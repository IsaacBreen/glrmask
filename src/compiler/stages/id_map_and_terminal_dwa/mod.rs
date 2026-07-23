//! Top-level id_map + terminal DWA builder.
//!
//! The canonical path splits the vocabulary into character-type partitions,
//! preserves each partition's L1 and L2P outputs, merges like families across
//! all partitions in parallel, then merges the two family DWAs.

use crate::automata::lexer::Lexer;
use crate::automata::lexer::compile::{
    compile_further_synthesized_tokenizer_with_structural_map,
    compile_partitioned_expression_pair_with_structural_map, factor_regex_expr,
    precompile_further_synthesis_pairs, PrecompiledFurtherSynthesisPairs,
    VocabularyRepeatHorizonCache,
};
pub(crate) mod classify;
mod finalize_ignore;
pub(crate) mod grammar_helpers;
pub(crate) mod l1;
pub(crate) mod l2p;
pub(crate) mod synthetic_state_map;
pub(crate) mod merge;
pub(crate) mod partition;
pub(crate) mod types;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::regex::Expr;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap, MappedArtifact};
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;
use crate::Vocab;

use classify::classify_vocab_char_type;
use finalize_ignore::erase_ignore_after_ti;
use grammar_helpers::{compute_always_allowed_follows, ignore_transparent_disallowed_follows};
use l2p::equivalence_analysis::state_equivalence::{
    resolve_global_pipeline_config, run_state_equivalence_pipeline, StateEquivalenceScope,
};
use types::{
    compile_profile_enabled, compile_profile_uses_serial_partition_schedule,
    LocalIdMapTerminalDwa, TerminalColoring, TerminalDwaFamilies, TerminalDwaPhaseProfile,
};

#[derive(Clone)]
pub(crate) struct PartitionLocalSynthesisPlan {
    pub(crate) expressions: Arc<[Expr]>,
    pub(crate) partition_ids: Arc<[u32]>,
    pub(crate) residual_isolation_classes: Arc<[Option<u32>]>,
    pub(crate) protected_terminal_ids: Arc<[u32]>,
    pub(crate) labels: Arc<[String]>,
    pub(crate) adaptive: bool,
    pub(crate) global_max_token_len: usize,
}

struct PartitionLocalTokenizer {
    tokenizer: Tokenizer,
    global_to_local: synthetic_state_map::CertifiedFullToSynthesizedStateMap,
    build_ms: f64,
}

struct PreparedPartitionLocalTokenizer {
    rebuilt_expressions: Arc<[Expr]>,
    local_expressions: Arc<[Expr]>,
    protected_terminal_ids: Arc<[u32]>,
    relevant_bytes: Arc<[u8]>,
    pairs: Arc<PrecompiledFurtherSynthesisPairs>,
    max_token_len: usize,
    build_ms: f64,
}

pub(crate) struct PreparedPartitionLocalTokenizers {
    entries: Vec<Mutex<Option<PreparedPartitionLocalTokenizer>>>,
}

impl PreparedPartitionLocalTokenizers {
    fn take(&self, partition: usize) -> Option<PreparedPartitionLocalTokenizer> {
        self.entries
            .get(partition)?
            .lock()
            .expect("prepared partition-local tokenizer slot poisoned")
            .take()
    }
}

fn partition_local_synthesis_enabled() -> bool {
    std::env::var("GLRMASK_PARTITION_LOCAL_SYNTHESIS")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(true)
}

pub(crate) fn prebuild_partition_local_synthesis_enabled() -> bool {
    std::env::var("GLRMASK_PREBUILD_PARTITION_LOCAL_SYNTHESIS")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(true)
}

fn fast_partition_local_synthesis_enabled() -> bool {
    std::env::var("GLRMASK_FAST_PARTITION_LOCAL_SYNTHESIS")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(true)
}

fn partition_local_synthesis_selected(partition_label: &str) -> bool {
    let Ok(filter) = std::env::var("GLRMASK_PARTITION_LOCAL_SYNTHESIS_FILTER") else {
        return true;
    };
    filter
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .any(|item| partition_label == item)
}

fn branch_active_state_map_enabled() -> bool {
    std::env::var("GLRMASK_BRANCH_ACTIVE_STATE_MAP")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(true)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutomaticBranchActiveStateMapStrategy {
    None,
    VeryLargeProfile,
    DenseRequiresFastProjection,
}

fn automatic_branch_active_state_map_strategy(
    branch_label: &str,
    vocab_tokens: usize,
    active_terminals: usize,
    source_reps: usize,
) -> AutomaticBranchActiveStateMapStrategy {
    if !branch_label.ends_with(".l1") {
        return AutomaticBranchActiveStateMapStrategy::None;
    }
    let work = source_reps.saturating_mul(vocab_tokens);
    if vocab_tokens >= 50_000 && work >= 300_000_000 {
        return AutomaticBranchActiveStateMapStrategy::VeryLargeProfile;
    }
    let dense_protected_profile = active_terminals >= 180
        && vocab_tokens >= 2_000
        && work >= 50_000_000
        && (source_reps >= 40_000 || vocab_tokens <= 8_000);
    if dense_protected_profile {
        AutomaticBranchActiveStateMapStrategy::DenseRequiresFastProjection
    } else {
        AutomaticBranchActiveStateMapStrategy::None
    }
}

pub(crate) fn build_branch_active_state_map(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    active: &[bool],
    initial_state_map: Option<&ManyToOneIdMap>,
    branch_label: &str,
    force: bool,
) -> Option<(ManyToOneIdMap, f64)> {
    if !branch_active_state_map_enabled() || vocab.is_empty() {
        return None;
    }
    let source_reps = initial_state_map
        .map(ManyToOneIdMap::num_internal_ids)
        .unwrap_or(tokenizer.num_states()) as usize;
    if source_reps <= 1 {
        return None;
    }
    let mut automatic_dense_requires_fast_projection = false;
    if let Ok(filter) = std::env::var("GLRMASK_BRANCH_ACTIVE_STATE_MAP_FILTER") {
        if !filter
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .any(|item| branch_label.contains(item))
        {
            return None;
        }
    } else if !force {
        // Stable active-language refinement has a real fixed cost. Select it
        // only when exact whole-token state profiling is predictably dominant.
        //
        // Two structural regimes amortize the quotient on protected-residual
        // workloads:
        //   * a very large vocabulary/state product (the existing broad gate);
        //   * a dense L1 terminal family with at least 50M raw state-token
        //     pairs and at least 2k tokens, provided either the state frontier
        //     is large or the vocabulary is compact enough that quotient
        //     construction does not contend with another medium/large token
        //     lane. The lower vocabulary bound avoids long-horizon lanes whose
        //     quotient remains above the fast-projected cutoff and would still
        //     pay a second exact-equivalence pass.
        //
        // The second clause deliberately excludes the medium-state,
        // medium-vocabulary regime: paired measurements show that adding a
        // quotient there shifts the critical path and worsens tail latency even
        // though its local CPU work falls.
        match automatic_branch_active_state_map_strategy(
            branch_label,
            vocab.len(),
            active.iter().filter(|&&value| value).count(),
            source_reps,
        ) {
            AutomaticBranchActiveStateMapStrategy::None => return None,
            AutomaticBranchActiveStateMapStrategy::VeryLargeProfile => {}
            AutomaticBranchActiveStateMapStrategy::DenseRequiresFastProjection => {
                automatic_dense_requires_fast_projection = true;
            }
        }
    }
    let started_at = Instant::now();
    let statistic =
        l2p::equivalence_analysis::state_equivalence::max_length::cached_statistic(vocab);
    let requested_mode =
        std::env::var("GLRMASK_BRANCH_ACTIVE_STATE_MAP_MODE").unwrap_or_else(|_| "stable".into());
    let (state_map, mode_label) = match requested_mode.as_str() {
        "empty" => (
            synthetic_state_map::empty_active_language_state_map(tokenizer, active),
            "empty",
        ),
        "structural" => {
            if initial_state_map.is_some() {
                return None;
            }
            (
                synthetic_state_map::inactive_dispatch_component_state_map(tokenizer, active)?,
                "structural",
            )
        }
        "stable" => (
            l2p::equivalence_analysis::state_equivalence::max_length::compute_state_map(
                tokenizer,
                &statistic,
                initial_state_map,
                Some(active),
                l2p::equivalence_analysis::state_equivalence::max_length::MaxLengthMode::StableByteRestricted,
                None,
                None,
            ),
            "stable",
        ),
        "kbounded" => (
            l2p::equivalence_analysis::state_equivalence::max_length::compute_state_map(
                tokenizer,
                &statistic,
                initial_state_map,
                Some(active),
                l2p::equivalence_analysis::state_equivalence::max_length::MaxLengthMode::KBoundedByteRestricted,
                None,
                None,
            ),
            "kbounded",
        ),
        other => panic!(
            "invalid GLRMASK_BRANCH_ACTIVE_STATE_MAP_MODE={other:?}; expected empty, structural, stable, or kbounded"
        ),
    };
    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    let reps = state_map.num_internal_ids() as usize;
    let enters_fast_projected_path =
        reps <= l1::fast_projected_l1_id_map_max_tsids();
    let selected = reps < source_reps
        && source_reps.saturating_sub(reps) >= 512
        && (!automatic_dense_requires_fast_projection || enters_fast_projected_path);
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][branch_active_state_map] branch={} mode={} active_terminals={} vocab_tokens={} horizon={} source_reps={} reps={} reduction_pct={:.2} ms={:.3} requires_fast_projected={} enters_fast_projected={} fast_projected_max_tsids={} selected={}",
            branch_label,
            mode_label,
            active.iter().filter(|&&value| value).count(),
            vocab.len(),
            statistic.max_token_len(),
            source_reps,
            reps,
            100.0 * source_reps.saturating_sub(reps) as f64 / source_reps as f64,
            elapsed_ms,
            automatic_dense_requires_fast_projection,
            enters_fast_projected_path,
            l1::fast_projected_l1_id_map_max_tsids(),
            selected,
        );
    }
    selected.then_some((state_map, elapsed_ms))
}

fn build_partition_local_tokenizer(
    global_tokenizer: &Tokenizer,
    vocab: &Vocab,
    plan: &PartitionLocalSynthesisPlan,
) -> Option<PartitionLocalTokenizer> {
    let reject = |stage: &str, horizon: usize| {
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][partition_local_synthesis] selected=false stage={} horizon={} global_states={}",
                stage,
                horizon,
                global_tokenizer.num_states(),
            );
        }
        None
    };
    if !partition_local_synthesis_enabled() || vocab.is_empty() {
        return None;
    }
    let max_token_len = vocab.entries.values().map(Vec::len).max().unwrap_or(0);
    if max_token_len == 0 || max_token_len >= plan.global_max_token_len {
        return None;
    }
    let allow_half_horizon = std::env::var("GLRMASK_PARTITION_LOCAL_SYNTHESIS_ALLOW_HALF_HORIZON")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(true);
    // A local structural pair has a non-trivial fixed construction cost. Very
    // small vocabulary partitions are already cheap, while a horizon close to
    // the global maximum cannot remove enough protected residual depth to
    // amortize that cost. This gate chooses an optimization strategy only; the
    // ordinary global-tokenizer path remains the exact fallback.
    let min_local_horizon = std::env::var("GLRMASK_PARTITION_LOCAL_SYNTHESIS_MIN_HORIZON")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(8);
    let min_local_vocab = std::env::var("GLRMASK_PARTITION_LOCAL_SYNTHESIS_MIN_VOCAB")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(2_000);
    if vocab.len() < min_local_vocab
        || max_token_len < min_local_horizon
        || if allow_half_horizon {
            max_token_len.saturating_mul(2) > plan.global_max_token_len
        } else {
            max_token_len.saturating_mul(2) >= plan.global_max_token_len
        }
    {
        return None;
    }

    let started_at = Instant::now();
    let protected = plan
        .protected_terminal_ids
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let mut candidate = synthetic_state_map::synthesize_terminal_expressions_for_horizon(
        &plan.expressions,
        max_token_len,
    );
    let mut changed = false;
    for terminal in 0..candidate.expressions.len() {
        if protected.contains(&(terminal as u32)) {
            changed |= candidate.expressions[terminal] != plan.expressions[terminal];
        } else {
            candidate.expressions[terminal] = plan.expressions[terminal].clone();
        }
    }
    if !changed {
        return reject("unchanged_candidate", max_token_len);
    }

    let rebuilt_expressions = plan
        .expressions
        .iter()
        .cloned()
        .map(factor_regex_expr)
        .collect::<Vec<_>>();
    let local_expressions = candidate
        .expressions
        .into_iter()
        .map(factor_regex_expr)
        .collect::<Vec<_>>();
    let mut relevant_bytes = vocab
        .entries
        .values()
        .flat_map(|bytes| bytes.iter().copied())
        .collect::<Vec<_>>();
    relevant_bytes.sort_unstable();
    relevant_bytes.dedup();
    if fast_partition_local_synthesis_enabled()
        && let Some((mut local_tokenizer, global_to_local)) =
            compile_further_synthesized_tokenizer_with_structural_map(
                global_tokenizer,
                &rebuilt_expressions,
                &local_expressions,
                &plan.protected_terminal_ids,
                vocab,
                &VocabularyRepeatHorizonCache::new(),
                max_token_len,
                &relevant_bytes,
                None,
            )
    {
        let local_nullable = local_tokenizer.isolate_start_state_and_drain_nullable_terminals();
        if local_nullable.is_empty() {
            let local_states = local_tokenizer.num_states() as usize;
            let global_states = global_tokenizer.num_states() as usize;
            let ratio_rejected = if allow_half_horizon {
                local_states.saturating_mul(20) > global_states.saturating_mul(19)
            } else {
                local_states.saturating_mul(4) > global_states.saturating_mul(3)
            };
            if local_states < global_states
                && global_states.saturating_sub(local_states) >= 1_024
                && !ratio_rejected
            {
                if compile_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][partition_local_synthesis_fast] horizon={} global_states={} local_states={} local_transitions={} build_ms={:.3} selected=true",
                        max_token_len,
                        global_states,
                        local_states,
                        local_tokenizer.transition_count(),
                        started_at.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                return Some(PartitionLocalTokenizer {
                    tokenizer: local_tokenizer,
                    global_to_local: synthetic_state_map::CertifiedFullToSynthesizedStateMap {
                        full_to_synthesized: global_to_local,
                    },
                    build_ms: started_at.elapsed().as_secs_f64() * 1000.0,
                });
            }
        }
    }
    let Some(pair) = compile_partitioned_expression_pair_with_structural_map(
        &rebuilt_expressions,
        &local_expressions,
        Some(&plan.labels),
        &plan.partition_ids,
        &plan.residual_isolation_classes,
        plan.adaptive,
        vocab,
        &VocabularyRepeatHorizonCache::new(),
        max_token_len,
        &relevant_bytes,
    ) else {
        return reject("structural_pair", max_token_len);
    };
    let terminal_count = plan.expressions.len() as u32;
    let mut rebuilt_tokenizer = pair.full.into_tokenizer(
        terminal_count,
        Some(Arc::from(rebuilt_expressions.into_boxed_slice())),
    );
    let rebuilt_nullable = rebuilt_tokenizer.isolate_start_state_and_drain_nullable_terminals();
    if !rebuilt_nullable.is_empty() {
        return reject("rebuilt_nullable", max_token_len);
    }
    let mut local_tokenizer = pair.synthesized.into_tokenizer(
        terminal_count,
        Some(Arc::from(local_expressions.into_boxed_slice())),
    );
    let local_nullable = local_tokenizer.isolate_start_state_and_drain_nullable_terminals();
    if !local_nullable.is_empty() {
        return reject("local_nullable", max_token_len);
    }
    let Some(global_to_local) = local_tokenizer.augment_from_verified_component_prefixes(
        global_tokenizer,
        &rebuilt_tokenizer,
        &pair.full_to_synthesized,
    ) else {
        return reject("component_prefix", max_token_len);
    };
    let local_states = local_tokenizer.num_states() as usize;
    let global_states = global_tokenizer.num_states() as usize;
    let ratio_rejected = if allow_half_horizon {
        local_states.saturating_mul(20) > global_states.saturating_mul(19)
    } else {
        local_states.saturating_mul(4) > global_states.saturating_mul(3)
    };
    if local_states >= global_states
        || global_states.saturating_sub(local_states) < 1_024
        || ratio_rejected
    {
        return reject("insufficient_reduction", max_token_len);
    }
    Some(PartitionLocalTokenizer {
        tokenizer: local_tokenizer,
        global_to_local: synthetic_state_map::CertifiedFullToSynthesizedStateMap {
            full_to_synthesized: global_to_local,
        },
        build_ms: started_at.elapsed().as_secs_f64() * 1000.0,
    })
}

fn prepare_partition_local_tokenizer(
    vocab: &Vocab,
    plan: &PartitionLocalSynthesisPlan,
) -> Option<PreparedPartitionLocalTokenizer> {
    if !partition_local_synthesis_enabled() || vocab.is_empty() {
        return None;
    }
    let max_token_len = vocab.entries.values().map(Vec::len).max().unwrap_or(0);
    if max_token_len == 0 || max_token_len >= plan.global_max_token_len {
        return None;
    }
    let allow_half_horizon = std::env::var("GLRMASK_PARTITION_LOCAL_SYNTHESIS_ALLOW_HALF_HORIZON")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(true);
    let min_local_horizon = std::env::var("GLRMASK_PARTITION_LOCAL_SYNTHESIS_MIN_HORIZON")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(16);
    let min_local_vocab = std::env::var("GLRMASK_PARTITION_LOCAL_SYNTHESIS_MIN_VOCAB")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(2_000);
    if vocab.len() < min_local_vocab
        || max_token_len < min_local_horizon
        || if allow_half_horizon {
            max_token_len.saturating_mul(2) > plan.global_max_token_len
        } else {
            max_token_len.saturating_mul(2) >= plan.global_max_token_len
        }
    {
        return None;
    }

    let protected = plan
        .protected_terminal_ids
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let mut candidate = synthetic_state_map::synthesize_terminal_expressions_for_horizon(
        &plan.expressions,
        max_token_len,
    );
    let mut changed = false;
    for terminal in 0..candidate.expressions.len() {
        if protected.contains(&(terminal as u32)) {
            changed |= candidate.expressions[terminal] != plan.expressions[terminal];
        } else {
            candidate.expressions[terminal] = plan.expressions[terminal].clone();
        }
    }
    if !changed {
        return None;
    }

    let rebuilt_expressions: Arc<[Expr]> = Arc::from(
        plan.expressions
            .iter()
            .cloned()
            .map(factor_regex_expr)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    let local_expressions: Arc<[Expr]> = Arc::from(
        candidate
            .expressions
            .into_iter()
            .map(factor_regex_expr)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    let mut relevant_bytes = vocab
        .entries
        .values()
        .flat_map(|bytes| bytes.iter().copied())
        .collect::<Vec<_>>();
    relevant_bytes.sort_unstable();
    relevant_bytes.dedup();
    let relevant_bytes: Arc<[u8]> = Arc::from(relevant_bytes.into_boxed_slice());
    let pairs = precompile_further_synthesis_pairs(
        &rebuilt_expressions,
        &local_expressions,
        &plan.protected_terminal_ids,
        vocab,
        &VocabularyRepeatHorizonCache::new(),
        max_token_len,
        &relevant_bytes,
    )?;
    let build_ms = pairs.build_ms;
    Some(PreparedPartitionLocalTokenizer {
        rebuilt_expressions,
        local_expressions,
        protected_terminal_ids: Arc::clone(&plan.protected_terminal_ids),
        relevant_bytes,
        pairs,
        max_token_len,
        build_ms,
    })
}

fn finish_prepared_partition_local_tokenizer(
    global_tokenizer: &Tokenizer,
    prepared: PreparedPartitionLocalTokenizer,
    vocab: &Vocab,
) -> Option<PartitionLocalTokenizer> {
    let finish_started_at = Instant::now();
    let (mut local_tokenizer, global_to_local) =
        compile_further_synthesized_tokenizer_with_structural_map(
            global_tokenizer,
            &prepared.rebuilt_expressions,
            &prepared.local_expressions,
            &prepared.protected_terminal_ids,
            vocab,
            &VocabularyRepeatHorizonCache::new(),
            prepared.max_token_len,
            &prepared.relevant_bytes,
            Some(&prepared.pairs),
        )?;
    if !local_tokenizer
        .isolate_start_state_and_drain_nullable_terminals()
        .is_empty()
    {
        return None;
    }
    let local_states = local_tokenizer.num_states() as usize;
    let global_states = global_tokenizer.num_states() as usize;
    let allow_half_horizon = std::env::var("GLRMASK_PARTITION_LOCAL_SYNTHESIS_ALLOW_HALF_HORIZON")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(true);
    let ratio_rejected = if allow_half_horizon {
        local_states.saturating_mul(20) > global_states.saturating_mul(19)
    } else {
        local_states.saturating_mul(4) > global_states.saturating_mul(3)
    };
    if local_states >= global_states
        || global_states.saturating_sub(local_states) < 1_024
        || ratio_rejected
    {
        return None;
    }
    let finish_ms = finish_started_at.elapsed().as_secs_f64() * 1000.0;
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][partition_local_synthesis_prebuilt] horizon={} global_states={} local_states={} local_transitions={} precompile_ms={:.3} finish_ms={:.3} selected=true",
            prepared.max_token_len,
            global_states,
            local_states,
            local_tokenizer.transition_count(),
            prepared.build_ms,
            finish_ms,
        );
    }
    Some(PartitionLocalTokenizer {
        tokenizer: local_tokenizer,
        global_to_local: synthetic_state_map::CertifiedFullToSynthesizedStateMap {
            full_to_synthesized: global_to_local,
        },
        build_ms: finish_ms,
    })
}

pub(crate) fn prepare_partition_local_tokenizers(
    vocab: &Vocab,
    plan: &PartitionLocalSynthesisPlan,
) -> Option<Arc<PreparedPartitionLocalTokenizers>> {
    if !prebuild_partition_local_synthesis_enabled()
        || std::env::var("GLRMASK_PARTITION_SCHEME")
            .as_deref()
            .unwrap_or("char_type")
            != "char_type"
    {
        return None;
    }
    use rayon::prelude::*;
    let sub_vocabs = build_char_type_sub_vocabs(vocab, true);
    let entries = sub_vocabs
        .par_iter()
        .enumerate()
        .map(|(partition, sub_vocab)| {
            let label = format!("p{partition}");
            let selected = std::env::var("GLRMASK_PREBUILD_PARTITION_LOCAL_SYNTHESIS_FILTER")
                .map(|filter| {
                    filter
                        .split(',')
                        .map(str::trim)
                        .filter(|item| !item.is_empty())
                        .any(|item| item == label)
                })
                .unwrap_or_else(|_| matches!(partition, 0 | 1 | 2 | 4));
            Mutex::new(
                selected
                    .then(|| prepare_partition_local_tokenizer(sub_vocab, plan))
                    .flatten(),
            )
        })
        .collect();
    Some(Arc::new(PreparedPartitionLocalTokenizers { entries }))
}

fn lift_partition_terminal_dwas_to_global(
    parts: &mut types::PartitionTerminalDwas,
    global_to_local: &synthetic_state_map::CertifiedFullToSynthesizedStateMap,
) -> Option<()> {
    for part in [
        parts.l1.as_mut(),
        parts.l2p.as_mut(),
        parts.l2p_single_l1.as_mut(),
    ]
    .into_iter()
    .flatten()
    {
        part.id_map.tokenizer_states = global_to_local
            .lift_internal_tsid_map(&part.id_map.tokenizer_states)?;
    }
    Some(())
}

fn l2p_partition_cost_fn_from_env() -> classify::L2pPartitionCostFn {
    match std::env::var("GLRMASK_L2P_COST_FN").as_deref() {
        Ok("size") | Err(_) => classify::L2pPartitionCostFn::Size,
        Ok("size_log") => classify::L2pPartitionCostFn::SizeLog,
        Ok("log_log") => classify::L2pPartitionCostFn::LogLog,
        Ok("union_size") => classify::L2pPartitionCostFn::UnionSize,
        Ok(other) => panic!(
            "Invalid GLRMASK_L2P_COST_FN={other}; expected one of: size, size_log, log_log, union_size"
        ),
    }
}

fn l2p_partition_objective_from_env() -> classify::L2pPartitionObjective {
    match std::env::var("GLRMASK_L2P_COST_OBJECTIVE").as_deref() {
        Ok("max") | Err(_) => classify::L2pPartitionObjective::Max,
        Ok("sum") => classify::L2pPartitionObjective::Sum,
        Ok(other) => panic!(
            "Invalid GLRMASK_L2P_COST_OBJECTIVE={other}; expected one of: max, sum"
        ),
    }
}

fn l2p_partition_count_from_env() -> usize {
    std::env::var("GLRMASK_L2P_COST_PARTITIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&count| count > 0)
        .unwrap_or(10)
}

fn l2p_auto_second_largest_limit_from_env() -> usize {
    std::env::var("GLRMASK_L2P_AUTO_SECOND_LARGEST_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&count| count > 0)
        .unwrap_or(12_000)
}

fn l2p_auto_max_estimated_l2p_terminals_from_env() -> usize {
    std::env::var("GLRMASK_L2P_AUTO_MAX_ESTIMATED_L2P_TERMINALS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&count| count > 0)
        .unwrap_or(7)
}

fn l2p_auto_min_estimated_l2p_terminals_from_env() -> usize {
    std::env::var("GLRMASK_L2P_AUTO_MIN_ESTIMATED_L2P_TERMINALS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(6)
}

fn l2p_auto_min_grammar_terminals_from_env() -> usize {
    std::env::var("GLRMASK_L2P_AUTO_MIN_GRAMMAR_TERMINALS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(12)
}

#[derive(Debug)]
struct CharTypeSubVocabs {
    p0_overflow_threshold: Option<usize>,
    p1_overflow_threshold: Option<usize>,
    p2_overflow_threshold: Option<usize>,
    p4_overflow_threshold: Option<usize>,
    sub_vocabs: Arc<[Vocab]>,
}

impl crate::vocab::VocabDerivedArtifact for CharTypeSubVocabs {}

fn long_token_overflow_threshold(
    name: &str,
    automatic: Option<usize>,
) -> Option<usize> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|&threshold| threshold > 0),
        Err(_) => automatic,
    }
}

fn char_type_partition_index(
    bytes: &[u8],
    p0_overflow_threshold: Option<usize>,
    p1_overflow_threshold: Option<usize>,
    p2_overflow_threshold: Option<usize>,
    p4_overflow_threshold: Option<usize>,
) -> usize {
    let partition = classify_vocab_char_type(bytes) as usize;
    if partition == 0
        && p0_overflow_threshold.is_some_and(|threshold| bytes.len() > threshold)
    {
        12
    } else if partition == 1
        && p1_overflow_threshold.is_some_and(|threshold| bytes.len() > threshold)
    {
        10
    } else if partition == 2
        && p2_overflow_threshold.is_some_and(|threshold| bytes.len() > threshold)
    {
        9
    } else if partition == 4
        && p4_overflow_threshold.is_some_and(|threshold| bytes.len() > threshold)
    {
        11
    } else {
        partition
    }
}

fn vocab_from_token_partitions(vocab: &Vocab, token_partitions: Vec<Vec<u32>>) -> Arc<[Vocab]> {
    token_partitions
        .into_iter()
        .map(|token_ids| {
            let entries = token_ids
                .into_iter()
                .filter_map(|token_id| vocab.entries.get(&token_id).map(|bytes| (token_id, bytes.clone())))
                .collect();
            Vocab::new(entries)
        })
        .collect::<Vec<_>>()
        .into()
}

fn build_char_type_sub_vocabs(
    vocab: &Vocab,
    automatic_bounded_synthesis_overflow: bool,
) -> Arc<[Vocab]> {
    let p0_overflow_threshold = long_token_overflow_threshold(
        "GLRMASK_P0_LONG_TOKEN_OVERFLOW_THRESHOLD",
        automatic_bounded_synthesis_overflow.then_some(16),
    );
    let p1_overflow_threshold = long_token_overflow_threshold(
        "GLRMASK_P1_LONG_TOKEN_OVERFLOW_THRESHOLD",
        automatic_bounded_synthesis_overflow.then_some(20),
    );
    let p2_overflow_threshold = long_token_overflow_threshold(
        "GLRMASK_P2_LONG_TOKEN_OVERFLOW_THRESHOLD",
        automatic_bounded_synthesis_overflow.then_some(32),
    );
    let p4_overflow_threshold = long_token_overflow_threshold(
        "GLRMASK_P4_LONG_TOKEN_OVERFLOW_THRESHOLD",
        automatic_bounded_synthesis_overflow.then_some(32),
    );
    if let Some(cached) = vocab.vocab_derived_cache_get::<CharTypeSubVocabs>() {
        if cached.p0_overflow_threshold == p0_overflow_threshold
            && cached.p1_overflow_threshold == p1_overflow_threshold
            && cached.p2_overflow_threshold == p2_overflow_threshold
            && cached.p4_overflow_threshold == p4_overflow_threshold
        {
            return Arc::clone(&cached.sub_vocabs);
        }
    }

    let partition_count = if p0_overflow_threshold.is_some() {
        13
    } else if p4_overflow_threshold.is_some() {
        12
    } else if p1_overflow_threshold.is_some() {
        11
    } else if p2_overflow_threshold.is_some() {
        10
    } else {
        9
    };
    let mut partition_entries: Vec<Vec<(u32, Vec<u8>)>> =
        (0..partition_count).map(|_| Vec::new()).collect();
    let mut partition_bytes = vec![U8Set::empty(); partition_count];
    let mut partition_follow_bytes: Vec<[U8Set; 256]> =
        (0..partition_count).map(|_| [U8Set::empty(); 256]).collect();
    for (&token_id, bytes) in vocab.entries.iter() {
        let idx = char_type_partition_index(
            bytes,
            p0_overflow_threshold,
            p1_overflow_threshold,
            p2_overflow_threshold,
            p4_overflow_threshold,
        );
        for &byte in bytes {
            partition_bytes[idx].insert(byte);
        }
        for pair in bytes.windows(2) {
            partition_follow_bytes[idx][pair[0] as usize].insert(pair[1]);
        }
        partition_entries[idx].push((token_id, bytes.clone()));
    }
    let sub_vocabs: Arc<[Vocab]> = partition_entries
        .into_iter()
        .enumerate()
        .map(|(idx, entries)| {
            let vocab = Vocab::new(entries);
            classify::cache_vocab_classification_facts(
                &vocab,
                partition_bytes[idx],
                partition_follow_bytes[idx],
            );
            vocab
        })
        .collect::<Vec<_>>()
        .into();
    vocab.vocab_derived_cache_set(Arc::new(CharTypeSubVocabs {
        p0_overflow_threshold,
        p1_overflow_threshold,
        p2_overflow_threshold,
        p4_overflow_threshold,
        sub_vocabs: Arc::clone(&sub_vocabs),
    }));
    sub_vocabs
}

pub(crate) fn prepare_vocab_for_terminal_dwa(vocab: &Vocab) {
    classify::prepare_vocab_for_terminal_classification(vocab);
    l1::prepare_l1_identity_vocab_order(vocab);
    l1::prepare_l1_token_bounded_analysis_trie(vocab);

    if std::env::var("GLRMASK_PARTITION_SCHEME").as_deref().unwrap_or("char_type") == "char_type" {
        for sub_vocab in build_char_type_sub_vocabs(vocab, false).iter() {
            classify::prepare_vocab_for_terminal_classification(sub_vocab);
            l1::prepare_l1_identity_vocab_order(sub_vocab);
            l1::prepare_l1_token_bounded_analysis_trie(sub_vocab);
        }
    }
}

fn global_max_length_env_override() -> Option<bool> {
    static OVERRIDE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("GLRMASK_USE_GLOBAL_MAX_LENGTH")
            .ok()
            .map(|value| {
                let trimmed = value.trim();
                !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
            })
    })
}

const DEFAULT_GLOBAL_MAX_LENGTH_STABLE_SIGNATURE_CELL_LIMIT: usize = 2_000_000;

fn global_max_length_stable_signature_cell_limit() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("GLRMASK_GLOBAL_MAX_LENGTH_STABLE_SIGNATURE_CELL_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_GLOBAL_MAX_LENGTH_STABLE_SIGNATURE_CELL_LIMIT)
    })
}

fn global_max_length_stable_signature_cells(
    tokenizer: &Tokenizer,
    statistic: &l2p::equivalence_analysis::state_equivalence::max_length::MaxLengthStatistic,
) -> usize {
    (tokenizer.num_states() as usize)
        .saturating_mul(1 + statistic.relevant_byte_count())
}

fn should_auto_use_global_max_length(
    num_states: usize,
    relevant_byte_count: usize,
    stable_signature_cell_limit: usize,
) -> bool {
    num_states > 50_000
        && num_states.saturating_mul(1 + relevant_byte_count) <= stable_signature_cell_limit
}

fn use_global_max_length(
    tokenizer: &Tokenizer,
    statistic: &l2p::equivalence_analysis::state_equivalence::max_length::MaxLengthStatistic,
) -> bool {
    match global_max_length_env_override() {
        Some(enabled) => enabled,
        None => {
            // Stable refinement is exact, but its cost is unbounded in the
            // number of refinement rounds. It is only a compile-time
            // acceleration: returning the identity map preserves all states
            // and therefore exactness. Do not launch the global stable pass
            // when even one signature matrix exceeds the structural budget.
            should_auto_use_global_max_length(
                tokenizer.num_states() as usize,
                statistic.relevant_byte_count(),
                global_max_length_stable_signature_cell_limit(),
            )
        }
    }
}

pub(crate) fn build_global_max_length_state_map(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    _flat_trans: &Arc<[u32]>,
) -> ManyToOneIdMap {
    let started_at = Instant::now();
    let num_states_u32 = tokenizer.num_states();
    let num_states = num_states_u32 as usize;
    let global_statistic =
        l2p::equivalence_analysis::state_equivalence::max_length::cached_statistic(vocab);
    let max_token_len = global_statistic.max_token_len();
    let stable_signature_cells = global_max_length_stable_signature_cells(tokenizer, &global_statistic);
    let stable_signature_cell_limit = global_max_length_stable_signature_cell_limit();

    let config = resolve_global_pipeline_config(use_global_max_length(tokenizer, &global_statistic));
    let (state_map, profile) = run_state_equivalence_pipeline(
        tokenizer,
        vocab,
        None,
        None,
        StateEquivalenceScope::Global,
        &config,
        None,
        None,
        None,
    );

    if compile_profile_enabled() {
        if profile.max_length_skipped {
            eprintln!(
                "[glrmask/profile][global_max_length] mode=identity skipped=true states={} reps={} tokens_included=0 max_token_len=0 stable_signature_cells={} stable_signature_cell_limit={} ms={:.3}",
                num_states,
                state_map.representative_original_ids.len(),
                stable_signature_cells,
                stable_signature_cell_limit,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        } else {
            eprintln!(
                "[glrmask/profile][global_max_length] mode=stable skipped=false states={} reps={} tokens_included={} max_token_len={} stable_signature_cells={} stable_signature_cell_limit={} ms={:.3}",
                num_states,
                state_map.representative_original_ids.len(),
                vocab.len(),
                max_token_len,
                stable_signature_cells,
                stable_signature_cell_limit,
                profile.max_length_state_equiv_ms,
            );
        }
    }

    state_map
}

/// Build the global `(InternalIdMap, DWA)` for the full vocabulary.
///
/// IMPORTANT: the JSON-schema importer must not feed this stage large numbers
/// of short, grammar-visible alnum-ish terminals. A terminal is
/// "grammar-visible" when it is either a named terminal or an inline
/// literal/pattern that appears directly in a nonterminal rule body.
///
/// Those terminals are pathological for terminal-DWA construction when all of
/// the following hold:
///
/// 1. they match one of a broad character class but only a small bounded
///    number of characters (classic bad cases: `[a-z]`, `[a-z]{1,3}`,
///    bare URI/hostname/JSON-string body fragments);
/// 2. they do not carry stabilizing punctuation with them, especially on the
///    trailing edge; and
/// 3. they are grammar-visible rather than internal-only.
///
/// This creates explosive same-prefix ambiguity in the terminal DWA: e.g. a
/// visible `[a-z]` and `[a-z][a-z]` force the DWA to keep many competing
/// terminal continuations alive. The importer therefore deliberately fuses
/// punctuation into visible terminals when possible, and keeps short generic
/// bodies internal-only whenever possible. Do not "simplify" that structure
/// away without re-checking schemas like `Github_hard---o1051` and
/// `uuid_maxlength5000`.
///
/// `JSON_NUMBER` is the known exception: it technically fits the heuristic, but
/// in practice it has not shown the same DWA blow-up.
///
/// 1. Splits vocab into 3 partitions by leading-byte character type.
/// 2. Builds each partition's L1/L2P DWA pieces in parallel via
///    [`partition::build_partition_id_map_and_terminal_dwa`].
/// 3. Merges every L1-style result and every L2P result in parallel.
/// 4. Merges the two family results into the final terminal DWA.
pub(crate) fn build_terminal_dwa_families_with_precomputed_global_max_length(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    flat_trans: Arc<[u32]>,
    global_max_length_state_map: &ManyToOneIdMap,
    external_classify_cache: Option<&classify::SharedClassifyCache>,
    external_transition_cache: Option<
        &OnceLock<l2p::equivalence_analysis::compat::FlatTransitionCache>,
    >,
    partition_local_synthesis_plan: Option<&PartitionLocalSynthesisPlan>,
    prepared_partition_local_tokenizers: Option<&PreparedPartitionLocalTokenizers>,
) -> (TerminalDwaFamilies, TerminalDwaPhaseProfile) {
    let total_started_at = Instant::now();
    let mut profile = TerminalDwaPhaseProfile::default();

    // Shared cache for terminal classification byte sets. The DFA scanning
    // (reachable_bytes, first_bytes, last_bytes) is identical across partitions;
    // only the vocab-dependent classification differs. Reuse external cache if
    // provided (already populated by compile.rs pre-classification), otherwise
    // create a fresh one for partition sharing.
    let owned_classify_cache = classify::SharedClassifyCache::new();
    let shared_classify_cache: &classify::SharedClassifyCache =
        external_classify_cache.unwrap_or(&owned_classify_cache);
    let token_path_disallowed_follows = Arc::new(
        ignore_transparent_disallowed_follows(disallowed_follows, ignore_terminal),
    );
    let normalized_token_path_disallowed_follows: Arc<[BitSet]> = Arc::from(
        l2p::equivalence_analysis::disallowed_follows::normalize_disallowed_follows(
            grammar.num_terminals as usize,
            &token_path_disallowed_follows,
        )
        .into_boxed_slice(),
    );
    let stage_setup_ms = total_started_at.elapsed().as_secs_f64() * 1000.0;

    let partition_vocab_started_at = Instant::now();
    let requested_partition_scheme =
        std::env::var("GLRMASK_PARTITION_SCHEME").unwrap_or_else(|_| "char_type".to_string());
    let partition_scheme = requested_partition_scheme.as_str();
    let sub_vocabs: Arc<[Vocab]> = match partition_scheme {
        "char_type" => build_char_type_sub_vocabs(
            vocab,
            partition_local_synthesis_plan.is_some(),
        ),
        "l2p_cost" => {
            let cost_fn = l2p_partition_cost_fn_from_env();
            let objective = l2p_partition_objective_from_env();
            let num_partitions = l2p_partition_count_from_env();
            let bytesets = shared_classify_cache.get_or_init(|| {
                classify::SharedClassifyBytesets::build(tokenizer, grammar.num_terminals)
            });
            let partitioning = classify::partition_vocab_by_l2p_cost(
                vocab,
                bytesets,
                &token_path_disallowed_follows,
                grammar.num_terminals,
                num_partitions,
                cost_fn,
                objective,
            );

            if compile_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][l2p_cost_partitioning] cost_fn={} objective={} partitions={} estimated_costs={:?} estimated_l2p_terminals={:?} objective_score={:.3}",
                    cost_fn.as_str(),
                    objective.as_str(),
                    num_partitions,
                    partitioning.estimated_partition_costs,
                    partitioning.estimated_l2p_terminals,
                    partitioning.objective_score,
                );
            }

            vocab_from_token_partitions(vocab, partitioning.partitions)
        }
        "auto_l2p_cost" => {
            let cost_fn = l2p_partition_cost_fn_from_env();
            let objective = l2p_partition_objective_from_env();
            let num_partitions = l2p_partition_count_from_env();
            let min_grammar_terminals_limit = l2p_auto_min_grammar_terminals_from_env();
            let char_token_partitions = classify::partition_vocab_char_type_tokens(
                vocab,
                partition_local_synthesis_plan.is_some(),
            );
            let char_partition_sizes = char_token_partitions
                .iter()
                .map(|partition| partition.len())
                .collect::<Vec<_>>();

            if (grammar.num_terminals as usize) < min_grammar_terminals_limit {
                if compile_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][auto_l2p_partition] cost_fn={} objective={} l2p_partitions={} grammar_terminals={} disallowed_follows_len={} min_grammar_terminals_limit={} char_partition_sizes={:?} chosen=char_type reason=low_grammar_terminal_count",
                        cost_fn.as_str(),
                        objective.as_str(),
                        num_partitions,
                        grammar.num_terminals,
                        token_path_disallowed_follows.len(),
                        min_grammar_terminals_limit,
                        char_partition_sizes,
                    );
                }
                vocab_from_token_partitions(vocab, char_token_partitions)
            } else {
                let bytesets = shared_classify_cache.get_or_init(|| {
                    classify::SharedClassifyBytesets::build(tokenizer, grammar.num_terminals)
                });
                let second_largest_limit = l2p_auto_second_largest_limit_from_env();
                let max_estimated_l2p_terminals_limit =
                    l2p_auto_max_estimated_l2p_terminals_from_env();
                let min_estimated_l2p_terminals_limit =
                    l2p_auto_min_estimated_l2p_terminals_from_env();

                let (l2p_partitioning, token_l2p_map) =
                    classify::partition_vocab_by_l2p_cost_with_token_map(
                        vocab,
                        bytesets,
                        &token_path_disallowed_follows,
                        grammar.num_terminals,
                        num_partitions,
                        cost_fn,
                        objective,
                    );

                let l2p_partition_sizes = l2p_partitioning
                    .partitions
                    .iter()
                    .map(|token_ids| token_ids.len())
                    .collect::<Vec<_>>();
                let mut sorted_sizes = l2p_partition_sizes.clone();
                sorted_sizes.sort_unstable_by(|left, right| right.cmp(left));
                let second_largest = sorted_sizes.get(1).copied().unwrap_or(0);
                let max_estimated_l2p_terminals = l2p_partitioning
                    .estimated_l2p_terminals
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0);

                let mut char_costs = Vec::new();
                let mut char_l2p_terminals = Vec::new();
                let mut char_score = f64::INFINITY;

                let use_l2p = if second_largest <= second_largest_limit
                    && max_estimated_l2p_terminals >= min_estimated_l2p_terminals_limit
                    && max_estimated_l2p_terminals <= max_estimated_l2p_terminals_limit
                {
                    let (computed_char_costs, computed_char_l2p_terminals, computed_char_score) =
                        classify::estimate_l2p_objective_for_token_partitions(
                            &char_token_partitions,
                            &token_l2p_map,
                            cost_fn,
                            objective,
                        );
                    char_costs = computed_char_costs;
                    char_l2p_terminals = computed_char_l2p_terminals;
                    char_score = computed_char_score;
                    l2p_partitioning.objective_score < char_score
                } else {
                    false
                };

                if compile_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][auto_l2p_partition] cost_fn={} objective={} l2p_partitions={} l2p_score={:.3} char_score={:.3} second_largest={} second_largest_limit={} disallowed_follows_len={} max_estimated_l2p_terminals={} min_estimated_l2p_terminals_limit={} max_estimated_l2p_terminals_limit={} char_partition_sizes={:?} chosen={} l2p_sizes={:?} l2p_costs={:?} char_costs={:?} l2p_l2p_terminals={:?} char_l2p_terminals={:?}",
                        cost_fn.as_str(),
                        objective.as_str(),
                        num_partitions,
                        l2p_partitioning.objective_score,
                        char_score,
                        second_largest,
                        second_largest_limit,
                        token_path_disallowed_follows.len(),
                        max_estimated_l2p_terminals,
                        min_estimated_l2p_terminals_limit,
                        max_estimated_l2p_terminals_limit,
                        char_partition_sizes,
                        if use_l2p { "l2p_cost" } else { "char_type" },
                        l2p_partition_sizes,
                        l2p_partitioning.estimated_partition_costs,
                        char_costs,
                        l2p_partitioning.estimated_l2p_terminals,
                        char_l2p_terminals,
                    );
                }

                if use_l2p {
                    vocab_from_token_partitions(vocab, l2p_partitioning.partitions)
                } else {
                    vocab_from_token_partitions(vocab, char_token_partitions)
                }
            }
        }
        other => panic!(
            "Invalid GLRMASK_PARTITION_SCHEME={other}; expected one of: char_type, l2p_cost, auto_l2p_cost"
        ),
    };
    let partition_vocab_ms = partition_vocab_started_at.elapsed().as_secs_f64() * 1000.0;
    profile.id_map_ms += partition_vocab_ms;

    // Lazily-initialized shared compact transition table cache over the one
    // raw lexer DFA used by every L2P partition. Subsequent partitions reuse
    // it, avoiding repeated transpose and byte-class computation.
    let shared_cache_setup_started_at = Instant::now();
    let shared_vocab_dfa_cache = l2p::equivalence_analysis::vocab::fast::SharedVocabDfaCache::new();
    // Shared raw-tokenizer cache for L2P vocabulary equivalence.
    let shared_original_vocab_dfa_cache =
        l2p::equivalence_analysis::vocab::fast::SharedVocabDfaCache::new();
    let shared_original_vocab_analysis_dfa_cache =
        l2p::equivalence_analysis::vocab::fast::SharedVocabAnalysisDfaCache::default();
    let owned_transition_cache = OnceLock::new();
    let shared_transition_cache = external_transition_cache.unwrap_or(&owned_transition_cache);
    // Grammar-wide and identical for every L2P vocabulary partition. Compute
    // once before parallel partition construction rather than repeating the
    // same FIRST/FOLLOW occurrence traversal in each partition.
    let always_allowed_follows = compute_always_allowed_follows(grammar);
    let shared_ti_output_cache = shared_classify_cache
        .get()
        .and_then(l2p::SharedTiTokenizerOutputCache::new_with_classify_bytesets)
        .unwrap_or_else(l2p::SharedTiTokenizerOutputCache::new);
    let shared_cache_setup_ms =
        shared_cache_setup_started_at.elapsed().as_secs_f64() * 1000.0;

    use rayon::prelude::*;
    let build_partition = |idx: usize,
                           sub_vocab: &Vocab|
     -> (Option<(types::PartitionTerminalDwas, f64)>, usize) {
        let started_at = Instant::now();
        let label = format!("p{}", idx);

        let prepared_local = prepared_partition_local_tokenizers
            .and_then(|prepared| prepared.take(idx))
            .and_then(|prepared| {
                finish_prepared_partition_local_tokenizer(tokenizer, prepared, sub_vocab)
            });
        if partition_local_synthesis_selected(&label)
            && let Some(local) = prepared_local.or_else(|| {
                partition_local_synthesis_plan
                    .and_then(|plan| build_partition_local_tokenizer(tokenizer, sub_vocab, plan))
            })
        {
            let local_flat_trans: Arc<[u32]> =
                Arc::from(l1::build_flat_transition_table(&local.tokenizer));
            let local_vocab_dfa_cache =
                l2p::equivalence_analysis::vocab::fast::SharedVocabDfaCache::new();
            let local_original_vocab_dfa_cache =
                l2p::equivalence_analysis::vocab::fast::SharedVocabDfaCache::new();
            let local_original_vocab_analysis_dfa_cache =
                l2p::equivalence_analysis::vocab::fast::SharedVocabAnalysisDfaCache::default();
            let local_transition_cache = OnceLock::new();
            let local_ti_output_cache = l2p::SharedTiTokenizerOutputCache::new();
            let local_classify_cache = classify::SharedClassifyCache::new();
            let mut local_result = partition::build_partition_id_map_and_terminal_dwa(
                &label,
                &local.tokenizer,
                sub_vocab,
                terminal_coloring,
                use_terminal_coloring,
                ignore_terminal,
                grammar,
                &always_allowed_follows,
                disallowed_follows,
                &token_path_disallowed_follows,
                &normalized_token_path_disallowed_follows,
                &local_flat_trans,
                None,
                Some(&local_vocab_dfa_cache),
                Some(&local_original_vocab_dfa_cache),
                Some(&local_original_vocab_analysis_dfa_cache),
                Some(&local_transition_cache),
                Some(&local_ti_output_cache),
                Some(&local_classify_cache),
            );
            if let Some(parts) = local_result.as_mut()
                && lift_partition_terminal_dwas_to_global(parts, &local.global_to_local).is_some()
            {
                parts.profile.id_map_ms += local.build_ms;
                if compile_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][partition_local_synthesis] partition={} horizon={} global_states={} local_states={} local_transitions={} build_ms={:.3} selected=true",
                        label,
                        sub_vocab.entries.values().map(Vec::len).max().unwrap_or(0),
                        tokenizer.num_states(),
                        local.tokenizer.num_states(),
                        local.tokenizer.transition_count(),
                        local.build_ms,
                    );
                }
                return (
                    Some((
                        local_result.expect("selected local partition result"),
                        started_at.elapsed().as_secs_f64() * 1000.0,
                    )),
                    idx,
                );
            }
            if compile_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][partition_local_synthesis] partition={} horizon={} global_states={} local_states={} build_ms={:.3} selected=false reason=build_or_lift_failed",
                    label,
                    sub_vocab.entries.values().map(Vec::len).max().unwrap_or(0),
                    tokenizer.num_states(),
                    local.tokenizer.num_states(),
                    local.build_ms,
                );
            }
        }

        let result = partition::build_partition_id_map_and_terminal_dwa(
            &label,
            tokenizer,
            sub_vocab,
            terminal_coloring,
            use_terminal_coloring,
            ignore_terminal,
            grammar,
            &always_allowed_follows,
            disallowed_follows,
            &token_path_disallowed_follows,
            &normalized_token_path_disallowed_follows,
            &flat_trans,
            Some(global_max_length_state_map),
            Some(&shared_vocab_dfa_cache),
            Some(&shared_original_vocab_dfa_cache),
            Some(&shared_original_vocab_analysis_dfa_cache),
            Some(shared_transition_cache),
            Some(&shared_ti_output_cache),
            Some(&shared_classify_cache),
        )
        .map(|pair| (pair, started_at.elapsed().as_secs_f64() * 1000.0));
        (result, idx)
    };
    let serial_profile_partition_schedule = compile_profile_uses_serial_partition_schedule();
    let partition_build_started_at = Instant::now();
    let partition_results: Vec<(Option<(types::PartitionTerminalDwas, f64)>, usize)> =
        if serial_profile_partition_schedule {
            sub_vocabs
                .iter()
                .enumerate()
                .map(|(idx, sub_vocab)| build_partition(idx, sub_vocab))
                .collect()
        } else {
            sub_vocabs
                .par_iter()
                .enumerate()
                .map(|(idx, sub_vocab)| build_partition(idx, sub_vocab))
                .collect()
        };
    let partition_build_wall_ms =
        partition_build_started_at.elapsed().as_secs_f64() * 1000.0;


    let partition_result_finalize_started_at = Instant::now();
    let partition_ms: Vec<f64> = {
        let mut ms = vec![0.0; sub_vocabs.len()];
        for (result, idx) in &partition_results {
            ms[*idx] = result.as_ref().map(|(_, m)| *m).unwrap_or(0.0);
        }
        ms
    };
    let dominant_partition_profile = partition_results
        .iter()
        .filter_map(|(result, _)| result.as_ref().map(|(parts, ms)| (parts.profile, *ms)))
        .max_by(|(_, left_ms), (_, right_ms)| left_ms.total_cmp(right_ms))
        .map(|(phase_profile, _)| phase_profile)
        .unwrap_or_default();

    // Gather like construction families across every vocabulary partition.
    // The split-off L2P vocabulary uses the L1 builder and therefore belongs in
    // the L1 family even though its active terminal mask is the L2P mask.
    let mut l1_pairs: Vec<LocalIdMapTerminalDwa> = Vec::new();
    let mut l2p_pairs: Vec<LocalIdMapTerminalDwa> = Vec::new();
    for (result, _idx) in partition_results {
        if let Some((parts, _)) = result {
            if let Some(l1) = parts.l1 {
                l1_pairs.push(l1);
            }
            if let Some(split_l1) = parts.l2p_single_l1 {
                l1_pairs.push(split_l1);
            }
            if let Some(l2p) = parts.l2p {
                l2p_pairs.push(l2p);
            }
        }
    }

    if l1_pairs.is_empty() && l2p_pairs.is_empty() {
        let num_states = tokenizer.num_states() as usize;
        let empty_map = InternalIdMap {
            tokenizer_states: ManyToOneIdMap {
                original_to_internal: vec![0u32; num_states],
                internal_to_originals: vec![(0..num_states as u32).collect()],
                representative_original_ids: vec![0],
            },
            vocab_tokens: ManyToOneIdMap {
                original_to_internal: Vec::new(),
                internal_to_originals: Vec::new(),
                representative_original_ids: Vec::new(),
            },
            deferred_vocab_singleton_original_ids: None,
        };
        return (
            TerminalDwaFamilies {
                l1: Some(MappedArtifact::new(
                    TerminalAutomaton::Dwa(DWA::new(1, 0)),
                    empty_map,
                )),
                l2p: None,
                special: None,
            },
            profile,
        );
    }

    let partition_result_finalize_ms =
        partition_result_finalize_started_at.elapsed().as_secs_f64() * 1000.0;
    let num_tokenizer_states = tokenizer.num_states() as usize;
    let max_token_id = vocab.max_token_id();

    let did_global_merge = l1_pairs.len() > 1 || l2p_pairs.len() > 1;
    let family_merge_started_at = Instant::now();
    let (l1_family, l2p_family) = rayon::join(
        || {
            (!l1_pairs.is_empty()).then(|| {
                let family = merge::merge_id_maps_and_terminal_dwas(
                    l1_pairs,
                    num_tokenizer_states,
                    max_token_id,
                );
                let profile = family.profile;
                (
                    MappedArtifact::new(TerminalAutomaton::Dwa(family.dwa), family.id_map),
                    profile,
                )
            })
        },
        || {
            (!l2p_pairs.is_empty()).then(|| {
                if let Some((nwa, id_map, profile)) =
                    merge::try_merge_id_maps_and_token_deterministic_nwa(
                        &l2p_pairs,
                        num_tokenizer_states,
                        max_token_id,
                    )
                {
                    return (
                        MappedArtifact::new(TerminalAutomaton::TokenDeterministicNwa(nwa), id_map),
                        profile,
                    );
                }
                let family = merge::merge_id_maps_and_terminal_dwas(
                    l2p_pairs,
                    num_tokenizer_states,
                    max_token_id,
                );
                let profile = family.profile;
                (
                    MappedArtifact::new(TerminalAutomaton::Dwa(family.dwa), family.id_map),
                    profile,
                )
            })
        },
    );
    let family_merge_wall_ms = family_merge_started_at.elapsed().as_secs_f64() * 1000.0;
    let dominant_family_profile = [l1_family.as_ref(), l2p_family.as_ref()]
        .into_iter()
        .flatten()
        .map(|(_, profile)| *profile)
        .max_by(|left, right| left.global_merge_ms.total_cmp(&right.global_merge_ms))
        .unwrap_or_default();
    let post_ti_ignore_started_at = Instant::now();
    let erase_family_ignore = |family: Option<(MappedArtifact<TerminalAutomaton>, TerminalDwaPhaseProfile)>| {
        family.map(|(family, _)| {
            let (automaton, id_map) = family.into_parts();
            MappedArtifact::new(erase_ignore_after_ti(automaton, ignore_terminal), id_map)
        })
    };
    let terminal_families = TerminalDwaFamilies {
        l1: erase_family_ignore(l1_family),
        l2p: erase_family_ignore(l2p_family),
        special: None,
    };
    let post_ti_ignore_ms = post_ti_ignore_started_at.elapsed().as_secs_f64() * 1000.0;
    let merge_ms = family_merge_wall_ms;

    let post_merge_bookkeeping_started_at = Instant::now();
    profile.add_assign(dominant_partition_profile);
    profile.add_assign(dominant_family_profile);
    profile.terminal_dwa_ms += post_ti_ignore_ms;
    profile.global_merge_ms = if did_global_merge { merge_ms } else { 0.0 };
    let post_merge_bookkeeping_ms =
        post_merge_bookkeeping_started_at.elapsed().as_secs_f64() * 1000.0;
    let split_terminal_dwa_total_ms = total_started_at.elapsed().as_secs_f64() * 1000.0;
    profile.split_terminal_dwa_total_ms = split_terminal_dwa_total_ms;
    let accounted_wall_ms = stage_setup_ms
        + partition_vocab_ms
        + shared_cache_setup_ms
        + partition_build_wall_ms
        + partition_result_finalize_ms
        + merge_ms
        + post_ti_ignore_ms
        + post_merge_bookkeeping_ms;
    let timing_residual_ms = (split_terminal_dwa_total_ms - accounted_wall_ms).max(0.0);

    if compile_profile_enabled() {
        let partition_detail: String = sub_vocabs
            .iter()
            .enumerate()
            .map(|(i, sv)| {
                format!(
                    "p{}_tokens={} p{}_ms={:.3}",
                    i,
                    sv.entries.len(),
                    i,
                    partition_ms[i]
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!(
            "[glrmask/profile][split_terminal_dwa] partition_vocab_ms={:.3} {} family_merge_wall_ms={:.3} global_merge_ms={:.3} split_terminal_dwa_total_ms={:.3} critical_path_id_map_ms={:.3} critical_path_terminal_dwa_ms={:.3} critical_path_compact_ms={:.3} critical_path_profile_ms={:.3} total_ms={:.3}",
            partition_vocab_ms,
            partition_detail,
            family_merge_wall_ms,
            merge_ms,
            split_terminal_dwa_total_ms,
            profile.id_map_ms,
            profile.terminal_dwa_ms,
            profile.compact_ms,
            profile.total_ms(),
            split_terminal_dwa_total_ms,
        );
        eprintln!(
            "[glrmask/profile][split_terminal_dwa_wall] scheduler={} stage_setup_ms={:.3} partition_vocab_ms={:.3} shared_cache_setup_ms={:.3} partition_build_wall_ms={:.3} partition_result_finalize_ms={:.3} family_merge_wall_ms={:.3} global_merge_ms={:.3} post_ti_ignore_ms={:.3} post_merge_bookkeeping_ms={:.3} accounted_wall_ms={:.3} timing_residual_ms={:.3} total_ms={:.3}",
            if serial_profile_partition_schedule { "serial_profile_1t" } else { "rayon" },
            stage_setup_ms,
            partition_vocab_ms,
            shared_cache_setup_ms,
            partition_build_wall_ms,
            partition_result_finalize_ms,
            family_merge_wall_ms,
            merge_ms,
            post_ti_ignore_ms,
            post_merge_bookkeeping_ms,
            accounted_wall_ms,
            timing_residual_ms,
            split_terminal_dwa_total_ms,
        );
    }

    (terminal_families, profile)
}

/// Compatibility wrapper for callers that still require one terminal
/// automaton.  The compiler pipeline consumes the families directly and merges
/// their parser DWAs instead.
pub(crate) fn build_id_map_and_terminal_dwa_with_precomputed_global_max_length(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    flat_trans: Arc<[u32]>,
    global_max_length_state_map: &ManyToOneIdMap,
    external_classify_cache: Option<&classify::SharedClassifyCache>,
) -> (MappedArtifact<TerminalAutomaton>, TerminalDwaPhaseProfile) {
    let (families, mut profile) =
        build_terminal_dwa_families_with_precomputed_global_max_length(
            tokenizer,
            vocab,
            terminal_coloring,
            use_terminal_coloring,
            ignore_terminal,
            grammar,
            disallowed_follows,
            flat_trans,
            global_max_length_state_map,
            external_classify_cache,
            None,
            None,
            None,
        );
    let family_count = families.len();
    let final_merge_started_at = Instant::now();
    let mapped_dwas = families
        .into_vec()
        .into_iter()
        .map(|family| {
            let (automaton, id_map) = family.into_parts();
            let dwa = match automaton {
                TerminalAutomaton::Dwa(dwa) => dwa,
                TerminalAutomaton::TokenDeterministicNwa(nwa)
                | TerminalAutomaton::EpsilonNwa(nwa) => {
                    crate::automata::weighted::determinize::determinize(&nwa)
                        .expect("terminal family compatibility merge requires an acyclic NWA")
                }
            };
            MappedArtifact::new(dwa, id_map)
        })
        .collect();
    let merged = merge::merge_mapped_dwas(
        mapped_dwas,
        tokenizer.num_states() as usize,
        vocab.max_token_id(),
    );
    let final_merge_ms = if family_count > 1 {
        final_merge_started_at.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };
    profile.global_merge_ms += final_merge_ms;
    profile.split_terminal_dwa_total_ms += final_merge_ms;
    let (dwa, id_map) = merged.into_parts();
    (
        MappedArtifact::new(TerminalAutomaton::Dwa(dwa), id_map),
        profile,
    )
}

pub(crate) fn build_id_map_and_terminal_dwa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    external_classify_cache: Option<&classify::SharedClassifyCache>,
) -> (MappedArtifact<TerminalAutomaton>, TerminalDwaPhaseProfile, ManyToOneIdMap) {
    let mut profile = TerminalDwaPhaseProfile::default();

    let flat_trans_started_at = Instant::now();
    let flat_trans: Arc<[u32]> = Arc::from(l1::build_flat_transition_table(tokenizer));
    let flat_trans_ms = flat_trans_started_at.elapsed().as_secs_f64() * 1000.0;

    let global_max_length_started_at = Instant::now();
    let global_max_length_state_map =
        build_global_max_length_state_map(tokenizer, vocab, &flat_trans);
    let global_max_length_ms = global_max_length_started_at.elapsed().as_secs_f64() * 1000.0;

    let (mapped_dwa, mut inner_profile) =
        build_id_map_and_terminal_dwa_with_precomputed_global_max_length(
            tokenizer,
            vocab,
            terminal_coloring,
            use_terminal_coloring,
            ignore_terminal,
            grammar,
            disallowed_follows,
            flat_trans,
            &global_max_length_state_map,
            external_classify_cache,
        );
    inner_profile.terminal_dwa_ms += flat_trans_ms;
    inner_profile.id_map_ms += global_max_length_ms;
    profile.add_assign(inner_profile);

    (mapped_dwa, profile, global_max_length_state_map)
}

#[cfg(test)]
mod tests {
    use super::{
        AutomaticBranchActiveStateMapStrategy,
        DEFAULT_GLOBAL_MAX_LENGTH_STABLE_SIGNATURE_CELL_LIMIT,
        automatic_branch_active_state_map_strategy, should_auto_use_global_max_length,
    };

    #[test]
    fn branch_active_state_map_auto_gate_selects_only_amortized_l1_regimes() {
        use AutomaticBranchActiveStateMapStrategy::{
            DenseRequiresFastProjection, None, VeryLargeProfile,
        };

        assert_eq!(
            automatic_branch_active_state_map_strategy("p4.l1", 21_308, 190, 45_180),
            DenseRequiresFastProjection,
        );
        assert_eq!(
            automatic_branch_active_state_map_strategy("p5.l1", 4_261, 233, 26_624),
            DenseRequiresFastProjection,
        );
        assert_eq!(
            automatic_branch_active_state_map_strategy("p2.l1", 82_266, 229, 48_002),
            VeryLargeProfile,
        );

        // Medium-state/medium-vocabulary work shifts the critical path, while
        // the small long-horizon lane remains above the fast-projected cutoff.
        assert_eq!(
            automatic_branch_active_state_map_strategy("p1.l1", 15_224, 201, 37_079),
            None,
        );
        assert_eq!(
            automatic_branch_active_state_map_strategy("p6.l1", 630, 192, 97_024),
            None,
        );
        assert_eq!(
            automatic_branch_active_state_map_strategy("p4.l2p", 17_646, 4, 45_180),
            None,
        );
    }

    #[test]
    fn global_max_length_auto_gate_bounds_stable_signature_matrix() {
        assert!(should_auto_use_global_max_length(
            50_001,
            1,
            DEFAULT_GLOBAL_MAX_LENGTH_STABLE_SIGNATURE_CELL_LIMIT,
        ));
        assert!(!should_auto_use_global_max_length(
            50_000,
            1,
            DEFAULT_GLOBAL_MAX_LENGTH_STABLE_SIGNATURE_CELL_LIMIT,
        ));

        // o9802 has 71,429 states and a near-full byte alphabet. The stable
        // global refinement is optional, and its first signature matrix alone
        // exceeds the bounded auto budget by almost an order of magnitude.
        assert!(!should_auto_use_global_max_length(
            71_429,
            89,
            DEFAULT_GLOBAL_MAX_LENGTH_STABLE_SIGNATURE_CELL_LIMIT,
        ));
    }
}
