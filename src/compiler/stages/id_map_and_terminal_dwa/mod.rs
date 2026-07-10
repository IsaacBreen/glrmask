//! Top-level id_map + terminal DWA builder.
//!
//! The canonical path splits the vocabulary into 3 character-type partitions,
//! builds a per-partition `(InternalIdMap, DWA)`, and merges the results into
//! the final global `(InternalIdMap, DWA)`.

use crate::automata::lexer::Lexer;
pub(crate) mod classify;
pub(crate) mod grammar_helpers;
pub(crate) mod l1;
pub(crate) mod l2p;
pub(crate) mod merge;
pub(crate) mod partition;
pub(crate) mod types;

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap, MappedArtifact};
use crate::ds::bitset::BitSet;
use crate::grammar::flat::TerminalID;
use crate::Vocab;

use classify::classify_vocab_char_type;
use grammar_helpers::ignore_transparent_disallowed_follows;
use l2p::equivalence_analysis::state_equivalence::{
    resolve_global_pipeline_config, run_state_equivalence_pipeline, StateEquivalenceScope,
};
use types::{
    compile_profile_enabled, compile_profile_uses_serial_partition_schedule,
    LocalIdMapTerminalDwa, TerminalColoring, TerminalDwaPhaseProfile,
};

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
    sub_vocabs: Arc<[Vocab]>,
}

impl crate::vocab::VocabDerivedArtifact for CharTypeSubVocabs {}

fn vocab_from_token_partitions(vocab: &Vocab, token_partitions: Vec<Vec<u32>>) -> Arc<[Vocab]> {
    token_partitions
        .into_iter()
        .map(|token_ids| {
            let entries = token_ids
                .into_iter()
                .filter_map(|token_id| vocab.entries.get(&token_id).map(|bytes| (token_id, bytes.clone())))
                .collect();
            Vocab::new(entries, None)
        })
        .collect::<Vec<_>>()
        .into()
}

fn build_char_type_sub_vocabs(vocab: &Vocab) -> Arc<[Vocab]> {
    if let Some(cached) = vocab.vocab_derived_cache_get::<CharTypeSubVocabs>() {
        return Arc::clone(&cached.sub_vocabs);
    }

    let mut partition_entries: Vec<Vec<(u32, Vec<u8>)>> = (0..9).map(|_| Vec::new()).collect();
    for (&token_id, bytes) in vocab.entries.iter() {
        let idx = classify_vocab_char_type(bytes) as usize;
        partition_entries[idx].push((token_id, bytes.clone()));
    }
    let sub_vocabs: Arc<[Vocab]> = partition_entries
        .into_iter()
        .map(|entries| Vocab::new(entries, None))
        .collect::<Vec<_>>()
        .into();
    vocab.vocab_derived_cache_set(Arc::new(CharTypeSubVocabs {
        sub_vocabs: Arc::clone(&sub_vocabs),
    }));
    sub_vocabs
}

pub(crate) fn prepare_vocab_for_terminal_dwa(vocab: &Vocab) {
    classify::prepare_vocab_for_terminal_classification(vocab);
    l1::prepare_l1_identity_vocab_order(vocab);

    if std::env::var("GLRMASK_PARTITION_SCHEME").as_deref().unwrap_or("char_type") == "char_type" {
        for sub_vocab in build_char_type_sub_vocabs(vocab).iter() {
            classify::prepare_vocab_for_terminal_classification(sub_vocab);
            l1::prepare_l1_identity_vocab_order(sub_vocab);
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
    if tokenizer.has_epsilon_transitions() {
        let state_ids = (0..num_states_u32).collect::<Vec<_>>();
        let state_map =
            ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                state_ids.clone(),
                state_ids,
            );
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][global_max_length] mode=identity skipped=true reason=epsilon_nfa states={} reps={} tokens_included=0 max_token_len=0 stable_signature_cells=0 stable_signature_cell_limit={} ms={:.3}",
                num_states,
                state_map.representative_original_ids.len(),
                global_max_length_stable_signature_cell_limit(),
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return state_map;
    }
    let token_bytes: Vec<&[u8]> = vocab
        .entries
        .values()
        .map(|bytes| bytes.as_slice())
        .collect();
    let max_token_len = token_bytes
        .iter()
        .map(|bytes| bytes.len())
        .max()
        .unwrap_or(0);
    let global_statistic =
        l2p::equivalence_analysis::state_equivalence::max_length::compute_statistic(vocab);
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
                token_bytes.len(),
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
/// 2. Builds each partition's `(InternalIdMap, DWA)` in parallel via
///    [`partition::build_partition_id_map_and_terminal_dwa`].
/// 3. Merges the 3 results via [`merge::merge_id_maps_and_terminal_dwas`].
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
    let stage_setup_ms = total_started_at.elapsed().as_secs_f64() * 1000.0;

    let partition_vocab_started_at = Instant::now();
    let requested_partition_scheme =
        std::env::var("GLRMASK_PARTITION_SCHEME").unwrap_or_else(|_| "char_type".to_string());
    // The l2p_cost scheme depends on deterministic classifier byte tables.
    // Vocabulary partitioning affects performance only, so retain exactness by
    // falling back to the ordinary byte-class partition for epsilon tokenizers.
    let partition_scheme = if tokenizer.has_epsilon_transitions()
        && matches!(
            requested_partition_scheme.as_str(),
            "l2p_cost" | "auto_l2p_cost"
        )
    {
        "char_type"
    } else {
        requested_partition_scheme.as_str()
    };
    let sub_vocabs: Arc<[Vocab]> = match partition_scheme {
        "char_type" => build_char_type_sub_vocabs(vocab),
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
            let char_token_partitions = classify::partition_vocab_char_type_tokens(vocab);
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
    let shared_transition_cache = OnceLock::new();
    let shared_cache_setup_ms =
        shared_cache_setup_started_at.elapsed().as_secs_f64() * 1000.0;

    use rayon::prelude::*;
    let build_partition = |idx: usize,
                           sub_vocab: &Vocab|
     -> (Option<(LocalIdMapTerminalDwa, f64)>, usize) {
        let started_at = Instant::now();
        let label = format!("p{}", idx);
        let result = partition::build_partition_id_map_and_terminal_dwa(
            &label,
            tokenizer,
            sub_vocab,
            terminal_coloring,
            use_terminal_coloring,
            ignore_terminal,
            grammar,
            disallowed_follows,
            &token_path_disallowed_follows,
            &flat_trans,
            Some(global_max_length_state_map),
            Some(&shared_vocab_dfa_cache),
            Some(&shared_original_vocab_dfa_cache),
            Some(&shared_original_vocab_analysis_dfa_cache),
            Some(&shared_transition_cache),
            Some(&shared_classify_cache),
        )
        .map(|pair| (pair, started_at.elapsed().as_secs_f64() * 1000.0));
        (result, idx)
    };
    let serial_profile_partition_schedule = compile_profile_uses_serial_partition_schedule();
    let partition_build_started_at = Instant::now();
    let partition_results: Vec<(Option<(LocalIdMapTerminalDwa, f64)>, usize)> =
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
        .filter_map(|(result, _)| result.as_ref().map(|(pair, ms)| (pair.profile, *ms)))
        .max_by(|(_, left_ms), (_, right_ms)| left_ms.total_cmp(right_ms))
        .map(|(phase_profile, _)| phase_profile)
        .unwrap_or_default();

    // Collect non-None results.
    let mut pairs: Vec<LocalIdMapTerminalDwa> = Vec::new();
    for (result, _idx) in partition_results {
        if let Some((pair, _)) = result {
            pairs.push(pair);
        }
    }

    if pairs.is_empty() {
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
        };
        return (
            MappedArtifact::new(TerminalAutomaton::Dwa(DWA::new(1, 0)), empty_map),
            profile,
        );
    }

    let partition_result_finalize_ms =
        partition_result_finalize_started_at.elapsed().as_secs_f64() * 1000.0;
    let num_tokenizer_states = tokenizer.num_states() as usize;
    let max_token_id = vocab.max_token_id();

    let did_global_merge = pairs.len() > 1;
    let merge_started_at = Instant::now();
    let token_deterministic_nwa_enabled = std::env::var_os(
        "GLRMASK_EXPERIMENTAL_TOKEN_DETERMINISTIC_TERMINAL_NWA",
    )
    .is_some();
    let (merged_terminal_automaton, merged_id_map, global_merge_profile) = if !did_global_merge {
        // Single partition — already compacted by partition merge. Skip redundant global compact.
        let pair = pairs.into_iter().next().unwrap();
        (
            TerminalAutomaton::Dwa(pair.dwa),
            pair.id_map,
            TerminalDwaPhaseProfile::default(),
        )
    } else if token_deterministic_nwa_enabled {
        if let Some((nwa, id_map, merge_profile)) =
            merge::try_merge_id_maps_and_token_deterministic_nwa(
                &pairs,
                num_tokenizer_states,
                max_token_id,
            )
        {
            (
                TerminalAutomaton::TokenDeterministicNwa(nwa),
                id_map,
                merge_profile,
            )
        } else {
            let merged =
                merge::merge_id_maps_and_terminal_dwas(pairs, num_tokenizer_states, max_token_id);
            (
                TerminalAutomaton::Dwa(merged.dwa),
                merged.id_map,
                merged.profile,
            )
        }
    } else {
        let merged = merge::merge_id_maps_and_terminal_dwas(pairs, num_tokenizer_states, max_token_id);
        (
            TerminalAutomaton::Dwa(merged.dwa),
            merged.id_map,
            merged.profile,
        )
    };
    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;

    let post_merge_bookkeeping_started_at = Instant::now();
    profile.add_assign(dominant_partition_profile);
    profile.add_assign(global_merge_profile);
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
            "[glrmask/profile][split_terminal_dwa] partition_vocab_ms={:.3} {} global_merge_ms={:.3} split_terminal_dwa_total_ms={:.3} critical_path_id_map_ms={:.3} critical_path_terminal_dwa_ms={:.3} critical_path_compact_ms={:.3} critical_path_profile_ms={:.3} total_ms={:.3}",
            partition_vocab_ms,
            partition_detail,
            merge_ms,
            split_terminal_dwa_total_ms,
            profile.id_map_ms,
            profile.terminal_dwa_ms,
            profile.compact_ms,
            profile.total_ms(),
            split_terminal_dwa_total_ms,
        );
        eprintln!(
            "[glrmask/profile][split_terminal_dwa_wall] scheduler={} stage_setup_ms={:.3} partition_vocab_ms={:.3} shared_cache_setup_ms={:.3} partition_build_wall_ms={:.3} partition_result_finalize_ms={:.3} global_merge_ms={:.3} post_merge_bookkeeping_ms={:.3} accounted_wall_ms={:.3} timing_residual_ms={:.3} total_ms={:.3}",
            if serial_profile_partition_schedule { "serial_profile_1t" } else { "rayon" },
            stage_setup_ms,
            partition_vocab_ms,
            shared_cache_setup_ms,
            partition_build_wall_ms,
            partition_result_finalize_ms,
            merge_ms,
            post_merge_bookkeeping_ms,
            accounted_wall_ms,
            timing_residual_ms,
            split_terminal_dwa_total_ms,
        );
    }

    (MappedArtifact::new(merged_terminal_automaton, merged_id_map), profile)
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
        should_auto_use_global_max_length,
        DEFAULT_GLOBAL_MAX_LENGTH_STABLE_SIGNATURE_CELL_LIMIT,
    };

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
