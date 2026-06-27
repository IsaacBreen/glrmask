//! L2+ terminal DWA: full NWA-based construction for terminals with path length ≥ 2.
//!
//! Uses the same structure as the pre-partition/path-length code (commit 67146d8):
//! build vocab trie → compute possible_matches → seed root nodes → trie-walk
//! NWA build → postprocess (always_allowed → collapse → disallowed → prune →
//! canonicalize) → determinize → minimize.
//!
//! The only structural difference from the old code is `active_terminals`
//! filtering: terminals not in the L2+ set are skipped during the trie walk.

use crate::automata::lexer::Lexer;
pub(crate) mod equivalence_analysis;
pub(crate) mod nwa_builder;
pub(crate) mod postprocess;
mod terminal_interchangeability;

use std::collections::BTreeMap;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::minimize::minimize_owned;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::possible_matches::PossibleMatchesComputer;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::id_map_and_terminal_dwa::types::LocalIdMapTerminalDwa;
use crate::ds::bitset::BitSet;
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;
use crate::Vocab;
use rustc_hash::FxHashMap;

use super::grammar_helpers::compute_always_allowed_follows;
use super::types::{compile_profile_enabled, TerminalColoring, TerminalDwaPhaseProfile};
use nwa_builder::{build_nwa_via_trie_walk, internal_vocab_entries, seed_root_nodes};
use terminal_interchangeability::TerminalInterchangeability;
use postprocess::{
    apply_disallowed_follow_constraints, canonicalize_acyclic_nwa, collapse_always_allowed,
    dwa_to_nwa_with_ignore_as_epsilon,
    prune_non_coreachable_states, SharedDisallowedFollowDfaCache,
};

fn l2p_timing_profile_enabled() -> bool {
    compile_profile_enabled() || std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some()
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct SimplifyCacheKey {
    active_words: Vec<u64>,
    relevant_words: [u64; 4],
}

#[derive(Default)]
pub(crate) struct SharedSimplifyCache {
    entries: Mutex<FxHashMap<SimplifyCacheKey, Arc<SimplifyCacheEntry>>>,
}

struct SimplifyCacheEntry {
    result: Mutex<Option<Result<Arc<(Tokenizer, ManyToOneIdMap)>, Arc<str>>>>,
    ready: Condvar,
}

impl SimplifyCacheEntry {
    fn new() -> Self {
        Self {
            result: Mutex::new(None),
            ready: Condvar::new(),
        }
    }
}

impl SharedSimplifyCache {
    fn key(active_terminals: &[bool], relevant_bytes: &[bool; 256]) -> SimplifyCacheKey {
        let mut active_words = vec![0u64; active_terminals.len().div_ceil(64)];
        for (idx, &active) in active_terminals.iter().enumerate() {
            if active {
                active_words[idx >> 6] |= 1u64 << (idx & 63);
            }
        }

        let mut relevant_words = [0u64; 4];
        for (idx, &relevant) in relevant_bytes.iter().enumerate() {
            if relevant {
                relevant_words[idx >> 6] |= 1u64 << (idx & 63);
            }
        }

        SimplifyCacheKey {
            active_words,
            relevant_words,
        }
    }

    fn simplify_for_terminals(
        &self,
        tokenizer: &Tokenizer,
        active_terminals: &[bool],
        relevant_bytes: &[bool; 256],
    ) -> (Tokenizer, ManyToOneIdMap, bool) {
        let key = Self::key(active_terminals, relevant_bytes);
        let (entry, owns_compute) = {
            let mut entries = self.entries.lock().unwrap();
            if let Some(entry) = entries.get(&key) {
                (entry.clone(), false)
            } else {
                let entry = Arc::new(SimplifyCacheEntry::new());
                entries.insert(key, entry.clone());
                (entry, true)
            }
        };

        if owns_compute {
            match catch_unwind(AssertUnwindSafe(|| {
                Arc::new(tokenizer.simplify_for_terminals(active_terminals, Some(relevant_bytes)))
            })) {
                Ok(computed) => {
                    *entry.result.lock().unwrap() = Some(Ok(computed.clone()));
                    entry.ready.notify_all();
                    return (computed.0.clone(), computed.1.clone(), false);
                }
                Err(payload) => {
                    *entry.result.lock().unwrap() =
                        Some(Err("tokenizer.simplify_for_terminals panicked".into()));
                    entry.ready.notify_all();
                    resume_unwind(payload);
                }
            }
        }

        let mut result = entry.result.lock().unwrap();
        loop {
            match result.as_ref() {
                Some(Ok(cached)) => {
                    return (cached.0.clone(), cached.1.clone(), true);
                }
                Some(Err(message)) => {
                    panic!("{message}");
                }
                None => {
                    result = entry.ready.wait(result).unwrap();
                }
            }
        }
    }
}

fn project_initial_state_map_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_L2P_PROJECT_INITIAL_STATE_MAP")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true)
    })
}

fn l2p_tokenizer_simplify_disabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_L2P_DISABLE_TOKENIZER_SIMPLIFY")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true)
    })
}

/// Enable the deliberately slow generated-swap reference construction.
/// It is opt-in until it has been validated against full-terminal builds.
fn l2p_terminal_interchangeability_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY")
            .map(|value| {
                let trimmed = value.trim();
                !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
            })
            .unwrap_or(false)
    })
}

fn l2p_terminal_interchangeability_enabled_for_partition(partition: &str) -> bool {
    if !l2p_terminal_interchangeability_enabled() {
        return false;
    }
    match std::env::var("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_PARTITIONS") {
        Ok(value) if !value.trim().is_empty() => value.split(',').any(|entry| entry.trim() == partition),
        _ => true,
    }
}

/// Reference-only: expand the local vocabulary map to singleton original
/// tokens before building the representative-terminal NWA. This deliberately
/// removes the token quotient while validating terminal expansion semantics.
fn singleton_vocab_map(existing: &ManyToOneIdMap) -> ManyToOneIdMap {
    let mut original_to_internal = vec![u32::MAX; existing.original_to_internal.len()];
    let mut representatives = Vec::new();
    for (original, &old_internal) in existing.original_to_internal.iter().enumerate() {
        if old_internal == u32::MAX {
            continue;
        }
        let internal = representatives.len() as u32;
        original_to_internal[original] = internal;
        representatives.push(original as u32);
    }
    ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
        original_to_internal,
        representatives,
    )
}


#[derive(Clone, Copy)]
struct L2PTokenLengthStats {
    max_len: usize,
    gt_4: usize,
    gt_8: usize,
    gt_16: usize,
    gt_32: usize,
    gt_64: usize,
}

fn l2p_token_length_stats(vocab: &Vocab) -> L2PTokenLengthStats {
    let mut stats = L2PTokenLengthStats {
        max_len: 0,
        gt_4: 0,
        gt_8: 0,
        gt_16: 0,
        gt_32: 0,
        gt_64: 0,
    };

    for bytes in vocab.entries.values() {
        let len = bytes.len();
        stats.max_len = stats.max_len.max(len);
        if len > 4 {
            stats.gt_4 += 1;
        }
        if len > 8 {
            stats.gt_8 += 1;
        }
        if len > 16 {
            stats.gt_16 += 1;
        }
        if len > 32 {
            stats.gt_32 += 1;
        }
        if len > 64 {
            stats.gt_64 += 1;
        }
    }

    stats
}

struct ProjectInitialStateMapProfile {
    used: bool,
    reason: &'static str,
    simplified_state_count: usize,
    projected_simplified_states: usize,
    unmapped_simplified_states_before_fill: usize,
    projected_initial_classes_before_compaction: usize,
    projected_initial_classes_after_compaction: usize,
    dead_class_added: bool,
}

impl ProjectInitialStateMapProfile {
    fn unused(reason: &'static str, simplified_state_count: usize) -> Self {
        Self {
            used: false,
            reason,
            simplified_state_count,
            projected_simplified_states: 0,
            unmapped_simplified_states_before_fill: 0,
            projected_initial_classes_before_compaction: 0,
            projected_initial_classes_after_compaction: 0,
            dead_class_added: false,
        }
    }
}

fn project_initial_state_map_for_simplified_tokenizer(
    initial_state_map: &ManyToOneIdMap,
    simplify_state_map: &ManyToOneIdMap,
) -> (Option<ManyToOneIdMap>, ProjectInitialStateMapProfile) {
    let simplified_state_count = simplify_state_map.num_internal_ids() as usize;
    if simplified_state_count == 0 {
        return (
            None,
            ProjectInitialStateMapProfile::unused("empty_simplified", simplified_state_count),
        );
    }

    let mut projected = vec![u32::MAX; simplified_state_count];
    let mut has_projected_state = false;

    for (original_state, &simplified_state) in
        simplify_state_map.original_to_internal.iter().enumerate()
    {
        if simplified_state == u32::MAX {
            continue;
        }

        let initial_class = initial_state_map
            .original_to_internal
            .get(original_state)
            .copied()
            .unwrap_or(u32::MAX);
        if initial_class == u32::MAX {
            continue;
        }

        let slot = &mut projected[simplified_state as usize];
        if *slot == u32::MAX {
            *slot = initial_class;
            has_projected_state = true;
        } else if *slot != initial_class {
            let projected_simplified_states = projected
                .iter()
                .filter(|&&initial_class| initial_class != u32::MAX)
                .count();
            let projected_initial_classes_before_compaction = projected
                .iter()
                .copied()
                .filter(|&initial_class| initial_class != u32::MAX)
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            let unmapped_simplified_states_before_fill =
                simplified_state_count.saturating_sub(projected_simplified_states);
            return (
                None,
                ProjectInitialStateMapProfile {
                    used: false,
                    reason: "mixed_initial_class",
                    simplified_state_count,
                    projected_simplified_states,
                    unmapped_simplified_states_before_fill,
                    projected_initial_classes_before_compaction,
                    projected_initial_classes_after_compaction: 0,
                    dead_class_added: false,
                },
            );
        }
    }

    if !has_projected_state {
        return (
            None,
            ProjectInitialStateMapProfile::unused("no_projected_states", simplified_state_count),
        );
    }

    let projected_simplified_states = projected
        .iter()
        .filter(|&&initial_class| initial_class != u32::MAX)
        .count();
    let projected_initial_classes_before_compaction = projected
        .iter()
        .copied()
        .filter(|&initial_class| initial_class != u32::MAX)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let unmapped_simplified_states_before_fill =
        simplified_state_count.saturating_sub(projected_simplified_states);

    let mut remapped_classes = vec![u32::MAX; initial_state_map.num_internal_ids() as usize];
    let mut next_class = 0u32;
    let compacted_projected: Vec<u32> = projected
        .into_iter()
        .map(|initial_class| {
            if initial_class == u32::MAX {
                return u32::MAX;
            }
            let slot = &mut remapped_classes[initial_class as usize];
            if *slot == u32::MAX {
                *slot = next_class;
                next_class += 1;
            }
            *slot
        })
        .collect();

    let dead_class_added = compacted_projected
        .iter()
        .any(|&initial_class| initial_class == u32::MAX);
    let projected_initial_classes_after_compaction = next_class as usize;
    (
        Some(
            ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                compacted_projected,
                next_class,
            )
            .fill_unmapped_with_new_class(),
        ),
        ProjectInitialStateMapProfile {
            used: true,
            reason: "used",
            simplified_state_count,
            projected_simplified_states,
            unmapped_simplified_states_before_fill,
            projected_initial_classes_before_compaction,
            projected_initial_classes_after_compaction,
            dead_class_added,
        },
    )
}

/// Build an L2+ id_map and terminal DWA for the given vocab and terminal set.
///
/// Builds its own id_map via `InternalIdMap::build_with_group_filter` (full DFA-
/// based equivalence analysis restricted to L2+ terminal groups). Then builds
/// the terminal DWA using the old-shaped trie-walk NWA pipeline matching the
/// 67146d8 code shape:
///
/// 1. Build internal vocab entries
/// 2. Build vocab prefix trie
/// 3. Compute possible_matches (for root node seeding)
/// 4. Create NWA, seed root nodes
/// 5. Trie-walk NWA build
/// 6. Postprocess: always_allowed → collapse → disallowed → prune → canonicalize
/// 7. Determinize → minimize, unless an enclosing local merge will perform the
///    exact determinize/minimize pass immediately afterwards.
///
/// `disallowed_follows` is threaded explicitly for id_map building.
///
/// Returns `None` if the vocab is empty.
pub(crate) fn build_l2p_id_map_and_terminal_dwa(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_interchangeability_bytes: &[bool; 256],
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    active_terminals: &[bool],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    shared_vocab_dfa_cache: Option<&equivalence_analysis::vocab::fast::SharedVocabDfaCache>,
    shared_original_vocab_dfa_cache: Option<&equivalence_analysis::vocab::fast::SharedVocabDfaCache>,
    shared_original_vocab_analysis_dfa_cache: Option<&equivalence_analysis::vocab::fast::SharedVocabAnalysisDfaCache>,
    shared_transition_cache: Option<&OnceLock<equivalence_analysis::compat::FlatTransitionCache>>,
    shared_simplify_cache: Option<&SharedSimplifyCache>,
    shared_disallowed_follow_dfa_cache: Option<&SharedDisallowedFollowDfaCache>,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    initial_state_map: Option<&ManyToOneIdMap>,
    defer_minimization_to_local_merge: bool,
) -> Option<LocalIdMapTerminalDwa> {
    if vocab.is_empty() {
        return None;
    }

    let total_started_at = Instant::now();
    let num_original_states = tokenizer.num_states() as usize;
    let num_active_terminals = active_terminals.iter().filter(|&&active| active).count();

    let mut relevant_bytes = [false; 256];
    for bytes in vocab.entries.values() {
        for &byte in bytes {
            relevant_bytes[byte as usize] = true;
        }
    }

    let terminal_interchangeability = if l2p_terminal_interchangeability_enabled_for_partition(partition_label) {
        TerminalInterchangeability::build(
            tokenizer,
            active_terminals,
            ignore_terminal,
            terminal_interchangeability_bytes,
        )
    } else {
        TerminalInterchangeability::identity(active_terminals)
    };
    // Keep representatives through raw terminal-NWA/DWA construction. Concrete
    // terminal labels are restored by the post-DWA copy/union/determinize path.
    let reference_terminal_expansion = !terminal_interchangeability.is_identity();
    if reference_terminal_expansion && std::env::var_os("GLRMASK_DEBUG_TERMINAL_INTERCHANGEABILITY").is_some() {
        for members in terminal_interchangeability.nontrivial_classes() {
            let labels = members
                .iter()
                .map(|&terminal| format!("{}:{}", terminal, grammar.terminal_display_name(terminal)))
                .collect::<Vec<_>>();
            eprintln!(
                "[glrmask/debug][terminal_interchangeability_class] partition={} members={}",
                partition_label,
                labels.join(" | "),
            );
        }
    }
    let terminal_relabel_map = (!terminal_interchangeability.is_identity())
        .then(|| terminal_interchangeability.terminal_representative_map());
    // Keep every member matcher path in the lexer, but make it emit its
    // representative label.  This is distinct from filtering inactive bits:
    // filtering erases paths that the post-DWA expansion must later restore.
    let relabelled_tokenizer_storage = terminal_relabel_map
        .as_ref()
        .map(|map| tokenizer.relabel_terminals(map));
    let tokenizer_for_terminal_partition = relabelled_tokenizer_storage
        .as_ref()
        .unwrap_or(tokenizer);
    let tokenizer_was_relabelled = relabelled_tokenizer_storage.is_some();
    let analysis_active_terminals = terminal_interchangeability.active_representatives();
    let num_analysis_active_terminals = analysis_active_terminals
        .iter()
        .filter(|&&active| active)
        .count();

    // Strip non-active terminal bits from DFA finalizers and minimize.
    // When every terminal remains active, reuse the original tokenizer.
    // Otherwise simplify to the smaller partition-local DFA.
    //
    // Unmapped original states (states with no active-terminal future
    // under this partition) are filled into a dead class via
    // `fill_unmapped_with_new_class` after composition, so we always
    // use the simplified tokenizer.
    let simplify_started_at = Instant::now();
    let can_skip_simplify = l2p_tokenizer_simplify_disabled() || num_analysis_active_terminals == analysis_active_terminals.len();
    let (simplified_tok_storage, simplify_state_map, simplify_cache_hit) = if can_skip_simplify {
        (None, None, false)
    } else if !tokenizer_was_relabelled {
        if let Some(cache) = shared_simplify_cache {
            let (tok, map, cache_hit) = cache.simplify_for_terminals(
                tokenizer_for_terminal_partition,
                analysis_active_terminals,
                &relevant_bytes,
            );
            (Some(tok), Some(map), cache_hit)
        } else {
            let (tok, map) = tokenizer_for_terminal_partition
                .simplify_for_terminals(analysis_active_terminals, Some(&relevant_bytes));
            (Some(tok), Some(map), false)
        }
    } else {
        let (tok, map) = tokenizer_for_terminal_partition
            .simplify_for_terminals(analysis_active_terminals, Some(&relevant_bytes));
        (Some(tok), Some(map), false)
    };
    let simplify_ms = simplify_started_at.elapsed().as_secs_f64() * 1000.0;
    let use_simplified_tok = simplified_tok_storage.is_some();
    let tokenizer_for_build = simplified_tok_storage
        .as_ref()
        .unwrap_or(tokenizer_for_terminal_partition);
    let candidate_unmapped_original_states = simplify_state_map.as_ref().map_or(0, |state_map| {
        state_map
            .original_to_internal
            .iter()
            .filter(|&&state| state == u32::MAX)
            .count()
    });
    let projection_enabled = project_initial_state_map_enabled();
    let (projected_initial_state_map, projection_profile) = if !projection_enabled {
        (
            None,
            ProjectInitialStateMapProfile::unused(
                "env_disabled",
                simplify_state_map
                    .as_ref()
                    .map(|simplified| simplified.num_internal_ids() as usize)
                    .unwrap_or(0),
            ),
        )
    } else if initial_state_map.is_none() {
        (
            None,
            ProjectInitialStateMapProfile::unused(
                "no_initial_map",
                simplify_state_map
                    .as_ref()
                    .map(|simplified| simplified.num_internal_ids() as usize)
                    .unwrap_or(0),
            ),
        )
    } else if simplify_state_map.is_none() {
        (
            None,
            ProjectInitialStateMapProfile::unused("no_simplify_map", 0),
        )
    } else {
        project_initial_state_map_for_simplified_tokenizer(
            initial_state_map.expect("checked above"),
            simplify_state_map.as_ref().expect("checked above"),
        )
    };
    let equivalence_initial_state_map = if use_simplified_tok {
        projected_initial_state_map.as_ref()
    } else {
        initial_state_map
    };
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l2p_projection] partition={} projection_enabled={} simplify_branch_active={} projected_initial_state_map_used={} reason={} simplified_state_count={} projected_simplified_states={} unmapped_simplified_states_before_fill={} projected_initial_classes_before_compaction={} projected_initial_classes_after_compaction={} dead_class_added={}",
            partition_label,
            projection_enabled,
            use_simplified_tok,
            projection_profile.used,
            projection_profile.reason,
            projection_profile.simplified_state_count,
            projection_profile.projected_simplified_states,
            projection_profile.unmapped_simplified_states_before_fill,
            projection_profile.projected_initial_classes_before_compaction,
            projection_profile.projected_initial_classes_after_compaction,
            projection_profile.dead_class_added,
        );
    }

    // ---- Step 1: Equivalence analysis (on simplified tokenizer) ----
    let id_map_started_at = Instant::now();
    // The original tokenizer has a single transition relation across every
    // unsimplified partition. Derive its exact byte partition lazily from
    // sparse lexer edges once; simplified tokenizers retain their local cache.
    let shared_base_setup_started_at = Instant::now();
    let equivalence_vocab_dfa_cache = if tokenizer_was_relabelled {
        None
    } else if use_simplified_tok {
        shared_vocab_dfa_cache
    } else if let (Some(original_cache), Some(transition_cache), Some(flat_trans)) = (
        shared_original_vocab_dfa_cache,
        shared_transition_cache,
        flat_trans,
    ) {
        original_cache.get_or_init(|| {
            let transition_cache = transition_cache.get_or_init(|| {
                equivalence_analysis::compat::derive_flat_transition_cache(
                    tokenizer_for_build,
                    Arc::clone(flat_trans),
                )
            });
            equivalence_analysis::vocab::fast::SharedVocabDfaBase::build_from_flat_transition_cache(
                transition_cache,
            )
        });
        Some(original_cache)
    } else {
        shared_vocab_dfa_cache
    };
    let shared_base_setup_ms = shared_base_setup_started_at.elapsed().as_secs_f64() * 1000.0;
    let shared_analysis_dfa_cache = (!use_simplified_tok && !tokenizer_was_relabelled)
        .then_some(shared_original_vocab_analysis_dfa_cache)
        .flatten();
    // Do not remove this note. Do not remove the instruction saying not to
    // remove this note. L2P state/vocab equivalence must always run fully.
    // Max-length may be skipped in controlled cases, but the full exact
    // state/vocab equivalence pass must not be bypassed. Do not reintroduce
    // fast-sound, identity, lex-dedup, or similar shortcut id-map paths.
    let fast_sound_id_map_used = false;
    // Keep equivalence analysis in the same terminal universe as the tokenizer
    // consumed by the L2P NWA builder.  When simplification is enabled, inactive
    // terminal bits have already been removed from `tokenizer_for_build`, so no
    // extra active-group filter is needed.  When simplification is disabled, the
    // builder still consults full-tokenizer future-terminal sets in several
    // paths, so filtering equivalence by `active_terminals` would be unsound: it
    // could merge states that the later NWA construction still distinguishes.
    let (simplified_id_map, equiv_profile) =
        equivalence_analysis::combined::analyze_equivalences_with_group_filter(
            partition_label,
            tokenizer_for_build,
            vocab,
            disallowed_follows,
            ignore_terminal,
            None,
            equivalence_vocab_dfa_cache,
            shared_analysis_dfa_cache,
            shared_base_setup_ms,
            if use_simplified_tok || tokenizer_was_relabelled { None } else { flat_trans },
            equivalence_initial_state_map,
        );
    let mut simplified_id_map = simplified_id_map;
    if reference_terminal_expansion {
        simplified_id_map.vocab_tokens = singleton_vocab_map(&simplified_id_map.vocab_tokens);
    }
    let nwa_build_id_map = simplified_id_map.clone();
    let tokenizer_for_nwa_build = tokenizer_for_build;
    let terminals_for_nwa_build = analysis_active_terminals;
    let id_map_ms = id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    // tsid_fallback is independent of the NWA build / postprocess /
    // determinize / minimize pipeline: it only feeds into the final
    // `LocalIdMapTerminalDwa` struct. Run it in parallel with steps 2-8
    // via rayon::join. On the critical partition this cuts ~150-200ms
    // off the per-partition sequential critical path.
    let (
        dwa,
        vocab_tree_ms,
        possible_matches_ms,
        seed_ms,
        trie_build_ms,
        always_allowed_ms,
        collapse_ms,
        disallowed_ms,
        prune_ms,
        canonicalize_ms,
        determinize_ms,
        minimize_ms,
        internal_vocab_count,
        nwa_states_after_build,
        nwa_states_after_collapse,
        nwa_states_after_disallowed,
        nwa_states_after_prune,
        nwa_states_after_canonicalize,
        dwa_stats_before_compact,
        dwa_stats_after_compact,
        early_none,
    ) = {
        // ---- Step 2-3: Internal vocab + prefix tree ----
        let vocab_tree_started_at = Instant::now();
        let internal_vocab = internal_vocab_entries(vocab, &nwa_build_id_map);
        let internal_vocab_count = internal_vocab.len();
        if internal_vocab.is_empty() {
            // Signal early-None via a sentinel. Build a dummy DWA;
            // outer code will observe `early_none=true` and return.
            (
                crate::automata::weighted_u32::dwa::DWA::new(0, 0),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0usize,
                0usize,
                0usize,
                0usize,
                0usize,
                0usize,
                crate::automata::weighted_u32::dwa::DWA::new(0, 0).stats(),
                crate::automata::weighted_u32::dwa::DWA::new(0, 0).stats(),
                true,
            )
        } else {
            let full_tree = VocabPrefixTree::build_owned(
                internal_vocab
                    .iter()
                    .map(|(token_id, bytes)| (*token_id as usize, bytes.clone()))
                    .collect(),
            );
            let vocab_tree_ms = vocab_tree_started_at.elapsed().as_secs_f64() * 1000.0;

            // ---- Step 4: Possible matches (lazy via computer) ----
            let mut pm_computer = PossibleMatchesComputer::new(tokenizer_for_nwa_build);
            let possible_matches_ms = 0.0;

            // ---- Step 5: Create NWA and seed root nodes ----
            let seed_started_at = Instant::now();
            let mut nwa = NWA::new(
                nwa_build_id_map.num_tsids(),
                nwa_build_id_map.max_internal_token_id(),
            );
            let leaf_state = nwa.add_state();
            nwa.set_final_weight(leaf_state, Weight::all());
            let start_state = nwa.add_state();
            nwa.start_states_mut().push(start_state);

            let roots = seed_root_nodes(&mut nwa, start_state, &nwa_build_id_map);
            let seed_ms = seed_started_at.elapsed().as_secs_f64() * 1000.0;

            // ---- Step 6: Trie-walk NWA build ----
            let trie_build_started_at = Instant::now();
            let _build_profile = build_nwa_via_trie_walk(
                tokenizer_for_nwa_build,
                terminal_coloring,
                use_terminal_coloring && !reference_terminal_expansion,
                ignore_terminal,
                reference_terminal_expansion,
                &mut nwa,
                leaf_state,
                nwa_build_id_map.num_tsids(),
                &full_tree.root,
                &roots,
                &mut pm_computer,
                Some(terminals_for_nwa_build),
            );
            let trie_build_ms = trie_build_started_at.elapsed().as_secs_f64() * 1000.0;

            let always_allowed_started_at = Instant::now();
            let always_allowed = (!reference_terminal_expansion)
                .then(|| compute_always_allowed_follows(grammar))
                .unwrap_or_default();
            let always_allowed_ms = always_allowed_started_at.elapsed().as_secs_f64() * 1000.0;
            let nwa_states_after_build = nwa.states().len();

            let collapse_started_at = Instant::now();
            if !reference_terminal_expansion {
                collapse_always_allowed(&mut nwa, &always_allowed, grammar.num_terminals as usize);
            }
            let collapse_ms = collapse_started_at.elapsed().as_secs_f64() * 1000.0;
            let nwa_states_after_collapse = nwa.states().len();

            let disallowed_started_at = Instant::now();
            if !reference_terminal_expansion {
                apply_disallowed_follow_constraints(
                    &mut nwa,
                    disallowed_follows,
                    grammar.num_terminals as usize,
                    shared_disallowed_follow_dfa_cache,
                );
            }
            let disallowed_ms = disallowed_started_at.elapsed().as_secs_f64() * 1000.0;
            let nwa_states_after_disallowed = nwa.states().len();

            let prune_started_at = Instant::now();
            if !reference_terminal_expansion {
                prune_non_coreachable_states(&mut nwa);
            }
            let prune_ms = prune_started_at.elapsed().as_secs_f64() * 1000.0;
            let nwa_states_after_prune = nwa.states().len();

            let canonicalize_started_at = Instant::now();
            if !reference_terminal_expansion {
                canonicalize_acyclic_nwa(&mut nwa);
            }
            let canonicalize_ms = canonicalize_started_at.elapsed().as_secs_f64() * 1000.0;
            let nwa_states_after_canonicalize = nwa.states().len();

            let determinize_started_at = Instant::now();
            let det = determinize(&nwa).expect("L2+ terminal NWA determinization failed");
            let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;

            let minimize_started_at = Instant::now();
            let skip_minimize = reference_terminal_expansion
                || defer_minimization_to_local_merge
                || std::env::var("GLRMASK_SKIP_L2P_MINIMIZE")
                .map(|value| {
                    let trimmed = value.trim();
                    trimmed.is_empty() || trimmed == "1" || trimmed.eq_ignore_ascii_case("true")
                })
                .unwrap_or(false);
            let dwa = if skip_minimize { det } else { minimize_owned(det) };
            let minimize_ms = minimize_started_at.elapsed().as_secs_f64() * 1000.0;
            let dwa_stats_before_compact = dwa.stats();
            let dwa_stats_after_compact = dwa.stats();

            (
                dwa,
                vocab_tree_ms,
                possible_matches_ms,
                seed_ms,
                trie_build_ms,
                always_allowed_ms,
                collapse_ms,
                disallowed_ms,
                prune_ms,
                canonicalize_ms,
                determinize_ms,
                minimize_ms,
                internal_vocab_count,
                nwa_states_after_build,
                nwa_states_after_collapse,
                nwa_states_after_disallowed,
                nwa_states_after_prune,
                nwa_states_after_canonicalize,
                dwa_stats_before_compact,
                dwa_stats_after_compact,
                false,
            )
        }
    };
    if early_none {
        return None;
    }
    let mut dwa = dwa;
    let mut determinize_ms = determinize_ms;
    let mut minimize_ms = minimize_ms;
    let mut reference_postprocess_ms = 0.0;
    let mut reference_expand_profile = None;
    let mut composed_id_map = if use_simplified_tok {
        InternalIdMap {
            tokenizer_states: simplify_state_map
                .as_ref()
                .expect("simplify_state_map missing for simplified tokenizer")
                .compose(&simplified_id_map.tokenizer_states)
                .fill_unmapped_with_new_class(),
            vocab_tokens: simplified_id_map.vocab_tokens.clone(),
        }
    } else {
        simplified_id_map.clone()
    };

    if reference_terminal_expansion {
        let (views, expansion_profile) = terminal_interchangeability
            .expand_reference_dwa_views(&mut dwa, &mut composed_id_map.tokenizer_states);
        let reference_postprocess_started_at = Instant::now();
        let mut views = views.into_iter();
        let first_view = views
            .next()
            .expect("reference terminal interchangeability requires an identity view");
        let mut expanded_nwa = dwa_to_nwa_with_ignore_as_epsilon(&first_view, ignore_terminal);
        let mut expanded_body = expanded_nwa.body();
        for view in views {
            let branch = dwa_to_nwa_with_ignore_as_epsilon(&view, ignore_terminal);
            expanded_body = expanded_nwa.union_in_place(&branch, &expanded_body);
        }
        expanded_nwa.set_start_states(expanded_body.start_states);
        apply_disallowed_follow_constraints(
            &mut expanded_nwa,
            disallowed_follows,
            grammar.num_terminals as usize,
            shared_disallowed_follow_dfa_cache,
        );
        prune_non_coreachable_states(&mut expanded_nwa);
        canonicalize_acyclic_nwa(&mut expanded_nwa);
        let final_determinize_started_at = Instant::now();
        let final_det = determinize(&expanded_nwa)
            .expect("terminal interchangeability post-expansion determinization failed");
        determinize_ms += final_determinize_started_at.elapsed().as_secs_f64() * 1000.0;
        let final_minimize_started_at = Instant::now();
        dwa = minimize_owned(final_det);
        minimize_ms += final_minimize_started_at.elapsed().as_secs_f64() * 1000.0;
        reference_postprocess_ms = reference_postprocess_started_at.elapsed().as_secs_f64() * 1000.0;
        reference_expand_profile = Some(expansion_profile);
    }

    let dwa_stats_before_compact = dwa.stats();
    let dwa_stats_after_compact = dwa.stats();
    let postprocess_ms = always_allowed_ms
        + collapse_ms
        + disallowed_ms
        + prune_ms
        + canonicalize_ms
        + reference_postprocess_ms;
    if let Some(profile) = &reference_expand_profile {
        if l2p_timing_profile_enabled() {
            eprintln!(
                "[glrmask/profile][l2p_terminal_interchangeability] partition={} active_terminals={} equivalence_classes={} inactive_members={} row_classes={} swap_generators={} group_elements={} concrete_tsids_before={} concrete_tsids_after={} expanded_transition_copies={} initial_substitutions_applied={} initial_substitutions_missing={} continuation_initial_moved={} weight_remap_ms={:.3} expansion_ms={:.3}",
                partition_label,
                profile.active_terminals,
                profile.equivalence_classes,
                profile.inactive_members,
                profile.row_classes,
                profile.swap_generators,
                profile.group_elements,
                profile.concrete_tsids_before,
                profile.concrete_tsids_after,
                profile.expanded_transition_copies,
                profile.initial_substitutions_applied,
                profile.initial_substitutions_missing,
                profile.continuation_initial_moved,
                profile.weight_remap_ms,
                profile.expansion_ms,
            );
        }
    }

    let max_length_reduction_pct = if equiv_profile.initial_states_considered == 0 {
        0.0
    } else {
        100.0
            * (1.0
                - equiv_profile.max_length_reps as f64
                    / equiv_profile.initial_states_considered as f64)
    };
    let exact_reduction_pct = if equiv_profile.initial_states_considered == 0 {
        0.0
    } else {
        100.0
            * (1.0
                - equiv_profile.exact_reps as f64 / equiv_profile.initial_states_considered as f64)
    };

    if l2p_timing_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l2p] partition={} vocab_tokens={} active_terminals={} original_states={} tsids={} internal_vocab_entries={} initial_states_considered={} max_length_skipped={} max_token_len={} token_len_gt_4={} token_len_gt_8={} token_len_gt_16={} token_len_gt_32={} token_len_gt_64={} prepare_inputs_ms={:.3} byte_class_setup_ms={:.3} token_dedup_ms={:.3} max_length_state_equiv_ms={:.3} vocab_equiv_ms={:.3} exact_state_equiv_ms={:.3} id_map_finalize_ms={:.3} max_length_reps={} exact_reps={} exact_rep_confirmation_used={} fast_sound_id_map_used={} max_length_reduction_pct={:.2} exact_reduction_pct={:.2} simplify_ms={:.3} simplify_cache_hit={} simplified_states={} id_map_ms={:.3} tsid_fallback_ms={:.3} vocab_tree_ms={:.3} possible_matches_ms={:.3} seed_ms={:.3} terminal_nwa_build_ms={:.3} nwa_states={}->{}->{}->{}->{} always_allowed_ms={:.3} collapse_ms={:.3} disallowed_ms={:.3} prune_ms={:.3} canonicalize_ms={:.3} postprocess_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} minimize_states={} dwa_states={} dwa_transitions={} dwa_transition_pairs={} dwa_interned_ranges_before_compact={} dwa_interned_ranges_after_compact={} total_ms={:.3}",
            partition_label,
            vocab.entries.len(),
            num_active_terminals,
            num_original_states,
            simplified_id_map.num_tsids(),
            internal_vocab_count,
            equiv_profile.initial_states_considered,
            equiv_profile.max_length_skipped,
            equiv_profile.max_token_len,
            equiv_profile.token_len_gt_4,
            equiv_profile.token_len_gt_8,
            equiv_profile.token_len_gt_16,
            equiv_profile.token_len_gt_32,
            equiv_profile.token_len_gt_64,
            equiv_profile.prepare_inputs_ms,
            equiv_profile.byte_class_setup_ms,
            equiv_profile.token_dedup_ms,
            equiv_profile.max_length_state_equiv_ms,
            equiv_profile.vocab_equiv_ms,
            equiv_profile.exact_state_equiv_ms,
            equiv_profile.id_map_finalize_ms,
            equiv_profile.max_length_reps,
            equiv_profile.exact_reps,
            equiv_profile.exact_rep_confirmation_used,
            fast_sound_id_map_used,
            max_length_reduction_pct,
            exact_reduction_pct,
            simplify_ms,
            simplify_cache_hit,
            tokenizer_for_build.num_states(),
            id_map_ms,
            0.0,
            vocab_tree_ms,
            possible_matches_ms,
            seed_ms,
            trie_build_ms,
            nwa_states_after_build,
            nwa_states_after_collapse,
            nwa_states_after_disallowed,
            nwa_states_after_prune,
            nwa_states_after_canonicalize,
            always_allowed_ms,
            collapse_ms,
            disallowed_ms,
            prune_ms,
            canonicalize_ms,
            postprocess_ms,
            determinize_ms,
            minimize_ms,
            dwa_stats_before_compact.states,
            dwa_stats_before_compact.states,
            dwa_stats_before_compact.transitions,
            dwa_stats_before_compact.transition_pairs,
            dwa_stats_before_compact.interned_ranges,
            dwa_stats_after_compact.interned_ranges,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Some(LocalIdMapTerminalDwa {
        id_map: composed_id_map,
        dwa,
        profile: TerminalDwaPhaseProfile {
            id_map_ms: simplify_ms + id_map_ms,
            terminal_dwa_ms: vocab_tree_ms
                + possible_matches_ms
                + seed_ms
                + trie_build_ms
                + postprocess_ms
                + determinize_ms
                + minimize_ms,
            compact_ms: 0.0,
            ..TerminalDwaPhaseProfile::default()
        },
    })
}
