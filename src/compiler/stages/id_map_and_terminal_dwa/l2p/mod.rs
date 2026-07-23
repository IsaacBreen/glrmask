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
mod terminal_dwa_equivalence;
mod terminal_interchangeability;

pub(crate) use terminal_interchangeability::warm_ti_pool;
pub(crate) use terminal_interchangeability::with_ti_pool;
pub(crate) use terminal_interchangeability::SharedTiTokenizerOutputCache;

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::determinize::{determinize, determinize_depth2};
use crate::automata::weighted::equivalence::find_difference;
use crate::automata::weighted::minimize::{
    PointwiseClassOrder, minimize_owned, minimize_owned_with_pointwise_class_order,
};
use crate::automata::weighted::nwa::NWA;
use crate::automata::weighted_u32::dwa::DWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::possible_matches::PossibleMatchesComputer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::compiler::stages::mapped_artifact::MappedArtifact;
use crate::compiler::stages::id_map_and_terminal_dwa::types::LocalIdMapTerminalDwa;
use crate::ds::bitset::BitSet;
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;
use crate::Vocab;

use super::types::{
    compile_profile_enabled, TerminalColoring, TerminalDwaBuildProfile, TerminalDwaPhaseProfile,
};
use nwa_builder::{build_nwa_via_trie_walk, internal_vocab_entries, seed_root_nodes};
use terminal_interchangeability::{
    active_terminals_for_partition, binary_transport_modes_from_witnesses,
    canonicalize_transport_mode_states, coalesced_disallowed_follows,
    discover_one_round_with_transport_witnesses_in_context, fold_one_round_partition,
    expand_representative_dwa_after_minimization, partition_has_merges,
    restrict_weights_to_forward_domains_in_place, restore_raw_follow_constraints_after_expansion,
    singleton_partition, transport_coordinate_quotient, visible_output_raw_labels,
    TiDiscoveryContext,
};
use postprocess::{
    apply_disallowed_follow_constraints, canonicalize_acyclic_nwa, collapse_always_allowed,
    max_structural_label_depth_to_final, prune_non_coreachable_states,
};

fn l2p_timing_profile_enabled() -> bool {
    compile_profile_enabled() || std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some()
}

thread_local! {
    static TERMINAL_INTERCHANGEABILITY_SUPPRESS_DEPTH: Cell<u32> = const { Cell::new(0) };
    static TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE_SUPPRESS_DEPTH: Cell<u32> = const { Cell::new(0) };
    static GLOBAL_TOKEN_POSITION_SUPPRESS_DEPTH: Cell<u32> = const { Cell::new(0) };
    static FIRST_BYTE_VOCAB_FACTOR_SUPPRESS_DEPTH: Cell<u32> = const { Cell::new(0) };
}

struct SuppressTerminalInterchangeability;
struct SuppressTerminalInterchangeabilityStrictReference;
struct SuppressGlobalTokenPosition;
struct SuppressFirstByteVocabFactor;

impl SuppressTerminalInterchangeability {
    fn new() -> Self {
        TERMINAL_INTERCHANGEABILITY_SUPPRESS_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self
    }
}

impl Drop for SuppressTerminalInterchangeability {
    fn drop(&mut self) {
        TERMINAL_INTERCHANGEABILITY_SUPPRESS_DEPTH.with(|depth| {
            depth.set(
                depth
                    .get()
                    .checked_sub(1)
                    .expect("unbalanced terminal interchangeability suppression"),
            );
        });
    }
}

impl SuppressTerminalInterchangeabilityStrictReference {
    fn new() -> Self {
        TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE_SUPPRESS_DEPTH
            .with(|depth| depth.set(depth.get() + 1));
        Self
    }
}

impl Drop for SuppressTerminalInterchangeabilityStrictReference {
    fn drop(&mut self) {
        TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE_SUPPRESS_DEPTH.with(|depth| {
            depth.set(
                depth
                    .get()
                    .checked_sub(1)
                    .expect("unbalanced terminal interchangeability strict-reference suppression"),
            );
        });
    }
}

impl SuppressGlobalTokenPosition {
    fn new() -> Self {
        GLOBAL_TOKEN_POSITION_SUPPRESS_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self
    }
}

impl Drop for SuppressGlobalTokenPosition {
    fn drop(&mut self) {
        GLOBAL_TOKEN_POSITION_SUPPRESS_DEPTH.with(|depth| {
            depth.set(
                depth
                    .get()
                    .checked_sub(1)
                    .expect("unbalanced global token-position suppression"),
            );
        });
    }
}

impl SuppressFirstByteVocabFactor {
    fn new() -> Self {
        FIRST_BYTE_VOCAB_FACTOR_SUPPRESS_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self
    }
}

impl Drop for SuppressFirstByteVocabFactor {
    fn drop(&mut self) {
        FIRST_BYTE_VOCAB_FACTOR_SUPPRESS_DEPTH.with(|depth| {
            depth.set(
                depth
                    .get()
                    .checked_sub(1)
                    .expect("unbalanced first-byte vocabulary factor suppression"),
            );
        });
    }
}

fn l2p_env_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn l2p_partition_selector_enabled(name: &str, partition_label: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() || trimmed == "1" || trimmed.eq_ignore_ascii_case("true") {
                return true;
            }
            if trimmed == "0" || trimmed.eq_ignore_ascii_case("false") {
                return false;
            }
            trimmed
                .split(',')
                .map(str::trim)
                .any(|label| label == partition_label)
        })
        .unwrap_or(false)
}

pub(crate) fn l2p_first_byte_vocab_factor_enabled_for_partition(
    partition_label: &str,
) -> bool {
    FIRST_BYTE_VOCAB_FACTOR_SUPPRESS_DEPTH.with(|depth| depth.get() == 0)
        && l2p_partition_selector_enabled(
            "GLRMASK_ENABLE_L2P_FIRST_BYTE_VOCAB_FACTOR",
            partition_label,
        )
}

pub(crate) fn l2p_first_byte_vocab_factor_strict_reference_enabled_for_partition(
    partition_label: &str,
) -> bool {
    l2p_first_byte_vocab_factor_enabled_for_partition(partition_label)
        && l2p_partition_selector_enabled(
            "GLRMASK_L2P_FIRST_BYTE_VOCAB_FACTOR_STRICT_REFERENCE",
            partition_label,
        )
}

/// Production terminal interchangeability is enabled by default. Recursive
/// strict-reference rebuilds suppress it with the thread-local depth guard;
/// `GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY=1` is the explicit
/// diagnostic escape hatch.
fn l2p_terminal_interchangeability_enabled() -> bool {
    TERMINAL_INTERCHANGEABILITY_SUPPRESS_DEPTH.with(|depth| depth.get() == 0)
        && !l2p_env_enabled("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY")
}

fn l2p_terminal_interchangeability_enabled_for_partition(_partition_label: &str) -> bool {
    l2p_terminal_interchangeability_enabled()
}

/// Rebuild the TI-off local artifact and symbolically compare it with TI-on.
/// This is deliberately slow and is intended for tests and explicit validation,
/// not ordinary TI compilation.
fn l2p_terminal_interchangeability_strict_reference_enabled() -> bool {
    TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE_SUPPRESS_DEPTH.with(|depth| depth.get() == 0)
        && l2p_env_enabled("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE")
}

/// The structural p8 factorization remains a production optimization. Strict
/// global-token-position validation suppresses C itself while retaining all
/// other local optimizations on both sides of the comparison.
pub(crate) fn p8_first_byte_factorization_allowed() -> bool {
    // Forced-all-L2P is a diagnostic route that does not preserve the normal
    // classifier invariants of the structural-boundary partition.
    !l2p_env_enabled("GLRMASK_FORCE_ALL_L2P")
}

fn l2p_global_token_position_enabled() -> bool {
    GLOBAL_TOKEN_POSITION_SUPPRESS_DEPTH.with(|depth| depth.get() == 0)
        && !l2p_env_enabled("GLRMASK_DISABLE_L2P_GLOBAL_TOKEN_POSITION")
}

fn l2p_global_token_position_strict_reference_enabled() -> bool {
    l2p_env_enabled("GLRMASK_L2P_GLOBAL_TOKEN_POSITION_STRICT_REFERENCE")
}

fn l2p_strict_partition_matches(partition_label: &str) -> bool {
    std::env::var("GLRMASK_L2P_STRICT_REFERENCE_PARTITION")
        .map(|requested| requested == partition_label)
        .unwrap_or(true)
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

/// Collapse byte-identical states in a TI-expanded DWA before minimization.
///
/// TI mode expansion materializes many states that are exact duplicates: same
/// `final_weight` and same `(label -> dst, weight)` transitions, where weight
/// equality is exact because the lifted weights and their token sets are
/// interned (Arc-ptr equality == value equality). Merging such states is a
/// sound bisimulation quotient — it preserves the language and every weight —
/// so the subsequent pointwise minimization reaches the identical canonical
/// automaton (byte-identical digest) from a smaller input. This shrinks the
/// input to `push_weights` and the pointwise coloring, which dominate the
/// post-DWA minimize cost, without attempting the deeper pointwise-value
/// equivalences that only that machinery can find.
///
/// The partition is refined to a fixpoint over `(final_weight ptr,
/// [(label, class(dst), weight ptr)])`; the destination is compared up to its
/// class so that duplicate states whose only difference is pointing at
/// (mutually duplicate) destinations still collapse.
fn quotient_exact_duplicate_states(dwa: DWA) -> DWA {
    use crate::automata::weighted_u32::dwa::DWAState;
    use std::collections::BTreeMap as StdBTreeMap;
    use std::sync::Arc;

    let states = dwa.states();
    let n = states.len();
    if n <= 1 {
        return dwa;
    }
    let wptr = |w: &Weight| Arc::as_ptr(&w.0) as u64;

    // Initial partition by local output signature (ignores destinations).
    let mut class: Vec<u32> = vec![0; n];
    let mut init_map: StdBTreeMap<Vec<u64>, u32> = StdBTreeMap::new();
    for (i, state) in states.iter().enumerate() {
        let mut sig: Vec<u64> = Vec::with_capacity(1 + state.transitions.len() * 2);
        sig.push(match &state.final_weight {
            Some(fw) => wptr(fw),
            None => 0,
        });
        for (label, (_dst, w)) in state.transitions.iter() {
            sig.push(*label as u64);
            sig.push(wptr(w));
        }
        let next = init_map.len() as u32;
        class[i] = *init_map.entry(sig).or_insert(next);
    }
    let mut num_classes = init_map.len();

    // Refine to fixpoint, now distinguishing destinations by their class.
    loop {
        let mut refine_map: StdBTreeMap<Vec<u64>, u32> = StdBTreeMap::new();
        let mut new_class: Vec<u32> = vec![0; n];
        for (i, state) in states.iter().enumerate() {
            let mut sig: Vec<u64> = Vec::with_capacity(1 + state.transitions.len() * 3);
            sig.push(class[i] as u64);
            for (label, (dst, w)) in state.transitions.iter() {
                sig.push(*label as u64);
                sig.push(wptr(w));
                sig.push(class[*dst as usize] as u64);
            }
            let next = refine_map.len() as u32;
            new_class[i] = *refine_map.entry(sig).or_insert(next);
        }
        let new_num = refine_map.len();
        class = new_class;
        if new_num == num_classes {
            break;
        }
        num_classes = new_num;
    }

    if num_classes == n {
        return dwa;
    }

    // Representative (first-seen) state per class; new state id == class id.
    let mut rep_of_class: Vec<Option<usize>> = vec![None; num_classes];
    for (i, &c) in class.iter().enumerate() {
        rep_of_class[c as usize].get_or_insert(i);
    }

    let mut new_states: Vec<DWAState> = Vec::with_capacity(num_classes);
    for c in 0..num_classes {
        let rep = rep_of_class[c].expect("every class has a representative");
        let old = &states[rep];
        let transitions = old
            .transitions
            .iter()
            .map(|(label, (dst, w))| (*label, (class[*dst as usize], w.clone())))
            .collect();
        new_states.push(DWAState {
            transitions,
            final_weight: old.final_weight.clone(),
        });
    }
    let new_start = class[dwa.start_state() as usize];
    DWA::from_parts(new_states, new_start)
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
/// 7. Determinize → minimize → compact.
///
/// `disallowed_follows` is threaded explicitly for id_map building.
///
/// Returns `None` if the vocab is empty.
pub(crate) fn build_l2p_id_map_and_terminal_dwa(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    always_allowed_follows: &[Vec<TerminalID>],
    active_terminals: &[bool],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    token_path_disallowed_follows: Option<&BTreeMap<u32, BitSet>>,
    normalized_token_path_disallowed_follows: Option<&[BitSet]>,
    shared_vocab_dfa_cache: Option<&equivalence_analysis::vocab::fast::SharedVocabDfaCache>,
    shared_original_vocab_dfa_cache: Option<&equivalence_analysis::vocab::fast::SharedVocabDfaCache>,
    shared_original_vocab_analysis_dfa_cache: Option<&equivalence_analysis::vocab::fast::SharedVocabAnalysisDfaCache>,
    shared_transition_cache: Option<&OnceLock<equivalence_analysis::compat::FlatTransitionCache>>,
    shared_ti_output_cache: Option<&SharedTiTokenizerOutputCache>,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    prebuilt_token_trie: Option<
        &equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie,
    >,
    initial_state_map: Option<&ManyToOneIdMap>,
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

    let token_position_partition_started_at = l2p_timing_profile_enabled().then(Instant::now);
    // Global token-position partition C. C combines partial partitions over
    // states active at explicit vocabulary-byte positions with pairwise
    // identity on the collapsed suffix tail. Inactive states are wildcards. A
    // state maps only to a maximal partial signature that extends all of that
    // state's known positional classes, so the selected raw representative is
    // valid everywhere the member can occur under the position-wise frontier
    // propagation for this vocabulary partition. TI consumes exactly this
    // directional member-to-representative substitution property; members of a
    // C class need not be pairwise interchangeable as raw DFA states.
    // The pre-TI quotient and representative-core token-position partition are
    // two wrappers over the same C map. Build the positional analysis once.
    let (global_state_quotient, token_position_partition) =
        if l2p_global_token_position_enabled() && matches!(partition_label, "p7" | "p8") {
            match equivalence_analysis::state_equivalence::global_token_position::
                compute_global_token_position_state_views(
                    tokenizer,
                    vocab,
                    flat_trans.map(AsRef::as_ref),
                )
            {
                Some((quotient, partition, _profile)) => (Some(quotient), Some(partition)),
                None => (None, None),
            }
        } else {
            (None, None)
        };
    let token_position_partition_ms = token_position_partition_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    if l2p_timing_profile_enabled() {
        eprintln!(
            "[glrmask/profile][global_token_position_partition] partition={} raw_states={} representatives={} enabled={} build_ms={:.3}",
            partition_label,
            num_original_states,
            global_state_quotient.as_ref().map_or(0, |quotient| {
                quotient.as_many_to_one().representative_original_ids.len()
            }),
            global_state_quotient.is_some(),
            token_position_partition_ms,
        );
    }

    // Repeatedly discover and immediately fold each transient exact partition.
    // Only the final flat original-member partition survives this loop.
    let ti_profile_timing = l2p_timing_profile_enabled();
    let ti_discovery_started_at = ti_profile_timing.then(Instant::now);
    let (
        terminal_partition,
        ti_transport_witness_rounds,
        ti_round_count,
        ti_discovery_ms,
        ti_additional_merged_members,
        ti_raw_observations,
        ti_restricted_observation_seed,
        ti_restricted_observation_seed_ms,
    ) =
        if l2p_terminal_interchangeability_enabled_for_partition(partition_label) {
            let mut active = active_terminals.to_vec();
            for &terminal in grammar.residual_isolation_classes.keys() {
                if let Some(slot) = active.get_mut(terminal as usize) {
                    *slot = false;
                }
            }
            let mut classes = singleton_partition(&active);
            // Always use the byte-level exact discovery oracle. When the global
            // token-position quotient C is available (p7/p8), build discovery
            // evidence over C representatives; otherwise over raw states.
            let discovery_context = match global_state_quotient.as_ref() {
                Some(quotient) => {
                    TiDiscoveryContext::try_new_with_global_state_quotient_and_output_cache(
                        tokenizer,
                        &relevant_bytes,
                        quotient,
                        shared_ti_output_cache,
                    )
                }
                None => TiDiscoveryContext::try_new_with_output_cache(
                    tokenizer,
                    &relevant_bytes,
                    shared_ti_output_cache,
                ),
            };
            match discovery_context {
                Ok(discovery_context) => {
                    let mut transport_witness_rounds = Vec::new();
                    let mut round_count = 0usize;
                    let mut first_round_class_count = None;
                    loop {
                        let round = discover_one_round_with_transport_witnesses_in_context(
                            tokenizer,
                            &active,
                            &discovery_context,
                            ignore_terminal,
                        );
                        let next_active =
                            active_terminals_for_partition(&round.partition, active.len());
                        let next_classes = fold_one_round_partition(&classes, &round.partition);
                        round_count += 1;
                        first_round_class_count.get_or_insert(next_classes.len());
                        classes = next_classes;
                        transport_witness_rounds.push(round);
                        if next_active == active {
                            break;
                        }
                        active = next_active;
                    }
                    let additional_merged_members = first_round_class_count
                        .unwrap_or(classes.len())
                        .saturating_sub(classes.len());
                    let discovery_ms = ti_discovery_started_at
                        .as_ref()
                        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);
                    let restricted_observation_seed_started_at =
                        ti_profile_timing.then(Instant::now);
                    let restricted_observation_seed = discovery_context
                        .reusable_nfa_restricted_observation_state_map(
                            tokenizer,
                            initial_state_map,
                        );
                    let restricted_observation_seed_ms = restricted_observation_seed_started_at
                        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);
                    let raw_observations = discovery_context
                        .final_raw_observation_ids(tokenizer.num_states() as usize);
                    (
                        Some(classes),
                        Some(transport_witness_rounds),
                        round_count,
                        discovery_ms,
                        additional_merged_members,
                        raw_observations,
                        restricted_observation_seed,
                        restricted_observation_seed_ms,
                    )
                }
                Err(_) => {
                    let discovery_ms = ti_discovery_started_at
                        .as_ref()
                        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);
                    (None, None, 0, discovery_ms, 0, None, None, 0.0)
                }
            }
        } else {
            (None, None, 0, 0.0, 0, None, None, 0.0)
        };
    let reference_terminal_expansion = terminal_partition
        .as_ref()
        .is_some_and(|partition| partition_has_merges(partition));
    // A successful merge activates production TI. The recursive TI-off rebuild
    // and symbolic comparison are a separate, explicit validation mode.
    let strict_reference = reference_terminal_expansion
        && l2p_terminal_interchangeability_strict_reference_enabled()
        && l2p_strict_partition_matches(partition_label);
    let global_token_position_strict_reference = token_position_partition.is_some()
        && l2p_global_token_position_strict_reference_enabled()
        && l2p_strict_partition_matches(partition_label);
    let first_byte_vocab_factor_strict_reference =
        l2p_first_byte_vocab_factor_strict_reference_enabled_for_partition(partition_label);
    let analysis_active_terminals = terminal_partition
        .as_ref()
        .map(|partition| active_terminals_for_partition(partition, active_terminals.len()))
        .unwrap_or_else(|| active_terminals.to_vec());
    // TI narrows equivalence observations to final representatives, but the
    // representative core must still emit terminals outside the TI-active
    // partition. Only true nonrepresentative class members are hidden.
    let representative_core_output_labels = reference_terminal_expansion.then(|| {
        visible_output_raw_labels(
            terminal_partition
                .as_ref()
                .expect("active TI transport must retain its partition"),
            active_terminals,
        )
    });
    // Equivalence and terminal-DWA construction must observe the same terminal
    // alphabet. `None` means "all terminals" to the NWA builder; using it on
    // the ordinary non-TI path after analyzing only `analysis_active_terminals`
    // can reintroduce inactive terminals from quotient representatives and make
    // the weighted language unsound. TI has a broader explicit output mask
    // because only true nonrepresentative class members are hidden there.
    let representative_core_active_terminals = representative_core_output_labels
        .as_deref()
        .unwrap_or(&analysis_active_terminals);
    let coalesced_disallowed_follows_started_at = ti_profile_timing.then(Instant::now);
    let coalesced_disallowed_follows = reference_terminal_expansion.then(|| {
        coalesced_disallowed_follows(
            terminal_partition
                .as_ref()
                .expect("active TI partition must be present"),
            disallowed_follows,
            grammar.num_terminals as usize,
        )
    });
    let ti_coalesced_disallowed_follows_ms = coalesced_disallowed_follows_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    if std::env::var("GLRMASK_DUMP_TI_CLASSES")
        .ok()
        .is_some_and(|requested| requested == "1" || requested == partition_label)
    {
        if let Some(partition) = terminal_partition.as_ref() {
            for (&representative, members) in partition {
                let member_exprs = members
                    .iter()
                    .map(|&member| {
                        format!(
                            "{}:{:?}",
                            member,
                            tokenizer.terminal_expr(member),
                        )
                    })
                    .collect::<Vec<_>>();
                eprintln!(
                    "[glrmask/dump][ti_class] partition={} representative={} members=[{}]",
                    partition_label,
                    representative,
                    member_exprs.join(", "),
                );
            }
        }
    }
    if std::env::var("GLRMASK_PROFILE_L2P_EXIT_AFTER_TI_DISCOVERY")
        .ok()
        .as_deref()
        == Some(partition_label)
    {
        let class_count = terminal_partition.as_ref().map_or(0, |partition| partition.len());
        eprintln!(
            "[glrmask/profile][terminal_interchangeability_plan] partition={} ti_active={} strict_reference={} rounds={} classes={} additional_merged_members={} discovery_ms={:.3} coalesced_disallowed_follows_ms={:.3} early_exit=after_ti_discovery",
            partition_label,
            reference_terminal_expansion,
            strict_reference,
            ti_round_count,
            class_count,
            ti_additional_merged_members,
            ti_discovery_ms,
            ti_coalesced_disallowed_follows_ms,
        );
        std::process::exit(0);
    }
    let equivalence_uses_pre_normalized_follows = coalesced_disallowed_follows.is_none()
        && token_path_disallowed_follows.is_some()
        && normalized_token_path_disallowed_follows.is_some();
    let equivalence_disallowed_follows = coalesced_disallowed_follows
        .as_ref()
        .or(token_path_disallowed_follows)
        .unwrap_or(disallowed_follows);
    // The representative core begins after TI discovery/coalescing. It includes
    // ordinary representative-only equivalence through representative DWA
    // compaction, but deliberately excludes replay and post-DWA expansion.
    // Time the ordinary representative-core section for both TI-on and TI-off
    // builds. This makes the profile compare the same work directly while
    // still excluding discovery, replay, and the strict baseline comparator.
    let ti_representative_core_started_at = l2p_timing_profile_enabled().then(Instant::now);
    let num_analysis_active_terminals = analysis_active_terminals
        .iter()
        .filter(|&&active| active)
        .count();
    let tokenizer_for_build = tokenizer;
    let token_position_partition_for_analysis = matches!(partition_label, "p1" | "p7" | "p8")
        .then_some(token_position_partition.as_ref())
        .flatten();
    let equivalence_initial_state_map = ti_restricted_observation_seed
        .as_ref()
        .or(initial_state_map);

    // ---- Step 1: Equivalence analysis (raw tokenizer state IDs) ----
    let id_map_started_at = Instant::now();
    // The raw shared base is demand-driven. The representative-only quotient
    // can prove its exact byte relation without materializing raw layouts; a
    // legacy analysis view still initializes the same cache on first use.
    let equivalence_vocab_dfa_cache = shared_original_vocab_dfa_cache.or(shared_vocab_dfa_cache);
    let shared_base_setup_ms = 0.0;
    let shared_analysis_dfa_cache = shared_original_vocab_analysis_dfa_cache;
    // Do not remove this note. Do not remove the instruction saying not to
    // remove this note. L2P state/vocab equivalence must always run fully.
    // Max-length may be skipped in controlled cases, but the full exact
    // state/vocab equivalence pass must not be bypassed. Do not reintroduce
    // fast-sound, identity, lex-dedup, or similar shortcut id-map paths.
    let fast_sound_id_map_used = false;
    // Keep raw lexer-state coordinates for the final scanner. Ordinarily the
    // exact equivalence analysis can project observations to this L2P family's
    // active terminals. Importer-level complex-pattern decomposition creates
    // parser-visible residual terminals whose byte languages can overlap a
    // terminal in another construction family. In that case family-local
    // projection can erase a vocabulary distinction that reconciliation cannot
    // reconstruct, so observe every terminal residual conservatively.
    let equivalence_active_groups = (!grammar.requires_global_terminal_observation)
        .then_some(analysis_active_terminals.as_slice());
    let (simplified_id_map, equiv_profile) =
        equivalence_analysis::combined::analyze_equivalences_with_group_filter(
            partition_label,
            tokenizer_for_build,
            vocab,
            equivalence_disallowed_follows,
            ignore_terminal,
            equivalence_uses_pre_normalized_follows,
            equivalence_uses_pre_normalized_follows
                .then_some(normalized_token_path_disallowed_follows)
                .flatten(),
            equivalence_active_groups,
            equivalence_vocab_dfa_cache,
            shared_analysis_dfa_cache,
            shared_base_setup_ms,
            flat_trans,
            shared_transition_cache,
            equivalence_initial_state_map,
            token_position_partition_for_analysis,
            std::env::var_os("GLRMASK_TI_DISABLE_RAW_OBSERVATION_REUSE")
                .is_none()
                .then(|| {
                    ti_raw_observations.as_ref().map(|(ids, representatives)| {
                        (ids.as_slice(), representatives.as_slice())
                    })
                })
                .flatten(),
            prebuilt_token_trie,
        );

    // Replay and transport-coordinate refinement are intentionally deferred
    // until after the representative-only DWA is minimized and compacted.
    // Keeping this ordinary quotient here is what makes the core small.
    let mut ti_transport_modes_ms = 0.0;
    let mut ti_canonicalize_transport_modes_ms = 0.0;
    let mut ti_transport_coordinate_quotient_ms = 0.0;

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
        nwa_build_profile,
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
        early_none,
    ) = {
        // ---- Step 2-3: Internal vocab + prefix tree ----
        let vocab_tree_started_at = Instant::now();
        let internal_vocab = internal_vocab_entries(vocab, &simplified_id_map);
        let internal_vocab_count = internal_vocab.len();
        if internal_vocab.is_empty() {
            // Signal early-None via a sentinel. Build a dummy DWA;
            // outer code will observe `early_none=true` and return.
            (
                crate::automata::weighted_u32::dwa::DWA::new(0, 0),
                0.0, // vocab_tree_ms
                0.0, // possible_matches_ms
                0.0, // seed_ms
                0.0, // trie_build_ms
                TerminalDwaBuildProfile::default(),
                0.0, // always_allowed_ms
                0.0, // collapse_ms
                0.0, // disallowed_ms
                0.0, // prune_ms
                0.0, // canonicalize_ms
                0.0, // determinize_ms
                0.0, // minimize_ms
                0usize, // internal_vocab_count
                0usize, // nwa_states_after_build
                0usize, // nwa_states_after_collapse
                0usize, // nwa_states_after_disallowed
                0usize, // nwa_states_after_prune
                0usize, // nwa_states_after_canonicalize
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
            let mut pm_computer = PossibleMatchesComputer::new(tokenizer_for_build);
            let possible_matches_ms = 0.0;

            // ---- Step 5: Create NWA and seed root nodes ----
            let seed_started_at = Instant::now();
            let mut nwa = NWA::new(
                simplified_id_map.num_tsids(),
                simplified_id_map.max_internal_token_id(),
            );
            let leaf_state = nwa.add_state();
            nwa.set_final_weight(leaf_state, Weight::all());
            let start_state = nwa.add_state();
            nwa.start_states_mut().push(start_state);

            let seed_ms;

            // ---- Step 6: Trie-walk NWA build ----
            let trie_build_started_at = Instant::now();
            let roots = seed_root_nodes(tokenizer, &mut nwa, start_state, &simplified_id_map);
            seed_ms = seed_started_at.elapsed().as_secs_f64() * 1000.0;
            let build_profile = build_nwa_via_trie_walk(
                tokenizer_for_build,
                terminal_coloring,
                use_terminal_coloring,
                ignore_terminal,
                &mut nwa,
                leaf_state,
                simplified_id_map.num_tsids(),
                &full_tree.root,
                &roots,
                &mut pm_computer,
                flat_trans.map(AsRef::as_ref),
                representative_core_active_terminals,
            );
            #[cfg(debug_assertions)]
            if !use_terminal_coloring {
                for state in nwa.states() {
                    for &label in state.transitions.keys() {
                        if label >= 0 {
                            debug_assert!(
                                representative_core_active_terminals
                                    .get(label as usize)
                                    .copied()
                                    .unwrap_or(false),
                                "L2P representative core emitted terminal {label} outside its active output mask",
                            );
                        }
                    }
                }
            }
            let trie_build_ms = trie_build_started_at.elapsed().as_secs_f64() * 1000.0;

            let always_allowed_ms = 0.0;
            let nwa_states_after_build = nwa.states().len();

            let collapse_started_at = Instant::now();
            collapse_always_allowed(
                &mut nwa,
                always_allowed_follows,
                grammar.num_terminals as usize,
            );
            let collapse_ms = collapse_started_at.elapsed().as_secs_f64() * 1000.0;
            let nwa_states_after_collapse = nwa.states().len();

            let disallowed_started_at = Instant::now();
            apply_disallowed_follow_constraints(
                &mut nwa,
                equivalence_disallowed_follows,
                grammar.num_terminals as usize,
                ignore_terminal,
            );
            let disallowed_ms = disallowed_started_at.elapsed().as_secs_f64() * 1000.0;
            let nwa_states_after_disallowed = nwa.states().len();

            let prune_started_at = Instant::now();
            prune_non_coreachable_states(&mut nwa);
            let prune_ms = prune_started_at.elapsed().as_secs_f64() * 1000.0;
            let nwa_states_after_prune = nwa.states().len();

            let canonicalize_started_at = Instant::now();
            canonicalize_acyclic_nwa(&mut nwa);
            let canonicalize_ms = canonicalize_started_at.elapsed().as_secs_f64() * 1000.0;
            let nwa_states_after_canonicalize = nwa.states().len();

            let structural_depth = max_structural_label_depth_to_final(&nwa);
            let determinize_started_at = Instant::now();
            let use_depth2 = structural_depth.is_some_and(|depth| depth <= 2);
            let det = if use_depth2 {
                let depth2 = determinize_depth2(&nwa)
                    .expect("depth-2 terminal NWA determinization failed");
                if std::env::var_os("GLRMASK_ASSERT_L2P_DEPTH2_DETERMINIZE").is_some() {
                    let reference =
                        determinize(&nwa).expect("L2+ terminal NWA determinization failed");
                    assert_eq!(
                        find_difference(&depth2, &reference)
                            .expect("depth-2 terminal DWA equivalence check failed"),
                        None,
                        "depth-2 terminal DWA differs from generic determinization",
                    );
                }
                depth2
            } else {
                determinize(&nwa).expect("L2+ terminal NWA determinization failed")
            };
            let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;

            let minimize_started_at = Instant::now();
            let skip_minimize = std::env::var("GLRMASK_SKIP_L2P_MINIMIZE")
                .map(|value| {
                    let trimmed = value.trim();
                    trimmed.is_empty() || trimmed == "1" || trimmed.eq_ignore_ascii_case("true")
                })
                .unwrap_or(false);
            let dwa = if skip_minimize { det } else { minimize_owned(det) };
            let minimize_ms = minimize_started_at.elapsed().as_secs_f64() * 1000.0;
            let dwa_stats_before_compact = dwa.stats();

            (
                dwa,
                vocab_tree_ms,
                possible_matches_ms,
                seed_ms,
                trie_build_ms,
                build_profile,
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
                false,
            )
        }
    };
    if early_none {
        return None;
    }
    let composed_id_map = simplified_id_map.clone();
    let postprocess_ms =
        always_allowed_ms + collapse_ms + disallowed_ms + prune_ms + canonicalize_ms;
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

    let profiling = compile_profile_enabled();
    let mut mapped_dwa = MappedArtifact::new(dwa, composed_id_map);
    let core_compact_started_at = Instant::now();
    if profiling {
        mapped_dwa.compact_dimensions_fast_with_stats();
    } else {
        mapped_dwa.compact_dimensions_fast();
    }
    let core_compact_ms = core_compact_started_at.elapsed().as_secs_f64() * 1000.0;
    let core_dwa_stats_after_compact = mapped_dwa.artifact().stats();
    let (core_dwa, core_id_map) = mapped_dwa.into_parts();
    let core_tsid_count = core_id_map.num_tsids();
    let ti_representative_core_total_ms = ti_representative_core_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    // Expand only the already minimized representative DWA. Replay witnesses
    // are temporary and are deliberately built here rather than before core
    // NWA construction. The final retained TI result remains the flat
    // terminal partition.
    let mut ti_post_dwa_expansion_ms = 0.0;
    let mut ti_raw_follow_restore_ms = 0.0;
    let mut ti_forward_domain_normalize_ms = 0.0;
    let mut ti_post_dwa_minimize_ms = 0.0;
    let mut ti_post_dwa_compact_ms = 0.0;
    let ti_post_dwa_started_at = reference_terminal_expansion.then(Instant::now);
    let (dwa, id_map, dwa_stats_after_compact) = if reference_terminal_expansion {
        let partition = terminal_partition
            .as_ref()
            .expect("active TI transport must retain its partition");
        let transport_modes_started_at = ti_profile_timing.then(Instant::now);
        let mut transport_active_terminals = active_terminals.to_vec();
        for &terminal in grammar.residual_isolation_classes.keys() {
            if let Some(slot) = transport_active_terminals.get_mut(terminal as usize) {
                *slot = false;
            }
        }
        let mut modes = binary_transport_modes_from_witnesses(
            tokenizer_for_build,
            &transport_active_terminals,
            partition,
            ti_transport_witness_rounds
                .as_ref()
                .expect("active TI transport must retain its transient exact round witnesses"),
        );
        ti_transport_modes_ms = transport_modes_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        let transport_coordinate_map = {
            let ordinary_state_map = &core_id_map.tokenizer_states;
            let canonicalize_started_at = ti_profile_timing.then(Instant::now);
            canonicalize_transport_mode_states(&mut modes, ordinary_state_map);
            ti_canonicalize_transport_modes_ms = canonicalize_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);

            let quotient_started_at = ti_profile_timing.then(Instant::now);
            let mut quotient = transport_coordinate_quotient(ordinary_state_map, &modes);
            // Cluster the final tsids by their core coordinate. Many final tsids
            // collapse onto a few core coordinates during expansion; ordering
            // them so equal-coordinate tsids are contiguous lets each lifted
            // weight collapse to one range per coordinate instead of scattering
            // into many. Pure relabel (masks are token-indexed, tsid names are
            // internal), so the global digest is unchanged.
            quotient.reorder_internal_by_representative_key(|representative| {
                ordinary_state_map
                    .original_to_internal
                    .get(representative as usize)
                    .copied()
                    .unwrap_or(u32::MAX)
            });
            ti_transport_coordinate_quotient_ms = quotient_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            quotient
        };

        let expansion_started_at = ti_profile_timing.then(Instant::now);
        let expanded_dwa = expand_representative_dwa_after_minimization(
            &core_dwa,
            &core_id_map.tokenizer_states,
            &transport_coordinate_map,
            &modes,
        );
        ti_post_dwa_expansion_ms = expansion_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        let raw_follow_started_at = ti_profile_timing.then(Instant::now);
        let raw_follow_restoration = restore_raw_follow_constraints_after_expansion(
            &expanded_dwa,
            disallowed_follows,
            grammar.num_terminals as usize,
            ignore_terminal,
        );
        let used_follow_row_quotient = raw_follow_restoration.used_follow_row_quotient;
        let expanded_dwa = raw_follow_restoration.dwa;
        ti_raw_follow_restore_ms = raw_follow_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        // Exact duplicate states remain equivalent under any subsequent
        // source-domain restriction. Quotient them before forward normalization
        // so both normalization and weighted minimization see the smaller graph.
        let premerge_disabled = std::env::var("GLRMASK_DISABLE_TI_PREMERGE").is_ok();
        let mut expanded_dwa = if premerge_disabled {
            expanded_dwa
        } else {
            quotient_exact_duplicate_states(expanded_dwa)
        };

        let forward_domain_started_at = ti_profile_timing.then(Instant::now);
        restrict_weights_to_forward_domains_in_place(&mut expanded_dwa);
        ti_forward_domain_normalize_ms = forward_domain_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        // The transport-coordinate map is still finer than the final raw
        // coordinate domain. Compact it before minimization: otherwise the
        // minimizer cannot see state equivalences enabled by the final TSID
        // quotient and can retain an avoidable TI-only state split.
        let mut final_id_map = core_id_map.clone();
        final_id_map.tokenizer_states = transport_coordinate_map;
        let mut expanded_artifact = MappedArtifact::new(expanded_dwa, final_id_map);
        let post_dwa_compact_started_at = Instant::now();
        // The compact follow-row product used by p1 already exposes the exact
        // pointwise grouping to minimization. Avoid rebuilding an equivalent
        // token layout before that pass; p0 retains the established precompact
        // canonicalization path.
        //
        // The pre-minimization `compact_dimensions_fast` used to canonicalize
        // the token layout so the minimizer could see TSID-quotient-enabled
        // state equivalences. Since the run-based lift (`from_tsid_runs_shared`)
        // and clustering relabel now emit canonical SharedTokenSet storage
        // directly, this pass reduces nothing (tsids 2602->2602, tokens 67->67
        // on p0) yet costs ~5ms rewriting every weight. Skipping it leaves the
        // digest AND num_parser_states byte-identical while cutting p0
        // post_dwa_compact ~7.5ms->2.9ms. A `GLRMASK_FORCE_PRECOMPACT` escape
        // hatch keeps the old path available for schemas that might still need
        // the extra canonicalization for state minimality.
        let force_precompact = std::env::var("GLRMASK_FORCE_PRECOMPACT").is_ok();
        if force_precompact
            && !(used_follow_row_quotient && expanded_artifact.artifact().stats().states <= 64)
        {
            if profiling {
                expanded_artifact.compact_dimensions_fast_with_stats();
            } else {
                expanded_artifact.compact_dimensions_fast();
            }
        }
        ti_post_dwa_compact_ms = post_dwa_compact_started_at.elapsed().as_secs_f64() * 1000.0;
        let (expanded_dwa, final_id_map) = expanded_artifact.into_parts();

        let post_dwa_minimize_started_at = ti_profile_timing.then(Instant::now);
        // TI lifting creates partial source domains. The precompacted local
        // layout already carries the useful density information, so retain
        // stable exact pointwise class order and avoid a second domain-ordering
        // pass. This is scoped to the post-DWA artifact.
        let pointwise_order = PointwiseClassOrder::Stable;
        let minimized_dwa = minimize_owned_with_pointwise_class_order(expanded_dwa, pointwise_order);
        ti_post_dwa_minimize_ms = post_dwa_minimize_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        // The minimizer can expose a little extra dimension locality. This
        // second pass is intentionally cheap because the final TSID domain was
        // already established before minimization.
        let mut minimized_artifact = MappedArtifact::new(minimized_dwa, final_id_map);
        let final_compact_started_at = Instant::now();
        let skip_final_compact = std::env::var("GLRMASK_SKIP_TI_FINAL_COMPACT")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true);
        if !skip_final_compact {
            if profiling {
                minimized_artifact.compact_dimensions_merge_only_fast_with_stats();
            } else {
                minimized_artifact.compact_dimensions_merge_only_fast();
            }
        }
        ti_post_dwa_compact_ms += final_compact_started_at.elapsed().as_secs_f64() * 1000.0;
        let stats = minimized_artifact.artifact().stats();
        let (dwa, id_map) = minimized_artifact.into_parts();
        (dwa, id_map, stats)
    } else {
        (core_dwa, core_id_map, core_dwa_stats_after_compact)
    };
    let ti_post_dwa_total_ms = ti_post_dwa_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let compact_ms = core_compact_ms + ti_post_dwa_compact_ms;

    if ti_profile_timing {
        eprintln!(
            "[glrmask/profile][ti_representative_core] partition={} ti_active={} ordinary_exact_tsids={} core_tsids={} id_map_ms={:.3} representative_nwa_build_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} core_compact_ms={:.3} core_total_ms={:.3}",
            partition_label,
            reference_terminal_expansion,
            equiv_profile.exact_reps,
            core_tsid_count,
            id_map_ms,
            trie_build_ms,
            determinize_ms,
            minimize_ms,
            core_compact_ms,
            ti_representative_core_total_ms,
        );
        eprintln!(
            "[glrmask/profile][ti_post_dwa_expansion] partition={} ti_active={} replay_maps_ms={:.3} canonicalize_transport_modes_ms={:.3} transport_coordinate_quotient_ms={:.3} expansion_ms={:.3} raw_follow_restore_ms={:.3} forward_domain_normalize_ms={:.3} post_dwa_minimize_ms={:.3} post_dwa_compact_ms={:.3} final_tsids={} expansion_total_ms={:.3}",
            partition_label,
            reference_terminal_expansion,
            ti_transport_modes_ms,
            ti_canonicalize_transport_modes_ms,
            ti_transport_coordinate_quotient_ms,
            ti_post_dwa_expansion_ms,
            ti_raw_follow_restore_ms,
            ti_forward_domain_normalize_ms,
            ti_post_dwa_minimize_ms,
            ti_post_dwa_compact_ms,
            id_map.num_tsids(),
            ti_post_dwa_total_ms,
        );
        eprintln!(
            "[glrmask/profile][terminal_interchangeability_plan] partition={} ti_active={} strict_reference={} rounds={} classes={} additional_merged_members={} discovery_ms={:.3} restricted_observation_seed_ms={:.3} transport_modes_ms={:.3} coalesced_disallowed_follows_ms={:.3} canonicalize_transport_modes_ms={:.3} transport_coordinate_quotient_ms={:.3}",
            partition_label,
            reference_terminal_expansion,
            strict_reference,
            ti_round_count,
            terminal_partition.as_ref().map_or(0, |partition| partition.len()),
            ti_additional_merged_members,
            ti_discovery_ms,
            ti_restricted_observation_seed_ms,
            ti_transport_modes_ms,
            ti_coalesced_disallowed_follows_ms,
            ti_canonicalize_transport_modes_ms,
            ti_transport_coordinate_quotient_ms,
        );
    }

    let id_map_attributed_ms = equiv_profile.raw_analysis_base_init_ms
        + equiv_profile.analysis_view_build_ms
        + equiv_profile.effective_follows_normalize_ms
        + equiv_profile.prepare_inputs_ms
        + equiv_profile.byte_class_setup_ms
        + equiv_profile.token_dedup_ms
        + equiv_profile.restricted_observation_state_equiv_ms
        + equiv_profile.max_length_state_equiv_ms
        + equiv_profile.vocab_equiv_ms
        + equiv_profile.exact_state_equiv_ms
        + equiv_profile.id_map_finalize_ms;
    let id_map_unattributed_ms = (id_map_ms - id_map_attributed_ms).max(0.0);

    if l2p_timing_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l2p] partition={} vocab_tokens={} active_terminals={} original_states={} tsids={} internal_vocab_entries={} initial_states_considered={} max_length_skipped={} max_token_len={} token_len_gt_4={} token_len_gt_8={} token_len_gt_16={} token_len_gt_32={} token_len_gt_64={} raw_analysis_base_init_ms={:.3} analysis_view_build_ms={:.3} active_mask_filter_ms={:.3} effective_follows_normalize_ms={:.3} prepare_inputs_ms={:.3} byte_class_setup_ms={:.3} vocab_analysis_dfa_build_ms={:.3} token_dedup_ms={:.3} max_length_state_equiv_ms={:.3} vocab_equiv_ms={:.3} exact_state_equiv_ms={:.3} id_map_finalize_ms={:.3} id_map_unattributed_ms={:.3} max_length_reps={} exact_reps={} exact_rep_confirmation_used={} fast_sound_id_map_used={} max_length_reduction_pct={:.2} exact_reduction_pct={:.2} restricted_observation_state_equiv_ms={:.3} restricted_observation_reps={} id_map_ms={:.3} tsid_fallback_ms={:.3} vocab_tree_ms={:.3} possible_matches_ms={:.3} seed_ms={:.3} terminal_nwa_build_ms={:.3} nwa_trie_walk_ms={:.3} nwa_flush_ms={:.3} nwa_flush_leaf_ms={:.3} nwa_flush_future_ms={:.3} nwa_flush_weight_ms={:.3} nwa_trie_self_loop_ms={:.3} nwa_trie_execute_ms={:.3} nwa_trie_match_filter_ms={:.3} nwa_trie_end_state_ms={:.3} nwa_trie_match_process_ms={:.3} nwa_trie_continuation_weight_ms={:.3} nwa_trie_execute_calls={} nwa_trie_execute_input_bytes={} nwa_trie_matches={} nwa_trie_end_states={} nwa_trie_self_loop_checks={} nwa_trie_self_loop_skips={} nwa_trie_self_loop_source_nodes={} nwa_trie_self_loop_skipped_source_nodes={} nwa_trie_self_loop_cache_misses={} nwa_future_terminal_additions={} nwa_match_transition_additions={} nwa_states={}->{}->{}->{}->{} always_allowed_ms={:.3} collapse_ms={:.3} disallowed_ms={:.3} prune_ms={:.3} canonicalize_ms={:.3} postprocess_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} compact_ms={:.3} minimize_states={} dwa_states={} dwa_transitions={} dwa_transition_pairs={} dwa_interned_ranges_before_compact={} dwa_interned_ranges_after_compact={} total_ms={:.3}",
            partition_label,
            vocab.entries.len(),
            num_active_terminals,
            num_original_states,
            id_map.num_tsids(),
            internal_vocab_count,
            equiv_profile.initial_states_considered,
            equiv_profile.max_length_skipped,
            equiv_profile.max_token_len,
            equiv_profile.token_len_gt_4,
            equiv_profile.token_len_gt_8,
            equiv_profile.token_len_gt_16,
            equiv_profile.token_len_gt_32,
            equiv_profile.token_len_gt_64,
            equiv_profile.raw_analysis_base_init_ms,
            equiv_profile.analysis_view_build_ms,
            equiv_profile.active_mask_filter_ms,
            equiv_profile.effective_follows_normalize_ms,
            equiv_profile.prepare_inputs_ms,
            equiv_profile.byte_class_setup_ms,
            equiv_profile.vocab_analysis_dfa_build_ms,
            equiv_profile.token_dedup_ms,
            equiv_profile.max_length_state_equiv_ms,
            equiv_profile.vocab_equiv_ms,
            equiv_profile.exact_state_equiv_ms,
            equiv_profile.id_map_finalize_ms,
            id_map_unattributed_ms,
            equiv_profile.max_length_reps,
            equiv_profile.exact_reps,
            equiv_profile.exact_rep_confirmation_used,
            fast_sound_id_map_used,
            max_length_reduction_pct,
            exact_reduction_pct,
            equiv_profile.restricted_observation_state_equiv_ms,
            equiv_profile.restricted_observation_reps,
            id_map_ms,
            0.0,
            vocab_tree_ms,
            possible_matches_ms,
            seed_ms,
            trie_build_ms,
            nwa_build_profile.trie_walk_ms,
            nwa_build_profile.flush_ms,
            nwa_build_profile.flush_leaf_ms,
            nwa_build_profile.flush_future_ms,
            nwa_build_profile.flush_weight_ms,
            nwa_build_profile.trie_self_loop_ms,
            nwa_build_profile.trie_execute_ms,
            nwa_build_profile.trie_match_filter_ms,
            nwa_build_profile.trie_end_state_ms,
            nwa_build_profile.trie_match_process_ms,
            nwa_build_profile.trie_continuation_weight_ms,
            nwa_build_profile.trie_execute_calls,
            nwa_build_profile.trie_execute_input_bytes,
            nwa_build_profile.trie_matches,
            nwa_build_profile.trie_end_states,
            nwa_build_profile.trie_self_loop_checks,
            nwa_build_profile.trie_self_loop_skips,
            nwa_build_profile.trie_self_loop_source_nodes,
            nwa_build_profile.trie_self_loop_skipped_source_nodes,
            nwa_build_profile.trie_self_loop_cache_misses,
            nwa_build_profile.future_terminal_additions,
            nwa_build_profile.match_transition_additions,
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
            compact_ms,
            dwa_stats_before_compact.states,
            dwa_stats_after_compact.states,
            dwa_stats_after_compact.transitions,
            dwa_stats_after_compact.transition_pairs,
            dwa_stats_before_compact.interned_ranges,
            dwa_stats_after_compact.interned_ranges,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let output = LocalIdMapTerminalDwa {
        id_map,
        dwa,
        profile: TerminalDwaPhaseProfile {
            id_map_ms,
            terminal_dwa_ms: vocab_tree_ms
                + possible_matches_ms
                + seed_ms
                + trie_build_ms
                + postprocess_ms
                + determinize_ms
                + minimize_ms
                + ti_post_dwa_total_ms,
            compact_ms,
            ..TerminalDwaPhaseProfile::default()
        },
    };

    if strict_reference
        || global_token_position_strict_reference
        || first_byte_vocab_factor_strict_reference
    {
        // Rebuild the local artifact under the appropriate suppressed feature
        // set, then compare completed weighted terminal languages in original
        // tokenizer-state and token coordinates. Global token-position strict
        // mode validates C against the pre-C raw construction; TI strict mode
        // validates only TI against the same C-seeded construction.
        let strict_baseline_started_at = Instant::now();
        let baseline = {
            let _strict_reference_suppress =
                SuppressTerminalInterchangeabilityStrictReference::new();
            let _suppress_ti = strict_reference.then(SuppressTerminalInterchangeability::new);
            let _suppress_global = global_token_position_strict_reference
                .then(SuppressGlobalTokenPosition::new);
            let _suppress_first_byte_vocab_factor = first_byte_vocab_factor_strict_reference
                .then(SuppressFirstByteVocabFactor::new);
            build_l2p_id_map_and_terminal_dwa(
                partition_label,
                tokenizer,
                vocab,
                terminal_coloring,
                use_terminal_coloring,
                ignore_terminal,
                grammar,
                always_allowed_follows,
                active_terminals,
                disallowed_follows,
                token_path_disallowed_follows,
                normalized_token_path_disallowed_follows,
                shared_vocab_dfa_cache,
                shared_original_vocab_dfa_cache,
                shared_original_vocab_analysis_dfa_cache,
                shared_transition_cache,
                shared_ti_output_cache,
                flat_trans,
                prebuilt_token_trie,
                initial_state_map,
            )
            .expect("terminal interchangeability baseline L2P build unexpectedly returned None")
        };
        let strict_baseline_build_ms = strict_baseline_started_at.elapsed().as_secs_f64() * 1000.0;
        let strict_compare_started_at = Instant::now();
        terminal_dwa_equivalence::compare(&baseline, &output).unwrap_or_else(|mismatch| {
            panic!(
                "optimized L2P artifact differed from exact baseline: partition={} {}",
                partition_label,
                mismatch,
            )
        });
        let strict_compare_ms = strict_compare_started_at.elapsed().as_secs_f64() * 1000.0;
        if ti_profile_timing {
            eprintln!(
                "[glrmask/profile][l2p_strict_reference] partition={} ti_reference={} global_token_position_reference={} first_byte_vocab_factor_reference={} baseline_build_ms={:.3} terminal_dwa_equivalence_ms={:.3} differs=false",
                partition_label,
                strict_reference,
                global_token_position_strict_reference,
                first_byte_vocab_factor_strict_reference,
                strict_baseline_build_ms,
                strict_compare_ms,
            );
        }
    }

    Some(output)
}

#[cfg(test)]
mod ti_mre_tests {
    use std::{env, ffi::OsString, sync::Mutex};

    use crate::{Constraint, Vocab};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        original: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let original = env::var_os(key);
            unsafe {
                env::set_var(key, value);
            }
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(value) => unsafe {
                    env::set_var(self.key, value);
                },
                None => unsafe {
                    env::remove_var(self.key);
                },
            }
        }
    }

    #[test]
    fn p7_and_p8_use_terminal_interchangeability_by_default() {
        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let _enabled = EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "0");

        assert!(super::l2p_terminal_interchangeability_enabled_for_partition("p7"));
        assert!(super::l2p_terminal_interchangeability_enabled_for_partition("p8"));
    }

    #[test]
    fn terminal_interchangeability_policy_leaves_generic_partitions_unchanged() {
        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let _enabled = EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "0");

        assert!(super::l2p_terminal_interchangeability_enabled_for_partition("p0"));
        assert!(super::l2p_terminal_interchangeability_enabled_for_partition("p7"));
    }

    #[test]
    fn terminal_interchangeability_policy_defaults_enabled_and_honors_explicit_disable() {
        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let original = env::var_os("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY");
        unsafe {
            env::remove_var("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY");
        }
        let _restore = EnvVarGuard {
            key: "GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY",
            original,
        };

        assert!(super::l2p_terminal_interchangeability_enabled_for_partition("p0"));
        let _disabled = EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "1");
        assert!(!super::l2p_terminal_interchangeability_enabled_for_partition("p0"));
    }

    #[test]
    fn p7_boundary_bypass_matches_forced_full_ti_reference() {
        let grammar = r#"
start S;
t TRUE ::= "true";
t FALSE ::= "false";
t NULL ::= "null";
nt S ::= TRUE | FALSE | NULL;
"#;
        let vocab = Vocab::new(
            vec![
                (0, b" true".to_vec()),
                (1, b" false".to_vec()),
                (2, b" null".to_vec()),
                (3, b"[true".to_vec()),
                (4, b" -".to_vec()),
            ]);

        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let _structural = EnvVarGuard::set("GLRMASK_STRUCTURAL_BOUNDARY_LEXICAL_PARTITION", "1");
        let _enabled = EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "0");
        let _strict = EnvVarGuard::set(
            "GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE",
            "1",
        );
        Constraint::from_glrm_grammar(grammar, &vocab)
            .expect("P7 local TI bypass must match the forced full-TI artifact");
    }

    #[test]
    fn p8_boundary_bypass_matches_forced_full_ti_reference() {
        let grammar = r#"
start S;
t QUOTE ::= "\"";
t IDENT ::= /[A-Za-z_][A-Za-z0-9_]*/;
nt S ::= QUOTE IDENT;
"#;
        let vocab = Vocab::new(
            vec![
                (0, b"\"A".to_vec()),
                (1, b"\"Z".to_vec()),
                (2, b"\"_".to_vec()),
            ]);

        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let _structural = EnvVarGuard::set("GLRMASK_STRUCTURAL_BOUNDARY_LEXICAL_PARTITION", "1");
        let _enabled = EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "0");
        let _strict = EnvVarGuard::set(
            "GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE",
            "1",
        );
        Constraint::from_glrm_grammar(grammar, &vocab)
            .expect("P8 local TI bypass must match the forced full-TI artifact");
    }

    #[test]
    fn forced_all_l2p_enum_skips_p8_first_byte_factorization() {
        let vocab = Vocab::new(
            vec![
                (0, b"\"red\"".to_vec()),
                (1, b"\"blue\"".to_vec()),
                (2, b"\"green\"".to_vec()),
            ]);

        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let _force_l2p = EnvVarGuard::set("GLRMASK_FORCE_ALL_L2P", "1");
        let _disable_vocab_split = EnvVarGuard::set("GLRMASK_SPLIT_L2P_VOCAB", "0");
        let _enabled = EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "0");
        let _strict = EnvVarGuard::set(
            "GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE",
            "1",
        );

        Constraint::from_json_schema(r#"{"enum":["red","blue"]}"#, &vocab)
            .expect("forced-all-L2P enum must match the strict TI reference");
    }

    #[test]
    fn partitioned_epsilon_ti_matches_strict_reference() {
        let grammar = r#"
start S;
lexer group a ::= A;
lexer group b ::= B;
lexer group c ::= C;
t A ::= "x";
t B ::= [xy] & [xz];
t C ::= "z";
nt S ::= A | B | C;
"#;
        let vocab = Vocab::new(
            vec![
                (0, b"x".to_vec()),
                (1, b"xx".to_vec()),
                (2, b"z".to_vec()),
            ]);

        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let _adaptive = EnvVarGuard::set("GLRMASK_LEXER_ADAPTIVE", "0");
        let _force_l2p = EnvVarGuard::set("GLRMASK_FORCE_ALL_L2P", "1");
        let _disable_vocab_split = EnvVarGuard::set("GLRMASK_SPLIT_L2P_VOCAB", "0");
        let _enabled = EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "0");
        let _strict = EnvVarGuard::set(
            "GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE",
            "1",
        );

        Constraint::from_glrm_grammar(grammar, &vocab)
            .expect("epsilon-NFA TI must match its TI-off symbolic reference");
    }

    #[test]
    fn partitioned_epsilon_p8_global_token_position_matches_strict_reference() {
        let grammar = r#"
start S;
lexer group quote ::= QUOTE;
lexer group ident ::= IDENT;
t QUOTE ::= "\"";
t IDENT ::= /[A-Za-z_][A-Za-z0-9_]*/;
nt S ::= QUOTE IDENT;
"#;
        let vocab = Vocab::new(
            vec![
                (0, b"\"A".to_vec()),
                (1, b"\"Z".to_vec()),
                (2, b"\"_".to_vec()),
            ]);

        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let _adaptive = EnvVarGuard::set("GLRMASK_LEXER_ADAPTIVE", "0");
        let _structural = EnvVarGuard::set("GLRMASK_STRUCTURAL_BOUNDARY_LEXICAL_PARTITION", "1");
        let _disable_ti =
            EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "1");
        let _strict = EnvVarGuard::set(
            "GLRMASK_L2P_GLOBAL_TOKEN_POSITION_STRICT_REFERENCE",
            "1",
        );

        Constraint::from_glrm_grammar(grammar, &vocab)
            .expect("epsilon-NFA C quotient must match its suppressed-C symbolic reference");
    }

    #[test]
    fn representative_only_vocab_equivalence_mre() {
        // `b" _"` completes SPACE then is a live prefix of CLASS. With TI
        // enabled, CLASS is hidden behind representative FROM during equivalence
        // analysis. Because FROM cannot follow SPACE but CLASS can, the
        // representative-labeled follow table must be COALESCED (a follow is
        // disallowed for the class only if disallowed for every member);
        // otherwise equivalence prunes the FROM-class continuation, merges
        // `b" !"`/`b" _"`, and the completed terminal DWA underaccepts
        // `[SPACE, CLASS]`. Regression guard for that coalescing fix.
        let grammar = r#"
start S;
t V ::= /.+/;
t SPACE ::= " ";
t FROM ::= /_a_/;
t CLASS ::= /_b_/;
nt S ::= FROM V | SPACE V SPACE CLASS;
"#;
        let vocab = Vocab::new(vec![(0, b" !".to_vec()), (1, b" _".to_vec())]);

        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let _enabled = EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "0");
        Constraint::from_glrm_grammar(grammar, &vocab)
            .expect("TI must preserve the completed terminal-DWA language");
    }
}
