use crate::automata::lexer::Lexer;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use super::state_equivalence::global_token_position::GlobalTokenPositionStatePartition;
use crate::compiler::stages::id_map_and_terminal_dwa::grammar_helpers::ignore_transparent_disallowed_follows;
use super::state_equivalence::{
    build_state_map_from_subset_representatives, resolve_l2p_pipeline_config,
    run_state_equivalence_pipeline, StateEquivalenceScope,
};
use super::state_equivalence::nfa::{
    PrebuiltSparsePowersetRefinement, build_bounded_analysis_view,
    build_relevant_powerset_view,
};
use crate::ds::bitset::BitSet;
use super::compat::TokenizerView;
use super::disallowed_follows::normalize_disallowed_follows;
use super::shared::{
    TokenDedup,
    expand_vocab_classes,
    hash_byte_class_seq,
    representative_tokens_for_vocab_classes,
    tokenizer_group_count,
};
use super::state::fast as state_equivalence_analysis;
use super::vocab::fast as vocab_equivalence_analysis;

fn deduplicate_tokens_by_byte_class<'a, S: AsRef<[u8]>>(
    tokens: &'a [S],
    byte_to_class: &[u8; 256],
) -> TokenDedup<'a> {
    let mut hash_to_repr = HashMap::with_capacity(tokens.len() / 2);
    let mut representative_token_bytes = Vec::new();
    let mut original_to_repr = Vec::with_capacity(tokens.len());

    for token in tokens {
        let bytes = token.as_ref();
        let repr_idx = *hash_to_repr
            .entry(hash_byte_class_seq(bytes, byte_to_class))
            .or_insert_with(|| {
                let idx = representative_token_bytes.len();
                representative_token_bytes.push(bytes);
                idx
            });
        original_to_repr.push(repr_idx);
    }

    TokenDedup {
        representative_token_bytes,
        original_to_repr,
    }
}

/// Exact byte congruence induced by the active terminal languages themselves.
///
/// The enclosing tokenizer may retain transition distinctions from inactive
/// terminal components. Those distinctions are unobservable to an L2P
/// partition whose outputs are projected to `active_groups`. Recompiling only
/// the active expressions yields the language product we actually observe.
/// Two bytes in the same transition column of that DFA have identical action
/// from every residual of every active terminal language, so replacing either
/// byte by the other preserves all active-terminal match positions and future
/// residuals. Follow constraints only filter terminal labels and therefore do
/// not invalidate this byte congruence.
fn active_terminal_language_byte_classes(
    tokenizer: &Tokenizer,
    active_groups: &[bool],
) -> Option<[u8; 256]> {
    let active_exprs = active_groups
        .iter()
        .enumerate()
        .filter_map(|(terminal, &active)| {
            active
                .then(|| tokenizer.terminal_expr(terminal as u32).cloned())
                .flatten()
        })
        .collect::<Vec<_>>();
    if active_exprs.is_empty() {
        return None;
    }
    let regex = crate::automata::lexer::compile::build_regex(&active_exprs);
    let active_tokenizer = regex.into_tokenizer(
        active_exprs.len() as u32,
        Some(Arc::from(active_exprs)),
    );
    let active_view = TokenizerView::new(&active_tokenizer);
    Some(super::compat::compute_byte_classes(active_view.dfa()))
}

fn active_terminal_language_tokenizer_and_follows(
    tokenizer: &Tokenizer,
    active_groups: &[bool],
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> Option<(Tokenizer, BTreeMap<u32, BitSet>)> {
    let active_global_ids = active_groups
        .iter()
        .enumerate()
        .filter_map(|(terminal, &active)| active.then_some(terminal as u32))
        .collect::<Vec<_>>();
    let active_exprs = active_global_ids
        .iter()
        .map(|&terminal| tokenizer.terminal_expr(terminal).cloned())
        .collect::<Option<Vec<_>>>()?;
    if active_exprs.is_empty() {
        return None;
    }
    let regex = crate::automata::lexer::compile::build_regex(&active_exprs);
    let active_tokenizer = regex.into_tokenizer(
        active_exprs.len() as u32,
        Some(Arc::from(active_exprs)),
    );
    let mut local_disallowed = BTreeMap::<u32, BitSet>::new();
    for (local_source, &global_source) in active_global_ids.iter().enumerate() {
        let Some(global_row) = disallowed_follows.get(&global_source) else {
            continue;
        };
        let mut local_row = BitSet::new(active_global_ids.len());
        for (local_target, &global_target) in active_global_ids.iter().enumerate() {
            if global_row.contains(global_target as usize) {
                local_row.set(local_target);
            }
        }
        if !local_row.is_zero() {
            local_disallowed.insert(local_source as u32, local_row);
        }
    }
    Some((active_tokenizer, local_disallowed))
}

fn projected_sparse_prepass_edges(
    powerset: &super::state_equivalence::nfa::RelevantPowersetView,
    relevant_bytes: &[bool; 256],
    byte_to_class: Option<&[u8; 256]>,
) -> (Vec<u32>, Vec<(u8, u32)>) {
    let mut keep_byte = [false; 256];
    if let Some(byte_to_class) = byte_to_class {
        let class_count = byte_to_class
            .iter()
            .copied()
            .max()
            .map_or(0, |class| class as usize + 1);
        let mut seen = vec![false; class_count];
        for byte in 0..256usize {
            if !relevant_bytes[byte] {
                continue;
            }
            let class = byte_to_class[byte] as usize;
            if !seen[class] {
                seen[class] = true;
                keep_byte[byte] = true;
            }
        }
    } else {
        keep_byte = *relevant_bytes;
    }

    let live = powerset
        .states
        .iter()
        .map(|state| {
            !state.finalizers.is_empty() || !state.possible_future_group_ids.is_empty()
        })
        .collect::<Vec<_>>();
    let mut edge_offsets = Vec::with_capacity(powerset.states.len() + 1);
    let mut edges = Vec::new();
    edge_offsets.push(0u32);
    for source in 0..powerset.states.len() {
        if live[source] {
            let start = powerset.edge_offsets[source] as usize;
            let end = powerset.edge_offsets[source + 1] as usize;
            edges.extend(
                powerset.edges[start..end]
                    .iter()
                    .copied()
                    .filter(|&(byte, target)| {
                        keep_byte[byte as usize] && live[target as usize]
                    }),
            );
        }
        edge_offsets.push(edges.len() as u32);
    }
    (edge_offsets, edges)
}

fn byte_classes_on_state_quotient(
    tokenizer: &TokenizerView,
    state_map: &ManyToOneIdMap,
) -> [u8; 256] {
    let dfa = tokenizer.dfa();
    let mut classes_by_column = HashMap::<Vec<u32>, u8>::new();
    let mut byte_to_class = [0u8; 256];
    let mut column = Vec::with_capacity(state_map.representative_original_ids.len());

    for byte in 0..=u8::MAX {
        column.clear();
        for &representative in &state_map.representative_original_ids {
            let target = dfa.trans(representative as usize, byte as usize);
            column.push(if target == u32::MAX {
                u32::MAX
            } else {
                state_map.original_to_internal[target as usize]
            });
        }
        let next = classes_by_column.len() as u8;
        byte_to_class[byte as usize] = *classes_by_column.entry(column.clone()).or_insert(next);
    }

    if std::env::var_os("GLRMASK_DUMP_ACTIVE_BYTE_CLASSES").is_some() {
        let probes = b" abcxyz_AZ09\\\"";
        eprintln!(
            "[glrmask/dump][active_byte_classes] classes={} probes={:?}",
            classes_by_column.len(),
            probes
                .iter()
                .map(|&byte| (byte, byte_to_class[byte as usize]))
                .collect::<Vec<_>>(),
        );
        for &(left, right) in &[(b'a', b'b'), (b'a', b'_'), (b'a', b'0')] {
            let witness = state_map
                .representative_original_ids
                .iter()
                .copied()
                .find_map(|representative| {
                    let left_target = dfa.trans(representative as usize, left as usize);
                    let right_target = dfa.trans(representative as usize, right as usize);
                    let left_class = if left_target == u32::MAX {
                        u32::MAX
                    } else {
                        state_map.original_to_internal[left_target as usize]
                    };
                    let right_class = if right_target == u32::MAX {
                        u32::MAX
                    } else {
                        state_map.original_to_internal[right_target as usize]
                    };
                    (left_class != right_class).then_some((
                        representative,
                        left_target,
                        left_class,
                        right_target,
                        right_class,
                    ))
                });
            eprintln!(
                "[glrmask/dump][active_byte_class_witness] left={} right={} witness={:?}",
                left, right, witness,
            );
        }
    }

    byte_to_class
}

/// Diagnostic oracle for the literal construction:
///
/// 1. start from the deterministic, active-terminal-filtered analysis DFA;
/// 2. observe finalizers only (as the lexer DFA minimizer does);
/// 3. delete bytes absent from this vocabulary partition;
/// 4. compute the stable deterministic Moore quotient;
/// 5. compare byte transition columns on that minimized quotient.
pub(crate) fn literal_active_finalizer_minimized_byte_classes(
    tokenizer: &TokenizerView,
    relevant_bytes: &[bool; 256],
) -> ([u8; 256], usize) {
    let dfa = tokenizer.dfa();
    let num_states = dfa.states.len();

    let mut label_ids = HashMap::<Vec<usize>, u32>::new();
    let mut classes = vec![0u32; num_states];
    for (state, dfa_state) in dfa.states.iter().enumerate() {
        let next = label_ids.len() as u32;
        classes[state] = *label_ids
            .entry(dfa_state.finalizers.clone())
            .or_insert(next);
    }

    let active_bytes = relevant_bytes
        .iter()
        .enumerate()
        .filter_map(|(byte, &active)| active.then_some(byte))
        .collect::<Vec<_>>();

    loop {
        let mut by_signature = HashMap::<Vec<u32>, u32>::new();
        let mut next_classes = vec![0u32; num_states];
        let mut signature = Vec::with_capacity(1 + active_bytes.len());
        for state in 0..num_states {
            signature.clear();
            signature.push(classes[state]);
            for &byte in &active_bytes {
                let target = dfa.trans(state, byte);
                signature.push(if target == u32::MAX {
                    u32::MAX
                } else {
                    classes[target as usize]
                });
            }
            let next = by_signature.len() as u32;
            next_classes[state] = *by_signature.entry(signature.clone()).or_insert(next);
        }
        if next_classes == classes {
            break;
        }
        classes = next_classes;
    }

    let class_count = classes
        .iter()
        .copied()
        .max()
        .map_or(0, |class| class as usize + 1);
    let mut representatives = vec![usize::MAX; class_count];
    for (state, &class) in classes.iter().enumerate() {
        representatives[class as usize] = representatives[class as usize].min(state);
    }

    let mut classes_by_column = HashMap::<Vec<u32>, u8>::new();
    let mut byte_to_class = [0u8; 256];
    let mut column = Vec::with_capacity(class_count);
    for byte in 0..=u8::MAX {
        column.clear();
        if relevant_bytes[byte as usize] {
            for &representative in &representatives {
                let target = dfa.trans(representative, byte as usize);
                column.push(if target == u32::MAX {
                    u32::MAX
                } else {
                    classes[target as usize]
                });
            }
        } else {
            column.push(u32::MAX);
        }
        let next = classes_by_column.len() as u8;
        byte_to_class[byte as usize] = *classes_by_column.entry(column.clone()).or_insert(next);
    }

    if std::env::var_os("GLRMASK_DUMP_ACTIVE_BYTE_CLASSES").is_some() {
        let probes = b" abcxyz_AZ09\\\"";
        eprintln!(
            "[glrmask/dump][literal_active_byte_classes] classes={} probes={:?}",
            classes_by_column.len(),
            probes
                .iter()
                .map(|&byte| (byte, byte_to_class[byte as usize]))
                .collect::<Vec<_>>(),
        );
        for &(left, right) in &[(b'a', b'b'), (b'a', b'_'), (b'a', b'0')] {
            let witness = representatives.iter().copied().find_map(|representative| {
                let left_target = dfa.trans(representative, left as usize);
                let right_target = dfa.trans(representative, right as usize);
                let left_class = if left_target == u32::MAX {
                    u32::MAX
                } else {
                    classes[left_target as usize]
                };
                let right_class = if right_target == u32::MAX {
                    u32::MAX
                } else {
                    classes[right_target as usize]
                };
                (left_class != right_class).then_some((
                    representative,
                    left_target,
                    left_class,
                    right_target,
                    right_class,
                    dfa.states[representative].finalizers.clone(),
                ))
            });
            eprintln!(
                "[glrmask/dump][literal_active_byte_class_witness] left={} right={} witness={:?}",
                left, right, witness,
            );
        }
    }

    (byte_to_class, class_count)
}

fn build_state_map(
    state_classes: &BTreeSet<BTreeSet<usize>>,
    num_dfa_states: usize,
) -> ManyToOneIdMap {
    let mut original_to_internal = vec![u32::MAX; num_dfa_states];
    let mut internal_to_originals = Vec::new();
    let mut representative_original_ids = Vec::new();

    for class in state_classes {
        let internal_id = internal_to_originals.len() as u32;
        let originals: Vec<u32> = class.iter().map(|&state| state as u32).collect();
        for &state in &originals {
            original_to_internal[state as usize] = internal_id;
        }
        representative_original_ids
            .push(*originals.first().expect("state class must be non-empty"));
        internal_to_originals.push(originals);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

fn build_state_map_composed(
    state_classes: &BTreeSet<BTreeSet<usize>>,
    num_dfa_states: usize,
    initial_state_map: &ManyToOneIdMap,
) -> ManyToOneIdMap {
    let mut rep_to_new_internal = vec![u32::MAX; num_dfa_states];
    let mut new_internal_to_originals: Vec<Vec<u32>> = Vec::new();
    for class in state_classes {
        let internal_id = new_internal_to_originals.len() as u32;
        let originals: Vec<u32> = class.iter().map(|&state| state as u32).collect();
        for &state in &originals {
            rep_to_new_internal[state as usize] = internal_id;
        }
        new_internal_to_originals.push(originals);
    }

    let mut composed_original_to_internal = vec![u32::MAX; num_dfa_states];
    let mut composed_internal_to_originals = vec![Vec::new(); new_internal_to_originals.len()];
    let mut composed_reps = vec![u32::MAX; new_internal_to_originals.len()];

    for (orig_state, &init_internal) in initial_state_map.original_to_internal.iter().enumerate() {
        if init_internal == u32::MAX
            || (init_internal as usize) >= initial_state_map.representative_original_ids.len()
        {
            continue;
        }
        let init_rep = initial_state_map.representative_original_ids[init_internal as usize] as usize;
        let new_internal = rep_to_new_internal[init_rep];
        if new_internal != u32::MAX {
            composed_original_to_internal[orig_state] = new_internal;
            let bucket = &mut composed_internal_to_originals[new_internal as usize];
            if bucket.is_empty() {
                composed_reps[new_internal as usize] = init_rep as u32;
            }
            bucket.push(orig_state as u32);
        }
    }

    ManyToOneIdMap {
        original_to_internal: composed_original_to_internal,
        internal_to_originals: composed_internal_to_originals,
        representative_original_ids: composed_reps,
    }
}

fn build_vocab_map(
    vocab_classes: &BTreeSet<Vec<usize>>,
    token_ids: &[u32],
    max_token_id: u32,
) -> ManyToOneIdMap {
    let mut ordered_vocab_classes: Vec<(u32, Vec<u32>)> = vocab_classes
        .iter()
        .map(|class| {
            let mut min_tid = u32::MAX;
            let mut originals: Vec<u32> = Vec::with_capacity(class.len());
            for &idx in class {
                let tid = token_ids[idx];
                originals.push(tid);
                if tid < min_tid {
                    min_tid = tid;
                }
            }
            (min_tid, originals)
        })
        .collect();
    ordered_vocab_classes.sort_unstable_by_key(|(rep_tid, _)| *rep_tid);

    let mut original_to_internal = vec![u32::MAX; (max_token_id + 1) as usize];
    let mut internal_to_originals = Vec::with_capacity(ordered_vocab_classes.len());
    let mut representative_original_ids = Vec::with_capacity(ordered_vocab_classes.len());

    for (internal_id, (rep_tid, originals)) in ordered_vocab_classes.into_iter().enumerate() {
        for &token_id in &originals {
            original_to_internal[token_id as usize] = internal_id as u32;
        }
        representative_original_ids.push(rep_tid);
        internal_to_originals.push(originals);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

/// An identity map is an exact conservative token quotient. It is useful for
/// small structural partitions where vocab equivalence has no materialization
/// benefit, while its analysis cost dominates the enclosing partition wall.
fn build_identity_vocab_map(token_ids: &[u32], max_token_id: u32) -> ManyToOneIdMap {
    let mut token_ids = token_ids.to_vec();
    token_ids.sort_unstable();
    token_ids.dedup();

    let mut original_to_internal = vec![u32::MAX; max_token_id as usize + 1];
    let mut internal_to_originals = Vec::with_capacity(token_ids.len());
    let mut representatives = Vec::with_capacity(token_ids.len());
    for (internal, token_id) in token_ids.into_iter().enumerate() {
        original_to_internal[token_id as usize] = internal as u32;
        internal_to_originals.push(vec![token_id]);
        representatives.push(token_id);
    }
    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids: representatives,
    }
}

/// Conservative vocabulary quotient for epsilon tokenizers: only byte-for-byte
/// aliases are merged. They necessarily have identical lexer behaviour, and
/// merging them is required because the vocabulary prefix tree has one leaf
/// per byte string. The final mask mapping expands the class back to every
/// original token id.
fn build_exact_byte_vocab_map(
    token_ids: &[u32],
    token_bytes: &[&[u8]],
    max_token_id: u32,
) -> ManyToOneIdMap {
    debug_assert_eq!(token_ids.len(), token_bytes.len());
    let mut classes = BTreeMap::<Vec<u8>, Vec<u32>>::new();
    for (&token_id, &bytes) in token_ids.iter().zip(token_bytes) {
        classes.entry(bytes.to_vec()).or_default().push(token_id);
    }

    let mut ordered = classes.into_values().collect::<Vec<_>>();
    for class in &mut ordered {
        class.sort_unstable();
    }
    ordered.sort_unstable_by_key(|class| class[0]);

    let mut original_to_internal = vec![u32::MAX; max_token_id as usize + 1];
    let mut representatives = Vec::with_capacity(ordered.len());
    for (internal, class) in ordered.iter().enumerate() {
        representatives.push(class[0]);
        for &original in class {
            original_to_internal[original as usize] = internal as u32;
        }
    }
    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals: ordered,
        representative_original_ids: representatives,
    }
}

struct PreparedEquivalenceInputs<'a> {
    max_token_id: u32,
    token_ids: Vec<u32>,
    token_bytes: Vec<&'a [u8]>,
    initial_states: Vec<usize>,
}

fn prepare_equivalence_inputs<'a>(
    tokenizer: &Tokenizer,
    vocab: &'a Vocab,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> PreparedEquivalenceInputs<'a> {
    let max_token_id = vocab.max_token_id();
    let mut token_bytes: Vec<&[u8]> = Vec::with_capacity(vocab.len());
    let mut token_ids: Vec<u32> = Vec::with_capacity(vocab.len());
    for (&tid, bytes) in vocab.entries.iter() {
        token_ids.push(tid);
        token_bytes.push(bytes.as_slice());
    }

    let initial_states = match initial_state_map {
        Some(map) => map.representative_original_ids.iter().map(|&s| s as usize).collect(),
        None => (0..tokenizer.num_states() as usize).collect(),
    };

    PreparedEquivalenceInputs {
        max_token_id,
        token_ids,
        token_bytes,
        initial_states,
    }
}


struct CombinedEquivalenceResult {
    vocab_classes: BTreeSet<Vec<usize>>,
    state_classes: BTreeSet<BTreeSet<usize>>,
}

pub(crate) struct CombinedEquivalenceProfile {
    pub(crate) initial_states_considered: usize,
    pub(crate) max_length_skipped: bool,
    pub(crate) max_token_len: usize,
    pub(crate) token_len_gt_4: usize,
    pub(crate) token_len_gt_8: usize,
    pub(crate) token_len_gt_16: usize,
    pub(crate) token_len_gt_32: usize,
    pub(crate) token_len_gt_64: usize,
    pub(crate) raw_analysis_base_init_ms: f64,
    pub(crate) analysis_view_build_ms: f64,
    /// Sub-timing within `analysis_view_build_ms`; do not add separately.
    pub(crate) active_mask_filter_ms: f64,
    pub(crate) effective_follows_normalize_ms: f64,
    pub(crate) prepare_inputs_ms: f64,
    pub(crate) byte_class_setup_ms: f64,
    pub(crate) vocab_analysis_dfa_build_ms: f64,
    pub(crate) token_dedup_ms: f64,
    pub(crate) restricted_observation_state_equiv_ms: f64,
    pub(crate) max_length_state_equiv_ms: f64,
    pub(crate) vocab_equiv_ms: f64,
    pub(crate) exact_state_equiv_ms: f64,
    pub(crate) id_map_finalize_ms: f64,
    pub(crate) restricted_observation_reps: usize,
    pub(crate) max_length_reps: usize,
    pub(crate) exact_reps: usize,
    pub(crate) exact_rep_confirmation_used: bool,
}

struct TokenLengthStats {
    gt_4: usize,
    gt_8: usize,
    gt_16: usize,
    gt_32: usize,
    gt_64: usize,
}

fn skip_max_length_for_partition(partition_label: &str) -> bool {
    static SKIPPED_PARTITIONS: OnceLock<Vec<String>> = OnceLock::new();
    SKIPPED_PARTITIONS
        .get_or_init(|| {
            std::env::var("GLRMASK_SKIP_MAX_LENGTH_PARTITIONS")
                .ok()
                .map(|value| {
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|label| !label.is_empty())
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default()
        })
        .iter()
        .any(|label| label == partition_label)
}

#[inline]
fn direct_refinement_work_is_no_larger(
    direct_token_bytes: usize,
    max_token_len: usize,
    active_byte_count: usize,
) -> bool {
    direct_token_bytes <= max_token_len.saturating_mul(active_byte_count)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum L2pNfaAnalysisViewPolicy {
    Adaptive,
    Bounded,
    Powerset,
}

impl L2pNfaAnalysisViewPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::Bounded => "bounded",
            Self::Powerset => "powerset",
        }
    }
}

#[inline]
fn l2p_nfa_analysis_view_policy() -> L2pNfaAnalysisViewPolicy {
    let Ok(value) = std::env::var("GLRMASK_L2P_NFA_RELEVANT_POWERSET_VIEW") else {
        return L2pNfaAnalysisViewPolicy::Adaptive;
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" | "adaptive" => L2pNfaAnalysisViewPolicy::Adaptive,
        "" | "1" | "true" | "yes" | "on" | "powerset" => {
            L2pNfaAnalysisViewPolicy::Powerset
        }
        "0" | "false" | "no" | "off" | "bounded" => L2pNfaAnalysisViewPolicy::Bounded,
        other => panic!(
            "invalid GLRMASK_L2P_NFA_RELEVANT_POWERSET_VIEW={other:?}; expected auto/adaptive, powerset/1/true/on, or bounded/0/false/off"
        ),
    }
}

#[inline]
fn l2p_nfa_relevant_powerset_max_states() -> usize {
    std::env::var("GLRMASK_L2P_NFA_RELEVANT_POWERSET_MAX_STATES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(8_192)
}

#[inline]
fn l2p_nfa_relevant_powerset_min_bounded_pairs() -> usize {
    std::env::var("GLRMASK_L2P_NFA_RELEVANT_POWERSET_MIN_BOUNDED_PAIRS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(500_000)
}

#[inline]
fn should_probe_l2p_nfa_powerset(
    policy: L2pNfaAnalysisViewPolicy,
    bounded_pair_estimate: usize,
    min_bounded_pairs: usize,
) -> bool {
    match policy {
        L2pNfaAnalysisViewPolicy::Adaptive => bounded_pair_estimate >= min_bounded_pairs,
        L2pNfaAnalysisViewPolicy::Bounded => false,
        L2pNfaAnalysisViewPolicy::Powerset => true,
    }
}

#[inline]
fn should_use_l2p_nfa_powerset(
    policy: L2pNfaAnalysisViewPolicy,
    candidate_present: bool,
    powerset_states: usize,
    max_states: usize,
) -> bool {
    match policy {
        L2pNfaAnalysisViewPolicy::Adaptive => {
            candidate_present && powerset_states <= max_states
        }
        L2pNfaAnalysisViewPolicy::Bounded => false,
        L2pNfaAnalysisViewPolicy::Powerset => true,
    }
}

fn should_skip_max_length_for_partition(
    partition_label: &str,
    initial_state_count: usize,
    projected_by_global: bool,
    direct_token_bytes: usize,
    max_token_len: usize,
    active_byte_count: usize,
) -> bool {
    if skip_max_length_for_partition(partition_label)
        || (projected_by_global && initial_state_count <= 8192)
    {
        return true;
    }

    // Both routes are exact. The k-bounded prepass performs up to one
    // state-transition refinement for every (length, active-byte) pair;
    // direct finite-vocabulary refinement must at least read every token byte.
    // Prefer the direct route when its input bound is no larger, avoiding a
    // prepass that cannot amortize its own full-DFA scans.
    direct_refinement_work_is_no_larger(
        direct_token_bytes,
        max_token_len,
        active_byte_count,
    )
}

const EXACT_REP_CONFIRMATION_MIN_STATES: usize = 2_000;
const EXACT_REP_CONFIRMATION_MIN_TOKENS: usize = 200;
const RAW_QUOTIENT_TINY_VOCAB_MAX_TOKENS: usize = 16;
const RAW_QUOTIENT_TINY_VOCAB_MAX_BYTES: usize = 64;
const RAW_QUOTIENT_STRUCTURAL_BOUNDARY_MAX_TOKENS: usize = 48;
const RAW_QUOTIENT_STRUCTURAL_BOUNDARY_MAX_BYTES: usize = 256;

fn token_length_stats(tokens: &[&[u8]]) -> TokenLengthStats {
    let mut stats = TokenLengthStats {
        gt_4: 0,
        gt_8: 0,
        gt_16: 0,
        gt_32: 0,
        gt_64: 0,
    };
    for token in tokens {
        let len = token.len();
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

/// Compose the raw restricted-observation quotient with its final exact
/// representative map without materializing ordered sets of raw states.
///
/// Scanning raw states in ascending order assigns dense IDs in the same order
/// as the generic `BTreeSet` path: each class is ordered by its first raw
/// member, and that member is also the retained raw representative.
fn compose_raw_quotient_state_map(
    pre_state_map: &ManyToOneIdMap,
    final_representative_for_preclass: &[usize],
) -> ManyToOneIdMap {
    compose_raw_quotient_state_map_impl(
        pre_state_map,
        final_representative_for_preclass,
        None,
    )
}

fn compose_raw_quotient_state_map_preserving_directional_representatives(
    pre_state_map: &ManyToOneIdMap,
    final_representative_for_preclass: &[usize],
) -> ManyToOneIdMap {
    compose_raw_quotient_state_map_impl(
        pre_state_map,
        final_representative_for_preclass,
        Some(&pre_state_map.representative_original_ids),
    )
}

fn compose_raw_quotient_state_map_impl(
    pre_state_map: &ManyToOneIdMap,
    final_representative_for_preclass: &[usize],
    representative_original_for_final_key: Option<&[u32]>,
) -> ManyToOneIdMap {
    assert_eq!(
        pre_state_map.internal_to_originals.len(),
        final_representative_for_preclass.len(),
        "raw quotient and exact representative map disagree",
    );
    let preclass_count = pre_state_map.internal_to_originals.len();
    let mut final_key_to_internal = vec![u32::MAX; preclass_count];
    let mut original_to_internal = vec![u32::MAX; pre_state_map.original_to_internal.len()];
    let mut internal_to_originals = Vec::<Vec<u32>>::new();
    let mut representative_original_ids = Vec::<u32>::new();

    for (raw_state, &preclass) in pre_state_map.original_to_internal.iter().enumerate() {
        if preclass == u32::MAX {
            continue;
        }
        let preclass = preclass as usize;
        assert!(preclass < preclass_count, "invalid raw quotient class");
        let final_key = final_representative_for_preclass[preclass];
        assert!(final_key < preclass_count, "invalid exact representative");
        let internal = if final_key_to_internal[final_key] == u32::MAX {
            let next = internal_to_originals.len() as u32;
            final_key_to_internal[final_key] = next;
            internal_to_originals.push(Vec::new());
            representative_original_ids.push(
                representative_original_for_final_key
                    .map_or(raw_state as u32, |representatives| representatives[final_key]),
            );
            next
        } else {
            final_key_to_internal[final_key]
        };
        original_to_internal[raw_state] = internal;
        internal_to_originals[internal as usize].push(raw_state as u32);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

fn build_internal_id_map_from_combined_result(
    tokenizer: &Tokenizer,
    initial_state_map: Option<&ManyToOneIdMap>,
    prepared: &PreparedEquivalenceInputs<'_>,
    result: &CombinedEquivalenceResult,
) -> InternalIdMap {
    let num_dfa_states = tokenizer.num_states() as usize;
    let state_map = match initial_state_map {
        Some(init_map) => build_state_map_composed(&result.state_classes, num_dfa_states, init_map),
        None => build_state_map(&result.state_classes, num_dfa_states),
    };
    let vocab_map = build_vocab_map(&result.vocab_classes, &prepared.token_ids, prepared.max_token_id);

    InternalIdMap {
        tokenizer_states: state_map,
        vocab_tokens: vocab_map,
    }
}

fn try_analyze_equivalences_with_token_position_partition(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    effective_disallowed: &BTreeMap<u32, BitSet>,
    active_groups: Option<&[bool]>,
    _shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
    _shared_analysis_dfa_cache: Option<&super::vocab::fast::SharedVocabAnalysisDfaCache>,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    token_position_partition: &GlobalTokenPositionStatePartition,
    effective_follows_prepare_ms: f64,
    pre_normalized_disallowed_follows: Option<&[BitSet]>,
) -> Option<(InternalIdMap, CombinedEquivalenceProfile)> {
    let active_groups = active_groups?;
    let seed = token_position_partition.as_many_to_one();
    let num_states = tokenizer.num_states() as usize;
    if seed.original_to_internal.len() != num_states
        || seed.representative_original_ids.is_empty()
        || vocab.entries.values().any(Vec::is_empty)
    {
        return None;
    }

    let prepare_inputs_started_at = Instant::now();
    let prepared = prepare_equivalence_inputs(tokenizer, vocab, None);
    if prepared.token_bytes.is_empty() {
        return None;
    }
    let token_len_stats = token_length_stats(&prepared.token_bytes);
    let max_token_len = prepared
        .token_bytes
        .iter()
        .map(|token| token.len())
        .max()
        .unwrap_or(0);
    let prepare_inputs_ms = prepare_inputs_started_at.elapsed().as_secs_f64() * 1000.0;

    let seed_states = seed
        .representative_original_ids
        .iter()
        .map(|&state| state as usize)
        .collect::<Vec<_>>();
    if seed_states.iter().any(|&state| state >= num_states) {
        return None;
    }

    let compatible_flat_trans = flat_trans.filter(|transitions| {
        transitions.len() == tokenizer.num_states() as usize * 256
    });
    let analysis_view_started_at = Instant::now();
    // C is a token-start partition, not an absolute lexer-state equivalence.
    // For epsilon lexers, analyze its representatives as epsilon-closed scanner
    // configurations in a bounded powerset view. These view IDs stay local to
    // token-boundary analysis and are projected back through C before returning.
    let bounded_analysis = tokenizer.has_epsilon_transitions().then(|| {
        build_bounded_analysis_view(
            tokenizer,
            &seed_states,
            &prepared.token_bytes,
            Some(active_groups),
        )
    });
    let raw_analysis_view = bounded_analysis.is_none().then(|| match compatible_flat_trans {
        Some(transitions) => TokenizerView::new_filtered_from_flat_trans(
            transitions,
            tokenizer,
            active_groups,
        ),
        None => TokenizerView::new_filtered(tokenizer, active_groups),
    });
    let tokenizer_view = bounded_analysis
        .as_ref()
        .map(|bounded| &bounded.tokenizer_view)
        .or(raw_analysis_view.as_ref())
        .expect("token-boundary analysis view must exist");
    let analysis_view_build_ms = analysis_view_started_at.elapsed().as_secs_f64() * 1000.0;

    // The C-seeded state/vocab equivalence walks only ever follow the
    // partition's token bytes, so byte classes only need to be exact for the
    // bytes that actually appear in the vocabulary. Restricting the byte-class
    // scan to those relevant bytes makes base construction proportional to the
    // partition's tiny alphabet instead of the full 18k-state x 256-byte table.
    // The base is partition-local (its byte classes depend on this partition's
    // relevant bytes), so it uses local caches rather than the cross-partition
    // shared ones to avoid contaminating other partitions.
    let mut relevant_bytes = [false; 256];
    for token in &prepared.token_bytes {
        for &byte in token.iter() {
            relevant_bytes[byte as usize] = true;
        }
    }
    let local_vocab_dfa_cache = vocab_equivalence_analysis::SharedVocabDfaCache::new();
    let local_analysis_dfa_cache =
        vocab_equivalence_analysis::SharedVocabAnalysisDfaCache::default();
    let compatible_shared_base = Some(local_vocab_dfa_cache.get_or_init(|| {
        vocab_equivalence_analysis::SharedVocabDfaBase::build_from_dfa_relevant(
            tokenizer_view.dfa(),
            &relevant_bytes,
        )
    }));

    let follows_normalize_started_at = Instant::now();
    let owned_normalized_disallowed_follows;
    let normalized_disallowed_follows = if let Some(rows) = pre_normalized_disallowed_follows {
        rows
    } else {
        owned_normalized_disallowed_follows =
            normalize_disallowed_follows(tokenizer_group_count(&tokenizer_view), effective_disallowed);
        &owned_normalized_disallowed_follows
    };
    let effective_follows_normalize_ms = effective_follows_prepare_ms
        + follows_normalize_started_at.elapsed().as_secs_f64() * 1000.0;

    let byte_class_started_at = Instant::now();
    let byte_to_class = compatible_shared_base
        .map(vocab_equivalence_analysis::SharedVocabDfaBase::byte_to_class)
        .unwrap_or_else(|| super::compat::compute_byte_classes(tokenizer_view.dfa()));
    let byte_class_setup_ms = byte_class_started_at.elapsed().as_secs_f64() * 1000.0;
    let token_dedup_started_at = Instant::now();
    let dedup = deduplicate_tokens_by_byte_class(&prepared.token_bytes, &byte_to_class);
    let token_dedup_ms = token_dedup_started_at.elapsed().as_secs_f64() * 1000.0;

    let seed_analysis_states = if let Some(bounded) = bounded_analysis.as_ref() {
        seed_states
            .iter()
            .map(|&raw| bounded.view_state_for_raw_start(raw))
            .collect::<Vec<_>>()
    } else {
        seed_states.clone()
    };
    let mut query_states = seed_analysis_states.clone();
    query_states.sort_unstable();
    query_states.dedup();
    let exact_rep_confirmation_used = query_states.len() >= EXACT_REP_CONFIRMATION_MIN_STATES
        && dedup.representative_token_bytes.len() >= EXACT_REP_CONFIRMATION_MIN_TOKENS;
    let exact_started_at = Instant::now();
    let shared_first_byte = bounded_analysis.as_ref().and_then(|_| {
        let first = dedup
            .representative_token_bytes
            .first()
            .and_then(|token| token.first())
            .copied()?;
        dedup
            .representative_token_bytes
            .iter()
            .all(|token| token.len() > 1 && token.first().copied() == Some(first))
            .then_some(first)
    });
    let state_representatives = if let Some(common_first) = shared_first_byte {
        let mut prefix_targets = BTreeSet::<usize>::new();
        let mut target_by_source = Vec::<u32>::with_capacity(query_states.len());
        for &source in &query_states {
            let target = tokenizer_view.dfa().trans(source, common_first as usize);
            target_by_source.push(target);
            if target != u32::MAX {
                prefix_targets.insert(target as usize);
            }
        }
        let prefix_targets = prefix_targets.into_iter().collect::<Vec<_>>();
        let suffix_tokens = dedup
            .representative_token_bytes
            .iter()
            .map(|token| &token[1..])
            .collect::<Vec<_>>();
        let prefix_target_representatives = if exact_rep_confirmation_used {
            state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base_with_initial_finalizers(
                    &tokenizer_view,
                    &suffix_tokens,
                    &prefix_targets,
                    &normalized_disallowed_follows,
                    None,
                    None,
                    Some(true),
                    compatible_shared_base,
                )
        } else {
            state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base_with_initial_finalizers(
                    &tokenizer_view,
                    &suffix_tokens,
                    &prefix_targets,
                    &normalized_disallowed_follows,
                    compatible_shared_base,
                )
        };
        let behavior_for_target = prefix_targets
            .iter()
            .copied()
            .zip(prefix_target_representatives)
            .collect::<BTreeMap<_, _>>();
        let mut representative_for_behavior = BTreeMap::<Option<usize>, usize>::new();
        let behavior_by_source = target_by_source
            .iter()
            .map(|&target| {
                (target != u32::MAX).then(|| behavior_for_target[&(target as usize)])
            })
            .collect::<Vec<_>>();
        for (&source, &behavior) in query_states.iter().zip(&behavior_by_source) {
            representative_for_behavior.entry(behavior).or_insert(source);
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][token_boundary_first_byte_factorization] partition={} first_byte={} source_states={} prefix_targets={} behavior_classes={}",
                partition_label,
                common_first,
                query_states.len(),
                prefix_targets.len(),
                representative_for_behavior.len(),
            );
        }
        behavior_by_source
            .into_iter()
            .map(|behavior| representative_for_behavior[&behavior])
            .collect()
    } else if exact_rep_confirmation_used {
        state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base(
            &tokenizer_view,
            &dedup.representative_token_bytes,
            &query_states,
            &normalized_disallowed_follows,
            None,
            None,
            Some(true),
            compatible_shared_base,
        )
    } else {
        state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base(
            &tokenizer_view,
            &dedup.representative_token_bytes,
            &query_states,
            &normalized_disallowed_follows,
            compatible_shared_base,
        )
    };
    let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;

    let tokenizer_states = if bounded_analysis.is_some() {
        let analysis_to_exact = query_states
            .iter()
            .copied()
            .zip(state_representatives.iter().copied())
            .collect::<BTreeMap<_, _>>();
        let first_seed_class_for_analysis = seed_analysis_states
            .iter()
            .copied()
            .enumerate()
            .fold(
                BTreeMap::<usize, usize>::new(),
                |mut map, (seed_class, analysis_state)| {
                    map.entry(analysis_state).or_insert(seed_class);
                    map
                },
            );
        let final_seed_class_representatives = seed_analysis_states
            .iter()
            .map(|analysis_state| {
                let exact_analysis_state = analysis_to_exact[analysis_state];
                first_seed_class_for_analysis[&exact_analysis_state]
            })
            .collect::<Vec<_>>();
        // C is directional: each seed representative extends every member's
        // positional observations. Exact powerset refinement may merge C seed
        // classes, but the resulting class must retain the chosen C
        // representative. Picking the first raw member, as ordinary
        // equivalence composition does, can replace the representative with a
        // strictly less-defined member and lose terminal-NWA behavior.
        compose_raw_quotient_state_map_preserving_directional_representatives(
            seed,
            &final_seed_class_representatives,
        )
    } else {
        build_state_map_from_subset_representatives(
            &seed_states,
            &state_representatives,
            num_states,
            Some(seed),
        )
    };
    let mut final_state_representatives = if bounded_analysis.is_some() {
        // Exact refinement above is already expressed in bounded powerset-view
        // coordinates. The composed raw quotient deliberately chooses the first
        // raw member of each final class as its stored representative; that raw
        // member need not itself be one of C's seeded token-start representatives.
        // Re-projecting it into the bounded view is therefore invalid. Use the
        // exact analysis representatives directly for the subsequent vocab pass.
        state_representatives.clone()
    } else {
        tokenizer_states
            .representative_original_ids
            .iter()
            .map(|&state| state as usize)
            .collect::<Vec<_>>()
    };
    final_state_representatives.sort_unstable();
    final_state_representatives.dedup();

    let vocab_equiv_started_at = Instant::now();
    let (dedup_vocab_classes, vocab_analysis_dfa_build_ms) =
        vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
            &tokenizer_view,
            &dedup.representative_token_bytes,
            &final_state_representatives,
            effective_disallowed,
            Some(&byte_to_class),
            None,
            Some(&local_vocab_dfa_cache),
            Some(&local_analysis_dfa_cache),
        );
    let vocab_classes = expand_vocab_classes(
        dedup_vocab_classes,
        &dedup.original_to_repr,
        dedup.representative_token_bytes.len(),
    );
    let vocab_equiv_ms = vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0;

    let finalize_started_at = Instant::now();
    let internal_id_map = InternalIdMap {
        tokenizer_states,
        vocab_tokens: build_vocab_map(&vocab_classes, &prepared.token_ids, prepared.max_token_id),
    };
    let id_map_finalize_ms = finalize_started_at.elapsed().as_secs_f64() * 1000.0;
    let exact_reps = internal_id_map.tokenizer_states.num_internal_ids() as usize;

    Some((
        internal_id_map,
        CombinedEquivalenceProfile {
            initial_states_considered: seed_states.len(),
            max_length_skipped: true,
            max_token_len,
            token_len_gt_4: token_len_stats.gt_4,
            token_len_gt_8: token_len_stats.gt_8,
            token_len_gt_16: token_len_stats.gt_16,
            token_len_gt_32: token_len_stats.gt_32,
            token_len_gt_64: token_len_stats.gt_64,
            raw_analysis_base_init_ms: 0.0,
            analysis_view_build_ms,
            active_mask_filter_ms: 0.0,
            effective_follows_normalize_ms,
            prepare_inputs_ms,
            byte_class_setup_ms,
            vocab_analysis_dfa_build_ms,
            token_dedup_ms,
            restricted_observation_state_equiv_ms: 0.0,
            max_length_state_equiv_ms: 0.0,
            vocab_equiv_ms,
            exact_state_equiv_ms,
            id_map_finalize_ms,
            restricted_observation_reps: seed_states.len(),
            max_length_reps: seed_states.len(),
            exact_reps,
            exact_rep_confirmation_used,
        },
    ))
}

fn try_analyze_equivalences_with_raw_quotient(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    effective_disallowed: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    active_groups: Option<&[bool]>,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    initial_state_map: Option<&ManyToOneIdMap>,
    effective_follows_prepare_ms: f64,
    precomputed_raw_observations: Option<(&[u32], &[u32])>,
) -> Option<(InternalIdMap, CombinedEquivalenceProfile)> {
    let active_groups = active_groups?;
    let flat_trans = flat_trans?;
    if flat_trans.len() != tokenizer.num_states() as usize * 256 {
        return None;
    }

    let prepare_inputs_started_at = Instant::now();
    let prepared = prepare_equivalence_inputs(tokenizer, vocab, initial_state_map);
    let token_len_stats = token_length_stats(&prepared.token_bytes);
    let prepare_inputs_ms = prepare_inputs_started_at.elapsed().as_secs_f64() * 1000.0;

    // The direct raw refinement sees every byte that can occur in this local
    // vocabulary. It is deliberately at least as fine as the later deduped
    // quotient analysis.
    let mut raw_relevant_bytes = [false; 256];
    let mut direct_token_bytes = 0usize;
    let mut max_token_len = 0usize;
    for token in &prepared.token_bytes {
        direct_token_bytes += token.len();
        max_token_len = max_token_len.max(token.len());
        for &byte in *token {
            raw_relevant_bytes[byte as usize] = true;
        }
    }
    let active_byte_count = raw_relevant_bytes.iter().filter(|&&active| active).count();
    let projected_by_global = prepared.initial_states.len() < tokenizer.num_states() as usize;
    let skip_raw_quotient = should_skip_max_length_for_partition(
        partition_label,
        prepared.initial_states.len(),
        projected_by_global,
        direct_token_bytes,
        max_token_len,
        active_byte_count,
    );
    // The generic route still performs restricted observation, but for a tiny
    // local vocabulary it then keeps the full raw analysis DFA alive for the
    // exact state and token phases.  Materializing the exact restricted raw
    // quotient is cheaper in that regime: it avoids the 18k-state shared-base
    // setup and lets both remaining phases operate on the quotient.  Keep the
    // bound deliberately narrow; larger short-token partitions (notably p7)
    // can retain too much raw topology for this to win.
    let tiny_raw_quotient = prepared.token_bytes.len() <= RAW_QUOTIENT_TINY_VOCAB_MAX_TOKENS
        && direct_token_bytes <= RAW_QUOTIENT_TINY_VOCAB_MAX_BYTES;
    let structural_boundary_raw_quotient = matches!(partition_label, "p7" | "p8")
        && prepared.token_bytes.len() <= RAW_QUOTIENT_STRUCTURAL_BOUNDARY_MAX_TOKENS
        && direct_token_bytes <= RAW_QUOTIENT_STRUCTURAL_BOUNDARY_MAX_BYTES;
    if skip_raw_quotient && !tiny_raw_quotient && !structural_boundary_raw_quotient {
        return None;
    }

    // P8's local vocabulary has one common leading quote byte. For this one
    // partition, the full behavior of a source state on every local token is
    // determined by its quote successor. Evaluate one source representative
    // per distinct successor, then lift the exact result back to all raw
    // sources. This is a conservative factorization: equal quote successors
    // have identical complete token paths by definition.
    if partition_label == "p8" && super::super::p8_first_byte_factorization_allowed() {
        let common_first = prepared
            .token_bytes
            .first()
            .and_then(|token| token.first())
            .copied()?;
        if prepared
            .token_bytes
            .iter()
            .any(|token| token.first().copied() != Some(common_first))
        {
            return None;
        }

        // Every production P8 token has a nonempty identifier prefix after
        // the quote. Model terminal matches at the quote successor as
        // zero-position events, then analyze only the remaining suffix.
        // A one-byte quote token invalidates that factorization: its acceptance
        // is observed at the quote successor itself, not after a nonempty
        // suffix. Let the generic exact route handle such atypical partitions.
        let use_seeded_suffix_factorization =
            prepared.token_bytes.iter().all(|token| token.len() > 1);
        if !use_seeded_suffix_factorization {
            return None;
        }
        let mut target_to_source = BTreeMap::<u32, usize>::new();
        let mut source_representative = vec![usize::MAX; tokenizer.num_states() as usize];
        let mut target_by_source = vec![u32::MAX; tokenizer.num_states() as usize];
        for source in 0..tokenizer.num_states() as usize {
            let target = flat_trans[source * 256 + common_first as usize];
            if target == u32::MAX {
                continue;
            }
            let representative = *target_to_source.entry(target).or_insert(source);
            source_representative[source] = representative;
            target_by_source[source] = target;
        }
        let representative_sources: Vec<usize> = target_to_source.values().copied().collect();
        let quote_targets: Vec<usize> = target_to_source.keys().map(|&target| target as usize).collect();
        let suffix_tokens: Vec<&[u8]> = prepared
            .token_bytes
            .iter()
            .map(|token| &token[1..])
            .collect();
        debug_assert!(suffix_tokens.iter().all(|suffix| !suffix.is_empty()));
        let view_started_at = Instant::now();
        let analysis_view = TokenizerView::new_filtered_from_flat_trans(
            flat_trans,
            tokenizer,
            active_groups,
        );
        let analysis_view_build_ms = view_started_at.elapsed().as_secs_f64() * 1000.0;
        let exact_started_at = Instant::now();
        let representative_state_reps = if use_seeded_suffix_factorization {
            state_equivalence_analysis::find_state_equivalence_classes_with_sparse_disallowed_and_raw_transitions_with_initial_finalizers(
                &analysis_view,
                &suffix_tokens,
                &quote_targets,
                effective_disallowed,
            )
        } else {
            state_equivalence_analysis::find_state_equivalence_classes_with_sparse_disallowed_and_raw_transitions(
                &analysis_view,
                &prepared.token_bytes,
                &representative_sources,
                effective_disallowed,
            )
        };
        let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
        let mut source_rep_to_behavior_rep = HashMap::<usize, usize>::new();
        if use_seeded_suffix_factorization {
            for (&target, &behavior_rep) in quote_targets.iter().zip(&representative_state_reps) {
                source_rep_to_behavior_rep.insert(target, behavior_rep);
            }
        } else {
            for (&source, &behavior_rep) in representative_sources
                .iter()
                .zip(&representative_state_reps)
            {
                source_rep_to_behavior_rep.insert(source, behavior_rep);
            }
        }

        let id_map_finalize_started_at = Instant::now();
        let mut behavior_rep_to_internal = HashMap::<Option<usize>, u32>::new();
        let mut original_to_internal = vec![u32::MAX; tokenizer.num_states() as usize];
        let mut internal_to_originals = Vec::<Vec<u32>>::new();
        let mut representative_original_ids = Vec::<u32>::new();
        for source in 0..tokenizer.num_states() as usize {
            let behavior_rep = if use_seeded_suffix_factorization {
                (target_by_source[source] != u32::MAX)
                    .then(|| source_rep_to_behavior_rep[&(target_by_source[source] as usize)])
            } else {
                (source_representative[source] != usize::MAX).then(|| {
                    source_rep_to_behavior_rep[&source_representative[source]]
                })
            };
            let next = internal_to_originals.len() as u32;
            let internal = *behavior_rep_to_internal.entry(behavior_rep).or_insert_with(|| {
                internal_to_originals.push(Vec::new());
                representative_original_ids.push(source as u32);
                next
            });
            original_to_internal[source] = internal;
            internal_to_originals[internal as usize].push(source as u32);
        }
        let tokenizer_states = ManyToOneIdMap {
            original_to_internal,
            internal_to_originals,
            representative_original_ids,
        };
        let exact_reps = tokenizer_states.num_internal_ids() as usize;
        let internal_id_map = InternalIdMap {
            tokenizer_states,
            vocab_tokens: build_identity_vocab_map(&prepared.token_ids, prepared.max_token_id),
        };
        let id_map_finalize_ms = id_map_finalize_started_at.elapsed().as_secs_f64() * 1000.0;
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][p8_first_byte_factorization] first_byte={} quote_targets={} seeded_suffix_factorized={} exact_reps={} view_ms={:.3} state_equiv_ms={:.3}",
                common_first,
                representative_sources.len(),
                use_seeded_suffix_factorization,
                exact_reps,
                analysis_view_build_ms,
                exact_state_equiv_ms,
            );
        }
        return Some((
            internal_id_map,
            CombinedEquivalenceProfile {
                initial_states_considered: prepared.initial_states.len(),
                max_length_skipped: false,
                max_token_len,
                token_len_gt_4: token_len_stats.gt_4,
                token_len_gt_8: token_len_stats.gt_8,
                token_len_gt_16: token_len_stats.gt_16,
                token_len_gt_32: token_len_stats.gt_32,
                token_len_gt_64: token_len_stats.gt_64,
                raw_analysis_base_init_ms: 0.0,
                analysis_view_build_ms,
                active_mask_filter_ms: 0.0,
                effective_follows_normalize_ms: effective_follows_prepare_ms,
                prepare_inputs_ms,
                byte_class_setup_ms: 0.0,
                vocab_analysis_dfa_build_ms: 0.0,
                token_dedup_ms: 0.0,
                restricted_observation_state_equiv_ms: 0.0,
                max_length_state_equiv_ms: 0.0,
                vocab_equiv_ms: 0.0,
                exact_state_equiv_ms,
                id_map_finalize_ms,
                restricted_observation_reps: representative_sources.len(),
                max_length_reps: representative_sources.len(),
                exact_reps,
                exact_rep_confirmation_used: false,
            },
        ));
    }

    let restricted_started_at = Instant::now();
    let raw_restricted = super::state_equivalence::restricted_observation::compute_state_map_raw(
        tokenizer,
        flat_trans,
        active_groups,
        &raw_relevant_bytes,
        precomputed_raw_observations,
    )?;
    let restricted_observation_state_equiv_ms =
        restricted_started_at.elapsed().as_secs_f64() * 1000.0;
    if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][raw_restricted_observation] partition={} labels_ms={:.3} refine_ms={:.3} certificate_ms={:.3} rounds={} reps={}",
            partition_label,
            raw_restricted.label_ms,
            raw_restricted.refine_ms,
            raw_restricted.certificate_ms,
            raw_restricted.rounds,
            raw_restricted.state_map.num_internal_ids(),
        );
    }
    let raw_observation_ids = raw_restricted.raw_observation_ids;
    let observation_representatives = raw_restricted.observation_representatives;
    let pre_state_map = raw_restricted.state_map;

    let use_compact_quotient_view = partition_label == "p7"
        && !super::super::l2p_terminal_interchangeability_strict_reference_enabled();
    let view_build_started_at = Instant::now();
    let (analysis_view, prebuilt_local_base, active_mask_filter_ms) = if use_compact_quotient_view {
        if let Some((view, base, build_ms)) =
            TokenizerView::new_filtered_quotient_from_flat_trans_with_observation_cache_and_relevant_base(
                flat_trans,
                tokenizer,
                active_groups,
                &pre_state_map,
                &raw_observation_ids,
                &observation_representatives,
                &raw_relevant_bytes,
            )
        {
            (view, Some(base), build_ms)
        } else {
            let (view, build_ms) =
                TokenizerView::new_filtered_quotient_from_flat_trans_with_observation_cache(
                    flat_trans,
                    tokenizer,
                    active_groups,
                    &pre_state_map,
                    &raw_observation_ids,
                    &observation_representatives,
                );
            (view, None, build_ms)
        }
    } else {
        let (view, build_ms) =
            TokenizerView::new_filtered_quotient_from_flat_trans_with_observation_cache(
                flat_trans,
                tokenizer,
                active_groups,
                &pre_state_map,
                &raw_observation_ids,
                &observation_representatives,
            );
        (view, None, build_ms)
    };
    let analysis_view_build_ms = view_build_started_at.elapsed().as_secs_f64() * 1000.0;

    // This exact byte partition is now over at most a few hundred quotient
    // rows. It replaces the previous 18k-state shared-base cold setup.
    let byte_class_setup_started_at = Instant::now();
    let local_shared_base = prebuilt_local_base.or_else(|| {
        super::vocab::fast::SharedVocabDfaBase::build_from_dfa_for_relevant_bytes(
            analysis_view.dfa(),
            &raw_relevant_bytes,
        )
    });
    let byte_to_class = local_shared_base
        .as_ref()
        .map(|base| base.byte_to_class())
        .unwrap_or_else(|| super::compat::compute_byte_classes(analysis_view.dfa()));
    let byte_class_setup_ms = byte_class_setup_started_at.elapsed().as_secs_f64() * 1000.0;

    let token_dedup_started_at = Instant::now();
    let dedup = deduplicate_tokens_by_byte_class(&prepared.token_bytes, &byte_to_class);
    let token_dedup_ms = token_dedup_started_at.elapsed().as_secs_f64() * 1000.0;

    // In strict-reference mode the primary build retains the old cloned dense
    // representation. The recursive reference runs with the guard suppressed
    // and therefore exercises the production borrowed/sparse form, so the
    // existing terminal-DWA equality check crosses the representations.
    let strict_reference_mode =
        super::super::l2p_terminal_interchangeability_strict_reference_enabled();
    let use_sparse_follow_rows = partition_label == "p8"
        && !strict_reference_mode;
    let use_borrowed_follow_rows = partition_label == "p7" && !strict_reference_mode;
    let follows_normalize_started_at = Instant::now();
    let group_count = tokenizer_group_count(&analysis_view);
    let normalized_disallowed_follows = (!use_sparse_follow_rows && !use_borrowed_follow_rows).then(|| {
        normalize_disallowed_follows(group_count, effective_disallowed)
    });
    let borrowed_disallowed_follows = use_borrowed_follow_rows.then(|| {
        (0..group_count)
            .map(|terminal| {
                effective_disallowed
                    .get(&(terminal as u32))
                    .filter(|bits| !bits.is_zero())
            })
            .collect::<Vec<_>>()
    });
    let effective_follows_normalize_ms = if use_sparse_follow_rows {
        effective_follows_prepare_ms
    } else {
        effective_follows_prepare_ms
            + follows_normalize_started_at.elapsed().as_secs_f64() * 1000.0
    };
    let pre_reduced_states: Vec<usize> = (0..pre_state_map.internal_to_originals.len()).collect();
    let run_state_refinement = |tokens: &[&[u8]]| {
        let exact_rep_confirmation_used = pre_reduced_states.len()
            >= EXACT_REP_CONFIRMATION_MIN_STATES
            && tokens.len() >= EXACT_REP_CONFIRMATION_MIN_TOKENS;
        let representatives = if use_sparse_follow_rows {
            if exact_rep_confirmation_used {
                state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_sparse_disallowed_and_shared_base(
                    &analysis_view,
                    tokens,
                    &pre_reduced_states,
                    effective_disallowed,
                    None,
                    None,
                    Some(true),
                    local_shared_base.as_ref(),
                )
            } else {
                state_equivalence_analysis::find_state_equivalence_classes_with_sparse_disallowed_and_shared_base(
                    &analysis_view,
                    tokens,
                    &pre_reduced_states,
                    effective_disallowed,
                    local_shared_base.as_ref(),
                )
            }
        } else if use_borrowed_follow_rows && exact_rep_confirmation_used {
            state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_borrowed_disallowed_and_shared_base(
                &analysis_view,
                tokens,
                &pre_reduced_states,
                borrowed_disallowed_follows
                    .as_deref()
                    .expect("borrowed follow rows must be present"),
                None,
                None,
                Some(true),
                local_shared_base.as_ref(),
            )
        } else if use_borrowed_follow_rows {
            state_equivalence_analysis::find_state_equivalence_classes_with_borrowed_disallowed_and_shared_base(
                &analysis_view,
                tokens,
                &pre_reduced_states,
                borrowed_disallowed_follows
                    .as_deref()
                    .expect("borrowed follow rows must be present"),
                local_shared_base.as_ref(),
            )
        } else if exact_rep_confirmation_used {
            state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base(
                &analysis_view,
                tokens,
                &pre_reduced_states,
                normalized_disallowed_follows
                    .as_deref()
                    .expect("dense follow rows must be present"),
                None,
                None,
                Some(true),
                local_shared_base.as_ref(),
            )
        } else {
            state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base(
                &analysis_view,
                tokens,
                &pre_reduced_states,
                normalized_disallowed_follows
                    .as_deref()
                    .expect("dense follow rows must be present"),
                local_shared_base.as_ref(),
            )
        };
        (representatives, exact_rep_confirmation_used)
    };

    let vocab_first = !matches!(partition_label, "p7" | "p8")
        && dedup.representative_token_bytes.len() >= 512
        && pre_reduced_states.len() >= 256;
    if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][raw_quotient_equivalence_order] partition={} dedup_tokens={} pre_states={} vocab_first={}",
            partition_label,
            dedup.representative_token_bytes.len(),
            pre_reduced_states.len(),
            vocab_first,
        );
    }

    let (
        reduced_state_reps_for_pre_reduced,
        exact_rep_confirmation_used,
        precomputed_vocab,
        exact_state_equiv_ms,
        vocab_equiv_ms,
    ) = if vocab_first {
        let vocab_equiv_started_at = Instant::now();
        let precomputed_vocab =
            vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                &analysis_view,
                &dedup.representative_token_bytes,
                &pre_reduced_states,
                effective_disallowed,
                Some(&byte_to_class),
                None,
                None,
                None,
            );
        let vocab_equiv_ms = vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0;
        let representative_tokens = representative_tokens_for_vocab_classes(
            &precomputed_vocab.0,
            &dedup.representative_token_bytes,
        );
        let exact_started_at = Instant::now();
        let (state_reps, exact_rep_confirmation_used) =
            run_state_refinement(&representative_tokens);
        let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
        (
            state_reps,
            exact_rep_confirmation_used,
            Some(precomputed_vocab),
            exact_state_equiv_ms,
            vocab_equiv_ms,
        )
    } else {
        let exact_started_at = Instant::now();
        let (state_reps, exact_rep_confirmation_used) =
            run_state_refinement(&dedup.representative_token_bytes);
        let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
        (
            state_reps,
            exact_rep_confirmation_used,
            None,
            exact_state_equiv_ms,
            0.0,
        )
    };

    let mut final_state_representatives = reduced_state_reps_for_pre_reduced.clone();
    final_state_representatives.sort_unstable();
    final_state_representatives.dedup();

    // P8 is the structural-boundary route. Its local source vocabulary is
    // deliberately tiny, while the generic token quotient can cost more than
    // the entire NWA/DWA construction and frequently leaves every local token
    // distinct. Retaining the identity token map is exact by construction: it
    // only declines a possible compression, and leaves the independently exact
    // scanner-state quotient unchanged.
    if matches!(partition_label, "p7" | "p8") {
        let id_map_finalize_started_at = Instant::now();
        let tokenizer_states = compose_raw_quotient_state_map(
            &pre_state_map,
            &reduced_state_reps_for_pre_reduced,
        );
        let exact_reps = tokenizer_states.num_internal_ids() as usize;
        let internal_id_map = InternalIdMap {
            tokenizer_states,
            vocab_tokens: build_identity_vocab_map(&prepared.token_ids, prepared.max_token_id),
        };
        let id_map_finalize_ms = id_map_finalize_started_at.elapsed().as_secs_f64() * 1000.0;
        return Some((
            internal_id_map,
            CombinedEquivalenceProfile {
                initial_states_considered: prepared.initial_states.len(),
                max_length_skipped: false,
                max_token_len,
                token_len_gt_4: token_len_stats.gt_4,
                token_len_gt_8: token_len_stats.gt_8,
                token_len_gt_16: token_len_stats.gt_16,
                token_len_gt_32: token_len_stats.gt_32,
                token_len_gt_64: token_len_stats.gt_64,
                raw_analysis_base_init_ms: 0.0,
                analysis_view_build_ms,
                active_mask_filter_ms,
                effective_follows_normalize_ms,
                prepare_inputs_ms,
                byte_class_setup_ms,
                vocab_analysis_dfa_build_ms: 0.0,
                token_dedup_ms,
                restricted_observation_state_equiv_ms,
                max_length_state_equiv_ms: 0.0,
                vocab_equiv_ms: 0.0,
                exact_state_equiv_ms,
                id_map_finalize_ms,
                restricted_observation_reps: pre_state_map.num_internal_ids() as usize,
                max_length_reps: pre_state_map.num_internal_ids() as usize,
                exact_reps,
                exact_rep_confirmation_used,
            },
        ));
    }

    let (dedup_vocab_classes, vocab_analysis_dfa_build_ms, vocab_equiv_ms) =
        if let Some((classes, build_ms)) = precomputed_vocab {
            (classes, build_ms, vocab_equiv_ms)
        } else {
            let vocab_equiv_started_at = Instant::now();
            let (classes, build_ms) =
                vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                    &analysis_view,
                    &dedup.representative_token_bytes,
                    &final_state_representatives,
                    effective_disallowed,
                    Some(&byte_to_class),
                    None,
                    None,
                    None,
                );
            (
                classes,
                build_ms,
                vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0,
            )
        };

    let id_map_finalize_started_at = Instant::now();
    let vocab_classes = expand_vocab_classes(
        dedup_vocab_classes,
        &dedup.original_to_repr,
        dedup.representative_token_bytes.len(),
    );
    let tokenizer_states = compose_raw_quotient_state_map(
        &pre_state_map,
        &reduced_state_reps_for_pre_reduced,
    );
    let exact_reps = tokenizer_states.num_internal_ids() as usize;
    let internal_id_map = InternalIdMap {
        tokenizer_states,
        vocab_tokens: build_vocab_map(&vocab_classes, &prepared.token_ids, prepared.max_token_id),
    };
    let id_map_finalize_ms = id_map_finalize_started_at.elapsed().as_secs_f64() * 1000.0;

    Some((
        internal_id_map,
        CombinedEquivalenceProfile {
            initial_states_considered: prepared.initial_states.len(),
            max_length_skipped: false,
            max_token_len,
            token_len_gt_4: token_len_stats.gt_4,
            token_len_gt_8: token_len_stats.gt_8,
            token_len_gt_16: token_len_stats.gt_16,
            token_len_gt_32: token_len_stats.gt_32,
            token_len_gt_64: token_len_stats.gt_64,
            raw_analysis_base_init_ms: 0.0,
            analysis_view_build_ms,
            active_mask_filter_ms,
            effective_follows_normalize_ms,
            prepare_inputs_ms,
            byte_class_setup_ms,
            vocab_analysis_dfa_build_ms,
            token_dedup_ms,
            restricted_observation_state_equiv_ms,
            max_length_state_equiv_ms: 0.0,
            vocab_equiv_ms,
            exact_state_equiv_ms,
            id_map_finalize_ms,
            restricted_observation_reps: pre_state_map.num_internal_ids() as usize,
            max_length_reps: pre_state_map.num_internal_ids() as usize,
            exact_reps,
            exact_rep_confirmation_used,
        },
    ))
}

pub(crate) fn analyze_equivalences_with_group_filter(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    disallowed_follows_are_ignore_transparent: bool,
    pre_normalized_disallowed_follows: Option<&[BitSet]>,
    active_groups: Option<&[bool]>,
    shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
    shared_analysis_dfa_cache: Option<&super::vocab::fast::SharedVocabAnalysisDfaCache>,
    shared_base_setup_ms: f64,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    initial_state_map: Option<&ManyToOneIdMap>,
    token_position_partition: Option<&GlobalTokenPositionStatePartition>,
    precomputed_raw_observations: Option<(&[u32], &[u32])>,
) -> (InternalIdMap, CombinedEquivalenceProfile) {
    analyze_equivalences_impl(
        partition_label,
        tokenizer,
        vocab,
        disallowed_follows,
        ignore_terminal,
        disallowed_follows_are_ignore_transparent,
        pre_normalized_disallowed_follows,
        active_groups,
        shared_vocab_dfa_cache,
        shared_analysis_dfa_cache,
        shared_base_setup_ms,
        flat_trans,
        initial_state_map,
        token_position_partition,
        precomputed_raw_observations,
    )
}

/// Combined equivalence analysis over a flattened tokenizer DFA.
///
/// Uses state equivalence (k-step hashing plus token-based refinement) followed
/// by vocab equivalence (parallel batched with byte-class compression).
fn analyze_equivalences_impl(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    disallowed_follows_are_ignore_transparent: bool,
    pre_normalized_disallowed_follows: Option<&[BitSet]>,
    active_groups: Option<&[bool]>,
    shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
    shared_analysis_dfa_cache: Option<&super::vocab::fast::SharedVocabAnalysisDfaCache>,
    shared_base_setup_ms: f64,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    initial_state_map: Option<&ManyToOneIdMap>,
    token_position_partition: Option<&GlobalTokenPositionStatePartition>,
    precomputed_raw_observations: Option<(&[u32], &[u32])>,
) -> (InternalIdMap, CombinedEquivalenceProfile) {
    if tokenizer.has_epsilon_transitions() {
        let follows_prepare_started_at = Instant::now();
        let token_path_disallowed_follows = (!disallowed_follows_are_ignore_transparent)
            .then(|| ignore_transparent_disallowed_follows(disallowed_follows, ignore_terminal));
        let effective_follows_prepare_ms =
            follows_prepare_started_at.elapsed().as_secs_f64() * 1000.0;
        let effective_disallowed = token_path_disallowed_follows
            .as_ref()
            .unwrap_or(disallowed_follows);

        // C is valid at LLM-token start boundaries. The C-seeded route performs
        // only whole-token analysis, so it is sound to use before the absolute
        // restricted-observation pipeline. Epsilon lexers use a bounded
        // configuration view inside that route and project back to raw TSIDs.
        if let Some(token_position_partition) = token_position_partition {
            if let Some(result) = try_analyze_equivalences_with_token_position_partition(
                partition_label,
                tokenizer,
                vocab,
                effective_disallowed,
                active_groups,
                shared_vocab_dfa_cache,
                shared_analysis_dfa_cache,
                flat_trans,
                token_position_partition,
                effective_follows_prepare_ms,
                pre_normalized_disallowed_follows,
            ) {
                return result;
            }
        }

        let prepare_started_at = Instant::now();
        let prepared = prepare_equivalence_inputs(tokenizer, vocab, initial_state_map);
        let token_len_stats = token_length_stats(&prepared.token_bytes);
        let prepare_inputs_ms = prepare_started_at.elapsed().as_secs_f64() * 1000.0;

        let token_dedup_started_at = Instant::now();
        let identity_byte_class: [u8; 256] = std::array::from_fn(|byte| byte as u8);
        let identity_dedup =
            deduplicate_tokens_by_byte_class(&prepared.token_bytes, &identity_byte_class);
        let token_dedup_ms = token_dedup_started_at.elapsed().as_secs_f64() * 1000.0;
        let max_token_len = identity_dedup
            .representative_token_bytes
            .iter()
            .map(|token| token.len())
            .max()
            .unwrap_or(0);
        let mut relevant_bytes = [false; 256];
        for token in &identity_dedup.representative_token_bytes {
            for &byte in *token {
                relevant_bytes[byte as usize] = true;
            }
        }
        let active_language_byte_classes = (partition_label == "p2"
            && (std::env::var_os("GLRMASK_L2P_ACTIVE_LANGUAGE_BYTE_DEDUP").is_some()
                || std::env::var_os("GLRMASK_L2P_ACTIVE_LANGUAGE_PREPASS_BYTES").is_some()))
        .then(|| {
            active_groups
                .and_then(|groups| active_terminal_language_byte_classes(tokenizer, groups))
        })
        .flatten();
        let direct_token_bytes = identity_dedup
            .representative_token_bytes
            .iter()
            .map(|token| token.len())
            .sum();
        let active_byte_count = relevant_bytes.iter().filter(|&&active| active).count();
        let projected_by_global = prepared.initial_states.len() < tokenizer.num_states() as usize;
        let max_length_skipped = should_skip_max_length_for_partition(
            partition_label,
            prepared.initial_states.len(),
            projected_by_global,
            direct_token_bytes,
            max_token_len,
            active_byte_count,
        );
        let analysis_view_policy = l2p_nfa_analysis_view_policy();
        let powerset_max_states = l2p_nfa_relevant_powerset_max_states();
        let powerset_min_bounded_pairs = l2p_nfa_relevant_powerset_min_bounded_pairs();
        let prepass_pair_estimate = prepared
            .initial_states
            .len()
            .saturating_mul(identity_dedup.representative_token_bytes.len());
        let should_probe_powerset = should_probe_l2p_nfa_powerset(
            analysis_view_policy,
            prepass_pair_estimate,
            powerset_min_bounded_pairs,
        );
        let powerset_started_at = Instant::now();
        let powerset_candidate = should_probe_powerset.then(|| {
            build_relevant_powerset_view(tokenizer, &relevant_bytes, active_groups, None)
        });
        let powerset_build_ms = powerset_started_at.elapsed().as_secs_f64() * 1000.0;
        let powerset_output_classes = powerset_candidate.as_ref().map(|powerset| {
            let mut output_ids = HashMap::<(Vec<usize>, Vec<usize>), u32>::new();
            powerset
                .states
                .iter()
                .map(|state| {
                    let key = (
                        state.finalizers.clone(),
                        state.possible_future_group_ids.clone(),
                    );
                    let next = output_ids.len() as u32;
                    *output_ids.entry(key).or_insert(next)
                })
                .collect::<Vec<_>>()
        });
        let projected_prepass_edges = powerset_candidate.as_ref().and_then(|powerset| {
            (partition_label == "p2"
                && std::env::var_os("GLRMASK_L2P_PRUNE_INACTIVE_PREPASS").is_some())
            .then(|| {
                projected_sparse_prepass_edges(
                    powerset,
                    &relevant_bytes,
                    std::env::var_os("GLRMASK_L2P_ACTIVE_LANGUAGE_PREPASS_BYTES")
                        .is_some()
                        .then_some(active_language_byte_classes.as_ref())
                        .flatten(),
                )
            })
        });
        let prebuilt_nfa_refinement = powerset_candidate
            .as_ref()
            .zip(powerset_output_classes.as_ref())
            .map(|(powerset, output_class_by_config)| PrebuiltSparsePowersetRefinement {
                raw_start_to_view: powerset.raw_start_to_view.as_ref(),
                configurations: powerset.configurations.as_ref(),
                output_class_by_config,
                edge_offsets: projected_prepass_edges
                    .as_ref()
                    .map_or(powerset.edge_offsets.as_slice(), |(offsets, _)| offsets.as_slice()),
                edges: projected_prepass_edges
                    .as_ref()
                    .map_or(powerset.edges.as_slice(), |(_, edges)| edges.as_slice()),
            });
        if partition_label == "p2"
            && std::env::var_os("GLRMASK_PROFILE_DIRECT_BOUNDED_PREPASS").is_some()
            && let Some(prebuilt) = prebuilt_nfa_refinement.as_ref()
        {
            let direct_started_at = Instant::now();
            let direct = prebuilt.compute_state_map(
                tokenizer,
                initial_state_map,
                super::state_equivalence::nfa::RefinementDepth::Bounded(max_token_len),
            );
            eprintln!(
                "[glrmask/profile][direct_bounded_prepass] partition={} max_token_len={} reps={} ms={:.3}",
                partition_label,
                max_token_len,
                direct.num_internal_ids(),
                direct_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        let pipeline_config = resolve_l2p_pipeline_config(!max_length_skipped);
        let (tokenizer_states, pipeline_profile) = run_state_equivalence_pipeline(
            tokenizer,
            vocab,
            initial_state_map,
            active_groups,
            StateEquivalenceScope::L2p,
            &pipeline_config,
            prebuilt_nfa_refinement.as_ref(),
            None,
            None,
        );
        if partition_label == "p2"
            && std::env::var_os("GLRMASK_PROFILE_RESTRICTED_ACTIVE_MASK_COMPARE").is_some()
        {
            let compare_started_at = Instant::now();
            let unfiltered = super::state_equivalence::nfa::compute_state_map(
                tokenizer,
                &relevant_bytes,
                None,
                initial_state_map,
                super::state_equivalence::nfa::RefinementDepth::Stable,
            );
            eprintln!(
                "[glrmask/profile][restricted_active_mask_compare] partition={} active_terminals={} active_reps={} all_terminal_reps={} all_terminal_ms={:.3}",
                partition_label,
                active_groups.map_or(tokenizer.num_terminals() as usize, |active| active.iter().filter(|&&value| value).count()),
                pipeline_profile.restricted_observation_reps,
                unfiltered.num_internal_ids(),
                compare_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }

        let raw_pre_representatives = tokenizer_states
            .representative_original_ids
            .iter()
            .map(|&state| state as usize)
            .collect::<Vec<_>>();
        let active_stable_state_map = if partition_label == "p2"
            && std::env::var_os("GLRMASK_L2P_ACTIVE_QUOTIENT_VIEW").is_some()
        {
            prebuilt_nfa_refinement.as_ref().map(|prebuilt| {
                prebuilt.compute_state_map(
                    tokenizer,
                    initial_state_map,
                    super::state_equivalence::nfa::RefinementDepth::Stable,
                )
            })
        } else {
            None
        };
        let analysis_view_started_at = Instant::now();
        let bounded_pair_estimate = raw_pre_representatives
            .len()
            .saturating_mul(identity_dedup.representative_token_bytes.len());
        let powerset_probed = powerset_candidate.is_some();
        let powerset_states = powerset_candidate
            .as_ref()
            .map_or(0, |powerset| powerset.states.len());
        let use_powerset = should_use_l2p_nfa_powerset(
            analysis_view_policy,
            powerset_probed,
            powerset_states,
            powerset_max_states,
        );
        let (analysis_view_owned, raw_start_to_view) = if use_powerset {
            let powerset = if let Some(state_map) = active_stable_state_map.as_ref() {
                build_relevant_powerset_view(
                    tokenizer,
                    &relevant_bytes,
                    active_groups,
                    Some(state_map),
                )
            } else {
                powerset_candidate
                    .expect("powerset analysis policy must build a powerset candidate")
            };
            let raw_start_to_view = Arc::clone(&powerset.raw_start_to_view);
            (powerset.into_tokenizer_view(), raw_start_to_view)
        } else {
            let bounded = build_bounded_analysis_view(
                tokenizer,
                &raw_pre_representatives,
                &identity_dedup.representative_token_bytes,
                active_groups,
            );
            let mut raw_start_to_view = vec![u32::MAX; tokenizer.num_states() as usize];
            for &raw in &raw_pre_representatives {
                raw_start_to_view[raw] = bounded.view_state_for_raw_start(raw) as u32;
            }
            (bounded.tokenizer_view, Arc::from(raw_start_to_view))
        };
        let analysis_view_build_ms = powerset_build_ms
            + analysis_view_started_at.elapsed().as_secs_f64() * 1000.0;
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][l2p_nfa_analysis_view] partition={} policy={} selected={} bounded_pair_estimate={} powerset_min_bounded_pairs={} powerset_probed={} powerset_states={} powerset_max_states={} powerset_build_ms={:.3} total_build_ms={:.3}",
                partition_label,
                analysis_view_policy.as_str(),
                if use_powerset { "powerset" } else { "bounded" },
                bounded_pair_estimate,
                powerset_min_bounded_pairs,
                powerset_probed,
                powerset_states,
                powerset_max_states,
                powerset_build_ms,
                analysis_view_build_ms,
            );
        }
        let analysis_view = &analysis_view_owned;

        let byte_class_started_at = Instant::now();
        let byte_to_class = if partition_label == "p2"
            && std::env::var_os("GLRMASK_L2P_LITERAL_ACTIVE_MINIMIZED_BYTE_CLASSES").is_some()
        {
            let (byte_to_class, minimized_states) =
                literal_active_finalizer_minimized_byte_classes(analysis_view, &relevant_bytes);
            if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
                eprintln!(
                    "[glrmask/profile][literal_active_minimized_byte_classes] partition={} view_states={} minimized_states={} byte_classes={}",
                    partition_label,
                    analysis_view.dfa().states.len(),
                    minimized_states,
                    byte_to_class.iter().copied().max().map_or(0, |class| class as usize + 1),
                );
            }
            byte_to_class
        } else if partition_label == "p2"
            && std::env::var_os("GLRMASK_L2P_ACTIVE_MINIMIZED_BYTE_CLASSES").is_some()
        {
            let active_minimized_state_map =
                super::state_equivalence::restricted_observation::compute_state_map(
                    analysis_view,
                    &relevant_bytes,
                    None,
                    None,
                    true,
                );
            if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
                eprintln!(
                    "[glrmask/profile][active_minimized_byte_classes] partition={} view_states={} minimized_states={}",
                    partition_label,
                    analysis_view.dfa().states.len(),
                    active_minimized_state_map.num_internal_ids(),
                );
            }
            byte_classes_on_state_quotient(analysis_view, &active_minimized_state_map)
        } else {
            super::compat::compute_byte_classes(analysis_view.dfa())
        };
        let byte_class_setup_ms = byte_class_started_at.elapsed().as_secs_f64() * 1000.0;

        // The epsilon analysis view is already filtered to this partition's
        // active terminals. Bytes in the same DFA byte class therefore induce
        // exactly the same state transition from every analysis state. Token
        // behaviour (including match positions and follow-aware suffixes) can
        // only depend on the resulting byte-class sequence, so collapse those
        // sequences before the expensive whole-vocabulary equivalence scan.
        let token_byte_classes = active_language_byte_classes
            .as_ref()
            .unwrap_or(&byte_to_class);
        let class_dedup = deduplicate_tokens_by_byte_class(
            &identity_dedup.representative_token_bytes,
            token_byte_classes,
        );
        if active_language_byte_classes.is_some()
            && std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some()
        {
            eprintln!(
                "[glrmask/profile][active_language_byte_dedup] partition={} identity_tokens={} canonical_tokens={} byte_classes={}",
                partition_label,
                identity_dedup.representative_token_bytes.len(),
                class_dedup.representative_token_bytes.len(),
                token_byte_classes
                    .iter()
                    .copied()
                    .max()
                    .map_or(0, |class| class as usize + 1),
            );
        }
        let original_to_class_repr = identity_dedup
            .original_to_repr
            .iter()
            .map(|&identity_repr| class_dedup.original_to_repr[identity_repr])
            .collect::<Vec<_>>();
        let dedup = TokenDedup {
            representative_token_bytes: class_dedup.representative_token_bytes,
            original_to_repr: original_to_class_repr,
        };

        if partition_label == "p2"
            && std::env::var_os("GLRMASK_PROFILE_ACTIVE_LANGUAGE_TOKEN_PREQUOTIENT").is_some()
            && let Some(active_groups) = active_groups
            && let Some((active_tokenizer, local_disallowed)) =
                active_terminal_language_tokenizer_and_follows(
                    tokenizer,
                    active_groups,
                    effective_disallowed,
                )
        {
            let active_view = TokenizerView::new(&active_tokenizer);
            let active_byte_classes = super::compat::compute_byte_classes(active_view.dfa());
            let active_dedup = deduplicate_tokens_by_byte_class(
                &identity_dedup.representative_token_bytes,
                &active_byte_classes,
            );
            let active_states = (0..active_view.dfa().states.len()).collect::<Vec<_>>();
            let started_at = Instant::now();
            let (active_classes, build_ms) =
                vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                    &active_view,
                    &active_dedup.representative_token_bytes,
                    &active_states,
                    &local_disallowed,
                    Some(&active_byte_classes),
                    None,
                    None,
                    None,
                );
            eprintln!(
                "[glrmask/profile][active_language_token_prequotient] partition={} identity_tokens={} byte_canonical_tokens={} action_classes={} active_states={} active_terminals={} build_ms={:.3} total_ms={:.3}",
                partition_label,
                identity_dedup.representative_token_bytes.len(),
                active_dedup.representative_token_bytes.len(),
                active_classes.len(),
                active_states.len(),
                active_groups.iter().filter(|&&active| active).count(),
                build_ms,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }

        let follows_normalize_started_at = Instant::now();
        let normalized_disallowed_follows =
            normalize_disallowed_follows(tokenizer_group_count(analysis_view), effective_disallowed);
        let effective_follows_normalize_ms = effective_follows_prepare_ms
            + follows_normalize_started_at.elapsed().as_secs_f64() * 1000.0;

        let preclass_view_states = raw_pre_representatives
            .iter()
            .map(|&raw| raw_start_to_view[raw] as usize)
            .collect::<Vec<_>>();
        let mut query_view_states = preclass_view_states.clone();
        query_view_states.sort_unstable();
        query_view_states.dedup();
        let force_state_first = std::env::var_os("GLRMASK_L2P_FORCE_STATE_FIRST").is_some();
        let vocab_first = !force_state_first
            && dedup.representative_token_bytes.len() >= 512
            && query_view_states.len() >= 256;
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][epsilon_equivalence_order] partition={} dedup_tokens={} pre_states={} vocab_first={}",
                partition_label,
                dedup.representative_token_bytes.len(),
                query_view_states.len(),
                vocab_first,
            );
        }
        let (precomputed_vocab, state_tokens, vocab_equiv_ms) = if vocab_first {
            let vocab_equiv_started_at = Instant::now();
            let precomputed_vocab =
                vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                    analysis_view,
                    &dedup.representative_token_bytes,
                    &query_view_states,
                    effective_disallowed,
                    Some(&byte_to_class),
                    None,
                    None,
                    None,
                );
            let vocab_equiv_ms = vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0;
            let state_tokens = representative_tokens_for_vocab_classes(
                &precomputed_vocab.0,
                &dedup.representative_token_bytes,
            );
            (Some(precomputed_vocab), state_tokens, vocab_equiv_ms)
        } else {
            (None, dedup.representative_token_bytes.clone(), 0.0)
        };
        let exact_rep_confirmation_used = query_view_states.len() >= EXACT_REP_CONFIRMATION_MIN_STATES
            && state_tokens.len() >= EXACT_REP_CONFIRMATION_MIN_TOKENS;
        let exact_started_at = Instant::now();
        let query_representatives = if exact_rep_confirmation_used {
            state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base(
                analysis_view,
                &state_tokens,
                &query_view_states,
                &normalized_disallowed_follows,
                None,
                None,
                Some(true),
                None,
            )
        } else {
            state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base(
                analysis_view,
                &state_tokens,
                &query_view_states,
                &normalized_disallowed_follows,
                None,
            )
        };
        let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;

        let view_to_exact_view = query_view_states
            .iter()
            .copied()
            .zip(query_representatives.iter().copied())
            .collect::<BTreeMap<_, _>>();
        let first_preclass_for_view = preclass_view_states
            .iter()
            .copied()
            .enumerate()
            .fold(BTreeMap::<usize, usize>::new(), |mut map, (preclass, view_state)| {
                map.entry(view_state).or_insert(preclass);
                map
            });
        let final_preclass_representatives = preclass_view_states
            .iter()
            .map(|view_state| {
                let exact_view = view_to_exact_view[view_state];
                first_preclass_for_view[&exact_view]
            })
            .collect::<Vec<_>>();
        let tokenizer_states =
            compose_raw_quotient_state_map(&tokenizer_states, &final_preclass_representatives);

        let mut final_view_states = tokenizer_states
            .representative_original_ids
            .iter()
            .map(|&raw| raw_start_to_view[raw as usize] as usize)
            .collect::<Vec<_>>();
        final_view_states.sort_unstable();
        final_view_states.dedup();

        let (dedup_vocab_classes, vocab_analysis_dfa_build_ms, vocab_equiv_ms) =
            if let Some((classes, build_ms)) = precomputed_vocab {
                (classes, build_ms, vocab_equiv_ms)
            } else {
                let vocab_equiv_started_at = Instant::now();
                let (classes, build_ms) =
                    vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                        analysis_view,
                        &dedup.representative_token_bytes,
                        &final_view_states,
                        effective_disallowed,
                        Some(&byte_to_class),
                        None,
                        None,
                        None,
                    );
                (
                    classes,
                    build_ms,
                    vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0,
                )
            };

        let id_map_finalize_started_at = Instant::now();
        let vocab_classes = expand_vocab_classes(
            dedup_vocab_classes,
            &dedup.original_to_repr,
            dedup.representative_token_bytes.len(),
        );
        let vocab_tokens = build_vocab_map(
            &vocab_classes,
            &prepared.token_ids,
            prepared.max_token_id,
        );
        let id_map = InternalIdMap {
            tokenizer_states,
            vocab_tokens,
        };
        let id_map_finalize_ms = id_map_finalize_started_at.elapsed().as_secs_f64() * 1000.0;
        let exact_reps = id_map.tokenizer_states.num_internal_ids() as usize;
        return (
            id_map,
            CombinedEquivalenceProfile {
                initial_states_considered: prepared.initial_states.len(),
                max_length_skipped: pipeline_profile.max_length_skipped,
                max_token_len,
                token_len_gt_4: token_len_stats.gt_4,
                token_len_gt_8: token_len_stats.gt_8,
                token_len_gt_16: token_len_stats.gt_16,
                token_len_gt_32: token_len_stats.gt_32,
                token_len_gt_64: token_len_stats.gt_64,
                raw_analysis_base_init_ms: 0.0,
                analysis_view_build_ms,
                active_mask_filter_ms: 0.0,
                effective_follows_normalize_ms,
                prepare_inputs_ms,
                byte_class_setup_ms,
                vocab_analysis_dfa_build_ms,
                token_dedup_ms,
                restricted_observation_state_equiv_ms: pipeline_profile
                    .restricted_observation_state_equiv_ms,
                max_length_state_equiv_ms: pipeline_profile.max_length_state_equiv_ms,
                vocab_equiv_ms,
                exact_state_equiv_ms,
                id_map_finalize_ms,
                restricted_observation_reps: pipeline_profile.restricted_observation_reps,
                max_length_reps: pipeline_profile.max_length_reps,
                exact_reps,
                exact_rep_confirmation_used,
            },
        );
    }

    let follows_prepare_started_at = Instant::now();
    let token_path_disallowed_follows = (!disallowed_follows_are_ignore_transparent)
        .then(|| ignore_transparent_disallowed_follows(disallowed_follows, ignore_terminal));
    let effective_follows_prepare_ms = follows_prepare_started_at.elapsed().as_secs_f64() * 1000.0;
    let effective_disallowed = token_path_disallowed_follows
        .as_ref()
        .unwrap_or(disallowed_follows);
    if let Some(token_position_partition) = token_position_partition {
        if let Some(result) = try_analyze_equivalences_with_token_position_partition(
            partition_label,
            tokenizer,
            vocab,
            effective_disallowed,
            active_groups,
            shared_vocab_dfa_cache,
            shared_analysis_dfa_cache,
            flat_trans,
            token_position_partition,
            effective_follows_prepare_ms,
            pre_normalized_disallowed_follows,
        ) {
            return result;
        }
    }
    if let Some(result) = try_analyze_equivalences_with_raw_quotient(
        partition_label,
        tokenizer,
        vocab,
        effective_disallowed,
        ignore_terminal,
        active_groups,
        flat_trans,
        initial_state_map,
        effective_follows_prepare_ms,
        precomputed_raw_observations,
    ) {
        return result;
    }
    // The raw tokenizer is the only lexer coordinate in L2P. Retain the
    // compatibility check defensively, since this cache is shared by callers.
    let compatible_flat_trans = flat_trans.filter(|ft| {
        ft.len() == tokenizer.num_states() as usize * 256
    });
    let analysis_view_build_started_at = Instant::now();
    let tokenizer_view = match (active_groups, compatible_flat_trans) {
        (Some(active_groups), Some(ft)) => TokenizerView::new_filtered_from_flat_trans(ft, tokenizer, active_groups),
        (Some(active_groups), None) => TokenizerView::new_filtered(tokenizer, active_groups),
        (None, Some(ft)) => TokenizerView::new_from_flat_trans(ft, tokenizer),
        _ => TokenizerView::new(tokenizer),
    };
    let analysis_view_build_ms = analysis_view_build_started_at.elapsed().as_secs_f64() * 1000.0;

    let prepare_inputs_started_at = Instant::now();
    let prepared = prepare_equivalence_inputs(tokenizer, vocab, initial_state_map);
    let token_len_stats = token_length_stats(&prepared.token_bytes);
    let prepare_inputs_ms = prepare_inputs_started_at.elapsed().as_secs_f64() * 1000.0;

    let raw_analysis_base_started_at = Instant::now();
    if let Some(cache) = shared_vocab_dfa_cache {
        cache.get_or_init(|| vocab_equivalence_analysis::SharedVocabDfaBase::build_from_dfa(tokenizer_view.dfa()));
    }
    let raw_analysis_base_init_ms = shared_base_setup_ms
        + raw_analysis_base_started_at.elapsed().as_secs_f64() * 1000.0;

    let byte_class_setup_started_at = Instant::now();
    let compatible_cache = shared_vocab_dfa_cache
        .and_then(|cache| cache.get())
        .filter(|base| base.is_compatible_with_dfa(tokenizer_view.dfa()));
    let byte_to_class = compatible_cache
        .map(|base| base.byte_to_class())
        .unwrap_or_else(|| super::compat::compute_byte_classes(tokenizer_view.dfa()));
    let byte_class_setup_ms = byte_class_setup_started_at.elapsed().as_secs_f64() * 1000.0;

    let token_dedup_started_at = Instant::now();
    let dedup = deduplicate_tokens_by_byte_class(&prepared.token_bytes, &byte_to_class);
    let token_dedup_ms = token_dedup_started_at.elapsed().as_secs_f64() * 1000.0;
    let max_token_len = dedup
        .representative_token_bytes
        .iter()
        .map(|token| token.len())
        .max()
        .unwrap_or(0);
    let mut relevant_bytes = [false; 256];
    for token in &dedup.representative_token_bytes {
        for &byte in *token {
            relevant_bytes[byte as usize] = true;
        }
    }
    let direct_token_bytes: usize = dedup
        .representative_token_bytes
        .iter()
        .map(|token| token.len())
        .sum();
    let active_byte_count = relevant_bytes.iter().filter(|&&active| active).count();
    let projected_by_global = prepared.initial_states.len() < tokenizer.num_states() as usize;
    let max_length_skipped = should_skip_max_length_for_partition(
        partition_label,
        prepared.initial_states.len(),
        projected_by_global,
        direct_token_bytes,
        max_token_len,
        active_byte_count,
    );
    let pipeline_config = resolve_l2p_pipeline_config(!max_length_skipped);
    let (pre_state_map, pipeline_profile) = run_state_equivalence_pipeline(
        tokenizer,
        vocab,
        initial_state_map,
        active_groups,
        StateEquivalenceScope::L2p,
        &pipeline_config,
        None,
        Some(&tokenizer_view),
        Some(&byte_to_class),
    );
    // Restricted observation is a fixed point over the vocabulary byte
    // alphabet. When its map is also output-labelled congruent, token paths can
    // be evaluated exactly on the quotient instead of reinitializing 18k raw
    // lexer states for the exact and vocabulary phases.
    let analysis_quotient = tokenizer_view
        .is_relevant_byte_congruent(&pre_state_map, &relevant_bytes)
        .then(|| tokenizer_view.quotient_by_state_map(&pre_state_map));
    let uses_analysis_quotient = analysis_quotient.is_some();
    let analysis_view = analysis_quotient.as_ref().unwrap_or(&tokenizer_view);
    let pre_reduced_states: Vec<usize> = if uses_analysis_quotient {
        (0..pre_state_map.internal_to_originals.len()).collect()
    } else {
        pre_state_map
            .representative_original_ids
            .iter()
            .map(|&state| state as usize)
            .collect()
    };
    let follows_normalize_started_at = Instant::now();
    let normalized_disallowed_follows =
        normalize_disallowed_follows(tokenizer_group_count(analysis_view), effective_disallowed);
    let effective_follows_normalize_ms = effective_follows_prepare_ms
        + follows_normalize_started_at.elapsed().as_secs_f64() * 1000.0;

    // State and vocabulary equivalence are commuting exact quotients.  For a
    // large vocabulary, classify tokens on the pre-state quotient first, then
    // refine states using one representative token per exact token class.  The
    // final state partition and final token partition are identical to the
    // historical state-then-vocab order (see the commutativity regression
    // below), but the expensive state trellis sees hundreds of tokens instead
    // of tens of thousands.
    let vocab_first = dedup.representative_token_bytes.len() >= 8_192
        && pre_reduced_states.len() >= 256;
    if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][combined_equivalence_order] partition={} dedup_tokens={} pre_states={} vocab_first={}",
            partition_label,
            dedup.representative_token_bytes.len(),
            pre_reduced_states.len(),
            vocab_first,
        );
    }
    let (
        reduced_state_reps_for_pre_reduced,
        dedup_vocab_classes,
        vocab_analysis_dfa_build_ms,
        exact_state_equiv_ms,
        vocab_equiv_ms,
        exact_rep_confirmation_used,
    ) = if vocab_first {
        let vocab_equiv_started_at = Instant::now();
        let (dedup_vocab_classes, vocab_analysis_dfa_build_ms) =
            vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                analysis_view,
                &dedup.representative_token_bytes,
                &pre_reduced_states,
                effective_disallowed,
                Some(&byte_to_class),
                if uses_analysis_quotient { None } else { active_groups },
                if uses_analysis_quotient { None } else { shared_vocab_dfa_cache },
                if uses_analysis_quotient { None } else { shared_analysis_dfa_cache },
            );
        let vocab_equiv_ms = vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0;
        let representative_tokens = representative_tokens_for_vocab_classes(
            &dedup_vocab_classes,
            &dedup.representative_token_bytes,
        );
        let exact_rep_confirmation_used = pre_reduced_states.len()
            >= EXACT_REP_CONFIRMATION_MIN_STATES
            && representative_tokens.len() >= EXACT_REP_CONFIRMATION_MIN_TOKENS;
        let exact_started_at = Instant::now();
        let reduced_state_reps_for_pre_reduced = if exact_rep_confirmation_used {
            state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base(
                analysis_view,
                &representative_tokens,
                &pre_reduced_states,
                &normalized_disallowed_follows,
                None,
                None,
                Some(true),
                (!uses_analysis_quotient).then_some(compatible_cache).flatten(),
            )
        } else {
            state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base(
                analysis_view,
                &representative_tokens,
                &pre_reduced_states,
                &normalized_disallowed_follows,
                (!uses_analysis_quotient).then_some(compatible_cache).flatten(),
            )
        };
        let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
        (
            reduced_state_reps_for_pre_reduced,
            dedup_vocab_classes,
            vocab_analysis_dfa_build_ms,
            exact_state_equiv_ms,
            vocab_equiv_ms,
            exact_rep_confirmation_used,
        )
    } else {
        let exact_rep_confirmation_used = pre_reduced_states.len()
            >= EXACT_REP_CONFIRMATION_MIN_STATES
            && dedup.representative_token_bytes.len() >= EXACT_REP_CONFIRMATION_MIN_TOKENS;
        let exact_started_at = Instant::now();
        let reduced_state_reps_for_pre_reduced = if exact_rep_confirmation_used {
            state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base(
                analysis_view,
                &dedup.representative_token_bytes,
                &pre_reduced_states,
                &normalized_disallowed_follows,
                None,
                None,
                Some(true),
                (!uses_analysis_quotient).then_some(compatible_cache).flatten(),
            )
        } else {
            state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base(
                analysis_view,
                &dedup.representative_token_bytes,
                &pre_reduced_states,
                &normalized_disallowed_follows,
                (!uses_analysis_quotient).then_some(compatible_cache).flatten(),
            )
        };
        let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
        let mut final_state_representatives = reduced_state_reps_for_pre_reduced.clone();
        final_state_representatives.sort_unstable();
        final_state_representatives.dedup();
        let vocab_equiv_started_at = Instant::now();
        let (dedup_vocab_classes, vocab_analysis_dfa_build_ms) =
            vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                analysis_view,
                &dedup.representative_token_bytes,
                &final_state_representatives,
                effective_disallowed,
                Some(&byte_to_class),
                if uses_analysis_quotient { None } else { active_groups },
                if uses_analysis_quotient { None } else { shared_vocab_dfa_cache },
                if uses_analysis_quotient { None } else { shared_analysis_dfa_cache },
            );
        let vocab_equiv_ms = vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0;
        (
            reduced_state_reps_for_pre_reduced,
            dedup_vocab_classes,
            vocab_analysis_dfa_build_ms,
            exact_state_equiv_ms,
            vocab_equiv_ms,
            exact_rep_confirmation_used,
        )
    };

    let representative_states = if uses_analysis_quotient {
        prepared
            .initial_states
            .iter()
            .map(|&state| {
                let pre_internal = pre_state_map.original_to_internal[state] as usize;
                let final_internal = reduced_state_reps_for_pre_reduced[pre_internal];
                pre_state_map.representative_original_ids[final_internal] as usize
            })
            .collect::<Vec<_>>()
    } else {
        let rep_to_final: BTreeMap<usize, usize> = pre_reduced_states
            .iter()
            .copied()
            .zip(reduced_state_reps_for_pre_reduced.iter().copied())
            .collect();
        prepared
            .initial_states
            .iter()
            .map(|&state| {
                let pre_internal = pre_state_map.original_to_internal[state];
                let pre_rep = pre_state_map.representative_original_ids[pre_internal as usize] as usize;
                rep_to_final[&pre_rep]
            })
            .collect::<Vec<_>>()
    };
    let id_map_finalize_started_at = Instant::now();
    let vocab_classes = expand_vocab_classes(
        dedup_vocab_classes,
        &dedup.original_to_repr,
        dedup.representative_token_bytes.len(),
    );
    let state_classes =
        state_equivalence_analysis::mapping_to_equivalence_classes(&prepared.initial_states, &representative_states);
    let exact_reps = state_classes.len();
    let result = CombinedEquivalenceResult {
        vocab_classes,
        state_classes,
    };
    let internal_id_map =
        build_internal_id_map_from_combined_result(tokenizer, initial_state_map, &prepared, &result);
    let id_map_finalize_ms = id_map_finalize_started_at.elapsed().as_secs_f64() * 1000.0;

    (
        internal_id_map,
        CombinedEquivalenceProfile {
            initial_states_considered: prepared.initial_states.len(),
            max_length_skipped: pipeline_profile.max_length_skipped,
            max_token_len,
            token_len_gt_4: token_len_stats.gt_4,
            token_len_gt_8: token_len_stats.gt_8,
            token_len_gt_16: token_len_stats.gt_16,
            token_len_gt_32: token_len_stats.gt_32,
            token_len_gt_64: token_len_stats.gt_64,
            raw_analysis_base_init_ms,
            analysis_view_build_ms,
            active_mask_filter_ms: 0.0,
            effective_follows_normalize_ms,
            prepare_inputs_ms,
            byte_class_setup_ms,
            vocab_analysis_dfa_build_ms,
            token_dedup_ms,
            restricted_observation_state_equiv_ms: pipeline_profile.restricted_observation_state_equiv_ms,
            max_length_state_equiv_ms: pipeline_profile.max_length_state_equiv_ms,
            vocab_equiv_ms,
            exact_state_equiv_ms,
            id_map_finalize_ms,
            restricted_observation_reps: pipeline_profile.restricted_observation_reps,
            max_length_reps: pipeline_profile.max_length_reps,
            exact_reps,
            exact_rep_confirmation_used,
        },
    )
}

#[cfg(test)]
mod prepass_selection_tests {
    use super::*;
    use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::compat::{
        FlatDfa, FlatDfaState,
    };
    use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::shared::representative_tokens_for_vocab_classes;
    use std::sync::Arc;

    #[test]
    fn adaptive_nfa_view_skips_powerset_probe_below_bounded_work_threshold() {
        assert!(!should_probe_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Adaptive,
            499_999,
            500_000,
        ));
        assert!(should_probe_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Adaptive,
            500_000,
            500_000,
        ));
    }

    #[test]
    fn adaptive_nfa_view_uses_only_small_probed_powersets() {
        assert!(should_use_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Adaptive,
            true,
            8_192,
            8_192,
        ));
        assert!(!should_use_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Adaptive,
            true,
            8_193,
            8_192,
        ));
        assert!(!should_use_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Adaptive,
            false,
            0,
            8_192,
        ));
    }

    #[test]
    fn forced_nfa_view_policies_override_adaptive_gates() {
        assert!(!should_probe_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Bounded,
            usize::MAX,
            500_000,
        ));
        assert!(!should_use_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Bounded,
            true,
            1,
            8_192,
        ));
        assert!(should_probe_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Powerset,
            0,
            500_000,
        ));
        assert!(should_use_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Powerset,
            true,
            usize::MAX,
            8_192,
        ));
    }

    fn partition_from_representatives<T: Ord + Copy>(
        values: &[T],
        representatives: &[T],
    ) -> BTreeSet<BTreeSet<T>> {
        let mut by_representative = BTreeMap::<T, BTreeSet<T>>::new();
        for (&value, &representative) in values.iter().zip(representatives) {
            by_representative.entry(representative).or_default().insert(value);
        }
        by_representative.into_values().collect()
    }

    fn synthetic_view() -> TokenizerView {
        let state_count = 5usize;
        let mut transitions = vec![u32::MAX; state_count * 256];
        let set = |transitions: &mut [u32], state: usize, byte: u8, target: u32| {
            transitions[state * 256 + byte as usize] = target;
        };
        set(&mut transitions, 0, b'a', 1);
        set(&mut transitions, 0, b'b', 2);
        set(&mut transitions, 1, b'a', 1);
        set(&mut transitions, 1, b'b', 3);
        set(&mut transitions, 2, b'a', 3);
        set(&mut transitions, 2, b'b', 2);
        set(&mut transitions, 3, b'a', 3);
        set(&mut transitions, 3, b'b', 3);
        // State 4 is behaviorally identical to state 1.
        set(&mut transitions, 4, b'a', 4);
        set(&mut transitions, 4, b'b', 3);

        TokenizerView {
            flat_dfa: FlatDfa {
                start_state: 0,
                transitions: Arc::from(transitions),
                states: vec![
                    FlatDfaState {
                        finalizers: vec![],
                        possible_future_group_ids: vec![0, 1],
                    },
                    FlatDfaState {
                        finalizers: vec![0],
                        possible_future_group_ids: vec![0, 1],
                    },
                    FlatDfaState {
                        finalizers: vec![1],
                        possible_future_group_ids: vec![0, 1],
                    },
                    FlatDfaState {
                        finalizers: vec![0, 1],
                        possible_future_group_ids: vec![0, 1],
                    },
                    FlatDfaState {
                        finalizers: vec![0],
                        possible_future_group_ids: vec![0, 1],
                    },
                ],
            },
        }
    }

    #[test]
    fn state_then_vocab_equivalence_matches_vocab_then_state() {
        let view = synthetic_view();
        let tokens: Vec<&[u8]> = vec![
            b"a", b"b", b"aa", b"ab", b"ba", b"bb", b"x", b"y",
        ];
        let states: Vec<usize> = (0..view.dfa().states.len()).collect();
        let byte_to_class = super::super::compat::compute_byte_classes(view.dfa());
        let disallowed = BTreeMap::<u32, BitSet>::new();
        let normalized = normalize_disallowed_follows(2, &disallowed);

        let old_vocab = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &states,
            &disallowed,
            Some(&byte_to_class),
            None,
            None,
            None,
        );
        let old_token_reps = representative_tokens_for_vocab_classes(&old_vocab, &tokens);
        let old_state_reps =
            state_equivalence_analysis::find_state_equivalence_classes_with_disallowed(
                &view,
                &old_token_reps,
                &states,
                &normalized,
            );

        let reversed_state_reps =
            state_equivalence_analysis::find_state_equivalence_classes_with_disallowed(
                &view,
                &tokens,
                &states,
                &normalized,
            );
        let final_state_reps: Vec<usize> = {
            let mut reps = reversed_state_reps.clone();
            reps.sort_unstable();
            reps.dedup();
            reps
        };
        let reversed_vocab = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &final_state_reps,
            &disallowed,
            Some(&byte_to_class),
            None,
            None,
            None,
        );

        assert_eq!(
            partition_from_representatives(&states, &old_state_reps),
            partition_from_representatives(&states, &reversed_state_reps),
        );
        assert_eq!(old_vocab, reversed_vocab);
    }

    #[test]
    fn one_byte_common_prefix_factorization_matches_full_token_state_equivalence() {
        let view = synthetic_view();
        let tokens: Vec<&[u8]> = vec![b"aa", b"ab", b"aaa", b"abb"];
        let suffix_tokens: Vec<&[u8]> = tokens.iter().map(|token| &token[1..]).collect();
        let states: Vec<usize> = (0..view.dfa().states.len()).collect();
        let normalized = normalize_disallowed_follows(2, &BTreeMap::new());

        let full = state_equivalence_analysis::find_state_equivalence_classes_with_disallowed(
            &view,
            &tokens,
            &states,
            &normalized,
        );

        let mut prefix_targets = BTreeSet::<usize>::new();
        let targets = states
            .iter()
            .map(|&source| view.dfa().trans(source, b'a' as usize))
            .collect::<Vec<_>>();
        for &target in &targets {
            if target != u32::MAX {
                prefix_targets.insert(target as usize);
            }
        }
        let prefix_targets = prefix_targets.into_iter().collect::<Vec<_>>();
        let target_representatives = state_equivalence_analysis::
            find_state_equivalence_classes_with_disallowed_and_shared_base_with_initial_finalizers(
                &view,
                &suffix_tokens,
                &prefix_targets,
                &normalized,
                None,
            );
        let behavior_for_target = prefix_targets
            .iter()
            .copied()
            .zip(target_representatives)
            .collect::<BTreeMap<_, _>>();
        let behaviors = targets
            .iter()
            .map(|&target| {
                (target != u32::MAX).then(|| behavior_for_target[&(target as usize)])
            })
            .collect::<Vec<_>>();
        let mut representative_for_behavior = BTreeMap::<Option<usize>, usize>::new();
        for (&source, &behavior) in states.iter().zip(&behaviors) {
            representative_for_behavior.entry(behavior).or_insert(source);
        }
        let factored = behaviors
            .into_iter()
            .map(|behavior| representative_for_behavior[&behavior])
            .collect::<Vec<_>>();

        assert_eq!(
            partition_from_representatives(&states, &full),
            partition_from_representatives(&states, &factored),
        );
    }

    #[test]
    fn directional_quotient_composition_retains_preclass_representative() {
        let directional = ManyToOneIdMap {
            original_to_internal: vec![0, 1, 0, 1],
            internal_to_originals: vec![vec![0, 2], vec![1, 3]],
            representative_original_ids: vec![2, 3],
        };

        let composed = compose_raw_quotient_state_map_preserving_directional_representatives(
            &directional,
            &[1, 1],
        );

        assert_eq!(composed.original_to_internal, vec![0, 0, 0, 0]);
        assert_eq!(composed.internal_to_originals, vec![vec![0, 1, 2, 3]]);
        assert_eq!(composed.representative_original_ids, vec![3]);
    }

    #[test]
    fn selects_direct_refinement_when_byte_bounded_prepass_cannot_amortize() {
        // Small vocabulary with many relevant bytes: direct token walks are
        // cheaper than a full k-bounded byte refinement per lexer state.
        assert!(direct_refinement_work_is_no_larger(100, 10, 41));
        // Larger vocabulary over a smaller byte alphabet amortizes the exact
        // prepass and should retain it.
        assert!(!direct_refinement_work_is_no_larger(900, 14, 19));
    }
}
