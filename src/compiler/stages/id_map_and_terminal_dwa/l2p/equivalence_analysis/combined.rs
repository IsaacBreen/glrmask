use crate::automata::lexer::Lexer;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::OnceLock;
use std::time::Instant;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::id_map_and_terminal_dwa::grammar_helpers::ignore_transparent_disallowed_follows;
use super::state_equivalence::{
    resolve_l2p_pipeline_config, run_state_equivalence_pipeline, StateEquivalenceScope,
};
use crate::ds::bitset::BitSet;
use super::compat::TokenizerView;
use super::disallowed_follows::normalize_disallowed_follows;
use super::shared::{
    TokenDedup,
    expand_vocab_classes,
    hash_byte_class_seq,
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
    pub(crate) prepare_inputs_ms: f64,
    pub(crate) byte_class_setup_ms: f64,
    pub(crate) token_dedup_ms: f64,
    pub(crate) max_length_state_equiv_ms: f64,
    pub(crate) vocab_equiv_ms: f64,
    pub(crate) exact_state_equiv_ms: f64,
    pub(crate) id_map_finalize_ms: f64,
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

#[inline]
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

pub(crate) fn analyze_equivalences_with_group_filter(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    active_groups: Option<&[bool]>,
    shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
    shared_base_setup_ms: f64,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> (InternalIdMap, CombinedEquivalenceProfile) {
    analyze_equivalences_impl(
        partition_label,
        tokenizer,
        vocab,
        disallowed_follows,
        ignore_terminal,
        active_groups,
        shared_vocab_dfa_cache,
        shared_base_setup_ms,
        flat_trans,
        initial_state_map,
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
    active_groups: Option<&[bool]>,
    shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
    shared_base_setup_ms: f64,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> (InternalIdMap, CombinedEquivalenceProfile) {
    let token_path_disallowed_follows =
        ignore_transparent_disallowed_follows(disallowed_follows, ignore_terminal);
    let effective_disallowed = &token_path_disallowed_follows;
    // Only use shared flat_trans when state count matches the (possibly
    // simplified) tokenizer. If simplify_for_terminals minimized the DFA,
    // the original flat_trans has different state numbering and must be
    // discarded.
    let compatible_flat_trans = flat_trans.filter(|ft| {
        ft.len() == tokenizer.num_states() as usize * 256
    });
    let tokenizer_view = match (active_groups, compatible_flat_trans) {
        (Some(active_groups), Some(ft)) => TokenizerView::new_filtered_from_flat_trans(ft, tokenizer, active_groups),
        (Some(active_groups), None) => TokenizerView::new_filtered(tokenizer, active_groups),
        (None, Some(ft)) => TokenizerView::new_from_flat_trans(ft, tokenizer),
        _ => TokenizerView::new(tokenizer),
    };

    let prepare_inputs_started_at = Instant::now();
    let prepared = prepare_equivalence_inputs(tokenizer, vocab, initial_state_map);
    let token_len_stats = token_length_stats(&prepared.token_bytes);
    let prepare_inputs_ms = prepare_inputs_started_at.elapsed().as_secs_f64() * 1000.0;

    let byte_class_setup_started_at = Instant::now();
    if let Some(cache) = shared_vocab_dfa_cache {
        cache.get_or_init(|| vocab_equivalence_analysis::SharedVocabDfaBase::build_from_dfa(tokenizer_view.dfa()));
    }

    let compatible_cache = shared_vocab_dfa_cache
        .and_then(|cache| cache.get())
        .filter(|base| base.is_compatible_with_dfa(tokenizer_view.dfa()));
    let byte_to_class = compatible_cache
        .map(|base| base.byte_to_class())
        .unwrap_or_else(|| super::compat::compute_byte_classes(tokenizer_view.dfa()));
    let byte_class_setup_ms = shared_base_setup_ms
        + byte_class_setup_started_at.elapsed().as_secs_f64() * 1000.0;

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
        Some(&tokenizer_view),
        Some(&byte_to_class),
    );
    let pre_reduced_states: Vec<usize> = pre_state_map
        .representative_original_ids
        .iter()
        .map(|&state| state as usize)
        .collect();
    let normalized_disallowed_follows =
        normalize_disallowed_follows(tokenizer_group_count(&tokenizer_view), effective_disallowed);

    // First finalize state equivalence against every deduplicated token.
    // A final state representative is indistinguishable from each state it
    // represents for every token, so token equivalence may then be computed
    // over only those representatives. This reverses two exact quotients and
    // avoids classifying tokens over the much larger pre-refinement state set.
    let exact_rep_confirmation_used = pre_reduced_states.len() >= EXACT_REP_CONFIRMATION_MIN_STATES
        && dedup.representative_token_bytes.len() >= EXACT_REP_CONFIRMATION_MIN_TOKENS;
    let exact_started_at = Instant::now();
    let reduced_state_reps_for_pre_reduced = if exact_rep_confirmation_used {
        state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base(
            &tokenizer_view,
            &dedup.representative_token_bytes,
            &pre_reduced_states,
            &normalized_disallowed_follows,
            None,
            None,
            Some(true),
            compatible_cache,
        )
    } else {
        state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base(
            &tokenizer_view,
            &dedup.representative_token_bytes,
            &pre_reduced_states,
            &normalized_disallowed_follows,
            compatible_cache,
        )
    };
    let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;

    let rep_to_final: BTreeMap<usize, usize> = pre_reduced_states
        .iter()
        .copied()
        .zip(reduced_state_reps_for_pre_reduced.iter().copied())
        .collect();
    let representative_states = prepared.initial_states
        .iter()
        .map(|&state| {
            let pre_internal = pre_state_map.original_to_internal[state];
            let pre_rep = pre_state_map.representative_original_ids[pre_internal as usize] as usize;
            rep_to_final[&pre_rep]
        })
        .collect::<Vec<_>>();
    let mut final_state_representatives = reduced_state_reps_for_pre_reduced;
    final_state_representatives.sort_unstable();
    final_state_representatives.dedup();

    let vocab_equiv_started_at = Instant::now();
    let dedup_vocab_classes = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter(
        &tokenizer_view,
        &dedup.representative_token_bytes,
        &final_state_representatives,
        effective_disallowed,
        Some(&byte_to_class),
        active_groups,
        shared_vocab_dfa_cache,
    );
    let vocab_equiv_ms = vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0;

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
            prepare_inputs_ms,
            byte_class_setup_ms,
            token_dedup_ms,
            max_length_state_equiv_ms: pipeline_profile.max_length_state_equiv_ms,
            vocab_equiv_ms,
            exact_state_equiv_ms,
            id_map_finalize_ms,
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
        );

        assert_eq!(
            partition_from_representatives(&states, &old_state_reps),
            partition_from_representatives(&states, &reversed_state_reps),
        );
        assert_eq!(old_vocab, reversed_vocab);
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
