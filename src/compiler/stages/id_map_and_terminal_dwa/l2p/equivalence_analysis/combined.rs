use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::bitset::BitSet;
use super::compat::TokenizerView;
use super::disallowed_follows::normalize_disallowed_follows;
use super::state::fast as state_equivalence_analysis;
use super::vocab::fast as vocab_equivalence_analysis;

fn elapsed_ms(started_at: std::time::Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

struct TokenDedup<'a> {
    representative_token_bytes: Vec<&'a [u8]>,
    original_to_repr: Vec<usize>,
}

#[inline]
pub(crate) fn hash_byte_class_seq(bytes: &[u8], byte_to_class: &[u8; 256]) -> u128 {
    let mut hash: u128 = 0xFF51_AFD7_ED55_8CCD;
    hash = hash.wrapping_mul(0xC4CE_B9FE_1A85_EC53).wrapping_add(bytes.len() as u128);
    for &byte in bytes {
        hash = hash
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(byte_to_class[byte as usize] as u128);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    hash ^= hash >> 29;
    hash
}

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

fn expand_vocab_classes(
    dedup_classes: vocab_equivalence_analysis::VocabEquivalenceResult,
    original_to_repr: &[usize],
    num_representatives: usize,
) -> vocab_equivalence_analysis::VocabEquivalenceResult {
    let mut repr_to_class = vec![usize::MAX; num_representatives];
    let mut original_classes: Vec<Vec<usize>> = Vec::with_capacity(dedup_classes.len());

    for (class_idx, dedup_class) in dedup_classes.iter().enumerate() {
        for &repr_idx in dedup_class {
            repr_to_class[repr_idx] = class_idx;
        }
        original_classes.push(Vec::new());
    }

    for (original_idx, &repr_idx) in original_to_repr.iter().enumerate() {
        let class_idx = repr_to_class[repr_idx];
        debug_assert!(class_idx != usize::MAX);
        original_classes[class_idx].push(original_idx);
    }

    original_classes
        .into_iter()
        .filter(|class| !class.is_empty())
        .collect()
}

fn representative_tokens_for_vocab_classes<'a>(
    dedup_vocab_classes: &vocab_equivalence_analysis::VocabEquivalenceResult,
    representative_token_bytes: &'a [&'a [u8]],
) -> Vec<&'a [u8]> {
    dedup_vocab_classes
        .iter()
        .map(|dedup_class| representative_token_bytes[dedup_class[0]])
        .collect()
}

fn tokenizer_group_count(tokenizer: &TokenizerView) -> usize {
    tokenizer
        .dfa()
        .states
        .iter()
        .flat_map(|state| {
            state
                .finalizers
                .iter()
                .copied()
                .chain(state.possible_future_group_ids.iter().copied())
        })
        .max()
        .map_or(0, |max_group| max_group + 1)
}

fn adjust_disallowed_follows(
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
) -> Option<BTreeMap<u32, BitSet>> {
    let ignore_terminal = ignore_terminal?;
    let mut adjusted = disallowed_follows.clone();
    adjusted.remove(&ignore_terminal);
    for bits in adjusted.values_mut() {
        if (ignore_terminal as usize) < bits.len() {
            bits.clear(ignore_terminal as usize);
        }
    }
    adjusted.retain(|_, bits| !bits.is_zero());
    Some(adjusted)
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

pub(crate) fn analyze_equivalences(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
) -> InternalIdMap {
    analyze_equivalences_impl(tokenizer, vocab, disallowed_follows, ignore_terminal, None, shared_vocab_dfa_cache, None, None)
}

pub(crate) fn analyze_equivalences_with_group_filter(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    active_groups: Option<&[bool]>,
    shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> InternalIdMap {
    analyze_equivalences_impl(tokenizer, vocab, disallowed_follows, ignore_terminal, active_groups, shared_vocab_dfa_cache, flat_trans, initial_state_map)
}

/// Combined equivalence analysis over a flattened tokenizer DFA.
///
/// Uses state equivalence (k-step hashing plus token-based refinement) followed
/// by vocab equivalence (parallel batched with byte-class compression).
fn analyze_equivalences_impl(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    active_groups: Option<&[bool]>,
    shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> InternalIdMap {
    let adjusted_disallowed = adjust_disallowed_follows(disallowed_follows, ignore_terminal);
    let effective_disallowed = adjusted_disallowed.as_ref().unwrap_or(disallowed_follows);
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
    if let Some(cache) = shared_vocab_dfa_cache {
        cache.get_or_init(|| vocab_equivalence_analysis::SharedVocabDfaBase::build_from_dfa(tokenizer_view.dfa()));
    }

    let compatible_cache = shared_vocab_dfa_cache
        .and_then(|cache| cache.get())
        .filter(|base| base.is_compatible_with_dfa(tokenizer_view.dfa()));
    let byte_to_class = compatible_cache
        .map(|base| base.byte_to_class())
        .unwrap_or_else(|| super::compat::compute_byte_classes(tokenizer_view.dfa()));

    // Extract vocab tokens as byte slices, ordered by token ID.
    let max_token_id = vocab.max_token_id();
    let mut token_bytes: Vec<&[u8]> = Vec::with_capacity(vocab.len());
    let mut token_ids: Vec<u32> = Vec::with_capacity(vocab.len());
    for (&tid, bytes) in &vocab.entries {
        token_ids.push(tid);
        token_bytes.push(bytes.as_slice());
    }
    // All DFA states as initial states (use initial_state_map representatives if available)
    let initial_states: Vec<usize> = match initial_state_map {
        Some(map) => map.representative_original_ids.iter().map(|&s| s as usize).collect(),
        None => (0..tokenizer.num_states() as usize).collect(),
    };
    let dedup = deduplicate_tokens_by_byte_class(&token_bytes, &byte_to_class);
    let mut relevant_bytes = [false; 256];
    for token in &dedup.representative_token_bytes {
        for &byte in *token {
            relevant_bytes[byte as usize] = true;
        }
    }
    let pre_state_reps = super::state::max_length::find_state_equivalence_classes_byte_restricted(
        &tokenizer_view,
        &dedup.representative_token_bytes,
        &initial_states,
        Some(&byte_to_class),
        active_groups,
        Some(&relevant_bytes),
    );
    let pre_reduced_states: Vec<usize> = pre_state_reps
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let normalized_disallowed_follows =
        normalize_disallowed_follows(tokenizer_group_count(&tokenizer_view), effective_disallowed);

    let dedup_vocab_classes = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter(
        &tokenizer_view,
        &dedup.representative_token_bytes,
        &pre_reduced_states,
        effective_disallowed,
        Some(&byte_to_class),
        active_groups,
        shared_vocab_dfa_cache,
    );
    let vocab_representative_tokens = representative_tokens_for_vocab_classes(
        &dedup_vocab_classes,
        &dedup.representative_token_bytes,
    );
    let reduced_state_reps_for_pre_reduced =
        state_equivalence_analysis::find_state_equivalence_classes_with_disallowed(
            &tokenizer_view,
            &vocab_representative_tokens,
            &pre_reduced_states,
            &normalized_disallowed_follows,
        );
    let rep_to_final: BTreeMap<usize, usize> = pre_reduced_states
        .iter()
        .copied()
        .zip(reduced_state_reps_for_pre_reduced.iter().copied())
        .collect();
    let representative_states = pre_state_reps
        .iter()
        .map(|pre_rep| rep_to_final[pre_rep])
        .collect::<Vec<_>>();
    let vocab_classes = expand_vocab_classes(
        dedup_vocab_classes,
        &dedup.original_to_repr,
        dedup.representative_token_bytes.len(),
    );
    let state_classes =
        state_equivalence_analysis::mapping_to_equivalence_classes(&initial_states, &representative_states);
    let num_dfa_states = tokenizer.num_states() as usize;
    let state_map = match initial_state_map {
        Some(init_map) => build_state_map_composed(&state_classes, num_dfa_states, init_map),
        None => build_state_map(&state_classes, num_dfa_states),
    };
    let vocab_map = build_vocab_map(&vocab_classes, &token_ids, max_token_id);

    InternalIdMap {
        tokenizer_states: state_map,
        vocab_tokens: vocab_map,
    }
}

