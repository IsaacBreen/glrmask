use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use super::compat::TokenizerView;
use super::disallowed_follows::normalize_disallowed_follows;
use super::state::fast as state_equivalence_analysis;
use super::vocab::fast as vocab_equivalence_analysis;
use crate::ds::bitset::BitSet;

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
    // Build rep → new_internal from state_classes
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

    // Compose: original state → initial_state_map → new_internal
    let mut composed_original_to_internal = vec![u32::MAX; num_dfa_states];
    let mut composed_internal_to_originals = vec![Vec::new(); new_internal_to_originals.len()];
    let mut composed_reps = vec![u32::MAX; new_internal_to_originals.len()];

    for (orig_state, &init_internal) in initial_state_map.original_to_internal.iter().enumerate() {
        if init_internal == u32::MAX || (init_internal as usize) >= initial_state_map.representative_original_ids.len() {
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
            // Use the token with the smallest token_id as representative.
            // No need to sort by bytes — the ordering within each class
            // doesn't affect correctness (only used for bitmask construction).
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
    // Sort classes by representative token_id (fast integer comparison).
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

/// Fast L1 equivalence analysis using transposed DFA with batched state
/// processing and rayon parallelism.
///
/// For L1 terminals (each token matches ≤1 terminal from each state), token
/// equivalence is determined by the ending DFA state from every starting state.
/// This function processes states in cache-friendly batches using a byte-class
/// compressed transposed transition table, achieving much better performance
/// than the naive per-token-per-state DFA walk.
pub(crate) fn analyze_equivalences_l1_fast(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> InternalIdMap {
    use rayon::prelude::*;
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};

    let num_states = tokenizer.num_states() as usize;
    let max_token_id = vocab.max_token_id();

    // Extract vocab token IDs and byte slices.
    let mut token_ids: Vec<u32> = Vec::with_capacity(vocab.len());
    let mut token_bytes_list: Vec<&[u8]> = Vec::with_capacity(vocab.len());
    for (&tid, bytes) in &vocab.entries {
        token_ids.push(tid);
        token_bytes_list.push(bytes.as_slice());
    }
    let num_tokens = token_ids.len();

    // Byte-class token deduplication.
    let tokenizer_view = TokenizerView::new(tokenizer);
    let byte_to_class = super::compat::compute_byte_classes(tokenizer_view.dfa());
    let mut bc_hash_to_repr: HashMap<u128, usize> = HashMap::with_capacity(num_tokens / 2);
    let mut repr_indices: Vec<usize> = Vec::new();
    let mut original_to_repr: Vec<usize> = Vec::with_capacity(num_tokens);
    for (idx, &bytes) in token_bytes_list.iter().enumerate() {
        let h = hash_byte_class_seq(bytes, &byte_to_class);
        let repr = *bc_hash_to_repr.entry(h).or_insert_with(|| {
            let r = repr_indices.len();
            repr_indices.push(idx);
            r
        });
        original_to_repr.push(repr);
    }
    let num_repr = repr_indices.len();
    // Build transposed transition table for cache-optimal batched access.
    let dfa = tokenizer_view.dfa();
    let num_byte_classes = byte_to_class.iter().copied().max().map_or(0usize, |m| m as usize + 1);
    let mut class_repr_byte = vec![0u8; num_byte_classes];
    let mut class_seen = vec![false; num_byte_classes];
    for b in 0..=255u8 {
        let c = byte_to_class[b as usize] as usize;
        if !class_seen[c] {
            class_seen[c] = true;
            class_repr_byte[c] = b;
        }
    }
    // trans_by_class[class * num_states + state] = next_state (or u32::MAX for dead)
    // Row-major construction: one pass over states instead of num_byte_classes passes.
    let mut trans_by_class: Vec<u32> = vec![u32::MAX; num_byte_classes * num_states];
    for s in 0..num_states {
        for c in 0..num_byte_classes {
            trans_by_class[c * num_states + s] = dfa.trans(s, class_repr_byte[c] as usize);
        }
    }
    // Compute fingerprints: for each repr token, hash the ending state from
    // state group representatives only. States with identical transition rows
    // produce identical token walks, so grouping them first avoids redundant work.
    // Initial state grouping by transition row: states with identical
    // 1-byte transitions for all byte classes are provably equivalent
    // for all tokens (since the DFA is deterministic).
    let mut row_hash_to_group: HashMap<u64, u32> = HashMap::with_capacity(num_states / 2);
    let mut initial_state_reps: Vec<usize> = Vec::new();
    let mut initial_state_group: Vec<u32> = Vec::with_capacity(num_states);
    for state in 0..num_states {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for c in 0..num_byte_classes {
            trans_by_class[c * num_states + state].hash(&mut hasher);
        }
        let h = hasher.finish();
        let next_id = initial_state_reps.len() as u32;
        let group = *row_hash_to_group.entry(h).or_insert_with(|| {
            initial_state_reps.push(state);
            next_id
        });
        initial_state_group.push(group);
    }
    // Each repr token gets a fingerprint: hash of ending state from
    // initial state group representatives only.
    let repr_fps: Vec<u64> = repr_indices
        .par_iter()
        .map(|&idx| {
            let bytes = token_bytes_list[idx];
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            for &state in &initial_state_reps {
                let mut s = state as u32;
                let mut dead = false;
                for &b in bytes {
                    let class = byte_to_class[b as usize] as usize;
                    let next = trans_by_class[class * num_states + s as usize];
                    if next == u32::MAX {
                        dead = true;
                        break;
                    }
                    s = next;
                }
                if dead {
                    0u32.hash(&mut hasher);
                } else {
                    1u32.hash(&mut hasher);
                    s.hash(&mut hasher);
                }
            }
            let sig: u64 = hasher.finish();
            sig
        })
        .collect();
    // Group representative tokens by fingerprint → token classes.
    let mut repr_sig_to_class: HashMap<u64, u32> = HashMap::new();
    let mut repr_class: Vec<u32> = Vec::with_capacity(num_repr);
    let mut repr_class_reps: Vec<usize> = Vec::new();
    for (r, &fp) in repr_fps.iter().enumerate() {
        let next_id = repr_class_reps.len() as u32;
        let class = *repr_sig_to_class.entry(fp).or_insert_with(|| {
            repr_class_reps.push(r);
            next_id
        });
        repr_class.push(class);
    }
    let num_repr_classes = repr_class_reps.len();

    // Expand back to original tokens.
    let mut token_original_to_internal = vec![u32::MAX; (max_token_id + 1) as usize];
    let mut token_internal_to_originals: Vec<Vec<u32>> = vec![Vec::new(); num_repr_classes];
    let mut token_representative_ids: Vec<u32> = vec![u32::MAX; num_repr_classes];
    for (orig_idx, &tid) in token_ids.iter().enumerate() {
        let repr = original_to_repr[orig_idx];
        let class = repr_class[repr];
        token_original_to_internal[tid as usize] = class;
        token_internal_to_originals[class as usize].push(tid);
        if tid < token_representative_ids[class as usize] {
            token_representative_ids[class as usize] = tid;
        }
    }
    // State equivalence: group states by behavior across token class representatives.
    // For state grouping, we need per-state per-token-class fingerprints.
    // Compute ending state for each (state, repr_token_class_representative).
    let class_repr_tokens: Vec<&[u8]> = repr_class_reps
        .iter()
        .map(|&r| token_bytes_list[repr_indices[r]])
        .collect();

    // Compute state fingerprint: hash of (ending_state_for_class_0, ending_state_for_class_1, ...)
    let state_fps: Vec<u64> = (0..num_states)
        .into_par_iter()
        .map(|state| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            for token_bytes in &class_repr_tokens {
                let mut s = state as u32;
                let mut dead = false;
                for &b in *token_bytes {
                    let class = byte_to_class[b as usize] as usize;
                    let next = trans_by_class[class * num_states + s as usize];
                    if next == u32::MAX {
                        dead = true;
                        break;
                    }
                    s = next;
                }
                if dead {
                    0u32.hash(&mut hasher);
                } else {
                    1u32.hash(&mut hasher);
                    s.hash(&mut hasher);
                }
            }
            hasher.finish()
        })
        .collect();

    let mut state_sig_to_class: HashMap<u64, u32> = HashMap::new();
    let mut state_original_to_internal = vec![0u32; num_states];
    let mut state_internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut state_representative_ids: Vec<u32> = Vec::new();
    for state in 0..num_states {
        let next_id = state_internal_to_originals.len() as u32;
        let class = *state_sig_to_class.entry(state_fps[state]).or_insert_with(|| {
            state_internal_to_originals.push(Vec::new());
            state_representative_ids.push(state as u32);
            next_id
        });
        state_original_to_internal[state] = class;
        state_internal_to_originals[class as usize].push(state as u32);
    }
    let num_state_classes = state_internal_to_originals.len();
    // Re-group tokens using only state class representative fingerprints.
    // Skip if no state reduction occurred (same fingerprints guaranteed).
    let (refined_num_token_classes, refined_token_original_to_internal, refined_token_internal_to_originals, refined_token_representative_ids) = if num_state_classes == num_states {
        // No state reduction: raw token classes are already optimal.
        (num_repr_classes, token_original_to_internal.clone(), token_internal_to_originals.clone(), token_representative_ids.clone())
    } else {
    let state_class_rep_indices: Vec<usize> = state_representative_ids
        .iter()
        .map(|&sid| sid as usize)
        .collect();

    let refined_repr_fps: Vec<u64> = repr_indices
        .par_iter()
        .map(|&idx| {
            let bytes = token_bytes_list[idx];
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            for &state in &state_class_rep_indices {
                let mut s = state as u32;
                let mut dead = false;
                for &b in bytes {
                    let class = byte_to_class[b as usize] as usize;
                    let next = trans_by_class[class * num_states + s as usize];
                    if next == u32::MAX {
                        dead = true;
                        break;
                    }
                    s = next;
                }
                if dead {
                    0u32.hash(&mut hasher);
                } else {
                    1u32.hash(&mut hasher);
                    s.hash(&mut hasher);
                }
            }
            hasher.finish()
        })
        .collect();

    let mut refined_repr_sig_to_class: HashMap<u64, u32> = HashMap::new();
    let mut refined_repr_class: Vec<u32> = Vec::with_capacity(num_repr);
    let mut refined_class_reps: Vec<usize> = Vec::new();
    for (r, &fp) in refined_repr_fps.iter().enumerate() {
        let next_id = refined_class_reps.len() as u32;
        let class = *refined_repr_sig_to_class.entry(fp).or_insert_with(|| {
            refined_class_reps.push(r);
            next_id
        });
        refined_repr_class.push(class);
    }
    let refined_num_tc = refined_class_reps.len();

    // Expand refined classes.
    let mut ref_token_o2i = vec![u32::MAX; (max_token_id + 1) as usize];
    let mut ref_token_i2o: Vec<Vec<u32>> = vec![Vec::new(); refined_num_tc];
    let mut ref_token_reps: Vec<u32> = vec![u32::MAX; refined_num_tc];
    for (orig_idx, &tid) in token_ids.iter().enumerate() {
        let repr = original_to_repr[orig_idx];
        let class = refined_repr_class[repr];
        ref_token_o2i[tid as usize] = class;
        ref_token_i2o[class as usize].push(tid);
        if tid < ref_token_reps[class as usize] {
            ref_token_reps[class as usize] = tid;
        }
    }
    (refined_num_tc, ref_token_o2i, ref_token_i2o, ref_token_reps)
    }; // end of if/else for refinement
    InternalIdMap {
        tokenizer_states: ManyToOneIdMap {
            original_to_internal: state_original_to_internal,
            internal_to_originals: state_internal_to_originals,
            representative_original_ids: state_representative_ids,
        },
        vocab_tokens: ManyToOneIdMap {
            original_to_internal: refined_token_original_to_internal,
            internal_to_originals: refined_token_internal_to_originals,
            representative_original_ids: refined_token_representative_ids,
        },
    }
}

/// L1 fast path for equivalence analysis.
///
/// When all terminals have max path length ≤ 1, no multi-terminal token paths
/// exist. Token equivalence reduces to "same finalizer set from every state"
/// and state equivalence to "same token-class behavior for every token class".
///
/// Uses byte-class token deduplication then direct DFA fingerprinting,
/// avoiding the product-DFA construction used in the general case.
pub(crate) fn analyze_equivalences_l1(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> InternalIdMap {
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};

    let num_states = tokenizer.num_states() as usize;
    let max_token_id = vocab.max_token_id();

    // Extract vocab token IDs and byte slices.
    let mut token_ids: Vec<u32> = Vec::with_capacity(vocab.len());
    let mut token_bytes_list: Vec<&[u8]> = Vec::with_capacity(vocab.len());
    for (&tid, bytes) in &vocab.entries {
        token_ids.push(tid);
        token_bytes_list.push(bytes.as_slice());
    }
    let num_tokens = token_ids.len();

    // Byte-class token deduplication: tokens with the same byte-class sequence
    // produce identical DFA behavior from every state.
    let tokenizer_view = TokenizerView::new(tokenizer);
    let byte_to_class = super::compat::compute_byte_classes(tokenizer_view.dfa());
    let mut bc_hash_to_repr: HashMap<u128, usize> = HashMap::with_capacity(num_tokens / 2);
    let mut repr_indices: Vec<usize> = Vec::new(); // indices into token_ids/token_bytes_list
    let mut original_to_repr: Vec<usize> = Vec::with_capacity(num_tokens);
    for (idx, &bytes) in token_bytes_list.iter().enumerate() {
        let h = hash_byte_class_seq(bytes, &byte_to_class);
        let repr = *bc_hash_to_repr.entry(h).or_insert_with(|| {
            let r = repr_indices.len();
            repr_indices.push(idx);
            r
        });
        original_to_repr.push(repr);
    }
    let num_repr = repr_indices.len();
    // Fingerprint per representative token: one u64 per state.
    let mut repr_fps: Vec<Vec<u64>> = Vec::with_capacity(num_repr);
    for &idx in &repr_indices {
        let bytes = token_bytes_list[idx];
        let mut fps = Vec::with_capacity(num_states);
        for state in 0..num_states as u32 {
            let mut s = state;
            let mut dead = false;
            for &b in bytes {
                match tokenizer.step(s, b) {
                    Some(next) => s = next,
                    None => {
                        dead = true;
                        break;
                    }
                }
            }
            if dead {
                fps.push(0u64);
            } else {
                let finalizers = tokenizer.dfa.finalizers(s);
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                finalizers.hash(&mut hasher);
                1u8.hash(&mut hasher);
                fps.push(hasher.finish());
            }
        }
        repr_fps.push(fps);
    }
    // Group representative tokens by fingerprint vector → token dedup classes.
    let mut repr_sig_to_class: HashMap<u64, u32> = HashMap::new();
    let mut repr_class: Vec<u32> = Vec::with_capacity(num_repr);
    let mut repr_class_reps: Vec<usize> = Vec::new(); // repr index for each class
    for (r, fps) in repr_fps.iter().enumerate() {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        fps.hash(&mut hasher);
        let sig = hasher.finish();
        let next_id = repr_class_reps.len() as u32;
        let class = *repr_sig_to_class.entry(sig).or_insert_with(|| {
            repr_class_reps.push(r);
            next_id
        });
        repr_class.push(class);
    }
    let num_repr_classes = repr_class_reps.len();

    // Expand back to original tokens.
    let mut token_original_to_internal = vec![u32::MAX; (max_token_id + 1) as usize];
    let mut token_internal_to_originals: Vec<Vec<u32>> = vec![Vec::new(); num_repr_classes];
    let mut token_representative_ids: Vec<u32> = vec![u32::MAX; num_repr_classes];
    for (orig_idx, &tid) in token_ids.iter().enumerate() {
        let repr = original_to_repr[orig_idx];
        let class = repr_class[repr];
        token_original_to_internal[tid as usize] = class;
        let bucket = &mut token_internal_to_originals[class as usize];
        bucket.push(tid);
        if tid < token_representative_ids[class as usize] {
            token_representative_ids[class as usize] = tid;
        }
    }
    // Group states by behavior across token class representative fingerprints.
    let mut state_sig_to_class: HashMap<u64, u32> = HashMap::new();
    let mut state_original_to_internal = vec![0u32; num_states];
    let mut state_internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut state_representative_ids: Vec<u32> = Vec::new();

    for state in 0..num_states {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &rep_r in &repr_class_reps {
            repr_fps[rep_r][state].hash(&mut hasher);
        }
        let sig = hasher.finish();
        let next_id = state_internal_to_originals.len() as u32;
        let class = *state_sig_to_class.entry(sig).or_insert_with(|| {
            state_internal_to_originals.push(Vec::new());
            state_representative_ids.push(state as u32);
            next_id
        });
        state_original_to_internal[state] = class;
        state_internal_to_originals[class as usize].push(state as u32);
    }
    let num_state_classes = state_internal_to_originals.len();
    // Re-group tokens using only state class representative fingerprints.
    let state_class_reps: Vec<usize> = state_representative_ids
        .iter()
        .map(|&sid| sid as usize)
        .collect();

    let mut refined_repr_sig_to_class: HashMap<u64, u32> = HashMap::new();
    let mut refined_repr_class: Vec<u32> = Vec::with_capacity(num_repr);
    let mut refined_class_reps: Vec<usize> = Vec::new();
    for (r, fps) in repr_fps.iter().enumerate() {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &state in &state_class_reps {
            fps[state].hash(&mut hasher);
        }
        let sig = hasher.finish();
        let next_id = refined_class_reps.len() as u32;
        let class = *refined_repr_sig_to_class.entry(sig).or_insert_with(|| {
            refined_class_reps.push(r);
            next_id
        });
        refined_repr_class.push(class);
    }
    let refined_num_token_classes = refined_class_reps.len();

    // Expand refined classes back to original tokens.
    let mut refined_token_original_to_internal = vec![u32::MAX; (max_token_id + 1) as usize];
    let mut refined_token_internal_to_originals: Vec<Vec<u32>> = vec![Vec::new(); refined_num_token_classes];
    let mut refined_token_representative_ids: Vec<u32> = vec![u32::MAX; refined_num_token_classes];
    for (orig_idx, &tid) in token_ids.iter().enumerate() {
        let repr = original_to_repr[orig_idx];
        let class = refined_repr_class[repr];
        refined_token_original_to_internal[tid as usize] = class;
        let bucket = &mut refined_token_internal_to_originals[class as usize];
        bucket.push(tid);
        if tid < refined_token_representative_ids[class as usize] {
            refined_token_representative_ids[class as usize] = tid;
        }
    }
    InternalIdMap {
        tokenizer_states: ManyToOneIdMap {
            original_to_internal: state_original_to_internal,
            internal_to_originals: state_internal_to_originals,
            representative_original_ids: state_representative_ids,
        },
        vocab_tokens: ManyToOneIdMap {
            original_to_internal: refined_token_original_to_internal,
            internal_to_originals: refined_token_internal_to_originals,
            representative_original_ids: refined_token_representative_ids,
        },
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

