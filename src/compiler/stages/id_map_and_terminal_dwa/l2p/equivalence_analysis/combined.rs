use std::collections::{BTreeMap, BTreeSet};

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use super::compat::TokenizerView;
use super::combined_equivalence_analysis;
use super::combined_equivalence_analysis::hash_byte_class_seq;
use crate::ds::bitset::BitSet;

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let trimmed = value.trim();
            !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn compile_profile_enabled() -> bool {
    env_flag_enabled("GLRMASK_PROFILE_COMPILE") || env_flag_enabled("GLRMASK_PROFILE_COMPILE_SUMMARY")
}

fn elapsed_ms(started_at: std::time::Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
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

    let profile_compile = compile_profile_enabled();
    let total_started_at = std::time::Instant::now();

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
    let dedup_started_at = std::time::Instant::now();
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
    let dedup_ms = elapsed_ms(dedup_started_at);

    // Build transposed transition table for cache-optimal batched access.
    let build_dfa_started_at = std::time::Instant::now();
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
    for (s, state) in dfa.states.iter().enumerate() {
        for c in 0..num_byte_classes {
            trans_by_class[c * num_states + s] = state.transitions[class_repr_byte[c] as usize];
        }
    }
    let build_dfa_ms = elapsed_ms(build_dfa_started_at);

    // Compute fingerprints: for each repr token, hash the ending state from
    // state group representatives only. States with identical transition rows
    // produce identical token walks, so grouping them first avoids redundant work.
    let fp_started_at = std::time::Instant::now();

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
    let num_initial_reps = initial_state_reps.len();

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
    let fp_ms = elapsed_ms(fp_started_at);

    // Group representative tokens by fingerprint → token classes.
    let token_group_started_at = std::time::Instant::now();
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
    let token_group_ms = elapsed_ms(token_group_started_at);

    // State equivalence: group states by behavior across token class representatives.
    let state_group_started_at = std::time::Instant::now();

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
    let state_group_ms = elapsed_ms(state_group_started_at);

    // Re-group tokens using only state class representative fingerprints.
    // Skip if no state reduction occurred (same fingerprints guaranteed).
    let refine_started_at = std::time::Instant::now();
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
    let refine_ms = elapsed_ms(refine_started_at);

    if profile_compile {
        eprintln!(
            "[glrmask/profile][id_map_l1_fast] dedup_ms={:.3} tokens={}->{} build_dfa_ms={:.3} fp_ms={:.3} initial_state_reps={} token_group_ms={:.3} state_group_ms={:.3} refine_ms={:.3} states={} state_classes={} token_classes_raw={} token_classes_refined={} total_ms={:.3}",
            dedup_ms, num_tokens, num_repr,
            build_dfa_ms,
            fp_ms, num_initial_reps, token_group_ms, state_group_ms, refine_ms,
            num_states, num_state_classes,
            num_repr_classes, refined_num_token_classes,
            elapsed_ms(total_started_at),
        );
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

    let profile_compile = compile_profile_enabled();
    let total_started_at = std::time::Instant::now();

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
    let dedup_started_at = std::time::Instant::now();
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
    let dedup_ms = elapsed_ms(dedup_started_at);

    // Fingerprint per representative token: one u64 per state.
    let fp_started_at = std::time::Instant::now();
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
    let fp_ms = elapsed_ms(fp_started_at);

    // Group representative tokens by fingerprint vector → token dedup classes.
    let token_group_started_at = std::time::Instant::now();
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
    let token_group_ms = elapsed_ms(token_group_started_at);

    // Group states by behavior across token class representative fingerprints.
    let state_group_started_at = std::time::Instant::now();
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
    let state_group_ms = elapsed_ms(state_group_started_at);

    // Re-group tokens using only state class representative fingerprints.
    let refine_started_at = std::time::Instant::now();
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
    let refine_ms = elapsed_ms(refine_started_at);

    if profile_compile {
        eprintln!(
            "[glrmask/profile][id_map_l1] dedup_ms={:.3} tokens={}->{} fp_ms={:.3} token_group_ms={:.3} state_group_ms={:.3} refine_ms={:.3} states={} state_classes={} token_classes_raw={} token_classes_refined={} total_ms={:.3}",
            dedup_ms, num_tokens, num_repr,
            fp_ms, token_group_ms, state_group_ms, refine_ms,
            num_states, num_state_classes,
            num_repr_classes, refined_num_token_classes,
            elapsed_ms(total_started_at),
        );
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
    analyze_equivalences_impl(tokenizer, vocab, disallowed_follows, ignore_terminal, None, shared_vocab_dfa_cache, None)
}

pub(crate) fn analyze_equivalences_with_group_filter(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    active_groups: Option<&[bool]>,
    shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
    flat_trans: Option<&[u32]>,
) -> InternalIdMap {
    analyze_equivalences_impl(tokenizer, vocab, disallowed_follows, ignore_terminal, active_groups, shared_vocab_dfa_cache, flat_trans)
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
    flat_trans: Option<&[u32]>,
) -> InternalIdMap {
    let profile_compile = compile_profile_enabled();
    let total_started_at = std::time::Instant::now();

    let adjust_started_at = std::time::Instant::now();
    let adjusted_disallowed = adjust_disallowed_follows(disallowed_follows, ignore_terminal);
    let effective_disallowed = adjusted_disallowed.as_ref().unwrap_or(disallowed_follows);
    let adjust_ms = elapsed_ms(adjust_started_at);

    let tokenizer_view_started_at = std::time::Instant::now();
    let tokenizer_view = match (active_groups, flat_trans) {
        (Some(active_groups), Some(ft)) => TokenizerView::new_filtered_from_flat_trans(ft, tokenizer, active_groups),
        (Some(active_groups), None) => TokenizerView::new_filtered(tokenizer, active_groups),
        _ => TokenizerView::new(tokenizer),
    };
    let tokenizer_view_ms = elapsed_ms(tokenizer_view_started_at);

    // Extract vocab tokens as byte slices, ordered by token ID.
    let vocab_extract_started_at = std::time::Instant::now();
    let max_token_id = vocab.max_token_id();
    let mut token_bytes: Vec<&[u8]> = Vec::with_capacity(vocab.len());
    let mut token_ids: Vec<u32> = Vec::with_capacity(vocab.len());
    for (&tid, bytes) in &vocab.entries {
        token_ids.push(tid);
        token_bytes.push(bytes.as_slice());
    }
    let vocab_extract_ms = elapsed_ms(vocab_extract_started_at);

    // All DFA states as initial states
    let initial_states_started_at = std::time::Instant::now();
    let initial_states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();
    let initial_states_ms = elapsed_ms(initial_states_started_at);

    let combined_started_at = std::time::Instant::now();
    let result = combined_equivalence_analysis::compute_combined_equivalence_with_group_filter(
        &tokenizer_view,
        &token_bytes,
        &initial_states,
        effective_disallowed,
        ignore_terminal,
        active_groups,
        shared_vocab_dfa_cache,
    );
    let combined_ms = elapsed_ms(combined_started_at);

    let state_map_started_at = std::time::Instant::now();
    let num_dfa_states = tokenizer.num_states() as usize;
    let state_map = build_state_map(&result.state_classes, num_dfa_states);
    let state_map_ms = elapsed_ms(state_map_started_at);

    let vocab_map_started_at = std::time::Instant::now();
    let vocab_map = build_vocab_map(
        &result.vocab_classes,
        &token_ids,
        max_token_id,
    );
    let vocab_map_ms = elapsed_ms(vocab_map_started_at);

    if profile_compile {
        eprintln!(
            "[glrmask/profile][id_map] adjust_disallowed_ms={:.3} tokenizer_view_ms={:.3} vocab_extract_ms={:.3} initial_states_ms={:.3} combined_equiv_ms={:.3} build_state_map_ms={:.3} build_vocab_map_ms={:.3} total_ms={:.3}",
            adjust_ms,
            tokenizer_view_ms,
            vocab_extract_ms,
            initial_states_ms,
            combined_ms,
            state_map_ms,
            vocab_map_ms,
            elapsed_ms(total_started_at),
        );
    }

    InternalIdMap {
        tokenizer_states: state_map,
        vocab_tokens: vocab_map,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::compile::build_tokenizer;
    use crate::compiler::grammar::model::{GrammarDef, Rule, Symbol, Terminal};

    #[test]
    fn test_internal_id_map_shape() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..Default::default()
        };
        let tok = build_tokenizer(&gdef);
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"a".to_vec()),
                (2, b"b".to_vec()),
            ],
            None,
        );
        let id_map = analyze_equivalences(&tok, &vocab, &BTreeMap::new(), None, None);

        assert!(id_map.num_tsids() >= 1);
        assert_eq!(id_map.max_token_id(), 2);
    }

    #[test]
    fn test_json_schema_equivalence_classes() {
        use crate::import::json_schema::json_schema_to_grammar;

        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }"#;
        let grammar = json_schema_to_grammar(schema).expect("Schema should convert");
        let tok = build_tokenizer(&grammar);
        let vocab_strs = vec![
            "{", "}", "\"", ":", ",", "n", "a", "m", "e", "s", "t", "r", "i", "g",
            "{\"", "\":",
        ];
        let vocab_entries: Vec<(u32, Vec<u8>)> = vocab_strs
            .iter()
            .enumerate()
            .map(|(i, s)| (i as u32, s.as_bytes().to_vec()))
            .collect();
        let vocab = Vocab::new(vocab_entries, None);
        let id_map = analyze_equivalences(&tok, &vocab, &BTreeMap::new(), None, None);
        let classes = &id_map.vocab_tokens.internal_to_originals;
        let expected: Vec<Vec<usize>> = vec![
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![4],
            vec![5],
            vec![6],
            vec![7],
            vec![8],
            vec![9],
            vec![10],
            vec![11],
            vec![12, 13],
            vec![14],
            vec![15],
        ];
        let mut expected_sorted: Vec<Vec<usize>> = expected
            .iter()
            .map(|class| {
                let mut sorted = class.clone();
                sorted.sort();
                sorted
            })
            .collect();
        expected_sorted.sort();
        let mut actual_sorted: Vec<Vec<usize>> = classes
            .iter()
            .map(|class| {
                let mut sorted: Vec<usize> = class.iter().map(|&id| id as usize).collect();
                sorted.sort();
                sorted
            })
            .collect();
        actual_sorted.sort();
        assert_eq!(
            actual_sorted,
            expected_sorted,
            "Equivalence classes don't match expected!\nExpected: {:?}\nActual:   {:?}",
            expected_sorted,
            actual_sorted,
        );
    }

    #[test]
    fn test_json_schema_equivalence_classes_simpler() {
        let grammar = crate::import::ebnf::parse_ebnf("root ::= '{' '}'")
            .expect("Grammar should build");
        let tok = build_tokenizer(&grammar);
        let vocab_entries = vec![(0, b"{".to_vec()), (1, b"}".to_vec())];
        let vocab = Vocab::new(vocab_entries, None);
        let id_map = analyze_equivalences(&tok, &vocab, &BTreeMap::new(), None, None);
        let classes = &id_map.vocab_tokens.internal_to_originals;
        let expected = vec![vec![0], vec![1]];
        let mut expected_sorted: Vec<Vec<usize>> = expected
            .into_iter()
            .map(|mut class| {
                class.sort();
                class
            })
            .collect();
        expected_sorted.sort();
        let mut actual_sorted: Vec<Vec<usize>> = classes
            .iter()
            .map(|class| {
                let mut sorted: Vec<usize> = class.iter().map(|&id| id as usize).collect();
                sorted.sort();
                sorted
            })
            .collect();
        actual_sorted.sort();
        assert_eq!(
            actual_sorted,
            expected_sorted,
            "Equivalence classes don't match expected!\nExpected: {:?}\nActual:   {:?}",
            expected_sorted,
            actual_sorted,
        );
    }
}

