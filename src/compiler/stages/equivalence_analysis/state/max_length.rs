//! Max-length bounded state equivalence prepass.
//!
//! Two states are equivalent up to length `k` iff for every byte string of
//! length `0..=k`, either:
//! - that path exists from both states and ends in states with identical
//!   `finalizers` and identical `possible_future_group_ids`, or
//! - that path exists from neither state.
//!
//! This implementation matches that definition modulo the usual extremely
//! unlikely possibility of 64-bit hash collisions.
//!
//! Refinement recurrence:
//! - depth 0 class = hash of the state's own label
//! - depth d+1 class = hash of:
//!     * the state's own label
//!     * the full byte -> depth-d class map of its outgoing transitions
//!       (with a distinguished dead class for missing transitions)
//!
//! Important correctness property:
//! - transitions are compared by byte -> previous-class behavior
//! - never by concrete target state id
//!
//! Performance notes:
//! - sparse transitions are precomputed once
//! - states with identical concrete transition patterns share a cached
//!   transition-hash computation each round
//! - refinement uses dense integer class IDs, not raw prior hashes

use std::collections::HashMap;

use rayon::prelude::*;

use super::super::compat::Sep1Tokenizer;

#[inline(always)]
fn mix_u64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

#[inline(always)]
fn hash_sorted_set(values: &[usize], tag: u64) -> u64 {
    let mut hash = mix_u64((values.len() as u64) ^ tag);
    for &value in values {
        hash = hash.wrapping_add(mix_u64((value as u64) ^ tag.rotate_left(17)));
    }
    hash
}

#[inline(always)]
fn hash_state_label(finalizers: &[usize], possible_futures: &[usize]) -> u64 {
    const FINALIZER_TAG: u64 = 0xF11A_F11A_F11A_F11A;
    const FUTURE_TAG: u64 = 0xF0C7_F0C7_F0C7_F0C7;

    let finalizer_hash = hash_sorted_set(finalizers, FINALIZER_TAG);
    let future_hash = hash_sorted_set(possible_futures, FUTURE_TAG);
    mix_u64(finalizer_hash.wrapping_add(future_hash))
}

#[inline(always)]
fn byte_tag(byte: u8) -> u64 {
    ((byte as u64) << 1) | 1
}

#[inline(always)]
fn build_classes_from_hashes(hashes: &[u64]) -> (Vec<usize>, usize) {
    let mut class_of_hash: HashMap<u64, usize> = HashMap::with_capacity(hashes.len());
    let mut classes = vec![0usize; hashes.len()];
    let mut next_class = 0usize;

    for (state_id, &hash) in hashes.iter().enumerate() {
        let class_id = match class_of_hash.get(&hash) {
            Some(&existing) => existing,
            None => {
                let new_id = next_class;
                class_of_hash.insert(hash, new_id);
                next_class += 1;
                new_id
            }
        };
        classes[state_id] = class_id;
    }

    (classes, next_class)
}

#[inline(always)]
fn build_subset_mapping(states: &[usize], classes: &[usize]) -> Vec<usize> {
    let mut rep_for_class: HashMap<usize, usize> = HashMap::new();
    let mut mapping = Vec::with_capacity(states.len());

    for &state_id in states {
        let rep = *rep_for_class.entry(classes[state_id]).or_insert(state_id);
        mapping.push(rep);
    }

    mapping
}

fn find_state_equivalence_classes_kstep(
    regex: &Sep1Tokenizer,
    states: &[usize],
    k: usize,
) -> Vec<usize> {
    if states.is_empty() {
        return Vec::new();
    }

    let dfa = regex.dfa();
    let num_states = dfa.states.len();

    if num_states == 0 {
        return states.to_vec();
    }

    // Sparse outgoing transitions in byte order.
    let transitions: Vec<Vec<(u8, usize)>> = dfa
        .states
        .iter()
        .map(|state| {
            state.transitions
                .iter()
                .enumerate()
                .filter(|(_, target)| **target != u32::MAX)
                .map(|(byte, &target)| (byte as u8, target as usize))
                .collect()
        })
        .collect();

    // Optional transition-pattern cache:
    // if many states share the exact same concrete transition pattern,
    // compute one transition hash per pattern class per refinement round.
    let mut class_reps: Vec<usize> = Vec::new();
    let class_for_state: Vec<usize>;
    let mut use_trans_class_cache = false;

    {
        let mut trans_pattern_to_class: HashMap<Vec<(u8, usize)>, usize> = HashMap::new();
        let mut tmp_class_for_state = vec![0usize; num_states];

        for (state_id, trans) in transitions.iter().enumerate() {
            let class_id = match trans_pattern_to_class.get(trans) {
                Some(&existing) => existing,
                None => {
                    let new_id = class_reps.len();
                    trans_pattern_to_class.insert(trans.clone(), new_id);
                    class_reps.push(state_id);
                    new_id
                }
            };
            tmp_class_for_state[state_id] = class_id;
        }

        let num_classes = class_reps.len();
        let shared = num_states.saturating_sub(num_classes);

        if num_states > 0 && shared * 2 >= num_states {
            use_trans_class_cache = true;
            class_for_state = tmp_class_for_state;
        } else {
            class_reps.clear();
            class_for_state = Vec::new();
        }
    }

    // Depth-0 label hashes.
    let label_hashes: Vec<u64> = dfa
        .states
        .iter()
        .map(|state| hash_state_label(&state.finalizers, &state.possible_future_group_ids))
        .collect();

    // Depth-0 classes: exact label equivalence modulo hash collisions.
    let (mut prev_classes, mut prev_num_classes) = build_classes_from_hashes(&label_hashes);

    if k == 0 {
        return build_subset_mapping(states, &prev_classes);
    }

    // Distinguished dead class contribution for missing transitions.
    const DEAD_CLASS_CODE: u64 = 0xDEAD_BEEF_DEAD_BEEF;

    let mut dead_byte_mix = [0u64; 256];
    let mut dead_base_sum = 0u64;
    for byte in 0u8..=255 {
        let contrib = mix_u64(DEAD_CLASS_CODE ^ byte_tag(byte));
        dead_byte_mix[byte as usize] = contrib;
        dead_base_sum = dead_base_sum.wrapping_add(contrib);
    }

    let mut next_hashes = vec![0u64; num_states];

    for _depth in 0..k {
        let compute_transition_hash = |state_id: usize, classes: &[usize]| -> u64 {
            let mut trans_sum = dead_base_sum;

            for &(byte, target) in &transitions[state_id] {
                let byte_idx = byte as usize;
                trans_sum = trans_sum.wrapping_sub(dead_byte_mix[byte_idx]);

                let child_class = classes[target] as u64;
                let contrib = mix_u64(child_class ^ byte_tag(byte));
                trans_sum = trans_sum.wrapping_add(contrib);
            }

            trans_sum
        };

        if use_trans_class_cache {
            let trans_hash_per_class: Vec<u64> = (0..class_reps.len())
                .into_par_iter()
                .map(|class_id| compute_transition_hash(class_reps[class_id], &prev_classes))
                .collect();

            next_hashes
                .par_iter_mut()
                .enumerate()
                .for_each(|(state_id, out)| {
                    let trans_hash = trans_hash_per_class[class_for_state[state_id]];
                    *out = mix_u64(
                        label_hashes[state_id]
                            ^ mix_u64(trans_hash ^ 0xA5A5_A5A5_5A5A_5A5A),
                    );
                });
        } else {
            next_hashes
                .par_iter_mut()
                .enumerate()
                .for_each(|(state_id, out)| {
                    let trans_hash = compute_transition_hash(state_id, &prev_classes);
                    *out = mix_u64(
                        label_hashes[state_id]
                            ^ mix_u64(trans_hash ^ 0xA5A5_A5A5_5A5A_5A5A),
                    );
                });
        }

        let (new_classes, new_num_classes) = build_classes_from_hashes(&next_hashes);

        // Refinement is monotone. Once the class count stops increasing,
        // the partition is stable and deeper rounds will not change it.
        if new_num_classes == prev_num_classes {
            prev_classes = new_classes;
            break;
        }

        prev_classes = new_classes;
        prev_num_classes = new_num_classes;
    }

    build_subset_mapping(states, &prev_classes)
}

pub fn find_state_equivalence_classes<S: AsRef<[u8]>>(
    regex: &Sep1Tokenizer,
    tokens: &[S],
    states: &[usize],
) -> Vec<usize> {
    let max_len = tokens
        .iter()
        .map(|token| token.as_ref().len())
        .max()
        .unwrap_or(0);

    find_state_equivalence_classes_kstep(regex, states, max_len)
}