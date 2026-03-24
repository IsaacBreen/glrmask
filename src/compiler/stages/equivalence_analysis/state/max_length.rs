//! Max-length bounded state equivalence prepass.
//!
//! This pass ignores actual token contents and uses only the maximum token
//! length. Two states remain equivalent at depth `d` iff they have the same
//! label and the same byte-to-previous-hash behavior for all paths up to
//! depth `d`.
//!
//! This implementation is intentionally simple:
//! - depth 0 hashes only each state's own label
//! - each later depth groups outgoing bytes by the previous hash of the target
//! - the final subset mapping is built from the depth-k hashes

use std::collections::HashMap;

use super::super::compat::{FlatDfaState, Sep1Tokenizer};

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
fn hash_state_label(state: &FlatDfaState) -> u64 {
    const FINALIZER_TAG: u64 = 0xF11A_F11A_F11A_F11A;
    const FUTURE_TAG: u64 = 0xF0C7_F0C7_F0C7_F0C7;

    let finalizers = hash_sorted_set(&state.finalizers, FINALIZER_TAG);
    let futures = hash_sorted_set(&state.possible_future_group_ids, FUTURE_TAG);
    mix_u64(finalizers.wrapping_add(futures))
}

#[inline(always)]
fn hash_transition_partition(state: &FlatDfaState, prev_hashes: &[u64], dead_hash: u64) -> u64 {
    // Group bytes by the previous hash of their destination.
    //
    // This is the key property needed for equivalence with the original:
    // transitions are partitioned by child hash class, not by concrete state id.
    let mut byte_sum_by_child_hash: HashMap<u64, u64> = HashMap::new();

    for (byte, &target) in state.transitions.iter().enumerate() {
        let child_hash = if target == u32::MAX {
            dead_hash
        } else {
            prev_hashes[target as usize]
        };

        let byte_hash = mix_u64((byte as u64) ^ 0xD6E8_FD93_5E6C_A271);

        byte_sum_by_child_hash
            .entry(child_hash)
            .and_modify(|acc| *acc = acc.wrapping_add(byte_hash))
            .or_insert(byte_hash);
    }

    // Make the result independent of HashMap iteration order.
    let mut buckets: Vec<(u64, u64)> = byte_sum_by_child_hash.into_iter().collect();
    buckets.sort_unstable_by_key(|&(child_hash, byte_sum)| (child_hash, byte_sum));

    let mut hash = mix_u64((buckets.len() as u64) ^ 0x7F4A_7C15_9E37_79B9);
    for (child_hash, byte_sum) in buckets {
        hash = mix_u64(
            hash ^ mix_u64(child_hash ^ 0xA24B_AED4_963E_E407)
                ^ mix_u64(byte_sum ^ 0xC3A5_C85C_97CB_3127),
        );
    }
    hash
}

fn build_subset_mapping(states: &[usize], hashes: &[u64]) -> Vec<usize> {
    let mut rep_for_hash: HashMap<u64, usize> = HashMap::new();
    let mut mapping = Vec::with_capacity(states.len());

    for &state_id in states {
        let rep = *rep_for_hash.entry(hashes[state_id]).or_insert(state_id);
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
    let dead_hash = mix_u64(0xDEAD_BEEF_DEAD_BEEF);

    // Depth 0: hash only the state's own label.
    let label_hashes: Vec<u64> = dfa.states.iter().map(hash_state_label).collect();

    if k == 0 {
        return build_subset_mapping(states, &label_hashes);
    }

    let mut prev_hashes = label_hashes.clone();
    let mut next_hashes = vec![0u64; dfa.states.len()];

    // Depth d > 0:
    // hash_d(state) = H(label(state), partition of bytes by hash_{d-1}(dest))
    for _depth in 0..k {
        for (state_id, state) in dfa.states.iter().enumerate() {
            let trans_hash = hash_transition_partition(state, &prev_hashes, dead_hash);
            next_hashes[state_id] =
                mix_u64(label_hashes[state_id] ^ mix_u64(trans_hash ^ 0xA5A5_A5A5_5A5A_5A5A));
        }
        std::mem::swap(&mut prev_hashes, &mut next_hashes);
    }

    build_subset_mapping(states, &prev_hashes)
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