//! Max-length bounded state equivalence prepass.
//!
//! Depth 0 hashes each state's own label data plus its outgoing label layout.
//! Each later depth only mixes in the previous hashes of that state's unique
//! outgoing destinations.

use std::collections::HashMap;

use rayon::prelude::*;

use super::super::l2p::equivalence_analysis::compat::{FlatDfa, FlatDfaState, TokenizerView};

fn debug_max_length_enabled() -> bool {
    std::env::var("GLRMASK_DEBUG_MAX_LENGTH")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

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
    let finalizers = hash_sorted_set(&state.finalizers, 0xF11A_F11A_F11A_F11A);
    let futures = hash_sorted_set(&state.possible_future_group_ids, 0xF0C7_F0C7_F0C7_F0C7);
    mix_u64(finalizers.wrapping_add(futures))
}

#[inline(always)]
fn hash_transition_labels(label_hashes: &[u64]) -> u64 {
    let mut hash = mix_u64((label_hashes.len() as u64) ^ 0x7F4A_7C15_9E37_79B9);
    for &label_hash in label_hashes {
        hash = mix_u64(hash ^ label_hash.wrapping_add(0xA24B_AED4_963E_E407));
    }
    hash
}

#[inline(always)]
fn hash_transition_targets(targets: &[usize], prev_hashes: &[u64]) -> u64 {
    let mut hash = mix_u64((targets.len() as u64) ^ 0xA5A5_A5A5_5A5A_5A5A);
    for &target in targets {
        hash = mix_u64(hash ^ prev_hashes[target].rotate_left(17));
    }
    hash
}

fn build_state_shape(dfa: &FlatDfa, state_idx: usize) -> (Vec<usize>, u64) {
    let mut targets: Vec<usize> = Vec::new();
    let mut label_hashes: Vec<u64> = Vec::new();
    let mut index_by_target: HashMap<usize, usize> = HashMap::new();

    for (byte, &target) in dfa.transitions_for(state_idx).iter().enumerate() {
        if target == u32::MAX {
            continue;
        }

        let byte_hash = mix_u64((byte as u64) ^ 0xD6E8_FD93_5E6C_A271);
        let target = target as usize;

        if let Some(&index) = index_by_target.get(&target) {
            label_hashes[index] = label_hashes[index].wrapping_add(byte_hash);
        } else {
            index_by_target.insert(target, targets.len());
            targets.push(target);
            label_hashes.push(byte_hash);
        }
    }

    (targets, hash_transition_labels(&label_hashes))
}

fn build_subset_mapping(states: &[usize], hashes: &[u64]) -> Vec<usize> {
    let mut indexed_hashes: Vec<(u64, usize, usize)> = states
        .par_iter()
        .enumerate()
        .map(|(position, &state_id)| (hashes[state_id], state_id, position))
        .collect();
    indexed_hashes.par_sort_unstable_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });

    let mut mapping = vec![0usize; states.len()];
    let mut current_hash = None;
    let mut current_rep = 0usize;
    for (hash, state_id, position) in indexed_hashes {
        if current_hash != Some(hash) {
            current_hash = Some(hash);
            current_rep = state_id;
        }
        mapping[position] = current_rep;
    }

    mapping
}

fn count_distinct_hashes(hashes: &[u64]) -> usize {
    let mut seen = std::collections::HashSet::with_capacity(hashes.len());
    for &h in hashes {
        seen.insert(h);
    }
    seen.len()
}

fn find_state_equivalence_classes_kstep(
    tokenizer: &TokenizerView,
    states: &[usize],
    k: usize,
) -> Vec<usize> {
    if states.is_empty() {
        return Vec::new();
    }

    let dfa = tokenizer.dfa();
    let state_shapes: Vec<(Vec<usize>, u64)> = (0..dfa.states.len())
        .into_par_iter()
        .map(|s| {
            let (targets, transition_label_hash) = build_state_shape(dfa, s);
            (targets, mix_u64(hash_state_label(&dfa.states[s]) ^ transition_label_hash))
        })
        .collect();
    let (outgoing_targets, mut prev_hashes): (Vec<_>, Vec<_>) = state_shapes.into_iter().unzip();

    let check_interval = 16.min(k);
    let mut prev_distinct = 0usize;
    let mut next_hashes = vec![0u64; dfa.states.len()];
    for step in 0..k {
        next_hashes
            .par_iter_mut()
            .enumerate()
            .for_each(|(state_id, next_hash)| {
                *next_hash = mix_u64(
                    prev_hashes[state_id]
                        ^ hash_transition_targets(&outgoing_targets[state_id], &prev_hashes),
                );
            });
        std::mem::swap(&mut prev_hashes, &mut next_hashes);
        if (step + 1) % check_interval == 0 {
            let distinct = count_distinct_hashes(&prev_hashes);
            if distinct == prev_distinct {
                break;
            }
            prev_distinct = distinct;
        }
    }

    build_subset_mapping(states, &prev_hashes)
}

/// Compute a cheap hash of a state's transitions without allocating a HashMap.
/// This is used as a fast precheck: if all states have unique cheap hashes,
/// the expensive `build_state_shape` + kstep iteration can be skipped entirely.
#[inline]
fn cheap_state_hash(dfa: &FlatDfa, state_idx: usize) -> u64 {
    let label = hash_state_label(&dfa.states[state_idx]);
    let mut transition_hash: u64 = 0;
    for (byte, &target) in dfa.transitions_for(state_idx).iter().enumerate() {
        if target != u32::MAX {
            transition_hash = transition_hash.wrapping_add(
                mix_u64((byte as u64) ^ ((target as u64) << 16) ^ 0xD6E8_FD93_5E6C_A271),
            );
        }
    }
    mix_u64(label ^ transition_hash)
}

pub fn find_state_equivalence_classes<S: AsRef<[u8]>>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
) -> Vec<usize> {
    let max_len = tokens.iter().map(|token| token.as_ref().len()).max().unwrap_or(0);
    let mapping = find_state_equivalence_classes_kstep(tokenizer, states, max_len);

    if debug_max_length_enabled() {
        let mut representatives = mapping.clone();
        representatives.sort_unstable();
        representatives.dedup();
        eprintln!(
            "[glrmask/debug][max_length] max_token_len={} input_states={} tokenizer_dfa_states={} representative_states={}",
            max_len,
            states.len(),
            tokenizer.dfa().states.len(),
            representatives.len(),
        );
    }

    mapping
}

/// Build a state shape considering only a restricted set of bytes.
///
/// This is used for pre-vocab state reduction: tokens in a given partition
/// only contain a subset of bytes (e.g. alnum tokens use only a-z, A-Z, 0-9).
/// States that differ only on transitions for irrelevant bytes can be merged.
fn build_state_shape_restricted(dfa: &FlatDfa, state_idx: usize, relevant_bytes: &[bool; 256]) -> (Vec<usize>, u64) {
    let mut targets: Vec<usize> = Vec::new();
    let mut label_hashes: Vec<u64> = Vec::new();
    let mut index_by_target: HashMap<usize, usize> = HashMap::new();

    for (byte, &target) in dfa.transitions_for(state_idx).iter().enumerate() {
        if target == u32::MAX || !relevant_bytes[byte] {
            continue;
        }

        let byte_hash = mix_u64((byte as u64) ^ 0xD6E8_FD93_5E6C_A271);
        let target = target as usize;

        if let Some(&index) = index_by_target.get(&target) {
            label_hashes[index] = label_hashes[index].wrapping_add(byte_hash);
        } else {
            index_by_target.insert(target, targets.len());
            targets.push(target);
            label_hashes.push(byte_hash);
        }
    }

    (targets, hash_transition_labels(&label_hashes))
}

fn find_state_equivalence_classes_kstep_restricted(
    tokenizer: &TokenizerView,
    states: &[usize],
    k: usize,
    relevant_bytes: &[bool; 256],
) -> Vec<usize> {
    if states.is_empty() {
        return Vec::new();
    }

    let dfa = tokenizer.dfa();
    let state_shapes: Vec<(Vec<usize>, u64)> = (0..dfa.states.len())
        .into_par_iter()
        .map(|s| {
            let (targets, transition_label_hash) = build_state_shape_restricted(dfa, s, relevant_bytes);
            (targets, mix_u64(hash_state_label(&dfa.states[s]) ^ transition_label_hash))
        })
        .collect();
    let (outgoing_targets, mut prev_hashes): (Vec<_>, Vec<_>) = state_shapes.into_iter().unzip();

    let check_interval = 16.min(k);
    let mut prev_distinct = 0usize;
    let mut next_hashes = vec![0u64; dfa.states.len()];
    for step in 0..k {
        next_hashes
            .par_iter_mut()
            .enumerate()
            .for_each(|(state_id, next_hash)| {
                *next_hash = mix_u64(
                    prev_hashes[state_id]
                        ^ hash_transition_targets(&outgoing_targets[state_id], &prev_hashes),
                );
            });
        std::mem::swap(&mut prev_hashes, &mut next_hashes);
        if (step + 1) % check_interval == 0 {
            let distinct = count_distinct_hashes(&prev_hashes);
            if distinct == prev_distinct {
                break;
            }
            prev_distinct = distinct;
        }
    }

    build_subset_mapping(states, &prev_hashes)
}

/// Byte-restricted state equivalence: merge states that behave identically
/// under the subset of bytes actually used by the partition's tokens.
///
/// This is much more effective than unrestricted k-step when the partition
/// tokens use a small alphabet (e.g. only alphanumeric bytes).
pub fn find_state_equivalence_classes_byte_restricted<S: AsRef<[u8]>>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
) -> Vec<usize> {
    let max_len = tokens.iter().map(|token| token.as_ref().len()).max().unwrap_or(0);

    let mut relevant_bytes = [false; 256];
    for token in tokens {
        for &b in token.as_ref() {
            relevant_bytes[b as usize] = true;
        }
    }

    find_state_equivalence_classes_kstep_restricted(tokenizer, states, max_len, &relevant_bytes)
}
