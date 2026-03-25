//! Max-length bounded state equivalence prepass.
//!
//! Depth 0 hashes each state's own label data plus its outgoing label layout.
//! Each later depth only mixes in the previous hashes of that state's unique
//! outgoing destinations.

use std::collections::HashMap;

use super::super::compat::{FlatDfaState, TokenizerView};

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

fn build_state_shape(state: &FlatDfaState) -> (Vec<usize>, u64) {
    let mut targets: Vec<usize> = Vec::new();
    let mut label_hashes: Vec<u64> = Vec::new();
    let mut index_by_target: HashMap<usize, usize> = HashMap::new();

    for (byte, &target) in state.transitions.iter().enumerate() {
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
    let mut rep_for_hash = HashMap::new();
    let mut mapping = Vec::with_capacity(states.len());

    for &state_id in states {
        let rep = *rep_for_hash.entry(hashes[state_id]).or_insert(state_id);
        mapping.push(rep);
    }

    mapping
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
    let mut outgoing_targets = Vec::with_capacity(dfa.states.len());
    let mut prev_hashes = Vec::with_capacity(dfa.states.len());

    for state in &dfa.states {
        let (targets, transition_label_hash) = build_state_shape(state);
        outgoing_targets.push(targets);
        prev_hashes.push(mix_u64(hash_state_label(state) ^ transition_label_hash));
    }

    let mut next_hashes = vec![0u64; dfa.states.len()];
    for _ in 0..k {
        for state_id in 0..dfa.states.len() {
            next_hashes[state_id] = mix_u64(
                prev_hashes[state_id]
                    ^ hash_transition_targets(&outgoing_targets[state_id], &prev_hashes),
            );
        }
        std::mem::swap(&mut prev_hashes, &mut next_hashes);
    }

    build_subset_mapping(states, &prev_hashes)
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
