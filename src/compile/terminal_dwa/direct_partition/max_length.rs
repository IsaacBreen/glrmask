//! Max-length bounded state equivalence prepass.
//!
//! Depth 0 hashes each state's own label data plus its outgoing label layout.
//! Each later depth only mixes in the previous hashes of that state's unique
//! outgoing destinations.

use rayon::prelude::*;

use super::super::pair_partition::equivalence_analysis::compat::{FlatDfa, FlatDfaState, TokenizerView};

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
fn hash_filtered_sorted_set(values: &[usize], active_groups: Option<&[bool]>, tag: u64) -> u64 {
    let active_len = values
        .iter()
        .filter(|&&value| active_groups.map_or(true, |groups| groups.get(value).copied().unwrap_or(false)))
        .count();
    let mut hash = mix_u64((active_len as u64) ^ tag);
    for &value in values {
        if !active_groups.map_or(true, |groups| groups.get(value).copied().unwrap_or(false)) {
            continue;
        }
        hash = hash.wrapping_add(mix_u64((value as u64) ^ tag.rotate_left(17)));
    }
    hash
}

#[inline(always)]
fn hash_state_label(state: &FlatDfaState, active_groups: Option<&[bool]>) -> u64 {
    let finalizers = hash_filtered_sorted_set(&state.finalizers, active_groups, 0xF11A_F11A_F11A_F11A);
    let futures = hash_filtered_sorted_set(
        &state.possible_future_group_ids,
        active_groups,
        0xF0C7_F0C7_F0C7_F0C7,
    );
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

fn build_state_shape(
    dfa: &FlatDfa,
    state_idx: usize,
    relevant_bytes: Option<&[bool; 256]>,
) -> (Vec<usize>, u64) {
    let mut targets: Vec<usize> = Vec::new();
    let mut label_hashes: Vec<u64> = Vec::new();

    for (byte, &target) in dfa.transitions_for(state_idx).iter().enumerate() {
        if target == u32::MAX || relevant_bytes.is_some_and(|bytes| !bytes[byte]) {
            continue;
        }

        let byte_hash = mix_u64((byte as u64) ^ 0xD6E8_FD93_5E6C_A271);
        let target = target as usize;

        if let Some(pos) = targets.iter().position(|&t| t == target) {
            label_hashes[pos] = label_hashes[pos].wrapping_add(byte_hash);
        } else {
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
    active_groups: Option<&[bool]>,
    relevant_bytes: Option<&[bool; 256]>,
) -> Vec<usize> {
    if states.is_empty() {
        return Vec::new();
    }

    let dfa = tokenizer.dfa();
    let state_shapes: Vec<(Vec<usize>, u64)> = (0..dfa.states.len())
        .into_par_iter()
        .map(|s| {
            let (targets, transition_label_hash) = build_state_shape(dfa, s, relevant_bytes);
            (
                targets,
                mix_u64(hash_state_label(&dfa.states[s], active_groups) ^ transition_label_hash),
            )
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
fn cheap_state_hash(
    dfa: &FlatDfa,
    state_idx: usize,
    active_groups: Option<&[bool]>,
    relevant_bytes: Option<&[bool; 256]>,
) -> u64 {
    let label = hash_state_label(&dfa.states[state_idx], active_groups);
    let mut transition_hash: u64 = 0;
    for (byte, &target) in dfa.transitions_for(state_idx).iter().enumerate() {
        if target != u32::MAX && relevant_bytes.is_none_or(|bytes| bytes[byte]) {
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
    active_groups: Option<&[bool]>,
    relevant_bytes: Option<&[bool; 256]>,
) -> Vec<usize> {
    let max_len = tokens.iter().map(|token| token.as_ref().len()).max().unwrap_or(0);
    let mapping = find_state_equivalence_classes_kstep(
        tokenizer,
        states,
        max_len,
        active_groups,
        relevant_bytes,
    );

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

    for (byte, &target) in dfa.transitions_for(state_idx).iter().enumerate() {
        if target == u32::MAX || !relevant_bytes[byte] {
            continue;
        }

        let byte_hash = mix_u64((byte as u64) ^ 0xD6E8_FD93_5E6C_A271);
        let target = target as usize;

        if let Some(pos) = targets.iter().position(|&t| t == target) {
            label_hashes[pos] = label_hashes[pos].wrapping_add(byte_hash);
        } else {
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
    active_groups: Option<&[bool]>,
) -> Vec<usize> {
    if states.is_empty() {
        return Vec::new();
    }

    let t0 = std::time::Instant::now();
    let dfa = tokenizer.dfa();
    let num_states = dfa.states.len();

    // ── Reachability: only process states reachable from any state in `states`
    //    via the restricted byte set within k transitions.
    let relevant_byte_list: Vec<u8> = (0..=255u8).filter(|&b| relevant_bytes[b as usize]).collect();
    let mut reachable = vec![false; num_states];
    let mut frontier: Vec<usize> = Vec::new();
    for &s in states {
        if s < num_states && !reachable[s] {
            reachable[s] = true;
            frontier.push(s);
        }
    }
    for _depth in 0..k {
        let mut next_frontier = Vec::new();
        for &state in &frontier {
            for &byte in &relevant_byte_list {
                let target = dfa.trans(state, byte as usize);
                if target != u32::MAX {
                    let t = target as usize;
                    if !reachable[t] {
                        reachable[t] = true;
                        next_frontier.push(t);
                    }
                }
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    // Collect reachable state indices and build dense mapping.
    let mut reachable_indices: Vec<usize> = Vec::new();
    // sparse_to_dense[original_state] = index in reachable_indices (or u32::MAX)
    let mut sparse_to_dense = vec![u32::MAX; num_states];
    for s in 0..num_states {
        if reachable[s] {
            sparse_to_dense[s] = reachable_indices.len() as u32;
            reachable_indices.push(s);
        }
    }
    let n_reachable = reachable_indices.len();
    let reachability_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // ── Build shapes for reachable states only, using CSR (flat) target storage.
    // Targets are stored as dense indices into `reachable_indices`.
    // Unreachable targets are mapped to a sentinel dense index.
    let unreachable_sentinel = n_reachable; // one past the last valid dense index
    let mut target_data: Vec<u32> = Vec::new();
    let mut target_offsets: Vec<u32> = Vec::with_capacity(n_reachable + 1);
    let mut dense_hashes: Vec<u64> = Vec::with_capacity(n_reachable);

    for &s in &reachable_indices {
        target_offsets.push(target_data.len() as u32);

        let mut targets_local: Vec<u32> = Vec::new();
        let mut label_hashes_local: Vec<u64> = Vec::new();

        for (byte, &target) in dfa.transitions_for(s).iter().enumerate() {
            if target == u32::MAX || !relevant_bytes[byte] {
                continue;
            }
            let byte_hash = mix_u64((byte as u64) ^ 0xD6E8_FD93_5E6C_A271);
            let dense_target = if reachable[target as usize] {
                sparse_to_dense[target as usize]
            } else {
                unreachable_sentinel as u32
            };

            if let Some(pos) = targets_local.iter().position(|&t| t == dense_target) {
                label_hashes_local[pos] = label_hashes_local[pos].wrapping_add(byte_hash);
            } else {
                targets_local.push(dense_target);
                label_hashes_local.push(byte_hash);
            }
        }

        let transition_label_hash = hash_transition_labels(&label_hashes_local);
        target_data.extend_from_slice(&targets_local);
        dense_hashes.push(mix_u64(
            hash_state_label(&dfa.states[s], active_groups) ^ transition_label_hash,
        ));
    }
    target_offsets.push(target_data.len() as u32);

    let shape_ms = t0.elapsed().as_secs_f64() * 1000.0 - reachability_ms;

    // ── kstep iteration on dense reachable set + one sentinel slot.
    // prev_hashes has n_reachable + 1 slots: [0..n_reachable) for reachable states,
    // [n_reachable] = sentinel hash for all unreachable targets.
    let sentinel_hash: u64 = 0;
    let mut prev_dense_hashes = dense_hashes;
    prev_dense_hashes.push(sentinel_hash);
    let mut next_dense_hashes = vec![0u64; n_reachable + 1];

    let check_interval = 16.min(k);
    let mut prev_distinct = 0usize;
    for step in 0..k {
        // Single-threaded iteration is faster for typical reachable set sizes
        // (avoids rayon scheduling overhead across 128 iterations).
        for dense_id in 0..n_reachable {
            let start = target_offsets[dense_id] as usize;
            let end = target_offsets[dense_id + 1] as usize;
            let targets = &target_data[start..end];
            let mut target_hash: u64 = mix_u64((targets.len() as u64) ^ 0xA5A5_A5A5_5A5A_5A5A);
            for &t in targets {
                target_hash = mix_u64(target_hash ^ prev_dense_hashes[t as usize].rotate_left(17));
            }
            next_dense_hashes[dense_id] = mix_u64(prev_dense_hashes[dense_id] ^ target_hash);
        }
        next_dense_hashes[n_reachable] = sentinel_hash; // sentinel unchanged
        std::mem::swap(&mut prev_dense_hashes, &mut next_dense_hashes);
        if (step + 1) % check_interval == 0 {
            let distinct = count_distinct_hashes(&prev_dense_hashes[..n_reachable]);
            if distinct == prev_distinct {
                break;
            }
            prev_distinct = distinct;
        }
    }
    let iter_ms = t0.elapsed().as_secs_f64() * 1000.0 - reachability_ms - shape_ms;

    // ── Map dense hashes back to sparse and build result.
    // All unreachable states get the sentinel hash (ensuring they form one class).
    let mut full_hashes = vec![sentinel_hash; num_states];
    for (dense_id, &orig_state) in reachable_indices.iter().enumerate() {
        full_hashes[orig_state] = prev_dense_hashes[dense_id];
    }

    let result = build_subset_mapping(states, &full_hashes);
    result
}

/// Byte-restricted state equivalence: merge states that behave identically
/// under the subset of bytes actually used by the partition's tokens.
///
/// This is much more effective than unrestricted k-step when the partition
/// tokens use a small alphabet (e.g. only alphanumeric bytes).
///
/// DO NOT cap or reduce k below `max_len`.  See the pair-partition variant's doc
/// comment for the full rationale: kstep merges are permanent and no
/// downstream stage re-splits them.
pub fn find_state_equivalence_classes_byte_restricted<S: AsRef<[u8]>>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    active_groups: Option<&[bool]>,
    relevant_bytes: Option<&[bool; 256]>,
) -> Vec<usize> {
    let max_len = tokens.iter().map(|token| token.as_ref().len()).max().unwrap_or(0);

    let derived_relevant_bytes;
    let relevant_bytes = match relevant_bytes {
        Some(bytes) => bytes,
        None => {
            let mut bytes = [false; 256];
            for token in tokens {
                for &b in token.as_ref() {
                    bytes[b as usize] = true;
                }
            }
            derived_relevant_bytes = bytes;
            &derived_relevant_bytes
        }
    };

    find_state_equivalence_classes_kstep_restricted(
        tokenizer,
        states,
        max_len,
        relevant_bytes,
        active_groups,
    )
}
