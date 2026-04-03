//! Max-length bounded state equivalence prepass.
//!
//! Depth 0 hashes each state's own label data plus its outgoing label layout.
//! Each later depth only mixes in the previous hashes of that state's unique
//! outgoing destinations.

use rayon::prelude::*;

use super::super::compat::{FlatDfa, FlatDfaState, TokenizerView};

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

fn build_state_shape(dfa: &FlatDfa, state_idx: usize) -> (Vec<usize>, u64) {
    let mut targets: Vec<usize> = Vec::new();
    let mut label_hashes: Vec<u64> = Vec::new();
    let trans = dfa.transitions_for(state_idx);

    for (byte, &target) in trans.iter().enumerate() {
        if target == u32::MAX {
            continue;
        }

        let byte_hash = mix_u64((byte as u64) ^ 0xD6E8_FD93_5E6C_A271);
        let target = target as usize;

        if let Some(index) = targets.iter().position(|&t| t == target) {
            label_hashes[index] = label_hashes[index].wrapping_add(byte_hash);
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

/// Flattened outgoing targets for all states in contiguous memory.
/// `targets_flat[offsets[i]..offsets[i+1]]` gives the outgoing targets for state `i`.
struct FlatTargets {
    targets: Vec<usize>,
    offsets: Vec<u32>,
}

impl FlatTargets {
    fn from_vec_of_vecs(per_state: Vec<Vec<usize>>) -> Self {
        let n = per_state.len();
        let total: usize = per_state.iter().map(|v| v.len()).sum();
        let mut targets = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(n + 1);
        for v in per_state {
            offsets.push(targets.len() as u32);
            targets.extend_from_slice(&v);
        }
        offsets.push(targets.len() as u32);
        Self { targets, offsets }
    }

    #[inline(always)]
    fn targets_for(&self, state: usize) -> &[usize] {
        let start = self.offsets[state] as usize;
        let end = self.offsets[state + 1] as usize;
        &self.targets[start..end]
    }
}

#[allow(dead_code)]
#[inline(always)]
fn hash_transition_targets_flat(targets: &[usize], prev_hashes: &[u64]) -> u64 {
    let mut hash = mix_u64((targets.len() as u64) ^ 0xA5A5_A5A5_5A5A_5A5A);
    for &target in targets {
        hash = mix_u64(hash ^ prev_hashes[target].rotate_left(17));
    }
    hash
}

/// Fast commutative hash of target hashes for the kstep inner loop.
///
/// Unlike `hash_transition_targets_flat` which has a serial dependency chain
/// (each mix_u64 depends on the previous), this uses commutative accumulation
/// so all target loads can be issued and completed in parallel by the CPU.
/// The serial chain in hash_transition_targets_flat is:
///   mix(mix(mix(... ^ load[0]) ^ load[1]) ^ load[2])
/// requiring ~8 cycles per target (mix latency). With 10 targets, that's ~80 cycles.
///
/// This version issues all loads independently and accumulates with wrapping_add,
/// then finalizes with a single mix. Total: ~load_latency + ~6 cycles.
#[inline(always)]
fn hash_transition_targets_fast(targets: &[usize], prev_hashes: &[u64]) -> u64 {
    let mut acc: u64 = 0;
    for &target in targets {
        acc = acc.wrapping_add(prev_hashes[target]);
    }
    mix_u64(acc ^ (targets.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn count_distinct_hashes(hashes: &[u64]) -> usize {
    let mut seen = std::collections::HashSet::with_capacity(hashes.len());
    for &h in hashes {
        seen.insert(h);
    }
    seen.len()
}

/// Run kstep hash propagation.
///
/// Each iteration refines hash signatures across all states.
/// Uses fast commutative target hashing for CPU load pipelining.
fn run_kstep_parallel(
    flat: &FlatTargets,
    prev_hashes: &mut Vec<u64>,
    k: usize,
) -> usize {
    let n = prev_hashes.len();
    if n == 0 || k == 0 {
        return 0;
    }

    let mut next_hashes = vec![0u64; n];

    for _step in 0..k {
        for state_id in 0..n {
            next_hashes[state_id] = mix_u64(
                prev_hashes[state_id]
                    ^ hash_transition_targets_fast(flat.targets_for(state_id), prev_hashes),
            );
        }
        std::mem::swap(prev_hashes, &mut next_hashes);
    }

    k
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
    let (outgoing_targets_vec, mut prev_hashes): (Vec<_>, Vec<_>) = state_shapes.into_iter().unzip();
    let flat = FlatTargets::from_vec_of_vecs(outgoing_targets_vec);

    run_kstep_parallel(&flat, &mut prev_hashes, k);

    build_subset_mapping(states, &prev_hashes)
}

/// Compute a cheap hash of a state's transitions without allocating a HashMap.
/// This is used as a fast precheck: if all states have unique cheap hashes,
/// the expensive `build_state_shape` + kstep iteration can be skipped entirely.
#[inline]
fn cheap_state_hash(dfa: &FlatDfa, state_idx: usize) -> u64 {
    let label = hash_state_label(&dfa.states[state_idx]);
    let mut transition_hash: u64 = 0;
    let trans = dfa.transitions_for(state_idx);
    for (byte, &target) in trans.iter().enumerate() {
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
    let trans = dfa.transitions_for(state_idx);

    for (byte, &target) in trans.iter().enumerate() {
        if target == u32::MAX || !relevant_bytes[byte] {
            continue;
        }

        let byte_hash = mix_u64((byte as u64) ^ 0xD6E8_FD93_5E6C_A271);
        let target = target as usize;

        if let Some(index) = targets.iter().position(|&t| t == target) {
            label_hashes[index] = label_hashes[index].wrapping_add(byte_hash);
        } else {
            targets.push(target);
            label_hashes.push(byte_hash);
        }
    }

    (targets, hash_transition_labels(&label_hashes))
}

/// Build a state shape using precomputed byte-class representatives.
///
/// Each entry in `active_classes` is `(representative_byte, combined_hash)`.
/// All bytes in a byte class have identical transitions from every state,
/// so we only need to look up one representative byte per class.
fn build_state_shape_by_class(
    dfa: &FlatDfa,
    state_idx: usize,
    active_classes: &[(u8, u64)],
) -> (Vec<usize>, u64) {
    let mut targets: Vec<usize> = Vec::new();
    let mut label_hashes: Vec<u64> = Vec::new();

    for &(rep_byte, class_hash) in active_classes {
        let target = dfa.trans(state_idx, rep_byte as usize);
        if target == u32::MAX {
            continue;
        }
        let target = target as usize;

        // Linear search for target — faster than HashMap for small target counts
        if let Some(index) = targets.iter().position(|&t| t == target) {
            label_hashes[index] = label_hashes[index].wrapping_add(class_hash);
        } else {
            targets.push(target);
            label_hashes.push(class_hash);
        }
    }

    (targets, hash_transition_labels(&label_hashes))
}

/// Precompute active byte classes: for each DFA byte class that contains at
/// least one relevant byte, compute a representative byte (for transition
/// lookup) and the combined hash of all relevant bytes in that class.
fn precompute_active_classes(
    byte_to_class: &[u8; 256],
    relevant_bytes: &[bool; 256],
) -> Vec<(u8, u64)> {
    let num_classes = *byte_to_class.iter().max().unwrap_or(&0) as usize + 1;
    let mut class_rep: Vec<Option<u8>> = vec![None; num_classes];
    let mut class_hash: Vec<u64> = vec![0u64; num_classes];

    for byte in 0..256u16 {
        let b = byte as u8;
        if !relevant_bytes[b as usize] {
            continue;
        }
        let class = byte_to_class[b as usize] as usize;
        if class_rep[class].is_none() {
            class_rep[class] = Some(b);
        }
        class_hash[class] = class_hash[class].wrapping_add(
            mix_u64((b as u64) ^ 0xD6E8_FD93_5E6C_A271),
        );
    }

    (0..num_classes)
        .filter_map(|c| {
            class_rep[c].map(|rep| (rep, class_hash[c]))
        })
        .collect()
}

fn find_state_equivalence_classes_kstep_restricted(
    tokenizer: &TokenizerView,
    states: &[usize],
    k: usize,
    relevant_bytes: &[bool; 256],
    byte_to_class: Option<&[u8; 256]>,
) -> Vec<usize> {
    if states.is_empty() {
        return Vec::new();
    }

    let profile = std::env::var("GLRMASK_PROFILE_COMPILE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let shapes_start = std::time::Instant::now();

    let dfa = tokenizer.dfa();
    let num_relevant = relevant_bytes.iter().filter(|&&b| b).count();
    let active_classes = byte_to_class.map(|btc| precompute_active_classes(btc, relevant_bytes));
    let state_shapes: Vec<(Vec<usize>, u64)> = (0..dfa.states.len())
        .into_par_iter()
        .map(|s| {
            let (targets, transition_label_hash) = if let Some(ref classes) = active_classes {
                build_state_shape_by_class(dfa, s, classes)
            } else {
                build_state_shape_restricted(dfa, s, relevant_bytes)
            };
            (targets, mix_u64(hash_state_label(&dfa.states[s]) ^ transition_label_hash))
        })
        .collect();
    let (outgoing_targets_vec, mut prev_hashes): (Vec<_>, Vec<_>) = state_shapes.into_iter().unzip();
    let flat = FlatTargets::from_vec_of_vecs(outgoing_targets_vec);

    let shapes_ms = shapes_start.elapsed().as_secs_f64() * 1000.0;
    let kstep_start = std::time::Instant::now();

    let converged_at = run_kstep_parallel(&flat, &mut prev_hashes, k);

    if profile {
        let kstep_ms = kstep_start.elapsed().as_secs_f64() * 1000.0;
        let distinct = count_distinct_hashes(&prev_hashes);
        eprintln!(
            "[glrmask/profile][max_length_kstep] dfa_states={} k={} converged_at={} relevant_bytes={} distinct={} shapes_ms={:.3} kstep_ms={:.3}",
            dfa.states.len(), k, converged_at, num_relevant, distinct, shapes_ms, kstep_ms,
        );
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
    byte_to_class: Option<&[u8; 256]>,
) -> Vec<usize> {
    let max_len = tokens.iter().map(|token| token.as_ref().len()).max().unwrap_or(0);
    let k = max_len;

    let mut relevant_bytes = [false; 256];
    for token in tokens {
        for &b in token.as_ref() {
            relevant_bytes[b as usize] = true;
        }
    }

    find_state_equivalence_classes_kstep_restricted(tokenizer, states, k, &relevant_bytes, byte_to_class)
}
