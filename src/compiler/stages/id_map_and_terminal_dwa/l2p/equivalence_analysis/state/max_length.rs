//! Max-length bounded state equivalence prepass.
//!
//! This implementation computes an exact Moore-style finite-depth partition
//! refinement over a filtered DFA view instead of using hash-defined
//! equivalence classes.

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::OnceLock;

use super::super::compat::{FlatDfa, TokenizerView};

const MISSING_BLOCK: u32 = u32::MAX;

fn max_length_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

struct ActiveTransitionTable {
    width: usize,
    targets_flat: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RefineMode {
    Sorted,
    Interned,
    Auto,
}

fn refine_mode() -> RefineMode {
    static MODE: OnceLock<RefineMode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("GLRMASK_MAX_LENGTH_REFINE_MODE") {
        Ok(value) if value.trim().eq_ignore_ascii_case("sorted") => RefineMode::Sorted,
        Ok(value) if value.trim().eq_ignore_ascii_case("interned") => RefineMode::Interned,
        _ => RefineMode::Auto,
    })
}

fn is_full_state_query(states: &[usize], total_states: usize) -> bool {
    if states.len() != total_states {
        return false;
    }
    states
        .iter()
        .enumerate()
        .all(|(index, &state)| state == index)
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
fn hash_signature_row(row: &[u32]) -> u64 {
    let mut hash = mix_u64((row.len() as u64) ^ 0x9E37_79B9_7F4A_7C15);
    for &cell in row {
        hash = mix_u64(hash ^ (cell as u64).wrapping_add(0xA24B_AED4_963E_E407));
    }
    hash
}

#[inline]
fn usize_to_u32(value: usize, what: &str) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{} exceeds u32::MAX", what))
}

#[inline]
fn is_active_group(group_id: usize, active_groups: Option<&[bool]>) -> bool {
    active_groups.map_or(true, |groups| {
        groups.get(group_id).copied().unwrap_or(false)
    })
}

fn filtered_group_ids(values: &[usize], active_groups: Option<&[bool]>) -> Vec<usize> {
    values
        .iter()
        .copied()
        .filter(|&group_id| is_active_group(group_id, active_groups))
        .collect()
}

fn build_filtered_finalizer_labels(
    dfa: &FlatDfa,
    active_groups: Option<&[bool]>,
) -> Vec<Vec<usize>> {
    dfa.states
        .par_iter()
        .map(|state| filtered_group_ids(&state.finalizers, active_groups))
        .collect()
}

fn build_filtered_possible_future_labels(
    dfa: &FlatDfa,
    active_groups: Option<&[bool]>,
) -> Vec<Vec<usize>> {
    dfa.states
        .par_iter()
        .map(|state| filtered_group_ids(&state.possible_future_group_ids, active_groups))
        .collect()
}

fn build_has_any_transition_labels(dfa: &FlatDfa) -> Vec<bool> {
    (0..dfa.states.len())
        .into_par_iter()
        .map(|state| {
            dfa.transitions_for(state)
                .iter()
                .any(|&target| target != u32::MAX)
        })
        .collect()
}

#[inline]
fn byte_is_relevant(byte: usize, relevant_bytes: Option<&[bool; 256]>) -> bool {
    relevant_bytes.map_or(true, |bytes| bytes[byte])
}

fn active_byte_representatives(
    relevant_bytes: Option<&[bool; 256]>,
    byte_to_class: Option<&[u8; 256]>,
) -> Vec<u8> {
    if let Some(byte_to_class) = byte_to_class {
        let num_classes = *byte_to_class.iter().max().unwrap_or(&0) as usize + 1;
        let mut class_rep: Vec<Option<u8>> = vec![None; num_classes];

        for byte in 0..256usize {
            if !byte_is_relevant(byte, relevant_bytes) {
                continue;
            }
            let class = byte_to_class[byte] as usize;
            if class_rep[class].is_none() {
                class_rep[class] = Some(byte as u8);
            }
        }

        class_rep.into_iter().flatten().collect()
    } else {
        (0..256usize)
            .filter(|&byte| byte_is_relevant(byte, relevant_bytes))
            .map(|byte| byte as u8)
            .collect()
    }
}

fn build_initial_label_partition(
    dfa: &FlatDfa,
    active_groups: Option<&[bool]>,
) -> (Vec<u32>, usize) {
    let n = dfa.states.len();
    if n == 0 {
        return (Vec::new(), 0);
    }

    let finalizer_labels = build_filtered_finalizer_labels(dfa, active_groups);
    let future_labels = build_filtered_possible_future_labels(dfa, active_groups);
    let has_any_transition = build_has_any_transition_labels(dfa);

    let mut order: Vec<usize> = (0..n).collect();
    order.par_sort_unstable_by(|&left, &right| {
        finalizer_labels[left]
            .cmp(&finalizer_labels[right])
            .then_with(|| future_labels[left].cmp(&future_labels[right]))
            .then_with(|| has_any_transition[left].cmp(&has_any_transition[right]))
            .then_with(|| left.cmp(&right))
    });

    let mut label_ids = vec![0u32; n];
    let mut label_count = 0usize;
    let mut previous_state: Option<usize> = None;

    for state in order {
        let starts_new_label = previous_state.map_or(true, |prev| {
            finalizer_labels[state] != finalizer_labels[prev]
                || future_labels[state] != future_labels[prev]
                || has_any_transition[state] != has_any_transition[prev]
        });
        if starts_new_label {
            label_count += 1;
        }
        label_ids[state] = usize_to_u32(label_count - 1, "initial label id");
        previous_state = Some(state);
    }

    (label_ids, label_count)
}

fn same_partition(left: &[u32], left_count: usize, right: &[u32], right_count: usize) -> bool {
    if left.len() != right.len() || left_count != right_count {
        return false;
    }

    let mut left_to_right = vec![u32::MAX; left_count];
    let mut right_to_left = vec![u32::MAX; right_count];

    for (&l, &r) in left.iter().zip(right.iter()) {
        let li = l as usize;
        let ri = r as usize;
        if li >= left_count || ri >= right_count {
            return false;
        }

        if left_to_right[li] == u32::MAX {
            left_to_right[li] = r;
        } else if left_to_right[li] != r {
            return false;
        }

        if right_to_left[ri] == u32::MAX {
            right_to_left[ri] = l;
        } else if right_to_left[ri] != l {
            return false;
        }
    }

    true
}

fn build_active_transition_table(dfa: &FlatDfa, active_bytes: &[u8]) -> ActiveTransitionTable {
    let width = active_bytes.len();
    let n = dfa.states.len();
    let mut targets_flat = vec![MISSING_BLOCK; n * width];

    targets_flat
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(state, row)| {
            for (slot, &byte) in active_bytes.iter().enumerate() {
                row[slot] = dfa.trans(state, byte as usize);
            }
        });

    ActiveTransitionTable {
        width,
        targets_flat,
    }
}

fn refine_once_sorted(
    active_targets: &ActiveTransitionTable,
    label_ids: &[u32],
    prev_blocks: &[u32],
    signatures: &mut [u32],
    row_hashes: &mut [u64],
    order: &mut [usize],
) -> (Vec<u32>, usize) {
    let n = prev_blocks.len();
    let width = 1 + active_targets.width;
    debug_assert_eq!(signatures.len(), n * width);
    debug_assert_eq!(row_hashes.len(), n);

    signatures
        .par_chunks_mut(width)
        .zip(row_hashes.par_iter_mut())
        .enumerate()
        .for_each(|(state, (row, row_hash))| {
            row[0] = label_ids[state];
            let target_start = state * active_targets.width;
            let targets =
                &active_targets.targets_flat[target_start..target_start + active_targets.width];
            for (i, &target) in targets.iter().enumerate() {
                row[i + 1] = if target == MISSING_BLOCK {
                    MISSING_BLOCK
                } else {
                    prev_blocks[target as usize]
                };
            }
            *row_hash = hash_signature_row(row);
        });

    order.par_sort_unstable_by(|&left, &right| {
        let hash_cmp = row_hashes[left].cmp(&row_hashes[right]);
        if hash_cmp != std::cmp::Ordering::Equal {
            return hash_cmp;
        }

        let left_start = left * width;
        let right_start = right * width;
        signatures[left_start..left_start + width]
            .cmp(&signatures[right_start..right_start + width])
            .then_with(|| left.cmp(&right))
    });

    let mut next_blocks = vec![0u32; n];
    let mut block_count = 0usize;
    let mut previous_state: Option<usize> = None;

    for &state in order.iter() {
        let starts_new_block = previous_state.map_or(true, |prev| {
            if row_hashes[state] != row_hashes[prev] {
                return true;
            }

            let state_start = state * width;
            let prev_start = prev * width;
            signatures[state_start..state_start + width]
                != signatures[prev_start..prev_start + width]
        });
        if starts_new_block {
            block_count += 1;
        }
        next_blocks[state] = usize_to_u32(block_count - 1, "partition block id");
        previous_state = Some(state);
    }

    (next_blocks, block_count)
}

#[inline(always)]
fn row_hash(
    state: usize,
    label_ids: &[u32],
    prev_blocks: &[u32],
    active_targets: &ActiveTransitionTable,
) -> u64 {
    let mut hash = mix_u64(((1 + active_targets.width) as u64) ^ 0x9E37_79B9_7F4A_7C15);
    hash = mix_u64(hash ^ (label_ids[state] as u64).wrapping_add(0xA24B_AED4_963E_E407));

    let start = state * active_targets.width;
    let end = start + active_targets.width;
    for &target in &active_targets.targets_flat[start..end] {
        let block = if target == MISSING_BLOCK {
            MISSING_BLOCK
        } else {
            prev_blocks[target as usize]
        };
        hash = mix_u64(hash ^ (block as u64).wrapping_add(0xA24B_AED4_963E_E407));
    }
    hash
}

#[inline(always)]
fn rows_equal(
    state_a: usize,
    state_b: usize,
    label_ids: &[u32],
    prev_blocks: &[u32],
    active_targets: &ActiveTransitionTable,
) -> bool {
    if label_ids[state_a] != label_ids[state_b] {
        return false;
    }

    let start_a = state_a * active_targets.width;
    let start_b = state_b * active_targets.width;
    for slot in 0..active_targets.width {
        let target_a = active_targets.targets_flat[start_a + slot];
        let target_b = active_targets.targets_flat[start_b + slot];
        let block_a = if target_a == MISSING_BLOCK {
            MISSING_BLOCK
        } else {
            prev_blocks[target_a as usize]
        };
        let block_b = if target_b == MISSING_BLOCK {
            MISSING_BLOCK
        } else {
            prev_blocks[target_b as usize]
        };
        if block_a != block_b {
            return false;
        }
    }

    true
}

fn refine_once_interned(
    label_ids: &[u32],
    prev_blocks: &[u32],
    active_targets: &ActiveTransitionTable,
    row_hashes: &mut [u64],
) -> (Vec<u32>, usize) {
    let n = prev_blocks.len();
    debug_assert_eq!(label_ids.len(), n);
    debug_assert_eq!(row_hashes.len(), n);

    row_hashes
        .par_iter_mut()
        .enumerate()
        .for_each(|(state, row_hash_out)| {
            *row_hash_out = row_hash(state, label_ids, prev_blocks, active_targets);
        });

    let mut next_blocks = vec![0u32; n];
    let mut block_count = 0usize;
    let mut buckets = HashMap::<u64, Vec<(usize, u32)>>::new();

    for state in 0..n {
        let hash = row_hashes[state];
        let reps = buckets.entry(hash).or_default();
        let mut assigned_block = None;
        for &(representative_state, block_id) in reps.iter() {
            if rows_equal(
                state,
                representative_state,
                label_ids,
                prev_blocks,
                active_targets,
            ) {
                assigned_block = Some(block_id);
                break;
            }
        }

        let block_id = if let Some(block_id) = assigned_block {
            block_id
        } else {
            let block_id = usize_to_u32(block_count, "partition block id");
            block_count += 1;
            reps.push((state, block_id));
            block_id
        };
        next_blocks[state] = block_id;
    }

    (next_blocks, block_count)
}

#[inline]
fn auto_prefers_sorted_refinement(
    is_full_state_query: bool,
    num_states: usize,
    active_byte_count: usize,
) -> bool {
    if !is_full_state_query {
        return false;
    }

    // Sorting whole signature rows is poor for very narrow alphabets: the sort
    // overhead dominates, and the interned path usually wins.  Broad rows can
    // favor sorting on small tokenizers, but on large preserved-length product
    // DFAs the row sort becomes the bottleneck (`Github_hard---o62060`).
    // Prefer the interned path once the full query is large enough.
    num_states <= 32_768 && (active_byte_count >= 58 || active_byte_count <= 16)
}

fn compute_kbounded_partition(
    dfa: &FlatDfa,
    k: usize,
    active_groups: Option<&[bool]>,
    active_bytes: &[u8],
    is_full_state_query: bool,
) -> (Vec<u32>, usize, usize) {
    let n = dfa.states.len();
    if n == 0 {
        return (Vec::new(), 0, 0);
    }

    let (label_ids, mut block_count) = build_initial_label_partition(dfa, active_groups);
    let mut blocks = label_ids.clone();

    if block_count == n || active_bytes.is_empty() {
        return (blocks, block_count, 0);
    }

    let active_targets = build_active_transition_table(dfa, active_bytes);
    let width = 1 + active_bytes.len();
    let mut signatures = vec![0u32; n * width];
    let mut row_hashes = vec![0u64; n];
    let mut order: Vec<usize> = (0..n).collect();
    let mode = refine_mode();

    for step in 0..k {
        let use_sorted = match mode {
            RefineMode::Sorted => true,
            RefineMode::Interned => false,
            RefineMode::Auto => {
                auto_prefers_sorted_refinement(is_full_state_query, n, active_bytes.len())
            }
        };

        let (next_blocks, next_count) = if use_sorted {
            refine_once_sorted(
                &active_targets,
                &label_ids,
                &blocks,
                &mut signatures,
                &mut row_hashes,
                &mut order,
            )
        } else {
            refine_once_interned(&label_ids, &blocks, &active_targets, &mut row_hashes)
        };

        let iteration = step + 1;
        let stable = same_partition(&blocks, block_count, &next_blocks, next_count);
        blocks = next_blocks;
        block_count = next_count;

        if stable || block_count == n {
            return (blocks, block_count, iteration);
        }
    }

    (blocks, block_count, k)
}

fn build_subset_mapping(states: &[usize], blocks: &[u32]) -> Vec<usize> {
    let mut indexed_blocks: Vec<(u32, usize, usize)> = states
        .par_iter()
        .enumerate()
        .map(|(position, &state_id)| (blocks[state_id], state_id, position))
        .collect();

    indexed_blocks.par_sort_unstable_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });

    let mut mapping = vec![0usize; states.len()];
    let mut current_block: Option<u32> = None;
    let mut current_rep = 0usize;

    for (block, state_id, position) in indexed_blocks {
        if current_block != Some(block) {
            current_block = Some(block);
            current_rep = state_id;
        }
        mapping[position] = current_rep;
    }

    mapping
}

pub(crate) fn find_state_equivalence_classes_kbounded(
    tokenizer: &TokenizerView,
    states: &[usize],
    k: usize,
    active_groups: Option<&[bool]>,
    relevant_bytes: Option<&[bool; 256]>,
    byte_to_class: Option<&[u8; 256]>,
    mode: &str,
) -> Vec<usize> {
    if states.is_empty() {
        return Vec::new();
    }

    let profile_enabled = max_length_profile_enabled();
    let total_started_at = profile_enabled.then(std::time::Instant::now);
    let dfa = tokenizer.dfa();
    let active_bytes_started_at = profile_enabled.then(std::time::Instant::now);
    let active_bytes = active_byte_representatives(relevant_bytes, byte_to_class);
    let active_bytes_ms = active_bytes_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let full_state_query = is_full_state_query(states, dfa.states.len());
    let compute_started_at = profile_enabled.then(std::time::Instant::now);
    let (blocks, block_count, iterations_run) =
        compute_kbounded_partition(dfa, k, active_groups, &active_bytes, full_state_query);
    let compute_ms = compute_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let mapping_started_at = profile_enabled.then(std::time::Instant::now);
    let mapping = build_subset_mapping(states, &blocks);
    let mapping_ms = mapping_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][max_length_equiv] mode={} query_states={} dfa_states={} k={} active_bytes={} full_state_query={} iterations={} reps={} active_bytes_ms={:.3} compute_ms={:.3} mapping_ms={:.3} total_ms={:.3}",
            mode,
            states.len(),
            dfa.states.len(),
            k,
            active_bytes.len(),
            full_state_query,
            iterations_run,
            block_count,
            active_bytes_ms,
            compute_ms,
            mapping_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    mapping
}

pub fn find_state_equivalence_classes_byte_restricted<S: AsRef<[u8]>>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    byte_to_class: Option<&[u8; 256]>,
    active_groups: Option<&[bool]>,
    relevant_bytes: Option<&[bool; 256]>,
) -> Vec<usize> {
    let max_len = tokens
        .iter()
        .map(|token| token.as_ref().len())
        .max()
        .unwrap_or(0);

    let derived_relevant_bytes;
    let relevant_bytes = match relevant_bytes {
        Some(bytes) => bytes,
        None => {
            let mut bytes = [false; 256];
            for token in tokens {
                for &byte in token.as_ref() {
                    bytes[byte as usize] = true;
                }
            }
            derived_relevant_bytes = bytes;
            &derived_relevant_bytes
        }
    };

    let mapping = find_state_equivalence_classes_kbounded(
        tokenizer,
        states,
        max_len,
        active_groups,
        Some(relevant_bytes),
        byte_to_class,
        "byte_restricted",
    );

    mapping
}
