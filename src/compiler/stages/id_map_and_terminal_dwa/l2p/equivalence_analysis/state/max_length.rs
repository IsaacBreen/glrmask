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
    dfa: &FlatDfa,
    active_bytes: &[u8],
    label_ids: &[u32],
    prev_blocks: &[u32],
    signatures: &mut [u32],
    row_hashes: &mut [u64],
    order: &mut [usize],
) -> (Vec<u32>, usize) {
    let n = prev_blocks.len();
    let width = 1 + active_bytes.len();
    debug_assert_eq!(signatures.len(), n * width);
    debug_assert_eq!(row_hashes.len(), n);

    signatures
        .par_chunks_mut(width)
        .zip(row_hashes.par_iter_mut())
        .enumerate()
        .for_each(|(state, (row, row_hash))| {
            row[0] = label_ids[state];
            for (i, &byte) in active_bytes.iter().enumerate() {
                let target = dfa.trans(state, byte as usize);
                row[i + 1] = if target == u32::MAX {
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
    is_full_state_query && num_states <= 16_384 && active_byte_count <= 16
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
                dfa,
                active_bytes,
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

struct ReverseSymbolTransitions {
    offsets: Vec<u32>,
    predecessors: Vec<u32>,
}

impl ReverseSymbolTransitions {
    #[inline]
    fn predecessors_for_target(&self, target: usize) -> &[u32] {
        let start = self.offsets[target] as usize;
        let end = self.offsets[target + 1] as usize;
        &self.predecessors[start..end]
    }
}

struct ReverseTransitionTable {
    by_symbol: Vec<ReverseSymbolTransitions>,
}

fn build_reverse_symbol_transitions(dfa: &FlatDfa, byte: u8) -> ReverseSymbolTransitions {
    let n = dfa.states.len();
    let mut counts = vec![0u32; n];

    for state in 0..n {
        let target = dfa.trans(state, byte as usize);
        if target != MISSING_BLOCK {
            let target = target as usize;
            if target < n {
                counts[target] += 1;
            }
        }
    }

    let mut offsets = vec![0u32; n + 1];
    for state in 0..n {
        offsets[state + 1] = offsets[state] + counts[state];
    }

    let mut next_offsets = offsets[..n].to_vec();
    let mut predecessors = vec![0u32; offsets[n] as usize];
    for state in 0..n {
        let target = dfa.trans(state, byte as usize);
        if target == MISSING_BLOCK {
            continue;
        }
        let target = target as usize;
        if target >= n {
            continue;
        }
        let slot = next_offsets[target] as usize;
        predecessors[slot] = usize_to_u32(state, "reverse transition predecessor state");
        next_offsets[target] += 1;
    }

    ReverseSymbolTransitions {
        offsets,
        predecessors,
    }
}

fn build_reverse_transition_table(dfa: &FlatDfa, active_bytes: &[u8]) -> ReverseTransitionTable {
    let by_symbol = active_bytes
        .par_iter()
        .map(|&byte| build_reverse_symbol_transitions(dfa, byte))
        .collect();
    ReverseTransitionTable { by_symbol }
}

fn build_blocks_from_partition(block_ids: &[u32], block_count: usize) -> Vec<Vec<u32>> {
    let mut sizes = vec![0usize; block_count];
    for &block_id in block_ids {
        sizes[block_id as usize] += 1;
    }

    let mut blocks: Vec<Vec<u32>> = sizes.into_iter().map(Vec::with_capacity).collect();
    for (state, &block_id) in block_ids.iter().enumerate() {
        blocks[block_id as usize].push(usize_to_u32(state, "partition state id"));
    }

    blocks
}

#[inline]
fn pack_work_item(block_id: u32, symbol: usize) -> u64 {
    debug_assert!(symbol < 256);
    ((block_id as u64) << 8) | symbol as u64
}

#[inline]
fn unpack_work_item(item: u64) -> (usize, usize) {
    ((item >> 8) as usize, (item & 0xff) as usize)
}

fn stable_refinement_blocks(
    dfa: &FlatDfa,
    active_groups: Option<&[bool]>,
    active_bytes: &[u8],
) -> Vec<Vec<u32>> {
    let n = dfa.states.len();
    let (initial_blocks, initial_block_count) = build_initial_label_partition(dfa, active_groups);
    let mut blocks = build_blocks_from_partition(&initial_blocks, initial_block_count);

    if initial_block_count == n || active_bytes.is_empty() {
        return blocks;
    }

    let reverse = build_reverse_transition_table(dfa, active_bytes);
    let width = active_bytes.len();
    debug_assert!(width <= 256);

    let mut state_to_block = initial_blocks;
    let mut worklist = Vec::with_capacity(blocks.len().saturating_mul(width));
    for block_id in 0..blocks.len() {
        let block_id = usize_to_u32(block_id, "partition block id");
        for symbol in 0..width {
            worklist.push(pack_work_item(block_id, symbol));
        }
    }

    let mut mark_epoch = vec![0u32; n];
    let mut epoch = 1u32;
    let mut marked_count_by_block = vec![0u32; blocks.len()];
    let mut touched_blocks = Vec::<u32>::new();

    while let Some(item) = worklist.pop() {
        let (splitter_block, symbol) = unpack_work_item(item);
        if splitter_block >= blocks.len() || blocks[splitter_block].is_empty() {
            continue;
        }

        for &target in &blocks[splitter_block] {
            for &predecessor in reverse.by_symbol[symbol].predecessors_for_target(target as usize) {
                let predecessor = predecessor as usize;
                if mark_epoch[predecessor] == epoch {
                    continue;
                }
                mark_epoch[predecessor] = epoch;
                let block_id = state_to_block[predecessor] as usize;
                if marked_count_by_block[block_id] == 0 {
                    touched_blocks.push(block_id as u32);
                }
                marked_count_by_block[block_id] += 1;
            }
        }

        for &block_id_u32 in &touched_blocks {
            let block_id = block_id_u32 as usize;
            let marked_count = marked_count_by_block[block_id] as usize;
            marked_count_by_block[block_id] = 0;

            let block_len = blocks[block_id].len();
            if marked_count == 0 || marked_count == block_len {
                continue;
            }

            // Keep the larger side in the existing block id and create a new
            // block for the smaller side.  Since every old block starts with
            // all symbols in the worklist and old ids are never re-added, this
            // is the usual Hopcroft smaller-splitter rule without needing a
            // separate pending-work membership table.
            let unmarked_count = block_len - marked_count;
            let move_marked_to_new = marked_count <= unmarked_count;
            let new_capacity = if move_marked_to_new {
                marked_count
            } else {
                unmarked_count
            };
            let kept_capacity = block_len - new_capacity;
            let old_states = std::mem::take(&mut blocks[block_id]);
            let mut kept_states = Vec::with_capacity(kept_capacity);
            let mut new_states = Vec::with_capacity(new_capacity);

            for state in old_states {
                let is_marked = mark_epoch[state as usize] == epoch;
                if is_marked == move_marked_to_new {
                    new_states.push(state);
                } else {
                    kept_states.push(state);
                }
            }

            debug_assert_eq!(new_states.len(), new_capacity);
            debug_assert_eq!(kept_states.len(), kept_capacity);
            debug_assert!(!new_states.is_empty());
            debug_assert!(!kept_states.is_empty());

            let new_block_id = usize_to_u32(blocks.len(), "partition block id");
            for &state in &new_states {
                state_to_block[state as usize] = new_block_id;
            }

            blocks[block_id] = kept_states;
            blocks.push(new_states);
            marked_count_by_block.push(0);

            for symbol in 0..width {
                worklist.push(pack_work_item(new_block_id, symbol));
            }

            if blocks.len() == n {
                break;
            }
        }
        touched_blocks.clear();

        if blocks.len() == n {
            break;
        }

        epoch = epoch.wrapping_add(1);
        if epoch == 0 {
            mark_epoch.fill(0);
            epoch = 1;
        }
    }

    blocks
}

fn build_full_mapping_from_blocks(blocks: &[Vec<u32>], num_states: usize) -> Vec<usize> {
    let mut mapping = vec![0usize; num_states];
    for block in blocks {
        if block.is_empty() {
            continue;
        }
        let representative = block.iter().copied().min().unwrap() as usize;
        for &state in block {
            mapping[state as usize] = representative;
        }
    }
    mapping
}

/// Compute a stable Moore partition over the active byte classes.
///
/// This is intentionally stronger (finer) than the finite `k = max_token_len`
/// prepass used by [`find_state_equivalence_classes_byte_restricted`]: states
/// in the same returned class have identical labels and transition to the same
/// returned class for every active byte class, so they are equivalent for all
/// byte strings over that alphabet and therefore for every vocabulary token.
/// It is used for the global pre-map where a conservative refinement is safe
/// and avoids the `max_token_len` factor that dominates large-token schemas.
pub fn find_state_equivalence_classes_stable_byte_restricted(
    tokenizer: &TokenizerView,
    byte_to_class: Option<&[u8; 256]>,
    active_groups: Option<&[bool]>,
    relevant_bytes: Option<&[bool; 256]>,
) -> Vec<usize> {
    let dfa = tokenizer.dfa();
    if dfa.states.is_empty() {
        return Vec::new();
    }

    let active_bytes = active_byte_representatives(relevant_bytes, byte_to_class);
    let blocks = stable_refinement_blocks(dfa, active_groups, &active_bytes);
    build_full_mapping_from_blocks(&blocks, dfa.states.len())
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

fn find_state_equivalence_classes_kbounded(
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

    let dfa = tokenizer.dfa();
    let active_bytes = active_byte_representatives(relevant_bytes, byte_to_class);
    let full_state_query = is_full_state_query(states, dfa.states.len());
    let (blocks, block_count, iterations_run) =
        compute_kbounded_partition(dfa, k, active_groups, &active_bytes, full_state_query);
    let _ = (block_count, iterations_run, mode);

    build_subset_mapping(states, &blocks)
}

pub fn find_state_equivalence_classes<S: AsRef<[u8]>>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    active_groups: Option<&[bool]>,
    relevant_bytes: Option<&[bool; 256]>,
) -> Vec<usize> {
    let max_len = tokens
        .iter()
        .map(|token| token.as_ref().len())
        .max()
        .unwrap_or(0);
    let mapping = find_state_equivalence_classes_kbounded(
        tokenizer,
        states,
        max_len,
        active_groups,
        relevant_bytes,
        None,
        "default",
    );

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
