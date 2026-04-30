//! Max-length bounded state equivalence prepass.
//!
//! This implementation computes an exact Moore-style finite-depth partition
//! refinement over a filtered DFA view instead of using hash-defined
//! equivalence classes.

use rayon::prelude::*;
use std::collections::VecDeque;

use super::super::compat::{FlatDfa, TokenizerView};

const MISSING_BLOCK: u32 = u32::MAX;

fn debug_max_length_enabled() -> bool {
    std::env::var("GLRMASK_DEBUG_MAX_LENGTH")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

fn profile_compile_enabled() -> bool {
    std::env::var("GLRMASK_PROFILE_COMPILE")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

#[inline]
fn usize_to_u32(value: usize, what: &str) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{} exceeds u32::MAX", what))
}

#[inline]
fn is_active_group(group_id: usize, active_groups: Option<&[bool]>) -> bool {
    active_groups.map_or(true, |groups| groups.get(group_id).copied().unwrap_or(false))
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

#[inline]
fn byte_is_relevant(byte: usize, relevant_bytes: Option<&[bool; 256]>) -> bool {
    relevant_bytes.map_or(true, |bytes| bytes[byte])
}

fn count_relevant_bytes(relevant_bytes: Option<&[bool; 256]>) -> usize {
    relevant_bytes.map_or(256, |bytes| bytes.iter().filter(|&&b| b).count())
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

fn build_dedup_adjacency(dfa: &FlatDfa, active_bytes: &[u8]) -> Vec<Vec<usize>> {
    let n = dfa.states.len();

    (0..n)
        .into_par_iter()
        .map(|state| {
            let mut targets = Vec::new();
            for &byte in active_bytes {
                let target = dfa.trans(state, byte as usize);
                if target != u32::MAX {
                    targets.push(target as usize);
                }
            }
            targets.sort_unstable();
            targets.dedup();
            targets
        })
        .collect()
}

fn compute_scc_ids(adj: &[Vec<usize>]) -> (Vec<u32>, usize) {
    let n = adj.len();
    let mut reverse_adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (source, targets) in adj.iter().enumerate() {
        for &target in targets {
            reverse_adj[target].push(source);
        }
    }

    let mut visited = vec![false; n];
    let mut finish_order = Vec::with_capacity(n);

    for start in 0..n {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];

        while let Some((state, next_child)) = stack.pop() {
            if next_child < adj[state].len() {
                stack.push((state, next_child + 1));
                let target = adj[state][next_child];
                if !visited[target] {
                    visited[target] = true;
                    stack.push((target, 0));
                }
            } else {
                finish_order.push(state);
            }
        }
    }

    let mut scc_id = vec![u32::MAX; n];
    let mut scc_count = 0usize;

    for &start in finish_order.iter().rev() {
        if scc_id[start] != u32::MAX {
            continue;
        }

        let current_scc = usize_to_u32(scc_count, "SCC id");
        scc_count += 1;
        scc_id[start] = current_scc;
        let mut stack = vec![start];

        while let Some(state) = stack.pop() {
            for &pred in &reverse_adj[state] {
                if scc_id[pred] == u32::MAX {
                    scc_id[pred] = current_scc;
                    stack.push(pred);
                }
            }
        }
    }

    (scc_id, scc_count)
}

fn union_sorted_in_place(dst: &mut Vec<usize>, src: &[usize]) {
    if src.is_empty() {
        return;
    }
    if dst.is_empty() {
        dst.extend_from_slice(src);
        return;
    }

    let mut merged = Vec::with_capacity(dst.len() + src.len());
    let mut i = 0usize;
    let mut j = 0usize;

    while i < dst.len() && j < src.len() {
        match dst[i].cmp(&src[j]) {
            std::cmp::Ordering::Less => {
                merged.push(dst[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                merged.push(src[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                merged.push(dst[i]);
                i += 1;
                j += 1;
            }
        }
    }

    merged.extend_from_slice(&dst[i..]);
    merged.extend_from_slice(&src[j..]);
    *dst = merged;
}

fn compute_strict_future_label_ids(
    dfa: &FlatDfa,
    active_bytes: &[u8],
    finalizer_labels: &[Vec<usize>],
) -> Vec<u32> {
    let n = dfa.states.len();
    if n == 0 {
        return Vec::new();
    }
    if active_bytes.is_empty() || finalizer_labels.iter().all(|label| label.is_empty()) {
        return vec![0u32; n];
    }

    let adj = build_dedup_adjacency(dfa, active_bytes);
    let (scc_id, scc_count) = compute_scc_ids(&adj);

    let mut scc_finalizers: Vec<Vec<usize>> = vec![Vec::new(); scc_count];
    let mut scc_sizes = vec![0usize; scc_count];
    let mut scc_has_self_loop = vec![false; scc_count];

    for state in 0..n {
        let sid = scc_id[state] as usize;
        scc_sizes[sid] += 1;
        scc_finalizers[sid].extend_from_slice(&finalizer_labels[state]);
        if adj[state].iter().any(|&target| target == state) {
            scc_has_self_loop[sid] = true;
        }
    }

    for finalizers in &mut scc_finalizers {
        finalizers.sort_unstable();
        finalizers.dedup();
    }

    let mut scc_successors: Vec<Vec<u32>> = vec![Vec::new(); scc_count];
    for (source, targets) in adj.iter().enumerate() {
        let source_scc = scc_id[source];
        for &target in targets {
            let target_scc = scc_id[target];
            if source_scc != target_scc {
                scc_successors[source_scc as usize].push(target_scc);
            }
        }
    }
    for successors in &mut scc_successors {
        successors.sort_unstable();
        successors.dedup();
    }

    let mut scc_predecessors: Vec<Vec<u32>> = vec![Vec::new(); scc_count];
    let mut remaining_successors = vec![0usize; scc_count];
    for (sid, successors) in scc_successors.iter().enumerate() {
        remaining_successors[sid] = successors.len();
        let sid_u32 = usize_to_u32(sid, "SCC id");
        for &succ in successors {
            scc_predecessors[succ as usize].push(sid_u32);
        }
    }

    let mut scc_futures: Vec<Vec<usize>> = vec![Vec::new(); scc_count];
    for sid in 0..scc_count {
        let cyclic = scc_sizes[sid] > 1 || scc_has_self_loop[sid];
        if cyclic {
            scc_futures[sid] = scc_finalizers[sid].clone();
        }
    }

    let mut queue: VecDeque<u32> = remaining_successors
        .iter()
        .enumerate()
        .filter_map(|(sid, &count)| (count == 0).then(|| usize_to_u32(sid, "SCC id")))
        .collect();

    while let Some(sid_u32) = queue.pop_front() {
        let sid = sid_u32 as usize;
        let sid_finalizers = scc_finalizers[sid].clone();
        let sid_futures = scc_futures[sid].clone();

        for &pred_u32 in &scc_predecessors[sid] {
            let pred = pred_u32 as usize;
            union_sorted_in_place(&mut scc_futures[pred], &sid_finalizers);
            union_sorted_in_place(&mut scc_futures[pred], &sid_futures);

            remaining_successors[pred] -= 1;
            if remaining_successors[pred] == 0 {
                queue.push_back(pred_u32);
            }
        }
    }

    let mut scc_order: Vec<usize> = (0..scc_count).collect();
    scc_order.sort_unstable_by(|&left, &right| {
        scc_futures[left]
            .cmp(&scc_futures[right])
            .then_with(|| left.cmp(&right))
    });

    let mut scc_future_ids = vec![0u32; scc_count];
    let mut future_label_count = 0usize;
    let mut previous_scc: Option<usize> = None;

    for sid in scc_order {
        let starts_new_label = previous_scc.map_or(true, |prev| scc_futures[sid] != scc_futures[prev]);
        if starts_new_label {
            future_label_count += 1;
        }
        scc_future_ids[sid] = usize_to_u32(future_label_count - 1, "future-label id");
        previous_scc = Some(sid);
    }

    let mut state_future_ids = vec![0u32; n];
    for state in 0..n {
        state_future_ids[state] = scc_future_ids[scc_id[state] as usize];
    }

    state_future_ids
}

fn build_initial_label_partition(
    dfa: &FlatDfa,
    active_groups: Option<&[bool]>,
    active_bytes: &[u8],
) -> (Vec<u32>, usize) {
    let n = dfa.states.len();
    if n == 0 {
        return (Vec::new(), 0);
    }

    let finalizer_labels = build_filtered_finalizer_labels(dfa, active_groups);
    let future_label_ids = compute_strict_future_label_ids(dfa, active_bytes, &finalizer_labels);

    let mut order: Vec<usize> = (0..n).collect();
    order.par_sort_unstable_by(|&left, &right| {
        finalizer_labels[left]
            .cmp(&finalizer_labels[right])
            .then_with(|| future_label_ids[left].cmp(&future_label_ids[right]))
            .then_with(|| left.cmp(&right))
    });

    let mut label_ids = vec![0u32; n];
    let mut label_count = 0usize;
    let mut previous_state: Option<usize> = None;

    for state in order {
        let starts_new_label = previous_state.map_or(true, |prev| {
            finalizer_labels[state] != finalizer_labels[prev]
                || future_label_ids[state] != future_label_ids[prev]
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

fn refine_once(
    dfa: &FlatDfa,
    active_bytes: &[u8],
    label_ids: &[u32],
    prev_blocks: &[u32],
    signatures: &mut [u32],
    order: &mut [usize],
) -> (Vec<u32>, usize) {
    let n = prev_blocks.len();
    let width = 1 + active_bytes.len();
    debug_assert_eq!(signatures.len(), n * width);

    signatures
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(state, row)| {
            row[0] = label_ids[state];
            for (i, &byte) in active_bytes.iter().enumerate() {
                let target = dfa.trans(state, byte as usize);
                row[i + 1] = if target == u32::MAX {
                    MISSING_BLOCK
                } else {
                    prev_blocks[target as usize]
                };
            }
        });

    order.par_sort_unstable_by(|&left, &right| {
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
            let state_start = state * width;
            let prev_start = prev * width;
            signatures[state_start..state_start + width] != signatures[prev_start..prev_start + width]
        });
        if starts_new_block {
            block_count += 1;
        }
        next_blocks[state] = usize_to_u32(block_count - 1, "partition block id");
        previous_state = Some(state);
    }

    (next_blocks, block_count)
}

fn compute_kbounded_partition(
    dfa: &FlatDfa,
    k: usize,
    active_groups: Option<&[bool]>,
    active_bytes: &[u8],
) -> (Vec<u32>, usize, usize) {
    let n = dfa.states.len();
    if n == 0 {
        return (Vec::new(), 0, 0);
    }

    let debug = debug_max_length_enabled();
    let (label_ids, mut block_count) = build_initial_label_partition(dfa, active_groups, active_bytes);
    let mut blocks = label_ids.clone();

    if debug {
        eprintln!("[glrmask/debug][max_length_partition] depth=0 blocks={}", block_count);
    }

    if k == 0 || block_count == n {
        return (blocks, block_count, 0);
    }

    let width = 1 + active_bytes.len();
    let mut signatures = vec![0u32; n * width];
    let mut order: Vec<usize> = (0..n).collect();

    for step in 0..k {
        let (next_blocks, next_count) = refine_once(
            dfa,
            active_bytes,
            &label_ids,
            &blocks,
            &mut signatures,
            &mut order,
        );

        let iteration = step + 1;
        let stable = same_partition(&blocks, block_count, &next_blocks, next_count);
        blocks = next_blocks;
        block_count = next_count;

        if debug {
            eprintln!(
                "[glrmask/debug][max_length_partition] depth={} blocks={}",
                iteration,
                block_count,
            );
        }

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

fn count_mapping_representatives(mapping: &[usize]) -> usize {
    let mut representatives = mapping.to_vec();
    representatives.sort_unstable();
    representatives.dedup();
    representatives.len()
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

    let profile = profile_compile_enabled();
    let start = std::time::Instant::now();
    let (blocks, block_count, iterations_run) =
        compute_kbounded_partition(dfa, k, active_groups, &active_bytes);

    if profile {
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[glrmask/profile][max_length_partition] mode={} dfa_states={} input_states={} k={} iterations_run={} relevant_bytes={} byte_representatives={} blocks={} analysis_ms={:.3}",
            mode,
            dfa.states.len(),
            states.len(),
            k,
            iterations_run,
            count_relevant_bytes(relevant_bytes),
            active_bytes.len(),
            block_count,
            elapsed_ms,
        );
    }

    build_subset_mapping(states, &blocks)
}

pub fn find_state_equivalence_classes<S: AsRef<[u8]>>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    active_groups: Option<&[bool]>,
    relevant_bytes: Option<&[bool; 256]>,
) -> Vec<usize> {
    let max_len = tokens.iter().map(|token| token.as_ref().len()).max().unwrap_or(0);
    let mapping = find_state_equivalence_classes_kbounded(
        tokenizer,
        states,
        max_len,
        active_groups,
        relevant_bytes,
        None,
        "default",
    );

    if debug_max_length_enabled() {
        eprintln!(
            "[glrmask/debug][max_length] max_token_len={} input_states={} tokenizer_dfa_states={} representative_states={}",
            max_len,
            states.len(),
            tokenizer.dfa().states.len(),
            count_mapping_representatives(&mapping),
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
    let max_len = tokens.iter().map(|token| token.as_ref().len()).max().unwrap_or(0);

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

    if debug_max_length_enabled() {
        eprintln!(
            "[glrmask/debug][max_length_byte_restricted] max_token_len={} input_states={} tokenizer_dfa_states={} relevant_bytes={} representative_states={}",
            max_len,
            states.len(),
            tokenizer.dfa().states.len(),
            relevant_bytes.iter().filter(|&&b| b).count(),
            count_mapping_representatives(&mapping),
        );
    }

    mapping
}
