//! Hopcroft-based DFA minimization for the lexer DFA.
//!
//! DFA minimization for byte-oriented lexer automata.
//! Uses topology-aware pre-refinement before the final Hopcroft pass.

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ds::bitset::BitSet;
use crate::ds::char_transitions::CharTransitions;

use super::dfa::DFA;

enum TopologyPrerefine {
    AlreadyMinimal(Vec<Vec<u32>>),
    Refined {
        partition: Vec<u32>,
        blocks: Vec<Vec<u32>>,
    },
    Skip,
}

fn partition_by_finalizers(dfa: &DFA) -> (Vec<u32>, Vec<Vec<u32>>) {
    let num_states = dfa.states().len();
    let mut partition = vec![0u32; num_states];
    let mut blocks: Vec<Vec<u32>> = Vec::new();
    let mut finalizer_to_block: FxHashMap<BitSet, u32> = FxHashMap::default();

    for (state_idx, state) in dfa.states().iter().enumerate() {
        let key = state.finalizers.clone();
        let block_idx = *finalizer_to_block.entry(key).or_insert_with(|| {
            let idx = blocks.len() as u32;
            blocks.push(Vec::new());
            idx
        });
        partition[state_idx] = block_idx;
        blocks[block_idx as usize].push(state_idx as u32);
    }

    (partition, blocks)
}

/// Refine finalizer classes by frozen future-terminal observations.
fn refine_partition_by_possible_futures(
    dfa: &DFA,
    blocks: Vec<Vec<u32>>,
) -> (Vec<u32>, Vec<Vec<u32>>) {
    let mut refined = Vec::with_capacity(blocks.len());
    for block in blocks {
        if block.len() <= 1 {
            refined.push(block);
            continue;
        }
        let mut by_future = rustc_hash::FxHashMap::default();
        for state in block {
            by_future
                .entry(dfa.possible_future_group_ids(state).clone())
                .or_insert_with(Vec::new)
                .push(state);
        }
        let mut groups = by_future.into_values().collect::<Vec<Vec<u32>>>();
        groups.sort_unstable_by_key(|group| group[0]);
        refined.extend(groups);
    }
    let mut partition = vec![0u32; dfa.num_states() as usize];
    for (class, block) in refined.iter().enumerate() {
        for &state in block {
            partition[state as usize] = class as u32;
        }
    }
    (partition, refined)
}

fn clear_possible_futures_for_minimization(dfa: &mut DFA) {
    let empty = BitSet::new(dfa.num_groups());
    dfa.mask_possible_futures(&empty);
}

fn dedup_adjacency(dfa: &DFA) -> Vec<Vec<usize>> {
    dfa.states()
        .iter()
        .map(|state| {
            let mut targets: Vec<usize> =
                state.transitions.iter().map(|(_, &target)| target as usize).collect();
            targets.sort_unstable();
            targets.dedup();
            targets
        })
        .collect()
}

fn compute_post_order(adj: &[Vec<usize>]) -> Vec<usize> {
    let num_states = adj.len();
    let mut post_order = Vec::with_capacity(num_states);
    let mut visited = vec![0u8; num_states];
    let mut dfs_stack: Vec<(usize, usize)> = Vec::new();

    for root in 0..num_states {
        if visited[root] != 0 {
            continue;
        }
        dfs_stack.push((root, 0));
        visited[root] = 1;

        while let Some((state, edge_index)) = dfs_stack.last_mut() {
            let state = *state;
            if *edge_index < adj[state].len() {
                let target = adj[state][*edge_index];
                *edge_index += 1;
                if visited[target] == 0 {
                    visited[target] = 1;
                    dfs_stack.push((target, 0));
                }
            } else {
                visited[state] = 2;
                post_order.push(state);
                dfs_stack.pop();
            }
        }
    }

    post_order
}

fn reverse_adjacency(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut reverse_adj = vec![Vec::new(); adj.len()];
    for (state, targets) in adj.iter().enumerate() {
        for &target in targets {
            reverse_adj[target].push(state);
        }
    }
    reverse_adj
}

fn compute_kosaraju_scc_ids(adj: &[Vec<usize>], post_order: &[usize]) -> Vec<u32> {
    let reverse_adj = reverse_adjacency(adj);
    let mut scc_id = vec![u32::MAX; adj.len()];
    let mut current_scc = 0u32;

    for &state in post_order.iter().rev() {
        if scc_id[state] != u32::MAX {
            continue;
        }

        let mut scc_stack = vec![state];
        scc_id[state] = current_scc;
        while let Some(node) = scc_stack.pop() {
            for &pred in &reverse_adj[node] {
                if scc_id[pred] == u32::MAX {
                    scc_id[pred] = current_scc;
                    scc_stack.push(pred);
                }
            }
        }

        current_scc += 1;
    }

    scc_id
}

fn build_blocks_from_labels(labels: &[u32], num_labels: u32) -> (Vec<u32>, Vec<Vec<u32>>) {
    let mut partition = vec![0u32; labels.len()];
    let mut blocks = vec![Vec::new(); num_labels as usize];

    for (state, &label) in labels.iter().enumerate() {
        partition[state] = label;
        blocks[label as usize].push(state as u32);
    }

    (partition, blocks)
}

fn has_self_loops(dfa: &DFA) -> bool {
    dfa.states().iter().enumerate().any(|(state_idx, state)| {
        state
            .transitions
            .iter()
            .any(|(_, &target)| target as usize == state_idx)
    })
}

/// Fast iterative partition refinement.
///
/// Starting from an initial partition (e.g. by finalizers), iteratively refine
/// by computing a signature for each state = (block_id, [(byte, target_block_id)])
/// and re-partitioning by signature. Converges in O(depth) iterations where
/// depth is the DFA's longest shortest path. Each iteration is O(n * avg_degree).
///
/// This is semantically equivalent to Hopcroft but much faster when the DFA
/// is large and the result has few equivalence classes (common after clearing
/// many terminal groups).
fn iterative_signature_refine(dfa: &DFA, initial_blocks: Vec<Vec<u32>>, max_iterations: u32) -> Option<Vec<Vec<u32>>> {
    let n = dfa.states().len();
    let mut partition = vec![0u32; n];
    for (block_id, block) in initial_blocks.iter().enumerate() {
        for &state in block {
            partition[state as usize] = block_id as u32;
        }
    }
    let mut num_blocks = initial_blocks.len() as u32;

    let mut label_map: FxHashMap<u64, u32> = FxHashMap::default();
    let mut iterations = 0u32;

    loop {
        if iterations >= max_iterations {
            return None;
        }

        label_map.clear();
        let mut next_label = 0u32;
        let mut new_partition = vec![0u32; n];

        for (state_idx, state) in dfa.states().iter().enumerate() {
            // Build signature: (current_block, [(byte, target_block)])
            // Use FxHash for speed instead of full signature comparison.
            let mut hasher = rustc_hash::FxHasher::default();
            partition[state_idx].hash(&mut hasher);
            state.transitions.len().hash(&mut hasher);
            for (byte, &target) in state.transitions.iter() {
                byte.hash(&mut hasher);
                partition[target as usize].hash(&mut hasher);
            }
            let sig_hash = hasher.finish();

            let label = *label_map.entry(sig_hash).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            });
            new_partition[state_idx] = label;
        }

        if next_label == num_blocks {
            break;
        }
        num_blocks = next_label;
        partition = new_partition;
        iterations += 1;
    }

    // Build final blocks
    let mut blocks = vec![Vec::new(); num_blocks as usize];
    for (state_idx, &block_id) in partition.iter().enumerate() {
        blocks[block_id as usize].push(state_idx as u32);
    }
    Some(blocks)
}

fn topology_prerefine_partition(dfa: &DFA, partition: &[u32]) -> TopologyPrerefine {
    let adj = dedup_adjacency(dfa);
    let post_order = compute_post_order(&adj);
    let scc_id = compute_kosaraju_scc_ids(&adj, &post_order);

    let mut labels = vec![u32::MAX; dfa.states().len()];
    let mut label_map: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
    label_map.reserve(dfa.states().len().min(200_000));
    let mut next_label = 0u32;
    let mut signature = Vec::with_capacity(32);

    for &state in &post_order {
        signature.clear();
        signature.push(partition[state]);

        for (byte, &target) in dfa.states()[state].transitions.iter() {
            let target = target as usize;
            signature.push(byte as u32);
            if scc_id[state] == scc_id[target] {
                signature.push(partition[target]);
            } else if labels[target] != u32::MAX {
                signature.push(labels[target] | 0x8000_0000);
            } else {
                signature.push(partition[target]);
            }
        }

        let label = *label_map.entry(signature.clone()).or_insert_with(|| {
            let label = next_label;
            next_label += 1;
            label
        });
        labels[state] = label;
    }

    let (partition, blocks) = build_blocks_from_labels(&labels, next_label);
    if next_label as usize == dfa.states().len() {
        if has_self_loops(dfa) {
            TopologyPrerefine::Skip
        } else {
            TopologyPrerefine::AlreadyMinimal(blocks)
        }
    } else {
        TopologyPrerefine::Refined { partition, blocks }
    }
}

fn build_inverse_transitions(dfa: &DFA) -> Vec<Vec<(u8, u32)>> {
    let mut inverse = vec![Vec::new(); dfa.states().len()];
    for (src, state) in dfa.states().iter().enumerate() {
        for (input, &target) in state.transitions.iter() {
            inverse[target as usize].push((input, src as u32));
        }
    }
    inverse
}

fn hopcroft_refine_partition(
    dfa: &DFA,
    mut partition: Vec<u32>,
    mut blocks: Vec<Vec<u32>>,
) -> Vec<Vec<u32>> {
    let num_states = dfa.states().len();
    let inverse = build_inverse_transitions(dfa);

    let mut worklist: VecDeque<u32> = (0..blocks.len() as u32).collect();
    let mut in_worklist = vec![true; blocks.len()];

    let mut source_set = vec![false; num_states];
    let mut sources_to_clear: Vec<u32> = Vec::with_capacity(num_states.min(10_000));
    let mut touched_blocks: Vec<u32> = Vec::with_capacity(1024);
    let mut block_touched = vec![false; blocks.len()];
    let mut block_sources: Vec<Vec<u32>> = vec![Vec::new(); blocks.len()];
    let mut input_sources: Vec<Vec<u32>> = vec![Vec::new(); 256];
    let mut touched_inputs: Vec<u8> = Vec::with_capacity(64);

    while let Some(splitter_block) = worklist.pop_front() {
        let splitter_idx = splitter_block as usize;
        if splitter_idx >= in_worklist.len() {
            continue;
        }
        in_worklist[splitter_idx] = false;

        if splitter_idx >= blocks.len() || blocks[splitter_idx].is_empty() {
            continue;
        }
        let splitter_states = blocks[splitter_idx].clone();

        touched_inputs.clear();
        for &target in &splitter_states {
            for &(input, src) in &inverse[target as usize] {
                let bucket = &mut input_sources[input as usize];
                if bucket.is_empty() {
                    touched_inputs.push(input);
                }
                bucket.push(src);
            }
        }

        if touched_inputs.is_empty() {
            continue;
        }

        for &input in &touched_inputs {
            sources_to_clear.clear();
            let bucket = &mut input_sources[input as usize];
            for &src in bucket.iter() {
                if !source_set[src as usize] {
                    source_set[src as usize] = true;
                    sources_to_clear.push(src);

                    let block_id = partition[src as usize] as usize;
                    if block_id < block_touched.len() && !block_touched[block_id] {
                        block_touched[block_id] = true;
                        touched_blocks.push(block_id as u32);
                    }
                    block_sources[block_id].push(src);
                }
            }
            bucket.clear();

            for &block_id in &touched_blocks {
                let block_idx = block_id as usize;
                if block_idx >= blocks.len() {
                    continue;
                }
                let block_len = blocks[block_idx].len();
                if block_len <= 1 {
                    continue;
                }

                let source_count = block_sources[block_idx].len();
                if source_count == 0 || source_count == block_len {
                    continue;
                }

                let new_block_idx = blocks.len();
                let move_sources = source_count <= block_len - source_count;
                let old_block = std::mem::take(&mut blocks[block_idx]);

                let (remaining, new_block) = if move_sources {
                    let mut remaining = Vec::with_capacity(block_len - source_count);
                    for state in old_block {
                        if !source_set[state as usize] {
                            remaining.push(state);
                        }
                    }
                    (remaining, std::mem::take(&mut block_sources[block_idx]))
                } else {
                    let mut new_block = Vec::with_capacity(block_len - source_count);
                    for state in old_block {
                        if !source_set[state as usize] {
                            new_block.push(state);
                        }
                    }
                    (std::mem::take(&mut block_sources[block_idx]), new_block)
                };

                for &state in &new_block {
                    partition[state as usize] = new_block_idx as u32;
                }

                blocks[block_idx] = remaining;
                blocks.push(new_block);

                in_worklist.push(false);
                block_touched.push(false);
                block_sources.push(Vec::new());

                if in_worklist[block_idx] {
                    in_worklist[new_block_idx] = true;
                    worklist.push_back(new_block_idx as u32);
                } else if blocks[block_idx].len() <= blocks[new_block_idx].len() {
                    in_worklist[block_idx] = true;
                    worklist.push_back(block_idx as u32);
                } else {
                    in_worklist[new_block_idx] = true;
                    worklist.push_back(new_block_idx as u32);
                }
            }

            for &src in &sources_to_clear {
                source_set[src as usize] = false;
            }

            for &block_id in &touched_blocks {
                if (block_id as usize) < block_touched.len() {
                    block_touched[block_id as usize] = false;
                    block_sources[block_id as usize].clear();
                }
            }
            touched_blocks.clear();
        }
    }

    blocks
}

fn compute_tarjan_scc_ids(adj: &[Vec<usize>]) -> (Vec<u32>, u32) {
    let num_states = adj.len();
    let mut scc_id = vec![u32::MAX; num_states];
    let mut scc_count: u32 = 0;
    let mut index_counter: u32 = 0;
    let mut stack: Vec<usize> = Vec::new();
    let mut on_stack = vec![false; num_states];
    let mut lowlink = vec![0u32; num_states];
    let mut disc = vec![u32::MAX; num_states];
    let mut dfs_stack: Vec<(usize, usize)> = Vec::new();

    for root in 0..num_states {
        if disc[root] != u32::MAX {
            continue;
        }
        dfs_stack.push((root, 0));
        disc[root] = index_counter;
        lowlink[root] = index_counter;
        index_counter += 1;
        stack.push(root);
        on_stack[root] = true;

        while let Some(&mut (state, ref mut edge_index)) = dfs_stack.last_mut() {
            if *edge_index < adj[state].len() {
                let target = adj[state][*edge_index];
                *edge_index += 1;
                if disc[target] == u32::MAX {
                    disc[target] = index_counter;
                    lowlink[target] = index_counter;
                    index_counter += 1;
                    stack.push(target);
                    on_stack[target] = true;
                    dfs_stack.push((target, 0));
                } else if on_stack[target] {
                    lowlink[state] = lowlink[state].min(disc[target]);
                }
            } else {
                if lowlink[state] == disc[state] {
                    while let Some(member) = stack.pop() {
                        on_stack[member] = false;
                        scc_id[member] = scc_count;
                        if member == state {
                            break;
                        }
                    }
                    scc_count += 1;
                }
                dfs_stack.pop();
                if let Some(&mut (parent, _)) = dfs_stack.last_mut() {
                    lowlink[parent] = lowlink[parent].min(lowlink[state]);
                }
            }
        }
    }

    (scc_id, scc_count)
}

impl DFA {
    /// Count distinct (transitions, finalizers) fingerprints via FxHasher.
    /// This is a LOWER bound on the number of truly distinct states (hash
    /// collisions can only reduce the count). Useful for predicting whether
    /// minimize would produce any reduction.
    pub(super) fn distinct_fingerprint_count(&self) -> usize {
        let n = self.states().len();
        if n <= 1 {
            return n;
        }

        let states = self.states();
        let mut seen = FxHashSet::default();
        seen.reserve(n);
        for state in states {
            let mut hasher = rustc_hash::FxHasher::default();
            for (byte, &target) in state.transitions.iter() {
                hasher.write_u8(byte);
                hasher.write_u32(target);
            }
            hasher.write_u8(0xFF);
            hasher.write_usize(state.transitions.len());
            state.finalizers.hash(&mut hasher);
            seen.insert(hasher.finish());
        }

        seen.len()
    }

    /// Minimize this DFA using Hopcroft's algorithm.
    /// Returns a new, minimized DFA.  State 0 remains the start state.
    pub(super) fn minimize(&self) -> DFA {
        self.minimize_impl(true, false).0
    }

    /// Minimize this DFA and return the mapping from original states to
    /// minimized states.  `mapping[old_state] = new_state`.
    /// Unreachable original states map to `u32::MAX`.
    pub(super) fn minimize_with_state_mapping(&self) -> (DFA, Vec<u32>) {
        self.minimize_impl(true, false)
    }

    /// Minimize while preserving current future-terminal labels as frozen observations.
    pub(super) fn minimize_with_state_mapping_preserving_possible_futures(
        &self,
    ) -> (DFA, Vec<u32>) {
        self.minimize_impl(true, true)
    }

    /// Byte-restricted partition DFAs are entered from raw lexer continuation
    /// states produced by other partitions. Retain every original state while
    /// also preserving its frozen future-terminal observation label.
    pub(super) fn minimize_with_state_mapping_preserving_all_states_and_possible_futures(
        &self,
    ) -> (DFA, Vec<u32>) {
        self.minimize_impl(false, true)
    }

    /// Minimize this DFA and return the mapping from original states to
    /// minimized states without first dropping states unreachable from state 0.
    ///
    /// This is useful when callers still need mappings for continuation states
    /// that are only reachable after resuming from inside a partially matched
    /// terminal.
    pub(super) fn minimize_with_state_mapping_preserve_all_states(&self) -> (DFA, Vec<u32>) {
        self.minimize_impl(false, false)
    }

    fn minimize_impl(&self, drop_unreachable: bool, preserve_possible_futures: bool) -> (DFA, Vec<u32>) {
        let orig_n = self.states().len();
        if orig_n == 0 {
            return (self.clone(), Vec::new());
        }

        let mut working = self.clone();
        let old_to_working = if drop_unreachable {
            working.remove_unreachable_states_with_mapping()
        } else {
            (0..orig_n as u32).collect()
        };
        if !preserve_possible_futures {
            clear_possible_futures_for_minimization(&mut working);
        }
        let n = working.states().len();

        if n <= 1 {
            working.recompute_possible_futures();
            return (working, old_to_working);
        }

        let (partition, blocks) = partition_by_finalizers(&working);
        let (partition, blocks) = if preserve_possible_futures {
            refine_partition_by_possible_futures(&working, blocks)
        } else {
            (partition, blocks)
        };
        let mut minimality_check_blocks = blocks.clone();

        match topology_prerefine_partition(&working, &partition) {
            TopologyPrerefine::AlreadyMinimal(blocks) => {
                let (result, block_map) = working.rebuild_from_blocks_with_mapping_and_future_mode(
                    blocks,
                    preserve_possible_futures,
                );
                let composed = compose_mappings(&old_to_working, &block_map);
                return (result, composed);
            }
            TopologyPrerefine::Refined { blocks: refined_blocks, .. } => {
                // Previously bailed out when refined_blocks.len() > n*9/10,
                // assuming the DFA was near-minimal. That heuristic was
                // unsound: topology_prerefine uses a one-pass signature that
                // over-splits when blocks depend on each other's refinement
                // (product-DFA states whose active language is equal but
                // whose one-pass signatures differ due to inactive-dimension
                // coupling). Always fall through to Hopcroft.
                minimality_check_blocks = refined_blocks;
            }
            TopologyPrerefine::Skip => {}
        }

        if minimality_check_blocks.iter().all(|block| block.len() <= 1) {
            let (result, block_map) = working.rebuild_from_blocks_with_mapping(minimality_check_blocks);
            let composed = compose_mappings(&old_to_working, &block_map);
            return (result, composed);
        }

        let blocks = hopcroft_refine_partition(&working, partition, blocks);

        let (result, block_map) = working.rebuild_from_blocks_with_mapping_and_future_mode(
                    blocks,
                    preserve_possible_futures,
                );
        let composed = compose_mappings(&old_to_working, &block_map);
        (result, composed)
    }

    /// Remove unreachable states, returning old→new mapping.
    /// Unreachable states map to `u32::MAX`.
    fn remove_unreachable_states_with_mapping(&mut self) -> Vec<u32> {
        self.remove_unreachable_states_with_roots_with_mapping(&[])
    }

    /// Remove states unreachable from state 0 or any provided extra roots,
    /// returning old→new mapping. Unreachable states map to `u32::MAX`.
    fn remove_unreachable_states_with_roots_with_mapping(&mut self, extra_roots: &[u32]) -> Vec<u32> {
        let n = self.states().len();
        let mut reachable = vec![false; n];
        let mut queue = vec![0usize];
        reachable[0] = true;
        for &root in extra_roots {
            let root = root as usize;
            if root < n && !reachable[root] {
                reachable[root] = true;
                queue.push(root);
            }
        }

        while let Some(state) = queue.pop() {
            for (_, &next) in self.states()[state].transitions.iter() {
                let next = next as usize;
                if !reachable[next] {
                    reachable[next] = true;
                    queue.push(next);
                }
            }
        }

        let mut state_mapping = vec![u32::MAX; n];
        if reachable.iter().all(|&is_reachable| is_reachable) {
            // All reachable — identity mapping.
            for i in 0..n {
                state_mapping[i] = i as u32;
            }
            return state_mapping;
        }

        let mut new_index: u32 = 0;
        for (old_index, &is_reachable) in reachable.iter().enumerate() {
            if is_reachable {
                state_mapping[old_index] = new_index;
                new_index += 1;
            }
        }

        let old_states = std::mem::take(self.states_mut());
        let mut new_states = Vec::with_capacity(new_index as usize);
        for (old_index, state) in old_states.into_iter().enumerate() {
            if reachable[old_index] {
                let mut new_state = state;
                let entries: Vec<(u8, u32)> = new_state
                    .transitions
                    .iter()
                    .map(|(byte, &next)| (byte, state_mapping[next as usize]))
                    .collect();
                new_state.transitions = CharTransitions::from_sorted_entries(entries);
                new_states.push(new_state);
            }
        }

        *self.states_mut() = new_states;
        state_mapping
    }

    /// Recompute `possible_future_group_ids` for all states via fixpoint.
    pub(super) fn recompute_possible_futures(&mut self) {
        let n = self.states().len();
        let num_groups = self.num_groups();
        if n == 0 {
            return;
        }

        let adj = dedup_adjacency(self);
        let (scc_id, scc_count) = compute_tarjan_scc_ids(&adj);

        // `possible_future_group_ids` is strict: it should include only
        // groups reachable after consuming at least one more byte. That means
        // an accepting sink state has no possible futures, while a cyclic SCC
        // can include its own finalizers because they are reachable again via
        // a non-empty path through the cycle.
        let mut scc_finalizers: Vec<BitSet> = (0..scc_count as usize)
            .map(|_| BitSet::new(num_groups))
            .collect();
        let mut scc_sizes = vec![0usize; scc_count as usize];
        let mut scc_has_self_loop = vec![false; scc_count as usize];
        for (state_idx, state) in self.states().iter().enumerate() {
            let sid = scc_id[state_idx] as usize;
            scc_sizes[sid] += 1;
            for bit in state.finalizers.iter() {
                scc_finalizers[sid].set(bit);
            }
            if state
                .transitions
                .iter()
                .any(|(_, &target)| target as usize == state_idx)
            {
                scc_has_self_loop[sid] = true;
            }
        }

        let mut scc_futures: Vec<BitSet> = (0..scc_count as usize)
            .map(|_| BitSet::new(num_groups))
            .collect();
        for sid in 0..scc_count as usize {
            let is_cyclic = scc_sizes[sid] > 1 || scc_has_self_loop[sid];
            if is_cyclic {
                scc_futures[sid].union_with(&scc_finalizers[sid]);
            }
        }

        // Build SCC adjacency (successor SCCs for each SCC)
        let mut scc_successors: Vec<Vec<u32>> = vec![vec![]; scc_count as usize];
        for (state_idx, targets) in adj.iter().enumerate() {
            let src_scc = scc_id[state_idx];
            for &target in targets {
                let dst_scc = scc_id[target];
                if src_scc != dst_scc {
                    scc_successors[src_scc as usize].push(dst_scc);
                }
            }
        }
        // Dedup successors
        for succs in &mut scc_successors {
            succs.sort_unstable();
            succs.dedup();
        }

        let mut scc_predecessors: Vec<Vec<u32>> = vec![vec![]; scc_count as usize];
        let mut remaining_successors = vec![0usize; scc_count as usize];
        for (sid, successors) in scc_successors.iter().enumerate() {
            remaining_successors[sid] = successors.len();
            for &succ in successors {
                scc_predecessors[succ as usize].push(sid as u32);
            }
        }

        // Process SCCs from sinks upward so every predecessor sees fully
        // computed successor futures.
        let mut queue: VecDeque<u32> = remaining_successors
            .iter()
            .enumerate()
            .filter_map(|(sid, &count)| (count == 0).then_some(sid as u32))
            .collect();

        while let Some(sid) = queue.pop_front() {
            let sid = sid as usize;
            let sid_finalizers = scc_finalizers[sid].clone();
            let sid_futures = scc_futures[sid].clone();

            for &pred in &scc_predecessors[sid] {
                let pred = pred as usize;
                scc_futures[pred].union_with(&sid_finalizers);
                scc_futures[pred].union_with(&sid_futures);
                remaining_successors[pred] -= 1;
                if remaining_successors[pred] == 0 {
                    queue.push_back(pred as u32);
                }
            }
        }

        // Assign futures to states
        for state_idx in 0..n {
            let sid = scc_id[state_idx] as usize;
            self.set_possible_future_group_ids(state_idx as u32, scc_futures[sid].clone());
        }
    }

    /// Rebuild DFA from partition blocks.
    /// Ensures state 0 in the new DFA corresponds to the block
    /// containing old state 0.
    fn rebuild_from_blocks(&self, partition_blocks: Vec<Vec<u32>>) -> DFA {
        self.rebuild_from_blocks_with_mapping(partition_blocks).0
    }

    /// Like `rebuild_from_blocks` but also returns old→new state mapping.
    fn rebuild_from_blocks_with_mapping(&self, partition_blocks: Vec<Vec<u32>>) -> (DFA, Vec<u32>) {
        self.rebuild_from_blocks_with_mapping_and_future_mode(partition_blocks, false)
    }

    fn rebuild_from_blocks_with_mapping_and_future_mode(
        &self,
        mut partition_blocks: Vec<Vec<u32>>,
        preserve_possible_futures: bool,
    ) -> (DFA, Vec<u32>) {
        let n = self.states().len();
        let mut state_mapping = vec![0u32; n];

        partition_blocks.retain(|block| !block.is_empty());

        // Ensure the block containing start state (0) is first.
        if let Some(start_part_idx) = partition_blocks
            .iter()
            .position(|block| block.iter().any(|&state| state == 0))
        {
            partition_blocks.swap(0, start_part_idx);
        }

        for (new_idx, block) in partition_blocks.iter().enumerate() {
            for &old_idx in block {
                state_mapping[old_idx as usize] = new_idx as u32;
            }
        }

        let num_groups = self.num_groups();
        let mut result = DFA::new(0);
        result.ensure_group_capacity(num_groups);
        // Copy group_id_to_u8set
        for gid in 0..num_groups {
            result.set_group_u8set(gid as u32, self.group_id_to_u8set(gid as u32).clone());
        }

        for block in &partition_blocks {
            let representative = block[0] as usize;
            let old_state = &self.states()[representative];

            let new_id = result.add_state();
            let new_state = &mut result.states_mut()[new_id as usize];
            new_state.finalizers = old_state.finalizers.clone();
            let entries: Vec<(u8, u32)> = old_state
                .transitions
                .iter()
                .map(|(byte, &old_next)| (byte, state_mapping[old_next as usize]))
                .collect();
            new_state.transitions = CharTransitions::from_sorted_entries(entries);
            if preserve_possible_futures {
                result.set_possible_future_group_ids(
                    new_id,
                    self.possible_future_group_ids(representative as u32).clone(),
                );
            }
        }

        if !preserve_possible_futures {
            result.recompute_possible_futures();
        }
        (result, state_mapping)
    }
}

/// Compose two state mappings: first[i] → second[first[i]].
/// Entries with `u32::MAX` in `first` stay as `u32::MAX`.
fn compose_mappings(first: &[u32], second: &[u32]) -> Vec<u32> {
    first
        .iter()
        .map(|&f| {
            if f == u32::MAX {
                u32::MAX
            } else {
                second[f as usize]
            }
        })
        .collect()
}
