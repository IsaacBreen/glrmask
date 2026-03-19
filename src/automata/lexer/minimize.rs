//! Hopcroft-based DFA minimization for the lexer DFA.
//!
//! Ported from grammars2024/src/dfa_u8/dfa.rs with minimal adaptation.
//! Two–phase approach:
//!   Phase 1: topology-aware pre-refinement (fast path for DAG portions)
//!   Phase 2: Hopcroft refinement (handles cycles)

use std::collections::{BTreeSet, VecDeque};

use rustc_hash::FxHashMap;

use crate::ds::bitset::BitSet;

use super::dfa::DFA;

impl DFA {
    /// Minimize this DFA using Hopcroft's algorithm.
    /// Returns a new, minimized DFA.  State 0 remains the start state.
    pub fn minimize(&self) -> DFA {
        if self.states().is_empty() {
            return self.clone();
        }

        // --- Remove unreachable states ---
        let mut working = self.clone();
        working.remove_unreachable_states();
        let n = working.states().len();

        if n <= 1 {
            working.recompute_possible_futures();
            return working;
        }

        // --- Initial partition: group states by their finalizer set ---
        let mut partition = vec![0u32; n]; // partition[state] = block_id
        let mut blocks: Vec<Vec<u32>> = Vec::new();

        {
            let mut finalizer_to_block: FxHashMap<Vec<usize>, u32> = FxHashMap::default();
            for (state_idx, state) in working.states().iter().enumerate() {
                let key: Vec<usize> = state.finalizers.iter().collect();
                let block_idx = *finalizer_to_block.entry(key).or_insert_with(|| {
                    let idx = blocks.len() as u32;
                    blocks.push(Vec::new());
                    idx
                });
                partition[state_idx] = block_idx;
                blocks[block_idx as usize].push(state_idx as u32);
            }
        }

        // --- Phase 1: Topology-aware pre-refinement ---
        // Compute DFS post-order and detect SCCs

        let adj: Vec<Vec<usize>> = working
            .states()
            .iter()
            .map(|state| {
                let mut targets: Vec<usize> =
                    state.transitions.iter().map(|(_, &t)| t as usize).collect();
                targets.sort_unstable();
                targets.dedup();
                targets
            })
            .collect();

        // Iterative DFS for post-order
        let mut post_order: Vec<usize> = Vec::with_capacity(n);
        let mut visited = vec![0u8; n]; // 0=unvisited, 1=in_stack, 2=done
        let mut dfs_stack: Vec<(usize, usize)> = Vec::new();

        for root in 0..n {
            if visited[root] != 0 {
                continue;
            }
            dfs_stack.push((root, 0));
            visited[root] = 1;

            while let Some((state, ai)) = dfs_stack.last_mut() {
                let state = *state;
                if *ai < adj[state].len() {
                    let target = adj[state][*ai];
                    *ai += 1;
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

        // Kosaraju's pass 2: compute SCCs
        let mut reverse_adj: Vec<Vec<usize>> = vec![vec![]; n];
        for (state, targets) in adj.iter().enumerate() {
            for &target in targets {
                reverse_adj[target].push(state);
            }
        }
        let mut scc_id = vec![u32::MAX; n];
        let mut current_scc: u32 = 0;
        for &state in post_order.iter().rev() {
            if scc_id[state] != u32::MAX {
                continue;
            }
            let mut scc_stack = vec![state];
            scc_id[state] = current_scc;
            while let Some(u) = scc_stack.pop() {
                for &pred in &reverse_adj[u] {
                    if scc_id[pred] == u32::MAX {
                        scc_id[pred] = current_scc;
                        scc_stack.push(pred);
                    }
                }
            }
            current_scc += 1;
        }

        // Compute labels in post-order
        let mut label = vec![u32::MAX; n];
        let mut label_map: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
        label_map.reserve(n.min(200_000));
        let mut num_labels: u32 = 0;
        let mut sig_buf: Vec<u32> = Vec::with_capacity(32);

        for &state in post_order.iter() {
            sig_buf.clear();
            sig_buf.push(partition[state]);

            for (byte, &target) in working.states()[state].transitions.iter() {
                let target = target as usize;
                sig_buf.push(byte as u32);
                if scc_id[state] == scc_id[target] {
                    sig_buf.push(partition[target]);
                } else if label[target] != u32::MAX {
                    sig_buf.push(label[target] | 0x8000_0000);
                } else {
                    sig_buf.push(partition[target]);
                }
            }

            let lbl = match label_map.get(&sig_buf) {
                Some(&l) => l,
                None => {
                    let l = num_labels;
                    num_labels += 1;
                    label_map.insert(sig_buf.clone(), l);
                    l
                }
            };
            label[state] = lbl;
        }

        // Rebuild partition and blocks from labels
        partition = vec![0u32; n];
        blocks = Vec::new();
        blocks.resize(num_labels as usize, Vec::new());
        for state in 0..n {
            let lbl = label[state];
            partition[state] = lbl;
            blocks[lbl as usize].push(state as u32);
        }

        if num_labels as usize == n {
            // Check for self-loops
            let has_self_loop = (0..n).any(|s| {
                working.states()[s]
                    .transitions
                    .iter()
                    .any(|(_, &t)| t as usize == s)
            });

            if !has_self_loop {
                // All singletons AND true DAG: provably already minimal
                let partition_list: Vec<BTreeSet<usize>> = blocks
                    .into_iter()
                    .filter(|b| !b.is_empty())
                    .map(|b| b.into_iter().map(|s| s as usize).collect())
                    .collect();

                let result = working.rebuild_from_partitions(partition_list);
                return result;
            }
            // Self-loops: fall through to Hopcroft
            partition = vec![0u32; n];
            blocks = Vec::new();
            {
                let mut finalizer_to_block: FxHashMap<Vec<usize>, u32> = FxHashMap::default();
                for (state_idx, state) in working.states().iter().enumerate() {
                    let key: Vec<usize> = state.finalizers.iter().collect();
                    let block_idx = *finalizer_to_block.entry(key).or_insert_with(|| {
                        let idx = blocks.len() as u32;
                        blocks.push(Vec::new());
                        idx
                    });
                    partition[state_idx] = block_idx;
                    blocks[block_idx as usize].push(state_idx as u32);
                }
            }
        }

        let non_singletons = blocks.iter().filter(|b| b.len() > 1).count();
        if non_singletons == 0 {
            let partition_list: Vec<BTreeSet<usize>> = blocks
                .into_iter()
                .filter(|b| !b.is_empty())
                .map(|b| b.into_iter().map(|s| s as usize).collect())
                .collect();
            return working.rebuild_from_partitions(partition_list);
        }

        // --- Phase 2: Hopcroft refinement ---
        // Rebuild initial partition from scratch
        partition = vec![0u32; n];
        blocks = Vec::new();
        {
            let mut finalizer_to_block: FxHashMap<Vec<usize>, u32> = FxHashMap::default();
            for (state_idx, state) in working.states().iter().enumerate() {
                let key: Vec<usize> = state.finalizers.iter().collect();
                let block_idx = *finalizer_to_block.entry(key).or_insert_with(|| {
                    let idx = blocks.len() as u32;
                    blocks.push(Vec::new());
                    idx
                });
                partition[state_idx] = block_idx;
                blocks[block_idx as usize].push(state_idx as u32);
            }
        }

        // Build inverse transition table
        let mut inverse: Vec<Vec<(u8, u32)>> = vec![Vec::new(); n];
        for (src, state) in working.states().iter().enumerate() {
            for (input, &target) in state.transitions.iter() {
                inverse[target as usize].push((input, src as u32));
            }
        }
        for inv in &mut inverse {
            inv.sort_unstable_by_key(|&(input, _)| input);
        }

        let mut worklist: VecDeque<u32> = (0..blocks.len() as u32).collect();
        let mut in_worklist = vec![true; blocks.len()];

        let mut source_set = vec![false; n];
        let mut sources_to_clear: Vec<u32> = Vec::with_capacity(n.min(10000));
        let mut touched_blocks: Vec<u32> = Vec::with_capacity(1024);
        let mut block_touched = vec![false; blocks.len()];

        while let Some(splitter_block) = worklist.pop_front() {
            let splitter_idx = splitter_block as usize;
            if splitter_idx >= in_worklist.len() {
                continue;
            }
            in_worklist[splitter_idx] = false;

            if splitter_idx >= blocks.len() || blocks[splitter_idx].is_empty() {
                continue;
            }
            let splitter_states: Vec<u32> = blocks[splitter_idx].clone();

            let mut all_pairs: Vec<(u8, u32)> = Vec::new();
            for &target in &splitter_states {
                all_pairs.extend_from_slice(&inverse[target as usize]);
            }

            if all_pairs.is_empty() {
                continue;
            }

            all_pairs.sort_unstable_by_key(|&(input, _)| input);

            let mut i = 0;
            while i < all_pairs.len() {
                let current_input = all_pairs[i].0;

                sources_to_clear.clear();
                while i < all_pairs.len() && all_pairs[i].0 == current_input {
                    let src = all_pairs[i].1;
                    if !source_set[src as usize] {
                        source_set[src as usize] = true;
                        sources_to_clear.push(src);

                        let block_id = partition[src as usize] as usize;
                        if block_id < block_touched.len() && !block_touched[block_id] {
                            block_touched[block_id] = true;
                            touched_blocks.push(block_id as u32);
                        }
                    }
                    i += 1;
                }

                for &block_id in &touched_blocks {
                    let block_idx = block_id as usize;
                    if block_idx >= blocks.len() {
                        continue;
                    }
                    let block_len = blocks[block_idx].len();
                    if block_len <= 1 {
                        continue;
                    }

                    let mut source_count = 0usize;
                    for &state in &blocks[block_idx] {
                        if source_set[state as usize] {
                            source_count += 1;
                        }
                    }

                    if source_count == 0 || source_count == block_len {
                        continue;
                    }

                    let new_block_idx = blocks.len();
                    let move_sources = source_count <= block_len - source_count;

                    let mut new_block = Vec::with_capacity(if move_sources {
                        source_count
                    } else {
                        block_len - source_count
                    });
                    let mut remaining =
                        Vec::with_capacity(block_len - new_block.capacity());

                    for &state in &blocks[block_idx] {
                        let is_source = source_set[state as usize];
                        if move_sources == is_source {
                            new_block.push(state);
                        } else {
                            remaining.push(state);
                        }
                    }

                    for &state in &new_block {
                        partition[state as usize] = new_block_idx as u32;
                    }

                    blocks[block_idx] = remaining;
                    blocks.push(new_block);

                    in_worklist.push(false);
                    block_touched.push(false);

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
                    }
                }
                touched_blocks.clear();
            }
        }

        let partition_list: Vec<BTreeSet<usize>> = blocks
            .into_iter()
            .filter(|b| !b.is_empty())
            .map(|b| b.into_iter().map(|s| s as usize).collect())
            .collect();

        working.rebuild_from_partitions(partition_list)
    }

    /// Remove states not reachable from state 0.
    fn remove_unreachable_states(&mut self) {
        let n = self.states().len();
        let mut reachable = vec![false; n];
        let mut queue = vec![0usize];
        reachable[0] = true;

        while let Some(state) = queue.pop() {
            for (_, &next) in self.states()[state].transitions.iter() {
                let next = next as usize;
                if !reachable[next] {
                    reachable[next] = true;
                    queue.push(next);
                }
            }
        }

        let mut state_mapping = vec![0u32; n];
        let mut new_index: u32 = 0;
        for (old_index, &is_reachable) in reachable.iter().enumerate() {
            if is_reachable {
                state_mapping[old_index] = new_index;
                new_index += 1;
            }
        }

        let old_states = std::mem::take(self.states_mut());
        let mut new_states = Vec::new();
        for (old_index, state) in old_states.into_iter().enumerate() {
            if reachable[old_index] {
                let mut new_state = state;
                new_state.transitions = new_state
                    .transitions
                    .iter()
                    .map(|(byte, &next)| (byte, state_mapping[next as usize]))
                    .collect();
                new_states.push(new_state);
            }
        }

        *self.states_mut() = new_states;
    }

    /// Recompute `possible_future_group_ids` for all states via fixpoint.
    fn recompute_possible_futures(&mut self) {
        let n = self.states().len();
        let num_groups = self.num_groups();
        if n == 0 {
            return;
        }

        // SCC-based O(n+m) algorithm:
        // 1. Compute SCCs via Tarjan's
        // 2. All states in an SCC share the same futures (mutually reachable)
        // 3. Process DAG of SCCs in reverse topological order (sinks first)

        // Pre-collect adjacency lists for indexed access in Tarjan's
        let adj: Vec<Vec<usize>> = self
            .states()
            .iter()
            .map(|s| {
                let mut targets: Vec<usize> = s.transitions.iter().map(|(_, &t)| t as usize).collect();
                targets.sort_unstable();
                targets.dedup();
                targets
            })
            .collect();

        // Tarjan's SCC (iterative)
        let mut scc_id = vec![u32::MAX; n];
        let mut scc_count: u32 = 0;
        let mut index_counter: u32 = 0;
        let mut stack: Vec<usize> = Vec::new();
        let mut on_stack = vec![false; n];
        let mut lowlink = vec![0u32; n];
        let mut disc = vec![u32::MAX; n]; // discovery index; u32::MAX = undefined
        let mut dfs_stack: Vec<(usize, usize)> = Vec::new(); // (node, adj_idx)

        for root in 0..n {
            if disc[root] != u32::MAX {
                continue;
            }
            dfs_stack.push((root, 0));
            disc[root] = index_counter;
            lowlink[root] = index_counter;
            index_counter += 1;
            stack.push(root);
            on_stack[root] = true;

            while let Some(&mut (v, ref mut ci)) = dfs_stack.last_mut() {
                if *ci < adj[v].len() {
                    let w = adj[v][*ci];
                    *ci += 1;
                    if disc[w] == u32::MAX {
                        disc[w] = index_counter;
                        lowlink[w] = index_counter;
                        index_counter += 1;
                        stack.push(w);
                        on_stack[w] = true;
                        dfs_stack.push((w, 0));
                    } else if on_stack[w] {
                        lowlink[v] = lowlink[v].min(disc[w]);
                    }
                } else {
                    // Backtrack
                    if lowlink[v] == disc[v] {
                        // v is root of SCC
                        while let Some(w) = stack.pop() {
                            on_stack[w] = false;
                            scc_id[w] = scc_count;
                            if w == v {
                                break;
                            }
                        }
                        scc_count += 1;
                    }
                    dfs_stack.pop();
                    if let Some(&mut (parent, _)) = dfs_stack.last_mut() {
                        lowlink[parent] = lowlink[parent].min(lowlink[v]);
                    }
                }
            }
        }

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

    /// Rebuild DFA from partition equivalence classes.
    /// Ensures state 0 in the new DFA corresponds to the partition
    /// containing old state 0.
    fn rebuild_from_partitions(&self, mut partition_list: Vec<BTreeSet<usize>>) -> DFA {
        let n = self.states().len();
        let mut state_mapping = vec![0u32; n];

        // Ensure partition containing start state (0) is first
        if let Some(start_part_idx) = partition_list.iter().position(|p| p.contains(&0)) {
            partition_list.swap(0, start_part_idx);
        }

        for (new_idx, partition) in partition_list.iter().enumerate() {
            for &old_idx in partition {
                state_mapping[old_idx] = new_idx as u32;
            }
        }

        let num_groups = self.num_groups();
        let mut result = DFA::new(0);
        result.ensure_group_capacity(num_groups);
        // Copy group_id_to_u8set
        for gid in 0..num_groups {
            result.set_group_u8set(gid as u32, self.group_id_to_u8set(gid as u32).clone());
        }

        for partition in &partition_list {
            let representative = *partition.iter().next().unwrap();
            let old_state = &self.states()[representative];

            let new_id = result.add_state();
            let new_state = &mut result.states_mut()[new_id as usize];
            new_state.finalizers = old_state.finalizers.clone();
            new_state.transitions = old_state
                .transitions
                .iter()
                .map(|(byte, &old_next)| (byte, state_mapping[old_next as usize]))
                .collect();
        }

        result.recompute_possible_futures();
        result
    }
}
