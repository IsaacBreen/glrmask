//! Lexer NFA -> DFA determinization for glrmask's byte-oriented DFA types.

use std::collections::VecDeque;
use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::ds::bitset::BitSet;
use crate::ds::compressed_state_set::{CompressedStateSet, SparseStateSet};
use crate::ds::u8set::U8Set;

use super::dfa::DFA;
use super::nfa::{CompactNFA, NFA};

const EPSILON_CLOSURE_PRECOMPUTE_THRESHOLD: u32 = 1;

fn debug_profile_enabled() -> bool {
    std::env::var("GLRMASK_DEBUG_PROFILE")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn sparse_to_sorted_vec(set: &SparseStateSet) -> Vec<u32> {
    let mut states = Vec::new();
    for &word_idx in &set.dirty_words {
        let mut word = set.words[word_idx];
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            word &= !(1u64 << bit);
            states.push((word_idx * 64 + bit) as u32);
        }
    }
    states.sort_unstable();
    states
}

fn compute_subset_metadata(
    nfa: &NFA,
    subset: &CompressedStateSet,
    group_count: usize,
    reachable_groups: &[BitSet],
) -> (BitSet, BitSet) {
    let mut finalizers = BitSet::new(group_count);
    let mut future = BitSet::new(group_count);

    for state in subset.iter() {
        for &group in &nfa.states[state].finalizers {
            finalizers.set(group as usize);
        }
        future.union_with(&reachable_groups[state]);
    }

    (finalizers, future)
}

fn compute_reachable_groups(nfa: &NFA, group_count: usize) -> Vec<BitSet> {
    let num_states = nfa.states.len();
    if num_states == 0 {
        return Vec::new();
    }
    if group_count == 0 {
        return vec![BitSet::new(0); num_states];
    }
    if group_count == 1 {
        let mut reverse_adj = vec![Vec::new(); num_states];
        let mut reaches_final = vec![false; num_states];
        let mut queue = VecDeque::new();

        for (state_idx, state) in nfa.states.iter().enumerate() {
            for &target in &state.epsilon_transitions {
                reverse_adj[target as usize].push(state_idx);
            }
            for (_, target) in &state.transitions {
                reverse_adj[*target as usize].push(state_idx);
            }
            if !state.finalizers.is_empty() {
                reaches_final[state_idx] = true;
                queue.push_back(state_idx);
            }
        }

        while let Some(state) = queue.pop_front() {
            for &pred in &reverse_adj[state] {
                if !reaches_final[pred] {
                    reaches_final[pred] = true;
                    queue.push_back(pred);
                }
            }
        }

        return reaches_final
            .into_iter()
            .map(|reachable| {
                let mut bits = BitSet::new(1);
                if reachable {
                    bits.set(0);
                }
                bits
            })
            .collect();
    }

    let adj: Vec<Vec<usize>> = nfa
        .states
        .iter()
        .map(|state| {
            let mut targets: Vec<usize> = state
                .epsilon_transitions
                .iter()
                .map(|&target| target as usize)
                .collect();
            targets.extend(state.transitions.iter().map(|(_, target)| *target as usize));
            targets.sort_unstable();
            targets.dedup();
            targets
        })
        .collect();

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

        while let Some(&mut (state, ref mut child_index)) = dfs_stack.last_mut() {
            if *child_index < adj[state].len() {
                let next = adj[state][*child_index];
                *child_index += 1;
                if disc[next] == u32::MAX {
                    disc[next] = index_counter;
                    lowlink[next] = index_counter;
                    index_counter += 1;
                    stack.push(next);
                    on_stack[next] = true;
                    dfs_stack.push((next, 0));
                } else if on_stack[next] {
                    lowlink[state] = lowlink[state].min(disc[next]);
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

    let mut scc_reachable: Vec<BitSet> = (0..scc_count as usize)
        .map(|_| BitSet::new(group_count))
        .collect();
    for (state_idx, state) in nfa.states.iter().enumerate() {
        let sid = scc_id[state_idx] as usize;
        for &group in &state.finalizers {
            scc_reachable[sid].set(group as usize);
        }
    }

    let mut scc_successors: Vec<Vec<u32>> = vec![Vec::new(); scc_count as usize];
    for (state_idx, targets) in adj.iter().enumerate() {
        let src_scc = scc_id[state_idx];
        for &target in targets {
            let dst_scc = scc_id[target];
            if src_scc != dst_scc {
                scc_successors[src_scc as usize].push(dst_scc);
            }
        }
    }
    for successors in &mut scc_successors {
        successors.sort_unstable();
        successors.dedup();
    }

    let mut scc_predecessors: Vec<Vec<u32>> = vec![Vec::new(); scc_count as usize];
    let mut remaining_successors = vec![0usize; scc_count as usize];
    for (sid, successors) in scc_successors.iter().enumerate() {
        remaining_successors[sid] = successors.len();
        for &succ in successors {
            scc_predecessors[succ as usize].push(sid as u32);
        }
    }

    let mut queue: VecDeque<u32> = remaining_successors
        .iter()
        .enumerate()
        .filter_map(|(sid, &count)| (count == 0).then_some(sid as u32))
        .collect();

    while let Some(sid) = queue.pop_front() {
        let sid = sid as usize;
        let sid_reachable = scc_reachable[sid].clone();

        for &pred in &scc_predecessors[sid] {
            let pred = pred as usize;
            scc_reachable[pred].union_with(&sid_reachable);
            remaining_successors[pred] -= 1;
            if remaining_successors[pred] == 0 {
                queue.push_back(pred as u32);
            }
        }
    }

    (0..num_states)
        .map(|state_idx| scc_reachable[scc_id[state_idx] as usize].clone())
        .collect()
}

fn is_epsilon_free_deterministic(nfa: &NFA) -> bool {
    nfa.states.iter().all(|state| {
        if !state.epsilon_transitions.is_empty() {
            return false;
        }

        let mut covered = U8Set::empty();
        for (set, _) in &state.transitions {
            if !covered.is_disjoint(set) {
                return false;
            }
            covered |= *set;
        }
        true
    })
}

fn determinize_epsilon_free_deterministic(nfa: &NFA, group_count: usize, reachable_groups: &[BitSet]) -> DFA {
    let num_states = nfa.states.len().max(1);
    let mut order = Vec::with_capacity(num_states);
    order.push(nfa.start_state as usize);
    for state_id in 0..nfa.states.len() {
        if state_id != nfa.start_state as usize {
            order.push(state_id);
        }
    }

    let mut remap = vec![0u32; nfa.states.len()];
    for (new_state, &old_state) in order.iter().enumerate() {
        remap[old_state] = new_state as u32;
    }

    let mut dfa = DFA::new(num_states);
    dfa.ensure_group_capacity(group_count);

    for (new_state, &old_state) in order.iter().enumerate() {
        let nfa_state = &nfa.states[old_state];
        let mut finalizers = BitSet::new(group_count);
        for &group in &nfa_state.finalizers {
            finalizers.set(group as usize);
        }

        let mut transitions = Vec::new();
        for (set, target) in &nfa_state.transitions {
            transitions.reserve(set.len());
            for byte in set.iter() {
                transitions.push((byte, remap[*target as usize]));
            }
        }
        if transitions.len() > 1 {
            transitions.sort_unstable_by_key(|entry| entry.0);
        }

        dfa.overwrite_state_metadata(new_state as u32, finalizers, reachable_groups[old_state].clone());
        dfa.set_transitions_from_sorted_entries(new_state as u32, transitions);
    }

    dfa
}

fn build_remapped_transitions(nfa: &NFA, class_map: &[u8]) -> Vec<Vec<(U8Set, u32)>> {
    let mut remapped_set_cache: FxHashMap<U8Set, U8Set> = FxHashMap::default();

    nfa.states
        .iter()
        .map(|state| {
            state
                .transitions
                .iter()
                .map(|(set, target)| {
                    let class_set = *remapped_set_cache.entry(*set).or_insert_with(|| {
                        let mut class_set = U8Set::empty();
                        for byte in set.iter() {
                            class_set.insert(class_map[byte as usize]);
                        }
                        class_set
                    });
                    (class_set, *target)
                })
                .collect()
        })
        .collect()
}

fn dfs_selective_post_order(
    state: usize,
    compact_nfa: &CompactNFA,
    out_degree: &[u32],
    threshold: u32,
    visited: &mut [bool],
    post_order: &mut Vec<usize>,
) {
    if visited[state] {
        return;
    }
    visited[state] = true;

    let start = compact_nfa.epsilon_offsets[state] as usize;
    let end = compact_nfa.epsilon_offsets[state + 1] as usize;
    for &target in &compact_nfa.epsilon_targets[start..end] {
        dfs_selective_post_order(
            target as usize,
            compact_nfa,
            out_degree,
            threshold,
            visited,
            post_order,
        );
    }

    if out_degree[state] >= threshold {
        post_order.push(state);
    }
}

fn precompute_epsilon_closures(
    compact_nfa: &CompactNFA,
    out_degree: &[u32],
) -> Vec<Option<Vec<u32>>> {
    let num_states = out_degree.len();
    let mut high_degree_closures: Vec<Option<Vec<u32>>> = vec![None; num_states];
    let mut visited = vec![false; num_states];
    let mut post_order = Vec::with_capacity(num_states);

    for state in 0..num_states {
        if out_degree[state] < EPSILON_CLOSURE_PRECOMPUTE_THRESHOLD {
            continue;
        }

        dfs_selective_post_order(
            state,
            compact_nfa,
            out_degree,
            EPSILON_CLOSURE_PRECOMPUTE_THRESHOLD,
            &mut visited,
            &mut post_order,
        );
    }

    let mut epsilon_stack = Vec::with_capacity(num_states);
    let mut closure = SparseStateSet::new(num_states);
    for &state in &post_order {
        closure.clear();
        closure.insert(state);
        epsilon_stack.push(state);

        while let Some(current) = epsilon_stack.pop() {
            let start = compact_nfa.epsilon_offsets[current] as usize;
            let end = compact_nfa.epsilon_offsets[current + 1] as usize;
            for index in start..end {
                let next = compact_nfa.epsilon_targets[index] as usize;
                if closure.insert(next) {
                    if let Some(ref precomputed) = high_degree_closures[next] {
                        closure.insert_many(precomputed);
                    } else if out_degree[next] > 0 {
                        epsilon_stack.push(next);
                    }
                }
            }
        }

        high_degree_closures[state] = Some(sparse_to_sorted_vec(&closure));
    }

    high_degree_closures
}

fn build_start_closure(
    compact_nfa: &CompactNFA,
    start_state: usize,
    out_degree: &[u32],
) -> SparseStateSet {
    let num_states = out_degree.len();
    let mut closure = SparseStateSet::new(num_states);
    let mut epsilon_stack = Vec::with_capacity(num_states);

    closure.insert(start_state);
    if out_degree[start_state] > 0 {
        epsilon_stack.push(start_state);
    }

    while let Some(current) = epsilon_stack.pop() {
        let start = compact_nfa.epsilon_offsets[current] as usize;
        let end = compact_nfa.epsilon_offsets[current + 1] as usize;
        for index in start..end {
            let next = compact_nfa.epsilon_targets[index] as usize;
            if closure.insert(next) && out_degree[next] > 0 {
                epsilon_stack.push(next);
            }
        }
    }

    closure
}

fn collect_transition_targets(
    subset: &CompressedStateSet,
    remapped_transitions: &[Vec<(U8Set, u32)>],
    transition_targets: &mut [SparseStateSet],
    used_classes: &mut Vec<usize>,
    seen_class: &mut [bool],
) {
    for state in subset.iter() {
        for (class_set, next_state) in &remapped_transitions[state] {
            for class_id in class_set.iter() {
                let class_index = class_id as usize;
                if !seen_class[class_index] {
                    seen_class[class_index] = true;
                    used_classes.push(class_index);
                }
                transition_targets[class_index].insert(*next_state as usize);
            }
        }
    }
}

fn fast_singleton_without_epsilon(
    target_set: &SparseStateSet,
    out_degree: &[u32],
    precomputed_closures: &[Option<Vec<u32>>],
) -> Option<usize> {
    if target_set.dirty_words.len() != 1 {
        return None;
    }

    let word_index = target_set.dirty_words[0];
    let word = target_set.words[word_index];
    if word == 0 || (word & (word - 1)) != 0 {
        return None;
    }

    let bit = word.trailing_zeros() as usize;
    let state = word_index * 64 + bit;
    (out_degree[state] == 0 && precomputed_closures[state].is_none()).then_some(state)
}

fn expand_transition_closure(
    target_set: &SparseStateSet,
    compact_nfa: &CompactNFA,
    out_degree: &[u32],
    precomputed_closures: &[Option<Vec<u32>>],
    closure: &mut SparseStateSet,
    epsilon_stack: &mut Vec<usize>,
) {
    closure.clear();
    let mut needs_bfs = false;

    for &word_index in &target_set.dirty_words {
        let mut word = target_set.words[word_index];
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            word &= !(1u64 << bit);
            let next_state = word_index * 64 + bit;

            if let Some(ref precomputed) = precomputed_closures[next_state] {
                closure.insert_many(precomputed);
            } else {
                closure.insert(next_state);
                if out_degree[next_state] > 0 {
                    needs_bfs = true;
                    epsilon_stack.push(next_state);
                }
            }
        }
    }

    if !needs_bfs {
        return;
    }

    while let Some(current) = epsilon_stack.pop() {
        let start = compact_nfa.epsilon_offsets[current] as usize;
        let end = compact_nfa.epsilon_offsets[current + 1] as usize;
        for index in start..end {
            let next = compact_nfa.epsilon_targets[index] as usize;
            if closure.insert(next) {
                if let Some(ref precomputed) = precomputed_closures[next] {
                    closure.insert_many(precomputed);
                } else if out_degree[next] > 0 {
                    epsilon_stack.push(next);
                }
            }
        }
    }
}

impl NFA {
    pub fn to_dfa(&self) -> DFA {
        let debug_profile = debug_profile_enabled();
        let total_started_at = Instant::now();
        let group_count = self
            .states
            .iter()
            .flat_map(|state| state.finalizers.iter())
            .max()
            .map(|group| *group as usize + 1)
            .unwrap_or(0);

        let reachable_started_at = Instant::now();
        let reachable_groups = compute_reachable_groups(self, group_count);
        let reachable_ms = elapsed_ms(reachable_started_at);

        let epsilon_edges: usize = self
            .states
            .iter()
            .map(|state| state.epsilon_transitions.len())
            .sum();
        let deterministic_no_epsilon = is_epsilon_free_deterministic(self);

        if deterministic_no_epsilon {
            let fast_started_at = Instant::now();
            let dfa = determinize_epsilon_free_deterministic(self, group_count, &reachable_groups);
            if debug_profile {
                eprintln!(
                    "[glrmask/debug][determinize] states={} epsilon_edges={} fast_path=epsilon_free_deterministic reachable_ms={:.3} fast_ms={:.3} total_ms={:.3} dfa_states={}",
                    self.states.len(),
                    epsilon_edges,
                    reachable_ms,
                    elapsed_ms(fast_started_at),
                    elapsed_ms(total_started_at),
                    dfa.num_states(),
                );
            }
            return dfa;
        }

        let mut dfa = DFA::new(1);
        dfa.ensure_group_capacity(group_count);

        let classes_started_at = Instant::now();
        let (class_map, num_classes, class_members) = self.compute_equivalence_classes();
        let remapped_transitions = build_remapped_transitions(self, &class_map);
        let classes_ms = elapsed_ms(classes_started_at);

        let epsilon_setup_started_at = Instant::now();
        let compact_nfa = self.build_compact_nfa();
        let num_nfa_states = self.states.len();
        let mut epsilon_stack = Vec::with_capacity(num_nfa_states);

        let out_degree: Vec<u32> = (0..num_nfa_states)
            .map(|state| compact_nfa.epsilon_offsets[state + 1] - compact_nfa.epsilon_offsets[state])
            .collect();

        let high_degree_closures = precompute_epsilon_closures(&compact_nfa, &out_degree);
        let start_closure = build_start_closure(
            &compact_nfa,
            self.start_state as usize,
            &out_degree,
        );
        let epsilon_setup_ms = elapsed_ms(epsilon_setup_started_at);

        let start_key = CompressedStateSet::from_sparse(&start_closure);
        let (start_finalizers, start_future) =
            compute_subset_metadata(self, &start_key, group_count, &reachable_groups);
        dfa.overwrite_state_metadata(0, start_finalizers, start_future);

        let mut subset_map = FxHashMap::<CompressedStateSet, u32>::default();
        subset_map.reserve(num_nfa_states);
        let mut worklist = Vec::with_capacity(num_nfa_states);
        subset_map.insert(start_key.clone(), 0);
        worklist.push((0, start_key));

        let mut transition_targets: Vec<SparseStateSet> = (0..num_classes)
            .map(|_| SparseStateSet::new(num_nfa_states))
            .collect();
        let mut used_classes = Vec::with_capacity(num_classes);
        let mut seen_class = vec![false; num_classes];
        let mut closure_set = SparseStateSet::new(num_nfa_states);
        let mut scratch_closure = CompressedStateSet::new();
        let subset_started_at = Instant::now();
        let mut processed_subsets = 0u64;
        let mut processed_targets = 0u64;
        let mut singleton_targets = 0u64;
        let mut total_subset_words = 0u64;
        let mut total_target_words = 0u64;
        let mut max_subset_words = 0usize;
        let mut max_target_words = 0usize;

        while let Some((current_dfa_state, current_set)) = worklist.pop() {
            processed_subsets += 1;
            let subset_words = current_set.words.len();
            total_subset_words += subset_words as u64;
            max_subset_words = max_subset_words.max(subset_words);

            collect_transition_targets(
                &current_set,
                &remapped_transitions,
                &mut transition_targets,
                &mut used_classes,
                &mut seen_class,
            );

            let mut dfa_transitions_vec = Vec::with_capacity(used_classes.len() * 2);
            for &class_id in &used_classes {
                processed_targets += 1;
                let target_set = &transition_targets[class_id];

                if let Some(state) = fast_singleton_without_epsilon(
                    target_set,
                    &out_degree,
                    &high_degree_closures,
                ) {
                    singleton_targets += 1;
                    scratch_closure.words.clear();
                    let word_idx = (state >> 6) as u32;
                    let mask = 1u64 << (state & 0x3f);
                    scratch_closure.words.push((word_idx, mask));
                    scratch_closure.hash = (word_idx as u64).wrapping_mul(0x517c_c1b7_2722_0a95)
                        ^ mask.wrapping_mul(0x9e37_79b9_7f4a_7c15);
                } else {
                    expand_transition_closure(
                        target_set,
                        &compact_nfa,
                        &out_degree,
                        &high_degree_closures,
                        &mut closure_set,
                        &mut epsilon_stack,
                    );
                    CompressedStateSet::reuse_from_sparse(&closure_set, &mut scratch_closure);
                }

                let target_words = scratch_closure.words.len();
                total_target_words += target_words as u64;
                max_target_words = max_target_words.max(target_words);

                let next_dfa_state = if let Some(&existing) = subset_map.get(&scratch_closure) {
                    existing
                } else {
                    let new_state = dfa.add_state();
                    let key = scratch_closure.clone();
                    let (finalizers, future) =
                        compute_subset_metadata(self, &key, group_count, &reachable_groups);
                    dfa.overwrite_state_metadata(new_state, finalizers, future);
                    subset_map.insert(key.clone(), new_state);
                    worklist.push((new_state, key));
                    new_state
                };

                for &byte in &class_members[class_id] {
                    dfa_transitions_vec.push((byte, next_dfa_state));
                }
            }

            if dfa_transitions_vec.len() > 1 {
                dfa_transitions_vec.sort_unstable_by_key(|entry| entry.0);
            }
            dfa.set_transitions_from_sorted_entries(current_dfa_state, dfa_transitions_vec);

            for &idx in &used_classes {
                seen_class[idx] = false;
                transition_targets[idx].clear();
            }
            used_classes.clear();
        }

        if debug_profile {
            let avg_subset_words = if processed_subsets == 0 {
                0.0
            } else {
                total_subset_words as f64 / processed_subsets as f64
            };
            let avg_target_words = if processed_targets == 0 {
                0.0
            } else {
                total_target_words as f64 / processed_targets as f64
            };
            eprintln!(
                "[glrmask/debug][determinize] states={} epsilon_edges={} fast_path=generic reachable_ms={:.3} classes_ms={:.3} epsilon_setup_ms={:.3} subset_ms={:.3} total_ms={:.3} dfa_states={} processed_subsets={} processed_targets={} singleton_targets={} avg_subset_words={:.2} max_subset_words={} avg_target_words={:.2} max_target_words={}",
                self.states.len(),
                epsilon_edges,
                reachable_ms,
                classes_ms,
                epsilon_setup_ms,
                elapsed_ms(subset_started_at),
                elapsed_ms(total_started_at),
                dfa.num_states(),
                processed_subsets,
                processed_targets,
                singleton_targets,
                avg_subset_words,
                max_subset_words,
                avg_target_words,
                max_target_words,
            );
        }

        dfa
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_dfa_fast_path_remaps_nonzero_start_state() {
        let mut nfa = NFA::new(3);
        nfa.start_state = 1;
        nfa.add_transition(1, b'a', 2);
        nfa.add_finalizer(2, 0);

        let dfa = nfa.to_dfa();

        let next = dfa.step(0, b'a').expect("expected remapped start transition");
        assert!(dfa.finalizers(next).contains(0));
    }
}
