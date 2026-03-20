//! sep1-style lexer NFA → DFA determinization, adapted to glrmask's leaner DFA types.

use rustc_hash::FxHashMap;

use crate::ds::bitset::BitSet;
use crate::ds::compressed_state_set::{CompressedStateSet, SparseStateSet};
use crate::ds::u8set::U8Set;

use super::dfa::DFA;
use super::nfa::{CompactNFA, NFA};

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

impl NFA {
    pub fn to_dfa(&self) -> DFA {
        let group_count = self
            .states
            .iter()
            .flat_map(|state| state.finalizers.iter())
            .max()
            .map(|group| *group as usize + 1)
            .unwrap_or(0);

        let mut reachable_groups: Vec<BitSet> = (0..self.states.len())
            .map(|_| BitSet::new(group_count))
            .collect();
        let mut changed = true;
        while changed {
            changed = false;
            for state_id in (0..self.states.len()).rev() {
                let mut next = reachable_groups[state_id].clone();
                for &group in &self.states[state_id].finalizers {
                    next.set(group as usize);
                }
                for &next_state in &self.states[state_id].epsilon_transitions {
                    next.union_with(&reachable_groups[next_state as usize]);
                }
                for (_, next_state) in &self.states[state_id].transitions {
                    next.union_with(&reachable_groups[*next_state as usize]);
                }
                if next != reachable_groups[state_id] {
                    reachable_groups[state_id] = next;
                    changed = true;
                }
            }
        }

        let mut dfa = DFA::new(1);
        dfa.ensure_group_capacity(group_count);

        let (class_map, num_classes, class_members) = self.compute_equivalence_classes();
        let remapped_transitions: Vec<Vec<(U8Set, u32)>> = self
            .states
            .iter()
            .map(|state| {
                state
                    .transitions
                    .iter()
                    .map(|(set, target)| {
                        let mut class_set = U8Set::empty();
                        for byte in set.iter() {
                            class_set.insert(class_map[byte as usize]);
                        }
                        (class_set, *target)
                    })
                    .collect()
            })
            .collect();

        let compact_nfa = self.build_compact_nfa();
        let num_nfa_states = self.states.len();
        let mut stack: Vec<usize> = Vec::with_capacity(num_nfa_states);

        let out_degree: Vec<u32> = (0..num_nfa_states)
            .map(|state| compact_nfa.epsilon_offsets[state + 1] - compact_nfa.epsilon_offsets[state])
            .collect();

        let precompute_threshold: u32 = std::env::var("DFA_EPS_PRECOMPUTE_THRESHOLD")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(4);

        let mut high_degree_closures: Vec<Option<Vec<u32>>> = vec![None; num_nfa_states];
        let mut visited = vec![false; num_nfa_states];
        let mut post_order = Vec::with_capacity(num_nfa_states);

        fn dfs_postorder_selective(
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
                dfs_postorder_selective(
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

        for state in 0..num_nfa_states {
            if out_degree[state] >= precompute_threshold {
                dfs_postorder_selective(
                    state,
                    &compact_nfa,
                    &out_degree,
                    precompute_threshold,
                    &mut visited,
                    &mut post_order,
                );
            }
        }

        let mut temp_set = SparseStateSet::new(num_nfa_states);
        for &state in &post_order {
            temp_set.clear();
            temp_set.insert(state);
            stack.push(state);

            while let Some(current) = stack.pop() {
                let start = compact_nfa.epsilon_offsets[current] as usize;
                let end = compact_nfa.epsilon_offsets[current + 1] as usize;
                for index in start..end {
                    let next = compact_nfa.epsilon_targets[index] as usize;
                    if temp_set.insert(next) {
                        if let Some(ref closure) = high_degree_closures[next] {
                            temp_set.insert_many(closure);
                        } else if out_degree[next] > 0 {
                            stack.push(next);
                        }
                    }
                }
            }

            high_degree_closures[state] = Some(sparse_to_sorted_vec(&temp_set));
        }

        let mut start_closure = SparseStateSet::new(num_nfa_states);
        start_closure.insert(self.start_state as usize);
        if out_degree[self.start_state as usize] > 0 {
            stack.push(self.start_state as usize);
            while let Some(current) = stack.pop() {
                let start = compact_nfa.epsilon_offsets[current] as usize;
                let end = compact_nfa.epsilon_offsets[current + 1] as usize;
                for index in start..end {
                    let next = compact_nfa.epsilon_targets[index] as usize;
                    if start_closure.insert(next) {
                        if out_degree[next] > 0 {
                            stack.push(next);
                        }
                    }
                }
            }
        }

        let start_key = CompressedStateSet::from_sparse(&start_closure);
        let (start_finalizers, start_future) =
            compute_subset_metadata(self, &start_key, group_count, &reachable_groups);
        dfa.overwrite_state_metadata(0, start_finalizers, start_future);

        let mut subset_map = FxHashMap::<CompressedStateSet, u32>::default();
        let mut worklist = Vec::new();
        subset_map.insert(start_key.clone(), 0);
        worklist.push(start_key);

        let mut transition_targets: Vec<SparseStateSet> = (0..num_classes)
            .map(|_| SparseStateSet::new(num_nfa_states))
            .collect();
        let mut used_classes = Vec::with_capacity(num_classes);
        let mut seen_class = vec![false; num_classes];
        let mut closure_set = SparseStateSet::new(num_nfa_states);
        let mut scratch_closure = CompressedStateSet::new();
        let mut sort_scratch = Vec::with_capacity(1024);

        while let Some(current_set) = worklist.pop() {
            let current_dfa_state = subset_map[&current_set];

            for state in current_set.iter() {
                for (class_set, next_state) in &remapped_transitions[state] {
                    for class_id in class_set.iter() {
                        let idx = class_id as usize;
                        if !seen_class[idx] {
                            seen_class[idx] = true;
                            used_classes.push(idx);
                        }
                        transition_targets[idx].insert(*next_state as usize);
                    }
                }
            }

            let mut dfa_transitions_vec = Vec::with_capacity(used_classes.len() * 2);
            for &class_id in &used_classes {
                let target_set = &transition_targets[class_id];

                let mut fast_singleton_state = None;
                if target_set.dirty_words.len() == 1 {
                    let word_idx = target_set.dirty_words[0];
                    let word = target_set.words[word_idx];
                    if word != 0 && (word & (word - 1)) == 0 {
                        let bit = word.trailing_zeros() as usize;
                        let state = word_idx * 64 + bit;
                        if out_degree[state] == 0 && high_degree_closures[state].is_none() {
                            fast_singleton_state = Some(state);
                        }
                    }
                }

                if fast_singleton_state.is_none() {
                    closure_set.clear();
                    let mut needs_bfs = false;

                    for &word_idx in &target_set.dirty_words {
                        let mut word = target_set.words[word_idx];
                        while word != 0 {
                            let bit = word.trailing_zeros() as usize;
                            word &= !(1u64 << bit);
                            let next_state = word_idx * 64 + bit;

                            if let Some(ref closure) = high_degree_closures[next_state] {
                                closure_set.insert_many(closure);
                            } else {
                                closure_set.insert(next_state);
                                if out_degree[next_state] > 0 {
                                    needs_bfs = true;
                                    stack.push(next_state);
                                }
                            }
                        }
                    }

                    if needs_bfs {
                        while let Some(current) = stack.pop() {
                            let start = compact_nfa.epsilon_offsets[current] as usize;
                            let end = compact_nfa.epsilon_offsets[current + 1] as usize;
                            for index in start..end {
                                let next = compact_nfa.epsilon_targets[index] as usize;
                                if closure_set.insert(next) {
                                    if let Some(ref closure) = high_degree_closures[next] {
                                        closure_set.insert_many(closure);
                                    } else if out_degree[next] > 0 {
                                        stack.push(next);
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some(state) = fast_singleton_state {
                    scratch_closure.words.clear();
                    let word_idx = (state >> 6) as u32;
                    let mask = 1u64 << (state & 0x3f);
                    scratch_closure.words.push((word_idx, mask));
                    scratch_closure.hash = (word_idx as u64).wrapping_mul(0x517c_c1b7_2722_0a95)
                        ^ mask.wrapping_mul(0x9e37_79b9_7f4a_7c15);
                } else {
                    CompressedStateSet::reuse_from_sparse(
                        &closure_set,
                        &mut scratch_closure,
                        &mut sort_scratch,
                    );
                }

                let next_dfa_state = if let Some(&existing) = subset_map.get(&scratch_closure) {
                    existing
                } else {
                    let new_state = dfa.add_state();
                    let key = scratch_closure.clone();
                    let (finalizers, future) =
                        compute_subset_metadata(self, &key, group_count, &reachable_groups);
                    dfa.overwrite_state_metadata(new_state, finalizers, future);
                    subset_map.insert(key.clone(), new_state);
                    worklist.push(key);
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

        dfa
    }
}
