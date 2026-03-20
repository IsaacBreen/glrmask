//! sep1-style lexer NFA → DFA determinization, adapted to glrmask's leaner DFA types.

use std::collections::VecDeque;

use rustc_hash::FxHashMap;

use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;

use super::dfa::DFA;
use super::nfa::{CompactNFA, NFA};

fn precompute_epsilon_closures_dag(compact_nfa: &CompactNFA, num_states: usize) -> Vec<Vec<u32>> {
    let mut visited = vec![false; num_states];
    let mut post_order = Vec::with_capacity(num_states);

    fn dfs_postorder(
        state: usize,
        compact_nfa: &CompactNFA,
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
            dfs_postorder(target as usize, compact_nfa, visited, post_order);
        }
        post_order.push(state);
    }

    for state in 0..num_states {
        dfs_postorder(state, compact_nfa, &mut visited, &mut post_order);
    }

    let mut closures = vec![Vec::new(); num_states];
    let mut seen_generation = vec![0u32; num_states];
    let mut generation = 0u32;

    for &state in &post_order {
        generation = generation.wrapping_add(1);
        if generation == 0 {
            seen_generation.fill(0);
            generation = 1;
        }

        let current_generation = generation;
        let mut closure = Vec::new();
        seen_generation[state] = current_generation;
        closure.push(state as u32);

        let start = compact_nfa.epsilon_offsets[state] as usize;
        let end = compact_nfa.epsilon_offsets[state + 1] as usize;
        for &target in &compact_nfa.epsilon_targets[start..end] {
            for &member in &closures[target as usize] {
                let slot = &mut seen_generation[member as usize];
                if *slot == current_generation {
                    continue;
                }
                *slot = current_generation;
                closure.push(member);
            }
        }

        closure.sort_unstable();
        closures[state] = closure;
    }

    closures
}

fn closure_key_from_targets(
    targets: &[u32],
    single_closures: &[Vec<u32>],
    seen_generation: &mut [u32],
    generation: &mut u32,
) -> Vec<u32> {
    *generation = generation.wrapping_add(1);
    if *generation == 0 {
        seen_generation.fill(0);
        *generation = 1;
    }

    let current_generation = *generation;
    let mut closure = Vec::new();
    for &target in targets {
        for &state_id in &single_closures[target as usize] {
            let slot = &mut seen_generation[state_id as usize];
            if *slot == current_generation {
                continue;
            }
            *slot = current_generation;
            closure.push(state_id);
        }
    }
    closure.sort_unstable();
    closure
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
        let single_closures = precompute_epsilon_closures_dag(&compact_nfa, self.states.len());
        let mut seen_generation = vec![0u32; self.states.len()];
        let mut generation = 0u32;
        let mut transition_generation = 0u32;
        let mut transition_marks = vec![0u32; num_classes * self.states.len().max(1)];

        let mut subset_map = FxHashMap::<Vec<u32>, u32>::default();
        let mut worklist = VecDeque::new();

        let start_key = single_closures[self.start_state as usize].clone();
        subset_map.insert(start_key.clone(), 0);
        worklist.push_back(start_key);

        while let Some(subset_key) = worklist.pop_front() {
            let dfa_state = subset_map[&subset_key];

            let mut finalizers = BitSet::new(group_count);
            let mut future = BitSet::new(group_count);
            let mut transitions = vec![Vec::<u32>::new(); num_classes];

            transition_generation = transition_generation.wrapping_add(1);
            if transition_generation == 0 {
                transition_marks.fill(0);
                transition_generation = 1;
            }
            let current_transition_generation = transition_generation;

            for &nfa_state_id in &subset_key {
                let nfa_state = &self.states[nfa_state_id as usize];
                for &group in &nfa_state.finalizers {
                    finalizers.set(group as usize);
                }
                future.union_with(&reachable_groups[nfa_state_id as usize]);
                for (set, next) in &remapped_transitions[nfa_state_id as usize] {
                    for class_id in set.iter() {
                        let mark_index = class_id as usize * self.states.len() + *next as usize;
                        if transition_marks[mark_index] == current_transition_generation {
                            continue;
                        }
                        transition_marks[mark_index] = current_transition_generation;
                        transitions[class_id as usize].push(*next);
                    }
                }
            }

            dfa.overwrite_state_metadata(dfa_state, finalizers, future);

            for (class_id, targets) in transitions.into_iter().enumerate() {
                if targets.is_empty() {
                    continue;
                }
                let key = closure_key_from_targets(
                    &targets,
                    &single_closures,
                    &mut seen_generation,
                    &mut generation,
                );
                let next_dfa_state = if let Some(existing) = subset_map.get(&key).copied() {
                    existing
                } else {
                    let new_state = dfa.add_state();
                    subset_map.insert(key.clone(), new_state);
                    worklist.push_back(key);
                    new_state
                };
                for &byte in &class_members[class_id] {
                    dfa.add_transition(dfa_state, byte, next_dfa_state);
                }
            }
        }

        dfa
    }
}
