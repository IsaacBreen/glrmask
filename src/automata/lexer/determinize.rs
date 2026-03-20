//! NOTE: lexer NFA → DFA determinization is intentionally deferred until the
//! sep1-style DFA rewrite.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rustc_hash::FxHashMap;

use crate::ds::bitset::BitSet;

use super::dfa::DFA;
use super::nfa::NFA;

fn single_state_epsilon_closures(nfa: &NFA) -> Vec<Vec<u32>> {
    let mut closures = Vec::with_capacity(nfa.states.len());
    for state_id in 0..nfa.states.len() {
        let mut closure = BTreeSet::new();
        let mut stack = vec![state_id as u32];
        closure.insert(state_id as u32);
        while let Some(current) = stack.pop() {
            if let Some(state) = nfa.states.get(current as usize) {
                for &next in &state.epsilon_transitions {
                    if closure.insert(next) {
                        stack.push(next);
                    }
                }
            }
        }
        closures.push(closure.into_iter().collect());
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

        let single_closures = single_state_epsilon_closures(self);
        let mut seen_generation = vec![0u32; self.states.len()];
        let mut generation = 0u32;
        let mut transition_generation = 0u32;
        let mut transition_marks = vec![0u32; 256 * self.states.len().max(1)];

        let mut subset_map = FxHashMap::<Vec<u32>, u32>::default();
        let mut worklist = VecDeque::new();

        let start_key = single_closures[0].clone();
        subset_map.insert(start_key.clone(), 0);
        worklist.push_back(start_key);

        while let Some(subset_key) = worklist.pop_front() {
            let dfa_state = subset_map[&subset_key];

            let mut finalizers = BitSet::new(group_count);
            let mut future = BitSet::new(group_count);
            let mut transitions = vec![Vec::<u32>::new(); 256];

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
                for (set, next) in &nfa_state.transitions {
                    for byte in set.iter() {
                        let mark_index = byte as usize * self.states.len() + *next as usize;
                        if transition_marks[mark_index] == current_transition_generation {
                            continue;
                        }
                        transition_marks[mark_index] = current_transition_generation;
                        transitions[byte as usize].push(*next);
                    }
                }
            }

            dfa.overwrite_state_metadata(dfa_state, finalizers, future);

            for (byte, targets) in transitions.into_iter().enumerate() {
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
                dfa.add_transition(dfa_state, byte as u8, next_dfa_state);
            }
        }

        dfa
    }
}
