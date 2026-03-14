//! NOTE: lexer NFA → DFA determinization is intentionally deferred until the
//! sep1-style DFA rewrite.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::ds::bitset::BitSet;

use super::dfa::DFA;
use super::nfa::NFA;

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

        let mut subset_map = HashMap::<Vec<u32>, u32>::new();
        let mut worklist = VecDeque::new();

        let mut start = BTreeSet::new();
        start.insert(0);
        let start_closure = self.epsilon_closure(&start);
        let start_key: Vec<u32> = start_closure.iter().copied().collect();
        subset_map.insert(start_key.clone(), 0);
        worklist.push_back(start_key);

        while let Some(subset_key) = worklist.pop_front() {
            let dfa_state = subset_map[&subset_key];
            let subset: BTreeSet<u32> = subset_key.iter().copied().collect();

            let mut finalizers = BitSet::new(group_count);
            let mut future = BitSet::new(group_count);
            let mut transitions = BTreeMap::<u8, BTreeSet<u32>>::new();

            for &nfa_state_id in &subset {
                let nfa_state = &self.states[nfa_state_id as usize];
                for &group in &nfa_state.finalizers {
                    finalizers.set(group as usize);
                }
                future.union_with(&reachable_groups[nfa_state_id as usize]);
                for (set, next) in &nfa_state.transitions {
                    for byte in set.iter() {
                        transitions.entry(byte).or_default().insert(*next);
                    }
                }
            }

            dfa.overwrite_state_metadata(dfa_state, finalizers, future);

            for (byte, targets) in transitions {
                let closure = self.epsilon_closure(&targets);
                let key: Vec<u32> = closure.iter().copied().collect();
                let next_dfa_state = if let Some(existing) = subset_map.get(&key).copied() {
                    existing
                } else {
                    let new_state = dfa.add_state();
                    subset_map.insert(key.clone(), new_state);
                    worklist.push_back(key);
                    new_state
                };
                dfa.add_transition(dfa_state, byte, next_dfa_state);
            }
        }

        dfa
    }
}
