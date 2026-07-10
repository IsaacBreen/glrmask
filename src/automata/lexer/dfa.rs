//! Byte-oriented lexer DFA used by the tokenizer and lexer compiler.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::ds::char_transitions::CharTransitions;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;

pub(super) type GroupId = u32;
pub(super) const DEAD: u32 = u32::MAX;

fn resized_bitset(bits: &BitSet, num_groups: usize) -> BitSet {
    let mut resized = BitSet::new(num_groups);
    for bit in bits.iter() {
        resized.set(bit);
    }
    resized
}

fn project_bitset(bits: &BitSet, num_groups: usize) -> BitSet {
    let mut projected = BitSet::new(num_groups);
    for group_id in bits.iter().filter(|group_id| *group_id < num_groups) {
        projected.set(group_id);
    }
    projected
}

fn excluded_group_indices(
    finalizers: &BitSet,
    excludes: &BTreeMap<GroupId, BTreeSet<GroupId>>,
) -> Vec<usize> {
    let mut to_clear = Vec::new();
    for (&group_id, blocked_by) in excludes {
        let group_index = group_id as usize;
        if !finalizers.contains(group_index) {
            continue;
        }
        if blocked_by
            .iter()
            .any(|blocked_by_id| finalizers.contains(*blocked_by_id as usize))
        {
            to_clear.push(group_index);
        }
    }
    to_clear
}

fn intersection_missing_group_indices(
    finalizers: &BitSet,
    intersections: &BTreeMap<GroupId, BTreeSet<GroupId>>,
) -> Vec<usize> {
    let mut to_clear = Vec::new();
    for (&group_id, required) in intersections {
        let group_index = group_id as usize;
        if !finalizers.contains(group_index) {
            continue;
        }
        if required
            .iter()
            .any(|required_id| !finalizers.contains(*required_id as usize))
        {
            to_clear.push(group_index);
        }
    }
    to_clear
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct DFAState {
    pub(super) transitions: CharTransitions<u32>,
    /// Epsilon transitions are the only source of lexer nondeterminism. Byte
    /// transitions remain deterministic within an individual physical state.
    #[serde(default)]
    pub(super) epsilon_transitions: Vec<u32>,
    pub(super) finalizers: BitSet,
    possible_future_group_ids: BitSet,
}

#[derive(Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DFA {
    states: Vec<DFAState>,
    group_id_to_u8set: Vec<U8Set>,
}

impl std::fmt::Debug for DFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DFA { .. }")
    }
}

impl DFA {
    pub(super) fn new(num_states: usize) -> Self {
        Self {
            states: vec![DFAState::default(); num_states],
            group_id_to_u8set: Vec::new(),
        }
    }

    pub(super) fn num_states(&self) -> usize {
        self.states.len()
    }

    pub(super) fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        let groups = self.group_id_to_u8set.len();
        self.states.push(DFAState {
            transitions: CharTransitions::default(),
            epsilon_transitions: Vec::new(),
            finalizers: BitSet::new(groups),
            possible_future_group_ids: BitSet::new(groups),
        });
        id
    }

    pub(super) fn ensure_group_capacity(&mut self, num_groups: usize) {
        if self.group_id_to_u8set.len() < num_groups {
            self.group_id_to_u8set.resize(num_groups, U8Set::empty());
        }
        for state in &mut self.states {
            Self::resize_state_group_bits(state, num_groups);
        }
    }

    pub(super) fn add_transition(&mut self, from: u32, byte: u8, to: u32) {
        if let Some(state) = self.states.get_mut(from as usize) {
            state.transitions.insert(byte, to);
        }
    }

    pub(super) fn add_epsilon_transition(&mut self, from: u32, to: u32) {
        if let Some(state) = self.states.get_mut(from as usize) {
            if !state.epsilon_transitions.contains(&to) {
                state.epsilon_transitions.push(to);
            }
        }
    }

    pub(super) fn has_epsilon_transitions(&self) -> bool {
        self.states
            .iter()
            .any(|state| !state.epsilon_transitions.is_empty())
    }

    pub(super) fn epsilon_closure(&self, roots: &[u32]) -> SmallVec<[u32; 1]> {
        if roots.len() == 1 {
            let root = roots[0];
            if self
                .states
                .get(root as usize)
                .is_some_and(|state| state.epsilon_transitions.is_empty())
            {
                return SmallVec::from_buf([root]);
            }
        }
        let mut closure = SmallVec::<[u32; 1]>::new();
        let mut seen = vec![false; self.states.len()];
        let mut stack = Vec::with_capacity(roots.len());
        for &root in roots {
            if (root as usize) < self.states.len() && !seen[root as usize] {
                seen[root as usize] = true;
                stack.push(root);
            }
        }
        while let Some(state) = stack.pop() {
            closure.push(state);
            for &target in &self.states[state as usize].epsilon_transitions {
                if !seen[target as usize] {
                    seen[target as usize] = true;
                    stack.push(target);
                }
            }
        }
        closure.sort_unstable();
        closure
    }

    pub(super) fn step_all(&self, states: &[u32], byte: u8) -> SmallVec<[u32; 1]> {
        if states.len() == 1 {
            let state = states[0];
            if self.states[state as usize].epsilon_transitions.is_empty() {
                let Some(target) = self.step(state, byte) else {
                    return SmallVec::new();
                };
                if self.states[target as usize].epsilon_transitions.is_empty() {
                    return SmallVec::from_buf([target]);
                }
            }
        }
        let closure = self.epsilon_closure(states);
        let mut targets = SmallVec::<[u32; 1]>::new();
        for state in closure {
            if let Some(target) = self.step(state, byte) {
                targets.push(target);
            }
        }
        if targets.is_empty() {
            return targets;
        }
        targets.sort_unstable();
        targets.dedup();
        self.epsilon_closure(&targets)
    }

    pub(super) fn set_transitions_from_sorted_entries(
        &mut self,
        state: u32,
        entries: Vec<(u8, u32)>,
    ) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.transitions = CharTransitions::from_sorted_entries(entries);
        }
    }

    pub(super) fn clear_finalizers_for_state(&mut self, state: u32) -> BitSet {
        let num_groups = self.group_id_to_u8set.len();
        if let Some(entry) = self.state_mut(state) {
            std::mem::replace(&mut entry.finalizers, BitSet::new(num_groups))
        } else {
            BitSet::empty(0)
        }
    }

    pub(super) fn overwrite_state_metadata(
        &mut self,
        state: u32,
        finalizers: BitSet,
        possible_future_group_ids: BitSet,
    ) {
        if let Some(entry) = self.state_mut(state) {
            entry.finalizers = finalizers;
            entry.possible_future_group_ids = possible_future_group_ids;
        }
    }

    pub(super) fn set_group_u8set(&mut self, group_id: GroupId, set: U8Set) {
        if let Some(entry) = self.group_id_to_u8set.get_mut(group_id as usize) {
            *entry = set;
        }
    }

    pub(super) fn step(&self, state: u32, byte: u8) -> Option<u32> {
        self.states
            .get(state as usize)
            .and_then(|state| state.transitions.get(byte).copied())
    }

    pub(super) fn get_u8set(&self, state: u32) -> U8Set {
        let mut out = U8Set::empty();
        if let Some(state) = self.states.get(state as usize) {
            for (byte, _) in state.transitions.iter() {
                out.insert(byte);
            }
        }
        out
    }

    pub(super) fn get_transition(&self, state: u32, byte: u8) -> u32 {
        self.step(state, byte).unwrap_or(DEAD)
    }

    pub(super) fn group_id_to_u8set(&self, group_id: GroupId) -> &U8Set {
        &self.group_id_to_u8set[group_id as usize]
    }

    pub(super) fn finalizers(&self, state: u32) -> &BitSet {
        &self.states[state as usize].finalizers
    }

    pub(super) fn possible_future_group_ids(&self, state: u32) -> &BitSet {
        &self.states[state as usize].possible_future_group_ids
    }

    pub(super) fn states(&self) -> &[DFAState] {
        &self.states
    }

    pub(super) fn states_mut(&mut self) -> &mut Vec<DFAState> {
        &mut self.states
    }

    pub(super) fn num_groups(&self) -> usize {
        self.group_id_to_u8set.len()
    }

    pub(super) fn set_possible_future_group_ids(&mut self, state: u32, ids: BitSet) {
        if let Some(entry) = self.state_mut(state) {
            entry.possible_future_group_ids = ids;
        }
    }
    /// Mask all states' possible_future_group_ids with the given bitset.
    pub(super) fn mask_possible_futures(&mut self, mask: &BitSet) {
        for state in &mut self.states {
            state.possible_future_group_ids.intersect_with(mask);
        }
    }
    /// Create a clone of an existing state (transitions, finalizers,
    /// possible_future_group_ids) and return the new state's id.
    pub(super) fn clone_state(&mut self, source: u32) -> u32 {
        let cloned = self.states[source as usize].clone();
        let id = self.states.len() as u32;
        self.states.push(cloned);
        id
    }

    /// Rewrite every transition that targets `old_target` so it targets
    /// `new_target` instead.
    /// Redirect every incoming edge to `old_target`, returning whether any
    /// edge changed. The caller may use this to speculatively clone a state
    /// and discard the clone when no incoming edge exists.
    pub(super) fn redirect_transitions(&mut self, old_target: u32, new_target: u32) -> bool {
        let mut changed = false;
        for state in &mut self.states {
            for (_, target) in state.transitions.iter_mut() {
                if *target == old_target {
                    *target = new_target;
                    changed = true;
                }
            }
            for target in &mut state.epsilon_transitions {
                if *target == old_target {
                    *target = new_target;
                    changed = true;
                }
            }
        }
        changed
    }

    /// Remove the final state when it is the expected freshly-created ID.
    pub(super) fn discard_last_state(&mut self, expected: u32) {
        debug_assert_eq!(self.states.len(), expected as usize + 1);
        self.states.pop();
    }

    pub(super) fn apply_group_exclusions(
        &mut self,
        excludes: &BTreeMap<GroupId, BTreeSet<GroupId>>,
    ) -> bool {
        let mut changed = false;
        for state in &mut self.states {
            if state.finalizers.count_ones() < 2 {
                continue;
            }

            for group_index in excluded_group_indices(&state.finalizers, excludes) {
                if state.finalizers.contains(group_index) {
                    state.finalizers.clear(group_index);
                    changed = true;
                }
            }
        }
        changed
    }

    pub(super) fn apply_group_intersections(
        &mut self,
        intersections: &BTreeMap<GroupId, BTreeSet<GroupId>>,
    ) -> bool {
        let mut changed = false;
        for state in &mut self.states {
            for group_index in intersection_missing_group_indices(&state.finalizers, intersections) {
                if state.finalizers.contains(group_index) {
                    state.finalizers.clear(group_index);
                    changed = true;
                }
            }
        }
        changed
    }

    pub(super) fn project_groups(&self, num_groups: usize) -> DFA {
        let mut projected = DFA::new(self.num_states());
        projected.ensure_group_capacity(num_groups);

        for (state_index, state) in self.states.iter().enumerate() {
            let transitions = state
                .transitions
                .iter()
                .map(|(byte, &target)| (byte, target))
                .collect();
            projected.set_transitions_from_sorted_entries(state_index as u32, transitions);
            projected.states_mut()[state_index].epsilon_transitions =
                state.epsilon_transitions.clone();

            let finalizers = project_bitset(&state.finalizers, num_groups);
            let future = project_bitset(&state.possible_future_group_ids, num_groups);

            projected.overwrite_state_metadata(state_index as u32, finalizers, future);
        }

        for group_id in 0..num_groups {
            projected.set_group_u8set(group_id as u32, self.group_id_to_u8set[group_id]);
        }

        projected
    }

    fn state_mut(&mut self, state: u32) -> Option<&mut DFAState> {
        self.states.get_mut(state as usize)
    }

    fn resize_state_group_bits(state: &mut DFAState, num_groups: usize) {
        if state.finalizers.len() < num_groups {
            state.finalizers = resized_bitset(&state.finalizers, num_groups);
            state.possible_future_group_ids =
                resized_bitset(&state.possible_future_group_ids, num_groups);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DFA;
    use crate::ds::bitset::BitSet;

    #[test]
    fn epsilon_closure_handles_fanout_chains_and_cycles() {
        let mut automaton = DFA::new(5);
        automaton.ensure_group_capacity(1);
        automaton.add_epsilon_transition(0, 1);
        automaton.add_epsilon_transition(0, 2);
        automaton.add_epsilon_transition(1, 3);
        automaton.add_epsilon_transition(2, 3);
        automaton.add_epsilon_transition(3, 1);
        automaton.add_transition(3, b'x', 4);

        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        automaton.overwrite_state_metadata(4, finalizers, BitSet::new(1));
        automaton.recompute_possible_futures();

        assert_eq!(automaton.epsilon_closure(&[0]).as_slice(), &[0, 1, 2, 3]);
        assert_eq!(automaton.step_all(&[0], b'x').as_slice(), &[4]);
        assert!(automaton.possible_future_group_ids(0).contains(0));
    }
}
