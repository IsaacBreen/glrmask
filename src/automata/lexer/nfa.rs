//! Byte-oriented lexer NFA used as the determinization input.

use std::collections::BTreeSet;

use rustc_hash::FxHashSet;

use crate::sets::byte_set::U8Set;

use super::dfa::GroupId;

#[derive(Debug, Clone)]
pub struct NFAState {
    pub transitions: Vec<(U8Set, u32)>,
    pub epsilon_transitions: Vec<u32>,
    pub finalizers: BTreeSet<GroupId>,
}

impl NFAState {
    fn new() -> Self {
        Self {
            transitions: Vec::new(),
            epsilon_transitions: Vec::new(),
            finalizers: BTreeSet::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CompactNFA {
    pub epsilon_offsets: Vec<u32>,
    pub epsilon_targets: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct NFA {
    pub(crate) states: Vec<NFAState>,
    pub(crate) start_state: u32,
}

fn build_states(count: usize) -> Vec<NFAState> {
    let mut states = Vec::with_capacity(count);
    for _ in 0..count {
        states.push(NFAState::new());
    }
    states
}

impl NFA {
    pub fn new(num_states: usize) -> Self {
        let count = num_states.max(1);
        Self {
            states: build_states(count),
            start_state: 0,
        }
    }

    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    pub fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        self.states.push(NFAState::new());
        id
    }

    pub fn add_transition(&mut self, from: u32, byte: u8, to: u32) {
        self.add_u8set_transition(from, U8Set::single(byte), to)
    }

    pub fn add_u8set_transition(&mut self, from: u32, set: U8Set, to: u32) {
        if let Some(state) = self.states.get_mut(from as usize) {
            state.transitions.push((set, to));
        }
    }

    pub fn add_epsilon(&mut self, from: u32, to: u32) {
        if let Some(state) = self.states.get_mut(from as usize) {
            state.epsilon_transitions.push(to);
        }
    }

    pub fn add_finalizer(&mut self, state: u32, group_id: GroupId) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.finalizers.insert(group_id);
        }
    }

    pub fn epsilon_closure(&self, states: &BTreeSet<u32>) -> BTreeSet<u32> {
        let mut closure = states.clone();
        let mut stack: Vec<u32> = states.iter().copied().collect();
        while let Some(state_id) = stack.pop() {
            if let Some(state) = self.states.get(state_id as usize) {
                for &next in &state.epsilon_transitions {
                    if closure.insert(next) {
                        stack.push(next);
                    }
                }
            }
        }
        closure
    }

    pub fn condense_epsilon_sccs(&mut self) {
        let num_states = self.states.len();
        if num_states <= 1 {
            return;
        }

        let mut disc = vec![-1i32; num_states];
        let mut low = vec![-1i32; num_states];
        let mut on_stack = vec![false; num_states];
        let mut stack: Vec<usize> = Vec::new();
        let mut time = 0i32;
        let mut scc_map = vec![0usize; num_states];
        let mut scc_count = 0usize;
        let mut work_stack: Vec<(usize, usize)> = Vec::new();

        for root in 0..num_states {
            if disc[root] != -1 {
                continue;
            }

            work_stack.push((root, 0));
            while let Some((state, edge_index)) = work_stack.pop() {
                if edge_index == 0 {
                    disc[state] = time;
                    low[state] = time;
                    time += 1;
                    stack.push(state);
                    on_stack[state] = true;
                }

                let neighbors = &self.states[state].epsilon_transitions;
                if edge_index < neighbors.len() {
                    let next = neighbors[edge_index] as usize;
                    work_stack.push((state, edge_index + 1));
                    if disc[next] == -1 {
                        work_stack.push((next, 0));
                    } else if on_stack[next] {
                        low[state] = low[state].min(disc[next]);
                    }
                    continue;
                }

                if low[state] == disc[state] {
                    loop {
                        let member = stack.pop().expect("epsilon SCC stack underflow");
                        on_stack[member] = false;
                        scc_map[member] = scc_count;
                        if member == state {
                            break;
                        }
                    }
                    scc_count += 1;
                }

                if let Some((parent, _)) = work_stack.last() {
                    low[*parent] = low[*parent].min(low[state]);
                }
            }
        }

        if scc_count == num_states {
            return;
        }

        let mut new_states = self.build_condensed_states(&scc_map, scc_count);
        Self::dedup_epsilon_transitions(&mut new_states);

        self.states = new_states;
        self.start_state = scc_map[self.start_state as usize] as u32;
    }

    pub(crate) fn build_compact_nfa(&self) -> CompactNFA {
        let mut epsilon_offsets = Vec::with_capacity(self.states.len() + 1);
        let mut epsilon_targets = Vec::new();

        for state in &self.states {
            epsilon_offsets.push(epsilon_targets.len() as u32);
            epsilon_targets.extend(state.epsilon_transitions.iter().copied());
        }
        epsilon_offsets.push(epsilon_targets.len() as u32);

        CompactNFA {
            epsilon_offsets,
            epsilon_targets,
        }
    }

    pub(crate) fn compute_equivalence_classes(&self) -> (Vec<u8>, usize, Vec<Vec<u8>>) {
        let mut partitions = vec![U8Set::all()];
        let mut seen_sets = FxHashSet::default();

        for state in &self.states {
            for (set, _) in &state.transitions {
                if seen_sets.insert(*set) {
                    partitions = Self::refine_partitions(partitions, *set);
                }
            }
        }

        let (class_map, class_members) = Self::build_equivalence_class_outputs(&partitions);

        (class_map, partitions.len(), class_members)
    }

    fn build_condensed_states(&self, scc_map: &[usize], scc_count: usize) -> Vec<NFAState> {
        let mut new_states = build_states(scc_count);
        for (old_id, state) in self.states.iter().enumerate() {
            Self::merge_condensed_state(old_id, state, scc_map, &mut new_states);
        }
        new_states
    }

    fn merge_condensed_state(
        old_id: usize,
        state: &NFAState,
        scc_map: &[usize],
        new_states: &mut [NFAState],
    ) {
        let new_id = scc_map[old_id];
        let new_state = &mut new_states[new_id];
        new_state.finalizers.extend(state.finalizers.iter().copied());

        for (set, target) in &state.transitions {
            let new_target = scc_map[*target as usize] as u32;
            new_state.transitions.push((*set, new_target));
        }

        for &target in &state.epsilon_transitions {
            let new_target = scc_map[target as usize] as u32;
            if new_target != new_id as u32 {
                new_state.epsilon_transitions.push(new_target);
            }
        }
    }

    fn dedup_epsilon_transitions(states: &mut [NFAState]) {
        for state in states {
            state.epsilon_transitions.sort_unstable();
            state.epsilon_transitions.dedup();
        }
    }

    fn refine_partitions(partitions: Vec<U8Set>, split: U8Set) -> Vec<U8Set> {
        let mut next_partitions = Vec::with_capacity(partitions.len() * 2);
        for partition in partitions {
            let intersection = partition.intersection(&split);
            let difference = partition.difference(&split);
            if !intersection.is_empty() {
                next_partitions.push(intersection);
            }
            if !difference.is_empty() {
                next_partitions.push(difference);
            }
        }
        next_partitions
    }

    fn build_equivalence_class_outputs(partitions: &[U8Set]) -> (Vec<u8>, Vec<Vec<u8>>) {
        let mut class_map = vec![0u8; 256];
        let mut class_members = vec![Vec::new(); partitions.len()];
        for (index, partition) in partitions.iter().enumerate() {
            for byte in partition.iter() {
                class_map[byte as usize] = index as u8;
                class_members[index].push(byte);
            }
        }
        (class_map, class_members)
    }
}
