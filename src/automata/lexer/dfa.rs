//! NOTE: this file now uses the real sep1-style `CharTransitions` structure,
//! while the broader lexer DFA still remains a trimmed-down version of sep1.
//! Keep the intended shape: explicit `CharTransitions`, `BitSet`-backed
//! finalizers and possible-future-group IDs, `DFAState`-owned
//! `possible_future_group_ids` behind a non-public `DFA` accessor, and
//! `DFA`-owned `group_id_to_u8set`.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::ds::char_transitions::CharTransitions;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;

pub type GroupId = u32;
pub const DEAD: u32 = u32::MAX;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DFAState {
    pub transitions: CharTransitions<u32>,
    pub finalizers: BitSet,
    possible_future_group_ids: BitSet,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DFA {
    states: Vec<DFAState>,
    group_id_to_u8set: Vec<U8Set>,
}

impl DFA {
    pub fn new(num_states: usize) -> Self {
        Self {
            states: vec![DFAState::default(); num_states],
            group_id_to_u8set: Vec::new(),
        }
    }

    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    pub(super) fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        let groups = self.group_id_to_u8set.len();
        self.states.push(DFAState {
            transitions: CharTransitions::default(),
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
            if state.finalizers.len() < num_groups {
                let mut finalizers = BitSet::new(num_groups);
                for bit in state.finalizers.iter() {
                    finalizers.set(bit);
                }
                state.finalizers = finalizers;

                let mut future = BitSet::new(num_groups);
                for bit in state.possible_future_group_ids.iter() {
                    future.set(bit);
                }
                state.possible_future_group_ids = future;
            }
        }
    }

    pub(super) fn add_transition(&mut self, from: u32, byte: u8, to: u32) {
        if let Some(state) = self.states.get_mut(from as usize) {
            state.transitions.insert(byte, to);
        }
    }

    pub(crate) fn set_transitions_from_sorted_entries(
        &mut self,
        state: u32,
        entries: Vec<(u8, u32)>,
    ) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.transitions = CharTransitions::from_sorted_entries(entries);
        }
    }

    pub(super) fn clear_finalizers_for_state(&mut self, state: u32) -> BitSet {
        if let Some(entry) = self.states.get_mut(state as usize) {
            std::mem::take(&mut entry.finalizers)
        } else {
            BitSet::empty(0)
        }
    }

    pub(crate) fn overwrite_state_metadata(
        &mut self,
        state: u32,
        finalizers: BitSet,
        possible_future_group_ids: BitSet,
    ) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.finalizers = finalizers;
            entry.possible_future_group_ids = possible_future_group_ids;
        }
    }

    pub(super) fn set_group_u8set(&mut self, group_id: GroupId, set: U8Set) {
        if let Some(entry) = self.group_id_to_u8set.get_mut(group_id as usize) {
            *entry = set;
        }
    }

    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        self.states
            .get(state as usize)
            .and_then(|state| state.transitions.get(byte).copied())
    }

    pub fn get_u8set(&self, state: u32) -> U8Set {
        let mut out = U8Set::empty();
        if let Some(state) = self.states.get(state as usize) {
            for (byte, _) in state.transitions.iter() {
                out.insert(byte);
            }
        }
        out
    }

    pub fn get_transition(&self, state: u32, byte: u8) -> u32 {
        self.step(state, byte).unwrap_or(DEAD)
    }

    pub fn group_id_to_u8set(&self, group_id: GroupId) -> &U8Set {
        &self.group_id_to_u8set[group_id as usize]
    }

    pub fn finalizers(&self, state: u32) -> &BitSet {
        &self.states[state as usize].finalizers
    }

    pub(crate) fn possible_future_group_ids(&self, state: u32) -> &BitSet {
        &self.states[state as usize].possible_future_group_ids
    }

    pub fn states(&self) -> &[DFAState] {
        &self.states
    }

    pub(super) fn states_mut(&mut self) -> &mut Vec<DFAState> {
        &mut self.states
    }

    pub(super) fn num_groups(&self) -> usize {
        self.group_id_to_u8set.len()
    }

    pub(super) fn set_possible_future_group_ids(&mut self, state: u32, ids: BitSet) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.possible_future_group_ids = ids;
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
    pub(super) fn redirect_transitions(&mut self, old_target: u32, new_target: u32) {
        for state in &mut self.states {
            for (_, target) in state.transitions.iter_mut() {
                if *target == old_target {
                    *target = new_target;
                }
            }
        }
    }

    pub(crate) fn apply_group_exclusions(
        &mut self,
        excludes: &BTreeMap<GroupId, BTreeSet<GroupId>>,
    ) -> bool {
        let mut changed = false;
        for state in &mut self.states {
            if state.finalizers.count_ones() < 2 {
                continue;
            }

            let mut to_clear = Vec::new();
            for (&group_id, blocked_by) in excludes {
                let group_index = group_id as usize;
                if !state.finalizers.contains(group_index) {
                    continue;
                }
                if blocked_by
                    .iter()
                    .any(|blocked_by_id| state.finalizers.contains(*blocked_by_id as usize))
                {
                    to_clear.push(group_index);
                }
            }

            for group_index in to_clear {
                if state.finalizers.contains(group_index) {
                    state.finalizers.clear(group_index);
                    changed = true;
                }
            }
        }
        changed
    }

    pub(crate) fn project_groups(&self, num_groups: usize) -> DFA {
        let mut projected = DFA::new(self.num_states());
        projected.ensure_group_capacity(num_groups);

        for (state_index, state) in self.states.iter().enumerate() {
            let transitions = state
                .transitions
                .iter()
                .map(|(byte, &target)| (byte, target))
                .collect();
            projected.set_transitions_from_sorted_entries(state_index as u32, transitions);

            let mut finalizers = BitSet::new(num_groups);
            for group_id in state.finalizers.iter().filter(|group_id| *group_id < num_groups) {
                finalizers.set(group_id);
            }

            let mut future = BitSet::new(num_groups);
            for group_id in state
                .possible_future_group_ids
                .iter()
                .filter(|group_id| *group_id < num_groups)
            {
                future.set(group_id);
            }

            projected.overwrite_state_metadata(state_index as u32, finalizers, future);
        }

        for group_id in 0..num_groups {
            projected.set_group_u8set(group_id as u32, self.group_id_to_u8set[group_id]);
        }

        projected
    }
}
