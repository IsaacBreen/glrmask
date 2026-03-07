//! NOTE: this file now uses the real sep1-style `CharTransitions` structure,
//! while the broader lexer DFA still remains a trimmed-down version of sep1.
//! Keep the intended shape: explicit `CharTransitions`, `BitSet`-backed
//! finalizers and possible-future-group IDs, `DFAState`-owned
//! `possible_future_group_ids` behind a non-public `DFA` accessor, and
//! `DFA`-owned `group_id_to_u8set`.
// SEP1_MAP: This file follows sep1's `dfa_u8/dfa.rs` storage shape, with
// `CharTransitions` now mirrored from sep1's `datastructures/char_transitions.rs`.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

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
    pub non_greedy_finalizers: BitSet,
    possible_future_group_ids: BitSet,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DFA {
    states: Vec<DFAState>,
    group_id_to_u8set: Vec<U8Set>,
}

impl DFA {
    pub fn new(_num_states: usize) -> Self {
        Self {
            states: vec![DFAState::default(); _num_states],
            group_id_to_u8set: Vec::new(),
        }
    }

    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    pub fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        let groups = self.group_id_to_u8set.len();
        self.states.push(DFAState {
            transitions: CharTransitions::default(),
            finalizers: BitSet::new(groups),
            non_greedy_finalizers: BitSet::new(groups),
            possible_future_group_ids: BitSet::new(groups),
        });
        id
    }

    pub fn ensure_group_capacity(&mut self, num_groups: usize) {
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

                let mut non_greedy = BitSet::new(num_groups);
                for bit in state.non_greedy_finalizers.iter() {
                    non_greedy.set(bit);
                }
                state.non_greedy_finalizers = non_greedy;

                let mut future = BitSet::new(num_groups);
                for bit in state.possible_future_group_ids.iter() {
                    future.set(bit);
                }
                state.possible_future_group_ids = future;
            }
        }
    }

    pub fn add_transition(&mut self, from: u32, byte: u8, to: u32) {
        if let Some(state) = self.states.get_mut(from as usize) {
            state.transitions.insert(byte, to);
        }
    }

    pub fn mark_finalizer(&mut self, state: u32, group_id: GroupId) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.finalizers.set(group_id as usize);
        }
    }

    pub fn mark_non_greedy_finalizer(&mut self, state: u32, group_id: GroupId) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.non_greedy_finalizers.set(group_id as usize);
        }
    }

    pub fn mark_possible_future_group(&mut self, state: u32, group_id: GroupId) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.possible_future_group_ids.set(group_id as usize);
        }
    }

    pub(crate) fn overwrite_state_metadata(
        &mut self,
        state: u32,
        finalizers: BitSet,
        non_greedy_finalizers: BitSet,
        possible_future_group_ids: BitSet,
    ) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.finalizers = finalizers;
            entry.non_greedy_finalizers = non_greedy_finalizers;
            entry.possible_future_group_ids = possible_future_group_ids;
        }
    }

    pub fn set_group_u8set(&mut self, group_id: GroupId, set: U8Set) {
        if let Some(entry) = self.group_id_to_u8set.get_mut(group_id as usize) {
            *entry = set;
        }
    }

    pub fn step(&self, _state: u32, _byte: u8) -> Option<u32> {
        self.states
            .get(_state as usize)
            .and_then(|state| state.transitions.get(_byte).copied())
    }

    pub fn get_u8set(&self, _state: u32) -> U8Set {
        let mut out = U8Set::empty();
        if let Some(state) = self.states.get(_state as usize) {
            for (byte, _) in state.transitions.iter() {
                out.insert(byte);
            }
        }
        out
    }

    pub fn get_transition(&self, state: u32, byte: u8) -> u32 {
        self.step(state, byte).unwrap_or(DEAD)
    }

    pub fn group_id_to_u8set(&self, _group_id: GroupId) -> &U8Set {
        &self.group_id_to_u8set[_group_id as usize]
    }

    pub fn finalizers(&self, _state: u32) -> &BitSet {
        &self.states[_state as usize].finalizers
    }

    pub fn non_greedy_finalizers(&self, _state: u32) -> &BitSet {
        &self.states[_state as usize].non_greedy_finalizers
    }

    pub(crate) fn possible_future_group_ids(&self, _state: u32) -> &BitSet {
        &self.states[_state as usize].possible_future_group_ids
    }

    pub fn states(&self) -> &[DFAState] {
        &self.states
    }
}
