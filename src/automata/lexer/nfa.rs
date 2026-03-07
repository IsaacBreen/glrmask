//! NOTE: this file is intentionally gutted.
//! Reintroduce only the minimal lexer NFA needed as input to the future
//! sep1-style DFA pipeline.
// SEP1_MAP: This placeholder corresponds directly to the NFA half of sep1's `dfa_u8/dfa.rs`.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeSet;

use crate::ds::u8set::U8Set;

use super::dfa::{DFA, GroupId};

#[derive(Debug, Clone)]
pub struct NFAState {
    pub transitions: Vec<(U8Set, u32)>,
    pub epsilon_transitions: Vec<u32>,
    pub finalizers: BTreeSet<GroupId>,
    pub non_greedy_finalizers: BTreeSet<GroupId>,
}

#[derive(Debug, Clone)]
pub struct NFA {
    pub(crate) states: Vec<NFAState>,
}

impl NFA {
    pub fn new(num_states: usize) -> Self {
        let mut states = Vec::with_capacity(num_states.max(1));
        states.push(NFAState {
            transitions: Vec::new(),
            epsilon_transitions: Vec::new(),
            finalizers: BTreeSet::new(),
            non_greedy_finalizers: BTreeSet::new(),
        });
        Self { states }
    }

    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    pub fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        self.states.push(NFAState {
            transitions: Vec::new(),
            epsilon_transitions: Vec::new(),
            finalizers: BTreeSet::new(),
            non_greedy_finalizers: BTreeSet::new(),
        });
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

    pub fn add_non_greedy_finalizer(&mut self, state: u32, group_id: GroupId) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.non_greedy_finalizers.insert(group_id);
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
}
