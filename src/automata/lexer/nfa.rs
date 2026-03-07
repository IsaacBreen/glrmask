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
    states: Vec<NFAState>,
}

impl NFA {
    pub fn new(_num_states: usize) -> Self {
        todo!("lexer NFA storage is intentionally deferred during automata cleanup")
    }

    pub fn num_states(&self) -> usize {
        todo!("lexer NFA storage is intentionally deferred during automata cleanup")
    }

    pub fn add_state(&mut self) -> u32 {
        todo!("lexer NFA storage is intentionally deferred during automata cleanup")
    }

    pub fn add_transition(&mut self, _from: u32, _byte: u8, _to: u32) {
        todo!("lexer NFA storage is intentionally deferred during automata cleanup")
    }

    pub fn add_u8set_transition(&mut self, _from: u32, _set: U8Set, _to: u32) {
        todo!("lexer NFA storage is intentionally deferred during automata cleanup")
    }

    pub fn add_epsilon(&mut self, _from: u32, _to: u32) {
        todo!("lexer NFA storage is intentionally deferred during automata cleanup")
    }

    pub fn add_finalizer(&mut self, _state: u32, _group_id: GroupId) {
        todo!("lexer NFA storage is intentionally deferred during automata cleanup")
    }

    pub fn add_non_greedy_finalizer(&mut self, _state: u32, _group_id: GroupId) {
        todo!("lexer NFA storage is intentionally deferred during automata cleanup")
    }

    pub fn epsilon_closure(&self, _states: &BTreeSet<u32>) -> BTreeSet<u32> {
        todo!("lexer NFA epsilon handling is intentionally deferred during automata cleanup")
    }
}
