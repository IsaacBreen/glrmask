//! NOTE: this file is intentionally gutted for a future `sep1`-style rewrite.
//! Keep the intended shape: explicit `CharTransitions`, `BitSet`-backed
//! finalizers and possible-future-group IDs, `DFAState`-owned
//! `possible_future_group_ids` behind a non-public `DFA` accessor, and
//! `DFA`-owned `group_id_to_u8set`.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use serde::{Deserialize, Serialize};

use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;

pub type GroupId = u32;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CharTransitions;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DFAState {
    pub transitions: CharTransitions,
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
        todo!("lexer DFA storage is being redesigned around the sep1-style shape")
    }

    pub fn num_states(&self) -> usize {
        todo!("lexer DFA storage is being redesigned around the sep1-style shape")
    }

    pub fn step(&self, _state: u32, _byte: u8) -> Option<u32> {
        todo!("lexer DFA transitions are being redesigned around CharTransitions")
    }

    pub fn get_u8set(&self, _state: u32) -> U8Set {
        todo!("lexer DFA transitions are being redesigned around CharTransitions")
    }

    pub fn group_id_to_u8set(&self, _group_id: GroupId) -> &U8Set {
        todo!("group_id_to_u8set will live on the sep1-style DFA")
    }

    pub fn finalizers(&self, _state: u32) -> &BitSet {
        todo!("lexer DFA finalizers are moving to BitSet-backed state storage")
    }

    pub fn non_greedy_finalizers(&self, _state: u32) -> &BitSet {
        todo!("lexer DFA finalizers are moving to BitSet-backed state storage")
    }

    pub(crate) fn possible_future_group_ids(&self, _state: u32) -> &BitSet {
        todo!("possible_future_group_ids should stay non-public on the sep1-style DFA")
    }

    pub fn states(&self) -> &[DFAState] {
        todo!("lexer DFA state layout is being redesigned around the sep1-style shape")
    }
}
