//! Unweighted DFA skeleton used by parser templates.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

pub type Label = i32;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DFAState {
    pub is_accepting: bool,
    pub transitions: BTreeMap<Label, u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DFA {
    pub states: Vec<DFAState>,
    pub start_state: u32,
}

impl DFA {
    #[allow(unused_variables, unused_mut, dead_code)]
    pub fn new() -> Self {
        unimplemented!()
    }

    #[allow(unused_variables, unused_mut, dead_code)]
    pub fn num_states(&self) -> usize {
        unimplemented!()
    }

    #[allow(unused_variables, unused_mut, dead_code)]
    pub fn add_state(&mut self) -> u32 {
        unimplemented!()
    }

    #[allow(unused_variables, unused_mut, dead_code)]
    pub fn add_transition(&mut self, from: u32, label: Label, to: u32) {
        unimplemented!()
    }

    #[allow(unused_variables, unused_mut, dead_code)]
    pub fn set_accepting(&mut self, state: u32, is_accepting: bool) {
        unimplemented!()
    }
}

impl std::fmt::Display for DFA {
    #[allow(unused_variables, unused_mut, dead_code)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}
