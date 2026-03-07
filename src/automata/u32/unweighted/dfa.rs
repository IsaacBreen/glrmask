//! Unweighted DFA skeleton used by parser templates.

use std::collections::BTreeMap;

pub type Label = i32;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DfaState {
    pub is_accepting: bool,
    pub transitions: BTreeMap<Label, u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Dfa {
    pub states: Vec<DfaState>,
    pub start_state: u32,
}

impl Dfa {
    #[allow(unused_variables, unused_mut, dead_code)]
    pub fn new() -> Self {
        unimplemented!("cargo-check-only stub")
    }

    #[allow(unused_variables, unused_mut, dead_code)]
    pub fn num_states(&self) -> usize {
        unimplemented!("cargo-check-only stub")
    }

    #[allow(unused_variables, unused_mut, dead_code)]
    pub fn add_state(&mut self) -> u32 {
        unimplemented!("cargo-check-only stub")
    }

    #[allow(unused_variables, unused_mut, dead_code)]
    pub fn add_transition(&mut self, from: u32, label: Label, to: u32) {
        unimplemented!("cargo-check-only stub")
    }

    #[allow(unused_variables, unused_mut, dead_code)]
    pub fn set_accepting(&mut self, state: u32, is_accepting: bool) {
        unimplemented!("cargo-check-only stub")
    }
}

impl std::fmt::Display for Dfa {
    #[allow(unused_variables, unused_mut, dead_code)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!("cargo-check-only stub")
    }
}
