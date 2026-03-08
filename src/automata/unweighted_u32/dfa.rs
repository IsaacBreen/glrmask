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
    pub fn new() -> Self {
        Self {
            states: vec![DFAState::default()],
            start_state: 0,
        }
    }

    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    pub fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        self.states.push(DFAState::default());
        id
    }

    pub fn add_transition(&mut self, from: u32, label: Label, to: u32) {
        if let Some(state) = self.states.get_mut(from as usize) {
            state.transitions.insert(label, to);
        }
    }

    pub fn set_accepting(&mut self, state: u32, is_accepting: bool) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.is_accepting = is_accepting;
        }
    }
}

impl std::fmt::Display for DFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DFA: {} states, start=State {}", self.states.len(), self.start_state)?;
        for (state_id, state) in self.states.iter().enumerate() {
            if state.transitions.is_empty() && !state.is_accepting {
                continue;
            }

            let start_mark = if state_id as u32 == self.start_state { " [START]" } else { "" };
            let accept_mark = if state.is_accepting { " [ACCEPT]" } else { "" };
            writeln!(f, "  State {state_id}{start_mark}{accept_mark}")?;
            for (label, target) in &state.transitions {
                writeln!(f, "    {label} → State {target}")?;
            }
        }
        Ok(())
    }
}
