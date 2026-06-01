use std::collections::BTreeMap;

pub type Label = i32;

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct DFAState {
    pub is_accepting: bool,
    pub transitions: BTreeMap<Label, u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct DFA {
    pub states: Vec<DFAState>,
    pub start_state: u32,
}

fn has_self_loop(state_id: usize, state: &DFAState) -> bool {
    state.transitions.values().any(|&target| target as usize == state_id)
}

fn visit_successors(
    state_id: usize,
    states: &[DFAState],
    colors: &mut [u8],
) -> bool {
    colors[state_id] = 1;
    for target in states[state_id].transitions.values() {
        let target = *target as usize;
        if target >= colors.len() {
            continue;
        }
        match colors[target] {
            1 => return false,
            0 => {
                if !visit_successors(target, states, colors) {
                    return false;
                }
            }
            _ => {}
        }
    }
    colors[state_id] = 2;
    true
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

    /// Returns `true` if the DFA's transition graph contains no cycles.
    pub fn is_acyclic(&self) -> bool {
        let num_states = self.states.len();

        for (state_id, state) in self.states.iter().enumerate() {
            if has_self_loop(state_id, state) {
                return false;
            }
        }

        let mut colors = vec![0u8; num_states];
        for state_id in 0..num_states {
            if colors[state_id] == 0 && !visit_successors(state_id, &self.states, &mut colors) {
                return false;
            }
        }
        true
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
