use std::collections::BTreeMap;

pub type Label = i32;

#[derive(
    Debug, Clone, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct DFAState {
    pub is_accepting: bool,
    pub transitions: BTreeMap<Label, u32>,
}

#[derive(
    Debug, Clone, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
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

    /// Compute whether the DFA's transition graph contains no cycles.
    ///
    /// The graph is publicly mutable, so this cannot be safely cached. The
    /// method name deliberately exposes the O(states + transitions) cost.
    pub fn compute_is_acyclic(&self) -> bool {
        let num_states = self.states.len();
        let mut indegree = vec![0u32; num_states];
        for state in &self.states {
            for &target in state.transitions.values() {
                if let Some(degree) = indegree.get_mut(target as usize) {
                    *degree += 1;
                }
            }
        }
        let mut queue = std::collections::VecDeque::new();
        for (state, &degree) in indegree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(state);
            }
        }
        let mut visited = 0usize;
        while let Some(state) = queue.pop_front() {
            visited += 1;
            for &target in self.states[state].transitions.values() {
                let Some(degree) = indegree.get_mut(target as usize) else {
                    continue;
                };
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(target as usize);
                }
            }
        }
        visited == num_states
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

#[cfg(test)]
mod acyclicity_tests {
    use super::*;

    #[test]
    fn iterative_acyclicity_handles_deep_graphs_and_cycles() {
        let mut dfa = DFA::new();
        let mut previous = 0u32;
        for label in 0..100_000i32 {
            let next = dfa.add_state();
            dfa.add_transition(previous, label, next);
            previous = next;
        }
        assert!(dfa.compute_is_acyclic());
        dfa.add_transition(previous, i32::MIN, 0);
        assert!(!dfa.compute_is_acyclic());
    }
}
