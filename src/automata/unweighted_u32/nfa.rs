//! Non-deterministic Finite Automaton (NFA) — unweighted, `u32` state IDs.
//!
//! Provides a lightweight NFA type with epsilon transitions that is used
//! primarily for template-DFA construction.  The template builder creates one
//! NFA per terminal characterization (with fresh intermediate states for each
//! path) and then determinizes it into an acyclic `DFA`.
//!
//! Unweighted NFA representation for integer-labeled automata.

use std::collections::BTreeMap;

use super::dfa::Label;

/// A single NFA state with non-deterministic transitions and epsilon edges.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NFAState {
    /// Whether this state is accepting (final).
    pub is_accepting: bool,
    /// Non-deterministic transitions: label → list of destination states.
    pub transitions: BTreeMap<Label, Vec<u32>>,
    /// Epsilon (unlabeled) transitions.
    pub epsilons: Vec<u32>,
}

/// Non-deterministic Finite Automaton with i32 labels and epsilon transitions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NFA {
    pub states: Vec<NFAState>,
    pub start_states: Vec<u32>,
}

impl NFA {
    /// Create a new NFA with a single start state (state 0).
    pub fn new() -> Self {
        Self {
            states: vec![NFAState::default()],
            start_states: vec![0],
        }
    }

    /// Create an empty NFA with no states.
    pub fn new_empty() -> Self {
        Self {
            states: Vec::new(),
            start_states: Vec::new(),
        }
    }

    /// Allocate a new state and return its ID.
    pub fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        self.states.push(NFAState::default());
        id
    }

    /// Number of states.
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    /// Add a labeled transition from `from` to `to`.
    pub fn add_transition(&mut self, from: u32, label: Label, to: u32) {
        self.states[from as usize]
            .transitions
            .entry(label)
            .or_default()
            .push(to);
    }

    /// Add an epsilon (unlabeled) transition from `from` to `to`.
    pub fn add_epsilon(&mut self, from: u32, to: u32) {
        self.states[from as usize].epsilons.push(to);
    }

    /// Mark a state as accepting.
    pub fn set_accepting(&mut self, state: u32) {
        self.states[state as usize].is_accepting = true;
    }

    /// Check if a state is accepting.
    pub fn is_accepting(&self, state: u32) -> bool {
        self.states
            .get(state as usize)
            .map_or(false, |s| s.is_accepting)
    }

    /// Returns `true` if the NFA's transition graph (including epsilon edges)
    /// contains no cycles.  Uses DFS 3-coloring.
    pub fn is_acyclic(&self) -> bool {
        let n = self.states.len();
        // 0 = white, 1 = gray, 2 = black
        let mut color = vec![0u8; n];

        fn visit(s: usize, states: &[NFAState], color: &mut [u8]) -> bool {
            color[s] = 1;
            // Check labeled transitions
            for targets in states[s].transitions.values() {
                for &t in targets {
                    let t = t as usize;
                    if t >= color.len() {
                        continue;
                    }
                    match color[t] {
                        1 => return false,
                        0 => {
                            if !visit(t, states, color) {
                                return false;
                            }
                        }
                        _ => {}
                    }
                }
            }
            // Check epsilon transitions
            for &t in &states[s].epsilons {
                let t = t as usize;
                if t >= color.len() {
                    continue;
                }
                match color[t] {
                    1 => return false,
                    0 => {
                        if !visit(t, states, color) {
                            return false;
                        }
                    }
                    _ => {}
                }
            }
            color[s] = 2;
            true
        }

        for s in 0..n {
            if color[s] == 0 && !visit(s, &self.states, &mut color) {
                return false;
            }
        }
        true
    }
}

impl std::fmt::Display for NFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "NFA: {} states, starts={:?}",
            self.states.len(),
            self.start_states
        )?;
        for (id, state) in self.states.iter().enumerate() {
            if state.transitions.is_empty() && state.epsilons.is_empty() && !state.is_accepting {
                continue;
            }
            let accept_mark = if state.is_accepting { " [ACCEPT]" } else { "" };
            writeln!(f, "  State {id}{accept_mark}")?;
            for (&label, targets) in &state.transitions {
                for &t in targets {
                    writeln!(f, "    {label} → State {t}")?;
                }
            }
            for &t in &state.epsilons {
                writeln!(f, "    ε → State {t}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nfa_basic() {
        let mut nfa = NFA::new();
        let s1 = nfa.add_state();
        let s2 = nfa.add_state();
        let start = nfa.start_states[0];

        nfa.add_transition(start, 1, s1);
        nfa.add_transition(s1, 2, s2);
        nfa.set_accepting(s2);

        assert_eq!(nfa.num_states(), 3);
        assert!(nfa.is_accepting(s2));
        assert!(!nfa.is_accepting(start));
    }

    #[test]
    fn test_nfa_acyclic_simple() {
        let mut nfa = NFA::new();
        let s1 = nfa.add_state();
        let s2 = nfa.add_state();
        nfa.add_transition(0, 1, s1);
        nfa.add_transition(s1, 2, s2);
        assert!(nfa.is_acyclic());
    }

    #[test]
    fn test_nfa_cyclic_labeled() {
        let mut nfa = NFA::new();
        let s1 = nfa.add_state();
        nfa.add_transition(0, 1, s1);
        nfa.add_transition(s1, 2, 0); // cycle: 0 → s1 → 0
        assert!(!nfa.is_acyclic());
    }

    #[test]
    fn test_nfa_cyclic_epsilon() {
        let mut nfa = NFA::new();
        let s1 = nfa.add_state();
        nfa.add_epsilon(0, s1);
        nfa.add_epsilon(s1, 0); // epsilon cycle
        assert!(!nfa.is_acyclic());
    }

    #[test]
    fn test_nfa_self_loop() {
        let mut nfa = NFA::new();
        nfa.add_transition(0, 1, 0); // self-loop
        assert!(!nfa.is_acyclic());
    }

    #[test]
    fn test_nfa_empty() {
        let nfa = NFA::new_empty();
        assert!(nfa.is_acyclic());
    }
}
