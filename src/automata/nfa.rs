//! Byte-level Nondeterministic Finite Automaton (NFA).
//!
//! Supports epsilon transitions. Can be converted to a DFA via subset construction.

use std::collections::{BTreeSet, HashMap, VecDeque};

use super::dfa::Dfa;

/// A byte-level NFA with epsilon transitions.
///
/// States are numbered `0..num_states`. State 0 is the start state.
#[derive(Debug, Clone)]
pub struct Nfa {
    /// `byte_transitions[state]` maps byte -> set of target states.
    byte_transitions: Vec<HashMap<u8, Vec<u32>>>,
    /// `epsilon_transitions[state]` = list of epsilon-reachable states.
    epsilon_transitions: Vec<Vec<u32>>,
    /// Which states are accepting.
    accepting: Vec<bool>,
    /// Number of states.
    num_states: usize,
}

impl Nfa {
    /// Create an NFA with the given number of states.
    pub fn new(num_states: usize) -> Self {
        Self {
            byte_transitions: vec![HashMap::new(); num_states],
            epsilon_transitions: vec![Vec::new(); num_states],
            accepting: vec![false; num_states],
            num_states,
        }
    }

    /// Number of states.
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Add a byte transition.
    pub fn add_transition(&mut self, from: u32, byte: u8, to: u32) {
        self.byte_transitions[from as usize]
            .entry(byte)
            .or_default()
            .push(to);
    }

    /// Add an epsilon transition.
    pub fn add_epsilon(&mut self, from: u32, to: u32) {
        self.epsilon_transitions[from as usize].push(to);
    }

    /// Set whether a state is accepting.
    pub fn set_accepting(&mut self, state: u32, accepting: bool) {
        self.accepting[state as usize] = accepting;
    }

    /// Whether a state is accepting.
    pub fn is_accepting(&self, state: u32) -> bool {
        self.accepting[state as usize]
    }

    /// Compute the epsilon closure of a set of states.
    pub fn epsilon_closure(&self, states: &BTreeSet<u32>) -> BTreeSet<u32> {
        let mut closure = states.clone();
        let mut stack: Vec<u32> = states.iter().cloned().collect();
        while let Some(s) = stack.pop() {
            for &t in &self.epsilon_transitions[s as usize] {
                if closure.insert(t) {
                    stack.push(t);
                }
            }
        }
        closure
    }

    /// Convert this NFA to a DFA via subset construction.
    pub fn to_dfa(&self) -> Dfa {
        let start_set = {
            let mut s = BTreeSet::new();
            s.insert(0);
            self.epsilon_closure(&s)
        };

        let mut state_map: HashMap<BTreeSet<u32>, u32> = HashMap::new();
        let mut dfa_states: Vec<BTreeSet<u32>> = Vec::new();
        let mut queue: VecDeque<u32> = VecDeque::new();

        let start_id = 0u32;
        state_map.insert(start_set.clone(), start_id);
        dfa_states.push(start_set);
        queue.push_back(start_id);

        // Process all reachable subsets
        while let Some(dfa_state) = queue.pop_front() {
            for byte in 0..=255u8 {
                let mut next_set = BTreeSet::new();
                for &nfa_state in &dfa_states[dfa_state as usize] {
                    if let Some(targets) = self.byte_transitions[nfa_state as usize].get(&byte) {
                        for &t in targets {
                            next_set.insert(t);
                        }
                    }
                }
                if next_set.is_empty() {
                    continue;
                }
                let next_set = self.epsilon_closure(&next_set);
                let next_id = if let Some(&id) = state_map.get(&next_set) {
                    id
                } else {
                    let id = dfa_states.len() as u32;
                    state_map.insert(next_set.clone(), id);
                    dfa_states.push(next_set);
                    queue.push_back(id);
                    id
                };
                // We'll set transitions later once we know the total count
                // For now, record in a temp structure
                // Actually, we need to build the DFA after processing
                let _ = next_id; // handled below
            }
        }

        // Rebuild: we need transitions. Let's redo with a proper builder.
        // Reset and rebuild properly
        state_map.clear();
        dfa_states.clear();

        let start_set_2 = {
            let mut s = BTreeSet::new();
            s.insert(0);
            self.epsilon_closure(&s)
        };
        state_map.insert(start_set_2.clone(), 0);
        dfa_states.push(start_set_2);
        let mut transitions: Vec<Vec<(u8, u32)>> = vec![Vec::new()];
        let mut queue2: VecDeque<u32> = VecDeque::new();
        queue2.push_back(0);

        while let Some(dfa_state) = queue2.pop_front() {
            for byte in 0..=255u8 {
                let mut next_set = BTreeSet::new();
                for &nfa_state in &dfa_states[dfa_state as usize] {
                    if let Some(targets) = self.byte_transitions[nfa_state as usize].get(&byte) {
                        for &t in targets {
                            next_set.insert(t);
                        }
                    }
                }
                if next_set.is_empty() {
                    continue;
                }
                let next_set = self.epsilon_closure(&next_set);
                let next_id = if let Some(&id) = state_map.get(&next_set) {
                    id
                } else {
                    let id = dfa_states.len() as u32;
                    state_map.insert(next_set.clone(), id);
                    dfa_states.push(next_set);
                    transitions.push(Vec::new());
                    queue2.push_back(id);
                    id
                };
                transitions[dfa_state as usize].push((byte, next_id));
            }
        }

        // Build final DFA
        let num_dfa_states = dfa_states.len();
        let mut dfa = Dfa::new(num_dfa_states);

        for (dfa_id, nfa_states) in dfa_states.iter().enumerate() {
            // Mark accepting if any NFA state in the subset is accepting
            if nfa_states.iter().any(|&s| self.is_accepting(s)) {
                dfa.set_accepting(dfa_id as u32, true);
            }
            // Set transitions
            for &(byte, target) in &transitions[dfa_id] {
                dfa.set_transition(dfa_id as u32, byte, target);
            }
        }

        dfa
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_nfa_to_dfa() {
        // NFA for "a|b"
        let mut nfa = Nfa::new(4);
        nfa.add_epsilon(0, 1);
        nfa.add_epsilon(0, 2);
        nfa.add_transition(1, b'a', 3);
        nfa.add_transition(2, b'b', 3);
        nfa.set_accepting(3, true);

        let dfa = nfa.to_dfa();
        assert!(dfa.accepts(b"a"));
        assert!(dfa.accepts(b"b"));
        assert!(!dfa.accepts(b"c"));
        assert!(!dfa.accepts(b""));
        assert!(!dfa.accepts(b"ab"));
    }

    #[test]
    fn test_epsilon_closure() {
        let mut nfa = Nfa::new(4);
        nfa.add_epsilon(0, 1);
        nfa.add_epsilon(1, 2);
        nfa.add_epsilon(2, 3);

        let mut start = BTreeSet::new();
        start.insert(0);
        let closure = nfa.epsilon_closure(&start);
        assert_eq!(closure, [0, 1, 2, 3].iter().cloned().collect());
    }
}
