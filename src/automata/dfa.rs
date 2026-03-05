//! Byte-level Deterministic Finite Automaton (DFA).
//!
//! Operates on individual bytes (0..=255). Used to represent tokenizer patterns
//! and grammar terminal symbols at the byte level.

use serde::{Deserialize, Serialize};

/// A dead/reject state sentinel.
pub const DEAD_STATE: u32 = u32::MAX;

/// A byte-level DFA with 256-way branching per state.
///
/// States are numbered `0..num_states`. Transitions are stored in a flat
/// array of size `num_states * 256`. State 0 is the start state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dfa {
    /// Flat transition table: `transitions[state * 256 + byte] = next_state`.
    /// `DEAD_STATE` means no transition.
    transitions: Vec<u32>,
    /// Which states are accepting.
    accepting: Vec<bool>,
    /// Number of states.
    num_states: usize,
}

impl Dfa {
    /// Create a DFA with the given number of states (all transitions dead, no accepting states).
    pub fn new(num_states: usize) -> Self {
        Self {
            transitions: vec![DEAD_STATE; num_states * 256],
            accepting: vec![false; num_states],
            num_states,
        }
    }

    /// Number of states.
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Set a transition.
    pub fn set_transition(&mut self, from: u32, byte: u8, to: u32) {
        self.transitions[from as usize * 256 + byte as usize] = to;
    }

    /// Get a transition.
    pub fn get_transition(&self, from: u32, byte: u8) -> u32 {
        self.transitions[from as usize * 256 + byte as usize]
    }

    /// Set whether a state is accepting.
    pub fn set_accepting(&mut self, state: u32, accepting: bool) {
        self.accepting[state as usize] = accepting;
    }

    /// Whether a state is accepting.
    pub fn is_accepting(&self, state: u32) -> bool {
        self.accepting[state as usize]
    }

    /// Run the DFA on a byte sequence from state 0. Returns the final state
    /// (or `DEAD_STATE` if any transition was dead).
    pub fn run(&self, input: &[u8]) -> u32 {
        let mut state = 0u32;
        for &byte in input {
            state = self.get_transition(state, byte);
            if state == DEAD_STATE {
                return DEAD_STATE;
            }
        }
        state
    }

    /// Whether the DFA accepts the given input.
    pub fn accepts(&self, input: &[u8]) -> bool {
        let final_state = self.run(input);
        final_state != DEAD_STATE && self.is_accepting(final_state)
    }

    /// Minimize this DFA using Hopcroft's algorithm. Returns a new minimized DFA.
    pub fn minimize(&self) -> Dfa {
        hopcroft_minimize(self)
    }

    /// Access the full transition table.
    pub fn transitions(&self) -> &[u32] {
        &self.transitions
    }
}

/// Hopcroft's DFA minimization algorithm.
fn hopcroft_minimize(dfa: &Dfa) -> Dfa {
    let n = dfa.num_states();
    if n == 0 {
        return Dfa::new(0);
    }

    // Identify reachable states
    let mut reachable = vec![false; n];
    let mut stack = vec![0u32];
    reachable[0] = true;
    while let Some(s) = stack.pop() {
        for byte in 0..=255u8 {
            let t = dfa.get_transition(s, byte);
            if t != DEAD_STATE && !reachable[t as usize] {
                reachable[t as usize] = true;
                stack.push(t);
            }
        }
    }

    // Initial partition: accepting vs non-accepting (only reachable states)
    let mut partition = vec![0u32; n]; // partition[state] = class
    let mut num_classes = 1u32;

    // Separate accepting from non-accepting
    let has_accepting = (0..n).any(|s| reachable[s] && dfa.is_accepting(s as u32));
    let has_non_accepting = (0..n).any(|s| reachable[s] && !dfa.is_accepting(s as u32));

    if has_accepting && has_non_accepting {
        num_classes = 2;
        for s in 0..n {
            if reachable[s] && dfa.is_accepting(s as u32) {
                partition[s] = 1;
            }
        }
    } else if has_accepting {
        // All reachable states are accepting
    } else {
        // All reachable states are non-accepting
    }

    // Refine partitions until stable
    loop {
        let mut changed = false;
        let mut new_num_classes = num_classes;

        for class in 0..num_classes {
            let members: Vec<usize> = (0..n)
                .filter(|&s| reachable[s] && partition[s] == class)
                .collect();

            if members.len() <= 1 {
                continue;
            }

            // Try to distinguish states in this class by their transitions
            let reference = members[0];
            let mut split_off = Vec::new();

            for &s in &members[1..] {
                let mut differs = false;
                for byte in 0..=255u8 {
                    let t_ref = dfa.get_transition(reference as u32, byte);
                    let t_s = dfa.get_transition(s as u32, byte);
                    let class_ref = if t_ref == DEAD_STATE {
                        u32::MAX
                    } else {
                        partition[t_ref as usize]
                    };
                    let class_s = if t_s == DEAD_STATE {
                        u32::MAX
                    } else {
                        partition[t_s as usize]
                    };
                    if class_ref != class_s {
                        differs = true;
                        break;
                    }
                }
                if differs {
                    split_off.push(s);
                }
            }

            if !split_off.is_empty() {
                for &s in &split_off {
                    partition[s] = new_num_classes;
                }
                new_num_classes += 1;
                changed = true;
            }
        }

        num_classes = new_num_classes;
        if !changed {
            break;
        }
    }

    // Build minimized DFA
    // Remap classes so that the start state's class is 0
    let start_class = partition[0];
    let mut class_remap = vec![u32::MAX; num_classes as usize];
    class_remap[start_class as usize] = 0;
    let mut next_id = 1u32;
    for c in 0..num_classes {
        if class_remap[c as usize] == u32::MAX {
            class_remap[c as usize] = next_id;
            next_id += 1;
        }
    }
    let final_num_states = next_id as usize;

    let mut result = Dfa::new(final_num_states);

    // For each class, pick a representative and build transitions
    for class in 0..num_classes {
        let new_class = class_remap[class as usize];
        if let Some(rep) = (0..n).find(|&s| reachable[s] && partition[s] == class) {
            if dfa.is_accepting(rep as u32) {
                result.set_accepting(new_class, true);
            }
            for byte in 0..=255u8 {
                let t = dfa.get_transition(rep as u32, byte);
                if t != DEAD_STATE {
                    let target_class = class_remap[partition[t as usize] as usize];
                    result.set_transition(new_class, byte, target_class);
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_dfa() {
        // DFA that accepts "ab"
        let mut dfa = Dfa::new(3);
        dfa.set_transition(0, b'a', 1);
        dfa.set_transition(1, b'b', 2);
        dfa.set_accepting(2, true);

        assert!(dfa.accepts(b"ab"));
        assert!(!dfa.accepts(b"a"));
        assert!(!dfa.accepts(b"abc"));
        assert!(!dfa.accepts(b""));
    }

    #[test]
    fn test_minimize_identity() {
        // Already minimal DFA
        let mut dfa = Dfa::new(2);
        dfa.set_transition(0, b'a', 1);
        dfa.set_accepting(1, true);

        let min = dfa.minimize();
        assert_eq!(min.num_states(), 2);
        assert!(min.accepts(b"a"));
        assert!(!min.accepts(b""));
    }

    #[test]
    fn test_minimize_merges() {
        // DFA with two equivalent accepting states
        let mut dfa = Dfa::new(3);
        dfa.set_transition(0, b'a', 1);
        dfa.set_transition(0, b'b', 2);
        dfa.set_accepting(1, true);
        dfa.set_accepting(2, true);

        let min = dfa.minimize();
        assert_eq!(min.num_states(), 2); // merged states 1 and 2
        assert!(min.accepts(b"a"));
        assert!(min.accepts(b"b"));
        assert!(!min.accepts(b""));
    }
}
