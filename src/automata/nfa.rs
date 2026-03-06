//! Byte-level Nondeterministic Finite Automaton (NFA).
//!
//! Supports epsilon transitions and U8Set-based byte transitions.
//! Can be converted to a DFA via subset construction.
#![allow(dead_code)]

use std::collections::{BTreeSet, HashMap, VecDeque};

use crate::ds::u8set::U8Set;

use super::dfa::{Dfa, GroupId};

/// An NFA state.
#[derive(Debug, Clone)]
struct NfaState {
    /// Byte transitions: `(byte_set, target_state)`.
    /// Multiple entries can overlap (nondeterminism).
    transitions: Vec<(U8Set, u32)>,
    /// Epsilon transitions: target states reachable without consuming input.
    epsilon_transitions: Vec<u32>,
    /// Finalizer group IDs (which groups match at this state).
    finalizers: BTreeSet<GroupId>,
    /// Subset of finalizers corresponding to non-greedy groups.
    non_greedy_finalizers: BTreeSet<GroupId>,
}

impl NfaState {
    fn new() -> Self {
        Self {
            transitions: Vec::new(),
            epsilon_transitions: Vec::new(),
            finalizers: BTreeSet::new(),
            non_greedy_finalizers: BTreeSet::new(),
        }
    }
}

/// A byte-level NFA with epsilon transitions.
///
/// States are numbered `0..num_states()`. State 0 is the start state.
#[derive(Debug, Clone)]
pub struct Nfa {
    states: Vec<NfaState>,
}

impl Nfa {
    /// Create an NFA with the given number of states.
    pub fn new(num_states: usize) -> Self {
        Self {
            states: (0..num_states).map(|_| NfaState::new()).collect(),
        }
    }

    /// Number of states.
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    /// Add a new state and return its ID.
    pub fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        self.states.push(NfaState::new());
        id
    }

    /// Add a single-byte transition.
    pub fn add_transition(&mut self, from: u32, byte: u8, to: u32) {
        self.states[from as usize]
            .transitions
            .push((U8Set::from_byte(byte), to));
    }

    /// Add a U8Set transition (transition on any byte in the set).
    pub fn add_u8set_transition(&mut self, from: u32, set: U8Set, to: u32) {
        if !set.is_empty() {
            self.states[from as usize].transitions.push((set, to));
        }
    }

    /// Add an epsilon transition.
    pub fn add_epsilon(&mut self, from: u32, to: u32) {
        self.states[from as usize].epsilon_transitions.push(to);
    }

    /// Mark a state as finalizing for a group.
    pub fn add_finalizer(&mut self, state: u32, group_id: GroupId) {
        self.states[state as usize].finalizers.insert(group_id);
    }

    /// Mark a state as finalizing for a non-greedy group.
    pub fn add_non_greedy_finalizer(&mut self, state: u32, group_id: GroupId) {
        self.states[state as usize].finalizers.insert(group_id);
        self.states[state as usize]
            .non_greedy_finalizers
            .insert(group_id);
    }

    /// Set whether a state is accepting (convenience, uses group 0).
    pub fn set_accepting(&mut self, state: u32, accepting: bool) {
        if accepting {
            self.states[state as usize].finalizers.insert(0);
        } else {
            self.states[state as usize].finalizers.clear();
        }
    }

    /// Whether a state is accepting.
    pub fn is_accepting(&self, state: u32) -> bool {
        !self.states[state as usize].finalizers.is_empty()
    }

    /// Compute the epsilon closure of a set of states.
    pub fn epsilon_closure(&self, states: &BTreeSet<u32>) -> BTreeSet<u32> {
        let mut closure = states.clone();
        let mut stack: Vec<u32> = states.iter().cloned().collect();
        while let Some(s) = stack.pop() {
            for &t in &self.states[s as usize].epsilon_transitions {
                if closure.insert(t) {
                    stack.push(t);
                }
            }
        }
        closure
    }

    /// Convert this NFA to a DFA via subset construction.
    ///
    /// Uses input equivalence classes to reduce the alphabet size,
    /// then builds the DFA using the standard powerset/subset construction.
    pub fn to_dfa(&self) -> Dfa {
        self.subset_construction()
    }

    /// Standard subset construction NFA → DFA.
    fn subset_construction(&self) -> Dfa {
        // Compute input equivalence classes
        // Two bytes are equivalent if they trigger the exact same set of transitions
        // in every NFA state. This reduces the 256 iterations per state to typically
        // ~20-40.
        let (class_map, num_classes, class_members) = self.compute_equivalence_classes();

        let start_set = {
            let mut s = BTreeSet::new();
            s.insert(0u32);
            self.epsilon_closure(&s)
        };

        let mut state_map: HashMap<BTreeSet<u32>, u32> = HashMap::new();
        let mut dfa_states: Vec<BTreeSet<u32>> = Vec::new();
        let mut transitions: Vec<Vec<(u8, u32)>> = Vec::new();
        let mut queue: VecDeque<u32> = VecDeque::new();

        state_map.insert(start_set.clone(), 0);
        dfa_states.push(start_set);
        transitions.push(Vec::new());
        queue.push_back(0);

        while let Some(dfa_state) = queue.pop_front() {
            // For each equivalence class, compute reachable NFA states
            for class in 0..num_classes {
                let representative_byte = class_members[class as usize];

                let mut next_set = BTreeSet::new();
                for &nfa_state in &dfa_states[dfa_state as usize] {
                    for &(ref byte_set, target) in &self.states[nfa_state as usize].transitions {
                        if byte_set.contains(representative_byte) {
                            next_set.insert(target);
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
                    queue.push_back(id);
                    id
                };

                // Map all bytes in this class to this target
                for b in 0..=255u8 {
                    if class_map[b as usize] == class {
                        transitions[dfa_state as usize].push((b, next_id));
                    }
                }
            }
        }

        // Build final DFA
        let num_dfa_states = dfa_states.len();
        let mut dfa = Dfa::new(num_dfa_states);

        for (dfa_id, nfa_states) in dfa_states.iter().enumerate() {
            // Merge finalizers from all NFA states in this subset
            for &nfa_s in nfa_states {
                for &group in &self.states[nfa_s as usize].finalizers {
                    dfa.add_finalizer(dfa_id as u32, group);
                }
                for &group in &self.states[nfa_s as usize].non_greedy_finalizers {
                    dfa.add_non_greedy_finalizer(dfa_id as u32, group);
                }
            }
            // Set transitions
            for &(byte, target) in &transitions[dfa_id] {
                dfa.set_transition(dfa_id as u32, byte, target);
            }
        }

        dfa.recompute_possible_future_group_ids();

        dfa
    }

    /// Compute input equivalence classes.
    ///
    /// Returns `(class_map, num_classes, class_members)` where:
    /// - `class_map[byte]` = class ID for that byte
    /// - `num_classes` = number of distinct classes
    /// - `class_members[class]` = one representative byte for each class
    fn compute_equivalence_classes(&self) -> (Vec<u8>, u8, Vec<u8>) {
        // Build a signature for each byte: which (state, transition_index) pairs include it
        let mut byte_signatures: Vec<Vec<(u32, usize)>> = vec![Vec::new(); 256];

        for (state_idx, state) in self.states.iter().enumerate() {
            for (trans_idx, &(ref byte_set, _target)) in state.transitions.iter().enumerate() {
                for b in byte_set.iter() {
                    byte_signatures[b as usize].push((state_idx as u32, trans_idx));
                }
            }
        }

        // Group bytes by identical signatures
        let mut sig_to_class: HashMap<&Vec<(u32, usize)>, u8> = HashMap::new();
        let mut class_map = vec![0u8; 256];
        let mut class_members = Vec::new();
        let mut num_classes = 0u8;

        for b in 0..=255u8 {
            let sig = &byte_signatures[b as usize];
            if let Some(&class) = sig_to_class.get(sig) {
                class_map[b as usize] = class;
            } else {
                let class = num_classes;
                num_classes = num_classes.saturating_add(1);
                sig_to_class.insert(sig, class);
                class_map[b as usize] = class;
                class_members.push(b);
            }
        }

        (class_map, num_classes, class_members)
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

    #[test]
    fn test_u8set_transition() {
        // NFA that accepts any digit
        let mut nfa = Nfa::new(2);
        let digits = U8Set::from_range(b'0', b'9');
        nfa.add_u8set_transition(0, digits, 1);
        nfa.set_accepting(1, true);

        let dfa = nfa.to_dfa();
        assert!(dfa.accepts(b"0"));
        assert!(dfa.accepts(b"5"));
        assert!(dfa.accepts(b"9"));
        assert!(!dfa.accepts(b"a"));
        assert!(!dfa.accepts(b""));
        assert!(!dfa.accepts(b"12"));
    }

    #[test]
    fn test_multi_group() {
        // NFA with two groups: group 0 matches "a", group 1 matches "b"
        let mut nfa = Nfa::new(4);
        nfa.add_epsilon(0, 1);
        nfa.add_epsilon(0, 2);
        nfa.add_transition(1, b'a', 3);
        nfa.add_finalizer(3, 0);
        let s4 = nfa.add_state();
        nfa.add_transition(2, b'b', s4);
        nfa.add_finalizer(s4, 1);

        let dfa = nfa.to_dfa();
        let m_a = dfa.find_matches(b"a");
        assert!(m_a.contains(&0));
        assert!(!m_a.contains(&1));

        let m_b = dfa.find_matches(b"b");
        assert!(!m_b.contains(&0));
        assert!(m_b.contains(&1));
    }

    #[test]
    fn test_non_greedy_finalizers_propagate_to_dfa() {
        let mut nfa = Nfa::new(2);
        nfa.add_transition(0, b'a', 1);
        nfa.add_non_greedy_finalizer(1, 3);

        let dfa = nfa.to_dfa();
        let accept = dfa.run(b"a");
        assert!(dfa.finalizers(accept).contains(&3));
        assert!(dfa.non_greedy_finalizers(accept).contains(&3));
    }

    #[test]
    fn test_star() {
        // NFA for a*
        let mut nfa = Nfa::new(2);
        nfa.add_transition(0, b'a', 1);
        nfa.add_epsilon(1, 0); // loop back
        nfa.set_accepting(0, true); // accept empty
        nfa.set_accepting(1, true);

        let dfa = nfa.to_dfa();
        assert!(dfa.accepts(b""));
        assert!(dfa.accepts(b"a"));
        assert!(dfa.accepts(b"aaa"));
        assert!(!dfa.accepts(b"b"));
        assert!(!dfa.accepts(b"ab"));
    }

    #[test]
    fn test_equivalence_classes() {
        // NFA with transitions on [a-z] and [0-9]
        // Should produce ~3 classes: letters, digits, everything else
        let mut nfa = Nfa::new(3);
        nfa.add_u8set_transition(0, U8Set::from_range(b'a', b'z'), 1);
        nfa.add_u8set_transition(0, U8Set::from_range(b'0', b'9'), 2);

        let (class_map, num_classes, _) = nfa.compute_equivalence_classes();
        // All letters should have the same class
        let letter_class = class_map[b'a' as usize];
        for b in b'b'..=b'z' {
            assert_eq!(class_map[b as usize], letter_class);
        }
        // All digits should have the same class
        let digit_class = class_map[b'0' as usize];
        for b in b'1'..=b'9' {
            assert_eq!(class_map[b as usize], digit_class);
        }
        // Letters and digits should be different classes
        assert_ne!(letter_class, digit_class);
        // Should be 3 classes: letters, digits, other
        assert_eq!(num_classes, 3);
    }
}
