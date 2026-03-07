//! Byte-level Nondeterministic Finite Automaton (NFA).
//!
//! Supports epsilon transitions and U8Set-based byte transitions.
//! Can be converted to a DFA via subset construction.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

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
        unimplemented!()
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
        unimplemented!()
    }

    /// Number of states.
    pub fn num_states(&self) -> usize {
        unimplemented!()
    }

    /// Add a new state and return its ID.
    pub fn add_state(&mut self) -> u32 {
        unimplemented!()
    }

    /// Add a single-byte transition.
    pub fn add_transition(&mut self, from: u32, byte: u8, to: u32) {
        unimplemented!()
    }

    /// Add a U8Set transition (transition on any byte in the set).
    pub fn add_u8set_transition(&mut self, from: u32, set: U8Set, to: u32) {
        unimplemented!()
    }

    /// Add an epsilon transition.
    pub fn add_epsilon(&mut self, from: u32, to: u32) {
        unimplemented!()
    }

    /// Mark a state as finalizing for a group.
    pub fn add_finalizer(&mut self, state: u32, group_id: GroupId) {
        unimplemented!()
    }

    /// Mark a state as finalizing for a non-greedy group.
    pub fn add_non_greedy_finalizer(&mut self, state: u32, group_id: GroupId) {
        unimplemented!()
    }

    /// Set whether a state is accepting (convenience, uses group 0).
    pub fn set_accepting(&mut self, state: u32, accepting: bool) {
        unimplemented!()
    }

    /// Whether a state is accepting.
    pub fn is_accepting(&self, state: u32) -> bool {
        unimplemented!()
    }

    /// Compute the epsilon closure of a set of states.
    pub fn epsilon_closure(&self, states: &BTreeSet<u32>) -> BTreeSet<u32> {
        unimplemented!()
    }

    /// Convert this NFA to a DFA via subset construction.
    ///
    /// Uses input equivalence classes to reduce the alphabet size,
    /// then builds the DFA using the standard powerset/subset construction.
    pub fn to_dfa(&self) -> Dfa {
        unimplemented!()
    }

    /// Standard subset construction NFA → DFA.
    fn subset_construction(&self) -> Dfa {
        unimplemented!()
    }

    /// Compute input equivalence classes.
    ///
    /// Returns `(class_map, num_classes, class_members)` where:
    /// - `class_map[byte]` = class ID for that byte
    /// - `num_classes` = number of distinct classes
    /// - `class_members[class]` = one representative byte for each class
    fn compute_equivalence_classes(&self) -> (Vec<u8>, u8, Vec<u8>) {
        unimplemented!()
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
