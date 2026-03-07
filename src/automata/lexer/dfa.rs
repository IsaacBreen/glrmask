//! Byte-level Deterministic Finite Automaton (DFA).
//!
//! Operates on individual bytes (0..=255). Used to represent tokenizer patterns
//! and grammar terminal symbols at the byte level.
//!
//! Each DFA state has:
//! - A 256-entry transition table (one per byte)
//! - A set of "finalizer" group IDs (which regex groups are matched at this state)
#![allow(dead_code)]
#![allow(unused_imports, unused_variables, dead_code)]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::ds::u8set::U8Set;

/// A dead/reject state sentinel.
pub const DEAD: u32 = u32::MAX;

/// A group ID identifying which regex alternative is matched.
pub type GroupId = usize;

/// A byte-level DFA with 256-way branching per state.
///
/// States are numbered `0..num_states`. Transitions are stored in a flat
/// array of size `num_states * 256`. State 0 is the start state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dfa {
    /// Flat transition table: `transitions[state * 256 + byte] = next_state`.
    /// `DEAD` means no transition.
    transitions: Vec<u32>,
    /// Per-state finalizer group IDs. `finalizers[state]` is the set of groups
    /// that match at this state. Empty means non-accepting.
    finalizers: Vec<BTreeSet<GroupId>>,
    /// Per-state subset of finalizers that came from non-greedy regex groups.
    non_greedy_finalizers: Vec<BTreeSet<GroupId>>,
    /// Per-state group IDs that remain reachable on some non-empty continuation.
    possible_future_group_ids: Vec<BTreeSet<GroupId>>,
    /// Number of states.
    num_states: usize,
}

impl Dfa {
    /// Create a DFA with the given number of states (all transitions dead, no finalizers).
    pub fn new(num_states: usize) -> Self {
        unimplemented!("cargo-check-only stub")
    }

    /// Number of states.
    pub fn num_states(&self) -> usize {
        unimplemented!("cargo-check-only stub")
    }

    /// Set a transition.
    #[inline]
    pub fn set_transition(&mut self, from: u32, byte: u8, to: u32) {
        unimplemented!("cargo-check-only stub")
    }

    /// Get a transition. Returns `DEAD` if no transition.
    #[inline]
    pub fn get_transition(&self, from: u32, byte: u8) -> u32 {
        unimplemented!("cargo-check-only stub")
    }

    /// Get the set of bytes that have transitions from a state.
    pub fn get_u8set(&self, state: u32) -> U8Set {
        unimplemented!("cargo-check-only stub")
    }

    /// Add a finalizer group ID to a state.
    pub fn add_finalizer(&mut self, state: u32, group_id: GroupId) {
        unimplemented!("cargo-check-only stub")
    }

    /// Add a non-greedy finalizer group ID to a state.
    pub fn add_non_greedy_finalizer(&mut self, state: u32, group_id: GroupId) {
        unimplemented!("cargo-check-only stub")
    }

    /// Set the finalizers for a state.
    pub fn set_finalizers(&mut self, state: u32, groups: BTreeSet<GroupId>) {
        unimplemented!("cargo-check-only stub")
    }

    /// Get the finalizer group IDs for a state.
    pub fn finalizers(&self, state: u32) -> &BTreeSet<GroupId> {
        unimplemented!("cargo-check-only stub")
    }

    /// Get the non-greedy finalizer group IDs for a state.
    pub fn non_greedy_finalizers(&self, state: u32) -> &BTreeSet<GroupId> {
        unimplemented!("cargo-check-only stub")
    }

    /// Get the group IDs reachable from this state on some non-empty suffix.
    pub fn possible_future_group_ids(&self, state: u32) -> &BTreeSet<GroupId> {
        unimplemented!("cargo-check-only stub")
    }

    /// Whether a state is accepting (has any finalizer).
    pub fn is_accepting(&self, state: u32) -> bool {
        unimplemented!("cargo-check-only stub")
    }

    /// Set whether a state is accepting (convenience, uses group 0).
    pub fn set_accepting(&mut self, state: u32, accepting: bool) {
        unimplemented!("cargo-check-only stub")
    }

    /// Run the DFA on a byte sequence from state 0. Returns the final state
    /// (or `DEAD` if any transition was dead).
    pub fn run(&self, input: &[u8]) -> u32 {
        unimplemented!("cargo-check-only stub")
    }

    /// Whether the DFA accepts the given input.
    pub fn accepts(&self, input: &[u8]) -> bool {
        unimplemented!("cargo-check-only stub")
    }

    /// Which group IDs match the given input (empty if no match).
    pub fn find_matches(&self, input: &[u8]) -> BTreeSet<GroupId> {
        unimplemented!("cargo-check-only stub")
    }

    /// Get the next state for a byte, returning `None` for dead.
    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        unimplemented!("cargo-check-only stub")
    }

    /// Access the full transition table.
    pub fn transitions(&self) -> &[u32] {
        unimplemented!("cargo-check-only stub")
    }

    /// Minimize this DFA using Hopcroft's algorithm. Returns a new minimized DFA.
    pub fn minimize(&self) -> Dfa {
        unimplemented!("cargo-check-only stub")
    }

    /// Recompute `possible_future_group_ids` from the current transition graph.
    pub fn recompute_possible_future_group_ids(&mut self) {
        unimplemented!("cargo-check-only stub")
    }
}

/// Hopcroft's DFA minimization algorithm.
///
/// Groups states into equivalence classes based on their transition behavior
/// and finalizer sets. States with different finalizers or different transition
/// signatures (w.r.t. equivalence classes) are separated.
fn hopcroft_minimize(dfa: &Dfa) -> Dfa {
    unimplemented!("cargo-check-only stub")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_dfa() {
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
        let mut dfa = Dfa::new(3);
        dfa.set_transition(0, b'a', 1);
        dfa.set_transition(0, b'b', 2);
        dfa.set_accepting(1, true);
        dfa.set_accepting(2, true);
        let min = dfa.minimize();
        assert_eq!(min.num_states(), 2);
        assert!(min.accepts(b"a"));
        assert!(min.accepts(b"b"));
    }

    #[test]
    fn test_group_ids() {
        let mut dfa = Dfa::new(3);
        dfa.set_transition(0, b'a', 1);
        dfa.set_transition(0, b'b', 2);
        dfa.add_finalizer(1, 0);
        dfa.add_finalizer(2, 1);
        let m1 = dfa.find_matches(b"a");
        assert!(m1.contains(&0));
        assert!(!m1.contains(&1));
        let m2 = dfa.find_matches(b"b");
        assert!(m2.contains(&1));
    }

    #[test]
    fn test_non_greedy_and_possible_future_metadata() {
        let mut dfa = Dfa::new(3);
        dfa.set_transition(0, b'a', 1);
        dfa.set_transition(1, b'b', 2);
        dfa.add_non_greedy_finalizer(1, 0);
        dfa.add_finalizer(2, 1);
        dfa.recompute_possible_future_group_ids();

        assert!(dfa.non_greedy_finalizers(1).contains(&0));
        assert!(dfa.possible_future_group_ids(0).contains(&0));
        assert!(dfa.possible_future_group_ids(0).contains(&1));
        assert!(dfa.possible_future_group_ids(1).contains(&1));
    }

    #[test]
    fn test_different_groups_not_merged() {
        let mut dfa = Dfa::new(3);
        dfa.set_transition(0, b'a', 1);
        dfa.set_transition(0, b'b', 2);
        dfa.add_finalizer(1, 0);
        dfa.add_finalizer(2, 1);
        let min = dfa.minimize();
        assert_eq!(min.num_states(), 3);
    }

    #[test]
    fn test_get_u8set() {
        let mut dfa = Dfa::new(2);
        dfa.set_transition(0, b'a', 1);
        dfa.set_transition(0, b'b', 1);
        dfa.set_transition(0, b'c', 1);
        let set = dfa.get_u8set(0);
        assert_eq!(set.len(), 3);
        assert!(set.contains(b'a'));
        assert!(set.contains(b'c'));
    }

    #[test]
    fn test_dfa_star_minimal() {
        // Minimal reproduction: x(\[ab])*y — after "x\c", DFA should be DEAD
        use crate::automata::regex::{ExprGroup, ExprGroups, class, seq, byte, star};
        use crate::ds::u8set::U8Set;
        use crate::automata::dfa::DEAD;
    
        // Pattern: x (\[ab])* y
        let pattern = seq(vec![
            byte(b'x'),
            star(seq(vec![
                byte(b'\\'),
                class(U8Set::from_bytes(&[b'a', b'b'])),
            ])),
            byte(b'y'),
        ]);
        let dfa = ExprGroups { groups: vec![ExprGroup { expr: pattern, is_non_greedy: false }] }.build();
    
        // "x" → alive (in star, could match \ab or y)
        let s1 = dfa.dfa.get_transition(0, b'x');
        eprintln!("After 'x': state={}", s1);
        assert_ne!(s1, DEAD, "should be alive after x");
    
        // "x\" → alive (started escape)
        let s2 = dfa.dfa.get_transition(s1, b'\\');
        eprintln!("After 'x\\': state={}", s2);
        assert_ne!(s2, DEAD, "should be alive after x\\backslash");
    
        // "x\a" → alive (valid escape, back in star)
        let s3 = dfa.dfa.get_transition(s2, b'a');
        eprintln!("After 'x\\a': state={}", s3);
        assert_ne!(s3, DEAD, "should be alive after valid escape");
    
        // "x\c" → should be DEAD (invalid escape char)
        let s4 = dfa.dfa.get_transition(s2, b'c');
        eprintln!("After 'x\\c': state={}", s4);
        assert_eq!(s4, DEAD, "MUST be DEAD after invalid escape char");
    
        // "x\ay" → should match (accepting)
        let s5 = dfa.dfa.get_transition(s3, b'y');
        eprintln!("After 'x\\ay': state={}, finalizers={:?}", s5, dfa.dfa.finalizers(s5));
        assert_ne!(s5, DEAD);
        assert!(!dfa.dfa.finalizers(s5).is_empty(), "should be accepting");
    }

    #[test]
    fn test_dfa_escape_simple() {
        // Simplified escape pattern: "(a|\b)*"
        use crate::automata::regex::{ExprGroup, ExprGroups, seq, byte, star, choice, class};
        use crate::ds::u8set::U8Set;
        use crate::automata::dfa::DEAD;
    
        // Pattern: "(a|\[bc])*"
        let pattern = seq(vec![
            byte(b'"'),
            star(choice(vec![
                byte(b'a'),
                seq(vec![
                    byte(b'\\'),
                    class(U8Set::from_bytes(&[b'b', b'c'])),
                ]),
            ])),
            byte(b'"'),
        ]);
        let dfa = ExprGroups { groups: vec![ExprGroup { expr: pattern, is_non_greedy: false }] }.build();
    
        eprintln!("DFA states: {}", dfa.dfa.num_states());
        eprintln!("\"\"\" → {}", dfa.is_match(b"\"\""));
        eprintln!("\"a\" → {}", dfa.is_match(b"\"a\""));
        eprintln!("\"\\b\" → {}", dfa.is_match(b"\"\\b\""));
        eprintln!("\"\\c\" → {}", dfa.is_match(b"\"\\c\""));
        eprintln!("\"\\.\" → {}", dfa.is_match(b"\"\\.\""));
        eprintln!("\"\\d\" → {}", dfa.is_match(b"\"\\d\""));
        eprintln!("\"a\\b\" → {}", dfa.is_match(b"\"a\\b\""));
        
        assert!(dfa.is_match(b"\"\""), "empty string match");
        assert!(dfa.is_match(b"\"a\""), "letter a match");
        assert!(dfa.is_match(b"\"\\b\""), "escape b match");
        assert!(dfa.is_match(b"\"\\c\""), "escape c match");
        assert!(!dfa.is_match(b"\"\\.\""), "\\. must NOT match (invalid escape)");
        assert!(!dfa.is_match(b"\"\\d\""), "\\d must NOT match (invalid escape)");
        
        // Trace DFA states
        let s0 = 0u32;
        let s1 = dfa.dfa.get_transition(s0, b'"');
        eprintln!("\nDFA trace:");
        eprintln!("  0 + '\"' -> {}", s1);
        let s2 = dfa.dfa.get_transition(s1, b'\\');
        eprintln!("  {} + '\\\\' -> {}", s1, s2);
        let s3 = dfa.dfa.get_transition(s2, b'.');
        eprintln!("  {} + '.' -> {} (DEAD={})", s2, s3, DEAD);
        let s4 = dfa.dfa.get_transition(s2, b'b');
        eprintln!("  {} + 'b' -> {}", s2, s4);
        let s5 = dfa.dfa.get_transition(s2, b'd');
        eprintln!("  {} + 'd' -> {} (DEAD={})", s2, s5, DEAD);
    }

}
