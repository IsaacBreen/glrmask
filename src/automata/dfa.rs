//! Byte-level Deterministic Finite Automaton (DFA).
//!
//! Operates on individual bytes (0..=255). Used to represent tokenizer patterns
//! and grammar terminal symbols at the byte level.
//!
//! Each DFA state has:
//! - A 256-entry transition table (one per byte)
//! - A set of "finalizer" group IDs (which regex groups are matched at this state)

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
    /// Number of states.
    num_states: usize,
}

impl Dfa {
    /// Create a DFA with the given number of states (all transitions dead, no finalizers).
    pub fn new(num_states: usize) -> Self {
        Self {
            transitions: vec![DEAD; num_states * 256],
            finalizers: vec![BTreeSet::new(); num_states],
            num_states,
        }
    }

    /// Number of states.
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Set a transition.
    #[inline]
    pub fn set_transition(&mut self, from: u32, byte: u8, to: u32) {
        self.transitions[from as usize * 256 + byte as usize] = to;
    }

    /// Get a transition. Returns `DEAD` if no transition.
    #[inline]
    pub fn get_transition(&self, from: u32, byte: u8) -> u32 {
        self.transitions[from as usize * 256 + byte as usize]
    }

    /// Get the set of bytes that have transitions from a state.
    pub fn get_u8set(&self, state: u32) -> U8Set {
        let base = state as usize * 256;
        let mut set = U8Set::empty();
        for b in 0..=255u8 {
            if self.transitions[base + b as usize] != DEAD {
                set.insert(b);
            }
        }
        set
    }

    /// Add a finalizer group ID to a state.
    pub fn add_finalizer(&mut self, state: u32, group_id: GroupId) {
        self.finalizers[state as usize].insert(group_id);
    }

    /// Set the finalizers for a state.
    pub fn set_finalizers(&mut self, state: u32, groups: BTreeSet<GroupId>) {
        self.finalizers[state as usize] = groups;
    }

    /// Get the finalizer group IDs for a state.
    pub fn finalizers(&self, state: u32) -> &BTreeSet<GroupId> {
        &self.finalizers[state as usize]
    }

    /// Whether a state is accepting (has any finalizer).
    pub fn is_accepting(&self, state: u32) -> bool {
        !self.finalizers[state as usize].is_empty()
    }

    /// Set whether a state is accepting (convenience, uses group 0).
    pub fn set_accepting(&mut self, state: u32, accepting: bool) {
        if accepting {
            self.finalizers[state as usize].insert(0);
        } else {
            self.finalizers[state as usize].clear();
        }
    }

    /// Run the DFA on a byte sequence from state 0. Returns the final state
    /// (or `DEAD` if any transition was dead).
    pub fn run(&self, input: &[u8]) -> u32 {
        let mut state = 0u32;
        for &byte in input {
            state = self.get_transition(state, byte);
            if state == DEAD {
                return DEAD;
            }
        }
        state
    }

    /// Whether the DFA accepts the given input.
    pub fn accepts(&self, input: &[u8]) -> bool {
        let final_state = self.run(input);
        final_state != DEAD && self.is_accepting(final_state)
    }

    /// Which group IDs match the given input (empty if no match).
    pub fn find_matches(&self, input: &[u8]) -> BTreeSet<GroupId> {
        let final_state = self.run(input);
        if final_state == DEAD {
            BTreeSet::new()
        } else {
            self.finalizers[final_state as usize].clone()
        }
    }

    /// Get the next state for a byte, returning `None` for dead.
    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        let next = self.get_transition(state, byte);
        if next == DEAD { None } else { Some(next) }
    }

    /// Access the full transition table.
    pub fn transitions(&self) -> &[u32] {
        &self.transitions
    }

    /// Minimize this DFA using Hopcroft's algorithm. Returns a new minimized DFA.
    pub fn minimize(&self) -> Dfa {
        hopcroft_minimize(self)
    }
}

/// Hopcroft's DFA minimization algorithm.
///
/// Groups states into equivalence classes based on their transition behavior
/// and finalizer sets. States with different finalizers or different transition
/// signatures (w.r.t. equivalence classes) are separated.
fn hopcroft_minimize(dfa: &Dfa) -> Dfa {
    let n = dfa.num_states();
    if n == 0 {
        return Dfa::new(0);
    }

    // Identify reachable states via BFS from state 0
    let mut reachable = vec![false; n];
    let mut stack = vec![0u32];
    reachable[0] = true;
    while let Some(s) = stack.pop() {
        for byte in 0..=255u8 {
            let t = dfa.get_transition(s, byte);
            if t != DEAD && !reachable[t as usize] {
                reachable[t as usize] = true;
                stack.push(t);
            }
        }
    }

    let reachable_states: Vec<usize> = (0..n).filter(|&s| reachable[s]).collect();
    if reachable_states.is_empty() {
        return Dfa::new(0);
    }

    // Initial partition: group by finalizer sets
    use std::collections::HashMap;
    let mut finalizer_to_class: HashMap<&BTreeSet<GroupId>, u32> = HashMap::new();
    let mut partition = vec![0u32; n];
    let mut num_classes = 0u32;

    for &s in &reachable_states {
        let fin = &dfa.finalizers[s];
        let class = *finalizer_to_class.entry(fin).or_insert_with(|| {
            let c = num_classes;
            num_classes += 1;
            c
        });
        partition[s] = class;
    }

    // Refine partitions until stable
    loop {
        let mut changed = false;
        let mut new_num_classes = num_classes;

        for class in 0..num_classes {
            let members: Vec<usize> = reachable_states
                .iter()
                .copied()
                .filter(|&s| partition[s] == class)
                .collect();

            if members.len() <= 1 {
                continue;
            }

            // Reference state
            let reference = members[0];
            let ref_sig: Vec<u32> = (0..=255u8)
                .map(|byte| {
                    let t = dfa.get_transition(reference as u32, byte);
                    if t == DEAD { u32::MAX } else { partition[t as usize] }
                })
                .collect();

            let mut split_groups: HashMap<Vec<u32>, Vec<usize>> = HashMap::new();

            for &s in &members[1..] {
                let sig: Vec<u32> = (0..=255u8)
                    .map(|byte| {
                        let t = dfa.get_transition(s as u32, byte);
                        if t == DEAD { u32::MAX } else { partition[t as usize] }
                    })
                    .collect();
                if sig != ref_sig {
                    split_groups.entry(sig).or_default().push(s);
                }
            }

            for (_sig, states) in &split_groups {
                let new_class = new_num_classes;
                new_num_classes += 1;
                for &s in states {
                    partition[s] = new_class;
                }
                changed = true;
            }
        }

        num_classes = new_num_classes;
        if !changed {
            break;
        }
    }

    // Remap classes so start state's class is 0
    let start_class = partition[0];
    let mut class_remap = vec![u32::MAX; num_classes as usize];
    class_remap[start_class as usize] = 0;
    let mut next_id = 1u32;
    for c in 0..num_classes {
        if class_remap[c as usize] == u32::MAX {
            if reachable_states.iter().any(|&s| partition[s] == c) {
                class_remap[c as usize] = next_id;
                next_id += 1;
            }
        }
    }
    let final_num_states = next_id as usize;
    let mut result = Dfa::new(final_num_states);

    for class in 0..num_classes {
        let new_class = class_remap[class as usize];
        if new_class == u32::MAX {
            continue;
        }
        if let Some(&rep) = reachable_states.iter().find(|&&s| partition[s] == class) {
            result.finalizers[new_class as usize] = dfa.finalizers[rep].clone();
            for byte in 0..=255u8 {
                let t = dfa.get_transition(rep as u32, byte);
                if t != DEAD {
                    let tc = class_remap[partition[t as usize] as usize];
                    if tc != u32::MAX {
                        result.set_transition(new_class, byte, tc);
                    }
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
}
