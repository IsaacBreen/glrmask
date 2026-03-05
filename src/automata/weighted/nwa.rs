//! Nondeterministic Weighted Automaton (NWA).
//!
//! The NWA is an intermediate representation produced by the compiler
//! before determinization into a DWA.

use serde::{Deserialize, Serialize};

use super::weight::Tsid;

/// A transition in the NWA.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NwaTransition {
    /// Target state.
    pub target: u32,
    /// Token-set ID that triggers this transition.
    pub tsid: Tsid,
    /// Weight of this transition.
    pub weight: i32,
}

/// A Nondeterministic Weighted Automaton.
///
/// Multiple transitions may be possible for the same (state, tsid) pair.
/// This is resolved by determinization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nwa {
    /// Transitions indexed by source state.
    pub transitions: Vec<Vec<NwaTransition>>,
    /// The start state.
    pub start_state: u32,
    /// Which states are accepting.
    pub accepting: Vec<bool>,
    /// Number of states.
    pub num_states: u32,
    /// Number of token-set IDs.
    pub num_tsids: u32,
}

impl Nwa {
    /// Create a new NWA.
    pub fn new(num_states: u32, num_tsids: u32) -> Self {
        Self {
            transitions: vec![Vec::new(); num_states as usize],
            start_state: 0,
            accepting: vec![false; num_states as usize],
            num_states,
            num_tsids,
        }
    }

    /// Add a transition.
    pub fn add_transition(&mut self, from: u32, tsid: Tsid, to: u32, weight: i32) {
        self.transitions[from as usize].push(NwaTransition {
            target: to,
            tsid,
            weight,
        });
    }

    /// Set a state as accepting.
    pub fn set_accepting(&mut self, state: u32, accepting: bool) {
        self.accepting[state as usize] = accepting;
    }

    /// Number of transitions.
    pub fn num_transitions(&self) -> usize {
        self.transitions.iter().map(|t| t.len()).sum()
    }
}
