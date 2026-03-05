//! Deterministic Weighted Automaton (DWA).
//!
//! The DWA is the final compiled form used at inference time.
//! Each state + token-set ID maps to a (next_state, weight) pair.

use serde::{Deserialize, Serialize};

use super::weight::WeightTable;

/// A Deterministic Weighted Automaton operating over token-set IDs.
///
/// At each step, given the current state and a token-set ID (TSID),
/// the DWA produces a next state and an integer weight. The weight
/// is used to determine whether the token is allowed (weight >= 0 means allowed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dwa {
    /// The weight/transition table.
    pub weights: WeightTable,
    /// The start state.
    pub start_state: u32,
    /// Which states are accepting (valid end-of-sequence states).
    pub accepting: Vec<bool>,
}

impl Dwa {
    /// Create a new DWA.
    pub fn new(weights: WeightTable, start_state: u32, accepting: Vec<bool>) -> Self {
        Self {
            weights,
            start_state,
            accepting,
        }
    }

    /// Number of states.
    pub fn num_states(&self) -> u32 {
        self.weights.num_states
    }

    /// Number of token-set IDs.
    pub fn num_tsids(&self) -> u32 {
        self.weights.num_tsids
    }

    /// Get the transition for `(state, tsid)`.
    #[inline]
    pub fn step(&self, state: u32, tsid: u32) -> (u32, i32) {
        self.weights.get(tsid, state)
    }

    /// Whether a state is accepting.
    pub fn is_accepting(&self, state: u32) -> bool {
        self.accepting
            .get(state as usize)
            .copied()
            .unwrap_or(false)
    }
}
