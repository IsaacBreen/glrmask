//! Constraint and ConstraintState — the main runtime types.

use serde::{Deserialize, Serialize};

use crate::automata::weighted::dwa::Dwa;
use crate::ds::bitset::BitSet;
use crate::GlrMaskError;

/// A compiled grammar constraint, ready for inference.
///
/// Immutable after creation. Thread-safe (`Send + Sync`).
/// Create [`ConstraintState`] instances from this to track per-sequence state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// The compiled DWA.
    pub(crate) dwa: Dwa,
    /// Token-to-TSID mapping.
    pub(crate) vocab_mapping: Vec<u32>,
    /// Number of tokens in the vocabulary.
    pub(crate) vocab_size: usize,
    /// EOS token ID, if any.
    pub(crate) eos_token_id: Option<u32>,
}

impl Constraint {
    /// Number of DWA states.
    pub fn num_states(&self) -> u32 {
        self.dwa.num_states()
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Create a new `ConstraintState` at the start position.
    pub fn start(&self) -> ConstraintState {
        ConstraintState {
            state: self.dwa.start_state,
        }
    }

    /// Serialize to bytes (bincode).
    pub fn to_bytes(&self) -> Result<Vec<u8>, GlrMaskError> {
        bincode::serialize(self).map_err(|e| GlrMaskError::Serialization(e.to_string()))
    }

    /// Deserialize from bytes (bincode).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, GlrMaskError> {
        bincode::deserialize(bytes).map_err(|e| GlrMaskError::Serialization(e.to_string()))
    }
}

/// Per-sequence constraint state.
///
/// Lightweight (just a DWA state ID). Computes token masks and advances
/// state when tokens are committed.
#[derive(Debug, Clone)]
pub struct ConstraintState {
    /// Current DWA state.
    state: u32,
}

impl ConstraintState {
    /// Get the current DWA state.
    pub fn current_state(&self) -> u32 {
        self.state
    }

    /// Commit a token: advance the DWA state.
    ///
    /// Returns the weight (>= 0 means the token was allowed).
    pub fn commit(&mut self, constraint: &Constraint, token_id: u32) -> i32 {
        let tsid = constraint.vocab_mapping[token_id as usize];
        let (next_state, weight) = constraint.dwa.step(self.state, tsid);
        self.state = next_state;
        weight
    }

    /// Compute the allowed-token mask for the current state.
    ///
    /// Returns a `BitSet` where bit `i` is set iff token `i` is allowed.
    pub fn compute_mask(&self, constraint: &Constraint) -> BitSet {
        super::mask::compute_mask(&constraint.dwa, self.state, &constraint.vocab_mapping, constraint.vocab_size)
    }

    /// Compute the forced byte prefix (if any).
    ///
    /// When only a single token (or a set of tokens with a common byte prefix)
    /// is allowed, returns that prefix. Used for speculative decoding.
    pub fn forced_prefix(&self, constraint: &Constraint) -> Option<Vec<u8>> {
        super::force::forced_prefix(&constraint.dwa, self.state, &constraint.vocab_mapping, constraint.vocab_size)
    }

    /// Whether the current state is accepting (valid end-of-sequence).
    pub fn is_accepting(&self, constraint: &Constraint) -> bool {
        constraint.dwa.is_accepting(self.state)
    }
}
