#![allow(dead_code)]
//! State equivalence analysis — slow (reference) tier.
//!
//! Not yet implemented. The original sep1 reference implementation requires
//! trellis types (`TokenTrellisWithCompletion`) that are not available in the
//! glrmask port.
//!
//! When implemented, this should be a straightforward partition-refinement
//! over the full DFA transition table, suitable for cross-validation against
//! the fast implementation.
