//! State equivalence analysis implementations.
//!
//! - `max_length`: bounded-depth path-hash prepass using only the maximum token length
//! - `fast`: token-based refinement on the surviving representative states

pub mod fast;
pub mod max_length;
