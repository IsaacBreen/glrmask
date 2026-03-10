//! State equivalence analysis implementations.
//!
//! - `slow`: Reference implementation (not yet implemented — placeholder)
//! - `medium`: Intermediate fidelity (not yet implemented — placeholder)
//! - `fast`: K-step hash mixing + token-based refinement (production runtime default)

pub mod slow;
pub mod medium;
pub mod fast;
