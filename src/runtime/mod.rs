//! Runtime: inference-time constraint state machine.
//!
//! This module contains the hot path: `ConstraintState` processes tokens
//! and computes allowed-token masks in microseconds.

pub mod force;
mod gss;
pub mod gss_acc;
pub mod leveled_gss;
mod mask;
mod state;

// Re-export the main types
pub use state::{Constraint, ConstraintState};
