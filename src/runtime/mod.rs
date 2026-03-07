//! Runtime: inference-time constraint state machine.
//!
//! This module contains the hot path: `ConstraintState` processes tokens
//! and computes allowed-token masks in microseconds.
#![allow(unused_imports, unused_variables, dead_code)]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]

pub(crate) mod force;
mod gss;
pub(crate) mod gss_acc;
pub(crate) mod leveled_gss;
mod mask;
mod state;

// Re-export the main types
pub use state::{Constraint, ConstraintState};
