//! Runtime: inference-time constraint state machine.
//!
//! This module contains the hot path: `ConstraintState` processes tokens
//! and computes allowed-token masks in microseconds.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

pub(crate) mod force;
pub(crate) mod gss_acc;
pub(crate) mod leveled_gss;
mod mask;
mod state;

// Re-export the main types
pub use state::{Constraint, ConstraintState};
