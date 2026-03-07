//! Runtime: inference-time constraint state machine.
//!
//! This module contains the hot path: `ConstraintState` processes tokens
//! and computes allowed-token masks in microseconds.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

mod actions;
mod constraint;
mod debug;
mod serde;
mod state;

// Re-export the main types
pub use constraint::Constraint;
pub use state::ConstraintState;
