//! Minimization passes for DWA and NWA.
//!
//! This module provides optimization passes for weighted automata:
//! - Pruning unreachable and dead-end states
//! - Weight pushing (toward start/final)
//! - State minimization via partition refinement

pub mod common;
pub mod dwa;
pub mod nwa;

pub use common::{Partition, MAX_OPTIMIZE_ITERATIONS};
pub use dwa::DwaPass;
pub use nwa::NwaPass;
