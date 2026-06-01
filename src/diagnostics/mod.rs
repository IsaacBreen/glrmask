//! Diagnostics, cache management, and compatibility helpers.
//!
//! Nothing in this module is needed for the mathematical mask/commit API.  These
//! functions exist for benchmark harnesses, profiling scripts, cache hygiene, and
//! inspection of frontend lowering.  Keeping them here prevents implementation
//! levers from looking like core decoding concepts.

pub mod cache;
pub mod frontend;
pub(crate) mod logging;
