//! L1 path: terminals with max path length ≤ 1.
//!
//! L1 terminals never co-occur with another terminal within a single vocab token.
//! This allows simplified equivalence analysis (max_length only) and a fast
//! direct terminal DWA construction path.

pub mod equivalence_analysis;
pub mod terminal_dwa;
