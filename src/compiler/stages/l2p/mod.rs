//! L2+ path: terminals with max path length ≥ 2.
//!
//! L2+ terminals can co-occur with other terminals within a single vocab token.
//! This requires full NWA-based terminal DWA construction and vocab equivalence
//! analysis with DFA-based signature computation.

pub mod equivalence_analysis;
pub mod terminal_dwa;
