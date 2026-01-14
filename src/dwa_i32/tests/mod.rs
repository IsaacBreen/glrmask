//! Tests for the DWA (Deterministic Weighted Automaton) module.

// Re-export types from dwa_i32 for convenience in tests
pub use crate::dwa_i32::{
    DWA, DWABody, DWABuildError, DWAState, DWAStates,
    NWA, NWABuildError, NWAState, NWAStates,
    Weight, StateID, RangeSet, format_word,
};
pub use crate::dwa_i32::common::Label;

mod test_debug_min;
mod test_determinization;
mod test_minimization;
mod test_minimization_failure;
mod test_nwa_pipeline;
mod test_push;
mod test_repro;
mod test_rm_epsilon_effect;
mod test_weight_loosening;
mod test_weight_storage;
