//! Deterministic Weighted Automata (DWA) with i32 labels.
//!
//! This module provides weighted finite automata types used for grammar constraint
//! enforcement. The automata use i32 labels (for parser state IDs) and bitset weights.
//!
//! Main types:
//! - `DWA`: Deterministic Weighted Automaton
//! - `NWA`: Non-deterministic Weighted Automaton
//! - `Weight`: Bitset-based weight for tracking valid token sets

#![allow(dead_code)]

pub mod rangeset;
pub mod common;
pub mod determinization;
pub mod determinization_acyclic;
pub mod determinization_cyclic;
pub mod determinization_rustfst;
pub mod dwa;
pub mod json;
pub mod nwa;
pub mod minimization;
pub mod minimization_config;
pub mod unroll;
pub mod weight_expansion;
pub mod factored_weight;
pub mod shared_bdd;
pub mod heavy_weight;
pub mod reorder;
pub mod test_weighted_automata;

#[cfg(test)]
mod tests;

pub use self::rangeset::RangeSet;
pub use self::common::{format_i16_char, format_pos_code, format_word, NWAStateID, StateID, Weight, Label};
pub use self::dwa::{DWABody, DWABuildError, DWAState, DWAStates, DWA};
pub use self::nwa::{NWABody, NWABuildError, NWAState, NWAStates, NWA};
pub use self::heavy_weight::{HeavyWeight, WeightDimensions};
pub use self::minimization_config::{DeterminizeAndMinimizeProfile, DwaOptimizeConfig};
