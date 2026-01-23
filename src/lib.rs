//! Grammar-Constrained Generation (GCG) Library
//!
//! This library provides efficient grammar-constrained decoding for language models.
//!
//! # Module Organization
//!
//! ## Automata Modules
//! - [`dfa_u8`]: Tokenizer DFA/NFA with u8 (byte) labels
//! - [`dfa_i32`]: Unweighted DFA/NFA with i32 (parser state) labels  
//! - [`dwa_i32`]: Weighted DWA/NWA with i32 labels and bitset weights
//!
//! ## Core Modules
//! - [`constraint`]: Grammar constraint implementation and state management
//! - [`precompute4`]: Parser DWA construction from grammar
//! - [`constraint_precompute`]: Terminal DWA construction from tokenizer
//!
//! ## Supporting Modules
//! - [`interface`]: Grammar parsing (EBNF, JSON Schema)
//! - [`glr`]: GLR parser for grammar analysis
//! - [`datastructures`]: Efficient data structures (bitsets, GSS, etc.)

#![allow(warnings)]

#[cfg(test)]
pub static GLOBAL_DIMS_MUTEX: once_cell::sync::Lazy<std::sync::Mutex<()>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(()));

pub mod dfa_u8;
/// Backward-compatibility re-export from dfa_u8::dfa
pub mod finite_automata;
pub mod equivalence_analysis;
pub mod glr;
pub mod constraint;
pub mod datastructures;
pub mod interface;
pub mod r#macro;
mod types;
pub mod json_serialization;
/// Backward-compatibility re-export from interface::json_schema
pub mod json_schema {
    pub use crate::interface::json_schema::*;
}
mod test_constraint_basic;
// mod test_constraint_python;
mod profiler;

// Automata modules
pub mod dwa_i32;
pub mod dfa_i32;

// Parser DWA construction
pub mod precompute4;
pub mod constraint_fns;
pub mod bruteforce_constraint;

pub mod constraint_vocab;
pub mod constraint_precompute;
mod test_finite_automata;
mod fill_benchmark;
mod test_equivalence_analysis;

// Pipeline module for staged constraint building
pub mod pipeline;
