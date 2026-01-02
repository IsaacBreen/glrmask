#![allow(warnings)]
pub mod tokenizer;
/// Backward-compatibility re-export from tokenizer::dfa
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

// New lightweight pass framework for Trie3 optimization
pub mod precompute4;
pub mod constraint_fns;
pub mod bruteforce_constraint;

pub mod constraint_vocab;
pub mod constraint_precompute;
mod test_finite_automata;
mod fill_benchmark;
mod test_json;
mod test_equivalence_analysis;

// Pipeline module for staged constraint building
pub mod pipeline;
