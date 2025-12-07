#![allow(warnings)]
pub mod finite_automata;
mod equivalence_analysis_reference;
mod equivalence_analysis_fast;
mod equivalence_analysis_fast_new;
mod equivalence_analysis;
pub mod glr;
pub mod constraint;
pub mod datastructures;
pub mod interface;
pub mod r#macro;
pub mod tokenizer;
mod types;
pub mod json_serialization;
pub mod json_schema;
mod test_constraint_basic;
// mod test_constraint_python;
mod profiler;

// New lightweight pass framework for Trie3 optimization
pub mod precompute4;
mod constraint_fns;
mod state_equivalence_analysis_finite_automata;

pub mod constraint_vocab;
pub mod constraint_precompute;
mod test_finite_automata;
mod fill_benchmark;

// Pipeline module for staged constraint building
pub mod pipeline;
