#![allow(warnings)]
pub mod finite_automata;
pub mod equivalence_analysis_finite_automata;
pub mod glr;
pub mod constraint;
mod constraint_extra;
pub mod datastructures;
pub mod interface;
mod r#macro;
pub mod tokenizer;
mod types;
mod tests_apr25;
mod multi_dfa;
pub mod json_serialization;
mod test_constraint_basic;
// mod test_constraint_python;
mod profiler;
mod test_constraint_js;
mod test_precompute_optimizations;
mod constraint_precompute2_utils;
mod constraint_precompute1_utils;
mod constraint_precompute0_utils;
mod constraint_stored_cache_utils;
mod constraint_precompute3_challenge_elimination;
mod constraint_precompute3_intermediate_utils;
mod constraint_special_precompute;

// New lightweight pass framework for Trie3 optimization
pub mod trie3_opt;
pub mod weighted_automata;
pub mod precompute4;
