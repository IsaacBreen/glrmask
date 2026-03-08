#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

// SEP1_MAP: This module matches sep1's template-DFA surface in `precompute4/`, but glrmask splits characterization, DFA compilation, and bundle assembly into separate files.

pub mod characterize;
pub mod compile_bundle;
pub mod compile_dfa;

pub use compile_dfa::Templates;
