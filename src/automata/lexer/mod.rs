
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

// SEP1_MAP: This submodule is the closest glrmask counterpart to sep1's byte-labeled automata in `dfa_u8/` plus the regex expression surface in `finite_automata.rs`.

pub mod ast;
pub mod compile;
pub mod determinize;
pub mod dfa;
pub mod minimize;
pub mod nfa;
pub mod tokenizer;
pub mod tokenizer_regex;

pub use ast as regex;
