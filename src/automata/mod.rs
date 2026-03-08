#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

// SEP1_MAP: This automata umbrella corresponds to sep1's `dfa_u8`, `dfa_i32`, `dwa_i32`, and `finite_automata` modules, regrouped under one crate-local tree.

pub mod lexer;
pub mod weighted_u32;
pub mod unweighted_u32;

pub use lexer::{ast as regex, compile, dfa, nfa, tokenizer};
pub use unweighted_u32 as unweighted;
pub use weighted_u32 as weighted;
