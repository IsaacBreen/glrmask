//! Lexer-side automata modules.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

pub mod ast;
pub mod compile;
pub mod determinize;
pub mod dfa;
pub mod minimize;
pub mod nfa;
pub mod tokenizer;
pub mod tokenizer_regex;

pub use ast as regex;
