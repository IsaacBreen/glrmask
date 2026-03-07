//! Lexer-side automata modules.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

pub mod ast;
pub mod compile;
pub mod dfa;
pub mod nfa;
pub mod tokenizer;

pub use ast as regex;
