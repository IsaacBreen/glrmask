//! Automata module layout.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

pub mod lexer;
pub mod u32;

pub use lexer::{dfa, nfa, regex};
pub use u32::weighted;
