//! Automata module layout.
#![allow(unused_imports, unused_variables, dead_code)]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]

pub mod lexer;
pub mod u32;

pub use lexer::{dfa, nfa, regex};
pub use u32::weighted;
