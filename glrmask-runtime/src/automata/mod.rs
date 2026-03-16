#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

pub mod lexer;
pub mod weighted_u32;

pub use lexer::{ast as regex, dfa, tokenizer};
pub use weighted_u32 as weighted;
