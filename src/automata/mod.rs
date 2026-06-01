//! Finite automata used by grammar lowering, tokenizer scanning, and compiled weighted decoder constraints.

pub mod lexer;
pub mod weighted;
pub mod unweighted;

#[doc(hidden)] pub use weighted as weighted_u32;
#[doc(hidden)] pub use unweighted as unweighted_u32;

pub use lexer::{ast as regex, dfa};
