pub(crate) mod ast;
pub(crate) mod compile;
mod determinize;
pub(crate) mod dfa;

#[cfg(feature = "internal-api")]
pub use dfa::DFA;
mod lightweight;
mod minimize;
mod nfa;
pub(crate) mod tokenizer;
pub(crate) mod regex;
#[doc(hidden)]
pub mod runtime_unit_repeat;
#[doc(hidden)]
pub mod runtime_repeat_product;
pub(crate) mod runtime_residual;

pub use tokenizer::Lexer;
