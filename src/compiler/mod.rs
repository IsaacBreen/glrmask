//! Compiler: transforms grammars + vocabularies into compiled constraints.
//!
//! Pipeline: grammar IR → GLR table → tokenizer DFA → parser DWA → optimize → Constraint
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

pub mod debug;
pub mod glr;
pub mod grammar;
pub mod labels;
pub mod pipeline;

pub use crate::automata::lexer::tokenizer as tokenizer_dfa;
pub use grammar::ast as grammar_def;
pub use pipeline::vocab_pre;

pub use pipeline::parser_dwa;
pub use pipeline::resolve_negatives;
pub use pipeline::template_dfa as template;
pub use pipeline::terminal_dwa;
