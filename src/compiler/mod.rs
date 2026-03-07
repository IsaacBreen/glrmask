//! Compiler: transforms grammars + vocabularies into compiled constraints.
//!
//! Pipeline: grammar IR → GLR table → tokenizer DFA → parser DWA → optimize → Constraint
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

pub mod debug;
pub mod glr;
pub mod grammar;
pub mod import;
pub mod labels;
pub mod stages;

pub use crate::automata::lexer::tokenizer as tokenizer_dfa;
pub use grammar::ast as grammar_def;
pub use stages::id_map;

pub use stages::parser_dwa;
pub use stages::resolve_negatives;
pub use stages::template_dfa as template;
pub use stages::terminal_dwa;
