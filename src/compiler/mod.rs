//! Compiler: transforms grammars + vocabularies into compiled constraints.
//!
//! Pipeline: grammar IR → GLR table → tokenizer DFA → parser DWA → optimize → Constraint
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

pub mod debug;
pub mod glr;
pub mod grammar_def;
pub mod labels;
pub mod pipeline;
pub mod tokenizer_dfa;
pub mod vocab_pre;

pub use pipeline::a_terminal_dwa as terminal_dwa;
pub use pipeline::b_template_dfa as template;
pub use pipeline::c_parser_dwa as parser_dwa;
pub use pipeline::d_resolve_negatives as resolve_negatives;
