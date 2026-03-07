//! Compiler: transforms grammars + vocabularies into compiled constraints.
//!
//! Pipeline: grammar IR → GLR table → tokenizer DFA → parser DWA → optimize → Constraint
#![allow(unused_imports, unused_variables, dead_code)]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]

pub mod debug;
pub mod glr;
pub mod grammar_def;
pub mod labels;
pub mod optimize;
pub mod parser_dwa;
pub mod pipeline;
pub mod resolve_negatives;
pub mod template;
pub mod terminal_dwa;
pub mod tokenizer_dfa;
pub mod vocab_pre;
