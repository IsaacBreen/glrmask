//! Compiler: transforms grammars + vocabularies into compiled constraints.
//!
//! Pipeline: grammar IR → GLR table → tokenizer DFA → parser DWA → optimize → Constraint

pub mod glr;
pub mod grammar_def;
pub mod optimize;
pub mod parser_dwa;
pub mod pipeline;
pub mod template;
pub mod terminal_dwa;
pub mod tokenizer_dfa;
pub mod vocab_pre;
