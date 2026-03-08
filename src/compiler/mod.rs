#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]


pub mod compile;
pub mod debug;
pub mod glr;
pub mod grammar;
pub(crate) mod possible_matches;
pub mod stages;

pub use crate::automata::lexer::tokenizer as tokenizer_dfa;
pub use compile::compile;
pub(crate) use compile::compile_with_debug;
pub use grammar::model as grammar_def;
pub use glr::labels as parser_labels;
pub use stages::equivalence_analysis;

pub use stages::parser_dwa;
pub use stages::resolve_negatives;
pub use stages::templates::compile_dfa as template;
pub use stages::terminal_dwa;
