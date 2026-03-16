#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

pub mod labels;
pub mod parser;
pub mod table;

pub(crate) const EOF: crate::compiler::grammar_def::TerminalID = u32::MAX;
