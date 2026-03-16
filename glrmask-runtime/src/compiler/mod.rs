#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

pub mod glr;

pub use crate::grammar::flat as grammar_def;
pub use glr::labels as parser_labels;
