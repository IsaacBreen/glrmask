#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub type GlrMaskError = glrmask_grammar::Error;
pub type Result<T> = std::result::Result<T, GlrMaskError>;

pub mod grammar {
    pub use glrmask_grammar::grammar::*;
}

pub mod automata {
    pub use glrmask_lexer::automata::lexer;
    pub use glrmask_lexer::automata::regex;
}

pub mod import {
    pub use glrmask_grammar::grammar::ast;
    pub mod numeric_range;
}

pub mod json_schema;
