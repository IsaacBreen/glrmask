#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

use thiserror::Error as ThisError;

#[derive(ThisError, Debug)]
pub enum Error {
    #[error("Grammar parse error: {0}")]
    GrammarParse(String),
}

pub type GlrMaskError = Error;
pub type Result<T> = std::result::Result<T, Error>;

pub mod automata {
    pub use glrmask_finite_automata::automata::unweighted_u32;
    pub use glrmask_lexer::automata::lexer;
    pub use glrmask_lexer::automata::regex;
}

pub mod ds {
    pub use glrmask_lexer::ds::u8set;
}

pub mod grammar;

pub mod import {
    pub use crate::grammar::ast;
    pub mod ebnf;
    pub mod lark;

    pub fn choice_or_single(mut options: Vec<ast::GrammarExpr>) -> ast::GrammarExpr {
        if options.len() == 1 {
            options.pop().unwrap()
        } else {
            ast::GrammarExpr::Choice(options)
        }
    }

    pub fn sequence_or_single(mut items: Vec<ast::GrammarExpr>) -> ast::GrammarExpr {
        match items.len() {
            0 => ast::GrammarExpr::Sequence(Vec::new()),
            1 => items.pop().unwrap(),
            _ => ast::GrammarExpr::Sequence(items),
        }
    }
}
