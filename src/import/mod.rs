#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

pub use crate::grammar::ast as ast;
pub mod ebnf;
pub mod json_schema;
pub mod lark;
pub mod numeric_range;

#[cfg(test)]
mod test_grammar_import;

#[cfg(test)]
mod test_json_schema;

pub use crate::grammar::ast as grammar_expr;

use crate::compiler::debug::CompileDebug;
use crate::compiler::{compile, compile_with_debug};
use crate::runtime::Constraint;

fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Constraint> {
    let grammar = ebnf::parse_ebnf(ebnf)?;
    Ok(compile(&grammar, vocab))
}

fn from_ebnf_with_debug(
    ebnf: &str,
    vocab: &crate::Vocab,
) -> crate::Result<(Constraint, CompileDebug)> {
    let grammar = ebnf::parse_ebnf(ebnf)?;
    Ok(compile_with_debug(&grammar, vocab))
}

fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Constraint> {
    let grammar = lark::parse_lark(lark)?;
    Ok(compile(&grammar, vocab))
}

fn from_lark_with_debug(
    lark: &str,
    vocab: &crate::Vocab,
) -> crate::Result<(Constraint, CompileDebug)> {
    let grammar = lark::parse_lark(lark)?;
    Ok(compile_with_debug(&grammar, vocab))
}

fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Constraint> {
    let grammar = json_schema::json_schema_to_grammar(schema)?;
    Ok(compile(&grammar, vocab))
}

fn from_json_schema_with_debug(
    schema: &str,
    vocab: &crate::Vocab,
) -> crate::Result<(Constraint, CompileDebug)> {
    let grammar = json_schema::json_schema_to_grammar(schema)?;
    Ok(compile_with_debug(&grammar, vocab))
}

impl Constraint {
    pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        if std::env::var("GLRMASK_COMPILE_DEBUG").is_ok() {
            let (constraint, debug) = from_ebnf_with_debug(ebnf, vocab)?;
            eprintln!("{}", debug);
            return Ok(constraint);
        }
        from_ebnf(ebnf, vocab)
    }

    pub fn from_ebnf_with_debug(
        ebnf: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDebug)> {
        from_ebnf_with_debug(ebnf, vocab)
    }

    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        if std::env::var("GLRMASK_COMPILE_DEBUG").is_ok() {
            let (constraint, debug) = from_lark_with_debug(lark, vocab)?;
            eprintln!("{}", debug);
            return Ok(constraint);
        }
        from_lark(lark, vocab)
    }

    pub(crate) fn from_lark_with_debug(
        lark: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDebug)> {
        from_lark_with_debug(lark, vocab)
    }

    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        if std::env::var("GLRMASK_COMPILE_DEBUG").is_ok() {
            let (constraint, debug) = from_json_schema_with_debug(schema, vocab)?;
            eprintln!("{}", debug);
            return Ok(constraint);
        }
        from_json_schema(schema, vocab)
    }

    pub(crate) fn from_json_schema_with_debug(
        schema: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDebug)> {
        from_json_schema_with_debug(schema, vocab)
    }
}
