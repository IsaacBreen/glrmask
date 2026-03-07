//! External-spec ingestion for compiled constraints.
//!
//! This module owns constructor-side responsibility for turning user-facing
//! grammar/spec inputs into compiled runtime artifacts. `runtime` executes a
//! compiled `Constraint`; `import` owns how specs are parsed and compiled.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

pub mod ast;
pub mod ebnf;
pub mod json_schema;
pub mod lark;

pub use ast as grammar_expr;

use crate::compiler::debug::CompileDebug;
use crate::runtime::Constraint;

/// Compile a `Constraint` directly from EBNF input.
fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Constraint> {
    let _ = (ebnf, vocab);
    unimplemented!()
}

/// Compile a `Constraint` from EBNF input and return a debug bundle.
fn from_ebnf_with_debug(
    ebnf: &str,
    vocab: &crate::Vocab,
) -> crate::Result<(Constraint, CompileDebug)> {
    let _ = (ebnf, vocab);
    unimplemented!()
}

/// Compile a `Constraint` directly from Lark input.
fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Constraint> {
    let _ = (lark, vocab);
    unimplemented!()
}

/// Compile a `Constraint` from Lark input and return a debug bundle.
fn from_lark_with_debug(
    lark: &str,
    vocab: &crate::Vocab,
) -> crate::Result<(Constraint, CompileDebug)> {
    let _ = (lark, vocab);
    unimplemented!()
}

/// Compile a `Constraint` directly from JSON Schema input.
fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Constraint> {
    let _ = (schema, vocab);
    unimplemented!()
}

/// Compile a `Constraint` from JSON Schema input and return a debug bundle.
fn from_json_schema_with_debug(
    schema: &str,
    vocab: &crate::Vocab,
) -> crate::Result<(Constraint, CompileDebug)> {
    let _ = (schema, vocab);
    unimplemented!()
}

impl Constraint {
    /// Compile a constraint from an EBNF grammar string.
    pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        from_ebnf(ebnf, vocab)
    }

    /// Compile a constraint from an EBNF grammar string, returning a
    /// [`CompileDebug`](crate::compiler::debug::CompileDebug) bundle
    /// alongside the constraint.
    pub(crate) fn from_ebnf_with_debug(
        ebnf: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDebug)> {
        from_ebnf_with_debug(ebnf, vocab)
    }

    /// Compile a constraint from a Lark grammar string.
    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        from_lark(lark, vocab)
    }

    /// Compile a constraint from a Lark grammar string, returning a debug bundle.
    pub(crate) fn from_lark_with_debug(
        lark: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDebug)> {
        from_lark_with_debug(lark, vocab)
    }

    /// Compile a constraint from a JSON Schema string.
    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        from_json_schema(schema, vocab)
    }

    /// Compile a constraint from a JSON Schema string, returning a debug bundle.
    pub(crate) fn from_json_schema_with_debug(
        schema: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDebug)> {
        from_json_schema_with_debug(schema, vocab)
    }
}
