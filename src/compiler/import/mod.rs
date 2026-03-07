//! External-spec ingestion for compiled constraints.
//!
//! This module owns constructor-side responsibility for turning user-facing
//! grammar/spec inputs into compiled runtime artifacts. `runtime` executes a
//! compiled `Constraint`; `import` owns how specs are parsed and compiled.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::compiler::debug::CompileDebug;
use crate::runtime::Constraint;

/// Compile a `Constraint` directly from EBNF input.
pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Constraint> {
    let _ = (ebnf, vocab);
    unimplemented!()
}

/// Compile a `Constraint` from EBNF input and return a debug bundle.
pub fn from_ebnf_with_debug(
    ebnf: &str,
    vocab: &crate::Vocab,
) -> crate::Result<(Constraint, CompileDebug)> {
    let _ = (ebnf, vocab);
    unimplemented!()
}

/// Compile a `Constraint` directly from Lark input.
pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Constraint> {
    let _ = (lark, vocab);
    unimplemented!()
}

/// Compile a `Constraint` directly from JSON Schema input.
pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Constraint> {
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
    pub fn from_ebnf_with_debug(
        ebnf: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDebug)> {
        from_ebnf_with_debug(ebnf, vocab)
    }

    /// Compile a constraint from a Lark grammar string.
    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        from_lark(lark, vocab)
    }

    /// Compile a constraint from a JSON Schema string.
    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        from_json_schema(schema, vocab)
    }
}
