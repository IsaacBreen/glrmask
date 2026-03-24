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

use crate::compiler::debug::{CompileDebug, CompileDiagnostics};
use crate::compiler::{compile_owned, compile_with_diagnostics};
use crate::grammar::flat::GrammarDef;
use crate::runtime::Constraint;

type GrammarParser = fn(&str) -> crate::Result<GrammarDef>;

fn compile_from_source(
    source: &str,
    vocab: &crate::Vocab,
    parse: GrammarParser,
) -> crate::Result<Constraint> {
    let grammar = parse(source)?;
    Ok(compile_owned(grammar, vocab))
}

fn compile_from_source_with_diagnostics(
    source: &str,
    vocab: &crate::Vocab,
    parse: GrammarParser,
) -> crate::Result<(Constraint, CompileDiagnostics)> {
    let grammar = parse(source)?;
    Ok(compile_with_diagnostics(&grammar, vocab))
}

impl Constraint {
    pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(ebnf, vocab, ebnf::parse_ebnf)
    }

    pub fn from_ebnf_with_diagnostics(
        ebnf: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDiagnostics)> {
        compile_from_source_with_diagnostics(ebnf, vocab, ebnf::parse_ebnf)
    }

    pub fn from_ebnf_with_debug(
        ebnf: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDebug)> {
        Self::from_ebnf_with_diagnostics(ebnf, vocab)
    }

    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(lark, vocab, lark::parse_lark)
    }

    pub(crate) fn from_lark_with_diagnostics(
        lark: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDiagnostics)> {
        compile_from_source_with_diagnostics(lark, vocab, lark::parse_lark)
    }

    pub(crate) fn from_lark_with_debug(
        lark: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDebug)> {
        Self::from_lark_with_diagnostics(lark, vocab)
    }

    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(schema, vocab, json_schema::json_schema_to_grammar)
    }

    pub(crate) fn from_json_schema_with_diagnostics(
        schema: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDiagnostics)> {
        compile_from_source_with_diagnostics(schema, vocab, json_schema::json_schema_to_grammar)
    }

    pub(crate) fn from_json_schema_with_debug(
        schema: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDebug)> {
        Self::from_json_schema_with_diagnostics(schema, vocab)
    }
}
