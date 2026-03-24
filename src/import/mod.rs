pub use crate::grammar::ast as ast;
pub mod ebnf;
pub mod json_schema;
pub mod lark;
pub mod numeric_range;

#[cfg(test)]
mod test_grammar_import;

#[cfg(test)]
mod test_json_schema;

use crate::compiler::compile::{compile_owned_profiled, compile_profile_enabled, emit_compile_profile_summary};
use crate::compiler::compile_owned;
use crate::grammar::flat::GrammarDef;
use crate::runtime::Constraint;

type GrammarParser = fn(&str) -> crate::Result<GrammarDef>;

fn compile_from_source(
    source: &str,
    vocab: &crate::Vocab,
    source_kind: &str,
    parse: GrammarParser,
) -> crate::Result<Constraint> {
    if compile_profile_enabled() {
        let parse_started_at = std::time::Instant::now();
        let grammar = parse(source)?;
        let import_ms = parse_started_at.elapsed().as_secs_f64() * 1000.0;
        let (constraint, profile) = compile_owned_profiled(grammar, vocab);
        emit_compile_profile_summary(Some(source_kind), Some(import_ms), &profile);
        return Ok(constraint);
    }

    let grammar = parse(source)?;
    Ok(compile_owned(grammar, vocab))
}

impl Constraint {
    pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(ebnf, vocab, "ebnf", ebnf::parse_ebnf)
    }

    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(lark, vocab, "lark", lark::parse_lark)
    }

    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(schema, vocab, "json_schema", json_schema::json_schema_to_grammar)
    }
}
