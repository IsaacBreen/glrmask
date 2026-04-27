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
use crate::grammar::factoring::factor_named_grammar;
use crate::runtime::Constraint;

type GrammarParser = fn(&str) -> crate::Result<GrammarDef>;
type NamedGrammarParser = fn(&str) -> crate::Result<ast::NamedGrammar>;

fn env_var_is_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

fn maybe_print_grammar_glrm(source_kind: &str, grammar: &ast::NamedGrammar) {
    if !env_var_is_truthy("GLRMASK_PRINT_GRAMMAR_GLRM") {
        return;
    }

    let printable = grammar.prune_unreachable();
    eprintln!(
        "[glrmask/grammar][{source_kind}]\n{}",
        crate::grammar::glrm::to_glrm(&printable)
    );
}

fn lower_factored_named_grammar(
    source: &str,
    source_kind: &str,
    parse_named: NamedGrammarParser,
) -> crate::Result<GrammarDef> {
    let debug_import = env_var_is_truthy("GLRMASK_DEBUG_IMPORT_TIMES");
    let phase_started_at = std::time::Instant::now();
    let named = parse_named(source)?;
    let parse_ms = phase_started_at.elapsed().as_secs_f64() * 1000.0;

    let factor_started_at = std::time::Instant::now();
    let factored = factor_named_grammar(named);
    let factor_ms = factor_started_at.elapsed().as_secs_f64() * 1000.0;

    maybe_print_grammar_glrm(source_kind, &factored);

    let lower_started_at = std::time::Instant::now();
    let lowered = ast::lower(&factored)?;
    let lower_ms = lower_started_at.elapsed().as_secs_f64() * 1000.0;

    if debug_import {
        eprintln!(
            "[glrmask/debug][import] source={} parse_ms={:.3} factor_ms={:.3} lower_ms={:.3}",
            source_kind,
            parse_ms,
            factor_ms,
            lower_ms,
        );
    }

    Ok(lowered)
}

fn compile_from_source(
    source: &str,
    vocab: &crate::Vocab,
    source_kind: &str,
    parse: NamedGrammarParser,
) -> crate::Result<Constraint> {
    if compile_profile_enabled() {
        let parse_started_at = std::time::Instant::now();
        let grammar = lower_factored_named_grammar(source, source_kind, parse)?;
        let import_ms = parse_started_at.elapsed().as_secs_f64() * 1000.0;
        let (constraint, profile) = compile_owned_profiled(grammar, vocab);
        emit_compile_profile_summary(Some(source_kind), Some(import_ms), &profile);
        return Ok(constraint);
    }

    let grammar = lower_factored_named_grammar(source, source_kind, parse)?;
    Ok(compile_owned(grammar, vocab))
}

fn glrm_to_grammar_def(source: &str) -> crate::Result<GrammarDef> {
    let named = crate::grammar::glrm::from_glrm(source)?;
    let factored = factor_named_grammar(named);
    ast::lower(&factored)
}

fn parse_json_schema_to_named(schema_json: &str) -> crate::Result<ast::NamedGrammar> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| crate::GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    json_schema::schema_to_named_grammar(&schema)
}

impl Constraint {
    pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(ebnf, vocab, "ebnf", ebnf::parse_ebnf_to_named)
    }

    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(lark, vocab, "lark", lark::parse_lark_to_named)
    }

    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(schema, vocab, "json_schema", parse_json_schema_to_named)
    }

    /// Load a grammar from the GLRM format (see [`crate::grammar::glrm`]).
    pub fn from_glrm_grammar(glrm: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(glrm, vocab, "glrm", crate::grammar::glrm::from_glrm)
    }
}
