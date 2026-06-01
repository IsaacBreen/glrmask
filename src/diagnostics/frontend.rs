//! Frontend and compile diagnostics.
//!
//! These helpers expose intermediate representations for tests, benchmarks, and
//! paper/debugging workflows.  They are deliberately outside the main facade:
//! normal users should call `Constraint::from_*` constructors instead.

use crate::api::{Constraint, GlrMaskError, Result, Vocab};

/// Compile a [`Constraint`] from a serialized internal `GrammarDef` JSON string.
///
/// This bypasses the source-language frontends and runs the full compile
/// pipeline directly on the lowered grammar IR.
pub fn compile_grammar_def_json(grammar_def_json: &str, vocab: &Vocab) -> Result<Constraint> {
    let gdef: crate::grammar::flat::GrammarDef = serde_json::from_str(grammar_def_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid GrammarDef JSON: {e}")))?;
    Ok(crate::compile::pipeline::compile_owned(gdef, vocab))
}

/// Populate compile-time artifacts that are pure functions of the vocabulary.
///
/// This intentionally does not compile any grammar/schema-dependent artifact.
pub fn prepare_vocab_for_compile(vocab: &Vocab) {
    crate::compiler::compile::prepare_vocab_for_compile(vocab);
}

/// Dump the imported JSON Schema grammar in GLRM format after frontend lowering.
///
/// The output is useful for comparing JSON Schema lowering against the grammar
/// model used by the paper and by the other source-language frontends.
pub fn dump_json_schema_grammar_glrm(schema_json: &str) -> Result<String> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    let named = crate::import::json_schema::schema_to_named_grammar(&schema)?;
    let mut factored = crate::grammar::factoring::factor_named_grammar(named);
    if crate::import::json_schema::simplify_grammar_enabled() {
        crate::grammar::named_simplify::simplify_named_grammar(&mut factored);
    }
    if crate::import::json_schema::lower_exact_subtractions_enabled() {
        crate::grammar::exact_subtraction_lowering::lower_exact_subtractions(&mut factored)?;
    }
    if crate::import::json_schema::promote_literal_choices_enabled() {
        crate::grammar::terminal_choice_promotion::promote_choice_terminals_exact(&mut factored, false);
    }
    Ok(crate::grammar::glrm::to_glrm(&factored))
}
