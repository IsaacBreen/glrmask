#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub(crate) mod automata;
pub(crate) mod compiler;
pub(crate) mod ds;
mod error;
pub(crate) mod grammar;
pub(crate) mod import;
pub(crate) mod runtime;
mod vocab;

pub use runtime::{Constraint, ConstraintState};
pub use vocab::Vocab;

pub(crate) use error::{GlrMaskError, Result};

/// Compile a Constraint from a serialized GrammarDef JSON + vocab.
/// This runs the full compile pipeline (equivalence analysis, terminal DWA, parser DWA).
pub(crate) fn compile_grammar_def_json(grammar_def_json: &str, vocab: &Vocab) -> Result<Constraint> {
    let gdef: grammar::flat::GrammarDef = serde_json::from_str(grammar_def_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid GrammarDef JSON: {e}")))?;
    Ok(compiler::stages::id_map_and_terminal_dwa::l2p::with_ti_pool(|| {
        compiler::compile_owned(gdef, vocab)
    }))
}

/// Populate compile-time artifacts that are pure functions of the vocabulary.
///
/// This intentionally does not compile any grammar/schema-dependent artifact.
pub(crate) fn prepare_vocab_for_compile(vocab: &Vocab) {
    compiler::compile::prepare_vocab_for_compile(vocab);
}

/// Build (and, if configured, start the keepalive for) the terminal
/// interchangeability certification thread pool ahead of first use.
///
/// Calling this at Python module import warms the pool so discovery does not
/// pay the first-use worker-wake handoff (a large latency on macOS).
pub(crate) fn warm_ti_pool() {
    compiler::stages::id_map_and_terminal_dwa::l2p::warm_ti_pool();
}

/// Dump the imported JSON Schema grammar in GLRM format.
///
/// This intentionally preserves exact subtraction syntax so dumps reflect the
/// source-level structure. The compile/import pipeline may still apply exact
/// subtraction lowering.
pub(crate) fn dump_json_schema_grammar_glrm(schema_json: &str) -> Result<String> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    let named = import::json_schema::schema_to_named_grammar(&schema)?;
    let mut factored = grammar::factoring::factor_named_grammar(named);
    if import::json_schema::simplify_grammar_enabled() {
        grammar::named_simplify::simplify_named_grammar(&mut factored);
    }
    if import::json_schema::promote_literal_choices_enabled() {
        grammar::terminal_choice_promotion::promote_choice_terminals_exact(&mut factored, false);
    }
    import::json_schema::assign_default_lexer_partitions(&mut factored);
    Ok(grammar::glrm::to_glrm(&factored))
}

pub(crate) fn set_test_compat_mode(enabled: bool) {
    crate::import::json_schema::string::TEST_COMPAT_MODE.with(|cell| {
        cell.set(if enabled {
            crate::import::json_schema::string::JsonStringCompatMode::LlGuidanceNative
        } else {
            crate::import::json_schema::string::JsonStringCompatMode::JsonSchema
        });
    });
}

impl Constraint {
    #[doc(hidden)]
    pub fn __compile_grammar_def_json(
        grammar_def_json: &str,
        vocab: &Vocab,
    ) -> Result<Self> {
        compile_grammar_def_json(grammar_def_json, vocab)
    }

    #[doc(hidden)]
    pub fn __dump_json_schema_grammar_glrm(schema_json: &str) -> Result<String> {
        dump_json_schema_grammar_glrm(schema_json)
    }

    #[doc(hidden)]
    pub fn __profile_json_schema_import(schema_json: &str) -> Result<()> {
        import::__profile_json_schema_import(schema_json)
    }

    #[doc(hidden)]
    pub fn __warm_ti_pool() {
        warm_ti_pool();
    }

    #[doc(hidden)]
    pub fn __clear_stale_weights() {
        ds::weight::clear_stale_weights();
    }

    #[doc(hidden)]
    pub fn __clear_weight_interners() {
        ds::weight::clear_weight_interners();
    }

    #[doc(hidden)]
    pub fn __clear_weight_op_caches() {
        ds::weight::clear_weight_op_caches();
    }

    #[doc(hidden)]
    pub fn __set_test_compat_mode(enabled: bool) {
        set_test_compat_mode(enabled);
    }
}

impl Vocab {
    #[doc(hidden)]
    pub fn __prepare_for_compile(&self) {
        prepare_vocab_for_compile(self);
    }
}

#[cfg(feature = "python-bindings")]
#[doc(hidden)]
pub mod __private {
    pub use crate::runtime::{AdvanceTrace, AdvanceTraceStep, GssProfileSummary, MaskProfile};
}
