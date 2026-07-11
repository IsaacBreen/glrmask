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

pub use ds::weight::{
    clear_stale_weights,
    clear_weight_interners,
    clear_weight_op_caches,
};
pub use error::{Error, GlrMaskError, Result};
pub use compiler::glr::table::{TableAmbiguity, TableAmbiguityKind};
pub use runtime::{
    AdvanceProfile,
    AdvanceTrace,
    AdvanceTraceGoto,
    AdvanceTraceReduce,
    AdvanceTraceStep,
    AdvanceTraceWave,
    CommitProfile,
    Constraint,
    ConstraintState,
    FinalMaskMapping,
    GssProfileSummary,
    MaskProfile,
    PerAdvanceEntry,
};
pub use vocab::Vocab;

#[doc(hidden)]
pub use import::__profile_json_schema_import;
/// Compile a Constraint from a serialized GrammarDef JSON + vocab.
/// This runs the full compile pipeline (equivalence analysis, terminal DWA, parser DWA).
pub fn compile_grammar_def_json(grammar_def_json: &str, vocab: &Vocab) -> Result<Constraint> {
    let gdef: grammar::flat::GrammarDef = serde_json::from_str(grammar_def_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid GrammarDef JSON: {e}")))?;
    Ok(compiler::stages::id_map_and_terminal_dwa::l2p::with_ti_pool(|| {
        compiler::compile_owned(gdef, vocab)
    }))
}

/// Populate compile-time artifacts that are pure functions of the vocabulary.
///
/// This intentionally does not compile any grammar/schema-dependent artifact.
pub fn prepare_vocab_for_compile(vocab: &Vocab) {
    compiler::compile::prepare_vocab_for_compile(vocab);
}

/// Build (and, if configured, start the keepalive for) the terminal
/// interchangeability certification thread pool ahead of first use.
///
/// Calling this at Python module import warms the pool so discovery does not
/// pay the first-use worker-wake handoff (a large latency on macOS).
pub fn warm_ti_pool() {
    compiler::stages::id_map_and_terminal_dwa::l2p::warm_ti_pool();
}

/// Dump the imported JSON Schema grammar in GLRM format.
///
/// This intentionally preserves exact subtraction syntax so dumps reflect the
/// source-level structure. The compile/import pipeline may still apply exact
/// subtraction lowering.
pub fn dump_json_schema_grammar_glrm(schema_json: &str) -> Result<String> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    let named = import::json_schema::schema_to_named_grammar(&schema)?;
    let mut factored = grammar::factoring::factor_named_grammar(named);
    import::json_schema::prepare_named_grammar_for_dump(&mut factored)?;
    Ok(grammar::glrm::to_glrm(&factored))
}

#[doc(hidden)]
pub fn set_test_compat_mode(enabled: bool) {
    crate::import::json_schema::string::TEST_COMPAT_MODE.with(|cell| {
        cell.set(if enabled {
            crate::import::json_schema::string::JsonStringCompatMode::LlGuidanceNative
        } else {
            crate::import::json_schema::string::JsonStringCompatMode::JsonSchema
        });
    });
}
