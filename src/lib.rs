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
    clear_weight_op_caches,
};
pub use error::{Error, GlrMaskError, Result};
pub use runtime::{
    CommitProfile,
    Constraint,
    ConstraintState,
    GssProfileSummary,
    PerAdvanceEntry,
};
pub use vocab::Vocab;

/// Compile a Constraint from a serialized GrammarDef JSON + vocab.
/// This runs the full compile pipeline (equivalence analysis, terminal DWA, parser DWA).
pub fn compile_grammar_def_json(grammar_def_json: &str, vocab: &Vocab) -> Result<Constraint> {
    let gdef: grammar::flat::GrammarDef = serde_json::from_str(grammar_def_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid GrammarDef JSON: {e}")))?;
    Ok(compiler::compile_owned(gdef, vocab))
}

/// Dump the imported JSON Schema grammar in GLRM format.
pub fn dump_json_schema_grammar_glrm(schema_json: &str) -> Result<String> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    let named = import::json_schema::schema_to_named_grammar(&schema)?;
    let factored = grammar::factoring::factor_named_grammar(named);
    Ok(grammar::glrm::to_glrm(&factored))
}
