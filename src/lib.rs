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
    clear_all_weights,
    clear_stale_weights,
    clear_weight_caches,
    clear_weight_op_caches,
};
pub use error::{Error, GlrMaskError, Result};
pub use runtime::{
    CommitProfile,
    Constraint,
    ConstraintState,
    FillMaskTimings,
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
