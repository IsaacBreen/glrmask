#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

//! `glrmask`: grammar-constrained token masking for language-model decoding.
//!
//! The publication-facing API is intentionally small.  A [`Vocab`] describes the
//! tokenizer byte vocabulary; a [`Constraint`] compiles a grammar against that
//! vocabulary; a [`ConstraintState`] alternates **Mask** and **Commit** during
//! generation.
//!
//! Implementation details such as Terminal DWA construction, Parser DWA
//! construction, template DFAs, GLR tables, and compact token-space maps are kept
//! behind crate-private modules.  Diagnostic entry points live under
//! [`diagnostics`] rather than at the root.

pub mod api;
pub mod diagnostics;

pub(crate) mod automata;
pub(crate) mod compile;
pub(crate) mod config;
pub(crate) mod compiler;
pub(crate) mod ds;
mod error;
pub(crate) mod grammar;
pub(crate) mod grammar_ir;
pub(crate) mod import;
pub(crate) mod invariants;
pub(crate) mod parser;
pub(crate) mod runtime;
pub(crate) mod scan;
pub(crate) mod sets;
mod vocab;

#[doc(inline)]
pub use api::{
    CompileOptions,
    Constraint,
    ConstraintState,
    Error,
    GlrMaskError,
    Result,
    RuntimeOptions,
    State,
    TableAmbiguity,
    TableAmbiguityKind,
    Vocab,
};

#[doc(inline)]
pub use api::profiles::{
    AdvanceProfile,
    AdvanceTrace,
    AdvanceTraceGoto,
    AdvanceTraceReduce,
    AdvanceTraceStep,
    AdvanceTraceWave,
    CommitProfile,
    GssProfileSummary,
    MaskProfile,
    PerAdvanceEntry,
};

// Compatibility shims for existing tests and benchmark harnesses.  New code
// should prefer `glrmask::diagnostics::frontend::*`.
#[doc(hidden)]
pub use diagnostics::frontend::{
    compile_grammar_def_json,
    dump_json_schema_grammar_glrm,
    prepare_vocab_for_compile,
};
