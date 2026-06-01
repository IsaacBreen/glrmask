//! Compile-time construction of the paper's automata and scan relations.
//!
//! This module names the implementation after the mathematical objects used in
//! the paper.  The older `compiler::stages::*` layout grouped code by historical
//! build phases; this layout groups code by denotation and by compile phase:
//!
//! - [`pipeline`]: orchestrates the explicit phase graph from grammar/vocab to
//!   runtime constraint.
//! - [`terminal_dwa`]: builds the Terminal DWA, the weighted automaton over
//!   completed grammar-terminal sequences.
//! - [`scan_relation`]: computes the runtime Scan/CanMatch relation used to
//!   decide which terminals may complete partially scanned bytes.
//! - [`parser_dwa`]: builds the Parser DWA, the weighted automaton over parser
//!   stack prefixes.
//! - [`options`], [`profiling`], and [`thread_pool`]: isolate historical
//!   environment-variable configuration and reporting from the mathematical
//!   compile graph.
//!
//! The GLR table and grammar-analysis machinery still live under
//! `crate::compiler` during this refactor.  The important boundary is that code
//! implementing the paper's named compiled objects is no longer hidden under a
//! generic `stages` bucket.

pub(crate) mod options;
pub(crate) mod id_space;
pub(crate) mod mapped_artifact;
pub(crate) mod parser_dwa;
pub(crate) mod pipeline;
pub(crate) mod profiling;
pub(crate) mod scan_relation;
pub(crate) mod terminal_dwa;
pub(crate) mod template_dfa;
pub(crate) mod thread_pool;
pub(crate) mod tokenizer;

pub(crate) use pipeline::compile_owned;
