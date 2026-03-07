//! Compiler stage modules.
//!
//! Top-level orchestration lives in `src/compiler/compile.rs`. This module is
//! only the home for stage implementations used by that orchestration layer.
//!
//! Intended stage order:
//! `equivalence_analysis` → `templates::characterize` → `terminal_dwa`
//! → `templates::compile` → `parser_dwa` → `resolve_negatives`.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

pub mod equivalence_analysis;
pub mod templates;
pub mod terminal_dwa;
pub mod parser_dwa;
pub mod resolve_negatives;
