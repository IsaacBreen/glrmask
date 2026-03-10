//! Direct port of sep1/grammars2024's `src/equivalence_analysis/` folder.
//!
//! This module contains the full equivalence analysis pipeline from the sep1
//! reference codebase, adapted for glrmask's DFA types via the `compat` shim.
//!
//! Files are kept as close to the originals as possible for traceability.
//! The `compat` module provides `FlatDfa`/`Sep1Tokenizer` wrappers.

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]

pub mod compat;

pub mod state_equivalence_analysis_fast;
pub mod vocab_equivalence_analysis_fast;
pub mod vocab_equivalence_analysis_fast_simple;
pub mod vocab_equivalence_analysis_flat;
pub mod combined_equivalence_analysis;
