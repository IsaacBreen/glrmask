//! Compiler stage modules.
//!
//! Top-level orchestration lives in `src/compiler/compile.rs`. This module is
//! only the home for stage implementations used by that orchestration layer.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

pub mod id_map;
pub mod parser_labels;
pub mod terminal_dwa;
pub mod template_dfa;
pub mod parser_dwa;
pub mod resolve_negatives;
