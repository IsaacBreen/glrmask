//! Compatibility namespace for the older `crate::grammar::*` paths.
//!
//! Publication-facing source now lives under `crate::grammar_ir`.  This module
//! remains as a thin shim so existing compiler/import code can be migrated in
//! small, reviewable patches.

pub mod ast;
pub mod exact_subtraction_lowering;
pub mod expr_nfa;
pub mod factoring;
pub mod flat;
pub mod glrm;
pub mod named_simplify;
pub mod terminal_choice_promotion;
