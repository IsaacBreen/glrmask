//! Runtime sequence state.
//!
//! `ConstraintState` tracks per-sequence state and delegates to the runtime
//! helper layers for mask computation, commit handling, forcing, and GLR
//! execution.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::compiler::glr::parser::ParserGSS;

use super::constraint::Constraint;

/// Per-sequence constraint state.
///
/// Tracks the current parse + tokenizer state. Computes token masks and
/// advances state when tokens are committed.
///
/// State is a map from tokenizer DFA state → GSS of parser stacks.
/// The GSS provides structural sharing for efficient GLR parsing.
#[derive(Debug, Clone)]
pub struct ConstraintState<'a> {
    /// Borrowed reference to the compiled constraint.
    pub(crate) constraint: &'a Constraint,
    /// tokenizer DFA state → GSS of parser state stacks.
    pub(crate) state: BTreeMap<u32, ParserGSS>,
}

impl<'a> ConstraintState<'a> {
    /// Whether the grammar has been fully satisfied (EOS is valid at current position).
    pub fn is_finished(&self) -> bool {
        unimplemented!()
    }
}
