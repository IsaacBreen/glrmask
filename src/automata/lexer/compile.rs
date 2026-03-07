//! Regex compilation and compiled-regex wrapper.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ds::u8set::U8Set;

use super::ast::{Expr, ExprGroups};
use super::dfa::Dfa;
use super::nfa::Nfa;

/// A compiled regex (wraps a minimized DFA).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regex {
    /// The underlying DFA.
    pub dfa: Dfa,
}

impl Regex {
    /// Number of states in the DFA.
    pub fn num_states(&self) -> usize {
        unimplemented!()
    }

    /// Whether the regex matches the input completely.
    pub fn is_match(&self, input: &[u8]) -> bool {
        unimplemented!()
    }

    /// Get the next DFA state for a byte, starting from the given state.
    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        unimplemented!()
    }

    /// Whether a DFA state is accepting.
    pub fn is_accepting(&self, state: u32) -> bool {
        unimplemented!()
    }

    /// Get the set of valid next bytes from a DFA state.
    pub fn get_u8set(&self, state: u32) -> U8Set {
        unimplemented!()
    }
}

impl Expr {
    /// Build a single-group regex from this expression.
    pub fn build(self) -> Regex {
        unimplemented!()
    }

    /// CPS (Continuation-Passing Style) NFA compilation.
    ///
    /// Compiles this expression into NFA states such that entering at the returned
    /// state will recognize the expression and then flow to `cont`.
    pub(crate) fn compile_cps(
        expr: &Expr,
        nfa: &mut Nfa,
        cont: u32,
        cache: &mut HashMap<(usize, u32), u32>,
    ) -> u32 {
        unimplemented!()
    }
}

impl ExprGroups {
    /// Compile all groups into a single multi-group regex.
    pub fn build(self) -> Regex {
        unimplemented!()
    }

    /// Compile to NFA (without DFA conversion — useful for testing).
    pub fn build_nfa(self) -> Nfa {
        unimplemented!()
    }
}
