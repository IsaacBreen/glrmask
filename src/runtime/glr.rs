//! Runtime GLR support.
//!
//! These helpers are the runtime execution layer for the compiled GLR parse
//! table. They support mask viability checks, accept checks, and parser-state
//! stepping during commit.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet};

use crate::compiler::glr::table::GlrTable;
use crate::compiler::grammar_def::TerminalId;
use crate::ds::leveled_gss::{LeveledGSS, Merge};

/// Maps tokenizer state ID → set of disallowed terminal IDs.
///
/// Used as the GSS accumulator to track which (tsid, terminal) pairs
/// should be excluded during mask computation.
pub type TerminalsDisallowed = BTreeMap<u32, BTreeSet<u32>>;

/// Create a fresh (empty) `TerminalsDisallowed`.
pub(crate) fn terminals_disallowed_fresh() -> TerminalsDisallowed {
    unimplemented!()
}

impl Merge for TerminalsDisallowed {
    fn merge(&self, other: &Self) -> Self {
        unimplemented!()
    }
}

/// A GSS (Graph-Structured Stack) for the GLR parser.
///
/// Stack items are `u32` parser state IDs.
/// Accumulator is `TerminalsDisallowed` (currently unused but reserved for future mask pruning).
pub type ParserGSS = LeveledGSS<u32, TerminalsDisallowed>;

/// Step the GLR parser on a terminal using the GSS.
///
/// This is the core GLR stepping function. It:
/// 1. Groups stacks by top state via `peek()` + `isolate()`
/// 2. Looks up actions for each (state, terminal) pair
/// 3. Handles shifts with `push`, reduces with `popn` + goto + `push`
/// 4. Merges all results with balanced merge
///
/// This is equivalent to grammars2024's `process_token_gss`.
pub(crate) fn step_glr_gss(table: &GlrTable, gss: &ParserGSS, terminal: TerminalId) -> ParserGSS {
    unimplemented!()
}

/// Compute ε-reduce closure for a single stack.
///
/// For each ε-production (pop_count=0 rule) that can fire at the top state,
/// push the goto state to produce a new extended stack. This is applied
/// recursively until no more ε-reductions are possible.
///
/// The original stack is NOT included in `out` — only newly produced variants.
pub(crate) fn epsilon_reduce_stacks(table: &GlrTable, stack: &[u32], out: &mut Vec<Vec<u32>>) {
    unimplemented!()
}

/// Check if a stack can reach Accept via EOF (possibly after reduce cascades).
pub(crate) fn can_accept(table: &GlrTable, stack: &[u32], eof: TerminalId) -> bool {
    unimplemented!()
}

pub(crate) fn can_accept_inner(table: &GlrTable, stack: &[u32], eof: TerminalId, depth: usize) -> bool {
    unimplemented!()
}

/// Check if a state has viable continuations.
///
/// A state is viable if at least one (tok_state, gss) entry satisfies:
/// 1. tok_state is the initial tokenizer state (clean terminal boundary), OR
/// 2. At least one reachable terminal from tok_state has valid parser actions
///    for some top parser state in the GSS.
///
/// This filters out states where the tokenizer is mid-match but no reachable
/// terminal matches any parser action — such states are effectively dead.
pub(crate) fn has_viable_state(
    state: &BTreeMap<u32, ParserGSS>,
    table: &GlrTable,
    reachable: &[std::collections::BTreeSet<crate::compiler::grammar_def::TerminalId>],
    initial_tok_state: u32,
    tok_dfa: &crate::automata::dfa::Dfa,
) -> bool {
    unimplemented!()
}