use rustc_hash::{FxHashMap, FxHashSet};

use crate::automata::lexer::tokenizer::TokenizerExecResult;
use crate::scan::execution;

use super::super::artifact::Constraint;

pub(super) struct InitialCommitScan {
    pub exec_results: FxHashMap<u32, TokenizerExecResult>,
    pub remapped_tokenizer_states: FxHashMap<u32, u32>,
    pub accepted_terminals: FxHashMap<u32, FxHashSet<u32>>,
}

/// Execute the runtime lexer scan for a committed byte fragment.
///
/// Commit owns parser/GSS advancement, but it no longer owns the basic scan
/// primitive.  The actual scan helper lives under `crate::scan::execution` so
/// compile-time scan-relation construction and runtime commit use the same
/// vocabulary while remaining separate implementations.
pub(super) fn execute_tokenizer_from_state_small(
    constraint: &Constraint,
    bytes: &[u8],
    start_state: u32,
) -> TokenizerExecResult {
    execution::execute_tokenizer_from_state(
        &constraint.tokenizer,
        &constraint.tokenizer_fast_transitions,
        bytes,
        start_state,
    )
}
