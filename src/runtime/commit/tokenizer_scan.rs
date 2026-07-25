use crate::automata::lexer::Lexer;
use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::{TokenizerExecResult, TokenizerStateSet};

use super::super::artifact::Constraint;


/// Reusable exact tokenizer execution scratch for the runtime hot path.
/// Capacities are allocated when a ConstraintState is created. If an unusual
/// epsilon frontier or match set exceeds them, execution declines before
/// mutating parser state and the existing general path remains authoritative.
#[derive(Debug)]
pub(crate) struct ReusableTokenizerExecScratch {
    pub states: Vec<u32>,
    next_states: Vec<u32>,
    pub matches: Vec<crate::automata::lexer::tokenizer::TokenizerMatch>,
}

impl Default for ReusableTokenizerExecScratch {
    fn default() -> Self {
        Self {
            states: Vec::with_capacity(64),
            next_states: Vec::with_capacity(64),
            matches: Vec::with_capacity(64),
        }
    }
}

/// Execute the epsilon tokenizer exactly using only preallocated scratch.
/// Returns false if a bounded scratch capacity would be exceeded.
pub(crate) fn execute_tokenizer_reusable(
    constraint: &Constraint,
    bytes: &[u8],
    start_state: u32,
    scratch: &mut ReusableTokenizerExecScratch,
) -> bool {
    let closures = constraint.tokenizer.all_singleton_epsilon_closures();
    let Some(start_closure) = closures.get(start_state as usize) else {
        return false;
    };
    if start_closure.len() > scratch.states.capacity() {
        return false;
    }
    scratch.states.clear();
    scratch.states.extend_from_slice(start_closure);
    scratch.matches.clear();

    for (index, &byte) in bytes.iter().enumerate() {
        scratch.next_states.clear();
        for &state in &scratch.states {
            let Some(target) = constraint.tokenizer.step(state, byte) else {
                continue;
            };
            let Some(target_closure) = closures.get(target as usize) else {
                return false;
            };
            if scratch.next_states.len() + target_closure.len()
                > scratch.next_states.capacity()
            {
                return false;
            }
            scratch.next_states.extend_from_slice(target_closure);
        }
        if scratch.next_states.is_empty() {
            scratch.states.clear();
            return true;
        }
        scratch.next_states.sort_unstable();
        scratch.next_states.dedup();
        std::mem::swap(&mut scratch.states, &mut scratch.next_states);

        let width = index + 1;
        for &state in &scratch.states {
            for terminal in constraint.tokenizer.matched_terminals_iter(state) {
                let prior_width = scratch
                    .matches
                    .iter()
                    .find(|matched| matched.id == terminal)
                    .map(|matched| matched.width);
                match prior_width {
                    Some(prior) if prior > width => continue,
                    Some(prior) if prior < width => {
                        scratch.matches.retain(|matched| matched.id != terminal);
                    }
                    Some(_) if scratch
                        .matches
                        .iter()
                        .any(|matched| matched.id == terminal && matched.end_state == state) =>
                    {
                        continue;
                    }
                    _ => {}
                }
                if scratch.matches.len() == scratch.matches.capacity() {
                    return false;
                }
                scratch.matches.push(crate::automata::lexer::tokenizer::TokenizerMatch {
                    id: terminal,
                    width,
                    end_state: state,
                });
            }
        }
    }
    true
}

pub(super) struct InitialCommitScan {
	pub exec_results: FxHashMap<u32, TokenizerExecResult>,
}

pub(super) fn execute_tokenizer_from_state_small(
    constraint: &Constraint,
    bytes: &[u8],
    start_state: u32,
) -> TokenizerExecResult {
    let mut result = TokenizerExecResult {
        end_state: TokenizerStateSet::new(),
        matches: Vec::with_capacity(8),
    };
    execute_tokenizer_from_state_small_into(constraint, bytes, start_state, &mut result);
    result
}

pub(super) fn execute_tokenizer_from_state_small_into(
    constraint: &Constraint,
    bytes: &[u8],
    start_state: u32,
    result: &mut TokenizerExecResult,
) {
    if constraint.tokenizer.has_epsilon_transitions() {
        *result = constraint.tokenizer.execute_from_state(bytes, start_state);
        return;
    }
    result.end_state.clear();
    result.matches.clear();
    let mut tokenizer_state = start_state;

    for (index, &byte) in bytes.iter().enumerate() {
        let next_state = constraint.tokenizer_fast_transitions.transition(
            &constraint.tokenizer,
            tokenizer_state,
            byte,
        );
        if next_state == u32::MAX {
            return;
        }

        tokenizer_state = next_state;
        let width = index + 1;
        for terminal in constraint.tokenizer.matched_terminals_iter(tokenizer_state) {
            if let Some(existing) = result
                .matches
                .iter_mut()
                .find(|matched| matched.id == terminal)
            {
                existing.width = width;
                existing.end_state = tokenizer_state;
            } else {
                result.matches.push(crate::automata::lexer::tokenizer::TokenizerMatch {
                    id: terminal,
                    width,
                    end_state: tokenizer_state,
                });
            }
        }
    }

    result.end_state.push(tokenizer_state);
}
