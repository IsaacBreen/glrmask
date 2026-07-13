use crate::automata::lexer::Lexer;
use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::{TokenizerExecResult, TokenizerStateSet};

use super::super::artifact::Constraint;

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
		let next_state = constraint
			.tokenizer_fast_transitions
			.get(tokenizer_state as usize)
			.map_or(u32::MAX, |transitions| transitions[byte as usize]);
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
