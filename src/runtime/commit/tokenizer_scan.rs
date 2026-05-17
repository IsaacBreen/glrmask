use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::automata::lexer::tokenizer::TokenizerExecResult;

use super::super::artifact::Constraint;

pub(super) struct InitialCommitScan {
	pub exec_results: FxHashMap<u32, TokenizerExecResult>,
	pub remapped_tokenizer_states: FxHashMap<u32, u32>,
	pub accepted_terminals: FxHashMap<u32, FxHashSet<u32>>,
}

pub(super) fn execute_tokenizer_from_state_small(
	constraint: &Constraint,
	bytes: &[u8],
	start_state: u32,
) -> TokenizerExecResult {
	let mut tokenizer_state = start_state;
	let mut matches = SmallVec::<[(u32, usize, u32); 8]>::new();

	for (index, &byte) in bytes.iter().enumerate() {
		let next_state = constraint
			.tokenizer_fast_transitions
			.get(tokenizer_state as usize)
			.map_or(u32::MAX, |transitions| transitions[byte as usize]);
		if next_state == u32::MAX {
			return TokenizerExecResult {
				end_state: None,
				matches: matches
					.into_iter()
					.map(|(id, width, end_state)| crate::automata::lexer::tokenizer::TokenizerMatch {
						id,
						width,
						end_state,
					})
					.collect(),
			};
		}

		tokenizer_state = next_state;
		let width = index + 1;
		for terminal in constraint.tokenizer.matched_terminals_iter(tokenizer_state) {
			if let Some((_, existing_width, existing_end_state)) =
				matches.iter_mut().find(|(id, _, _)| *id == terminal)
			{
				*existing_width = width;
				*existing_end_state = tokenizer_state;
			} else {
				matches.push((terminal, width, tokenizer_state));
			}
		}
	}

	TokenizerExecResult {
		end_state: Some(tokenizer_state),
		matches: matches
			.into_iter()
			.map(|(id, width, end_state)| crate::automata::lexer::tokenizer::TokenizerMatch {
				id,
				width,
				end_state,
			})
			.collect(),
	}
}
