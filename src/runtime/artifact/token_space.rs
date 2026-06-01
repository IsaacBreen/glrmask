//! Runtime token-space and lexer-state quotients.
//!
//! The compiled runtime artifact uses four distinct coordinate systems:
//!
//! - original vocabulary token ids, the ids supplied by the model/tokenizer;
//! - final runtime-internal token ids, after Parser-DWA and CanMatch
//!   reconciliation;
//! - original lexer/tokenizer states;
//! - final runtime-internal tokenizer-state ids used inside weights.
//!
//! This module owns the conversion functions between those coordinate systems.

use std::collections::BTreeMap;

use range_set_blaze::RangeSetBlaze;

use crate::sets::weight::Weight;
use crate::grammar::flat::TerminalID;
use crate::parser::glr::accumulator::TerminalsDisallowed;
use crate::parser::glr::advance::ParserGSS;

use super::Constraint;

/// CanMatch relation keyed by grammar terminal id.
pub(crate) type CanMatchByTerminal = BTreeMap<TerminalID, Weight>;

/// Original-vocabulary token id supplied to or returned from the model.
pub(crate) type OriginalTokenId = u32;

/// Runtime-internal token id after final Parser-DWA/CanMatch reconciliation.
pub(crate) type InternalTokenId = u32;

/// Original tokenizer DFA state id.
pub(crate) type OriginalTokenizerStateId = u32;

/// Runtime-internal tokenizer-state class id used inside weights.
pub(crate) type InternalTokenizerStateId = u32;

impl Constraint {
	pub(crate) fn can_match_for_state(
		&self,
		tokenizer_state: u32,
	) -> BTreeMap<TerminalID, RangeSetBlaze<u32>> {
		let internal_tsid = self.internal_tsid_for_state(tokenizer_state);
		self.can_match
			.iter()
			.filter_map(|(&terminal, weight)| {
				let tokens = weight.tokens_for_tsid(internal_tsid);
				if tokens.is_empty() {
					None
				} else {
					Some((terminal, self.expand_internal_token_set(&tokens)))
				}
			})
			.collect()
	}

	pub(crate) fn internal_tsid_for_state(&self, tokenizer_state: u32) -> u32 {
		self.state_to_internal_tsid
			.get(tokenizer_state as usize)
			.copied()
			.unwrap_or(tokenizer_state)
	}

	pub(crate) fn internal_token_for_original(&self, token_id: u32) -> u32 {
		self.original_token_to_internal
			.get(token_id as usize)
			.copied()
			.filter(|internal_id| *internal_id != u32::MAX)
			.unwrap_or(token_id)
	}

	pub(crate) fn final_internal_token_for_original(&self, token_id: u32) -> Option<u32> {
		let internal = *self.original_token_to_internal.get(token_id as usize)?;

		if internal == u32::MAX {
			return None;
		}

		if !self.internal_token_to_tokens.is_empty()
			&& internal as usize >= self.internal_token_to_tokens.len()
		{
			return None;
		}

		Some(internal)
	}

	pub(crate) fn internal_token_universe(&self) -> RangeSetBlaze<u32> {
		if self.internal_token_to_tokens.is_empty() {
			let Some(max_token_id) = self.max_original_token_id() else {
				return RangeSetBlaze::new();
			};
			return RangeSetBlaze::from_iter([0..=max_token_id]);
		}

		RangeSetBlaze::from_iter([0..=self.internal_token_to_tokens.len().saturating_sub(1) as u32])
	}

	pub(crate) fn expand_internal_token_set(
		&self,
		internal_tokens: &RangeSetBlaze<u32>,
	) -> RangeSetBlaze<u32> {
		if self.internal_token_to_tokens.is_empty() {
			return internal_tokens.clone();
		}

		let all_ids = self.collect_original_token_ids(internal_tokens);
		Self::range_set_from_sorted_ids(&all_ids)
	}

	pub(crate) fn initial_state_map(&self) -> BTreeMap<u32, ParserGSS> {
		let initial_tok_state = self.tokenizer.initial_state();
		let parser_gss = ParserGSS::from_stacks(&[(vec![0u32], TerminalsDisallowed::new())]);
		BTreeMap::from([(initial_tok_state, parser_gss)])
	}

	fn collect_original_token_ids(&self, internal_tokens: &RangeSetBlaze<u32>) -> Vec<u32> {
		let total_estimate: usize = internal_tokens
			.iter()
			.filter_map(|token| self.internal_token_to_tokens.get(token as usize))
			.map(Vec::len)
			.sum();
		let mut all_ids = Vec::with_capacity(total_estimate);
		for internal_token in internal_tokens.iter() {
			if let Some(originals) = self.internal_token_to_tokens.get(internal_token as usize) {
				all_ids.extend_from_slice(originals);
			}
		}
		all_ids.sort_unstable();
		all_ids.dedup();
		all_ids
	}

	fn range_set_from_sorted_ids(ids: &[u32]) -> RangeSetBlaze<u32> {
		let Some((&first, rest)) = ids.split_first() else {
			return RangeSetBlaze::new();
		};

		let mut ranges = Vec::new();
		let mut start = first;
		let mut end = first;
		for &id in rest {
			if id == end + 1 {
				end = id;
			} else {
				ranges.push(start..=end);
				start = id;
				end = id;
			}
		}
		ranges.push(start..=end);
		RangeSetBlaze::from_iter(ranges)
	}
}
