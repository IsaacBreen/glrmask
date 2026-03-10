#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use range_set_blaze::RangeSetBlaze;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::possible_matches::build_possible_matches_from_token_bytes;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar_def::TerminalID;
use crate::ds::leveled_gss::LeveledGSS;

use super::state::ConstraintState;

pub(crate) type TokenizerStateID = u32;
pub(crate) type PossibleMatchesByState =
    BTreeMap<TokenizerStateID, BTreeMap<TerminalID, RangeSetBlaze<u32>>>;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct Constraint {
    pub(crate) parser_dwa: DWA,
    pub(crate) table: GLRTable,
    pub(crate) tokenizer: Tokenizer,
    #[serde(default)]
    pub(crate) ignore_terminal: Option<TerminalID>,

    #[serde(with = "crate::runtime::serde::serde_btmap_rsb")]
    pub(crate) possible_matches: PossibleMatchesByState,
    pub(crate) state_to_internal_tsid: Vec<u32>,
    pub(crate) internal_tsid_to_states: Vec<Vec<u32>>,
    #[serde(default)]
    pub(crate) original_token_to_internal: Vec<u32>,
    #[serde(default)]
    pub(crate) internal_token_to_tokens: Vec<Vec<u32>>,
    pub(crate) eos_token_id: Option<u32>,
    pub(crate) token_bytes: BTreeMap<u32, Vec<u8>>,
    #[serde(default)]
    pub(crate) internal_token_bytes: BTreeMap<u32, Vec<u8>>,
}

impl Clone for Constraint {
    fn clone(&self) -> Self {
        Self {
            parser_dwa: self.parser_dwa.clone(),
            table: self.table.clone(),
            tokenizer: self.tokenizer.clone(),
            ignore_terminal: self.ignore_terminal,
            possible_matches: self.possible_matches.clone(),
            state_to_internal_tsid: self.state_to_internal_tsid.clone(),
            internal_tsid_to_states: self.internal_tsid_to_states.clone(),
            original_token_to_internal: self.original_token_to_internal.clone(),
            internal_token_to_tokens: self.internal_token_to_tokens.clone(),
            eos_token_id: self.eos_token_id,
            token_bytes: self.token_bytes.clone(),
            internal_token_bytes: self.internal_token_bytes.clone(),
        }
    }
}

impl Constraint {
    fn possible_matches_use_internal_tokens(&self) -> bool {
        !self.internal_token_bytes.is_empty()
    }

    fn all_possible_matches(&self) -> &PossibleMatchesByState {
        &self.possible_matches
    }

    pub(crate) fn rebuild_possible_matches(&mut self) {
        let token_bytes = if self.possible_matches_use_internal_tokens() {
            &self.internal_token_bytes
        } else {
            &self.token_bytes
        };
        self.possible_matches = build_possible_matches_from_token_bytes(&self.tokenizer, token_bytes);
    }

    pub fn start(&self) -> ConstraintState<'_> {
        
        let initial_parser_state = 0u32;
        let initial_tok_state = self.tokenizer.initial_state();

        let mut state = BTreeMap::new();
        let gss = LeveledGSS::from_stacks(&[(vec![initial_parser_state], BTreeMap::new())]);
        state.insert(initial_tok_state, gss);

        ConstraintState { constraint: self, state }
    }

    pub fn mask_len(&self) -> usize {
        self.token_bytes
            .keys()
            .max()
            .map(|token_id| (*token_id as usize / 32) + 1)
            .unwrap_or(0)
    }

    pub(crate) fn parser_dwa(&self) -> &DWA {
        &self.parser_dwa
    }

    pub(crate) fn possible_matches_for_state(
        &self,
        tokenizer_state: u32,
    ) -> BTreeMap<TerminalID, RangeSetBlaze<u32>> {
        let possible_matches = self.all_possible_matches()
            .get(&tokenizer_state)
            .cloned()
            .unwrap_or_default();
        if self.possible_matches_use_internal_tokens() {
            possible_matches
                .into_iter()
                .map(|(terminal, internal_tokens)| {
                    (terminal, self.expand_internal_token_set(&internal_tokens))
                })
                .collect()
        } else {
            possible_matches
        }
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

    pub(crate) fn internal_token_universe(&self) -> RangeSetBlaze<u32> {
        if self.internal_token_to_tokens.is_empty() {
            let Some(max_token_id) = self.token_bytes.keys().next_back().copied() else {
                return RangeSetBlaze::new();
            };
            return RangeSetBlaze::from_iter([0..=max_token_id]);
        }

        RangeSetBlaze::from_iter([0..=self.internal_token_to_tokens.len().saturating_sub(1) as u32])
    }

    pub(crate) fn internalize_token_set(
        &self,
        original_tokens: &RangeSetBlaze<u32>,
    ) -> RangeSetBlaze<u32> {
        if self.original_token_to_internal.is_empty() {
            return original_tokens.clone();
        }

        let mut internal_tokens = RangeSetBlaze::new();
        for token_id in original_tokens.iter() {
            internal_tokens.insert(self.internal_token_for_original(token_id));
        }
        internal_tokens
    }

    pub(crate) fn expand_internal_token_set(
        &self,
        internal_tokens: &RangeSetBlaze<u32>,
    ) -> RangeSetBlaze<u32> {
        if self.internal_token_to_tokens.is_empty() {
            return internal_tokens.clone();
        }

        // Collect all original token IDs, sort, build ranges.
        let total_estimate: usize = internal_tokens.iter()
            .filter_map(|t| self.internal_token_to_tokens.get(t as usize))
            .map(|v| v.len())
            .sum();
        let mut all_ids = Vec::with_capacity(total_estimate);
        for internal_token in internal_tokens.iter() {
            if let Some(originals) = self.internal_token_to_tokens.get(internal_token as usize) {
                all_ids.extend_from_slice(originals);
            }
        }
        all_ids.sort_unstable();
        all_ids.dedup();
        if all_ids.is_empty() {
            return RangeSetBlaze::new();
        }
        let mut ranges = Vec::new();
        let mut start = all_ids[0];
        let mut end = all_ids[0];
        for &id in &all_ids[1..] {
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

    pub(crate) fn possible_matches_for_state_internal(
        &self,
        tokenizer_state: u32,
    ) -> BTreeMap<TerminalID, RangeSetBlaze<u32>> {
        let possible_matches = self
            .all_possible_matches()
            .get(&tokenizer_state)
            .cloned()
            .unwrap_or_default();
        if self.possible_matches_use_internal_tokens() {
            possible_matches
        } else {
            possible_matches
                .into_iter()
                .map(|(terminal, token_ids)| (terminal, self.internalize_token_set(&token_ids)))
                .collect()
        }
    }

    pub(crate) fn possible_matches_for_internal_tsid(
        &self,
        internal_tsid: u32,
    ) -> BTreeMap<TerminalID, RangeSetBlaze<u32>> {
        let mut merged = BTreeMap::new();
        let Some(original_states) = self.internal_tsid_to_states.get(internal_tsid as usize) else {
            return merged;
        };

        for &tokenizer_state in original_states {
            for (terminal, token_ids) in self.possible_matches_for_state_internal(tokenizer_state) {
                merged
                    .entry(terminal)
                    .and_modify(|existing: &mut RangeSetBlaze<u32>| {
                        *existing = existing.clone() | token_ids.clone();
                    })
                    .or_insert(token_ids);
            }
        }

        merged
    }
}
