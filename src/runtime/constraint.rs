#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
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

    /// Precomputed bitmask fragments for each internal token.
    /// `internal_token_buf_masks[i]` contains (word_index, or_mask) pairs
    /// for all original tokens that map to internal token `i`.
    #[serde(default)]
    pub(crate) internal_token_buf_masks: Vec<Vec<(u16, u32)>>,
    #[serde(skip)]
    pub(crate) internal_token_dense_words: usize,
    #[serde(skip)]
    pub(crate) weight_token_dense_masks: FxHashMap<usize, Box<[u64]>>,
    /// Precomputed dense bitmask for the seed phase: for each (tokenizer_state, terminal_id),
    /// the dense bitmap of internal tokens that terminal covers in that state.
    #[serde(skip)]
    pub(crate) seed_terminal_dense: FxHashMap<(u32, TerminalID), Box<[u64]>>,
    /// Dense bitmap of the full internal token universe.
    #[serde(skip)]
    pub(crate) seed_universe_dense: Box<[u64]>,
    /// Fast DWA transition lookup (FxHashMap instead of BTreeMap).
    /// Built from parser_dwa.states at load/build time.
    #[serde(skip)]
    pub(crate) dwa_fast_transitions: Vec<FxHashMap<i32, (u32, crate::ds::weight::Weight)>>,
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
            internal_token_buf_masks: self.internal_token_buf_masks.clone(),
            internal_token_dense_words: self.internal_token_dense_words,
            weight_token_dense_masks: self.weight_token_dense_masks.clone(),
            seed_terminal_dense: self.seed_terminal_dense.clone(),
            seed_universe_dense: self.seed_universe_dense.clone(),
            dwa_fast_transitions: self.dwa_fast_transitions.clone(),
        }
    }
}

impl Constraint {

    /// Build precomputed bitmask fragments for each internal token.
    pub(crate) fn build_buf_masks(&mut self) {
        if self.internal_token_to_tokens.is_empty() {
            self.internal_token_buf_masks = Vec::new();
            return;
        }
        self.internal_token_buf_masks = self.internal_token_to_tokens.iter().map(|originals| {
            let mut word_map = BTreeMap::<u16, u32>::new();
            for &original in originals {
                let word = (original / 32) as u16;
                let bit = original % 32;
                *word_map.entry(word).or_default() |= 1u32 << bit;
            }
            word_map.into_iter().collect()
        }).collect();
    }

    /// Build fast transition lookup tables from the DWA's BTreeMap transitions.
    pub(crate) fn build_fast_transitions(&mut self) {
        self.dwa_fast_transitions = self.parser_dwa.states.iter().map(|state| {
            state.transitions.iter().map(|(&label, (target, weight))| {
                (label, (*target, weight.clone()))
            }).collect()
        }).collect();
    }

    pub(crate) fn build_dense_token_masks(&mut self) {
        self.internal_token_dense_words = self.internal_token_to_tokens.len().div_ceil(64);
        if self.internal_token_dense_words == 0 {
            self.weight_token_dense_masks.clear();
            return;
        }

        let mut dense_masks = FxHashMap::default();
        for state in &self.parser_dwa.states {
            if let Some(final_weight) = &state.final_weight {
                self.collect_weight_dense_masks(final_weight, &mut dense_masks);
            }
            for (_, weight) in state.transitions.values() {
                self.collect_weight_dense_masks(weight, &mut dense_masks);
            }
        }
        self.weight_token_dense_masks = dense_masks;
    }

    /// Precompute dense bitmaps for the seed phase: one bitmap per (state, terminal)
    /// pair, plus the universe bitmap. This lets seed_weight_dense use bitwise ANDNOT
    /// instead of RangeSetBlaze subtraction.
    pub(crate) fn build_seed_dense_masks(&mut self) {
        let dw = self.internal_token_dense_words;
        if dw == 0 {
            self.seed_terminal_dense.clear();
            self.seed_universe_dense = Box::new([]);
            return;
        }

        // Universe bitmap.
        let universe = self.internal_token_universe();
        self.seed_universe_dense = self.dense_words_from_internal_set(&universe);

        // Per-(state, terminal) bitmaps.
        let mut terminal_masks = FxHashMap::default();
        for (&tok_state, terminals) in &self.possible_matches {
            for (&terminal_id, llm_tokens) in terminals {
                terminal_masks.insert(
                    (tok_state, terminal_id),
                    self.dense_words_from_internal_set(llm_tokens),
                );
            }
        }
        self.seed_terminal_dense = terminal_masks;
    }

    fn collect_weight_dense_masks(
        &self,
        weight: &crate::ds::weight::Weight,
        dense_masks: &mut FxHashMap<usize, Box<[u64]>>,
    ) {
        for token_set in weight.unique_token_sets() {
            let key = token_set as *const RangeSetBlaze<u32> as usize;
            dense_masks
                .entry(key)
                .or_insert_with(|| self.dense_words_from_internal_set(token_set));
        }
    }

    fn dense_words_from_internal_set(&self, internal_tokens: &RangeSetBlaze<u32>) -> Box<[u64]> {
        let mut words = vec![0u64; self.internal_token_dense_words];
        for internal_token in internal_tokens.iter() {
            let index = internal_token as usize / 64;
            let bit = internal_token as usize % 64;
            if let Some(word) = words.get_mut(index) {
                *word |= 1u64 << bit;
            }
        }
        words.into_boxed_slice()
    }

    pub(crate) fn or_dense_intersection_to_buf(
        &self,
        left_words: &[u64],
        right_words: &[u64],
        buf: &mut [u32],
    ) {
        for (word_index, (&left_word, &right_word)) in left_words.iter().zip(right_words.iter()).enumerate() {
            let mut overlap = left_word & right_word;
            while overlap != 0 {
                let bit = overlap.trailing_zeros() as usize;
                let internal_token = word_index * 64 + bit;
                let masks = &self.internal_token_buf_masks[internal_token];
                for &(buf_word, mask) in masks {
                    buf[buf_word as usize] |= mask;
                }
                overlap &= overlap - 1;
            }
        }
    }

    fn or_single_weight_intersection_to_buf_sparse(
        &self,
        start: u32,
        end: u32,
        single_tokens: &RangeSetBlaze<u32>,
        other: &crate::ds::weight::Weight,
        buf: &mut [u32],
    ) {
        other.for_each_intersection_tokens_with_single(start, end, single_tokens, |tokens| {
            self.or_internal_token_set_to_buf(tokens, buf);
        });
    }

    pub(crate) fn or_internal_token_set_to_buf(
        &self,
        internal_tokens: &RangeSetBlaze<u32>,
        buf: &mut [u32],
    ) {
        if !self.internal_token_buf_masks.is_empty() {
            for internal_token in internal_tokens.iter() {
                let masks = &self.internal_token_buf_masks[internal_token as usize];
                for &(word_idx, mask) in masks {
                    buf[word_idx as usize] |= mask;
                }
            }
        } else {
            for token_id in internal_tokens.iter() {
                let word = token_id as usize / 32;
                let bit = token_id as usize % 32;
                if let Some(slot) = buf.get_mut(word) {
                    *slot |= 1u32 << bit;
                }
            }
        }
    }

    pub(crate) fn or_single_weight_intersection_to_buf(
        &self,
        start: u32,
        end: u32,
        single_tokens: &RangeSetBlaze<u32>,
        other: &crate::ds::weight::Weight,
        buf: &mut [u32],
    ) {
        if self.internal_token_dense_words == 0
            || self.weight_token_dense_masks.is_empty()
            || self.internal_token_buf_masks.is_empty()
        {
            self.or_single_weight_intersection_to_buf_sparse(start, end, single_tokens, other, buf);
            return;
        }

        let single_dense = self.dense_words_from_internal_set(single_tokens);
        let mut seen_dense_keys = Vec::<usize>::new();

        for (range, other_tokens) in other.0.range_values() {
            if end < *range.start() || *range.end() < start {
                continue;
            }

            let key = std::sync::Arc::as_ptr(other_tokens) as usize;
            if seen_dense_keys.contains(&key) {
                continue;
            }
            seen_dense_keys.push(key);

            if single_tokens == other_tokens.as_ref() {
                self.or_internal_token_set_to_buf(single_tokens, buf);
                continue;
            }

            let Some(other_dense) = self.weight_token_dense_masks.get(&key) else {
                self.or_single_weight_intersection_to_buf_sparse(start, end, single_tokens, other, buf);
                return;
            };

            self.or_dense_intersection_to_buf(single_dense.as_ref(), other_dense.as_ref(), buf);
        }
    }

    pub fn start(&self) -> ConstraintState<'_> {
        crate::ds::weight::clear_stale_weights();
        crate::ds::weight::clear_weight_op_caches();

        
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
        let possible_matches = self.possible_matches
            .get(&tokenizer_state)
            .cloned()
            .unwrap_or_default();
        possible_matches
            .into_iter()
            .map(|(terminal, internal_tokens)| {
                (terminal, self.expand_internal_token_set(&internal_tokens))
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

    pub(crate) fn internal_token_universe(&self) -> RangeSetBlaze<u32> {
        if self.internal_token_to_tokens.is_empty() {
            let Some(max_token_id) = self.token_bytes.keys().next_back().copied() else {
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
    ) -> Option<&BTreeMap<TerminalID, RangeSetBlaze<u32>>> {
        self.possible_matches.get(&tokenizer_state)
    }

}
