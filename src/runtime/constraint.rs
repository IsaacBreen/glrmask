use std::collections::BTreeMap;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::parser::{ParserGSS, TerminalsDisallowed};
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar_def::TerminalID;
use crate::ds::weight::Weight;

use super::state::ConstraintState;

pub(crate) type TokenizerStateID = u32;

pub(crate) fn bitmap_to_rangeset(words: &[u64]) -> RangeSetBlaze<u32> {
    let mut result = RangeSetBlaze::new();
    for (word_idx, &word) in words.iter().enumerate() {
        if word == 0 { continue; }
        let base = (word_idx as u32) * 64;
        let mut w = word;
        let mut pos = 0u32;
        while w != 0 {
            let zeros = w.trailing_zeros();
            pos += zeros;
            w >>= zeros;
            let ones = if w == u64::MAX { 64 - pos % 64 } else { (!w).trailing_zeros() };
            let run_start = base + pos;
            let run_end = base + pos + ones - 1;
            pos += ones;
            if ones < 64 { w >>= ones; } else { w = 0; }
            result.ranges_insert(run_start..=run_end);
        }
    }
    result
}
pub(crate) type PossibleMatchesByState =
    BTreeMap<TokenizerStateID, BTreeMap<TerminalID, Box<[u64]>>>;
type DenseWords = Box<[u64]>;
type InternalTokenBufMasks = Vec<(u16, u32)>;
type DenseWeightMaskCache = FxHashMap<usize, DenseWords>;
type SeedTerminalDenseMasks = FxHashMap<(u32, TerminalID), DenseWords>;
type FastDwaTransitions = Vec<FxHashMap<i32, (u32, Weight)>>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Constraint {
    pub(crate) parser_dwa: DWA,
    pub(crate) table: GLRTable,
    pub(crate) tokenizer: Tokenizer,
    #[serde(default)]
    pub(crate) ignore_terminal: Option<TerminalID>,

    #[serde(with = "crate::runtime::serde::serde_btreemap_rangeset")]
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
    #[serde(skip)]
    pub(crate) token_bytes_dense: Vec<Option<Box<[u8]>>>,

    /// Precomputed bitmask fragments for each internal token.
    /// `internal_token_buf_masks[i]` contains (word_index, or_mask) pairs
    /// for all original tokens that map to internal token `i`.
    #[serde(default)]
    pub(crate) internal_token_buf_masks: Vec<InternalTokenBufMasks>,
    /// Precomputed combined buf output for each group of 64 internal tokens.
    /// `word_group_buf_masks[w]` is the combined mask for internal tokens [w*64 .. (w+1)*64).
    /// Used as a fast path in `or_to_buf` when a dense word is all-ones (!0u64).
    #[serde(skip)]
    pub(crate) word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed buf output for the full internal token universe (OR of all word_group_buf_masks).
    #[serde(skip)]
    pub(crate) all_tokens_buf_mask: Box<[u32]>,
    #[serde(skip)]
    pub(crate) internal_token_dense_words: usize,
    #[serde(skip)]
    pub(crate) weight_token_dense_masks: DenseWeightMaskCache,
    /// Precomputed dense bitmask for the seed phase: for each (tokenizer_state, terminal_id),
    /// the dense bitmap of internal tokens that terminal covers in that state.
    #[serde(skip)]
    pub(crate) seed_terminal_dense: SeedTerminalDenseMasks,
    /// Dense bitmap of the full internal token universe.
    #[serde(skip)]
    pub(crate) seed_universe_dense: DenseWords,
    /// Fast DWA transition lookup (FxHashMap instead of BTreeMap).
    /// Built from parser_dwa.states at load/build time.
    #[serde(skip)]
    pub(crate) dwa_fast_transitions: FastDwaTransitions,
}

impl Constraint {
    pub(crate) fn rebuild_runtime_caches(&mut self) {
        let (internal_token_buf_masks, ((dense_mask_words, dense_masks), fast_transitions)) = rayon::join(
            || self.compute_buf_masks(),
            || {
                rayon::join(
                    || self.compute_dense_token_masks(),
                    || self.compute_fast_transitions(),
                )
            },
        );

        self.internal_token_buf_masks = internal_token_buf_masks;
        self.word_group_buf_masks = self.compute_word_group_buf_masks();
        self.all_tokens_buf_mask = self.compute_all_tokens_buf_mask();
        self.token_bytes_dense = Vec::new();
        self.internal_token_dense_words = dense_mask_words;
        self.weight_token_dense_masks = dense_masks;
        self.dwa_fast_transitions = fast_transitions;
        self.build_seed_dense_masks();
    }

    fn compute_buf_masks(&self) -> Vec<InternalTokenBufMasks> {
        if self.internal_token_to_tokens.is_empty() {
            return Vec::new();
        }

        self.internal_token_to_tokens
            .iter()
            .map(|originals| Self::build_internal_token_buf_mask(originals))
            .collect()
    }

    fn compute_word_group_buf_masks(&self) -> Vec<Box<[u32]>> {
        if self.internal_token_buf_masks.is_empty() {
            return Vec::new();
        }
        let buf_words = self.mask_len();
        let n_groups = self.internal_token_buf_masks.len().div_ceil(64);
        let mut groups = Vec::with_capacity(n_groups);
        for group_start in (0..self.internal_token_buf_masks.len()).step_by(64) {
            let mut combined = vec![0u32; buf_words];
            let group_end = (group_start + 64).min(self.internal_token_buf_masks.len());
            for token_masks in &self.internal_token_buf_masks[group_start..group_end] {
                for &(word_idx, mask) in token_masks {
                    combined[word_idx as usize] |= mask;
                }
            }
            groups.push(combined.into_boxed_slice());
        }
        groups
    }

    fn compute_all_tokens_buf_mask(&self) -> Box<[u32]> {
        let buf_words = self.mask_len();
        let mut combined = vec![0u32; buf_words];
        for group in &self.word_group_buf_masks {
            for (i, &mask) in group.iter().enumerate() {
                combined[i] |= mask;
            }
        }
        combined.into_boxed_slice()
    }

    fn compute_dense_token_bytes(&self) -> Vec<Option<Box<[u8]>>> {
        let Some(max_token_id) = self.max_original_token_id() else {
            return Vec::new();
        };

        let mut dense = vec![None; max_token_id as usize + 1];
        for (&token_id, bytes) in &self.token_bytes {
            dense[token_id as usize] = Some(bytes.clone().into_boxed_slice());
        }
        dense
    }

    fn compute_fast_transitions(&self) -> FastDwaTransitions {
        self.parser_dwa
            .states
            .iter()
            .map(|state| {
                state
                    .transitions
                    .iter()
                    .map(|(&label, (target, weight))| (label, (*target, weight.clone())))
                    .collect()
            })
            .collect()
    }

    fn compute_dense_token_masks(&self) -> (usize, DenseWeightMaskCache) {
        let internal_token_dense_words = self.internal_token_to_tokens.len().div_ceil(64);
        if internal_token_dense_words == 0 {
            return (0, DenseWeightMaskCache::default());
        }

        let mut dense_masks = DenseWeightMaskCache::default();
        for state in &self.parser_dwa.states {
            if let Some(final_weight) = &state.final_weight {
                self.collect_weight_dense_masks(
                    final_weight,
                    internal_token_dense_words,
                    &mut dense_masks,
                );
            }
            for (_, weight) in state.transitions.values() {
                self.collect_weight_dense_masks(weight, internal_token_dense_words, &mut dense_masks);
            }
        }

        (internal_token_dense_words, dense_masks)
    }

    /// Build precomputed bitmask fragments for each internal token.
    pub(crate) fn build_buf_masks(&mut self) {
        self.internal_token_buf_masks = self.compute_buf_masks();
        self.word_group_buf_masks = self.compute_word_group_buf_masks();
        self.all_tokens_buf_mask = self.compute_all_tokens_buf_mask();
    }

    pub(crate) fn build_dense_token_bytes(&mut self) {
        self.token_bytes_dense = self.compute_dense_token_bytes();
    }

    /// Build fast transition lookup tables from the DWA's BTreeMap transitions.
    pub(crate) fn build_fast_transitions(&mut self) {
        self.dwa_fast_transitions = self.compute_fast_transitions();
    }

    pub(crate) fn build_dense_token_masks(&mut self) {
        let (internal_token_dense_words, dense_masks) = self.compute_dense_token_masks();
        self.internal_token_dense_words = internal_token_dense_words;
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

        let universe = self.internal_token_universe();
        self.seed_universe_dense = self.dense_words_from_internal_set(&universe);

        self.seed_terminal_dense = self.build_seed_terminal_dense_masks();
    }

    fn collect_weight_dense_masks(
        &self,
        weight: &Weight,
        internal_token_dense_words: usize,
        dense_masks: &mut DenseWeightMaskCache,
    ) {
        for token_set in weight.unique_token_sets() {
            let key = token_set as *const RangeSetBlaze<u32> as usize;
            dense_masks
                .entry(key)
                .or_insert_with(|| self.dense_words_from_internal_set_with_words(token_set, internal_token_dense_words));
        }
    }

    fn dense_words_from_internal_set_with_words(
        &self,
        internal_tokens: &RangeSetBlaze<u32>,
        dense_word_count: usize,
    ) -> Box<[u64]> {
        let mut words = vec![0u64; dense_word_count];
        for internal_token in internal_tokens.iter() {
            let index = internal_token as usize / 64;
            let bit = internal_token as usize % 64;
            if let Some(word) = words.get_mut(index) {
                *word |= 1u64 << bit;
            }
        }
        words.into_boxed_slice()
    }

    fn dense_words_from_internal_set(&self, internal_tokens: &RangeSetBlaze<u32>) -> Box<[u64]> {
        self.dense_words_from_internal_set_with_words(internal_tokens, self.internal_token_dense_words)
    }

    pub(crate) fn or_dense_intersection_to_buf(
        &self,
        left_words: &[u64],
        right_words: &[u64],
        buf: &mut [u32],
    ) {
        for (word_index, (&left_word, &right_word)) in
            left_words.iter().zip(right_words.iter()).enumerate()
        {
            let overlap = left_word & right_word;
            if overlap == 0 {
                continue;
            }
            // Fast path: all 64 tokens in this word set → use precomputed group mask.
            if overlap == !0u64 {
                if let Some(group) = self.word_group_buf_masks.get(word_index) {
                    for (i, &mask) in group.iter().enumerate() {
                        buf[i] |= mask;
                    }
                    continue;
                }
            }
            let mut bits = overlap;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let internal_token = word_index * 64 + bit;
                self.or_internal_token_masks_to_buf(internal_token, buf);
                bits &= bits - 1;
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
                self.or_internal_token_masks_to_buf(internal_token as usize, buf);
            }
        } else {
            for token_id in internal_tokens.iter() {
                self.or_original_token_to_buf(token_id, buf);
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
        let state = self.initial_state_map();
        ConstraintState { constraint: self, state, buffers: Default::default() }
    }

    pub fn mask_len(&self) -> usize {
        self.token_bytes
            .keys()
            .max()
            .map(|token_id| (*token_id as usize / 32) + 1)
            .unwrap_or(0)
    }

    pub fn num_parser_states(&self) -> u32 {
        self.table.num_states
    }

    pub(crate) fn parser_dwa(&self) -> &DWA {
        &self.parser_dwa
    }

    pub(crate) fn possible_matches_for_state(
        &self,
        tokenizer_state: u32,
    ) -> BTreeMap<TerminalID, RangeSetBlaze<u32>> {
        let internal_tsid = self.internal_tsid_for_state(tokenizer_state);
        self.possible_matches
            .get(&internal_tsid)
            .map(|terminals| {
                terminals
                    .iter()
                    .map(|(&terminal, bitmap)| {
                        let internal_tokens = bitmap_to_rangeset(bitmap);
                        (terminal, self.expand_internal_token_set(&internal_tokens))
                    })
                    .collect()
            })
            .unwrap_or_default()
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

    pub(crate) fn possible_matches_for_state_internal(
        &self,
        tokenizer_state: u32,
    ) -> Option<BTreeMap<TerminalID, RangeSetBlaze<u32>>> {
        let internal_tsid = self.internal_tsid_for_state(tokenizer_state);
        self.possible_matches.get(&internal_tsid).map(|terminals| {
            terminals
                .iter()
                .map(|(&terminal, bitmap)| (terminal, bitmap_to_rangeset(bitmap)))
                .collect()
        })
    }

    fn build_internal_token_buf_mask(originals: &[u32]) -> InternalTokenBufMasks {
        let mut word_map = BTreeMap::<u16, u32>::new();
        for &original in originals {
            let word = (original / 32) as u16;
            let bit = original % 32;
            *word_map.entry(word).or_default() |= 1u32 << bit;
        }
        word_map.into_iter().collect()
    }

    fn max_original_token_id(&self) -> Option<u32> {
        self.token_bytes.keys().next_back().copied()
    }

    fn build_seed_terminal_dense_masks(&self) -> SeedTerminalDenseMasks {
        let mut terminal_masks = SeedTerminalDenseMasks::default();
        for (&internal_tsid, terminals) in &self.possible_matches {
            for (&terminal_id, bitmap) in terminals {
                terminal_masks.insert(
                    (internal_tsid, terminal_id),
                    bitmap.clone(),
                );
            }
        }
        terminal_masks
    }

    fn or_internal_token_masks_to_buf(&self, internal_token: usize, buf: &mut [u32]) {
        let masks = &self.internal_token_buf_masks[internal_token];
        for &(word_idx, mask) in masks {
            buf[word_idx as usize] |= mask;
        }
    }

    fn or_original_token_to_buf(&self, token_id: u32, buf: &mut [u32]) {
        let word = token_id as usize / 32;
        let bit = token_id as usize % 32;
        if let Some(slot) = buf.get_mut(word) {
            *slot |= 1u32 << bit;
        }
    }

    fn initial_state_map(&self) -> BTreeMap<u32, ParserGSS> {
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

    // Temporary diagnostics
    pub fn debug_original_token_to_internal(&self) -> Vec<u32> {
        self.original_token_to_internal.clone()
    }

    pub fn debug_internal_token_to_tokens(&self) -> Vec<Vec<u32>> {
        self.internal_token_to_tokens.clone()
    }

    /// Walk a byte sequence through the raw tokenizer DFA from every state.
    /// Returns a Vec indexed by initial state. Each entry is (final_state, finalizers, possible_futures).
    /// finalizers and possible_futures are Vec<u32> of group IDs that are set.
    pub fn debug_walk_dfa(&self, token_bytes: &[u8]) -> Vec<(u32, Vec<u32>, Vec<u32>)> {
        let dfa = &self.tokenizer.dfa;
        let num_states = dfa.states().len();
        let mut results = Vec::with_capacity(num_states);
        for initial_state in 0..num_states {
            let mut state = initial_state as u32;
            let mut dead = false;
            for &byte in token_bytes {
                let next = dfa.states()[state as usize].transitions.get(byte);
                if let Some(&ns) = next {
                    state = ns;
                } else {
                    dead = true;
                    break;
                }
            }
            if dead {
                results.push((u32::MAX, vec![], vec![]));
            } else {
                let finalizers: Vec<u32> = dfa.finalizers(state).iter().map(|x| x as u32).collect();
                let futures: Vec<u32> = dfa.possible_future_group_ids(state).iter().map(|x| x as u32).collect();
                results.push((state, finalizers, futures));
            }
        }
        results
    }
}
