use std::sync::Mutex;
use std::collections::BTreeMap;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::ParserGSS;
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
    /// Dense buf masks for "heavy" internal tokens (those with many buf entries).
    /// Indexed by internal token ID; None for light tokens.
    #[serde(skip)]
    pub(crate) heavy_token_dense_masks: Vec<Option<Box<[u32]>>>,
    /// Flattened contiguous array of all internal token buf mask entries.
    /// All tokens' (word_index, or_mask) pairs concatenated in token order.
    /// Improves cache locality vs separate Vec allocations per token.
    #[serde(skip)]
    pub(crate) internal_token_buf_flat: Box<[(u16, u32)]>,
    /// Offsets into `internal_token_buf_flat` for each internal token.
    /// `internal_token_buf_flat[offsets[i]..offsets[i+1]]` gives token i's entries.
    /// Length = n_internal + 1 (sentinel at end).
    #[serde(skip)]
    pub(crate) internal_token_buf_offsets: Box<[u32]>,
    /// Pre-computed total cost (sum of entry counts) for all internal tokens.
    /// Used to avoid O(n_internal) cost analysis in the convert phase.
    #[serde(skip)]
    pub(crate) total_internal_buf_cost: usize,
    /// Number of heavy internal tokens (those stored as dense masks).
    #[serde(skip)]
    pub(crate) n_heavy_tokens: usize,
    /// Total cost of all heavy tokens combined (n_heavy × buf_len).
    #[serde(skip)]
    pub(crate) heavy_total_cost: usize,
    /// Average cost per light token: (total_cost - heavy_total) / n_light.
    /// Pre-multiplied by 256 for fixed-point arithmetic to avoid float.
    #[serde(skip)]
    pub(crate) light_avg_cost_x256: usize,
}

/// Dense buf OR: `buf[i] |= mask[i]` for all i in min(buf.len(), mask.len()).
/// Processes u64 chunks for reduced loop overhead and better throughput.
#[inline(always)]
fn or_dense_buf(buf: &mut [u32], mask: &[u32]) {
    let n = buf.len().min(mask.len());
    // Process pairs of u32 as u64 for 2× fewer iterations.
    let n_pairs = n / 2;
    if n_pairs > 0 {
        let buf64 = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u64, n_pairs) };
        let mask64 = unsafe { std::slice::from_raw_parts(mask.as_ptr() as *const u64, n_pairs) };
        for (b, &m) in buf64.iter_mut().zip(mask64.iter()) {
            *b |= m;
        }
    }
    // Handle trailing u32.
    if n % 2 == 1 {
        buf[n - 1] |= mask[n - 1];
    }
}

/// Dense buf AND-NOT: `buf[i] &= !mask[i]` for all i in min(buf.len(), mask.len()).
/// Processes u64 chunks for reduced loop overhead and better throughput.
#[inline(always)]
fn andnot_dense_buf(buf: &mut [u32], mask: &[u32]) {
    let n = buf.len().min(mask.len());
    let n_pairs = n / 2;
    if n_pairs > 0 {
        let buf64 = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u64, n_pairs) };
        let mask64 = unsafe { std::slice::from_raw_parts(mask.as_ptr() as *const u64, n_pairs) };
        for (b, &m) in buf64.iter_mut().zip(mask64.iter()) {
            *b &= !m;
        }
    }
    if n % 2 == 1 {
        buf[n - 1] &= !mask[n - 1];
    }
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
        self.heavy_token_dense_masks = self.compute_heavy_token_dense_masks();
        let (flat, offsets) = Self::compute_flat_buf_masks(&self.internal_token_buf_masks);
        self.internal_token_buf_flat = flat;
        self.internal_token_buf_offsets = offsets;
        self.total_internal_buf_cost = Self::compute_total_internal_buf_cost(
            &self.internal_token_buf_offsets,
            &self.heavy_token_dense_masks,
            self.mask_len(),
        );

        // Precompute heavy token stats for fast path decision in convert.
        let buf_len = self.mask_len();
        let n_internal = if self.internal_token_buf_offsets.len() > 1 {
            self.internal_token_buf_offsets.len() - 1
        } else {
            0
        };
        self.n_heavy_tokens = self.heavy_token_dense_masks.iter().filter(|m| m.is_some()).count();
        self.heavy_total_cost = self.n_heavy_tokens * buf_len;
        let n_light = n_internal.saturating_sub(self.n_heavy_tokens);
        let light_total = self.total_internal_buf_cost.saturating_sub(self.heavy_total_cost);
        self.light_avg_cost_x256 = if n_light > 0 { (light_total * 256) / n_light } else { 0 };

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

    /// Build dense buf masks for internal tokens with many sparse entries.
    /// A token with >THRESHOLD entries benefits from a sequential 16KB scan
    /// instead of thousands of indexed read-modify-writes.
    fn compute_heavy_token_dense_masks(&self) -> Vec<Option<Box<[u32]>>> {
        let buf_words = self.mask_len();
        if buf_words == 0 {
            return Vec::new();
        }
        // Threshold: use dense when sparse entries > buf_words / 2.
        // Dense OR costs ~buf_words ops; sparse OR costs ~n_entries ops.
        // With buf in L1 cache (≤16KB), sparse random writes are fast,
        // so we only go dense when entries exceed half the buffer size.
        let threshold = buf_words / 2;
        self.internal_token_buf_masks
            .iter()
            .map(|sparse| {
                if sparse.len() > threshold {
                    let mut dense = vec![0u32; buf_words];
                    for &(word_idx, mask) in sparse {
                        dense[word_idx as usize] |= mask;
                    }
                    Some(dense.into_boxed_slice())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Flatten all per-token sparse entries into a single contiguous array
    /// with an offset table. Improves cache locality during convert phase.
    fn compute_flat_buf_masks(masks: &[InternalTokenBufMasks]) -> (Box<[(u16, u32)]>, Box<[u32]>) {
        let total: usize = masks.iter().map(|m| m.len()).sum();
        let mut flat = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(masks.len() + 1);
        for m in masks {
            offsets.push(flat.len() as u32);
            flat.extend_from_slice(m);
        }
        offsets.push(flat.len() as u32);
        (flat.into_boxed_slice(), offsets.into_boxed_slice())
    }

    /// Pre-compute total cost for all internal tokens (sum of entry counts,
    /// with heavy tokens counted at buf_len).
    fn compute_total_internal_buf_cost(
        offsets: &[u32],
        heavy: &[Option<Box<[u32]>>],
        buf_len: usize,
    ) -> usize {
        let n_internal = if offsets.len() > 1 { offsets.len() - 1 } else { 0 };
        let mut total: usize = 0;
        for idx in 0..n_internal {
            if idx < heavy.len() && heavy[idx].is_some() {
                total += buf_len;
            } else {
                total += (offsets[idx + 1] - offsets[idx]) as usize;
            }
        }
        total
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

    pub fn start(&self) -> ConstraintState<'_> {
        let state = self.initial_state_map();
        ConstraintState { constraint: self, state, buffers: Default::default(), generation: 0, mask_cache: Mutex::new(None) }
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

    /// Convert a merged internal token dense bitmap to the output buffer.
    /// Uses a contiguous flat entry array for cache-friendly sequential access,
    /// with word_group fast paths for fully-set 64-bit words and heavy token
    /// dense masks for tokens with many buf entries.
    pub(crate) fn or_internal_dense_to_buf(&self, dense: &[u64], buf: &mut [u32]) {
        let all_mask = &self.all_tokens_buf_mask;
        let offsets = &self.internal_token_buf_offsets;
        let flat = &self.internal_token_buf_flat;
        let n_internal = if offsets.len() > 1 { offsets.len() - 1 } else { 0 };

        if n_internal == 0 || dense.is_empty() {
            return;
        }

        // Count set bits to choose path.
        let n_set: usize = dense.iter().map(|w| w.count_ones() as usize).sum();

        // Super-fast path: all internal tokens set → OR all_tokens_buf_mask.
        if n_set >= n_internal && !all_mask.is_empty() {
            or_dense_buf(buf, all_mask);
            return;
        }

        if n_set == 0 {
            return;
        }

        // Count total sparse entries for set vs missing tokens to pick the cheaper path.
        // Heavy tokens use dense masks — count them at dense cost (buf.len() per token).
        // Use precomputed averages + O(n_heavy) heavy check for fast path decision.
        let heavy = &self.heavy_token_dense_masks;
        let buf_len = buf.len();
        let word_groups = &self.word_group_buf_masks;
        let n_missing = n_internal - n_set;

        // Fast O(n_heavy) estimation: count heavy set tokens, estimate rest from average.
        let n_heavy_set = if self.n_heavy_tokens > 0 {
            let n_heavy_check = heavy.len().min(n_internal);
            let mut count = 0usize;
            for idx in 0..n_heavy_check {
                if heavy[idx].is_some() {
                    let wi = idx / 64;
                    let bit = idx % 64;
                    if wi < dense.len() && (dense[wi] >> bit) & 1 != 0 {
                        count += 1;
                    }
                }
            }
            count
        } else {
            0
        };
        let n_light_set = n_set.saturating_sub(n_heavy_set);
        let n_heavy_missing = self.n_heavy_tokens.saturating_sub(n_heavy_set);

        let est_set_cost = n_heavy_set * buf_len
            + (n_light_set * self.light_avg_cost_x256) / 256;
        let est_missing_cost = n_heavy_missing * buf_len
            + (n_missing.saturating_sub(n_heavy_missing) * self.light_avg_cost_x256) / 256;
        // Complement path overhead: 16KB all_mask copy ≈ buf.len() entry-equivalent writes.
        let copy_overhead = buf_len;

        if !all_mask.is_empty() && est_missing_cost + copy_overhead < est_set_cost {
            // Complement-sparse path: start from all_tokens, subtract missing tokens.
            or_dense_buf(buf, all_mask);
            for (wi, &w) in dense.iter().enumerate() {
                if wi * 64 >= n_internal {
                    break;
                }
                let remaining = n_internal - wi * 64;
                let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
                let missing = !w & valid_mask;
                let mut bits = missing;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let internal_token = wi * 64 + bit;
                    if internal_token < heavy.len() {
                        if let Some(ref dense_mask) = heavy[internal_token] {
                            andnot_dense_buf(buf, dense_mask);
                            bits &= bits - 1;
                            continue;
                        }
                    }
                    let start = offsets[internal_token] as usize;
                    let end = offsets[internal_token + 1] as usize;
                    for &(buf_word, mask) in &flat[start..end] {
                        buf[buf_word as usize] &= !mask;
                    }
                    bits &= bits - 1;
                }
            }
        } else {
            // Normal path: process sparse light tokens and dense heavy tokens.
            for (wi, &w) in dense.iter().enumerate() {
                if w == 0 {
                    continue;
                }
                // Word-group fast path: all 64 internal tokens in this word are set.
                if w == !0u64 {
                    if let Some(group) = word_groups.get(wi) {
                        or_dense_buf(buf, group);
                        continue;
                    }
                }
                let mut bits = w;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let internal_token = wi * 64 + bit;
                    if internal_token < n_internal {
                        if internal_token < heavy.len() {
                            if let Some(ref dense_mask) = heavy[internal_token] {
                                or_dense_buf(buf, dense_mask);
                                bits &= bits - 1;
                                continue;
                            }
                        }
                        let start = offsets[internal_token] as usize;
                        let end = offsets[internal_token + 1] as usize;
                        for &(buf_word, mask) in &flat[start..end] {
                            buf[buf_word as usize] |= mask;
                        }
                    }
                    bits &= bits - 1;
                }
            }
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

    /// Return action table entries for a parser state.
    /// Each entry is (terminal_id, action_debug_str).
    pub fn debug_actions_for_state(&self, state: u32) -> Vec<(u32, String)> {
        if let Some(actions) = self.table.action.get(state as usize) {
            actions.iter().map(|(&tid, action)| {
                (tid, format!("{:?}", action))
            }).collect()
        } else {
            Vec::new()
        }
    }

    /// Return rule info: (lhs_nonterminal, rhs_length, rhs_debug).
    pub fn debug_rules(&self) -> Vec<(u32, usize, String)> {
        self.table.rules.iter().map(|r| {
            (r.lhs, r.rhs.len(), format!("{:?}", r.rhs))
        }).collect()
    }

    pub fn debug_num_terminals(&self) -> u32 {
        self.table.num_terminals
    }

    pub fn debug_num_states(&self) -> u32 {
        self.table.num_states
    }
}
