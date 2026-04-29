use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::labels::{encode_positive_label, DEFAULT_LABEL};
use crate::ds::leveled_gss::{LeveledGSS, Merge};
use crate::ds::weight::Weight;
use crate::runtime::state::{ConstraintState, MaskCacheData};
use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

type DenseTokenMaskCache = FxHashMap<usize, Box<[u64]>>;
type DenseMaskGSS = LeveledGSS<u32, DenseMaskAcc>;
type MaskQueue = BTreeMap<u32, FxHashMap<u32, DenseMaskGSS>>;

/// Compatibility return type for [`ConstraintState::fill_mask_profiled`].
///
/// The old implementation reported many detailed profiling counters. This file
/// intentionally does not keep that profiling machinery. `fill_mask_profiled`
/// now behaves as a lightweight compatibility shim: `total_ns` is meaningful,
/// `cache_hit_ns`/`cache_miss_ns` are coarse whole-call timings, and the
/// remaining fields are left at their default values.
#[derive(Debug, Default, Clone)]
pub struct FillMaskTimings {
    pub cache_hit_ns: u64,
    pub cache_miss_ns: u64,
    pub seed_ns: u64,
    pub bfs_ns: u64,
    pub convert_ns: u64,
    pub total_ns: u64,

    pub bfs_queue_pops: u64,
    pub bfs_states_processed: u64,
    pub weight_intersections: u64,
    pub weight_pruned: u64,
    pub convert_incremental: bool,
    pub convert_delta_tokens: u64,
    pub seed_tokenizer_states: u64,
    pub seed_chain_hits: u64,
    pub seed_chain_misses: u64,

    pub bfs_fast_path_ns: u64,
    pub bfs_standard_path_ns: u64,
    pub bfs_fw_merge_ns: u64,
}

/// Dense bitmap accumulator used while walking the parser DWA.
///
/// The key is an internal tokenizer-state id. The value is a dense bitmap of
/// allowed internal tokens for that tokenizer state.
#[derive(Clone, PartialEq, Eq, Hash)]
struct DenseMaskAcc(BTreeMap<u32, Arc<[u64]>>);

impl DenseMaskAcc {
    fn from_dense(tsid: u32, dense: Vec<u64>) -> Option<Self> {
        if dense.iter().all(|&word| word == 0) {
            return None;
        }

        let dense: Arc<[u64]> = dense.into();
        Some(Self(BTreeMap::from([(tsid, dense)])))
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    fn bit_range_mask(lo_bit: usize, hi_bit: usize) -> u64 {
        debug_assert!(lo_bit <= hi_bit);
        debug_assert!(hi_bit < 64);

        let high_mask = if hi_bit == 63 {
            !0u64
        } else {
            (1u64 << (hi_bit + 1)) - 1
        };

        let low_mask = if lo_bit == 0 {
            0
        } else {
            (1u64 << lo_bit) - 1
        };

        high_mask & !low_mask
    }

    fn for_each_token_range_word<F>(tokens: &RangeSetBlaze<u32>, word_limit: usize, mut f: F)
    where
        F: FnMut(usize, u64),
    {
        if word_limit == 0 {
            return;
        }

        let max_token_exclusive = word_limit.saturating_mul(64);
        if max_token_exclusive == 0 {
            return;
        }

        for range in tokens.ranges() {
            let lo = *range.start() as usize;
            if lo >= max_token_exclusive {
                continue;
            }

            let hi = (*range.end() as usize).min(max_token_exclusive - 1);
            if lo > hi {
                continue;
            }

            let word_lo = lo / 64;
            let word_hi = hi / 64;

            for word_idx in word_lo..=word_hi {
                let lo_bit = if word_idx == word_lo { lo % 64 } else { 0 };
                let hi_bit = if word_idx == word_hi { hi % 64 } else { 63 };
                f(word_idx, Self::bit_range_mask(lo_bit, hi_bit));
            }
        }
    }

    fn intersect_dense_with_tokens(
        dense: &[u64],
        tokens: &RangeSetBlaze<u32>,
    ) -> Option<Arc<[u64]>> {
        if dense.is_empty() || tokens.is_empty() {
            return None;
        }

        let mut out = vec![0u64; dense.len()];
        let mut any = false;

        Self::for_each_token_range_word(tokens, dense.len(), |word_idx, token_mask| {
            let word = dense[word_idx] & token_mask;
            if word != 0 {
                out[word_idx] |= word;
                any = true;
            }
        });

        if any {
            Some(out.into())
        } else {
            None
        }
    }

    fn intersect_dense_with_token_set(
        dense: &[u64],
        token_set: &Arc<RangeSetBlaze<u32>>,
        precomputed: &DenseTokenMaskCache,
    ) -> Option<Arc<[u64]>> {
        let key = Arc::as_ptr(token_set) as usize;

        if let Some(mask) = precomputed.get(&key) {
            let mut out = vec![0u64; dense.len()];
            let mut any = false;

            for i in 0..dense.len() {
                let word = dense[i] & mask.get(i).copied().unwrap_or(0);
                if word != 0 {
                    any = true;
                }
                out[i] = word;
            }

            if any {
                Some(out.into())
            } else {
                None
            }
        } else {
            Self::intersect_dense_with_tokens(dense, token_set)
        }
    }

    fn or_dense_and_token_set_into(
        dense: &[u64],
        token_set: &Arc<RangeSetBlaze<u32>>,
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
    ) {
        let key = Arc::as_ptr(token_set) as usize;

        if let Some(mask) = precomputed.get(&key) {
            let n = dense.len().min(mask.len()).min(merged.len());
            for i in 0..n {
                merged[i] |= dense[i] & mask[i];
            }
        } else {
            let word_limit = dense.len().min(merged.len());
            Self::for_each_token_range_word(token_set, word_limit, |word_idx, token_mask| {
                merged[word_idx] |= dense[word_idx] & token_mask;
            });
        }
    }

    fn intersect_with_weight(
        &self,
        weight: &Weight,
        precomputed: &DenseTokenMaskCache,
    ) -> Option<Self> {
        if self.is_empty() {
            return None;
        }

        if weight.is_full() {
            return Some(self.clone());
        }

        let mut result = BTreeMap::new();

        for (&tsid, dense) in &self.0 {
            let Some(token_set) = weight.0.get(tsid) else {
                continue;
            };

            if let Some(intersection) =
                Self::intersect_dense_with_token_set(dense, token_set, precomputed)
            {
                result.insert(tsid, intersection);
            }
        }

        if result.is_empty() {
            None
        } else {
            Some(Self(result))
        }
    }

    fn or_into_merged(&self, merged: &mut [u64]) {
        for dense in self.0.values() {
            let n = dense.len().min(merged.len());
            for i in 0..n {
                merged[i] |= dense[i];
            }
        }
    }

    fn or_intersection_into_merged(
        &self,
        final_weight: &Weight,
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
    ) {
        if final_weight.is_full() {
            self.or_into_merged(merged);
            return;
        }

        for (&tsid, dense) in &self.0 {
            let Some(token_set) = final_weight.0.get(tsid) else {
                continue;
            };

            Self::or_dense_and_token_set_into(dense, token_set, precomputed, merged);
        }
    }
}

impl Merge for DenseMaskAcc {
    fn merge(&self, other: &Self) -> Self {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }

        let mut merged = self.0.clone();

        for (&tsid, other_dense) in &other.0 {
            merged
                .entry(tsid)
                .and_modify(|dense| {
                    let len = dense.len().max(other_dense.len());
                    let mut combined = vec![0u64; len];

                    for (i, &word) in dense.iter().enumerate() {
                        combined[i] |= word;
                    }
                    for (i, &word) in other_dense.iter().enumerate() {
                        combined[i] |= word;
                    }

                    *dense = combined.into();
                })
                .or_insert_with(|| Arc::clone(other_dense));
        }

        Self(merged)
    }
}

fn enqueue_gss(queue: &mut MaskQueue, target: u32, gss: DenseMaskGSS) {
    if gss.is_empty() {
        return;
    }

    let depth = gss.max_depth();

    queue
        .entry(depth)
        .or_default()
        .entry(target)
        .and_modify(|existing| {
            *existing = existing.merge(&gss);
        })
        .or_insert(gss);
}

fn enqueue_weighted_transition(
    queue: &mut MaskQueue,
    popped: &DenseMaskGSS,
    target: u32,
    weight: &Weight,
    precomputed: &DenseTokenMaskCache,
) {
    if weight.is_full() {
        enqueue_gss(queue, target, popped.clone());
        return;
    }

    let pruned = popped.apply_and_prune_no_promote(|allowed| {
        allowed.intersect_with_weight(weight, precomputed)
    });

    enqueue_gss(queue, target, pruned);
}

fn enqueue_parser_state_transition(
    queue: &mut MaskQueue,
    fast_transitions: &FxHashMap<i32, (u32, Weight)>,
    parser_state: u32,
    popped: &DenseMaskGSS,
    precomputed: &DenseTokenMaskCache,
) {
    let positive_label = encode_positive_label(parser_state);

    let Some((target, weight)) = fast_transitions
        .get(&positive_label)
        .or_else(|| fast_transitions.get(&DEFAULT_LABEL))
    else {
        return;
    };

    enqueue_weighted_transition(queue, popped, *target, weight, precomputed);
}

fn update_eos_mask(buf: &mut [u32], eos_token_id: Option<u32>, is_complete: bool) {
    let Some(eos_token_id) = eos_token_id else {
        return;
    };

    let word = eos_token_id as usize / 32;
    let bit = eos_token_id as usize % 32;

    let Some(slot) = buf.get_mut(word) else {
        return;
    };

    *slot &= !(1u32 << bit);

    if is_complete {
        *slot |= 1u32 << bit;
    }
}

impl<'a> ConstraintState<'a> {
    fn try_fill_mask_from_cache(&self, buf: &mut [u32]) -> bool {
        let cache = self.mask_cache.lock().unwrap();

        let Some(cache_data) = cache.as_ref() else {
            return false;
        };

        if cache_data.generation != self.generation {
            return false;
        }

        buf.copy_from_slice(&cache_data.mask);
        true
    }

    fn store_mask_cache(&self, buf: &[u32], merged_dense: &[u64]) {
        let mut cache = self.mask_cache.lock().unwrap();

        match cache.as_mut() {
            Some(cache_data) => {
                cache_data.generation = self.generation;

                cache_data.mask.clear();
                cache_data.mask.extend_from_slice(buf);

                cache_data.merged_dense.clear();
                cache_data.merged_dense.extend_from_slice(merged_dense);
            }
            None => {
                *cache = Some(MaskCacheData {
                    generation: self.generation,
                    mask: buf.to_vec(),
                    merged_dense: merged_dense.to_vec(),
                });
            }
        }
    }

    fn terminals_disallowed_to_dense_acc(
        &self,
        terminals_disallowed: &TerminalsDisallowed,
        internal_tsid: u32,
    ) -> Option<DenseMaskAcc> {
        let universe = &self.constraint.seed_universe_dense;
        let terminal_masks = &self.constraint.seed_terminal_dense;

        let no_disallowed_terminals = terminals_disallowed.is_empty()
            || terminals_disallowed
                .values()
                .all(|disallowed| disallowed.is_empty());

        if no_disallowed_terminals {
            return DenseMaskAcc::from_dense(internal_tsid, universe.to_vec());
        }

        let mut dense = vec![0u64; universe.len()];

        for (&original_tokenizer_state, disallowed_in_state) in terminals_disallowed.iter() {
            let mut allowed_for_state = universe.to_vec();

            for &terminal_id in disallowed_in_state {
                if let Some(mask) = terminal_masks.get(&(original_tokenizer_state, terminal_id)) {
                    for (allowed_word, mask_word) in allowed_for_state.iter_mut().zip(mask.iter()) {
                        *allowed_word &= !mask_word;
                    }
                }
            }

            for (dense_word, allowed_word) in dense.iter_mut().zip(allowed_for_state.iter()) {
                *dense_word |= *allowed_word;
            }
        }

        DenseMaskAcc::from_dense(internal_tsid, dense)
    }

    fn merge_final_weight_to_internal(
        &self,
        final_weight: &Weight,
        acc: &DenseMaskAcc,
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
    ) {
        if final_weight.is_full() {
            acc.or_into_merged(merged);
        } else {
            acc.or_intersection_into_merged(final_weight, precomputed, merged);
        }
    }

    fn merge_final_weight_for_accs(
        &self,
        final_weight: &Weight,
        accs: &[DenseMaskAcc],
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
    ) {
        for acc in accs {
            self.merge_final_weight_to_internal(final_weight, acc, precomputed, merged);
        }
    }

    fn merge_final_weight_for_gss(
        &self,
        final_weight: &Weight,
        gss: &DenseMaskGSS,
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
    ) {
        gss.for_each_acc(|acc| {
            self.merge_final_weight_to_internal(final_weight, acc, precomputed, merged);
        });
    }

    fn seed_mask_queue_merged(
        &self,
        start_final_weight: Option<&Weight>,
        start_fast_transitions: &FxHashMap<i32, (u32, Weight)>,
        precomputed: &DenseTokenMaskCache,
        queue: &mut MaskQueue,
        merged: &mut [u64],
    ) {
        for (&tokenizer_state, gss) in &self.state {
            if gss.is_empty() {
                continue;
            }

            let internal_tsid = self.constraint.internal_tsid_for_state(tokenizer_state);

            let (decomposed, root_accs) =
                gss.apply_transform_and_decompose(|terminals_disallowed| {
                    self.terminals_disallowed_to_dense_acc(terminals_disallowed, internal_tsid)
                });

            if decomposed.is_empty() && root_accs.is_empty() {
                continue;
            }

            if let Some(final_weight) = start_final_weight {
                self.merge_final_weight_for_accs(final_weight, &root_accs, precomputed, merged);

                for (_, sub_gss) in &decomposed {
                    self.merge_final_weight_for_gss(final_weight, sub_gss, precomputed, merged);
                }
            }

            for (parser_state, popped) in &decomposed {
                enqueue_parser_state_transition(
                    queue,
                    start_fast_transitions,
                    *parser_state,
                    popped,
                    precomputed,
                );
            }
        }
    }

    fn fill_mask_uncached(&self, buf: &mut [u32]) {
        let parser_dwa = self.constraint.parser_dwa();

        if self.state.is_empty() || parser_dwa.states().is_empty() {
            buf.fill(0);
            update_eos_mask(buf, self.constraint.eos_token_id, self.is_complete());
            self.store_mask_cache(buf, &[]);
            return;
        }

        let precomputed = &self.constraint.weight_token_dense_masks;
        let dense_words = self.constraint.internal_token_dense_words;

        let mut merged = {
            let mut scratch = self.mask_scratch.lock().unwrap();
            std::mem::take(&mut scratch.merged_dense)
        };

        merged.clear();
        merged.resize(dense_words, 0);

        let mut queue = MaskQueue::new();

        let start_state = parser_dwa.start_state();
        let start_dwa_state = &parser_dwa.states()[start_state as usize];
        let start_fast_transitions = &self.constraint.dwa_fast_transitions[start_state as usize];

        self.seed_mask_queue_merged(
            start_dwa_state.final_weight.as_ref(),
            start_fast_transitions,
            precomputed,
            &mut queue,
            &mut merged,
        );

        while let Some((_depth, states_at_depth)) = queue.pop_last() {
            for (wa_state, gss) in states_at_depth {
                let dwa_state = &parser_dwa.states()[wa_state as usize];
                let fast_transitions = &self.constraint.dwa_fast_transitions[wa_state as usize];

                if let Some(final_weight) = &dwa_state.final_weight {
                    self.merge_final_weight_for_gss(final_weight, &gss, precomputed, &mut merged);
                }

                gss.for_each_decomposed(|parser_state, popped| {
                    enqueue_parser_state_transition(
                        &mut queue,
                        fast_transitions,
                        parser_state,
                        &popped,
                        precomputed,
                    );
                });
            }
        }

        buf.fill(0);
        self.constraint.or_internal_dense_to_buf(&merged, buf, true);
        update_eos_mask(buf, self.constraint.eos_token_id, self.is_complete());

        self.store_mask_cache(buf, &merged);

        let mut scratch = self.mask_scratch.lock().unwrap();
        scratch.merged_dense = merged;
        scratch.chain_merged_dense.clear();
    }

    pub fn mask(&self) -> Vec<u32> {
        let mut buf = vec![0u32; self.constraint.mask_len()];
        self.fill_mask(&mut buf);
        buf
    }

    pub fn fill_mask(&self, buf: &mut [u32]) {
        if self.try_fill_mask_from_cache(buf) {
            return;
        }

        self.fill_mask_uncached(buf);
    }

    pub fn fill_mask_timed_ns(&self, buf: &mut [u32]) -> u64 {
        let start = Instant::now();
        self.fill_mask(buf);
        start.elapsed().as_nanos() as u64
    }

    pub fn fill_mask_profiled(&self, buf: &mut [u32]) -> FillMaskTimings {
        let start = Instant::now();

        if self.try_fill_mask_from_cache(buf) {
            let total_ns = start.elapsed().as_nanos() as u64;
            return FillMaskTimings {
                cache_hit_ns: total_ns,
                total_ns,
                ..Default::default()
            };
        }

        self.fill_mask_uncached(buf);

        let total_ns = start.elapsed().as_nanos() as u64;
        FillMaskTimings {
            cache_miss_ns: total_ns,
            total_ns,
            ..Default::default()
        }
    }
}
