//! Runtime-cache construction for compiled artifacts.
//!
//! The functions in this file rebuild all `#[serde(skip)]` data from the
//! immutable compiled artifact.  They are called after compilation finalization
//! and after deserialization.  No mask or commit algorithm should depend on how
//! these caches are built; they should only consume the resulting fields.

use std::collections::BTreeMap;
use std::sync::Arc;

use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::grammar::flat::TerminalID;
use crate::runtime::bitmask_ops::{copy_dense_buf, or_dense_buf, or_sparse_buf_entries};
use crate::runtime::artifact::dense::empty_dense_words;
use crate::runtime::token_space::final_mask_mapping::{
    DenseToBufProfileStats,
    FinalMaskMapping,
};
use crate::sets::weight::Weight;

use super::cache_types::{
    DenseWeightBufMaskCache,
    DenseWeightMaskCache,
    FastDwaTransitions,
    FastTokenizerTransitions,
    InternalTokenBufMasks,
    SeedStateBufMasks,
    SeedStateDenseMasks,
    SeedTerminalDenseMasks,
    SparseWeightBufMaskCache,
};
use super::dense::DenseWords;
use super::Constraint;

impl Constraint {
    pub(crate) fn rebuild_runtime_caches_impl(&mut self) {
        self.table.rebuild_guarded_shift_index();
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let total_started_at = profile.then(std::time::Instant::now);
        let primary_started_at = profile.then(std::time::Instant::now);
        let (
            internal_token_buf_masks,
            tokenizer_fast_transitions,
            (dense_mask_words, dense_masks),
            fast_transitions,
        ) = if rayon::current_num_threads() == 1 {
            (
                self.compute_buf_masks(),
                self.compute_tokenizer_fast_transitions(),
                self.compute_dense_token_masks(),
                self.compute_fast_transitions(),
            )
        } else {
            let ((internal_token_buf_masks, tokenizer_fast_transitions), ((dense_mask_words, dense_masks), fast_transitions)) = rayon::join(
                || rayon::join(
                    || self.compute_buf_masks(),
                    || self.compute_tokenizer_fast_transitions(),
                ),
                || rayon::join(
                    || self.compute_dense_token_masks(),
                    || self.compute_fast_transitions(),
                ),
            );
            (
                internal_token_buf_masks,
                tokenizer_fast_transitions,
                (dense_mask_words, dense_masks),
                fast_transitions,
            )
        };
        let primary_ms = primary_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        self.internal_token_buf_masks = internal_token_buf_masks;
        self.word_group_buf_masks = Vec::new();
        let block_started_at = profile.then(std::time::Instant::now);
        let block_masks = if rayon::current_num_threads() == 1 {
            (
                self.compute_token_block_sparse_masks(64),
                self.compute_token_block_sparse_masks(4),
                self.compute_token_block_sparse_masks(8),
            )
        } else {
            let (word, (quad, byte)) = rayon::join(
                || self.compute_token_block_sparse_masks(64),
                || rayon::join(
                    || self.compute_token_block_sparse_masks(4),
                    || self.compute_token_block_sparse_masks(8),
                ),
            );
            (word, quad, byte)
        };
        let (
            (word_group_sparse_masks, word_group_sparse_total_entries, word_group_sparse_max_entries),
            (quad_group_sparse_masks, _, _),
            (byte_group_sparse_masks, _, _),
        ) = block_masks;
        self.word_group_sparse_masks = word_group_sparse_masks;
        self.word_group_prefix_buf_masks = self.compute_word_group_prefix_buf_masks();
        self.word_group_sparse_prefix_entries =
            Self::compute_sparse_entry_prefix(&self.word_group_sparse_masks);
        self.quad_group_sparse_masks = quad_group_sparse_masks;
        self.byte_group_sparse_masks = byte_group_sparse_masks;
        self.word_group_sparse_total_entries = word_group_sparse_total_entries;
        self.word_group_sparse_max_entries = word_group_sparse_max_entries;
        let block_ms = block_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let derived_started_at = profile.then(std::time::Instant::now);
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        self.pair_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(2);
        let pair_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        self.quad_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(4);
        let quad_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        self.super_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(8);
        let super_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        self.mega_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(16);
        let mega_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        self.giga_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(32);
        let giga_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        self.all_tokens_buf_mask = self.compute_all_tokens_buf_mask();
        let all_tokens_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        self.json_escape_prefix_buf_mask = self.compute_json_escape_prefix_buf_mask();
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        self.heavy_token_dense_masks = self.compute_heavy_token_dense_masks();
        let heavy_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        let (flat, offsets) = Self::compute_flat_buf_masks(&self.internal_token_buf_masks);
        self.internal_token_buf_flat = flat;
        self.internal_token_buf_offsets = offsets;
        let flat_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let derived_piece_started_at = profile.then(std::time::Instant::now);
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
        self.heavy_token_indices = self.heavy_token_dense_masks.iter().enumerate().filter_map(|(i, m)| if m.is_some() { Some(i) } else { None }).collect();
        self.heavy_total_cost = self.heavy_token_indices.len() * buf_len;
        self.internal_token_buf_op_costs = Self::compute_internal_token_buf_op_costs(
            &self.internal_token_buf_offsets,
            &self.heavy_token_dense_masks,
            buf_len,
        );
        self.word_group_buf_op_costs =
            Self::compute_word_group_buf_op_costs(&self.internal_token_buf_op_costs);
        self.final_mask_mapping = FinalMaskMapping::default();
        let costs_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let n_light = n_internal.saturating_sub(self.heavy_token_indices.len());
        let light_total = self.total_internal_buf_cost.saturating_sub(self.heavy_total_cost);
        self.light_avg_cost_x256 = if n_light > 0 { (light_total * 256) / n_light } else { 0 };

        self.token_bytes_dense = Vec::new();
        self.internal_token_dense_words = dense_mask_words;
        self.weight_token_dense_masks = dense_masks;
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        self.weight_token_buf_masks = self.compute_weight_token_buf_masks();
        let weight_buf_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        self.weight_token_sparse_buf_masks = self.compute_weight_token_sparse_buf_masks();
        let weight_sparse_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        self.dwa_fast_transitions = fast_transitions;
        self.tokenizer_fast_transitions = tokenizer_fast_transitions;
        let derived_ms = derived_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let seed_started_at = profile.then(std::time::Instant::now);
        self.build_seed_dense_masks();
        let seed_ms = seed_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][runtime_finalize_derived] pair_ms={:.3} quad_ms={:.3} super_ms={:.3} mega_ms={:.3} giga_ms={:.3} all_tokens_ms={:.3} heavy_ms={:.3} flat_ms={:.3} costs_ms={:.3} weight_buf_ms={:.3} weight_sparse_ms={:.3} final_weight_sets={}",
                pair_ms,
                quad_ms,
                super_ms,
                mega_ms,
                giga_ms,
                all_tokens_ms,
                heavy_ms,
                flat_ms,
                costs_ms,
                weight_buf_ms,
                weight_sparse_ms,
                self.weight_token_buf_masks.len(),
            );
            eprintln!(
                "[glrmask/profile][runtime_finalize] primary_ms={:.3} block_masks_ms={:.3} derived_masks_ms={:.3} seed_dense_ms={:.3} total_ms={:.3}",
                primary_ms,
                block_ms,
                derived_ms,
                seed_ms,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    fn compute_tokenizer_fast_transitions(&self) -> FastTokenizerTransitions {
        let build = |state| {
                let dfa_state = &self.tokenizer.dfa.states()[state as usize];
                let mut flat = Box::new([u32::MAX; 256]);
                for (byte, &target) in dfa_state.transitions.iter() {
                    flat[byte as usize] = target;
                }
                flat
            };
        if rayon::current_num_threads() == 1 {
            (0..self.tokenizer.num_states()).map(build).collect()
        } else {
            (0..self.tokenizer.num_states()).into_par_iter().map(build).collect()
        }
    }

    fn compute_buf_masks(&self) -> Vec<InternalTokenBufMasks> {
        if self.internal_token_to_tokens.is_empty() {
            return Vec::new();
        }

        if !self.original_token_to_internal.is_empty() {
            let mut masks = vec![Vec::<(u16, u32)>::new(); self.internal_token_to_tokens.len()];
            for (original, &internal) in self.original_token_to_internal.iter().enumerate() {
                if internal == u32::MAX {
                    continue;
                }
                let internal = internal as usize;
                let Some(mask) = masks.get_mut(internal) else {
                    continue;
                };
                let word = (original as u32 / 32) as u16;
                let bit = original as u32 % 32;
                if let Some((last_word, last_mask)) = mask.last_mut() {
                    if *last_word == word {
                        *last_mask |= 1u32 << bit;
                        continue;
                    }
                }
                mask.push((word, 1u32 << bit));
            }
            return masks;
        }

        if rayon::current_num_threads() == 1 {
            self.internal_token_to_tokens
                .iter()
                .map(|originals| Self::build_internal_token_buf_mask(originals))
                .collect()
        } else {
            self.internal_token_to_tokens
                .par_iter()
                .map(|originals| Self::build_internal_token_buf_mask(originals))
                .collect()
        }
    }

    fn compute_token_block_sparse_masks(&self, block_size: usize) -> (Vec<InternalTokenBufMasks>, usize, usize) {
        if self.internal_token_buf_masks.is_empty() {
            return (Vec::new(), 0, 0);
        }
        let n_groups = self.internal_token_buf_masks.len().div_ceil(block_size);
        let mask_words = self.mask_len();
        let build_group = |group_id: usize| {
                let group_start = group_id * block_size;
                let group_end = (group_start + block_size).min(self.internal_token_buf_masks.len());
                let mut dense = vec![0u32; mask_words];
                let mut touched = Vec::<u16>::new();
                for token_masks in &self.internal_token_buf_masks[group_start..group_end] {
                    for &(word_idx, mask) in token_masks {
                        let slot = &mut dense[word_idx as usize];
                        if *slot == 0 {
                            touched.push(word_idx);
                        }
                        *slot |= mask;
                    }
                }
                touched.sort_unstable();
                touched
                    .into_iter()
                    .map(|word_idx| (word_idx, dense[word_idx as usize]))
                    .collect()
            };
        let groups: Vec<InternalTokenBufMasks> = if rayon::current_num_threads() == 1 {
            (0..n_groups).map(build_group).collect()
        } else {
            (0..n_groups).into_par_iter().map(build_group).collect()
        };
        let total_entries = groups.iter().map(Vec::len).sum();
        let max_entries = groups.iter().map(Vec::len).max().unwrap_or(0);
        (groups, total_entries, max_entries)
    }

    fn compute_sliding_word_group_dense_masks(&self, word_group_len: usize) -> Vec<Box<[u32]>> {
        if self.internal_token_buf_masks.is_empty() || word_group_len == 0 {
            return Vec::new();
        }
        let n_word_groups = self.internal_token_buf_masks.len().div_ceil(64);
        let mask_words = self.mask_len();
        let build_group = |word_group_start: usize| {
            let group_start = word_group_start * 64;
            let group_end = (group_start + word_group_len * 64).min(self.internal_token_buf_masks.len());
            let mut dense = vec![0u32; mask_words];
            for token_masks in &self.internal_token_buf_masks[group_start..group_end] {
                for &(word_idx, mask) in token_masks {
                    dense[word_idx as usize] |= mask;
                }
            }
            dense.into_boxed_slice()
        };
        if rayon::current_num_threads() == 1 {
            (0..n_word_groups).map(build_group).collect()
        } else {
            (0..n_word_groups).into_par_iter().map(build_group).collect()
        }
    }

    fn compute_all_tokens_buf_mask(&self) -> Box<[u32]> {
        let buf_words = self.mask_len();
        let mut combined = vec![0u32; buf_words];
        for group in &self.word_group_sparse_masks {
            for &(word_idx, mask) in group {
                combined[word_idx as usize] |= mask;
            }
        }
        combined.into_boxed_slice()
    }

    pub(crate) fn token_starts_json_escape_prefix(bytes: &[u8]) -> bool {
        bytes.len() >= 2
            && bytes[0] == b'\\'
            && matches!(bytes[1], b'"' | b'\\' | b'b' | b'f' | b'n' | b'r' | b't')
    }

    fn compute_json_escape_prefix_buf_mask(&self) -> Box<[u32]> {
        let mut mask = vec![0u32; self.mask_len()];
        for (&token_id, bytes) in self.token_bytes.iter() {
            if Self::token_starts_json_escape_prefix(bytes) {
                let word = token_id as usize / 32;
                let bit = token_id as usize % 32;
                if let Some(slot) = mask.get_mut(word) {
                    *slot |= 1u32 << bit;
                }
            }
        }
        mask.into_boxed_slice()
    }

    fn compute_word_group_prefix_buf_masks(&self) -> Vec<Box<[u32]>> {
        let buf_words = self.mask_len();
        let mut prefixes = Vec::with_capacity(self.word_group_sparse_masks.len() + 1);
        let mut current = vec![0u32; buf_words];
        prefixes.push(current.clone().into_boxed_slice());
        for group in &self.word_group_sparse_masks {
            for &(word_idx, mask) in group {
                current[word_idx as usize] |= mask;
            }
            prefixes.push(current.clone().into_boxed_slice());
        }
        prefixes
    }

    fn compute_sparse_entry_prefix(groups: &[InternalTokenBufMasks]) -> Vec<usize> {
        let mut prefix = Vec::with_capacity(groups.len() + 1);
        let mut total = 0usize;
        prefix.push(0);
        for group in groups {
            total += group.len();
            prefix.push(total);
        }
        prefix
    }

    fn dense_words_hash(words: &[u64]) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for &word in words {
            hash ^= word;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^ (words.len() as u64)
    }

    fn compute_seed_state_hashes(
        seed_state_dense: &[DenseWords],
    ) -> FxHashMap<u64, Vec<usize>> {
        let mut by_hash: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
        for (idx, dense) in seed_state_dense.iter().enumerate() {
            by_hash
                .entry(Self::dense_words_hash(dense))
                .or_default()
                .push(idx);
        }
        by_hash
    }

    pub(crate) fn seed_state_index_for_dense(&self, dense: &[u64]) -> Option<usize> {
        let hash = Self::dense_words_hash(dense);
        let candidates = self.seed_state_by_dense_hash.get(&hash)?;
        candidates
            .iter()
            .copied()
            .find(|&idx| self.seed_state_dense.get(idx).is_some_and(|seed| seed.as_ref() == dense))
    }

    #[inline(always)]
    pub(crate) fn or_seed_state_dense_to_buf(&self, seed_idx: usize, buf: &mut [u32]) -> bool {
        let Some(Some(mask)) = self.seed_state_buf_masks.get(seed_idx) else {
            return false;
        };
        or_dense_buf(buf, mask);
        true
    }

    #[inline(always)]
    pub(crate) fn has_seed_state_buf_mask(&self, seed_idx: usize) -> bool {
        self.seed_state_buf_masks
            .get(seed_idx)
            .is_some_and(Option::is_some)
    }

    pub(crate) fn or_seed_dense_token_set_to_buf(
        &self,
        seed_idx: usize,
        token_set: &Arc<RangeSetBlaze<u32>>,
        buf: &mut [u32],
    ) -> bool {
        let Some(seed_dense) = self.seed_state_dense.get(seed_idx) else {
            return false;
        };
        if seed_dense.is_empty() || token_set.is_empty() {
            return true;
        }

        let mut token_words = vec![0u64; seed_dense.len()];
        let max_token_exclusive = token_words.len().saturating_mul(64);
        for range in token_set.ranges() {
            let lo = *range.start() as usize;
            if lo >= max_token_exclusive {
                continue;
            }
            let hi = (*range.end() as usize).min(max_token_exclusive - 1);
            let word_lo = lo / 64;
            let word_hi = hi / 64;
            for wi in word_lo..=word_hi {
                let lo_bit = if wi == word_lo { lo % 64 } else { 0 };
                let hi_bit = if wi == word_hi { hi % 64 } else { 63 };
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
                token_words[wi] |= high_mask & !low_mask;
            }
        }

        let n_internal = self.internal_token_to_tokens.len();
        let mut stats = DenseToBufProfileStats::default();
        let mut wi = 0usize;
        while wi < seed_dense.len() && wi * 64 < n_internal {
            let remaining = n_internal - wi * 64;
            let valid_mask = if remaining >= 64 {
                !0u64
            } else {
                (1u64 << remaining) - 1
            };
            let bits = seed_dense[wi] & token_words[wi] & valid_mask;
            if bits == 0 {
                wi += 1;
                continue;
            }
            if bits == valid_mask && remaining >= 64 {
                let run_start = wi;
                wi += 1;
                while wi < seed_dense.len() && wi * 64 < n_internal {
                    let remaining = n_internal - wi * 64;
                    if remaining < 64 || (seed_dense[wi] & token_words[wi]) != !0u64 {
                        break;
                    }
                    wi += 1;
                }
                self.or_full_internal_word_run_to_buf(run_start, wi, buf, &mut stats);
                continue;
            }
            let mut remaining_bits = bits;
            while remaining_bits != 0 {
                let bit = remaining_bits.trailing_zeros() as usize;
                self.or_internal_token_to_buf_fast(
                    wi * 64 + bit,
                    buf,
                    &mut stats.normal_sparse_entries,
                );
                remaining_bits &= remaining_bits - 1;
            }
            wi += 1;
        }
        true
    }

    #[inline(always)]
    pub(crate) fn or_weight_token_set_to_buf_if_contained(
        &self,
        dense: &[u64],
        token_set: &Arc<RangeSetBlaze<u32>>,
        buf: &mut [u32],
    ) -> bool {
        let key = Arc::as_ptr(token_set) as usize;
        let Some(mask) = self.weight_token_buf_masks.get(&key) else {
            return false;
        };
        let Some(token_dense) = self.weight_token_dense_masks.get(&key) else {
            return false;
        };

        for (i, &token_word) in token_dense.iter().enumerate() {
            let dense_word = dense.get(i).copied().unwrap_or(0);
            if token_word & !dense_word != 0 {
                return false;
            }
        }

        if let Some(sparse_mask) = self.weight_token_sparse_buf_masks.get(&key) {
            or_sparse_buf_entries(buf, sparse_mask);
        } else {
            or_dense_buf(buf, mask);
        }
        true
    }

    #[inline(always)]
    pub(crate) fn or_dense_token_set_to_buf_sparse(
        &self,
        dense: &[u64],
        token_set: &Arc<RangeSetBlaze<u32>>,
        max_tokens: u64,
        buf: &mut [u32],
    ) -> Option<bool> {
        if dense.is_empty() || token_set.is_empty() {
            return Some(false);
        }

        let mut total = 0u64;
        for range in token_set.ranges() {
            total = total.saturating_add((*range.end() as u64).saturating_sub(*range.start() as u64) + 1);
            if total > max_tokens {
                return None;
            }
        }

        let n_internal = self.internal_token_to_tokens.len();
        let mut any = false;
        let mut stats_entries = 0u64;
        for range in token_set.ranges() {
            let start = *range.start() as usize;
            let end = (*range.end() as usize).min(n_internal.saturating_sub(1));
            if start > end {
                continue;
            }
            for internal_token in start..=end {
                let word_idx = internal_token / 64;
                let bit = internal_token % 64;
                if dense
                    .get(word_idx)
                    .is_some_and(|word| (word & (1u64 << bit)) != 0)
                {
                    self.or_internal_token_to_buf_fast(internal_token, buf, &mut stats_entries);
                    any = true;
                }
            }
        }

        Some(any)
    }

    #[inline(always)]
    pub(crate) fn has_weight_token_set_buf_if_contained(
        &self,
        dense: &[u64],
        token_set: &Arc<RangeSetBlaze<u32>>,
    ) -> bool {
        let key = Arc::as_ptr(token_set) as usize;
        if !self.weight_token_buf_masks.contains_key(&key) {
            return false;
        }
        let Some(token_dense) = self.weight_token_dense_masks.get(&key) else {
            return false;
        };

        for (i, &token_word) in token_dense.iter().enumerate() {
            let dense_word = dense.get(i).copied().unwrap_or(0);
            if token_word & !dense_word != 0 {
                return false;
            }
        }

        true
    }

    fn compute_weight_token_buf_masks(&self) -> DenseWeightBufMaskCache {
        let buf_words = self.mask_len();
        if buf_words == 0 {
            return FxHashMap::default();
        }

        let final_dense_masks = self.final_weight_token_dense_masks();
        let build = |(&key, dense): (&usize, &DenseWords)| {
            let estimated_cost = self.estimate_internal_dense_to_buf_cost(dense);
            if estimated_cost == 0 {
                return None;
            }

            let mut buf = vec![0u32; buf_words];
            self.or_internal_dense_to_buf(dense, &mut buf, true);
            Some((key, buf.into_boxed_slice()))
        };

        final_dense_masks.into_iter().filter_map(build).collect()
    }

    fn dense_buf_to_sparse_entries(buf: &[u32]) -> Box<[(u16, u32)]> {
        buf.iter()
            .enumerate()
            .filter_map(|(idx, &word)| {
                if word == 0 {
                    None
                } else {
                    Some((idx as u16, word))
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    fn compute_weight_token_sparse_buf_masks(&self) -> SparseWeightBufMaskCache {
        let buf_words = self.mask_len();
        if buf_words == 0 || buf_words > u16::MAX as usize {
            return FxHashMap::default();
        }

        let final_dense_masks = self.final_weight_token_dense_masks();
        let build = |(&key, dense): (&usize, &DenseWords)| {
            let estimated_cost = self.estimate_internal_dense_to_buf_cost(dense);
            if estimated_cost == 0 || estimated_cost >= (buf_words / 2) as u64 {
                return None;
            }

            let mut buf = vec![0u32; buf_words];
            self.or_internal_dense_to_buf(dense, &mut buf, true);
            let sparse = Self::dense_buf_to_sparse_entries(&buf);
            if sparse.len() < buf_words / 2 {
                Some((key, sparse))
            } else {
                None
            }
        };

        final_dense_masks.into_iter().filter_map(build).collect()
    }

    fn compute_seed_state_buf_masks(&self) -> SeedStateBufMasks {
        let buf_words = self.mask_len();
        if buf_words == 0 {
            return vec![None; self.seed_state_dense.len()];
        }

        let build = |dense: &DenseWords| {
            let estimated_cost = self.estimate_internal_dense_to_buf_cost(dense);
            if estimated_cost == 0 {
                return None;
            }

            let mut buf = vec![0u32; buf_words];
            self.or_internal_dense_to_buf(dense, &mut buf, true);
            Some(buf.into_boxed_slice())
        };

        if rayon::current_num_threads() == 1 {
            self.seed_state_dense.iter().map(build).collect()
        } else {
            self.seed_state_dense.par_iter().map(build).collect()
        }
    }

    /// Build dense buf masks for internal tokens with many sparse entries.
    /// A token with >THRESHOLD entries benefits from a sequential 16KB scan
    /// instead of thousands of indexed read-modify-writes.
    fn compute_heavy_token_dense_masks(&self) -> Vec<Option<Box<[u32]>>> {
        let buf_words = self.mask_len();
        if buf_words == 0 {
            return Vec::new();
        }
        // Threshold: use dense when sparse entries are large enough that a
        // sequential scan beats many indexed read-modify-writes.
        // Dense OR costs ~buf_words ops; sparse OR costs ~n_entries ops.
        // With buf in L1 cache (≤16KB), sparse random writes are fast,
        // so we only go dense when entries exceed half the buffer size.
        let threshold = buf_words / 4;
        let build = |sparse: &InternalTokenBufMasks| {
                if sparse.len() > threshold {
                    let mut dense = vec![0u32; buf_words];
                    for &(word_idx, mask) in sparse {
                        dense[word_idx as usize] |= mask;
                    }
                    Some(dense.into_boxed_slice())
                } else {
                    None
                }
            };
        if rayon::current_num_threads() == 1 {
            self.internal_token_buf_masks.iter().map(build).collect()
        } else {
            self.internal_token_buf_masks.par_iter().map(build).collect()
        }
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

    fn compute_internal_token_buf_op_costs(
        offsets: &[u32],
        heavy: &[Option<Box<[u32]>>],
        buf_len: usize,
    ) -> Vec<usize> {
        let n_internal = if offsets.len() > 1 { offsets.len() - 1 } else { 0 };
        (0..n_internal)
            .map(|idx| {
                if idx < heavy.len() && heavy[idx].is_some() {
                    buf_len
                } else {
                    (offsets[idx + 1] - offsets[idx]) as usize
                }
            })
            .collect()
    }

    fn compute_word_group_buf_op_costs(costs: &[usize]) -> Vec<usize> {
        costs
            .chunks(64)
            .map(|chunk| chunk.iter().copied().sum())
            .collect()
    }

    fn compute_dense_token_bytes(&self) -> Vec<Option<Box<[u8]>>> {
        let Some(max_token_id) = self.max_original_token_id() else {
            return Vec::new();
        };

        let mut dense = vec![None; max_token_id as usize + 1];
        for (&token_id, bytes) in self.token_bytes.iter() {
            dense[token_id as usize] = Some(bytes.clone().into_boxed_slice());
        }
        dense
    }

    fn compute_fast_transitions(&self) -> FastDwaTransitions {
        let build = |state: &crate::automata::weighted::dwa::DWAState| {
                state
                    .transitions
                    .iter()
                    .map(|(&label, (target, weight))| (label, (*target, weight.clone())))
                    .collect()
            };
        if rayon::current_num_threads() == 1 {
            self.parser_dwa.states().iter().map(build).collect()
        } else {
            self.parser_dwa.states().par_iter().map(build).collect()
        }
    }

    fn compute_dense_token_masks(&self) -> (usize, DenseWeightMaskCache) {
        let internal_token_dense_words = self.internal_token_to_tokens.len().div_ceil(64);
        if internal_token_dense_words == 0 {
            return (0, DenseWeightMaskCache::default());
        }

        let mut unique_sets: FxHashMap<usize, &RangeSetBlaze<u32>> = FxHashMap::default();
        for state in self.parser_dwa.states() {
            if let Some(final_weight) = &state.final_weight {
                Self::collect_weight_token_sets(final_weight, &mut unique_sets);
            }
            for (_, weight) in state.transitions.values() {
                Self::collect_weight_token_sets(weight, &mut unique_sets);
            }
        }

        let build = |(key, token_set): (usize, &RangeSetBlaze<u32>)| {
                (
                    key,
                    Self::dense_words_from_internal_set_with_words(token_set, internal_token_dense_words),
                )
            };
        let dense_masks: DenseWeightMaskCache = if rayon::current_num_threads() == 1 {
            unique_sets.into_iter().map(build).collect()
        } else {
            unique_sets.into_par_iter().map(build).collect()
        };

        (internal_token_dense_words, dense_masks)
    }

    /// Build precomputed bitmask fragments for each internal token.
    pub(crate) fn build_buf_masks(&mut self) {
        self.internal_token_buf_masks = self.compute_buf_masks();
        self.word_group_buf_masks = Vec::new();
        let (word_group_sparse_masks, word_group_sparse_total_entries, word_group_sparse_max_entries) =
            self.compute_token_block_sparse_masks(64);
        let (quad_group_sparse_masks, _, _) = self.compute_token_block_sparse_masks(4);
        let (byte_group_sparse_masks, _, _) = self.compute_token_block_sparse_masks(8);
        self.word_group_sparse_masks = word_group_sparse_masks;
        self.word_group_prefix_buf_masks = self.compute_word_group_prefix_buf_masks();
        self.word_group_sparse_prefix_entries =
            Self::compute_sparse_entry_prefix(&self.word_group_sparse_masks);
        self.quad_group_sparse_masks = quad_group_sparse_masks;
        self.byte_group_sparse_masks = byte_group_sparse_masks;
        self.word_group_sparse_total_entries = word_group_sparse_total_entries;
        self.word_group_sparse_max_entries = word_group_sparse_max_entries;
        self.pair_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(2);
        self.quad_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(4);
        self.super_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(8);
        self.mega_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(16);
        self.giga_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(32);
        self.all_tokens_buf_mask = self.compute_all_tokens_buf_mask();
        self.heavy_token_dense_masks = self.compute_heavy_token_dense_masks();
        let (flat, offsets) = Self::compute_flat_buf_masks(&self.internal_token_buf_masks);
        self.internal_token_buf_flat = flat;
        self.internal_token_buf_offsets = offsets;
        self.internal_token_buf_op_costs = Self::compute_internal_token_buf_op_costs(
            &self.internal_token_buf_offsets,
            &self.heavy_token_dense_masks,
            self.mask_len(),
        );
        self.word_group_buf_op_costs =
            Self::compute_word_group_buf_op_costs(&self.internal_token_buf_op_costs);
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
            self.seed_state_dense.clear();
            self.seed_state_by_dense_hash.clear();
            self.seed_state_buf_masks.clear();
            self.seed_universe_dense = empty_dense_words();
            return;
        }

        if self.tokenizer_fast_transitions.len() != self.tokenizer.num_states() as usize {
            self.tokenizer_fast_transitions = self.compute_tokenizer_fast_transitions();
        }

        let universe = self.internal_token_universe();
        self.seed_universe_dense = self.dense_words_from_internal_set(&universe);

        self.seed_terminal_dense = self.build_seed_terminal_dense_masks();
        self.seed_state_dense = self.build_seed_state_dense_masks();
        self.extend_seed_state_dense_with_single_terminal_exclusions();
        self.seed_state_by_dense_hash =
            Self::compute_seed_state_hashes(&self.seed_state_dense);
        self.seed_state_buf_masks = self.compute_seed_state_buf_masks();
    }

    fn extend_seed_state_dense_with_single_terminal_exclusions(&mut self) {
        let base_count = (self.tokenizer.num_states() as usize).min(self.seed_state_dense.len());
        if base_count == 0 || self.seed_terminal_dense.is_empty() {
            return;
        }

        let mut seen = Self::compute_seed_state_hashes(&self.seed_state_dense);
        let mut additions = Vec::new();

        for (&(tokenizer_state, _terminal_id), terminal_dense) in &self.seed_terminal_dense {
            let state_idx = tokenizer_state as usize;
            if state_idx >= base_count {
                continue;
            }
            let base = &self.seed_state_dense[state_idx];
            let mut excluded = base.to_vec();
            for (word, terminal_word) in excluded.iter_mut().zip(terminal_dense.iter()) {
                *word &= !*terminal_word;
            }
            if excluded.as_slice() == base.as_ref() {
                continue;
            }

            let hash = Self::dense_words_hash(&excluded);
            let duplicate = seen
                .get(&hash)
                .is_some_and(|indices| {
                    indices.iter().any(|&idx| {
                        idx < self.seed_state_dense.len()
                            && self.seed_state_dense[idx].as_ref() == excluded.as_slice()
                    })
                })
                || additions.iter().any(|dense: &DenseWords| dense.as_ref() == excluded.as_slice());
            if duplicate {
                continue;
            }

            seen.entry(hash).or_default().push(self.seed_state_dense.len() + additions.len());
            additions.push(excluded.into());
        }

        self.seed_state_dense.extend(additions);
    }

    fn collect_weight_token_sets<'a>(
        weight: &'a Weight,
        unique_sets: &mut FxHashMap<usize, &'a RangeSetBlaze<u32>>,
    ) {
        for token_set in weight.unique_token_sets() {
            let key = token_set as *const RangeSetBlaze<u32> as usize;
            unique_sets.entry(key).or_insert(token_set);
        }
    }

    fn final_weight_token_dense_masks(&self) -> Vec<(&usize, &DenseWords)> {
        let mut keys: FxHashMap<usize, ()> = FxHashMap::default();
        let mut dense_masks = Vec::new();

        for state in self.parser_dwa.states() {
            let Some(final_weight) = &state.final_weight else {
                continue;
            };

            for token_set in final_weight.unique_token_sets() {
                let key = token_set as *const RangeSetBlaze<u32> as usize;
                if keys.insert(key, ()).is_some() {
                    continue;
                }
                if let Some((stored_key, dense)) = self.weight_token_dense_masks.get_key_value(&key)
                {
                    dense_masks.push((stored_key, dense));
                }
            }
        }

        dense_masks
    }

    fn dense_words_from_internal_set_with_words(
        internal_tokens: &RangeSetBlaze<u32>,
        dense_word_count: usize,
    ) -> DenseWords {
        let mut words = vec![0u64; dense_word_count];
        for internal_token in internal_tokens.iter() {
            let index = internal_token as usize / 64;
            let bit = internal_token as usize % 64;
            if let Some(word) = words.get_mut(index) {
                *word |= 1u64 << bit;
            }
        }
        Arc::from(words.into_boxed_slice())
    }

    fn dense_words_from_internal_set(&self, internal_tokens: &RangeSetBlaze<u32>) -> DenseWords {
        Self::dense_words_from_internal_set_with_words(internal_tokens, self.internal_token_dense_words)
    }

    /// Start a new mutable generation state for this compiled constraint.
    ///
    /// This is the boundary between compile time and runtime.  The returned
    /// [`ConstraintState`] initially represents the empty byte prefix: its active
    /// lexer/parser frontier is the set from which the first **Mask** operation
    /// should be evaluated.
    ///
    fn build_internal_token_buf_mask(originals: &[u32]) -> InternalTokenBufMasks {
        let mut result = Vec::<(u16, u32)>::new();
        let mut current_word = None::<u16>;
        let mut current_mask = 0u32;
        for &original in originals {
            let word = (original / 32) as u16;
            let bit = original % 32;
            match current_word {
                None => {
                    current_word = Some(word);
                    current_mask = 1u32 << bit;
                }
                Some(current) if current == word => {
                    current_mask |= 1u32 << bit;
                }
                Some(current) if current < word => {
                    result.push((current, current_mask));
                    current_word = Some(word);
                    current_mask = 1u32 << bit;
                }
                Some(_) => {
                    return Self::build_internal_token_buf_mask_unsorted(originals);
                }
            }
        }
        if let Some(word) = current_word {
            result.push((word, current_mask));
        }
        result
    }

    fn build_internal_token_buf_mask_unsorted(originals: &[u32]) -> InternalTokenBufMasks {
        let mut word_map = BTreeMap::<u16, u32>::new();
        for &original in originals {
            let word = (original / 32) as u16;
            let bit = original % 32;
            *word_map.entry(word).or_default() |= 1u32 << bit;
        }
        word_map.into_iter().collect()
    }

    fn build_seed_terminal_dense_masks(&self) -> SeedTerminalDenseMasks {
        self.can_match
            .iter()
            .flat_map(|(&terminal_id, weight)| {
                weight
                    .compact_entries()
                    .unwrap_or_default()
                    .into_iter()
                    .flat_map(move |(start, end, token_set)| {
                        let dense = self.dense_words_from_internal_set(token_set.as_ref());
                        let mut entries = Vec::new();
                        for internal_tsid in start..=end {
                            if let Some(states) = self.internal_tsid_to_states.get(internal_tsid as usize) {
                                for &tokenizer_state in states {
                                    entries.push(((tokenizer_state, terminal_id), dense.clone()));
                                }
                            } else {
                                entries.push(((internal_tsid, terminal_id), dense.clone()));
                            }
                        }
                        entries.into_iter()
                    })
            })
            .collect()
    }

    fn precompute_node_reachable_dense(
        node: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
        dense_words: usize,
        cache: &mut FxHashMap<usize, Vec<u64>>,
    ) {
        let ptr = node as *const _ as usize;
        if cache.contains_key(&ptr) {
            return;
        }
        let mut dense = vec![0u64; dense_words];
        if node.has_token() {
            let id = node.token_id();
            let word = id / 64;
            let bit = id % 64;
            if word < dense_words {
                dense[word] |= 1u64 << bit;
            }
        }
        for (_, child) in node.iter_children() {
            Self::precompute_node_reachable_dense(child, dense_words, cache);
            let child_ptr = child as *const _ as usize;
            let child_dense = cache.get(&child_ptr).unwrap();
            for i in 0..dense_words {
                dense[i] |= child_dense[i];
            }
        }
        cache.insert(ptr, dense);
    }

    fn walk_seed_trie(
        node: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
        state: u32,
        mask: &mut [u64],
        flat_transitions: &[Box<[u32; 256]>],
        terminal_states: &[bool],
        node_reachable_dense: &FxHashMap<usize, Vec<u64>>,
    ) {
        if node.has_token() {
            let id = node.token_id();
            let word = id / 64;
            let bit = id % 64;
            if word < mask.len() {
                mask[word] |= 1u64 << bit;
            }
        }
        for (segment, child) in node.iter_children() {
            let mut current_state = state;
            let mut terminal_hit = false;
            let mut blocked = false;
            for &byte in segment {
                let next = flat_transitions[current_state as usize][byte as usize];
                if next == u32::MAX {
                    blocked = true;
                    break;
                }
                if terminal_states[next as usize] {
                    terminal_hit = true;
                    break;
                }
                current_state = next;
            }
            if blocked {
                continue;
            }
            if terminal_hit {
                let child_ptr = child as *const _ as usize;
                if let Some(child_dense) = node_reachable_dense.get(&child_ptr) {
                    for i in 0..mask.len() {
                        mask[i] |= child_dense[i];
                    }
                }
            } else {
                Self::walk_seed_trie(child, current_state, mask, flat_transitions, terminal_states, node_reachable_dense);
            }
        }
    }

    fn build_seed_state_dense_masks(&self) -> SeedStateDenseMasks {
        let state_count = self.tokenizer.num_states() as usize;
        let dense_words = self.internal_token_dense_words;
        if state_count == 0 || dense_words == 0 {
            return (0..state_count)
                .map(|_| Arc::from(vec![0u64; dense_words].into_boxed_slice()))
                .collect();
        }

        let terminal_state = |state| self.tokenizer.matched_terminals_iter(state).next().is_some();
        let terminal_states: Vec<bool> = if rayon::current_num_threads() == 1 {
            (0..self.tokenizer.num_states()).map(terminal_state).collect()
        } else {
            (0..self.tokenizer.num_states())
                .into_par_iter()
                .map(terminal_state)
                .collect()
        };
        let owned_flat_transitions;
        let flat_transitions: &[Box<[u32; 256]>] = if self.tokenizer_fast_transitions.len() == state_count {
            &self.tokenizer_fast_transitions
        } else {
            owned_flat_transitions = self.compute_tokenizer_fast_transitions();
            &owned_flat_transitions
        };

        // Build trie over internal token byte strings so common prefixes
        // are shared instead of re-simulated for every token.
        let trie = crate::ds::vocab_prefix_tree::VocabPrefixTree::build_owned(
            self.internal_token_bytes
                .iter()
                .map(|(&token_id, bytes)| (token_id as usize, bytes.clone()))
                .collect(),
        );

        // Precompute a dense reachable-token bitmap for every trie node.
        let mut node_reachable_dense: FxHashMap<usize, Vec<u64>> = FxHashMap::default();
        Self::precompute_node_reachable_dense(&trie.root, dense_words, &mut node_reachable_dense);

        let root_ptr = &trie.root as *const _ as usize;
        let all_tokens_dense: DenseWords = Arc::from(
            node_reachable_dense
                .get(&root_ptr)
                .expect("root reachable-token bitmap must be precomputed")
                .clone()
                .into_boxed_slice(),
        );

        let compute_for_state = |start_state: u32| -> DenseWords {
            let start_idx = start_state as usize;
            if terminal_states.get(start_idx).copied().unwrap_or(false) {
                return Arc::clone(&all_tokens_dense);
            }

            let mut mask = vec![0u64; dense_words];
            Self::walk_seed_trie(
                &trie.root,
                start_state,
                &mut mask,
                flat_transitions,
                &terminal_states,
                &node_reachable_dense,
            );
            Arc::from(mask.into_boxed_slice())
        };

        let mapping_covers_states = self.state_to_internal_tsid.len() >= state_count;
        if mapping_covers_states && !self.internal_tsid_to_states.is_empty() {
            let max_mapped_internal = self
                .state_to_internal_tsid
                .iter()
                .take(state_count)
                .filter(|&&internal| internal != u32::MAX)
                .map(|&internal| internal as usize)
                .max()
                .unwrap_or(0);
            let class_count = self.internal_tsid_to_states.len().max(max_mapped_internal + 1);
            let mut representatives: Vec<Option<u32>> = vec![None; class_count];

            for (internal_tsid, states) in self.internal_tsid_to_states.iter().enumerate() {
                if internal_tsid < representatives.len() {
                    representatives[internal_tsid] = states.first().copied();
                }
            }
            for (state, &internal_tsid) in self.state_to_internal_tsid.iter().take(state_count).enumerate() {
                if internal_tsid == u32::MAX {
                    continue;
                }
                let internal_idx = internal_tsid as usize;
                if let Some(slot) = representatives.get_mut(internal_idx) {
                    slot.get_or_insert(state as u32);
                }
            }

            let build_class_mask =
                |representative: &Option<u32>| (*representative).map(|state| compute_for_state(state));
            let class_masks: Vec<Option<DenseWords>> = if rayon::current_num_threads() == 1 {
                representatives.iter().map(build_class_mask).collect()
            } else {
                representatives.par_iter().map(build_class_mask).collect()
            };

            let build_state_mask = |state: usize| {
                    let internal_tsid = self.state_to_internal_tsid[state];
                    if internal_tsid != u32::MAX {
                        if let Some(Some(mask)) = class_masks.get(internal_tsid as usize) {
                            return Arc::clone(mask);
                        }
                    }
                    compute_for_state(state as u32)
                };
            return if rayon::current_num_threads() == 1 {
                (0..state_count).map(build_state_mask).collect()
            } else {
                (0..state_count).into_par_iter().map(build_state_mask).collect()
            };
        }

        // Fallback for older/deserialized constraints without state-class maps.
        if rayon::current_num_threads() == 1 {
            (0..state_count as u32).map(compute_for_state).collect()
        } else {
            (0..state_count as u32)
                .into_par_iter()
                .map(compute_for_state)
                .collect()
        }
    }


}
