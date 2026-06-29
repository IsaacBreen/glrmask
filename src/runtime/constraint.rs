use crate::automata::lexer::Lexer;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;
use rayon::prelude::*;

use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::table::TableAmbiguity;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;

use super::artifact::{
    empty_dense_words,
    DenseWeightBufMaskCache,
    DenseWeightMaskCache,
    DenseWords,
    DynamicMaskVocab,
    FastDwaTransitions,
    FastTokenizerTransitions,
    InternalTokenBufMasks,
    SeedTerminalDenseMasks,
    SparseWeightBufMaskCache,
};
pub use super::artifact::Constraint;
pub(crate) use super::mask_mapping::{DeltaReplayProfileStats, DenseToBufProfileStats};
use super::mask_mapping::FinalMaskMapping;
use super::state::ConstraintState;


/// Dense buf OR: `buf[i] |= mask[i]` for all i in min(buf.len(), mask.len()).
/// Processes u64 chunks for reduced loop overhead and better throughput.
#[inline(always)]
fn or_dense_buf(buf: &mut [u32], mask: &[u32]) {
    let n = buf.len().min(mask.len());
    let n_pairs = n / 2;
    unsafe {
        let buf_ptr = buf.as_mut_ptr();
        let mask_ptr = mask.as_ptr();
        for i in 0..n_pairs {
            let offset = i * 2;
            let b = std::ptr::read_unaligned(buf_ptr.add(offset) as *const u64);
            let m = std::ptr::read_unaligned(mask_ptr.add(offset) as *const u64);
            std::ptr::write_unaligned(buf_ptr.add(offset) as *mut u64, b | m);
        }
        for i in (n_pairs * 2)..n {
            *buf_ptr.add(i) |= *mask_ptr.add(i);
        }
    }
}

/// Dense buf AND-NOT: `buf[i] &= !mask[i]` for all i in min(buf.len(), mask.len()).
/// Processes u64 chunks for reduced loop overhead and better throughput.
#[inline(always)]
fn andnot_dense_buf(buf: &mut [u32], mask: &[u32]) {
    let n = buf.len().min(mask.len());
    let n_pairs = n / 2;
    unsafe {
        let buf_ptr = buf.as_mut_ptr();
        let mask_ptr = mask.as_ptr();
        for i in 0..n_pairs {
            let offset = i * 2;
            let b = std::ptr::read_unaligned(buf_ptr.add(offset) as *const u64);
            let m = std::ptr::read_unaligned(mask_ptr.add(offset) as *const u64);
            std::ptr::write_unaligned(buf_ptr.add(offset) as *mut u64, b & !m);
        }
        for i in (n_pairs * 2)..n {
            *buf_ptr.add(i) &= !*mask_ptr.add(i);
        }
    }
}

#[inline(always)]
fn copy_dense_buf(buf: &mut [u32], mask: &[u32]) {
    let n = buf.len().min(mask.len());
    unsafe {
        std::ptr::copy_nonoverlapping(mask.as_ptr(), buf.as_mut_ptr(), n);
    }
}

#[inline(always)]
fn or_sparse_buf_entries(buf: &mut [u32], entries: &[(u16, u32)]) {
    for &(word_idx, mask) in entries {
        unsafe {
            let slot = buf.get_unchecked_mut(word_idx as usize);
            *slot |= mask;
        }
    }
}

#[inline(always)]
fn andnot_sparse_buf_entries(buf: &mut [u32], entries: &[(u16, u32)]) {
    for &(word_idx, mask) in entries {
        unsafe {
            let slot = buf.get_unchecked_mut(word_idx as usize);
            *slot &= !mask;
        }
    }
}

#[inline(always)]
fn count_complement_subgroups(missing: u64, valid_mask: u64) -> (u32, u32, u32) {
    let mut byte_groups = 0u32;
    let mut nibble_groups = 0u32;
    let mut remaining_bits = 0u32;

    for byte_idx in 0..8 {
        let shift = byte_idx * 8;
        let byte_valid = ((valid_mask >> shift) & 0xff) as u8;
        if byte_valid == 0 {
            continue;
        }

        let byte_missing = ((missing >> shift) & 0xff) as u8;
        if byte_valid == 0xff && byte_missing == 0xff {
            byte_groups += 1;
            continue;
        }

        for nibble_idx in 0..2 {
            let nibble_shift = nibble_idx * 4;
            let nibble_valid = (byte_valid >> nibble_shift) & 0x0f;
            if nibble_valid == 0 {
                continue;
            }

            let nibble_missing = (byte_missing >> nibble_shift) & 0x0f;
            if nibble_valid == 0x0f && nibble_missing == 0x0f {
                nibble_groups += 1;
            } else {
                remaining_bits += nibble_missing.count_ones();
            }
        }
    }

    (byte_groups, nibble_groups, remaining_bits)
}

impl Constraint {
    fn build_dynamic_mask_vocab(&self) -> DynamicMaskVocab {
        let mut aliases_by_bytes = BTreeMap::<Vec<u8>, Vec<u32>>::new();
        for (&token_id, bytes) in self.token_bytes.iter() {
            aliases_by_bytes
                .entry(bytes.clone())
                .or_default()
                .push(token_id);
        }

        let mut aliases_by_canonical = BTreeMap::<u32, Box<[u32]>>::new();
        let mut trie_entries = Vec::with_capacity(aliases_by_bytes.len());
        for (bytes, mut token_ids) in aliases_by_bytes {
            token_ids.sort_unstable();
            let canonical = token_ids[0];
            aliases_by_canonical.insert(canonical, token_ids.into_boxed_slice());
            trie_entries.push((canonical as usize, bytes));
        }

        DynamicMaskVocab {
            trie: Arc::new(crate::ds::vocab_prefix_tree::VocabPrefixTree::build_owned(
                trie_entries,
            )),
            token_ids: Arc::new(aliases_by_canonical),
            terminal_self_loop_bytes: Arc::new(std::sync::OnceLock::new()),
        }
    }

    pub fn table_ambiguous_actions(&self) -> Vec<TableAmbiguity> {
        self.table.ambiguous_actions()
    }

    pub fn table_has_ambiguity(&self) -> bool {
        self.table.has_ambiguity()
    }

    pub fn terminal_display_names(&self) -> &[String] {
        &self.terminal_display_names
    }

    pub fn terminal_display_name(&self, terminal_id: TerminalID) -> Option<&str> {
        self.terminal_display_names
            .get(terminal_id as usize)
            .map(String::as_str)
    }

    pub(crate) fn internal_token_materialization_cost(&self, internal_token: usize) -> u64 {
        if internal_token < self.heavy_token_dense_masks.len()
            && self.heavy_token_dense_masks[internal_token].is_some()
        {
            return self.mask_len() as u64;
        }
        if internal_token + 1 >= self.internal_token_buf_offsets.len() {
            return 0;
        }
        (self.internal_token_buf_offsets[internal_token + 1]
            - self.internal_token_buf_offsets[internal_token]) as u64
    }

    pub(crate) fn estimate_internal_dense_to_buf_cost(&self, dense: &[u64]) -> u64 {
        if self.final_mask_mapping.internal_len() > 0 {
            return self.final_mask_mapping.estimate_dense_to_buf_cost(dense);
        }

        let all_mask = &self.all_tokens_buf_mask;
        let sparse_word_groups = &self.word_group_sparse_masks;
        let offsets = &self.internal_token_buf_offsets;
        let n_internal = if offsets.len() > 1 { offsets.len() - 1 } else { 0 };
        if n_internal == 0 || dense.is_empty() {
            return 0;
        }

        let n_set: usize = dense.iter().map(|w| w.count_ones() as usize).sum();
        let buf_len = self.mask_len();
        if n_set >= n_internal && !all_mask.is_empty() {
            return buf_len as u64;
        }
        if n_set == 0 {
            return 0;
        }

        let n_missing = n_internal - n_set;

        let dense_complement_fast_path = n_set.saturating_mul(5) >= n_internal.saturating_mul(4)
            && n_missing <= 128;

        if !all_mask.is_empty() && dense_complement_fast_path {
            let mut cost = buf_len as u64;
            for (wi, &w) in dense.iter().enumerate() {
                if wi * 64 >= n_internal {
                    break;
                }
                let remaining = n_internal - wi * 64;
                let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
                let missing = !w & valid_mask;
                if missing == 0 {
                    continue;
                }
                if missing == valid_mask {
                    if let Some(group_mask) = sparse_word_groups.get(wi) {
                        cost += group_mask.len() as u64;
                        continue;
                    }
                }
                cost += self.internal_bits_grouped_buf_op_cost(wi, missing, valid_mask, buf_len)
                    as u64;
            }
            cost
        } else {
            let mut cost = 0u64;
            for (wi, &w) in dense.iter().enumerate() {
                if wi * 64 >= n_internal {
                    break;
                }
                let remaining = n_internal - wi * 64;
                let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
                let valid_bits = w & valid_mask;
                if valid_bits == 0 {
                    continue;
                }
                if valid_bits == valid_mask {
                    if let Some(group_mask) = sparse_word_groups.get(wi) {
                        cost += group_mask.len() as u64;
                        continue;
                    }
                }
                cost += self.internal_bits_grouped_buf_op_cost(wi, valid_bits, valid_mask, buf_len)
                    as u64;
            }
            cost
        }
    }

    pub(crate) fn apply_internal_dense_delta_to_buf(
        &self,
        previous_dense: &[u64],
        current_dense: &[u64],
        buf: &mut [u32],
    ) -> DeltaReplayProfileStats {
        let mut stats = DeltaReplayProfileStats::default();
        let offsets = &self.internal_token_buf_offsets;
        let flat = &self.internal_token_buf_flat;
        let heavy = &self.heavy_token_dense_masks;
        let n_internal = if offsets.len() > 1 { offsets.len() - 1 } else { 0 };

        if n_internal == 0 {
            return stats;
        }

        let word_len = previous_dense.len().max(current_dense.len());
        for wi in 0..word_len {
            if wi * 64 >= n_internal {
                break;
            }
            let remaining = n_internal - wi * 64;
            let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
            let previous = previous_dense.get(wi).copied().unwrap_or(0) & valid_mask;
            let current = current_dense.get(wi).copied().unwrap_or(0) & valid_mask;

            let mut added = current & !previous;
            if added == valid_mask {
                if let Some(group_mask) = self.word_group_sparse_masks.get(wi) {
                    stats.added_word_group_hits += 1;
                    stats.added_word_group_entries += group_mask.len() as u64;
                    or_sparse_buf_entries(buf, group_mask);
                    continue;
                }
            }
            for byte_idx in 0..8 {
                let shift = byte_idx * 8;
                let byte_valid = (valid_mask >> shift) & 0xff;
                let byte_bits = (added >> shift) & 0xff;
                if byte_valid == 0xff && byte_bits == 0xff {
                    if let Some(group_mask) = self.byte_group_sparse_masks.get(wi * 8 + byte_idx) {
                        stats.added_byte_group_hits += 1;
                        stats.added_byte_group_entries += group_mask.len() as u64;
                        or_sparse_buf_entries(buf, group_mask);
                        added &= !(0xffu64 << shift);
                    }
                }
            }
            for quad_idx in 0..16 {
                let shift = quad_idx * 4;
                let quad_valid = (valid_mask >> shift) & 0x0f;
                let quad_bits = (added >> shift) & 0x0f;
                if quad_valid == 0x0f && quad_bits == 0x0f {
                    if let Some(group_mask) = self.quad_group_sparse_masks.get(wi * 16 + quad_idx) {
                        stats.added_byte_group_hits += 1;
                        stats.added_byte_group_entries += group_mask.len() as u64;
                        or_sparse_buf_entries(buf, group_mask);
                        added &= !(0x0fu64 << shift);
                    }
                }
            }
            while added != 0 {
                stats.added_token_iterations += 1;
                let bit = added.trailing_zeros() as usize;
                let internal_token = wi * 64 + bit;
                if internal_token < heavy.len() {
                    if let Some(ref dense_mask) = heavy[internal_token] {
                        stats.added_token_entries += dense_mask.len() as u64;
                        or_dense_buf(buf, dense_mask);
                        added &= added - 1;
                        continue;
                    }
                }
                let start = offsets[internal_token] as usize;
                let end = offsets[internal_token + 1] as usize;
                stats.added_token_entries += (end - start) as u64;
                or_sparse_buf_entries(buf, &flat[start..end]);
                added &= added - 1;
            }

        }

        for wi in 0..word_len {
            if wi * 64 >= n_internal {
                break;
            }
            let remaining = n_internal - wi * 64;
            let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
            let previous = previous_dense.get(wi).copied().unwrap_or(0) & valid_mask;
            let current = current_dense.get(wi).copied().unwrap_or(0) & valid_mask;

            let mut removed = previous & !current;
            if removed == valid_mask {
                if let Some(group_mask) = self.word_group_sparse_masks.get(wi) {
                    stats.removed_word_group_hits += 1;
                    stats.removed_word_group_entries += group_mask.len() as u64;
                    andnot_sparse_buf_entries(buf, group_mask);
                    continue;
                }
            }
            for byte_idx in 0..8 {
                let shift = byte_idx * 8;
                let byte_valid = (valid_mask >> shift) & 0xff;
                let byte_bits = (removed >> shift) & 0xff;
                if byte_valid == 0xff && byte_bits == 0xff {
                    if let Some(group_mask) = self.byte_group_sparse_masks.get(wi * 8 + byte_idx) {
                        stats.removed_byte_group_hits += 1;
                        stats.removed_byte_group_entries += group_mask.len() as u64;
                        andnot_sparse_buf_entries(buf, group_mask);
                        removed &= !(0xffu64 << shift);
                    }
                }
            }
            for quad_idx in 0..16 {
                let shift = quad_idx * 4;
                let quad_valid = (valid_mask >> shift) & 0x0f;
                let quad_bits = (removed >> shift) & 0x0f;
                if quad_valid == 0x0f && quad_bits == 0x0f {
                    if let Some(group_mask) = self.quad_group_sparse_masks.get(wi * 16 + quad_idx) {
                        stats.removed_byte_group_hits += 1;
                        stats.removed_byte_group_entries += group_mask.len() as u64;
                        andnot_sparse_buf_entries(buf, group_mask);
                        removed &= !(0x0fu64 << shift);
                    }
                }
            }
            while removed != 0 {
                stats.removed_token_iterations += 1;
                let bit = removed.trailing_zeros() as usize;
                let internal_token = wi * 64 + bit;
                if internal_token < heavy.len() {
                    if let Some(ref dense_mask) = heavy[internal_token] {
                        stats.removed_token_entries += dense_mask.len() as u64;
                        andnot_dense_buf(buf, dense_mask);
                        removed &= removed - 1;
                        continue;
                    }
                }
                let start = offsets[internal_token] as usize;
                let end = offsets[internal_token + 1] as usize;
                stats.removed_token_entries += (end - start) as u64;
                andnot_sparse_buf_entries(buf, &flat[start..end]);
                removed &= removed - 1;
            }
        }

        stats
    }

    pub(crate) fn rebuild_runtime_caches_impl(&mut self) {
        self.table.rebuild_guarded_shift_index();
        self.dynamic_mask_vocab = self.build_dynamic_mask_vocab();
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
        let (weight_token_buf_masks, weight_token_sparse_buf_masks) =
            self.compute_weight_token_buf_mask_caches();
        self.weight_token_buf_masks = weight_token_buf_masks;
        let weight_buf_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        self.weight_token_sparse_buf_masks = weight_token_sparse_buf_masks;
        let weight_sparse_ms = 0.0;
        self.dwa_fast_transitions = fast_transitions;
        self.tokenizer_fast_transitions = tokenizer_fast_transitions;
        let derived_ms = derived_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let seed_started_at = profile.then(std::time::Instant::now);
        self.build_seed_dense_masks();
        let seed_ms = seed_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][runtime_finalize_derived] pair_ms={:.3} quad_ms={:.3} super_ms={:.3} mega_ms={:.3} giga_ms={:.3} all_tokens_ms={:.3} heavy_ms={:.3} flat_ms={:.3} costs_ms={:.3} weight_buf_ms={:.3} weight_sparse_ms={:.3} final_weight_sets={} final_weight_sparse_sets={}",
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
                self.weight_token_sparse_buf_masks.len(),
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
        let build = |state| self.tokenizer.transition_row(state);
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

    fn direct_sparse_weight_buf_cache_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("GLRMASK_DIRECT_SPARSE_WEIGHT_BUF_CACHE")
                .map(|value| {
                    let value = value.trim();
                    !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
                })
                .unwrap_or(true)
        })
    }

    fn clear_sparse_buf_scratch(scratch: &mut [u32], touched: &mut Vec<u16>) {
        for word in touched.drain(..) {
            scratch[word as usize] = 0;
        }
    }

    fn build_sparse_buf_mask_from_internal_tokens(
        &self,
        internal_tokens: &RangeSetBlaze<u32>,
        scratch: &mut [u32],
        touched: &mut Vec<u16>,
    ) -> Box<[(u16, u32)]> {
        debug_assert!(touched.is_empty());
        for internal_token in internal_tokens.iter() {
            if let Some(token_masks) = self.internal_token_buf_masks.get(internal_token as usize) {
                for &(word, mask) in token_masks {
                    let slot = &mut scratch[word as usize];
                    if *slot == 0 {
                        touched.push(word);
                    }
                    *slot |= mask;
                }
            }
        }
        touched.sort_unstable();
        let sparse = touched
            .iter()
            .map(|&word| (word, scratch[word as usize]))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self::clear_sparse_buf_scratch(scratch, touched);
        sparse
    }

    fn token_set_cardinality_at_most(tokens: &RangeSetBlaze<u32>, limit: u64) -> bool {
        let mut total = 0u64;
        for range in tokens.ranges() {
            total = total.saturating_add(u64::from(*range.end() - *range.start()) + 1);
            if total > limit {
                return false;
            }
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
        let sparse_mask = self.weight_token_sparse_buf_masks.get(&key);
        let dense_mask = self.weight_token_buf_masks.get(&key);
        if sparse_mask.is_none() && dense_mask.is_none() {
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

        if let Some(sparse_mask) = sparse_mask {
            or_sparse_buf_entries(buf, sparse_mask);
        } else {
            or_dense_buf(buf, dense_mask.expect("cache presence checked"));
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
        if !self.weight_token_buf_masks.contains_key(&key)
            && !self.weight_token_sparse_buf_masks.contains_key(&key)
        {
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

    /// Build dense and sparse per-weight buffer-mask caches in one expansion.
    ///
    /// The sparse cache is a subset representation of the dense cache. Building
    /// both independently duplicated the internal-token-to-buffer expansion for
    /// every sparse candidate.
    fn compute_weight_token_buf_mask_caches(
        &self,
    ) -> (DenseWeightBufMaskCache, SparseWeightBufMaskCache) {
        let buf_words = self.mask_len();
        if buf_words == 0 {
            return (FxHashMap::default(), FxHashMap::default());
        }

        let can_store_sparse = buf_words <= u16::MAX as usize;
        let sparse_cost_limit = (buf_words / 2) as u64;
        let final_dense_masks = self.final_weight_token_dense_masks();
        let direct_sparse = Self::direct_sparse_weight_buf_cache_enabled()
            && self.final_mask_mapping.internal_len() == 0;
        let mut dense_masks = FxHashMap::default();
        let mut sparse_masks = FxHashMap::default();
        let mut sparse_scratch = vec![0u32; buf_words];
        let mut sparse_touched = Vec::<u16>::new();

        for (&key, token_set, dense) in final_dense_masks {
            // Most final-weight token sets are tiny. Build their sparse output
            // cache directly from the source set instead of scanning the dense
            // internal-token bitmap that the runtime keeps for containment.
            if can_store_sparse
                && direct_sparse
                && Self::token_set_cardinality_at_most(token_set, sparse_cost_limit)
            {
                let sparse = self.build_sparse_buf_mask_from_internal_tokens(
                    token_set,
                    &mut sparse_scratch,
                    &mut sparse_touched,
                );
                if sparse.len() < buf_words / 2 {
                    sparse_masks.insert(key, sparse);
                    continue;
                }
            }

            let estimated_cost = self.estimate_internal_dense_to_buf_cost(dense);
            if estimated_cost == 0 {
                continue;
            }

            let try_sparse = can_store_sparse && estimated_cost < sparse_cost_limit;
            let mut buf = vec![0u32; buf_words];
            self.or_internal_dense_to_buf(dense, &mut buf, true);

            if try_sparse {
                let sparse = Self::dense_buf_to_sparse_entries(&buf);
                if sparse.len() < buf_words / 2 {
                    sparse_masks.insert(key, sparse);
                    continue;
                }
            }

            dense_masks.insert(key, buf.into_boxed_slice());
        }

        (dense_masks, sparse_masks)
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
        let build = |state: &crate::automata::weighted_u32::dwa::DWAState| {
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
            self.seed_universe_dense = empty_dense_words();
            return;
        }

        let universe = self.internal_token_universe();
        self.seed_universe_dense = self.dense_words_from_internal_set(&universe);

        self.seed_terminal_dense = self.build_seed_terminal_dense_masks();
    }

    fn collect_weight_token_sets<'a>(
        weight: &'a Weight,
        unique_sets: &mut FxHashMap<usize, &'a RangeSetBlaze<u32>>,
    ) {
        if weight.is_full() || weight.is_empty() {
            return;
        }
        // `unique_sets` already deduplicates globally. Avoid allocating and
        // linearly deduplicating a temporary vector for every individual
        // weight before inserting the same pointers here.
        for (_tsid_range, token_set) in weight.0.range_values() {
            let token_set = token_set.as_ref();
            let key = token_set as *const RangeSetBlaze<u32> as usize;
            unique_sets.entry(key).or_insert(token_set);
        }
    }

    fn final_weight_token_dense_masks(&self) -> Vec<(&usize, &RangeSetBlaze<u32>, &DenseWords)> {
        let mut keys: FxHashMap<usize, ()> = FxHashMap::default();
        let mut dense_masks = Vec::new();

        for state in self.parser_dwa.states() {
            let Some(final_weight) = &state.final_weight else {
                continue;
            };
            if final_weight.is_full() || final_weight.is_empty() {
                continue;
            }
            // The global pointer set already removes duplicates. Do not build a
            // temporary per-weight `unique_token_sets` vector first.
            for (_tsid_range, token_set) in final_weight.0.range_values() {
                let token_set = token_set.as_ref();
                let key = token_set as *const RangeSetBlaze<u32> as usize;
                if keys.insert(key, ()).is_some() {
                    continue;
                }
                if let Some((stored_key, dense)) = self.weight_token_dense_masks.get_key_value(&key)
                {
                    dense_masks.push((stored_key, token_set, dense));
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
        let Some(max_token) = dense_word_count.checked_mul(64).and_then(|count| count.checked_sub(1)) else {
            return Arc::from(words.into_boxed_slice());
        };

        for token_range in internal_tokens.ranges() {
            let start = *token_range.start() as usize;
            if start > max_token {
                continue;
            }
            let end = (*token_range.end() as usize).min(max_token);
            let first_word = start / 64;
            let last_word = end / 64;
            let first_bit = start % 64;
            let last_bit = end % 64;

            if first_word == last_word {
                let high_mask = if last_bit == 63 {
                    u64::MAX
                } else {
                    (1u64 << (last_bit + 1)) - 1
                };
                words[first_word] |= (u64::MAX << first_bit) & high_mask;
                continue;
            }

            words[first_word] |= u64::MAX << first_bit;
            if first_word + 1 < last_word {
                words[first_word + 1..last_word].fill(u64::MAX);
            }
            let last_mask = if last_bit == 63 {
                u64::MAX
            } else {
                (1u64 << (last_bit + 1)) - 1
            };
            words[last_word] |= last_mask;
        }
        Arc::from(words.into_boxed_slice())
    }

    fn dense_words_from_internal_set(&self, internal_tokens: &RangeSetBlaze<u32>) -> DenseWords {
        Self::dense_words_from_internal_set_with_words(internal_tokens, self.internal_token_dense_words)
    }

    pub fn start(&self) -> ConstraintState<'_> {
        let state = self.initial_state_map();
        let state = ConstraintState {
            constraint: self,
            state,
            buffers: Default::default(),
            generation: 0,
            mask_cache: Mutex::new(None),
            mask_scratch: Mutex::new(Default::default()),
        };
        state.prefill_mask_cache();
        state
    }

    pub fn mask_len(&self) -> usize {
        self.token_bytes
            .keys()
            .max()
            .map(|token_id| (*token_id as usize / 32) + 1)
            .unwrap_or(0)
    }

    pub fn mask_game_internal_to_original(&self) -> &[Vec<u32>] {
        &self.internal_token_to_tokens
    }

    pub fn mask_game_original_to_internal(&self) -> &[u32] {
        &self.original_token_to_internal
    }

    pub fn num_parser_states(&self) -> u32 {
        self.table.num_states
    }

    pub fn num_tokenizer_states(&self) -> usize {
        self.tokenizer.num_states() as usize
    }

    pub fn num_forced_minimized_tokenizer_states(&self) -> usize {
        self.tokenizer.num_forced_minimized_states()
    }

    pub(crate) fn parser_dwa(&self) -> &DWA {
        &self.parser_dwa
    }


    pub(crate) fn possible_matches_for_state_internal(
        &self,
        tokenizer_state: u32,
    ) -> Option<BTreeMap<TerminalID, RangeSetBlaze<u32>>> {
        // Return possible_matches in the final shared constraint-internal vocab
        // space. These ids match parser-DWA weight token ids after reconciliation.
        let internal_tsid = self.internal_tsid_for_state(tokenizer_state);
        let mut result = BTreeMap::new();
        for (&terminal, weight) in &self.possible_matches {
            let tokens = weight.tokens_for_tsid(internal_tsid);
            if !tokens.is_empty() {
                result.insert(terminal, tokens);
            }
        }
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

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

    pub(crate) fn max_original_token_id(&self) -> Option<u32> {
        self.token_bytes.keys().next_back().copied()
    }

    fn build_seed_terminal_dense_masks(&self) -> SeedTerminalDenseMasks {
        self.possible_matches
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

    fn or_internal_token_masks_to_buf(&self, internal_token: usize, buf: &mut [u32]) {
        let masks = &self.internal_token_buf_masks[internal_token];
        for &(word_idx, mask) in masks {
            buf[word_idx as usize] |= mask;
        }
    }

    fn sparse_word_group_entries_in(&self, start: usize, len: usize) -> usize {
        let end = start + len;
        if end < self.word_group_sparse_prefix_entries.len() {
            self.word_group_sparse_prefix_entries[end] - self.word_group_sparse_prefix_entries[start]
        } else {
            self.word_group_sparse_masks[start..end]
                .iter()
                .map(Vec::len)
                .sum()
        }
    }

    #[inline(always)]
    fn prefer_dense_buf_scan(buf_words: usize, sparse_entries: usize) -> bool {
        sparse_entries > buf_words / 4
    }

    #[inline(always)]
    fn or_word_group_prefix_diff_to_buf(&self, start: usize, end: usize, buf: &mut [u32]) {
        let Some(start_mask) = self.word_group_prefix_buf_masks.get(start) else {
            return;
        };
        let Some(end_mask) = self.word_group_prefix_buf_masks.get(end) else {
            return;
        };
        let n = buf.len().min(start_mask.len()).min(end_mask.len());
        let n_pairs = n / 2;
        unsafe {
            let buf_ptr = buf.as_mut_ptr();
            let start_ptr = start_mask.as_ptr();
            let end_ptr = end_mask.as_ptr();
            for i in 0..n_pairs {
                let offset = i * 2;
                let b = std::ptr::read_unaligned(buf_ptr.add(offset) as *const u64);
                let s = std::ptr::read_unaligned(start_ptr.add(offset) as *const u64);
                let e = std::ptr::read_unaligned(end_ptr.add(offset) as *const u64);
                std::ptr::write_unaligned(buf_ptr.add(offset) as *mut u64, b | (e & !s));
            }
            for i in (n_pairs * 2)..n {
                *buf_ptr.add(i) |= *end_ptr.add(i) & !*start_ptr.add(i);
            }
        }
    }

    #[inline(always)]
    fn andnot_word_group_prefix_diff_from_buf(&self, start: usize, end: usize, buf: &mut [u32]) {
        let Some(start_mask) = self.word_group_prefix_buf_masks.get(start) else {
            return;
        };
        let Some(end_mask) = self.word_group_prefix_buf_masks.get(end) else {
            return;
        };
        let n = buf.len().min(start_mask.len()).min(end_mask.len());
        let n_pairs = n / 2;
        unsafe {
            let buf_ptr = buf.as_mut_ptr();
            let start_ptr = start_mask.as_ptr();
            let end_ptr = end_mask.as_ptr();
            for i in 0..n_pairs {
                let offset = i * 2;
                let b = std::ptr::read_unaligned(buf_ptr.add(offset) as *const u64);
                let s = std::ptr::read_unaligned(start_ptr.add(offset) as *const u64);
                let e = std::ptr::read_unaligned(end_ptr.add(offset) as *const u64);
                std::ptr::write_unaligned(buf_ptr.add(offset) as *mut u64, b & !(e & !s));
            }
            for i in (n_pairs * 2)..n {
                *buf_ptr.add(i) &= !(*end_ptr.add(i) & !*start_ptr.add(i));
            }
        }
    }

    fn or_full_internal_word_run_to_buf(
        &self,
        mut wi: usize,
        end: usize,
        buf: &mut [u32],
        stats: &mut DenseToBufProfileStats,
    ) {
        let run_len = end.saturating_sub(wi);
        if run_len > 0
            && end < self.word_group_prefix_buf_masks.len()
            && Self::prefer_dense_buf_scan(buf.len(), self.sparse_word_group_entries_in(wi, run_len))
        {
            stats.normal_full_word_hits += run_len as u64;
            stats.group_or_sparse_entries += buf.len() as u64;
            self.or_word_group_prefix_diff_to_buf(wi, end, buf);
            return;
        }

        while wi < end {
            let remaining = end - wi;
            let block = if remaining >= 32
                && self
                    .giga_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 32)))
            {
                Some((32, &self.giga_word_group_buf_masks[wi]))
            } else if remaining >= 16
                && self
                    .mega_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 16)))
            {
                Some((16, &self.mega_word_group_buf_masks[wi]))
            } else if remaining >= 8
                && self
                    .super_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 8)))
            {
                Some((8, &self.super_word_group_buf_masks[wi]))
            } else if remaining >= 4
                && self
                    .quad_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 4)))
            {
                Some((4, &self.quad_word_group_buf_masks[wi]))
            } else if remaining >= 2
                && self
                    .pair_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 2)))
            {
                Some((2, &self.pair_word_group_buf_masks[wi]))
            } else {
                None
            };

            if let Some((block_len, dense_mask)) = block {
                stats.normal_full_word_hits += block_len as u64;
                stats.group_or_sparse_entries += dense_mask.len() as u64;
                or_dense_buf(buf, dense_mask);
                wi += block_len;
                continue;
            }

            if let Some(group_mask) = self.word_group_sparse_masks.get(wi) {
                stats.normal_full_word_hits += 1;
                if Self::prefer_dense_buf_scan(buf.len(), group_mask.len())
                    && wi + 1 < self.word_group_prefix_buf_masks.len()
                {
                    stats.group_or_sparse_entries += buf.len() as u64;
                    self.or_word_group_prefix_diff_to_buf(wi, wi + 1, buf);
                } else {
                    stats.group_or_sparse_entries += group_mask.len() as u64;
                    or_sparse_buf_entries(buf, group_mask);
                }
            }
            wi += 1;
        }
    }

    fn andnot_full_internal_word_run_from_buf(
        &self,
        mut wi: usize,
        end: usize,
        buf: &mut [u32],
        stats: &mut DenseToBufProfileStats,
    ) {
        let run_len = end.saturating_sub(wi);
        if run_len > 0
            && end < self.word_group_prefix_buf_masks.len()
            && Self::prefer_dense_buf_scan(buf.len(), self.sparse_word_group_entries_in(wi, run_len))
        {
            stats.complement_full_word_hits += run_len as u64;
            stats.group_andnot_sparse_entries += buf.len() as u64;
            self.andnot_word_group_prefix_diff_from_buf(wi, end, buf);
            return;
        }

        while wi < end {
            let remaining = end - wi;
            let block = if remaining >= 32
                && self
                    .giga_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 32)))
            {
                Some((32, &self.giga_word_group_buf_masks[wi]))
            } else if remaining >= 16
                && self
                    .mega_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 16)))
            {
                Some((16, &self.mega_word_group_buf_masks[wi]))
            } else if remaining >= 8
                && self
                    .super_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 8)))
            {
                Some((8, &self.super_word_group_buf_masks[wi]))
            } else if remaining >= 4
                && self
                    .quad_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 4)))
            {
                Some((4, &self.quad_word_group_buf_masks[wi]))
            } else if remaining >= 2
                && self
                    .pair_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 2)))
            {
                Some((2, &self.pair_word_group_buf_masks[wi]))
            } else {
                None
            };

            if let Some((block_len, dense_mask)) = block {
                stats.complement_full_word_hits += block_len as u64;
                stats.group_andnot_sparse_entries += dense_mask.len() as u64;
                andnot_dense_buf(buf, dense_mask);
                wi += block_len;
                continue;
            }

            if let Some(group_mask) = self.word_group_sparse_masks.get(wi) {
                stats.complement_full_word_hits += 1;
                if Self::prefer_dense_buf_scan(buf.len(), group_mask.len())
                    && wi + 1 < self.word_group_prefix_buf_masks.len()
                {
                    stats.group_andnot_sparse_entries += buf.len() as u64;
                    self.andnot_word_group_prefix_diff_from_buf(wi, wi + 1, buf);
                } else {
                    stats.group_andnot_sparse_entries += group_mask.len() as u64;
                    andnot_sparse_buf_entries(buf, group_mask);
                }
            }
            wi += 1;
        }
    }

    #[inline(always)]
    fn internal_token_buf_op_cost(&self, internal_token: usize, buf_len: usize) -> usize {
        if let Some(&cost) = self.internal_token_buf_op_costs.get(internal_token) {
            return cost;
        }
        if internal_token < self.heavy_token_dense_masks.len()
            && self.heavy_token_dense_masks[internal_token].is_some()
        {
            buf_len
        } else {
            (self.internal_token_buf_offsets[internal_token + 1]
                - self.internal_token_buf_offsets[internal_token]) as usize
        }
    }

    #[inline(always)]
    fn internal_bits_buf_op_cost(&self, wi: usize, mut bits: u64, buf_len: usize) -> usize {
        let mut cost = 0usize;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            let internal_token = wi * 64 + bit;
            cost += self.internal_token_buf_op_cost(internal_token, buf_len);
            bits &= bits - 1;
        }
        cost
    }

    #[inline(always)]
    pub(crate) fn internal_bits_grouped_buf_op_cost(
        &self,
        wi: usize,
        mut bits: u64,
        valid_mask: u64,
        buf_len: usize,
    ) -> usize {
        let mut cost = 0usize;
        for byte_idx in 0..8 {
            let shift = byte_idx * 8;
            let byte_valid = (valid_mask >> shift) & 0xff;
            let byte_bits = (bits >> shift) & 0xff;
            if byte_valid == 0xff && byte_bits == 0xff {
                if let Some(group_mask) = self.byte_group_sparse_masks.get(wi * 8 + byte_idx) {
                    cost += group_mask.len();
                    bits &= !(0xffu64 << shift);
                }
            }
        }

        for quad_idx in 0..16 {
            let shift = quad_idx * 4;
            let quad_valid = (valid_mask >> shift) & 0x0f;
            let quad_bits = (bits >> shift) & 0x0f;
            if quad_valid == 0x0f && quad_bits == 0x0f {
                if let Some(group_mask) = self.quad_group_sparse_masks.get(wi * 16 + quad_idx) {
                    cost += group_mask.len();
                    bits &= !(0x0fu64 << shift);
                }
            }
        }

        cost + self.internal_bits_buf_op_cost(wi, bits, buf_len)
    }

    #[inline(always)]
    fn or_internal_token_to_buf_fast(
        &self,
        internal_token: usize,
        buf: &mut [u32],
        stats_entries: &mut u64,
    ) {
        if internal_token < self.heavy_token_dense_masks.len() {
            if let Some(ref dense_mask) = self.heavy_token_dense_masks[internal_token] {
                *stats_entries += dense_mask.len() as u64;
                or_dense_buf(buf, dense_mask);
                return;
            }
        }
        let start = self.internal_token_buf_offsets[internal_token] as usize;
        let end = self.internal_token_buf_offsets[internal_token + 1] as usize;
        *stats_entries += end.saturating_sub(start) as u64;
        or_sparse_buf_entries(buf, &self.internal_token_buf_flat[start..end]);
    }

    #[inline(always)]
    fn andnot_internal_token_from_buf_fast(
        &self,
        internal_token: usize,
        buf: &mut [u32],
        stats_entries: &mut u64,
    ) {
        if internal_token < self.heavy_token_dense_masks.len() {
            if let Some(ref dense_mask) = self.heavy_token_dense_masks[internal_token] {
                *stats_entries += dense_mask.len() as u64;
                andnot_dense_buf(buf, dense_mask);
                return;
            }
        }
        let start = self.internal_token_buf_offsets[internal_token] as usize;
        let end = self.internal_token_buf_offsets[internal_token + 1] as usize;
        *stats_entries += end.saturating_sub(start) as u64;
        andnot_sparse_buf_entries(buf, &self.internal_token_buf_flat[start..end]);
    }

    fn or_internal_bits_to_buf_grouped(
        &self,
        wi: usize,
        mut bits: u64,
        valid_mask: u64,
        buf: &mut [u32],
        stats: &mut DenseToBufProfileStats,
    ) {
        for byte_idx in 0..8 {
            let shift = byte_idx * 8;
            let byte_valid = (valid_mask >> shift) & 0xff;
            let byte_bits = (bits >> shift) & 0xff;
            if byte_valid == 0xff && byte_bits == 0xff {
                if let Some(group_mask) = self.byte_group_sparse_masks.get(wi * 8 + byte_idx) {
                    stats.group_or_sparse_entries += group_mask.len() as u64;
                    or_sparse_buf_entries(buf, group_mask);
                    bits &= !(0xffu64 << shift);
                }
            }
        }

        for quad_idx in 0..16 {
            let shift = quad_idx * 4;
            let quad_valid = (valid_mask >> shift) & 0x0f;
            let quad_bits = (bits >> shift) & 0x0f;
            if quad_valid == 0x0f && quad_bits == 0x0f {
                if let Some(group_mask) = self.quad_group_sparse_masks.get(wi * 16 + quad_idx) {
                    stats.group_or_sparse_entries += group_mask.len() as u64;
                    or_sparse_buf_entries(buf, group_mask);
                    bits &= !(0x0fu64 << shift);
                }
            }
        }

        while bits != 0 {
            stats.normal_token_iterations += 1;
            let bit = bits.trailing_zeros() as usize;
            let internal_token = wi * 64 + bit;
            if internal_token < self.internal_token_buf_offsets.len().saturating_sub(1) {
                self.or_internal_token_to_buf_fast(
                    internal_token,
                    buf,
                    &mut stats.normal_sparse_entries,
                );
            }
            bits &= bits - 1;
        }
    }

    fn andnot_internal_bits_from_buf_grouped(
        &self,
        wi: usize,
        mut bits: u64,
        valid_mask: u64,
        buf: &mut [u32],
        stats: &mut DenseToBufProfileStats,
    ) {
        for byte_idx in 0..8 {
            let shift = byte_idx * 8;
            let byte_valid = (valid_mask >> shift) & 0xff;
            let byte_bits = (bits >> shift) & 0xff;
            if byte_valid == 0xff && byte_bits == 0xff {
                if let Some(group_mask) = self.byte_group_sparse_masks.get(wi * 8 + byte_idx) {
                    stats.complement_full_byte_groups += 1;
                    stats.group_andnot_sparse_entries += group_mask.len() as u64;
                    andnot_sparse_buf_entries(buf, group_mask);
                    bits &= !(0xffu64 << shift);
                }
            }
        }

        for quad_idx in 0..16 {
            let shift = quad_idx * 4;
            let quad_valid = (valid_mask >> shift) & 0x0f;
            let quad_bits = (bits >> shift) & 0x0f;
            if quad_valid == 0x0f && quad_bits == 0x0f {
                if let Some(group_mask) = self.quad_group_sparse_masks.get(wi * 16 + quad_idx) {
                    stats.complement_full_nibble_groups += 1;
                    stats.group_andnot_sparse_entries += group_mask.len() as u64;
                    andnot_sparse_buf_entries(buf, group_mask);
                    bits &= !(0x0fu64 << shift);
                }
            }
        }

        while bits != 0 {
            stats.complement_token_iterations += 1;
            let bit = bits.trailing_zeros() as usize;
            let internal_token = wi * 64 + bit;
            if internal_token < self.internal_token_buf_offsets.len().saturating_sub(1) {
                self.andnot_internal_token_from_buf_fast(
                    internal_token,
                    buf,
                    &mut stats.complement_sparse_entries,
                );
            }
            bits &= bits - 1;
        }
    }

    /// Convert a merged internal token dense bitmap to the output buffer.
    /// Uses a contiguous flat entry array for cache-friendly sequential access,
    /// with word_group fast paths for fully-set 64-bit words and heavy token
    /// dense masks for tokens with many buf entries.
    pub(crate) fn or_internal_dense_to_buf(
        &self,
        dense: &[u64],
        buf: &mut [u32],
        buf_zeroed: bool,
    ) -> DenseToBufProfileStats {
        if self.final_mask_mapping.internal_len() > 0 {
            return self.final_mask_mapping.or_dense_to_buf(dense, buf, buf_zeroed);
        }

        let mut stats = DenseToBufProfileStats::default();
        let all_mask = &self.all_tokens_buf_mask;
        let sparse_word_groups = &self.word_group_sparse_masks;
        let offsets = &self.internal_token_buf_offsets;
        let n_internal = if offsets.len() > 1 { offsets.len() - 1 } else { 0 };

        if n_internal == 0 || dense.is_empty() {
            return stats;
        }

        // Count set bits to choose path.
        let n_set: usize = dense.iter().map(|w| w.count_ones() as usize).sum();

        // Super-fast path: all internal tokens set → OR all_tokens_buf_mask.
        if n_set >= n_internal && !all_mask.is_empty() {
            if buf_zeroed {
                copy_dense_buf(buf, all_mask);
            } else {
                or_dense_buf(buf, all_mask);
            }
            return stats;
        }

        if n_set == 0 {
            return stats;
        }

        let buf_len = buf.len();
        let n_missing = n_internal - n_set;

        let dense_complement_fast_path =
            n_set.saturating_mul(5) >= n_internal.saturating_mul(4) && n_missing <= 128;

        if !all_mask.is_empty() && dense_complement_fast_path {
            stats.complement_path_used = true;
            // Complement-sparse path: start from all_tokens, subtract missing tokens.
            if buf_zeroed {
                copy_dense_buf(buf, all_mask);
            } else {
                or_dense_buf(buf, all_mask);
            }
            let mut wi = 0usize;
            while wi < dense.len() {
                if wi * 64 >= n_internal {
                    break;
                }
                stats.dense_words_visited += 1;
                let w = dense[wi];
                let remaining = n_internal - wi * 64;
                let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
                let missing = !w & valid_mask;
                if missing == 0 {
                    wi += 1;
                    continue;
                }
                if missing == valid_mask {
                    let run_start = wi;
                    wi += 1;
                    while wi < dense.len() && wi * 64 < n_internal {
                        let remaining = n_internal - wi * 64;
                        if remaining < 64 || dense[wi] != 0 {
                            break;
                        }
                        stats.dense_words_visited += 1;
                        wi += 1;
                    }
                    self.andnot_full_internal_word_run_from_buf(run_start, wi, buf, &mut stats);
                    continue;
                }
                self.andnot_internal_bits_from_buf_grouped(wi, missing, valid_mask, buf, &mut stats);
                wi += 1;
            }
        } else {
            // Normal path: process sparse light tokens and dense heavy tokens.
            let mut wi = 0usize;
            while wi < dense.len() {
                if wi * 64 >= n_internal {
                    break;
                }
                stats.dense_words_visited += 1;
                let w = dense[wi];
                let remaining = n_internal - wi * 64;
                let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
                let valid_bits = w & valid_mask;
                if valid_bits == 0 {
                    wi += 1;
                    continue;
                }
                if valid_bits == valid_mask {
                    let run_start = wi;
                    wi += 1;
                    while wi < dense.len() && wi * 64 < n_internal {
                        let remaining = n_internal - wi * 64;
                        if remaining < 64 || dense[wi] != !0u64 {
                            break;
                        }
                        stats.dense_words_visited += 1;
                        wi += 1;
                    }
                    self.or_full_internal_word_run_to_buf(run_start, wi, buf, &mut stats);
                    continue;
                }
                let missing_bits = !valid_bits & valid_mask;
                if missing_bits != 0 {
                    if let Some(group_mask) = sparse_word_groups.get(wi) {
                        let selected_cost = self.internal_bits_buf_op_cost(wi, valid_bits, buf_len);
                        let missing_cost = self
                            .word_group_buf_op_costs
                            .get(wi)
                            .copied()
                            .unwrap_or_else(|| selected_cost + self.internal_bits_buf_op_cost(wi, missing_bits, buf_len))
                            .saturating_sub(selected_cost);
                        if group_mask.len() + missing_cost < selected_cost {
                            stats.normal_group_complement_hits += 1;
                            if Self::prefer_dense_buf_scan(buf_len, group_mask.len())
                                && wi + 1 < self.word_group_prefix_buf_masks.len()
                            {
                                stats.group_or_sparse_entries += buf_len as u64;
                                self.or_word_group_prefix_diff_to_buf(wi, wi + 1, buf);
                            } else {
                                stats.group_or_sparse_entries += group_mask.len() as u64;
                                or_sparse_buf_entries(buf, group_mask);
                            }
                            let mut missing_stats = DenseToBufProfileStats::default();
                            self.andnot_internal_bits_from_buf_grouped(
                                wi,
                                missing_bits,
                                valid_mask,
                                buf,
                                &mut missing_stats,
                            );
                            stats.normal_group_complement_sparse_entries +=
                                missing_stats.group_andnot_sparse_entries
                                    + missing_stats.complement_sparse_entries;
                            stats.complement_full_byte_groups +=
                                missing_stats.complement_full_byte_groups;
                            stats.complement_full_nibble_groups +=
                                missing_stats.complement_full_nibble_groups;
                            wi += 1;
                            continue;
                        }
                    }
                }

                self.or_internal_bits_to_buf_grouped(wi, valid_bits, valid_mask, buf, &mut stats);
                wi += 1;
            }
        }

        stats
    }

    pub(crate) fn or_internal_dense_to_buf_fast(
        &self,
        dense: &[u64],
        buf: &mut [u32],
        buf_zeroed: bool,
    ) {
        if self.final_mask_mapping.internal_len() > 0 {
            self.final_mask_mapping
                .or_dense_to_buf_fast(dense, buf, buf_zeroed);
            return;
        }

        let _ = self.or_internal_dense_to_buf(dense, buf, buf_zeroed);
    }

    fn or_original_token_to_buf(&self, token_id: u32, buf: &mut [u32]) {
        let word = token_id as usize / 32;
        let bit = token_id as usize % 32;
        if let Some(slot) = buf.get_mut(word) {
            *slot |= 1u32 << bit;
        }
    }

}

impl<'a> ConstraintState<'a> {
    /// Fill a mask directly from the lexer and parser stack, without using the
    /// parser DWA.
    pub fn fill_mask_dynamic(&self, buf: &mut [u32]) {
        super::dynamic_mask::fill_mask_dynamic(self, buf);
    }

}
#[cfg(test)]
mod dense_internal_token_mask_tests {
    use super::*;

    #[test]
    fn dense_internal_token_masks_match_reference_expansion() {
        let internal_tokens = RangeSetBlaze::from_iter([
            0u32..=0,
            3..=7,
            62..=65,
            127..=130,
            190..=192,
            300..=302,
        ]);
        let actual = Constraint::dense_words_from_internal_set_with_words(&internal_tokens, 5);
        let mut expected = vec![0u64; 5];
        for token in internal_tokens.iter() {
            let word = token as usize / 64;
            let bit = token as usize % 64;
            if let Some(slot) = expected.get_mut(word) {
                *slot |= 1u64 << bit;
            }
        }
        assert_eq!(actual.as_ref(), expected.as_slice());
    }

    #[test]
    fn dense_internal_token_masks_ignore_out_of_bounds_ranges() {
        let internal_tokens = RangeSetBlaze::from_iter([63u32..=65, 190..=400]);
        let actual = Constraint::dense_words_from_internal_set_with_words(&internal_tokens, 3);
        assert_eq!(actual.as_ref(), &[1u64 << 63, 0b11, 1u64 << 62 | 1u64 << 63]);
    }
}
