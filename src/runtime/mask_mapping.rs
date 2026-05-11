use rayon::prelude::*;

pub type InternalTokenBufMasks = Vec<(u16, u32)>;

#[derive(Default, Clone, Copy)]
pub(crate) struct DenseToBufProfileStats {
    pub dense_words_visited: u64,
    pub normal_full_word_hits: u64,
    pub normal_group_complement_hits: u64,
    pub complement_full_word_hits: u64,
    pub complement_full_byte_groups: u64,
    pub complement_full_nibble_groups: u64,
    pub complement_remaining_bits: u64,
    pub normal_token_iterations: u64,
    pub complement_token_iterations: u64,
    pub normal_sparse_entries: u64,
    pub normal_group_complement_sparse_entries: u64,
    pub complement_sparse_entries: u64,
    pub complement_heavy_dense_clears: u64,
    pub complement_max_sparse_span: u64,
    pub group_or_sparse_entries: u64,
    pub group_andnot_sparse_entries: u64,
    pub complement_path_used: bool,
}

#[derive(Default, Clone, Copy)]
pub(crate) struct DeltaReplayProfileStats {
    pub added_word_group_hits: u64,
    pub added_word_group_entries: u64,
    pub removed_word_group_hits: u64,
    pub removed_word_group_entries: u64,
    pub added_byte_group_hits: u64,
    pub added_byte_group_entries: u64,
    pub removed_byte_group_hits: u64,
    pub removed_byte_group_entries: u64,
    pub added_token_iterations: u64,
    pub added_token_entries: u64,
    pub removed_token_iterations: u64,
    pub removed_token_entries: u64,
}

#[derive(Debug, Clone, Default)]
pub struct FinalMaskMapping {
    token_entries: Vec<Box<[(usize, u32)]>>,
    quad_group_entries: Vec<Box<[(usize, u32)]>>,
    byte_group_entries: Vec<Box<[(usize, u32)]>>,
    word_group_entries: Vec<Box<[(usize, u32)]>>,
    word_group_entry_prefix: Vec<usize>,
    word_group_dense_prefix: Vec<Box<[u32]>>,
    all_tokens_mask: Box<[u32]>,
    buf_words: usize,
}

impl FinalMaskMapping {
    pub fn new(internal_to_original: &[Vec<u32>], buf_words: usize) -> Self {
        let token_entries = compute_token_entries(internal_to_original, buf_words);
        let quad_group_entries = compute_block_entries(&token_entries, buf_words, 4);
        let byte_group_entries = compute_block_entries(&token_entries, buf_words, 8);
        let word_group_entries = compute_block_entries(&token_entries, buf_words, 64);
        let word_group_entry_prefix = compute_entry_prefix(&word_group_entries);
        let word_group_dense_prefix = compute_dense_prefix(&word_group_entries, buf_words);
        let all_tokens_mask = word_group_dense_prefix
            .last()
            .cloned()
            .unwrap_or_else(|| vec![0u32; buf_words].into_boxed_slice());

        Self {
            token_entries,
            quad_group_entries,
            byte_group_entries,
            word_group_entries,
            word_group_entry_prefix,
            word_group_dense_prefix,
            all_tokens_mask,
            buf_words,
        }
    }

    pub fn buf_words(&self) -> usize {
        self.buf_words
    }

    pub fn internal_len(&self) -> usize {
        self.token_entries.len()
    }

    pub fn fill_internal_ids(&self, internal_ids: &[u32], out: &mut [u32]) {
        let n_internal = self.token_entries.len();
        let selected = internal_ids.len().min(n_internal);
        let missing = n_internal.saturating_sub(selected);

        if selected >= n_internal || (selected * 5 >= n_internal * 4 && missing <= 128) {
            copy_dense(out, &self.all_tokens_mask);
            self.andnot_missing_ids(internal_ids, out);
        } else {
            self.fill_internal_ids_by_runs(internal_ids, out);
        }
    }

    pub fn fill_dense_words(&self, dense: &[u64], out: &mut [u32]) {
        self.or_dense_to_buf(dense, out, true);
    }

    pub(crate) fn estimate_dense_to_buf_cost(&self, dense: &[u64]) -> u64 {
        let n_internal = self.token_entries.len();
        if n_internal == 0 || dense.is_empty() {
            return 0;
        }
        let n_set: usize = dense.iter().map(|w| w.count_ones() as usize).sum();
        if n_set >= n_internal && !self.all_tokens_mask.is_empty() {
            return self.buf_words as u64;
        }
        if n_set == 0 {
            return 0;
        }
        self.internal_ids_from_dense(dense)
            .iter()
            .map(|&id| self.token_entries.get(id as usize).map_or(0, |entries| entries.len()))
            .sum::<usize>() as u64
    }

    pub(crate) fn or_dense_to_buf(
        &self,
        dense: &[u64],
        buf: &mut [u32],
        buf_zeroed: bool,
    ) -> DenseToBufProfileStats {
        let mut stats = DenseToBufProfileStats::default();
        let n_internal = self.token_entries.len();
        if n_internal == 0 || dense.is_empty() {
            return stats;
        }

        let n_set: usize = dense.iter().map(|w| w.count_ones() as usize).sum();
        if n_set >= n_internal && !self.all_tokens_mask.is_empty() {
            if buf_zeroed {
                copy_dense(buf, &self.all_tokens_mask);
            } else {
                or_dense(buf, &self.all_tokens_mask);
            }
            return stats;
        }
        if n_set == 0 {
            return stats;
        }

        let n_missing = n_internal - n_set;
        if !self.all_tokens_mask.is_empty()
            && n_set.saturating_mul(5) >= n_internal.saturating_mul(4)
            && n_missing <= 128
        {
            stats.complement_path_used = true;
            if buf_zeroed {
                copy_dense(buf, &self.all_tokens_mask);
            } else {
                or_dense(buf, &self.all_tokens_mask);
            }
            self.andnot_missing_dense(dense, buf, &mut stats);
            return stats;
        }

        self.or_selected_dense(dense, buf, &mut stats);
        stats
    }

    fn internal_ids_from_dense(&self, dense: &[u64]) -> Vec<u32> {
        let n_internal = self.token_entries.len();
        let mut internal_ids = Vec::new();
        for (wi, &word) in dense.iter().enumerate() {
            let base = wi * 64;
            if base >= n_internal {
                break;
            }
            let valid_bits = (n_internal - base).min(64);
            let valid_mask = if valid_bits == 64 {
                !0u64
            } else {
                (1u64 << valid_bits) - 1
            };
            let mut bits = word & valid_mask;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                internal_ids.push((base + bit) as u32);
                bits &= bits - 1;
            }
        }
        internal_ids
    }

    fn fill_internal_ids_by_runs(&self, internal_ids: &[u32], out: &mut [u32]) {
        let mut idx = 0usize;
        let mut wrote = false;
        while idx < internal_ids.len() {
            let run_start = internal_ids[idx] as usize;
            let mut idx_end = idx + 1;
            let mut run_end = run_start + 1;
            while idx_end < internal_ids.len() && internal_ids[idx_end] as usize == run_end {
                idx_end += 1;
                run_end += 1;
            }

            if run_end > self.token_entries.len() {
                for &internal_id in &internal_ids[idx..idx_end] {
                    wrote |= self.or_token(internal_id as usize, out);
                }
                idx = idx_end;
                continue;
            }

            wrote |= self.or_internal_run(run_start, run_end, out, !wrote);
            idx = idx_end;
        }
    }

    fn or_selected_dense(&self, dense: &[u64], out: &mut [u32], stats: &mut DenseToBufProfileStats) {
        let n_internal = self.token_entries.len();
        for (wi, &word) in dense.iter().enumerate() {
            let base = wi * 64;
            if base >= n_internal {
                break;
            }
            stats.dense_words_visited += 1;
            let remaining = n_internal - base;
            let valid_bits = remaining.min(64);
            let valid_mask = if valid_bits == 64 { !0u64 } else { (1u64 << valid_bits) - 1 };
            let bits = word & valid_mask;
            if bits == 0 {
                continue;
            }
            if bits == valid_mask && valid_bits == 64 {
                self.or_full_group_run(wi, wi + 1, out, false);
                stats.normal_full_word_hits += 1;
                continue;
            }
            self.or_bits(base, bits, out, stats);
        }
    }

    fn andnot_missing_dense(
        &self,
        dense: &[u64],
        out: &mut [u32],
        stats: &mut DenseToBufProfileStats,
    ) {
        let n_internal = self.token_entries.len();
        for (wi, &word) in dense.iter().enumerate() {
            let base = wi * 64;
            if base >= n_internal {
                break;
            }
            stats.dense_words_visited += 1;
            let remaining = n_internal - base;
            let valid_bits = remaining.min(64);
            let valid_mask = if valid_bits == 64 { !0u64 } else { (1u64 << valid_bits) - 1 };
            let missing = !word & valid_mask;
            if missing == 0 {
                continue;
            }
            self.andnot_bits(base, missing, out, stats);
        }
    }

    fn or_internal_run(&self, mut start: usize, end: usize, out: &mut [u32], can_copy: bool) -> bool {
        let mut wrote = false;
        while start < end && !start.is_multiple_of(4) {
            wrote |= self.or_token(start, out);
            start += 1;
        }
        while start + 4 <= end && !start.is_multiple_of(8) {
            wrote |= self.or_quad_group(start / 4, out);
            start += 4;
        }
        while start < end && !start.is_multiple_of(8) {
            wrote |= self.or_token(start, out);
            start += 1;
        }
        while start + 8 <= end && !start.is_multiple_of(64) {
            wrote |= self.or_byte_group(start / 8, out);
            start += 8;
        }
        while start < end && !start.is_multiple_of(64) {
            wrote |= self.or_token(start, out);
            start += 1;
        }
        let group_start = start / 64;
        let group_end = end / 64;
        if group_start < group_end {
            wrote |= self.or_full_group_run(group_start, group_end, out, can_copy && !wrote);
            start = group_end * 64;
        }
        while start + 8 <= end {
            wrote |= self.or_byte_group(start / 8, out);
            start += 8;
        }
        while start + 4 <= end {
            wrote |= self.or_quad_group(start / 4, out);
            start += 4;
        }
        while start < end {
            wrote |= self.or_token(start, out);
            start += 1;
        }
        wrote
    }

    fn or_full_group_run(&self, start: usize, end: usize, out: &mut [u32], can_copy: bool) -> bool {
        if start >= end {
            return false;
        }
        let sparse_entries = self.group_entries_in(start, end);
        if sparse_entries > self.buf_words / 4 && end < self.word_group_dense_prefix.len() {
            let before = &self.word_group_dense_prefix[start];
            let after = &self.word_group_dense_prefix[end];
            if can_copy {
                copy_prefix_diff(out, before, after);
            } else {
                or_prefix_diff(out, before, after);
            }
            return true;
        }
        let mut wrote = false;
        for group_id in start..end {
            if let Some(entries) = self.word_group_entries.get(group_id) {
                or_sparse(out, entries);
                wrote |= !entries.is_empty();
            }
        }
        wrote
    }

    fn group_entries_in(&self, start: usize, end: usize) -> usize {
        self.word_group_entry_prefix
            .get(end)
            .zip(self.word_group_entry_prefix.get(start))
            .map(|(end, start)| end - start)
            .unwrap_or_else(|| {
                self.word_group_entries[start..end]
                    .iter()
                    .map(|entries| entries.len())
                    .sum()
            })
    }

    #[inline(always)]
    fn or_token(&self, internal_id: usize, out: &mut [u32]) -> bool {
        let Some(entries) = self.token_entries.get(internal_id) else {
            return false;
        };
        or_sparse(out, entries);
        !entries.is_empty()
    }

    #[inline(always)]
    fn andnot_token(&self, internal_id: usize, out: &mut [u32]) {
        let Some(entries) = self.token_entries.get(internal_id) else {
            return;
        };
        andnot_sparse(out, entries);
    }

    #[inline(always)]
    fn or_quad_group(&self, group_id: usize, out: &mut [u32]) -> bool {
        let Some(entries) = self.quad_group_entries.get(group_id) else {
            return false;
        };
        or_sparse(out, entries);
        !entries.is_empty()
    }

    #[inline(always)]
    fn or_byte_group(&self, group_id: usize, out: &mut [u32]) -> bool {
        let Some(entries) = self.byte_group_entries.get(group_id) else {
            return false;
        };
        or_sparse(out, entries);
        !entries.is_empty()
    }

    fn andnot_missing_ids(&self, internal_ids: &[u32], out: &mut [u32]) {
        let mut expected = 0usize;
        for &raw in internal_ids {
            let selected = raw as usize;
            if selected > self.token_entries.len() {
                break;
            }
            while expected < selected {
                self.andnot_token(expected, out);
                expected += 1;
            }
            expected = selected.saturating_add(1);
        }
        while expected < self.token_entries.len() {
            self.andnot_token(expected, out);
            expected += 1;
        }
    }

    fn or_bits(
        &self,
        base: usize,
        mut bits: u64,
        out: &mut [u32],
        stats: &mut DenseToBufProfileStats,
    ) {
        while bits != 0 {
            stats.normal_token_iterations += 1;
            let bit = bits.trailing_zeros() as usize;
            if self.or_token(base + bit, out) {
                stats.normal_sparse_entries += self
                    .token_entries
                    .get(base + bit)
                    .map_or(0, |entries| entries.len()) as u64;
            }
            bits &= bits - 1;
        }
    }

    fn andnot_bits(
        &self,
        base: usize,
        mut bits: u64,
        out: &mut [u32],
        stats: &mut DenseToBufProfileStats,
    ) {
        while bits != 0 {
            stats.complement_token_iterations += 1;
            let bit = bits.trailing_zeros() as usize;
            self.andnot_token(base + bit, out);
            stats.complement_sparse_entries += self
                .token_entries
                .get(base + bit)
                .map_or(0, |entries| entries.len()) as u64;
            bits &= bits - 1;
        }
    }
}

fn compute_token_entries(
    internal_to_original: &[Vec<u32>],
    buf_words: usize,
) -> Vec<Box<[(usize, u32)]>> {
    internal_to_original
        .iter()
        .map(|original_ids| {
            let mut entries = Vec::<(usize, u32)>::new();
            for &original in original_ids {
                let word_idx = (original / 32) as usize;
                if word_idx >= buf_words {
                    continue;
                }
                let mask = 1u32 << (original & 31);
                if let Some((_, existing)) = entries
                    .iter_mut()
                    .find(|(existing_word_idx, _)| *existing_word_idx == word_idx)
                {
                    *existing |= mask;
                } else {
                    entries.push((word_idx, mask));
                }
            }
            entries.into_boxed_slice()
        })
        .collect()
}

fn compute_block_entries(
    token_entries: &[Box<[(usize, u32)]>],
    buf_words: usize,
    block_size: usize,
) -> Vec<Box<[(usize, u32)]>> {
    let n_groups = token_entries.len().div_ceil(block_size);
    let build = |group_id: usize| {
        let start = group_id * block_size;
        let end = (start + block_size).min(token_entries.len());
        let mut dense = vec![0u32; buf_words];
        let mut touched = Vec::<usize>::new();
        for entries in &token_entries[start..end] {
            for &(word_idx, mask) in entries.iter() {
                if dense[word_idx] == 0 {
                    touched.push(word_idx);
                }
                dense[word_idx] |= mask;
            }
        }
        touched.sort_unstable();
        touched.dedup();
        touched
            .into_iter()
            .map(|word_idx| (word_idx, dense[word_idx]))
            .collect::<Vec<_>>()
            .into_boxed_slice()
    };
    if rayon::current_num_threads() == 1 {
        (0..n_groups).map(build).collect()
    } else {
        (0..n_groups).into_par_iter().map(build).collect()
    }
}

fn compute_entry_prefix(group_entries: &[Box<[(usize, u32)]>]) -> Vec<usize> {
    let mut prefix = Vec::with_capacity(group_entries.len() + 1);
    let mut total = 0usize;
    prefix.push(total);
    for group in group_entries {
        total += group.len();
        prefix.push(total);
    }
    prefix
}

fn compute_dense_prefix(group_entries: &[Box<[(usize, u32)]>], buf_words: usize) -> Vec<Box<[u32]>> {
    let mut prefixes = Vec::with_capacity(group_entries.len() + 1);
    let mut current = vec![0u32; buf_words];
    prefixes.push(current.clone().into_boxed_slice());
    for group in group_entries {
        for &(word_idx, mask) in group.iter() {
            current[word_idx] |= mask;
        }
        prefixes.push(current.clone().into_boxed_slice());
    }
    prefixes
}

#[inline(always)]
fn or_sparse(out: &mut [u32], entries: &[(usize, u32)]) {
    for &(word_idx, mask) in entries.iter() {
        out[word_idx] |= mask;
    }
}

#[inline(always)]
fn andnot_sparse(out: &mut [u32], entries: &[(usize, u32)]) {
    for &(word_idx, mask) in entries.iter() {
        out[word_idx] &= !mask;
    }
}

#[inline(always)]
fn or_dense(out: &mut [u32], mask: &[u32]) {
    let n = out.len().min(mask.len());
    let n_pairs = n / 2;
    unsafe {
        let out_ptr = out.as_mut_ptr();
        let mask_ptr = mask.as_ptr();
        for i in 0..n_pairs {
            let offset = i * 2;
            let b = std::ptr::read_unaligned(out_ptr.add(offset) as *const u64);
            let m = std::ptr::read_unaligned(mask_ptr.add(offset) as *const u64);
            std::ptr::write_unaligned(out_ptr.add(offset) as *mut u64, b | m);
        }
        for i in (n_pairs * 2)..n {
            *out_ptr.add(i) |= *mask_ptr.add(i);
        }
    }
}

#[inline(always)]
fn copy_dense(out: &mut [u32], dense: &[u32]) {
    let n = out.len().min(dense.len());
    out[..n].copy_from_slice(&dense[..n]);
}

#[inline(always)]
fn or_prefix_diff(out: &mut [u32], before: &[u32], after: &[u32]) {
    let n = out.len().min(before.len()).min(after.len());
    let n_pairs = n / 2;
    unsafe {
        let out_ptr = out.as_mut_ptr();
        let before_ptr = before.as_ptr();
        let after_ptr = after.as_ptr();
        for i in 0..n_pairs {
            let offset = i * 2;
            let b = std::ptr::read_unaligned(out_ptr.add(offset) as *const u64);
            let s = std::ptr::read_unaligned(before_ptr.add(offset) as *const u64);
            let e = std::ptr::read_unaligned(after_ptr.add(offset) as *const u64);
            std::ptr::write_unaligned(out_ptr.add(offset) as *mut u64, b | (e & !s));
        }
        for i in (n_pairs * 2)..n {
            *out_ptr.add(i) |= *after_ptr.add(i) & !*before_ptr.add(i);
        }
    }
}

#[inline(always)]
fn copy_prefix_diff(out: &mut [u32], before: &[u32], after: &[u32]) {
    let n = out.len().min(before.len()).min(after.len());
    let n_pairs = n / 2;
    unsafe {
        let out_ptr = out.as_mut_ptr();
        let before_ptr = before.as_ptr();
        let after_ptr = after.as_ptr();
        for i in 0..n_pairs {
            let offset = i * 2;
            let s = std::ptr::read_unaligned(before_ptr.add(offset) as *const u64);
            let e = std::ptr::read_unaligned(after_ptr.add(offset) as *const u64);
            std::ptr::write_unaligned(out_ptr.add(offset) as *mut u64, e & !s);
        }
        for i in (n_pairs * 2)..n {
            *out_ptr.add(i) = *after_ptr.add(i) & !*before_ptr.add(i);
        }
    }
}
