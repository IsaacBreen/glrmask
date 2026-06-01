use crate::parser::glr::table::TableAmbiguity;
use crate::grammar::flat::TerminalID;

pub use super::artifact::Constraint;
pub(crate) use super::token_space::final_mask_mapping::{
    DeltaReplayProfileStats,
    DenseToBufProfileStats,
};
use super::bitmask_ops::{
    andnot_dense_buf,
    andnot_sparse_buf_entries,
    copy_dense_buf,
    or_dense_buf,
    or_sparse_buf_entries,
};

impl Constraint {
    /// Return parser-table ambiguity records discovered during GLR table construction.
    ///
    /// This is a grammar diagnostic, not part of the runtime mask/commit loop.
    pub fn table_ambiguous_actions(&self) -> Vec<TableAmbiguity> {
        self.table.ambiguous_actions()
    }

    /// Return whether the compiled parser table contains any ambiguity records.
    pub fn table_has_ambiguity(&self) -> bool {
        self.table.has_ambiguity()
    }

    /// Return display names for grammar terminals by terminal id.
    pub fn terminal_display_names(&self) -> &[String] {
        &self.terminal_display_names
    }

    /// Return the display name for one grammar terminal id, if present.
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

    pub(crate) fn or_full_internal_word_run_to_buf(
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
    pub(crate) fn or_internal_token_to_buf_fast(
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

#[cfg(test)]
mod tests {
    use super::Constraint;

    #[test]
    fn json_escape_prefix_predicate_matches_supported_short_escapes() {
        for bytes in [b"\\\"".as_slice(), b"\\\\".as_slice(), b"\\b".as_slice(), b"\\f".as_slice(), b"\\n".as_slice(), b"\\r".as_slice(), b"\\t".as_slice()] {
            assert!(Constraint::token_starts_json_escape_prefix(bytes));
        }

        for bytes in [b"\\u".as_slice(), b"\\x".as_slice(), b"\\".as_slice(), b"abc".as_slice(), b"".as_slice()] {
            assert!(!Constraint::token_starts_json_escape_prefix(bytes));
        }
    }
}
