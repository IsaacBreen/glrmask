use crate::{Candidate, Case, Mapping};
use glrmask::FinalMaskMapping;
use rayon::prelude::*;

pub struct BaselineCandidate;

pub struct BaselinePrepared {
    entries_by_internal: Vec<Box<[(usize, u32)]>>,
}

impl Candidate for BaselineCandidate {
    type Prepared = BaselinePrepared;

    fn name() -> &'static str {
        "baseline_sparse_entries"
    }

    fn prepare(mapping: &Mapping, buf_words: usize) -> Self::Prepared {
        let entries_by_internal = mapping
            .internal_to_original
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
            .collect();

        BaselinePrepared {
            entries_by_internal,
        }
    }

    fn fill(prepared: &Self::Prepared, case: &Case, out: &mut [u32]) {
        let internal_ids = &case.internal_ids;
        for &internal_id in internal_ids {
            let Some(entries) = prepared.entries_by_internal.get(internal_id as usize) else {
                continue;
            };
            for &(word_idx, mask) in entries.iter() {
                out[word_idx] |= mask;
            }
        }
    }
}

pub struct GlrMaskLikeCandidate;

pub struct GlrMaskLikePrepared {
    token_entries: Vec<Box<[(usize, u32)]>>,
    quad_group_entries: Vec<Box<[(usize, u32)]>>,
    byte_group_entries: Vec<Box<[(usize, u32)]>>,
    group_entries: Vec<Box<[(usize, u32)]>>,
    group_entry_prefix: Vec<usize>,
    group_dense_prefix: Vec<Box<[u32]>>,
    all_tokens_mask: Box<[u32]>,
    buf_words: usize,
}

impl Candidate for GlrMaskLikeCandidate {
    type Prepared = GlrMaskLikePrepared;

    fn name() -> &'static str {
        "glrmask_like_group_runs"
    }

    fn prepare(mapping: &Mapping, buf_words: usize) -> Self::Prepared {
        let baseline = BaselineCandidate::prepare(mapping, buf_words);
        let token_entries = baseline.entries_by_internal;
        let quad_group_entries = compute_block_entries(&token_entries, buf_words, 4);
        let byte_group_entries = compute_block_entries(&token_entries, buf_words, 8);
        let group_entries = compute_block_entries(&token_entries, buf_words, 64);
        let group_entry_prefix = compute_entry_prefix(&group_entries);
        let group_dense_prefix = compute_dense_prefix(&group_entries, buf_words);
        let all_tokens_mask = group_dense_prefix
            .last()
            .cloned()
            .unwrap_or_else(|| vec![0u32; buf_words].into_boxed_slice());

        GlrMaskLikePrepared {
            token_entries,
            quad_group_entries,
            byte_group_entries,
            group_entries,
            group_entry_prefix,
            group_dense_prefix,
            all_tokens_mask,
            buf_words,
        }
    }

    fn fill(prepared: &Self::Prepared, case: &Case, out: &mut [u32]) {
        let internal_ids = &case.internal_ids;
        let mut idx = 0usize;
        while idx < internal_ids.len() {
            let run_start = internal_ids[idx] as usize;
            let mut idx_end = idx + 1;
            let mut run_end = run_start + 1;
            while idx_end < internal_ids.len() && internal_ids[idx_end] as usize == run_end {
                idx_end += 1;
                run_end += 1;
            }

            if run_end > prepared.token_entries.len() {
                for &internal_id in &internal_ids[idx..idx_end] {
                    or_token(prepared, internal_id as usize, out);
                }
                idx = idx_end;
                continue;
            }

            or_internal_run(prepared, run_start, run_end, out);
            idx = idx_end;
        }
    }
}

pub struct CopyFirstGroupRunCandidate;

impl Candidate for CopyFirstGroupRunCandidate {
    type Prepared = GlrMaskLikePrepared;

    fn name() -> &'static str {
        "copy_first_group_runs"
    }

    fn prepare(mapping: &Mapping, buf_words: usize) -> Self::Prepared {
        GlrMaskLikeCandidate::prepare(mapping, buf_words)
    }

    fn fill(prepared: &Self::Prepared, case: &Case, out: &mut [u32]) {
        let internal_ids = &case.internal_ids;
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

            if run_end > prepared.token_entries.len() {
                for &internal_id in &internal_ids[idx..idx_end] {
                    wrote |= or_token(prepared, internal_id as usize, out);
                }
                idx = idx_end;
                continue;
            }

            wrote |= or_internal_run_copy_first(prepared, run_start, run_end, out, !wrote);
            idx = idx_end;
        }
    }
}

pub struct ComplementCandidate;

impl Candidate for ComplementCandidate {
    type Prepared = GlrMaskLikePrepared;

    fn name() -> &'static str {
        "complement_missing_tokens"
    }

    fn prepare(mapping: &Mapping, buf_words: usize) -> Self::Prepared {
        GlrMaskLikeCandidate::prepare(mapping, buf_words)
    }

    fn fill(prepared: &Self::Prepared, case: &Case, out: &mut [u32]) {
        let internal_ids = &case.internal_ids;
        let n_internal = prepared.token_entries.len();
        let selected = internal_ids.len().min(n_internal);
        let missing = n_internal.saturating_sub(selected);

        if selected >= n_internal || (selected * 5 >= n_internal * 4 && missing <= 128) {
            copy_dense(out, &prepared.all_tokens_mask);
            andnot_missing_ids(prepared, internal_ids, out);
        } else {
            CopyFirstGroupRunCandidate::fill(prepared, case, out);
        }
    }
}

pub struct ParallelComplementCandidate;

impl Candidate for ParallelComplementCandidate {
    type Prepared = GlrMaskLikePrepared;

    fn name() -> &'static str {
        "parallel_complement_missing_tokens"
    }

    fn prepare(mapping: &Mapping, buf_words: usize) -> Self::Prepared {
        GlrMaskLikeCandidate::prepare(mapping, buf_words)
    }

    fn fill(prepared: &Self::Prepared, case: &Case, out: &mut [u32]) {
        let internal_ids = &case.internal_ids;
        let n_internal = prepared.token_entries.len();
        let selected = internal_ids.len().min(n_internal);
        let missing = n_internal.saturating_sub(selected);

        if selected >= n_internal || (selected * 5 >= n_internal * 4 && missing <= 128) {
            parallel_copy_dense(out, &prepared.all_tokens_mask);
            andnot_missing_ids(prepared, internal_ids, out);
        } else {
            CopyFirstGroupRunCandidate::fill(prepared, case, out);
        }
    }
}

pub struct GlrMaskFinalDenseCandidate;
pub struct GlrMaskFinalDenseComplementCandidate;

impl Candidate for GlrMaskFinalDenseCandidate {
    type Prepared = FinalMaskMapping;

    fn name() -> &'static str {
        "glrmask_final_dense"
    }

    fn prepare(mapping: &Mapping, buf_words: usize) -> Self::Prepared {
        FinalMaskMapping::new(&mapping.internal_to_original, buf_words)
    }

    fn fill(prepared: &Self::Prepared, case: &Case, out: &mut [u32]) {
        if case.internal_dense_words.is_empty() {
            prepared.fill_internal_ids(&case.internal_ids, out);
        } else {
            prepared.fill_dense_words(&case.internal_dense_words, out);
        }
    }
}

impl Candidate for GlrMaskFinalDenseComplementCandidate {
    type Prepared = FinalMaskMapping;

    fn name() -> &'static str {
        "glrmask_final_dense_force_complement"
    }

    fn prepare(mapping: &Mapping, buf_words: usize) -> Self::Prepared {
        FinalMaskMapping::new(&mapping.internal_to_original, buf_words)
    }

    fn fill(prepared: &Self::Prepared, case: &Case, out: &mut [u32]) {
        if case.internal_dense_words.is_empty() {
            prepared.fill_internal_ids(&case.internal_ids, out);
        } else {
            prepared.fill_dense_words_complement(&case.internal_dense_words, out);
        }
    }
}

fn compute_block_entries(
    token_entries: &[Box<[(usize, u32)]>],
    buf_words: usize,
    block_size: usize,
) -> Vec<Box<[(usize, u32)]>> {
    let n_groups = token_entries.len().div_ceil(block_size);
    let mut groups = Vec::with_capacity(n_groups);
    for group_id in 0..n_groups {
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
        groups.push(
            touched
                .into_iter()
                .map(|word_idx| (word_idx, dense[word_idx]))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
    }
    groups
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
fn or_token(prepared: &GlrMaskLikePrepared, internal_id: usize, out: &mut [u32]) -> bool {
    let Some(entries) = prepared.token_entries.get(internal_id) else {
        return false;
    };
    for &(word_idx, mask) in entries.iter() {
        out[word_idx] |= mask;
    }
    !entries.is_empty()
}

fn or_internal_run(prepared: &GlrMaskLikePrepared, mut start: usize, end: usize, out: &mut [u32]) {
    while start < end && !start.is_multiple_of(4) {
        or_token(prepared, start, out);
        start += 1;
    }

    while start + 4 <= end && !start.is_multiple_of(8) {
        or_quad_group(prepared, start / 4, out);
        start += 4;
    }

    while start < end && !start.is_multiple_of(8) {
        or_token(prepared, start, out);
        start += 1;
    }

    while start + 8 <= end && !start.is_multiple_of(64) {
        or_byte_group(prepared, start / 8, out);
        start += 8;
    }

    while start < end && !start.is_multiple_of(64) {
        or_token(prepared, start, out);
        start += 1;
    }

    let group_start = start / 64;
    let group_end = end / 64;
    if group_start < group_end {
        or_full_group_run(prepared, group_start, group_end, out);
        start = group_end * 64;
    }

    while start + 8 <= end {
        or_byte_group(prepared, start / 8, out);
        start += 8;
    }

    while start + 4 <= end {
        or_quad_group(prepared, start / 4, out);
        start += 4;
    }

    while start < end {
        or_token(prepared, start, out);
        start += 1;
    }
}

fn or_internal_run_copy_first(
    prepared: &GlrMaskLikePrepared,
    mut start: usize,
    end: usize,
    out: &mut [u32],
    can_copy: bool,
) -> bool {
    let mut wrote = false;
    while start < end && !start.is_multiple_of(4) {
        wrote |= or_token(prepared, start, out);
        start += 1;
    }

    while start + 4 <= end && !start.is_multiple_of(8) {
        wrote |= or_quad_group(prepared, start / 4, out);
        start += 4;
    }

    while start < end && !start.is_multiple_of(8) {
        wrote |= or_token(prepared, start, out);
        start += 1;
    }

    while start + 8 <= end && !start.is_multiple_of(64) {
        wrote |= or_byte_group(prepared, start / 8, out);
        start += 8;
    }

    while start < end && !start.is_multiple_of(64) {
        wrote |= or_token(prepared, start, out);
        start += 1;
    }

    let group_start = start / 64;
    let group_end = end / 64;
    if group_start < group_end {
        wrote |= or_full_group_run_copy_first(prepared, group_start, group_end, out, can_copy && !wrote);
        start = group_end * 64;
    }

    while start + 8 <= end {
        wrote |= or_byte_group(prepared, start / 8, out);
        start += 8;
    }

    while start + 4 <= end {
        wrote |= or_quad_group(prepared, start / 4, out);
        start += 4;
    }

    while start < end {
        wrote |= or_token(prepared, start, out);
        start += 1;
    }
    wrote
}

#[inline(always)]
fn or_quad_group(prepared: &GlrMaskLikePrepared, group_id: usize, out: &mut [u32]) -> bool {
    let Some(entries) = prepared.quad_group_entries.get(group_id) else {
        return false;
    };
    for &(word_idx, mask) in entries.iter() {
        out[word_idx] |= mask;
    }
    !entries.is_empty()
}

#[inline(always)]
fn or_byte_group(prepared: &GlrMaskLikePrepared, group_id: usize, out: &mut [u32]) -> bool {
    let Some(entries) = prepared.byte_group_entries.get(group_id) else {
        return false;
    };
    for &(word_idx, mask) in entries.iter() {
        out[word_idx] |= mask;
    }
    !entries.is_empty()
}

fn group_entries_in(prepared: &GlrMaskLikePrepared, start: usize, end: usize) -> usize {
    prepared
        .group_entry_prefix
        .get(end)
        .zip(prepared.group_entry_prefix.get(start))
        .map(|(e, s)| e - s)
        .unwrap_or_else(|| {
            prepared.group_entries[start..end]
                .iter()
                .map(|entries| entries.len())
                .sum()
        })
}

fn or_full_group_run(prepared: &GlrMaskLikePrepared, start: usize, end: usize, out: &mut [u32]) {
    if start >= end {
        return;
    }

    let sparse_entries = group_entries_in(prepared, start, end);
    if sparse_entries > prepared.buf_words
        && end < prepared.group_dense_prefix.len()
        && out.len() >= prepared.buf_words
    {
        let before = &prepared.group_dense_prefix[start];
        let after = &prepared.group_dense_prefix[end];
        or_prefix_diff(out, before, after);
        return;
    }

    for group_id in start..end {
        if let Some(entries) = prepared.group_entries.get(group_id) {
            for &(word_idx, mask) in entries.iter() {
                out[word_idx] |= mask;
            }
        }
    }
}

fn or_full_group_run_copy_first(
    prepared: &GlrMaskLikePrepared,
    start: usize,
    end: usize,
    out: &mut [u32],
    can_copy: bool,
) -> bool {
    if start >= end {
        return false;
    }

    let sparse_entries = group_entries_in(prepared, start, end);
    if sparse_entries > prepared.buf_words
        && end < prepared.group_dense_prefix.len()
        && out.len() >= prepared.buf_words
    {
        let before = &prepared.group_dense_prefix[start];
        let after = &prepared.group_dense_prefix[end];
        if can_copy {
            copy_prefix_diff(out, before, after);
        } else {
            or_prefix_diff(out, before, after);
        }
        return true;
    }

    let mut wrote = false;
    for group_id in start..end {
        if let Some(entries) = prepared.group_entries.get(group_id) {
            for &(word_idx, mask) in entries.iter() {
                out[word_idx] |= mask;
            }
            wrote |= !entries.is_empty();
        }
    }
    wrote
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

#[inline(always)]
fn copy_dense(out: &mut [u32], dense: &[u32]) {
    let n = out.len().min(dense.len());
    out[..n].copy_from_slice(&dense[..n]);
}

fn parallel_copy_dense(out: &mut [u32], dense: &[u32]) {
    let n = out.len().min(dense.len());
    if n < 16_384 {
        copy_dense(out, dense);
        return;
    }
    out[..n]
        .par_chunks_mut(4096)
        .zip(dense[..n].par_chunks(4096))
        .for_each(|(dst, src)| dst.copy_from_slice(src));
}

#[inline(always)]
fn andnot_token(prepared: &GlrMaskLikePrepared, internal_id: usize, out: &mut [u32]) {
    let Some(entries) = prepared.token_entries.get(internal_id) else {
        return;
    };
    for &(word_idx, mask) in entries.iter() {
        out[word_idx] &= !mask;
    }
}

fn andnot_missing_ids(prepared: &GlrMaskLikePrepared, internal_ids: &[u32], out: &mut [u32]) {
    let mut expected = 0usize;
    for &raw in internal_ids {
        let selected = raw as usize;
        if selected > prepared.token_entries.len() {
            break;
        }
        while expected < selected {
            andnot_token(prepared, expected, out);
            expected += 1;
        }
        expected = selected.saturating_add(1);
    }
    while expected < prepared.token_entries.len() {
        andnot_token(prepared, expected, out);
        expected += 1;
    }
}
