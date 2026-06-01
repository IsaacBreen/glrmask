//! Low-level mask-buffer operations shared by runtime cache building and mask materialization.
//!
//! These functions deliberately know nothing about Parser DWAs, lexer states,
//! grammar terminals, or ConstraintState.  They are the word-level Boolean
//! algebra over output-vocabulary masks.

/// Dense buf OR: `buf[i] |= mask[i]` for all i in min(buf.len(), mask.len()).
/// Processes u64 chunks for reduced loop overhead and better throughput.
#[inline(always)]
pub(crate) fn or_dense_buf(buf: &mut [u32], mask: &[u32]) {
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
pub(crate) fn andnot_dense_buf(buf: &mut [u32], mask: &[u32]) {
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
pub(crate) fn copy_dense_buf(buf: &mut [u32], mask: &[u32]) {
    let n = buf.len().min(mask.len());
    unsafe {
        std::ptr::copy_nonoverlapping(mask.as_ptr(), buf.as_mut_ptr(), n);
    }
}

#[inline(always)]
pub(crate) fn or_sparse_buf_entries(buf: &mut [u32], entries: &[(u16, u32)]) {
    for &(word_idx, mask) in entries {
        unsafe {
            let slot = buf.get_unchecked_mut(word_idx as usize);
            *slot |= mask;
        }
    }
}

#[inline(always)]
pub(crate) fn andnot_sparse_buf_entries(buf: &mut [u32], entries: &[(u16, u32)]) {
    for &(word_idx, mask) in entries {
        unsafe {
            let slot = buf.get_unchecked_mut(word_idx as usize);
            *slot &= !mask;
        }
    }
}

#[inline(always)]
pub(crate) fn count_complement_subgroups(missing: u64, valid_mask: u64) -> (u32, u32, u32) {
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
