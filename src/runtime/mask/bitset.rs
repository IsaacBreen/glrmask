//! Packed `Vec<u32>` vocabulary-mask bit operations.
//!
//! These helpers operate on the public mask layout: bit `t % 32` in word
//! `t / 32` represents original vocabulary token `t`.

pub(super) fn update_eos_mask(buf: &mut [u32], eos_token_id: Option<u32>, is_complete: bool) {
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

pub(super) fn set_token_bit(buf: &mut [u32], token_id: u32) {
    let word = token_id as usize / 32;
    let bit = token_id as usize % 32;
    if let Some(slot) = buf.get_mut(word) {
        *slot |= 1u32 << bit;
    }
}

pub(super) fn is_token_bit_set(buf: &[u32], token_id: u32) -> bool {
    let word = token_id as usize / 32;
    let bit = token_id as usize % 32;
    buf.get(word)
        .map(|slot| (*slot & (1u32 << bit)) != 0)
        .unwrap_or(false)
}

pub(super) fn for_each_set_token_bit(buf: &[u32], mut f: impl FnMut(u32)) {
    for (word_index, &word) in buf.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros();
            f((word_index as u32) * 32 + bit);
            remaining &= remaining - 1;
        }
    }
}

pub(super) fn eos_mask_bit(buf: &[u32], eos_token_id: Option<u32>) -> bool {
    let Some(eos_token_id) = eos_token_id else {
        return false;
    };
    let word = eos_token_id as usize / 32;
    let bit = eos_token_id as usize % 32;
    buf.get(word)
        .map(|slot| (*slot & (1u32 << bit)) != 0)
        .unwrap_or(false)
}

