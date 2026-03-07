//! Dense bitvector implementation.
//!
//! Used for token masks and set operations on token IDs.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use serde::{Deserialize, Serialize};

/// A dense bitvector stored as a `Vec<u64>`.
///
/// Bit `i` is stored in word `i / 64`, bit position `i % 64`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BitSet {
    words: Vec<u64>,
    len: usize,
}

impl BitSet {
    /// Create a new bitset that can hold at least `len` bits, all initially zero.
    pub fn new(len: usize) -> Self {
        unimplemented!()
    }

    /// Number of bits this bitset can hold.
    pub fn len(&self) -> usize {
        unimplemented!()
    }

    /// Whether the bitset has zero capacity.
    pub fn is_empty(&self) -> bool {
        unimplemented!()
    }

    /// Get the value of bit `i`.
    pub fn get(&self, i: usize) -> bool {
        unimplemented!()
    }

    /// Set bit `i` to 1.
    pub fn set(&mut self, i: usize) {
        unimplemented!()
    }

    /// Clear bit `i` to 0.
    pub fn clear(&mut self, i: usize) {
        unimplemented!()
    }

    /// Set all bits to 0.
    pub fn clear_all(&mut self) {
        unimplemented!()
    }

    /// Number of set bits (population count).
    pub fn count_ones(&self) -> usize {
        unimplemented!()
    }

    /// Whether all bits are zero.
    pub fn is_zero(&self) -> bool {
        unimplemented!()
    }

    /// In-place OR: `self |= other`.
    pub fn union_with(&mut self, other: &BitSet) {
        unimplemented!()
    }

    /// In-place AND: `self &= other`.
    pub fn intersect_with(&mut self, other: &BitSet) {
        unimplemented!()
    }

    /// Iterator over set bit indices.
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(word_idx, &word)| {
            let base = word_idx * 64;
            BitIter { word, base }
        })
    }

    /// Access the underlying word slice (for filling masks as `&[u64]`).
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Mutable access to underlying words.
    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }

    /// Fill a `&mut [u32]` mask buffer from this bitset.
    ///
    /// The mask uses the convention: token `i` is allowed iff
    /// `buf[i / 32] & (1u32 << (i % 32)) != 0`.
    pub fn fill_u32_mask(&self, buf: &mut [u32]) {
        // Each u64 word maps to two consecutive u32 entries
        for (i, &word) in self.words.iter().enumerate() {
            let base = i * 2;
            if base < buf.len() {
                buf[base] = word as u32;
            }
            if base + 1 < buf.len() {
                buf[base + 1] = (word >> 32) as u32;
            }
        }
    }
}

/// Iterator over set bits in a u64 word.
struct BitIter {
    word: u64,
    base: usize,
}

impl Iterator for BitIter {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.word == 0 {
            return None;
        }
        let tz = self.word.trailing_zeros() as usize;
        self.word &= self.word - 1; // clear lowest set bit
        Some(self.base + tz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        let mut bs = BitSet::new(128);
        assert!(!bs.get(0));
        bs.set(0);
        assert!(bs.get(0));
        bs.set(63);
        bs.set(64);
        bs.set(127);
        assert_eq!(bs.count_ones(), 4);

        let ones: Vec<usize> = bs.iter_ones().collect();
        assert_eq!(ones, vec![0, 63, 64, 127]);
    }

    #[test]
    fn test_union_intersect() {
        let mut a = BitSet::new(64);
        let mut b = BitSet::new(64);
        a.set(0);
        a.set(1);
        b.set(1);
        b.set(2);

        let mut u = a.clone();
        u.union_with(&b);
        assert_eq!(u.count_ones(), 3);

        let mut i = a.clone();
        i.intersect_with(&b);
        assert_eq!(i.count_ones(), 1);
        assert!(i.get(1));
    }

    #[test]
    fn test_fill_u32_mask() {
        let mut bs = BitSet::new(128);
        bs.set(0);
        bs.set(31);
        bs.set(32);
        bs.set(95);
        let mut buf = vec![0u32; 4];
        bs.fill_u32_mask(&mut buf);
        assert!(buf[0] & (1u32 << 0) != 0);
        assert!(buf[0] & (1u32 << 31) != 0);
        assert!(buf[1] & (1u32 << 0) != 0);
        assert!(buf[2] & (1u32 << 31) != 0);
    }
}
