use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BitSet {
    words: SmallVec<[u64; 1]>,
    len: usize,
}

impl BitSet {
    #[inline]
    fn assert_same_len(&self, other: &Self) {
        debug_assert_eq!(self.len, other.len);
    }

    #[inline]
    fn bit_position(&self, index: usize) -> Option<(usize, u32)> {
        (index < self.len).then_some((index / 64, (index % 64) as u32))
    }

    pub fn new(len: usize) -> Self {
        let mut words = SmallVec::new();
        words.resize(len.div_ceil(64), 0);
        Self {
            words,
            len,
        }
    }

    pub fn empty(len: usize) -> Self {
        Self::new(len)
    }

    pub fn all(len: usize) -> Self {
        let mut words = SmallVec::new();
        words.resize(len.div_ceil(64), u64::MAX);
        let mut set = Self {
            words,
            len,
        };
        set.mask_unused_bits();
        set
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.is_zero()
    }

    pub fn get(&self, i: usize) -> bool {
        let Some((word_index, bit_index)) = self.bit_position(i) else {
            return false;
        };
        (self.words[word_index] & (1u64 << bit_index)) != 0
    }

    pub fn contains(&self, i: usize) -> bool {
        self.get(i)
    }

    pub fn set(&mut self, i: usize) {
        if let Some((word_index, bit_index)) = self.bit_position(i) {
            self.words[word_index] |= 1u64 << bit_index;
        }
    }

    pub fn clear(&mut self, i: usize) {
        if let Some((word_index, bit_index)) = self.bit_position(i) {
            self.words[word_index] &= !(1u64 << bit_index);
        }
    }

    pub fn clear_all(&mut self) {
        self.words.fill(0);
    }

    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|word| word.count_ones() as usize).sum()
    }

    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|&word| word == 0)
    }

    pub fn union_with(&mut self, other: &BitSet) {
        self.assert_same_len(other);
        for (lhs, rhs) in self.words.iter_mut().zip(&other.words) {
            *lhs |= *rhs;
        }
    }

    /// Union `left ∩ right` into this set and report whether the intersection
    /// contained any bits.
    ///
    /// This avoids allocating a temporary intersection and is useful when a
    /// large collection of bitset pairs contributes to one accumulated set.
    pub fn union_intersection_with(&mut self, left: &BitSet, right: &BitSet) -> bool {
        self.assert_same_len(left);
        self.assert_same_len(right);
        let mut any = false;
        for ((dst, lhs), rhs) in self.words.iter_mut().zip(&left.words).zip(&right.words) {
            let intersection = *lhs & *rhs;
            any |= intersection != 0;
            *dst |= intersection;
        }
        any
    }

    /// Union `other` into this set and return exactly the bits newly added.
    ///
    /// This combines the common `other.difference(self)` followed by
    /// `self.union_with(delta)` pattern into one pass over the words.
    pub fn union_with_delta(&mut self, other: &BitSet) -> BitSet {
        self.assert_same_len(other);
        let mut delta = Self::new(self.len);
        for ((lhs, delta_word), rhs) in self
            .words
            .iter_mut()
            .zip(delta.words.iter_mut())
            .zip(&other.words)
        {
            *delta_word = *rhs & !*lhs;
            *lhs |= *rhs;
        }
        delta
    }

    pub fn union(&self, other: &Self) -> Self {
        self.assert_same_len(other);
        let mut out = self.clone();
        out.union_with(other);
        out
    }

    pub fn intersect_with(&mut self, other: &BitSet) {
        self.assert_same_len(other);
        for (lhs, rhs) in self.words.iter_mut().zip(&other.words) {
            *lhs &= *rhs;
        }
    }

    pub fn intersection(&self, other: &Self) -> Self {
        self.assert_same_len(other);
        let mut out = self.clone();
        out.intersect_with(other);
        out
    }

    pub fn difference(&self, other: &Self) -> Self {
        self.assert_same_len(other);
        let mut out = self.clone();
        for (lhs, rhs) in out.words.iter_mut().zip(&other.words) {
            *lhs &= !*rhs;
        }
        out.mask_unused_bits();
        out
    }

    pub fn complement(&self) -> Self {
        let mut out = self.clone();
        for word in &mut out.words {
            *word = !*word;
        }
        out.mask_unused_bits();
        out
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.assert_same_len(other);
        self.words
            .iter()
            .zip(&other.words)
            .all(|(lhs, rhs)| (*lhs & *rhs) == 0)
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        self.assert_same_len(other);
        self.words
            .iter()
            .zip(&other.words)
            .all(|(lhs, rhs)| (*lhs & !*rhs) == 0)
    }

    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(word_idx, &word)| {
            let base = word_idx * 64;
            BitIter { word, base }
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.iter_ones()
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn fill_u32_mask(&self, buf: &mut [u32]) {
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

    fn mask_unused_bits(&mut self) {
        let rem = self.len % 64;
        if rem == 0 {
            return;
        }
        if let Some(last) = self.words.last_mut() {
            *last &= (1u64 << rem) - 1;
        }
    }
}

impl Default for BitSet {
    fn default() -> Self {
        Self::new(0)
    }
}

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
        self.word &= self.word - 1;
        Some(self.base + tz)
    }
}



#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::BitSet;

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct LegacyBitSet {
        words: Vec<u64>,
        len: usize,
    }

    #[test]
    fn inline_words_preserve_legacy_bincode_wire_shape() {
        for len in [0usize, 1, 64, 65, 130] {
            let mut current = BitSet::new(len);
            for bit in [0usize, 63, 64, 129] {
                if bit < len {
                    current.set(bit);
                }
            }
            let legacy = LegacyBitSet {
                words: current.words().to_vec(),
                len,
            };
            let current_bytes = bincode::serialize(&current).unwrap();
            let legacy_bytes = bincode::serialize(&legacy).unwrap();
            assert_eq!(current_bytes, legacy_bytes);
            assert_eq!(
                bincode::deserialize::<BitSet>(&legacy_bytes).unwrap(),
                current,
            );
            assert_eq!(
                bincode::deserialize::<LegacyBitSet>(&current_bytes).unwrap(),
                legacy,
            );
        }
    }

    #[test]
    fn union_with_delta_reports_only_new_bits() {
        let mut left = BitSet::new(130);
        left.set(0);
        left.set(64);
        left.set(129);

        let mut right = BitSet::new(130);
        right.set(0);
        right.set(63);
        right.set(64);
        right.set(65);
        right.set(129);

        let delta = left.union_with_delta(&right);

        assert_eq!(delta.iter_ones().collect::<Vec<_>>(), vec![63, 65]);
        assert_eq!(left.iter_ones().collect::<Vec<_>>(), vec![0, 63, 64, 65, 129]);
    }

    #[test]
    fn union_intersection_with_accumulates_exact_intersection() {
        let mut accumulated = BitSet::new(130);
        accumulated.set(1);

        let mut left = BitSet::new(130);
        for bit in [0, 63, 64, 65, 129] {
            left.set(bit);
        }
        let mut right = BitSet::new(130);
        for bit in [63, 65, 66, 129] {
            right.set(bit);
        }

        assert!(accumulated.union_intersection_with(&left, &right));
        assert_eq!(
            accumulated.iter_ones().collect::<Vec<_>>(),
            vec![1, 63, 65, 129],
        );

        let empty = BitSet::new(130);
        assert!(!accumulated.union_intersection_with(&left, &empty));
    }
}
