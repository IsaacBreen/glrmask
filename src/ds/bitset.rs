use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BitSet {
    words: Vec<u64>,
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
        Self {
            words: vec![0; len.div_ceil(64)],
            len,
        }
    }

    pub fn empty(len: usize) -> Self {
        Self::new(len)
    }

    pub fn all(len: usize) -> Self {
        let mut set = Self {
            words: vec![u64::MAX; len.div_ceil(64)],
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

    pub(crate) fn intersect_with(&mut self, other: &BitSet) {
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
    use super::BitSet;

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
}
