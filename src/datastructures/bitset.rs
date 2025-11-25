#![allow(dead_code)]

use crate::datastructures::hybrid_bitset::RangeSet;
use range_set_blaze::RangeSetBlaze;
use std::iter::FromIterator;
use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Sub, SubAssign,
};

const WORD_SIZE: usize = 64;

/// A bitset implementation using a vector of u64s.
#[derive(Default, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct Bitset {
    pub(crate) words: Vec<u64>,
}

impl Bitset {
    /// Creates a new, empty Bitset.
    pub fn zeros() -> Self {
        Bitset { words: Vec::new() }
    }

    /// Creates a new Bitset with all indices from 0 up to `len` (exclusive) set to true.
    pub fn ones(len: usize) -> Self {
        if len == 0 {
            return Self::zeros();
        }
        let num_words = (len + WORD_SIZE - 1) / WORD_SIZE;
        let mut words = vec![0; num_words];
        let full_words = len / WORD_SIZE;
        for i in 0..full_words {
            words[i] = u64::MAX;
        }
        let remaining_bits = len % WORD_SIZE;
        if remaining_bits > 0 {
            words[full_words] = (1u64 << remaining_bits).wrapping_sub(1);
        }
        let mut set = Bitset { words };
        set.trim();
        set
    }

    fn word_index(index: usize) -> usize {
        index / WORD_SIZE
    }
    fn bit_index(index: usize) -> usize {
        index % WORD_SIZE
    }

    /// Checks if a specific index is set.
    pub fn contains(&self, index: usize) -> bool {
        let word_idx = Self::word_index(index);
        if word_idx >= self.words.len() {
            return false;
        }
        let bit_idx = Self::bit_index(index);
        (self.words[word_idx] & (1u64 << bit_idx)) != 0
    }

    /// Inserts an index into the set. Returns true if the index was not already present.
    pub fn insert(&mut self, index: usize) -> bool {
        let word_idx = Self::word_index(index);
        if word_idx >= self.words.len() {
            self.words.resize(word_idx + 1, 0);
        }
        let bit_idx = Self::bit_index(index);
        let mask = 1u64 << bit_idx;
        let was_set = (self.words[word_idx] & mask) != 0;
        self.words[word_idx] |= mask;
        !was_set
    }

    /// Removes an index from the set. Returns true if the index was present.
    pub fn remove(&mut self, index: usize) -> bool {
        let word_idx = Self::word_index(index);
        if word_idx >= self.words.len() {
            return false;
        }
        let bit_idx = Self::bit_index(index);
        let mask = 1u64 << bit_idx;
        let was_set = (self.words[word_idx] & mask) != 0;
        self.words[word_idx] &= !mask;
        self.trim();
        was_set
    }

    /// Removes trailing zero words.
    fn trim(&mut self) {
        while self.words.last() == Some(&0) {
            self.words.pop();
        }
    }

    /// Returns true if the bitset contains no set bits.
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Returns the exact number of set bits (cardinality).
    pub fn len(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Removes all elements from the set.
    pub fn clear(&mut self) {
        self.words.clear();
    }

    /// Returns an iterator over the indices of the set bits.
    pub fn iter_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(|(word_idx, &word)| {
                (0..WORD_SIZE).filter_map(move |bit_idx| {
                    if (word & (1u64 << bit_idx)) != 0 {
                        Some(word_idx * WORD_SIZE + bit_idx)
                    } else {
                        None
                    }
                })
            })
    }

    pub fn from_words_vec(words: Vec<u64>) -> Self {
        Bitset {
            words,
        }
    }
}

impl FromIterator<usize> for Bitset {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let mut set = Bitset::zeros();
        for i in iter {
            set.insert(i);
        }
        set
    }
}

// --- Bitwise Operations ---

impl BitAnd for &Bitset {
    type Output = Bitset;
    fn bitand(self, rhs: Self) -> Self::Output {
        let min_len = self.words.len().min(rhs.words.len());
        let mut new_words = vec![0; min_len];
        for i in 0..min_len {
            new_words[i] = self.words[i] & rhs.words[i];
        }
        let mut result = Bitset { words: new_words };
        result.trim();
        result
    }
}

impl BitOr for &Bitset {
    type Output = Bitset;
    fn bitor(self, rhs: Self) -> Self::Output {
        let max_len = self.words.len().max(rhs.words.len());
        let mut new_words = vec![0; max_len];
        for i in 0..max_len {
            let w1 = self.words.get(i).copied().unwrap_or(0);
            let w2 = rhs.words.get(i).copied().unwrap_or(0);
            new_words[i] = w1 | w2;
        }
        let mut result = Bitset { words: new_words };
        result.trim();
        result
    }
}

impl BitXor for &Bitset {
    type Output = Bitset;
    fn bitxor(self, rhs: Self) -> Self::Output {
        let max_len = self.words.len().max(rhs.words.len());
        let mut new_words = vec![0; max_len];
        for i in 0..max_len {
            let w1 = self.words.get(i).copied().unwrap_or(0);
            let w2 = rhs.words.get(i).copied().unwrap_or(0);
            new_words[i] = w1 ^ w2;
        }
        let mut result = Bitset { words: new_words };
        result.trim();
        result
    }
}

impl Sub for &Bitset {
    type Output = Bitset;
    fn sub(self, rhs: Self) -> Self::Output {
        let max_len = self.words.len().max(rhs.words.len());
        let mut new_words = vec![0; max_len];
        for i in 0..max_len {
            let w1 = self.words.get(i).copied().unwrap_or(0);
            let w2 = rhs.words.get(i).copied().unwrap_or(0);
            new_words[i] = w1 & !w2;
        }
        let mut result = Bitset { words: new_words };
        result.trim();
        result
    }
}

// --- In-place Bitwise Operations ---
impl BitAndAssign for Bitset {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = &*self & &rhs;
    }
}
impl BitOrAssign for Bitset {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = &*self | &rhs;
    }
}
impl BitXorAssign for Bitset {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = &*self ^ &rhs;
    }
}
impl SubAssign for Bitset {
    fn sub_assign(&mut self, rhs: Self) {
        *self = &*self - &rhs;
    }
}

impl BitAndAssign<&Bitset> for Bitset {
    fn bitand_assign(&mut self, rhs: &Bitset) {
        *self = &*self & rhs;
    }
}
impl BitOrAssign<&Bitset> for Bitset {
    fn bitor_assign(&mut self, rhs: &Bitset) {
        *self = &*self | rhs;
    }
}
impl BitXorAssign<&Bitset> for Bitset {
    fn bitxor_assign(&mut self, rhs: &Bitset) {
        *self = &*self ^ rhs;
    }
}
impl SubAssign<&Bitset> for Bitset {
    fn sub_assign(&mut self, rhs: &Bitset) {
        *self = &*self - rhs;
    }
}

// --- Conversions ---

impl From<&RangeSet> for Bitset {
    fn from(hybrid: &RangeSet) -> Self {
        hybrid.iter_indices().collect()
    }
}

impl From<RangeSet> for Bitset {
    fn from(hybrid: RangeSet) -> Self {
        Self::from(&hybrid)
    }
}

impl From<&RangeSetBlaze<usize>> for Bitset {
    fn from(rsb: &RangeSetBlaze<usize>) -> Self {
        rsb.iter().collect()
    }
}

impl From<RangeSetBlaze<usize>> for Bitset {
    fn from(rsb: RangeSetBlaze<usize>) -> Self {
        Self::from(&rsb)
    }
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_contains() {
        let mut set = Bitset::zeros();
        assert!(!set.contains(0));
        assert!(!set.contains(63));
        assert!(!set.contains(64));
        assert!(!set.contains(100));

        set.insert(63);
        assert!(set.contains(63));
        assert_eq!(set.words.len(), 1);
        assert_eq!(set.words[0], 1u64 << 63);

        set.insert(100);
        assert!(set.contains(100));
        assert_eq!(set.words.len(), 2);
        assert_eq!(set.words[0], 1u64 << 63);
        assert_eq!(set.words[1], 1u64 << (100 % 64));

        assert!(!set.contains(0));
        assert!(!set.contains(64));
    }

    #[test]
    fn test_remove() {
        let mut set = Bitset::zeros();
        set.insert(10);
        set.insert(20);
        set.insert(70);

        assert!(set.contains(10));
        assert!(set.contains(20));
        assert!(set.contains(70));

        assert!(set.remove(20));
        assert!(!set.contains(20));
        assert!(!set.remove(20)); // Already removed

        assert!(set.remove(70));
        assert!(!set.contains(70));
        assert_eq!(set.words.len(), 1); // Should trim
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut set = Bitset::zeros();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);

        set.insert(5);
        set.insert(10);
        assert!(!set.is_empty());
        assert_eq!(set.len(), 2);

        set.insert(5); // no change
        assert_eq!(set.len(), 2);

        set.remove(5);
        assert_eq!(set.len(), 1);

        set.remove(10);
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn test_iter() {
        let mut set = Bitset::zeros();
        set.insert(1);
        set.insert(65);
        set.insert(10);
        set.insert(127);

        let mut indices: Vec<_> = set.iter_indices().collect();
        indices.sort();
        assert_eq!(indices, vec![1, 10, 65, 127]);
    }

    #[test]
    fn test_bitwise_ops() {
        let set1 = Bitset::from_iter(vec![1, 2, 3, 100]);
        let set2 = Bitset::from_iter(vec![2, 3, 4, 101]);

        let and = &set1 & &set2;
        assert_eq!(and.iter_indices().collect::<Vec<_>>(), vec![2, 3]);

        let or = &set1 | &set2;
        let mut or_vec = or.iter_indices().collect::<Vec<_>>();
        or_vec.sort();
        assert_eq!(or_vec, vec![1, 2, 3, 4, 100, 101]);

        let xor = &set1 ^ &set2;
        let mut xor_vec = xor.iter_indices().collect::<Vec<_>>();
        xor_vec.sort();
        assert_eq!(xor_vec, vec![1, 4, 100, 101]);

        let sub = &set1 - &set2;
        let mut sub_vec = sub.iter_indices().collect::<Vec<_>>();
        sub_vec.sort();
        assert_eq!(sub_vec, vec![1, 100]);
    }
    
    #[test]
    fn test_ones() {
        let set = Bitset::ones(65);
        assert_eq!(set.len(), 65);
        for i in 0..65 {
            assert!(set.contains(i));
        }
        assert!(!set.contains(65));
        assert_eq!(set.words.len(), 2);
        assert_eq!(set.words[0], u64::MAX);
        assert_eq!(set.words[1], 1);
    }
}
