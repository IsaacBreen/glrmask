#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::hash::{Hash, Hasher};

/// Sparse bitset that tracks which words became non-zero so clearing is cheap.
pub struct SparseStateSet {
    pub(crate) words: Vec<u64>,
    pub(crate) dirty_words: Vec<usize>,
}

impl SparseStateSet {
    pub fn new(num_bits: usize) -> Self {
        Self {
            words: vec![0; num_bits.div_ceil(64)],
            dirty_words: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, bit: usize) -> bool {
        let word_idx = bit / 64;
        let bit_mask = 1u64 << (bit % 64);
        let word = &mut self.words[word_idx];
        if (*word & bit_mask) != 0 {
            return false;
        }
        if *word == 0 {
            self.dirty_words.push(word_idx);
        }
        *word |= bit_mask;
        true
    }

    #[inline]
    pub fn insert_many(&mut self, states: &[u32]) {
        for &state in states {
            self.insert(state as usize);
        }
    }

    pub fn clear(&mut self) {
        for &idx in &self.dirty_words {
            self.words[idx] = 0;
        }
        self.dirty_words.clear();
    }
}

#[derive(Clone, Eq, Debug, Default)]
pub struct CompressedStateSet {
    /// Sorted `(word_index, word_bits)` pairs for non-zero words only.
    pub(crate) words: Vec<(u32, u64)>,
    pub(crate) hash: u64,
}

impl PartialEq for CompressedStateSet {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.words == other.words
    }
}

impl Hash for CompressedStateSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

impl CompressedStateSet {
    pub fn new() -> Self {
        Self {
            words: Vec::new(),
            hash: 0,
        }
    }

    #[inline]
    pub fn from_sparse(sparse: &SparseStateSet) -> Self {
        let mut result = Self::new();
        Self::reuse_from_sparse(sparse, &mut result, &mut Vec::new());
        result
    }

    #[inline]
    pub fn reuse_from_sparse(
        sparse: &SparseStateSet,
        buffer: &mut Self,
        _sort_scratch: &mut Vec<usize>,
    ) {
        buffer.words.clear();

        let mut hash = 0u64;
        for &idx in &sparse.dirty_words {
            let word = sparse.words[idx];
            buffer.words.push((idx as u32, word));
            hash ^= (idx as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
            hash ^= word.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }

        buffer.words.sort_unstable_by_key(|&(idx, _)| idx);
        buffer.hash = hash;
    }

    #[inline]
    pub fn iter(&self) -> CompressedStateSetIter<'_> {
        CompressedStateSetIter {
            set: self,
            idx: 0,
            current_word: 0,
            current_word_idx: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.words
            .iter()
            .map(|(_, word)| word.count_ones() as usize)
            .sum()
    }
}

pub struct CompressedStateSetIter<'a> {
    set: &'a CompressedStateSet,
    idx: usize,
    current_word: u64,
    current_word_idx: u32,
}

impl<'a> Iterator for CompressedStateSetIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_word != 0 {
                let trailing = self.current_word.trailing_zeros();
                self.current_word &= !(1u64 << trailing);
                return Some(self.current_word_idx as usize * 64 + trailing as usize);
            }

            if self.idx >= self.set.words.len() {
                return None;
            }

            let (word_idx, word) = self.set.words[self.idx];
            self.current_word = word;
            self.current_word_idx = word_idx;
            self.idx += 1;
        }
    }
}