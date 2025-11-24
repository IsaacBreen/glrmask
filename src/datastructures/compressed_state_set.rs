use std::hash::{Hash, Hasher};
use ahash::AHasher;
use profiler_macro::time_it;
use crate::datastructures::state_set::StateSet;

#[derive(Clone)]
pub struct DenseStateSet {
    pub words: Vec<u64>,
}

impl DenseStateSet {
    #[time_it]
    pub fn new(num_bits: usize) -> Self {
        let num_words = (num_bits + 63) / 64;
        Self {
            words: vec![0; num_words],
        }
    }
}

pub struct SparseStateSet {
    pub dense: DenseStateSet,
    pub dirty_words: Vec<usize>,
}

impl SparseStateSet {
    #[time_it]
    pub fn new(num_bits: usize) -> Self {
        Self {
            dense: DenseStateSet::new(num_bits),
            dirty_words: Vec::new(),
        }
    }

    // #[time_it]
    #[inline(always)]
    pub fn insert(&mut self, bit: usize) -> bool {
        let word_idx = bit / 64;
        let bit_mask = 1u64 << (bit % 64);
        unsafe {
            let word = self.dense.words.get_unchecked_mut(word_idx);
            if (*word & bit_mask) == 0 {
                if *word == 0 {
                    self.dirty_words.push(word_idx);
                }
                *word |= bit_mask;
                true
            } else {
                false
            }
        }
    }

    pub fn clear(&mut self) {
        for &idx in &self.dirty_words {
            self.dense.words[idx] = 0;
        }
        self.dirty_words.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.dirty_words.is_empty()
    }
}

#[derive(Clone, Eq, Debug, Default)]
pub struct CompressedStateSet {
    // Sorted by word index. (word_index, word_value)
    pub words: Vec<(u32, u64)>,
    pub hash: u64,
}

impl PartialEq for CompressedStateSet {
    #[time_it]
    fn eq(&self, other: &Self) -> bool {
        if self.hash != other.hash {
            return false;
        }
        self.words == other.words
    }
}

impl Hash for CompressedStateSet {
    #[time_it]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

impl CompressedStateSet {
    #[inline]
    pub fn new() -> Self {
        Self {
            words: Vec::new(),
            hash: 0,
        }
    }

    #[inline]
    #[time_it]
    pub fn from_sparse(sparse: &SparseStateSet) -> Self {
        // Optimization: Sort indices first, then build words.
        let mut indices = sparse.dirty_words.clone();
        indices.sort_unstable();

        let mut words = Vec::with_capacity(indices.len());
        for &idx in &indices {
            words.push((idx as u32, sparse.dense.words[idx]));
        }

        let mut hasher = AHasher::default();
        // Optimization: Hash as raw bytes
        unsafe {
            let slice = std::slice::from_raw_parts(
                words.as_ptr() as *const u8,
                words.len() * std::mem::size_of::<(u32, u64)>()
            );
            hasher.write(slice);
        }
        let hash = hasher.finish();

        Self { words, hash }
    }

    #[inline]
    #[time_it]
    pub fn reuse_from_sparse(sparse: &SparseStateSet, buffer: &mut Self) {
        buffer.words.clear();

        // We can't easily reuse a buffer for sorting indices without allocating,
        // unless we pass one in. For now, let's just clone dirty_words which is small (Vec<usize>).
        let mut indices = sparse.dirty_words.clone();
        indices.sort_unstable();

        for &idx in &indices {
            buffer.words.push((idx as u32, sparse.dense.words[idx]));
        }

        // Recompute hash for the buffer
        buffer.recompute_hash();
    }

    #[inline(always)]
    // #[time_it]
    pub fn insert(&mut self, bit: usize) -> bool {
        let word_idx = (bit >> 6) as u32;
        let mask = 1u64 << (bit & 0x3F);
        match self.words.binary_search_by_key(&word_idx, |&(w, _)| w) {
            Ok(idx) => {
                let old = self.words[idx].1;
                if (old & mask) == 0 {
                    self.words[idx].1 |= mask;
                    true
                } else {
                    false
                }
            }
            Err(idx) => {
                self.words.insert(idx, (word_idx, mask));
                true
            }
        }
    }

    #[time_it]
    pub fn contains(&self, bit: usize) -> bool {
        let word_idx = (bit >> 6) as u32;
        let mask = 1u64 << (bit & 0x3F);
        match self.words.binary_search_by_key(&word_idx, |&(w, _)| w) {
            Ok(idx) => (self.words[idx].1 & mask) != 0,
            Err(_) => false,
        }
    }

    pub fn clear(&mut self) {
        self.words.clear();
        self.hash = 0;
    }

    #[inline]
    #[time_it]
    pub fn recompute_hash(&mut self) {
        let mut hasher = AHasher::default();
        // Optimization: Hash as raw bytes for speed
        unsafe {
            let slice = std::slice::from_raw_parts(
                self.words.as_ptr() as *const u8,
                self.words.len() * std::mem::size_of::<(u32, u64)>()
            );
            hasher.write(slice);
        }
        self.hash = hasher.finish();
    }

    #[inline]
    #[time_it]
    pub fn iter(&self) -> CompressedStateSetIter {
        CompressedStateSetIter {
            set: self,
            idx: 0,
            current_word: 0,
            current_word_idx: 0,
        }
    }

    #[inline]
    #[time_it]
    pub fn len(&self) -> usize {
        self.words.iter().map(|(_, w)| w.count_ones() as usize).sum()
    }
}

impl FromIterator<usize> for CompressedStateSet {
    #[time_it]
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        let mut set = CompressedStateSet::new();
        for i in iter {
            set.insert(i);
        }
        set
    }
}

impl StateSet for CompressedStateSet {
    type Iter<'a> = CompressedStateSetIter<'a>;

    fn with_capacity(_: usize) -> Self { Self::new() }
    fn insert(&mut self, state: usize) -> bool { self.insert(state) }
    fn contains(&self, state: usize) -> bool { self.contains(state) }
    fn len(&self) -> usize { self.len() }
    fn is_empty(&self) -> bool { self.words.is_empty() }
    fn clear(&mut self) { self.clear() }
    fn iter<'a>(&'a self) -> Self::Iter<'a> { self.iter() }
    fn recompute_hash(&mut self) { self.recompute_hash() }
}

pub struct CompressedStateSetIter<'a> {
    set: &'a CompressedStateSet,
    idx: usize,
    current_word: u64,
    current_word_idx: u32,
}

impl<'a> Iterator for CompressedStateSetIter<'a> {
    type Item = usize;

    // #[time_it]
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
            let (w_idx, w) = self.set.words[self.idx];
            self.current_word = w;
            self.current_word_idx = w_idx;
            self.idx += 1;
        }
    }
}