use std::hash::{Hash, Hasher};
use rustc_hash::FxHasher;
use crate::datastructures::state_set::StateSet;
use crate::datastructures::bitset::Bitset;

/// Type alias for backward compatibility.
/// DenseStateSet is now unified with Bitset.
pub type DenseStateSet = Bitset;

/// A sparse state set that tracks which words have been modified for efficient clearing.
/// Uses a dense bitset internally but maintains a list of dirty word indices.
pub struct SparseStateSet {
    pub dense: Bitset,
    pub dirty_words: Vec<usize>,
}

impl SparseStateSet {
    pub fn new(num_bits: usize) -> Self {
        Self {
            dense: Bitset::new(num_bits),
            dirty_words: Vec::new(),
        }
    }

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

    /// Bulk insert multiple states from a sorted slice of u32.
    /// More efficient than individual inserts for large closures.
    #[inline]
    pub fn insert_many(&mut self, states: &[u32]) {
        for &state in states {
            self.insert(state as usize);
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

/// A compressed representation of a state set using sparse storage.
/// Stores only non-zero words as (word_index, word_value) pairs.
#[derive(Clone, Eq, Debug, Default)]
pub struct CompressedStateSet {
    /// Sorted by word index. (word_index, word_value)
    pub words: Vec<(u32, u64)>,
    pub hash: u64,
}

impl PartialEq for CompressedStateSet {
    fn eq(&self, other: &Self) -> bool {
        if self.hash != other.hash {
            return false;
        }
        self.words == other.words
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
        let mut words = Vec::with_capacity(sparse.dirty_words.len());
        for &idx in &sparse.dirty_words {
            words.push((idx as u32, sparse.dense.words[idx]));
        }
        
        // Sort for binary search compatibility
        words.sort_unstable_by_key(|&(idx, _)| idx);

        // Compute order-independent XOR hash
        let mut hash = 0u64;
        for &(idx, word) in &words {
            hash ^= (idx as u64).wrapping_mul(0x517cc1b727220a95);
            hash ^= word.wrapping_mul(0x9e3779b97f4a7c15);
        }

        Self { words, hash }
    }

    #[inline]
    pub fn reuse_from_sparse(sparse: &SparseStateSet, buffer: &mut Self, _sort_scratch: &mut Vec<usize>) {
        buffer.words.clear();

        // Compute hash while copying - single pass optimization
        let mut hash = 0u64;
        for &idx in &sparse.dirty_words {
            let word = sparse.dense.words[idx];
            buffer.words.push((idx as u32, word));
            // XOR-based order-independent hash
            hash ^= (idx as u64).wrapping_mul(0x517cc1b727220a95);
            hash ^= word.wrapping_mul(0x9e3779b97f4a7c15);
        }
        buffer.hash = hash;

        // Sort for binary search compatibility
        buffer.words.sort_unstable_by_key(|&(idx, _)| idx);
    }

    #[inline(always)]
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
    pub fn recompute_hash(&mut self) {
        let mut hasher = FxHasher::default();
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
    pub fn iter(&self) -> CompressedStateSetIter {
        CompressedStateSetIter {
            set: self,
            idx: 0,
            current_word: 0,
            current_word_idx: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.words.iter().map(|(_, w)| w.count_ones() as usize).sum()
    }

    /// Create a normalized version of this state set by mapping each state to its representative.
    /// This is used for NFA bisimulation: states that are bisimilar (have identical future behavior)
    /// map to the same representative, so equivalent NFA subsets become identical after normalization.
    /// 
    /// The state_to_rep slice maps state_id -> representative_state_id.
    /// Only states present in this set are mapped.
    #[inline]
    pub fn normalized(&self, state_to_rep: &[usize]) -> Self {
        // Collect all mapped states
        let mut mapped_states: Vec<usize> = Vec::with_capacity(self.len());
        for state in self.iter() {
            if state < state_to_rep.len() {
                mapped_states.push(state_to_rep[state]);
            } else {
                mapped_states.push(state);
            }
        }
        
        // Sort and deduplicate (multiple states may map to same representative)
        mapped_states.sort_unstable();
        mapped_states.dedup();
        
        // Build compressed set from sorted unique states
        let mut result = Self::new();
        for state in mapped_states {
            result.insert(state);
        }
        result.recompute_hash();
        result
    }

    /// Efficiently normalize and store into a preallocated buffer.
    /// Same as normalized() but reuses allocations.
    #[inline]
    pub fn normalize_into(&self, state_to_rep: &[usize], buffer: &mut Self, scratch: &mut Vec<usize>) {
        scratch.clear();
        scratch.reserve(self.len());
        
        for state in self.iter() {
            if state < state_to_rep.len() {
                scratch.push(state_to_rep[state]);
            } else {
                scratch.push(state);
            }
        }
        
        scratch.sort_unstable();
        scratch.dedup();
        
        buffer.words.clear();
        for &state in scratch.iter() {
            buffer.insert(state);
        }
        buffer.recompute_hash();
    }
}

impl FromIterator<usize> for CompressedStateSet {
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
