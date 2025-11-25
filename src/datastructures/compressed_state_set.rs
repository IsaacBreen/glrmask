use std::hash::{Hash, Hasher};
use rustc_hash::FxHasher;
use crate::datastructures::state_set::StateSet;
use crate::json_serialization::{JSONConvertible, JSONNode};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DenseStateSet {
    pub words: Vec<u64>,
}

impl DenseStateSet {
    // #[time_it]
    pub fn new(num_bits: usize) -> Self {
        let num_words = (num_bits + 63) / 64;
        Self {
            words: vec![0; num_words],
        }
    }

    pub fn new_from_slice(num_bits: usize, slice: &[usize]) -> Self {
        let num_words = (num_bits + 63) / 64;
        let mut words = vec![0; num_words];
        for &bit in slice {
            words[bit / 64] |= 1u64 << (bit % 64);
        }
        Self { words }
    }

    pub fn empty() -> Self {
        Self { words: Vec::new() }
    }

    #[inline]
    pub fn insert(&mut self, bit: usize) {
        let word_idx = bit / 64;
        if word_idx >= self.words.len() {
            self.words.resize(word_idx + 1, 0);
        }
        unsafe {
            *self.words.get_unchecked_mut(word_idx) |= 1u64 << (bit % 64);
        }
    }

    #[inline]
    pub fn contains(&self, bit: usize) -> bool {
        let word_idx = bit / 64;
        if word_idx >= self.words.len() {
            return false;
        }
        unsafe {
            (*self.words.get_unchecked(word_idx) & (1u64 << (bit % 64))) != 0
        }
    }

    #[inline]
    pub fn union_with(&mut self, other: &DenseStateSet) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            self.union_with_avx2(other);
        }
        
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            self.union_with_neon(other);
        }
        
        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "avx2"),
            all(target_arch = "aarch64", target_feature = "neon")
        )))]
        {
            // Scalar fallback - process 8 words at a time for better pipelining
            let other_words = &other.words;
            let len = other_words.len();
            let chunks = len / 8;
            let remainder = len % 8;
            
            for i in 0..chunks {
                let base = i * 8;
                unsafe {
                    *self.words.get_unchecked_mut(base) |= *other_words.get_unchecked(base);
                    *self.words.get_unchecked_mut(base + 1) |= *other_words.get_unchecked(base + 1);
                    *self.words.get_unchecked_mut(base + 2) |= *other_words.get_unchecked(base + 2);
                    *self.words.get_unchecked_mut(base + 3) |= *other_words.get_unchecked(base + 3);
                    *self.words.get_unchecked_mut(base + 4) |= *other_words.get_unchecked(base + 4);
                    *self.words.get_unchecked_mut(base + 5) |= *other_words.get_unchecked(base + 5);
                    *self.words.get_unchecked_mut(base + 6) |= *other_words.get_unchecked(base + 6);
                    *self.words.get_unchecked_mut(base + 7) |= *other_words.get_unchecked(base + 7);
                }
            }
            
            // Handle remaining words
            for i in (chunks * 8)..len {
                unsafe {
                    *self.words.get_unchecked_mut(i) |= *other_words.get_unchecked(i);
                }
            }
        }
    }
    
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline]
    unsafe fn union_with_avx2(&mut self, other: &DenseStateSet) {
        use std::arch::x86_64::*;
        
        let other_words = &other.words;
        let len = other_words.len();
        let simd_len = (len / 4) * 4; // Process 4 u64s (256 bits) at a time
        
        for i in (0..simd_len).step_by(4) {
            let a = _mm256_loadu_si256(self.words.as_ptr().add(i) as *const __m256i);
            let b = _mm256_loadu_si256(other_words.as_ptr().add(i) as *const __m256i);
            let result = _mm256_or_si256(a, b);
            _mm256_storeu_si256(self.words.as_mut_ptr().add(i) as *mut __m256i, result);
        }
        
        // Handle remaining words
        for i in simd_len..len {
            *self.words.get_unchecked_mut(i) |= *other_words.get_unchecked(i);
        }
    }
    
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[inline]
    unsafe fn union_with_neon(&mut self, other: &DenseStateSet) {
        use std::arch::aarch64::*;
        
        let other_words = &other.words;
        let len = other_words.len();
        let simd_len = (len / 2) * 2; // Process 2 u64s (128 bits) at a time
        
        for i in (0..simd_len).step_by(2) {
            let a = vld1q_u64(self.words.as_ptr().add(i));
            let b = vld1q_u64(other_words.as_ptr().add(i));
            let result = vorrq_u64(a, b);
            vst1q_u64(self.words.as_mut_ptr().add(i), result);
        }
        
        // Handle remaining words
        for i in simd_len..len {
            *self.words.get_unchecked_mut(i) |= *other_words.get_unchecked(i);
        }
    }

    pub fn iter(&self) -> DenseStateSetIter {
        DenseStateSetIter {
            set: self,
            word_idx: 0,
            current_word: if self.words.is_empty() { 0 } else { self.words[0] },
        }
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    pub fn len(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }
}

pub struct DenseStateSetIter<'a> {
    set: &'a DenseStateSet,
    word_idx: usize,
    current_word: u64,
}

impl<'a> Iterator for DenseStateSetIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_word != 0 {
                let t = self.current_word.trailing_zeros();
                self.current_word &= !(1u64 << t);
                return Some(self.word_idx * 64 + t as usize);
            }
            self.word_idx += 1;
            if self.word_idx >= self.set.words.len() {
                return None;
            }
            self.current_word = self.set.words[self.word_idx];
        }
    }
}

impl<'a> IntoIterator for &'a DenseStateSet {
    type Item = usize;
    type IntoIter = DenseStateSetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl FromIterator<usize> for DenseStateSet {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        let mut set = DenseStateSet::empty();
        for i in iter {
            set.insert(i);
        }
        set
    }
}

impl JSONConvertible for DenseStateSet {
    fn to_json(&self) -> JSONNode {
        let items: Vec<usize> = self.iter().collect();
        items.to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let items = Vec::<usize>::from_json(node)?;
        let mut set = DenseStateSet::empty();
        for item in items {
            set.insert(item);
        }
        Ok(set)
    }
}

pub struct SparseStateSet {
    pub dense: DenseStateSet,
    pub dirty_words: Vec<usize>,
}

impl SparseStateSet {
    // #[time_it]
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
    // #[inline]
    pub fn new() -> Self {
        Self {
            words: Vec::new(),
            hash: 0,
        }
    }

    #[inline]
    // #[time_it]
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
    // #[time_it]
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