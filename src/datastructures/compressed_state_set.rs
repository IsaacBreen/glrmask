use std::hash::{Hash, Hasher};
use ahash::AHasher;
use profiler_macro::time_it;

#[derive(Clone, Eq, Debug)]
pub struct CompressedStateSet {
    // Sorted by word index. (word_index, word_value)
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