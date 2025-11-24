use std::hash::{Hash, Hasher};
use crate::datastructures::state_set::StateSet;

#[derive(Debug, Clone, Default)]
pub struct BitSet {
    data: Vec<u64>,
    dirty_words: Vec<usize>,
    capacity_bits: usize,
}

impl PartialEq for BitSet {
    fn eq(&self, other: &Self) -> bool {
        // Two BitSets are equal if their data vectors are equal.
        // We assume unused capacity in 'data' is always zeroed via clear/insert logic.
        self.data == other.data
    }
}

impl Eq for BitSet {}

impl Hash for BitSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl BitSet {
    pub fn new(capacity_bits: usize) -> Self {
        let words = (capacity_bits + 63) / 64;
        Self {
            data: vec![0; words],
            dirty_words: Vec::with_capacity(words),
            capacity_bits,
        }
    }

    pub fn len(&self) -> usize {
        self.data.iter().map(|&x| x.count_ones() as usize).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn insert(&mut self, bit: usize) -> bool {
        if bit >= self.capacity_bits {
            return false;
        }
        let word_idx = bit / 64;
        let mask = 1 << (bit % 64);
        if self.data[word_idx] & mask == 0 {
            if self.data[word_idx] == 0 {
                self.dirty_words.push(word_idx);
            }
            self.data[word_idx] |= mask;
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn contains(&self, bit: usize) -> bool {
        if bit >= self.capacity_bits {
            return false;
        }
        (self.data[bit / 64] & (1 << (bit % 64))) != 0
    }

    pub fn clear(&mut self) {
        for &w in &self.dirty_words {
            self.data[w] = 0;
        }
        self.dirty_words.clear();
    }

    pub fn iter(&self) -> BitSetIter {
        BitSetIter {
            bitset: self,
            dirty_idx: 0,
            current_word_idx: 0,
            current_word: 0,
        }
    }
}

impl FromIterator<usize> for BitSet {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        // Note: This default impl creates a small BitSet and might panic if indices exceed capacity.
        // For proper usage, construct with `with_capacity` first.
        // Here we assume a default large enough for typical small tests or resizes (if we added resizing).
        // Since our BitSet doesn't resize, this is risky if not used carefully.
        // However, standard `collect()` usage is rare in the hot path compared to `insert`.
        let mut set = BitSet::new(1024); 
        for i in iter {
            if i >= set.capacity_bits {
                // In a real impl, we'd resize here. For now, we just panic or ignore?
                // Let's create a new larger one? No, cannot replace self easily.
                // We'll panic for now as this is a specific optimization struct.
                panic!("BitSet::from_iter index out of bounds (cap 1024)");
            }
            set.insert(i);
        }
        set
    }
}

impl StateSet for BitSet {
    type Iter<'a> = BitSetIter<'a>;

    fn with_capacity(capacity: usize) -> Self { Self::new(capacity) }
    fn insert(&mut self, state: usize) -> bool { self.insert(state) }
    fn contains(&self, state: usize) -> bool { self.contains(state) }
    fn len(&self) -> usize { self.len() }
    fn is_empty(&self) -> bool { self.is_empty() }
    fn clear(&mut self) { self.clear() }
    fn iter<'a>(&'a self) -> Self::Iter<'a> { self.iter() }
}

pub struct BitSetIter<'a> {
    bitset: &'a BitSet,
    dirty_idx: usize,
    current_word_idx: usize,
    current_word: u64,
}

impl<'a> Iterator for BitSetIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_word != 0 {
                let trailing = self.current_word.trailing_zeros();
                self.current_word &= !(1u64 << trailing);
                return Some(self.current_word_idx * 64 + trailing as usize);
            }
            
            if self.dirty_idx >= self.bitset.dirty_words.len() {
                return None;
            }
            
            let w_idx = self.bitset.dirty_words[self.dirty_idx];
            self.dirty_idx += 1;
            
            let word = self.bitset.data[w_idx];
            if word != 0 {
                self.current_word = word;
                self.current_word_idx = w_idx;
            }
        }
    }
}
