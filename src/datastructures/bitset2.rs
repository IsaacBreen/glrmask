#[derive(Debug, Clone)]
pub struct BitSet {
    data: Vec<u64>,
    dirty_words: Vec<usize>,
    capacity_bits: usize,
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

    pub fn iter(&mut self) -> impl Iterator<Item = usize> + '_ {
        self.dirty_words.sort_unstable();
        self.dirty_words.iter().flat_map(|&w| {
            let word = self.data[w];
            let base = w * 64;
            (0..64).filter(move |&b| (word & (1 << b)) != 0).map(move |b| base + b)
        })
    }
}
