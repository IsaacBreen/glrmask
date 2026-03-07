


#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: `BitSet` is the direct glrmask analogue of sep1's compact bitset type in `datastructures/bitset.rs`.

use serde::{Deserialize, Serialize};




#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BitSet {
    words: Vec<u64>,
    len: usize,
}

impl BitSet {
    
    pub fn new(len: usize) -> Self {
        unimplemented!()
    }

    
    pub fn empty(len: usize) -> Self {
        Self::new(len)
    }

    
    pub fn all(len: usize) -> Self {
        let _ = len;
        unimplemented!()
    }

    
    pub fn len(&self) -> usize {
        unimplemented!()
    }

    
    pub fn is_empty(&self) -> bool {
        unimplemented!()
    }

    
    pub fn get(&self, i: usize) -> bool {
        unimplemented!()
    }

    
    pub fn contains(&self, i: usize) -> bool {
        self.get(i)
    }

    
    pub fn set(&mut self, i: usize) {
        unimplemented!()
    }

    
    pub fn clear(&mut self, i: usize) {
        unimplemented!()
    }

    
    pub fn clear_all(&mut self) {
        unimplemented!()
    }

    
    pub fn count_ones(&self) -> usize {
        unimplemented!()
    }

    
    pub fn is_zero(&self) -> bool {
        unimplemented!()
    }

    
    fn union_with(&mut self, other: &BitSet) {
        unimplemented!()
    }

    
    pub fn union(&self, other: &Self) -> Self {
        let _ = other;
        unimplemented!()
    }

    
    fn intersect_with(&mut self, other: &BitSet) {
        unimplemented!()
    }

    
    pub fn intersection(&self, other: &Self) -> Self {
        let _ = other;
        unimplemented!()
    }

    
    pub fn difference(&self, other: &Self) -> Self {
        let _ = other;
        unimplemented!()
    }

    
    pub fn complement(&self) -> Self {
        unimplemented!()
    }

    
    pub fn is_disjoint(&self, other: &Self) -> bool {
        let _ = other;
        unimplemented!()
    }

    
    pub fn is_subset(&self, other: &Self) -> bool {
        let _ = other;
        unimplemented!()
    }

    
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(word_idx, &word)| {
            let base = word_idx * 64;
            BitIter { word, base }
        })
    }

    
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    
    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
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
    use super::*;

    #[test]
    fn test_basic_ops() {
        let mut bs = BitSet::new(128);
        assert!(!bs.get(0));
        bs.set(0);
        assert!(bs.get(0));
        bs.set(63);
        bs.set(64);
        bs.set(127);
        assert_eq!(bs.count_ones(), 4);

        let ones: Vec<usize> = bs.iter_ones().collect();
        assert_eq!(ones, vec![0, 63, 64, 127]);
    }

    #[test]
    fn test_union_intersect() {
        let mut a = BitSet::new(64);
        let mut b = BitSet::new(64);
        a.set(0);
        a.set(1);
        b.set(1);
        b.set(2);

        let u = a.union(&b);
        assert_eq!(u.count_ones(), 3);

        let i = a.intersection(&b);
        assert_eq!(i.count_ones(), 1);
        assert!(i.get(1));
    }

    #[test]
    fn test_fill_u32_mask() {
        let mut bs = BitSet::new(128);
        bs.set(0);
        bs.set(31);
        bs.set(32);
        bs.set(95);
        let mut buf = vec![0u32; 4];
        bs.fill_u32_mask(&mut buf);
        assert!(buf[0] & (1u32 << 0) != 0);
        assert!(buf[0] & (1u32 << 31) != 0);
        assert!(buf[1] & (1u32 << 0) != 0);
        assert!(buf[2] & (1u32 << 31) != 0);
    }
}
