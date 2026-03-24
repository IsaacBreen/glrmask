use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct U8Set {
    lo: u128,
    hi: u128,
}

#[inline]
fn split_byte(byte: u8) -> (bool, u32) {
    if byte < 128 {
        (true, byte as u32)
    } else {
        (false, (byte - 128) as u32)
    }
}

#[inline]
fn pop_lowest_bit(word: &mut u128, offset: u8) -> Option<u8> {
    if *word == 0 {
        return None;
    }
    let trailing = word.trailing_zeros() as u8;
    *word &= *word - 1;
    Some(offset + trailing)
}

impl U8Set {
    pub const fn empty() -> Self {
        Self { lo: 0, hi: 0 }
    }

    pub const fn all() -> Self {
        Self {
            lo: u128::MAX,
            hi: u128::MAX,
        }
    }

    pub fn single(b: u8) -> Self {
        Self::from_byte(b)
    }

    pub fn from_byte(b: u8) -> Self {
        let (is_low_word, bit_index) = split_byte(b);
        if is_low_word {
            Self {
                lo: 1u128 << bit_index,
                hi: 0,
            }
        } else {
            Self {
                lo: 0,
                hi: 1u128 << bit_index,
            }
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut set = Self::empty();
        for &b in bytes {
            set.insert(b);
        }
        set
    }

    pub fn from_words(words: [u64; 4]) -> Self {
        Self {
            lo: words[0] as u128 | ((words[1] as u128) << 64),
            hi: words[2] as u128 | ((words[3] as u128) << 64),
        }
    }

    pub fn from_range(lo: u8, hi: u8) -> Self {
        let mut set = Self::empty();
        for b in lo..=hi {
            set.insert(b);
        }
        set
    }

    pub fn from_predicate(f: impl Fn(u8) -> bool) -> Self {
        let mut set = Self::empty();
        for b in 0..=u8::MAX {
            if f(b) {
                set.insert(b);
            }
        }
        set
    }

    pub fn is_empty(&self) -> bool {
        self.lo == 0 && self.hi == 0
    }

    pub fn len(&self) -> usize {
        self.lo.count_ones() as usize + self.hi.count_ones() as usize
    }

    pub fn is_full(&self) -> bool {
        self.lo == u128::MAX && self.hi == u128::MAX
    }

    pub fn contains(&self, b: u8) -> bool {
        let (is_low_word, bit_index) = split_byte(b);
        if is_low_word {
            (self.lo & (1u128 << bit_index)) != 0
        } else {
            (self.hi & (1u128 << bit_index)) != 0
        }
    }

    pub fn insert(&mut self, b: u8) -> bool {
        let old = self.contains(b);
        let (is_low_word, bit_index) = split_byte(b);
        if is_low_word {
            self.lo |= 1u128 << bit_index;
        } else {
            self.hi |= 1u128 << bit_index;
        }
        !old
    }

    pub fn remove(&mut self, b: u8) -> bool {
        let old = self.contains(b);
        let (is_low_word, bit_index) = split_byte(b);
        if is_low_word {
            self.lo &= !(1u128 << bit_index);
        } else {
            self.hi &= !(1u128 << bit_index);
        }
        old
    }

    pub fn union(&self, other: &Self) -> Self {
        Self {
            lo: self.lo | other.lo,
            hi: self.hi | other.hi,
        }
    }

    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            lo: self.lo & other.lo,
            hi: self.hi & other.hi,
        }
    }

    pub fn difference(&self, other: &Self) -> Self {
        Self {
            lo: self.lo & !other.lo,
            hi: self.hi & !other.hi,
        }
    }

    pub fn complement(&self) -> Self {
        Self {
            lo: !self.lo,
            hi: !self.hi,
        }
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        (self.lo & other.lo) == 0 && (self.hi & other.hi) == 0
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        (self.lo & !other.lo) == 0 && (self.hi & !other.hi) == 0
    }

    pub fn iter(&self) -> U8SetIter {
        U8SetIter {
            lo: self.lo,
            hi: self.hi,
        }
    }
}

impl Default for U8Set {
    fn default() -> Self {
        Self::empty()
    }
}

impl BitOr for U8Set {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(&rhs)
    }
}

impl BitOrAssign for U8Set {
    fn bitor_assign(&mut self, rhs: Self) {
        self.lo |= rhs.lo;
        self.hi |= rhs.hi;
    }
}

impl BitAnd for U8Set {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        self.intersection(&rhs)
    }
}

impl BitAndAssign for U8Set {
    fn bitand_assign(&mut self, rhs: Self) {
        self.lo &= rhs.lo;
        self.hi &= rhs.hi;
    }
}

impl Not for U8Set {
    type Output = Self;
    fn not(self) -> Self {
        self.complement()
    }
}

impl fmt::Debug for U8Set {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes: Vec<u8> = self.iter().collect();
        if bytes.len() <= 16 {
            write!(f, "U8Set({:?})", bytes)
        } else {
            write!(f, "U8Set({} bytes)", bytes.len())
        }
    }
}

impl fmt::Display for U8Set {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (i, b) in self.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "0x{b:02X}")?;
        }
        write!(f, "}}")
    }
}

pub struct U8SetIter {
    lo: u128,
    hi: u128,
}

impl Iterator for U8SetIter {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        pop_lowest_bit(&mut self.lo, 0).or_else(|| pop_lowest_bit(&mut self.hi, 128))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_full() {
        let empty = U8Set::empty();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let full = U8Set::all();
        assert!(full.is_full());
        assert_eq!(full.len(), 256);
    }

    #[test]
    fn test_insert_contains() {
        let mut s = U8Set::empty();
        assert!(!s.contains(42));
        assert!(s.insert(42));
        assert!(s.contains(42));
        assert!(!s.insert(42));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn test_byte_range() {
        let s = U8Set::from_range(b'a', b'z');
        assert_eq!(s.len(), 26);
        assert!(s.contains(b'a'));
        assert!(s.contains(b'z'));
        assert!(!s.contains(b'A'));
    }

    #[test]
    fn test_boundary_bytes() {
        let mut s = U8Set::empty();
        s.insert(0);
        s.insert(127);
        s.insert(128);
        s.insert(255);
        assert_eq!(s.len(), 4);
        let vals: Vec<u8> = s.iter().collect();
        assert_eq!(vals, vec![0, 127, 128, 255]);
    }

    #[test]
    fn test_set_ops() {
        let a = U8Set::from_range(0, 10);
        let b = U8Set::from_range(5, 15);

        let u = a.union(&b);
        assert_eq!(u.len(), 16);

        let i = a.intersection(&b);
        assert_eq!(i.len(), 6);

        let d = a.difference(&b);
        assert_eq!(d.len(), 5);

        let c = a.complement();
        assert_eq!(c.len(), 256 - 11);
    }

    #[test]
    fn test_from_predicate() {
        let digits = U8Set::from_predicate(|b| b.is_ascii_digit());
        assert_eq!(digits.len(), 10);
        assert!(digits.contains(b'0'));
        assert!(digits.contains(b'9'));
        assert!(!digits.contains(b'a'));
    }

    #[test]
    fn test_iter_order() {
        let s = U8Set::from_bytes(&[200, 100, 50, 150]);
        let vals: Vec<u8> = s.iter().collect();
        assert_eq!(vals, vec![50, 100, 150, 200]);
    }
}
