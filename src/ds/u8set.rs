//! 256-bit set for byte values.
//!
//! `U8Set` represents a set of byte values (0..=255) using two `u128` words.
//! This is the fundamental building block for byte-level automata transitions.
#![allow(dead_code)]

use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};

use serde::{Deserialize, Serialize};

/// A set of byte values stored as 256 bits (two `u128`s).
///
/// - `lo` covers bytes 0..128
/// - `hi` covers bytes 128..256
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct U8Set {
    lo: u128,
    hi: u128,
}

impl U8Set {
    /// Empty set.
    pub const fn empty() -> Self {
        Self { lo: 0, hi: 0 }
    }

    /// Full set (all 256 bytes).
    pub const fn full() -> Self {
        Self {
            lo: u128::MAX,
            hi: u128::MAX,
        }
    }

    /// Singleton set containing just one byte.
    pub fn from_byte(b: u8) -> Self {
        let mut s = Self::empty();
        s.insert(b);
        s
    }

    /// Set from a byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut s = Self::empty();
        for &b in bytes {
            s.insert(b);
        }
        s
    }

    /// Set from an inclusive byte range `[lo, hi]`.
    pub fn from_range(lo: u8, hi: u8) -> Self {
        let mut s = Self::empty();
        for b in lo..=hi {
            s.insert(b);
        }
        s
    }

    /// Set from a predicate function.
    pub fn from_predicate(f: impl Fn(u8) -> bool) -> Self {
        let mut s = Self::empty();
        for b in 0..=255u8 {
            if f(b) {
                s.insert(b);
            }
        }
        s
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.lo == 0 && self.hi == 0
    }

    /// Number of bytes in the set.
    pub fn len(&self) -> usize {
        (self.lo.count_ones() + self.hi.count_ones()) as usize
    }

    /// Whether the set contains all 256 bytes.
    pub fn is_full(&self) -> bool {
        self.lo == u128::MAX && self.hi == u128::MAX
    }

    /// Check if a byte is in the set.
    pub fn contains(&self, b: u8) -> bool {
        if b < 128 {
            (self.lo >> b) & 1 != 0
        } else {
            (self.hi >> (b - 128)) & 1 != 0
        }
    }

    /// Insert a byte into the set. Returns true if the byte was not already present.
    pub fn insert(&mut self, b: u8) -> bool {
        let was = self.contains(b);
        if b < 128 {
            self.lo |= 1u128 << b;
        } else {
            self.hi |= 1u128 << (b - 128);
        }
        !was
    }

    /// Remove a byte from the set. Returns true if the byte was present.
    pub fn remove(&mut self, b: u8) -> bool {
        let was = self.contains(b);
        if b < 128 {
            self.lo &= !(1u128 << b);
        } else {
            self.hi &= !(1u128 << (b - 128));
        }
        was
    }

    /// Union of two sets.
    pub fn union(&self, other: &Self) -> Self {
        Self {
            lo: self.lo | other.lo,
            hi: self.hi | other.hi,
        }
    }

    /// Intersection of two sets.
    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            lo: self.lo & other.lo,
            hi: self.hi & other.hi,
        }
    }

    /// Set difference: `self \ other`.
    pub fn difference(&self, other: &Self) -> Self {
        Self {
            lo: self.lo & !other.lo,
            hi: self.hi & !other.hi,
        }
    }

    /// Complement: all bytes NOT in this set.
    pub fn complement(&self) -> Self {
        Self {
            lo: !self.lo,
            hi: !self.hi,
        }
    }

    /// Iterator over all bytes in the set, in ascending order.
    pub fn iter(&self) -> U8SetIter {
        U8SetIter {
            lo: self.lo,
            hi: self.hi,
            phase: 0,
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
        *self = self.union(&rhs);
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
        *self = self.intersection(&rhs);
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
        // Display as byte ranges like [a-z0-9]
        let mut ranges: Vec<(u8, u8)> = Vec::new();
        for b in self.iter() {
            if let Some(last) = ranges.last_mut()
                && b == last.1 + 1
            {
                last.1 = b;
                continue;
            }
            ranges.push((b, b));
        }
        write!(f, "[")?;
        for (lo, hi) in &ranges {
            if lo == hi {
                if lo.is_ascii_graphic() {
                    write!(f, "{}", *lo as char)?;
                } else {
                    write!(f, "\\x{:02x}", lo)?;
                }
            } else {
                let lo_s = if lo.is_ascii_graphic() {
                    format!("{}", *lo as char)
                } else {
                    format!("\\x{:02x}", lo)
                };
                let hi_s = if hi.is_ascii_graphic() {
                    format!("{}", *hi as char)
                } else {
                    format!("\\x{:02x}", hi)
                };
                write!(f, "{}-{}", lo_s, hi_s)?;
            }
        }
        write!(f, "]")
    }
}

/// Iterator over bytes in a `U8Set`.
pub struct U8SetIter {
    lo: u128,
    hi: u128,
    phase: u8, // 0 = processing lo, 1 = processing hi, 2 = done
}

impl Iterator for U8SetIter {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        loop {
            match self.phase {
                0 => {
                    if self.lo == 0 {
                        self.phase = 1;
                        continue;
                    }
                    let tz = self.lo.trailing_zeros() as u8;
                    self.lo &= self.lo - 1;
                    return Some(tz);
                }
                1 => {
                    if self.hi == 0 {
                        self.phase = 2;
                        return None;
                    }
                    let tz = self.hi.trailing_zeros() as u8;
                    self.hi &= self.hi - 1;
                    return Some(128 + tz);
                }
                _ => return None,
            }
        }
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

        let full = U8Set::full();
        assert!(full.is_full());
        assert_eq!(full.len(), 256);
    }

    #[test]
    fn test_insert_contains() {
        let mut s = U8Set::empty();
        assert!(!s.contains(42));
        assert!(s.insert(42));
        assert!(s.contains(42));
        assert!(!s.insert(42)); // already present
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
        assert_eq!(u.len(), 16); // 0..=15

        let i = a.intersection(&b);
        assert_eq!(i.len(), 6); // 5..=10

        let d = a.difference(&b);
        assert_eq!(d.len(), 5); // 0..=4

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
