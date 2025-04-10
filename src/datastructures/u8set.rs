use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

#[derive(Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct U8Set {
    pub(crate) x: u128,
    pub(crate) y: u128,
}

impl Default for U8Set {
    fn default() -> Self {
        Self::none()
    }
}

impl U8Set {
    #[inline]
    fn is_set(&self, index: u8) -> bool {
        if index < 128 {
            self.x & (1 << index) != 0
        } else {
            self.y & (1 << (index - 128)) != 0
        }
    }

    #[inline]
    fn set_bit(&mut self, index: u8) {
        if index < 128 {
            self.x |= 1 << index;
        } else {
            self.y |= 1 << (index - 128);
        }
    }

    #[inline]
    fn clear_bit(&mut self, index: u8) {
        if index < 128 {
            self.x &= !(1 << index);
        } else {
            self.y &= !(1 << (index - 128));
        }
    }

    #[inline]
    fn update(&mut self, other: &Self) {
        self.x |= other.x;
        self.y |= other.y;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.x.count_ones() as usize + self.y.count_ones() as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.x == 0 && self.y == 0
    }

    #[inline]
    pub fn clear(&mut self) {
        self.x = 0;
        self.y = 0;
    }

    #[inline]
    pub fn all() -> Self {
        U8Set { x: u128::MAX, y: u128::MAX }
    }

    #[inline]
    pub fn none() -> Self {
        U8Set { x: 0, y: 0 }
    }

    #[inline]
    pub fn new() -> Self {
        Self::none()
    }


    pub fn from_u8(p0: u8) -> U8Set {
        let mut result = U8Set::none();
        result.insert(p0);
        result
    }


    pub fn from_u8_range(start: u8, end: u8) -> U8Set {
        Self::from_match_fn(move |i| start <= i && i <= end)
    }

    pub fn from_char(p0: char) -> U8Set {
        Self::from_chars(&p0.to_string())
    }

    pub fn from_char_negation(p0: char) -> U8Set {
        let mut result = U8Set::none();
        result.insert(p0 as u8);
        result.complement()
    }

    pub fn from_byte_range(range: impl IntoIterator<Item = u8>) -> U8Set {
        let mut result = U8Set::none();
        for c in range {
            assert!(c <= 255, "Character {} is not a valid u8 value", c);
            result.insert(c);
        }
        result
    }

    pub fn from_char_negation_range(range: impl IntoIterator<Item = u8>) -> U8Set {
        Self::from_byte_range(range).complement()
    }

    #[inline]
    pub fn insert(&mut self, value: u8) -> bool {
        if self.contains(value) {
            false
        } else {
            self.set_bit(value);
            true
        }
    }


    #[inline]
    pub fn remove(&mut self, value: u8) -> bool {
        if !self.contains(value) {
            false
        } else {
            self.clear_bit(value);
            true
        }
    }


    #[inline]
    pub fn contains(&self, value: impl Into<u8>) -> bool {
        self.is_set(value.into())
    }


    pub fn from_byte(byte: u8) -> Self {
        let mut result = Self::none();
        result.insert(byte);
        result
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut result = Self::none();
        for byte in bytes {
            result.insert(*byte);
        }
        result
    }

    pub fn from_chars(chars: &str) -> Self {
        let mut result = Self::none();
        for c in chars.chars() {
            assert!(c as usize <= 255, "Character {} is not a valid u8 value", c);
            result.insert(c as u8);
        }
        result
    }


    pub fn from_chars_negation(chars: &str) -> Self {
        Self::from_chars(chars).complement()
    }


    pub fn from_str(s: &str) -> Self {
        Self::from_chars(s)
    }

    pub fn from_match_fn<F>(f: F) -> Self
    where
        F: Fn(u8) -> bool,
    {
        let mut result = Self::none();
        for i in 0..=255 {
            if f(i) {
                result.insert(i);
            }
        }
        result
    }

    pub fn from_range(start: u8, end: u8) -> Self {
        Self::from_match_fn(move |i| start <= i && i <= end)
    }

    pub fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        (0..=255).filter(move |&i| self.contains(i))
    }

    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        U8Set {
            x: self.x | other.x,
            y: self.y | other.y,
        }
    }

    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        U8Set {
            x: self.x & other.x,
            y: self.y & other.y,
        }
    }


    #[inline]
    pub fn complement(&self) -> Self {
        U8Set {
            x: !self.x,
            y: !self.y,
        }
    }
}


impl BitOr for &U8Set {
    type Output = U8Set;

    fn bitor(self, other: &U8Set) -> U8Set {
        self.union(other)
    }
}

impl BitAnd for &U8Set {
    type Output = U8Set;

    fn bitand(self, other: &U8Set) -> U8Set {
        self.intersection(other)
    }
}

impl BitOr for U8Set {
    type Output = U8Set;

    fn bitor(self, other: U8Set) -> U8Set {
        &self | &other
    }
}

impl BitAnd for U8Set {
    type Output = U8Set;

    fn bitand(self, other: U8Set) -> U8Set {
        &self & &other
    }
}

impl BitOrAssign for U8Set {
    fn bitor_assign(&mut self, other: U8Set) {
        self.update(&other);
    }
}

impl BitAndAssign for U8Set {
    fn bitand_assign(&mut self, other: U8Set) {
        self.x &= other.x;
        self.y &= other.y;
    }
}

impl std::fmt::Debug for U8Set {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "U8Set({})", self)
    }
}

impl std::fmt::Display for U8Set {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ranges = Vec::new();
        let mut start = None;
        let mut prev = None;

        for i in self.iter() {
            match (start, prev) {
                (None, None) => {
                    start = Some(i);
                }
                (Some(_), Some(p)) if i == p + 1 => {}
                (Some(s), Some(p)) => {
                    ranges.push((s, p));
                    start = Some(i);
                }
                _ => unreachable!(),
            }
            prev = Some(i);
        }

        if let Some(s) = start {
            ranges.push((s, prev.unwrap()));
        }

        let mut output = String::new();
        for (i, (start, end)) in ranges.iter().enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            if start == end {
                output.push_str(&format!("{:?}", *start as char));
            } else if end - start == 1 {
                output.push_str(&format!("{:?}, {:?}", *start as char, *end as char));
            } else {
                output.push_str(&format!("{:?}..{:?}", *start as char, *end as char));
            }
        }

        write!(f, "[{}]", output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u8set() {
        let mut set = U8Set::none();
        assert!(set.insert(b'a'));
        assert!(set.insert(b'b'));
        assert!(!set.insert(b'a'));
        assert!(set.contains(b'a'));
        assert!(set.contains(b'b'));
        assert!(!set.contains(b'c'));
        assert_eq!(set.len(), 2);
        assert!(set.remove(b'a'));
        assert!(!set.remove(b'c'));
        assert_eq!(set.len(), 1);
        assert!(!set.is_empty());
        set.clear();
        assert!(set.is_empty());

        let set1 = U8Set::from_chars("abc");
        let set2 = U8Set::from_chars("bcd");
        let union = &set1 | &set2;
        let intersection = &set1 & &set2;
        assert_eq!(union.len(), 4);
        assert_eq!(intersection.len(), 2);

        let even_set = U8Set::from_match_fn(|x| x % 2 == 0);
        assert!(even_set.contains(0));
        assert!(even_set.contains(2));
        assert!(!even_set.contains(1));
        assert_eq!(even_set.len(), 128);
    }
}