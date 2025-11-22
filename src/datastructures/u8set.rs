use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Sub, SubAssign};
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use std::fmt::Display;

/// A bitset for u8 values (0-255).
/// Uses two u128s for storage: x for 0-127, y for 128-255.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct U8Set {
    x: u128,
    y: u128,
}

impl JSONConvertible for U8Set {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("x".to_string(), self.x.to_json());
        obj.insert("y".to_string(), self.y.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let x = obj
                    .remove("x")
                    .ok_or_else(|| "Missing field x for U8Set".to_string())
                    .and_then(u128::from_json)?;
                let y = obj
                    .remove("y")
                    .ok_or_else(|| "Missing field y for U8Set".to_string())
                    .and_then(u128::from_json)?;
                Ok(U8Set { x, y })
            }
            _ => Err("Expected JSONNode::Object for U8Set".to_string()),
        }
    }
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
    pub(crate) fn update(&mut self, other: &Self) {
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

    #[inline]
    pub fn from_u8(index: u8) -> Self {
        let mut set = Self::none();
        set.insert(index);
        set
    }

    #[inline]
    pub fn from_slice(slice: &[u8]) -> Self {
        let mut set = Self::none();
        for &i in slice {
            set.insert(i);
        }
        set
    }

    #[inline]
    pub fn from_match_fn<F>(f: F) -> Self
    where
        F: Fn(u8) -> bool,
    {
        let mut set = Self::none();
        for i in 0..=255 {
            if f(i) {
                set.insert(i);
            }
        }
        set
    }

    #[inline]
    pub fn insert(&mut self, index: u8) -> bool {
        let old = self.contains(index);
        self.set_bit(index);
        !old
    }

    #[inline]
    pub fn remove(&mut self, index: u8) -> bool {
        let old = self.contains(index);
        self.clear_bit(index);
        old
    }

    #[inline]
    pub fn contains(&self, index: u8) -> bool {
        self.is_set(index)
    }

    #[inline]
    pub fn complement(&self) -> Self {
        !*self
    }

    #[inline]
    pub fn difference(&self, other: &Self) -> Self {
        *self - *other
    }

    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        *self & *other
    }

    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        *self | *other
    }

    pub fn from_range(start: u8, end: u8) -> Self {
        Self::from_match_fn(move |i| start <= i && i <= end)
    }

    pub fn iter(&self) -> U8SetIter {
        U8SetIter { x: self.x, y: self.y }
    }

    #[inline]
    pub fn without(&self, index: u8) -> Self {
        let mut set = *self;
        set.remove(index);
        set
    }
}

impl Not for U8Set {
    type Output = Self;
    fn not(self) -> Self::Output {
        U8Set { x: !self.x, y: !self.y }
    }
}

impl BitAnd for U8Set {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        U8Set {
            x: self.x & rhs.x,
            y: self.y & rhs.y,
        }
    }
}

impl BitOr for U8Set {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        U8Set {
            x: self.x | rhs.x,
            y: self.y | rhs.y,
        }
    }
}

impl BitXor for U8Set {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        U8Set {
            x: self.x ^ rhs.x,
            y: self.y ^ rhs.y,
        }
    }
}

impl Sub for U8Set {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        U8Set {
            x: self.x & !rhs.x,
            y: self.y & !rhs.y,
        }
    }
}

impl BitAndAssign for U8Set {
    fn bitand_assign(&mut self, rhs: Self) {
        self.x &= rhs.x;
        self.y &= rhs.y;
    }
}

impl BitOrAssign for U8Set {
    fn bitor_assign(&mut self, rhs: Self) {
        self.x |= rhs.x;
        self.y |= rhs.y;
    }
}

impl BitXorAssign for U8Set {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.x ^= rhs.x;
        self.y ^= rhs.y;
    }
}

impl SubAssign for U8Set {
    fn sub_assign(&mut self, rhs: Self) {
        self.x &= !rhs.x;
        self.y &= !rhs.y;
    }
}

pub struct U8SetIter {
    x: u128,
    y: u128,
}

impl Iterator for U8SetIter {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.x != 0 {
            let t = self.x.trailing_zeros();
            self.x &= !(1 << t);
            return Some(t as u8);
        }
        if self.y != 0 {
            let t = self.y.trailing_zeros();
            self.y &= !(1 << t);
            return Some((128 + t) as u8);
        }
        None
    }
}

impl std::fmt::Debug for U8Set {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ranges = Vec::new();
        let mut current_range_start: Option<u8> = None;

        for i in 0..=255 {
            if self.contains(i) {
                if current_range_start.is_none() {
                    current_range_start = Some(i);
                }
            } else if let Some(start) = current_range_start.take() {
                let end = i - 1;
                if start == end {
                    ranges.push(format!("{}", start));
                } else {
                    ranges.push(format!("{}-{}", start, end));
                }
            }
        }

        if let Some(start) = current_range_start.take() {
            let end = 255;
            if start == end {
                ranges.push(format!("{}", start));
            } else {
                ranges.push(format!("{}-{}", start, end));
            }
        }

        write!(f, "U8Set({})", ranges.join(", "))
    }
}

impl Display for U8Set {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}
