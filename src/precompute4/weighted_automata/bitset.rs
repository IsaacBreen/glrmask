// src/precompute4/weighted_automata/bitset.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use range_set_blaze::RangeSetBlaze;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, Not, Sub, SubAssign,
};

// SimpleBitset: a RangeSetBlaze with cached fingerprint and fast "is_all" flag.
// - fp is a content fingerprint used only for hashing; Eq remains content-based via rsb.
// - is_all tracks whether it is the universe (0..=usize::MAX), enabling fast short-circuits.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Default)]
pub struct SimpleBitset {
    pub(crate) rsb: RangeSetBlaze<usize>,
    pub(crate) fp: u64,
    is_all: bool,
}

// Fingerprint utilities
pub(crate) const FP_ZERO: u64 = 0x9E37_79B9_7F4A_7C15;
const FP_ALL: u64 = 0xD6E8_FEB8_6659_FD93;
pub(crate) const FP_K1: u64 = 0xC2B2_AE3D_27D4_EB4F;
pub(crate) const FP_K2: u64 = 0x1656_67B1_F3E1_5A6D;
const FP_K3: u64 = 0x9E37_79B1_1F37_9B97;

#[inline]
pub(crate) fn mix3(a: u64, b: u64, c: u64) -> u64 {
    let mut x = a ^ b.wrapping_mul(FP_K1);
    x = x.rotate_left(27) ^ c.wrapping_mul(FP_K2);
    x ^= x >> 33;
    x = x.wrapping_mul(FP_K3);
    x ^ (x >> 29)
}

#[inline]
fn calc_is_all_and_fp(rsb: &RangeSetBlaze<usize>) -> (bool, u64) {
    let mut it = rsb.ranges();
    if let Some(first) = it.next() {
        if *first.start() == 0 && *first.end() == usize::MAX && it.next().is_none() {
            return (true, FP_ALL);
        }
        let mut fp = mix3(
            FP_ZERO,
            *first.start() as u64,
            (*first.end() as u64).wrapping_mul(FP_K1),
        );
        for r in it {
            let s = *r.start() as u64;
            let e = *r.end() as u64;
            fp = mix3(fp, s.wrapping_mul(FP_K2), e.wrapping_mul(FP_K3));
        }
        (false, fp)
    } else {
        (false, FP_ZERO)
    }
}

#[inline]
fn universe_rsb() -> RangeSetBlaze<usize> {
    RangeSetBlaze::from_iter([0usize..=usize::MAX])
}

impl SimpleBitset {
    #[inline]
    fn from_rsb_inner(rsb: RangeSetBlaze<usize>) -> Self {
        let (is_all, fp) = calc_is_all_and_fp(&rsb);
        SimpleBitset { rsb, fp, is_all }
    }

    #[inline]
    fn update_cached(&mut self) {
        let (is_all, fp) = calc_is_all_and_fp(&self.rsb);
        self.is_all = is_all;
        self.fp = fp;
    }

    pub fn zeros() -> Self {
        SimpleBitset { rsb: RangeSetBlaze::new(), fp: FP_ZERO, is_all: false }
    }
    pub fn all() -> Self {
        SimpleBitset { rsb: universe_rsb(), fp: FP_ALL, is_all: true }
    }
    pub fn from_item(item: usize) -> Self {
        SimpleBitset::from_rsb_inner(RangeSetBlaze::from_iter([item]))
    }
    pub fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self {
        SimpleBitset::from_rsb_inner(rsb)
    }
    pub fn len(&self) -> usize {
        self.rsb.len().try_into().unwrap_or(usize::MAX)
    }
    pub fn is_empty(&self) -> bool {
        self.rsb.is_empty()
    }
    pub fn is_disjoint(&self, other: &SimpleBitset) -> bool {
        if self.is_empty() || other.is_empty() {
            return true;
        }
        if self.is_all_fast() || other.is_all_fast() {
            return false;
        }
        (&self.rsb & &other.rsb).is_empty()
    }
    pub fn contains(&self, index: usize) -> bool {
        self.rsb.contains(index)
    }
    pub fn max_item(&self) -> Option<usize> {
        self.rsb.iter().last()
    }
    /// Iterate over items, truncated by `max` to prevent accidental ALL iteration.
    pub fn iter_up_to(&self, max: usize) -> impl Iterator<Item = usize> {
        (&self.rsb & &RangeSetBlaze::from_iter([0..=max])).into_iter()
    }
    #[inline]
    fn with_new_rsb_and_op(_lhs: &SimpleBitset, _rhs: &SimpleBitset, rsb: RangeSetBlaze<usize>, _op_tag: u64) -> SimpleBitset {
        let (is_all, fp) = calc_is_all_and_fp(&rsb);
        SimpleBitset { rsb, fp, is_all }
    }
    #[inline]
    fn with_new_rsb_unary(_src: &SimpleBitset, rsb: RangeSetBlaze<usize>, _op_tag: u64) -> SimpleBitset {
        let (is_all, fp) = calc_is_all_and_fp(&rsb);
        SimpleBitset { rsb, fp, is_all }
    }
    #[inline]
    pub fn is_all_fast(&self) -> bool { self.is_all }

    #[inline]
    pub fn is_subset_of(&self, rhs: &SimpleBitset) -> bool {
        (self & rhs) == *self
    }

    pub fn complement(&self) -> SimpleBitset {
        if self.is_empty() { return SimpleBitset::all(); }
        if self.is_all_fast() { return SimpleBitset::zeros(); }
        let rsb = &universe_rsb() - &self.rsb;
        SimpleBitset::with_new_rsb_unary(self, rsb, 0xD3)
    }

    pub fn insert(&mut self, item: usize) {
        if self.is_all_fast() { return; }
        self.rsb.insert(item);
        self.update_cached();
    }

    pub fn add(&mut self, item: usize) {
        self.insert(item);
    }

    pub fn remove(&mut self, item: usize) {
        if self.is_empty() { return; }
        if self.is_all_fast() {
            self.rsb = universe_rsb();
        }
        self.rsb.remove(item);
        self.update_cached();
    }

    pub fn set(&mut self, item: usize, value: bool) {
        if value {
            self.insert(item);
        } else {
            self.remove(item);
        }
    }

    pub fn clear(&mut self) {
        self.rsb.clear();
        self.fp = FP_ZERO;
        self.is_all = false;
    }

    pub fn clip_to_range(&mut self, min: usize, max: usize) {
        if self.is_empty() { return; }
        if self.is_all_fast() {
            self.rsb = universe_rsb();
        }
        let clip_rsb = RangeSetBlaze::from_iter([min..=max]);
        self.rsb = &self.rsb & &clip_rsb;
        self.update_cached();
    }

    pub fn clip_min(&mut self, min: usize) {
        self.clip_to_range(min, usize::MAX);
    }

    pub fn clip_max(&mut self, max: usize) {
        self.clip_to_range(0, max);
    }
}

impl Hash for SimpleBitset {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.fp);
    }
}

impl Debug for SimpleBitset {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_all_fast() {
            write!(f, "SimpleBitset(ALL)")
        } else {
            Debug::fmt(&self.rsb, f)
        }
    }
}

impl Display for SimpleBitset {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_all_fast() {
            return write!(f, "ALL");
        }
        write!(f, "[")?;
        let mut ranges = self.rsb.ranges().peekable();
        while let Some(range) = ranges.next() {
            if range.start() == range.end() {
                write!(f, "{}", range.start())?;
            } else {
                write!(f, "{}..={}", range.start(), range.end())?;
            }
            if ranges.peek().is_some() {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

impl FromIterator<usize> for SimpleBitset {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        SimpleBitset::from_rsb_inner(RangeSetBlaze::from_iter(iter))
    }
}

impl FromIterator<std::ops::RangeInclusive<usize>> for SimpleBitset {
    fn from_iter<T: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(iter: T) -> Self {
        SimpleBitset::from_rsb_inner(RangeSetBlaze::from_iter(iter))
    }
}

// Borrowed bit-ops
impl<'a> BitAnd<&'a SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn bitand(self, rhs: &'a SimpleBitset) -> Self::Output {
        if self.is_empty() || rhs.is_empty() {
            return SimpleBitset::zeros();
        }
        if self.is_all_fast() { return rhs.clone(); }
        if rhs.is_all_fast() { return self.clone(); }
        let rsb = &self.rsb & &rhs.rsb;
        SimpleBitset::with_new_rsb_and_op(self, rhs, rsb, 0xA1)
    }
}
impl<'a> BitOr<&'a SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn bitor(self, rhs: &'a SimpleBitset) -> Self::Output {
        if self.is_all_fast() || rhs.is_all_fast() {
            return SimpleBitset::all();
        }
        if self.is_empty() { return rhs.clone(); }
        if rhs.is_empty() { return self.clone(); }
        let rsb = &self.rsb | &rhs.rsb;
        SimpleBitset::with_new_rsb_and_op(self, rhs, rsb, 0xB1)
    }
}

// Assign ops (borrowed RHS)
impl BitAndAssign<&SimpleBitset> for SimpleBitset {
    fn bitand_assign(&mut self, rhs: &SimpleBitset) {
        if self.is_empty() { return; }
        if rhs.is_empty() {
            *self = SimpleBitset::zeros();
            return;
        }
        if rhs.is_all_fast() { return; }
        if self.is_all_fast() {
            *self = rhs.clone();
            return;
        }
        self.rsb = &self.rsb & &rhs.rsb;
        self.update_cached();
    }
}
impl BitOrAssign<&SimpleBitset> for SimpleBitset {
    fn bitor_assign(&mut self, rhs: &SimpleBitset) {
        if self.is_all_fast() || rhs.is_all_fast() {
            *self = SimpleBitset::all();
            return;
        }
        if rhs.is_empty() { return; }
        if self.is_empty() {
            *self = rhs.clone();
            return;
        }
        self.rsb |= &rhs.rsb;
        self.update_cached();
    }
}

// Owned fallbacks via borrowed ops
impl BitAnd<SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn bitand(self, rhs: SimpleBitset) -> Self::Output {
        (&self) & (&rhs)
    }
}
impl BitOr<SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn bitor(self, rhs: SimpleBitset) -> Self::Output {
        (&self) | (&rhs)
    }
}
impl<'a> BitAnd<&'a SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn bitand(self, rhs: &'a SimpleBitset) -> Self::Output {
        (&self) & rhs
    }
}
impl<'a> BitOr<&'a SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn bitor(self, rhs: &'a SimpleBitset) -> Self::Output {
        (&self) | rhs
    }
}
impl<'a> BitAnd<SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn bitand(self, rhs: SimpleBitset) -> Self::Output {
        self & (&rhs)
    }
}
impl<'a> BitOr<SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn bitor(self, rhs: SimpleBitset) -> Self::Output {
        self | (&rhs)
    }
}

impl SubAssign<&SimpleBitset> for SimpleBitset {
    fn sub_assign(&mut self, rhs: &SimpleBitset) {
        if self.is_empty() || rhs.is_empty() { return; }
        if self.is_all_fast() && rhs.is_all_fast() {
            *self = SimpleBitset::zeros();
            return;
        }
        self.rsb = &self.rsb - &rhs.rsb;
        self.update_cached();
    }
}
impl SubAssign<SimpleBitset> for SimpleBitset {
    fn sub_assign(&mut self, rhs: SimpleBitset) {
        *self -= &rhs
    }
}
impl Sub<SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn sub(self, rhs: SimpleBitset) -> Self::Output {
        (&self) - (&rhs)
    }
}
impl<'a> Sub<&'a SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn sub(self, rhs: &'a SimpleBitset) -> Self::Output {
        if self.is_empty() || rhs.is_empty() { return self.clone(); }
        if self.is_all_fast() && rhs.is_all_fast() { return SimpleBitset::zeros(); }
        let rsb = &self.rsb - &rhs.rsb;
        SimpleBitset::with_new_rsb_and_op(self, rhs, rsb, 0xC1)
    }
}

impl Not for SimpleBitset {
    type Output = SimpleBitset;
    fn not(self) -> Self::Output {
        if self.is_empty() { return SimpleBitset::all(); }
        if self.is_all_fast() { return SimpleBitset::zeros(); }
        let rsb = &universe_rsb() - &self.rsb;
        SimpleBitset::with_new_rsb_unary(&self, rsb, 0xD1)
    }
}
impl Not for &SimpleBitset {
    type Output = SimpleBitset;
    fn not(self) -> Self::Output {
        if self.is_empty() { return SimpleBitset::all(); }
        if self.is_all_fast() { return SimpleBitset::zeros(); }
        let rsb = &universe_rsb() - &self.rsb;
        SimpleBitset::with_new_rsb_unary(self, rsb, 0xD2)
    }
}

impl Serialize for SimpleBitset {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.rsb.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SimpleBitset {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let rsb = RangeSetBlaze::<usize>::deserialize(deserializer)?;
        Ok(SimpleBitset::from_rsb_inner(rsb))
    }
}
