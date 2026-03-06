//! Sparse range-based set backed by [`range_set_blaze::RangeSetBlaze`].
//!
//! Represents a set of `u32` integers as sorted, non-overlapping, inclusive ranges.
//! The underlying `RangeSetBlaze` provides efficient set operations (union,
//! intersection, difference, complement) with optimal range merging.

use range_set_blaze::RangeSetBlaze;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A set of `u32` integers represented as sorted, non-overlapping inclusive ranges.
///
/// Wraps `Arc<RangeSetBlaze<u32>>` for O(1) clone and efficient set operations.
#[derive(Debug, Clone)]
pub struct RangeSet {
    inner: Arc<RangeSetBlaze<u32>>,
}

impl RangeSet {
    /// Create an empty range set.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RangeSetBlaze::new()),
        }
    }

    /// Create from a single inclusive range `[lo, hi]`.
    pub fn from_range(lo: u32, hi: u32) -> Self {
        if lo > hi {
            return Self::new();
        }
        Self {
            inner: Arc::new(RangeSetBlaze::from_iter([lo..=hi])),
        }
    }

    /// Create from an iterator of inclusive `(lo, hi)` pairs.
    ///
    /// Ranges need not be sorted or non-overlapping; `RangeSetBlaze` handles merging.
    pub fn from_ranges(iter: impl IntoIterator<Item = (u32, u32)>) -> Self {
        let rsb: RangeSetBlaze<u32> = iter
            .into_iter()
            .filter(|(lo, hi)| lo <= hi)
            .map(|(lo, hi)| lo..=hi)
            .collect();
        Self {
            inner: Arc::new(rsb),
        }
    }

    /// Create from pre-sorted, non-overlapping ranges.
    ///
    /// For API compatibility. The ranges are fed into RangeSetBlaze which
    /// handles any needed merging.
    pub fn from_sorted(ranges: Vec<(u32, u32)>) -> Self {
        Self::from_ranges(ranges)
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Number of contiguous ranges.
    pub fn num_ranges(&self) -> usize {
        self.inner.ranges_len() as usize
    }

    /// Total number of elements in the set.
    pub fn cardinality(&self) -> u64 {
        self.inner.len() as u64
    }

    /// Check if a value is in the set.
    pub fn contains(&self, val: u32) -> bool {
        self.inner.contains(val)
    }

    /// Insert a single value.
    pub fn insert(&mut self, val: u32) {
        let mut new = (*self.inner).clone();
        new.insert(val);
        self.inner = Arc::new(new);
    }

    /// Insert an inclusive range `[lo, hi]`.
    pub fn insert_range(&mut self, lo: u32, hi: u32) {
        if lo > hi {
            return;
        }
        let mut new = (*self.inner).clone();
        new.ranges_insert(lo..=hi);
        self.inner = Arc::new(new);
    }

    /// Iterate over contiguous ranges as `(lo, hi)` inclusive pairs.
    pub fn iter_ranges(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.inner.ranges().map(|r| (*r.start(), *r.end()))
    }

    /// Iterate over all individual values.
    pub fn iter_values(&self) -> impl Iterator<Item = u32> + '_ {
        self.inner.iter()
    }

    /// Get the underlying ranges as a Vec of `(lo, hi)` pairs for compatibility.
    pub fn ranges(&self) -> Vec<(u32, u32)> {
        self.iter_ranges().collect()
    }

    /// Number of contiguous ranges (alias for `num_ranges`).
    pub fn len(&self) -> usize {
        self.num_ranges()
    }

    /// Compute the union of two sets.
    pub fn union(&self, other: &Self) -> Self {
        Self {
            inner: Arc::new(&*self.inner | &*other.inner),
        }
    }

    /// Compute the intersection of two sets.
    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            inner: Arc::new(&*self.inner & &*other.inner),
        }
    }

    /// Compute the set difference `self − other`.
    pub fn difference(&self, other: &Self) -> Self {
        Self {
            inner: Arc::new(&*self.inner - &*other.inner),
        }
    }

    /// Compute the complement within `[0, max]`.
    pub fn complement(&self, max: u32) -> Self {
        let universe = RangeSetBlaze::from_iter([0..=max]);
        Self {
            inner: Arc::new(&universe - &*self.inner),
        }
    }

    /// Check if two sets are disjoint.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.inner.is_disjoint(&other.inner)
    }

    /// Check whether `self ⊆ other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.inner.is_subset(&other.inner)
    }

    /// Access the underlying `RangeSetBlaze`.
    pub fn as_inner(&self) -> &RangeSetBlaze<u32> {
        &self.inner
    }
}

impl Default for RangeSet {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for RangeSet {
    fn eq(&self, other: &Self) -> bool {
        // Arc pointer equality first, then content equality
        Arc::ptr_eq(&self.inner, &other.inner) || *self.inner == *other.inner
    }
}

impl Eq for RangeSet {}

impl std::hash::Hash for RangeSet {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for r in self.inner.ranges() {
            r.start().hash(state);
            r.end().hash(state);
        }
    }
}

// ---- Operator impls ----

impl std::ops::BitOr for &RangeSet {
    type Output = RangeSet;
    fn bitor(self, rhs: Self) -> RangeSet {
        self.union(rhs)
    }
}

impl std::ops::BitAnd for &RangeSet {
    type Output = RangeSet;
    fn bitand(self, rhs: Self) -> RangeSet {
        self.intersection(rhs)
    }
}

impl std::ops::Sub for &RangeSet {
    type Output = RangeSet;
    fn sub(self, rhs: Self) -> RangeSet {
        self.difference(rhs)
    }
}

impl std::ops::BitOrAssign<&RangeSet> for RangeSet {
    fn bitor_assign(&mut self, rhs: &RangeSet) {
        *self = self.union(rhs);
    }
}

impl std::fmt::Display for RangeSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        for (i, (lo, hi)) in self.iter_ranges().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if lo == hi {
                write!(f, "{lo}")?;
            } else {
                write!(f, "{lo}..={hi}")?;
            }
        }
        write!(f, "}}")
    }
}

// ---- Serde ----

impl Serialize for RangeSet {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Serialize as flat array: [lo0, hi0, lo1, hi1, ...]
        let flat: Vec<u32> = self
            .iter_ranges()
            .flat_map(|(lo, hi)| [lo, hi])
            .collect();
        flat.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RangeSet {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let flat: Vec<u32> = Vec::deserialize(deserializer)?;
        let ranges: Vec<(u32, u32)> = flat
            .chunks(2)
            .filter_map(|c| {
                if c.len() == 2 {
                    Some((c[0], c[1]))
                } else {
                    None
                }
            })
            .collect();
        Ok(Self::from_ranges(ranges))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let s = RangeSet::new();
        assert!(s.is_empty());
        assert_eq!(s.cardinality(), 0);
        assert!(!s.contains(0));
    }

    #[test]
    fn test_from_range() {
        let s = RangeSet::from_range(3, 7);
        assert!(!s.is_empty());
        assert_eq!(s.cardinality(), 5);
        assert!(s.contains(3));
        assert!(s.contains(5));
        assert!(s.contains(7));
        assert!(!s.contains(2));
        assert!(!s.contains(8));
    }

    #[test]
    fn test_union() {
        let a = RangeSet::from_range(1, 5);
        let b = RangeSet::from_range(4, 8);
        let u = a.union(&b);
        assert_eq!(u.cardinality(), 8); // 1..=8
        assert!(u.contains(1));
        assert!(u.contains(8));
    }

    #[test]
    fn test_intersection() {
        let a = RangeSet::from_range(1, 5);
        let b = RangeSet::from_range(4, 8);
        let i = a.intersection(&b);
        assert_eq!(i.cardinality(), 2); // 4, 5
        assert!(i.contains(4));
        assert!(i.contains(5));
        assert!(!i.contains(3));
        assert!(!i.contains(6));
    }

    #[test]
    fn test_difference() {
        let a = RangeSet::from_range(1, 5);
        let b = RangeSet::from_range(4, 8);
        let d = a.difference(&b);
        assert_eq!(d.cardinality(), 3); // 1, 2, 3
        assert!(d.contains(1));
        assert!(d.contains(3));
        assert!(!d.contains(4));
    }

    #[test]
    fn test_complement() {
        let s = RangeSet::from_range(3, 5);
        let c = s.complement(9);
        assert_eq!(c.cardinality(), 7); // 0,1,2,6,7,8,9
        assert!(c.contains(0));
        assert!(c.contains(2));
        assert!(!c.contains(3));
        assert!(!c.contains(5));
        assert!(c.contains(6));
        assert!(c.contains(9));
    }

    #[test]
    fn test_insert() {
        let mut s = RangeSet::new();
        s.insert(5);
        s.insert(3);
        s.insert(4);
        assert_eq!(s.cardinality(), 3);
        assert_eq!(s.num_ranges(), 1); // merged into 3..=5
    }

    #[test]
    fn test_insert_range() {
        let mut s = RangeSet::new();
        s.insert_range(1, 3);
        s.insert_range(5, 7);
        assert_eq!(s.cardinality(), 6);
        assert_eq!(s.num_ranges(), 2);
        s.insert_range(3, 5);
        assert_eq!(s.cardinality(), 7);
        assert_eq!(s.num_ranges(), 1); // merged into 1..=7
    }

    #[test]
    fn test_from_ranges() {
        let s = RangeSet::from_ranges([(1, 3), (5, 7), (2, 6)]);
        assert_eq!(s.cardinality(), 7); // 1..=7 merged
        assert_eq!(s.num_ranges(), 1);
    }

    #[test]
    fn test_is_subset() {
        let a = RangeSet::from_range(3, 5);
        let b = RangeSet::from_range(1, 9);
        assert!(a.is_subset(&b));
        assert!(!b.is_subset(&a));
    }

    #[test]
    fn test_is_disjoint() {
        let a = RangeSet::from_range(1, 3);
        let b = RangeSet::from_range(5, 7);
        assert!(a.is_disjoint(&b));
        let c = RangeSet::from_range(3, 5);
        assert!(!a.is_disjoint(&c));
    }

    #[test]
    fn test_serde_roundtrip() {
        let s = RangeSet::from_ranges([(1, 3), (7, 9)]);
        let json = serde_json::to_string(&s).unwrap();
        let s2: RangeSet = serde_json::from_str(&json).unwrap();
        assert_eq!(s, s2);
    }

    #[test]
    fn test_equality() {
        let a = RangeSet::from_range(1, 5);
        let b = RangeSet::from_range(1, 5);
        assert_eq!(a, b);

        let c = RangeSet::from_range(1, 4);
        assert_ne!(a, c);
    }
}
