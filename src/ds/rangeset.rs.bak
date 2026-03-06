//! Sparse range-based set.
//!
//! Represents a set of integers as sorted, non-overlapping, inclusive ranges.
//! Efficient for sets with large contiguous runs (e.g., Unicode character classes).

use serde::{Deserialize, Serialize};

/// A set of integers represented as sorted, non-overlapping inclusive ranges `[lo, hi]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RangeSet {
    /// Sorted, non-overlapping, non-adjacent ranges: `[(lo0, hi0), (lo1, hi1), ...]`
    /// where `lo_i <= hi_i` and `hi_i + 1 < lo_{i+1}`.
    ranges: Vec<(u32, u32)>,
}

impl RangeSet {
    /// Create an empty range set.
    pub fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    /// Create from pre-sorted, non-overlapping ranges.
    ///
    /// # Safety (logical)
    /// Caller must ensure ranges are sorted and non-overlapping.
    pub fn from_sorted(ranges: Vec<(u32, u32)>) -> Self {
        Self { ranges }
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Number of ranges.
    pub fn num_ranges(&self) -> usize {
        self.ranges.len()
    }

    /// Total number of elements in the set.
    pub fn cardinality(&self) -> u64 {
        self.ranges
            .iter()
            .map(|&(lo, hi)| (hi - lo + 1) as u64)
            .sum()
    }

    /// Check if a value is in the set.
    pub fn contains(&self, val: u32) -> bool {
        self.ranges
            .binary_search_by(|&(lo, hi)| {
                if val < lo {
                    std::cmp::Ordering::Greater
                } else if val > hi {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .is_ok()
    }

    /// Insert a single value.
    pub fn insert(&mut self, val: u32) {
        self.insert_range(val, val);
    }

    /// Insert an inclusive range `[lo, hi]`.
    pub fn insert_range(&mut self, lo: u32, hi: u32) {
        if lo > hi {
            return;
        }
        // Find ranges that overlap or are adjacent to [lo, hi]
        let mut new_lo = lo;
        let mut new_hi = hi;
        let mut first = self.ranges.len();
        let mut last = 0;

        for (i, &(rlo, rhi)) in self.ranges.iter().enumerate() {
            // Check if range overlaps or is adjacent
            if rhi + 1 >= lo && rlo <= hi + 1 {
                if i < first {
                    first = i;
                }
                last = i + 1;
                new_lo = new_lo.min(rlo);
                new_hi = new_hi.max(rhi);
            }
        }

        if first >= last {
            // No overlapping ranges, insert new one
            let pos = self.ranges.partition_point(|&(rlo, _)| rlo < lo);
            self.ranges.insert(pos, (new_lo, new_hi));
        } else {
            // Replace overlapping ranges with merged one
            self.ranges[first] = (new_lo, new_hi);
            self.ranges.drain(first + 1..last);
        }
    }

    /// Iterate over all ranges.
    pub fn iter_ranges(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.ranges.iter().copied()
    }

    /// Iterate over all values in the set.
    pub fn iter_values(&self) -> impl Iterator<Item = u32> + '_ {
        self.ranges.iter().flat_map(|&(lo, hi)| lo..=hi)
    }

    /// Access the underlying ranges slice.
    pub fn ranges(&self) -> &[(u32, u32)] {
        &self.ranges
    }

    // ---- Factory methods ----

    /// Create a set containing a single inclusive range `[lo, hi]`.
    pub fn from_range(lo: u32, hi: u32) -> Self {
        if lo > hi {
            Self::new()
        } else {
            Self {
                ranges: vec![(lo, hi)],
            }
        }
    }

    /// Create a set from an iterator of inclusive `(lo, hi)` pairs.
    /// The input does not need to be sorted or non-overlapping.
    pub fn from_ranges(iter: impl IntoIterator<Item = (u32, u32)>) -> Self {
        let mut ranges: Vec<(u32, u32)> = iter.into_iter().filter(|&(lo, hi)| lo <= hi).collect();
        if ranges.is_empty() {
            return Self::new();
        }
        ranges.sort_unstable();
        // Coalesce overlapping / adjacent ranges.
        let mut result = Vec::with_capacity(ranges.len());
        let mut cur = ranges[0];
        for &(lo, hi) in &ranges[1..] {
            if lo <= cur.1.saturating_add(1) {
                cur.1 = cur.1.max(hi);
            } else {
                result.push(cur);
                cur = (lo, hi);
            }
        }
        result.push(cur);
        Self { ranges: result }
    }

    /// Number of individual elements in the set (as `usize`).
    pub fn len(&self) -> usize {
        self.cardinality() as usize
    }

    // ---- Set operations ----

    /// Compute the union of two range sets.
    pub fn union(&self, other: &Self) -> Self {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        // Merge two sorted range lists, then coalesce.
        let mut merged = Vec::with_capacity(self.ranges.len() + other.ranges.len());
        let (mut i, mut j) = (0, 0);
        while i < self.ranges.len() && j < other.ranges.len() {
            if self.ranges[i].0 <= other.ranges[j].0 {
                merged.push(self.ranges[i]);
                i += 1;
            } else {
                merged.push(other.ranges[j]);
                j += 1;
            }
        }
        merged.extend_from_slice(&self.ranges[i..]);
        merged.extend_from_slice(&other.ranges[j..]);

        let mut result = Vec::with_capacity(merged.len());
        let mut cur = merged[0];
        for &(lo, hi) in &merged[1..] {
            if lo <= cur.1.saturating_add(1) {
                cur.1 = cur.1.max(hi);
            } else {
                result.push(cur);
                cur = (lo, hi);
            }
        }
        result.push(cur);
        Self { ranges: result }
    }

    /// Compute the intersection of two range sets.
    pub fn intersection(&self, other: &Self) -> Self {
        let mut result = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < self.ranges.len() && j < other.ranges.len() {
            let (a_lo, a_hi) = self.ranges[i];
            let (b_lo, b_hi) = other.ranges[j];
            let lo = a_lo.max(b_lo);
            let hi = a_hi.min(b_hi);
            if lo <= hi {
                result.push((lo, hi));
            }
            if a_hi <= b_hi {
                i += 1;
            } else {
                j += 1;
            }
        }
        Self { ranges: result }
    }

    /// Compute the set difference `self - other`.
    pub fn difference(&self, other: &Self) -> Self {
        if other.is_empty() {
            return self.clone();
        }
        if self.is_empty() {
            return Self::new();
        }
        let mut result = Vec::new();
        let mut b_idx = 0;

        for &(a_lo, a_hi) in &self.ranges {
            let mut cur = a_lo;

            // Advance past other-ranges that end before our start.
            while b_idx < other.ranges.len() && other.ranges[b_idx].1 < cur {
                b_idx += 1;
            }

            let mut k = b_idx;
            let mut consumed = false;

            while k < other.ranges.len() && cur <= a_hi {
                let (b_lo, b_hi) = other.ranges[k];
                if b_lo > a_hi {
                    break;
                }
                // Emit the part of self before this subtraction range.
                if cur < b_lo {
                    result.push((cur, b_lo - 1));
                }
                // Skip past the subtraction range.
                if b_hi >= a_hi {
                    consumed = true;
                    break;
                }
                cur = b_hi + 1; // Safe: b_hi < a_hi, so b_hi < u32::MAX.
                k += 1;
            }

            if !consumed && cur <= a_hi {
                result.push((cur, a_hi));
            }
        }
        Self { ranges: result }
    }

    /// Complement within `[0, max]` (inclusive).
    pub fn complement(&self, max: u32) -> Self {
        Self::from_range(0, max).difference(self)
    }

    /// Check whether two range sets are disjoint.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        let (mut i, mut j) = (0, 0);
        while i < self.ranges.len() && j < other.ranges.len() {
            let (a_lo, a_hi) = self.ranges[i];
            let (b_lo, b_hi) = other.ranges[j];
            if a_lo <= b_hi && b_lo <= a_hi {
                return false;
            }
            if a_hi < b_lo {
                i += 1;
            } else {
                j += 1;
            }
        }
        true
    }

    /// Check whether `self ⊆ other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.difference(other).is_empty()
    }
}

impl Default for RangeSet {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::BitOr for &RangeSet {
    type Output = RangeSet;
    fn bitor(self, rhs: &RangeSet) -> RangeSet {
        self.union(rhs)
    }
}

impl std::ops::BitAnd for &RangeSet {
    type Output = RangeSet;
    fn bitand(self, rhs: &RangeSet) -> RangeSet {
        self.intersection(rhs)
    }
}

impl std::ops::Sub for &RangeSet {
    type Output = RangeSet;
    fn sub(self, rhs: &RangeSet) -> RangeSet {
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
        for (i, &(lo, hi)) in self.ranges.iter().enumerate() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let mut rs = RangeSet::new();
        rs.insert(5);
        rs.insert(3);
        rs.insert(4);
        assert_eq!(rs.num_ranges(), 1);
        assert_eq!(rs.ranges(), &[(3, 5)]);
        assert!(rs.contains(3));
        assert!(rs.contains(4));
        assert!(rs.contains(5));
        assert!(!rs.contains(2));
        assert!(!rs.contains(6));
    }

    #[test]
    fn test_merge() {
        let mut rs = RangeSet::new();
        rs.insert_range(1, 3);
        rs.insert_range(5, 7);
        assert_eq!(rs.num_ranges(), 2);
        rs.insert_range(3, 5);
        assert_eq!(rs.num_ranges(), 1);
        assert_eq!(rs.ranges(), &[(1, 7)]);
    }

    #[test]
    fn test_from_range() {
        let rs = RangeSet::from_range(3, 7);
        assert_eq!(rs.ranges(), &[(3, 7)]);
        assert_eq!(rs.len(), 5);

        let empty = RangeSet::from_range(5, 3);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_from_ranges() {
        let rs = RangeSet::from_ranges(vec![(5, 7), (1, 3), (3, 6)]);
        assert_eq!(rs.ranges(), &[(1, 7)]);
    }

    #[test]
    fn test_union() {
        let a = RangeSet::from_ranges(vec![(1, 3), (7, 9)]);
        let b = RangeSet::from_ranges(vec![(2, 5), (11, 13)]);
        let u = a.union(&b);
        assert_eq!(u.ranges(), &[(1, 5), (7, 9), (11, 13)]);
    }

    #[test]
    fn test_union_adjacent() {
        let a = RangeSet::from_range(1, 3);
        let b = RangeSet::from_range(4, 6);
        let u = a.union(&b);
        assert_eq!(u.ranges(), &[(1, 6)]);
    }

    #[test]
    fn test_intersection() {
        let a = RangeSet::from_ranges(vec![(1, 5), (10, 15)]);
        let b = RangeSet::from_ranges(vec![(3, 12)]);
        let i = a.intersection(&b);
        assert_eq!(i.ranges(), &[(3, 5), (10, 12)]);
    }

    #[test]
    fn test_intersection_disjoint() {
        let a = RangeSet::from_range(1, 3);
        let b = RangeSet::from_range(5, 7);
        assert!(a.intersection(&b).is_empty());
    }

    #[test]
    fn test_difference() {
        let a = RangeSet::from_range(1, 10);
        let b = RangeSet::from_ranges(vec![(3, 5), (8, 8)]);
        let d = a.difference(&b);
        assert_eq!(d.ranges(), &[(1, 2), (6, 7), (9, 10)]);
    }

    #[test]
    fn test_difference_complete() {
        let a = RangeSet::from_range(1, 5);
        let b = RangeSet::from_range(0, 10);
        assert!(a.difference(&b).is_empty());
    }

    #[test]
    fn test_complement() {
        let a = RangeSet::from_ranges(vec![(2, 4), (7, 8)]);
        let c = a.complement(10);
        assert_eq!(c.ranges(), &[(0, 1), (5, 6), (9, 10)]);
    }

    #[test]
    fn test_is_disjoint() {
        let a = RangeSet::from_range(1, 3);
        let b = RangeSet::from_range(4, 6);
        assert!(a.is_disjoint(&b));

        let c = RangeSet::from_range(3, 5);
        assert!(!a.is_disjoint(&c));
    }

    #[test]
    fn test_is_subset() {
        let a = RangeSet::from_range(3, 5);
        let b = RangeSet::from_range(1, 10);
        assert!(a.is_subset(&b));
        assert!(!b.is_subset(&a));
    }

    #[test]
    fn test_operators() {
        let a = RangeSet::from_range(1, 5);
        let b = RangeSet::from_range(3, 8);
        assert_eq!((&a | &b).ranges(), &[(1, 8)]);
        assert_eq!((&a & &b).ranges(), &[(3, 5)]);
        assert_eq!((&a - &b).ranges(), &[(1, 2)]);
    }

    #[test]
    fn test_bitor_assign() {
        let mut a = RangeSet::from_range(1, 3);
        a |= &RangeSet::from_range(5, 7);
        assert_eq!(a.ranges(), &[(1, 3), (5, 7)]);
        a |= &RangeSet::from_range(3, 5);
        assert_eq!(a.ranges(), &[(1, 7)]);
    }

    #[test]
    fn test_display() {
        let rs = RangeSet::from_ranges(vec![(1, 1), (3, 5)]);
        assert_eq!(format!("{rs}"), "{1, 3..=5}");
    }
}
