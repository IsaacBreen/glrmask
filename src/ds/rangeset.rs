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
        self.ranges
            .iter()
            .flat_map(|&(lo, hi)| lo..=hi)
    }

    /// Access the underlying ranges slice.
    pub fn ranges(&self) -> &[(u32, u32)] {
        &self.ranges
    }
}

impl Default for RangeSet {
    fn default() -> Self {
        Self::new()
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
}
