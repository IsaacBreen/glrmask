use crate::datastructures::hybrid_bitset::HybridBitset;
use range_set_blaze::prelude::*;
use range_set_blaze::RangeMapBlaze;
use std::ops::{RangeInclusive, SubAssign};
use std::collections::BTreeSet;
use std::iter::FromIterator;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Sub};
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::cmp::Ordering;

/// A two-dimensional bitset, conceptually a map from `usize` to `HybridBitset`.
///
/// This structure uses a `RangeMapBlaze` to efficiently store `HybridBitset`s
/// for ranges of first-level indices. This is efficient when many consecutive
/// first-level indices map to the same `HybridBitset` or are empty.
///
/// An empty `HybridBitset` is never stored; if a row becomes empty, it is
/// removed from the map.
#[derive(Clone, PartialEq, Eq)]
pub struct HybridL2Bitset {
    /// The underlying map from usize (L1 index) to a HybridBitset (L2 indices).
    inner: RangeMapBlaze<usize, HybridBitset>,
}

impl Debug for HybridL2Bitset {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let f_alternate = f.alternate();
        let mut ds = f.debug_struct("HybridL2Bitset");

        const MAX_RANGES_TO_SHOW: usize = 5;
        let total_ranges = self.inner.ranges_len();

        if f_alternate || total_ranges <= MAX_RANGES_TO_SHOW {
            // In alternate mode or for small sets, show full inner map.
            // The inner HybridBitsets already have a truncating Debug impl.
            ds.field("inner", &self.inner);
        } else {
            // For normal mode with many ranges, show a preview.
            let ranges_to_show: Vec<_> = self.inner.range_values().take(MAX_RANGES_TO_SHOW).collect();
            ds.field("inner_preview", &ranges_to_show);
            ds.field("...", &format_args!("and {} more ranges", total_ranges - MAX_RANGES_TO_SHOW));
        }

        ds.finish()
    }
}

impl PartialOrd for HybridL2Bitset {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HybridL2Bitset {
    fn cmp(&self, other: &Self) -> Ordering {
        let mut self_iter = self.inner.range_values();
        let mut other_iter = other.inner.range_values();

        loop {
            match (self_iter.next(), other_iter.next()) {
                (Some((r1, v1)), Some((r2, v2))) => {
                    // Compare ranges lexicographically by start, then by end.
                    let range_cmp = r1.start().cmp(r2.start()).then_with(|| r1.end().cmp(r2.end()));
                    if range_cmp != Ordering::Equal {
                        return range_cmp;
                    }
                    // If ranges are identical, compare the associated HybridBitset values.
                    let value_cmp = v1.cmp(v2);
                    if value_cmp != Ordering::Equal {
                        return value_cmp;
                    }
                    // Continue to the next element if the current ones are equal.
                }
                (Some(_), None) => return Ordering::Greater, // self has more elements
                (None, Some(_)) => return Ordering::Less,    // other has more elements
                (None, None) => return Ordering::Equal,     // Both iterators are exhausted
            }
        }
    }
}

impl Hash for HybridL2Bitset {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (range, value) in self.inner.range_values() {
            range.hash(state);
            value.hash(state);
        }
    }
}

impl HybridL2Bitset {
    /// Creates a new, empty `HybridL2Bitset`.
    pub fn new() -> Self {
        HybridL2Bitset {
            inner: RangeMapBlaze::new(),
        }
    }

    /// Creates a new `HybridL2Bitset` with all entries allowed.
    pub fn all() -> Self {
        HybridL2Bitset {
            inner: RangeMapBlaze::from_iter(std::iter::once((
                0..=usize::MAX,
                HybridBitset::max_ones(),
            ))),
        }
    }

    /// Returns `true` if the set contains no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the total number of set bits in the entire 2D bitset.
    ///
    /// This can be an expensive operation as it iterates through all
    /// ranges and sums the lengths of the contained `HybridBitset`s.
    pub fn len(&self) -> usize {
        self.inner.iter().map(|(_, bitset)| bitset.len()).sum()
    }

    /// Clears the entire set, removing all points.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Inserts a 2D point (l1_index, l2_index) into the set.
    pub fn insert(&mut self, l1_index: usize, l2_index: usize) {
        let mut bitset = self.inner.remove(l1_index).unwrap_or_else(HybridBitset::zeros);
        bitset.insert(l2_index);
        self.inner.insert(l1_index, bitset);
    }

    pub fn insert_l2_bitset(&mut self, l1_index: usize, bitset: HybridBitset) {
        if !bitset.is_empty() {
            self.inner.insert(l1_index, bitset);
        } else {
            self.inner.remove(l1_index);
        }
    }

    /// Removes a 2D point (l1_index, l2_index) from the set.
    ///
    /// Returns `true` if the point was present in the set.
    pub fn remove(&mut self, l1_index: usize, l2_index: usize) -> bool {
        if let Some(mut bitset) = self.inner.remove(l1_index) {
            let was_present = bitset.remove(l2_index);
            // If the bitset is not empty after removal, we must insert it back.
            if !bitset.is_empty() {
                self.inner.insert(l1_index, bitset);
            }
            was_present
        } else {
            false // No bitset at l1_index.
        }
    }

    pub fn remove_l1(&mut self, l1_index: usize) -> Option<HybridBitset> {
        self.inner.remove(l1_index)
    }

    /// Checks if a 2D point (l1_index, l2_index) is present in the set.
    pub fn contains(&self, l1_index: usize, l2_index: usize) -> bool {
        self.inner
            .get(l1_index)
            .map_or(false, |bitset| bitset.contains(l2_index))
    }

    /// Returns the `HybridBitset` for a given first-level index.
    ///
    /// If no bits are set for this `l1_index`, it returns `None`.
    pub fn get_l2_bitset(&self, l1_index: usize) -> Option<&HybridBitset> {
        self.inner.get(l1_index)
    }

    /// Returns an iterator over all set points `(l1_index, l2_index)`.
    /// The points are yielded in lexicographical order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.inner.iter().flat_map(|(l1_index, bitset)| {
            bitset.iter().map(move |l2_index| (l1_index, l2_index))
        })
    }

    pub fn iter_l1_bitsets(&self) -> impl Iterator<Item = (usize, &HybridBitset)> {
        self.inner.iter()
    }

    /// Returns an iterator over ranges of L1 indices and their corresponding L2 bitsets.
    pub fn range_values(&self) -> impl Iterator<Item = (RangeInclusive<usize>, &HybridBitset)> {
        self.inner.range_values()
    }

    /// Computes the complement of the set.
    ///
    /// For every first-level index (`l1_index`) present in the set, its
    /// corresponding `HybridBitset` is inverted.
    ///
    /// First-level indices that are not present in the original set will not be
    /// present in the complement.
    pub fn complement(&self) -> Self {
        let complemented_values = self
            .inner
            .range_values()
            .map(|(range, bitset)| (range, bitset.inverted()));

        HybridL2Bitset {
            inner: complemented_values.collect(),
        }
    }

    /// A generalized intersection operation.
    ///
    /// For L1 keys present in both sets, the corresponding L2 bitsets are intersected.
    /// For L1 keys present in only one set, the L2 bitset is intersected with the `default`.
    /// If `default` is `None`, keys present in only one set are excluded from the result.
    pub fn intersection_with(&self, other: &Self, default: Option<HybridBitset>) -> Self {
        self.zip_op(other, default, |a, b| a & b)
    }

    /// A generalized union operation.
    ///
    /// For L1 keys present in both sets, the corresponding L2 bitsets are unioned.
    /// For L1 keys present in only one set, the L2 bitset is unioned with the `default`.
    /// If `default` is `None`, keys present in only one set are excluded from the result.
    pub fn union_with(&self, other: &Self, default: Option<HybridBitset>) -> Self {
        self.zip_op(other, default, |a, b| a | b)
    }

    /// A generalized symmetric difference operation.
    ///
    /// For L1 keys present in both sets, the corresponding L2 bitsets are XORed.
    /// For L1 keys present in only one set, the L2 bitset is XORed with the `default`.
    /// If `default` is `None`, keys present in only one set are excluded from the result.
    pub fn symmetric_difference_with(&self, other: &Self, default: Option<HybridBitset>) -> Self {
        self.zip_op(other, default, |a, b| a ^ b)
    }

    /// Helper function to perform a zip operation between two `HybridL2Bitset`s.
    fn zip_op<F>(&self, other: &Self, default: Option<HybridBitset>, op: F) -> Self
    where
        F: Fn(&HybridBitset, &HybridBitset) -> HybridBitset,
    {
        let mut points = BTreeSet::new();
        if !self.inner.is_empty() || !other.inner.is_empty() || default.is_some() {
            points.insert(0);
        }

        for (range, _) in self.inner.range_values() {
            points.insert(*range.start());
            if *range.end() < usize::MAX {
                points.insert(range.end() + 1);
            }
        }
        for (range, _) in other.inner.range_values() {
            points.insert(*range.start());
            if *range.end() < usize::MAX {
                points.insert(range.end() + 1);
            }
        }

        let mut new_ranges = Vec::new();
        let empty_bs = HybridBitset::zeros();

        let mut process_interval = |start, end| {
            if start > end {
                return;
            }
            let mid = start;

            let self_bs_opt = self.inner.get(mid);
            let other_bs_opt = other.inner.get(mid);

            let result_bs = match (self_bs_opt, other_bs_opt) {
                (Some(s), Some(o)) => op(s, o),
                (Some(s), None) => {
                    if let Some(ref d) = default {
                        op(s, d)
                    } else {
                        return; // Exclude if no default
                    }
                }
                (None, Some(o)) => {
                    if let Some(ref d) = default {
                        op(d, o)
                    } else {
                        return; // Exclude if no default
                    }
                }
                (None, None) => {
                    // Both are implicitly empty.
                    // The `default` is for when a key is in one but not the other.
                    // When both are missing, it's op(empty, empty).
                    op(&empty_bs, &empty_bs)
                }
            };

            if !result_bs.is_empty() {
                new_ranges.push((start..=end, result_bs));
            }
        };

        let points_vec: Vec<_> = points.into_iter().collect();
        for window in points_vec.windows(2) {
            process_interval(window[0], window[1] - 1);
        }

        if let Some(&last_point) = points_vec.last() {
            if last_point <= usize::MAX {
                process_interval(last_point, usize::MAX);
            }
        }

        HybridL2Bitset {
            inner: RangeMapBlaze::from_iter(new_ranges),
        }
    }
}

impl Default for HybridL2Bitset {
    fn default() -> Self {
        Self::new()
    }
}

// --- Bitwise Operations ---

impl BitAnd for &HybridL2Bitset {
    type Output = HybridL2Bitset;

    fn bitand(self, rhs: Self) -> Self::Output {
        // Standard intersection excludes keys not present in both sets.
        // This is equivalent to intersection_with a default of None.
        self.intersection_with(rhs, None)
    }
}

impl BitOr for &HybridL2Bitset {
    type Output = HybridL2Bitset;

    fn bitor(self, rhs: Self) -> Self::Output {
        // Standard union includes keys from both. If a key is missing from one,
        // it's treated as an empty set. This is equivalent to union_with a
        // default of an empty HybridBitset.
        self.union_with(rhs, Some(HybridBitset::zeros()))
    }
}

impl BitXor for &HybridL2Bitset {
    type Output = HybridL2Bitset;

    fn bitxor(self, rhs: Self) -> Self::Output {
        // Standard symmetric difference includes keys from both sets.
        // If a key is missing from one, it's treated as an empty set.
        // This is equivalent to symmetric_difference_with a default of Some(HybridBitset::zeros()).
        self.symmetric_difference_with(rhs, Some(HybridBitset::zeros()))
    }
}

impl Sub for &HybridL2Bitset {
    type Output = HybridL2Bitset;
    fn sub(self, rhs: Self) -> Self::Output {
        self.zip_op(rhs, Some(HybridBitset::zeros()), |a, b| a - b)
    }
}

// --- In-place Bitwise Operations ---

impl BitAndAssign for HybridL2Bitset {
    fn bitand_assign(&mut self, rhs: Self) {
        *self &= &rhs;
    }
}

impl BitAndAssign<&HybridL2Bitset> for HybridL2Bitset {
    fn bitand_assign(&mut self, rhs: &HybridL2Bitset) {
        *self = &*self & rhs;
    }
}

impl BitOrAssign for HybridL2Bitset {
    fn bitor_assign(&mut self, rhs: Self) {
        *self |= &rhs;
    }
}

impl BitOrAssign<&HybridL2Bitset> for HybridL2Bitset {
    fn bitor_assign(&mut self, rhs: &HybridL2Bitset) {
        *self = &*self | rhs;
    }
}

impl BitXorAssign for HybridL2Bitset {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self ^= &rhs;
    }
}

impl BitXorAssign<&HybridL2Bitset> for HybridL2Bitset {
    fn bitxor_assign(&mut self, rhs: &HybridL2Bitset) {
        *self = &*self ^ rhs;
    }
}

impl SubAssign for HybridL2Bitset {
    fn sub_assign(&mut self, rhs: Self) {
        *self = &*self - &rhs;
    }
}

impl SubAssign<&HybridL2Bitset> for HybridL2Bitset {
    fn sub_assign(&mut self, rhs: &HybridL2Bitset) {
        *self = &*self - rhs;
    }
}

// --- Operations on owned values ---
impl<'a> BitAnd<&'a HybridL2Bitset> for HybridL2Bitset {
    type Output = HybridL2Bitset;
    fn bitand(self, rhs: &'a HybridL2Bitset) -> Self::Output {
        &self & rhs
    }
}

impl<'a> BitOr<&'a HybridL2Bitset> for HybridL2Bitset {
    type Output = HybridL2Bitset;
    fn bitor(self, rhs: &'a HybridL2Bitset) -> Self::Output {
        &self | rhs
    }
}

impl<'a> BitXor<&'a HybridL2Bitset> for HybridL2Bitset {
    type Output = HybridL2Bitset;
    fn bitxor(self, rhs: &'a HybridL2Bitset) -> Self::Output {
        &self ^ rhs
    }
}

impl<'a> Sub<&'a HybridL2Bitset> for HybridL2Bitset {
    type Output = HybridL2Bitset;
    fn sub(self, rhs: &'a HybridL2Bitset) -> Self::Output {
        &self - rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn test_new_and_is_empty() {
        let set = HybridL2Bitset::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn test_insert_and_contains() {
        let mut set = HybridL2Bitset::new();
        assert!(!set.contains(10, 100));

        set.insert(10, 100);
        assert!(set.contains(10, 100));
        assert!(!set.contains(10, 101));
        assert!(!set.contains(11, 100));
        assert!(!set.is_empty());

        set.insert(10, 101);
        assert!(set.contains(10, 100));
        assert!(set.contains(10, 101));

        set.insert(11, 100);
        assert!(set.contains(11, 100));
    }

    #[test]
    fn test_remove() {
        let mut set = HybridL2Bitset::new();
        set.insert(10, 100);
        set.insert(10, 101);
        set.insert(20, 200);

        assert!(set.remove(10, 100));
        assert!(!set.contains(10, 100));
        assert!(set.contains(10, 101));
        assert!(set.contains(20, 200));

        assert!(!set.remove(10, 99)); // Was not present

        // Removing the last element of a row should remove the row
        assert!(set.get_l2_bitset(10).is_some());
        assert!(set.remove(10, 101));
        assert!(!set.contains(10, 101));
        assert!(set.get_l2_bitset(10).is_none());

        assert!(!set.remove(30, 300)); // Row not present
    }

    #[test]
    fn test_len_and_clear() {
        let mut set = HybridL2Bitset::new();
        set.insert(1, 10);
        set.insert(1, 20);
        set.insert(1, 30);
        set.insert(100, 10);
        set.insert(100, 20);

        assert_eq!(set.len(), 5);

        set.remove(1, 20);
        assert_eq!(set.len(), 4);

        set.clear();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }

    #[test]
    fn test_iter() {
        let mut set = HybridL2Bitset::new();
        set.insert(10, 100);
        set.insert(2, 50);
        set.insert(10, 101);
        set.insert(5, 80);

        let expected: BTreeSet<(usize, usize)> =
            vec![(2, 50), (5, 80), (10, 100), (10, 101)]
                .into_iter()
                .collect();

        let collected: BTreeSet<(usize, usize)> = set.iter().collect();
        assert_eq!(collected, expected);

        let empty_set = HybridL2Bitset::new();
        assert_eq!(empty_set.iter().count(), 0);
    }

    #[test]
    fn test_bitwise_ops() {
        let mut set1 = HybridL2Bitset::new();
        set1.insert(10, 100);
        set1.insert(10, 101);
        set1.insert(20, 200);

        let mut set2 = HybridL2Bitset::new();
        set2.insert(11, 101);
        set2.insert(11, 102);
        set2.insert(20, 200);
        set2.insert(30, 300);

        // Intersection
        let intersection = &set1 & &set2;
        let expected_intersection: BTreeSet<(usize, usize)> = vec![(20, 200)].into_iter().collect();
        assert_eq!(
            intersection.iter().collect::<BTreeSet<_>>(),
            expected_intersection
        );

        // Union
        let union = &set1 | &set2;
        let expected_union: BTreeSet<(usize, usize)> = vec![
            (10, 100),
            (10, 101),
            (11, 101),
            (11, 102),
            (20, 200),
            (30, 300),
        ]
        .into_iter()
        .collect();
        assert_eq!(union.iter().collect::<BTreeSet<_>>(), expected_union);

        // Symmetric Difference (XOR)
        let xor = &set1 ^ &set2;
        let expected_xor: BTreeSet<(usize, usize)> = vec![
            (10, 100),
            (10, 101),
            (11, 101),
            (11, 102),
            (30, 300),
        ]
        .into_iter()
        .collect();
        assert_eq!(xor.iter().collect::<BTreeSet<_>>(), expected_xor);
    }

    #[test]
    fn test_bitwise_assign_ops() {
        let mut set1_orig = HybridL2Bitset::new();
        set1_orig.insert(10, 100);
        set1_orig.insert(20, 200);

        let mut set2 = HybridL2Bitset::new();
        set2.insert(20, 200);
        set2.insert(30, 300);

        let mut set1_and = set1_orig.clone();
        set1_and &= &set2;
        assert_eq!(
            set1_and.iter().collect::<BTreeSet<_>>(),
            vec![(20, 200)].into_iter().collect()
        );

        let mut set1_or = set1_orig.clone();
        set1_or |= &set2;
        assert_eq!(
            set1_or.iter().collect::<BTreeSet<_>>(),
            vec![(10, 100), (20, 200), (30, 300)]
                .into_iter()
                .collect()
        );

        let mut set1_xor = set1_orig.clone();
        set1_xor ^= &set2;
        assert_eq!(
            set1_xor.iter().collect::<BTreeSet<_>>(),
            vec![(10, 100), (30, 300)].into_iter().collect()
        );
    }

    #[test]
    fn test_get_l2_bitset() {
        let mut set = HybridL2Bitset::new();
        set.insert(5, 50);
        set.insert(5, 51);

        let l2_set = set.get_l2_bitset(5).unwrap();
        assert_eq!(l2_set.len(), 2);
        assert!(l2_set.contains(50));
        assert!(l2_set.contains(51));

        assert!(set.get_l2_bitset(99).is_none());
    }

    #[test]
    fn test_with_ops() {
        let mut set1 = HybridL2Bitset::new();
        set1.insert(10, 100); // in set1 only
        set1.insert(20, 200); // in both

        let mut set2 = HybridL2Bitset::new();
        set2.insert(20, 201); // in both
        set2.insert(30, 300); // in set2 only

        // intersection_with
        // with None default (same as bitand)
        let inter_none = set1.intersection_with(&set2, None);
        let mut expected_inter = HybridL2Bitset::new();
        // key 20 is in both. intersection of {200} and {201} is empty.
        assert!(inter_none.is_empty());
        assert_eq!(inter_none, &set1 & &set2);

        // with non-empty default
        let default_bs = HybridBitset::from_iter(vec![100, 200, 300]);
        let inter_default = set1.intersection_with(&set2, Some(default_bs.clone()));
        // key 10: {100} & {100,200,300} -> {100}
        // key 20: {200} & {201} -> {}
        // key 30: {100,200,300} & {300} -> {300}
        let mut expected_inter_default = HybridL2Bitset::new();
        expected_inter_default.insert(10, 100);
        expected_inter_default.insert(30, 300);
        assert_eq!(inter_default, expected_inter_default);

        // union_with
        // with None default (only common keys)
        let union_none = set1.union_with(&set2, None);
        // key 10: excluded
        // key 20: {200} | {201} -> {200, 201}
        // key 30: excluded
        let mut expected_union_none = HybridL2Bitset::new();
        expected_union_none.insert(20, 200);
        expected_union_none.insert(20, 201);
        assert_eq!(union_none, expected_union_none);

        // with zeros default (same as bitor)
        let union_zeros = set1.union_with(&set2, Some(HybridBitset::zeros()));
        let mut expected_union_zeros = HybridL2Bitset::new();
        expected_union_zeros.insert(10, 100);
        expected_union_zeros.insert(20, 200);
        expected_union_zeros.insert(20, 201);
        expected_union_zeros.insert(30, 300);
        assert_eq!(union_zeros, expected_union_zeros);
        assert_eq!(union_zeros, &set1 | &set2);
    }
}
