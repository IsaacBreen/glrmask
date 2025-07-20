use range_set_blaze::RangeMapBlaze;
use crate::datastructures::hybrid_bitset::HybridBitset;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign};

/// A two-dimensional bitset, conceptually a map from `usize` to `HybridBitset`.
///
/// This structure uses a `RangeMapBlaze` to efficiently store `HybridBitset`s
/// for ranges of first-level indices. This is efficient when many consecutive
/// first-level indices map to the same `HybridBitset` or are empty.
///
/// An empty `HybridBitset` is never stored; if a row becomes empty, it is
/// removed from the map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HybridL2Bitset {
    /// The underlying map from usize (L1 index) to a HybridBitset (L2 indices).
    inner: RangeMapBlaze<usize, HybridBitset>,
}

impl HybridL2Bitset {
    /// Creates a new, empty `HybridL2Bitset`.
    pub fn new() -> Self {
        HybridL2Bitset {
            inner: RangeMapBlaze::new(),
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
        todo!()
    }
}

impl BitOr for &HybridL2Bitset {
    type Output = HybridL2Bitset;

    fn bitor(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl BitXor for &HybridL2Bitset {
    type Output = HybridL2Bitset;

    fn bitxor(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

// --- In-place Bitwise Operations ---

impl BitAndAssign for HybridL2Bitset {
    fn bitand_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl BitAndAssign<&HybridL2Bitset> for HybridL2Bitset {
    fn bitand_assign(&mut self, rhs: &HybridL2Bitset) {
        todo!()
    }
}

impl BitOrAssign for HybridL2Bitset {
    fn bitor_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl BitOrAssign<&HybridL2Bitset> for HybridL2Bitset {
    fn bitor_assign(&mut self, rhs: &HybridL2Bitset) {
        todo!()
    }
}

impl BitXorAssign for HybridL2Bitset {
    fn bitxor_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl BitXorAssign<&HybridL2Bitset> for HybridL2Bitset {
    fn bitxor_assign(&mut self, rhs: &HybridL2Bitset) {
        todo!()
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
        assert_eq!(intersection.iter().collect::<BTreeSet<_>>(), expected_intersection);

        // Union
        let union = &set1 | &set2;
        let expected_union: BTreeSet<(usize, usize)> = vec![
            (10, 100), (10, 101),
            (11, 101), (11, 102),
            (20, 200),
            (30, 300),
        ].into_iter().collect();
        assert_eq!(union.iter().collect::<BTreeSet<_>>(), expected_union);

        // Symmetric Difference (XOR)
        let xor = &set1 ^ &set2;
        let expected_xor: BTreeSet<(usize, usize)> = vec![
            (10, 100), (10, 101),
            (11, 101), (11, 102),
            (30, 300),
        ].into_iter().collect();
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
        assert_eq!(set1_and.iter().collect::<BTreeSet<_>>(), vec![(20, 200)].into_iter().collect());

        let mut set1_or = set1_orig.clone();
        set1_or |= &set2;
        assert_eq!(set1_or.iter().collect::<BTreeSet<_>>(), vec![(10, 100), (20, 200), (30, 300)].into_iter().collect());

        let mut set1_xor = set1_orig.clone();
        set1_xor ^= &set2;
        assert_eq!(set1_xor.iter().collect::<BTreeSet<_>>(), vec![(10, 100), (30, 300)].into_iter().collect());
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
}
