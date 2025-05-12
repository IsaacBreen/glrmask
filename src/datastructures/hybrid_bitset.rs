#![allow(dead_code)] // Allow unused code for the example

use range_set_blaze::{RangeSetBlaze, SortedDisjoint};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index, IndexMut, Sub, SubAssign};
use std::iter::FromIterator; // Needed for collect

// For Into<BitVec> and From<BitVec>
use bitvec::prelude::*;


// --- The Hybrid Bitset Struct ---

#[derive(Debug, Clone)]
pub struct HybridBitset {
    inner: RangeSetBlaze<usize>,
}

// --- Core Implementation (`impl HybridBitset`) ---
impl HybridBitset {
    /// Creates a new, empty HybridBitset.
    pub fn new() -> Self {
        HybridBitset {
            inner: RangeSetBlaze::new(),
        }
    }

    /// Creates a new HybridBitset with all indices from 0 up to `max_value` (inclusive) set to true.
    pub fn ones(max_value: usize) -> Self {
        // RangeSetBlaze can efficiently represent a continuous range.
        // (0..=max_value) creates a range from 0 to max_value, inclusive.
        // If max_value is usize::MAX, this will create a RangeSetBlaze covering the entire usize range.
        HybridBitset {
            inner: (0..=max_value).collect(),
        }
    }

    /// Creates a HybridBitset from an iterator of indices.
    pub fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        HybridBitset {
            inner: iter.into_iter().collect(),
        }
    }

    /// Returns the exact number of set bits (cardinality).
    pub fn len(&self) -> usize {
        self.inner.len() as usize // RangeSetBlaze::len() returns u64
    }

    /// Returns true if the bitset contains no set bits.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Checks if a specific index is set.
    pub fn contains(&self, index: usize) -> bool {
        self.inner.contains(index)
    }

    /// Inserts an index into the set. Returns true if the index was not already present.
    pub fn insert(&mut self, index: usize) -> bool {
        self.inner.insert(index)
    }

    /// Sets the bit at `index` to `value`.
    /// Returns true if the value of the bit was changed.
    pub fn set(&mut self, index: usize, value: bool) -> bool {
        if value {
            self.insert(index) // insert returns true if newly inserted
        } else {
            self.remove(index) // remove returns true if was present
        }
    }

    /// Removes an index from the set. Returns true if the index was present.
    pub fn remove(&mut self, index: usize) -> bool {
        self.inner.remove(index)
    }

    /// Removes all elements from the set.
    pub fn clear(&mut self) {
         self.inner.clear();
    }

    /// Returns an iterator over the indices of the set bits.
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            inner_iter: self.inner.iter(),
        }
    }

    /// Returns an iterator over booleans, indicating for each index from 0
    /// up to the largest index present in the set (inclusive) whether it's set or not.
    /// If the set is empty, the iterator is empty.
    pub fn iter_bools(&self) -> BoolIter<'_> {
        if self.inner.is_empty() {
            BoolIter {
                set: &self.inner,
                current_idx: 1, // current > max makes it empty
                max_idx_to_iterate: 0,
                is_empty_set: true,
            }
        } else {
            // unwrap is safe due to is_empty check
            let max_val_in_set = self.inner.max().unwrap();
            BoolIter {
                set: &self.inner,
                current_idx: 0,
                max_idx_to_iterate: max_val_in_set,
                is_empty_set: false,
            }
        }
    }
}

// --- Default Implementation ---
impl Default for HybridBitset {
    fn default() -> Self {
        Self::new()
    }
}

// --- Iterator ---
pub struct Iter<I: SortedDisjoint<usize>> {
    inner_iter: range_set_blaze::Iter<usize, I>,
}

impl<I: SortedDisjoint<usize>> Iterator for Iter<I> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner_iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner_iter.len() as usize;
        (len, Some(len))
    }
}

impl<I: range_set_blaze::SortedDisjoint<usize>> std::iter::ExactSizeIterator for Iter<I> {}


// Implement IntoIterator for references to HybridBitset
impl IntoIterator for HybridBitset {
    type Item = usize;
    type IntoIter = Iter<HybridBitset>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Implement IntoIterator for HybridBitset by value
impl IntoIterator for HybridBitset {
     type Item = usize;
     type IntoIter = range_set_blaze::IntoIter<usize>; // This is std::vec::IntoIter<usize>

     fn into_iter(self) -> Self::IntoIter {
         self.inner.into_iter()
     }
}

// --- Boolean Iterator ---
pub struct BoolIter<'a> {
    set: &'a RangeSetBlaze<usize>,
    current_idx: usize,
    max_idx_to_iterate: usize,
    is_empty_set: bool,
}

impl<'a> Iterator for BoolIter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_empty_set {
            return None;
        }
        if self.current_idx > self.max_idx_to_iterate {
            None
        } else {
            let val_to_yield = self.set.contains(self.current_idx);
            self.current_idx += 1;
            Some(val_to_yield)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.is_empty_set {
            return (0, Some(0));
        }
        let remaining = if self.current_idx > self.max_idx_to_iterate {
            0
        } else {
            (self.max_idx_to_iterate - self.current_idx) + 1
        };
        (remaining, Some(remaining))
    }
}

impl<'a> std::iter::ExactSizeIterator for BoolIter<'a> {}


// Implement FromIterator for HybridBitset
impl FromIterator<usize> for HybridBitset {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        HybridBitset {
            inner: iter.into_iter().collect(),
        }
    }
}


// --- Bitwise Operations (Creating New Sets) ---
// RangeSetBlaze implements these operations.

impl BitAnd for &HybridBitset {
    type Output = HybridBitset;

    fn bitand(self, rhs: Self) -> Self::Output {
        HybridBitset {
            inner: &self.inner & &rhs.inner,
        }
    }
}

impl BitOr for &HybridBitset {
     type Output = HybridBitset;

    fn bitor(self, rhs: Self) -> Self::Output {
        HybridBitset {
            inner: &self.inner | &rhs.inner,
        }
    }
}

impl BitXor for &HybridBitset {
     type Output = HybridBitset;

    fn bitxor(self, rhs: Self) -> Self::Output {
        HybridBitset {
            inner: &self.inner ^ &rhs.inner,
        }
    }
}

// Set Difference (A - B or A \ B)
impl Sub for &HybridBitset {
    type Output = HybridBitset;

    fn sub(self, rhs: Self) -> Self::Output {
        HybridBitset {
            inner: &self.inner - &rhs.inner,
        }
    }
}

// --- In-place Bitwise Operations ---
// RangeSetBlaze implements these assign operations.

impl BitAndAssign for HybridBitset {
    fn bitand_assign(&mut self, rhs: Self) {
        self.inner &= rhs.inner;
    }
}

impl BitOrAssign for HybridBitset {
     fn bitor_assign(&mut self, rhs: Self) {
        self.inner |= rhs.inner;
    }
}

impl BitXorAssign for HybridBitset {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.inner ^= rhs.inner;
    }
}

impl SubAssign for HybridBitset {
    fn sub_assign(&mut self, rhs: Self) {
        self.inner -= rhs.inner;
    }
}

// --- In-place Bitwise Operations with References ---

impl BitAndAssign<&HybridBitset> for HybridBitset {
    fn bitand_assign(&mut self, rhs: &HybridBitset) {
        self.inner &= &rhs.inner;
    }
}

impl BitOrAssign<&HybridBitset> for HybridBitset {
     fn bitor_assign(&mut self, rhs: &HybridBitset) {
        self.inner |= &rhs.inner;
    }
}

impl BitXorAssign<&HybridBitset> for HybridBitset {
    fn bitxor_assign(&mut self, rhs: &HybridBitset) {
        self.inner ^= &rhs.inner;
    }
}

impl SubAssign<&HybridBitset> for HybridBitset {
    fn sub_assign(&mut self, rhs: &HybridBitset) {
        self.inner -= &rhs.inner;
    }
}


// --- Equality, Hashing, Ordering ---

impl PartialEq for HybridBitset {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for HybridBitset {}

impl Hash for HybridBitset {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl PartialOrd for HybridBitset {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}

impl Ord for HybridBitset {
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.cmp(&other.inner)
    }
}


// --- Conversions with BitVec ---

impl From<HybridBitset> for BitVec<usize, Lsb0> {
    /// Convert a HybridBitset into a BitVec.
    /// The BitVec will be sized to include the maximum element in the HybridBitset.
    /// If the HybridBitset is empty, an empty BitVec is returned.
    fn from(hybrid_set: HybridBitset) -> Self {
        if hybrid_set.inner.is_empty() {
            return BitVec::new();
        }
        // unwrap is safe due to is_empty check
        let max_val = hybrid_set.inner.max().unwrap();
        let mut bv = bitvec![usize, Lsb0; 0; max_val + 1];
        for val in hybrid_set.inner.iter() {
            // Safety: val is within 0..=max_val due to how max_val is determined
            // and iter() yields elements from the set.
            bv.set(val, true);
        }
        bv
    }
}

impl From<BitVec<usize, Lsb0>> for HybridBitset {
    /// Convert a BitVec into a HybridBitset.
    fn from(bitvec: BitVec<usize, Lsb0>) -> Self {
        HybridBitset {
            inner: bitvec.iter_ones().collect(),
        }
    }
}

// --- Indexing (Kept as todo! due to complexities with Output = bool) ---

impl Index<usize> for HybridBitset {
    type Output = bool;

    /// Checks if an index is present.
    /// Note: This trait conventionally returns a reference (`&Self::Output`).
    /// Returning `&bool` for a computed value is non-trivial without proxy types
    /// or static bools (which `contains` doesn't provide).
    /// `range-set-blaze` itself does not implement `Index<usize, Output=bool>`.
    /// Consider using `contains()` instead.
    fn index(&self, index: usize) -> &Self::Output {
        // self.contains(index) // This would return bool, not &bool.
        // A proper implementation would require returning a static reference e.g. `&true` or `&false`
        // or a more complex proxy type.
        // For now, matches original `todo!()`.
        todo!("Direct indexing returning &bool is complex; use .contains() or consider API change.")
    }
}

impl IndexMut<usize> for HybridBitset {
    /// Allows mutable access to a bit.
    /// Note: Similar to `Index`, `IndexMut` for `Output = bool` is complex.
    /// `range-set-blaze` does not implement `IndexMut<usize, Output=bool>`.
    /// Consider using `set()` or `insert()`/`remove()` instead.
    fn index_mut(&mut self, _index: usize) -> &mut Self::Output {
        todo!("Direct mutable indexing for bool is complex; use .set(), .insert(), or .remove().")
    }
}


// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet as StdSet; // For comparison, renamed to avoid confusion
    use std::iter::FromIterator;

    // Threshold constants are no longer used by HybridBitset itself
    const SPARSE_TO_DENSE_THRESHOLD: usize = 128; // Kept for test logic if needed, but not for HybridBitset internals
    const DENSE_TO_SPARSE_THRESHOLD: usize = 64;  // Kept for test logic if needed

    #[test]
    fn test_new_empty_len() {
        let set = HybridBitset::new();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }

    #[test]
    fn test_insert_basic() {
        let mut set = HybridBitset::new();
        assert!(set.insert(10));
        assert!(!set.insert(10)); // Already present
        assert!(set.insert(20));
        assert_eq!(set.len(), 2);
        assert!(set.contains(10));
        assert!(set.contains(20));
        assert!(!set.contains(5));
    }

     #[test]
    fn test_remove_basic() {
        let mut set = HybridBitset::from_iter(vec![10, 20, 30]);
        assert_eq!(set.len(), 3);
        assert!(set.remove(20));
        assert_eq!(set.len(), 2);
        assert!(!set.contains(20));
        assert!(set.contains(10));
        assert!(!set.remove(50)); // Not present
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_set_method() {
        let mut set = HybridBitset::new();
        assert!(set.set(10, true)); // Changed from false to true
        assert!(set.contains(10));
        assert_eq!(set.len(), 1);

        assert!(!set.set(10, true)); // No change
        assert!(set.contains(10));
        assert_eq!(set.len(), 1);

        assert!(set.set(10, false)); // Changed from true to false
        assert!(!set.contains(10));
        assert_eq!(set.len(), 0);

        assert!(!set.set(10, false)); // No change
        assert!(!set.contains(10));
        assert_eq!(set.len(), 0);

        assert!(set.set(20, true));
        assert!(set.set(30, true));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_iteration() {
        let indices = vec![5, 1, 100, 42];
        let set = HybridBitset::from_iter(indices.clone());
        let mut collected: Vec<usize> = set.iter().collect();
        collected.sort_unstable(); // range-set-blaze iter() is sorted
        let mut expected = indices;
        expected.sort_unstable();
        assert_eq!(collected, expected);
        assert_eq!(set.iter().len(), expected.len());
    }

    #[test]
    fn test_into_iter() {
        let indices = vec![5, 1, 100, 42];
        let set = HybridBitset::from_iter(indices.clone());
        let mut collected: Vec<usize> = set.into_iter().collect(); // Consumes set
        collected.sort_unstable();
        let mut expected = indices;
        expected.sort_unstable();
        assert_eq!(collected, expected);
    }


    #[test]
    fn test_set_ops() { // Combined sparse/dense tests as distinction is internal to range-set-blaze
        let set1 = HybridBitset::from_iter(vec![1, 2, 3, 10]);
        let set2 = HybridBitset::from_iter(vec![3, 4, 5, 10]);

        let intersection = &set1 & &set2;
        let union = &set1 | &set2;
        let difference = &set1 - &set2; // set1 \ set2
        let sym_diff = &set1 ^ &set2;

        assert_eq!(intersection.iter().collect::<StdSet<usize>>(), StdSet::from_iter(vec![3, 10]));
        assert_eq!(union.iter().collect::<StdSet<usize>>(), StdSet::from_iter(vec![1, 2, 3, 4, 5, 10]));
        assert_eq!(difference.iter().collect::<StdSet<usize>>(), StdSet::from_iter(vec![1, 2]));
        assert_eq!(sym_diff.iter().collect::<StdSet<usize>>(), StdSet::from_iter(vec![1, 2, 4, 5]));
    }

    #[test]
    fn test_set_ops_with_larger_sets() {
        // These numbers are arbitrary, not tied to old thresholds
        let mut set1_items: Vec<usize> = (0..150).collect();
        set1_items.extend(200..250); // Gaps
        let set1 = HybridBitset::from_iter(set1_items.clone());

        let mut set2_items: Vec<usize> = (100..220).collect();
        let set2 = HybridBitset::from_iter(set2_items.clone());

        // Intersection: (100..150) U (200..220)
        let intersection = &set1 & &set2;
        let mut expected_intersection: StdSet<usize> = (100..150).collect();
        expected_intersection.extend(200..220);
        assert_eq!(intersection.iter().collect::<StdSet<usize>>(), expected_intersection);

        // Union: (0..250)
        let union = &set1 | &set2;
        let expected_union: StdSet<usize> = (0..250).collect();
        assert_eq!(union.iter().collect::<StdSet<usize>>(), expected_union);

        // Difference: set1 - set2 = (0..100) U (220..250)
        let diff1 = &set1 - &set2;
        let mut expected_diff1: StdSet<usize> = (0..100).collect();
        expected_diff1.extend(220..250);
        assert_eq!(diff1.iter().collect::<StdSet<usize>>(), expected_diff1);

        // Symmetric Difference
        let sym_diff = &set1 ^ &set2;
        // (set1 - set2) U (set2 - set1)
        // set2 - set1 = (150..200)
        let mut expected_sym_diff = expected_diff1.clone();
        expected_sym_diff.extend(150..200);
        assert_eq!(sym_diff.iter().collect::<StdSet<usize>>(), expected_sym_diff);
    }


     #[test]
    fn test_equality_and_hash() {
        let set1 = HybridBitset::from_iter(vec![1, 5, 10, 1000]);
        let set1_clone = HybridBitset::from_iter(vec![1, 5, 10, 1000]);
        let set2 = HybridBitset::from_iter(vec![1, 5, 11, 1000]);
        let empty_set = HybridBitset::new();
        let empty_set_clone = HybridBitset::new();

        assert_eq!(set1, set1_clone);
        assert_ne!(set1, set2);
        assert_ne!(set1, empty_set);
        assert_eq!(empty_set, empty_set_clone);

        use std::collections::hash_map::DefaultHasher;
        let hash_fn = |s: &HybridBitset| -> u64 {
            let mut hasher = DefaultHasher::new();
            s.hash(&mut hasher);
            hasher.finish()
        };

        assert_eq!(hash_fn(&set1), hash_fn(&set1_clone));
        assert_ne!(hash_fn(&set1), hash_fn(&set2));
        assert_ne!(hash_fn(&set1), hash_fn(&empty_set));
        assert_eq!(hash_fn(&empty_set), hash_fn(&empty_set_clone));

        // Test in StdSet (requires Ord, Eq, Hash)
        let mut std_set = StdSet::new();
        std_set.insert(set1.clone());
        assert!(std_set.contains(&set1));
        assert!(std_set.contains(&set1_clone));

        std_set.insert(set1_clone); // Should not increase size
        assert_eq!(std_set.len(), 1);

        std_set.insert(set2.clone());
        assert_eq!(std_set.len(), 2);
        assert!(std_set.contains(&set2));

        std_set.insert(empty_set.clone());
        assert_eq!(std_set.len(), 3);
        assert!(std_set.contains(&empty_set));
    }

     #[test]
    fn test_large_index() {
        let mut set = HybridBitset::new();
        let large_idx = 1_000_000_000; // A billion
        set.insert(large_idx);
        set.insert(0);

        assert_eq!(set.len(), 2);
        assert!(set.contains(0));
        assert!(set.contains(large_idx));
        assert!(!set.contains(1));
        assert!(!set.contains(large_idx - 1));

        set.remove(large_idx);
        assert_eq!(set.len(), 1);
        assert!(set.contains(0));
        assert!(!set.contains(large_idx));
    }

     #[test]
    fn test_clear() {
        let mut set = HybridBitset::from_iter(0..200);
        assert!(!set.is_empty());
        assert_eq!(set.len(), 200);
        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);

        let mut set2 = HybridBitset::from_iter(vec![1,2,3]);
        assert!(!set2.is_empty());
        set2.clear();
        assert!(set2.is_empty());
    }

    #[test]
    fn test_assign_ops() {
        // Or Assign
        let mut set1 = HybridBitset::from_iter(vec![1, 2, 10]);
        let set2 = HybridBitset::from_iter(vec![2, 3, 20]);
        set1 |= set2; // Consumes set2
        assert_eq!(set1.iter().collect::<StdSet<_>>(), StdSet::from_iter(vec![1, 2, 3, 10, 20]));

        // And Assign
        let mut set3 = HybridBitset::from_iter(0..100);
        let set4 = HybridBitset::from_iter(50..150);
        let expected_and = (50..100).collect::<StdSet<_>>();
        set3 &= set4; // Consumes set4
        assert_eq!(set3.iter().collect::<StdSet<_>>(), expected_and);

        // Xor Assign
        let mut set5 = HybridBitset::from_iter(vec![1, 2, 3]);
        let set6 = HybridBitset::from_iter(vec![3, 4, 5]);
        set5 ^= set6; // Consumes set6
        assert_eq!(set5.iter().collect::<StdSet<_>>(), StdSet::from_iter(vec![1, 2, 4, 5]));

        // Sub Assign
        let mut set7 = HybridBitset::from_iter(vec![1, 2, 3, 4, 5]);
        let set8 = HybridBitset::from_iter(vec![2, 4, 6]);
        set7 -= set8; // Consumes set8
        assert_eq!(set7.iter().collect::<StdSet<_>>(), StdSet::from_iter(vec![1, 3, 5]));
    }

    #[test]
    fn test_assign_ops_ref() {
         // Or Assign Ref
        let mut set1 = HybridBitset::from_iter(vec![1, 2, 10]);
        let set2 = HybridBitset::from_iter(vec![2, 3, 20]);
        set1 |= &set2;
        assert_eq!(set1.iter().collect::<StdSet<_>>(), StdSet::from_iter(vec![1, 2, 3, 10, 20]));

        // And Assign Ref
        let mut set3 = HybridBitset::from_iter(0..100);
        let set4 = HybridBitset::from_iter(50..150);
        let expected_and = (50..100).collect::<StdSet<_>>();
        set3 &= &set4;
        assert_eq!(set3.iter().collect::<StdSet<_>>(), expected_and);

        // Xor Assign Ref
        let mut set5 = HybridBitset::from_iter(vec![1, 2, 3]);
        let set6 = HybridBitset::from_iter(vec![3, 4, 5]);
        set5 ^= &set6;
        assert_eq!(set5.iter().collect::<StdSet<_>>(), StdSet::from_iter(vec![1, 2, 4, 5]));

        // Sub Assign Ref
        let mut set7 = HybridBitset::from_iter(vec![1, 2, 3, 4, 5]);
        let set8 = HybridBitset::from_iter(vec![2, 4, 6]);
        set7 -= &set8;
        assert_eq!(set7.iter().collect::<StdSet<_>>(), StdSet::from_iter(vec![1, 3, 5]));
    }

    #[test]
    fn test_edge_case_ops() { // Renamed from test_dense_dense_edge_cases
        let d1 = HybridBitset::new();
        let d2 = HybridBitset::new();
        let d3 = HybridBitset::from_iter(0..100);

        assert_eq!(&d1 & &d2, d1);
        assert_eq!(&d1 | &d2, d1);
        assert_eq!(&d1 ^ &d2, d1);
        assert_eq!(&d1 - &d2, d1);

        assert_eq!(&d1 & &d3, d1);
        assert_eq!(&d3 & &d1, d1);
        assert_eq!(&d1 | &d3, d3);
        assert_eq!(&d3 | &d1, d3);
        assert_eq!(&d1 ^ &d3, d3);
        assert_eq!(&d3 ^ &d1, d3);
        assert_eq!(&d1 - &d3, d1);
        assert_eq!(&d3 - &d1, d3);

        let d4 = HybridBitset::from_iter(0..5);
        let d5 = HybridBitset::from_iter(3..10);

        let inter = &d4 & &d5; // {3, 4}
        assert_eq!(inter.iter().collect::<StdSet<_>>(), StdSet::from_iter(vec![3, 4]));

        let union_val = &d4 | &d5; // {0..10}
        assert_eq!(union_val.iter().collect::<StdSet<_>>(), (0..10).collect::<StdSet<_>>());

        let diff = &d4 - &d5; // {0, 1, 2}
        assert_eq!(diff.iter().collect::<StdSet<_>>(), StdSet::from_iter(vec![0, 1, 2]));

        let sym_diff = &d4 ^ &d5; // {0,1,2} U {5,6,7,8,9}
        assert_eq!(sym_diff.iter().collect::<StdSet<_>>(), StdSet::from_iter(vec![0,1,2,5,6,7,8,9]));
    }

     #[test]
    fn test_from_iterator_trait() {
        let data = vec![10, 20, 10, 30, 20];
        let set: HybridBitset = data.into_iter().collect();

        let expected: StdSet<usize> = vec![10, 20, 30].into_iter().collect();
        assert_eq!(set.iter().collect::<StdSet<_>>(), expected);
    }

    #[test]
    fn test_iter_bools() {
        let empty_set = HybridBitset::new();
        assert_eq!(empty_set.iter_bools().collect::<Vec<bool>>(), Vec::<bool>::new());
        assert_eq!(empty_set.iter_bools().len(), 0);

        let sparse_set = HybridBitset::from_iter(vec![1, 3]); // Max index is 3
        let expected_sparse_bools = vec![false, true, false, true];
        assert_eq!(sparse_set.iter_bools().collect::<Vec<bool>>(), expected_sparse_bools);
        assert_eq!(sparse_set.iter_bools().len(), expected_sparse_bools.len());

        let set_with_zero = HybridBitset::from_iter(vec![0, 2]); // Max index is 2
        let expected_zero_bools = vec![true, false, true];
        assert_eq!(set_with_zero.iter_bools().collect::<Vec<bool>>(), expected_zero_bools);
        assert_eq!(set_with_zero.iter_bools().len(), expected_zero_bools.len());

        let single_zero_set = HybridBitset::from_iter(vec![0]); // Max index is 0
        assert_eq!(single_zero_set.iter_bools().collect::<Vec<bool>>(), vec![true]);
        assert_eq!(single_zero_set.iter_bools().len(), 1);
    }

    #[test]
    fn test_ones() {
        let set_ones_0 = HybridBitset::ones(0); // {0}
        assert_eq!(set_ones_0.len(), 1);
        assert!(set_ones_0.contains(0));
        assert_eq!(set_ones_0.iter().collect::<Vec<_>>(), vec![0]);

        let set_ones_5 = HybridBitset::ones(5); // {0,1,2,3,4,5}
        assert_eq!(set_ones_5.len(), 6);
        for i in 0..=5 {
            assert!(set_ones_5.contains(i));
        }
        assert!(!set_ones_5.contains(6));
        assert_eq!(set_ones_5.iter().collect::<Vec<_>>(), (0..=5).collect::<Vec<_>>());

        let empty_by_ones_like_construction = HybridBitset::from_iter(0..0); // empty
        assert!(empty_by_ones_like_construction.is_empty());
    }

    #[test]
    fn test_from_into_bitvec() {
        // Empty set
        let hb_empty = HybridBitset::new();
        let bv_empty: BitVec<usize, Lsb0> = hb_empty.clone().into();
        assert!(bv_empty.is_empty());
        let hb_from_bv_empty = HybridBitset::from(bv_empty);
        assert!(hb_from_bv_empty.is_empty());
        assert_eq!(hb_empty, hb_from_bv_empty);

        // Set with some values
        let hb_orig = HybridBitset::from_iter(vec![1, 3, 5, 10]);
        let bv: BitVec<usize, Lsb0> = hb_orig.clone().into();

        assert_eq!(bv.len(), 11); // Max element is 10, so length is 10+1
        assert!(!bv[0]);
        assert!(bv[1]);
        assert!(!bv[2]);
        assert!(bv[3]);
        assert!(!bv[4]);
        assert!(bv[5]);
        assert!(!bv[6]);
        assert!(!bv[7]);
        assert!(!bv[8]);
        assert!(!bv[9]);
        assert!(bv[10]);
        assert_eq!(bv.count_ones(), 4);

        let hb_from_bv = HybridBitset::from(bv);
        assert_eq!(hb_orig, hb_from_bv);

        // BitVec with leading/trailing zeros
        let mut bv_complex = bitvec![usize, Lsb0; 0; 20]; // len 20, all 0
        bv_complex.set(2, true);
        bv_complex.set(8, true);
        bv_complex.set(15, true);
        // Effective elements {2, 8, 15}

        let hb_from_bv_complex = HybridBitset::from(bv_complex.clone());
        assert_eq!(hb_from_bv_complex.len(), 3);
        assert!(hb_from_bv_complex.contains(2));
        assert!(hb_from_bv_complex.contains(8));
        assert!(hb_from_bv_complex.contains(15));
        assert!(!hb_from_bv_complex.contains(0));

        let bv_rt: BitVec<usize, Lsb0> = hb_from_bv_complex.clone().into();
        assert_eq!(bv_rt.len(), 16); // Max element 15, so len 15+1
        assert!(bv_rt[2]);
        assert!(bv_rt[8]);
        assert!(bv_rt[15]);
        assert_eq!(bv_rt.count_ones(), 3);
    }
}