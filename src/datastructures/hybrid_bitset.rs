#![allow(dead_code)] // Allow unused code for the example

use bitvec::prelude::*;
use std::cmp::{max, min};
// use std::hash::Hasher; // Not directly used, derive(Hash) handles its needs.
use std::iter::FromIterator;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index, Sub, SubAssign};


// --- Static Booleans for Index Trait ---
static STATIC_TRUE: bool = true;
static STATIC_FALSE: bool = false;

// --- Hybrid Bitset Struct ---

/// A bitset that uses `bitvec::prelude::BitVec` as its underlying storage.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HybridBitset {
    bits: BitVec<usize, Lsb0>,
}

// --- Core Implementation (`impl HybridBitset`) ---
impl HybridBitset {
    /// Creates a new, empty HybridBitset.
    pub fn new() -> Self {
        HybridBitset {
            bits: BitVec::new(),
        }
    }

    /// Creates a new HybridBitset with all indices from 0 up to `max_index` (inclusive) set to true.
    /// If `max_index` is `usize::MAX`, an empty bitset is returned as it's impossible to represent
    /// `usize::MAX + 1` elements with a 0-indexed length.
    pub fn ones(max_index: usize) -> Self {
        if let Some(len) = max_index.checked_add(1) {
            HybridBitset {
                bits: bitvec![usize, Lsb0; 1; len],
            }
        } else {
            HybridBitset::new()
        }
    }

    /// Returns the exact number of set bits (cardinality).
    pub fn len(&self) -> usize {
        self.bits.count_ones()
    }

    /// Returns true if the bitset contains no set bits.
    pub fn is_empty(&self) -> bool {
        self.bits.not_any()
    }

    /// Checks if a specific index is set.
    /// Returns false if the index is out of bounds of the current bitvector length.
    pub fn contains(&self, index: usize) -> bool {
        self.bits.get(index).map_or(false, |bitref| *bitref)
    }

    /// Inserts an index into the set. Returns true if the index was not already present.
    /// If the index is outside the current bounds, the bitvector will be resized.
    pub fn insert(&mut self, index: usize) -> bool {
        if index >= self.bits.len() {
            self.bits.resize(index + 1, false);
            self.bits.set(index, true);
            return true;
        }
        let old_value = self.bits.replace(index, true);
        !old_value
    }

    /// Sets the bit at `index` to `value`.
    /// If `value` is true and the index is outside bounds, the bitvector resizes.
    /// If `value` is false and the index is outside bounds, no change occurs.
    pub fn set(&mut self, index: usize, value: bool) {
        if index >= self.bits.len() {
            if value {
                self.bits.resize(index + 1, false);
            } else {
                return;
            }
        }
        self.bits.set(index, value);
    }

    /// Removes an index from the set. Returns true if the index was present (i.e., set to true).
    /// If the index is outside bounds, it's considered not present, returns false.
    pub fn remove(&mut self, index: usize) -> bool {
        if index < self.bits.len() {
            let old_value = self.bits.replace(index, false);
            return old_value;
        }
        false
    }

    /// Sets all bits to false. The underlying storage capacity is retained.
    pub fn clear(&mut self) {
        self.bits.fill(false);
    }

    /// Returns an iterator over the indices of the set bits.
    /// Indices are yielded in ascending order.
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            inner_iter: self.bits.iter_ones(),
        }
    }

    /// Returns an iterator over booleans, indicating for each index from 0
    /// up to `self.domain_len() - 1` whether it's set or not.
    pub fn iter_bools(&self) -> BoolIter<'_> {
        BoolIter {
            inner_iter: self.bits.iter(),
        }
    }

    /// Returns the current capacity of the underlying BitVec.
    pub fn capacity(&self) -> usize {
        self.bits.capacity()
    }

    /// Returns the length of the underlying BitVec (the maximum index ever set + 1, or 0 if empty).
    /// This defines the domain [0, domain_len - 1] for which bits are explicitly stored.
    pub fn domain_len(&self) -> usize {
        self.bits.len()
    }
}

// --- Default Implementation ---
impl Default for HybridBitset {
    fn default() -> Self {
        Self::new()
    }
}

// --- Iterator ---
pub struct Iter<'a> {
    inner_iter: bitvec::slice::IterOnes<'a, usize, Lsb0>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner_iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner_iter.size_hint()
    }
}

impl<'a> std::iter::ExactSizeIterator for Iter<'a> {}

// Implement IntoIterator for references to HybridBitset
impl<'a> IntoIterator for &'a HybridBitset {
    type Item = usize;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Implement IntoIterator for HybridBitset by value
impl IntoIterator for HybridBitset {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<usize>;

    fn into_iter(self) -> Self::IntoIter {
        // self.bits is BitVec. It derefs to BitSlice.
        // iter_ones() is on BitSlice and returns an iterator of indices.
        // collect() will consume this iterator.
        self.bits.iter_ones().collect::<Vec<usize>>().into_iter()
    }
}

// --- Boolean Iterator ---
pub struct BoolIter<'a> {
    inner_iter: bitvec::slice::Iter<'a, usize, Lsb0>,
}

impl<'a> Iterator for BoolIter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner_iter.next().map(|bit_ref| *bit_ref)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner_iter.size_hint()
    }
}

impl<'a> std::iter::ExactSizeIterator for BoolIter<'a> {}

// Implement FromIterator for HybridBitset
impl FromIterator<usize> for HybridBitset {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let mut set = HybridBitset::new();
        for i in iter {
            set.insert(i);
        }
        set
    }
}

// --- Bitwise Operations (Creating New Sets) ---

impl<'a, 'b> BitAnd<&'b HybridBitset> for &'a HybridBitset {
    type Output = HybridBitset;

    fn bitand(self, rhs: &'b HybridBitset) -> Self::Output {
        let len1 = self.bits.len();
        let len2 = rhs.bits.len();
        let op_len = min(len1, len2);

        if op_len == 0 {
            return HybridBitset::new();
        }

        let mut result_bits = self.bits[..op_len].to_bitvec();
        result_bits &= &rhs.bits[..op_len];
        HybridBitset { bits: result_bits }
    }
}

impl<'a, 'b> BitOr<&'b HybridBitset> for &'a HybridBitset {
    type Output = HybridBitset;

    fn bitor(self, rhs: &'b HybridBitset) -> Self::Output {
        let len1 = self.bits.len();
        let len2 = rhs.bits.len();
        let max_len = max(len1, len2);

        if max_len == 0 {
            return HybridBitset::new();
        }

        let mut new_bits = self.bits.clone();
        new_bits.resize(max_len, false);

        let mut rhs_bits_padded = rhs.bits.clone();
        rhs_bits_padded.resize(max_len, false);

        new_bits |= rhs_bits_padded;
        HybridBitset { bits: new_bits }
    }
}

impl<'a, 'b> BitXor<&'b HybridBitset> for &'a HybridBitset {
    type Output = HybridBitset;

    fn bitxor(self, rhs: &'b HybridBitset) -> Self::Output {
        let len1 = self.bits.len();
        let len2 = rhs.bits.len();
        let max_len = max(len1, len2);

        if max_len == 0 {
            return HybridBitset::new();
        }

        let mut new_bits = self.bits.clone();
        new_bits.resize(max_len, false);

        let mut rhs_bits_padded = rhs.bits.clone();
        rhs_bits_padded.resize(max_len, false);

        new_bits ^= rhs_bits_padded;
        HybridBitset { bits: new_bits }
    }
}

impl<'a, 'b> Sub<&'b HybridBitset> for &'a HybridBitset {
    type Output = HybridBitset;

    fn sub(self, rhs: &'b HybridBitset) -> Self::Output {
        let self_len = self.bits.len();
        if self_len == 0 {
            return HybridBitset::new();
        }

        let mut result_bits = self.bits.clone();
        let rhs_len = rhs.bits.len();
        let op_len = min(self_len, rhs_len);

        if op_len > 0 {
            // A - B is equivalent to A & (!B)
            // let not_rhs_prefix: BitVec<_, _> = !(&rhs.bits[..op_len]);
            // result_bits[..op_len] &= &not_rhs_prefix;
        }
        HybridBitset { bits: result_bits }
    }
}

// --- In-place Bitwise Operations with References ---

impl<'b> BitAndAssign<&'b HybridBitset> for HybridBitset {
    fn bitand_assign(&mut self, rhs: &'b HybridBitset) {
        let self_len = self.bits.len();
        if self_len == 0 {
            return;
        }
        let rhs_len = rhs.bits.len();
        let op_len = min(self_len, rhs_len);

        if op_len > 0 {
            self.bits[..op_len] &= &rhs.bits[..op_len];
        }
        if self_len > op_len {
            self.bits[op_len..self_len].fill(false);
        }
    }
}

impl<'b> BitOrAssign<&'b HybridBitset> for HybridBitset {
    fn bitor_assign(&mut self, rhs: &'b HybridBitset) {
        let rhs_len = rhs.bits.len();
        if rhs_len == 0 { return; }

        if rhs_len > self.bits.len() {
            self.bits.resize(rhs_len, false);
        }
        self.bits[..rhs_len] |= &rhs.bits[..rhs_len];
    }
}

impl<'b> BitXorAssign<&'b HybridBitset> for HybridBitset {
    fn bitxor_assign(&mut self, rhs: &'b HybridBitset) {
        let rhs_len = rhs.bits.len();
        if rhs_len == 0 { return; }

        if rhs_len > self.bits.len() {
            self.bits.resize(rhs_len, false);
        }
        self.bits[..rhs_len] ^= &rhs.bits[..rhs_len];
    }
}

impl<'b> SubAssign<&'b HybridBitset> for HybridBitset {
    fn sub_assign(&mut self, rhs: &'b HybridBitset) {
        let self_len = self.bits.len();
        if self_len == 0 { return; }

        let rhs_len = rhs.bits.len();
        let op_len = min(self_len, rhs_len);

        if op_len > 0 {
            // let not_rhs_prefix: BitVec<_, _> = !(&rhs.bits[..op_len]);
            // self.bits[..op_len] &= &not_rhs_prefix;
        }
    }
}

// --- In-place Bitwise Operations (Consuming RHS) ---
impl BitAndAssign<HybridBitset> for HybridBitset {
    fn bitand_assign(&mut self, rhs: HybridBitset) {
        self.bitand_assign(&rhs);
    }
}

impl BitOrAssign<HybridBitset> for HybridBitset {
    fn bitor_assign(&mut self, rhs: HybridBitset) {
        self.bitor_assign(&rhs);
    }
}

impl BitXorAssign<HybridBitset> for HybridBitset {
    fn bitxor_assign(&mut self, rhs: HybridBitset) {
        self.bitxor_assign(&rhs);
    }
}

impl SubAssign<HybridBitset> for HybridBitset {
    fn sub_assign(&mut self, rhs: HybridBitset) {
        self.sub_assign(&rhs);
    }
}


// --- Conversions ---
impl From<HybridBitset> for BitVec<usize, Lsb0> {
    fn from(hybrid_set: HybridBitset) -> Self {
        hybrid_set.bits
    }
}

impl From<BitVec<usize, Lsb0>> for HybridBitset {
    fn from(bits: BitVec<usize, Lsb0>) -> Self {
        HybridBitset { bits }
    }
}

// --- Indexing ---
impl Index<usize> for HybridBitset {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        if self.bits[index] {
            &STATIC_TRUE
        } else {
            &STATIC_FALSE
        }
    }
}


// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet; // For comparison

    const TEST_SET_SIZE_LARGE: usize = 128;
    const TEST_SET_SIZE_SMALL: usize = 64;


    #[test]
    fn test_new_empty_len() {
        let set = HybridBitset::new();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
        assert_eq!(set.domain_len(), 0);
    }

    #[test]
    fn test_insert_basic() {
        let mut set = HybridBitset::new();
        assert!(set.insert(10));
        assert_eq!(set.domain_len(), 11);
        assert!(!set.insert(10));
        assert!(set.insert(20));
        assert_eq!(set.domain_len(), 21);
        assert_eq!(set.len(), 2);
        assert!(set.contains(10));
        assert!(set.contains(20));
        assert!(!set.contains(5));
        assert!(!set.contains(0));
    }

    #[test]
    fn test_remove_basic() {
        let mut set = HybridBitset::from_iter(vec![10, 20, 30]);
        assert_eq!(set.len(), 3);
        assert_eq!(set.domain_len(), 31);

        assert!(set.remove(20));
        assert_eq!(set.len(), 2);
        assert!(!set.contains(20));
        assert!(set.contains(10));
        assert_eq!(set.domain_len(), 31);

        assert!(!set.remove(50));
        assert_eq!(set.len(), 2);
        assert_eq!(set.domain_len(), 31);

        assert!(!set.remove(15));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_iteration() {
        let indices = vec![5, 1, 100, 42];
        let set = HybridBitset::from_iter(indices.clone());
        let collected: Vec<usize> = set.iter().collect();

        let mut expected = indices;
        expected.sort_unstable();

        assert_eq!(collected, expected);
        assert_eq!(set.iter().len(), expected.len());
    }

    #[test]
    fn test_set_method() {
        let mut set = HybridBitset::new();
        set.set(5, true);
        assert!(set.contains(5));
        assert_eq!(set.len(), 1);
        assert_eq!(set.domain_len(), 6);

        set.set(5, false);
        assert!(!set.contains(5));
        assert_eq!(set.len(), 0);
        assert_eq!(set.domain_len(), 6);

        set.set(10, true);
        assert!(set.contains(10));
        assert_eq!(set.len(), 1);
        assert_eq!(set.domain_len(), 11);

        set.set(100, false);
        assert!(!set.contains(100));
        assert_eq!(set.len(), 1);
        assert_eq!(set.domain_len(), 11);
    }


    #[test]
    fn test_set_ops() {
        let set1 = HybridBitset::from_iter(vec![1, 2, 3, 10]);
        let set2 = HybridBitset::from_iter(vec![3, 4, 5, 10]);

        let intersection = &set1 & &set2;
        assert_eq!(intersection.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![3, 10]));
        assert_eq!(intersection.domain_len(), min(set1.domain_len(), set2.domain_len()));


        let union = &set1 | &set2;
        assert_eq!(union.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![1, 2, 3, 4, 5, 10]));
        assert_eq!(union.domain_len(), max(set1.domain_len(), set2.domain_len()));


        let sym_diff = &set1 ^ &set2;
        assert_eq!(sym_diff.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![1, 2, 4, 5]));
        assert_eq!(sym_diff.domain_len(), max(set1.domain_len(), set2.domain_len()));

        let difference = &set1 - &set2;
        assert_eq!(difference.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![1, 2]));
        assert_eq!(difference.domain_len(), set1.domain_len());
    }

    #[test]
    fn test_set_ops_different_lengths() {
        let set1 = HybridBitset::from_iter(vec![1, 2, 3, 10]);
        let set2 = HybridBitset::from_iter(vec![3, 4, 5, 20]);

        let intersection = &set1 & &set2;
        assert_eq!(intersection.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![3]));
        assert_eq!(intersection.domain_len(), min(set1.domain_len(), set2.domain_len()));

        let union = &set1 | &set2;
        assert_eq!(union.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![1, 2, 3, 4, 5, 10, 20]));
        assert_eq!(union.domain_len(), max(set1.domain_len(), set2.domain_len()));

        let sym_diff = &set1 ^ &set2;
        assert_eq!(sym_diff.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![1, 2, 4, 5, 10, 20]));
        assert_eq!(sym_diff.domain_len(), max(set1.domain_len(), set2.domain_len()));

        let diff12 = &set1 - &set2;
        assert_eq!(diff12.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![1, 2, 10]));
        assert_eq!(diff12.domain_len(), set1.domain_len());

        let diff21 = &set2 - &set1;
        assert_eq!(diff21.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![4, 5, 20]));
        assert_eq!(diff21.domain_len(), set2.domain_len());
    }


    #[test]
    fn test_equality_and_hash() {
        let set1 = HybridBitset::from_iter(vec![1, 5, 10]);
        let set1_clone = HybridBitset::from_iter(vec![1, 5, 10]);
        let set2 = HybridBitset::from_iter(vec![1, 5, 11]);

        let mut set3_trailing_zeros = HybridBitset::from_iter(vec![1, 5, 10]);
        set3_trailing_zeros.bits.resize(20, false);

        assert_eq!(set1, set1_clone);
        assert_ne!(set1, set2);
        assert_eq!(set1, set3_trailing_zeros, "Sets should be equal despite different underlying BitVec lengths if trailing bits are zero");

        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_val = |s: &HybridBitset| -> u64 {
            let mut hasher = DefaultHasher::new();
            s.hash(&mut hasher);
            hasher.finish()
        };

        assert_eq!(hash_val(&set1), hash_val(&set1_clone));
        assert_ne!(hash_val(&set1), hash_val(&set2));
        assert_eq!(hash_val(&set1), hash_val(&set3_trailing_zeros), "Hashes should be equal for equivalent sets");

        let mut btree_map = std::collections::BTreeMap::new();
        btree_map.insert(set1.clone(), "set1");
        assert!(btree_map.contains_key(&set1_clone));
        assert!(btree_map.contains_key(&set3_trailing_zeros));

        btree_map.insert(set3_trailing_zeros.clone(), "set3_equivalent_to_set1");
        assert_eq!(btree_map.len(), 1, "Inserting equivalent set should replace");
        assert_eq!(btree_map.get(&set1), Some(&"set3_equivalent_to_set1"));

        btree_map.insert(set2.clone(), "set2");
        assert_eq!(btree_map.len(), 2);
    }

    #[test]
    fn test_large_index() {
        let mut set = HybridBitset::new();
        let large_idx = 1_000_000;
        set.insert(large_idx);
        set.insert(0);

        assert_eq!(set.len(), 2);
        assert_eq!(set.domain_len(), large_idx + 1);
        assert!(set.contains(0));
        assert!(set.contains(large_idx));
        assert!(!set.contains(1));
        assert!(!set.contains(large_idx - 1));

        set.remove(large_idx);
        assert_eq!(set.len(), 1);
        assert!(set.contains(0));
        assert!(!set.contains(large_idx));
        assert_eq!(set.domain_len(), large_idx + 1);
    }

    #[test]
    fn test_clear() {
        let mut set = HybridBitset::from_iter(0..TEST_SET_SIZE_LARGE);
        assert!(!set.is_empty());
        assert_eq!(set.len(), TEST_SET_SIZE_LARGE);
        let original_domain_len = set.domain_len();
        let original_capacity = set.capacity();

        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert_eq!(set.domain_len(), original_domain_len, "domain_len should be retained after clear");
        assert!(set.capacity() >= original_capacity, "capacity should be retained or grow");
        assert!(set.bits.not_any(), "All bits should be false after clear");
    }

    #[test]
    fn test_assign_ops() {
        let mut set1 = HybridBitset::from_iter(vec![1, 2, 10]);
        let set2 = HybridBitset::from_iter(vec![2, 3, 20]);
        set1 |= &set2;
        assert_eq!(set1.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 2, 3, 10, 20]));
        assert_eq!(set1.domain_len(), 21);

        let mut set3 = HybridBitset::from_iter(0..TEST_SET_SIZE_LARGE);
        let set4 = HybridBitset::from_iter( (TEST_SET_SIZE_SMALL)..TEST_SET_SIZE_LARGE + 10);
        let expected_and = (TEST_SET_SIZE_SMALL..TEST_SET_SIZE_LARGE).collect::<BTreeSet<_>>();
        set3 &= &set4;
        assert_eq!(set3.iter().collect::<BTreeSet<_>>(), expected_and);
        assert_eq!(set3.domain_len(), TEST_SET_SIZE_LARGE);

        let mut set5 = HybridBitset::from_iter(vec![1, 2, 3]);
        let set6 = HybridBitset::from_iter(vec![3, 4, 5]);
        set5 ^= &set6;
        assert_eq!(set5.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 2, 4, 5]));
        assert_eq!(set5.domain_len(), 6);

        let mut set7 = HybridBitset::from_iter(vec![1, 2, 3, 4, 5]);
        let set8 = HybridBitset::from_iter(vec![2, 4, 6]);
        set7 -= &set8;
        assert_eq!(set7.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 3, 5]));
        assert_eq!(set7.domain_len(), 6);
    }

    #[test]
    fn test_assign_ops_consuming_rhs() {
        let mut set1 = HybridBitset::from_iter(vec![1, 2, 10]);
        let set2 = HybridBitset::from_iter(vec![2, 3, 20]);
        let expected_domain_len = max(set1.domain_len(), set2.domain_len());
        set1 |= set2;
        assert_eq!(set1.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 2, 3, 10, 20]));
        assert_eq!(set1.domain_len(), expected_domain_len);

        let mut set3 = HybridBitset::from_iter(0..TEST_SET_SIZE_LARGE);
        let set4 = HybridBitset::from_iter( (TEST_SET_SIZE_SMALL)..TEST_SET_SIZE_LARGE + 10);
        let expected_and = (TEST_SET_SIZE_SMALL..TEST_SET_SIZE_LARGE).collect::<BTreeSet<_>>();
        let original_set3_domain_len = set3.domain_len();
        set3 &= set4;
        assert_eq!(set3.iter().collect::<BTreeSet<_>>(), expected_and);
        assert_eq!(set3.domain_len(), original_set3_domain_len);
    }


    #[test]
    fn test_edge_cases_ops() {
        let d1 = HybridBitset::new();
        let d3 = HybridBitset::from_iter(0..TEST_SET_SIZE_SMALL);

        assert_eq!(&d1 & &d1, d1, "E & E = E");
        assert_eq!((&d1 & &d1).domain_len(), 0);
        assert_eq!(&d1 | &d1, d1, "E | E = E");
        assert_eq!((&d1 | &d1).domain_len(), 0);
        assert_eq!(&d1 ^ &d1, d1, "E ^ E = E");
        assert_eq!((&d1 ^ &d1).domain_len(), 0);
        assert_eq!(&d1 - &d1, d1, "E - E = E");
        assert_eq!((&d1 - &d1).domain_len(), 0);

        assert_eq!(&d1 & &d3, d1, "E & NE = E");
        assert_eq!((&d1 & &d3).domain_len(), 0);
        assert_eq!(&d3 & &d1, d1, "NE & E = E");
        assert_eq!((&d3 & &d1).domain_len(), 0);


        assert_eq!(&d1 | &d3, d3, "E | NE = NE");
        assert_eq!((&d1 | &d3).domain_len(), d3.domain_len());
        assert_eq!(&d3 | &d1, d3, "NE | E = NE");
        assert_eq!((&d3 | &d1).domain_len(), d3.domain_len());


        assert_eq!(&d1 ^ &d3, d3, "E ^ NE = NE");
        assert_eq!((&d1 ^ &d3).domain_len(), d3.domain_len());
        assert_eq!(&d3 ^ &d1, d3, "NE ^ E = NE");
        assert_eq!((&d3 ^ &d1).domain_len(), d3.domain_len());

        assert_eq!(&d1 - &d3, d1, "E - NE = E");
        assert_eq!((&d1 - &d3).domain_len(), 0);
        assert_eq!(&d3 - &d1, d3, "NE - E = NE");
        assert_eq!((&d3 - &d1).domain_len(), d3.domain_len());
    }

    #[test]
    fn test_from_iterator() {
        let data = vec![10, 20, 10, 30, 20];
        let set: HybridBitset = data.into_iter().collect();

        let expected_elements: BTreeSet<usize> = vec![10, 20, 30].into_iter().collect();
        assert_eq!(set.iter().collect::<BTreeSet<_>>(), expected_elements);
        assert_eq!(set.len(), 3);
        assert_eq!(set.domain_len(), 31);
    }

    #[test]
    fn test_iter_bools() {
        let empty_set = HybridBitset::new();
        assert_eq!(empty_set.iter_bools().collect::<Vec<bool>>(), Vec::<bool>::new());
        assert_eq!(empty_set.iter_bools().len(), 0);

        let set = HybridBitset::from_iter(vec![1, 3]);
        let expected_bools = vec![false, true, false, true];
        assert_eq!(set.iter_bools().collect::<Vec<bool>>(), expected_bools);
        assert_eq!(set.iter_bools().len(), expected_bools.len());
        assert_eq!(set.domain_len(), 4);
    }

    #[test]
    fn test_ones() {
        let set_none = HybridBitset::ones(usize::MAX);
        assert!(set_none.is_empty(), "ones(usize::MAX) should be empty");
        assert_eq!(set_none.domain_len(), 0);

        let set_zero = HybridBitset::ones(0);
        assert_eq!(set_zero.len(), 1);
        assert!(set_zero.contains(0));
        assert_eq!(set_zero.domain_len(), 1);

        let set_five = HybridBitset::ones(4);
        assert_eq!(set_five.len(), 5);
        for i in 0..5 {
            assert!(set_five.contains(i));
        }
        assert!(!set_five.contains(5));
        assert_eq!(set_five.domain_len(), 5);
    }

    #[test]
    fn test_index_trait_access() {
        let set = HybridBitset::from_iter(vec![1, 3, 5]);
        assert_eq!(set[0], false);
        assert_eq!(set[1], true);
        assert_eq!(set[2], false);
        assert_eq!(set[3], true);
        assert_eq!(set[4], false);
        assert_eq!(set[5], true);
    }

    #[test]
    #[should_panic]
    fn test_index_panic_out_of_bounds() {
        let set = HybridBitset::from_iter(vec![1, 3]);
        let _ = set[4];
    }

    #[test]
    fn test_into_from_bitvec() {
        let mut bv = bitvec![usize, Lsb0; 0; 10];
        bv.set(1, true);
        bv.set(5, true);

        let hb_from_bv = HybridBitset::from(bv.clone());
        assert!(hb_from_bv.contains(1));
        assert!(hb_from_bv.contains(5));
        assert!(!hb_from_bv.contains(2));
        assert_eq!(hb_from_bv.len(), 2);
        assert_eq!(hb_from_bv.domain_len(), 10);

        let bv_from_hb: BitVec<usize, Lsb0> = hb_from_bv.into();
        assert_eq!(bv, bv_from_hb);
    }
}