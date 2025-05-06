#![allow(dead_code)] // Allow unused code for the example

use bitvec::prelude::*;
use std::collections::BTreeSet;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index, IndexMut, Sub, SubAssign};
use std::cmp::{max, min};
use std::hash::{Hash, Hasher};
use std::iter::FromIterator; // Needed for collect into BTreeSet in tests

// --- Static Assertions Dependency (Optional but Recommended) ---
// Add `static_assertions = "1.1"` to your Cargo.toml if you want this compile-time check
// use static_assertions::const_assert;

// --- Constants for Switching Thresholds ---
// If a Sparse set grows >= this, convert to Dense
const SPARSE_TO_DENSE_THRESHOLD: usize = 128;
// If a Dense set shrinks < this, convert to Sparse
const DENSE_TO_SPARSE_THRESHOLD: usize = 64;
// Ensure hysteresis: DENSE_TO_SPARSE < SPARSE_TO_DENSE
// const_assert!(DENSE_TO_SPARSE_THRESHOLD < SPARSE_TO_DENSE_THRESHOLD); // Uncomment if using static_assertions


// --- Enum for Internal Representation ---

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
enum BitsetRepr {
    Sparse(BTreeSet<usize>),
    Dense {
        bits: BitVec<usize, Lsb0>,
        // Inclusive bounds on the number of set bits.
        // If lower == upper, the count is exact.
        // Otherwise, the exact count is unknown but within [lower, upper].
        lower_bound_count: usize,
        upper_bound_count: usize,
    },
}

// --- The Hybrid Bitset Struct ---

#[derive(Debug, Clone, Ord, PartialOrd)]
pub struct HybridBitset {
    inner: BitsetRepr,
}

// --- Core Implementation (`impl HybridBitset`) ---
impl HybridBitset {
    /// Creates a new, empty HybridBitset. Starts as Sparse.
    pub fn new() -> Self {
        // Ensure thresholds make sense at runtime if static_assertions is not used
        assert!(DENSE_TO_SPARSE_THRESHOLD < SPARSE_TO_DENSE_THRESHOLD, "Thresholds misconfigured");
        HybridBitset {
            inner: BitsetRepr::Sparse(BTreeSet::new()),
        }
    }

    /// Creates a HybridBitset from an iterator of indices.
    pub fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let mut set = HybridBitset::new();
        for index in iter {
            set.insert(index);
        }
        set
    }

    /// Returns the exact number of set bits (cardinality).
    /// May trigger a recount in the Dense variant if bounds are not exact.
    pub fn len(&self) -> usize {
        match &self.inner {
            BitsetRepr::Sparse(set) => set.len(),
            BitsetRepr::Dense { bits, lower_bound_count, upper_bound_count } => {
                if *lower_bound_count == *upper_bound_count {
                    // Bounds are exact, return the known count
                    *lower_bound_count
                } else {
                    // Bounds are not exact, recalculate
                    let exact_count = bits.count_ones();
                     // Can't do this - needs mutable reference to self
                    // *lower_bound_count = exact_count;
                    // *upper_bound_count = exact_count;
                    exact_count
                }
            }
        }
    }

    /// Returns true if the bitset contains no set bits.
    pub fn is_empty(&self) -> bool {
        match &self.inner { // Use mutable borrow to potentially update count
            BitsetRepr::Sparse(set) => set.is_empty(),
            BitsetRepr::Dense { lower_bound_count, upper_bound_count, .. } => {
                 // If bounds are exact and 0, it's empty.
                 // If lower bound is > 0, it's not empty.
                 // Otherwise, we need the exact count.
                 if *lower_bound_count == *upper_bound_count {
                     *lower_bound_count == 0
                 } else if *lower_bound_count > 0 {
                     false
                 } else {
                     // Need exact count if lower is 0 but upper > 0
                     self.len() == 0 // len() will recalculate and update bounds if needed
                 }
            }
        }
    }

    /// Checks if a specific index is set.
    pub fn contains(&self, index: usize) -> bool {
        match &self.inner {
            BitsetRepr::Sparse(set) => set.contains(&index),
            BitsetRepr::Dense { bits, .. } => {
                // Use get() for safe bounds checking
                bits.get(index).map_or(false, |bitref| *bitref)
            }
        }
    }

    /// Inserts an index into the set. Returns true if the index was not already present.
    /// May trigger a conversion from Sparse to Dense.
    pub fn insert(&mut self, index: usize) -> bool {
        let was_present;
        let mut needs_conversion_check = false;
        match &mut self.inner {
            BitsetRepr::Sparse(set) => {
                was_present = set.contains(&index);
                if !was_present {
                    set.insert(index);
                    // Check if we need to convert to Dense
                    if set.len() >= SPARSE_TO_DENSE_THRESHOLD {
                        needs_conversion_check = true; // Defer conversion until outside match
                    }
                }
            }
            BitsetRepr::Dense { bits, lower_bound_count, upper_bound_count } => {
                // Ensure the BitVec is large enough
                if index >= bits.len() {
                    // Calculate required capacity increase. Avoid excessive overallocation for single large indices.
                    let new_len = index + 1;
                    bits.resize(new_len, false);
                    // Resizing might invalidate upper bound if we didn't know the exact count
                     if *lower_bound_count != *upper_bound_count {
                         // The safest upper bound after resize is the new capacity,
                         // but adding 1 is also a valid (though potentially looser) upper bound update.
                         // Let's stick to the simple increment for consistency, len() will fix it if needed.
                         // *upper_bound_count = bits.len(); // Safest upper bound
                         *upper_bound_count = upper_bound_count.saturating_add(1);
                     }
                }

                // Check current state before setting using immutable borrow first
                let current_bit = bits.get(index).map_or(false, |b| *b);
                was_present = current_bit;

                if !was_present {
                    // Now get mutable borrow to set
                    bits.set(index, true);
                    // Increment bounds only if we actually changed the bit from 0 to 1
                    *lower_bound_count = lower_bound_count.saturating_add(1);
                    *upper_bound_count = upper_bound_count.saturating_add(1);
                }
                // No conversion check needed on insert for Dense
            }
        }

        if needs_conversion_check {
            self.convert_to_dense();
        }

        !was_present // Return true if it was newly inserted
    }

    pub fn set(&mut self, index: usize, value: bool) {
        todo!()
    }

    /// Removes an index from the set. Returns true if the index was present.
    /// May trigger a conversion from Dense to Sparse.
    pub fn remove(&mut self, index: usize) -> bool {
        let was_present;
        let mut needs_conversion_check = false;
        let mut current_upper_bound = 0; // To check threshold outside match

        match &mut self.inner {
            BitsetRepr::Sparse(set) => {
                was_present = set.remove(&index);
                // No conversion check needed on remove for Sparse
            }
            BitsetRepr::Dense { bits, lower_bound_count, upper_bound_count } => {
                // Check if index is within bounds first
                if index < bits.len() {
                     // Check current state before clearing
                    let current_bit = bits.get(index).map_or(false, |b| *b);
                    was_present = current_bit;

                    if was_present {
                        bits.set(index, false);
                        // Decrement bounds only if we actually changed the bit from 1 to 0
                        // If bounds were exact, they remain exact after decrement.
                        // If bounds were not exact, decrementing both is safe.
                        *lower_bound_count = lower_bound_count.saturating_sub(1);
                        *upper_bound_count = upper_bound_count.saturating_sub(1);

                        // Check if we should convert to Sparse
                        // Use the upper bound for the check, as it's the max possible count
                        current_upper_bound = *upper_bound_count; // Store for check outside match
                        if current_upper_bound < DENSE_TO_SPARSE_THRESHOLD {
                            needs_conversion_check = true;
                        }
                    }
                } else {
                    // Index out of bounds, definitely wasn't present
                    was_present = false;
                }
            }
        }

        if needs_conversion_check {
             // Recalculate exact count before potentially converting
             // This requires getting a mutable borrow again
             if let BitsetRepr::Dense { bits, lower_bound_count, upper_bound_count } = &mut self.inner {
                 if *lower_bound_count != *upper_bound_count || *upper_bound_count < DENSE_TO_SPARSE_THRESHOLD {
                     let exact_count = bits.count_ones();
                     *lower_bound_count = exact_count;
                     *upper_bound_count = exact_count;
                     if exact_count < DENSE_TO_SPARSE_THRESHOLD {
                         self.convert_to_sparse();
                     }
                 }
             }
        }

        was_present
    }

    /// Removes all elements from the set.
    pub fn clear(&mut self) {
         // Easiest way is to replace with a new empty set
         *self = HybridBitset::new();
    }

    /// Returns an iterator over the indices of the set bits.
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            inner: match &self.inner {
                BitsetRepr::Sparse(set) => IterInner::Sparse(set.iter()),
                BitsetRepr::Dense { bits, .. } => IterInner::Dense(bits.iter_ones()),
            },
        }
    }

    /// Returns an iterator over booleans, indicating for each index from 0
    /// up to a certain limit whether it's set or not.
    /// For a Dense set, the limit is its current capacity (`bits.len() - 1`),
    /// meaning it yields `bits.len()` booleans.
    /// For a Sparse set, the limit is the largest index present in the set.
    /// If the set is empty (either Sparse or Dense), the iterator is empty.
    pub fn iter_bools(&self) -> BoolIter<'_> {
        match &self.inner {
            BitsetRepr::Sparse(set) => {
                if set.is_empty() {
                    // For an empty set, create an iterator that yields nothing.
                    // current_idx starts beyond max_idx_to_iterate.
                    BoolIter {
                        inner: BoolIterInner::Sparse {
                            set,
                            current_idx: 1,
                            max_idx_to_iterate: 0,
                        }
                    }
                } else {
                    // Find the maximum element to define the iteration range.
                    // BTreeSet is sorted, so last() is efficient.
                    // unwrap() is safe here because we've checked is_empty().
                    let max_val_in_set = set.last().copied().unwrap();
                    BoolIter {
                        inner: BoolIterInner::Sparse {
                            set,
                            current_idx: 0,
                            max_idx_to_iterate: max_val_in_set,
                        }
                    }
                }
            }
            BitsetRepr::Dense { bits, .. } => {
                // bits.iter() yields bool values for each position in the bitvector.
                // The length of this iterator is bits.len().
                BoolIter {
                    inner: BoolIterInner::Dense(bits.iter()),
                }
            }
        }
    }


    // --- Helper: Force conversion to Dense ---
    fn ensure_dense(&mut self) {
        if matches!(self.inner, BitsetRepr::Sparse(_)) {
            self.convert_to_dense();
        }
    }

    // --- Helper: Force conversion to Sparse ---
    // fn ensure_sparse(&mut self) { // Less commonly needed, but could be added
    //     if matches!(self.inner, BitsetRepr::Dense { .. }) {
    //         self.convert_to_sparse();
    //     }
    // }


    // --- Helper: Convert Sparse -> Dense ---
    fn convert_to_dense(&mut self) {
        if let BitsetRepr::Sparse(set) = &self.inner {
            let count = set.len();
            if count == 0 {
                // Handle empty set case
                 self.inner = BitsetRepr::Dense {
                    bits: BitVec::new(),
                    lower_bound_count: 0,
                    upper_bound_count: 0,
                };
                return;
            }

            // Find the maximum index to determine BitVec size
            // Handle the case where the set might be non-empty but only contains 0
            let max_index = set.iter().max().copied().unwrap_or(0);
            let mut bits = bitvec![usize, Lsb0; 0; max_index + 1];

            for &index in set {
                // Safety: index is guaranteed to be <= max_index here
                bits.set(index, true);
            }

            self.inner = BitsetRepr::Dense {
                bits,
                lower_bound_count: count, // Exact count known after conversion
                upper_bound_count: count,
            };
        }
        // If already Dense, do nothing
    }

    // --- Helper: Convert Dense -> Sparse ---
    fn convert_to_sparse(&mut self) {
        if let BitsetRepr::Dense { bits, lower_bound_count, .. } = &self.inner {
            // Use lower_bound as a hint for capacity, though exact count is better if available
            let capacity_hint = *lower_bound_count;
            let mut set = BTreeSet::new();
            for index in bits.iter_ones() {
                set.insert(index);
            }
            self.inner = BitsetRepr::Sparse(set);
        }
        // If already Sparse, do nothing
    }

     // --- Helper: Recalculate count for Dense and update bounds ---
    fn recalculate_dense_count(&mut self) {
        if let BitsetRepr::Dense { bits, lower_bound_count, upper_bound_count } = &mut self.inner {
            if *lower_bound_count != *upper_bound_count { // Only recalculate if needed
                let exact_count = bits.count_ones();
                *lower_bound_count = exact_count;
                *upper_bound_count = exact_count;
            }
        }
    }

    // --- Helper: Check and potentially convert after an operation ---
    // This should be called after operations that might change the count significantly.
    fn check_representation(&mut self) {
        match &mut self.inner {
            BitsetRepr::Sparse(set) => {
                if set.len() >= SPARSE_TO_DENSE_THRESHOLD {
                    self.convert_to_dense();
                }
            }
            BitsetRepr::Dense { lower_bound_count, upper_bound_count, bits } => {
                 // Use upper bound for quick check, but recalculate if near threshold
                 if *upper_bound_count < DENSE_TO_SPARSE_THRESHOLD {
                     // Recalculate to be sure before converting
                     let exact_count = bits.count_ones();
                     *lower_bound_count = exact_count;
                     *upper_bound_count = exact_count;
                     if exact_count < DENSE_TO_SPARSE_THRESHOLD {
                         self.convert_to_sparse();
                     }
                 }
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

// Need an inner enum for the iterator state
enum IterInner<'a> {
    Sparse(std::collections::btree_set::Iter<'a, usize>),
    Dense(bitvec::slice::IterOnes<'a, usize, Lsb0>),
}

pub struct Iter<'a> {
    inner: IterInner<'a>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            IterInner::Sparse(iter) => iter.next().copied(), // Need copied() because BTreeSet::Iter yields &usize
            IterInner::Dense(iter) => iter.next(),
        }
    }

    // Optional: Provide size_hint if possible
    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            IterInner::Sparse(iter) => iter.size_hint(), // BTreeSet::Iter provides exact size hint
            IterInner::Dense(iter) => iter.size_hint(), // bitvec::IterOnes also provides exact size hint
        }
    }
}

// Implement ExactSizeIterator if the underlying iterators support it
impl<'a> std::iter::ExactSizeIterator for Iter<'a> {}


// Implement IntoIterator for references to HybridBitset
impl<'a> IntoIterator for &'a HybridBitset {
    type Item = usize;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Optional: Implement IntoIterator for HybridBitset by value
// This consumes the bitset.
impl IntoIterator for HybridBitset {
     type Item = usize;
     // This requires owning iterators or collecting.
     // Let's return a Vec for simplicity here, though a custom owning iterator is possible.
     type IntoIter = std::vec::IntoIter<usize>;

     fn into_iter(self) -> Self::IntoIter {
         let collected: Vec<usize> = match self.inner {
             BitsetRepr::Sparse(set) => set.into_iter().collect(),
             BitsetRepr::Dense { bits, .. } => {
                 // Use iter_ones which is efficient
                 bits.iter_ones().collect()
             }
         };
         collected.into_iter()
     }
}

// --- Boolean Iterator ---

enum BoolIterInner<'a> {
    Sparse {
        set: &'a BTreeSet<usize>,
        current_idx: usize,
        // The iterator will yield values for indices from 0 up to max_idx_to_iterate (inclusive).
        max_idx_to_iterate: usize,
    },
    Dense(bitvec::slice::Iter<'a, usize, Lsb0>), // This iterator from bitvec yields bool
}

pub struct BoolIter<'a> {
    inner: BoolIterInner<'a>,
}

impl<'a> Iterator for BoolIter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            BoolIterInner::Sparse { set, current_idx, max_idx_to_iterate } => {
                if *current_idx > *max_idx_to_iterate {
                    None
                } else {
                    let val_to_yield = set.contains(current_idx);
                    *current_idx += 1;
                    Some(val_to_yield)
                }
            }
            BoolIterInner::Dense(iter) => iter.next(), // bitvec::slice::Iter yields bool directly
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            BoolIterInner::Sparse { current_idx, max_idx_to_iterate, .. } => {
                let remaining = if *current_idx > *max_idx_to_iterate {
                    0
                } else {
                    (*max_idx_to_iterate - *current_idx) + 1
                };
                (remaining, Some(remaining))
            }
            BoolIterInner::Dense(iter) => iter.size_hint(), // bitvec::slice::Iter provides exact size hint
        }
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

impl BitAnd for &HybridBitset {
    type Output = HybridBitset;

    fn bitand(self, rhs: Self) -> Self::Output {
        match (&self.inner, &rhs.inner) {
            // Sparse & Sparse
            (BitsetRepr::Sparse(set1), BitsetRepr::Sparse(set2)) => {
                // Optimize for the smaller set driving the intersection
                let (smaller, larger) = if set1.len() < set2.len() { (set1, set2) } else { (set2, set1) };
                let mut result_set = BTreeSet::new();
                for &item in smaller {
                    if larger.contains(&item) {
                        result_set.insert(item);
                    }
                }
                // Alternative: let result_set: BTreeSet<usize> = set1.intersection(set2).copied().collect();
                let mut result = HybridBitset { inner: BitsetRepr::Sparse(result_set) };
                result.check_representation(); // Check if result should be Dense
                result
            }
            // Dense & Dense
            (BitsetRepr::Dense { bits: bits1, .. }, BitsetRepr::Dense { bits: bits2, .. }) => {
                let len1 = bits1.len();
                let len2 = bits2.len();
                // Result length is the minimum of the two lengths for intersection
                let min_len = min(len1, len2);
                if min_len == 0 {
                    // Intersection with empty is empty
                    return HybridBitset::new();
                }

                // Take a slice of the relevant parts and perform AND
                let slice1: BitVec = bits1[..min_len].into();
                let slice2: BitVec = bits2[..min_len].into();
                let result_bits = slice1 & slice2; // This creates a new BitVec

                let exact_count = result_bits.count_ones();
                let mut result = HybridBitset {
                    inner: BitsetRepr::Dense {
                        bits: result_bits,
                        lower_bound_count: exact_count,
                        upper_bound_count: exact_count,
                    }
                };
                result.check_representation(); // Check if result should be Sparse
                result
            }
            // Mixed: Convert Sparse to Dense temporarily is often easiest
            (BitsetRepr::Sparse(set1), BitsetRepr::Dense { bits: bits2, .. }) => {
                // Optimization: Iterate sparse set and check against dense set
                let mut result_set = BTreeSet::new(); // Capacity hint
                for &item in set1 {
                    if bits2.get(item).map_or(false, |b| *b) {
                        result_set.insert(item);
                    }
                }
                 let mut result = HybridBitset { inner: BitsetRepr::Sparse(result_set) };
                 result.check_representation(); // Check if result should be Dense (unlikely)
                 result
            }
            (BitsetRepr::Dense { bits: bits1, .. }, BitsetRepr::Sparse(set2)) => {
                 // Symmetric to the above case
                 let mut result_set = BTreeSet::new();
                 for &item in set2 {
                     if bits1.get(item).map_or(false, |b| *b) {
                         result_set.insert(item);
                     }
                 }
                 let mut result = HybridBitset { inner: BitsetRepr::Sparse(result_set) };
                 result.check_representation();
                 result
            }
        }
    }
}

impl BitOr for &HybridBitset {
     type Output = HybridBitset;

    fn bitor(self, rhs: Self) -> Self::Output {
         match (&self.inner, &rhs.inner) {
            // Sparse | Sparse
            (BitsetRepr::Sparse(set1), BitsetRepr::Sparse(set2)) => {
                // Optimization: clone larger, extend with smaller
                let (larger, smaller) = if set1.len() >= set2.len() { (set1, set2) } else { (set2, set1) };
                let mut result_set = larger.clone();
                result_set.extend(smaller.iter().copied());
                // Alternative: let result_set: BTreeSet<usize> = set1.union(set2).copied().collect();
                let mut result = HybridBitset { inner: BitsetRepr::Sparse(result_set) };
                result.check_representation();
                result
            }
            // Dense | Dense
            (BitsetRepr::Dense { bits: bits1, .. }, BitsetRepr::Dense { bits: bits2, .. }) => {
                let len1 = bits1.len();
                let len2 = bits2.len();
                let max_len = max(len1, len2);

                // Clone the longer one, resize if necessary, then OR with the shorter one
                let mut result_bits;
                let other_bits;

                if len1 >= len2 {
                    result_bits = bits1.clone();
                    other_bits = bits2;
                    // result_bits already has max_len (or more)
                } else {
                    result_bits = bits2.clone();
                    other_bits = bits1;
                    // result_bits already has max_len (or more)
                }

                // OR the common prefix. The rest of result_bits remains unchanged (which is correct for OR).
                let min_len = min(len1, len2);
                if min_len > 0 {
                    result_bits[..min_len] |= &other_bits[..min_len];
                }

                // Estimate bounds or recalculate. Recalculate is simpler.
                let exact_count = result_bits.count_ones();
                let mut result = HybridBitset {
                    inner: BitsetRepr::Dense {
                        bits: result_bits,
                        lower_bound_count: exact_count,
                        upper_bound_count: exact_count,
                    }
                };
                result.check_representation(); // Check if result should be Sparse (e.g., union of small dense sets)
                result
            }
            // Mixed: Convert Sparse to Dense is usually required here
            (BitsetRepr::Sparse(_), BitsetRepr::Dense { .. }) => {
                let mut lhs_clone = self.clone();
                lhs_clone.ensure_dense(); // Convert lhs to Dense
                // Now it's Dense | Dense
                // Need mutable borrow for ensure_dense, so use reference for op
                &lhs_clone | rhs
            }
            (BitsetRepr::Dense { .. }, BitsetRepr::Sparse(_)) => {
                let mut rhs_clone = rhs.clone();
                rhs_clone.ensure_dense();
                self | &rhs_clone
            }
        }
    }
}

impl BitXor for &HybridBitset {
     type Output = HybridBitset;

    fn bitxor(self, rhs: Self) -> Self::Output {
         match (&self.inner, &rhs.inner) {
            // Sparse ^ Sparse
            (BitsetRepr::Sparse(set1), BitsetRepr::Sparse(set2)) => {
                let result_set: BTreeSet<usize> = set1.symmetric_difference(set2).copied().collect();
                let mut result = HybridBitset { inner: BitsetRepr::Sparse(result_set) };
                result.check_representation();
                result
            }
            // Dense ^ Dense
            (BitsetRepr::Dense { bits: bits1, .. }, BitsetRepr::Dense { bits: bits2, .. }) => {
                 let len1 = bits1.len();
                let len2 = bits2.len();
                let max_len = max(len1, len2);

                // Clone the longer one, resize if necessary, then XOR with the shorter one
                let mut result_bits;
                let other_bits;

                if len1 >= len2 {
                    result_bits = bits1.clone();
                    other_bits = bits2;
                    result_bits.resize(max_len, false); // Ensure result has max_len
                } else {
                    result_bits = bits2.clone();
                    other_bits = bits1;
                    result_bits.resize(max_len, false); // Ensure result has max_len
                }

                // XOR the common prefix
                let min_len = min(len1, len2);
                 if min_len > 0 {
                    result_bits[..min_len] ^= &other_bits[..min_len];
                 }
                 // The bits beyond min_len in the longer original vector remain unchanged,
                 // which is correct for XOR (since the shorter vector has implicit zeros there).

                let exact_count = result_bits.count_ones();
                let mut result = HybridBitset {
                    inner: BitsetRepr::Dense {
                        bits: result_bits,
                        lower_bound_count: exact_count,
                        upper_bound_count: exact_count,
                    }
                };
                result.check_representation(); // Result size could be small or large
                result
            }
            // Mixed: Convert Sparse to Dense
            (BitsetRepr::Sparse(_), BitsetRepr::Dense { .. }) => {
                let mut lhs_clone = self.clone();
                lhs_clone.ensure_dense();
                &lhs_clone ^ rhs
            }
            (BitsetRepr::Dense { .. }, BitsetRepr::Sparse(_)) => {
                let mut rhs_clone = rhs.clone();
                rhs_clone.ensure_dense();
                self ^ &rhs_clone
            }
        }
    }
}

// Set Difference (A - B or A \ B)
impl Sub for &HybridBitset {
    type Output = HybridBitset;

    // Computes self - rhs (elements in self but not in rhs)
    fn sub(self, rhs: Self) -> Self::Output {
         match (&self.inner, &rhs.inner) {
            // Sparse - Sparse
            (BitsetRepr::Sparse(set1), BitsetRepr::Sparse(set2)) => {
                let result_set: BTreeSet<usize> = set1.difference(set2).copied().collect();
                // Result can only be sparse or stay sparse (cannot grow)
                HybridBitset { inner: BitsetRepr::Sparse(result_set) }
            }
            // Dense - Dense ( A & !B )
            (BitsetRepr::Dense { bits: bits1, .. }, BitsetRepr::Dense { bits: bits2, .. }) => {
                let len1 = bits1.len();
                let len2 = bits2.len();

                // Result size is bounded by bits1's length. Start with a clone of bits1.
                let mut result_bits = bits1.clone();

                // We only need to modify bits in result_bits where bits2 has a '1'.
                // Iterate up to the minimum length where both might have set bits.
                let op_len = min(len1, len2);
                if op_len > 0 {
                    // Create a temporary negated view/copy of bits2's prefix
                    let mut negated_rhs_prefix = !bits2[..op_len].to_bitvec(); // Copy slice

                    // Apply the difference: result = result & !rhs_prefix
                    result_bits[..op_len] &= negated_rhs_prefix;
                }
                // Bits in result_bits beyond op_len (if len1 > len2) remain unchanged, which is correct.

                let exact_count = result_bits.count_ones();
                let mut result = HybridBitset {
                    inner: BitsetRepr::Dense {
                        bits: result_bits,
                        lower_bound_count: exact_count,
                        upper_bound_count: exact_count,
                    }
                };
                result.check_representation(); // Check if result should be Sparse
                result
            }
            // Dense - Sparse
            (BitsetRepr::Dense { bits, .. }, BitsetRepr::Sparse(set2)) => {
                // More efficient to clone dense and remove elements from sparse
                let mut result_bits = bits.clone();
                let mut count_decreased = false; // Track if count might have changed
                for &index in set2 {
                    if index < result_bits.len() {
                        // Use set(.., false) which returns previous state
                        if result_bits.replace(index, false) {
                             count_decreased = true; // A bit was actually cleared
                        }
                    }
                }

                // If count decreased, bounds are invalidated. Recalculate.
                // Otherwise, original bounds are still valid (though maybe not tight).
                // Always recalculating is simpler.
                let exact_count = result_bits.count_ones();
                let mut result = HybridBitset {
                    inner: BitsetRepr::Dense {
                        bits: result_bits,
                        lower_bound_count: exact_count,
                        upper_bound_count: exact_count,
                    }
                };
                result.check_representation();
                result
            }
             // Sparse - Dense
            (BitsetRepr::Sparse(set1), BitsetRepr::Dense { bits: bits2, .. }) => {
                 // Iterate sparse set, keep elements not present in dense set
                 let mut result_set = BTreeSet::new();
                 for &index in set1 {
                     // Keep index if it's NOT in bits2 (either out of bounds or bit is false)
                     if bits2.get(index).map_or(true, |b| !*b) {
                         result_set.insert(index);
                     }
                 }
                 // Result can only be sparse or stay sparse
                 HybridBitset { inner: BitsetRepr::Sparse(result_set) }
            }
        }
    }
}

// --- In-place Bitwise Operations ---

impl BitAndAssign for HybridBitset {
    fn bitand_assign(&mut self, rhs: Self) {
        // Avoid clone if possible, but logic is complex.
        // Easiest is often to calculate the result and assign back.
        // Need to borrow rhs immutably for the operation.
        *self = &*self & &rhs; // Use the non-assign version
    }
}

impl BitOrAssign for HybridBitset {
     fn bitor_assign(&mut self, rhs: Self) {
        // Optimization: If self is Dense, rhs is Sparse, can be faster
        match (&mut self.inner, &rhs.inner) {
            (BitsetRepr::Dense { bits, lower_bound_count, upper_bound_count }, BitsetRepr::Sparse(set2)) => {
                let mut new_bits_set = 0;
                // let original_len = bits.len(); // Not needed for logic
                for &index in set2 {
                    if index >= bits.len() {
                        bits.resize(index + 1, false);
                    }
                    if !bits[index] { // Check before setting
                        bits.set(index, true);
                        new_bits_set += 1;
                    }
                }
                if new_bits_set > 0 {
                    // Bounds updated approximately
                    *lower_bound_count = lower_bound_count.saturating_add(new_bits_set);
                    *upper_bound_count = upper_bound_count.saturating_add(new_bits_set);
                    // If resize happened, upper bound might be too low, recalculate?
                    // For simplicity, let's assume len() will fix it later if needed.
                }
                // No representation check needed (usually grows)
                return; // Done with optimized path
            }
            _ => {
                 // Fallback to default implementation
                 *self = &*self | &rhs;
            }
        }
    }
}

impl BitXorAssign for HybridBitset {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = &*self ^ &rhs;
    }
}

impl SubAssign for HybridBitset {
    fn sub_assign(&mut self, rhs: Self) {
         // Optimization: If self is Dense, rhs is Sparse, can be faster
        match (&mut self.inner, &rhs.inner) {
            (BitsetRepr::Dense { bits, lower_bound_count, upper_bound_count }, BitsetRepr::Sparse(set2)) => {
                let mut bits_cleared = 0;
                for &index in set2 {
                    if index < bits.len() {
                        if bits.replace(index, false) { // Returns previous value
                            bits_cleared += 1;
                        }
                    }
                }
                if bits_cleared > 0 {
                    // Update bounds approximately
                    *lower_bound_count = lower_bound_count.saturating_sub(bits_cleared);
                    *upper_bound_count = upper_bound_count.saturating_sub(bits_cleared);
                    // Check representation
                    self.check_representation();
                }
                return; // Done with optimized path
            }
             _ => {
                 // Fallback to default implementation
                 *self = &*self - &rhs;
            }
        }
    }
}

// --- In-place Bitwise Operations with References ---

impl BitAndAssign<&HybridBitset> for HybridBitset {
    fn bitand_assign(&mut self, rhs: &HybridBitset) {
        // Easiest is often to calculate the result and assign back.
        *self = &*self & rhs; // Use the non-assign version with references
    }
}

impl BitOrAssign<&HybridBitset> for HybridBitset {
     fn bitor_assign(&mut self, rhs: &HybridBitset) {
        // Optimization: If self is Dense, rhs is Sparse, can be faster
        match (&mut self.inner, &rhs.inner) {
            (BitsetRepr::Dense { bits, lower_bound_count, upper_bound_count }, BitsetRepr::Sparse(set2)) => {
                let mut new_bits_set = 0;
                // let original_len = bits.len(); // Not needed for logic
                for &index in set2 {
                    if index >= bits.len() {
                        bits.resize(index + 1, false);
                    }
                    if !bits[index] { // Check before setting
                        bits.set(index, true);
                        new_bits_set += 1;
                    }
                }
                if new_bits_set > 0 {
                    // Bounds updated approximately
                    *lower_bound_count = lower_bound_count.saturating_add(new_bits_set);
                    *upper_bound_count = upper_bound_count.saturating_add(new_bits_set);
                    // If resize happened, upper bound might be too low, recalculate?
                    // For simplicity, let's assume len() will fix it later if needed.
                }
                // No representation check needed (usually grows)
                return; // Done with optimized path
            }
            _ => {
                 // Fallback to default implementation
                 *self = &*self | rhs;
            }
        }
    }
}

impl BitXorAssign<&HybridBitset> for HybridBitset {
    fn bitxor_assign(&mut self, rhs: &HybridBitset) {
        *self = &*self ^ rhs;
    }
}

impl SubAssign<&HybridBitset> for HybridBitset {
    fn sub_assign(&mut self, rhs: &HybridBitset) {
         // Optimization: If self is Dense, rhs is Sparse, can be faster
        match (&mut self.inner, &rhs.inner) {
            (BitsetRepr::Dense { bits, lower_bound_count, upper_bound_count }, BitsetRepr::Sparse(set2)) => {
                let mut bits_cleared = 0;
                for &index in set2 {
                    if index < bits.len() {
                        if bits.replace(index, false) { // Returns previous value
                            bits_cleared += 1;
                        }
                    }
                }
                if bits_cleared > 0 {
                    // Update bounds approximately
                    *lower_bound_count = lower_bound_count.saturating_sub(bits_cleared);
                    *upper_bound_count = upper_bound_count.saturating_sub(bits_cleared);
                    // Check representation
                    self.check_representation();
                }
                return; // Done with optimized path
            }
             _ => {
                 // Fallback to default implementation
                 *self = &*self - rhs;
            }
        }
    }
}


// --- Equality and Hashing ---
// Note: Equality must be independent of the internal representation.

impl PartialEq for HybridBitset {
    fn eq(&self, other: &Self) -> bool {
        // The most reliable way is to iterate and compare elements,
        // but this can be slow. Let's try optimizing common cases.
        match (&self.inner, &other.inner) {
            (BitsetRepr::Sparse(s1), BitsetRepr::Sparse(s2)) => s1 == s2,
            (BitsetRepr::Dense { bits: b1, lower_bound_count: lc1, upper_bound_count: uc1 },
             BitsetRepr::Dense { bits: b2, lower_bound_count: lc2, upper_bound_count: uc2 }) => {
                // Quick check: if counts are known exact and differ, they aren't equal
                if lc1 == uc1 && lc2 == uc2 && lc1 != lc2 {
                    return false;
                }
                // Compare underlying bitvecs, considering trailing zeros implicitly
                let len1 = b1.len();
                let len2 = b2.len();
                let min_len = min(len1, len2);

                // Compare the common prefix
                if min_len > 0 && b1[..min_len] != b2[..min_len] {
                    return false;
                }

                // Check trailing parts - they must be all zeros
                if len1 > len2 {
                    if b1[min_len..].any() { return false; }
                } else if len2 > len1 {
                    if b2[min_len..].any() { return false; }
                }
                // If we reach here, they are equal
                true
            }
            // Mixed case: Use the general iterator comparison
            _ => {
                // Optimization: Check rough size estimates first if available without recalc?
                // let size_hint1 = self.iter().size_hint();
                // let size_hint2 = other.iter().size_hint();
                // if size_hint1.1.is_some() && size_hint2.1.is_some() && size_hint1.1 != size_hint2.1 {
                //     return false; // If exact sizes known and differ
                // }
                // Fallback to iterating both and comparing elements
                let mut iter1 = self.iter();
                let mut iter2 = other.iter();
                loop {
                    match (iter1.next(), iter2.next()) {
                        (Some(v1), Some(v2)) => if v1 != v2 { return false; }, // Elements differ
                        (None, None) => return true, // Both iterators exhausted simultaneously
                        _ => return false, // One iterator exhausted before the other (lengths differ)
                    }
                }
            }
        }
    }
}

impl Eq for HybridBitset {}

// Hashing must also be representation-independent.
// Hash the elements in a defined order (sorted).
impl Hash for HybridBitset {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Collect elements and sort them to ensure consistent hash order
        // This is potentially expensive for large dense sets.
        // Alternative: Hash based on the dense representation if possible?
        // But that requires converting sparse to dense just for hashing.
        // Sticking to sorted element hashing for correctness.
        let mut elements: Vec<usize> = self.iter().collect();
        elements.sort_unstable(); // Use unstable sort for performance

        // Hash the number of elements first (important!)
        elements.len().hash(state);
        // Then hash each element in order
        for element in elements {
            element.hash(state);
        }
    }
}

impl Into<BitVec> for HybridBitset {
    /// Convert a HybridBitset into a BitVec
    fn into(self) -> BitVec {
        todo!()
    }
}

impl From<BitVec> for HybridBitset {
    // Convert a BitVec into a HybridBitset
    fn from(bitvec: BitVec) -> Self {
        todo!()
    }
}

impl Index<usize> for HybridBitset {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        todo!()
    }
}

impl IndexMut<usize> for HybridBitset {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        todo!()
    }
}


// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet; // For comparison
    use std::iter::FromIterator; // Ensure FromIterator is in scope

    #[test]
    fn test_new_empty_len() {
        let mut set = HybridBitset::new();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
        assert!(matches!(set.inner, BitsetRepr::Sparse(_)));
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
        assert!(matches!(set.inner, BitsetRepr::Sparse(_)));
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
         assert!(matches!(set.inner, BitsetRepr::Sparse(_)));
    }

    #[test]
    fn test_iteration() {
        let indices = vec![5, 1, 100, 42];
        let set = HybridBitset::from_iter(indices.clone());
        let mut collected: Vec<usize> = set.iter().collect();
        collected.sort_unstable();
        let mut expected = indices;
        expected.sort_unstable();
        assert_eq!(collected, expected);
        assert_eq!(set.iter().len(), expected.len()); // Check ExactSizeIterator impl
    }

    #[test]
    fn test_sparse_to_dense_conversion() {
        let mut set = HybridBitset::new();
        // Insert just under the threshold
        for i in 0..(SPARSE_TO_DENSE_THRESHOLD - 1) {
            set.insert(i * 2); // Spread them out a bit
        }
        assert_eq!(set.len(), SPARSE_TO_DENSE_THRESHOLD - 1);
        assert!(matches!(set.inner, BitsetRepr::Sparse(_)), "Should be Sparse before threshold");

        // Insert one more to trigger conversion
        set.insert(SPARSE_TO_DENSE_THRESHOLD * 2);
        assert_eq!(set.len(), SPARSE_TO_DENSE_THRESHOLD);
        assert!(matches!(set.inner, BitsetRepr::Dense { .. }), "Should be Dense after threshold");

        // Check contains works on Dense
        assert!(set.contains(0));
        assert!(set.contains(2));
        assert!(set.contains(SPARSE_TO_DENSE_THRESHOLD * 2));
        assert!(!set.contains(1));

        // Check bounds are exact after conversion
         if let BitsetRepr::Dense { lower_bound_count, upper_bound_count, .. } = &set.inner {
             assert_eq!(*lower_bound_count, SPARSE_TO_DENSE_THRESHOLD);
             assert_eq!(*upper_bound_count, SPARSE_TO_DENSE_THRESHOLD);
         }
    }

     #[test]
    fn test_dense_to_sparse_conversion() {
        let mut set = HybridBitset::new();
        // Force Dense by inserting enough elements
        for i in 0..SPARSE_TO_DENSE_THRESHOLD {
            set.insert(i);
        }
        assert!(matches!(set.inner, BitsetRepr::Dense { .. }), "Should be Dense initially");
        assert_eq!(set.len(), SPARSE_TO_DENSE_THRESHOLD);


        // Remove elements until just above the lower threshold
        for i in (DENSE_TO_SPARSE_THRESHOLD..SPARSE_TO_DENSE_THRESHOLD).rev() {
             assert!(set.remove(i));
        }
        // At this point, len() might not have been called, bounds might be inexact
        // Let's call len() to force recalculation if needed before checking state
        assert_eq!(set.len(), DENSE_TO_SPARSE_THRESHOLD);
        assert!(matches!(set.inner, BitsetRepr::Dense { .. }), "Should still be Dense at threshold");

        // Remove one more element to trigger conversion check
        assert!(set.remove(DENSE_TO_SPARSE_THRESHOLD - 1));
        // remove() should trigger the conversion
        assert!(matches!(set.inner, BitsetRepr::Sparse(_)), "Should be Sparse below threshold");
        assert_eq!(set.len(), DENSE_TO_SPARSE_THRESHOLD - 1); // len() on sparse is cheap

        // Check contains works on Sparse
        assert!(set.contains(0));
        assert!(!set.contains(DENSE_TO_SPARSE_THRESHOLD - 1));
    }

    #[test]
    fn test_dense_bounds_update() {
        let mut set = HybridBitset::new();
        // Force dense
        for i in 0..SPARSE_TO_DENSE_THRESHOLD { set.insert(i); }
        assert!(matches!(set.inner, BitsetRepr::Dense { .. }));
        // Ensure bounds are exact initially
        assert_eq!(set.len(), SPARSE_TO_DENSE_THRESHOLD);

        // Insert existing - bounds shouldn't change if exact
        set.insert(10);
         if let BitsetRepr::Dense { lower_bound_count, upper_bound_count, .. } = &set.inner {
             assert_eq!(*lower_bound_count, SPARSE_TO_DENSE_THRESHOLD);
             assert_eq!(*upper_bound_count, SPARSE_TO_DENSE_THRESHOLD);
         }

        // Insert new
        set.insert(SPARSE_TO_DENSE_THRESHOLD + 100); // Also tests resize
         if let BitsetRepr::Dense { lower_bound_count, upper_bound_count, .. } = &set.inner {
             assert_eq!(*lower_bound_count, SPARSE_TO_DENSE_THRESHOLD + 1);
             assert_eq!(*upper_bound_count, SPARSE_TO_DENSE_THRESHOLD + 1);
         }
         assert_eq!(set.len(), SPARSE_TO_DENSE_THRESHOLD + 1); // len() forces exact count check

         // Remove existing
         set.remove(0);
          if let BitsetRepr::Dense { lower_bound_count, upper_bound_count, .. } = &set.inner {
             assert_eq!(*lower_bound_count, SPARSE_TO_DENSE_THRESHOLD);
             assert_eq!(*upper_bound_count, SPARSE_TO_DENSE_THRESHOLD);
         }
         assert_eq!(set.len(), SPARSE_TO_DENSE_THRESHOLD);

         // Remove non-existing (within bounds)
         set.remove(1); // Was present, now removed
         set.remove(1); // Try removing again
          if let BitsetRepr::Dense { lower_bound_count, upper_bound_count, .. } = &set.inner {
             assert_eq!(*lower_bound_count, SPARSE_TO_DENSE_THRESHOLD -1);
             assert_eq!(*upper_bound_count, SPARSE_TO_DENSE_THRESHOLD -1);
         }
         assert_eq!(set.len(), SPARSE_TO_DENSE_THRESHOLD-1);

         // Remove non-existing (out of bounds)
         set.remove(999999);
          if let BitsetRepr::Dense { lower_bound_count, upper_bound_count, .. } = &set.inner {
             assert_eq!(*lower_bound_count, SPARSE_TO_DENSE_THRESHOLD -1);
             assert_eq!(*upper_bound_count, SPARSE_TO_DENSE_THRESHOLD -1);
         }
         assert_eq!(set.len(), SPARSE_TO_DENSE_THRESHOLD-1);
    }

    #[test]
    fn test_set_ops_sparse_sparse() {
        let set1 = HybridBitset::from_iter(vec![1, 2, 3, 10]);
        let set2 = HybridBitset::from_iter(vec![3, 4, 5, 10]);

        let intersection = &set1 & &set2;
        let union = &set1 | &set2;
        let difference = &set1 - &set2; // set1 \ set2
        let sym_diff = &set1 ^ &set2;

        assert_eq!(intersection.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![3, 10]));
        assert_eq!(union.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![1, 2, 3, 4, 5, 10]));
        assert_eq!(difference.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![1, 2]));
        assert_eq!(sym_diff.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![1, 2, 4, 5]));

        assert!(matches!(intersection.inner, BitsetRepr::Sparse(_)));
        assert!(matches!(union.inner, BitsetRepr::Sparse(_)));
        assert!(matches!(difference.inner, BitsetRepr::Sparse(_)));
        assert!(matches!(sym_diff.inner, BitsetRepr::Sparse(_)));
    }

     #[test]
    fn test_set_ops_dense_dense() {
        let mut set1 = HybridBitset::new();
        let mut set2 = HybridBitset::new();
        // Make lengths different but overlapping
        for i in 0..SPARSE_TO_DENSE_THRESHOLD + 10 { set1.insert(i); } // [0, N+10) Dense
        for i in 5..SPARSE_TO_DENSE_THRESHOLD + 20 { set2.insert(i); } // [5, N+20) Dense

        assert!(matches!(set1.inner, BitsetRepr::Dense { .. }));
        assert!(matches!(set2.inner, BitsetRepr::Dense { .. }));

        let intersection = &set1 & &set2; // [5..SPARSE_TO_DENSE_THRESHOLD + 10)
        let union = &set1 | &set2;       // [0..SPARSE_TO_DENSE_THRESHOLD + 20)
        let difference = &set1 - &set2; // [0..5)
        let sym_diff = &set1 ^ &set2;   // [0..5) U [SPARSE_TO_DENSE_THRESHOLD+10..SPARSE_TO_DENSE_THRESHOLD+20)

        let intersection_expected: BTreeSet<usize> = (5..SPARSE_TO_DENSE_THRESHOLD + 10).collect();
        let union_expected: BTreeSet<usize> = (0..SPARSE_TO_DENSE_THRESHOLD + 20).collect();
        let difference_expected: BTreeSet<usize> = (0..5).collect();
        let sym_diff_expected: BTreeSet<usize> = (0..5).chain(SPARSE_TO_DENSE_THRESHOLD + 10..SPARSE_TO_DENSE_THRESHOLD + 20).collect();


        assert_eq!(intersection.iter().collect::<BTreeSet<usize>>(), intersection_expected);
        assert_eq!(union.iter().collect::<BTreeSet<usize>>(), union_expected);
        assert_eq!(difference.iter().collect::<BTreeSet<usize>>(), difference_expected);
        assert_eq!(sym_diff.iter().collect::<BTreeSet<usize>>(), sym_diff_expected);

        // Check representation of results (depends on thresholds)
        // Intersection might become sparse if overlap is small
        if intersection_expected.len() < DENSE_TO_SPARSE_THRESHOLD {
             assert!(matches!(intersection.inner, BitsetRepr::Sparse { .. }));
        } else {
             assert!(matches!(intersection.inner, BitsetRepr::Dense { .. }));
        }
        assert!(matches!(union.inner, BitsetRepr::Dense { .. })); // Union is large -> Dense
        // Difference is small -> Sparse
        assert!(matches!(difference.inner, BitsetRepr::Sparse { .. }));
        // Sym Diff might be sparse or dense depending on threshold and exact values
         if sym_diff_expected.len() < DENSE_TO_SPARSE_THRESHOLD {
             assert!(matches!(sym_diff.inner, BitsetRepr::Sparse { .. }));
        } else {
             assert!(matches!(sym_diff.inner, BitsetRepr::Dense { .. }));
        }
    }

     #[test]
    fn test_set_ops_mixed() {
        let set1_sparse = HybridBitset::from_iter(vec![1, 2, 3, SPARSE_TO_DENSE_THRESHOLD + 100]); // Sparse, includes large value
        let mut set2_dense = HybridBitset::new(); // Dense
        for i in 0..SPARSE_TO_DENSE_THRESHOLD + 5 { set2_dense.insert(i); } // [0..N+5)

        assert!(matches!(set1_sparse.inner, BitsetRepr::Sparse(_)));
        assert!(matches!(set2_dense.inner, BitsetRepr::Dense { .. }));

        // --- Sparse & Dense ---
        let intersection1 = &set1_sparse & &set2_dense; // {1, 2, 3}
        let intersection1_expected: BTreeSet<usize> = vec![1, 2, 3].into_iter().collect();
        assert_eq!(intersection1.iter().collect::<BTreeSet<usize>>(), intersection1_expected);
        assert!(matches!(intersection1.inner, BitsetRepr::Sparse(_))); // Result is small

        // --- Dense & Sparse ---
        let intersection2 = &set2_dense & &set1_sparse; // {1, 2, 3}
        assert_eq!(intersection2.iter().collect::<BTreeSet<usize>>(), intersection1_expected);
         assert!(matches!(intersection2.inner, BitsetRepr::Sparse(_)));

        // --- Sparse | Dense ---
        let union1 = &set1_sparse | &set2_dense; // {0..N+5} U {N+100}
        let mut union1_expected: BTreeSet<usize> = (0..SPARSE_TO_DENSE_THRESHOLD + 5).collect();
        union1_expected.insert(SPARSE_TO_DENSE_THRESHOLD + 100);
        assert_eq!(union1.iter().collect::<BTreeSet<usize>>(), union1_expected);
        assert!(matches!(union1.inner, BitsetRepr::Dense { .. })); // Result is large and has large index

         // --- Dense | Sparse ---
        let union2 = &set2_dense | &set1_sparse;
        assert_eq!(union2.iter().collect::<BTreeSet<usize>>(), union1_expected);
        assert!(matches!(union2.inner, BitsetRepr::Dense { .. }));

        // --- Sparse - Dense ---
        let diff1 = &set1_sparse - &set2_dense; // {N+100}
        let diff1_expected: BTreeSet<usize> = vec![SPARSE_TO_DENSE_THRESHOLD + 100].into_iter().collect();
        assert_eq!(diff1.iter().collect::<BTreeSet<usize>>(), diff1_expected);
        assert!(matches!(diff1.inner, BitsetRepr::Sparse(_)));

        // --- Dense - Sparse ---
        let diff2 = &set2_dense - &set1_sparse; // {0, 4..N+5}
        let mut diff2_expected: BTreeSet<usize> = (0..SPARSE_TO_DENSE_THRESHOLD + 5).collect();
        diff2_expected.remove(&1);
        diff2_expected.remove(&2);
        diff2_expected.remove(&3);
        // N+100 is not in set2_dense, so it doesn't affect the result
        assert_eq!(diff2.iter().collect::<BTreeSet<usize>>(), diff2_expected);
        // Result is large -> Dense
        assert!(matches!(diff2.inner, BitsetRepr::Dense { .. }));

        // --- Sparse ^ Dense ---
        let xor1 = &set1_sparse ^ &set2_dense; // {0, 4..N+5} U {N+100}
        let mut xor1_expected = diff2_expected.clone(); // Elements only in Dense
        xor1_expected.insert(SPARSE_TO_DENSE_THRESHOLD + 100); // Element only in Sparse
        assert_eq!(xor1.iter().collect::<BTreeSet<usize>>(), xor1_expected);
        assert!(matches!(xor1.inner, BitsetRepr::Dense { .. })); // Result is large

        // --- Dense ^ Sparse ---
        let xor2 = &set2_dense ^ &set1_sparse;
        assert_eq!(xor2.iter().collect::<BTreeSet<usize>>(), xor1_expected);
        assert!(matches!(xor2.inner, BitsetRepr::Dense { .. }));
    }

     #[test]
    fn test_equality_and_hash() {
        let set1_s = HybridBitset::from_iter(vec![1, 5, 10]); // Sparse
        // Create the same set and force it to be dense for comparison
        let mut set1_d = HybridBitset::from_iter(vec![1, 5, 10]); // Start sparse
        set1_d.ensure_dense(); // Force to dense representation


        let set2_s = HybridBitset::from_iter(vec![1, 5, 11]); // Sparse, different
        let mut set3_d_empty = HybridBitset::from_iter(0..SPARSE_TO_DENSE_THRESHOLD); // Dense
        set3_d_empty.clear(); // Now sparse empty
        set3_d_empty.ensure_dense(); // Force dense empty

        assert!(matches!(set1_s.inner, BitsetRepr::Sparse(_)));
        // Check that set1_d actually became dense
        assert!(matches!(set1_d.inner, BitsetRepr::Dense { .. }), "set1_d should be dense");
        assert_eq!(set1_d.iter().collect::<Vec<_>>(), vec![1, 5, 10]);


        // Test equality
        assert_eq!(set1_s, set1_s);
        assert_eq!(set1_d, set1_d);
        assert_eq!(set1_s, set1_d, "Sparse and Dense representations of the same set should be equal");
        assert_eq!(set1_d, set1_s, "Equality should be symmetric");
        assert_ne!(set1_s, set2_s);
        assert_ne!(set1_d, set2_s);
        assert_ne!(set1_s, set3_d_empty);
        assert_ne!(set1_d, set3_d_empty);

        // Test hashing
        use std::collections::hash_map::DefaultHasher;
        let hash = |s: &HybridBitset| -> u64 {
            let mut hasher = DefaultHasher::new();
            s.hash(&mut hasher);
            hasher.finish()
        };

        let hash1_s = hash(&set1_s);
        let hash1_d = hash(&set1_d);
        let hash2_s = hash(&set2_s);
        let hash3_d = hash(&set3_d_empty);


        assert_eq!(hash1_s, hash1_d, "Hashes of Sparse and Dense representations of the same set should be equal");
        assert_ne!(hash1_s, hash2_s);
        assert_ne!(hash1_d, hash2_s);
        assert_ne!(hash1_s, hash3_d);

        // Test in BTreeSet
        let mut map = BTreeSet::new();
        map.insert(set1_s.clone());
        assert!(map.contains(&set1_s));
        assert!(map.contains(&set1_d), "BTreeSet should find equivalent Dense set using Sparse key's hash");

        map.insert(set1_d.clone()); // Should replace the previous one or do nothing
        assert_eq!(map.len(), 1);

        map.insert(set2_s.clone());
        assert_eq!(map.len(), 2);
        assert!(map.contains(&set2_s));

        map.insert(set3_d_empty.clone());
        assert_eq!(map.len(), 3);
        assert!(map.contains(&set3_d_empty));
    }

     #[test]
    fn test_large_index() {
        let mut set = HybridBitset::new();
        // Use a reasonably large index, but avoid usize::MAX/2 which might cause allocation issues
        // depending on memory. Let's use something like 1 million.
        let large_idx = 1_000_000;
        set.insert(large_idx);
        set.insert(0);

        // Should still be Sparse because count (2) is less than SPARSE_TO_DENSE_THRESHOLD
        assert!(matches!(set.inner, BitsetRepr::Sparse(_)), "Should remain Sparse based on count");
        assert_eq!(set.len(), 2);
        assert!(set.contains(0));
        assert!(set.contains(large_idx));
        assert!(!set.contains(1));
        assert!(!set.contains(large_idx - 1));
        // This block won't be reached if Sparse, which is expected now
        // if let BitsetRepr::Dense{ bits, ..} = &set.inner {
        // assert!(bits.len() > large_idx); // Check bitvec was resized appropriately
        // }


        // Check if it converts back to sparse if large index removed
        // (It's already sparse, so this just removes the element)
        set.remove(large_idx);
        assert_eq!(set.len(), 1);
        assert!(set.contains(0));
        assert!(!set.contains(large_idx));

        // It should remain sparse
        assert!(matches!(set.inner, BitsetRepr::Sparse(_)), "Should still be sparse");
    }

     #[test]
    fn test_clear() {
        let mut set = HybridBitset::from_iter(0..SPARSE_TO_DENSE_THRESHOLD + 10); // Dense
        assert!(matches!(set.inner, BitsetRepr::Dense { .. }));
        assert!(!set.is_empty());
        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert!(matches!(set.inner, BitsetRepr::Sparse(_)), "Clear should reset to Sparse empty");

        let mut set2 = HybridBitset::from_iter(vec![1,2,3]); // Sparse
        assert!(matches!(set2.inner, BitsetRepr::Sparse(_)));
        assert!(!set2.is_empty());
        set2.clear();
        assert!(set2.is_empty());
        assert_eq!(set2.len(), 0);
        assert!(matches!(set2.inner, BitsetRepr::Sparse(_)));
    }

    #[test]
    fn test_assign_ops() {
        // Or Assign
        let mut set1 = HybridBitset::from_iter(vec![1, 2, 10]); // Sparse
        let set2 = HybridBitset::from_iter(vec![2, 3, 20]); // Sparse
        set1 |= set2;
        assert_eq!(set1.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 2, 3, 10, 20]));
        assert!(matches!(set1.inner, BitsetRepr::Sparse(_))); // Still sparse

        // And Assign
        let mut set3 = HybridBitset::from_iter(0..SPARSE_TO_DENSE_THRESHOLD); // Dense
        let set4 = HybridBitset::from_iter( (SPARSE_TO_DENSE_THRESHOLD/2)..SPARSE_TO_DENSE_THRESHOLD + 10); // Dense overlap
        let expected_and = (SPARSE_TO_DENSE_THRESHOLD/2..SPARSE_TO_DENSE_THRESHOLD).collect::<BTreeSet<_>>();
        set3 &= set4;
        assert_eq!(set3.iter().collect::<BTreeSet<_>>(), expected_and);
        // The operation was Dense &= Sparse. The result size is 64.
        // The Dense & Sparse path creates a Sparse result. check_representation checks
        // if 64 >= SPARSE_TO_DENSE_THRESHOLD (128), which is false. So it stays Sparse.
        assert!(matches!(set3.inner, BitsetRepr::Sparse { .. }), "Result of Dense &= Sparse with size 64 should be Sparse");

        // Xor Assign
        let mut set5 = HybridBitset::from_iter(vec![1, 2, 3]); // Sparse
        let set6 = HybridBitset::from_iter(vec![3, 4, 5]); // Sparse
        set5 ^= set6;
        assert_eq!(set5.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 2, 4, 5]));
        assert!(matches!(set5.inner, BitsetRepr::Sparse(_)));

        // Sub Assign
        let mut set7 = HybridBitset::from_iter(vec![1, 2, 3, 4, 5]); // Sparse
        let set8 = HybridBitset::from_iter(vec![2, 4, 6]); // Sparse
        set7 -= set8;
        assert_eq!(set7.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 3, 5]));
        assert!(matches!(set7.inner, BitsetRepr::Sparse(_)));
    }

    #[test]
    fn test_assign_ops_ref() {
         // Or Assign Ref
        let mut set1 = HybridBitset::from_iter(vec![1, 2, 10]); // Sparse
        let set2 = HybridBitset::from_iter(vec![2, 3, 20]); // Sparse
        set1 |= &set2;
        assert_eq!(set1.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 2, 3, 10, 20]));
        assert!(matches!(set1.inner, BitsetRepr::Sparse(_))); // Still sparse

        // And Assign Ref
        let mut set3 = HybridBitset::from_iter(0..SPARSE_TO_DENSE_THRESHOLD); // Dense
        let set4 = HybridBitset::from_iter( (SPARSE_TO_DENSE_THRESHOLD/2)..SPARSE_TO_DENSE_THRESHOLD + 10); // Dense overlap
        let expected_and = (SPARSE_TO_DENSE_THRESHOLD/2..SPARSE_TO_DENSE_THRESHOLD).collect::<BTreeSet<_>>();
        set3 &= &set4;
        assert_eq!(set3.iter().collect::<BTreeSet<_>>(), expected_and);
        assert!(matches!(set3.inner, BitsetRepr::Sparse { .. }), "Result of Dense &= Sparse with size 64 should be Sparse");

        // Xor Assign Ref
        let mut set5 = HybridBitset::from_iter(vec![1, 2, 3]); // Sparse
        let set6 = HybridBitset::from_iter(vec![3, 4, 5]); // Sparse
        set5 ^= &set6;
        assert_eq!(set5.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 2, 4, 5]));
        assert!(matches!(set5.inner, BitsetRepr::Sparse(_)));

        // Sub Assign Ref
        let mut set7 = HybridBitset::from_iter(vec![1, 2, 3, 4, 5]); // Sparse
        let set8 = HybridBitset::from_iter(vec![2, 4, 6]); // Sparse
        set7 -= &set8;
        assert_eq!(set7.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 3, 5]));
        assert!(matches!(set7.inner, BitsetRepr::Sparse(_)));
    }

    #[test]
    fn test_dense_dense_edge_cases() {
        // Empty sets
        let mut d1 = HybridBitset::new(); d1.ensure_dense();
        let mut d2 = HybridBitset::new(); d2.ensure_dense();
        let d3 = HybridBitset::from_iter(0..SPARSE_TO_DENSE_THRESHOLD); // Non-empty dense

        assert_eq!(&d1 & &d2, d1);
        assert_eq!(&d1 | &d2, d1);
        assert_eq!(&d1 ^ &d2, d1);
        assert_eq!(&d1 - &d2, d1);

        assert_eq!(&d1 & &d3, d1); // Empty & NonEmpty = Empty
        assert_eq!(&d3 & &d1, d1);
        assert_eq!(&d1 | &d3, d3); // Empty | NonEmpty = NonEmpty
        assert_eq!(&d3 | &d1, d3);
        assert_eq!(&d1 ^ &d3, d3); // Empty ^ NonEmpty = NonEmpty
        assert_eq!(&d3 ^ &d1, d3);
        assert_eq!(&d1 - &d3, d1); // Empty - NonEmpty = Empty
        assert_eq!(&d3 - &d1, d3); // NonEmpty - Empty = NonEmpty

        // Different lengths
        let mut d4 = HybridBitset::from_iter(0..5); d4.ensure_dense();
        let mut d5 = HybridBitset::from_iter(3..10); d5.ensure_dense();

        let inter = &d4 & &d5; // {3, 4}
        assert_eq!(inter.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![3, 4]));
        assert!(matches!(inter.inner, BitsetRepr::Sparse(_)));

        let union = &d4 | &d5; // {0..10}
        assert_eq!(union.iter().collect::<BTreeSet<_>>(), (0..10).collect::<BTreeSet<_>>());
        // Dense | Dense now calls check_representation.
        // Result count is 10. 10 < DENSE_TO_SPARSE_THRESHOLD (64).
        // Should convert to Sparse.
        assert!(matches!(union.inner, BitsetRepr::Sparse(_)), "Union result (size 10) should become Sparse");

        let diff = &d4 - &d5; // {0, 1, 2}
        assert_eq!(diff.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![0, 1, 2]));
        assert!(matches!(diff.inner, BitsetRepr::Sparse(_)));

        let sym_diff = &d4 ^ &d5; // {0,1,2} U {5,6,7,8,9}
        assert_eq!(sym_diff.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![0,1,2,5,6,7,8,9]));
        assert!(matches!(sym_diff.inner, BitsetRepr::Sparse(_)));
    }

     #[test]
    fn test_from_iterator() {
        let data = vec![10, 20, 10, 30, 20];
        let set: HybridBitset = data.into_iter().collect(); // Use FromIterator trait

        let expected: BTreeSet<usize> = vec![10, 20, 30].into_iter().collect();
        assert_eq!(set.iter().collect::<BTreeSet<_>>(), expected);
        assert!(matches!(set.inner, BitsetRepr::Sparse(_)));
    }

    // Implement FromIterator for HybridBitset - Already implemented above the test module
    // impl FromIterator<usize> for HybridBitset {
    //     fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
    //         let mut set = HybridBitset::new();
    //         for i in iter {
    //             set.insert(i);
    //         }
    //         set
    //     }
    // }

    #[test]
    fn test_iter_bools() {
        // Test with an empty set (starts sparse)
        let empty_set = HybridBitset::new();
        assert_eq!(empty_set.iter_bools().collect::<Vec<bool>>(), Vec::<bool>::new());
        assert_eq!(empty_set.iter_bools().len(), 0);

        // Test with a sparse set
        let sparse_set = HybridBitset::from_iter(vec![1, 3]); // Max index is 3
        // Expected: [false (for 0), true (for 1), false (for 2), true (for 3)]
        let expected_sparse_bools = vec![false, true, false, true];
        assert_eq!(sparse_set.iter_bools().collect::<Vec<bool>>(), expected_sparse_bools);
        assert_eq!(sparse_set.iter_bools().len(), expected_sparse_bools.len());

        // Test with a dense set
        let mut dense_set_forced = HybridBitset::new();
        for i in 0..5 { // Small range, will be sparse initially
            if i == 1 || i == 3 {
                dense_set_forced.insert(i);
            }
        }
        // Manually convert to dense for this test case, assuming it has some internal length.
        // If we want to test the dense path of iter_bools, we need a dense set.
        // Let's assume convert_to_dense works and use it.
        dense_set_forced.convert_to_dense(); // Now it's dense. Max index is 3, length might be 4 or more.
                                             // If it was {1,3}, after convert_to_dense, bits might be [0,1,0,1] (len 4)

        // Let's re-evaluate the dense test for iter_bools.
        // The iter_bools for dense iterates up to bits.len().
        // If we have a dense set with bits representing [false, true, false, true] (len 4)
        // then iter_bools should yield exactly that.

        let mut test_dense = HybridBitset::new();
        test_dense.insert(1);
        test_dense.insert(3); // sparse: {1, 3}
        test_dense.convert_to_dense(); // dense: bits for [0,1,0,1], len=4
                                       // lower_bound_count=2, upper_bound_count=2

        let expected_dense_bools = vec![false, true, false, true];
        assert_eq!(test_dense.iter_bools().collect::<Vec<bool>>(), expected_dense_bools);
        assert_eq!(test_dense.iter_bools().len(), expected_dense_bools.len());

        // Test with an empty dense set
        let mut empty_dense_set = HybridBitset::new();
        empty_dense_set.convert_to_dense(); // Becomes Dense { bits: [], ... }
        assert_eq!(empty_dense_set.iter_bools().collect::<Vec<bool>>(), Vec::<bool>::new());
        assert_eq!(empty_dense_set.iter_bools().len(), 0);
    }
}
