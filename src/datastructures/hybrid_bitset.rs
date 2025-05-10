#![allow(dead_code)] // Allow unused code for the example

use bitvec::prelude::*;
use std::collections::BTreeSet;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index, IndexMut, Sub, SubAssign};
use std::cmp::{max, min};
use std::hash::{Hash, Hasher};
use std::iter::FromIterator; // Needed for collect into BTreeSet in tests
use std::cell::RefCell; // Needed for caching

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

// New thresholds for Holey representation
const DENSE_TO_HOLEY_THRESHOLD: usize = 9 * SPARSE_TO_DENSE_THRESHOLD; // e.g., 90% full
const HOLEY_TO_SPARSE_THRESHOLD: usize = 64; // if “missing” grows beyond this

// --- Enum for Internal Representation ---

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq)]
enum BitsetRepr {
    Sparse(BTreeSet<usize>),
    Holey {
        max_index: usize,
        // indices that are FALSE inside the otherwise full prefix 0..=max_index
        missing: BTreeSet<usize>,
        // cached_exact_count is not needed because it is trivial:
        //     (max_index + 1) - missing.len()
    },
    Dense {
        bits: BitVec<usize, Lsb0>,
        // Stores the exact count if known. None if dirty and needs recalculation.
        cached_exact_count: RefCell<Option<usize>>,
    },
}

impl Hash for BitsetRepr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            BitsetRepr::Sparse(set) => set.hash(state),
            BitsetRepr::Holey { max_index, missing } => {
                max_index.hash(state);
                missing.hash(state);
            }
            BitsetRepr::Dense { bits, .. } => bits.hash(state),
        }
    }
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

    /// Creates a new HybridBitset with all indices from 0 up to `max_value` (inclusive) set to true.
    pub fn ones(max_value: usize) -> Self {
        assert!(DENSE_TO_SPARSE_THRESHOLD < SPARSE_TO_DENSE_THRESHOLD, "Thresholds misconfigured");
        let num_elements = max_value.saturating_add(1); // Number of elements from 0 to max_value

        if num_elements == 0 { // max_value was usize::MAX, and num_elements overflowed to 0
            // Or if max_value was passed as a very large number that makes num_elements effectively 0 after saturation.
            // This case implies an attempt to create an impossibly large set.
            // For safety, return an empty set, though this scenario is unlikely with typical token IDs.
            return Self::new();
        }

        // If the target size is large enough, start with Holey (initially empty missing set)
        // or Dense, depending on size vs. DENSE_TO_HOLEY_THRESHOLD
        if max_value >= DENSE_TO_HOLEY_THRESHOLD {
             HybridBitset {
                 inner: BitsetRepr::Holey {
                     max_index: max_value,
                     missing: BTreeSet::new(),
                 }
             }
        } else if num_elements >= SPARSE_TO_DENSE_THRESHOLD {
            // Use Dense representation
            let bits = bitvec![usize, Lsb0; 1; num_elements]; // Create BitVec of `num_elements` length, all set to 1
            HybridBitset {
                inner: BitsetRepr::Dense { bits, cached_exact_count: RefCell::new(Some(num_elements)) }
            }
        } else {
            // Use Sparse representation
            HybridBitset::from_iter(0..num_elements)
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
            BitsetRepr::Holey { max_index, missing } => {
                 max_index.saturating_add(1).saturating_sub(missing.len())
            }
            BitsetRepr::Dense { bits, cached_exact_count } => {
                let mut count_opt = cached_exact_count.borrow_mut(); // Borrow mutably to update
                if let Some(count) = *count_opt {
                    // Count is known
                    count
                } else {
                    // Count is not known, recalculate and cache it
                    let exact_count = bits.count_ones();
                    *count_opt = Some(exact_count);
                    exact_count
                }
            }
        }
    }

    /// Returns true if the bitset contains no set bits.
    pub fn is_empty(&self) -> bool {
        match &self.inner { // Use mutable borrow to potentially update count
            BitsetRepr::Sparse(set) => set.is_empty(),
            BitsetRepr::Holey { max_index, missing } => {
                 (*max_index).saturating_add(1) == missing.len()
            }
            BitsetRepr::Dense { .. } => { // We don't need to destructure cached_exact_count here
                 self.len() == 0 // len() now handles caching
            }
        }
    }

    /// Checks if a specific index is set.
    pub fn contains(&self, index: usize) -> bool {
        match &self.inner {
            BitsetRepr::Sparse(set) => set.contains(&index),
            BitsetRepr::Holey { max_index, missing } => {
                index <= *max_index && !missing.contains(&index)
            }
            BitsetRepr::Dense { bits, .. } => {
                // Use get() for safe bounds checking
                bits.get(index).map_or(false, |bitref| *bitref)
            }
        }
    }

    /// Inserts an index into the set. Returns true if the index was not already present.
    /// May trigger a conversion from Sparse to Dense or Dense/Sparse to Holey.
    pub fn insert(&mut self, index: usize) -> bool {
        let was_present;
        let mut needs_conversion_check = false;
        match &mut self.inner {
            BitsetRepr::Sparse(set) => {
                was_present = set.contains(&index);
                if !was_present {
                    set.insert(index);
                    // Check if we need to convert
                    needs_conversion_check = true; // Always check after insert into sparse
                }
            }
            BitsetRepr::Holey { max_index, missing } => {
                 let old_max_index = *max_index;
                 was_present = if index <= old_max_index {
                     // Index is within the implicitly true range.
                     // If it's in 'missing', removing it means it is now set.
                     // If it's not in 'missing', it was already set.
                    !missing.remove(&index) // returns true if it was in `missing` (was unset)
                 } else {
                    // Index is outside the implicitly true range.
                    // Inserting it extends the implicitly true range.
                    // All indices from old_max_index + 1 up to index - 1 are now implicitly true,
                    // but they were previously *not* in the set (implicitly false in sparse/dense
                    // before conversion). We need to add them to 'missing'.
                    for gap in (old_max_index + 1)..index {
                         missing.insert(gap);
                    }
                    *max_index = index;
                    false // it was not present before
                 };
                 // Insert into Holey can increase the effective size or decrease the number of missing elements.
                 // Either way, density might change or max_index increases, so check representation.
                 needs_conversion_check = true;
            }
            BitsetRepr::Dense { bits, cached_exact_count } => {
                // Ensure the BitVec is large enough
                if index >= bits.len() {
                    // Calculate required capacity increase. Avoid excessive overallocation for single large indices.
                    let new_len = index + 1;
                    bits.resize(new_len, false);
                    // Invalidate cache as the length of the BitVec changed
                    *cached_exact_count.borrow_mut() = None;
                }

                // Check current state before setting using immutable borrow first
                let current_bit = bits.get(index).map_or(false, |b| *b);
                was_present = current_bit;

                if !was_present {
                    // Now get mutable borrow to set
                    bits.set(index, true);
                    // Update cached count if known
                    if let Some(count_ref) = cached_exact_count.borrow_mut().as_mut() {
                        *count_ref += 1;
                    }
                    // Density might increase, check representation
                    needs_conversion_check = true;
                }
                // If already Dense and element was present, nothing changes, no conversion check needed.
            }
        }

        if needs_conversion_check {
            self.check_representation();
        }

        !was_present // Return true if it was newly inserted
    }

    /// Sets the bit at `index` to `value`. Returns true if the bit was changed.
    pub fn set(&mut self, index: usize, value: bool) -> bool {
        if value {
            self.insert(index) // insert returns true if it was NOT present
        } else {
            self.remove(index) // remove returns true if it WAS present
        }
    }

    /// Removes an index from the set. Returns true if the index was present.
    /// May trigger a conversion from Dense/Holey to Sparse or Dense to Holey.
    pub fn remove(&mut self, index: usize) -> bool {
        let was_present;
        let mut needs_conversion_check = false;

        match &mut self.inner {
            BitsetRepr::Sparse(set) => {
                was_present = set.remove(&index);
                // No conversion check needed on remove for Sparse - can only shrink
            }
            BitsetRepr::Holey { max_index, missing } => {
                was_present = if index <= *max_index {
                    // Index is within the implicitly true range.
                    // Adding it to 'missing' means it's now unset.
                    missing.insert(index) // returns true if it was *not* in `missing` (i.e., was set)
                } else {
                    // Index is outside the implicitly true range, already unset.
                    false // not present
                };
                if was_present {
                     // Density might decrease or missing set grows, check representation
                    needs_conversion_check = true;
                }
                 // If index was already missing (index <= max_index && index in missing),
                 // or if index was outside max_index, nothing changes, no conversion check needed.
            }
            BitsetRepr::Dense { bits, cached_exact_count } => {
                // Check if index is within bounds first
                if index < bits.len() {
                     // Check current state before clearing
                    let current_bit = bits.get(index).map_or(false, |b| *b);
                    was_present = current_bit;

                    if was_present {
                        bits.set(index, false);
                        // Update cached count if known
                        if let Some(count_ref) = cached_exact_count.borrow_mut().as_mut() {
                            *count_ref -= 1;
                        }
                        // Signal that representation might need to be checked
                        needs_conversion_check = true;
                    }
                } else {
                    // Index out of bounds, definitely wasn't present
                    was_present = false;
                }
            }
        }

        if needs_conversion_check {
            // This flag is set true only if the set was modified and might need conversion.
            self.check_representation();
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
                BitsetRepr::Holey { max_index, missing } => IterInner::Holey {
                    current: 0,
                    max_index: *max_index,
                    missing_iter: missing.iter(),
                },
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
    /// For a Holey set, the limit is `max_index`.
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
                    let max_val_in_set = set.last().copied().unwrap_or(0);
                    BoolIter {
                        inner: BoolIterInner::Sparse {
                            set,
                            current_idx: 0,
                            max_idx_to_iterate: max_val_in_set,
                        }
                    }
                }
            }
            BitsetRepr::Holey { max_index, missing } => {
                 BoolIter {
                     inner: BoolIterInner::Holey {
                         current: 0,
                         max_index: *max_index,
                         missing: missing,
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
        if matches!(self.inner, BitsetRepr::Sparse(_)) || matches!(self.inner, BitsetRepr::Holey { .. }) {
            self.convert_to_dense();
        }
    }

    // --- Helper: Force conversion to Sparse ---
     fn ensure_sparse(&mut self) {
         if matches!(self.inner, BitsetRepr::Dense { .. }) || matches!(self.inner, BitsetRepr::Holey { .. }){
             self.convert_to_sparse();
         }
     }

    // --- Helper: Force conversion to Holey ---
     fn ensure_holey(&mut self) {
         if matches!(self.inner, BitsetRepr::Sparse(_)) || matches!(self.inner, BitsetRepr::Dense { .. }){
             self.convert_to_holey();
         }
     }


    // --- Helper: Convert Sparse -> Dense ---
    fn convert_to_dense(&mut self) {
        let (elements, max_index_opt) = match &self.inner {
             BitsetRepr::Sparse(set) => (set.iter().copied().collect::<Vec<_>>(), set.iter().max().copied()),
             BitsetRepr::Holey { max_index, missing } => {
                 // Collect all set elements from Holey: 0..=max_index minus missing
                 let mut elements = Vec::new();
                 for i in 0..=*max_index {
                     if !missing.contains(&i) {
                         elements.push(i);
                     }
                 }
                 (elements, Some(*max_index)) // Max index is max_index from Holey
             }
             BitsetRepr::Dense { .. } => return, // Already Dense
         };

        if elements.is_empty() {
             self.inner = BitsetRepr::Dense {
                bits: BitVec::new(),
                cached_exact_count: RefCell::new(Some(0)),
            };
            return;
        }

        let max_index = max_index_opt.unwrap_or_else(|| elements.iter().max().copied().unwrap_or(0));

        let mut bits = bitvec![usize, Lsb0; 0; max_index + 1];
        for index in elements {
            // Safety: index is guaranteed to be <= max_index here
            bits.set(index, true);
        }

        self.inner = BitsetRepr::Dense {
            bits,
            cached_exact_count: RefCell::new(Some(elements.len())),
        };
    }

    // --- Helper: Convert Dense -> Sparse ---
    fn convert_to_sparse(&mut self) {
        let elements: Vec<usize> = match &self.inner {
             BitsetRepr::Sparse(_) => return, // Already Sparse
             BitsetRepr::Holey { max_index, missing } => {
                 // Collect all set elements from Holey: 0..=max_index minus missing
                 let mut elements = Vec::new();
                 for i in 0..=*max_index {
                     if !missing.contains(&i) {
                         elements.push(i);
                     }
                 }
                 elements
             }
             BitsetRepr::Dense { bits, .. } => { // removed cached_exact_count from destructuring
                 bits.iter_ones().collect()
             }
         };

        let mut set = BTreeSet::new();
        for index in elements {
            set.insert(index);
        }
        self.inner = BitsetRepr::Sparse(set);
    }

    // --- Helper: Convert to Holey ---
     fn convert_to_holey(&mut self) {
         let (elements, max_index_opt) = match &self.inner {
             BitsetRepr::Sparse(set) => (set.iter().copied().collect::<Vec<_>>(), set.iter().max().copied()),
             BitsetRepr::Holey { .. } => return, // Already Holey
             BitsetRepr::Dense { bits, .. } => (bits.iter_ones().collect::<Vec<_>>(), bits.len().checked_sub(1)),
         };

         if elements.is_empty() {
             // Converting an empty set to Holey is somewhat ambiguous regarding max_index.
             // The simplest is to represent it as empty Sparse.
             self.inner = BitsetRepr::Sparse(BTreeSet::new());
             return;
         }

         // Determine the maximum index. This is the range end for the Holey set.
         let max_index = max_index_opt.unwrap_or_else(|| elements.iter().max().copied().unwrap_or(0));

         let mut missing = BTreeSet::new();
         let mut elements_iter = elements.into_iter().peekable();

         for i in 0..=max_index {
             if let Some(&element) = elements_iter.peek() {
                 if i == element {
                     elements_iter.next(); // This element is present, skip it
                 } else {
                     missing.insert(i); // This element is missing
                 }
             } else {
                 missing.insert(i); // All remaining elements in the range are missing
             }
         }

         self.inner = BitsetRepr::Holey { max_index, missing };
     }


    // --- Helper: Check and potentially convert after an operation ---
    // This should be called after operations that might change the count or max_index significantly.
    fn check_representation(&mut self) {
        match &mut self.inner {
            BitsetRepr::Sparse(set) => {
                let count = set.len();
                if count >= SPARSE_TO_DENSE_THRESHOLD {
                    // Calculate potential max index if it were dense
                    let max_index_opt = set.last().copied();
                    if let Some(max_index) = max_index_opt {
                         // Estimate dense size needed
                         let required_dense_len = max_index.saturating_add(1);
                         // If the sparse set, when dense, is large and dense enough, convert to Holey
                         if required_dense_len >= DENSE_TO_HOLEY_THRESHOLD && count * 10 >= required_dense_len * 9 { // 90% density
                            self.convert_to_holey();
                            return;
                         }
                    }
                    // Otherwise, convert to Dense
                    self.convert_to_dense();
                }
            }
             BitsetRepr::Holey { max_index, missing } => {
                 let count = (*max_index).saturating_add(1).saturating_sub(missing.len());
                 // Convert to Sparse if the number of missing elements is large
                 if missing.len() >= HOLEY_TO_SPARSE_THRESHOLD {
                     self.convert_to_sparse();
                 }
                 // Convert to Dense if density is low (Holey is not efficient)
                 else if (*max_index).saturating_add(1) > 0 && count * 2 < (*max_index).saturating_add(1) { // < 50% density
                    self.convert_to_dense();
                 }
             }
            BitsetRepr::Dense { bits, cached_exact_count } => {
                 let count = {
                     let mut count_opt = cached_exact_count.borrow_mut();
                     if let Some(c) = *count_opt {
                         c
                     } else {
                         let exact_c = bits.count_ones();
                         *count_opt = Some(exact_c);
                         exact_c
                     }
                 };
                 let dense_len = bits.len();

                 // Convert to Sparse if count is below threshold
                 if count < DENSE_TO_SPARSE_THRESHOLD {
                     self.convert_to_sparse();
                 }
                 // Convert to Holey if dense and density is high
                 else if dense_len > 0 && count * 10 >= dense_len * 9 { // >= 90% density
                     self.convert_to_holey();
                 }
            }
        }
    }

    /// Helper for bitwise ops when one or both are Holey. Converts to Dense and performs op.
    fn bitwise_op_fallback<F>(&self, rhs: &Self, op: F) -> Self
    where
        F: FnOnce(&BitVec<usize, Lsb0>, &BitVec<usize, Lsb0>) -> BitVec<usize, Lsb0>,
    {
        // Convert both to Dense for the operation
        let mut self_dense = self.clone();
        self_dense.ensure_dense();
        let mut rhs_dense = rhs.clone();
        rhs_dense.ensure_dense();

        if let (BitsetRepr::Dense { bits: bits1, .. }, BitsetRepr::Dense { bits: bits2, .. }) = (&self_dense.inner, &rhs_dense.inner) {
             // Apply the bitwise operation
            let result_bits = op(bits1, bits2);

            let exact_count = result_bits.count_ones();
            let mut result = HybridBitset {
                inner: BitsetRepr::Dense {
                    bits: result_bits,
                    cached_exact_count: RefCell::new(Some(exact_count)),
                }
            };
            result.check_representation(); // Check if result should be Sparse or Holey
            result

        } else {
             // This case should ideally not happen if ensure_dense works,
             // but as a fallback, return empty or handle error. Empty is safer.
             HybridBitset::new()
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
    Holey {
        current: usize,
        max_index: usize,
        missing_iter: std::collections::btree_set::Iter<'a, usize>,
    },
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
            IterInner::Holey { current, max_index, missing_iter } => {
                 while *current <= *max_index {
                     let i = *current;
                     *current += 1;
                     // advance the “peek” on missing_iter (it is sorted)
                     if let Some(&m) = missing_iter.clone().next() {
                         if i == m {
                             missing_iter.next();  // skip missing element
                             continue;             // look at next i
                         }
                     }
                     return Some(i);
                 }
                 None
            }
            IterInner::Dense(iter) => iter.next(),
        }
    }

    // Optional: Provide size_hint if possible
    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            IterInner::Sparse(iter) => iter.size_hint(), // BTreeSet::Iter provides exact size hint
            IterInner::Holey { current, max_index, missing_iter } => {
                let remaining_in_range = if *current > *max_index {
                     0
                } else {
                     (*max_index - *current).saturating_add(1)
                };
                let missing_remaining = missing_iter.size_hint().0; // Exact size hint is available
                let remaining = remaining_in_range.saturating_sub(missing_remaining);
                (remaining, Some(remaining))
            }
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
             BitsetRepr::Holey { max_index, missing } => {
                 let mut elements = Vec::new();
                 let mut missing_iter = missing.into_iter().peekable();
                 for i in 0..=max_index {
                     if let Some(&m) = missing_iter.peek() {
                         if i == m {
                             missing_iter.next();
                         } else {
                             elements.push(i);
                         }
                     } else {
                         elements.push(i);
                     }
                 }
                 elements
             }
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
    Holey {
        current: usize,
        max_index: usize,
        missing: &'a BTreeSet<usize>,
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
            BoolIterInner::Holey { current, max_index, missing } => {
                 if *current > *max_index { return None; }
                 let val = !missing.contains(current);
                 *current += 1;
                 Some(val)
            }
            BoolIterInner::Dense(iter) => iter.next().map(|bit_ref| *bit_ref),
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
             BoolIterInner::Holey { current, max_index, .. } => {
                 let remaining = if *current > *max_index {
                     0
                 } else {
                     (*max_index - *current) + 1
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
        // Fallback for Holey combinations to simplify logic
        if matches!(&self.inner, BitsetRepr::Holey { .. }) || matches!(&rhs.inner, BitsetRepr::Holey { .. }) {
            return self.bitwise_op_fallback(rhs, |b1, b2| b1 & b2);
        }

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
                result.check_representation(); // Check if result should be Dense or Holey
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
                        cached_exact_count: RefCell::new(Some(exact_count)),
                    }
                };
                result.check_representation(); // Check if result should be Sparse or Holey
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
                 result.check_representation(); // Check if result should be Dense or Holey (unlikely)
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
                 result.check_representation(); // Check if result should be Dense or Holey
                 result
            }
            // Holey vs other handled by fallback
            _ => unreachable!("Holey combinations should be handled by fallback"),
        }
    }
}

impl BitOr for &HybridBitset {
     type Output = HybridBitset;

    fn bitor(self, rhs: Self) -> Self::Output {
        // Fallback for Holey combinations to simplify logic
        if matches!(&self.inner, BitsetRepr::Holey { .. }) || matches!(&rhs.inner, BitsetRepr::Holey { .. }) {
            return self.bitwise_op_fallback(rhs, |b1, b2| b1 | b2);
        }

         match (&self.inner, &rhs.inner) {
            // Sparse | Sparse
            (BitsetRepr::Sparse(set1), BitsetRepr::Sparse(set2)) => {
                // Optimization: clone larger, extend with smaller
                let (larger, smaller) = if set1.len() >= set2.len() { (set1, set2) } else { (set2, set1) };
                let mut result_set = larger.clone();
                result_set.extend(smaller.iter().copied());
                // Alternative: let result_set: BTreeSet<usize> = set1.union(set2).copied().collect();
                let mut result = HybridBitset { inner: BitsetRepr::Sparse(result_set) };
                result.check_representation(); // Check if result should be Dense or Holey
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
                        cached_exact_count: RefCell::new(Some(exact_count)),
                    }
                };
                result.check_representation(); // Check if result should be Sparse or Holey
                result
            }
            // Mixed: Sparse | Dense (self is Sparse, rhs is Dense)
            (BitsetRepr::Sparse(s_set), BitsetRepr::Dense { bits: d_bits, .. }) => {
                let mut result_bits = d_bits.clone();
                let mut count_increased = false;
                for &index in s_set {
                    if index >= result_bits.len() {
                        result_bits.resize(index + 1, false);
                         // Invalidate cache as len changed
                        if let Some(count_ref) = result.get_cached_count_mut() {
                           *count_ref = None; // Mark as dirty
                        }
                        count_increased = true;
                    }
                     // Check if bit was false before setting
                    if !result_bits.get(index).map_or(true, |b| *b) {
                         result_bits.set(index, true); // Set bit to true
                         // Increment cached count if it was known
                         if let Some(count_ref) = result.get_cached_count_mut() {
                             if let Some(c) = count_ref {
                                 *c += 1;
                             }
                         }
                         count_increased = true;
                    }
                }
                 // If count increased or dense length changed, check representation
                if count_increased {
                     result.check_representation();
                }
                 result
            }
            // Mixed: Dense | Sparse (self is Dense, rhs is Sparse)
            (BitsetRepr::Dense { bits: d_bits, cached_exact_count }, BitsetRepr::Sparse(s_set)) => {
                let mut result_bits = d_bits.clone();
                 let mut count_increased = false;
                 let mut cached_count_mut = cached_exact_count.borrow_mut(); // Borrow mutably once

                for &index in s_set {
                    if index >= result_bits.len() {
                        result_bits.resize(index + 1, false);
                        // Invalidate cache as len changed
                         *cached_count_mut = None; // Mark as dirty
                         count_increased = true;
                    }
                     // Check if bit was false before setting
                    if !result_bits.get(index).map_or(true, |b| *b) {
                         result_bits.set(index, true); // Set bit to true
                         // Increment cached count if it was known
                         if let Some(c) = cached_count_mut.as_mut().and_then(|opt| opt.as_mut()) {
                             *c += 1;
                         }
                         count_increased = true;
                    }
                }
                drop(cached_count_mut); // Drop the mutable borrow

                let mut result = HybridBitset { inner: BitsetRepr::Dense { bits: result_bits, cached_exact_count: RefCell::new(None) } }; // Cache is potentially stale
                // If count increased or dense length changed, check representation
                if count_increased {
                     result.check_representation();
                }
                 result
            }
            // Holey vs other handled by fallback
             _ => unreachable!("Holey combinations should be handled by fallback"),
        }
    }
}

impl BitXor for &HybridBitset {
     type Output = HybridBitset;

    fn bitxor(self, rhs: Self) -> Self::Output {
        // Fallback for Holey combinations to simplify logic
        if matches!(&self.inner, BitsetRepr::Holey { .. }) || matches!(&rhs.inner, BitsetRepr::Holey { .. }) {
            return self.bitwise_op_fallback(rhs, |b1, b2| b1 ^ b2);
        }

         match (&self.inner, &rhs.inner) {
            // Sparse ^ Sparse
            (BitsetRepr::Sparse(set1), BitsetRepr::Sparse(set2)) => {
                let result_set: BTreeSet<usize> = set1.symmetric_difference(set2).copied().collect();
                let mut result = HybridBitset { inner: BitsetRepr::Sparse(result_set) };
                result.check_representation(); // Result size could be small or large -> conversion check
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
                        cached_exact_count: RefCell::new(Some(exact_count)),
                    }
                };
                result.check_representation(); // Result size could be small or large -> conversion check
                result
            }
            // Mixed: Sparse ^ Dense (self is Sparse, rhs is Dense)
            (BitsetRepr::Sparse(s_set), BitsetRepr::Dense { bits: d_bits, .. }) => {
                let mut result_bits = d_bits.clone();
                // Determine the maximum index from the sparse set to correctly size the result_bits.
                // BTreeSet::last() is efficient for getting the max element.
                let max_s_idx_plus_1 = s_set.last().map_or(0, |&v| v.saturating_add(1));

                let current_len = result_bits.len();
                if max_s_idx_plus_1 > current_len {
                    result_bits.resize(max_s_idx_plus_1, false);
                     // Invalidate cache as len changed
                    if let BitsetRepr::Dense{ cached_exact_count, .. } = &mut result_bits.to_owned().into_inner() { // Need to get it from the BitVec somehow... tricky.
                       // Let's just recalculate count below.
                    }
                }

                for &index in s_set {
                    // Index is guaranteed to be within the (potentially resized) bounds of result_bits.
                    let current_val = result_bits[index];
                    result_bits.set(index, !current_val); // Flip the bit
                }

                let exact_count = result_bits.count_ones();
                let mut result = HybridBitset {
                    inner: BitsetRepr::Dense {
                        bits: result_bits,
                        cached_exact_count: RefCell::new(Some(exact_count)),
                    }
                };
                result.check_representation(); // Check if result should be Sparse or Holey
                result
            }
            // Mixed: Dense ^ Sparse (self is Dense, rhs is Sparse)
            (BitsetRepr::Dense { bits: d_bits, .. }, BitsetRepr::Sparse(s_set)) => {
                let mut result_bits = d_bits.clone();
                let max_s_idx_plus_1 = s_set.last().map_or(0, |&v| v.saturating_add(1));

                let current_len = result_bits.len();
                if max_s_idx_plus_1 > current_len {
                    result_bits.resize(max_s_idx_plus_1, false);
                     // Invalidate cache
                    if let BitsetRepr::Dense{ cached_exact_count, .. } = &mut result_bits.to_owned().into_inner() {
                       // Let's just recalculate count below.
                    }
                }

                for &index in s_set {
                    let current_val = result_bits[index];
                    result_bits.set(index, !current_val); // Flip the bit
                }

                let exact_count = result_bits.count_ones();
                let mut result = HybridBitset {
                    inner: BitsetRepr::Dense {
                        bits: result_bits,
                        cached_exact_count: RefCell::new(Some(exact_count)),
                    }
                };
                result.check_representation(); // Check if result should be Sparse or Holey
                result
            }
            // Holey vs other handled by fallback
            _ => unreachable!("Holey combinations should be handled by fallback"),
        }
    }
}

// Set Difference (A - B or A \ B)
impl Sub for &HybridBitset {
    type Output = HybridBitset;

    // Computes self - rhs (elements in self but not in rhs)
    fn sub(self, rhs: Self) -> Self::Output {
        // Fallback for Holey combinations to simplify logic
        if matches!(&self.inner, BitsetRepr::Holey { .. }) || matches!(&rhs.inner, BitsetRepr::Holey { .. }) {
            return self.bitwise_op_fallback(rhs, |b1, b2| b1 & (!b2.to_bitvec())); // Need !rhs
        }

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
                        cached_exact_count: RefCell::new(Some(exact_count)),
                    }
                };
                result.check_representation(); // Check if result should be Sparse or Holey
                result
            }
            // Dense - Sparse
            (BitsetRepr::Dense { bits, cached_exact_count }, BitsetRepr::Sparse(set2)) => {
                // More efficient to clone dense and remove elements from sparse
                let mut result_bits = bits.clone();
                let mut bits_cleared = 0; // Track if count might have changed
                 let mut cached_count_mut = cached_exact_count.borrow_mut(); // Borrow mutably once

                for &index in set2 {
                    if index < result_bits.len() {
                        // Use set(.., false) which returns previous state
                        if result_bits.replace(index, false) {
                             bits_cleared += 1; // A bit was actually cleared
                        }
                    }
                }

                if bits_cleared > 0 {
                     // Update cached count if known
                     if let Some(c) = cached_count_mut.as_mut().and_then(|opt| opt.as_mut()) {
                        *c -= bits_cleared;
                    }
                    drop(cached_count_mut); // Drop the mutable borrow
                    let mut result = HybridBitset { inner: BitsetRepr::Dense { bits: result_bits, cached_exact_count: RefCell::new(None) } }; // Cache is potentially stale
                    result.check_representation(); // This will use the updated cache or recompute
                    result
                } else {
                    drop(cached_count_mut);
                    // No bits cleared, return clone of original Dense set
                    HybridBitset { inner: BitsetRepr::Dense { bits: result_bits, cached_exact_count: cached_exact_count.clone() } }
                }
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
            // Holey vs other handled by fallback
            _ => unreachable!("Holey combinations should be handled by fallback"),
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
         // Fallback for Holey combinations to simplify logic
        if matches!(&self.inner, BitsetRepr::Holey { .. }) || matches!(&rhs.inner, BitsetRepr::Holey { .. }) {
            *self = self.bitwise_op_fallback(&rhs, |b1, b2| b1 | b2);
            return;
        }

        // Optimization: If self is Dense, rhs is Sparse, can be faster
        match (&mut self.inner, &rhs.inner) {
            (BitsetRepr::Dense { bits, cached_exact_count }, BitsetRepr::Sparse(set2)) => {
                let mut new_bits_set = 0;
                 let mut cached_count_mut = cached_exact_count.borrow_mut(); // Borrow mutably once

                for &index in set2 {
                    if index >= bits.len() {
                        bits.resize(index + 1, false);
                         // Invalidate cache as len changed
                        *cached_count_mut = None; // Mark as dirty
                    }
                    if !bits[index] { // Check before setting
                        bits.set(index, true);
                        new_bits_set += 1;
                    }
                }
                if new_bits_set > 0 {
                     // Increment cached count if it was known
                    if let Some(c) = cached_count_mut.as_mut().and_then(|opt| opt.as_mut()) {
                        *c += new_bits_set;
                    }
                }
                 drop(cached_count_mut);
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
        // Fallback for Holey combinations to simplify logic
        if matches!(&self.inner, BitsetRepr::Holey { .. }) || matches!(&rhs.inner, BitsetRepr::Holey { .. }) {
            *self = self.bitwise_op_fallback(&rhs, |b1, b2| b1 & (!b2.to_bitvec()));
            return;
        }

         // Optimization: If self is Dense, rhs is Sparse, can be faster
        match (&mut self.inner, &rhs.inner) {
            (BitsetRepr::Dense { bits, cached_exact_count }, BitsetRepr::Sparse(set2)) => {
                let mut bits_cleared = 0;
                 let mut cached_count_mut = cached_exact_count.borrow_mut(); // Borrow mutably once

                for &index in set2 {
                    if index < bits.len() {
                        if bits.replace(index, false) { // Returns previous value
                            bits_cleared += 1;
                        }
                    }
                }
                if bits_cleared > 0 {
                     // Update cached count if known
                     if let Some(c) = cached_count_mut.as_mut().and_then(|opt| opt.as_mut()) {
                        *c -= bits_cleared;
                    }
                    drop(cached_count_mut);
                    self.check_representation(); // This will use the updated cache or recompute
                } else {
                     drop(cached_count_mut);
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
         // Fallback for Holey combinations to simplify logic
        if matches!(&self.inner, BitsetRepr::Holey { .. }) || matches!(&rhs.inner, BitsetRepr::Holey { .. }) {
            *self = self.bitwise_op_fallback(rhs, |b1, b2| b1 | b2);
            return;
        }

        // Optimization: If self is Dense, rhs is Sparse, can be faster
        match (&mut self.inner, &rhs.inner) {
            (BitsetRepr::Dense { bits, cached_exact_count }, BitsetRepr::Sparse(set2)) => {
                let mut new_bits_set = 0;
                 let mut cached_count_mut = cached_exact_count.borrow_mut(); // Borrow mutably once

                for &index in set2 {
                    if index >= bits.len() {
                        bits.resize(index + 1, false);
                         // Invalidate cache as len changed
                        *cached_count_mut = None; // Mark as dirty
                    }
                    if !bits[index] { // Check before setting
                        bits.set(index, true);
                        new_bits_set += 1;
                    }
                }
                if new_bits_set > 0 {
                     // Increment cached count if it was known
                    if let Some(c) = cached_count_mut.as_mut().and_then(|opt| opt.as_mut()) {
                        *c += new_bits_set;
                    }
                }
                 drop(cached_count_mut);
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
         // Fallback for Holey combinations to simplify logic
        if matches!(&self.inner, BitsetRepr::Holey { .. }) || matches!(&rhs.inner, BitsetRepr::Holey { .. }) {
            *self = self.bitwise_op_fallback(rhs, |b1, b2| b1 & (!b2.to_bitvec()));
            return;
        }

         // Optimization: If self is Dense, rhs is Sparse, can be faster
        match (&mut self.inner, &rhs.inner) {
            (BitsetRepr::Dense { bits, cached_exact_count }, BitsetRepr::Sparse(set2)) => {
                let mut bits_cleared = 0;
                 let mut cached_count_mut = cached_exact_count.borrow_mut(); // Borrow mutably once

                for &index in set2 {
                    if index < bits.len() {
                        if bits.replace(index, false) { // Returns previous value
                            bits_cleared += 1;
                        }
                    }
                }
                if bits_cleared > 0 {
                     // Update cached count if known
                    if let Some(c) = cached_count_mut.as_mut().and_then(|opt| opt.as_mut()) {
                        *c -= bits_cleared;
                    }
                    drop(cached_count_mut);
                    self.check_representation();
                } else {
                     drop(cached_count_mut);
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

impl HybridBitset {
     // Helper to get a mutable reference to cached_exact_count Option
     fn get_cached_count_mut(&self) -> Option<std::cell::RefMut<'_, Option<usize>>> {
         if let BitsetRepr::Dense { cached_exact_count, .. } = &self.inner {
             Some(cached_exact_count.borrow_mut())
         } else {
             None
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
            (BitsetRepr::Holey { max_index: m1, missing: mm1 }, BitsetRepr::Holey { max_index: m2, missing: mm2 }) => {
                 // If max_index is different, or the number of missing elements is different, they aren't equal.
                 // Also, if one missing set is empty and the other isn't, they can't be equal if max_index > 0.
                 if m1 != m2 || mm1.len() != mm2.len() {
                     return false;
                 }
                 // If max_index and missing counts are the same, check if the missing sets are the same.
                 mm1 == mm2
            }
            (BitsetRepr::Dense { bits: b1, cached_exact_count: c1_rc },
             BitsetRepr::Dense { bits: b2, cached_exact_count: c2_rc }) => {
                // Try to use cached counts for an early exit if they are known and different.
                let c1_opt_borrow = c1_rc.borrow();
                let c2_opt_borrow = c2_rc.borrow();
                if let (Some(count1), Some(count2)) = (*c1_opt_borrow, *c2_opt_borrow) {
                    if count1 != count2 {
                        // Drop borrows before returning
                        drop(c1_opt_borrow);
                        drop(c2_opt_borrow);
                        return false;
                    }
                    // If counts are known and equal, we still need to compare bits.
                }
                // Drop borrows if not already dropped
                drop(c1_opt_borrow);
                drop(c2_opt_borrow);

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
        match self.inner {
            BitsetRepr::Sparse(set) => {
                if set.is_empty() {
                    bitvec![]
                } else {
                    let max_index = set.last().copied().unwrap_or(0);
                    let mut bits = bitvec![usize, Lsb0; 0; max_index + 1];
                    for index in set {
                        bits.set(index, true);
                    }
                    bits
                }
            }
            BitsetRepr::Holey { max_index, missing } => {
                let mut bits = bitvec![usize, Lsb0; 1; max_index + 1];
                for index in missing {
                     if index <= max_index { // Ensure index is within the declared range
                        bits.set(index, false);
                     }
                }
                bits
            }
            BitsetRepr::Dense { bits, .. } => {
                 bits // Just return the underlying BitVec
            }
        }
    }
}

impl From<BitVec> for HybridBitset {
    // Convert a BitVec into a HybridBitset
    fn from(bitvec: BitVec) -> Self {
        // Decide whether to convert to Sparse, Holey, or Dense based on content
        let count = bitvec.count_ones();
        let len = bitvec.len();

        if count == 0 {
            // Empty bitvec should be Sparse
            HybridBitset::new()
        } else if len > 0 && (len - count) < HOLEY_TO_SPARSE_THRESHOLD && len >= DENSE_TO_HOLEY_THRESHOLD {
            // High density and large enough - convert to Holey
            let max_index = len - 1;
            let mut missing = BTreeSet::new();
            for (index, bit) in bitvec.iter().enumerate() {
                if !*bit {
                    missing.insert(index);
                }
            }
            HybridBitset { inner: BitsetRepr::Holey { max_index, missing } }
        } else if count < SPARSE_TO_DENSE_THRESHOLD {
            // Low count - convert to Sparse
            let mut set = BTreeSet::new();
            for index in bitvec.iter_ones() {
                set.insert(index);
            }
            HybridBitset { inner: BitsetRepr::Sparse(set) }
        }
         else {
            // Default to Dense
            HybridBitset {
                inner: BitsetRepr::Dense {
                    bits: bitvec,
                    cached_exact_count: RefCell::new(Some(count)),
                }
            }
        }
    }
}

impl Index<usize> for HybridBitset {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
         match &self.inner {
             BitsetRepr::Sparse(set) => {
                 if set.contains(&index) { &true } else { &false }
             }
             BitsetRepr::Holey { max_index, missing } => {
                 if index <= *max_index && !missing.contains(&index) { &true } else { &false }
             }
             BitsetRepr::Dense { bits, .. } => {
                 // bitvec's get() returns Option<&bool>, need to handle bounds.
                 // For Index, panic on out of bounds is standard.
                 // Use bitvec's index impl directly which handles bounds.
                 &bits[index]
             }
         }
    }
}

impl IndexMut<usize> for HybridBitset {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
         // IndexMut is trickier because we need to potentially change the representation
         // on set. We'll implement set() and use that internally.
         // A simple approach for IndexMut is to get the current value,
         // return a mutable reference to a temporary static bool, and then
         // apply the change after the reference is dropped (which is not possible
         // with the standard trait signature).
         // The standard IndexMut requires returning a reference into self,
         // which is hard with our dynamic representation and potential reallocations/conversions.
         // A common pattern is to not implement IndexMut for structures with complex internal
         // mutations like this, and instead provide a `set(index, value)` method.
         // However, if we must, we can force a conversion to Dense which supports IndexMut,
         // but this might be inefficient for Sparse/Holey.

         // Forcing Dense to support IndexMut is the most straightforward approach
         // that fits the trait signature requirements without complex lifetime issues.
         // It means indexing into a Sparse/Holey set mutably will convert it to Dense.

         self.ensure_dense();
         if let BitsetRepr::Dense { bits, .. } = &mut self.inner {
             // Ensure dense BitVec is large enough if index is out of bounds
             if index >= bits.len() {
                  bits.resize(index + 1, false);
                  // Invalidate cached count as len changed
                  if let Some(count_ref) = self.get_cached_count_mut() {
                      *count_ref = None;
                  }
             }
             // bitvec's IndexMut impl handles the mutable reference
             &mut bits[index]
         } else {
             // This should not happen after ensure_dense()
             panic!("Internal error: Failed to convert to Dense for IndexMut");
         }
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
         if let BitsetRepr::Dense { cached_exact_count, .. } = &set.inner {
             assert_eq!(*cached_exact_count.borrow(), Some(SPARSE_TO_DENSE_THRESHOLD));
         } else {
             panic!("Expected Dense representation");
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
        // At this point, len() might not have been called, cached count might be updated approximately
        // Call len() to force recalculation and caching if needed before checking state
        assert_eq!(set.len(), DENSE_TO_SPARSE_THRESHOLD);
        assert!(matches!(set.inner, BitsetRepr::Dense { .. }), "Should still be Dense at threshold");
         if let BitsetRepr::Dense { cached_exact_count, .. } = &set.inner {
             assert_eq!(*cached_exact_count.borrow(), Some(DENSE_TO_SPARSE_THRESHOLD));
         } else {
              panic!("Expected Dense representation");
         }


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
        if let BitsetRepr::Dense { cached_exact_count, .. } = &set.inner {
            assert_eq!(*cached_exact_count.borrow(), Some(SPARSE_TO_DENSE_THRESHOLD));
        }


        // Insert existing - cached count shouldn't change if known
        set.insert(10);
        assert_eq!(set.len(), SPARSE_TO_DENSE_THRESHOLD);
         if let BitsetRepr::Dense { cached_exact_count, .. } = &set.inner {
             assert_eq!(*cached_exact_count.borrow(), Some(SPARSE_TO_DENSE_THRESHOLD));
         }


        // Insert new
        set.insert(SPARSE_TO_DENSE_THRESHOLD + 100); // Also tests resize
        assert_eq!(set.len(), SPARSE_TO_DENSE_THRESHOLD + 1); // len() forces exact count check
         if let BitsetRepr::Dense { cached_exact_count, .. } = &set.inner {
             assert_eq!(*cached_exact_count.borrow(), Some(SPARSE_TO_DENSE_THRESHOLD + 1));
         }


         // Remove existing
         set.remove(0);
         assert_eq!(set.len(), SPARSE_TO_DENSE_THRESHOLD);
          if let BitsetRepr::Dense { cached_exact_count, .. } = &set.inner {
             assert_eq!(*cached_exact_count.borrow(), Some(SPARSE_TO_DENSE_THRESHOLD));
         }


         // Remove non-existing (within bounds)
         set.remove(1); // Was present, now removed
         set.remove(1); // Try removing again
         assert_eq!(set.len(), SPARSE_TO_DENSE_THRESHOLD-1);
          if let BitsetRepr::Dense { cached_exact_count, .. } = &set.inner {
             assert_eq!(*cached_exact_count.borrow(), Some(SPARSE_TO_DENSE_THRESHOLD -1));
         }


         // Remove non-existing (out of bounds)
         set.remove(999999);
         assert_eq!(set.len(), SPARSE_TO_DENSE_THRESHOLD-1);
          if let BitsetRepr::Dense { cached_exact_count, .. } = &set.inner {
             assert_eq!(*cached_exact_count.borrow(), Some(SPARSE_TO_DENSE_THRESHOLD -1));
         }
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
        assert!(matches!(union.inner, BitsetRepr::Dense { .. })); // Union is large -> Dense or Holey
         // Union size is N+20. If N+20 >= DENSE_TO_HOLEY_THRESHOLD * 9/10, it should be Holey.
         // 128+20=148. 9*128=1152. 148 < 1152. So it should be Dense.
        assert!(matches!(union.inner, BitsetRepr::Dense { .. }));


        // Difference is small -> Sparse
        assert!(matches!(difference.inner, BitsetRepr::Sparse { .. }));
        // Sym Diff might be sparse or dense depending on threshold and exact values
         if sym_diff_expected.len() < DENSE_TO_SPARSE_THRESHOLD {
             assert!(matches!(sym_diff.inner, BitsetRepr::Sparse { .. }));
        } else {
             // Sym Diff is size 5 + 10 = 15. 15 < DENSE_TO_SPARSE_THRESHOLD. Should be Sparse.
             assert!(matches!(sym_diff.inner, BitsetRepr::Sparse { .. }));
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
        // Result size is (N+5) + 1 = N+6. 128+6 = 134.
        // 134 >= SPARSE_TO_DENSE_THRESHOLD. Should be Dense.
        // Max index is N+100. Density is (N+6)/(N+101). (134)/(229) approx 0.58.
        // 0.58*10 < 9. So not Holey.
        assert!(matches!(union1.inner, BitsetRepr::Dense { .. }));

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
        // Result is size (N+5) - 3 = N+2. 128+2=130. >= SPARSE_TO_DENSE_THRESHOLD.
        // Max index is N+4. Density is (N+2)/(N+5). (130)/(133) approx 0.97.
        // 0.97 * 10 > 9. Should be Holey.
        assert!(matches!(diff2.inner, BitsetRepr::Holey { .. }));


        // --- Sparse ^ Dense ---
        let xor1 = &set1_sparse ^ &set2_dense; // {0, 4..N+5} U {N+100}
        let mut xor1_expected = diff2_expected.clone(); // Elements only in Dense
        xor1_expected.insert(SPARSE_TO_DENSE_THRESHOLD + 100); // Element only in Sparse
        assert_eq!(xor1.iter().collect::<BTreeSet<usize>>(), xor1_expected);
        // Result size is (N+2) + 1 = N+3. 128+3=131. >= SPARSE_TO_DENSE_THRESHOLD.
        // Max index is N+100. Density is (N+3)/(N+101). (131)/(229) approx 0.57.
        // 0.57 * 10 < 9. Should be Dense.
        assert!(matches!(xor1.inner, BitsetRepr::Dense { .. }));

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
         // Create the same set and force it to be holey for comparison
        let mut set1_h = HybridBitset::ones(10); // Holey { max_index: 10, missing: {0, 2, 3, 4, 6, 7, 8, 9} }
        set1_h.insert(0); set1_h.insert(2); set1_h.insert(3); set1_h.insert(4);
        set1_h.insert(6); set1_h.insert(7); set1_h.insert(8); set1_h.insert(9);
        // Now set1_h should be Holey { max_index: 10, missing: {0,2,3,4,6,7,8,9} }
        // The set bits are 1, 5, 10.
        assert!(matches!(set1_h.inner, BitsetRepr::Holey { .. }));


        let set2_s = HybridBitset::from_iter(vec![1, 5, 11]); // Sparse, different
        let mut set3_d_empty = HybridBitset::from_iter(0..SPARSE_TO_DENSE_THRESHOLD); // Dense
        set3_d_empty.clear(); // Now sparse empty
        set3_d_empty.ensure_dense(); // Force dense empty

        assert!(matches!(set1_s.inner, BitsetRepr::Sparse(_)));
        // Check that set1_d actually became dense
        assert!(matches!(set1_d.inner, BitsetRepr::Dense { .. }), "set1_d should be dense");
        assert_eq!(set1_d.iter().collect::<Vec<_>>(), vec![1, 5, 10]);
         assert_eq!(set1_h.iter().collect::<Vec<_>>(), vec![1, 5, 10]);


        // Test equality
        assert_eq!(set1_s, set1_s);
        assert_eq!(set1_d, set1_d);
        assert_eq!(set1_h, set1_h);
        assert_eq!(set1_s, set1_d, "Sparse and Dense representations of the same set should be equal");
        assert_eq!(set1_d, set1_s, "Equality should be symmetric");
         assert_eq!(set1_s, set1_h, "Sparse and Holey representations of the same set should be equal");
         assert_eq!(set1_h, set1_s, "Equality should be symmetric");
         assert_eq!(set1_d, set1_h, "Dense and Holey representations of the same set should be equal");
         assert_eq!(set1_h, set1_d, "Equality should be symmetric");


        assert_ne!(set1_s, set2_s);
        assert_ne!(set1_d, set2_s);
         assert_ne!(set1_h, set2_s);

        assert_ne!(set1_s, set3_d_empty);
        assert_ne!(set1_d, set3_d_empty);
         assert_ne!(set1_h, set3_d_empty);

        // Test hashing
        use std::collections::hash_map::DefaultHasher;
        let hash = |s: &HybridBitset| -> u64 {
            let mut hasher = DefaultHasher::new();
            s.hash(&mut hasher);
            hasher.finish()
        };

        let hash1_s = hash(&set1_s);
        let hash1_d = hash(&set1_d);
        let hash1_h = hash(&set1_h);
        let hash2_s = hash(&set2_s);
        let hash3_d = hash(&set3_d_empty);


        assert_eq!(hash1_s, hash1_d, "Hashes of Sparse and Dense representations of the same set should be equal");
        assert_eq!(hash1_s, hash1_h, "Hashes of Sparse and Holey representations of the same set should be equal");
        assert_eq!(hash1_d, hash1_h, "Hashes of Dense and Holey representations of the same set should be equal");

        assert_ne!(hash1_s, hash2_s);
        assert_ne!(hash1_d, hash2_s);
        assert_ne!(hash1_h, hash2_s);

        assert_ne!(hash1_s, hash3_d);


        // Test in BTreeSet
        let mut map = BTreeSet::new();
        map.insert(set1_s.clone());
        assert!(map.contains(&set1_s));
        assert!(map.contains(&set1_d), "BTreeSet should find equivalent Dense set using Sparse key's hash");
         assert!(map.contains(&set1_h), "BTreeSet should find equivalent Holey set using Sparse key's hash");


        map.insert(set1_d.clone()); // Should replace the previous one or do nothing
        assert_eq!(map.len(), 1);

        map.insert(set1_h.clone()); // Should replace the previous one or do nothing
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


        // It should remain sparse after removal as well
        set.remove(large_idx);
        assert_eq!(set.len(), 1);
        assert!(set.contains(0));
        assert!(!set.contains(large_idx));
        assert!(matches!(set.inner, BitsetRepr::Sparse(_)), "Should still be sparse");

         // Test insertion of a large index into a Dense set, forcing resize
         let mut dense_set = HybridBitset::from_iter(0..SPARSE_TO_DENSE_THRESHOLD);
         assert!(matches!(dense_set.inner, BitsetRepr::Dense { .. }));
         let large_idx_dense = SPARSE_TO_DENSE_THRESHOLD + 1000;
         dense_set.insert(large_idx_dense);
         // Should still be Dense, but larger
         assert!(matches!(dense_set.inner, BitsetRepr::Dense { bits, .. }));
         if let BitsetRepr::Dense{ bits, ..} = &dense_set.inner {
             assert!(bits.len() > large_idx_dense);
         }
         assert_eq!(dense_set.len(), SPARSE_TO_DENSE_THRESHOLD + 1);
         assert!(dense_set.contains(large_idx_dense));
         assert!(dense_set.contains(0));
         assert!(!dense_set.contains(large_idx_dense-1)); // Assuming large_idx_dense is > original dense size

         // Test insertion into Holey that extends max_index
         let mut holey_set = HybridBitset::ones(DENSE_TO_HOLEY_THRESHOLD); // Holey
         assert!(matches!(holey_set.inner, BitsetRepr::Holey { .. }));
         let large_idx_holey = DENSE_TO_HOLEY_THRESHOLD + 500;
         holey_set.insert(large_idx_holey);
         // Should still be Holey, max_index increased, intermediate values are missing
         assert!(matches!(holey_set.inner, BitsetRepr::Holey { max_index, missing }));
         assert_eq!(*max_index, large_idx_holey);
         // Check some intermediate indices are now missing
         assert!(missing.contains(&DENSE_TO_HOLEY_THRESHOLD));
         assert!(missing.contains(& (large_idx_holey - 1) ));
         // Count should be (old_max + 1 - missing before) + 1 - (new missing)
         let expected_len = (DENSE_TO_HOLEY_THRESHOLD + 1) + 1 - (large_idx_holey - DENSE_TO_HOLEY_THRESHOLD);
         assert_eq!(holey_set.len(), expected_len);
         assert!(holey_set.contains(large_idx_holey));
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

        let mut set3 = HybridBitset::ones(DENSE_TO_HOLEY_THRESHOLD); // Holey
        assert!(matches!(set3.inner, BitsetRepr::Holey { .. }));
        assert!(!set3.is_empty());
        set3.clear();
        assert!(set3.is_empty());
        assert_eq!(set3.len(), 0);
        assert!(matches!(set3.inner, BitsetRepr::Sparse(_)), "Clear should reset to Sparse empty");
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
        // The operation was Dense &= Dense. The result size is 64.
        // The Dense & Dense path creates a Dense result initially. check_representation checks
        // if 64 < DENSE_TO_SPARSE_THRESHOLD (64), which is false. So it should stay Dense.
        // Wait, check_representation after Dense & Dense should be called on the RESULT.
        // Result count 64. 64 < DENSE_TO_SPARSE_THRESHOLD. Should convert to Sparse.
        assert!(matches!(set3.inner, BitsetRepr::Sparse { .. }), "Result of Dense &= Dense with size 64 should be Sparse");

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
         // Result count 64. 64 < DENSE_TO_SPARSE_THRESHOLD. Should convert to Sparse.
        assert!(matches!(set3.inner, BitsetRepr::Sparse { .. }), "Result of Dense &= Dense with size 64 should be Sparse");

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
        let mut test_dense = HybridBitset::new();
        test_dense.insert(1);
        test_dense.insert(3); // sparse: {1, 3}
        test_dense.convert_to_dense(); // dense: bits for [0,1,0,1], len=4
                                       // cached_exact_count should be Some(2) now

        let expected_dense_bools = vec![false, true, false, true];
        assert_eq!(test_dense.iter_bools().collect::<Vec<bool>>(), expected_dense_bools);
        assert_eq!(test_dense.iter_bools().len(), expected_dense_bools.len());

        // Test with an empty dense set
        let mut empty_dense_set = HybridBitset::new();
        empty_dense_set.convert_to_dense(); // Becomes Dense { bits: [], cached_exact_count: Some(0) }
        assert_eq!(empty_dense_set.iter_bools().collect::<Vec<bool>>(), Vec::<bool>::new());
        assert_eq!(empty_dense_set.iter_bools().len(), 0);

         // Test with a Holey set
         let mut test_holey = HybridBitset::ones(10); // 0..=10 true
         test_holey.remove(1);
         test_holey.remove(3); // Holey { max_index: 10, missing: {1, 3} }
         let expected_holey_bools = vec![true, false, true, false, true, true, true, true, true, true, true];
         assert_eq!(test_holey.iter_bools().collect::<Vec<bool>>(), expected_holey_bools);
         assert_eq!(test_holey.iter_bools().len(), expected_holey_bools.len());

         // Test with an empty Holey set (should convert to Sparse empty)
         let mut empty_holey = HybridBitset::ones(10);
         for i in 0..=10 {
             empty_holey.remove(i); // Remove all, should become Sparse empty
         }
         assert!(matches!(empty_holey.inner, BitsetRepr::Sparse(_))); // Verify conversion
         assert_eq!(empty_holey.iter_bools().collect::<Vec<bool>>(), Vec::<bool>::new());
         assert_eq!(empty_holey.iter_bools().len(), 0);
    }

     #[test]
    fn holey_representation_basic() {
        let mut h = HybridBitset::ones(1000);     // everything 0..=1000 is true
        h.remove(37); h.remove(123);
        assert!(matches!(h.inner, BitsetRepr::Holey { max_index: m, missing: miss } if *m == 1000 && miss.contains(&37) && miss.contains(&123) && miss.len() == 2));
        assert_eq!(h.len(), 999);                 // (1000 + 1) – 2 holes = 1001 - 2 = 999
        assert!(h.contains(1000));
        assert!(!h.contains(37));
        assert!(!h.contains(123));
        assert!(h.contains(0));
        assert!(h.contains(1001)); // Index out of bounds is false

         // Add an element that was missing
         h.insert(37);
         assert!(h.contains(37));
         assert_eq!(h.len(), 1000); // Now 1001 - 1 hole
         assert!(matches!(h.inner, BitsetRepr::Holey { max_index: m, missing: miss } if *m == 1000 && !miss.contains(&37) && miss.contains(&123) && miss.len() == 1));

         // Add an element outside the max_index
         let large_idx = 10000;
         h.insert(large_idx);
         assert!(h.contains(large_idx));
         assert_eq!(h.len(), (large_idx + 1 - 1) - (large_idx - 1000 - 1) ); // old_len + 1 - new_missing_gap
         // Recalculate expected len more carefully:
         // Before: range 0..=1000, missing {123}. Len = 1001 - 1 = 1000. Max index = 1000.
         // Insert 10000. New max_index = 10000.
         // Missing indices in the new range 0..=10000 are:
         // Old missing: {123}
         // New missing in gap 1001..=9999: {1001, 1002, ..., 9999}. Count = 9999 - 1001 + 1 = 8999.
         // Total missing = {123} U {1001..=9999}. Count = 1 + 8999 = 9000.
         // New len = (10000 + 1) - 9000 = 10001 - 9000 = 1001.
         assert_eq!(h.len(), 1001);

         assert!(matches!(h.inner, BitsetRepr::Holey { max_index: m, missing: miss } if *m == large_idx && miss.contains(&123) && miss.contains(&1001) && miss.contains(&9999) && !miss.contains(&large_idx) && miss.len() == 9000));

         // Remove an element that was already missing (no change)
         assert!(!h.remove(123));
         assert_eq!(h.len(), 1001); // No change
         assert!(matches!(h.inner, BitsetRepr::Holey { max_index: m, missing: miss } if *m == large_idx && miss.len() == 9000));

         // Remove an element within the implicit range that was present
         assert!(h.remove(50)); // 50 was in 0..=1000 and not missing
         assert!(!h.contains(50));
         assert_eq!(h.len(), 1000); // Count decreases
          assert!(matches!(h.inner, BitsetRepr::Holey { max_index: m, missing: miss } if *m == large_idx && miss.contains(&50) && miss.len() == 9001));

         // Check conversion from Holey to Sparse if missing set grows large
         let mut large_missing_holey = HybridBitset::ones(DENSE_TO_HOLEY_THRESHOLD); // Holey
         assert!(matches!(large_missing_holey.inner, BitsetRepr::Holey { .. }));
         // Remove elements until missing set exceeds threshold
         for i in 0..HOLEY_TO_SPARSE_THRESHOLD + 5 {
             large_missing_holey.remove(i * 2); // Remove some even indices
         }
          assert!(matches!(large_missing_holey.inner, BitsetRepr::Sparse(_)), "Holey should convert to Sparse if missing set is too large");
          assert_eq!(large_missing_holey.len(), (DENSE_TO_HOLEY_THRESHOLD + 1) - (HOLEY_TO_SPARSE_THRESHOLD + 5));

          // Check conversion from Holey to Dense if density drops too much
           let mut low_density_holey = HybridBitset::ones(DENSE_TO_HOLEY_THRESHOLD * 2); // Large Holey
           assert!(matches!(low_density_holey.inner, BitsetRepr::Holey { .. }));
           // Remove a large block to reduce density
           for i in (DENSE_TO_HOLEY_THRESHOLD..DENSE_TO_HOLEY_THRESHOLD + SPARSE_TO_DENSE_THRESHOLD).rev() {
               low_density_holey.remove(i);
           }
           // Density check is count * 2 < max_index + 1
           // Max index is DENSE_TO_HOLEY_THRESHOLD * 2. Count is (DENSE_TO_HOLEY_THRESHOLD * 2 + 1) - (SPARSE_TO_DENSE_THRESHOLD).
           // If (DENSE_TO_HOLEY_THRESHOLD * 2 + 1 - SPARSE_TO_DENSE_THRESHOLD) * 2 < DENSE_TO_HOLEY_THRESHOLD * 2 + 1, convert to Dense.
           // Example: DENSE_TO_HOLEY_THRESHOLD = 128*9 = 1152. SPARSE_TO_DENSE_THRESHOLD = 128.
           // Max index = 2304. Count = (2305 - 128) = 2177.
           // 2177 * 2 = 4354. Max_index + 1 = 2305. 4354 > 2305. Does not convert to Dense yet.
           // Need to remove more elements to drop density below 50%.
            for i in (DENSE_TO_HOLEY_THRESHOLD + SPARSE_TO_DENSE_THRESHOLD)..DENSE_TO_HOLEY_THRESHOLD * 2 {
                low_density_holey.remove(i);
            }
            // Now remove a block such that count is < (max_index+1)/2
            let target_count = (DENSE_TO_HOLEY_THRESHOLD * 2 + 1) / 2 - 10;
            let removed_count = (DENSE_TO_HOLEY_THRESHOLD * 2 + 1) - target_count;
            for i in (DENSE_TO_HOLEY_THRESHOLD * 2 - removed_count)..DENSE_TO_HOLEY_THRESHOLD * 2 {
                low_density_holey.remove(i);
            }
             assert!(matches!(low_density_holey.inner, BitsetRepr::Dense { .. }), "Holey should convert to Dense if density is too low");
    }

     #[test]
    fn test_ones() {
        // Small value -> Sparse
        let small_ones = HybridBitset::ones(10); // 0..=10, count 11
        assert!(matches!(small_ones.inner, BitsetRepr::Sparse(_)));
        assert_eq!(small_ones.len(), 11);
        assert!(small_ones.contains(0));
        assert!(small_ones.contains(10));
        assert!(!small_ones.contains(11));

        // Value above SPARSE_TO_DENSE_THRESHOLD but below DENSE_TO_HOLEY_THRESHOLD -> Dense
        let medium_ones = HybridBitset::ones(SPARSE_TO_DENSE_THRESHOLD + 10); // 0..=N+10, count N+11
        assert!(matches!(medium_ones.inner, BitsetRepr::Dense { .. }));
        assert_eq!(medium_ones.len(), SPARSE_TO_DENSE_THRESHOLD + 11);
        assert!(medium_ones.contains(0));
        assert!(medium_ones.contains(SPARSE_TO_DENSE_THRESHOLD + 10));
        assert!(!medium_ones.contains(SPARSE_TO_DENSE_THRESHOLD + 11));

        // Large value >= DENSE_TO_HOLEY_THRESHOLD * 2 (or just >= DENSE_TO_HOLEY_THRESHOLD depending on strategy) -> Holey
        let large_ones = HybridBitset::ones(DENSE_TO_HOLEY_THRESHOLD * 2); // 0..=D*2, count D*2+1
        assert!(matches!(large_ones.inner, BitsetRepr::Holey { max_index, missing }));
        assert_eq!(*max_index, DENSE_TO_HOLEY_THRESHOLD * 2);
        assert!(missing.is_empty());
        assert_eq!(large_ones.len(), DENSE_TO_HOLEY_THRESHOLD * 2 + 1);
        assert!(large_ones.contains(0));
        assert!(large_ones.contains(DENSE_TO_HOLEY_THRESHOLD * 2));
        assert!(!large_ones.contains(DENSE_TO_HOLEY_THRESHOLD * 2 + 1));
    }

    #[test]
    fn test_into_from_bitvec() {
        let sparse_set = HybridBitset::from_iter(vec![1, 5, 10]);
        let bitvec_from_sparse: BitVec = sparse_set.into();
        assert_eq!(bitvec_from_sparse.len(), 11); // Max index 10 + 1
        assert_eq!(bitvec_from_sparse.count_ones(), 3);
        assert!(!bitvec_from_sparse[0]); assert!(bitvec_from_sparse[1]); !bitvec_from_sparse[2..=4].any(); assert!(bitvec_from_sparse[5]); !bitvec_from_sparse[6..=9].any(); assert!(bitvec_from_sparse[10]);

        let dense_set = HybridBitset::from_iter(0..SPARSE_TO_DENSE_THRESHOLD);
        let bitvec_from_dense: BitVec = dense_set.into();
        assert_eq!(bitvec_from_dense.len(), SPARSE_TO_DENSE_THRESHOLD);
        assert_eq!(bitvec_from_dense.count_ones(), SPARSE_TO_DENSE_THRESHOLD);

        let holey_set = HybridBitset::ones(100);
        holey_set.remove(10);
        holey_set.remove(20);
        let bitvec_from_holey: BitVec = holey_set.into();
        assert_eq!(bitvec_from_holey.len(), 101); // Max index 100 + 1
        assert_eq!(bitvec_from_holey.count_ones(), 99); // 101 - 2 missing
        assert!(bitvec_from_holey[0]); assert!(!bitvec_from_holey[10]); assert!(!bitvec_from_holey[20]); assert!(bitvec_from_holey[100]);

        // Test From<BitVec>
        let bv_sparse: BitVec = bitvec![usize, Lsb0; 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]; // Corresponds to {1, 5, 10}
        let set_from_bv_sparse: HybridBitset = bv_sparse.into();
        assert!(matches!(set_from_bv_sparse.inner, BitsetRepr::Sparse(_))); // Should be Sparse (count 3 < 128)
        assert_eq!(set_from_bv_sparse.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 5, 10]));

        let bv_dense: BitVec = bitvec![usize, Lsb0; 1; SPARSE_TO_DENSE_THRESHOLD];
        let set_from_bv_dense: HybridBitset = bv_dense.into();
         // Count is 128, len is 128. Density 100%. Len 128 < DENSE_TO_HOLEY_THRESHOLD*9. Should be Dense.
        assert!(matches!(set_from_bv_dense.inner, BitsetRepr::Dense { .. }));
        assert_eq!(set_from_bv_dense.len(), SPARSE_TO_DENSE_THRESHOLD);

        let mut bv_holey: BitVec = bitvec![usize, Lsb0; 1; DENSE_TO_HOLEY_THRESHOLD + 10]; // len 1162
        bv_holey.set(10, false); bv_holey.set(20, false); // 2 missing
        let set_from_bv_holey: HybridBitset = bv_holey.into();
         // len 1162, count 1160. len > 0 && (len - count) < HOLEY_TO_SPARSE_THRESHOLD (2 < 64) && len >= DENSE_TO_HOLEY_THRESHOLD (1162 >= 1152). Should be Holey.
        assert!(matches!(set_from_bv_holey.inner, BitsetRepr::Holey { .. }));
        assert_eq!(set_from_bv_holey.len(), 1160);
        assert!(!set_from_bv_holey.contains(10));
        assert!(!set_from_bv_holey.contains(20));
        assert!(set_from_bv_holey.contains(0));
        assert!(set_from_bv_holey.contains(DENSE_TO_HOLEY_THRESHOLD + 9)); // last index set
    }

     #[test]
    fn test_index_ops() {
        let mut set = HybridBitset::new(); // Sparse
        set.insert(5);
        set.insert(10);

        assert_eq!(set[5], true);
        assert_eq!(set[10], true);
        assert_eq!(set[0], false);
        assert_eq!(set[6], false);
        // Indexing out of bounds returns false for Sparse/Holey, but panics for Dense.
        // Our Index impl forces Dense for mutable access, so immutable access should match Dense behavior.
        // Bitvec's Index impl panics on out of bounds.
         // assert_eq!(set[100], false); // This will panic with the current Index impl

        // Force to Dense
        set.insert(SPARSE_TO_DENSE_THRESHOLD);
        assert!(matches!(set.inner, BitsetRepr::Dense { .. }));
        assert_eq!(set[5], true);
        assert_eq!(set[10], true);
        assert_eq!(set[0], false);
        assert_eq!(set[SPARSE_TO_DENSE_THRESHOLD], true);
         // Indexing out of bounds panics for BitVec
         // set[100000]; // This should panic

        // Mutable access (forces Dense)
        let mut set2 = HybridBitset::from_iter(vec![1, 3]); // Sparse
        set2[1] = false; // set(1, false)
        assert!(!set2.contains(1));
        assert_eq!(set2.len(), 1);
        assert!(set2.contains(3));
        // set2 is still sparse because count (1) is low
        assert!(matches!(set2.inner, BitsetRepr::Sparse(_)));

        set2[50] = true; // set(50, true)
        assert!(set2.contains(50));
        assert_eq!(set2.len(), 2);
         // Still sparse

         set2[SPARSE_TO_DENSE_THRESHOLD] = true; // set(N, true)
         // Should convert to Dense
         assert!(matches!(set2.inner, BitsetRepr::Dense { .. }));
         assert_eq!(set2.len(), 3); // 3, 50, N are set
         assert!(set2.contains(3));
         assert!(set2.contains(50));
         assert!(set2.contains(SPARSE_TO_DENSE_THRESHOLD));
         assert!(!set2.contains(0));
         assert!(!set2.contains(1));
    }

     #[test]
    fn test_set_method() {
        let mut set = HybridBitset::new(); // Sparse
        set.set(5, true);
        assert!(set.contains(5));
        assert_eq!(set.len(), 1);

        set.set(5, true); // Setting already set
        assert!(set.contains(5));
        assert_eq!(set.len(), 1);

        set.set(5, false); // Unset
        assert!(!set.contains(5));
        assert_eq!(set.len(), 0);

        set.set(5, false); // Unsetting already unset
        assert!(!set.contains(5));
        assert_eq!(set.len(), 0);

        // Test on Dense
        let mut dense_set = HybridBitset::from_iter(0..SPARSE_TO_DENSE_THRESHOLD);
        assert!(matches!(dense_set.inner, BitsetRepr::Dense { .. }));
        assert!(dense_set.contains(10));
        dense_set.set(10, false);
        assert!(!dense_set.contains(10));
        assert_eq!(dense_set.len(), SPARSE_TO_DENSE_THRESHOLD - 1);
        // Should still be Dense as len is still >= 64.

        dense_set.set(SPARSE_TO_DENSE_THRESHOLD + 100, true); // Insert outside bounds
        assert!(dense_set.contains(SPARSE_TO_DENSE_THRESHOLD + 100));
         assert!(matches!(dense_set.inner, BitsetRepr::Dense { .. })); // Should still be Dense, resized

        // Test on Holey
        let mut holey_set = HybridBitset::ones(100);
        holey_set.remove(10); // Make it Holey
        assert!(matches!(holey_set.inner, BitsetRepr::Holey { .. }));

        holey_set.set(10, true); // Set a missing element
        assert!(holey_set.contains(10));
        assert_eq!(holey_set.len(), 101); // Should be full 0..=100
        assert!(matches!(holey_set.inner, BitsetRepr::Holey { max_index: m, missing: miss } if *m == 100 && miss.is_empty()));

        holey_set.set(50, false); // Unset a present element in Holey
        assert!(!holey_set.contains(50));
        assert_eq!(holey_set.len(), 100); // 101 - 1 missing
        assert!(matches!(holey_set.inner, BitsetRepr::Holey { max_index: m, missing: miss } if *m == 100 && miss.contains(&50) && miss.len() == 1));

        holey_set.set(1000, true); // Set outside max_index
         assert!(holey_set.contains(1000));
          assert!(matches!(holey_set.inner, BitsetRepr::Holey { max_index: m, missing: miss } if *m == 1000 && miss.contains(&50) && miss.contains(&101) && miss.contains(&999)));
          // Len check calculation similar to test_large_index
         let expected_len = (100 + 1 - 1) + 1 - (1000 - 100 - 1); // Old len (100) + 1 - gap
          // Old len was 100. New max_index 1000. Gap 101..999. Gap size 999 - 101 + 1 = 899.
          // Missing: {50} U {101..999}. Total missing 1 + 899 = 900.
          // New len: (1000 + 1) - 900 = 1001 - 900 = 101.
          assert_eq!(holey_set.len(), 101);
    }
}
