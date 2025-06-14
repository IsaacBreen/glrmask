#![allow(dead_code)] // Allow unused code for the example

use range_set_blaze::{MultiwayRangeSetBlaze, RangeSetBlaze}; // Import RangeSetBlaze
use std::convert::TryInto;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher}; // Added
use std::iter::FromIterator; // Needed for collect into BTreeSet in tests
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index, IndexMut, RangeInclusive, Sub, SubAssign};
use crate::json_serialization::{JSONConvertible, JSONNode}; // Added

// --- The Hybrid Bitset Struct ---
// Ord and PartialOrd will now rely on RangeSetBlaze's implementation (lexicographical on ranges)
#[derive(Clone, Ord, PartialOrd, Eq)]
pub struct HybridBitset {
    inner: RangeSetBlaze<usize>,
}

impl JSONConvertible for HybridBitset {
    fn to_json(&self) -> JSONNode {
        // Serialize as an array of [start, end] inclusive ranges
        let ranges_vec: Vec<Vec<usize>> = self.inner.ranges().map(|range_inclusive| {
            vec![*range_inclusive.start(), *range_inclusive.end()]
        }).collect();
        ranges_vec.to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let ranges_vec: Vec<Vec<usize>> = Vec::from_json(node)?;
        let mut ranges = Vec::new();
        for mut range_vec in ranges_vec {
            if range_vec.len() != 2 {
                return Err(format!("Expected 2-element array for HybridBitset range, got {:?}", range_vec));
            }
            let end = range_vec.pop().unwrap();
            let start = range_vec.pop().unwrap();
            ranges.push(start..=end);
        }
        Ok(HybridBitset { inner: RangeSetBlaze::from_iter(ranges) })
    }
}

// Helper struct for custom Debug formatting of the inner RangeSetBlaze's ranges.
// This struct is not intended for public use and is defined here for the Debug impl of HybridBitset.
struct DebugRangesTruncated<'a> {
    set: &'a RangeSetBlaze<usize>,
    limit: usize,
    is_alternate: bool, // True if the formatter is in alternate mode (e.g., {:#?})
}

impl<'a> Debug for DebugRangesTruncated<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let total_ranges = self.set.ranges_len();

        // If not truncating (either few ranges, or alternate mode which usually means "show all")
        if total_ranges <= self.limit || self.is_alternate {
            // Use RangeSetBlaze's own Debug impl, which formats as a list of RangeInclusive<usize>
            self.set.fmt(f)
        } else {
            // Truncate: format a list of the first `limit` ranges, then an ellipsis entry
            let mut list_formatter = f.debug_list();
            list_formatter.entries(self.set.ranges().take(self.limit));
            list_formatter.entry(&format_args!("... and {} more ranges", total_ranges - self.limit));
            list_formatter.finish()
        }
    }
}

impl Debug for HybridBitset {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        const MAX_RANGES_IN_DEBUG: usize = 10; // Threshold for truncation in normal debug mode
        let is_alternate_mode = f.alternate(); // Call alternate() before the mutable borrow for debug_struct

        f.debug_struct("HybridBitset")
            .field("inner", &DebugRangesTruncated {
                set: &self.inner,
                limit: MAX_RANGES_IN_DEBUG,
                is_alternate: is_alternate_mode,
            })
            .finish()
    }
}

// --- Core Implementation (`impl HybridBitset`) ---
impl HybridBitset {
    /// Creates a new, empty HybridBitset.
    pub fn zeros() -> Self {
        HybridBitset {
            inner: RangeSetBlaze::new(),
        }
    }

    /// Creates a new HybridBitset with all indices from 0 up to `max_value` (inclusive) set to true.
    pub fn ones(len: usize) -> Self {
        if len == 0 {
            HybridBitset::zeros()
        } else {
            HybridBitset {
                inner: RangeSetBlaze::from_iter([0..=len - 1]),
            }
        }
    }

    pub fn max_ones() -> Self {
        HybridBitset {
            inner: RangeSetBlaze::from_iter([0..=usize::MAX]),
        }
    }

    /// Creates a HybridBitset from an iterator of indices.
    pub fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        HybridBitset {
            inner: RangeSetBlaze::from_iter(iter),
        }
    }

    /// Returns the exact number of set bits (cardinality).
    /// The count is expected to fit within a `usize`.
    /// If the actual count (which can be `u128`) exceeds `usize::MAX`,
    /// this will saturate at `usize::MAX`.
    pub fn len(&self) -> usize {
        let count_u128 = self.inner.len(); // <usize as Integer>::SafeLen is u128
        count_u128.try_into().unwrap_or(usize::MAX)
    }

    /// Returns true if the bitset contains no set bits.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Checks if a specific index is set.
    pub fn contains(&self, index: usize) -> bool {
        self.inner.contains(index)
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        self.inner.is_subset(&other.inner)
    }

    /// Inserts an index into the set. Returns true if the index was not already present.
    pub fn insert(&mut self, index: usize) -> bool {
        self.inner.insert(index)
    }

    /// Sets or clears an index.
    pub fn set(&mut self, index: usize, value: bool) {
        if value {
            self.inner.insert(index);
        } else {
            self.inner.remove(index);
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

    pub fn inverted(&self) -> Self {
        &Self::max_ones() - self
    }

    /// Returns an iterator over the indices of the set bits.
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            // The full type for range_set_blaze::Iter is:
            // range_set_blaze::Iter<T, range_set_blaze::RangesIter<'_, T>>
            // where T is usize in this case.
            iter_inner: self.inner.iter(),
            remaining: self.len(), // Use HybridBitset's len() which returns usize
        }
    }

    /// Returns an iterator over booleans, indicating for each index from 0
    /// up to the largest index present in the set (inclusive) whether it's set or not.
    /// If the set is empty, the iterator is empty.
    pub fn iter_bits(&self) -> BitsIter<'_> {
        if self.is_empty() {
            BitsIter {
                bitset: self,
                current_idx: 1, // Start beyond max_idx_to_iterate to yield nothing
                max_idx_to_iterate: 0,
            }
        } else {
            // self.inner.last() is Option<usize>
            let max_val_in_set = self.inner.last().unwrap_or(0); // unwrap is safe due to is_empty check
            BitsIter {
                bitset: self,
                current_idx: 0,
                max_idx_to_iterate: max_val_in_set,
            }
        }
    }

    pub fn inner(&self) -> &RangeSetBlaze<usize> {
        &self.inner
    }

    /// Given a collection of HybridBitsets, this function finds a "good" permutation
    /// of the underlying integers to reduce the total number of disjoint intervals
    /// across all sets. This is a heuristic approach to the NP-hard problem of
    /// minimizing "total fragmentation".
    ///
    /// The algorithm works by:
    /// 1. Building a graph where nodes are the integers and edge weights represent
    ///    how often two integers are consecutive in the input sets.
    /// 2. Greedily forming chains of integers based on the heaviest edges.
    /// 3. Concatenating these chains to produce a final linear ordering.
    ///
    /// The returned permutation is a map from the original integer values to their
    /// new positions (0, 1, 2, ...).
    pub fn find_good_permutation(sets: &[&Self]) -> std::collections::HashMap<usize, usize> {
        use std::collections::{HashMap, HashSet, VecDeque};

        // --- Step 1: Build Adjacency Graph and Collect All Nodes ---
        let mut adj_graph: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
        let mut all_nodes: HashSet<usize> = HashSet::new();

        for set in sets.iter() {
            let mut iter = set.iter().peekable();
            while let Some(u) = iter.next() {
                all_nodes.insert(u);
                if let Some(&v) = iter.peek() {
                    // u and v are consecutive in this set, increment edge weight
                    *adj_graph.entry(u).or_default().entry(v).or_default() += 1;
                    *adj_graph.entry(v).or_default().entry(u).or_default() += 1;
                }
            }
        }

        // --- Step 2: Generate Permutation Order ---

        // 2a. Create a list of all edges and sort by weight descending
        let mut edges: Vec<(usize, usize, usize)> = Vec::new();
        for (&u, neighbors) in &adj_graph {
            for (&v, &weight) in neighbors {
                if u < v { // Add each edge only once
                    edges.push((u, v, weight));
                }
            }
        }
        edges.sort_unstable_by_key(|&(_, _, weight)| std::cmp::Reverse(weight));

        // 2b. Build chains of nodes by processing edges from heaviest to lightest
        let mut used_nodes: HashSet<usize> = HashSet::new();
        let mut chains: Vec<VecDeque<usize>> = Vec::new();

        for &(u, v, _) in &edges {
            if used_nodes.contains(&u) || used_nodes.contains(&v) {
                continue; // Skip if either node is already in a chain
            }

            // Start a new chain with this edge
            let mut current_chain = VecDeque::from([u, v]);
            used_nodes.insert(u);
            used_nodes.insert(v);

            // Greedily extend the chain to the right
            let mut current_end = v;
            loop {
                let best_neighbor = adj_graph.get(&current_end).and_then(|neighbors| {
                    neighbors.iter()
                        .filter(|(node, _)| !used_nodes.contains(node))
                        .max_by_key(|(_, weight)| **weight)
                        .map(|(node, _)| *node)
                });

                if let Some(neighbor) = best_neighbor {
                    current_chain.push_back(neighbor);
                    used_nodes.insert(neighbor);
                    current_end = neighbor;
                } else { break; }
            }

            // Greedily extend the chain to the left
            let mut current_start = u;
            loop {
                let best_neighbor = adj_graph.get(&current_start).and_then(|neighbors| {
                    neighbors.iter()
                        .filter(|(node, _)| !used_nodes.contains(node))
                        .max_by_key(|(_, weight)| **weight)
                        .map(|(node, _)| *node)
                });

                if let Some(neighbor) = best_neighbor {
                    current_chain.push_front(neighbor);
                    used_nodes.insert(neighbor);
                    current_start = neighbor;
                } else { break; }
            }
            chains.push(current_chain);
        }

        // 2c. Handle nodes that were not part of any edge (singletons)
        for &node in &all_nodes {
            if !used_nodes.contains(&node) {
                chains.push(VecDeque::from([node]));
            }
        }

        // --- Step 3: Produce the Final Permutation Map ---
        chains.into_iter().flat_map(|chain| chain.into_iter()).enumerate().map(|(new_idx, old_idx)| (old_idx, new_idx)).collect()
    }
}

// --- Iterator ---
pub struct Iter<'a> {
    iter_inner: range_set_blaze::Iter<usize, range_set_blaze::RangesIter<'a, usize>>,
    remaining: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter_inner.next() {
            Some(item) => {
                self.remaining -= 1;
                Some(item)
            }
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
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
    type IntoIter = range_set_blaze::IntoIter<usize>; // Use RangeSetBlaze's owning iterator

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

// --- Boolean Iterator ---
pub struct BitsIter<'a> {
    bitset: &'a HybridBitset,
    current_idx: usize,
    max_idx_to_iterate: usize, // Inclusive
}

impl<'a> Iterator for BitsIter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx > self.max_idx_to_iterate {
            None
        } else {
            let val_to_yield = self.bitset.contains(self.current_idx);
            self.current_idx += 1;
            Some(val_to_yield)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = if self.current_idx > self.max_idx_to_iterate {
            0
        } else {
            (self.max_idx_to_iterate - self.current_idx) + 1
        };
        (remaining, Some(remaining))
    }
}

impl<'a> std::iter::ExactSizeIterator for BitsIter<'a> {}

// Implement FromIterator for HybridBitset
impl FromIterator<usize> for HybridBitset {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        HybridBitset {
            inner: RangeSetBlaze::from_iter(iter),
        }
    }
}

// --- Bitwise Operations (Creating New Sets) ---

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

impl Sub for HybridBitset {
    type Output = HybridBitset;

    fn sub(self, rhs: Self) -> Self::Output {
        HybridBitset {
            inner: &self.inner - &rhs.inner,
        }
    }
}

impl Sub for &HybridBitset {
    type Output = HybridBitset;

    fn sub(self, rhs: Self) -> Self::Output {
        HybridBitset {
            inner: &self.inner - &rhs.inner,
        }
    }
}

// --- In-place Bitwise Operations ---

impl BitAndAssign for HybridBitset {
    fn bitand_assign(&mut self, rhs: Self) {
        let start_time = std::time::Instant::now();
        self.inner = &self.inner & &rhs.inner;
        let duration = start_time.elapsed();
        if duration.as_millis() > 1 {
            println!("HybridBitset::bitand_assign (owned) took {:?}", duration);
        }
        if duration.as_millis() > 10 {
            // panic!("HybridBitset::bitand_assign (owned) took {:?}", duration);
        }
    }
}

impl BitOrAssign for HybridBitset {
    fn bitor_assign(&mut self, rhs: Self) {
        // let self_ranges_len = self.inner.ranges_len();
        // let rhs_ranges_len = rhs.inner.ranges_len();
        // let self_len = self.inner.len();
        // let rhs_len = rhs.inner.len();
        // let self_compression_ratio: f64 = self_len as f64 / self_ranges_len as f64;
        // let rhs_compression_ratio: f64 = rhs_len as f64 / rhs_ranges_len as f64;
        // let MAX_RANGES_LEN = 1024;
        // let MIN_COMPRESSION_RATIO = 2.0;
        // if self_compression_ratio < MIN_COMPRESSION_RATIO && self_ranges_len > MAX_RANGES_LEN || rhs_compression_ratio < MIN_COMPRESSION_RATIO && rhs_ranges_len > MAX_RANGES_LEN {
        //     panic!("BitOr of HybridBitset with more than {} ranges is not supported. Got hybrid bitsets with length {} and {} but ranges lengths {} and {}, implying compression ratios of {:.2} and {:.2}, respectively", MAX_RANGES_LEN, self_len, rhs_len, self_ranges_len, rhs_ranges_len, self_compression_ratio, rhs_compression_ratio);
        // }
        // // Option 2: Using named arguments for better readability (especially with many args)
        // println!(
        //     "BitOr of HybridBitsets with lengths {slen:>5} and {rlen:>5} and ranges lengths {srlen:>5} and {rrlen:>5}, implying compression ratios of {scratio:>7.2} and {rcratio:>7.2}, respectively",
        //     slen = self_len,
        //     rlen = rhs_len,
        //     srlen = self_ranges_len,
        //     rrlen = rhs_ranges_len,
        //     scratio = self_compression_ratio,
        //     rcratio = rhs_compression_ratio
        // );
        self.inner |= rhs.inner; // RangeSetBlaze has BitOrAssign<RangeSetBlaze>
    }
}

impl BitXorAssign for HybridBitset {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.inner = &self.inner ^ &rhs.inner;
    }
}

impl SubAssign for HybridBitset {
    fn sub_assign(&mut self, rhs: Self) {
        self.inner = &self.inner - &rhs.inner;
    }
}

// --- In-place Bitwise Operations with References ---

impl BitAndAssign<&HybridBitset> for HybridBitset {
    fn bitand_assign(&mut self, rhs: &HybridBitset) {
        let start_time = std::time::Instant::now();
        self.inner = &self.inner & &rhs.inner;
        let duration = start_time.elapsed();
        if duration.as_millis() > 100 {
            println!(
                "HybridBitset::bitand_assign (ref) took {}ms",
                duration.as_millis()
            );
        }
    }
}

impl BitOrAssign<&HybridBitset> for HybridBitset {
    fn bitor_assign(&mut self, rhs: &HybridBitset) {
        self.inner |= &rhs.inner; // RangeSetBlaze has BitOrAssign<&RangeSetBlaze>
    }
}

impl BitXorAssign<&HybridBitset> for HybridBitset {
    fn bitxor_assign(&mut self, rhs: &HybridBitset) {
        self.inner = &self.inner ^ &rhs.inner;
    }
}

impl SubAssign<&HybridBitset> for HybridBitset {
    fn sub_assign(&mut self, rhs: &HybridBitset) {
        self.inner = &self.inner - &rhs.inner;
    }
}

// --- Equality and Hashing ---
impl PartialEq for HybridBitset {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
// Eq is derived.

impl Hash for HybridBitset {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

// --- Conversions and Indexing (Kept as todo! as per original) ---
use bitvec::prelude::*; // Keep for BitVec type in signatures

impl Into<BitVec<usize, Lsb0>> for HybridBitset {
    fn into(self) -> BitVec<usize, Lsb0> {
        todo!("Conversion from HybridBitset (RangeSetBlaze based) to BitVec is not directly implemented yet.")
    }
}

impl From<BitVec<usize, Lsb0>> for HybridBitset {
    fn from(bitvec: BitVec<usize, Lsb0>) -> Self {
        HybridBitset { inner: RangeSetBlaze::from_iter(bitvec.iter_ones()) }
    }
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::iter::FromIterator;
    use deterministic_hash::DeterministicHasher;

    // Thresholds are no longer used by HybridBitset directly
    const SPARSE_TO_DENSE_THRESHOLD: usize = 128; // Keep for test logic if needed, but not for HybridBitset
    const DENSE_TO_SPARSE_THRESHOLD: usize = 64;   // Keep for test logic if needed

    #[test]
    fn test_new_empty_len() {
        let set = HybridBitset::zeros();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }

    #[test]
    fn test_insert_basic() {
        let mut set = HybridBitset::zeros();
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
    fn test_into_iteration() {
        let indices = vec![5, 1, 100, 42];
        let set = HybridBitset::from_iter(indices.clone());
        let mut collected: Vec<usize> = set.into_iter().collect(); // Consumes set
        collected.sort_unstable();
        let mut expected = indices;
        expected.sort_unstable();
        assert_eq!(collected, expected);
    }


    #[test]
    fn test_set_ops_sparse_sparse() { // Names are now conceptual, as internal repr is opaque
        let set1 = HybridBitset::from_iter(vec![1, 2, 3, 10]);
        let set2 = HybridBitset::from_iter(vec![3, 4, 5, 10]);

        let intersection = &set1 & &set2;
        let union = &set1 | &set2;
        let difference = &set1 - &set2;
        let sym_diff = &set1 ^ &set2;

        assert_eq!(intersection.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![3, 10]));
        assert_eq!(union.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![1, 2, 3, 4, 5, 10]));
        assert_eq!(difference.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![1, 2]));
        assert_eq!(sym_diff.iter().collect::<BTreeSet<usize>>(), BTreeSet::from_iter(vec![1, 2, 4, 5]));
    }

    #[test]
    fn test_set_ops_dense_dense() { // Names are now conceptual
        let set1 = HybridBitset::from_iter(0..SPARSE_TO_DENSE_THRESHOLD + 10);
        let set2 = HybridBitset::from_iter(5..SPARSE_TO_DENSE_THRESHOLD + 20);

        let intersection = &set1 & &set2;
        let union = &set1 | &set2;
        let difference = &set1 - &set2;
        let sym_diff = &set1 ^ &set2;

        let intersection_expected: BTreeSet<usize> = (5..SPARSE_TO_DENSE_THRESHOLD + 10).collect();
        let union_expected: BTreeSet<usize> = (0..SPARSE_TO_DENSE_THRESHOLD + 20).collect();
        let difference_expected: BTreeSet<usize> = (0..5).collect();
        let sym_diff_expected: BTreeSet<usize> = (0..5).chain(SPARSE_TO_DENSE_THRESHOLD + 10..SPARSE_TO_DENSE_THRESHOLD + 20).collect();

        assert_eq!(intersection.iter().collect::<BTreeSet<usize>>(), intersection_expected);
        assert_eq!(union.iter().collect::<BTreeSet<usize>>(), union_expected);
        assert_eq!(difference.iter().collect::<BTreeSet<usize>>(), difference_expected);
        assert_eq!(sym_diff.iter().collect::<BTreeSet<usize>>(), sym_diff_expected);
    }

    #[test]
    fn test_set_ops_mixed() { // Names are now conceptual
        let set1_conceptually_sparse = HybridBitset::from_iter(vec![1, 2, 3, SPARSE_TO_DENSE_THRESHOLD + 100]);
        let set2_conceptually_dense = HybridBitset::from_iter(0..SPARSE_TO_DENSE_THRESHOLD + 5);

        let intersection1 = &set1_conceptually_sparse & &set2_conceptually_dense;
        let intersection1_expected: BTreeSet<usize> = vec![1, 2, 3].into_iter().collect();
        assert_eq!(intersection1.iter().collect::<BTreeSet<usize>>(), intersection1_expected);

        let union1 = &set1_conceptually_sparse | &set2_conceptually_dense;
        let mut union1_expected: BTreeSet<usize> = (0..SPARSE_TO_DENSE_THRESHOLD + 5).collect();
        union1_expected.insert(SPARSE_TO_DENSE_THRESHOLD + 100);
        assert_eq!(union1.iter().collect::<BTreeSet<usize>>(), union1_expected);

        let diff1 = &set1_conceptually_sparse - &set2_conceptually_dense;
        let diff1_expected: BTreeSet<usize> = vec![SPARSE_TO_DENSE_THRESHOLD + 100].into_iter().collect();
        assert_eq!(diff1.iter().collect::<BTreeSet<usize>>(), diff1_expected);

        let diff2 = &set2_conceptually_dense - &set1_conceptually_sparse;
        let mut diff2_expected: BTreeSet<usize> = (0..SPARSE_TO_DENSE_THRESHOLD + 5).collect();
        diff2_expected.remove(&1); diff2_expected.remove(&2); diff2_expected.remove(&3);
        assert_eq!(diff2.iter().collect::<BTreeSet<usize>>(), diff2_expected);

        let xor1 = &set1_conceptually_sparse ^ &set2_conceptually_dense;
        let mut xor1_expected = diff2_expected.clone();
        xor1_expected.insert(SPARSE_TO_DENSE_THRESHOLD + 100);
        assert_eq!(xor1.iter().collect::<BTreeSet<usize>>(), xor1_expected);
    }

    #[test]
    fn test_equality_and_hash() {
        let set1 = HybridBitset::from_iter(vec![1, 5, 10]);
        let set1_clone = HybridBitset::from_iter(vec![1, 5, 10]); // Same elements
        let set2 = HybridBitset::from_iter(vec![1, 5, 11]);    // Different elements
        let empty_set = HybridBitset::zeros();

        assert_eq!(set1, set1_clone);
        assert_ne!(set1, set2);
        assert_ne!(set1, empty_set);

        use std::collections::hash_map::DefaultHasher;
        let hash = |s: &HybridBitset| -> u64 {
            let mut hasher = DeterministicHasher::new(DefaultHasher::new());
            s.hash(&mut hasher);
            hasher.finish()
        };

        assert_eq!(hash(&set1), hash(&set1_clone));
        assert_ne!(hash(&set1), hash(&set2));
        assert_ne!(hash(&set1), hash(&empty_set));

        let mut btree_map = BTreeSet::new();
        btree_map.insert(set1.clone());
        assert!(btree_map.contains(&set1));
        assert!(btree_map.contains(&set1_clone));

        btree_map.insert(set1_clone.clone());
        assert_eq!(btree_map.len(), 1);

        btree_map.insert(set2.clone());
        assert_eq!(btree_map.len(), 2);
        assert!(btree_map.contains(&set2));
    }

    #[test]
    fn test_large_index() {
        let mut set = HybridBitset::zeros();
        let large_idx = 1_000_000;
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
        let mut set = HybridBitset::from_iter(0..SPARSE_TO_DENSE_THRESHOLD + 10);
        assert!(!set.is_empty());
        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);

        let mut set2 = HybridBitset::from_iter(vec![1,2,3]);
        assert!(!set2.is_empty());
        set2.clear();
        assert!(set2.is_empty());
        assert_eq!(set2.len(), 0);
    }

    #[test]
    fn test_assign_ops() {
        let mut set1_orig = HybridBitset::from_iter(vec![1, 2, 10]);
        let set2 = HybridBitset::from_iter(vec![2, 3, 20]);

        let mut set1 = set1_orig.clone();
        set1 |= set2.clone();
        assert_eq!(set1.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 2, 3, 10, 20]));

        let set3_orig = HybridBitset::from_iter(0..DENSE_TO_SPARSE_THRESHOLD); // Conceptual dense
        let set4 = HybridBitset::from_iter( (DENSE_TO_SPARSE_THRESHOLD/2)..DENSE_TO_SPARSE_THRESHOLD + 10);
        let expected_and = (DENSE_TO_SPARSE_THRESHOLD/2..DENSE_TO_SPARSE_THRESHOLD).collect::<BTreeSet<_>>();
        let mut set3 = set3_orig.clone();
        set3 &= set4.clone();
        assert_eq!(set3.iter().collect::<BTreeSet<_>>(), expected_and);

        let mut set5 = HybridBitset::from_iter(vec![1, 2, 3]);
        let set6 = HybridBitset::from_iter(vec![3, 4, 5]);
        set5 ^= set6.clone();
        assert_eq!(set5.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 2, 4, 5]));

        let mut set7 = HybridBitset::from_iter(vec![1, 2, 3, 4, 5]);
        let set8 = HybridBitset::from_iter(vec![2, 4, 6]);
        set7 -= set8.clone();
        assert_eq!(set7.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 3, 5]));
    }

    #[test]
    fn test_assign_ops_ref() {
        let mut set1_orig = HybridBitset::from_iter(vec![1, 2, 10]);
        let set2 = HybridBitset::from_iter(vec![2, 3, 20]);

        let mut set1 = set1_orig.clone();
        set1 |= &set2;
        assert_eq!(set1.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 2, 3, 10, 20]));

        let set3_orig = HybridBitset::from_iter(0..DENSE_TO_SPARSE_THRESHOLD);
        let set4 = HybridBitset::from_iter( (DENSE_TO_SPARSE_THRESHOLD/2)..DENSE_TO_SPARSE_THRESHOLD + 10);
        let expected_and = (DENSE_TO_SPARSE_THRESHOLD/2..DENSE_TO_SPARSE_THRESHOLD).collect::<BTreeSet<_>>();
        let mut set3 = set3_orig.clone();
        set3 &= &set4;
        assert_eq!(set3.iter().collect::<BTreeSet<_>>(), expected_and);

        let mut set5 = HybridBitset::from_iter(vec![1, 2, 3]);
        let set6 = HybridBitset::from_iter(vec![3, 4, 5]);
        set5 ^= &set6;
        assert_eq!(set5.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 2, 4, 5]));

        let mut set7 = HybridBitset::from_iter(vec![1, 2, 3, 4, 5]);
        let set8 = HybridBitset::from_iter(vec![2, 4, 6]);
        set7 -= &set8;
        assert_eq!(set7.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![1, 3, 5]));
    }

    #[test]
    fn test_dense_dense_edge_cases() { // Conceptual names
        let d1 = HybridBitset::zeros();
        let d2 = HybridBitset::zeros();
        let d3 = HybridBitset::from_iter(0..DENSE_TO_SPARSE_THRESHOLD);

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

        let inter = &d4 & &d5;
        assert_eq!(inter.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![3, 4]));

        let union = &d4 | &d5;
        assert_eq!(union.iter().collect::<BTreeSet<_>>(), (0..10).collect::<BTreeSet<_>>());

        let diff = &d4 - &d5;
        assert_eq!(diff.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![0, 1, 2]));

        let sym_diff = &d4 ^ &d5;
        assert_eq!(sym_diff.iter().collect::<BTreeSet<_>>(), BTreeSet::from_iter(vec![0,1,2,5,6,7,8,9]));
    }

    #[test]
    fn test_from_iterator_trait() { // Renamed to avoid conflict
        let data = vec![10, 20, 10, 30, 20];
        let set: HybridBitset = data.into_iter().collect();

        let expected: BTreeSet<usize> = vec![10, 20, 30].into_iter().collect();
        assert_eq!(set.iter().collect::<BTreeSet<_>>(), expected);
    }

    #[test]
    fn test_iter_bits() {
        let empty_set = HybridBitset::zeros();
        assert_eq!(empty_set.iter_bits().collect::<Vec<bool>>(), Vec::<bool>::new());
        assert_eq!(empty_set.iter_bits().len(), 0);

        let sparse_set = HybridBitset::from_iter(vec![1, 3]);
        let expected_sparse_bools = vec![false, true, false, true];
        assert_eq!(sparse_set.iter_bits().collect::<Vec<bool>>(), expected_sparse_bools);
        assert_eq!(sparse_set.iter_bits().len(), expected_sparse_bools.len());

        // Test with a set that would have been dense
        let dense_like_set = HybridBitset::from_iter(vec![1,3]); // Max index 3
        // RangeSetBlaze doesn't have an explicit dense conversion, iter_bits uses .last()
        assert_eq!(dense_like_set.iter_bits().collect::<Vec<bool>>(), expected_sparse_bools);
        assert_eq!(dense_like_set.iter_bits().len(), expected_sparse_bools.len());

        let empty_set_from_non_empty = HybridBitset::from_iter(vec![5]);
        let _ = empty_set_from_non_empty.inner.last(); // just to use it
        let mut empty_set_cleared = HybridBitset::from_iter(vec![5]);
        empty_set_cleared.clear();
        assert_eq!(empty_set_cleared.iter_bits().collect::<Vec<bool>>(), Vec::<bool>::new());
        assert_eq!(empty_set_cleared.iter_bits().len(), 0);
    }

    #[test]
    fn test_ones() {
        let set_ones_small = HybridBitset::ones(4); // 0, 1, 2, 3
        assert_eq!(set_ones_small.len(), 4);
        assert!(set_ones_small.contains(0));
        assert!(set_ones_small.contains(1));
        assert!(set_ones_small.contains(2));
        assert!(set_ones_small.contains(3));
        assert!(!set_ones_small.contains(4));

        let len = SPARSE_TO_DENSE_THRESHOLD + 5;
        let set_ones_large = HybridBitset::ones(len + 1); // Corrected: len is exclusive upper bound for RangeSetBlaze
        assert_eq!(set_ones_large.len(), SPARSE_TO_DENSE_THRESHOLD + 6);
        for i in 0..=(SPARSE_TO_DENSE_THRESHOLD + 5) {
            assert!(set_ones_large.contains(i));
        }
        assert!(!set_ones_large.contains(SPARSE_TO_DENSE_THRESHOLD + 6));

        // Test edge case for usize::MAX
        // This test might be very slow or OOM with RangeSetBlaze if it tries to create a huge range.
        // let set_ones_max = HybridBitset::ones(usize::MAX);
        // assert!(!set_ones_max.is_empty());
        // assert_eq!(set_ones_max.len(), usize::MAX);

        let set_ones_one = HybridBitset::ones(1); // Should contain only 0
        assert_eq!(set_ones_one.len(), 1);
        assert!(set_ones_one.contains(0));
        assert!(!set_ones_one.contains(1));

        let set_ones_zero = HybridBitset::ones(0); // Should be empty
        assert_eq!(set_ones_zero.len(), 0);
        assert!(set_ones_zero.is_empty());
    }

    #[test]
    fn test_find_good_permutation() {
        // Example from the problem description
        let s1 = HybridBitset::from_iter(vec![1, 2, 3, 4, 8, 9]);
        let s2 = HybridBitset::from_iter(vec![2, 3, 5, 7, 8, 9, 15]);
        let s3 = HybridBitset::from_iter(vec![1, 4, 5, 6, 7, 8, 12, 13]);

        let sets = vec![&s1, &s2, &s3];

        // Calculate original cost
        let original_cost: usize = sets.iter().map(|s| s.inner().ranges_len()).sum();
        assert_eq!(original_cost, 2 + 4 + 3); // 9

        // Find the permutation
        let perm_map = HybridBitset::find_good_permutation(&sets);

        // Apply the permutation
        let apply_permutation = |set: &HybridBitset, map: &std::collections::HashMap<usize, usize>| -> HybridBitset {
            set.iter().map(|val| *map.get(&val).unwrap()).collect()
        };

        let pi_s1 = apply_permutation(&s1, &perm_map);
        let pi_s2 = apply_permutation(&s2, &perm_map);
        let pi_s3 = apply_permutation(&s3, &perm_map);

        // Calculate new cost
        let new_cost = pi_s1.inner().ranges_len() + pi_s2.inner().ranges_len() + pi_s3.inner().ranges_len();

        // The heuristic should improve the cost for this non-trivial case.
        assert!(new_cost < original_cost, "Expected new cost {} to be less than original cost {}", new_cost, original_cost);

        // Test with sets that have no shared adjacencies
        let s4 = HybridBitset::from_iter(vec![100, 200, 300]);
        let s5 = HybridBitset::from_iter(vec![400, 500]);
        let disjoint_sets = vec![&s4, &s5];
        let original_disjoint_cost: usize = disjoint_sets.iter().map(|s| s.inner().ranges_len()).sum();
        let perm_map_disjoint = HybridBitset::find_good_permutation(&disjoint_sets);
        let new_disjoint_cost: usize = disjoint_sets.iter().map(|s| apply_permutation(s, &perm_map_disjoint).inner().ranges_len()).sum();
        assert_eq!(new_disjoint_cost, original_disjoint_cost, "Cost should not change for sets with no adjacencies");
    }
}
