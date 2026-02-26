#![allow(dead_code)] // Allow unused code for the example

use crate::datastructures::cache::{self, Acc};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::datastructures::bitset::Bitset;
use lru::LruCache;
use once_cell::sync::Lazy;
// Added
use range_set_blaze::RangeSetBlaze;
// Import RangeSetBlaze
use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
// Added
use std::iter::FromIterator;
use std::num::NonZeroUsize;
// Needed for collect into BTreeSet in tests
use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Sub, SubAssign,
};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
// ---------------------------------------------------------------------------
// Thread-local dimensions for weight-heavy/symbol-heavy mode
// ---------------------------------------------------------------------------

thread_local! {
    /// Maximum LLM token ID (internal). Set during constraint initialization.
    /// Default is 1_000_000 — large enough for any LLM vocab while preventing
    /// catastrophic memory blowup when Weight::all() is used without explicit
    /// set_global_dims (e.g. in unit tests).
    static MAX_LLM_TOKEN: Cell<usize> = Cell::new(1_000_000);

    /// Number of tokenizer states. Set during constraint initialization.
    /// Default is 1 for symbol-heavy mode.
    static NUM_TSIDS: Cell<usize> = Cell::new(1);
}

const L1_OP_CACHE_CAPACITY: usize = 100_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct L1OpKey {
    op: cache::BinOp,
    a: usize,
    b: usize,
}

static L1_OP_CACHE: Lazy<Mutex<LruCache<L1OpKey, Acc<RangeSetBlaze<usize>>>>> = Lazy::new(|| {
    Mutex::new(LruCache::new(NonZeroUsize::new(L1_OP_CACHE_CAPACITY).unwrap()))
});
static L1_OP_CACHE_INDEX: Lazy<Mutex<HashMap<usize, HashSet<L1OpKey>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static L1_INTERNED_PTRS: Lazy<Mutex<HashSet<usize>>> = Lazy::new(|| Mutex::new(HashSet::new()));

static L1_OP_CACHE_PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);
static L1_OP_CACHE_OR_HITS: AtomicU64 = AtomicU64::new(0);
static L1_OP_CACHE_OR_MISSES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn set_l1_op_cache_profile_enabled(enabled: bool) {
    L1_OP_CACHE_PROFILE_ENABLED.store(enabled, AtomicOrdering::Relaxed);
}

pub(crate) fn reset_l1_op_cache_counters() {
    L1_OP_CACHE_OR_HITS.store(0, AtomicOrdering::Relaxed);
    L1_OP_CACHE_OR_MISSES.store(0, AtomicOrdering::Relaxed);
}

pub(crate) fn l1_op_cache_or_counters() -> (u64, u64) {
    (
        L1_OP_CACHE_OR_HITS.load(AtomicOrdering::Relaxed),
        L1_OP_CACHE_OR_MISSES.load(AtomicOrdering::Relaxed),
    )
}

/// Set the global LLM token dimension.
/// 
/// This should be called once during constraint initialization with the actual
/// maximum internal LLM token ID.
pub fn set_global_dims(max_llm_token: usize, num_tsids: usize) {
    crate::debug!(4, "set_global_dims: max_llm_token={}, num_tsids={}", max_llm_token, num_tsids);
    MAX_LLM_TOKEN.with(|value| value.set(max_llm_token));
    NUM_TSIDS.with(|value| value.set(num_tsids));
}

/// Set global dims for the current thread and propagate to all Rayon workers.
pub fn set_global_dims_all_threads(max_llm_token: usize, num_tsids: usize) {
    set_global_dims(max_llm_token, num_tsids);
    rayon::broadcast(|_| {
        MAX_LLM_TOKEN.with(|value| value.set(max_llm_token));
        NUM_TSIDS.with(|value| value.set(num_tsids));
    });
}

/// Get the current global max LLM token ID.
pub fn get_max_llm_token() -> usize {
    MAX_LLM_TOKEN.with(|value| value.get())
}

/// Get the current global number of tokenizer states.
pub fn get_num_tsids() -> usize {
    NUM_TSIDS.with(|value| value.get())
}

// --- Profiling ---
// Legacy weight-op profiling removed; keep no-op hooks for callers.
pub fn reset_profiling() {}

pub fn print_profiling(_label: &str) {}

fn l1_op_key(op: cache::BinOp, a: &Acc<RangeSetBlaze<usize>>, b: &Acc<RangeSetBlaze<usize>>) -> L1OpKey {
    L1OpKey {
        op,
        a: Arc::as_ptr(a) as usize,
        b: Arc::as_ptr(b) as usize,
    }
}

fn is_interned_l1(acc: &Acc<RangeSetBlaze<usize>>) -> bool {
    let ptr = Arc::as_ptr(acc) as usize;
    L1_INTERNED_PTRS.lock().unwrap().contains(&ptr)
}

fn remove_l1_op_key_from_index(index: &mut HashMap<usize, HashSet<L1OpKey>>, key: L1OpKey) {
    if let Some(set) = index.get_mut(&key.a) {
        set.remove(&key);
        if set.is_empty() {
            index.remove(&key.a);
        }
    }
    if let Some(set) = index.get_mut(&key.b) {
        set.remove(&key);
        if set.is_empty() {
            index.remove(&key.b);
        }
    }
}

fn invalidate_l1_op_cache_for_ptr(ptr: usize) {
    let mut cache = L1_OP_CACHE.lock().unwrap();
    let mut index = L1_OP_CACHE_INDEX.lock().unwrap();
    let Some(keys) = index.remove(&ptr) else { return; };
    for key in keys {
        cache.pop(&key);
        remove_l1_op_key_from_index(&mut index, key);
    }
}

fn intern_l1_tracked(rs: RangeSetBlaze<usize>) -> Acc<RangeSetBlaze<usize>> {
    // Use thread-local interning only. The global L1 op cache tracking
    // (invalidation + pointer set) was pure overhead — profiling shows
    // 0 L1 cache hits because all RangeSets are "simple" (< 16 ranges)
    // and bypass the cache. Removing the 3 Mutex locks per interning call
    // eliminates massive contention during parallel determinize.
    cache::intern_l1(rs)
}

fn get_l1_op_cache_tracked(
    op: cache::BinOp,
    a: &Acc<RangeSetBlaze<usize>>,
    b: &Acc<RangeSetBlaze<usize>>,
) -> Option<Acc<RangeSetBlaze<usize>>> {
    if !is_interned_l1(a) || !is_interned_l1(b) {
        return None;
    }
    let mut cache = L1_OP_CACHE.lock().unwrap();
    let key = l1_op_key(op, a, b);
    cache.get(&key).cloned()
}

fn put_l1_op_cache_tracked(
    op: cache::BinOp,
    a: Acc<RangeSetBlaze<usize>>,
    b: Acc<RangeSetBlaze<usize>>,
    result: Acc<RangeSetBlaze<usize>>,
) {
    if !is_interned_l1(&a) || !is_interned_l1(&b) {
        return;
    }
    let key = l1_op_key(op, &a, &b);
    let mut cache = L1_OP_CACHE.lock().unwrap();
    let mut index = L1_OP_CACHE_INDEX.lock().unwrap();
    if let Some((evicted_key, _)) = cache.push(key, result) {
        remove_l1_op_key_from_index(&mut index, evicted_key);
    }
    index.entry(key.a).or_default().insert(key);
    index.entry(key.b).or_default().insert(key);
}



// --- The Hybrid Bitset Struct ---
#[derive(Default, Clone, Eq)]
pub struct RangeSet {
    pub(crate) inner: Arc<RangeSetBlaze<usize>>,
}

impl serde::Serialize for RangeSet {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let ranges: Vec<_> = self.inner.ranges().collect();
        let mut seq = serializer.serialize_seq(Some(ranges.len() * 2))?;
        for range in ranges {
            seq.serialize_element(range.start())?;
            seq.serialize_element(range.end())?;
        }
        seq.end()
    }
}

impl<'de> serde::Deserialize<'de> for RangeSet {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let flat: Vec<usize> = Vec::deserialize(deserializer)?;
        let ranges: Vec<_> = flat.chunks(2)
            .filter_map(|c| if c.len() == 2 { Some(c[0]..=c[1]) } else { None })
            .collect();
        let inner = RangeSetBlaze::from_iter(ranges);
        Ok(RangeSet { inner: Arc::new(inner) })
    }
}

impl JSONConvertible for RangeSet {
    fn to_json(&self) -> JSONNode {
        // Flattened array format: [start1, end1, start2, end2, ...]
        let mut flat = Vec::new();
        for range in self.inner.ranges() {
            flat.push(JSONNode::UInt(*range.start() as u128));
            flat.push(JSONNode::UInt(*range.end() as u128));
        }
        JSONNode::Array(flat)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => {
                if arr.len() % 2 != 0 {
                    return Err(format!(
                        "Expected even number of elements in flattened HybridBitset array, got {}",
                        arr.len()
                    ));
                }
                let mut ranges = Vec::new();
                for chunk in arr.chunks(2) {
                    let start = usize::from_json(chunk[0].clone())?;
                    let end = usize::from_json(chunk[1].clone())?;
                    ranges.push(start..=end);
                }
                Ok(RangeSet {
                    inner: intern_l1_tracked(RangeSetBlaze::from_iter(ranges)),
                })
            }
            _ => Err("Expected JSONNode::Array for HybridBitset".to_string()),
        }
    }
}

// Helper struct for custom Debug formatting of the inner RangeSetBlaze's ranges.
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
            Debug::fmt(&self.set, f)
        } else {
            // Truncate: format a list of the first `limit` ranges, then an ellipsis entry
            let mut list_formatter = f.debug_list();
            list_formatter.entries(self.set.ranges().take(self.limit));
            list_formatter.entry(&format_args!(
                "... and {} more ranges",
                total_ranges - self.limit
            ));
            list_formatter.finish()
        }
    }
}

impl Debug for RangeSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_all() {
            return f
                .debug_struct("HybridBitset")
                .field("inner", &format_args!("0..=usize::MAX"))
                .finish();
        }

        const MAX_RANGES_IN_DEBUG: usize = 10; // Threshold for truncation in normal debug mode
        let is_alternate_mode = f.alternate(); // Call alternate() before the mutable borrow for debug_struct

        f.debug_struct("HybridBitset")
            .field(
                "inner",
                &DebugRangesTruncated {
                    set: &self.inner,
                    limit: MAX_RANGES_IN_DEBUG,
                    is_alternate: is_alternate_mode,
                },
            )
            .finish()
    }
}

impl Display for RangeSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;

        let mut ranges = self.inner.ranges().peekable();
        while let Some(range) = ranges.next() {
            let start = *range.start();
            let end = *range.end();

            if start == end {
                write!(f, "{}", start)?;
            } else if end == usize::MAX {
                write!(f, "{}..", start)?;
            } else {
                write!(f, "{}..{}", start, end)?;
            }

            if ranges.peek().is_some() {
                write!(f, ", ")?;
            }
        }

        write!(f, "]")
    }
}

// --- Core Implementation (`impl HybridBitset`) ---
impl RangeSet {
    /// Creates a new, empty HybridBitset.
    pub fn zeros() -> Self {
        RangeSet {
            inner: intern_l1_tracked(RangeSetBlaze::new()),
        }
    }

    pub fn new_empty(len: usize) -> Self {
        Self::zeros()
    }

    /// Creates a new HybridBitset with all indices from 0 up to `max_value` (inclusive) set to true.
    pub fn ones(len: usize) -> Self {
        if len == 0 {
            RangeSet::zeros()
        } else {
            RangeSet {
                inner: intern_l1_tracked(RangeSetBlaze::from_iter([0..=len - 1])),
            }
        }
    }

    pub fn max_ones() -> Self {
        // Use global dimensions if set, otherwise fall back to usize::MAX for backwards compatibility
        // In weight-heavy mode (num_tsids > 1), the domain is expanded to N×M space
        let max_token = get_max_llm_token();
        let num_tsids = get_num_tsids();
        let domain_max = if num_tsids > 1 {
            // N×M space: max position is (N-1)*M + (M-1) = N*M - 1
            // where N = max_token + 1, M = num_tsids
            max_token.saturating_mul(num_tsids).saturating_add(num_tsids.saturating_sub(1))
        } else {
            max_token
        };
        RangeSet {
            inner: intern_l1_tracked(RangeSetBlaze::from_iter([0..=domain_max])),
        }
    }
    
    /// Creates a RangeSet containing all LLM tokens based on global dimensions.
    /// 
    /// This should be used instead of `max_ones()` when the global dimensions have been set.
    /// In weight-heavy mode (num_tsids > 1), returns the full expanded N×M domain.
    pub fn all_llm_tokens() -> Self {
        let max_token = get_max_llm_token();
        let num_tsids = get_num_tsids();
        let domain_max = if num_tsids > 1 {
            // N×M space: max position is (N-1)*M + (M-1) = N*M - 1
            max_token.saturating_mul(num_tsids).saturating_add(num_tsids.saturating_sub(1))
        } else {
            max_token
        };
        RangeSet {
            inner: intern_l1_tracked(RangeSetBlaze::from_iter([0..=domain_max])),
        }
    }

    /// Creates a HybridBitset from an iterator of indices.
    pub fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        RangeSet {
            inner: intern_l1_tracked(RangeSetBlaze::from_iter(iter)),
        }
    }

    pub fn from_item(item: usize) -> Self {
        RangeSet {
            inner: intern_l1_tracked(RangeSetBlaze::from_iter([item..=item])),
        }
    }

    /// Union multiple RangeSets in a single operation.
    pub fn bulk_union(sets: &[&RangeSet]) -> Self {
        if sets.is_empty() {
            return RangeSet::zeros();
        }
        if sets.len() == 1 {
            return sets[0].clone();
        }

        let total_ranges: usize = sets.iter().map(|s| s.ranges_len()).sum();
        if total_ranges == 0 {
            return RangeSet::zeros();
        }

        let mut all_ranges: Vec<(usize, usize)> = Vec::with_capacity(total_ranges);
        for set in sets {
            for range in set.ranges() {
                all_ranges.push((*range.start(), *range.end()));
            }
        }
        if all_ranges.is_empty() {
            return RangeSet::zeros();
        }

        all_ranges.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut merged: Vec<std::ops::RangeInclusive<usize>> = Vec::with_capacity(all_ranges.len());
        let mut current_start = all_ranges[0].0;
        let mut current_end = all_ranges[0].1;
        for (start, end) in all_ranges.into_iter().skip(1) {
            if start <= current_end.saturating_add(1) {
                if end > current_end {
                    current_end = end;
                }
            } else {
                merged.push(current_start..=current_end);
                current_start = start;
                current_end = end;
            }
        }
        merged.push(current_start..=current_end);

        RangeSet {
            inner: intern_l1_tracked(RangeSetBlaze::from_iter(merged)),
        }
    }

    /// A bitset is simple if it has a small number of ranges, making operations fast
    /// enough that caching overhead is not worthwhile.
    pub fn is_simple(&self) -> bool {
        self.inner.ranges_len() < cache::SIMPLE_BITSET_THRESHOLD
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
    
    /// Returns true if the bitset is "all" (contains 0..=max where max is global max_llm_token).
    /// 
    /// This checks if the bitset spans from 0 to the current global max_llm_token.
    /// For backwards compatibility, also returns true if it spans 0..=usize::MAX.
    pub fn is_all(&self) -> bool {
        if self.inner.ranges_len() != 1 {
            return false;
        }
        if let Some(range) = self.inner.ranges().next() {
            let start = *range.start();
            let end = *range.end();
            if start != 0 {
                return false;
            }
            // Check against global max OR usize::MAX for backwards compatibility
            let max_token = get_max_llm_token();
            end == max_token || end == usize::MAX
        } else {
            false
        }
    }

    /// Checks if a specific index is set.
    pub fn contains(&self, index: usize) -> bool {
        self.inner.contains(index)
    }

    /// Returns an iterator over the inclusive [start, end] ranges of the set.
    pub fn ranges(&self) -> impl Iterator<Item = std::ops::RangeInclusive<usize>> + '_ {
        self.inner.ranges()
    }

    /// Returns the number of ranges in the set.
    pub fn ranges_len(&self) -> usize {
        self.inner.ranges_len()
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        self.inner.is_subset(&other.inner)
    }
    pub fn is_superset(&self, other: &Self) -> bool {
        self.inner.is_superset(&other.inner)
    }
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.inner.is_disjoint(&other.inner)
    }
    pub fn intersects(&self, other: &Self) -> bool {
        !self.is_disjoint(other)
    }

    /// Inserts an index into the set. Returns true if the index was not already present.
    pub fn insert_with_intern(&mut self, index: usize) -> bool {
        if self.inner.contains(index) {
            return false;
        }
        let mut new_inner = (*self.inner).clone();
        let result = new_inner.insert(index);
        self.inner = intern_l1_tracked(new_inner);
        result
    }

    pub fn insert(&mut self, index: usize) -> bool {
        if self.inner.contains(index) {
            return false;
        }
        Arc::make_mut(&mut self.inner).insert(index)
    }

    /// Sets or clears an index.
    pub fn set_with_intern(&mut self, index: usize, value: bool) {
        let mut new_inner = (*self.inner).clone();
        if value {
            new_inner.insert(index);
        } else {
            new_inner.remove(index);
        }
        self.inner = intern_l1_tracked(new_inner);
    }

    pub fn set(&mut self, index: usize, value: bool) {
        let mut new_inner = (*self.inner).clone();
        if value {
            new_inner.insert(index);
        } else {
            new_inner.remove(index);
        }
        self.inner = Arc::new(new_inner);
    }

    /// Removes an index from the set. Returns true if the index was present.
    pub fn remove_with_intern(&mut self, index: usize) -> bool {
        if !self.inner.contains(index) {
            return false;
        }
        let result = Arc::make_mut(&mut self.inner).remove(index);
        self.inner = intern_l1_tracked(Arc::unwrap_or_clone(std::mem::take(&mut self.inner)));
        result
    }

    pub fn remove(&mut self, index: usize) -> bool {
        self.remove_with_intern(index)
    }

    pub fn ensure_interned(&mut self) {
        self.inner = intern_l1_tracked(Arc::unwrap_or_clone(std::mem::take(&mut self.inner)));
    }

    /// Removes all elements from the set.
    pub fn clear(&mut self) {
        self.inner = intern_l1_tracked(RangeSetBlaze::new());
    }

    pub fn inverted(&self) -> Self {
        &Self::max_ones() - self
    }

    pub fn union_with(&mut self, other: &Self) {
        *Arc::make_mut(&mut self.inner) |= &*other.inner;
    }

    pub fn intersection_with(&mut self, other: &Self) {
        *Arc::make_mut(&mut self.inner) = &*self.inner & &*other.inner;
    }

    /// Returns an iterator over the indices of the set bits.
    pub fn iter_indices(&self) -> Iter<'_> {
        Iter {
            iter_inner: self.inner.iter(),
            remaining: self.len(),
            is_all: self.is_all(),
            count: 0,
            max_val: usize::MAX,
        }
    }

    pub fn iter_up_to(&self, max: usize) -> Iter<'_> {
        let mut count: u128 = 0;
        for range in self.inner.ranges() {
            let start = *range.start();
            if start > max {
                break;
            }
            let end = *range.end();
            let effective_end = if end > max { max } else { end };
            count += (effective_end as u128 - start as u128) + 1;
        }

        Iter {
            iter_inner: self.inner.iter(),
            remaining: count.try_into().unwrap_or(usize::MAX),
            is_all: self.is_all(),
            count: 0,
            max_val: max,
        }
    }

    /// Returns an iterator over the inclusive [start, end] ranges of the set.
    pub fn iter_ranges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.inner.ranges().map(|r| (*r.start(), *r.end()))
    }

    /// Returns an iterator over booleans, indicating for each index from 0
    /// up to the largest index present in the set (inclusive) whether it's set or not.
    /// If the set is empty, the iterator is empty.
    pub fn iter_bits(&self) -> BitsIter<'_> {
        let is_all = self.is_all();
        if self.is_empty() {
            BitsIter {
                bitset: self,
                current_idx: 1, // Start beyond max_idx_to_iterate to yield nothing
                max_idx_to_iterate: 0,
                is_all,
                count: 0,
            }
        } else {
            let max_val_in_set = self.inner.last().unwrap_or(0); // unwrap is safe due to is_empty check
            BitsIter {
                bitset: self,
                current_idx: 0,
                max_idx_to_iterate: max_val_in_set,
                is_all,
                count: 0,
            }
        }
    }

    pub fn inner(&self) -> &RangeSetBlaze<usize> {
        &self.inner
    }

    pub fn find_good_permutation(
        sets: &[&Self],
    ) -> std::collections::HashMap<usize, usize> {
        use std::collections::{HashMap, HashSet, VecDeque};

        // --- Step 1: Build Adjacency Graph and Collect All Nodes ---
        let mut adj_graph: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
        let mut all_nodes: HashSet<usize> = HashSet::new();

        for set in sets.iter() {
            let mut iter = set.iter_indices().peekable();
            while let Some(u) = iter.next() {
                all_nodes.insert(u);
                if let Some(&v) = iter.peek() {
                    *adj_graph.entry(u).or_default().entry(v).or_default() += 1;
                    *adj_graph.entry(v).or_default().entry(u).or_default() += 1;
                }
            }
        }

        // --- Step 2: Generate Permutation Order ---
        let mut edges: Vec<(usize, usize, usize)> = Vec::new();
        for (&u, neighbors) in &adj_graph {
            for (&v, &weight) in neighbors {
                if u < v {
                    edges.push((u, v, weight));
                }
            }
        }
        edges.sort_unstable_by_key(|&(_, _, weight)| std::cmp::Reverse(weight));

        let mut used_nodes: HashSet<usize> = HashSet::new();
        let mut chains: Vec<VecDeque<usize>> = Vec::new();

        for &(u, v, _) in &edges {
            if used_nodes.contains(&u) || used_nodes.contains(&v) {
                continue;
            }

            let mut current_chain = VecDeque::from([u, v]);
            used_nodes.insert(u);
            used_nodes.insert(v);

            let mut current_end = v;
            loop {
                let best_neighbor = adj_graph.get(&current_end).and_then(|neighbors| {
                    neighbors
                        .iter()
                        .filter(|(node, _)| !used_nodes.contains(node))
                        .max_by_key(|(_, weight)| **weight)
                        .map(|(node, _)| *node)
                });

                if let Some(neighbor) = best_neighbor {
                    current_chain.push_back(neighbor);
                    used_nodes.insert(neighbor);
                    current_end = neighbor;
                } else {
                    break;
                }
            }

            let mut current_start = u;
            loop {
                let best_neighbor = adj_graph.get(&current_start).and_then(|neighbors| {
                    neighbors
                        .iter()
                        .filter(|(node, _)| !used_nodes.contains(node))
                        .max_by_key(|(_, weight)| **weight)
                        .map(|(node, _)| *node)
                });

                if let Some(neighbor) = best_neighbor {
                    current_chain.push_front(neighbor);
                    used_nodes.insert(neighbor);
                    current_start = neighbor;
                } else {
                    break;
                }
            }
            chains.push(current_chain);
        }

        for &node in &all_nodes {
            if !used_nodes.contains(&node) {
                chains.push(VecDeque::from([node]));
            }
        }

        chains
            .into_iter()
            .flat_map(|chain| chain.into_iter())
            .enumerate()
            .map(|(new_idx, old_idx)| (old_idx, new_idx))
            .collect()
    }

    pub fn symmetric_difference(&self, other: &Self) -> Self {
        self ^ other
    }

    /// Constrains the bitset to a maximum value (inclusive).
    /// Any bits set for indices greater than `max` will be cleared.
    pub fn constrain(&mut self, max: usize) {
        if let Some(len) = max.checked_add(1) {
            *self &= &Self::ones(len);
        }
        // If max is usize::MAX, checked_add returns None, and we do nothing,
        // which is correct because all values are already <= usize::MAX.
    }
}

// --- Iterator ---
const FULL_ITER_WARNING_THRESHOLD: usize = 1_000_000;

pub struct Iter<'a> {
    iter_inner: range_set_blaze::Iter<usize, range_set_blaze::RangesIter<'a, usize>>,
    remaining: usize,
    is_all: bool,
    count: usize,
    max_val: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        if self.is_all {
            self.count += 1;
            if self.count == FULL_ITER_WARNING_THRESHOLD {
                // eprintln!(
                //     "Warning: Iterating over a full HybridBitset. This may take a very long time."
                // );
                panic!(
                    "Warning: Iterating over a full HybridBitset. This may take a very long time."
                );
            }
        }
        match self.iter_inner.next() {
            Some(item) => {
                if item > self.max_val {
                    self.remaining = 0;
                    return None;
                }
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

impl<'a> IntoIterator for &'a RangeSet {
    type Item = usize;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_indices()
    }
}

impl IntoIterator for RangeSet {
    type Item = usize;
    type IntoIter = range_set_blaze::IntoIter<usize>;

    fn into_iter(self) -> Self::IntoIter {
        // To get an owning iterator, we need to take ownership of the RangeSetBlaze.
        // We can do this by cloning the inner data if the Arc is shared.
        Arc::try_unwrap(self.inner)
            .unwrap_or_else(|arc| (*arc).clone())
            .into_iter()
    }
}

// --- Boolean Iterator ---
pub struct BitsIter<'a> {
    bitset: &'a RangeSet,
    current_idx: usize,
    max_idx_to_iterate: usize, // Inclusive
    is_all: bool,
    count: usize,
}

impl<'a> Iterator for BitsIter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_all {
            self.count += 1;
            if self.count == FULL_ITER_WARNING_THRESHOLD {
                eprintln!(
                    "Warning: Iterating bits over a full HybridBitset. This may take a very long time."
                );
            }
        }
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

impl FromIterator<usize> for RangeSet {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        Self::from_iter(iter)
    }
}

impl Extend<usize> for RangeSet {
    fn extend<I: IntoIterator<Item = usize>>(&mut self, iter: I) {
        // Avoids clone if Arc is unique
        Arc::make_mut(&mut self.inner).extend(iter);
        // Re-intern the potentially modified RangeSetBlaze
        self.inner = intern_l1_tracked((*self.inner).clone());
    }
}

// --- Bitwise Operations (Creating New Sets) ---

impl BitAnd for &RangeSet {
    type Output = RangeSet;

    fn bitand(self, rhs: Self) -> Self::Output {
        // Optimization: if pointers are equal, return clone
        if Arc::ptr_eq(&self.inner, &rhs.inner) {
            return self.clone();
        }

        if self.is_simple() || rhs.is_simple() {
            let result_inner = &*self.inner & &*rhs.inner;
            let result = RangeSet {
                inner: intern_l1_tracked(result_inner),
            };
            return result;
        }
        if let Some(cached) = get_l1_op_cache_tracked(cache::BinOp::And, &self.inner, &rhs.inner) {
            return RangeSet { inner: cached };
        }
        if let Some(cached) = get_l1_op_cache_tracked(cache::BinOp::And, &rhs.inner, &self.inner) {
            return RangeSet { inner: cached };
        }
        let result_inner = &*self.inner & &*rhs.inner;
        let result_acc = intern_l1_tracked(result_inner);

        put_l1_op_cache_tracked(
            cache::BinOp::And,
            self.inner.clone(),
            rhs.inner.clone(),
            result_acc.clone(),
        );
        RangeSet { inner: result_acc }
    }
}

impl BitOr for &RangeSet {
    type Output = RangeSet;

    fn bitor(self, rhs: Self) -> Self::Output {
        let profile_enabled = L1_OP_CACHE_PROFILE_ENABLED.load(AtomicOrdering::Relaxed);
        if Arc::ptr_eq(&self.inner, &rhs.inner) {
            return self.clone();
        }

        if self.is_simple() || rhs.is_simple() {
            let result_inner = &*self.inner | &*rhs.inner;
            let result = RangeSet {
                inner: intern_l1_tracked(result_inner),
            };
            return result;
        }
        if let Some(cached) = get_l1_op_cache_tracked(cache::BinOp::Or, &self.inner, &rhs.inner) {
            if profile_enabled {
                L1_OP_CACHE_OR_HITS.fetch_add(1, AtomicOrdering::Relaxed);
            }
            return RangeSet { inner: cached };
        }
        if let Some(cached) = get_l1_op_cache_tracked(cache::BinOp::Or, &rhs.inner, &self.inner) {
            if profile_enabled {
                L1_OP_CACHE_OR_HITS.fetch_add(1, AtomicOrdering::Relaxed);
            }
            return RangeSet { inner: cached };
        }
        if profile_enabled {
            L1_OP_CACHE_OR_MISSES.fetch_add(1, AtomicOrdering::Relaxed);
        }
        let result_inner = &*self.inner | &*rhs.inner;
        let result_acc = intern_l1_tracked(result_inner);

        put_l1_op_cache_tracked(
            cache::BinOp::Or,
            self.inner.clone(),
            rhs.inner.clone(),
            result_acc.clone(),
        );
        RangeSet { inner: result_acc }
    }
}

impl BitXor for &RangeSet {
    type Output = RangeSet;

    fn bitxor(self, rhs: Self) -> Self::Output {
        if Arc::ptr_eq(&self.inner, &rhs.inner) {
            return RangeSet::zeros();
        }
        if self.is_simple() || rhs.is_simple() {
            let result_inner = &*self.inner ^ &*rhs.inner;
            return RangeSet {
                inner: intern_l1_tracked(result_inner),
            };
        }
        if let Some(cached) = get_l1_op_cache_tracked(cache::BinOp::Xor, &self.inner, &rhs.inner) {
            return RangeSet { inner: cached };
        }
        if let Some(cached) = get_l1_op_cache_tracked(cache::BinOp::Xor, &rhs.inner, &self.inner) {
            return RangeSet { inner: cached };
        }

        let result_inner = &*self.inner ^ &*rhs.inner;
        let result_acc = intern_l1_tracked(result_inner);

        put_l1_op_cache_tracked(
            cache::BinOp::Xor,
            self.inner.clone(),
            rhs.inner.clone(),
            result_acc.clone(),
        );

        RangeSet { inner: result_acc }
    }
}

impl Sub for &RangeSet {
    type Output = RangeSet;

    fn sub(self, rhs: Self) -> Self::Output {
        if Arc::ptr_eq(&self.inner, &rhs.inner) {
            return RangeSet::zeros();
        }

        if self.is_simple() || rhs.is_simple() {
            let result_inner = &*self.inner - &*rhs.inner;
            let result = RangeSet {
                inner: intern_l1_tracked(result_inner),
            };
            return result;
        }
        if let Some(cached) = get_l1_op_cache_tracked(cache::BinOp::Sub, &self.inner, &rhs.inner) {
            return RangeSet { inner: cached };
        }
        let result_inner = &*self.inner - &*rhs.inner;
        let result_acc = intern_l1_tracked(result_inner);

        put_l1_op_cache_tracked(
            cache::BinOp::Sub,
            self.inner.clone(),
            rhs.inner.clone(),
            result_acc.clone(),
        );

        RangeSet { inner: result_acc }
    }
}

// --- In-place Bitwise Operations ---
impl BitAndAssign for RangeSet {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = &*self & &rhs;
    }
}
impl BitOrAssign for RangeSet {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = &*self | &rhs;
    }
}
impl BitXorAssign for RangeSet {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = &*self ^ &rhs;
    }
}
impl SubAssign for RangeSet {
    fn sub_assign(&mut self, rhs: Self) {
        *self = &*self - &rhs;
    }
}

impl BitAndAssign<&RangeSet> for RangeSet {
    fn bitand_assign(&mut self, rhs: &RangeSet) {
        *self = &*self & rhs;
    }
}
impl BitOrAssign<&RangeSet> for RangeSet {
    fn bitor_assign(&mut self, rhs: &RangeSet) {
        if Arc::ptr_eq(&self.inner, &rhs.inner) {
            return;
        }
        // A clone-modify-reintern pattern is safest with interning.
        // This avoids the overhead of the full caching logic in the `bitor` operator.
        let mut new_inner = (*self.inner).clone();
        new_inner |= (*rhs.inner).clone();
        self.inner = intern_l1_tracked(new_inner);
    }
}
impl BitXorAssign<&RangeSet> for RangeSet {
    fn bitxor_assign(&mut self, rhs: &RangeSet) {
        *self = &*self ^ rhs;
    }
}
impl SubAssign<&RangeSet> for RangeSet {
    fn sub_assign(&mut self, rhs: &RangeSet) {
        *self = &*self - rhs;
    }
}

// --- Equality, Hashing, Ordering ---
impl PartialEq for RangeSet {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner) || *self.inner == *other.inner
    }
}

impl PartialOrd for RangeSet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RangeSet {
    fn cmp(&self, other: &Self) -> Ordering {
        if Arc::ptr_eq(&self.inner, &other.inner) {
            return Ordering::Equal;
        }
        self.inner.cmp(&other.inner)
    }
}

impl Hash for RangeSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

// --- Conversions ---
use bitvec::prelude::*;

impl Into<BitVec<usize, Lsb0>> for RangeSet {
    fn into(self) -> BitVec<usize, Lsb0> {
        todo!("Conversion from HybridBitset (RangeSetBlaze based) to BitVec is not directly implemented yet.")
    }
}

impl From<BitVec<usize, Lsb0>> for RangeSet {
    fn from(bitvec: BitVec<usize, Lsb0>) -> Self {
        RangeSet {
            inner: intern_l1_tracked(RangeSetBlaze::from_iter(bitvec.iter_ones())),
        }
    }
}

impl From<RangeSetBlaze<usize>> for RangeSet {
    fn from(range_set: RangeSetBlaze<usize>) -> Self {
        RangeSet {
            inner: intern_l1_tracked(range_set),
        }
    }
}

impl From<&RangeSetBlaze<usize>> for RangeSet {
    fn from(range_set: &RangeSetBlaze<usize>) -> Self {
        RangeSet {
            inner: intern_l1_tracked(range_set.clone()),
        }
    }
}

impl From<&Bitset> for RangeSet {
    fn from(bitset: &Bitset) -> Self {
        RangeSet::from_iter(bitset.iter_indices())
    }
}

impl From<Bitset> for RangeSet {
    fn from(bitset: Bitset) -> Self {
        RangeSet::from_iter(bitset.iter_indices())
    }
}

impl From<crate::datastructures::AbstractWeight> for RangeSet {
    fn from(aw: crate::datastructures::AbstractWeight) -> Self {
        use crate::datastructures::AbstractWeight;
        match aw {
            AbstractWeight::RangeSet(rsb) => rsb,
            AbstractWeight::Factorized(fw) => {
                // For factorized weights, we can extract the token sets directly
                // without the expensive N×M expansion if we're in symbol-heavy mode
                // (num_tsids == 1) where each pair represents just token positions.
                let num_tsids = get_num_tsids();
                if num_tsids <= 1 {
                    // Symbol-heavy mode: pairs are (tsid_set, token_set) where token_set
                    // contains the actual token positions directly
                    let mut result = RangeSet::zeros();
                    for (_tsid_set, token_set) in fw.pairs() {
                        result |= token_set;
                    }
                    result
                } else {
                    // Weight-heavy mode: need full expansion (O(N×M))
                    // This triggers the guard if ALLOW_FACTORIZED_EXPANSION is not set
                    RangeSet::from(fw.expand_to_rsb())
                }
            }
            AbstractWeight::RangeMap(rm) => {
                let num_tsids = get_num_tsids();
                if num_tsids <= 1 {
                    let mut result = RangeSet::zeros();
                    for (token_range, tsid_set) in rm.map.range_values() {
                        if tsid_set.contains(0) {
                            result |= &RangeSet::from(RangeSetBlaze::from_iter([token_range.clone()]));
                        }
                    }
                    result
                } else {
                    RangeSet::from(rm.expand_to_rsb())
                }
            }
        }
    }
}

// --- Operations on owned values ---
impl BitAnd<RangeSet> for RangeSet {
    type Output = RangeSet;
    fn bitand(self, rhs: RangeSet) -> Self::Output {
        &self & &rhs
    }
}
impl BitOr<RangeSet> for RangeSet {
    type Output = RangeSet;
    fn bitor(self, rhs: RangeSet) -> Self::Output {
        &self | &rhs
    }
}
impl BitXor<RangeSet> for RangeSet {
    type Output = RangeSet;
    fn bitxor(self, rhs: RangeSet) -> Self::Output {
        &self ^ &rhs
    }
}
impl Sub<RangeSet> for RangeSet {
    type Output = RangeSet;
    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl<'a> BitAnd<&'a RangeSet> for RangeSet {
    type Output = RangeSet;
    fn bitand(self, rhs: &'a RangeSet) -> Self::Output {
        &self & rhs
    }
}
impl<'a> BitOr<&'a RangeSet> for RangeSet {
    type Output = RangeSet;
    fn bitor(self, rhs: &'a RangeSet) -> Self::Output {
        &self | rhs
    }
}
impl<'a> BitXor<&'a RangeSet> for RangeSet {
    type Output = RangeSet;
    fn bitxor(self, rhs: &'a RangeSet) -> Self::Output {
        &self ^ rhs
    }
}
impl<'a> Sub<&'a RangeSet> for RangeSet {
    type Output = RangeSet;
    fn sub(self, rhs: &'a RangeSet) -> Self::Output {
        &self - rhs
    }
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use deterministic_hash::DeterministicHasher;
    use std::collections::BTreeSet;
    use std::iter::FromIterator;

    // Thresholds are now defined in the cache module, but we can keep these for test logic.
    const SPARSE_TO_DENSE_THRESHOLD: usize = 128;
    const DENSE_TO_SPARSE_THRESHOLD: usize = 64;

    #[test]
    fn test_new_empty_len() {
        let set = RangeSet::zeros();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }

    #[test]
    fn test_insert_basic() {
        let mut set = RangeSet::zeros();
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
        let mut set = RangeSet::from_iter(vec![10, 20, 30]);
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
        let set = RangeSet::from_iter(indices.clone());
        let mut collected: Vec<usize> = set.iter_indices().collect();
        collected.sort_unstable();
        let mut expected = indices;
        expected.sort_unstable();
        assert_eq!(collected, expected);
        assert_eq!(set.iter_up_to(100).len(), expected.len()); // Check ExactSizeIterator impl
    }

    #[test]
    fn test_into_iteration() {
        let indices = vec![5, 1, 100, 42];
        let set = RangeSet::from_iter(indices.clone());
        let mut collected: Vec<usize> = set.into_iter().collect(); // Consumes set
        collected.sort_unstable();
        let mut expected = indices;
        expected.sort_unstable();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_set_ops_sparse_sparse() {
        // Names are now conceptual, as internal repr is opaque
        let set1 = RangeSet::from_iter(vec![1, 2, 3, 10]);
        let set2 = RangeSet::from_iter(vec![3, 4, 5, 10]);

        let intersection = &set1 & &set2;
        let union = &set1 | &set2;
        let difference = &set1 - &set2;
        let sym_diff = &set1 ^ &set2;

        assert_eq!(
            intersection.iter_indices().collect::<BTreeSet<usize>>(),
            BTreeSet::from_iter(vec![3, 10])
        );
        assert_eq!(
            union.iter_indices().collect::<BTreeSet<usize>>(),
            BTreeSet::from_iter(vec![1, 2, 3, 4, 5, 10])
        );
        assert_eq!(
            difference.iter_indices().collect::<BTreeSet<usize>>(),
            BTreeSet::from_iter(vec![1, 2])
        );
        assert_eq!(
            sym_diff.iter_indices().collect::<BTreeSet<usize>>(),
            BTreeSet::from_iter(vec![1, 2, 4, 5])
        );
    }

    #[test]
    fn test_set_ops_dense_dense() {
        // Names are now conceptual
        let set1 = RangeSet::from_iter(0..SPARSE_TO_DENSE_THRESHOLD + 10);
        let set2 = RangeSet::from_iter(5..SPARSE_TO_DENSE_THRESHOLD + 20);

        let intersection = &set1 & &set2;
        let union = &set1 | &set2;
        let difference = &set1 - &set2;
        let sym_diff = &set1 ^ &set2;

        let intersection_expected: BTreeSet<usize> = (5..SPARSE_TO_DENSE_THRESHOLD + 10).collect();
        let union_expected: BTreeSet<usize> = (0..SPARSE_TO_DENSE_THRESHOLD + 20).collect();
        let difference_expected: BTreeSet<usize> = (0..5).collect();
        let sym_diff_expected: BTreeSet<usize> = (0..5)
            .chain(SPARSE_TO_DENSE_THRESHOLD + 10..SPARSE_TO_DENSE_THRESHOLD + 20)
            .collect();

        assert_eq!(
            intersection.iter_indices().collect::<BTreeSet<usize>>(),
            intersection_expected
        );
        assert_eq!(
            union.iter_indices().collect::<BTreeSet<usize>>(),
            union_expected
        );
        assert_eq!(
            difference.iter_indices().collect::<BTreeSet<usize>>(),
            difference_expected
        );
        assert_eq!(
            sym_diff.iter_indices().collect::<BTreeSet<usize>>(),
            sym_diff_expected
        );
    }

    #[test]
    fn test_set_ops_mixed() {
        // Names are now conceptual
        let set1_conceptually_sparse =
            RangeSet::from_iter(vec![1, 2, 3, SPARSE_TO_DENSE_THRESHOLD + 100]);
        let set2_conceptually_dense = RangeSet::from_iter(0..SPARSE_TO_DENSE_THRESHOLD + 5);

        let intersection1 = &set1_conceptually_sparse & &set2_conceptually_dense;
        let intersection1_expected: BTreeSet<usize> = vec![1, 2, 3].into_iter().collect();
        assert_eq!(
            intersection1.iter_indices().collect::<BTreeSet<usize>>(),
            intersection1_expected
        );

        let union1 = &set1_conceptually_sparse | &set2_conceptually_dense;
        let mut union1_expected: BTreeSet<usize> = (0..SPARSE_TO_DENSE_THRESHOLD + 5).collect();
        union1_expected.insert(SPARSE_TO_DENSE_THRESHOLD + 100);
        assert_eq!(
            union1.iter_indices().collect::<BTreeSet<usize>>(),
            union1_expected
        );

        let diff1 = &set1_conceptually_sparse - &set2_conceptually_dense;
        let diff1_expected: BTreeSet<usize> =
            vec![SPARSE_TO_DENSE_THRESHOLD + 100].into_iter().collect();
        assert_eq!(
            diff1.iter_indices().collect::<BTreeSet<usize>>(),
            diff1_expected
        );

        let diff2 = &set2_conceptually_dense - &set1_conceptually_sparse;
        let mut diff2_expected: BTreeSet<usize> = (0..SPARSE_TO_DENSE_THRESHOLD + 5).collect();
        diff2_expected.remove(&1);
        diff2_expected.remove(&2);
        diff2_expected.remove(&3);
        assert_eq!(
            diff2.iter_indices().collect::<BTreeSet<usize>>(),
            diff2_expected
        );

        let xor1 = &set1_conceptually_sparse ^ &set2_conceptually_dense;
        let mut xor1_expected = diff2_expected.clone();
        xor1_expected.insert(SPARSE_TO_DENSE_THRESHOLD + 100);
        assert_eq!(
            xor1.iter_indices().collect::<BTreeSet<usize>>(),
            xor1_expected
        );
    }

    #[test]
    fn test_equality_and_hash() {
        let set1 = RangeSet::from_iter(vec![1, 5, 10]);
        let set1_clone = RangeSet::from_iter(vec![1, 5, 10]); // Same elements
        let set2 = RangeSet::from_iter(vec![1, 5, 11]); // Different elements
        let empty_set = RangeSet::zeros();

        assert_eq!(set1, set1_clone);
        assert_ne!(set1, set2);
        assert_ne!(set1, empty_set);

        // Test pointer equality for interned values
        assert!(Arc::ptr_eq(&set1.inner, &set1_clone.inner));

        use std::collections::hash_map::DefaultHasher;
        let hash = |s: &RangeSet| -> u64 {
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
        let mut set = RangeSet::zeros();
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
        let mut set = RangeSet::from_iter(0..SPARSE_TO_DENSE_THRESHOLD + 10);
        assert!(!set.is_empty());
        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);

        let mut set2 = RangeSet::from_iter(vec![1, 2, 3]);
        assert!(!set2.is_empty());
        set2.clear();
        assert!(set2.is_empty());
        assert_eq!(set2.len(), 0);
    }

    #[test]
    fn test_assign_ops() {
        let set1_orig = RangeSet::from_iter(vec![1, 2, 10]);
        let set2 = RangeSet::from_iter(vec![2, 3, 20]);

        let mut set1 = set1_orig.clone();
        set1 |= set2.clone();
        assert_eq!(
            set1.iter_indices().collect::<BTreeSet<_>>(),
            BTreeSet::from_iter(vec![1, 2, 3, 10, 20])
        );

        let set3_orig = RangeSet::from_iter(0..DENSE_TO_SPARSE_THRESHOLD); // Conceptual dense
        let set4 =
            RangeSet::from_iter((DENSE_TO_SPARSE_THRESHOLD / 2)..DENSE_TO_SPARSE_THRESHOLD + 10);
        let expected_and =
            (DENSE_TO_SPARSE_THRESHOLD / 2..DENSE_TO_SPARSE_THRESHOLD).collect::<BTreeSet<_>>();
        let mut set3 = set3_orig.clone();
        set3 &= set4.clone();
        assert_eq!(set3.iter_indices().collect::<BTreeSet<_>>(), expected_and);

        let mut set5 = RangeSet::from_iter(vec![1, 2, 3]);
        let set6 = RangeSet::from_iter(vec![3, 4, 5]);
        set5 ^= set6.clone();
        assert_eq!(
            set5.iter_indices().collect::<BTreeSet<_>>(),
            BTreeSet::from_iter(vec![1, 2, 4, 5])
        );

        let mut set7 = RangeSet::from_iter(vec![1, 2, 3, 4, 5]);
        let set8 = RangeSet::from_iter(vec![2, 4, 6]);
        set7 -= set8.clone();
        assert_eq!(
            set7.iter_indices().collect::<BTreeSet<_>>(),
            BTreeSet::from_iter(vec![1, 3, 5])
        );
    }

    #[test]
    fn test_assign_ops_ref() {
        let set1_orig = RangeSet::from_iter(vec![1, 2, 10]);
        let set2 = RangeSet::from_iter(vec![2, 3, 20]);

        let mut set1 = set1_orig.clone();
        set1 |= &set2;
        assert_eq!(
            set1.iter_indices().collect::<BTreeSet<_>>(),
            BTreeSet::from_iter(vec![1, 2, 3, 10, 20])
        );

        let set3_orig = RangeSet::from_iter(0..DENSE_TO_SPARSE_THRESHOLD);
        let set4 =
            RangeSet::from_iter((DENSE_TO_SPARSE_THRESHOLD / 2)..DENSE_TO_SPARSE_THRESHOLD + 10);
        let expected_and =
            (DENSE_TO_SPARSE_THRESHOLD / 2..DENSE_TO_SPARSE_THRESHOLD).collect::<BTreeSet<_>>();
        let mut set3 = set3_orig.clone();
        set3 &= &set4;
        assert_eq!(set3.iter_indices().collect::<BTreeSet<_>>(), expected_and);

        let mut set5 = RangeSet::from_iter(vec![1, 2, 3]);
        let set6 = RangeSet::from_iter(vec![3, 4, 5]);
        set5 ^= &set6;
        assert_eq!(
            set5.iter_indices().collect::<BTreeSet<_>>(),
            BTreeSet::from_iter(vec![1, 2, 4, 5])
        );

        let mut set7 = RangeSet::from_iter(vec![1, 2, 3, 4, 5]);
        let set8 = RangeSet::from_iter(vec![2, 4, 6]);
        set7 -= &set8;
        assert_eq!(
            set7.iter_indices().collect::<BTreeSet<_>>(),
            BTreeSet::from_iter(vec![1, 3, 5])
        );
    }

    #[test]
    fn test_dense_dense_edge_cases() {
        // Conceptual names
        let d1 = RangeSet::zeros();
        let d2 = RangeSet::zeros();
        let d3 = RangeSet::from_iter(0..DENSE_TO_SPARSE_THRESHOLD);

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

        let d4 = RangeSet::from_iter(0..5);
        let d5 = RangeSet::from_iter(3..10);

        let inter = &d4 & &d5;
        assert_eq!(
            inter.iter_indices().collect::<BTreeSet<_>>(),
            BTreeSet::from_iter(vec![3, 4])
        );

        let union = &d4 | &d5;
        assert_eq!(
            union.iter_indices().collect::<BTreeSet<_>>(),
            (0..10).collect::<BTreeSet<_>>()
        );

        let diff = &d4 - &d5;
        assert_eq!(
            diff.iter_indices().collect::<BTreeSet<_>>(),
            BTreeSet::from_iter(vec![0, 1, 2])
        );

        let sym_diff = &d4 ^ &d5;
        assert_eq!(
            sym_diff.iter_indices().collect::<BTreeSet<_>>(),
            BTreeSet::from_iter(vec![0, 1, 2, 5, 6, 7, 8, 9])
        );
    }

    #[test]
    fn test_from_iterator_trait() {
        // Renamed to avoid conflict
        let data = vec![10, 20, 10, 30, 20];
        let set: RangeSet = data.into_iter().collect();

        let expected: BTreeSet<usize> = vec![10, 20, 30].into_iter().collect();
        assert_eq!(set.iter_indices().collect::<BTreeSet<_>>(), expected);
    }

    #[test]
    fn test_iter_bits() {
        let empty_set = RangeSet::zeros();
        assert_eq!(
            empty_set.iter_bits().collect::<Vec<bool>>(),
            Vec::<bool>::new()
        );
        assert_eq!(empty_set.iter_bits().len(), 0);

        let sparse_set = RangeSet::from_iter(vec![1, 3]);
        let expected_sparse_bools = vec![false, true, false, true];
        assert_eq!(
            sparse_set.iter_bits().collect::<Vec<bool>>(),
            expected_sparse_bools
        );
        assert_eq!(sparse_set.iter_bits().len(), expected_sparse_bools.len());

        // Test with a set that would have been dense
        let dense_like_set = RangeSet::from_iter(vec![1, 3]); // Max index 3
                                                                  // RangeSetBlaze doesn't have an explicit dense conversion, iter_bits uses .last()
        assert_eq!(
            dense_like_set.iter_bits().collect::<Vec<bool>>(),
            expected_sparse_bools
        );
        assert_eq!(
            dense_like_set.iter_bits().len(),
            expected_sparse_bools.len()
        );

        let empty_set_from_non_empty = RangeSet::from_iter(vec![5]);
        let _ = empty_set_from_non_empty.inner.last(); // just to use it
        let mut empty_set_cleared = RangeSet::from_iter(vec![5]);
        empty_set_cleared.clear();
        assert_eq!(
            empty_set_cleared.iter_bits().collect::<Vec<bool>>(),
            Vec::<bool>::new()
        );
        assert_eq!(empty_set_cleared.iter_bits().len(), 0);
    }

    #[test]
    fn test_ones() {
        let set_ones_small = RangeSet::ones(4); // 0, 1, 2, 3
        assert_eq!(set_ones_small.len(), 4);
        assert!(set_ones_small.contains(0));
        assert!(set_ones_small.contains(1));
        assert!(set_ones_small.contains(2));
        assert!(set_ones_small.contains(3));
        assert!(!set_ones_small.contains(4));

        let len = SPARSE_TO_DENSE_THRESHOLD + 5;
        let set_ones_large = RangeSet::ones(len + 1); // Corrected: len is exclusive upper bound for RangeSetBlaze
        assert_eq!(set_ones_large.len(), SPARSE_TO_DENSE_THRESHOLD + 6);
        for i in 0..=(SPARSE_TO_DENSE_THRESHOLD + 5) {
            assert!(set_ones_large.contains(i));
        }
        assert!(!set_ones_large.contains(SPARSE_TO_DENSE_THRESHOLD + 6));

        // Test edge case for usize::MAX
        // This test might be very slow or OOM with RangeSetBlaze if it tries to create a huge range.
        // let set_ones_max = HybridBitset::ones(usize::MAX); // This would be 0..=usize::MAX-1
        // assert!(!set_ones_max.is_empty()); //
        // assert_eq!(set_ones_max.len(), usize::MAX); // This is correct

        let set_ones_one = RangeSet::ones(1); // Should contain only 0
        assert_eq!(set_ones_one.len(), 1);
        assert!(set_ones_one.contains(0));
        assert!(!set_ones_one.contains(1));

        let set_ones_zero = RangeSet::ones(0); // Should be empty
        assert_eq!(set_ones_zero.len(), 0);
        assert!(set_ones_zero.is_empty());
    }

    #[ignore]
    #[test]
    fn test_find_good_permutation() {
        // Example from the problem description
        let s1 = RangeSet::from_iter(vec![1, 2, 3, 4, 8, 9]);
        let s2 = RangeSet::from_iter(vec![2, 3, 5, 7, 8, 9, 15]);
        let s3 = RangeSet::from_iter(vec![1, 4, 5, 6, 7, 8, 12, 13]);

        let sets = vec![&s1, &s2, &s3];

        // Calculate original cost
        let original_cost: usize = sets.iter().map(|s| s.inner().ranges_len()).sum();
        assert_eq!(original_cost, 2 + 4 + 3); // 9

        // Find the permutation
        let perm_map = RangeSet::find_good_permutation(&sets);

        // Apply the permutation
        let apply_permutation =
            |set: &RangeSet, map: &std::collections::HashMap<usize, usize>| -> RangeSet {
                set.iter_indices().map(|val| *map.get(&val).unwrap()).collect()
            };

        let pi_s1 = apply_permutation(&s1, &perm_map);
        let pi_s2 = apply_permutation(&s2, &perm_map);
        let pi_s3 = apply_permutation(&s3, &perm_map);

        // Calculate new cost
        let new_cost = pi_s1.inner().ranges_len()
            + pi_s2.inner().ranges_len()
            + pi_s3.inner().ranges_len();

        // The heuristic should improve the cost for this non-trivial case.
        assert!(
            new_cost < original_cost,
            "Expected new cost {} to be less than original cost {}",
            new_cost,
            original_cost
        );

        // Test with sets that have no shared adjacencies
        let s4 = RangeSet::from_iter(vec![100, 200, 300]);
        let s5 = RangeSet::from_iter(vec![400, 500]);
        let disjoint_sets = vec![&s4, &s5];
        let original_disjoint_cost: usize =
            disjoint_sets.iter().map(|s| s.inner().ranges_len()).sum();
        let perm_map_disjoint = RangeSet::find_good_permutation(&disjoint_sets);
        let new_disjoint_cost: usize = disjoint_sets
            .iter()
            .map(|s| {
                apply_permutation(s, &perm_map_disjoint)
                    .inner()
                    .ranges_len()
            })
            .sum();
        assert!(
            new_disjoint_cost < original_disjoint_cost,
            "Expected new cost {} to be less than original cost {}",
            new_disjoint_cost,
            original_disjoint_cost
        );
    }

    #[test]
    fn test_sub_is_intersection_with_inverted() {
        let set_a = RangeSet::from_iter(vec![1, 2, 10, 100]);
        let set_b = RangeSet::from_iter(vec![2, 3, 20, 100]);

        let diff = &set_a - &set_b;
        let diff_with_inverted = &set_a & &set_b.inverted();

        let expected_diff = RangeSet::from_iter(vec![1, 10]);

        assert_eq!(diff, expected_diff);
        assert_eq!(diff_with_inverted, expected_diff);
    }

    #[test]
    fn test_constraint() {
        let original_set = RangeSet::from_iter(vec![1, 10, 100, 1000]);

        // Constraint that includes all elements
        let mut constrained1 = original_set.clone();
        constrained1.constrain(2000);
        assert_eq!(constrained1, original_set);

        // Constraint that cuts off some elements
        let mut constrained2 = original_set.clone();
        constrained2.constrain(500);
        let expected2 = RangeSet::from_iter(vec![1, 10, 100]);
        assert_eq!(constrained2, expected2);

        // Constraint that cuts off all elements
        let mut constrained3 = original_set.clone();
        constrained3.constrain(0);
        assert!(constrained3.is_empty());

        // Constraint at an edge
        let mut constrained4 = original_set.clone();
        constrained4.constrain(100);
        let expected4 = RangeSet::from_iter(vec![1, 10, 100]);
        assert_eq!(constrained4, expected4);

        // Constraint on an empty set
        let mut empty_set = RangeSet::zeros();
        empty_set.constrain(100);
        assert!(empty_set.is_empty());

        // Constraint with usize::MAX
        let mut constrained_max = original_set.clone();
        constrained_max.constrain(usize::MAX);
        assert_eq!(constrained_max, original_set);
    }

    #[test]
    fn test_extend() {
        let mut set = RangeSet::from_iter(vec![1, 5]);
        let new_elements = vec![5, 10, 20];
        set.extend(new_elements);

        let expected: BTreeSet<usize> = vec![1, 5, 10, 20].into_iter().collect();
        assert_eq!(set.iter_indices().collect::<BTreeSet<_>>(), expected);

        let mut empty_set = RangeSet::zeros();
        empty_set.extend(vec![100, 200]);
        let expected_empty: BTreeSet<usize> = vec![100, 200].into_iter().collect();
        assert_eq!(empty_set.iter_indices().collect::<BTreeSet<_>>(), expected_empty);

        let mut large_set = RangeSet::from_iter(0..10);
        large_set.extend(15..20);
        let expected_large: BTreeSet<usize> = (0..10).chain(15..20).collect();
        assert_eq!(large_set.iter_indices().collect::<BTreeSet<_>>(), expected_large);
    }

    #[test]
    fn test_display_format() {
        let set1 = RangeSet::from_iter(vec![0, 1, 2, 3, 4, 7, 9, 10, 11]);
        assert_eq!(format!("{}", set1), "[0..4, 7, 9..11]");

        let set2 = RangeSet::zeros();
        assert_eq!(format!("{}", set2), "[]");

        let set3 = RangeSet::from_iter(vec![42]);
        assert_eq!(format!("{}", set3), "[42]");

        let set4 = RangeSet::from_iter(vec![10, 12, 14]);
        assert_eq!(format!("{}", set4), "[10, 12, 14]");

        let set5 = RangeSet::ones(5); // 0..=4
        assert_eq!(format!("{}", set5), "[0..4]");
    }
}

