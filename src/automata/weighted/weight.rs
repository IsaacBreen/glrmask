//! RangeMap-based weight backend.
//!
//! The **Weight** type stores a set of (token, TSID) positions using TSID-outer
//! layout: a sorted map from TSID ranges to token `RangeSet`s.  This is the
//! sole weight representation used during DWA determinization and minimization.
//!
//! The **WeightTable** is the flat, cache-friendly layout used at inference
//! time.  It stores `(target_state, weight)` pairs indexed by `(tsid, state)`.
//!
//! The **RangeMap** is a generic sorted-interval-to-value map used for
//! vocabulary preprocessing (token → TSID mapping).

use serde::{Deserialize, Serialize};

use crate::ds::RangeSet;

/// A token-set ID.  Groups of tokens that behave identically through a
/// DWA state transition share the same TSID.
pub type Tsid = u32;

// ---------------------------------------------------------------------------
// RangeMap<V> — generic sorted interval map
// ---------------------------------------------------------------------------

/// A mapping from non-overlapping, half-open `[start, end)` ranges to values.
///
/// Used for vocabulary-level mappings such as token-ID → TSID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RangeMap<V> {
    /// Sorted entries: `(start, end, value)` where range is `[start, end)`.
    entries: Vec<(u32, u32, V)>,
}

impl<V: Clone + Eq> RangeMap<V> {
    /// Create an empty range map.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create from pre-sorted entries.
    pub fn from_sorted(entries: Vec<(u32, u32, V)>) -> Self {
        Self { entries }
    }

    /// Number of range entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up the value for a given key using binary search.
    pub fn get(&self, key: u32) -> Option<&V> {
        let idx = self
            .entries
            .binary_search_by(|&(start, end, _)| {
                if key < start {
                    std::cmp::Ordering::Greater
                } else if key >= end {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .ok()?;
        Some(&self.entries[idx].2)
    }

    /// Iterate over all entries as `(start, end, &value)`.
    pub fn iter(&self) -> impl Iterator<Item = (u32, u32, &V)> {
        self.entries.iter().map(|&(s, e, ref v)| (s, e, v))
    }

    /// Access entries as a slice.
    pub fn entries(&self) -> &[(u32, u32, V)] {
        &self.entries
    }
}

impl<V: Clone + Eq> Default for RangeMap<V> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WeightTable — flat TSID×state table for runtime
// ---------------------------------------------------------------------------

/// Weight layout using TSID-outer organization.
///
/// For each `(tsid, state)` pair, stores the resulting DWA transition
/// `(target_state, weight)`.  The outer dimension is TSID so that computing
/// a mask for a single token set requires a contiguous memory scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightTable {
    /// Number of DWA states.
    pub num_states: u32,
    /// Number of token-set IDs.
    pub num_tsids: u32,
    /// Flat table: `data[tsid * num_states + state] = (target_state, weight)`.
    /// `target_state == u32::MAX` means dead/no transition.
    pub data: Vec<(u32, i32)>,
}

impl WeightTable {
    /// Create a new weight table with all dead transitions.
    pub fn new(num_states: u32, num_tsids: u32) -> Self {
        let size = num_states as usize * num_tsids as usize;
        Self {
            num_states,
            num_tsids,
            data: vec![(u32::MAX, 0); size],
        }
    }

    /// Get the transition for `(tsid, state)`.
    #[inline]
    pub fn get(&self, tsid: u32, state: u32) -> (u32, i32) {
        self.data[tsid as usize * self.num_states as usize + state as usize]
    }

    /// Set the transition for `(tsid, state)`.
    #[inline]
    pub fn set(&mut self, tsid: u32, state: u32, target: u32, weight: i32) {
        self.data[tsid as usize * self.num_states as usize + state as usize] = (target, weight);
    }
}

// ---------------------------------------------------------------------------
// Weight — TSID-outer weight set for compilation
// ---------------------------------------------------------------------------

/// A weight set using TSID-outer layout.
///
/// Stores a sorted, non-overlapping map from TSID ranges to token `RangeSet`s.
/// In TSID-outer layout the outer key is the TSID and the value is a set of
/// token IDs.  This enables O(log n) lookup of the token set for a given TSID,
/// which is the hot path during mask computation.
///
/// A "position" in the flat DWA weight space is
/// `token_id * num_tsids + tsid`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weight {
    /// Sorted entries: `(tsid_lo, tsid_hi, token_set)` with **inclusive**
    /// ranges.  Non-overlapping and sorted by `tsid_lo`.
    entries: Vec<(u32, u32, RangeSet)>,
    /// Number of token-set IDs (always ≥ 1).
    num_tsids: u32,
}

impl Weight {
    // ---- Construction ----

    /// Create an empty weight.
    pub fn empty(num_tsids: u32) -> Self {
        Self {
            entries: Vec::new(),
            num_tsids: num_tsids.max(1),
        }
    }

    /// Construct from raw sorted entries (TSID-outer layout).
    ///
    /// Entries must be non-overlapping and sorted by TSID range.
    /// Empty token sets are silently filtered out.
    pub fn from_entries(entries: Vec<(u32, u32, RangeSet)>, num_tsids: u32) -> Self {
        let entries: Vec<_> = entries
            .into_iter()
            .filter(|(_, _, rs)| !rs.is_empty())
            .collect();
        Self {
            entries,
            num_tsids: num_tsids.max(1),
        }
    }

    /// Create a weight containing a single flat position.
    ///
    /// Position `p` decodes as `token = p / num_tsids`, `tsid = p % num_tsids`.
    pub fn from_position(pos: u32, num_tsids: u32) -> Self {
        let num_tsids = num_tsids.max(1);
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        Self {
            entries: vec![(tsid, tsid, RangeSet::from_range(token, token))],
            num_tsids,
        }
    }

    /// Create a weight where every TSID in `tsid_set` maps to the same
    /// token range `[token_start, token_end]`.
    pub fn from_uniform_tsid_set(
        token_start: u32,
        token_end: u32,
        tsid_set: &RangeSet,
        num_tsids: u32,
    ) -> Self {
        let num_tsids = num_tsids.max(1);
        if tsid_set.is_empty() || token_start > token_end {
            return Self::empty(num_tsids);
        }
        let token_rs = RangeSet::from_range(token_start, token_end);
        let entries: Vec<_> = tsid_set
            .iter_ranges()
            .map(|(lo, hi)| (lo, hi, token_rs.clone()))
            .collect();
        Self { entries, num_tsids }
    }

    /// Create a weight covering all positions from 0 to `max_position`
    /// (inclusive).
    pub fn all(max_position: u32, num_tsids: u32) -> Self {
        let num_tsids = num_tsids.max(1);
        if num_tsids == 1 {
            return Self {
                entries: vec![(0, 0, RangeSet::from_range(0, max_position))],
                num_tsids,
            };
        }

        let max_token = max_position / num_tsids;
        let max_tsid = max_position % num_tsids;
        let full_tokens = RangeSet::from_range(0, max_token);

        if max_tsid == num_tsids - 1 {
            return Self {
                entries: vec![(0, max_tsid, full_tokens)],
                num_tsids,
            };
        }

        let mut entries = Vec::with_capacity(2);
        // TSIDs 0..=max_tsid get all tokens 0..=max_token.
        entries.push((0, max_tsid, full_tokens));
        // TSIDs max_tsid+1..=num_tsids-1 get tokens 0..=max_token-1.
        if max_token > 0 && max_tsid + 1 <= num_tsids - 1 {
            let prefix_tokens = RangeSet::from_range(0, max_token - 1);
            entries.push((max_tsid + 1, num_tsids - 1, prefix_tokens));
        }
        Self { entries, num_tsids }
    }

    /// Construct from flat position ranges (position = token × num_tsids + tsid).
    ///
    /// The input is a `RangeSet` of flat positions.
    pub fn from_positions(positions: &RangeSet, num_tsids: u32) -> Self {
        let num_tsids = num_tsids.max(1);
        if positions.is_empty() {
            return Self::empty(num_tsids);
        }

        // Collect per-TSID token ranges.
        let mut tsid_tokens: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_tsids as usize];

        for (lo, hi) in positions.iter_ranges() {
            let lo_token = lo / num_tsids;
            let lo_tsid = lo % num_tsids;
            let hi_token = hi / num_tsids;
            let hi_tsid = hi % num_tsids;

            if lo_token == hi_token {
                for tsid in lo_tsid..=hi_tsid {
                    tsid_tokens[tsid as usize].push((lo_token, lo_token));
                }
            } else {
                // First token: TSIDs lo_tsid..=num_tsids-1.
                for tsid in lo_tsid..num_tsids {
                    tsid_tokens[tsid as usize].push((lo_token, lo_token));
                }
                // Middle tokens (full TSID range).
                if lo_token + 1 <= hi_token.saturating_sub(1) {
                    for bucket in tsid_tokens.iter_mut() {
                        bucket.push((lo_token + 1, hi_token - 1));
                    }
                }
                // Last token: TSIDs 0..=hi_tsid.
                for tsid in 0..=hi_tsid {
                    tsid_tokens[tsid as usize].push((hi_token, hi_token));
                }
            }
        }

        // Build entries, coalescing consecutive TSIDs with the same token set.
        let mut entries = Vec::new();
        let mut cur_start: Option<u32> = None;
        let mut cur_end: u32 = 0;
        let mut cur_rs = RangeSet::new();

        for (tsid, token_ranges) in tsid_tokens.into_iter().enumerate() {
            if token_ranges.is_empty() {
                if let Some(start) = cur_start.take() {
                    entries.push((start, cur_end, std::mem::take(&mut cur_rs)));
                }
                continue;
            }
            let rs = RangeSet::from_ranges(token_ranges);
            if let Some(start) = cur_start {
                if rs == cur_rs && cur_end + 1 == tsid as u32 {
                    cur_end = tsid as u32;
                    continue;
                }
                entries.push((start, cur_end, std::mem::take(&mut cur_rs)));
            }
            cur_start = Some(tsid as u32);
            cur_end = tsid as u32;
            cur_rs = rs;
        }
        if let Some(start) = cur_start {
            entries.push((start, cur_end, cur_rs));
        }

        Self { entries, num_tsids }
    }

    // ---- Queries ----

    /// Number of TSIDs.
    pub fn num_tsids(&self) -> u32 {
        self.num_tsids
    }

    /// Whether the weight is empty (no positions).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of outer range entries.
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Total number of sub-ranges (outer + sum of inner).
    pub fn num_ranges(&self) -> usize {
        self.entries
            .iter()
            .map(|(_, _, rs)| rs.num_ranges())
            .sum::<usize>()
            + self.entries.len()
    }

    /// Count the total number of positions in this weight.
    pub fn len(&self) -> u64 {
        let mut total: u64 = 0;
        for (tsid_lo, tsid_hi, token_set) in &self.entries {
            let tsid_span = (tsid_hi - tsid_lo + 1) as u64;
            total += tsid_span * token_set.cardinality();
        }
        total
    }

    /// Look up the token set for a specific TSID.
    pub fn tokens_for_tsid(&self, tsid: u32) -> RangeSet {
        match self.get_value(tsid) {
            Some(rs) => rs.clone(),
            None => RangeSet::new(),
        }
    }

    /// Check if a flat position is contained.
    ///
    /// Position `p` decodes as `token = p / num_tsids`, `tsid = p % num_tsids`.
    pub fn contains(&self, pos: u32) -> bool {
        let token = pos / self.num_tsids;
        let tsid = pos % self.num_tsids;
        self.get_value(tsid)
            .map_or(false, |rs| rs.contains(token))
    }

    /// Iterate over entries as `(tsid_lo, tsid_hi, &token_set)`.
    pub fn iter_entries(&self) -> impl Iterator<Item = (u32, u32, &RangeSet)> {
        self.entries.iter().map(|(lo, hi, rs)| (*lo, *hi, rs))
    }

    /// Access entries as a slice.
    pub fn entries(&self) -> &[(u32, u32, RangeSet)] {
        &self.entries
    }

    /// Expand to a sorted list of non-overlapping inclusive flat-position
    /// ranges `(lo, hi)` where `position = token * num_tsids + tsid`.
    pub fn expand_to_positions(&self) -> Vec<(u32, u32)> {
        let nt = self.num_tsids;
        let mut ranges = Vec::new();

        for (tsid_lo, tsid_hi, token_set) in &self.entries {
            let tsid_span = tsid_hi - tsid_lo + 1;
            for (t_lo, t_hi) in token_set.iter_ranges() {
                if nt <= 1 || tsid_span == nt {
                    // Full TSID coverage ⇒ contiguous positions.
                    let pos_lo = t_lo.saturating_mul(nt).saturating_add(*tsid_lo);
                    let pos_hi = t_hi.saturating_mul(nt).saturating_add(*tsid_hi);
                    ranges.push((pos_lo, pos_hi));
                } else {
                    // Partial TSID range ⇒ per-token blocks.
                    for token in t_lo..=t_hi {
                        let base = token.saturating_mul(nt);
                        ranges.push((
                            base.saturating_add(*tsid_lo),
                            base.saturating_add(*tsid_hi),
                        ));
                    }
                }
            }
        }

        // Sort and coalesce.
        ranges.sort_unstable();
        coalesce_ranges(&mut ranges);
        ranges
    }

    // ---- Set operations ----

    /// Compute the union of two weights.
    pub fn union(&self, other: &Self) -> Self {
        debug_assert_eq!(self.num_tsids, other.num_tsids);
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        Self::merge(self, other, |a, b| match (a, b) {
            (Some(a), Some(b)) => a.union(b),
            (Some(x), None) | (None, Some(x)) => x.clone(),
            (None, None) => RangeSet::new(),
        })
    }

    /// Compute the intersection of two weights.
    pub fn intersection(&self, other: &Self) -> Self {
        debug_assert_eq!(self.num_tsids, other.num_tsids);
        if self.is_empty() || other.is_empty() {
            return Self::empty(self.num_tsids);
        }
        Self::merge(self, other, |a, b| match (a, b) {
            (Some(a), Some(b)) => a.intersection(b),
            _ => RangeSet::new(),
        })
    }

    /// Compute the set difference `self − other`.
    pub fn difference(&self, other: &Self) -> Self {
        debug_assert_eq!(self.num_tsids, other.num_tsids);
        if self.is_empty() || other.is_empty() {
            return self.clone();
        }
        Self::merge(self, other, |a, b| match (a, b) {
            (Some(a), Some(b)) => a.difference(b),
            (Some(a), None) => a.clone(),
            _ => RangeSet::new(),
        })
    }

    /// Compute the complement within `[0, max_position]`.
    pub fn complement(&self, max_position: u32) -> Self {
        Self::all(max_position, self.num_tsids).difference(self)
    }

    /// Compute `self | !other` (divide).
    ///
    /// For each TSID, the result token set is
    /// `self_tokens | (full_tokens − other_tokens)` where `full_tokens` is
    /// `[0, max_token]`.  Requires an explicit `max_token` to define the
    /// complement universe.
    pub fn divide(&self, other: &Self, max_token: u32) -> Self {
        debug_assert_eq!(self.num_tsids, other.num_tsids);
        let full = RangeSet::from_range(0, max_token);
        Self::merge(self, other, |a, b| {
            let rhs_comp = match b {
                Some(b) => full.difference(b),
                None => full.clone(),
            };
            match a {
                Some(a) => a.union(&rhs_comp),
                None => rhs_comp,
            }
        })
    }

    /// Check whether two weights are disjoint.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        debug_assert_eq!(self.num_tsids, other.num_tsids);
        let mut ai = 0;
        let mut bi = 0;
        while ai < self.entries.len() && bi < other.entries.len() {
            let (a_lo, a_hi, ref a_rs) = self.entries[ai];
            let (b_lo, b_hi, ref b_rs) = other.entries[bi];
            if a_hi < b_lo {
                ai += 1;
            } else if b_hi < a_lo {
                bi += 1;
            } else {
                if !a_rs.is_disjoint(b_rs) {
                    return false;
                }
                if a_hi <= b_hi {
                    ai += 1;
                } else {
                    bi += 1;
                }
            }
        }
        true
    }

    /// Check whether `self ⊆ other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.difference(other).is_empty()
    }

    // ---- Internal ----

    /// Binary-search for the token set at a given TSID.
    fn get_value(&self, tsid: u32) -> Option<&RangeSet> {
        match self.entries.binary_search_by(|&(lo, hi, _)| {
            if tsid < lo {
                std::cmp::Ordering::Greater
            } else if tsid > hi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        }) {
            Ok(idx) => Some(&self.entries[idx].2),
            Err(_) => None,
        }
    }

    /// Merge two weights by sweeping over the TSID key space.
    ///
    /// All boundary points from both maps are collected, the key space is
    /// partitioned into uniform sub-intervals, and the `combine` closure is
    /// called for each.  Adjacent intervals with identical results are
    /// coalesced.
    fn merge<F>(a: &Weight, b: &Weight, combine: F) -> Weight
    where
        F: Fn(Option<&RangeSet>, Option<&RangeSet>) -> RangeSet,
    {
        let num_tsids = a.num_tsids;

        // Collect all boundary points (start and end+1 of each entry).
        let cap = 2 * (a.entries.len() + b.entries.len());
        let mut boundaries = Vec::with_capacity(cap);
        for &(lo, hi, _) in &a.entries {
            boundaries.push(lo);
            if let Some(next) = hi.checked_add(1) {
                boundaries.push(next);
            }
        }
        for &(lo, hi, _) in &b.entries {
            boundaries.push(lo);
            if let Some(next) = hi.checked_add(1) {
                boundaries.push(next);
            }
        }
        boundaries.sort_unstable();
        boundaries.dedup();

        if boundaries.is_empty() {
            return Weight::empty(num_tsids);
        }

        let mut entries: Vec<(u32, u32, RangeSet)> = Vec::new();
        let mut cur_start: Option<u32> = None;
        let mut cur_end: u32 = 0;
        let mut cur_value = RangeSet::new();

        for (idx, &start) in boundaries.iter().enumerate() {
            let end = if idx + 1 < boundaries.len() {
                boundaries[idx + 1] - 1
            } else {
                // Last boundary — clip to valid TSID range.
                num_tsids - 1
            };
            if start > end {
                continue;
            }

            let a_val = a.get_value(start);
            let b_val = b.get_value(start);
            let combined = combine(a_val, b_val);

            if combined.is_empty() {
                if let Some(range_start) = cur_start.take() {
                    entries.push((range_start, cur_end, std::mem::take(&mut cur_value)));
                }
                continue;
            }

            if let Some(range_start) = cur_start {
                if cur_value == combined && cur_end.checked_add(1) == Some(start) {
                    cur_end = end;
                    continue;
                }
                entries.push((range_start, cur_end, std::mem::take(&mut cur_value)));
            }

            cur_start = Some(start);
            cur_end = end;
            cur_value = combined;
        }

        if let Some(range_start) = cur_start {
            entries.push((range_start, cur_end, cur_value));
        }

        Weight { entries, num_tsids }
    }
}

// ---- Trait impls ----

impl PartialEq for Weight {
    fn eq(&self, other: &Self) -> bool {
        self.num_tsids == other.num_tsids && self.entries == other.entries
    }
}

impl Eq for Weight {}

impl std::hash::Hash for Weight {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.num_tsids.hash(state);
        self.entries.len().hash(state);
        for (lo, hi, rs) in &self.entries {
            lo.hash(state);
            hi.hash(state);
            rs.ranges().hash(state);
        }
    }
}

impl std::fmt::Display for Weight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Weight(tsids={}, [", self.num_tsids)?;
        for (i, (lo, hi, rs)) in self.entries.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if lo == hi {
                write!(f, "{lo}→{rs}")?;
            } else {
                write!(f, "{lo}..={hi}→{rs}")?;
            }
        }
        write!(f, "])")
    }
}

// ---- Helpers ----

/// Sort and coalesce a `Vec<(u32, u32)>` of inclusive ranges **in-place**,
/// assuming the input is already sorted.
fn coalesce_ranges(ranges: &mut Vec<(u32, u32)>) {
    if ranges.len() <= 1 {
        return;
    }
    let mut write = 0;
    for read in 1..ranges.len() {
        if ranges[read].0 <= ranges[write].1.saturating_add(1) {
            ranges[write].1 = ranges[write].1.max(ranges[read].1);
        } else {
            write += 1;
            ranges[write] = ranges[read];
        }
    }
    ranges.truncate(write + 1);
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- RangeMap tests --

    #[test]
    fn test_range_map_lookup() {
        let rm = RangeMap::from_sorted(vec![(0, 10, "a"), (10, 20, "b"), (30, 40, "c")]);
        assert_eq!(rm.get(0), Some(&"a"));
        assert_eq!(rm.get(9), Some(&"a"));
        assert_eq!(rm.get(10), Some(&"b"));
        assert_eq!(rm.get(25), None);
        assert_eq!(rm.get(35), Some(&"c"));
    }

    // -- WeightTable tests --

    #[test]
    fn test_weight_table() {
        let mut wt = WeightTable::new(3, 2);
        wt.set(0, 1, 2, 5);
        assert_eq!(wt.get(0, 1), (2, 5));
        assert_eq!(wt.get(1, 0), (u32::MAX, 0));
    }

    // -- Weight construction tests --

    #[test]
    fn test_weight_empty() {
        let w = Weight::empty(4);
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);
        assert_eq!(w.num_tsids(), 4);
    }

    #[test]
    fn test_weight_from_position() {
        // 2 TSIDs.  Position 5 = token 2, tsid 1.
        let w = Weight::from_position(5, 2);
        assert!(w.contains(5));
        assert!(!w.contains(4));
        assert!(!w.contains(6));
        assert_eq!(w.len(), 1);
        assert_eq!(w.tokens_for_tsid(1), RangeSet::from_range(2, 2));
        assert!(w.tokens_for_tsid(0).is_empty());
    }

    #[test]
    fn test_weight_from_uniform_tsid_set() {
        let tsids = RangeSet::from_range(0, 1);
        let w = Weight::from_uniform_tsid_set(10, 20, &tsids, 3);
        assert_eq!(w.tokens_for_tsid(0), RangeSet::from_range(10, 20));
        assert_eq!(w.tokens_for_tsid(1), RangeSet::from_range(10, 20));
        assert!(w.tokens_for_tsid(2).is_empty());
    }

    #[test]
    fn test_weight_all_simple() {
        let w = Weight::all(9, 1);
        assert_eq!(w.len(), 10);
        for p in 0..=9 {
            assert!(w.contains(p));
        }
        assert!(!w.contains(10));
    }

    #[test]
    fn test_weight_all_multi_tsid() {
        // 3 TSIDs, max_position = 7.
        let w = Weight::all(7, 3);
        assert_eq!(w.len(), 8);
        for p in 0..=7 {
            assert!(w.contains(p), "should contain position {p}");
        }
        assert!(!w.contains(8));
    }

    #[test]
    fn test_weight_from_positions() {
        let positions = RangeSet::from_ranges(vec![(0, 1), (4, 5)]);
        let w = Weight::from_positions(&positions, 2);
        assert_eq!(w.len(), 4);
        assert!(w.contains(0));
        assert!(w.contains(1));
        assert!(!w.contains(2));
        assert!(!w.contains(3));
        assert!(w.contains(4));
        assert!(w.contains(5));
        assert_eq!(
            w.tokens_for_tsid(0),
            RangeSet::from_ranges(vec![(0, 0), (2, 2)])
        );
        assert_eq!(
            w.tokens_for_tsid(1),
            RangeSet::from_ranges(vec![(0, 0), (2, 2)])
        );
    }

    // -- Set operation tests --

    #[test]
    fn test_weight_union() {
        let a = Weight::from_position(0, 2);
        let b = Weight::from_position(3, 2);
        let u = a.union(&b);
        assert_eq!(u.len(), 2);
        assert!(u.contains(0));
        assert!(u.contains(3));
        assert!(!u.contains(1));
        assert!(!u.contains(2));
    }

    #[test]
    fn test_weight_union_overlapping() {
        let a = Weight::from_position(5, 2);
        let b = Weight::from_position(5, 2);
        let u = a.union(&b);
        assert_eq!(u.len(), 1);
        assert!(u.contains(5));
    }

    #[test]
    fn test_weight_intersection() {
        let nt = 2u32;
        let a = Weight::from_positions(&RangeSet::from_range(0, 3), nt);
        let b = Weight::from_positions(&RangeSet::from_range(2, 5), nt);
        let i = a.intersection(&b);
        assert_eq!(i.len(), 2);
        assert!(i.contains(2));
        assert!(i.contains(3));
        assert!(!i.contains(0));
        assert!(!i.contains(4));
    }

    #[test]
    fn test_weight_difference() {
        let nt = 2u32;
        let a = Weight::from_positions(&RangeSet::from_range(0, 5), nt);
        let b = Weight::from_positions(&RangeSet::from_range(2, 3), nt);
        let d = a.difference(&b);
        assert_eq!(d.len(), 4);
        assert!(d.contains(0));
        assert!(d.contains(1));
        assert!(!d.contains(2));
        assert!(!d.contains(3));
        assert!(d.contains(4));
        assert!(d.contains(5));
    }

    #[test]
    fn test_weight_complement() {
        let nt = 2u32;
        let w = Weight::from_positions(&RangeSet::from_range(2, 3), nt);
        let c = w.complement(5);
        assert_eq!(c.len(), 4);
        assert!(c.contains(0));
        assert!(c.contains(1));
        assert!(!c.contains(2));
        assert!(!c.contains(3));
        assert!(c.contains(4));
        assert!(c.contains(5));
    }

    #[test]
    fn test_weight_divide() {
        let nt = 1u32;
        let a = Weight::from_positions(&RangeSet::from_range(1, 2), nt);
        let b = Weight::from_positions(&RangeSet::from_range(3, 4), nt);
        let d = a.divide(&b, 5);
        assert_eq!(d.len(), 4);
        assert!(d.contains(0));
        assert!(d.contains(1));
        assert!(d.contains(2));
        assert!(!d.contains(3));
        assert!(!d.contains(4));
        assert!(d.contains(5));
    }

    #[test]
    fn test_weight_is_disjoint() {
        let nt = 2u32;
        let a = Weight::from_positions(&RangeSet::from_range(0, 1), nt);
        let b = Weight::from_positions(&RangeSet::from_range(2, 3), nt);
        assert!(a.is_disjoint(&b));

        let c = Weight::from_positions(&RangeSet::from_range(1, 2), nt);
        assert!(!a.is_disjoint(&c));
    }

    #[test]
    fn test_weight_is_subset() {
        let nt = 2u32;
        let small = Weight::from_positions(&RangeSet::from_range(2, 3), nt);
        let big = Weight::from_positions(&RangeSet::from_range(0, 5), nt);
        assert!(small.is_subset(&big));
        assert!(!big.is_subset(&small));
    }

    // -- Expansion tests --

    #[test]
    fn test_expand_to_positions_simple() {
        let w = Weight::from_position(5, 2);
        let positions = w.expand_to_positions();
        assert_eq!(positions, vec![(5, 5)]);
    }

    #[test]
    fn test_expand_to_positions_contiguous() {
        let w = Weight::from_positions(&RangeSet::from_range(0, 5), 2);
        let positions = w.expand_to_positions();
        assert_eq!(positions, vec![(0, 5)]);
    }

    #[test]
    fn test_expand_roundtrip() {
        let nt = 3u32;
        let original = RangeSet::from_ranges(vec![(0, 2), (5, 8), (12, 14)]);
        let w = Weight::from_positions(&original, nt);
        let positions = w.expand_to_positions();
        let expanded = RangeSet::from_ranges(positions);
        assert_eq!(expanded, original);
    }

    // -- Equality and display tests --

    #[test]
    fn test_weight_equality() {
        let a = Weight::from_position(5, 2);
        let b = Weight::from_position(5, 2);
        assert_eq!(a, b);
        let c = Weight::from_position(3, 2);
        assert_ne!(a, c);
    }

    #[test]
    fn test_weight_display() {
        let w = Weight::from_position(5, 2);
        let s = format!("{w}");
        assert!(s.contains("Weight"));
        assert!(s.contains("tsids=2"));
    }

    // -- Coalesce helper test --

    #[test]
    fn test_coalesce_ranges() {
        let mut r = vec![(1, 3), (2, 5), (7, 9), (8, 10)];
        r.sort_unstable();
        coalesce_ranges(&mut r);
        assert_eq!(r, vec![(1, 5), (7, 10)]);
    }
}
