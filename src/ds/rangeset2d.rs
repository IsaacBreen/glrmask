//! RangeMapBlaze-based weight backend.
//!
//! The **Weight** type stores a set of (token, TSID) positions using TSID-outer
//! layout: a `RangeMapBlaze<u32, RangeSetBlaze<u32>>` mapping TSID ranges to
//! token sets.  This is the sole weight representation used during DWA
//! determinization and minimization.
//!
//! The backing representation is the shape-only 2D token/TSID range-set used
//! throughout the weighted-u32 automata skeleton.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};
use serde::{Deserialize, Serialize};

/// A token-set ID.  Groups of tokens that behave identically through a
/// DWA state transition share the same TSID.
pub type Tsid = u32;

/// A set of `u32` token IDs, backed by `RangeSetBlaze<u32>`.
///
/// This type alias keeps `range_set_blaze` contained to this module;
/// consumers import `TokenSet` instead of depending on the upstream crate
/// directly.
pub type TokenSet = RangeSetBlaze<u32>;

// ---------------------------------------------------------------------------
// Weight — TSID-outer weight set for compilation
// ---------------------------------------------------------------------------

/// A weight set using TSID-outer layout.
///
/// Stores a `RangeMapBlaze<u32, RangeSetBlaze<u32>>` mapping TSID ranges to
/// token sets.  In TSID-outer layout the outer key is the TSID and the value
/// is a set of token IDs.  This enables O(log n) lookup of the token set for
/// a given TSID, which is the hot path during mask computation.
///
/// A "position" in the flat DWA weight space is
/// `token_id * num_tsids + tsid`.
///
/// The weight is represented as an enum with three variants:
/// - `Empty` — no positions (the zero element).
/// - `Full` — all positions (the universal element); lazy, carries no data.
/// - `Concrete` — explicit TSID-outer entries backed by `RangeMapBlaze`.
///
/// Dimension bounds (`num_tsids`, `max_token`) are **not** stored in the
/// weight.  Operations that need them (`complement`, `divide`,
/// `expand_to_positions`, etc.) accept them as explicit parameters.
#[derive(Debug, Clone)]
pub enum Weight {
    /// No positions.
    Empty,
    /// All positions (universal weight).  Carries no concrete data; the
    /// actual extent is determined by the automaton context (`num_tsids`,
    /// `max_token`).
    Full,
    /// Concrete TSID-outer entries: `RangeMapBlaze<u32, RangeSetBlaze<u32>>`
    /// where key = TSID, value = token set.
    Concrete(RangeMapBlaze<u32, RangeSetBlaze<u32>>),
}

impl Weight {
    // ---- Construction ----

    /// Create an empty weight (no positions).
    pub fn empty() -> Self {
        unimplemented!()
    }

    /// Create a full / universal weight (all positions).
    ///
    /// This is lazy: no concrete entries are stored.  Use
    /// [`materialize_full`](Self::materialize_full) when concrete entries
    /// are needed (e.g. for `complement` or `divide_complement`).
    pub fn full() -> Self {
        unimplemented!()
    }

    /// Construct from raw sorted entries (TSID-outer layout).
    ///
    /// Entries must be non-overlapping and sorted by TSID range.
    /// Empty token sets are silently filtered out.
    pub fn from_entries(entries: Vec<(u32, u32, RangeSetBlaze<u32>)>) -> Self {
        unimplemented!()
    }

    /// Construct a `Concrete` weight directly from a `RangeMapBlaze`.
    pub fn from_map(map: RangeMapBlaze<u32, RangeSetBlaze<u32>>) -> Self {
        unimplemented!()
    }

    /// Create a weight containing a single flat position.
    ///
    /// Position `p` decodes as `token = p / num_tsids`, `tsid = p % num_tsids`.
    pub fn from_position(pos: u32, num_tsids: u32) -> Self {
        unimplemented!()
    }

    /// Create a weight where every TSID in `tsid_set` maps to the same
    /// token range `[token_start, token_end]`.
    pub fn from_uniform_tsid_set(
        token_start: u32,
        token_end: u32,
        tsid_set: &RangeSetBlaze<u32>,
    ) -> Self {
        unimplemented!()
    }

    /// Create a weight covering all positions from 0 to `max_position`
    /// (inclusive), materialized as concrete `Concrete` entries.
    ///
    /// Use [`full()`](Self::full) for the lazy variant that carries no data.
    pub fn materialize_full(max_position: u32, num_tsids: u32) -> Self {
        unimplemented!()
    }

    /// Backward-compatible alias: create a full weight with concrete entries.
    ///
    /// **Prefer [`full()`](Self::full) or [`materialize_full()`](Self::materialize_full).**
    #[deprecated(note = "use Weight::full() for lazy or Weight::materialize_full() for concrete")]
    pub fn all(max_position: u32, num_tsids: u32) -> Self {
        unimplemented!()
    }

    /// Construct from flat position ranges (position = token × num_tsids + tsid).
    ///
    /// The input is a `RangeSetBlaze<u32>` of flat positions.
    pub fn from_positions(positions: &RangeSetBlaze<u32>, num_tsids: u32) -> Self {
        unimplemented!()
    }

    // ---- Queries ----

    /// Whether this is the universal (full) weight.
    pub fn is_full(&self) -> bool {
        unimplemented!()
    }

    /// Whether the weight is empty (no positions).
    pub fn is_empty(&self) -> bool {
        unimplemented!()
    }

    /// Access the concrete map, or `None` for `Empty`/`Full`.
    pub fn as_map(&self) -> Option<&RangeMapBlaze<u32, RangeSetBlaze<u32>>> {
        unimplemented!()
    }

    /// Collect entries as `Vec<(tsid_lo, tsid_hi, token_set)>`.
    ///
    /// For `Empty`/`Full`, returns an empty Vec.
    pub fn collect_entries(&self) -> Vec<(u32, u32, RangeSetBlaze<u32>)> {
        unimplemented!()
    }

    /// Number of outer range entries.
    pub fn num_entries(&self) -> usize {
        unimplemented!()
    }

    /// Total number of sub-ranges (outer + sum of inner).
    pub fn num_ranges(&self) -> usize {
        unimplemented!()
    }

    /// Count the total number of positions in this weight.
    ///
    /// For `Full`, this returns 0 because the actual count depends on
    /// the automaton context.  Use `materialize_full` first if you need
    /// the concrete count.
    pub fn len(&self) -> u64 {
        unimplemented!()
    }

    /// Look up the token set for a specific TSID.
    ///
    /// For `Full`, returns an empty set (materialize first for concrete
    /// results).
    pub fn tokens_for_tsid(&self, tsid: u32) -> RangeSetBlaze<u32> {
        unimplemented!()
    }

    /// Check if a flat position is contained.
    ///
    /// Position `p` decodes as `token = p / num_tsids`, `tsid = p % num_tsids`.
    ///
    /// For `Full`, always returns `true`.
    pub fn contains(&self, pos: u32, num_tsids: u32) -> bool {
        unimplemented!()
    }

    /// Iterate over entries as `(tsid_lo, tsid_hi, &token_set)`.
    ///
    /// For `Full`, yields nothing (materialize first).
    pub fn iter_entries(&self) -> Box<dyn Iterator<Item = (u32, u32, &RangeSetBlaze<u32>)> + '_> {
        unimplemented!()
    }

    /// Expand to a sorted list of non-overlapping inclusive flat-position
    /// ranges `(lo, hi)` where `position = token * num_tsids + tsid`.
    ///
    /// Panics if called on `Full` (materialize first).
    pub fn expand_to_positions(&self, num_tsids: u32) -> Vec<(u32, u32)> {
        unimplemented!()
    }

    // ---- Set operations ----

    /// Compute the union of two weights.
    pub fn union(&self, other: &Self) -> Self {
        unimplemented!()
    }

    /// Compute the intersection of two weights.
    pub fn intersection(&self, other: &Self) -> Self {
        unimplemented!()
    }

    /// Compute the set difference `self − other`.
    ///
    /// Panics if `self` is `Full` and `other` is `Concrete` (use
    /// [`complement`](Self::complement) with explicit bounds instead).
    pub fn difference(&self, other: &Self) -> Self {
        unimplemented!()
    }

    /// Compute the complement within `[0, max_position]`.
    pub fn complement(&self, max_position: u32, num_tsids: u32) -> Self {
        unimplemented!()
    }

    /// Compute `self | !other` (divide).
    ///
    /// For each TSID, the result token set is
    /// `self_tokens | (full_tokens − other_tokens)` where `full_tokens` is
    /// `[0, max_token]`.  Requires explicit `max_token` and `num_tsids`
    /// to define the complement universe.
    pub fn divide(&self, other: &Self, max_token: u32, num_tsids: u32) -> Self {
        unimplemented!()
    }

    /// Compute the "divide complement" of `self` within `[0, max_token]`.
    ///
    /// Returns a weight covering ALL TSIDs (0..num_tsids-1) where:
    /// - TSIDs with entries get `full.difference(entry_value)`
    /// - TSIDs without entries get `full` (the full token range)
    ///
    /// This can be precomputed once for a divisor and reused across
    /// multiple `divide_with_complement` calls.
    pub fn divide_complement(&self, max_token: u32, num_tsids: u32) -> Self {
        unimplemented!()
    }

    /// Divide using a precomputed complement: `self | complement`.
    ///
    /// `complement` must be the result of `divisor.divide_complement(max_token, num_tsids)`.
    /// This computes `self | complement` which equals `self.divide(divisor, max_token, num_tsids)`.
    ///
    /// Specialized implementation that exploits the complement's structure:
    /// most complement entries are `full` (where `x | full = full`), so we
    /// only compute actual unions for the few non-`full` entries.
    pub fn divide_with_complement(&self, complement: &Self, full: &RangeSetBlaze<u32>) -> Self {
        unimplemented!()
    }

    /// Check whether two weights are disjoint.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        unimplemented!()
    }

    /// Check whether `self ⊆ other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        unimplemented!()
    }

    // ---- Internal ----

    /// Merge two `RangeMapBlaze`s by sweeping over the TSID key space.
    ///
    /// Follows the sep1 `merge_maps` pattern: collects all boundary
    /// points from both maps, partitions the key space into uniform
    /// sub-intervals, calls the `combine` closure for each, and
    /// builds the result RangeMapBlaze with coalescing.
        fn merge_maps<F>(
        a: &RangeMapBlaze<u32, RangeSetBlaze<u32>>,
        b: &RangeMapBlaze<u32, RangeSetBlaze<u32>>,
        combine: F,
    ) -> RangeMapBlaze<u32, RangeSetBlaze<u32>>
    where
        F: Fn(Option<&RangeSetBlaze<u32>>, Option<&RangeSetBlaze<u32>>) -> RangeSetBlaze<u32>,
    {
            unimplemented!()
        }
}

// ---- Trait impls ----

impl PartialEq for Weight {
        fn eq(&self, other: &Self) -> bool {
            unimplemented!()
        }
}

impl Eq for Weight {}

impl std::hash::Hash for Weight {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            unimplemented!()
        }
}

impl std::fmt::Display for Weight {
    /// Compact structural display: `{tsid_range: token_set, ...}`
    ///
    /// Examples:
    /// - `{0: {0, 3, 5}, 1..=3: {1..=5, 7, 9..=11}}`
    /// - `∅` (empty weight)
    /// - `ALL` (full weight)
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}

/// Maximum number of entries before falling back to compact display in
/// the symbol-aware weight formatter.
const WEIGHT_SYMBOL_EXPAND_LIMIT: usize = 64;

/// Wrapper to display a [`Weight`] with human-readable names for both
/// the TSID dimension and the token dimension.
///
/// If either dimension exceeds [`WEIGHT_SYMBOL_EXPAND_LIMIT`], falls back
/// to the compact/default representation.
pub struct WeightDisplayWithMaps<'a> {
    weight: &'a Weight,
    /// TSID → name (e.g. "root", "state3").
    tsid_names: &'a std::collections::BTreeMap<u32, String>,
    /// token_id → name (e.g. `"a"`, `"$"`).
    token_names: &'a std::collections::BTreeMap<u32, String>,
}

impl<'a> Weight {
    /// Return a wrapper that prints this weight using human-readable names
    /// for TSIDs and tokens.
    pub fn display_with_maps(
        &'a self,
        tsid_names: &'a std::collections::BTreeMap<u32, String>,
        token_names: &'a std::collections::BTreeMap<u32, String>,
    ) -> WeightDisplayWithMaps<'a> {
        unimplemented!()
    }
}

impl std::fmt::Display for WeightDisplayWithMaps<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}

// ---- Serde ----

/// Serde proxy for Weight (since RangeMapBlaze/RangeSetBlaze don't impl
/// Serialize/Deserialize).
#[derive(Serialize, Deserialize)]
enum WeightSerde {
    Empty,
    Full,
    /// Entries as `Vec<(tsid_lo, tsid_hi, token_ranges)>` where
    /// `token_ranges` is a flat `[lo0, hi0, lo1, hi1, ...]` array.
    Concrete(Vec<(u32, u32, Vec<u32>)>),
}

impl Serialize for Weight {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        unimplemented!()
    }
}

impl<'de> Deserialize<'de> for Weight {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        unimplemented!()
    }
}

// ---- Helpers ----

/// Sort and coalesce a `Vec<(u32, u32)>` of inclusive ranges **in-place**,
/// assuming the input is already sorted.
fn coalesce_ranges(ranges: &mut Vec<(u32, u32)>) {
    unimplemented!()
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a `RangeSetBlaze<u32>` from a single inclusive range.
    fn rsb(lo: u32, hi: u32) -> RangeSetBlaze<u32> {
        RangeSetBlaze::from_iter([lo..=hi])
    }

    /// Helper to build from multiple inclusive ranges.
    fn rsb_multi(ranges: &[(u32, u32)]) -> RangeSetBlaze<u32> {
        ranges.iter().map(|&(lo, hi)| lo..=hi).collect()
    }

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
        let w = Weight::empty();
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);
    }

    #[test]
    fn test_weight_from_position() {
        // 2 TSIDs.  Position 5 = token 2, tsid 1.
        let w = Weight::from_position(5, 2);
        assert!(w.contains(5, 2));
        assert!(!w.contains(4, 2));
        assert!(!w.contains(6, 2));
        assert_eq!(w.len(), 1);
        assert_eq!(w.tokens_for_tsid(1), rsb(2, 2));
        assert!(w.tokens_for_tsid(0).is_empty());
    }

    #[test]
    fn test_weight_from_uniform_tsid_set() {
        let tsids = rsb(0, 1);
        let w = Weight::from_uniform_tsid_set(10, 20, &tsids);
        assert_eq!(w.tokens_for_tsid(0), rsb(10, 20));
        assert_eq!(w.tokens_for_tsid(1), rsb(10, 20));
        assert!(w.tokens_for_tsid(2).is_empty());
    }

    #[test]
    fn test_weight_materialize_full_simple() {
        let w = Weight::materialize_full(9, 1);
        assert_eq!(w.len(), 10);
        for p in 0..=9 {
            assert!(w.contains(p, 1));
        }
        assert!(!w.contains(10, 1));
    }

    #[test]
    fn test_weight_materialize_full_multi_tsid() {
        // 3 TSIDs, max_position = 7.
        let w = Weight::materialize_full(7, 3);
        assert_eq!(w.len(), 8);
        for p in 0..=7 {
            assert!(w.contains(p, 3), "should contain position {p}");
        }
        assert!(!w.contains(8, 3));
    }

    #[test]
    fn test_weight_full_is_full() {
        let w = Weight::full();
        assert!(w.is_full());
        assert!(!w.is_empty());
        // Full contains everything
        assert!(w.contains(0, 1));
        assert!(w.contains(999, 2));
    }

    #[test]
    fn test_weight_from_positions() {
        let positions = rsb_multi(&[(0, 1), (4, 5)]);
        let w = Weight::from_positions(&positions, 2);
        assert_eq!(w.len(), 4);
        assert!(w.contains(0, 2));
        assert!(w.contains(1, 2));
        assert!(!w.contains(2, 2));
        assert!(!w.contains(3, 2));
        assert!(w.contains(4, 2));
        assert!(w.contains(5, 2));
        assert_eq!(
            w.tokens_for_tsid(0),
            rsb_multi(&[(0, 0), (2, 2)])
        );
        assert_eq!(
            w.tokens_for_tsid(1),
            rsb_multi(&[(0, 0), (2, 2)])
        );
    }

    // -- Set operation tests --

    #[test]
    fn test_weight_union() {
        let a = Weight::from_position(0, 2);
        let b = Weight::from_position(3, 2);
        let u = a.union(&b);
        assert_eq!(u.len(), 2);
        assert!(u.contains(0, 2));
        assert!(u.contains(3, 2));
        assert!(!u.contains(1, 2));
        assert!(!u.contains(2, 2));
    }

    #[test]
    fn test_weight_union_overlapping() {
        let a = Weight::from_position(5, 2);
        let b = Weight::from_position(5, 2);
        let u = a.union(&b);
        assert_eq!(u.len(), 1);
        assert!(u.contains(5, 2));
    }

    #[test]
    fn test_weight_intersection() {
        let nt = 2u32;
        let a = Weight::from_positions(&rsb(0, 3), nt);
        let b = Weight::from_positions(&rsb(2, 5), nt);
        let i = a.intersection(&b);
        assert_eq!(i.len(), 2);
        assert!(i.contains(2, nt));
        assert!(i.contains(3, nt));
        assert!(!i.contains(0, nt));
        assert!(!i.contains(4, nt));
    }

    #[test]
    fn test_weight_difference() {
        let nt = 2u32;
        let a = Weight::from_positions(&rsb(0, 5), nt);
        let b = Weight::from_positions(&rsb(2, 3), nt);
        let d = a.difference(&b);
        assert_eq!(d.len(), 4);
        assert!(d.contains(0, nt));
        assert!(d.contains(1, nt));
        assert!(!d.contains(2, nt));
        assert!(!d.contains(3, nt));
        assert!(d.contains(4, nt));
        assert!(d.contains(5, nt));
    }

    #[test]
    fn test_weight_complement() {
        let nt = 2u32;
        let w = Weight::from_positions(&rsb(2, 3), nt);
        let c = w.complement(5, nt);
        assert_eq!(c.len(), 4);
        assert!(c.contains(0, nt));
        assert!(c.contains(1, nt));
        assert!(!c.contains(2, nt));
        assert!(!c.contains(3, nt));
        assert!(c.contains(4, nt));
        assert!(c.contains(5, nt));
    }

    #[test]
    fn test_weight_divide() {
        let nt = 1u32;
        let a = Weight::from_positions(&rsb(1, 2), nt);
        let b = Weight::from_positions(&rsb(3, 4), nt);
        let d = a.divide(&b, 5, nt);
        assert_eq!(d.len(), 4);
        assert!(d.contains(0, nt));
        assert!(d.contains(1, nt));
        assert!(d.contains(2, nt));
        assert!(!d.contains(3, nt));
        assert!(!d.contains(4, nt));
        assert!(d.contains(5, nt));
    }

    #[test]
    fn test_weight_is_disjoint() {
        let nt = 2u32;
        let a = Weight::from_positions(&rsb(0, 1), nt);
        let b = Weight::from_positions(&rsb(2, 3), nt);
        assert!(a.is_disjoint(&b));

        let c = Weight::from_positions(&rsb(1, 2), nt);
        assert!(!a.is_disjoint(&c));
    }

    #[test]
    fn test_weight_is_subset() {
        let nt = 2u32;
        let small = Weight::from_positions(&rsb(2, 3), nt);
        let big = Weight::from_positions(&rsb(0, 5), nt);
        assert!(small.is_subset(&big));
        assert!(!big.is_subset(&small));
    }

    // -- Expansion tests --

    #[test]
    fn test_expand_to_positions_simple() {
        let w = Weight::from_position(5, 2);
        let positions = w.expand_to_positions(2);
        assert_eq!(positions, vec![(5, 5)]);
    }

    #[test]
    fn test_expand_to_positions_contiguous() {
        let w = Weight::from_positions(&rsb(0, 5), 2);
        let positions = w.expand_to_positions(2);
        assert_eq!(positions, vec![(0, 5)]);
    }

    #[test]
    fn test_expand_roundtrip() {
        let nt = 3u32;
        let original = rsb_multi(&[(0, 2), (5, 8), (12, 14)]);
        let w = Weight::from_positions(&original, nt);
        let positions = w.expand_to_positions(nt);
        let expanded: RangeSetBlaze<u32> = positions.into_iter().map(|(lo, hi)| lo..=hi).collect();
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
        // New compact format: {tsid: {token_ranges}}
        assert!(s.contains("{"));
        assert!(s.contains("}"));

        // Empty weight
        let empty = Weight::empty();
        assert_eq!(format!("{empty}"), "∅");

        // Full weight
        let full = Weight::full();
        assert_eq!(format!("{full}"), "ALL");
    }

    // -- Serde roundtrip tests --

    #[test]
    fn test_weight_serde_empty() {
        let w = Weight::empty();
        let json = serde_json::to_string(&w).unwrap();
        let w2: Weight = serde_json::from_str(&json).unwrap();
        assert_eq!(w, w2);
    }

    #[test]
    fn test_weight_serde_full() {
        let w = Weight::full();
        let json = serde_json::to_string(&w).unwrap();
        let w2: Weight = serde_json::from_str(&json).unwrap();
        assert_eq!(w, w2);
    }

    #[test]
    fn test_weight_serde_concrete() {
        let w = Weight::from_entries(vec![
            (0, 2, rsb(10, 20)),
            (5, 5, rsb_multi(&[(1, 3), (7, 9)])),
        ]);
        let json = serde_json::to_string(&w).unwrap();
        let w2: Weight = serde_json::from_str(&json).unwrap();
        assert_eq!(w, w2);
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


/// Compatibility alias for the relocated 2D token/TSID range-set representation.
pub type RangeSet2D = Weight;

// ------------------------------------------------------------------
// Serde helpers relocated from the deleted range_set_serde.rs
// ------------------------------------------------------------------

/// `#[serde(with = "crate::ds::rangeset2d::bare")]`
#[allow(dead_code)]
pub mod bare {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(rs: &TokenSet, s: S) -> Result<S::Ok, S::Error> {
        unimplemented!()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<TokenSet, D::Error> {
        unimplemented!()
    }
}

/// `#[serde(with = "crate::ds::rangeset2d::vec_rsb")]`
pub mod vec_rsb {
    use super::*;
    use serde::ser::SerializeSeq;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &[TokenSet], s: S) -> Result<S::Ok, S::Error> {
        unimplemented!()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<TokenSet>, D::Error> {
        unimplemented!()
    }

    pub(super) fn rsb_from_flat(flat: Vec<u32>) -> TokenSet {
        unimplemented!()
    }
}

/// `#[serde(with = "crate::ds::rangeset2d::vec_btmap_rsb")]`
pub mod vec_btmap_rsb {
    use super::vec_rsb::rsb_from_flat;
    use super::TokenSet;
    use serde::ser::SerializeSeq;
    use serde::{Deserialize, Deserializer, Serializer};
    use std::collections::BTreeMap;

    pub fn serialize<S: Serializer>(
        v: &[BTreeMap<u32, TokenSet>],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        unimplemented!()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Vec<BTreeMap<u32, TokenSet>>, D::Error> {
        unimplemented!()
    }
}
