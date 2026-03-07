//! RangeMapBlaze-based weight backend.
//!
//! The **Weight** type stores a set of (token, TSID) positions using TSID-outer
//! layout: a `RangeMapBlaze<u32, RangeSetBlaze<u32>>` mapping TSID ranges to
//! token sets.  This is the sole weight representation used during DWA
//! determinization and minimization.
//!
//! The **WeightTable** is the flat, cache-friendly layout used at inference
//! time.  It stores `(target_state, weight)` pairs indexed by `(tsid, state)`.
//!
//! The **RangeMap** is a generic sorted-interval-to-value map used for
//! vocabulary preprocessing (token → TSID mapping).
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
        unimplemented!()
    }

    /// Create from pre-sorted entries.
    pub fn from_sorted(entries: Vec<(u32, u32, V)>) -> Self {
        unimplemented!()
    }

    /// Number of range entries.
    pub fn len(&self) -> usize {
        unimplemented!()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        unimplemented!()
    }

    /// Look up the value for a given key using binary search.
    pub fn get(&self, key: u32) -> Option<&V> {
        unimplemented!()
    }

    /// Iterate over all entries as `(start, end, &value)`.
    pub fn iter(&self) -> impl Iterator<Item = (u32, u32, &V)> {
        std::iter::empty()
    }

    /// Access entries as a slice.
    pub fn entries(&self) -> &[(u32, u32, V)] {
        unimplemented!()
    }
}

impl<V: Clone + Eq> Default for RangeMap<V> {
    fn default() -> Self {
        unimplemented!()
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
        unimplemented!()
    }

    /// Get the transition for `(tsid, state)`.
    #[inline]
    pub fn get(&self, tsid: u32, state: u32) -> (u32, i32) {
        unimplemented!()
    }

    /// Set the transition for `(tsid, state)`.
    #[inline]
    pub fn set(&mut self, tsid: u32, state: u32, target: u32, weight: i32) {
        unimplemented!()
    }
}

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
        match self {
            Weight::Concrete(m) => {
                Box::new(m.range_values().map(|(r, v)| (*r.start(), *r.end(), v)))
            }
            _ => Box::new(std::iter::empty()),
        }
    }

    /// Expand to a sorted list of non-overlapping inclusive flat-position
    /// ranges `(lo, hi)` where `position = token * num_tsids + tsid`.
    ///
    /// Panics if called on `Full` (materialize first).
    pub fn expand_to_positions(&self, num_tsids: u32) -> Vec<(u32, u32)> {
        let map = match self {
            Weight::Empty => return Vec::new(),
            Weight::Full => panic!("expand_to_positions called on Full weight; materialize first"),
            Weight::Concrete(m) => m,
        };
        let nt = num_tsids.max(1);
        let mut ranges = Vec::new();

        for (tsid_range, token_set) in map.range_values() {
            let tsid_lo = *tsid_range.start();
            let tsid_hi = *tsid_range.end();
            let tsid_span = tsid_hi - tsid_lo + 1;
            for tok_range in token_set.ranges() {
                let t_lo = *tok_range.start();
                let t_hi = *tok_range.end();
                if nt <= 1 || tsid_span == nt {
                    // Full TSID coverage ⇒ contiguous positions.
                    let pos_lo = t_lo.saturating_mul(nt).saturating_add(tsid_lo);
                    let pos_hi = t_hi.saturating_mul(nt).saturating_add(tsid_hi);
                    ranges.push((pos_lo, pos_hi));
                } else {
                    // Partial TSID range ⇒ per-token blocks.
                    for token in t_lo..=t_hi {
                        let base = token.saturating_mul(nt);
                        ranges.push((base.saturating_add(tsid_lo), base.saturating_add(tsid_hi)));
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
        match (self, other) {
            (Weight::Full, _) | (_, Weight::Full) => Weight::Full,
            (Weight::Empty, _) => other.clone(),
            (_, Weight::Empty) => self.clone(),
            (Weight::Concrete(a), Weight::Concrete(b)) => {
                Self::from_map(Self::merge_maps(a, b, |a, b| match (a, b) {
                    (Some(a), Some(b)) => a | b,
                    (Some(x), None) | (None, Some(x)) => x.clone(),
                    (None, None) => RangeSetBlaze::new(),
                }))
            }
        }
    }

    /// Compute the intersection of two weights.
    pub fn intersection(&self, other: &Self) -> Self {
        match (self, other) {
            (Weight::Empty, _) | (_, Weight::Empty) => Weight::Empty,
            (Weight::Full, _) => other.clone(),
            (_, Weight::Full) => self.clone(),
            (Weight::Concrete(a), Weight::Concrete(b)) => {
                // Specialized two-pointer sweep for intersection (only
                // overlapping TSID ranges can contribute).
                let mut result = RangeMapBlaze::new();
                let mut a_iter = a.range_values().peekable();
                let mut b_iter = b.range_values().peekable();

                // Track pending coalesce state.
                let mut cur_start: Option<u32> = None;
                let mut cur_end: u32 = 0;
                let mut cur_val = RangeSetBlaze::<u32>::new();

                while let (Some(&(ref a_range, ref a_rs)), Some(&(ref b_range, ref b_rs))) =
                    (a_iter.peek(), b_iter.peek())
                {
                    let a_lo = *a_range.start();
                    let a_hi = *a_range.end();
                    let b_lo = *b_range.start();
                    let b_hi = *b_range.end();

                    if a_hi < b_lo {
                        a_iter.next();
                        continue;
                    }
                    if b_hi < a_lo {
                        b_iter.next();
                        continue;
                    }

                    // Overlapping TSID interval.
                    let lo = a_lo.max(b_lo);
                    let hi = a_hi.min(b_hi);
                    let rs: RangeSetBlaze<u32> = (*a_rs) & (*b_rs);
                    if !rs.is_empty() {
                        if let Some(start) = cur_start {
                            if cur_val == rs && cur_end + 1 == lo {
                                cur_end = hi;
                            } else {
                                result.ranges_insert(start..=cur_end, cur_val.clone());
                                cur_start = Some(lo);
                                cur_end = hi;
                                cur_val = rs;
                            }
                        } else {
                            cur_start = Some(lo);
                            cur_end = hi;
                            cur_val = rs;
                        }
                    } else if let Some(start) = cur_start.take() {
                        result.ranges_insert(start..=cur_end, cur_val.clone());
                    }

                    if a_hi <= b_hi {
                        a_iter.next();
                    } else {
                        b_iter.next();
                    }
                }
                if let Some(start) = cur_start {
                    result.ranges_insert(start..=cur_end, cur_val);
                }

                Self::from_map(result)
            }
        }
    }

    /// Compute the set difference `self − other`.
    ///
    /// Panics if `self` is `Full` and `other` is `Concrete` (use
    /// [`complement`](Self::complement) with explicit bounds instead).
    pub fn difference(&self, other: &Self) -> Self {
        match (self, other) {
            (Weight::Empty, _) | (_, Weight::Full) => Weight::Empty,
            (_, Weight::Empty) => self.clone(),
            (Weight::Full, Weight::Concrete(_)) => {
                panic!("difference(Full, Concrete) requires explicit bounds — use complement() instead")
            }
            (Weight::Concrete(a), Weight::Concrete(b)) => {
                Self::from_map(Self::merge_maps(a, b, |a, b| match (a, b) {
                    (Some(a), Some(b)) => a - b,
                    (Some(a), None) => a.clone(),
                    _ => RangeSetBlaze::new(),
                }))
            }
        }
    }

    /// Compute the complement within `[0, max_position]`.
    pub fn complement(&self, max_position: u32, num_tsids: u32) -> Self {
        match self {
            Weight::Full => Weight::Empty,
            Weight::Empty => Weight::materialize_full(max_position, num_tsids),
            Weight::Concrete(_) => {
                Weight::materialize_full(max_position, num_tsids).difference(self)
            }
        }
    }

    /// Compute `self | !other` (divide).
    ///
    /// For each TSID, the result token set is
    /// `self_tokens | (full_tokens − other_tokens)` where `full_tokens` is
    /// `[0, max_token]`.  Requires explicit `max_token` and `num_tsids`
    /// to define the complement universe.
    pub fn divide(&self, other: &Self, max_token: u32, num_tsids: u32) -> Self {
        // Fast path: divide(full, any) = full
        if self.is_full() {
            return Weight::Full;
        }
        // Fast path: divide(w, full) = w  (a | (full - full) = a | ∅ = a)
        if other.is_full() {
            return self.clone();
        }
        let full = RangeSetBlaze::from_iter([0..=max_token]);
        let comp = other.divide_complement(max_token, num_tsids);
        self.divide_with_complement(&comp, &full)
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
        if self.is_full() {
            return Weight::Empty;
        }
        let full = RangeSetBlaze::from_iter([0..=max_token]);
        let nt = num_tsids.max(1);

        let map = match self {
            Weight::Empty => {
                // Complement of empty across all TSIDs = full for every TSID.
                return Weight::Concrete(RangeMapBlaze::from_iter([(
                    0..=(nt - 1),
                    full,
                )]));
            }
            Weight::Full => unreachable!(),
            Weight::Concrete(m) => m,
        };

        let mut result = RangeMapBlaze::new();
        let mut pos = 0u32;

        for (tsid_range, rs) in map.range_values() {
            let lo = *tsid_range.start();
            let hi = *tsid_range.end();

            // Gap before this entry: fill with `full`
            if pos < lo {
                result.ranges_insert(pos..=(lo - 1), full.clone());
            }
            // This entry: complement
            let comp = &full - rs;
            if !comp.is_empty() {
                result.ranges_insert(lo..=hi, comp);
            }
            pos = hi + 1;
        }
        // Trailing gap: fill with `full`
        if pos < nt {
            result.ranges_insert(pos..=(nt - 1), full);
        }

        Self::from_map(result)
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
        match self {
            Weight::Empty => write!(f, "∅"),
            Weight::Full => write!(f, "ALL"),
            Weight::Concrete(m) => {
                write!(f, "{{")?;
                for (i, (range, rs)) in m.range_values().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    let lo = range.start();
                    let hi = range.end();
                    if lo == hi {
                        write!(f, "{lo}: ")?;
                    } else {
                        write!(f, "{lo}..={hi}: ")?;
                    }
                    // Display the token set
                    write!(f, "{{")?;
                    for (j, tok_range) in rs.ranges().enumerate() {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        let tlo = tok_range.start();
                        let thi = tok_range.end();
                        if tlo == thi {
                            write!(f, "{tlo}")?;
                        } else {
                            write!(f, "{tlo}..={thi}")?;
                        }
                    }
                    write!(f, "}}")?;
                }
                write!(f, "}}")
            }
        }
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
        WeightDisplayWithMaps {
            weight: self,
            tsid_names,
            token_names,
        }
    }
}

impl std::fmt::Display for WeightDisplayWithMaps<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let w = self.weight;
        if w.is_empty() {
            return write!(f, "∅");
        }
        if w.is_full() {
            return write!(f, "ALL");
        }

        let map = match w {
            Weight::Concrete(m) => m,
            _ => return write!(f, "{w}"),
        };

        // Size guard: if too many entries, fall back to compact form.
        let total_token_ranges: usize = map.range_values().map(|(_, rs)| rs.ranges_len()).sum();
        if map.range_values_len() + total_token_ranges > WEIGHT_SYMBOL_EXPAND_LIMIT {
            return write!(f, "{w}");
        }

        write!(f, "{{")?;
        for (i, (range, rs)) in map.range_values().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            let lo = *range.start();
            let hi = *range.end();
            // TSID part
            if lo == hi {
                match self.tsid_names.get(&lo) {
                    Some(name) => write!(f, "{name}")?,
                    None => write!(f, "tsid{lo}")?,
                }
            } else {
                write!(f, "tsid{lo}..={hi}")?;
            }
            write!(f, ": [")?;
            // Token part — expand individual values when small
            let mut first = true;
            for tok_range in rs.ranges() {
                if !first {
                    write!(f, ", ")?;
                }
                first = false;
                let tlo = *tok_range.start();
                let thi = *tok_range.end();
                if tlo == thi {
                    match self.token_names.get(&tlo) {
                        Some(name) => write!(f, "{name}")?,
                        None => write!(f, "tok{tlo}")?,
                    }
                } else {
                    write!(f, "tok{tlo}..={thi}")?;
                }
            }
            write!(f, "]")?;
        }
        write!(f, "}}")
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
        let proxy = match self {
            Weight::Empty => WeightSerde::Empty,
            Weight::Full => WeightSerde::Full,
            Weight::Concrete(m) => {
                let entries: Vec<(u32, u32, Vec<u32>)> = m
                    .range_values()
                    .map(|(range, rs)| {
                        let flat: Vec<u32> = rs
                            .ranges()
                            .flat_map(|r| [*r.start(), *r.end()])
                            .collect();
                        (*range.start(), *range.end(), flat)
                    })
                    .collect();
                WeightSerde::Concrete(entries)
            }
        };
        proxy.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Weight {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let proxy = WeightSerde::deserialize(deserializer)?;
        match proxy {
            WeightSerde::Empty => Ok(Weight::Empty),
            WeightSerde::Full => Ok(Weight::Full),
            WeightSerde::Concrete(entries) => {
                let mut map = RangeMapBlaze::new();
                for (lo, hi, flat) in entries {
                    let rs: RangeSetBlaze<u32> = flat
                        .chunks(2)
                        .filter_map(|c| {
                            if c.len() == 2 {
                                Some(c[0]..=c[1])
                            } else {
                                None
                            }
                        })
                        .collect();
                    if !rs.is_empty() {
                        map.ranges_insert(lo..=hi, rs);
                    }
                }
                Ok(Weight::from_map(map))
            }
        }
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
