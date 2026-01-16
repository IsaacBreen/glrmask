//! Abstract weight type for DWA/NWA operations.
//! 
//! This module provides a unified weight type that can be used across
//! different weight representations. Currently only supports RangeSetBlaze,
//! but is designed to allow future extensions to other backends.
//!
//! # Adding a new backend
//! 1. Implement `WeightBackend` for your backend type.
//! 2. Add a new variant to `AbstractWeight` to wrap the backend.
//! 3. Update `AbstractWeight` constructors, accessors, and bitwise ops to
//!    dispatch to the new backend (mirroring the `RangeSet` variant).
//! 4. Add backend-specific tests that exercise the `WeightBackend` trait
//!    methods (ranges, set ops, complement, min/max, clip).
//!
//! # Backend selection
//! Use the `ABSTRACT_WEIGHT_BACKEND` environment variable to choose between
//! `rangeset` (default) and `factorized` backends.

use range_set_blaze::RangeSetBlaze;
use std::collections::BTreeMap;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};

/// Dimensions for weight-heavy mode encoding.
/// 
/// In weight-heavy mode, weights are encoded in an N×M space where:
/// - N = number of LLM tokens (vocab_size)
/// - M = number of tokenizer states (num_tsids)
/// 
/// A position p in this space represents:
/// - Token ID: p / num_tsids
/// - Tokenizer state ID (tsid): p % num_tsids
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WeightDimensions {
    /// Number of LLM tokens (vocab size). 0 for symbol-heavy mode.
    pub num_tokens: usize,
    /// Number of tokenizer states. 0 or 1 for symbol-heavy mode.
    pub num_tsids: usize,
}

impl WeightDimensions {
    /// Create new dimensions.
    pub fn new(num_tokens: usize, num_tsids: usize) -> Self {
        Self { num_tokens, num_tsids }
    }

    /// Check if this is weight-heavy mode (has tsid dimension).
    pub fn is_weight_heavy(&self) -> bool {
        self.num_tsids > 1
    }

    /// Total domain size: num_tokens × num_tsids.
    pub fn domain_size(&self) -> usize {
        if self.num_tsids == 0 {
            self.num_tokens
        } else {
            self.num_tokens * self.num_tsids
        }
    }

    /// Get the maximum position value (domain_size - 1).
    pub fn max_position(&self) -> usize {
        self.domain_size().saturating_sub(1)
    }

    /// Encode a token and tsid into a position.
    pub fn encode(&self, token: usize, tsid: usize) -> usize {
        debug_assert!(token < self.num_tokens, "token {} >= num_tokens {}", token, self.num_tokens);
        debug_assert!(tsid < self.num_tsids, "tsid {} >= num_tsids {}", tsid, self.num_tsids);
        token * self.num_tsids + tsid
    }

    /// Decode a position into (token, tsid).
    pub fn decode(&self, pos: usize) -> (usize, usize) {
        if self.num_tsids == 0 {
            (pos, 0)
        } else {
            (pos / self.num_tsids, pos % self.num_tsids)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    RangeSet,
    Factorized,
}

fn backend_choice() -> BackendChoice {
    match std::env::var("ABSTRACT_WEIGHT_BACKEND") {
        Ok(value) if value.eq_ignore_ascii_case("factorized") => BackendChoice::Factorized,
        _ => BackendChoice::RangeSet,
    }
}

fn normalize_num_tsids(num_tsids: usize) -> usize {
    if num_tsids == 0 { 1 } else { num_tsids }
}

fn current_num_tsids() -> usize {
    normalize_num_tsids(crate::datastructures::get_num_tsids())
}

// ---------------------------------------------------------------------------
// Factorized Weight Backend
// ---------------------------------------------------------------------------

/// Factorized weight representation as a union of (tsid_set × token_set) pairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactorizedWeight {
    pairs: Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)>,
    num_tsids: usize,
}

impl FactorizedWeight {
    fn new(num_tsids: usize) -> Self {
        Self {
            pairs: Vec::new(),
            num_tsids: normalize_num_tsids(num_tsids),
        }
    }

    fn num_tsids(&self) -> usize {
        normalize_num_tsids(self.num_tsids)
    }

    fn add_pair(&mut self, tsid_set: RangeSetBlaze<usize>, token_set: RangeSetBlaze<usize>) {
        if tsid_set.is_empty() || token_set.is_empty() {
            return;
        }
        for (existing_tsids, existing_tokens) in &mut self.pairs {
            if *existing_tsids == tsid_set {
                *existing_tokens |= &token_set;
                return;
            }
        }
        self.pairs.push((tsid_set, token_set));
    }

    fn normalize_pairs(&mut self) {
        let mut normalized = Vec::with_capacity(self.pairs.len());
        for (tsid_set, token_set) in std::mem::take(&mut self.pairs) {
            if tsid_set.is_empty() || token_set.is_empty() {
                continue;
            }
            let mut merged = false;
            for (existing_tsids, existing_tokens) in &mut normalized {
                if *existing_tsids == tsid_set {
                    *existing_tokens |= &token_set;
                    merged = true;
                    break;
                }
            }
            if !merged {
                normalized.push((tsid_set, token_set));
            }
        }
        self.pairs = normalized;
    }

    fn from_position_with_num_tsids(pos: usize, num_tsids: usize) -> Self {
        let num_tsids = normalize_num_tsids(num_tsids);
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        let tsid_set = RangeSetBlaze::from_iter([tsid..=tsid]);
        let token_set = RangeSetBlaze::from_iter([token..=token]);
        let mut weight = Self {
            pairs: vec![(tsid_set, token_set)],
            num_tsids,
        };
        weight.normalize_pairs();
        weight
    }

    fn all_with_max_position(max_position: usize, num_tsids: usize) -> Self {
        let num_tsids = normalize_num_tsids(num_tsids);
        if max_position == 0 {
            return Self::from_position_with_num_tsids(0, num_tsids);
        }

        let full_tsids = RangeSetBlaze::from_iter([0..=num_tsids - 1]);
        let full_tokens = max_position / num_tsids;
        let last_tsid = max_position % num_tsids;

        let mut weight = Self::new(num_tsids);
        if last_tsid == num_tsids - 1 {
            let token_set = RangeSetBlaze::from_iter([0..=full_tokens]);
            weight.add_pair(full_tsids, token_set);
        } else {
            if full_tokens > 0 {
                let token_set = RangeSetBlaze::from_iter([0..=full_tokens - 1]);
                weight.add_pair(full_tsids.clone(), token_set);
            }
            let token_set = RangeSetBlaze::from_iter([full_tokens..=full_tokens]);
            let tsid_set = RangeSetBlaze::from_iter([0..=last_tsid]);
            weight.add_pair(tsid_set, token_set);
        }
        weight.normalize_pairs();
        weight
    }

    fn from_rsb_with_num_tsids(rsb: &RangeSetBlaze<usize>, num_tsids: usize) -> Self {
        let num_tsids = normalize_num_tsids(num_tsids);
        if rsb.is_empty() {
            return Self::new(num_tsids);
        }

        let mut token_to_tsids: BTreeMap<usize, RangeSetBlaze<usize>> = BTreeMap::new();
        let full_tsid_set = RangeSetBlaze::from_iter([0..=num_tsids - 1]);

        for range in rsb.ranges() {
            let start = *range.start();
            let end = *range.end();
            let start_token = start / num_tsids;
            let end_token = end / num_tsids;
            let start_tsid = start % num_tsids;
            let end_tsid = end % num_tsids;

            if start_token == end_token {
                let entry = token_to_tsids.entry(start_token).or_insert_with(RangeSetBlaze::new);
                *entry |= &RangeSetBlaze::from_iter([start_tsid..=end_tsid]);
                continue;
            }

            let entry = token_to_tsids.entry(start_token).or_insert_with(RangeSetBlaze::new);
            *entry |= &RangeSetBlaze::from_iter([start_tsid..=num_tsids - 1]);

            if start_token + 1 <= end_token.saturating_sub(1) {
                for token in (start_token + 1)..=end_token - 1 {
                    let entry = token_to_tsids.entry(token).or_insert_with(RangeSetBlaze::new);
                    *entry |= &full_tsid_set;
                }
            }

            let entry = token_to_tsids.entry(end_token).or_insert_with(RangeSetBlaze::new);
            *entry |= &RangeSetBlaze::from_iter([0..=end_tsid]);
        }

        let mut weight = Self::new(num_tsids);
        for (token, tsid_set) in token_to_tsids {
            let token_set = RangeSetBlaze::from_iter([token..=token]);
            weight.add_pair(tsid_set, token_set);
        }
        weight.normalize_pairs();
        weight
    }

    pub fn expand_to_rsb(&self) -> RangeSetBlaze<usize> {
        if self.pairs.is_empty() {
            return RangeSetBlaze::new();
        }
        let num_tsids = self.num_tsids();
        let mut ranges: Vec<std::ops::RangeInclusive<usize>> = Vec::new();

        for (tsid_set, token_set) in &self.pairs {
            for token_range in token_set.ranges() {
                let token_start = *token_range.start();
                let token_end = *token_range.end();
                for tsid_range in tsid_set.ranges() {
                    let tsid_start = *tsid_range.start();
                    let tsid_end = *tsid_range.end();
                    for token in token_start..=token_end {
                        let base = token.saturating_mul(num_tsids);
                        ranges.push(base.saturating_add(tsid_start)..=base.saturating_add(tsid_end));
                    }
                }
            }
        }

        RangeSetBlaze::from_iter(ranges)
    }
}

// ---------------------------------------------------------------------------
// WeightBackend Trait - defines all operations weight backends must implement
// ---------------------------------------------------------------------------

/// Trait defining the interface for weight backends.
/// 
/// Implementors of this trait provide the underlying representation and
/// operations for weights. Multiple backends can exist (e.g., RangeSetBlaze,
/// BDD, etc.) and the `AbstractWeight` enum dispatches to the appropriate
/// backend based on the variant.
pub trait WeightBackend: Clone + PartialEq + Eq + std::fmt::Debug {
    /// Create an empty weight.
    fn empty() -> Self;
    
    /// Create a weight containing all positions from 0 to max_position (inclusive).
    fn all(max_position: usize) -> Self;
    
    /// Create a weight from a single position.
    fn from_position(pos: usize) -> Self;
    
    /// Create a weight from an iterator of inclusive ranges.
    fn from_ranges<I: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(ranges: I) -> Self;
    
    /// Check if the weight is empty.
    fn is_empty(&self) -> bool;
    
    /// Get the number of positions in the weight.
    fn len(&self) -> usize;
    
    /// Check if a position is in the weight.
    fn contains(&self, pos: usize) -> bool;
    
    /// Get the number of ranges in the weight.
    fn ranges_len(&self) -> usize;
    
    /// Iterate over ranges as (start, end) inclusive pairs.
    fn iter_ranges(&self) -> Box<dyn Iterator<Item = (usize, usize)> + '_>;
    
    /// Insert a position into the weight.
    fn insert(&mut self, pos: usize);
    
    /// Compute intersection with another weight.
    fn intersect(&self, other: &Self) -> Self;
    
    /// Compute intersection and mutate self.
    fn intersect_assign(&mut self, other: &Self);
    
    /// Compute union with another weight.
    fn union(&self, other: &Self) -> Self;
    
    /// Compute union and mutate self.
    fn union_assign(&mut self, other: &Self);
    
    /// Compute set difference (self - other).
    fn difference(&self, other: &Self) -> Self;
    
    /// Compute complement within the given max position (0..=max_position).
    fn complement(&self, max_position: usize) -> Self;
    
    /// Get the minimum position, if any.
    fn min_item(&self) -> Option<usize>;
    
    /// Get the maximum position, if any.
    fn max_item(&self) -> Option<usize>;
    
    /// Clip to positions <= max.
    fn clip_max(&mut self, max: usize);
}

// ---------------------------------------------------------------------------
// RangeSetBlaze Backend Implementation
// ---------------------------------------------------------------------------

impl WeightBackend for RangeSetBlaze<usize> {
    fn empty() -> Self {
        RangeSetBlaze::new()
    }
    
    fn all(max_position: usize) -> Self {
        RangeSetBlaze::from_iter([0..=max_position])
    }
    
    fn from_position(pos: usize) -> Self {
        RangeSetBlaze::from_iter([pos..=pos])
    }
    
    fn from_ranges<I: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(ranges: I) -> Self {
        RangeSetBlaze::from_iter(ranges)
    }
    
    fn is_empty(&self) -> bool {
        RangeSetBlaze::is_empty(self)
    }
    
    fn len(&self) -> usize {
        RangeSetBlaze::len(self) as usize
    }
    
    fn contains(&self, pos: usize) -> bool {
        RangeSetBlaze::contains(self, pos)
    }
    
    fn ranges_len(&self) -> usize {
        RangeSetBlaze::ranges_len(self)
    }
    
    fn iter_ranges(&self) -> Box<dyn Iterator<Item = (usize, usize)> + '_> {
        Box::new(self.ranges().map(|r| (*r.start(), *r.end())))
    }
    
    fn insert(&mut self, pos: usize) {
        RangeSetBlaze::insert(self, pos);
    }
    
    fn intersect(&self, other: &Self) -> Self {
        self & other
    }
    
    fn intersect_assign(&mut self, other: &Self) {
        *self = self.intersect(other);
    }
    
    fn union(&self, other: &Self) -> Self {
        self | other
    }
    
    fn union_assign(&mut self, other: &Self) {
        *self |= other;
    }
    
    fn difference(&self, other: &Self) -> Self {
        self - other
    }
    
    fn complement(&self, max_position: usize) -> Self {
        let all = RangeSetBlaze::from_iter([0..=max_position]);
        &all - self
    }
    
    fn min_item(&self) -> Option<usize> {
        self.ranges().next().map(|r| *r.start())
    }
    
    fn max_item(&self) -> Option<usize> {
        self.ranges().last().map(|r| *r.end())
    }
    
    fn clip_max(&mut self, max: usize) {
        let clipped = self.intersect(&RangeSetBlaze::from_iter([0..=max]));
        *self = clipped;
    }
}

impl WeightBackend for FactorizedWeight {
    fn empty() -> Self {
        FactorizedWeight::new(current_num_tsids())
    }

    fn all(max_position: usize) -> Self {
        FactorizedWeight::all_with_max_position(max_position, current_num_tsids())
    }

    fn from_position(pos: usize) -> Self {
        FactorizedWeight::from_position_with_num_tsids(pos, current_num_tsids())
    }

    fn from_ranges<I: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(ranges: I) -> Self {
        let rsb = RangeSetBlaze::from_iter(ranges);
        FactorizedWeight::from_rsb_with_num_tsids(&rsb, current_num_tsids())
    }

    fn is_empty(&self) -> bool {
        self.pairs.is_empty() || self.pairs.iter().all(|(a, b)| a.is_empty() || b.is_empty())
    }

    fn len(&self) -> usize {
        self.expand_to_rsb().len() as usize
    }

    fn contains(&self, pos: usize) -> bool {
        if self.pairs.is_empty() {
            return false;
        }
        let num_tsids = self.num_tsids();
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        self.pairs.iter().any(|(tsid_set, token_set)| {
            tsid_set.contains(tsid) && token_set.contains(token)
        })
    }

    fn ranges_len(&self) -> usize {
        self.expand_to_rsb().ranges_len()
    }

    fn iter_ranges(&self) -> Box<dyn Iterator<Item = (usize, usize)> + '_> {
        let ranges: Vec<(usize, usize)> = self
            .expand_to_rsb()
            .ranges()
            .map(|r| (*r.start(), *r.end()))
            .collect();
        Box::new(ranges.into_iter())
    }

    fn insert(&mut self, pos: usize) {
        let num_tsids = self.num_tsids();
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        let tsid_set = RangeSetBlaze::from_iter([tsid..=tsid]);
        let token_set = RangeSetBlaze::from_iter([token..=token]);
        self.add_pair(tsid_set, token_set);
        self.normalize_pairs();
    }

    fn intersect(&self, other: &Self) -> Self {
        assert_eq!(self.num_tsids(), other.num_tsids(), "FactorizedWeight num_tsids mismatch");
        let mut out = FactorizedWeight::new(self.num_tsids());
        for (tsid_a, token_a) in &self.pairs {
            for (tsid_b, token_b) in &other.pairs {
                let tsid_inter = tsid_a & tsid_b;
                let token_inter = token_a & token_b;
                if !tsid_inter.is_empty() && !token_inter.is_empty() {
                    out.add_pair(tsid_inter, token_inter);
                }
            }
        }
        out.normalize_pairs();
        out
    }

    fn intersect_assign(&mut self, other: &Self) {
        *self = self.intersect(other);
    }

    fn union(&self, other: &Self) -> Self {
        assert_eq!(self.num_tsids(), other.num_tsids(), "FactorizedWeight num_tsids mismatch");
        let mut out = self.clone();
        for (tsid_set, token_set) in &other.pairs {
            out.add_pair(tsid_set.clone(), token_set.clone());
        }
        out.normalize_pairs();
        out
    }

    fn union_assign(&mut self, other: &Self) {
        assert_eq!(self.num_tsids(), other.num_tsids(), "FactorizedWeight num_tsids mismatch");
        for (tsid_set, token_set) in &other.pairs {
            self.add_pair(tsid_set.clone(), token_set.clone());
        }
        self.normalize_pairs();
    }

    fn difference(&self, other: &Self) -> Self {
        assert_eq!(self.num_tsids(), other.num_tsids(), "FactorizedWeight num_tsids mismatch");
        let expanded_self = self.expand_to_rsb();
        let expanded_other = other.expand_to_rsb();
        let diff = &expanded_self - &expanded_other;
        FactorizedWeight::from_rsb_with_num_tsids(&diff, self.num_tsids())
    }

    fn complement(&self, max_position: usize) -> Self {
        let all = RangeSetBlaze::from_iter([0..=max_position]);
        let expanded_self = self.expand_to_rsb();
        let diff = &all - &expanded_self;
        FactorizedWeight::from_rsb_with_num_tsids(&diff, self.num_tsids())
    }

    fn min_item(&self) -> Option<usize> {
        let num_tsids = self.num_tsids();
        self.pairs
            .iter()
            .filter_map(|(tsid_set, token_set)| {
                let min_token = token_set.ranges().next().map(|r| *r.start())?;
                let min_tsid = tsid_set.ranges().next().map(|r| *r.start())?;
                Some(min_token.saturating_mul(num_tsids).saturating_add(min_tsid))
            })
            .min()
    }

    fn max_item(&self) -> Option<usize> {
        let num_tsids = self.num_tsids();
        self.pairs
            .iter()
            .filter_map(|(tsid_set, token_set)| {
                let max_token = token_set.ranges().last().map(|r| *r.end())?;
                let max_tsid = tsid_set.ranges().last().map(|r| *r.end())?;
                Some(max_token.saturating_mul(num_tsids).saturating_add(max_tsid))
            })
            .max()
    }

    fn clip_max(&mut self, max: usize) {
        let expanded = self.expand_to_rsb();
        let clipped = expanded.intersect(&RangeSetBlaze::from_iter([0..=max]));
        *self = FactorizedWeight::from_rsb_with_num_tsids(&clipped, self.num_tsids());
    }
}

// ---------------------------------------------------------------------------
// AbstractWeight Enum - dispatches to backends
// ---------------------------------------------------------------------------

/// Abstract weight type wrapping different backends.
/// 
/// This enum provides a unified interface for weight operations.
/// Operations between different variants will panic - all weights in a
/// computation must use the same backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbstractWeight {
    /// Weight represented as a RangeSetBlaze.
    RangeSet(RangeSetBlaze<usize>),
    /// Weight represented as a factorized (tsid_set × token_set) union.
    Factorized(FactorizedWeight),
    // Future variants can be added here, e.g.:
    // Bdd(BddWeight),
    // Explicit(Vec<usize>),
}


impl Default for AbstractWeight {
    fn default() -> Self {
        AbstractWeight::empty()
    }
}

impl AbstractWeight {
    /// Create an empty weight.
    pub fn empty() -> Self {
        match backend_choice() {
            BackendChoice::RangeSet => AbstractWeight::RangeSet(RangeSetBlaze::<usize>::empty()),
            BackendChoice::Factorized => {
                AbstractWeight::Factorized(FactorizedWeight::new(current_num_tsids()))
            }
        }
    }

    /// Create a weight containing all positions in the given range.
    pub fn all(dims: WeightDimensions) -> Self {
        if dims.domain_size() == 0 {
            return Self::empty();
        }
        match backend_choice() {
            BackendChoice::RangeSet => {
                AbstractWeight::RangeSet(RangeSetBlaze::<usize>::all(dims.max_position()))
            }
            BackendChoice::Factorized => AbstractWeight::Factorized(
                FactorizedWeight::all_with_max_position(
                    dims.max_position(),
                    normalize_num_tsids(dims.num_tsids),
                ),
            ),
        }
    }

    /// Create a weight from a single position.
    pub fn from_position(pos: usize) -> Self {
        match backend_choice() {
            BackendChoice::RangeSet => {
                AbstractWeight::RangeSet(RangeSetBlaze::<usize>::from_position(pos))
            }
            BackendChoice::Factorized => AbstractWeight::Factorized(
                FactorizedWeight::from_position_with_num_tsids(pos, current_num_tsids()),
            ),
        }
    }

    /// Create a weight from an iterator of positions.
    pub fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        AbstractWeight::from_rsb(RangeSetBlaze::from_iter(iter))
    }
    
    /// Create a weight from an iterator of inclusive ranges.
    pub fn from_ranges<I: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(ranges: I) -> Self {
        AbstractWeight::from_rsb(RangeSetBlaze::<usize>::from_ranges(ranges))
    }

    /// Create a weight from a RangeSetBlaze.
    pub fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self {
        match backend_choice() {
            BackendChoice::RangeSet => AbstractWeight::RangeSet(rsb),
            BackendChoice::Factorized => AbstractWeight::Factorized(
                FactorizedWeight::from_rsb_with_num_tsids(&rsb, current_num_tsids()),
            ),
        }
    }

    /// Check if the weight is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::is_empty(rsb),
            AbstractWeight::Factorized(fw) => WeightBackend::is_empty(fw),
        }
    }

    /// Get the number of positions in the weight.
    pub fn len(&self) -> usize {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::len(rsb),
            AbstractWeight::Factorized(fw) => WeightBackend::len(fw),
        }
    }

    /// Check if a position is in the weight.
    pub fn contains(&self, pos: usize) -> bool {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::contains(rsb, pos),
            AbstractWeight::Factorized(fw) => WeightBackend::contains(fw, pos),
        }
    }

    /// Expand to a RangeSetBlaze representation.
    pub fn to_rsb(&self) -> RangeSetBlaze<usize> {
        match self {
            AbstractWeight::RangeSet(rsb) => rsb.clone(),
            AbstractWeight::Factorized(fw) => fw.expand_to_rsb(),
        }
    }

    /// Expand to a RangeSetBlaze representation (alias for `to_rsb`).
    pub fn expand_to_rsb(&self) -> RangeSetBlaze<usize> {
        self.to_rsb()
    }

    /// Get the number of ranges in the weight.
    pub fn ranges_len(&self) -> usize {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::ranges_len(rsb),
            AbstractWeight::Factorized(fw) => WeightBackend::ranges_len(fw),
        }
    }

    /// Iterate over ranges as (start, end) inclusive pairs.
    pub fn iter_ranges(&self) -> Box<dyn Iterator<Item = (usize, usize)> + '_> {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::iter_ranges(rsb),
            AbstractWeight::Factorized(fw) => WeightBackend::iter_ranges(fw),
        }
    }

    /// Iterate over ranges.
    pub fn ranges(&self) -> Box<dyn Iterator<Item = std::ops::RangeInclusive<usize>> + '_> {
        match self {
            AbstractWeight::RangeSet(rsb) => Box::new(rsb.ranges()),
            AbstractWeight::Factorized(fw) => {
                let ranges: Vec<_> = fw.expand_to_rsb().ranges().collect();
                Box::new(ranges.into_iter())
            }
        }
    }

    /// Insert a position.
    pub fn insert(&mut self, pos: usize) {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::insert(rsb, pos),
            AbstractWeight::Factorized(fw) => WeightBackend::insert(fw, pos),
        }
    }
    
    /// Get the minimum position, if any.
    pub fn min_item(&self) -> Option<usize> {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::min_item(rsb),
            AbstractWeight::Factorized(fw) => WeightBackend::min_item(fw),
        }
    }
    
    /// Get the maximum position, if any.
    pub fn max_item(&self) -> Option<usize> {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::max_item(rsb),
            AbstractWeight::Factorized(fw) => WeightBackend::max_item(fw),
        }
    }
    
    /// Compute set difference (self - other).
    /// 
    /// Panics if self and other are different variants.
    pub fn difference(&self, other: &Self) -> Self {
        match (self, other) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::difference(a, b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::difference(a, b))
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
    
    /// Clip to positions <= max.
    pub fn clip_max(&mut self, max: usize) {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::clip_max(rsb, max),
            AbstractWeight::Factorized(fw) => WeightBackend::clip_max(fw, max),
        }
    }
}

// ---------------------------------------------------------------------------
// Bitwise Operations - dispatch to backends with variant checking
// ---------------------------------------------------------------------------

impl BitAnd for AbstractWeight {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::intersect(&a, &b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::intersect(&a, &b))
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl BitAnd for &AbstractWeight {
    type Output = AbstractWeight;

    fn bitand(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::intersect(a, b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::intersect(a, b))
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl BitAndAssign for AbstractWeight {
    fn bitand_assign(&mut self, rhs: Self) {
        match (self, &rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                WeightBackend::intersect_assign(a, b);
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                WeightBackend::intersect_assign(a, b);
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl BitOr for AbstractWeight {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::union(&a, &b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::union(&a, &b))
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl BitOr for &AbstractWeight {
    type Output = AbstractWeight;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::union(a, b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::union(a, b))
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl BitOrAssign for AbstractWeight {
    fn bitor_assign(&mut self, rhs: Self) {
        match (self, &rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                WeightBackend::union_assign(a, b);
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                WeightBackend::union_assign(a, b);
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl BitOrAssign<&AbstractWeight> for AbstractWeight {
    fn bitor_assign(&mut self, rhs: &AbstractWeight) {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                WeightBackend::union_assign(a, b);
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                WeightBackend::union_assign(a, b);
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl Not for AbstractWeight {
    type Output = Self;

    fn not(self) -> Self::Output {
        // Note: This requires knowing the domain size, which we don't have here.
        // For now, this is a placeholder that should be used carefully.
        panic!("Not operation on AbstractWeight requires domain knowledge. Use complement_with_dims instead.");
    }
}

impl AbstractWeight {
    /// Compute the complement within the given dimensions.
    pub fn complement_with_dims(&self, dims: WeightDimensions) -> Self {
        if dims.domain_size() == 0 {
            return Self::empty();
        }
        match self {
            AbstractWeight::RangeSet(rsb) => {
                AbstractWeight::RangeSet(WeightBackend::complement(rsb, dims.max_position()))
            }
            AbstractWeight::Factorized(fw) => {
                assert_eq!(
                    fw.num_tsids(),
                    normalize_num_tsids(dims.num_tsids),
                    "FactorizedWeight dimensions mismatch in complement_with_dims"
                );
                AbstractWeight::Factorized(WeightBackend::complement(fw, dims.max_position()))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Conversion Traits
// ---------------------------------------------------------------------------

impl From<RangeSetBlaze<usize>> for AbstractWeight {
    fn from(rsb: RangeSetBlaze<usize>) -> Self {
        AbstractWeight::from_rsb(rsb)
    }
}

impl From<&RangeSetBlaze<usize>> for AbstractWeight {
    fn from(rsb: &RangeSetBlaze<usize>) -> Self {
        AbstractWeight::from_rsb(rsb.clone())
    }
}

impl From<FactorizedWeight> for AbstractWeight {
    fn from(weight: FactorizedWeight) -> Self {
        AbstractWeight::Factorized(weight)
    }
}

impl From<&FactorizedWeight> for AbstractWeight {
    fn from(weight: &FactorizedWeight) -> Self {
        AbstractWeight::Factorized(weight.clone())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    type Backend = RangeSetBlaze<usize>;

    #[test]
    fn test_weight_dimensions_encoding() {
        let dims = WeightDimensions::new(100, 10);
        
        // Token 5, tsid 3 -> position 53
        assert_eq!(dims.encode(5, 3), 53);
        assert_eq!(dims.decode(53), (5, 3));
        
        // Token 0, tsid 0 -> position 0
        assert_eq!(dims.encode(0, 0), 0);
        assert_eq!(dims.decode(0), (0, 0));
        
        // Token 99, tsid 9 -> position 999
        assert_eq!(dims.encode(99, 9), 999);
        assert_eq!(dims.decode(999), (99, 9));
    }

    #[test]
    fn test_weight_dimensions_domain_size() {
        let dims = WeightDimensions::new(100, 10);
        assert_eq!(dims.domain_size(), 1000);
        assert_eq!(dims.max_position(), 999);
        assert!(dims.is_weight_heavy());
        
        let symbol_heavy = WeightDimensions::new(100, 1);
        assert_eq!(symbol_heavy.domain_size(), 100);
        assert!(!symbol_heavy.is_weight_heavy());
    }

    #[test]
    fn test_abstract_weight_basic() {
        let w = AbstractWeight::from_iter([1, 2, 3, 5, 6, 7]);
        assert_eq!(w.len(), 6);
        assert!(!w.is_empty());
        assert!(w.contains(1));
        assert!(w.contains(5));
        assert!(!w.contains(4));
    }

    #[test]
    fn test_abstract_weight_intersection() {
        let a = AbstractWeight::from_iter([1, 2, 3, 4, 5]);
        let b = AbstractWeight::from_iter([3, 4, 5, 6, 7]);
        let c = &a & &b;
        assert_eq!(c.len(), 3);
        assert!(c.contains(3));
        assert!(c.contains(4));
        assert!(c.contains(5));
    }

    #[test]
    fn test_abstract_weight_union() {
        let a = AbstractWeight::from_iter([1, 2, 3]);
        let b = AbstractWeight::from_iter([3, 4, 5]);
        let c = &a | &b;
        assert_eq!(c.len(), 5);
    }

    #[test]
    fn test_abstract_weight_all() {
        let dims = WeightDimensions::new(5, 3);
        let w = AbstractWeight::all(dims);
        assert_eq!(w.len(), 15);
        assert!(w.contains(0));
        assert!(w.contains(14));
    }

    #[test]
    fn test_backend_ranges_and_len() {
        let backend = <Backend as WeightBackend>::from_ranges([1..=3, 7..=9]);
        assert_eq!(<Backend as WeightBackend>::ranges_len(&backend), 2);
        let ranges: Vec<(usize, usize)> = <Backend as WeightBackend>::iter_ranges(&backend).collect();
        assert_eq!(ranges, vec![(1, 3), (7, 9)]);
        assert_eq!(<Backend as WeightBackend>::len(&backend), 6);
    }

    #[test]
    fn test_backend_set_ops_and_complement() {
        let a = <Backend as WeightBackend>::from_ranges([1..=4]);
        let b = <Backend as WeightBackend>::from_ranges([3..=6]);
        let intersect = <Backend as WeightBackend>::intersect(&a, &b);
        let union = <Backend as WeightBackend>::union(&a, &b);
        let diff = <Backend as WeightBackend>::difference(&a, &b);
        let comp = <Backend as WeightBackend>::complement(&a, 6);

        assert!(<Backend as WeightBackend>::contains(&intersect, 3));
        assert!(!<Backend as WeightBackend>::contains(&intersect, 2));
        assert!(<Backend as WeightBackend>::contains(&union, 6));
        assert!(<Backend as WeightBackend>::contains(&diff, 1));
        assert!(!<Backend as WeightBackend>::contains(&diff, 5));
        assert!(!<Backend as WeightBackend>::contains(&comp, 2));
        assert!(<Backend as WeightBackend>::contains(&comp, 6));
    }

    #[test]
    fn test_backend_min_max_and_clip() {
        let mut backend = <Backend as WeightBackend>::from_ranges([2..=4, 8..=10]);
        assert_eq!(<Backend as WeightBackend>::min_item(&backend), Some(2));
        assert_eq!(<Backend as WeightBackend>::max_item(&backend), Some(10));

        <Backend as WeightBackend>::clip_max(&mut backend, 5);
        assert_eq!(<Backend as WeightBackend>::max_item(&backend), Some(4));
        assert!(<Backend as WeightBackend>::contains(&backend, 2));
        assert!(!<Backend as WeightBackend>::contains(&backend, 8));
    }

    #[test]
    fn test_factorized_expand_roundtrip() {
        let num_tsids = 3;
        let rsb = RangeSetBlaze::from_iter([0..=2, 4..=5, 8..=8, 10..=12]);
        let fw = FactorizedWeight::from_rsb_with_num_tsids(&rsb, num_tsids);
        assert_eq!(fw.expand_to_rsb(), rsb);
    }

    #[test]
    fn test_factorized_set_ops_match_rsb() {
        let num_tsids = 4;
        let a_rsb = RangeSetBlaze::from_iter([0..=7, 10..=12]);
        let b_rsb = RangeSetBlaze::from_iter([5..=9, 12..=15]);
        let a = FactorizedWeight::from_rsb_with_num_tsids(&a_rsb, num_tsids);
        let b = FactorizedWeight::from_rsb_with_num_tsids(&b_rsb, num_tsids);

        let inter = <FactorizedWeight as WeightBackend>::intersect(&a, &b);
        let union = <FactorizedWeight as WeightBackend>::union(&a, &b);

        assert_eq!(inter.expand_to_rsb(), &a_rsb & &b_rsb);
        assert_eq!(union.expand_to_rsb(), &a_rsb | &b_rsb);
    }
}
