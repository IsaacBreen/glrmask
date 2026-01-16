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
use crate::datastructures::factorized_weight::FactorizedWeight;
use crate::json_serialization::{JSONConvertible, JSONNode};
use serde::{Deserialize, Serialize};
use serde::de::Error;
use serde_json::Value as JsonValue;
use std::hash::{Hash, Hasher};
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

pub(crate) fn normalize_num_tsids(num_tsids: usize) -> usize {
    if num_tsids == 0 { 1 } else { num_tsids }
}

pub(crate) fn current_num_tsids() -> usize {
    normalize_num_tsids(crate::datastructures::get_num_tsids())
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

impl std::hash::Hash for AbstractWeight {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            AbstractWeight::RangeSet(rsb) => {
                0u8.hash(state);
                for range in rsb.ranges() {
                    range.start().hash(state);
                    range.end().hash(state);
                }
            }
            AbstractWeight::Factorized(fw) => {
                1u8.hash(state);
                fw.hash(state);
            }
        }
    }
}

impl PartialOrd for AbstractWeight {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Compare by len, then by first range
        match self.len().cmp(&other.len()) {
            std::cmp::Ordering::Equal => {
                let self_first = self.min_item();
                let other_first = other.min_item();
                self_first.partial_cmp(&other_first)
            }
            ord => Some(ord),
        }
    }
}

impl Ord for AbstractWeight {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl std::fmt::Display for AbstractWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AbstractWeight::RangeSet(rsb) => {
                let ranges: Vec<_> = rsb.ranges().take(5).collect();
                if ranges.len() < rsb.ranges_len() {
                    write!(f, "Weight({} ranges, {} items)", rsb.ranges_len(), self.len())
                } else {
                    write!(f, "Weight({:?})", ranges)
                }
            }
            AbstractWeight::Factorized(fw) => {
                write!(f, "FactorizedWeight({} pairs)", fw.pairs.len())
            }
        }
    }
}

impl Serialize for AbstractWeight {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let ranges: Vec<(usize, usize)> = self
            .to_rsb()
            .ranges()
            .map(|r| (*r.start(), *r.end()))
            .collect();
        ranges.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AbstractWeight {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = JsonValue::deserialize(deserializer)?;
        let arr = match value {
            JsonValue::Array(arr) => arr,
            other => {
                return Err(D::Error::custom(format!(
                    "Expected weight as JSON array, got {}",
                    other
                )))
            }
        };

        if arr.is_empty() {
            return Ok(AbstractWeight::empty());
        }

        if arr.iter().all(|v| v.is_array()) {
            let mut ranges = Vec::with_capacity(arr.len());
            for item in arr {
                let pair = item
                    .as_array()
                    .ok_or_else(|| D::Error::custom("Expected array for range pair"))?;
                if pair.len() != 2 {
                    return Err(D::Error::custom(format!(
                        "Expected 2-element range pair, got {}",
                        pair.len()
                    )));
                }
                let start = pair[0]
                    .as_u64()
                    .ok_or_else(|| D::Error::custom("Range start must be a non-negative integer"))?
                    as usize;
                let end = pair[1]
                    .as_u64()
                    .ok_or_else(|| D::Error::custom("Range end must be a non-negative integer"))?
                    as usize;
                ranges.push(start..=end);
            }
            let rsb = RangeSetBlaze::from_iter(ranges);
            return Ok(AbstractWeight::from_rsb(rsb));
        }

        if arr.iter().all(|v| v.is_number()) {
            if arr.len() % 2 != 0 {
                return Err(D::Error::custom(format!(
                    "Expected even number of flattened range endpoints, got {}",
                    arr.len()
                )));
            }
            let mut ranges = Vec::with_capacity(arr.len() / 2);
            let mut iter = arr.into_iter();
            while let (Some(start_val), Some(end_val)) = (iter.next(), iter.next()) {
                let start = start_val
                    .as_u64()
                    .ok_or_else(|| D::Error::custom("Range start must be a non-negative integer"))?
                    as usize;
                let end = end_val
                    .as_u64()
                    .ok_or_else(|| D::Error::custom("Range end must be a non-negative integer"))?
                    as usize;
                ranges.push(start..=end);
            }
            let rsb = RangeSetBlaze::from_iter(ranges);
            return Ok(AbstractWeight::from_rsb(rsb));
        }

        Err(D::Error::custom(
            "Expected weight as array of [start,end] pairs or flattened endpoints",
        ))
    }
}

impl JSONConvertible for AbstractWeight {
    fn to_json(&self) -> JSONNode {
        let rsb = self.to_rsb();
        crate::datastructures::hybrid_bitset::RangeSet::from(rsb).to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let rsb = crate::datastructures::hybrid_bitset::RangeSet::from_json(node)?;
        Ok(AbstractWeight::from_rsb(std::sync::Arc::unwrap_or_clone(rsb.inner)))
    }
}

impl std::ops::Not for AbstractWeight {
    type Output = AbstractWeight;
    
    fn not(self) -> Self::Output {
        self.complement()
    }
}

impl std::ops::Not for &AbstractWeight {
    type Output = AbstractWeight;
    
    fn not(self) -> Self::Output {
        self.complement()
    }
}

impl Default for AbstractWeight {
    fn default() -> Self {
        AbstractWeight::empty()
    }
}

impl AbstractWeight {
    /// Create an empty weight (no positions).
    pub fn empty() -> Self {
        match backend_choice() {
            BackendChoice::RangeSet => AbstractWeight::RangeSet(RangeSetBlaze::<usize>::empty()),
            BackendChoice::Factorized => {
                AbstractWeight::Factorized(FactorizedWeight::new(current_num_tsids()))
            }
        }
    }
    
    /// Create an empty weight (alias for `empty()`).
    pub fn zeros() -> Self {
        Self::empty()
    }
    
    /// Create a weight containing all positions.
    /// 
    /// Uses the global dims (from set_global_dims) to determine the domain.
    /// This is the preferred no-arg version.
    pub fn ones() -> Self {
        let max_llm_token = crate::datastructures::get_max_llm_token();
        let num_tsids = crate::datastructures::get_num_tsids();
        let domain_max = if num_tsids > 1 {
            max_llm_token
                .saturating_mul(num_tsids)
                .saturating_add(num_tsids.saturating_sub(1))
        } else {
            max_llm_token
        };
        match backend_choice() {
            BackendChoice::RangeSet => {
                AbstractWeight::RangeSet(RangeSetBlaze::<usize>::from_iter([0..=domain_max]))
            }
            BackendChoice::Factorized => AbstractWeight::Factorized(
                FactorizedWeight::all_with_max_position(
                    domain_max,
                    normalize_num_tsids(num_tsids),
                ),
            ),
        }
    }
    
    /// Create a weight containing all positions (alias for `ones()`).
    /// 
    /// Uses the global dims (from set_global_dims) to determine the domain.
    pub fn all() -> Self {
        Self::ones()
    }

    /// Create a weight containing all positions in the given range.
    pub fn all_with_dims(dims: WeightDimensions) -> Self {
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

    /// Create a weight from a single position (alias for `from_position`).
    pub fn from_item(pos: usize) -> Self {
        Self::from_position(pos)
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
    
    /// Alias for `ranges_len()`.
    pub fn num_ranges(&self) -> usize {
        self.ranges_len()
    }
    
    /// Fast check if weight represents all positions in the domain.
    /// 
    /// Uses global dims to determine domain size.
    pub fn is_all_fast(&self) -> bool {
        let max_llm_token = crate::datastructures::get_max_llm_token();
        let num_tsids = normalize_num_tsids(crate::datastructures::get_num_tsids());
        let domain_size = max_llm_token
            .saturating_add(1)
            .saturating_mul(num_tsids);
        // Check if ranges_len == 1 and len == domain_size
        self.ranges_len() == 1 && self.len() == domain_size
    }
    
    /// Check if self is a subset of other.
    pub fn is_subset_of(&self, other: &Self) -> bool {
        // self is subset of other if (self - other) is empty
        match (self, other) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                a.clone().difference(&b.clone()).is_empty()
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                WeightBackend::difference(a, b).is_empty()
            }
            _ => {
                // Mixed variants: expand both to RangeSet and compare
                let a_rsb = self.to_rsb();
                let b_rsb = other.to_rsb();
                a_rsb.difference(&b_rsb).is_empty()
            }
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

    /// Remove a position.
    pub fn remove(&mut self, pos: usize) {
        let single = AbstractWeight::from_position(pos);
        *self = self.difference(&single);
    }

    /// Set or clear a position.
    pub fn set(&mut self, pos: usize, value: bool) {
        if value {
            self.insert(pos);
        } else {
            self.remove(pos);
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

impl BitAnd<&AbstractWeight> for AbstractWeight {
    type Output = AbstractWeight;

    fn bitand(self, rhs: &AbstractWeight) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::intersect(&a, b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::intersect(&a, b))
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

impl BitAndAssign<&AbstractWeight> for AbstractWeight {
    fn bitand_assign(&mut self, rhs: &AbstractWeight) {
        match (self, rhs) {
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

impl BitOr<&AbstractWeight> for AbstractWeight {
    type Output = AbstractWeight;

    fn bitor(self, rhs: &AbstractWeight) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::union(&a, b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::union(&a, b))
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

impl BitOrAssign<&&AbstractWeight> for AbstractWeight {
    fn bitor_assign(&mut self, rhs: &&AbstractWeight) {
        self.bitor_assign(*rhs);
    }
}

impl std::ops::Sub for AbstractWeight {
    type Output = AbstractWeight;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::difference(&a, &b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::difference(&a, &b))
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl std::ops::Sub for &AbstractWeight {
    type Output = AbstractWeight;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::difference(a, b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::difference(a, b))
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl std::ops::SubAssign<&AbstractWeight> for AbstractWeight {
    fn sub_assign(&mut self, rhs: &AbstractWeight) {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                *a = WeightBackend::difference(a, b);
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                *a = WeightBackend::difference(a, b);
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl AbstractWeight {
    /// Compute the complement using global dimensions.
    pub fn complement(&self) -> Self {
        let max_llm_token = crate::datastructures::get_max_llm_token();
        let num_tsids = crate::datastructures::get_num_tsids();
        let domain_max = if num_tsids > 1 {
            max_llm_token
                .saturating_mul(num_tsids)
                .saturating_add(num_tsids.saturating_sub(1))
        } else {
            max_llm_token
        };
        match self {
            AbstractWeight::RangeSet(rsb) => {
                AbstractWeight::RangeSet(WeightBackend::complement(rsb, domain_max))
            }
            AbstractWeight::Factorized(fw) => {
                assert_eq!(
                    fw.num_tsids(),
                    normalize_num_tsids(num_tsids),
                    "FactorizedWeight dimensions mismatch in complement"
                );
                AbstractWeight::Factorized(WeightBackend::complement(fw, domain_max))
            }
        }
    }

    /// Compute a stable fingerprint for hashing/grouping weights.
    pub fn fingerprint(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Iterate over positions up to and including max.
    pub fn iter_up_to(&self, max: usize) -> impl Iterator<Item = usize> + '_ {
        let rsb = self.to_rsb();
        let clipped = &rsb & &RangeSetBlaze::from_iter([0..=max]);
        clipped.into_iter()
    }
    
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

impl From<crate::datastructures::hybrid_bitset::RangeSet> for AbstractWeight {
    fn from(rsb: crate::datastructures::hybrid_bitset::RangeSet) -> Self {
        AbstractWeight::from_rsb(std::sync::Arc::unwrap_or_clone(rsb.inner))
    }
}

impl From<&crate::datastructures::hybrid_bitset::RangeSet> for AbstractWeight {
    fn from(rsb: &crate::datastructures::hybrid_bitset::RangeSet) -> Self {
        AbstractWeight::from_rsb(std::sync::Arc::unwrap_or_clone(rsb.inner.clone()))
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
        let w = AbstractWeight::all_with_dims(dims);
        assert_eq!(w.len(), 15);
        assert!(w.contains(0));
        assert!(w.contains(14));
    }

    #[test]
    fn test_backend_ranges_and_len() {
        let backend = <Backend as WeightBackend>::from_ranges([1..=3, 7..=9]);
        assert_eq!(<Backend as WeightBackend>::ranges_len(&backend), 2);
        let ranges: Vec<(usize, usize)> = backend
            .ranges()
            .map(|r| (*r.start(), *r.end()))
            .collect();
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
    fn test_backend_min_max() {
        let backend = <Backend as WeightBackend>::from_ranges([2..=4, 8..=10]);
        assert_eq!(<Backend as WeightBackend>::min_item(&backend), Some(2));
        assert_eq!(<Backend as WeightBackend>::max_item(&backend), Some(10));
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
