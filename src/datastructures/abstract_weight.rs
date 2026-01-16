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

use range_set_blaze::RangeSetBlaze;
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
    // Future variants can be added here, e.g.:
    // Bdd(BddWeight),
    // Explicit(Vec<usize>),
}

/// Helper macro to extract matching variants or panic.
macro_rules! match_variant {
    ($a:expr, $b:expr, $variant:ident) => {
        match ($a, $b) {
            (AbstractWeight::$variant(a), AbstractWeight::$variant(b)) => (a, b),
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    };
}

impl Default for AbstractWeight {
    fn default() -> Self {
        AbstractWeight::RangeSet(RangeSetBlaze::new())
    }
}

impl AbstractWeight {
    /// Create an empty weight.
    pub fn empty() -> Self {
        AbstractWeight::RangeSet(RangeSetBlaze::<usize>::empty())
    }

    /// Create a weight containing all positions in the given range.
    pub fn all(dims: WeightDimensions) -> Self {
        if dims.domain_size() == 0 {
            return Self::empty();
        }
        AbstractWeight::RangeSet(RangeSetBlaze::<usize>::all(dims.max_position()))
    }

    /// Create a weight from a single position.
    pub fn from_position(pos: usize) -> Self {
        AbstractWeight::RangeSet(RangeSetBlaze::<usize>::from_position(pos))
    }

    /// Create a weight from an iterator of positions.
    pub fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        AbstractWeight::RangeSet(RangeSetBlaze::from_iter(iter))
    }
    
    /// Create a weight from an iterator of inclusive ranges.
    pub fn from_ranges<I: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(ranges: I) -> Self {
        AbstractWeight::RangeSet(RangeSetBlaze::<usize>::from_ranges(ranges))
    }

    /// Create a weight from a RangeSetBlaze.
    pub fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self {
        AbstractWeight::RangeSet(rsb)
    }

    /// Check if the weight is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::is_empty(rsb),
        }
    }

    /// Get the number of positions in the weight.
    pub fn len(&self) -> usize {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::len(rsb),
        }
    }

    /// Check if a position is in the weight.
    pub fn contains(&self, pos: usize) -> bool {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::contains(rsb, pos),
        }
    }

    /// Get the underlying RangeSetBlaze (if applicable).
    /// 
    /// Panics if this weight is not a RangeSet variant.
    pub fn to_rsb(&self) -> RangeSetBlaze<usize> {
        match self {
            AbstractWeight::RangeSet(rsb) => rsb.clone(),
        }
    }

    /// Get the number of ranges in the weight.
    pub fn ranges_len(&self) -> usize {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::ranges_len(rsb),
        }
    }

    /// Iterate over ranges as (start, end) inclusive pairs.
    pub fn iter_ranges(&self) -> Box<dyn Iterator<Item = (usize, usize)> + '_> {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::iter_ranges(rsb),
        }
    }

    /// Iterate over ranges (convenience for RangeSet variant).
    pub fn ranges(&self) -> impl Iterator<Item = std::ops::RangeInclusive<usize>> + '_ {
        match self {
            AbstractWeight::RangeSet(rsb) => rsb.ranges(),
        }
    }

    /// Insert a position.
    pub fn insert(&mut self, pos: usize) {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::insert(rsb, pos),
        }
    }
    
    /// Get the minimum position, if any.
    pub fn min_item(&self) -> Option<usize> {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::min_item(rsb),
        }
    }
    
    /// Get the maximum position, if any.
    pub fn max_item(&self) -> Option<usize> {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::max_item(rsb),
        }
    }
    
    /// Compute set difference (self - other).
    /// 
    /// Panics if self and other are different variants.
    pub fn difference(&self, other: &Self) -> Self {
        let (a, b) = match_variant!(self, other, RangeSet);
        AbstractWeight::RangeSet(WeightBackend::difference(a, b))
    }
    
    /// Clip to positions <= max.
    pub fn clip_max(&mut self, max: usize) {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::clip_max(rsb, max),
        }
    }
}

// ---------------------------------------------------------------------------
// Bitwise Operations - dispatch to backends with variant checking
// ---------------------------------------------------------------------------

impl BitAnd for AbstractWeight {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        let (a, b) = match_variant!(&self, &rhs, RangeSet);
        AbstractWeight::RangeSet(WeightBackend::intersect(a, b))
    }
}

impl BitAnd for &AbstractWeight {
    type Output = AbstractWeight;

    fn bitand(self, rhs: Self) -> Self::Output {
        let (a, b) = match_variant!(self, rhs, RangeSet);
        AbstractWeight::RangeSet(WeightBackend::intersect(a, b))
    }
}

impl BitAndAssign for AbstractWeight {
    fn bitand_assign(&mut self, rhs: Self) {
        match (self, &rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                WeightBackend::intersect_assign(a, b);
            }
        }
    }
}

impl BitOr for AbstractWeight {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        let (a, b) = match_variant!(&self, &rhs, RangeSet);
        AbstractWeight::RangeSet(WeightBackend::union(a, b))
    }
}

impl BitOr for &AbstractWeight {
    type Output = AbstractWeight;

    fn bitor(self, rhs: Self) -> Self::Output {
        let (a, b) = match_variant!(self, rhs, RangeSet);
        AbstractWeight::RangeSet(WeightBackend::union(a, b))
    }
}

impl BitOrAssign for AbstractWeight {
    fn bitor_assign(&mut self, rhs: Self) {
        match (self, &rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                WeightBackend::union_assign(a, b);
            }
        }
    }
}

impl BitOrAssign<&AbstractWeight> for AbstractWeight {
    fn bitor_assign(&mut self, rhs: &AbstractWeight) {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                WeightBackend::union_assign(a, b);
            }
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
        }
    }
}

// ---------------------------------------------------------------------------
// Conversion Traits
// ---------------------------------------------------------------------------

impl From<RangeSetBlaze<usize>> for AbstractWeight {
    fn from(rsb: RangeSetBlaze<usize>) -> Self {
        AbstractWeight::RangeSet(rsb)
    }
}

impl From<&RangeSetBlaze<usize>> for AbstractWeight {
    fn from(rsb: &RangeSetBlaze<usize>) -> Self {
        AbstractWeight::RangeSet(rsb.clone())
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
}
