//! Abstract weight type for DWA/NWA operations.
//! 
//! This module provides a unified weight type that can be used across
//! different weight representations. Currently only supports RangeSetBlaze.

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

/// Abstract weight type wrapping RangeSetBlaze.
/// 
/// This enum provides a unified interface for weight operations.
/// Currently only supports RangeSetBlaze, but the enum structure
/// allows for future extensions to other representations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbstractWeight {
    /// Weight represented as a RangeSetBlaze.
    RangeSet(RangeSetBlaze<usize>),
}

impl Default for AbstractWeight {
    fn default() -> Self {
        AbstractWeight::RangeSet(RangeSetBlaze::new())
    }
}

impl AbstractWeight {
    /// Create an empty weight.
    pub fn empty() -> Self {
        AbstractWeight::RangeSet(RangeSetBlaze::new())
    }

    /// Create a weight containing all positions in the given range.
    pub fn all(dims: WeightDimensions) -> Self {
        if dims.domain_size() == 0 {
            return Self::empty();
        }
        AbstractWeight::RangeSet(RangeSetBlaze::from_iter([0..=dims.max_position()]))
    }

    /// Create a weight from a single position.
    pub fn from_position(pos: usize) -> Self {
        AbstractWeight::RangeSet(RangeSetBlaze::from_iter([pos..=pos]))
    }

    /// Create a weight from an iterator of positions.
    pub fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        AbstractWeight::RangeSet(RangeSetBlaze::from_iter(iter))
    }

    /// Create a weight from a RangeSetBlaze.
    pub fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self {
        AbstractWeight::RangeSet(rsb)
    }

    /// Check if the weight is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            AbstractWeight::RangeSet(rsb) => rsb.is_empty(),
        }
    }

    /// Get the number of positions in the weight.
    pub fn len(&self) -> usize {
        match self {
            AbstractWeight::RangeSet(rsb) => rsb.len() as usize,
        }
    }

    /// Check if a position is in the weight.
    pub fn contains(&self, pos: usize) -> bool {
        match self {
            AbstractWeight::RangeSet(rsb) => rsb.contains(pos),
        }
    }

    /// Get the underlying RangeSetBlaze (if applicable).
    pub fn to_rsb(&self) -> RangeSetBlaze<usize> {
        match self {
            AbstractWeight::RangeSet(rsb) => rsb.clone(),
        }
    }

    /// Get the number of ranges in the weight.
    pub fn ranges_len(&self) -> usize {
        match self {
            AbstractWeight::RangeSet(rsb) => rsb.ranges_len(),
        }
    }

    /// Iterate over ranges.
    pub fn ranges(&self) -> impl Iterator<Item = std::ops::RangeInclusive<usize>> + '_ {
        match self {
            AbstractWeight::RangeSet(rsb) => rsb.ranges(),
        }
    }

    /// Insert a position.
    pub fn insert(&mut self, pos: usize) {
        match self {
            AbstractWeight::RangeSet(rsb) => { rsb.insert(pos); },
        }
    }
}

// Implement bitwise operations

impl BitAnd for AbstractWeight {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(&a & &b)
            }
        }
    }
}

impl BitAnd for &AbstractWeight {
    type Output = AbstractWeight;

    fn bitand(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(a & b)
            }
        }
    }
}

impl BitAndAssign for AbstractWeight {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = std::mem::take(self) & rhs;
    }
}

impl BitOr for AbstractWeight {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(&a | &b)
            }
        }
    }
}

impl BitOr for &AbstractWeight {
    type Output = AbstractWeight;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(a | b)
            }
        }
    }
}

impl BitOrAssign for AbstractWeight {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = std::mem::take(self) | rhs;
    }
}

impl BitOrAssign<&AbstractWeight> for AbstractWeight {
    fn bitor_assign(&mut self, rhs: &AbstractWeight) {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                *a |= b;
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
        let all = RangeSetBlaze::from_iter([0..=dims.max_position()]);
        match self {
            AbstractWeight::RangeSet(rsb) => AbstractWeight::RangeSet(&all - rsb),
        }
    }
}

// Conversion traits

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
