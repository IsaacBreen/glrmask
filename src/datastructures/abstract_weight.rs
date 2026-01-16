//! Abstract weight type for DWA/NWA operations.
//! 
//! This module provides a unified weight type that can be used across
//! different weight representations. Currently only supports RangeSet.

use crate::datastructures::hybrid_bitset::RangeSet as HybridRangeSet;
use crate::datastructures::leveled_gss::Merge;
use crate::dwa_i32::rangeset::RangeSet;
use crate::json_serialization::{JSONConvertible, JSONNode};
use once_cell::sync::Lazy;
use range_set_blaze::RangeSetBlaze;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Deref, DerefMut, Not, Sub,
    SubAssign,
};
use std::sync::RwLock;

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
    /// Number of LLM tokens (vocab size). 0 for unset.
    pub num_tokens: usize,
    /// Number of tokenizer states. Symbol-heavy mode uses 1.
    pub num_tsids: usize,
}

impl WeightDimensions {
    /// Create new dimensions.
    pub fn new(num_tokens: usize, num_tsids: usize) -> Self {
        Self { num_tokens, num_tsids: num_tsids.max(1) }
    }

    /// Check if this is weight-heavy mode (has tsid dimension > 1).
    pub fn is_weight_heavy(&self) -> bool {
        self.num_tsids > 1
    }

    /// Total domain size: num_tokens × num_tsids.
    pub fn domain_size(&self) -> usize {
        if self.num_tokens == 0 || self.num_tsids == 0 {
            return 0;
        }
        self.num_tokens
            .checked_mul(self.num_tsids)
            .expect("weight domain size overflow")
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

const DEFAULT_TEST_TOKENS: usize = 4096;

static WEIGHT_DIMS: Lazy<RwLock<WeightDimensions>> = Lazy::new(|| {
    let dims = if cfg!(test) {
        WeightDimensions::new(DEFAULT_TEST_TOKENS, 1)
    } else {
        WeightDimensions::new(0, 1)
    };
    RwLock::new(dims)
});

/// Abstract weight type wrapping RangeSet.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AbstractWeight(RangeSet);

impl Default for AbstractWeight {
    fn default() -> Self {
        Self::zeros()
    }
}

impl AbstractWeight {
    /// Update global weight dimensions.
    pub fn set_weight_dimensions(dims: WeightDimensions) {
        *WEIGHT_DIMS.write().expect("weight dims lock poisoned") = dims;
    }

    /// Read global weight dimensions.
    pub fn weight_dimensions() -> WeightDimensions {
        *WEIGHT_DIMS.read().expect("weight dims lock poisoned")
    }

    /// Create an empty weight.
    pub fn empty() -> Self {
        Self::zeros()
    }

    /// Create an empty weight.
    pub fn zeros() -> Self {
        Self(RangeSet::zeros())
    }

    /// Create a weight containing all positions in a length.
    pub fn ones(len: usize) -> Self {
        Self(RangeSet::ones(len))
    }

    /// Create a weight containing all positions in the configured domain.
    pub fn all() -> Self {
        let size = Self::weight_dimensions().domain_size();
        if size == 0 {
            return Self::zeros();
        }
        Self::ones(size)
    }

    /// Create a weight containing all positions in the given domain.
    pub fn all_for_dims(dims: WeightDimensions) -> Self {
        let size = dims.domain_size();
        if size == 0 {
            return Self::zeros();
        }
        Self::ones(size)
    }

    /// Create a weight from a single position.
    pub fn from_item(pos: usize) -> Self {
        Self(RangeSet::from_item(pos))
    }

    /// Create a weight from a range list.
    pub fn from_ranges(ranges: &[(usize, usize)]) -> Self {
        Self(RangeSet::from_ranges(ranges))
    }

    /// Create a weight from an iterator of positions.
    pub fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        Self(RangeSet::from_iter(iter))
    }

    /// Create a weight from a RangeSetBlaze.
    pub fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self {
        Self(RangeSet::from_rsb(rsb))
    }

    /// Create a weight from an existing RangeSet.
    pub fn from_rangeset(rangeset: RangeSet) -> Self {
        Self(rangeset)
    }

    /// Borrow the underlying RangeSetBlaze.
    pub fn as_rsb(&self) -> &RangeSetBlaze<usize> {
        &self.0.rsb
    }

    /// Clone the underlying RangeSetBlaze.
    pub fn to_rsb(&self) -> RangeSetBlaze<usize> {
        self.0.rsb.clone()
    }

    /// Compute the bounded complement within the configured domain.
    pub fn complement(&self) -> Self {
        &Self::all() - self
    }

    /// Consume into the underlying RangeSet.
    pub fn into_rangeset(self) -> RangeSet {
        self.0
    }
}

impl Deref for AbstractWeight {
    type Target = RangeSet;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for AbstractWeight {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Display for AbstractWeight {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

// Implement bitwise operations

impl BitAnd for AbstractWeight {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        AbstractWeight(&self.0 & &rhs.0)
    }
}

impl BitAnd for &AbstractWeight {
    type Output = AbstractWeight;

    fn bitand(self, rhs: Self) -> Self::Output {
        AbstractWeight(&self.0 & &rhs.0)
    }
}

impl<'a> BitAnd<&'a AbstractWeight> for AbstractWeight {
    type Output = AbstractWeight;

    fn bitand(self, rhs: &'a AbstractWeight) -> Self::Output {
        AbstractWeight(&self.0 & &rhs.0)
    }
}

impl<'a> BitAnd<AbstractWeight> for &'a AbstractWeight {
    type Output = AbstractWeight;

    fn bitand(self, rhs: AbstractWeight) -> Self::Output {
        AbstractWeight(&self.0 & &rhs.0)
    }
}

impl BitAndAssign for AbstractWeight {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= &rhs.0;
    }
}

impl BitAndAssign<&AbstractWeight> for AbstractWeight {
    fn bitand_assign(&mut self, rhs: &AbstractWeight) {
        self.0 &= &rhs.0;
    }
}

impl BitOr for AbstractWeight {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        AbstractWeight(&self.0 | &rhs.0)
    }
}

impl BitOr for &AbstractWeight {
    type Output = AbstractWeight;

    fn bitor(self, rhs: Self) -> Self::Output {
        AbstractWeight(&self.0 | &rhs.0)
    }
}

impl<'a> BitOr<&'a AbstractWeight> for AbstractWeight {
    type Output = AbstractWeight;

    fn bitor(self, rhs: &'a AbstractWeight) -> Self::Output {
        AbstractWeight(&self.0 | &rhs.0)
    }
}

impl<'a> BitOr<AbstractWeight> for &'a AbstractWeight {
    type Output = AbstractWeight;

    fn bitor(self, rhs: AbstractWeight) -> Self::Output {
        AbstractWeight(&self.0 | &rhs.0)
    }
}

impl BitOrAssign for AbstractWeight {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= &rhs.0;
    }
}

impl BitOrAssign<&AbstractWeight> for AbstractWeight {
    fn bitor_assign(&mut self, rhs: &AbstractWeight) {
        self.0 |= &rhs.0;
    }
}

impl BitXor for AbstractWeight {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        AbstractWeight(&self.0 ^ &rhs.0)
    }
}

impl BitXor for &AbstractWeight {
    type Output = AbstractWeight;

    fn bitxor(self, rhs: Self) -> Self::Output {
        AbstractWeight(&self.0 ^ &rhs.0)
    }
}

impl<'a> BitXor<&'a AbstractWeight> for AbstractWeight {
    type Output = AbstractWeight;

    fn bitxor(self, rhs: &'a AbstractWeight) -> Self::Output {
        AbstractWeight(&self.0 ^ &rhs.0)
    }
}

impl<'a> BitXor<AbstractWeight> for &'a AbstractWeight {
    type Output = AbstractWeight;

    fn bitxor(self, rhs: AbstractWeight) -> Self::Output {
        AbstractWeight(&self.0 ^ &rhs.0)
    }
}

impl BitXorAssign for AbstractWeight {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= &rhs.0;
    }
}

impl BitXorAssign<&AbstractWeight> for AbstractWeight {
    fn bitxor_assign(&mut self, rhs: &AbstractWeight) {
        self.0 ^= &rhs.0;
    }
}

impl Sub for AbstractWeight {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        AbstractWeight(&self.0 - &rhs.0)
    }
}

impl Sub for &AbstractWeight {
    type Output = AbstractWeight;

    fn sub(self, rhs: Self) -> Self::Output {
        AbstractWeight(&self.0 - &rhs.0)
    }
}

impl<'a> Sub<&'a AbstractWeight> for AbstractWeight {
    type Output = AbstractWeight;

    fn sub(self, rhs: &'a AbstractWeight) -> Self::Output {
        AbstractWeight(&self.0 - &rhs.0)
    }
}

impl<'a> Sub<AbstractWeight> for &'a AbstractWeight {
    type Output = AbstractWeight;

    fn sub(self, rhs: AbstractWeight) -> Self::Output {
        AbstractWeight(&self.0 - &rhs.0)
    }
}

impl SubAssign for AbstractWeight {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= &rhs.0;
    }
}

impl SubAssign<&AbstractWeight> for AbstractWeight {
    fn sub_assign(&mut self, rhs: &AbstractWeight) {
        self.0 -= &rhs.0;
    }
}

impl Not for AbstractWeight {
    type Output = Self;

    fn not(self) -> Self::Output {
        &AbstractWeight::all() - &self
    }
}

impl Not for &AbstractWeight {
    type Output = AbstractWeight;

    fn not(self) -> Self::Output {
        &AbstractWeight::all() - self
    }
}

impl Merge for AbstractWeight {
    fn merge(&self, other: &Self) -> Self {
        self | other
    }
}

// Conversion traits

impl From<RangeSet> for AbstractWeight {
    fn from(rsb: RangeSet) -> Self {
        AbstractWeight(rsb)
    }
}

impl From<&RangeSet> for AbstractWeight {
    fn from(rsb: &RangeSet) -> Self {
        AbstractWeight(rsb.clone())
    }
}

impl From<RangeSetBlaze<usize>> for AbstractWeight {
    fn from(rsb: RangeSetBlaze<usize>) -> Self {
        AbstractWeight(RangeSet::from_rsb(rsb))
    }
}

impl From<&RangeSetBlaze<usize>> for AbstractWeight {
    fn from(rsb: &RangeSetBlaze<usize>) -> Self {
        AbstractWeight(RangeSet::from_rsb(rsb.clone()))
    }
}

impl From<HybridRangeSet> for AbstractWeight {
    fn from(rsb: HybridRangeSet) -> Self {
        AbstractWeight(RangeSet::from_rsb(rsb.inner.as_ref().clone()))
    }
}

impl From<&HybridRangeSet> for AbstractWeight {
    fn from(rsb: &HybridRangeSet) -> Self {
        AbstractWeight(RangeSet::from_rsb(rsb.inner.as_ref().clone()))
    }
}

impl From<AbstractWeight> for HybridRangeSet {
    fn from(weight: AbstractWeight) -> Self {
        HybridRangeSet::from(weight.0)
    }
}

impl From<&AbstractWeight> for HybridRangeSet {
    fn from(weight: &AbstractWeight) -> Self {
        HybridRangeSet::from(weight.0.clone())
    }
}

impl From<AbstractWeight> for RangeSet {
    fn from(weight: AbstractWeight) -> Self {
        weight.0
    }
}

impl From<&AbstractWeight> for RangeSet {
    fn from(weight: &AbstractWeight) -> Self {
        weight.0.clone()
    }
}

impl JSONConvertible for AbstractWeight {
    fn to_json(&self) -> JSONNode {
        self.0.to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        Ok(AbstractWeight(RangeSet::from_json(node)?))
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
        let prev_dims = AbstractWeight::weight_dimensions();
        AbstractWeight::set_weight_dimensions(dims);
        let w = AbstractWeight::all();
        assert_eq!(w.len(), 15);
        assert!(w.contains(0));
        assert!(w.contains(14));
        AbstractWeight::set_weight_dimensions(prev_dims);
    }
}
