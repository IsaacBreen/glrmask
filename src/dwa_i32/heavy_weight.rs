//! Heavy Weight: A dimension-aware weight representation for 2D (token × tsid) space.
//!
//! This module provides `HeavyWeight`, a wrapper around `RangeSet` that tracks
//! the dimensions of the underlying N×M space. This enables experimentation with
//! factored representations that require knowing both dimensions.
//!
//! ## Encoding
//!
//! A position in the 2D space is encoded as:
//! ```text
//! pos = token * num_tsids + tsid
//! ```
//!
//! Where:
//! - `token` is the LLM token ID (0 to num_tokens-1)
//! - `tsid` is the tokenizer state ID (0 to num_tsids-1)
//! - `num_tsids` is M (the second dimension)
//! - `num_tokens` is N (the first dimension)
//!
//! ## Usage
//!
//! ```ignore
//! let dims = WeightDimensions::new(4096, 4476);  // 4096 tokens × 4476 tsids
//! let weight = HeavyWeight::from_item(pos, &dims);
//! let expanded = weight.to_rangeset();
//! ```
//!
//! ## Factored Representation
//!
//! HeavyWeight supports conversion to/from `FactoredWeight`, a compact representation
//! that stores weights as union of `(TokenRangeSet, TsidRangeSet)` Cartesian products.
//! This can achieve ~5-10x compression for weights with structured 2D patterns.

use super::factored_weight::FactoredWeight;
use super::rangeset::RangeSet;
use range_set_blaze::RangeSetBlaze;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display, Formatter};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};
use std::sync::Arc;

/// Dimensions for the 2D weight space (token × tsid).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WeightDimensions {
    /// Number of LLM tokens (N dimension)
    pub num_tokens: usize,
    /// Number of tokenizer states (M dimension)
    pub num_tsids: usize,
}

impl Default for WeightDimensions {
    /// Default returns TEST dimensions (1000, 100).
    /// This provides sensible defaults for tests and legacy code.
    fn default() -> Self {
        Self::TEST
    }
}

impl WeightDimensions {
    /// Test/fallback dimensions (1000 tokens × 100 tsids).
    /// Use this for tests or when dimensions aren't critical.
    pub const TEST: Self = Self { num_tokens: 1000, num_tsids: 100 };
    
    /// Create new weight dimensions.
    pub fn new(num_tokens: usize, num_tsids: usize) -> Self {
        assert!(num_tokens > 0, "num_tokens must be positive");
        assert!(num_tsids > 0, "num_tsids must be positive");
        Self { num_tokens, num_tsids }
    }
    
    /// Total size of the 2D space (N × M).
    pub fn total_size(&self) -> usize {
        self.num_tokens.checked_mul(self.num_tsids)
            .expect("dimension overflow")
    }
    
    /// Maximum valid position in the encoded space.
    pub fn max_pos(&self) -> usize {
        self.total_size().saturating_sub(1)
    }
    
    /// Encode a (token, tsid) pair as a 1D position.
    #[inline]
    pub fn encode(&self, token: usize, tsid: usize) -> usize {
        debug_assert!(token < self.num_tokens, "token {} out of range (max {})", token, self.num_tokens);
        debug_assert!(tsid < self.num_tsids, "tsid {} out of range (max {})", tsid, self.num_tsids);
        token * self.num_tsids + tsid
    }
    
    /// Decode a 1D position to (token, tsid).
    #[inline]
    pub fn decode(&self, pos: usize) -> (usize, usize) {
        let token = pos / self.num_tsids;
        let tsid = pos % self.num_tsids;
        (token, tsid)
    }
    
    /// Encode a 2D rectangle as 1D ranges.
    /// 
    /// Returns an iterator of (start, end) pairs representing the 1D encoding
    /// of all positions where token ∈ [t1, t2] and tsid ∈ [s1, s2].
    pub fn encode_rect(&self, t1: usize, t2: usize, s1: usize, s2: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        // A rectangle in 2D becomes multiple ranges in 1D
        // Each row is a contiguous range
        (t1..=t2).map(move |token| {
            let start = token * self.num_tsids + s1;
            let end = token * self.num_tsids + s2;
            (start, end)
        })
    }
    
    /// Check if this dimension matches another.
    pub fn matches(&self, other: &WeightDimensions) -> bool {
        self.num_tokens == other.num_tokens && self.num_tsids == other.num_tsids
    }
}

impl Display for WeightDimensions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}×{}", self.num_tokens, self.num_tsids)
    }
}

/// A dimension-aware weight that wraps RangeSet with explicit dimension tracking.
///
/// This enables experiments with factored representations that need to know
/// both the token and tsid dimensions to decompose weights properly.
#[derive(Clone)]
pub struct HeavyWeight {
    /// The underlying range set
    inner: RangeSet,
    /// Dimensions of the weight space
    dims: WeightDimensions,
}

impl HeavyWeight {
    // ========== Constructors ==========
    
    /// Create an empty weight.
    pub fn zeros(dims: WeightDimensions) -> Self {
        Self {
            inner: RangeSet::zeros(),
            dims,
        }
    }
    
    /// Create a weight containing all valid positions.
    pub fn all(dims: WeightDimensions) -> Self {
        // Create a RangeSet containing [0, max_pos]
        let inner = RangeSet::from_iter(0..=dims.max_pos());
        Self { inner, dims }
    }
    
    /// Create a weight containing a single position.
    pub fn from_item(pos: usize, dims: WeightDimensions) -> Self {
        debug_assert!(pos <= dims.max_pos(), "position out of bounds");
        Self {
            inner: RangeSet::from_item(pos),
            dims,
        }
    }
    
    /// Create a weight from a single 2D point (token, tsid).
    pub fn from_point(token: usize, tsid: usize, dims: WeightDimensions) -> Self {
        Self::from_item(dims.encode(token, tsid), dims)
    }
    
    /// Create a weight from a 1D range [start, end].
    pub fn from_range(start: usize, end: usize, dims: WeightDimensions) -> Self {
        debug_assert!(end <= dims.max_pos(), "range end out of bounds");
        Self {
            inner: RangeSet::from_ranges(&[(start, end)]),
            dims,
        }
    }
    
    /// Create a weight from multiple 1D ranges.
    pub fn from_ranges(ranges: impl IntoIterator<Item = (usize, usize)>, dims: WeightDimensions) -> Self {
        let mut rsb = RangeSetBlaze::new();
        for (start, end) in ranges {
            debug_assert!(end <= dims.max_pos(), "range end out of bounds");
            rsb.ranges_insert(start..=end);
        }
        Self {
            inner: RangeSet::from_rsb(rsb),
            dims,
        }
    }
    
    /// Create a weight from an existing RangeSet.
    /// 
    /// The RangeSet is assumed to already be in valid bounds for the given dimensions.
    /// Use `from_rangeset_clamped` if you need to clip to valid range.
    pub fn from_rangeset(inner: RangeSet, dims: WeightDimensions) -> Self {
        Self { inner, dims }
    }
    
    /// Create a weight from a RangeSet, clamping to valid bounds.
    pub fn from_rangeset_clamped(mut inner: RangeSet, dims: WeightDimensions) -> Self {
        inner.clip_max(dims.max_pos());
        Self { inner, dims }
    }
    
    /// Create a weight representing a 2D rectangle [t1, t2] × [s1, s2].
    pub fn from_rect(t1: usize, t2: usize, s1: usize, s2: usize, dims: WeightDimensions) -> Self {
        Self::from_ranges(dims.encode_rect(t1, t2, s1, s2), dims)
    }
    
    // ========== Accessors ==========
    
    /// Get the underlying RangeSet.
    pub fn as_rangeset(&self) -> &RangeSet {
        &self.inner
    }
    
    /// Get the underlying RangeSet, consuming self.
    pub fn into_rangeset(self) -> RangeSet {
        self.inner
    }
    
    /// Get the weight dimensions.
    pub fn dimensions(&self) -> WeightDimensions {
        self.dims
    }
    
    /// Get the number of tsids (M dimension).
    pub fn num_tsids(&self) -> usize {
        self.dims.num_tsids
    }
    
    /// Get the number of tokens (N dimension).
    pub fn num_tokens(&self) -> usize {
        self.dims.num_tokens
    }
    
    // ========== Properties ==========
    
    /// Check if the weight is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    
    /// Check if the weight contains all valid positions.
    pub fn is_all(&self) -> bool {
        // Check if it contains [0, max_pos]
        let min = self.inner.min_item();
        let max = self.inner.max_item();
        if let (Some(min), Some(max)) = (min, max) {
            min == 0 && max >= self.dims.max_pos() && self.inner.len() == self.dims.total_size()
        } else {
            false
        }
    }
    
    /// Get the number of positions in the weight.
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    
    
    /// Get the minimum and maximum positions, if any.
    pub fn bounds(&self) -> Option<(usize, usize)> {
        let min = self.inner.min_item()?;
        let max = self.inner.max_item()?;
        Some((min, max))
    }
    
    // ========== Membership Tests ==========
    
    /// Check if a position is in the weight.
    pub fn contains(&self, pos: usize) -> bool {
        self.inner.contains(pos)
    }
    
    /// Check if a (token, tsid) point is in the weight.
    pub fn contains_point(&self, token: usize, tsid: usize) -> bool {
        if token >= self.dims.num_tokens || tsid >= self.dims.num_tsids {
            return false;
        }
        self.inner.contains(self.dims.encode(token, tsid))
    }
    
    // ========== Set Operations ==========
    
    /// Union of two weights (OR).
    pub fn union(&self, other: &Self) -> Self {
        self.check_dims(other);
        Self {
            inner: &self.inner | &other.inner,
            dims: self.dims,
        }
    }
    
    /// Intersection of two weights (AND).
    pub fn intersection(&self, other: &Self) -> Self {
        self.check_dims(other);
        Self {
            inner: &self.inner & &other.inner,
            dims: self.dims,
        }
    }
    
    /// Complement within the valid domain.
    pub fn complement(&self) -> Self {
        let all = Self::all(self.dims);
        Self {
            inner: &all.inner - &self.inner,
            dims: self.dims,
        }
    }
    
    /// Set difference (self - other).
    pub fn difference(&self, other: &Self) -> Self {
        self.check_dims(other);
        Self {
            inner: &self.inner - &other.inner,
            dims: self.dims,
        }
    }
    
    /// Clip the weight to a maximum position.
    pub fn clip_max(&mut self, max: usize) {
        self.inner.clip_max(max);
    }
    
    // ========== Iteration ==========
    
    /// Iterate over all positions in the weight (up to a given max).
    /// Note: For large weights, this can be very slow.
    pub fn iter_up_to(&self, max: usize) -> impl Iterator<Item = usize> + '_ {
        self.inner.iter_up_to(max)
    }
    
    /// Iterate over ranges [start, end] in the weight.
    pub fn ranges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.inner.rsb.ranges().map(|r| (*r.start(), *r.end()))
    }
    
    /// Iterate over (token, tsid) pairs in the weight (up to a given max).
    /// Note: For large weights, this can be very slow.
    pub fn iter_points_up_to(&self, max: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let dims = self.dims;
        self.inner.iter_up_to(max).map(move |pos| dims.decode(pos))
    }
    
    // ========== 2D Operations ==========
    
    /// Project the weight to token space (first dimension).
    /// Returns a RangeSetBlaze of unique token values.
    pub fn project_tokens(&self) -> RangeSetBlaze<usize> {
        let mut result = RangeSetBlaze::new();
        for (start, end) in self.ranges() {
            let t_start = start / self.dims.num_tsids;
            let t_end = end / self.dims.num_tsids;
            result.ranges_insert(t_start..=t_end);
        }
        result
    }
    
    /// Project the weight to tsid space (second dimension).
    /// Returns a RangeSetBlaze of unique tsid values.
    /// 
    /// Note: This is a union of all tsid values across all tokens,
    /// not the tsid values for any specific token.
    pub fn project_tsids(&self) -> RangeSetBlaze<usize> {
        let mut result = RangeSetBlaze::new();
        let m = self.dims.num_tsids;
        for (start, end) in self.ranges() {
            // For each range, compute which tsids are covered
            let t_start = start / m;
            let t_end = end / m;
            let s_start = start % m;
            let s_end = end % m;
            
            if t_start == t_end {
                // Single row: just this range of tsids
                result.ranges_insert(s_start..=s_end);
            } else {
                // Multiple rows: first partial, middle full, last partial
                // First row: s_start to m-1
                result.ranges_insert(s_start..=(m - 1));
                // Middle rows: all tsids (0 to m-1)
                if t_end > t_start + 1 {
                    result.ranges_insert(0..=(m - 1));  // Already covers all
                }
                // Last row: 0 to s_end
                result.ranges_insert(0..=s_end);
            }
        }
        result
    }
    
    // ========== Factored Representation ==========
    
    /// Convert this weight to a factored representation.
    /// 
    /// The factored representation stores the weight as a union of
    /// `(TokenRangeSet, TsidRangeSet)` Cartesian products, which can
    /// achieve significant compression when weights have structured
    /// 2D patterns.
    /// 
    /// # Example
    /// 
    /// ```ignore
    /// let hw = HeavyWeight::from_ranges(ranges, dims);
    /// let factored = hw.to_factored();
    /// println!("Compressed: {} terms, {} ranges", 
    ///          factored.num_terms(), factored.total_ranges());
    /// ```
    pub fn to_factored(&self) -> FactoredWeight {
        FactoredWeight::from_1d_ranges(
            self.ranges(),
            self.dims.num_tsids,
        )
    }
    
    /// Create a HeavyWeight from a factored representation.
    /// 
    /// This expands the factored representation back to a flat RangeSet.
    pub fn from_factored(factored: FactoredWeight, dims: WeightDimensions) -> Self {
        debug_assert_eq!(
            factored.num_tsids as usize, dims.num_tsids,
            "num_tsids mismatch: factored has {}, dims has {}",
            factored.num_tsids, dims.num_tsids
        );
        let rsb = factored.expand();
        Self::from_rangeset(RangeSet::from_rsb(rsb), dims)
    }
    
    /// Estimate the storage bytes if this weight were stored as factored.
    /// 
    /// Returns (factored_bytes, rangeset_bytes, num_terms).
    /// 
    /// This is useful for deciding whether factored representation would
    /// save space for this particular weight.
    pub fn factored_storage_estimate(&self) -> (usize, usize, usize) {
        let factored = self.to_factored();
        let factored_bytes = factored.estimated_storage_bytes();
        
        // RangeSet: 2 × usize per range = 16 bytes per range
        let num_ranges: usize = self.inner.rsb.ranges_len();
        let rangeset_bytes = num_ranges * 16;
        
        (factored_bytes, rangeset_bytes, factored.num_terms())
    }
    
    // ========== Internal Helpers ==========
    
    fn check_dims(&self, other: &Self) {
        assert!(
            self.dims.matches(&other.dims),
            "dimension mismatch: {} vs {}",
            self.dims, other.dims
        );
    }
}

// ========== Trait Implementations ==========

impl Debug for HeavyWeight {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeavyWeight")
            .field("dims", &format!("{}", self.dims))
            .field("cardinality", &self.len())
            .finish()
    }
}

impl Display for HeavyWeight {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "HeavyWeight({}, {} items)", 
               self.dims, self.len())
    }
}

impl PartialEq for HeavyWeight {
    fn eq(&self, other: &Self) -> bool {
        self.dims == other.dims && self.inner == other.inner
    }
}

impl Eq for HeavyWeight {}

impl BitOr for &HeavyWeight {
    type Output = HeavyWeight;
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl BitOr for HeavyWeight {
    type Output = HeavyWeight;
    fn bitor(self, rhs: Self) -> Self::Output {
        (&self).union(&rhs)
    }
}

impl BitOrAssign<&HeavyWeight> for HeavyWeight {
    fn bitor_assign(&mut self, rhs: &HeavyWeight) {
        self.check_dims(rhs);
        self.inner |= &rhs.inner;
    }
}

impl BitAnd for &HeavyWeight {
    type Output = HeavyWeight;
    fn bitand(self, rhs: Self) -> Self::Output {
        self.intersection(rhs)
    }
}

impl BitAnd for HeavyWeight {
    type Output = HeavyWeight;
    fn bitand(self, rhs: Self) -> Self::Output {
        (&self).intersection(&rhs)
    }
}

impl BitAndAssign<&HeavyWeight> for HeavyWeight {
    fn bitand_assign(&mut self, rhs: &HeavyWeight) {
        self.check_dims(rhs);
        self.inner &= &rhs.inner;
    }
}

impl Not for &HeavyWeight {
    type Output = HeavyWeight;
    fn not(self) -> Self::Output {
        self.complement()
    }
}

impl Not for HeavyWeight {
    type Output = HeavyWeight;
    fn not(self) -> Self::Output {
        (&self).complement()
    }
}

// ========== DWA Integration ==========

use super::dwa::DWA;
use std::collections::HashMap;
use std::ptr;

impl HeavyWeight {
    /// Extract unique weights from a DWA as HeavyWeights.
    /// 
    /// This is useful for analyzing weights with dimension awareness.
    pub fn extract_unique_from_dwa(dwa: &DWA, dims: WeightDimensions) -> Vec<HeavyWeight> {
        let mut unique: HashMap<usize, RangeSet> = HashMap::new();
        
        for state in &dwa.states.0 {
            if let Some(fw) = &state.final_weight {
                let p = fw.intern_id();
                unique.entry(p).or_insert_with(|| fw.clone().into());
            }
            for w in state.trans_weights.values() {
                let p = w.intern_id();
                unique.entry(p).or_insert_with(|| w.clone().into());
            }
        }
        
        unique.into_values()
            .map(|rs| HeavyWeight::from_rangeset(rs, dims))
            .collect()
    }
    
    /// Compute aggregate statistics for a set of HeavyWeights.
    pub fn aggregate_stats(weights: &[HeavyWeight]) -> WeightStats {
        let total_cardinality: usize = weights.iter().map(|w| w.len()).sum();
        
        WeightStats {
            num_weights: weights.len(),
            total_cardinality,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WeightStats {
    /// Number of unique weights
    pub num_weights: usize,
    /// Total cardinality (sum of all items across weights)
    pub total_cardinality: usize,
}

// ========== Tests ==========

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dimensions() {
        let dims = WeightDimensions::new(4096, 4476);
        assert_eq!(dims.total_size(), 4096 * 4476);
        assert_eq!(dims.max_pos(), 4096 * 4476 - 1);
        
        // Test encode/decode roundtrip
        for (token, tsid) in [(0, 0), (100, 200), (4095, 4475)] {
            let pos = dims.encode(token, tsid);
            let (t2, s2) = dims.decode(pos);
            assert_eq!((t2, s2), (token, tsid));
        }
    }
    
    #[test]
    fn test_zeros_and_all() {
        let dims = WeightDimensions::new(100, 10);
        
        let zeros = HeavyWeight::zeros(dims);
        assert!(zeros.is_empty());
        assert_eq!(zeros.len(), 0);
        
        let all = HeavyWeight::all(dims);
        assert!(all.is_all());
        assert_eq!(all.len(), 100 * 10);
    }
    
    #[test]
    fn test_from_point() {
        let dims = WeightDimensions::new(100, 10);
        let w = HeavyWeight::from_point(5, 3, dims);
        
        assert!(w.contains_point(5, 3));
        assert!(!w.contains_point(5, 4));
        assert!(!w.contains_point(6, 3));
        assert_eq!(w.len(), 1);
    }
    
    #[test]
    fn test_from_rect() {
        let dims = WeightDimensions::new(100, 10);
        let w = HeavyWeight::from_rect(2, 4, 3, 5, dims);  // tokens 2-4, tsids 3-5
        
        // Should contain 3 tokens × 3 tsids = 9 points
        assert_eq!(w.len(), 9);
        
        // Check corners
        assert!(w.contains_point(2, 3));
        assert!(w.contains_point(4, 5));
        
        // Check outside
        assert!(!w.contains_point(1, 3));
        assert!(!w.contains_point(5, 3));
        assert!(!w.contains_point(2, 2));
        assert!(!w.contains_point(2, 6));
    }
    
    #[test]
    fn test_union() {
        let dims = WeightDimensions::new(100, 10);
        let w1 = HeavyWeight::from_rect(0, 1, 0, 2, dims);  // 2×3 = 6
        let w2 = HeavyWeight::from_rect(1, 2, 0, 2, dims);  // 2×3 = 6, overlaps
        
        let u = w1.union(&w2);
        // Overlap is row 1, so 6 + 6 - 3 = 9
        assert_eq!(u.len(), 9);
        
        assert!(u.contains_point(0, 0));
        assert!(u.contains_point(2, 2));
    }
    
    #[test]
    fn test_intersection() {
        let dims = WeightDimensions::new(100, 10);
        let w1 = HeavyWeight::from_rect(0, 2, 0, 3, dims);  // 3×4
        let w2 = HeavyWeight::from_rect(1, 3, 2, 5, dims);  // 3×4
        
        let i = w1.intersection(&w2);
        // Intersection: tokens 1-2 (2), tsids 2-3 (2) = 4
        assert_eq!(i.len(), 4);
        assert!(i.contains_point(1, 2));
        assert!(i.contains_point(2, 3));
        assert!(!i.contains_point(0, 0));
    }
    
    #[test]
    fn test_complement() {
        let dims = WeightDimensions::new(10, 10);  // 100 total
        let w = HeavyWeight::from_rect(0, 0, 0, 0, dims);  // Just one point
        
        let c = w.complement();
        assert_eq!(c.len(), 99);
        assert!(!c.contains_point(0, 0));
        assert!(c.contains_point(0, 1));
    }
    
    #[test]
    fn test_projection() {
        let dims = WeightDimensions::new(100, 10);
        let w = HeavyWeight::from_rect(2, 5, 3, 7, dims);
        
        let tokens = w.project_tokens();
        let tsids = w.project_tsids();
        
        // Should have tokens 2-5
        assert!(tokens.contains(2));
        assert!(tokens.contains(5));
        assert!(!tokens.contains(1));
        assert!(!tokens.contains(6));
        
        // Should have tsids 3-7
        assert!(tsids.contains(3));
        assert!(tsids.contains(7));
        assert!(!tsids.contains(2));
        assert!(!tsids.contains(8));
    }
    
    #[test]
    fn test_rangeset_conversion() {
        let dims = WeightDimensions::new(100, 10);
        let w = HeavyWeight::from_range(15, 25, dims);
        
        // Get rangeset
        let rs = w.as_rangeset();
        assert!(rs.contains(15));
        assert!(rs.contains(25));
        assert!(!rs.contains(14));
        
        // Convert back
        let w2 = HeavyWeight::from_rangeset(rs.clone(), dims);
        assert_eq!(w, w2);
    }
    
    #[test]
    fn test_iter_points() {
        let dims = WeightDimensions::new(100, 10);
        let w = HeavyWeight::from_rect(1, 2, 3, 4, dims);
        
        let points: Vec<_> = w.iter_points_up_to(dims.max_pos()).collect();
        assert_eq!(points.len(), 4);  // 2×2
        assert!(points.contains(&(1, 3)));
        assert!(points.contains(&(1, 4)));
        assert!(points.contains(&(2, 3)));
        assert!(points.contains(&(2, 4)));
    }
    
    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_dimension_mismatch() {
        let dims1 = WeightDimensions::new(100, 10);
        let dims2 = WeightDimensions::new(100, 20);
        
        let w1 = HeavyWeight::zeros(dims1);
        let w2 = HeavyWeight::zeros(dims2);
        
        let _ = w1.union(&w2);  // Should panic
    }
    
    #[test]
    fn test_factored_roundtrip() {
        let dims = WeightDimensions::new(100, 10);
        
        // Create a complex weight with multiple rectangles
        let w1 = HeavyWeight::from_rect(0, 10, 0, 5, dims);
        let w2 = HeavyWeight::from_rect(5, 20, 3, 9, dims);
        let original = w1.union(&w2);
        
        // Convert to factored
        let factored = original.to_factored();
        
        // Check factored contains correct points
        for (start, end) in original.ranges() {
            for pos in start..=end {
                let (token, tsid) = dims.decode(pos);
                assert!(factored.contains(token as u16, tsid as u16),
                    "factored missing ({}, {})", token, tsid);
            }
        }
        
        // Convert back
        let recovered = HeavyWeight::from_factored(factored, dims);
        
        // Should be identical
        assert_eq!(original.len(), recovered.len());
        for (start, end) in original.ranges() {
            for pos in start..=end {
                assert!(recovered.contains(pos), "recovered missing position {}", pos);
            }
        }
    }
    
    #[test]
    fn test_factored_storage_estimate() {
        let dims = WeightDimensions::new(100, 10);
        
        // Create a weight that benefits from factoring (full rows)
        let w = HeavyWeight::from_rect(10, 30, 0, 9, dims);  // 21 tokens × 10 tsids = full rows
        
        let (factored_bytes, rangeset_bytes, num_terms) = w.factored_storage_estimate();
        
        // Full rows should factor into 1 term
        assert_eq!(num_terms, 1, "full rows should produce 1 term");
        
        // Factored should be smaller than rangeset for this pattern
        // 21 full rows = 21 ranges in RangeSet, but 1 term in factored
        assert!(factored_bytes < rangeset_bytes,
            "factored ({}) should be smaller than rangeset ({}) for full rows",
            factored_bytes, rangeset_bytes);
    }
}
