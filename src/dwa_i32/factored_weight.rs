//! Factored Weight representation for weight-heavy DWAs.
//!
//! Instead of expanding weights from N-space to N×M-space (which causes O(N×M) ranges),
//! we store weights as `(base_weight, tsid_mask)` pairs. This allows:
//! - Compact storage: O(N) + O(M) instead of O(N×M)
//! - Fast AND/OR: Operate on base and mask separately
//! - Lazy expansion: Only expand when needed for queries
//!
//! Semantics:
//! - A factored weight `(base, mask)` represents the set:
//!   `{ llm_token * num_tsids + tsid : llm_token ∈ base, tsid ∈ mask }`
//!
//! Operations:
//! - AND: `(b1, m1) & (b2, m2) = (b1 & b2, m1 & m2)`
//! - OR:  `(b1, m1) | (b2, m2)` = requires normalization or expansion
//!        (OR is only exact if masks are identical)

use super::rangeset::RangeSet;
use std::sync::Arc;

/// A factored weight that represents a set of (llm_token, tsid) pairs compactly.
#[derive(Clone, Debug)]
pub struct FactoredWeight {
    /// The base weight in N-space (LLM token IDs)
    pub base: RangeSet,
    /// The tsid mask (which tokenizer states this weight applies to)
    pub tsid_mask: RangeSet,
    /// Number of tokenizer states (M in the N×M expansion)
    pub num_tsids: usize,
}

impl FactoredWeight {
    /// Create a new factored weight.
    pub fn new(base: RangeSet, tsid_mask: RangeSet, num_tsids: usize) -> Self {
        Self { base, tsid_mask, num_tsids }
    }
    
    /// Create a factored weight that covers all LLM tokens for specific tsids.
    pub fn all_tokens_for_tsids(tsid_mask: RangeSet, num_tsids: usize) -> Self {
        Self {
            base: RangeSet::all(),
            tsid_mask,
            num_tsids,
        }
    }
    
    /// Create a factored weight that covers specific LLM tokens for all tsids.
    pub fn tokens_for_all_tsids(base: RangeSet, num_tsids: usize) -> Self {
        Self {
            base,
            tsid_mask: RangeSet::ones(num_tsids),
            num_tsids,
        }
    }
    
    /// Create a factored weight covering all (token, tsid) pairs.
    pub fn all(num_tsids: usize) -> Self {
        Self {
            base: RangeSet::all(),
            tsid_mask: RangeSet::ones(num_tsids),
            num_tsids,
        }
    }
    
    /// Create an empty factored weight.
    pub fn empty(num_tsids: usize) -> Self {
        Self {
            base: RangeSet::zeros(),
            tsid_mask: RangeSet::zeros(),
            num_tsids,
        }
    }
    
    /// Check if this weight is empty.
    pub fn is_empty(&self) -> bool {
        self.base.is_empty() || self.tsid_mask.is_empty()
    }
    
    /// Compute the intersection of two factored weights.
    /// This is exact: (b1 & b2, m1 & m2).
    pub fn intersect(&self, other: &FactoredWeight) -> FactoredWeight {
        debug_assert_eq!(self.num_tsids, other.num_tsids);
        FactoredWeight {
            base: &self.base & &other.base,
            tsid_mask: &self.tsid_mask & &other.tsid_mask,
            num_tsids: self.num_tsids,
        }
    }
    
    /// Compute the union of two factored weights.
    /// 
    /// NOTE: Union is only exact if the masks are identical.
    /// If masks differ, this returns a conservative over-approximation
    /// that may include extra (token, tsid) pairs.
    pub fn union(&self, other: &FactoredWeight) -> FactoredWeight {
        debug_assert_eq!(self.num_tsids, other.num_tsids);
        
        // If masks are identical, union is exact
        if self.tsid_mask == other.tsid_mask {
            return FactoredWeight {
                base: &self.base | &other.base,
                tsid_mask: self.tsid_mask.clone(),
                num_tsids: self.num_tsids,
            };
        }
        
        // If either base covers all tokens, we can be smarter
        if self.base.is_all_fast() {
            return FactoredWeight {
                base: RangeSet::all(),
                tsid_mask: &self.tsid_mask | &other.tsid_mask,
                num_tsids: self.num_tsids,
            };
        }
        if other.base.is_all_fast() {
            return FactoredWeight {
                base: RangeSet::all(),
                tsid_mask: &self.tsid_mask | &other.tsid_mask,
                num_tsids: self.num_tsids,
            };
        }
        
        // General case: over-approximate by taking union of both components
        // This is NOT exact but is safe (never drops valid pairs)
        FactoredWeight {
            base: &self.base | &other.base,
            tsid_mask: &self.tsid_mask | &other.tsid_mask,
            num_tsids: self.num_tsids,
        }
    }
    
    /// Expand this factored weight to the full N×M-space representation.
    /// This is the inverse of factorization and produces the actual RangeSet
    /// that would be stored in the traditional representation.
    /// 
    /// Each (llm_token, tsid) pair becomes position: llm_token * num_tsids + tsid
    pub fn expand(&self) -> RangeSet {
        use range_set_blaze::RangeSetBlaze;
        
        if self.is_empty() {
            return RangeSet::zeros();
        }
        
        if self.base.is_all_fast() && self.tsid_mask.len() == self.num_tsids {
            return RangeSet::all();
        }
        
        let mut result = RangeSetBlaze::new();
        
        // For each range in base
        for base_range in self.base.rsb.ranges() {
            let start = *base_range.start();
            let end = *base_range.end();
            
            // For each tsid range in mask
            for tsid_range in self.tsid_mask.rsb.ranges() {
                let ts = *tsid_range.start();
                let te = *tsid_range.end();
                
                // Add expanded positions
                // Position = token * num_tsids + tsid
                // For a contiguous range of tokens and tsids, we can compute the range
                for token in start..=end {
                    let exp_start = token * self.num_tsids + ts;
                    let exp_end = token * self.num_tsids + te;
                    result.ranges_insert(exp_start..=exp_end);
                }
            }
        }
        
        RangeSet::from_rsb(result)
    }
    
    /// Count the number of ranges in the base weight.
    pub fn base_range_count(&self) -> usize {
        self.base.num_ranges()
    }
    
    /// Count the number of ranges in the tsid mask.
    pub fn mask_range_count(&self) -> usize {
        self.tsid_mask.num_ranges()
    }
    
    /// Estimate the total number of ranges if this weight were expanded.
    /// This is an upper bound: actual count may be lower due to range merging.
    pub fn estimated_expanded_ranges(&self) -> usize {
        self.base.num_ranges() * self.tsid_mask.num_ranges()
    }
}

impl std::ops::BitAnd for &FactoredWeight {
    type Output = FactoredWeight;
    
    fn bitand(self, rhs: Self) -> Self::Output {
        self.intersect(rhs)
    }
}

impl std::ops::BitOr for &FactoredWeight {
    type Output = FactoredWeight;
    
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_factored_weight_basic() {
        let num_tsids = 4;
        
        // Create a weight: tokens {0, 1, 2}, tsids {1, 3}
        let base = RangeSet::from_ranges(&[(0, 2)]);
        let mask = RangeSet::from_ranges(&[(1, 1), (3, 3)]);
        let fw = FactoredWeight::new(base, mask, num_tsids);
        
        assert!(!fw.is_empty());
        assert_eq!(fw.base_range_count(), 1);
        assert_eq!(fw.mask_range_count(), 2);
        
        // Expand and check positions
        // Expected: (0*4+1), (0*4+3), (1*4+1), (1*4+3), (2*4+1), (2*4+3)
        //         = 1, 3, 5, 7, 9, 11
        let expanded = fw.expand();
        assert!(expanded.contains(1));
        assert!(expanded.contains(3));
        assert!(expanded.contains(5));
        assert!(expanded.contains(7));
        assert!(expanded.contains(9));
        assert!(expanded.contains(11));
        assert!(!expanded.contains(0));
        assert!(!expanded.contains(2));
        assert!(!expanded.contains(4));
    }
    
    #[test]
    fn test_factored_weight_intersection() {
        let num_tsids = 4;
        
        let fw1 = FactoredWeight::new(
            RangeSet::from_ranges(&[(0, 5)]),
            RangeSet::from_ranges(&[(0, 1)]),
            num_tsids,
        );
        
        let fw2 = FactoredWeight::new(
            RangeSet::from_ranges(&[(3, 10)]),
            RangeSet::from_ranges(&[(1, 2)]),
            num_tsids,
        );
        
        let result = &fw1 & &fw2;
        
        // Base: {0..5} & {3..10} = {3..5}
        // Mask: {0,1} & {1,2} = {1}
        assert!(result.base.contains(3));
        assert!(result.base.contains(4));
        assert!(result.base.contains(5));
        assert!(!result.base.contains(2));
        assert!(!result.base.contains(6));
        
        assert!(result.tsid_mask.contains(1));
        assert!(!result.tsid_mask.contains(0));
        assert!(!result.tsid_mask.contains(2));
    }
}
