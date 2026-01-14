//! Factored Weight representation for weight-heavy DWAs.
//!
//! A FactoredWeight represents a weight as a union of 2D profiles:
//!   Weight = OR( TokenSet_i × TsidSet_i )
//!
//! This achieves ~10x compression compared to 1D RangeSet representation
//! by exploiting the 2D structure of the N×M weight space.
//!
//! Position encoding: `pos = llm_token * num_tsids + tsid`
//!
//! Operations:
//! - AND: Pairwise intersection of terms, filter empty results
//! - OR:  Concatenate terms, optionally merge same-profile terms
//! - contains: Check if any term contains the (token, tsid) pair

use range_set_blaze::RangeSetBlaze;

/// A weight represented as a union of 2D profiles.
/// Each term is a Cartesian product: TokenSet × TsidSet.
#[derive(Clone, Debug)]
pub struct FactoredWeight {
    /// List of (TokenRangeSet, TsidRangeSet) pairs.
    /// The weight is the union of all these Cartesian products.
    pub terms: Vec<(RangeSetBlaze<u16>, RangeSetBlaze<u16>)>,
    /// Number of tokenizer states (M in the N×M expansion)
    pub num_tsids: u16,
}

impl FactoredWeight {
    /// Create a new factored weight with the given terms.
    pub fn new(terms: Vec<(RangeSetBlaze<u16>, RangeSetBlaze<u16>)>, num_tsids: u16) -> Self {
        Self { terms, num_tsids }
    }
    
    /// Create an empty factored weight.
    pub fn empty(num_tsids: u16) -> Self {
        Self { terms: Vec::new(), num_tsids }
    }
    
    /// Create a factored weight from a single Cartesian product.
    pub fn from_product(tokens: RangeSetBlaze<u16>, tsids: RangeSetBlaze<u16>, num_tsids: u16) -> Self {
        if tokens.is_empty() || tsids.is_empty() {
            return Self::empty(num_tsids);
        }
        Self { terms: vec![(tokens, tsids)], num_tsids }
    }
    
    /// Check if this weight is empty.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty() || self.terms.iter().all(|(t, s)| t.is_empty() || s.is_empty())
    }
    
    /// Check if a (token, tsid) pair is contained in this weight.
    pub fn contains(&self, token: u16, tsid: u16) -> bool {
        self.terms.iter().any(|(tok_set, tsid_set)| {
            tok_set.contains(token) && tsid_set.contains(tsid)
        })
    }
    
    /// Compute the union of two factored weights.
    /// Simply concatenates the terms and optionally merges same-profile terms.
    pub fn union(&self, other: &FactoredWeight) -> FactoredWeight {
        debug_assert_eq!(self.num_tsids, other.num_tsids);
        
        let mut terms = self.terms.clone();
        terms.extend(other.terms.iter().cloned());
        
        // Optional: merge terms with identical TsidSets
        let merged = merge_same_profile_terms(terms);
        
        FactoredWeight { terms: merged, num_tsids: self.num_tsids }
    }
    
    /// Compute the intersection of two factored weights.
    /// Returns a new weight containing all (token, tsid) pairs in both weights.
    pub fn intersection(&self, other: &FactoredWeight) -> FactoredWeight {
        debug_assert_eq!(self.num_tsids, other.num_tsids);
        
        let mut result = Vec::new();
        for (tok_a, tsid_a) in &self.terms {
            for (tok_b, tsid_b) in &other.terms {
                let tok_inter = tok_a & tok_b;
                let tsid_inter = tsid_a & tsid_b;
                if !tok_inter.is_empty() && !tsid_inter.is_empty() {
                    result.push((tok_inter, tsid_inter));
                }
            }
        }
        
        FactoredWeight { terms: result, num_tsids: self.num_tsids }
    }
    
    /// Convert from 1D ranges (N×M space) to factored representation.
    /// 
    /// This decomposes each 1D range into 2D rectangles and groups them
    /// by TSID profile for compact storage.
    pub fn from_1d_ranges(ranges: impl Iterator<Item = (usize, usize)>, num_tsids: usize) -> Self {
        use std::collections::HashMap;
        
        let num_tsids_u16 = num_tsids as u16;
        
        // Group by (tsid_lo, tsid_hi) profile
        let mut profile_map: HashMap<(u16, u16), RangeSetBlaze<u16>> = HashMap::new();
        
        for (start, end) in ranges {
            // Decompose to 2D rectangles
            let tok_s = (start / num_tsids) as u16;
            let tsid_s = (start % num_tsids) as u16;
            let tok_e = (end / num_tsids) as u16;
            let tsid_e = (end % num_tsids) as u16;
            
            // Generate rectangles for this range
            let rects: Vec<(u16, u16, u16, u16)> = decompose_range_to_rects(
                tok_s, tsid_s, tok_e, tsid_e, num_tsids_u16
            );
            
            for (t1, t2, s1, s2) in rects {
                profile_map.entry((s1, s2))
                    .or_insert_with(RangeSetBlaze::new)
                    .ranges_insert(t1..=t2);
            }
        }
        
        // Convert profile map to terms
        let terms: Vec<_> = profile_map.into_iter()
            .map(|((tsid_lo, tsid_hi), tok_set)| {
                let mut tsid_set = RangeSetBlaze::new();
                tsid_set.ranges_insert(tsid_lo..=tsid_hi);
                (tok_set, tsid_set)
            })
            .collect();
        
        FactoredWeight { terms, num_tsids: num_tsids_u16 }
    }
    
    /// Expand this factored weight to the full 1D N×M-space representation.
    /// This is the inverse of from_1d_ranges.
    /// 
    /// Position = token * num_tsids + tsid
    pub fn expand(&self) -> RangeSetBlaze<usize> {
        let mut result = RangeSetBlaze::new();
        let num_tsids = self.num_tsids as usize;
        
        for (tok_set, tsid_set) in &self.terms {
            // For each token range and tsid range, compute expanded positions
            for tok_range in tok_set.ranges() {
                let tok_start = *tok_range.start() as usize;
                let tok_end = *tok_range.end() as usize;
                
                for tsid_range in tsid_set.ranges() {
                    let tsid_start = *tsid_range.start() as usize;
                    let tsid_end = *tsid_range.end() as usize;
                    
                    // Each token in the range maps to a contiguous range of positions
                    for token in tok_start..=tok_end {
                        let pos_start = token * num_tsids + tsid_start;
                        let pos_end = token * num_tsids + tsid_end;
                        result.ranges_insert(pos_start..=pos_end);
                    }
                }
            }
        }
        
        result
    }
    
    /// Count the total number of terms.
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }
    
    /// Count total ranges across all terms (tokens + tsids).
    pub fn total_ranges(&self) -> usize {
        self.terms.iter()
            .map(|(t, s)| t.ranges().count() + s.ranges().count())
            .sum()
    }
    
    /// Estimate storage size in bytes.
    /// Each range is (u16, u16) = 4 bytes per range.
    pub fn estimated_storage_bytes(&self) -> usize {
        self.total_ranges() * 4
    }
}

/// Decompose a 1D range into 2D rectangles.
/// 
/// A 1D range [start, end] in N×M space decomposes into up to 3 rectangles:
/// 1. Partial first row: [tok_s, tok_s] × [tsid_s, M-1]  (if not starting at tsid 0)
/// 2. Full middle rows: [tok_s+1, tok_e-1] × [0, M-1]   (if there are full rows)
/// 3. Partial last row:  [tok_e, tok_e] × [0, tsid_e]    (if not ending at tsid M-1)
fn decompose_range_to_rects(tok_s: u16, tsid_s: u16, tok_e: u16, tsid_e: u16, num_tsids: u16) -> Vec<(u16, u16, u16, u16)> {
    let max_tsid = num_tsids - 1;
    
    if tok_s == tok_e {
        // Single row: just one rectangle
        return vec![(tok_s, tok_s, tsid_s, tsid_e)];
    }
    
    let mut rects = Vec::new();
    
    // Partial first row (if not starting at tsid 0)
    if tsid_s > 0 {
        rects.push((tok_s, tok_s, tsid_s, max_tsid));
    } else {
        // First row is complete, include it in full rows
    }
    
    // Full middle rows
    let first_full_row = if tsid_s > 0 { tok_s + 1 } else { tok_s };
    let last_full_row = if tsid_e < max_tsid { tok_e.saturating_sub(1) } else { tok_e };
    
    if first_full_row <= last_full_row {
        rects.push((first_full_row, last_full_row, 0, max_tsid));
    }
    
    // Partial last row (if not ending at max tsid and not same as first row)
    if tsid_e < max_tsid && tok_e > tok_s {
        rects.push((tok_e, tok_e, 0, tsid_e));
    }
    
    rects
}

/// Merge terms that have identical TSID profiles by unioning their token sets.
fn merge_same_profile_terms(terms: Vec<(RangeSetBlaze<u16>, RangeSetBlaze<u16>)>) -> Vec<(RangeSetBlaze<u16>, RangeSetBlaze<u16>)> {
    use std::collections::HashMap;
    
    // Group by serialized tsid_set (since RangeSetBlaze doesn't impl Hash/Eq directly)
    let mut groups: HashMap<Vec<(u16, u16)>, RangeSetBlaze<u16>> = HashMap::new();
    let mut tsid_sets: HashMap<Vec<(u16, u16)>, RangeSetBlaze<u16>> = HashMap::new();
    
    for (tok_set, tsid_set) in terms {
        let key: Vec<(u16, u16)> = tsid_set.ranges().map(|r| (*r.start(), *r.end())).collect();
        groups.entry(key.clone())
            .and_modify(|existing| *existing |= &tok_set)
            .or_insert_with(|| tok_set.clone());
        tsid_sets.entry(key).or_insert(tsid_set);
    }
    
    groups.into_iter()
        .map(|(key, tok_set)| (tok_set, tsid_sets.remove(&key).unwrap()))
        .collect()
}

impl std::ops::BitAnd for &FactoredWeight {
    type Output = FactoredWeight;
    
    fn bitand(self, rhs: Self) -> Self::Output {
        self.intersection(rhs)
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
    fn test_factored_weight_contains() {
        let num_tsids = 4;
        
        // Create a weight: tokens {0, 1, 2}, tsids {1, 3}
        let mut tokens = RangeSetBlaze::new();
        tokens.ranges_insert(0..=2);
        let mut tsids = RangeSetBlaze::new();
        tsids.insert(1);
        tsids.insert(3);
        
        let fw = FactoredWeight::from_product(tokens, tsids, num_tsids);
        
        // Should contain (0,1), (0,3), (1,1), (1,3), (2,1), (2,3)
        assert!(fw.contains(0, 1));
        assert!(fw.contains(0, 3));
        assert!(fw.contains(1, 1));
        assert!(fw.contains(2, 3));
        
        // Should NOT contain
        assert!(!fw.contains(0, 0));
        assert!(!fw.contains(0, 2));
        assert!(!fw.contains(3, 1));
    }
    
    #[test]
    fn test_factored_weight_expand() {
        let num_tsids: u16 = 4;
        
        // Create a weight: tokens {0, 1}, tsids {1, 2}
        let mut tokens = RangeSetBlaze::new();
        tokens.ranges_insert(0..=1);
        let mut tsids = RangeSetBlaze::new();
        tsids.ranges_insert(1..=2);
        
        let fw = FactoredWeight::from_product(tokens, tsids, num_tsids);
        
        // Expected positions: 
        // (0,1) = 0*4+1 = 1
        // (0,2) = 0*4+2 = 2
        // (1,1) = 1*4+1 = 5
        // (1,2) = 1*4+2 = 6
        let expanded = fw.expand();
        
        assert!(expanded.contains(1));
        assert!(expanded.contains(2));
        assert!(expanded.contains(5));
        assert!(expanded.contains(6));
        
        assert!(!expanded.contains(0));
        assert!(!expanded.contains(3));
        assert!(!expanded.contains(4));
    }
    
    #[test]
    fn test_from_1d_ranges_single_row() {
        let num_tsids = 4;
        
        // Range [5, 7] in 1D = token 1, tsids 1-3
        let fw = FactoredWeight::from_1d_ranges([(5, 7)].iter().copied(), num_tsids);
        
        assert!(fw.contains(1, 1));
        assert!(fw.contains(1, 2));
        assert!(fw.contains(1, 3));
        assert!(!fw.contains(1, 0));
        assert!(!fw.contains(0, 1));
    }
    
    #[test]
    fn test_from_1d_ranges_multi_row() {
        let num_tsids = 4;
        
        // Range [2, 9] in 1D:
        // - token 0: tsids 2-3 (positions 2-3)
        // - token 1: tsids 0-3 (positions 4-7) - full row
        // - token 2: tsids 0-1 (positions 8-9)
        let fw = FactoredWeight::from_1d_ranges([(2, 9)].iter().copied(), num_tsids);
        
        // Verify by expanding back
        let expanded = fw.expand();
        for pos in 2..=9 {
            assert!(expanded.contains(pos), "Position {} should be in expanded", pos);
        }
        assert!(!expanded.contains(0));
        assert!(!expanded.contains(1));
        assert!(!expanded.contains(10));
    }
    
    #[test]
    fn test_intersection() {
        let num_tsids: u16 = 4;
        
        // fw1: tokens {0..5}, tsids {0,1}
        let mut t1 = RangeSetBlaze::new();
        t1.ranges_insert(0..=5);
        let mut s1 = RangeSetBlaze::new();
        s1.ranges_insert(0..=1);
        let fw1 = FactoredWeight::from_product(t1, s1, num_tsids);
        
        // fw2: tokens {3..10}, tsids {1,2}
        let mut t2 = RangeSetBlaze::new();
        t2.ranges_insert(3..=10);
        let mut s2 = RangeSetBlaze::new();
        s2.ranges_insert(1..=2);
        let fw2 = FactoredWeight::from_product(t2, s2, num_tsids);
        
        let result = &fw1 & &fw2;
        
        // Intersection should be: tokens {3..5}, tsids {1}
        assert!(result.contains(3, 1));
        assert!(result.contains(4, 1));
        assert!(result.contains(5, 1));
        
        // Should not contain
        assert!(!result.contains(2, 1)); // token 2 not in intersection
        assert!(!result.contains(3, 0)); // tsid 0 not in intersection
        assert!(!result.contains(3, 2)); // tsid 2 not in fw1
    }
    
    #[test]
    fn test_union_same_profile() {
        let num_tsids: u16 = 4;
        
        // Two weights with same tsid profile should merge nicely
        let mut t1 = RangeSetBlaze::new();
        t1.ranges_insert(0..=5);
        let mut s1 = RangeSetBlaze::new();
        s1.ranges_insert(0..=1);
        let fw1 = FactoredWeight::from_product(t1, s1.clone(), num_tsids);
        
        let mut t2 = RangeSetBlaze::new();
        t2.ranges_insert(10..=15);
        let fw2 = FactoredWeight::from_product(t2, s1, num_tsids);
        
        let result = &fw1 | &fw2;
        
        // Should have merged into one term
        assert_eq!(result.num_terms(), 1);
        
        // Should contain all expected pairs
        assert!(result.contains(0, 0));
        assert!(result.contains(5, 1));
        assert!(result.contains(10, 0));
        assert!(result.contains(15, 1));
    }
    
    #[test]
    fn test_roundtrip() {
        let num_tsids = 100;
        
        // Create some 1D ranges
        let ranges = vec![
            (0, 99),     // Full first row
            (100, 299),  // Two full rows
            (350, 375),  // Partial row
        ];
        
        let fw = FactoredWeight::from_1d_ranges(ranges.clone().into_iter(), num_tsids);
        let expanded = fw.expand();
        
        // Verify all original positions are present
        for (start, end) in &ranges {
            for pos in *start..=*end {
                assert!(expanded.contains(pos), "Position {} should be in expanded", pos);
            }
        }
        
        // Verify no extra positions
        // Check some positions that should NOT be present
        assert!(!expanded.contains(300)); // Between range 2 and 3
        assert!(!expanded.contains(376)); // After range 3
    }
}
