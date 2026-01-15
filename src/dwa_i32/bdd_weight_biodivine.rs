//! Per-Weight BDD Storage using biodivine_lib_bdd.
//!
//! This module implements a compact BDD representation for individual weights
//! using the `biodivine_lib_bdd` crate for efficient BDD operations.
//!
//! ## Key Features
//!
//! - **TSID-First Ordering**: TSID bits first (MSB to LSB), then token bits.
//!   This exploits the fact that many tokens share similar TSID patterns.
//!   Variable order: `tsid_12, tsid_11, ..., tsid_0, tok_11, tok_10, ..., tok_0`
//!
//! - **Uses biodivine_lib_bdd**: Leverages a well-tested BDD library for
//!   efficient operations, node sharing, and memory management.
//!
//! ## Position Encoding
//! Position = token * num_tsids + tsid

use biodivine_lib_bdd::{Bdd, BddValuation, BddVariable, BddVariableSet};
use range_set_blaze::RangeSetBlaze;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// Thread-local cache for BddVariableSets to avoid repeated allocation.
// Key is (tsid_bits, token_bits).
thread_local! {
    static VAR_SET_CACHE: RefCell<HashMap<(u8, u8), Arc<BddVariableSet>>> = RefCell::new(HashMap::new());
}

/// Per-weight BDD representation using biodivine_lib_bdd.
///
/// Each weight stores its own independent BDD.
#[derive(Clone, Debug)]
pub struct BddWeightBiodivine {
    /// The BDD representing this weight.
    bdd: Bdd,
    /// Variable set (shared via Arc for cheap cloning).
    vars: Arc<BddVariableSet>,
    /// Number of TSID values (M dimension).
    tsid_dim: u16,
    /// Number of Token values (N dimension).
    token_dim: u16,
    /// Number of bits for TSID encoding.
    tsid_bits: u8,
    /// Number of bits for Token encoding.
    token_bits: u8,
}

impl PartialEq for BddWeightBiodivine {
    fn eq(&self, other: &Self) -> bool {
        // Two BDDs are equal if they represent the same boolean function
        // and have the same dimensions
        self.tsid_dim == other.tsid_dim 
            && self.token_dim == other.token_dim
            && self.bdd.iff(&other.bdd).is_true()
    }
}

impl Eq for BddWeightBiodivine {}

impl BddWeightBiodivine {
    /// Calculate bits needed to represent values 0..max_val (inclusive).
    pub fn bits_for(max_val: u16) -> u8 {
        if max_val == 0 {
            1
        } else {
            (16 - max_val.leading_zeros()) as u8
        }
    }

    /// Create variable set with TSID-first ordering.
    fn create_vars(tsid_bits: u8, token_bits: u8) -> BddVariableSet {
        let total_bits = (tsid_bits + token_bits) as u16;
        BddVariableSet::new_anonymous(total_bits)
    }

    /// Get or create a cached variable set for the given bit widths.
    fn get_vars(tsid_bits: u8, token_bits: u8) -> Arc<BddVariableSet> {
        VAR_SET_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.entry((tsid_bits, token_bits))
                .or_insert_with(|| Arc::new(Self::create_vars(tsid_bits, token_bits)))
                .clone()
        })
    }

    /// Build a BDD for the interval [lo, hi] over k bits, MSB to LSB.
    /// `var_offset` is the starting variable index.
    fn interval(
        vars: &BddVariableSet,
        var_offset: u8,
        k: u8,
        lo: u16,
        hi: u16,
    ) -> Bdd {
        if k == 0 {
            return if lo == 0 && hi == 0 {
                vars.mk_true()
            } else {
                vars.mk_false()
            };
        }

        // Full domain for k bits
        let full_hi = if k >= 16 { u16::MAX } else { (1u16 << k) - 1 };

        if lo == 0 && hi >= full_hi {
            return vars.mk_true();
        }

        if lo > hi {
            return vars.mk_false();
        }

        let msb_mask = 1u16 << (k - 1);
        let lo_msb = (lo & msb_mask) != 0;
        let hi_msb = (hi & msb_mask) != 0;

        // Variable index for this bit (MSB first)
        let var = BddVariable::from_index(var_offset as usize);
        let x = vars.mk_var(var);
        let nx = vars.mk_not_var(var);

        match (lo_msb, hi_msb) {
            (false, false) => {
                // Stay in 0* half
                let child = Self::interval(vars, var_offset + 1, k - 1, lo, hi);
                nx.and(&child)
            }
            (true, true) => {
                // Stay in 1* half
                let child = Self::interval(vars, var_offset + 1, k - 1, lo - msb_mask, hi - msb_mask);
                x.and(&child)
            }
            (false, true) => {
                // Cross the boundary: [lo..msb_mask-1] U [msb_mask..hi]
                let left_hi = msb_mask - 1;
                let left = Self::interval(vars, var_offset + 1, k - 1, lo, left_hi);
                let right = Self::interval(vars, var_offset + 1, k - 1, 0, hi - msb_mask);
                nx.and(&left).or(&x.and(&right))
            }
            (true, false) => {
                // Impossible if lo <= hi
                vars.mk_false()
            }
        }
    }

    /// Build a BDD for rectangle: token in [t1, t2] AND tsid in [s1, s2].
    /// This is a static method for efficient construction.
    fn rect_static(vars: &BddVariableSet, tsid_bits: u8, token_bits: u8, t1: u16, t2: u16, s1: u16, s2: u16) -> Bdd {
        // TSID interval (first tsid_bits variables)
        let tsid_bdd = Self::interval(vars, 0, tsid_bits, s1, s2);
        
        // Token interval (next token_bits variables)
        let token_bdd = Self::interval(vars, tsid_bits, token_bits, t1, t2);
        
        tsid_bdd.and(&token_bdd)
    }

    /// Build a BDD for rectangle: token in [t1, t2] AND tsid in [s1, s2].
    fn rect(&self, t1: u16, t2: u16, s1: u16, s2: u16) -> Bdd {
        Self::rect_static(&self.vars, self.tsid_bits, self.token_bits, t1, t2, s1, s2)
    }

    /// Create from 1D ranges using dimension info.
    ///
    /// Position encoding: pos = token * tsid_dim + tsid
    pub fn from_ranges(
        ranges: impl Iterator<Item = (usize, usize)>,
        tsid_dim: u16,
        token_dim: u16,
    ) -> Self {
        let tsid_bits = Self::bits_for(tsid_dim.saturating_sub(1));
        let token_bits = Self::bits_for(token_dim.saturating_sub(1));
        // Use cached variable set for efficiency
        let vars = Self::get_vars(tsid_bits, token_bits);

        let mut result = vars.mk_false();

        for (start, end) in ranges {
            // Decompose 1D range into 2D rectangles
            let tok_s = (start / tsid_dim as usize) as u16;
            let tsid_s = (start % tsid_dim as usize) as u16;
            let tok_e = (end / tsid_dim as usize) as u16;
            let tsid_e = (end % tsid_dim as usize) as u16;

            let rects = decompose_range_to_rects(tok_s, tsid_s, tok_e, tsid_e, tsid_dim);

            for (t1, t2, s1, s2) in rects {
                let rect_bdd = Self::rect_static(&vars, tsid_bits, token_bits, t1, t2, s1, s2);
                result = result.or(&rect_bdd);

                // Short-circuit if result is TRUE
                if result.is_true() {
                    break;
                }
            }

            if result.is_true() {
                break;
            }
        }

        Self {
            bdd: result,
            vars,
            tsid_dim,
            token_dim,
            tsid_bits,
            token_bits,
        }
    }

    /// Create an empty BDD (accepts nothing).
    pub fn empty(tsid_dim: u16, token_dim: u16) -> Self {
        let tsid_bits = Self::bits_for(tsid_dim.saturating_sub(1));
        let token_bits = Self::bits_for(token_dim.saturating_sub(1));
        let vars = Self::get_vars(tsid_bits, token_bits);
        
        Self {
            bdd: vars.mk_false(),
            vars,
            tsid_dim,
            token_dim,
            tsid_bits,
            token_bits,
        }
    }

    /// Create a full BDD (accepts everything).
    pub fn full(tsid_dim: u16, token_dim: u16) -> Self {
        let tsid_bits = Self::bits_for(tsid_dim.saturating_sub(1));
        let token_bits = Self::bits_for(token_dim.saturating_sub(1));
        let vars = Self::get_vars(tsid_bits, token_bits);
        
        Self {
            bdd: vars.mk_true(),
            vars,
            tsid_dim,
            token_dim,
            tsid_bits,
            token_bits,
        }
    }

    /// Create a TSID column mask: all tokens for a specific TSID value.
    /// This is much more efficient than from_ranges for strided patterns.
    /// 
    /// Represents: {t, t+M, t+2M, ..., t+N*M} where M = tsid_dim, N = token_dim
    /// In BDD terms: tsid == specific_tsid AND token in [0, token_dim)
    pub fn tsid_column(tsid: u16, tsid_dim: u16, token_dim: u16) -> Self {
        let tsid_bits = Self::bits_for(tsid_dim.saturating_sub(1));
        let token_bits = Self::bits_for(token_dim.saturating_sub(1));
        let vars = Self::get_vars(tsid_bits, token_bits);

        // tsid must equal the specific value (interval [tsid, tsid])
        let tsid_bdd = Self::interval(&vars, 0, tsid_bits, tsid, tsid);
        
        // token can be any valid value [0, token_dim - 1]
        let token_bdd = Self::interval(&vars, tsid_bits, token_bits, 0, token_dim.saturating_sub(1));
        
        Self {
            bdd: tsid_bdd.and(&token_bdd),
            vars,
            tsid_dim,
            token_dim,
            tsid_bits,
            token_bits,
        }
    }

    /// Create a multi-TSID column mask: all tokens for a set of TSID values.
    /// More efficient than building from ranges for strided patterns.
    pub fn tsid_columns<I: IntoIterator<Item = u16>>(tsids: I, tsid_dim: u16, token_dim: u16) -> Self {
        let tsid_bits = Self::bits_for(tsid_dim.saturating_sub(1));
        let token_bits = Self::bits_for(token_dim.saturating_sub(1));
        let vars = Self::get_vars(tsid_bits, token_bits);

        // token can be any valid value [0, token_dim - 1]
        let token_bdd = Self::interval(&vars, tsid_bits, token_bits, 0, token_dim.saturating_sub(1));

        // Union of all tsid values
        let mut tsid_union = vars.mk_false();
        for tsid in tsids {
            let tsid_bdd = Self::interval(&vars, 0, tsid_bits, tsid, tsid);
            tsid_union = tsid_union.or(&tsid_bdd);
        }
        
        Self {
            bdd: tsid_union.and(&token_bdd),
            vars,
            tsid_dim,
            token_dim,
            tsid_bits,
            token_bits,
        }
    }

    /// Check if this BDD is empty.
    pub fn is_empty(&self) -> bool {
        self.bdd.is_false()
    }

    /// Check if this BDD is full.
    pub fn is_full(&self) -> bool {
        self.bdd.is_true()
    }

    /// Get the number of BDD nodes.
    pub fn num_nodes(&self) -> usize {
        self.bdd.size()
    }

    /// Get storage bytes (approximate).
    pub fn storage_bytes(&self) -> usize {
        // biodivine_lib_bdd uses more complex internal structures
        // This is an approximation based on typical node sizes
        self.bdd.size() * 24  // Rough estimate: 24 bytes per node
    }

    /// Get the TSID dimension.
    pub fn tsid_dim(&self) -> u16 {
        self.tsid_dim
    }

    /// Get the Token dimension.
    pub fn token_dim(&self) -> u16 {
        self.token_dim
    }

    /// Check if (token, tsid) is in this weight.
    pub fn contains(&self, token: u16, tsid: u16) -> bool {
        if token >= self.token_dim || tsid >= self.tsid_dim {
            return false;
        }

        // Build valuation: TSID bits first (MSB to LSB), then Token bits
        let total_bits = (self.tsid_bits + self.token_bits) as usize;
        let mut valuation = vec![false; total_bits];

        // Set TSID bits (MSB first)
        for i in 0..self.tsid_bits {
            let bit_idx = self.tsid_bits - 1 - i;
            valuation[i as usize] = (tsid >> bit_idx) & 1 == 1;
        }

        // Set Token bits (MSB first)
        for i in 0..self.token_bits {
            let bit_idx = self.token_bits - 1 - i;
            valuation[(self.tsid_bits + i) as usize] = (token >> bit_idx) & 1 == 1;
        }

        self.bdd.eval_in(&BddValuation::new(valuation))
    }

    /// Check if a 1D position is in this weight.
    pub fn contains_pos(&self, pos: usize) -> bool {
        let token = (pos / self.tsid_dim as usize) as u16;
        let tsid = (pos % self.tsid_dim as usize) as u16;
        self.contains(token, tsid)
    }

    /// Convert to RangeSetBlaze.
    pub fn to_rangeset(&self) -> RangeSetBlaze<usize> {
        if self.is_empty() {
            return RangeSetBlaze::new();
        }
        if self.is_full() {
            return RangeSetBlaze::from_iter([0..=(self.token_dim as usize * self.tsid_dim as usize - 1)]);
        }

        let mut rsb = RangeSetBlaze::new();
        self.enumerate_positions(&mut |pos| { rsb.insert(pos); });
        rsb
    }

    /// Enumerate all positions in this BDD.
    fn enumerate_positions(&self, callback: &mut impl FnMut(usize)) {
        if self.is_empty() {
            return;
        }
        if self.is_full() {
            for pos in 0..(self.token_dim as usize * self.tsid_dim as usize) {
                callback(pos);
            }
            return;
        }

        // Enumerate satisfying assignments
        for valuation in self.bdd.sat_valuations() {
            // Decode TSID from first tsid_bits
            let mut tsid: u16 = 0;
            for i in 0..self.tsid_bits {
                let bit_idx = self.tsid_bits - 1 - i;
                let var = BddVariable::from_index(i as usize);
                if valuation[var] {
                    tsid |= 1 << bit_idx;
                }
            }

            // Decode Token from next token_bits
            let mut token: u16 = 0;
            for i in 0..self.token_bits {
                let bit_idx = self.token_bits - 1 - i;
                let var = BddVariable::from_index((self.tsid_bits + i) as usize);
                if valuation[var] {
                    token |= 1 << bit_idx;
                }
            }

            // Skip out-of-range values (due to bit padding)
            if token >= self.token_dim || tsid >= self.tsid_dim {
                continue;
            }

            let pos = token as usize * self.tsid_dim as usize + tsid as usize;
            callback(pos);
        }
    }

    /// Union of two BDDs.
    pub fn union(&self, other: &Self) -> Self {
        assert_eq!(self.tsid_dim, other.tsid_dim);
        assert_eq!(self.token_dim, other.token_dim);

        Self {
            bdd: self.bdd.or(&other.bdd),
            vars: self.vars.clone(),
            tsid_dim: self.tsid_dim,
            token_dim: self.token_dim,
            tsid_bits: self.tsid_bits,
            token_bits: self.token_bits,
        }
    }

    /// Intersection of two BDDs.
    pub fn intersection(&self, other: &Self) -> Self {
        assert_eq!(self.tsid_dim, other.tsid_dim);
        assert_eq!(self.token_dim, other.token_dim);

        Self {
            bdd: self.bdd.and(&other.bdd),
            vars: self.vars.clone(),
            tsid_dim: self.tsid_dim,
            token_dim: self.token_dim,
            tsid_bits: self.tsid_bits,
            token_bits: self.token_bits,
        }
    }

    /// Complement of this BDD.
    ///
    /// Note: The complement is computed over the full bit domain,
    /// which may include positions outside [0, token_dim * tsid_dim).
    /// Use `complement_clipped` for domain-aware complement.
    pub fn complement(&self) -> Self {
        // Build domain mask: token < token_dim AND tsid < tsid_dim
        let domain_mask = self.rect(0, self.token_dim - 1, 0, self.tsid_dim - 1);
        
        Self {
            bdd: self.bdd.not().and(&domain_mask),
            vars: self.vars.clone(),
            tsid_dim: self.tsid_dim,
            token_dim: self.token_dim,
            tsid_bits: self.tsid_bits,
            token_bits: self.token_bits,
        }
    }

    /// Subtract another BDD from this one.
    pub fn subtract(&self, other: &Self) -> Self {
        self.intersection(&other.complement())
    }

    /// Iterate over all positions in this BDD weight.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        let mut positions = Vec::new();
        self.enumerate_positions(&mut |pos| positions.push(pos));
        positions.sort_unstable();
        positions.into_iter()
    }

    /// Get the number of positions in this BDD.
    pub fn len(&self) -> usize {
        if self.is_empty() {
            return 0;
        }
        if self.is_full() {
            return self.token_dim as usize * self.tsid_dim as usize;
        }
        self.iter().count()
    }
}

/// Decompose a 1D range into 2D rectangles.
fn decompose_range_to_rects(
    tok_s: u16,
    tsid_s: u16,
    tok_e: u16,
    tsid_e: u16,
    tsid_dim: u16,
) -> Vec<(u16, u16, u16, u16)> {
    let mut rects = Vec::new();

    if tok_s == tok_e {
        // Single row
        rects.push((tok_s, tok_e, tsid_s, tsid_e));
    } else if tok_s + 1 == tok_e && tsid_s == 0 && tsid_e == tsid_dim - 1 {
        // Two full rows
        rects.push((tok_s, tok_e, 0, tsid_dim - 1));
    } else {
        // Multi-row case
        // First partial row
        if tsid_s > 0 || tok_s + 1 > tok_e {
            rects.push((tok_s, tok_s, tsid_s, tsid_dim - 1));
        } else {
            rects.push((tok_s, tok_s, 0, tsid_dim - 1));
        }

        // Middle full rows
        if tok_s + 1 < tok_e {
            rects.push((tok_s + 1, tok_e - 1, 0, tsid_dim - 1));
        }

        // Last partial row
        if tok_s < tok_e {
            rects.push((tok_e, tok_e, 0, tsid_e));
        }
    }

    rects
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_bdd() {
        let bdd = BddWeightBiodivine::empty(100, 100);
        assert!(bdd.is_empty());
        assert!(!bdd.is_full());
        assert!(!bdd.contains(0, 0));
    }

    #[test]
    fn test_full_bdd() {
        let bdd = BddWeightBiodivine::full(100, 100);
        assert!(bdd.is_full());
        assert!(!bdd.is_empty());
        assert!(bdd.contains(0, 0));
        assert!(bdd.contains(50, 50));
        assert!(bdd.contains(99, 99));
        assert!(!bdd.contains(100, 0));
    }

    #[test]
    fn test_single_point() {
        let tsid_dim = 100u16;
        let token_dim = 100u16;
        
        let pos = 5 * tsid_dim as usize + 3; // token=5, tsid=3
        let bdd = BddWeightBiodivine::from_ranges(
            std::iter::once((pos, pos)),
            tsid_dim,
            token_dim,
        );
        
        assert!(bdd.contains(5, 3));
        assert!(!bdd.contains(5, 4));
        assert!(!bdd.contains(4, 3));
    }

    #[test]
    fn test_single_row() {
        let tsid_dim = 10u16;
        let token_dim = 10u16;
        
        // Token 5, tsid 0-9
        let start = 5 * tsid_dim as usize + 0;
        let end = 5 * tsid_dim as usize + 9;
        let bdd = BddWeightBiodivine::from_ranges(
            std::iter::once((start, end)),
            tsid_dim,
            token_dim,
        );
        
        for tsid in 0..10 {
            assert!(bdd.contains(5, tsid), "Should contain (5, {})", tsid);
        }
        assert!(!bdd.contains(4, 0));
        assert!(!bdd.contains(6, 0));
    }

    #[test]
    fn test_union_disjoint() {
        let tsid_dim = 10u16;
        let token_dim = 10u16;
        
        let a = BddWeightBiodivine::from_ranges(vec![(0, 5)].into_iter(), tsid_dim, token_dim);
        let b = BddWeightBiodivine::from_ranges(vec![(10, 15)].into_iter(), tsid_dim, token_dim);
        
        let c = a.union(&b);
        
        for pos in 0..=5 {
            assert!(c.contains_pos(pos), "Position {} should be in union", pos);
        }
        for pos in 10..=15 {
            assert!(c.contains_pos(pos), "Position {} should be in union", pos);
        }
        for pos in 6..10 {
            assert!(!c.contains_pos(pos), "Position {} should NOT be in union", pos);
        }
    }

    #[test]
    fn test_intersection() {
        let tsid_dim = 10u16;
        let token_dim = 10u16;
        
        let a = BddWeightBiodivine::from_ranges(vec![(0, 10)].into_iter(), tsid_dim, token_dim);
        let b = BddWeightBiodivine::from_ranges(vec![(5, 15)].into_iter(), tsid_dim, token_dim);
        
        let c = a.intersection(&b);
        
        for pos in 0..5 {
            assert!(!c.contains_pos(pos));
        }
        for pos in 5..=10 {
            assert!(c.contains_pos(pos));
        }
        for pos in 11..=15 {
            assert!(!c.contains_pos(pos));
        }
    }

    #[test]
    fn test_complement() {
        let tsid_dim = 10u16;
        let token_dim = 10u16;
        
        let a = BddWeightBiodivine::from_ranges(vec![(0, 4)].into_iter(), tsid_dim, token_dim);
        let not_a = a.complement();
        
        for pos in 0..=4 {
            assert!(!not_a.contains_pos(pos));
        }
        for pos in 5..20 {
            assert!(not_a.contains_pos(pos));
        }
    }

    #[test]
    fn test_len() {
        let tsid_dim = 10u16;
        let token_dim = 10u16;
        
        let a = BddWeightBiodivine::from_ranges(vec![(0, 5)].into_iter(), tsid_dim, token_dim);
        assert_eq!(a.len(), 6);
        
        let empty = BddWeightBiodivine::empty(tsid_dim, token_dim);
        assert_eq!(empty.len(), 0);
        
        let full = BddWeightBiodivine::full(tsid_dim, token_dim);
        assert_eq!(full.len(), 100);
    }

    #[test]
    fn test_compare_with_custom_bdd() {
        use crate::dwa_i32::bdd_weight::BddWeight;
        
        let tsid_dim = 100u16;
        let token_dim = 100u16;
        
        // Test various range patterns
        let test_cases: Vec<Vec<(usize, usize)>> = vec![
            // Single points
            vec![(503, 503)],
            // Single range
            vec![(100, 200)],
            // Multiple disjoint ranges
            vec![(10, 50), (100, 150), (500, 600)],
            // Adjacent ranges
            vec![(100, 149), (150, 200)],
            // Large ranges
            vec![(0, 999), (5000, 5999)],
        ];
        
        for (i, ranges) in test_cases.iter().enumerate() {
            let custom = BddWeight::from_ranges(
                ranges.iter().cloned(),
                tsid_dim,
                token_dim,
            );
            let biodivine = BddWeightBiodivine::from_ranges(
                ranges.iter().cloned(),
                tsid_dim,
                token_dim,
            );
            
            // Verify they have the same positions
            let custom_rs = custom.to_rangeset();
            let biodivine_rs = biodivine.to_rangeset();
            
            assert_eq!(
                custom_rs, biodivine_rs,
                "Test case {} failed: wrapper and biodivine produce different results",
                i
            );
            
            // Print stats for comparison.
            // Note: `BddWeight` is a compatibility wrapper; both are biodivine-backed.
            let custom_bytes = custom.storage_bytes();
            let biodivine_bytes = biodivine.storage_bytes();
            eprintln!(
                "Test case {}: wrapper={} nodes ({} bytes), biodivine={} nodes ({} bytes)",
                i,
                custom.num_nodes(),
                custom_bytes,
                biodivine.num_nodes(),
                biodivine_bytes
            );
        }
    }

    #[test]
    fn test_tsid_column() {
        // Test single TSID column
        let tsid_dim = 100u16;
        let token_dim = 1000u16;
        
        // Create column for tsid=5
        let column = BddWeightBiodivine::tsid_column(5, tsid_dim, token_dim);
        
        // Should contain (token, tsid=5) for all tokens
        for t in 0..token_dim {
            assert!(column.contains(t, 5), "should contain token {} with tsid 5", t);
            assert!(!column.contains(t, 6), "should not contain token {} with tsid 6", t);
        }
        
        // Length should be token_dim (one per token)
        assert_eq!(column.len(), token_dim as usize);
    }

    #[test]
    fn test_tsid_columns() {
        // Test multiple TSID columns
        let tsid_dim = 100u16;
        let token_dim = 1000u16;
        
        // Create columns for tsids 10, 20, 30
        let columns = BddWeightBiodivine::tsid_columns(
            vec![10u16, 20, 30],
            tsid_dim,
            token_dim,
        );
        
        // Should contain (token, tsid) for tsid in {10, 20, 30}
        for t in 0..token_dim {
            assert!(columns.contains(t, 10));
            assert!(columns.contains(t, 20));
            assert!(columns.contains(t, 30));
            assert!(!columns.contains(t, 5));
            assert!(!columns.contains(t, 15));
        }
        
        // Length should be 3 * token_dim
        assert_eq!(columns.len(), 3 * token_dim as usize);
        
        // Node count should be small (efficient representation)
        println!("tsid_columns node count: {}", columns.num_nodes());
        // With good BDD structure, should be roughly O(tsid_bits + token_bits)
        assert!(columns.num_nodes() < 100, "BDD should be compact, got {} nodes", columns.num_nodes());
    }

    #[test]
    fn test_typical_grammar_dimensions() {
        // Test with dimensions similar to real grammar constraints
        // From analysis: tsid_dim ≈ 1119, token_dim ≈ 50257 (GPT-2)
        // Using smaller values for faster testing
        let tsid_dim = 100u16;
        let token_dim = 1000u16;
        
        // Position = token * tsid_dim + tsid
        // Token 0: positions 0-99
        // Token 10: positions 1000-1099
        // Token 500: positions 50000-50099
        let ranges = vec![
            (0, 99),           // Token 0, all tsids (0-99)
            (1000, 1099),      // Token 10, all tsids (1000-1099)
            (50000, 50099),    // Token 500, all tsids (50000-50099)
        ];
        
        let biodivine = BddWeightBiodivine::from_ranges(
            ranges.iter().cloned(),
            tsid_dim,
            token_dim,
        );
        
        assert!(!biodivine.is_empty());
        assert!(!biodivine.is_full());
        assert_eq!(biodivine.len(), 300);
        
        // Check some specific positions
        assert!(biodivine.contains(0, 0));      // Token 0, tsid 0
        assert!(biodivine.contains(0, 99));     // Token 0, tsid 99
        assert!(!biodivine.contains(0, 100));   // tsid out of range
        assert!(biodivine.contains(10, 0));     // Token 10, tsid 0 (pos=1000)
        assert!(biodivine.contains(10, 99));    // Token 10, tsid 99 (pos=1099)
        assert!(!biodivine.contains(1, 0));     // Token 1 not in ranges
        assert!(biodivine.contains(500, 0));    // Token 500, tsid 0 (pos=50000)
        assert!(biodivine.contains(500, 99));   // Token 500, tsid 99 (pos=50099)
        assert!(!biodivine.contains(2, 0));     // Token 2 not in ranges
    }
}