//! Per-Weight BDD Storage with TSID-First Variable Ordering.
//!
//! This module implements a compact BDD representation for individual weights.
//! Unlike `SharedBdd` which shares nodes across weights, `BddWeight` stores
//! an independent BDD per weight, optimized for space efficiency.
//!
//! ## Key Features
//!
//! - **TSID-First Ordering**: TSID bits first (MSB to LSB), then token bits.
//!   This exploits the fact that many tokens share similar TSID patterns.
//!   Variable order: `tsid_12, tsid_11, ..., tsid_0, tok_11, tok_10, ..., tok_0`
//!
//! - **5-Byte Compact Nodes**: Each node is:
//!   - 1 byte: variable index (0-24), or 0xFF for terminal
//!   - 2 bytes: low child index (0 = FALSE, 1 = TRUE for terminals)
//!   - 2 bytes: high child index
//!
//! - **Per-Weight Independence**: Each weight has its own node array,
//!   enabling parallel construction and avoiding lock contention.
//!
//! ## Position Encoding
//! Position = token * num_tsids + tsid

use range_set_blaze::RangeSetBlaze;
use std::collections::HashMap;

/// Compact BDD node representation (5 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct BddNode {
    /// Variable index (0-24), or 0xFF for terminal
    pub var: u8,
    /// Index of low child (when variable bit = 0)
    pub lo: u16,
    /// Index of high child (when variable bit = 1)
    pub hi: u16,
}

impl BddNode {
    /// Check if this is a terminal node.
    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.var == 0xFF
    }
}

/// Per-weight BDD representation.
///
/// Each weight stores its own independent BDD node array.
/// This enables parallel construction and avoids lock contention.
#[derive(Clone, Debug)]
pub struct BddWeight {
    /// This weight's BDD nodes. Node 0 = FALSE, Node 1 = TRUE.
    nodes: Vec<BddNode>,
    /// Root node index.
    root: u16,
    /// Number of TSID values (M dimension).
    tsid_dim: u16,
    /// Number of Token values (N dimension).
    token_dim: u16,
}

/// Builder for constructing BDDs with unique table.
struct BddBuilder {
    nodes: Vec<BddNode>,
    unique: HashMap<(u8, u16, u16), u16>,
    tsid_bits: u8,
    token_bits: u8,
}

impl BddBuilder {
    /// FALSE terminal constant.
    const FALSE: u16 = 0;
    /// TRUE terminal constant.
    const TRUE: u16 = 1;
    
    /// Create a new builder with terminal nodes.
    fn new(tsid_bits: u8, token_bits: u8) -> Self {
        let mut nodes = Vec::with_capacity(256);
        // Add terminal nodes
        nodes.push(BddNode { var: 0xFF, lo: 0, hi: 0 }); // FALSE
        nodes.push(BddNode { var: 0xFF, lo: 0, hi: 1 }); // TRUE
        
        BddBuilder {
            nodes,
            unique: HashMap::new(),
            tsid_bits,
            token_bits,
        }
    }
    
    /// Get or create a BDD node.
    /// Applies the reduction rule: if lo == hi, return lo (skip this variable).
    fn mk(&mut self, var: u8, lo: u16, hi: u16) -> u16 {
        // BDD reduction rule
        if lo == hi {
            return lo;
        }
        
        // Check unique table
        let key = (var, lo, hi);
        if let Some(&id) = self.unique.get(&key) {
            return id;
        }
        
        // Create new node
        let id = self.nodes.len() as u16;
        if id >= u16::MAX {
            // Overflow protection - shouldn't happen for reasonable weights
            panic!("BDD node overflow");
        }
        self.nodes.push(BddNode { var, lo, hi });
        self.unique.insert(key, id);
        id
    }
    
    /// Build a BDD for interval constraint: val in [lo, hi].
    /// Tests bits from MSB to LSB starting at `first_var`.
    fn interval(&mut self, first_var: u8, num_bits: u8, lo: u16, hi: u16) -> u16 {
        if lo > hi {
            return Self::FALSE;
        }
        
        let max_val = (1u32 << num_bits).saturating_sub(1) as u16;
        if lo == 0 && hi >= max_val {
            return Self::TRUE;
        }
        
        self.interval_rec(first_var, 0, num_bits, lo, hi, 0, max_val)
    }
    
    /// Recursive interval construction.
    fn interval_rec(
        &mut self,
        first_var: u8,
        bit: u8,
        num_bits: u8,
        lo: u16,
        hi: u16,
        cur_lo: u16,
        cur_hi: u16,
    ) -> u16 {
        // Completely contained → TRUE
        if cur_lo >= lo && cur_hi <= hi {
            return Self::TRUE;
        }
        
        // Completely outside → FALSE
        if cur_hi < lo || cur_lo > hi {
            return Self::FALSE;
        }
        
        // All bits processed
        if bit >= num_bits {
            return if cur_lo >= lo && cur_lo <= hi { Self::TRUE } else { Self::FALSE };
        }
        
        // Split the range
        let var = first_var + bit;
        let pivot = (1u32 << (num_bits - bit - 1)) as u16 + cur_lo;
        
        let lo_child = self.interval_rec(first_var, bit + 1, num_bits, lo, hi, cur_lo, pivot - 1);
        let hi_child = self.interval_rec(first_var, bit + 1, num_bits, lo, hi, pivot, cur_hi);
        
        self.mk(var, lo_child, hi_child)
    }
    
    /// Build a BDD for a 2D rectangle: token in [t1, t2] AND tsid in [s1, s2].
    /// Uses TSID-first ordering.
    fn rect(&mut self, t1: u16, t2: u16, s1: u16, s2: u16) -> u16 {
        // Build TSID constraint (first tsid_bits variables)
        let tsid_bdd = self.interval(0, self.tsid_bits, s1, s2);
        
        // Build Token constraint (next token_bits variables)
        let token_bdd = self.interval(self.tsid_bits, self.token_bits, t1, t2);
        
        // AND them together
        self.apply_and(tsid_bdd, token_bdd)
    }
    
    /// Apply AND operation to two BDDs.
    fn apply_and(&mut self, a: u16, b: u16) -> u16 {
        self.apply(a, b, false)
    }
    
    /// Apply OR operation to two BDDs.
    fn apply_or(&mut self, a: u16, b: u16) -> u16 {
        self.apply(a, b, true)
    }
    
    /// Generic apply operation (AND or OR).
    fn apply(&mut self, a: u16, b: u16, is_or: bool) -> u16 {
        // Use a memo table for this apply call
        let mut memo: HashMap<(u16, u16), u16> = HashMap::new();
        self.apply_memo(a, b, is_or, &mut memo)
    }
    
    fn apply_memo(&mut self, a: u16, b: u16, is_or: bool, memo: &mut HashMap<(u16, u16), u16>) -> u16 {
        // Terminal cases
        if is_or {
            if a == Self::TRUE || b == Self::TRUE { return Self::TRUE; }
            if a == Self::FALSE { return b; }
            if b == Self::FALSE { return a; }
        } else {
            if a == Self::FALSE || b == Self::FALSE { return Self::FALSE; }
            if a == Self::TRUE { return b; }
            if b == Self::TRUE { return a; }
        }
        
        // Normalize order for cache
        let (a, b) = if a > b { (b, a) } else { (a, b) };
        
        // Check memo
        if let Some(&result) = memo.get(&(a, b)) {
            return result;
        }
        
        // Get nodes
        let node_a = self.nodes[a as usize];
        let node_b = self.nodes[b as usize];
        
        // Determine variable to split on
        let var = node_a.var.min(node_b.var);
        
        // Get children
        let (a_lo, a_hi) = if node_a.var == var {
            (node_a.lo, node_a.hi)
        } else {
            (a, a)
        };
        
        let (b_lo, b_hi) = if node_b.var == var {
            (node_b.lo, node_b.hi)
        } else {
            (b, b)
        };
        
        // Recursive apply
        let lo = self.apply_memo(a_lo, b_lo, is_or, memo);
        let hi = self.apply_memo(a_hi, b_hi, is_or, memo);
        
        let result = self.mk(var, lo, hi);
        memo.insert((a, b), result);
        result
    }
    
    /// Finish building and return the BddWeight.
    fn finish(self, root: u16, tsid_dim: u16, token_dim: u16) -> BddWeight {
        BddWeight {
            nodes: self.nodes,
            root,
            tsid_dim,
            token_dim,
        }
    }
}

impl BddWeight {
    /// Calculate the number of bits needed to represent values up to max.
    fn bits_for(max: u16) -> u8 {
        if max == 0 {
            return 1;
        }
        (16 - max.leading_zeros()) as u8
    }
    
    /// Create a BDD from 1D ranges using dimension information.
    ///
    /// Each range [start, end] in N×M space is decomposed into 2D rectangles
    /// and combined via OR.
    pub fn from_ranges(
        ranges: impl Iterator<Item = (usize, usize)>,
        tsid_dim: u16,
        token_dim: u16,
    ) -> Self {
        let tsid_bits = Self::bits_for(tsid_dim.saturating_sub(1));
        let token_bits = Self::bits_for(token_dim.saturating_sub(1));
        
        let mut builder = BddBuilder::new(tsid_bits, token_bits);
        let mut root = BddBuilder::FALSE;
        
        for (start, end) in ranges {
            // Decompose 1D range to 2D coordinates
            let tok_s = (start / tsid_dim as usize) as u16;
            let tsid_s = (start % tsid_dim as usize) as u16;
            let tok_e = (end / tsid_dim as usize) as u16;
            let tsid_e = (end % tsid_dim as usize) as u16;
            
            // Decompose into rectangles
            let rects = decompose_range_to_rects(tok_s, tsid_s, tok_e, tsid_e, tsid_dim);
            
            for (t1, t2, s1, s2) in rects {
                let rect_bdd = builder.rect(t1, t2, s1, s2);
                root = builder.apply_or(root, rect_bdd);
            }
        }
        
        builder.finish(root, tsid_dim, token_dim)
    }
    
    /// Create an empty BDD (accepts nothing).
    pub fn empty(tsid_dim: u16, token_dim: u16) -> Self {
        let tsid_bits = Self::bits_for(tsid_dim.saturating_sub(1));
        let token_bits = Self::bits_for(token_dim.saturating_sub(1));
        let builder = BddBuilder::new(tsid_bits, token_bits);
        builder.finish(BddBuilder::FALSE, tsid_dim, token_dim)
    }
    
    /// Create a full BDD (accepts everything).
    pub fn full(tsid_dim: u16, token_dim: u16) -> Self {
        let tsid_bits = Self::bits_for(tsid_dim.saturating_sub(1));
        let token_bits = Self::bits_for(token_dim.saturating_sub(1));
        let builder = BddBuilder::new(tsid_bits, token_bits);
        builder.finish(BddBuilder::TRUE, tsid_dim, token_dim)
    }
    
    /// Check if the BDD is empty (accepts nothing).
    pub fn is_empty(&self) -> bool {
        self.root == 0  // FALSE terminal
    }
    
    /// Check if the BDD is full (accepts everything).
    pub fn is_full(&self) -> bool {
        self.root == 1  // TRUE terminal
    }
    
    /// Get the number of nodes in this BDD.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
    
    /// Get the storage size in bytes (5 bytes per node).
    pub fn storage_bytes(&self) -> usize {
        self.nodes.len() * 5
    }
    
    /// Get the TSID dimension.
    pub fn tsid_dim(&self) -> u16 {
        self.tsid_dim
    }
    
    /// Get the Token dimension.
    pub fn token_dim(&self) -> u16 {
        self.token_dim
    }
    
    /// Test if a (token, tsid) pair is contained in the BDD.
    pub fn contains(&self, token: u16, tsid: u16) -> bool {
        let tsid_bits = Self::bits_for(self.tsid_dim.saturating_sub(1));
        let token_bits = Self::bits_for(self.token_dim.saturating_sub(1));
        
        let mut node = self.root;
        
        // Traverse TSID bits (variables 0 to tsid_bits-1)
        for bit in 0..tsid_bits {
            if node <= 1 {
                return node == 1;
            }
            let n = &self.nodes[node as usize];
            if n.var > bit {
                // Variable not present, continue
                continue;
            }
            if n.var < bit {
                // Out of order - should not happen
                panic!("BDD has out-of-order variables");
            }
            let bit_val = (tsid >> (tsid_bits - 1 - bit)) & 1;
            node = if bit_val == 1 { n.hi } else { n.lo };
        }
        
        // Traverse Token bits (variables tsid_bits to tsid_bits+token_bits-1)
        for bit in 0..token_bits {
            if node <= 1 {
                return node == 1;
            }
            let n = &self.nodes[node as usize];
            let expected_var = tsid_bits + bit;
            if n.var > expected_var {
                // Variable not present
                continue;
            }
            if n.var < expected_var {
                panic!("BDD has out-of-order variables");
            }
            let bit_val = (token >> (token_bits - 1 - bit)) & 1;
            node = if bit_val == 1 { n.hi } else { n.lo };
        }
        
        node == 1
    }
    
    /// Test if a 1D position is contained in the BDD.
    /// Position = token * tsid_dim + tsid
    pub fn contains_pos(&self, pos: usize) -> bool {
        let token = (pos / self.tsid_dim as usize) as u16;
        let tsid = (pos % self.tsid_dim as usize) as u16;
        self.contains(token, tsid)
    }
    
    /// Convert the BDD back to a RangeSetBlaze.
    ///
    /// This iterates over all (token, tsid) pairs to find accepted ones.
    /// For large dimensions, this can be slow - prefer using `enumerate_positions`.
    pub fn to_rangeset(&self) -> RangeSetBlaze<usize> {
        let mut positions: Vec<usize> = Vec::new();
        
        // Use BDD traversal to enumerate accepted values
        self.enumerate_positions(&mut positions);
        
        // Convert positions to ranges efficiently
        RangeSetBlaze::from_iter(positions)
    }
    
    /// Enumerate all accepted positions into a vector.
    pub fn enumerate_positions(&self, positions: &mut Vec<usize>) {
        let tsid_bits = Self::bits_for(self.tsid_dim.saturating_sub(1));
        let token_bits = Self::bits_for(self.token_dim.saturating_sub(1));
        
        self.enumerate_rec(self.root, 0, 0, 0, tsid_bits, token_bits, positions);
    }
    
    /// Recursive enumeration helper.
    fn enumerate_rec(
        &self,
        node: u16,
        var: u8,
        tsid_acc: u16,
        token_acc: u16,
        tsid_bits: u8,
        token_bits: u8,
        positions: &mut Vec<usize>,
    ) {
        // Terminal cases
        if node == 0 {
            return; // FALSE - no positions
        }
        if node == 1 {
            // TRUE - add all positions reachable from current accumulators
            let total_vars = tsid_bits + token_bits;
            let remaining = total_vars - var;
            
            // Enumerate all 2^remaining combinations
            for i in 0..(1u32 << remaining) {
                let mut tsid = tsid_acc;
                let mut token = token_acc;
                
                for b in 0..remaining {
                    let bit_val = ((i >> (remaining as u32 - 1 - b as u32)) & 1) as u16;
                    let v = var + b as u8;
                    if v < tsid_bits {
                        tsid = (tsid << 1) | bit_val;
                    } else {
                        token = (token << 1) | bit_val;
                    }
                }
                
                // Check bounds
                if tsid < self.tsid_dim && token < self.token_dim {
                    let pos = token as usize * self.tsid_dim as usize + tsid as usize;
                    positions.push(pos);
                }
            }
            return;
        }
        
        let n = &self.nodes[node as usize];
        let node_var = n.var;
        
        // Handle skipped variables (all values accepted for those bits)
        if node_var > var {
            // Need to enumerate both branches for skipped variables
            let skipped = node_var - var;
            for i in 0..(1u32 << skipped) {
                let mut tsid = tsid_acc;
                let mut token = token_acc;
                
                for b in 0..skipped {
                    let bit_val = ((i >> (skipped as u32 - 1 - b as u32)) & 1) as u16;
                    let v = var + b as u8;
                    if v < tsid_bits {
                        tsid = (tsid << 1) | bit_val;
                    } else {
                        token = (token << 1) | bit_val;
                    }
                }
                
                self.enumerate_rec(node, node_var, tsid, token, tsid_bits, token_bits, positions);
            }
            return;
        }
        
        // Process current variable
        let (new_tsid_lo, new_token_lo) = if var < tsid_bits {
            ((tsid_acc << 1), token_acc)
        } else {
            (tsid_acc, (token_acc << 1))
        };
        
        let (new_tsid_hi, new_token_hi) = if var < tsid_bits {
            ((tsid_acc << 1) | 1, token_acc)
        } else {
            (tsid_acc, (token_acc << 1) | 1)
        };
        
        // Recurse on both branches
        self.enumerate_rec(n.lo, var + 1, new_tsid_lo, new_token_lo, tsid_bits, token_bits, positions);
        self.enumerate_rec(n.hi, var + 1, new_tsid_hi, new_token_hi, tsid_bits, token_bits, positions);
    }
}

/// Decompose a 1D range into 2D rectangles.
///
/// A 1D range [start, end] in N×M space decomposes into up to 3 rectangles:
/// 1. Partial first row: [tok_s, tok_s] × [tsid_s, M-1]  (if not starting at tsid 0)
/// 2. Full middle rows: [tok_s+1, tok_e-1] × [0, M-1]   (if there are full rows)
/// 3. Partial last row:  [tok_e, tok_e] × [0, tsid_e]    (if not ending at tsid M-1)
fn decompose_range_to_rects(
    tok_s: u16,
    tsid_s: u16,
    tok_e: u16,
    tsid_e: u16,
    num_tsids: u16,
) -> Vec<(u16, u16, u16, u16)> {
    let max_tsid = num_tsids.saturating_sub(1);
    
    if tok_s == tok_e {
        // Single row: one rectangle
        return vec![(tok_s, tok_e, tsid_s, tsid_e)];
    }
    
    let mut rects = Vec::with_capacity(3);
    
    // Partial first row (if not starting at tsid 0)
    if tsid_s > 0 {
        rects.push((tok_s, tok_s, tsid_s, max_tsid));
        
        // Full middle rows
        if tok_s + 1 < tok_e {
            rects.push((tok_s + 1, tok_e - 1, 0, max_tsid));
        }
        
        // Last row (partial or full)
        if tsid_e < max_tsid {
            rects.push((tok_e, tok_e, 0, tsid_e));
        } else {
            // Last row is full, merge with middle
            if tok_s + 1 < tok_e {
                // Extend middle row to include last row
                let last = rects.len() - 1;
                rects[last].1 = tok_e;
            } else {
                // No middle rows, just add full last row
                rects.push((tok_e, tok_e, 0, max_tsid));
            }
        }
    } else if tsid_s == 0 {
        // First row starts at tsid 0
        if tsid_e == max_tsid {
            // Entire range is full rows
            rects.push((tok_s, tok_e, 0, max_tsid));
        } else {
            // First row(s) are full, last row is partial
            if tok_s < tok_e {
                rects.push((tok_s, tok_e - 1, 0, max_tsid));
            }
            rects.push((tok_e, tok_e, 0, tsid_e));
        }
    }
    
    rects
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_empty_bdd() {
        let bdd = BddWeight::empty(100, 100);
        assert!(bdd.is_empty());
        assert!(!bdd.contains(0, 0));
        assert!(!bdd.contains(50, 50));
    }
    
    #[test]
    fn test_full_bdd() {
        let bdd = BddWeight::full(100, 100);
        assert!(bdd.is_full());
        assert!(bdd.contains(0, 0));
        assert!(bdd.contains(50, 50));
        assert!(bdd.contains(99, 99));
    }
    
    #[test]
    fn test_single_point() {
        let tsid_dim = 100u16;
        let token_dim = 100u16;
        
        // Create range for single point (token=5, tsid=3)
        let pos = 5 * 100 + 3; // position 503
        let ranges = vec![(pos, pos)];
        
        let bdd = BddWeight::from_ranges(ranges.into_iter(), tsid_dim, token_dim);
        
        assert!(!bdd.is_empty());
        assert!(bdd.contains(5, 3));
        assert!(!bdd.contains(5, 2));
        assert!(!bdd.contains(4, 3));
        assert!(!bdd.contains(0, 0));
    }
    
    #[test]
    fn test_single_row() {
        let tsid_dim = 10u16;
        let token_dim = 10u16;
        
        // Token=3, all tsids (positions 30-39)
        let ranges = vec![(30, 39)];
        let bdd = BddWeight::from_ranges(ranges.into_iter(), tsid_dim, token_dim);
        
        for tsid in 0..10 {
            assert!(bdd.contains(3, tsid), "Should contain (3, {})", tsid);
        }
        
        for token in 0..10 {
            if token != 3 {
                for tsid in 0..10 {
                    assert!(!bdd.contains(token, tsid), "Should not contain ({}, {})", token, tsid);
                }
            }
        }
    }
    
    #[test]
    fn test_rectangle() {
        let tsid_dim = 10u16;
        let token_dim = 10u16;
        
        // Tokens 2-4, tsids 3-5
        // This spans multiple rows, so multiple ranges
        let ranges: Vec<(usize, usize)> = vec![
            (2 * 10 + 3, 2 * 10 + 9),  // Token 2, tsid 3-9
            (3 * 10 + 0, 3 * 10 + 9),  // Token 3, tsid 0-9
            (4 * 10 + 0, 4 * 10 + 5),  // Token 4, tsid 0-5
        ];
        
        let bdd = BddWeight::from_ranges(ranges.into_iter(), tsid_dim, token_dim);
        
        // Check expected contents
        assert!(bdd.contains(2, 5));
        assert!(bdd.contains(3, 0));
        assert!(bdd.contains(4, 5));
        
        // Check exclusions
        assert!(!bdd.contains(2, 2)); // Before tsid range
        assert!(!bdd.contains(4, 6)); // After tsid range  
        assert!(!bdd.contains(1, 5)); // Before token range
        assert!(!bdd.contains(5, 5)); // After token range
    }
    
    #[test]
    fn test_to_rangeset_roundtrip() {
        let tsid_dim = 20u16;
        let token_dim = 20u16;
        
        // Create a complex set of ranges
        let original_ranges: Vec<(usize, usize)> = vec![
            (5, 15),   // Mixed positions
            (40, 45),  // Another range
            (100, 100), // Single point
        ];
        
        let bdd = BddWeight::from_ranges(original_ranges.clone().into_iter(), tsid_dim, token_dim);
        let recovered = bdd.to_rangeset();
        
        // Check all original positions are in recovered
        for (start, end) in &original_ranges {
            for pos in *start..=*end {
                assert!(recovered.contains(pos), "Position {} should be in recovered set", pos);
            }
        }
        
        // Check recovered doesn't have extra positions
        for pos in recovered.iter() {
            let mut found = false;
            for (start, end) in &original_ranges {
                if pos >= *start && pos <= *end {
                    found = true;
                    break;
                }
            }
            assert!(found, "Position {} shouldn't be in recovered set", pos);
        }
    }
    
    #[test]
    fn test_node_count() {
        let tsid_dim = 100u16;
        let token_dim = 100u16;
        
        // Empty BDD has 2 nodes (terminals)
        let empty = BddWeight::empty(tsid_dim, token_dim);
        assert_eq!(empty.num_nodes(), 2);
        
        // Full BDD has 2 nodes (terminals)
        let full = BddWeight::full(tsid_dim, token_dim);
        assert_eq!(full.num_nodes(), 2);
        
        // Single point should have more nodes
        let single = BddWeight::from_ranges(vec![(500, 500)].into_iter(), tsid_dim, token_dim);
        assert!(single.num_nodes() > 2);
    }
    
    #[test]
    fn test_storage_bytes() {
        let bdd = BddWeight::empty(100, 100);
        assert_eq!(bdd.storage_bytes(), 2 * 5); // 2 nodes × 5 bytes
    }
    
    #[test]
    fn test_decompose_single_row() {
        let rects = decompose_range_to_rects(5, 3, 5, 7, 10);
        assert_eq!(rects.len(), 1);
        assert_eq!(rects[0], (5, 5, 3, 7));
    }
    
    #[test]
    fn test_decompose_multiple_rows() {
        // Token 2, tsid 3 to token 4, tsid 5 with 10 tsids
        let rects = decompose_range_to_rects(2, 3, 4, 5, 10);
        
        // Should decompose into:
        // 1. Token 2, tsid 3-9 (partial first row)
        // 2. Token 3, tsid 0-9 (full middle row)
        // 3. Token 4, tsid 0-5 (partial last row)
        assert!(rects.len() <= 3);
    }
    
    #[test]
    fn test_bits_for() {
        assert_eq!(BddWeight::bits_for(0), 1);
        assert_eq!(BddWeight::bits_for(1), 1);
        assert_eq!(BddWeight::bits_for(2), 2);
        assert_eq!(BddWeight::bits_for(3), 2);
        assert_eq!(BddWeight::bits_for(4), 3);
        assert_eq!(BddWeight::bits_for(255), 8);
        assert_eq!(BddWeight::bits_for(256), 9);
    }
}
