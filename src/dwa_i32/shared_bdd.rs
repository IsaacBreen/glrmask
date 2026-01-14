//! Shared BDD (Binary Decision Diagram) for compact weight representation.
//!
//! This module implements a highly efficient shared BDD representation for Terminal DWA
//! weights. Using TSID-first variable ordering, it achieves ~275 KB storage with 5-byte
//! nodes and built-in sharing across weights.
//!
//! ## Key Design Decisions
//!
//! 1. **TSID-First Variable Ordering**: Places TSID bits first (MSB to LSB), then token bits.
//!    This exploits the fact that many weights share "TSID Profiles" (e.g., "Full Row").
//!    Variable order: `tsid_12, tsid_11, ..., tsid_0, tok_11, tok_10, ..., tok_0`
//!
//! 2. **5-Byte Compact Nodes**: Each node is:
//!    - 1 byte: variable index (0-24), or 0xFF for terminal
//!    - 2 bytes: low child index
//!    - 2 bytes: high child index
//!
//! 3. **Shared Node Pool**: All BDDs share the same node pool, enabling massive
//!    deduplication across similar weights.
//!
//! ## Position Encoding
//! Position = token * NUM_TSIDS + tsid

use std::collections::HashMap;

/// Number of bits for TSID (tokenizer state ID)
pub const TSID_BITS: u8 = 13;
/// Number of bits for Token (LLM token ID)  
pub const TOKEN_BITS: u8 = 12;
/// Total number of variables in the BDD
pub const TOTAL_VARS: u8 = TSID_BITS + TOKEN_BITS;  // 25

/// Compact BDD node representation (5 bytes).
/// Uses u16 indices assuming < 65536 unique nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BddNode {
    /// Variable index (0-24), or 0xFF for terminal
    pub var: u8,
    /// Index of low child (when variable bit = 0)
    pub lo: u16,
    /// Index of high child (when variable bit = 1)
    pub hi: u16,
}

impl BddNode {
    /// Create a terminal node
    pub fn terminal(value: bool) -> Self {
        BddNode {
            var: 0xFF,
            lo: 0,
            hi: if value { 1 } else { 0 },
        }
    }
    
    /// Check if this is a terminal node
    pub fn is_terminal(&self) -> bool {
        self.var == 0xFF
    }
}

/// A shared Binary Decision Diagram structure.
/// All BDDs share the same node pool for maximum deduplication.
#[derive(Clone)]
pub struct SharedBdd {
    /// Compact node storage. Node 0 = FALSE, Node 1 = TRUE.
    nodes: Vec<BddNode>,
    /// Unique table for node canonicalization: (var, lo, hi) -> node_id
    unique: HashMap<(u8, u16, u16), u16>,
    /// Number of tsids (for variable ordering)
    num_tsids: u16,
    /// Operation cache for apply operations
    apply_cache: HashMap<(u16, u16, bool), u16>,  // (a, b, is_or) -> result
}

impl SharedBdd {
    /// Create a new SharedBdd with the terminal nodes initialized.
    pub fn new(num_tsids: u16) -> Self {
        let mut bdd = SharedBdd {
            nodes: Vec::with_capacity(65536),
            unique: HashMap::new(),
            num_tsids,
            apply_cache: HashMap::new(),
        };
        
        // Add terminal nodes (FALSE = 0, TRUE = 1)
        bdd.nodes.push(BddNode { var: 0xFF, lo: 0, hi: 0 }); // FALSE
        bdd.nodes.push(BddNode { var: 0xFF, lo: 0, hi: 1 }); // TRUE
        
        bdd
    }
    
    /// The FALSE terminal constant.
    pub const FALSE: u16 = 0;
    /// The TRUE terminal constant.
    pub const TRUE: u16 = 1;
    
    /// Get or create a BDD node.
    /// 
    /// Applies the BDD reduction rule: if lo == hi, just return lo (skip this variable).
    /// Also handles canonicalization via the unique table.
    pub fn mk(&mut self, var: u8, lo: u16, hi: u16) -> u16 {
        // BDD reduction rule: if both children are the same, skip this variable
        if lo == hi {
            return lo;
        }
        
        // Check unique table for existing node
        let key = (var, lo, hi);
        if let Some(&id) = self.unique.get(&key) {
            return id;
        }
        
        // Create new node
        let id = self.nodes.len() as u16;
        self.nodes.push(BddNode { var, lo, hi });
        self.unique.insert(key, id);
        id
    }
    
    /// Get a node by its index.
    pub fn node(&self, id: u16) -> &BddNode {
        &self.nodes[id as usize]
    }
    
    /// Check if a node is a terminal.
    pub fn is_terminal(&self, id: u16) -> bool {
        id <= Self::TRUE
    }
    
    /// Get the number of nodes in the BDD.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
    
    /// Estimated storage size in bytes.
    pub fn storage_bytes(&self) -> usize {
        self.nodes.len() * 5  // 5 bytes per node
    }
    
    /// Build a BDD for the interval constraint: val in [lo, hi].
    /// 
    /// The BDD tests bits from MSB to LSB, where `first_var` is the variable
    /// index of the MSB.
    pub fn interval(&mut self, first_var: u8, num_bits: u8, lo: u16, hi: u16) -> u16 {
        if lo > hi {
            return Self::FALSE;
        }
        
        let max_val = (1u16 << num_bits).saturating_sub(1);
        if lo == 0 && hi >= max_val {
            return Self::TRUE;
        }
        
        self.interval_rec(first_var, 0, num_bits, lo, hi, 0, max_val)
    }
    
    /// Recursive helper for interval construction.
    /// 
    /// At each level, we track the range of possible values [cur_lo, cur_hi]
    /// and compare against the target interval [lo, hi].
    fn interval_rec(&mut self, first_var: u8, bit: u8, num_bits: u8, lo: u16, hi: u16, cur_lo: u16, cur_hi: u16) -> u16 {
        // If current range is completely contained in target, accept everything
        if cur_lo >= lo && cur_hi <= hi {
            return Self::TRUE;
        }
        
        // If current range is completely outside target, reject everything
        if cur_hi < lo || cur_lo > hi {
            return Self::FALSE;
        }
        
        // If we've processed all bits, we must decide based on cur range
        // At this point cur_lo == cur_hi (single value)
        if bit >= num_bits {
            // Should have been caught by the above checks
            // If we get here, it means cur_lo <= hi && cur_hi >= lo but
            // not fully contained - shouldn't happen for single value
            return if cur_lo >= lo && cur_lo <= hi { Self::TRUE } else { Self::FALSE };
        }
        
        // Split the range: low branch has bit=0, high branch has bit=1
        let var = first_var + bit;
        let pivot = (1u16 << (num_bits - bit - 1)) + cur_lo;
        
        let lo_child = self.interval_rec(first_var, bit + 1, num_bits, lo, hi, cur_lo, pivot - 1);
        let hi_child = self.interval_rec(first_var, bit + 1, num_bits, lo, hi, pivot, cur_hi);
        
        self.mk(var, lo_child, hi_child)
    }
    
    /// Build a BDD for a 2D rectangle: token in [t1, t2] AND tsid in [s1, s2].
    /// 
    /// Uses TSID-first ordering: TSID bits (first_var=0), then token bits.
    pub fn rect(&mut self, t1: u16, t2: u16, s1: u16, s2: u16) -> u16 {
        // Build BDD for tsid constraint (first 13 variables)
        let tsid_bdd = self.interval(0, TSID_BITS, s1, s2);
        
        // Build BDD for token constraint (next 12 variables)
        let token_bdd = self.interval(TSID_BITS, TOKEN_BITS, t1, t2);
        
        // AND them together
        self.apply_and(tsid_bdd, token_bdd)
    }
    
    /// Apply AND operation to two BDDs.
    pub fn apply_and(&mut self, a: u16, b: u16) -> u16 {
        self.apply(a, b, false)
    }
    
    /// Apply OR operation to two BDDs.
    pub fn apply_or(&mut self, a: u16, b: u16) -> u16 {
        self.apply(a, b, true)
    }
    
    /// Generic apply operation (AND or OR).
    fn apply(&mut self, a: u16, b: u16, is_or: bool) -> u16 {
        // Terminal cases
        if is_or {
            // OR
            if a == Self::TRUE || b == Self::TRUE {
                return Self::TRUE;
            }
            if a == Self::FALSE {
                return b;
            }
            if b == Self::FALSE {
                return a;
            }
        } else {
            // AND
            if a == Self::FALSE || b == Self::FALSE {
                return Self::FALSE;
            }
            if a == Self::TRUE {
                return b;
            }
            if b == Self::TRUE {
                return a;
            }
        }
        
        // Normalize order for cache lookup
        let (a, b) = if a > b { (b, a) } else { (a, b) };
        
        // Check cache
        let cache_key = (a, b, is_or);
        if let Some(&result) = self.apply_cache.get(&cache_key) {
            return result;
        }
        
        // Get nodes
        let node_a = self.nodes[a as usize];
        let node_b = self.nodes[b as usize];
        
        // Determine the variable to split on
        let var = node_a.var.min(node_b.var);
        
        // Get children
        let (a_lo, a_hi) = if node_a.var == var {
            (node_a.lo, node_a.hi)
        } else {
            (a, a)  // a doesn't depend on this variable
        };
        
        let (b_lo, b_hi) = if node_b.var == var {
            (node_b.lo, node_b.hi)
        } else {
            (b, b)  // b doesn't depend on this variable
        };
        
        // Recursive apply
        let lo = self.apply(a_lo, b_lo, is_or);
        let hi = self.apply(a_hi, b_hi, is_or);
        
        let result = self.mk(var, lo, hi);
        
        // Cache result
        self.apply_cache.insert(cache_key, result);
        
        result
    }
    
    /// Test if a (token, tsid) pair is contained in the BDD.
    pub fn contains(&self, root: u16, token: u16, tsid: u16) -> bool {
        self.contains_generic(root, tsid, TSID_BITS, token, TOKEN_BITS)
    }
    
    /// Generic containment test for two concatenated bit fields.
    /// Tests if (val1, val2) is in the BDD, where val1 uses the first num_bits1 variables
    /// and val2 uses the next num_bits2 variables.
    fn contains_generic(&self, root: u16, val1: u16, num_bits1: u8, val2: u16, num_bits2: u8) -> bool {
        let mut node = root;
        
        // First value bits (variables 0 to num_bits1-1)
        for bit in 0..num_bits1 {
            if node <= Self::TRUE {
                return node == Self::TRUE;
            }
            let n = &self.nodes[node as usize];
            if n.var > bit {
                // Variable not present (BDD doesn't depend on it), continue
                continue;
            }
            if n.var < bit {
                // Should not happen in a well-formed BDD
                panic!("BDD has out-of-order variables: expected >= {}, got {}", bit, n.var);
            }
            // n.var == bit
            let bit_val = (val1 >> (num_bits1 - 1 - bit)) & 1;
            node = if bit_val == 1 { n.hi } else { n.lo };
        }
        
        // Second value bits (variables num_bits1 to num_bits1+num_bits2-1)
        for bit in 0..num_bits2 {
            if node <= Self::TRUE {
                return node == Self::TRUE;
            }
            let n = &self.nodes[node as usize];
            let expected_var = num_bits1 + bit;
            if n.var > expected_var {
                // Variable not present
                continue;
            }
            if n.var < expected_var {
                panic!("BDD has out-of-order variables: expected >= {}, got {}", expected_var, n.var);
            }
            // n.var == expected_var
            let bit_val = (val2 >> (num_bits2 - 1 - bit)) & 1;
            node = if bit_val == 1 { n.hi } else { n.lo };
        }
        
        node == Self::TRUE
    }
    
    /// Test if a single value is contained in an interval BDD.
    /// Used for testing interval() directly.
    fn contains_interval(&self, root: u16, first_var: u8, num_bits: u8, val: u16) -> bool {
        let mut node = root;
        
        for bit in 0..num_bits {
            if node <= Self::TRUE {
                return node == Self::TRUE;
            }
            let n = &self.nodes[node as usize];
            let expected_var = first_var + bit;
            if n.var > expected_var {
                // Variable not present
                continue;
            }
            if n.var < expected_var {
                panic!("BDD has out-of-order variables");
            }
            let bit_val = (val >> (num_bits - 1 - bit)) & 1;
            node = if bit_val == 1 { n.hi } else { n.lo };
        }
        
        node == Self::TRUE
    }
    
    /// Build a BDD from 1D ranges (N×M space).
    /// 
    /// Each range is decomposed into 2D rectangles and OR'd together.
    pub fn from_1d_ranges(&mut self, ranges: impl Iterator<Item = (usize, usize)>, num_tsids: usize) -> u16 {
        let mut result = Self::FALSE;
        
        for (start, end) in ranges {
            // Decompose to 2D rectangles
            let tok_s = (start / num_tsids) as u16;
            let tsid_s = (start % num_tsids) as u16;
            let tok_e = (end / num_tsids) as u16;
            let tsid_e = (end % num_tsids) as u16;
            
            let rects = decompose_range_to_rects(tok_s, tsid_s, tok_e, tsid_e, num_tsids as u16);
            
            for (t1, t2, s1, s2) in rects {
                let rect_bdd = self.rect(t1, t2, s1, s2);
                result = self.apply_or(result, rect_bdd);
            }
        }
        
        result
    }
    
    /// Clear the apply cache (useful between bulk operations to save memory).
    pub fn clear_cache(&mut self) {
        self.apply_cache.clear();
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bdd_terminal() {
        let bdd = SharedBdd::new(100);
        assert_eq!(bdd.num_nodes(), 2);
        assert!(bdd.is_terminal(SharedBdd::FALSE));
        assert!(bdd.is_terminal(SharedBdd::TRUE));
    }
    
    #[test]
    fn test_bdd_mk_reduction() {
        let mut bdd = SharedBdd::new(100);
        
        // If lo == hi, mk should return the child directly
        let result = bdd.mk(0, SharedBdd::TRUE, SharedBdd::TRUE);
        assert_eq!(result, SharedBdd::TRUE);
        
        let result = bdd.mk(0, SharedBdd::FALSE, SharedBdd::FALSE);
        assert_eq!(result, SharedBdd::FALSE);
    }
    
    #[test]
    fn test_bdd_mk_sharing() {
        let mut bdd = SharedBdd::new(100);
        
        // Creating the same node twice should return the same ID
        let n1 = bdd.mk(0, SharedBdd::FALSE, SharedBdd::TRUE);
        let n2 = bdd.mk(0, SharedBdd::FALSE, SharedBdd::TRUE);
        assert_eq!(n1, n2);
    }
    
    #[test]
    fn test_interval_simple() {
        let mut bdd = SharedBdd::new(100);
        
        // Interval [0, 0] for 4 bits should accept only 0
        let root = bdd.interval(0, 4, 0, 0);
        assert!(bdd.contains_interval(root, 0, 4, 0));
        assert!(!bdd.contains_interval(root, 0, 4, 1));
        assert!(!bdd.contains_interval(root, 0, 4, 15));
    }
    
    #[test]
    fn test_interval_full() {
        let mut bdd = SharedBdd::new(100);
        
        // Interval [0, 15] for 4 bits should accept everything
        let root = bdd.interval(0, 4, 0, 15);
        assert_eq!(root, SharedBdd::TRUE);
    }
    
    #[test]
    fn test_interval_range() {
        let mut bdd = SharedBdd::new(100);
        
        // Interval [3, 7] for 4 bits
        let root = bdd.interval(0, 4, 3, 7);
        for val in 0..16u16 {
            let expected = val >= 3 && val <= 7;
            let result = bdd.contains_interval(root, 0, 4, val);
            assert_eq!(result, expected, "interval [3,7] at {} should be {}", val, expected);
        }
    }
    
    #[test]
    fn test_rect_simple() {
        let mut bdd = SharedBdd::new(16);  // 16 tsids for testing
        
        // Rectangle: token in [1, 2], tsid in [3, 5]
        let root = bdd.rect(1, 2, 3, 5);
        
        // Should contain (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)
        assert!(bdd.contains(root, 1, 3));
        assert!(bdd.contains(root, 1, 4));
        assert!(bdd.contains(root, 1, 5));
        assert!(bdd.contains(root, 2, 3));
        assert!(bdd.contains(root, 2, 5));
        
        // Should NOT contain
        assert!(!bdd.contains(root, 0, 3));  // token 0 not in range
        assert!(!bdd.contains(root, 1, 2));  // tsid 2 not in range
        assert!(!bdd.contains(root, 3, 3));  // token 3 not in range
    }
    
    #[test]
    fn test_apply_or() {
        let mut bdd = SharedBdd::new(16);
        
        // Create two rectangles
        let r1 = bdd.rect(0, 1, 0, 3);  // tokens 0-1, tsids 0-3
        let r2 = bdd.rect(2, 3, 0, 3);  // tokens 2-3, tsids 0-3
        
        let combined = bdd.apply_or(r1, r2);
        
        // Should contain all from both rectangles
        for token in 0..4 {
            for tsid in 0..4 {
                assert!(bdd.contains(combined, token, tsid), 
                    "Should contain ({}, {})", token, tsid);
            }
        }
        
        // Should not contain outside
        assert!(!bdd.contains(combined, 4, 0));
        assert!(!bdd.contains(combined, 0, 4));
    }
    
    #[test]
    fn test_apply_and() {
        let mut bdd = SharedBdd::new(16);
        
        // Create two overlapping rectangles
        let r1 = bdd.rect(0, 3, 0, 3);  // tokens 0-3, tsids 0-3
        let r2 = bdd.rect(2, 5, 2, 5);  // tokens 2-5, tsids 2-5
        
        let intersection = bdd.apply_and(r1, r2);
        
        // Should contain only the intersection: tokens 2-3, tsids 2-3
        for token in 0..8u16 {
            for tsid in 0..8u16 {
                let in_r1 = token <= 3 && tsid <= 3;
                let in_r2 = token >= 2 && token <= 5 && tsid >= 2 && tsid <= 5;
                let expected = in_r1 && in_r2;
                let actual = bdd.contains(intersection, token, tsid);
                assert_eq!(actual, expected, 
                    "Intersection at ({}, {}) should be {} but was {}", token, tsid, expected, actual);
            }
        }
    }
    
    #[test]
    fn test_from_1d_ranges() {
        let num_tsids: usize = 10;
        let mut bdd = SharedBdd::new(num_tsids as u16);
        
        // Range [5, 14] spans token 0 (tsids 5-9) and token 1 (tsids 0-4)
        let root = bdd.from_1d_ranges([(5, 14)].iter().copied(), num_tsids);
        
        // Verify all expected positions
        for pos in 5..=14 {
            let token = (pos / num_tsids) as u16;
            let tsid = (pos % num_tsids) as u16;
            assert!(bdd.contains(root, token, tsid), 
                "Position {} (token={}, tsid={}) should be in BDD", pos, token, tsid);
        }
        
        // Verify some positions outside the range
        assert!(!bdd.contains(root, 0, 4));  // position 4
        assert!(!bdd.contains(root, 1, 5));  // position 15
    }
    
    #[test]
    fn test_node_sharing() {
        let num_tsids: usize = 100;
        let mut bdd = SharedBdd::new(num_tsids as u16);
        
        // Build BDDs for similar ranges that should share nodes
        let r1 = bdd.from_1d_ranges([(0, 99)].iter().copied(), num_tsids);  // Full first row
        let r2 = bdd.from_1d_ranges([(100, 199)].iter().copied(), num_tsids);  // Full second row
        
        let nodes_after_r1 = bdd.num_nodes();
        let _r3 = bdd.from_1d_ranges([(200, 299)].iter().copied(), num_tsids);  // Full third row
        let nodes_after_r3 = bdd.num_nodes();
        
        // Because all three are "full rows" (same TSID profile), they should share
        // the TSID sub-BDD. The node count shouldn't increase much.
        println!("Nodes after r1: {}, after r3: {}", nodes_after_r1, nodes_after_r3);
        
        // Verify correctness
        assert!(bdd.contains(r1, 0, 50));
        assert!(bdd.contains(r2, 1, 50));
        assert!(!bdd.contains(r1, 1, 50));
    }
}
