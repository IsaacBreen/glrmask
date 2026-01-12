#![allow(dead_code)]

//! Weight Factorization Metrics
//!
//! Analyzes unique weights in a DWA to determine factorization potential.
//! This helps evaluate whether a weight optimization mapping could reduce
//! the total range count across all weights.
//!
//! Key concepts:
//! - **Partition atoms**: Atomic intervals formed by all range endpoints
//! - **Sharing ratio**: How many weights contain each atom
//! - **Theoretical minimum basis**: Lower bound on factorized representation
//!
//! Enable with `WEIGHT_FACTORIZATION_METRICS=1`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ptr;

use range_set_blaze::RangeSetBlaze;

use crate::dwa_i32::{DWA, Weight};

/// Statistics about weight factorization potential.
#[derive(Debug, Clone)]
pub struct FactorizationStats {
    /// Number of unique weights analyzed
    pub num_unique_weights: usize,
    /// Total ranges across all unique weights (baseline)
    pub total_ranges: usize,
    /// Number of partition atoms (atomic intervals from all endpoints)
    pub num_atoms: usize,
    /// Max overlap: maximum number of weights containing the same atom
    pub max_overlap: usize,
    /// Atoms with high overlap (>10% of weights)
    pub high_overlap_atoms: usize,
    /// Theoretical minimum ranges if perfectly factorized
    /// (each atom appears in basis once, weights just refer to it)
    pub theoretical_min_ranges: usize,
    /// Compression ratio: theoretical_min / total_ranges
    pub compression_ratio: f64,
    /// Number of distinct "patterns" (unique atom-bitmasks across weights)
    pub num_distinct_patterns: usize,
}

/// Computes partition atoms from a set of RangeSetBlaze weights.
/// 
/// Returns a sorted list of atomic intervals derived from all range endpoints.
/// Each atom [a, b) is such that every weight either fully contains it or doesn't overlap at all.
fn compute_partition_atoms(weights: &[&RangeSetBlaze<usize>]) -> Vec<(usize, usize)> {
    // Collect all endpoints
    let mut endpoints: BTreeSet<usize> = BTreeSet::new();
    endpoints.insert(0); // Include 0 as a boundary
    
    for w in weights {
        for range in w.ranges() {
            endpoints.insert(*range.start());
            endpoints.insert(range.end().saturating_add(1)); // Exclusive end
        }
    }
    
    // Convert to sorted vec
    let endpoints_vec: Vec<usize> = endpoints.into_iter().collect();
    
    // Create atoms from consecutive pairs
    let mut atoms = Vec::new();
    for window in endpoints_vec.windows(2) {
        if window[0] < window[1] {
            atoms.push((window[0], window[1]));
        }
    }
    
    atoms
}

/// Check if a weight contains a given atom (half-open interval [lo, hi))
fn weight_contains_atom(weight: &RangeSetBlaze<usize>, lo: usize, hi: usize) -> bool {
    // An atom is contained if the weight contains all points in [lo, hi)
    // Since atoms are derived from endpoints, we just check if lo is in the weight
    weight.contains(lo)
}

/// Analyze factorization potential for a set of weights.
pub fn analyze_factorization(unique_weights: &[Weight]) -> FactorizationStats {
    if unique_weights.is_empty() {
        return FactorizationStats {
            num_unique_weights: 0,
            total_ranges: 0,
            num_atoms: 0,
            max_overlap: 0,
            high_overlap_atoms: 0,
            theoretical_min_ranges: 0,
            compression_ratio: 1.0,
            num_distinct_patterns: 0,
        };
    }
    
    let num_unique_weights = unique_weights.len();
    let total_ranges: usize = unique_weights.iter().map(|w| w.num_ranges()).sum();
    
    // Get references to inner RangeSetBlaze
    let rsbs: Vec<&RangeSetBlaze<usize>> = unique_weights.iter().map(|w| &w.rsb).collect();
    
    // Compute partition atoms
    let atoms = compute_partition_atoms(&rsbs);
    let num_atoms = atoms.len();
    
    if num_atoms == 0 {
        return FactorizationStats {
            num_unique_weights,
            total_ranges,
            num_atoms: 0,
            max_overlap: 0,
            high_overlap_atoms: 0,
            theoretical_min_ranges: total_ranges,
            compression_ratio: 1.0,
            num_distinct_patterns: 0,
        };
    }
    
    // Count how many weights contain each atom
    let mut atom_overlap: Vec<usize> = vec![0; num_atoms];
    
    // Also track each weight's "pattern" (which atoms it contains)
    let mut weight_patterns: Vec<Vec<bool>> = Vec::with_capacity(num_unique_weights);
    
    for weight in &rsbs {
        let mut pattern = Vec::with_capacity(num_atoms);
        for (i, &(lo, hi)) in atoms.iter().enumerate() {
            let contains = weight_contains_atom(weight, lo, hi);
            pattern.push(contains);
            if contains {
                atom_overlap[i] += 1;
            }
        }
        weight_patterns.push(pattern);
    }
    
    // Compute statistics
    let max_overlap = atom_overlap.iter().copied().max().unwrap_or(0);
    let high_overlap_threshold = (num_unique_weights as f64 * 0.1).ceil() as usize;
    let high_overlap_atoms = atom_overlap.iter().filter(|&&c| c > high_overlap_threshold).count();
    
    // Theoretical minimum: each atom that appears in ANY weight contributes 1 range to basis
    // This is a lower bound if we can perfectly factor
    let atoms_used = atom_overlap.iter().filter(|&&c| c > 0).count();
    let theoretical_min_ranges = atoms_used;
    
    let compression_ratio = if total_ranges > 0 {
        theoretical_min_ranges as f64 / total_ranges as f64
    } else {
        1.0
    };
    
    // Count distinct patterns
    let mut pattern_set: BTreeSet<Vec<bool>> = BTreeSet::new();
    for pattern in weight_patterns {
        pattern_set.insert(pattern);
    }
    let num_distinct_patterns = pattern_set.len();
    
    FactorizationStats {
        num_unique_weights,
        total_ranges,
        num_atoms,
        max_overlap,
        high_overlap_atoms,
        theoretical_min_ranges,
        compression_ratio,
        num_distinct_patterns,
    }
}

/// Print factorization metrics for a DWA's unique weights.
/// Enabled only when `WEIGHT_FACTORIZATION_METRICS=1`.
pub fn maybe_print_dwa_weight_factorization_metrics(dwa: &DWA, name: &str) {
    if std::env::var("WEIGHT_FACTORIZATION_METRICS")
        .map(|v| v != "1")
        .unwrap_or(true)
    {
        return;
    }
    
    // Collect unique weights by Arc pointer address
    let mut unique: HashMap<usize, Weight> = HashMap::new();
    
    for state in &dwa.states.0 {
        if let Some(fw) = &state.final_weight {
            let p = ptr::addr_of!(**fw) as usize;
            unique.entry(p).or_insert_with(|| fw.clone());
        }
        for w in state.trans_weights.values() {
            let p = ptr::addr_of!(**w) as usize;
            unique.entry(p).or_insert_with(|| w.clone());
        }
    }
    
    let unique_weights: Vec<Weight> = unique.into_values().collect();
    let stats = analyze_factorization(&unique_weights);
    
    crate::debug!(5,
        "[WEIGHT_FACTORIZATION_METRICS] {}: unique_weights={} total_ranges={} num_atoms={} max_overlap={} high_overlap_atoms={} theoretical_min={} compression_ratio={:.3} distinct_patterns={}",
        name,
        stats.num_unique_weights,
        stats.total_ranges,
        stats.num_atoms,
        stats.max_overlap,
        stats.high_overlap_atoms,
        stats.theoretical_min_ranges,
        stats.compression_ratio,
        stats.num_distinct_patterns,
    );
    
    // Additional detailed analysis at debug level 6
    if crate::r#macro::is_debug_level_enabled(6) {
        // Show distribution of atom overlaps
        let mut overlap_histogram: BTreeMap<usize, usize> = BTreeMap::new();
        
        let rsbs: Vec<&RangeSetBlaze<usize>> = unique_weights.iter().map(|w| &w.rsb).collect();
        let atoms = compute_partition_atoms(&rsbs);
        
        for weight in &rsbs {
            for &(lo, hi) in &atoms {
                if weight_contains_atom(weight, lo, hi) {
                    *overlap_histogram.entry(1).or_default() += 1;
                }
            }
        }
        
        // Count overlap distribution
        let mut atom_counts: Vec<usize> = Vec::new();
        for weight in &rsbs {
            let mut count = 0;
            for &(lo, _hi) in &atoms {
                if weight_contains_atom(weight, lo, _hi) {
                    count += 1;
                }
            }
            atom_counts.push(count);
        }
        
        let avg_atoms_per_weight = if !atom_counts.is_empty() {
            atom_counts.iter().sum::<usize>() as f64 / atom_counts.len() as f64
        } else {
            0.0
        };
        
        crate::debug!(6,
            "[WEIGHT_FACTORIZATION_METRICS] {} detail: avg_atoms_per_weight={:.2} ranges_per_weight={:.2}",
            name,
            avg_atoms_per_weight,
            if stats.num_unique_weights > 0 { stats.total_ranges as f64 / stats.num_unique_weights as f64 } else { 0.0 },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_weight(ranges: &[(usize, usize)]) -> Weight {
        let mut rsb = RangeSetBlaze::new();
        for &(start, end) in ranges {
            rsb |= RangeSetBlaze::from_iter(start..=end);
        }
        Weight::from_rsb(rsb)
    }
    
    #[test]
    fn test_partition_atoms_simple() {
        let w1 = make_weight(&[(0, 10)]);
        let w2 = make_weight(&[(5, 15)]);
        let rsbs: Vec<&RangeSetBlaze<usize>> = vec![&w1.rsb, &w2.rsb];
        
        let atoms = compute_partition_atoms(&rsbs);
        
        // Endpoints: 0, 5, 11, 16
        // Atoms: [0,5), [5,11), [11,16)
        assert_eq!(atoms.len(), 3);
        assert_eq!(atoms[0], (0, 5));
        assert_eq!(atoms[1], (5, 11));
        assert_eq!(atoms[2], (11, 16));
    }
    
    #[test]
    fn test_analyze_factorization_sharing() {
        // Two weights with a common prefix
        let w1 = make_weight(&[(0, 10), (100, 200)]);
        let w2 = make_weight(&[(0, 10), (300, 400)]);
        let weights = vec![w1, w2];
        
        let stats = analyze_factorization(&weights);
        
        assert_eq!(stats.num_unique_weights, 2);
        assert_eq!(stats.total_ranges, 4); // 2 ranges each
        // The [0,10] atom is shared, so max_overlap should be 2
        assert!(stats.max_overlap >= 2);
        // Theoretical min should be 3 (three distinct contiguous regions: 0-10, 100-200, 300-400)
        assert!(stats.theoretical_min_ranges <= stats.total_ranges);
    }
    
    #[test]
    fn test_analyze_factorization_no_sharing() {
        // Two weights with no overlap
        let w1 = make_weight(&[(0, 10)]);
        let w2 = make_weight(&[(100, 110)]);
        let weights = vec![w1, w2];
        
        let stats = analyze_factorization(&weights);
        
        assert_eq!(stats.num_unique_weights, 2);
        assert_eq!(stats.total_ranges, 2);
        assert_eq!(stats.max_overlap, 1); // Each atom in only one weight
    }
}
