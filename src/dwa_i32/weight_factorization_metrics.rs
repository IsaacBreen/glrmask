#![allow(dead_code)]

//! Weight Factorization Metrics
//!
//! Analyzes unique weights in a DWA to determine factorization potential using
//! a Greedy Dictionary Compression (Re-Pair style) approach.
//!
//! Key Algorithm:
//! 1. **Initial Basis**: Unique Ranges (interned from all weights).
//! 2. **Initial Map**: Weights -> Sets of Range IDs.
//! 3. **Greedy Merge**: Iteratively find pair (A, B) that appears most frequently.
//! 4. **Update**: Create new basis element C = A U B. Replace (A, B) with C in weights.
//! 5. **Cost Metric**: Total Ranges = Ranges(Basis) + sum(len(MappedWeight)).
//!
//! Enable with `WEIGHT_FACTORIZATION_METRICS=1`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ptr;

use range_set_blaze::{RangeSetBlaze, CheckSortedDisjoint};

use crate::dwa_i32::{DWA, Weight};

/// Computes partition atoms (for debugging/detailed stats only)
fn compute_partition_atoms(weights: &[&RangeSetBlaze<usize>]) -> Vec<(usize, usize)> {
    let mut endpoints: BTreeSet<usize> = BTreeSet::new();
    endpoints.insert(0);
    
    for w in weights {
        for range in w.ranges() {
            endpoints.insert(*range.start());
            endpoints.insert(range.end().saturating_add(1));
        }
    }
    
    let endpoints_vec: Vec<usize> = endpoints.into_iter().collect();
    let mut atoms = Vec::new();
    for window in endpoints_vec.windows(2) {
        if window[0] < window[1] {
            atoms.push((window[0], window[1]));
        }
    }
    atoms
}

/// Check if a weight contains a given atom (half-open interval [lo, hi))
fn weight_contains_atom(weight: &RangeSetBlaze<usize>, lo: usize, _hi: usize) -> bool {
    weight.contains(lo)
}

/// Print factorization metrics for a DWA's unique weights.
pub fn maybe_print_dwa_weight_factorization_metrics(dwa: &DWA, name: &str) {
    if std::env::var("WEIGHT_FACTORIZATION_METRICS")
        .map(|v| v != "1")
        .unwrap_or(true)
    {
        return;
    }
    
    // Collect unique weights
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
    if unique_weights.is_empty() { return; }

    // Log basic stats
    let total_ranges: usize = unique_weights.iter().map(|w| w.num_ranges()).sum();
    crate::debug!(5, "[WEIGHT_FACTORIZATION_METRICS] {}: unique_weights={} total_ranges={}", 
        name, unique_weights.len(), total_ranges);

    // Run verified greedy analysis
    analyze_greedy_factorization(&unique_weights, name);
}

/// Run Verified Greedy Dictionary Factorization (Range-Based).
fn analyze_greedy_factorization(unique_weights: &[Weight], name: &str) {
    // eprintln!("DEBUG: analyze_greedy_factorization start for {}", name);
    if unique_weights.is_empty() { return; }

    // 1. Intern all ranges
    let mut range_to_id: HashMap<(usize, usize), usize> = HashMap::new();
    let mut id_to_range: Vec<(usize, usize)> = Vec::new();
    let mut next_id = 0;
    
    // Mapped weights: Sets of basis IDs
    let mut mapped_weights: Vec<BTreeSet<usize>> = Vec::with_capacity(unique_weights.len());
    
    // eprintln!("DEBUG: starting intern loop");
    for w in unique_weights {
        let mut entry = BTreeSet::new();
        for range in w.rsb.ranges() {
            let r = (*range.start(), *range.end());
            let id = if let Some(&id) = range_to_id.get(&r) {
                id
            } else {
                range_to_id.insert(r, next_id);
                id_to_range.push(r);
                let id = next_id;
                next_id += 1;
                id
            };
            entry.insert(id);
        }
        mapped_weights.push(entry);
    }
    // eprintln!("DEBUG: intern loop done, found {} unique ranges", id_to_range.len());
    
    // Initial Basis: Single ranges (constructed efficiently O(1))
    let mut basis: Vec<RangeSetBlaze<usize>> = id_to_range.iter()
        .map(|&(s, e)| RangeSetBlaze::from_sorted_disjoint(CheckSortedDisjoint::new(std::iter::once(s..=e))))
        .collect();
    
    let original_total_ranges: usize = unique_weights.iter().map(|w| w.num_ranges()).sum();
    
    // Cost calculation
    let mut current_basis_ranges: usize = basis.iter().map(|b| b.ranges().count()).sum();
    let mut current_mapping_refs: usize = mapped_weights.iter().map(|w| w.len()).sum();
    let start_total_cost = current_basis_ranges + current_mapping_refs;
    
    // eprintln!("DEBUG: Cost calc done. Printing debug log.");

    crate::debug!(5, "[GREEDY_FACTORIZATION] {} Start: original={} start_cost={} (basis={} + map={}) unique_ranges={}", 
        name, original_total_ranges, start_total_cost, current_basis_ranges, current_mapping_refs, basis.len());

    // Greedy Loop
    let max_iterations = 2000; 
    
    for _iter in 0..max_iterations {
        let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();
        
        for w in &mapped_weights {
            if w.len() < 2 { continue; }
            let elements: Vec<usize> = w.iter().copied().collect();
            // Optimization: Only count ADJACENT pairs in the current mapping.
            // This is O(L) instead of O(L^2), preventing huge weights from stalling the analysis.
            for i in 0..(elements.len() - 1) {
                let pair = (elements[i], elements[i+1]);
                *pair_counts.entry(pair).or_default() += 1;
            }
        }
        
        if pair_counts.is_empty() { break; }
        
        let mut best_pair = None;
        let mut best_net_savings: isize = std::isize::MIN;
        
        for (&(a, b), &count) in pair_counts.iter() {
            if count < 2 { continue; }
            
            let c_rsb = &basis[a] | &basis[b];
            let c_ranges = c_rsb.ranges().count();
            
            // Savings: (count refs removed) - (basis ranges added)
            let net_savings = (count as isize) - (c_ranges as isize);
            
            if net_savings > best_net_savings {
                best_net_savings = net_savings;
                best_pair = Some((a, b));
            }
        }
        
        if let Some((a, b)) = best_pair {
            if best_net_savings <= 0 { break; }
            
            let new_id = basis.len();
            let new_rsb = &basis[a] | &basis[b];
            basis.push(new_rsb);
            
            for w in &mut mapped_weights {
                if w.contains(&a) && w.contains(&b) {
                    w.remove(&a);
                    w.remove(&b);
                    w.insert(new_id);
                }
            }
        } else {
            break;
        }
    }
    
    // Final Cost
    let mut used_indices = BTreeSet::new();
    for w in &mapped_weights {
        for &idx in w { used_indices.insert(idx); }
    }
    
    let final_basis_ranges: usize = used_indices.iter().map(|&i| basis[i].ranges().count()).sum();
    let final_mapping_refs: usize = mapped_weights.iter().map(|w| w.len()).sum();
    let final_total_cost = final_basis_ranges + final_mapping_refs;
    let ratio = final_total_cost as f64 / original_total_ranges as f64;
    
    crate::debug!(5, 
        "[GREEDY_FACTORIZATION] {} Result: final_cost={} (basis={} map={}) ratio={:.3}",
        name, final_total_cost, final_basis_ranges, final_mapping_refs, ratio
    );

    // Verification
    for (i, w_idxs) in mapped_weights.iter().enumerate() {
        let mut reconstructed = RangeSetBlaze::new();
        for &idx in w_idxs {
            reconstructed |= &basis[idx];
        }
        if reconstructed != unique_weights[i].rsb {
            crate::debug!(1, "[GREEDY_FACTORIZATION] FATAL: Verification failed for weight {}!", i);
            panic!("Greedy factorization verification failed!");
        }
    }
    // eprintln!("DEBUG: analyze_greedy_factorization done for {}", name);
}

// Stub for 2D metrics call
pub fn maybe_print_2d_factorization_metrics(_: &DWA, _: usize, _: usize, _: &str) {}

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
    fn test_greedy_factorization_simple() {
        let w1 = make_weight(&[(0, 10), (100, 200)]);
        let w2 = make_weight(&[(0, 10), (300, 400)]); 
        let weights = vec![w1, w2];
        analyze_greedy_factorization(&weights, "Test");
    }

    #[test]
    fn test_rsb_perf() {
        let start = std::time::Instant::now();
        let s = 0;
        let e = 1_000_000_000;
        // Verify O(1) construction using CheckSortedDisjoint
        let _rsb = RangeSetBlaze::from_sorted_disjoint(CheckSortedDisjoint::new(std::iter::once(s..=e)));
        let duration = start.elapsed();
        println!("RSB construction took {:?}", duration);
        assert!(duration.as_millis() < 100, "Construction too slow: {:?}", duration);
    }
}
