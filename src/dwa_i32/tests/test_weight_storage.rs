//! Integration tests for weight storage benchmarking.
//!
//! These tests measure the performance of different weight storage implementations
//! on realistic data extracted from DWAs.

use super::*;
use crate::dwa_i32::weight_storage::*;
use crate::dwa_i32::rangeset::RangeSet;
use std::time::Instant;

/// Extract unique weights from a DWA and measure storage implementations.
pub fn benchmark_dwa_weights(dwa: &DWA, name: &str) {
    // Extract unique weights
    let mut unique_weights: Vec<RangeSet> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    
    for state in &dwa.states.0 {
        if let Some(fw) = &state.final_weight {
            if !fw.is_empty() && !fw.is_all_fast() {
                let key = fw.rsb.ranges().map(|r| (*r.start(), *r.end())).collect::<Vec<_>>();
                if !seen.contains(&key) {
                    seen.insert(key);
                    unique_weights.push(fw.clone());
                }
            }
        }
        
        for tw in state.trans_weights.values() {
            if !tw.is_empty() && !tw.is_all_fast() {
                let key = tw.rsb.ranges().map(|r| (*r.start(), *r.end())).collect::<Vec<_>>();
                if !seen.contains(&key) {
                    seen.insert(key);
                    unique_weights.push(tw.clone());
                }
            }
        }
    }
    
    println!("\n=== {} Weight Storage Benchmark ===", name);
    println!("Unique weights: {}", unique_weights.len());
    
    // Get dimensions from DWA
    let dims = dwa.dims;
    println!("Dimensions: {} tokens × {} tsids", dims.num_tokens, dims.num_tsids);
    
    // Benchmark RangeSet storage
    let rs_factory = RangeSetStorageFactory::new(dims.num_tsids, dims.num_tokens);
    let mut rs_metrics = StorageMetrics::new("RangeSet");
    
    let start = Instant::now();
    for w in &unique_weights {
        let storage = rs_factory.from_rangeset(w);
        rs_metrics.add(&storage);
    }
    rs_metrics.construction_time_us = start.elapsed().as_micros() as u64;
    
    // Benchmark BDD storage
    let bdd_factory = BddStorageFactory::new(dims.num_tsids, dims.num_tokens);
    let mut bdd_metrics = StorageMetrics::new("PerWeightBDD");
    
    let start = Instant::now();
    for w in &unique_weights {
        let storage = bdd_factory.from_rangeset(w);
        bdd_metrics.add(&storage);
    }
    bdd_metrics.construction_time_us = start.elapsed().as_micros() as u64;
    
    // Print metrics
    rs_metrics.print_summary();
    println!();
    bdd_metrics.print_summary();
    
    // Compression ratio
    if bdd_metrics.total_bytes > 0 {
        let ratio = rs_metrics.total_bytes as f64 / bdd_metrics.total_bytes as f64;
        println!("\nCompression: RangeSet/BDD = {:.2}x", ratio);
    }
    
    // Verify correctness (sample a few weights)
    let sample_size = unique_weights.len().min(10);
    let mut all_correct = true;
    
    for (i, w) in unique_weights.iter().take(sample_size).enumerate() {
        let rs = rs_factory.from_rangeset(w);
        let bdd = bdd_factory.from_rangeset(w);
        
        if !storages_equal(&rs, &bdd) {
            println!("ERROR: Weight {} differs between storages!", i);
            all_correct = false;
        }
    }
    
    if all_correct {
        println!("Correctness check: PASSED ({} weights sampled)", sample_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    /// Helper to create a test DWA with various weight patterns.
    fn create_test_dwa() -> DWA {
        use crate::dwa_i32::dwa::{DWABody, DWAStates};
        use crate::dwa_i32::heavy_weight::WeightDimensions;
        
        let mut states = DWAStates::default();
        
        // Create 10 states with different weight patterns
        for i in 0..10 {
            let sid = states.add_state();
            
            // Final weight: different pattern per state
            let final_ranges: Vec<usize> = ((i * 100)..((i + 1) * 100))
                .step_by(3)
                .collect();
            if !final_ranges.is_empty() {
                states[sid].final_weight = Some(RangeSet::from_iter(final_ranges));
            }
            
            // Transition weights
            if i < 9 {
                let trans_ranges: Vec<usize> = ((i * 50)..((i + 2) * 50))
                    .step_by(2)
                    .collect();
                states[sid].transitions.insert(i as i32, sid + 1);
                states[sid].trans_weights.insert(i as i32, RangeSet::from_iter(trans_ranges));
            }
        }
        
        DWA {
            states,
            body: DWABody { start_state: 0 },
            dims: WeightDimensions::new(100, 100),  // 100 tokens × 100 tsids
        }
    }
    
    #[test]
    fn test_benchmark_small_dwa() {
        let dwa = create_test_dwa();
        benchmark_dwa_weights(&dwa, "SmallTest");
    }
}
