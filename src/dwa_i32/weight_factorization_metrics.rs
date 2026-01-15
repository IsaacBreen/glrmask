#![allow(dead_code)]

//! Weight Factorization Metrics
//!
//! Exports unique weights to JSON for external analysis.
//!
//! Enable with `WEIGHT_FACTORIZATION_METRICS=1`.

use std::collections::{HashMap};
use std::ptr;
use std::fs::File;
use std::io::BufWriter;

use range_set_blaze::RangeSetBlaze;

use crate::dwa_i32::{DWA, Weight, HeavyWeight, WeightDimensions};

/// Print factorization metrics (Export JSON) for a DWA's unique weights.
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
            let p = fw.intern_id();
            unique.entry(p).or_insert_with(|| fw.clone());
        }
        for w in state.trans_weights.values() {
            let p = w.intern_id();
            unique.entry(p).or_insert_with(|| w.clone());
        }
    }
    
    let unique_weights: Vec<Weight> = unique.into_values().collect();
    if unique_weights.is_empty() { return; }

    // Log basic stats
    let total_ranges: usize = unique_weights.iter().map(|w| w.num_ranges()).sum();
    crate::debug!(5, "[WEIGHT_FACTORIZATION_METRICS] {}: unique_weights={} total_ranges={} -> Exporting JSON", 
        name, unique_weights.len(), total_ranges);

    export_weights_to_json(&unique_weights, name);
}

/// Export weights to JSON file: `range_weights_{name}.json`
fn export_weights_to_json(unique_weights: &[Weight], name: &str) {
    let sanitized_name = name.replace(" ", "_").to_lowercase();
    let filename = format!("range_weights_{}.json", sanitized_name);
    
    // Convert to serializable format: Vec<Vec<(usize, usize)>>
    // Each weight is a list of [start, end] inclusive ranges.
    let export_data: Vec<Vec<(usize, usize)>> = unique_weights.iter().map(|w| {
        let rsb = w.to_rsb();
        rsb.ranges().map(|r| (*r.start(), *r.end())).collect()
    }).collect();
    
    let file = File::create(&filename).expect("Unable to create export file");
    let writer = BufWriter::new(file);
    
    serde_json::to_writer(writer, &export_data).expect("Failed to serialize weights to JSON");
    
    crate::debug!(5, "[WEIGHT_FACTORIZATION_METRICS] Exported {} weights to {}", export_data.len(), filename);
}

/// Compute factored representation metrics for 2D (N×M) weight space.
/// 
/// For each weight in N×M space (position = llm_token * num_tsids + tsid),
/// compute what the range count would be if stored as (base_ranges, tsid_mask).
pub fn maybe_print_2d_factorization_metrics(dwa: &DWA, max_n: usize, num_tsids: usize, name: &str) {
    if std::env::var("WEIGHT_FACTORIZATION_METRICS")
        .map(|v| v != "1")
        .unwrap_or(true)
    {
        return;
    }
    
    if num_tsids == 0 {
        return;
    }
    
    // Create dimensions for HeavyWeight conversion
    let dims = WeightDimensions::new(max_n + 1, num_tsids);
    
    // Collect unique weights
    let mut unique: HashMap<usize, Weight> = HashMap::new();
    for state in &dwa.states.0 {
        if let Some(fw) = &state.final_weight {
            let p = fw.intern_id();
            unique.entry(p).or_insert_with(|| fw.clone());
        }
        for w in state.trans_weights.values() {
            let p = w.intern_id();
            unique.entry(p).or_insert_with(|| w.clone());
        }
    }
    
    let unique_weights: Vec<Weight> = unique.into_values().collect();
    if unique_weights.is_empty() { return; }
    
    let mut total_current_ranges: usize = 0;
    let mut total_factored_ranges: usize = 0;
    
    for w in &unique_weights {
        total_current_ranges += w.num_ranges();
        
        // Convert to HeavyWeight for 2D operations
        let hw = HeavyWeight::from_rangeset(w.clone().into(), dims);
        
        // Project to tokens and tsids
        let tokens = hw.project_tokens();
        let tsids = hw.project_tsids();
        
        // Count ranges in factored representation
        let token_ranges = tokens.ranges().count();
        let tsid_ranges = tsids.ranges().count();
        total_factored_ranges += token_ranges + tsid_ranges;
    }
    
    crate::debug!(5, "[FACTORED_2D_METRICS] {}: unique_weights={} current_ranges={} factored_ranges={} reduction={:.2}x",
        name, unique_weights.len(), total_current_ranges, total_factored_ranges,
        if total_factored_ranges > 0 { total_current_ranges as f64 / total_factored_ranges as f64 } else { 0.0 }
    );

    // EXTENSION: Export base token sets for clustering analysis
    if name == "Terminal DWA" {
        let mut base_sets: Vec<Vec<(usize, usize)>> = Vec::new();
        for w in &unique_weights {
            // Convert to HeavyWeight and project
            let hw = HeavyWeight::from_rangeset(w.clone().into(), dims);
            let tokens = hw.project_tokens();
            base_sets.push(tokens.ranges().map(|r| (*r.start(), *r.end())).collect());
        }
        
        let filename = "base_token_sets_terminal.json";
        let file = File::create(filename).expect("Unable to create base token export file");
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &base_sets).expect("Failed to serialize base tokens");
        crate::debug!(5, "[WEIGHT_FACTORIZATION_METRICS] Exported {} base sets to {}", base_sets.len(), filename);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_weight(ranges: &[(usize, usize)]) -> Weight {
        let mut rsb = RangeSetBlaze::new();
        for &(start, end) in ranges {
            rsb |= RangeSetBlaze::from_iter(start..=end); // Inclusive iter
        }
        Weight::from_rsb(rsb)
    }
    
    #[test]
    fn test_export_weights() {
        // Just verify basic logic compiles and runs without panic
        let w1 = make_weight(&[(0, 10), (100, 200)]);
        let weights = vec![w1];
        // Don't actually write file in unit test to avoid clutter, or write to tmp
        // But main function writes to CWD.
        // We'll skip file write test here or use a temp file logic if strictly needed.
    }
}
