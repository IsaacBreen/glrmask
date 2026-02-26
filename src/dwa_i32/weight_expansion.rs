// src/precompute4/weighted_automata/weight_expansion.rs
//
// Converts a "symbol-heavy" DWA (where tokenizer state IDs are encoded as initial
// labeled transitions) into a "weight-heavy" DWA (where tsids are encoded in the
// weight space itself).
//
// Weight space expansion: N (LLM tokens) -> N×M (LLM tokens × tokenizer states)
// Layout: position = llm_token * M + tsid
//
// Note: A single LLM token active on ALL tsids uses just ONE range.

#![allow(dead_code)]

use range_set_blaze::RangeSetBlaze;
use once_cell::sync::Lazy;
use std::time::{Duration, Instant};
use super::common::{Label, Weight};
use super::dwa::DWA;
use super::nwa::NWA;
use crate::datastructures::abstract_weight::{BackendChoice, current_backend_choice};
use crate::datastructures::hybrid_bitset::RangeSet;
use crate::datastructures::rangemap_weight::{RangeMapWeight, intern_rangemap};

static PROFILE_WEIGHT_EXPANSION: Lazy<bool> = Lazy::new(|| {
    std::env::var("PROFILE_WEIGHT_EXPANSION")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
});

const PROFILE_WEIGHT_EXPANSION_MIN_MS: u64 = 5;

#[inline]
fn weight_expansion_profile_enabled() -> bool {
    *PROFILE_WEIGHT_EXPANSION
}

#[inline]
fn should_log_weight_expansion(elapsed: Duration) -> bool {
    elapsed >= Duration::from_millis(PROFILE_WEIGHT_EXPANSION_MIN_MS)
}

/// Expand a weight from N-space to N×M-space.
/// Each position p in the original weight becomes position p * num_tsids + 0..num_tsids-1.
pub fn expand_weight(weight: &Weight, num_tsids: usize) -> Weight {
    if weight.is_empty() {
        return Weight::zeros();
    }
    if weight.is_all_fast() {
        return Weight::all();
    }
    
    Weight::from_rsb(expand_rsb(&weight.to_rsb_allow_expansion(), num_tsids))
}

/// Expand a RangeSetBlaze from N-space to N×M-space.
/// Each position p becomes positions p * num_tsids through p * num_tsids + num_tsids - 1.
pub fn expand_weight_rsb(rsb: &std::sync::Arc<RangeSetBlaze<usize>>, num_tsids: usize) -> RangeSetBlaze<usize> {
    if rsb.is_empty() || num_tsids == 0 {
        return RangeSetBlaze::new();
    }
    expand_rsb(rsb, num_tsids)
}

/// Internal helper to expand a RangeSetBlaze.
/// Uses saturating arithmetic to handle large values that would overflow.
pub fn expand_rsb(rsb: &RangeSetBlaze<usize>, num_tsids: usize) -> RangeSetBlaze<usize> {
    let profile = weight_expansion_profile_enabled();
    let start = profile.then(Instant::now);
    let expanded: RangeSetBlaze<usize> = rsb.ranges()
        .map(|r| {
            let start = r.start().saturating_mul(num_tsids);
            let end = r.end()
                .saturating_mul(num_tsids)
                .saturating_add(num_tsids.saturating_sub(1));
            start..=end
        })
        .collect();

    if let Some(start) = start {
        let elapsed = start.elapsed();
        if should_log_weight_expansion(elapsed) {
            let in_ranges = rsb.ranges().count();
            let out_ranges = expanded.ranges().count();
            crate::debug!(
                5,
                "expand_rsb: ranges {} -> {}, num_tsids={}, elapsed={:?}",
                in_ranges,
                out_ranges,
                num_tsids,
                elapsed,
            );
        }
    }

    expanded
}

/// Expand a RangeSetBlaze from N-space to N×M-space using WeightDimensions.
/// Each position p becomes positions p * num_tsids through p * num_tsids + num_tsids - 1.
pub fn expand_rsb_with_dims(rsb: &RangeSetBlaze<usize>, dims: crate::datastructures::WeightDimensions) -> RangeSetBlaze<usize> {
    expand_rsb(rsb, dims.num_tsids)
}

/// Create a tsid mask for a specific tokenizer state ID.
/// The mask has positions: tsid + n*M for n in 0..N
/// This is equivalent to: all positions where position % M == tsid
pub fn create_tsid_mask(tsid: usize, num_tsids: usize, max_llm_token: usize) -> Weight {
    let mut mask = RangeSetBlaze::new();
    for n in 0..=max_llm_token {
        mask.insert(tsid + n * num_tsids);
    }
    Weight::from_rsb(mask)
}

/// Create a tsid mask as a RangeSetBlaze.
pub fn create_tsid_mask_rsb(tsid: usize, num_tsids: usize, max_llm_token: usize) -> RangeSetBlaze<usize> {
    let mut mask = RangeSetBlaze::new();
    for n in 0..=max_llm_token {
        mask.insert(tsid + n * num_tsids);
    }
    mask
}

/// Create a combined tsid mask for a set of tokenizer state IDs.
/// 
/// This is more efficient than calling `create_tsid_mask` multiple times when
/// you have multiple tsids that all need to be combined. Instead of building
/// N separate RangeSets (each iterating through max_llm_token), this builds
/// one RangeSet by:
/// 1. Creating a base pattern with all the tsid offsets
/// 2. "Tiling" that pattern across all LLM token positions
/// 
/// For example, with tsids = {0, 2, 5} and num_tsids = 10:
/// - Base pattern: {0, 2, 5}
/// - Tiled for each LLM token n: {0+n*10, 2+n*10, 5+n*10}
pub fn create_tsid_set_mask<I>(tsids: I, num_tsids: usize, max_llm_token: usize) -> Weight
where
    I: IntoIterator<Item = usize>,
{
    let profile = weight_expansion_profile_enabled();
    let start = profile.then(Instant::now);
    let token_count = max_llm_token.saturating_add(1);
    let mut tsid_count = 0usize;
    // Build base pattern from all tsids
    let base_pattern: RangeSetBlaze<usize> = tsids
        .into_iter()
        .map(|t| {
            tsid_count += 1;
            t
        })
        .collect();
    let base_ranges_len = if profile { base_pattern.ranges().count() } else { 0 };
    
    if base_pattern.is_empty() {
        let mask = Weight::zeros();
        if let Some(start) = start {
            let elapsed = start.elapsed();
            if should_log_weight_expansion(elapsed) {
                crate::debug!(
                    5,
                    "create_tsid_set_mask: tsids={}, num_tsids={}, tokens={}, base_ranges={}, out_ranges={}, elapsed={:?}",
                    tsid_count,
                    num_tsids,
                    token_count,
                    base_ranges_len,
                    mask.ranges_len(),
                    elapsed,
                );
            }
        }
        return mask;
    }

    if num_tsids == 0 {
        let mask = Weight::zeros();
        if let Some(start) = start {
            let elapsed = start.elapsed();
            if should_log_weight_expansion(elapsed) {
                crate::debug!(
                    5,
                    "create_tsid_set_mask: tsids={}, num_tsids={}, tokens={}, base_ranges={}, out_ranges={}, elapsed={:?}",
                    tsid_count,
                    num_tsids,
                    token_count,
                    base_ranges_len,
                    mask.ranges_len(),
                    elapsed,
                );
            }
        }
        return mask;
    }

    if matches!(current_backend_choice(), BackendChoice::RangeMap) {
        let tsid_set = RangeSet::from(base_pattern.clone());
        let mask = Weight::RangeMap(intern_rangemap(
            RangeMapWeight::from_uniform_tsid_set(0, max_llm_token, tsid_set, num_tsids),
        ));
        if let Some(start) = start {
            let elapsed = start.elapsed();
            if should_log_weight_expansion(elapsed) {
                crate::debug!(
                    5,
                    "create_tsid_set_mask: tsids={}, num_tsids={}, tokens={}, base_ranges={}, out_ranges={}, elapsed={:?}",
                    tsid_count,
                    num_tsids,
                    token_count,
                    base_ranges_len,
                    mask.ranges_len(),
                    elapsed,
                );
            }
        }
        return mask;
    }
    if token_count == 1 {
        let mask = Weight::from_rsb(base_pattern);
        if let Some(start) = start {
            let elapsed = start.elapsed();
            if should_log_weight_expansion(elapsed) {
                crate::debug!(
                    5,
                    "create_tsid_set_mask: tsids={}, num_tsids={}, tokens={}, base_ranges={}, out_ranges={}, elapsed={:?}",
                    tsid_count,
                    num_tsids,
                    token_count,
                    base_ranges_len,
                    mask.ranges_len(),
                    elapsed,
                );
            }
        }
        return mask;
    }

    // Fast path: base pattern covers the full tsid block, so the result is one contiguous range.
    if base_ranges_len == 1 {
        if let Some(r) = base_pattern.ranges().next() {
            if *r.start() == 0 && r.end().saturating_add(1) == num_tsids {
                let end = max_llm_token
                    .saturating_mul(num_tsids)
                    .saturating_add(num_tsids.saturating_sub(1));
                let mask: RangeSetBlaze<usize> = std::iter::once(0..=end).collect();
                let mask = Weight::from_rsb(mask);
                if let Some(start) = start {
                    let elapsed = start.elapsed();
                    if should_log_weight_expansion(elapsed) {
                        crate::debug!(
                            5,
                            "create_tsid_set_mask: tsids={}, num_tsids={}, tokens={}, base_ranges={}, out_ranges={}, elapsed={:?}",
                            tsid_count,
                            num_tsids,
                            token_count,
                            base_ranges_len,
                            mask.ranges_len(),
                            elapsed,
                        );
                    }
                }
                return mask;
            }
        }
    }
    
    // Tile the pattern across all LLM tokens
    // For each LLM token n, shift the base pattern by n * num_tsids
    let base_ranges: Vec<(usize, usize)> = base_pattern
        .ranges()
        .map(|r| (*r.start(), *r.end()))
        .collect();
    let mut ranges = Vec::with_capacity(base_ranges.len().saturating_mul(token_count));
    for (start, end) in base_ranges {
        for n in 0..token_count {
            let offset = n.saturating_mul(num_tsids);
            let shifted_start = start.saturating_add(offset);
            let shifted_end = end.saturating_add(offset);
            ranges.push(shifted_start..=shifted_end);
        }
    }

    let mask: RangeSetBlaze<usize> = ranges.into_iter().collect();
    let mask = Weight::from_rsb(mask);
    if let Some(start) = start {
        let elapsed = start.elapsed();
        if should_log_weight_expansion(elapsed) {
            crate::debug!(
                5,
                "create_tsid_set_mask: tsids={}, num_tsids={}, tokens={}, base_ranges={}, out_ranges={}, elapsed={:?}",
                tsid_count,
                num_tsids,
                token_count,
                base_ranges_len,
                mask.ranges_len(),
                elapsed,
            );
        }
    }
    mask
}

/// Create a combined tsid mask using WeightDimensions.
/// 
/// This version takes WeightDimensions to specify the N×M space.
pub fn create_tsid_set_mask_with_dims<I>(
    tsids: I,
    dims: crate::datastructures::WeightDimensions,
) -> Weight
where
    I: IntoIterator<Item = usize>,
{
    // max_llm_token is num_tokens - 1, or 0 if num_tokens is 0
    let max_llm_token = dims.num_tokens.saturating_sub(1);
    create_tsid_set_mask(tsids, dims.num_tsids, max_llm_token)
}

/// Collapse a weight from N×M-space back to N-space.
/// Given a weight in N×M-space (already restricted to a specific tsid via intersection),
/// convert positions back to LLM token IDs: position / num_tsids.
pub fn collapse_weight_rsb(rsb: &RangeSetBlaze<usize>, num_tsids: usize) -> RangeSetBlaze<usize> {
    if num_tsids == 0 || rsb.is_empty() {
        return rsb.clone();
    }
    // Process ranges in bulk rather than individual positions.
    // Each range [start, end] collapses to [start/M, end/M].
    // Collapsed ranges may overlap/merge, handled by collect.
    rsb.ranges()
        .map(|r| {
            let start_token = r.start() / num_tsids;
            let end_token = r.end() / num_tsids;
            start_token..=end_token
        })
        .collect()
}

/// Expands all weights in a DWA from N-space to NxM-space in-place.
pub fn expand_dwa_weights(dwa: &mut DWA, num_tsids: usize) {
    if num_tsids == 0 {
        return;
    }
    
    for state in &mut dwa.states.0 {
        // Expand final weight
        if let Some(ref fw) = state.final_weight {
            state.final_weight = Some(expand_weight(fw, num_tsids));
        }
        
        // Expand transition weights
        for weight in state.trans_weights.values_mut() {
            *weight = expand_weight(weight, num_tsids);
        }
    }
}



/// Find the maximum position set in any weight in the DWA.
fn find_max_weight_position(dwa: &DWA) -> usize {
    let mut max_pos = 0usize;
    
    for state in &dwa.states.0 {
        if let Some(ref fw) = state.final_weight {
            if !fw.is_all_fast() {
                if let Some(m) = fw.max_item() {
                    max_pos = max_pos.max(m);
                }
            }
        }
        
        for weight in state.trans_weights.values() {
            if !weight.is_all_fast() {
                if let Some(m) = weight.max_item() {
                    max_pos = max_pos.max(m);
                }
            }
        }
    }
    
    max_pos
}

/// Collapse a weight from N×M-space back to N-space.
/// For each original position n, the output has bit set if any position
/// in range [n*M, (n+1)*M) is set in the input weight.
pub fn collapse_weight(weight: &Weight, num_tsids: usize) -> Weight {
    if weight.is_empty() {
        return Weight::zeros();
    }
    if num_tsids == 0 {
        return weight.clone();
    }
    if weight.is_all_fast() {
        return Weight::all();
    }
    
    Weight::from_rsb(collapse_weight_rsb(&weight.to_rsb_allow_expansion(), num_tsids))
}

/// Create an initial weight for weight-heavy mode given an active tokenizer state ID.
/// This creates a weight where position indices are: tsid + n*M for n in 0..N
pub fn create_initial_weight_for_tsid(tsid: usize, num_tsids: usize, max_llm_token: usize) -> Weight {
    create_tsid_mask(tsid, num_tsids, max_llm_token)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_expand_weight() {
        // Weight [0, 1, 0, 1] meaning positions 1 and 3 are set
        let weight = Weight::from_iter([1usize, 3]);
        let num_tsids = 3;
        
        // After expansion with M=3:
        // Position 1 -> 3
        // Position 3 -> 9
        // But expand_weight expands ranges, so [1] becomes [3..=5] and [3] becomes [9..=11]
        let expanded = expand_weight(&weight, num_tsids);
        
        // Should have positions 3,4,5,9,10,11
        assert!(expanded.contains(3));
        assert!(expanded.contains(4));
        assert!(expanded.contains(5));
        assert!(expanded.contains(9));
        assert!(expanded.contains(10));
        assert!(expanded.contains(11));
        assert!(!expanded.contains(0));
        assert!(!expanded.contains(1));
        assert!(!expanded.contains(2));
        assert!(!expanded.contains(6));
        assert!(!expanded.contains(7));
        assert!(!expanded.contains(8));
    }
    
    #[test]
    fn test_create_tsid_mask() {
        let num_tsids = 3;
        let max_llm_token = 3; // 4 tokens: 0, 1, 2, 3
        
        // For tsid 1, should have positions: 1, 4, 7, 10
        let mask = create_tsid_mask(1, num_tsids, max_llm_token);
        
        assert!(mask.contains(1));
        assert!(mask.contains(4));
        assert!(mask.contains(7));
        assert!(mask.contains(10));
        assert!(!mask.contains(0));
        assert!(!mask.contains(2));
        assert!(!mask.contains(3));
        assert!(!mask.contains(5));
    }
    
    #[test]
    fn test_create_tsid_set_mask() {
        let num_tsids = 5;
        let max_llm_token = 3; // 4 tokens: 0, 1, 2, 3
        
        // Test with tsids {0, 2, 4}
        let mask = create_tsid_set_mask([0usize, 2, 4], num_tsids, max_llm_token);
        
        // Should have positions for each tsid across all tokens:
        // tsid 0: 0, 5, 10, 15
        // tsid 2: 2, 7, 12, 17
        // tsid 4: 4, 9, 14, 19
        assert!(mask.contains(0));
        assert!(mask.contains(5));
        assert!(mask.contains(10));
        assert!(mask.contains(15));
        assert!(mask.contains(2));
        assert!(mask.contains(7));
        assert!(mask.contains(12));
        assert!(mask.contains(17));
        assert!(mask.contains(4));
        assert!(mask.contains(9));
        assert!(mask.contains(14));
        assert!(mask.contains(19));
        
        // Should NOT have positions for tsids 1 or 3
        assert!(!mask.contains(1));
        assert!(!mask.contains(3));
        assert!(!mask.contains(6));
        assert!(!mask.contains(8));
        
        // Verify equivalence with calling create_tsid_mask individually
        let mask0 = create_tsid_mask(0, num_tsids, max_llm_token);
        let mask2 = create_tsid_mask(2, num_tsids, max_llm_token);
        let mask4 = create_tsid_mask(4, num_tsids, max_llm_token);
        let combined = &(&mask0 | &mask2) | &mask4;
        
        assert_eq!(mask.len(), combined.len());
        for pos in combined.to_rsb_allow_expansion().iter() {
            assert!(mask.contains(pos), "mask missing position {}", pos);
        }
    }
    
    #[test]
    fn test_collapse_weight() {
        let num_tsids = 3;
        // Weight with positions 4, 5 (which is token 1) and 10 (which is token 3)
        let weight = Weight::from_iter([4usize, 5, 10]);
        
        let collapsed = collapse_weight(&weight, num_tsids);
        
        // Should have tokens 1 and 3
        assert!(collapsed.contains(1));
        assert!(collapsed.contains(3));
        assert!(!collapsed.contains(0));
        assert!(!collapsed.contains(2));
    }
    
    #[test]
    fn test_expand_and_intersect_example() {
        // Example from the user request:
        // tsid=1, 3 tsids total, vocab size=4
        // Original weight [0,1,0,1] (positions 1 and 3)
        let original_weight = Weight::from_iter([1usize, 3]);
        let num_tsids = 3;
        let max_llm_token = 3;
        
        // After expansion: each position p -> range [p*M, p*M + M - 1]
        // Position 1 -> [3, 4, 5]
        // Position 3 -> [9, 10, 11]
        // So expanded = [3,4,5,9,10,11] which is [0,0,0,1,1,1,0,0,0,1,1,1]
        let expanded = expand_weight(&original_weight, num_tsids);
        
        // tsid mask for tsid=1: positions 1, 4, 7, 10 (every 1+M*i)
        // Which is [0,1,0,0,1,0,0,1,0,0,1,0]
        let tsid_mask = create_tsid_mask(1, num_tsids, max_llm_token);
        
        // Intersection: [0,0,0,1,1,1,0,0,0,1,1,1] & [0,1,0,0,1,0,0,1,0,0,1,0]
        // = [0,0,0,0,1,0,0,0,0,0,1,0]
        // Positions 4 and 10
        let final_weight = &expanded & &tsid_mask;
        
        assert!(final_weight.contains(4));
        assert!(final_weight.contains(10));
        assert!(!final_weight.contains(0));
        assert!(!final_weight.contains(1));
        assert!(!final_weight.contains(3));
        assert!(!final_weight.contains(5));
        assert!(!final_weight.contains(7));
        assert!(!final_weight.contains(9));
        assert!(!final_weight.contains(11));
        assert_eq!(final_weight.len(), 2);
    }
}
