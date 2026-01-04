// src/precompute4/weighted_automata/weight_expansion.rs
//
// Converts a "symbol-heavy" DWA (where tokenizer state IDs are encoded as initial
// labeled transitions) into a "weight-heavy" DWA (where tsids are encoded in the
// weight space itself).
//
// Weight space expansion: N (LLM tokens) -> N×M (LLM tokens × tokenizer states)
// Layout: position = llm_token * M + tsid
// This means iterating through weight indices iterates tsids first, so a single
// LLM token active on ALL tsids uses just ONE range.

#![allow(dead_code)]

use range_set_blaze::RangeSetBlaze;
use super::common::{Label, Weight};
use super::dwa::DWA;
use super::nwa::NWA;

/// Expand a weight from N-space to N×M-space.
/// Each position p in the original weight becomes position p * num_tsids + 0..num_tsids-1.
pub fn expand_weight(weight: &Weight, num_tsids: usize) -> Weight {
    if weight.is_empty() {
        return Weight::zeros();
    }
    if weight.is_all_fast() {
        return Weight::all();
    }
    
    Weight::from_rsb(expand_rsb(&weight.rsb, num_tsids))
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
fn expand_rsb(rsb: &RangeSetBlaze<usize>, num_tsids: usize) -> RangeSetBlaze<usize> {
    let mut expanded = RangeSetBlaze::new();
    for range in rsb.ranges() {
        let start = range.start().saturating_mul(num_tsids);
        let end_base = range.end().saturating_mul(num_tsids);
        let end = end_base.saturating_add(num_tsids.saturating_sub(1));
        expanded.extend([start..=end]);
    }
    expanded
}

/// Create a tsid mask for a specific tokenizer state ID.
/// The mask has positions: tsid + n*M for n in 0..N
/// This is equivalent to: all positions where position % M == tsid
fn create_tsid_mask(tsid: usize, num_tsids: usize, max_llm_token: usize) -> Weight {
    // For tsid m, we want positions: m, m+M, m+2M, ..., m+n*M where n*M + m <= max_llm_token * M
    let mut mask = RangeSetBlaze::new();
    for n in 0..=max_llm_token {
        mask.insert(tsid + n * num_tsids);
    }
    Weight::from_rsb(mask)
}

/// Convert a symbol-heavy DWA to a weight-heavy DWA.
/// 
/// In symbol-heavy mode:
/// - Start state has labeled transitions for each tokenizer state ID
/// - Labels >= terminals_count represent tsids (label - terminals_count = tsid)
/// - Weights encode just LLM tokens (N-space)
///
/// In weight-heavy mode:
/// - Start state has no special tsid transitions
/// - All weights are in N×M space (LLM tokens × tokenizer states)
/// - Initial transitions become epsilon transitions with tsid-masked weights
pub fn convert_symbol_heavy_to_weight_heavy(
    dwa: &DWA,
    num_tsids: usize,
    terminals_count: usize,
) -> DWA {
    if num_tsids == 0 {
        return dwa.clone();
    }
    
    // Find the max LLM token in the existing weights to know our N
    let max_llm_token = find_max_weight_position(dwa);
    
    // Create NWA from DWA for manipulation
    let mut nwa = NWA::from_dwa(dwa);
    
    // Step 1: Expand all weights (transitions and finals) by multiplying positions by M
    for state in &mut nwa.states.0 {
        // Expand final weight
        if let Some(ref fw) = state.final_weight {
            state.final_weight = Some(expand_weight(fw, num_tsids));
        }
        
        // Expand transition weights
        for targets in state.transitions.values_mut() {
            for (_, weight) in targets {
                *weight = expand_weight(weight, num_tsids);
            }
        }
        
        // Expand epsilon weights
        for (_, weight) in &mut state.epsilons {
            *weight = expand_weight(weight, num_tsids);
        }
    }
    
    // Step 2: Convert tsid labeled transitions from start state to epsilon transitions
    let start_state = nwa.body.start_states[0];
    let start_transitions = std::mem::take(&mut nwa.states[start_state].transitions);
    
    for (label, targets) in start_transitions {
        let is_tsid_label = label >= terminals_count as Label;
        
        if is_tsid_label {
            // This is a tsid transition - convert to epsilon with masked weight
            let tsid = (label - terminals_count as Label) as usize;
            let tsid_mask = create_tsid_mask(tsid, num_tsids, max_llm_token);
            
            for (target, weight) in targets {
                // Intersect the expanded weight with the tsid mask
                let masked_weight = &weight & &tsid_mask;
                if !masked_weight.is_empty() {
                    nwa.states[start_state].epsilons.push((target, masked_weight));
                }
            }
        } else {
            // Regular terminal transition - keep as is (already expanded)
            nwa.states[start_state].transitions.insert(label, targets);
        }
    }
    
    // Step 3: Determinize and simplify
    let mut result = nwa.determinize();
    result.simplify();
    
    result
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
    
    // For each position in the expanded weight, divide by M to get the original position
    let mut collapsed = RangeSetBlaze::new();
    for range in weight.rsb.ranges() {
        let start = *range.start() / num_tsids;
        let end = *range.end() / num_tsids;
        collapsed.extend([start..=end]);
    }
    Weight::from_rsb(collapsed)
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
