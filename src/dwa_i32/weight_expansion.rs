// src/precompute4/weighted_automata/weight_expansion.rs
//
// Converts a "symbol-heavy" DWA (where tokenizer state IDs are encoded as initial
// labeled transitions) into a "weight-heavy" DWA (where tsids are encoded in the
// weight space itself).
//
// Weight space expansion: N (LLM tokens) -> N×M (LLM tokens × tokenizer states)
// Layout: position = llm_token * M + tsid_offset
// By default, tsid_offset == tsid. However, callers may provide a permutation
// tsid_offset(tsid) to reduce RangeSet fragmentation (by making frequently-used
// tsid groups contiguous in the offset space).
//
// Note: A single LLM token active on ALL tsids uses just ONE range.

#![allow(dead_code)]

use range_set_blaze::RangeSetBlaze;
use super::common::{Label, Weight, weight_all};
use super::dwa::DWA;
use super::nwa::NWA;
use super::heavy_weight::WeightDimensions;

#[inline]
fn tsid_to_offset(tsid: usize, tsid_offset_map: Option<&[usize]>) -> usize {
    if let Some(map) = tsid_offset_map {
        debug_assert!(tsid < map.len());
        map[tsid]
    } else {
        tsid
    }
}

/// Expand a weight from N-space to N×M-space.
/// Each position p in the original weight becomes position p * num_tsids + 0..num_tsids-1.
pub fn expand_weight(weight: &Weight, num_tsids: usize) -> Weight {
    if weight.is_empty() {
        return Weight::zeros();
    }
    if weight.is_all_fast() {
        return weight_all();
    }
    
    let rsb = weight.to_rsb();
    Weight::from_rsb(expand_rsb(&rsb, num_tsids))
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
    rsb.ranges()
        .map(|r| {
            let start = r.start().saturating_mul(num_tsids);
            let end = r.end()
                .saturating_mul(num_tsids)
                .saturating_add(num_tsids.saturating_sub(1));
            start..=end
        })
        .collect()
}

/// Create a tsid mask for a specific tokenizer state ID.
/// The mask has positions: tsid + n*M for n in 0..N
/// This is equivalent to: all positions where position % M == tsid
pub fn create_tsid_mask(tsid: usize, num_tsids: usize, max_llm_token: usize) -> Weight {
    create_tsid_mask_with_offset_map(tsid, num_tsids, max_llm_token, None)
}

/// Create a tsid mask for a specific tokenizer state ID, using an optional tsid->offset map.
///
/// The mask has positions: off(tsid) + n*M for n in 0..N.
/// This is equivalent to: all positions where (position % M) == off(tsid).
pub fn create_tsid_mask_with_offset_map(
    tsid: usize,
    num_tsids: usize,
    max_llm_token: usize,
    tsid_offset_map: Option<&[usize]>,
) -> Weight {
    // For tsid off, we want positions: off, off+M, off+2M, ..., off+n*M where n <= max_llm_token
    let off = tsid_to_offset(tsid, tsid_offset_map);
    let mut mask = RangeSetBlaze::new();
    for n in 0..=max_llm_token {
        mask.insert(off + n * num_tsids);
    }
    Weight::from_rsb(mask)
}

/// Create a tsid mask as a RangeSetBlaze.
pub fn create_tsid_mask_rsb(tsid: usize, num_tsids: usize, max_llm_token: usize) -> RangeSetBlaze<usize> {
    create_tsid_mask_rsb_with_offset_map(tsid, num_tsids, max_llm_token, None)
}

/// Create a tsid mask as a RangeSetBlaze, using an optional tsid->offset map.
pub fn create_tsid_mask_rsb_with_offset_map(
    tsid: usize,
    num_tsids: usize,
    max_llm_token: usize,
    tsid_offset_map: Option<&[usize]>,
) -> RangeSetBlaze<usize> {
    let off = tsid_to_offset(tsid, tsid_offset_map);
    let mut mask = RangeSetBlaze::new();
    for n in 0..=max_llm_token {
        mask.insert(off + n * num_tsids);
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
    create_tsid_set_mask_with_offset_map(tsids, num_tsids, max_llm_token, None)
}

/// Create a combined tsid mask for a set of tokenizer state IDs, using an optional tsid->offset map.
pub fn create_tsid_set_mask_with_offset_map<I>(
    tsids: I,
    num_tsids: usize,
    max_llm_token: usize,
    tsid_offset_map: Option<&[usize]>,
) -> Weight
where
    I: IntoIterator<Item = usize>,
{
    // Build base pattern from all tsids (with offset mapping if provided)
    let mapped_tsids: Vec<usize> = tsids
        .into_iter()
        .map(|t| tsid_to_offset(t, tsid_offset_map))
        .collect();
    
    if mapped_tsids.is_empty() {
        return Weight::zeros();
    }
    
    // Use optimized tsid_columns construction with explicit dimensions
    // This is O(|tsids|) for BDD backends instead of O(|tsids| * |tokens|)
    Weight::tsid_columns_with_dims(mapped_tsids, num_tsids, max_llm_token + 1)
}

/// Collapse a weight from N×M-space back to N-space.
/// Given a weight in N×M-space (already restricted to a specific tsid via intersection),
/// convert positions back to LLM token IDs: position / num_tsids.
pub fn collapse_weight_rsb(rsb: &RangeSetBlaze<usize>, num_tsids: usize) -> RangeSetBlaze<usize> {
    if num_tsids == 0 || rsb.is_empty() {
        return rsb.clone();
    }
    let mut collapsed = RangeSetBlaze::new();
    for pos in rsb.iter() {
        collapsed.insert(pos / num_tsids);
    }
    collapsed
}

/// Convert a symbol-heavy DWA to a weight-heavy DWA.
/// 
/// In symbol-heavy mode:
/// - Start state has labeled transitions for each tokenizer state ID
/// - Labels at start state are tsid values (0, 1, 2, ..., M-1)
/// - Labels at internal states are parser_state_id values
/// - Weights encode just LLM tokens (N-space)
///
/// In weight-heavy mode:
/// - Start state has no tsid-labeled transitions (converted to epsilon)
/// - All weights are in N×M space (LLM tokens × tokenizer states)
/// - For a given tsid, only bits at positions {i*M + tsid : i ∈ 0..N} are relevant
///
/// Key insight: At the DWA start state, ALL labeled transitions are tsid transitions.
/// The `terminals_count` parameter is not used because tsid labels and parser_state_id
/// labels are distinguished by position (start vs internal), not by label value.
pub fn convert_symbol_heavy_to_weight_heavy(
    dwa: &DWA,
    num_tsids: usize,
    _terminals_count: usize,  // Not used, kept for API compatibility
) -> DWA {
    if num_tsids == 0 {
        return dwa.clone();
    }
    
    // Find the max LLM token in the existing weights to know our N
    let max_llm_token = find_max_weight_position(dwa);
    
    // Create NWA from DWA for manipulation
    let mut nwa = NWA::from_dwa(dwa);
    
    // Step 1: Expand all weights (transitions and finals) from N-space to N×M-space
    for state in &mut nwa.states.0 {
        if let Some(ref fw) = state.final_weight {
            state.final_weight = Some(expand_weight(fw, num_tsids));
        }
        for targets in state.transitions.values_mut() {
            for (_, weight) in targets {
                *weight = expand_weight(weight, num_tsids);
            }
        }
        for (_, weight) in &mut state.epsilons {
            *weight = expand_weight(weight, num_tsids);
        }
    }
    
    crate::debug!(3, "convert_symbol_heavy_to_weight_heavy: After expansion, NWA has {} states", nwa.states.0.len());
    
    // Step 2: At start state, ALL labeled transitions are tsid transitions.
    // Convert them to epsilon transitions with tsid-masked weights.
    let start_state = nwa.body.start_states[0];
    let start_transitions = std::mem::take(&mut nwa.states[start_state].transitions);
    
    crate::debug!(3, "convert: Start state {} had {} labeled transitions (all are tsid)", 
        start_state, start_transitions.len());
    
    for (label, targets) in start_transitions {
        // Label IS the tsid (0, 1, 2, ...)
        let tsid = label as usize;
        
        if tsid >= num_tsids {
            // This shouldn't happen - skip if tsid is out of range
            crate::debug!(2, "WARNING: tsid {} >= num_tsids {} at start state", tsid, num_tsids);
            continue;
        }
        
        let tsid_mask = create_tsid_mask(tsid, num_tsids, max_llm_token);
        
        for (target, weight) in targets {
            // Weight has already been expanded to N×M space.
            // Intersect with tsid mask to keep only bits for this specific tsid.
            let masked_weight = &weight & &tsid_mask;
            if !masked_weight.is_empty() {
                nwa.states[start_state].epsilons.push((target, masked_weight));
            }
        }
    }
    
    crate::debug!(3, "convert: After conversion, start state has {} epsilon transitions", 
        nwa.states[start_state].epsilons.len());
    
    // Step 3: Determinize and minimize
    let mut result = nwa.determinize();
    result.minimize();
    
    crate::debug!(3, "convert: After determinize, DWA has {} states", result.states.len());
    
    result
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
        return weight_all();
    }
    
    let rsb = weight.to_rsb();
    Weight::from_rsb(collapse_weight_rsb(&rsb, num_tsids))
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
        let combined_rsb = combined.to_rsb();
        for pos in combined_rsb.iter() {
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
    
    #[test]
    fn test_convert_symbol_heavy_to_weight_heavy() {
        use super::*;
        use crate::dwa_i32::dwa::{DWA, DWAState, DWAStates, DWABody};
        use crate::dwa_i32::test_weighted_automata::stochastic_equivalence_test;
        
        // Build the input DWA as specified:
        // DWA (start: 0)
        //   State 0:
        //     2 -> 1 (weight: [0..=1])   // label 2 = terminals_count + tsid 0
        //   State 1:
        //     0 -> 2 (weight: [0..=1])   // terminal A
        //     1 -> 3 (weight: [0..=1])   // terminal EOF
        //   State 2:
        //     final_weight: [1]
        //   State 3:
        //     final_weight: [0]
        
        let num_tsids = 3;
        let terminals_count = 2; // terminals A=0, EOF=1
        
        // Create input states
        let mut state0 = DWAState::default();
        state0.trans_weights.insert(2, Weight::from_rsb(RangeSetBlaze::from_iter([0, 1])));
        state0.transitions.insert(2, 1);
        
        let mut state1 = DWAState::default();
        state1.trans_weights.insert(0, Weight::from_rsb(RangeSetBlaze::from_iter([0, 1])));
        state1.trans_weights.insert(1, Weight::from_rsb(RangeSetBlaze::from_iter([0, 1])));
        state1.transitions.insert(0, 2);
        state1.transitions.insert(1, 3);
        
        let mut state2 = DWAState::default();
        state2.final_weight = Some(Weight::from_iter([1usize]));
        
        let mut state3 = DWAState::default();
        state3.final_weight = Some(Weight::from_iter([0usize]));
        
        let input_dwa = DWA {
            states: DWAStates(vec![state0, state1, state2, state3]),
            body: DWABody { start_state: 0 },
            dims: WeightDimensions::TEST,
        };
        
        println!("INPUT DWA:");
        println!("{}", input_dwa);
        
        // Build expected output DWA:
        // DWA (start: 0)
        //   State 0:
        //     0 -> 1 (weight: [2, 5])   // terminal A with tsid mask
        //     1 -> 2 (weight: [2, 5])   // terminal EOF with tsid mask
        //   State 1:
        //     final_weight: [5]            // from path: tsid0 + terminal EOF
        //   State 2:
        //     final_weight: [2]            // from path: tsid0 + terminal A
        
        let mut exp_state0 = DWAState::default();
        exp_state0.trans_weights.insert(0, Weight::from_iter([2usize, 5]));
        exp_state0.trans_weights.insert(1, Weight::from_iter([2usize, 5]));
        exp_state0.transitions.insert(0, 1);
        exp_state0.transitions.insert(1, 2);
        
        let mut exp_state1 = DWAState::default();
        exp_state1.final_weight = Some(Weight::from_iter([5usize]));
        
        let mut exp_state2 = DWAState::default();
        exp_state2.final_weight = Some(Weight::from_iter([2usize]));
        
        let expected_dwa = DWA {
            states: DWAStates(vec![exp_state0, exp_state1, exp_state2]),
            body: DWABody { start_state: 0 },
            dims: WeightDimensions::TEST,
        };
        
        println!("EXPECTED DWA:");
        println!("{}", expected_dwa);
        
        // Convert
        let output_dwa = convert_symbol_heavy_to_weight_heavy(&input_dwa, num_tsids, terminals_count);
        
        println!("OUTPUT DWA:");
        println!("{}", output_dwa);
        
        // Test for semantic equivalence
        stochastic_equivalence_test(output_dwa, expected_dwa);
    }


    #[test]
    fn test_convert_symbol_heavy_to_weight_heavy2() {
        use super::*;
        use crate::dwa_i32::dwa::{DWA, DWAState, DWAStates, DWABody};
        use crate::dwa_i32::test_weighted_automata::stochastic_equivalence_test;

        // Build the input DWA as specified:
        // DWA (start: 0)
        //   State 0:
        //     0 -> 1 (weight: [0..=1])
        //     1 -> 1 (weight: [0..=1])
        //   State 1:
        //     final_weight: [1]

        let num_tsids = 2;
        let terminals_count = 0; // No labeled parser-state transitions

        let mut state0 = DWAState::default();
        state0.trans_weights.insert(0, Weight::from_rsb(RangeSetBlaze::from_iter([0, 1])));
        state0.trans_weights.insert(1, Weight::from_rsb(RangeSetBlaze::from_iter([0, 1])));
        state0.transitions.insert(0, 1);
        state0.transitions.insert(1, 1);

        let mut state1 = DWAState::default();
        state1.final_weight = Some(Weight::from_iter([1usize]));

        let input_dwa = DWA {
            states: DWAStates(vec![state0, state1]),
            body: DWABody { start_state: 0 },
            dims: WeightDimensions::TEST,
        };

        println!("INPUT DWA:");
        println!("{}", input_dwa);

        // Build expected output DWA:
        // DWA (start: 0)
        //   State 0:
        //     final_weight: [2..=3]

        let mut exp_state0 = DWAState::default();
        exp_state0.final_weight = Some(Weight::from_rsb(RangeSetBlaze::from_iter([2, 3])));

        let expected_dwa = DWA {
            states: DWAStates(vec![exp_state0]),
            body: DWABody { start_state: 0 },
            dims: WeightDimensions::TEST,
        };

        println!("EXPECTED DWA:");
        println!("{}", expected_dwa);

        // Convert
        let output_dwa = convert_symbol_heavy_to_weight_heavy(&input_dwa, num_tsids, terminals_count);

        println!("OUTPUT DWA:");
        println!("{}", output_dwa);

        // Test for semantic equivalence
        stochastic_equivalence_test(output_dwa, expected_dwa);
    }
}
