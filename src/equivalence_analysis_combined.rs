//! Simple equivalence analysis for LLM token grouping - MASK ONLY.
//!
//! The key insight: For DWA weight equivalence, two tokens A and B are equivalent
//! if they appear in exactly the same weights. Weights correspond to (tokenizer_state,
//! grammar_terminal) pairs. So equivalence is determined purely by:
//!
//! - MASK EQUIVALENCE: Same set of (initial_state_idx, group_id) pairs can produce a match
//!   + Same future group signature for the final state(s)

use crate::finite_automata::Regex;
use hashbrown::HashMap;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

/// Result of equivalence analysis
pub struct CombinedEquivalenceResult {
    /// Equivalence classes for get_mask
    pub mask_classes: BTreeMap<Vec<usize>, Vec<usize>>,
    /// Equivalence classes for commit (placeholder - same as mask for now)
    pub commit_classes: BTreeMap<Vec<usize>, Vec<usize>>,
}

/// Signature for mask equivalence
#[derive(Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
struct MaskSignature {
    /// Set of (initial_state_idx, group_id) pairs where this token can trigger a match
    matches: Vec<(u16, u16)>,
    /// Signature(s) of possible final states (for determining future behavior)
    final_state_signatures: Vec<u64>,
}

/// Compute signature for a single string from all initial states.
/// Only tracks matches for group IDs in grammar_group_ids.
fn compute_string_signature(
    regex: &Regex,
    s: &[u8],
    initial_states: &[usize],
    state_signatures: &[u64],
    grammar_group_ids: &BTreeSet<usize>,
) -> MaskSignature {
    let mut matches: BTreeSet<(u16, u16)> = BTreeSet::new();
    let mut final_state_sigs: BTreeSet<u64> = BTreeSet::new();
    
    // For each initial state, run the tokenizer on the string
    for (state_idx, &init_state) in initial_states.iter().enumerate() {
        let state_idx = state_idx as u16;
        
        // Run the DFA on the string starting from this state
        let mut current_state = init_state;
        let mut valid = true;
        
        for &byte in s.iter() {
            let state_data = &regex.dfa.states[current_state];
            
            // Transition to next state
            if let Some(&next) = state_data.transitions.get(byte) {
                current_state = next;
                
                // Check for matches after consuming this byte
                // ONLY track grammar-relevant group IDs
                let next_state_data = &regex.dfa.states[current_state];
                for gid in next_state_data.finalizers.iter_indices() {
                    if grammar_group_ids.contains(&gid) {
                        matches.insert((state_idx, gid as u16));
                    }
                }
            } else {
                valid = false;
                break;
            }
        }
        
        // Record final state info if we consumed the entire string
        if valid {
            final_state_sigs.insert(state_signatures[current_state]);
        }
    }
    
    MaskSignature {
        matches: matches.into_iter().collect(),
        final_state_signatures: final_state_sigs.into_iter().collect(),
    }
}

/// Compute state signatures based on possible future group IDs
fn compute_state_signatures(regex: &Regex) -> Vec<u64> {
    regex.dfa.states.iter().map(|s| {
        let mut h = 0xcbf29ce484222325u64; // FNV-1a init
        for &gid in &s.possible_future_group_ids {
            h ^= gid as u64;
            h = h.wrapping_mul(0x100000001b3u64);
        }
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccdu64);
        h ^= h >> 33;
        h
    }).collect()
}

/// Main entry point: compute equivalence classes for a set of strings.
pub fn find_equivalence_classes_combined(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
    grammar_group_ids: &BTreeSet<usize>,
) -> CombinedEquivalenceResult {
    crate::debug!(3, "Simple equivalence analysis for {} strings, {} initial states, {} grammar groups", 
                 strings.len(), initial_states.len(), grammar_group_ids.len());
    
    let t0 = std::time::Instant::now();
    let state_signatures = compute_state_signatures(regex);
    crate::debug!(4, "Equiv: State signatures computed: {:?}", t0.elapsed());
    
    // Compute signatures for all strings in parallel
    let t1 = std::time::Instant::now();
    let signatures: Vec<MaskSignature> = strings
        .par_iter()
        .map(|s| compute_string_signature(regex, s, initial_states, &state_signatures, grammar_group_ids))
        .collect();
    crate::debug!(4, "Equiv: String signatures computed: {:?}", t1.elapsed());
    
    // Group by signature
    let t2 = std::time::Instant::now();
    let mut mask_groups: HashMap<MaskSignature, Vec<usize>> = HashMap::new();
    
    for (idx, sig) in signatures.into_iter().enumerate() {
        mask_groups.entry(sig).or_default().push(idx);
    }
    
    // Convert to output format
    let mask_classes: BTreeMap<Vec<usize>, Vec<usize>> = mask_groups
        .into_iter()
        .enumerate()
        .map(|(id, (_, indices))| (vec![id], indices))
        .collect();
    
    crate::debug!(4, "Equiv: Grouped into {} mask classes: {:?}", mask_classes.len(), t2.elapsed());
    crate::debug!(3, "Equivalence analysis complete: {} mask classes", mask_classes.len());
    
    // Return same classes for both mask and commit (commit can be refined later if needed)
    CombinedEquivalenceResult {
        mask_classes: mask_classes.clone(),
        commit_classes: mask_classes,
    }
}
