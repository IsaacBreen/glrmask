//! Simple equivalence analysis based on terminal behavior - v3.
//!
//! Key insight: After a terminal match at position P, the tokenizer conceptually
//! "restarts" from the initial state to process the suffix from position P.
//!
//! So the equivalence signature needs to capture:
//! 1. For each initial state: what matches occur and at what positions
//! 2. For each match at position P: what ACCESSIBLE TERMINALS are available from
//!    the suffix's final state (when executed from the INITIAL state)
//! 3. The accessible terminals from the final tokenizer state after the entire token
//!
//! Two tokens are equivalent if all these behaviors are identical for ALL initial states.

use crate::finite_automata::Regex;
use hashbrown::HashMap;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Result of simple equivalence analysis
pub struct SimpleEquivalenceResult {
    /// Equivalence classes: signature -> list of string indices
    pub mask_classes: BTreeMap<Vec<usize>, Vec<usize>>,
    /// Same as mask_classes for commit equivalence
    pub commit_classes: BTreeMap<Vec<usize>, Vec<usize>>,
}

// 128-bit hash for better collision resistance
#[inline(always)]
fn mix_u128(mut x: u128) -> u128 {
    const C1: u128 = 0x9e3779b97f4a7c15_bf58476d1ce4e5b9;
    const C2: u128 = 0x94d049bb133111eb_ff51afd7ed558ccd;
    x ^= x >> 33;
    x = x.wrapping_mul(C1);
    x ^= x >> 33;
    x = x.wrapping_mul(C2);
    x ^= x >> 33;
    x
}

/// Hash for match with suffix
#[inline]
fn hash_match(group_id: usize, suffix_accessible: &[usize]) -> u128 {
    let mut h = mix_u128((group_id as u128) << 8 | 1);
    for &tid in suffix_accessible {
        h = h.wrapping_add(mix_u128(tid as u128 | 2));
    }
    h
}

/// Hash for state outcome
#[inline]
fn hash_outcome(matches_hash: u128, final_accessible: &[usize]) -> u128 {
    let mut h = matches_hash;
    for &tid in final_accessible {
        h = h.wrapping_add(mix_u128(tid as u128 | 4));
    }
    h
}

/// Execute suffix from a given state (typically the initial state)
fn execute_suffix_accessible(
    regex: &Regex,
    token: &[u8],
    from_position: usize,
    from_state: usize,
) -> Vec<usize> {
    let mut current_state = from_state;
    for &byte in &token[from_position..] {
        let sd = &regex.dfa.states[current_state];
        if let Some(&next_state) = sd.transitions.get(byte) {
            current_state = next_state;
        } else {
            return vec![]; // Dead end
        }
    }
    // Return accessible terminals (group IDs) from the final state
    regex.dfa.states[current_state].possible_future_group_ids.iter().copied().collect()
}

/// Compute hash-based signature for a single initial state
fn compute_outcome_hash(
    regex: &Regex,
    token: &[u8],
    initial_state: usize,
) -> u128 {
    if token.is_empty() {
        // Empty token: accessible terminals from the initial state
        let final_accessible: Vec<usize> = regex.dfa.states[initial_state]
            .possible_future_group_ids.iter().copied().collect();
        return hash_outcome(0, &final_accessible);
    }
    
    let mut current_state = initial_state;
    let mut matches_hash: u128 = 0;
    
    for (i, &byte) in token.iter().enumerate() {
        let sd = &regex.dfa.states[current_state];
        if let Some(&next_state) = sd.transitions.get(byte) {
            current_state = next_state;
            let next_data = &regex.dfa.states[current_state];
            for gid in next_data.finalizers.iter_indices() {
                // For each match, compute the suffix accessible terminals from INITIAL state
                let suffix_accessible = execute_suffix_accessible(
                    regex,
                    token,
                    i + 1, // Position after match
                    regex.dfa.start_state, // Execute suffix from INITIAL state
                );
                // Hash this match and accumulate
                matches_hash = matches_hash.wrapping_add(hash_match(gid, &suffix_accessible));
            }
        } else {
            // Dead end - return with current matches hash and empty final accessible
            return hash_outcome(matches_hash, &[]);
        }
    }
    
    // Accessible terminals from the final state
    let final_accessible: Vec<usize> = regex.dfa.states[current_state]
        .possible_future_group_ids.iter().copied().collect();
    
    hash_outcome(matches_hash, &final_accessible)
}

/// Compute combined hash signature for a token across all initial states
fn compute_hash_signature(
    regex: &Regex,
    token: &[u8],
    initial_states: &[usize],
) -> u128 {
    let mut combined_hash: u128 = 0;
    
    for (state_idx, &init_state) in initial_states.iter().enumerate() {
        let outcome_hash = compute_outcome_hash(regex, token, init_state);
        // Mix state index into the hash to ensure order matters
        combined_hash = combined_hash.wrapping_add(
            mix_u128(outcome_hash ^ (state_idx as u128) << 64)
        );
    }
    
    combined_hash
}

/// Main entry point
pub fn find_equivalence_classes_simple(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> SimpleEquivalenceResult {
    crate::debug!(3, "Simple equivalence v3 analysis for {} strings, {} initial states",
                 strings.len(), initial_states.len());
    
    let t0 = std::time::Instant::now();
    
    // Compute 128-bit hash signatures in parallel
    let signatures: Vec<u128> = strings
        .par_iter()
        .map(|s| compute_hash_signature(regex, s, initial_states))
        .collect();
    
    crate::debug!(4, "Simple equiv v3: signatures computed in {:?}", t0.elapsed());
    
    let t1 = std::time::Instant::now();
    
    // Group by hash signature (very fast now since hashes are just u128)
    let mut groups: HashMap<u128, Vec<usize>> = HashMap::new();
    for (idx, &sig) in signatures.iter().enumerate() {
        groups.entry(sig).or_default().push(idx);
    }
    
    crate::debug!(4, "Simple equiv v3: grouping took {:?}", t1.elapsed());
    
    // Convert to output format
    let mask_classes: BTreeMap<Vec<usize>, Vec<usize>> = groups
        .into_iter()
        .enumerate()
        .map(|(id, (_, indices))| (vec![id], indices))
        .collect();
    
    crate::debug!(3, "Simple equivalence v3: {} classes in {:?}", mask_classes.len(), t0.elapsed());
    
    SimpleEquivalenceResult {
        mask_classes: mask_classes.clone(),
        commit_classes: mask_classes,
    }
}
