//! Simple equivalence analysis without trie optimization.
//!
//! This version computes signatures independently for each token without
//! using a trie to share prefix computation. It's slower than the trie version
//! but simpler and easier to verify.
//!
//! The signature computation uses hashing similar to the fast version but
//! processes each token independently.

use crate::finite_automata::Regex;
use hashbrown::HashMap;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

/// Result of simple equivalence analysis
pub struct SimpleEquivalenceResult {
    /// Equivalence classes: signature -> list of string indices
    pub mask_classes: BTreeMap<Vec<usize>, Vec<usize>>,
    /// Same as mask_classes for commit equivalence
    pub commit_classes: BTreeMap<Vec<usize>, Vec<usize>>,
}

/// A match that occurred during token processing
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct Match {
    group_id: usize,
    position: usize,
}

/// Signature for a single token from all initial states
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct SimpleSignature {
    /// For each initial state: (matches_that_occurred, final_state_signature)
    /// Sorted by initial state index for consistency
    state_outcomes: Vec<(usize, Vec<Match>, u64)>,
}

/// Compute state signature based on possible future group IDs
fn compute_state_signature(regex: &Regex, state: usize) -> u64 {
    let state_data = &regex.dfa.states[state];
    let mut h = 0xcbf29ce484222325u64; // FNV-1a init
    for &gid in &state_data.possible_future_group_ids {
        h ^= gid as u64;
        h = h.wrapping_mul(0x100000001b3u64);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccdu64);
    h ^= h >> 33;
    h
}

/// Process a single token from a single initial state.
/// Returns (matches, final_state_signature) or None if dead end.
fn process_token_from_state(
    regex: &Regex,
    token: &[u8],
    initial_state: usize,
) -> Option<(Vec<Match>, u64)> {
    let mut current_state = initial_state;
    let mut matches = Vec::new();
    
    for (pos, &byte) in token.iter().enumerate() {
        let state_data = &regex.dfa.states[current_state];
        
        if let Some(&next_state) = state_data.transitions.get(byte) {
            current_state = next_state;
            
            // Record matches
            let next_data = &regex.dfa.states[current_state];
            for gid in next_data.finalizers.iter_indices() {
                matches.push(Match {
                    group_id: gid,
                    position: pos + 1,
                });
            }
        } else {
            // Dead end
            return None;
        }
    }
    
    let final_sig = compute_state_signature(regex, current_state);
    Some((matches, final_sig))
}

/// Compute signature for a single token across all initial states
fn compute_simple_signature(
    regex: &Regex,
    token: &[u8],
    initial_states: &[usize],
) -> SimpleSignature {
    let mut state_outcomes = Vec::with_capacity(initial_states.len());
    
    for (state_idx, &init_state) in initial_states.iter().enumerate() {
        if let Some((matches, final_sig)) = process_token_from_state(regex, token, init_state) {
            state_outcomes.push((state_idx, matches, final_sig));
        }
        // Dead ends are not recorded - they're implicitly "no outcome for this state"
    }
    
    // Sort by state index for consistent ordering
    state_outcomes.sort_by_key(|(idx, _, _)| *idx);
    
    SimpleSignature { state_outcomes }
}

/// Main entry point: compute equivalence classes using simple (no trie) method.
pub fn find_equivalence_classes_simple(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> SimpleEquivalenceResult {
    crate::debug!(3, "Simple equivalence analysis for {} strings, {} initial states",
                 strings.len(), initial_states.len());
    
    let t0 = std::time::Instant::now();
    
    // Compute signatures in parallel
    let signatures: Vec<SimpleSignature> = strings
        .par_iter()
        .map(|s| compute_simple_signature(regex, s, initial_states))
        .collect();
    
    crate::debug!(4, "Simple equiv: signatures computed in {:?}", t0.elapsed());
    
    // Group by signature
    let mut groups: HashMap<SimpleSignature, Vec<usize>> = HashMap::new();
    for (idx, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_default().push(idx);
    }
    
    // Convert to output format
    let mask_classes: BTreeMap<Vec<usize>, Vec<usize>> = groups
        .into_iter()
        .enumerate()
        .map(|(id, (_, indices))| (vec![id], indices))
        .collect();
    
    crate::debug!(3, "Simple equivalence: {} classes in {:?}", mask_classes.len(), t0.elapsed());
    
    SimpleEquivalenceResult {
        mask_classes: mask_classes.clone(),
        commit_classes: mask_classes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_match_ordering() {
        let m1 = Match { group_id: 1, position: 0 };
        let m2 = Match { group_id: 1, position: 1 };
        assert!(m1 < m2);
    }
}
