//! Brute-force equivalence analysis for validation.
//!
//! This is the slowest but most correct implementation. For each token, it computes
//! the complete set of possible match sequences from each initial tokenizer state.
//!
//! A token's signature consists of:
//! - For each initial tokenizer state:
//!   - All possible sequences of (group_id, position) pairs that can match
//!   - The final tokenizer state after consuming the token
//!   - Whether the token ends in a "clean" state (can continue) vs dead end
//!
//! Two tokens are equivalent if they have identical signatures.

use crate::finite_automata::Regex;
use std::collections::{BTreeMap, BTreeSet};

/// Result of brute-force equivalence analysis
pub struct BruteForceEquivalenceResult {
    /// Equivalence classes: signature -> list of string indices
    pub mask_classes: BTreeMap<Vec<usize>, Vec<usize>>,
    /// Same as mask_classes (commit equivalence uses same logic here)
    pub commit_classes: BTreeMap<Vec<usize>, Vec<usize>>,
}

/// A single match event: (group_id, position within string)
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct MatchEvent {
    group_id: usize,
    position: usize,
}

/// Outcome of running a token from a specific initial state
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct TokenOutcome {
    /// Sequence of matches that occurred (in order)
    match_sequence: Vec<MatchEvent>,
    /// Final DFA state after consuming the token (None if dead end)
    final_state: Option<usize>,
    /// Signature of final state's possible future groups
    final_state_signature: u64,
}

/// Complete signature for a token across all initial states
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct BruteForceSignature {
    /// Map from initial_state_idx to outcome
    outcomes: BTreeMap<usize, TokenOutcome>,
}

/// Compute signature for final state based on possible future group IDs
fn compute_state_signature(regex: &Regex, state: usize) -> u64 {
    let mut h = 0xcbf29ce484222325u64; // FNV-1a init
    for &gid in &regex.dfa.states[state].possible_future_group_ids {
        h ^= gid as u64;
        h = h.wrapping_mul(0x100000001b3u64);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccdu64);
    h ^= h >> 33;
    h
}

/// Run a single token through the DFA from a given initial state.
/// Returns the outcome: match sequence, final state, and final state signature.
fn compute_token_outcome(
    regex: &Regex,
    token: &[u8],
    initial_state: usize,
) -> TokenOutcome {
    let mut current_state = initial_state;
    let mut match_sequence = Vec::new();
    
    for (pos, &byte) in token.iter().enumerate() {
        let state_data = &regex.dfa.states[current_state];
        
        // Try to transition
        if let Some(&next_state) = state_data.transitions.get(byte) {
            current_state = next_state;
            
            // Record any matches at this position
            let next_data = &regex.dfa.states[current_state];
            for gid in next_data.finalizers.iter_indices() {
                match_sequence.push(MatchEvent {
                    group_id: gid,
                    position: pos + 1, // Position after consuming byte
                });
            }
        } else {
            // Dead end - no valid transition
            return TokenOutcome {
                match_sequence,
                final_state: None,
                final_state_signature: 0,
            };
        }
    }
    
    // Successfully consumed entire token
    let final_sig = compute_state_signature(regex, current_state);
    TokenOutcome {
        match_sequence,
        final_state: Some(current_state),
        final_state_signature: final_sig,
    }
}

/// Compute the complete brute-force signature for a token
fn compute_bruteforce_signature(
    regex: &Regex,
    token: &[u8],
    initial_states: &[usize],
) -> BruteForceSignature {
    let mut outcomes = BTreeMap::new();
    
    for (state_idx, &init_state) in initial_states.iter().enumerate() {
        let outcome = compute_token_outcome(regex, token, init_state);
        outcomes.insert(state_idx, outcome);
    }
    
    BruteForceSignature { outcomes }
}

/// Main entry point: compute equivalence classes using brute force.
/// 
/// This is O(n * m * k) where:
/// - n = number of tokens
/// - m = max token length  
/// - k = number of initial states
///
/// Should only be used for small vocabularies (< 1000 tokens).
pub fn find_equivalence_classes_bruteforce(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> BruteForceEquivalenceResult {
    crate::debug!(3, "Brute-force equivalence analysis for {} strings, {} initial states",
                 strings.len(), initial_states.len());
    
    // Compute signature for each string
    let signatures: Vec<BruteForceSignature> = strings
        .iter()
        .map(|s| compute_bruteforce_signature(regex, s, initial_states))
        .collect();
    
    // Group by signature
    let mut groups: BTreeMap<BruteForceSignature, Vec<usize>> = BTreeMap::new();
    for (idx, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_default().push(idx);
    }
    
    // Convert to output format
    let mask_classes: BTreeMap<Vec<usize>, Vec<usize>> = groups
        .into_iter()
        .enumerate()
        .map(|(id, (_, indices))| (vec![id], indices))
        .collect();
    
    crate::debug!(3, "Brute-force equivalence: {} classes", mask_classes.len());
    
    BruteForceEquivalenceResult {
        mask_classes: mask_classes.clone(),
        commit_classes: mask_classes,
    }
}

/// Compare two equivalence results to check if they produce the same partition.
/// Returns Ok(()) if equivalent, Err with details if not.
pub fn compare_partitions(
    name1: &str,
    classes1: &BTreeMap<Vec<usize>, Vec<usize>>,
    name2: &str,
    classes2: &BTreeMap<Vec<usize>, Vec<usize>>,
    strings: &[Vec<u8>],
) -> Result<(), String> {
    // Build mapping from string index to class ID for both
    let mut string_to_class1: BTreeMap<usize, usize> = BTreeMap::new();
    for (class_id, (_, indices)) in classes1.iter().enumerate() {
        for &idx in indices {
            string_to_class1.insert(idx, class_id);
        }
    }
    
    let mut string_to_class2: BTreeMap<usize, usize> = BTreeMap::new();
    for (class_id, (_, indices)) in classes2.iter().enumerate() {
        for &idx in indices {
            string_to_class2.insert(idx, class_id);
        }
    }
    
    // Check that same strings are grouped together
    // For each pair of strings in the same class in result1, 
    // they should also be in the same class in result2
    for indices in classes1.values() {
        if indices.len() < 2 {
            continue;
        }
        let first = indices[0];
        let class2_for_first = string_to_class2.get(&first);
        
        for &idx in &indices[1..] {
            let class2_for_idx = string_to_class2.get(&idx);
            if class2_for_first != class2_for_idx {
                let s1 = String::from_utf8_lossy(&strings[first]);
                let s2 = String::from_utf8_lossy(&strings[idx]);
                return Err(format!(
                    "{} groups {:?} and {:?} together, but {} separates them",
                    name1, s1, s2, name2
                ));
            }
        }
    }
    
    // Check the reverse: strings in same class in result2 should be in same class in result1
    for indices in classes2.values() {
        if indices.len() < 2 {
            continue;
        }
        let first = indices[0];
        let class1_for_first = string_to_class1.get(&first);
        
        for &idx in &indices[1..] {
            let class1_for_idx = string_to_class1.get(&idx);
            if class1_for_first != class1_for_idx {
                let s1 = String::from_utf8_lossy(&strings[first]);
                let s2 = String::from_utf8_lossy(&strings[idx]);
                return Err(format!(
                    "{} groups {:?} and {:?} together, but {} separates them",
                    name2, s1, s2, name1
                ));
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_match_event_ordering() {
        let e1 = MatchEvent { group_id: 1, position: 0 };
        let e2 = MatchEvent { group_id: 1, position: 1 };
        let e3 = MatchEvent { group_id: 2, position: 0 };
        
        assert!(e1 < e2);
        assert!(e1 < e3);
    }
}
