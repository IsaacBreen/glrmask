//! Simple equivalence analysis based on terminal SEQUENCES.
//!
//! Two LLM tokens are equivalent iff for ALL tokenizer states, they can generate
//! the same set of terminal SEQUENCES. A sequence is a list of grammar terminals
//! that could be produced by consuming the token.
//!
//! Key insight: When a terminal completes mid-token, the tokenizer resets to
//! initial state for the remaining bytes. So terminal sequences can be multiple
//! terminals long.
//!
//! The algorithm:
//! 1. For each (token, tokenizer_state), compute all possible terminal sequences
//! 2. A sequence is built by: execute tokenizer, find matches (terminals that complete)
//! 3. For intermediate matches (before end), recurse from initial state
//! 4. For matches at end, that's a complete sequence entry  
//! 5. For end_state (partial match), accessible_terminals are possible final terminals

use crate::finite_automata::Regex;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Result of simple equivalence analysis
pub struct SimpleEquivalenceResult {
    /// Equivalence classes: signature -> list of string indices
    pub mask_classes: BTreeMap<Vec<usize>, Vec<usize>>,
    /// Same as mask_classes for commit equivalence
    pub commit_classes: BTreeMap<Vec<usize>, Vec<usize>>,
}

/// A terminal sequence - list of group IDs that could be produced
type TerminalSequence = Vec<usize>;

/// Outcome for one initial state: set of possible terminal sequences + final accessible terminals
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct StateOutcome {
    /// All possible terminal sequences from this state
    /// Sorted for consistent hashing
    sequences: Vec<TerminalSequence>,
    /// Accessible terminals from final state (if any partial match)
    /// This captures what terminals could combine with future tokens
    final_accessible: Vec<usize>,
}

/// Compute all possible terminal sequences for a token from a given initial state.
/// 
/// Algorithm uses DFS with memoization to explore all match paths:
/// - When a terminal matches before end of token, continue from INITIAL state (reset)
/// - When a terminal matches at end of token, that's a complete sequence
/// - When there's a partial match (end_state), record accessible terminals
///
/// Uses hashing to avoid storing full sequences (which can be exponential).
fn compute_sequences_for_state(
    regex: &Regex,
    token: &[u8],
    initial_state: usize,
) -> StateOutcome {
    if token.is_empty() {
        // Empty token - check what's accessible from initial state
        let state_data = &regex.dfa.states[initial_state];
        let mut accessible: Vec<usize> = state_data.possible_future_group_ids.iter().cloned().collect();
        accessible.sort();
        return StateOutcome {
            sequences: vec![],
            final_accessible: accessible,
        };
    }
    
    // Memoization: position -> (sequence_hashes, final_accessible) from that position
    // Since we always reset to state 0 after intermediate matches, we only need position as key
    let mut memo: HashMap<usize, (HashSet<u64>, BTreeSet<usize>)> = HashMap::new();
    
    fn dfs(
        regex: &Regex,
        token: &[u8],
        pos: usize,
        memo: &mut HashMap<usize, (HashSet<u64>, BTreeSet<usize>)>,
    ) -> (HashSet<u64>, BTreeSet<usize>) {
        // Check memo
        if let Some(cached) = memo.get(&pos) {
            return cached.clone();
        }
        
        let mut result_hashes: HashSet<u64> = HashSet::new();
        let mut final_accessible: BTreeSet<usize> = BTreeSet::new();
        
        if pos >= token.len() {
            // End of token - no more sequences, but record accessible from initial state
            let state_data = &regex.dfa.states[0];  // State 0 after reset
            for &gid in &state_data.possible_future_group_ids {
                final_accessible.insert(gid);
            }
            memo.insert(pos, (result_hashes.clone(), final_accessible.clone()));
            return (result_hashes, final_accessible);
        }
        
        // Execute tokenizer from current position, starting from state 0
        // (Except for initial call which may use a different state - handled by caller)
        let remaining = &token[pos..];
        let mut current_state = 0usize;  // Always start from initial state for continuation
        let mut matches: Vec<(usize, usize)> = vec![]; // (group_id, end_position)
        
        for (i, &byte) in remaining.iter().enumerate() {
            let state_data = &regex.dfa.states[current_state];
            
            if let Some(&next_state) = state_data.transitions.get(byte) {
                current_state = next_state;
                
                // Check for finalizers (completed terminals)
                let next_data = &regex.dfa.states[current_state];
                for gid in next_data.finalizers.iter_indices() {
                    matches.push((gid, pos + i + 1)); // absolute position
                }
            } else {
                break;
            }
        }
        
        // Process matches
        for (gid, end_pos) in matches {
            let hash = gid as u64;
            
            if end_pos < token.len() {
                // Intermediate match - recurse to get continuations
                let (cont_hashes, cont_accessible) = dfs(regex, token, end_pos, memo);
                
                // Combine this terminal with all continuations
                for &cont_hash in &cont_hashes {
                    let combined = hash.wrapping_mul(0x100000001b3).wrapping_add(cont_hash);
                    result_hashes.insert(combined);
                }
                
                // Also record this terminal as a complete sequence if there's a partial at end
                if cont_accessible.len() > 0 {
                    result_hashes.insert(hash);
                }
                
                final_accessible.extend(cont_accessible);
            } else {
                // Match at end of token - this is a complete sequence
                result_hashes.insert(hash);
            }
        }
        
        // Handle partial match at end
        if pos + remaining.len() == token.len() {
            let state_data = &regex.dfa.states[current_state];
            for &gid in &state_data.possible_future_group_ids {
                final_accessible.insert(gid);
            }
        }
        
        memo.insert(pos, (result_hashes.clone(), final_accessible.clone()));
        (result_hashes, final_accessible)
    }
    
    // For the initial call, we need to handle the specific initial_state
    // Execute from the given initial state first
    let remaining = &token[..];
    let mut current_state = initial_state;
    let mut matches: Vec<(usize, usize)> = vec![]; // (group_id, end_position)
    
    for (i, &byte) in remaining.iter().enumerate() {
        let state_data = &regex.dfa.states[current_state];
        
        if let Some(&next_state) = state_data.transitions.get(byte) {
            current_state = next_state;
            
            // Check for finalizers (completed terminals)
            let next_data = &regex.dfa.states[current_state];
            for gid in next_data.finalizers.iter_indices() {
                matches.push((gid, i + 1)); // position is 1-indexed
            }
        } else {
            break;
        }
    }
    
    let mut result_hashes: HashSet<u64> = HashSet::new();
    let mut final_accessible_set: BTreeSet<usize> = BTreeSet::new();
    
    // Process matches
    for (gid, end_pos) in matches {
        let hash = gid as u64;
        
        if end_pos < token.len() {
            // Intermediate match - recurse to get continuations (from state 0)
            let (cont_hashes, cont_accessible) = dfs(regex, token, end_pos, &mut memo);
            
            // Combine this terminal with all continuations
            for &cont_hash in &cont_hashes {
                let combined = hash.wrapping_mul(0x100000001b3).wrapping_add(cont_hash);
                result_hashes.insert(combined);
            }
            
            // Also record this terminal as a complete sequence if there's partial continuation
            if cont_accessible.len() > 0 {
                result_hashes.insert(hash);
            }
            
            final_accessible_set.extend(cont_accessible);
        } else {
            // Match at end of token - this is a complete sequence
            result_hashes.insert(hash);
        }
    }
    
    // Handle partial match at end for initial traversal
    let state_data = &regex.dfa.states[current_state];
    for &gid in &state_data.possible_future_group_ids {
        final_accessible_set.insert(gid);
    }
    
    // Convert to sorted vectors for consistent representation
    let mut sequences: Vec<TerminalSequence> = result_hashes.iter().map(|&h| vec![h as usize]).collect();
    sequences.sort();
    
    let final_accessible: Vec<usize> = final_accessible_set.into_iter().collect();
    
    StateOutcome {
        sequences,
        final_accessible,
    }
}

/// Signature for a single token across all initial states
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct SimpleSignature {
    /// For each initial state: the outcome (sequences + final accessible)
    /// Keyed by state index for consistent ordering
    outcomes: Vec<(usize, StateOutcome)>,
}

/// Compute signature for a single token across all initial states
fn compute_simple_signature(
    regex: &Regex,
    token: &[u8],
    initial_states: &[usize],
) -> SimpleSignature {
    let mut outcomes = Vec::with_capacity(initial_states.len());
    
    for (state_idx, &init_state) in initial_states.iter().enumerate() {
        let outcome = compute_sequences_for_state(regex, token, init_state);
        // Only include non-trivial outcomes
        if !outcome.sequences.is_empty() || !outcome.final_accessible.is_empty() {
            outcomes.push((state_idx, outcome));
        }
    }
    
    SimpleSignature { outcomes }
}

/// Main entry point: compute equivalence classes using simple method.
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
    fn test_empty_token() {
        // Just a basic sanity check
        let outcome = StateOutcome {
            sequences: vec![],
            final_accessible: vec![1, 2],
        };
        assert_eq!(outcome.final_accessible.len(), 2);
    }
}
