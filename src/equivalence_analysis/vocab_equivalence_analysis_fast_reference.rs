//! Fast Reference Implementation of Vocab Equivalence Analysis
//!
//! This is a simple, correct implementation that is reasonably fast (< 1 second
//! for most cases). It uses basic parallelism and straightforward hashing without
//! the advanced optimizations of the main fast implementation.
//!
//! Use this as a reference for testing and validation against the optimized version.
//!
//! Complexity: O(tokens × states × avg_token_length) with parallelism

use crate::finite_automata::Regex;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;

/// Compute a signature for a token based on its behavior from all initial states.
///
/// The signature captures:
/// - For each initial state, where does the token end up?
/// - What finalizers are encountered along the way (with positions)?
fn compute_token_signature(
    dfa_transitions: &[[u32; 256]],
    dfa_finalizers: &[Vec<usize>],
    possible_futures: &[Vec<usize>],
    token: &[u8],
    initial_states: &[usize],
) -> u64 {
    const NONE_STATE: u32 = u32::MAX;
    
    let mut hasher = DefaultHasher::new();
    
    for &start_state in initial_states {
        let mut current = start_state as u32;
        let mut path_hash: u64 = 0;
        
        for (depth, &byte) in token.iter().enumerate() {
            if current == NONE_STATE {
                break;
            }
            let next = dfa_transitions[current as usize][byte as usize];
            if next == NONE_STATE {
                current = NONE_STATE;
                break;
            }
            current = next;
            
            // Hash finalizers at this position
            let finalizers = &dfa_finalizers[current as usize];
            if !finalizers.is_empty() {
                let depth_u64 = (depth + 1) as u64;
                for &gid in finalizers {
                    // Position-sensitive hash of finalizer
                    path_hash = path_hash.wrapping_add(
                        mix64(depth_u64 ^ ((gid as u64) << 32))
                    );
                }
            }
        }
        
        // Hash the end state's possible future groups
        let end_hash = if current == NONE_STATE {
            mix64(0xDEADBEEF_u64)
        } else {
            let futures = &possible_futures[current as usize];
            let mut h: u64 = 0;
            for &gid in futures {
                h = h.wrapping_add(mix64(gid as u64));
            }
            h | (1 << 63)
        };
        
        // Combine end state and path info for this initial state
        let state_sig = end_hash.wrapping_add(path_hash);
        state_sig.hash(&mut hasher);
    }
    
    hasher.finish()
}

/// Fast 64-bit mixing function
#[inline(always)]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

/// Find vocab equivalence classes using a simple but fast algorithm.
///
/// # Arguments
/// * `regex` - The tokenizer DFA
/// * `tokens` - Vocabulary tokens to analyze
/// * `initial_states` - Tokenizer states to consider for equivalence
///
/// # Returns
/// Sets of token indices that are equivalent (produce identical parsing behavior).
pub fn find_vocab_equivalence_classes(
    regex: &Regex,
    tokens: &[Vec<u8>],
    initial_states: &[usize],
) -> VocabEquivalenceResult {
    let start = std::time::Instant::now();
    let dfa = &regex.dfa;
    
    // Precompute packed transition tables for cache efficiency
    const NONE_STATE: u32 = u32::MAX;
    let dfa_transitions: Vec<[u32; 256]> = dfa.states
        .iter()
        .map(|state| {
            let mut table = [NONE_STATE; 256];
            for (byte, &target) in state.transitions.iter() {
                table[byte as usize] = target as u32;
            }
            table
        })
        .collect();
    
    let dfa_finalizers: Vec<Vec<usize>> = dfa.states
        .iter()
        .map(|state| state.finalizers.iter().collect())
        .collect();
    
    let possible_futures: Vec<Vec<usize>> = dfa.states
        .iter()
        .map(|state| state.possible_future_group_ids.iter().copied().collect())
        .collect();
    
    // Compute signatures for all tokens in parallel
    let signatures: Vec<u64> = tokens
        .par_iter()
        .map(|token| {
            compute_token_signature(
                &dfa_transitions,
                &dfa_finalizers,
                &possible_futures,
                token,
                initial_states,
            )
        })
        .collect();
    
    // Group tokens by signature
    let mut signature_groups: HashMap<u64, Vec<usize>> = HashMap::new();
    for (token_idx, &sig) in signatures.iter().enumerate() {
        signature_groups.entry(sig).or_default().push(token_idx);
    }
    
    // Convert to result format
    let result: VocabEquivalenceResult = signature_groups
        .into_values()
        .collect();
    
    crate::debug!(
        3,
        "Fast reference vocab equiv: {} tokens -> {} classes in {:?}",
        tokens.len(),
        result.len(),
        start.elapsed(),
    );
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Basic smoke test - actual testing requires integration tests with real DFAs
    #[test]
    fn test_mix64() {
        // Just ensure mix64 is deterministic
        assert_eq!(mix64(42), mix64(42));
        assert_ne!(mix64(42), mix64(43));
    }
}
