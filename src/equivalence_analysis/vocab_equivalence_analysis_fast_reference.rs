//! Reference Implementation of Vocab Equivalence Analysis
//!
//! This is a simple, correct implementation for testing and validation.
//! It properly computes token equivalence including recursive suffix behavior,
//! matching the semantics of the optimized implementation.
//!
//! This implementation is SLOWER than the optimized version (vocab_equivalence_analysis_fast.rs)
//! but serves as a reference for correctness testing. Use the environment variable
//! `USE_FAST_REFERENCE_VOCAB=1` to enable this implementation instead of the optimized one.
//!
//! Complexity: O(tokens × states × avg_token_length²) with parallelism
//! The squared factor comes from computing suffix hashes from each finalization point.

use crate::finite_automata::Regex;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;

const NONE_STATE: u32 = u32::MAX;

/// Compute a recursive suffix hash from a given position in the token.
/// This captures what happens if the tokenizer resumes parsing from this position.
///
/// Returns (end_state_hash, edges_with_suffix_hashes)
fn compute_suffix_hash(
    dfa_transitions: &[[u32; 256]],
    dfa_finalizers: &[Vec<usize>],
    possible_futures: &[Vec<usize>],
    token: &[u8],
    start_pos: usize,
    start_state: usize,
    dfa_start_state: usize,  // The DFA's initial state for suffix computations
    memo: &mut HashMap<usize, u64>,  // Keyed only by position for suffix hashes
) -> u64 {
    // For suffix hashes (pos > 0), we always start from dfa_start_state
    // So the memo key only needs the position
    if start_pos > 0 {
        if let Some(&cached) = memo.get(&start_pos) {
            return cached;
        }
    }
    
    let mut hasher = DefaultHasher::new();
    
    // Run DFA from start_state on token[start_pos..]
    let mut current = start_state as u32;
    let mut edges: Vec<(usize, usize)> = Vec::new(); // (group_id, position)
    
    for (rel_pos, &byte) in token[start_pos..].iter().enumerate() {
        if current == NONE_STATE {
            break;
        }
        let next = dfa_transitions[current as usize][byte as usize];
        if next == NONE_STATE {
            current = NONE_STATE;
            break;
        }
        current = next;
        let abs_pos = start_pos + rel_pos + 1;
        
        // Track finalizers at this position
        let finalizers = &dfa_finalizers[current as usize];
        for &gid in finalizers {
            edges.push((gid, abs_pos));
        }
    }
    
    // Hash the end state
    let end_hash = if current == NONE_STATE {
        mix64(0xDEADBEEF_u64)
    } else {
        // Hash possible futures of end state
        let futures = &possible_futures[current as usize];
        let mut h: u64 = 0;
        for &gid in futures {
            h = h.wrapping_add(mix64(gid as u64));
        }
        h | (1 << 63)
    };
    end_hash.hash(&mut hasher);
    
    // For each edge, compute suffix hash recursively and include it
    // Sort edges for determinism
    edges.sort_unstable();
    
    for &(gid, pos) in &edges {
        if pos <= token.len() {
            // The suffix hash is computed from the DFA's start state
            // because when the tokenizer resumes, it restarts from the initial state
            let suffix_hash = compute_suffix_hash(
                dfa_transitions,
                dfa_finalizers,
                possible_futures,
                token,
                pos,
                dfa_start_state,
                dfa_start_state,
                memo,
            );
            (gid as u64).hash(&mut hasher);
            suffix_hash.hash(&mut hasher);
        }
    }
    
    let result = hasher.finish();
    if start_pos > 0 {
        memo.insert(start_pos, result);
    }
    result
}

/// Compute a signature for a token based on its behavior from all initial states.
///
/// The signature captures:
/// - For each initial state, where does the token end up?
/// - What finalizers are encountered along the way (with positions)?
/// - Recursive suffix behavior from each finalization point
fn compute_token_signature(
    dfa_transitions: &[[u32; 256]],
    dfa_finalizers: &[Vec<usize>],
    possible_futures: &[Vec<usize>],
    token: &[u8],
    initial_states: &[usize],
    dfa_start_state: usize,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    let mut memo: HashMap<usize, u64> = HashMap::new();
    
    for &start_state in initial_states {
        // Compute suffix hash starting from position 0 with this initial state
        let state_sig = compute_suffix_hash(
            dfa_transitions,
            dfa_finalizers,
            possible_futures,
            token,
            0,
            start_state,
            dfa_start_state,
            &mut memo,
        );
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
    let dfa_start_state = dfa.start_state;
    
    // Precompute packed transition tables for cache efficiency
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
                dfa_start_state,
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
        "Reference vocab equiv: {} tokens -> {} classes in {:?}",
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
