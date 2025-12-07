//! State Equivalence Analysis - Reference Implementation
//!
//! A simple, correct implementation for testing and validation.
//! States are equivalent if they have identical behavior on ALL tokens.
//!
//! Complexity: O(states × tokens × avg_token_length) with parallelism

use std::collections::{BTreeSet, HashMap};
use rayon::prelude::*;
use crate::finite_automata::Regex;

/// The result of state equivalence analysis: sets of state IDs that behave identically.
pub type StateEquivalenceResult = BTreeSet<BTreeSet<usize>>;

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

/// Compute signature for a state based on its behavior on all tokens.
fn compute_state_signature(
    dfa_transitions: &[[u32; 256]],
    dfa_finalizers: &[Vec<usize>],
    possible_futures: &[Vec<usize>],
    tokens: &[Vec<u8>],
    state: usize,
) -> u64 {
    const NONE_STATE: u32 = u32::MAX;
    let mut hash: u64 = 0;
    
    for (token_idx, token) in tokens.iter().enumerate() {
        // Run token through DFA from this state
        let mut current = state as u32;
        let mut finalizers_hash: u64 = 0;
        
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
                for &gid in finalizers {
                    finalizers_hash = finalizers_hash.wrapping_add(
                        mix64(((depth + 1) as u64) ^ ((gid as u64) << 32))
                    );
                }
            }
        }
        
        // Hash end state's possible futures
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
        
        // Combine into token result, weighted by token index
        let token_hash = end_hash.wrapping_add(finalizers_hash);
        let weight = mix64((token_idx + 1) as u64);
        hash = hash.wrapping_add(token_hash.wrapping_mul(weight));
    }
    
    hash
}

/// Find state equivalence classes for a tokenizer.
///
/// # Arguments
/// * `regex` - The tokenizer DFA
/// * `tokens` - Vocabulary tokens to consider
/// * `states` - List of state IDs to analyze
///
/// # Returns
/// A vector where `result[i]` is the representative state for `states[i]`.
/// States with the same representative are equivalent.
pub fn find_state_equivalence_classes(
    regex: &Regex,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> Vec<usize> {
    let start = std::time::Instant::now();
    let dfa = &regex.dfa;
    
    // Precompute packed transition tables
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
    
    // Compute signatures for all states in parallel
    let signatures: Vec<u64> = states
        .par_iter()
        .map(|&state| {
            compute_state_signature(
                &dfa_transitions,
                &dfa_finalizers,
                &possible_futures,
                tokens,
                state,
            )
        })
        .collect();
    
    // Group states by signature
    let mut sig_groups: HashMap<u64, Vec<usize>> = HashMap::new();
    for (idx, &sig) in signatures.iter().enumerate() {
        sig_groups.entry(sig).or_default().push(idx);
    }
    
    // Build mapping: state index -> representative state ID
    let mut mapping = vec![0usize; states.len()];
    for members in sig_groups.values() {
        let rep_state_id = states[members[0]];
        for &idx in members {
            mapping[idx] = rep_state_id;
        }
    }
    
    let num_groups = sig_groups.len();
    crate::debug!(
        3,
        "State equiv reference: {} states -> {} groups in {:?}",
        states.len(),
        num_groups,
        start.elapsed(),
    );
    
    mapping
}

/// Convert a state-to-representative mapping to StateEquivalenceResult format.
pub fn mapping_to_equivalence_classes(states: &[usize], mapping: &[usize]) -> StateEquivalenceResult {
    let mut rep_to_class: std::collections::BTreeMap<usize, BTreeSet<usize>> = std::collections::BTreeMap::new();
    
    for (i, &rep) in mapping.iter().enumerate() {
        rep_to_class.entry(rep).or_default().insert(states[i]);
    }
    
    rep_to_class.into_values().collect()
}
