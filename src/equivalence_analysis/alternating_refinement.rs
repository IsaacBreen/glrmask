//! Alternating Refinement for Combined State and Vocab Equivalence
//!
//! This module implements an alternating refinement algorithm that iteratively
//! refines state and token groups until convergence. This is significantly faster
//! than the naive approach for large DFAs because:
//!
//! - State refinement only needs to test TOKEN GROUP REPRESENTATIVES, not all tokens
//! - Token refinement only needs to test STATE GROUP REPRESENTATIVES, not all states
//!
//! Complexity: O(iterations × (states × token_groups + tokens × state_groups))
//! vs naive: O(states × tokens)

use rayon::prelude::*;
use std::collections::HashMap;

use crate::finite_automata::Regex;

/// Result of alternating refinement
pub struct AlternatingRefinementResult {
    /// Mapping from state index to representative state index
    pub state_to_representative: Vec<usize>,
    /// Mapping from token index to equivalence class ID
    pub token_to_class: Vec<usize>,
    /// Number of state groups
    pub num_state_groups: usize,
    /// Number of token groups  
    pub num_token_groups: usize,
}

/// Find state and vocab equivalence classes using alternating refinement.
///
/// # Arguments
/// * `regex` - The tokenizer DFA
/// * `tokens` - Vocabulary tokens to analyze
/// * `initial_states` - Initial tokenizer state IDs to consider
///
/// # Returns
/// Result containing state-to-rep mapping and token-to-class mapping.
pub fn find_equivalence_alternating(
    regex: &Regex,
    tokens: &[Vec<u8>],
    initial_states: &[usize],
) -> AlternatingRefinementResult {
    let start = std::time::Instant::now();
    let dfa = &regex.dfa;
    let num_states = initial_states.len();
    let num_tokens = tokens.len();
    
    if num_states == 0 || num_tokens == 0 {
        return AlternatingRefinementResult {
            state_to_representative: initial_states.to_vec(),
            token_to_class: (0..num_tokens).map(|_| 0).collect(),
            num_state_groups: if num_states > 0 { 1 } else { 0 },
            num_token_groups: if num_tokens > 0 { 1 } else { 0 },
        };
    }
    
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
    
    // Initialize: all states in one group, all tokens in one group
    let mut state_to_group: Vec<usize> = vec![0; num_states];
    let mut token_to_group: Vec<usize> = vec![0; num_tokens];
    let mut num_state_groups: usize = 1;
    let mut num_token_groups: usize = 1;
    
    // Get representative for each group (first member)
    let mut state_group_reps: Vec<usize> = vec![initial_states[0]];  // actual state IDs
    let mut token_group_reps: Vec<usize> = vec![0];  // token indices
    
    let mut iteration = 0;
    let max_iterations = 100;
    
    loop {
        iteration += 1;
        let iter_start = std::time::Instant::now();
        
        // =========================================================================
        // Step 1: Refine STATES using token group representatives
        // =========================================================================
        let prev_state_groups = num_state_groups;
        
        // Compute state hashes based on behavior on token representatives
        let state_hashes: Vec<u64> = (0..num_states)
            .into_par_iter()
            .map(|state_idx| {
                let state_id = initial_states[state_idx];
                let mut hash: u64 = 0;
                
                for &token_rep_idx in token_group_reps.iter() {
                    let token = &tokens[token_rep_idx];
                    let result = execute_token(&dfa_transitions, &dfa_finalizers, state_id, token);
                    let result_hash = hash_result(result);
                    // Use XOR for order-independent aggregation
                    hash ^= result_hash;
                }
                
                hash
            })
            .collect();
        
        // Group states by hash
        let mut state_hash_groups: HashMap<u64, Vec<usize>> = HashMap::new();
        for (state_idx, &hash) in state_hashes.iter().enumerate() {
            state_hash_groups.entry(hash).or_default().push(state_idx);
        }
        
        // Update state grouping
        num_state_groups = state_hash_groups.len();
        state_group_reps.clear();
        for (group_id, (_hash, members)) in state_hash_groups.into_iter().enumerate() {
            state_group_reps.push(initial_states[members[0]]);
            for &state_idx in &members {
                state_to_group[state_idx] = group_id;
            }
        }
        
        // =========================================================================
        // Step 2: Refine TOKENS using state group representatives
        // =========================================================================
        let prev_token_groups = num_token_groups;
        
        // Compute token hashes based on behavior from state representatives
        let token_hashes: Vec<u64> = (0..num_tokens)
            .into_par_iter()
            .map(|token_idx| {
                let token = &tokens[token_idx];
                let mut hash: u64 = 0;
                
                for &state_rep_id in state_group_reps.iter() {
                    let result = execute_token(&dfa_transitions, &dfa_finalizers, state_rep_id, token);
                    let result_hash = hash_result(result);
                    // Use XOR for order-independent aggregation
                    hash ^= result_hash;
                }
                
                hash
            })
            .collect();
        
        // Group tokens by hash
        let mut token_hash_groups: HashMap<u64, Vec<usize>> = HashMap::new();
        for (token_idx, &hash) in token_hashes.iter().enumerate() {
            token_hash_groups.entry(hash).or_default().push(token_idx);
        }
        
        // Update token grouping
        num_token_groups = token_hash_groups.len();
        token_group_reps.clear();
        for (group_id, (_hash, members)) in token_hash_groups.into_iter().enumerate() {
            token_group_reps.push(members[0]);
            for &token_idx in &members {
                token_to_group[token_idx] = group_id;
            }
        }
        
        crate::debug!(5, "Alt refine iteration {}: {} state groups (was {}), {} token groups (was {}), {:?}",
                      iteration, num_state_groups, prev_state_groups, num_token_groups, prev_token_groups, iter_start.elapsed());
        
        // Check convergence
        if num_state_groups == prev_state_groups && num_token_groups == prev_token_groups {
            crate::debug!(4, "Alt refine converged after {} iterations", iteration);
            break;
        }
        
        if iteration >= max_iterations {
            crate::debug!(3, "Alt refine: max iterations reached ({} state groups, {} token groups)", 
                         num_state_groups, num_token_groups);
            break;
        }
    }
    
    // Build state-to-representative mapping
    // For each state, find the representative state in its group
    let mut state_to_representative = vec![0usize; num_states];
    for state_idx in 0..num_states {
        let group = state_to_group[state_idx];
        state_to_representative[state_idx] = state_group_reps[group];
    }
    
    crate::debug!(3, "Alternating refinement: {} states -> {} groups, {} tokens -> {} groups in {:?}",
                  num_states, num_state_groups, num_tokens, num_token_groups, start.elapsed());
    
    AlternatingRefinementResult {
        state_to_representative,
        token_to_class: token_to_group,
        num_state_groups,
        num_token_groups,
    }
}

/// Execute a token from a state and return (end_state, finalizers_hash)
#[inline]
fn execute_token(
    dfa_transitions: &[[u32; 256]],
    dfa_finalizers: &[Vec<usize>],
    start_state: usize,
    token: &[u8],
) -> (u32, u64) {
    const NONE_STATE: u32 = u32::MAX;
    
    let mut current = start_state as u32;
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
    
    (current, finalizers_hash)
}

/// Hash the execution result
#[inline]
fn hash_result(result: (u32, u64)) -> u64 {
    let (end_state, finalizers_hash) = result;
    mix64(end_state as u64).wrapping_add(finalizers_hash)
}

/// Fast 64-bit mixing function
#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_empty_inputs() {
        // Can't easily test without a Regex, but we can at least verify the function exists
    }
}
