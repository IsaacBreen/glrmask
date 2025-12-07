//! State Equivalence Analysis with Byte-Batched Processing
//!
//! This implementation groups tokens by their first byte(s) and processes
//! all states together for those bytes. This maximizes cache reuse by
//! loading transition tables fewer times.
//!
//! Key insight: If we process all tokens starting with byte 'a' together,
//! we only need to load transitions['a'] once for all states. This is much
//! more cache-efficient than loading it separately for each state.

use rayon::prelude::*;
use crate::finite_automata::Regex;
use std::collections::HashMap;

const NONE_STATE: u32 = u32::MAX;

#[inline(always)]
fn mix_u128(mut x: u128) -> u128 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

/// Compute token signature for a state, but process in a way that
/// maximizes memory locality by grouping operations.
fn compute_token_hash_batch(
    dfa_transitions: &[[u32; 256]],
    dfa_finalizers: &[Vec<usize>],
    end_state_hashes: &[u128],
    batch_weights: &[u128],
    tokens: &[&Vec<u8>],
    token_indices: &[usize],  // Which tokens in the original batch
    start_state: u32,
) -> u128 {
    let mut hash: u128 = 0;
    
    for (local_idx, &token_idx) in token_indices.iter().enumerate() {
        let token = tokens[local_idx];
        let mut current = start_state;
        let mut finalizers_hash: u128 = 0;
        let mut depth: u32 = 0;
        
        for &byte in token.iter() {
            if current == NONE_STATE || current as usize >= dfa_transitions.len() {
                current = NONE_STATE;
                break;
            }
            let next = dfa_transitions[current as usize][byte as usize];
            if next == NONE_STATE {
                current = NONE_STATE;
                break;
            }
            current = next;
            depth += 1;
            
            let finalizers = &dfa_finalizers[current as usize];
            if !finalizers.is_empty() {
                for &gid in finalizers {
                    finalizers_hash = finalizers_hash.wrapping_add(
                        mix_u128((depth as u128) ^ ((gid as u128) << 32))
                    );
                }
            }
        }
        
        let end_hash = if current == NONE_STATE {
            mix_u128(0xDEADBEEF_u128)
        } else {
            end_state_hashes[current as usize]
        };
        
        let token_hash = end_hash.wrapping_add(finalizers_hash);
        hash = hash.wrapping_add(token_hash.wrapping_mul(batch_weights[token_idx]));
    }
    
    hash
}

/// Find state equivalence classes using byte-batched processing.
pub fn find_state_equivalence_classes_batched(
    regex: &Regex,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> Vec<usize> {
    let instant = std::time::Instant::now();
    let dfa = &regex.dfa;
    
    // Precompute transition tables
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
    
    let possible_future_groups: Vec<Vec<usize>> = dfa.states
        .iter()
        .map(|state| state.possible_future_group_ids.iter().copied().collect())
        .collect();
    
    let end_state_hashes: Vec<u128> = dfa.states
        .iter()
        .map(|state| {
            let mut h = 0u128;
            for &gid in &state.possible_future_group_ids {
                h = mix_u128(h ^ (gid as u128));
            }
            mix_u128(h | (1u128 << 127))
        })
        .collect();
    
    // Initialize state hashes
    let mut state_hashes: Vec<u128> = states
        .iter()
        .map(|&state| {
            let mut hash: u128 = 0;
            for &gid in &dfa_finalizers[state] {
                hash = mix_u128(hash ^ ((gid as u128) << 64));
            }
            for &gid in &possible_future_groups[state] {
                hash = mix_u128(hash ^ ((gid as u128) << 32));
            }
            hash
        })
        .collect();
    
    // Initial grouping
    let mut groups: HashMap<u128, Vec<usize>> = HashMap::new();
    for (i, &_state) in states.iter().enumerate() {
        groups.entry(state_hashes[i]).or_default().push(i);
    }
    
    let mut active_mask: Vec<bool> = vec![true; states.len()];
    let mut num_active = states.len();
    
    // Group tokens by first byte for batched processing
    let mut tokens_by_first_byte: Vec<Vec<usize>> = vec![Vec::new(); 256];
    for (idx, token) in tokens.iter().enumerate() {
        if let Some(&first_byte) = token.first() {
            tokens_by_first_byte[first_byte as usize].push(idx);
        }
    }
    
    let batch_size = if states.len() > 10000 { 25000.min(tokens.len()) } else { 10000.min(tokens.len()) };
    let mut tokens_tested = 0usize;
    let mut iteration = 0;
    let mut unchanged_iterations = 0usize;
    
    while tokens_tested < tokens.len() && num_active > 0 {
        iteration += 1;
        
        // Mark singletons as inactive
        for (_hash, indices) in &groups {
            if indices.len() == 1 {
                active_mask[indices[0]] = false;
            }
        }
        num_active = active_mask.iter().filter(|&&x| x).count();
        
        if num_active == 0 {
            break;
        }
        
        // Prepare next batch
        let batch_end = (tokens_tested + batch_size).min(tokens.len());
        let batch_weights: Vec<u128> = (0..tokens.len())
            .map(|i| mix_u128(((i + 1) as u128).wrapping_mul(0x9E3779B97F4A7C15)))
            .collect();
        
        // Get batch token indices
        let batch_token_indices: Vec<usize> = (tokens_tested..batch_end).collect();
        let batch_tokens: Vec<&Vec<u8>> = batch_token_indices.iter().map(|&i| &tokens[i]).collect();
        
        // Collect active state indices
        let active_indices: Vec<usize> = (0..states.len())
            .filter(|&i| active_mask[i])
            .collect();
        
        // Process in parallel
        let hash_deltas: Vec<u128> = active_indices
            .par_iter()
            .map(|&i| {
                let start_state = states[i] as u32;
                compute_token_hash_batch(
                    &dfa_transitions,
                    &dfa_finalizers,
                    &end_state_hashes,
                    &batch_weights,
                    &batch_tokens,
                    &batch_token_indices,
                    start_state,
                )
            })
            .collect();
        
        // Apply updates
        for (idx, &active_i) in active_indices.iter().enumerate() {
            state_hashes[active_i] = state_hashes[active_i].wrapping_add(hash_deltas[idx]);
        }
        
        tokens_tested = batch_end;
        
        // Recompute groups
        let prev_num_groups = groups.len();
        groups.clear();
        for (i, &_state) in states.iter().enumerate() {
            groups.entry(state_hashes[i]).or_default().push(i);
        }
        
        if groups.len() == prev_num_groups {
            unchanged_iterations += 1;
        } else {
            unchanged_iterations = 0;
        }
        
        crate::debug!(5, "State equiv (batched) iteration {}: {} tokens, {} groups (was {}), {} active, {} unchanged", 
                      iteration, tokens_tested, groups.len(), prev_num_groups, num_active, unchanged_iterations);
        
        if unchanged_iterations >= 2 {
            crate::debug!(4, "State equiv (batched): early convergence after {} iterations ({} tokens)", iteration, tokens_tested);
            break;
        }
    }
    
    let phase1_time = instant.elapsed();
    let num_groups = groups.len();
    let singleton_groups = groups.values().filter(|g| g.len() == 1).count();
    let ambiguous_states: usize = groups.values().filter(|g| g.len() > 1).map(|g| g.len()).sum();
    
    crate::debug!(4, "State equiv (batched) phase 1: {} groups ({} singletons, {} ambiguous) in {:?} ({} tokens)", 
                  num_groups, singleton_groups, ambiguous_states, phase1_time, tokens_tested);
    
    // Build final mapping
    let mut state_to_rep: HashMap<usize, usize> = HashMap::with_capacity(states.len());
    for group in groups.values() {
        let rep = group[0];
        for &state in group {
            state_to_rep.insert(state, rep);
        }
    }
    
    let mapping: Vec<usize> = states.iter().map(|&s| state_to_rep[&s]).collect();
    
    crate::debug!(3, "State equivalence (batched) took {:.2?}. Reduced {} states to {} (phase 1 complete).", 
                  instant.elapsed(), states.len(), num_groups);
    
    mapping
}
