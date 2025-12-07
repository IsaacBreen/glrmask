//! State Equivalence Analysis with u16 State IDs
//!
//! This implementation uses u16 instead of u32 for state IDs to halve memory bandwidth.
//! Since the diff grammar has ~37K states (< 65536), u16 is sufficient.
//!
//! Memory reduction:
//! - Transition table: 37K × 256 × 4 = 38MB → 37K × 256 × 2 = 19MB
//! - Better cache utilization for random accesses

use rayon::prelude::*;
use crate::finite_automata::Regex;
use std::collections::HashMap;

const NONE_STATE_U16: u16 = u16::MAX;

#[inline(always)]
fn mix_u128(mut x: u128) -> u128 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

/// Find state equivalence classes using u16 state IDs
pub fn find_state_equivalence_classes_u16(
    regex: &Regex,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> Vec<usize> {
    let instant = std::time::Instant::now();
    let dfa = &regex.dfa;
    
    // Check if u16 is sufficient
    if dfa.states.len() > 65535 {
        crate::debug!(3, "DFA has {} states, falling back to u32 implementation", dfa.states.len());
        return super::state_equivalence_analysis_fast::find_state_equivalence_classes(regex, tokens, states);
    }
    
    // Precompute packed transition tables using u16
    let dfa_transitions: Vec<[u16; 256]> = dfa.states
        .iter()
        .map(|state| {
            let mut table = [NONE_STATE_U16; 256];
            for (byte, &target) in state.transitions.iter() {
                table[byte as usize] = target as u16;
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
    
    // Precompute end state hashes
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
        .par_iter()
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
    
    let mut active_mask: Vec<bool> = vec![true; states.len()];
    let mut num_active = states.len();
    
    let mut groups: HashMap<u128, Vec<usize>> = HashMap::new();
    for (i, _) in states.iter().enumerate() {
        groups.entry(state_hashes[i]).or_default().push(i);
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
        
        // Prepare batch
        let batch_end = (tokens_tested + batch_size).min(tokens.len());
        let batch_tokens: Vec<&Vec<u8>> = (tokens_tested..batch_end).map(|i| &tokens[i]).collect();
        let batch_weights: Vec<u128> = (0..batch_tokens.len())
            .map(|i| mix_u128(((tokens_tested + i + 1) as u128).wrapping_mul(0x9E3779B97F4A7C15)))
            .collect();
        
        // Update hashes - using u16 transitions
        let updates: Vec<(usize, u128)> = (0..states.len())
            .into_par_iter()
            .filter_map(|i| {
                if !active_mask[i] {
                    return None;
                }
                
                let state = states[i];
                let mut hash_delta: u128 = 0;
                
                for (token_idx, token) in batch_tokens.iter().enumerate() {
                    let mut current = state as u16;
                    let mut finalizers_hash: u128 = 0;
                    let mut depth: u32 = 0;
                    
                    for &byte in *token {
                        let next = dfa_transitions[current as usize][byte as usize];
                        if next == NONE_STATE_U16 {
                            current = NONE_STATE_U16;
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
                    
                    let end_hash = if current == NONE_STATE_U16 {
                        mix_u128(0xDEADBEEF_u128)
                    } else {
                        end_state_hashes[current as usize]
                    };
                    
                    let token_hash = end_hash.wrapping_add(finalizers_hash);
                    hash_delta = hash_delta.wrapping_add(token_hash.wrapping_mul(batch_weights[token_idx]));
                }
                
                Some((i, hash_delta))
            })
            .collect();
        
        // Apply updates
        for (i, delta) in updates {
            state_hashes[i] = state_hashes[i].wrapping_add(delta);
        }
        
        tokens_tested = batch_end;
        
        // Recompute groups
        let prev_num_groups = groups.len();
        groups.clear();
        for (i, _) in states.iter().enumerate() {
            groups.entry(state_hashes[i]).or_default().push(i);
        }
        
        if groups.len() == prev_num_groups {
            unchanged_iterations += 1;
        } else {
            unchanged_iterations = 0;
        }
        
        crate::debug!(5, "State equiv (u16) iteration {}: {} tokens, {} groups (was {}), {} active, {} unchanged", 
                      iteration, tokens_tested, groups.len(), prev_num_groups, num_active, unchanged_iterations);
        
        if unchanged_iterations >= 2 {
            crate::debug!(4, "State equiv (u16): early convergence after {} iterations ({} tokens)", iteration, tokens_tested);
            break;
        }
    }
    
    let phase1_time = instant.elapsed();
    let num_groups = groups.len();
    let singleton_groups = groups.values().filter(|g| g.len() == 1).count();
    let ambiguous_states: usize = groups.values().filter(|g| g.len() > 1).map(|g| g.len()).sum();
    
    crate::debug!(4, "State equiv (u16) phase 1: {} groups ({} singletons, {} ambiguous) in {:?} ({} tokens)", 
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
    
    crate::debug!(3, "State equivalence (u16) took {:.2?}. Reduced {} states to {} (phase 1 complete).", 
                  instant.elapsed(), states.len(), num_groups);
    
    mapping
}
