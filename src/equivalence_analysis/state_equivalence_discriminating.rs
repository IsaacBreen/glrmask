//! Discriminating Token State Equivalence Analysis
//!
//! This module implements an optimized state equivalence analysis using
//! "discriminating tokens" - a small subset of tokens that are sufficient
//! to distinguish between different equivalence classes.
//!
//! Key insight: Not all tokens are needed to distinguish states. If two states
//! behave the same for a diverse sample of tokens, they're likely equivalent.
//!
//! Algorithm:
//! 1. Test a small batch of diverse tokens
//! 2. If hash groups are stable (no changes), we're done
//! 3. Otherwise, add more tokens and repeat

use std::collections::{BTreeSet, HashMap};
use rayon::prelude::*;
use crate::finite_automata::Regex;

pub type StateEquivalenceResult = BTreeSet<BTreeSet<usize>>;

// -----------------------------------------------------------------------------
// Hashing Utilities
// -----------------------------------------------------------------------------

#[inline(always)]
fn mix_u128(mut x: u128) -> u128 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

const NONE_STATE: u32 = u32::MAX;

// -----------------------------------------------------------------------------
// Compute Token Hash for a State
// -----------------------------------------------------------------------------

#[inline]
fn compute_token_hash_for_state(
    token: &[u8],
    start_state: u32,
    token_weight: u128,
    dfa_transitions: &[[u32; 256]],
    dfa_finalizers: &[Vec<usize>],
    end_state_hashes: &[u128],
) -> u128 {
    let mut current = start_state;
    let mut finalizers_hash: u128 = 0;
    let mut depth: u32 = 0;
    
    for &byte in token {
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
    token_hash.wrapping_mul(token_weight)
}

// -----------------------------------------------------------------------------
// Discriminating Token Selection
// -----------------------------------------------------------------------------

/// Find state equivalence classes using discriminating token selection.
///
/// This approach iteratively tests batches of tokens until equivalence classes
/// stabilize. For most grammars, a small fraction of tokens is sufficient.
pub fn find_state_equivalence_classes_discriminating(
    regex: &Regex,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> Vec<usize> {
    let instant = std::time::Instant::now();
    let dfa = &regex.dfa;
    let num_states = states.len();
    let num_tokens = tokens.len();
    
    // Build packed transition tables
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
    
    // Generate token weights (same as original for consistency)
    let token_weights: Vec<u128> = (0..num_tokens)
        .map(|i| mix_u128(((i + 1) as u128).wrapping_mul(0x9E3779B97F4A7C15)))
        .collect();
    
    // Initialize state hashes with intrinsic properties
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
    
    // Track which states are still active (not in singleton groups)
    let mut active_mask: Vec<bool> = vec![true; num_states];
    
    // Group by hash
    let mut groups: HashMap<u128, Vec<usize>> = HashMap::new();
    for (i, &hash) in state_hashes.iter().enumerate() {
        groups.entry(hash).or_default().push(i);
    }
    
    // Process tokens in batches
    // Use larger batches for large state counts to reduce iteration overhead
    let batch_size = if num_states > 10000 { 
        25000.min(num_tokens) 
    } else { 
        10000.min(num_tokens) 
    };
    let mut tokens_tested = 0usize;
    let mut iteration = 0;
    let mut unchanged_iterations = 0usize;
    
    while tokens_tested < num_tokens {
        iteration += 1;
        
        // Mark singletons as inactive
        for indices in groups.values() {
            if indices.len() == 1 {
                active_mask[indices[0]] = false;
            }
        }
        let num_active = active_mask.iter().filter(|&&x| x).count();
        
        if num_active == 0 {
            crate::debug!(4, "Discriminating: all states in singleton groups after {} tokens", tokens_tested);
            break;
        }
        
        // Prepare next batch
        let batch_end = (tokens_tested + batch_size).min(num_tokens);
        let batch_weights: Vec<u128> = (tokens_tested..batch_end)
            .map(|i| token_weights[i])
            .collect();
        
        // Update hashes for active states only
        let updates: Vec<(usize, u128)> = (0..num_states)
            .into_par_iter()
            .filter_map(|i| {
                if !active_mask[i] {
                    return None;
                }
                
                let state = states[i] as u32;
                let mut hash_delta: u128 = 0;
                
                for (batch_idx, token_idx) in (tokens_tested..batch_end).enumerate() {
                    hash_delta = hash_delta.wrapping_add(
                        compute_token_hash_for_state(
                            &tokens[token_idx],
                            state,
                            batch_weights[batch_idx],
                            &dfa_transitions,
                            &dfa_finalizers,
                            &end_state_hashes,
                        )
                    );
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
        let prev_group_count = groups.len();
        groups.clear();
        for (i, &hash) in state_hashes.iter().enumerate() {
            groups.entry(hash).or_default().push(i);
        }
        
        if groups.len() == prev_group_count {
            unchanged_iterations += 1;
        } else {
            unchanged_iterations = 0;
        }
        
        crate::debug!(5, "Discriminating iteration {}: {} tokens, {} groups (was {}), {} active", 
                      iteration, tokens_tested, groups.len(), prev_group_count, num_active);
        
        // Early convergence: if groups haven't changed for 2 iterations, likely stable
        if unchanged_iterations >= 2 {
            crate::debug!(4, "Discriminating: early convergence after {} iterations ({} tokens)", 
                          iteration, tokens_tested);
            break;
        }
    }
    
    // Build final mapping
    let mut mapping: Vec<usize> = vec![0; num_states];
    for group in groups.values() {
        let rep = group[0];
        for &state in group {
            mapping[state] = rep;
        }
    }
    
    let result: Vec<usize> = mapping.iter().map(|&rep| states[rep]).collect();
    
    let num_representatives = groups.len();
    crate::debug!(3, "Discriminating state equivalence: {} states -> {} classes using {} tokens in {:.2?}", 
                  states.len(), num_representatives, tokens_tested, instant.elapsed());
    
    result
}

/// Convert a state-to-representative mapping to StateEquivalenceResult format.
pub fn mapping_to_equivalence_classes(states: &[usize], mapping: &[usize]) -> StateEquivalenceResult {
    let mut rep_to_class: std::collections::BTreeMap<usize, BTreeSet<usize>> = std::collections::BTreeMap::new();
    
    for (i, &rep) in mapping.iter().enumerate() {
        rep_to_class.entry(rep).or_default().insert(states[i]);
    }
    
    rep_to_class.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equivalence_analysis::state_equivalence_analysis_fast;
    use crate::finite_automata::{ExprGroups, ExprGroup, eat_u8_seq};
    
    fn build_simple_regex(patterns: &[&str]) -> crate::finite_automata::Regex {
        let groups: Vec<ExprGroup> = patterns.iter().map(|p| {
            ExprGroup {
                expr: eat_u8_seq(p.as_bytes().to_vec()),
                is_non_greedy: false,
            }
        }).collect();
        ExprGroups { groups }.build()
    }
    
    #[test]
    fn test_discriminating_correctness() {
        let regex = build_simple_regex(&["function", "func", "for", "if"]);
        
        let tokens = vec![
            b"function".to_vec(),
            b"functional".to_vec(),
            b"func".to_vec(),
            b"for".to_vec(),
            b"forEach".to_vec(),
            b"if".to_vec(),
            b"x".to_vec(),
        ];
        
        let states: Vec<usize> = regex.iter_states().map(|s| s.0).collect();
        
        let disc_result = find_state_equivalence_classes_discriminating(&regex, &tokens, &states);
        let orig_result = state_equivalence_analysis_fast::find_state_equivalence_classes(&regex, &tokens, &states);
        
        println!("Disc result: {:?}", disc_result);
        println!("Orig result: {:?}", orig_result);
        
        // They should produce the same equivalence classes
        assert_eq!(disc_result, orig_result);
    }
}
