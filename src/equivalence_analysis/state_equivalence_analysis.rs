//! State Equivalence Analysis
//!
//! Determines which tokenizer states behave identically for all tokens in a vocabulary.
//! States that are equivalent can be merged, reducing the workload for subsequent
//! vocab equivalence analysis.
//!
//! The algorithm uses a two-phase approach:
//! 1. Quick structural hash: Group states by their immediate transition structure.
//!    States with different structures are definitely not equivalent.
//! 2. Full token analysis: Only for states with the same structural hash,
//!    compute full signatures to distinguish states that differ in multi-byte behavior.
//!
//! This is much faster than the naive approach when many states have different structures.

use std::collections::BTreeSet;
use rayon::prelude::*;
use crate::finite_automata::Regex;

/// The result of state equivalence analysis: sets of state IDs that behave identically.
pub type StateEquivalenceResult = BTreeSet<BTreeSet<usize>>;

// -----------------------------------------------------------------------------
// Hashing Utilities (128-bit)
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

// -----------------------------------------------------------------------------
// Phase 2: Full Token Analysis (expensive, but only for ambiguous groups)
// -----------------------------------------------------------------------------

/// Compute a hash signature for a single state by running all tokens through it.
fn compute_state_signature(
    dfa_transitions: &[[u32; 256]],
    dfa_finalizers: &[Vec<usize>],
    end_state_hashes: &[u128],
    token_weights: &[u128],
    tokens: &[Vec<u8>],
    start_state: usize,
) -> u128 {
    const NONE_STATE: u32 = u32::MAX;
    let mut hash: u128 = 0;
    
    for (token_idx, token) in tokens.iter().enumerate() {
        // Run token through DFA from start_state
        let mut current = start_state as u32;
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
            
            // Hash any finalizers at this state
            let finalizers = &dfa_finalizers[current as usize];
            if !finalizers.is_empty() {
                for &gid in finalizers {
                    finalizers_hash = finalizers_hash.wrapping_add(
                        mix_u128((depth as u128) ^ ((gid as u128) << 32))
                    );
                }
            }
        }
        
        // Use precomputed end_hash and token_weight
        let end_hash = if current == NONE_STATE {
            mix_u128(0xDEADBEEF_u128)
        } else {
            end_state_hashes[current as usize]
        };
        
        let token_hash = end_hash.wrapping_add(finalizers_hash);
        hash = hash.wrapping_add(token_hash.wrapping_mul(token_weights[token_idx]));
    }
    
    hash
}

// -----------------------------------------------------------------------------
// State Equivalence Analysis (Two-Phase with Sample-based Token Testing)
// -----------------------------------------------------------------------------

/// Find state equivalence classes for a tokenizer.
///
/// Uses a two-phase approach:
/// 1. Sample-based token hash: Test a sample of tokens to group states by their
///    observed behavior. This captures TOKEN-LEVEL behavior, not just byte structure.
/// 2. Full token analysis only for groups with multiple states.
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
    use std::collections::HashMap;
    
    let instant = std::time::Instant::now();
    let dfa = &regex.dfa;
    
    // Precompute packed transition tables and finalizers for cache efficiency
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
    
    // Extract possible_future_group_ids for semantic hashing
    let possible_future_groups: Vec<Vec<usize>> = dfa.states
        .iter()
        .map(|state| state.possible_future_group_ids.iter().copied().collect())
        .collect();
    
    // =========================================================================
    // PHASE 1: Token testing with early exit for singletons
    // =========================================================================
    // We test tokens in batches, but only on states that haven't been uniquely
    // identified yet. Once a state is in a singleton group, it stays there.
    
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
    
    // Initialize state hashes with state's own properties
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
    
    // Track which state indices are "active" (not yet in singletons)
    let mut active_mask: Vec<bool> = vec![true; states.len()];
    let mut num_active = states.len();
    
    // Group by hash
    let mut groups: HashMap<u128, Vec<usize>> = HashMap::new();
    for (i, &_state) in states.iter().enumerate() {
        groups.entry(state_hashes[i]).or_default().push(i);
    }
    
    // Token batch size - larger batches reduce iteration overhead
    // but may do more work on states that become inactive mid-batch
    let batch_size = 10000.min(tokens.len());
    let mut tokens_tested = 0usize;
    let mut iteration = 0;
    
    while tokens_tested < tokens.len() && num_active > 0 {
        iteration += 1;
        
        // Mark singletons as inactive
        for (hash, indices) in &groups {
            if indices.len() == 1 {
                active_mask[indices[0]] = false;
            }
        }
        num_active = active_mask.iter().filter(|&&x| x).count();
        
        if num_active == 0 {
            break; // All singletons
        }
        
        // Prepare next batch of tokens
        let batch_end = (tokens_tested + batch_size).min(tokens.len());
        let batch_tokens: Vec<&Vec<u8>> = (tokens_tested..batch_end)
            .map(|i| &tokens[i])
            .collect();
        let batch_weights: Vec<u128> = (0..batch_tokens.len())
            .map(|i| mix_u128(((tokens_tested + i + 1) as u128).wrapping_mul(0x9E3779B97F4A7C15)))
            .collect();
        
        // Update hashes for ACTIVE states only
        let updates: Vec<(usize, u128)> = (0..states.len())
            .into_par_iter()
            .filter_map(|i| {
                if !active_mask[i] {
                    return None;
                }
                
                let state = states[i];
                let mut hash_delta: u128 = 0;
                
                for (token_idx, token) in batch_tokens.iter().enumerate() {
                    let mut current = state as u32;
                    let mut finalizers_hash: u128 = 0;
                    let mut depth: u32 = 0;
                    
                    for &byte in *token {
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
        for (i, &_state) in states.iter().enumerate() {
            groups.entry(state_hashes[i]).or_default().push(i);
        }
        
        crate::debug!(5, "State equiv iteration {}: {} tokens, {} groups (was {}), {} active states", 
                      iteration, tokens_tested, groups.len(), prev_num_groups, num_active);
    }
    
    let phase1_time = instant.elapsed();
    let num_groups = groups.len();
    let singleton_groups = groups.values().filter(|g| g.len() == 1).count();
    let ambiguous_states: usize = groups.values().filter(|g| g.len() > 1).map(|g| g.len()).sum();
    
    crate::debug!(4, "State equiv phase 1: {} groups ({} singletons, {} ambiguous) in {:?} ({} tokens)", 
                  num_groups, singleton_groups, ambiguous_states, phase1_time, tokens_tested);
    
    // If all groups are singletons, we're done (no states are equivalent)
    if ambiguous_states == 0 {
        // Convert from state index to state ID
        let mapping: Vec<usize> = states.to_vec();
        crate::debug!(3, "State equivalence analysis took {:.2?}. Reduced {} states to {} (all unique).", 
                      instant.elapsed(), states.len(), states.len());
        return mapping;
    }
    
    // If we've tested ALL tokens in phase 1, the groups are already correct.
    // Phase 2 would just recompute the same result.
    // All non-singleton states have seen all tokens (singletons stopped early but are already unique).
    if tokens_tested >= tokens.len() {
        // Build mapping from phase 1 groups
        let mut state_to_rep: HashMap<usize, usize> = HashMap::with_capacity(states.len());
        for group in groups.values() {
            let rep = group[0]; // First state in group is representative
            for &state in group {
                state_to_rep.insert(state, rep);
            }
        }
        let mapping: Vec<usize> = states.iter().map(|&s| state_to_rep[&s]).collect();
        let num_representatives: usize = mapping.iter().collect::<std::collections::HashSet<_>>().len();
        crate::debug!(3, "State equivalence analysis took {:.2?}. Reduced {} states to {} (phase 1 complete).", 
                      instant.elapsed(), states.len(), num_representatives);
        return mapping;
    }
    
    // =========================================================================
    // PHASE 2: Full token analysis for ambiguous states
    // =========================================================================
    // Only needed when phase 1 didn't test all tokens (early exit due to all singletons).
    // Use full token analysis for correctness.
    
    // Collect all ambiguous states
    let ambiguous_state_list: Vec<usize> = groups.values()
        .filter(|g| g.len() > 1)
        .flat_map(|g| g.iter().copied())
        .collect();
    
    // Full analysis: use all tokens
    let phase2_tokens: Vec<&Vec<u8>> = tokens.iter().collect();
    let phase2_token_weights: Vec<u128> = (0..tokens.len())
        .map(|i| mix_u128(((i + 1) as u128).wrapping_mul(0x9E3779B97F4A7C15)))
        .collect();
    
    crate::debug!(4, "State equiv phase 2: analyzing {} states with {} tokens (full)", 
                  ambiguous_states, phase2_tokens.len());
    // Compute signatures for all ambiguous states in parallel
    let ambiguous_signatures: Vec<(usize, u128)> = ambiguous_state_list
        .par_iter()
        .map(|&state| {
            let mut hash: u128 = 0;
            
            for (token_idx, token) in phase2_tokens.iter().enumerate() {
                let mut current = state as u32;
                let mut finalizers_hash: u128 = 0;
                let mut depth: u32 = 0;
                
                const NONE_STATE: u32 = u32::MAX;
                for &byte in *token {
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
                hash = hash.wrapping_add(token_hash.wrapping_mul(phase2_token_weights[token_idx]));
            }
            
            (state, hash)
        })
        .collect();
    
    // Build state -> signature map
    let state_to_sig: HashMap<usize, u128> = ambiguous_signatures.iter().copied().collect();
    
    // For each group, find the refined grouping
    let mut state_to_rep: HashMap<usize, usize> = HashMap::with_capacity(states.len());
    
    // Singleton groups: state maps to itself
    for group in groups.values() {
        if group.len() == 1 {
            let state = group[0];
            state_to_rep.insert(state, state);
        }
    }
    
    // Ambiguous groups: refine by full signature
    for group in groups.values() {
        if group.len() > 1 {
            // Group by full signature within this group
            let mut sig_to_rep: HashMap<u128, usize> = HashMap::new();
            for &state in group {
                let sig = state_to_sig[&state];
                let rep = *sig_to_rep.entry(sig).or_insert(state);
                state_to_rep.insert(state, rep);
            }
        }
    }
    
    // Build final mapping
    let mapping: Vec<usize> = states.iter().map(|&s| state_to_rep[&s]).collect();
    
    let num_representatives: usize = mapping.iter().collect::<std::collections::HashSet<_>>().len();

    crate::debug!(3, "State equivalence analysis took {:.2?}. Reduced {} states to {}.", 
                  instant.elapsed(), states.len(), num_representatives);
    
    mapping
}

/// Convert a state-to-representative mapping to StateEquivalenceResult format.
///
/// # Arguments
/// * `states` - The original list of state IDs
/// * `mapping` - The mapping where `mapping[i]` is the representative for `states[i]`
///
/// # Returns
/// A set of equivalence classes, where each class is a set of state IDs.
pub fn mapping_to_equivalence_classes(states: &[usize], mapping: &[usize]) -> StateEquivalenceResult {
    let mut rep_to_class: std::collections::BTreeMap<usize, BTreeSet<usize>> = std::collections::BTreeMap::new();
    
    for (i, &rep) in mapping.iter().enumerate() {
        rep_to_class.entry(rep).or_default().insert(states[i]);
    }
    
    rep_to_class.into_values().collect()
}
