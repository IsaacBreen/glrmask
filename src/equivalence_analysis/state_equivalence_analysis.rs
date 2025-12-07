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
// State Equivalence Analysis (Two-Phase)
// -----------------------------------------------------------------------------

/// Find state equivalence classes for a tokenizer.
///
/// Uses a two-phase approach:
/// 1. Quick semantic hash to group states by their immediate behavior.
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
    // PHASE 1: Quick dual hash (structural + semantic)
    // =========================================================================
    // For diff-like grammars: structural hash gives many unique groups
    // For JS-like grammars: semantic hash gives meaningful groups
    // We compute both and use whichever gives FEWER ambiguous states.
    
    let hashes: Vec<(u128, u128)> = states
        .par_iter()
        .map(|&state| {
            let trans = &dfa_transitions[state];
            let fin = &dfa_finalizers[state];
            let pfg = &possible_future_groups[state];
            
            let mut structural_h = 0u128;
            let mut semantic_h = 0u128;
            
            // Hash this state's own behavior (same for both)
            for &gid in fin {
                structural_h = mix_u128(structural_h ^ ((gid as u128) << 64));
                semantic_h = mix_u128(semantic_h ^ ((gid as u128) << 64));
            }
            for &gid in pfg {
                structural_h = mix_u128(structural_h ^ ((gid as u128) << 32));
                semantic_h = mix_u128(semantic_h ^ ((gid as u128) << 32));
            }
            
            // Hash each byte transition
            const NONE_STATE: u32 = u32::MAX;
            for (byte, &target) in trans.iter().enumerate() {
                if target != NONE_STATE {
                    // Structural: use target ID
                    structural_h = mix_u128(structural_h ^ ((byte as u128) << 40) ^ (target as u128));
                    
                    // Semantic: use target's behavior
                    let target_fin = &dfa_finalizers[target as usize];
                    let target_pfg = &possible_future_groups[target as usize];
                    
                    let mut target_hash = byte as u128;
                    for &gid in target_fin {
                        target_hash = mix_u128(target_hash ^ ((gid as u128) << 64));
                    }
                    for &gid in target_pfg {
                        target_hash = mix_u128(target_hash ^ (gid as u128));
                    }
                    semantic_h = mix_u128(semantic_h ^ target_hash);
                }
            }
            (structural_h, semantic_h)
        })
        .collect();
    
    // Group by structural hash
    let mut structural_groups: HashMap<u128, Vec<usize>> = HashMap::new();
    for (i, &state) in states.iter().enumerate() {
        structural_groups.entry(hashes[i].0).or_default().push(state);
    }
    let structural_ambiguous: usize = structural_groups.values().filter(|g| g.len() > 1).map(|g| g.len()).sum();
    
    // Group by semantic hash
    let mut semantic_groups: HashMap<u128, Vec<usize>> = HashMap::new();
    for (i, &state) in states.iter().enumerate() {
        semantic_groups.entry(hashes[i].1).or_default().push(state);
    }
    let semantic_ambiguous: usize = semantic_groups.values().filter(|g| g.len() > 1).map(|g| g.len()).sum();
    
    // Use whichever hash gives fewer ambiguous states, BUT:
    // - If semantic would be fast (< 5000 states), prefer it because it may find more equivalences
    // - Only prefer structural if semantic would be very expensive
    let semantic_threshold = 5000; // ~100ms for 5000 states
    let (groups, hash_type) = if structural_ambiguous < semantic_ambiguous && semantic_ambiguous > semantic_threshold {
        // Semantic is expensive and structural is cheaper -> use structural
        (structural_groups, "structural")
    } else if structural_ambiguous == 0 && semantic_ambiguous > 0 && semantic_ambiguous <= semantic_threshold {
        // Structural gives no info but semantic is cheap -> use semantic
        (semantic_groups, "semantic")
    } else if structural_ambiguous <= semantic_ambiguous {
        (structural_groups, "structural")
    } else {
        (semantic_groups, "semantic")
    };
    
    let phase1_time = instant.elapsed();
    let num_groups = groups.len();
    let singleton_groups = groups.values().filter(|g| g.len() == 1).count();
    let ambiguous_states: usize = groups.values().filter(|g| g.len() > 1).map(|g| g.len()).sum();
    
    crate::debug!(4, "State equiv phase 1 ({}): {} groups ({} singletons, {} states need full analysis) in {:?}", 
                  hash_type, num_groups, singleton_groups, ambiguous_states, phase1_time);
    
    // If all groups are singletons, we're done (no states are equivalent)
    if ambiguous_states == 0 {
        let mapping: Vec<usize> = states.to_vec();
        crate::debug!(3, "State equivalence analysis took {:.2?}. Reduced {} states to {} (all unique).", 
                      instant.elapsed(), states.len(), states.len());
        return mapping;
    }
    
    // =========================================================================
    // PHASE 2: Full token analysis for ambiguous states
    // =========================================================================
    // Compute full signatures for all ambiguous states in parallel.
    
    // Precompute token weights and end state hashes
    let token_weights: Vec<u128> = (0..tokens.len())
        .map(|i| mix_u128(((i + 1) as u128).wrapping_mul(0x9E3779B97F4A7C15)))
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
    
    // Collect all ambiguous states
    let ambiguous_state_list: Vec<usize> = groups.values()
        .filter(|g| g.len() > 1)
        .flat_map(|g| g.iter().copied())
        .collect();
    
    // Compute signatures for all ambiguous states in parallel
    let ambiguous_signatures: Vec<(usize, u128)> = ambiguous_state_list
        .par_iter()
        .map(|&state| {
            let sig = compute_state_signature(
                &dfa_transitions,
                &dfa_finalizers,
                &end_state_hashes,
                &token_weights,
                tokens,
                state,
            );
            (state, sig)
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
