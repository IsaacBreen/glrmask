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
use crate::dfa_u8::Tokenizer;

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
/// 
/// The signature captures the (group_id, final_position) pairs and the final end_state.
/// - Greedy groups: use LAST position
/// - Non-greedy groups: use FIRST position
/// This matches the trellis semantics where only one match per group is recorded.
fn compute_state_signature(
    dfa_transitions: &[[u32; 256]],
    dfa_finalizers: &[Vec<usize>],
    _possible_futures: &[Vec<usize>],
    end_state_hashes: &[u128],
    token_weights: &[u128],
    tokens: &[Vec<u8>],
    non_greedy_finalizers: &std::collections::BTreeSet<usize>,
    start_state: usize,
) -> u128 {
    const NONE_STATE: u32 = u32::MAX;
    let mut hash: u128 = 0;
    
    for (token_idx, token) in tokens.iter().enumerate() {
        // Run token through DFA from start_state
        let mut current = start_state as u32;
        let mut dead_at_depth: Option<usize> = None;
        
        // Track (group_id, position) - we'll need to hash this at the end
        // Using BTreeMap for ordered hashing (matches reference)
        let mut matches: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        
        for (depth, &byte) in token.iter().enumerate() {
            let next = dfa_transitions[current as usize][byte as usize];
            if next == NONE_STATE {
                dead_at_depth = Some(depth + 1);
                current = NONE_STATE;
                break;
            }
            current = next;
            let position = depth + 1;
            
            // Record matches with proper greedy/non-greedy semantics
            for &gid in &dfa_finalizers[current as usize] {
                if non_greedy_finalizers.contains(&gid) {
                    // Non-greedy: keep first
                    matches.entry(gid).or_insert(position);
                } else {
                    // Greedy: keep last
                    matches.insert(gid, position);
                }
            }
        }
        
        // Hash structure: dead position OR (matches + end state)
        let structure_hash: u128;
        let end_hash: u128;
        
        if let Some(dead_depth) = dead_at_depth {
            structure_hash = mix_u128((dead_depth as u128) ^ 0xDEAD_DEAD_DEAD_DEAD);
            end_hash = mix_u128(0xDEADBEEF_u128);
        } else {
            // Hash (group_id, position) pairs
            let mut sh = mix_u128(matches.len() as u128 | (1u128 << 48));
            for (&gid, &pos) in &matches {
                sh = sh.wrapping_add(mix_u128((gid as u128) | ((pos as u128) << 32)));
            }
            structure_hash = sh;
            end_hash = end_state_hashes[current as usize];
        }
        
        let token_hash = end_hash.wrapping_add(structure_hash);
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
    regex: &Tokenizer,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> Vec<usize> {
    use std::collections::HashMap;
    
    let instant = std::time::Instant::now();
    let dfa = regex.dfa();
    
    // Note: Token sampling (STATE_EQUIV_MAX_TOKENS) was tested but causes correctness issues.
    // Sampled state equivalence doesn't fully capture distinguishing states,
    // leading to incorrect vocab class merging. Keep this disabled.
    // 
    // let max_tokens = std::env::var("STATE_EQUIV_MAX_TOKENS")
    //     .ok()
    //     .and_then(|s| s.parse::<usize>().ok())
    //     .unwrap_or(tokens.len());
    
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
    
    // Count states with finalizers for optimization insight
    let states_with_finalizers = dfa_finalizers.iter().filter(|f| !f.is_empty()).count();
    crate::debug!(5, "DFA stats: {} states, {} with finalizers ({:.1}%)", 
                  dfa.states.len(), 
                  states_with_finalizers, 
                  100.0 * states_with_finalizers as f64 / dfa.states.len() as f64);
    
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
    // CRITICAL: Must include the COUNT of possible futures in the hash!
    // Otherwise sets like [0, 6] and [6] would hash the same because mix(0) = 0.
    let end_state_hashes: Vec<u128> = dfa.states
        .iter()
        .map(|state| {
            // Seed with the length to distinguish sets of different sizes
            let futures = &state.possible_future_group_ids;
            let mut h = mix_u128(futures.len() as u128 | (1u128 << 48));
            // Add (not XOR!) each element's hash for commutativity and collision resistance
            // NOTE: Match reference by NOT adding extra bits to gid
            for &gid in futures {
                h = h.wrapping_add(mix_u128(gid as u128));
            }
            // Flag to distinguish from dead state hash
            h | (1u128 << 127)
        })
        .collect();
    
    // Initialize state hashes to zero (like reference implementation)
    let mut state_hashes: Vec<u128> = vec![0u128; states.len()];
    
    // Get non-greedy finalizers for proper position tracking
    let non_greedy_finalizers = &dfa.non_greedy_finalizers;
    
    // Process tokens in batches for memory efficiency, but process ALL tokens for ALL states
    // to ensure correct equivalence (no early singleton exit optimization)
    let batch_size = if states.len() > 10000 { 
        25000.min(tokens.len())
    } else { 
        10000.min(tokens.len())
    };
    let mut tokens_tested = 0usize;
    
    while tokens_tested < tokens.len() {
        // Prepare next batch of tokens
        let batch_end = (tokens_tested + batch_size).min(tokens.len());
        let batch_tokens: Vec<&Vec<u8>> = (tokens_tested..batch_end)
            .map(|i| &tokens[i])
            .collect();
        
        // Precompute token weights based on global token index (for consistent hashing)
        let batch_weights: Vec<u128> = (tokens_tested..batch_end)
            .map(|i| mix_u128((i + 1) as u128))  // Match reference: mix(token_index + 1)
            .collect();
        
        // Update hashes for ALL states
        let updates: Vec<(usize, u128)> = (0..states.len())
            .into_par_iter()
            .map(|i| {
                let state = states[i];
                let mut hash_delta: u128 = 0;
                
                for (batch_idx, token) in batch_tokens.iter().enumerate() {
                    let mut current = state as u32;
                    let mut dead_at_depth: Option<usize> = None;
                    
                    // Track (group_id, position) with proper greedy/non-greedy semantics
                    let mut matches: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
                    
                    for (depth, &byte) in (*token).iter().enumerate() {
                        // Safety check - should never trigger since we break after setting NONE_STATE
                        if current == NONE_STATE {
                            dead_at_depth = Some(depth);
                            break;
                        }
                        let next = dfa_transitions[current as usize][byte as usize];
                        if next == NONE_STATE {
                            dead_at_depth = Some(depth + 1);
                            current = NONE_STATE;
                            break;
                        }
                        current = next;
                        let position = depth + 1;
                        
                        // Record matches with proper greedy/non-greedy semantics
                        for &gid in &dfa_finalizers[current as usize] {
                            if non_greedy_finalizers.contains(&gid) {
                                // Non-greedy: keep first
                                matches.entry(gid).or_insert(position);
                            } else {
                                // Greedy: keep last
                                matches.insert(gid, position);
                            }
                        }
                    }
                    
                    // Hash structure: dead position OR (matches + end state)
                    let structure_hash: u128;
                    let end_hash: u128;
                    
                    if let Some(dead_depth) = dead_at_depth {
                        // Token leads to dead state - hash the dead depth
                        structure_hash = mix_u128((dead_depth as u128) ^ 0xDEAD_DEAD_DEAD_DEAD);
                        end_hash = mix_u128(0xDEADBEEF_u128);
                    } else {
                        // Token is valid - hash the (group_id, position) pairs
                        let mut sh = mix_u128(matches.len() as u128 | (1u128 << 48));
                        for (&gid, &pos) in &matches {
                            sh = sh.wrapping_add(mix_u128((gid as u128) | ((pos as u128) << 32)));
                        }
                        structure_hash = sh;
                        
                        // Hash end state possible_futures (precomputed)
                        end_hash = end_state_hashes[current as usize];
                    }
                    
                    let token_hash = end_hash.wrapping_add(structure_hash);
                    hash_delta = hash_delta.wrapping_add(token_hash.wrapping_mul(batch_weights[batch_idx]));
                }
                
                (i, hash_delta)
            })
            .collect();
        
        // Apply updates
        for (i, delta) in updates {
            state_hashes[i] = state_hashes[i].wrapping_add(delta);
        }
        
        tokens_tested = batch_end;
    }
    
    // Group by final hash
    let mut groups: HashMap<u128, Vec<usize>> = HashMap::new();
    for (i, &hash) in state_hashes.iter().enumerate() {
        groups.entry(hash).or_default().push(i);
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
        // Note: groups contains state indices (positions in `states`), not state IDs
        let mut mapping = vec![0usize; states.len()];
        for group in groups.values() {
            let rep_state_id = states[group[0]]; // Convert representative index to state ID
            for &idx in group {
                mapping[idx] = rep_state_id;
            }
        }
        let num_representatives: usize = mapping.iter().collect::<std::collections::HashSet<_>>().len();
        crate::debug!(3, "State equivalence analysis took {:.2?}. Reduced {} states to {} (phase 1 complete).", 
                      instant.elapsed(), states.len(), num_representatives);
        return mapping;
    }
    
    // PHASE 2: Full token analysis for ambiguous states
    // =========================================================================
    // Only needed when phase 1 didn't test all tokens (early exit due to all singletons).
    // Use full token analysis for correctness.
    
    // Collect all ambiguous state indices (positions in `states` array)
    let ambiguous_idx_list: Vec<usize> = groups.values()
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
    // Returns (state_index, signature) pairs
    let ambiguous_signatures: Vec<(usize, u128)> = ambiguous_idx_list
        .par_iter()
        .map(|&idx| {
            let state = states[idx];  // Convert index to actual state ID
            let mut hash: u128 = 0;
            
            for (token_idx, token) in phase2_tokens.iter().enumerate() {
                let mut current = state as u32;
                let mut dead_at_depth: Option<usize> = None;
                
                // Track (group_id, position) with proper greedy/non-greedy semantics
                let mut matches: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
                
                const NONE_STATE: u32 = u32::MAX;
                for (depth, &byte) in (*token).iter().enumerate() {
                    let next = dfa_transitions[current as usize][byte as usize];
                    if next == NONE_STATE {
                        dead_at_depth = Some(depth + 1);
                        current = NONE_STATE;
                        break;
                    }
                    current = next;
                    let position = depth + 1;
                    
                    // Record matches with proper greedy/non-greedy semantics
                    for &gid in &dfa_finalizers[current as usize] {
                        if non_greedy_finalizers.contains(&gid) {
                            // Non-greedy: keep first
                            matches.entry(gid).or_insert(position);
                        } else {
                            // Greedy: keep last
                            matches.insert(gid, position);
                        }
                    }
                }
                
                // Hash structure: dead position OR (matches + end state)
                let structure_hash: u128;
                let end_hash: u128;
                
                if let Some(dead_depth) = dead_at_depth {
                    // Token leads to dead state - hash the dead depth
                    structure_hash = mix_u128((dead_depth as u128) ^ 0xDEAD_DEAD_DEAD_DEAD);
                    end_hash = mix_u128(0xDEADBEEF_u128);
                } else {
                    // Token is valid - hash the (group_id, position) pairs
                    let mut sh = mix_u128(matches.len() as u128 | (1u128 << 48));
                    for (&gid, &pos) in &matches {
                        sh = sh.wrapping_add(mix_u128((gid as u128) | ((pos as u128) << 32)));
                    }
                    structure_hash = sh;
                    
                    // Hash end state possible_futures (precomputed)
                    end_hash = end_state_hashes[current as usize];
                }
                
                let token_hash = end_hash.wrapping_add(structure_hash);
                hash = hash.wrapping_add(token_hash.wrapping_mul(phase2_token_weights[token_idx]));
            }
            
            (idx, hash)  // Return index, not state ID
        })
        .collect();
    
    // Build index -> signature map
    let idx_to_sig: HashMap<usize, u128> = ambiguous_signatures.iter().copied().collect();
    
    // Build final mapping: mapping[index] = representative state ID
    let mut mapping = vec![0usize; states.len()];
    
    // Singleton groups: state maps to itself
    for group in groups.values() {
        if group.len() == 1 {
            let idx = group[0];
            mapping[idx] = states[idx];  // Map to own state ID
        }
    }
    
    // Ambiguous groups: refine by full signature
    for group in groups.values() {
        if group.len() > 1 {
            // Group by full signature within this group
            let mut sig_to_rep: HashMap<u128, usize> = HashMap::new();
            for &idx in group {
                let sig = idx_to_sig[&idx];
                let rep_idx = *sig_to_rep.entry(sig).or_insert(idx);
                mapping[idx] = states[rep_idx];  // Map to representative's state ID
            }
        }
    }
    
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
