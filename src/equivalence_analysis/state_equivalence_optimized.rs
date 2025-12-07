//! State Equivalence Analysis with Flattened Token Buffer
//!
//! This implementation pre-flattens all tokens into a single contiguous buffer
//! to improve cache locality during token processing.
//!
//! Key optimizations:
//! 1. Flattened token buffer: All token bytes in one contiguous array
//! 2. Reduced branching: Pre-check for empty finalizers
//! 3. Optimized inner loop: Minimize per-byte overhead

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

/// Flattened representation of tokens for cache-efficient access
struct FlatTokens {
    /// All token bytes concatenated
    bytes: Vec<u8>,
    /// Start offset for each token in bytes
    offsets: Vec<usize>,
}

impl FlatTokens {
    fn new(tokens: &[Vec<u8>]) -> Self {
        let total_bytes: usize = tokens.iter().map(|t| t.len()).sum();
        let mut bytes = Vec::with_capacity(total_bytes);
        let mut offsets = Vec::with_capacity(tokens.len() + 1);
        
        offsets.push(0);
        for token in tokens {
            bytes.extend_from_slice(token);
            offsets.push(bytes.len());
        }
        
        Self { bytes, offsets }
    }
    
    #[inline]
    fn get(&self, idx: usize) -> &[u8] {
        let start = self.offsets[idx];
        let end = self.offsets[idx + 1];
        &self.bytes[start..end]
    }
    
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }
}

/// Precomputed DFA data with optimized layout
struct OptimizedDfa {
    /// Transition table: transitions[state * 256 + byte] = next_state
    /// Flat layout for better cache behavior
    transitions_flat: Vec<u32>,
    /// Number of states
    num_states: usize,
    /// Finalizers for each state (empty vec if none)
    finalizers: Vec<Vec<usize>>,
    /// Precomputed end-state hashes
    end_hashes: Vec<u128>,
    /// Whether state has non-empty finalizers (for branch prediction)
    has_finalizers: Vec<bool>,
}

impl OptimizedDfa {
    fn new(regex: &Regex) -> Self {
        let dfa = &regex.dfa;
        let num_states = dfa.states.len();
        
        // Flat transition table: transitions_flat[state * 256 + byte] = next_state
        let mut transitions_flat = vec![NONE_STATE; num_states * 256];
        for (state_id, state) in dfa.states.iter().enumerate() {
            let base = state_id * 256;
            for (byte, &target) in state.transitions.iter() {
                transitions_flat[base + byte as usize] = target as u32;
            }
        }
        
        let finalizers: Vec<Vec<usize>> = dfa.states
            .iter()
            .map(|state| state.finalizers.iter().collect())
            .collect();
        
        let has_finalizers: Vec<bool> = finalizers.iter().map(|f| !f.is_empty()).collect();
        
        let end_hashes: Vec<u128> = dfa.states
            .iter()
            .map(|state| {
                let mut h = 0u128;
                for &gid in &state.possible_future_group_ids {
                    h = mix_u128(h ^ (gid as u128));
                }
                mix_u128(h | (1u128 << 127))
            })
            .collect();
        
        Self {
            transitions_flat,
            num_states,
            finalizers,
            end_hashes,
            has_finalizers,
        }
    }
    
    #[inline(always)]
    fn transition(&self, state: u32, byte: u8) -> u32 {
        if state == NONE_STATE || state as usize >= self.num_states {
            NONE_STATE
        } else {
            self.transitions_flat[(state as usize) * 256 + byte as usize]
        }
    }
}

/// Compute hash delta for a single state over a batch of tokens
#[inline]
fn compute_state_hash_delta(
    dfa: &OptimizedDfa,
    flat_tokens: &FlatTokens,
    batch_range: std::ops::Range<usize>,
    batch_weights: &[u128],
    batch_start: usize,
    start_state: u32,
) -> u128 {
    let mut hash_delta: u128 = 0;
    
    for token_idx in batch_range {
        let token = flat_tokens.get(token_idx);
        let mut current = start_state;
        let mut finalizers_hash: u128 = 0;
        let mut depth: u32 = 0;
        
        for &byte in token {
            let next = dfa.transition(current, byte);
            if next == NONE_STATE {
                current = NONE_STATE;
                break;
            }
            current = next;
            depth += 1;
            
            // Only check finalizers if state has them (branch prediction hint)
            if dfa.has_finalizers[current as usize] {
                for &gid in &dfa.finalizers[current as usize] {
                    finalizers_hash = finalizers_hash.wrapping_add(
                        mix_u128((depth as u128) ^ ((gid as u128) << 32))
                    );
                }
            }
        }
        
        let end_hash = if current == NONE_STATE {
            mix_u128(0xDEADBEEF_u128)
        } else {
            dfa.end_hashes[current as usize]
        };
        
        let token_hash = end_hash.wrapping_add(finalizers_hash);
        let weight_idx = token_idx - batch_start;
        hash_delta = hash_delta.wrapping_add(token_hash.wrapping_mul(batch_weights[weight_idx]));
    }
    
    hash_delta
}

/// Find state equivalence classes using optimized data layout
pub fn find_state_equivalence_classes_optimized(
    regex: &Regex,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> Vec<usize> {
    let instant = std::time::Instant::now();
    let dfa = &regex.dfa;
    
    // Build optimized DFA
    let opt_dfa = OptimizedDfa::new(regex);
    
    // Flatten tokens for cache-efficient access
    let flat_tokens = FlatTokens::new(tokens);
    
    // Get possible_future_groups for initial hashing
    let possible_future_groups: Vec<Vec<usize>> = dfa.states
        .iter()
        .map(|state| state.possible_future_group_ids.iter().copied().collect())
        .collect();
    
    // Initialize state hashes
    let mut state_hashes: Vec<u128> = states
        .iter()
        .map(|&state| {
            let mut hash: u128 = 0;
            for &gid in &opt_dfa.finalizers[state] {
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
    for (i, _) in states.iter().enumerate() {
        groups.entry(state_hashes[i]).or_default().push(i);
    }
    
    let mut active_mask: Vec<bool> = vec![true; states.len()];
    let mut num_active = states.len();
    
    let batch_size = if states.len() > 10000 { 25000.min(flat_tokens.len()) } else { 10000.min(flat_tokens.len()) };
    let mut tokens_tested = 0usize;
    let mut iteration = 0;
    let mut unchanged_iterations = 0usize;
    
    while tokens_tested < flat_tokens.len() && num_active > 0 {
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
        let batch_start = tokens_tested;
        let batch_end = (tokens_tested + batch_size).min(flat_tokens.len());
        let batch_weights: Vec<u128> = (0..(batch_end - batch_start))
            .map(|i| mix_u128(((batch_start + i + 1) as u128).wrapping_mul(0x9E3779B97F4A7C15)))
            .collect();
        
        // Collect active state indices
        let active_indices: Vec<usize> = (0..states.len())
            .filter(|&i| active_mask[i])
            .collect();
        
        // Process in parallel
        let hash_deltas: Vec<u128> = active_indices
            .par_iter()
            .map(|&i| {
                compute_state_hash_delta(
                    &opt_dfa,
                    &flat_tokens,
                    batch_start..batch_end,
                    &batch_weights,
                    batch_start,
                    states[i] as u32,
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
        for (i, _) in states.iter().enumerate() {
            groups.entry(state_hashes[i]).or_default().push(i);
        }
        
        if groups.len() == prev_num_groups {
            unchanged_iterations += 1;
        } else {
            unchanged_iterations = 0;
        }
        
        crate::debug!(5, "State equiv (optimized) iteration {}: {} tokens, {} groups (was {}), {} active, {} unchanged", 
                      iteration, tokens_tested, groups.len(), prev_num_groups, num_active, unchanged_iterations);
        
        if unchanged_iterations >= 2 {
            crate::debug!(4, "State equiv (optimized): early convergence after {} iterations ({} tokens)", iteration, tokens_tested);
            break;
        }
    }
    
    let phase1_time = instant.elapsed();
    let num_groups = groups.len();
    let singleton_groups = groups.values().filter(|g| g.len() == 1).count();
    let ambiguous_states: usize = groups.values().filter(|g| g.len() > 1).map(|g| g.len()).sum();
    
    crate::debug!(4, "State equiv (optimized) phase 1: {} groups ({} singletons, {} ambiguous) in {:?} ({} tokens)", 
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
    
    crate::debug!(3, "State equivalence (optimized) took {:.2?}. Reduced {} states to {} (phase 1 complete).", 
                  instant.elapsed(), states.len(), num_groups);
    
    mapping
}
