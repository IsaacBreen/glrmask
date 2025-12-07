//! Trie-based State Equivalence Analysis
//!
//! This module provides an optimized version of state equivalence analysis
//! that uses a trie to share prefix computations across tokens.
//!
//! The key insight is that many tokens share common prefixes. By building a
//! trie of all tokens and walking each state through the trie, we avoid
//! redundant DFA transitions for shared prefixes.
//!
//! Complexity analysis:
//! - Original: O(states × tokens × avg_token_len)
//! - Trie: O(states × trie_nodes)
//! - With ~2x prefix sharing, trie approach is ~2x faster

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

// -----------------------------------------------------------------------------
// Compact Trie Structure (no per-node state storage)
// -----------------------------------------------------------------------------

/// A node in the token trie (compact representation)
struct TrieNode {
    /// Children: (byte, child_index) pairs, sorted by byte
    children: Vec<(u8, usize)>,
    /// Token indices that end at this node
    token_indices: Vec<usize>,
}

/// Token trie for efficient prefix sharing
struct TokenTrie {
    nodes: Vec<TrieNode>,
}

impl TokenTrie {
    /// Build a trie from tokens
    fn build(tokens: &[Vec<u8>]) -> Self {
        let mut nodes = vec![TrieNode { children: Vec::new(), token_indices: Vec::new() }];
        
        for (idx, token) in tokens.iter().enumerate() {
            let mut current = 0usize;
            for &byte in token {
                // Find or create child for this byte
                let child_idx = {
                    if let Some(pos) = nodes[current].children.iter().position(|(b, _)| *b == byte) {
                        nodes[current].children[pos].1
                    } else {
                        let new_idx = nodes.len();
                        nodes.push(TrieNode { children: Vec::new(), token_indices: Vec::new() });
                        nodes[current].children.push((byte, new_idx));
                        new_idx
                    }
                };
                current = child_idx;
            }
            nodes[current].token_indices.push(idx);
        }
        
        TokenTrie { nodes }
    }
    
    /// Count total nodes
    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

const NONE_STATE: u32 = u32::MAX;

// -----------------------------------------------------------------------------
// Trie-based State Equivalence (no per-node storage)
// -----------------------------------------------------------------------------

/// Find state equivalence classes using trie-based prefix sharing.
/// 
/// This version walks the trie for each state in parallel, avoiding the need
/// to store per-state data at each trie node.
pub fn find_state_equivalence_classes_trie(
    regex: &Regex,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> Vec<usize> {
    let instant = std::time::Instant::now();
    let dfa = &regex.dfa;
    let num_states = states.len();
    
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
    
    // Precompute end state hashes (encodes possible_future_groups)
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
    
    // Build trie
    let trie_start = std::time::Instant::now();
    let trie = TokenTrie::build(tokens);
    crate::debug!(4, "Built token trie: {} nodes for {} tokens in {:?}", 
                  trie.num_nodes(), tokens.len(), trie_start.elapsed());
    
    // Generate token weights
    let token_weights: Vec<u128> = (0..tokens.len())
        .map(|i| mix_u128(((i + 1) as u128).wrapping_mul(0x9E3779B97F4A7C15)))
        .collect();
    
    // Compute state hashes in parallel - each state walks the entire trie
    let hash_start = std::time::Instant::now();
    let state_hashes: Vec<u128> = states
        .par_iter()
        .map(|&start_state| {
            let mut hash: u128 = 0;
            
            // Hash initial state properties (same as original algorithm)
            for &gid in &dfa_finalizers[start_state] {
                hash = mix_u128(hash ^ ((gid as u128) << 64));
            }
            for &gid in &possible_future_groups[start_state] {
                hash = mix_u128(hash ^ ((gid as u128) << 32));
            }
            
            // Walk trie for this state
            walk_trie_for_state(
                &trie,
                0,  // start at root
                start_state as u32,
                0u128,  // accumulated finalizer hash
                1u32,   // depth
                &dfa_transitions,
                &dfa_finalizers,
                &end_state_hashes,
                &token_weights,
                &mut hash,
            );
            
            hash
        })
        .collect();
    
    crate::debug!(4, "Computed {} state hashes in {:?}", num_states, hash_start.elapsed());
    
    // Group states by hash
    let mut groups: HashMap<u128, Vec<usize>> = HashMap::new();
    for (i, &hash) in state_hashes.iter().enumerate() {
        groups.entry(hash).or_default().push(i);
    }
    
    // Build mapping
    let mut mapping: Vec<usize> = vec![0; num_states];
    for group in groups.values() {
        let rep = group[0];
        for &state in group {
            mapping[state] = rep;
        }
    }
    
    // Convert to state IDs
    let result: Vec<usize> = mapping.iter().map(|&rep| states[rep]).collect();
    
    let num_representatives = mapping.iter().collect::<std::collections::HashSet<_>>().len();
    crate::debug!(3, "Trie-based state equivalence: {} states -> {} classes in {:.2?}", 
                  states.len(), num_representatives, instant.elapsed());
    
    result
}

/// Recursively walk the trie for a single state, accumulating hash contributions
#[inline(never)]
fn walk_trie_for_state(
    trie: &TokenTrie,
    node_idx: usize,
    current_dfa_state: u32,
    accumulated_finalizer_hash: u128,
    depth: u32,
    dfa_transitions: &[[u32; 256]],
    dfa_finalizers: &[Vec<usize>],
    end_state_hashes: &[u128],
    token_weights: &[u128],
    hash_accumulator: &mut u128,
) {
    let node = &trie.nodes[node_idx];
    
    // Process any tokens that end at this node
    for &token_idx in &node.token_indices {
        let end_hash = if current_dfa_state == NONE_STATE {
            mix_u128(0xDEADBEEF_u128)
        } else {
            end_state_hashes[current_dfa_state as usize]
        };
        
        let token_hash = end_hash.wrapping_add(accumulated_finalizer_hash);
        *hash_accumulator = hash_accumulator.wrapping_add(token_hash.wrapping_mul(token_weights[token_idx]));
    }
    
    // If we're in a dead state, all children are also dead
    if current_dfa_state == NONE_STATE {
        // Still need to process children for tokens, but all will be dead
        for &(byte, child_idx) in &node.children {
            walk_trie_for_state(
                trie,
                child_idx,
                NONE_STATE,
                accumulated_finalizer_hash,
                depth + 1,
                dfa_transitions,
                dfa_finalizers,
                end_state_hashes,
                token_weights,
                hash_accumulator,
            );
        }
        return;
    }
    
    // Process children
    for &(byte, child_idx) in &node.children {
        let next_state = dfa_transitions[current_dfa_state as usize][byte as usize];
        
        // Update finalizer hash if we reached a valid state with finalizers
        let mut new_finalizer_hash = accumulated_finalizer_hash;
        if next_state != NONE_STATE {
            let finalizers = &dfa_finalizers[next_state as usize];
            for &gid in finalizers {
                new_finalizer_hash = new_finalizer_hash.wrapping_add(
                    mix_u128((depth as u128) ^ ((gid as u128) << 32))
                );
            }
        }
        
        walk_trie_for_state(
            trie,
            child_idx,
            next_state,
            new_finalizer_hash,
            depth + 1,
            dfa_transitions,
            dfa_finalizers,
            end_state_hashes,
            token_weights,
            hash_accumulator,
        );
    }
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
    
    /// Helper to build a simple regex from string patterns
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
    fn test_trie_build() {
        let tokens = vec![
            b"hello".to_vec(),
            b"hell".to_vec(),
            b"help".to_vec(),
            b"world".to_vec(),
        ];
        
        let trie = TokenTrie::build(&tokens);
        println!("Trie has {} nodes for {} tokens", trie.num_nodes(), tokens.len());
        assert!(trie.num_nodes() < tokens.len() * 5); // Should have fewer nodes due to sharing
    }
    
    #[test]
    fn test_trie_correctness_simple_regex() {
        // Create a simple regex that matches "a", "ab", "abc"
        let regex = build_simple_regex(&["a", "ab", "abc"]);
        
        // Some test tokens that exercise different paths
        let tokens = vec![
            b"a".to_vec(),
            b"ab".to_vec(),
            b"abc".to_vec(),
            b"abcd".to_vec(),
            b"b".to_vec(),
        ];
        
        // Get all states
        let states: Vec<usize> = regex.iter_states().map(|s| s.0).collect();
        
        println!("Testing regex with {} states, {} tokens", states.len(), tokens.len());
        
        // Compare trie-based vs original
        let trie_result = find_state_equivalence_classes_trie(&regex, &tokens, &states);
        let orig_result = state_equivalence_analysis_fast::find_state_equivalence_classes(&regex, &tokens, &states);
        
        println!("Trie result: {:?}", trie_result);
        println!("Orig result: {:?}", orig_result);
        
        // Both should produce the same state -> representative mapping
        assert_eq!(trie_result, orig_result, "Trie and original should produce same mappings");
    }
    
    #[test]
    fn test_trie_correctness_with_shared_prefixes() {
        // Regex with overlapping patterns
        let regex = build_simple_regex(&["function", "func", "for", "if"]);
        
        // Tokens with shared prefixes
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
        
        println!("Testing regex with {} states, {} tokens", states.len(), tokens.len());
        
        let trie_result = find_state_equivalence_classes_trie(&regex, &tokens, &states);
        let orig_result = state_equivalence_analysis_fast::find_state_equivalence_classes(&regex, &tokens, &states);
        
        println!("Trie result: {:?}", trie_result);
        println!("Orig result: {:?}", orig_result);
        
        assert_eq!(trie_result, orig_result, "Trie and original should produce same mappings");
    }
}
