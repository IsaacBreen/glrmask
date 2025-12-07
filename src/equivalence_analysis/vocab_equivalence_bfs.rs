//! Level-Order Trie-Based Vocab Equivalence Analysis
//!
//! This module implements a BFS (level-order) trie traversal for computing
//! vocabulary token equivalence classes. Unlike DFS which has poor cache locality,
//! BFS processes all nodes at the same depth together, allowing:
//! - Batch processing of state transitions
//! - Better cache utilization
//! - Potential for SIMD optimization
//!
//! The algorithm:
//! 1. Build a trie from all tokens
//! 2. Process trie level-by-level (BFS)
//! 3. At each level, batch-process byte transitions for all active nodes
//! 4. Compute signatures for tokens ending at each level

use crate::finite_automata::Regex;
use crate::r#macro::is_debug_level_enabled;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::HashMap;

// =============================================================================
// TRIE DATA STRUCTURE
// =============================================================================

#[derive(Default)]
struct TokenTrieNode {
    /// Transitions: (byte, child_idx)
    transitions: SmallVec<[(u8, u32); 8]>,
    /// Token indices ending at this node
    token_indices: SmallVec<[usize; 1]>,
}

struct TokenTrie {
    nodes: Vec<TokenTrieNode>,
}

impl TokenTrie {
    fn new() -> Self {
        TokenTrie {
            nodes: vec![TokenTrieNode::default()],
        }
    }

    fn insert(&mut self, token: &[u8], token_idx: usize) {
        let mut node_idx = 0usize;
        for &byte in token {
            let mut found = None;
            for &(b, child) in &self.nodes[node_idx].transitions {
                if b == byte {
                    found = Some(child as usize);
                    break;
                }
            }
            match found {
                Some(child) => node_idx = child,
                None => {
                    let new_idx = self.nodes.len();
                    self.nodes.push(TokenTrieNode::default());
                    self.nodes[node_idx].transitions.push((byte, new_idx as u32));
                    node_idx = new_idx;
                }
            }
        }
        self.nodes[node_idx].token_indices.push(token_idx);
    }
}

// =============================================================================
// PRECOMPUTED DFA (simplified for this algorithm)
// =============================================================================

const NONE_STATE: u32 = u32::MAX;
const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;

#[inline(always)]
fn mix_u64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

struct PrecomputedDfa {
    transitions: Vec<[u32; 256]>,
    /// Finalizer mask: which groups can match at each state
    /// For efficiency, stored as a u64 bitmask (supports up to 64 groups)
    finalizer_mask: Vec<u64>,
    /// has_transitions[state] = true if state has any outgoing transitions
    has_transitions: Vec<bool>,
    num_groups: usize,
    /// Completion hash for each state
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
}

fn precompute_dfa(dfa: &crate::finite_automata::DFA, num_groups: usize) -> PrecomputedDfa {
    let num_states = dfa.states.len();
    
    let mut transitions = vec![[NONE_STATE; 256]; num_states];
    for (state_idx, state) in dfa.states.iter().enumerate() {
        for (byte, &next_state) in &state.transitions {
            transitions[state_idx][byte as usize] = next_state as u32;
        }
    }
    
    let finalizer_mask: Vec<u64> = dfa.states
        .iter()
        .map(|s| {
            let mut mask = 0u64;
            for gid in s.finalizers.iter() {
                if gid < 64 && gid < num_groups {
                    mask |= 1u64 << gid;
                }
            }
            mask
        })
        .collect();
    
    let has_transitions: Vec<bool> = dfa.states
        .iter()
        .map(|s| !s.transitions.is_empty())
        .collect();
    
    let none_completion_hash = mix_u64(HASH_SEED1 ^ 0xDEAD);
    let completion_hash: Vec<u64> = dfa.states
        .iter()
        .map(|s| {
            let mut h = 0u64;
            for &gid in &s.possible_future_group_ids {
                h = mix_u64(h ^ (gid as u64));
            }
            mix_u64(h | (1 << 63))
        })
        .collect();
    
    PrecomputedDfa {
        transitions,
        finalizer_mask,
        has_transitions,
        num_groups,
        completion_hash,
        none_completion_hash,
    }
}

// =============================================================================
// LEVEL-ORDER BFS ALGORITHM
// =============================================================================

/// State for a single "active" trie node being processed
#[derive(Clone)]
struct ActiveNode {
    /// Index in the trie
    trie_idx: u32,
    /// Index into the state_arrays storage
    /// state_arrays[states_offset..states_offset+num_initial_states] = current DFA states
    states_offset: u32,
}

/// Find vocab equivalence classes using level-order trie traversal.
pub fn find_vocab_equivalence_classes_bfs(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> Vec<Vec<usize>> {
    let instant = std::time::Instant::now();
    
    if strings.is_empty() || initial_states.is_empty() {
        return vec![];
    }
    
    let num_tokens = strings.len();
    let num_states = initial_states.len();
    
    let num_groups = regex.dfa.states.iter()
        .flat_map(|s| s.finalizers.iter())
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    
    crate::debug!(4, "  BFS trie vocab equiv: {} tokens, {} states, {} groups",
                  num_tokens, num_states, num_groups);
    
    // Build token trie
    let trie_start = std::time::Instant::now();
    let mut trie = TokenTrie::new();
    for (idx, token) in strings.iter().enumerate() {
        trie.insert(token, idx);
    }
    crate::debug!(5, "    Trie: {} nodes in {:?}", trie.nodes.len(), trie_start.elapsed());
    
    // Precompute DFA
    let pre = precompute_dfa(&regex.dfa, num_groups);
    
    // State storage: flat array for all active nodes' DFA states
    // Each active node needs num_states u32 values
    let mut state_storage: Vec<u32> = Vec::with_capacity(num_states * 256);
    let mut match_storage: Vec<u64> = Vec::with_capacity(256);  // Bitmask of matched groups
    
    // Initialize with root node
    let init_offset = state_storage.len() as u32;
    for &s in initial_states {
        state_storage.push(s as u32);
    }
    match_storage.push(0);  // No groups matched initially
    
    let mut current_level: Vec<ActiveNode> = vec![ActiveNode {
        trie_idx: 0,
        states_offset: init_offset,
    }];
    let mut next_level: Vec<ActiveNode> = Vec::new();
    
    // Signature map
    let mut signature_groups: HashMap<u64, Vec<usize>> = HashMap::new();
    
    // Process level by level
    let mut depth = 0u32;
    let bfs_start = std::time::Instant::now();
    
    while !current_level.is_empty() {
        // Process tokens ending at this level
        for active in &current_level {
            let trie_node = &trie.nodes[active.trie_idx as usize];
            if !trie_node.token_indices.is_empty() {
                let sig = compute_signature(
                    &pre,
                    &state_storage[active.states_offset as usize..(active.states_offset as usize + num_states)],
                    match_storage[active.states_offset as usize / num_states],
                    num_groups,
                );
                for &token_idx in &trie_node.token_indices {
                    signature_groups.entry(sig).or_default().push(token_idx);
                }
            }
        }
        
        // Clear storage for next level (will reallocate as needed)
        let next_storage_start = state_storage.len();
        let next_match_start = match_storage.len();
        
        // Process transitions to next level
        for active in &current_level {
            let trie_node = &trie.nodes[active.trie_idx as usize];
            
            for &(byte, child_trie_idx) in &trie_node.transitions {
                // Compute new states after this byte transition
                let new_offset = state_storage.len() as u32;
                let mut any_active = false;
                let mut new_match_mask = match_storage[active.states_offset as usize / num_states];
                
                for i in 0..num_states {
                    let current = state_storage[active.states_offset as usize + i];
                    if current == NONE_STATE {
                        state_storage.push(NONE_STATE);
                    } else {
                        let next = pre.transitions[current as usize][byte as usize];
                        state_storage.push(next);
                        if next != NONE_STATE {
                            any_active = true;
                            // Update match mask
                            new_match_mask |= pre.finalizer_mask[next as usize];
                        }
                    }
                }
                
                match_storage.push(new_match_mask);
                
                // Only keep active nodes (nodes where at least one DFA state can still transition)
                if any_active {
                    next_level.push(ActiveNode {
                        trie_idx: child_trie_idx,
                        states_offset: new_offset,
                    });
                } else {
                    // Still need to process this node for tokens ending here
                    // But no need to continue deeper
                    let child_trie_node = &trie.nodes[child_trie_idx as usize];
                    if !child_trie_node.token_indices.is_empty() {
                        // Compute signature for tokens at this dead end
                        let sig = compute_signature(
                            &pre,
                            &state_storage[new_offset as usize..(new_offset as usize + num_states)],
                            new_match_mask,
                            num_groups,
                        );
                        for &token_idx in &child_trie_node.token_indices {
                            signature_groups.entry(sig).or_default().push(token_idx);
                        }
                    }
                }
            }
        }
        
        // Swap levels
        std::mem::swap(&mut current_level, &mut next_level);
        next_level.clear();
        depth += 1;
        
        // Trim old storage (we don't need previous level's states anymore)
        // This is a heuristic - we keep the storage but could trim if memory is tight
    }
    
    crate::debug!(5, "    BFS traverse in {:?}, max depth {}", bfs_start.elapsed(), depth);
    crate::debug!(4, "  BFS trie: {} equiv classes in {:?}", 
                  signature_groups.len(), instant.elapsed());
    
    signature_groups.into_values().collect()
}

fn compute_signature(
    pre: &PrecomputedDfa,
    states: &[u32],
    match_mask: u64,
    num_groups: usize,
) -> u64 {
    let mut h = HASH_SEED1;
    
    for &state in states {
        let end_hash = if state == NONE_STATE || !pre.has_transitions[state as usize] {
            pre.none_completion_hash
        } else {
            pre.completion_hash[state as usize]
        };
        h = mix_u64(h ^ end_hash);
    }
    
    // Include match information
    h = mix_u64(h ^ match_mask);
    
    h
}
