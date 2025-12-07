//! State Equivalence Analysis
//!
//! Determines which tokenizer states behave identically for all tokens in a vocabulary.
//! States that are equivalent can be merged, reducing the workload for subsequent
//! vocab equivalence analysis.
//!
//! The algorithm uses a trie-based approach with weighted contributions:
//! - Build a trie from all vocabulary tokens
//! - For each state, compute a hash signature by traversing the trie
//! - States with identical signatures are equivalent
//!
//! Complexity: O(trie_size × unique_groups) where groups are DFA states reachable from initial states.

use std::collections::BTreeSet;
use crate::finite_automata::Regex;

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

#[inline(always)]
fn hash_match_event(gid: u32, depth: u32) -> u128 {
    let packed = ((depth as u128) << 64) | (gid as u128);
    mix_u128(packed)
}

// -----------------------------------------------------------------------------
// Trie Definition for Tokens
// -----------------------------------------------------------------------------

#[derive(Default, Clone)]
struct TokenTrieNode {
    transitions: Vec<(u8, u32)>, // (byte, child_idx)
    // Sum of weights of all tokens in this subtree
    subtree_weight: u128,
    // Weight of the token ending exactly at this node (if any)
    token_weight: u128,
}

struct TokenTrie {
    nodes: Vec<TokenTrieNode>,
}

impl TokenTrie {
    fn new() -> Self {
        TokenTrie { nodes: vec![TokenTrieNode::default()] }
    }

    fn insert(&mut self, s: &[u8], weight: u128) {
        let mut node_idx = 0;
        for &b in s {
            let mut found = None;
            for &(byte, child) in &self.nodes[node_idx].transitions {
                if byte == b {
                    found = Some(child as usize);
                    break;
                }
            }
            match found {
                Some(child) => node_idx = child,
                None => {
                    let new_node_idx = self.nodes.len();
                    self.nodes.push(TokenTrieNode::default());
                    self.nodes[node_idx].transitions.push((b, new_node_idx as u32));
                    node_idx = new_node_idx;
                }
            }
        }
        self.nodes[node_idx].token_weight = self.nodes[node_idx].token_weight.wrapping_add(weight);
    }

    fn compute_subtree_weights(&mut self, node_idx: usize) {
        let mut sum = self.nodes[node_idx].token_weight;
        // Clone transitions to avoid borrow checker issues during recursion
        let transitions = self.nodes[node_idx].transitions.clone();
        for &(_, child_idx) in &transitions {
            self.compute_subtree_weights(child_idx as usize);
            sum = sum.wrapping_add(self.nodes[child_idx as usize].subtree_weight);
        }
        self.nodes[node_idx].subtree_weight = sum;
    }
}

// -----------------------------------------------------------------------------
// State Equivalence Analysis
// -----------------------------------------------------------------------------

/// Find state equivalence classes for a tokenizer.
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
    let instant = std::time::Instant::now();
    
    // 1. Build Token Trie
    let mut trie = TokenTrie::new();
    for (i, token) in tokens.iter().enumerate() {
        // Assign a random weight to each token index to distinguish them
        // Use i+1 to avoid zero weight for first token
        let w = mix_u128(((i + 1) as u128).wrapping_mul(0x9E3779B97F4A7C15));
        trie.insert(token, w);
    }
    trie.compute_subtree_weights(0);

    // 2. Precompute end state hashes based on possible_future_group_ids
    // When we reach the end of a token, it's not the end state that matters,
    // but the possible terminals accessible from that end state.
    let mut end_state_hashes = Vec::with_capacity(regex.dfa.states.len());
    for state in &regex.dfa.states {
        let mut h = 0u128;
        for &gid in &state.possible_future_group_ids {
            // Mix gid into hash. BTreeSet iteration is deterministic (sorted).
            h = mix_u128(h ^ (gid as u128));
        }
        // Distinguish end state hash from match hash by setting high bit
        end_state_hashes.push(mix_u128(h | (1u128 << 127)));
    }

    // 3. Compute signatures (single pass)
    let mut accumulators = vec![0u128; states.len()];

    // Prepare initial groups for DFS
    // Map: current_dfa_state -> List of original_state_indices (u32)
    let mut initial_groups: Vec<(Vec<u32>, u32)> = Vec::new();
    let mut map: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
    for (idx_in_input, &s) in states.iter().enumerate() {
        map.entry(s as u32).or_default().push(idx_in_input as u32);
    }
    for (dfa_st, list) in map {
        initial_groups.push((list, dfa_st));
    }

    process_node(regex, &trie, 0, initial_groups, &mut accumulators, 0, &end_state_hashes);

    // 4. Generate final mapping to representatives
    let mut hash_to_rep: std::collections::HashMap<u128, usize> = std::collections::HashMap::new();
    let mut mapping = vec![0; states.len()];
    
    for (i, &s) in states.iter().enumerate() {
        let h = accumulators[i];
        let rep = *hash_to_rep.entry(h).or_insert(s);
        mapping[i] = rep;
    }

    crate::debug!(3, "State equivalence analysis took {:.2?}. Reduced {} states to {}.", 
                  instant.elapsed(), states.len(), hash_to_rep.len());
    
    mapping
}

fn process_node(
    regex: &Regex,
    trie: &TokenTrie,
    node_idx: usize,
    active_groups: Vec<(Vec<u32>, u32)>, // (list of indices into `states`, current dfa state)
    accumulators: &mut Vec<u128>,
    depth: u32,
    end_state_hashes: &[u128],
) {
    let node = &trie.nodes[node_idx];

    // 1. Handle token end at this node
    if node.token_weight != 0 {
        for (list, dfa_state) in &active_groups {
            // If we are here, we are at the end of a token.
            // Use the precomputed hash of the current state (based on possible_future_group_ids).
            let h = end_state_hashes[*dfa_state as usize].wrapping_mul(node.token_weight);
            for &idx_in_input in list {
                accumulators[idx_in_input as usize] = accumulators[idx_in_input as usize].wrapping_add(h);
            }
        }
    }

    if node.transitions.is_empty() { return; }

    // 2. Transitions
    // We need to regroup by *next* DFA state for each child
    for &(byte, child_idx) in &node.transitions {
        let child_node = &trie.nodes[child_idx as usize];
        let mut next_groups_map: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();

        for (list, dfa_state) in &active_groups {
            let dfa_node = &regex.dfa.states[*dfa_state as usize];
            if let Some(&next_state) = dfa_node.transitions.get(byte) {
                let next_state = next_state as u32;
                let next_data = &regex.dfa.states[next_state as usize];
                
                // Handle match events on this transition
                if !next_data.finalizers.is_empty() {
                    let mut event_hash = 0u128;
                    for gid in &next_data.finalizers {
                        event_hash = event_hash.wrapping_add(hash_match_event(gid as u32, depth + 1));
                    }
                    // Apply this event to all subtree tokens
                    let contrib = event_hash.wrapping_mul(child_node.subtree_weight);
                    for &idx_in_input in list {
                        accumulators[idx_in_input as usize] = accumulators[idx_in_input as usize].wrapping_add(contrib);
                    }
                }
                
                next_groups_map.entry(next_state).or_default().extend_from_slice(list);
            } 
            // If no transition (dead), we do nothing. 
            // The accumulated hash for the dead state will simply lack the contributions from this subtree,
            // effectively distinguishing it from states that continue.
        }

        if !next_groups_map.is_empty() {
            let next_groups: Vec<_> = next_groups_map.into_iter().map(|(k, v)| (v, k)).collect();
            process_node(regex, trie, child_idx as usize, next_groups, accumulators, depth + 1, end_state_hashes);
        }
    }
}
