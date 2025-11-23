use crate::finite_automata::Regex;
use smallvec::SmallVec;

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

#[inline(always)]
fn hash_end_state(class_id: u64) -> u128 {
    // Distinguish end state hash from match hash by setting high bit or using different mix
    mix_u128((class_id as u128) | (1u128 << 127))
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

pub fn find_state_equivalence_classes(
    regex: &Regex,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> Vec<usize> {
    // 1. Build Token Trie
    let mut trie = TokenTrie::new();
    for (i, token) in tokens.iter().enumerate() {
        // Assign a random weight to each token index to distinguish them
        let w = mix_u128((i as u128).wrapping_mul(0x9E3779B97F4A7C15));
        trie.insert(token, w);
    }
    trie.compute_subtree_weights(0);

    // 2. Partition Refinement Loop
    // Initialize classes to 0. We rely on the refinement loop to distinguish states
    // based on their behavior with respect to tokens.
    let max_state_id = regex.dfa.states.len() - 1;
    let mut class_ids = vec![0u64; regex.dfa.states.len()];

    // Prepare initial groups for DFS (invariant across iterations)
    // Map: current_dfa_state -> List of original_state_indices (u32)
    let mut initial_groups: Vec<(Vec<u32>, u32)> = Vec::new();
    let mut map: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
    for (idx_in_input, &s) in states.iter().enumerate() {
        map.entry(s as u32).or_default().push(idx_in_input as u32);
    }
    for (dfa_st, list) in map {
        initial_groups.push((list, dfa_st));
    }

    // Loop until fixpoint
    for _iteration in 0..100 {
        let mut accumulators = vec![0u128; states.len()];
        
        process_node(regex, &trie, 0, initial_groups.clone(), &mut accumulators, 0, &class_ids);

        // Re-classify based on new hashes + previous class
        let mut new_class_map: std::collections::HashMap<u128, u64> = std::collections::HashMap::new();
        let mut next_class = 0u64;
        let mut changed = false;
        let mut new_class_ids = vec![0u64; max_state_id + 1];

        for (idx, &s) in states.iter().enumerate() {
            let h = accumulators[idx];
            // Mix with previous class to ensure refinement only splits, never merges
            let combined_hash = mix_u128(h ^ (class_ids[s] as u128));
            
            let id = *new_class_map.entry(combined_hash).or_insert_with(|| {
                let c = next_class;
                next_class += 1;
                c
            });
            
            if id != class_ids[s] {
                changed = true;
            }
            new_class_ids[s] = id;
        }

        class_ids = new_class_ids;
        if !changed {
            break;
        }
    }

    // 3. Generate final mapping to representatives
    let mut class_to_rep: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    let mut mapping = vec![0; states.len()];
    
    for (i, &s) in states.iter().enumerate() {
        let cls = class_ids[s];
        let rep = *class_to_rep.entry(cls).or_insert(s);
        mapping[i] = rep;
    }
    
    mapping
}

fn process_node(
    regex: &Regex,
    trie: &TokenTrie,
    node_idx: usize,
    active_groups: Vec<(Vec<u32>, u32)>, // (list of indices into `states`, current dfa state)
    accumulators: &mut Vec<u128>,
    depth: u32,
    class_ids: &[u64],
) {
    let node = &trie.nodes[node_idx];

    // 1. Handle token end at this node
    if node.token_weight != 0 {
        for (list, dfa_state) in &active_groups {
            // If we are here, we are at the end of a token.
            // Use the CLASS of the current state, not the state ID itself.
            let cls = class_ids[*dfa_state as usize];
            let h = hash_end_state(cls).wrapping_mul(node.token_weight);
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
                    for &gid in &next_data.finalizers {
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
            process_node(regex, trie, child_idx as usize, next_groups, accumulators, depth + 1, class_ids);
        }
    }
}
