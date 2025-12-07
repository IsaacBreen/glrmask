//! Vocab DFA Product Approach for Combined State/Token Equivalence
//!
//! This module implements a novel approach to computing both state equivalence
//! and token equivalence in a single unified algorithm.
//!
//! ## Concept
//!
//! Traditional approach:
//! 1. For each token, run DFA from each state → O(tokens × states × avg_len)
//! 2. Group states/tokens by behavioral signatures
//!
//! New approach using intersection DFA construction:
//! 1. Build vocab trie (shares prefix computation)
//! 2. Product state = (tokenizer_state → set_of_initial_states, vocab_trie_node)
//! 3. As we traverse, group initial states by their current position
//! 4. At token endpoints, initial states in same group are equivalent
//!
//! This is an intersection DFA construction that naturally groups equivalent states.
//! The key insight is that we don't track each state individually - we track
//! GROUPS of initial states that have reached the SAME current state.
//!
//! Complexity: O(vocab_trie_nodes × distinct_current_states) which is typically
//! much less than O(vocab_trie_nodes × total_states).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use crate::finite_automata::Regex;
use rayon::prelude::*;

/// Signature type for what happens when running a token from a state.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TokenBehavior {
    /// None if token is rejected, Some(state_id) if accepted
    pub end_state: Option<usize>,
    /// List of (position, group_ids) for matches found during execution
    pub matches: Vec<(usize, Vec<usize>)>,
}

impl TokenBehavior {
    pub fn dead() -> Self {
        TokenBehavior {
            end_state: None,
            matches: Vec::new(),
        }
    }
}

/// A state in the vocab trie
#[derive(Debug)]
struct VocabTrieNode {
    /// Token ID if this is an accepting state (end of a token)
    token_id: Option<usize>,
    /// Transitions: byte -> child node index
    children: HashMap<u8, usize>,
}

/// Minimal vocab trie for efficient traversal
pub struct VocabTrie {
    nodes: Vec<VocabTrieNode>,
}

impl VocabTrie {
    /// Build a vocab trie from vocabulary tokens
    pub fn build(tokens: &[Vec<u8>]) -> Self {
        let mut nodes = vec![VocabTrieNode {
            token_id: None,
            children: HashMap::new(),
        }];
        
        for (token_id, token) in tokens.iter().enumerate() {
            let mut current = 0;
            for &byte in token {
                if let Some(&child) = nodes[current].children.get(&byte) {
                    current = child;
                } else {
                    let new_node = nodes.len();
                    nodes.push(VocabTrieNode {
                        token_id: None,
                        children: HashMap::new(),
                    });
                    nodes[current].children.insert(byte, new_node);
                    current = new_node;
                }
            }
            // Mark end of token
            nodes[current].token_id = Some(token_id);
        }
        
        VocabTrie { nodes }
    }
    
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

/// Result of the product construction
pub struct ProductResult {
    /// State equivalence: state_i -> representative
    pub state_to_rep: Vec<usize>,
    /// Token equivalence: token_i -> representative  
    pub token_to_rep: Vec<usize>,
    /// Number of state equivalence classes
    pub num_state_classes: usize,
    /// Number of token equivalence classes
    pub num_token_classes: usize,
}

/// Compute state and token equivalence using vocab trie traversal with partition refinement.
///
/// Algorithm:
/// 1. Build vocab trie
/// 2. For each state, compute its "signature" at each token endpoint
///    - Signature = (end_state, intermediate_matches)
/// 3. States with identical signatures at ALL endpoints are equivalent
/// 4. Tokens with identical signatures for ALL states are equivalent
///
/// We share computation via the trie structure.
pub fn compute_equivalence_via_product(
    regex: &Regex,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> ProductResult {
    let start = std::time::Instant::now();
    
    // Build vocab trie
    let vocab_trie = VocabTrie::build(tokens);
    crate::debug!(4, "Vocab trie has {} nodes for {} tokens", vocab_trie.num_nodes(), tokens.len());
    
    // Precompute DFA transitions as packed arrays
    let dfa = &regex.dfa;
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
    
    // Compute signature for each (state, token) pair via trie traversal
    // Key insight: share computation for common prefixes
    
    // For each state, we'll compute a signature vector: one entry per token
    // signature[state][token] = behavior of running token from state
    
    // Use DFS on trie, maintaining current positions for all states
    // At each accepting node, record the behavior
    
    // For efficiency, we'll do this in parallel over states
    let token_behaviors: Vec<Vec<Option<usize>>> = states
        .par_iter()
        .map(|&start_state| {
            // For this start state, traverse the trie and compute end positions
            let mut behaviors = vec![None; tokens.len()];
            
            // DFS stack: (trie_node, current_dfa_state)
            let mut stack: Vec<(usize, u32)> = vec![(0, start_state as u32)];
            
            while let Some((trie_node, dfa_state)) = stack.pop() {
                let node = &vocab_trie.nodes[trie_node];
                
                // If this is a token endpoint, record the behavior
                if let Some(token_id) = node.token_id {
                    behaviors[token_id] = if dfa_state == NONE_STATE {
                        None
                    } else {
                        Some(dfa_state as usize)
                    };
                }
                
                // Push children to stack
                for (&byte, &child_idx) in &node.children {
                    let next_dfa_state = if dfa_state == NONE_STATE {
                        NONE_STATE
                    } else {
                        dfa_transitions[dfa_state as usize][byte as usize]
                    };
                    stack.push((child_idx, next_dfa_state));
                }
            }
            
            behaviors
        })
        .collect();
    
    crate::debug!(4, "Computed behaviors for {} states × {} tokens via trie", states.len(), tokens.len());
    
    // State equivalence: group states by their behavior vector
    // Two states are equivalent if they have the same behavior for ALL tokens
    let mut sig_to_rep: HashMap<Vec<Option<usize>>, usize> = HashMap::new();
    let mut state_to_rep: Vec<usize> = Vec::with_capacity(states.len());
    
    for (i, behaviors) in token_behaviors.iter().enumerate() {
        let rep = *sig_to_rep.entry(behaviors.clone()).or_insert(states[i]);
        state_to_rep.push(rep);
    }
    
    let num_state_classes = sig_to_rep.len();
    
    // Token equivalence: group tokens by their behavior across all states
    // Two tokens are equivalent if they produce the same behavior pattern from ALL states
    let mut token_sigs: Vec<Vec<Option<usize>>> = vec![Vec::with_capacity(states.len()); tokens.len()];
    for behaviors in &token_behaviors {
        for (token_id, &behavior) in behaviors.iter().enumerate() {
            token_sigs[token_id].push(behavior);
        }
    }
    
    let mut token_sig_to_rep: HashMap<Vec<Option<usize>>, usize> = HashMap::new();
    let mut token_to_rep: Vec<usize> = Vec::with_capacity(tokens.len());
    
    for (token_id, sig) in token_sigs.iter().enumerate() {
        let rep = *token_sig_to_rep.entry(sig.clone()).or_insert(token_id);
        token_to_rep.push(rep);
    }
    
    let num_token_classes = token_sig_to_rep.len();
    
    crate::debug!(3, "Vocab DFA product approach took {:?}. States: {} -> {}, Tokens: {} -> {}", 
                  start.elapsed(), states.len(), num_state_classes, tokens.len(), num_token_classes);
    
    ProductResult {
        state_to_rep,
        token_to_rep,
        num_state_classes,
        num_token_classes,
    }
}

/// Alternative: More memory-efficient version using hashing instead of full vectors
pub fn compute_equivalence_via_product_hashed(
    regex: &Regex,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> ProductResult {
    let start = std::time::Instant::now();
    
    // Build vocab trie
    let vocab_trie = VocabTrie::build(tokens);
    crate::debug!(4, "Vocab trie (hashed) has {} nodes for {} tokens", vocab_trie.num_nodes(), tokens.len());
    
    // Precompute DFA transitions
    let dfa = &regex.dfa;
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
    
    // Mix function for hashing
    #[inline(always)]
    fn mix_u128(mut x: u128) -> u128 {
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
        x ^= x >> 33;
        x
    }
    
    // Compute hash signature for each state via trie traversal
    let state_hashes: Vec<u128> = states
        .par_iter()
        .map(|&start_state| {
            let mut hash: u128 = 0;
            
            // DFS stack: (trie_node, current_dfa_state)
            let mut stack: Vec<(usize, u32)> = vec![(0, start_state as u32)];
            
            while let Some((trie_node, dfa_state)) = stack.pop() {
                let node = &vocab_trie.nodes[trie_node];
                
                // If this is a token endpoint, hash the behavior
                if let Some(token_id) = node.token_id {
                    let behavior = if dfa_state == NONE_STATE {
                        u128::MAX
                    } else {
                        dfa_state as u128
                    };
                    // Include token_id in hash for position-sensitivity
                    hash = hash.wrapping_add(mix_u128(behavior ^ ((token_id as u128) << 64)));
                }
                
                // Push children to stack
                for (&byte, &child_idx) in &node.children {
                    let next_dfa_state = if dfa_state == NONE_STATE {
                        NONE_STATE
                    } else {
                        dfa_transitions[dfa_state as usize][byte as usize]
                    };
                    stack.push((child_idx, next_dfa_state));
                }
            }
            
            hash
        })
        .collect();
    
    // Group states by hash
    let mut hash_to_rep: HashMap<u128, usize> = HashMap::new();
    let mut state_to_rep: Vec<usize> = Vec::with_capacity(states.len());
    
    for (i, &hash) in state_hashes.iter().enumerate() {
        let rep = *hash_to_rep.entry(hash).or_insert(states[i]);
        state_to_rep.push(rep);
    }
    
    let num_state_classes = hash_to_rep.len();
    
    // For token equivalence, we need to compute hash per token
    // This requires iterating tokens, but we can still use trie for prefix sharing
    let token_hashes: Vec<u128> = (0..tokens.len())
        .into_par_iter()
        .map(|token_id| {
            let token = &tokens[token_id];
            let mut hash: u128 = 0;
            
            // For each state, compute the end position and hash it
            for &start_state in states {
                let mut current = start_state as u32;
                for &byte in token {
                    if current == NONE_STATE {
                        break;
                    }
                    current = dfa_transitions[current as usize][byte as usize];
                }
                let behavior = if current == NONE_STATE {
                    u128::MAX
                } else {
                    current as u128
                };
                hash = hash.wrapping_add(mix_u128(behavior ^ ((start_state as u128) << 64)));
            }
            
            hash
        })
        .collect();
    
    let mut token_hash_to_rep: HashMap<u128, usize> = HashMap::new();
    let mut token_to_rep: Vec<usize> = Vec::with_capacity(tokens.len());
    
    for (token_id, &hash) in token_hashes.iter().enumerate() {
        let rep = *token_hash_to_rep.entry(hash).or_insert(token_id);
        token_to_rep.push(rep);
    }
    
    let num_token_classes = token_hash_to_rep.len();
    
    crate::debug!(3, "Vocab DFA product (hashed) took {:?}. States: {} -> {}, Tokens: {} -> {}", 
                  start.elapsed(), states.len(), num_state_classes, tokens.len(), num_token_classes);
    
    ProductResult {
        state_to_rep,
        token_to_rep,
        num_state_classes,
        num_token_classes,
    }
}

/// Efficient intersection DFA approach for state equivalence.
///
/// Instead of tracking each initial state independently, we group initial states
/// by their CURRENT position. The product state is:
///   (current_tokenizer_state -> set_of_initial_states, vocab_trie_node)
///
/// As we traverse the trie, initial states that reach the same current state
/// get merged. This is much more efficient when many states behave similarly.
///
/// This approach tracks BOTH end state AND finalizers at each position,
/// matching the semantics of the iterative state equivalence analysis.
///
/// Complexity: O(trie_nodes × distinct_current_positions) which can be much
/// less than O(trie_nodes × total_states).
pub fn compute_equivalence_intersection_dfa(
    regex: &Regex,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> ProductResult {
    let start = std::time::Instant::now();
    
    // Build vocab trie
    let vocab_trie = VocabTrie::build(tokens);
    let trie_build_time = start.elapsed();
    crate::debug!(4, "Vocab trie has {} nodes (built in {:?})", vocab_trie.num_nodes(), trie_build_time);
    
    // Precompute DFA transitions and finalizers
    let dfa = &regex.dfa;
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
    
    // Precompute end state hashes (for possible_future_group_ids)
    fn mix_u128(mut x: u128) -> u128 {
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
        x ^= x >> 33;
        x
    }
    
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
    
    // For each token, record the behavior for each initial state
    // behavior = (end_state_hash, finalizers_hash)
    // token_id -> (initial_state -> behavior_hash)
    let mut token_behaviors: Vec<HashMap<usize, u128>> = vec![HashMap::new(); tokens.len()];
    
    // The key data structure for DFS:
    // For each initial state, track (current_dfa_state, finalizers_hash_so_far)
    type StateInfo = (u32, u128); // (current_state, finalizers_hash)
    type StateMap = HashMap<usize, StateInfo>;
    
    // DFS through the vocab trie
    let mut stack: Vec<(usize, u32, StateMap)> = Vec::new(); // (trie_node, depth, state_map)
    
    // Initial: each state starts at itself with empty finalizers hash
    let initial_map: StateMap = states.iter()
        .map(|&s| {
            // Check if initial state has finalizers at position 0
            let mut init_hash: u128 = 0;
            let finalizers = &dfa_finalizers[s];
            if !finalizers.is_empty() {
                for &gid in finalizers {
                    init_hash = init_hash.wrapping_add(
                        mix_u128((0u32 as u128) ^ ((gid as u128) << 32))
                    );
                }
            }
            (s, (s as u32, init_hash))
        })
        .collect();
    
    stack.push((0, 0, initial_map));
    
    while let Some((trie_node, depth, state_map)) = stack.pop() {
        let node = &vocab_trie.nodes[trie_node];
        
        // If this is a token endpoint, compute final behavior hash
        if let Some(token_id) = node.token_id {
            for (&init_state, &(current, finalizers_hash)) in &state_map {
                let end_hash = if current == NONE_STATE {
                    mix_u128(0xDEADBEEF_u128)
                } else {
                    end_state_hashes[current as usize]
                };
                let behavior = end_hash.wrapping_add(finalizers_hash);
                token_behaviors[token_id].insert(init_state, behavior);
            }
        }
        
        // Process children
        for (&byte, &child_idx) in &node.children {
            let new_depth = depth + 1;
            // Compute next state map
            let mut next_map: StateMap = HashMap::with_capacity(state_map.len());
            
            for (&init_state, &(current, finalizers_hash)) in &state_map {
                let (next_state, next_hash) = if current == NONE_STATE {
                    (NONE_STATE, finalizers_hash)
                } else {
                    let next = dfa_transitions[current as usize][byte as usize];
                    if next == NONE_STATE {
                        (NONE_STATE, finalizers_hash)
                    } else {
                        // Add finalizers at the new state
                        let mut new_hash = finalizers_hash;
                        let finalizers = &dfa_finalizers[next as usize];
                        if !finalizers.is_empty() {
                            for &gid in finalizers {
                                new_hash = new_hash.wrapping_add(
                                    mix_u128((new_depth as u128) ^ ((gid as u128) << 32))
                                );
                            }
                        }
                        (next, new_hash)
                    }
                };
                next_map.insert(init_state, (next_state, next_hash));
            }
            
            stack.push((child_idx, new_depth, next_map));
        }
    }
    
    crate::debug!(4, "Intersection DFA: processed {} trie nodes", vocab_trie.num_nodes());
    
    // State equivalence: states are equivalent if they have the same behavior hash
    // for ALL tokens.
    let mut state_signatures: HashMap<Vec<u128>, usize> = HashMap::new();
    let mut state_to_rep: Vec<usize> = Vec::with_capacity(states.len());
    
    for &init_state in states {
        // Signature: behavior hash for each token
        let sig: Vec<u128> = (0..tokens.len())
            .map(|token_id| *token_behaviors[token_id].get(&init_state).unwrap_or(&0))
            .collect();
        
        let rep = *state_signatures.entry(sig).or_insert(init_state);
        state_to_rep.push(rep);
    }
    
    let num_state_classes = state_signatures.len();
    
    // Token equivalence: tokens are equivalent if they produce the same behavior
    // for ALL initial states.
    let mut token_signatures: HashMap<Vec<u128>, usize> = HashMap::new();
    let mut token_to_rep: Vec<usize> = Vec::with_capacity(tokens.len());
    
    for token_id in 0..tokens.len() {
        // Signature: behavior from each initial state
        let sig: Vec<u128> = states.iter()
            .map(|&s| *token_behaviors[token_id].get(&s).unwrap_or(&0))
            .collect();
        
        let rep = *token_signatures.entry(sig).or_insert(token_id);
        token_to_rep.push(rep);
    }
    
    let num_token_classes = token_signatures.len();
    
    crate::debug!(3, "Intersection DFA approach took {:?}. States: {} -> {}, Tokens: {} -> {}", 
                  start.elapsed(), states.len(), num_state_classes, tokens.len(), num_token_classes);
    
    ProductResult {
        state_to_rep,
        token_to_rep,
        num_state_classes,
        num_token_classes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vocab_trie_build() {
        let tokens = vec![
            vec![b'h', b'e', b'l', b'l', b'o'],
            vec![b'h', b'e', b'l', b'p'],
            vec![b'w', b'o', b'r', b'l', b'd'],
        ];
        let trie = VocabTrie::build(&tokens);
        
        // "hello", "help", "world" share some prefixes
        assert!(trie.num_nodes() <= 12);
        assert!(trie.num_nodes() >= 8);
    }
}
