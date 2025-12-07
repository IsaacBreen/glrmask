//! Trie-Based Vocab Equivalence Analysis
//!
//! This module implements a trie-based algorithm for computing vocabulary
//! token equivalence classes. Instead of processing each token independently,
//! it builds a trie from all tokens and performs a single DFS traversal.
//!
//! The key insight is that tokens sharing prefixes share DFA execution up to that prefix.
//! By using a trie, we only process each unique prefix once across all tokens.
//!
//! Complexity: O(total_token_bytes × states) instead of O(tokens × states × avg_len)
//!
//! For a vocabulary with many shared prefixes (like GPT-2 where 66% of tokens start
//! with the same byte), this can be significantly faster.
//!
//! This version uses an iterative DFS with explicit stack to avoid allocation overhead.

use crate::finite_automata::Regex;
use crate::r#macro::is_debug_level_enabled;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;

// =============================================================================
// TRIE DATA STRUCTURE
// =============================================================================

#[derive(Default)]
struct TokenTrieNode {
    /// Map from byte to child node index
    transitions: SmallVec<[(u8, u32); 8]>,
    /// Token indices that end at this node (usually 0 or 1)
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
    
    /// Get children of root node, for parallelization
    fn root_children(&self) -> Vec<(u8, u32)> {
        self.nodes[0].transitions.iter().cloned().collect()
    }
}

// =============================================================================
// HASHING UTILITIES
// =============================================================================

const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;

#[inline(always)]
fn mix_u64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

// =============================================================================
// PRECOMPUTED DFA
// =============================================================================

const NONE_STATE: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Finalizer {
    gid: usize,
    non_greedy: bool,
}

/// Precomputed DFA with optimized data layout
struct PrecomputedDfa {
    transitions: Vec<[u32; 256]>,
    finalizers: Vec<SmallVec<[Finalizer; 4]>>,
    has_transitions: Vec<bool>,
    num_groups: usize,
    /// Hash of possible_future_group_ids for each state
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
}

fn precompute_dfa(dfa: &crate::finite_automata::DFA, num_groups: usize) -> PrecomputedDfa {
    let num_states = dfa.states.len();
    
    // Transition table with NONE_STATE for missing transitions
    let mut transitions = vec![[NONE_STATE; 256]; num_states];
    for (state_idx, state) in dfa.states.iter().enumerate() {
        for (byte, &next_state) in &state.transitions {
            transitions[state_idx][byte as usize] = next_state as u32;
        }
    }
    
    // Finalizers - iterate over bitset to get group IDs
    let mut finalizers: Vec<SmallVec<[Finalizer; 4]>> = Vec::with_capacity(num_states);
    for state in &dfa.states {
        let mut fs: SmallVec<[Finalizer; 4]> = SmallVec::new();
        for gid in state.finalizers.iter() {
            if gid < num_groups {
                // Check if non-greedy
                let ng = dfa.non_greedy_finalizers.contains(&gid);
                fs.push(Finalizer { gid, non_greedy: ng });
            }
        }
        finalizers.push(fs);
    }
    
    // has_transitions
    let has_transitions: Vec<bool> = dfa.states
        .iter()
        .map(|s| !s.transitions.is_empty())
        .collect();
    
    // Completion hashes
    let none_completion_hash = mix_u64(HASH_SEED2);
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
        finalizers,
        has_transitions,
        num_groups,
        completion_hash,
        none_completion_hash,
    }
}

// =============================================================================
// MAIN ALGORITHM - PARALLEL TRIE DFS
// =============================================================================

/// Find vocab equivalence classes using trie-based traversal.
///
/// This version parallelizes over first-byte subtrees for good performance.
pub fn find_vocab_equivalence_classes_trie(
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
    
    // Compute number of groups from finalizers (which are Bitsets)
    let num_groups = regex.dfa.states.iter()
        .flat_map(|s| s.finalizers.iter())
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    
    crate::debug!(4, "  Trie-based vocab equiv: {} tokens, {} states, {} groups",
                  num_tokens, num_states, num_groups);
    
    // Build token trie
    let trie_start = std::time::Instant::now();
    let mut trie = TokenTrie::new();
    for (idx, token) in strings.iter().enumerate() {
        trie.insert(token, idx);
    }
    let trie_time = trie_start.elapsed();
    
    // Precompute DFA
    let pre_start = std::time::Instant::now();
    let pre = std::sync::Arc::new(precompute_dfa(&regex.dfa, num_groups));
    let pre_time = pre_start.elapsed();
    
    crate::debug!(5, "    Trie: {} nodes in {:?}", trie.nodes.len(), trie_time);
    crate::debug!(5, "    DFA precompute in {:?}", pre_time);
    
    // Convert initial states to u32
    let init_states: Vec<u32> = initial_states.iter().map(|&s| s as u32).collect();
    
    // Get root children for parallelization
    let root_children = trie.root_children();
    let trie = std::sync::Arc::new(trie);
    
    // Process root's token_indices first (empty tokens)
    let dfs_start = std::time::Instant::now();
    let mut all_results: Vec<(usize, u64)> = Vec::new();
    
    // Empty tokens at root
    let root_sig = compute_initial_signature(&pre, &init_states, num_groups);
    for &token_idx in &trie.nodes[0].token_indices {
        all_results.push((token_idx, root_sig));
    }
    
    // Parallel traversal of subtrees
    let subtree_results: Vec<Vec<(usize, u64)>> = root_children
        .par_iter()
        .map(|&(first_byte, child_idx)| {
            // Initialize state after first byte
            let (child_states, child_done, child_matches) = 
                process_first_byte(&pre, &init_states, first_byte, num_groups);
            
            // DFS this subtree
            let mut results = Vec::new();
            dfs_iterative(
                &trie,
                &pre,
                child_idx as usize,
                &child_states,
                &child_done,
                &child_matches,
                num_groups,
                1, // depth starts at 1
                &mut results,
            );
            results
        })
        .collect();
    
    // Merge results
    for subtree in subtree_results {
        all_results.extend(subtree);
    }
    
    let dfs_time = dfs_start.elapsed();
    
    // Group by signature
    let mut signature_groups: HashMap<u64, Vec<usize>> = HashMap::new();
    for (token_idx, sig) in all_results {
        signature_groups.entry(sig).or_default().push(token_idx);
    }
    
    crate::debug!(5, "    DFS traverse in {:?}", dfs_time);
    crate::debug!(4, "  Trie-based: {} equiv classes in {:?}", 
                  signature_groups.len(), instant.elapsed());
    
    signature_groups.into_values().collect()
}

/// Compute signature for initial states (depth 0)
fn compute_initial_signature(pre: &PrecomputedDfa, init_states: &[u32], num_groups: usize) -> u64 {
    let mut h = HASH_SEED1;
    for &state in init_states {
        let s = state as usize;
        // End state hash
        let end_hash = if pre.has_transitions[s] {
            pre.completion_hash[s]
        } else {
            pre.none_completion_hash
        };
        h = mix_u64(h ^ end_hash);
        // No match positions at depth 0
        for gid in 0..num_groups {
            let mp_hash = mix_u64((gid as u64) << 32 | (NONE_POS as u64));
            h = mix_u64(h ^ mp_hash);
        }
    }
    h
}

const NONE_POS: u32 = u32::MAX;

/// Process first byte transition from initial states
fn process_first_byte(
    pre: &PrecomputedDfa,
    init_states: &[u32],
    byte: u8,
    num_groups: usize,
) -> (Vec<u32>, Vec<bool>, Vec<u32>) {
    let n = init_states.len();
    let mut current = Vec::with_capacity(n);
    let mut done = Vec::with_capacity(n);
    let mut matches = vec![NONE_POS; n * num_groups.max(1)];
    
    for (i, &state) in init_states.iter().enumerate() {
        let s = state as usize;
        let next = pre.transitions[s][byte as usize];
        
        if next == NONE_STATE {
            current.push(NONE_STATE);
            done.push(true);
        } else {
            let ns = next as usize;
            current.push(next);
            
            // Process finalizers
            for f in &pre.finalizers[ns] {
                let idx = i * num_groups.max(1) + f.gid;
                if f.non_greedy {
                    if matches[idx] == NONE_POS {
                        matches[idx] = 1;
                    }
                } else {
                    matches[idx] = 1;
                }
            }
            
            done.push(!pre.has_transitions[ns]);
        }
    }
    
    (current, done, matches)
}

/// Iterative DFS traversal of trie subtree
fn dfs_iterative(
    trie: &TokenTrie,
    pre: &PrecomputedDfa,
    start_node: usize,
    start_states: &[u32],
    start_done: &[bool],
    start_matches: &[u32],
    num_groups: usize,
    start_depth: u32,
    results: &mut Vec<(usize, u64)>,
) {
    let n = start_states.len();
    let match_size = n * num_groups.max(1);
    
    // Stack entry: (node_idx, child_iter_pos, states, done, matches, depth)
    // We store Vec indices into reusable buffers
    struct StackFrame {
        node_idx: usize,
        child_pos: usize,
        states_offset: usize,
        done_offset: usize,
        matches_offset: usize,
        depth: u32,
    }
    
    // Reusable buffers for state storage
    let mut states_pool: Vec<u32> = Vec::with_capacity(n * 32);
    let mut done_pool: Vec<bool> = Vec::with_capacity(n * 32);
    let mut matches_pool: Vec<u32> = Vec::with_capacity(match_size * 32);
    
    // Push initial state
    let init_states_offset = states_pool.len();
    states_pool.extend_from_slice(start_states);
    let init_done_offset = done_pool.len();
    done_pool.extend_from_slice(start_done);
    let init_matches_offset = matches_pool.len();
    matches_pool.extend_from_slice(start_matches);
    
    let mut stack: Vec<StackFrame> = vec![StackFrame {
        node_idx: start_node,
        child_pos: 0,
        states_offset: init_states_offset,
        done_offset: init_done_offset,
        matches_offset: init_matches_offset,
        depth: start_depth,
    }];
    
    while let Some(frame) = stack.last_mut() {
        let node = &trie.nodes[frame.node_idx];
        
        // First visit: process tokens at this node
        if frame.child_pos == 0 {
            if !node.token_indices.is_empty() {
                let sig = compute_signature_from_pools(
                    pre,
                    &states_pool[frame.states_offset..frame.states_offset + n],
                    &done_pool[frame.done_offset..frame.done_offset + n],
                    &matches_pool[frame.matches_offset..frame.matches_offset + match_size],
                    num_groups,
                );
                for &token_idx in &node.token_indices {
                    results.push((token_idx, sig));
                }
            }
        }
        
        // Try to advance to next child
        if frame.child_pos < node.transitions.len() {
            let (byte, child_idx) = node.transitions[frame.child_pos];
            frame.child_pos += 1;
            
            // Compute child state
            let new_depth = frame.depth + 1;
            let child_states_offset = states_pool.len();
            let child_done_offset = done_pool.len();
            let child_matches_offset = matches_pool.len();
            
            // Extend pools
            for i in 0..n {
                let state = states_pool[frame.states_offset + i];
                let is_done = done_pool[frame.done_offset + i];
                
                if is_done || state == NONE_STATE {
                    states_pool.push(NONE_STATE);
                    done_pool.push(true);
                } else {
                    let s = state as usize;
                    let next = pre.transitions[s][byte as usize];
                    
                    if next == NONE_STATE {
                        states_pool.push(NONE_STATE);
                        done_pool.push(true);
                    } else {
                        states_pool.push(next);
                        done_pool.push(!pre.has_transitions[next as usize]);
                    }
                }
            }
            
            // Copy matches (avoid borrow conflict)
            let old_len = matches_pool.len();
            matches_pool.reserve(match_size);
            for i in 0..match_size {
                let v = matches_pool[frame.matches_offset + i];
                matches_pool.push(v);
            }
            
            // Process finalizers for transitions that succeeded
            for i in 0..n {
                let next = states_pool[child_states_offset + i];
                if next != NONE_STATE {
                    let ns = next as usize;
                    for f in &pre.finalizers[ns] {
                        let idx = child_matches_offset + i * num_groups.max(1) + f.gid;
                        if f.non_greedy {
                            if matches_pool[idx] == NONE_POS {
                                matches_pool[idx] = new_depth;
                            }
                        } else {
                            matches_pool[idx] = new_depth;
                        }
                    }
                }
            }
            
            // Push child frame
            stack.push(StackFrame {
                node_idx: child_idx as usize,
                child_pos: 0,
                states_offset: child_states_offset,
                done_offset: child_done_offset,
                matches_offset: child_matches_offset,
                depth: new_depth,
            });
        } else {
            // Pop this frame, reclaim memory
            let popped = stack.pop().unwrap();
            
            // Shrink pools back (only if this wasn't the initial frame)
            if popped.states_offset > init_states_offset {
                states_pool.truncate(popped.states_offset);
                done_pool.truncate(popped.done_offset);
                matches_pool.truncate(popped.matches_offset);
            }
        }
    }
}

fn compute_signature_from_pools(
    pre: &PrecomputedDfa,
    states: &[u32],
    done: &[bool],
    matches: &[u32],
    num_groups: usize,
) -> u64 {
    let n = states.len();
    let mut h = HASH_SEED1;
    
    for i in 0..n {
        let state = states[i];
        let is_done = done[i];
        
        // Hash the final state
        let end_hash = if state == NONE_STATE || is_done || !pre.has_transitions[state as usize] {
            pre.none_completion_hash
        } else {
            pre.completion_hash[state as usize]
        };
        h = mix_u64(h ^ end_hash);
        
        // Hash match positions for this state
        if num_groups > 0 {
            let base = i * num_groups;
            for gid in 0..num_groups {
                let pos = matches[base + gid];
                let mp_hash = mix_u64((gid as u64) << 32 | (pos as u64));
                h = mix_u64(h ^ mp_hash);
            }
        }
    }
    
    h
}
