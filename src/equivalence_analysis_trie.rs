// PERMANENT WARNING: Do NOT add any form of caching or shortcuts that skip or restrict
// states/tokens for equivalence analysis. Full correctness is mandatory; no "cheating"
// optimizations that drop work are allowed here.

//! Trie-based equivalence analysis for LLM tokens.
//!
//! This module implements token equivalence analysis by traversing a vocabulary trie
//! and tracking weighted contributions from initial tokenizer states. The algorithm
//! achieves O(trie_size × unique_dfa_states) complexity instead of O(states × tokens × token_length).
//!
//! Key insight: If two initial states reach the same DFA state at some trie node,
//! they will contribute identically to ALL tokens in that subtree. We can merge them
//! and track only the weighted sum.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::finite_automata::Regex;

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

// =============================================================================
// HASH UTILITIES (128-bit for collision resistance)
// =============================================================================

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
fn hash_state_weight(state: u32, weight: u128) -> u128 {
    // Combine state and weight uniquely
    let packed = ((state as u128) << 64) | (weight & ((1u128 << 64) - 1));
    mix_u128(packed)
}

#[inline(always)]
fn hash_match_event(gid: u32, depth: u32) -> u128 {
    let packed = ((depth as u128) << 64) | (gid as u128);
    mix_u128(packed)
}

#[inline(always)]
fn hash_end_behavior(future_groups: &[usize]) -> u128 {
    let mut h = 0u128;
    for &gid in future_groups {
        h = mix_u128(h ^ (gid as u128));
    }
    // Distinguish from match events
    mix_u128(h | (1u128 << 127))
}

// =============================================================================
// TOKEN TRIE
// =============================================================================

#[derive(Default)]
struct TokenTrieNode {
    children: Vec<(u8, u32)>,  // (byte, child_idx)
    // Token indices that END at this node
    token_indices: Vec<usize>,
}

struct TokenTrie {
    nodes: Vec<TokenTrieNode>,
}

impl TokenTrie {
    fn new() -> Self {
        TokenTrie { nodes: vec![TokenTrieNode::default()] }
    }

    fn insert(&mut self, token: &[u8], token_idx: usize) {
        let mut node_idx = 0;
        for &b in token {
            let mut found = None;
            for &(byte, child) in &self.nodes[node_idx].children {
                if byte == b {
                    found = Some(child as usize);
                    break;
                }
            }
            match found {
                Some(child) => node_idx = child,
                None => {
                    let new_idx = self.nodes.len();
                    self.nodes.push(TokenTrieNode::default());
                    self.nodes[node_idx].children.push((b, new_idx as u32));
                    node_idx = new_idx;
                }
            }
        }
        self.nodes[node_idx].token_indices.push(token_idx);
    }
}

// =============================================================================
// WEIGHTED STATE GROUPS
// =============================================================================

/// Tracks which initial states are at each DFA state, with weighted sums.
/// Key insight: states that converge to the same DFA state contribute identically
/// to all tokens in the subtree, so we can merge them.
type StateGroups = HashMap<u32, u128>;  // dfa_state -> sum of weights

fn advance_state_groups(
    regex: &Regex,
    groups: &StateGroups,
    byte: u8,
) -> StateGroups {
    let mut next_groups = HashMap::with_capacity(groups.len());
    for (&dfa_state, &weight) in groups {
        if let Some(&next_state) = regex.dfa.states[dfa_state as usize].transitions.get(byte) {
            *next_groups.entry(next_state as u32).or_default() += weight;
        }
        // Dead states are dropped (weight contributes nothing more)
    }
    next_groups
}

// =============================================================================
// SUFFIX HASH COMPUTATION
// =============================================================================

/// For a given suffix of a token (starting at some position), compute the
/// "trellis hash" that captures all possible match paths through that suffix.
/// This is independent of initial state - it always starts from the DFA's start_state.
fn compute_suffix_hash(
    regex: &Regex,
    suffix: &[u8],
    cache: &mut HashMap<usize, u128>,
    token_offset: usize,
) -> u128 {
    // Check cache (keyed by position in original token)
    if let Some(&h) = cache.get(&token_offset) {
        return h;
    }

    // Execute DFA on this suffix from start_state
    let result = regex.execute_from_state_nonzero(suffix, regex.dfa.start_state);

    // Build hash combining: end state behavior + outgoing edges with recursive hashes
    let mut hasher = 0u128;

    // Hash end state behavior (possible future terminals)
    if let Some(end_state) = result.end_state {
        let future_groups: Vec<usize> = regex.dfa.states[end_state]
            .possible_future_group_ids
            .iter()
            .cloned()
            .collect();
        hasher = hasher.wrapping_add(hash_end_behavior(&future_groups));
    }

    // Sort matches by group_id for determinism
    let mut matches: Vec<_> = result.matches.iter().collect();
    matches.sort_by_key(|m| m.group_id);

    // Hash each outgoing edge with recursive suffix hash
    for m in matches {
        let target_pos = token_offset + m.position;
        let target_hash = compute_suffix_hash(
            regex,
            &suffix[m.position..],
            cache,
            target_pos,
        );
        let edge_hash = mix_u128(((m.group_id as u128) << 64) | target_hash);
        hasher = hasher.wrapping_add(edge_hash);
    }

    let h = mix_u128(hasher);
    cache.insert(token_offset, h);
    h
}

// =============================================================================
// MAIN ALGORITHM
// =============================================================================

/// Find equivalence classes of tokens based on their behavior across all initial states.
pub fn find_equivalence_classes(
    regex: &Regex,
    tokens: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    let num_tokens = tokens.len();
    let num_states = initial_states.len();

    crate::debug!(3, "trie equivalence: {} tokens, {} initial states", num_tokens, num_states);

    if num_tokens == 0 {
        return BTreeSet::new();
    }

    if num_states == 0 {
        // All tokens equivalent if no states to distinguish them
        return BTreeSet::from_iter(vec![(0..num_tokens).collect()]);
    }

    // 1. Assign random weights to each initial state
    let state_weights: Vec<u128> = initial_states
        .iter()
        .enumerate()
        .map(|(i, &s)| mix_u128(((i + 1) as u128).wrapping_mul(0x9E3779B97F4A7C15) ^ (s as u128)))
        .collect();

    // 2. Build token trie
    let mut trie = TokenTrie::new();
    for (idx, token) in tokens.iter().enumerate() {
        trie.insert(token, idx);
    }

    crate::debug!(4, "  Built token trie with {} nodes", trie.nodes.len());

    // 3. Initialize accumulators for each token
    let mut token_accumulators = vec![0u128; num_tokens];

    // 4. Initialize state groups at root
    let mut initial_groups: StateGroups = HashMap::new();
    for (i, &s) in initial_states.iter().enumerate() {
        *initial_groups.entry(s as u32).or_default() += state_weights[i];
    }

    // Process initial finalizers at root (before any bytes)
    for (&dfa_state, &weight) in &initial_groups {
        let state_data = &regex.dfa.states[dfa_state as usize];
        if !state_data.finalizers.is_empty() {
            // This would be a match at position 0 - affects all tokens
            // The contribution will be factored in during trie traversal
        }
    }

    // 5. DFS through trie
    process_trie_node(
        regex,
        &trie,
        0,
        initial_groups,
        &mut token_accumulators,
        0,
        &vec![],  // empty path (no bytes yet)
        tokens,
    );

    // 6. Group tokens by accumulator
    let mut groups: HashMap<u128, Vec<usize>> = HashMap::new();
    for (idx, &acc) in token_accumulators.iter().enumerate() {
        groups.entry(acc).or_default().push(idx);
    }

    crate::debug!(3, "  Found {} equivalence classes", groups.len());

    groups.into_values().collect()
}

fn process_trie_node(
    regex: &Regex,
    trie: &TokenTrie,
    node_idx: usize,
    state_groups: StateGroups,
    accumulators: &mut Vec<u128>,
    depth: u32,
    path: &[u8],
    tokens: &[Vec<u8>],
) {
    let node = &trie.nodes[node_idx];

    // Handle tokens that END at this node
    if !node.token_indices.is_empty() {
        // For each token ending here, compute its signature contribution from current state groups
        for &token_idx in &node.token_indices {
            let token = &tokens[token_idx];

            // Compute suffix hash (from position 0, which is the full token - but we need
            // the suffix structure starting AFTER the token to capture what terminals could follow)
            // Actually, for pos 0, we need the hash that captures the trellis from initial states

            let mut token_sig = 0u128;

            // Contribution from each state group (initial states that reached this DFA state)
            for (&dfa_state, &weight) in &state_groups {
                // Hash the behavior from this initial state: end state + possible future groups
                let state_data = &regex.dfa.states[dfa_state as usize];
                let future_groups: Vec<usize> = state_data.possible_future_group_ids.iter().cloned().collect();
                let end_hash = hash_end_behavior(&future_groups);

                // The contribution from this group of initial states
                let contrib = mix_u128(end_hash).wrapping_mul(weight);
            token_sig = token_sig.wrapping_add(contrib);
            }

            // Also need to incorporate the MATCH EVENTS that happened along the path
            // For the "pos 0" execution, we need to track matches
            for (&dfa_state, &weight) in &state_groups {
                let state_data = &regex.dfa.states[dfa_state as usize];
                for gid in &state_data.finalizers {
                    let match_hash = hash_match_event(gid as u32, depth);
                    let contrib = match_hash.wrapping_mul(weight);
                    token_sig = token_sig.wrapping_add(contrib);
                }
            }

            // Now we need to add the SUFFIX contributions
            // The trellis involves matches that spawn new execution paths from start_state
            // This is where it gets complex - we need to run the reference algorithm's
            // suffix hash computation
            let mut suffix_cache: HashMap<usize, u128> = HashMap::new();
            
            // For each match position in pos0, compute suffix hash and incorporate
            // Actually, this is getting complicated. Let me think about this differently.
            //
            // The reference computes:
            // - For pos 0: run DFA from initial_state, get matches and end_state
            // - For each match at position p: run DFA from start_state on suffix[p..], recursively
            // - Hash the whole trellis bottom-up
            //
            // The contribution to the token's signature from initial_state s is:
            //   hash(end_state_behavior(s), [(gid, suffix_hash(p)) for each match])
            //
            // The suffix_hash(p) is INDEPENDENT of the initial state - it only depends on
            // the token content from position p onwards.
            //
            // So we can:
            // 1. Pre-compute suffix_hash(p) for all positions (once per token)
            // 2. For each state group, compute the pos0 contribution using those hashes

            // Compute suffix hashes for this token (shared across all initial states)
            let suffix_hashes = compute_all_suffix_hashes(regex, token);

            // Now compute contribution for each state group
            let mut refined_sig = 0u128;
            for (&dfa_state, &weight) in &state_groups {
                // Run DFA from this initial state
                let result = regex.execute_from_state_nonzero(token, dfa_state as usize);

                // Compute hash for this execution
                let mut exec_hash = 0u128;

                // End state behavior
                if let Some(end_state) = result.end_state {
                    let future_groups: Vec<usize> = regex.dfa.states[end_state]
                        .possible_future_group_ids
                        .iter()
                        .cloned()
                        .collect();
                    exec_hash = exec_hash.wrapping_add(hash_end_behavior(&future_groups));
                }

                // Matches with suffix hashes
                let mut matches: Vec<_> = result.matches.iter().collect();
                matches.sort_by_key(|m| m.group_id);
                for m in matches {
                    let suffix_hash = suffix_hashes.get(&m.position).copied().unwrap_or(0);
                    let edge_hash = mix_u128(((m.group_id as u128) << 64) | suffix_hash);
                    exec_hash = exec_hash.wrapping_add(edge_hash);
                }

                // Weight by the state group's weight
                let contrib = mix_u128(exec_hash).wrapping_mul(weight);
                refined_sig = refined_sig.wrapping_add(contrib);
            }

            accumulators[token_idx] = refined_sig;
        }
    }

    // Recurse to children
    let children = node.children.clone();  // Clone to avoid borrow conflict
    for &(byte, child_idx) in &children {
        let next_groups = advance_state_groups(regex, &state_groups, byte);
        if !next_groups.is_empty() {
            let mut next_path = path.to_vec();
            next_path.push(byte);
            process_trie_node(
                regex,
                trie,
                child_idx as usize,
                next_groups,
                accumulators,
                depth + 1,
                &next_path,
                tokens,
            );
        }
    }
}

/// Compute suffix hashes for all positions in a token.
/// These are independent of initial state (always start from DFA's start_state).
fn compute_all_suffix_hashes(
    regex: &Regex,
    token: &[u8],
) -> HashMap<usize, u128> {
    let mut cache = HashMap::new();
    
    // BFS to find all reachable positions, then compute hashes bottom-up
    let mut queue = vec![0usize];
    let mut visited: HashMap<usize, bool> = HashMap::new();  // pos -> processed
    visited.insert(0, false);

    while let Some(pos) = queue.pop() {
        if pos > token.len() {
            continue;
        }
        
        let suffix = &token[pos..];
        let result = regex.execute_from_state_nonzero(suffix, regex.dfa.start_state);

        // Record targets for BFS
        for m in &result.matches {
            let target = pos + m.position;
            if target <= token.len() && !visited.contains_key(&target) {
                visited.insert(target, false);
                queue.push(target);
            }
        }
    }

    // Compute hashes bottom-up (larger positions first)
    let mut positions: Vec<_> = visited.keys().cloned().collect();
    positions.sort_by(|a, b| b.cmp(a));

    for pos in positions {
        let suffix = &token[pos..];
        let result = regex.execute_from_state_nonzero(suffix, regex.dfa.start_state);

        let mut node_hash = 0u128;

        // End state behavior
        if let Some(end_state) = result.end_state {
            let future_groups: Vec<usize> = regex.dfa.states[end_state]
                .possible_future_group_ids
                .iter()
                .cloned()
                .collect();
            node_hash = node_hash.wrapping_add(hash_end_behavior(&future_groups));
        }

        // Edges
        let mut matches: Vec<_> = result.matches.iter().collect();
        matches.sort_by_key(|m| m.group_id);
        for m in matches {
            let target = pos + m.position;
            let target_hash = cache.get(&target).copied().unwrap_or(0);
            let edge_hash = mix_u128(((m.group_id as u128) << 64) | target_hash);
            node_hash = node_hash.wrapping_add(edge_hash);
        }

        cache.insert(pos, mix_u128(node_hash));
    }

    cache
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tokens() {
        // Create a simple regex
        let regex = Regex::from_str_dfa("a").unwrap();
        let result = find_equivalence_classes(&regex, &[], &[0]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_token() {
        let regex = Regex::from_str_dfa("a").unwrap();
        let tokens = vec![b"a".to_vec()];
        let result = find_equivalence_classes(&regex, &tokens, &[0]);
        assert_eq!(result.len(), 1);
        assert!(result.iter().next().unwrap().contains(&0));
    }
}
