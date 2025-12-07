//! Reference Implementation of Vocab Equivalence Analysis
//!
//! This is a slow but correct reference implementation used for testing
//! and validation. It computes token signatures by building and hashing
//! the full parse graph for each token.
//!
//! Complexity: O(tokens × states × token_length × graph_size)

use crate::finite_automata::Regex;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, VecDeque};
use std::hash::{Hash, Hasher};

pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;

/// Computes a deterministic hash representing the parsing structure of the string.
pub fn compute_signature(regex: &Regex, slice: &[u8], start_state: usize) -> u64 {
    // 1. Forward Pass: Build a graph of valid state transitions (nodes and edges).
    // We Map: Position -> (EndStateData, List of Outgoing Edges)
    let mut graph = HashMap::new();
    let mut queue = VecDeque::from([0]);
    let mut visited: HashSet<usize> = HashSet::from([0]);

    while let Some(pos) = queue.pop_front() {
        if pos > slice.len() {
            continue;
        }

        let exec_start = if pos == 0 {
            start_state
        } else {
            regex.dfa.start_state
        };
        let result = regex.execute_from_state_nonzero(&slice[pos..], exec_start);

        let mut edges = Vec::with_capacity(result.matches.len());
        for m in result.matches {
            let target = pos + m.position;
            edges.push((m.group_id, target));

            if visited.insert(target) {
                queue.push_back(target);
            }
        }

        // Sort edges by Group ID so the hash is consistent regardless of execution order
        edges.sort_unstable_by_key(|e| e.0);

        let completion = result
            .end_state
            .map(|id| &regex.dfa.states[id].possible_future_group_ids);
        graph.insert(pos, (completion, edges));
    }

    // 2. Backward Pass: Calculate hashes from the end of the string back to the start.
    // We sort positions descending to ensure we hash a target node before the node pointing to it.
    let mut positions: Vec<_> = graph.keys().copied().collect();
    positions.sort_unstable_by(|a, b| b.cmp(a));

    let mut node_hashes = HashMap::with_capacity(graph.len());

    for pos in positions {
        let (completion, edges) = &graph[&pos];
        let mut hasher = DefaultHasher::new();

        // Hash the local state (completion data)
        completion.hash(&mut hasher);

        // Hash the structural connections (outgoing edges + hash of target nodes)
        for (group_id, target) in edges {
            let target_hash = node_hashes
                .get(target)
                .expect("Target must be processed before Source");
            (group_id, target_hash).hash(&mut hasher);
        }

        node_hashes.insert(pos, hasher.finish());
    }

    // The signature is the hash of the root node (position 0)
    node_hashes.get(&0).copied().unwrap_or(0)
}

/// Find vocab equivalence classes using the reference algorithm.
///
/// # Arguments
/// * `regex` - The tokenizer DFA
/// * `strings` - Vocabulary tokens to analyze
/// * `initial_states` - Tokenizer states to consider for equivalence
///
/// # Returns
/// Sets of token indices that are equivalent (produce identical parsing behavior).
pub fn find_vocab_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> VocabEquivalenceResult {
    // Compute a unique signature for every string in parallel
    let signatures: Vec<u64> = strings
        .par_iter()
        .map(|s| {
            let mut hasher = DefaultHasher::new();
            // Combine signatures for all requested start states into one final hash
            for &state in initial_states {
                compute_signature(regex, s, state).hash(&mut hasher);
            }
            hasher.finish()
        })
        .collect();

    // Group string indices by their computed signature
    let mut groups = HashMap::new();
    for (index, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_insert_with(Vec::new).push(index);
    }

    groups.into_values().collect()
}
