//! State Equivalence Analysis - Graph-Signature Implementation
//!
//! This is a simple specialization of the reference graph-signature algorithm
//! for state equivalence analysis only.
//!
//! For each tokenizer state, we build a combined signature across all tokens.
//! States with identical signatures are equivalent.
//!
//! Complexity: O(states × tokens × token_length × graph_size)

use crate::dfa_u8::Tokenizer;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, VecDeque};
use std::hash::{Hash, Hasher};

/// The result of state equivalence analysis: sets of state IDs that behave identically.
pub type StateEquivalenceResult = BTreeSet<BTreeSet<usize>>;

/// Computes a deterministic hash representing the parsing structure of a token
/// from a particular start state.
///
/// This is a direct specialization of the reference vocab-equivalence
/// signature computation. It builds the reachable parse graph over token
/// positions, then hashes it from the end of the token back to the start.
pub fn compute_signature(tokenizer: &Tokenizer, slice: &[u8], start_state: usize) -> u64 {
    // 1. Forward pass: build a graph of reachable token-position nodes.
    //    We map: position -> (completion data, outgoing edges).
    let mut graph: HashMap<usize, (Option<Vec<usize>>, Vec<(usize, usize)>)> = HashMap::new();
    let mut queue = VecDeque::from([0usize]);
    let mut visited: HashSet<usize> = HashSet::from([0usize]);

    while let Some(pos) = queue.pop_front() {
        if pos > slice.len() {
            continue;
        }

        let exec_start = if pos == 0 {
            start_state
        } else {
            tokenizer.initial_state_id().0
        };
        let result = tokenizer.execute_from_state_nonzero(&slice[pos..], exec_start);

        let mut edges = Vec::with_capacity(result.matches.len());
        for m in result.matches {
            let target = pos + m.position;
            if target != slice.len() || target <= pos {
                continue;
            }
            edges.push((m.group_id, target));

            if visited.insert(target) {
                queue.push_back(target);
            }
        }

        // Sort edges by group ID so the hash is consistent regardless of execution order.
        edges.sort_unstable_by_key(|e| e.0);

        let completion = result.end_state.map(|id| {
            tokenizer.dfa().states[id]
                .possible_future_group_ids
                .iter()
                .copied()
                .collect::<Vec<_>>()
        });

        graph.insert(pos, (completion, edges));
    }

    // 2. Backward pass: hash nodes from the end of the token back to the start.
    let mut positions: Vec<_> = graph.keys().copied().collect();
    positions.sort_unstable_by(|a, b| b.cmp(a));

    let mut node_hashes: HashMap<usize, u64> = HashMap::with_capacity(graph.len());

    for pos in positions {
        let (completion, edges) = graph.get(&pos).expect("graph node must exist");
        let mut hasher = DefaultHasher::new();

        // Hash the local state (completion data).
        completion.hash(&mut hasher);

        // Hash the structural connections (outgoing edges + target hashes).
        for (group_id, target) in edges {
            let target_hash = node_hashes
                .get(target)
                .expect("target must be processed before source");
            (group_id, target_hash).hash(&mut hasher);
        }

        node_hashes.insert(pos, hasher.finish());
    }

    // The signature is the hash of the root node (position 0).
    node_hashes.get(&0).copied().unwrap_or(0)
}

/// Find state equivalence classes using the graph-signature algorithm.
///
/// # Arguments
/// * `tokenizer` - The tokenizer DFA
/// * `tokens` - Vocabulary tokens to analyze
/// * `states` - Tokenizer states to consider for equivalence
///
/// # Returns
/// A vector where `result[i]` is the representative state for `states[i]`.
/// States with the same representative are equivalent.
pub fn find_state_equivalence_classes(
    tokenizer: &Tokenizer,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> Vec<usize> {
    if states.is_empty() {
        return Vec::new();
    }

    let start = std::time::Instant::now();

    // Compute a combined signature for every state in parallel.
    let state_signatures: Vec<u64> = states
        .par_iter()
        .map(|&state| {
            let mut hasher = DefaultHasher::new();
            for token in tokens {
                compute_signature(tokenizer, token, state).hash(&mut hasher);
            }
            hasher.finish()
        })
        .collect();

    // Group states by signature and choose representatives.
    let mut sig_to_rep: HashMap<u64, usize> = HashMap::new();
    let mut mapping: Vec<usize> = Vec::with_capacity(states.len());

    for (idx, sig) in state_signatures.into_iter().enumerate() {
        let rep = *sig_to_rep.entry(sig).or_insert(states[idx]);
        mapping.push(rep);
    }

    crate::debug!(
        3,
        "State equivalence analysis (graph-signature) took {:?}. Reduced {} states to {}.",
        start.elapsed(),
        states.len(),
        mapping.iter().copied().collect::<BTreeSet<_>>().len(),
    );

    mapping
}

/// Convert a state-to-representative mapping to StateEquivalenceResult format.
pub fn mapping_to_equivalence_classes(
    states: &[usize],
    mapping: &[usize],
) -> StateEquivalenceResult {
    let mut rep_to_class: std::collections::BTreeMap<usize, BTreeSet<usize>> =
        std::collections::BTreeMap::new();

    for (i, &rep) in mapping.iter().enumerate() {
        rep_to_class.entry(rep).or_default().insert(states[i]);
    }

    rep_to_class.into_values().collect()
}
