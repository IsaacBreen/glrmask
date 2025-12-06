use crate::finite_automata::Regex;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::collections::{BTreeSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

// A node in our parse graph, stored in a flat Vec for cache efficiency.
// The lifetime 'a is tied to the lifetime of the Regex object.
struct GraphNode<'a> {
    completion: Option<&'a BTreeSet<u16>>,
    edges: Vec<(u16, usize)>,
}

/// Computes a deterministic hash representing the parsing structure of the string.
/// This version is optimized for performance by prioritizing cache locality and reducing allocations.
fn compute_signature(regex: &Regex, slice: &[u8], start_state: usize) -> u64 {
    let len = slice.len();
    // Optimization: Use Vecs instead of HashMaps/HashSets for graph data.
    // Since our keys are string positions (0..=len), a Vec is a perfect,
    // cache-friendly replacement for a hash map. This is known as a direct-address table.
    let mut graph: Vec<Option<GraphNode>> = vec![None; len + 1];
    let mut queue = VecDeque::with_capacity(len + 1);
    // A Vec<bool> is significantly faster and more memory-efficient than a HashSet<usize>
    // for dense integer sets.
    let mut visited = vec![false; len + 1];

    // --- 1. Forward Pass: Build a graph of valid state transitions ---
    queue.push_back(0);
    visited[0] = true;

    // Optimization: Reuse the `edges` allocation.
    // Instead of creating a new Vec on every loop iteration, we create it once
    // and clear it. This avoids repeated calls to the allocator.
    let mut temp_edges = Vec::new();

    while let Some(pos) = queue.pop_front() {
        // The check `pos > len` from the original code is now implicitly handled
        // by the bounds of our `visited` and `graph` Vecs.

        let exec_start = if pos == 0 { start_state } else { regex.dfa.start_state };
        let result = regex.execute_from_state_nonzero(&slice[pos..], exec_start);

        temp_edges.clear();
        for m in result.matches {
            let target = pos + m.position;
            if target <= len { // Ensure we don't go out of bounds
                temp_edges.push((m.group_id, target));

                if !visited[target] {
                    visited[target] = true;
                    queue.push_back(target);
                }
            }
        }

        // Sort edges by Group ID so the hash is consistent regardless of execution order.
        temp_edges.sort_unstable_by_key(|e| e.0);

        let completion = result.end_state.map(|id| &regex.dfa.states[id].possible_future_group_ids);

        // We must clone `temp_edges` here, but we've still avoided N-1 allocations.
        graph[pos] = Some(GraphNode { completion, edges: temp_edges.clone() });
    }

    // --- 2. Backward Pass: Calculate hashes from the end to the start ---
    let mut node_hashes = vec![0u64; len + 1];

    // Optimization: Iterate backwards over the Vec indices.
    // This is much faster than collecting keys from a HashMap and sorting them.
    // It works because edges only point forward (pos -> target where target > pos),
    // so a reverse traversal is a valid topological sort.
    for pos in (0..=len).rev() {
        if let Some(node) = &graph[pos] {
            let mut hasher = DefaultHasher::new();

            // Hash the local state (completion data).
            node.completion.hash(&mut hasher);

            // Hash the structural connections (outgoing edges + hash of target nodes).
            for (group_id, target) in &node.edges {
                // Accessing the target hash is a simple, fast array lookup.
                let target_hash = node_hashes[*target];
                (group_id, target_hash).hash(&mut hasher);
            }

            node_hashes[pos] = hasher.finish();
        }
    }

    // The signature is the hash of the root node (position 0).
    node_hashes[0]
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    // Compute a unique signature for every string in parallel.
    // This part of the logic is already well-parallelized and doesn't need changes.
    let signatures: Vec<u64> = strings
        .par_iter()
        .map(|s| {
            let mut hasher = DefaultHasher::new();
            // Combine signatures for all requested start states into one final hash.
            for &state in initial_states {
                compute_signature(regex, s, state).hash(&mut hasher);
            }
            hasher.finish()
        })
        .collect();

    // Group string indices by their computed signature.
    let mut groups = HashMap::new();
    for (index, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_insert_with(Vec::new).push(index);
    }

    // The final collection into BTreeSet is required by the function signature.
    groups.into_values().collect()
}