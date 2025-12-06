use crate::finite_automata::Regex;
use hashbrown::{hash_map::DefaultHashBuilder, HashMap};
use rayon::prelude::*;
use std::collections::{BTreeSet, VecDeque};
use std::hash::{BuildHasher, Hash, Hasher};
use std::mem;

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

/// A node in our parse graph. To improve performance, node data is stored in a
/// struct-of-arrays style: node metadata is in a `Vec<FlatGraphNode>`, and all
/// edges are stored contiguously in a single separate `Vec`.
struct FlatGraphNode<'a> {
    completion: Option<&'a BTreeSet<usize>>,
    edges_start: usize,
    edges_len: usize,
}

/// Computes a deterministic hash representing the parsing structure of the string.
/// This version is optimized for performance by prioritizing cache locality and reducing allocations.
fn compute_signature(regex: &Regex, slice: &[u8], start_state: usize) -> u64 {
    let len = slice.len();

    // By pre-filling the graph with "empty" nodes, we avoid checks later on.
    let mut graph_nodes: Vec<FlatGraphNode> = std::iter::repeat_with(|| FlatGraphNode {
        completion: None,
        edges_start: 0,
        edges_len: 0,
    })
    .take(len + 1)
    .collect();
    let mut all_edges: Vec<(usize, usize)> = Vec::with_capacity(len * 2); // Heuristic

    // Use a Vec as a stack for a DFS traversal. This is often faster than BFS
    // due to better cache locality.
    let mut stack = Vec::with_capacity(len + 1);
    let mut visited = vec![false; len + 1];

    // --- 1. Forward Pass: Build a graph of valid state transitions ---
    stack.push(0);
    visited[0] = true;

    let mut temp_edges = Vec::new();

    while let Some(pos) = stack.pop() {
        let exec_start = if pos == 0 { start_state } else { regex.dfa.start_state };
        let result = regex.execute_from_state_nonzero(&slice[pos..], exec_start);

        temp_edges.clear();
        for m in result.matches {
            let target = pos + m.position;
            if target <= len {
                temp_edges.push((m.group_id, target));

                if !visited[target] {
                    visited[target] = true;
                    stack.push(target);
                }
            }
        }

        temp_edges.sort_unstable_by_key(|e| e.0);

        // Store edges in the flat `all_edges` vector and record the slice info.
        let node = &mut graph_nodes[pos];
        node.completion = result.end_state.map(|id| &regex.dfa.states[id].possible_future_group_ids);
        node.edges_start = all_edges.len();
        node.edges_len = temp_edges.len();
        all_edges.extend_from_slice(&temp_edges);
    }

    // --- 2. Backward Pass: Calculate hashes from the end to the start ---
    let mut node_hashes = vec![0u64; len + 1];

    // Optimization: Use a faster hashing algorithm. `hashbrown`'s default hasher (AHasher)
    // is significantly faster than the standard library's `DefaultHasher` (SipHash-1-3)
    // for non-cryptographic use cases like this.
    let hasher_builder = DefaultHashBuilder::default();

    // The reverse iteration is a valid topological sort of the graph, as edges only
    // point to greater indices. This ensures that when we hash a node, the hashes
    // of all nodes it points to have already been computed.
    for pos in (0..=len).rev() {
        let node = &graph_nodes[pos];
        let mut hasher = hasher_builder.build_hasher();

        // Hash the local state (completion data).
        node.completion.hash(&mut hasher);

        // Hash the structural connections (outgoing edges + hash of target nodes).
        let edges_slice = &all_edges[node.edges_start..node.edges_start + node.edges_len];
        for (group_id, target) in edges_slice {
            let target_hash = node_hashes[*target];
            (group_id, target_hash).hash(&mut hasher);
        }

        node_hashes[pos] = hasher.finish();
    }

    // The signature is the hash of the root node (position 0).
    node_hashes[0]
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    // Optimization: Use a faster hashing algorithm via `hashbrown`.
    let hasher_builder = DefaultHashBuilder::default();

    let signatures: Vec<u64> = strings
        .par_iter()
        .map(|s| {
            // Optimization: If there's only one start state, its signature is the final
            // signature. This avoids a redundant re-hashing of the u64.
            if initial_states.len() == 1 {
                return compute_signature(regex, s, initial_states[0]);
            }

            // Combine signatures for all requested start states into one final hash.
            let mut hasher = hasher_builder.build_hasher();
            for &state in initial_states {
                compute_signature(regex, s, state).hash(&mut hasher);
            }
            hasher.finish()
        })
        .collect();

    // Group string indices by their computed signature. This is efficient.
    let mut groups = HashMap::new();
    for (index, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_insert_with(Vec::new).push(index);
    }

    // The final collection into BTreeSet is required by the function signature.
    groups.into_values().collect()
}