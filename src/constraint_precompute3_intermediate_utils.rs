// src/constraint_precompute3_intermediate_utils.rs
use crate::{
    constraint::{
        IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey,
        IntermediateTrie3GodWrapper, LLMTokenBV,
    },
    datastructures::trie::Trie,
};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Normalizes a path for comparison purposes.
/// - Removes NoOp edges.
/// - Collects all CheckLLM bitvectors, intersects them, and prepends a single CheckLLM.
pub(crate) fn normalize_path(path: Vec<IntermediateTrie3EdgeKey>) -> Vec<IntermediateTrie3EdgeKey> {
    let mut combined_llm_bv = LLMTokenBV::max_ones();
    let mut has_llm_check = false;

    let mut other_ops: Vec<_> = path
        .into_iter()
        .filter_map(|ek| match ek {
            IntermediateTrie3EdgeKey::CheckLLM(bv) => {
                combined_llm_bv &= bv;
                has_llm_check = true;
                None
            }
            IntermediateTrie3EdgeKey::NoOp => None,
            IntermediateTrie3EdgeKey::Push(_) | IntermediateTrie3EdgeKey::Pop(_, _) => Some(ek),
        })
        .collect();

    if has_llm_check {
        other_ops.insert(0, IntermediateTrie3EdgeKey::CheckLLM(combined_llm_bv));
    }

    other_ops
}

/// Compares two Intermediate Trie3 graphs for equivalence by comparing their sets of normalized paths.
/// This is a strong equivalence check, suitable for testing optimization passes.
pub fn are_intermediate_trie3_graphs_equal<F>(
    roots_a: &[IntermediatePrecomputeNode3Index],
    god_a: &IntermediateTrie3GodWrapper,
    roots_b: &[IntermediatePrecomputeNode3Index],
    god_b: &IntermediateTrie3GodWrapper,
    is_end: &F,
    max_path_length: usize,
) -> bool
where
    F: Fn(IntermediatePrecomputeNode3Index, &IntermediatePrecomputeNode3) -> bool,
{
    // Only Pop and Push operations count towards path length for cycle detection.
    let is_path_edge: fn(&IntermediateTrie3EdgeKey, &(), IntermediatePrecomputeNode3Index) -> bool =
        |ek, _, _| {
            !matches!(ek, IntermediateTrie3EdgeKey::NoOp)
        };

    let get_normalized_paths = |god, roots| {
        Trie::get_all_paths_with_cycles(god, roots, is_end, is_path_edge, max_path_length)
            .into_iter()
            .map(|(_, path)| normalize_path(path.into_iter().map(|(ek, ..)| ek).collect()))
            .collect::<HashSet<_>>()
    };

    get_normalized_paths(god_a, roots_a) == get_normalized_paths(god_b, roots_b)
}

pub fn optimize_intermediate_trie3(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
    is_end: impl Fn(IntermediatePrecomputeNode3Index, &IntermediatePrecomputeNode3) -> bool,
) -> BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> {
    let original_god = god.deep_clone();
    let original_roots = roots.to_vec();

    // The signature of a node, used for content-addressing.
    // It includes the node's own value and a canonical representation of its children.
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct NodeSignature {
        value: crate::constraint::IntermediatePrecomputedNodeContents3,
        children: BTreeMap<IntermediateTrie3EdgeKey, BTreeMap<IntermediatePrecomputeNode3Index, ()>>,
    }

    // 1. Recompute depths for bottom-up traversal
    Trie::recompute_all_max_depths(god, roots);

    // 2. Get all nodes and group them by depth
    let all_nodes = Trie::all_nodes(god, roots);
    let mut nodes_by_depth: BTreeMap<usize, Vec<_>> = BTreeMap::new();
    for node_idx in &all_nodes {
        // It's safe to unwrap as all_nodes are guaranteed to be in the arena.
        god.with(node_idx.as_index(), |node| {
            nodes_by_depth.entry(node.max_depth).or_default().push(*node_idx);
        })
        .unwrap();
    }

    // 3. Initialize maps for the optimization process
    // node_map: maps original node index to its new, optimized index
    let mut node_map: BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> =
        BTreeMap::new();
    // signature_map: maps a node's signature to its optimized index
    let mut signature_map: HashMap<NodeSignature, IntermediatePrecomputeNode3Index> =
        HashMap::new();

    // 4. Progress reporting setup
    let total_nodes = all_nodes.len();
    eprintln!("\n--- Starting Intermediate Trie3 Optimization ---");
    eprintln!("Found {} total nodes to process.", total_nodes);

    // 5. Main optimization loop: process nodes from leaves up to roots (bottom-up)
    for (depth, nodes) in nodes_by_depth.iter().rev() {
        eprintln!("> Processing depth {}: {} nodes...", depth, nodes.len());
        for &original_idx in nodes {
            // To avoid holding a lock while building the signature, we clone the node's data.
            let original_node_clone = god.get(original_idx.as_index()).unwrap();

            // Build the signature for the current node.
            // Children's indices are replaced with their already-computed optimized indices from node_map.
            let mut new_children = BTreeMap::new();
            for (ek, child_map) in original_node_clone.children() {
                let mut new_child_map = BTreeMap::new();
                for &child_idx in child_map.keys() {
                    let &new_child_idx = node_map
                        .get(&child_idx)
                        .expect("Child not yet processed; bug in bottom-up traversal logic.");
                    new_child_map.insert(new_child_idx, ());
                }
                new_children.insert(ek.clone(), new_child_map);
            }

            let signature = NodeSignature {
                value: original_node_clone.value.clone(),
                children: new_children,
            };

            // Look up the signature to see if we've already created an equivalent node.
            if let Some(&existing_idx) = signature_map.get(&signature) {
                // We found a duplicate. Map the original node to the existing optimized one.
                node_map.insert(original_idx, existing_idx);
            } else {
                // This is the first time we've seen this structure. Create a new optimized node.
                let mut new_node =
                    IntermediatePrecomputeNode3::new(original_node_clone.value.clone());
                for (ek, child_map) in signature.children.iter() {
                    for &new_child_idx in child_map.keys() {
                        // The edge value for IntermediatePrecomputeNode3 is `()`.
                        new_node.force_insert_to_node(ek.clone(), (), new_child_idx);
                    }
                }

                let new_idx = IntermediatePrecomputeNode3Index::new(god.insert(new_node));
                node_map.insert(original_idx, new_idx);
                signature_map.insert(signature, new_idx);
            }
        }
    }
    eprintln!("> Optimization pass complete.");

    // 6. Determine the new roots and perform garbage collection
    let new_roots: Vec<_> = original_roots
        .iter()
        .map(|r| *node_map.get(r).unwrap_or(r))
        .collect();

    eprintln!("> Garbage collecting unused nodes...");
    let stats_before = Trie::stats(god, &new_roots);
    Trie::gc(god, &new_roots);
    let stats_after = Trie::stats(god, &new_roots);
    let node_pct_change = if stats_before.num_reachable_nodes == 0 { 0.0 } else { (stats_after.num_reachable_nodes as f64 / stats_before.num_reachable_nodes as f64 - 1.0) * 100.0 };
    let edge_pct_change = if stats_before.num_reachable_edges == 0 { 0.0 } else { (stats_after.num_reachable_edges as f64 / stats_before.num_reachable_edges as f64 - 1.0) * 100.0 };
    eprintln!(
        "> GC complete. Nodes: {} -> {} ({:+.2}%), Edges: {} -> {} ({:+.2}%)",
        stats_before.num_reachable_nodes,
        stats_after.num_reachable_nodes,
        node_pct_change,
        stats_before.num_reachable_edges,
        stats_after.num_reachable_edges,
        edge_pct_change
    );
    eprintln!("--- Optimization Finished ---\n");

    // 7. Check equivalence after optimization
    assert!(
        are_intermediate_trie3_graphs_equal(&original_roots, &original_god, &new_roots, god, &is_end, 25),
        "Optimization failed to preserve graph equivalence for all roots"
    );

    node_map
}
