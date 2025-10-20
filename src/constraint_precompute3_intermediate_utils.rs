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

    // Helper for post-order traversal
    fn post_order_dfs(
        idx: IntermediatePrecomputeNode3Index,
        god: &IntermediateTrie3GodWrapper,
        visited: &mut HashSet<IntermediatePrecomputeNode3Index>,
        post_order: &mut Vec<IntermediatePrecomputeNode3Index>,
    ) {
        if !visited.insert(idx) {
            return;
        }
        if god
            .with(idx.as_index(), |node| {
                for (_, children_map) in node.children() {
                    for (child_idx, _) in children_map {
                        post_order_dfs(*child_idx, god, visited, post_order);
                    }
                }
            })
            .is_some()
        {
            post_order.push(idx);
        }
    }

    // 1. Get all reachable nodes and build a post-order traversal list
    let all_original_nodes = Trie::all_nodes(&original_god, &original_roots);
    let mut post_order = Vec::new();
    let mut visited = HashSet::new();
    for root in &all_original_nodes {
        post_order_dfs(*root, &original_god, &mut visited, &mut post_order);
    }

    // 2. Set up for optimization
    #[derive(PartialEq, Eq, Hash, Clone, PartialOrd, Ord)]
    struct NodeSignature {
        is_end: bool,
        edges: Vec<(IntermediateTrie3EdgeKey, IntermediatePrecomputeNode3Index)>,
    }

    let mut node_map: BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> =
        BTreeMap::new();
    let mut signature_map: HashMap<NodeSignature, IntermediatePrecomputeNode3Index> =
        HashMap::new();
    let optimized_god = IntermediateTrie3GodWrapper::new();

    let total_nodes = post_order.len();
    println!("Optimizing intermediate trie3: {} nodes to process", total_nodes);

    // 3. Process nodes in post-order
    for (i, &original_idx) in post_order.iter().enumerate() {
        if (i + 1) % 5000 == 0 || i == total_nodes - 1 {
            println!("  ...processed {}/{} nodes", i + 1, total_nodes);
        }

        let original_node = original_god
            .with(original_idx.as_index(), |n| n.clone())
            .unwrap();

        // Build canonical list of optimized child edges
        let mut optimized_edges = Vec::new();
        for (edge_key, children_map) in original_node.children() {
            for (original_child_idx, _) in children_map {
                let optimized_child_idx = node_map.get(original_child_idx).expect(
                    "Child node should have been processed in post-order traversal",
                );
                optimized_edges.push((edge_key.clone(), *optimized_child_idx));
            }
        }
        optimized_edges.sort_unstable();

        let signature = NodeSignature {
            is_end: original_node.value.end,
            edges: optimized_edges,
        };

        if let Some(&existing_optimized_idx) = signature_map.get(&signature) {
            node_map.insert(original_idx, existing_optimized_idx);
        } else {
            let new_optimized_idx = optimized_god.insert(original_node.value.clone()).into();

            optimized_god
                .with_mut(new_optimized_idx.as_index(), |new_node| {
                    for (edge_key, optimized_child_idx) in &signature.edges {
                        new_node.force_insert_to_node(edge_key.clone(), (), *optimized_child_idx);
                    }
                })
                .unwrap();

            node_map.insert(original_idx, new_optimized_idx);
            signature_map.insert(signature, new_optimized_idx);
        }
    }

    let optimized_roots: Vec<_> = original_roots
        .iter()
        .map(|r| *node_map.get(r).unwrap())
        .collect();

    let stats_before = Trie::stats(&original_god, &original_roots);
    let stats_after = Trie::stats(&optimized_god, &optimized_roots);
    println!(
        "Optimization complete. Nodes: {} -> {}, Edges: {} -> {}",
        stats_before.num_reachable_nodes, stats_after.num_reachable_nodes,
        stats_before.num_reachable_edges, stats_after.num_reachable_edges
    );

    god.clear();
    let (_, optimized_to_final_map) =
        Trie::deep_copy_subtrees_into(&optimized_god, god, &optimized_roots);

    let final_node_map: BTreeMap<_, _> = node_map
        .into_iter()
        .filter_map(|(original_idx, optimized_idx)| {
            optimized_to_final_map
                .get(&optimized_idx)
                .map(|&final_idx| (original_idx, final_idx))
        })
        .collect();

    let new_roots: Vec<_> = original_roots
        .iter()
        .map(|r| *final_node_map.get(r).unwrap_or(r))
        .collect();

    assert!(
        are_intermediate_trie3_graphs_equal(&original_roots, &original_god, &new_roots, god, &is_end, 25),
        "Optimization failed to preserve graph equivalence for all roots"
    );

    final_node_map
}
