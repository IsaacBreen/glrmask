// src/constraint_precompute3_intermediate_utils.rs
use crate::constraint::{
    IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey,
    IntermediateTrie3GodWrapper, LLMTokenBV,
};
use std::collections::BTreeMap;
use crate::datastructures::trie::Trie;
use std::collections::HashSet;

/// Normalizes a path for comparison purposes.
/// - Removes NoOp edges.
/// - Collects all CheckLLM bitvectors, intersects them, and prepends a single CheckLLM.
pub(crate) fn normalize_path(path: Vec<IntermediateTrie3EdgeKey>) -> Vec<IntermediateTrie3EdgeKey> {
    let mut combined_llm_bv = LLMTokenBV::max_ones();
    let mut has_llm_check = false;

    let mut other_ops: Vec<IntermediateTrie3EdgeKey> = path
        .into_iter()
        .filter(|ek| {
            if let IntermediateTrie3EdgeKey::CheckLLM(bv) = ek {
                combined_llm_bv &= bv;
                has_llm_check = true;
                false // remove from path
            } else {
                !matches!(ek, IntermediateTrie3EdgeKey::NoOp)
            }
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
    let counts_toward_length: fn(&IntermediateTrie3EdgeKey, &(), IntermediatePrecomputeNode3Index) -> bool = |ek, _, _| {
        // Only Pop and Push operations count towards path length for cycle detection.
        // CheckLLM and NoOp are considered "free" moves.
        matches!(ek, IntermediateTrie3EdgeKey::Pop(_, _)) || matches!(ek, IntermediateTrie3EdgeKey::Push(_))
    };

    // 1. Get all paths for graph A
    let paths_a = Trie::get_all_paths_with_cycles(
        god_a,
        roots_a,
        is_end,
        &counts_toward_length,
        max_path_length,
    );

    // 2. Get all paths for graph B
    let paths_b = Trie::get_all_paths_with_cycles(
        god_b,
        roots_b,
        is_end,
        &counts_toward_length,
        max_path_length,
    );

    // 3. Normalize and collect paths into sets
    let normalized_paths_a: HashSet<Vec<IntermediateTrie3EdgeKey>> = paths_a.into_iter().map(|(_, path_edges)| {
        let edge_keys: Vec<IntermediateTrie3EdgeKey> = path_edges.into_iter().map(|(ek, _, _)| ek).collect();
        normalize_path(edge_keys)
    }).collect();

    let normalized_paths_b: HashSet<Vec<IntermediateTrie3EdgeKey>> = paths_b.into_iter().map(|(_, path_edges)| {
        let edge_keys: Vec<IntermediateTrie3EdgeKey> = path_edges.into_iter().map(|(ek, _, _)| ek).collect();
        normalize_path(edge_keys)
    }).collect();

    // 4. Compare the sets
    normalized_paths_a == normalized_paths_b
}

pub fn optimize_intermediate_trie3(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
    is_end: impl Fn(IntermediatePrecomputeNode3Index, &IntermediatePrecomputeNode3) -> bool,
    // The return type is correct for a node mapping (old node -> new node)
) -> BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> {
    let original_god = god.deep_clone();
    let original_roots = roots.to_vec();

    let node_map: BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> = BTreeMap::default();

    // TODO

    // Check equivalence after optimization (currently no-op)
    let new_roots: Vec<_> = original_roots.iter()
        .map(|r| node_map.get(r).unwrap_or(r).clone())
        .collect();

    assert!(
        are_intermediate_trie3_graphs_equal(&original_roots, &original_god, &new_roots, god, &is_end, 25),
        "Optimization failed to preserve graph equivalence for all roots"
    );

    node_map
}
