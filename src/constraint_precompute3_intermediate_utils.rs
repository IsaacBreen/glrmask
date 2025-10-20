// src/constraint_precompute3_intermediate_utils.rs
use crate::{
    constraint::{
        IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey,
        IntermediateTrie3GodWrapper, LLMTokenBV,
    },
    datastructures::trie::Trie,
};
use std::collections::{BTreeMap, HashSet};

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
            _ => Some(ek),
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
            matches!(ek, IntermediateTrie3EdgeKey::Pop(_, _) | IntermediateTrie3EdgeKey::Push(_))
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

    let node_map: BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> =
        BTreeMap::default();

    // TODO

    // Check equivalence after optimization (currently no-op)
    let new_roots: Vec<_> = original_roots
        .iter()
        .map(|r| *node_map.get(r).unwrap_or(r))
        .collect();

    assert!(
        are_intermediate_trie3_graphs_equal(&original_roots, &original_god, &new_roots, god, &is_end, 25),
        "Optimization failed to preserve graph equivalence for all roots"
    );

    node_map
}
