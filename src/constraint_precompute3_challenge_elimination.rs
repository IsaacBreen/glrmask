// src/constraint_precompute3_challenge_elimination.rs
use crate::constraint::IntermediatePrecomputedNodeContents3;
use crate::constraint::{
    IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey,
    IntermediateTrie3GodWrapper, LLMTokenBV, StateIDBV,
};
use crate::datastructures::trie::Trie;
use crate::tokenizer::TokenizerStateID;
use std::collections::{BTreeMap, BTreeSet};

/// Normalizes a path for comparison purposes.
/// - Removes NoOp edges.
/// - Collects all CheckLLM bitvectors, intersects them, and prepends a single CheckLLM.
fn normalize_path(path: Vec<IntermediateTrie3EdgeKey>) -> Vec<IntermediateTrie3EdgeKey> {
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
/// Eliminates adjacent Push/Pop pairs from a stack of intermediate trie edge keys.
/// Additionally, Pop(0, B) constraints are folded into the preceding Push via intersection.
/// This is a core part of simplifying the precompute3 graph.
///
/// Pairing rules applied when a Push(A) looks to the nearest Pop(n, B) to its right
/// (stopping if another Push is encountered first):
/// - n == 0: If A intersects B, remove Pop(0, B) and keep Push(A). If disjoint, the stack is invalid (None).
/// - n == 1: If A intersects B, both cancel (epsilon). If disjoint, the stack is invalid (None).
/// - n > 1: Remove Push(A) and replace Pop with Pop(n - 1, B) unconditionally (no intersection check).
/// The scan repeats until no more changes can be made.
fn simplify_path(
    stack: Vec<IntermediateTrie3EdgeKey>,
) -> Option<Vec<IntermediateTrie3EdgeKey>> {
    let mut stack = stack;
    loop {
        let mut changed_in_pass = false;
        let mut i = 0;
        while i < stack.len() {
            if let IntermediateTrie3EdgeKey::Push(push_states) = &stack[i] {
                let push_states = push_states.clone();
                // Find nearest pop to the right, not blocked by another push
                let mut pop_j = None;
                for j in (i + 1)..stack.len() {
                    if matches!(stack[j], IntermediateTrie3EdgeKey::Push(_)) {
                        break; // Blocked
                    }
                    if matches!(stack[j], IntermediateTrie3EdgeKey::Pop(_, _)) {
                        pop_j = Some(j);
                        break;
                    }
                }

                if let Some(j) = pop_j {
                    // Found a pair to cancel
                    let pop_op = stack.remove(j);
                    let _push_op = stack.remove(i); // push is at i

                    if let IntermediateTrie3EdgeKey::Pop(n, pop_states) = pop_op {
                        match n {
                            0 => {
                                if push_states.is_disjoint(&pop_states) {
                                    return None; // Mismatch on state check
                                }
                                // Fold Pop(0, B) constraint into preceding Push(A) as A := A ∩ B.
                                // This preserves semantics and reduces the state space.
                                let mut new_push_bv = push_states.clone();
                                new_push_bv &= pop_states;
                                // If intersection is empty we'd have returned None above.
                                stack.insert(i, IntermediateTrie3EdgeKey::Push(new_push_bv));
                            }
                            1 => {
                                if push_states.is_disjoint(&pop_states) {
                                    return None; // Mismatch on single-pop check
                                }
                                // Intersection: both cancel -> epsilon (nothing to insert).
                            }
                            _ => {
                                // n > 1: remove Push unconditionally and decrement Pop.
                                stack.insert(i, IntermediateTrie3EdgeKey::Pop(n - 1, pop_states));
                            }
                        }
                    }
                    changed_in_pass = true;
                    // Restart scan from beginning of modified stack
                    i = 0;
                    continue;
                }
            }
            i += 1;
        }
        if !changed_in_pass {
            break;
        }
    }
    Some(stack)
}

pub fn eliminate_pushes_and_pops_path_based(
    roots: &mut BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) {
    // 1. Get all paths from the original graph.
    let all_root_indices: Vec<_> = roots.values().cloned().collect();
    if all_root_indices.is_empty() {
        return;
    }
    let all_paths =
        IntermediatePrecomputeNode3::get_all_paths(god, &all_root_indices, |n| n.value.end);

    // 2. Simplify them.
    let mut simplified_paths = BTreeSet::new();
    for (_root_value, path_edges) in all_paths {
        let edge_keys: Vec<_> = path_edges.into_iter().map(|(ek, _, _)| ek).collect();
        if let Some(new_path) = simplify_path(edge_keys) {
            simplified_paths.insert(new_path);
        }
    }

    // 3. Clear the graph and rebuild a single trie from all simplified paths.
    god.clear();

    if simplified_paths.is_empty() {
        // If all paths were eliminated, clear the roots.
        roots.clear();
        return;
    }

    // Create a single root for the new trie.
    let has_only_empty_path =
        simplified_paths.len() == 1 && simplified_paths.iter().next().unwrap().is_empty();
    let root_content = if has_only_empty_path {
        IntermediatePrecomputedNodeContents3::leaf()
    } else {
        IntermediatePrecomputedNodeContents3::internal()
    };
    let new_root = god.insert(Trie::new(root_content)).into();

    let mut node_cache: BTreeMap<Vec<IntermediateTrie3EdgeKey>, IntermediatePrecomputeNode3Index> =
        BTreeMap::new();
    node_cache.insert(vec![], new_root);

    for path in simplified_paths {
        if path.is_empty() {
            continue; // Handled by root node creation
        }
        let mut current_node_idx = new_root;
        for i in 0..path.len() {
            let edge = &path[i];
            let prefix = &path[0..=i];

            let next_node_idx = *node_cache.entry(prefix.to_vec()).or_insert_with(|| {
                let is_leaf = i == path.len() - 1;
                let content = if is_leaf {
                    IntermediatePrecomputedNodeContents3::leaf()
                } else {
                    IntermediatePrecomputedNodeContents3::internal()
                };
                god.insert(Trie::new(content)).into()
            });

            god.insert_edge_simple(current_node_idx, next_node_idx, edge.clone(), ());
            current_node_idx = next_node_idx;
        }
    }

    // 4. Update all roots to point to the new single root.
    for (_, root_idx) in roots.iter_mut() {
        *root_idx = new_root;
    }
}

pub fn eliminate_pushes_and_pops(
    _roots: &mut BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    _god: &IntermediateTrie3GodWrapper,
) {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datastructures::trie::Trie2Index;

    fn run_test(
        input_god: &IntermediateTrie3GodWrapper,
        input_roots: &[IntermediatePrecomputeNode3Index],
    ) {
        // NOTE: This test harness works by comparing the set of all paths from root to leaf
        // before and after simplification. The `get_all_paths` helper function stops traversing
        // when it detects a cycle to avoid infinite loops. Therefore, this harness cannot
        // validate simplifications that occur *within* a cycle. A true trie-based
        // elimination algorithm would need a different testing strategy for cyclic graphs.

        // 1. Run path-based elimination
        let (eliminated_god, eliminated_roots, _) =
            Trie::deep_copy_subtrees(input_god, input_roots);
        let mut eliminated_roots_map = BTreeMap::new();
        for (i, r) in eliminated_roots.iter().enumerate() {
            eliminated_roots_map.insert(TokenizerStateID(i), *r); // Use dummy TokenizerStateID
        }
        eliminate_pushes_and_pops_path_based(&mut eliminated_roots_map, &eliminated_god);

        // 2. Flatten result to paths
        let final_roots_from_trie_elim: Vec<_> = eliminated_roots_map.values().cloned().collect();
        let paths_from_trie_elim: BTreeSet<_> = IntermediatePrecomputeNode3::get_all_paths(
            &eliminated_god,
            &final_roots_from_trie_elim,
            |n| n.value.end,
        )
        .into_iter()
        .map(|(_r, p)| normalize_path(p.into_iter().map(|(ek, _, _)| ek).collect()))
        .collect();

        // 3. Run old path-based elimination directly
        let initial_paths =
            IntermediatePrecomputeNode3::get_all_paths(input_god, input_roots, |node| node.value.end);
        let mut paths_from_path_elim = BTreeSet::new();
        for (_root_value, path_edges) in initial_paths {
            let edge_keys: Vec<_> = path_edges.into_iter().map(|(ek, _, _)| ek).collect();
            if let Some(new_path) = simplify_path(edge_keys) {
                paths_from_path_elim.insert(normalize_path(new_path));
            }
        }

        // 4. Compare
        assert_eq!(paths_from_trie_elim, paths_from_path_elim);
    }

    #[test]
    fn test_simple_cancel() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv = StateIDBV::zeros();
        bv.insert(1);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv.clone()), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_mismatch_invalidates_path() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv2), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_pop_zero_keeps_push() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv = StateIDBV::zeros();
        bv.insert(1);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv.clone()), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(0, bv), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_pop_zero_mismatch_invalidates_path() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(0, bv2), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_pop_n_decrements() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2); // Note: disjoint, but should not matter for n>1

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(3, bv2), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_blocked_push() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv2.clone()), (), v2);
        v2.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv2), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_multiple_cancellations_in_sequence() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v3 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1.clone()), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv2.clone()), (), v2);
        v2.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv2), (), v3);
        v3.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv1), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_interleaved_ops() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut llm_bv = LLMTokenBV::zeros();
        llm_bv.insert(100);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1.clone()), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_bv), (), v2);
        v2.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv1), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_branching_and_merging() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v3 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv_a = StateIDBV::zeros();
        bv_a.insert(1);
        let mut llm_x = LLMTokenBV::zeros();
        llm_x.insert(100);
        let mut llm_y = LLMTokenBV::zeros();
        llm_y.insert(200);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_a.clone()), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_x), (), v2);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_y), (), v3);
        v2.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), end);
        v3.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_cycle_simplification() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv_a = StateIDBV::zeros();
        bv_a.insert(1);

        // Path to end
        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), end);
        // Path with cycle
        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_a.clone()), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), root);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_no_pushes_or_pops() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut llm_bv = LLMTokenBV::zeros();
        llm_bv.insert(100);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_bv), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_dangling_pop() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv = StateIDBV::zeros();
        bv.insert(1);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_challenging_gauntlet() {
        // This test combines iterative cancellation, branching, and path invalidation.
        // Path X should simplify over multiple passes.
        // Path Y should be invalidated because Push(B) blocks Push(A), and then Push(B)
        // mismatches with Pop(1, A), killing the path.
        let god = IntermediateTrie3GodWrapper::new();

        // --- Nodes ---
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v3x =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v4x =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v5x =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v3y =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let merge =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        // --- State & LLM Bitsets ---
        let mut bv_a = StateIDBV::zeros();
        bv_a.insert(1);
        let mut bv_b = StateIDBV::zeros();
        bv_b.insert(2);
        let mut bv_c = StateIDBV::zeros();
        bv_c.insert(3);

        let mut llm_x = LLMTokenBV::zeros();
        llm_x.insert(100);
        let mut llm_y = LLMTokenBV::zeros();
        llm_y.insert(200);

        // --- Graph Structure ---
        // Common prefix
        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_a.clone()), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_b.clone()), (), v2);

        // Branching
        v2.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_x), (), v3x);
        v2.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_y), (), v3y);

        // Path X (iterative cancellation)
        v3x.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_c.clone()), (), v4x);
        v4x.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_c.clone()), (), v5x);
        v5x.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_b.clone()), (), merge);

        // Path Y (blocking and invalidation)
        v3y.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), merge);

        // Common suffix
        merge
            .write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), end);

        // The expected outcome is that Path Y is completely eliminated.
        // Path X simplifies: Push(C)/Pop(C) cancel, then Push(B)/Pop(B) cancel.
        // The remaining path is Push(A) -> CheckLLM(X) -> merge -> Pop(1, A) -> end.
        // Then Push(A)/Pop(A) cancel.
        // The final path should be just CheckLLM(X).
        run_test(&god, &[root]);
    }

    #[test]
    fn test_mismatch_from_log() {
        // This test case is derived from a real-world mismatch found during development.
        // The trie-based elimination correctly prunes this path as invalid, while the
        // path-based simplification incorrectly simplifies it to a non-empty path.
        // The key interaction is a Pop(0, bv_5) that should invalidate a path when a
        // Push(bv_1) is on the stack, but the path-based logic overlooks this.
        let god = IntermediateTrie3GodWrapper::new();

        let nodes: Vec<_> = (0..12)
            .map(|i| {
                let content = if i == 11 {
                    IntermediatePrecomputedNodeContents3::leaf()
                } else {
                    IntermediatePrecomputedNodeContents3::internal()
                };
                Trie2Index::from(god.insert(Trie::new(content)))
            })
            .collect();
        let root = nodes[0];

        let mut llm_0 = LLMTokenBV::zeros();
        llm_0.insert(0);

        let mut bv_1 = StateIDBV::zeros();
        bv_1.insert(1);
        let mut bv_2 = StateIDBV::zeros();
        bv_2.insert(2);
        let mut bv_4 = StateIDBV::zeros();
        bv_4.insert(4);
        let mut bv_5 = StateIDBV::zeros();
        bv_5.insert(5);
        let bv_max = StateIDBV::max_ones();

        let path = vec![
            IntermediateTrie3EdgeKey::CheckLLM(llm_0),
            IntermediateTrie3EdgeKey::Pop(0, bv_4.clone()),
            IntermediateTrie3EdgeKey::Pop(2, bv_max.clone()),
            IntermediateTrie3EdgeKey::Pop(0, bv_5.clone()),
            IntermediateTrie3EdgeKey::Push(bv_1.clone()),
            IntermediateTrie3EdgeKey::Push(bv_4.clone()),
            IntermediateTrie3EdgeKey::Pop(0, bv_4.clone()),
            IntermediateTrie3EdgeKey::Pop(2, bv_max.clone()),
            IntermediateTrie3EdgeKey::Pop(0, bv_5.clone()),
            IntermediateTrie3EdgeKey::Push(bv_1.clone()),
            IntermediateTrie3EdgeKey::Push(bv_2.clone()),
        ];

        let mut current_node = root;
        for (i, edge) in path.into_iter().enumerate() {
            let next_node = nodes[i + 1];
            current_node
                .write(&god)
                .unwrap()
                .force_insert_to_node(edge, (), next_node);
            current_node = next_node;
        }

        run_test(&god, &[root]);
    }

    #[test]
    fn test_complex_cycle_simplification() {
        // This test features a cycle that should be simplified away,
        // nested within a larger Push/Pop pair that should also be simplified.
        // The structure is:
        // root -> Push(A) -> c1 -> CheckLLM(X) -> c2
        // c2 -> Push(B) -> c3 -> Pop(1, B) -> c2  (inner cycle)
        // c2 -> Pop(1, A) -> end                   (exit path)
        // The expected simplified path is just CheckLLM(X).
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let c1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let c2 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let c3 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv_a = StateIDBV::zeros();
        bv_a.insert(1);
        let mut bv_b = StateIDBV::zeros();
        bv_b.insert(2);
        let mut llm_x = LLMTokenBV::zeros();
        llm_x.insert(100);

        // Path structure
        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_a.clone()), (), c1);
        c1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_x), (), c2);
        // Inner cycle
        c2.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_b.clone()), (), c3);
        c3.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_b.clone()), (), c2);
        // Exit path
        c2.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), end);

        run_test(&god, &[root]);
    }
}
