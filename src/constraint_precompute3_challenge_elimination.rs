// src/constraint_precompute3_challenge_elimination.rs
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use crate::constraint::{IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey, IntermediateTrie3GodWrapper, StateIDBV};
use crate::constraint::IntermediatePrecomputedNodeContents3;
use crate::datastructures::trie::Trie;
use crate::r#macro::is_debug_level_enabled;
use crate::tokenizer::TokenizerStateID;

/// Eliminates adjacent Push/Pop pairs from a stack of intermediate trie edge keys.
/// This is a core part of simplifying the precompute3 graph.
///
/// Pairing rules applied when a Push(A) looks to the nearest Pop(n, B) to its right
/// (stopping if another Push is encountered first):
/// - n == 0: If A intersects B, remove Pop(0, B) and keep Push(A). If disjoint, the stack is invalid (None).
/// - n == 1: If A intersects B, both cancel (epsilon). If disjoint, the stack is invalid (None).
/// - n > 1: Remove Push(A) and replace Pop with Pop(n - 1, B) unconditionally (no intersection check).
/// The scan repeats until no more changes can be made.
pub fn eliminate_pushes_and_pops_path_based(
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
                                // Keep Push(A); only Pop(0, B) is removed.
                                stack.insert(i, IntermediateTrie3EdgeKey::Push(push_states));
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

fn get_or_create_node(
    src_idx: IntermediatePrecomputeNode3Index,
    pending: Option<StateIDBV>,
    pair_cache: &mut BTreeMap<(IntermediatePrecomputeNode3Index, Option<StateIDBV>), IntermediatePrecomputeNode3Index>,
    work: &mut VecDeque<(IntermediatePrecomputeNode3Index, Option<StateIDBV>)>,
    god: &IntermediateTrie3GodWrapper,
    source: &IntermediateTrie3GodWrapper,
) -> IntermediatePrecomputeNode3Index {
    if let Some(existing) = pair_cache.get(&(src_idx, pending.clone())) {
        return *existing;
    }
    let src_guard = src_idx.read(source).expect("source read");
    let is_end = src_guard.value.end && pending.is_none();
    drop(src_guard);
    let node_val = if is_end {
        IntermediatePrecomputedNodeContents3::leaf()
    } else {
        IntermediatePrecomputedNodeContents3::internal()
    };
    let dest_idx = IntermediatePrecomputeNode3Index::new(god.insert(Trie::new(node_val)));
    pair_cache.insert((src_idx, pending.clone()), dest_idx);
    work.push_back((src_idx, pending));
    dest_idx
}

/// Placeholder for a future, more efficient trie-based implementation of push/pop elimination.
/// Currently, it uses the path-based approach internally by flattening the trie,
/// processing paths, and rebuilding the trie. This maintains the correct logic while
/// providing the desired API for a true trie-based replacement.
pub fn eliminate_pushes_and_pops(
    roots: &mut BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) {
    // We build a new graph directly in `god` by traversing a read-only snapshot (`source`)
    // of the original reachable subgraph, using a product over (node, pending_push).
    // pending_push is Option<StateIDBV>: None means no pending; Some(A) means a Push(A) is pending
    // and eligible to be canceled by the next Pop encountered before any subsequent Push.

    // 1) Snapshot the reachable subgraph from the provided roots.
    let mut sids: Vec<TokenizerStateID> = Vec::with_capacity(roots.len());
    let mut old_root_vec: Vec<IntermediatePrecomputeNode3Index> = Vec::with_capacity(roots.len());
    for (sid, idx) in roots.iter() {
        sids.push(*sid);
        old_root_vec.push(*idx);
    }
    let (source, source_roots, _map) = Trie::deep_copy_subtrees(god, &old_root_vec);

    // 2) Prepare destination arena (clear existing graph).
    crate::debug!(2, "Building trie-native elimination (product graph)...");
    god.clear();

    // 3) Memoization: (source_idx, pending) -> dest_idx
    let mut pair_cache: BTreeMap<(IntermediatePrecomputeNode3Index, Option<StateIDBV>), IntermediatePrecomputeNode3Index> = BTreeMap::new();
    let mut work: VecDeque<(IntermediatePrecomputeNode3Index, Option<StateIDBV>)> = VecDeque::new();

    // Helper to create or fetch a product-state node in the destination graph. (Refactored to get_or_create_node)
    let mut get_or_create = |src_idx: IntermediatePrecomputeNode3Index, pending: Option<StateIDBV>| -> IntermediatePrecomputeNode3Index {
        get_or_create_node(src_idx, pending, &mut pair_cache, &mut work, god, &source)
    };

    // 4) Create new roots at (source_root, None)
    let mut new_roots: BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index> = BTreeMap::new();
    for (sid, src_root) in sids.into_iter().zip(source_roots.into_iter()) {
        let new_root = get_or_create(src_root, None);
        new_roots.insert(sid, new_root);
    }

    // 5) BFS over product states
    while let Some((src_idx, pending)) = work.pop_front() {
        let dest_idx = *pair_cache.get(&(src_idx, pending.clone())).expect("dest exists");
        let src_guard = src_idx.read(&source).expect("source read");

        // If this source node is an end, and we still have a pending push, we must
        // commit it before termination: emit Push(A) to (src_idx, None).
        if src_guard.value.end {
            if let Some(a_pending) = pending.clone() {
                let none_state = get_or_create_node(src_idx, None, &mut pair_cache, &mut work, god, &source);
                god.insert_edge_simple(dest_idx, none_state, IntermediateTrie3EdgeKey::Push(a_pending.clone()), ());
            }
        }

        for (ek, dests) in src_guard.children().iter() {
            match ek {
                IntermediateTrie3EdgeKey::Push(bv_new) => {
                    // A new Push blocks any earlier pending Push.
                    // If we have a pending A, commit it now at this position:
                    //   (src, Some(A)) --Push(A)--> (src, None)
                    // then follow the actual Push(B) via NoOp to (dst, Some(B)).
                    if let Some(a_pending) = pending.clone() {
                        let none_state = get_or_create_node(src_idx, None, &mut pair_cache, &mut work, god, &source);
                        god.insert_edge_simple(dest_idx, none_state, IntermediateTrie3EdgeKey::Push(a_pending.clone()), ());
                        // Now process the Push(B) from the none_state
                        for (dst_src_idx, _) in dests.iter() {
                            let next_state = get_or_create_node(*dst_src_idx, Some(bv_new.clone()), &mut pair_cache, &mut work, god, &source);
                            god.insert_edge_simple(none_state, next_state, IntermediateTrie3EdgeKey::NoOp, ());
                        }
                    } else {
                        // No pending: defer emission by using NoOp to carry pending=Some(B)
                        for (dst_src_idx, _) in dests.iter() {
                            let next_state = get_or_create_node(*dst_src_idx, Some(bv_new.clone()), &mut pair_cache, &mut work, god, &source);
                            god.insert_edge_simple(dest_idx, next_state, IntermediateTrie3EdgeKey::NoOp, ());
                        }
                    }
                }
                IntermediateTrie3EdgeKey::Pop(n, pop_bv) => {
                    match &pending {
                        None => {
                            // No pending: forward Pop as-is.
                            for (dst_src_idx, _) in dests.iter() {
                                let next_state = get_or_create_node(*dst_src_idx, None, &mut pair_cache, &mut work, god, &source);
                                god.insert_edge_simple(dest_idx, next_state, IntermediateTrie3EdgeKey::Pop(*n, pop_bv.clone()), ());
                            }
                        }
                        Some(a_pending) => {
                            if *n == 0 {
                                if a_pending.is_disjoint(pop_bv) {
                                    // Invalid path; drop this branch (no edge emitted).
                                    continue;
                                }
                                // Intersect: remove Pop(0), keep pending A.
                                for (dst_src_idx, _) in dests.iter() {
                                    let next_state = get_or_create_node(*dst_src_idx, Some(a_pending.clone()), &mut pair_cache, &mut work, god, &source);
                                    god.insert_edge_simple(dest_idx, next_state, IntermediateTrie3EdgeKey::NoOp, ());
                                }
                            } else if *n == 1 {
                                if a_pending.is_disjoint(pop_bv) {
                                    // Invalid path; drop this branch.
                                    continue;
                                }
                                // Intersect: both cancel -> epsilon, clear pending.
                                for (dst_src_idx, _) in dests.iter() {
                                    let next_state = get_or_create_node(*dst_src_idx, None, &mut pair_cache, &mut work, god, &source);
                                    god.insert_edge_simple(dest_idx, next_state, IntermediateTrie3EdgeKey::NoOp, ());
                                }
                            } else {
                                // n > 1: drop pending Push unconditionally and decrement Pop.
                                for (dst_src_idx, _) in dests.iter() {
                                    let next_state = get_or_create_node(*dst_src_idx, None, &mut pair_cache, &mut work, god, &source);
                                    god.insert_edge_simple(
                                        dest_idx,
                                        next_state,
                                        IntermediateTrie3EdgeKey::Pop(n - 1, pop_bv.clone()),
                                        (),
                                    );
                                }
                            }
                        }
                    }
                }
                // Non-push/pop edges (e.g., CheckLLM, NoOp) just forward; pending unchanged.
                other => {
                    for (dst_src_idx, _) in dests.iter() {
                        let next_state = get_or_create_node(*dst_src_idx, pending.clone(), &mut pair_cache, &mut work, god, &source);
                        god.insert_edge_simple(dest_idx, next_state, other.clone(), ());
                    }
                }
            }
        }
    }

    // 6) Replace input roots with new roots (pending == None states).
    *roots = new_roots;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::LLMTokenBV;
    use crate::datastructures::trie::Trie2Index;

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

    fn run_test(
        input_god: &IntermediateTrie3GodWrapper,
        input_roots: &[IntermediatePrecomputeNode3Index],
    ) {
        // NOTE: This test harness works by comparing the set of all paths from root to leaf
        // before and after simplification. The `get_all_paths` helper function stops traversing
        // when it detects a cycle to avoid infinite loops. Therefore, this harness cannot
        // validate simplifications that occur *within* a cycle. A true trie-based
        // elimination algorithm would need a different testing strategy for cyclic graphs.

        // 1. Run new trie-based elimination (which currently calls the path-based one)
        let (eliminated_god, eliminated_roots, _) = Trie::deep_copy_subtrees(input_god, input_roots);
        let mut eliminated_roots_map = BTreeMap::new();
        for (i, r) in eliminated_roots.iter().enumerate() {
            eliminated_roots_map.insert(TokenizerStateID(i), *r); // Use dummy TokenizerStateID
        }
        eliminate_pushes_and_pops(&mut eliminated_roots_map, &eliminated_god);

        // 2. Flatten result to paths
        let final_roots_from_trie_elim: Vec<_> = eliminated_roots_map.values().cloned().collect();
        let paths_from_trie_elim: BTreeSet<_> = IntermediatePrecomputeNode3::get_all_paths(&eliminated_god, &final_roots_from_trie_elim, |n| n.value.end)
            .into_iter()
            .map(|(_r, p)| normalize_path(p.into_iter().map(|(ek, _, _)| ek).collect()))
            .collect();

        // 3. Run old path-based elimination directly
        let initial_paths = IntermediatePrecomputeNode3::get_all_paths(input_god, input_roots, |node| node.value.end);
        let mut paths_from_path_elim = BTreeSet::new();
        for (_root_value, path_edges) in initial_paths {
            let edge_keys: Vec<_> = path_edges.into_iter().map(|(ek, _, _)| ek).collect();
            if let Some(new_path) = eliminate_pushes_and_pops_path_based(edge_keys) {
                paths_from_path_elim.insert(normalize_path(new_path));
            }
        }

        // 4. Compare
        assert_eq!(paths_from_trie_elim, paths_from_path_elim);
    }

    #[test]
    fn test_simple_cancel() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv = StateIDBV::zeros();
        bv.insert(1);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv.clone()), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_mismatch_invalidates_path() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv2), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_pop_zero_keeps_push() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv = StateIDBV::zeros();
        bv.insert(1);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv.clone()), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(0, bv), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_pop_zero_mismatch_invalidates_path() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(0, bv2), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_pop_n_decrements() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2); // Note: disjoint, but should not matter for n>1

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(3, bv2), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_blocked_push() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv2.clone()), (), v2);
        v2.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv2), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_multiple_cancellations_in_sequence() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v3 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1.clone()), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv2.clone()), (), v2);
        v2.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv2), (), v3);
        v3.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv1), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_interleaved_ops() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut llm_bv = LLMTokenBV::zeros();
        llm_bv.insert(100);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1.clone()), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_bv), (), v2);
        v2.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv1), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_branching_and_merging() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v3 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv_a = StateIDBV::zeros();
        bv_a.insert(1);
        let mut llm_x = LLMTokenBV::zeros();
        llm_x.insert(100);
        let mut llm_y = LLMTokenBV::zeros();
        llm_y.insert(200);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_a.clone()), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_x), (), v2);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_y), (), v3);
        v2.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), end);
        v3.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_cycle_simplification() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv_a = StateIDBV::zeros();
        bv_a.insert(1);

        // Path to end
        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), end);
        // Path with cycle
        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_a.clone()), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), root);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_no_pushes_or_pops() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut llm_bv = LLMTokenBV::zeros();
        llm_bv.insert(100);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_bv), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_dangling_pop() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv = StateIDBV::zeros();
        bv.insert(1);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv), (), end);

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
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v3x = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v4x = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v5x = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v3y = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let merge = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

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
        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_a.clone()), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_b.clone()), (), v2);

        // Branching
        v2.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_x), (), v3x);
        v2.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_y), (), v3y);

        // Path X (iterative cancellation)
        v3x.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_c.clone()), (), v4x);
        v4x.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_c.clone()), (), v5x);
        v5x.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_b.clone()), (), merge);

        // Path Y (blocking and invalidation)
        v3y.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), merge);

        // Common suffix
        merge.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), end);

        // The expected outcome is that Path Y is completely eliminated.
        // Path X simplifies: Push(C)/Pop(C) cancel, then Push(B)/Pop(B) cancel.
        // The remaining path is Push(A) -> CheckLLM(X) -> merge -> Pop(1, A) -> end.
        // Then Push(A)/Pop(A) cancel.
        // The final path should be just CheckLLM(X).
        run_test(&god, &[root]);
    }
}
