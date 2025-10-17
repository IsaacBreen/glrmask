// src/constraint_precompute3_challenge_elimination.rs
use std::time::{Duration, Instant};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use crate::constraint::{IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey, IntermediateTrie3GodWrapper, StateIDBV};
use crate::constraint::IntermediatePrecomputedNodeContents3;
use crate::datastructures::trie::Trie;
use crate::r#macro::is_debug_level_enabled;
use crate::tokenizer::TokenizerStateID;

/// If true, runs a much slower path-based elimination algorithm and compares its output
/// against the trie-based algorithm. If they mismatch, it attempts to find a minimal
/// failing subgraph and prints it.
const DEBUG_CHALLENGE_ELIMINATION: bool = true;
/// Eliminates adjacent Push/Pop pairs from a stack of intermediate trie edge keys.
/// This is a core part of simplifying the precompute3 graph.
///
/// Pairing rules applied when a Push(A) looks to the nearest Pop(n, B) to its right
/// (stopping if another Push is encountered first):
/// - n == 0: If A intersects B, remove Pop(0, B) and keep Push(A). If disjoint, the stack is invalid (None).
/// - n == 1: If A intersects B, both cancel (epsilon). If disjoint, the stack is invalid (None).
/// - n > 1: Remove Push(A) and replace Pop with Pop(n - 1, B) unconditionally (no intersection check).
/// The scan repeats until no more changes can be made.
fn simplify_path_edges(
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

pub fn eliminate_pushes_and_pops_path_based(
    roots: &mut BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) {
    // 1. Get all paths for each root and simplify them.
    let mut simplified_paths_by_root: BTreeMap<TokenizerStateID, Vec<Vec<IntermediateTrie3EdgeKey>>> = BTreeMap::new();
    for (sid, root_idx) in roots.iter() {
        let paths_for_root = IntermediatePrecomputeNode3::get_all_paths(god, &[*root_idx], |node| node.value.end);
        let mut simplified_paths = Vec::new();
        for (_, path_edges) in paths_for_root {
            let edge_keys: Vec<_> = path_edges.into_iter().map(|(ek, _, _)| ek).collect();
            if let Some(new_path) = simplify_path_edges(edge_keys) {
                simplified_paths.push(new_path);
            }
        }
        simplified_paths_by_root.insert(*sid, simplified_paths);
    }

    // 2. Clear the graph and rebuild from simplified paths.
    god.clear();
    let mut new_roots = BTreeMap::new();

    for (sid, paths) in simplified_paths_by_root {
        let new_root = IntermediatePrecomputeNode3Index::new(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        new_roots.insert(sid, new_root);

        for path in paths {
            if path.is_empty() {
                // This path simplified to epsilon. The root is now a leaf.
                new_root.write(god).unwrap().value.end = true;
                continue;
            }

            let mut current_node_idx = new_root;
            for (i, edge) in path.iter().enumerate() {
                let is_last_edge = i == path.len() - 1;

                let mut write_guard = current_node_idx.write(god).unwrap();
                let maybe_dest = write_guard.children().get(edge).and_then(|dests| dests.get(0));

                let next_node_idx = if let Some((dest_idx, _)) = maybe_dest {
                    // Path prefix already exists, just move to the next node.
                    *dest_idx
                } else {
                    // Create new node and edge.
                    let new_node_val = if is_last_edge {
                        IntermediatePrecomputedNodeContents3::leaf()
                    } else {
                        IntermediatePrecomputedNodeContents3::internal()
                    };
                    let new_node_idx = IntermediatePrecomputeNode3Index::new(god.insert(Trie::new(new_node_val)));
                    // Drop guard before getting a new write guard.
                    drop(write_guard);
                    current_node_idx.write(god).unwrap().force_insert_to_node(edge.clone(), (), new_node_idx);
                    new_node_idx
                };
                current_node_idx = next_node_idx;
            }
        }
    }
    *roots = new_roots;
}

fn eliminate_pushes_and_pops_trie_based(
    roots: &mut BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) {
    // pending_stack is a Vec<StateIDBV> that behaves like a stack of Pushes seen so far but not yet
    // materialized or canceled. This enables nested cancellations like:
    //   Push(A), Push(B), Pop(1, B), Pop(1, A)  -> epsilon
    //
    // The transition rules mirror eliminate_pushes_and_pops_path_based but operate locally:
    // - Push(X): push X onto the stack; propagate via NoOp.
    // - Pop(0, B): if top intersects B, drop Pop (NoOp), keep stack; else branch dies.
    // - Pop(1, B): if top intersects B, pop stack and NoOp; else branch dies.
    // - Pop(n>1, B): unconditionally pop stack and forward Pop(n-1, B).
    // - Other edges: forward unchanged with the same stack.
    // When at an end node, flush remaining stack by emitting Push edges in original order.

    // 1) Snapshot the reachable subgraph from the provided roots.
    let mut sids: Vec<TokenizerStateID> = Vec::with_capacity(roots.len());
    let mut old_root_vec: Vec<IntermediatePrecomputeNode3Index> = Vec::with_capacity(roots.len());
    for (sid, idx) in roots.iter() {
        sids.push(*sid);
        old_root_vec.push(*idx);
    }
    let (source, source_roots, _map) = Trie::deep_copy_subtrees(god, &old_root_vec);

    // 2) Prepare destination arena (clear existing graph).
    god.clear();

    // 3) Memoization: (source_idx, pending_stack) -> dest_idx
    let mut pair_cache: BTreeMap<(IntermediatePrecomputeNode3Index, Vec<StateIDBV>), IntermediatePrecomputeNode3Index> = BTreeMap::new();
    let mut work: VecDeque<(IntermediatePrecomputeNode3Index, Vec<StateIDBV>)> = VecDeque::new();

    macro_rules! get_or_create {
        ($src_idx:expr, $stack:expr) => {
            {
                let key = ($src_idx, $stack.clone());
                if let Some(&existing) = pair_cache.get(&key) {
                    existing
                } else {
                    let src_guard = key.0.read(&source).expect("source read");
                    let is_end = src_guard.value.end && key.1.is_empty();
                    drop(src_guard);
                    let node_val = if is_end {
                        IntermediatePrecomputedNodeContents3::leaf()
                    } else {
                        IntermediatePrecomputedNodeContents3::internal()
                    };
                    let dest_idx = IntermediatePrecomputeNode3Index::new(god.insert(Trie::new(node_val)));
                    pair_cache.insert(key.clone(), dest_idx);
                    work.push_back(key);
                    dest_idx
                }
            }
        };
    }

    // 4) Create new roots at (source_root, empty stack)
    let mut new_roots: BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index> = BTreeMap::new();
    for (sid, src_root) in sids.into_iter().zip(source_roots.into_iter()) {
        let new_root = get_or_create!(src_root, Vec::<StateIDBV>::new());
        new_roots.insert(sid, new_root);
    }

    // 5) BFS over product states
    while let Some((src_idx, stack)) = work.pop_front() {
        let dest_idx = *pair_cache.get(&(src_idx, stack.clone())).expect("dest exists");
        let src_guard = src_idx.read(&source).expect("source read");

        // If this source node is an end, flush the entire pending stack in order:
        // (src, [A, B, C]) --Push(A)--> (src, [B, C]) --Push(B)--> (src, [C]) --Push(C)--> (src, [])
        if src_guard.value.end && !stack.is_empty() {
            let mut from_idx = dest_idx;
            // Emit pushes in original encounter order (bottom-to-top).
            for i in 0..stack.len() {
                let label = IntermediateTrie3EdgeKey::Push(stack[i].clone());
                let next_stack = stack[i+1..].to_vec();
                let to_idx = get_or_create!(src_idx, next_stack);
                god.insert_edge_simple(from_idx, to_idx, label, ());
                from_idx = to_idx;
            }
        }

        for (ek, dests) in src_guard.children().iter() {
            match ek {
                IntermediateTrie3EdgeKey::Push(bv_new) => {
                    // Push: defer emission by pushing onto the stack and carrying via NoOp.
                    let mut new_stack = stack.clone();
                    new_stack.push(bv_new.clone());
                    for (dst_src_idx, _) in dests.iter() {
                        let next_state = get_or_create!(*dst_src_idx, new_stack.clone());
                        god.insert_edge_simple(dest_idx, next_state, IntermediateTrie3EdgeKey::NoOp, ());
                    }
                }
                IntermediateTrie3EdgeKey::Pop(n, pop_bv) => {
                    if stack.is_empty() {
                        // No pending: forward Pop as-is.
                        for (dst_src_idx, _) in dests.iter() {
                            let next_state = get_or_create!(*dst_src_idx, Vec::<StateIDBV>::new());
                            god.insert_edge_simple(dest_idx, next_state, IntermediateTrie3EdgeKey::Pop(*n, pop_bv.clone()), ());
                        }
                    } else {
                        let top = stack.last().expect("non-empty").clone();
                        if *n == 0 {
                            if top.is_disjoint(pop_bv) {
                                // Invalid path; drop this branch (no edge emitted).
                                continue;
                            }
                            // Intersect: remove Pop(0), keep stack unchanged.
                            for (dst_src_idx, _) in dests.iter() {
                                let next_state = get_or_create!(*dst_src_idx, stack.clone());
                                god.insert_edge_simple(dest_idx, next_state, IntermediateTrie3EdgeKey::NoOp, ());
                            }
                        } else if *n == 1 {
                            if top.is_disjoint(pop_bv) {
                                // Invalid path; drop this branch.
                                continue;
                            }
                            // Intersect: both cancel -> epsilon, pop top of stack.
                            let mut new_stack = stack.clone();
                            new_stack.pop();
                            for (dst_src_idx, _) in dests.iter() {
                                let next_state = get_or_create!(*dst_src_idx, new_stack.clone());
                                god.insert_edge_simple(dest_idx, next_state, IntermediateTrie3EdgeKey::NoOp, ());
                            }
                        } else {
                            // n > 1: drop top Push unconditionally and decrement Pop.
                            let mut new_stack = stack.clone();
                            new_stack.pop();
                            for (dst_src_idx, _) in dests.iter() {
                                let next_state = get_or_create!(*dst_src_idx, new_stack.clone());
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
                // Non-push/pop edges (e.g., CheckLLM, NoOp) just forward; pending unchanged.
                other => {
                    for (dst_src_idx, _) in dests.iter() {
                        let next_state = get_or_create!(*dst_src_idx, stack.clone());
                        god.insert_edge_simple(dest_idx, next_state, other.clone(), ());
                    }
                }
            }
        }
    }

    // 6) Replace input roots with new roots (pending_stack is empty).
    *roots = new_roots;
}

/// Normalizes a path for comparison purposes.
/// - Removes NoOp edges.
/// - Collects all CheckLLM bitvectors, intersects them, and prepends a single CheckLLM.
fn normalize_path_for_test(path: Vec<IntermediateTrie3EdgeKey>) -> Vec<IntermediateTrie3EdgeKey> {
    let mut combined_llm_bv = crate::constraint::LLMTokenBV::max_ones();
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

fn get_normalized_paths(
    god: &IntermediateTrie3GodWrapper,
    roots: &[IntermediatePrecomputeNode3Index],
) -> BTreeSet<Vec<IntermediateTrie3EdgeKey>> {
    IntermediatePrecomputeNode3::get_all_paths(god, roots, |n| n.value.end)
        .into_iter()
        .map(|(_r, p)| normalize_path_for_test(p.into_iter().map(|(ek, _, _)| ek).collect()))
        .collect()
}

fn get_all_edges(god: &IntermediateTrie3GodWrapper, roots: &[IntermediatePrecomputeNode3Index]) -> Vec<(IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey, IntermediatePrecomputeNode3Index)> {
    let mut edges = Vec::new();
    let mut visited = BTreeSet::new();
    let mut work: VecDeque<_> = roots.to_vec().into();

    while let Some(idx) = work.pop_front() {
        if !visited.insert(idx) {
            continue;
        }
        let guard = idx.read(god).unwrap();
        for (ek, dests) in guard.children().iter() {
            for (dest_idx, _) in dests.iter() {
                edges.push((idx, ek.clone(), *dest_idx));
                work.push_back(*dest_idx);
            }
        }
    }
    edges
}

fn remove_edge(god: &IntermediateTrie3GodWrapper, from: IntermediatePrecomputeNode3Index, ek: &IntermediateTrie3EdgeKey, to: IntermediatePrecomputeNode3Index) {
    let mut guard = from.write(god).unwrap();
    // NOTE: This assumes a `children_mut()` method or similar exists on the Trie write guard.
    // Based on `force_insert_to_node`, mutable access to children is expected to be possible.
    // If the Trie API is different, this part will need adjustment.
    if let Some(dests) = guard.children_mut().get_mut(ek) {
        dests.retain(|(dest_idx, _)| *dest_idx != to);
        if dests.is_empty() {
            guard.children_mut().remove(ek);
        }
    }
}

fn check_consistency(
    god: &IntermediateTrie3GodWrapper,
    roots_map: &BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
) -> bool {
    let roots_vec: Vec<_> = roots_map.values().cloned().collect();

    // Path-based
    let (path_god, path_roots_vec, _) = Trie::deep_copy_subtrees(god, &roots_vec);
    let mut path_roots_map_copy: BTreeMap<_,_> = roots_map.keys().cloned().zip(path_roots_vec.into_iter()).collect();
    eliminate_pushes_and_pops_path_based(&mut path_roots_map_copy, &path_god);
    let expected_paths = get_normalized_paths(&path_god, &path_roots_map_copy.values().cloned().collect::<Vec<_>>());

    // Trie-based
    let (trie_god, trie_roots_vec, _) = Trie::deep_copy_subtrees(god, &roots_vec);
    let mut trie_roots_map_copy: BTreeMap<_,_> = roots_map.keys().cloned().zip(trie_roots_vec.into_iter()).collect();
    eliminate_pushes_and_pops_trie_based(&mut trie_roots_map_copy, &trie_god);
    let actual_paths = get_normalized_paths(&trie_god, &trie_roots_map_copy.values().cloned().collect::<Vec<_>>());

    expected_paths == actual_paths
}

pub fn eliminate_pushes_and_pops(
    roots: &mut BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) {
    if !DEBUG_CHALLENGE_ELIMINATION {
        eliminate_pushes_and_pops_trie_based(roots, god);
        return;
    }

    // --- PRE-COMPUTATION (for debugging) ---
    // 1. Get expected paths from the slower, path-based implementation.
    let (path_god, path_roots_vec, _) = Trie::deep_copy_subtrees(god, &roots.values().cloned().collect::<Vec<_>>());
    let mut path_roots_map: BTreeMap<_, _> = roots.keys().cloned().zip(path_roots_vec.into_iter()).collect();
    eliminate_pushes_and_pops_path_based(&mut path_roots_map, &path_god);
    let final_roots_from_path_elim: Vec<_> = path_roots_map.values().cloned().collect();
    let expected_paths = get_normalized_paths(&path_god, &final_roots_from_path_elim);

    // 2. Keep a pristine copy of the original graph for refinement on mismatch.
    let (original_god, original_roots_vec, _) = Trie::deep_copy_subtrees(god, &roots.values().cloned().collect::<Vec<_>>());
    let original_roots_map: BTreeMap<_, _> = roots.keys().cloned().zip(original_roots_vec.into_iter()).collect();

    // --- MAIN IMPLEMENTATION (TRIE-BASED) ---
    eliminate_pushes_and_pops_trie_based(roots, god);

    // --- POST-COMPUTATION (DEBUG CHECK) ---
    let actual_paths = get_normalized_paths(god, &roots.values().cloned().collect::<Vec<_>>());

    if actual_paths != expected_paths {
        println!("\n!!! MISMATCH DETECTED in eliminate_pushes_and_pops !!!");
        println!("Expected (path-based) != Actual (trie-based)");

        let mut min_god = original_god;
        let mut min_roots_map = original_roots_map;
        let mut changed = true;

        let start_time = Instant::now();
        let timeout = Duration::from_secs(1);
        let mut last_progress_time = Instant::now();
        let no_progress_timeout = Duration::from_millis(200);

        while changed && start_time.elapsed() < timeout && last_progress_time.elapsed() < no_progress_timeout {
            changed = false;
            let min_roots_vec: Vec<_> = min_roots_map.values().cloned().collect();
            let edges = get_all_edges(&min_god, &min_roots_vec);

            for (from, ek, to) in edges {
                if start_time.elapsed() >= timeout { break; }

                let (smaller_god, smaller_roots_vec, map) = Trie::deep_copy_subtrees(&min_god, &min_roots_vec);
                let mut smaller_roots_map: BTreeMap<_, _> = min_roots_map.keys().cloned().zip(smaller_roots_vec.into_iter()).collect();

                let new_from = *map.get(&from).unwrap();
                let new_to = *map.get(&to).unwrap();

                remove_edge(&smaller_god, new_from, &ek, new_to);

                if !check_consistency(&smaller_god, &smaller_roots_map) {
                    min_god = smaller_god;
                    min_roots_map = smaller_roots_map;
                    changed = true;
                    last_progress_time = Instant::now();
                    break; // Restart with the smaller graph
                }
            }
        }

        println!("\n--- Minimal Failing Graph Found (or timeout reached) ---");
        let final_failing_roots: Vec<_> = min_roots_map.values().cloned().collect();
        let paths_before = IntermediatePrecomputeNode3::get_all_paths(&min_god, &final_failing_roots, |n| n.value.end);
        println!("\nPaths in minimal failing graph (BEFORE simplification):");
        for (root, path) in &paths_before {
            println!("  Root {:?}: {:?}", root.idx, path.iter().map(|(ek,_,_)| ek).collect::<Vec<_>>());
        }

        let (path_god_min, path_roots_vec_min, _) = Trie::deep_copy_subtrees(&min_god, &final_failing_roots);
        let mut path_roots_map_min: BTreeMap<_,_> = min_roots_map.keys().cloned().zip(path_roots_vec_min.into_iter()).collect();
        eliminate_pushes_and_pops_path_based(&mut path_roots_map_min, &path_god_min);
        let expected_paths_min = get_normalized_paths(&path_god_min, &path_roots_map_min.values().cloned().collect::<Vec<_>>());
        println!("\nMinimal failing graph, EXPECTED paths (path-based): {:#?}", expected_paths_min);

        let (trie_god_min, trie_roots_vec_min, _) = Trie::deep_copy_subtrees(&min_god, &final_failing_roots);
        let mut trie_roots_map_min: BTreeMap<_,_> = min_roots_map.keys().cloned().zip(trie_roots_vec_min.into_iter()).collect();
        eliminate_pushes_and_pops_trie_based(&mut trie_roots_map_min, &trie_god_min);
        let actual_paths_min = get_normalized_paths(&trie_god_min, &trie_roots_map_min.values().cloned().collect::<Vec<_>>());
        println!("\nMinimal failing graph, ACTUAL paths (trie-based): {:#?}", actual_paths_min);
        println!("\n--- End of Mismatch Report ---\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::LLMTokenBV;
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

        // 1. Run new trie-based elimination
        let (trie_god, trie_roots, _) = Trie::deep_copy_subtrees(input_god, input_roots);
        let mut trie_roots_map = BTreeMap::new();
        for (i, r) in trie_roots.iter().enumerate() {
            trie_roots_map.insert(TokenizerStateID(i), *r); // Use dummy TokenizerStateID
        }
        eliminate_pushes_and_pops(&mut trie_roots_map, &trie_god);

        // 2. Flatten trie-based result to paths
        let final_roots_from_trie_elim: Vec<_> = trie_roots_map.values().cloned().collect();
        let paths_from_trie_elim = get_normalized_paths(&trie_god, &final_roots_from_trie_elim);

        // 3. Run path-based elimination
        let (path_god, path_roots, _) = Trie::deep_copy_subtrees(input_god, input_roots);
        let mut path_roots_map = BTreeMap::new();
        for (i, r) in path_roots.iter().enumerate() {
            path_roots_map.insert(TokenizerStateID(i), *r);
        }
        eliminate_pushes_and_pops_path_based(&mut path_roots_map, &path_god);

        // 4. Flatten path-based result to paths
        let final_roots_from_path_elim: Vec<_> = path_roots_map.values().cloned().collect();
        let paths_from_path_elim = get_normalized_paths(&path_god, &final_roots_from_path_elim);

        // 5. Compare
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
