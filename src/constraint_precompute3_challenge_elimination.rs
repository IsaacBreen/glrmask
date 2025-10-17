// src/constraint_precompute3_challenge_elimination.rs
use std::collections::{BTreeMap, BTreeSet};
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

/// Placeholder for a future, more efficient trie-based implementation of push/pop elimination.
/// Currently, it uses the path-based approach internally by flattening the trie,
/// processing paths, and rebuilding the trie. This maintains the correct logic while
/// providing the desired API for a true trie-based replacement.
pub fn eliminate_pushes_and_pops(
    roots: &mut BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) {
    // --- Path extraction, elimination, and trie rebuilding ---
    let mut paths_by_sid = BTreeMap::new();
    crate::debug!(2, "Processing paths for each intermediate trie3 state...");
    for (sid, root_idx) in &*roots {
        let paths = IntermediatePrecomputeNode3::get_all_paths(god, &[*root_idx], |node| node.value.end);
        let mut processed_paths_for_sid = BTreeSet::new();
        for (_root_value, path_edges) in paths {
            let edge_keys: Vec<_> = path_edges.into_iter().map(|(ek, _, _)| ek).collect();
            if let Some(new_path) = eliminate_pushes_and_pops_path_based(edge_keys) {
                processed_paths_for_sid.insert(new_path);
            }
        }
        paths_by_sid.insert(*sid, processed_paths_for_sid);
    }

    if is_debug_level_enabled(3) {
        println!("Processed paths after elimination:");
        for (sid, paths) in &paths_by_sid {
            println!("  SID {}:", sid.0);
            for path in paths {
                let edge_keys_str: Vec<_> = path.iter()
                    .filter(|ek| !matches!(ek, &IntermediateTrie3EdgeKey::NoOp))
                    .map(|ek| format!("{}", ek))
                    .collect();
                if !edge_keys_str.is_empty() {
                    println!("    [{}]", edge_keys_str.join(", "));
                }
            }
        }
    }

    // Rebuild the intermediate trie from the processed paths.
    crate::debug!(2, "Rebuilding intermediate trie3 from processed paths...");
    god.clear();
    let mut new_root_map: BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index> = BTreeMap::new();
    for (sid, _old_root) in &*roots {
        let new_root = IntermediatePrecomputeNode3Index::new(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        new_root_map.insert(*sid, new_root);
    }
    *roots = new_root_map;

    // Create a single shared leaf node.
    let leaf_node = IntermediatePrecomputeNode3Index::new(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

    for (sid, paths) in &paths_by_sid {
        let root_idx = roots.get(sid).unwrap();
        for path in paths {
            let mut current_idx = *root_idx;
            for edge_key in path {
                let next_idx = {
                    let guard = current_idx.read(god).unwrap();
                    if let Some(dest_map) = guard.children().get(edge_key) {
                        // Path processing should result in deterministic single-destination edges.
                        *dest_map.keys().next().unwrap()
                    } else {
                        drop(guard);
                        let new_node = Trie::new(IntermediatePrecomputedNodeContents3::internal());
                        let new_idx = IntermediatePrecomputeNode3Index::from(god.insert(new_node));
                        current_idx.write(god).unwrap().force_insert_to_node(edge_key.clone(), (), new_idx);
                        new_idx
                    }
                };
                current_idx = next_idx;
            }
            // After the path is built, connect the last node to the shared leaf.
            current_idx.write(god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), leaf_node);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datastructures::trie::Trie2Index;

    fn run_test(
        input_god: &IntermediateTrie3GodWrapper,
        input_roots: &[IntermediatePrecomputeNode3Index],
    ) {
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
            .map(|(_r, p)| {
                p.into_iter()
                    .map(|(ek, _, _)| ek)
                    .filter(|ek| !matches!(ek, IntermediateTrie3EdgeKey::NoOp))
                    .collect::<Vec<_>>()
            })
            .collect();

        // 3. Run old path-based elimination directly
        let initial_paths = IntermediatePrecomputeNode3::get_all_paths(input_god, input_roots, |node| node.value.end);
        let mut paths_from_path_elim = BTreeSet::new();
        for (_root_value, path_edges) in initial_paths {
            let edge_keys: Vec<_> = path_edges.into_iter().map(|(ek, _, _)| ek).collect();
            if let Some(new_path) = eliminate_pushes_and_pops_path_based(edge_keys) {
                let normalized_path: Vec<_> = new_path.into_iter()
                    .filter(|ek| !matches!(ek, IntermediateTrie3EdgeKey::NoOp))
                    .collect();
                paths_from_path_elim.insert(normalized_path);
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
}
