use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use profiler_macro::time_it;

use crate::constraint::{LLMTokenBV, PrecomputeNode3Index, PrecomputedNodeContents, StateIDBV};
use crate::datastructures::gss::{Acc, GSSInternal, GSSNode, GSSRoot, PruneAndTransformRecursiveMemo, StoredPrecomputeNode, StoredPrecomputeNodeIndex, StoredTrieGodWrapper};
use crate::datastructures::gss_pruning;
use crate::datastructures::trie::EdgeInserter;

pub(crate) fn merge_stored_trie_nodes(
    root_arc: &mut Arc<GSSNode>,
    memo: &mut PruneAndTransformRecursiveMemo,
    stored_trie_god: &StoredTrieGodWrapper,
) {
    let mut new_destinations = BTreeMap::new();

    let mut internal_closure = |internal: &GSSInternal| Some((internal.acc.clone(), true));
    let mut root_closure = |root: &GSSRoot| -> Option<Arc<Acc>> {
        if !root.acc.stored_trie_nodes.iter().any(
            // TODO: can this condition be relaxed to a subset or something?
            |n| n.as_arc().read(stored_trie_god).expect("poison").value.live_tokens != root.acc.llm_tokens_union
        ) {
            return Some(root.acc.clone());
        }
        let mut new_acc = (*root.acc).clone();
        // Create a single new destination for this merge operation.
        let new_destination = new_destinations.entry((new_acc.stored_trie_nodes.clone(), root.acc.llm_tokens_union.clone()))
            .or_insert_with(|| StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal()))))
            .clone();
        let edge_key = (0, new_acc.llm_tokens_union.clone());
        let edge_value = StateIDBV::max_ones();
        let tokens_for_edge = new_acc.llm_tokens_union.clone();

        for source_wrapper in &new_acc.stored_trie_nodes {
            let source_arc = source_wrapper.as_arc().clone();

            let inserter = EdgeInserter::new(
                &stored_trie_god,
                source_arc,
                edge_key.clone(),
                edge_value.clone(),
                |e, n| *e |= n,
                |node_value, _edge_value| node_value.live_tokens |= &tokens_for_edge,
                |_, _| {}, // Unconditional insertion
            );
            // Insert a strong edge to the new shared destination.
            inserter.try_destination(new_destination.clone()).expect("Cycle detected when merging stored_trie nodes; this should be impossible.");
        }

        // Update the live tokens on the new destination node.
        new_destination.write(stored_trie_god).expect("poison").value.live_tokens |= &tokens_for_edge;

        // The acc now points only to this new merged destination.
        new_acc.stored_trie_nodes = BTreeSet::from([new_destination]);
        Some(Arc::new(new_acc))
    };
    if let Some(new_root) = gss_pruning::prune_and_transform_recursive(root_arc, &mut internal_closure, &mut root_closure, memo) {
        *root_arc = new_root;
    } else {
        unreachable!();
    }
}

/// Recursively traverses a GSS, and for each node (both internal and root) that has
/// `stored_trie_nodes`, it:
/// 1. Gets a new destination trie node from `destination_provider`.
/// 2. Adds edges from all of the node's `stored_trie_nodes` to this new destination.
/// 3. Replaces the node's `stored_trie_nodes` with a set containing only the new destination.
///
/// This is used to apply shared constraints (like a StateID bitvector) across an entire GSS branch
/// by adding a filtered edge to the underlying precomputation trie.
#[time_it]
pub(crate) fn deep_add_precompute_trie_edges(
    root_arc: &mut Arc<GSSNode>,
    god: &StoredTrieGodWrapper,
    edge_key: &(usize, LLMTokenBV),
    edge_value: &StateIDBV,
    tokens_for_update: &LLMTokenBV,
    destination_provider: &mut impl FnMut() -> PrecomputeNode3Index,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    if let Some(new_root) = deep_add_precompute_trie_edges_recursive(
        root_arc,
        god,
        edge_key,
        edge_value,
        tokens_for_update,
        destination_provider,
        memo,
    ) {
        *root_arc = new_root;
    } else {
        // This function should not prune the root unless it becomes completely empty.
        // If all paths are pruned, it becomes a fresh root.
        *root_arc = Arc::new(GSSNode::new_fresh());
    }
}

fn deep_add_precompute_trie_edges_recursive(
    node_arc: &Arc<GSSNode>,
    god: &StoredTrieGodWrapper,
    edge_key: &(usize, LLMTokenBV),
    edge_value: &StateIDBV,
    tokens_for_update: &LLMTokenBV,
    destination_provider: &mut impl FnMut() -> StoredPrecomputeNodeIndex,
    memo: &mut PruneAndTransformRecursiveMemo,
) -> Option<Arc<GSSNode>> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(cached_result) = memo.get(&node_ptr) {
        return cached_result.clone();
    }

    // 1. Process the current node's Acc
    let local_acc = node_arc.local_acc();
    let (new_acc_arc, acc_changed) = if !local_acc.stored_trie_nodes().is_empty() {
        let destination = destination_provider();

        for source_wrapper in local_acc.stored_trie_nodes() {
            let source_arc = source_wrapper.as_arc().clone();

            let inserter = EdgeInserter::new(
                god,
                source_arc,
                edge_key.clone(),
                edge_value.clone(),
                |e, n| *e |= n,
                |node_value, _edge_value| node_value.live_tokens |= tokens_for_update,
                |_, _| {}, // Unconditional insertion
            );
            inserter.try_destination(destination.clone()).expect("Cycle detected when adding precompute trie edges");
        }

        destination.write(god).expect("poison").value.live_tokens |= tokens_for_update;

        let mut new_acc = (*local_acc).clone();
        *new_acc.stored_trie_nodes_mut() = BTreeSet::from([destination]);
        (Arc::new(new_acc), true)
    } else {
        (local_acc.clone(), false)
    };

    // 2. Recurse into children (for internal nodes) and rebuild the node
    let result = match node_arc.as_ref() {
        GSSNode::Root(_) => {
            if acc_changed {
                Some(Arc::new(GSSNode::new((*new_acc_arc).clone())))
            } else {
                Some(node_arc.clone())
            }
        }
        GSSNode::Internal(internal) => {
            let mut any_child_changed = false;
            let mut new_predecessors_map = BTreeMap::new();

            for (edge_val, preds_by_depth) in &internal.predecessors {
                let mut new_preds_by_depth = BTreeMap::new();
                for (dest_key, pred_vec) in preds_by_depth {
                    let mut new_vec: Vec<Arc<GSSNode>> = Vec::with_capacity(pred_vec.len());
                    for pred_arc in pred_vec {
                        if let Some(new_pred_arc) = deep_add_precompute_trie_edges_recursive(
                            pred_arc, god, edge_key, edge_value, tokens_for_update, destination_provider, memo
                        ) {
                            if !Arc::ptr_eq(&new_pred_arc, pred_arc) {
                                any_child_changed = true;
                            }
                            new_vec.push(new_pred_arc);
                        } else {
                            any_child_changed = true; // Child was pruned
                        }
                    }
                    if !new_vec.is_empty() {
                        new_preds_by_depth.insert(*dest_key, new_vec);
                    }
                }
                if !new_preds_by_depth.is_empty() {
                    new_predecessors_map.insert(edge_val.clone(), new_preds_by_depth);
                }
            }

            if new_predecessors_map.is_empty() {
                // All children pruned, so this node becomes a root with its (possibly new) acc.
                Some(Arc::new(GSSNode::new((*new_acc_arc).clone())))
            } else if !any_child_changed && !acc_changed {
                Some(node_arc.clone())
            } else {
                let transformed_node = GSSNode::new_with_map(new_acc_arc, new_predecessors_map);
                Some(Arc::new(transformed_node))
            }
        }
    };

    memo.insert(node_ptr, result.clone());
    result
}
