use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::constraint::{LLMTokenBV, TerminalBV};
use crate::datastructures::gss::{Acc, GSSInternal, GSSNode, GSSRoot};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::hybrid_l2_bitset::HybridL2Bitset;
use crate::tokenizer::TokenizerStateID;
use crate::types::TerminalID;

pub(crate) type PruneAndTransformRecursiveMemo = HashMap<*const GSSNode, Option<Arc<GSSNode>>>;

/// Prunes and/or transforms a GSS by:
/// - Invoking `internal_closure` on internal nodes to decide if they should be pruned entirely.
/// - Invoking `root_closure` on root nodes to determine the replacement Acc or prune the root.
///
/// Note:
/// - There is no early-continue/stop: recursion always traverses into children of internal nodes
///   unless `internal_closure` prunes that node.
/// - Internal nodes never hold Acc; only roots do.
pub(crate) fn prune_and_transform_recursive(
    node_arc: &Arc<GSSNode>,
    internal_closure: &mut impl FnMut(&GSSInternal) -> Option<(Arc<Acc>, bool)>,
    root_closure: &mut impl FnMut(&GSSRoot) -> Option<Arc<Acc>>,
    memo: &mut PruneAndTransformRecursiveMemo,
) -> Option<Arc<GSSNode>> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(cached_result) = memo.get(&node_ptr) {
        return cached_result.clone();
    }
    let result = match node_arc.as_ref() {
        GSSNode::Root(root) => {
            match root_closure(root) {
                None => None, // Prune
                Some(new_acc_arc) => {
                    if Arc::ptr_eq(&new_acc_arc, root.acc()) {
                        // No change
                        Some(node_arc.clone())
                    } else {
                        // Acc changed, create new root node
                        let new_node = GSSNode::new((*new_acc_arc).clone());
                        Some(Arc::new(new_node))
                    }
                }
            }
        }
        GSSNode::Internal(internal) => {
            match internal_closure(internal) {
                None => { // Prune
                    memo.insert(node_ptr, None);
                    return None;
                }
                Some((new_acc_arc, recurse)) => {
                    let acc_changed = !Arc::ptr_eq(&new_acc_arc, internal.acc());

                    if !recurse {
                        let result = if acc_changed {
                            // Acc changed, but no recursion. Rebuild node with old predecessors.
                            let new_node = GSSNode::new_with_map(new_acc_arc, internal.predecessors().clone());
                            Some(Arc::new(new_node))
                        } else {
                            // No change at all.
                            Some(node_arc.clone())
                        };
                        memo.insert(node_ptr, result.clone());
                        return result;
                    }

                    // Recurse into children.
                    let mut any_child_changed = false;
                    let mut new_predecessors_map = BTreeMap::new();

                    for (edge_val, preds_by_depth) in internal.predecessors() {
                        let mut new_preds_by_depth = BTreeMap::new();
                        for (dest_key, pred_vec) in preds_by_depth {
                            let mut new_vec: Vec<Arc<GSSNode>> = Vec::new();
                            for pred_arc in pred_vec {
                                match prune_and_transform_recursive(pred_arc, internal_closure, root_closure, memo) {
                                    Some(new_pred_arc) => {
                                        if !Arc::ptr_eq(&new_pred_arc, pred_arc) {
                                            any_child_changed = true;
                                        }
                                        new_vec.push(new_pred_arc);
                                    }
                                    None => {
                                        // Child was pruned.
                                        any_child_changed = true;
                                    }
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
                        // All children pruned, so prune this node.
                        None
                    } else if !any_child_changed && !acc_changed {
                        // No change in children or acc, so no change in this node.
                        Some(node_arc.clone())
                    } else {
                        // Children or acc changed, create new internal node.
                        let transformed_node = GSSNode::new_with_map(new_acc_arc, new_predecessors_map);
                        Some(Arc::new(transformed_node))
                    }
                }
            }
        }
    };

    memo.insert(node_ptr, result.clone());
    result
}

pub fn allow_only_llm_tokens_and_prune(
    root_arc: &mut Arc<GSSNode>,
    allowed_tokens: &LLMTokenBV,
) {
    let mut memo = HashMap::new();
    allow_only_llm_tokens_and_prune_arc(root_arc, allowed_tokens, &mut memo);
}

pub(crate) fn allow_only_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    allowed_tokens: &LLMTokenBV,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let node_ptr = Arc::as_ptr(root_arc);
    if let Some(cached) = memo.get(&node_ptr) {
        *root_arc = cached.clone().unwrap_or_else(|| Arc::new(GSSNode::new_dead()));
        return;
    }
    let new_arc_opt = match root_arc.as_ref() {
        GSSNode::Root(root) => {
            let mut new_acc = (**root.acc()).clone();
            new_acc.llm_tokens_union &= allowed_tokens;
            if new_acc.llm_tokens_union.is_empty() {
                None
            } else {
                Some(Arc::new(GSSNode::new(new_acc)))
            }
        }
        GSSNode::Internal(internal) => {
            let mut new_acc = (**internal.acc()).clone();
            new_acc.llm_tokens_union &= allowed_tokens;
            if new_acc.llm_tokens_union.is_empty() {
                None
            } else {
                Some(Arc::new(GSSNode::new_with_map(
                    Arc::new(new_acc),
                    internal.predecessors().clone(),
                )))
            }
        }
    };
    memo.insert(node_ptr, new_arc_opt.clone());
    *root_arc = new_arc_opt.unwrap_or_else(|| Arc::new(GSSNode::new_dead()));
}

pub(crate) fn disallow_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    tokens_to_disallow: &LLMTokenBV,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let allowed_mask = HybridBitset::max_ones() - tokens_to_disallow.clone();
    allow_only_llm_tokens_and_prune_arc(root_arc, &allowed_mask, memo);
}

pub fn reset_llm_tokens(
    root_arc: &mut Arc<GSSNode>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let mut internal_closure = |internal: &GSSInternal| {
        let mut new_acc = (**internal.acc()).clone();
        new_acc.llm_tokens_union = HybridBitset::max_ones();
        Some((Arc::new(new_acc), true))
    };
    let mut root_closure = |root: &GSSRoot| {
        let mut new_acc = (**root.acc()).clone();
        new_acc.llm_tokens_union = HybridBitset::max_ones();
        Some(Arc::new(new_acc))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &mut internal_closure, &mut root_closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_dead());
    }
}

pub(crate) fn reset_terminals(
    root_arc: &mut Arc<GSSNode>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let mut internal_closure = |internal: &GSSInternal| {
        let mut new_acc = (**internal.acc()).clone();
        new_acc.terminals_union = HybridL2Bitset::all();
        Some((Arc::new(new_acc), true))
    };
    let mut root_closure = |root: &GSSRoot| {
        let mut new_acc = (**root.acc()).clone();
        new_acc.terminals_union = HybridL2Bitset::all();
        Some(Arc::new(new_acc))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &mut internal_closure, &mut root_closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_dead());
    }
}

pub(crate) fn disallow_terminals_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    disallowed_terminals: &HybridL2Bitset,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let node_ptr = Arc::as_ptr(root_arc);
    if let Some(cached) = memo.get(&node_ptr) {
        *root_arc = cached.clone().unwrap_or_else(|| Arc::new(GSSNode::new_dead()));
        return;
    }

    let new_node = match root_arc.as_ref() {
        GSSNode::Root(root) => {
            let mut new_acc = (**root.acc()).clone();
            new_acc.terminals_union -= disallowed_terminals;
            GSSNode::new(new_acc)
        }
        GSSNode::Internal(internal) => {
            let mut new_acc = (**internal.acc()).clone();
            new_acc.terminals_union -= disallowed_terminals;
            GSSNode::new_with_map(Arc::new(new_acc), internal.predecessors().clone())
        }
    };
    let new_arc = Arc::new(new_node);
    memo.insert(node_ptr, Some(new_arc.clone()));
    *root_arc = new_arc;
}

pub fn prune_llm_tokens_by_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    possible_matches: &BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let transform_acc = |acc: &Arc<Acc>| -> Option<Arc<Acc>> {
        if acc.terminals_union == HybridL2Bitset::all() {
            return Some(acc.clone());
        }

        let mut forbidden_llm_tokens = LLMTokenBV::zeros();
        let disallowed_terminals_l2 = acc.terminals_union.complement();

        for (tokenizer_state_range, disallowed_terminals_for_range) in disallowed_terminals_l2.range_values() {
            if disallowed_terminals_for_range.is_empty() {
                continue;
            }

            let relevant_possible_matches = possible_matches.range(TokenizerStateID(*tokenizer_state_range.start())..=TokenizerStateID(*tokenizer_state_range.end()));

            for (_tokenizer_state_id, possible_matches_for_state) in relevant_possible_matches {
                for (terminal_id, llm_tokens_that_match_this_terminal) in possible_matches_for_state {
                    if disallowed_terminals_for_range.contains(terminal_id.0) {
                        forbidden_llm_tokens |= llm_tokens_that_match_this_terminal;
                    }
                }
            }
        }

        if forbidden_llm_tokens.is_empty() {
            return Some(acc.clone());
        }

        let mut new_acc = (**acc).clone();
        new_acc.llm_tokens_union -= &forbidden_llm_tokens;

        if new_acc.llm_tokens_union.is_empty() {
            None // Prune this path
        } else {
            Some(Arc::new(new_acc))
        }
    };

    let mut internal_closure = |internal: &GSSInternal| transform_acc(internal.acc()).map(|new_acc| (new_acc, true));
    let mut root_closure = |root: &GSSRoot| transform_acc(root.acc());

    if let Some(new_root) = prune_and_transform_recursive(root_arc, &mut internal_closure, &mut root_closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_dead());
    }
}

pub fn prune_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    matched_terminals: &BTreeMap<TokenizerStateID, TerminalBV>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let check_and_prune = |node_acc: &Arc<Acc>| -> bool {
        // Returns true if the node should be pruned.
        for (state_id, matched_bv) in matched_terminals {
            let allowed_terminals_union = node_acc.terminals_union.get_l2_bitset(state_id.0).unwrap();
            if !matched_bv.is_subset(allowed_terminals_union) {
                return true;
            }
        }
        false
    };

    let mut internal_closure = |internal: &GSSInternal| {
        if check_and_prune(&internal.acc()) {
            None
        } else {
            Some((internal.acc().clone(), true))
        }
    };
    let mut root_closure = |root: &GSSRoot| -> Option<Arc<Acc>> {
        if check_and_prune(&root.acc()) { None } else { Some(root.acc().clone()) }
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &mut internal_closure, &mut root_closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_dead());
    }
}

pub fn map_allowed_terminals_tokenizer_states(
    root_arc: &mut Arc<GSSNode>,
    map: &BTreeMap<TokenizerStateID, TokenizerStateID>,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let map_one = |terminals: &HybridL2Bitset| -> HybridL2Bitset {
        let mut new_terminals_btreemap = BTreeMap::new();

        for (old_state_id, new_state_id) in map {
            let bv_source = terminals.get_l2_bitset(old_state_id.0).unwrap();
            new_terminals_btreemap.entry(*new_state_id)
                .and_modify(|bv| *bv |= bv_source)
                .or_insert_with(|| bv_source.clone());
        }

        let mut new_terminals_l2_bitset = HybridL2Bitset::all();
        for (state_id, bv) in new_terminals_btreemap {
            new_terminals_l2_bitset.insert_l2_bitset(state_id.0, bv);
        }

        new_terminals_l2_bitset
    };

    let transform_acc = |acc: &Arc<Acc>| -> Option<Arc<Acc>> {
        let mut new_acc = (**acc).clone();
        let new_terminals_union = map_one(&acc.terminals_union);
        new_acc.terminals_union = new_terminals_union;
        Some(Arc::new(new_acc))
    };

    let mut internal_closure = |internal: &GSSInternal| transform_acc(&internal.acc()).map(|acc| (acc, true));
    let mut root_closure = |root: &GSSRoot| transform_acc(&root.acc());

    if let Some(new_root) = prune_and_transform_recursive(root_arc, &mut internal_closure, &mut root_closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new_dead());
    }
}
