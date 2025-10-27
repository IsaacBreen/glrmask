use std::collections::{BTreeMap, HashMap, VecDeque};

use indicatif::{ProgressBar, ProgressStyle};
use ordered_hash_map::OrderedHashMap;

use crate::constraint::{LLMTokenBV, PrecomputeNode3Index, StateIDBV, Trie3GodWrapper};
use crate::datastructures::EntryApi;
use crate::datastructures::trie::{Trie, Trie2Index};
use crate::profiler::PROGRESS_BAR_ENABLED;

/// Backwards reachability pruning on token-liveness through edge masks.
/// Removes edges that cannot lead to an end (w.r.t. tokens and states).
pub fn prune_dead_paths_trie3(
    roots: &mut BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
) {
    crate::debug!(2, "Pruning dead paths from precomputed trie 3.");

    let all_nodes = Trie::all_nodes(trie3_god, &roots.values().cloned().collect::<Vec<_>>());
    if all_nodes.is_empty() {
        return;
    }

    let mut predecessors: HashMap<
        PrecomputeNode3Index,
        Vec<(PrecomputeNode3Index, (isize, LLMTokenBV))>,
    > = HashMap::new();
    let mut worklist = VecDeque::new();
    let mut live: HashMap<PrecomputeNode3Index, LLMTokenBV> = HashMap::new();

    // 1. Initialize live sets and build predecessor map.
    for node_arc in &all_nodes {
        let node_ptr = *node_arc;
        live.insert(node_ptr, LLMTokenBV::zeros());

        let guard = node_arc.read(trie3_god).unwrap();
        if guard.value.end {
            // Seed end nodes with 'all tokens' to allow backward propagation through edge masks.
            live.insert(node_ptr, LLMTokenBV::max_ones());
            worklist.push_back(node_ptr);
        }

        for (edge_key, dest_map) in guard.children() {
            for child_wrap in dest_map.keys() {
                predecessors
                    .entry(*child_wrap)
                    .or_default()
                    .push((node_ptr, edge_key.clone()));
            }
        }
    }

    #[cfg(not(rustrover))]
    let pb = {
        let pb = ProgressBar::new(all_nodes.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [Trie3 Prune] [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap(),
        );
        if !PROGRESS_BAR_ENABLED {
            pb.set_draw_target(indicatif::ProgressDrawTarget::hidden());
        }
        pb.set_position(0);
        pb
    };
    #[cfg(rustrover)]
    let pb = ProgressBar::hidden();

    // 2. Propagate liveness until a fixed point is reached.
    while let Some(node_ptr) = worklist.pop_front() {
        pb.inc(1);

        let live_at_node = live.get(&node_ptr).unwrap().clone();
        if let Some(preds) = predecessors.get(&node_ptr) {
            for (pred_ptr, edge_key) in preds {
                let live_from_edge = &live_at_node & &edge_key.1;
                if live_from_edge.is_empty() {
                    continue;
                }

                let pred_live = live.get_mut(pred_ptr).unwrap();
                let old_len = pred_live.len();
                *pred_live |= &live_from_edge;
                if pred_live.len() > old_len {
                    worklist.push_back(*pred_ptr);
                }
            }
        }
    }
    pb.finish_and_clear();

    // 3. Prune the graph based on the computed live sets.
    for node_arc in &all_nodes {
        let mut guard = node_arc.write(trie3_god).unwrap();
        let mut new_children: BTreeMap<
            (isize, LLMTokenBV),
            OrderedHashMap<Trie2Index, StateIDBV>,
        > = BTreeMap::new();

        for (edge_key, dest_map) in guard.children() {
            for (child_wrapper, edge_value_sids) in dest_map {
                let live_from_child = live.get(child_wrapper).unwrap();

                let live_on_edge = &edge_key.1 & live_from_child;

                if !live_on_edge.is_empty() {
                    let new_edge_key = (edge_key.0, live_on_edge);
                    let new_dest_map_for_key = new_children.entry(new_edge_key).or_default();
                    new_dest_map_for_key
                        .entry(*child_wrapper)
                        .and_modify(|v| *v |= edge_value_sids)
                        .or_insert_with(|| edge_value_sids.clone());
                }
            }
        }
        *guard.children_mut() = new_children;

        // Update the node's own live_tokens field with the final computed value.
        let node_ptr = *node_arc;
        guard.value.live_tokens = live.get(&node_ptr).unwrap().clone();
    }
    crate::debug!(2, "Finished pruning dead paths from trie 3.");
}
