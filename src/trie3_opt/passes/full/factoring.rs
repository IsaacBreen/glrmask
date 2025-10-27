use std::collections::{BTreeMap, HashMap};

use ordered_hash_map::OrderedHashMap;

use crate::constraint::{
    LLMTokenBV, PrecomputeNode3Index, PrecomputedNodeContents, StateIDBV, Trie3GodWrapper,
};
use crate::datastructures::EntryApi;
use crate::datastructures::trie::Trie;

/// Reduces state fanout by factoring edges. For each node, it groups outgoing edges
/// by (pop, destination). For each group, it merges the edges by unioning their
/// token sets and state ID sets. This can reduce the number of distinct edges
/// a single (pop, state_id) transition can take, thus lowering state fanout.
pub fn factor_state_fanout_trie3(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
) {
    crate::debug!(2, "Factoring state fanout in Trie3...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    for node_idx in &all_nodes {
        let mut w = if let Some(guard) = node_idx.write(trie3_god) {
            guard
        } else {
            continue;
        };
        if w.children().is_empty() {
            continue;
        }

        let old_children = std::mem::take(w.children_mut());

        // Group by (pop, destination)
        let mut grouped_edges: BTreeMap<(isize, PrecomputeNode3Index), (LLMTokenBV, StateIDBV)> =
            BTreeMap::new();

        for ((pop, llm_bv), dm) in old_children.iter() {
            for (dest, sids) in dm.iter() {
                let key = (*pop, *dest);
                let entry = grouped_edges
                    .entry(key)
                    .or_insert_with(|| (LLMTokenBV::zeros(), StateIDBV::zeros()));
                entry.0 |= llm_bv;
                entry.1 |= sids;
            }
        }

        // Rebuild children from factored edges.
        // Group again by (pop, tokens) to form valid edges.
        let mut new_children: BTreeMap<
            (isize, LLMTokenBV),
            OrderedHashMap<PrecomputeNode3Index, StateIDBV>,
        > = BTreeMap::new();

        for ((pop, dest), (tokens, sids)) in grouped_edges {
            if !tokens.is_empty() && !sids.is_empty() {
                let dm = new_children.entry((pop, tokens)).or_default();
                dm.insert(dest, sids);
            }
        }

        // Recompute live tokens and update children
        let mut new_live = LLMTokenBV::zeros();
        for ((_, llm_bv), _) in &new_children {
            new_live |= llm_bv;
        }
        w.value.live_tokens = new_live;
        *w.children_mut() = new_children;
    }
}

/// Factor out common destinations: create intermediates when many sources share the same dest map.
pub fn factor_common_destinations_trie3(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    max_llm_token_id: usize,
    max_state_id: usize,
    min_incoming: usize,
) {
    crate::debug!(2, "Factoring out common destinations in Trie3.");
    // use dynamic threshold provided by config via min_incoming
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    let all_llm_bv = LLMTokenBV::ones(max_llm_token_id + 1);
    let all_sids_bv = StateIDBV::ones(max_state_id + 1);

    // Map: dest -> { (pop, llm_bv) -> { state_id_bv -> [sources] } }
    let mut incoming_map: HashMap<
        PrecomputeNode3Index,
        HashMap<(isize, LLMTokenBV), HashMap<StateIDBV, Vec<PrecomputeNode3Index>>>,
    > = HashMap::new();

    for src_idx in &all_nodes {
        let guard = src_idx.read(trie3_god).expect("read");
        for (edge_key, dest_map) in guard.children() {
            for (dest_idx, sids_bv) in dest_map {
                incoming_map
                    .entry(*dest_idx)
                    .or_default()
                    .entry(edge_key.clone())
                    .or_default()
                    .entry(sids_bv.clone())
                    .or_default()
                    .push(*src_idx);
            }
        }
    }

    for (dest_idx, edges_by_key) in incoming_map {
        for (edge_key, sources_by_sids) in edges_by_key {
            for (sids_bv, sources) in sources_by_sids {
                if sources.len() >= min_incoming {
                    // Create intermediate node
                    let intermediate_node = PrecomputeNode3Index::new(
                        trie3_god.insert(Trie::new(PrecomputedNodeContents::internal())),
                    );

                    // Add edge from intermediate to original destination
                    {
                        let mut intermediate_guard =
                            intermediate_node.write(trie3_god).expect("write");
                        let dest_map = intermediate_guard.children_mut().entry(edge_key.clone()).or_default();
                        dest_map.insert(dest_idx, sids_bv.clone());
                        intermediate_guard.value.live_tokens |= &edge_key.1;
                    }

                    // Reroute sources to point to intermediate node
                    for src_idx in &sources {
                        let mut src_guard = src_idx.write(trie3_god).expect("write");

                        // Remove old edge
                        if let Some(dest_map_for_key) = src_guard.children_mut().get_mut(&edge_key) {
                            dest_map_for_key.remove(&dest_idx);
                            if dest_map_for_key.is_empty() {
                                src_guard.children_mut().remove(&edge_key);
                            }
                        }

                        // Add new edge to intermediate node. This is a "None-like" edge.
                        // pop=0, all llm tokens, all state ids.
                        let none_like_edge_key = (0, all_llm_bv.clone());
                        let dest_map =
                            src_guard.children_mut().entry(none_like_edge_key).or_default();
                        dest_map.insert(intermediate_node, all_sids_bv.clone());
                        // Recompute live tokens from scratch after modifying edges.
                        let mut new_live = LLMTokenBV::zeros();
                        for ((_, llm_bv), _) in src_guard.children() {
                            new_live |= llm_bv;
                        }
                        src_guard.value.live_tokens = new_live;
                    }
                }
            }
        }
    }
    crate::debug!(2, "Finished factoring common destinations in Trie3.");
}
