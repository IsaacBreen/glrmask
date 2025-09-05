use std::collections::{BTreeMap, HashMap, VecDeque};
use ordered_hash_map::OrderedHashMap;
use crate::constraint::{GrammarConstraintConfig, PrecomputeNode3Index, StateIDBV, Trie3GodWrapper};
use crate::datastructures::EntryApi;
use crate::datastructures::gss::LLMTokenBV;
use crate::datastructures::trie::{EdgeInserter, Trie, Trie2Index};
use crate::tokenizer::TokenizerStateID;

pub fn optimize_trie3_size(
    roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    config: &GrammarConstraintConfig,
) {
    crate::debug!(2, "Optimizing Trie 3 size...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let _all_nodes_pinner = Trie::all_nodes(&trie3_god, &roots_vec);

    if config.optimize_trie2_prune_dead_paths { // Reusing config flags from trie2
        prune_dead_paths_trie3(roots, &trie3_god);
    }
    if config.optimize_trie2_merge_nodes {
        merge_nodes_trie3(roots, &trie3_god);
    }
    if config.optimize_trie2_compress_edges {
        compress_trie3_edges(roots, &trie3_god);
    }
    if config.optimize_trie2_prune_dead_paths {
        prune_dead_paths_trie3(roots, &trie3_god);
    }
    if config.optimize_trie2_merge_nodes {
        merge_nodes_trie3(roots, &trie3_god);
    }
    if config.optimize_trie2_gc {
        Trie::gc(&trie3_god, &roots.values().cloned().collect::<Vec<_>>());
    }
    Trie::recompute_all_max_depths(&trie3_god, &roots.values().cloned().collect::<Vec<_>>());
}

pub fn prune_dead_paths_trie3(roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>, trie3_god: &Trie3GodWrapper) {
    crate::debug!(2, "Pruning dead paths from precomputed trie 3.");

    let all_nodes = Trie::all_nodes(trie3_god, &roots.values().cloned().collect::<Vec<_>>());
    if all_nodes.is_empty() { return; }

    let mut predecessors: HashMap<PrecomputeNode3Index, Vec<(PrecomputeNode3Index, (usize, LLMTokenBV))>> = HashMap::new();
    let mut worklist = VecDeque::new();
    let mut live: HashMap<PrecomputeNode3Index, LLMTokenBV> = HashMap::new();

    // 1. Initialize live sets and build predecessor map.
    for node_arc in &all_nodes {
        let node_ptr = *node_arc;
        live.insert(node_ptr, LLMTokenBV::zeros());

        let guard = node_arc.read(trie3_god).unwrap();
        if guard.value.end {
            let initial_live = guard.value.live_tokens.clone();
            if !initial_live.is_empty() {
                live.insert(node_ptr, initial_live);
                worklist.push_back(node_ptr);
            }
        }

        for (edge_key, dest_map) in guard.children() {
            for child_wrap in dest_map.keys() {
                let child_arc = child_wrap.as_arc().clone();
                let child_ptr = child_arc;
                predecessors.entry(child_ptr).or_default().push((node_ptr, edge_key.clone()));
            }
        }
    }

    // 2. Propagate liveness until a fixed point is reached.
    while let Some(node_ptr) = worklist.pop_front() {
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

    // 3. Prune the graph based on the computed live sets.
    for node_arc in &all_nodes {
        let mut guard = node_arc.write(trie3_god).unwrap();
        let mut new_children: BTreeMap<(usize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>> = BTreeMap::new();

        for (edge_key, dest_map) in guard.children() {
            for (child_wrapper, edge_value_sids) in dest_map {
                let child_arc = child_wrapper.as_arc().clone();
                let child_ptr = child_arc;
                let live_from_child = live.get(&child_ptr).unwrap();

                let live_on_edge = &edge_key.1 & live_from_child;

                if !live_on_edge.is_empty() {
                    let new_edge_key = (edge_key.0, live_on_edge);
                    let new_dest_map_for_key = new_children.entry(new_edge_key).or_default();
                    new_dest_map_for_key.entry(*child_wrapper)
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

pub fn merge_nodes_trie3(roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>, trie3_god: &Trie3GodWrapper) {
    crate::debug!(2, "Merging identical subtrees in precomputed trie 3.");

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() { return; }

    let mut dense_of: HashMap<Trie2Index, usize> = HashMap::new();
    let mut old_of: Vec<Trie2Index> = Vec::with_capacity(all_nodes.len());
    for (i, node_idx) in all_nodes.iter().enumerate() {
        dense_of.insert(*node_idx, i);
        old_of.push(*node_idx);
    }
    let n = all_nodes.len();

    let mut ends: Vec<bool> = vec![false; n];
    type RawEdge3 = (usize, LLMTokenBV, usize, StateIDBV);
    let mut raw_edges: Vec<Vec<RawEdge3>> = vec![Vec::new(); n];

    for (u_dense, u_idx) in old_of.iter().enumerate() {
        let guard = u_idx.read(trie3_god).unwrap();
        ends[u_dense] = guard.value.end;
        for (ek, dest_map) in guard.children() {
            for (v_idx, bv) in dest_map {
                if let Some(&v_dense) = dense_of.get(v_idx) {
                    raw_edges[u_dense].push((ek.0, ek.1.clone(), v_dense, bv.clone()));
                }
            }
        }
    }

    let mut prev_class: Vec<usize> = (0..n).map(|i| if ends[i] { 1 } else { 0 }).collect();

    const MAX_ITERS: usize = 40;
    for it in 0..MAX_ITERS {
        type AggregatedEdge3 = ((usize, LLMTokenBV, usize), StateIDBV);
        type Signature3 = (bool, Vec<AggregatedEdge3>);

        let mut sig_to_id: HashMap<Signature3, usize> = HashMap::new();
        let mut new_class = vec![0; n];
        let mut next_id = 0;
        let mut changes = 0;

        for u in 0..n {
            let mut aggr: BTreeMap<(usize, LLMTokenBV, usize), StateIDBV> = BTreeMap::new();
            for (p, bv_key, v_dense, sids) in &raw_edges[u] {
                let dest_class = prev_class[*v_dense];
                let key = (*p, bv_key.clone(), dest_class);
                aggr.entry(key).and_modify(|e| *e |= sids).or_insert_with(|| sids.clone());
            }
            let agg_edges: Vec<AggregatedEdge3> = aggr.into_iter().collect();

            let sig: Signature3 = (ends[u], agg_edges);

            let cid = *sig_to_id.entry(sig).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });

            new_class[u] = cid;
            if new_class[u] != prev_class[u] {
                changes += 1;
            }
        }

        crate::debug!(3, "Trie3 merge iter {}: classes={}, changes={}", it + 1, next_id, changes);
        prev_class = new_class;
        if changes == 0 { break; }
    }

    let final_partition = prev_class;
    let num_classes = final_partition.iter().max().map_or(0, |m| m + 1);

    let mut representatives: Vec<Option<Trie2Index>> = vec![None; num_classes];
    for (u_dense, &class_id) in final_partition.iter().enumerate() {
        if representatives[class_id].is_none() {
            representatives[class_id] = Some(old_of[u_dense]);
        }
    }

    let mut node_to_rep: HashMap<Trie2Index, Trie2Index> = HashMap::new();
    for (u_dense, &class_id) in final_partition.iter().enumerate() {
        node_to_rep.insert(old_of[u_dense], representatives[class_id].unwrap());
    }

    for class_id in 0..num_classes {
        if let Some(rep_idx) = representatives[class_id] {
            let u_dense = final_partition.iter().position(|&c| c == class_id).unwrap();

            let mut aggr: BTreeMap<(usize, LLMTokenBV, usize), StateIDBV> = BTreeMap::new();
            for (p, bv_key, v_dense, sids) in &raw_edges[u_dense] {
                let dest_class = final_partition[*v_dense];
                aggr.entry((*p, bv_key.clone(), dest_class)).and_modify(|e| *e |= sids).or_insert_with(|| sids.clone());
            }

            let mut new_children = BTreeMap::new();
            let mut new_live_tokens = LLMTokenBV::zeros();
            for ((p, bv_key, dest_class), sids) in aggr {
                if let Some(dest_rep_idx) = representatives[dest_class] {
                    new_children.entry((p, bv_key.clone())).or_insert_with(OrderedHashMap::new).insert(dest_rep_idx, sids);
                    new_live_tokens |= &bv_key;
                }
            }

            for (i, &c) in final_partition.iter().enumerate() {
                if c == class_id {
                    new_live_tokens |= &old_of[i].read(trie3_god).unwrap().value.live_tokens;
                }
            }

            let mut guard = rep_idx.write(trie3_god).unwrap();
            *guard.children_mut() = new_children;
            guard.value.live_tokens = new_live_tokens;
        }
    }

    for root_idx in roots.values_mut() {
        *root_idx = *node_to_rep.get(root_idx).unwrap();
    }

    let final_roots_vec: Vec<_> = roots.values().cloned().collect();
    Trie::recompute_all_max_depths(trie3_god, &final_roots_vec);
}

pub fn compress_trie3_edges(roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>, trie3_god: &Trie3GodWrapper) {
    crate::debug!(2, "Compressing Trie 3 edges...");
    let mut changed = true;
    let mut iter = 0;
    const MAX_ITERS: usize = 10;

    while changed && iter < MAX_ITERS {
        iter += 1;
        changed = false;

        let all_nodes = Trie::all_nodes(trie3_god, &roots.values().cloned().collect::<Vec<_>>());
        if all_nodes.is_empty() { return; }

        let root_indices: std::collections::HashSet<_> = roots.values().cloned().collect();

        // 1. Build predecessor map
        let mut predecessors: HashMap<PrecomputeNode3Index, Vec<(PrecomputeNode3Index, (usize, LLMTokenBV), StateIDBV)>> = HashMap::new();
        for node_arc in &all_nodes {
            let guard = node_arc.read(trie3_god).unwrap();
            for (edge_key, dest_map) in guard.children() {
                for (child_wrap, edge_value) in dest_map {
                    predecessors.entry(*child_wrap).or_default().push((*node_arc, edge_key.clone(), edge_value.clone()));
                }
            }
        }

        // 2. Find and process candidates
        for node_b_arc in &all_nodes {
            let node_b_idx = *node_b_arc;

            // Skip roots and end nodes
            if root_indices.contains(&node_b_idx) { continue; }
            let is_end_node = { node_b_arc.read(trie3_god).unwrap().value.end };
            if is_end_node { continue; }

            // Candidate must have only pop=0 outgoing edges
            let successors = {
                let guard = node_b_arc.read(trie3_god).unwrap();
                let mut succs = Vec::new();
                let mut is_candidate = !guard.children().is_empty();
                for (edge_key, dest_map) in guard.children() {
                    if edge_key.0 != 0 {
                        is_candidate = false;
                        break;
                    }
                    for (child_wrap, edge_value) in dest_map {
                        succs.push((*child_wrap, edge_key.clone(), edge_value.clone()));
                    }
                }
                if is_candidate { Some(succs) } else { None }
            };

            if successors.is_none() { continue; }
            let successors = successors.unwrap();
            if successors.is_empty() { continue; }

            let preds = match predecessors.get(&node_b_idx) {
                Some(p) if !p.is_empty() => p.clone(),
                _ => continue,
            };

            // Heuristic: compress if in-degree or out-degree is 1.
            let in_degree = preds.iter().map(|(p, _, _)| p).collect::<std::collections::HashSet<_>>().len();
            let out_degree = successors.iter().map(|(s, _, _)| s).collect::<std::collections::HashSet<_>>().len();

            if in_degree > 1 && out_degree > 1 {
                continue;
            }

            changed = true;

            // For each predecessor A, create new edges to all successors C
            for (pred_a_idx, (pop_a, llm_bv_a), sids_a) in &preds {
                for (succ_c_idx, (_pop_b, llm_bv_b), sids_b) in &successors {
                    let new_llm_bv = llm_bv_a & llm_bv_b;
                    if new_llm_bv.is_empty() { continue; }

                    let new_sids = sids_a & sids_b;
                    if new_sids.is_empty() { continue; }

                    let new_edge_key = (*pop_a, new_llm_bv);

                    let mut pred_a_guard = pred_a_idx.write(trie3_god).unwrap();
                    let dest_map = pred_a_guard.children_mut().entry(new_edge_key).or_default();
                    dest_map.entry(*succ_c_idx).and_modify(|v| *v |= &new_sids).or_insert(new_sids);
                }
            }

            // Remove A -> B edges
            for (pred_a_idx, edge_key_ab, _) in &preds {
                let mut pred_a_guard = pred_a_idx.write(trie3_god).unwrap();
                if let Some(dest_map) = pred_a_guard.children_mut().get_mut(edge_key_ab) {
                    dest_map.remove(&node_b_idx);
                    if dest_map.is_empty() {
                        pred_a_guard.children_mut().remove(edge_key_ab);
                    }
                }
            }

            // Remove B -> C edges by clearing all children of B
            node_b_arc.write(trie3_god).unwrap().children_mut().clear();
        }
        if changed {
            crate::debug!(3, "Trie3 compression iter {}: made changes.", iter);
        } else {
            crate::debug!(3, "Trie3 compression iter {}: no changes, fixed point reached.", iter);
        }
    }
    crate::debug!(2, "Finished compressing Trie 3 edges after {} iterations.", iter);
}
