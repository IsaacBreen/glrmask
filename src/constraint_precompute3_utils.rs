use std::collections::{BTreeMap, HashMap, VecDeque, HashSet};
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
    if config.optimize_trie2_compress_edges {
        compress_trie3_edges(roots, &trie3_god);
    }
    if config.optimize_trie2_merge_nodes {
        merge_nodes_trie3(roots, &trie3_god);
    }
    if config.optimize_trie2_prune_dead_paths {
        prune_dead_paths_trie3(roots, &trie3_god);
    }
    if config.optimize_trie2_compress_edges {
        compress_trie3_edges(roots, &trie3_god);
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
    crate::debug!(2, "Compressing Trie 3 edges (conservative edge-reducing transforms)...");

    /// Remove dominated edges within each node.
    /// For a fixed (pop p) and child C, if there are multiple edges with (LLM_a, SIDs_a) and (LLM_b, SIDs_b)
    /// such that LLM_a ⊆ LLM_b and SIDs_a ⊆ SIDs_b, then the (LLM_a, SIDs_a) edge is redundant and can be removed.
    fn remove_dominated_edges_within_nodes(trie3_god: &Trie3GodWrapper, roots_vec: &[PrecomputeNode3Index]) -> bool {
        crate::debug!(2, "Removing dominated edges within nodes (Trie 3)...");

        let nodes = Trie::all_nodes(trie3_god, roots_vec);
        if nodes.is_empty() { return false; }
        let mut changed_any = false;

        // Helper subset checks via bitwise-and equality
        fn bv_subset_llm(a: &LLMTokenBV, b: &LLMTokenBV) -> bool {
            // a ⊆ b  iff (a & b) == a
            let tmp = a & b;
            tmp == *a
        }
        fn bv_subset_sid(a: &StateIDBV, b: &StateIDBV) -> bool {
            let tmp = a & b;
            tmp == *a
        }

        for u in nodes {
            let old_children = {
                let g = u.read(trie3_god).expect("read");
                g.children().clone()
            };
            if old_children.is_empty() { continue; }

            // Group per pop p, then per child
            // For each pop p: child -> Vec<(llm_bv, sids)>
            let mut groups: HashMap<usize, HashMap<Trie2Index, Vec<(LLMTokenBV, StateIDBV)>>> = HashMap::new();
            for ((pop, llm_bv), dest_map) in &old_children {
                let entry = groups.entry(*pop).or_default();
                for (child_idx, sids) in dest_map.iter() {
                    entry.entry(*child_idx).or_default().push((llm_bv.clone(), sids.clone()));
                }
            }

            // Build new children by removing dominated entries
            let mut new_children: BTreeMap<(usize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>> = BTreeMap::new();
            let mut local_changed = false;

            for (pop, by_child) in groups {
                for (child, mut entries) in by_child {
                    if entries.len() <= 1 {
                        // Reinsert as-is
                        if let Some((llm0, sids0)) = entries.pop() {
                            new_children.entry((pop, llm0)).or_default().insert(child, sids0);
                        }
                        continue;
                    }
                    // Mark dominated
                    let mut keep = vec![true; entries.len()];
                    for i in 0..entries.len() {
                        if !keep[i] { continue; }
                        for j in 0..entries.len() {
                            if i == j || !keep[j] { continue; } // check keep[j] to avoid comparing against already-dominated
                            let (ref li, ref si) = entries[i];
                            let (ref lj, ref sj) = entries[j];
                            // If (li, si) dominated by (lj, sj), drop i
                            if bv_subset_llm(li, lj) && bv_subset_sid(si, sj) {
                                keep[i] = false;
                                break; // No need to check further for i
                            }
                        }
                    }
                    // Reinsert all kept entries
                    for (k, (llm, sids)) in entries.into_iter().enumerate() {
                        if keep[k] {
                            new_children.entry((pop, llm)).or_default().insert(child, sids);
                        } else {
                            local_changed = true;
                        }
                    }
                }
            }

            if new_children != old_children {
                let mut w = u.write(trie3_god).expect("write");
                *w.children_mut() = new_children;
                changed_any = true;
            } else if local_changed {
                // This case can happen if the BTreeMap order changes but content is same
                changed_any = true;
            }
        }

        changed_any
    }

    /// Generalized zero-pop shortcut across multiple outgoing edges of the middle node,
    /// applied conservatively only when it doesn't increase edge count.
    ///
    /// For U --(p1,L1)-> V (with SIDs S1), and V has only pop-0 outgoing edges:
    ///   For each V --(0,L2)-> C with SIDs S2:
    ///     Produce (p1, L1∩L2) -> C with (S1∩S2), if intersections are non-empty.
    /// Aggregate per child C (union of LLM and union of SIDs) across all eligible V under that (p1, L1) key.
    /// Apply only if new_edges_count <= removed_edges_count for that key to avoid edge explosion.
    fn shortcut_zero_pop_chains_batched(trie3_god: &Trie3GodWrapper, roots_vec: &[PrecomputeNode3Index]) -> bool {
        crate::debug!(2, "Shortcutting zero-pop chains in batch (Trie 3)...");
        let nodes = Trie::all_nodes(trie3_god, roots_vec);
        if nodes.is_empty() { return false; }
        let mut changed_any = false;

        for u in nodes {
            // Snapshot children
            let children_snapshot: Vec<((usize, LLMTokenBV), Vec<(Trie2Index, StateIDBV)>)> = {
                let g = u.read(trie3_god).expect("read");
                g.children()
                    .iter()
                    .map(|(ek, dm)| {
                        let dests = dm.iter().map(|(d, sids)| (*d, sids.clone())).collect::<Vec<_>>();
                        (ek.clone(), dests)
                    })
                    .collect()
            };
            if children_snapshot.is_empty() { continue; }

            let mut local_changed = false;
            let mut w = u.write(trie3_god).expect("write");

            for ((p1, llm1), dests) in &children_snapshot {
                // Aggregate candidates per child, and track which V are eligible to be removed
                let mut aggregated: BTreeMap<Trie2Index, (LLMTokenBV, StateIDBV)> = BTreeMap::new();
                let mut eligible_vs: Vec<Trie2Index> = Vec::new();

                for (v, sids1) in dests {
                    // Inspect V
                    let v_guard = v.read(trie3_god).expect("read");
                    let mut has_nonzero_pop = false;
                    let mut zero_pop_edges: Vec<(LLMTokenBV, Vec<(Trie2Index, StateIDBV)>)> = Vec::new();
                    for (ek2, dm2) in v_guard.children() {
                        if ek2.0 == 0 {
                            let v2 = dm2.iter().map(|(c, s2)| (*c, s2.clone())).collect::<Vec<_>>();
                            zero_pop_edges.push((ek2.1.clone(), v2));
                        } else {
                            has_nonzero_pop = true;
                            break;
                        }
                    }
                    drop(v_guard);

                    if has_nonzero_pop || zero_pop_edges.is_empty() {
                        // Not eligible: either has non-zero-pop edges, or no outgoing edges
                        continue;
                    }

                    // Eligible V: compose through all pop-0 edges
                    let mut contributed_any = false;
                    for (llm2, list) in zero_pop_edges {
                        let new_llm = llm1.clone() & &llm2;
                        if new_llm.is_empty() { continue; }
                        for (c, s2) in list {
                            let new_sids = sids1 & &s2;
                            if new_sids.is_empty() { continue; }
                            let entry = aggregated.entry(c).or_insert_with(|| (LLMTokenBV::zeros(), StateIDBV::zeros()));
                            entry.0 |= &new_llm;
                            *&mut entry.1 |= &new_sids;
                            contributed_any = true;
                        }
                    }
                    if contributed_any {
                        eligible_vs.push(*v);
                    }
                }

                if eligible_vs.is_empty() {
                    continue;
                }

                // Decide if beneficial: remove |eligible_vs| edges; add |aggregated| edges
                let removed_count = eligible_vs.len();
                let added_count = aggregated.len();
                if added_count > removed_count {
                    // Avoid blow-up
                    continue;
                }

                // Apply: remove U --(p1,llm1)--> V for eligible V
                if let Some(dm) = w.children_mut().get_mut(&(*p1, llm1.clone())) {
                    let mut removed_any = false;
                    for v in eligible_vs {
                        if dm.remove(&v).is_some() {
                            removed_any = true;
                        }
                    }
                    if removed_any {
                        if dm.is_empty() {
                            w.children_mut().remove(&(*p1, llm1.clone()));
                        }
                        local_changed = true;
                    }
                }

                // Add aggregated edges: U --(p1, aggregated_llm)--> C with aggregated_sids
                if !aggregated.is_empty() {
                    for (c, (llm_u, sids_u)) in aggregated {
                        if llm_u.is_empty() || sids_u.is_empty() { continue; }
                        let dest_map = w.children_mut().entry((*p1, llm_u)).or_default();
                        dest_map.entry(c)
                            .and_modify(|s| { *s |= &sids_u; })
                            .or_insert(sids_u);
                    }
                }
            }

            if local_changed {
                changed_any = true;
            }
        }

        changed_any
    }

    // Pass 1: local coalesce within each node
    fn coalesce_edges_within_nodes(trie3_god: &Trie3GodWrapper, roots_vec: &[PrecomputeNode3Index]) -> bool {
        let nodes = Trie::all_nodes(trie3_god, roots_vec);
        if nodes.is_empty() { return false; }
        let mut changed_any = false;

        for node_idx in nodes {
            // Snapshot current children
            let old_children = {
                let g = node_idx.read(trie3_god).expect("read");
                g.children().clone() // BTreeMap<(usize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>>
            };
            if old_children.is_empty() { continue; }

            // Aggregate per (pop, child, sids): union LLM-token BVs
            let mut by_pop: HashMap<usize, Vec<(Trie2Index, StateIDBV, LLMTokenBV)>> = HashMap::new();
            for ((pop, llm_bv), dest_map) in &old_children {
                for (child_idx, sids) in dest_map.iter() {
                    let items = by_pop.entry(*pop).or_default();
                    let mut found = false;
                    for (c, c_sids, llm_union) in items.iter_mut() {
                        if c == child_idx && c_sids == sids {
                            *llm_union |= llm_bv;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        items.push((*child_idx, sids.clone(), llm_bv.clone()));
                    }
                }
            }

            // Rebuild children from aggregates
            let mut new_children: BTreeMap<(usize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>> = BTreeMap::new();
            for (pop, vec_items) in by_pop {
                for (child, sids, llm_union) in vec_items {
                    if llm_union.is_empty() || sids.is_empty() {
                        continue;
                    }
                    new_children.entry((pop, llm_union)).or_default().insert(child, sids);
                }
            }

            if new_children != old_children {
                let mut w = node_idx.write(trie3_god).expect("write");
                *w.children_mut() = new_children;
                changed_any = true;
            }
        }

        changed_any
    }

    // Pass 2: shortcut zero-pop chains.
    // Contracts sequences V --(pop 0, L2, S2)--> ... --(pop 0, Lk, Sk)--> Z
    // into U --(p1, L1∩L2∩...∩Lk, S1∩S2∩...∩Sk)--> Z where U --(p1, L1, S1)--> V.
    // Only applies when each intermediate has exactly one outgoing (pop 0) edge with exactly one destination (no fanout), avoiding edge explosion.
    fn shortcut_zero_pop_chains(trie3_god: &Trie3GodWrapper, roots_vec: &[PrecomputeNode3Index]) -> bool {
        let nodes = Trie::all_nodes(trie3_god, roots_vec);
        if nodes.is_empty() { return false; }

        // Snapshot summaries for quick lookups
        type DestList = Vec<(Trie2Index, StateIDBV)>;
        type EdgeList = Vec<(usize, LLMTokenBV, DestList)>;
        let mut summary: HashMap<Trie2Index, (bool, EdgeList)> = HashMap::new();
        for n in &nodes {
            let g = n.read(trie3_god).expect("read");
            let edges: EdgeList = g.children()
                .iter()
                .map(|(ek, dm)| {
                    let dests = dm.iter().map(|(d, sids)| (*d, sids.clone())).collect::<DestList>();
                    (ek.0, ek.1.clone(), dests)
                })
                .collect();
            summary.insert(*n, (g.value.end, edges));
        }

        // Memoization for zero-pop chain results
        #[derive(Clone)]
        struct ChainRes {
            last: Trie2Index,
            llm: LLMTokenBV,
            sids: StateIDBV,
        }
        let mut memo: HashMap<Trie2Index, Option<ChainRes>> = HashMap::new();

        fn follow_zero_chain(
            v: Trie2Index,
            summary: &HashMap<Trie2Index, (bool, EdgeList)>,
            memo: &mut HashMap<Trie2Index, Option<ChainRes>>,
        ) -> Option<ChainRes> {
            if let Some(cached) = memo.get(&v) {
                return cached.clone();
            }
            let (_is_end, edges) = match summary.get(&v) {
                Some(x) => x,
                None => {
                    memo.insert(v, None);
                    return None;
                }
            };
            // Must be exactly one outgoing edge, pop == 0, with exactly one destination.
            let mut pop0_edges = edges.iter().filter(|(p, _, _)| *p == 0);
            let next = match pop0_edges.next() {
                Some(x) => x,
                None => {
                    memo.insert(v, None);
                    return None;
                }
            };
            // Ensure it is the only outgoing edge and has a single destination.
            if edges.len() != 1 || next.2.len() != 1 {
                memo.insert(v, None);
                return None;
            }
            let (_p0, llm2, dests) = next;
            let (w, sids2) = &dests[0];

            // Recurse forward
            let res = if let Some(tail) = follow_zero_chain(*w, summary, memo) {
                Some(ChainRes {
                    last: tail.last,
                    llm: llm2 & &tail.llm,
                    sids: sids2 & &tail.sids,
                })
            } else {
                Some(ChainRes {
                    last: *w,
                    llm: llm2.clone(),
                    sids: sids2.clone(),
                })
            };
            memo.insert(v, res.clone());
            res
        }

        let mut changed_any = false;

        for u in &nodes {
            // Snapshot children (stable during this node's rewrite)
            let children_snapshot: Vec<((usize, LLMTokenBV), Vec<(Trie2Index, StateIDBV)>)> = {
                let g = u.read(trie3_god).expect("read");
                g.children()
                    .iter()
                    .map(|(ek, dm)| {
                        let dests = dm.iter().map(|(d, sids)| (*d, sids.clone())).collect::<Vec<_>>();
                        (ek.clone(), dests)
                    })
                    .collect()
            };
            if children_snapshot.is_empty() { continue; }

            let mut local_changed = false;
            let mut w = u.write(trie3_god).expect("write");

            for ((p1, llm1), dests) in &children_snapshot {
                // We will remove/replace individual destinations for this key.
                for (v, sids1) in dests {
                    if let Some(chain) = follow_zero_chain(*v, &summary, &mut memo) {
                        // Compose new filters
                        let new_llm = llm1 & &chain.llm;
                        let new_sids = sids1 & &chain.sids;

                        // Remove old edge U --(p1, llm1)--> V
                        if let Some(dm) = w.children_mut().get_mut(&(p1.clone(), llm1.clone())) {
                            if dm.remove(v).is_some() {
                                local_changed = true;
                            }
                            if dm.is_empty() {
                                w.children_mut().remove(&(p1.clone(), llm1.clone()));
                            }
                        }

                        // If empty, nothing to add; drop the path.
                        if new_llm.is_empty() || new_sids.is_empty() {
                            continue;
                        }

                        // Insert U --(p1, new_llm)--> chain.last with new_sids
                        let dest_map = w.children_mut().entry((*p1, new_llm)).or_default();
                        dest_map.entry(chain.last)
                            .and_modify(|s| *s |= &new_sids)
                            .or_insert(new_sids);
                    }
                }
            }

            if local_changed {
                changed_any = true;
            }
        }

        changed_any
    }

    // Pass 3: shortcut when the middle has a single outgoing edge (with exactly one destination).
    // A --(p1, L1, S1)--> B and B --(p2>0, L2, S2)--> C (only outgoing, only one destination)
    // becomes A --(p1+p2, L1∩L2, S1∩S2)--> C (if intersections are not empty).
    fn shortcut_single_out_pop_step(trie3_god: &Trie3GodWrapper, roots_vec: &[PrecomputeNode3Index]) -> bool {
        let nodes = Trie::all_nodes(trie3_god, roots_vec);
        if nodes.is_empty() { return false; }

        // Summaries
        type DestList = Vec<(Trie2Index, StateIDBV)>;
        type EdgeList = Vec<(usize, LLMTokenBV, DestList)>;
        let mut summary: HashMap<Trie2Index, (bool, EdgeList)> = HashMap::new();
        for n in &nodes {
            let g = n.read(trie3_god).expect("read");
            let edges: EdgeList = g.children()
                .iter()
                .map(|(ek, dm)| {
                    let dests = dm.iter().map(|(d, sids)| (*d, sids.clone())).collect::<DestList>();
                    (ek.0, ek.1.clone(), dests)
                })
                .collect();
            summary.insert(*n, (g.value.end, edges));
        }

        // Identify "compressible" middle nodes: exactly one outgoing edge, with exactly one destination, pop > 0
        let mut middle_info: HashMap<Trie2Index, (usize, LLMTokenBV, Trie2Index, StateIDBV)> = HashMap::new();
        for n in &nodes {
            let (is_end, edges) = summary.get(n).unwrap();
            if *is_end { continue; }
            if edges.len() != 1 { continue; }
            let (p2, llm2, dests) = &edges[0];
            if *p2 == 0 { continue; } // leave pop=0 to other pass
            if dests.len() != 1 { continue; }
            let (c, sids2) = &dests[0];
            middle_info.insert(*n, (*p2, llm2.clone(), *c, sids2.clone()));
        }

        let mut changed_any = false;

        for u in &nodes {
            // Snapshot children
            let children_snapshot: Vec<((usize, LLMTokenBV), Vec<(Trie2Index, StateIDBV)>)> = {
                let g = u.read(trie3_god).expect("read");
                g.children()
                    .iter()
                    .map(|(ek, dm)| {
                        let dests = dm.iter().map(|(d, sids)| (*d, sids.clone())).collect::<Vec<_>>();
                        (ek.clone(), dests)
                    })
                    .collect()
            };
            if children_snapshot.is_empty() { continue; }

            let mut local_changed = false;
            let mut w = u.write(trie3_god).expect("write");

            for ((p1, llm1), dests) in &children_snapshot {
                for (v, sids1) in dests {
                    if let Some((p2, llm2, c, sids2)) = middle_info.get(v).cloned() {
                        // Remove old edge U --(p1, llm1)--> V
                        if let Some(dm) = w.children_mut().get_mut(&(p1.clone(), llm1.clone())) {
                            if dm.remove(v).is_some() {
                                local_changed = true;
                            }
                            if dm.is_empty() {
                                w.children_mut().remove(&(p1.clone(), llm1.clone()));
                            }
                        }
                        // Intersections for new edge
                        let new_llm = llm1 & &llm2;
                        let new_sids = sids1 & &sids2;
                        if !new_llm.is_empty() && !new_sids.is_empty() {
                            // Insert U --(p1+p2, new_llm)--> C with new_sids
                            let key_new = (p1 + p2, new_llm);
                            let dest_map = w.children_mut().entry(key_new).or_default();
                            dest_map.entry(c)
                                .and_modify(|s| *s |= &new_sids)
                                .or_insert(new_sids);
                        }
                    }
                }
            }

            if local_changed {
                changed_any = true;
            }
        }

        changed_any
    }

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    if Trie::all_nodes(trie3_god, &roots_vec).is_empty() {
        return;
    }

    // Iterate to a (small) fixpoint so that local changes enable further opportunities.
    const MAX_PASSES: usize = 4;
    let mut any_changed = false;
    for pass in 0..MAX_PASSES {
        let mut pass_changed = false;
        // 1) Coalesce within nodes (cheap win)
        if coalesce_edges_within_nodes(trie3_god, &roots_vec) {
            pass_changed = true;
        }
        // 2) Shortcut strict pop=0 chains (safe, non-expanding)
        if shortcut_zero_pop_chains(trie3_god, &roots_vec) {
            pass_changed = true;
        }
        // 3) Batched pop=0 composition (conservative, only if edges do not increase)
        if shortcut_zero_pop_chains_batched(trie3_god, &roots_vec) {
            pass_changed = true;
        }
        // 4) Shortcut single-out pop>0 step from middle nodes (safe, non-expanding)
        if shortcut_single_out_pop_step(trie3_god, &roots_vec) {
            pass_changed = true;
        }
        // 5) Remove dominated edges after rewrites
        if remove_dominated_edges_within_nodes(trie3_god, &roots_vec) {
            pass_changed = true;
        }
        // 6) Coalesce again after transformations
        if coalesce_edges_within_nodes(trie3_god, &roots_vec) {
            pass_changed = true;
        }
        if pass_changed {
            any_changed = true;
            crate::debug!(3, "compress_trie3_edges: pass {} applied changes", pass + 1);
        } else {
            break;
        }
    }

    if any_changed {
        crate::debug!(2, "compress_trie3_edges: changes applied; prune/merge/gc will follow in optimize_trie3_size");
    } else {
        crate::debug!(2, "compress_trie3_edges: no changes");
    }
}
