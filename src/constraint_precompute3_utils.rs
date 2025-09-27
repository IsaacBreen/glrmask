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
    max_state_id: usize,
    max_llm_token_id: usize,
) {
    crate::debug!(2, "Optimizing Trie 3 size...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let _all_nodes_pinner = Trie::all_nodes(&trie3_god, &roots_vec);

    if config.optimize_trie3_constrain_bitvecs {
        constrain_bitvecs_trie3(trie3_god, &roots_vec, max_state_id, max_llm_token_id);
    }

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

fn constrain_bitvecs_trie3(
    trie3_god: &Trie3GodWrapper,
    roots_vec: &[PrecomputeNode3Index],
    max_state_id: usize,
    max_llm_token_id: usize,
) {
    crate::debug!(2, "Constraining bitvectors in Trie 3...");
    let all_nodes = Trie::all_nodes(trie3_god, roots_vec);
    if all_nodes.is_empty() { return; }

    for node_arc in all_nodes {
        let mut guard = node_arc.write(trie3_god).unwrap();

        // Constrain live_tokens on the node value
        guard.value.live_tokens.constrain(max_llm_token_id);

        let old_children = std::mem::take(guard.children_mut());
        let mut new_children = BTreeMap::new();

        for ((pop, mut llm_bv), dest_map) in old_children {
            llm_bv.constrain(max_llm_token_id);

            let mut new_dest_map = OrderedHashMap::new();
            for (dest_wrapper, mut sids_bv) in dest_map {
                sids_bv.constrain(max_state_id);
                if !sids_bv.is_empty() {
                    new_dest_map.insert(dest_wrapper, sids_bv);
                }
            }

            if !llm_bv.is_empty() && !new_dest_map.is_empty() {
                // Need to merge if the key (with constrained llm_bv) already exists
                let entry = new_children.entry((pop, llm_bv)).or_insert_with(OrderedHashMap::new);
                for (dest, sids) in new_dest_map {
                    entry.entry(dest)
                        .and_modify(|existing_sids| *existing_sids |= &sids)
                        .or_insert(sids);
                }
            }
        }
        *guard.children_mut() = new_children;
    }
    crate::debug!(2, "Finished constraining bitvectors.");
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

    // Helper: is the LLM-token BV "all tokens"?
    fn is_all_llm(bv: &LLMTokenBV) -> bool {
        bv == &LLMTokenBV::max_ones()
    }
    // Helper: is the StateIDBV "all states"?
    fn is_all_sids(bv: &StateIDBV) -> bool {
        bv == &StateIDBV::max_ones()
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

    // Pass 3: shortcut when the first edge is "universal" and the middle has a single outgoing edge.
    // A --(p1, ALL_LLM, ALL_SID)--> B and B --(p2, L2, SID2)--> C (only outgoing)
    // becomes A --(p1+p2, L2, SID2)--> C. (Do not apply when p2 == 0; zero-pop handled by pass 2.)
    fn shortcut_universal_pop_step(trie3_god: &Trie3GodWrapper, roots_vec: &[PrecomputeNode3Index]) -> bool {
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
            if *p2 == 0 { continue; } // leave to zero-pop pass
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
                // Only when the first edge is universal in both LLM and SIDs.
                if !is_all_llm(llm1) {
                    continue;
                }
                for (v, sids1) in dests {
                    if !is_all_sids(sids1) {
                        continue;
                    }
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
                        // Insert U --(p1+p2, llm2)--> C with sids2
                        let key_new = (p1 + p2, llm2);
                        let dest_map = w.children_mut().entry(key_new).or_default();
                        dest_map.entry(c)
                            .and_modify(|s| *s |= &sids2)
                            .or_insert(sids2);
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
        // 2) Shortcut pop=0 chains (safe, non-expanding)
        if shortcut_zero_pop_chains(trie3_god, &roots_vec) {
            pass_changed = true;
        }
        // 3) Shortcut universal-first edges by adding pops (safe, non-expanding)
        if shortcut_universal_pop_step(trie3_god, &roots_vec) {
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
