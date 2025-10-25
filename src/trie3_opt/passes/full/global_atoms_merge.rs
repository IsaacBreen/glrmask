use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use ordered_hash_map::OrderedHashMap;
use range_set_blaze::RangeSetBlaze;

use crate::constraint::{LLMTokenBV, PrecomputeNode3Index, StateIDBV, Trie3GodWrapper};
use crate::datastructures::trie::{Trie, Trie2Index};
use crate::datastructures::EntryApi;
use crate::passes::full::atoms::build_global_token_atoms_by_pop;
use crate::passes::full::rewire::rewire_all_edges_to_representatives;

/// NEW MERGE PASS:
/// Global-atoms bisimulation merge. Compared to existing WL / structural merges, this pass:
///  - Derives a global partition of tokens ("atoms") per pop across the entire graph by splitting a
///    universe set with all observed edge masks at that pop (with a cap).
///  - Iteratively refines node classes: two nodes are equivalent iff for every pop and every atom,
///    the aggregated semantics (mapping from destination class to union of SIDs) match.
///  - Rewires all edges to representatives and reconstructs representative edges deterministically.
pub fn merge_nodes_trie3_global_atoms(
    roots: &mut BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    max_llm_token_id: usize,
    max_iters: usize,
    max_atoms_per_pop: usize,
) {
    crate::debug!(2, "Merging nodes (global-atoms bisimulation) in precomputed trie 3 (iters={}, cap={} atoms/pop).",
        max_iters, max_atoms_per_pop);

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    // Derive global atoms by pop.
    let atoms_by_pop =
        build_global_token_atoms_by_pop(trie3_god, &all_nodes, max_llm_token_id, max_atoms_per_pop);
    if atoms_by_pop.is_empty() {
        crate::debug!(3, "Global-atoms merge: no edges present; skipping.");
        return;
    }

    // Dense ids for nodes
    let n = all_nodes.len();
    let mut dense_of: HashMap<Trie2Index, usize> = HashMap::with_capacity(n);
    let mut old_of: Vec<Trie2Index> = Vec::with_capacity(n);
    for (i, idx) in all_nodes.iter().enumerate() {
        dense_of.insert(*idx, i);
        old_of.push(*idx);
    }

    // Ends and raw edges (pop, llm_bv, dest_dense, sids)
    type RawEdge3 = (isize, LLMTokenBV, usize, StateIDBV);
    let mut ends: Vec<bool> = vec![false; n];
    let mut raw_edges: Vec<Vec<RawEdge3>> = vec![Vec::new(); n];
    for (u_dense, u_idx) in old_of.iter().enumerate() {
        let g = u_idx.read(trie3_god).expect("read");
        ends[u_dense] = g.value.end;
        for (ek, dm) in g.children() {
            for (dst, sids) in dm {
                if let Some(v_dense) = dense_of.get(dst) {
                    raw_edges[u_dense].push((ek.0, ek.1.clone(), *v_dense, sids.clone()));
                }
            }
        }
    }

    // Precompute, once, for every unique (pop, LLMTokenBV) mask pointer, which atoms (by index) it overlaps.
    // This avoids scanning all atoms for every edge repeatedly.
    use std::collections::BTreeMap as BTM;
    let mut unique_bvs_per_pop: BTM<isize, Vec<LLMTokenBV>> = BTM::new();
    let mut seen_ptrs_per_pop: BTM<isize, HashSet<*const RangeSetBlaze<usize>>> = BTM::new();
    for edges in &raw_edges {
        for (p, llm_bv, _v, _sids) in edges {
            if !atoms_by_pop.contains_key(p) {
                continue;
            }
            let ptr = Arc::as_ptr(&llm_bv.inner);
            let seen = seen_ptrs_per_pop.entry(*p).or_default();
            if seen.insert(ptr) {
                unique_bvs_per_pop
                    .entry(*p)
                    .or_default()
                    .push(llm_bv.clone());
            }
        }
    }
    let mut atom_idxs_by_pop_ptr: BTM<isize, HashMap<*const RangeSetBlaze<usize>, Vec<usize>>> =
        BTM::new();
    for (p, bvs) in unique_bvs_per_pop {
        if let Some(atoms) = atoms_by_pop.get(&p) {
            let mut map: HashMap<*const RangeSetBlaze<usize>, Vec<usize>> = HashMap::new();
            for bv in bvs {
                let ptr = Arc::as_ptr(&bv.inner);
                let mut idxs: Vec<usize> = Vec::new();
                for (j, a) in atoms.iter().enumerate() {
                    if !(&bv & a).is_empty() {
                        idxs.push(j);
                    }
                }
                map.insert(ptr, idxs);
            }
            atom_idxs_by_pop_ptr.insert(p, map);
        }
    }

    // Compute longest path distance in pop=0 subgraph to prevent merging nodes in a chain.
    let mut pop0_adj: Vec<Vec<usize>> = vec![vec![]; n];
    let mut pop0_rev_adj: Vec<Vec<usize>> = vec![vec![]; n];
    let mut pop0_out_degree = vec![0; n];
    for u in 0..n {
        for (p, _, v_dense, _) in &raw_edges[u] {
            if *p == 0 {
                pop0_adj[u].push(*v_dense);
                pop0_rev_adj[*v_dense].push(u);
                pop0_out_degree[u] += 1;
            }
        }
    }

    let mut dist = vec![0; n];
    let mut q: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        if pop0_out_degree[i] == 0 {
            q.push_back(i);
        }
    }

    let mut processed_count = 0;
    while let Some(v) = q.pop_front() {
        processed_count += 1;
        for &u in &pop0_rev_adj[v] {
            dist[u] = dist[u].max(1 + dist[v]);
            pop0_out_degree[u] -= 1;
            if pop0_out_degree[u] == 0 {
                q.push_back(u);
            }
        }
    }

    // Mark nodes in or that can reach a cycle with max distance
    if processed_count < n {
        let max_dist = n + 1;
        for i in 0..n {
            if pop0_out_degree[i] > 0 {
                q.push_back(i);
            }
        }
        while let Some(v) = q.pop_front() {
            if dist[v] != max_dist {
                dist[v] = max_dist;
                for &u in &pop0_rev_adj[v] {
                    q.push_back(u);
                }
            }
        }
    }

    // Partition refinement (optimized)
    // Initial partition based on end flag and pop=0 distance.
    let mut prev_class: Vec<usize> = vec![0; n];
    let mut class_map: HashMap<(bool, usize), usize> = HashMap::new();
    let mut next_class_id = 0;
    for i in 0..n {
        let key = (ends[i], dist[i]);
        let class_id = *class_map.entry(key).or_insert_with(|| {
            let id = next_class_id;
            next_class_id += 1;
            id
        });
        prev_class[i] = class_id;
    }
    for it in 0..max_iters {
        // Signature (compact): (end_flag, Vec<((pop, atom_idx), Vec<(dest_class, StateIDBV)>)>)
        // Only atoms actually hit by this node are included; canonicalization is via sorted BTreeMaps below.
        type SigKey = (bool, Vec<((isize, usize), Vec<(usize, StateIDBV)>)>);
        let mut sig_to_id: HashMap<SigKey, usize> = HashMap::new();
        let mut next_id = 0usize;
        let mut new_class = vec![0usize; n];
        let mut changes = 0usize;

        #[cfg(not(rustrover))]
        let itn = kdam::tqdm!(
            0..n,
            desc = format!("Trie3 Merge Global-Atoms Iter {}", it + 1),
            total = n,
            disable = !crate::profiler::PROGRESS_BAR_ENABLED,
            leave = true
        );
        #[cfg(rustrover)]
        let itn = 0..n;
        for u in itn {
            // 1) Aggregate once per (pop, LLM mask pointer) to dest_class -> SIDs.
            // This removes duplicate work across atoms for the same mask.
            let mut per_bv_aggr: HashMap<
                (isize, *const RangeSetBlaze<usize>),
                BTreeMap<usize, StateIDBV>,
            > = HashMap::new();
            for (p, llm_bv, v_dense, sids) in &raw_edges[u] {
                if !atoms_by_pop.contains_key(p) {
                    continue;
                }
                let dest_class = prev_class[*v_dense];
                let ptr = Arc::as_ptr(&llm_bv.inner);
                per_bv_aggr
                    .entry((*p, ptr))
                    .or_insert_with(BTreeMap::new)
                    .entry(dest_class)
                    .and_modify(|e| *e |= sids)
                    .or_insert_with(|| sids.clone());
            }

            // 2) Fan out each (pop, mask-pointer) aggregated map to only the atoms it overlaps.
            // Temporary aggregation: (pop, atom_idx) -> BTreeMap<dest_class, SIDs>, sorted for determinism.
            let mut per_atom_aggr: BTreeMap<(isize, usize), BTreeMap<usize, StateIDBV>> =
                BTreeMap::new();
            for ((pop, ptr), dm) in per_bv_aggr {
                if let Some(pop_map) = atom_idxs_by_pop_ptr.get(&pop) {
                    if let Some(atom_idxs) = pop_map.get(&ptr) {
                        for &j in atom_idxs {
                            let entry = per_atom_aggr
                                .entry((pop, j))
                                .or_insert_with(BTreeMap::new);
                            for (dest_class, sids) in &dm {
                                entry
                                    .entry(*dest_class)
                                    .and_modify(|e| *e |= sids)
                                    .or_insert_with(|| sids.clone());
                            }
                        }
                    }
                }
            }

            // 3) Build compact signature. We only include non-empty atoms that this node touches.
            let mut sig_entries: Vec<((isize, usize), Vec<(usize, StateIDBV)>)> =
                Vec::with_capacity(per_atom_aggr.len());
            for (key, m) in per_atom_aggr {
                sig_entries.push((key, m.into_iter().collect()));
            }
            let sig: SigKey = (ends[u], sig_entries);
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
        crate::debug!(
            3,
            "Global-atoms merge iter {}: classes={}, changes={}",
            it + 1,
            new_class.iter().max().map(|m| m + 1).unwrap_or(0),
            changes
        );
        prev_class = new_class;
        if changes == 0 {
            break;
        }
    }

    // Representatives for final classes
    let final_partition = prev_class;
    let num_classes = final_partition.iter().max().map_or(0, |m| m + 1);
    let mut representatives: Vec<Option<Trie2Index>> = vec![None; num_classes];
    for (u_dense, &class_id) in final_partition.iter().enumerate() {
        if representatives[class_id].is_none() {
            representatives[class_id] = Some(old_of[u_dense]);
        }
    }
    let mut node_to_rep: HashMap<Trie2Index, Trie2Index> = HashMap::with_capacity(n);
    for (u_dense, &class_id) in final_partition.iter().enumerate() {
        node_to_rep.insert(old_of[u_dense], representatives[class_id].unwrap());
    }

    // Rewrite all edges in the graph to class representatives
    rewire_all_edges_to_representatives(trie3_god, roots, &node_to_rep);

    // Rebuild representative edges deterministically by aggregating to dest classes and mapping to reps.
    for class_id in 0..num_classes {
        if let Some(rep_idx) = representatives[class_id] {
            // Use a single exemplar node from the class to rebuild representative edges
            let u_dense = final_partition.iter().position(|&c| c == class_id).unwrap();
            // Aggregate by (pop, llm_bv, dest_class) unioning SIDs
            let mut aggr: BTreeMap<(isize, LLMTokenBV, usize), StateIDBV> = BTreeMap::new();
            for (p, llm_bv, v_dense, sids) in &raw_edges[u_dense] {
                let dest_class = final_partition[*v_dense];
                aggr.entry((*p, llm_bv.clone(), dest_class))
                    .and_modify(|e| *e |= sids)
                    .or_insert_with(|| sids.clone());
            }
            let mut new_children: BTreeMap<
                (isize, LLMTokenBV),
                OrderedHashMap<Trie2Index, StateIDBV>,
            > = BTreeMap::new();
            for ((p, bv_key, dest_class), sids) in aggr {
                if let Some(dst_rep) = representatives[dest_class] {
                    new_children
                        .entry((p, bv_key.clone()))
                        .or_insert_with(OrderedHashMap::new)
                        .entry(dst_rep)
                        .and_modify(|e| *e |= &sids)
                        .or_insert(sids);
                }
            }
            // Recompute live tokens
            let mut new_live = LLMTokenBV::zeros();
            for ((_, l), _) in &new_children {
                new_live |= l;
            }
            let mut w = rep_idx.write(trie3_god).expect("write");
            *w.children_mut() = new_children;
            w.value.live_tokens = new_live;
        }
    }

    // Remap roots to representatives
    for r in roots.values_mut() {
        *r = *node_to_rep.get(r).unwrap();
    }
    // Cleanup
    let roots_vec2: Vec<_> = roots.values().cloned().collect();
    Trie::gc(trie3_god, &roots_vec2);
    Trie::recompute_all_max_depths(trie3_god, &roots_vec2);
}
