use std::collections::{BTreeMap, HashMap, HashSet};

use ordered_hash_map::OrderedHashMap;

use crate::constraint::{LLMTokenBV, PrecomputeNode3Index, StateIDBV, Trie3GodWrapper};
use crate::datastructures::EntryApi;
use crate::datastructures::trie::{Trie, Trie2Index};
use crate::passes::full::rewire::rewire_all_edges_to_representatives;
use crate::profiler::PROGRESS_BAR_ENABLED;

/// Extremely fast, cycle-safe node merging using WL-style refinement with a cheap signature
/// (ignoring StateIDBV during coarse iterations) followed by a single exact refinement within
/// candidate equivalence classes that compares aggregated SIDs exactly. Finally, only representatives
/// are rewritten to point to representative destinations; later GC/pruning removes non-reps.
pub fn merge_nodes_trie3_ultrafast(
    roots: &mut BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
) {
    crate::debug!(
        2,
        "Merging nodes (ultrafast WL + exact refine) in precomputed trie 3."
    );

    // Collect all nodes reachable from roots
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    // Build dense index for nodes
    let n = all_nodes.len();
    let mut dense_of: HashMap<Trie2Index, u32> =
        HashMap::with_capacity(n.checked_mul(2).unwrap_or(n));
    let mut old_of: Vec<Trie2Index> = Vec::with_capacity(n);
    for (i, node_idx) in all_nodes.iter().enumerate() {
        dense_of.insert(*node_idx, i as u32);
        old_of.push(*node_idx);
    }

    // First pass: count edges per node and capture ends
    let mut out_counts: Vec<usize> = vec![0; n];
    let mut ends: Vec<u8> = vec![0; n];
    #[cfg(not(rustrover))]
    let it = kdam::tqdm!(
        old_of.iter().enumerate(),
        desc = "Trie3 Merge Ultra (Pass1 count)",
        total = n,
        disable = !PROGRESS_BAR_ENABLED,
        leave = true
    );
    #[cfg(rustrover)]
    let it = old_of.iter().enumerate();
    for (i, node_idx) in it {
        let g = node_idx.read(trie3_god).expect("read");
        ends[i] = if g.value.end { 1 } else { 0 };

        let mut cnt = 0usize;
        for (_ek, dm) in g.children() {
            cnt += dm.len();
        }
        out_counts[i] = cnt;
    }

    // Build offsets for flat edge storage
    let mut offsets: Vec<usize> = vec![0; n + 1];
    for i in 0..n {
        offsets[i + 1] = offsets[i] + out_counts[i];
    }
    let m = offsets[n];

    // LLM bitset pointer-id map (for edge-key LLM masks only)
    use range_set_blaze::RangeSetBlaze;
    let mut llm_id_map: HashMap<*const RangeSetBlaze<usize>, u32> = HashMap::new();
    let mut next_llm_id: u32 = 0;
    let mut get_or_insert_llm_id = |ptr: *const RangeSetBlaze<usize>| -> u32 {
        if let Some(id) = llm_id_map.get(&ptr) {
            *id
        } else {
            let id = next_llm_id;
            next_llm_id = next_llm_id.wrapping_add(1);
            llm_id_map.insert(ptr, id);
            id
        }
    };

    // Flat edge structure: pop, llm_id, dest_dense
    #[derive(Copy, Clone)]
    struct EdgeLight {
        pop: u32,
        llm_id: u32,
        dest: u32,
    }
    let mut edges: Vec<EdgeLight> = vec![EdgeLight {
        pop: 0,
        llm_id: 0,
        dest: 0,
    }; m];

    // Second pass: fill flat edges
    #[cfg(not(rustrover))]
    let it = kdam::tqdm!(
        old_of.iter().enumerate(),
        desc = "Trie3 Merge Ultra (Pass2 fill)",
        total = n,
        disable = !PROGRESS_BAR_ENABLED,
        leave = true
    );
    #[cfg(rustrover)]
    let it = old_of.iter().enumerate();
    for (i, node_idx) in it {
        let g = node_idx.read(trie3_god).expect("read");
        let mut p = offsets[i];
        for (ek, dm) in g.children() {
            let pop_u32 = ek.0 as u32;
            let llm_ptr = std::sync::Arc::as_ptr(&ek.1.inner);
            let llm_id = get_or_insert_llm_id(llm_ptr);
            for (dst, _sids) in dm {
                let dest_dense = *dense_of.get(dst).expect("dense id") as u32;
                edges[p] = EdgeLight {
                    pop: pop_u32,
                    llm_id,
                    dest: dest_dense,
                };
                p += 1;
            }
        }
        debug_assert_eq!(p, offsets[i + 1]);
    }

    // Coarse WL refinement ignoring SIDs
    let max_iters_coarse: usize = 6; // small number of iterations for speed
    let mut prev_class: Vec<u32> = vec![0; n];
    // Initialize coarse classes by (end flag, degree, live_llm_id)
    {
        // Ignore live_tokens entirely: they are derived and must not partition nodes.
        let mut init_map: HashMap<(u8, usize), u32> =
            HashMap::with_capacity(n.checked_div(2).unwrap_or(1024));
        let mut next_c: u32 = 0;
        for i in 0..n {
            let deg = offsets[i + 1] - offsets[i];
            let key = (ends[i], deg);
            let c = init_map.entry(key).or_insert_with(|| {
                let id = next_c;
                next_c = next_c.wrapping_add(1);
                id
            });
            prev_class[i] = *c;
        }
    }

    // Workspace vectors to avoid reallocations
    let mut tmp_items: Vec<(u32, u32, u32)> = Vec::with_capacity(16); // (pop, llm_id, dest_class)
    let mut agg_items: Vec<((u32, u32, u32), u32)> = Vec::with_capacity(16); // ((pop,llm,dest_class), count)
    let mut new_class: Vec<u32> = vec![0; n];
    for it in 0..max_iters_coarse {
        // Phase: compute coarse signature hashes per node
        // We'll use a simple FNV-1a over a sorted aggregated vector
        let mut h_of: Vec<u64> = vec![0; n];
        #[cfg(not(rustrover))]
        let itn = kdam::tqdm!(
            0..n,
            desc = format!("Trie3 Merge Ultra (WL coarse {} / {})", it + 1, max_iters_coarse),
            total = n,
            disable = !PROGRESS_BAR_ENABLED,
            leave = true
        );
        #[cfg(rustrover)]
        let itn = 0..n;
        for u in itn {
            tmp_items.clear();
            let begin = offsets[u];
            let end = offsets[u + 1];
            for idx in begin..end {
                let e = edges[idx];
                let dclass = prev_class[e.dest as usize];
                tmp_items.push((e.pop, e.llm_id, dclass));
            }
            if tmp_items.len() > 1 {
                tmp_items.sort_unstable();
            }
            agg_items.clear();
            let mut i2 = 0usize;
            while i2 < tmp_items.len() {
                let key = tmp_items[i2];
                let mut cnt: u32 = 1;
                i2 += 1;
                while i2 < tmp_items.len() && tmp_items[i2] == key {
                    cnt = cnt.wrapping_add(1);
                    i2 += 1;
                }
                agg_items.push((key, cnt));
            }
            // FNV-1a hash
            let mut h: u64 = 0xcbf29ce484222325;
            // end flag
            h ^= ends[u] as u64;
            h = h.wrapping_mul(0x100000001b3);
            // length
            h ^= agg_items.len() as u64;
            h = h.wrapping_mul(0x100000001b3);
            for &((p, lid, dc), cnt) in &agg_items {
                h ^= p as u64;
                h = h.wrapping_mul(0x100000001b3);
                h ^= lid as u64;
                h = h.wrapping_mul(0x100000001b3);
                h ^= dc as u64;
                h = h.wrapping_mul(0x100000001b3);
                h ^= cnt as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h_of[u] = h;
        }

        // Compress hashes to new classes
        let mut map: HashMap<u64, u32> =
            HashMap::with_capacity(n.checked_div(2).unwrap_or(1024));
        let mut next_c: u32 = 0;
        let mut changes = 0usize;
        for u in 0..n {
            let cid = *map.entry(h_of[u]).or_insert_with(|| {
                let id = next_c;
                next_c = next_c.wrapping_add(1);
                id
            });
            new_class[u] = cid;
            if new_class[u] != prev_class[u] {
                changes += 1;
            }
        }
        crate::debug!(
            3,
            "Trie3 ultrafast coarse iter {}: classes={}, changes={}",
            it + 1,
            map.len(),
            changes
        );
        prev_class = new_class;
        new_class = vec![0; n];
        if changes == 0 {
            break;
        }
    }

    // Now do an exact refinement within each coarse class by aggregating SIDs exactly.
    // Build membership list as sorted pairs (class, node)
    let mut membership: Vec<(u32, u32)> =
        (0..n as u32).map(|u| (prev_class[u as usize], u)).collect();
    membership.sort_unstable_by_key(|x| x.0);

    // We'll produce a node->representative map (dense ids)
    let mut node_to_rep_dense: Vec<u32> = (0..n as u32).collect();

    #[derive(Hash, Eq, PartialEq, Clone)]
    struct KeyNoSids {
        end: u8,
        // Sorted vector of (pop, llm_id, dest_class) triples
        edges: Vec<(u32, u32, u32)>,
    }
    struct Prototype {
        sids: Vec<StateIDBV>, // aligned with edges order
        rep_dense: u32,
    }

    // Iterate groups
    let mut i = 0usize;
    #[cfg(not(rustrover))]
    let total_groups = membership.len();
    #[cfg(not(rustrover))]
    let mut processed_nodes = 0usize;
    while i < membership.len() {
        let class_id = membership[i].0;
        let start = i;
        while i < membership.len() && membership[i].0 == class_id {
            i += 1;
        }
        let end_span = i;
        let span_len = end_span - start;
        if span_len <= 1 {
            // Single node - it is its own representative
            let u_dense = membership[start].1;
            node_to_rep_dense[u_dense as usize] = u_dense;
            #[cfg(not(rustrover))]
            {
                processed_nodes += 1;
            }
            continue;
        }

        // Build prototype map keyed by (end, live, keys_without_sids)
        let mut key_map: HashMap<KeyNoSids, Vec<Prototype>> = HashMap::new();
        #[cfg(not(rustrover))]
        let it = kdam::tqdm!(
            start..end_span,
            desc = "Trie3 Merge Ultra (Exact refine group)",
            disable = !PROGRESS_BAR_ENABLED,
            leave = false
        );
        #[cfg(rustrover)]
        let it = start..end_span;
        for idx in it {
            let u_dense = membership[idx].1 as usize;
            let node_idx = old_of[u_dense];
            let g = node_idx.read(trie3_god).expect("read");

            // Aggregate SIDs per (pop, llm_id, coarse_dest_class)
            let mut aggr: BTreeMap<(u32, u32, u32), StateIDBV> = BTreeMap::new();
            for (ek, dm) in g.children() {
                let pop_u32 = ek.0 as u32;
                let llm_ptr = std::sync::Arc::as_ptr(&ek.1.inner);
                let llm_id = *llm_id_map.get(&llm_ptr).expect("llm_id present");
                for (dst, sids) in dm {
                    let dest_dense = *dense_of.get(dst).expect("dense of dst") as usize;
                    let coarse_dest_class = prev_class[dest_dense];
                    aggr.entry((pop_u32, llm_id, coarse_dest_class))
                        .and_modify(|v| *v |= sids)
                        .or_insert_with(|| sids.clone());
                }
            }

            // Build key without sids
            let mut keys_vec: Vec<(u32, u32, u32)> = aggr.keys().cloned().collect();
            // BTreeMap iteration is already sorted; we can rely on that
            let key = KeyNoSids {
                end: ends[u_dense],
                edges: keys_vec.clone(),
            };
            let sids_vec: Vec<StateIDBV> = aggr.into_values().collect();

            // Try to match an existing prototype
            let entry = key_map.entry(key).or_insert_with(Vec::new);
            let mut found = None;
            for proto in entry.iter() {
                if proto.sids.len() != sids_vec.len() {
                    continue;
                }
                let mut ok = true;
                for (a, b) in proto.sids.iter().zip(sids_vec.iter()) {
                    if a != b {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    found = Some(proto.rep_dense);
                    break;
                }
            }
            if let Some(rep) = found {
                node_to_rep_dense[u_dense] = rep;
            } else {
                // New prototype
                let rep = u_dense as u32;
                node_to_rep_dense[u_dense] = rep;
                entry.push(Prototype {
                    sids: sids_vec,
                    rep_dense: rep,
                });
            }
            #[cfg(not(rustrover))]
            {
                processed_nodes += 1;
            }
        }
    }
    #[cfg(not(rustrover))]
    {
        let _ = processed_nodes; // avoid unused warning
    }

    // Representatives (unique)
    let mut rep_set: HashSet<u32> = HashSet::new();
    for (u_dense, &rep) in node_to_rep_dense.iter().enumerate() {
        if u_dense as u32 == rep {
            rep_set.insert(rep);
        }
    }

    // Build a mapping from node index to its representative index for rewiring.
    let mut node_to_rep_map: HashMap<Trie2Index, Trie2Index> = HashMap::with_capacity(n);
    for u_dense in 0..n {
        let u_idx = old_of[u_dense];
        let rep_dense = node_to_rep_dense[u_dense] as usize;
        let rep_idx = old_of[rep_dense];
        node_to_rep_map.insert(u_idx, rep_idx);
    }
    // Rewire all graph edges to point to representatives.
    rewire_all_edges_to_representatives(trie3_god, roots, &node_to_rep_map);

    // Rewrite representatives' children to point to representatives and recompute live tokens
    #[cfg(not(rustrover))]
    let it = kdam::tqdm!(
        rep_set.iter(),
        desc = "Trie3 Merge Ultra (Rewrite reps)",
        total = rep_set.len(),
        disable = !PROGRESS_BAR_ENABLED,
        leave = true
    );
    #[cfg(rustrover)]
    let it = rep_set.iter();
    for rep_dense in it {
        let rep_idx = old_of[*rep_dense as usize];
        let mut w = rep_idx.write(trie3_god).expect("write");
        let mut new_children: BTreeMap<
            (isize, LLMTokenBV),
            OrderedHashMap<Trie2Index, StateIDBV>,
        > = BTreeMap::new();

        // Build new children by remapping destinations to their representatives
        for (ek, dm) in w.children().clone() {
            let (pop, llm_bv) = ek;
            let mut dest_map = OrderedHashMap::new();
            for (dst, sids) in dm {
                let dst_dense = *dense_of.get(&dst).expect("dense of dst") as usize;
                let rep_dst_dense = node_to_rep_dense[dst_dense] as usize;
                let rep_dst_idx = old_of[rep_dst_dense];
                dest_map
                    .entry(rep_dst_idx)
                    .and_modify(|v| *v |= &sids)
                    .or_insert(sids);
            }
            if !dest_map.is_empty() {
                new_children.insert((pop, llm_bv), dest_map);
            }
        }
        // Commit the rewritten children and recompute live_tokens
        let mut new_live = LLMTokenBV::zeros();
        for ((_, llm_bv), _) in &new_children {
            new_live |= llm_bv;
        }
        w.value.live_tokens = new_live;
        *w.children_mut() = new_children;
    }

    // Remap roots to their representatives
    for root_idx in roots.values_mut() {
        if let Some(dense) = dense_of.get(root_idx) {
            let rep_dense = node_to_rep_dense[*dense as usize] as usize;
            *root_idx = old_of[rep_dense];
        }
    }

    // Finalize
    let final_roots_vec: Vec<_> = roots.values().cloned().collect();
    Trie::gc(trie3_god, &final_roots_vec);
    Trie::recompute_all_max_depths(trie3_god, &final_roots_vec);
    crate::debug!(
        2,
        "Ultrafast merge completed: representatives kept = {}",
        rep_set.len()
    );
}

pub fn merge_nodes_trie3_fast(
    roots: &mut BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
) {
    merge_nodes_trie3_impl(roots, trie3_god, 2);
}

pub fn merge_nodes_trie3(
    roots: &mut BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    max_iters: usize,
) {
    merge_nodes_trie3_impl(roots, trie3_god, max_iters);
}

fn merge_nodes_trie3_impl(
    roots: &mut BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    max_iters: usize,
) {
    crate::debug!(
        2,
        "Merging identical subtrees in precomputed trie 3 (max_iters={}).",
        max_iters
    );

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    // NOTE: This pass already ignores 'live_tokens' (derived) in its signature.
    // The new global-atoms pass complements this by aligning semantics globally
    // across nodes for each pop and token atom.

    let mut dense_of: HashMap<Trie2Index, usize> = HashMap::new();
    let mut old_of: Vec<Trie2Index> = Vec::with_capacity(all_nodes.len());
    for (i, node_idx) in all_nodes.iter().enumerate() {
        dense_of.insert(*node_idx, i);
        old_of.push(*node_idx);
    }
    let n = all_nodes.len();

    let mut ends: Vec<bool> = vec![false; n];
    type RawEdge3 = (isize, LLMTokenBV, usize, StateIDBV);
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

    let mut it: usize = 0;
    loop {
        type AggregatedEdge3 = ((isize, LLMTokenBV, usize), StateIDBV);
        type Signature3 = (bool, Vec<AggregatedEdge3>);

        let mut sig_to_id: HashMap<Signature3, usize> = HashMap::new();
        let mut new_class = vec![0; n];
        let mut next_id = 0;
        let mut changes = 0;

        #[cfg(not(rustrover))]
        let its = kdam::tqdm!(
            0..n,
            desc = format!("Trie3 Merge Iter {}", it + 1),
            total = n,
            disable = !PROGRESS_BAR_ENABLED,
            leave = true
        );
        #[cfg(rustrover)]
        let its = 0..n;
        for u in its {
            let mut aggr: BTreeMap<(isize, LLMTokenBV, usize), StateIDBV> = BTreeMap::new();
            for (p, bv_key, v_dense, sids) in &raw_edges[u] {
                let dest_class = prev_class[*v_dense];
                let key = (*p, bv_key.clone(), dest_class);
                aggr.entry(key)
                    .and_modify(|e| *e |= sids)
                    .or_insert_with(|| sids.clone());
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

        crate::debug!(
            3,
            "Trie3 merge iter {}: classes={}, changes={}",
            it + 1,
            next_id,
            changes
        );
        prev_class = new_class;
        if changes == 0 {
            break;
        }
        it += 1;
        if max_iters > 0 && it >= max_iters {
            break;
        }
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

    // Rewrite all edges in the graph to point to the representatives.
    rewire_all_edges_to_representatives(trie3_god, roots, &node_to_rep);

    for class_id in 0..num_classes {
        if let Some(rep_idx) = representatives[class_id] {
            let u_dense = final_partition.iter().position(|&c| c == class_id).unwrap();

            let mut aggr: BTreeMap<(isize, LLMTokenBV, usize), StateIDBV> = BTreeMap::new();
            for (p, bv_key, v_dense, sids) in &raw_edges[u_dense] {
                let dest_class = final_partition[*v_dense];
                aggr.entry((*p, bv_key.clone(), dest_class))
                    .and_modify(|e| *e |= sids)
                    .or_insert_with(|| sids.clone());
            }

            let mut new_children = BTreeMap::new();
            for ((p, bv_key, dest_class), sids) in aggr {
                if let Some(dest_rep_idx) = representatives[dest_class] {
                    new_children
                        .entry((p, bv_key.clone()))
                        .or_insert_with(OrderedHashMap::new)
                        .insert(dest_rep_idx, sids);
                }
            }

            // Recompute live tokens from the new merged edges.
            let mut new_live_tokens = LLMTokenBV::zeros();
            for ((_, llm_bv), _) in &new_children {
                new_live_tokens |= llm_bv;
            }

            let mut guard = rep_idx.write(trie3_god).unwrap();
            *guard.children_mut() = new_children;
            guard.value.live_tokens = new_live_tokens;
        }
    }

    for root_idx in roots.values_mut() {
        *root_idx = *node_to_rep.get(root_idx).unwrap();
    }

    // GC unreachable non-representatives and recompute depths.
    let final_roots_vec: Vec<_> = roots.values().cloned().collect();
    Trie::gc(trie3_god, &final_roots_vec);
    Trie::recompute_all_max_depths(trie3_god, &final_roots_vec);
}

pub fn merge_nodes_trie3_structural(
    roots: &mut BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    max_iters: usize,
) {
    crate::debug!(
        2,
        "Merging structurally equivalent subtrees in precomputed trie 3 (max_iters={}).",
        max_iters
    );

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    let mut dense_of: HashMap<Trie2Index, usize> = HashMap::new();
    let mut old_of: Vec<Trie2Index> = Vec::with_capacity(all_nodes.len());
    for (i, node_idx) in all_nodes.iter().enumerate() {
        dense_of.insert(*node_idx, i);
        old_of.push(*node_idx);
    }
    let n = all_nodes.len();

    let mut ends: Vec<bool> = vec![false; n];
    type RawEdge3 = (isize, LLMTokenBV, usize, StateIDBV);
    let mut raw_edges: Vec<Vec<RawEdge3>> = vec![Vec::new(); n];

    for (u_dense, u_idx) in old_of.iter().enumerate() {
        let guard = u_idx.read(trie3_god).unwrap();
        ends[u_dense] = guard.value.end;
        for (ek, dest_map) in guard.children() {
            for (v_idx, bv) in dest_map {
                if let Some(&v_dense) = dense_of.get(v_idx) {
                    // Note: we capture the LLMTokenBV here but won't use it in the signature.
                    raw_edges[u_dense].push((ek.0, ek.1.clone(), v_dense, bv.clone()));
                }
            }
        }
    }

    let mut prev_class: Vec<usize> = (0..n).map(|i| if ends[i] { 1 } else { 0 }).collect();

    let mut it: usize = 0;
    loop {
        // Signature is (end_flag, aggregated edges keyed by (pop, llm_bv, dest_class)).
        // Including the LLMTokenBV ensures we don't merge nodes that differ on token distributions.
        type SignatureStructural3 = (bool, Vec<((isize, LLMTokenBV, usize), StateIDBV)>);

        let mut sig_to_id: HashMap<SignatureStructural3, usize> = HashMap::new();
        let mut new_class = vec![0; n];
        let mut next_id = 0;
        let mut changes = 0;

        #[cfg(not(rustrover))]
        let its = kdam::tqdm!(
            0..n,
            desc = format!("Trie3 Merge Structural Iter {}", it + 1),
            total = n,
            disable = !PROGRESS_BAR_ENABLED,
            leave = true
        );
        #[cfg(rustrover)]
        let its = 0..n;
        for u in its {
            let mut aggr: BTreeMap<(isize, LLMTokenBV, usize), StateIDBV> = BTreeMap::new();
            for (p, bv_key, v_dense, sids) in &raw_edges[u] {
                let dest_class = prev_class[*v_dense];
                let key = (*p, bv_key.clone(), dest_class);
                aggr.entry(key)
                    .and_modify(|e| *e |= sids)
                    .or_insert_with(|| sids.clone());
            }

            let agg_edges: Vec<((isize, LLMTokenBV, usize), StateIDBV)> = aggr.into_iter().collect();
            let sig: SignatureStructural3 = (ends[u], agg_edges);

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
            "Trie3 structural merge iter {}: classes={}, changes={}",
            it + 1,
            next_id,
            changes
        );
        prev_class = new_class;
        if changes == 0 {
            break;
        }
        it += 1;
        if max_iters > 0 && it >= max_iters {
            break;
        }
    }

    // Reconstruction
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

    // Rewrite all edges in the graph to point to the representatives.
    rewire_all_edges_to_representatives(trie3_god, roots, &node_to_rep);

    for class_id in 0..num_classes {
        if let Some(rep_idx) = representatives[class_id] {
            // Use a single exemplar node from the class to rebuild the representative's edges
            // based on the final partition (safe and deterministic).
            let u_dense = final_partition.iter().position(|&c| c == class_id).unwrap();

            // Aggregate edges by (pop, llm_bv, dest_class)
            let mut aggr: BTreeMap<(isize, LLMTokenBV, usize), StateIDBV> = BTreeMap::new();
            for (p, bv_key, v_dense, sids) in &raw_edges[u_dense] {
                let dest_class = final_partition[*v_dense];
                aggr.entry((*p, bv_key.clone(), dest_class))
                    .and_modify(|e| *e |= sids)
                    .or_insert_with(|| sids.clone());
            }

            // Build representative children by mapping dest classes to their representatives.
            let mut new_children: BTreeMap<
                (isize, LLMTokenBV),
                OrderedHashMap<Trie2Index, StateIDBV>,
            > = BTreeMap::new();
            for ((p, bv_key, dest_class), sids) in aggr {
                if let Some(dest_rep_idx) = representatives[dest_class] {
                    let dm = new_children
                        .entry((p, bv_key.clone()))
                        .or_insert_with(OrderedHashMap::new);
                    dm.entry(dest_rep_idx)
                        .and_modify(|e| *e |= &sids)
                        .or_insert(sids);
                }
            }

            // Recompute live tokens from the new merged edges.
            let mut new_live_tokens = LLMTokenBV::zeros();
            for ((_, llm_bv), _) in &new_children {
                new_live_tokens |= llm_bv;
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
    Trie::gc(trie3_god, &final_roots_vec);
    Trie::recompute_all_max_depths(trie3_god, &final_roots_vec);
}
