use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::constraint::{IntermediateTrie3GodWrapper, PrecomputeNode3Index};
use crate::datastructures::gss_leveled_adapter::{map_trie3_node_ids, GSSNode};
use crate::datastructures::trie::{Trie, Trie2Index};
use crate::glr::table::{NonTerminalID, TerminalID};
use ordered_hash_map::OrderedHashMap;

// Key for the stored cache
pub type StoredCacheKey = (NonTerminalID, TerminalID);
// Value for the stored cache: (Trie Root, GSS Root)
pub type StoredCacheValue = (PrecomputeNode3Index, Arc<GSSNode>);

/// Performs structural deduplication (merging) on the stored trie graph
/// and updates the cache entries accordingly.
pub fn optimize_stored_cache(
    cache: &mut HashMap<StoredCacheKey, StoredCacheValue>,
    trie3_god: &IntermediateTrie3GodWrapper,
    max_iters: usize,
) {
    if cache.is_empty() || max_iters == 0 {
        return;
    }

    // 1. Collect all unique trie roots to define the graph scope.
    let unique_roots: HashSet<PrecomputeNode3Index> = cache.values().map(|(r, _)| *r).collect();
    let roots_vec: Vec<PrecomputeNode3Index> = unique_roots.into_iter().collect();
    
    // 2. Perform WL refinement on the reachable graph to find structural equivalence classes.
    let node_to_rep = merge_trie3_nodes_and_get_map_internal(trie3_god, &roots_vec, max_iters);

    if node_to_rep.is_empty() {
        return;
    }
    
    // 3. Update cache entries using the Trie mapping.
    for (_key, (trie_root, gss_root)) in cache.iter_mut() {
        // Update Trie Root
        if let Some(rep) = node_to_rep.get(trie_root) {
            *trie_root = *rep;
        }
        // Update GSS Root (remap internal trie node indices stored in Accs)
        map_trie3_node_ids(gss_root, &node_to_rep);
    }

    // 4. GSS Normalization and Sharing
    let gss_roots_vec: Vec<Arc<GSSNode>> = cache.values().map(|(_, gss)| gss.clone()).collect();
    let canonical_gss_roots = GSSNode::normalize_many(gss_roots_vec.clone());

    let mut gss_ptr_to_canonical: HashMap<*const GSSNode, Arc<GSSNode>> = HashMap::new();
    for (old_arc, new_arc) in gss_roots_vec.iter().zip(canonical_gss_roots.into_iter()) {
        gss_ptr_to_canonical.insert(Arc::as_ptr(old_arc), new_arc);
    }

    // Update cache entries with canonical GSS roots
    for (_key, (_trie_root, gss_root)) in cache.iter_mut() {
        let old_ptr = Arc::as_ptr(gss_root);
        if let Some(canonical_arc) = gss_ptr_to_canonical.get(&old_ptr) {
            *gss_root = canonical_arc.clone();
        }
    }

    // 5. Final cleanup: GC unreachable nodes.
    let final_roots: Vec<PrecomputeNode3Index> = cache.values().map(|(r, _)| *r).collect();
    Trie::gc(trie3_god, &final_roots);
    Trie::recompute_all_max_depths(trie3_god, &final_roots);
}

// --- Internal WL Refinement Implementation (adapted from Trie3 optimization) ---

/// Performs WL refinement on the graph defined by `roots_vec` and returns the mapping.
fn merge_trie3_nodes_and_get_map_internal(
    trie3_god: &IntermediateTrie3GodWrapper,
    roots_vec: &[PrecomputeNode3Index],
    max_iters: usize,
) -> HashMap<Trie2Index, Trie2Index> {
    let all_nodes = Trie::all_nodes(trie3_god, roots_vec);
    if all_nodes.is_empty() { return HashMap::new(); }

    let n = all_nodes.len();
    let mut dense_of: HashMap<Trie2Index, usize> = HashMap::with_capacity(n);
    let mut old_of: Vec<Trie2Index> = Vec::with_capacity(n);
    for (i, node_idx) in all_nodes.iter().enumerate() {
        dense_of.insert(*node_idx, i);
        old_of.push(*node_idx);
    }

    let mut ends: Vec<bool> = vec![false; n];
    type RawEdge3 = (crate::constraint::IntermediateTrie3EdgeKey, usize);
    let mut raw_edges: Vec<Vec<RawEdge3>> = vec![Vec::new(); n];

    for (u_dense, u_idx) in old_of.iter().enumerate() {
        let guard = u_idx.read(trie3_god).unwrap();
        ends[u_dense] = guard.value.end;
        for (ek, dest_map) in guard.children() {
            for (v_idx, _bv) in dest_map {
                if let Some(&v_dense) = dense_of.get(v_idx) {
                    raw_edges[u_dense].push((ek.clone(), v_dense));
                }
            }
        }
    }

    // Initial classification based on end status
    let mut prev_class: Vec<usize> = (0..n).map(|i| if ends[i] { 1 } else { 0 }).collect();

    for it in 0..max_iters {
        // Signature: (end_flag, sorted_list_of_aggregated_edges)
        // Aggregated edge: ((pop, LLMTokenBV, dest_class), StateIDBV)
        type Signature3 = (bool, Vec<(crate::constraint::IntermediateTrie3EdgeKey, usize)>);

        let mut sig_to_id: HashMap<Signature3, usize> = HashMap::new();
        let mut new_class = vec![0; n];
        let mut next_id = 0;
        let mut changes = 0;

        #[cfg(not(rustrover))]
        let its = kdam::tqdm!(0..n, desc = format!("Stored Cache Merge Iter {}", it + 1), total = n, disable = !crate::profiler::PROGRESS_BAR_ENABLED, leave = true);
        #[cfg(rustrover)]
        let its = 0..n;
        for u in its {
            let mut agg_edges: Vec<(crate::constraint::IntermediateTrie3EdgeKey, usize)> = raw_edges[u]
                .iter()
                .map(|(ek, v_dense)| (ek.clone(), prev_class[*v_dense]))
                .collect();
            agg_edges.sort();
            agg_edges.dedup();

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

        prev_class = new_class;
        if changes == 0 { break; }
    }

    // 3. Build node_to_rep map and rewrite representatives.
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

    // Rewrite representatives' children and live tokens
    for class_id in 0..num_classes {
        if let Some(rep_idx) = representatives[class_id] {
            let mut new_children = BTreeMap::new();
            // Collect all edges from all nodes in the class
            for (i, &c) in final_partition.iter().enumerate() {
                if c == class_id {
                    let u_idx = old_of[i];
                    let guard = u_idx.read(trie3_god).unwrap();
                    for (ek, dest_map) in guard.children() {
                        for (v_idx, _ev) in dest_map {
                            let v_dense = dense_of[v_idx];
                            let dest_class = final_partition[v_dense];
                            if let Some(dest_rep_idx) = representatives[dest_class] {
                                new_children.entry(ek.clone()).or_insert_with(OrderedHashMap::new).insert(dest_rep_idx, ());
                            }
                        }
                    }
                }
            }

            let mut guard = rep_idx.write(trie3_god).unwrap();
            *guard.children_mut() = new_children;
        }
    }

    node_to_rep
}
