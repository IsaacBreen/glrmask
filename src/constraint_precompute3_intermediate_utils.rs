// src/constraint_precompute3_intermediate_utils.rs
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use ordered_hash_map::OrderedHashMap;
use crate::constraint::{IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey, IntermediateTrie3GodWrapper};
use crate::datastructures::ordered_hash_map::Retain;
use crate::datastructures::trie::{Trie, Trie2Index};
use crate::r#macro::is_debug_level_enabled;

pub fn optimize_intermediate_trie3_template(
    start_node: &IntermediatePrecomputeNode3Index,
    end_node: &IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
) {
    // A few passes of optimization.
    for _ in 0..2 {
        let changed = prune_unproductive_nodes(&[*start_node], end_node, god);
        // GC is needed to remove nodes that become unreachable after pruning edges.
        // NOTE: GC is disabled here because it was causing issues with multiple templates
        // in the same arena. Dangling nodes are acceptable for now.
        // Trie::gc(god, &[*start_node]);
        if !changed {
            break;
        }
    }
}

pub fn optimize_intermediate_trie3(
    roots: &[IntermediatePrecomputeNode3Index],
    end_node: &IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
) {
    if is_debug_level_enabled(2) {
        let mut stats = crate::constraint_extra::PrecomputeStats::default();
        crate::constraint_extra::calculate_intermediate_stats3(roots, &mut stats, god);
        crate::constraint_extra::print_intermediate_stats3(&stats, god);
    }
    for _ in 0..2 {
        let changed = prune_unproductive_nodes(roots, end_node, god);
        if !changed {
            break;
        }
    }
}

/// Prunes nodes in a graph that cannot reach the specified `end_node`.
/// Returns true if any edges were pruned.
fn prune_unproductive_nodes(
    start_nodes: &[IntermediatePrecomputeNode3Index],
    end_node: &IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
) -> bool {
    let all_nodes_vec = Trie::all_nodes(god, start_nodes);
    if all_nodes_vec.is_empty() {
        return false;
    }
    let all_nodes_in_subgraph: HashSet<_> = all_nodes_vec.into_iter().collect();

    // Build reverse adjacency map for the subgraph
    let mut incoming: HashMap<IntermediatePrecomputeNode3Index, Vec<IntermediatePrecomputeNode3Index>> = HashMap::new();
    for src in &all_nodes_in_subgraph {
        if let Some(g) = src.read(god) {
            for (_ek, dm) in g.children() {
                for (dst, _) in dm {
                    // Only consider edges within the subgraph
                    if all_nodes_in_subgraph.contains(dst) {
                        incoming.entry(*dst).or_default().push(*src);
                    }
                }
            }
        }
    }

    // Reverse BFS from end_node to find all productive nodes
    let mut productive: HashSet<IntermediatePrecomputeNode3Index> = HashSet::new();
    let mut q: VecDeque<IntermediatePrecomputeNode3Index> = VecDeque::new();

    if all_nodes_in_subgraph.contains(end_node) {
        productive.insert(*end_node);
        q.push_back(*end_node);
    }

    while let Some(d) = q.pop_front() {
        if let Some(srcs) = incoming.get(&d) {
            for s in srcs {
                if productive.insert(*s) {
                    q.push_back(*s);
                }
            }
        }
    }

    let prunable_count = all_nodes_in_subgraph.len() - productive.len();
    if prunable_count == 0 {
        return false;
    }

    let mut changed = false;
    // Remove any edge pointing to a non-productive destination
    for n in &all_nodes_in_subgraph {
        if !productive.contains(n) {
            continue; // This node will be GC'd anyway, no need to edit its edges.
        }
        if let Some(mut w) = n.write(god) {
            let original_edge_count: usize = w.children().values().map(|dm| dm.len()).sum();
            w.children_mut().retain(|_ek, dm| {
                dm.retain(|dst, _| productive.contains(dst));
                !dm.is_empty()
            });
            let new_edge_count: usize = w.children().values().map(|dm| dm.len()).sum();
            if new_edge_count < original_edge_count {
                changed = true;
            }
        }
    }

    changed
}

pub fn merge_intermediate_nodes_globally(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
    max_iters: usize,
) -> HashMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> {
    let all_nodes = Trie::all_nodes(god, roots);
    if all_nodes.is_empty() { return HashMap::new(); }

    let mut dense_of: HashMap<IntermediatePrecomputeNode3Index, usize> = HashMap::new();
    let mut old_of: Vec<IntermediatePrecomputeNode3Index> = Vec::with_capacity(all_nodes.len());
    for (i, node_idx) in all_nodes.iter().enumerate() {
        dense_of.insert(*node_idx, i);
        old_of.push(*node_idx);
    }
    let n = all_nodes.len();

    let mut ends: Vec<bool> = vec![false; n];
    type RawEdgeIntermediate = (IntermediateTrie3EdgeKey, usize); // (edge_key, dest_dense)
    let mut raw_edges: Vec<Vec<RawEdgeIntermediate>> = vec![Vec::new(); n];

    for (u_dense, u_idx) in old_of.iter().enumerate() {
        let guard = u_idx.read(god).unwrap();
        ends[u_dense] = guard.value.end;
        for (ek, dest_map) in guard.children() {
            for (v_idx, _) in dest_map {
                if let Some(&v_dense) = dense_of.get(v_idx) {
                    raw_edges[u_dense].push((ek.clone(), v_dense));
                }
            }
        }
    }

    let mut prev_class: Vec<usize> = (0..n).map(|i| if ends[i] { 1 } else { 0 }).collect();

    for _it in 0..max_iters {
        type SignatureIntermediate = (bool, Vec<(IntermediateTrie3EdgeKey, usize)>);

        let mut sig_to_id: HashMap<SignatureIntermediate, usize> = HashMap::new();
        let mut new_class = vec![0; n];
        let mut next_id = 0;
        let mut changes = 0;

        for u in 0..n {
            let mut edges_for_sig: Vec<(IntermediateTrie3EdgeKey, usize)> = raw_edges[u]
                .iter()
                .map(|(ek, v_dense)| (ek.clone(), prev_class[*v_dense]))
                .collect();
            edges_for_sig.sort_unstable();

            let sig: SignatureIntermediate = (ends[u], edges_for_sig);

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

    let final_partition = prev_class;
    let num_classes = final_partition.iter().max().map_or(0, |m| m + 1);

    let mut representatives: Vec<Option<IntermediatePrecomputeNode3Index>> = vec![None; num_classes];
    for (u_dense, &class_id) in final_partition.iter().enumerate() {
        if representatives[class_id].is_none() {
            representatives[class_id] = Some(old_of[u_dense]);
        }
    }

    let mut node_to_rep: HashMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> = HashMap::new();
    for (u_dense, &class_id) in final_partition.iter().enumerate() {
        node_to_rep.insert(old_of[u_dense], representatives[class_id].unwrap());
    }

    // Rewrite representatives
    for class_id in 0..num_classes {
        if let Some(rep_idx) = representatives[class_id] {
            let mut guard = rep_idx.write(god).unwrap();
            let old_children = std::mem::take(guard.children_mut());
            let mut new_children = BTreeMap::new();

            for (ek, dest_map) in old_children {
                let new_dest_map: &mut OrderedHashMap<Trie2Index, ()> = new_children.entry(ek).or_default();
                for (old_dest, val) in dest_map {
                    let new_dest = *node_to_rep.get(&old_dest).unwrap();
                    // The value is `()`, so we can just insert. If there are duplicates, they are fine.
                    // `insert` will just overwrite, which is ok since value is `()`.
                    new_dest_map.insert(new_dest, val);
                }
            }
            *guard.children_mut() = new_children;
        }
    }

    node_to_rep
}
