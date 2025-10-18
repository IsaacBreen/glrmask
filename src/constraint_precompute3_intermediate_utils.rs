// src/constraint_precompute3_intermediate_utils.rs
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use ordered_hash_map::OrderedHashMap;
use crate::constraint::{IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey, IntermediateTrie3GodWrapper};
use crate::datastructures::ordered_hash_map::Retain;
use crate::datastructures::trie::{Trie, Trie2Index};
use crate::r#macro::is_debug_level_enabled;
use crate::tokenizer::TokenizerStateID;

pub fn optimize_intermediate_trie3_template(
    mut start_node: IntermediatePrecomputeNode3Index,
    mut end_node: IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
) -> (IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index) {
    // A few passes of optimization.
    for _ in 0..3 {
        let changed1 = prune_unproductive_nodes(&[start_node.clone()], &end_node, god);

        let (node_map, reps) = merge_nodes_intermediate_trie3(&[start_node.clone()], god, 40);
        let changed2 = node_map.len() > reps.len();
        start_node = *node_map.get(&start_node).unwrap_or(&start_node);
        end_node = *node_map.get(&end_node).unwrap_or(&end_node);

        if !changed1 && !changed2 {
            break;
        }
    }
    (start_node, end_node)
}

pub fn optimize_intermediate_trie3(
    roots: &mut BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    end_node: &IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
) {
    if is_debug_level_enabled(2) {
        let roots_vec: Vec<_> = roots.values().cloned().collect();
        let mut stats = crate::constraint_extra::PrecomputeStats::default();
        crate::constraint_extra::calculate_intermediate_stats3(&roots_vec, &mut stats, god);
        crate::constraint_extra::print_intermediate_stats3(&stats, god);
    }
    for _ in 0..2 {
        let roots_vec: Vec<_> = roots.values().cloned().collect();
        let changed = prune_unproductive_nodes(&roots_vec, end_node, god);
        if !changed {
            break;
        }
    }
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let (node_map, _) = merge_nodes_intermediate_trie3(&roots_vec, god, 40);
    for root_idx in roots.values_mut() {
        *root_idx = *node_map.get(root_idx).unwrap_or(root_idx);
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

fn merge_nodes_intermediate_trie3(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
    max_iters: usize,
) -> (HashMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index>, Vec<IntermediatePrecomputeNode3Index>) {
    let all_nodes = Trie::all_nodes(god, roots);
    if all_nodes.is_empty() {
        return (HashMap::new(), vec![]);
    }

    let mut dense_of: HashMap<IntermediatePrecomputeNode3Index, usize> = HashMap::new();
    let mut old_of: Vec<IntermediatePrecomputeNode3Index> = Vec::with_capacity(all_nodes.len());
    for (i, node_idx) in all_nodes.iter().enumerate() {
        dense_of.insert(*node_idx, i);
        old_of.push(*node_idx);
    }
    let n = all_nodes.len();

    let mut ends: Vec<bool> = vec![false; n];
    type RawEdge = (IntermediateTrie3EdgeKey, usize);
    let mut raw_edges: Vec<Vec<RawEdge>> = vec![Vec::new(); n];

    for (u_dense, u_idx) in old_of.iter().enumerate() {
        if let Some(guard) = u_idx.read(god) {
            ends[u_dense] = guard.value.end;
            for (ek, dest_map) in guard.children() {
                for (v_idx, _) in dest_map {
                    if let Some(&v_dense) = dense_of.get(v_idx) {
                        raw_edges[u_dense].push((ek.clone(), v_dense));
                    }
                }
            }
        }
    }

    let mut prev_class: Vec<usize> = (0..n).map(|i| if ends[i] { 1 } else { 0 }).collect();

    for _it in 0..max_iters {
        type Signature = (bool, Vec<(IntermediateTrie3EdgeKey, usize)>);

        let mut sig_to_id: HashMap<Signature, usize> = HashMap::new();
        let mut new_class = vec![0; n];
        let mut next_id = 0;
        let mut changes = 0;

        for u in 0..n {
            let mut sig_edges: Vec<(IntermediateTrie3EdgeKey, usize)> = raw_edges[u]
                .iter()
                .map(|(ek, v_dense)| (ek.clone(), prev_class[*v_dense]))
                .collect();
            sig_edges.sort();

            let sig: Signature = (ends[u], sig_edges);

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
        if changes == 0 {
            break;
        }
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
        if let Some(rep) = representatives[class_id] {
            node_to_rep.insert(old_of[u_dense], rep);
        }
    }

    for class_id in 0..num_classes {
        if let Some(rep_idx) = representatives[class_id] {
            let u_dense = final_partition.iter().position(|&c| c == class_id).unwrap();

            let mut new_children: BTreeMap<IntermediateTrie3EdgeKey, OrderedHashMap<Trie2Index, ()>> = BTreeMap::new();
            for (ek, v_dense) in &raw_edges[u_dense] {
                let dest_class = final_partition[*v_dense];
                if let Some(dest_rep_idx) = representatives[dest_class] {
                    new_children.entry(ek.clone()).or_default().insert(dest_rep_idx, ());
                }
            }

            if let Some(mut guard) = rep_idx.write(god) {
                *guard.children_mut() = new_children;
            }
        }
    }

    let reps: Vec<_> = representatives.into_iter().filter_map(|x| x).collect();
    (node_to_rep, reps)
}
