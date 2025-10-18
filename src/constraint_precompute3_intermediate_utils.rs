// src/constraint_precompute3_intermediate_utils.rs
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use crate::constraint::{IntermediatePrecomputeNode3Index, IntermediateTrie3GodWrapper};
use crate::datastructures::ordered_hash_map::Retain;
use crate::datastructures::trie::Trie;
use crate::r#macro::is_debug_level_enabled;
use crate::constraint::IntermediateTrie3EdgeKey;
use crate::tokenizer::TokenizerStateID;

pub fn optimize_intermediate_trie3_template(
    start_node: &mut IntermediatePrecomputeNode3Index,
    end_node: &mut IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
) {
    // A few passes of optimization.
    for _ in 0..3 {
        let mut changed = false;
        changed |= bypass_noop_nodes(&[*start_node], god);
        changed |= prune_unproductive_nodes(&[*start_node], end_node, god);

        let (map, merged) = merge_equivalent_nodes(&[*start_node], Some(end_node), god);
        if merged {
            changed = true;
            *start_node = map.get(start_node).cloned().unwrap_or(*start_node);
            *end_node = map.get(end_node).cloned().unwrap_or(*end_node);
        }

        if !changed {
            break;
        }
    }
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

    let mut roots_vec: Vec<_> = roots.values().cloned().collect();

    for _ in 0..3 {
        let mut changed = false;
        changed |= bypass_noop_nodes(&roots_vec, god);
        changed |= prune_unproductive_nodes(&roots_vec, end_node, god);

        let (map, merged) = merge_equivalent_nodes(&roots_vec, Some(end_node), god);
        if merged {
            changed = true;
            // Remap roots in the BTreeMap
            for root in roots.values_mut() {
                if let Some(new_root) = map.get(root) {
                    *root = *new_root;
                }
            }
            // Update the local roots_vec for the next iteration
            roots_vec = roots.values().cloned().collect();
        }

        if !changed {
            break;
        }
        Trie::gc(god, &roots_vec);
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

fn bypass_noop_nodes(
    start_nodes: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) -> bool {
    let all_nodes = Trie::all_nodes(god, start_nodes);
    let mut changes_to_make: HashMap<IntermediatePrecomputeNode3Index, BTreeMap<_, _>> = HashMap::new();
    let mut changed = false;

    // Phase 1: Read graph and determine changes, without holding locks for long.
    for node_idx in &all_nodes {
        let old_children = if let Some(guard) = node_idx.read(god) {
            guard.children().clone()
        } else {
            continue;
        };
        
        if old_children.is_empty() { continue; }

        let mut new_children = old_children.clone();
        let mut local_change = false;

        for (edge_key, dest_map) in &old_children {
            if let IntermediateTrie3EdgeKey::NoOp = &edge_key {
                let mut bypassed_something = false;
                for (dest_idx, _) in dest_map {
                    if let Some(dest_guard) = dest_idx.read(god) {
                        // Don't bypass end nodes.
                        if dest_guard.value.end { continue; }

                        bypassed_something = true;
                        
                        // Add dest's children to new_children
                        for (child_edge_key, child_dest_map) in dest_guard.children() {
                            let entry = new_children.entry(child_edge_key.clone()).or_default();
                            for (child_dest, val) in child_dest_map {
                                entry.insert(*child_dest, val.clone());
                            }
                        }
                    }
                }
                if bypassed_something {
                    // Remove the NoOp edge from new_children
                    new_children.remove(edge_key);
                    local_change = true;
                }
            }
        }

        if local_change {
            changed = true;
            changes_to_make.insert(*node_idx, new_children);
        }
    }

    if !changed {
        return false;
    }

    // Phase 2: Apply all collected changes.
    for (node_idx, new_children) in changes_to_make {
        if let Some(mut guard) = node_idx.write(god) {
            *guard.children_mut() = new_children;
        }
    }

    true
}

fn merge_equivalent_nodes(
    start_nodes: &[IntermediatePrecomputeNode3Index],
    end_node: Option<&IntermediatePrecomputeNode3Index>, // Optional end node for partitioning
    god: &IntermediateTrie3GodWrapper,
) -> (HashMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index>, bool) {
    let all_nodes = Trie::all_nodes(god, start_nodes);
    let n = all_nodes.len();
    if n == 0 { return (HashMap::new(), false); }

    let mut dense_of: HashMap<IntermediatePrecomputeNode3Index, usize> = HashMap::new();
    let mut old_of: Vec<IntermediatePrecomputeNode3Index> = Vec::with_capacity(n);
    for (i, node_idx) in all_nodes.iter().enumerate() {
        dense_of.insert(*node_idx, i);
        old_of.push(*node_idx);
    }

    let end_node_dense_idx = end_node.and_then(|en| dense_of.get(en).copied());

    // Initial partition: 0 for normal, 1 for end_node (if provided), 2 for other end nodes
    let mut prev_class: Vec<usize> = (0..n).map(|i| {
        let node_idx = old_of[i];
        let is_end = if let Some(g) = node_idx.read(god) { g.value.end } else { false };
        if Some(i) == end_node_dense_idx { 1 }
        else if is_end { 2 }
        else { 0 }
    }).collect();

    for _ in 0..10 {
        type Signature = BTreeMap<IntermediateTrie3EdgeKey, BTreeSet<usize>>;
        let mut sig_to_id: HashMap<Signature, usize> = HashMap::new();
        let mut new_class = vec![0; n];
        let mut next_id = 0;
        let mut changes = 0;

        for u_dense in 0..n {
            let u_idx = old_of[u_dense];
            let guard = if let Some(g) = u_idx.read(god) { g } else { continue };
            let mut sig: Signature = BTreeMap::new();
            for (edge_key, dest_map) in guard.children() {
                let dest_classes = sig.entry(edge_key.clone()).or_default();
                for (dest_idx, _) in dest_map {
                    if let Some(&v_dense) = dense_of.get(dest_idx) {
                        dest_classes.insert(prev_class[v_dense]);
                    }
                }
            }

            let cid = *sig_to_id.entry(sig).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });

            new_class[u_dense] = cid;
            if new_class[u_dense] != prev_class[u_dense] {
                changes += 1;
            }
        }

        prev_class = new_class;
        if changes == 0 { break; }
    }

    let final_partition = prev_class;
    let num_classes = final_partition.iter().max().map_or(0, |m| m + 1);

    if num_classes == n { return (HashMap::new(), false); }

    let mut representatives: Vec<Option<IntermediatePrecomputeNode3Index>> = vec![None; num_classes];
    // Prioritize keeping start_nodes and end_node as representatives
    let special_nodes: HashSet<_> = start_nodes.iter().chain(end_node).copied().collect();
    for (u_dense, &class_id) in final_partition.iter().enumerate() {
        let u_idx = old_of[u_dense];
        if special_nodes.contains(&u_idx) {
            representatives[class_id] = Some(u_idx);
        }
    }
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

    for rep_idx_opt in representatives {
        if let Some(rep_idx) = rep_idx_opt {
            if let Some(mut guard) = rep_idx.write(god) {
                let old_children = std::mem::take(guard.children_mut());
                for (edge_key, dest_map) in old_children {
                    let new_dest_map = guard.children_mut().entry(edge_key).or_default();
                    for (dest_idx, val) in dest_map {
                        if let Some(rep_dest) = node_to_rep.get(&dest_idx) {
                            new_dest_map.insert(*rep_dest, val);
                        } else {
                            // This can happen if a node was unreachable and not in all_nodes
                            new_dest_map.insert(dest_idx, val);
                        }
                    }
                }
            }
        }
    }

    (node_to_rep, true)
}
