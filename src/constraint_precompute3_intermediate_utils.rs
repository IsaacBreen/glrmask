// src/constraint_precompute3_intermediate_utils.rs
use std::collections::{HashMap, HashSet, VecDeque};
use crate::constraint::{IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey, IntermediateTrie3GodWrapper, LLMTokenBV, StateIDBV};
use crate::datastructures::ordered_hash_map::Retain;
use crate::datastructures::trie::Trie;
use crate::r#macro::is_debug_level_enabled;
use std::collections::BTreeMap;

fn coalesce_parallel_edges(
    nodes: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) -> bool {
    let mut changed = false;
    for node_idx in nodes {
        let old_children = {
            let r = node_idx.read(god).unwrap();
            if r.children().len() <= 1 { continue; }
            r.children().clone()
        };

        // Aggregate keys by destination
        let mut keys_by_dest: HashMap<IntermediatePrecomputeNode3Index, Vec<IntermediateTrie3EdgeKey>> = HashMap::new();
        for (edge_key, dest_map) in &old_children {
            for (dest, _) in dest_map.iter() {
                keys_by_dest.entry(*dest).or_default().push(edge_key.clone());
            }
        }

        let mut needs_rebuild = false;
        for keys in keys_by_dest.values() {
            if keys.len() > 1 {
                needs_rebuild = true;
                break;
            }
        }
        if !needs_rebuild { continue; }

        let mut new_children = BTreeMap::new();
        let mut processed_dests = HashSet::new();

        // Copy over edges that are not part of any merge
        for (edge_key, dest_map) in &old_children {
            for (dest, _) in dest_map {
                if keys_by_dest.get(dest).map_or(true, |keys| keys.len() <= 1) {
                    new_children.entry(edge_key.clone()).or_default().insert(*dest, ());
                    processed_dests.insert(*dest);
                }
            }
        }

        // Process merges
        for (dest, keys) in keys_by_dest {
            if keys.len() <= 1 || processed_dests.contains(&dest) { continue; }

            let mut pushes = StateIDBV::zeros();
            let mut pops: HashMap<usize, StateIDBV> = HashMap::new();
            let mut checks = LLMTokenBV::zeros();
            let mut has_noop = false;

            for key in keys {
                match key {
                    IntermediateTrie3EdgeKey::Push(s) => pushes |= &s,
                    IntermediateTrie3EdgeKey::Pop(n, s) => *pops.entry(n).or_default() |= &s,
                    IntermediateTrie3EdgeKey::CheckLLM(l) => checks |= &l,
                    IntermediateTrie3EdgeKey::NoOp => has_noop = true,
                }
            }

            if !pushes.is_empty() {
                new_children.entry(IntermediateTrie3EdgeKey::Push(pushes)).or_default().insert(dest, ());
            }
            for (n, s) in pops {
                new_children.entry(IntermediateTrie3EdgeKey::Pop(n, s)).or_default().insert(dest, ());
            }
            if !checks.is_empty() {
                new_children.entry(IntermediateTrie3EdgeKey::CheckLLM(checks)).or_default().insert(dest, ());
            }
            if has_noop {
                new_children.entry(IntermediateTrie3EdgeKey::NoOp).or_default().insert(dest, ());
            }
        }

        if old_children != new_children {
            let mut w = node_idx.write(god).unwrap();
            *w.children_mut() = new_children;
            changed = true;
        }
    }
    changed
}

fn bypass_noop_nodes(
    nodes: &[IntermediatePrecomputeNode3Index],
    end_node: &IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
) -> bool {
    let mut changed = false;
    let mut predecessors: HashMap<IntermediatePrecomputeNode3Index, usize> = HashMap::new();
    for node_idx in nodes {
        let r = node_idx.read(god).unwrap();
        for (_, dm) in r.children() {
            for (dest, _) in dm {
                *predecessors.entry(*dest).or_default() += 1;
            }
        }
    }

    for node_idx in nodes {
        let (edges_to_add, noop_edges_to_remove) = {
            let w = node_idx.read(god).unwrap();
            let mut edges_to_add = Vec::new();
            let mut noop_edges_to_remove = Vec::new();

            if let Some(dest_map) = w.children().get(&IntermediateTrie3EdgeKey::NoOp) {
                for (dest, _) in dest_map {
                    if dest == end_node { continue; }
                    if predecessors.get(dest).cloned().unwrap_or(0) != 1 { continue; }

                    let dest_guard = dest.read(god).unwrap();
                    if !dest_guard.value.end {
                        noop_edges_to_remove.push(*dest);
                        for (succ_key, succ_dest_map) in dest_guard.children() {
                            for (succ_dest, _) in succ_dest_map {
                                edges_to_add.push((succ_key.clone(), *succ_dest));
                            }
                        }
                    }
                }
            }
            (edges_to_add, noop_edges_to_remove)
        };


        if !edges_to_add.is_empty() {
            changed = true;
            let mut w = node_idx.write(god).unwrap();
            if let Some(dm) = w.children_mut().get_mut(&IntermediateTrie3EdgeKey::NoOp) {
                for dest_to_remove in noop_edges_to_remove {
                    dm.remove(&dest_to_remove);
                }
            }
            if w.children().get(&IntermediateTrie3EdgeKey::NoOp).map_or(false, |dm| dm.is_empty()) {
                w.children_mut().remove(&IntermediateTrie3EdgeKey::NoOp);
            }

            for (key, dest) in edges_to_add {
                w.children_mut().entry(key).or_default().insert(dest, ());
            }
        }
    }
    changed
}

pub fn optimize_intermediate_trie3_template(
    start_node: &IntermediatePrecomputeNode3Index,
    end_node: &IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
) {
    // A few passes of optimization.
    for _ in 0..3 {
        let all_nodes = Trie::all_nodes(god, &[*start_node]);
        if all_nodes.is_empty() { break; }
        let mut changed_this_pass = false;

        if bypass_noop_nodes(&all_nodes, end_node, god) {
            changed_this_pass = true;
        }
        if coalesce_parallel_edges(&all_nodes, god) {
            changed_this_pass = true;
        }
        if prune_unproductive_nodes(&[*start_node], end_node, god) {
            changed_this_pass = true;
        }

        if !changed_this_pass {
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
        println!("Before optimization:");
        let mut stats = crate::constraint_extra::PrecomputeStats::default();
        crate::constraint_extra::calculate_intermediate_stats3(roots, &mut stats, god);
        crate::constraint_extra::print_intermediate_stats3(&stats, god);
    }
    for _ in 0..3 {
        let all_nodes = Trie::all_nodes(god, roots);
        if all_nodes.is_empty() { break; }
        let mut changed_this_pass = false;

        if bypass_noop_nodes(&all_nodes, end_node, god) {
            changed_this_pass = true;
        }
        if coalesce_parallel_edges(&all_nodes, god) {
            changed_this_pass = true;
        }
        if prune_unproductive_nodes(roots, end_node, god) {
            changed_this_pass = true;
        }

        if !changed_this_pass {
            break;
        }
    }
    if is_debug_level_enabled(2) {
        println!("After optimization:");
        let mut stats = crate::constraint_extra::PrecomputeStats::default();
        crate::constraint_extra::calculate_intermediate_stats3(roots, &mut stats, god);
        crate::constraint_extra::print_intermediate_stats3(&stats, god);
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
