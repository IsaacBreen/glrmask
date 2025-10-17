// src/constraint_precompute3_intermediate_utils.rs
use std::collections::{HashMap, HashSet, VecDeque};
use crate::constraint::{IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey, IntermediateTrie3GodWrapper, LLMTokenBV, StateIDBV};
use crate::datastructures::ordered_hash_map::Retain;
use crate::datastructures::trie::Trie;
use crate::r#macro::is_debug_level_enabled;
use std::collections::BTreeMap;
use crate::datastructures::ordered_hash_map::OrderedHashMap;

pub fn optimize_intermediate_trie3_template(
    start_node: &IntermediatePrecomputeNode3Index,
    end_node: &IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
) {
    let all_nodes = Trie::all_nodes(god, &[*start_node]);
    if all_nodes.is_empty() {
        return;
    }

    // A few passes of optimization.
    for _ in 0..3 {
        let mut changed_this_pass = false;
        changed_this_pass |= bypass_noop_nodes(&all_nodes, god);
        changed_this_pass |= coalesce_parallel_edges_intermediate(&all_nodes, god);
        changed_this_pass |= prune_unproductive_nodes(&[*start_node], end_node, god);

        // GC is needed to remove nodes that become unreachable after pruning edges.
        // NOTE: GC is disabled here because it was causing issues with multiple templates
        // in the same arena. Dangling nodes are acceptable for now.
        // Trie::gc(god, &[*start_node]);

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
        let mut stats = crate::constraint_extra::PrecomputeStats::default();
        crate::constraint_extra::calculate_intermediate_stats3(roots, &mut stats, god);
        crate::constraint_extra::print_intermediate_stats3(&stats, god);
    }

    let all_nodes = Trie::all_nodes(god, roots);
    if all_nodes.is_empty() {
        return;
    }

    for i in 0..3 {
        let mut changed_this_pass = false;
        if is_debug_level_enabled(2) {
            println!("--- Intermediate Trie3 Opt Pass {} ---", i + 1);
        }
        changed_this_pass |= bypass_noop_nodes(&all_nodes, god);
        changed_this_pass |= coalesce_parallel_edges_intermediate(&all_nodes, god);
        changed_this_pass |= prune_unproductive_nodes(roots, end_node, god);

        if !changed_this_pass {
            if is_debug_level_enabled(2) {
                println!("--- Intermediate Trie3 Opt finished in pass {} (no changes) ---", i + 1);
            }
            break;
        }
        if is_debug_level_enabled(2) {
            println!("--- After Intermediate Trie3 Opt Pass {} ---", i + 1);
            let mut stats = crate::constraint_extra::PrecomputeStats::default();
            crate::constraint_extra::calculate_intermediate_stats3(roots, &mut stats, god);
            crate::constraint_extra::print_intermediate_stats3(&stats, god);
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

fn coalesce_parallel_edges_intermediate(
    nodes: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) -> bool {
    let mut changed = false;
    for node_idx in nodes {
        let old_children = if let Some(g) = node_idx.read(god) {
            g.children().clone()
        } else {
            continue;
        };

        if old_children.is_empty() {
            continue;
        }

        // Aggregate edges by destination
        // dest -> Vec<EdgeKey>
        let mut edges_by_dest: HashMap<IntermediatePrecomputeNode3Index, Vec<IntermediateTrie3EdgeKey>> =
            HashMap::new();
        for (edge_key, dest_map) in &old_children {
            for (dest_idx, _) in dest_map {
                edges_by_dest
                    .entry(*dest_idx)
                    .or_default()
                    .push(edge_key.clone());
            }
        }

        let mut has_merges = false;
        // For each destination, merge compatible edge keys
        let mut new_edges_by_dest: HashMap<IntermediatePrecomputeNode3Index, Vec<IntermediateTrie3EdgeKey>> =
            HashMap::new();
        for (dest_idx, keys) in edges_by_dest {
            if keys.len() <= 1 {
                new_edges_by_dest.insert(dest_idx, keys);
                continue;
            }

            has_merges = true;
            let mut merged_keys: Vec<IntermediateTrie3EdgeKey> = Vec::new();
            let mut pushes: Vec<StateIDBV> = Vec::new();
            let mut pops: HashMap<usize, Vec<StateIDBV>> = HashMap::new();
            let mut checks: Vec<LLMTokenBV> = Vec::new();
            let mut has_noop = false;

            for key in keys {
                match key {
                    IntermediateTrie3EdgeKey::Push(s) => pushes.push(s),
                    IntermediateTrie3EdgeKey::Pop(n, s) => pops.entry(n).or_default().push(s),
                    IntermediateTrie3EdgeKey::CheckLLM(l) => checks.push(l),
                    IntermediateTrie3EdgeKey::NoOp => has_noop = true,
                }
            }

            if !pushes.is_empty() {
                let mut union_s = StateIDBV::zeros();
                for s in pushes {
                    union_s |= &s;
                }
                merged_keys.push(IntermediateTrie3EdgeKey::Push(union_s));
            }
            for (n, s_vec) in pops {
                let mut union_s = StateIDBV::zeros();
                for s in s_vec {
                    union_s |= &s;
                }
                merged_keys.push(IntermediateTrie3EdgeKey::Pop(n, union_s));
            }
            if !checks.is_empty() {
                let mut union_l = LLMTokenBV::zeros();
                for l in checks {
                    union_l |= &l;
                }
                merged_keys.push(IntermediateTrie3EdgeKey::CheckLLM(union_l));
            }
            if has_noop {
                merged_keys.push(IntermediateTrie3EdgeKey::NoOp);
            }

            new_edges_by_dest.insert(dest_idx, merged_keys);
        }

        if has_merges {
            changed = true;
            let mut new_children = BTreeMap::new();
            for (dest_idx, keys) in new_edges_by_dest {
                for key in keys {
                    new_children
                        .entry(key)
                        .or_insert_with(OrderedHashMap::new)
                        .insert(dest_idx, ());
                }
            }
            if let Some(mut g) = node_idx.write(god) {
                *g.children_mut() = new_children;
            }
        }
    }
    changed
}

fn bypass_noop_nodes(
    nodes: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) -> bool {
    let mut changed = false;

    // Build reverse adjacency map
    let mut incoming: HashMap<IntermediatePrecomputeNode3Index, Vec<IntermediatePrecomputeNode3Index>> =
        HashMap::new();
    for src in nodes {
        if let Some(g) = src.read(god) {
            for (_ek, dm) in g.children() {
                for (dst, _) in dm {
                    incoming.entry(*dst).or_default().push(*src);
                }
            }
        }
    }

    for node_idx in nodes {
        let (old_children, is_end) = if let Some(g) = node_idx.read(god) {
            (g.children().clone(), g.value.end)
        } else {
            continue;
        };

        if is_end {
            continue;
        }

        // Find children reachable via NoOp that are candidates for bypass
        let mut bypass_candidates = Vec::new();
        if let Some(dest_map) = old_children.get(&IntermediateTrie3EdgeKey::NoOp) {
            for (child_idx, _) in dest_map {
                // A child is a candidate if it's not an end node and its only parent is the current node.
                if let Some(child_g) = child_idx.read(god) {
                    if !child_g.value.end {
                        if let Some(parents) = incoming.get(child_idx) {
                            if parents.len() == 1 && parents[0] == *node_idx {
                                bypass_candidates.push(*child_idx);
                            }
                        }
                    }
                }
            }
        }

        if bypass_candidates.is_empty() {
            continue;
        }

        changed = true;
        let mut new_children = old_children;
        // Remove the NoOp edge to bypassed children
        if let Some(dest_map) = new_children.get_mut(&IntermediateTrie3EdgeKey::NoOp) {
            dest_map.retain(|k, _| !bypass_candidates.contains(k));
        }
        new_children.retain(|_, v| !v.is_empty());

        // Add children of bypassed nodes to current node
        for bypassed_child_idx in bypass_candidates {
            if let Some(bypassed_g) = bypassed_child_idx.read(god) {
                for (edge_key, dest_map) in bypassed_g.children() {
                    let entry = new_children.entry(edge_key.clone()).or_default();
                    for (dest, val) in dest_map {
                        entry.insert(*dest, *val);
                    }
                }
            }
        }

        if let Some(mut g) = node_idx.write(god) {
            *g.children_mut() = new_children;
        }
    }

    changed
}
