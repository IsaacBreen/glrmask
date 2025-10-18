// src/constraint_precompute3_intermediate_utils.rs
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};
use crate::constraint::{IntermediatePrecomputeNode3Index, IntermediateTrie3GodWrapper, IntermediateTrie3EdgeKey};
use crate::datastructures::ordered_hash_map::Retain;
use crate::datastructures::trie::Trie;
use crate::r#macro::is_debug_level_enabled;

pub struct GlobalInternState { by_fingerprint: HashMap<u64, IntermediatePrecomputeNode3Index>, total_inserted: usize, total_reused: usize, total_merged: usize }
impl GlobalInternState { pub fn new() -> Self { Self { by_fingerprint: HashMap::new(), total_inserted: 0, total_reused: 0, total_merged: 0 } } }

pub fn optimize_intermediate_trie3_template(
    start_node: &IntermediatePrecomputeNode3Index,
    end_node: &IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
    interner: &mut GlobalInternState,
) {
    // Iterate until convergence or max passes. GC remains disabled in this arena.
    let max_passes = 4;
    for _ in 0..max_passes {
        let mut changed = false;

        if prune_unproductive_nodes(&[*start_node], end_node, god) {
            changed = true;
        }
        if compress_noop_edges(&[*start_node], end_node, god) {
            changed = true;
            if prune_unproductive_nodes(&[*start_node], end_node, god) {
                changed = true;
            }
        }
        if dedup_structurally_and_share(&[*start_node], end_node, god, interner) {
            changed = true;
            if prune_unproductive_nodes(&[*start_node], end_node, god) {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    if is_debug_level_enabled(3) {
        eprintln!(
            "[template-opt] interner: inserted={}, reused={}, merged={}",
            interner.total_inserted, interner.total_reused, interner.total_merged
        );
    }
}

pub fn optimize_intermediate_trie3(
    roots: &[IntermediatePrecomputeNode3Index],
    end_node: &IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
    interner: &mut GlobalInternState,
) {
    if is_debug_level_enabled(2) {
        let mut stats = crate::constraint_extra::PrecomputeStats::default();
        crate::constraint_extra::calculate_intermediate_stats3(roots, &mut stats, god);
        crate::constraint_extra::print_intermediate_stats3(&stats, god);
    }
    let max_passes = 3;
    for _ in 0..max_passes {
        let mut changed = false;
        if prune_unproductive_nodes(roots, end_node, god) {
            changed = true;
        }
        if compress_noop_edges(roots, end_node, god) {
            changed = true;
            if prune_unproductive_nodes(roots, end_node, god) {
                changed = true;
            }
        }
        if dedup_structurally_and_share(roots, end_node, god, interner) {
            changed = true;
            if prune_unproductive_nodes(roots, end_node, god) {
                changed = true;
            }
        }
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

fn collect_subgraph_nodes_and_incoming(
    start_nodes: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
)->(
    HashSet<IntermediatePrecomputeNode3Index>,
    HashMap<IntermediatePrecomputeNode3Index, Vec<IntermediatePrecomputeNode3Index>>,
    HashMap<IntermediatePrecomputeNode3Index, HashSet<IntermediatePrecomputeNode3Index>>,
){
    let all_nodes_vec = Trie::all_nodes(god, start_nodes);
    let all_nodes_in_subgraph: HashSet<_> = all_nodes_vec.into_iter().collect();
    let mut incoming: HashMap<IntermediatePrecomputeNode3Index, Vec<IntermediatePrecomputeNode3Index>> = HashMap::new();
    let mut child_sets: HashMap<IntermediatePrecomputeNode3Index, HashSet<IntermediatePrecomputeNode3Index>> = HashMap::new();
    for src in &all_nodes_in_subgraph {
        if let Some(g) = src.read(god) {
            for (_ek, dm) in g.children() {
                for (dst, _) in dm {
                    if all_nodes_in_subgraph.contains(dst) {
                        incoming.entry(*dst).or_default().push(*src);
                        child_sets.entry(*src).or_default().insert(*dst);
                    }
                }
            }
        }
    }
    (all_nodes_in_subgraph, incoming, child_sets)
}

fn topo_postorder_by_unique_children(
    nodes: &HashSet<IntermediatePrecomputeNode3Index>,
    incoming: &HashMap<IntermediatePrecomputeNode3Index, Vec<IntermediatePrecomputeNode3Index>>,
    child_sets: &HashMap<IntermediatePrecomputeNode3Index, HashSet<IntermediatePrecomputeNode3Index>>,
) -> Vec<IntermediatePrecomputeNode3Index> {
    let mut outdeg: HashMap<IntermediatePrecomputeNode3Index, usize> = HashMap::new();
    for n in nodes {
        let d = child_sets.get(n).map(|s| s.len()).unwrap_or(0);
        outdeg.insert(*n, d);
    }
    let mut q: VecDeque<IntermediatePrecomputeNode3Index> = outdeg
        .iter()
        .filter_map(|(n, d)| if *d == 0 { Some(*n) } else { None })
        .collect();
    let mut order: Vec<IntermediatePrecomputeNode3Index> = Vec::new();
    let mut seen: HashSet<IntermediatePrecomputeNode3Index> = HashSet::new();
    while let Some(n) = q.pop_front() {
        if !seen.insert(n) { continue; }
        order.push(n);
        if let Some(parents) = incoming.get(&n) {
            for p in parents {
                if let Some(e) = outdeg.get_mut(p) {
                    if *e > 0 {
                        *e -= 1;
                        if *e == 0 {
                            q.push_back(*p);
                        }
                    }
                }
            }
        }
    }
    if order.len() != nodes.len() {
        let mut fallback = Vec::new();
        for n in nodes { fallback.push(*n); }
        return fallback;
    }
    order
}

fn compute_structural_fingerprints(
    order: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
    cache: &mut HashMap<IntermediatePrecomputeNode3Index, u64>,
) {
    for n in order {
        let g = match n.read(god) { Some(guard) => guard, None => continue };
        let mut pairs: Vec<(IntermediateTrie3EdgeKey, Vec<u64>)> = Vec::new();
        for (ek, dm) in g.children() {
            let mut child_sigs: Vec<u64> = dm.keys().filter_map(|dst| cache.get(dst)).cloned().collect();
            child_sigs.sort_unstable();
            pairs.push((ek.clone(), child_sigs));
        }
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        g.value.end.hash(&mut hasher);
        for (ek, cs) in &pairs {
            ek.hash(&mut hasher);
            cs.len().hash(&mut hasher);
            for s in cs { s.hash(&mut hasher); }
        }
        let sig = hasher.finish();
        cache.insert(*n, sig);
    }
}

fn redirect_incoming_to(
    from: IntermediatePrecomputeNode3Index,
    to: IntermediatePrecomputeNode3Index,
    incoming: &mut HashMap<IntermediatePrecomputeNode3Index, Vec<IntermediatePrecomputeNode3Index>>,
    god: &IntermediateTrie3GodWrapper,
) -> bool {
    if from == to { return false; }
    let mut any = false;
    let sources: Vec<IntermediatePrecomputeNode3Index> = incoming.get(&from).cloned().unwrap_or_default();
    for src in sources {
        let keys_to_update: Vec<IntermediateTrie3EdgeKey> = {
            let sg = match src.read(god) { Some(g) => g, None => continue };
            sg.children().iter().filter_map(|(ek, dm)| if dm.contains_key(&from) { Some(ek.clone()) } else { None }).collect()
        };
        if keys_to_update.is_empty() { continue; }
        if let Some(mut sw) = src.write(god) {
            for ek in keys_to_update {
                if let Some(dm) = sw.children_mut().get_mut(&ek) {
                    if dm.remove(&from).is_some() {
                        dm.insert(to, ());
                        any = true;
                    }
                }
            }
            sw.children_mut().retain(|_ek, dm| !dm.is_empty());
        }
        if let Some(v) = incoming.get_mut(&from) { v.retain(|p| *p != src); }
        incoming.entry(to).or_default().push(src);
    }
    any
}

fn compress_noop_edges(
    start_nodes: &[IntermediatePrecomputeNode3Index],
    end_node: &IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
) -> bool {
    let (nodes, _incoming, _child_sets) = collect_subgraph_nodes_and_incoming(start_nodes, god);
    if nodes.is_empty() { return false; }
    let mut changed = false;
    for n in &nodes {
        let noop_targets: Vec<IntermediatePrecomputeNode3Index> = {
            let g = match n.read(god) { Some(guard) => guard, None => continue };
            let mut targets = Vec::new();
            if let Some(dm) = g.children().get(&IntermediateTrie3EdgeKey::NoOp) {
                for (dst, _) in dm.iter() {
                    if dst != end_node && *dst != *n { targets.push(*dst); }
                }
            }
            targets
        };
        if noop_targets.is_empty() { continue; }
        let mut edges_to_add: Vec<(IntermediateTrie3EdgeKey, IntermediatePrecomputeNode3Index)> = Vec::new();
        for t in &noop_targets {
            if let Some(tg) = t.read(god) {
                for (ek2, dm2) in tg.children() {
                    for (dst2, _) in dm2 {
                        edges_to_add.push((ek2.clone(), *dst2));
                    }
                }
            }
        }
        if let Some(mut w) = n.write(god) {
            if let Some(dm) = w.children_mut().get_mut(&IntermediateTrie3EdgeKey::NoOp) {
                for t in &noop_targets {
                    if dm.remove(t).is_some() { changed = true; }
                }
                if dm.is_empty() {
                    w.children_mut().remove(&IntermediateTrie3EdgeKey::NoOp);
                }
            }
            for (ek2, dst2) in edges_to_add {
                let dest_map = w.children_mut().entry(ek2).or_default();
                if !dest_map.contains_key(&dst2) {
                    dest_map.insert(dst2, ());
                    changed = true;
                }
            }
        }
    }
    changed
}

fn dedup_structurally_and_share(
    start_nodes: &[IntermediatePrecomputeNode3Index],
    end_node: &IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
    interner: &mut GlobalInternState,
) -> bool {
    let (nodes, mut incoming, child_sets) = collect_subgraph_nodes_and_incoming(start_nodes, god);
    if nodes.is_empty() { return false; }
    let order = topo_postorder_by_unique_children(&nodes, &incoming, &child_sets);
    let mut fingerprints: HashMap<IntermediatePrecomputeNode3Index, u64> = HashMap::new();
    compute_structural_fingerprints(&order, god, &mut fingerprints);
    let mut changed = false;
    let mut local_canonical: HashMap<u64, IntermediatePrecomputeNode3Index> = HashMap::new();
    for n in &order {
        let sig = match fingerprints.get(n) { Some(s) => *s, None => continue };
        if n == end_node {
            interner.by_fingerprint.insert(sig, *n);
            local_canonical.insert(sig, *n);
            continue;
        }
        let is_start = start_nodes.iter().any(|s| s == n);
        let mut canonical = if let Some(c) = local_canonical.get(&sig) {
            *c
        } else if let Some(c) = interner.by_fingerprint.get(&sig) {
            *c
        } else {
            interner.by_fingerprint.insert(sig, *n);
            interner.total_inserted += 1;
            local_canonical.insert(sig, *n);
            *n
        };
        if let Some(local_c) = local_canonical.get(&sig) {
            canonical = *local_c;
        }
        if canonical != *n && !is_start {
            let rewired = redirect_incoming_to(*n, canonical, &mut incoming, god);
            if rewired {
                interner.total_merged += 1;
                changed = true;
            } else {
                interner.total_reused += 1;
            }
        } else {
            local_canonical.insert(sig, *n);
        }
    }
    changed
}
