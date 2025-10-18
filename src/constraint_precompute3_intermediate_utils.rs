// src/constraint_precompute3_intermediate_utils.rs
use std::collections::{HashMap, HashSet, VecDeque};
use crate::constraint::{IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3GodWrapper};
use crate::constraint_precompute3_challenge_elimination::{get_normalized_paths_for_vec, normalize_path};
use crate::datastructures::ordered_hash_map::Retain;
use crate::datastructures::trie::Trie;
use crate::r#macro::is_debug_level_enabled;

pub fn optimize_intermediate_trie3_template(
    start_node: &IntermediatePrecomputeNode3Index,
    end_node: &IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
) {
    use std::collections::HashSet;
    let mut pinned: HashSet<IntermediatePrecomputeNode3Index> = HashSet::new();
    pinned.insert(*start_node);
    pinned.insert(*end_node);

    for _ in 0..3 {
        let mut changed = false;
        changed |= prune_unproductive_nodes(&[*start_node], end_node, god);
        changed |= compress_noop_only_nodes(&[*start_node], &pinned, god);
        changed |= structural_merge_nodes_in_subgraph(&[*start_node], &pinned, god);
        changed |= prune_unproductive_nodes(&[*start_node], end_node, god);
        if !changed { break; }
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

// Compress nodes whose outgoing edges are all NoOp by bypassing them.
// Pinned nodes are never removed (e.g., template start/end).
fn compress_noop_only_nodes(
    start_nodes: &[IntermediatePrecomputeNode3Index],
    pinned: &std::collections::HashSet<IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) -> bool {
    use std::collections::HashMap;

    let all_nodes_vec = Trie::all_nodes(god, start_nodes);
    if all_nodes_vec.is_empty() {
        return false;
    }
    let all_nodes_in_subgraph: std::collections::HashSet<_> = all_nodes_vec.into_iter().collect();

    // Build incoming map with edge keys
    let mut incoming: HashMap<
        IntermediatePrecomputeNode3Index,
        Vec<(IntermediatePrecomputeNode3Index, crate::constraint::IntermediateTrie3EdgeKey)>
    > = HashMap::new();

    for src in &all_nodes_in_subgraph {
        if let Some(g) = src.read(god) {
            for (ek, dm) in g.children() {
                for (dst, _) in dm {
                    if all_nodes_in_subgraph.contains(dst) {
                        incoming.entry(*dst).or_default().push((*src, ek.clone()));
                    }
                }
            }
        }
    }

    // Identify compressible nodes: non-pinned, non-end, and all outgoing edges are NoOp
    let mut compressible: Vec<IntermediatePrecomputeNode3Index> = Vec::new();
    for n in &all_nodes_in_subgraph {
        if pinned.contains(n) { continue; }
        if let Some(g) = n.read(god) {
            if g.value.end {
                continue;
            }
            let mut has_edges = false;
            let mut only_noop = true;
            for (ek, dm) in g.children() {
                if !dm.is_empty() {
                    has_edges = true;
                }
                if !matches!(ek, crate::constraint::IntermediateTrie3EdgeKey::NoOp) {
                    only_noop = false;
                    break;
                }
            }
            if has_edges && only_noop {
                compressible.push(*n);
            }
        }
    }

    if compressible.is_empty() {
        return false;
    }

    let mut changed = false;

    // For each compressible node, redirect all incoming edges to its NoOp destinations
    for n in compressible {
        // Snapshot NoOp destinations
        let noop_dests: Vec<IntermediatePrecomputeNode3Index> = if let Some(g) = n.read(god) {
            g.children()
                .get(&crate::constraint::IntermediateTrie3EdgeKey::NoOp)
                .map(|dm| dm.keys().cloned().collect::<Vec<_>>())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        if noop_dests.is_empty() {
            continue;
        }

        if let Some(preds) = incoming.get(&n) {
            for (pred, ek) in preds {
                if let Some(mut w) = pred.write(god) {
                    let dest_map = w.children_mut().entry(ek.clone()).or_default();
                    // Remove pred -> n under this key
                    dest_map.remove(&n);
                    // Add pred -> each NoOp destination under the same key
                    for d in &noop_dests {
                        dest_map.insert(*d, ());
                    }
                    changed = true;
                }
            }
        }
    }

    changed
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct NodeSignature {
    end: bool,
    // For determinism: edge keys sorted; each has sorted child-colors
    edges: Vec<(crate::constraint::IntermediateTrie3EdgeKey, Vec<u64>)>,
}

// Merge structurally equivalent nodes within the reachable subgraph, except pinned nodes.
// Works across multiple roots if provided.
fn structural_merge_nodes_in_subgraph(
    start_nodes: &[IntermediatePrecomputeNode3Index],
    pinned: &std::collections::HashSet<IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) -> bool {
    // return false;
    use std::collections::HashMap;

    let all_nodes_vec = Trie::all_nodes(god, start_nodes);
    if all_nodes_vec.is_empty() {
        return false;
    }
    let all_nodes_in_subgraph: std::collections::HashSet<_> = all_nodes_vec.clone().into_iter().collect();

    // Snapshot outgoing children for each node within the subgraph for stable iteration
    let mut outgoing: HashMap<
        IntermediatePrecomputeNode3Index,
        Vec<(crate::constraint::IntermediateTrie3EdgeKey, Vec<IntermediatePrecomputeNode3Index>)>
    > = HashMap::new();

    for n in &all_nodes_in_subgraph {
        if let Some(g) = n.read(god) {
            let mut edges: Vec<(crate::constraint::IntermediateTrie3EdgeKey, Vec<IntermediatePrecomputeNode3Index>)> = Vec::new();
            for (ek, dm) in g.children() {
                let mut kids: Vec<_> = dm.keys().cloned().filter(|k| all_nodes_in_subgraph.contains(k)).collect();
                kids.sort(); // ensure deterministic order of children
                edges.push((ek.clone(), kids));
            }
            edges.sort_by(|a, b| a.0.cmp(&b.0)); // deterministic by edge key
            outgoing.insert(*n, edges);
        }
    }

    // Build incoming (pred, edge_key) list for rewrites
    let mut incoming: HashMap<
        IntermediatePrecomputeNode3Index,
        Vec<(IntermediatePrecomputeNode3Index, crate::constraint::IntermediateTrie3EdgeKey)>
    > = HashMap::new();
    for (src, edges) in &outgoing {
        for (ek, kids) in edges {
            for k in kids {
                incoming.entry(*k).or_default().push((*src, ek.clone()));
            }
        }
    }

    // Iterative color refinement on DAG-like structure (robust even if cycles appear).
    let mut color: HashMap<IntermediatePrecomputeNode3Index, u64> = HashMap::new();
    let mut next_color: HashMap<NodeSignature, u64> = HashMap::new();

    // Seed: base color by end flag and out-degree patterns only (ignore incoming context).
    for n in &all_nodes_in_subgraph {
        let end_flag = if let Some(g) = n.read(god) { g.value.end } else { false };
        let out_deg_summary: Vec<(crate::constraint::IntermediateTrie3EdgeKey, usize)> = outgoing.get(n)
            .map(|v| v.iter().map(|(ek, kids)| (ek.clone(), kids.len())).collect())
            .unwrap_or_default();
        let mut out_sig_part: Vec<_> = out_deg_summary.into_iter().map(|(ek, cnt)| (ek, vec![cnt as u64])).collect();
        out_sig_part.sort_by(|a, b| a.0.cmp(&b.0));

        let sig = NodeSignature { end: end_flag, edges: out_sig_part };
        let len = next_color.len();
        let id = *next_color.entry(sig).or_insert(len as u64 + 1);
        color.insert(*n, id);
    }

    // Refine up to a bounded number of iterations or until convergence
    let max_iters = 16;
    for _ in 0..max_iters {
        let mut changed = false;
        let mut interner: HashMap<NodeSignature, u64> = HashMap::new();
        let mut new_color: HashMap<IntermediatePrecomputeNode3Index, u64> = HashMap::new();

        for n in &all_nodes_in_subgraph {
            let end_flag = if let Some(g) = n.read(god) { g.value.end } else { false };
            let mut edges_sig: Vec<(crate::constraint::IntermediateTrie3EdgeKey, Vec<u64>)> = Vec::new();
            if let Some(edges) = outgoing.get(n) {
                for (ek, kids) in edges {
                    let mut kid_colors: Vec<u64> = kids.iter().map(|k| *color.get(k).unwrap_or(&0)).collect();
                    kid_colors.sort_unstable();
                    edges_sig.push((ek.clone(), kid_colors));
                }
                edges_sig.sort_by(|a, b| a.0.cmp(&b.0));
            }

            // Ignore incoming context for equivalence; only outgoing structure + end flag matters.
            let sig = NodeSignature { end: end_flag, edges: edges_sig };
            let len = interner.len();
            let id = *interner.entry(sig).or_insert(len as u64 + 1);
            if color.get(n).copied().unwrap_or(0) != id {
                changed = true;
            }
            new_color.insert(*n, id);
        }

        color = new_color;
        if !changed {
            break;
        }
    }

    // Group nodes by final color
    let mut groups: HashMap<u64, Vec<IntermediatePrecomputeNode3Index>> = HashMap::new();
    for (n, c) in &color {
        groups.entry(*c).or_default().push(*n);
    }

    let mut any_changed = false;

    // For each group, pick a canonical node (avoid pinned nodes) and redirect predecessors of others.
    for (_c, nodes) in groups {
        let mut non_pinned: Vec<_> = nodes.into_iter().filter(|n| !pinned.contains(n)).collect();
        if non_pinned.len() < 2 {
            continue;
        }
        non_pinned.sort(); // deterministic canonical choice
        let canonical = non_pinned[0];

        for &victim in non_pinned.iter().skip(1) {
            if let Some(preds) = incoming.get(&victim) {
                for (pred, ek) in preds {
                    if let Some(mut w) = pred.write(god) {
                        let dest_map = w.children_mut().entry(ek.clone()).or_default();
                        if dest_map.remove(&victim).is_some() {
                            dest_map.insert(canonical, ());
                            any_changed = true;
                        }
                    }
                }
            }
        }
    }

    any_changed
}

fn compute_and_print_template_stats(
    templates: &[(IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index)],
    god: &IntermediateTrie3GodWrapper,
    phase: &str,
) {
    if !is_debug_level_enabled(2) { return; }

    let mut total_nodes_sum = 0; // Sum of nodes in each template if treated separately
    let mut union_nodes: HashSet<IntermediatePrecomputeNode3Index> = HashSet::new();
    let mut node_coverage: HashMap<IntermediatePrecomputeNode3Index, usize> = HashMap::new();

    for (start_node, _end_node) in templates {
        let nodes_in_template = Trie::all_nodes(god, &[*start_node]);
        total_nodes_sum += nodes_in_template.len();
        for node in nodes_in_template {
            union_nodes.insert(node);
            *node_coverage.entry(node).or_insert(0) += 1;
        }
    }

    let shared_nodes_count = node_coverage.values().filter(|&&count| count > 1).count();
    let unique_nodes_count = union_nodes.len();
    let sharing_factor = if total_nodes_sum > 0 {
        (total_nodes_sum as f64) / (unique_nodes_count as f64)
    } else {
        1.0
    };
    let shared_pct = if unique_nodes_count > 0 {
        (shared_nodes_count as f64) * 100.0 / (unique_nodes_count as f64)
    } else {
        0.0
    };

    let mut total_edges = 0;
    for node in &union_nodes {
        if let Some(g) = node.read(god) {
            for (_ek, dm) in g.children() {
                total_edges += dm.len();
            }
        }
    }

    println!("\n--- Global Template Stats ({}) ---", phase);
    println!("  Total templates: {}", templates.len());
    println!("  Sum of nodes (if unshared): {}", total_nodes_sum);
    println!("  Unique nodes across all templates: {}", unique_nodes_count);
    println!("  Total edges across all templates: {}", total_edges);
    println!("  Nodes shared by >= 2 templates: {} ({:.1}%)", shared_nodes_count, shared_pct);
    println!("  Sharing factor (sum_nodes / unique_nodes): {:.2}x", sharing_factor);
    println!("--------------------------------------------");
}

// Run a global optimization across all per-terminal templates.
// Pins all (start,end) nodes to keep external references valid.
pub fn optimize_intermediate_trie3_templates_global(
    templates: &[(IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index)],
    god: &IntermediateTrie3GodWrapper,
) {
    if templates.is_empty() { return; }

    compute_and_print_template_stats(templates, god, "Before Optimization");

    let start_nodes: Vec<_> = templates.iter().map(|(s, _)| *s).collect();
    let mut pinned: std::collections::HashSet<IntermediatePrecomputeNode3Index> = std::collections::HashSet::new();
    for (s, e) in templates {
        pinned.insert(*s);
        pinned.insert(*e);
    }

    // A few global passes: compress NoOp chains and merge identical subgraphs across all templates,
    // then prune per template to drop detritus.
    for _ in 0..3 {
        let mut changed = false;
        changed |= compress_noop_only_nodes(&start_nodes, &pinned, god);
        changed |= structural_merge_nodes_in_subgraph(&start_nodes, &pinned, god);
        for (s, e) in templates {
            changed |= prune_unproductive_nodes(&[*s], e, god);
        }
        if !changed {
            break;
        }
    }

    compute_and_print_template_stats(templates, god, "After Optimization");
}
