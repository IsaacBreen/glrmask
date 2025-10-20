// src/constraint_precompute3_intermediate_utils.rs
use crate::{
    constraint::{
        IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey,
        IntermediateTrie3GodWrapper, LLMTokenBV,
    },
    datastructures::trie::{MergeableEdgeValue, Trie},
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Normalizes a path for comparison purposes.
/// - Removes NoOp edges.
/// - Collects all CheckLLM bitvectors, intersects them, and prepends a single CheckLLM.
pub(crate) fn normalize_path(path: Vec<IntermediateTrie3EdgeKey>) -> Vec<IntermediateTrie3EdgeKey> {
    let mut combined_llm_bv = LLMTokenBV::max_ones();
    let mut has_llm_check = false;

    let mut other_ops: Vec<_> = path
        .into_iter()
        .filter_map(|ek| match ek {
            IntermediateTrie3EdgeKey::CheckLLM(bv) => {
                combined_llm_bv &= bv;
                has_llm_check = true;
                None
            }
            IntermediateTrie3EdgeKey::NoOp => None,
            IntermediateTrie3EdgeKey::Push(_) | IntermediateTrie3EdgeKey::Pop(_, _) => Some(ek),
        })
        .collect();

    if has_llm_check {
        other_ops.insert(0, IntermediateTrie3EdgeKey::CheckLLM(combined_llm_bv));
    }

    other_ops
}

/// Compares two Intermediate Trie3 graphs for equivalence by comparing their sets of normalized paths.
/// This is a strong equivalence check, suitable for testing optimization passes.
pub fn are_intermediate_trie3_graphs_equal<F>(
    roots_a: &[IntermediatePrecomputeNode3Index],
    god_a: &IntermediateTrie3GodWrapper,
    roots_b: &[IntermediatePrecomputeNode3Index],
    god_b: &IntermediateTrie3GodWrapper,
    is_end: &F,
    max_path_length: usize,
) -> bool
where
    F: Fn(IntermediatePrecomputeNode3Index, &IntermediatePrecomputeNode3) -> bool,
{
    // Only Pop and Push operations count towards path length for cycle detection.
    let is_path_edge: fn(&IntermediateTrie3EdgeKey, &(), IntermediatePrecomputeNode3Index) -> bool =
        |ek, _, _| {
            !matches!(ek, IntermediateTrie3EdgeKey::NoOp)
        };

    let get_normalized_paths = |god, roots| {
        Trie::get_all_paths_with_cycles(god, roots, is_end, is_path_edge, max_path_length)
            .into_iter()
            .map(|(_, path)| normalize_path(path.into_iter().map(|(ek, ..)| ek).collect()))
            .collect::<HashSet<_>>()
    };

    get_normalized_paths(god_a, roots_a) == get_normalized_paths(god_b, roots_b)
}

fn reachable_nodes(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) -> Vec<IntermediatePrecomputeNode3Index> {
    Trie::all_nodes(god, roots)
        .into_iter()
        .collect()
}

#[derive(Clone)]
struct NodeAdj {
    // Flattened list of outgoing labeled edges (one per destination)
    edges: Vec<(IntermediateTrie3EdgeKey, IntermediatePrecomputeNode3Index)>,
    end: bool,
}

fn snapshot_reachable(
    nodes: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) -> BTreeMap<IntermediatePrecomputeNode3Index, NodeAdj> {
    let mut map: BTreeMap<IntermediatePrecomputeNode3Index, NodeAdj> = BTreeMap::new();
    for idx in nodes {
        let idxv = *idx;
        let (end, edges) = god
            .with(idxv.as_index(), |node: &IntermediatePrecomputeNode3| {
                let end = node.value.end;
                // Collect all edges: one entry per (edge_key, dst) pair
                let mut edges: Vec<(IntermediateTrie3EdgeKey, IntermediatePrecomputeNode3Index)> =
                    Vec::new();
                for (ek, dsts) in node.children().iter() {
                    // OrderedHashMap<Trie2Index, EV> -> iterate keys
                    for (dst, _ev) in dsts.iter() {
                        edges.push((ek.clone(), *dst));
                    }
                }
                (end, edges)
            })
            .unwrap_or((false, Vec::new()));
        map.insert(
            idxv,
            NodeAdj {
                end,
                edges,
            },
        );
    }
    map
}

fn compute_labels_and_groups(
    nodes: &[IntermediatePrecomputeNode3Index],
    adj: &BTreeMap<IntermediatePrecomputeNode3Index, NodeAdj>,
) -> (BTreeMap<IntermediatePrecomputeNode3Index, u64>, BTreeMap<u64, Vec<IntermediatePrecomputeNode3Index>>) {
    // Iterative color refinement (Weisfeiler-Lehman style) until fixpoint.
    // Start with label based on (end flag, outdegree signature).
    let mut labels: BTreeMap<IntermediatePrecomputeNode3Index, u64> = BTreeMap::new();
    for idx in nodes {
        let na = adj.get(idx).unwrap();
        let mut h = DefaultHasher::new();
        na.end.hash(&mut h);
        // A quick initial signature: multiset of edge keys only (ignoring dst); stable but coarse.
        let mut keys: Vec<_> = na.edges.iter().map(|(ek, _)| ek).cloned().collect();
        keys.sort();
        keys.hash(&mut h);
        labels.insert(*idx, h.finish());
    }

    let mut changed = true;
    let mut iter = 0usize;
    while changed && iter < 64 {
        iter += 1;
        changed = false;
        let mut new_labels: BTreeMap<IntermediatePrecomputeNode3Index, u64> = BTreeMap::new();
        for idx in nodes {
            let na = adj.get(idx).unwrap();
            // Build a sorted signature over (edge_key, label(dst))
            let mut sig: Vec<(IntermediateTrie3EdgeKey, u64)> = Vec::with_capacity(na.edges.len());
            for (ek, dst) in na.edges.iter() {
                let dl = *labels.get(dst).unwrap_or(&0);
                sig.push((ek.clone(), dl));
            }
            sig.sort_unstable_by(|a, b| {
                let ord = a.0.cmp(&b.0);
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
                a.1.cmp(&b.1)
            });
            // Hash the node's own "end" plus its outgoing signature
            let mut h = DefaultHasher::new();
            na.end.hash(&mut h);
            sig.hash(&mut h);
            new_labels.insert(*idx, h.finish());
        }
        // Check if changed
        if new_labels != labels {
            changed = true;
            let unique = {
                let mut s = BTreeSet::new();
                for v in new_labels.values() {
                    s.insert(*v);
                }
                s.len()
            };
            println!(
                "[optimize_trie3] iteration {}: refined to {} label classes",
                iter, unique
            );
            labels = new_labels;
        }
    }
    // Group by label
    let mut groups: BTreeMap<u64, Vec<IntermediatePrecomputeNode3Index>> = BTreeMap::new();
    for (idx, l) in labels.iter() {
        groups.entry(*l).or_default().push(*idx);
    }
    (labels, groups)
}

fn build_canonical_map(
    groups: &BTreeMap<u64, Vec<IntermediatePrecomputeNode3Index>>,
) -> BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> {
    let mut map: BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> =
        BTreeMap::new();
    for (_label, members) in groups.iter() {
        if members.is_empty() {
            continue;
        }
        // Choose canonical representative: the minimum index for stability
        let mut best = members[0];
        for m in members.iter().copied() {
            if m < best {
                best = m;
            }
        }
        for m in members {
            map.insert(*m, best);
        }
    }
    map
}

fn retarget_edges_to_canonical(
    nodes: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
    canon: &BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index>,
) -> (usize, usize) {
    // Returns (num_nodes_changed, num_edges_retargeted)
    let mut nodes_changed = 0usize;
    let mut edges_changed = 0usize;
    for idx in nodes {
        let idxv = *idx;
        // Snapshot current edges
        let (end, edges) = god
            .with(idxv.as_index(), |node: &IntermediatePrecomputeNode3| {
                let end = node.value.end;
                let mut edges: Vec<(IntermediateTrie3EdgeKey, IntermediatePrecomputeNode3Index)> =
                    Vec::new();
                for (ek, dsts) in node.children().iter() {
                    for (dst, _ev) in dsts.iter() {
                        edges.push((ek.clone(), *dst));
                    }
                }
                (end, edges)
            })
            .unwrap_or((false, Vec::new()));

        // Compute the new target list under canonical mapping; deduplicate (edge_key, dst)
        let mut seen: BTreeSet<(IntermediateTrie3EdgeKey, IntermediatePrecomputeNode3Index)> =
            BTreeSet::new();
        let mut new_edges: Vec<(IntermediateTrie3EdgeKey, IntermediatePrecomputeNode3Index)> =
            Vec::with_capacity(edges.len());
        let mut changed_here = false;
        for (ek, dst) in edges.iter().cloned() {
            let mapped = *canon.get(&dst).unwrap_or(&dst);
            if mapped != dst {
                changed_here = true;
                edges_changed += 1;
            }
            let pair = (ek.clone(), mapped);
            if !seen.contains(&pair) {
                seen.insert(pair.clone());
                new_edges.push(pair);
            }
        }

        if changed_here {
            nodes_changed += 1;
            // Rewrite edges: clear and re-insert
            let _ = god.with_mut(idxv.as_index(), |node: &mut IntermediatePrecomputeNode3| {
                // Rebuild adjacency from scratch to avoid tricky in-place map edits
                node.children_mut().clear();
                for (ek, dst) in new_edges.iter().cloned() {
                    let mut ev = Some(());
                    node.try_insert_unchecked(ek, &mut ev, dst);
                }
                // end flag remains unchanged
            });
        }
        // else: no change needed for this node
        let _ = end; // keep end for clarity (not modified)
    }
    (nodes_changed, edges_changed)
}

fn gc_with_mapped_roots(
    roots: &[IntermediatePrecomputeNode3Index],
    canon: &BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) -> Vec<IntermediatePrecomputeNode3Index> {
    let new_roots: Vec<_> = roots
        .iter()
        .map(|r| *canon.get(r).unwrap_or(r))
        .collect();
    // GC unreachable nodes after rewiring to canonical subgraphs
    IntermediatePrecomputeNode3::gc(god, &new_roots);
    new_roots
}

pub fn optimize_intermediate_trie3(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
    is_end: impl Fn(IntermediatePrecomputeNode3Index, &IntermediatePrecomputeNode3) -> bool,
) -> BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> {
    let original_god = god.deep_clone();
    let original_roots = roots.to_vec();

    let node_map: BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> =
        BTreeMap::default();

    // Progress: initial stats
    let stats_before = IntermediatePrecomputeNode3::stats(god, roots);
    println!(
        "[optimize_trie3] start: nodes={}, edges={}",
        stats_before.num_reachable_nodes, stats_before.num_reachable_edges
    );

    // Phase 1: collect reachable nodes and snapshot their adjacency
    let reachable = reachable_nodes(roots, god);
    println!(
        "[optimize_trie3] reachable nodes from {} roots: {}",
        roots.len(),
        reachable.len()
    );
    let adj = snapshot_reachable(&reachable, god);

    // Phase 2: iterative color refinement (bisimulation-ish) to identify isomorphic subgraphs
    let (_labels, groups) = compute_labels_and_groups(&reachable, &adj);

    // Phase 3: build canonical mapping (old -> canonical representative)
    let canonical_map = build_canonical_map(&groups);
    let classes = groups.len();
    let unified = reachable.len().saturating_sub(classes);
    println!(
        "[optimize_trie3] dedup candidates: {} classes over {} nodes (unify {} nodes)",
        classes,
        reachable.len(),
        unified
    );

    // Phase 4: rewire all edges to canonical representatives (no structural change in semantics)
    let (nodes_changed, edges_rewired) = retarget_edges_to_canonical(&reachable, god, &canonical_map);
    println!(
        "[optimize_trie3] edges rewired: {} across {} nodes",
        edges_rewired, nodes_changed
    );

    // Phase 5: GC with canonicalized roots to drop unreachable duplicates
    let canonical_roots = gc_with_mapped_roots(roots, &canonical_map, god);
    let stats_after = IntermediatePrecomputeNode3::stats(god, &canonical_roots);
    println!(
        "[optimize_trie3] after GC: nodes={}, edges={}",
        stats_after.num_reachable_nodes, stats_after.num_reachable_edges
    );

    // Build the final node_map to return: for all original reachable nodes
    let mut result_map: BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> =
        BTreeMap::new();
    for idx in reachable.iter().copied() {
        let mapped = *canonical_map.get(&idx).unwrap_or(&idx);
        result_map.insert(idx, mapped);
    }

    // Check equivalence after optimization (currently no-op)
    let new_roots: Vec<_> = original_roots
        .iter()
        .map(|r| *result_map.get(r).unwrap_or(r))
        .collect();

    assert!(
        are_intermediate_trie3_graphs_equal(&original_roots, &original_god, &new_roots, god, &is_end, 25),
        "Optimization failed to preserve graph equivalence for all roots"
    );

    println!(
        "[optimize_trie3] done: nodes -{} ({} -> {}), edges -{} ({} -> {})",
        stats_before
            .num_reachable_nodes
            .saturating_sub(stats_after.num_reachable_nodes),
        stats_before.num_reachable_nodes,
        stats_after.num_reachable_nodes,
        stats_before
            .num_reachable_edges
            .saturating_sub(stats_after.num_reachable_edges),
        stats_before.num_reachable_edges,
        stats_after.num_reachable_edges
    );

    result_map
}

