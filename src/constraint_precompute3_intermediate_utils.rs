// src/constraint_precompute3_intermediate_utils.rs
use crate::{
    constraint::{
        IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey,
        IntermediateTrie3GodWrapper, LLMTokenBV,
    },
    datastructures::trie::Trie,
};
use std::collections::{BTreeMap, HashMap, HashSet, BTreeSet};
use ordered_hash_map::OrderedHashMap;

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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum StructuralEdgeKey {
    Pop(usize),
    Push,
    CheckLLM,
    NoOp,
}

impl From<&IntermediateTrie3EdgeKey> for StructuralEdgeKey {
    fn from(ek: &IntermediateTrie3EdgeKey) -> Self {
        match ek {
            IntermediateTrie3EdgeKey::Pop(n, _) => Self::Pop(*n),
            IntermediateTrie3EdgeKey::Push(_) => Self::Push,
            IntermediateTrie3EdgeKey::CheckLLM(_) => Self::CheckLLM,
            IntermediateTrie3EdgeKey::NoOp => Self::NoOp,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct NodeSignature {
    end: bool,
    // For each structural edge type, store the sorted multiset of child colors.
    // Using BTreeMap ensures canonical representation.
    edges: BTreeMap<StructuralEdgeKey, Vec<usize>>,
}
fn build_node_signature(
    idx: IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
    color_map: &BTreeMap<IntermediatePrecomputeNode3Index, usize>,
) -> NodeSignature {
    let guard = idx.read(god).expect("Invalid index during signature build");
    let end = guard.value.end;
    // Aggregate child colors under their structural edge key.
    let mut edges_aggr: BTreeMap<StructuralEdgeKey, Vec<usize>> = BTreeMap::new();
    for (ek, dsts) in guard.children().iter() {
        let key = StructuralEdgeKey::from(ek);
        let cols = dsts.keys().map(|d| *color_map.get(d).unwrap_or(&0));
        edges_aggr.entry(key).or_default().extend(cols);
    }

    // Sort the color lists to make the signature canonical.
    let mut edges: BTreeMap<StructuralEdgeKey, Vec<usize>> = BTreeMap::new();
    for (key, mut cols) in edges_aggr {
        cols.sort_unstable();
        edges.insert(key, cols);
    }

    NodeSignature { end, edges }
}

fn wl_color_refine(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) -> (BTreeMap<IntermediatePrecomputeNode3Index, usize>, usize, usize, usize) {
    // Restrict to reachable nodes to keep work bounded.
    let mut reachable = Trie::all_nodes(god, roots);
    reachable.sort_unstable();
    reachable.dedup();

    // Compute an edge count for progress reporting.
    let mut edge_count: usize = 0;
    for idx in &reachable {
        if let Some(g) = idx.read(god) {
            for (_ek, dsts) in g.children().iter() {
                edge_count += dsts.len();
            }
        }
    }
    println!(
        "[optimize_intermediate_trie3] Reachable nodes: {}, edges: {}",
        reachable.len(),
        edge_count
    );

    // Initial colors: separate by 'end' flag only (0 = not end, 1 = end).
    let mut colors: BTreeMap<IntermediatePrecomputeNode3Index, usize> = BTreeMap::new();
    for idx in &reachable {
        let end = idx.read(god).map(|g| g.value.end).unwrap_or(false);
        colors.insert(*idx, if end { 1 } else { 0 });
    }

    // Iteratively refine until stable.
    let mut iter = 0usize;
    loop {
        iter += 1;
        let mut sigs: Vec<(IntermediatePrecomputeNode3Index, NodeSignature)> = Vec::with_capacity(reachable.len());
        for idx in &reachable {
            let sig = build_node_signature(*idx, god, &colors);
            sigs.push((*idx, sig));
        }

        // Intern signatures to compact color IDs.
        let mut intern: BTreeMap<NodeSignature, usize> = BTreeMap::new();
        let mut next_color_id: usize = 0;
        let mut new_colors: BTreeMap<IntermediatePrecomputeNode3Index, usize> = BTreeMap::new();

        for (idx, sig) in sigs {
            let cid = if let Some(c) = intern.get(&sig) {
                *c
            } else {
                let c = next_color_id;
                intern.insert(sig, c);
                next_color_id += 1;
                c
            };
            new_colors.insert(idx, cid);
        }

        // Compare with old colors to check for stability.
        let mut changed = 0usize;
        for (idx, new_c) in &new_colors {
            if colors.get(idx).copied().unwrap_or(usize::MAX) != *new_c {
                changed += 1;
            }
        }

        println!(
            "[optimize_intermediate_trie3] WL iteration {}: classes={}, changed={}",
            iter,
            new_colors.values().copied().collect::<HashSet<_>>().len(),
            changed
        );

        colors = new_colors;
        if changed == 0 {
            break;
        }
    }

    // Count classes and returns stats.
    let classes = colors.values().copied().collect::<HashSet<_>>().len();
    (colors, iter, classes, reachable.len())
}
fn merge_and_rewire_nodes(
    colors: &BTreeMap<IntermediatePrecomputeNode3Index, usize>,
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
    node_map: &mut BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index>,
) -> (usize, usize) {
    // Compute reachable again to scope work.
    let mut reachable = Trie::all_nodes(god, roots);
    reachable.sort_unstable();
    reachable.dedup();

    // Build canonical representative per color (pick minimum index).
    let mut canon_by_color: HashMap<usize, IntermediatePrecomputeNode3Index> = HashMap::new();
    for (idx, &c) in colors {
        canon_by_color.entry(c).or_insert(*idx);
    }
    // Build node->canonical map (only store changed ones into node_map)
    let mut merges = 0usize;
    for idx in &reachable {
        let c = colors[idx];
        let canon = *canon_by_color.get(&c).expect("canonical missing");
        if *idx != canon {
            merges += 1;
            node_map.insert(*idx, canon);
        }
    }
    println!(
        "[optimize_intermediate_trie3] Canonicalization: classes={}, merges={}",
        canon_by_color.len(),
        merges
    );
    // For each canonical representative, rebuild its edges by merging all edges
    // from all nodes in its equivalence class.
    for (color, canon_idx) in &canon_by_color {
        let nodes_in_class: Vec<_> = colors.iter()
            .filter(|(_, &c)| c == *color)
            .map(|(idx, _)| *idx)
            .collect();

        if nodes_in_class.len() <= 1 {
            continue;
        }

        // Aggregate all outgoing edges from all nodes in the class.
        // New edges will point to the canonical representatives of their original destinations.
        let mut merged_children = BTreeMap::new();
        for node_idx in &nodes_in_class {
            if let Some(guard) = node_idx.read(god) {
                for (ek, dsts) in guard.children().iter() {
                    for (dst, ev) in dsts.iter() {
                        // Remap destination to its canonical representative
                        let canon_dst = *node_map.get(dst).unwrap_or(dst);
                        let dest_map_for_key: &mut OrderedHashMap<_, _> = merged_children.entry(ek.clone()).or_default();
                        // The event value is just (), so we can just insert.
                        dest_map_for_key.insert(canon_dst, *ev);
                    }
                }
            }
        }

        // Now, write the merged edges to the canonical node.
        if let Some(mut writer) = canon_idx.write(god) {
            let mut new_children_ordered = BTreeMap::new();
            for (ek, dsts) in merged_children {
                let mut new_dsts_map = ordered_hash_map::OrderedHashMap::new();
                for (dst, ev) in dsts {
                    new_dsts_map.insert(dst, ev);
                }
                new_children_ordered.insert(ek, new_dsts_map);
            }
            *writer.children_mut() = new_children_ordered;
        }
    }

    (merges, 0)
}
pub fn optimize_intermediate_trie3(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) -> BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> {
    let original_god = god.deep_clone();
    let original_roots = roots.to_vec();

    let node_map: BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> =
        BTreeMap::default();

    // Pass 1: color-refinement-based structural deduplication with progress reporting.
    println!("[optimize_intermediate_trie3] Starting structural deduplication (WL color refinement)...");
    let (colors, iters, classes, total_nodes) = wl_color_refine(roots, god);
    println!(
        "[optimize_intermediate_trie3] Refinement complete: iterations={}, classes={}, nodes={}",
        iters, classes, total_nodes
    );
    let mut node_map = node_map; // make mutable for update
    let (merges, edges_rewired) = merge_and_rewire_nodes(&colors, roots, god, &mut node_map);
    println!(
        "[optimize_intermediate_trie3] Rewiring done: merges={}, edges_rewired={}",
        merges, edges_rewired
    );
    // After merging, non-canonical nodes are unreachable from canonical ones.
    // We need to update the roots and then GC.
    let new_roots: Vec<_> = roots.iter().map(|r| *node_map.get(r).unwrap_or(r)).collect();
    Trie::gc(god, &new_roots);
    // assert!(
    //     are_intermediate_trie3_graphs_equal(&original_roots, &original_god, &new_roots, god, &|_, n| n.value.end, 25),
    //     "Optimization failed to preserve graph equivalence for all roots"
    // );
    node_map
}

