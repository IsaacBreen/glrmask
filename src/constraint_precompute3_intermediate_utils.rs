// src/constraint_precompute3_intermediate_utils.rs
use crate::{
    constraint::{
        IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey,
        IntermediateTrie3GodWrapper, LLMTokenBV,
    },
    datastructures::trie::Trie,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

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
            matches!(ek, IntermediateTrie3EdgeKey::Pop(_, _) | IntermediateTrie3EdgeKey::Push(_))
        };

    let get_normalized_paths = |god, roots| {
        Trie::get_all_paths_with_cycles(god, roots, is_end, is_path_edge, max_path_length)
            .into_iter()
            .map(|(_, path)| normalize_path(path.into_iter().map(|(ek, ..)| ek).collect()))
            .collect::<HashSet<_>>()
    };

    get_normalized_paths(god_a, roots_a) == get_normalized_paths(god_b, roots_b)
}

fn count_graph_stats(
    god: &IntermediateTrie3GodWrapper,
    nodes: &[IntermediatePrecomputeNode3Index],
) -> (usize, usize, usize) {
    // (total_edges, noop_edges, check_edges)
    let mut total_edges = 0usize;
    let mut noop_edges = 0usize;
    let mut check_edges = 0usize;
    for &idx in nodes {
        if let Some(g) = idx.read(god) {
            for (ek, dsts) in g.children().iter() {
                let n = dsts.len();
                total_edges += n;
                match ek {
                    IntermediateTrie3EdgeKey::NoOp => noop_edges += n,
                    IntermediateTrie3EdgeKey::CheckLLM(_) => check_edges += n,
                    _ => {}
                }
            }
        }
    }
    (total_edges, noop_edges, check_edges)
}

fn flatten_noops(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) {
    eprintln!("[opt-trie3] Flattening NoOp closures...");
    let nodes = Trie::all_nodes(god, roots);

    // Build NoOp adjacency once for faster closure queries
    let mut noop_adj: HashMap<IntermediatePrecomputeNode3Index, Vec<IntermediatePrecomputeNode3Index>> =
        HashMap::with_capacity(nodes.len());
    for &idx in &nodes {
        if let Some(g) = idx.read(god) {
            let mut v = Vec::new();
            if let Some(map) = g.get(&IntermediateTrie3EdgeKey::NoOp) {
                for (dst, _) in map.iter() {
                    v.push(*dst);
                }
            }
            noop_adj.insert(idx, v);
        }
    }

    // For each node, compute closure via NoOp-only edges, then rewrite:
    // - end flag := OR of end flags in closure
    // - children := union of all non-NoOp edges from nodes in closure
    let mut processed = 0usize;
    for &idx in &nodes {
        // BFS closure over NoOp
        let mut closure: Vec<IntermediatePrecomputeNode3Index> = Vec::new();
        let mut stack = Vec::new();
        let mut seen: HashSet<IntermediatePrecomputeNode3Index> = HashSet::new();
        stack.push(idx);
        seen.insert(idx);
        while let Some(cur) = stack.pop() {
            closure.push(cur);
            if let Some(neigh) = noop_adj.get(&cur) {
                for &n in neigh {
                    if seen.insert(n) {
                        stack.push(n);
                    }
                }
            }
        }

        // Compute aggregated end flag and edge list
        let mut new_end = false;
        let mut new_edges: BTreeMap<IntermediateTrie3EdgeKey, BTreeSet<IntermediatePrecomputeNode3Index>> =
            BTreeMap::new();

        for c in &closure {
            if let Some(gc) = c.read(god) {
                new_end |= gc.value.end;
                for (ek, dsts) in gc.children().iter() {
                    if matches!(ek, IntermediateTrie3EdgeKey::NoOp) {
                        continue;
                    }
                    let entry = new_edges.entry(ek.clone()).or_default();
                    for (dst, _) in dsts.iter() {
                        entry.insert(*dst);
                    }
                }
            }
        }

        // Rewrite node in-place
        if let Some(mut gw) = idx.write(god) {
            gw.value.end = new_end;
            gw.children_mut().clear();
            for (ek, dsts) in new_edges.into_iter() {
                for dst in dsts {
                    let mut ev = Some(());
                    gw.try_insert_unchecked(ek.clone(), &mut ev, dst);
                }
            }
        }

        processed += 1;
        if processed % 500 == 0 || processed == nodes.len() {
            eprintln!(
                "[opt-trie3] NoOp flatten progress: {}/{} nodes",
                processed,
                nodes.len()
            );
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct IsoSig {
    end: bool,
    edges: Vec<(IntermediateTrie3EdgeKey, Vec<usize>)>,
}

fn structural_dedup_partition_refinement(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) -> BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> {
    eprintln!("[opt-trie3] Structural dedup (partition refinement)...");
    let nodes = Trie::all_nodes(god, roots);

    // Initial class: by end flag only
    let mut class_of: BTreeMap<IntermediatePrecomputeNode3Index, usize> = BTreeMap::new();
    for &idx in &nodes {
        if let Some(g) = idx.read(god) {
            class_of.insert(idx, if g.value.end { 1 } else { 0 });
        }
    }

    let mut iteration = 0usize;
    loop {
        iteration += 1;
        let mut interner: HashMap<IsoSig, usize> = HashMap::new();
        let mut next_class_of: BTreeMap<IntermediatePrecomputeNode3Index, usize> = BTreeMap::new();
        let mut next_classes = 0usize;

        for &idx in &nodes {
            if let Some(g) = idx.read(god) {
                let mut edges_summarized: Vec<(IntermediateTrie3EdgeKey, Vec<usize>)> = Vec::new();
                for (ek, dsts) in g.children().iter() {
                    // Map dst -> current class; collect and sort
                    let mut child_classes: Vec<usize> = dsts
                        .iter()
                        .map(|(dst, _)| class_of.get(dst).copied().unwrap_or(0))
                        .collect();
                    child_classes.sort_unstable();
                    edges_summarized.push((ek.clone(), child_classes));
                }
                // children() is BTreeMap, so keys are already ordered; we keep that order.
                let sig = IsoSig {
                    end: g.value.end,
                    edges: edges_summarized,
                };
                let cid = if let Some(&id) = interner.get(&sig) {
                    id
                } else {
                    let id = next_classes;
                    interner.insert(sig, id);
                    next_classes += 1;
                    id
                };
                next_class_of.insert(idx, cid);
            }
        }

        let changed = next_class_of != class_of;
        eprintln!(
            "[opt-trie3]  iteration {} -> {} equivalence classes (changed: {})",
            iteration,
            next_classes,
            changed
        );
        class_of = next_class_of;
        if !changed {
            break;
        }
        // Safety net in case of unforeseen oscillation; should not happen.
        if iteration > 64 {
            eprintln!("[opt-trie3]  reached iteration limit; stopping refinement.");
            break;
        }
    }

    // Build representative per class (min index)
    let mut repr_of_class: HashMap<usize, IntermediatePrecomputeNode3Index> = HashMap::new();
    for (&idx, &cid) in class_of.iter() {
        repr_of_class
            .entry(cid)
            .and_modify(|cur| {
                if idx < *cur {
                    *cur = idx;
                }
            })
            .or_insert(idx);
    }

    // Old -> representative mapping
    let mut node_map: BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> =
        BTreeMap::new();
    for (&idx, &cid) in class_of.iter() {
        let rep = repr_of_class.get(&cid).copied().unwrap_or(idx);
        node_map.insert(idx, rep);
    }

    node_map
}

fn rewire_to_canonical(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
    node_map: &BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index>,
) {
    eprintln!("[opt-trie3] Rewiring edges to canonical representatives...");
    let nodes = Trie::all_nodes(god, roots);
    let mut processed = 0usize;
    for &idx in &nodes {
        // Snapshot existing edges
        let mut by_key: BTreeMap<
            IntermediateTrie3EdgeKey,
            BTreeSet<IntermediatePrecomputeNode3Index>,
        > = BTreeMap::new();
        if let Some(g) = idx.read(god) {
            for (ek, dsts) in g.children().iter() {
                let entry = by_key.entry(ek.clone()).or_default();
                for (dst, _) in dsts.iter() {
                    let canonical = node_map.get(dst).copied().unwrap_or(*dst);
                    entry.insert(canonical);
                }
            }
        }

        // Rewrite node edges
        if let Some(mut gw) = idx.write(god) {
            gw.children_mut().clear();
            for (ek, dsts) in by_key.into_iter() {
                for dst in dsts {
                    let mut ev = Some(());
                    gw.try_insert_unchecked(ek.clone(), &mut ev, dst);
                }
            }
        }

        processed += 1;
        if processed % 500 == 0 || processed == nodes.len() {
            eprintln!(
                "[opt-trie3] Rewire progress: {}/{} nodes",
                processed,
                nodes.len()
            );
        }
    }
}

pub fn optimize_intermediate_trie3(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
    is_end: impl Fn(IntermediatePrecomputeNode3Index, &IntermediatePrecomputeNode3) -> bool,
) -> BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> {
    let original_god = god.deep_clone();
    let original_roots = roots.to_vec();

    eprintln!("--- optimize_intermediate_trie3: start ---");
    let initial_nodes = Trie::all_nodes(god, roots);
    let (edges_total_before, noop_before, check_before) = count_graph_stats(god, &initial_nodes);
    eprintln!(
        "[opt-trie3] Initial: nodes={}, edges={} (noop={}, check={})",
        initial_nodes.len(),
        edges_total_before,
        noop_before,
        check_before
    );

    // 1) Flatten NoOp edges
    flatten_noops(roots, god);

    let after_noop_nodes = Trie::all_nodes(god, roots);
    let (edges_total_after_noop, noop_after_noop, check_after_noop) =
        count_graph_stats(god, &after_noop_nodes);
    eprintln!(
        "[opt-trie3] After NoOp flatten: nodes={}, edges={} (noop={}, check={})",
        after_noop_nodes.len(),
        edges_total_after_noop,
        noop_after_noop,
        check_after_noop
    );

    // 2) Structural dedup using partition refinement
    let node_map = structural_dedup_partition_refinement(roots, god);
    let identity_count = node_map
        .iter()
        .filter(|(k, v)| k == v)
        .count();
    eprintln!(
        "[opt-trie3] Dedup map: {} entries (identity: {}, non-identity: {})",
        node_map.len(),
        identity_count,
        node_map.len().saturating_sub(identity_count)
    );

    // 3) Rewire edges to canonical representatives
    rewire_to_canonical(roots, god, &node_map);

    // Compute new roots for equivalence test and for GC
    let new_roots: Vec<_> = original_roots
        .iter()
        .map(|r| *node_map.get(r).unwrap_or(r))
        .collect();

    // 4) GC to drop unreachable duplicates
    Trie::gc(god, &new_roots);
    let final_nodes = Trie::all_nodes(god, &new_roots);
    let (edges_total_after, noop_after, check_after) = count_graph_stats(god, &final_nodes);
    eprintln!(
        "[opt-trie3] After GC: nodes={}, edges={} (noop={}, check={})",
        final_nodes.len(),
        edges_total_after,
        noop_after,
        check_after
    );
    eprintln!("--- optimize_intermediate_trie3: end ---");

    // Check equivalence after optimization
    // The path-length limiter only counts Pop/Push edges (see are_intermediate_trie3_graphs_equal),
    // so setting this high should still pass as we preserved normalized paths.
    let max_path_length = 1_000_000usize;

    assert!(
        are_intermediate_trie3_graphs_equal(
            &original_roots,
            &original_god,
            &new_roots,
            god,
            &is_end,
            max_path_length
        ),
        "Optimization failed to preserve graph equivalence for all roots"
    );

    node_map
}

