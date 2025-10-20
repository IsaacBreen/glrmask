// src/constraint_precompute3_intermediate_utils.rs
use crate::{
    constraint::{
        IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey,
        IntermediateTrie3GodWrapper, LLMTokenBV,
    },
    datastructures::trie::Trie,
};
use kdam::tqdm;
use ordered_hash_map::OrderedHashMap;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use crate::{profiler::PROGRESS_BAR_ENABLED};

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
struct NodeSignature {
    end: bool,
    // For each edge key, store the sorted list of child colors.
    edges: Vec<(IntermediateTrie3EdgeKey, Vec<usize>)>,
}

fn build_node_signature(
    idx: IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
    color_map: &BTreeMap<IntermediatePrecomputeNode3Index, usize>,
) -> NodeSignature {
    let guard = idx.read(god).expect("Invalid index during signature build");
    let end = guard.value.end;

    // Collect children grouped by edge key, mapped to their current colors.
    let mut edges: Vec<(IntermediateTrie3EdgeKey, Vec<usize>)> = Vec::new();
    for (ek, dsts) in guard.children().iter() {
        let mut cols: Vec<usize> = dsts
            .keys()
            .map(|d| *color_map.get(d).unwrap_or(&0usize))
            .collect();
        cols.sort_unstable();
        edges.push((ek.clone(), cols));
    }
    edges.sort(); // ensure deterministic order

    NodeSignature { end, edges }
}

fn contract_noop_chains(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) -> bool {
    let mut changed_anything = false;
    let all_nodes = Trie::all_nodes(god, roots);
    if all_nodes.is_empty() {
        return false;
    }

    // Phase 1: Identify all nodes that are part of a simple NoOp chain.
    // A node is a candidate if it's not an end node and has a single outgoing NoOp edge
    // to a single destination.
    let mut bypass_target: HashMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> =
        HashMap::new();
    for idx in &all_nodes {
        let guard = idx.read(god).expect("read");
        if guard.value.end {
            continue;
        }

        if guard.children().len() == 1 {
            if let Some((key, dests)) = guard.children().iter().next() {
                if matches!(key, IntermediateTrie3EdgeKey::NoOp) && dests.len() == 1 {
                    let dest_idx = *dests.keys().next().unwrap();
                    bypass_target.insert(*idx, dest_idx);
                }
            }
        }
    }

    if bypass_target.is_empty() {
        return false;
    }

    // Phase 2: Resolve bypass chains. e.g., if A->B and B->C, update map to A->C.
    // This makes rewiring direct to the end of the chain.
    for node_idx in bypass_target.keys().copied().collect::<Vec<_>>() {
        let mut target = bypass_target[&node_idx];
        let mut visited = HashSet::from([node_idx]);
        while let Some(next_target) = bypass_target.get(&target) {
            if !visited.insert(target) {
                // Cycle of NoOp nodes detected. This is a "true cycle" and should have been caught.
                // Break to prevent an infinite loop.
                break;
            }
            target = *next_target;
        }
        bypass_target.insert(node_idx, target);
    }

    // Phase 3: Rewire the graph.
    // For each node, check if any of its children should be bypassed.
    for idx in &all_nodes {
        if bypass_target.contains_key(idx) {
            continue;
        }

        let mut write_guard = idx.write(god).expect("write");
        let old_children = std::mem::take(write_guard.children_mut());
        let mut local_change = false;

        for (key, dests) in old_children {
            let mut new_dests = OrderedHashMap::new();
            for (dest_idx, ev) in dests {
                let final_dest = bypass_target.get(&dest_idx).copied().unwrap_or(dest_idx);
                if final_dest != dest_idx {
                    local_change = true;
                }
                new_dests.insert(final_dest, ev);
            }
            if !new_dests.is_empty() {
                // It's possible for multiple old (key, dests) to now have the same key.
                // We must merge them correctly.
                let entry = write_guard.children_mut().entry(key).or_default();
                for (dest, ev) in new_dests {
                    entry.insert(dest, ev);
                }
            }
        }
        if local_change {
            changed_anything = true;
        }
    }

    changed_anything
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

    // Initial colors: based on 'end' flag and out-degree for faster convergence.
    let mut colors: BTreeMap<IntermediatePrecomputeNode3Index, usize> = BTreeMap::new();
    let mut initial_color_map: BTreeMap<(bool, usize), usize> = BTreeMap::new();
    let mut next_initial_color = 0;
    for idx in &reachable {
        let g = idx.read(god).expect("read");
        let end = g.value.end;
        let out_degree = g.children().values().map(|dests| dests.len()).sum();
        let key = (end, out_degree);
        let color = *initial_color_map.entry(key).or_insert_with(|| {
            let c = next_initial_color;
            next_initial_color += 1;
            c
        });
        colors.insert(*idx, color);
    }

    println!("[optimize_intermediate_trie3] Initial classes: {}", next_initial_color);

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

fn rewire_to_canonical(
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
    for idx in &reachable {
        let c = colors[idx];
        match canon_by_color.get(&c) {
            Some(existing) => {
                if idx.as_usize() < existing.as_usize() {
                    canon_by_color.insert(c, *idx);
                }
            }
            None => {
                canon_by_color.insert(c, *idx);
            }
        }
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

    // Prepare rewiring plan: for each src and edge key, move edges to canonical destination.
    let mut plan: BTreeMap<
        IntermediatePrecomputeNode3Index,
        BTreeMap<IntermediateTrie3EdgeKey, Vec<(IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index)>>
    > = BTreeMap::new();

    for src in &reachable {
        if let Some(r) = src.read(god) {
            for (ek, dsts) in r.children().iter() {
                for (dst, _ev) in dsts.iter() {
                    let c = colors[dst];
                    let canon_dst = *canon_by_color.get(&c).unwrap();
                    if canon_dst != *dst {
                        plan.entry(*src)
                            .or_default()
                            .entry(ek.clone())
                            .or_default()
                            .push((*dst, canon_dst));
                    }
                }
            }
        }
    }

    // Apply rewiring plan.
    let mut edges_rewired = 0usize;
    for (src, by_key) in plan {
        if let Some(mut w) = src.write(god) {
            for (ek, mods) in by_key {
                if let Some(map) = w.get_mut(&ek) {
                    for (old_dst, new_dst) in mods {
                        if old_dst == new_dst {
                            continue;
                        }
                        if let Some(ev) = map.remove(&old_dst) {
                            // Insert the new edge. This correctly handles cases where multiple children
                            // are remapped to the same canonical destination.
                            map.insert(new_dst, ev);
                            edges_rewired += 1;
                        }
                    }
                }
            }
        }
    }

    (merges, edges_rewired)
}

fn prune_unproductive_paths_intermediate_trie3(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) {
    println!("[optimize_intermediate_trie3] Pruning nodes that cannot reach an end node...");
    if roots.is_empty() {
        return;
    }

    let all_nodes = Trie::all_nodes(god, roots);
    if all_nodes.is_empty() {
        return;
    }

    // 1. Build reverse adjacency list: dest -> sources
    let mut incoming: HashMap<IntermediatePrecomputeNode3Index, Vec<IntermediatePrecomputeNode3Index>> = HashMap::new();
    for src in &all_nodes {
        let g = src.read(god).expect("read");
        for (_ek, dm) in g.children() {
            for (dst, _ev) in dm {
                incoming.entry(*dst).or_default().push(*src);
            }
        }
    }

    // 2. Initialize worklist with all end nodes
    let mut productive: HashSet<IntermediatePrecomputeNode3Index> = HashSet::new();
    let mut q: VecDeque<IntermediatePrecomputeNode3Index> = VecDeque::new();
    let mut end_nodes_count = 0usize;
    for n in &all_nodes {
        let r = n.read(god).expect("read");
        if r.value.end {
            end_nodes_count += 1;
            if productive.insert(*n) {
                q.push_back(*n);
            }
        }
    }

    if end_nodes_count == 0 {
        println!("[optimize_intermediate_trie3] No end nodes found; skipping pruning.");
        return;
    }

    // 3. Reverse BFS to find all productive nodes
    while let Some(d) = q.pop_front() {
        if let Some(srcs) = incoming.get(&d) {
            for s in srcs {
                if productive.insert(*s) {
                    q.push_back(*s);
                }
            }
        }
    }

    let total_nodes = all_nodes.len();
    let productive_nodes = productive.len();
    let prunable = total_nodes.saturating_sub(productive_nodes);
    println!(
        "[optimize_intermediate_trie3] End-reachability: total={}, productive={}, prunable={}",
        total_nodes, productive_nodes, prunable
    );
    if prunable == 0 {
        return;
    }

    // 4. Remove any edge to a non-productive destination
    for n in &all_nodes {
        let mut w = n.write(god).expect("write");
        let old_children = std::mem::take(w.children_mut());
        for (ek, dm) in old_children {
            let mut new_dm = ordered_hash_map::OrderedHashMap::new();
            for (dst, ev) in dm {
                if productive.contains(&dst) {
                    new_dm.insert(dst, ev);
                }
            }
            if !new_dm.is_empty() {
                w.children_mut().insert(ek, new_dm);
            }
        }
    }

    // 5. Recompute all max depths
    Trie::recompute_all_max_depths(god, roots);

    println!("[optimize_intermediate_trie3] Finished end-reachability pruning.");
}

fn gc_preserving_ends(
    god: &IntermediateTrie3GodWrapper,
    roots: &[IntermediatePrecomputeNode3Index],
) {
    let mut effective_roots = roots.to_vec();

    // Arena::to_vec() is expensive, but it's the way to get all items.
    // We need to preserve all end nodes, even if they become disconnected.
    let all_entries = god.to_vec();
    for (idx, trie) in all_entries {
        if trie.value.end {
            effective_roots.push(IntermediatePrecomputeNode3Index::new(idx));
        }
    }
    effective_roots.sort();
    effective_roots.dedup();
    Trie::gc(god, &effective_roots);
}

pub fn optimize_intermediate_trie3(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
    is_end: impl Fn(IntermediatePrecomputeNode3Index, &IntermediatePrecomputeNode3) -> bool,
) -> BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> {
    let original_god = god.deep_clone();
    let original_roots = roots.to_vec();

    has_true_cycle_intermediate_trie3(god, roots);

    prune_unproductive_paths_intermediate_trie3(roots, god);

    println!("[optimize_intermediate_trie3] Contracting NoOp chains...");
    let mut passes = 0;
    while contract_noop_chains(roots, god) {
        passes += 1;
        println!("[optimize_intermediate_trie3] NoOp chain contraction pass {} complete.", passes);
        if passes > 10 {
             println!("[optimize_intermediate_trie3] WARN: NoOp chain contraction took too many passes, breaking.");
             break;
        }
    }
    if passes > 0 {
        println!("[optimize_intermediate_trie3] NoOp chains contracted. Running GC.");
        gc_preserving_ends(god, roots);
    } else {
        println!("[optimize_intermediate_trie3] NoOp chains contracted.");
    }
    has_true_cycle_intermediate_trie3(god, roots);

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
    let (merges, edges_rewired) = rewire_to_canonical(&colors, roots, god, &mut node_map);
    println!(
        "[optimize_intermediate_trie3] Rewiring done: merges={}, edges_rewired={}",
        merges, edges_rewired
    );

    // Check equivalence after optimization (currently no-op)
    let new_roots: Vec<_> = original_roots
        .iter()
        .map(|r| *node_map.get(r).unwrap_or(r))
        .collect();

    println!("[optimize_intermediate_trie3] Running final GC.");
    gc_preserving_ends(god, &new_roots);

    // assert!(
    //     are_intermediate_trie3_graphs_equal(&original_roots, &original_god, &new_roots, god, &is_end, 25),
    //     "Optimization failed to preserve graph equivalence for all roots"
    // );

    has_true_cycle_intermediate_trie3(god, &new_roots);

    node_map
}

type CycleReportIntermediate3 = Vec<(IntermediatePrecomputeNode3Index, Option<IntermediateTrie3EdgeKey>)>;

fn detect_true_cycle_recursive_intermediate_trie3(
    node_idx: IntermediatePrecomputeNode3Index,
    edge_key_opt: Option<IntermediateTrie3EdgeKey>,
    god: &IntermediateTrie3GodWrapper,
    recursion_stack: &mut HashMap<IntermediatePrecomputeNode3Index, usize>,
    visited: &mut HashSet<IntermediatePrecomputeNode3Index>,
    path: &mut Vec<(IntermediatePrecomputeNode3Index, Option<IntermediateTrie3EdgeKey>)>,
) -> Option<CycleReportIntermediate3> {
    path.push((node_idx, edge_key_opt));

    if let Some(&path_start_idx) = recursion_stack.get(&node_idx) {
        let cycle_path = path[path_start_idx..].to_vec();
        path.pop();
        return Some(cycle_path);
    }

    if visited.contains(&node_idx) {
        path.pop();
        return None;
    }

    recursion_stack.insert(node_idx, path.len() - 1);

    let children_to_visit = if let Some(guard) = node_idx.read(god) {
        guard.children().clone()
    } else {
        recursion_stack.remove(&node_idx);
        path.pop();
        return None;
    };

    for (edge_key, dest_map) in children_to_visit.iter() {
        match edge_key {
            IntermediateTrie3EdgeKey::CheckLLM(_) | IntermediateTrie3EdgeKey::NoOp => {
                for (child_idx, _) in dest_map.iter() {
                    if let Some(report) = detect_true_cycle_recursive_intermediate_trie3(
                        *child_idx,
                        Some(edge_key.clone()),
                        god,
                        recursion_stack,
                        visited,
                        path,
                    ) {
                        return Some(report);
                    }
                }
            }
            IntermediateTrie3EdgeKey::Pop(_, _) | IntermediateTrie3EdgeKey::Push(_) => {
                // These edges break "true" cycles, so we don't traverse them.
            }
        }
    }

    recursion_stack.remove(&node_idx);
    visited.insert(node_idx);
    path.pop();
    None
}

pub fn has_true_cycle_intermediate_trie3(
    god: &IntermediateTrie3GodWrapper,
    roots: &[IntermediatePrecomputeNode3Index],
) {
    let mut visited: HashSet<IntermediatePrecomputeNode3Index> = HashSet::new();
    for &root in roots {
        if !visited.contains(&root) {
            continue;
        }
        if let Some(cycle_path) = detect_true_cycle_recursive_intermediate_trie3(
            root,
            None,
            god,
            &mut HashMap::new(),
            &mut visited,
            &mut Vec::new(),
        ) {
            let mut report = format!(
                "Stack-neutral cycle detected in intermediate precompute3 trie.\nCycle path:\n"
            );
            for i in 0..cycle_path.len() {
                let (node_idx, _) = &cycle_path[i];
                let next_i = (i + 1) % cycle_path.len();
                let (next_node_idx, edge_to_next_opt) = &cycle_path[next_i];
                let edge_str = edge_to_next_opt.as_ref().map_or_else(
                    || " (root edge)".to_string(),
                    |ek| format!("{}", ek),
                );
                report.push_str(&format!("  {} --[{}]--> {}\n", node_idx, edge_str, next_node_idx));
            }
            panic!("{}", report);
        }
    }
}
