// src/constraint_precompute3_intermediate_utils.rs
use crate::{
    constraint::{
        IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey,
        IntermediateTrie3GodWrapper, LLMTokenBV,
    },
    datastructures::trie::Trie,
};
use ordered_hash_map::OrderedHashMap;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

const DEBUG_LOG: bool = cfg!(debug_assertions);

#[derive(Debug, Clone)]
pub struct IntermediateTrie3Config {
    pub enabled: bool,
    pub verbose: bool,
    pub prune_unproductive_paths: bool,
    pub contract_noop_chains: bool,
    pub normalize_checkllm_edges: bool,
    pub contract_checkllm_chains: bool,
    pub structural_deduplication: bool,
    pub gc: bool,
}

impl Default for IntermediateTrie3Config {
    fn default() -> Self {
        Self {
            enabled: true,
            verbose: false,
            prune_unproductive_paths: false,
            contract_noop_chains: false,
            normalize_checkllm_edges: false,
            contract_checkllm_chains: false,
            structural_deduplication: true,
            gc: false,
        }
    }
}

impl IntermediateTrie3Config {
    pub fn off() -> Self {
        Self {
            enabled: false,
            verbose: false,
            prune_unproductive_paths: false,
            contract_noop_chains: false,
            normalize_checkllm_edges: false,
            contract_checkllm_chains: false,
            structural_deduplication: false,
            gc: false,
        }
    }
}

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
            matches!(ek, IntermediateTrie3EdgeKey::Pop(_, _) | IntermediateTrie3EdgeKey::Push(_) | IntermediateTrie3EdgeKey::CheckLLM(_))
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

/// Compute in-degree for all nodes reachable from roots.
fn compute_in_degrees(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) -> HashMap<IntermediatePrecomputeNode3Index, usize> {
    let nodes = Trie::all_nodes(god, roots);
    let mut indeg: HashMap<IntermediatePrecomputeNode3Index, usize> = HashMap::new();
    for n in &nodes {
        indeg.entry(*n).or_insert(0);
    }
    for n in &nodes {
        if let Some(r) = n.read(god) {
            for (_ek, dm) in r.children() {
                for (dst, _ev) in dm {
                    *indeg.entry(*dst).or_insert(0) += 1;
                }
            }
        }
    }
    indeg
}

/// Normalize CheckLLM edges on a single node:
/// 1) Aggregate the union of bitvectors per destination (merges duplicate/subsumed edges).
/// 2) Regroup by the aggregated bitvector so that a single CheckLLM(BV) key maps to a set of destinations.
///    This yields a canonical representation independent of original partitioning.
fn normalize_checkllm_edges_on_node(
    idx: IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
) -> bool {
    let mut changed = false;
    let mut w = if let Some(w) = idx.write(god) { w } else { return false };

    // Take children out to rebuild.
    let old_children = std::mem::take(w.children_mut());

    // Gather old stats for "did we change?"
    let mut old_ck_keys = 0usize;
    let mut old_ck_edges = 0usize;
    for (ek, dm) in old_children.iter() {
        if matches!(ek, IntermediateTrie3EdgeKey::CheckLLM(_)) {
            old_ck_keys += 1;
            old_ck_edges += dm.len();
        }
    }

    // Step 1: union BV per destination across all CheckLLM entries.
    let mut union_per_dest: BTreeMap<IntermediatePrecomputeNode3Index, LLMTokenBV> = BTreeMap::new();
    let mut other_edges: BTreeMap<IntermediateTrie3EdgeKey, OrderedHashMap<IntermediatePrecomputeNode3Index, ()>> = BTreeMap::new();
    for (ek, dm) in old_children.into_iter() {
        match ek {
            IntermediateTrie3EdgeKey::CheckLLM(bv) => {
                for (dst, _ev) in dm.into_iter() {
                    union_per_dest
                        .entry(dst)
                        .and_modify(|acc| *acc |= bv.clone())
                        .or_insert(bv.clone());
                }
            }
            _ => {
                other_edges.insert(ek, dm);
            }
        }
    }

    // Step 2: group destinations by their aggregated bitvector.
    let mut grouped: BTreeMap<LLMTokenBV, Vec<IntermediatePrecomputeNode3Index>> = BTreeMap::new();
    for (dst, bv) in union_per_dest.into_iter() {
        grouped.entry(bv).or_default().push(dst);
    }
    let mut new_ck_keys = 0usize;
    let mut new_ck_edges = 0usize;

    // Rebuild children.
    for (ek, dm) in other_edges.into_iter() {
        w.children_mut().insert(ek, dm);
    }
    for (bv, mut dests) in grouped.into_iter() {
        dests.sort_unstable();
        let ek = IntermediateTrie3EdgeKey::CheckLLM(bv);
        let mut dm: OrderedHashMap<IntermediatePrecomputeNode3Index, ()> = OrderedHashMap::new();
        for d in dests {
            dm.insert(d, ());
            new_ck_edges += 1;
        }
        w.children_mut().insert(ek, dm);
        new_ck_keys += 1;
    }

    if old_ck_keys != new_ck_keys || old_ck_edges != new_ck_edges {
        changed = true;
    }
    changed
}

/// Apply CheckLLM normalization to all reachable nodes. Returns true if any change occurred.
fn normalize_checkllm_edges_for_all(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) -> bool {
    let nodes = Trie::all_nodes(god, roots);
    let mut changed_any = false;
    for n in nodes {
        if normalize_checkllm_edges_on_node(n, god) {
            changed_any = true;
        }
    }
    changed_any
}

#[derive(Debug, Clone)]
struct CheckLLMChainTask {
    src: IntermediatePrecomputeNode3Index,
    src_ek: IntermediateTrie3EdgeKey, // CheckLLM(bv1)
    mid: IntermediatePrecomputeNode3Index,
    new_bv: LLMTokenBV,               // bv1 & bv2
    dst: IntermediatePrecomputeNode3Index,
}

/// Contract simple linear chains of consecutive CheckLLM checks:
///   src --[CheckLLM(bv1)]--> mid --[CheckLLM(bv2)]--> dst
/// with constraints:
///   - mid has in-degree == 1,
///   - mid is not an end node,
///   - each step has a single destination (no branching along this chain).
/// Then replace the first edge with CheckLLM(bv1 & bv2) directly to dst.
fn contract_checkllm_chains(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) -> bool {
    let reachable = Trie::all_nodes(god, roots);
    if reachable.is_empty() {
        return false;
    }
    let mut indeg = compute_in_degrees(roots, god);
    let mut tasks: Vec<CheckLLMChainTask> = Vec::new();

    for src in &reachable {
        let Some(rsrc) = src.read(god) else { continue };
        for (ek1, dm1) in rsrc.children().iter() {
            let bv1 = if let IntermediateTrie3EdgeKey::CheckLLM(bv) = ek1 {
                bv.clone()
            } else {
                continue;
            };
            if dm1.len() != 1 {
                continue;
            }
            let (&mid, _) = dm1.iter().next().unwrap();
            if indeg.get(&mid).copied().unwrap_or(0) != 1 {
                continue;
            }
            let Some(rmid) = mid.read(god) else { continue };
            if rmid.value.end {
                continue; // cannot bypass acceptance point
            }
            if rmid.children().len() != 1 {
                continue;
            }
            let (ek2, dm2) = rmid.children().iter().next().unwrap();
            let bv2 = if let IntermediateTrie3EdgeKey::CheckLLM(bv) = ek2 {
                bv.clone()
            } else {
                continue;
            };
            if dm2.len() != 1 {
                continue;
            }
            let (&dst, _) = dm2.iter().next().unwrap();
            let mut new_bv = bv1.clone();
            new_bv &= bv2.clone();
            tasks.push(CheckLLMChainTask {
                src: *src,
                src_ek: ek1.clone(),
                mid,
                new_bv,
                dst,
            });
        }
    }

    if tasks.is_empty() {
        return false;
    }

    // Apply rewiring tasks.
    let mut changed = false;
    for t in tasks {
        if let Some(mut wsrc) = t.src.write(god) {
            if let Some(dm) = wsrc.get_mut(&t.src_ek) {
                if dm.remove(&t.mid).is_some() {
                    changed = true;
                }
                if dm.is_empty() {
                    wsrc.children_mut().remove(&t.src_ek);
                }
            }
            let new_ek = IntermediateTrie3EdgeKey::CheckLLM(t.new_bv);
            wsrc.children_mut().entry(new_ek).or_default().insert(t.dst, ());
        }
        // Update in-degree snapshot for subsequent rewires in the same pass.
        if let Some(v) = indeg.get_mut(&t.mid) {
            *v = v.saturating_sub(1);
        }
        *indeg.entry(t.dst).or_insert(0) += 1;
    }
    changed
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
    if DEBUG_LOG {
        println!(
            "[optimize_intermediate_trie3] Reachable nodes: {}, edges: {}",
            reachable.len(),
            edge_count
        );
    }

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

    if DEBUG_LOG {
        println!(
            "[optimize_intermediate_trie3] Initial classes: {}",
            next_initial_color
        );
    }

    // Iteratively refine until stable.
    let mut iter = 0usize;
    loop {
        iter += 1;
        let mut sigs: Vec<(IntermediatePrecomputeNode3Index, NodeSignature)> =
            Vec::with_capacity(reachable.len());
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

        if DEBUG_LOG {
            println!(
                "[optimize_intermediate_trie3] WL iteration {}: classes={}, changed={}",
                iter,
                new_colors.values().copied().collect::<HashSet<_>>().len(),
                changed
            );
        }

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
    if DEBUG_LOG {
        println!(
            "[optimize_intermediate_trie3] Canonicalization: classes={}, merges={}",
            canon_by_color.len(),
            merges
        );
    }

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
    if DEBUG_LOG { println!("[optimize_intermediate_trie3] Pruning nodes that cannot reach an end node..."); }
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
        if DEBUG_LOG { println!("[optimize_intermediate_trie3] No end nodes found; skipping pruning."); }
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
    if DEBUG_LOG {
        // Log only summary above; the per-node pruning happens below.
    }
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

    if DEBUG_LOG { println!("[optimize_intermediate_trie3] Finished end-reachability pruning."); }
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
    config: &IntermediateTrie3Config,
) -> BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> {
    if !config.enabled {
        return BTreeMap::new();
    }
    let original_roots = roots.to_vec();

    // Helper for verbose printing
    let print_graph = |pass_name: &str| {
        if config.verbose {
            println!("\n--- Graph state after: {} ---", pass_name);
            let options = crate::datastructures::trie::PrettyPrintOptions::default()
                .display_edge_keys_only()
                .omit_depth()
                .display_nodes();
            println!("{}", Trie::pretty_print_with_options(god, roots, &options));
        }
    };

    if config.verbose {
        print_graph("Initial State");
    }

    if DEBUG_LOG { has_true_cycle_intermediate_trie3(god, roots); }

    if config.prune_unproductive_paths {
        prune_unproductive_paths_intermediate_trie3(roots, god);
        print_graph("Prune Unproductive Paths");
    }

    // Pass A: aggressively remove simple NoOp chains (epsilon-like edges).
    if config.contract_noop_chains {
        if DEBUG_LOG { println!("[optimize_intermediate_trie3] Contracting NoOp chains..."); }
        let mut passes = 0;
        let mut changed_in_loop = false;
        while contract_noop_chains(roots, god) {
            passes += 1;
            changed_in_loop = true;
            if DEBUG_LOG { println!("[optimize_intermediate_trie3] NoOp chain contraction pass {} complete.", passes); }
            if passes > 10 {
                 if DEBUG_LOG { println!("[optimize_intermediate_trie3] WARN: NoOp chain contraction took too many passes, breaking."); }
                 break;
            }
        }
        if changed_in_loop {
            print_graph("Contract NoOp Chains");
        }
        if passes > 0 {
            if DEBUG_LOG { println!("[optimize_intermediate_trie3] NoOp chains contracted. Running GC."); }
            if config.gc {
                gc_preserving_ends(god, roots);
                print_graph("GC after NoOp Contraction");
            }
        } else {
            if DEBUG_LOG { println!("[optimize_intermediate_trie3] NoOp chains contracted."); }
        }
    }
    if DEBUG_LOG { has_true_cycle_intermediate_trie3(god, roots); }

    // Pass B: normalize CheckLLM partitions, then contract linear CheckLLM chains.
    let mut changed_any = false;
    if config.normalize_checkllm_edges {
        if DEBUG_LOG { println!("[optimize_intermediate_trie3] Normalizing CheckLLM edges..."); }
        if normalize_checkllm_edges_for_all(roots, god) {
            changed_any = true;
            print_graph("Normalize CheckLLM Edges");
        }
    }
    if config.contract_checkllm_chains {
        if DEBUG_LOG { println!("[optimize_intermediate_trie3] Contracting CheckLLM chains..."); }
        let mut chain_passes = 0usize;
        let mut contracted_chains = false;
        while contract_checkllm_chains(roots, god) {
            chain_passes += 1;
            changed_any = true;
            contracted_chains = true;
            if chain_passes > 10 {
                if DEBUG_LOG { println!("[optimize_intermediate_trie3] WARN: CheckLLM chain contraction took too many passes, breaking."); }
                break;
            }
        }
        if contracted_chains {
            print_graph("Contract CheckLLM Chains");
        }
    }
    if changed_any {
        if DEBUG_LOG { println!("[optimize_intermediate_trie3] CheckLLM normalization/chain contraction made changes. Running GC."); }
        if config.gc {
            gc_preserving_ends(god, roots);
            print_graph("GC after CheckLLM Ops");
        }
    }
    if DEBUG_LOG { has_true_cycle_intermediate_trie3(god, roots); }

    let mut node_map: BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> =
        BTreeMap::default();

    if config.structural_deduplication {
        // Pass 1: color-refinement-based structural deduplication with progress reporting.
        if DEBUG_LOG { println!("[optimize_intermediate_trie3] Starting structural deduplication (WL color refinement)..."); }
        let (colors, iters, classes, total_nodes) = wl_color_refine(roots, god);
        if DEBUG_LOG {
            println!(
                "[optimize_intermediate_trie3] Refinement complete: iterations={}, classes={}, nodes={}",
                iters, classes, total_nodes
            );
        }

        // Normalize again before canonical rewiring to maximize equality exposure.
        if config.normalize_checkllm_edges && normalize_checkllm_edges_for_all(roots, god) {
            if DEBUG_LOG { println!("[optimize_intermediate_trie3] Normalization revealed new opportunities pre-canonicalization."); }
            print_graph("Pre-Deduplication CheckLLM Normalization");
        }

        let (merges, edges_rewired) = rewire_to_canonical(&colors, roots, god, &mut node_map);
        if merges > 0 || edges_rewired > 0 {
            print_graph("Structural Deduplication (Rewire)");
        }
        if DEBUG_LOG {
            println!(
                "[optimize_intermediate_trie3] Rewiring done: merges={}, edges_rewired={}",
                merges, edges_rewired
            );
        }

        // Quick polish pass: normalization + WL + canonicalization can sometimes collapse further.
        let mut polish_changed = false;
        if config.normalize_checkllm_edges && normalize_checkllm_edges_for_all(roots, god) {
            polish_changed = true;
            print_graph("Polish Pass: CheckLLM Normalization");
        }
        if polish_changed {
            if DEBUG_LOG { println!("[optimize_intermediate_trie3] Polishing: rerunning WL refinement after normalization..."); }
            let (colors2, iters2, classes2, total_nodes2) = wl_color_refine(roots, god);
            if DEBUG_LOG {
                println!(
                    "[optimize_intermediate_trie3] Polish refinement complete: iterations={}, classes={}, nodes={}",
                    iters2, classes2, total_nodes2
                );
            }
            let (merges2, edges_rewired2) = rewire_to_canonical(&colors2, roots, god, &mut node_map);
            if merges2 > 0 || edges_rewired2 > 0 {
                print_graph("Structural Deduplication (Polish Rewire)");
            }
            if DEBUG_LOG {
                println!(
                    "[optimize_intermediate_trie3] Polish rewiring done: merges={}, edges_rewired={}",
                    merges2, edges_rewired2
                );
            }
        }
    }

    // Check equivalence after optimization (currently no-op)
    let new_roots: Vec<_> = original_roots
        .iter()
        .map(|r| *node_map.get(r).unwrap_or(r))
        .collect();

    if config.gc {
        if DEBUG_LOG { println!("[optimize_intermediate_trie3] Running final GC."); }
        gc_preserving_ends(god, &new_roots);
        print_graph("Final GC");
    }

    // assert!(
    //     are_intermediate_trie3_graphs_equal(&original_roots, &original_god, &new_roots, god, &is_end, 25),
    //     "Optimization failed to preserve graph equivalence for all roots"
    // );

    if DEBUG_LOG { has_true_cycle_intermediate_trie3(god, &new_roots); }

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
        if visited.contains(&root) {
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
