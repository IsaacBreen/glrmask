// src/constraint_precompute3_intermediate_utils.rs
use crate::{
    constraint::{
        IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey,
        IntermediateTrie3GodWrapper, LLMTokenBV,
    },
    datastructures::trie::Trie,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

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
    // Count all edges except NoOp towards the max_path_length bound. This keeps CheckLLM cycles in check.
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct NormalizedNodeSignature {
    // End-frontiers reachable via only NoOp/CheckLLM from this node.
    // None => no CheckLLM seen; Some(bv) => normalized CheckLLM(bv) precedes termination.
    end_frontiers: Vec<Option<LLMTokenBV>>,
    // First non-(NoOp|CheckLLM) frontier ops (Push/Pop), with optional accumulated CheckLLM,
    // mapped to the sorted multiset of child colors behind that op.
    edges: Vec<(IntermediateTrie3EdgeKey, Option<LLMTokenBV>, Vec<usize>)>,
}

fn build_normalized_node_signature(
    idx: IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
    color_map: &BTreeMap<IntermediatePrecomputeNode3Index, usize>,
) -> NormalizedNodeSignature {
    // BFS over (node, accumulated_check) states following only NoOp and CheckLLM,
    // emitting frontiers on first Push/Pop or "end" nodes.
    let mut end_frontiers: BTreeSet<Option<LLMTokenBV>> = BTreeSet::new();
    let mut frontier: BTreeMap<(IntermediateTrie3EdgeKey, Option<LLMTokenBV>), Vec<usize>> =
        BTreeMap::new();

    let mut visited: BTreeMap<IntermediatePrecomputeNode3Index, BTreeSet<Option<LLMTokenBV>>> =
        BTreeMap::new();
    let mut q: VecDeque<(IntermediatePrecomputeNode3Index, Option<LLMTokenBV>)> = VecDeque::new();

    visited.entry(idx).or_default().insert(None);
    q.push_back((idx, None));

    while let Some((cur, acc_opt)) = q.pop_front() {
        let guard = cur
            .read(god)
            .unwrap_or_else(|| panic!("Invalid index during normalized signature build: {:?}", cur));

        // If we can terminate here (before any Push/Pop), record an end-frontier with current accumulated check.
        if guard.value.end {
            end_frontiers.insert(acc_opt.clone());
        }

        for (ek, dsts) in guard.children().iter() {
            match ek {
                IntermediateTrie3EdgeKey::NoOp => {
                    for (dst, _ev) in dsts.iter() {
                        let seen = visited.entry(*dst).or_default();
                        if seen.insert(acc_opt.clone()) {
                            q.push_back((*dst, acc_opt.clone()));
                        }
                    }
                }
                IntermediateTrie3EdgeKey::CheckLLM(bv) => {
                    // Accumulate CheckLLM by intersection; None means "no check yet".
                    let new_acc_opt = if let Some(ref a) = acc_opt {
                        let mut m = a.clone();
                        m &= bv.clone();
                        Some(m)
                    } else {
                        Some(bv.clone())
                    };
                    for (dst, _ev) in dsts.iter() {
                        let seen = visited.entry(*dst).or_default();
                        if seen.insert(new_acc_opt.clone()) {
                            q.push_back((*dst, new_acc_opt.clone()));
                        }
                    }
                }
                // First non-(NoOp|CheckLLM) operation => frontier edge.
                IntermediateTrie3EdgeKey::Push(_) | IntermediateTrie3EdgeKey::Pop(_, _) => {
                    // Collect colors of destinations reached via this (op, acc_opt).
                    let key = (ek.clone(), acc_opt.clone());
                    let entry = frontier.entry(key).or_default();
                    for (dst, _ev) in dsts.iter() {
                        entry.push(*color_map.get(dst).unwrap_or(&0usize));
                    }
                }
            }
        }
    }

    // Normalize frontier child color buckets.
    let mut edges: Vec<(IntermediateTrie3EdgeKey, Option<LLMTokenBV>, Vec<usize>)> = Vec::new();
    for ((ek, acc), mut cols) in frontier {
        cols.sort_unstable();
        edges.push((ek, acc, cols));
    }
    edges.sort(); // deterministic

    let end_frontiers: Vec<Option<LLMTokenBV>> = end_frontiers.into_iter().collect();

    NormalizedNodeSignature {
        end_frontiers,
        edges,
    }
}

fn wl_color_refine_normalized(
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

    // Initial colors: coarse (all zero). The normalized frontier signature will refine it rapidly.
    let mut colors: BTreeMap<IntermediatePrecomputeNode3Index, usize> = BTreeMap::new();
    for idx in &reachable {
        colors.insert(*idx, 0);
    }

    // Iteratively refine until stable.
    let mut iter = 0usize;
    loop {
        iter += 1;
        let mut sigs: Vec<(IntermediatePrecomputeNode3Index, NormalizedNodeSignature)> =
            Vec::with_capacity(reachable.len());
        for idx in &reachable {
            let sig = build_normalized_node_signature(*idx, god, &colors);
            sigs.push((*idx, sig));
        }

        // Intern signatures to compact color IDs.
        let mut intern: BTreeMap<NormalizedNodeSignature, usize> = BTreeMap::new();
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
            "[optimize_intermediate_trie3] Normalized WL iteration {}: classes={}, changed={}",
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
                            // Only insert if not already present; otherwise drop the ev.
                            if !map.contains_key(&new_dst) {
                                map.insert(new_dst, ev);
                            }
                            edges_rewired += 1;
                        }
                    }
                }
            }
        }
    }

    (merges, edges_rewired)
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

    // Pass 1: color-refinement-based structural deduplication with progress reporting.
    println!("[optimize_intermediate_trie3] Starting normalized-path deduplication (WL color refinement)...");
    let (colors, iters, classes, total_nodes) = wl_color_refine_normalized(roots, god);
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

    // assert!(
    //     are_intermediate_trie3_graphs_equal(&original_roots, &original_god, &new_roots, god, &is_end, 25),
    //     "Optimization failed to preserve graph equivalence for all roots"
    // );

    node_map
}

/// Perform normalized-path-aware optimization across the union of all templates.
/// This enables maximal sharing across templates in the same arena.
pub fn optimize_intermediate_trie3_templates_global(
    all_roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
    is_end: impl Fn(IntermediatePrecomputeNode3Index, &IntermediatePrecomputeNode3) -> bool,
) -> BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> {
    let original_god = god.deep_clone();
    let original_roots = all_roots.to_vec();

    println!(
        "[optimize_intermediate_trie3] Global optimization over {} roots",
        all_roots.len()
    );

    // Global normalized WL refinement.
    let (colors, iters, classes, total_nodes) = wl_color_refine_normalized(all_roots, god);
    println!(
        "[optimize_intermediate_trie3] Global refinement: iterations={}, classes={}, nodes={}",
        iters, classes, total_nodes
    );

    // Canonicalize globally.
    let mut node_map: BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> =
        BTreeMap::default();
    let (merges, edges_rewired) = rewire_to_canonical(&colors, all_roots, god, &mut node_map);
    println!(
        "[optimize_intermediate_trie3] Global rewiring: merges={}, edges_rewired={}",
        merges, edges_rewired
    );

    // Post-check (keep ready if you want to assert equivalence globally).
    let new_roots: Vec<_> = original_roots
        .iter()
        .map(|r| *node_map.get(r).unwrap_or(r))
        .collect();
    // assert!(
    //     are_intermediate_trie3_graphs_equal(&original_roots, &original_god, &new_roots, god, &is_end, 25),
    //     "Global optimization failed to preserve graph equivalence"
    // );

    node_map
}

