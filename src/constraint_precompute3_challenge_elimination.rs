// src/constraint_precompute3_challenge_elimination.rs
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::Instant;
use crate::constraint::{IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey, IntermediateTrie3GodWrapper, LLMTokenBV, StateIDBV};
use crate::constraint::IntermediatePrecomputedNodeContents3;
use crate::datastructures::trie::Trie;
use crate::profiler::PROGRESS_BAR_ENABLED;
use crate::r#macro::is_debug_level_enabled;
use crate::tokenizer::TokenizerStateID;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

/// If true, runs both trie-based and path-based elimination, compares them,
/// and if a mismatch is found, attempts to find a minimal failing input graph.
/// This adds significant overhead and should only be used for debugging the
/// elimination logic itself.
const DEBUG_MISMATCHES: bool = true;

/// Maximum stack depth during trie-based elimination. Prevents infinite loops
/// on graphs with cycles that contain unbalanced Push operations.
const MAX_STACK_DEPTH: usize = 64;
const MAX_ELIMINATION_PASSES: usize = 16;

/// Mark nodes hot early by visit count; lower is safer to avoid blow-ups.
const HOT_NODE_VISIT_THRESHOLD: u64 = 2048;
/// If a node accumulates too many distinct pending stacks, mark it hot.
/// This is counted when a new (source, stack) pair is first seen.
const UNIQUE_STACKS_PER_NODE_THRESHOLD: usize = 256;
/// Global hard cap on distinct product states. Once exceeded, force immediate push emission globally.
/// This preserves correctness (we just stop deferring pushes everywhere).
const PAIR_CACHE_HARD_LIMIT: usize = 1_000_000;

/// Normalizes a path for comparison purposes.
/// - Removes NoOp edges.
/// - Collects all CheckLLM bitvectors, intersects them, and prepends a single CheckLLM.
fn normalize_path(path: Vec<IntermediateTrie3EdgeKey>) -> Vec<IntermediateTrie3EdgeKey> {
    let mut combined_llm_bv = LLMTokenBV::max_ones();
    let mut has_llm_check = false;

    let mut other_ops: Vec<IntermediateTrie3EdgeKey> = path
        .into_iter()
        .filter(|ek| {
            if let IntermediateTrie3EdgeKey::CheckLLM(bv) = ek {
                combined_llm_bv &= bv;
                has_llm_check = true;
                false // remove from path
            } else {
                !matches!(ek, IntermediateTrie3EdgeKey::NoOp)
            }
        })
        .collect();

    if has_llm_check {
        other_ops.insert(0, IntermediateTrie3EdgeKey::CheckLLM(combined_llm_bv));
    }

    other_ops
}
/// Eliminates adjacent Push/Pop pairs from a stack of intermediate trie edge keys.
/// Additionally, Pop(0, B) constraints are folded into the preceding Push via intersection.
/// This is a core part of simplifying the precompute3 graph.
///
/// Pairing rules applied when a Push(A) looks to the nearest Pop(n, B) to its right
/// (stopping if another Push is encountered first):
/// - n == 0: If A intersects B, remove Pop(0, B) and keep Push(A). If disjoint, the stack is invalid (None).
/// - n == 1: If A intersects B, both cancel (epsilon). If disjoint, the stack is invalid (None).
/// - n > 1: Remove Push(A) and replace Pop with Pop(n - 1, B) unconditionally (no intersection check).
/// The scan repeats until no more changes can be made.
fn simplify_path(
    stack: Vec<IntermediateTrie3EdgeKey>,
) -> Option<Vec<IntermediateTrie3EdgeKey>> {
    let mut stack = stack;
    loop {
        let mut changed_in_pass = false;
        let mut i = 0;
        while i < stack.len() {
            if let IntermediateTrie3EdgeKey::Push(push_states) = &stack[i] {
                let push_states = push_states.clone();
                // Find nearest pop to the right, not blocked by another push
                let mut pop_j = None;
                for j in (i + 1)..stack.len() {
                    if matches!(stack[j], IntermediateTrie3EdgeKey::Push(_)) {
                        break; // Blocked
                    }
                    if matches!(stack[j], IntermediateTrie3EdgeKey::Pop(_, _)) {
                        pop_j = Some(j);
                        break;
                    }
                }

                if let Some(j) = pop_j {
                    // Found a pair to cancel
                    let pop_op = stack.remove(j);
                    let _push_op = stack.remove(i); // push is at i

                    if let IntermediateTrie3EdgeKey::Pop(n, pop_states) = pop_op {
                        match n {
                            0 => {
                                if push_states.is_disjoint(&pop_states) {
                                    return None; // Mismatch on state check
                                }
                                // Fold Pop(0, B) constraint into preceding Push(A) as A := A ∩ B.
                                // This preserves semantics and reduces the state space.
                                let mut new_push_bv = push_states.clone();
                                new_push_bv &= pop_states;
                                // If intersection is empty we'd have returned None above.
                                stack.insert(i, IntermediateTrie3EdgeKey::Push(new_push_bv));
                            }
                            1 => {
                                if push_states.is_disjoint(&pop_states) {
                                    return None; // Mismatch on single-pop check
                                }
                                // Intersection: both cancel -> epsilon (nothing to insert).
                            }
                            _ => {
                                // n > 1: remove Push unconditionally and decrement Pop.
                                stack.insert(i, IntermediateTrie3EdgeKey::Pop(n - 1, pop_states));
                            }
                        }
                    }
                    changed_in_pass = true;
                    // Restart scan from beginning of modified stack
                    i = 0;
                    continue;
                }
            }
            i += 1;
        }
        if !changed_in_pass {
            break;
        }
    }
    Some(stack)
}

pub fn eliminate_pushes_and_pops_path_based(
    roots: &mut BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) {
    // 1. Get all paths from the original graph.
    let all_root_indices: Vec<_> = roots.values().cloned().collect();
    if all_root_indices.is_empty() {
        return;
    }
    let all_paths = IntermediatePrecomputeNode3::get_all_paths(god, &all_root_indices, |n| n.value.end);

    // 2. Simplify them.
    let mut simplified_paths = BTreeSet::new();
    for (_root_value, path_edges) in all_paths {
        let edge_keys: Vec<_> = path_edges.into_iter().map(|(ek, _, _)| ek).collect();
        if let Some(new_path) = simplify_path(edge_keys) {
            simplified_paths.insert(new_path);
        }
    }

    // 3. Clear the graph and rebuild a single trie from all simplified paths.
    god.clear();

    if simplified_paths.is_empty() {
        // If all paths were eliminated, clear the roots.
        roots.clear();
        return;
    }

    // Create a single root for the new trie.
    let has_only_empty_path = simplified_paths.len() == 1 && simplified_paths.iter().next().unwrap().is_empty();
    let root_content = if has_only_empty_path {
        IntermediatePrecomputedNodeContents3::leaf()
    } else {
        IntermediatePrecomputedNodeContents3::internal()
    };
    let new_root = god.insert(Trie::new(root_content)).into();

    let mut node_cache: BTreeMap<Vec<IntermediateTrie3EdgeKey>, IntermediatePrecomputeNode3Index> = BTreeMap::new();
    node_cache.insert(vec![], new_root);

    for path in simplified_paths {
        if path.is_empty() {
            continue; // Handled by root node creation
        }
        let mut current_node_idx = new_root;
        for i in 0..path.len() {
            let edge = &path[i];
            let prefix = &path[0..=i];

            let next_node_idx = *node_cache.entry(prefix.to_vec()).or_insert_with(|| {
                let is_leaf = i == path.len() - 1;
                let content = if is_leaf { IntermediatePrecomputedNodeContents3::leaf() } else { IntermediatePrecomputedNodeContents3::internal() };
                god.insert(Trie::new(content)).into()
            });

            god.insert_edge_simple(current_node_idx, next_node_idx, edge.clone(), ());
            current_node_idx = next_node_idx;
        }
    }

    // 4. Update all roots to point to the new single root.
    for (_, root_idx) in roots.iter_mut() {
        *root_idx = new_root;
    }
}

pub fn eliminate_pushes_and_pops(
    roots: &mut BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) {
    if !DEBUG_MISMATCHES {
        run_trie_based_elimination(roots, god);
        return;
    }

    // --- DEBUG MODE ---
    // 1. Snapshot original graph
    let old_root_vec: Vec<_> = roots.values().cloned().collect();
    if old_root_vec.is_empty() {
        run_trie_based_elimination(roots, god); // Handle empty case, should do nothing.
        return;
    }

    // Check for cycles. The path-based comparison is only valid for acyclic graphs,
    // as get_all_paths would not terminate on a cyclic graph.
    if Trie::has_cycle(god, old_root_vec.clone()) {
        if is_debug_level_enabled(3) {
            println!("[Push/Pop Elimination] Skipping debug comparison for cyclic graph.");
        }
        run_trie_based_elimination(roots, god);
        return;
    }

    let (source_god, source_roots_vec, _) = Trie::deep_copy_subtrees(god, &old_root_vec);
    let mut source_roots_map = BTreeMap::new();
    for (sid, r_idx) in roots.keys().zip(source_roots_vec.iter()) {
        source_roots_map.insert(*sid, *r_idx);
    }

    // 2. Compare both implementations
    if check_mismatch(&source_god, &source_roots_map) {
        println!("!!! MISMATCH DETECTED BETWEEN TRIE-BASED AND PATH-BASED ELIMINATION !!!");
        println!("Starting refinement of failing input graph...");

        let (minimal_god, minimal_roots_map) = refine_mismatch(&source_god, &source_roots_map);

        // Rerun on minimal failing input to get the differing outputs for printing
        let (min_trie_paths, min_trie_god, _) = {
            let (god_copy, roots_vec_copy, _) = Trie::deep_copy_subtrees(&minimal_god, &minimal_roots_map.values().cloned().collect::<Vec<_>>());
            let mut roots_map_copy = BTreeMap::new();
            for (sid, r_idx) in minimal_roots_map.keys().zip(roots_vec_copy.iter()) {
                roots_map_copy.insert(*sid, *r_idx);
            }
            run_trie_based_elimination(&mut roots_map_copy, &god_copy);
            let paths = get_normalized_paths(&roots_map_copy, &god_copy);
            (paths, god_copy, roots_map_copy)
        };

        let (min_path_paths, min_path_god, _) = {
            let (god_copy, roots_vec_copy, _) = Trie::deep_copy_subtrees(&minimal_god, &minimal_roots_map.values().cloned().collect::<Vec<_>>());
            let mut roots_map_copy = BTreeMap::new();
            for (sid, r_idx) in minimal_roots_map.keys().zip(roots_vec_copy.iter()) {
                roots_map_copy.insert(*sid, *r_idx);
            }
            eliminate_pushes_and_pops_path_based(&mut roots_map_copy, &god_copy);
            let paths = get_normalized_paths(&roots_map_copy, &god_copy);
            (paths, god_copy, roots_map_copy)
        };

        let minimal_input_paths = get_normalized_paths(&minimal_roots_map, &minimal_god);
        println!("\n--- MINIMAL FAILING INPUT ({} paths) ---", minimal_input_paths.len());
        for (i, path) in minimal_input_paths.iter().enumerate() {
            println!("  Path {}: {:?}", i, path);
        }

        println!("\n--- TRIE-BASED OUTPUT ({} paths) ---", min_trie_paths.len());
        for (i, path) in min_trie_paths.iter().enumerate() {
            println!("  Path {}: {:?}", i, path);
        }
        println!("\n--- PATH-BASED OUTPUT ({} paths) ---", min_path_paths.len());
        for (i, path) in min_path_paths.iter().enumerate() {
            println!("  Path {}: {:?}", i, path);
        }

        panic!("Push/Pop elimination mismatch detected. See logs for details.");
    }

    // 5. If no mismatch, run the trie-based version on the actual input `god` to modify it.
    run_trie_based_elimination(roots, god);
}

/// Runs both elimination algorithms on a graph and returns true if their outputs differ.
fn check_mismatch(
    god: &IntermediateTrie3GodWrapper,
    roots: &BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
) -> bool {
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    if roots_vec.is_empty() { return false; }

    let (trie_paths, _, _) = {
        let (god_copy, roots_vec_copy, _) = Trie::deep_copy_subtrees(god, &roots_vec);
        let mut roots_map_copy = BTreeMap::new();
        for (sid, r_idx) in roots.keys().zip(roots_vec_copy.iter()) {
            roots_map_copy.insert(*sid, *r_idx);
        }
        run_trie_based_elimination(&mut roots_map_copy, &god_copy);
        (get_normalized_paths(&roots_map_copy, &god_copy), god_copy, roots_map_copy)
    };

    let (path_paths, _, _) = {
        let (god_copy, roots_vec_copy, _) = Trie::deep_copy_subtrees(god, &roots_vec);
        let mut roots_map_copy = BTreeMap::new();
        for (sid, r_idx) in roots.keys().zip(roots_vec_copy.iter()) {
            roots_map_copy.insert(*sid, *r_idx);
        }
        eliminate_pushes_and_pops_path_based(&mut roots_map_copy, &god_copy);
        (get_normalized_paths(&roots_map_copy, &god_copy), god_copy, roots_map_copy)
    };

    trie_paths != path_paths
}

/// Given a failing graph, randomly removes edges to find a smaller subgraph that still fails.
fn refine_mismatch(
    initial_god: &IntermediateTrie3GodWrapper,
    initial_roots: &BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
) -> (IntermediateTrie3GodWrapper, BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>) {
    let (mut minimal_god, minimal_roots_vec, old_to_new) = Trie::deep_copy_subtrees(initial_god, &initial_roots.values().cloned().collect::<Vec<_>>());
    let mut minimal_roots = BTreeMap::new();
    for (sid, old_root) in initial_roots {
        minimal_roots.insert(*sid, *old_to_new.get(old_root).unwrap());
    }

    let mut changed_in_pass = true;
    while changed_in_pass {
        changed_in_pass = false;

        // --- Pass 1: Try to remove edge groups systematically ---
        let current_roots_vec: Vec<_> = minimal_roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(&minimal_god, &current_roots_vec);

        let mut all_edge_groups = Vec::new();
        for node_idx in all_nodes {
            if let Some(node_r) = node_idx.read(&minimal_god) {
                for ek in node_r.children().keys() {
                    all_edge_groups.push((node_idx, ek.clone()));
                }
            }
        }

        for (source_idx, edge_key) in all_edge_groups {
            let (mut candidate_god, candidate_roots_vec, old_to_new) = Trie::deep_copy_subtrees(&minimal_god, &current_roots_vec);
            let mut candidate_roots = BTreeMap::new();
            for (sid, old_root) in &minimal_roots {
                candidate_roots.insert(*sid, *old_to_new.get(old_root).unwrap());
            }
            
            let mapped_source_idx = *old_to_new.get(&source_idx).unwrap();

            if let Some(mut node_w) = mapped_source_idx.write(&candidate_god) {
                node_w.children_mut().remove(&edge_key);
            }

            Trie::gc(&candidate_god, &candidate_roots_vec);

            if check_mismatch(&candidate_god, &candidate_roots) {
                minimal_god = candidate_god;
                minimal_roots = candidate_roots;
                changed_in_pass = true;
                let new_size = Trie::all_nodes(&minimal_god, &minimal_roots.values().cloned().collect::<Vec<_>>()).len();
                println!("... Refined mismatch by removing edge group. New size: {} nodes.", new_size);
                break; // Restart the whole process with the smaller graph
            }
        }

        if changed_in_pass {
            continue; // Restart the while loop
        }

        // --- Pass 2: Try to remove roots (if more than one) ---
        if minimal_roots.len() > 1 {
            let root_sids_to_try: Vec<_> = minimal_roots.keys().cloned().collect();
            for sid_to_remove in root_sids_to_try {
                let mut candidate_roots = minimal_roots.clone();
                candidate_roots.remove(&sid_to_remove);

                if candidate_roots.is_empty() { continue; }

                if check_mismatch(&minimal_god, &candidate_roots) {
                    minimal_roots = candidate_roots;
                    changed_in_pass = true;
                    println!("... Refined mismatch by removing a root. New root count: {}.", minimal_roots.len());
                    break; // Restart the while loop
                }
            }
        }
    }

    (minimal_god, minimal_roots)
}

/// Compute nodes that lie in SCCs (subgraph without Pop(0)/Pop(1)) that contain at least one Push edge internally.
/// Deferring Push in such SCCs can cause unbounded growth of the pending stack. We will therefore not defer Push
/// at those nodes – instead we emit Push edges immediately.
fn compute_push_cycle_nodes(
    source: &IntermediateTrie3GodWrapper,
    source_roots: &[IntermediatePrecomputeNode3Index],
) -> BTreeSet<IntermediatePrecomputeNode3Index> {
    let nodes = Trie::all_nodes(source, source_roots);
    if nodes.is_empty() {
        return BTreeSet::new();
    }
    let mut id_of: BTreeMap<IntermediatePrecomputeNode3Index, usize> = BTreeMap::new();
    for (i, ni) in nodes.iter().enumerate() {
        id_of.insert(*ni, i);
    }

    // Build adjacency for the FULL subgraph (include all edges).
    let n = nodes.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, node_idx) in nodes.iter().enumerate() {
        if let Some(node_r) = node_idx.read(source) {
            for (ek, dests) in node_r.children().iter() {
                for (dst_idx, _) in dests.iter() {
                    if let Some(&j) = id_of.get(dst_idx) {
                        adj[i].push(j);
                    }
                }
            }
        }
    }

    // Tarjan's SCC
    let mut index: usize = 0;
    let mut indices: Vec<Option<usize>> = vec![None; n];
    let mut lowlink: Vec<usize> = vec![0; n];
    let mut on_stack: Vec<bool> = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut scc_id: Vec<usize> = vec![usize::MAX; n];
    let mut scc_count: usize = 0;

    fn strongconnect(
        v: usize,
        index: &mut usize,
        indices: &mut [Option<usize>],
        lowlink: &mut [usize],
        on_stack: &mut [bool],
        stack: &mut Vec<usize>,
        scc_id: &mut [usize],
        scc_count: &mut usize,
        adj: &Vec<Vec<usize>>,
    ) {
        indices[v] = Some(*index);
        lowlink[v] = *index;
        *index += 1;
        stack.push(v);
        on_stack[v] = true;

        for &w in &adj[v] {
            if indices[w].is_none() {
                strongconnect(
                    w, index, indices, lowlink, on_stack, stack, scc_id, scc_count, adj,
                );
                lowlink[v] = lowlink[v].min(lowlink[w]);
            } else if on_stack[w] {
                lowlink[v] = lowlink[v].min(indices[w].unwrap());
            }
        }

        if Some(lowlink[v]) == indices[v] {
            loop {
                let w = stack.pop().expect("stack not empty");
                on_stack[w] = false;
                scc_id[w] = *scc_count;
                if w == v {
                    break;
                }
            }
            *scc_count += 1;
        }
    }

    for v in 0..n {
        if indices[v].is_none() {
            strongconnect(
                v,
                &mut index,
                &mut indices,
                &mut lowlink,
                &mut on_stack,
                &mut stack,
                &mut scc_id,
                &mut scc_count,
                &adj,
            );
        }
    }

    // For each SCC, check if there's an internal Push edge (source and destination both in SCC).
    let mut scc_has_internal_push: Vec<bool> = vec![false; scc_count.max(1)];
    for (i, node_idx) in nodes.iter().enumerate() {
        if let Some(node_r) = node_idx.read(source) {
            for (ek, dests) in node_r.children().iter() {
                if matches!(ek, IntermediateTrie3EdgeKey::Push(_)) {
                    for (dst_idx, _) in dests.iter() {
                        if let (Some(&from), Some(&to)) = (id_of.get(node_idx), id_of.get(dst_idx))
                        {
                            if scc_id[from] == scc_id[to] {
                                scc_has_internal_push[scc_id[from]] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut result = BTreeSet::new();
    for (i, node_idx) in nodes.iter().enumerate() {
        if scc_id[i] != usize::MAX && scc_has_internal_push[scc_id[i]] {
            result.insert(*node_idx);
        }
    }
    result
}

/// Cheap signature of the reachable graph, used to detect fixpoint across passes.
fn compute_graph_signature(
    god: &IntermediateTrie3GodWrapper,
    roots: &BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
) -> (u64, u64, u64, u64, u64, u64) {
    let root_indices: Vec<_> = roots.values().cloned().collect();
    if root_indices.is_empty() {
        return (0, 0, 0, 0, 0, 0);
    }
    let nodes = Trie::all_nodes(god, &root_indices);
    let mut pushes = 0u64;
    let mut pop0 = 0u64;
    let mut pop1 = 0u64;
    let mut popgt = 0u64;
    let mut popgt_sum_n = 0u64;
    let mut others = 0u64;
    for idx in nodes {
        if let Some(nr) = idx.read(god) {
            for (ek, dests) in nr.children().iter() {
                let multiplicity = dests.len() as u64;
                match ek {
                    IntermediateTrie3EdgeKey::Push(_) => pushes += multiplicity,
                    IntermediateTrie3EdgeKey::Pop(n, _) => {
                        if *n == 0 {
                            pop0 += multiplicity;
                        } else if *n == 1 {
                            pop1 += multiplicity;
                        } else {
                            popgt += multiplicity;
                            popgt_sum_n += (*n as u64) * multiplicity;
                        }
                    }
                    _ => others += multiplicity,
                }
            }
        }
    }
    (pushes, pop0, pop1, popgt, popgt_sum_n, others)
}

fn run_trie_based_elimination(
    roots: &mut BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) {
    // 1) Snapshot reachable subgraph under roots and operate on a copy.
    let mut sids: Vec<TokenizerStateID> = Vec::with_capacity(roots.len());
    let mut old_root_vec: Vec<IntermediatePrecomputeNode3Index> = Vec::with_capacity(roots.len());
    for (sid, idx) in roots.iter() {
        sids.push(*sid);
        old_root_vec.push(*idx);
    }
    if old_root_vec.is_empty() {
        return;
    }
    let (source, source_roots, _map) = Trie::deep_copy_subtrees(god, &old_root_vec);

    // 2) Prepare destination arena (clear, we'll rebuild).
    god.clear();

    // 3) Exact product-graph BFS over (source_node, pending_stack as Vec<StateIDBV>).
    let mut pair_cache: BTreeMap<(IntermediatePrecomputeNode3Index, Vec<StateIDBV>), IntermediatePrecomputeNode3Index> = BTreeMap::new();
    let mut work: VecDeque<(IntermediatePrecomputeNode3Index, Vec<StateIDBV>)> = VecDeque::new();

    // 4) Initialize new roots as (source_root, empty stack).
    let mut new_roots: BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index> = BTreeMap::new();
    for (sid, src_root) in sids.into_iter().zip(source_roots.into_iter()) {
        let stack = Vec::new();
        let key = (src_root, stack.clone());
        let new_root = *pair_cache.entry(key).or_insert_with(|| {
            let is_end = src_root
                .read(&source)
                .map(|r| r.value.end)
                .unwrap_or(false) && stack.is_empty();
            let node_val = if is_end {
                IntermediatePrecomputedNodeContents3::leaf()
            } else {
                IntermediatePrecomputedNodeContents3::internal()
            };
            let dest_idx = IntermediatePrecomputeNode3Index::new(god.insert(Trie::new(node_val)));
            work.push_back((src_root, stack));
            dest_idx
        });
        new_roots.insert(sid, new_root);
    }

    // Setup progress bar
    let pb = ProgressBar::new(0);
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] states={pos} {msg}")
            .expect("progress-bar"),
    );
    if !PROGRESS_BAR_ENABLED {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }
    pb.set_message("Eliminating push/pop pairs");

    // 5) BFS loop.
    let mut processed: u64 = 0;
    while let Some((src_idx, stack)) = work.pop_front() {
        processed += 1;
        pb.set_position(processed);
        if processed & 0xfff == 0 {
            pb.set_message(format!("queue={}", work.len()));
        }
        let dest_idx = *pair_cache.get(&(src_idx, stack.clone())).expect("dest exists");
        let src_guard = src_idx.read(&source).expect("source read");

        let mut new_work_items = Vec::new();
        let mut get_or_create = |src_idx: IntermediatePrecomputeNode3Index, stack: Vec<StateIDBV>| {
            let key = (src_idx, stack.clone());
            *pair_cache.entry(key).or_insert_with(|| {
                let is_end = src_idx
                    .read(&source)
                    .map(|r| r.value.end)
                    .unwrap_or(false) && stack.is_empty();
                let node_val = if is_end {
                    IntermediatePrecomputedNodeContents3::leaf()
                } else {
                    IntermediatePrecomputedNodeContents3::internal()
                };
                let dest_idx = IntermediatePrecomputeNode3Index::new(god.insert(Trie::new(node_val)));
                new_work_items.push((src_idx, stack));
                dest_idx
            })
        };

        // If we are on a source leaf (end == true) but with a non-empty pending stack,
        // flush the remaining pushes as a chain of Push edges to the (src_idx, empty) state.
        if src_guard.value.end && !stack.is_empty() {
            let final_dest_idx = get_or_create(src_idx, Vec::new());
            let items = &stack; // bottom -> top
            let mut cur = dest_idx;
            for (i, bv) in items.iter().enumerate() {
                let next = if i == items.len() - 1 {
                    final_dest_idx
                } else {
                    IntermediatePrecomputeNode3Index::new(
                        god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())),
                    )
                };
                god.insert_edge_simple(cur, next, IntermediateTrie3EdgeKey::Push(bv.clone()), ());
                cur = next;
            }
        }

        // Traverse outgoing edges and build transformed graph.
        for (ek, dests) in src_guard.children().iter() {
            match ek {
                IntermediateTrie3EdgeKey::Push(bv_new) => {
                    // Defer push by stacking and carrying via NoOp.
                    let mut new_stack = stack.clone();
                    if new_stack.len() + 1 > MAX_STACK_DEPTH {
                        // Safety cap: drop too-deep paths.
                        continue;
                    }
                    new_stack.push(bv_new.clone());
                    for (dst_src_idx, _) in dests.iter() {
                        let next_state = get_or_create(*dst_src_idx, new_stack.clone());
                        god.insert_edge_simple(dest_idx, next_state, IntermediateTrie3EdgeKey::NoOp, ());
                    }
                }
                IntermediateTrie3EdgeKey::Pop(n, pop_bv) => {
                    if *n == 0 {
                        // Non-consuming: intersect with top if present; otherwise forward as-is.
                        if stack.is_empty() {
                            for (dst_src_idx, _) in dests.iter() {
                                let next_state = get_or_create(*dst_src_idx, Vec::new());
                                god.insert_edge_simple(
                                    dest_idx,
                                    next_state,
                                    IntermediateTrie3EdgeKey::Pop(0, pop_bv.clone()),
                                    (),
                                );
                            }
                        } else {
                            let mut new_stack = stack.clone();
                            let top = new_stack.last().expect("non-empty").clone();
                            if top.is_disjoint(pop_bv) {
                                // Invalid branch: filtered out.
                                continue;
                            }
                            let mut narrowed = top;
                            narrowed &= pop_bv.clone();
                            *new_stack.last_mut().expect("non-empty") = narrowed;
                            for (dst_src_idx, _) in dests.iter() {
                                let next_state = get_or_create(*dst_src_idx, new_stack.clone());
                                god.insert_edge_simple(dest_idx, next_state, IntermediateTrie3EdgeKey::NoOp, ());
                            }
                        }
                    } else {
                        // n >= 1
                        if stack.is_empty() {
                            // No pending pushes: forward the Pop(n, ...) unchanged.
                            for (dst_src_idx, _) in dests.iter() {
                                let next_state = get_or_create(*dst_src_idx, Vec::new());
                                god.insert_edge_simple(
                                    dest_idx,
                                    next_state,
                                    IntermediateTrie3EdgeKey::Pop(*n, pop_bv.clone()),
                                    (),
                                );
                            }
                        } else {
                            // Consume as many as possible from the stack.
                            let mut k = *n;
                            let mut new_stack = stack.clone();
                            while k > 1 && !new_stack.is_empty() {
                                new_stack.pop();
                                k -= 1;
                            }
                            if k == 1 {
                                if let Some(top2) = new_stack.last() {
                                    if top2.is_disjoint(pop_bv) {
                                        // Invalid branch
                                        continue;
                                    }
                                    // Pop the remaining top (consuming).
                                    new_stack.pop();
                                    for (dst_src_idx, _) in dests.iter() {
                                        let next_state = get_or_create(*dst_src_idx, new_stack.clone());
                                        god.insert_edge_simple(dest_idx, next_state, IntermediateTrie3EdgeKey::NoOp, ());
                                    }
                                } else {
                                    // No pushes left to match: forward Pop(1, ...) as-is.
                                    for (dst_src_idx, _) in dests.iter() {
                                        let next_state = get_or_create(*dst_src_idx, new_stack.clone());
                                        god.insert_edge_simple(
                                            dest_idx,
                                            next_state,
                                            IntermediateTrie3EdgeKey::Pop(1, pop_bv.clone()),
                                            (),
                                        );
                                    }
                                }
                            } else {
                                // k > 1 and stack empty: forward the remaining Pop(k, ...) as-is.
                                for (dst_src_idx, _) in dests.iter() {
                                    let next_state = get_or_create(*dst_src_idx, new_stack.clone());
                                    god.insert_edge_simple(
                                        dest_idx,
                                        next_state,
                                        IntermediateTrie3EdgeKey::Pop(k, pop_bv.clone()),
                                        (),
                                    );
                                }
                            }
                        }
                    }
                }
                // NoOp and CheckLLM (and any other non-Push/Pop edges) are forwarded unchanged with the same pending stack.
                other => {
                    for (dst_src_idx, _) in dests.iter() {
                        let next_state = get_or_create(*dst_src_idx, stack.clone());
                        god.insert_edge_simple(dest_idx, next_state, other.clone(), ());
                    }
                }
            }
        }
        work.extend(new_work_items);
    }

    pb.finish_with_message("Done eliminating push/pop pairs");

    // 6) Replace input roots with new roots (pending stack is empty at roots).
    *roots = new_roots;
}

fn get_normalized_paths(
    roots: &BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) -> BTreeSet<Vec<IntermediateTrie3EdgeKey>> {
    let root_indices: Vec<_> = roots.values().cloned().collect();
    IntermediatePrecomputeNode3::get_all_paths(god, &root_indices, |n| n.value.end)
        .into_iter()
        .map(|(_r, p)| normalize_path(p.into_iter().map(|(ek, _, _)| ek).collect()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datastructures::trie::Trie2Index;

    fn run_test(
        input_god: &IntermediateTrie3GodWrapper,
        input_roots: &[IntermediatePrecomputeNode3Index],
    ) {
        // NOTE: This test harness works by comparing the set of all paths from root to leaf
        // before and after simplification. The `get_all_paths` helper function stops traversing
        // when it detects a cycle to avoid infinite loops. Therefore, this harness cannot
        // validate simplifications that occur *within* a cycle. A true trie-based
        // elimination algorithm would need a different testing strategy for cyclic graphs.

        // 1. Run new trie-based elimination (which currently calls the path-based one)
        let (eliminated_god, eliminated_roots, _) = Trie::deep_copy_subtrees(input_god, input_roots);
        let mut eliminated_roots_map = BTreeMap::new();
        for (i, r) in eliminated_roots.iter().enumerate() {
            eliminated_roots_map.insert(TokenizerStateID(i), *r); // Use dummy TokenizerStateID
        }
        eliminate_pushes_and_pops(&mut eliminated_roots_map, &eliminated_god);

        // 2. Flatten result to paths
        let final_roots_from_trie_elim: Vec<_> = eliminated_roots_map.values().cloned().collect();
        let paths_from_trie_elim: BTreeSet<_> = IntermediatePrecomputeNode3::get_all_paths(&eliminated_god, &final_roots_from_trie_elim, |n| n.value.end)
            .into_iter()
            .map(|(_r, p)| normalize_path(p.into_iter().map(|(ek, _, _)| ek).collect()))
            .collect();

        // 3. Run old path-based elimination directly
        let initial_paths = IntermediatePrecomputeNode3::get_all_paths(input_god, input_roots, |node| node.value.end);
        let mut paths_from_path_elim = BTreeSet::new();
        for (_root_value, path_edges) in initial_paths {
            let edge_keys: Vec<_> = path_edges.into_iter().map(|(ek, _, _)| ek).collect();
            if let Some(new_path) = simplify_path(edge_keys) {
                paths_from_path_elim.insert(normalize_path(new_path));
            }
        }

        // 4. Compare
        assert_eq!(paths_from_trie_elim, paths_from_path_elim);
    }

    #[test]
    fn test_simple_cancel() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv = StateIDBV::zeros();
        bv.insert(1);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv.clone()), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_mismatch_invalidates_path() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv2), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_pop_zero_keeps_push() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv = StateIDBV::zeros();
        bv.insert(1);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv.clone()), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(0, bv), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_pop_zero_mismatch_invalidates_path() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(0, bv2), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_pop_n_decrements() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2); // Note: disjoint, but should not matter for n>1

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(3, bv2), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_blocked_push() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv2.clone()), (), v2);
        v2.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv2), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_multiple_cancellations_in_sequence() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v3 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1.clone()), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv2.clone()), (), v2);
        v2.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv2), (), v3);
        v3.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv1), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_interleaved_ops() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut llm_bv = LLMTokenBV::zeros();
        llm_bv.insert(100);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1.clone()), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_bv), (), v2);
        v2.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv1), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_branching_and_merging() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v3 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv_a = StateIDBV::zeros();
        bv_a.insert(1);
        let mut llm_x = LLMTokenBV::zeros();
        llm_x.insert(100);
        let mut llm_y = LLMTokenBV::zeros();
        llm_y.insert(200);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_a.clone()), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_x), (), v2);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_y), (), v3);
        v2.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), end);
        v3.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_cycle_simplification() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv_a = StateIDBV::zeros();
        bv_a.insert(1);

        // Path to end
        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), end);
        // Path with cycle
        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_a.clone()), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), root);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_no_pushes_or_pops() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut llm_bv = LLMTokenBV::zeros();
        llm_bv.insert(100);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_bv), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_dangling_pop() {
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv = StateIDBV::zeros();
        bv.insert(1);

        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_challenging_gauntlet() {
        // This test combines iterative cancellation, branching, and path invalidation.
        // Path X should simplify over multiple passes.
        // Path Y should be invalidated because Push(B) blocks Push(A), and then Push(B)
        // mismatches with Pop(1, A), killing the path.
        let god = IntermediateTrie3GodWrapper::new();

        // --- Nodes ---
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v3x = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v4x = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v5x = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v3y = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let merge = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        // --- State & LLM Bitsets ---
        let mut bv_a = StateIDBV::zeros();
        bv_a.insert(1);
        let mut bv_b = StateIDBV::zeros();
        bv_b.insert(2);
        let mut bv_c = StateIDBV::zeros();
        bv_c.insert(3);

        let mut llm_x = LLMTokenBV::zeros();
        llm_x.insert(100);
        let mut llm_y = LLMTokenBV::zeros();
        llm_y.insert(200);

        // --- Graph Structure ---
        // Common prefix
        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_a.clone()), (), v1);
        v1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_b.clone()), (), v2);

        // Branching
        v2.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_x), (), v3x);
        v2.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_y), (), v3y);

        // Path X (iterative cancellation)
        v3x.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_c.clone()), (), v4x);
        v4x.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_c.clone()), (), v5x);
        v5x.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_b.clone()), (), merge);

        // Path Y (blocking and invalidation)
        v3y.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), merge);

        // Common suffix
        merge.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), end);

        // The expected outcome is that Path Y is completely eliminated.
        // Path X simplifies: Push(C)/Pop(C) cancel, then Push(B)/Pop(B) cancel.
        // The remaining path is Push(A) -> CheckLLM(X) -> merge -> Pop(1, A) -> end.
        // Then Push(A)/Pop(A) cancel.
        // The final path should be just CheckLLM(X).
        run_test(&god, &[root]);
    }

    #[test]
    fn test_mismatch_from_log() {
        // This test case is derived from a real-world mismatch found during development.
        // The trie-based elimination correctly prunes this path as invalid, while the
        // path-based simplification incorrectly simplifies it to a non-empty path.
        // The key interaction is a Pop(0, bv_5) that should invalidate a path when a
        // Push(bv_1) is on the stack, but the path-based logic overlooks this.
        let god = IntermediateTrie3GodWrapper::new();

        let nodes: Vec<_> = (0..12)
            .map(|i| {
                let content = if i == 11 {
                    IntermediatePrecomputedNodeContents3::leaf()
                } else {
                    IntermediatePrecomputedNodeContents3::internal()
                };
                Trie2Index::from(god.insert(Trie::new(content)))
            })
            .collect();
        let root = nodes[0];

        let mut llm_0 = LLMTokenBV::zeros();
        llm_0.insert(0);

        let mut bv_1 = StateIDBV::zeros();
        bv_1.insert(1);
        let mut bv_2 = StateIDBV::zeros();
        bv_2.insert(2);
        let mut bv_4 = StateIDBV::zeros();
        bv_4.insert(4);
        let mut bv_5 = StateIDBV::zeros();
        bv_5.insert(5);
        let bv_max = StateIDBV::max_ones();

        let path = vec![
            IntermediateTrie3EdgeKey::CheckLLM(llm_0),
            IntermediateTrie3EdgeKey::Pop(0, bv_4.clone()),
            IntermediateTrie3EdgeKey::Pop(2, bv_max.clone()),
            IntermediateTrie3EdgeKey::Pop(0, bv_5.clone()),
            IntermediateTrie3EdgeKey::Push(bv_1.clone()),
            IntermediateTrie3EdgeKey::Push(bv_4.clone()),
            IntermediateTrie3EdgeKey::Pop(0, bv_4.clone()),
            IntermediateTrie3EdgeKey::Pop(2, bv_max.clone()),
            IntermediateTrie3EdgeKey::Pop(0, bv_5.clone()),
            IntermediateTrie3EdgeKey::Push(bv_1.clone()),
            IntermediateTrie3EdgeKey::Push(bv_2.clone()),
        ];

        let mut current_node = root;
        for (i, edge) in path.into_iter().enumerate() {
            let next_node = nodes[i + 1];
            current_node.write(&god).unwrap().force_insert_to_node(edge, (), next_node);
            current_node = next_node;
        }

        run_test(&god, &[root]);
    }

    #[test]
    fn test_complex_cycle_simplification() {
        // This test features a cycle that should be simplified away,
        // nested within a larger Push/Pop pair that should also be simplified.
        // The structure is:
        // root -> Push(A) -> c1 -> CheckLLM(X) -> c2
        // c2 -> Push(B) -> c3 -> Pop(1, B) -> c2  (inner cycle)
        // c2 -> Pop(1, A) -> end                   (exit path)
        // The expected simplified path is just CheckLLM(X).
        let god = IntermediateTrie3GodWrapper::new();
        let root = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let c1 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let c2 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let c3 = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end = Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv_a = StateIDBV::zeros();
        bv_a.insert(1);
        let mut bv_b = StateIDBV::zeros();
        bv_b.insert(2);
        let mut llm_x = LLMTokenBV::zeros();
        llm_x.insert(100);

        // Path structure
        root.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_a.clone()), (), c1);
        c1.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_x), (), c2);
        // Inner cycle
        c2.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_b.clone()), (), c3);
        c3.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_b.clone()), (), c2);
        // Exit path
        c2.write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), end);

        run_test(&god, &[root]);
    }
}
