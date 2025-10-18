// src/constraint_precompute3_challenge_elimination.rs
use crate::constraint::IntermediatePrecomputedNodeContents3;
use crate::constraint::{
    IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey,
    IntermediateTrie3GodWrapper, LLMTokenBV, StateIDBV,
};
use crate::datastructures::trie::Trie;
use crate::r#macro::is_debug_level_enabled;
use crate::tokenizer::TokenizerStateID;
use std::collections::{BTreeMap, BTreeSet};

use std::collections::VecDeque;
use std::collections::btree_map::Entry;

/// If true, runs both trie-based and path-based elimination, compares them,
/// and if a mismatch is found, attempts to find a minimal failing input graph.
/// This adds significant overhead and should only be used for debugging the
/// elimination logic itself.
const DEBUG_MISMATCHES: bool = true;

fn debug_mismatches_enabled() -> bool {
    if DEBUG_MISMATCHES {
        return true;
    }
    match std::env::var("GRAMMARS_DEBUG_MISMATCHES").or_else(|_| std::env::var("DEBUG_MISMATCHES")) {
        Ok(v) => {
            let v = v.to_ascii_lowercase();
            v == "1" || v == "true" || v == "yes" || v == "on"
        }
        Err(_) => false,
    }
}

/// Normalizes a path for comparison purposes.
/// - Removes NoOp edges.
/// - Collects all CheckLLM bitvectors, intersects them, and prepends a single CheckLLM.
pub(crate) fn normalize_path(path: Vec<IntermediateTrie3EdgeKey>) -> Vec<IntermediateTrie3EdgeKey> {
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

/// Compute the set of nodes that are part of any directed cycle in the subgraph induced by `nodes`.
/// Uses Kahn's algorithm (iterative topological pruning) to identify nodes not removed => in cycles.
pub(crate) fn nodes_in_cycles_subgraph(
    god: &IntermediateTrie3GodWrapper,
    nodes: &[IntermediatePrecomputeNode3Index],
) -> BTreeSet<usize> {
    // Build a set for quick membership checks.
    let node_set: BTreeSet<usize> = nodes.iter().map(|n| n.as_usize()).collect();

    // Build adjacency and in-degree within the induced subgraph.
    let mut indeg: BTreeMap<usize, usize> = BTreeMap::new();
    let mut adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for u in &node_set {
        indeg.entry(*u).or_insert(0);
        adj.entry(*u).or_insert_with(Vec::new);
    }

    for src_idx in nodes {
        let u = src_idx.as_usize();
        if let Some(read_guard) = src_idx.read(god) {
            for (_ek, dsts) in read_guard.children().iter() {
                for (dst_idx, _ev) in dsts.iter() {
                    let v = dst_idx.as_usize();
                    if node_set.contains(&v) {
                        // u -> v is an edge in the induced subgraph
                        adj.get_mut(&u).unwrap().push(v);
                        *indeg.entry(v).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    // Kahn's algorithm: remove all nodes with indegree 0 iteratively.
    let mut q: VecDeque<usize> = VecDeque::new();
    for (&u, &d) in indeg.iter() {
        if d == 0 {
            q.push_back(u);
        }
    }

    while let Some(u) = q.pop_front() {
        if let Some(nei) = adj.get_mut(&u) {
            for &v in nei.iter() {
                if let Some(d) = indeg.get_mut(&v) {
                    *d -= 1;
                    if *d == 0 {
                        q.push_back(v);
                    }
                }
            }
            nei.clear();
        }
        indeg.remove(&u);
    }

    // Nodes left in indeg are part of (at least one) cycle.
    indeg.keys().cloned().collect()
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
    let all_paths =
        IntermediatePrecomputeNode3::get_all_paths_with_cycles(god, &all_root_indices, |_idx, n| n.value.end, 1000000);

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
    let has_only_empty_path =
        simplified_paths.len() == 1 && simplified_paths.iter().next().unwrap().is_empty();
    let root_content = if has_only_empty_path {
        IntermediatePrecomputedNodeContents3::leaf()
    } else {
        IntermediatePrecomputedNodeContents3::internal()
    };
    let new_root = god.insert(Trie::new(root_content)).into();

    let mut node_cache: BTreeMap<Vec<IntermediateTrie3EdgeKey>, IntermediatePrecomputeNode3Index> =
        BTreeMap::new();
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
                let content = if is_leaf {
                    IntermediatePrecomputedNodeContents3::leaf()
                } else {
                    IntermediatePrecomputedNodeContents3::internal()
                };
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
    if !debug_mismatches_enabled() {
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
        let (min_trie_paths, _, _) = {
            let (god_copy, roots_vec_copy, _) = Trie::deep_copy_subtrees(
                &minimal_god,
                &minimal_roots_map.values().cloned().collect::<Vec<_>>(),
            );
            let mut roots_map_copy = BTreeMap::new();
            for (sid, r_idx) in minimal_roots_map.keys().zip(roots_vec_copy.iter()) {
                roots_map_copy.insert(*sid, *r_idx);
            }
            run_trie_based_elimination(&mut roots_map_copy, &god_copy);
            (
                get_normalized_paths(&roots_map_copy, &god_copy),
                god_copy,
                roots_map_copy,
            )
        };

        let (min_path_paths, _, _) = {
            let (god_copy, roots_vec_copy, _) = Trie::deep_copy_subtrees(
                &minimal_god,
                &minimal_roots_map.values().cloned().collect::<Vec<_>>(),
            );
            let mut roots_map_copy = BTreeMap::new();
            for (sid, r_idx) in minimal_roots_map.keys().zip(roots_vec_copy.iter()) {
                roots_map_copy.insert(*sid, *r_idx);
            }
            eliminate_pushes_and_pops_path_based(&mut roots_map_copy, &god_copy);
            (
                get_normalized_paths(&roots_map_copy, &god_copy),
                god_copy,
                roots_map_copy,
            )
        };

        println!(
            "\n--- MINIMAL FAILING INPUT (graph) ---\n{}",
            Trie::pretty_print(&minimal_god, &minimal_roots_map.values().cloned().collect::<Vec<_>>())
        );
        let minimal_input_paths = get_normalized_paths(&minimal_roots_map, &minimal_god);
        println!(
            "\n--- MINIMAL FAILING INPUT ({} paths) ---",
            minimal_input_paths.len()
        );
        for (i, path) in minimal_input_paths.iter().enumerate() {
            println!("  Path {}: {:?}", i, path);
        }

        println!(
            "\n--- TRIE-BASED OUTPUT ({} paths) ---",
            min_trie_paths.len()
        );
        for (i, path) in min_trie_paths.iter().enumerate() {
            println!("  Path {}: {:?}", i, path);
        }
        println!(
            "\n--- PATH-BASED OUTPUT ({} paths) ---",
            min_path_paths.len()
        );
        for (i, path) in min_path_paths.iter().enumerate() {
            println!("  Path {}: {:?}", i, path);
        }

        panic!("Push/Pop elimination mismatch detected. See logs for details.");
    }

    // 5. If no mismatch, run the trie-based version on the actual input `god` to modify it.
    run_trie_based_elimination(roots, god);
}

/// Compute the final set of normalized paths from a graph, for comparison.
fn get_normalized_paths(
    roots: &BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) -> BTreeSet<Vec<IntermediateTrie3EdgeKey>> {
    let root_indices: Vec<_> = roots.values().cloned().collect();
    get_normalized_paths_for_vec(&root_indices, god)
}

pub(crate) fn get_normalized_paths_for_vec(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) -> BTreeSet<Vec<IntermediateTrie3EdgeKey>> {
    IntermediatePrecomputeNode3::get_all_paths_with_cycles(god, &roots, |_idx, n| n.value.end, 1000000)
        .into_iter()
        .map(|(_r, p)| normalize_path(p.into_iter().map(|(ek, _, _)| ek).collect()))
        .collect()
}

/// Runs both elimination algorithms on a graph and returns true if their outputs differ.
fn check_mismatch(
    god: &IntermediateTrie3GodWrapper,
    roots: &BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
) -> bool {
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    if roots_vec.is_empty() {
        return false;
    }

    let (trie_paths, _, _) = {
        let (god_copy, roots_vec_copy, _) = Trie::deep_copy_subtrees(god, &roots_vec);
        let mut roots_map_copy = BTreeMap::new();
        for (sid, r_idx) in roots.keys().zip(roots_vec_copy.iter()) {
            roots_map_copy.insert(*sid, *r_idx);
        }
        run_trie_based_elimination(&mut roots_map_copy, &god_copy);
        (
            get_normalized_paths(&roots_map_copy, &god_copy),
            god_copy,
            roots_map_copy,
        )
    };

    let (path_paths, _, _) = {
        let (god_copy, roots_vec_copy, _) = Trie::deep_copy_subtrees(god, &roots_vec);
        let mut roots_map_copy = BTreeMap::new();
        for (sid, r_idx) in roots.keys().zip(roots_vec_copy.iter()) {
            roots_map_copy.insert(*sid, *r_idx);
        }
        eliminate_pushes_and_pops_path_based(&mut roots_map_copy, &god_copy);
        (
            get_normalized_paths(&roots_map_copy, &god_copy),
            god_copy,
            roots_map_copy,
        )
    };

    trie_paths != path_paths
}

/// Given a failing graph, iteratively removes edges and roots to find a smaller subgraph that still fails.
fn refine_mismatch(
    initial_god: &IntermediateTrie3GodWrapper,
    initial_roots: &BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
) -> (
    IntermediateTrie3GodWrapper,
    BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
) {
    let (mut minimal_god, minimal_roots_vec, old_to_new) = Trie::deep_copy_subtrees(
        initial_god,
        &initial_roots.values().cloned().collect::<Vec<_>>(),
    );
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
            let (candidate_god, candidate_roots_vec, old_to_new) =
                Trie::deep_copy_subtrees(&minimal_god, &current_roots_vec);
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
                let new_size = Trie::all_nodes(
                    &minimal_god,
                    &minimal_roots.values().cloned().collect::<Vec<_>>(),
                )
                .len();
                println!(
                    "... Refined mismatch by removing edge group. New size: {} nodes.",
                    new_size
                );
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

                if candidate_roots.is_empty() {
                    continue;
                }

                if check_mismatch(&minimal_god, &candidate_roots) {
                    minimal_roots = candidate_roots;
                    changed_in_pass = true;
                    println!(
                        "... Refined mismatch by removing a root. New root count: {}.",
                        minimal_roots.len()
                    );
                    break; // Restart the while loop
                }
            }
        }
    }

    (minimal_god, minimal_roots)
}

#[derive(Clone, Debug)]
enum Exit {
    // Both cancel -> epsilon; we attach a NoOp from the aggregator node to dst.
    Cancel {
        llm: LLMTokenBV,
        dst: IntermediatePrecomputeNode3Index,
    },
    // Remove Push unconditionally and decrement Pop(n>1) -> Pop(n-1).
    DegradePop {
        llm: LLMTokenBV,
        new_n: usize,
        pop_bv: StateIDBV,
        dst: IntermediatePrecomputeNode3Index,
    },
    // Elimination is blocked by a nested Push, or by reaching a leaf with no Pop(n>=1).
    // We must keep the Push with the possibly-restricted bitset.
    BlockedPush {
        llm: LLMTokenBV,
        push_bv: StateIDBV,
        dst: IntermediatePrecomputeNode3Index,
        on_cycle: bool,
    },
}

#[derive(Clone, Debug)]
struct BFSState {
    node: IntermediatePrecomputeNode3Index,
    push_bv: StateIDBV,
    llm_bv: LLMTokenBV,
}

fn get_or_create_aggregator_node(
    src: IntermediatePrecomputeNode3Index,
    llm: &LLMTokenBV,
    god: &IntermediateTrie3GodWrapper,
    cache: &mut BTreeMap<LLMTokenBV, IntermediatePrecomputeNode3Index>,
) -> IntermediatePrecomputeNode3Index {
    // If no checks to aggregate, the aggregator node is just the source itself.
    if *llm == LLMTokenBV::max_ones() {
        return src;
    }
    if let Some(idx) = cache.get(llm) {
        return *idx;
    }
    let new_node = god
        .insert(Trie::new(IntermediatePrecomputedNodeContents3::internal()))
        .into();
    god.insert_edge_simple(
        src,
        new_node,
        IntermediateTrie3EdgeKey::CheckLLM(llm.clone()),
        (),
    );
    cache.insert(llm.clone(), new_node);
    new_node
}

fn compute_push_elim_exits(
    start: IntermediatePrecomputeNode3Index,
    initial_push_bv: &StateIDBV,
    god: &IntermediateTrie3GodWrapper,
    nodes_on_cycle: &BTreeSet<usize>,
) -> Vec<Exit> {
    crate::debug!(5,
        "[challenge_elim]   - compute_push_elim_exits(start: {}, initial_push_bv: {:?})",
        start, initial_push_bv
    );
    let mut exits: Vec<Exit> = Vec::new();
    let mut q: VecDeque<BFSState> = VecDeque::new();
    q.push_back(BFSState {
        node: start,
        push_bv: initial_push_bv.clone(),
        llm_bv: LLMTokenBV::max_ones(),
    });

    // We include node, push_bv, llm_bv in visited to avoid infinite exploration across cycles.
    // This is finite because push_bv and llm_bv only ever intersect with finitely many constants.
    let mut visited: BTreeSet<(usize, StateIDBV, LLMTokenBV)> = BTreeSet::new();
    // Safety guard (should not normally trigger thanks to visited)
    let mut steps: usize = 0;
    let max_steps: usize = 1_000_000;

    while let Some(state) = q.pop_front() {
        steps += 1;
        if steps % 10_000 == 0 {
            crate::debug!(5,
                "[challenge_elim]   - BFS progress: {} states explored (q_len: {}, visited: {}, exits: {}), guard {}",
                steps, q.len(), visited.len(), exits.len(), max_steps
            );
        }
        if steps > max_steps {
            crate::debug!(5, "[challenge_elim] Warning: BFS step guard hit while eliminating a Push; breaking exploration to avoid non-termination.");
            break;
        }

        let key = (state.node.as_usize(), state.push_bv.clone(), state.llm_bv.clone());
        if !visited.insert(key) {
            if steps < 100 { // Log first few skips
                 crate::debug!(5, "[challenge_elim]     - BFS skip visited state: node {}, push_bv {:?}, llm_bv {:?}", state.node, state.push_bv, state.llm_bv);
            }
            continue;
        }
        if steps < 100 { // Log first few processed
            crate::debug!(5, "[challenge_elim]     - BFS processing state: node {}, push_bv {:?}, llm_bv {:?}", state.node, state.push_bv, state.llm_bv);
        }

        // If this node is an end, then along this branch we can only preserve the Push (blocked).
        if let Some(read_guard) = state.node.read(god) {
            if read_guard.value.end {
                if steps < 100 {
                    crate::debug!(5, "[challenge_elim]       - Found end node, creating BlockedPush exit.");
                }
                exits.push(Exit::BlockedPush {
                    llm: state.llm_bv.clone(),
                    push_bv: state.push_bv.clone(),
                    dst: state.node,
                    on_cycle: nodes_on_cycle.contains(&state.node.as_usize()),
                });
                // Do not explore past an end marker; treat as terminal for this branch
                continue;
            }
            // Explore outgoing edges
            for (ek, dsts) in read_guard.children().iter() {
                if steps < 100 {
                    crate::debug!(5, "[challenge_elim]       - Exploring edge {:?} to {} dests", ek, dsts.len());
                }
                match ek {
                    IntermediateTrie3EdgeKey::NoOp => {
                        for (dst_idx, _ev) in dsts.iter() {
                            if steps < 100 {
                                crate::debug!(5, "[challenge_elim]         - Enqueueing NoOp -> {}", dst_idx);
                            }
                            q.push_back(BFSState {
                                node: *dst_idx,
                                push_bv: state.push_bv.clone(),
                                llm_bv: state.llm_bv.clone(),
                            });
                        }
                    }
                    IntermediateTrie3EdgeKey::CheckLLM(llm2) => {
                        let mut next_llm = state.llm_bv.clone();
                        // Aggregate checks by intersection; do not prune on emptiness:
                        // normalize_path keeps empty intersections too.
                        next_llm &= llm2.clone();
                        for (dst_idx, _ev) in dsts.iter() {
                            if steps < 100 {
                                crate::debug!(5, "[challenge_elim]         - Enqueueing CheckLLM -> {} with new llm_bv", dst_idx);
                            }
                            q.push_back(BFSState {
                                node: *dst_idx,
                                push_bv: state.push_bv.clone(),
                                llm_bv: next_llm.clone(),
                            });
                        }
                    }
                    IntermediateTrie3EdgeKey::Push(_nested) => {
                        // Blocked by a nested push: keep our (possibly intersected) push
                        // anchored at the current node (after any aggregated CheckLLM),
                        // and do not traverse past the nested push for this elimination.
                        //
                        // Important: We DO NOT move the push forward to the nested push's
                        // destination. That would violate the stack semantics used by the
                        // path-based simplifier (which blocks when encountering another push).
                        if steps < 100 {
                            crate::debug!(5, "[challenge_elim]       - Blocked by nested push. Creating BlockedPush exit.");
                        }
                        exits.push(Exit::BlockedPush {
                            llm: state.llm_bv.clone(),
                            push_bv: state.push_bv.clone(),
                            dst: state.node,
                            on_cycle: nodes_on_cycle.contains(&state.node.as_usize()),
                        });
                        // Do not traverse past a nested push for this elimination.
                    }
                    IntermediateTrie3EdgeKey::Pop(n, pop_bv) => {
                        let n_val = *n;
                        for (dst_idx, _ev) in dsts.iter() {
                            match n_val {
                                0 => {
                                    // Fold into push: A := A ∩ B; prune branch if disjoint.
                                    if state.push_bv.is_disjoint(pop_bv) {
                                        if steps < 100 {
                                            crate::debug!(5, "[challenge_elim]         - Pruning Pop(0) branch due to disjoint BVs.");
                                        }
                                        // Invalid on this branch
                                        continue;
                                    }
                                    let mut next_push = state.push_bv.clone();
                                    next_push &= pop_bv.clone();
                                    if steps < 100 {
                                        crate::debug!(5, "[challenge_elim]         - Enqueueing Pop(0) -> {} with restricted push_bv", dst_idx);
                                    }
                                    q.push_back(BFSState {
                                        node: *dst_idx,
                                        push_bv: next_push,
                                        llm_bv: state.llm_bv.clone(),
                                    });
                                }
                                1 => {
                                    // Cancel if intersect; else branch invalid
                                    if state.push_bv.is_disjoint(pop_bv) {
                                        if steps < 100 {
                                            crate::debug!(5, "[challenge_elim]         - Pruning Pop(1) branch due to disjoint BVs.");
                                        }
                                        continue;
                                    }
                                    if steps < 100 {
                                        crate::debug!(5, "[challenge_elim]       - Found Pop(1). Creating Cancel exit to {}.", dst_idx);
                                    }
                                    exits.push(Exit::Cancel {
                                        llm: state.llm_bv.clone(),
                                        dst: *dst_idx,
                                    });
                                    // Do not explore past a Pop(1) for this elimination.
                                }
                                _ => {
                                    // Remove Push and decrement Pop.
                                    if steps < 100 {
                                        crate::debug!(5, "[challenge_elim]       - Found Pop(>1). Creating DegradePop exit to {}.", dst_idx);
                                    }
                                    exits.push(Exit::DegradePop {
                                        llm: state.llm_bv.clone(),
                                        new_n: n_val - 1,
                                        pop_bv: pop_bv.clone(),
                                        dst: *dst_idx,
                                    });
                                    // Do not explore past a Pop(n>1) for this elimination.
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    exits
}

fn remove_specific_edge(
    god: &IntermediateTrie3GodWrapper,
    src: IntermediatePrecomputeNode3Index,
    key: IntermediateTrie3EdgeKey,
    dst: IntermediatePrecomputeNode3Index,
) -> bool {
    if let Some(mut guard) = src.write(god) {
        let children = guard.children_mut();
        match children.entry(key) {
            Entry::Occupied(mut occ) => {
                let map = occ.get_mut();
                let removed = map.remove(&dst).is_some();
                if map.is_empty() {
                    occ.remove();
                }
                return removed;
            }
            Entry::Vacant(_) => {}
        }
    }
    false
}

fn run_trie_based_elimination(
    roots: &mut BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) {
    // Fixpoint elimination. We never exit early; we iterate until a whole round removes none.
    let root_indices: Vec<_> = roots.values().cloned().collect();
    if root_indices.is_empty() {
        return;
    }

    let mut round: usize = 0;
    loop {
        round += 1;
        // Collect all reachable nodes for this round.
        let nodes = Trie::all_nodes(god, &root_indices);
        let nodes_on_cycle = nodes_in_cycles_subgraph(god, &nodes);
        // Collect all push edges in the current snapshot.
        let mut push_edges: Vec<(IntermediatePrecomputeNode3Index, StateIDBV, IntermediatePrecomputeNode3Index)> =
            Vec::new();
        for src in &nodes {
            if let Some(read_guard) = src.read(god) {
                for (ek, dsts) in read_guard.children().iter() {
                    if let IntermediateTrie3EdgeKey::Push(bv) = ek {
                        for (dst_idx, _ev) in dsts.iter() {
                            push_edges.push((*src, bv.clone(), *dst_idx));
                        }
                    }
                }
            }
        }

        if push_edges.is_empty() {
            crate::debug!(5, "[challenge_elim] No pushes found; done.");
            break;
        }

        eprintln!(
            "[challenge_elim] Round {}: attempting to eliminate {} push edge(s).",
            round,
            push_edges.len()
        );
        // Simple progress bar: print at 10% increments
        let total = push_edges.len().max(1);
        let mut processed = 0usize;
        let mut next_mark = 10usize;

        // For each src, memoize aggregator nodes by LLM BV to avoid node blowup.
        let mut per_src_agg_cache: BTreeMap<
            IntermediatePrecomputeNode3Index,
            BTreeMap<LLMTokenBV, IntermediatePrecomputeNode3Index>,
        > = BTreeMap::new();

        // Cache exits per (dst, push_bv) to avoid repeated BFS work in this round.
        let mut exit_cache: BTreeMap<(usize, StateIDBV), Vec<Exit>> = BTreeMap::new();

        let mut removed_this_round: usize = 0;

        for (src, push_bv, dst) in push_edges {
            processed += 1;
            let pct = (processed * 100) / total;
            if pct >= next_mark {
                eprintln!(
                    "[challenge_elim] Round {} progress: {}/{} ({}%)",
                    round, processed, total, pct
                );
                next_mark += 10;
            }
            crate::debug!(5, "[challenge_elim]  - Processing push edge {} --Push({:?})--> {}", src, push_bv, dst);

            // Compute or reuse exits for this (dst, push_bv)
            let exits = match exit_cache.get(&(dst.as_usize(), push_bv.clone())) {
                Some(v) => {
                    crate::debug!(5, "[challenge_elim]    - Reusing {} exits from cache for (dst {}, push_bv {:?})", v.len(), dst, push_bv);
                    v.clone()
                },
                None => {
                    crate::debug!(5, "[challenge_elim]    - Computing exits for (dst {}, push_bv {:?})", dst, push_bv);
                    // BFS exploration to compute exits for (dst, push_bv)
                    // Results are memoized per round to avoid repetition.
                    let e = compute_push_elim_exits(dst, &push_bv, god, &nodes_on_cycle);
                    crate::debug!(5, "[challenge_elim]    - Found {} exits.", e.len());
                    exit_cache.insert((dst.as_usize(), push_bv.clone()), e.clone());
                    e
                }
            };
            // Deduplicate exits and detect any blocked branches.
            if exits.is_empty() {
                crate::debug!(5, "[challenge_elim]    - No exits found. Removing original push edge.");
                // No viable continuations were found for this push under the stack semantics
                // (e.g., every branch mismatched). This path is dead; remove the original push.
                if remove_specific_edge(
                    god,
                    src,
                    IntermediateTrie3EdgeKey::Push(push_bv.clone()),
                    dst,
                ) {
                    removed_this_round += 1;
                }
                continue;
            }

            // Check for stability. A push is stable if all its exits are BlockedPush and they
            // represent no-op transformations (i.e., same destination, same push bitvector, no LLM checks).
            let mut is_stable = true;
            for ex in &exits {
                if let Exit::BlockedPush { llm, push_bv: exit_push_bv, dst: exit_dst, .. } = ex {
                    if *llm != LLMTokenBV::max_ones() || *exit_dst != dst || *exit_push_bv != push_bv {
                        is_stable = false;
                        break;
                    }
                } else {
                    is_stable = false;
                    break;
                }
            }

            if is_stable {
                crate::debug!(5, "[challenge_elim]    - Push edge is stable, no changes made.");
                continue;
            }

            // If not stable and not complex-blocked, rewire all exits and remove the original edge.
            // New policy:
            // - Always rewire Cancel/Degrade.
            // - Partition BlockedPush into cyclic vs acyclic.
            //   If any cyclic: do NOT rewire any BlockedPush; keep original Push but restrict its BV
            //   to the union of all blocked push_bv (fold Pop(0) constraints), to avoid cycle blow-up.
            let cache = per_src_agg_cache
                .entry(src)
                .or_insert_with(BTreeMap::new);

            // Deduplicate exits before wiring.
            let mut cancel_set = BTreeSet::new();
            let mut degrade_set = BTreeSet::new();
            let mut blocked_acyclic_set = BTreeSet::new();
            let mut has_cyclic_blocked = false;
            let mut union_blocked_push_bv = StateIDBV::zeros();
            for ex in exits.iter() {
                match ex {
                    Exit::Cancel { llm, dst } => {
                        cancel_set.insert((llm.clone(), *dst));
                    }
                    Exit::DegradePop { llm, new_n, pop_bv, dst } => {
                        degrade_set.insert((llm.clone(), *new_n, pop_bv.clone(), *dst));
                    }
                    Exit::BlockedPush { llm, push_bv: exit_push_bv, dst, on_cycle } => {
                        // Track union of push_bv across ALL blocked exits to fold Pop(0) into the push label.
                        union_blocked_push_bv |= exit_push_bv.clone();
                        if *on_cycle {
                            has_cyclic_blocked = true;
                        } else {
                            blocked_acyclic_set.insert((llm.clone(), exit_push_bv.clone(), *dst));
                        }
                    }
                }
            }

            for (llm, cancel_dst) in cancel_set {
                crate::debug!(5, "[challenge_elim]    - Applying Cancel exit to {} via LLM {:?}", cancel_dst, llm);
                let agg = get_or_create_aggregator_node(src, &llm, god, cache);
                god.insert_edge_simple(agg, cancel_dst, IntermediateTrie3EdgeKey::NoOp, ());
            }
            for (llm, new_n, pop_bv, degrade_dst) in degrade_set {
                crate::debug!(5, "[challenge_elim]    - Applying DegradePop exit to {} via LLM {:?}", degrade_dst, llm);
                let agg = get_or_create_aggregator_node(src, &llm, god, cache);
                god.insert_edge_simple(
                    agg,
                    degrade_dst,
                    IntermediateTrie3EdgeKey::Pop(new_n, pop_bv),
                    (),
                );
            }

            if has_cyclic_blocked {
                // Avoid creating new Push edges that feed back into cycles.
                // Keep the original Push edge but fold Pop(0) constraints into its label
                // using the union of all blocked push_bv, to remain semantically accurate.
                if union_blocked_push_bv != push_bv {
                    crate::debug!(5,
                        "[challenge_elim]    - Cyclic blocked detected. Restricting original push label from {:?} to {:?} (keeping edge).",
                        push_bv, union_blocked_push_bv
                    );
                    // Replace the edge key: remove old (do NOT count as removal), then reinsert with new label.
                    remove_specific_edge(
                        god,
                        src,
                        IntermediateTrie3EdgeKey::Push(push_bv.clone()),
                        dst,
                    );
                    god.insert_edge_simple(
                        src,
                        dst,
                        IntermediateTrie3EdgeKey::Push(union_blocked_push_bv.clone()),
                        (),
                    );
                } else {
                    crate::debug!(5,
                        "[challenge_elim]    - Cyclic blocked detected. Keeping original push edge unchanged (label already minimal)."
                    );
                }
                // Do NOT rewire any BlockedPush exits in cyclic case.
                continue;
            } else {
                // No cyclic blocked: safe to rewire all BlockedPush exits.
                for (llm, exit_push_bv, exit_dst) in blocked_acyclic_set {
                    crate::debug!(5,
                        "[challenge_elim]      - Rewiring BlockedPush (acyclic) to {}, push_bv: {:?}, llm: {:?}",
                        exit_dst, exit_push_bv, llm
                    );
                    let agg = get_or_create_aggregator_node(src, &llm, god, cache);
                    god.insert_edge_simple(
                        agg,
                        exit_dst,
                        IntermediateTrie3EdgeKey::Push(exit_push_bv.clone()),
                        (),
                    );
                }
                // Remove the original push edge; rewriting is complete.
                crate::debug!(5,
                    "[challenge_elim]    - Decision: Rewiring complete. Removing original edge {} --Push({:?})--> {}",
                    src, push_bv.clone(), dst
                );
                if remove_specific_edge(
                    god,
                    src,
                    IntermediateTrie3EdgeKey::Push(push_bv.clone()),
                    dst,
                ) {
                    removed_this_round += 1;
                }
            }
        }

        eprintln!(
            "[challenge_elim] Round {} removed {} push edge(s).",
            round, removed_this_round
        );

        Trie::gc(god, &root_indices);

        if removed_this_round == 0 {
            // Fixpoint reached: no more eliminations possible.
            break;
        }
    }

    // Optional: recompute depths for diagnostics or downstream heuristics.
    Trie::recompute_all_max_depths(god, &root_indices);
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
        let (eliminated_god, eliminated_roots, _) =
            Trie::deep_copy_subtrees(input_god, input_roots);
        let mut eliminated_roots_map = BTreeMap::new();
        for (i, r) in eliminated_roots.iter().enumerate() {
            eliminated_roots_map.insert(TokenizerStateID(i), *r); // Use dummy TokenizerStateID
        }
        eliminate_pushes_and_pops(&mut eliminated_roots_map, &eliminated_god);

        // 2. Flatten result to paths
        let final_roots_from_trie_elim: Vec<_> = eliminated_roots_map.values().cloned().collect();
        let paths_from_trie_elim: BTreeSet<_> = IntermediatePrecomputeNode3::get_all_paths_with_cycles(
            &eliminated_god,
            &final_roots_from_trie_elim,
            |_idx, n| n.value.end,
            1000000,
        )
        .into_iter()
        .map(|(_r, p)| normalize_path(p.into_iter().map(|(ek, _, _)| ek).collect()))
        .collect();

        // 3. Run old path-based elimination directly
        let initial_paths =
            IntermediatePrecomputeNode3::get_all_paths_with_cycles(input_god, input_roots, |_idx, node| node.value.end, 1000000);
        let mut paths_from_path_elim = BTreeSet::new();
        for (_root_value, path_edges) in initial_paths {
            let edge_keys: Vec<_> = path_edges.into_iter().map(|(ek, _, _)| ek).collect();
            if let Some(new_path) = simplify_path(edge_keys) {
                paths_from_path_elim.insert(normalize_path(new_path));
            }
        }

        // 4. Compare
        if paths_from_trie_elim != paths_from_path_elim {
            eprintln!("\n--- MISMATCH DETECTED IN TEST ---");
            eprintln!("EXPECTED (path-based):");
            for path in &paths_from_path_elim {
                eprintln!("  {:?}", path);
            }
            eprintln!("\nACTUAL (trie-based):");
            for path in &paths_from_trie_elim {
                eprintln!("  {:?}", path);
            }
            eprintln!("---------------------------------\n");
        }
        assert_eq!(paths_from_trie_elim, paths_from_path_elim);
    }

    #[test]
    fn test_simple_cancel() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv = StateIDBV::zeros();
        bv.insert(1);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv.clone()), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_mismatch_invalidates_path() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv2), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_pop_zero_keeps_push() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv = StateIDBV::zeros();
        bv.insert(1);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv.clone()), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(0, bv), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_pop_zero_mismatch_invalidates_path() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(0, bv2), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_pop_n_decrements() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2); // Note: disjoint, but should not matter for n>1

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(3, bv2), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_blocked_push() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv2.clone()), (), v2);
        v2.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv2), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_multiple_cancellations_in_sequence() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v3 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1.clone()), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv2.clone()), (), v2);
        v2.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv2), (), v3);
        v3.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv1), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_interleaved_ops() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut llm_bv = LLMTokenBV::zeros();
        llm_bv.insert(100);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv1.clone()), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_bv), (), v2);
        v2.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv1), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_branching_and_merging() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v3 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv_a = StateIDBV::zeros();
        bv_a.insert(1);
        let mut llm_x = LLMTokenBV::zeros();
        llm_x.insert(100);
        let mut llm_y = LLMTokenBV::zeros();
        llm_y.insert(200);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_a.clone()), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_x), (), v2);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_y), (), v3);
        v2.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), end);
        v3.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_cycle_simplification() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv_a = StateIDBV::zeros();
        bv_a.insert(1);

        // Path to end
        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), end);
        // Path with cycle
        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_a.clone()), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), root);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_no_pushes_or_pops() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut llm_bv = LLMTokenBV::zeros();
        llm_bv.insert(100);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_bv), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), end);

        run_test(&god, &[root]);
    }

    #[test]
    fn test_dangling_pop() {
        let god = IntermediateTrie3GodWrapper::new();
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv = StateIDBV::zeros();
        bv.insert(1);

        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv), (), end);

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
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v2 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v3x =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v4x =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v5x =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let v3y =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let merge =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

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
        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_a.clone()), (), v1);
        v1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_b.clone()), (), v2);

        // Branching
        v2.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_x), (), v3x);
        v2.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_y), (), v3y);

        // Path X (iterative cancellation)
        v3x.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_c.clone()), (), v4x);
        v4x.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_c.clone()), (), v5x);
        v5x.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_b.clone()), (), merge);

        // Path Y (blocking and invalidation)
        v3y.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), merge);

        // Common suffix
        merge
            .write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), end);

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
            current_node
                .write(&god)
                .unwrap()
                .force_insert_to_node(edge, (), next_node);
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
        let root =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let c1 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let c2 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let c3 =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal())));
        let end =
            Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::leaf())));

        let mut bv_a = StateIDBV::zeros();
        bv_a.insert(1);
        let mut bv_b = StateIDBV::zeros();
        bv_b.insert(2);
        let mut llm_x = LLMTokenBV::zeros();
        llm_x.insert(100);

        // Path structure
        root.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_a.clone()), (), c1);
        c1.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_x), (), c2);
        // Inner cycle
        c2.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv_b.clone()), (), c3);
        c3.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_b.clone()), (), c2);
        // Exit path
        c2.write(&god)
            .unwrap()
            .force_insert_to_node(IntermediateTrie3EdgeKey::Pop(1, bv_a.clone()), (), end);

        run_test(&god, &[root]);
    }

    /// Helper to build a simple graph with a single path for testing.
    fn build_graph_from_path(
        god: &IntermediateTrie3GodWrapper,
        path: Vec<IntermediateTrie3EdgeKey>,
    ) -> IntermediatePrecomputeNode3Index {
        let root: IntermediatePrecomputeNode3Index = god
            .insert(Trie::new(IntermediatePrecomputedNodeContents3::internal()))
            .into();
        if path.is_empty() {
            root.write(god).unwrap().value.end = true;
            return root;
        }

        let mut current_node = root;
        for (i, edge) in path.iter().enumerate() {
            let is_last = i == path.len() - 1;
            let content = if is_last {
                IntermediatePrecomputedNodeContents3::leaf()
            } else {
                IntermediatePrecomputedNodeContents3::internal()
            };
            let next_node = god.insert(Trie::new(content)).into();
            current_node
                .write(god)
                .unwrap()
                .force_insert_to_node(edge.clone(), (), next_node);
            current_node = next_node;
        }
        root
    }

    #[test]
    fn test_minimal_push_push_pop_cancel() {
        // Path: Push(1) -> Push(2) -> Pop(1, 2). Should simplify to Push(1).
        let god = IntermediateTrie3GodWrapper::new();
        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);
        let root = build_graph_from_path(
            &god,
            vec![
                IntermediateTrie3EdgeKey::Push(bv1),
                IntermediateTrie3EdgeKey::Push(bv2.clone()),
                IntermediateTrie3EdgeKey::Pop(1, bv2),
            ],
        );
        run_test(&god, &[root]);
    }

    #[test]
    fn test_minimal_push_pop_mismatch_invalidates() {
        // Path: Push(1) -> Pop(1, 2). Should invalidate the path (empty result).
        let god = IntermediateTrie3GodWrapper::new();
        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);
        let root = build_graph_from_path(
            &god,
            vec![
                IntermediateTrie3EdgeKey::Push(bv1),
                IntermediateTrie3EdgeKey::Pop(1, bv2),
            ],
        );
        run_test(&god, &[root]);
    }

    #[test]
    fn test_minimal_push_pop_zero_keeps_push() {
        // Path: Push(1) -> Pop(0, 1). Should simplify to Push(1).
        let god = IntermediateTrie3GodWrapper::new();
        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let root = build_graph_from_path(
            &god,
            vec![
                IntermediateTrie3EdgeKey::Push(bv1.clone()),
                IntermediateTrie3EdgeKey::Pop(0, bv1),
            ],
        );
        run_test(&god, &[root]);
    }

    #[test]
    fn test_minimal_push_pop_zero_mismatch_invalidates() {
        // Path: Push(1) -> Pop(0, 2). Should invalidate the path (empty result).
        let god = IntermediateTrie3GodWrapper::new();
        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);
        let root = build_graph_from_path(
            &god,
            vec![
                IntermediateTrie3EdgeKey::Push(bv1),
                IntermediateTrie3EdgeKey::Pop(0, bv2),
            ],
        );
        run_test(&god, &[root]);
    }

    #[test]
    fn test_minimal_challenging_gauntlet_failure() {
        // A minimal failing case derived from the challenging gauntlet.
        // Path: CheckLLM(200) -> Push(1) -> Push(2) -> Pop(1, 1) -> Pop(1, 1).
        // Should be invalidated because Push(2) is blocked by Pop(1, 1) which mismatches.
        let god = IntermediateTrie3GodWrapper::new();
        let mut bv1 = StateIDBV::zeros();
        bv1.insert(1);
        let mut bv2 = StateIDBV::zeros();
        bv2.insert(2);
        let mut llm200 = LLMTokenBV::zeros();
        llm200.insert(200);
        let root = build_graph_from_path(
            &god,
            vec![
                IntermediateTrie3EdgeKey::CheckLLM(llm200),
                IntermediateTrie3EdgeKey::Push(bv1.clone()),
                IntermediateTrie3EdgeKey::Push(bv2),
                IntermediateTrie3EdgeKey::Pop(1, bv1.clone()),
                IntermediateTrie3EdgeKey::Pop(1, bv1),
            ],
        );
        run_test(&god, &[root]);
    }

    #[test]
    fn test_mismatch_from_user_report() {
        // This test is based on a mismatch found in production logs.
        // Path-based simplification reduces this path, while the trie-based
        // one (at the time of writing) does not.
        // Path: CheckLLM -> Pop(0) -> Push(A) -> Pop(0, A) -> Push(B)
        // Path-based simplifies Push(A) -> Pop(0, A) into just Push(A),
        // resulting in: CheckLLM -> Pop(0) -> Push(A) -> Push(B)
        let god = IntermediateTrie3GodWrapper::new();

        let mut llm_bv = LLMTokenBV::zeros();
        llm_bv.insert(0);
        llm_bv.insert(1);

        let mut bv0 = StateIDBV::zeros();
        bv0.insert(0);

        let mut bv3 = StateIDBV::zeros();
        bv3.insert(3);

        let mut bv4 = StateIDBV::zeros();
        bv4.insert(4);

        let path = vec![
            IntermediateTrie3EdgeKey::CheckLLM(llm_bv),
            IntermediateTrie3EdgeKey::Pop(0, bv0),
            IntermediateTrie3EdgeKey::Push(bv3.clone()),
            IntermediateTrie3EdgeKey::Pop(0, bv3),
            IntermediateTrie3EdgeKey::Push(bv4),
        ];

        let root = build_graph_from_path(&god, path);
        run_test(&god, &[root]);
    }

    #[test]
    fn test_mismatch_from_log_2() {
        // This test is based on a mismatch found during development, using the exact
        // graph structure from the logs.
        // The path-based simplifier incorrectly reduces `Push(A), Pop(0, A)` to `Push(A)`.
        // The correct behavior, exhibited by the trie-based approach, is to leave this
        // sequence unmodified.
        let god = IntermediateTrie3GodWrapper::new();

        // --- Bitsets ---
        let mut llm_bv = LLMTokenBV::zeros();
        llm_bv.insert(0);
        llm_bv.insert(1);

        let mut bv0 = StateIDBV::zeros();
        bv0.insert(0);

        let mut bv3 = StateIDBV::zeros();
        bv3.insert(3);

        let mut bv4 = StateIDBV::zeros();
        bv4.insert(4);

        // --- Nodes ---
        // The log shows nodes 0-17.
        let nodes: Vec<_> = (0..18)
            .map(|i| {
                let content = if i == 15 {
                    IntermediatePrecomputedNodeContents3::leaf()
                } else {
                    IntermediatePrecomputedNodeContents3::internal()
                };
                Trie2Index::from(god.insert(Trie::new(content)))
            })
            .collect();

        let root = nodes[0];

        // --- Graph Structure from log ---
        // Main path
        nodes[0].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), nodes[1]);
        nodes[1].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), nodes[2]);
        nodes[2].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(0, bv0), (), nodes[3]);
        nodes[3].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), nodes[4]);
        nodes[4].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv3.clone()), (), nodes[5]);
        nodes[5].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), nodes[6]);
        nodes[6].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_bv.clone()), (), nodes[7]);
        nodes[7].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), nodes[8]);
        nodes[8].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), nodes[9]);
        nodes[9].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Pop(0, bv3), (), nodes[10]);
        nodes[10].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), nodes[11]);
        nodes[11].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(bv4), (), nodes[12]);
        nodes[12].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), nodes[13]);
        nodes[13].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(llm_bv), (), nodes[14]);
        nodes[14].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), nodes[15]);

        // Other branches from root
        nodes[0].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), nodes[16]);
        nodes[0].write(&god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), nodes[17]);

        // The root in the log is not node 0, but an implicit root pointing to node 0.
        // The test harness takes a slice of roots. So I'll pass &[nodes[0]].
        run_test(&god, &[root]);
    }
}
