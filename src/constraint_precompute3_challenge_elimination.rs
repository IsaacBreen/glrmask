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
use rand::Rng;
use crate::datastructures::ordered_hash_map::Pop;

/// If true, runs both trie-based and path-based elimination, compares them,
/// and if a mismatch is found, attempts to find a minimal failing input graph.
/// This adds significant overhead and should only be used for debugging the
/// elimination logic itself.
const DEBUG_MISMATCHES: bool = true;

/// Maximum stack depth during trie-based elimination. Prevents infinite loops
/// on graphs with cycles that contain unbalanced Push operations.
const MAX_STACK_DEPTH: usize = 64;
const MAX_ELIMINATION_PASSES: usize = 16;

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

    // Build adjacency for the subgraph WITHOUT Pop(0)/Pop(1) edges.
    let n = nodes.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, node_idx) in nodes.iter().enumerate() {
        if let Some(node_r) = node_idx.read(source) {
            for (ek, dests) in node_r.children().iter() {
                let include = match ek {
                    IntermediateTrie3EdgeKey::Pop(k, _) if *k == 0 || *k == 1 => false,
                    _ => true,
                };
                if include {
                    for (dst_idx, _) in dests.iter() {
                        if let Some(&j) = id_of.get(dst_idx) {
                            adj[i].push(j);
                        }
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

/// Persistent, interned stack representation for pending StateIDBV pushes.
/// This replaces cloning large Vec<StateIDBV> everywhere.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct StackID(usize);

#[derive(Clone)]
struct StackNode {
    prev: Option<StackID>,
    top: StateIDBV,
    depth: usize,
}

struct StackInterner {
    nodes: Vec<StackNode>,
    // Intern map: (prev_stack_id, top_item) -> new_stack_id
    // Use BTreeMap to avoid requiring Hash on StateIDBV (it should implement Ord already,
    // since it participates in ordered keys via IntermediateTrie3EdgeKey).
    map: BTreeMap<(StackID, StateIDBV), StackID>,
}

impl StackInterner {
    fn new() -> Self {
        // Index 0 is the empty stack sentinel.
        let mut nodes = Vec::new();
        nodes.push(StackNode {
            prev: None,
            top: StateIDBV::zeros(), // unused for sentinel
            depth: 0,
        });
        Self {
            nodes,
            map: BTreeMap::new(),
        }
    }

    #[inline]
    fn empty(&self) -> StackID {
        StackID(0)
    }

    #[inline]
    fn is_empty(&self, id: StackID) -> bool {
        id.0 == 0
    }

    #[inline]
    fn depth(&self, id: StackID) -> usize {
        self.nodes[id.0].depth
    }

    #[inline]
    fn top(&self, id: StackID) -> Option<&StateIDBV> {
        if id.0 == 0 { None } else { Some(&self.nodes[id.0].top) }
    }

    #[inline]
    fn pop(&self, id: StackID) -> Option<StackID> {
        if id.0 == 0 { None } else { self.nodes[id.0].prev }
    }

    fn push(&mut self, id: StackID, bv: StateIDBV) -> StackID {
        if let Some(&existing) = self.map.get(&(id, bv.clone())) {
            return existing;
        }
        let new_id = StackID(self.nodes.len());
        let depth = self.nodes[id.0].depth + 1;
        self.nodes.push(StackNode { prev: Some(id), top: bv.clone(), depth });
        self.map.insert((id, bv), new_id);
        new_id
    }

    fn to_vec_bottom_up(&self, mut id: StackID) -> Vec<StateIDBV> {
        let mut rev: Vec<StateIDBV> = Vec::with_capacity(self.depth(id));
        while id.0 != 0 {
            rev.push(self.nodes[id.0].top.clone());
            id = self.nodes[id.0].prev.expect("non-empty must have prev");
        }
        rev.reverse();
        rev
    }

    fn from_bottom_slice(&mut self, slice: &[StateIDBV]) -> StackID {
        let mut id = self.empty();
        for bv in slice {
            id = self.push(id, bv.clone());
        }
        id
    }

    fn to_string(&self, mut id: StackID) -> String {
        if self.is_empty(id) {
            return "[]".to_string();
        }
        let mut items = Vec::new();
        while id.0 != 0 {
            items.push(format!("{:?}", self.nodes[id.0].top));
            id = self.nodes[id.0].prev.expect("non-empty must have prev");
        }
        items.reverse();
        format!("[{}]", items.join(", "))
    }
}

fn run_trie_based_elimination(
    roots: &mut BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) {
    // Outer fixpoint loop: iterate passes to converge even when we avoid deferring pushes in “push-cycling” SCCs.
    for pass in 0..MAX_ELIMINATION_PASSES {
        // Snapshot the reachable subgraph from the provided roots.
        let mut sids: Vec<TokenizerStateID> = Vec::with_capacity(roots.len());
        let mut old_root_vec: Vec<IntermediatePrecomputeNode3Index> = Vec::with_capacity(roots.len());
        for (sid, idx) in roots.iter() {
            sids.push(*sid);
            old_root_vec.push(*idx);
        }
        let before_sig = compute_graph_signature(god, roots);
        let (source, source_roots, _map) = Trie::deep_copy_subtrees(god, &old_root_vec);

        // Precompute push-cycling nodes to avoid unbounded pending stacks in SCCs that can keep pushing without encountering Pop(0/1).
        let push_cycle_nodes: BTreeSet<IntermediatePrecomputeNode3Index> =
            compute_push_cycle_nodes(&source, &source_roots);


        // Diagnostic and safety limits
        const HOT_NODE_THRESHOLD: u64 = 16384;
        const MAX_BFS_ITERATIONS: u64 = 75_000_000; // Failsafe to prevent hangs

        // Prepare destination arena (clear existing graph).
        god.clear();

        // Setup progress bar (spinner, no moving denominator).
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

        // Memoization: (source_idx, pending_stack) -> dest_idx
        // Memoization: (source_idx, pending_stack) -> dest_idx
        let mut stack_interner = StackInterner::new();
        let mut pair_cache: BTreeMap<(IntermediatePrecomputeNode3Index, StackID), IntermediatePrecomputeNode3Index> = BTreeMap::new();
        let mut hot_nodes = BTreeSet::new();
        let mut visit_counts: BTreeMap<IntermediatePrecomputeNode3Index, u64> = BTreeMap::new();
        let mut work: VecDeque<(IntermediatePrecomputeNode3Index, StackID)> = VecDeque::new();


        macro_rules! get_or_create {
            ($src_idx:expr, $stack:expr) => {
                {
                    let key = ($src_idx, $stack);
                    if let Some(&existing) = pair_cache.get(&key) {
                        existing
                    } else {
                        let src_guard = key.0.read(&source).expect("source read");
                        let is_end = src_guard.value.end && stack_interner.is_empty(key.1);
                        drop(src_guard);
                        let node_val = if is_end {
                            IntermediatePrecomputedNodeContents3::leaf()
                        } else {
                            IntermediatePrecomputedNodeContents3::internal()
                        };
                        let dest_idx = IntermediatePrecomputeNode3Index::new(god.insert(Trie::new(node_val)));
                        pair_cache.insert(key, dest_idx);
                        work.push_back(($src_idx, $stack));
                        dest_idx
                    }
                }
            };
        }

        // Create new roots at (source_root, empty stack)
        let mut new_roots: BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index> = BTreeMap::new();
        for (sid, src_root) in sids.into_iter().zip(source_roots.into_iter()) {
            let new_root = get_or_create!(src_root, stack_interner.empty());
            new_roots.insert(sid, new_root);
        }

        // BFS over product states
        let mut processed: u64 = 0;
        while let Some((src_idx, stack_id)) = work.pop_front() {
            processed += 1;

            // --- Diagnostics and Safeguards ---
            if processed > MAX_BFS_ITERATIONS {
                panic!(
                    "Push/Pop elimination exceeded maximum BFS iterations ({}). This indicates a likely infinite loop or state explosion. Last processed node: {:?}, stack depth: {}, stack: {}. Queue size: {}.",
                    MAX_BFS_ITERATIONS, src_idx, stack_interner.depth(stack_id), stack_interner.to_string(stack_id), work.len(),
                );
            }
            let count = visit_counts.entry(src_idx).or_insert(0);
            *count += 1;
            if !hot_nodes.contains(&src_idx) && *count > HOT_NODE_THRESHOLD {
                if is_debug_level_enabled(1) {
                    println!("[WARN] Push/Pop elimination: Node {:?} became hot (visited > {} times). Flushing stacks on subsequent visits to prevent state explosion.", src_idx, HOT_NODE_THRESHOLD);
                }
                hot_nodes.insert(src_idx);
            }

            pb.set_position(processed);
            if processed & 0xfff == 0 {
                pb.set_message(format!("queue={}", work.len()));
            }
            let dest_idx = *pair_cache.get(&(src_idx, stack_id)).expect("dest exists");

            // --- Hot Node Stack Flushing Heuristic ---
            if hot_nodes.contains(&src_idx) && !stack_interner.is_empty(stack_id) {
                // This node is hot and has a pending stack. Flush the stack to the graph
                // and then let the re-queued (src_idx, empty_stack) state handle propagation.
                // This converts stack complexity into graph complexity locally, containing the explosion.
                let empty_stack_id = stack_interner.empty();
                let final_dest_idx = get_or_create!(src_idx, empty_stack_id);

                let items = stack_interner.to_vec_bottom_up(stack_id);
                let mut from_idx = dest_idx;

                // Create a chain of intermediate nodes for all but the last push.
                for i in 0..(items.len().saturating_sub(1)) {
                    let new_intermediate_node = Trie::new(IntermediatePrecomputedNodeContents3::internal());
                    let to_idx = god.insert(new_intermediate_node).into();
                    god.insert_edge_simple(from_idx, to_idx, IntermediateTrie3EdgeKey::Push(items[i].clone()), ());
                    from_idx = to_idx;
                }

                // The last push connects to the final destination.
                if let Some(last_item) = items.last() {
                    god.insert_edge_simple(from_idx, final_dest_idx, IntermediateTrie3EdgeKey::Push(last_item.clone()), ());
                } else {
                    // This case (empty stack) is already handled by the `if` condition, but for completeness:
                    god.insert_edge_simple(from_idx, final_dest_idx, IntermediateTrie3EdgeKey::NoOp, ());
                }

                continue; // Stop processing this complex state; the empty-stack version will take over.
            }

            let src_guard = src_idx.read(&source).expect("source read");

            // If this source node is an end, flush the entire pending stack in order.
            // Note: pending stack bitvectors may already have been narrowed by Pop(0) constraints.
            if src_guard.value.end && !stack_interner.is_empty(stack_id) {
                let mut from_idx = dest_idx;
                // Bottom-to-top order
                let items = stack_interner.to_vec_bottom_up(stack_id);
                for i in 0..items.len() {
                    let label = IntermediateTrie3EdgeKey::Push(items[i].clone());
                    let next_stack_id = stack_interner.from_bottom_slice(&items[i+1..]);
                    let to_idx = get_or_create!(src_idx, next_stack_id);
                    god.insert_edge_simple(from_idx, to_idx, label, ());
                    from_idx = to_idx;
                }
            }

            for (ek, dests) in src_guard.children().iter() {
                match ek {
                    IntermediateTrie3EdgeKey::Push(bv_new) => {
                        let is_hot = hot_nodes.contains(&src_idx);
                        // Heuristic: If a source node is part of a pre-calculated "push cycle" OR
                        // if it has become dynamically "hot" (visited too many times), we stop
                        // deferring its pushes. Instead, we emit them directly into the graph.
                        // This prevents combinatorial explosion of stack states from complex cycles.
                        if push_cycle_nodes.contains(&src_idx) || is_hot {
                            if is_hot && !push_cycle_nodes.contains(&src_idx) && is_debug_level_enabled(3) {
                                println!("[Push/Pop Elimination] Dynamically forcing immediate push for hot source {:?}.", src_idx);
                            }

                            for (dst_src_idx, _) in dests.iter() {
                                let next_state = get_or_create!(*dst_src_idx, stack_id);
                                god.insert_edge_simple(dest_idx, next_state, IntermediateTrie3EdgeKey::Push(bv_new.clone()), ());
                            }
                        } else {
                            // Default: defer emission by pushing onto the stack and carrying via NoOp.
                            let new_stack_id = stack_interner.push(stack_id, bv_new.clone());
                            if stack_interner.depth(new_stack_id) > MAX_STACK_DEPTH {
                                if is_debug_level_enabled(1) {
                                    println!(
                                        "[WARN] Push/Pop elimination: stack depth > {} exceeded. Path dropped.",
                                        MAX_STACK_DEPTH
                                    );
                                }
                                continue;
                            }

                            for (dst_src_idx, _) in dests.iter() {
                                let next_state = get_or_create!(*dst_src_idx, new_stack_id);
                                god.insert_edge_simple(dest_idx, next_state, IntermediateTrie3EdgeKey::NoOp, ());
                            }
                        }
                    }
                    IntermediateTrie3EdgeKey::Pop(n, pop_bv) => {
                        if stack_interner.is_empty(stack_id) {
                            // No pending: forward Pop as-is.
                            for (dst_src_idx, _) in dests.iter() {
                                let next_state = get_or_create!(*dst_src_idx, stack_interner.empty());
                                god.insert_edge_simple(dest_idx, next_state, IntermediateTrie3EdgeKey::Pop(*n, pop_bv.clone()), ());
                            }
                        } else {
                            if *n == 0 {
                                let top = stack_interner.top(stack_id).expect("non-empty");
                                if top.is_disjoint(pop_bv) {
                                    // Invalid path; drop this branch (no edge emitted).
                                    continue;
                                }
                                // Intersect: fold Pop(0, B) into the top-of-stack A := A ∩ B (non-consuming).
                                // Replace the top-of-stack with the narrowed bitvector.
                                let mut narrowed = top.clone();
                                narrowed &= pop_bv.clone();
                                // If intersection is empty we'd have continued above.
                                let suffix = stack_interner.pop(stack_id).expect("non-empty");
                                let new_stack_id = stack_interner.push(suffix, narrowed);

                                for (dst_src_idx, _) in dests.iter() {
                                    let next_state = get_or_create!(*dst_src_idx, new_stack_id);
                                    god.insert_edge_simple(dest_idx, next_state, IntermediateTrie3EdgeKey::NoOp, ());
                                }
                            } else {
                                // Consume as many pops as possible immediately.
                                // For k > 1, we can pop unconditionally; for the final k == 1,
                                // we must check intersection against the current top.
                                let mut k = *n;
                                let mut sid2 = stack_id;

                                // Unconditional pops for k > 1 while we have a stack.
                                while k > 1 && !stack_interner.is_empty(sid2) {
                                    sid2 = stack_interner.pop(sid2).expect("non-empty");
                                    k -= 1;
                                }

                                if k == 1 {
                                    if let Some(top2) = stack_interner.top(sid2) {
                                        if top2.is_disjoint(pop_bv) {
                                            // Invalid path; drop this branch.
                                            continue;
                                        }
                                        // Intersection: pop the remaining top and emit NoOp.
                                        let final_sid = stack_interner.pop(sid2).expect("non-empty");
                                        for (dst_src_idx, _) in dests.iter() {
                                            let next_state = get_or_create!(*dst_src_idx, final_sid);
                                            god.insert_edge_simple(dest_idx, next_state, IntermediateTrie3EdgeKey::NoOp, ());
                                        }
                                    } else {
                                        // No pushes left to match k == 1; forward Pop(1, ...) as-is.
                                        for (dst_src_idx, _) in dests.iter() {
                                            let next_state = get_or_create!(*dst_src_idx, sid2);
                                            god.insert_edge_simple(
                                                dest_idx,
                                                next_state,
                                                IntermediateTrie3EdgeKey::Pop(1, pop_bv.clone()),
                                                (),
                                            );
                                        }
                                    }
                                } else {
                                    // k > 1 and stack is empty: forward the remaining Pop(k, ...) as-is.
                                    for (dst_src_idx, _) in dests.iter() {
                                        let next_state = get_or_create!(*dst_src_idx, sid2);
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
                    // Non-push/pop edges (e.g., CheckLLM, NoOp) just forward; pending unchanged.
                    other => {
                        for (dst_src_idx, _) in dests.iter() {
                            let next_state = get_or_create!(*dst_src_idx, stack_id);
                            god.insert_edge_simple(dest_idx, next_state, other.clone(), ());
                        }
                    }
                }
            }
        }

        pb.finish_with_message("Done eliminating push/pop pairs");

        // Replace input roots with new roots (pending_stack is empty).
        *roots = new_roots;

        // Fixpoint check: if the signature did not change, we converged.
        let after_sig = compute_graph_signature(god, roots);
        if after_sig == before_sig {
            if is_debug_level_enabled(2) {
                println!("[Push/Pop Elimination] Converged after {} pass(es).", pass + 1);
            }
            break;
        } else if is_debug_level_enabled(3) {
            println!(
                "[Push/Pop Elimination] Pass {} changed signature: before={:?} after={:?}",
                pass + 1,
                before_sig,
                after_sig
            );
        }
    }
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
