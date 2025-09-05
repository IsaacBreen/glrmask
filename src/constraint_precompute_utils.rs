use crate::constraint::{GrammarConstraintConfig, PrecomputeNode2, PrecomputeNode2Index, PrecomputeNodeIndex, Trie2GodWrapper, PrecomputeNode3, PrecomputeNode3Index, Trie3GodWrapper, StateIDBV};
use crate::datastructures::gss::{LLMTokenBV, PrecomputedNodeContents};
use crate::datastructures::ordered_hash_map::Retain;
use crate::datastructures::trie::{EdgeInserter, Trie, Trie2Index};
use crate::datastructures::{ArcPtrWrapper, EntryApi};
use crate::glr::table::StateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use crate::tokenizer::TokenizerStateID;
use deterministic_hash::DeterministicHasher;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use ordered_hash_map::OrderedHashMap;
use rand::prelude::IndexedRandom;
use rand::Rng;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::hash::{Hash};
use std::sync::{Arc, RwLock};

type NormalizedPath = Vec<(usize, StateID)>;
type PathMap = BTreeMap<NormalizedPath, LLMTokenBV>;

/// Samples a single normalized path by performing a random walk from the root.
fn sample_normalized_path(
    root: &Trie2Index,
    rng: &mut impl Rng,
    max_len: usize,
    trie2_god: &Trie2GodWrapper,
) -> Option<NormalizedPath> {
    let mut current_node = root.clone();
    let mut path = NormalizedPath::new();
    let mut current_k = 0;
    let mut bv = root.read(trie2_god).unwrap().value.live_tokens.clone();

    while path.len() < max_len {
        let can_terminate = current_node.read(trie2_god).unwrap().value.end;
        let can_continue = !current_node.read(trie2_god).unwrap().children().is_empty();

        if !can_continue {
            return if can_terminate { Some(path) } else { None };
        }

        if can_terminate && rng.gen_bool(0.2) { // 20% chance to terminate at an end node
            return Some(path);
        }

        let all_outgoing_edges: Vec<_> = current_node.read(trie2_god).unwrap()
            .children()
            .iter()
            .flat_map(|(ek, dest_map)| {
                dest_map.iter().map(move |(dest_ptr, edge_bv)| (ek.clone(), dest_ptr.clone(), edge_bv.clone()))
            })
            .collect();

        if all_outgoing_edges.is_empty() {
            return if current_node.read(trie2_god).unwrap().value.end { Some(path) } else { None };
        }

        let (ek, dest_ptr, edge_bv) = all_outgoing_edges.choose(rng)?;

        bv &= edge_bv;
        if bv.is_empty() {
            return None; // Path became invalid
        }

        let (k, sid_opt) = ek;
        current_k += k;
        if let Some(sid) = sid_opt {
            path.push((current_k, *sid));
            current_k = 0;
        }

        current_node = dest_ptr.as_arc().clone();
    }

    Some(path)
}

/// For a given normalized path, computes the union of LLM token bitvectors for all
/// possible ways to traverse that path in the trie.
fn get_bv_for_normalized_path(
    root: &Trie2Index,
    path: &NormalizedPath,
    trie2_god: &Trie2GodWrapper,
) -> LLMTokenBV {
    // State: (current_node, path_segment_index, accumulated_k, current_bv)
    let mut q: VecDeque<(Trie2Index, usize, usize, LLMTokenBV)> = VecDeque::new();
    let mut final_bv = LLMTokenBV::zeros();

    let initial_bv = root.read(trie2_god).unwrap().value.live_tokens.clone();
    q.push_back((root.clone(), 0, 0, initial_bv.clone()));

    // To handle cycles and redundant exploration
    let mut visited: HashMap<(PrecomputeNode2Index, usize, usize), LLMTokenBV> = HashMap::new();
    visited.insert((*root, 0, 0), initial_bv);

    while let Some((node, path_idx, k_so_far, bv)) = q.pop_front() {
        // Check if we've completed the path
        if path_idx == path.len() {
            // We have successfully traversed the path. Now we need to reach an `end` node from here
            // with only `(k, None)` edges.
            let end_bv = find_end_bv_from_node_via_none_edges(node, bv, &trie2_god);
            final_bv |= &end_bv;
            continue;
        }

        let (target_k, target_sid) = path[path_idx];

        // Explore children
        let guard = node.read(trie2_god).unwrap();
        for (ek, dest_map) in guard.children() {
            for (dest_ptr, edge_bv) in dest_map {
                let new_bv = &bv & edge_bv;
                if new_bv.is_empty() { continue; }

                let child_arc = dest_ptr.as_arc().clone();
                let (k, sid_opt) = ek;
                let new_k = k_so_far + *k;

                if let Some(sid) = sid_opt {
                    if new_k == target_k && sid == &target_sid {
                        // Matched a path segment. Advance.
                        let visited_key = (child_arc, path_idx + 1, 0);
                        if let Some(existing_bv) = visited.get_mut(&visited_key) {
                            let diff = &new_bv - &*existing_bv;
                            if !diff.is_empty() {
                                *existing_bv |= &diff;
                                q.push_back((child_arc, path_idx + 1, 0, diff));
                            }
                        } else {
                            visited.insert(visited_key, new_bv.clone());
                            q.push_back((child_arc, path_idx + 1, 0, new_bv));
                        }
                    }
                } else { // sid_opt is None
                    if new_k <= target_k {
                        // Continue accumulating k
                        let visited_key = (child_arc, path_idx, new_k);
                        if let Some(existing_bv) = visited.get_mut(&visited_key) {
                            let diff = &new_bv - &*existing_bv;
                            if !diff.is_empty() {
                                *existing_bv |= &diff;
                                q.push_back((child_arc, path_idx, new_k, diff));
                            }
                        } else {
                            visited.insert(visited_key, new_bv.clone());
                            q.push_back((child_arc, path_idx, new_k, new_bv));
                        }
                    }
                }
            }
        }
    }
    final_bv
}

/// Helper to find the union of BVs for all paths from a start node to any `end` node
/// that consist solely of `(k, None)` edges.
fn find_end_bv_from_node_via_none_edges(
    start_node: Trie2Index,
    initial_bv: LLMTokenBV,
    trie2_god: &Trie2GodWrapper,
) -> LLMTokenBV {
    let mut end_bv = LLMTokenBV::zeros();
    let mut q = VecDeque::new();
    q.push_back((start_node, initial_bv));
    let mut visited: HashMap<PrecomputeNode2Index, LLMTokenBV> = HashMap::new();

    while let Some((node, bv)) = q.pop_front() {
        let guard = node.read(trie2_god).unwrap();
        if guard.value.end {
            end_bv |= &bv;
        }

        for (ek, dest_map) in guard.children() {
            let (_k, sid_opt) = ek;
            if sid_opt.is_none() { // Only (k, None) edges
                for (dest_ptr, edge_bv) in dest_map {
                    let new_bv = &bv & edge_bv;
                    if new_bv.is_empty() { continue; }

                    let child_arc = dest_ptr.as_arc().clone();
                    let child_ptr = child_arc;
                    if let Some(existing_bv) = visited.get_mut(&child_ptr) {
                        let diff = &new_bv - &*existing_bv;
                        if !diff.is_empty() {
                            *existing_bv |= &diff;
                            q.push_back((child_arc, diff));
                        }
                    } else {
                        visited.insert(child_ptr, new_bv.clone());
                        q.push_back((child_arc, new_bv));
                    }
                }
            }
        }
    }
    end_bv
}

/// Checks for semantic equivalence between two `precompute2` trees.
///
/// Two trees are considered equivalent if they generate the same set of "normalized paths",
/// where each path is associated with a bitvector of applicable LLM tokens.
/// A normalized path collapses consecutive edge keys of the form `(k, None)`.
///
/// # Arguments
/// * `a`: An `Arc` to the first trie's root node.
/// * `b`: An `Arc` to the second trie's root node.
///
/// # Returns
/// `true` if the tries are semantically equivalent, `false` otherwise.
pub fn are_precompute2_trees_equivalent(
    a: &Trie2Index,
    trie2_god_a: &Trie2GodWrapper,
    b: &Trie2Index,
    trie2_god_b: &Trie2GodWrapper,
) -> bool {
    // Stochastic version
    if a == b && trie2_god_a == trie2_god_b {
        return true;
    }

    const NUM_SAMPLES: usize = 100;
    const MAX_PATH_LEN: usize = 32;
    let mut rng = rand::thread_rng();

    // Sample from A, check in B
    for i in 0..NUM_SAMPLES {
        if let Some(path) = sample_normalized_path(a, &mut rng, MAX_PATH_LEN, trie2_god_a) {
            let bv_a = get_bv_for_normalized_path(a, &path, trie2_god_a);
            if bv_a.is_empty() && i > 0 {
                continue;
            } // Skip trivial paths, but always check the empty path
            let bv_b = get_bv_for_normalized_path(b, &path, trie2_god_b);
            if bv_a != bv_b {
                println!("\n--- Precompute2 Equivalence Mismatch ---");
                println!("Path sampled from Tree A:");
                println!("  Path: {:?}", path);
                println!("  BV from A: {:?}", bv_a);
                println!("  BV from B: {:?}", bv_b);
                println!("  Difference (A ^ B): {:?}", bv_a.symmetric_difference(&bv_b));
                return false;
            }
        }
    }

    // Sample from B, check in A
    for i in 0..NUM_SAMPLES {
        if let Some(path) = sample_normalized_path(b, &mut rng, MAX_PATH_LEN, trie2_god_b) {
            let bv_b = get_bv_for_normalized_path(b, &path, trie2_god_b);
            if bv_b.is_empty() && i > 0 {
                continue;
            } // Skip trivial paths, but always check the empty path
            let bv_a = get_bv_for_normalized_path(a, &path, trie2_god_a);
            if bv_a != bv_b {
                println!("\n--- Precompute2 Equivalence Mismatch ---");
                println!("Path sampled from Tree B:");
                println!("  Path: {:?}", path);
                println!("  BV from A: {:?}", bv_a);
                println!("  BV from B: {:?}", bv_b);
                println!("  Difference (A ^ B): {:?}", bv_a.symmetric_difference(&bv_b));
                return false;
            }
        }
    }

    true
}

pub fn prune_dead_paths_trie2(roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode2Index>, trie2_god: &Trie2GodWrapper) {
    crate::debug!(2, "Pruning dead paths from precomputed trie 2.");

    // Use a worklist algorithm to propagate "liveness" backwards from end nodes.
    // This correctly handles cycles, iterating until a fixed point is reached.
    let all_nodes = Trie::all_nodes(trie2_god, &roots.values().cloned().collect::<Vec<_>>());
    let mut predecessors: HashMap<PrecomputeNode2Index, Vec<(PrecomputeNode2Index, LLMTokenBV)>> = HashMap::new();
    let mut worklist = VecDeque::new();
    let mut live: HashMap<PrecomputeNode2Index, LLMTokenBV> = HashMap::new();

    // 1. Initialize live sets and build predecessor map.
    for node_arc in &all_nodes {
        let node_ptr = *node_arc;
        live.insert(node_ptr, LLMTokenBV::zeros());

        let guard = node_arc.read(trie2_god).unwrap();
        if guard.value.end {
            let initial_live = guard.value.live_tokens.clone();
            if !initial_live.is_empty() {
                live.insert(node_ptr, initial_live);
                worklist.push_back(node_ptr);
            }
        }

        for dest_map in guard.children().values() {
            for (child_wrap, edge_bv) in dest_map {
                let child_arc = child_wrap.as_arc().clone();
                let child_ptr = child_arc;
                predecessors.entry(child_ptr).or_default().push((node_ptr, edge_bv.clone()));
            }
        }
    }

    // 2. Propagate liveness until a fixed point is reached.
    while let Some(node_ptr) = worklist.pop_front() {
        let live_at_node = live.get(&node_ptr).unwrap().clone();
        if let Some(preds) = predecessors.get(&node_ptr) {
            for (pred_ptr, edge_bv) in preds {
                let live_from_edge = &live_at_node & edge_bv;
                if live_from_edge.is_empty() {
                    continue;
                }

                let pred_live = live.get_mut(pred_ptr).unwrap();
                let old_len = pred_live.len();
                *pred_live |= &live_from_edge;
                if pred_live.len() > old_len {
                    worklist.push_back(*pred_ptr);
                }
            }
        }
    }

    // 3. Prune the graph based on the computed live sets.
    for node_arc in &all_nodes {
        let mut guard = node_arc.write(trie2_god).unwrap();
        guard.children_mut().retain(|_edge_key, dest_map| {
            dest_map.retain(|child_wrapper, edge_value_bv| {
                let child_arc = child_wrapper.as_arc().clone();
                let child_ptr = child_arc;
                let live_from_child = live.get(&child_ptr).unwrap();
                let live_on_edge = &*edge_value_bv & live_from_child;
                if live_on_edge.is_empty() {
                    false
                } else {
                    *edge_value_bv = live_on_edge;
                    true
                }
            });
            !dest_map.is_empty()
        });
        // Update the node's own live_tokens field with the final computed value.
        let node_ptr = *node_arc;
        guard.value.live_tokens = live.get(&node_ptr).unwrap().clone();
    }
    crate::debug!(2, "Finished pruning dead paths from trie 2.");
}

pub fn simplify_trie2_factor_common_destinations(roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode2Index>, trie2_god: &Trie2GodWrapper) {
    crate::debug!(2, "Simplifying trie 2 by factoring common destinations.");

    const MIN_INCOMING_EDGES_FOR_FACTORING: usize = 3;

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie2_god, &roots_vec);
    let arc_map: HashMap<_, _> = all_nodes.iter().map(|n| (*n, n.clone())).collect();

    type EdgeKey2 = (usize, Option<StateID>);
    let mut incoming_map: HashMap<
        PrecomputeNode2Index,
        HashMap<
            EdgeKey2,
            Vec<(PrecomputeNode2Index, LLMTokenBV)>,
        >,
    > = HashMap::new();

    for src_arc in &all_nodes {
        let src_ptr = *src_arc;
        let guard = src_arc.read(trie2_god).expect("poison");
        for (ek, dest_map) in guard.children() {
            for (dest_wrapper, bv) in dest_map {
                let dest_arc = dest_wrapper.as_arc().clone();
                let dest_ptr = dest_arc;
                incoming_map
                    .entry(dest_ptr)
                    .or_default()
                    .entry(ek.clone())
                    .or_default()
                    .push((src_ptr, bv.clone()));
            }
        }
    }

    for (dest_ptr, edges_by_key) in incoming_map {
        for (edge_key, sources) in edges_by_key {
            if sources.len() >= MIN_INCOMING_EDGES_FOR_FACTORING {
                let dest_arc = arc_map.get(&dest_ptr).unwrap().clone();

                let intermediate_node = PrecomputeNode2Index::new(trie2_god.insert(PrecomputeNode2::new(PrecomputedNodeContents::internal())));

                let mut union_bv = LLMTokenBV::zeros();
                for (_, bv) in &sources {
                    union_bv |= bv;
                }

                {
                    let mut intermediate_guard = intermediate_node.write(trie2_god).expect("poison");
                    let mut edge_val_opt = Some(union_bv.clone());
                    intermediate_guard.try_insert_unchecked(edge_key.clone(), &mut edge_val_opt, dest_arc.clone());
                    intermediate_guard.value.live_tokens |= &union_bv;
                }

                let identity_edge_key = (0, None);
                for (src_ptr, bv) in &sources {
                    let src_arc = arc_map.get(src_ptr).unwrap();
                    let mut src_guard = src_arc.write(trie2_god).expect("poison");

                    let mut remove_ek = false;
                    if let Some(dest_map_for_ek) = src_guard.children_mut().get_mut(&edge_key) {
                        let strong_key = dest_arc.clone();
                        dest_map_for_ek.remove(&strong_key);
                        if dest_map_for_ek.is_empty() {
                            remove_ek = true;
                        }
                    }
                    if remove_ek {
                        src_guard.children_mut().remove(&edge_key);
                    }

                    let mut edge_val_opt = Some(bv.clone());
                    src_guard.try_insert_unchecked(identity_edge_key.clone(), &mut edge_val_opt, intermediate_node.clone()); // ignore cycle error, should not happen
                    src_guard.value.live_tokens |= bv;
                }
            }
        }
    }
    crate::debug!(2, "Finished factoring common destinations in trie 2.");
}

pub fn optimize_trie2_size(
    roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode2Index>,
    trie2_god: &Trie2GodWrapper,
    config: &GrammarConstraintConfig,
) {
    crate::debug!(2, "Optimizing Trie 2 size...");
    // Pin all nodes to prevent dangling weak pointers while we rewire.
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes_pinner = Trie::all_nodes(&trie2_god, &roots_vec);

    if config.optimize_trie2_prune_dead_paths {
        prune_dead_paths_trie2(roots, &trie2_god);
    }
    if config.optimize_trie2_merge_nodes {
        merge_nodes_trie2(roots, &trie2_god);
    }
    if config.optimize_trie2_factor_common_destinations {
        simplify_trie2_factor_common_destinations(roots, &trie2_god);
    }
    if config.optimize_trie2_compress_edges {
        compress_trie2_edges(roots, &trie2_god);
    }
    if config.optimize_trie2_prune_dead_paths {
        prune_dead_paths_trie2(roots, &trie2_god);
    }
    if config.optimize_trie2_merge_nodes {
        merge_nodes_trie2(roots, &trie2_god);
    }
    if config.optimize_trie2_gc {
        Trie::gc(&trie2_god, &roots.values().cloned().collect::<Vec<_>>());
    }
    Trie::recompute_all_max_depths(&trie2_god, &roots.values().cloned().collect::<Vec<_>>());
}

pub fn merge_nodes_trie2(roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode2Index>, trie2_god: &Trie2GodWrapper) {
    crate::debug!(2, "Merging identical subtrees in precomputed trie 2.");

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie2_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    // 1. Densify indices
    let mut dense_of: HashMap<Trie2Index, usize> = HashMap::new();
    let mut old_of: Vec<Trie2Index> = Vec::with_capacity(all_nodes.len());
    for (i, node_idx) in all_nodes.iter().enumerate() {
        dense_of.insert(*node_idx, i);
        old_of.push(*node_idx);
    }
    let n = all_nodes.len();

    // 2. Extract raw edges and end flags
    let mut ends: Vec<bool> = vec![false; n];
    type RawEdge = (usize, Option<StateID>, usize, LLMTokenBV);
    let mut raw_edges: Vec<Vec<RawEdge>> = vec![Vec::new(); n];

    for (u_dense, u_idx) in old_of.iter().enumerate() {
        let guard = u_idx.read(trie2_god).unwrap();
        ends[u_dense] = guard.value.end;
        for (ek, dest_map) in guard.children() {
            for (v_idx, bv) in dest_map {
                if let Some(&v_dense) = dense_of.get(v_idx) {
                    raw_edges[u_dense].push((ek.0, ek.1, v_dense, bv.clone()));
                }
            }
        }
    }

    // 3. Initialize classes by end flag
    let mut prev_class: Vec<usize> = (0..n).map(|i| if ends[i] { 1 } else { 0 }).collect();

    // 4. Refinement loop
    const MAX_ITERS: usize = 40;
    for it in 0..MAX_ITERS {
        type AggregatedEdge = ((usize, Option<StateID>, usize), LLMTokenBV);
        type Signature = (bool, Vec<AggregatedEdge>);

        let mut sig_to_id: HashMap<Signature, usize> = HashMap::new();
        let mut new_class = vec![0; n];
        let mut next_id = 0;
        let mut changes = 0;

        for u in 0..n {
            // Aggregate edges for node u
            let mut aggr: BTreeMap<(usize, Option<StateID>, usize), LLMTokenBV> = BTreeMap::new();
            for (p, s, v_dense, bv) in &raw_edges[u] {
                let dest_class = prev_class[*v_dense];
                let key = (*p, *s, dest_class);
                aggr.entry(key).and_modify(|e| *e |= bv).or_insert_with(|| bv.clone());
            }
            let agg_edges: Vec<AggregatedEdge> = aggr.into_iter().collect();

            let sig: Signature = (ends[u], agg_edges);

            let cid = *sig_to_id.entry(sig).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });

            new_class[u] = cid;
            if new_class[u] != prev_class[u] {
                changes += 1;
            }
        }

        crate::debug!(3, "Trie2 merge iter {}: classes={}, changes={}", it + 1, next_id, changes);
        prev_class = new_class;
        if changes == 0 {
            break;
        }
    }

    let final_partition = prev_class;
    let num_classes = final_partition.iter().max().map_or(0, |m| m + 1);

    // 5. Build quotient graph (in-place modification)
    let mut representatives: Vec<Option<Trie2Index>> = vec![None; num_classes];
    for (u_dense, &class_id) in final_partition.iter().enumerate() {
        if representatives[class_id].is_none() {
            representatives[class_id] = Some(old_of[u_dense]);
        }
    }

    let mut node_to_rep: HashMap<Trie2Index, Trie2Index> = HashMap::new();
    for (u_dense, &class_id) in final_partition.iter().enumerate() {
        node_to_rep.insert(old_of[u_dense], representatives[class_id].unwrap());
    }

    // Update representatives' children and live_tokens
    for class_id in 0..num_classes {
        if let Some(rep_idx) = representatives[class_id] {
            // Find any node `u` in this class to compute aggregated edges
            let u_dense = final_partition.iter().position(|&c| c == class_id).unwrap();

            let mut aggr: BTreeMap<(usize, Option<StateID>, usize), LLMTokenBV> = BTreeMap::new();
            for (p, s, v_dense, bv) in &raw_edges[u_dense] {
                let dest_class = final_partition[*v_dense];
                aggr.entry((*p, *s, dest_class)).and_modify(|e| *e |= bv).or_insert_with(|| bv.clone());
            }

            let mut new_children = BTreeMap::new();
            let mut new_live_tokens = LLMTokenBV::zeros();
            for ((p, s, dest_class), bv) in aggr {
                if let Some(dest_rep_idx) = representatives[dest_class] {
                    new_children.entry((p, s)).or_insert_with(OrderedHashMap::new).insert(dest_rep_idx, bv.clone());
                    new_live_tokens |= &bv;
                }
            }

            // Also union live_tokens from all nodes in the class
            for (i, &c) in final_partition.iter().enumerate() {
                if c == class_id {
                    new_live_tokens |= &old_of[i].read(trie2_god).unwrap().value.live_tokens;
                }
            }

            let mut guard = rep_idx.write(trie2_god).unwrap();
            *guard.children_mut() = new_children;
            guard.value.live_tokens = new_live_tokens;
        }
    }

    // Update roots
    for root_idx in roots.values_mut() {
        *root_idx = *node_to_rep.get(root_idx).unwrap();
    }

    let pb = ProgressBar::new(all_nodes.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta})")
            .expect("progress-bar"),
    );
    if !PROGRESS_BAR_ENABLED {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    // Recompute depths after structural changes from merging
    let final_roots_vec: Vec<_> = roots.values().cloned().collect();
    Trie::recompute_all_max_depths(trie2_god, &final_roots_vec);

    pb.finish_with_message("Finished merging Trie 2 nodes");
}

/// Compress linear chains in Trie by merging consecutive edges where safe.
///
/// Given A -(k1,s1)-> B and B -(k2,s2)-> C, this pass replaces with
/// A -(k1+k2, s1.or(s2))-> C with edge BV = bv1 ∧ bv2,
/// provided:
///   - B is not an end node.
///   - B has exactly one outgoing destination (across all keys).
///   - B has exactly one incoming edge (across all parents, strong or weak).
///   - Not both s1 and s2 are Some(...) (i.e., at most one has a state ID).
/// This reduces redundant intermediate nodes introduced during construction.
pub fn compress_trie2_edges(
    roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode2Index>,
    trie2_god: &Trie2GodWrapper,
) {
    crate::debug!(2, "Compressing Trie 2 by merging linear chains...");
    type EdgeKey2 = (usize, Option<StateID>);

    // Helper to count incoming edges for each node (both strong and weak).
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let mut changed = true;
    let mut iterations = 0usize;
    let _all_nodes = Trie::all_nodes(trie2_god, &roots_vec);

    while changed && iterations < 5 { // Add iteration limit to prevent infinite loops in buggy cases
        iterations += 1;
        changed = false;
        let all_nodes = Trie::all_nodes(trie2_god, &roots_vec);
        let mut arc_map: HashMap<PrecomputeNode2Index, PrecomputeNode2Index> = HashMap::new();
        for n in &all_nodes {
            arc_map.insert(*n, n.clone());
        }

        // Build incoming counts
        let mut incoming_count: HashMap<PrecomputeNode2Index, usize> = HashMap::new();
        for src_arc in &all_nodes {
            let guard = src_arc.read(trie2_god).expect("poison");
            for (_ek, dest_map) in guard.children() {
                for (node_ptr, _ev) in dest_map {
                    let child_arc = node_ptr.as_arc().clone();
                    let ptr = child_arc;
                    *incoming_count.entry(ptr).or_insert(0) += 1;
                }
            }
        }

        // Try to compress from each source node
        'src_loop: for src_arc in &all_nodes {
            // Snapshot children
            let children_snapshot: Vec<(EdgeKey2, Vec<(PrecomputeNode2Index, LLMTokenBV)>)> = {
                let g = src_arc.read(trie2_god).expect("poison");
                g.children()
                    .iter()
                    .map(|(ek, dest_map)| {
                        let entries = dest_map
                            .iter()
                            .map(|(np, ev)| (np.clone(), ev.clone()))
                            .collect::<Vec<_>>();
                        (ek.clone(), entries)
                    })
                    .collect()
            };

            for (ek1, entries) in children_snapshot {
                for (child_ptr, bv1) in entries {
                    let child_arc = child_ptr.as_arc().clone();
                    let child_ptr_raw = child_arc;

                    // Preconditions: B has in-degree 1, not end, and exactly one outgoing edge overall.
                    if incoming_count.get(&child_ptr_raw).cloned().unwrap_or(0) != 1 {
                        continue;
                    }
                    let (is_end, child_outgoing): (bool, Vec<(EdgeKey2, Vec<(PrecomputeNode2Index, LLMTokenBV)>)>) = {
                        let cg = child_arc.read(trie2_god).expect("poison");
                        let mut out = Vec::new();
                        for (ek, dest_map) in cg.children() {
                            let mut v = Vec::new();
                            for (np, ev) in dest_map {
                                v.push((np.clone(), ev.clone()));
                            }
                            if !v.is_empty() {
                                out.push((ek.clone(), v));
                            }
                        }
                        (cg.value.end, out)
                    };
                    if is_end {
                        continue;
                    }
                    // Exactly one outgoing destination (across all keys)
                    if child_outgoing.len() != 1 {
                        continue;
                    }
                    let (ek2, dests2) = &child_outgoing[0];
                    if dests2.len() != 1 {
                        continue;
                    }
                    let (grand_ptr, bv2) = &dests2[0];
                    let grand_arc = grand_ptr.as_arc().clone();
                    // Check state-id merge safety: not both Some(...)
                    let s1 = ek1.1;
                    // The first edge in a compressible chain must not have a state ID check.
                    // A state ID check on the first edge (s1) is an intermediate validation
                    // that is lost if we merge the edges. We can only merge if the first
                    // edge is just a pop (s1 is None).
                    if s1.is_some() {
                        continue;
                    }
                    let s2 = ek2.1;

                    // Compute merged edge
                    let merged_k = ek1.0 + ek2.0;
                    let merged_sid = s1.or(s2);
                    let merged_key: EdgeKey2 = (merged_k, merged_sid);
                    let merged_bv = &bv1 & bv2;
                    if merged_bv.is_empty() {
                        continue;
                    }

                    // Perform rewire on src: subtract moved tokens from src->child edge, add/merge src->grand edge
                    {
                        let mut src_w = src_arc.write(trie2_god).expect("poison");
                        // 1) Reduce/remove src --ek1--> child by merged_bv
                        if let Some(dest_map_for_ek1) = src_w.children_mut().get_mut(&ek1) {
                            let child_key_in_map = child_arc.clone();
                            let mut removed = false;
                            if let Some(ev) = dest_map_for_ek1.get_mut(&child_key_in_map) {
                                *ev -= &merged_bv;
                                if ev.is_empty() {
                                    dest_map_for_ek1.remove(&child_key_in_map);
                                    removed = true;
                                }
                            }
                            if removed && dest_map_for_ek1.is_empty() {
                                src_w.children_mut().remove(&ek1);
                            }
                        }
                    }

                    // 2) Add/merge src --merged_key--> grand with merged_bv
                    {
                        let inserter = EdgeInserter::new(
                            trie2_god,
                            src_arc.clone(),
                            merged_key.clone(),
                            merged_bv.clone(),
                            |e, n| *e |= n,
                            |node_value, edge_value| node_value.live_tokens |= edge_value,
                            |ev, t| *ev &= &t.live_tokens,
                        );
                        let _ = inserter.try_destination(grand_arc.clone()).into_option();
                    }

                    // Mark progress; we'll iterate again to catch further compressible segments.
                    changed = true;
                    // Move on to next source after mutation to avoid borrow complexity.
                    continue 'src_loop;
                }
            }
        }

        // After a full pass, prune trivial dead ends introduced by compression.
        if changed {
            prune_dead_paths_trie2(roots, trie2_god);
            merge_nodes_trie2(roots, trie2_god);
        }
    }
    crate::debug!(2, "Finished compressing Trie 2 in {} iteration(s).", iterations);
}

pub fn clone_trie2_graph(
    root: &Trie2Index,
    trie2_god: &Trie2GodWrapper,
) -> (
    Trie2Index,
    HashMap<PrecomputeNode2Index, PrecomputeNode2Index>,
) {
    // old_ptr -> new arc
    let mut map: HashMap<PrecomputeNode2Index, PrecomputeNode2Index> = HashMap::new();
    let mut q: VecDeque<PrecomputeNode2Index> = VecDeque::new();

    let root_ptr = *root;
    let root_value = { root.read(trie2_god).expect("poison").value.clone() };
    let new_root = PrecomputeNode2Index::new(trie2_god.insert(PrecomputeNode2::new(root_value)));
    map.insert(root_ptr, new_root.clone());
    q.push_back(root.clone());

    while let Some(old_arc) = q.pop_front() {
        let old_ptr = old_arc;
        let new_arc = map.get(&old_ptr).expect("parent must be created").clone();

        // Snapshot children outside of lock to avoid recursive lock explosion.
        let children_snapshot: Vec<( (usize, Option<StateID>), Vec<(PrecomputeNode2Index, LLMTokenBV)> )> = {
            let g = old_arc.read(trie2_god).expect("poison");
            g.children()
                .iter()
                .map(|(ek, dest_map)| {
                    let entries = dest_map
                        .iter()
                        .map(|(node_ptr, ev)| {
                            (node_ptr.clone(), ev.clone())
                        })
                        .collect::<Vec<_>>();
                    (ek.clone(), entries)
                })
                .collect()
        };

        // For each child, ensure it exists in map (create a blank new node with same value).
        for (_ek, entries) in &children_snapshot {
            for (node_ptr, _ev) in entries {
                let child_arc_old = node_ptr.as_arc().clone();
                let child_ptr_old = child_arc_old;
                if !map.contains_key(&child_ptr_old) {
                    let child_value = { child_arc_old.read(trie2_god).expect("poison").value.clone() };
                    let child_arc_new = PrecomputeNode2Index::new(trie2_god.insert(PrecomputeNode2::new(child_value)));
                    map.insert(child_ptr_old, child_arc_new);
                    q.push_back(child_arc_old);
                }
            }
        }

        // Now wire edges on new_arc
        {
            let mut new_g = new_arc.write(trie2_god).expect("poison");
            for (ek, entries) in children_snapshot {
                let dest_map = new_g.children_mut().entry(ek).or_default();
                for (old_node_ptr, ev) in entries {
                    let child_arc_old = old_node_ptr.as_arc().clone();
                    let child_ptr_old = child_arc_old;
                    let child_arc_new = map.get(&child_ptr_old).expect("must exist").clone(); // With weak refs removed, all edges are strong.
                    let new_key = child_arc_new;
                    dest_map.insert(new_key, ev);
                }
            }
        }
    }

    // Recompute max_depths in the clone to keep invariants consistent.
    Trie::recompute_all_max_depths(trie2_god, &[new_root.clone()]);
    (new_root, map)
}

pub fn optimize_trie3_size(
    roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    config: &GrammarConstraintConfig,
) {
    crate::debug!(2, "Optimizing Trie 3 size...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let _all_nodes_pinner = Trie::all_nodes(&trie3_god, &roots_vec);

    if config.optimize_trie2_prune_dead_paths { // Reusing config flags from trie2
        prune_dead_paths_trie3(roots, &trie3_god);
    }
    if config.optimize_trie2_merge_nodes {
        merge_nodes_trie3(roots, &trie3_god);
    }
    if config.optimize_trie2_compress_edges {
        compress_trie3_edges(roots, &trie3_god);
    }
    if config.optimize_trie2_prune_dead_paths {
        prune_dead_paths_trie3(roots, &trie3_god);
    }
    if config.optimize_trie2_merge_nodes {
        merge_nodes_trie3(roots, &trie3_god);
    }
    if config.optimize_trie2_gc {
        Trie::gc(&trie3_god, &roots.values().cloned().collect::<Vec<_>>());
    }
    Trie::recompute_all_max_depths(&trie3_god, &roots.values().cloned().collect::<Vec<_>>());
}

pub fn prune_dead_paths_trie3(roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>, trie3_god: &Trie3GodWrapper) {
    crate::debug!(2, "Pruning dead paths from precomputed trie 3.");

    let all_nodes = Trie::all_nodes(trie3_god, &roots.values().cloned().collect::<Vec<_>>());
    if all_nodes.is_empty() { return; }

    let mut predecessors: HashMap<PrecomputeNode3Index, Vec<(PrecomputeNode3Index, (usize, LLMTokenBV))>> = HashMap::new();
    let mut worklist = VecDeque::new();
    let mut live: HashMap<PrecomputeNode3Index, LLMTokenBV> = HashMap::new();

    // 1. Initialize live sets and build predecessor map.
    for node_arc in &all_nodes {
        let node_ptr = *node_arc;
        live.insert(node_ptr, LLMTokenBV::zeros());

        let guard = node_arc.read(trie3_god).unwrap();
        if guard.value.end {
            let initial_live = guard.value.live_tokens.clone();
            if !initial_live.is_empty() {
                live.insert(node_ptr, initial_live);
                worklist.push_back(node_ptr);
            }
        }

        for (edge_key, dest_map) in guard.children() {
            for child_wrap in dest_map.keys() {
                let child_arc = child_wrap.as_arc().clone();
                let child_ptr = child_arc;
                predecessors.entry(child_ptr).or_default().push((node_ptr, edge_key.clone()));
            }
        }
    }

    // 2. Propagate liveness until a fixed point is reached.
    while let Some(node_ptr) = worklist.pop_front() {
        let live_at_node = live.get(&node_ptr).unwrap().clone();
        if let Some(preds) = predecessors.get(&node_ptr) {
            for (pred_ptr, edge_key) in preds {
                let live_from_edge = &live_at_node & &edge_key.1;
                if live_from_edge.is_empty() {
                    continue;
                }

                let pred_live = live.get_mut(pred_ptr).unwrap();
                let old_len = pred_live.len();
                *pred_live |= &live_from_edge;
                if pred_live.len() > old_len {
                    worklist.push_back(*pred_ptr);
                }
            }
        }
    }

    // 3. Prune the graph based on the computed live sets.
    for node_arc in &all_nodes {
        let mut guard = node_arc.write(trie3_god).unwrap();
        let mut new_children: BTreeMap<(usize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>> = BTreeMap::new();

        for (edge_key, dest_map) in guard.children() {
            for (child_wrapper, edge_value_sids) in dest_map {
                let child_arc = child_wrapper.as_arc().clone();
                let child_ptr = child_arc;
                let live_from_child = live.get(&child_ptr).unwrap();

                let live_on_edge = &edge_key.1 & live_from_child;

                if !live_on_edge.is_empty() {
                    let new_edge_key = (edge_key.0, live_on_edge);
                    let new_dest_map_for_key = new_children.entry(new_edge_key).or_default();
                    new_dest_map_for_key.entry(*child_wrapper)
                        .and_modify(|v| *v |= edge_value_sids)
                        .or_insert_with(|| edge_value_sids.clone());
                }
            }
        }
        *guard.children_mut() = new_children;

        // Update the node's own live_tokens field with the final computed value.
        let node_ptr = *node_arc;
        guard.value.live_tokens = live.get(&node_ptr).unwrap().clone();
    }
    crate::debug!(2, "Finished pruning dead paths from trie 3.");
}

pub fn merge_nodes_trie3(roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>, trie3_god: &Trie3GodWrapper) {
    crate::debug!(2, "Merging identical subtrees in precomputed trie 3.");

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() { return; }

    let mut dense_of: HashMap<Trie2Index, usize> = HashMap::new();
    let mut old_of: Vec<Trie2Index> = Vec::with_capacity(all_nodes.len());
    for (i, node_idx) in all_nodes.iter().enumerate() {
        dense_of.insert(*node_idx, i);
        old_of.push(*node_idx);
    }
    let n = all_nodes.len();

    let mut ends: Vec<bool> = vec![false; n];
    type RawEdge3 = (usize, LLMTokenBV, usize, StateIDBV);
    let mut raw_edges: Vec<Vec<RawEdge3>> = vec![Vec::new(); n];

    for (u_dense, u_idx) in old_of.iter().enumerate() {
        let guard = u_idx.read(trie3_god).unwrap();
        ends[u_dense] = guard.value.end;
        for (ek, dest_map) in guard.children() {
            for (v_idx, bv) in dest_map {
                if let Some(&v_dense) = dense_of.get(v_idx) {
                    raw_edges[u_dense].push((ek.0, ek.1.clone(), v_dense, bv.clone()));
                }
            }
        }
    }

    let mut prev_class: Vec<usize> = (0..n).map(|i| if ends[i] { 1 } else { 0 }).collect();

    const MAX_ITERS: usize = 40;
    for it in 0..MAX_ITERS {
        type AggregatedEdge3 = ((usize, LLMTokenBV, usize), StateIDBV);
        type Signature3 = (bool, Vec<AggregatedEdge3>);

        let mut sig_to_id: HashMap<Signature3, usize> = HashMap::new();
        let mut new_class = vec![0; n];
        let mut next_id = 0;
        let mut changes = 0;

        for u in 0..n {
            let mut aggr: BTreeMap<(usize, LLMTokenBV, usize), StateIDBV> = BTreeMap::new();
            for (p, bv_key, v_dense, sids) in &raw_edges[u] {
                let dest_class = prev_class[*v_dense];
                let key = (*p, bv_key.clone(), dest_class);
                aggr.entry(key).and_modify(|e| *e |= sids).or_insert_with(|| sids.clone());
            }
            let agg_edges: Vec<AggregatedEdge3> = aggr.into_iter().collect();

            let sig: Signature3 = (ends[u], agg_edges);

            let cid = *sig_to_id.entry(sig).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });

            new_class[u] = cid;
            if new_class[u] != prev_class[u] {
                changes += 1;
            }
        }

        crate::debug!(3, "Trie3 merge iter {}: classes={}, changes={}", it + 1, next_id, changes);
        prev_class = new_class;
        if changes == 0 { break; }
    }

    let final_partition = prev_class;
    let num_classes = final_partition.iter().max().map_or(0, |m| m + 1);

    let mut representatives: Vec<Option<Trie2Index>> = vec![None; num_classes];
    for (u_dense, &class_id) in final_partition.iter().enumerate() {
        if representatives[class_id].is_none() {
            representatives[class_id] = Some(old_of[u_dense]);
        }
    }

    let mut node_to_rep: HashMap<Trie2Index, Trie2Index> = HashMap::new();
    for (u_dense, &class_id) in final_partition.iter().enumerate() {
        node_to_rep.insert(old_of[u_dense], representatives[class_id].unwrap());
    }

    for class_id in 0..num_classes {
        if let Some(rep_idx) = representatives[class_id] {
            let u_dense = final_partition.iter().position(|&c| c == class_id).unwrap();

            let mut aggr: BTreeMap<(usize, LLMTokenBV, usize), StateIDBV> = BTreeMap::new();
            for (p, bv_key, v_dense, sids) in &raw_edges[u_dense] {
                let dest_class = final_partition[*v_dense];
                aggr.entry((*p, bv_key.clone(), dest_class)).and_modify(|e| *e |= sids).or_insert_with(|| sids.clone());
            }

            let mut new_children = BTreeMap::new();
            let mut new_live_tokens = LLMTokenBV::zeros();
            for ((p, bv_key, dest_class), sids) in aggr {
                if let Some(dest_rep_idx) = representatives[dest_class] {
                    new_children.entry((p, bv_key.clone())).or_insert_with(OrderedHashMap::new).insert(dest_rep_idx, sids);
                    new_live_tokens |= &bv_key;
                }
            }

            for (i, &c) in final_partition.iter().enumerate() {
                if c == class_id {
                    new_live_tokens |= &old_of[i].read(trie3_god).unwrap().value.live_tokens;
                }
            }

            let mut guard = rep_idx.write(trie3_god).unwrap();
            *guard.children_mut() = new_children;
            guard.value.live_tokens = new_live_tokens;
        }
    }

    for root_idx in roots.values_mut() {
        *root_idx = *node_to_rep.get(root_idx).unwrap();
    }

    let final_roots_vec: Vec<_> = roots.values().cloned().collect();
    Trie::recompute_all_max_depths(trie3_god, &final_roots_vec);
}

pub fn compress_trie3_edges(roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>, trie3_god: &Trie3GodWrapper) {
    crate::debug!(2, "Compressing Trie 3 by merging linear chains...");
    type EdgeKey3 = (usize, LLMTokenBV);

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let mut changed = true;
    let mut iterations = 0usize;

    while changed && iterations < 5 {
        iterations += 1;
        changed = false;
        let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);

        let mut incoming_count: HashMap<PrecomputeNode3Index, usize> = HashMap::new();
        for src_arc in &all_nodes {
            let guard = src_arc.read(trie3_god).expect("poison");
            for (_ek, dest_map) in guard.children() {
                for (node_ptr, _ev) in dest_map {
                    *incoming_count.entry(*node_ptr).or_insert(0) += 1;
                }
            }
        }

        'src_loop: for src_arc in &all_nodes {
            let children_snapshot: Vec<(EdgeKey3, Vec<(PrecomputeNode3Index, StateIDBV)>)> = {
                let g = src_arc.read(trie3_god).expect("poison");
                g.children().iter().map(|(ek, dest_map)| (ek.clone(), dest_map.iter().map(|(np, ev)| (*np, ev.clone())).collect())).collect()
            };

            for (ek1, entries) in children_snapshot {
                if entries.len() != 1 { continue; }
                let (child_ptr, sids1) = &entries[0];

                if incoming_count.get(child_ptr).cloned().unwrap_or(0) != 1 { continue; }

                let (is_end, child_outgoing): (bool, Vec<(EdgeKey3, Vec<(PrecomputeNode3Index, StateIDBV)>)>) = {
                    let cg = child_ptr.read(trie3_god).expect("poison");
                    (cg.value.end, cg.children().iter().map(|(ek, dm)| (ek.clone(), dm.iter().map(|(np, ev)| (*np, ev.clone())).collect())).collect())
                };
                if is_end { continue; }
                if child_outgoing.iter().map(|(_,dests)| dests.len()).sum::<usize>() != 1 { continue; }

                let (ek2, dests2) = &child_outgoing[0];
                let (grand_ptr, sids2) = &dests2[0];

                let mut merged_key: Option<EdgeKey3> = None;
                let mut merged_sids: Option<StateIDBV> = None;

                let is_all_sids1 = sids1 == &StateIDBV::max_ones();
                let is_all_bv1 = ek1.1 == LLMTokenBV::max_ones();
                let is_all_sids2 = sids2 == &StateIDBV::max_ones();
                let is_all_bv2 = ek2.1 == LLMTokenBV::max_ones();

                if is_all_sids1 && is_all_bv1 && is_all_sids2 && is_all_bv2 {
                    merged_key = Some((ek1.0 + ek2.0, LLMTokenBV::max_ones()));
                    merged_sids = Some(StateIDBV::max_ones());
                }
                else if ek2.0 == 0 {
                    let new_bv = &ek1.1 & &ek2.1;
                    if !new_bv.is_empty() {
                        merged_key = Some((ek1.0, new_bv));
                        merged_sids = Some(sids1 & sids2);
                    }
                }

                if let (Some(merged_key), Some(merged_sids)) = (merged_key, merged_sids) {
                    {
                        let mut src_w = src_arc.write(trie3_god).expect("poison");
                        if let Some(dest_map_for_ek1) = src_w.children_mut().get_mut(&ek1) {
                            dest_map_for_ek1.remove(child_ptr);
                            if dest_map_for_ek1.is_empty() {
                                src_w.children_mut().remove(&ek1);
                            }
                        }
                    }

                    {
                        let inserter = EdgeInserter::new(
                            trie3_god,
                            *src_arc,
                            merged_key.clone(),
                            merged_sids.clone(),
                            |e, n| *e |= n,
                            |node_value, _edge_value| {
                                node_value.live_tokens |= &merged_key.1;
                            },
                            |_, _| {},
                        );
                        let _ = inserter.try_destination(*grand_ptr).into_option();
                    }

                    changed = true;
                    continue 'src_loop;
                }
            }
        }

        if changed {
            prune_dead_paths_trie3(roots, trie3_god);
            merge_nodes_trie3(roots, trie3_god);
        }
    }
    crate::debug!(2, "Finished compressing Trie 3 in {} iteration(s).", iterations);
}
