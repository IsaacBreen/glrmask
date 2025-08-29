use crate::constraint::{PrecomputeNode2, Trie2GodWrapper};
use crate::datastructures::gss::{LLMTokenBV, PrecomputedNodeContents};
use crate::datastructures::ordered_hash_map::Retain;
use crate::datastructures::trie::{EdgeInserter, Trie2};
use crate::datastructures::ArcPtrWrapper;
use crate::glr::table::StateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use crate::tokenizer::TokenizerStateID;
use deterministic_hash::DeterministicHasher;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use ordered_hash_map::OrderedHashMap;
use rand::prelude::IndexedRandom;
use rand::Rng;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

type NormalizedPath = Vec<(usize, StateID)>;
type PathMap = BTreeMap<NormalizedPath, LLMTokenBV>;

/// Samples a single normalized path by performing a random walk from the root.
fn sample_normalized_path(
    root: &Arc<RwLock<PrecomputeNode2>>,
    rng: &mut impl Rng,
    max_len: usize,
) -> Option<NormalizedPath> {
    let mut current_node = root.clone();
    let mut path = NormalizedPath::new();
    let mut current_k = 0;
    let mut bv = root.read().unwrap().value.live_tokens.clone();

    while path.len() < max_len {
        let can_terminate = current_node.read().unwrap().value.end;
        let can_continue = !current_node.read().unwrap().children().is_empty();

        if !can_continue {
            return if can_terminate { Some(path) } else { None };
        }

        if can_terminate && rng.gen_bool(0.2) { // 20% chance to terminate at an end node
            return Some(path);
        }

        let all_outgoing_edges: Vec<_> = current_node.read().unwrap()
            .children()
            .iter()
            .flat_map(|(ek, dest_map)| {
                dest_map.iter().map(move |(dest_ptr, edge_bv)| (ek.clone(), dest_ptr.clone(), edge_bv.clone()))
            })
            .collect();

        if all_outgoing_edges.is_empty() {
            return if current_node.read().unwrap().value.end { Some(path) } else { None };
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
    root: &Arc<RwLock<PrecomputeNode2>>,
    path: &NormalizedPath,
) -> LLMTokenBV {
    // State: (current_node, path_segment_index, accumulated_k, current_bv)
    let mut q: VecDeque<(Arc<RwLock<PrecomputeNode2>>, usize, usize, LLMTokenBV)> = VecDeque::new();
    let mut final_bv = LLMTokenBV::zeros();

    let initial_bv = root.read().unwrap().value.live_tokens.clone();
    q.push_back((root.clone(), 0, 0, initial_bv.clone()));

    // To handle cycles and redundant exploration
    let mut visited: HashMap<(*const RwLock<PrecomputeNode2>, usize, usize), LLMTokenBV> = HashMap::new();
    visited.insert((Arc::as_ptr(root), 0, 0), initial_bv);

    while let Some((node, path_idx, k_so_far, bv)) = q.pop_front() {
        // Check if we've completed the path
        if path_idx == path.len() {
            // We have successfully traversed the path. Now we need to reach an `end` node from here
            // with only `(k, None)` edges.
            let end_bv = find_end_bv_from_node_via_none_edges(node, bv);
            final_bv |= &end_bv;
            continue;
        }

        let (target_k, target_sid) = path[path_idx];

        // Explore children
        let guard = node.read().unwrap();
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
                        let visited_key = (Arc::as_ptr(&child_arc), path_idx + 1, 0);
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
                        let visited_key = (Arc::as_ptr(&child_arc), path_idx, new_k);
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
    start_node: Arc<RwLock<PrecomputeNode2>>,
    initial_bv: LLMTokenBV,
) -> LLMTokenBV {
    let mut end_bv = LLMTokenBV::zeros();
    let mut q = VecDeque::new();
    q.push_back((start_node, initial_bv));
    let mut visited: HashMap<*const RwLock<PrecomputeNode2>, LLMTokenBV> = HashMap::new();

    while let Some((node, bv)) = q.pop_front() {
        let guard = node.read().unwrap();
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
                    let child_ptr = Arc::as_ptr(&child_arc);
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
pub fn are_precompute2_trees_equivalent(a: &Arc<RwLock<PrecomputeNode2>>, b: &Arc<RwLock<PrecomputeNode2>>) -> bool {
    // Stochastic version
    if Arc::ptr_eq(a, b) { return true; }

    const NUM_SAMPLES: usize = 100;
    const MAX_PATH_LEN: usize = 32;
    let mut rng = rand::thread_rng();

    // Sample from A, check in B
    for i in 0..NUM_SAMPLES {
        if let Some(path) = sample_normalized_path(a, &mut rng, MAX_PATH_LEN) {
            let bv_a = get_bv_for_normalized_path(a, &path);
            if bv_a.is_empty() && i > 0 { continue; } // Skip trivial paths, but always check the empty path
            let bv_b = get_bv_for_normalized_path(b, &path);
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
        if let Some(path) = sample_normalized_path(b, &mut rng, MAX_PATH_LEN) {
            let bv_b = get_bv_for_normalized_path(b, &path);
            if bv_b.is_empty() && i > 0 { continue; } // Skip trivial paths, but always check the empty path
            let bv_a = get_bv_for_normalized_path(a, &path);
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

pub fn prune_dead_paths_trie2(roots: &mut BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode2>>>) {
    crate::debug!(2, "Pruning dead paths from precomputed trie 2.");

    // Use a worklist algorithm to propagate "liveness" backwards from end nodes.
    // This correctly handles cycles, iterating until a fixed point is reached.
    let all_nodes = Trie2::all_nodes(&roots.values().cloned().collect::<Vec<_>>());
    let mut predecessors: HashMap<*const RwLock<PrecomputeNode2>, Vec<(*const RwLock<PrecomputeNode2>, LLMTokenBV)>> = HashMap::new();
    let mut worklist = VecDeque::new();
    let mut live: HashMap<*const RwLock<PrecomputeNode2>, LLMTokenBV> = HashMap::new();

    // 1. Initialize live sets and build predecessor map.
    for node_arc in &all_nodes {
        let node_ptr = Arc::as_ptr(node_arc);
        live.insert(node_ptr, LLMTokenBV::zeros());

        let guard = node_arc.read().unwrap();
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
                let child_ptr = Arc::as_ptr(&child_arc);
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
        let mut guard = node_arc.write().unwrap();
        guard.children_mut().retain(|_edge_key, dest_map| {
            dest_map.retain(|child_wrapper, edge_value_bv| {
                let child_arc = child_wrapper.as_arc().clone();
                let child_ptr = Arc::as_ptr(&child_arc);
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
        let node_ptr = Arc::as_ptr(node_arc);
        guard.value.live_tokens = live.get(&node_ptr).unwrap().clone();
    }
    crate::debug!(2, "Finished pruning dead paths from trie 2.");
}

pub fn simplify_trie2_factor_common_destinations(roots: &mut BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode2>>>) {
    crate::debug!(2, "Simplifying trie 2 by factoring common destinations.");

    const MIN_INCOMING_EDGES_FOR_FACTORING: usize = 3;

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie2::all_nodes(&roots_vec);
    let arc_map: HashMap<_, _> = all_nodes.iter().map(|n| (Arc::as_ptr(n), n.clone())).collect();

    type EdgeKey2 = (usize, Option<StateID>);
    let mut incoming_map: HashMap<
        *const RwLock<PrecomputeNode2>,
        HashMap<
            EdgeKey2,
            Vec<(*const RwLock<PrecomputeNode2>, LLMTokenBV)>,
        >,
    > = HashMap::new();

    for src_arc in &all_nodes {
        let src_ptr = Arc::as_ptr(src_arc);
        let guard = src_arc.read().expect("poison");
        for (ek, dest_map) in guard.children() {
            for (dest_wrapper, bv) in dest_map {
                let dest_arc = dest_wrapper.as_arc().clone();
                let dest_ptr = Arc::as_ptr(&dest_arc);
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

                let intermediate_node = Arc::new(RwLock::new(PrecomputeNode2::new(
                    PrecomputedNodeContents::internal(),
                )));

                let mut union_bv = LLMTokenBV::zeros();
                for (_, bv) in &sources {
                    union_bv |= bv;
                }

                {
                    let mut intermediate_guard = intermediate_node.write().expect("poison");
                    let mut edge_val_opt = Some(union_bv.clone());
                    intermediate_guard.try_insert_unchecked(edge_key.clone(), &mut edge_val_opt, dest_arc.clone());
                    intermediate_guard.value.live_tokens |= &union_bv;
                }

                let identity_edge_key = (0, None);
                for (src_ptr, bv) in &sources {
                    let src_arc = arc_map.get(src_ptr).unwrap();
                    let mut src_guard = src_arc.write().expect("poison");

                    let mut remove_ek = false;
                    if let Some(dest_map_for_ek) = src_guard.children_mut().get_mut(&edge_key) {
                        let strong_key = ArcPtrWrapper::new(dest_arc.clone());
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
    roots: &mut BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode2>>>,
    god: Trie2GodWrapper,

) {
    crate::debug!(2, "Optimizing Trie2 2 size...");
    // Pin all nodes to prevent dangling weak pointers while we rewire.
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes_pinner = Trie2::all_nodes(&roots_vec);

    prune_dead_paths_trie2(roots);
    merge_nodes_trie2(roots);
    simplify_trie2_factor_common_destinations(roots);
    compress_trie2_edges(roots, &god);
    prune_dead_paths_trie2(roots);
    merge_nodes_trie2(roots);
    let final_roots: Vec<_> = roots.values().cloned().collect();
    Trie2::recompute_all_max_depths(&final_roots);
}

fn trie2_shape_hash(
    arc: &Arc<RwLock<PrecomputeNode2>>,
    memo: &mut HashMap<*const RwLock<PrecomputeNode2>, u64>,
) -> u64 {
    let ptr = Arc::as_ptr(arc);
    if let Some(&h) = memo.get(&ptr) {
        return h;
    }

    // Insert a placeholder to break cycles. A fixed value like 0 is fine.
    memo.insert(ptr, 0);

    let node_guard = arc.read().unwrap();
    let mut hasher = DeterministicHasher::new(std::collections::hash_map::DefaultHasher::new());

    // Hash shape-defining value fields
    node_guard.value.end.hash(&mut hasher);

    // Hash children structure
    let mut edge_hashes = Vec::new();
    for (ek, dest_map) in node_guard.children() {
        for (np, ev) in dest_map {
            let child = np.as_arc().clone();
            let child_h = trie2_shape_hash(&child, memo);
            let mut pair_hasher = DeterministicHasher::new(std::collections::hash_map::DefaultHasher::new());
            ek.hash(&mut pair_hasher);
            ev.hash(&mut pair_hasher);
            child_h.hash(&mut pair_hasher);
            edge_hashes.push(pair_hasher.finish());
        }
    }

    edge_hashes.sort_unstable();
    for h in edge_hashes {
        h.hash(&mut hasher);
    }

    let final_hash = hasher.finish();
    // Update the memo with the real hash.
    memo.insert(ptr, final_hash);
    final_hash
}

/// Cycle-safe, depth-bounded structural hash for Trie2 nodes.
/// This "skeleton" hash is intended only for bucketing candidates before
/// running exact structural equality. It never recurses indefinitely.
fn trie2_skeleton_hash(
    arc: &Arc<RwLock<PrecomputeNode2>>,
    memo: &mut HashMap<*const RwLock<PrecomputeNode2>, u64>,
) -> u64 {
    const MAX_DEPTH: usize = 64; // generous, but bounded
    fn inner(
        node: &Arc<RwLock<PrecomputeNode2>>,
        memo: &mut HashMap<*const RwLock<PrecomputeNode2>, u64>,
        visiting: &mut HashSet<*const RwLock<PrecomputeNode2>>,
        depth_left: usize,
    ) -> u64 {
        let ptr = Arc::as_ptr(node);
        if let Some(&h) = memo.get(&ptr) {
            return h;
        }
        if depth_left == 0 {
            // Depth cutoff: use stable mix of pointer and end flag (non-deterministic across runs is fine for bucketing).
            let guard = node.read().expect("poison");
            let mut h = DeterministicHasher::new(std::collections::hash_map::DefaultHasher::new());
            (ptr as usize).hash(&mut h);
            guard.value.end.hash(&mut h);
            let out = h.finish();
            memo.insert(ptr, out);
            return out;
        }
        if !visiting.insert(ptr) {
            // Cycle detected on current recursion path: fall back to pointer + end flag.
            let guard = node.read().expect("poison");
            let mut h = DeterministicHasher::new(std::collections::hash_map::DefaultHasher::new());
            (ptr as usize).hash(&mut h);
            guard.value.end.hash(&mut h);
            let out = h.finish();
            memo.insert(ptr, out);
            return out;
        }

        let guard = node.read().expect("poison");
        let mut edge_hashes = Vec::new();
        for (ek, dest_map) in guard.children() {
            for (np, _ev) in dest_map {
                let child = np.as_arc().clone();
                let child_h = inner(&child, memo, visiting, depth_left - 1);
                let mut pair_hasher = DeterministicHasher::new(std::collections::hash_map::DefaultHasher::new());
                // Only hash the "kind" of edge key: (k, sid_is_some) and whether strong/weak.
                let (k, sid_opt) = ek;
                k.hash(&mut pair_hasher);
                sid_opt.is_some().hash(&mut pair_hasher);
                child_h.hash(&mut pair_hasher);
                edge_hashes.push(pair_hasher.finish());
            }
        }
        drop(guard);

        edge_hashes.sort_unstable();
        let mut hasher = DeterministicHasher::new(std::collections::hash_map::DefaultHasher::new());
        {
            let guard2 = node.read().expect("poison");
            guard2.value.end.hash(&mut hasher);
        }
        for h in edge_hashes {
            h.hash(&mut hasher);
        }
        let out = hasher.finish();
        visiting.remove(&ptr);
        memo.insert(ptr, out);
        out
    }

    let mut visiting: HashSet<*const RwLock<PrecomputeNode2>> = HashSet::new();
    inner(arc, memo, &mut visiting, MAX_DEPTH)
}

fn trie2_shape_eq(
    a: &Arc<RwLock<PrecomputeNode2>>,
    b: &Arc<RwLock<PrecomputeNode2>>,
    cache: &mut HashMap<(*const RwLock<PrecomputeNode2>, *const RwLock<PrecomputeNode2>), bool>,
) -> bool {
    if Arc::ptr_eq(a, b) {
        return true;
    }

    let (p1, p2) = if Arc::as_ptr(a) < Arc::as_ptr(b) {
        (Arc::as_ptr(a), Arc::as_ptr(b))
    } else {
        (Arc::as_ptr(b), Arc::as_ptr(a))
    };

    if let Some(&res) = cache.get(&(p1, p2)) {
        return res;
    }

    cache.insert((p1, p2), true); // Optimistic insertion for cycles

    let guard_a = a.read().unwrap();
    let guard_b = b.read().unwrap();

    // Compare shape-defining value fields
    if guard_a.value.end != guard_b.value.end {
        cache.insert((p1, p2), false);
        return false;
    }

    // Compare children
    if guard_a.children().len() != guard_b.children().len() {
        cache.insert((p1, p2), false);
        return false;
    }

    for (ek, dest_map_a) in guard_a.children() {
        if let Some(dest_map_b) = guard_b.children().get(ek) {
            if dest_map_a.len() != dest_map_b.len() {
                cache.insert((p1, p2), false);
                return false;
            }

            let mut pairs_b: Vec<_> = dest_map_b.iter().map(|(np, ev)| (ev, np.as_arc().clone())).collect();

            for (np_a, ev_a) in dest_map_a.iter() {
                let arc_a = np_a.as_arc().clone();
                let mut found_match = false;
                for i in 0..pairs_b.len() {
                    let (ev_b, ref arc_b) = pairs_b[i];
                    if ev_a == ev_b {
                        if trie2_shape_eq(&arc_a, arc_b, cache) {
                            pairs_b.remove(i);
                            found_match = true;
                            break;
                        }
                    }
                }
                if !found_match {
                    cache.insert((p1, p2), false);
                    return false;
                }
            }
        } else {
            cache.insert((p1, p2), false);
            return false;
        }
    }

    true
}

pub fn merge_nodes_trie2(roots: &mut BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode2>>>) {
    crate::debug!(2, "Merging identical subtrees in precomputed trie 2.");

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie2::all_nodes(&roots_vec);

    let pb = ProgressBar::new(all_nodes.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta})")
            .expect("progress-bar"),
    );
    if !PROGRESS_BAR_ENABLED {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    let mut canonical_nodes: HashMap<u64, Vec<Arc<RwLock<PrecomputeNode2>>>> = HashMap::new();
    let mut visited: HashMap<*const RwLock<PrecomputeNode2>, Arc<RwLock<PrecomputeNode2>>> = HashMap::new();
    let mut shape_hash_memo: HashMap<*const RwLock<PrecomputeNode2>, u64> = HashMap::new();
    let mut shape_eq_cache: HashMap<(*const RwLock<PrecomputeNode2>, *const RwLock<PrecomputeNode2>), bool> = HashMap::new();

    let mut new_roots = BTreeMap::new();
    for (sid, root_arc) in roots.iter() {
        let canonical_root = deduplicate_recursive_trie2(
            root_arc.clone(),
            &mut canonical_nodes,
            &mut visited,
            &mut shape_hash_memo,
            &mut shape_eq_cache,
            &pb,
        );
        new_roots.insert(*sid, canonical_root);
    }
    *roots = new_roots;

    // Recompute depths after structural changes from merging
    let final_roots_vec: Vec<_> = roots.values().cloned().collect();
    Trie2::recompute_all_max_depths(&final_roots_vec);

    pb.finish_with_message("Finished merging Trie2 2 nodes");
    crate::debug!(2, "Finished merging subtrees in trie 2. Canonical nodes: {}", canonical_nodes.values().map(|v| v.len()).sum::<usize>());
}

fn deduplicate_recursive_trie2(
    node_arc: Arc<RwLock<PrecomputeNode2>>,
    canonical_nodes: &mut HashMap<u64, Vec<Arc<RwLock<PrecomputeNode2>>>>,
    visited: &mut HashMap<*const RwLock<PrecomputeNode2>, Arc<RwLock<PrecomputeNode2>>>,
    shape_hash_memo: &mut HashMap<*const RwLock<PrecomputeNode2>, u64>,
    shape_eq_cache: &mut HashMap<(*const RwLock<PrecomputeNode2>, *const RwLock<PrecomputeNode2>), bool>,
    pb: &ProgressBar,
) -> Arc<RwLock<PrecomputeNode2>> {
    let node_ptr = Arc::as_ptr(&node_arc);
    if let Some(cached_node) = visited.get(&node_ptr) {
        return cached_node.clone();
    }

    // Pre-emptively insert to break cycles.
    // We will update this later if we find a different canonical node.
    visited.insert(node_ptr, node_arc.clone());

    pb.inc(1);

    // Post-order: canonicalize children first
    let mut new_children_map = BTreeMap::new();
    let mut children_changed = false;

    {
        let node_guard = node_arc.read().unwrap();
        for (edge_key, dest_map) in node_guard.children() {
            let mut new_dest_map = OrderedHashMap::new();
            for (node_ptr_wrapper, edge_val) in dest_map.iter() {
                let child_arc = node_ptr_wrapper.as_arc().clone();
                let canonical_child_arc = deduplicate_recursive_trie2(
                    child_arc.clone(),
                    canonical_nodes,
                    visited,
                    shape_hash_memo,
                    shape_eq_cache,
                    pb,
                );
                if !Arc::ptr_eq(&child_arc, &canonical_child_arc) {
                    children_changed = true;
                }
                let new_node_ptr_wrapper = ArcPtrWrapper::new(canonical_child_arc);
                new_dest_map.insert(new_node_ptr_wrapper, edge_val.clone());
            }
            if !new_dest_map.is_empty() {
                new_children_map.insert(edge_key.clone(), new_dest_map);
            }
        }
    }

    if children_changed {
        let mut node_guard = node_arc.write().unwrap();
        *node_guard.children_mut() = new_children_map;
        // max_depth will be recomputed globally at the end
    }

    // Now find a canonical representative for the current node using cycle-safe skeleton hash
    let fp = trie2_skeleton_hash(&node_arc, shape_hash_memo);
    let bucket = canonical_nodes.entry(fp).or_default();

    for candidate_arc in bucket.iter() {
        if trie2_shape_eq(&node_arc, candidate_arc, shape_eq_cache) {
            // Found a match. Merge live_tokens and return the canonical version.
            let node_live_tokens = { node_arc.read().unwrap().value.live_tokens.clone() };
            if !node_live_tokens.is_empty() {
                let mut candidate_guard = candidate_arc.write().unwrap();
                candidate_guard.value.live_tokens |= node_live_tokens;
            }
            // Update visited map with the true canonical node.
            visited.insert(node_ptr, candidate_arc.clone());
            return candidate_arc.clone();
        }
    }

    // No match found. This node becomes a new canonical representative.
    bucket.push(node_arc.clone());
    // The visited map already contains (node_ptr, node_arc), which is correct in this case.
    node_arc
}

/// Compress linear chains in Trie2 by merging consecutive edges where safe.
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
    roots: &mut BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode2>>>,
    god: &Trie2GodWrapper,
) {
    crate::debug!(2, "Compressing Trie2 2 by merging linear chains...");
    type EdgeKey2 = (usize, Option<StateID>);

    // Helper to count incoming edges for each node (both strong and weak).
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let mut changed = true;
    let mut iterations = 0usize;
    let _all_nodes = Trie2::all_nodes(&roots_vec);

    while changed {
        iterations += 1;
        changed = false;
        let all_nodes = Trie2::all_nodes(&roots_vec);
        let mut arc_map: HashMap<*const RwLock<PrecomputeNode2>, Arc<RwLock<PrecomputeNode2>>> = HashMap::new();
        for n in &all_nodes {
            arc_map.insert(Arc::as_ptr(n), n.clone());
        }

        // Build incoming counts
        let mut incoming_count: HashMap<*const RwLock<PrecomputeNode2>, usize> = HashMap::new();
        for src_arc in &all_nodes {
            let guard = src_arc.read().expect("poison");
            for (_ek, dest_map) in guard.children() {
                for (node_ptr, _ev) in dest_map {
                    let child_arc = node_ptr.as_arc().clone();
                    let ptr = Arc::as_ptr(&child_arc);
                    *incoming_count.entry(ptr).or_insert(0) += 1;
                }
            }
        }

        // Try to compress from each source node
        'src_loop: for src_arc in &all_nodes {
            // Snapshot children
            let children_snapshot: Vec<(EdgeKey2, Vec<(ArcPtrWrapper<RwLock<PrecomputeNode2>>, LLMTokenBV)>)> = {
                let g = src_arc.read().expect("poison");
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
                    let child_ptr_raw = Arc::as_ptr(&child_arc);

                    // Preconditions: B has in-degree 1, not end, and exactly one outgoing edge overall.
                    if incoming_count.get(&child_ptr_raw).cloned().unwrap_or(0) != 1 {
                        continue;
                    }
                    let (is_end, child_outgoing): (bool, Vec<(EdgeKey2, Vec<(ArcPtrWrapper<RwLock<PrecomputeNode2>>, LLMTokenBV)>)>) = {
                        let cg = child_arc.read().expect("poison");
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
                        let mut src_w = src_arc.write().expect("poison");
                        // 1) Reduce/remove src --ek1--> child by merged_bv
                        if let Some(dest_map_for_ek1) = src_w.children_mut().get_mut(&ek1) {
                            let child_key_in_map = ArcPtrWrapper::new(child_arc.clone());
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
                            god,
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
            prune_dead_paths_trie2(roots);
            merge_nodes_trie2(roots);
        }
    }
    crate::debug!(2, "Finished compressing Trie2 2 in {} iteration(s).", iterations);
}

pub fn clone_trie2_graph(
    root: &Arc<RwLock<PrecomputeNode2>>,
) -> (
    Arc<RwLock<PrecomputeNode2>>,
    HashMap<*const RwLock<PrecomputeNode2>, Arc<RwLock<PrecomputeNode2>>>,
) {
    // old_ptr -> new arc
    let mut map: HashMap<*const RwLock<PrecomputeNode2>, Arc<RwLock<PrecomputeNode2>>> = HashMap::new();
    let mut q: VecDeque<Arc<RwLock<PrecomputeNode2>>> = VecDeque::new();

    let root_ptr = Arc::as_ptr(root);
    let root_value = { root.read().expect("poison").value.clone() };
    let new_root = Arc::new(RwLock::new(PrecomputeNode2::new(root_value)));
    map.insert(root_ptr, new_root.clone());
    q.push_back(root.clone());

    while let Some(old_arc) = q.pop_front() {
        let old_ptr = Arc::as_ptr(&old_arc);
        let new_arc = map.get(&old_ptr).expect("parent must be created").clone();

        // Snapshot children outside of lock to avoid recursive lock explosion.
        let children_snapshot: Vec<( (usize, Option<StateID>), Vec<(ArcPtrWrapper<RwLock<PrecomputeNode2>>, LLMTokenBV)> )> = {
            let g = old_arc.read().expect("poison");
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
                let child_ptr_old = Arc::as_ptr(&child_arc_old);
                if !map.contains_key(&child_ptr_old) {
                    let child_value = { child_arc_old.read().expect("poison").value.clone() };
                    let child_arc_new = Arc::new(RwLock::new(PrecomputeNode2::new(child_value)));
                    map.insert(child_ptr_old, child_arc_new);
                    q.push_back(child_arc_old);
                }
            }
        }

        // Now wire edges on new_arc
        {
            let mut new_g = new_arc.write().expect("poison");
            for (ek, entries) in children_snapshot {
                let dest_map = new_g.children_mut().entry(ek).or_default();
                for (old_node_ptr, ev) in entries {
                    let child_arc_old = old_node_ptr.as_arc().clone();
                    let child_ptr_old = Arc::as_ptr(&child_arc_old);
                    let child_arc_new = map.get(&child_ptr_old).expect("must exist").clone(); // With weak refs removed, all edges are strong.
                    let new_key = ArcPtrWrapper::new(child_arc_new);
                    dest_map.insert(new_key, ev);
                }
            }
        }
    }

    // Recompute max_depths in the clone to keep invariants consistent.
    Trie2::recompute_all_max_depths(&[new_root.clone()]);
    (new_root, map)
}