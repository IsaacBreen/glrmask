use crate::constraint::{LLMTokenBV, PrecomputeNode2, PrecomputeNode2Index, Trie2GodWrapper};
use crate::datastructures::trie::{Trie, Trie2Index};
use crate::glr::table::StateID;
use crate::tokenizer::TokenizerStateID;
use rand::prelude::IndexedRandom;
use rand::Rng;
use std::collections::{BTreeMap, HashMap, VecDeque};

#[derive(Debug, Clone)]
pub struct Trie2Config {
    pub enabled: bool,
}

impl Default for Trie2Config {
    fn default() -> Self {
        Self {
            enabled: true,
        }
    }
}

impl Trie2Config {
    pub fn off() -> Self {
        Self {
            enabled: false,
        }
    }
}

type NormalizedPath = Vec<(usize, StateID)>;

/// Samples a single normalized path by performing a random walk from the root.
fn sample_normalized_path(
    root: &Trie2Index,
    rng: &mut impl Rng,
    max_len: usize,
    trie2_god: &Trie2GodWrapper,
) -> Option<NormalizedPath> {
    let mut current_node = root.clone();
    let mut path = NormalizedPath::new();
    let mut current_k: usize = 0;
    let mut bv = root.read(trie2_god).value.live_tokens.clone();

    while path.len() < max_len {
        let can_terminate = current_node.read(trie2_god).value.end;
        let can_continue = !current_node.read(trie2_god).children().is_empty();

        if !can_continue {
            return if can_terminate { Some(path) } else { None };
        }

        if can_terminate && rng.gen_bool(0.2) { // 20% chance to terminate at an end node
            return Some(path);
        }

        let all_outgoing_edges: Vec<_> = current_node.read(trie2_god)
            .children()
            .iter()
            .flat_map(|(ek, dest_map)| {
                dest_map.iter().map(move |(dest_ptr, edge_bv)| (ek.clone(), dest_ptr.clone(), edge_bv.clone()))
            })
            .collect();

        if all_outgoing_edges.is_empty() {
            return if current_node.read(trie2_god).value.end { Some(path) } else { None };
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

    let initial_bv = root.read(trie2_god).value.live_tokens.clone();
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
        let guard = node.read(trie2_god);
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
        let guard = node.read(trie2_god);
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

pub fn optimize_trie2_size(
    _roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode2Index>,
    _trie2_god: &Trie2GodWrapper,
    _config: &Trie2Config,
) {
    // All optimizations have been removed.
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
    let root_value = { root.read(trie2_god).value.clone() };
    let new_root = PrecomputeNode2Index::new(trie2_god.insert(PrecomputeNode2::new(root_value)));
    map.insert(root_ptr, new_root.clone());
    q.push_back(root.clone());

    while let Some(old_arc) = q.pop_front() {
        let old_ptr = old_arc;
        let new_arc = *map.get(&old_ptr).expect("parent must be created");

        // Snapshot children outside of lock to avoid recursive lock explosion.
        let children_snapshot: Vec<( (usize, Option<StateID>), Vec<(PrecomputeNode2Index, LLMTokenBV)> )> = {
            let g = old_arc.read(trie2_god);
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
                    let child_value = { child_arc_old.read(trie2_god).value.clone() };
                    let child_arc_new = PrecomputeNode2Index::new(trie2_god.insert(PrecomputeNode2::new(child_value)));
                    map.insert(child_ptr_old, child_arc_new);
                    q.push_back(child_arc_old);
                }
            }
        }

        // Now wire edges on new_arc
        {
            let mut new_g = new_arc.write(trie2_god);
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
