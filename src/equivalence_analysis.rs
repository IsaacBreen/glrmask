use crate::finite_automata::{ExecutionResult, GroupID, Match, Regex};
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

// A high-performance, inlined mixing function (Variant of FNV/WyHash logic)
// drastically faster than SipHash for integer/graph mixing.
#[inline(always)]
fn mix(hash: u64, data: u64) -> u64 {
    (hash ^ data).wrapping_mul(0x517cc1b727220a95).rotate_left(13)
}

#[inline(always)]
fn fold_slice<T: Hash>(mut hash: u64, slice: &BTreeSet<T>) -> u64 {
    // We use DefaultHasher here just to process the distinct types (like BTreeSet/Vec)
    // cleanly into a u64, but we only instantiate it once per static structure.
    let mut hasher = DefaultHasher::new();
    slice.hash(&mut hasher);
    mix(hash, hasher.finish())
}

/// Computes a deterministic hash representing the parsing structure of the string.
/// heavily optimized for allocation reuse and cache locality.
fn compute_signature(regex: &Regex, slice: &[u8], start_state: usize) -> u64 {
    let dfa = &regex.dfa;
    let text_len = slice.len();

    // -- Pre-computation Phase --
    // Determine bounds for flat arrays to avoid resizing checks in the hot loop
    let max_group_id = dfa.states.iter()
        .flat_map(|s| s.finalizers.iter())
        .max()
        .unwrap_or(0);

    let mut non_greedy_mask = vec![false; max_group_id + 1];
    for &gid in &dfa.non_greedy_finalizers {
        if gid <= max_group_id {
            non_greedy_mask[gid] = true;
        }
    }

    // -- Working Memory (Reused) --
    // Map: GroupID -> match length (0 means unset)
    // Using 0 as unset is safe because match length is position + local_position + 1
    // (we store 1-based index to distinguish match at 0 from no match).
    let mut matches_sparse = vec![0usize; max_group_id + 1];
    let mut matches_dense = Vec::with_capacity(16); // Keeps track of which IDs are set

    // Graph storage: Position -> (NodeHash, Edges)
    // We store the "static" hash of the node immediately.
    let mut graph_data: HashMap<usize, (u64, Vec<(GroupID, usize)>)> = HashMap::new();

    // BFS State
    let mut queue = VecDeque::new();
    queue.push_back(0);
    // Use FxHashSet implicitly via hashbrown for visited
    let mut visited: HashSet<usize> = HashSet::new();
    visited.insert(0);

    // -- Forward Pass: BFS Simulation --
    while let Some(pos) = queue.pop_front() {
        if pos > text_len {
            continue;
        }

        let exec_start = if pos == 0 { start_state } else { dfa.start_state };
        let text = &slice[pos..];

        // --- Inlined & Optimized Execution Logic ---
        let mut current_state = exec_start;
        let mut done = dfa.states[exec_start].transitions.is_empty();

        // Clear working memory
        for &id in &matches_dense {
            matches_sparse[id] = 0;
        }
        matches_dense.clear();

        // Initial state finalizers
        for group_id in &dfa.states[exec_start].finalizers {
            if group_id <= max_group_id {
                // Initialize match at relative 0. Stored as 0 + 1.
                matches_sparse[group_id] = 1;
                matches_dense.push(group_id);
            }
        }

        let mut current_pos_offset = 0;
        let mut result_end_state = if done { None } else { Some(current_state) };

        if !done {
            let mut local_position = 0;
            let limit = text.len();

            while local_position < limit {
                let state_data = &dfa.states[current_state];
                let next_u8 = text[local_position];

                // Fast transition lookup
                // Assuming transitions is HashMap/BTreeMap. If it's a sparse Vec, this is fast.
                if let Some(&next_state) = state_data.transitions.get(next_u8) {
                    current_state = next_state;
                    local_position += 1;

                    let match_val = pos + local_position + 1; // 1-based logic

                    // Process finalizers
                    for group_id in &dfa.states[current_state].finalizers {
                        if group_id > max_group_id { continue; }

                        let is_set = matches_sparse[group_id] != 0;
                        if !is_set {
                            matches_sparse[group_id] = match_val;
                            matches_dense.push(group_id);
                        } else if !non_greedy_mask[group_id] {
                            // Greedy: overwrite existing
                            matches_sparse[group_id] = match_val;
                        }
                        // If non-greedy and set, keep existing (first match wins)
                    }

                    // Optimized Termination Check
                    // Original: all possible futures are (non-greedy AND matched)
                    let futures = &dfa.states[current_state].possible_future_group_ids;
                    let should_terminate = !futures.is_empty() && futures.iter().all(|&gid| {
                         gid <= max_group_id && non_greedy_mask[gid] && matches_sparse[gid] != 0
                    });

                    if should_terminate {
                        current_pos_offset += text.len(); // "position += text.len()"
                        done = true;
                        result_end_state = None; // Terminated early
                        break;
                    }
                } else {
                    current_pos_offset += text.len();
                    done = true;
                    result_end_state = None; // No transition
                    break;
                }
            }
            if !done {
                current_pos_offset += text.len();
                if dfa.states[current_state].transitions.is_empty() {
                    result_end_state = None; // Reached dead end naturally
                } else {
                    result_end_state = Some(current_state);
                }
            }
        }

        // --- Collect Edges ---
        // Iterate dense list, sort by ID to ensure deterministic hash order
        matches_dense.sort_unstable();

        let mut edges = Vec::with_capacity(matches_dense.len());
        for &gid in &matches_dense {
            let len = matches_sparse[gid];
            if len > 1 { // Filter len != 0 (unset) and len != 1 (empty match at start - handled by logic?)
                // Note: Original logic `filter(|token| token.position != 0)`.
                // Our stored `len` is `real_pos + 1`. So `real_pos` != 0 means `len` > 1.
                let target = (len - 1); // Recover absolute position
                edges.push((gid, target));

                if visited.insert(target) {
                    queue.push_back(target);
                }
            }
        }

        // --- Compute Static Node Hash ---
        // Hash the "completion" data (possible future groups of end state)
        let mut node_static_hash = 0u64;
        if let Some(end_id) = result_end_state {
            let possible_futures = &dfa.states[end_id].possible_future_group_ids;
            // Using DefaultHasher for the set to ensure strict compatibility with complex keys if needed,
            // but mixing the result efficiently.
            node_static_hash = fold_slice(node_static_hash, possible_futures);
        }

        graph_data.insert(pos, (node_static_hash, edges));
    }

    // -- Backward Pass: Compute Final Signatures --
    // Sort positions descending
    let mut positions: Vec<_> = graph_data.keys().copied().collect();
    positions.sort_unstable_by(|a, b| b.cmp(a));

    // Map: Position -> Computed Hash
    // We reuse the capacity from graph_data which is roughly the same size
    let mut node_hashes = HashMap::with_capacity(graph_data.len());

    for pos in positions {
        let (static_hash, edges) = graph_data.get(&pos).unwrap();

        let mut final_hash = *static_hash;

        for (group_id, target) in edges {
            let target_hash = *node_hashes.get(target).unwrap_or(&0); // Should always exist due to sort

            // Mix: Group ID then Target Hash
            final_hash = mix(final_hash, *group_id as u64);
            final_hash = mix(final_hash, target_hash);
        }

        node_hashes.insert(pos, final_hash);
    }

    *node_hashes.get(&0).unwrap_or(&0)
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    // Parallel computation of signatures
    let signatures: Vec<u64> = strings
        .par_iter()
        .map(|s| {
            // Mix the signatures of multiple start states into one hash
            let mut combined_hash = 0u64;
            for &state in initial_states {
                let sig = compute_signature(regex, s, state);
                combined_hash = mix(combined_hash, sig);
            }
            combined_hash
        })
        .collect();

    // Grouping
    let mut groups = HashMap::new();
    for (index, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_insert_with(Vec::new).push(index);
    }

    groups.into_values().collect()
}