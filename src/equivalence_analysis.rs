use crate::finite_automata::{GroupID, Regex};
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    // 1. Compute signatures in parallel
    let signatures: Vec<u64> = strings
        .par_iter()
        .map(|s| {
            let mut hasher = DefaultHasher::new();
            for &state in initial_states {
                compute_signature(regex, s, state).hash(&mut hasher);
            }
            hasher.finish()
        })
        .collect();

    // 2. Group indices by signature
    let mut groups: HashMap<u64, Vec<usize>> = HashMap::with_capacity(signatures.len());
    for (idx, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_default().push(idx);
    }

    groups.into_values().collect()
}

fn compute_signature(regex: &Regex, slice: &[u8], start_state: usize) -> u64 {
    // Graph maps: Position -> (StateHash, Edges)
    let mut graph = HashMap::new();
    let mut visited = HashSet::from([0]);
    let mut queue = VecDeque::from([0]);

    // 1. Forward Pass: Build the parsing graph (BFS)
    while let Some(pos) = queue.pop_front() {
        if pos > slice.len() { continue; }

        let exec_start = if pos == 0 { start_state } else { regex.dfa.start_state };
        let (state_hash, mut edges) = scan_transitions(regex, &slice[pos..], exec_start);

        // Convert relative offsets to absolute positions for the graph
        for edge in &mut edges {
            edge.1 += pos;
            if visited.insert(edge.1) {
                queue.push_back(edge.1);
            }
        }

        // Sort for deterministic hashing
        edges.sort_unstable_by_key(|e| e.0);
        graph.insert(pos, (state_hash, edges));
    }

    // 2. Backward Pass: Compute deterministic hashes
    // Sorting descending ensures we process children before parents
    let mut positions: Vec<_> = graph.keys().copied().collect();
    positions.sort_unstable_by(|a, b| b.cmp(a));

    let mut node_hashes = HashMap::with_capacity(graph.len());

    for pos in positions {
        let (state_hash, edges) = &graph[&pos];
        let mut hasher = DefaultHasher::new();

        // Hash the local state info
        state_hash.hash(&mut hasher);

        // Hash outgoing edges + the hash of the target node
        for (group_id, target_pos) in edges {
            let target_hash = node_hashes.get(target_pos).expect("Broken DAG order");
            (group_id, target_hash).hash(&mut hasher);
        }

        node_hashes.insert(pos, hasher.finish());
    }

    node_hashes.get(&0).copied().unwrap_or(0)
}

/// Simulates the DFA from a specific state and text slice.
/// Returns: (Hash of potential future groups, List of matches/edges)
#[inline(always)]
fn scan_transitions(
    regex: &Regex,
    text: &[u8],
    start_state: usize,
) -> (Option<u64>, Vec<(GroupID, usize)>) {
    let dfa = &regex.dfa;
    let mut current_state = start_state;

    // Initialize matches with 0-width for current state finalizers
    let mut matches: BTreeMap<GroupID, usize> = dfa.states[current_state]
        .finalizers
        .iter()
        .map(|&g| (g, 0))
        .collect();

    let mut done = dfa.states[current_state].transitions.is_empty();
    let mut position = 0;

    // Run the DFA
    if !done {
        while position < text.len() {
            let next_u8 = text[position];

            if let Some(&next_state) = dfa.states[current_state].transitions.get(next_u8) {
                current_state = next_state;
                position += 1;

                // Update matches
                for &group_id in &dfa.states[current_state].finalizers {
                    if dfa.non_greedy_finalizers.contains(&group_id) {
                        matches.entry(group_id).or_insert(position);
                    } else {
                        matches.insert(group_id, position);
                    }
                }

                // Check early termination: if all future potentials are already satisfied non-greedily
                let futures = &dfa.states[current_state].possible_future_group_ids;
                let should_terminate = !futures.is_empty() && futures.iter().all(|gid| {
                    dfa.non_greedy_finalizers.contains(gid) && matches.contains_key(gid)
                });

                if should_terminate {
                    done = true;
                    break;
                }
            } else {
                done = true;
                break;
            }
        }

        // If ran out of text but state has nowhere to go, mark done
        if !done && dfa.states[current_state].transitions.is_empty() {
            done = true;
        }
    }

    // Format output
    let edges = matches
        .into_iter()
        .filter(|&(_, width)| width != 0) // Filter 0-width matches
        .collect();

    let state_hash = if done {
        None
    } else {
        let mut h = DefaultHasher::new();
        dfa.states[current_state].possible_future_group_ids.hash(&mut h);
        Some(h.finish())
    };

    (state_hash, edges)
}