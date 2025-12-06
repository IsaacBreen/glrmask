use crate::finite_automata::Regex;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

pub type EquivalenceResult = std::collections::BTreeSet<Vec<usize>>;

fn compute_structural_hash(
    regex: &Regex,
    slice: &[u8],
    start_state: usize,
    hasher: &mut DefaultHasher,
) {
    // Optimization: Replicate the structural hash of the trellis without allocating
    // the heavy recursive Trellis/Arc/BTreeMap objects.
    
    // 1. Build a lightweight "flat" graph of the parse structure via BFS.
    // Map: Position -> (EndStateCompletion, Edges)
    // Edges: Vec<(GroupID, TargetPosition)>
    let mut nodes = HashMap::new();
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();

    queue.push_back(0);
    visited.insert(0);

    let dfa_start = regex.dfa.start_state;
    let dfa_states = &regex.dfa.states;

    while let Some(pos) = queue.pop_front() {
        if pos > slice.len() {
            continue;
        }

        let sub_slice = &slice[pos..];
        let exec_start = if pos == 0 { start_state } else { dfa_start };

        // Reuse existing regex execution logic to find matches and completion states
        let result = regex.execute_from_state_nonzero(sub_slice, exec_start);

        let mut edges = Vec::with_capacity(result.matches.len());
        for m in result.matches {
            let target = pos + m.position;
            edges.push((m.group_id, target));
            if visited.insert(target) {
                queue.push_back(target);
            }
        }

        // We only track the completion set (future group IDs) for the hash
        let completion = result.end_state.map(|sid| &dfa_states[sid].possible_future_group_ids);
        nodes.insert(pos, (completion, edges));
    }

    // 2. Compute hashes bottom-up (from the end of the string back to 0).
    // Sorting positions descending ensures we process targets before sources.
    let mut sorted_positions: Vec<usize> = nodes.keys().copied().collect();
    sorted_positions.sort_unstable_by(|a, b| b.cmp(a));

    let mut node_hashes: HashMap<usize, u64> = HashMap::with_capacity(nodes.len());

    for pos in sorted_positions {
        let (completion, edges) = nodes.get(&pos).unwrap();
        let mut h = DefaultHasher::new();

        // Hash Completion Set
        if let Some(comp) = completion {
            1u8.hash(&mut h);
            comp.hash(&mut h);
        } else {
            0u8.hash(&mut h);
        }

        // Hash Edges
        // Sort edges by GroupID to ensure the hash is deterministic (order-independent)
        let mut sorted_edges = edges.clone();
        sorted_edges.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        for (gid, target) in sorted_edges {
            gid.hash(&mut h);
            if let Some(target_hash) = node_hashes.get(&target) {
                target_hash.hash(&mut h);
            } else {
                // In a valid trellis (DAG), the target must be processed before the source 
                // because matches always advance position (execute_from_state_nonzero filters 0-width).
                panic!("Logic error: Target position {} not processed before source {}", target, pos);
            }
        }

        node_hashes.insert(pos, h.finish());
    }

    // Mix the root node's hash into the final hasher
    if let Some(root_hash) = node_hashes.get(&0) {
        root_hash.hash(hasher);
    } else {
        0u64.hash(hasher);
    }
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    // TEMP: disable
    return (0..strings.len()).map(|i| vec![i]).collect();
    let signatures: Vec<u64> = strings
        .par_iter()
        .enumerate()
        .map(|(i, s)| {
            if i > 0 && i % 1000 == 0 {
                // Optional: println!("Computing equivalence signatures: processing string {}/{}", i, strings.len());
            }
            let mut h = DefaultHasher::new();
            for &start in initial_states.iter() {
                compute_structural_hash(regex, s, start, &mut h);
            }
            h.finish()
        })
        .collect();

    let mut groups = HashMap::new();
    for (idx, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_insert_with(Vec::new).push(idx);
    }

    groups.into_values().collect()
}
