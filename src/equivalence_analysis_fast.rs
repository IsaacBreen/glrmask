use crate::finite_automata::Regex;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::collections::{BTreeSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

/// Compute the suffix hashes for positions > 0.
/// This is the same for all initial states since positions > 0 always start from dfa.start_state.
fn compute_suffix_hashes(regex: &Regex, slice: &[u8]) -> HashMap<usize, u64> {
    // Build the graph for all positions >= 1
    let mut graph: HashMap<usize, (Option<usize>, Vec<(usize, usize)>)> = HashMap::new();
    let mut queue = VecDeque::new();
    let mut visited: HashSet<usize> = HashSet::new();
    
    // Start from all positions 1..=slice.len()
    for pos in 1..=slice.len() {
        if visited.insert(pos) {
            queue.push_back(pos);
        }
    }
    
    while let Some(pos) = queue.pop_front() {
        if pos > slice.len() { continue; }
        
        let result = regex.execute_from_state_nonzero(&slice[pos..], regex.dfa.start_state);
        
        let mut edges: Vec<(usize, usize)> = result.matches.iter()
            .map(|m| (m.group_id, pos + m.position))
            .collect();
        edges.sort_unstable_by_key(|e| e.0);
        
        for (_, target) in &edges {
            if visited.insert(*target) {
                queue.push_back(*target);
            }
        }
        
        graph.insert(pos, (result.end_state, edges));
    }
    
    // Backward pass to compute hashes
    let mut positions: Vec<_> = graph.keys().copied().collect();
    positions.sort_unstable_by(|a, b| b.cmp(a));
    
    let mut node_hashes: HashMap<usize, u64> = HashMap::with_capacity(graph.len());
    
    for pos in positions {
        let (end_state, edges) = &graph[&pos];
        let mut hasher = DefaultHasher::new();
        
        // Hash completion info
        let completion = end_state.map(|id| &regex.dfa.states[id].possible_future_group_ids);
        completion.hash(&mut hasher);
        
        for (group_id, target) in edges {
            let target_hash = node_hashes.get(target).copied().unwrap_or(0);
            (group_id, target_hash).hash(&mut hasher);
        }
        
        node_hashes.insert(pos, hasher.finish());
    }
    
    node_hashes
}

/// Compute position-0 info for a given start_state.
/// Returns a hash representing (completion, edges with suffix hashes).
fn compute_pos0_hash(
    regex: &Regex,
    slice: &[u8],
    start_state: usize,
    suffix_hashes: &HashMap<usize, u64>
) -> u64 {
    let result = regex.execute_from_state_nonzero(slice, start_state);
    
    let completion = result.end_state.map(|id| &regex.dfa.states[id].possible_future_group_ids);
    
    let mut edges: Vec<(usize, usize)> = result.matches.iter()
        .map(|m| (m.group_id, m.position))
        .collect();
    edges.sort_unstable_by_key(|e| e.0);
    
    let mut hasher = DefaultHasher::new();
    completion.hash(&mut hasher);
    
    for (group_id, target) in &edges {
        let target_hash = suffix_hashes.get(target).copied().unwrap_or(0);
        (group_id, target_hash).hash(&mut hasher);
    }
    
    hasher.finish()
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    // Compute signatures in parallel
    let signatures: Vec<u64> = strings
        .par_iter()
        .map(|s| {
            // Precompute suffix hashes for this token (same for all initial states)
            let suffix_hashes = compute_suffix_hashes(regex, s);
            
            // Compute pos0 hash for each initial state (caching to avoid recomputation)
            let pos0_hashes: Vec<u64> = initial_states
                .iter()
                .map(|&state| compute_pos0_hash(regex, s, state, &suffix_hashes))
                .collect();
            
            // Combine all pos0 hashes into final signature
            let mut combined_hasher = DefaultHasher::new();
            for hash in pos0_hashes {
                hash.hash(&mut combined_hasher);
            }
            combined_hasher.finish()
        })
        .collect();

    // Group string indices by their computed signature
    let mut groups = HashMap::new();
    for (index, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_insert_with(Vec::new).push(index);
    }

    groups.into_values().collect()
}