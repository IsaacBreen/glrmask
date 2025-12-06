use crate::finite_automata::Regex;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::collections::{BTreeSet, VecDeque};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

fn compute_hash(regex: &Regex, text: &[u8], start: usize) -> u64 {
    let mut visited: HashSet<usize> = HashSet::from([0]);
    let mut queue = VecDeque::from([0]);

    // 1. Discover all reachable positions (BFS)
    while let Some(pos) = queue.pop_front() {
        if pos > text.len() { continue; }
        let state = if pos == 0 { start } else { regex.dfa.start_state };

        for m in regex.execute_from_state_nonzero(&text[pos..], state).matches {
            if visited.insert(pos + m.position) {
                queue.push_back(pos + m.position);
            }
        }
    }

    // 2. Sort positions descending (leaves first) to hash bottom-up
    let mut nodes: Vec<_> = visited.into_iter().collect();
    nodes.sort_unstable_by(|a, b| b.cmp(a));

    // 3. Compute structural hashes
    let mut hashes = HashMap::with_capacity(nodes.len());
    for pos in nodes {
        let state = if pos == 0 { start } else { regex.dfa.start_state };
        let res = regex.execute_from_state_nonzero(&text[pos..], state);

        let mut edges: Vec<_> = res.matches.iter()
            .map(|m| (m.group_id, *hashes.get(&(pos + m.position)).unwrap_or(&0)))
            .collect();
        edges.sort_unstable(); // Ensure deterministic hashing

        let future = res.end_state.map(|id| &regex.dfa.states[id].possible_future_group_ids);

        let mut h = DefaultHasher::new();
        (future, edges).hash(&mut h);
        hashes.insert(pos, h.finish());
    }

    *hashes.get(&0).unwrap_or(&0)
}

pub fn find_equivalence_classes(regex: &Regex, strings: &[Vec<u8>], starts: &[usize]) -> EquivalenceResult {
    let mut map = HashMap::new();

    // Compute combined signature for every string in parallel
    let computed: Vec<_> = strings.par_iter().enumerate().map(|(i, s)| {
        let mut h = DefaultHasher::new();
        starts.iter().for_each(|&st| compute_hash(regex, s, st).hash(&mut h));
        (h.finish(), i)
    }).collect();

    // Group indices by signature
    for (sig, i) in computed {
        map.entry(sig).or_insert_with(Vec::new).push(i);
    }

    map.into_values().collect()
}