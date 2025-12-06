use crate::finite_automata::Regex;
use hashbrown::HashMap;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct SimpleEquivalenceResult {
    pub mask_classes: BTreeMap<Vec<usize>, Vec<usize>>,
}

fn hash_u64<T: Hash>(t: T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

/// Simulates the tokenizer: walks the DFA on the slice.
/// Returns all matches found (group_id, width) and the final state if not dead.
fn execute_from_state(
    regex: &Regex,
    slice: &[u8],
    start_node: usize
) -> (Vec<(usize, usize)>, Option<usize>) {
    let mut curr = start_node;
    let mut matches = Vec::new();

    for (i, &b) in slice.iter().enumerate() {
        match regex.dfa.states[curr].transitions.get(b) {
            Some(&next) => {
                curr = next;
                for gid in regex.dfa.states[curr].finalizers.iter_indices() {
                    matches.push((gid, i + 1));
                }
            }
            None => return (matches, None),
        }
    }
    (matches, Some(curr))
}

/// Recursively sums the hashes of all execution branches.
fn compute_structural_hash(regex: &Regex, slice: &[u8], start_node: usize) -> u64 {
    let mut total_hash: u64 = 0;

    // 1. Run the DFA for this segment
    let (matches, end_state) = execute_from_state(regex, slice, start_node);

    // 2. Add branches commutatively (Order of parallel matches doesn't matter)
    for (gid, width) in matches {
        let branch_hash = hash_u64(gid).wrapping_add(
            compute_structural_hash(regex, &slice[width..], regex.dfa.start_state)
        );
        total_hash = total_hash.wrapping_add(branch_hash);
    }

    // 3. Add leaf state commutatively
    if let Some(final_state) = end_state {
        for future_id in &regex.dfa.states[final_state].possible_future_group_ids {
            total_hash = total_hash.wrapping_add(hash_u64(future_id));
        }
    } else {
        total_hash = total_hash.wrapping_add(0xDEAD_BEEF);
    }

    total_hash
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> SimpleEquivalenceResult {
    let signatures: Vec<u64> = strings.par_iter().map(|s| {
        let mut h: u64 = 0;
        for (i, &start) in initial_states.iter().enumerate() {
            // Mix index to separate contexts, add structurally
            h = h.wrapping_add(
                compute_structural_hash(regex, s, start).wrapping_mul(hash_u64(i) | 1)
            );
        }
        h
    }).collect();

    let mut groups = HashMap::new();
    for (idx, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_insert_with(Vec::new).push(idx);
    }

    SimpleEquivalenceResult {
        mask_classes: groups.into_iter()
            .enumerate()
            .map(|(id, (_, idxs))| (vec![id], idxs))
            .collect(),
    }
}