use crate::finite_automata::Regex;
use hashbrown::HashMap;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub type EquivalenceResult = BTreeMap<Vec<usize>, Vec<usize>>;

fn hash_u64<T: Hash>(t: T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

fn compute_structural_hash(regex: &Regex, slice: &[u8], start_state: usize) -> u64 {
    todo!()
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
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

    groups.into_iter()
        .enumerate()
        .map(|(id, (_, idxs))| (vec![id], idxs))
        .collect()
}