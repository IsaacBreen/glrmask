use crate::finite_automata::Regex;
use hashbrown::HashMap;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub type EquivalenceResult = Vec<Vec<usize>>;

fn compute_structural_hash(regex: &Regex, slice: &[u8], start_state: usize, hasher: &mut DefaultHasher) {
    let trellis = regex.generate_token_trellis(slice, start_state);
    // Trellis is hashable, so just hash it!
    trellis.hash(hasher);
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    let signatures: Vec<u64> = strings.par_iter().map(|s| {
        let mut h = DefaultHasher::new();
        for &start in initial_states.iter() {
            compute_structural_hash(regex, s, start, &mut h);
        }
        h.finish()
    }).collect();

    let mut groups = HashMap::new();
    for (idx, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_insert_with(Vec::new).push(idx);
    }

    groups.into_values().collect()
}