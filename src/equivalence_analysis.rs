use crate::finite_automata::Regex;
use hashbrown::HashMap;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct SimpleEquivalenceResult {
    pub mask_classes: BTreeMap<Vec<usize>, Vec<usize>>,
}

fn compute_signature(regex: &Regex, token: &[u8], initial_states: &[usize]) -> u64 {
    let mut hasher = DefaultHasher::new();

    for &start_node in initial_states {
        start_node.hash(&mut hasher);

        let mut pending_offsets = BTreeSet::from([0]);

        while let Some(offset) = pending_offsets.pop_first() {
            let mut curr = if offset == 0 { start_node } else { regex.dfa.start_state };
            let mut survived = true;

            for (i, &byte) in token.iter().enumerate().skip(offset) {
                match regex.dfa.states[curr].transitions.get(byte) {
                    Some(&next) => {
                        curr = next;
                        for group_id in regex.dfa.states[curr].finalizers.iter_indices() {
                            // Record intermediate match
                            (group_id, i).hash(&mut hasher);

                            // Schedule branch reset
                            if i + 1 < token.len() {
                                pending_offsets.insert(i + 1);
                            }
                        }
                    }
                    None => {
                        survived = false;
                        break;
                    }
                }
            }

            if survived {
                curr.hash(&mut hasher);
                for future_id in &regex.dfa.states[curr].possible_future_group_ids {
                    future_id.hash(&mut hasher);
                }
            } else {
                // Hash failure state to distinguish from survival
                0xFF_u8.hash(&mut hasher);
            }
        }
    }

    hasher.finish()
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> SimpleEquivalenceResult {
    crate::debug!(3, "Equiv Check: {} strings, {} states", strings.len(), initial_states.len());

    let signatures: Vec<u64> = strings.par_iter()
        .map(|s| compute_signature(regex, s, initial_states))
        .collect();

    let mut groups = HashMap::new();
    for (idx, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_insert_with(Vec::new).push(idx);
    }

    let mask_classes: BTreeMap<_, _> = groups.into_iter()
        .enumerate()
        .map(|(id, (_, indices))| (vec![id], indices))
        .collect();

    crate::debug!(4, "Equiv Check: Found {} equivalence classes", mask_classes.len());

    SimpleEquivalenceResult { mask_classes }
}