use crate::finite_automata::Regex;
use hashbrown::HashMap;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct SimpleEquivalenceResult {
    pub mask_classes: BTreeMap<Vec<usize>, Vec<usize>>,
}

/// Simulates `tokenizer.execute_from_state`: walks the DFA on the slice.
/// Returns all matches found along the way and the final state (if valid).
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
                // Collect matches (Group ID, Width)
                for gid in regex.dfa.states[curr].finalizers.iter_indices() {
                    matches.push((gid, i + 1));
                }
            }
            None => return (matches, None), // Dead end
        }
    }
    (matches, Some(curr))
}

fn compute_signature(regex: &Regex, token: &[u8], initial_states: &[usize]) -> u64 {
    let mut hasher = DefaultHasher::new();

    for &start_node in initial_states {
        start_node.hash(&mut hasher);
        let mut queue = BTreeSet::from([0]);

        while let Some(offset) = queue.pop_first() {
            // 1. Setup State: Offset 0 uses context; others reset to start_state.
            let state_at_offset = if offset == 0 { start_node } else { regex.dfa.start_state };

            // 2. Execution Phase: Run the DFA on the remaining bytes.
            let (matches, end_state) = execute_from_state(regex, &token[offset..], state_at_offset);

            // 3. Match Processing Phase: Hash matches and schedule branches.
            for (group_id, width) in matches {
                (group_id, offset + width).hash(&mut hasher); // Hash match event

                // Add branch to queue (mimics `processing_queue` in commit_bytes)
                if offset + width < token.len() {
                    queue.insert(offset + width);
                }
            }

            // 4. End State Phase: Hash the final outcome if we survived.
            if let Some(final_state) = end_state {
                final_state.hash(&mut hasher);
                // Hash lookahead constraints (mimics valid next tokens)
                for future in &regex.dfa.states[final_state].possible_future_group_ids {
                    future.hash(&mut hasher);
                }
            } else {
                0xFF_u8.hash(&mut hasher); // Hash dead end
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
    let signatures: Vec<u64> = strings.par_iter()
        .map(|s| compute_signature(regex, s, initial_states))
        .collect();

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