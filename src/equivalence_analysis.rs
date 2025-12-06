use crate::finite_automata::Regex;
use hashbrown::HashMap;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

pub struct SimpleEquivalenceResult {
    pub mask_classes: BTreeMap<Vec<usize>, Vec<usize>>,
}

#[inline(always)]
fn mix(mut x: u128) -> u128 {
    x ^= x >> 33; x = x.wrapping_mul(0x9e3779b97f4a7c15_bf58476d1ce4e5b9);
    x ^= x >> 33; x = x.wrapping_mul(0x94d049bb133111eb_ff51afd7ed558ccd);
    x ^ (x >> 33)
}

fn compute_signature(regex: &Regex, token: &[u8], initial_states: &[usize]) -> u128 {
    let mut overall_hash: u128 = 0;

    // We must ensure the token behaves identically for EVERY possible start state
    // in the current context.
    for (idx, &start_node) in initial_states.iter().enumerate() {
        let mut path_hash: u128 = 0;

        // In `commit_bytes`, processing is handled via a queue sorted by offset.
        // We replicate this to ensure we hash side-effects in the same deterministic order.
        // `BTreeSet` automatically handles the sorting (pop_first) and deduplication.
        let mut pending_offsets = BTreeSet::new();
        pending_offsets.insert(0);

        while let Some(offset) = pending_offsets.pop_first() {
            // Logic from `commit_bytes`:
            // 1. Initial execution (offset 0) starts from the specific `tokenizer_s_id_at_offset`.
            // 2. All subsequent branches (triggered by matches) reset to `tokenizer.initial_state_id()`.
            let current_start_node = if offset == 0 {
                start_node
            } else {
                regex.dfa.start_state
            };

            let mut curr = current_start_node;
            let mut dead = false;

            // Mix in the offset so identical behaviors at different positions hash differently
            path_hash = mix(path_hash.wrapping_add(offset as u128).wrapping_add(1));

            // Simulate `tokenizer.execute_from_state`
            // We walk the remaining bytes. Matches found along the way trigger queue insertions.
            for i in offset..token.len() {
                let b = token[i];
                match regex.dfa.states[curr].transitions.get(b) {
                    Some(&next) => {
                        curr = next;

                        // Check for matches (Finalizers)
                        // In `commit_bytes`, this loops over `exec_result.matches`
                        if !regex.dfa.states[curr].finalizers.is_empty() {
                            for gid in regex.dfa.states[curr].finalizers.iter_indices() {
                                // 1. Record that a match occurred here (relevant for parser GSS)
                                path_hash = path_hash.wrapping_add(mix((gid as u128) << 16 | 2));

                                // 2. Simulate the branch: `new_offset = offset + match_info.width`
                                // Since we are walking byte-by-byte, the width is (i - offset + 1),
                                // so the new absolute offset is i + 1.
                                let next_offset = i + 1;

                                if next_offset < token.len() {
                                    // Branch continues within the token
                                    pending_offsets.insert(next_offset);
                                } else {
                                    // Branch hit the end of the token (Complete match)
                                    // In `commit_bytes`, this merges into `new_overall_state`.
                                    path_hash = path_hash.wrapping_add(mix(99999));
                                }
                            }
                        }
                    }
                    None => {
                        dead = true;
                        break;
                    }
                }
            }

            // Simulate `exec_result.end_state` logic.
            // If the tokenizer survived to the end of the slice, it contributes to the next state.
            if !dead {
                // Hash the resulting state ID
                path_hash = path_hash.wrapping_add(mix((curr as u128) << 32 | 3));

                // Hash the lookahead constraints (`tokens_accessible_from_state`)
                // This determines if this token is valid based on what MUST follow it.
                for &id in &regex.dfa.states[curr].possible_future_group_ids {
                    path_hash = path_hash.wrapping_add(mix((id as u128) | 4));
                }
            }
        }

        // Combine this start_state's path into the overall signature
        overall_hash = overall_hash.wrapping_add(mix(path_hash ^ (idx as u128)));
    }

    overall_hash
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> SimpleEquivalenceResult {
    crate::debug!(3, "Commit-logic equiv: {} strings, {} states", strings.len(), initial_states.len());
    let t0 = std::time::Instant::now();

    let mut groups = HashMap::new();

    // Compute signatures in parallel
    let signatures: Vec<u128> = strings.par_iter()
        .map(|s| compute_signature(regex, s, initial_states))
        .collect();

    // Group strings by signature
    for (i, h) in signatures.into_iter().enumerate() {
        groups.entry(h).or_insert_with(Vec::new).push(i);
    }

    let classes: BTreeMap<_, _> = groups.into_iter()
        .enumerate()
        .map(|(id, (_, v))| (vec![id], v)) // Arbitrary deterministic keys
        .collect();

    crate::debug!(3, "Commit-logic equiv done in {:?}", t0.elapsed());
    SimpleEquivalenceResult { mask_classes: classes }
}