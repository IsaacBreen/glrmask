//! Max-length bounded state equivalence prepass.
//!
//! Two states are equivalent up to length `k` iff for every byte string of
//! length `0..=k`, either:
//! - that path exists from both states and ends in states with identical
//!   `finalizers` and identical `possible_future_group_ids`, or
//! - that path exists from neither state.
//!
//! This pass must be conservative: it may keep distinguishable states separate,
//! but it must never merge states that differ under the bounded view. To keep
//! the partition exact, we intern structural keys directly instead of relying on
//! 64-bit hashes.

use std::collections::HashMap;
use std::hash::Hash;

use rayon::prelude::*;

use super::super::compat::TokenizerView;

fn build_classes_from_keys<K>(keys: impl IntoIterator<Item = K>, expected_len: usize) -> (Vec<usize>, usize)
where
    K: Eq + Hash,
{
    let mut class_of_key: HashMap<K, usize> = HashMap::with_capacity(expected_len);
    let mut classes = Vec::with_capacity(expected_len);
    let mut next_class = 0usize;

    for key in keys {
        let class_id = match class_of_key.get(&key) {
            Some(&existing) => existing,
            None => {
                let new_id = next_class;
                class_of_key.insert(key, new_id);
                next_class += 1;
                new_id
            }
        };
        classes.push(class_id);
    }

    (classes, next_class)
}

#[inline(always)]
fn build_subset_mapping(states: &[usize], classes: &[usize]) -> Vec<usize> {
    let mut rep_for_class: HashMap<usize, usize> = HashMap::new();
    let mut mapping = Vec::with_capacity(states.len());

    for &state_id in states {
        let rep = *rep_for_class.entry(classes[state_id]).or_insert(state_id);
        mapping.push(rep);
    }

    mapping
}

fn build_label_classes(tokenizer: &TokenizerView) -> (Vec<usize>, usize) {
    let dfa = tokenizer.dfa();
    build_classes_from_keys(
        dfa.states
            .iter()
            .map(|state| (state.finalizers.clone(), state.possible_future_group_ids.clone())),
        dfa.states.len(),
    )
}

fn find_state_equivalence_classes_kstep(
    tokenizer: &TokenizerView,
    states: &[usize],
    k: usize,
) -> Vec<usize> {
    if states.is_empty() {
        return Vec::new();
    }

    let dfa = tokenizer.dfa();
    let num_states = dfa.states.len();

    if num_states == 0 {
        return states.to_vec();
    }

    let transitions: Vec<Vec<(u8, usize)>> = dfa
        .states
        .iter()
        .map(|state| {
            state
                .transitions
                .iter()
                .enumerate()
                .filter(|(_, target)| **target != u32::MAX)
                .map(|(byte, &target)| (byte as u8, target as usize))
                .collect()
        })
        .collect();

    let mut class_reps: Vec<usize> = Vec::new();
    let class_for_state: Vec<usize>;
    let mut use_trans_class_cache = false;

    {
        let mut trans_pattern_to_class: HashMap<Vec<(u8, usize)>, usize> = HashMap::new();
        let mut tmp_class_for_state = vec![0usize; num_states];

        for (state_id, trans) in transitions.iter().enumerate() {
            let class_id = match trans_pattern_to_class.get(trans) {
                Some(&existing) => existing,
                None => {
                    let new_id = class_reps.len();
                    trans_pattern_to_class.insert(trans.clone(), new_id);
                    class_reps.push(state_id);
                    new_id
                }
            };
            tmp_class_for_state[state_id] = class_id;
        }

        let num_classes = class_reps.len();
        let shared = num_states.saturating_sub(num_classes);

        if num_states > 0 && shared * 2 >= num_states {
            use_trans_class_cache = true;
            class_for_state = tmp_class_for_state;
        } else {
            class_reps.clear();
            class_for_state = Vec::new();
        }
    }

    let (label_classes, _) = build_label_classes(tokenizer);
    let (mut prev_classes, mut prev_num_classes) = build_label_classes(tokenizer);

    if k == 0 {
        return build_subset_mapping(states, &prev_classes);
    }

    for _depth in 0..k {
        let transition_signature_ids = if use_trans_class_cache {
            let class_signatures: Vec<Vec<(u8, usize)>> = (0..class_reps.len())
                .into_par_iter()
                .map(|class_id| {
                    transitions[class_reps[class_id]]
                        .iter()
                        .map(|(byte, target)| (*byte, prev_classes[*target]))
                        .collect()
                })
                .collect();

            let mut signature_to_id: HashMap<Vec<(u8, usize)>, usize> =
                HashMap::with_capacity(class_signatures.len());
            let mut next_signature_id = 0usize;
            let mut signature_ids_for_class = vec![0usize; class_signatures.len()];

            for (class_id, signature) in class_signatures.into_iter().enumerate() {
                let signature_id = match signature_to_id.get(&signature) {
                    Some(&existing) => existing,
                    None => {
                        let new_id = next_signature_id;
                        signature_to_id.insert(signature, new_id);
                        next_signature_id += 1;
                        new_id
                    }
                };
                signature_ids_for_class[class_id] = signature_id;
            }

            (0..num_states)
                .map(|state_id| signature_ids_for_class[class_for_state[state_id]])
                .collect::<Vec<_>>()
        } else {
            let state_signatures: Vec<Vec<(u8, usize)>> = transitions
                .par_iter()
                .map(|state_transitions| {
                    state_transitions
                        .iter()
                        .map(|(byte, target)| (*byte, prev_classes[*target]))
                        .collect()
                })
                .collect();

            let mut signature_to_id: HashMap<Vec<(u8, usize)>, usize> =
                HashMap::with_capacity(state_signatures.len());
            let mut next_signature_id = 0usize;
            let mut signature_ids = vec![0usize; num_states];

            for (state_id, signature) in state_signatures.into_iter().enumerate() {
                let signature_id = match signature_to_id.get(&signature) {
                    Some(&existing) => existing,
                    None => {
                        let new_id = next_signature_id;
                        signature_to_id.insert(signature, new_id);
                        next_signature_id += 1;
                        new_id
                    }
                };
                signature_ids[state_id] = signature_id;
            }

            signature_ids
        };

        let (new_classes, new_num_classes) = build_classes_from_keys(
            (0..num_states).map(|state_id| (label_classes[state_id], transition_signature_ids[state_id])),
            num_states,
        );

        if new_num_classes == prev_num_classes {
            prev_classes = new_classes;
            break;
        }

        prev_classes = new_classes;
        prev_num_classes = new_num_classes;
    }

    build_subset_mapping(states, &prev_classes)
}

pub fn find_state_equivalence_classes<S: AsRef<[u8]>>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
) -> Vec<usize> {
    let max_len = tokens
        .iter()
        .map(|token| token.as_ref().len())
        .max()
        .unwrap_or(0);

    find_state_equivalence_classes_kstep(tokenizer, states, max_len)
}
