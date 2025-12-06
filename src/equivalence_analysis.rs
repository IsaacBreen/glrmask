use crate::finite_automata::Regex;
use hashbrown::HashMap;
use rayon::prelude::*;
use std::collections::BTreeMap;

pub struct SimpleEquivalenceResult {
    pub mask_classes: BTreeMap<Vec<usize>, Vec<usize>>,
    pub commit_classes: BTreeMap<Vec<usize>, Vec<usize>>,
}

#[inline(always)]
fn mix(mut x: u128) -> u128 {
    x ^= x >> 33; x = x.wrapping_mul(0x9e3779b97f4a7c15_bf58476d1ce4e5b9);
    x ^= x >> 33; x = x.wrapping_mul(0x94d049bb133111eb_ff51afd7ed558ccd);
    x ^ (x >> 33)
}

/// Computes the hash contribution of accessible terminals from a restart
fn hash_suffix(regex: &Regex, token: &[u8], from: usize) -> u128 {
    let mut s = regex.dfa.start_state;
    for &b in &token[from..] {
        match regex.dfa.states[s].transitions.get(b) {
            Some(&n) => s = n,
            None => return 0,
        }
    }
    regex.dfa.states[s].possible_future_group_ids.iter()
        .fold(0, |acc, &id| acc.wrapping_add(mix(id as u128 | 2)))
}

fn compute_signature(regex: &Regex, token: &[u8], initial_states: &[usize]) -> u128 {
    initial_states.iter().enumerate().fold(0, |acc, (idx, &start_node)| {
        let mut curr = start_node;
        let mut outcome: u128 = 0;
        let mut dead = false;

        for (i, &b) in token.iter().enumerate() {
            match regex.dfa.states[curr].transitions.get(b) {
                Some(&next) => {
                    curr = next;
                    for gid in regex.dfa.states[curr].finalizers.iter_indices() {
                        let match_h = mix((gid as u128) << 8 | 1)
                            .wrapping_add(hash_suffix(regex, token, i + 1));
                        outcome = outcome.wrapping_add(match_h);
                    }
                }
                None => { dead = true; break; }
            }
        }

        if !dead {
            for &id in &regex.dfa.states[curr].possible_future_group_ids {
                outcome = outcome.wrapping_add(mix(id as u128 | 4));
            }
        }
        acc.wrapping_add(mix(outcome ^ (idx as u128) << 64))
    })
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> SimpleEquivalenceResult {
    crate::debug!(3, "Simple equiv v3: {} strings, {} states", strings.len(), initial_states.len());
    let t0 = std::time::Instant::now();

    let mut groups = HashMap::new();
    strings.par_iter()
        .map(|s| compute_signature(regex, s, initial_states))
        .collect::<Vec<_>>()
        .into_iter().enumerate()
        .for_each(|(i, h)| groups.entry(h).or_insert_with(Vec::new).push(i));

    let classes: BTreeMap<_, _> = groups.into_iter()
        .enumerate().map(|(id, (_, v))| (vec![id], v)).collect();

    crate::debug!(3, "Simple equiv v3 done in {:?}", t0.elapsed());
    SimpleEquivalenceResult { mask_classes: classes.clone(), commit_classes: classes }
}