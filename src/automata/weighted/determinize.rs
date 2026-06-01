//! Weighted determinization for acyclic NWAs.
//!
//! Cyclic inputs are rejected and must be handled by the caller.

use std::collections::{VecDeque, hash_map::Entry as HashMapEntry};

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::dwa::DWA;
use super::nwa::NWA;
use crate::sets::weight::Weight;
use crate::GlrMaskError;

fn union_state_weight(weights: &mut FxHashMap<u32, Weight>, state_id: u32, add: Weight) {
    if add.is_empty() {
        return;
    }

    match weights.entry(state_id) {
        HashMapEntry::Occupied(mut occupied) => {
            let existing = occupied.get_mut();
            *existing = existing.union(&add);
        }
        HashMapEntry::Vacant(vacant) => {
            vacant.insert(add);
        }
    }
}

fn subset_final_weight(nwa: &NWA, subset_entries: &[(u32, Weight)]) -> Weight {
    subset_entries.iter().fold(Weight::empty(), |final_weight, (state_id, path_weight)| {
        let Some(state_final) = nwa.states()[*state_id as usize].final_weight.as_ref() else {
            return final_weight;
        };

        final_weight.union(&path_weight.intersection(state_final))
    })
}

fn seed_start_subset(nwa: &NWA) -> FxHashMap<u32, Weight> {
    let mut start_subset = FxHashMap::default();
    for &state_id in nwa.start_states() {
        union_state_weight(&mut start_subset, state_id, Weight::all());
    }
    start_subset
}

pub fn determinize(nwa: &NWA) -> Result<DWA, GlrMaskError> {
    if !nwa.is_acyclic() {
        return Err(GlrMaskError::Compilation(
            "weighted determinization currently supports only acyclic NWAs".into(),
        ));
    }

    let profile = std::env::var("GLRMASK_PROFILE_DETERMINIZE").map(|v| v == "1").unwrap_or(false);

    fn canonicalize(subset: &FxHashMap<u32, Weight>) -> Vec<(u32, Weight)> {
        let mut entries: Vec<_> = subset
            .iter()
            .filter_map(|(&state_id, weight)| (!weight.is_empty()).then_some((state_id, weight.clone())))
            .collect();
        entries.sort_by_key(|(state_id, _)| *state_id);
        entries
    }

    fn epsilon_closure(nwa: &NWA, seed: FxHashMap<u32, Weight>) -> FxHashMap<u32, Weight> {
        // Fast path: single-state seed with no epsilon transitions (99.6% of calls)
        if seed.len() == 1 {
            let (&state_id, _) = seed.iter().next().unwrap();
            if let Some(state) = nwa.states().get(state_id as usize) {
                if state.epsilons.is_empty() {
                    return seed;
                }
            }
        }

        let mut closure = seed;
        let mut queue: VecDeque<u32> = closure.keys().copied().collect();

        while let Some(state_id) = queue.pop_front() {
            let Some(current_weight) = closure.get(&state_id).cloned() else {
                continue;
            };
            let Some(state) = nwa.states().get(state_id as usize) else {
                continue;
            };
            for (dst, edge_weight) in &state.epsilons {
                let contribution = current_weight.intersection(edge_weight);
                if contribution.is_empty() {
                    continue;
                }
                let existing = closure.get(dst).cloned().unwrap_or_else(Weight::empty);
                if !contribution.is_subset(&existing) {
                    closure.insert(*dst, existing.union(&contribution));
                    queue.push_back(*dst);
                }
            }
        }

        closure
    }

    let mut dwa = DWA::new(0, 0);
    let start_id = dwa.start_state();

    let start_subset = epsilon_closure(nwa, seed_start_subset(nwa));

    if start_subset.is_empty() {
        return Ok(dwa);
    }

    let mut subset_map: FxHashMap<Vec<(u32, Weight)>, u32> = FxHashMap::default();
    let mut worklist: VecDeque<(Vec<(u32, Weight)>, Vec<(u32, Weight)>)> = VecDeque::new();
    let start_entries = canonicalize(&start_subset);
    let start_key = start_entries.clone();
    subset_map.insert(start_key.clone(), start_id);
    worklist.push_back((start_key, start_entries));

    let mut raw_targets: FxHashMap<i32, FxHashMap<u32, Vec<Weight>>> = FxHashMap::default();

    while let Some((subset_key, subset_entries)) = worklist.pop_front() {
        let from_state = subset_map[&subset_key];

        // Final weight computation is deferred to after the main loop
        // and parallelized across all states.

        for (nwa_state_id, path_weight) in &subset_entries {
            let state = &nwa.states()[*nwa_state_id as usize];
            for (&label, targets) in &state.transitions {
                for (dst, trans_weight) in targets {
                    let next_weight = path_weight.intersection(trans_weight);
                    if next_weight.is_empty() {
                        continue;
                    }

                    raw_targets.entry(label).or_default().entry(*dst).or_default().push(next_weight);
                }
            }
        }

        for (label, target_contributions) in raw_targets.drain() {
            if target_contributions.is_empty() {
                continue;
            }

            let mut target_subset: FxHashMap<u32, Weight> = FxHashMap::default();
            for (dst, weights) in target_contributions {
                let combined = Weight::union_all(weights.iter());
                if !combined.is_empty() {
                    target_subset.insert(dst, combined);
                }
            }

            if target_subset.is_empty() {
                continue;
            }

            let edge_weight = Weight::union_all(target_subset.values());
            if edge_weight.is_empty() {
                continue;
            }

            let expanded = epsilon_closure(nwa, target_subset);
            if expanded.is_empty() {
                continue;
            }

            let edge_complement = edge_weight.complement();
            let normalized: FxHashMap<u32, Weight> = if edge_complement.is_empty() {
                expanded
            } else {
                expanded
                    .into_iter()
                    .filter_map(|(state_id, weight)| {
                        let normalized_weight = weight.union(&edge_complement);
                        (!normalized_weight.is_empty()).then_some((state_id, normalized_weight))
                    })
                    .collect()
            };
            let next_key = canonicalize(&normalized);
            if next_key.is_empty() {
                continue;
            }
            let to_state = if let Some(existing) = subset_map.get(&next_key).copied() {
                existing
            } else {
                let new_id = dwa.add_state();
                subset_map.insert(next_key.clone(), new_id);
                worklist.push_back((next_key.clone(), next_key));
                new_id
            };

            dwa.add_transition(from_state, label, to_state, edge_weight);
        }
    }

    // Compute final weights in parallel after the main loop.
    // The subset_map already stores all (entries, state_id) pairs.
    let final_weights: Vec<(u32, Weight)> = subset_map
        .par_iter()
        .filter_map(|(entries, &state_id)| {
            let fw = subset_final_weight(nwa, entries);
            (!fw.is_empty()).then_some((state_id, fw))
        })
        .collect();
    for (state_id, fw) in final_weights {
        dwa.set_final_weight(state_id, fw);
    }

    if profile {
        let max_weight_dim = dwa.states().iter()
            .filter_map(|s| s.final_weight.as_ref())
            .map(|w| w.0.ranges_len())
            .max()
            .unwrap_or(0);
        eprintln!(
            "[glrmask/profile][determinize] nwa_states={} dwa_states={} subset_map_entries={} max_weight_dim={}",
            nwa.states().len(),
            dwa.states().len(),
            subset_map.len(),
            max_weight_dim,
        );
    }

    Ok(dwa)
}
