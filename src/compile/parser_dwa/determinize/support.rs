//! Support-preserving weighted subset construction.
//!
//! This is the first determinization pass.  Besides producing a DWA, it keeps
//! for each determinized state the set of source-NWA states that contributed
//! to that state.  Those supports are required by fallback/default handling.

use std::collections::{hash_map::Entry, VecDeque};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::nwa::NWA;
use crate::parser::glr::labels::DEFAULT_LABEL;
use crate::ds::weight::Weight;

use super::epsilon::local_epsilon_closure;
use super::super::types::{
    add_target_contribution, CachedClosure, DeterminizedDwaWithSupports, TargetContribs,
};

pub(crate) fn determinize_with_supports(
    nwa: &NWA,
    dense_positive_label_limit: Option<u32>,
) -> DeterminizedDwaWithSupports {
    fn subset_key(entries: &[(u32, Weight)]) -> Vec<(u32, usize)> {
        entries.iter().map(|(sid, w)| (*sid, w.ptr_key())).collect()
    }

    let num_nwa_states = nwa.states().len();

    // Use flat arrays for epsilon closure when NWA is small enough.
    // weight_by_state[i] = Some(weight) means state i is in the closure.
    let mut weight_by_state: Vec<Option<Weight>> = vec![None; num_nwa_states];
    let mut closure_queue: VecDeque<u32> = VecDeque::new();
    // Reusable buffer for canonicalized entries.
    let mut canon_buf: Vec<(u32, Weight)> = Vec::new();

    // Epsilon closure using flat arrays instead of FxHashMap.
    let epsilon_closure = |weight_by_state: &mut Vec<Option<Weight>>,
                           closure_queue: &mut VecDeque<u32>,
                           seed: &mut FxHashMap<u32, Weight>| {
        // Initialize flat array from seed.
        let mut seed_states: Vec<u32> = Vec::new();
        for (&state_id, weight) in seed.iter() {
            weight_by_state[state_id as usize] = Some(weight.clone());
            closure_queue.push_back(state_id);
            seed_states.push(state_id);
        }

        // Fast path: single seed with no epsilons.
        if seed.len() == 1 {
            let state_id = seed_states[0];
            if let Some(state) = nwa.states().get(state_id as usize) {
                if state.epsilons.is_empty() {
                    // Clean up and return early — seed is already populated.
                    closure_queue.clear();
                    for &s in &seed_states {
                        weight_by_state[s as usize] = None;
                    }
                    return;
                }
            }
        }

        while let Some(state_id) = closure_queue.pop_front() {
            let Some(current_weight) = weight_by_state[state_id as usize].clone() else {
                continue;
            };
            let Some(state) = nwa.states().get(state_id as usize) else {
                continue;
            };
            for (target, edge_weight) in &state.epsilons {
                let contribution = current_weight.intersection(edge_weight);
                if contribution.is_empty() {
                    continue;
                }
                let target_idx = *target as usize;
                if let Some(existing) = &weight_by_state[target_idx] {
                    if !contribution.is_subset(existing) {
                        weight_by_state[target_idx] = Some(existing.union(&contribution));
                        closure_queue.push_back(*target);
                    }
                } else {
                    weight_by_state[target_idx] = Some(contribution);
                    closure_queue.push_back(*target);
                    seed_states.push(*target);
                }
            }
        }

        // Write results back to seed map.
        seed.clear();
        for &s in &seed_states {
            if let Some(w) = weight_by_state[s as usize].take() {
                seed.insert(s, w);
            }
        }
    };

    // Canonicalize from FxHashMap into reusable buffer.
    let canonicalize_into =
        |map: &FxHashMap<u32, Weight>, buf: &mut Vec<(u32, Weight)>| {
            buf.clear();
            for (&state_id, weight) in map.iter() {
                if !weight.is_empty() {
                    buf.push((state_id, weight.clone()));
                }
            }
            buf.sort_unstable_by_key(|(state_id, _)| *state_id);
        };

    let mut dwa = DWA::new(0, 0);
    let mut supports = vec![Vec::new()];

    let mut start_subset = FxHashMap::default();
    for &state_id in nwa.start_states() {
        let existing = start_subset.get(&state_id).cloned().unwrap_or_else(Weight::empty);
        start_subset.insert(state_id, existing.union(&Weight::all()));
    }
    epsilon_closure(&mut weight_by_state, &mut closure_queue, &mut start_subset);
    if start_subset.is_empty() {
        return DeterminizedDwaWithSupports { dwa, supports };
    }

    canonicalize_into(&start_subset, &mut canon_buf);
    supports[0] = canon_buf.iter().map(|(state_id, _)| *state_id).collect();

    let mut subset_map: FxHashMap<Vec<(u32, usize)>, u32> = FxHashMap::default();
    let mut worklist: VecDeque<Vec<(u32, Weight)>> = VecDeque::new();
    subset_map.insert(subset_key(&canon_buf), dwa.start_state());
    worklist.push_back(canon_buf.clone());

    let dense_label_limit = dense_positive_label_limit.map(|n| n as usize).unwrap_or(0);
    let mut dense_raw_targets: Vec<TargetContribs> =
        (0..dense_label_limit).map(|_| TargetContribs::new()).collect();
    let mut default_raw_targets: TargetContribs = TargetContribs::new();
    let mut sparse_raw_targets: FxHashMap<i32, TargetContribs> = FxHashMap::default();
    let mut touched_dense_labels: Vec<usize> = Vec::new();
    let mut dense_label_touched: Vec<bool> = vec![false; dense_label_limit];
    let mut default_touched = false;
    // Memoize local epsilon-closure outputs keyed by pre-closure weighted subsets.
    let mut closure_cache: FxHashMap<Vec<(u32, usize)>, CachedClosure> = FxHashMap::default();
    let mut key_buf: Vec<(u32, usize)> = Vec::new();

    // Deferred final weight computation: store subset entries for each DWA state
    // and compute final weights in parallel after the main loop.
    let mut deferred_final_entries: Vec<(u32, Vec<(u32, Weight)>)> = Vec::new();

    while let Some(subset_entries) = worklist.pop_front() {
        let from_state = subset_map[&subset_key(&subset_entries)];

        // Save subset entries for deferred parallel final weight computation.
        // Only save entries whose NWA states have final weights.
        let has_finals: Vec<(u32, Weight)> = subset_entries.iter()
            .filter(|(nwa_state_id, _)| nwa.states()[*nwa_state_id as usize].final_weight.is_some())
            .map(|(id, w)| (*id, w.clone()))
            .collect();
        if !has_finals.is_empty() {
            deferred_final_entries.push((from_state, has_finals));
        }
        for (nwa_state_id, path_weight) in &subset_entries {
            let state = &nwa.states()[*nwa_state_id as usize];
            for (&label, targets) in &state.transitions {
                for (target, transition_weight) in targets {
                    let next_weight = path_weight.intersection(transition_weight);
                    if next_weight.is_empty() {
                        continue;
                    }

                    let target_weights = if label >= 0 && (label as usize) < dense_label_limit {
                        let label_idx = label as usize;
                        if !dense_label_touched[label_idx] {
                            dense_label_touched[label_idx] = true;
                            touched_dense_labels.push(label_idx);
                        }
                        &mut dense_raw_targets[label_idx]
                    } else if label == DEFAULT_LABEL {
                        default_touched = true;
                        &mut default_raw_targets
                    } else {
                        sparse_raw_targets.entry(label).or_default()
                    };
                    add_target_contribution(target_weights, *target, next_weight);
                }
            }
        }

        let mut pre_closure_key: Vec<(u32, usize)> = Vec::new();

        let mut process_label = |label: i32, mut contribs: TargetContribs| {
            if contribs.is_empty() {
                return;
            }

            debug_assert!(contribs.iter().all(|(_, weight)| !weight.is_empty()));

            contribs.sort_unstable_by_key(|(state_id, _)| *state_id);

            if contribs.len() == 1 {
                let (only_state, only_weight) = &contribs[0];
                if nwa.states()[*only_state as usize].epsilons.is_empty() {
                    key_buf.clear();
                    key_buf.push((*only_state, only_weight.ptr_key()));
                    let to_state = if let Some(existing) = subset_map.get(&key_buf).copied() {
                        existing
                    } else {
                        let new_state = dwa.add_state();
                        subset_map.insert(key_buf.clone(), new_state);
                        worklist.push_back(vec![(*only_state, only_weight.clone())]);
                        supports.push(vec![*only_state]);
                        new_state
                    };
                    dwa.add_transition(from_state, label, to_state, only_weight.clone());
                    return;
                }
            }

            pre_closure_key.clear();
            pre_closure_key.extend(contribs.iter().map(|(sid, w)| (*sid, w.ptr_key())));

            let cached = match closure_cache.entry(pre_closure_key.clone()) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let edge_weight = Weight::union_all(contribs.iter().map(|(_, weight)| weight));
                    if edge_weight.is_empty() {
                        return;
                    }
                    let mut target_subset: FxHashMap<u32, Weight> = contribs
                        .iter()
                        .map(|(state_id, weight)| (*state_id, weight.clone()))
                        .collect();
                    local_epsilon_closure(
                        nwa,
                        &mut weight_by_state,
                        &mut closure_queue,
                        &mut target_subset,
                    );
                    if target_subset.is_empty() {
                        return;
                    }
                    let mut canon: Vec<(u32, Weight)> = target_subset
                        .iter()
                        .filter(|(_, w)| !w.is_empty())
                        .map(|(id, w)| (*id, w.clone()))
                        .collect();
                    canon.sort_unstable_by_key(|(state_id, _)| *state_id);
                    if canon.is_empty() {
                        return;
                    }
                    entry.insert(CachedClosure { canon, edge_weight })
                }
            };

            key_buf.clear();
            key_buf.extend(cached.canon.iter().map(|(sid, w)| (*sid, w.ptr_key())));
            let to_state = if let Some(existing) = subset_map.get(&key_buf).copied() {
                existing
            } else {
                let new_state = dwa.add_state();
                subset_map.insert(key_buf.clone(), new_state);
                worklist.push_back(cached.canon.clone());
                supports.push(cached.canon.iter().map(|(sid, _)| *sid).collect());
                new_state
            };
            dwa.add_transition(from_state, label, to_state, cached.edge_weight.clone());
        };

        for label_idx in touched_dense_labels.drain(..) {
            dense_label_touched[label_idx] = false;
            process_label(label_idx as i32, std::mem::take(&mut dense_raw_targets[label_idx]));
        }

        if default_touched {
            default_touched = false;
            process_label(DEFAULT_LABEL, std::mem::take(&mut default_raw_targets));
        }

        for (label, contribs) in sparse_raw_targets.drain() {
            process_label(label, contribs);
        }
    }

    // Compute final weights in parallel using rayon.
    {
        use rayon::prelude::*;
        let final_weights: Vec<(u32, Weight)> = deferred_final_entries
            .par_iter()
            .filter_map(|(state_id, entries)| {
                // Group by final weight pointer to leverage distributivity.
                let mut final_groups: SmallVec<[(usize, &Weight, SmallVec<[&Weight; 4]>); 4]> = SmallVec::new();
                for (nwa_state_id, path_weight) in entries {
                    if let Some(state_final) = nwa.states()[*nwa_state_id as usize].final_weight.as_ref() {
                        let key = state_final.ptr_key();
                        if let Some(group) = final_groups.iter_mut().find(|(k, _, _)| *k == key) {
                            group.2.push(path_weight);
                        } else {
                            let mut pws = SmallVec::new();
                            pws.push(path_weight);
                            final_groups.push((key, state_final, pws));
                        }
                    }
                }
                let final_contributions: SmallVec<[Weight; 4]> = final_groups.into_iter()
                    .filter_map(|(_, final_w, path_weights)| {
                        let pw_union = Weight::union_all(path_weights.into_iter());
                        let contribution = pw_union.intersection(final_w);
                        if contribution.is_empty() { None } else { Some(contribution) }
                    })
                    .collect();
                let final_weight = Weight::union_all(final_contributions.iter());
                if final_weight.is_empty() { None } else { Some((*state_id, final_weight)) }
            })
            .collect();
        for (state_id, weight) in final_weights {
            dwa.set_final_weight(state_id, weight);
        }
    }

    DeterminizedDwaWithSupports { dwa, supports }
}

