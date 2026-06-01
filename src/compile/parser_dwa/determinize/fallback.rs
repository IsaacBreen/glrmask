//! Fallback/default determinization.
//!
//! After default-edge optimization, the automaton has fallback semantics that
//! are convenient for construction but inconvenient for runtime walking.  This
//! pass makes those semantics explicit in an ordinary deterministic weighted
//! automaton.

use std::collections::VecDeque;

use rustc_hash::FxHashMap;

use crate::automata::weighted::dwa::DWA;
use crate::parser::glr::labels::DEFAULT_LABEL;
use crate::ds::weight::Weight;

use super::super::types::{
    add_target_contribution, extend_target_contribs, PossibleOutgoingIds, TargetContribs,
};

pub(crate) fn determinize_parser_dwa_with_fallbacks(
    dwa: &DWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
) -> DWA {
    fn subset_key(entries: &[(u32, Weight)]) -> Vec<(u32, usize)> {
        entries.iter().map(|(sid, w)| (*sid, w.ptr_key())).collect()
    }

    let dense_label_limit = num_parser_states as usize;
    let mut result = DWA::new(0, 0);

    let mut start_subset = FxHashMap::default();
    start_subset.insert(dwa.start_state(), Weight::all());

    let mut canon_buf: Vec<(u32, Weight)> = start_subset
        .iter()
        .map(|(state_id, weight)| (*state_id, weight.clone()))
        .collect();
    canon_buf.sort_unstable_by_key(|(state_id, _)| *state_id);

    let mut subset_map: FxHashMap<Vec<(u32, usize)>, u32> = FxHashMap::default();
    let mut worklist: VecDeque<Vec<(u32, Weight)>> = VecDeque::new();
    subset_map.insert(subset_key(&canon_buf), result.start_state());
    worklist.push_back(canon_buf.clone());

    let mut dense_raw_targets: Vec<TargetContribs> =
        (0..dense_label_limit).map(|_| TargetContribs::new()).collect();
    let mut default_raw_targets: TargetContribs = TargetContribs::new();
    let mut sparse_raw_targets: FxHashMap<i32, TargetContribs> = FxHashMap::default();
    let mut touched_dense_labels: Vec<usize> = Vec::new();
    let mut dense_label_touched: Vec<bool> = vec![false; dense_label_limit];
    let mut default_touched = false;
    let mut dense_default_all_raw_targets: TargetContribs = TargetContribs::new();
    let mut key_buf: Vec<(u32, usize)> = Vec::new();
    let mut final_contributions: Vec<Weight> = Vec::new();

    while let Some(subset_entries) = worklist.pop_front() {
        dense_default_all_raw_targets.clear();
        let from_state = subset_map[&subset_key(&subset_entries)];

        final_contributions.clear();
        for (state_id, path_weight) in &subset_entries {
            let Some(state_final) = dwa.states()[*state_id as usize].final_weight.as_ref() else {
                continue;
            };
            let contribution = path_weight.intersection(state_final);
            if !contribution.is_empty() {
                final_contributions.push(contribution);
            }
        }
        let final_weight = Weight::union_all(final_contributions.iter());
        if !final_weight.is_empty() {
            result.set_final_weight(from_state, final_weight);
        }

        for (dwa_state_id, path_weight) in &subset_entries {
            let state = &dwa.states()[*dwa_state_id as usize];

            for (&label, (target, transition_weight)) in &state.transitions {
                if label == DEFAULT_LABEL {
                    continue;
                }
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
                } else {
                    sparse_raw_targets.entry(label).or_default()
                };
                add_target_contribution(target_weights, *target, next_weight);
            }

            let Some((default_target, default_weight)) = state.transitions.get(&DEFAULT_LABEL) else {
                continue;
            };

            let fallback_weight = path_weight.intersection(default_weight);
            if fallback_weight.is_empty() {
                continue;
            }

            default_touched = true;
            add_target_contribution(&mut default_raw_targets, *default_target, fallback_weight.clone());

            for &label in state.transitions.keys() {
                if label == DEFAULT_LABEL {
                    continue;
                }
                if label >= 0 && (label as usize) < dense_label_limit {
                    let label_idx = label as usize;
                    if !dense_label_touched[label_idx] {
                        dense_label_touched[label_idx] = true;
                        touched_dense_labels.push(label_idx);
                    }
                    let target_weights = &mut dense_raw_targets[label_idx];
                    add_target_contribution(target_weights, *default_target, fallback_weight.clone());
                } else {
                    let target_weights = sparse_raw_targets.entry(label).or_default();
                    add_target_contribution(target_weights, *default_target, fallback_weight.clone());
                }
            }

            match possible_by_state.get(*dwa_state_id as usize) {
                Some(PossibleOutgoingIds::All) => {
                    add_target_contribution(
                        &mut dense_default_all_raw_targets,
                        *default_target,
                        fallback_weight.clone(),
                    );
                }
                Some(PossibleOutgoingIds::Some(ids)) => {
                    for parser_state_id in ids.iter_ones() {
                        let label_idx = parser_state_id;
                        if !dense_label_touched[label_idx] {
                            dense_label_touched[label_idx] = true;
                            touched_dense_labels.push(label_idx);
                        }
                        let target_weights = &mut dense_raw_targets[label_idx];
                        add_target_contribution(target_weights, *default_target, fallback_weight.clone());
                    }
                }
                Some(PossibleOutgoingIds::Empty) | None => {}
            }
        }

        let mut process_label = |label: i32, mut contribs: TargetContribs| {
            if contribs.is_empty() {
                return;
            }

            debug_assert!(contribs.iter().all(|(_, weight)| !weight.is_empty()));
            contribs.sort_unstable_by_key(|(state_id, _)| *state_id);

            let edge_weight = Weight::union_all(contribs.iter().map(|(_, weight)| weight));
            if edge_weight.is_empty() {
                return;
            }

            key_buf.clear();
            if contribs.len() == 1 {
                let (only_state, only_weight) = &contribs[0];
                key_buf.push((*only_state, only_weight.ptr_key()));
            } else {
                key_buf.extend(contribs.iter().map(|(sid, w)| (*sid, w.ptr_key())));
            }

            let to_state = if let Some(existing) = subset_map.get(&key_buf).copied() {
                existing
            } else {
                let new_state = result.add_state();
                subset_map.insert(key_buf.clone(), new_state);
                let next_entries: Vec<(u32, Weight)> = contribs.into_iter().collect();
                worklist.push_back(next_entries);
                new_state
            };

            result.add_transition(from_state, label, to_state, edge_weight);
        };

        for label_idx in touched_dense_labels.drain(..) {
            dense_label_touched[label_idx] = false;
            if !dense_default_all_raw_targets.is_empty() {
                extend_target_contribs(
                    &mut dense_raw_targets[label_idx],
                    &dense_default_all_raw_targets,
                );
            }
            process_label(
                label_idx as i32,
                std::mem::take(&mut dense_raw_targets[label_idx]),
            );
        }
        if default_touched {
            default_touched = false;
            process_label(DEFAULT_LABEL, std::mem::take(&mut default_raw_targets));
        }
        for (label, contribs) in sparse_raw_targets.drain() {
            process_label(label, contribs);
        }
    }

    result
}
