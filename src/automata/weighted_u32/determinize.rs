//! Weighted determinization for acyclic NWAs.
//!
//! Cyclic inputs are rejected and must be handled by the caller.

use std::collections::{VecDeque, hash_map::Entry as HashMapEntry};
use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::dwa::DWA;
use super::nwa::NWA;
use crate::ds::weight::Weight;
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

#[derive(Default)]
struct FinalWeightProfile {
    subsets: usize,
    subset_entries: usize,
    final_entries: usize,
    nonempty_contributions: usize,
    max_final_entries: usize,
    intersection_ms: f64,
    union_ms: f64,
}

impl FinalWeightProfile {
    fn merge(&mut self, other: Self) {
        self.subsets += other.subsets;
        self.subset_entries += other.subset_entries;
        self.final_entries += other.final_entries;
        self.nonempty_contributions += other.nonempty_contributions;
        self.max_final_entries = self.max_final_entries.max(other.max_final_entries);
        self.intersection_ms += other.intersection_ms;
        self.union_ms += other.union_ms;
    }
}

fn subset_final_weight_profiled(
    nwa: &NWA,
    subset_entries: &[(u32, Weight)],
) -> (Weight, FinalWeightProfile) {
    let mut profile = FinalWeightProfile {
        subsets: 1,
        subset_entries: subset_entries.len(),
        ..FinalWeightProfile::default()
    };
    let mut final_weight = Weight::empty();

    for (state_id, path_weight) in subset_entries {
        let Some(state_final) = nwa.states()[*state_id as usize].final_weight.as_ref() else {
            continue;
        };
        profile.final_entries += 1;
        profile.max_final_entries = profile.max_final_entries.max(profile.final_entries);

        let intersection_started_at = Instant::now();
        let contribution = path_weight.intersection(state_final);
        profile.intersection_ms += intersection_started_at.elapsed().as_secs_f64() * 1000.0;
        if contribution.is_empty() {
            continue;
        }
        profile.nonempty_contributions += 1;

        let union_started_at = Instant::now();
        final_weight = final_weight.union(&contribution);
        profile.union_ms += union_started_at.elapsed().as_secs_f64() * 1000.0;
    }

    (final_weight, profile)
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

    // Almost every label has one surviving destination. Keep that common
    // case inline instead of allocating a nested hash map and a Vec per label.
    let mut raw_targets: FxHashMap<i32, SmallVec<[(u32, Weight); 1]>> = FxHashMap::default();
    let mut profile_subset_entries = 0usize;
    let mut profile_max_subset_entries = 0usize;
    let mut profile_raw_transition_visits = 0usize;
    let mut profile_labels = 0usize;
    let mut profile_target_contributions = 0usize;
    let mut profile_expand_ms = 0.0;
    let mut profile_combine_ms = 0.0;
    let mut profile_edge_union_ms = 0.0;
    let mut profile_closure_ms = 0.0;
    let mut profile_normalize_ms = 0.0;
    let mut profile_canonicalize_ms = 0.0;
    let mut profile_subset_lookup_ms = 0.0;

    while let Some((subset_key, subset_entries)) = worklist.pop_front() {
        if profile {
            profile_subset_entries += subset_entries.len();
            profile_max_subset_entries = profile_max_subset_entries.max(subset_entries.len());
        }
        let from_state = subset_map[&subset_key];

        // Final weight computation is deferred to after the main loop
        // and parallelized across all states.
        let expand_started_at = profile.then(Instant::now);

        for (nwa_state_id, path_weight) in &subset_entries {
            let state = &nwa.states()[*nwa_state_id as usize];
            for (&label, targets) in &state.transitions {
                for (dst, trans_weight) in targets {
                    if profile {
                        profile_raw_transition_visits += 1;
                    }
                    let next_weight = path_weight.intersection(trans_weight);
                    if next_weight.is_empty() {
                        continue;
                    }

                    raw_targets.entry(label).or_default().push((*dst, next_weight));
                }
            }
        }

        if let Some(expand_started_at) = expand_started_at {
            profile_expand_ms += expand_started_at.elapsed().as_secs_f64() * 1000.0;
        }

        for (label, target_contributions) in raw_targets.drain() {
            if profile {
                profile_labels += 1;
                profile_target_contributions += target_contributions.len();
            }
            if target_contributions.is_empty() {
                continue;
            }

            let combine_started_at = profile.then(Instant::now);
            let mut target_subset: FxHashMap<u32, Weight> = FxHashMap::default();
            if target_contributions.len() == 1 {
                let (dst, weight) = target_contributions.into_iter().next().unwrap();
                target_subset.insert(dst, weight);
            } else {
                for (dst, weight) in target_contributions {
                    union_state_weight(&mut target_subset, dst, weight);
                }
            }

            if let Some(combine_started_at) = combine_started_at {
                profile_combine_ms += combine_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            if target_subset.is_empty() {
                continue;
            }

            let edge_union_started_at = profile.then(Instant::now);
            let edge_weight = Weight::union_all(target_subset.values());
            if let Some(edge_union_started_at) = edge_union_started_at {
                profile_edge_union_ms += edge_union_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            if edge_weight.is_empty() {
                continue;
            }

            let closure_started_at = profile.then(Instant::now);
            let expanded = epsilon_closure(nwa, target_subset);
            if let Some(closure_started_at) = closure_started_at {
                profile_closure_ms += closure_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            if expanded.is_empty() {
                continue;
            }

            let normalize_started_at = profile.then(Instant::now);
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
            if let Some(normalize_started_at) = normalize_started_at {
                profile_normalize_ms += normalize_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            let canonicalize_started_at = profile.then(Instant::now);
            let next_key = canonicalize(&normalized);
            if let Some(canonicalize_started_at) = canonicalize_started_at {
                profile_canonicalize_ms += canonicalize_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            if next_key.is_empty() {
                continue;
            }
            let subset_lookup_started_at = profile.then(Instant::now);
            let to_state = if let Some(existing) = subset_map.get(&next_key).copied() {
                existing
            } else {
                let new_id = dwa.add_state();
                subset_map.insert(next_key.clone(), new_id);
                worklist.push_back((next_key.clone(), next_key));
                new_id
            };
            if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                profile_subset_lookup_ms += subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
            }

            dwa.add_transition(from_state, label, to_state, edge_weight);
        }
    }

    // Compute final weights in parallel after the main loop.
    // The subset_map already stores all (entries, state_id) pairs.
    let final_weights_started_at = profile.then(Instant::now);
    let mut final_weight_profile = FinalWeightProfile::default();
    let final_weights: Vec<(u32, Weight)> = if profile {
        let profiled: Vec<(u32, Weight, FinalWeightProfile)> = subset_map
            .par_iter()
            .map(|(entries, &state_id)| {
                let (fw, stats) = subset_final_weight_profiled(nwa, entries);
                (state_id, fw, stats)
            })
            .collect();
        profiled
            .into_iter()
            .filter_map(|(state_id, fw, stats)| {
                final_weight_profile.merge(stats);
                (!fw.is_empty()).then_some((state_id, fw))
            })
            .collect()
    } else {
        subset_map
            .par_iter()
            .filter_map(|(entries, &state_id)| {
                let fw = subset_final_weight(nwa, entries);
                (!fw.is_empty()).then_some((state_id, fw))
            })
            .collect()
    };
    for (state_id, fw) in final_weights {
        dwa.set_final_weight(state_id, fw);
    }
    let final_weights_ms = final_weights_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    if profile {
        let max_weight_dim = dwa.states().iter()
            .filter_map(|s| s.final_weight.as_ref())
            .map(|w| w.0.ranges_len())
            .max()
            .unwrap_or(0);
        let mut final_pairs = FxHashSet::default();
        let mut final_path_weights = FxHashSet::default();
        let mut final_state_weights = FxHashSet::default();
        let mut final_path_outer_ranges = 0usize;
        let mut final_state_outer_ranges = 0usize;
        let mut max_final_path_outer_ranges = 0usize;
        let mut max_final_state_outer_ranges = 0usize;
        for entries in subset_map.keys() {
            for (state_id, path_weight) in entries {
                let Some(state_final) = nwa.states()[*state_id as usize].final_weight.as_ref() else {
                    continue;
                };
                let path_key = path_weight.ptr_key();
                let state_key = state_final.ptr_key();
                final_pairs.insert((path_key, state_key));
                final_path_weights.insert(path_key);
                final_state_weights.insert(state_key);
                let path_ranges = path_weight.outer_range_count();
                let state_ranges = state_final.outer_range_count();
                final_path_outer_ranges += path_ranges;
                final_state_outer_ranges += state_ranges;
                max_final_path_outer_ranges = max_final_path_outer_ranges.max(path_ranges);
                max_final_state_outer_ranges = max_final_state_outer_ranges.max(state_ranges);
            }
        }
        eprintln!(
            "[glrmask/profile][determinize_final_shape] pairs={} unique_pairs={} unique_path_weights={} unique_state_weights={} path_outer_ranges={} state_outer_ranges={} max_path_outer_ranges={} max_state_outer_ranges={}",
            final_weight_profile.final_entries,
            final_pairs.len(),
            final_path_weights.len(),
            final_state_weights.len(),
            final_path_outer_ranges,
            final_state_outer_ranges,
            max_final_path_outer_ranges,
            max_final_state_outer_ranges,
        );

        eprintln!(
            "[glrmask/profile][determinize] nwa_states={} dwa_states={} subset_map_entries={} max_weight_dim={} subset_entries={} max_subset_entries={} raw_transition_visits={} labels={} target_contributions={} expand_ms={:.3} combine_ms={:.3} edge_union_ms={:.3} closure_ms={:.3} normalize_ms={:.3} canonicalize_ms={:.3} subset_lookup_ms={:.3} final_weights_ms={:.3} final_subsets={} final_subset_entries={} final_entries={} final_nonempty_contributions={} final_max_entries={} final_intersection_ms={:.3} final_union_ms={:.3}",
            nwa.states().len(),
            dwa.states().len(),
            subset_map.len(),
            max_weight_dim,
            profile_subset_entries,
            profile_max_subset_entries,
            profile_raw_transition_visits,
            profile_labels,
            profile_target_contributions,
            profile_expand_ms,
            profile_combine_ms,
            profile_edge_union_ms,
            profile_closure_ms,
            profile_normalize_ms,
            profile_canonicalize_ms,
            profile_subset_lookup_ms,
            final_weights_ms,
            final_weight_profile.subsets,
            final_weight_profile.subset_entries,
            final_weight_profile.final_entries,
            final_weight_profile.nonempty_contributions,
            final_weight_profile.max_final_entries,
            final_weight_profile.intersection_ms,
            final_weight_profile.union_ms,
        );
    }

    Ok(dwa)
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_set_blaze::RangeSetBlaze;

    fn tokens(values: impl IntoIterator<Item = u32>) -> Weight {
        Weight::from_per_tsid_token_sets(std::iter::once((
            0,
            RangeSetBlaze::from_iter(values.into_iter().map(|value| value..=value)),
        )))
    }

    #[test]
    fn determinize_unions_duplicate_label_target_contributions() {
        let mut nwa = NWA::new(1, 2);
        let start = nwa.add_state();
        let accept = nwa.add_state();
        nwa.set_start_states(vec![start]);
        nwa.add_transition(start, 7, accept, tokens([0]));
        nwa.add_transition(start, 7, accept, tokens([1]));
        nwa.set_final_weight(accept, tokens([0, 1]));

        let dwa = determinize(&nwa).unwrap();
        assert_eq!(dwa.eval_word(&[7]), tokens([0, 1]));
    }
}
