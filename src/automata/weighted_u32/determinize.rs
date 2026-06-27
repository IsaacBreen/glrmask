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
use crate::ds::weight::{SharedTokenSet, ScopedWeightOpCache, Weight, WeightIntersectionIndex};
use crate::GlrMaskError;

const MAX_INDEXED_FINAL_PATH_RANGES: usize = 8;
const MIN_INDEXED_FINAL_WEIGHT_RANGES: usize = 32;

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

fn final_weight_intersection(
    path_weight: &Weight,
    state_final: &Weight,
    final_weight_indices: &FxHashMap<usize, WeightIntersectionIndex>,
    max_indexed_path_ranges: usize,
) -> Weight {
    if path_weight.outer_range_count() <= max_indexed_path_ranges {
        if let Some(index) = final_weight_indices.get(&state_final.ptr_key()) {
            return path_weight.intersection_with_index(index);
        }
    }
    path_weight.intersection(state_final)
}

fn subset_final_weight(
    nwa: &NWA,
    subset_entries: &[(u32, Weight)],
    final_weight_indices: &FxHashMap<usize, WeightIntersectionIndex>,
    max_indexed_path_ranges: usize,
) -> Weight {
    subset_entries.iter().fold(Weight::empty(), |final_weight, (state_id, path_weight)| {
        let Some(state_final) = nwa.states()[*state_id as usize].final_weight.as_ref() else {
            return final_weight;
        };

        final_weight.union(&final_weight_intersection(
            path_weight,
            state_final,
            final_weight_indices,
            MAX_INDEXED_FINAL_PATH_RANGES,
        ))
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
    final_weight_indices: &FxHashMap<usize, WeightIntersectionIndex>,
    max_indexed_path_ranges: usize,
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
        let contribution = final_weight_intersection(
            path_weight,
            state_final,
            final_weight_indices,
            MAX_INDEXED_FINAL_PATH_RANGES,
        );
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

/// Aggregate direct epsilon-free point-entry contributions without repeatedly
/// rebuilding whole weight maps. Each contribution carries exactly one TSID;
/// sorting lets us reduce token sets locally for a `(destination, TSID)` pair
/// and construct the canonical per-destination and edge weights in one pass.
fn aggregate_direct_point_entries(
    target_contributions: SmallVec<[(u32, Weight); 1]>,
) -> (Vec<(u32, Weight)>, Weight) {
    let mut points: Vec<(u32, u32, SharedTokenSet)> = target_contributions
        .into_iter()
        .map(|(dst, weight)| {
            let (tsid, tokens) = weight
                .single_tsid_shared_entry()
                .expect("point-entry aggregation requires point weights");
            (dst, tsid, tokens)
        })
        .collect();

    let mut edge_entries: Vec<(u32, SharedTokenSet)> = points
        .iter()
        .map(|(_, tsid, tokens)| (*tsid, std::sync::Arc::clone(tokens)))
        .collect();
    edge_entries.sort_unstable_by_key(|(tsid, _)| *tsid);
    let edge_weight = Weight::union_sorted_point_entries(edge_entries);

    points.sort_unstable_by_key(|(dst, tsid, _)| (*dst, *tsid));
    let mut next_key = Vec::new();
    let mut group_start = 0usize;
    while group_start < points.len() {
        let dst = points[group_start].0;
        let mut group_end = group_start + 1;
        while group_end < points.len() && points[group_end].0 == dst {
            group_end += 1;
        }
        let weight = Weight::union_sorted_point_entries(
            points[group_start..group_end]
                .iter()
                .map(|(_, tsid, tokens)| (*tsid, std::sync::Arc::clone(tokens))),
        );
        debug_assert!(!weight.is_empty());
        next_key.push((dst, weight));
        group_start = group_end;
    }

    (next_key, edge_weight)
}

fn intern_determinized_subset(
    next_key: Vec<(u32, Weight)>,
    subset_map: &mut FxHashMap<Vec<(u32, Weight)>, u32>,
    worklist: &mut VecDeque<(Vec<(u32, Weight)>, Vec<(u32, Weight)>)>,
    dwa: &mut DWA,
) -> u32 {
    if let Some(existing) = subset_map.get(&next_key).copied() {
        existing
    } else {
        let new_id = dwa.add_state();
        subset_map.insert(next_key.clone(), new_id);
        worklist.push_back((next_key.clone(), next_key));
        new_id
    }
}

pub fn determinize(nwa: &NWA) -> Result<DWA, GlrMaskError> {
    determinize_impl(nwa, true)
}

fn determinize_impl(
    nwa: &NWA,
    direct_single_target_enabled: bool,
) -> Result<DWA, GlrMaskError> {
    if !nwa.is_acyclic() {
        return Err(GlrMaskError::Compilation(
            "weighted determinization currently supports only acyclic NWAs".into(),
        ));
    }

    let profile = std::env::var("GLRMASK_PROFILE_DETERMINIZE")
        .map(|v| v == "1")
        .unwrap_or(false);

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
    // All operands in the expansion loop remain owned by the NWA or subset_map
    // for the lifetime of this determinization, so a local exact cache avoids
    // thread-local memo overhead on the heavily reused intersection pairs.
    let mut scoped_determinize_weight_cache = ScopedWeightOpCache::default();
    let mut profile_subset_entries = 0usize;
    let mut profile_max_subset_entries = 0usize;
    let mut profile_raw_transition_visits = 0usize;
    let mut profile_labels = 0usize;
    let mut profile_target_contributions = 0usize;
    let mut profile_single_contribution_labels = 0usize;
    let mut profile_single_contribution_no_epsilon_labels = 0usize;
    let mut profile_direct_single_target_labels = 0usize;
    let mut profile_multi_contribution_single_target_no_epsilon_labels = 0usize;
    let mut profile_multi_contribution_single_target_no_epsilon_contributions = 0usize;
    let mut profile_direct_multi_target_labels = 0usize;
    let mut profile_multi_contribution_all_no_epsilon_labels = 0usize;
    let mut profile_multi_contribution_all_no_epsilon_contributions = 0usize;
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
                    let next_weight = scoped_determinize_weight_cache
                        .intersection(path_weight, trans_weight);
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
                if target_contributions.len() == 1 {
                    profile_single_contribution_labels += 1;
                    let (dst, _) = &target_contributions[0];
                    if nwa
                        .states()
                        .get(*dst as usize)
                        .is_some_and(|state| state.epsilons.is_empty())
                    {
                        profile_single_contribution_no_epsilon_labels += 1;
                    }
                } else {
                    let dst = target_contributions[0].0;
                    let all_no_epsilon = target_contributions.iter().all(|(candidate, _)| {
                        nwa
                            .states()
                            .get(*candidate as usize)
                            .is_some_and(|state| state.epsilons.is_empty())
                    });
                    if all_no_epsilon {
                        profile_multi_contribution_all_no_epsilon_labels += 1;
                        profile_multi_contribution_all_no_epsilon_contributions +=
                            target_contributions.len();
                    }
                    if target_contributions.iter().all(|(candidate, _)| *candidate == dst)
                        && all_no_epsilon
                    {
                        profile_multi_contribution_single_target_no_epsilon_labels += 1;
                        profile_multi_contribution_single_target_no_epsilon_contributions +=
                            target_contributions.len();
                    }
                }
            }
            if target_contributions.is_empty() {
                continue;
            }

            let direct_no_epsilon_targets = direct_single_target_enabled
                && target_contributions.iter().all(|(dst, _)| {
                    nwa
                        .states()
                        .get(*dst as usize)
                        .is_some_and(|state| state.epsilons.is_empty())
                });
            if direct_no_epsilon_targets {
                if target_contributions.len() == 1 {
                    if profile {
                        profile_direct_single_target_labels += 1;
                    }
                    let (dst, edge_weight) = target_contributions.into_iter().next().unwrap();
                    let subset_lookup_started_at = profile.then(Instant::now);
                    let to_state = intern_determinized_subset(
                        vec![(dst, edge_weight.clone())],
                        &mut subset_map,
                        &mut worklist,
                        &mut dwa,
                    );
                    if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                        profile_subset_lookup_ms +=
                            subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
                    }
                    dwa.add_transition(from_state, label, to_state, edge_weight);
                    continue;
                }

                let direct_point_entries = target_contributions
                    .iter()
                    .all(|(_, weight)| weight.single_tsid_shared_entry().is_some());
                if direct_point_entries {
                    let (next_key, edge_weight) = aggregate_direct_point_entries(target_contributions);
                    debug_assert!(!edge_weight.is_empty());
                    let subset_lookup_started_at = profile.then(Instant::now);
                    let to_state = intern_determinized_subset(
                        next_key,
                        &mut subset_map,
                        &mut worklist,
                        &mut dwa,
                    );
                    if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                        profile_subset_lookup_ms +=
                            subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
                    }
                    dwa.add_transition(from_state, label, to_state, edge_weight);
                    continue;
                }

                if profile {
                    profile_direct_multi_target_labels += 1;
                }
                let mut sorted_targets = target_contributions;
                sorted_targets.sort_unstable_by_key(|(dst, _)| *dst);
                let mut next_key: Vec<(u32, Weight)> =
                    Vec::with_capacity(sorted_targets.len());
                for (dst, weight) in sorted_targets {
                    if let Some((last_dst, last_weight)) = next_key.last_mut() {
                        if *last_dst == dst {
                            *last_weight = last_weight.union(&weight);
                            continue;
                        }
                    }
                    next_key.push((dst, weight));
                }
                let edge_weight = Weight::union_all(next_key.iter().map(|(_, weight)| weight));
                debug_assert!(!edge_weight.is_empty());

                let subset_lookup_started_at = profile.then(Instant::now);
                let to_state = intern_determinized_subset(
                    next_key,
                    &mut subset_map,
                    &mut worklist,
                    &mut dwa,
                );
                if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                    profile_subset_lookup_ms +=
                        subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
                }
                dwa.add_transition(from_state, label, to_state, edge_weight);
                continue;
            }

            let combine_started_at = profile.then(Instant::now);
            let mut target_subset: FxHashMap<u32, Weight> = FxHashMap::default();
            if target_contributions.len() == 1 {
                let (dst, weight) = target_contributions.into_iter().next().unwrap();
                target_subset.insert(dst, weight);
            } else {
                // Keep the first contribution per destination in the compact
                // target map. On the first repeat, move that destination's
                // operands into a side bucket and union them once at the end.
                // This preserves the exact subset construction while avoiding
                // quadratic growth from repeatedly rebuilding wide weights.
                let mut duplicate_weights: Option<FxHashMap<u32, SmallVec<[Weight; 2]>>> = None;
                for (dst, weight) in target_contributions {
                    if let Some(duplicates) = duplicate_weights.as_mut() {
                        if let Some(weights) = duplicates.get_mut(&dst) {
                            weights.push(weight);
                            continue;
                        }
                    }

                    match target_subset.entry(dst) {
                        HashMapEntry::Vacant(vacant) => {
                            vacant.insert(weight);
                        }
                        HashMapEntry::Occupied(occupied) => {
                            let previous = occupied.remove();
                            let mut weights = SmallVec::<[Weight; 2]>::new();
                            weights.push(previous);
                            weights.push(weight);
                            let duplicates = duplicate_weights.get_or_insert_with(FxHashMap::default);
                            debug_assert!(duplicates.insert(dst, weights).is_none());
                        }
                    }
                }
                if let Some(duplicates) = duplicate_weights {
                    for (dst, weights) in duplicates {
                        target_subset.insert(dst, Weight::union_all(weights.iter()));
                    }
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
            let to_state = intern_determinized_subset(
                next_key,
                &mut subset_map,
                &mut worklist,
                &mut dwa,
            );
            if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                profile_subset_lookup_ms += subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
            }

            dwa.add_transition(from_state, label, to_state, edge_weight);
        }
    }

    if profile {
        eprintln!(
            "[glrmask/profile][determinize_scoped_weight_cache] intersections={}",
            scoped_determinize_weight_cache.intersection_entry_count(),
        );
    }

    // Compute final weights in parallel after the main loop.
    // The subset_map already stores all (entries, state_id) pairs.
    // Sparse path weights recur against a few wide final weights. Index those
    // wide maps once so each sparse range can seek directly to its overlaps.
    let final_weight_indices: FxHashMap<usize, WeightIntersectionIndex> = nwa
        .states()
        .iter()
        .filter_map(|state| state.final_weight.as_ref())
        .filter(|weight| weight.outer_range_count() >= MIN_INDEXED_FINAL_WEIGHT_RANGES)
        .map(|weight| (weight.ptr_key(), weight.intersection_index()))
        .collect();
    let final_weights_started_at = profile.then(Instant::now);
    let mut final_weight_profile = FinalWeightProfile::default();
    let final_weights: Vec<(u32, Weight)> = if profile {
        let profiled: Vec<(u32, Weight, FinalWeightProfile)> = subset_map
            .par_iter()
            .map(|(entries, &state_id)| {
                let (fw, stats) = subset_final_weight_profiled(
                    nwa,
                    entries,
                    &final_weight_indices,
                    MAX_INDEXED_FINAL_PATH_RANGES,
                );
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
                let fw = subset_final_weight(
                    nwa,
                    entries,
                    &final_weight_indices,
                    MAX_INDEXED_FINAL_PATH_RANGES,
                );
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
        let mut final_group_entries = 0usize;
        let mut final_group_count = 0usize;
        let mut final_subsets_with_reused_group = 0usize;
        let mut max_final_groups_per_subset = 0usize;
        let mut max_final_entries_per_group = 0usize;
        for entries in subset_map.keys() {
            let mut groups = FxHashMap::<usize, usize>::default();
            for (state_id, _) in entries {
                let Some(state_final) = nwa.states()[*state_id as usize].final_weight.as_ref() else {
                    continue;
                };
                *groups.entry(state_final.ptr_key()).or_default() += 1;
            }
            let final_entry_count: usize = groups.values().sum();
            if final_entry_count > 0 {
                final_group_entries += final_entry_count;
                final_group_count += groups.len();
                max_final_groups_per_subset = max_final_groups_per_subset.max(groups.len());
                let max_group = groups.values().copied().max().unwrap_or(0);
                max_final_entries_per_group = max_final_entries_per_group.max(max_group);
                final_subsets_with_reused_group += usize::from(max_group > 1);
            }
        }
        eprintln!(
            "[glrmask/profile][determinize_final_groups] entries={} groups={} saved_intersections={} subsets_with_reused_group={} max_groups_per_subset={} max_entries_per_group={}",
            final_group_entries,
            final_group_count,
            final_group_entries.saturating_sub(final_group_count),
            final_subsets_with_reused_group,
            max_final_groups_per_subset,
            max_final_entries_per_group,
        );

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
            "[glrmask/profile][determinize] nwa_states={} dwa_states={} subset_map_entries={} max_weight_dim={} subset_entries={} max_subset_entries={} raw_transition_visits={} labels={} target_contributions={} single_contribution_labels={} single_contribution_no_epsilon_labels={} direct_single_target_labels={} multi_contribution_single_target_no_epsilon_labels={} multi_contribution_single_target_no_epsilon_contributions={} direct_multi_target_labels={} multi_contribution_all_no_epsilon_labels={} multi_contribution_all_no_epsilon_contributions={} expand_ms={:.3} combine_ms={:.3} edge_union_ms={:.3} closure_ms={:.3} normalize_ms={:.3} canonicalize_ms={:.3} subset_lookup_ms={:.3} final_weights_ms={:.3} final_subsets={} final_subset_entries={} final_entries={} final_nonempty_contributions={} final_max_entries={} final_intersection_ms={:.3} final_union_ms={:.3}",
            nwa.states().len(),
            dwa.states().len(),
            subset_map.len(),
            max_weight_dim,
            profile_subset_entries,
            profile_max_subset_entries,
            profile_raw_transition_visits,
            profile_labels,
            profile_target_contributions,
            profile_single_contribution_labels,
            profile_single_contribution_no_epsilon_labels,
            profile_direct_single_target_labels,
            profile_multi_contribution_single_target_no_epsilon_labels,
            profile_multi_contribution_single_target_no_epsilon_contributions,
            profile_direct_multi_target_labels,
            profile_multi_contribution_all_no_epsilon_labels,
            profile_multi_contribution_all_no_epsilon_contributions,
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

        let fast = determinize_impl(&nwa, true).unwrap();
        let generic = determinize_impl(&nwa, false).unwrap();
        assert_eq!(bincode::serialize(&fast).unwrap(), bincode::serialize(&generic).unwrap());
        assert_eq!(fast.eval_word(&[7]), tokens([0, 1]));
    }
    #[test]
    fn determinize_batches_many_repeated_destinations_exactly() {
        let mut nwa = NWA::new(1, 7);
        let start = nwa.add_state();
        let left = nwa.add_state();
        let right = nwa.add_state();
        nwa.set_start_states(vec![start]);

        for token in 0..=4 {
            nwa.add_transition(start, 7, left, tokens([token]));
        }
        for token in 5..=6 {
            nwa.add_transition(start, 7, right, tokens([token]));
        }
        nwa.set_final_weight(left, tokens(0..=4));
        nwa.set_final_weight(right, tokens(5..=6));

        let dwa = determinize(&nwa).unwrap();
        assert_eq!(dwa.eval_word(&[7]), tokens(0..=6));
    }

    #[test]
    fn direct_epsilon_free_subsets_match_generic_across_acyclic_cases() {
        for case in 0u32..32 {
            let mut nwa = NWA::new(1, 8);
            let states: Vec<u32> = (0..5).map(|_| nwa.add_state()).collect();
            nwa.set_start_states(vec![states[0]]);

            for from in 0..4usize {
                for to in (from + 1)..5usize {
                    if (case + (from * 7 + to * 11) as u32) % 3 == 0 {
                        continue;
                    }
                    let label = ((case + (from * 3 + to) as u32) % 4) as i32;
                    let first = (case + (from * 5 + to) as u32) % 6;
                    let second = (first + 1 + case % 3) % 7;
                    nwa.add_transition(states[from], label, states[to], tokens([first, second]));
                    if (case + from as u32 + to as u32) % 5 == 0 {
                        let extra = (second + 2) % 8;
                        nwa.add_transition(states[from], label, states[to], tokens([extra]));
                    }
                }
            }

            for state in 0..5usize {
                if (case + state as u32) % 2 == 0 {
                    let first = (case + state as u32) % 7;
                    nwa.set_final_weight(states[state], tokens([first]));
                }
            }

            // Exercise generic fallback as well: a transition into this state
            // must retain epsilon closure, while the remaining direct targets
            // may use the optimized path.
            if case % 4 == 0 {
                nwa.add_epsilon(states[1], states[4], tokens([case % 6]));
            }

            let fast = determinize_impl(&nwa, true).unwrap();
            let generic = determinize_impl(&nwa, false).unwrap();
            assert_eq!(
                bincode::serialize(&fast).unwrap(),
                bincode::serialize(&generic).unwrap(),
                "case {case}",
            );
        }
    }

    #[test]
    fn point_entry_aggregation_matches_generic_determinization() {
        let mut nwa = NWA::new(1, 8);
        let start = nwa.add_state();
        let first_accept = nwa.add_state();
        let second_accept = nwa.add_state();
        nwa.set_start_states(vec![start]);
        nwa.add_transition(start, 5, first_accept, tokens([0, 2]));
        nwa.add_transition(start, 5, first_accept, tokens([1, 3]));
        nwa.add_transition(start, 5, second_accept, tokens([2, 4]));
        nwa.add_transition(start, 5, second_accept, tokens([5]));
        nwa.set_final_weight(first_accept, tokens([0, 1, 2, 3]));
        nwa.set_final_weight(second_accept, tokens([2, 4, 5]));

        let fast = determinize_impl(&nwa, true).unwrap();
        let generic = determinize_impl(&nwa, false).unwrap();
        assert_eq!(bincode::serialize(&fast).unwrap(), bincode::serialize(&generic).unwrap());
        assert_eq!(fast.eval_word(&[5]), tokens([0, 1, 2, 3, 4, 5]));
    }

    #[test]
    fn direct_single_target_path_matches_generic_determinization() {
        let mut nwa = NWA::new(1, 4);
        let start = nwa.add_state();
        let first_accept = nwa.add_state();
        let second_accept = nwa.add_state();
        nwa.set_start_states(vec![start]);
        nwa.add_transition(start, 7, first_accept, tokens([0, 1]));
        nwa.add_transition(start, 8, second_accept, tokens([2]));
        nwa.set_final_weight(first_accept, tokens([0, 1]));
        nwa.set_final_weight(second_accept, tokens([2]));

        let fast = determinize_impl(&nwa, true).unwrap();
        let generic = determinize_impl(&nwa, false).unwrap();

        assert_eq!(bincode::serialize(&fast).unwrap(), bincode::serialize(&generic).unwrap());
        assert_eq!(fast.eval_word(&[7]), tokens([0, 1]));
        assert_eq!(fast.eval_word(&[8]), tokens([2]));

        let mut multi_destination = NWA::new(1, 4);
        let start = multi_destination.add_state();
        let first_accept = multi_destination.add_state();
        let second_accept = multi_destination.add_state();
        multi_destination.set_start_states(vec![start]);
        multi_destination.add_transition(start, 9, first_accept, tokens([0]));
        multi_destination.add_transition(start, 9, second_accept, tokens([1]));
        multi_destination.set_final_weight(first_accept, tokens([0]));
        multi_destination.set_final_weight(second_accept, tokens([1]));

        let fast = determinize_impl(&multi_destination, true).unwrap();
        let generic = determinize_impl(&multi_destination, false).unwrap();
        assert_eq!(bincode::serialize(&fast).unwrap(), bincode::serialize(&generic).unwrap());
        assert_eq!(fast.eval_word(&[9]), tokens([0, 1]));
    }
}
