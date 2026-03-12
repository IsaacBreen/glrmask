//! IMPORTANT: this should only be implemented for **acyclic** weighted
//! automata. Cyclic input should panic rather than trying to determinize.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, VecDeque};
use std::hash::Hasher;

use rustc_hash::FxHashMap;

use super::dwa::DWA;
use super::nwa::NWA;
use crate::ds::weight::{Weight, WeightBuilder};
use crate::GlrMaskError;

type WeightId = u32;

/// Like TransitionWeight in minimize: stores either a single weight or
/// multiple merged via WeightBuilder. Avoids the expand/compress cycle
/// when there's only one contribution.
enum RawTargetWeight {
    Single(Weight),
    Merged(WeightBuilder),
}

impl RawTargetWeight {
    fn add(&mut self, weight: Weight) {
        match self {
            RawTargetWeight::Single(existing) => {
                let mut builder = WeightBuilder::new();
                builder.union_weight(existing);
                builder.union_weight(&weight);
                *self = RawTargetWeight::Merged(builder);
            }
            RawTargetWeight::Merged(builder) => {
                builder.union_weight(&weight);
            }
        }
    }

    fn build(self) -> Weight {
        match self {
            RawTargetWeight::Single(w) => w,
            RawTargetWeight::Merged(builder) => builder.build(),
        }
    }
}

#[derive(Default)]
struct DeterminizeProfile {
    subsets_processed: usize,
    epsilon_closure_calls: usize,
    epsilon_closure_seed_states: usize,
    epsilon_closure_output_states: usize,
    raw_target_labels: usize,
    raw_target_contributions: usize,
    raw_target_edges: usize,
    raw_target_collisions: usize,
    final_weight_ms: std::time::Duration,
    raw_targets_ms: std::time::Duration,
    epsilon_closure_ms: std::time::Duration,
    edge_weight_ms: std::time::Duration,
    normalize_ms: std::time::Duration,
    subset_lookup_ms: std::time::Duration,
}

#[derive(Default)]
struct LocalWeightInterner {
    weights: Vec<Weight>,
    buckets: FxHashMap<u64, Vec<WeightId>>,
}

impl LocalWeightInterner {
    fn intern(&mut self, weight: &Weight) -> WeightId {
        let fingerprint = weight_fingerprint(weight);
        let bucket = self.buckets.entry(fingerprint).or_default();
        for &weight_id in bucket.iter() {
            if self.weights[weight_id as usize] == *weight {
                return weight_id;
            }
        }

        let weight_id = self.weights.len() as WeightId;
        self.weights.push(weight.clone());
        bucket.push(weight_id);
        weight_id
    }

    fn intern_subset_key(&mut self, subset: &[(u32, Weight)]) -> Vec<(u32, WeightId)> {
        subset
            .iter()
            .map(|(state_id, weight)| (*state_id, self.intern(weight)))
            .collect()
    }
}

fn weight_fingerprint(weight: &Weight) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    if weight.is_full() {
        hasher.write_u8(1);
        return hasher.finish();
    }

    hasher.write_u8(0);
    for (range, tokens) in weight.0.range_values() {
        hasher.write_u32(*range.start());
        hasher.write_u32(*range.end());
        for token_range in tokens.ranges() {
            hasher.write_u32(*token_range.start());
            hasher.write_u32(*token_range.end());
        }
        hasher.write_u8(0xff);
    }
    hasher.finish()
}

pub fn determinize(nwa: &NWA) -> Result<DWA, GlrMaskError> {
    if !nwa.is_acyclic() {
        return Err(GlrMaskError::Compilation(
            "weighted determinization currently supports only acyclic NWAs".into(),
        ));
    }

    fn canonicalize(subset: &BTreeMap<u32, Weight>) -> Vec<(u32, Weight)> {
        subset
            .iter()
            .filter_map(|(&state_id, weight)| (!weight.is_empty()).then_some((state_id, weight.clone())))
            .collect()
    }

    fn epsilon_closure(nwa: &NWA, seed: &BTreeMap<u32, Weight>) -> BTreeMap<u32, Weight> {
        let mut closure = seed.clone();
        let mut queue: VecDeque<u32> = seed.keys().copied().collect();

        while let Some(state_id) = queue.pop_front() {
            let Some(current_weight) = closure.get(&state_id).cloned() else {
                continue;
            };
            let Some(state) = nwa.states.get(state_id as usize) else {
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
    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_WEIGHTED_DETERMINIZE").is_some();
    let mut profile = profile_enabled.then(DeterminizeProfile::default);
    let mut weight_interner = LocalWeightInterner::default();

    let mut dwa = DWA::new(0, 0);
    let start_id = dwa.start_state;

    let mut start_subset = BTreeMap::new();
    for &state_id in &nwa.start_states {
        let existing = start_subset
            .get(&state_id)
            .cloned()
            .unwrap_or_else(Weight::empty);
        start_subset.insert(state_id, existing.union(&Weight::all()));
    }
    let start_closure_started_at = profile_enabled.then(std::time::Instant::now);
    let start_subset = epsilon_closure(nwa, &start_subset);
    if let (Some(profile), Some(started_at)) = (profile.as_mut(), start_closure_started_at) {
        profile.epsilon_closure_calls += 1;
        profile.epsilon_closure_seed_states += nwa.start_states.len();
        profile.epsilon_closure_output_states += start_subset.len();
        profile.epsilon_closure_ms += started_at.elapsed();
    }

    if start_subset.is_empty() {
        return Ok(dwa);
    }

    let mut subset_map: FxHashMap<Vec<(u32, WeightId)>, u32> = FxHashMap::default();
    let mut worklist: VecDeque<(Vec<(u32, WeightId)>, Vec<(u32, Weight)>)> = VecDeque::new();
    let start_entries = canonicalize(&start_subset);
    let start_key = weight_interner.intern_subset_key(&start_entries);
    subset_map.insert(start_key.clone(), start_id);
    worklist.push_back((start_key, start_entries));

    while let Some((subset_key_ids, subset_entries)) = worklist.pop_front() {
        if let Some(profile) = profile.as_mut() {
            profile.subsets_processed += 1;
        }
        let from_state = subset_map[&subset_key_ids];

        let final_weight_started_at = profile_enabled.then(std::time::Instant::now);
        let mut final_weight = Weight::empty();
        for (nwa_state_id, path_weight) in &subset_entries {
            if let Some(state_final) = nwa.states[*nwa_state_id as usize].final_weight.as_ref() {
                final_weight = final_weight.union(&path_weight.intersection(state_final));
            }
        }
        if let (Some(profile), Some(started_at)) = (profile.as_mut(), final_weight_started_at) {
            profile.final_weight_ms += started_at.elapsed();
        }
        if !final_weight.is_empty() {
            dwa.set_final_weight(from_state, final_weight);
        }

        let mut raw_targets: BTreeMap<i32, BTreeMap<u32, RawTargetWeight>> = BTreeMap::new();

        let raw_targets_started_at = profile_enabled.then(std::time::Instant::now);
        for (nwa_state_id, path_weight) in &subset_entries {
            let state = &nwa.states[*nwa_state_id as usize];
            for (&label, targets) in &state.transitions {
                for (dst, trans_weight) in targets {
                    let next_weight = path_weight.intersection(trans_weight);
                    if next_weight.is_empty() {
                        continue;
                    }

                    if let Some(profile) = profile.as_mut() {
                        profile.raw_target_contributions += 1;
                    }

                    let target_entry = raw_targets.entry(label).or_default();
                    match target_entry.entry(*dst) {
                        std::collections::btree_map::Entry::Occupied(mut occupied) => {
                            if let Some(profile) = profile.as_mut() {
                                profile.raw_target_collisions += 1;
                            }
                            occupied.get_mut().add(next_weight);
                        }
                        std::collections::btree_map::Entry::Vacant(vacant) => {
                            vacant.insert(RawTargetWeight::Single(next_weight));
                        }
                    }
                }
            }
        }
        if let (Some(profile), Some(started_at)) = (profile.as_mut(), raw_targets_started_at) {
            profile.raw_targets_ms += started_at.elapsed();
        }

        for (label, target_subset) in raw_targets {
            if target_subset.is_empty() {
                continue;
            }
            let target_subset: BTreeMap<u32, Weight> = target_subset
                .into_iter()
                .filter_map(|(state_id, rtw)| {
                    let w = rtw.build();
                    (!w.is_empty()).then_some((state_id, w))
                })
                .collect();
            if target_subset.is_empty() {
                continue;
            }
            if let Some(profile) = profile.as_mut() {
                profile.raw_target_labels += 1;
                profile.raw_target_edges += target_subset.len();
            }

            let edge_weight_started_at = profile_enabled.then(std::time::Instant::now);
            let edge_weight = Weight::union_all(target_subset.values());
            if let (Some(profile), Some(started_at)) = (profile.as_mut(), edge_weight_started_at) {
                profile.edge_weight_ms += started_at.elapsed();
            }
            if edge_weight.is_empty() {
                continue;
            }

            let closure_started_at = profile_enabled.then(std::time::Instant::now);
            let expanded = epsilon_closure(nwa, &target_subset);
            if let (Some(profile), Some(started_at)) = (profile.as_mut(), closure_started_at) {
                profile.epsilon_closure_calls += 1;
                profile.epsilon_closure_seed_states += target_subset.len();
                profile.epsilon_closure_output_states += expanded.len();
                profile.epsilon_closure_ms += started_at.elapsed();
            }
            if expanded.is_empty() {
                continue;
            }

            let normalize_started_at = profile_enabled.then(std::time::Instant::now);
            let edge_complement = edge_weight.complement();
            let normalized: BTreeMap<u32, Weight> = if edge_complement.is_empty() {
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
            if let (Some(profile), Some(started_at)) = (profile.as_mut(), normalize_started_at) {
                profile.normalize_ms += started_at.elapsed();
            }
            let next_key = canonicalize(&normalized);
            if next_key.is_empty() {
                continue;
            }
            let next_key_ids = weight_interner.intern_subset_key(&next_key);

            let subset_lookup_started_at = profile_enabled.then(std::time::Instant::now);
            let to_state = if let Some(existing) = subset_map.get(&next_key_ids).copied() {
                existing
            } else {
                let new_id = dwa.add_state();
                subset_map.insert(next_key_ids.clone(), new_id);
                worklist.push_back((next_key_ids, next_key));
                new_id
            };
            if let (Some(profile), Some(started_at)) = (profile.as_mut(), subset_lookup_started_at) {
                profile.subset_lookup_ms += started_at.elapsed();
            }

            dwa.add_transition(from_state, label, to_state, edge_weight);
        }
    }

    if let Some(profile) = profile {
        let avg_seed_states = if profile.epsilon_closure_calls == 0 {
            0.0
        } else {
            profile.epsilon_closure_seed_states as f64 / profile.epsilon_closure_calls as f64
        };
        let avg_closure_states = if profile.epsilon_closure_calls == 0 {
            0.0
        } else {
            profile.epsilon_closure_output_states as f64 / profile.epsilon_closure_calls as f64
        };
        eprintln!(
            "[glrmask/profile][weighted_determinize] subsets_processed={} epsilon_closure_calls={} avg_seed_states={:.3} avg_closure_states={:.3} raw_target_labels={} raw_target_contributions={} raw_target_edges={} raw_target_collisions={} final_weight_ms={:.3} raw_targets_ms={:.3} epsilon_closure_ms={:.3} edge_weight_ms={:.3} normalize_ms={:.3} subset_lookup_ms={:.3}",
            profile.subsets_processed,
            profile.epsilon_closure_calls,
            avg_seed_states,
            avg_closure_states,
            profile.raw_target_labels,
            profile.raw_target_contributions,
            profile.raw_target_edges,
            profile.raw_target_collisions,
            profile.final_weight_ms.as_secs_f64() * 1000.0,
            profile.raw_targets_ms.as_secs_f64() * 1000.0,
            profile.epsilon_closure_ms.as_secs_f64() * 1000.0,
            profile.edge_weight_ms.as_secs_f64() * 1000.0,
            profile.normalize_ms.as_secs_f64() * 1000.0,
            profile.subset_lookup_ms.as_secs_f64() * 1000.0,
        );
    }

    Ok(dwa)
}
