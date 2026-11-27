//! Weighted determinization for acyclic NWAs.
//!
//! Cyclic inputs are rejected and must be handled by the caller.

use std::collections::{BTreeMap, VecDeque, hash_map::Entry as HashMapEntry};
use std::time::Instant;
use std::sync::Arc;

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::dwa::DWA;
use super::equivalence::find_difference;
use super::nwa::NWA;
use crate::ds::weight::{SharedTokenSet, ScopedWeightOpCache, Weight, WeightIntersectionIndex};
use crate::GlrMaskError;

const MAX_INDEXED_FINAL_PATH_RANGES: usize = 8;
const MIN_INDEXED_FINAL_WEIGHT_RANGES: usize = 32;

/// Outgoing edges from one NWA state, grouped by the exact shared `Weight`.
///
/// For a path weight `p`, every edge in a group contributes `p ∩ w`, so this
/// computes that intersection once and distributes the immutable result to the
/// group's distinct labels and destinations.
#[derive(Clone)]
struct WeightGroupedEdge {
    label: i32,
    dst: u32,
    /// This source label has exactly one target and that target is epsilon-free.
    /// A one-entry determinized subset can emit it without the label staging map.
    direct_singleton: bool,
}

#[derive(Clone)]
struct WeightGroupedTransitions {
    weight: Weight,
    edges: Vec<WeightGroupedEdge>,
}

fn build_weight_grouped_transitions(nwa: &NWA) -> Vec<Vec<WeightGroupedTransitions>> {
    nwa.states()
        .iter()
        .map(|state| {
            let mut groups = FxHashMap::<usize, usize>::default();
            let mut result = Vec::<WeightGroupedTransitions>::new();
            for (&label, targets) in &state.transitions {
                let label_has_single_target = targets.len() == 1;
                for (dst, weight) in targets {
                    let direct_singleton_edge = label_has_single_target
                        && nwa.states()[*dst as usize].epsilons.is_empty();
                    let key = weight.ptr_key();
                    let index = if let Some(&existing) = groups.get(&key) {
                        existing
                    } else {
                        let created = result.len();
                        groups.insert(key, created);
                        result.push(WeightGroupedTransitions {
                            weight: weight.clone(),
                            edges: Vec::new(),
                        });
                        created
                    };
                    result[index].edges.push(WeightGroupedEdge {
                        label,
                        dst: *dst,
                        direct_singleton: direct_singleton_edge,
                    });
                }
            }
            result
        })
        .collect()
}

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
    if path_weight.outer_range_count() <= max_indexed_path_ranges
        && let Some(index) = final_weight_indices.get(&state_final.ptr_key()) {
            return path_weight.intersection_with_index(index);
        }
    path_weight.intersection(state_final)
}

fn subset_final_weight(
    nwa: &NWA,
    subset_entries: &[(u32, Weight)],
    final_weight_indices: &FxHashMap<usize, WeightIntersectionIndex>,
    max_indexed_path_ranges: usize,
) -> Weight {
    let mut contributions = SmallVec::<[Weight; 4]>::new();
    for (state, path) in subset_entries {
        if let Some(final_weight) = nwa.states()[*state as usize].final_weight.as_ref() {
            let contribution = final_weight_intersection(path, final_weight,
                final_weight_indices, max_indexed_path_ranges);
            if contribution.is_full() { return contribution; }
            if !contribution.is_empty() { contributions.push(contribution); }
        }
    }
    Weight::union_all(contributions.iter())
}

fn seed_start_subset(nwa: &NWA) -> FxHashMap<u32, Weight> {
    let mut start_subset = FxHashMap::default();
    for &state_id in nwa.start_states() {
        union_state_weight(&mut start_subset, state_id, Weight::all());
    }
    start_subset
}

/// Exact determinization for an NWA whose structural accepted language has
/// labelled depth at most two.
///
/// Rather than constructing residual subsets, evaluate the accepted weight of
/// every word of length 0, 1, or 2 directly. The resulting prefix DWA stores
/// the exact one-label word weight as the depth-1 final weight, and the union
/// of all accepted words below a first label on its start edge. Two-label word
/// weights live on the second edge into one shared all-final leaf.
///
/// The caller must establish the structural depth bound. This function keeps
/// an acyclicity check because the reverse language DP requires a DAG.
pub fn determinize_depth2(nwa: &NWA) -> Result<DWA, GlrMaskError> {
    if !nwa.is_acyclic() {
        return Err(GlrMaskError::Compilation(
            "depth-2 weighted determinization supports only acyclic NWAs".into(),
        ));
    }

    let profile = determinize_profile_enabled();
    let total_started_at = profile.then(Instant::now);
    type Word = SmallVec<[i32; 2]>;

    fn union_word_weight(language: &mut BTreeMap<Word, Weight>, word: Word, add: Weight) {
        if add.is_empty() {
            return;
        }
        match language.entry(word) {
            std::collections::btree_map::Entry::Occupied(mut occupied) => {
                *occupied.get_mut() = occupied.get().union(&add);
            }
            std::collections::btree_map::Entry::Vacant(vacant) => {
                vacant.insert(add);
            }
        }
    }

    let mut in_degree = vec![0u32; nwa.states().len()];
    for state in nwa.states() {
        for (dst, _) in &state.epsilons {
            in_degree[*dst as usize] += 1;
        }
        for targets in state.transitions.values() {
            for (dst, _) in targets {
                in_degree[*dst as usize] += 1;
            }
        }
    }
    let mut queue = VecDeque::new();
    for (state_id, degree) in in_degree.iter().enumerate() {
        if *degree == 0 {
            queue.push_back(state_id);
        }
    }
    let mut topo_order = Vec::with_capacity(nwa.states().len());
    while let Some(state_id) = queue.pop_front() {
        topo_order.push(state_id);
        let state = &nwa.states()[state_id];
        for (dst, _) in &state.epsilons {
            in_degree[*dst as usize] -= 1;
            if in_degree[*dst as usize] == 0 {
                queue.push_back(*dst as usize);
            }
        }
        for targets in state.transitions.values() {
            for (dst, _) in targets {
                in_degree[*dst as usize] -= 1;
                if in_degree[*dst as usize] == 0 {
                    queue.push_back(*dst as usize);
                }
            }
        }
    }
    debug_assert_eq!(topo_order.len(), nwa.states().len());

    let dp_started_at = profile.then(Instant::now);
    let mut languages = vec![BTreeMap::<Word, Weight>::new(); nwa.states().len()];
    let mut weight_cache = ScopedWeightOpCache::default();
    let mut dp_word_contributions = 0usize;
    let mut max_state_words = 0usize;
    for state_id in topo_order.into_iter().rev() {
        let state = &nwa.states()[state_id];
        let mut contributions = BTreeMap::<Word, SmallVec<[Weight; 4]>>::new();
        if let Some(final_weight) = &state.final_weight {
            contributions
                .entry(Word::new())
                .or_default()
                .push(final_weight.clone());
        }
        for (dst, edge_weight) in &state.epsilons {
            for (word, suffix_weight) in &languages[*dst as usize] {
                dp_word_contributions += 1;
                let contribution = weight_cache.intersection(edge_weight, suffix_weight);
                if !contribution.is_empty() {
                    contributions
                        .entry(word.clone())
                        .or_default()
                        .push(contribution);
                }
            }
        }
        for (&label, targets) in &state.transitions {
            for (dst, edge_weight) in targets {
                for (suffix, suffix_weight) in &languages[*dst as usize] {
                    if suffix.len() >= 2 {
                        let contribution = weight_cache.intersection(edge_weight, suffix_weight);
                        if !contribution.is_empty() {
                            return Err(GlrMaskError::Compilation(
                                "depth-2 determinization received an accepted path deeper than two labels".into(),
                            ));
                        }
                        continue;
                    }
                    dp_word_contributions += 1;
                    let contribution = weight_cache.intersection(edge_weight, suffix_weight);
                    if contribution.is_empty() {
                        continue;
                    }
                    let mut word = Word::with_capacity(suffix.len() + 1);
                    word.push(label);
                    word.extend_from_slice(suffix);
                    contributions.entry(word).or_default().push(contribution);
                }
            }
        }
        let language = contributions
            .into_iter()
            .filter_map(|(word, weights)| {
                let weight = Weight::union_all(weights.iter());
                (!weight.is_empty()).then_some((word, weight))
            })
            .collect::<BTreeMap<_, _>>();
        max_state_words = max_state_words.max(language.len());
        languages[state_id] = language;
    }
    let dp_ms = dp_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let start_merge_started_at = profile.then(Instant::now);
    let mut accepted = BTreeMap::<Word, Weight>::new();
    for &start_state in nwa.start_states() {
        for (word, weight) in &languages[start_state as usize] {
            union_word_weight(&mut accepted, word.clone(), weight.clone());
        }
    }
    let start_merge_ms = start_merge_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let accepted_word_count = accepted.len();
    let build_started_at = profile.then(Instant::now);
    let mut by_first = BTreeMap::<i32, (Weight, BTreeMap<i32, Weight>)>::new();
    let mut empty_weight = Weight::empty();
    for (word, weight) in accepted {
        match word.as_slice() {
            [] => empty_weight = empty_weight.union(&weight),
            [first] => {
                let entry = by_first
                    .entry(*first)
                    .or_insert_with(|| (Weight::empty(), BTreeMap::new()));
                entry.0 = entry.0.union(&weight);
            }
            [first, second] => {
                let entry = by_first
                    .entry(*first)
                    .or_insert_with(|| (Weight::empty(), BTreeMap::new()));
                entry
                    .1
                    .entry(*second)
                    .and_modify(|existing| *existing = existing.union(&weight))
                    .or_insert(weight);
            }
            _ => unreachable!("depth-2 language contains an overlong word"),
        }
    }

    let mut dwa = DWA::new(0, 0);
    if !empty_weight.is_empty() {
        dwa.set_final_weight(dwa.start_state(), empty_weight);
    }
    let mut shared_leaf = None::<u32>;
    for (first_label, (first_final, second_words)) in by_first {
        let prefix_weight = Weight::union_all(
            std::iter::once(&first_final).chain(second_words.values()),
        );
        let first_state = dwa.add_state();
        dwa.add_transition(dwa.start_state(), first_label, first_state, prefix_weight);
        if !first_final.is_empty() {
            dwa.set_final_weight(first_state, first_final);
        }
        if !second_words.is_empty() {
            let leaf = *shared_leaf.get_or_insert_with(|| {
                let state = dwa.add_state();
                dwa.set_final_weight(state, Weight::all());
                state
            });
            for (second_label, word_weight) in second_words {
                dwa.add_transition(first_state, second_label, leaf, word_weight);
            }
        }
    }
    let build_ms = build_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][determinize_depth2] nwa_states={} dp_word_contributions={} max_state_words={} accepted_words={} intersection_cache_entries={} dp_ms={:.3} start_merge_ms={:.3} build_ms={:.3} total_ms={:.3}",
            nwa.states().len(),
            dp_word_contributions,
            max_state_words,
            accepted_word_count,
            weight_cache.intersection_entry_count(),
            dp_ms,
            start_merge_ms,
            build_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Ok(dwa)
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

type WeightedSubset = SmallVec<[(u32, Weight); 1]>;

struct PreparedLabel {
    edge_weight: Weight,
    subset: WeightedSubset,
}

fn canonicalize(subset: FxHashMap<u32, Weight>) -> WeightedSubset {
    let mut entries: WeightedSubset = subset.into_iter()
        .filter(|(_, weight)| !weight.is_empty()).collect();
    entries.sort_unstable_by_key(|(state, _)| *state);
    entries
}

fn epsilon_closure(nwa: &NWA, mut closure: FxHashMap<u32, Weight>) -> FxHashMap<u32, Weight> {
    if closure.len() == 1 && nwa.states()[*closure.keys().next().unwrap() as usize].epsilons.is_empty() {
        return closure;
    }
    let mut queue: VecDeque<u32> = closure.keys().copied().collect();
    while let Some(state) = queue.pop_front() {
        let weight = closure[&state].clone();
        for (target, epsilon) in &nwa.states()[state as usize].epsilons {
            let contribution = weight.intersection(epsilon);
            if contribution.is_empty() { continue; }
            match closure.entry(*target) {
                HashMapEntry::Vacant(entry) => { entry.insert(contribution); }
                HashMapEntry::Occupied(mut entry) => {
                    if contribution.is_subset(entry.get()) { continue; }
                    *entry.get_mut() = entry.get().union(&contribution);
                }
            }
            queue.push_back(*target);
        }
    }
    closure
}

/// The exact label image is the union of contributions at each destination.
/// The edge weight records their total domain; the subset describes residual
/// behavior on that domain. Direct epsilon-free singleton normalization is
/// an explicit representation policy, independent of the work schedule.
fn prepare_label(
    nwa: &NWA,
    mut contributions: WeightedSubset,
    direct_enabled: bool,
    normalize_singletons: bool,
) -> Option<PreparedLabel> {
    if contributions.is_empty() { return None; }
    let direct = direct_enabled && contributions.iter()
        .all(|(dst, _)| nwa.states()[*dst as usize].epsilons.is_empty());
    let (mut subset, edge_weight) = if contributions.len() == 1 {
        let edge_weight = contributions[0].1.clone();
        (contributions, edge_weight)
    } else if direct && contributions.iter()
        .all(|(_, weight)| weight.single_tsid_shared_entry().is_some()) {
        let (entries, edge) = aggregate_direct_point_entries(contributions);
        (SmallVec::from_vec(entries), edge)
    } else {
        contributions.sort_unstable_by_key(|(dst, _)| *dst);
        let mut entries = WeightedSubset::new();
        let mut start = 0;
        while start < contributions.len() {
            let dst = contributions[start].0;
            let mut end = start + 1;
            while end < contributions.len() && contributions[end].0 == dst { end += 1; }
            let weight = if end == start + 1 {
                contributions[start].1.clone()
            } else {
                Weight::union_all(contributions[start..end].iter().map(|(_, weight)| weight))
            };
            if !weight.is_empty() { entries.push((dst, weight)); }
            start = end;
        }
        let edge = Weight::union_all(entries.iter().map(|(_, weight)| weight));
        (entries, edge)
    };
    if edge_weight.is_empty() { return None; }
    if direct {
        if normalize_singletons && subset.len() == 1 {
            subset[0].1 = Weight::all();
        }
    } else {
        let closure = epsilon_closure(nwa, subset.into_iter().collect());
        let complement = edge_weight.complement();
        subset = if complement.is_empty() {
            canonicalize(closure)
        } else {
            // Outside the incoming edge domain the residual is don't-care.
            // Preserve the historical normalized generic subset semantics.
            canonicalize(closure.into_iter().map(|(state, weight)| {
                (state, weight.union(&complement))
            }).collect())
        };
    }
    (!subset.is_empty()).then_some(PreparedLabel { edge_weight, subset })
}

/// One exact structural interner, with the common one-entry subset indexed
/// by pointer identity. Retain the weight in the front-cache value even
/// when a structurally equal key was already interned under another pointer.
struct SubsetInterner {
    keys: Vec<Arc<[(u32, Weight)]>>,
    by_key: FxHashMap<Arc<[(u32, Weight)]>, u32>,
    singleton: FxHashMap<(u32, usize), (Weight, u32)>,
}

impl SubsetInterner {
    fn new(start: WeightedSubset) -> Self {
        let start: Arc<[(u32, Weight)]> = start.into_vec().into();
        let mut by_key = FxHashMap::default();
        by_key.insert(Arc::clone(&start), 0);
        Self { keys: vec![start], by_key, singleton: FxHashMap::default() }
    }

    fn intern(&mut self, key: WeightedSubset, dwa: &mut DWA) -> u32 {
        let single = match key.as_slice() {
            [(state, weight)] => Some(((*state, weight.ptr_key()), weight.clone())),
            _ => None,
        };
        if let Some((pointer, _)) = &single {
            if let Some((_, existing)) = self.singleton.get(pointer) { return *existing; }
        }
        let id = if let Some(&existing) = self.by_key.get(key.as_slice()) {
            existing
        } else {
            let id = dwa.add_state();
            debug_assert_eq!(id as usize, self.keys.len());
            let key: Arc<[(u32, Weight)]> = key.into_vec().into();
            self.by_key.insert(Arc::clone(&key), id);
            self.keys.push(key);
            id
        };
        if let Some((pointer, weight)) = single {
            self.singleton.insert(pointer, (weight, id));
        }
        id
    }

    fn emit(&mut self, dwa: &mut DWA, from: u32, label: i32, prepared: PreparedLabel) {
        let to = self.intern(prepared.subset, dwa);
        dwa.add_transition(from, label, to, prepared.edge_weight);
    }
}


fn determinize_profile_enabled() -> bool {
    std::env::var("GLRMASK_PROFILE_DETERMINIZE")
        .map(|value| value == "1")
        .unwrap_or(false)
}

/// Workspace union builders must share this choice with generic determinization.
/// Raw weighted language is invariant under normalization, but a parser's
/// structural DEFAULT_LABEL fallback can distinguish the resulting rows.
#[doc(hidden)]
pub fn normalize_direct_singletons_enabled(nwa_states: usize) -> bool {
    if std::env::var_os("GLRMASK_DISABLE_DETERMINIZE_NORMALIZE_SINGLETONS").is_some() {
        return false;
    }
    if std::env::var_os("GLRMASK_DETERMINIZE_NORMALIZE_SINGLETONS").is_some() {
        return true;
    }
    let min_states = std::env::var("GLRMASK_DETERMINIZE_NORMALIZE_SINGLETON_MIN_STATES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(16_384);
    nwa_states >= min_states
}

pub fn determinize(nwa: &NWA) -> Result<DWA, GlrMaskError> {
    let profile = determinize_profile_enabled();
    let normalize_direct_singletons = normalize_direct_singletons_enabled(nwa.states().len());
    let dwa = determinize_impl_with_options(
        nwa,
        true,
        true,
        profile,
        normalize_direct_singletons,
    )?;

    if normalize_direct_singletons
        && std::env::var_os("GLRMASK_VALIDATE_DETERMINIZE_NORMALIZE_SINGLETONS").is_some()
    {
        let reference = determinize_impl_with_options(nwa, true, true, false, false)?;
        if let Some(word) = find_difference(&dwa, &reference)? {
            return Err(GlrMaskError::Compilation(format!(
                "normalized-singleton determinization differs on labels {word:?}"
            )));
        }
        if profile {
            eprintln!("[glrmask/profile][determinize_normalized_singleton_equivalence] result=equivalent");
        }
    }

    if std::env::var_os("GLRMASK_ASSERT_GROUPED_DETERMINIZE_EQUIVALENCE").is_some() {
        let reference = determinize_impl_with_options(
            nwa,
            true,
            false,
            false,
            normalize_direct_singletons,
        )?;
        match find_difference(&dwa, &reference)? {
            Some(word) => {
                return Err(GlrMaskError::Compilation(format!(
                    "grouped-weight determinization differs from the ordinary path on labels {word:?}"
                )));
            }
            None if profile => eprintln!(
                "[glrmask/profile][determinize_grouped_weight_equivalence] result=equivalent"
            ),
            None => {}
        }
    }

    Ok(dwa)
}

fn determinize_impl(
    nwa: &NWA,
    direct_single_target_enabled: bool,
) -> Result<DWA, GlrMaskError> {
    determinize_impl_with_options(
        nwa,
        direct_single_target_enabled,
        true,
        determinize_profile_enabled(),
        normalize_direct_singletons_enabled(nwa.states().len()),
    )
}

fn determinize_impl_with_options(
    nwa: &NWA,
    direct_single_target_enabled: bool,
    group_transition_weights: bool,
    profile: bool,
    normalize_direct_singletons: bool,
) -> Result<DWA, GlrMaskError> {
    if !nwa.is_acyclic() {
        return Err(GlrMaskError::Compilation(
            "weighted determinization currently supports only acyclic NWAs".into(),
        ));
    }
    let started = Instant::now();
    let grouped = group_transition_weights.then(|| build_weight_grouped_transitions(nwa));
    let mut dwa = DWA::new(0, 0);
    let start = canonicalize(epsilon_closure(nwa, seed_start_subset(nwa)));
    if start.is_empty() { return Ok(dwa); }
    let mut subsets = SubsetInterner::new(start);
    let mut next_pending = 0usize;
    let mut raw_targets = FxHashMap::<i32, WeightedSubset>::default();
    let mut cache = ScopedWeightOpCache::default();

    // Schedule ready subsets in waves when there is enough independent work.
    // Both the direct emission and batched paths use exactly prepare_label.
    let parallel_waves = std::env::var_os("GLRMASK_DISABLE_DETERMINIZE_PARALLEL_DIRECT_WAVES").is_none()
        && rayon::current_num_threads() > 1
        && (std::env::var_os("GLRMASK_DETERMINIZE_PARALLEL_DIRECT_WAVES").is_some()
            || nwa.states().len() >= 128);
    let parallel_labels = rayon::current_num_threads() > 1
        && (parallel_waves || std::env::var_os("GLRMASK_DETERMINIZE_PARALLEL_DIRECT_LABELS").is_some());
    let wave_max = std::env::var("GLRMASK_DETERMINIZE_WAVE_MAX").ok()
        .and_then(|v| v.trim().parse::<usize>().ok()).filter(|&v| v > 0).unwrap_or(1_000);
    let mut transition_visits = 0usize;
    let mut group_visits = 0usize;
    let mut direct_labels = 0usize;
    let mut contributions = 0usize;
    let mut max_subset_entries = 0usize;

    while next_pending < subsets.keys.len() {
        let end = if parallel_waves {
            subsets.keys.len().min(next_pending.saturating_add(wave_max))
        } else {
            next_pending + 1
        };
        let mut batches = Vec::new();
        while next_pending < end {
            let from = next_pending as u32;
            let subset = Arc::clone(&subsets.keys[next_pending]);
            next_pending += 1;
            let inline = direct_single_target_enabled && subset.len() == 1 && !parallel_waves;
            if profile { max_subset_entries = max_subset_entries.max(subset.len()); }
            if let Some(grouped) = &grouped {
                for (state, path) in subset.iter() {
                    for group in &grouped[*state as usize] {
                        if profile {
                            group_visits += 1;
                            transition_visits += group.edges.len();
                        }
                        let contribution = cache.intersection(path, &group.weight);
                        if contribution.is_empty() { continue; }
                        for edge in &group.edges {
                            if profile { contributions += 1; }
                            if inline && edge.direct_singleton {
                                // A unique source label cannot receive any
                                // other contribution in this subset.
                                let prepared = prepare_label(nwa,
                                    smallvec::smallvec![(edge.dst, contribution.clone())],
                                    direct_single_target_enabled, normalize_direct_singletons,
                                ).expect("nonempty direct singleton");
                                subsets.emit(&mut dwa, from, edge.label, prepared);
                                if profile { direct_labels += 1; }
                            } else {
                                raw_targets.entry(edge.label).or_default()
                                    .push((edge.dst, contribution.clone()));
                            }
                        }
                    }
                }
            } else {
                for (state, path) in subset.iter() {
                    for (&label, targets) in &nwa.states()[*state as usize].transitions {
                        for (target, weight) in targets {
                            if profile { transition_visits += 1; }
                            let contribution = cache.intersection(path, weight);
                            if !contribution.is_empty() {
                                if profile { contributions += 1; }
                                raw_targets.entry(label).or_default().push((*target, contribution));
                            }
                        }
                    }
                }
            }
            batches.extend(raw_targets.drain().map(|(label, targets)| (from, label, targets)));
        }
        let prepare = |(from, label, targets)| {
            prepare_label(nwa, targets, direct_single_target_enabled, normalize_direct_singletons)
                .map(|prepared| (from, label, prepared))
        };
        // Preserve batch order when publishing, regardless of which workers
        // finished first. Newly interned subsets join the next ready wave.
        if parallel_labels && batches.len() >= 2 {
            let prepared = batches.into_par_iter().map(prepare).collect::<Vec<_>>();
            for (from, label, prepared) in prepared.into_iter().flatten() {
                subsets.emit(&mut dwa, from, label, prepared);
            }
        } else {
            for (from, label, prepared) in batches.into_iter().filter_map(prepare) {
                subsets.emit(&mut dwa, from, label, prepared);
            }
        }
    }

    // Final weights are a second independent reduction of the same residual
    // subset; no alternate profiled evaluator or duplicate subset traversal.
    let final_weight_indices: FxHashMap<usize, WeightIntersectionIndex> = nwa.states().iter()
        .filter_map(|state| state.final_weight.as_ref())
        .filter(|weight| weight.outer_range_count() >= MIN_INDEXED_FINAL_WEIGHT_RANGES)
        .map(|weight| (weight.ptr_key(), weight.intersection_index())).collect();
    let final_started = Instant::now();
    let final_weights = subsets.keys.par_iter().map(|entries| {
        subset_final_weight(nwa, entries, &final_weight_indices, MAX_INDEXED_FINAL_PATH_RANGES)
    }).collect::<Vec<_>>();
    for (state, weight) in final_weights.into_iter().enumerate() {
        if !weight.is_empty() { dwa.set_final_weight(state as u32, weight); }
    }
    if profile {
        eprintln!(
            "[glrmask/profile][determinize] nwa_states={} dwa_states={} max_subset_entries={} raw_transition_visits={} weight_group_visits={} direct_singleton_labels={} target_contributions={} intersection_cache_entries={} final_weights_ms={:.3} total_ms={:.3}",
            nwa.states().len(), dwa.states().len(), max_subset_entries, transition_visits,
            group_visits, direct_labels, contributions, cache.intersection_entry_count(),
            final_started.elapsed().as_secs_f64() * 1000.0,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(dwa)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::weighted_u32::equivalence::find_difference;
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
        let grouped = determinize_impl_with_options(&nwa, true, true, false, false).unwrap();
        assert_eq!(find_difference(&fast, &generic).unwrap(), None);
        assert_eq!(find_difference(&grouped, &generic).unwrap(), None);
        assert_eq!(fast.eval_word(&[7]), tokens([0, 1]));
    }

    #[test]
    fn depth2_language_dp_matches_generic_with_epsilon_paths() {
        let mut nwa = NWA::new(1, 4);
        let start = nwa.add_state();
        let left = nwa.add_state();
        let right = nwa.add_state();
        let middle = nwa.add_state();
        let accept = nwa.add_state();
        nwa.set_start_states(vec![start]);

        nwa.add_epsilon(start, left, tokens([0, 1, 2]));
        nwa.add_epsilon(start, right, tokens([1, 2, 3]));
        nwa.add_transition(left, 7, middle, tokens([0, 1]));
        nwa.add_transition(right, 7, middle, tokens([2, 3]));
        nwa.set_final_weight(middle, tokens([0, 2]));
        nwa.add_transition(middle, 8, accept, tokens([1, 2, 3]));
        nwa.set_final_weight(accept, tokens([1, 2]));

        let depth2 = determinize_depth2(&nwa).unwrap();
        let generic = determinize(&nwa).unwrap();
        assert_eq!(find_difference(&depth2, &generic).unwrap(), None);
        assert_eq!(depth2.eval_word(&[7]), tokens([0, 2]));
        assert_eq!(depth2.eval_word(&[7, 8]), tokens([1, 2]));
    }

    #[test]
    fn depth2_language_dp_rejects_accepted_three_label_path() {
        let mut nwa = NWA::new(1, 1);
        let states: Vec<u32> = (0..4).map(|_| nwa.add_state()).collect();
        nwa.set_start_states(vec![states[0]]);
        nwa.add_transition(states[0], 1, states[1], tokens([0]));
        nwa.add_transition(states[1], 2, states[2], tokens([0]));
        nwa.add_transition(states[2], 3, states[3], tokens([0]));
        nwa.set_final_weight(states[3], tokens([0]));

        let error = determinize_depth2(&nwa).unwrap_err();
        assert!(error.to_string().contains("deeper than two labels"), "{error}");
    }

    #[test]
    fn repeated_single_destination_uses_the_same_normalization_policy() {
        let mut nwa = NWA::new(1, 4);
        let start = nwa.add_state();
        let accept = nwa.add_state();
        nwa.set_start_states(vec![start]);
        nwa.add_transition(start, 7, accept, tokens([0]));
        nwa.add_transition(start, 7, accept, tokens([1]));
        nwa.set_final_weight(accept, tokens([0, 1, 2]));

        let normalized = determinize_impl_with_options(&nwa, true, true, false, true).unwrap();
        let direct = determinize_impl_with_options(&nwa, true, true, false, false).unwrap();
        assert_eq!(find_difference(&normalized, &direct).unwrap(), None);
        let target = normalized.states()[0].transitions.get(&7).unwrap().0;
        assert_eq!(normalized.states()[target as usize].final_weight.as_ref(), Some(&tokens([0, 1, 2])));
        assert_eq!(normalized.eval_word(&[7]), tokens([0, 1]));
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
                    if (case + (from * 7 + to * 11) as u32).is_multiple_of(3) {
                        continue;
                    }
                    let label = ((case + (from * 3 + to) as u32) % 4) as i32;
                    let first = (case + (from * 5 + to) as u32) % 6;
                    let second = (first + 1 + case % 3) % 7;
                    nwa.add_transition(states[from], label, states[to], tokens([first, second]));
                    if (case + from as u32 + to as u32).is_multiple_of(5) {
                        let extra = (second + 2) % 8;
                        nwa.add_transition(states[from], label, states[to], tokens([extra]));
                    }
                }
            }

            for (state, &state_id) in states.iter().enumerate().take(5) {
                if (case + state as u32).is_multiple_of(2) {
                    let first = (case + state as u32) % 7;
                    nwa.set_final_weight(state_id, tokens([first]));
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
            let grouped = determinize_impl_with_options(&nwa, true, true, false, false).unwrap();
            assert_eq!(
                find_difference(&fast, &generic).unwrap(),
                None,
                "case {case}",
            );
            assert_eq!(
                find_difference(&grouped, &generic).unwrap(),
                None,
                "grouped case {case}",
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
        let grouped = determinize_impl_with_options(&nwa, true, true, false, false).unwrap();
        assert_eq!(find_difference(&fast, &generic).unwrap(), None);
        assert_eq!(find_difference(&grouped, &generic).unwrap(), None);
        assert_eq!(fast.eval_word(&[5]), tokens([0, 1, 2, 3, 4, 5]));
    }

    #[test]
    fn grouped_singleton_edges_mix_direct_and_epsilon_fallback_exactly() {
        let mut nwa = NWA::new(1, 4);
        let start = nwa.add_state();
        let first_accept = nwa.add_state();
        let second_accept = nwa.add_state();
        let epsilon_source = nwa.add_state();
        let epsilon_accept = nwa.add_state();
        nwa.set_start_states(vec![start]);

        // The three transition labels share one exact Weight group. Two are
        // directly emit-able; the third must keep the epsilon-closure fallback.
        let shared = tokens([0, 1]);
        nwa.add_transition(start, 7, first_accept, shared.clone());
        nwa.add_transition(start, 8, second_accept, shared.clone());
        nwa.add_transition(start, 9, epsilon_source, shared);
        nwa.add_epsilon(epsilon_source, epsilon_accept, tokens([0, 1]));
        nwa.set_final_weight(first_accept, tokens([0, 1]));
        nwa.set_final_weight(second_accept, tokens([0, 1]));
        nwa.set_final_weight(epsilon_accept, tokens([0, 1]));

        let grouped = determinize_impl_with_options(&nwa, true, true, false, false).unwrap();
        let generic = determinize_impl_with_options(&nwa, false, false, false, false).unwrap();
        assert_eq!(find_difference(&grouped, &generic).unwrap(), None);
        assert_eq!(grouped.eval_word(&[7]), tokens([0, 1]));
        assert_eq!(grouped.eval_word(&[8]), tokens([0, 1]));
        assert_eq!(grouped.eval_word(&[9]), tokens([0, 1]));
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

        assert_eq!(find_difference(&fast, &generic).unwrap(), None);
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
        assert_eq!(find_difference(&fast, &generic).unwrap(), None);
        assert_eq!(fast.eval_word(&[9]), tokens([0, 1]));
    }
}
