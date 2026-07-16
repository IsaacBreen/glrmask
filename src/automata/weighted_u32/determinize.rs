//! Weighted determinization for acyclic NWAs.
//!
//! Cyclic inputs are rejected and must be handled by the caller.

use std::collections::{VecDeque, hash_map::Entry as HashMapEntry};
use std::hash::{Hash, Hasher};
use std::time::Instant;
use std::sync::Arc;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
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
    subset_map: &mut FxHashMap<Arc<[(u32, Weight)]>, u32>,
    worklist: &mut VecDeque<(u32, Arc<[(u32, Weight)]>)>,
    dwa: &mut DWA,
) -> u32 {
    if let Some(existing) = subset_map.get(next_key.as_slice()).copied() {
        return existing;
    }

    let new_id = dwa.add_state();
    let shared_key: Arc<[(u32, Weight)]> = next_key.into();
    subset_map.insert(Arc::clone(&shared_key), new_id);
    worklist.push_back((new_id, shared_key));
    new_id
}

/// Intern a one-entry subset through a pointer-identity front cache. The
/// structural `subset_map` remains authoritative on every cache miss.
fn intern_determinized_singleton(
    dst: u32,
    weight: &Weight,
    singleton_subsets: &mut FxHashMap<(u32, usize), u32>,
    subset_map: &mut FxHashMap<Arc<[(u32, Weight)]>, u32>,
    worklist: &mut VecDeque<(u32, Arc<[(u32, Weight)]>)>,
    dwa: &mut DWA,
) -> (u32, bool) {
    let singleton_key = (dst, weight.ptr_key());
    if let Some(existing) = singleton_subsets.get(&singleton_key).copied() {
        return (existing, true);
    }

    let state = intern_determinized_subset(
        vec![(dst, weight.clone())],
        subset_map,
        worklist,
        dwa,
    );
    singleton_subsets.insert(singleton_key, state);
    (state, false)
}

#[derive(Default)]
struct ScopedMultiwayUnionCache {
    entries: FxHashMap<Box<[Weight]>, Weight>,
    hits: usize,
    misses: usize,
}

impl ScopedMultiwayUnionCache {
    fn union_all<'a>(&mut self, weights: impl IntoIterator<Item = &'a Weight>) -> Weight {
        let mut operands = SmallVec::<[Weight; 8]>::new();
        for weight in weights {
            if weight.is_full() {
                return Weight::all();
            }
            if !weight.is_empty() {
                operands.push(weight.clone());
            }
        }
        match operands.len() {
            0 => return Weight::empty(),
            1 => return operands.pop().unwrap(),
            _ => {}
        }

        operands.sort_unstable_by_key(Weight::ptr_key);
        operands.dedup_by_key(|weight| weight.ptr_key());
        if operands.len() == 1 {
            return operands.pop().unwrap();
        }

        if let Some(existing) = self.entries.get(operands.as_slice()) {
            self.hits += 1;
            return existing.clone();
        }
        self.misses += 1;
        let key = operands.into_vec().into_boxed_slice();
        let result = Weight::union_all_direct(key.iter());
        self.entries.insert(key, result.clone());
        result
    }
}

fn determinize_profile_enabled() -> bool {
    std::env::var("GLRMASK_PROFILE_DETERMINIZE")
        .map(|value| value == "1")
        .unwrap_or(false)
}

fn nwa_topological_order(nwa: &NWA) -> Option<Vec<usize>> {
    let n = nwa.states().len();
    let mut indegree = vec![0usize; n];
    for state in nwa.states() {
        for (target, _) in state.transitions.values().flatten() {
            let target = *target as usize;
            if target >= n {
                return None;
            }
            indegree[target] += 1;
        }
        for (target, _) in &state.epsilons {
            let target = *target as usize;
            if target >= n {
                return None;
            }
            indegree[target] += 1;
        }
    }

    let mut topo: Vec<usize> = indegree
        .iter()
        .enumerate()
        .filter_map(|(state_id, &degree)| (degree == 0).then_some(state_id))
        .collect();
    let mut head = 0usize;
    while head < topo.len() {
        let state_id = topo[head];
        head += 1;
        for (target, _) in nwa.states()[state_id].transitions.values().flatten() {
            let target = *target as usize;
            indegree[target] -= 1;
            if indegree[target] == 0 {
                topo.push(target);
            }
        }
        for (target, _) in &nwa.states()[state_id].epsilons {
            let target = *target as usize;
            indegree[target] -= 1;
            if indegree[target] == 0 {
                topo.push(target);
            }
        }
    }
    (topo.len() == n).then_some(topo)
}

/// Exact token domains from which each NWA state can accept some suffix.
fn nwa_future_live_domains(nwa: &NWA) -> Option<Vec<Weight>> {
    let topo = nwa_topological_order(nwa)?;
    let mut future_live = vec![Weight::empty(); nwa.states().len()];
    let mut cache = ScopedWeightOpCache::default();

    for &state_id in topo.iter().rev() {
        let state = &nwa.states()[state_id];
        let mut live_parts = Vec::<Weight>::new();

        if let Some(final_weight) = state.final_weight.as_ref() {
            if !final_weight.is_empty() {
                live_parts.push(final_weight.clone());
            }
        }

        for branches in state.transitions.values() {
            for (target, weight) in branches {
                let target_live = &future_live[*target as usize];
                let contribution = if target_live.is_full() {
                    weight.clone()
                } else if target_live.is_empty() {
                    Weight::empty()
                } else {
                    cache.intersection(weight, target_live)
                };
                if !contribution.is_empty() {
                    live_parts.push(contribution);
                }
            }
        }

        for (target, weight) in &state.epsilons {
            let target_live = &future_live[*target as usize];
            let live_contribution = if target_live.is_full() {
                weight.clone()
            } else if target_live.is_empty() {
                Weight::empty()
            } else {
                cache.intersection(weight, target_live)
            };
            if !live_contribution.is_empty() {
                live_parts.push(live_contribution);
            }
        }

        future_live[state_id] = Weight::union_all(live_parts.iter());
    }

    Some(future_live)
}

/// The exact path-weight domain observable by one determinization expansion.
///
/// Subsets are epsilon-closed before they enter the worklist. The expansion
/// loop therefore observes one subset entry only by intersecting its path
/// weight with that state's labeled outgoing edge weights. If two path weights
/// agree on the union of those edge weights, every raw labeled contribution is
/// exactly equal. Final weights are deliberately excluded: they are computed
/// independently from the full subset after expansion.
fn nwa_labeled_transition_domains(nwa: &NWA) -> Vec<Weight> {
    nwa.states()
        .iter()
        .map(|state| {
            Weight::union_all(
                state
                    .transitions
                    .values()
                    .flatten()
                    .map(|(_, weight)| weight),
            )
        })
        .collect()
}

fn transition_expansion_key(
    subset_entries: &[(u32, Weight)],
    labeled_transition_domains: &[Weight],
    cache: &mut ScopedWeightOpCache,
) -> Box<[(u32, Weight)]> {
    let mut expansion_key = Vec::<(u32, Weight)>::with_capacity(subset_entries.len());
    for (state_id, path_weight) in subset_entries {
        let domain = &labeled_transition_domains[*state_id as usize];
        let restricted = if domain.is_full() {
            path_weight.clone()
        } else if domain.is_empty() {
            continue;
        } else {
            cache.intersection(path_weight, domain)
        };
        if !restricted.is_empty() {
            expansion_key.push((*state_id, restricted));
        }
    }
    expansion_key.into_boxed_slice()
}

/// Push each NWA edge weight backward through the exact token domain from
/// which its target can still reach acceptance.
///
/// For acyclic NWAs this is a language-preserving trim: a token coordinate
/// outside `future_live[target]` cannot contribute to any accepted suffix from
/// that target, so retaining it on an incoming edge only creates dead path
/// weight distinctions during subset construction.
pub(crate) fn push_nwa_weights_to_future_live(nwa: &mut NWA) -> Option<bool> {
    let future_live = nwa_future_live_domains(nwa)?;
    let mut cache = ScopedWeightOpCache::default();
    let mut changed = false;

    for state in nwa.states_mut() {
        for branches in state.transitions.values_mut() {
            for (target, weight) in branches.iter_mut() {
                let target_live = &future_live[*target as usize];
                let pushed = if target_live.is_full() {
                    weight.clone()
                } else if target_live.is_empty() {
                    Weight::empty()
                } else {
                    cache.intersection(weight, target_live)
                };
                if pushed != *weight {
                    *weight = pushed.clone();
                    changed = true;
                }
            }
            let old_len = branches.len();
            branches.retain(|(_, weight)| !weight.is_empty());
            changed |= branches.len() != old_len;
        }
        state.transitions.retain(|_, branches| !branches.is_empty());

        for (target, weight) in &mut state.epsilons {
            let target_live = &future_live[*target as usize];
            let pushed = if target_live.is_full() {
                weight.clone()
            } else if target_live.is_empty() {
                Weight::empty()
            } else {
                cache.intersection(weight, target_live)
            };
            if pushed != *weight {
                *weight = pushed.clone();
                changed = true;
            }
        }
        let old_epsilon_len = state.epsilons.len();
        state.epsilons.retain(|(_, weight)| !weight.is_empty());
        changed |= state.epsilons.len() != old_epsilon_len;

    }

    Some(changed)
}

pub fn determinize(nwa: &NWA) -> Result<DWA, GlrMaskError> {
    let profile = determinize_profile_enabled();
    let dwa = determinize_impl_with_options(nwa, true, true, profile)?;

    if std::env::var_os("GLRMASK_ASSERT_GROUPED_DETERMINIZE_EQUIVALENCE").is_some() {
        let reference = determinize_impl_with_options(nwa, true, false, false)?;
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
    )
}

fn determinize_impl_with_options(
    nwa: &NWA,
    direct_single_target_enabled: bool,
    group_transition_weights: bool,
    profile: bool,
) -> Result<DWA, GlrMaskError> {
    let cache_expansions = std::env::var_os("GLRMASK_EXPERIMENTAL_DETERMINIZE_EXPANSION_CACHE")
        .is_some();
    determinize_impl_with_options_and_cache(
        nwa,
        direct_single_target_enabled,
        group_transition_weights,
        profile,
        cache_expansions,
    )
}

fn determinize_impl_with_options_and_cache(
    nwa: &NWA,
    direct_single_target_enabled: bool,
    group_transition_weights: bool,
    profile: bool,
    cache_expansions: bool,
) -> Result<DWA, GlrMaskError> {
    if !nwa.is_acyclic() {
        return Err(GlrMaskError::Compilation(
            "weighted determinization currently supports only acyclic NWAs".into(),
        ));
    }

    let profile_expansion_keys = profile
        && std::env::var_os("GLRMASK_PROFILE_DETERMINIZE_EXPANSION_KEYS").is_some();
    let labeled_transition_domains =
        (profile_expansion_keys || cache_expansions).then(|| nwa_labeled_transition_domains(nwa));

    let weight_group_build_started_at = profile.then(Instant::now);
    let weight_grouped_transitions = group_transition_weights.then(|| build_weight_grouped_transitions(nwa));
    let weight_group_build_ms = weight_group_build_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

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

    let mut subset_map: FxHashMap<Arc<[(u32, Weight)]>, u32> = FxHashMap::default();
    // This is an identity-only front cache. A miss always falls back through
    // `subset_map`, which retains structural equality as the source of truth.
    let mut singleton_subsets: FxHashMap<(u32, usize), u32> = FxHashMap::default();
    let mut worklist: VecDeque<(u32, Arc<[(u32, Weight)]>)> = VecDeque::new();
    let start_entries: Arc<[(u32, Weight)]> = canonicalize(&start_subset).into();
    subset_map.insert(Arc::clone(&start_entries), start_id);
    worklist.push_back((start_id, start_entries));

    // Almost every label has one surviving destination. Keep that common
    // case inline instead of allocating a nested hash map and a Vec per label.
    let mut raw_targets: FxHashMap<i32, SmallVec<[(u32, Weight); 1]>> = FxHashMap::default();
    // All operands in the expansion loop remain owned by the NWA or subset_map
    // for the lifetime of this determinization, so a local exact cache avoids
    // thread-local memo overhead on the heavily reused intersection pairs.
    let mut scoped_determinize_weight_cache = ScopedWeightOpCache::default();
    let mut scoped_multiway_union_cache = ScopedMultiwayUnionCache::default();
    let determinize_started_at = profile.then(Instant::now);
    let mut processed_states = 0usize;
    let mut profile_subset_entries = 0usize;
    let mut profile_max_subset_entries = 0usize;
    let mut profile_raw_transition_visits = 0usize;
    let mut profile_weight_group_visits = 0usize;
    let mut profile_labels = 0usize;
    let mut profile_target_contributions = 0usize;
    let mut profile_single_contribution_labels = 0usize;
    let mut profile_single_contribution_no_epsilon_labels = 0usize;
    let mut profile_direct_single_target_labels = 0usize;
    let mut profile_direct_singleton_cache_hits = 0usize;
    let mut profile_direct_singleton_cache_misses = 0usize;
    let mut profile_direct_singleton_fast_path_groups = 0usize;
    let mut profile_direct_singleton_fast_path_labels = 0usize;
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
    let mut profile_subset_len_buckets = [0usize; 8];
    let mut profile_full_path_weights = 0usize;
    let mut profile_topology_variants = FxHashMap::<u64, usize>::default();
    let mut profile_topology_variant_excess = 0usize;
    let mut profile_max_topology_variants = 0usize;
    let mut profile_expansion_weight_cache = ScopedWeightOpCache::default();
    let mut profile_expansion_variants = FxHashMap::<Box<[(u32, Weight)]>, usize>::default();
    let mut profile_expansion_variant_excess = 0usize;
    let mut profile_max_expansion_variants = 0usize;
    let mut expansion_cache = FxHashMap::<Box<[(u32, Weight)]>, Arc<[(i32, u32, Weight)]>>::default();
    let mut profile_expansion_cache_hits = 0usize;
    let mut profile_expansion_cache_misses = 0usize;

    while let Some((from_state, subset_entries)) = worklist.pop_front() {
        processed_states += 1;
        if profile && processed_states % 25_000 == 0 {
            eprintln!(
                "[glrmask/profile][determinize_progress] processed_states={} discovered_states={} queued_states={} transitions={} subset_entries={} multiway_union_entries={} multiway_union_hits={} multiway_union_misses={} topology_hashes={} topology_variant_excess={} max_topology_variants={} expansion_keys={} expansion_variant_excess={} max_expansion_variants={} expansion_cache_entries={} expansion_cache_hits={} expansion_cache_misses={} elapsed_ms={:.3}",
                processed_states,
                subset_map.len(),
                worklist.len(),
                dwa.num_transitions(),
                profile_subset_entries,
                scoped_multiway_union_cache.entries.len(),
                scoped_multiway_union_cache.hits,
                scoped_multiway_union_cache.misses,
                profile_topology_variants.len(),
                profile_topology_variant_excess,
                profile_max_topology_variants,
                profile_expansion_variants.len(),
                profile_expansion_variant_excess,
                profile_max_expansion_variants,
                expansion_cache.len(),
                profile_expansion_cache_hits,
                profile_expansion_cache_misses,
                determinize_started_at.unwrap().elapsed().as_secs_f64() * 1000.0,
            );
        }
        let expansion_key = labeled_transition_domains.as_ref().map(|domains| {
            transition_expansion_key(
                subset_entries.as_ref(),
                domains,
                &mut profile_expansion_weight_cache,
            )
        });
        if profile {
            profile_subset_entries += subset_entries.len();
            profile_max_subset_entries = profile_max_subset_entries.max(subset_entries.len());
            let bucket = match subset_entries.len() {
                0 => 0,
                1 => 1,
                2 => 2,
                3..=4 => 3,
                5..=8 => 4,
                9..=16 => 5,
                17..=32 => 6,
                _ => 7,
            };
            profile_subset_len_buckets[bucket] += 1;
            profile_full_path_weights += subset_entries
                .iter()
                .filter(|(_, weight)| weight.is_full())
                .count();

            let mut topology_hasher = FxHasher::default();
            for (state_id, _) in subset_entries.iter() {
                state_id.hash(&mut topology_hasher);
            }
            let topology_hash = topology_hasher.finish();
            let variants = profile_topology_variants.entry(topology_hash).or_default();
            if *variants > 0 {
                profile_topology_variant_excess += 1;
            }
            *variants += 1;
            profile_max_topology_variants = profile_max_topology_variants.max(*variants);

            if profile_expansion_keys && let Some(expansion_key) = expansion_key.as_ref() {
                let variants = profile_expansion_variants
                    .entry(expansion_key.clone())
                    .or_default();
                if *variants > 0 {
                    profile_expansion_variant_excess += 1;
                }
                *variants += 1;
                profile_max_expansion_variants =
                    profile_max_expansion_variants.max(*variants);
            }
        }

        if cache_expansions {
            let expansion_key = expansion_key
                .as_ref()
                .expect("expansion caching requires labeled transition domains");
            if let Some(cached_transitions) = expansion_cache.get(expansion_key.as_ref()) {
                profile_expansion_cache_hits += 1;
                for (label, to_state, edge_weight) in cached_transitions.iter() {
                    dwa.add_transition(from_state, *label, *to_state, edge_weight.clone());
                }
                continue;
            }
            profile_expansion_cache_misses += 1;
        }
        // Final weight computation is deferred to after the main loop
        // and parallelized across all states.
        let expand_started_at = profile.then(Instant::now);

        if let Some(weight_grouped_transitions) = &weight_grouped_transitions {
            if direct_single_target_enabled && subset_entries.len() == 1 {
                let (nwa_state_id, path_weight) = &subset_entries[0];
                for group in &weight_grouped_transitions[*nwa_state_id as usize] {
                    if profile {
                        profile_weight_group_visits += 1;
                        profile_raw_transition_visits += group.edges.len();
                    }
                    let next_weight = scoped_determinize_weight_cache
                        .intersection(path_weight, &group.weight);
                    if next_weight.is_empty() {
                        continue;
                    }

                    let mut emitted_direct_singleton = false;
                    for edge in &group.edges {
                        if edge.direct_singleton {
                            emitted_direct_singleton = true;
                            if profile {
                                profile_labels += 1;
                                profile_target_contributions += 1;
                                profile_single_contribution_labels += 1;
                                profile_single_contribution_no_epsilon_labels += 1;
                                profile_direct_single_target_labels += 1;
                                profile_direct_singleton_fast_path_labels += 1;
                            }
                            let subset_lookup_started_at = profile.then(Instant::now);
                            let (to_state, cache_hit) = intern_determinized_singleton(
                                edge.dst,
                                &next_weight,
                                &mut singleton_subsets,
                                &mut subset_map,
                                &mut worklist,
                                &mut dwa,
                            );
                            if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                                profile_subset_lookup_ms +=
                                    subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
                            }
                            if profile {
                                if cache_hit {
                                    profile_direct_singleton_cache_hits += 1;
                                } else {
                                    profile_direct_singleton_cache_misses += 1;
                                }
                            }
                            dwa.add_transition(from_state, edge.label, to_state, next_weight.clone());
                        } else {
                            raw_targets
                                .entry(edge.label)
                                .or_default()
                                .push((edge.dst, next_weight.clone()));
                        }
                    }
                    if emitted_direct_singleton && profile {
                        profile_direct_singleton_fast_path_groups += 1;
                    }
                }
            } else {
                for (nwa_state_id, path_weight) in subset_entries.iter() {
                    for group in &weight_grouped_transitions[*nwa_state_id as usize] {
                        if profile {
                            profile_weight_group_visits += 1;
                            profile_raw_transition_visits += group.edges.len();
                        }
                        let next_weight = scoped_determinize_weight_cache
                            .intersection(path_weight, &group.weight);
                        if next_weight.is_empty() {
                            continue;
                        }
                        for edge in &group.edges {
                            raw_targets
                                .entry(edge.label)
                                .or_default()
                                .push((edge.dst, next_weight.clone()));
                        }
                    }
                }
            }
        } else {
            for (nwa_state_id, path_weight) in subset_entries.iter() {
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
                    let (to_state, cache_hit) = intern_determinized_singleton(
                        dst,
                        &edge_weight,
                        &mut singleton_subsets,
                        &mut subset_map,
                        &mut worklist,
                        &mut dwa,
                    );
                    if profile {
                        if cache_hit {
                            profile_direct_singleton_cache_hits += 1;
                        } else {
                            profile_direct_singleton_cache_misses += 1;
                        }
                    }
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
                let combine_started_at = profile.then(Instant::now);
                let mut sorted_targets = target_contributions;
                sorted_targets.sort_unstable_by_key(|(dst, _)| *dst);
                let mut next_key: Vec<(u32, Weight)> =
                    Vec::with_capacity(sorted_targets.len());
                let mut group_start = 0usize;
                while group_start < sorted_targets.len() {
                    let dst = sorted_targets[group_start].0;
                    let mut group_end = group_start + 1;
                    while group_end < sorted_targets.len()
                        && sorted_targets[group_end].0 == dst
                    {
                        group_end += 1;
                    }
                    let weight = if group_end == group_start + 1 {
                        sorted_targets[group_start].1.clone()
                    } else {
                        scoped_multiway_union_cache.union_all(
                            sorted_targets[group_start..group_end]
                                .iter()
                                .map(|(_, weight)| weight),
                        )
                    };
                    next_key.push((dst, weight));
                    group_start = group_end;
                }
                if let Some(combine_started_at) = combine_started_at {
                    profile_combine_ms += combine_started_at.elapsed().as_secs_f64() * 1000.0;
                }

                let edge_union_started_at = profile.then(Instant::now);
                let edge_weight = scoped_multiway_union_cache
                    .union_all(next_key.iter().map(|(_, weight)| weight));
                if let Some(edge_union_started_at) = edge_union_started_at {
                    profile_edge_union_ms +=
                        edge_union_started_at.elapsed().as_secs_f64() * 1000.0;
                }
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
                            let replaced = duplicates.insert(dst, weights);
                            debug_assert!(replaced.is_none());
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

        if cache_expansions {
            let expansion_key = expansion_key
                .expect("expansion caching requires labeled transition domains");
            let transitions: Arc<[(i32, u32, Weight)]> = dwa.states()[from_state as usize]
                .transitions
                .iter()
                .map(|(&label, (to_state, edge_weight))| {
                    (label, *to_state, edge_weight.clone())
                })
                .collect::<Vec<_>>()
                .into();
            let replaced = expansion_cache.insert(expansion_key, transitions);
            debug_assert!(replaced.is_none());
        }
    }

    if profile {
        eprintln!(
            "[glrmask/profile][determinize_scoped_weight_cache] intersections={} multiway_union_entries={} multiway_union_hits={} multiway_union_misses={}",
            scoped_determinize_weight_cache.intersection_entry_count(),
            scoped_multiway_union_cache.entries.len(),
            scoped_multiway_union_cache.hits,
            scoped_multiway_union_cache.misses,
        );
        eprintln!(
            "[glrmask/profile][determinize_expansion_cache] enabled={} entries={} hits={} misses={}",
            cache_expansions,
            expansion_cache.len(),
            profile_expansion_cache_hits,
            profile_expansion_cache_misses,
        );
        eprintln!(
            "[glrmask/profile][determinize_subset_shapes] len_0={} len_1={} len_2={} len_3_4={} len_5_8={} len_9_16={} len_17_32={} len_gt_32={} full_path_weights={} topology_hashes={} topology_variant_excess={} max_topology_variants={} expansion_keys={} expansion_variant_excess={} max_expansion_variants={}",
            profile_subset_len_buckets[0],
            profile_subset_len_buckets[1],
            profile_subset_len_buckets[2],
            profile_subset_len_buckets[3],
            profile_subset_len_buckets[4],
            profile_subset_len_buckets[5],
            profile_subset_len_buckets[6],
            profile_subset_len_buckets[7],
            profile_full_path_weights,
            profile_topology_variants.len(),
            profile_topology_variant_excess,
            profile_max_topology_variants,
            profile_expansion_variants.len(),
            profile_expansion_variant_excess,
            profile_max_expansion_variants,
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
            for (state_id, _) in entries.iter() {
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
            for (state_id, path_weight) in entries.iter() {
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
            "[glrmask/profile][determinize] nwa_states={} dwa_states={} subset_map_entries={} max_weight_dim={} subset_entries={} max_subset_entries={} raw_transition_visits={} weight_grouped_transitions={} weight_group_build_ms={:.3} weight_group_visits={} labels={} target_contributions={} single_contribution_labels={} single_contribution_no_epsilon_labels={} direct_single_target_labels={} direct_singleton_cache_hits={} direct_singleton_cache_misses={} direct_singleton_fast_path_groups={} direct_singleton_fast_path_labels={} multi_contribution_single_target_no_epsilon_labels={} multi_contribution_single_target_no_epsilon_contributions={} direct_multi_target_labels={} multi_contribution_all_no_epsilon_labels={} multi_contribution_all_no_epsilon_contributions={} expand_ms={:.3} combine_ms={:.3} edge_union_ms={:.3} closure_ms={:.3} normalize_ms={:.3} canonicalize_ms={:.3} subset_lookup_ms={:.3} final_weights_ms={:.3} final_subsets={} final_subset_entries={} final_entries={} final_nonempty_contributions={} final_max_entries={} final_intersection_ms={:.3} final_union_ms={:.3}",
            nwa.states().len(),
            dwa.states().len(),
            subset_map.len(),
            max_weight_dim,
            profile_subset_entries,
            profile_max_subset_entries,
            profile_raw_transition_visits,
            group_transition_weights,
            weight_group_build_ms,
            profile_weight_group_visits,
            profile_labels,
            profile_target_contributions,
            profile_single_contribution_labels,
            profile_single_contribution_no_epsilon_labels,
            profile_direct_single_target_labels,
            profile_direct_singleton_cache_hits,
            profile_direct_singleton_cache_misses,
            profile_direct_singleton_fast_path_groups,
            profile_direct_singleton_fast_path_labels,
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
    use crate::automata::weighted_u32::equivalence::find_difference;
    use range_set_blaze::RangeSetBlaze;

    fn tokens(values: impl IntoIterator<Item = u32>) -> Weight {
        Weight::from_per_tsid_token_sets(std::iter::once((
            0,
            RangeSetBlaze::from_iter(values.into_iter().map(|value| value..=value)),
        )))
    }

    fn two_tsid_tokens(left: impl IntoIterator<Item = u32>, right: impl IntoIterator<Item = u32>) -> Weight {
        Weight::from_per_tsid_token_sets([
            (
                0,
                RangeSetBlaze::from_iter(left.into_iter().map(|value| value..=value)),
            ),
            (
                1,
                RangeSetBlaze::from_iter(right.into_iter().map(|value| value..=value)),
            ),
        ])
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
        let grouped = determinize_impl_with_options(&nwa, true, true, false).unwrap();
        assert_eq!(find_difference(&fast, &generic).unwrap(), None);
        assert_eq!(find_difference(&grouped, &generic).unwrap(), None);
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
    fn direct_multi_target_multiway_union_matches_generic_determinization() {
        let mut nwa = NWA::new(2, 32);
        let start = nwa.add_state();
        let left = nwa.add_state();
        let right = nwa.add_state();
        nwa.set_start_states(vec![start]);

        for offset in 0..12u32 {
            nwa.add_transition(
                start,
                7,
                left,
                two_tsid_tokens([offset, offset + 12], [offset + 1, offset + 13]),
            );
        }
        for offset in 0..9u32 {
            nwa.add_transition(
                start,
                7,
                right,
                two_tsid_tokens([offset + 2], [offset + 16, offset + 20]),
            );
        }
        nwa.set_final_weight(left, Weight::all());
        nwa.set_final_weight(right, Weight::all());

        let fast = determinize_impl_with_options(&nwa, true, true, false).unwrap();
        let generic = determinize_impl_with_options(&nwa, false, false, false).unwrap();
        assert_eq!(find_difference(&fast, &generic).unwrap(), None);
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
            let grouped = determinize_impl_with_options(&nwa, true, true, false).unwrap();
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
    fn future_live_push_preserves_determinized_language_across_acyclic_cases() {
        for case in 0u32..64 {
            let mut nwa = NWA::new(1, 11);
            let states: Vec<u32> = (0..7).map(|_| nwa.add_state()).collect();
            nwa.set_start_states(vec![states[0]]);

            for from in 0..6usize {
                for to in (from + 1)..7usize {
                    if (case + (from * 13 + to * 17) as u32) % 4 == 0 {
                        continue;
                    }
                    let label = ((case + (from * 5 + to * 3) as u32) % 5) as i32;
                    let first = (case + (from * 7 + to * 11) as u32) % 10;
                    let second = (first + 1 + case % 4) % 11;
                    nwa.add_transition(states[from], label, states[to], tokens([first, second]));
                    if (case + from as u32 + to as u32) % 6 == 0 {
                        let extra = (second + 3) % 12;
                        nwa.add_transition(states[from], label, states[to], tokens([extra]));
                    }
                    if to > from + 1 && (case + from as u32 * 3 + to as u32) % 9 == 0 {
                        nwa.add_epsilon(states[from], states[to], tokens([first]));
                    }
                }
            }

            for state in 0..7usize {
                if (case + state as u32) % 3 == 0 {
                    let first = (case + state as u32 * 2) % 11;
                    nwa.set_final_weight(states[state], tokens([first, (first + 2) % 12]));
                }
            }

            let baseline = determinize_impl_with_options(&nwa, true, true, false).unwrap();
            let mut pushed = nwa.clone();
            assert!(push_nwa_weights_to_future_live(&mut pushed).is_some());
            let optimized = determinize_impl_with_options(&pushed, true, true, false).unwrap();
            assert_eq!(
                find_difference(&baseline, &optimized).unwrap(),
                None,
                "case {case}",
            );
        }
    }

    #[test]
    fn future_live_push_removes_dead_token_domains_from_incoming_edges() {
        let mut nwa = NWA::new(1, 3);
        let start = nwa.add_state();
        let left = nwa.add_state();
        let right = nwa.add_state();
        let accept = nwa.add_state();
        nwa.set_start_states(vec![start]);
        nwa.add_transition(start, 1, left, tokens([0, 1]));
        nwa.add_transition(start, 1, right, tokens([0, 1]));
        nwa.add_transition(left, 2, accept, tokens([0]));
        nwa.add_transition(right, 2, accept, tokens([1]));
        nwa.set_final_weight(accept, Weight::all());

        let baseline = determinize(&nwa).unwrap();
        assert_eq!(push_nwa_weights_to_future_live(&mut nwa), Some(true));
        let optimized = determinize(&nwa).unwrap();
        assert_eq!(find_difference(&baseline, &optimized).unwrap(), None);

        let branches = &nwa.states()[start as usize].transitions[&1];
        assert_eq!(branches[0].1, tokens([0]));
        assert_eq!(branches[1].1, tokens([1]));
    }

    #[test]
    fn expansion_cache_preserves_determinized_language_across_acyclic_cases() {
        for case in 0u32..96 {
            let mut nwa = NWA::new(2, 15);
            let states: Vec<u32> = (0..8).map(|_| nwa.add_state()).collect();
            nwa.set_start_states(vec![states[0]]);

            for from in 0..7usize {
                for to in (from + 1)..8usize {
                    if (case + (from * 11 + to * 19) as u32) % 5 == 0 {
                        continue;
                    }
                    let label = ((case + (from * 7 + to * 3) as u32) % 6) as i32;
                    let left_a = (case + (from * 5 + to * 13) as u32) % 14;
                    let left_b = (left_a + 1 + case % 5) % 16;
                    let right_a = (case * 3 + (from * 17 + to * 7) as u32) % 15;
                    let right_b = (right_a + 2 + case % 4) % 16;
                    nwa.add_transition(
                        states[from],
                        label,
                        states[to],
                        two_tsid_tokens([left_a, left_b], [right_a, right_b]),
                    );
                    if (case + from as u32 * 2 + to as u32) % 7 == 0 {
                        nwa.add_transition(
                            states[from],
                            label,
                            states[to],
                            two_tsid_tokens(
                                [(left_b + 3) % 16],
                                [(right_b + 5) % 16],
                            ),
                        );
                    }
                    if to > from + 1 && (case + from as u32 + to as u32 * 2) % 11 == 0 {
                        nwa.add_epsilon(
                            states[from],
                            states[to],
                            two_tsid_tokens([left_a], [right_a]),
                        );
                    }
                }
            }

            for state in 0..8usize {
                if (case + state as u32) % 3 != 1 {
                    let left = (case + state as u32 * 3) % 16;
                    let right = (case * 2 + state as u32 * 5) % 16;
                    nwa.set_final_weight(
                        states[state],
                        two_tsid_tokens([left], [right]),
                    );
                }
            }

            let baseline =
                determinize_impl_with_options_and_cache(&nwa, true, true, false, false).unwrap();
            let cached =
                determinize_impl_with_options_and_cache(&nwa, true, true, false, true).unwrap();
            assert_eq!(
                find_difference(&baseline, &cached).unwrap(),
                None,
                "case {case}",
            );
        }
    }

    #[test]
    fn expansion_cache_reuses_final_only_weight_variants_exactly() {
        let mut nwa = NWA::new(1, 1);
        let start = nwa.add_state();
        let continuation = nwa.add_state();
        let final_only = nwa.add_state();
        let accept = nwa.add_state();
        nwa.set_start_states(vec![start]);

        nwa.add_transition(start, 1, continuation, Weight::all());
        nwa.add_transition(start, 1, final_only, tokens([0]));
        nwa.add_transition(start, 2, continuation, Weight::all());
        nwa.add_transition(start, 2, final_only, tokens([1]));
        nwa.add_transition(continuation, 3, accept, Weight::all());
        nwa.set_final_weight(final_only, Weight::all());
        nwa.set_final_weight(accept, Weight::all());

        let domains = nwa_labeled_transition_domains(&nwa);
        let mut cache = ScopedWeightOpCache::default();
        let left_key = transition_expansion_key(
            &[(continuation, Weight::all()), (final_only, tokens([0]))],
            &domains,
            &mut cache,
        );
        let right_key = transition_expansion_key(
            &[(continuation, Weight::all()), (final_only, tokens([1]))],
            &domains,
            &mut cache,
        );
        assert_eq!(left_key, right_key);

        let baseline =
            determinize_impl_with_options_and_cache(&nwa, true, true, false, false).unwrap();
        let cached =
            determinize_impl_with_options_and_cache(&nwa, true, true, false, true).unwrap();
        assert_eq!(find_difference(&baseline, &cached).unwrap(), None);
        assert_eq!(cached.eval_word(&[1]), tokens([0]));
        assert_eq!(cached.eval_word(&[2]), tokens([1]));
        assert_eq!(cached.eval_word(&[1, 3]), Weight::all());
        assert_eq!(cached.eval_word(&[2, 3]), Weight::all());
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
        let grouped = determinize_impl_with_options(&nwa, true, true, false).unwrap();
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

        let grouped = determinize_impl_with_options(&nwa, true, true, false).unwrap();
        let generic = determinize_impl_with_options(&nwa, false, false, false).unwrap();
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
