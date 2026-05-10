//! Generic weight compaction for mapped artifacts.
//!
//! This pass has two separate jobs:
//! - merge only tokenizer-state/token IDs that are provably equivalent for the
//!   entire supplied weight collection;
//! - choose deterministic numeric orders for the merged classes that tend to
//!   reduce `RangeMapBlaze` / `RangeSetBlaze` fragmentation.
//!
//! Ordering is heuristic. Merging is not: every many-to-one mapping below is
//! derived from exact membership profiles, so the rewritten weights remain a
//! valid representation of the same relations under the updated `InternalIdMap`.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use rayon::prelude::*;
use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};

use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::weight::{Weight, finalize_weight_map, shared_rangeset};

#[derive(Clone, Debug)]
pub struct CompactReport {
    pub tsid_perm: Vec<u32>,
    pub token_perm: Vec<u32>,
    pub profile_stats: Option<CompactProfileStats>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InternedRangeCounts {
    pub tsid_ranges: usize,
    pub token_ranges: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompactProfileStats {
    pub tsids_before: usize,
    pub tsids_after: usize,
    pub tokens_before: usize,
    pub tokens_after: usize,
    pub weight_ranges_before: usize,
    pub weight_ranges_after: usize,
    pub token_ranges_before: usize,
    pub token_ranges_after: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct UniqueStorageCounts {
    weight_ranges: usize,
    token_ranges: usize,
}

struct DimensionCompaction {
    tsid_perm: Vec<u32>,
    ordered_num_tsids: usize,
    token_perm: Vec<u32>,
    ordered_num_tokens: usize,
}

pub(super) fn compact_weights_with_id_map(
    weights: &mut [&mut Weight],
    id_map: &mut InternalIdMap,
    collect_profile_stats: bool,
) -> CompactReport {
    let num_tsids = id_map.num_tsids() as usize;
    let num_tokens = id_map.num_internal_tokens() as usize;
    let storage_before = collect_profile_stats.then(|| {
        count_unique_storage_for_weight_refs(&weight_ref_slice(weights))
    });

    let unique_weights = collect_unique_weights_from_refs(weights);
    let compaction = build_dimension_compaction(&unique_weights, num_tsids, num_tokens);

    apply_permutations_to_weight_refs(
        weights,
        &unique_weights,
        &compaction.tsid_perm,
        &compaction.token_perm,
    );
    apply_perm_to_id_map(
        &mut id_map.tokenizer_states,
        &compaction.tsid_perm,
        compaction.ordered_num_tsids,
    );
    apply_perm_to_id_map(
        &mut id_map.vocab_tokens,
        &compaction.token_perm,
        compaction.ordered_num_tokens,
    );

    let profile_stats = storage_before.map(|storage_before| {
        let storage_after = count_unique_storage_for_weight_refs(&weight_ref_slice(weights));
        CompactProfileStats {
            tsids_before: num_tsids,
            tsids_after: compaction.ordered_num_tsids,
            tokens_before: num_tokens,
            tokens_after: compaction.ordered_num_tokens,
            weight_ranges_before: storage_before.weight_ranges,
            weight_ranges_after: storage_after.weight_ranges,
            token_ranges_before: storage_before.token_ranges,
            token_ranges_after: storage_after.token_ranges,
        }
    });

    CompactReport {
        tsid_perm: compaction.tsid_perm,
        token_perm: compaction.token_perm,
        profile_stats,
    }
}

pub(super) fn count_interned_ranges_for_weight_refs(weights: &[&Weight]) -> InternedRangeCounts {
    let counts = count_unique_storage_for_weight_refs(weights);
    InternedRangeCounts {
        tsid_ranges: counts.weight_ranges,
        token_ranges: counts.token_ranges,
    }
}

fn build_dimension_compaction(
    unique_weights: &[Weight],
    num_tsids: usize,
    num_tokens: usize,
) -> DimensionCompaction {
    let original_weight_refs = weight_refs(unique_weights);

    let (token_merge_perm, merged_num_tokens) =
        build_exact_token_merge_permutation(&original_weight_refs, num_tokens);
    let token_perm = order_token_groups(unique_weights, token_merge_perm, merged_num_tokens);

    let token_compacted_weights = apply_permutations_to_weight_set(
        unique_weights,
        &identity_perm(num_tsids),
        &token_perm,
    );
    let token_compacted_refs = weight_refs(&token_compacted_weights);
    let (tsid_merge_perm, merged_num_tsids) =
        build_exact_tsid_merge_permutation(&token_compacted_refs, num_tsids);
    let tsid_perm = order_tsid_groups(
        &token_compacted_refs,
        tsid_merge_perm,
        merged_num_tsids,
    );

    DimensionCompaction {
        tsid_perm,
        ordered_num_tsids: merged_num_tsids,
        token_perm,
        ordered_num_tokens: merged_num_tokens,
    }
}

fn build_exact_token_merge_permutation(weights: &[&Weight], num_tokens: usize) -> (Vec<u32>, usize) {
    if num_tokens == 0 {
        return (Vec::new(), 0);
    }

    let mut events: Vec<(u32, bool, u32)> = Vec::new();
    let mut context = 0u32;
    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        for (_tsid_range, token_set) in weight.0.range_values() {
            for token_range in token_set.ranges() {
                let lo = *token_range.start();
                if lo as usize >= num_tokens {
                    continue;
                }
                events.push((lo, true, context));
                let end = token_range.end().saturating_add(1).min(num_tokens as u32);
                if end < num_tokens as u32 {
                    events.push((end, false, context));
                }
            }
            context += 1;
        }
    }

    if events.is_empty() {
        return (vec![0; num_tokens], 1);
    }

    events.sort_unstable_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

    let mut perm = vec![0u32; num_tokens];
    let mut active = BTreeSet::<u32>::new();
    let mut profile_to_group = HashMap::<Vec<u32>, u32>::new();
    let mut current_group = group_for_profile(Vec::new(), &mut profile_to_group);
    let mut prev_pos = 0u32;
    let mut event_idx = 0usize;

    while event_idx < events.len() {
        let boundary = events[event_idx].0.min(num_tokens as u32);
        for token in prev_pos..boundary {
            perm[token as usize] = current_group;
        }
        prev_pos = boundary;

        while event_idx < events.len() && events[event_idx].0 == boundary {
            let (_, is_addition, context) = events[event_idx];
            if is_addition {
                active.insert(context);
            } else {
                active.remove(&context);
            }
            event_idx += 1;
        }

        let profile: Vec<u32> = active.iter().copied().collect();
        current_group = group_for_profile(profile, &mut profile_to_group);
    }

    for token in prev_pos..num_tokens as u32 {
        perm[token as usize] = current_group;
    }

    densify_used_group_ids(perm)
}

fn build_exact_tsid_merge_permutation(weights: &[&Weight], num_tsids: usize) -> (Vec<u32>, usize) {
    if num_tsids == 0 {
        return (Vec::new(), 0);
    }

    let mut profiles = vec![Vec::<u32>::new(); num_tsids];
    let mut context = 0u32;
    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        let mut contexts_by_token_set = HashMap::<Vec<(u32, u32)>, u32>::new();
        for (tsid_range, token_set) in weight.0.range_values() {
            let key = rangeset_key(token_set);
            let token_set_context = *contexts_by_token_set.entry(key).or_insert_with(|| {
                let current = context;
                context += 1;
                current
            });
            let start = *tsid_range.start();
            let end = (*tsid_range.end()).min(num_tsids.saturating_sub(1) as u32);
            for tsid in start..=end {
                profiles[tsid as usize].push(token_set_context);
            }
        }
    }

    build_profile_merge_permutation(&profiles)
}

fn order_token_groups(
    weights: &[Weight],
    initial_perm: Vec<u32>,
    num_groups: usize,
) -> Vec<u32> {
    if num_groups < 2 {
        return initial_perm;
    }

    let token_sets = collect_token_sets_after_permutation(weights, &initial_perm);
    if token_sets.is_empty() {
        return initial_perm;
    }

    let mut adjacency = vec![0usize; num_groups * num_groups];
    let mut first_seen = vec![usize::MAX; num_groups];
    let mut frequency = vec![0usize; num_groups];

    for (set_idx, token_set) in token_sets.iter().enumerate() {
        let groups: Vec<usize> = token_set
            .ranges()
            .flat_map(|range| *range.start()..=*range.end())
            .map(|group| group as usize)
            .filter(|&group| group < num_groups)
            .collect();
        for &group in &groups {
            first_seen[group] = first_seen[group].min(set_idx);
            frequency[group] += 1;
        }
        for pair in groups.windows(2) {
            let left = pair[0];
            let right = pair[1];
            adjacency[left * num_groups + right] += 1;
            adjacency[right * num_groups + left] += 1;
        }
    }

    let mut remaining: HashSet<usize> = (0..num_groups).collect();
    let mut layout = Vec::<usize>::with_capacity(num_groups);
    while !remaining.is_empty() {
        let next = if let Some(&last) = layout.last() {
            *remaining
                .iter()
                .max_by_key(|&&candidate| {
                    (
                        adjacency[last * num_groups + candidate],
                        frequency[candidate],
                        usize::MAX - first_seen[candidate],
                        usize::MAX - candidate,
                    )
                })
                .unwrap()
        } else {
            *remaining
                .iter()
                .max_by_key(|&&candidate| {
                    (
                        frequency[candidate],
                        usize::MAX - first_seen[candidate],
                        usize::MAX - candidate,
                    )
                })
                .unwrap()
        };
        remaining.remove(&next);
        layout.push(next);
    }

    compose_group_layout(initial_perm, &layout)
}

fn order_tsid_groups(
    weights: &[&Weight],
    initial_perm: Vec<u32>,
    num_groups: usize,
) -> Vec<u32> {
    if num_groups < 2 {
        return initial_perm;
    }

    let mut group_profiles = vec![Vec::<(u32, u32, u32)>::new(); num_groups];
    for (weight_idx, weight) in weights.iter().enumerate() {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        for (tsid_range, token_set) in weight.0.range_values() {
            let token_range_count = token_set.ranges().count() as u32;
            for tsid in *tsid_range.start()..=*tsid_range.end() {
                let Some(&group) = initial_perm.get(tsid as usize) else {
                    continue;
                };
                group_profiles[group as usize].push((
                    weight_idx as u32,
                    token_range_count,
                    token_set.iter().next().unwrap_or(u32::MAX),
                ));
            }
        }
    }
    for profile in &mut group_profiles {
        profile.sort_unstable();
        profile.dedup();
    }

    let mut layout: Vec<usize> = (0..num_groups).collect();
    layout.sort_by(|&left, &right| {
        group_profiles[left]
            .cmp(&group_profiles[right])
            .then(left.cmp(&right))
    });

    compose_group_layout(initial_perm, &layout)
}

fn compose_group_layout(initial_perm: Vec<u32>, layout: &[usize]) -> Vec<u32> {
    let mut group_to_position = vec![0u32; layout.len()];
    for (position, &group) in layout.iter().enumerate() {
        group_to_position[group] = position as u32;
    }
    initial_perm
        .into_iter()
        .map(|group| group_to_position[group as usize])
        .collect()
}

fn group_for_profile(profile: Vec<u32>, profile_to_group: &mut HashMap<Vec<u32>, u32>) -> u32 {
    let next_group = profile_to_group.len() as u32;
    *profile_to_group.entry(profile).or_insert(next_group)
}

fn build_profile_merge_permutation<P: Ord>(profiles: &[P]) -> (Vec<u32>, usize) {
    if profiles.is_empty() {
        return (Vec::new(), 0);
    }

    let mut indices: Vec<usize> = (0..profiles.len()).collect();
    indices.sort_by(|&left, &right| profiles[left].cmp(&profiles[right]));

    let mut perm = vec![0u32; profiles.len()];
    let mut group = 0u32;
    perm[indices[0]] = group;
    for pair in indices.windows(2) {
        if profiles[pair[0]] != profiles[pair[1]] {
            group += 1;
        }
        perm[pair[1]] = group;
    }

    (perm, group as usize + 1)
}

fn densify_used_group_ids(perm: Vec<u32>) -> (Vec<u32>, usize) {
    let mut remap = HashMap::<u32, u32>::new();
    let mut next_group = 0u32;
    let dense_perm = perm
        .into_iter()
        .map(|group| {
            *remap.entry(group).or_insert_with(|| {
                let dense = next_group;
                next_group += 1;
                dense
            })
        })
        .collect();

    (dense_perm, next_group as usize)
}

fn apply_permutations_to_weight_set(
    weights: &[Weight],
    tsid_perm: &[u32],
    token_perm: &[u32],
) -> Vec<Weight> {
    let mut cache = HashMap::new();
    dedup_weights_by_storage_ptr(
        weights
            .iter()
            .map(|weight| permute_weight_with_cache(weight, tsid_perm, token_perm, &mut cache))
            .collect(),
    )
}

fn apply_permutations_to_weight_refs(
    weights: &mut [&mut Weight],
    unique_weights: &[Weight],
    tsid_perm: &[u32],
    token_perm: &[u32],
) {
    let weight_entries: Vec<(usize, Weight)> = unique_weights
        .par_iter()
        .map(|weight| {
            let mut cache = HashMap::new();
            let new_weight = permute_weight_with_cache(weight, tsid_perm, token_perm, &mut cache);
            (Arc::as_ptr(&weight.0) as usize, new_weight)
        })
        .collect();
    let weight_map: HashMap<usize, Weight> = weight_entries.into_iter().collect();

    for weight in weights.iter_mut() {
        let ptr = Arc::as_ptr(&weight.0) as usize;
        if let Some(new_weight) = weight_map.get(&ptr) {
            **weight = new_weight.clone();
        }
    }
}

fn permute_weight_with_cache(
    weight: &Weight,
    tsid_perm: &[u32],
    token_perm: &[u32],
    permuted_token_cache: &mut HashMap<usize, RangeSetBlaze<u32>>,
) -> Weight {
    if weight.is_empty() {
        return Weight::empty();
    }
    if weight.is_full() {
        return Weight::all();
    }

    let new_tsid_count = tsid_perm.iter().copied().max().map_or(0, |max| max as usize + 1);
    let mut tokens_by_new_tsid = vec![None::<RangeSetBlaze<u32>>; new_tsid_count];
    let token_perm_is_identity = token_perm.iter().enumerate().all(|(idx, &value)| value == idx as u32);

    for (tsid_range, token_set) in weight.0.range_values() {
        let token_set_ptr = Arc::as_ptr(token_set) as usize;
        let mapped_tokens = permuted_token_cache
            .entry(token_set_ptr)
            .or_insert_with(|| {
                if token_perm_is_identity {
                    (**token_set).clone()
                } else {
                    permute_rangeset(token_set, token_perm)
                }
            })
            .clone();

        for tsid in *tsid_range.start()..=*tsid_range.end() {
            let Some(&new_tsid) = tsid_perm.get(tsid as usize) else {
                continue;
            };
            let slot = &mut tokens_by_new_tsid[new_tsid as usize];
            match slot {
                Some(existing) => *existing |= mapped_tokens.clone(),
                None => *slot = Some(mapped_tokens.clone()),
            }
        }
    }

    finalize_weight_map(build_weight_map_from_tsid_tokens(tokens_by_new_tsid))
}

fn build_weight_map_from_tsid_tokens(
    tokens_by_tsid: Vec<Option<RangeSetBlaze<u32>>>,
) -> RangeMapBlaze<u32, Arc<RangeSetBlaze<u32>>> {
    let mut map = RangeMapBlaze::new();
    let mut run: Option<(u32, u32, RangeSetBlaze<u32>)> = None;

    for (tsid, tokens) in tokens_by_tsid.into_iter().enumerate() {
        let Some(tokens) = tokens else {
            if let Some((start, end, run_tokens)) = run.take() {
                map.extend_simple(std::iter::once((start..=end, shared_rangeset(run_tokens))));
            }
            continue;
        };

        let tsid = tsid as u32;
        match run.as_mut() {
            Some((_start, end, run_tokens)) if *end + 1 == tsid && *run_tokens == tokens => {
                *end = tsid;
            }
            Some(_) => {
                let (start, end, run_tokens) = run.take().unwrap();
                map.extend_simple(std::iter::once((start..=end, shared_rangeset(run_tokens))));
                run = Some((tsid, tsid, tokens));
            }
            None => run = Some((tsid, tsid, tokens)),
        }
    }

    if let Some((start, end, run_tokens)) = run {
        map.extend_simple(std::iter::once((start..=end, shared_rangeset(run_tokens))));
    }

    map
}

fn permute_rangeset(set: &RangeSetBlaze<u32>, perm: &[u32]) -> RangeSetBlaze<u32> {
    let mut mapped: Vec<u32> = set
        .ranges()
        .flat_map(|range| *range.start()..=*range.end())
        .filter_map(|token| perm.get(token as usize).copied())
        .collect();
    mapped.sort_unstable();
    mapped.dedup();

    let mut ranges = Vec::new();
    let Some((&first, rest)) = mapped.split_first() else {
        return RangeSetBlaze::new();
    };
    let mut start = first;
    let mut end = first;
    for &token in rest {
        if token == end + 1 {
            end = token;
        } else {
            ranges.push(start..=end);
            start = token;
            end = token;
        }
    }
    ranges.push(start..=end);
    RangeSetBlaze::from_iter(ranges)
}

fn apply_perm_to_id_map(id_map: &mut ManyToOneIdMap, perm: &[u32], new_count: usize) {
    let old_internal_to_originals = std::mem::take(&mut id_map.internal_to_originals);
    let old_representatives = std::mem::take(&mut id_map.representative_original_ids);

    for internal in &mut id_map.original_to_internal {
        if *internal == u32::MAX {
            continue;
        }
        if let Some(&new_id) = perm.get(*internal as usize) {
            *internal = new_id;
        }
    }

    let mut new_internal_to_originals = vec![Vec::new(); new_count];
    let mut new_representatives = vec![u32::MAX; new_count];
    for (old_internal, originals) in old_internal_to_originals.into_iter().enumerate() {
        let Some(&new_internal) = perm.get(old_internal) else {
            continue;
        };
        let new_internal = new_internal as usize;
        if new_internal >= new_count {
            continue;
        }
        new_internal_to_originals[new_internal].extend(originals);
        if new_representatives[new_internal] == u32::MAX {
            new_representatives[new_internal] = old_representatives[old_internal];
        }
    }

    id_map.internal_to_originals = new_internal_to_originals;
    id_map.representative_original_ids = new_representatives;
}

fn collect_token_sets_after_permutation(
    weights: &[Weight],
    token_perm: &[u32],
) -> Vec<RangeSetBlaze<u32>> {
    let mut cache = HashMap::<usize, RangeSetBlaze<u32>>::new();
    let mut seen = HashSet::<Vec<(u32, u32)>>::new();
    let mut unique_sets = Vec::new();

    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        for (_tsid_range, token_set) in weight.0.range_values() {
            let ptr = Arc::as_ptr(token_set) as usize;
            let mapped = cache
                .entry(ptr)
                .or_insert_with(|| permute_rangeset(token_set, token_perm));
            if seen.insert(rangeset_key(mapped)) {
                unique_sets.push(mapped.clone());
            }
        }
    }

    unique_sets
}

fn collect_unique_weights_from_refs(weights: &[&mut Weight]) -> Vec<Weight> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for weight in weights {
        if seen.insert(Arc::as_ptr(&weight.0) as usize) {
            unique.push((**weight).clone());
        }
    }
    unique
}

fn dedup_weights_by_storage_ptr(weights: Vec<Weight>) -> Vec<Weight> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for weight in weights {
        if seen.insert(Arc::as_ptr(&weight.0) as usize) {
            unique.push(weight);
        }
    }
    unique
}

fn count_unique_storage_for_weight_refs(weights: &[&Weight]) -> UniqueStorageCounts {
    let mut seen_weights = HashSet::new();
    let mut seen_token_sets = HashSet::new();
    let mut storage = UniqueStorageCounts::default();

    for weight in weights {
        if seen_weights.insert(Arc::as_ptr(&weight.0) as usize) {
            storage.weight_ranges += weight.num_ranges();
        }
        for (_tsid_range, token_set) in weight.0.range_values() {
            if seen_token_sets.insert(Arc::as_ptr(token_set) as usize) {
                storage.token_ranges += token_set.ranges().count();
            }
        }
    }

    storage
}

fn weight_refs(weights: &[Weight]) -> Vec<&Weight> {
    weights.iter().collect()
}

fn weight_ref_slice<'a>(weights: &'a [&'a mut Weight]) -> Vec<&'a Weight> {
    weights.iter().map(|weight| &**weight).collect()
}

fn identity_perm(size: usize) -> Vec<u32> {
    (0..size as u32).collect()
}

fn rangeset_key(set: &RangeSetBlaze<u32>) -> Vec<(u32, u32)> {
    set.ranges()
        .map(|range| (*range.start(), *range.end()))
        .collect()
}
