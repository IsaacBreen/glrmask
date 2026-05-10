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

const EXACT_LAYOUT_MAX_GROUPS: usize = 20;
const GLOBALLY_EXACT_COMPONENT_MAX_GROUPS_DEFAULT: usize = EXACT_LAYOUT_MAX_GROUPS;

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

fn legacy_exact_adjacency_proxy_enabled() -> bool {
    // Historical name retained for compatibility.  This is *not* globally
    // exact compaction; it only solves the old adjacency-proxy layout when the
    // number of groups is tiny enough for Held-Karp DP.
    env_flag("GLRMASK_EXACT_COMPACTION")
}

fn globally_exact_compaction_enabled() -> bool {
    env_flag("GLRMASK_GLOBALLY_EXACT_COMPACTION")
}

fn globally_exact_component_max_groups() -> usize {
    static MAX_GROUPS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAX_GROUPS.get_or_init(|| {
        std::env::var("GLRMASK_GLOBALLY_EXACT_MAX_COMPONENT_GROUPS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(GLOBALLY_EXACT_COMPONENT_MAX_GROUPS_DEFAULT)
    })
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
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
    if globally_exact_compaction_enabled() {
        return build_globally_exact_dimension_compaction(unique_weights, num_tsids, num_tokens);
    }

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

fn build_globally_exact_dimension_compaction(
    unique_weights: &[Weight],
    num_tsids: usize,
    num_tokens: usize,
) -> DimensionCompaction {
    let original_weight_refs = weight_refs(unique_weights);

    let (token_merge_perm, merged_num_tokens) =
        build_exact_token_merge_permutation(&original_weight_refs, num_tokens);
    let token_perm = order_token_groups_globally_exact(
        unique_weights,
        token_merge_perm,
        merged_num_tokens,
    );

    let token_compacted_weights = apply_permutations_to_weight_set(
        unique_weights,
        &identity_perm(num_tsids),
        &token_perm,
    );
    let token_compacted_refs = weight_refs(&token_compacted_weights);
    let (tsid_merge_perm, merged_num_tsids) =
        build_exact_tsid_merge_permutation(&token_compacted_refs, num_tsids);
    let tsid_perm = order_tsid_groups_globally_exact(
        &token_compacted_weights,
        tsid_merge_perm,
        merged_num_tsids,
        merged_num_tokens,
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

fn order_token_groups_globally_exact(
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

    let pair_weights = build_token_cooccurrence_pair_weights(&token_sets, num_groups);
    let layout = exact_layout_from_pair_weights_or_panic(&pair_weights, num_groups, "token");
    compose_group_layout(initial_perm, &layout)
}

fn order_tsid_groups_globally_exact(
    token_compacted_weights: &[Weight],
    initial_perm: Vec<u32>,
    num_groups: usize,
    num_tokens: usize,
) -> Vec<u32> {
    if num_groups < 2 {
        return initial_perm;
    }

    // Rebuild through the TSID quotient before measuring the TSID objective.
    // This is important for exactness: two previously-distinct weights may
    // become the same interned weight after semantic TSID merging, and the
    // objective counts that final interned representative only once.
    let quotient_weights = apply_permutations_to_weight_set(
        token_compacted_weights,
        &initial_perm,
        &identity_perm(num_tokens),
    );
    if quotient_weights.is_empty() {
        return initial_perm;
    }

    let pair_weights = build_tsid_equal_value_pair_weights(&quotient_weights, num_groups);
    let layout = exact_layout_from_pair_weights_or_panic(&pair_weights, num_groups, "TSID");
    compose_group_layout(initial_perm, &layout)
}

fn build_token_cooccurrence_pair_weights(
    token_sets: &[RangeSetBlaze<u32>],
    num_groups: usize,
) -> Vec<usize> {
    let mut pair_weights = vec![0usize; num_groups * num_groups];
    for token_set in token_sets {
        let mut members = rangeset_members_below(token_set, num_groups);
        members.sort_unstable();
        members.dedup();
        add_unit_clique_pair_weights(&mut pair_weights, num_groups, &members);
    }
    pair_weights
}

fn build_tsid_equal_value_pair_weights(weights: &[Weight], num_groups: usize) -> Vec<usize> {
    let mut pair_weights = vec![0usize; num_groups * num_groups];
    if num_groups == 0 {
        return pair_weights;
    }

    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }

        let mut by_token_set = HashMap::<Vec<(u32, u32)>, Vec<usize>>::new();
        for (tsid_range, token_set) in weight.0.range_values() {
            let members = by_token_set.entry(rangeset_key(token_set)).or_default();
            let start = (*tsid_range.start() as usize).min(num_groups);
            let end = (*tsid_range.end() as usize).min(num_groups.saturating_sub(1));
            if start <= end {
                members.extend(start..=end);
            }
        }

        for members in by_token_set.values_mut() {
            members.sort_unstable();
            members.dedup();
            add_unit_clique_pair_weights(&mut pair_weights, num_groups, members);
        }
    }

    pair_weights
}

fn rangeset_members_below(set: &RangeSetBlaze<u32>, upper_exclusive: usize) -> Vec<usize> {
    let mut members = Vec::new();
    if upper_exclusive == 0 {
        return members;
    }
    for range in set.ranges() {
        let start = *range.start() as usize;
        let end = (*range.end() as usize).min(upper_exclusive.saturating_sub(1));
        if start <= end {
            members.extend(start..=end);
        }
    }
    members
}

fn add_unit_clique_pair_weights(
    pair_weights: &mut [usize],
    num_groups: usize,
    members: &[usize],
) {
    for left_idx in 0..members.len() {
        let left = members[left_idx];
        if left >= num_groups {
            continue;
        }
        for &right in &members[left_idx + 1..] {
            if right >= num_groups || right == left {
                continue;
            }
            pair_weights[left * num_groups + right] += 1;
            pair_weights[right * num_groups + left] += 1;
        }
    }
}

fn exact_layout_from_pair_weights_or_panic(
    pair_weights: &[usize],
    num_groups: usize,
    dimension_name: &str,
) -> Vec<usize> {
    debug_assert_eq!(pair_weights.len(), num_groups * num_groups);
    if num_groups < 2 {
        return (0..num_groups).collect();
    }

    let components = positive_pair_weight_components(pair_weights, num_groups);
    let max_component_groups = globally_exact_component_max_groups();
    let mut layout = Vec::with_capacity(num_groups);

    for component in components {
        if component.len() <= 1 {
            layout.extend(component);
            continue;
        }

        let local_weights = project_pair_weights(pair_weights, num_groups, &component);
        let local_layout = if component.len() <= max_component_groups {
            exact_max_adjacency_layout(&local_weights, component.len())
        } else {
            eprintln!(
                "[glrmask/profile][globally_exact_compaction] dimension={dimension_name} component_groups={} dp_limit={} using=branch_and_bound_exact",
                component.len(),
                max_component_groups,
            );
            exact_max_adjacency_layout_branch_and_bound(&local_weights, component.len())
        };
        layout.extend(local_layout.into_iter().map(|local| component[local]));
    }

    layout
}

fn positive_pair_weight_components(pair_weights: &[usize], num_groups: usize) -> Vec<Vec<usize>> {
    let mut visited = vec![false; num_groups];
    let mut components = Vec::new();

    for start in 0..num_groups {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![start];
        let mut component = Vec::new();

        while let Some(node) = stack.pop() {
            component.push(node);
            for next in 0..num_groups {
                if !visited[next] && pair_weights[node * num_groups + next] > 0 {
                    visited[next] = true;
                    stack.push(next);
                }
            }
        }

        component.sort_unstable();
        components.push(component);
    }

    components.sort_by(|left, right| {
        left.first()
            .copied()
            .unwrap_or(usize::MAX)
            .cmp(&right.first().copied().unwrap_or(usize::MAX))
            .then(left.len().cmp(&right.len()))
    });
    components
}

fn project_pair_weights(
    pair_weights: &[usize],
    num_groups: usize,
    component: &[usize],
) -> Vec<usize> {
    let mut projected = vec![0usize; component.len() * component.len()];
    for (local_left, &global_left) in component.iter().enumerate() {
        for (local_right, &global_right) in component.iter().enumerate() {
            projected[local_left * component.len() + local_right] =
                pair_weights[global_left * num_groups + global_right];
        }
    }
    projected
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

    if legacy_exact_adjacency_proxy_enabled() && num_groups <= EXACT_LAYOUT_MAX_GROUPS {
        let layout = exact_max_adjacency_layout(&adjacency, num_groups);
        return compose_group_layout(initial_perm, &layout);
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

    let mut adjacency = vec![0usize; num_groups * num_groups];
    let mut group_profiles = vec![Vec::<(u32, Vec<(u32, u32)>)>::new(); num_groups];
    for (weight_idx, weight) in weights.iter().enumerate() {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        for (tsid_range, token_set) in weight.0.range_values() {
            let token_key = rangeset_key(token_set);
            for tsid in *tsid_range.start()..=*tsid_range.end() {
                let Some(&group) = initial_perm.get(tsid as usize) else {
                    continue;
                };
                group_profiles[group as usize].push((weight_idx as u32, token_key.clone()));
            }
        }
    }
    for profile in &mut group_profiles {
        profile.sort_unstable();
        profile.dedup();
    }

    for left in 0..num_groups {
        for right in (left + 1)..num_groups {
            let bonus = shared_tsid_profile_entry_count(&group_profiles[left], &group_profiles[right]);
            adjacency[left * num_groups + right] = bonus;
            adjacency[right * num_groups + left] = bonus;
        }
    }

    if legacy_exact_adjacency_proxy_enabled() && num_groups <= EXACT_LAYOUT_MAX_GROUPS {
        let layout = exact_max_adjacency_layout(&adjacency, num_groups);
        return compose_group_layout(initial_perm, &layout);
    }

    let mut layout: Vec<usize> = (0..num_groups).collect();
    layout.sort_by(|&left, &right| {
        group_profiles[left]
            .cmp(&group_profiles[right])
            .then(left.cmp(&right))
    });

    compose_group_layout(initial_perm, &layout)
}

fn shared_tsid_profile_entry_count(
    left: &[(u32, Vec<(u32, u32)>)],
    right: &[(u32, Vec<(u32, u32)>)],
) -> usize {
    let mut left_idx = 0usize;
    let mut right_idx = 0usize;
    let mut shared = 0usize;
    while left_idx < left.len() && right_idx < right.len() {
        match left[left_idx].cmp(&right[right_idx]) {
            std::cmp::Ordering::Less => left_idx += 1,
            std::cmp::Ordering::Greater => right_idx += 1,
            std::cmp::Ordering::Equal => {
                shared += 1;
                left_idx += 1;
                right_idx += 1;
            }
        }
    }
    shared
}

fn exact_max_adjacency_layout(adjacency: &[usize], num_groups: usize) -> Vec<usize> {
    debug_assert!(num_groups <= EXACT_LAYOUT_MAX_GROUPS);
    if num_groups < 2 {
        return (0..num_groups).collect();
    }

    let states = 1usize << num_groups;
    let mut best = vec![0usize; states * num_groups];
    let mut reachable = vec![false; states * num_groups];
    let mut parent = vec![usize::MAX; states * num_groups];

    for group in 0..num_groups {
        reachable[(1usize << group) * num_groups + group] = true;
    }

    for mask in 1usize..states {
        for last in 0..num_groups {
            let state_idx = mask * num_groups + last;
            if !reachable[state_idx] {
                continue;
            }
            let current = best[state_idx];
            for next in 0..num_groups {
                let bit = 1usize << next;
                if mask & bit != 0 {
                    continue;
                }
                let next_mask = mask | bit;
                let next_score = current + adjacency[last * num_groups + next];
                let next_idx = next_mask * num_groups + next;
                if !reachable[next_idx]
                    || next_score > best[next_idx]
                    || (next_score == best[next_idx] && last < parent[next_idx])
                {
                    reachable[next_idx] = true;
                    best[next_idx] = next_score;
                    parent[next_idx] = last;
                }
            }
        }
    }

    let full_mask = states - 1;
    let mut last = (0..num_groups)
        .max_by_key(|&group| (best[full_mask * num_groups + group], usize::MAX - group))
        .unwrap();
    let mut mask = full_mask;
    let mut reversed = Vec::with_capacity(num_groups);

    loop {
        reversed.push(last);
        let prev = parent[mask * num_groups + last];
        if prev == usize::MAX {
            break;
        }
        mask &= !(1usize << last);
        last = prev;
    }

    reversed.reverse();
    reversed
}

fn exact_max_adjacency_layout_branch_and_bound(
    adjacency: &[usize],
    num_groups: usize,
) -> Vec<usize> {
    debug_assert_eq!(adjacency.len(), num_groups * num_groups);
    if num_groups < 2 {
        return (0..num_groups).collect();
    }

    let mut weighted_degree = vec![0usize; num_groups];
    for left in 0..num_groups {
        for right in 0..num_groups {
            let weight = adjacency[left * num_groups + right];
            weighted_degree[left] += weight;
        }
    }

    let mut best_layout = greedy_adjacency_layout(adjacency, &weighted_degree, num_groups);
    improve_layout_2opt(adjacency, &mut best_layout, num_groups);
    improve_layout_reinsert(adjacency, &mut best_layout, num_groups);
    let mut best_score = path_score(adjacency, &best_layout, num_groups);
    let initial_upper_bound =
        remaining_path_score_upper_bound(adjacency, num_groups, None, &vec![false; num_groups]);
    eprintln!(
        "[glrmask/profile][globally_exact_compaction_bnb] groups={num_groups} incumbent_score={best_score} initial_upper_bound={initial_upper_bound}"
    );
    if best_score == initial_upper_bound {
        eprintln!(
            "[glrmask/profile][globally_exact_compaction_bnb] groups={num_groups} proven=upper_bound_tight"
        );
        return best_layout;
    }

    let mut used = vec![false; num_groups];
    let mut path = Vec::with_capacity(num_groups);
    let mut starts: Vec<usize> = (0..num_groups).collect();
    starts.sort_by_key(|&group| (usize::MAX - weighted_degree[group], group));

    for start in starts {
        used[start] = true;
        path.push(start);
        exact_layout_branch_and_bound_dfs(
            adjacency,
            num_groups,
            &weighted_degree,
            &mut used,
            &mut path,
            0,
            &mut best_score,
            &mut best_layout,
        );
        path.pop();
        used[start] = false;
    }

    best_layout
}

fn exact_layout_branch_and_bound_dfs(
    adjacency: &[usize],
    num_groups: usize,
    weighted_degree: &[usize],
    used: &mut [bool],
    path: &mut Vec<usize>,
    score: usize,
    best_score: &mut usize,
    best_layout: &mut Vec<usize>,
) {
    let remaining = num_groups - path.len();
    if remaining == 0 {
        if score > *best_score || (score == *best_score && path.as_slice() < best_layout.as_slice()) {
            *best_score = score;
            best_layout.clear();
            best_layout.extend_from_slice(path);
        }
        return;
    }

    let optimistic = score.saturating_add(remaining_path_score_upper_bound(
        adjacency,
        num_groups,
        path.last().copied(),
        used,
    ));
    if optimistic < *best_score {
        return;
    }

    let last = *path.last().unwrap();
    let mut candidates: Vec<usize> = (0..num_groups).filter(|&group| !used[group]).collect();
    candidates.sort_by_key(|&candidate| {
        (
            usize::MAX - adjacency[last * num_groups + candidate],
            usize::MAX - weighted_degree[candidate],
            candidate,
        )
    });

    for next in candidates {
        used[next] = true;
        path.push(next);
        exact_layout_branch_and_bound_dfs(
            adjacency,
            num_groups,
            weighted_degree,
            used,
            path,
            score + adjacency[last * num_groups + next],
            best_score,
            best_layout,
        );
        path.pop();
        used[next] = false;
    }
}

fn remaining_path_score_upper_bound(
    adjacency: &[usize],
    num_groups: usize,
    last: Option<usize>,
    used: &[bool],
) -> usize {
    let unused_count = used.iter().filter(|&&is_used| !is_used).count();
    if unused_count == 0 {
        return 0;
    }

    let mut degree_capacity_sum = 0usize;
    let mut endpoint_loss_candidates = Vec::with_capacity(unused_count);

    if let Some(last) = last {
        let best_from_last = (0..num_groups)
            .filter(|&candidate| !used[candidate])
            .map(|candidate| adjacency[last * num_groups + candidate])
            .max()
            .unwrap_or(0);
        degree_capacity_sum += best_from_last;
    }

    for vertex in 0..num_groups {
        if used[vertex] {
            continue;
        }

        let mut best = 0usize;
        let mut second = 0usize;
        for other in 0..num_groups {
            if other == vertex {
                continue;
            }
            if used[other] && Some(other) != last {
                continue;
            }
            let weight = adjacency[vertex * num_groups + other];
            if weight >= best {
                second = best;
                best = weight;
            } else if weight > second {
                second = weight;
            }
        }
        degree_capacity_sum += best + second;
        endpoint_loss_candidates.push(second);
    }

    endpoint_loss_candidates.sort_unstable();
    let endpoint_losses_needed = if last.is_some() { 1 } else { 2 };
    let endpoint_loss: usize = endpoint_loss_candidates
        .iter()
        .take(endpoint_losses_needed.min(endpoint_loss_candidates.len()))
        .sum();

    degree_capacity_sum.saturating_sub(endpoint_loss) / 2
}

fn greedy_adjacency_layout(
    adjacency: &[usize],
    weighted_degree: &[usize],
    num_groups: usize,
) -> Vec<usize> {
    let mut best_layout = Vec::new();
    let mut best_score = 0usize;

    for start in 0..num_groups {
        let mut used = vec![false; num_groups];
        let mut layout = Vec::with_capacity(num_groups);
        used[start] = true;
        layout.push(start);

        while layout.len() < num_groups {
            let last = *layout.last().unwrap();
            let next = (0..num_groups)
                .filter(|&group| !used[group])
                .max_by_key(|&group| {
                    (
                        adjacency[last * num_groups + group],
                        weighted_degree[group],
                        usize::MAX - group,
                    )
                })
                .unwrap();
            used[next] = true;
            layout.push(next);
        }

        let score = path_score(adjacency, &layout, num_groups);
        if best_layout.is_empty()
            || score > best_score
            || (score == best_score && layout.as_slice() < best_layout.as_slice())
        {
            best_score = score;
            best_layout = layout;
        }
    }

    best_layout
}

fn improve_layout_2opt(adjacency: &[usize], layout: &mut [usize], num_groups: usize) {
    if layout.len() < 4 {
        return;
    }

    loop {
        let mut improved = false;
        for left_edge in 0..layout.len() - 2 {
            let a = layout[left_edge];
            let b = layout[left_edge + 1];
            for right_edge in left_edge + 2..layout.len() - 1 {
                let c = layout[right_edge];
                let d = layout[right_edge + 1];
                let old = adjacency[a * num_groups + b] + adjacency[c * num_groups + d];
                let new = adjacency[a * num_groups + c] + adjacency[b * num_groups + d];
                if new > old {
                    layout[left_edge + 1..=right_edge].reverse();
                    improved = true;
                    break;
                }
            }
            if improved {
                break;
            }
        }
        if !improved {
            break;
        }
    }
}

fn improve_layout_reinsert(adjacency: &[usize], layout: &mut Vec<usize>, num_groups: usize) {
    if layout.len() < 3 {
        return;
    }

    loop {
        let mut best_gain = 0isize;
        let mut best_from = 0usize;
        let mut best_to = 0usize;

        for from in 0..layout.len() {
            let removed = layout[from];
            let remove_loss = incident_path_score_at(adjacency, layout, num_groups, from);
            let close_gain = if from > 0 && from + 1 < layout.len() {
                adjacency[layout[from - 1] * num_groups + layout[from + 1]]
            } else {
                0
            };
            let base_gain = close_gain as isize - remove_loss as isize;

            let reduced_len = layout.len() - 1;
            for to in 0..=reduced_len {
                let insert_gain = if to == 0 {
                    let right = reduced_layout_at(layout, from, 0);
                    adjacency[removed * num_groups + right]
                } else if to == reduced_len {
                    let left = reduced_layout_at(layout, from, reduced_len - 1);
                    adjacency[left * num_groups + removed]
                } else {
                    let left = reduced_layout_at(layout, from, to - 1);
                    let right = reduced_layout_at(layout, from, to);
                    adjacency[left * num_groups + removed]
                        + adjacency[removed * num_groups + right]
                        - adjacency[left * num_groups + right]
                };
                let gain = base_gain + insert_gain as isize;
                if gain > best_gain {
                    best_gain = gain;
                    best_from = from;
                    best_to = to;
                }
            }
        }

        if best_gain <= 0 {
            break;
        }

        let value = layout.remove(best_from);
        layout.insert(best_to, value);
    }
}

fn reduced_layout_at(layout: &[usize], removed: usize, index: usize) -> usize {
    if index < removed {
        layout[index]
    } else {
        layout[index + 1]
    }
}

fn incident_path_score_at(
    adjacency: &[usize],
    layout: &[usize],
    num_groups: usize,
    index: usize,
) -> usize {
    let mut score = 0usize;
    if index > 0 {
        score += adjacency[layout[index - 1] * num_groups + layout[index]];
    }
    if index + 1 < layout.len() {
        score += adjacency[layout[index] * num_groups + layout[index + 1]];
    }
    score
}

fn path_score(adjacency: &[usize], layout: &[usize], num_groups: usize) -> usize {
    layout
        .windows(2)
        .map(|pair| adjacency[pair[0] * num_groups + pair[1]])
        .sum()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn singleton_set(token: u32) -> RangeSetBlaze<u32> {
        RangeSetBlaze::from_iter(std::iter::once(token..=token))
    }

    fn set_from_bits(bits: usize, width: usize) -> RangeSetBlaze<u32> {
        RangeSetBlaze::from_iter(
            (0..width)
                .filter(move |bit| bits & (1usize << bit) != 0)
                .map(|bit| bit as u32..=bit as u32),
        )
    }

    fn all_permutations(n: usize) -> Vec<Vec<usize>> {
        fn rec(pos: usize, values: &mut [usize], out: &mut Vec<Vec<usize>>) {
            if pos == values.len() {
                out.push(values.to_vec());
                return;
            }
            for idx in pos..values.len() {
                values.swap(pos, idx);
                rec(pos + 1, values, out);
                values.swap(pos, idx);
            }
        }

        let mut values: Vec<usize> = (0..n).collect();
        let mut out = Vec::new();
        rec(0, &mut values, &mut out);
        out
    }

    fn path_score(pair_weights: &[usize], num_groups: usize, layout: &[usize]) -> usize {
        layout
            .windows(2)
            .map(|pair| pair_weights[pair[0] * num_groups + pair[1]])
            .sum()
    }

    fn brute_force_best_path_score(pair_weights: &[usize], num_groups: usize) -> usize {
        all_permutations(num_groups)
            .into_iter()
            .map(|layout| path_score(pair_weights, num_groups, &layout))
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn exact_token_layout_matches_bruteforce_for_all_four_element_set_families() {
        // Exhaustive over every family of subsets on a four-element token
        // universe.  The exact objective for a family of token sets is the
        // constant total cardinality minus the Hamiltonian path score generated
        // by pair co-occurrence weights, so matching the brute-force path score
        // verifies the token-layout optimizer for this entire tiny universe.
        let num_groups = 4;
        let nonempty_subsets: Vec<_> = (1usize..(1usize << num_groups))
            .map(|bits| set_from_bits(bits, num_groups))
            .collect();

        for family_bits in 0usize..(1usize << nonempty_subsets.len()) {
            let family: Vec<_> = nonempty_subsets
                .iter()
                .enumerate()
                .filter_map(|(idx, set)| {
                    (family_bits & (1usize << idx) != 0).then(|| set.clone())
                })
                .collect();
            let pair_weights = build_token_cooccurrence_pair_weights(&family, num_groups);
            let exact_layout = exact_layout_from_pair_weights_or_panic(
                &pair_weights,
                num_groups,
                "test-token",
            );
            let exact_score = path_score(&pair_weights, num_groups, &exact_layout);
            let brute_score = brute_force_best_path_score(&pair_weights, num_groups);
            assert_eq!(exact_score, brute_score, "family_bits={family_bits:#x}");
        }
    }

    #[test]
    fn exact_tsid_layout_matches_bruteforce_for_all_single_weight_four_tsid_labelings() {
        // Exhaustive over every single-weight map from four TSIDs to
        // {empty, token-set-A, token-set-B}.  This directly validates the
        // outer RangeMap objective transformation used by globally exact mode.
        let num_groups = 4;
        for mut code in 0usize..3usize.pow(num_groups as u32) {
            let mut entries = Vec::new();
            let mut labels = Vec::new();
            for tsid in 0..num_groups {
                let label = code % 3;
                code /= 3;
                labels.push(label);
                match label {
                    0 => {}
                    1 => entries.push((tsid as u32, singleton_set(11))),
                    2 => entries.push((tsid as u32, singleton_set(17))),
                    _ => unreachable!(),
                }
            }

            let weight = Weight::from_per_tsid_token_sets(entries);
            let pair_weights = build_tsid_equal_value_pair_weights(&[weight], num_groups);
            let exact_layout = exact_layout_from_pair_weights_or_panic(
                &pair_weights,
                num_groups,
                "test-tsid",
            );
            let exact_score = path_score(&pair_weights, num_groups, &exact_layout);
            let brute_score = brute_force_best_path_score(&pair_weights, num_groups);
            assert_eq!(exact_score, brute_score, "labels={labels:?}");
        }
    }

    #[test]
    fn exact_layout_decomposes_zero_weight_components_without_losing_score() {
        let num_groups = 6;
        let mut pair_weights = vec![0usize; num_groups * num_groups];
        // Component {0, 1, 2}
        pair_weights[0 * num_groups + 1] = 7;
        pair_weights[1 * num_groups + 0] = 7;
        pair_weights[1 * num_groups + 2] = 5;
        pair_weights[2 * num_groups + 1] = 5;
        // Component {3, 4}; node 5 is an isolated singleton component.
        pair_weights[3 * num_groups + 4] = 9;
        pair_weights[4 * num_groups + 3] = 9;

        let exact_layout = exact_layout_from_pair_weights_or_panic(
            &pair_weights,
            num_groups,
            "test-components",
        );
        let exact_score = path_score(&pair_weights, num_groups, &exact_layout);
        let brute_score = brute_force_best_path_score(&pair_weights, num_groups);
        assert_eq!(exact_score, brute_score);
    }
}
