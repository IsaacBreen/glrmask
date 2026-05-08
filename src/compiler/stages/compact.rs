//! DWA dimension compaction: merge equivalent IDs and reorder for adjacency.
//!
//! Merges tsid (outer) and token (inner) IDs that have identical weight
//! profiles, then reorders the remaining IDs so that similar elements are
//! placed adjacently. This reduces the number of ranges in the underlying
//! `RangeMapBlaze` / `RangeSetBlaze` structures.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rayon::prelude::*;
use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};

use crate::automata::weighted_u32::dwa::DWA;
use crate::compiler::constraint_possible_matches::RuntimePossibleMatchesByTerminal;
use crate::ds::weight::{Weight, finalize_weight_map, shared_rangeset};

use super::equiv_types::{InternalIdMap, MappedArtifact, ManyToOneIdMap};

// ── public entry point ──────────────────────────────────────────────────────

const TOKEN_ORDER_LOCAL_SEARCH_PASSES: usize = 3;
const TOKEN_ORDER_FINISH_ITERS: usize = 20000;
const TOKEN_ORDER_FINISH_SEED: u64 = 7;
const TOKEN_ORDER_FINISH_PATIENCE_MIN: usize = 256;
const TOKEN_ORDER_FINISH_PATIENCE_FACTOR: usize = 16;
const FAST_TOKEN_ORDER_MAX_GROUPS: usize = 64;

pub struct CompactReport {
    pub tsid_perm: Vec<u32>,
    pub token_perm: Vec<u32>,
    pub profile_stats: Option<CompactProfileStats>,
}

pub(crate) trait WeightRefs {
    fn weight_refs_mut(&mut self) -> Vec<&mut Weight>;
}

impl WeightRefs for DWA {
    fn weight_refs_mut(&mut self) -> Vec<&mut Weight> {
        let mut weights = Vec::new();
        for state in self.states_mut() {
            if let Some(final_weight) = state.final_weight.as_mut() {
                weights.push(final_weight);
            }
            for (_, weight) in state.transitions.values_mut() {
                weights.push(weight);
            }
        }
        weights
    }
}

impl WeightRefs for RuntimePossibleMatchesByTerminal {
    fn weight_refs_mut(&mut self) -> Vec<&mut Weight> {
        self.values_mut().collect()
    }
}

impl<A, B> WeightRefs for (A, B)
where
    A: WeightRefs,
    B: WeightRefs,
{
    fn weight_refs_mut(&mut self) -> Vec<&mut Weight> {
        let mut weights = self.0.weight_refs_mut();
        weights.extend(self.1.weight_refs_mut());
        weights
    }
}

impl<A, B, C> WeightRefs for (A, B, C)
where
    A: WeightRefs,
    B: WeightRefs,
    C: WeightRefs,
{
    fn weight_refs_mut(&mut self) -> Vec<&mut Weight> {
        let mut weights = self.0.weight_refs_mut();
        weights.extend(self.1.weight_refs_mut());
        weights.extend(self.2.weight_refs_mut());
        weights
    }
}

impl<T> WeightRefs for [T]
where
    T: WeightRefs,
{
    fn weight_refs_mut(&mut self) -> Vec<&mut Weight> {
        let mut weights = Vec::new();
        for item in self.iter_mut() {
            weights.extend(item.weight_refs_mut());
        }
        weights
    }
}

impl<T> WeightRefs for Vec<T>
where
    T: WeightRefs,
{
    fn weight_refs_mut(&mut self) -> Vec<&mut Weight> {
        self.as_mut_slice().weight_refs_mut()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InternedRangeCounts {
    pub tsid_ranges: usize,
    pub token_ranges: usize,
}

struct DimensionCompaction {
    tsid_perm: Vec<u32>,
    ordered_num_tsids: usize,
    token_perm: Vec<u32>,
    ordered_num_tokens: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct UniqueStorageCounts {
    weight_ranges: usize,
    token_ranges: usize,
}

impl UniqueStorageCounts {
    fn total_ranges(self) -> usize {
        self.weight_ranges + self.token_ranges
    }
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

impl CompactProfileStats {
}

struct TokenOrderScorer {
    num_groups: usize,
    total_token_memberships: usize,
    pair_weights: Vec<u32>,
}

impl TokenOrderScorer {
    fn new(merged_unique_token_sets: &[RangeSetBlaze<u32>], num_groups: usize) -> Self {
        let mut pair_weights = vec![0u32; num_groups * num_groups];
        let mut total_token_memberships = 0usize;

        for token_set in merged_unique_token_sets {
            let groups: Vec<usize> = token_set
                .ranges()
                .flat_map(|range| *range.start()..=*range.end())
                .map(|group| group as usize)
                .collect();

            total_token_memberships += groups.len();
            for left_idx in 0..groups.len() {
                let left = groups[left_idx];
                for &right in &groups[(left_idx + 1)..] {
                    pair_weights[left * num_groups + right] += 1;
                    pair_weights[right * num_groups + left] += 1;
                }
            }
        }

        Self {
            num_groups,
            total_token_memberships,
            pair_weights,
        }
    }

    fn score_layout(&self, layout: &[u32]) -> usize {
        debug_assert_eq!(layout.len(), self.num_groups);

        let adjacency_bonus: usize = layout
            .windows(2)
            .map(|edge| {
                self.pair_weights[edge[0] as usize * self.num_groups + edge[1] as usize] as usize
            })
            .sum();
        self.total_token_memberships.saturating_sub(adjacency_bonus)
    }
}

pub fn compact_dwa_dimensions_fast_with_stats(
    dwa: &mut DWA,
    id_map: &mut InternalIdMap,
) -> CompactReport {
    compact_dwa_dimensions_inner(dwa, id_map, true, true)
}

pub fn compact_weights_with_id_map(
    weights: &mut [&mut Weight],
    id_map: &mut InternalIdMap,
    collect_profile_stats: bool,
) -> CompactReport {
    let num_tsids = id_map.num_tsids();
    let num_tokens = id_map.num_internal_tokens();
    let storage_before = collect_profile_stats.then(|| count_unique_storage_for_weight_refs(&weight_ref_slice(weights)));

    let unique_weights = collect_unique_weights_from_refs(weights);
    let compaction = build_dimension_compaction(&unique_weights, num_tsids as usize, num_tokens, true);

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
            tsids_before: num_tsids as usize,
            tsids_after: compaction.ordered_num_tsids,
            tokens_before: num_tokens as usize,
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

pub fn count_interned_ranges_for_weights(weights: &[&Weight]) -> InternedRangeCounts {
    let counts = count_unique_storage_for_weight_refs(weights);
    InternedRangeCounts {
        tsid_ranges: counts.weight_ranges,
        token_ranges: counts.token_ranges,
    }
}

pub fn reconcile_weight_id_maps(
    left_weights: &mut [&mut Weight],
    left_id_map: &mut InternalIdMap,
    right_weights: &mut [&mut Weight],
    right_id_map: &mut InternalIdMap,
) {
    let common_id_map = build_common_internal_id_map(&[left_id_map, right_id_map]);

    let left_tsid_map = build_local_to_common_tsid_map(left_id_map, &common_id_map);
    let left_token_map = build_local_to_common_token_map(left_id_map, &common_id_map);
    let right_tsid_map = build_local_to_common_tsid_map(right_id_map, &common_id_map);
    let right_token_map = build_local_to_common_token_map(right_id_map, &common_id_map);

    remap_weights_with_maps(
        left_weights,
        &left_tsid_map,
        &left_token_map,
        common_id_map.num_tsids() as usize,
    );
    remap_weights_with_maps(
        right_weights,
        &right_tsid_map,
        &right_token_map,
        common_id_map.num_tsids() as usize,
    );

    *left_id_map = common_id_map.clone();
    *right_id_map = common_id_map;
}

pub fn reconcile_mapped_weight_artifacts<L, R>(
    left: &mut MappedArtifact<L>,
    right: &mut MappedArtifact<R>,
) -> InternalIdMap
where
    L: WeightRefs,
    R: WeightRefs,
{
    let (left_artifact, left_id_map) = left.parts_mut();
    let (right_artifact, right_id_map) = right.parts_mut();
    let mut left_weights = left_artifact.weight_refs_mut();
    let mut right_weights = right_artifact.weight_refs_mut();
    reconcile_weight_id_maps(
        &mut left_weights,
        left_id_map,
        &mut right_weights,
        right_id_map,
    );
    left.id_map().clone()
}

pub(crate) fn reconcile_mapped_pair<A, B>(
    a: MappedArtifact<A>,
    b: MappedArtifact<B>,
) -> MappedArtifact<(A, B)>
where
    A: WeightRefs,
    B: WeightRefs,
{
    let mut a = a;
    let mut b = b;
    let common_id_map = reconcile_mapped_weight_artifacts(&mut a, &mut b);
    let artifact_a = a.into_artifact();
    let artifact_b = b.into_artifact();
    MappedArtifact::new((artifact_a, artifact_b), common_id_map)
}

pub(crate) fn reconcile_mapped_vec<T>(inputs: Vec<MappedArtifact<T>>) -> MappedArtifact<Vec<T>>
where
    T: WeightRefs,
{
    assert!(!inputs.is_empty(), "reconcile_mapped_vec called with empty inputs");

    let mut iter = inputs.into_iter();
    let first = iter.next().unwrap();
    let (first_artifact, first_id_map) = first.into_parts();
    let mut acc = MappedArtifact::new(vec![first_artifact], first_id_map);

    for next in iter {
        let mut next = next;
        let common_id_map = reconcile_mapped_weight_artifacts(&mut acc, &mut next);
        let (artifacts, id_map) = acc.parts_mut();
        artifacts.push(next.into_artifact());
        *id_map = common_id_map;
    }

    acc
}

fn compact_dwa_dimensions_inner(
    dwa: &mut DWA,
    id_map: &mut InternalIdMap,
    collect_profile_stats: bool,
    skip_token_ordering: bool,
) -> CompactReport {
    let num_tsids = id_map.num_tsids();
    let num_tokens = id_map.num_internal_tokens();
    let storage_before = collect_profile_stats.then(|| count_unique_storage(dwa));

    let t0 = std::time::Instant::now();
    let unique_weights = collect_unique_weights(dwa);
    let t1 = std::time::Instant::now();
    let compaction = build_dimension_compaction(&unique_weights, num_tsids as usize, num_tokens, skip_token_ordering);
    let t2 = std::time::Instant::now();

    apply_permutations_to_dwa(
        dwa,
        &unique_weights,
        &compaction.tsid_perm,
        &compaction.token_perm,
    );
    let t3 = std::time::Instant::now();
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
        let storage_after = count_unique_storage(dwa);
        CompactProfileStats {
            tsids_before: num_tsids as usize,
            tsids_after: compaction.ordered_num_tsids,
            tokens_before: num_tokens as usize,
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

// ── helpers ─────────────────────────────────────────────────────────────────

fn build_dimension_compaction(
    unique_weights: &[Weight],
    num_tsids: usize,
    num_tokens: u32,
    skip_token_ordering: bool,
) -> DimensionCompaction {
    let ((tsid_perm, ordered_num_tsids), (token_perm, ordered_num_tokens)) = rayon::join(
        || {
            let original_weight_refs = weight_refs(unique_weights);
            let tsid_merge_profiles = build_tsid_context_profiles(&original_weight_refs, num_tsids);
            let (tsid_merge_perm, merged_num_tsids) =
                build_profile_merge_permutation(&tsid_merge_profiles);

            if skip_token_ordering {
                (tsid_merge_perm, merged_num_tsids)
            } else {
                let merged_tsid_weights = apply_permutations_to_weight_set(
                    unique_weights,
                    &tsid_merge_perm,
                    &identity_perm(num_tokens as usize),
                );

                let merged_tsid_refs = weight_refs(&merged_tsid_weights);
                let tsid_order_profiles =
                    build_tsid_context_profiles(&merged_tsid_refs, merged_num_tsids);
                let (tsid_order_perm, ordered_num_tsids) =
                    build_profile_merge_permutation(&tsid_order_profiles);

                (compose_perm(&tsid_merge_perm, &tsid_order_perm), ordered_num_tsids)
            }
        },
        || {
            let original_weight_refs = weight_refs(unique_weights);

            let (token_perm, ordered_num_tokens) = if skip_token_ordering {
                let result = build_token_merge_permutation_ranged(&original_weight_refs, num_tokens);
                let (token_perm, _, _) =
                    maybe_optimize_fast_token_group_order(
                        unique_weights,
                        num_tsids,
                        result.0,
                        result.1,
                        false,
                    );
                (token_perm, result.1)
            } else {
                let token_profiles = build_token_profiles(&original_weight_refs, num_tokens);
                let (token_group_perm, ordered_num_tokens) =
                    build_profile_merge_permutation(&token_profiles);
                let merged_token_sets =
                    collect_token_sets_after_permutation(unique_weights, &token_group_perm);
                let token_perm = optimize_token_group_order(
                    &merged_token_sets,
                    token_group_perm,
                    ordered_num_tokens,
                );
                (token_perm, ordered_num_tokens)
            };

            (token_perm, ordered_num_tokens)
        },
    );

    DimensionCompaction {
        tsid_perm,
        ordered_num_tsids,
        token_perm,
        ordered_num_tokens,
    }
}

fn weight_refs(weights: &[Weight]) -> Vec<&Weight> {
    weights.iter().collect()
}

fn weight_ref_slice<'a>(weights: &'a [&'a mut Weight]) -> Vec<&'a Weight> {
    weights.iter().map(|weight| &**weight).collect()
}

fn collect_unique_weights(dwa: &DWA) -> Vec<Weight> {
    let mut seen = std::collections::HashSet::new();
    let mut unique = Vec::new();
    for state in dwa.states() {
        for (_, (_, weight)) in &state.transitions {
            if seen.insert(Arc::as_ptr(&weight.0) as usize) {
                unique.push(weight.clone());
            }
        }
        if let Some(fw) = &state.final_weight {
            if seen.insert(Arc::as_ptr(&fw.0) as usize) {
                unique.push(fw.clone());
            }
        }
    }
    unique
}

fn dedup_weights_by_storage_ptr(weights: Vec<Weight>) -> Vec<Weight> {
    let mut seen = std::collections::HashSet::new();
    let mut unique = Vec::new();
    for weight in weights {
        if seen.insert(Arc::as_ptr(&weight.0) as usize) {
            unique.push(weight);
        }
    }
    unique
}

fn collect_unique_weights_from_refs(weights: &[&mut Weight]) -> Vec<Weight> {
    let mut seen = std::collections::HashSet::new();
    let mut unique = Vec::new();
    for weight in weights {
        if seen.insert(Arc::as_ptr(&weight.0) as usize) {
            unique.push((**weight).clone());
        }
    }
    unique
}

fn identity_perm(size: usize) -> Vec<u32> {
    (0..size as u32).collect()
}

fn compose_perm(left: &[u32], right: &[u32]) -> Vec<u32> {
    left.iter().map(|&mid| right[mid as usize]).collect()
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

fn rangeset_key(set: &RangeSetBlaze<u32>) -> Vec<(u32, u32)> {
    set.ranges()
        .map(|range| (*range.start(), *range.end()))
        .collect()
}

/// Treat repeated equal token sets within the same weight as the same context.
/// This lets identical TSIDs merge cleanly and also keeps semantically similar
/// TSIDs near each other when only their ordering changes.
fn build_tsid_context_profiles(weights: &[&Weight], num_tsids: usize) -> Vec<Vec<u32>> {
    let mut profiles = vec![Vec::new(); num_tsids];
    let mut ctx = 0u32;
    for w in weights {
        let mut contexts_by_token_set: HashMap<Vec<(u32, u32)>, u32> = HashMap::new();
        for (tsid_range, token_set) in w.0.range_values() {
            let key = rangeset_key(token_set);
            let token_set_ctx = *contexts_by_token_set.entry(key).or_insert_with(|| {
                let current = ctx;
                ctx += 1;
                current
            });
            for tsid in *tsid_range.start()..=*tsid_range.end() {
                if (tsid as usize) < num_tsids {
                    profiles[tsid as usize].push(token_set_ctx);
                }
            }
        }
    }
    profiles
}

/// For each token, list the (weight, entry) context indices where it appears.
fn build_token_profiles(weights: &[&Weight], num_tokens: u32) -> Vec<Vec<u32>> {
    let n = num_tokens as usize;
    let mut profiles = vec![Vec::new(); n];
    let mut ctx = 0u32;
    for w in weights {
        for (_tsid_range, token_set) in w.0.range_values() {
            for token_range in token_set.ranges() {
                for token in *token_range.start()..=*token_range.end() {
                    if (token as usize) < n {
                        profiles[token as usize].push(ctx);
                    }
                }
            }
            ctx += 1;
        }
    }
    profiles
}

/// Range-aware token merge: sweep boundary events to produce the merge
/// permutation directly, without expanding ranges to individual tokens.
fn build_token_merge_permutation_ranged(weights: &[&Weight], num_tokens: u32) -> (Vec<u32>, usize) {
    let n = num_tokens as usize;
    if n == 0 {
        return (vec![], 0);
    }

    // Collect boundary events: (position, is_addition, ctx)
    let mut events: Vec<(u32, bool, u32)> = Vec::new();
    let mut max_ctx = 0u32;
    let mut ctx = 0u32;
    for w in weights {
        for (_tsid_range, token_set) in w.0.range_values() {
            for token_range in token_set.ranges() {
                let lo = *token_range.start();
                let hi = *token_range.end();
                if (lo as usize) < n {
                    events.push((lo, true, ctx));
                    let end = (hi + 1).min(num_tokens);
                    if (end as usize) < n {
                        events.push((end, false, ctx));
                    }
                }
            }
            max_ctx = ctx;
            ctx += 1;
        }
    }

    // Sort: by position, removals before additions at the same position
    events.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    // Pre-compute random Zobrist keys for each context
    let num_ctx = (max_ctx + 1) as usize;
    let zobrist_keys: Vec<u128> = (0..num_ctx)
        .map(|i| {
            // Deterministic pseudo-random keys from context index
            let seed = i as u128;
            let h = seed.wrapping_mul(0x9E3779B97F4A7C15_u128.wrapping_shl(64) | 0xF39CC0605CEDC835)
                .wrapping_add(0x6A09E667F3BCC908_u128.wrapping_shl(64) | 0xBB67AE8584CAA73B);
            h ^ (h >> 61) ^ (h >> 37)
        })
        .collect();

    // Sweep to assign group IDs using Zobrist hash
    let mut perm = vec![0u32; n];
    let mut hash_to_group: HashMap<u128, u32> = HashMap::new();
    let mut num_groups = 0u32;

    // Assign group for the empty profile (hash=0)
    let mut current_hash: u128 = 0;
    let empty_group = {
        let g = num_groups;
        num_groups += 1;
        hash_to_group.insert(0u128, g);
        g
    };
    let mut current_group = empty_group;
    let mut prev_pos = 0u32;
    let mut event_idx = 0;

    // Collect unique boundary positions
    let mut boundaries: Vec<u32> = events.iter().map(|e| e.0).collect();
    boundaries.sort_unstable();
    boundaries.dedup();

    for &boundary in &boundaries {
        // Fill perm[prev_pos..boundary] with current_group
        for i in prev_pos..boundary.min(num_tokens) {
            perm[i as usize] = current_group;
        }
        prev_pos = boundary;

        // Process all events at this boundary, updating hash incrementally
        while event_idx < events.len() && events[event_idx].0 == boundary {
            let (_, _is_start, c) = events[event_idx];
            // XOR toggles: add and remove are the same operation
            current_hash ^= zobrist_keys[c as usize];
            event_idx += 1;
        }

        // Determine group for the new active set (O(1) lookup)
        current_group = *hash_to_group.entry(current_hash).or_insert_with(|| {
            let g = num_groups;
            num_groups += 1;
            g
        });
    }

    // Fill remaining tokens
    for i in prev_pos..num_tokens {
        perm[i as usize] = current_group;
    }

    densify_used_group_ids(perm)
}

fn densify_used_group_ids(perm: Vec<u32>) -> (Vec<u32>, usize) {
    if perm.is_empty() {
        return (perm, 0);
    }

    let mut remap = rustc_hash::FxHashMap::default();
    let mut next_dense = 0u32;
    let dense_perm = perm
        .into_iter()
        .map(|group| {
            *remap.entry(group).or_insert_with(|| {
                let dense = next_dense;
                next_dense += 1;
                dense
            })
        })
        .collect();

    (dense_perm, next_dense as usize)
}

/// Merge elements with identical profiles, then sort by profile.
/// Returns `(perm, new_count)` where `perm[old_id] = new_id` (many-to-one)
/// and `new_count` is the number of unique merged IDs.
fn build_profile_merge_permutation<P: Ord + std::hash::Hash + Eq>(
    profiles: &[P],
) -> (Vec<u32>, usize) {
    let n = profiles.len();
    if n == 0 {
        return (vec![], 0);
    }

    // Group old IDs by profile: profile → list of old IDs with that profile
    // Use index-based comparison to avoid requiring Clone
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| profiles[a].cmp(&profiles[b]));

    // Walk sorted indices, grouping consecutive identical profiles
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current_group = vec![sorted_indices[0]];
    for &idx in &sorted_indices[1..] {
        if profiles[idx] == profiles[current_group[0]] {
            current_group.push(idx);
        } else {
            groups.push(std::mem::take(&mut current_group));
            current_group.push(idx);
        }
    }
    groups.push(current_group);

    // Assign new IDs: one per group, in sorted order
    let new_count = groups.len();
    let mut perm = vec![0u32; n];
    for (new_id, group) in groups.iter().enumerate() {
        for &old_id in group {
            perm[old_id] = new_id as u32;
        }
    }

    (perm, new_count)
}

fn collect_token_sets_after_permutation(
    weights: &[Weight],
    token_perm: &[u32],
) -> Vec<RangeSetBlaze<u32>> {
    // Cache permuted token sets by Arc pointer — many weights share the same
    // interned token set, so we avoid redundant permute_rangeset calls.
    let mut cache: HashMap<usize, RangeSetBlaze<u32>> = HashMap::new();
    let mut seen = std::collections::HashSet::new();
    let mut unique_sets = Vec::new();
    for weight in weights {
        for (_, token_set) in weight.0.range_values() {
            let ptr = Arc::as_ptr(token_set) as usize;
            let merged = cache
                .entry(ptr)
                .or_insert_with(|| permute_rangeset(token_set, token_perm));
            let key = rangeset_key(merged);
            if seen.insert(key) {
                unique_sets.push(merged.clone());
            }
        }
    }
    unique_sets
}

fn layout_to_group_positions(layout: &[u32]) -> Vec<u32> {
    let mut group_positions = vec![0u32; layout.len()];
    for (position, &group) in layout.iter().enumerate() {
        group_positions[group as usize] = position as u32;
    }
    group_positions
}

fn improve_layout_with_adjacent_swaps(
    scorer: &TokenOrderScorer,
    layout: &mut Vec<u32>,
    current_score: &mut usize,
) {
    for _ in 0..TOKEN_ORDER_LOCAL_SEARCH_PASSES {
        let mut improved = false;
        for left_pos in 0..(layout.len() - 1) {
            let right_pos = left_pos + 1;

            layout.swap(left_pos, right_pos);
            let candidate_score = scorer.score_layout(layout);
            if candidate_score < *current_score {
                *current_score = candidate_score;
                improved = true;
            } else {
                layout.swap(left_pos, right_pos);
            }
        }
        if !improved {
            break;
        }
    }
}

fn finish_layout_with_seeded_search(
    scorer: &TokenOrderScorer,
    initial_layout: Vec<u32>,
) -> Vec<u32> {
    if TOKEN_ORDER_FINISH_ITERS == 0 || initial_layout.len() < 2 {
        return initial_layout;
    }

    let mut rng = StdRng::seed_from_u64(TOKEN_ORDER_FINISH_SEED);
    let mut current_layout = initial_layout.clone();
    let mut current_score = scorer.score_layout(&current_layout);
    let mut best_layout = current_layout.clone();
    let mut best_score = current_score;
    let patience = TOKEN_ORDER_FINISH_PATIENCE_MIN.max(
        initial_layout
            .len()
            .saturating_mul(TOKEN_ORDER_FINISH_PATIENCE_FACTOR),
    );
    let mut iters_since_best = 0usize;

    let mut temperature = 8.0f64;
    // Reusable buffer to record the move for undo.
    let mut undo_buf: Vec<u32> = Vec::new();
    for _ in 0..TOKEN_ORDER_FINISH_ITERS {
        // Apply move in-place and record undo info
        apply_random_layout_move_with_undo(&mut current_layout, &mut rng, &mut undo_buf);
        let candidate_score = scorer.score_layout(&current_layout);

        if candidate_score < best_score {
            best_score = candidate_score;
            best_layout.copy_from_slice(&current_layout);
            iters_since_best = 0;
        } else {
            iters_since_best += 1;
            if iters_since_best >= patience {
                break;
            }
        }

        let delta = candidate_score as i64 - current_score as i64;
        let accept = if delta <= 0 {
            true
        } else {
            let probability = (-(delta as f64) / temperature.max(0.1)).exp().clamp(0.0, 1.0);
            rng.gen_bool(probability)
        };

        if accept {
            current_score = candidate_score;
        } else {
            // Rejected: undo the move
            undo_layout_move(&mut current_layout, &undo_buf);
        }

        temperature *= 0.995;
    }

    best_layout
}

fn optimize_token_group_order(
    merged_unique_token_sets: &[RangeSetBlaze<u32>],
    initial_token_perm: Vec<u32>,
    new_num_tokens: usize,
) -> Vec<u32> {
    if new_num_tokens < 2 || merged_unique_token_sets.is_empty() {
        return initial_token_perm;
    }

    let scorer = TokenOrderScorer::new(merged_unique_token_sets, new_num_tokens);
    let mut layout: Vec<u32> = (0..new_num_tokens as u32).collect();
    let mut current_score = scorer.score_layout(&layout);

    improve_layout_with_adjacent_swaps(&scorer, &mut layout, &mut current_score);

    layout = finish_layout_with_seeded_search(&scorer, layout);
    let group_positions = layout_to_group_positions(&layout);

    initial_token_perm
        .into_iter()
        .map(|group| group_positions[group as usize])
        .collect()
}

fn optimize_token_group_order_local_only(
    merged_unique_token_sets: &[RangeSetBlaze<u32>],
    initial_token_perm: Vec<u32>,
    new_num_tokens: usize,
) -> Vec<u32> {
    if new_num_tokens < 2 || merged_unique_token_sets.is_empty() {
        return initial_token_perm;
    }

    let scorer = TokenOrderScorer::new(merged_unique_token_sets, new_num_tokens);
    let mut layout: Vec<u32> = (0..new_num_tokens as u32).collect();
    let mut current_score = scorer.score_layout(&layout);
    improve_layout_with_adjacent_swaps(&scorer, &mut layout, &mut current_score);

    let group_positions = layout_to_group_positions(&layout);
    initial_token_perm
        .into_iter()
        .map(|group| group_positions[group as usize])
        .collect()
}

fn maybe_optimize_fast_token_group_order(
    unique_weights: &[Weight],
    num_tsids: usize,
    initial_token_perm: Vec<u32>,
    new_num_tokens: usize,
    profile_compact: bool,
) -> (Vec<u32>, f64, f64) {
    if new_num_tokens < 2 || new_num_tokens > FAST_TOKEN_ORDER_MAX_GROUPS {
        return (initial_token_perm, 0.0, 0.0);
    }

    let collect_started_at = std::time::Instant::now();
    let merged_unique_token_sets =
        collect_token_sets_after_permutation(unique_weights, &initial_token_perm);
    let collect_ms = collect_started_at.elapsed().as_secs_f64() * 1000.0;
    if merged_unique_token_sets.is_empty() {
        return (initial_token_perm, collect_ms, 0.0);
    }

    let optimize_started_at = std::time::Instant::now();
    let optimized_token_perm = optimize_token_group_order_local_only(
        &merged_unique_token_sets,
        initial_token_perm.clone(),
        new_num_tokens,
    );
    let optimize_ms = optimize_started_at.elapsed().as_secs_f64() * 1000.0;

    if profile_compact {
        emit_fast_token_order_gap_profile(
            unique_weights,
            num_tsids,
            &initial_token_perm,
            &optimized_token_perm,
            &merged_unique_token_sets,
            new_num_tokens,
            optimize_ms,
        );
    }

    (optimized_token_perm, collect_ms, optimize_ms)
}

fn emit_fast_token_order_gap_profile(
    unique_weights: &[Weight],
    num_tsids: usize,
    initial_token_perm: &[u32],
    optimized_token_perm: &[u32],
    merged_unique_token_sets: &[RangeSetBlaze<u32>],
    new_num_tokens: usize,
    optimize_ms: f64,
) {
    let scorer = TokenOrderScorer::new(merged_unique_token_sets, new_num_tokens);
    let baseline_layout: Vec<u32> = (0..new_num_tokens as u32).collect();
    let baseline_proxy_ranges = scorer.score_layout(&baseline_layout);

    let mut optimized_group_positions = vec![u32::MAX; new_num_tokens];
    for (&group, &position) in initial_token_perm.iter().zip(optimized_token_perm.iter()) {
        let slot = &mut optimized_group_positions[group as usize];
        if *slot == u32::MAX {
            *slot = position;
        }
    }
    if optimized_group_positions.iter().any(|&position| position == u32::MAX) {
        return;
    }

    let mut optimized_layout = vec![0u32; new_num_tokens];
    for (group, &position) in optimized_group_positions.iter().enumerate() {
        optimized_layout[position as usize] = group as u32;
    }
    let optimized_proxy_ranges = scorer.score_layout(&optimized_layout);

    let identity_tsid_perm = identity_perm(num_tsids);
    let baseline_storage = count_unique_storage_for_weights(&apply_permutations_to_weight_set(
        unique_weights,
        &identity_tsid_perm,
        initial_token_perm,
    ));
    let optimized_storage = count_unique_storage_for_weights(&apply_permutations_to_weight_set(
        unique_weights,
        &identity_tsid_perm,
        &optimized_token_perm,
    ));

    eprintln!(
        "[glrmask/profile][compact_token_gap] merged_tokens={} optimize_ms={:.3} proxy_token_ranges={}=>{} actual_token_ranges={}=>{} actual_total_ranges={}=>{}",
        new_num_tokens,
        optimize_ms,
        baseline_proxy_ranges,
        optimized_proxy_ranges,
        baseline_storage.token_ranges,
        optimized_storage.token_ranges,
        baseline_storage.total_ranges(),
        optimized_storage.total_ranges(),
    );
}

/// Apply tsid and token permutations (possibly many-to-one) to every weight.
fn apply_permutations_to_dwa(
    dwa: &mut DWA,
    unique_weights: &[Weight],
    tsid_perm: &[u32],
    token_perm: &[u32],
) {
    use rayon::prelude::*;

    // Permute each unique weight in parallel. Each worker holds its own
    // per-weight token-set cache (shared across that weight's internal
    // range-values but not across weights). For large DWAs (~2k unique
    // weights) this is a significant win — each weight's permutation is
    // O(num_ranges * num_tokens) and the work splits cleanly.
    let weight_entries: Vec<(usize, Weight)> = unique_weights
        .par_iter()
        .map(|w| {
            let mut cache = HashMap::new();
            let new_w = permute_weight_with_cache(w, tsid_perm, token_perm, &mut cache);
            (Arc::as_ptr(&w.0) as usize, new_w)
        })
        .collect();
    let weight_map: HashMap<usize, Weight> = weight_entries.into_iter().collect();

    dwa.states_mut().par_iter_mut().for_each(|state| {
        for (_, (_, weight)) in state.transitions.iter_mut() {
            let ptr = Arc::as_ptr(&weight.0) as usize;
            if let Some(new_w) = weight_map.get(&ptr) {
                *weight = new_w.clone();
            }
        }
        if let Some(fw) = &mut state.final_weight {
            let ptr = Arc::as_ptr(&fw.0) as usize;
            if let Some(new_w) = weight_map.get(&ptr) {
                *fw = new_w.clone();
            }
        }
    });
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

fn apply_random_layout_move(layout: &mut [u32], rng: &mut StdRng) {
    if layout.len() < 2 {
        return;
    }

    match rng.gen_range(0..3) {
        0 => {
            let left = rng.gen_range(0..layout.len());
            let mut right = rng.gen_range(0..layout.len());
            if left == right {
                right = (right + 1) % layout.len();
            }
            layout.swap(left, right);
        }
        1 => {
            let left = rng.gen_range(0..layout.len() - 1);
            layout.swap(left, left + 1);
        }
        _ => {
            let from = rng.gen_range(0..layout.len());
            let to = rng.gen_range(0..layout.len());
            if from != to {
                let value = layout[from];
                if from < to {
                    layout.copy_within((from + 1)..=to, from);
                } else {
                    layout.copy_within(to..from, to + 1);
                }
                layout[to] = value;
            }
        }
    }
}

/// Apply a random move and record undo info.
/// undo_buf format: [move_type, args...] where move_type:
///   0 → swap(left, right)
///   1 → adjacent swap(left)
///   2 → block move(from, to)
fn apply_random_layout_move_with_undo(
    layout: &mut [u32],
    rng: &mut StdRng,
    undo_buf: &mut Vec<u32>,
) {
    undo_buf.clear();
    if layout.len() < 2 {
        return;
    }

    match rng.gen_range(0..3) {
        0 => {
            let left = rng.gen_range(0..layout.len());
            let mut right = rng.gen_range(0..layout.len());
            if left == right {
                right = (right + 1) % layout.len();
            }
            undo_buf.extend_from_slice(&[0, left as u32, right as u32]);
            layout.swap(left, right);
        }
        1 => {
            let left = rng.gen_range(0..layout.len() - 1);
            undo_buf.extend_from_slice(&[1, left as u32]);
            layout.swap(left, left + 1);
        }
        _ => {
            let from = rng.gen_range(0..layout.len());
            let to = rng.gen_range(0..layout.len());
            undo_buf.extend_from_slice(&[2, from as u32, to as u32]);
            if from != to {
                let value = layout[from];
                if from < to {
                    layout.copy_within((from + 1)..=to, from);
                } else {
                    layout.copy_within(to..from, to + 1);
                }
                layout[to] = value;
            }
        }
    }
}

/// Undo a move recorded by `apply_random_layout_move_with_undo`.
fn undo_layout_move(layout: &mut [u32], undo_buf: &[u32]) {
    if undo_buf.is_empty() {
        return;
    }
    match undo_buf[0] {
        0 => {
            // Swap: just swap back
            layout.swap(undo_buf[1] as usize, undo_buf[2] as usize);
        }
        1 => {
            // Adjacent swap: swap back
            let left = undo_buf[1] as usize;
            layout.swap(left, left + 1);
        }
        2 => {
            // Block move: reverse the block move
            let from = undo_buf[1] as usize;
            let to = undo_buf[2] as usize;
            if from != to {
                // Original moved layout[from] to layout[to] by shifting.
                // To undo: move layout[to] back to layout[from].
                let value = layout[to];
                if from < to {
                    // Original: copy_within((from+1)..=to, from), layout[to] = value
                    // Undo: copy_within(from..to, from+1), layout[from] = value
                    layout.copy_within(from..to, from + 1);
                } else {
                    // Original: copy_within(to..from, to+1), layout[to] = value
                    // Undo: copy_within((to+1)..=from, to), layout[from] = value
                    layout.copy_within((to + 1)..=from, to);
                }
                layout[from] = value;
            }
        }
        _ => {}
    }
}

/// Create a new Weight from `w` with permuted (possibly many-to-one) coords.
fn permute_weight(w: &Weight, tsid_perm: &[u32], token_perm: &[u32]) -> Weight {
    let mut cache = HashMap::new();
    permute_weight_with_cache(w, tsid_perm, token_perm, &mut cache)
}

/// Like `permute_weight` but reuses a token-set permutation cache across calls.
fn permute_weight_with_cache(
    w: &Weight,
    tsid_perm: &[u32],
    token_perm: &[u32],
    permuted_token_cache: &mut HashMap<usize, RangeSetBlaze<u32>>,
) -> Weight {
    if w.is_empty() {
        return Weight::empty();
    }
    if w.is_full() {
        return Weight::all();
    }

    // Use a flat Vec for O(1) access by new_tsid instead of BTreeMap O(log n).
    let new_tsid_cap = tsid_perm.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut tokens_by_new_tsid: Vec<Option<RangeSetBlaze<u32>>> = vec![None; new_tsid_cap];

    // Check if token_perm is identity (common in intermediate compact steps).
    let token_perm_is_identity = token_perm.iter().enumerate().all(|(i, &v)| v == i as u32);

    for (tsid_range, token_set) in w.0.range_values() {
        let ts_ptr = Arc::as_ptr(token_set) as usize;
        let new_ts = permuted_token_cache
            .entry(ts_ptr)
            .or_insert_with(|| {
                if token_perm_is_identity {
                    (**token_set).clone()
                } else {
                    permute_rangeset(token_set, token_perm)
                }
            })
            .clone();

        for tsid in *tsid_range.start()..=*tsid_range.end() {
            if (tsid as usize) < tsid_perm.len() {
                let new_tsid = tsid_perm[tsid as usize] as usize;
                match &mut tokens_by_new_tsid[new_tsid] {
                    Some(existing) => *existing |= new_ts.clone(),
                    slot @ None => *slot = Some(new_ts.clone()),
                }
            }
        }
    }

    // Build the output weight from the Vec, iterating in order.
    let mut map = RangeMapBlaze::new();
    let mut run: Option<(u32, u32, RangeSetBlaze<u32>)> = None; // (start, end, tokens)

    for (new_tsid, slot) in tokens_by_new_tsid.into_iter().enumerate() {
        if let Some(tokens) = slot {
            let new_tsid = new_tsid as u32;
            match run {
                Some((_start, end, ref prev_tokens)) if new_tsid == end + 1 && tokens == *prev_tokens => {
                    run.as_mut().unwrap().1 = new_tsid;
                }
                Some((start, end, prev_tokens)) => {
                    map.extend_simple(std::iter::once((
                        start..=end,
                        shared_rangeset(prev_tokens),
                    )));
                    run = Some((new_tsid, new_tsid, tokens));
                }
                None => {
                    run = Some((new_tsid, new_tsid, tokens));
                }
            }
        }
    }
    if let Some((start, end, tokens)) = run {
        map.extend_simple(std::iter::once((start..=end, shared_rangeset(tokens))));
    }

    finalize_weight_map(map)
}

/// Map each element in `set` through the permutation (may be many-to-one).
fn permute_rangeset(set: &RangeSetBlaze<u32>, perm: &[u32]) -> RangeSetBlaze<u32> {
    let mut mapped: Vec<u32> = set
        .ranges()
        .flat_map(|r| *r.start()..=*r.end())
        .filter_map(|v| perm.get(v as usize).copied())
        .collect();
    mapped.sort_unstable();
    mapped.dedup();

    if mapped.is_empty() {
        return RangeSetBlaze::new();
    }

    let mut ranges = Vec::new();
    let mut start = mapped[0];
    let mut end = mapped[0];
    for &v in &mapped[1..] {
        if v == end + 1 {
            end = v;
        } else {
            ranges.push(start..=end);
            start = v;
            end = v;
        }
    }
    ranges.push(start..=end);
    RangeSetBlaze::from_iter(ranges)
}

/// Update a `ManyToOneIdMap` after a (possibly many-to-one) permutation.
fn apply_perm_to_id_map(id_map: &mut ManyToOneIdMap, perm: &[u32], new_count: usize) {
    let old_internal_to_originals = std::mem::take(&mut id_map.internal_to_originals);
    let old_representatives = std::mem::take(&mut id_map.representative_original_ids);

    for internal in &mut id_map.original_to_internal {
        if *internal != u32::MAX {
            if let Some(&new_id) = perm.get(*internal as usize) {
                *internal = new_id;
            }
        }
    }

    let mut new_internal_to_originals = vec![Vec::new(); new_count];
    let mut new_representatives = vec![u32::MAX; new_count];
    for (old_internal, originals) in old_internal_to_originals.into_iter().enumerate() {
        let Some(&new_internal) = perm.get(old_internal) else {
            continue;
        };
        if (new_internal as usize) >= new_count {
            continue;
        }
        new_internal_to_originals[new_internal as usize].extend(originals);
        if new_representatives[new_internal as usize] == u32::MAX {
            new_representatives[new_internal as usize] = old_representatives[old_internal];
        }
    }
    id_map.internal_to_originals = new_internal_to_originals;
    id_map.representative_original_ids = new_representatives;
}

fn count_unique_storage(dwa: &DWA) -> UniqueStorageCounts {
    let unique_weights = collect_unique_weights(dwa);
    count_unique_storage_for_weights(&unique_weights)
}

fn count_unique_storage_for_weights(weights: &[Weight]) -> UniqueStorageCounts {
    let refs = weight_refs(weights);
    count_unique_storage_for_weight_refs(&refs)
}

fn count_unique_storage_for_weight_refs(weights: &[&Weight]) -> UniqueStorageCounts {
    let mut seen_token_sets = std::collections::HashSet::new();
    let mut storage = UniqueStorageCounts::default();
    for weight in weights {
        storage.weight_ranges += weight.num_ranges();
        for (_, token_set) in weight.0.range_values() {
            if seen_token_sets.insert(Arc::as_ptr(token_set) as usize) {
                storage.token_ranges += token_set.ranges().count();
            }
        }
    }
    storage
}

fn build_common_internal_id_map(inputs: &[&InternalIdMap]) -> InternalIdMap {
    let num_tokenizer_states = inputs
        .iter()
        .map(|input| input.tokenizer_states.original_to_internal.len())
        .max()
        .unwrap_or(0);
    let num_original_tokens = inputs
        .iter()
        .map(|input| input.vocab_tokens.original_to_internal.len())
        .max()
        .unwrap_or(0);

    let tokenizer_states = build_common_many_to_one_id_map(
        inputs,
        num_tokenizer_states,
        |input| &input.tokenizer_states,
        false,
    );
    let vocab_tokens = build_common_many_to_one_id_map(
        inputs,
        num_original_tokens,
        |input| &input.vocab_tokens,
        true,
    );

    InternalIdMap {
        tokenizer_states,
        vocab_tokens,
    }
}

fn build_common_many_to_one_id_map(
    inputs: &[&InternalIdMap],
    num_originals: usize,
    project: impl Fn(&InternalIdMap) -> &ManyToOneIdMap,
    allow_unmapped: bool,
) -> ManyToOneIdMap {
    let mut composite_to_class: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut original_to_internal = vec![u32::MAX; num_originals];
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut representatives: Vec<u32> = Vec::new();

    for original in 0..num_originals {
        let composite: Vec<u32> = inputs
            .iter()
            .map(|input| {
                project(input)
                    .original_to_internal
                    .get(original)
                    .copied()
                    .unwrap_or(u32::MAX)
            })
            .collect();
        if allow_unmapped && composite.iter().all(|&value| value == u32::MAX) {
            continue;
        }

        let next_id = internal_to_originals.len() as u32;
        let class_id = *composite_to_class.entry(composite).or_insert_with(|| {
            internal_to_originals.push(Vec::new());
            representatives.push(original as u32);
            next_id
        });
        original_to_internal[original] = class_id;
        internal_to_originals[class_id as usize].push(original as u32);
    }

    reorder_common_classes(
        composite_to_class,
        &mut original_to_internal,
        &mut internal_to_originals,
        &mut representatives,
        allow_unmapped,
    );

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids: representatives,
    }
}

fn reorder_common_classes(
    composite_to_class: HashMap<Vec<u32>, u32>,
    original_to_internal: &mut [u32],
    internal_to_originals: &mut Vec<Vec<u32>>,
    representatives: &mut Vec<u32>,
    allow_unmapped: bool,
) {
    let num_classes = internal_to_originals.len();
    if num_classes <= 1 {
        return;
    }

    let mut sorted: Vec<(Vec<u32>, u32)> = composite_to_class.into_iter().collect();
    sorted.sort_by(|left, right| left.0.cmp(&right.0));

    let mut old_to_new = vec![0u32; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        old_to_new[*old_id as usize] = new_id as u32;
    }

    for value in original_to_internal.iter_mut() {
        if *value == u32::MAX && allow_unmapped {
            continue;
        }
        *value = old_to_new[*value as usize];
    }

    let mut new_internal_to_originals = vec![Vec::new(); num_classes];
    let mut new_representatives = vec![u32::MAX; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        new_internal_to_originals[new_id] = std::mem::take(&mut internal_to_originals[*old_id as usize]);
        new_representatives[new_id] = representatives[*old_id as usize];
    }
    *internal_to_originals = new_internal_to_originals;
    *representatives = new_representatives;
}

fn build_local_to_common_tsid_map(
    local_id_map: &InternalIdMap,
    common_id_map: &InternalIdMap,
) -> Vec<Vec<u32>> {
    let num_local = local_id_map.num_tsids() as usize;
    let mut local_to_common = vec![BTreeSet::new(); num_local];

    for (state, &local_tsid) in local_id_map
        .tokenizer_states
        .original_to_internal
        .iter()
        .enumerate()
    {
        if local_tsid == u32::MAX {
            continue;
        }
        let common_tsid = common_id_map
            .tokenizer_states
            .original_to_internal
            .get(state)
            .copied()
            .unwrap_or(u32::MAX);
        if common_tsid == u32::MAX {
            continue;
        }
        local_to_common[local_tsid as usize].insert(common_tsid);
    }

    local_to_common
        .into_iter()
        .map(|ids| ids.into_iter().collect())
        .collect()
}

fn build_local_to_common_token_map(
    local_id_map: &InternalIdMap,
    common_id_map: &InternalIdMap,
) -> Vec<Vec<u32>> {
    let num_local = local_id_map.num_internal_tokens() as usize;
    let mut local_to_common = vec![BTreeSet::new(); num_local];

    for (original, &local_token) in local_id_map.vocab_tokens.original_to_internal.iter().enumerate() {
        if local_token == u32::MAX {
            continue;
        }
        let common_token = common_id_map
            .vocab_tokens
            .original_to_internal
            .get(original)
            .copied()
            .unwrap_or(u32::MAX);
        if common_token == u32::MAX {
            continue;
        }
        local_to_common[local_token as usize].insert(common_token);
    }

    local_to_common
        .into_iter()
        .map(|ids| ids.into_iter().collect())
        .collect()
}

fn remap_weights_with_maps(
    weights: &mut [&mut Weight],
    local_to_common_tsids: &[Vec<u32>],
    local_to_common_tokens: &[Vec<u32>],
    common_tsid_count: usize,
) {
    let mut cache = HashMap::<usize, Weight>::new();
    for weight in weights.iter_mut() {
        let remapped = remap_weight_cached_general(
            weight,
            local_to_common_tsids,
            local_to_common_tokens,
            common_tsid_count,
            &mut cache,
        );
        **weight = remapped;
    }
}

fn remap_weight_cached_general(
    weight: &Weight,
    local_to_common_tsids: &[Vec<u32>],
    local_to_common_tokens: &[Vec<u32>],
    common_tsid_count: usize,
    cache: &mut HashMap<usize, Weight>,
) -> Weight {
    let ptr = Arc::as_ptr(&weight.0) as usize;
    if let Some(cached) = cache.get(&ptr) {
        return cached.clone();
    }

    let remapped = remap_weight_general(
        weight,
        local_to_common_tsids,
        local_to_common_tokens,
        common_tsid_count,
    );
    cache.insert(ptr, remapped.clone());
    remapped
}

fn remap_weight_general(
    weight: &Weight,
    local_to_common_tsids: &[Vec<u32>],
    local_to_common_tokens: &[Vec<u32>],
    common_tsid_count: usize,
) -> Weight {
    if weight.is_empty() {
        return weight.clone();
    }

    if weight.is_full() {
        let mut all_common_tokens = RangeSetBlaze::new();
        for common_tokens in local_to_common_tokens {
            for &common_token in common_tokens {
                all_common_tokens.insert(common_token);
            }
        }
        if all_common_tokens.is_empty() {
            return Weight::empty();
        }

        let mut all_common_tsids = BTreeSet::new();
        for common_tsids in local_to_common_tsids {
            for &common_tsid in common_tsids {
                if (common_tsid as usize) < common_tsid_count {
                    all_common_tsids.insert(common_tsid);
                }
            }
        }
        if all_common_tsids.is_empty() {
            return Weight::empty();
        }

        return Weight::from_per_tsid_token_sets(
            all_common_tsids
                .into_iter()
                .map(|common_tsid| (common_tsid, all_common_tokens.clone())),
        );
    }

    let Some(entries) = weight.compact_entries() else {
        return weight.clone();
    };

    let mut token_cache = HashMap::<usize, Arc<RangeSetBlaze<u32>>>::new();
    let mut tokens_by_common_tsid: Vec<Option<Arc<RangeSetBlaze<u32>>>> = vec![None; common_tsid_count];
    let mut any_set = false;

    for (start, end, tokens) in entries {
        let token_key = Arc::as_ptr(&tokens) as usize;
        let mapped_tokens = token_cache
            .entry(token_key)
            .or_insert_with(|| {
                let mut result = RangeSetBlaze::new();
                for local_token in tokens.iter() {
                    if let Some(common_tokens) = local_to_common_tokens.get(local_token as usize) {
                        for &common_token in common_tokens {
                            result.insert(common_token);
                        }
                    }
                }
                Arc::new(result)
            })
            .clone();

        for local_tsid in start..=end {
            let Some(common_tsids) = local_to_common_tsids.get(local_tsid as usize) else {
                continue;
            };
            for &common_tsid in common_tsids {
                let index = common_tsid as usize;
                if index >= common_tsid_count {
                    continue;
                }
                match &mut tokens_by_common_tsid[index] {
                    Some(existing) => {
                        let merged = existing.as_ref() | mapped_tokens.as_ref();
                        *existing = shared_rangeset(merged);
                    }
                    slot @ None => {
                        *slot = Some(Arc::clone(&mapped_tokens));
                    }
                }
                any_set = true;
            }
        }
    }

    if !any_set {
        return Weight::empty();
    }

    let mut map = RangeMapBlaze::<u32, Arc<RangeSetBlaze<u32>>>::new();
    let mut run_start: Option<u32> = None;
    let mut run_end = 0u32;
    let mut run_tokens: Option<Arc<RangeSetBlaze<u32>>> = None;

    for (index, slot) in tokens_by_common_tsid.iter().enumerate() {
        let common_tsid = index as u32;
        if let Some(tokens) = slot {
            if let Some(ref current) = run_tokens {
                if Arc::ptr_eq(current, tokens) || current.as_ref() == tokens.as_ref() {
                    run_end = common_tsid;
                    continue;
                }
                map.extend_simple(std::iter::once((
                    run_start.unwrap()..=run_end,
                    Arc::clone(current),
                )));
            }
            run_start = Some(common_tsid);
            run_end = common_tsid;
            run_tokens = Some(Arc::clone(tokens));
        } else if let Some(ref current) = run_tokens {
            map.extend_simple(std::iter::once((
                run_start.unwrap()..=run_end,
                Arc::clone(current),
            )));
            run_start = None;
            run_tokens = None;
        }
    }
    if let Some(tokens) = run_tokens {
        map.extend_simple(std::iter::once((run_start.unwrap()..=run_end, tokens)));
    }

    finalize_weight_map(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn token_set(tokens: &[u32]) -> RangeSetBlaze<u32> {
        RangeSetBlaze::from_iter(tokens.iter().copied().map(|token| token..=token))
    }

    fn weight_entries(weight: &Weight) -> Vec<(u32, Vec<u32>)> {
        let mut entries = Vec::new();
        for (start, end, tokens) in weight.compact_entries().unwrap_or_default() {
            for tsid in start..=end {
                entries.push((tsid, tokens.iter().collect()));
            }
        }
        entries
    }

    fn runtime_possible_matches(entries: &[(u32, Weight)]) -> RuntimePossibleMatchesByTerminal {
        entries
            .iter()
            .map(|(terminal_id, weight)| (*terminal_id, weight.clone()))
            .collect::<BTreeMap<_, _>>()
    }

    fn mapped_runtime_possible_matches(
        entries: &[(u32, Weight)],
        tokenizer_original_to_internal: Vec<u32>,
        tokenizer_num_internal: u32,
        token_original_to_internal: Vec<u32>,
        token_num_internal: u32,
    ) -> MappedArtifact<RuntimePossibleMatchesByTerminal> {
        MappedArtifact::new(
            runtime_possible_matches(entries),
            test_id_map(
                tokenizer_original_to_internal,
                tokenizer_num_internal,
                token_original_to_internal,
                token_num_internal,
            ),
        )
    }

    fn test_id_map(
        tokenizer_original_to_internal: Vec<u32>,
        tokenizer_num_internal: u32,
        token_original_to_internal: Vec<u32>,
        token_num_internal: u32,
    ) -> InternalIdMap {
        InternalIdMap {
            tokenizer_states: ManyToOneIdMap::from_original_to_internal_with_representatives(
                tokenizer_original_to_internal,
                tokenizer_num_internal,
                (0..tokenizer_num_internal).collect(),
            ),
            vocab_tokens: ManyToOneIdMap::from_original_to_internal_with_representatives(
                token_original_to_internal,
                token_num_internal,
                (0..token_num_internal).collect(),
            ),
        }
    }

    #[test]
    fn reconcile_weight_id_maps_splits_token_classes() {
        let mut left_weight = Weight::from_per_tsid_token_sets(std::iter::once((0, token_set(&[0]))));
        let mut right_weight_a =
            Weight::from_per_tsid_token_sets(std::iter::once((0, token_set(&[0]))));
        let mut right_weight_b =
            Weight::from_per_tsid_token_sets(std::iter::once((0, token_set(&[1]))));

        let mut left_id_map = InternalIdMap {
            tokenizer_states: ManyToOneIdMap::from_original_to_internal_with_representatives(
                vec![0],
                1,
                vec![0],
            ),
            vocab_tokens: ManyToOneIdMap::from_original_to_internal_with_representatives(
                vec![0, 0],
                1,
                vec![0],
            ),
        };
        let mut right_id_map = InternalIdMap {
            tokenizer_states: ManyToOneIdMap::from_original_to_internal_with_representatives(
                vec![0],
                1,
                vec![0],
            ),
            vocab_tokens: ManyToOneIdMap::from_original_to_internal_with_representatives(
                vec![0, 1],
                2,
                vec![0, 1],
            ),
        };

        reconcile_weight_id_maps(
            &mut [&mut left_weight],
            &mut left_id_map,
            &mut [&mut right_weight_a, &mut right_weight_b],
            &mut right_id_map,
        );

        assert_eq!(left_id_map.vocab_tokens.original_to_internal, vec![0, 1]);
        assert_eq!(left_id_map.vocab_tokens.original_to_internal, right_id_map.vocab_tokens.original_to_internal);
        assert_eq!(weight_entries(&left_weight), vec![(0, vec![0, 1])]);
        assert_eq!(weight_entries(&right_weight_a), vec![(0, vec![0])]);
        assert_eq!(weight_entries(&right_weight_b), vec![(0, vec![1])]);
    }

    #[test]
    fn reconcile_weight_id_maps_splits_tsid_classes() {
        let mut left_weight = Weight::from_per_tsid_token_sets(std::iter::once((0, token_set(&[0]))));
        let mut right_weight_left =
            Weight::from_per_tsid_token_sets(std::iter::once((0, token_set(&[0]))));
        let mut right_weight_right =
            Weight::from_per_tsid_token_sets(std::iter::once((1, token_set(&[0]))));

        let mut left_id_map = InternalIdMap {
            tokenizer_states: ManyToOneIdMap::from_original_to_internal_with_representatives(
                vec![0, 0],
                1,
                vec![0],
            ),
            vocab_tokens: ManyToOneIdMap::from_original_to_internal_with_representatives(
                vec![0],
                1,
                vec![0],
            ),
        };
        let mut right_id_map = InternalIdMap {
            tokenizer_states: ManyToOneIdMap::from_original_to_internal_with_representatives(
                vec![0, 1],
                2,
                vec![0, 1],
            ),
            vocab_tokens: ManyToOneIdMap::from_original_to_internal_with_representatives(
                vec![0],
                1,
                vec![0],
            ),
        };

        reconcile_weight_id_maps(
            &mut [&mut left_weight],
            &mut left_id_map,
            &mut [&mut right_weight_left, &mut right_weight_right],
            &mut right_id_map,
        );

        assert_eq!(left_id_map.tokenizer_states.original_to_internal, vec![0, 1]);
        assert_eq!(left_id_map.tokenizer_states.original_to_internal, right_id_map.tokenizer_states.original_to_internal);
        assert_eq!(weight_entries(&left_weight), vec![(0, vec![0]), (1, vec![0])]);
        assert_eq!(weight_entries(&right_weight_left), vec![(0, vec![0])]);
        assert_eq!(weight_entries(&right_weight_right), vec![(1, vec![0])]);
    }

    #[test]
    fn reconcile_mapped_pair_returns_shared_map_and_split_preserves_it() {
        let left = mapped_runtime_possible_matches(
            &[(7, Weight::from_per_tsid_token_sets(std::iter::once((0, token_set(&[0])))))],
            vec![0],
            1,
            vec![0, 0],
            1,
        );
        let right = mapped_runtime_possible_matches(
            &[
                (7, Weight::from_per_tsid_token_sets(std::iter::once((0, token_set(&[0]))))),
                (8, Weight::from_per_tsid_token_sets(std::iter::once((0, token_set(&[1]))))),
            ],
            vec![0],
            1,
            vec![0, 1],
            2,
        );

        let paired = reconcile_mapped_pair(left, right);
        assert_eq!(paired.id_map().vocab_tokens.original_to_internal, vec![0, 1]);

        let (left, right) = paired.split_pair();
        assert_eq!(left.id_map().vocab_tokens.original_to_internal, vec![0, 1]);
        assert_eq!(left.id_map().vocab_tokens.original_to_internal, right.id_map().vocab_tokens.original_to_internal);
        assert_eq!(weight_entries(left.artifact().get(&7).unwrap()), vec![(0, vec![0, 1])]);
        assert_eq!(weight_entries(right.artifact().get(&7).unwrap()), vec![(0, vec![0])]);
        assert_eq!(weight_entries(right.artifact().get(&8).unwrap()), vec![(0, vec![1])]);
    }

    #[test]
    fn reconcile_mapped_vec_and_split_vec_preserve_shared_map() {
        let inputs = vec![
            mapped_runtime_possible_matches(
                &[(7, Weight::from_per_tsid_token_sets(std::iter::once((0, token_set(&[0])))))],
                vec![0],
                1,
                vec![0, 0],
                1,
            ),
            mapped_runtime_possible_matches(
                &[(8, Weight::from_per_tsid_token_sets(std::iter::once((0, token_set(&[1])))))],
                vec![0],
                1,
                vec![0, 1],
                2,
            ),
        ];

        let reconciled = reconcile_mapped_vec(inputs);
        assert_eq!(reconciled.id_map().vocab_tokens.original_to_internal, vec![0, 1]);
        assert_eq!(reconciled.artifact().len(), 2);

        let split = reconciled.split_vec();
        assert_eq!(split.len(), 2);
        assert_eq!(split[0].id_map().vocab_tokens.original_to_internal, vec![0, 1]);
        assert_eq!(split[0].id_map().vocab_tokens.original_to_internal, split[1].id_map().vocab_tokens.original_to_internal);
        assert_eq!(weight_entries(split[0].artifact().get(&7).unwrap()), vec![(0, vec![0, 1])]);
        assert_eq!(weight_entries(split[1].artifact().get(&8).unwrap()), vec![(0, vec![1])]);
    }
}

