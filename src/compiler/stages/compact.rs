//! DWA dimension compaction: merge equivalent IDs and reorder for adjacency.
//!
//! Merges tsid (outer) and token (inner) IDs that have identical weight
//! profiles, then reorders the remaining IDs so that similar elements are
//! placed adjacently. This reduces the number of ranges in the underlying
//! `RangeMapBlaze` / `RangeSetBlaze` structures.

use std::collections::HashMap;
use std::sync::Arc;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};

use crate::automata::weighted_u32::dwa::DWA;
use crate::ds::weight::{Weight, finalize_weight_map, shared_rangeset};

use super::equiv_types::{InternalIdMap, ManyToOneIdMap};

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

/// Controls DWA dimension compaction behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompactMode {
    /// Skip compaction entirely.
    None,
    /// Merge equivalent IDs but skip token reordering.
    Fast,
    /// Full compaction with token reordering.
    Full,
}

/// Run compaction according to the selected default mode.
pub fn compact_from_env(
    dwa: &mut DWA,
    id_map: &mut InternalIdMap,
    _env_var: &str,
    default: CompactMode,
    collect_profile_stats: bool,
) -> CompactReport {
    match default {
        CompactMode::None => {
            let n_tsids = id_map.num_tsids() as usize;
            let n_tokens = id_map.num_internal_tokens() as usize;
            CompactReport {
                tsid_perm: (0..n_tsids as u32).collect(),
                token_perm: (0..n_tokens as u32).collect(),
                profile_stats: None,
            }
        }
        CompactMode::Fast => {
            compact_dwa_dimensions_inner(dwa, id_map, collect_profile_stats, true)
        }
        CompactMode::Full => {
            compact_dwa_dimensions_inner(dwa, id_map, collect_profile_stats, false)
        }
    }
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
    pub fn total_ranges_before(self) -> usize {
        self.weight_ranges_before + self.token_ranges_before
    }

    pub fn total_ranges_after(self) -> usize {
        self.weight_ranges_after + self.token_ranges_after
    }
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

/// Merge equivalent IDs and reorder both dimensions of every weight in `dwa`,
/// updating `id_map` to match.
pub fn compact_dwa_dimensions(
    dwa: &mut DWA,
    id_map: &mut InternalIdMap,
    collect_profile_stats: bool,
) -> CompactReport {
    compact_dwa_dimensions_inner(dwa, id_map, collect_profile_stats, false)
}

pub fn compact_dwa_dimensions_fast(
    dwa: &mut DWA,
    id_map: &mut InternalIdMap,
) -> CompactReport {
    compact_dwa_dimensions_inner(dwa, id_map, false, true)
}

pub fn compact_dwa_dimensions_fast_with_stats(
    dwa: &mut DWA,
    id_map: &mut InternalIdMap,
) -> CompactReport {
    compact_dwa_dimensions_inner(dwa, id_map, true, true)
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

#[cfg(test)]
fn unique_storage_better(candidate: UniqueStorageCounts, current: UniqueStorageCounts) -> bool {
    candidate.total_ranges() < current.total_ranges()
        || (candidate.total_ranges() == current.total_ranges()
            && (candidate.token_ranges < current.token_ranges
                || (candidate.token_ranges == current.token_ranges
                    && candidate.weight_ranges < current.weight_ranges)))
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

#[cfg(test)]
fn count_token_ranges_after_group_permutation_exact(
    merged_unique_token_sets: &[RangeSetBlaze<u32>],
    group_positions: &[u32],
) -> usize {
    merged_unique_token_sets
        .iter()
        .map(|token_set| permute_rangeset(token_set, group_positions).ranges().count())
        .sum()
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

#[cfg(test)]
fn score_permuted_weights(
    weights: &[Weight],
    tsid_perm: &[u32],
    token_perm: &[u32],
) -> UniqueStorageCounts {
    let permuted: Vec<_> = weights
        .iter()
        .map(|weight| permute_weight(weight, tsid_perm, token_perm))
        .collect();
    count_unique_storage_for_weights(&permuted)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ds::weight::Weight;

    #[test]
    fn permute_weight_unions_duplicate_new_tsids() {
        let weight = Weight::from_compact_ranges([
            (0..=0, [0..=0]),
            (1..=1, [1..=1]),
            (2..=2, [0..=0]),
            (3..=3, [1..=1]),
        ]);

        let permuted = permute_weight(&weight, &[0, 1, 0, 1], &[0, 1]);

        assert_eq!(permuted.tokens_for_tsid(0), RangeSetBlaze::from_iter([0..=0]));
        assert_eq!(permuted.tokens_for_tsid(1), RangeSetBlaze::from_iter([1..=1]));
        assert!(permuted.tokens_for_tsid(2).is_empty());
        assert!(permuted.tokens_for_tsid(3).is_empty());
    }

    #[test]
    fn tsid_merge_profiles_collapse_repeated_equal_token_sets() {
        let weight = Weight::from_compact_ranges([
            (0..=0, [0..=0]),
            (1..=1, [1..=1]),
            (2..=2, [0..=0]),
        ]);
        let weights = vec![&weight];

        let profiles = build_tsid_context_profiles(&weights, 3);
        let (perm, new_count) = build_profile_merge_permutation(&profiles);

        assert_eq!(new_count, 2);
        assert_eq!(perm[0], perm[2]);
        assert_ne!(perm[0], perm[1]);
    }

    #[test]
    fn permute_weight_reuses_interned_weight_storage() {
        let weight = Weight::from_compact_ranges([
            (0..=0, [0..=0]),
            (1..=1, [1..=1]),
            (2..=2, [0..=0]),
        ]);

        let permuted_a = permute_weight(&weight, &[0, 1, 2], &[0, 1]);
        let permuted_b = permute_weight(&weight, &[0, 1, 2], &[0, 1]);

        assert!(Arc::ptr_eq(&permuted_a.0, &permuted_b.0));
    }

    #[test]
    fn token_order_scorer_matches_exact_range_count() {
        let merged_unique_token_sets = vec![
            RangeSetBlaze::from_iter([0..=0, 2..=2]),
            RangeSetBlaze::from_iter([0..=1, 3..=3]),
            RangeSetBlaze::from_iter([1..=2]),
        ];
        let scorer = TokenOrderScorer::new(&merged_unique_token_sets, 4);
        let layout = vec![2, 0, 3, 1];
        let group_positions = layout_to_group_positions(&layout);

        assert_eq!(
            scorer.score_layout(&layout),
            count_token_ranges_after_group_permutation_exact(
                &merged_unique_token_sets,
                &group_positions,
            ),
        );
    }

    #[test]
    fn local_token_order_search_reduces_unique_token_ranges() {
        let weights = vec![
            Weight::from_uniform(0..=0, RangeSetBlaze::from_iter([0..=0, 2..=2])),
            Weight::from_uniform(0..=0, RangeSetBlaze::from_iter([1..=1])),
        ];
        let initial_token_perm = vec![0, 1, 2];
        let merged_unique_token_sets =
            collect_token_sets_after_permutation(&weights, &initial_token_perm);

        let baseline = score_permuted_weights(&weights, &[0], &initial_token_perm);
        let optimized = optimize_token_group_order(&merged_unique_token_sets, initial_token_perm, 3);
        let improved = score_permuted_weights(&weights, &[0], &optimized);

        assert!(unique_storage_better(improved, baseline));
        assert_eq!(improved.weight_ranges, baseline.weight_ranges);
        assert!(improved.token_ranges < baseline.token_ranges);
    }
}
