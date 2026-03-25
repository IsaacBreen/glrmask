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

use super::equivalence_analysis::{InternalIdMap, ManyToOneIdMap};

// ── public entry point ──────────────────────────────────────────────────────

const TOKEN_ORDER_LOCAL_SEARCH_PASSES: usize = 3;
const TOKEN_ORDER_FINISH_ITERS: usize = 20000;
const TOKEN_ORDER_FINISH_SEED: u64 = 7;

pub struct CompactReport {
    pub tsid_perm: Vec<u32>,
    pub token_perm: Vec<u32>,
}

struct DimensionCompaction {
    tsid_perm: Vec<u32>,
    ordered_num_tsids: usize,
    token_perm: Vec<u32>,
    ordered_num_tokens: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct UniqueStorageCounts {
    weight_ranges: usize,
    token_ranges: usize,
}

#[cfg(test)]
impl UniqueStorageCounts {
    fn total_ranges(self) -> usize {
        self.weight_ranges + self.token_ranges
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
) -> CompactReport {
    let num_tsids = id_map.num_tsids();
    let num_tokens = id_map.num_internal_tokens();

    let unique_weights = collect_unique_weights(dwa);
    let compaction = build_dimension_compaction(&unique_weights, num_tsids as usize, num_tokens);

    apply_permutations_to_dwa(
        dwa,
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

    CompactReport {
        tsid_perm: compaction.tsid_perm,
        token_perm: compaction.token_perm,
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn build_dimension_compaction(
    unique_weights: &[Weight],
    num_tsids: usize,
    num_tokens: u32,
) -> DimensionCompaction {
    let original_weight_refs = weight_refs(unique_weights);

    let tsid_merge_profiles = build_tsid_context_profiles(&original_weight_refs, num_tsids);
    let (tsid_merge_perm, merged_num_tsids) = build_profile_merge_permutation(&tsid_merge_profiles);
    let merged_tsid_weights = apply_permutations_to_weight_set(
        unique_weights,
        &tsid_merge_perm,
        &identity_perm(num_tokens as usize),
    );

    let merged_tsid_refs = weight_refs(&merged_tsid_weights);
    let tsid_order_profiles = build_tsid_context_profiles(&merged_tsid_refs, merged_num_tsids);
    let (tsid_order_perm, ordered_num_tsids) =
        build_profile_merge_permutation(&tsid_order_profiles);

    let token_profiles = build_token_profiles(&original_weight_refs, num_tokens);
    let (token_group_perm, ordered_num_tokens) =
        build_profile_merge_permutation(&token_profiles);
    let merged_token_sets = collect_token_sets_after_permutation(unique_weights, &token_group_perm);
    let token_perm = optimize_token_group_order(
        &merged_token_sets,
        token_group_perm,
        ordered_num_tokens,
    );

    DimensionCompaction {
        tsid_perm: compose_perm(&tsid_merge_perm, &tsid_order_perm),
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
    for state in &dwa.states {
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
    dedup_weights_by_storage_ptr(
        weights
            .iter()
            .map(|weight| permute_weight(weight, tsid_perm, token_perm))
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
    let mut seen = std::collections::HashSet::new();
    let mut unique_sets = Vec::new();
    for weight in weights {
        for (_, token_set) in weight.0.range_values() {
            let merged = permute_rangeset(token_set, token_perm);
            let key = rangeset_key(&merged);
            if seen.insert(key) {
                unique_sets.push(merged);
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

    let mut temperature = 8.0f64;
    for _ in 0..TOKEN_ORDER_FINISH_ITERS {
        let mut candidate_layout = current_layout.clone();
        apply_random_layout_move(&mut candidate_layout, &mut rng);
        let candidate_score = scorer.score_layout(&candidate_layout);

        if candidate_score < best_score {
            best_score = candidate_score;
            best_layout = candidate_layout.clone();
        }

        let delta = candidate_score as i64 - current_score as i64;
        let accept = if delta <= 0 {
            true
        } else {
            let probability = (-(delta as f64) / temperature.max(0.1)).exp().clamp(0.0, 1.0);
            rng.gen_bool(probability)
        };

        if accept {
            current_layout = candidate_layout;
            current_score = candidate_score;
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

/// Apply tsid and token permutations (possibly many-to-one) to every weight.
fn apply_permutations_to_dwa(
    dwa: &mut DWA,
    unique_weights: &[Weight],
    tsid_perm: &[u32],
    token_perm: &[u32],
) {
    let mut weight_map: HashMap<usize, Weight> = HashMap::with_capacity(unique_weights.len());
    for w in unique_weights {
        let old_ptr = Arc::as_ptr(&w.0) as usize;
        let new_w = permute_weight(w, tsid_perm, token_perm);
        weight_map.insert(old_ptr, new_w);
    }

    for state in &mut dwa.states {
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
    if w.is_empty() {
        return Weight::empty();
    }
    if w.is_full() {
        return Weight::all();
    }

    let mut permuted_token_cache: HashMap<usize, RangeSetBlaze<u32>> = HashMap::new();
    let mut tokens_by_new_tsid: std::collections::BTreeMap<u32, RangeSetBlaze<u32>> =
        std::collections::BTreeMap::new();

    for (tsid_range, token_set) in w.0.range_values() {
        let ts_ptr = Arc::as_ptr(token_set) as usize;
        let new_ts = permuted_token_cache
            .entry(ts_ptr)
            .or_insert_with(|| permute_rangeset(token_set, token_perm))
            .clone();

        for tsid in *tsid_range.start()..=*tsid_range.end() {
            if (tsid as usize) < tsid_perm.len() {
                let new_tsid = tsid_perm[tsid as usize];
                tokens_by_new_tsid
                    .entry(new_tsid)
                    .and_modify(|existing| *existing |= new_ts.clone())
                    .or_insert_with(|| new_ts.clone());
            }
        }
    }

    if tokens_by_new_tsid.is_empty() {
        return Weight::empty();
    }

    let mut ordered_pairs: Vec<(u32, RangeSetBlaze<u32>)> = tokens_by_new_tsid.into_iter().collect();
    ordered_pairs.sort_unstable_by_key(|(tsid, _)| *tsid);

    // Merge consecutive tsids with the same token set and rebuild through the
    // shared weight/token-set interner so unique-storage accounting is real.
    let mut map = RangeMapBlaze::new();
    let mut pairs = ordered_pairs.into_iter();
    let (mut run_start, mut run_tokens) = pairs.next().unwrap();
    let mut run_end = run_start;

    for (tsid, tokens) in pairs {
        if tsid == run_end + 1 && tokens == run_tokens {
            run_end = tsid;
        } else {
            map.extend_simple(std::iter::once((
                run_start..=run_end,
                shared_rangeset(std::mem::take(&mut run_tokens)),
            )));
            run_start = tsid;
            run_end = tsid;
            run_tokens = tokens;
        }
    }
    map.extend_simple(std::iter::once((run_start..=run_end, shared_rangeset(run_tokens))));

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

    let mut new_internal_to_originals = vec![RangeSetBlaze::new(); new_count];
    let mut new_representatives = vec![u32::MAX; new_count];
    for (old_internal, originals) in old_internal_to_originals.into_iter().enumerate() {
        let Some(&new_internal) = perm.get(old_internal) else {
            continue;
        };
        if (new_internal as usize) >= new_count {
            continue;
        }
        new_internal_to_originals[new_internal as usize] |= originals;
        if new_representatives[new_internal as usize] == u32::MAX {
            new_representatives[new_internal as usize] = old_representatives[old_internal];
        }
    }
    id_map.internal_to_originals = new_internal_to_originals;
    id_map.representative_original_ids = new_representatives;
}

#[cfg(test)]
fn count_unique_storage(dwa: &DWA) -> UniqueStorageCounts {
    let unique_weights = collect_unique_weights(dwa);
    count_unique_storage_for_weights(&unique_weights)
}

#[cfg(test)]
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
