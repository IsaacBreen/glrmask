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
    pub old_num_tsids: u32,
    pub new_num_tsids: u32,
    pub old_num_tokens: u32,
    pub new_num_tokens: u32,
    pub old_ranges: usize,
    pub new_ranges: usize,
    pub old_weight_ranges: usize,
    pub new_weight_ranges: usize,
    pub old_unique_token_ranges: usize,
    pub new_unique_token_ranges: usize,
    pub token_perm: Vec<u32>,
}

pub struct StochasticCompactProbeReport {
    pub baseline_ranges: usize,
    pub best_ranges: usize,
    pub baseline_weight_ranges: usize,
    pub best_weight_ranges: usize,
    pub baseline_token_ranges: usize,
    pub best_token_ranges: usize,
    pub iterations: usize,
    pub best_iteration: usize,
    pub seed: u64,
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

/// Merge equivalent IDs and reorder both dimensions of every weight in `dwa`,
/// updating `id_map` to match.
pub fn compact_dwa_dimensions(
    dwa: &mut DWA,
    id_map: &mut InternalIdMap,
) -> CompactReport {
    let num_tsids = id_map.num_tsids();
    let num_tokens = id_map.num_internal_tokens();

    let old_storage = count_unique_storage(dwa);
    let old_ranges = old_storage.total_ranges();

    // Step 1  — collect unique weights (by Arc pointer)
    let unique_weights = collect_unique_weights(dwa);
    let weight_refs: Vec<&Weight> = unique_weights.iter().collect();

    // Step 2  — tsid dimension: merge + reorder
    let tsid_merge_profiles = build_tsid_profiles(&weight_refs, num_tsids as usize);
    let tsid_order_profiles = build_tsid_order_profiles(&weight_refs, num_tsids as usize);
    let (tsid_perm, new_num_tsids) =
        merge_sort_perm_with_group_order(&tsid_merge_profiles, &tsid_order_profiles);

    // Step 3  — token dimension: merge + reorder
    let token_profiles = build_token_profiles(&weight_refs, num_tokens);
    let (token_perm, new_num_tokens) = merge_sort_perm(&token_profiles);
    let merged_unique_token_sets = collect_merged_unique_token_sets(&unique_weights, &token_perm);
    let token_perm = optimize_token_order_locally(
        &merged_unique_token_sets,
        token_perm,
        new_num_tokens,
    );

    // Step 4  — apply permutations to every weight in the DWA
    apply_permutations_to_dwa(dwa, &unique_weights, &tsid_perm, &token_perm);

    // Step 5  — update id_map so the runtime still maps correctly
    apply_perm_to_id_map(&mut id_map.tokenizer_states, &tsid_perm, new_num_tsids);
    apply_perm_to_id_map(&mut id_map.vocab_tokens, &token_perm, new_num_tokens);

    let new_storage = count_unique_storage(dwa);
    let new_ranges = new_storage.total_ranges();

    CompactReport {
        old_num_tsids: num_tsids,
        new_num_tsids: new_num_tsids as u32,
        old_num_tokens: num_tokens,
        new_num_tokens: new_num_tokens as u32,
        old_ranges,
        new_ranges,
        old_weight_ranges: old_storage.weight_ranges,
        new_weight_ranges: new_storage.weight_ranges,
        old_unique_token_ranges: old_storage.token_ranges,
        new_unique_token_ranges: new_storage.token_ranges,
        token_perm,
    }
}

pub fn probe_stochastic_token_reordering(
    dwa: &DWA,
    num_tsids: u32,
    num_tokens: u32,
    iterations: usize,
    seed: u64,
) -> StochasticCompactProbeReport {
    let unique_weights = collect_unique_weights(dwa);
    let identity_token_perm: Vec<u32> = (0..num_tokens).collect();
    let merged_unique_token_sets =
        collect_merged_unique_token_sets(&unique_weights, &identity_token_perm);
    let baseline_storage = count_unique_storage_for_weights(&unique_weights);
    let mut rng = StdRng::seed_from_u64(seed);

    let mut current_perm: Vec<u32> = (0..num_tokens).collect();
    let baseline_token_ranges =
        count_token_ranges_after_group_permutation(&merged_unique_token_sets, &current_perm);
    let baseline_ranges = baseline_storage.weight_ranges + baseline_token_ranges;

    let mut current_token_ranges = baseline_token_ranges;
    let mut current_ranges = baseline_ranges;

    let mut best_token_ranges = baseline_token_ranges;
    let mut best_ranges = baseline_ranges;
    let mut best_iteration = 0usize;

    let mut temperature = 8.0f64;
    for iteration in 1..=iterations {
        let mut candidate_perm = current_perm.clone();
        apply_random_token_move(&mut candidate_perm, &mut rng);
        let candidate_token_ranges =
            count_token_ranges_after_group_permutation(&merged_unique_token_sets, &candidate_perm);
        let candidate_ranges = baseline_storage.weight_ranges + candidate_token_ranges;

        let better_than_best = candidate_ranges < best_ranges
            || (candidate_ranges == best_ranges
                && candidate_token_ranges < best_token_ranges);
        if better_than_best {
            best_ranges = candidate_ranges;
            best_token_ranges = candidate_token_ranges;
            best_iteration = iteration;
        }

        let delta = candidate_ranges as i64 - current_ranges as i64;
        let accept = if delta <= 0 {
            true
        } else {
            let probability = (-(delta as f64) / temperature.max(0.1)).exp().clamp(0.0, 1.0);
            rng.gen_bool(probability)
        };

        if accept {
            current_perm = candidate_perm;
            current_ranges = candidate_ranges;
            current_token_ranges = candidate_token_ranges;
        }

        let _ = current_token_ranges;
        temperature *= 0.995;
    }

    StochasticCompactProbeReport {
        baseline_ranges,
        best_ranges,
        baseline_weight_ranges: baseline_storage.weight_ranges,
        best_weight_ranges: baseline_storage.weight_ranges,
        baseline_token_ranges,
        best_token_ranges,
        iterations,
        best_iteration,
        seed,
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

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

fn rangeset_key(set: &RangeSetBlaze<u32>) -> Vec<(u32, u32)> {
    set.ranges()
        .map(|range| (*range.start(), *range.end()))
        .collect()
}

/// For each tsid, list the (weight, entry) context indices where it appears.
fn build_tsid_profiles(weights: &[&Weight], num_tsids: usize) -> Vec<Vec<u32>> {
    let mut profiles = vec![Vec::new(); num_tsids];
    let mut ctx = 0u32;
    for w in weights {
        for (tsid_range, _token_set) in w.0.range_values() {
            for tsid in *tsid_range.start()..=*tsid_range.end() {
                if (tsid as usize) < num_tsids {
                    profiles[tsid as usize].push(ctx);
                }
            }
            ctx += 1;
        }
    }
    profiles
}

/// For ordering only: treat repeated equal token sets within the same weight as
/// the same context so semantically similar TSIDs can become adjacent even when
/// they must remain distinct IDs.
fn build_tsid_order_profiles(weights: &[&Weight], num_tsids: usize) -> Vec<Vec<u32>> {
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
fn merge_sort_perm<P: Ord + std::hash::Hash + Eq>(profiles: &[P]) -> (Vec<u32>, usize) {
    let n = profiles.len();
    if n == 0 {
        return (vec![], 0);
    }

    // Group old IDs by profile: profile → list of old IDs with that profile
    let mut profile_groups: HashMap<usize, Vec<usize>> = HashMap::new();
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
    drop(profile_groups);

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

fn merge_sort_perm_with_group_order<Pm, Po>(merge_profiles: &[Pm], order_profiles: &[Po]) -> (Vec<u32>, usize)
where
    Pm: Ord + std::hash::Hash + Eq,
    Po: Ord,
{
    let n = merge_profiles.len();
    if n == 0 {
        return (vec![], 0);
    }
    assert_eq!(merge_profiles.len(), order_profiles.len());

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| merge_profiles[a].cmp(&merge_profiles[b]));

    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current_group = vec![sorted_indices[0]];
    for &idx in &sorted_indices[1..] {
        if merge_profiles[idx] == merge_profiles[current_group[0]] {
            current_group.push(idx);
        } else {
            groups.push(std::mem::take(&mut current_group));
            current_group.push(idx);
        }
    }
    groups.push(current_group);

    groups.sort_by(|left, right| {
        order_profiles[left[0]]
            .cmp(&order_profiles[right[0]])
            .then_with(|| merge_profiles[left[0]].cmp(&merge_profiles[right[0]]))
    });

    let new_count = groups.len();
    let mut perm = vec![0u32; n];
    for (new_id, group) in groups.iter().enumerate() {
        for &old_id in group {
            perm[old_id] = new_id as u32;
        }
    }

    (perm, new_count)
}

fn unique_storage_better(candidate: UniqueStorageCounts, current: UniqueStorageCounts) -> bool {
    candidate.total_ranges() < current.total_ranges()
        || (candidate.total_ranges() == current.total_ranges()
            && (candidate.token_ranges < current.token_ranges
                || (candidate.token_ranges == current.token_ranges
                    && candidate.weight_ranges < current.weight_ranges)))
}

fn collect_merged_unique_token_sets(weights: &[Weight], merge_token_perm: &[u32]) -> Vec<RangeSetBlaze<u32>> {
    let mut seen = std::collections::HashSet::new();
    let mut unique_sets = Vec::new();
    for weight in weights {
        for (_, token_set) in weight.0.range_values() {
            let merged = permute_rangeset(token_set, merge_token_perm);
            let key = rangeset_key(&merged);
            if seen.insert(key) {
                unique_sets.push(merged);
            }
        }
    }
    unique_sets
}

fn count_token_ranges_after_group_permutation(
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

fn run_adjacent_swap_passes(
    merged_unique_token_sets: &[RangeSetBlaze<u32>],
    layout: &mut Vec<u32>,
    group_positions: &mut Vec<u32>,
    current_token_ranges: &mut usize,
) {
    for _ in 0..TOKEN_ORDER_LOCAL_SEARCH_PASSES {
        let mut improved = false;
        for left_pos in 0..(layout.len() - 1) {
            let right_pos = left_pos + 1;
            let left_group = layout[left_pos] as usize;
            let right_group = layout[right_pos] as usize;

            let mut candidate_positions = group_positions.clone();
            candidate_positions[left_group] = right_pos as u32;
            candidate_positions[right_group] = left_pos as u32;
            let candidate_token_ranges = count_token_ranges_after_group_permutation(
                merged_unique_token_sets,
                &candidate_positions,
            );
            if candidate_token_ranges < *current_token_ranges {
                *group_positions = candidate_positions;
                layout.swap(left_pos, right_pos);
                *current_token_ranges = candidate_token_ranges;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
}

fn finish_token_order_with_seeded_search(
    merged_unique_token_sets: &[RangeSetBlaze<u32>],
    initial_group_positions: Vec<u32>,
) -> Vec<u32> {
    if TOKEN_ORDER_FINISH_ITERS == 0 || merged_unique_token_sets.is_empty() {
        return initial_group_positions;
    }

    let mut rng = StdRng::seed_from_u64(TOKEN_ORDER_FINISH_SEED);
    let mut current_positions = initial_group_positions.clone();
    let mut current_token_ranges =
        count_token_ranges_after_group_permutation(merged_unique_token_sets, &current_positions);
    let mut best_positions = current_positions.clone();
    let mut best_token_ranges = current_token_ranges;

    let mut temperature = 8.0f64;
    for _ in 0..TOKEN_ORDER_FINISH_ITERS {
        let mut candidate_positions = current_positions.clone();
        apply_random_token_move(&mut candidate_positions, &mut rng);
        let candidate_token_ranges = count_token_ranges_after_group_permutation(
            merged_unique_token_sets,
            &candidate_positions,
        );

        if candidate_token_ranges < best_token_ranges {
            best_token_ranges = candidate_token_ranges;
            best_positions = candidate_positions.clone();
        }

        let delta = candidate_token_ranges as i64 - current_token_ranges as i64;
        let accept = if delta <= 0 {
            true
        } else {
            let probability = (-(delta as f64) / temperature.max(0.1)).exp().clamp(0.0, 1.0);
            rng.gen_bool(probability)
        };

        if accept {
            current_positions = candidate_positions;
            current_token_ranges = candidate_token_ranges;
        }

        temperature *= 0.995;
    }

    best_positions
}

fn optimize_token_order_locally(
    merged_unique_token_sets: &[RangeSetBlaze<u32>],
    initial_token_perm: Vec<u32>,
    new_num_tokens: usize,
) -> Vec<u32> {
    if new_num_tokens < 2 || merged_unique_token_sets.is_empty() {
        return initial_token_perm;
    }

    let mut layout: Vec<u32> = (0..new_num_tokens as u32).collect();
    let mut group_positions = layout.clone();
    let mut current_token_ranges =
        count_token_ranges_after_group_permutation(merged_unique_token_sets, &group_positions);

    run_adjacent_swap_passes(
        merged_unique_token_sets,
        &mut layout,
        &mut group_positions,
        &mut current_token_ranges,
    );

    group_positions = finish_token_order_with_seeded_search(
        merged_unique_token_sets,
        group_positions,
    );

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

fn apply_random_token_move(perm: &mut [u32], rng: &mut StdRng) {
    if perm.len() < 2 {
        return;
    }

    match rng.gen_range(0..3) {
        0 => {
            let left = rng.gen_range(0..perm.len());
            let mut right = rng.gen_range(0..perm.len());
            if left == right {
                right = (right + 1) % perm.len();
            }
            perm.swap(left, right);
        }
        1 => {
            let left = rng.gen_range(0..perm.len() - 1);
            perm.swap(left, left + 1);
        }
        _ => {
            let from = rng.gen_range(0..perm.len());
            let to = rng.gen_range(0..perm.len());
            if from != to {
                let value = perm[from];
                if from < to {
                    perm.copy_within((from + 1)..=to, from);
                } else {
                    perm.copy_within(to..from, to + 1);
                }
                perm[to] = value;
            }
        }
    }
}

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
    fn local_token_order_search_reduces_unique_token_ranges() {
        let weights = vec![
            Weight::from_uniform(0..=0, RangeSetBlaze::from_iter([0..=0, 2..=2])),
            Weight::from_uniform(0..=0, RangeSetBlaze::from_iter([1..=1])),
        ];
        let initial_token_perm = vec![0, 1, 2];
        let merged_unique_token_sets = collect_merged_unique_token_sets(&weights, &initial_token_perm);

        let baseline = score_permuted_weights(&weights, &[0], &initial_token_perm);
        let optimized = optimize_token_order_locally(&merged_unique_token_sets, initial_token_perm, 3);
        let improved = score_permuted_weights(&weights, &[0], &optimized);

        assert!(unique_storage_better(improved, baseline));
        assert_eq!(improved.weight_ranges, baseline.weight_ranges);
        assert!(improved.token_ranges < baseline.token_ranges);
    }
}
