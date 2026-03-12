//! DWA dimension compaction: merge equivalent IDs and reorder for adjacency.
//!
//! Merges tsid (outer) and token (inner) IDs that have identical weight
//! profiles, then reorders the remaining IDs so that similar elements are
//! placed adjacently. This reduces the number of ranges in the underlying
//! `RangeMapBlaze` / `RangeSetBlaze` structures.
//!
//! Token merging is additionally gated on matching possible-matches
//! fingerprints so that downstream mask correctness is preserved.

use std::collections::HashMap;
use std::sync::Arc;

use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};

use crate::automata::weighted_u32::dwa::DWA;
use crate::compiler::grammar_def::TerminalID;
use crate::compiler::possible_matches::PossibleMatchesByState;
use crate::ds::weight::Weight;

use super::equivalence_analysis::{InternalIdMap, ManyToOneIdMap};

// ── public entry point ──────────────────────────────────────────────────────

pub struct CompactReport {
    pub old_num_tsids: u32,
    pub new_num_tsids: u32,
    pub old_num_tokens: u32,
    pub new_num_tokens: u32,
    pub old_ranges: usize,
    pub new_ranges: usize,
}

/// Merge equivalent IDs and reorder both dimensions of every weight in `dwa`,
/// updating `id_map` to match.
///
/// `pre_compact_possible_matches` is the possible-matches map computed from the
/// pre-compact id_map; it is used only for token merge fingerprinting.
pub fn compact_dwa_dimensions(
    dwa: &mut DWA,
    id_map: &mut InternalIdMap,
    pre_compact_possible_matches: &PossibleMatchesByState,
) -> CompactReport {
    let num_tsids = id_map.num_tsids();
    let num_tokens = id_map.num_internal_tokens();

    let old_ranges = count_total_ranges(dwa);

    // Step 1  — collect unique weights (by Arc pointer)
    let unique_weights = collect_unique_weights(dwa);
    let weight_refs: Vec<&Weight> = unique_weights.iter().collect();

    // Step 2  — tsid dimension: merge + reorder
    let tsid_profiles = build_tsid_profiles(&weight_refs, num_tsids as usize);
    let (tsid_perm, new_num_tsids) = merge_sort_perm(&tsid_profiles);

    // Step 3  — token dimension: merge + reorder
    //           Gate merging on matching possible-matches fingerprints.
    let token_profiles = build_token_profiles(&weight_refs, num_tokens);
    let token_fingerprints = build_token_fingerprints(pre_compact_possible_matches, num_tokens);
    let augmented_token_profiles: Vec<(Vec<u32>, Vec<(u32, TerminalID)>)> = token_profiles
        .into_iter()
        .zip(token_fingerprints.into_iter())
        .collect();
    let (token_perm, new_num_tokens) = merge_sort_perm(&augmented_token_profiles);

    // Step 4  — apply permutations to every weight in the DWA
    apply_permutations_to_dwa(dwa, &unique_weights, &tsid_perm, &token_perm);

    // Step 5  — update id_map so the runtime still maps correctly
    apply_perm_to_id_map(&mut id_map.tokenizer_states, &tsid_perm, new_num_tsids);
    apply_perm_to_id_map(&mut id_map.vocab_tokens, &token_perm, new_num_tokens);

    let new_ranges = count_total_ranges(dwa);

    CompactReport {
        old_num_tsids: num_tsids,
        new_num_tsids: new_num_tsids as u32,
        old_num_tokens: num_tokens,
        new_num_tokens: new_num_tokens as u32,
        old_ranges,
        new_ranges,
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

/// For each internal token, collect a sorted fingerprint of which
/// (original_state, terminal) pairs it appears in within possible_matches.
fn build_token_fingerprints(
    pm: &PossibleMatchesByState,
    num_tokens: u32,
) -> Vec<Vec<(u32, TerminalID)>> {
    let n = num_tokens as usize;
    let mut fingerprints = vec![Vec::new(); n];
    for (&state, by_terminal) in pm {
        for (&terminal, token_set) in by_terminal {
            for token in token_set.iter() {
                if (token as usize) < n {
                    fingerprints[token as usize].push((state, terminal));
                }
            }
        }
    }
    // Sort each fingerprint for consistent comparison
    for fp in &mut fingerprints {
        fp.sort_unstable();
    }
    fingerprints
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

/// Create a new Weight from `w` with permuted (possibly many-to-one) coords.
fn permute_weight(w: &Weight, tsid_perm: &[u32], token_perm: &[u32]) -> Weight {
    if w.is_empty() {
        return Weight::empty();
    }
    if w.is_full() {
        return Weight::all();
    }

    let mut ts_cache: HashMap<usize, Arc<RangeSetBlaze<u32>>> = HashMap::new();
    let mut pairs: Vec<(u32, Arc<RangeSetBlaze<u32>>)> = Vec::new();

    for (tsid_range, token_set) in w.0.range_values() {
        let ts_ptr = Arc::as_ptr(token_set) as usize;
        let new_ts = ts_cache
            .entry(ts_ptr)
            .or_insert_with(|| Arc::new(permute_rangeset(token_set, token_perm)));

        for tsid in *tsid_range.start()..=*tsid_range.end() {
            if (tsid as usize) < tsid_perm.len() {
                pairs.push((tsid_perm[tsid as usize], Arc::clone(new_ts)));
            }
        }
    }

    pairs.sort_unstable_by_key(|(t, _)| *t);

    if pairs.is_empty() {
        return Weight::empty();
    }

    // Deduplicate: merged tsids may produce duplicate (new_tsid, token_set) pairs.
    // Since merged tsids share the same token_set (same profile → same entries),
    // duplicates will have the same token_set Arc.
    pairs.dedup_by(|b, a| a.0 == b.0);

    // Merge consecutive tsids with the same token_set Arc into ranges
    let mut map = RangeMapBlaze::<u32, Arc<RangeSetBlaze<u32>>>::new();
    let mut run_start = pairs[0].0;
    let mut run_end = pairs[0].0;
    let mut run_ts = Arc::clone(&pairs[0].1);

    for &(tsid, ref ts) in &pairs[1..] {
        if tsid == run_end + 1 && Arc::ptr_eq(&run_ts, ts) {
            run_end = tsid;
        } else {
            map.extend_simple(std::iter::once((run_start..=run_end, Arc::clone(&run_ts))));
            run_start = tsid;
            run_end = tsid;
            run_ts = Arc::clone(ts);
        }
    }
    map.extend_simple(std::iter::once((run_start..=run_end, run_ts)));

    Weight(Arc::new(map))
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
    for internal in &mut id_map.original_to_internal {
        if *internal != u32::MAX {
            if let Some(&new_id) = perm.get(*internal as usize) {
                *internal = new_id;
            }
        }
    }

    let mut new_internal_to_originals = vec![Vec::new(); new_count];
    for (original, &internal) in id_map.original_to_internal.iter().enumerate() {
        if internal != u32::MAX && (internal as usize) < new_count {
            new_internal_to_originals[internal as usize].push(original as u32);
        }
    }
    id_map.internal_to_originals = new_internal_to_originals;
}

/// Count total ranges across all unique weights in the DWA.
fn count_total_ranges(dwa: &DWA) -> usize {
    let mut seen = std::collections::HashSet::new();
    let mut total = 0;
    for state in &dwa.states {
        for (_, (_, w)) in &state.transitions {
            if seen.insert(Arc::as_ptr(&w.0) as usize) {
                total += count_weight_ranges(w);
            }
        }
        if let Some(fw) = &state.final_weight {
            if seen.insert(Arc::as_ptr(&fw.0) as usize) {
                total += count_weight_ranges(fw);
            }
        }
    }
    total
}

fn count_weight_ranges(w: &Weight) -> usize {
    let mut n = 0;
    for (_, token_set) in w.0.range_values() {
        n += 1;
        n += token_set.ranges().count();
    }
    n
}
