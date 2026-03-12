//! DWA dimension reordering and equivalence merging.
//!
//! Reorders the tsid (outer) and token (inner) dimensions of all weights in a
//! DWA so that elements with identical behaviour are merged and elements with
//! similar behaviour are placed adjacently. This reduces the number of ranges
//! in the underlying `RangeMapBlaze` / `RangeSetBlaze` structures.

use std::collections::HashMap;
use std::sync::Arc;

use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};

use crate::automata::weighted_u32::dwa::DWA;
use crate::ds::weight::Weight;

use super::equivalence_analysis::{InternalIdMap, ManyToOneIdMap};

// ── public entry point ──────────────────────────────────────────────────────

pub struct ReorderReport {
    pub old_num_tsids: u32,
    pub new_num_tsids: u32,
    pub old_num_tokens: u32,
    pub new_num_tokens: u32,
    pub old_ranges: usize,
    pub new_ranges: usize,
}

/// Reorder and merge along both the tsid and token dimensions of every weight
/// in `dwa`, updating `id_map` to match.
pub fn reorder_dwa_dimensions(
    dwa: &mut DWA,
    id_map: &mut InternalIdMap,
) -> ReorderReport {
    let num_tsids = id_map.num_tsids();
    let num_tokens = id_map.num_internal_tokens();

    let old_ranges = count_total_ranges(dwa);

    // Step 1  — collect unique weights (by Arc pointer)
    let unique_weights = collect_unique_weights(dwa);
    let weight_refs: Vec<&Weight> = unique_weights.iter().collect();

    // Step 2  — tsid (outer) dimension: build profiles, sort by profile
    //           (no merging — just reorder so similar tsids are adjacent)
    let tsid_profiles = build_tsid_profiles(&weight_refs, num_tsids as usize);
    let tsid_perm = profile_sort_perm(&tsid_profiles);

    // Step 3  — token (inner) dimension: build profiles, sort by profile
    let token_profiles = build_token_profiles(&weight_refs, num_tokens);
    let token_perm = profile_sort_perm(&token_profiles);

    // Step 4  — apply permutations to every weight in the DWA
    apply_permutations_to_dwa(dwa, &unique_weights, &tsid_perm, &token_perm);

    // Step 5  — update id_map so the runtime still maps correctly
    apply_perm_to_id_map(&mut id_map.tokenizer_states, &tsid_perm);
    apply_perm_to_id_map(&mut id_map.vocab_tokens, &token_perm);

    let new_ranges = count_total_ranges(dwa);

    ReorderReport {
        old_num_tsids: num_tsids,
        new_num_tsids: num_tsids,
        old_num_tokens: num_tokens,
        new_num_tokens: num_tokens,
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

/// Deduplicate profiles: returns (class_map, unique_profiles) where
/// class_map[old_id] = class_index and unique_profiles[class_index] is the
/// canonical profile for that class.
#[allow(dead_code)]
fn dedup_profiles(profiles: &[Vec<u32>]) -> (Vec<usize>, Vec<Vec<u32>>) {
    let mut class_map_hash: HashMap<&[u32], usize> = HashMap::new();
    let mut mapping = vec![0usize; profiles.len()];
    let mut unique = Vec::new();
    for (i, profile) in profiles.iter().enumerate() {
        let class = if let Some(&c) = class_map_hash.get(profile.as_slice()) {
            c
        } else {
            let c = unique.len();
            unique.push(profile.clone());
            class_map_hash.insert(profile.as_slice(), c);
            c
        };
        mapping[i] = class;
    }
    (mapping, unique)
}

/// Sort elements by their profile (lexicographic) and return a bijective
/// permutation `perm[old_id] = new_id`. Elements with identical profiles
/// end up adjacent but are NOT merged — every old ID gets a unique new ID.
fn profile_sort_perm(profiles: &[Vec<u32>]) -> Vec<u32> {
    let n = profiles.len();
    if n == 0 {
        return vec![];
    }
    // sorted_order[i] = old_id that should come at position i
    let mut sorted_order: Vec<usize> = (0..n).collect();
    sorted_order.sort_by(|&a, &b| profiles[a].cmp(&profiles[b]));
    // Invert: perm[old_id] = new_position
    let mut perm = vec![0u32; n];
    for (new_pos, &old_id) in sorted_order.iter().enumerate() {
        perm[old_id] = new_pos as u32;
    }
    perm
}

/// Apply tsid and token permutations to every weight in the DWA.
fn apply_permutations_to_dwa(
    dwa: &mut DWA,
    unique_weights: &[Weight],
    tsid_perm: &[u32],
    token_perm: &[u32],
) {
    // Build old Arc ptr → new Weight map
    let mut weight_map: HashMap<usize, Weight> = HashMap::with_capacity(unique_weights.len());
    for w in unique_weights {
        let old_ptr = Arc::as_ptr(&w.0) as usize;
        let new_w = permute_weight(w, tsid_perm, token_perm);
        weight_map.insert(old_ptr, new_w);
    }

    // Replace all weights in the DWA
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

/// Create a new Weight from `w` with permuted tsid and token coordinates.
fn permute_weight(w: &Weight, tsid_perm: &[u32], token_perm: &[u32]) -> Weight {
    if w.is_empty() {
        return Weight::empty();
    }
    if w.is_full() {
        return Weight::all();
    }

    // Cache: old token_set Arc ptr → new permuted token_set Arc
    let mut ts_cache: HashMap<usize, Arc<RangeSetBlaze<u32>>> = HashMap::new();

    // Collect (new_tsid, new_token_set Arc) pairs
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

    // Sort by new tsid to build the RangeMapBlaze
    pairs.sort_unstable_by_key(|(t, _)| *t);

    if pairs.is_empty() {
        return Weight::empty();
    }

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

/// Map each element in `set` through the permutation.
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

    // Build compact ranges from sorted values
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

/// Update a `ManyToOneIdMap` after a permutation of internal IDs.
fn apply_perm_to_id_map(id_map: &mut ManyToOneIdMap, perm: &[u32]) {
    // Update original_to_internal: each old internal → new internal
    for internal in &mut id_map.original_to_internal {
        if *internal != u32::MAX {
            if let Some(&new_id) = perm.get(*internal as usize) {
                *internal = new_id;
            }
        }
    }

    // Rebuild internal_to_originals from the updated mapping
    let new_count = perm.iter().copied().max().map_or(0, |m| m + 1) as usize;
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
        n += 1; // outer range
        n += token_set.ranges().count(); // inner ranges
    }
    n
}
