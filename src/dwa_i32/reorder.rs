//! Post-DWA dimension reordering to minimize total range counts in weights.
//!
//! After building a DWA with RangeMapWeight weights, the total number of
//! ranges in the weight pool depends on the ordering of token IDs and tsid
//! IDs. Tokens/tsids that co-occur in the same sets but have distant IDs
//! create many small ranges. Reordering so that co-occurring IDs are adjacent
//! can dramatically reduce the number of ranges (94%+ reduction observed).
//!
//! This module implements:
//! - Greedy nearest-neighbor reordering for both token and tsid dimensions
//! - Application of permutations to all weights in a DWA

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::datastructures::{
    hybrid_bitset::RangeSet as HybridRangeSet,
    rangemap_weight::{intern_rangemap, RangeMapWeight},
    AbstractWeight,
};
use crate::dwa_i32::dwa::DWA;
use range_set_blaze::RangeSetBlaze;

/// Compute greedy nearest-neighbor permutation for a dimension.
///
/// Given a list of profiles (one per element), where each profile is a set
/// of context indices, compute an ordering that places elements with similar
/// profiles adjacent to each other.
///
/// Returns a permutation array where `perm[old_id] = new_id`.
fn greedy_nearest_neighbor(profiles: &[Vec<u32>]) -> Vec<usize> {
    let n = profiles.len();
    if n == 0 {
        return vec![];
    }

    // Convert profiles to sorted vecs for fast intersection
    // (they should already be sorted, but ensure it)
    let sorted_profiles: Vec<&[u32]> = profiles.iter().map(|p| p.as_slice()).collect();

    let mut visited = vec![false; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);

    // Start with the element that has the largest profile
    let start = (0..n)
        .max_by_key(|&i| sorted_profiles[i].len())
        .unwrap_or(0);
    visited[start] = true;
    order.push(start);

    for _ in 1..n {
        let current = *order.last().unwrap();
        let current_prof = sorted_profiles[current];

        let mut best_sim: usize = 0;
        let mut best_next: Option<usize> = None;
        let mut fallback: Option<usize> = None;

        for j in 0..n {
            if visited[j] {
                continue;
            }
            if fallback.is_none() {
                fallback = Some(j);
            }
            if current_prof.is_empty() || sorted_profiles[j].is_empty() {
                continue;
            }
            // Compute intersection size of two sorted slices
            let inter = sorted_intersection_count(current_prof, sorted_profiles[j]);
            if inter > best_sim || best_next.is_none() {
                best_sim = inter;
                best_next = Some(j);
            }
        }

        let next = best_next.or(fallback).unwrap();
        visited[next] = true;
        order.push(next);
    }

    // Build permutation: perm[old_id] = new_id
    let mut perm = vec![0usize; n];
    for (new_pos, &old_id) in order.iter().enumerate() {
        perm[old_id] = new_pos;
    }
    perm
}

/// Count the number of common elements in two sorted slices.
#[inline]
fn sorted_intersection_count(a: &[u32], b: &[u32]) -> usize {
    let mut count = 0;
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

/// Build tsid profiles from the weight pool of a DWA.
///
/// For each tsid, returns a sorted list of context indices where that tsid
/// appears. A "context" is a (weight_pool_index, token_range_index) pair,
/// encoded as a single u32.
fn build_tsid_profiles(unique_weights: &[&RangeMapWeight], num_tsids: usize) -> Vec<Vec<u32>> {
    let mut profiles: Vec<Vec<u32>> = vec![Vec::new(); num_tsids];
    let mut context_idx: u32 = 0;

    for &weight in unique_weights {
        for (_token_range, tsid_set) in weight.map.range_values() {
            for tsid_range in tsid_set.ranges() {
                for tsid in *tsid_range.start()..=*tsid_range.end() {
                    if tsid < num_tsids {
                        profiles[tsid].push(context_idx);
                    }
                }
            }
            context_idx += 1;
        }
    }

    profiles
}

/// Build token profiles from the weight pool of a DWA.
///
/// For each token, returns a sorted list of context indices where that token
/// appears. A "context" is a (weight_pool_index, token_range_index) pair,
/// but since tokens share ranges, we just use a unique entry_idx per
/// (weight, range_entry) pair.
fn build_token_profiles(
    unique_weights: &[&RangeMapWeight],
    max_token: usize,
) -> Vec<Vec<u32>> {
    let mut profiles: Vec<Vec<u32>> = vec![Vec::new(); max_token + 1];
    let mut context_idx: u32 = 0;

    for &weight in unique_weights {
        for (token_range, _tsid_set) in weight.map.range_values() {
            for tok in *token_range.start()..=*token_range.end() {
                if tok <= max_token {
                    profiles[tok].push(context_idx);
                }
            }
            context_idx += 1;
        }
    }

    profiles
}

/// Apply a tsid permutation to a HybridRangeSet.
///
/// Maps each value through the permutation and returns a new HybridRangeSet.
fn permute_rangeset(set: &HybridRangeSet, perm: &[usize]) -> HybridRangeSet {
    if set.is_empty() {
        return HybridRangeSet::default();
    }
    let mut mapped: Vec<usize> = set
        .ranges()
        .flat_map(|r| *r.start()..=*r.end())
        .map(|v| perm[v])
        .collect();
    mapped.sort_unstable();
    // Build RangeSetBlaze from sorted values
    let rsb = if mapped.is_empty() {
        RangeSetBlaze::new()
    } else {
        // Merge adjacent values into ranges
        let mut ranges: Vec<std::ops::RangeInclusive<usize>> = Vec::new();
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
    };
    HybridRangeSet::from(rsb)
}

/// Apply both token and tsid permutations to a RangeMapWeight.
///
/// Returns a new RangeMapWeight with permuted dimensions.
fn permute_weight(
    weight: &RangeMapWeight,
    token_perm: &[usize],
    tsid_perm: &[usize],
) -> RangeMapWeight {
    // Collect all (old_token, tsid_set) pairs, mapping tokens through permutation
    let mut token_to_tsid: BTreeMap<usize, HybridRangeSet> = BTreeMap::new();

    for (token_range, tsid_set) in weight.map.range_values() {
        let new_tsid_set = permute_rangeset(tsid_set, tsid_perm);
        for old_token in *token_range.start()..=*token_range.end() {
            if old_token < token_perm.len() {
                let new_token = token_perm[old_token];
                token_to_tsid.insert(new_token, new_tsid_set.clone());
            }
        }
    }

    // Build new RangeMapBlaze by merging adjacent tokens with same tsid_set
    let mut new_map = range_set_blaze::RangeMapBlaze::new();

    let entries: Vec<(usize, HybridRangeSet)> = token_to_tsid.into_iter().collect();
    if !entries.is_empty() {
        let mut run_start = entries[0].0;
        let mut run_end = entries[0].0;
        let mut run_set = entries[0].1.clone();

        for &(token, ref set) in &entries[1..] {
            if token == run_end + 1 && *set == run_set {
                run_end = token;
            } else {
                new_map.ranges_insert(run_start..=run_end, run_set.clone());
                run_start = token;
                run_end = token;
                run_set = set.clone();
            }
        }
        new_map.ranges_insert(run_start..=run_end, run_set);
    }

    RangeMapWeight::from_map(new_map, weight.num_tsids())
}

/// Collect unique RangeMapWeight references from a DWA's weight pool.
fn collect_unique_weights(dwa: &DWA) -> Vec<Arc<RangeMapWeight>> {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    let mut unique = Vec::new();

    for state in &dwa.states.0 {
        for (_, w) in &state.trans_weights {
            if let AbstractWeight::RangeMap(rm) = w {
                let ptr = Arc::as_ptr(rm) as usize;
                if seen.insert(ptr) {
                    unique.push(Arc::clone(rm));
                }
            }
        }
        if let Some(AbstractWeight::RangeMap(rm)) = &state.final_weight {
            let ptr = Arc::as_ptr(rm) as usize;
            if seen.insert(ptr) {
                unique.push(Arc::clone(rm));
            }
        }
    }
    unique
}

/// Compute optimal token and tsid permutations for a DWA's weights,
/// then apply those permutations to all weights in the DWA.
///
/// Returns (token_perm, tsid_perm) where perm[old_id] = new_id.
pub fn reorder_dwa_dimensions(
    dwa: &mut DWA,
    max_token: usize,
    num_tsids: usize,
) -> (Vec<usize>, Vec<usize>) {
    let start = std::time::Instant::now();

    // Step 1: Collect unique weights
    let unique_weights_arc = collect_unique_weights(dwa);
    let unique_weights: Vec<&RangeMapWeight> =
        unique_weights_arc.iter().map(|arc| arc.as_ref()).collect();

    let num_unique = unique_weights.len();
    crate::debug!(
        3,
        "reorder_dwa_dimensions: {} unique weights, max_token={}, num_tsids={}",
        num_unique,
        max_token,
        num_tsids,
    );

    // Count baseline ranges (properly deduped: outer by weight Arc, inner by RangeSet Arc)
    let baseline_ranges = dwa.num_ranges_interned();

    // Step 2: Compute tsid permutation
    let tsid_profiles = build_tsid_profiles(&unique_weights, num_tsids);
    let tsid_perm = greedy_nearest_neighbor(&tsid_profiles);

    // Step 3: Compute token permutation
    let token_profiles = build_token_profiles(&unique_weights, max_token);
    let token_perm = greedy_nearest_neighbor(&token_profiles);

    crate::debug!(
        3,
        "reorder_dwa_dimensions: computed permutations in {:?}",
        start.elapsed()
    );

    // Step 4: Build mapping from old weight Arc ptr -> new weight
    let apply_start = std::time::Instant::now();
    let mut weight_map: std::collections::HashMap<usize, Arc<RangeMapWeight>> =
        std::collections::HashMap::new();
    for arc in &unique_weights_arc {
        let old_ptr = Arc::as_ptr(arc) as usize;
        let new_weight = permute_weight(arc.as_ref(), &token_perm, &tsid_perm);
        weight_map.insert(old_ptr, intern_rangemap(new_weight));
    }

    // Step 5: Apply to all weights in the DWA
    for state in &mut dwa.states.0 {
        for (_, w) in &mut state.trans_weights {
            if let AbstractWeight::RangeMap(rm) = w {
                let old_ptr = Arc::as_ptr(rm) as usize;
                if let Some(new_rm) = weight_map.get(&old_ptr) {
                    *rm = Arc::clone(new_rm);
                }
            }
        }
        if let Some(AbstractWeight::RangeMap(rm)) = &mut state.final_weight {
            let old_ptr = Arc::as_ptr(rm) as usize;
            if let Some(new_rm) = weight_map.get(&old_ptr) {
                *rm = Arc::clone(new_rm);
            }
        }
    }

    // Count new ranges (properly deduped: outer by weight Arc, inner by RangeSet Arc)
    let new_ranges = dwa.num_ranges_interned();

    crate::debug!(
        3,
        "REORDER_DWA: baseline_ranges={} -> new_ranges={} ({:.1}% reduction) in {:?}",
        baseline_ranges,
        new_ranges,
        if baseline_ranges > 0 {
            (1.0 - new_ranges as f64 / baseline_ranges as f64) * 100.0
        } else {
            0.0
        },
        start.elapsed()
    );

    crate::debug!(
        3,
        "reorder_dwa_dimensions: applied permutations in {:?} (total {:?})",
        apply_start.elapsed(),
        start.elapsed()
    );

    (token_perm, tsid_perm)
}

// ---------------------------------------------------------------------------
// Pre-DWA ordering prediction from possible_matches
// ---------------------------------------------------------------------------

use crate::constraint_vocab::LLMTokenBV;
use crate::dfa_u8::TokenizerStateID;
use crate::types::TerminalID;

/// Predict good token and tsid orderings BEFORE the terminal DWA is computed,
/// using the possible_matches data structure.
///
/// `possible_matches[state][terminal]` = set of internal token IDs that can
/// match that terminal from that state. This is essentially the same
/// information that ends up encoded in the DWA weights, just transposed.
///
/// For **tsid ordering**: two states should be adjacent if similar sets of
/// tokens match them across terminals.
///
/// For **token ordering**: two tokens should be adjacent if they appear in
/// similar state×terminal contexts.
///
/// Returns `(token_perm, tsid_perm)` where `perm[old_id] = new_id`.
/// `token_perm` has length `max_token + 1`, `tsid_perm` has length `num_states`.
pub fn predict_orderings_from_possible_matches(
    possible_matches: &BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    max_internal_token: usize,
    num_states: usize,
) -> (Vec<usize>, Vec<usize>) {
    let start = std::time::Instant::now();
    let num_tokens = max_internal_token + 1;

    // Build tsid profiles: for each state, the sorted list of
    // (context_id) where context_id encodes (terminal, token) pairs.
    // To keep profiles compact, we use a simpler encoding:
    // for each state s, collect ALL internal token IDs that appear in
    // any possible_matches[s][terminal].
    let mut tsid_profiles: Vec<Vec<u32>> = vec![Vec::new(); num_states];
    // Build token profiles: for each token, the sorted list of state IDs
    // where it appears in any possible_matches entry.
    let mut token_profiles: Vec<Vec<u32>> = vec![Vec::new(); num_tokens];

    for (&state_id, terminal_map) in possible_matches {
        let sid = state_id.0;
        if sid >= num_states {
            continue;
        }
        for (_terminal_id, token_bv) in terminal_map {
            for range in token_bv.ranges() {
                let lo = *range.start();
                let hi = *range.end();
                for tok in lo..=hi.min(max_internal_token) {
                    token_profiles[tok].push(sid as u32);
                }
            }
            // For tsid profile, collect token IDs
            for range in token_bv.ranges() {
                let lo = *range.start();
                let hi = *range.end();
                for tok in lo..=hi.min(max_internal_token) {
                    tsid_profiles[sid].push(tok as u32);
                }
            }
        }
    }

    // Deduplicate and sort profiles
    for p in &mut tsid_profiles {
        p.sort_unstable();
        p.dedup();
    }
    for p in &mut token_profiles {
        p.sort_unstable();
        p.dedup();
    }

    let profile_time = start.elapsed();

    // Run greedy NN on tsid profiles
    let tsid_perm = greedy_nearest_neighbor(&tsid_profiles);
    let tsid_time = start.elapsed();

    // Run greedy NN on token profiles
    let token_perm = greedy_nearest_neighbor(&token_profiles);
    let total_time = start.elapsed();

    crate::debug!(
        3,
        "predict_orderings: profiles={:?}, tsid_nn={:?}, token_nn={:?}, total={:?} (tokens={}, states={})",
        profile_time,
        tsid_time - profile_time,
        total_time - tsid_time,
        total_time,
        num_tokens,
        num_states,
    );

    (token_perm, tsid_perm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_nearest_neighbor_empty() {
        let profiles: Vec<Vec<u32>> = vec![];
        let perm = greedy_nearest_neighbor(&profiles);
        assert!(perm.is_empty());
    }

    #[test]
    fn test_greedy_nearest_neighbor_single() {
        let profiles: Vec<Vec<u32>> = vec![vec![0, 1, 2]];
        let perm = greedy_nearest_neighbor(&profiles);
        assert_eq!(perm, vec![0]);
    }

    #[test]
    fn test_greedy_nearest_neighbor_groups_similar() {
        // Elements 0,1 share context {0,1}, elements 2,3 share context {2,3}
        let profiles: Vec<Vec<u32>> = vec![
            vec![0, 1],
            vec![0, 1],
            vec![2, 3],
            vec![2, 3],
        ];
        let perm = greedy_nearest_neighbor(&profiles);
        // Similar elements should get adjacent positions
        // Verify: elements 0,1 are adjacent and elements 2,3 are adjacent
        assert!((perm[0] as isize - perm[1] as isize).abs() == 1);
        assert!((perm[2] as isize - perm[3] as isize).abs() == 1);
    }

    #[test]
    fn test_permute_rangeset() {
        // {0, 2, 4} with perm [3, 0, 1, 2, 4] -> {3, 1, 4} -> as range set {1, 3..=4}
        let rs = HybridRangeSet::from(RangeSetBlaze::from_iter([0..=0, 2..=2, 4..=4]));
        let perm = vec![3, 0, 1, 2, 4];
        let result = permute_rangeset(&rs, &perm);
        // {3, 1, 4} -> sorted {1, 3, 4} -> ranges {1..=1, 3..=4}
        assert_eq!(result.ranges_len(), 2);
    }

    #[test]
    fn test_sorted_intersection_count() {
        assert_eq!(sorted_intersection_count(&[1, 3, 5], &[2, 3, 5, 7]), 2);
        assert_eq!(sorted_intersection_count(&[], &[1, 2, 3]), 0);
        assert_eq!(sorted_intersection_count(&[1, 2, 3], &[1, 2, 3]), 3);
    }
}
