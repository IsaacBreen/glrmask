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
use hashbrown::HashMap;

/// Deduplicate profiles: group identical profiles into equivalence classes.
///
/// Returns:
/// - `mapping[old_id] = class_idx` — maps each element to its equivalence class
/// - `unique_profiles` — one representative profile per class (class_idx → profile)
fn dedup_profiles(profiles: &[Vec<u32>]) -> (Vec<usize>, Vec<Vec<u32>>) {
    let mut class_map: HashMap<&[u32], usize> = HashMap::new();
    let mut mapping = vec![0usize; profiles.len()];
    let mut unique_profiles: Vec<Vec<u32>> = Vec::new();
    for (i, profile) in profiles.iter().enumerate() {
        let class = if let Some(&c) = class_map.get(profile.as_slice()) {
            c
        } else {
            let c = unique_profiles.len();
            unique_profiles.push(profile.clone());
            class_map.insert(profile.as_slice(), c);
            c
        };
        mapping[i] = class;
    }
    (mapping, unique_profiles)
}

/// Compute greedy nearest-neighbor permutation for a dimension.
///
/// Given a list of profiles (one per element), where each profile is a set
/// of context indices, compute an ordering that places elements with similar
/// profiles adjacent to each other.
///
/// Only elements with non-empty profiles participate in the ordering.
/// Empty-profile elements are placed at the end in their original order.
///
/// For large inputs (>4000 active elements), uses a block-based approach:
/// sort by first context word, then run NN within local neighborhoods.
///
/// Returns a permutation array where `perm[old_id] = new_id`.
fn greedy_nearest_neighbor(profiles: &[Vec<u32>]) -> Vec<usize> {
    let n = profiles.len();
    if n == 0 {
        return vec![];
    }

    // Separate active (non-empty profile) from inactive (empty profile) elements
    let active_indices: Vec<usize> = (0..n).filter(|&i| !profiles[i].is_empty()).collect();
    let inactive_indices: Vec<usize> = (0..n).filter(|&i| profiles[i].is_empty()).collect();

    let active_count = active_indices.len();

    if active_count == 0 {
        // All profiles empty — identity permutation
        return (0..n).collect();
    }

    // Determine max context index for bitvector sizing
    let max_ctx = profiles.iter()
        .flat_map(|p| p.iter())
        .copied()
        .max()
        .unwrap_or(0) as usize;
    let num_words = (max_ctx + 64) / 64; // number of u64 words

    // Build bitvector representations for active profiles
    let active_bvs: Vec<Vec<u64>> = active_indices.iter().map(|&i| {
        let mut bv = vec![0u64; num_words];
        for &ctx in &profiles[i] {
            let idx = ctx as usize;
            bv[idx / 64] |= 1u64 << (idx % 64);
        }
        bv
    }).collect();

    // Choose strategy based on scale
    // Use total work estimate: O(n^2 * num_words) for full NN.
    // Budget ~10B ops (about 2s on modern hardware).
    let work_estimate = (active_count as u64) * (active_count as u64) * (num_words as u64);
    let order = if work_estimate > 10_000_000_000 {
        // Large scale: use lexicographic sort on bitvectors (locality-preserving)
        // then refine with local NN within a window
        greedy_nn_large_scale(&active_bvs, active_count, num_words)
    } else {
        // Small scale: full NN with bitvector popcount
        greedy_nn_small_scale(&active_bvs, active_count)
    };

    // Build permutation: active elements placed first (in NN order),
    // inactive elements placed after in original order.
    let mut perm = vec![0usize; n];
    for (new_pos, &active_local_idx) in order.iter().enumerate() {
        let original_idx = active_indices[active_local_idx];
        perm[original_idx] = new_pos;
    }
    let offset = active_count;
    for (i, &original_idx) in inactive_indices.iter().enumerate() {
        perm[original_idx] = offset + i;
    }
    perm
}

/// Full NN for small-scale problems using bitvector intersection.
fn greedy_nn_small_scale(active_bvs: &[Vec<u64>], active_count: usize) -> Vec<usize> {
    // Precompute popcount for each active BV
    let active_popcnts: Vec<u32> = active_bvs.iter()
        .map(|bv| bv.iter().map(|w| w.count_ones()).sum())
        .collect();

    let mut visited = vec![false; active_count];
    let mut order: Vec<usize> = Vec::with_capacity(active_count);

    let start = (0..active_count)
        .max_by_key(|&i| active_popcnts[i])
        .unwrap_or(0);
    visited[start] = true;
    order.push(start);

    for _ in 1..active_count {
        let current = *order.last().unwrap();
        let current_bv = &active_bvs[current];

        let mut best_sim: u32 = 0;
        let mut best_next: Option<usize> = None;
        let mut fallback: Option<usize> = None;

        for j in 0..active_count {
            if visited[j] {
                continue;
            }
            if fallback.is_none() {
                fallback = Some(j);
            }
            let inter: u32 = current_bv.iter()
                .zip(active_bvs[j].iter())
                .map(|(&a, &b)| (a & b).count_ones())
                .sum();
            if inter > best_sim || best_next.is_none() {
                best_sim = inter;
                best_next = Some(j);
            }
        }

        let next = best_next.or(fallback).unwrap();
        visited[next] = true;
        order.push(next);
    }

    order
}

/// Large-scale ordering using recursive bisection + 2-opt refinement.
///
/// Uses a decision-tree style recursive bisection: at each level, find the 
/// bitvector bit (context) that most evenly splits the elements, place elements
/// WITH that bit in the left half and WITHOUT in the right half, then recurse.
/// This produces orderings where elements sharing contexts are clustered together,
/// directly optimizing for range contiguity.
fn greedy_nn_large_scale(active_bvs: &[Vec<u64>], active_count: usize, _num_words: usize) -> Vec<usize> {
    let profile = std::env::var("PROFILE_BUILD_TOKENIZER").is_ok();

    // --- Strategy 1: Recursive bisection ---
    let bisection_order = recursive_bisection_order(active_bvs, active_count);
    let bisection_cost = total_adjacent_hamming(&bisection_order, active_bvs);

    // --- Strategy 2: Lexicographic sort (for comparison) ---
    let mut lex_order: Vec<usize> = (0..active_count).collect();
    lex_order.sort_unstable_by(|&a, &b| active_bvs[a].cmp(&active_bvs[b]));
    let lex_cost = total_adjacent_hamming(&lex_order, active_bvs);

    // --- Strategy 3: Popcount sort ---
    let popcnts: Vec<u32> = active_bvs.iter()
        .map(|bv| bv.iter().map(|w| w.count_ones()).sum())
        .collect();
    let mut pop_order: Vec<usize> = (0..active_count).collect();
    pop_order.sort_unstable_by_key(|&i| popcnts[i]);
    let pop_cost = total_adjacent_hamming(&pop_order, active_bvs);

    if profile {
        eprintln!("  reorder: bisection cost={}, lex cost={}, popcount cost={}", bisection_cost, lex_cost, pop_cost);
    }

    // Pick best initial ordering
    let mut order = if bisection_cost <= lex_cost && bisection_cost <= pop_cost {
        if profile { eprintln!("  reorder: using bisection ordering"); }
        bisection_order
    } else if lex_cost <= pop_cost {
        if profile { eprintln!("  reorder: using lex ordering"); }
        lex_order
    } else {
        if profile { eprintln!("  reorder: using popcount ordering"); }
        pop_order
    };

    if profile {
        let cost = total_adjacent_hamming(&order, active_bvs);
        eprintln!("  reorder: initial cost={}", cost);
    }

    // --- 2-opt local improvement ---
    let max_2opt_iters = 10;
    let max_segment = 500usize.min(order.len() / 2);
    // Use step>1 for very large problems to keep time bounded
    let step = if active_count > 10000 { 2 } else { 1 };
    let time_limit = std::time::Duration::from_secs(5);
    let opt_start = std::time::Instant::now();

    for iter in 0..max_2opt_iters {
        let iter_start = std::time::Instant::now();
        let mut improved = false;
        let mut i = 1;
        while i < order.len().saturating_sub(2) {
            let max_j = (i + max_segment).min(order.len() - 1);
            let mut best_delta: i64 = 0;
            let mut best_j = 0;

            let mut j = i + 1;
            while j <= max_j {
                let old_cost = bv_hamming(&active_bvs[order[i - 1]], &active_bvs[order[i]])
                    + if j + 1 < order.len() {
                        bv_hamming(&active_bvs[order[j]], &active_bvs[order[j + 1]])
                    } else { 0 };
                let new_cost = bv_hamming(&active_bvs[order[i - 1]], &active_bvs[order[j]])
                    + if j + 1 < order.len() {
                        bv_hamming(&active_bvs[order[i]], &active_bvs[order[j + 1]])
                    } else { 0 };
                let delta = old_cost as i64 - new_cost as i64;

                if delta > best_delta {
                    best_delta = delta;
                    best_j = j;
                }

                j += step;
            }

            if best_delta > 0 {
                order[i..=best_j].reverse();
                improved = true;
            }

            i += step;
        }

        if profile {
            let cost = total_adjacent_hamming(&order, active_bvs);
            eprintln!("  reorder: 2-opt iter {} improved={} cost={} ({:?})", iter, improved, cost, iter_start.elapsed());
        }

        if !improved || opt_start.elapsed() > time_limit {
            break;
        }
    }

    order
}

/// Recursive bisection ordering: at each level, find the BV bit that most
/// evenly splits elements, then recursively order each half.
fn recursive_bisection_order(bvs: &[Vec<u64>], n: usize) -> Vec<usize> {
    let indices: Vec<usize> = (0..n).collect();
    let num_words = bvs.first().map_or(0, |bv| bv.len());
    let num_bits = num_words * 64;
    
    // Precompute per-bit counts: how many elements have each bit set
    let mut bit_counts = vec![0u32; num_bits];
    for &idx in &indices {
        for (w, &word) in bvs[idx].iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit_pos = bits.trailing_zeros() as usize;
                bit_counts[w * 64 + bit_pos] += 1;
                bits &= bits - 1;
            }
        }
    }
    
    // Sort bits by how close they are to a 50/50 split
    let half = n as u32 / 2;
    let mut bit_order: Vec<usize> = (0..num_bits)
        .filter(|&b| bit_counts[b] > 0 && bit_counts[b] < n as u32) // skip all-0 and all-1 bits
        .collect();
    bit_order.sort_unstable_by_key(|&b| {
        let count = bit_counts[b];
        if count >= half { count - half } else { half - count }
    });
    
    // Limit recursion depth to ~log2(n) using the most balanced bits
    let max_depth = (n as f64).log2().ceil() as usize + 2;
    let useful_bits: Vec<usize> = bit_order.into_iter().take(max_depth * 2).collect();
    
    let mut order = Vec::with_capacity(n);
    bisect_recursive(bvs, &indices, &useful_bits, 0, &mut order);
    order
}

fn bisect_recursive(
    bvs: &[Vec<u64>],
    indices: &[usize],
    bits: &[usize],
    depth: usize,
    order: &mut Vec<usize>,
) {
    if indices.len() <= 2 || depth >= bits.len() {
        // Base case: add in current order
        order.extend_from_slice(indices);
        return;
    }
    
    let bit = bits[depth];
    let word_idx = bit / 64;
    let bit_mask = 1u64 << (bit % 64);
    
    let mut left = Vec::new();  // has bit set
    let mut right = Vec::new(); // doesn't have bit set
    
    for &idx in indices {
        if bvs[idx].get(word_idx).copied().unwrap_or(0) & bit_mask != 0 {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }
    
    // If split is too unbalanced (>90/10), skip this bit
    let threshold = indices.len() / 10;
    if left.len() < threshold || right.len() < threshold {
        bisect_recursive(bvs, indices, bits, depth + 1, order);
        return;
    }
    
    bisect_recursive(bvs, &left, bits, depth + 1, order);
    bisect_recursive(bvs, &right, bits, depth + 1, order);
}

/// Count interned ranges broken down into outer (token ranges) and inner (tsid sub-ranges).
fn count_ranges_breakdown(dwa: &DWA) -> (usize, usize) {
    use crate::datastructures::AbstractWeight;
    use std::collections::HashSet;

    let mut seen_weight_ptrs: HashSet<usize> = HashSet::new();
    let mut seen_rangeset_ptrs: HashSet<usize> = HashSet::new();
    let mut outer = 0usize;
    let mut inner = 0usize;

    let mut process = |w: &crate::dwa_i32::dwa::Weight| {
        if let AbstractWeight::RangeMap(rm) = w {
            let weight_ptr = Arc::as_ptr(rm) as usize;
            if seen_weight_ptrs.insert(weight_ptr) {
                outer += rm.map.range_values().count();
            }
            for (_, tsid_set) in rm.map.range_values() {
                let ptr = Arc::as_ptr(&tsid_set.inner) as usize;
                if seen_rangeset_ptrs.insert(ptr) {
                    inner += tsid_set.ranges_len();
                }
            }
        }
    };

    for state in &dwa.states.0 {
        if let Some(fw) = &state.final_weight {
            process(fw);
        }
        for w in state.trans_weights.values() {
            process(w);
        }
    }

    (outer, inner)
}

/// Compute total adjacent Hamming distance for an ordering.
fn total_adjacent_hamming(order: &[usize], bvs: &[Vec<u64>]) -> u64 {
    let mut total: u64 = 0;
    for i in 0..order.len().saturating_sub(1) {
        total += bv_hamming(&bvs[order[i]], &bvs[order[i + 1]]) as u64;
    }
    total
}

/// Build column-fingerprint sort order for tokens.
///
/// For each unique weight, each token maps to a specific column (tsid_set).
/// The fingerprint for a token is its column ID in each weight, ordered by
/// weight importance (fewest distinct columns first — these are the weights
/// where column grouping helps most).
///
/// Sorting by this fingerprint groups tokens that share columns in the most
/// impactful weights together, directly minimizing outer range count.
fn column_fingerprint_order(
    unique_weights: &[&RangeMapWeight],
    num_elements: usize,
    _dim: &str,
) -> Vec<usize> {
    column_fingerprint_order_with_class_map(unique_weights, num_elements, None)
}

/// Column-fingerprint with optional class map for deduped elements.
fn column_fingerprint_order_with_class_map(
    unique_weights: &[&RangeMapWeight],
    num_classes: usize,
    class_map: Option<(&[usize], usize)>, // (class_map, max_old_id+1)
) -> Vec<usize> {
    let not_present: u32 = u32::MAX;
    
    // Compute number of distinct columns per weight
    let mut weight_info: Vec<(usize, usize)> = Vec::new();
    for (w_idx, &weight) in unique_weights.iter().enumerate() {
        let num_entries = weight.map.range_values().count();
        weight_info.push((w_idx, num_entries));
    }
    // Sort by number of distinct columns (ascending) — most constrained first
    weight_info.sort_unstable_by_key(|&(_, num)| num);
    
    // Build fingerprints using top-32 most constrained weights
    let max_fp_len = 32usize.min(weight_info.len());
    let selected_weights: Vec<usize> = weight_info.iter().take(max_fp_len).map(|&(idx, _)| idx).collect();
    
    let mut fingerprints: Vec<Vec<u32>> = vec![vec![not_present; max_fp_len]; num_classes];
    
    for (fp_pos, &w_idx) in selected_weights.iter().enumerate() {
        let weight = unique_weights[w_idx];
        let mut col_id: u32 = 0;
        for (token_range, _) in weight.map.range_values() {
            for old_elem in *token_range.start()..=*token_range.end() {
                let elem = if let Some((cmap, max_old)) = class_map {
                    if old_elem < max_old { cmap[old_elem] } else { continue }
                } else {
                    if old_elem < num_classes { old_elem } else { continue }
                };
                if elem < num_classes {
                    fingerprints[elem][fp_pos] = col_id;
                }
            }
            col_id += 1;
        }
    }
    
    // Sort elements by fingerprint (lexicographic)
    let mut order: Vec<usize> = (0..num_classes).collect();
    order.sort_unstable_by(|&a, &b| fingerprints[a].cmp(&fingerprints[b]));
    order
}

/// Count outer ranges (token range entries) for a proposed token ordering.
fn count_outer_ranges_for_ordering(
    unique_weights: &[&RangeMapWeight],
    class_perm: &[usize],
    max_token: usize,
    class_map: &[usize],
) -> usize {
    let perm: Vec<usize> = (0..max_token + 1)
        .map(|i| class_perm[class_map[i]])
        .collect();
    
    let mut total_entries = 0usize;
    
    for &weight in unique_weights {
        // Use tsid_set Arc pointer as column ID (not entry_idx) so that
        // adjacent tokens with the same tsid_set merge properly.
        let mut new_positions: Vec<(usize, usize)> = Vec::new();
        for (token_range, tsid_set) in weight.map.range_values() {
            let col_id = Arc::as_ptr(&tsid_set.inner) as usize;
            for old_token in *token_range.start()..=*token_range.end() {
                if old_token < perm.len() {
                    new_positions.push((perm[old_token], col_id));
                }
            }
        }
        new_positions.sort_unstable();
        
        if !new_positions.is_empty() {
            let mut runs = 1;
            for i in 1..new_positions.len() {
                let (pos, col) = new_positions[i];
                let (prev_pos, prev_col) = new_positions[i - 1];
                if col != prev_col || pos != prev_pos + 1 {
                    runs += 1;
                }
            }
            total_entries += runs;
        }
    }
    
    total_entries
}

/// Column-fingerprint ordering for the tsid dimension.
/// For each weight entry, the tsid_set is a set of tsids. We build fingerprints
/// based on which "row" (token_range entry) each tsid belongs to in each weight.
fn column_fingerprint_tsid_order(
    unique_weights: &[&RangeMapWeight],
    num_classes: usize,
    class_map: Option<(&[usize], usize)>,
) -> Vec<usize> {
    let not_present: u32 = u32::MAX;
    
    // Rank weights by number of distinct tsid_sets (fewer = more constrained)
    let mut weight_info: Vec<(usize, usize)> = Vec::new();
    for (w_idx, &weight) in unique_weights.iter().enumerate() {
        // Count distinct tsid_sets in this weight
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (_, tsid_set) in weight.map.range_values() {
            seen.insert(Arc::as_ptr(&tsid_set.inner) as usize);
        }
        weight_info.push((w_idx, seen.len()));
    }
    weight_info.sort_unstable_by_key(|&(_, num)| num);
    
    let max_fp_len = 32usize.min(weight_info.len());
    let selected_weights: Vec<usize> = weight_info.iter().take(max_fp_len).map(|&(idx, _)| idx).collect();
    
    let mut fingerprints: Vec<Vec<u32>> = vec![vec![not_present; max_fp_len]; num_classes];
    
    for (fp_pos, &w_idx) in selected_weights.iter().enumerate() {
        let weight = unique_weights[w_idx];
        // Assign each unique tsid_set a column ID
        let mut set_to_col: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
        let mut next_col: u32 = 0;
        
        for (_, tsid_set) in weight.map.range_values() {
            let ptr = Arc::as_ptr(&tsid_set.inner) as usize;
            let col_id = *set_to_col.entry(ptr).or_insert_with(|| {
                let c = next_col;
                next_col += 1;
                c
            });
            
            // Map each tsid in this set to the column ID
            for tsid_range in tsid_set.ranges() {
                for old_tsid in *tsid_range.start()..=*tsid_range.end() {
                    let tsid = if let Some((cmap, max_old)) = class_map {
                        if old_tsid < max_old { cmap[old_tsid] } else { continue }
                    } else {
                        if old_tsid < num_classes { old_tsid } else { continue }
                    };
                    if tsid < num_classes {
                        fingerprints[tsid][fp_pos] = col_id;
                    }
                }
            }
        }
    }
    
    let mut order: Vec<usize> = (0..num_classes).collect();
    order.sort_unstable_by(|&a, &b| fingerprints[a].cmp(&fingerprints[b]));
    order
}

/// Count inner ranges (tsid sub-ranges) for a proposed tsid ordering.
fn count_inner_ranges_for_ordering(
    unique_weights: &[&RangeMapWeight],
    class_perm: &[usize],
    num_tsids: usize,
    class_map: &[usize],
) -> usize {
    let perm: Vec<usize> = (0..num_tsids)
        .map(|i| class_perm[class_map[i]])
        .collect();
    
    let mut total_ranges = 0usize;
    let mut seen_sets: std::collections::HashSet<usize> = std::collections::HashSet::new();
    
    for &weight in unique_weights {
        for (_, tsid_set) in weight.map.range_values() {
            let ptr = Arc::as_ptr(&tsid_set.inner) as usize;
            if !seen_sets.insert(ptr) {
                continue; // Already counted this tsid_set
            }
            
            // Apply permutation and count ranges
            let mut mapped: Vec<usize> = tsid_set.ranges()
                .flat_map(|r| *r.start()..=*r.end())
                .filter(|&v| v < perm.len())
                .map(|v| perm[v])
                .collect();
            mapped.sort_unstable();
            mapped.dedup();
            
            if mapped.is_empty() {
                continue;
            }
            
            let mut ranges = 1;
            for i in 1..mapped.len() {
                if mapped[i] != mapped[i - 1] + 1 {
                    ranges += 1;
                }
            }
            total_ranges += ranges;
        }
    }
    
    total_ranges
}

/// Or-opt local search to directly minimize outer range count for the token dimension.
///
/// Given an initial class permutation (from NN), iteratively tries relocating each class
/// to every other position, accepting moves that reduce the outer range count.
///
/// Uses precomputed same-column-neighbor array to handle inactive weights in O(1) per
/// position. Only active weights (where the element has col≠0) are iterated per position.
///
/// Returns the improved class permutation.
fn or_opt_outer_ranges(
    unique_weights: &[&RangeMapWeight],
    initial_perm: &[usize],
    max_token: usize,
    class_map: &[usize],
    time_limit: std::time::Duration,
) -> Vec<usize> {
    let profile = std::env::var("PROFILE_BUILD_TOKENIZER").is_ok();
    let start = std::time::Instant::now();
    let num_classes = initial_perm.len();
    if num_classes <= 2 {
        return initial_perm.to_vec();
    }

    // Build the inverse permutation: order[new_pos] = class_id
    let mut order: Vec<usize> = vec![0; num_classes];
    for (cls, &new_pos) in initial_perm.iter().enumerate() {
        if new_pos < num_classes {
            order[new_pos] = cls;
        }
    }

    // Build per-class column assignment matrix.
    // col_assign[class][weight_idx] = column_id (0 = absent)
    let num_weights = unique_weights.len();
    let mut col_assign: Vec<Vec<u16>> = vec![vec![0u16; num_weights]; num_classes];

    for (w_idx, &weight) in unique_weights.iter().enumerate() {
        let mut ptr_to_col: HashMap<usize, u16> = HashMap::new();
        let mut next_col: u16 = 1; // 0 = absent

        for (token_range, tsid_set) in weight.map.range_values() {
            let ptr = Arc::as_ptr(&tsid_set.inner) as usize;
            let col = *ptr_to_col.entry(ptr).or_insert_with(|| {
                let c = next_col;
                next_col = next_col.saturating_add(1);
                c
            });

            for old_token in *token_range.start()..=*token_range.end() {
                if old_token <= max_token {
                    let cls = class_map[old_token];
                    if cls < num_classes {
                        col_assign[cls][w_idx] = col;
                    }
                }
            }
        }
    }

    // Count outer ranges (runs of same non-zero column) in current ordering
    let count_breaks = |order: &[usize]| -> usize {
        let mut breaks = 0usize;
        for w_idx in 0..num_weights {
            let mut in_run = false;
            let mut prev_col = 0u16;
            for &cls in order.iter() {
                let col = col_assign[cls][w_idx];
                if col == 0 {
                    in_run = false;
                } else if !in_run || col != prev_col {
                    breaks += 1;
                    in_run = true;
                }
                prev_col = col;
            }
        }
        breaks
    };

    // same_col_between(a, b) = count of weights where col_assign[a][w] == col_assign[b][w] && col_assign[a][w] != 0
    let same_col_between = |a: usize, b: usize| -> i32 {
        let mut count = 0i32;
        let ca = &col_assign[a];
        let cb = &col_assign[b];
        for w in 0..num_weights {
            if ca[w] != 0 && ca[w] == cb[w] {
                count += 1;
            }
        }
        count
    };

    // Precompute same_col_adj_all[i] = same_col_between(order[i], order[i+1])
    // This counts ALL weights where adjacent elements share the same non-zero column.
    let build_same_col_adj = |order: &[usize]| -> Vec<i32> {
        let n = order.len();
        let mut adj = vec![0i32; n.saturating_sub(1)];
        for i in 0..n.saturating_sub(1) {
            adj[i] = same_col_between(order[i], order[i + 1]);
        }
        adj
    };

    let mut same_col_adj = build_same_col_adj(&order);
    let mut best_breaks = count_breaks(&order);
    if profile {
        eprintln!("  or_opt: initial breaks={}", best_breaks);
    }

    // For large dimensions, limit inner-loop search to a window around the
    // source to avoid O(n²) scanning per iteration. Full scan on first iter.
    let search_radius = if num_classes > 2000 { 250 } else { num_classes };

    let max_iters = 20;
    for iter in 0..max_iters {
        if start.elapsed() > time_limit {
            break;
        }
        let mut improved = false;

        for src_pos in 0..order.len() {
            if start.elapsed() > time_limit {
                break;
            }

            let cls = order[src_pos];
            let mut best_pos = src_pos;
            let mut best_delta: i64 = 0;

            // Active weights for this class (col != 0)
            let active_weights: Vec<usize> = (0..num_weights)
                .filter(|&w| col_assign[cls][w] != 0)
                .collect();

            // --- REMOVAL SIDE ---
            // Compute remove_save_active (active weights only)
            let mut remove_save_active = 0i64;
            let mut join_same_col_active = 0i32; // active weights where removal neighbors share same non-zero col
            for &w in &active_weights {
                let col = col_assign[cls][w];
                let left_col = if src_pos > 0 { col_assign[order[src_pos - 1]][w] } else { 0 };
                let right_col = if src_pos + 1 < order.len() { col_assign[order[src_pos + 1]][w] } else { 0 };

                let cur_break_at_src = if col != 0 && (left_col == 0 || left_col != col) { 1 } else { 0 };
                let cur_break_at_right = if right_col != 0 && (col == 0 || col != right_col) { 1 } else { 0 };
                let new_break_at_gap = if right_col != 0 && (left_col == 0 || left_col != right_col) { 1 } else { 0 };

                remove_save_active += (cur_break_at_src + cur_break_at_right - new_break_at_gap) as i64;

                // Track active weight contribution to join pair same_col
                if src_pos > 0 && src_pos + 1 < order.len() && left_col != 0 && left_col == right_col {
                    join_same_col_active += 1;
                }
            }

            // Compute remove_save_inactive = join_same_col_all - join_same_col_active
            let join_same_col_all = if src_pos > 0 && src_pos + 1 < order.len() {
                same_col_between(order[src_pos - 1], order[src_pos + 1])
            } else {
                0
            };
            let remove_save_inactive = (join_same_col_all - join_same_col_active) as i64;
            let remove_save = remove_save_active + remove_save_inactive;

            // --- INSERTION SIDE ---
            // Trial positions: 0..order.len()-1 (order with cls removed)
            // trial[i] = order[i] for i < src_pos
            // trial[i] = order[i+1] for i >= src_pos
            let trial_len = order.len() - 1;

            // For large dimensions, limit search to a window around src_pos
            // to reduce O(n²) scanning to O(n × window). NN ordering already
            // places similar elements nearby, so most moves are local.
            let ins_lo = src_pos.saturating_sub(search_radius);
            let ins_hi = (src_pos + search_radius + 1).min(trial_len + 1);

            for ins_pos in ins_lo..ins_hi {
                // Look up same_col_all for the pair at this insertion position.
                // ins_pos == 0 or ins_pos == trial_len: boundary → 0
                // ins_pos in 1..trial_len: depends on mapping back to order indices
                let same_col_all_pair = if ins_pos == 0 || ins_pos == trial_len {
                    0i32
                } else if ins_pos < src_pos {
                    // trial[ins_pos-1] = order[ins_pos-1], trial[ins_pos] = order[ins_pos]
                    // → same_col_adj[ins_pos - 1]
                    same_col_adj[ins_pos - 1]
                } else if ins_pos == src_pos {
                    // trial[ins_pos-1] = order[src_pos-1], trial[ins_pos] = order[src_pos+1]
                    // → join pair
                    join_same_col_all
                } else {
                    // ins_pos > src_pos:
                    // trial[ins_pos-1] = order[ins_pos], trial[ins_pos] = order[ins_pos+1]
                    // → same_col_adj[ins_pos]
                    if ins_pos < same_col_adj.len() {
                        same_col_adj[ins_pos]
                    } else {
                        0
                    }
                };

                // Get trial neighbor class IDs
                let ins_left_cls = if ins_pos > 0 {
                    if ins_pos - 1 < src_pos { order[ins_pos - 1] } else { order[ins_pos] }
                } else {
                    usize::MAX
                };
                let ins_right_cls = if ins_pos < trial_len {
                    if ins_pos < src_pos { order[ins_pos] } else { order[ins_pos + 1] }
                } else {
                    usize::MAX
                };

                // Active weight loop: compute full delta + same_col correction
                let mut insert_cost_active = 0i64;
                let mut active_same_col_correction = 0i32;
                for &w in &active_weights {
                    let col = col_assign[cls][w];
                    let left_col = if ins_left_cls != usize::MAX { col_assign[ins_left_cls][w] } else { 0 };
                    let right_col = if ins_right_cls != usize::MAX { col_assign[ins_right_cls][w] } else { 0 };

                    let old_break_at_ins = if right_col != 0 && (left_col == 0 || left_col != right_col) { 1 } else { 0 };
                    let new_break_at_cls = if col != 0 && (left_col == 0 || left_col != col) { 1 } else { 0 };
                    let new_break_at_right = if right_col != 0 && (col == 0 || col != right_col) { 1 } else { 0 };

                    insert_cost_active += (new_break_at_cls + new_break_at_right - old_break_at_ins) as i64;

                    // Correction: this active weight's contribution to same_col_all_pair
                    if left_col != 0 && left_col == right_col {
                        active_same_col_correction += 1;
                    }
                }

                let insert_cost_inactive = (same_col_all_pair - active_same_col_correction) as i64;
                let total_insert_cost = insert_cost_active + insert_cost_inactive;

                let delta = total_insert_cost - remove_save;
                if delta < best_delta {
                    best_delta = delta;
                    best_pos = ins_pos;
                }
            }

            if best_delta < 0 {
                let src = src_pos;
                let cls = order.remove(src);
                let dst = best_pos.min(order.len());
                order.insert(dst, cls);
                improved = true;

                // Incremental same_col_adj update.
                // After moving CLS from position `src` to `dst`, only 3 boundary
                // entries change; the rest shift by ±1 in index space.
                if dst < src {
                    // Shift middle entries right: new[i] = old[i-1] for i in [dst+1, src-1]
                    for i in (dst + 1..src).rev() {
                        same_col_adj[i] = same_col_adj[i - 1];
                    }
                    // Recompute boundary entries
                    if dst > 0 {
                        same_col_adj[dst - 1] = same_col_between(order[dst - 1], order[dst]);
                    }
                    if dst < same_col_adj.len() {
                        same_col_adj[dst] = same_col_between(order[dst], order[dst + 1]);
                    }
                    if src < same_col_adj.len() {
                        same_col_adj[src] = same_col_between(order[src], order[src + 1]);
                    }
                } else if dst > src {
                    // Shift middle entries left: new[i] = old[i+1] for i in [src, dst-2]
                    for i in src..dst.saturating_sub(1) {
                        same_col_adj[i] = same_col_adj[i + 1];
                    }
                    // Recompute boundary entries
                    if src > 0 {
                        same_col_adj[src - 1] = same_col_between(order[src - 1], order[src]);
                    }
                    if dst > 0 && dst - 1 < same_col_adj.len() {
                        same_col_adj[dst - 1] = same_col_between(order[dst - 1], order[dst]);
                    }
                    if dst < same_col_adj.len() {
                        same_col_adj[dst] = same_col_between(order[dst], order[dst + 1]);
                    }
                }

                // Verify incremental update matches full rebuild (debug only)
                debug_assert_eq!(same_col_adj, build_same_col_adj(&order),
                    "incremental same_col_adj update mismatch at src={} dst={}", src, dst);
            }
        }

        best_breaks = count_breaks(&order);
        if profile {
            eprintln!("  or_opt: iter {} breaks={} improved={} ({:?})", iter, best_breaks, improved, start.elapsed());
        }

        if !improved {
            break;
        }
    }

    // Convert order back to permutation: perm[class] = new_position
    let mut perm = vec![0usize; num_classes];
    for (new_pos, &cls) in order.iter().enumerate() {
        perm[cls] = new_pos;
    }
    perm
}

/// Compute Hamming distance between two bitvectors.
#[inline]
fn bv_hamming(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x ^ y).count_ones()).sum()
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
    mapped.dedup();
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

/// Compute optimal token and tsid mappings for a DWA's weights,
/// merging equivalent indices and reordering for locality.
///
/// Equivalent indices (those with identical profiles across all weights)
/// are mapped to the same new ID. Returns (token_mapping, tsid_mapping,
/// new_max_token, new_num_tsids) where mapping[old_id] = new_id.
/// The new_max_token and new_num_tsids reflect the reduced dimensions.
pub fn reorder_dwa_dimensions(
    dwa: &mut DWA,
    max_token: usize,
    num_tsids: usize,
) -> (Vec<usize>, Vec<usize>, usize, usize) {
    let start = std::time::Instant::now();
    let profile = std::env::var("PROFILE_BUILD_TOKENIZER").is_ok();

    // When RANGEMAP_TSID_OUTER is active, the RangeMap outer key is the tsid dimension
    // and the inner value is the token dimension. We swap the parameters so the rest
    // of the function operates generically on "outer" and "inner" dimensions.
    let tsid_outer = RangeMapWeight::tsid_outer_enabled();
    let (effective_max_outer, effective_inner_count) = if tsid_outer {
        (num_tsids.saturating_sub(1), max_token + 1)
    } else {
        (max_token, num_tsids)
    };

    // Shadow with effective values so all downstream code is dimension-generic.
    // "max_token" now means "max outer key" and "num_tsids" means "inner count".
    #[allow(unused_variables)]
    let (max_token, num_tsids) = (effective_max_outer, effective_inner_count);

    // Step 1: Collect unique weights
    let unique_weights_arc = collect_unique_weights(dwa);
    let unique_weights: Vec<&RangeMapWeight> =
        unique_weights_arc.iter().map(|arc| arc.as_ref()).collect();
    if profile { eprintln!("  reorder: collect_unique_weights = {:?}", start.elapsed()); }
    let step2_start = std::time::Instant::now();

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

    // Step 2: Compute tsid equivalence classes + permutation
    let sub_t = std::time::Instant::now();
    let tsid_profiles = build_tsid_profiles(&unique_weights, num_tsids);
    let (tsid_class_map, unique_tsid_profs) = dedup_profiles(&tsid_profiles);
    let new_num_tsids = unique_tsid_profs.len();
    let active_tsids = unique_tsid_profs.iter().filter(|p| !p.is_empty()).count();
    let max_tsid_profile = unique_tsid_profs.iter().map(|p| p.len()).max().unwrap_or(0);
    let total_tsid_ctx = unique_tsid_profs.iter().map(|p| p.len()).sum::<usize>();
    if profile { eprintln!("  reorder: tsid_build_profiles+dedup = {:?} (n_unique={})", sub_t.elapsed(), new_num_tsids); }
    crate::debug!(
        3,
        "reorder_dwa_dimensions: tsid profiles: {} total, {} unique (merged {}), {} active (non-empty), max_profile={}, total_ctx={}",
        num_tsids,
        new_num_tsids,
        num_tsids - new_num_tsids,
        active_tsids,
        max_tsid_profile,
        total_tsid_ctx,
    );
    
    // Column-fingerprint for tsid dimension (fast O(n log n))
    let sub_t = std::time::Instant::now();
    let tsid_class_perm_fp = column_fingerprint_tsid_order(
        &unique_weights,
        unique_tsid_profs.len(),
        Some((&tsid_class_map, num_tsids)),
    );
    if profile { eprintln!("  reorder: tsid_fp = {:?}", sub_t.elapsed()); }
    
    let tsid_class_perm = if unique_tsid_profs.len() <= 2000 {
        // Small enough for NN
        let sub_t = std::time::Instant::now();
        let tsid_class_perm_nn = greedy_nearest_neighbor(&unique_tsid_profs);
        if profile { eprintln!("  reorder: tsid_nn = {:?}", sub_t.elapsed()); }
        
        // Evaluate both
        let nn_inner = count_inner_ranges_for_ordering(&unique_weights, &tsid_class_perm_nn, num_tsids, &tsid_class_map);
        let fp_inner = count_inner_ranges_for_ordering(&unique_weights, &tsid_class_perm_fp, num_tsids, &tsid_class_map);
        
        if fp_inner < nn_inner {
            crate::debug!(3, "reorder_dwa_dimensions: tsid: using fingerprint ordering (inner: {} vs NN: {})", fp_inner, nn_inner);
            tsid_class_perm_fp 
        } else {
            crate::debug!(3, "reorder_dwa_dimensions: tsid: using NN ordering (inner: {} vs fingerprint: {})", nn_inner, fp_inner);
            tsid_class_perm_nn
        }
    } else {
        // Large tsid dimension — skip NN, use FP directly
        if profile { eprintln!("  reorder: tsid_nn = SKIPPED (n_unique={} > 2000)", new_num_tsids); }
        crate::debug!(3, "reorder_dwa_dimensions: tsid: using fingerprint ordering (NN skipped, n={})", new_num_tsids);
        tsid_class_perm_fp
    };
    
    // Compose: old_tsid → class → reordered_class_position
    let tsid_perm: Vec<usize> = (0..num_tsids)
        .map(|i| tsid_class_perm[tsid_class_map[i]])
        .collect();
    if profile { eprintln!("  reorder: tsid_profiles+perm = {:?}", step2_start.elapsed()); }
    let step3_start = std::time::Instant::now();

    // Step 3: Compute token equivalence classes + permutation
    let sub_t = std::time::Instant::now();
    let token_profiles = build_token_profiles(&unique_weights, max_token);
    let (token_class_map, unique_token_profs) = dedup_profiles(&token_profiles);
    if profile { eprintln!("  reorder: token_build_profiles+dedup = {:?} (n_unique={})", sub_t.elapsed(), unique_token_profs.len()); }
    let new_max_token = if unique_token_profs.is_empty() { 0 } else { unique_token_profs.len() - 1 };
    let active_tokens = unique_token_profs.iter().filter(|p| !p.is_empty()).count();
    let max_token_profile = unique_token_profs.iter().map(|p| p.len()).max().unwrap_or(0);
    let total_token_ctx = unique_token_profs.iter().map(|p| p.len()).sum::<usize>();
    crate::debug!(
        3,
        "reorder_dwa_dimensions: token profiles: {} total, {} unique (merged {}), {} active (non-empty), max_profile={}, total_ctx={}",
        max_token + 1,
        unique_token_profs.len(),
        (max_token + 1) - unique_token_profs.len(),
        active_tokens,
        max_token_profile,
        total_token_ctx,
    );
    
    // Column-fingerprint ordering is much faster than NN (O(n log n) vs O(n²×W))
    // and produces comparable results when followed by or-opt local search.
    // For large token dimensions (>500 classes), skip the expensive NN entirely.
    let sub_t = std::time::Instant::now();
    let token_class_perm_fp = column_fingerprint_order_with_class_map(
        &unique_weights,
        unique_token_profs.len(),
        Some((&token_class_map, max_token + 1)),
    );
    if profile { eprintln!("  reorder: token_fp = {:?}", sub_t.elapsed()); }
    
    let (token_class_perm, nn_outer, fp_outer) = if unique_token_profs.len() <= 2000 {
        // Small enough for NN to be fast — compute both and pick better
        let sub_t = std::time::Instant::now();
        let token_class_perm_nn = greedy_nearest_neighbor(&unique_token_profs);
        if profile { eprintln!("  reorder: token_nn = {:?}", sub_t.elapsed()); }
        
        let sub_t = std::time::Instant::now();
        let nn_outer = count_outer_ranges_for_ordering(&unique_weights, &token_class_perm_nn, max_token, &token_class_map);
        let fp_outer = count_outer_ranges_for_ordering(&unique_weights, &token_class_perm_fp, max_token, &token_class_map);
        if profile { eprintln!("  reorder: token_count_ranges = {:?}", sub_t.elapsed()); }
        
        if fp_outer < nn_outer {
            crate::debug!(3, "reorder_dwa_dimensions: using fingerprint ordering (outer: {} vs NN: {})", fp_outer, nn_outer);
            (token_class_perm_fp, nn_outer, fp_outer)
        } else {
            crate::debug!(3, "reorder_dwa_dimensions: using NN ordering (outer: {} vs fingerprint: {})", nn_outer, fp_outer);
            (token_class_perm_nn, nn_outer, fp_outer)
        }
    } else {
        // Large token dimension — skip NN (saves O(n²×W)), use FP directly
        if profile { eprintln!("  reorder: token_nn = SKIPPED (n_unique={} > 2000)", unique_token_profs.len()); }
        let fp_outer = count_outer_ranges_for_ordering(&unique_weights, &token_class_perm_fp, max_token, &token_class_map);
        crate::debug!(3, "reorder_dwa_dimensions: using fingerprint ordering (NN skipped, n={}), outer={}", unique_token_profs.len(), fp_outer);
        (token_class_perm_fp, fp_outer, fp_outer) // use fp_outer for both to avoid confusion
    };

    // Step 3b: Or-opt local search to directly minimize outer range count.
    // For large dimensions (tsid-outer), use a tight budget — NN ordering
    // already captures 97%+ of the benefit and the first or-opt iteration
    // provides most of the remaining improvement.
    let or_opt_budget = if unique_token_profs.len() > 2000 {
        std::time::Duration::from_millis(200)
    } else {
        std::time::Duration::from_millis(500)
    };
    let token_class_perm = or_opt_outer_ranges(
        &unique_weights,
        &token_class_perm,
        max_token,
        &token_class_map,
        or_opt_budget,
    );
    if profile {
        let opt_outer = count_outer_ranges_for_ordering(&unique_weights, &token_class_perm, max_token, &token_class_map);
        eprintln!("  reorder: after or_opt outer={} (was NN={}, FP={})", opt_outer, nn_outer, fp_outer);
    }

    // Compose: old_token → class → reordered_class_position
    let token_perm: Vec<usize> = (0..max_token + 1)
        .map(|i| token_class_perm[token_class_map[i]])
        .collect();
    if profile { eprintln!("  reorder: token_profiles+perm = {:?}", step3_start.elapsed()); }

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
    if profile { eprintln!("  reorder: build_weight_map = {:?}", apply_start.elapsed()); }
    let step5_start = std::time::Instant::now();

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
    if profile { eprintln!("  reorder: apply_weights = {:?}", step5_start.elapsed()); }

    // Count new ranges (properly deduped: outer by weight Arc, inner by RangeSet Arc)
    let new_ranges = dwa.num_ranges_interned();
    let (new_outer, new_inner) = if profile || crate::r#macro::is_debug_level_enabled(3) {
        count_ranges_breakdown(dwa)
    } else { (0, 0) };

    // Compute theoretical minimum outer ranges: for each weight, the minimum is
    // the number of DISTINCT tsid_set columns. This is achieved when all tokens
    // with the same column are adjacent.
    let (min_outer, baseline_outer, baseline_inner) = if profile || crate::r#macro::is_debug_level_enabled(3) {
        let mut min_out = 0usize;
        let mut seen = std::collections::HashSet::new();
        for state in &dwa.states.0 {
            let mut process = |w: &crate::dwa_i32::dwa::Weight| -> usize {
                use crate::datastructures::AbstractWeight;
                if let AbstractWeight::RangeMap(rm) = w {
                    let ptr = Arc::as_ptr(rm) as usize;
                    if seen.insert(ptr) {
                        let mut distinct: std::collections::HashSet<usize> = std::collections::HashSet::new();
                        for (_, tsid_set) in rm.map.range_values() {
                            distinct.insert(Arc::as_ptr(&tsid_set.inner) as usize);
                        }
                        return distinct.len();
                    }
                }
                0
            };
            if let Some(fw) = &state.final_weight {
                min_out += process(fw);
            }
            for w in state.trans_weights.values() {
                min_out += process(w);
            }
        }
        (min_out, new_outer, new_inner) // Use new_ as baseline since we already permuted
    } else { (0, 0, 0) };

    crate::debug!(
        3,
        "REORDER_DWA: baseline_ranges={} -> new_ranges={} ({:.1}% reduction, outer={} inner={}, min_outer={}) in {:?}",
        baseline_ranges,
        new_ranges,
        if baseline_ranges > 0 {
            (1.0 - new_ranges as f64 / baseline_ranges as f64) * 100.0
        } else {
            0.0
        },
        new_outer,
        new_inner,
        min_outer,
        start.elapsed()
    );

    crate::debug!(
        3,
        "reorder_dwa_dimensions: applied permutations in {:?} (total {:?})",
        apply_start.elapsed(),
        start.elapsed()
    );

    crate::debug!(
        3,
        "reorder_dwa_dimensions: dimensions: tokens {}→{} (merged {}), tsids {}→{} (merged {})",
        max_token + 1,
        new_max_token + 1,
        (max_token + 1) - (new_max_token + 1),
        num_tsids,
        new_num_tsids,
        num_tsids - new_num_tsids,
    );

    // Map back to caller's dimension semantics.
    // In the function body, "token_perm" = outer perm, "tsid_perm" = inner perm.
    // In tsid-outer mode: outer = tsid dimension, inner = token dimension.
    if tsid_outer {
        // Swap: return (inner_perm_as_token, outer_perm_as_tsid, inner_max_as_token, outer_count_as_tsid)
        (tsid_perm, token_perm, new_num_tsids.saturating_sub(1), new_max_token + 1)
    } else {
        (token_perm, tsid_perm, new_max_token, new_num_tsids)
    }
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
/// `state_to_internal_tsid` maps raw state IDs to internal tsid indices (0..num_internal_tsids-1).
///
/// For **tsid ordering**: two internal tsids should be adjacent if similar sets of
/// tokens match them across terminals.
///
/// For **token ordering**: two tokens should be adjacent if they appear in
/// similar internal_tsid×terminal contexts.
///
/// Returns `(token_perm, tsid_perm)` where `perm[old_id] = new_id`.
/// `token_perm` has length `max_token + 1`, `tsid_perm` has length `num_internal_tsids`.
pub fn predict_orderings_from_possible_matches(
    possible_matches: &BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    max_internal_token: usize,
    num_internal_tsids: usize,
    state_to_internal_tsid: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let start = std::time::Instant::now();
    let num_tokens = max_internal_token + 1;

    // Build tsid profiles: for each internal tsid, collect all token IDs
    // from any possible_matches entry that maps to this internal tsid.
    let mut tsid_profiles: Vec<Vec<u32>> = vec![Vec::new(); num_internal_tsids];
    // Build token profiles: for each token, collect internal tsid IDs
    // where it appears in any possible_matches entry.
    let mut token_profiles: Vec<Vec<u32>> = vec![Vec::new(); num_tokens];

    for (&state_id, terminal_map) in possible_matches {
        let sid = state_id.0;
        if sid >= state_to_internal_tsid.len() {
            continue;
        }
        let itid = state_to_internal_tsid[sid];
        if itid >= num_internal_tsids {
            continue;
        }
        for (_terminal_id, token_bv) in terminal_map {
            for range in token_bv.ranges() {
                let lo = *range.start();
                let hi = *range.end();
                for tok in lo..=hi.min(max_internal_token) {
                    token_profiles[tok].push(itid as u32);
                    tsid_profiles[itid].push(tok as u32);
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
        "predict_orderings: profiles={:?}, tsid_nn={:?}, token_nn={:?}, total={:?} (tokens={}, internal_tsids={})",
        profile_time,
        tsid_time - profile_time,
        total_time - tsid_time,
        total_time,
        num_tokens,
        num_internal_tsids,
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
