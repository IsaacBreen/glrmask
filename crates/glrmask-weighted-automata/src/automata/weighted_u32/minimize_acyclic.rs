//! Acyclic weighted DWA minimization via weight pushing and height-layered coloring.
//!
//! The pipeline pushes weights backward to discard dead token flow, groups
//! states by topological height, colors each height bucket subject to
//! compatibility constraints, and reconstructs the minimized automaton from the
//! merged buckets.
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::dwa::{DWA, DWAState};
use crate::ds::weight::{
    shared_rangeset, ScopedWeightOpCache, SharedTokenSet, Weight, WeightIntersectionIndex,
};

type Label = i32;

/// Ordering policy for exact pointwise merge groups during acyclic weighted
/// minimization. The policy affects only representation choices among already
/// compatible classes; it does not alter the accepted weighted language.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PointwiseClassOrder {
    /// Preserve the partition-refinement class order used historically.
    Stable,
    /// Place denser partial behavior functions first so their constraints are
    /// absorbed before smaller compatible fragments are grouped greedily.
    DescendingDomain,
}

const UNMAPPED: u32 = u32::MAX;
fn weighted_dwa_minimize_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

fn reconstruction_token_range_coalescing_enabled(
    automatic_large_minimizer: bool,
    pending_count: usize,
) -> bool {
    if std::env::var("GLRMASK_WEIGHT_UNION_COALESCE_TOKEN_RANGES")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
    {
        return true;
    }
    match std::env::var("GLRMASK_WEIGHT_UNION_COALESCE_MODE")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("off" | "disabled" | "merge") => false,
        Some("force" | "on" | "coalesce") => true,
        Some("auto") | None => automatic_large_minimizer && (16..=127).contains(&pending_count),
        Some(other) => panic!(
            "unknown GLRMASK_WEIGHT_UNION_COALESCE_MODE={other:?}; expected auto, off, or force"
        ),
    }
}

fn mapped_target(old_to_new: &[u32], target: u32) -> Option<u32> {
    let mapped = old_to_new.get(target as usize).copied().unwrap_or(UNMAPPED);
    (mapped != UNMAPPED).then_some(mapped)
}
fn compute_reachable_from_start(dwa: &DWA, start_state: usize) -> Vec<bool> {
    let mut reachable = vec![false; dwa.states().len()];
    if start_state >= dwa.states().len() {
        return reachable;
    }

    let mut stack = vec![start_state];
    while let Some(state_id) = stack.pop() {
        if reachable[state_id] {
            continue;
        }

        reachable[state_id] = true;
        for (target, _) in dwa.states()[state_id].transitions.values() {
            let target = *target as usize;
            if target < dwa.states().len() && !reachable[target] {
                stack.push(target);
            }
        }
    }

    reachable
}

fn weight_body_id(weight: &Weight) -> usize {
    weight.ptr_key()
}

fn intersection_memo_key(left: &Weight, right: &Weight) -> (usize, usize) {
    let left_id = weight_body_id(left);
    let right_id = weight_body_id(right);
    if left_id <= right_id {
        (left_id, right_id)
    } else {
        (right_id, left_id)
    }
}

fn memoized_intersection(
    cache: &mut FxHashMap<(usize, usize), Weight>,
    left: &Weight,
    right: &Weight,
    index: Option<&WeightIntersectionIndex>,
) -> Weight {
    if left.is_empty() || right.is_empty() {
        return Weight::empty();
    }
    if left.is_full() {
        return right.clone();
    }
    if right.is_full() {
        return left.clone();
    }

    let key = intersection_memo_key(left, right);
    if let Some(existing) = cache.get(&key) {
        return existing.clone();
    }

    let value = index.map_or_else(|| left.intersection(right), |index| left.intersection_with_index(index));
    cache.insert(key, value.clone());
    value
}

// Push weights backward before any topological analysis.

/// Push weights: intersect each transition weight with the backward-reachable
/// set of its target state.  This ensures transitions only carry tokens that
/// can actually reach acceptance, enabling more state merges.
///
/// Returns (changed, topo_order, reachable_sets) so callers can reuse them.
pub fn push_weights(dwa: &mut DWA) -> (bool, Option<Vec<usize>>, Vec<Weight>) {
    let profile_enabled = weighted_dwa_minimize_profile_enabled();
    let n = dwa.states().len();
    if n == 0 {
        return (false, Some(Vec::new()), Vec::new());
    }

    // 1. Topological order (Kahn's algorithm)
    let topo_started_at = profile_enabled.then(Instant::now);
    let Some(topo) = compute_topo_order(dwa) else {
        return (false, None, Vec::new()); // cyclic
    };
    let topo_ms = topo_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    // 2+3 combined: backward reachable sets + push transition weights.
    // In reverse topo order, each state's targets have already been processed,
    // so we compute reachable[u] and push transitions in a single pass.
    let mut reachable: Vec<Weight> = vec![Weight::empty(); n];
    let mut intersection_cache = FxHashMap::default();
    let mut incoming_transition_counts = vec![0usize; n];
    for state in dwa.states() {
        for (target, _) in state.transitions.values() {
            if (*target as usize) < n {
                incoming_transition_counts[*target as usize] += 1;
            }
        }
    }
    let mut intersection_indexes = FxHashMap::<usize, WeightIntersectionIndex>::default();
    let mut indexed_intersection_uses = 0usize;
    let mut changed = false;
    let mut target_full = 0usize;
    let mut target_empty = 0usize;
    let mut target_partial = 0usize;
    let mut reachable_parts = 0usize;
    let mut pushed_transitions = 0usize;
    let mut intersection_ms = 0.0;
    let pointer_fast_path =
        std::env::var_os("GLRMASK_WEIGHTED_MINIMIZE_DISABLE_PUSH_POINTER_FAST_PATH").is_none();
    let subset_fast_path = std::env::var("GLRMASK_WEIGHTED_MINIMIZE_PUSH_SUBSET_FAST_PATH")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false);
    let mut subset_checks = 0usize;
    let mut subset_hits = 0usize;
    let mut subset_ms = 0.0;
    let mut union_ms = 0.0;
    let mut apply_ms = 0.0;
    let mut union_size_histogram = [0usize; 7];
    let mut max_union_size = 0usize;
    let mut union_key_occurrences = 0usize;
    let mut union_key_repeats = 0usize;
    let mut union_keys_seen = FxHashSet::<Vec<usize>>::default();

    for &u in topo.iter().rev() {
        let state = &dwa.states()[u];
        let mut state_reachable_parts: Vec<Weight> =
            Vec::with_capacity(state.transitions.len() + 1);
        let mut acc_full = false;
        if let Some(final_weight) = &state.final_weight {
            if final_weight.is_full() {
                acc_full = true;
            } else if !final_weight.is_empty() {
                state_reachable_parts.push(final_weight.clone());
            }
        }
        let mut pushed: Vec<(Label, u32, Option<Weight>)> = Vec::new();
        for (&lbl, (target, w)) in &state.transitions {
            let t = *target as usize;
            if t >= n {
                continue;
            }
            if reachable[t].is_full() {
                target_full += 1;
                if !acc_full && !w.is_empty() {
                    if w.is_full() {
                        acc_full = true;
                        state_reachable_parts.clear();
                    } else {
                        state_reachable_parts.push(w.clone());
                    }
                }
                // w ∩ all = w, no push needed
            } else if reachable[t].is_empty() {
                target_empty += 1;
                // w ∩ empty = empty, remove transition
                pushed.push((lbl, *target, None));
                // Contributes nothing to acc
            } else {
                target_partial += 1;
                let intersection_started_at = profile_enabled.then(Instant::now);
                let new_w = if pointer_fast_path && w.storage_ptr_eq(&reachable[t]) {
                    subset_hits += 1;
                    w.clone()
                } else if subset_fast_path {
                    let subset_started_at = profile_enabled.then(Instant::now);
                    subset_checks += 1;
                    let contained = w.is_subset(&reachable[t]);
                    if let Some(subset_started_at) = subset_started_at {
                        subset_ms += subset_started_at.elapsed().as_secs_f64() * 1000.0;
                    }
                    if contained {
                        subset_hits += 1;
                        w.clone()
                    } else {
                        let index = (incoming_transition_counts[t] >= 8).then(|| {
                            let key = reachable[t].ptr_key();
                            intersection_indexes
                                .entry(key)
                                .or_insert_with(|| reachable[t].intersection_index())
                        });
                        if index.is_some() {
                            indexed_intersection_uses += 1;
                        }
                        memoized_intersection(
                            &mut intersection_cache,
                            w,
                            &reachable[t],
                            index.as_deref(),
                        )
                    }
                } else {
                    let index = (incoming_transition_counts[t] >= 8).then(|| {
                        let key = reachable[t].ptr_key();
                        intersection_indexes
                            .entry(key)
                            .or_insert_with(|| reachable[t].intersection_index())
                    });
                    if index.is_some() {
                        indexed_intersection_uses += 1;
                    }
                    memoized_intersection(
                        &mut intersection_cache,
                        w,
                        &reachable[t],
                        index.as_deref(),
                    )
                };
                if let Some(started_at) = intersection_started_at {
                    intersection_ms += started_at.elapsed().as_secs_f64() * 1000.0;
                }
                if !acc_full && !new_w.is_empty() {
                    if new_w.is_full() {
                        acc_full = true;
                        state_reachable_parts.clear();
                    } else {
                        state_reachable_parts.push(new_w.clone());
                    }
                }
                if new_w != *w {
                    pushed.push((lbl, *target, if new_w.is_empty() { None } else { Some(new_w) }));
                }
            }
        }
        reachable[u] = if acc_full {
            Weight::all()
        } else {
            reachable_parts += state_reachable_parts.len();
            if profile_enabled {
                let union_size = state_reachable_parts.len();
                max_union_size = max_union_size.max(union_size);
                let bucket = match union_size {
                    0 => 0,
                    1 => 1,
                    2 => 2,
                    3 => 3,
                    4 => 4,
                    5..=16 => 5,
                    _ => 6,
                };
                union_size_histogram[bucket] += 1;
                if union_size >= 2 {
                    let mut key: Vec<usize> = state_reachable_parts
                        .iter()
                        .map(weight_body_id)
                        .collect();
                    key.sort_unstable();
                    key.dedup();
                    union_key_occurrences += 1;
                    if !union_keys_seen.insert(key) {
                        union_key_repeats += 1;
                    }
                }
            }
            let union_started_at = profile_enabled.then(Instant::now);
            let result = Weight::union_all(state_reachable_parts.iter());
            if let Some(started_at) = union_started_at {
                union_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            }
            result
        };

        pushed_transitions += pushed.len();
        let apply_started_at = profile_enabled.then(Instant::now);
        for (lbl, target, new_w_opt) in pushed {
            if let Some(new_w) = new_w_opt {
                dwa.states_mut()[u].transitions.insert(lbl, (target, new_w));
            } else {
                dwa.states_mut()[u].transitions.remove(&lbl);
            }
            changed = true;
        }
        if let Some(started_at) = apply_started_at {
            apply_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
    }

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_push] states={} topo_ms={:.3} target_full={} target_empty={} target_partial={} subset_checks={} subset_hits={} subset_ms={:.3} intersection_ms={:.3} intersection_cache_entries={} intersection_indexes={} indexed_intersection_uses={} reachable_parts={} union_ms={:.3} union_sizes=[{},{},{},{},{},{},{}] max_union_size={} union_key_occurrences={} union_unique_keys={} union_key_repeats={} pushed_transitions={} apply_ms={:.3}",
            n,
            topo_ms,
            target_full,
            target_empty,
            target_partial,
            subset_checks,
            subset_hits,
            subset_ms,
            intersection_ms,
            intersection_cache.len(),
            intersection_indexes.len(),
            indexed_intersection_uses,
            reachable_parts,
            union_ms,
            union_size_histogram[0],
            union_size_histogram[1],
            union_size_histogram[2],
            union_size_histogram[3],
            union_size_histogram[4],
            union_size_histogram[5],
            union_size_histogram[6],
            max_union_size,
            union_key_occurrences,
            union_keys_seen.len(),
            union_key_repeats,
            pushed_transitions,
            apply_ms,
        );
    }

    (changed, Some(topo), reachable)
}

struct ParallelPushStateResult {
    state: usize,
    reachable: Weight,
    changes: Vec<(Label, u32, Option<Weight>)>,
    target_full: usize,
    target_empty: usize,
    target_partial: usize,
    pointer_hits: usize,
    intersections: usize,
    union_parts: usize,
}

/// Exact height-parallel form of [`push_weights`].
///
/// In an acyclic DWA, every transition from a state of height `h` targets a
/// strictly lower height. Therefore all states at one height read only live
/// domains finalized by earlier rounds and can compute their recurrence in
/// parallel. Applying their edge rewrites after the round cannot affect any
/// peer computation because no same-height edge exists.
fn push_weights_parallel_by_height(
    dwa: &mut DWA,
) -> (bool, Option<Vec<usize>>, Vec<Weight>) {
    let profile_enabled = weighted_dwa_minimize_profile_enabled();
    let total_started_at = profile_enabled.then(Instant::now);
    let n = dwa.states().len();
    if n == 0 {
        return (false, Some(Vec::new()), Vec::new());
    }
    let Some(topo) = compute_topo_order(dwa) else {
        return (false, None, Vec::new());
    };
    let heights = compute_heights(dwa, &topo);
    let max_height = heights.iter().copied().max().unwrap_or(0);
    let mut states_by_height = vec![Vec::<usize>::new(); max_height + 1];
    for (state, height) in heights.into_iter().enumerate() {
        states_by_height[height].push(state);
    }

    let mut reachable = vec![Weight::empty(); n];
    let mut changed = false;
    let mut target_full = 0usize;
    let mut target_empty = 0usize;
    let mut target_partial = 0usize;
    let mut pointer_hits = 0usize;
    let mut intersections = 0usize;
    let mut union_parts = 0usize;
    let mut changed_transitions = 0usize;

    for states in states_by_height {
        let results = states
            .par_iter()
            .map(|&u| {
                let state = &dwa.states()[u];
                let mut parts = Vec::<Weight>::with_capacity(state.transitions.len() + 1);
                let mut acc_full = false;
                if let Some(final_weight) = &state.final_weight {
                    if final_weight.is_full() {
                        acc_full = true;
                    } else if !final_weight.is_empty() {
                        parts.push(final_weight.clone());
                    }
                }
                let mut changes = Vec::new();
                let mut local_target_full = 0usize;
                let mut local_target_empty = 0usize;
                let mut local_target_partial = 0usize;
                let mut local_pointer_hits = 0usize;
                let mut local_intersections = 0usize;
                for (&label, (target, weight)) in &state.transitions {
                    let target_index = *target as usize;
                    if target_index >= n {
                        continue;
                    }
                    let target_domain = &reachable[target_index];
                    let contribution = if target_domain.is_full() {
                        local_target_full += 1;
                        weight.clone()
                    } else if target_domain.is_empty() {
                        local_target_empty += 1;
                        changes.push((label, *target, None));
                        continue;
                    } else if weight.storage_ptr_eq(target_domain) {
                        local_target_partial += 1;
                        local_pointer_hits += 1;
                        weight.clone()
                    } else {
                        local_target_partial += 1;
                        local_intersections += 1;
                        let intersection = weight.intersection(target_domain);
                        if intersection != *weight {
                            changes.push((
                                label,
                                *target,
                                (!intersection.is_empty()).then(|| intersection.clone()),
                            ));
                        }
                        intersection
                    };
                    if !acc_full && !contribution.is_empty() {
                        if contribution.is_full() {
                            acc_full = true;
                            parts.clear();
                        } else {
                            parts.push(contribution);
                        }
                    }
                }
                let union_part_count = parts.len();
                let state_reachable = if acc_full {
                    Weight::all()
                } else {
                    Weight::union_all(parts.iter())
                };
                ParallelPushStateResult {
                    state: u,
                    reachable: state_reachable,
                    changes,
                    target_full: local_target_full,
                    target_empty: local_target_empty,
                    target_partial: local_target_partial,
                    pointer_hits: local_pointer_hits,
                    intersections: local_intersections,
                    union_parts: union_part_count,
                }
            })
            .collect::<Vec<_>>();

        for result in results {
            reachable[result.state] = result.reachable;
            target_full += result.target_full;
            target_empty += result.target_empty;
            target_partial += result.target_partial;
            pointer_hits += result.pointer_hits;
            intersections += result.intersections;
            union_parts += result.union_parts;
            changed_transitions += result.changes.len();
            if !result.changes.is_empty() {
                changed = true;
                let state = &mut dwa.states_mut()[result.state];
                for (label, target, replacement) in result.changes {
                    if let Some(weight) = replacement {
                        state.transitions.insert(label, (target, weight));
                    } else {
                        state.transitions.remove(&label);
                    }
                }
            }
        }
    }

    if let Some(started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_push_parallel] states={} heights={} target_full={} target_empty={} target_partial={} pointer_hits={} intersections={} union_parts={} changed_transitions={} total_ms={:.3}",
            n,
            max_height + 1,
            target_full,
            target_empty,
            target_partial,
            pointer_hits,
            intersections,
            union_parts,
            changed_transitions,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    (changed, Some(topo), reachable)
}

// Topological analysis.

fn compute_topo_order(dwa: &DWA) -> Option<Vec<usize>> {
    let n = dwa.states().len();
    let mut in_degree = vec![0u32; n];
    for state in dwa.states() {
        for (_, (target, _)) in &state.transitions {
            let t = *target as usize;
            if t < n {
                in_degree[t] += 1;
            }
        }
    }

    let mut queue: Vec<usize> = in_degree
        .iter()
        .enumerate()
        .filter(|(_, d)| **d == 0)
        .map(|(i, _)| i)
        .collect();
    let mut head = 0;
    let mut topo = Vec::with_capacity(n);

    while head < queue.len() {
        let u = queue[head];
        head += 1;
        topo.push(u);
        for (_, (target, _)) in &dwa.states()[u].transitions {
            let t = *target as usize;
            if t < n {
                in_degree[t] -= 1;
                if in_degree[t] == 0 {
                    queue.push(t);
                }
            }
        }
    }

    if topo.len() == n {
        Some(topo)
    } else {
        None // cyclic
    }
}

/// Needed sets: for each state, the set of tokens that can flow from that
/// state to any accepting state.  Computed in topological order (leaves first).
fn compute_heights(dwa: &DWA, topo: &[usize]) -> Vec<usize> {
    let n = dwa.states().len();
    let mut heights = vec![0usize; n];
    // Process in reverse topo order so children are resolved before parents
    for &u in topo.iter().rev() {
        heights[u] = dwa.states()[u]
            .transitions
            .values()
            .filter_map(|(target, _)| {
                let t = *target as usize;
                (t < n).then(|| heights[t] + 1)
            })
            .max()
            .unwrap_or(0);
    }
    heights
}

#[derive(Clone)]
struct ProductiveTransition {
    label: Label,
    target: u32,
    weight: Weight,
}

fn compute_productive_transitions(
    dwa: &DWA,
    needed: &[Weight],
    edges_already_productive: bool,
) -> Vec<Vec<ProductiveTransition>> {
    let n = dwa.states().len();
    let mut result = Vec::with_capacity(n);
    let mut cache = ScopedWeightOpCache::default();

    for state in dwa.states() {
        let mut transitions = Vec::with_capacity(state.transitions.len());
        for (&label, (target, weight)) in &state.transitions {
            let t = *target as usize;
            if t >= n {
                continue;
            }
            if needed[t].is_empty() {
                continue;
            }
            if edges_already_productive {
                transitions.push(ProductiveTransition {
                    label,
                    target: *target,
                    weight: weight.clone(),
                });
                continue;
            }
            // The productive relation is the only transition relation observed
            // by minimization.  On the legacy path `weight` has already been
            // pushed and this is normally an identity operation.  On the fused
            // determinize/minimize path the DWA remains unmodified, so derive
            // the exact same relation explicitly from the supplied live domain.
            let productive_weight = if needed[t].is_full()
                || weight.storage_ptr_eq(&needed[t])
                || weight.is_subset(&needed[t])
            {
                weight.clone()
            } else {
                cache.intersection(weight, &needed[t])
            };
            if productive_weight.is_empty() {
                continue;
            }
            transitions.push(ProductiveTransition {
                label,
                target: *target,
                weight: productive_weight,
            });
        }
        result.push(transitions);
    }

    result
}

#[cfg(debug_assertions)]
fn debug_assert_pushed_weights_within_needed(dwa: &DWA, needed: &[Weight]) {
    debug_assert_eq!(dwa.states().len(), needed.len());

    let n = dwa.states().len();
    for (state_id, state) in dwa.states().iter().enumerate() {
        let source_needed = &needed[state_id];
        if let Some(final_weight) = &state.final_weight {
            debug_assert!(
                final_weight.is_subset(source_needed),
                "pushed DWA invariant violated: final weight at state {state_id} is not contained in needed[state]",
            );
        }

        for (&label, (target, weight)) in &state.transitions {
            let target_id = *target as usize;
            if target_id >= n {
                continue;
            }
            debug_assert!(
                weight.is_subset(source_needed),
                "pushed DWA invariant violated: transition weight at state {state_id}, label {label} is not contained in needed[source]",
            );
            debug_assert!(
                weight.is_subset(&needed[target_id]),
                "pushed DWA invariant violated: transition weight at state {state_id}, label {label} is not contained in needed[target]",
            );
        }
    }
}

// Incompatibility graph and coloring.

/// Check if `w_a.intersection(domain) == w_b.intersection(domain)` without
/// allocating intermediate Weight objects.  Three-way merge scan over the
/// tsid dimension; at each atomic sub-interval the token-level agreement is
/// verified with a zero-allocation sweep (`token_sets_agree_on_domain`).
fn weights_equal_on_domain(w_a: &Weight, w_b: &Weight, domain: &Weight) -> bool {
    // Fast path: identical Arc ⇒ trivially equal on any domain.
    if w_a.storage_ptr_eq(w_b) {
        return true;
    }
    if domain.is_empty() {
        return true;
    }

    let mut a_iter = w_a.raw_range_values();
    let mut b_iter = w_b.raw_range_values();
    let mut a_entry = a_iter.next();
    let mut b_entry = b_iter.next();

    for (d_range, d_tokens) in domain.raw_range_values() {
        let d_lo = *d_range.start();
        let d_hi = *d_range.end();

        if d_tokens.is_empty() {
            continue;
        }

        // Skip entries fully before this domain range.
        while a_entry.as_ref().is_some_and(|(r, _)| *r.end() < d_lo) {
            a_entry = a_iter.next();
        }
        while b_entry.as_ref().is_some_and(|(r, _)| *r.end() < d_lo) {
            b_entry = b_iter.next();
        }

        let mut pos = d_lo;
        while pos <= d_hi {
            // Determine w_a's token set at `pos` and how far it extends.
            let (a_tokens, a_bound): (Option<&Arc<RangeSetBlaze<u32>>>, u32) = match &a_entry {
                Some((r, tokens)) if *r.start() <= pos => (Some(tokens), *r.end()),
                Some((r, _)) => (None, r.start() - 1), // gap; r.start()>pos≥1
                None => (None, d_hi),
            };
            let (b_tokens, b_bound): (Option<&Arc<RangeSetBlaze<u32>>>, u32) = match &b_entry {
                Some((r, tokens)) if *r.start() <= pos => (Some(tokens), *r.end()),
                Some((r, _)) => (None, r.start() - 1),
                None => (None, d_hi),
            };

            let sub_end = d_hi.min(a_bound).min(b_bound);

            // Check: a_tokens ∩ d_tokens  ==  b_tokens ∩ d_tokens
            match (a_tokens, b_tokens) {
                (None, None) => { /* both empty → agree */ }
                (Some(at), None) => {
                    if !at.as_ref().is_disjoint(d_tokens.as_ref()) {
                        return false;
                    }
                }
                (None, Some(bt)) => {
                    if !bt.as_ref().is_disjoint(d_tokens.as_ref()) {
                        return false;
                    }
                }
                (Some(at), Some(bt)) => {
                    if !Arc::ptr_eq(at, bt) && at.as_ref() != bt.as_ref() {
                        if !token_sets_agree_on_domain(
                            at.as_ref(),
                            bt.as_ref(),
                            d_tokens.as_ref(),
                        ) {
                            return false;
                        }
                    }
                }
            }

            if sub_end == u32::MAX {
                break;
            }
            pos = sub_end + 1;
            while a_entry.as_ref().is_some_and(|(r, _)| *r.end() < pos) {
                a_entry = a_iter.next();
            }
            while b_entry.as_ref().is_some_and(|(r, _)| *r.end() < pos) {
                b_entry = b_iter.next();
            }
        }
    }

    true
}

/// Zero-allocation check: `a ∩ d == b ∩ d` for `RangeSetBlaze` values.
/// Sweep-scans the sorted ranges of `a`, `b`, and `d` in parallel.
fn token_sets_agree_on_domain(
    a: &RangeSetBlaze<u32>,
    b: &RangeSetBlaze<u32>,
    d: &RangeSetBlaze<u32>,
) -> bool {
    let mut a_ranges = a.ranges().peekable();
    let mut b_ranges = b.ranges().peekable();

    for d_range in d.ranges() {
        let d_lo = *d_range.start();
        let d_hi = *d_range.end();

        while a_ranges.peek().is_some_and(|r| *r.end() < d_lo) {
            a_ranges.next();
        }
        while b_ranges.peek().is_some_and(|r| *r.end() < d_lo) {
            b_ranges.next();
        }

        let mut pos = d_lo;
        while pos <= d_hi {
            let (in_a, a_end) = match a_ranges.peek() {
                Some(r) if *r.start() <= pos => (true, *r.end()),
                Some(r) => (false, r.start() - 1),
                None => (false, d_hi),
            };
            let (in_b, b_end) = match b_ranges.peek() {
                Some(r) if *r.start() <= pos => (true, *r.end()),
                Some(r) => (false, r.start() - 1),
                None => (false, d_hi),
            };

            if in_a != in_b {
                return false;
            }

            let sub_end = d_hi.min(a_end).min(b_end);
            if sub_end == u32::MAX {
                break;
            }
            pos = sub_end + 1;
            while a_ranges.peek().is_some_and(|r| *r.end() < pos) {
                a_ranges.next();
            }
            while b_ranges.peek().is_some_and(|r| *r.end() < pos) {
                b_ranges.next();
            }
        }
    }

    true
}

fn token_sets_intersect_three(
    a: &RangeSetBlaze<u32>,
    b: &RangeSetBlaze<u32>,
    c: &RangeSetBlaze<u32>,
) -> bool {
    let mut a_ranges = a.ranges().peekable();
    let mut b_ranges = b.ranges().peekable();
    let mut c_ranges = c.ranges().peekable();

    loop {
        let (Some(a_range), Some(b_range), Some(c_range)) =
            (a_ranges.peek(), b_ranges.peek(), c_ranges.peek())
        else {
            return false;
        };

        let start = (*a_range.start()).max(*b_range.start()).max(*c_range.start());
        let end = (*a_range.end()).min(*b_range.end()).min(*c_range.end());
        if start <= end {
            return true;
        }

        let min_end = (*a_range.end()).min(*b_range.end()).min(*c_range.end());
        if *a_range.end() == min_end {
            a_ranges.next();
        }
        if *b_range.end() == min_end {
            b_ranges.next();
        }
        if *c_range.end() == min_end {
            c_ranges.next();
        }
    }
}

fn token_sets_agree_on_domain_intersection(
    a: &RangeSetBlaze<u32>,
    b: &RangeSetBlaze<u32>,
    left: &RangeSetBlaze<u32>,
    right: &RangeSetBlaze<u32>,
) -> bool {
    let mut a_ranges = a.ranges().peekable();
    let mut b_ranges = b.ranges().peekable();
    let mut left_ranges = left.ranges().peekable();
    let mut right_ranges = right.ranges().peekable();

    loop {
        while let (Some(left_range), Some(right_range)) = (left_ranges.peek(), right_ranges.peek()) {
            if *left_range.end() < *right_range.start() {
                left_ranges.next();
            } else if *right_range.end() < *left_range.start() {
                right_ranges.next();
            } else {
                break;
            }
        }

        let (Some(left_range), Some(right_range)) = (left_ranges.peek(), right_ranges.peek()) else {
            return true;
        };

        let d_lo = (*left_range.start()).max(*right_range.start());
        let d_hi = (*left_range.end()).min(*right_range.end());
        while a_ranges.peek().is_some_and(|r| *r.end() < d_lo) {
            a_ranges.next();
        }
        while b_ranges.peek().is_some_and(|r| *r.end() < d_lo) {
            b_ranges.next();
        }

        let mut pos = d_lo;
        while pos <= d_hi {
            let (in_a, a_end) = match a_ranges.peek() {
                Some(r) if *r.start() <= pos => (true, *r.end()),
                Some(r) => (false, r.start() - 1),
                None => (false, d_hi),
            };
            let (in_b, b_end) = match b_ranges.peek() {
                Some(r) if *r.start() <= pos => (true, *r.end()),
                Some(r) => (false, r.start() - 1),
                None => (false, d_hi),
            };

            if in_a != in_b {
                return false;
            }

            let sub_end = d_hi.min(a_end).min(b_end);
            if sub_end == u32::MAX {
                break;
            }
            pos = sub_end + 1;
            while a_ranges.peek().is_some_and(|r| *r.end() < pos) {
                a_ranges.next();
            }
            while b_ranges.peek().is_some_and(|r| *r.end() < pos) {
                b_ranges.next();
            }
        }

        let left_ended = *left_range.end() == d_hi;
        let right_ended = *right_range.end() == d_hi;
        if left_ended {
            left_ranges.next();
        }
        if right_ended {
            right_ranges.next();
        }
    }
}

fn weight_is_disjoint_from_domain_intersection(
    weight: &Weight,
    left_domain: &Weight,
    right_domain: &Weight,
) -> bool {
    let mut weight_iter = weight.raw_range_values();
    let mut weight_entry = weight_iter.next();
    let mut left_iter = left_domain.raw_range_values();
    let mut right_iter = right_domain.raw_range_values();
    let mut left_entry = left_iter.next();
    let mut right_entry = right_iter.next();

    loop {
        while let (Some((left_range, _)), Some((right_range, _))) = (&left_entry, &right_entry) {
            if *left_range.end() < *right_range.start() {
                left_entry = left_iter.next();
            } else if *right_range.end() < *left_range.start() {
                right_entry = right_iter.next();
            } else {
                break;
            }
        }

        let (Some((left_range, left_tokens)), Some((right_range, right_tokens))) =
            (&left_entry, &right_entry)
        else {
            return true;
        };

        let d_lo = (*left_range.start()).max(*right_range.start());
        let d_hi = (*left_range.end()).min(*right_range.end());
        while weight_entry.as_ref().is_some_and(|(r, _)| *r.end() < d_lo) {
            weight_entry = weight_iter.next();
        }

        let mut pos = d_lo;
        while pos <= d_hi {
            let (weight_tokens, weight_end): (Option<&Arc<RangeSetBlaze<u32>>>, u32) = match &weight_entry {
                Some((r, tokens)) if *r.start() <= pos => (Some(tokens), *r.end()),
                Some((r, _)) => (None, r.start() - 1),
                None => (None, d_hi),
            };

            let sub_end = d_hi.min(weight_end);
            if let Some(weight_tokens) = weight_tokens {
                if token_sets_intersect_three(
                    weight_tokens.as_ref(),
                    left_tokens.as_ref(),
                    right_tokens.as_ref(),
                ) {
                    return false;
                }
            }

            if sub_end == u32::MAX {
                break;
            }
            pos = sub_end + 1;
            while weight_entry.as_ref().is_some_and(|(r, _)| *r.end() < pos) {
                weight_entry = weight_iter.next();
            }
        }

        let left_ended = *left_range.end() == d_hi;
        let right_ended = *right_range.end() == d_hi;
        if left_ended {
            left_entry = left_iter.next();
        }
        if right_ended {
            right_entry = right_iter.next();
        }
    }
}

fn weights_equal_on_domain_intersection(
    w_a: &Weight,
    w_b: &Weight,
    left_domain: &Weight,
    right_domain: &Weight,
) -> bool {
    if w_a.storage_ptr_eq(w_b) {
        return true;
    }

    let mut a_iter = w_a.raw_range_values();
    let mut b_iter = w_b.raw_range_values();
    let mut left_iter = left_domain.raw_range_values();
    let mut right_iter = right_domain.raw_range_values();
    let mut a_entry = a_iter.next();
    let mut b_entry = b_iter.next();
    let mut left_entry = left_iter.next();
    let mut right_entry = right_iter.next();

    loop {
        while let (Some((left_range, _)), Some((right_range, _))) = (&left_entry, &right_entry) {
            if *left_range.end() < *right_range.start() {
                left_entry = left_iter.next();
            } else if *right_range.end() < *left_range.start() {
                right_entry = right_iter.next();
            } else {
                break;
            }
        }

        let (Some((left_range, left_tokens)), Some((right_range, right_tokens))) =
            (&left_entry, &right_entry)
        else {
            return true;
        };

        if left_tokens.as_ref().is_disjoint(right_tokens.as_ref()) {
            let left_ended = *left_range.end() <= *right_range.end();
            let right_ended = *right_range.end() <= *left_range.end();
            if left_ended {
                left_entry = left_iter.next();
            }
            if right_ended {
                right_entry = right_iter.next();
            }
            continue;
        }

        let d_lo = (*left_range.start()).max(*right_range.start());
        let d_hi = (*left_range.end()).min(*right_range.end());
        while a_entry.as_ref().is_some_and(|(r, _)| *r.end() < d_lo) {
            a_entry = a_iter.next();
        }
        while b_entry.as_ref().is_some_and(|(r, _)| *r.end() < d_lo) {
            b_entry = b_iter.next();
        }

        let mut pos = d_lo;
        while pos <= d_hi {
            let (a_tokens, a_bound): (Option<&Arc<RangeSetBlaze<u32>>>, u32) = match &a_entry {
                Some((r, tokens)) if *r.start() <= pos => (Some(tokens), *r.end()),
                Some((r, _)) => (None, r.start() - 1),
                None => (None, d_hi),
            };
            let (b_tokens, b_bound): (Option<&Arc<RangeSetBlaze<u32>>>, u32) = match &b_entry {
                Some((r, tokens)) if *r.start() <= pos => (Some(tokens), *r.end()),
                Some((r, _)) => (None, r.start() - 1),
                None => (None, d_hi),
            };

            let sub_end = d_hi.min(a_bound).min(b_bound);
            match (a_tokens, b_tokens) {
                (None, None) => {}
                (Some(at), None) => {
                    if token_sets_intersect_three(
                        at.as_ref(),
                        left_tokens.as_ref(),
                        right_tokens.as_ref(),
                    ) {
                        return false;
                    }
                }
                (None, Some(bt)) => {
                    if token_sets_intersect_three(
                        bt.as_ref(),
                        left_tokens.as_ref(),
                        right_tokens.as_ref(),
                    ) {
                        return false;
                    }
                }
                (Some(at), Some(bt)) => {
                    if !Arc::ptr_eq(at, bt) && at.as_ref() != bt.as_ref() {
                        if !token_sets_agree_on_domain_intersection(
                            at.as_ref(),
                            bt.as_ref(),
                            left_tokens.as_ref(),
                            right_tokens.as_ref(),
                        ) {
                            return false;
                        }
                    }
                }
            }

            if sub_end == u32::MAX {
                break;
            }
            pos = sub_end + 1;
            while a_entry.as_ref().is_some_and(|(r, _)| *r.end() < pos) {
                a_entry = a_iter.next();
            }
            while b_entry.as_ref().is_some_and(|(r, _)| *r.end() < pos) {
                b_entry = b_iter.next();
            }
        }

        let left_ended = *left_range.end() == d_hi;
        let right_ended = *right_range.end() == d_hi;
        if left_ended {
            left_entry = left_iter.next();
        }
        if right_ended {
            right_entry = right_iter.next();
        }
    }
}

/// Check if two candidate states can be merged.
///
/// States are compatible if:
/// 1. Their needed sets don't overlap, OR
/// 2. On the overlapping domain, they have identical final weights and
///    identical transition targets (after remapping through old_to_new).
/// 3. Even when disjoint, transitions on the same label must go to the same
///    target (since the DWA can only store one target per label).
fn are_compatible(
    u: usize,
    v: usize,
    dwa: &DWA,
    needed: &[Weight],
    old_to_new: &[u32],
    productive_transitions: &[Vec<ProductiveTransition>],
    known_overlapping: bool,
) -> bool {
    let needed_u = &needed[u];
    let needed_v = &needed[v];

    let domain_disjoint = if known_overlapping {
        false
    } else {
        needed_u.is_disjoint(needed_v)
    };

    // Check transitions — do target conflict detection first (cheap).
    let n = dwa.states().len();
    let trans_u = &productive_transitions[u];
    let trans_v = &productive_transitions[v];

    // Quick target-conflict check first (no weight ops needed)
    {
        let mut idx_u = 0usize;
        let mut idx_v = 0usize;
        while idx_u < trans_u.len() || idx_v < trans_v.len() {
            let (entry_u, entry_v) = match (trans_u.get(idx_u), trans_v.get(idx_v)) {
                (Some(u_entry), Some(v_entry)) => {
                    if u_entry.label == v_entry.label {
                        idx_u += 1;
                        idx_v += 1;
                        (Some(u_entry), Some(v_entry))
                    } else if u_entry.label < v_entry.label {
                        idx_u += 1;
                        (Some(u_entry), None)
                    } else {
                        idx_v += 1;
                        (None, Some(v_entry))
                    }
                }
                (Some(u_entry), None) => { idx_u += 1; (Some(u_entry), None) }
                (None, Some(v_entry)) => { idx_v += 1; (None, Some(v_entry)) }
                (None, None) => break,
            };

            let target_u = entry_u.and_then(|e| ((e.target as usize) < n).then_some(e.target));
            let target_v = entry_v.and_then(|e| ((e.target as usize) < n).then_some(e.target));
            let mapped_u = target_u.and_then(|t| mapped_target(old_to_new, t));
            let mapped_v = target_v.and_then(|t| mapped_target(old_to_new, t));
            match (mapped_u, mapped_v) {
                (Some(mu), Some(mv)) if mu != mv => {
                    let has_u = entry_u.is_some_and(|e| !e.weight.is_empty());
                    let has_v = entry_v.is_some_and(|e| !e.weight.is_empty());
                    if has_u || has_v {
                        return false;
                    }
                }
                _ => {}
            }
        }
    }

    // If domains are disjoint we're done — no overlap weight checks needed.
    if domain_disjoint {
        return true;
    }

    // Check if all transition weights are identical (Arc-equal or value-equal).
    // This is a fast path that avoids computing the overlap weight entirely.
    {
        let mut all_equal = true;
        let fw_u = dwa.states()[u].final_weight.as_ref();
        let fw_v = dwa.states()[v].final_weight.as_ref();
        match (fw_u, fw_v) {
            (Some(wu), Some(wv)) if wu == wv => {}
            (None, None) => {}
            _ => { all_equal = false; }
        }
        if all_equal {
            let mut idx_u = 0usize;
            let mut idx_v = 0usize;
            while idx_u < trans_u.len() || idx_v < trans_v.len() {
                let (eu, ev) = match (trans_u.get(idx_u), trans_v.get(idx_v)) {
                    (Some(a), Some(b)) => {
                        if a.label == b.label { idx_u += 1; idx_v += 1; (Some(a), Some(b)) }
                        else if a.label < b.label { idx_u += 1; (Some(a), None) }
                        else { idx_v += 1; (None, Some(b)) }
                    }
                    (Some(a), None) => { idx_u += 1; (Some(a), None) }
                    (None, Some(b)) => { idx_v += 1; (None, Some(b)) }
                    (None, None) => break,
                };
                let w_u = eu.map(|e| &e.weight);
                let w_v = ev.map(|e| &e.weight);
                match (w_u, w_v) {
                    (Some(wu), Some(wv)) if wu == wv => {}
                    (None, None) => {}
                    _ => { all_equal = false; break; }
                }
            }
        }
        if all_equal {
            return true;
        }
    }

    // Slow path: compute overlap and check weight equality on the intersection
    let overlap = needed_u.intersection(needed_v);

    // Check final weights on the overlapping domain
    {
        let fw_u = dwa.states()[u].final_weight.as_ref();
        let fw_v = dwa.states()[v].final_weight.as_ref();
        match (fw_u, fw_v) {
            (Some(wu), Some(wv)) => {
                if !weights_equal_on_domain(wu, wv, &overlap) {
                    return false;
                }
            }
            (Some(fw), None) | (None, Some(fw)) => {
                if !fw.is_disjoint(&overlap) {
                    return false;
                }
            }
            (None, None) => {}
        }
    }

    // Slow path: check overlap weights per transition label.
    // Target conflicts were already caught in pass 1 above.
    {
        let mut idx_u = 0usize;
        let mut idx_v = 0usize;
        while idx_u < trans_u.len() || idx_v < trans_v.len() {
            let (entry_u, entry_v) = match (trans_u.get(idx_u), trans_v.get(idx_v)) {
                (Some(a), Some(b)) => {
                    if a.label == b.label { idx_u += 1; idx_v += 1; (Some(a), Some(b)) }
                    else if a.label < b.label { idx_u += 1; (Some(a), None) }
                    else { idx_v += 1; (None, Some(b)) }
                }
                (Some(a), None) => { idx_u += 1; (Some(a), None) }
                (None, Some(b)) => { idx_v += 1; (None, Some(b)) }
                (None, None) => break,
            };

            let w_u_full = entry_u.and_then(|e| ((e.target as usize) < n).then_some(&e.weight));
            let w_v_full = entry_v.and_then(|e| ((e.target as usize) < n).then_some(&e.weight));

            // Fast path: if both full weights are equal, overlap restrictions are too.
            match (w_u_full, w_v_full) {
                (Some(wu), Some(wv)) if wu == wv => continue,
                (None, None) => continue,
                _ => {}
            }

            let u_disjoint = w_u_full.map_or(true, |w| w.is_disjoint(&overlap));
            let v_disjoint = w_v_full.map_or(true, |w| w.is_disjoint(&overlap));

            if u_disjoint && v_disjoint {
                continue;
            }
            if u_disjoint != v_disjoint {
                return false;
            }

            // Both non-empty on overlap → check equality.
            if !weights_equal_on_domain(w_u_full.unwrap(), w_v_full.unwrap(), &overlap) {
                return false;
            }

            // Equal on overlap → targets must agree (re-check for this specific case).
            let target_u = entry_u.and_then(|e| ((e.target as usize) < n).then_some(e.target));
            let target_v = entry_v.and_then(|e| ((e.target as usize) < n).then_some(e.target));
            let mapped_u = target_u.and_then(|t| mapped_target(old_to_new, t));
            let mapped_v = target_v.and_then(|t| mapped_target(old_to_new, t));
            match (mapped_u, mapped_v) {
                (Some(mu), Some(mv)) if mu != mv => return false,
                (Some(_), None) | (None, Some(_)) => return false,
                _ => {}
            }
        }
    }

    true
}

#[derive(Clone)]
struct ClassProfile {
    targets: Vec<(Label, u32)>,
    weights: Vec<(Label, Weight)>,
    final_weight: Option<Weight>,
}

fn build_class_profile(
    rep: usize,
    old_to_new: &[u32],
    productive_transitions: &[Vec<ProductiveTransition>],
    dwa: &DWA,
) -> ClassProfile {
    let mut targets = Vec::with_capacity(productive_transitions[rep].len());
    let mut weights = Vec::with_capacity(productive_transitions[rep].len());
    for pt in &productive_transitions[rep] {
        if let Some(mapped) = mapped_target(old_to_new, pt.target) {
            targets.push((pt.label, mapped));
        }
        weights.push((pt.label, pt.weight.clone()));
    }
    targets.sort_unstable_by_key(|(label, _)| *label);
    weights.sort_unstable_by_key(|(label, _)| *label);
    ClassProfile {
        targets,
        weights,
        final_weight: dwa.states()[rep].final_weight.clone(),
    }
}

/// Sparse pointwise behavior used to compare one group against a candidate
/// without rescanning all overlapping members.
#[derive(Clone, Eq, PartialEq, Hash)]
struct PointwiseBehavior {
    final_active: bool,
    transitions: Vec<(Label, u32)>,
}

#[derive(Default)]
struct PointwiseBehaviorInterner {
    ids: FxHashMap<PointwiseBehavior, u32>,
    values: Vec<PointwiseBehavior>,
}

impl PointwiseBehaviorInterner {
    fn intern(&mut self, final_active: bool, transitions: Vec<(Label, u32)>) -> u32 {
        let behavior = PointwiseBehavior { final_active, transitions };
        if let Some(&id) = self.ids.get(&behavior) {
            return id;
        }
        let id = self.ids.len() as u32;
        self.values.push(behavior.clone());
        self.ids.insert(behavior, id);
        id
    }

    fn get(&self, id: u32) -> &PointwiseBehavior {
        &self.values[id as usize]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct TokenBehaviorRange {
    start: u32,
    end: u32,
    behavior: u32,
}

#[derive(Clone, Eq, PartialEq, Hash)]
struct PointwiseRegionBuildKey {
    domain_tokens: usize,
    final_tokens: usize,
    transitions: SmallVec<[(Label, u32, usize); 4]>,
}

#[derive(Default)]
struct PointwiseRegionBuildCache {
    entries: FxHashMap<PointwiseRegionBuildKey, Option<Arc<Vec<TokenBehaviorRange>>>>,
    hits: usize,
    misses: usize,
}

struct DirectOverlayCache {
    region_values: Vec<Arc<Vec<TokenBehaviorRange>>>,
    region_ids_by_ptr: FxHashMap<usize, u32>,
    slots: Vec<Option<(u64, u32)>>,
    hits: usize,
    misses: usize,
    replacements: usize,
}

struct PointwiseRegionInterner {
    // Keep the immutable region itself as the hash key. `Arc<Vec<_>>` hashes
    // and compares by its contents, so lookup can still borrow a fresh Vec,
    // but newly interned regions no longer need a second full Vec clone solely
    // for map ownership.
    regions: FxHashMap<Arc<Vec<TokenBehaviorRange>>, ()>,
    direct_overlay: Option<DirectOverlayCache>,
}

impl Default for PointwiseRegionInterner {
    fn default() -> Self {
        let requested_slots = std::env::var("GLRMASK_WEIGHTED_MINIMIZE_DIRECT_OVERLAY_SLOTS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(0);
        let slot_count = if requested_slots == 0 {
            0
        } else {
            requested_slots.next_power_of_two()
        };
        Self::with_direct_overlay_slots(slot_count)
    }
}

impl PointwiseRegionInterner {
    fn with_direct_overlay_slots(slot_count: usize) -> Self {
        debug_assert!(slot_count == 0 || slot_count.is_power_of_two());
        Self {
            regions: FxHashMap::default(),
            direct_overlay: (slot_count != 0).then(|| DirectOverlayCache {
                region_values: Vec::new(),
                region_ids_by_ptr: FxHashMap::default(),
                slots: vec![None; slot_count],
                hits: 0,
                misses: 0,
                replacements: 0,
            }),
        }
    }

    fn intern(&mut self, ranges: Vec<TokenBehaviorRange>) -> Arc<Vec<TokenBehaviorRange>> {
        if let Some((existing, _)) = self.regions.get_key_value(&ranges) {
            return Arc::clone(existing);
        }
        let ranges = Arc::new(ranges);
        self.regions.insert(Arc::clone(&ranges), ());
        if let Some(cache) = self.direct_overlay.as_mut() {
            let id = cache.region_values.len() as u32;
            cache
                .region_ids_by_ptr
                .insert(Arc::as_ptr(&ranges) as usize, id);
            cache.region_values.push(Arc::clone(&ranges));
        }
        ranges
    }

    fn overlay_compatible(
        &mut self,
        left: &Arc<Vec<TokenBehaviorRange>>,
        right: &Arc<Vec<TokenBehaviorRange>>,
    ) -> Arc<Vec<TokenBehaviorRange>> {
        if Arc::ptr_eq(left, right) {
            return Arc::clone(left);
        }
        let Some(cache) = self.direct_overlay.as_mut() else {
            return self.intern(overlay_compatible_token_behavior_ranges(
                left.as_ref(),
                right.as_ref(),
            ));
        };

        let left_id = *cache
            .region_ids_by_ptr
            .get(&(Arc::as_ptr(left) as usize))
            .expect("interned pointwise region must have a compact ID");
        let right_id = *cache
            .region_ids_by_ptr
            .get(&(Arc::as_ptr(right) as usize))
            .expect("interned pointwise region must have a compact ID");
        let (low, high) = if left_id <= right_id {
            (left_id, right_id)
        } else {
            (right_id, left_id)
        };
        let key = (u64::from(low) << 32) | u64::from(high);
        let mixed = key
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(23)
            ^ key.rotate_right(17);
        let slot_index = mixed as usize & (cache.slots.len() - 1);
        if let Some((cached_key, result_id)) = cache.slots[slot_index] {
            if cached_key == key {
                cache.hits += 1;
                return Arc::clone(&cache.region_values[result_id as usize]);
            }
            cache.replacements += 1;
        }

        cache.misses += 1;
        let result = self.intern(overlay_compatible_token_behavior_ranges(
            left.as_ref(),
            right.as_ref(),
        ));
        let cache = self
            .direct_overlay
            .as_mut()
            .expect("direct overlay cache must remain enabled");
        let result_id = *cache
            .region_ids_by_ptr
            .get(&(Arc::as_ptr(&result) as usize))
            .expect("newly interned pointwise region must have a compact ID");
        cache.slots[slot_index] = Some((key, result_id));
        result
    }

    fn direct_overlay_stats(&self) -> (usize, usize, usize, usize) {
        self.direct_overlay.as_ref().map_or((0, 0, 0, 0), |cache| {
            (
                cache.slots.len(),
                cache.hits,
                cache.misses,
                cache.replacements,
            )
        })
    }
}

#[derive(Default)]
struct PointwiseProfile {
    /// Sorted by TSID. Each behavior region is immutable and shared by every
    /// TSID whose weight behavior is identical.
    by_tsid: Vec<(u32, Arc<Vec<TokenBehaviorRange>>)>,
}

/// One constant pointwise behavior over an inclusive TSID interval.
///
/// The existing pointwise path expands this interval into one entry per TSID.
/// That is exact but pathological when a weight is constant across thousands
/// of compacted tokenizer states. The range form preserves the same partial
/// behavior function without that expansion.
#[derive(Clone)]
struct PointwiseTsidRange {
    start: u32,
    end: u32,
    region: Arc<Vec<TokenBehaviorRange>>,
}

#[derive(Default)]
struct PointwiseRangeProfile {
    /// Sorted, disjoint TSID intervals. Adjacent intervals with the same
    /// region are coalesced.
    by_tsid_range: Vec<PointwiseTsidRange>,
}

#[derive(Default)]
struct PointwiseRangeBehaviorMap {
    /// Sorted, disjoint TSID intervals encoding the exact union behavior of a
    /// merge group.
    ranges: Vec<PointwiseTsidRange>,
}

#[derive(Default)]
struct PointwiseRangeMergeGroup {
    targets_by_label: FxHashMap<Label, u32>,
    behavior_by_tsid: PointwiseRangeBehaviorMap,
    member_classes: Vec<usize>,
}

fn pointwise_tsid_ranges_enabled() -> bool {
    std::env::var("GLRMASK_POINTWISE_TSID_RANGES")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn pointwise_tsid_ranges_auto_enabled() -> bool {
    std::env::var("GLRMASK_POINTWISE_TSID_RANGES")
        .map(|value| value.trim().eq_ignore_ascii_case("auto"))
        .unwrap_or(false)
}

const POINTWISE_TSID_RANGE_MIN_COMPRESSION: usize = 4;

fn push_pointwise_tsid_range(
    ranges: &mut Vec<PointwiseTsidRange>,
    start: u32,
    end: u32,
    region: Arc<Vec<TokenBehaviorRange>>,
) {
    if start > end {
        return;
    }
    if let Some(previous) = ranges.last_mut() {
        if Arc::ptr_eq(&previous.region, &region)
            && previous.end != u32::MAX
            && previous.end + 1 == start
        {
            previous.end = end;
            return;
        }
    }
    ranges.push(PointwiseTsidRange { start, end, region });
}

const MAX_DENSE_POINTWISE_TSID_SLOTS: usize = 65_536;

#[derive(Clone, Copy)]
enum PointwiseBehaviorMapLayout {
    Sparse,
    Dense { slots: usize },
}

enum PointwiseBehaviorMap {
    Sparse(FxHashMap<u32, Arc<Vec<TokenBehaviorRange>>>),
    Dense(Vec<Option<Arc<Vec<TokenBehaviorRange>>>>),
}

impl Default for PointwiseBehaviorMap {
    fn default() -> Self {
        Self::Sparse(FxHashMap::default())
    }
}

impl PointwiseBehaviorMap {
    fn new(layout: PointwiseBehaviorMapLayout) -> Self {
        match layout {
            PointwiseBehaviorMapLayout::Sparse => Self::Sparse(FxHashMap::default()),
            PointwiseBehaviorMapLayout::Dense { slots } => Self::Dense(vec![None; slots]),
        }
    }

    fn get(&self, tsid: u32) -> Option<&Arc<Vec<TokenBehaviorRange>>> {
        match self {
            Self::Sparse(entries) => entries.get(&tsid),
            Self::Dense(entries) => entries.get(tsid as usize).and_then(Option::as_ref),
        }
    }

    fn merge_profile(
        &mut self,
        profile: &PointwiseProfile,
        regions: &mut PointwiseRegionInterner,
    ) {
        for (tsid, ranges) in &profile.by_tsid {
            match self {
                Self::Sparse(entries) => match entries.get_mut(tsid) {
                    Some(existing) if Arc::ptr_eq(existing, ranges) => {}
                    Some(existing) => {
                        *existing = regions.overlay_compatible(existing, ranges);
                    }
                    None => {
                        entries.insert(*tsid, Arc::clone(ranges));
                    }
                },
                Self::Dense(entries) => {
                    let entry = entries
                        .get_mut(*tsid as usize)
                        .expect("dense pointwise behavior map must cover every profile TSID");
                    match entry {
                        Some(existing) if Arc::ptr_eq(existing, ranges) => {}
                        Some(existing) => {
                            *existing = regions.overlay_compatible(existing, ranges);
                        }
                        None => {
                            *entry = Some(Arc::clone(ranges));
                        }
                    }
                }
            }
        }
    }

    fn region_entry_count(&self) -> usize {
        match self {
            Self::Sparse(entries) => entries.values().map(|ranges| ranges.len()).sum(),
            Self::Dense(entries) => entries
                .iter()
                .flatten()
                .map(|ranges| ranges.len())
                .sum(),
        }
    }
}

impl PointwiseRangeBehaviorMap {
    fn profile_compatible(&self, profile: &PointwiseRangeProfile) -> bool {
        let mut existing_index = 0usize;
        let mut profile_index = 0usize;
        while existing_index < self.ranges.len() && profile_index < profile.by_tsid_range.len() {
            let existing = &self.ranges[existing_index];
            let add = &profile.by_tsid_range[profile_index];
            if existing.end < add.start {
                existing_index += 1;
                continue;
            }
            if add.end < existing.start {
                profile_index += 1;
                continue;
            }
            if !Arc::ptr_eq(&existing.region, &add.region)
                && !token_behavior_ranges_compatible(existing.region.as_ref(), add.region.as_ref())
            {
                return false;
            }
            if existing.end <= add.end {
                existing_index += 1;
            }
            if add.end <= existing.end {
                profile_index += 1;
            }
        }
        true
    }

    fn merge_profile(
        &mut self,
        profile: &PointwiseRangeProfile,
        regions: &mut PointwiseRegionInterner,
    ) {
        let mut merged = Vec::with_capacity(self.ranges.len() + profile.by_tsid_range.len());
        let mut existing_index = 0usize;
        let mut profile_index = 0usize;
        let mut existing = self.ranges.get(existing_index).cloned();
        let mut add = profile.by_tsid_range.get(profile_index).cloned();

        loop {
            match (existing.take(), add.take()) {
                (None, None) => break,
                (Some(range), None) => {
                    push_pointwise_tsid_range(&mut merged, range.start, range.end, range.region);
                    existing_index += 1;
                    existing = self.ranges.get(existing_index).cloned();
                }
                (None, Some(range)) => {
                    push_pointwise_tsid_range(&mut merged, range.start, range.end, range.region);
                    profile_index += 1;
                    add = profile.by_tsid_range.get(profile_index).cloned();
                }
                (Some(mut left), Some(mut right)) => {
                    if left.end < right.start {
                        push_pointwise_tsid_range(&mut merged, left.start, left.end, left.region);
                        existing_index += 1;
                        existing = self.ranges.get(existing_index).cloned();
                        add = Some(right);
                        continue;
                    }
                    if right.end < left.start {
                        push_pointwise_tsid_range(&mut merged, right.start, right.end, right.region);
                        profile_index += 1;
                        existing = Some(left);
                        add = profile.by_tsid_range.get(profile_index).cloned();
                        continue;
                    }

                    if left.start < right.start {
                        push_pointwise_tsid_range(
                            &mut merged,
                            left.start,
                            right.start - 1,
                            Arc::clone(&left.region),
                        );
                        left.start = right.start;
                    } else if right.start < left.start {
                        push_pointwise_tsid_range(
                            &mut merged,
                            right.start,
                            left.start - 1,
                            Arc::clone(&right.region),
                        );
                        right.start = left.start;
                    }

                    debug_assert!(
                        Arc::ptr_eq(&left.region, &right.region)
                            || token_behavior_ranges_compatible(
                                left.region.as_ref(),
                                right.region.as_ref(),
                            ),
                        "range profiles must be checked for compatibility before merging",
                    );
                    let end = left.end.min(right.end);
                    let region = if Arc::ptr_eq(&left.region, &right.region) {
                        left.region.clone()
                    } else {
                        regions.overlay_compatible(&left.region, &right.region)
                    };
                    push_pointwise_tsid_range(&mut merged, left.start, end, region);

                    if left.end == end {
                        existing_index += 1;
                        existing = self.ranges.get(existing_index).cloned();
                    } else {
                        left.start = end + 1;
                        existing = Some(left);
                    }
                    if right.end == end {
                        profile_index += 1;
                        add = profile.by_tsid_range.get(profile_index).cloned();
                    } else {
                        right.start = end + 1;
                        add = Some(right);
                    }
                }
            }
        }
        self.ranges = merged;
    }

    fn region_entry_count(&self) -> usize {
        self.ranges.iter().map(|range| range.region.len()).sum()
    }
}

#[derive(Default)]
struct PointwiseMergeGroup {
    targets_by_label: FxHashMap<Label, u32>,
    /// Exact partial behavior function of all members already in this group.
    behavior_by_tsid: PointwiseBehaviorMap,
    member_classes: Vec<usize>,
}

fn range_set_contains(tokens: &RangeSetBlaze<u32>, value: u32) -> bool {
    tokens.contains(value)
}

fn profile_target_for_label(profile: &ClassProfile, label: Label) -> Option<u32> {
    profile
        .targets
        .binary_search_by_key(&label, |(candidate, _)| *candidate)
        .ok()
        .map(|index| profile.targets[index].1)
}

fn push_token_behavior_range(
    ranges: &mut Vec<TokenBehaviorRange>,
    start: u32,
    end: u32,
    behavior: u32,
) {
    if start > end {
        return;
    }
    if let Some(previous) = ranges.last_mut() {
        if previous.behavior == behavior
            && previous.end != u32::MAX
            && previous.end + 1 == start
        {
            previous.end = end;
            return;
        }
    }
    ranges.push(TokenBehaviorRange { start, end, behavior });
}

fn add_tsid_boundary_if_overlapping(
    boundaries: &mut Vec<u64>,
    domain_start: u32,
    domain_end: u32,
    range_start: u32,
    range_end: u32,
) {
    let start = domain_start.max(range_start);
    let end = domain_end.min(range_end);
    if start <= end {
        boundaries.push(u64::from(start));
        boundaries.push(u64::from(end) + 1);
    }
}

fn build_token_behavior_region(
    domain_tokens: &RangeSetBlaze<u32>,
    final_tokens: Option<&RangeSetBlaze<u32>>,
    active_transitions: &[(Label, u32, &RangeSetBlaze<u32>)],
    behaviors: &mut PointwiseBehaviorInterner,
    regions: &mut PointwiseRegionInterner,
    build_cache: &mut PointwiseRegionBuildCache,
) -> Option<Arc<Vec<TokenBehaviorRange>>> {
    // Token sets are immutable and retained by the DWA/needed weights for this
    // whole coloring pass, so their addresses form an exact local cache key.
    let key = PointwiseRegionBuildKey {
        domain_tokens: domain_tokens as *const RangeSetBlaze<u32> as usize,
        final_tokens: final_tokens
            .map(|tokens| tokens as *const RangeSetBlaze<u32> as usize)
            .unwrap_or(0),
        transitions: active_transitions
            .iter()
            .map(|(label, target, tokens)| {
                (*label, *target, *tokens as *const RangeSetBlaze<u32> as usize)
            })
            .collect(),
    };
    if let Some(existing) = build_cache.entries.get(&key) {
        build_cache.hits += 1;
        return existing.clone();
    }
    build_cache.misses += 1;
    let mut boundaries = Vec::<u64>::new();
    for range in domain_tokens.ranges() {
        boundaries.push(u64::from(*range.start()));
        boundaries.push(u64::from(*range.end()) + 1);
    }
    if let Some(tokens) = final_tokens {
        for range in tokens.ranges() {
            boundaries.push(u64::from(*range.start()));
            boundaries.push(u64::from(*range.end()) + 1);
        }
    }
    for (_, _, tokens) in active_transitions {
        for range in tokens.ranges() {
            boundaries.push(u64::from(*range.start()));
            boundaries.push(u64::from(*range.end()) + 1);
        }
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    let mut token_ranges = Vec::new();
    for pair in boundaries.windows(2) {
        let start64 = pair[0];
        let next = pair[1];
        if start64 > u64::from(u32::MAX) || start64 >= next {
            continue;
        }
        let start = start64 as u32;
        let end = (next - 1) as u32;
        if !range_set_contains(domain_tokens, start) {
            continue;
        }
        let final_active = final_tokens.is_some_and(|tokens| range_set_contains(tokens, start));
        let mut transitions = Vec::new();
        for (label, target, tokens) in active_transitions {
            if range_set_contains(tokens, start) {
                transitions.push((*label, *target));
            }
        }
        debug_assert!(final_active || !transitions.is_empty());
        if !final_active && transitions.is_empty() {
            return None;
        }
        let behavior = behaviors.intern(final_active, transitions);
        push_token_behavior_range(&mut token_ranges, start, end, behavior);
    }
    let result = (!token_ranges.is_empty()).then(|| regions.intern(token_ranges));
    build_cache.entries.insert(key, result.clone());
    result
}

/// Materialize a complete observable behavior function. A profile is constant
/// over large TSID intervals, so build its token behavior once per interval
/// then share that immutable region for every TSID in the interval.
fn build_pointwise_profile(
    domain: &Weight,
    profile: &ClassProfile,
    behaviors: &mut PointwiseBehaviorInterner,
    regions: &mut PointwiseRegionInterner,
    build_cache: &mut PointwiseRegionBuildCache,
) -> Option<PointwiseProfile> {
    if domain.is_full()
        || profile.final_weight.as_ref().is_some_and(Weight::is_full)
        || profile.weights.iter().any(|(_, weight)| weight.is_full())
    {
        return None;
    }

    let transitions: Vec<(Label, u32, &Weight)> = profile
        .weights
        .iter()
        .map(|(label, weight)| Some((*label, profile_target_for_label(profile, *label)?, weight)))
        .collect::<Option<_>>()?;

    // The domain is fragmented into many small compact windows. Collect each
    // source's full sorted TSID range list ONCE up front; re-scanning every
    // transition's complete range list per window would be O(windows * ranges).
    // Monotone cursors below then advance forward as windows advance, making the
    // per-window range gathering O(ranges + overlaps) overall.
    let final_full: Vec<(u32, u32, &RangeSetBlaze<u32>)> = profile
        .final_weight
        .as_ref()
        .map(|fw| {
            fw.raw_range_values()
                .map(|(r, tokens)| (*r.start(), *r.end(), tokens.as_ref()))
                .collect()
        })
        .unwrap_or_default();
    let transition_full: Vec<Vec<(u32, u32, &RangeSetBlaze<u32>)>> = transitions
        .iter()
        .map(|(_, _, weight)| {
            weight
                .raw_range_values()
                .map(|(r, tokens)| (*r.start(), *r.end(), tokens.as_ref()))
                .collect()
        })
        .collect();

    let mut by_tsid = Vec::new();
    let mut final_scan = 0usize;
    let mut transition_scan = vec![0usize; transitions.len()];
    for (domain_start, domain_end, domain_tokens) in domain.compact_entries()? {
        // Build the TSID boundaries and, in the same pass, collect each source's
        // token-set ranges clipped to the domain window. The clipped range lists
        // let a coordinated sweep read the active token set per interval with a
        // monotone cursor instead of a binary-search `get` per interval per
        // transition, which is the dominant color-step cost.
        let mut boundaries = vec![u64::from(domain_start), u64::from(domain_end) + 1];
        let mut final_ranges: Vec<(u32, u32, &RangeSetBlaze<u32>)> = Vec::new();
        // Advance the monotone scan cursor past ranges ending before this window;
        // windows are visited left to right so earlier ranges are never revisited.
        while final_scan < final_full.len() && final_full[final_scan].1 < domain_start {
            final_scan += 1;
        }
        let mut peek = final_scan;
        while peek < final_full.len() && final_full[peek].0 <= domain_end {
            let (range_start, range_end, tokens) = final_full[peek];
            let start = domain_start.max(range_start);
            let end = domain_end.min(range_end);
            if start <= end {
                boundaries.push(u64::from(start));
                boundaries.push(u64::from(end) + 1);
                final_ranges.push((start, end, tokens));
            }
            peek += 1;
        }
        let mut transition_ranges: Vec<Vec<(u32, u32, &RangeSetBlaze<u32>)>> =
            Vec::with_capacity(transitions.len());
        for (index, full) in transition_full.iter().enumerate() {
            let cursor = &mut transition_scan[index];
            while *cursor < full.len() && full[*cursor].1 < domain_start {
                *cursor += 1;
            }
            let mut clipped: Vec<(u32, u32, &RangeSetBlaze<u32>)> = Vec::new();
            let mut peek = *cursor;
            while peek < full.len() && full[peek].0 <= domain_end {
                let (range_start, range_end, tokens) = full[peek];
                let start = domain_start.max(range_start);
                let end = domain_end.min(range_end);
                if start <= end {
                    boundaries.push(u64::from(start));
                    boundaries.push(u64::from(end) + 1);
                    clipped.push((start, end, tokens));
                }
                peek += 1;
            }
            transition_ranges.push(clipped);
        }
        boundaries.sort_unstable();
        boundaries.dedup();

        // Only transitions with at least one clipped range in this window can
        // ever be active in an interval; skipping the empties keeps the sweep
        // inner loop proportional to the live transitions rather than the full
        // transition count. Indices stay ascending so push order is unchanged.
        let active_indices: Vec<usize> = transition_ranges
            .iter()
            .enumerate()
            .filter(|(_, ranges)| !ranges.is_empty())
            .map(|(index, _)| index)
            .collect();

        // Monotone cursors: TSID intervals are visited left to right, and every
        // clipped range list is sorted, so each cursor only advances forward.
        let mut final_cursor = 0usize;
        let mut transition_cursors = vec![0usize; transitions.len()];
        for pair in boundaries.windows(2) {
            let start64 = pair[0];
            let next = pair[1];
            if start64 > u64::from(u32::MAX) || start64 >= next {
                continue;
            }
            let tsid_start = start64 as u32;
            let tsid_end = (next - 1) as u32;
            while final_cursor < final_ranges.len() && final_ranges[final_cursor].1 < tsid_start {
                final_cursor += 1;
            }
            let final_tokens = final_ranges
                .get(final_cursor)
                .filter(|(start, _, _)| *start <= tsid_start)
                .map(|(_, _, tokens)| *tokens);
            let mut active_transitions = Vec::new();
            for &index in &active_indices {
                let (label, target, _) = &transitions[index];
                let ranges = &transition_ranges[index];
                let cursor = &mut transition_cursors[index];
                while *cursor < ranges.len() && ranges[*cursor].1 < tsid_start {
                    *cursor += 1;
                }
                if let Some((start, _, tokens)) = ranges.get(*cursor) {
                    if *start <= tsid_start {
                        active_transitions.push((*label, *target, *tokens));
                    }
                }
            }
            let region = build_token_behavior_region(
                domain_tokens.as_ref(),
                final_tokens,
                &active_transitions,
                behaviors,
                regions,
                build_cache,
            )?;
            for tsid in tsid_start..=tsid_end {
                by_tsid.push((tsid, Arc::clone(&region)));
            }
        }
    }
    Some(PointwiseProfile { by_tsid })
}

struct PointwiseProfileBuildOutput {
    profiles: Vec<PointwiseProfile>,
    behaviors: PointwiseBehaviorInterner,
    regions: PointwiseRegionInterner,
    cache_entries: usize,
    cache_hits: usize,
    cache_misses: usize,
    parallel_chunks: usize,
}

struct LocalPointwiseProfileBuild {
    profiles: Vec<PointwiseProfile>,
    behaviors: Vec<PointwiseBehavior>,
    cache_entries: usize,
    cache_hits: usize,
    cache_misses: usize,
}

fn build_pointwise_profiles_serial(
    class_needed_union: &[Weight],
    class_profiles: &[ClassProfile],
) -> Option<PointwiseProfileBuildOutput> {
    let mut behaviors = PointwiseBehaviorInterner::default();
    let mut regions = PointwiseRegionInterner::default();
    let mut build_cache = PointwiseRegionBuildCache::default();
    let mut profiles = Vec::with_capacity(class_profiles.len());
    for (domain, profile) in class_needed_union.iter().zip(class_profiles) {
        profiles.push(build_pointwise_profile(
            domain,
            profile,
            &mut behaviors,
            &mut regions,
            &mut build_cache,
        )?);
    }
    Some(PointwiseProfileBuildOutput {
        profiles,
        behaviors,
        regions,
        cache_entries: build_cache.entries.len(),
        cache_hits: build_cache.hits,
        cache_misses: build_cache.misses,
        parallel_chunks: 1,
    })
}

/// Build exact pointwise profiles in deterministic contiguous chunks.
///
/// Each chunk owns its behavior/region interners, so profile construction is
/// embarrassingly parallel. The merge phase maps every local behavior value to
/// one global ID and rewrites every local immutable region through that map.
/// Thus each resulting `(tsid, token) -> behavior value` function is identical
/// to serial construction; only internal IDs and allocation order differ.
fn build_pointwise_profiles_parallel(
    class_needed_union: &[Weight],
    class_profiles: &[ClassProfile],
) -> Option<PointwiseProfileBuildOutput> {
    debug_assert_eq!(class_needed_union.len(), class_profiles.len());
    if class_profiles.is_empty() {
        return Some(PointwiseProfileBuildOutput {
            profiles: Vec::new(),
            behaviors: PointwiseBehaviorInterner::default(),
            regions: PointwiseRegionInterner::default(),
            cache_entries: 0,
            cache_hits: 0,
            cache_misses: 0,
            parallel_chunks: 0,
        });
    }

    let workers = rayon::current_num_threads().max(1);
    let desired_chunks = workers.min(class_profiles.len().div_ceil(32)).max(1);
    if desired_chunks == 1 {
        return build_pointwise_profiles_serial(class_needed_union, class_profiles);
    }
    let chunk_size = class_profiles.len().div_ceil(desired_chunks);
    let local_chunks = class_needed_union
        .par_chunks(chunk_size)
        .zip(class_profiles.par_chunks(chunk_size))
        .map(|(domains, profiles)| {
            let mut behaviors = PointwiseBehaviorInterner::default();
            let mut regions = PointwiseRegionInterner::default();
            let mut build_cache = PointwiseRegionBuildCache::default();
            let profiles = domains
                .iter()
                .zip(profiles)
                .map(|(domain, profile)| {
                    build_pointwise_profile(
                        domain,
                        profile,
                        &mut behaviors,
                        &mut regions,
                        &mut build_cache,
                    )
                })
                .collect::<Option<Vec<_>>>()?;
            Some(LocalPointwiseProfileBuild {
                profiles,
                behaviors: behaviors.values,
                cache_entries: build_cache.entries.len(),
                cache_hits: build_cache.hits,
                cache_misses: build_cache.misses,
            })
        })
        .collect::<Vec<_>>();
    if local_chunks.iter().any(Option::is_none) {
        return None;
    }

    let mut global_behaviors = PointwiseBehaviorInterner::default();
    let mut global_regions = PointwiseRegionInterner::default();
    let mut merged_profiles = Vec::with_capacity(class_profiles.len());
    let mut cache_entries = 0usize;
    let mut cache_hits = 0usize;
    let mut cache_misses = 0usize;
    let parallel_chunks = local_chunks.len();

    for local in local_chunks.into_iter().map(Option::unwrap) {
        cache_entries += local.cache_entries;
        cache_hits += local.cache_hits;
        cache_misses += local.cache_misses;
        let local_to_global = local
            .behaviors
            .iter()
            .map(|behavior| {
                global_behaviors.intern(
                    behavior.final_active,
                    behavior.transitions.clone(),
                )
            })
            .collect::<Vec<_>>();
        let mut remapped_regions = FxHashMap::<usize, Arc<Vec<TokenBehaviorRange>>>::default();
        for profile in local.profiles {
            let mut by_tsid = Vec::with_capacity(profile.by_tsid.len());
            for (tsid, local_region) in profile.by_tsid {
                let key = Arc::as_ptr(&local_region) as usize;
                let global_region = remapped_regions
                    .entry(key)
                    .or_insert_with(|| {
                        let remapped = local_region
                            .iter()
                            .map(|range| TokenBehaviorRange {
                                start: range.start,
                                end: range.end,
                                behavior: local_to_global[range.behavior as usize],
                            })
                            .collect::<Vec<_>>();
                        global_regions.intern(remapped)
                    })
                    .clone();
                by_tsid.push((tsid, global_region));
            }
            merged_profiles.push(PointwiseProfile { by_tsid });
        }
    }

    Some(PointwiseProfileBuildOutput {
        profiles: merged_profiles,
        behaviors: global_behaviors,
        regions: global_regions,
        cache_entries,
        cache_hits,
        cache_misses,
        parallel_chunks,
    })
}

fn pointwise_profiles_equal_by_value(
    left_profiles: &[PointwiseProfile],
    left_behaviors: &PointwiseBehaviorInterner,
    right_profiles: &[PointwiseProfile],
    right_behaviors: &PointwiseBehaviorInterner,
) -> bool {
    left_profiles.len() == right_profiles.len()
        && left_profiles.iter().zip(right_profiles).all(|(left, right)| {
            left.by_tsid.len() == right.by_tsid.len()
                && left
                    .by_tsid
                    .iter()
                    .zip(&right.by_tsid)
                    .all(|((left_tsid, left_region), (right_tsid, right_region))| {
                        left_tsid == right_tsid
                            && left_region.len() == right_region.len()
                            && left_region.iter().zip(right_region.iter()).all(
                                |(left_range, right_range)| {
                                    left_range.start == right_range.start
                                        && left_range.end == right_range.end
                                        && left_behaviors.get(left_range.behavior)
                                            == right_behaviors.get(right_range.behavior)
                                },
                            )
                    })
        })
}

/// Range-compressed equivalent of [`build_pointwise_profile`].
///
/// Every point in one emitted TSID interval observes the same final and
/// transition token sets, so the token behavior region is constant throughout
/// that interval. Keeping the interval intact is exact.
fn build_pointwise_range_profile(
    domain: &Weight,
    profile: &ClassProfile,
    behaviors: &mut PointwiseBehaviorInterner,
    regions: &mut PointwiseRegionInterner,
    build_cache: &mut PointwiseRegionBuildCache,
) -> Option<PointwiseRangeProfile> {
    if domain.is_full()
        || profile.final_weight.as_ref().is_some_and(Weight::is_full)
        || profile.weights.iter().any(|(_, weight)| weight.is_full())
    {
        return None;
    }

    let transitions: Vec<(Label, u32, &Weight)> = profile
        .weights
        .iter()
        .map(|(label, weight)| Some((*label, profile_target_for_label(profile, *label)?, weight)))
        .collect::<Option<_>>()?;
    let mut by_tsid_range = Vec::new();
    for (domain_start, domain_end, domain_tokens) in domain.compact_entries()? {
        let mut boundaries = vec![u64::from(domain_start), u64::from(domain_end) + 1];
        if let Some(final_weight) = &profile.final_weight {
            for (tsid_range, _) in final_weight.raw_range_values() {
                add_tsid_boundary_if_overlapping(
                    &mut boundaries,
                    domain_start,
                    domain_end,
                    *tsid_range.start(),
                    *tsid_range.end(),
                );
            }
        }
        for (_, _, weight) in &transitions {
            for (tsid_range, _) in weight.raw_range_values() {
                add_tsid_boundary_if_overlapping(
                    &mut boundaries,
                    domain_start,
                    domain_end,
                    *tsid_range.start(),
                    *tsid_range.end(),
                );
            }
        }
        boundaries.sort_unstable();
        boundaries.dedup();

        for pair in boundaries.windows(2) {
            let start64 = pair[0];
            let next = pair[1];
            if start64 > u64::from(u32::MAX) || start64 >= next {
                continue;
            }
            let tsid_start = start64 as u32;
            let tsid_end = (next - 1) as u32;
            let final_tokens = profile
                .final_weight
                .as_ref()
                .and_then(|weight| weight.token_set_for_tsid_ref(tsid_start))
                .map(|tokens| tokens.as_ref());
            let mut active_transitions = Vec::new();
            for (label, target, weight) in &transitions {
                if let Some(tokens) = weight.token_set_for_tsid_ref(tsid_start) {
                    active_transitions.push((*label, *target, tokens.as_ref()));
                }
            }
            let region = build_token_behavior_region(
                domain_tokens.as_ref(),
                final_tokens,
                &active_transitions,
                behaviors,
                regions,
                build_cache,
            )?;
            push_pointwise_tsid_range(&mut by_tsid_range, tsid_start, tsid_end, region);
        }
    }
    Some(PointwiseRangeProfile { by_tsid_range })
}

fn token_behavior_ranges_compatible(
    left: &[TokenBehaviorRange],
    right: &[TokenBehaviorRange],
) -> bool {
    let mut left_index = 0usize;
    let mut right_index = 0usize;
    while left_index < left.len() && right_index < right.len() {
        let left_range = left[left_index];
        let right_range = right[right_index];
        if left_range.end < right_range.start {
            left_index += 1;
            continue;
        }
        if right_range.end < left_range.start {
            right_index += 1;
            continue;
        }
        if left_range.behavior != right_range.behavior {
            return false;
        }
        if left_range.end <= right_range.end {
            left_index += 1;
        }
        if right_range.end <= left_range.end {
            right_index += 1;
        }
    }
    true
}

/// Overlay two compatible sparse functions. Equal behavior is required where
/// both inputs are defined; the output preserves their exact union.
fn overlay_compatible_token_behavior_ranges(
    existing: &[TokenBehaviorRange],
    add: &[TokenBehaviorRange],
) -> Vec<TokenBehaviorRange> {
    let mut result = Vec::with_capacity(existing.len() + add.len());
    let mut existing_index = 0usize;
    let mut add_index = 0usize;
    let mut current_existing = existing.get(existing_index).copied();
    let mut current_add = add.get(add_index).copied();

    loop {
        match (current_existing, current_add) {
            (None, None) => break,
            (Some(range), None) => {
                push_token_behavior_range(&mut result, range.start, range.end, range.behavior);
                existing_index += 1;
                current_existing = existing.get(existing_index).copied();
            }
            (None, Some(range)) => {
                push_token_behavior_range(&mut result, range.start, range.end, range.behavior);
                add_index += 1;
                current_add = add.get(add_index).copied();
            }
            (Some(mut left), Some(mut right)) => {
                if left.end < right.start {
                    push_token_behavior_range(&mut result, left.start, left.end, left.behavior);
                    existing_index += 1;
                    current_existing = existing.get(existing_index).copied();
                    continue;
                }
                if right.end < left.start {
                    push_token_behavior_range(&mut result, right.start, right.end, right.behavior);
                    add_index += 1;
                    current_add = add.get(add_index).copied();
                    continue;
                }

                debug_assert_eq!(left.behavior, right.behavior);
                if left.start < right.start {
                    push_token_behavior_range(&mut result, left.start, right.start - 1, left.behavior);
                    left.start = right.start;
                } else if right.start < left.start {
                    push_token_behavior_range(&mut result, right.start, left.start - 1, right.behavior);
                    right.start = left.start;
                }

                let end = left.end.min(right.end);
                push_token_behavior_range(&mut result, left.start, end, left.behavior);
                if left.end == end {
                    existing_index += 1;
                    current_existing = existing.get(existing_index).copied();
                } else {
                    left.start = end + 1;
                    current_existing = Some(left);
                }
                if right.end == end {
                    add_index += 1;
                    current_add = add.get(add_index).copied();
                } else {
                    right.start = end + 1;
                    current_add = Some(right);
                }
            }
        }
    }
    result
}

fn pointwise_profile_compatible(group: &PointwiseMergeGroup, profile: &PointwiseProfile) -> bool {
    profile.by_tsid.iter().all(|(tsid, ranges)| {
        group.behavior_by_tsid.get(*tsid).is_none_or(|existing| {
            Arc::ptr_eq(existing, ranges)
                || token_behavior_ranges_compatible(existing.as_ref(), ranges.as_ref())
        })
    })
}

fn merge_pointwise_profile_into_group(
    group: &mut PointwiseMergeGroup,
    profile: &PointwiseProfile,
    regions: &mut PointwiseRegionInterner,
) {
    group.behavior_by_tsid.merge_profile(profile, regions);
}

fn pointwise_conflict_graph_coloring(
    class_profiles: &[ClassProfile],
    pointwise_profiles: &[PointwiseProfile],
    class_order: &[usize],
) -> (Vec<usize>, usize, usize, usize, usize) {
    fn profiles_compatible(left: &PointwiseProfile, right: &PointwiseProfile) -> bool {
        let mut left_index = 0usize;
        let mut right_index = 0usize;
        while left_index < left.by_tsid.len() && right_index < right.by_tsid.len() {
            let (left_tsid, left_region) = &left.by_tsid[left_index];
            let (right_tsid, right_region) = &right.by_tsid[right_index];
            if left_tsid < right_tsid {
                left_index += 1;
            } else if right_tsid < left_tsid {
                right_index += 1;
            } else {
                if !Arc::ptr_eq(left_region, right_region)
                    && !token_behavior_ranges_compatible(
                        left_region.as_ref(),
                        right_region.as_ref(),
                    )
                {
                    return false;
                }
                left_index += 1;
                right_index += 1;
            }
        }
        true
    }

    fn mark_conflict(conflicts: &mut [Vec<u64>], left: usize, right: usize) {
        debug_assert_ne!(left, right);
        let (left_words, right_words) = if left < right {
            let (before, after) = conflicts.split_at_mut(right);
            (&mut before[left], &mut after[0])
        } else {
            let (before, after) = conflicts.split_at_mut(left);
            (&mut after[0], &mut before[right])
        };
        left_words[right >> 6] |= 1u64 << (right & 63);
        right_words[left >> 6] |= 1u64 << (left & 63);
    }

    let class_count = class_profiles.len();
    let words = class_count.div_ceil(64);
    let mut conflicts = vec![vec![0u64; words]; class_count];
    let mut target_conflicts = 0usize;

    for left in 0..class_count {
        for right in left + 1..class_count {
            if !sorted_targets_compatible(
                &class_profiles[left].targets,
                &class_profiles[right].targets,
            ) {
                mark_conflict(&mut conflicts, left, right);
                target_conflicts += 1;
            }
        }
    }

    let max_tsid = pointwise_profiles
        .iter()
        .flat_map(|profile| profile.by_tsid.iter().map(|(tsid, _)| *tsid))
        .max()
        .unwrap_or(0);
    let mut region_ids = FxHashMap::<usize, usize>::default();
    let mut region_values = Vec::<Arc<Vec<TokenBehaviorRange>>>::new();
    let mut active_by_tsid = vec![Vec::<(usize, usize)>::new(); max_tsid as usize + 1];
    for (class, profile) in pointwise_profiles.iter().enumerate() {
        for (tsid, region) in &profile.by_tsid {
            let region_id = *region_ids
                .entry(Arc::as_ptr(region) as usize)
                .or_insert_with(|| {
                    let id = region_values.len();
                    region_values.push(Arc::clone(region));
                    id
                });
            active_by_tsid[*tsid as usize].push((class, region_id));
        }
    }
    let region_count = region_values.len();
    let mut compatible_regions = vec![false; region_count * region_count];
    for left in 0..region_count {
        for right in left..region_count {
            let compatible = left == right
                || token_behavior_ranges_compatible(
                    region_values[left].as_ref(),
                    region_values[right].as_ref(),
                );
            compatible_regions[left * region_count + right] = compatible;
            compatible_regions[right * region_count + left] = compatible;
        }
    }

    let accumulate = |active_slice: &[Vec<(usize, usize)>]| {
        let mut local_conflicts = vec![vec![0u64; words]; class_count];
        let mut behavior_conflicts = 0usize;
        let mut members_by_region = vec![Vec::<usize>::new(); region_count];
        let mut member_bits_by_region = vec![vec![0u64; words]; region_count];
        let mut used_regions = Vec::<usize>::new();
        for active in active_slice {
            used_regions.clear();
            for &(class, region) in active {
                if members_by_region[region].is_empty() {
                    used_regions.push(region);
                }
                members_by_region[region].push(class);
                member_bits_by_region[region][class >> 6] |= 1u64 << (class & 63);
            }
            for left_index in 0..used_regions.len() {
                let left_region = used_regions[left_index];
                for &right_region in &used_regions[left_index + 1..] {
                    if compatible_regions[left_region * region_count + right_region] {
                        continue;
                    }
                    behavior_conflicts += members_by_region[left_region].len()
                        * members_by_region[right_region].len();
                    for &class in &members_by_region[left_region] {
                        for (dst, add) in local_conflicts[class]
                            .iter_mut()
                            .zip(&member_bits_by_region[right_region])
                        {
                            *dst |= *add;
                        }
                    }
                    for &class in &members_by_region[right_region] {
                        for (dst, add) in local_conflicts[class]
                            .iter_mut()
                            .zip(&member_bits_by_region[left_region])
                        {
                            *dst |= *add;
                        }
                    }
                }
            }
            for region in used_regions.drain(..) {
                members_by_region[region].clear();
                member_bits_by_region[region].fill(0);
            }
        }
        (local_conflicts, behavior_conflicts)
    };

    let parallel_requested = std::env::var("GLRMASK_WEIGHTED_MINIMIZE_PARALLEL_CONFLICT_GRAPH")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false);
    let parallel_chunks = if parallel_requested && class_count >= 128 {
        rayon::current_num_threads()
            .min(active_by_tsid.len().div_ceil(32))
            .max(1)
    } else {
        1
    };
    let (behavior_matrix, behavior_conflicts) = if parallel_chunks > 1 {
        let chunk_size = active_by_tsid.len().div_ceil(parallel_chunks);
        active_by_tsid
            .par_chunks(chunk_size)
            .map(accumulate)
            .reduce(
                || (vec![vec![0u64; words]; class_count], 0usize),
                |(mut left_matrix, left_count), (right_matrix, right_count)| {
                    for (left_row, right_row) in left_matrix.iter_mut().zip(right_matrix) {
                        for (left, right) in left_row.iter_mut().zip(right_row) {
                            *left |= right;
                        }
                    }
                    (left_matrix, left_count + right_count)
                },
            )
    } else {
        accumulate(&active_by_tsid)
    };
    for (row, add) in conflicts.iter_mut().zip(behavior_matrix) {
        for (dst, value) in row.iter_mut().zip(add) {
            *dst |= value;
        }
    }

    if std::env::var_os("GLRMASK_VALIDATE_POINTWISE_CONFLICT_GRAPH").is_some() {
        for left in 0..class_count {
            for right in left + 1..class_count {
                let graph_conflict = conflicts[left][right >> 6] & (1u64 << (right & 63)) != 0;
                let direct_compatible = sorted_targets_compatible(
                    &class_profiles[left].targets,
                    &class_profiles[right].targets,
                ) && profiles_compatible(
                    &pointwise_profiles[left],
                    &pointwise_profiles[right],
                );
                assert_eq!(
                    graph_conflict,
                    !direct_compatible,
                    "pointwise conflict graph disagreed for classes {left} and {right}"
                );
            }
        }
    }

    let mut group_members = Vec::<Vec<u64>>::new();
    let mut class_to_group = vec![usize::MAX; class_count];
    for &class in class_order {
        let group = group_members
            .iter()
            .position(|members| {
                members
                    .iter()
                    .zip(&conflicts[class])
                    .all(|(members, conflicts)| members & conflicts == 0)
            })
            .unwrap_or_else(|| {
                group_members.push(vec![0u64; words]);
                group_members.len() - 1
            });
        group_members[group][class >> 6] |= 1u64 << (class & 63);
        class_to_group[class] = group;
    }

    (
        class_to_group,
        group_members.len(),
        target_conflicts,
        behavior_conflicts,
        parallel_chunks,
    )
}

/// Build the exact incompatibility graph directly from partial weighted
/// behaviors, without materializing a pointwise `(tsid, token) -> behavior`
/// profile for every class.
///
/// At one acyclic height all transition targets have already been quotiented.
/// Each class therefore denotes a partial deterministic function on its live
/// token domain: a final bit plus at most one quotient target per label. Two
/// classes are mergeable exactly when their target maps are compatible and
/// those partial functions agree on the intersection of their live domains.
/// Pairwise compatibility is sufficient for a whole group because pairwise
/// consistent partial functions have one well-defined union.
fn domain_conflict_graph_coloring(
    class_profiles: &[ClassProfile],
    class_domains: &[Weight],
    class_order: &[usize],
) -> (Vec<usize>, usize, usize, usize, usize) {
    fn mark_conflict(conflicts: &mut [Vec<u64>], left: usize, right: usize) {
        debug_assert_ne!(left, right);
        let (left_words, right_words) = if left < right {
            let (before, after) = conflicts.split_at_mut(right);
            (&mut before[left], &mut after[0])
        } else {
            let (before, after) = conflicts.split_at_mut(left);
            (&mut after[0], &mut before[right])
        };
        left_words[right >> 6] |= 1u64 << (right & 63);
        right_words[left >> 6] |= 1u64 << (left & 63);
    }

    debug_assert_eq!(class_profiles.len(), class_domains.len());
    let class_count = class_profiles.len();
    let words = class_count.div_ceil(64);
    let mut conflicts = vec![vec![0u64; words]; class_count];
    let mut target_conflicts = 0usize;
    let mut overlap_pairs = 0usize;
    let mut behavior_conflicts = 0usize;

    for left in 0..class_count {
        for right in left + 1..class_count {
            if !sorted_targets_compatible(
                &class_profiles[left].targets,
                &class_profiles[right].targets,
            ) {
                mark_conflict(&mut conflicts, left, right);
                target_conflicts += 1;
                continue;
            }

            if class_domains[left].is_disjoint(&class_domains[right]) {
                continue;
            }
            overlap_pairs += 1;

            let compatible = final_weights_compatible_on_domain_intersection(
                class_profiles[left].final_weight.as_ref(),
                class_profiles[right].final_weight.as_ref(),
                &class_domains[left],
                &class_domains[right],
            ) && sorted_weights_compatible_on_domain_intersection(
                &class_profiles[left].weights,
                &class_profiles[right].weights,
                &class_domains[left],
                &class_domains[right],
            );
            if !compatible {
                mark_conflict(&mut conflicts, left, right);
                behavior_conflicts += 1;
            }
        }
    }

    let mut group_members = Vec::<Vec<u64>>::new();
    let mut class_to_group = vec![usize::MAX; class_count];
    for &class in class_order {
        let group = group_members
            .iter()
            .position(|members| {
                members
                    .iter()
                    .zip(&conflicts[class])
                    .all(|(members, conflicts)| members & conflicts == 0)
            })
            .unwrap_or_else(|| {
                group_members.push(vec![0u64; words]);
                group_members.len() - 1
            });
        group_members[group][class >> 6] |= 1u64 << (class & 63);
        class_to_group[class] = group;
    }

    (
        class_to_group,
        group_members.len(),
        target_conflicts,
        overlap_pairs,
        behavior_conflicts,
    )
}

#[derive(Clone, Copy)]
struct PointwiseBehaviorEvent {
    position: u64,
    class: usize,
    component: i32,
    add: bool,
}

fn append_weight_behavior_events(
    events_by_tsid: &mut BTreeMap<u32, Vec<PointwiseBehaviorEvent>>,
    class: usize,
    component: i32,
    weight: &Weight,
) -> bool {
    if weight.is_full() {
        return false;
    }
    for (tsid_range, tokens) in weight.raw_range_values() {
        let mut tsid = *tsid_range.start();
        loop {
            let events = events_by_tsid.entry(tsid).or_default();
            for token_range in tokens.ranges() {
                events.push(PointwiseBehaviorEvent {
                    position: u64::from(*token_range.start()),
                    class,
                    component,
                    add: true,
                });
                events.push(PointwiseBehaviorEvent {
                    position: u64::from(*token_range.end()) + 1,
                    class,
                    component,
                    add: false,
                });
            }
            if tsid == *tsid_range.end() {
                break;
            }
            tsid += 1;
        }
    }
    true
}

/// Exact conflict graph from one global sweep of weighted behavior events.
///
/// For a fixed `(tsid, token)` point, every class denotes either no function
/// (outside its live domain) or one deterministic behavior: a final bit and a
/// sorted list of `(label, quotient target)` pairs. Classes with distinct
/// behaviors at the same point conflict; equal behaviors agree there. Sweeping
/// all event boundaries therefore records exactly the same incompatibility
/// relation as materializing one complete pointwise profile per class, but its
/// work is proportional to weight-range events and actual behavior changes.
fn event_conflict_graph_coloring(
    class_profiles: &[ClassProfile],
    class_domains: &[Weight],
    class_order: &[usize],
) -> Option<(Vec<usize>, usize, usize, usize, usize)> {
    fn mark_conflict(conflicts: &mut [Vec<u64>], left: usize, right: usize) {
        debug_assert_ne!(left, right);
        let (left_words, right_words) = if left < right {
            let (before, after) = conflicts.split_at_mut(right);
            (&mut before[left], &mut after[0])
        } else {
            let (before, after) = conflicts.split_at_mut(left);
            (&mut after[0], &mut before[right])
        };
        left_words[right >> 6] |= 1u64 << (right & 63);
        right_words[left >> 6] |= 1u64 << (left & 63);
    }

    fn mark_cross_conflicts(
        conflicts: &mut [Vec<u64>],
        left: &[u64],
        right: &[u64],
    ) -> usize {
        let mut pairs = 0usize;
        for (word_index, &word) in left.iter().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                let class = word_index * 64 + bit;
                for (dst, add) in conflicts[class].iter_mut().zip(right) {
                    *dst |= *add;
                }
                pairs += right.iter().map(|word| word.count_ones() as usize).sum::<usize>();
                remaining &= remaining - 1;
            }
        }
        for (word_index, &word) in right.iter().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                let class = word_index * 64 + bit;
                for (dst, add) in conflicts[class].iter_mut().zip(left) {
                    *dst |= *add;
                }
                remaining &= remaining - 1;
            }
        }
        pairs
    }

    const DOMAIN_COMPONENT: i32 = -2;
    const FINAL_COMPONENT: i32 = -1;

    debug_assert_eq!(class_profiles.len(), class_domains.len());
    let class_count = class_profiles.len();
    let words = class_count.div_ceil(64);
    let mut conflicts = vec![vec![0u64; words]; class_count];
    let mut target_conflicts = 0usize;
    for left in 0..class_count {
        for right in left + 1..class_count {
            if !sorted_targets_compatible(
                &class_profiles[left].targets,
                &class_profiles[right].targets,
            ) {
                mark_conflict(&mut conflicts, left, right);
                target_conflicts += 1;
            }
        }
    }

    let mut transition_descriptors = Vec::<Vec<(Label, u32)>>::with_capacity(class_count);
    let mut events_by_tsid = BTreeMap::<u32, Vec<PointwiseBehaviorEvent>>::new();
    for class in 0..class_count {
        if !append_weight_behavior_events(
            &mut events_by_tsid,
            class,
            DOMAIN_COMPONENT,
            &class_domains[class],
        ) {
            return None;
        }
        if let Some(final_weight) = &class_profiles[class].final_weight
            && !append_weight_behavior_events(
                &mut events_by_tsid,
                class,
                FINAL_COMPONENT,
                final_weight,
            )
        {
            return None;
        }

        let mut descriptors = Vec::with_capacity(class_profiles[class].weights.len());
        for (index, (label, weight)) in class_profiles[class].weights.iter().enumerate() {
            let target = profile_target_for_label(&class_profiles[class], *label)?;
            descriptors.push((*label, target));
            if !append_weight_behavior_events(
                &mut events_by_tsid,
                class,
                index as i32,
                weight,
            ) {
                return None;
            }
        }
        transition_descriptors.push(descriptors);
    }

    let mut interner = PointwiseBehaviorInterner::default();
    let mut behavior_conflict_pairs = 0usize;
    let mut interval_count = 0usize;
    for (_, mut events) in events_by_tsid {
        events.sort_unstable_by_key(|event| event.position);
        let mut domain_active = vec![false; class_count];
        let mut final_active = vec![false; class_count];
        let mut transition_active = transition_descriptors
            .iter()
            .map(|transitions| vec![false; transitions.len()])
            .collect::<Vec<_>>();
        let mut class_behavior = vec![None::<u32>; class_count];
        let mut members_by_behavior = Vec::<Vec<u64>>::new();
        let mut affected = Vec::<usize>::new();
        let mut affected_mark = vec![false; class_count];

        let mut event_index = 0usize;
        while event_index < events.len() {
            let position = events[event_index].position;
            let mut event_end = event_index + 1;
            while event_end < events.len() && events[event_end].position == position {
                event_end += 1;
            }

            affected.clear();
            for event in &events[event_index..event_end] {
                if !affected_mark[event.class] {
                    affected_mark[event.class] = true;
                    affected.push(event.class);
                    if let Some(behavior) = class_behavior[event.class].take() {
                        members_by_behavior[behavior as usize][event.class >> 6] &=
                            !(1u64 << (event.class & 63));
                    }
                }
            }

            for event in &events[event_index..event_end] {
                match event.component {
                    DOMAIN_COMPONENT => domain_active[event.class] = event.add,
                    FINAL_COMPONENT => final_active[event.class] = event.add,
                    component => {
                        transition_active[event.class][component as usize] = event.add;
                    }
                }
            }

            for &class in &affected {
                affected_mark[class] = false;
                if !domain_active[class] {
                    continue;
                }
                let transitions = transition_descriptors[class]
                    .iter()
                    .enumerate()
                    .filter_map(|(index, value)| transition_active[class][index].then_some(*value))
                    .collect::<Vec<_>>();
                if !final_active[class] && transitions.is_empty() {
                    return None;
                }
                let behavior = interner.intern(final_active[class], transitions);
                if members_by_behavior.len() <= behavior as usize {
                    members_by_behavior
                        .resize_with(behavior as usize + 1, || vec![0u64; words]);
                }
                members_by_behavior[behavior as usize][class >> 6] |=
                    1u64 << (class & 63);
                class_behavior[class] = Some(behavior);
            }

            let next_position = events.get(event_end).map(|event| event.position);
            if next_position.is_some_and(|next| next > position) {
                interval_count += 1;
                let active_behaviors = members_by_behavior
                    .iter()
                    .enumerate()
                    .filter_map(|(behavior, members)| {
                        members.iter().any(|word| *word != 0).then_some((behavior, members))
                    })
                    .collect::<Vec<_>>();
                for left in 0..active_behaviors.len() {
                    for right in left + 1..active_behaviors.len() {
                        behavior_conflict_pairs += mark_cross_conflicts(
                            &mut conflicts,
                            active_behaviors[left].1,
                            active_behaviors[right].1,
                        );
                    }
                }
            }
            event_index = event_end;
        }
    }

    let mut group_members = Vec::<Vec<u64>>::new();
    let mut class_to_group = vec![usize::MAX; class_count];
    for &class in class_order {
        let group = group_members
            .iter()
            .position(|members| {
                members
                    .iter()
                    .zip(&conflicts[class])
                    .all(|(members, conflicts)| members & conflicts == 0)
            })
            .unwrap_or_else(|| {
                group_members.push(vec![0u64; words]);
                group_members.len() - 1
            });
        group_members[group][class >> 6] |= 1u64 << (class & 63);
        class_to_group[class] = group;
    }

    Some((
        class_to_group,
        group_members.len(),
        target_conflicts,
        interval_count,
        behavior_conflict_pairs,
    ))
}

enum GroupCompatibilityWitness {
    Target(Label),
    Point { tsid: u32, token: u32 },
}

fn first_weight_point(weight: &Weight) -> Option<(u32, u32)> {
    if weight.is_empty() || weight.is_full() {
        return None;
    }
    let (tsid_range, tokens) = weight.raw_range_values().next()?;
    let token_range = tokens.ranges().next()?;
    Some((*tsid_range.start(), *token_range.start()))
}

fn weight_contains_point(weight: &Weight, tsid: u32, token: u32) -> bool {
    if weight.is_full() {
        return true;
    }
    weight
        .token_set_for_tsid_ref(tsid)
        .is_some_and(|tokens| tokens.contains(token))
}

fn profile_weight_for_label(profile: &ClassProfile, label: Label) -> Option<&Weight> {
    profile
        .weights
        .binary_search_by_key(&label, |(candidate, _)| *candidate)
        .ok()
        .map(|index| &profile.weights[index].1)
}

/// Prove that all classes in `members` define one compatible partial
/// deterministic behavior, or return one exact separating witness.
///
/// For a boolean observable component `c` (finality or one labelled edge), let
/// `W_i,c` be the tokens where class `i` exposes that component and `D_i` its
/// live domain. The component agrees across every overlapping class iff
///
///   union_i W_i,c  ∩  union_i (D_i \ W_i,c) = empty.
///
/// The forward direction is immediate: a point in the intersection witnesses
/// one active class with `c` and one active class without it. Conversely, any
/// disagreement supplies exactly such a point. Checking every component plus
/// the token-independent target-map condition is therefore necessary and
/// sufficient for whole-group compatibility.
fn prove_group_compatible_or_witness(
    members: &[usize],
    class_profiles: &[ClassProfile],
    class_domains: &[Weight],
) -> Result<(), GroupCompatibilityWitness> {
    if members.len() <= 1 {
        return Ok(());
    }
    if members
        .iter()
        .any(|&class| class_domains[class].is_full())
    {
        // The finite event/set representation deliberately does not invent a
        // universe for the all sentinel. Let the established pointwise path
        // handle that uncommon case.
        return Err(GroupCompatibilityWitness::Point {
            tsid: u32::MAX,
            token: u32::MAX,
        });
    }

    let mut target_by_label = BTreeMap::<Label, u32>::new();
    for &class in members {
        for &(label, target) in &class_profiles[class].targets {
            match target_by_label.entry(label) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(target);
                }
                std::collections::btree_map::Entry::Occupied(entry)
                    if *entry.get() != target =>
                {
                    return Err(GroupCompatibilityWitness::Target(label));
                }
                std::collections::btree_map::Entry::Occupied(_) => {}
            }
        }
    }

    let mut components = Vec::<Option<Label>>::with_capacity(target_by_label.len() + 1);
    components.push(None); // finality
    components.extend(target_by_label.keys().copied().map(Some));

    for component in components {
        let mut positive = Vec::<&Weight>::new();
        let mut negative = Vec::<Weight>::with_capacity(members.len());
        for &class in members {
            let domain = &class_domains[class];
            let component_weight = match component {
                None => class_profiles[class].final_weight.as_ref(),
                Some(label) => profile_weight_for_label(&class_profiles[class], label),
            };
            if let Some(weight) = component_weight {
                debug_assert!(weight.is_subset(domain));
                if !weight.is_empty() {
                    positive.push(weight);
                }
                let absent = domain.difference(weight);
                if !absent.is_empty() {
                    negative.push(absent);
                }
            } else {
                negative.push(domain.clone());
            }
        }
        if positive.is_empty() || negative.is_empty() {
            continue;
        }
        let positive_union = Weight::union_all(positive);
        let negative_union = Weight::union_all(negative.iter());
        let disagreement = positive_union.intersection(&negative_union);
        if let Some((tsid, token)) = first_weight_point(&disagreement) {
            return Err(GroupCompatibilityWitness::Point { tsid, token });
        }
    }
    Ok(())
}

fn class_behavior_at_point(
    class: usize,
    tsid: u32,
    token: u32,
    class_profiles: &[ClassProfile],
    class_domains: &[Weight],
) -> Option<PointwiseBehavior> {
    if !weight_contains_point(&class_domains[class], tsid, token) {
        return None;
    }
    let final_active = class_profiles[class]
        .final_weight
        .as_ref()
        .is_some_and(|weight| weight_contains_point(weight, tsid, token));
    let transitions = class_profiles[class]
        .weights
        .iter()
        .filter_map(|(label, weight)| {
            weight_contains_point(weight, tsid, token).then(|| {
                (
                    *label,
                    profile_target_for_label(&class_profiles[class], *label)
                        .expect("class transition weight must retain its target"),
                )
            })
        })
        .collect::<Vec<_>>();
    debug_assert!(final_active || !transitions.is_empty());
    Some(PointwiseBehavior {
        final_active,
        transitions,
    })
}

/// Recursively partition classes using only global compatibility proofs and
/// concrete counterexamples. Every emitted group has passed the complete set
/// identity above, so the quotient is exact even if the partition differs from
/// the historical greedy coloring.
fn recursive_witness_coloring(
    class_profiles: &[ClassProfile],
    class_domains: &[Weight],
    class_order: &[usize],
) -> Option<(Vec<usize>, usize, usize)> {
    let mut pending = vec![class_order.to_vec()];
    let mut groups = Vec::<Vec<usize>>::new();
    let mut proof_count = 0usize;
    let mut witness_count = 0usize;

    while let Some(members) = pending.pop() {
        proof_count += 1;
        match prove_group_compatible_or_witness(&members, class_profiles, class_domains) {
            Ok(()) => groups.push(members),
            Err(GroupCompatibilityWitness::Target(label)) => {
                witness_count += 1;
                let mut split = Vec::<(u32, Vec<usize>)>::new();
                let mut by_target = FxHashMap::<u32, usize>::default();
                let mut inactive = Vec::new();
                for class in members {
                    if let Some(target) = profile_target_for_label(&class_profiles[class], label) {
                        let index = *by_target.entry(target).or_insert_with(|| {
                            let index = split.len();
                            split.push((target, Vec::new()));
                            index
                        });
                        split[index].1.push(class);
                    } else {
                        inactive.push(class);
                    }
                }
                if split.len() < 2 {
                    return None;
                }
                let largest = split
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, (_, group))| group.len())
                    .map(|(index, _)| index)
                    .unwrap();
                split[largest].1.extend(inactive);
                pending.extend(split.into_iter().map(|(_, group)| group));
            }
            Err(GroupCompatibilityWitness::Point { tsid, token }) => {
                if tsid == u32::MAX && token == u32::MAX {
                    return None;
                }
                witness_count += 1;
                let mut behavior_ids = FxHashMap::<PointwiseBehavior, usize>::default();
                let mut split = Vec::<Vec<usize>>::new();
                let mut inactive = Vec::new();
                for class in members {
                    if let Some(behavior) = class_behavior_at_point(
                        class,
                        tsid,
                        token,
                        class_profiles,
                        class_domains,
                    ) {
                        let index = *behavior_ids.entry(behavior).or_insert_with(|| {
                            let index = split.len();
                            split.push(Vec::new());
                            index
                        });
                        split[index].push(class);
                    } else {
                        inactive.push(class);
                    }
                }
                if split.len() < 2 {
                    return None;
                }
                let largest = split
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, group)| group.len())
                    .map(|(index, _)| index)
                    .unwrap();
                split[largest].extend(inactive);
                pending.extend(split);
            }
        }
    }

    let mut class_to_group = vec![usize::MAX; class_profiles.len()];
    for (group, members) in groups.iter().enumerate() {
        // Re-prove every leaf independently. This is cheap for the final small
        // number of groups and makes the construction fail closed.
        if prove_group_compatible_or_witness(members, class_profiles, class_domains).is_err() {
            return None;
        }
        for &class in members {
            class_to_group[class] = group;
        }
    }
    Some((class_to_group, proof_count, witness_count))
}

struct HybridColoring {
    colors: Vec<usize>,
    direct_builders: Option<Vec<MergedStateBuilder>>,
}

impl HybridColoring {
    fn colors(colors: Vec<usize>) -> Self {
        Self {
            colors,
            direct_builders: None,
        }
    }
}

#[derive(Clone)]
struct DecodedPointwiseRegion {
    final_tokens: Option<SharedTokenSet>,
    transitions: Vec<(Label, u32, SharedTokenSet)>,
}

fn decode_pointwise_region(
    region: &Arc<Vec<TokenBehaviorRange>>,
    behaviors: &PointwiseBehaviorInterner,
) -> DecodedPointwiseRegion {
    let mut final_ranges = Vec::new();
    let mut transition_ranges = BTreeMap::<(Label, u32), Vec<std::ops::RangeInclusive<u32>>>::new();
    for token_range in region.iter() {
        let behavior = behaviors.get(token_range.behavior);
        if behavior.final_active {
            final_ranges.push(token_range.start..=token_range.end);
        }
        for &(label, target) in &behavior.transitions {
            transition_ranges
                .entry((label, target))
                .or_default()
                .push(token_range.start..=token_range.end);
        }
    }
    let final_tokens = (!final_ranges.is_empty())
        .then(|| shared_rangeset(final_ranges.into_iter().collect::<RangeSetBlaze<u32>>()));
    let transitions = transition_ranges
        .into_iter()
        .map(|((label, target), ranges)| {
            (
                label,
                target,
                shared_rangeset(ranges.into_iter().collect::<RangeSetBlaze<u32>>()),
            )
        })
        .collect();
    DecodedPointwiseRegion {
        final_tokens,
        transitions,
    }
}

fn direct_builders_from_pointwise_groups(
    class_to_group: &[usize],
    group_count: usize,
    pointwise_profiles: &[PointwiseProfile],
    behavior_map_layout: PointwiseBehaviorMapLayout,
    behaviors: &PointwiseBehaviorInterner,
    regions: &mut PointwiseRegionInterner,
) -> Vec<MergedStateBuilder> {
    let mut group_maps = (0..group_count)
        .map(|_| PointwiseBehaviorMap::new(behavior_map_layout))
        .collect::<Vec<_>>();
    for (class, profile) in pointwise_profiles.iter().enumerate() {
        group_maps[class_to_group[class]].merge_profile(profile, regions);
    }

    let mut decoded = FxHashMap::<usize, DecodedPointwiseRegion>::default();
    group_maps
        .into_iter()
        .map(|group_map| {
            let mut entries = match group_map {
                PointwiseBehaviorMap::Sparse(entries) => entries.into_iter().collect::<Vec<_>>(),
                PointwiseBehaviorMap::Dense(entries) => entries
                    .into_iter()
                    .enumerate()
                    .filter_map(|(tsid, region)| region.map(|region| (tsid as u32, region)))
                    .collect::<Vec<_>>(),
            };
            entries.sort_unstable_by_key(|(tsid, _)| *tsid);

            let mut final_entries = Vec::<(u32, SharedTokenSet)>::new();
            let mut transition_entries =
                BTreeMap::<Label, (u32, Vec<(u32, SharedTokenSet)>)>::new();
            for (tsid, region) in entries {
                let key = Arc::as_ptr(&region) as usize;
                let decoded_region = decoded
                    .entry(key)
                    .or_insert_with(|| decode_pointwise_region(&region, behaviors));
                if let Some(tokens) = &decoded_region.final_tokens {
                    final_entries.push((tsid, Arc::clone(tokens)));
                }
                for (label, target, tokens) in &decoded_region.transitions {
                    let (existing_target, entries) = transition_entries
                        .entry(*label)
                        .or_insert_with(|| (*target, Vec::new()));
                    debug_assert_eq!(*existing_target, *target);
                    entries.push((tsid, Arc::clone(tokens)));
                }
            }

            let mut builder = MergedStateBuilder::default();
            let final_weight = Weight::from_per_tsid_shared(final_entries);
            if !final_weight.is_empty() {
                builder.final_weights_pending.push(final_weight);
            }
            for (label, (target, entries)) in transition_entries {
                let weight = Weight::from_per_tsid_shared(entries);
                if !weight.is_empty() {
                    builder
                        .transitions_pending
                        .insert(label, (target, vec![weight]));
                }
            }
            builder
        })
        .collect()
}

fn validate_direct_builders(
    candidates: &[usize],
    coloring: &[usize],
    dwa: &DWA,
    productive_transitions: &[Vec<ProductiveTransition>],
    old_to_new: &[u32],
    direct_builders: &[MergedStateBuilder],
) {
    let mut legacy = (0..direct_builders.len())
        .map(|_| MergedStateBuilder::default())
        .collect::<Vec<_>>();
    for (index, &color) in coloring.iter().enumerate() {
        merge_productive_state_into_builder(
            candidates[index],
            color,
            dwa,
            productive_transitions,
            old_to_new,
            &mut legacy,
        );
    }

    for (group, (direct, legacy)) in direct_builders.iter().zip(&legacy).enumerate() {
        let direct_final = Weight::union_all(direct.final_weights_pending.iter());
        let legacy_final = Weight::union_all(legacy.final_weights_pending.iter());
        assert_eq!(
            direct_final, legacy_final,
            "direct quotient final weight disagreed for group {group}"
        );

        assert_eq!(
            direct.transitions_pending.len(),
            legacy.transitions_pending.len(),
            "direct quotient transition-label count disagreed for group {group}"
        );
        for (&label, (legacy_target, legacy_weights)) in &legacy.transitions_pending {
            let (direct_target, direct_weights) = direct.transitions_pending.get(&label).unwrap_or_else(|| {
                panic!("direct quotient omitted label {label} for group {group}")
            });
            assert_eq!(
                direct_target, legacy_target,
                "direct quotient target disagreed for group {group}, label {label}"
            );
            assert_eq!(
                Weight::union_all(direct_weights.iter()),
                Weight::union_all(legacy_weights.iter()),
                "direct quotient transition weight disagreed for group {group}, label {label}"
            );
        }
    }
}

/// Exact greedy grouping using one sparse partial behavior function per group.
/// The class order and target-map restriction are identical to the memberwise
/// path; only the witness representation changes.
fn try_build_and_color_pointwise(
    candidates: &[usize],
    class_coloring: &[usize],
    class_needed_union: &[Weight],
    class_profiles: &[ClassProfile],
    pointwise_class_order: PointwiseClassOrder,
    profile_enabled: bool,
) -> Option<HybridColoring> {
    let started_at = Instant::now();
    let profile_started_at = Instant::now();
    let parallel_requested = std::env::var("GLRMASK_WEIGHTED_MINIMIZE_PARALLEL_POINTWISE")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false);
    let PointwiseProfileBuildOutput {
        profiles: pointwise_profiles,
        behaviors: interner,
        mut regions,
        cache_entries: region_build_cache_entries,
        cache_hits: region_build_cache_hits,
        cache_misses: region_build_cache_misses,
        parallel_chunks,
    } = if parallel_requested && class_profiles.len() >= 128 {
        let parallel = build_pointwise_profiles_parallel(class_needed_union, class_profiles)?;
        if std::env::var_os("GLRMASK_VALIDATE_WEIGHTED_MINIMIZE_PARALLEL_POINTWISE")
            .is_some()
        {
            let serial = build_pointwise_profiles_serial(class_needed_union, class_profiles)?;
            assert!(
                pointwise_profiles_equal_by_value(
                    &parallel.profiles,
                    &parallel.behaviors,
                    &serial.profiles,
                    &serial.behaviors,
                ),
                "parallel pointwise profile construction differs from serial construction",
            );
        }
        parallel
    } else {
        build_pointwise_profiles_serial(class_needed_union, class_profiles)?
    };
    let profile_build_ms = profile_started_at.elapsed().as_secs_f64() * 1000.0;

    let profile_entries = pointwise_profiles
        .iter()
        .map(|profile| profile.by_tsid.len())
        .sum::<usize>();
    let max_tsid = pointwise_profiles
        .iter()
        .flat_map(|profile| profile.by_tsid.iter().map(|(tsid, _)| *tsid))
        .max();
    let behavior_map_layout = max_tsid
        .and_then(|max_tsid| usize::try_from(max_tsid).ok()?.checked_add(1))
        .filter(|&slots| slots <= MAX_DENSE_POINTWISE_TSID_SLOTS && slots <= profile_entries)
        .map(|slots| PointwiseBehaviorMapLayout::Dense { slots })
        .unwrap_or(PointwiseBehaviorMapLayout::Sparse);

    let merge_started_at = Instant::now();
    let mut groups = Vec::<PointwiseMergeGroup>::new();
    let mut group_attempts = 0usize;
    let mut target_rejects = 0usize;
    let mut behavior_rejects = 0usize;
    let mut class_order = (0..class_profiles.len()).collect::<Vec<_>>();
    if pointwise_class_order == PointwiseClassOrder::DescendingDomain {
        class_order.sort_unstable_by_key(|&class| {
            std::cmp::Reverse((
                pointwise_profiles[class].by_tsid.len(),
                class_profiles[class].targets.len(),
                class,
            ))
        });
    }
    let conflict_graph_mode = std::env::var("GLRMASK_WEIGHTED_MINIMIZE_POINTWISE_COLORING")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase());
    let use_conflict_graph = match conflict_graph_mode.as_deref() {
        Some("conflict" | "conflict_graph" | "graph") => true,
        Some("merge" | "overlay" | "legacy") => false,
        Some("auto") | None => (128..=1_024).contains(&class_profiles.len()),
        Some(other) => panic!(
            "unknown GLRMASK_WEIGHTED_MINIMIZE_POINTWISE_COLORING={other:?}; expected auto, merge, or conflict"
        ),
    } || std::env::var_os("GLRMASK_WEIGHTED_MINIMIZE_POINTWISE_CONFLICT_GRAPH").is_some();
    if use_conflict_graph {
        let conflict_started_at = Instant::now();
        let (
            class_to_group,
            group_count,
            target_conflicts,
            behavior_conflicts,
            conflict_chunks,
        ) =
            pointwise_conflict_graph_coloring(
                class_profiles,
                &pointwise_profiles,
                &class_order,
            );
        let coloring = class_coloring
            .iter()
            .map(|class| class_to_group[*class])
            .collect();
        let direct_builders = std::env::var_os("GLRMASK_WEIGHTED_MINIMIZE_DIRECT_QUOTIENT")
            .is_some()
            .then(|| {
                direct_builders_from_pointwise_groups(
                    &class_to_group,
                    group_count,
                    &pointwise_profiles,
                    behavior_map_layout,
                    &interner,
                    &mut regions,
                )
            });
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][weighted_dwa_minimize_pointwise_conflicts] candidates={} classes={} groups={} target_conflicts={} behavior_conflicts={} profile_chunks={} conflict_chunks={} profile_build_ms={:.3} conflict_ms={:.3} total_ms={:.3}",
                candidates.len(),
                class_profiles.len(),
                group_count,
                target_conflicts,
                behavior_conflicts,
                parallel_chunks,
                conflict_chunks,
                profile_build_ms,
                conflict_started_at.elapsed().as_secs_f64() * 1000.0,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Some(HybridColoring {
            colors: coloring,
            direct_builders,
        });
    }
    for class in class_order {
        let class_profile = &class_profiles[class];
        let pointwise_profile = &pointwise_profiles[class];
        let mut placed = false;
        for group in &mut groups {
            group_attempts += 1;
            if !targets_compatible_with_group_map(&class_profile.targets, &group.targets_by_label) {
                target_rejects += 1;
                continue;
            }
            if !pointwise_profile_compatible(group, pointwise_profile) {
                behavior_rejects += 1;
                continue;
            }
            #[cfg(debug_assertions)]
            debug_assert!(memberwise_group_compatible(
                &class_needed_union[class],
                class_profile,
                &group.member_classes,
                class_needed_union,
                class_profiles,
            ));
            for (label, target) in &class_profile.targets {
                group.targets_by_label.entry(*label).or_insert(*target);
            }
            merge_pointwise_profile_into_group(group, pointwise_profile, &mut regions);
            group.member_classes.push(class);
            placed = true;
            break;
        }
        if !placed {
            let mut targets_by_label = FxHashMap::default();
            targets_by_label.reserve(class_profile.targets.len());
            for (label, target) in &class_profile.targets {
                targets_by_label.insert(*label, *target);
            }
            let mut group = PointwiseMergeGroup {
                targets_by_label,
                behavior_by_tsid: PointwiseBehaviorMap::new(behavior_map_layout),
                member_classes: vec![class],
            };
            merge_pointwise_profile_into_group(&mut group, pointwise_profile, &mut regions);
            groups.push(group);
        }
    }
    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut class_to_group = vec![0usize; class_profiles.len()];
    for (group_id, group) in groups.iter().enumerate() {
        for &class in &group.member_classes {
            class_to_group[class] = group_id;
        }
    }
    let coloring = class_coloring
        .iter()
        .map(|class| class_to_group[*class])
        .collect();

    if profile_enabled {
        let (direct_overlay_slots, direct_overlay_hits, direct_overlay_misses, direct_overlay_replacements) =
            regions.direct_overlay_stats();
        let region_entries = groups
            .iter()
            .map(|group| {
                group.behavior_by_tsid.region_entry_count()
            })
            .sum::<usize>();
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_pointwise] candidates={} classes={} groups={} behaviors={} interned_regions={} regions={} region_build_cache_entries={} region_build_cache_hits={} region_build_cache_misses={} direct_overlay_slots={} direct_overlay_hits={} direct_overlay_misses={} direct_overlay_replacements={} profile_build_ms={:.3} merge_ms={:.3} total_ms={:.3} group_attempts={} target_rejects={} behavior_rejects={}",
            candidates.len(),
            class_profiles.len(),
            groups.len(),
            interner.ids.len(),
            regions.regions.len(),
            region_entries,
            region_build_cache_entries,
            region_build_cache_hits,
            region_build_cache_misses,
            direct_overlay_slots,
            direct_overlay_hits,
            direct_overlay_misses,
            direct_overlay_replacements,
            profile_build_ms,
            merge_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
            group_attempts,
            target_rejects,
            behavior_rejects,
        );
        if parallel_chunks > 1 {
            eprintln!(
                "[glrmask/profile][weighted_dwa_minimize_pointwise_parallel] classes={} chunks={} profile_build_ms={:.3} cache_entries={} cache_hits={} cache_misses={}",
                class_profiles.len(),
                parallel_chunks,
                profile_build_ms,
                region_build_cache_entries,
                region_build_cache_hits,
                region_build_cache_misses,
            );
        }
    }
    Some(HybridColoring::colors(coloring))
}

/// Exact greedy pointwise coloring using TSID intervals rather than one map
/// entry per TSID. This is equivalent to [`try_build_and_color_pointwise`]; it
/// changes only the representation of a partial behavior function.
fn try_build_and_color_pointwise_ranges(
    candidates: &[usize],
    class_coloring: &[usize],
    class_needed_union: &[Weight],
    class_profiles: &[ClassProfile],
    pointwise_class_order: PointwiseClassOrder,
    profile_enabled: bool,
) -> Option<Vec<usize>> {
    let started_at = Instant::now();
    let mut interner = PointwiseBehaviorInterner::default();
    let mut regions = PointwiseRegionInterner::default();
    let mut region_build_cache = PointwiseRegionBuildCache::default();
    let profile_started_at = Instant::now();
    let mut pointwise_profiles = Vec::with_capacity(class_profiles.len());
    for (domain, profile) in class_needed_union.iter().zip(class_profiles) {
        pointwise_profiles.push(build_pointwise_range_profile(
            domain,
            profile,
            &mut interner,
            &mut regions,
            &mut region_build_cache,
        )?);
    }
    let profile_build_ms = profile_started_at.elapsed().as_secs_f64() * 1000.0;

    let profile_ranges = pointwise_profiles
        .iter()
        .map(|profile| profile.by_tsid_range.len())
        .sum::<usize>();
    let profile_tsid_cells = pointwise_profiles
        .iter()
        .flat_map(|profile| profile.by_tsid_range.iter())
        .map(|range| (range.end as usize - range.start as usize) + 1)
        .sum::<usize>();

    if pointwise_tsid_ranges_auto_enabled()
        && profile_ranges.saturating_mul(POINTWISE_TSID_RANGE_MIN_COMPRESSION)
            > profile_tsid_cells
    {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][weighted_dwa_minimize_pointwise_ranges_fallback] candidates={} classes={} profile_ranges={} profile_tsid_cells={} profile_build_ms={:.3}",
                candidates.len(),
                class_profiles.len(),
                profile_ranges,
                profile_tsid_cells,
                profile_build_ms,
            );
        }
        return try_build_and_color_pointwise(
            candidates,
            class_coloring,
            class_needed_union,
            class_profiles,
            pointwise_class_order,
            profile_enabled,
        )
        .map(|result| result.colors);
    }

    let merge_started_at = Instant::now();
    let mut groups = Vec::<PointwiseRangeMergeGroup>::new();
    let mut group_attempts = 0usize;
    let mut target_rejects = 0usize;
    let mut behavior_rejects = 0usize;
    let mut class_order = (0..class_profiles.len()).collect::<Vec<_>>();
    if pointwise_class_order == PointwiseClassOrder::DescendingDomain {
        class_order.sort_unstable_by_key(|&class| {
            std::cmp::Reverse((
                pointwise_profiles[class].by_tsid_range.len(),
                class_profiles[class].targets.len(),
                class,
            ))
        });
    }
    for class in class_order {
        let class_profile = &class_profiles[class];
        let pointwise_profile = &pointwise_profiles[class];
        let mut placed = false;
        for group in &mut groups {
            group_attempts += 1;
            if !targets_compatible_with_group_map(&class_profile.targets, &group.targets_by_label) {
                target_rejects += 1;
                continue;
            }
            if !group.behavior_by_tsid.profile_compatible(pointwise_profile) {
                behavior_rejects += 1;
                continue;
            }
            #[cfg(debug_assertions)]
            debug_assert!(memberwise_group_compatible(
                &class_needed_union[class],
                class_profile,
                &group.member_classes,
                class_needed_union,
                class_profiles,
            ));
            for (label, target) in &class_profile.targets {
                group.targets_by_label.entry(*label).or_insert(*target);
            }
            group
                .behavior_by_tsid
                .merge_profile(pointwise_profile, &mut regions);
            group.member_classes.push(class);
            placed = true;
            break;
        }
        if !placed {
            let mut targets_by_label = FxHashMap::default();
            targets_by_label.reserve(class_profile.targets.len());
            for (label, target) in &class_profile.targets {
                targets_by_label.insert(*label, *target);
            }
            let mut group = PointwiseRangeMergeGroup {
                targets_by_label,
                behavior_by_tsid: PointwiseRangeBehaviorMap::default(),
                member_classes: vec![class],
            };
            group
                .behavior_by_tsid
                .merge_profile(pointwise_profile, &mut regions);
            groups.push(group);
        }
    }
    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut class_to_group = vec![0usize; class_profiles.len()];
    for (group_id, group) in groups.iter().enumerate() {
        for &class in &group.member_classes {
            class_to_group[class] = group_id;
        }
    }
    let coloring = class_coloring
        .iter()
        .map(|class| class_to_group[*class])
        .collect();

    if profile_enabled {
        let (direct_overlay_slots, direct_overlay_hits, direct_overlay_misses, direct_overlay_replacements) =
            regions.direct_overlay_stats();
        let group_tsid_ranges = groups
            .iter()
            .map(|group| group.behavior_by_tsid.ranges.len())
            .sum::<usize>();
        let region_entries = groups
            .iter()
            .map(|group| group.behavior_by_tsid.region_entry_count())
            .sum::<usize>();
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_pointwise_ranges] candidates={} classes={} groups={} behaviors={} interned_regions={} profile_ranges={} profile_tsid_cells={} group_tsid_ranges={} regions={} region_build_cache_entries={} region_build_cache_hits={} region_build_cache_misses={} direct_overlay_slots={} direct_overlay_hits={} direct_overlay_misses={} direct_overlay_replacements={} profile_build_ms={:.3} merge_ms={:.3} total_ms={:.3} group_attempts={} target_rejects={} behavior_rejects={}",
            candidates.len(),
            class_profiles.len(),
            groups.len(),
            interner.ids.len(),
            regions.regions.len(),
            profile_ranges,
            profile_tsid_cells,
            group_tsid_ranges,
            region_entries,
            region_build_cache.entries.len(),
            region_build_cache.hits,
            region_build_cache.misses,
            direct_overlay_slots,
            direct_overlay_hits,
            direct_overlay_misses,
            direct_overlay_replacements,
            profile_build_ms,
            merge_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
            group_attempts,
            target_rejects,
            behavior_rejects,
        );
    }
    Some(coloring)
}

fn sorted_targets_compatible(class_targets: &[(Label, u32)], group_targets: &[(Label, u32)]) -> bool {
    let mut class_idx = 0;
    let mut group_idx = 0;

    while class_idx < class_targets.len() && group_idx < group_targets.len() {
        let (class_label, class_target) = class_targets[class_idx];
        let (group_label, group_target) = group_targets[group_idx];
        if class_label == group_label {
            if class_target != group_target {
                return false;
            }
            class_idx += 1;
            group_idx += 1;
        } else if class_label < group_label {
            class_idx += 1;
        } else {
            group_idx += 1;
        }
    }

    true
}

fn sorted_weights_compatible_on_domain_intersection(
    class_weights: &[(Label, Weight)],
    group_weights: &[(Label, Weight)],
    left_domain: &Weight,
    right_domain: &Weight,
) -> bool {
    let mut class_idx = 0;
    let mut group_idx = 0;

    while class_idx < class_weights.len() && group_idx < group_weights.len() {
        let (class_label, class_weight) = &class_weights[class_idx];
        let (group_label, group_weight) = &group_weights[group_idx];
        if class_label == group_label {
            if class_weight != group_weight {
                let class_disjoint = weight_is_disjoint_from_domain_intersection(
                    class_weight,
                    left_domain,
                    right_domain,
                );
                let group_disjoint = weight_is_disjoint_from_domain_intersection(
                    group_weight,
                    left_domain,
                    right_domain,
                );
                if class_disjoint != group_disjoint {
                    return false;
                }
                if !class_disjoint
                    && !weights_equal_on_domain_intersection(
                        class_weight,
                        group_weight,
                        left_domain,
                        right_domain,
                    )
                {
                    return false;
                }
            }
            class_idx += 1;
            group_idx += 1;
        } else if class_label < group_label {
            if !weight_is_disjoint_from_domain_intersection(class_weight, left_domain, right_domain) {
                return false;
            }
            class_idx += 1;
        } else {
            if !weight_is_disjoint_from_domain_intersection(group_weight, left_domain, right_domain) {
                return false;
            }
            group_idx += 1;
        }
    }

    for (_, class_weight) in &class_weights[class_idx..] {
        if !weight_is_disjoint_from_domain_intersection(class_weight, left_domain, right_domain) {
            return false;
        }
    }
    for (_, group_weight) in &group_weights[group_idx..] {
        if !weight_is_disjoint_from_domain_intersection(group_weight, left_domain, right_domain) {
            return false;
        }
    }

    true
}

/// Compare two sparse label→weight profiles on one already-materialized domain.
///
/// The group path compares many labels against the same overlap. Materializing
/// that overlap once is exact and avoids recomputing its TSID/token intersection
/// for every label.
fn sorted_weights_compatible_on_domain(
    class_weights: &[(Label, Weight)],
    group_weights: &[(Label, Weight)],
    domain: &Weight,
) -> bool {
    let mut class_idx = 0;
    let mut group_idx = 0;

    while class_idx < class_weights.len() && group_idx < group_weights.len() {
        let (class_label, class_weight) = &class_weights[class_idx];
        let (group_label, group_weight) = &group_weights[group_idx];
        if class_label == group_label {
            if class_weight != group_weight {
                let class_disjoint = class_weight.is_disjoint(domain);
                let group_disjoint = group_weight.is_disjoint(domain);
                if class_disjoint != group_disjoint {
                    return false;
                }
                if !class_disjoint && !weights_equal_on_domain(class_weight, group_weight, domain) {
                    return false;
                }
            }
            class_idx += 1;
            group_idx += 1;
        } else if class_label < group_label {
            if !class_weight.is_disjoint(domain) {
                return false;
            }
            class_idx += 1;
        } else {
            if !group_weight.is_disjoint(domain) {
                return false;
            }
            group_idx += 1;
        }
    }

    for (_, class_weight) in &class_weights[class_idx..] {
        if !class_weight.is_disjoint(domain) {
            return false;
        }
    }
    for (_, group_weight) in &group_weights[group_idx..] {
        if !group_weight.is_disjoint(domain) {
            return false;
        }
    }

    true
}

fn targets_compatible_with_group_map(
    class_targets: &[(Label, u32)],
    group_targets_by_label: &FxHashMap<Label, u32>,
) -> bool {
    class_targets.iter().all(|(label, target)| {
        group_targets_by_label
            .get(label)
            .is_none_or(|group_target| *group_target == *target)
    })
}

fn final_weights_compatible_on_domain_intersection(
    class_final_weight: Option<&Weight>,
    member_final_weight: Option<&Weight>,
    class_domain: &Weight,
    member_domain: &Weight,
) -> bool {
    match (class_final_weight, member_final_weight) {
        (Some(class_fw), Some(member_fw)) => weights_equal_on_domain_intersection(
            class_fw,
            member_fw,
            class_domain,
            member_domain,
        ),
        (Some(class_fw), None) => {
            weight_is_disjoint_from_domain_intersection(class_fw, class_domain, member_domain)
        }
        (None, Some(member_fw)) => {
            weight_is_disjoint_from_domain_intersection(member_fw, class_domain, member_domain)
        }
        (None, None) => true,
    }
}

fn final_weights_compatible_on_domain(
    class_final_weight: Option<&Weight>,
    group_final_weight: Option<&Weight>,
    domain: &Weight,
) -> bool {
    match (class_final_weight, group_final_weight) {
        (Some(class_weight), Some(group_weight)) => {
            weights_equal_on_domain(class_weight, group_weight, domain)
        }
        (Some(weight), None) | (None, Some(weight)) => weight.is_disjoint(domain),
        (None, None) => true,
    }
}

fn merge_sorted_targets(existing: &mut Vec<(Label, u32)>, add: &[(Label, u32)]) {
    if add.is_empty() {
        return;
    }
    if existing.is_empty() {
        existing.extend_from_slice(add);
        return;
    }

    let mut merged = Vec::with_capacity(existing.len() + add.len());
    let mut existing_idx = 0;
    let mut add_idx = 0;
    while existing_idx < existing.len() && add_idx < add.len() {
        let existing_entry = existing[existing_idx];
        let add_entry = add[add_idx];
        if existing_entry.0 == add_entry.0 {
            debug_assert_eq!(existing_entry.1, add_entry.1);
            merged.push(existing_entry);
            existing_idx += 1;
            add_idx += 1;
        } else if existing_entry.0 < add_entry.0 {
            merged.push(existing_entry);
            existing_idx += 1;
        } else {
            merged.push(add_entry);
            add_idx += 1;
        }
    }
    merged.extend_from_slice(&existing[existing_idx..]);
    merged.extend_from_slice(&add[add_idx..]);
    *existing = merged;
}

fn merge_sorted_weights(existing: &mut Vec<(Label, Weight)>, add: &[(Label, Weight)]) {
    if add.is_empty() {
        return;
    }
    if existing.is_empty() {
        existing.extend(add.iter().cloned());
        return;
    }

    let mut merged = Vec::with_capacity(existing.len() + add.len());
    let mut existing_idx = 0;
    let mut add_idx = 0;
    while existing_idx < existing.len() && add_idx < add.len() {
        let (existing_label, existing_weight) = &existing[existing_idx];
        let (add_label, add_weight) = &add[add_idx];
        if existing_label == add_label {
            let merged_weight = if existing_weight == add_weight {
                existing_weight.clone()
            } else {
                existing_weight.union(add_weight)
            };
            merged.push((*existing_label, merged_weight));
            existing_idx += 1;
            add_idx += 1;
        } else if existing_label < add_label {
            merged.push((*existing_label, existing_weight.clone()));
            existing_idx += 1;
        } else {
            merged.push((*add_label, add_weight.clone()));
            add_idx += 1;
        }
    }
    merged.extend(existing[existing_idx..].iter().cloned());
    merged.extend(add[add_idx..].iter().cloned());
    *existing = merged;
}

fn tsid_coverage_disjoint(
    left: &Option<RangeSetBlaze<u32>>,
    right: &Option<RangeSetBlaze<u32>>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left.is_disjoint(right),
        _ => false,
    }
}

fn merge_tsid_coverage(
    target: &mut Option<RangeSetBlaze<u32>>,
    add: &Option<RangeSetBlaze<u32>>,
) {
    match (&mut *target, add) {
        (None, _) | (_, None) => *target = None,
        (Some(target_set), Some(add_set)) => {
            *target_set = target_set.clone() | add_set.clone();
        }
    }
}

const TSID_MEMBER_INDEX_ENUMERATION_LIMIT: usize = 256;

fn enumerate_tsid_coverage_limited(
    coverage: &Option<RangeSetBlaze<u32>>,
    limit: usize,
    mut visit: impl FnMut(u32),
) -> bool {
    let Some(coverage) = coverage else {
        return false;
    };
    let mut count = 0usize;
    for range in coverage.ranges() {
        let mut tsid = *range.start();
        loop {
            count += 1;
            if count > limit {
                return false;
            }
            visit(tsid);
            if tsid == *range.end() {
                break;
            }
            tsid += 1;
        }
    }
    true
}

/// The indexed memberwise path is considerably cheaper when each incoming
/// class has a small TSID footprint.  Broad classes cannot use that index and
/// can force repeated scans of a large group, so after several such probes we
/// promote the group to the exact pointwise summary used by the summary path.
/// Dense indexed probes can be just as expensive.  Promote compact profiles
/// after enough actual overlap checks, rather than using wall-clock time: the
/// latter is host-load dependent and can misclassify very wide profiles whose
/// eager summary would be more expensive than their indexed checks.  Both
/// representations encode the same compatibility condition.
const SUMMARY_PROMOTION_MIN_MEMBERS: usize = 64;
const SUMMARY_PROMOTION_BROAD_PROBES: usize = 4;
const SUMMARY_WORK_PROMOTION_MAX_PROFILE_WEIGHTS: usize = 128;
const SUMMARY_PROMOTION_MEMBERWISE_OVERLAP_CHECKS: usize = 4_096;

/// A summary is an exact union over a stable prefix of a merge group.  Updating
/// that union one member at a time repeatedly normalizes the same growing
/// weights, which is quadratic for wide terminal-DWA groups.  Keep subsequent
/// members as an exact bounded suffix and rebuild the immutable snapshot in
/// batches.  Compatibility checks cover both pieces, so this changes only the
/// construction schedule, not the merge relation.
const SUMMARY_SNAPSHOT_BATCH_SIZE: usize = 64;

struct ExactGroupSummary {
    needed_union: Weight,
    merged_final_weight: Option<Weight>,
    transition_weights: Vec<(Label, Weight)>,
}

struct OverlapMergeGroup {
    targets_by_label: FxHashMap<Label, u32>,
    needed_tsid_coverage: Option<RangeSetBlaze<u32>>,
    indexed_members_by_tsid: FxHashMap<u32, Vec<usize>>,
    unindexed_member_classes: Vec<usize>,
    member_classes: Vec<usize>,
    broad_probe_count: usize,
    max_profile_weights: usize,
    memberwise_overlap_checks: usize,
    /// Exact aggregate for every member before `summary_pending_classes`.
    summary: Option<ExactGroupSummary>,
    /// Exact suffix not yet folded into the immutable aggregate.
    summary_pending_classes: Vec<usize>,
}

fn should_promote_group_summary(group: &OverlapMergeGroup) -> bool {
    group.member_classes.len() >= SUMMARY_PROMOTION_MIN_MEMBERS
        && (group.broad_probe_count >= SUMMARY_PROMOTION_BROAD_PROBES
            || (group.max_profile_weights <= SUMMARY_WORK_PROMOTION_MAX_PROFILE_WEIGHTS
                && group.memberwise_overlap_checks
                    >= SUMMARY_PROMOTION_MEMBERWISE_OVERLAP_CHECKS))
}

fn build_exact_group_summary(
    member_classes: &[usize],
    class_needed_union: &[Weight],
    class_profiles: &[ClassProfile],
) -> ExactGroupSummary {
    let needed_union = Weight::union_all(
        member_classes
            .iter()
            .map(|&class| &class_needed_union[class]),
    );

    let final_weights: Vec<&Weight> = member_classes
        .iter()
        .filter_map(|&class| class_profiles[class].final_weight.as_ref())
        .collect();
    let merged_final_weight =
        (!final_weights.is_empty()).then(|| Weight::union_all(final_weights));

    let mut weights_by_label: BTreeMap<Label, Vec<&Weight>> = BTreeMap::new();
    for &class in member_classes {
        for (label, weight) in &class_profiles[class].weights {
            weights_by_label.entry(*label).or_default().push(weight);
        }
    }
    let transition_weights = weights_by_label
        .into_iter()
        .map(|(label, weights)| (label, Weight::union_all(weights)))
        .collect();

    ExactGroupSummary {
        needed_union,
        merged_final_weight,
        transition_weights,
    }
}

fn update_exact_group_summary(
    summary: &mut ExactGroupSummary,
    needed: &Weight,
    profile: &ClassProfile,
) {
    if summary.needed_union != *needed {
        summary.needed_union = summary.needed_union.union(needed);
    }
    if let Some(final_weight) = &profile.final_weight {
        summary.merged_final_weight = Some(match summary.merged_final_weight.take() {
            Some(existing) if existing == *final_weight => existing,
            Some(existing) => existing.union(final_weight),
            None => final_weight.clone(),
        });
    }
    merge_sorted_weights(&mut summary.transition_weights, &profile.weights);
}

fn memberwise_group_compatible(
    class_domain: &Weight,
    class_profile: &ClassProfile,
    member_classes: &[usize],
    class_needed_union: &[Weight],
    class_profiles: &[ClassProfile],
) -> bool {
    member_classes.iter().all(|&member_class| {
        let member_domain = &class_needed_union[member_class];
        class_domain.is_disjoint(member_domain)
            || (final_weights_compatible_on_domain_intersection(
                class_profile.final_weight.as_ref(),
                class_profiles[member_class].final_weight.as_ref(),
                class_domain,
                member_domain,
            ) && sorted_weights_compatible_on_domain_intersection(
                &class_profile.weights,
                &class_profiles[member_class].weights,
                class_domain,
                member_domain,
            ))
    })
}

/// Hybrid coloring: partition refinement to reduce candidates into classes,
/// then graph coloring among class representatives.
///
/// States within the same partition-refinement class have identical transition
/// structure (same labels, mapped targets, weights) and are guaranteed compatible.
/// We only need to check pairwise compatibility among classes, reducing O(N²) to O(K²)
/// where K is the number of classes (typically K << N).
///
/// To ensure correctness, we compute the union of needed sets for each class
/// and use those when checking inter-class compatibility. This guarantees that
/// if two class representatives are deemed compatible, ALL pairs across the
/// two classes are compatible (since transitions/weights are identical within
/// a class, only the needed-set overlap domain varies).
fn build_and_color_pairwise_greedy(
    dwa: &DWA,
    candidates: &[usize],
    needed: &[Weight],
    old_to_new: &[u32],
    productive_transitions: &[Vec<ProductiveTransition>],
) -> Vec<usize> {
    // A merge group is valid when every pair of members agrees on the
    // intersection of its live domains and has no conflicting label target.
    // Checking the full clique directly is exact: pairwise agreement of these
    // partial deterministic behaviors implies one well-defined union behavior
    // for the whole group. For small height buckets this avoids constructing
    // the much larger pointwise profile representation merely to discover that
    // almost every successful merge is domain-disjoint.
    let mut groups = Vec::<SmallVec<[usize; 16]>>::new();
    let mut coloring = Vec::with_capacity(candidates.len());
    for &candidate in candidates {
        let group = groups
            .iter()
            .position(|members| {
                members.iter().all(|&member| {
                    are_compatible(
                        candidate,
                        member,
                        dwa,
                        needed,
                        old_to_new,
                        productive_transitions,
                        false,
                    )
                })
            })
            .unwrap_or_else(|| {
                groups.push(SmallVec::new());
                groups.len() - 1
            });
        groups[group].push(candidate);
        coloring.push(group);
    }
    coloring
}

fn build_and_color_hybrid(
    dwa: &DWA,
    candidates: &[usize],
    needed: &[Weight],
    old_to_new: &[u32],
    productive_transitions: &[Vec<ProductiveTransition>],
    pointwise_class_order: PointwiseClassOrder,
) -> HybridColoring {
    let profile_enabled = weighted_dwa_minimize_profile_enabled();
    if candidates.len() <= 64
        && std::env::var_os("GLRMASK_WEIGHTED_MINIMIZE_DISABLE_PAIRWISE_SMALL").is_none()
    {
        return HybridColoring::colors(build_and_color_pairwise_greedy(
            dwa,
            candidates,
            needed,
            old_to_new,
            productive_transitions,
        ));
    }
    let total_started_at = Instant::now();

    // Step 1: Partition refinement to get fine-grained classes.
    let partition_refine_started_at = Instant::now();
    let class_coloring = partition_refine_coloring_productive(
        candidates,
        dwa,
        productive_transitions,
        old_to_new,
    );
    let partition_refine_ms = partition_refine_started_at.elapsed().as_secs_f64() * 1000.0;
    let num_classes = class_coloring.iter().max().map(|&c| c + 1).unwrap_or(0);
    if num_classes <= 1 {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][weighted_dwa_minimize_hybrid] candidates={} classes={} groups={} partition_refine_ms={:.3} class_union_ms={:.3} class_profiles_ms={:.3} greedy_merge_ms={:.3} map_ms={:.3} total_ms={:.3}",
                candidates.len(),
                num_classes,
                num_classes,
                partition_refine_ms,
                0.0,
                0.0,
                0.0,
                0.0,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return HybridColoring::colors(class_coloring);
    }

    // Step 2: Pick one representative state per class and compute the union
    // of needed sets for each class.
    let class_union_started_at = Instant::now();
    let mut class_rep_state: Vec<usize> = vec![usize::MAX; num_classes];
    let mut class_needed_union: Vec<Weight> = Vec::with_capacity(num_classes);
    class_needed_union.resize_with(num_classes, Weight::empty);
    for (idx, &class) in class_coloring.iter().enumerate() {
        let state_id = candidates[idx];
        if class_rep_state[class] == usize::MAX {
            class_rep_state[class] = state_id;
        }
        class_needed_union[class] = class_needed_union[class].union(&needed[state_id]);
    }
    let class_union_ms = class_union_started_at.elapsed().as_secs_f64() * 1000.0;

    let class_profiles_started_at = Instant::now();
    let class_profiles: Vec<ClassProfile> = class_rep_state
        .iter()
        .map(|&rep| build_class_profile(rep, old_to_new, productive_transitions, dwa))
        .collect();
    let class_profiles_ms = class_profiles_started_at.elapsed().as_secs_f64() * 1000.0;
    if profile_enabled {
        let mut coverage_cells = 0usize;
        let mut max_coverage_cells = 0usize;
        let mut coverage_over_256 = 0usize;
        for weight in &class_needed_union {
            let count = weight
                .tsid_coverage()
                .as_ref()
                .map(|set| set.ranges().map(|range| (*range.end() as usize - *range.start() as usize) + 1).sum())
                .unwrap_or(usize::MAX);
            coverage_cells = coverage_cells.saturating_add(count);
            max_coverage_cells = max_coverage_cells.max(count);
            coverage_over_256 += usize::from(count > 256);
        }
        let total_profile_weight_ranges: usize = class_profiles
            .iter()
            .flat_map(|profile| profile.weights.iter())
            .map(|(_, weight)| weight.outer_range_count())
            .sum();
        let total_profile_weight_cells: usize = class_profiles
            .iter()
            .flat_map(|profile| profile.weights.iter())
            .map(|(_, weight)| weight.tsid_coverage().map(|set| set.ranges().map(|range| (*range.end() as usize - *range.start() as usize) + 1).sum()).unwrap_or(usize::MAX))
            .fold(0usize, usize::saturating_add);
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_hybrid_shape] candidates={} classes={} coverage_cells={} max_coverage_cells={} coverage_over_256={} total_profile_weight_ranges={} total_profile_weight_cells={}",
            candidates.len(), num_classes, coverage_cells, max_coverage_cells, coverage_over_256, total_profile_weight_ranges, total_profile_weight_cells,
        );
    }

    if std::env::var_os("GLRMASK_WEIGHTED_MINIMIZE_RECURSIVE_WITNESS").is_some() {
        let witness_started_at = Instant::now();
        let mut class_order = (0..num_classes).collect::<Vec<_>>();
        if pointwise_class_order == PointwiseClassOrder::DescendingDomain {
            class_order.sort_unstable_by_key(|&class| {
                std::cmp::Reverse((
                    class_needed_union[class].outer_range_count(),
                    class_profiles[class].targets.len(),
                    class,
                ))
            });
        }
        if let Some((class_to_group, proof_count, witness_count)) = recursive_witness_coloring(
            &class_profiles,
            &class_needed_union,
            &class_order,
        ) {
            let group_count = class_to_group
                .iter()
                .copied()
                .max()
                .map_or(0, |group| group + 1);
            let coloring = class_coloring
                .iter()
                .map(|class| class_to_group[*class])
                .collect::<Vec<_>>();

            if std::env::var_os("GLRMASK_VALIDATE_WEIGHTED_MINIMIZE_RECURSIVE_WITNESS")
                .is_some()
            {
                let reference = try_build_and_color_pointwise(
                    candidates,
                    &class_coloring,
                    &class_needed_union,
                    &class_profiles,
                    pointwise_class_order,
                    false,
                )
                .expect("pointwise reference coloring must be finite");
                let reference_groups = reference
                    .colors
                    .iter()
                    .copied()
                    .max()
                    .map_or(0, |group| group + 1);
                assert_eq!(
                    group_count, reference_groups,
                    "recursive witness quotient produced a different group count",
                );
            }

            if profile_enabled {
                eprintln!(
                    "[glrmask/profile][weighted_dwa_minimize_recursive_witness] candidates={} classes={} groups={} proofs={} witnesses={} total_ms={:.3}",
                    candidates.len(),
                    num_classes,
                    group_count,
                    proof_count,
                    witness_count,
                    witness_started_at.elapsed().as_secs_f64() * 1000.0,
                );
                eprintln!(
                    "[glrmask/profile][weighted_dwa_minimize_pointwise_preamble] candidates={} classes={} partition_refine_ms={:.3} class_union_ms={:.3} class_profiles_ms={:.3}",
                    candidates.len(),
                    num_classes,
                    partition_refine_ms,
                    class_union_ms,
                    class_profiles_ms,
                );
            }
            return HybridColoring::colors(coloring);
        }
    }

    if std::env::var_os("GLRMASK_WEIGHTED_MINIMIZE_EVENT_CONFLICT_GRAPH").is_some() {
        let event_graph_started_at = Instant::now();
        let mut class_order = (0..num_classes).collect::<Vec<_>>();
        if pointwise_class_order == PointwiseClassOrder::DescendingDomain {
            class_order.sort_unstable_by_key(|&class| {
                std::cmp::Reverse((
                    class_needed_union[class].outer_range_count(),
                    class_profiles[class].targets.len(),
                    class,
                ))
            });
        }
        if let Some((
            class_to_group,
            group_count,
            target_conflicts,
            intervals,
            behavior_conflict_pairs,
        )) = event_conflict_graph_coloring(
            &class_profiles,
            &class_needed_union,
            &class_order,
        ) {
            let coloring = class_coloring
                .iter()
                .map(|class| class_to_group[*class])
                .collect::<Vec<_>>();

            if std::env::var_os("GLRMASK_VALIDATE_WEIGHTED_MINIMIZE_EVENT_CONFLICT_GRAPH")
                .is_some()
            {
                assert_eq!(
                    pointwise_class_order,
                    PointwiseClassOrder::Stable,
                    "event-conflict validation currently requires stable class order",
                );
                let reference = try_build_and_color_pointwise(
                    candidates,
                    &class_coloring,
                    &class_needed_union,
                    &class_profiles,
                    pointwise_class_order,
                    false,
                )
                .expect("pointwise reference coloring must be finite");
                assert_eq!(
                    coloring, reference.colors,
                    "event conflict graph differs from pointwise behavior coloring",
                );
            }

            if profile_enabled {
                eprintln!(
                    "[glrmask/profile][weighted_dwa_minimize_event_conflicts] candidates={} classes={} groups={} target_conflicts={} intervals={} behavior_conflict_pairs={} total_ms={:.3}",
                    candidates.len(),
                    num_classes,
                    group_count,
                    target_conflicts,
                    intervals,
                    behavior_conflict_pairs,
                    event_graph_started_at.elapsed().as_secs_f64() * 1000.0,
                );
                eprintln!(
                    "[glrmask/profile][weighted_dwa_minimize_pointwise_preamble] candidates={} classes={} partition_refine_ms={:.3} class_union_ms={:.3} class_profiles_ms={:.3}",
                    candidates.len(),
                    num_classes,
                    partition_refine_ms,
                    class_union_ms,
                    class_profiles_ms,
                );
            }
            return HybridColoring::colors(coloring);
        }
    }

    if std::env::var_os("GLRMASK_WEIGHTED_MINIMIZE_DOMAIN_CONFLICT_GRAPH").is_some() {
        let domain_graph_started_at = Instant::now();
        let mut class_order = (0..num_classes).collect::<Vec<_>>();
        if pointwise_class_order == PointwiseClassOrder::DescendingDomain {
            class_order.sort_unstable_by_key(|&class| {
                std::cmp::Reverse((
                    class_needed_union[class].outer_range_count(),
                    class_profiles[class].targets.len(),
                    class,
                ))
            });
        }
        let (
            class_to_group,
            group_count,
            target_conflicts,
            overlap_pairs,
            behavior_conflicts,
        ) = domain_conflict_graph_coloring(
            &class_profiles,
            &class_needed_union,
            &class_order,
        );
        let coloring = class_coloring
            .iter()
            .map(|class| class_to_group[*class])
            .collect::<Vec<_>>();

        if std::env::var_os("GLRMASK_VALIDATE_WEIGHTED_MINIMIZE_DOMAIN_CONFLICT_GRAPH")
            .is_some()
        {
            assert_eq!(
                pointwise_class_order,
                PointwiseClassOrder::Stable,
                "domain-conflict validation currently requires stable class order",
            );
            let reference = try_build_and_color_pointwise(
                candidates,
                &class_coloring,
                &class_needed_union,
                &class_profiles,
                pointwise_class_order,
                false,
            )
            .expect("pointwise reference coloring must be finite");
            assert_eq!(
                coloring, reference.colors,
                "domain conflict graph differs from pointwise behavior coloring",
            );
        }

        if profile_enabled {
            eprintln!(
                "[glrmask/profile][weighted_dwa_minimize_domain_conflicts] candidates={} classes={} groups={} target_conflicts={} overlap_pairs={} behavior_conflicts={} total_ms={:.3}",
                candidates.len(),
                num_classes,
                group_count,
                target_conflicts,
                overlap_pairs,
                behavior_conflicts,
                domain_graph_started_at.elapsed().as_secs_f64() * 1000.0,
            );
            eprintln!(
                "[glrmask/profile][weighted_dwa_minimize_pointwise_preamble] candidates={} classes={} partition_refine_ms={:.3} class_union_ms={:.3} class_profiles_ms={:.3}",
                candidates.len(),
                num_classes,
                partition_refine_ms,
                class_union_ms,
                class_profiles_ms,
            );
        }
        return HybridColoring::colors(coloring);
    }

    // Each class is a partial behavior function over its pushed-needed
    // TSID/token domain. When that representation is finite, compare against
    // the exact union function of each group instead of revisiting all members.
    // A full sentinel cannot be enumerated, so it retains the generic path.
    let estimated_pointwise_work = class_profiles
        .iter()
        .flat_map(|profile| profile.weights.iter())
        .map(|(_, weight)| weight.outer_range_count())
        .sum::<usize>();
    let prefer_overlap_indexed_merge = num_classes <= 64 && estimated_pointwise_work >= 8_192;
    let pointwise_coloring = if prefer_overlap_indexed_merge {
        None
    } else if pointwise_tsid_ranges_enabled() {
        try_build_and_color_pointwise_ranges(
            candidates,
            &class_coloring,
            &class_needed_union,
            &class_profiles,
            pointwise_class_order,
            profile_enabled,
        )
        .map(HybridColoring::colors)
    } else {
        try_build_and_color_pointwise(
            candidates,
            &class_coloring,
            &class_needed_union,
            &class_profiles,
            pointwise_class_order,
            profile_enabled,
        )
    };
    if profile_enabled && prefer_overlap_indexed_merge {
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_overlap_choice] candidates={} classes={} estimated_pointwise_work={} strategy=overlap_indexed",
            candidates.len(), num_classes, estimated_pointwise_work,
        );
    }
    if let Some(coloring) = pointwise_coloring {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][weighted_dwa_minimize_pointwise_preamble] candidates={} classes={} partition_refine_ms={:.3} class_union_ms={:.3} class_profiles_ms={:.3}",
                candidates.len(),
                num_classes,
                partition_refine_ms,
                class_union_ms,
                class_profiles_ms,
            );
        }
        return coloring;
    }

    // The generic fallback needs per-class TSID coverage for its indexed
    // overlap probes. Do not build it on the normal pointwise-success path.
    let class_tsid_coverage: Vec<Option<RangeSetBlaze<u32>>> = class_needed_union
        .iter()
        .map(Weight::tsid_coverage)
        .collect();
    let classes_with_final_weight = class_profiles
        .iter()
        .filter(|profile| profile.final_weight.is_some())
        .count();
    let min_targets = class_profiles
        .iter()
        .map(|profile| profile.targets.len())
        .min()
        .unwrap_or(0);
    let max_targets = class_profiles
        .iter()
        .map(|profile| profile.targets.len())
        .max()
        .unwrap_or(0);
    let avg_targets = class_profiles
        .iter()
        .map(|profile| profile.targets.len() as f64)
        .sum::<f64>()
        / num_classes as f64;
    let min_weights = class_profiles
        .iter()
        .map(|profile| profile.weights.len())
        .min()
        .unwrap_or(0);
    let max_weights = class_profiles
        .iter()
        .map(|profile| profile.weights.len())
        .max()
        .unwrap_or(0);
    let avg_weights = class_profiles
        .iter()
        .map(|profile| profile.weights.len() as f64)
        .sum::<f64>()
        / num_classes as f64;

    // Step 3: Greedy merge of classes, handling both disjoint and overlapping
    // needed sets. Instead of building an O(K²) incompatibility graph, we check
    // each class against only the small number of existing groups (~14 for kb_684).
    //
    let greedy_merge_started_at = Instant::now();
    let mut groups: Vec<OverlapMergeGroup> = Vec::new();
    let mut group_attempts = 0usize;
    let mut target_checks = 0usize;
    let mut target_rejects = 0usize;
    let mut target_check_ms = 0.0;
    let mut disjoint_checks = 0usize;
    let mut disjoint_true = 0usize;
    let mut disjoint_check_ms = 0.0;
    let mut final_weight_checks = 0usize;
    let mut final_weight_rejects = 0usize;
    let mut final_weight_check_ms = 0.0;
    let mut transition_weight_checks = 0usize;
    let mut transition_weight_rejects = 0usize;
    let mut transition_weight_check_ms = 0.0;
    let mut group_update_ms = 0.0;
    let mut memberwise_indexed_probes = 0usize;
    let mut memberwise_broad_probes = 0usize;
    let mut memberwise_member_scans = 0usize;
    let mut memberwise_overlap_checks = 0usize;
    let mut summary_checks = 0usize;
    let mut summary_promotions = 0usize;
    let mut summary_work_promotions = 0usize;
    let mut summary_snapshot_rebuilds = 0usize;
    let mut summary_pending_member_scans = 0usize;
    let mut overlap_candidate_marks = vec![0u32; num_classes];
    let mut overlap_candidate_mark = 0u32;
    let mut overlap_candidate_members = Vec::<usize>::new();

    for class in 0..num_classes {
        let cn = &class_needed_union[class];
        let class_profile = &class_profiles[class];

        let mut placed = false;
        for g in &mut groups {
            group_attempts += 1;

            target_checks += 1;
            let target_check_started_at = Instant::now();
            let targets_compatible =
                targets_compatible_with_group_map(&class_profile.targets, &g.targets_by_label);
            target_check_ms += target_check_started_at.elapsed().as_secs_f64() * 1000.0;
            if !targets_compatible {
                target_rejects += 1;
                continue;
            }

            disjoint_checks += 1;
            let disjoint_check_started_at = Instant::now();
            let compatible = if let Some(summary) = g.summary.as_ref() {
                summary_checks += 1;
                let is_disjoint = cn.is_disjoint(&summary.needed_union);
                if is_disjoint {
                    disjoint_true += 1;
                    true
                } else {
                    let overlap = cn.intersection(&summary.needed_union);
                    debug_assert!(!overlap.is_empty());

                    final_weight_checks += 1;
                    let final_weight_check_started_at = Instant::now();
                    let mut summary_compatible = final_weights_compatible_on_domain(
                        class_profile.final_weight.as_ref(),
                        summary.merged_final_weight.as_ref(),
                        &overlap,
                    );
                    final_weight_check_ms +=
                        final_weight_check_started_at.elapsed().as_secs_f64() * 1000.0;
                    if !summary_compatible {
                        final_weight_rejects += 1;
                    }

                    if summary_compatible {
                        transition_weight_checks += 1;
                        let transition_weight_check_started_at = Instant::now();
                        summary_compatible = sorted_weights_compatible_on_domain(
                            &class_profile.weights,
                            &summary.transition_weights,
                            &overlap,
                        );
                        transition_weight_check_ms +=
                            transition_weight_check_started_at.elapsed().as_secs_f64() * 1000.0;
                        if !summary_compatible {
                            transition_weight_rejects += 1;
                        }
                    }

                    // The immutable summary covers the stable prefix.  Check the
                    // bounded suffix memberwise until the next snapshot rebuild.
                    // This is exact: a candidate is compatible with the whole
                    // group iff it is compatible with the prefix union and every
                    // suffix member on their respective overlap domains.
                    if summary_compatible {
                        for &member_class in &g.summary_pending_classes {
                            summary_pending_member_scans += 1;
                            memberwise_member_scans += 1;
                            let member_needed = &class_needed_union[member_class];
                            if cn.is_disjoint(member_needed) {
                                continue;
                            }
                            memberwise_overlap_checks += 1;

                            final_weight_checks += 1;
                            let final_weight_check_started_at = Instant::now();
                            let final_weight_ok = final_weights_compatible_on_domain_intersection(
                                class_profile.final_weight.as_ref(),
                                class_profiles[member_class].final_weight.as_ref(),
                                cn,
                                member_needed,
                            );
                            final_weight_check_ms +=
                                final_weight_check_started_at.elapsed().as_secs_f64() * 1000.0;
                            if !final_weight_ok {
                                final_weight_rejects += 1;
                                summary_compatible = false;
                                break;
                            }

                            transition_weight_checks += 1;
                            let transition_weight_check_started_at = Instant::now();
                            let transition_weights_ok =
                                sorted_weights_compatible_on_domain_intersection(
                                    &class_profile.weights,
                                    &class_profiles[member_class].weights,
                                    cn,
                                    member_needed,
                                );
                            transition_weight_check_ms += transition_weight_check_started_at
                                .elapsed()
                                .as_secs_f64()
                                * 1000.0;
                            if !transition_weights_ok {
                                transition_weight_rejects += 1;
                                summary_compatible = false;
                                break;
                            }
                        }
                    }

                    #[cfg(debug_assertions)]
                    debug_assert_eq!(
                        summary_compatible,
                        memberwise_group_compatible(
                            cn,
                            class_profile,
                            &g.member_classes,
                            &class_needed_union,
                            &class_profiles,
                        ),
                        "group summary plus pending suffix must be equivalent to checking every member",
                    );
                    summary_compatible
                }
            } else {
                overlap_candidate_members.clear();
                overlap_candidate_mark = overlap_candidate_mark.wrapping_add(1);
                if overlap_candidate_mark == 0 {
                    overlap_candidate_marks.fill(0);
                    overlap_candidate_mark = 1;
                }

                for &member_class in &g.unindexed_member_classes {
                    overlap_candidate_marks[member_class] = overlap_candidate_mark;
                    overlap_candidate_members.push(member_class);
                }
                let indexed = enumerate_tsid_coverage_limited(
                    &class_tsid_coverage[class],
                    TSID_MEMBER_INDEX_ENUMERATION_LIMIT,
                    |tsid| {
                        if let Some(members) = g.indexed_members_by_tsid.get(&tsid) {
                            for &member_class in members {
                                if overlap_candidate_marks[member_class] != overlap_candidate_mark {
                                    overlap_candidate_marks[member_class] = overlap_candidate_mark;
                                    overlap_candidate_members.push(member_class);
                                }
                            }
                        }
                    },
                );

                let scan_members: &[usize] = if indexed {
                    memberwise_indexed_probes += 1;
                    if overlap_candidate_members.is_empty() {
                        disjoint_true += 1;
                    }
                    &overlap_candidate_members
                } else {
                    memberwise_broad_probes += 1;
                    g.broad_probe_count += 1;
                    if tsid_coverage_disjoint(
                        &class_tsid_coverage[class],
                        &g.needed_tsid_coverage,
                    ) {
                        disjoint_true += 1;
                        &[]
                    } else {
                        &g.member_classes
                    }
                };

                let mut saw_overlap = false;
                let mut memberwise_compatible = true;
                for &member_class in scan_members {
                    memberwise_member_scans += 1;
                    let member_needed = &class_needed_union[member_class];
                    if cn.is_disjoint(member_needed) {
                        continue;
                    }
                    saw_overlap = true;
                    g.memberwise_overlap_checks += 1;
                    memberwise_overlap_checks += 1;

                    final_weight_checks += 1;
                    let final_weight_check_started_at = Instant::now();
                    let final_weight_ok = final_weights_compatible_on_domain_intersection(
                        class_profile.final_weight.as_ref(),
                        class_profiles[member_class].final_weight.as_ref(),
                        cn,
                        member_needed,
                    );
                    final_weight_check_ms +=
                        final_weight_check_started_at.elapsed().as_secs_f64() * 1000.0;
                    if !final_weight_ok {
                        final_weight_rejects += 1;
                        memberwise_compatible = false;
                        break;
                    }

                    transition_weight_checks += 1;
                    let transition_weight_check_started_at = Instant::now();
                    let transition_weights_ok = sorted_weights_compatible_on_domain_intersection(
                        &class_profile.weights,
                        &class_profiles[member_class].weights,
                        cn,
                        member_needed,
                    );
                    transition_weight_check_ms +=
                        transition_weight_check_started_at.elapsed().as_secs_f64() * 1000.0;
                    if !transition_weights_ok {
                        transition_weight_rejects += 1;
                        memberwise_compatible = false;
                        break;
                    }
                }
                if !saw_overlap && !scan_members.is_empty() {
                    disjoint_true += 1;
                }
                memberwise_compatible
            };
            if g.summary.is_none() && should_promote_group_summary(g) {
                let promoted_by_work = g.broad_probe_count < SUMMARY_PROMOTION_BROAD_PROBES;
                g.summary = Some(build_exact_group_summary(
                    &g.member_classes,
                    &class_needed_union,
                    &class_profiles,
                ));
                summary_promotions += 1;
                summary_work_promotions += usize::from(promoted_by_work);
            }
            disjoint_check_ms += disjoint_check_started_at.elapsed().as_secs_f64() * 1000.0;
            if !compatible {
                continue;
            }

            // Compatible — merge into this group
            let group_update_started_at = Instant::now();
            for (label, target) in &class_profile.targets {
                if let Some(existing_target) = g.targets_by_label.get(label) {
                    debug_assert_eq!(*existing_target, *target);
                } else {
                    g.targets_by_label.insert(*label, *target);
                }
            }
            merge_tsid_coverage(
                &mut g.needed_tsid_coverage,
                &class_tsid_coverage[class],
            );
            if !enumerate_tsid_coverage_limited(
                &class_tsid_coverage[class],
                TSID_MEMBER_INDEX_ENUMERATION_LIMIT,
                |tsid| g.indexed_members_by_tsid.entry(tsid).or_default().push(class),
            ) {
                g.unindexed_member_classes.push(class);
            }
            g.member_classes.push(class);
            g.max_profile_weights = g.max_profile_weights.max(class_profile.weights.len());
            if g.summary.is_some() {
                g.summary_pending_classes.push(class);
                if g.summary_pending_classes.len() >= SUMMARY_SNAPSHOT_BATCH_SIZE {
                    g.summary = Some(build_exact_group_summary(
                        &g.member_classes,
                        &class_needed_union,
                        &class_profiles,
                    ));
                    g.summary_pending_classes.clear();
                    summary_snapshot_rebuilds += 1;
                }
            } else if should_promote_group_summary(g) {
                g.summary = Some(build_exact_group_summary(
                    &g.member_classes,
                    &class_needed_union,
                    &class_profiles,
                ));
                summary_promotions += 1;
                summary_work_promotions += usize::from(
                    g.broad_probe_count < SUMMARY_PROMOTION_BROAD_PROBES,
                );
            }
            group_update_ms += group_update_started_at.elapsed().as_secs_f64() * 1000.0;
            placed = true;
            break;
        }

        if !placed {
            let mut targets_by_label = FxHashMap::default();
            targets_by_label.reserve(class_profile.targets.len());
            for (label, target) in &class_profile.targets {
                targets_by_label.insert(*label, *target);
            }

            groups.push(OverlapMergeGroup {
                targets_by_label,
                needed_tsid_coverage: class_tsid_coverage[class].clone(),
                indexed_members_by_tsid: {
                    let mut by_tsid = FxHashMap::default();
                    if enumerate_tsid_coverage_limited(
                        &class_tsid_coverage[class],
                        TSID_MEMBER_INDEX_ENUMERATION_LIMIT,
                        |tsid| by_tsid.entry(tsid).or_insert_with(Vec::new).push(class),
                    ) {
                        by_tsid
                    } else {
                        FxHashMap::default()
                    }
                },
                unindexed_member_classes: {
                    if enumerate_tsid_coverage_limited(
                        &class_tsid_coverage[class],
                        TSID_MEMBER_INDEX_ENUMERATION_LIMIT,
                        |_| {},
                    ) {
                        Vec::new()
                    } else {
                        vec![class]
                    }
                },
                member_classes: vec![class],
                broad_probe_count: 0,
                max_profile_weights: class_profile.weights.len(),
                memberwise_overlap_checks: 0,
                summary: None,
                summary_pending_classes: Vec::new(),
            });
        }
    }

    let greedy_merge_ms = greedy_merge_started_at.elapsed().as_secs_f64() * 1000.0;
    let min_group_size = groups
        .iter()
        .map(|group| group.member_classes.len())
        .min()
        .unwrap_or(0);
    let max_group_size = groups
        .iter()
        .map(|group| group.member_classes.len())
        .max()
        .unwrap_or(0);

    // Step 4: Map each candidate through class -> merged color.
    let map_started_at = Instant::now();
    let mut class_to_group = vec![0usize; num_classes];
    for (gid, g) in groups.iter().enumerate() {
        for &c in &g.member_classes {
            class_to_group[c] = gid;
        }
    }
    let nc = candidates.len();
    let mut coloring = vec![0usize; nc];
    for (idx, &class) in class_coloring.iter().enumerate() {
        coloring[idx] = class_to_group[class];
    }

    let map_ms = map_started_at.elapsed().as_secs_f64() * 1000.0;

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_hybrid] candidates={} classes={} groups={} partition_refine_ms={:.3} class_union_ms={:.3} class_profiles_ms={:.3} greedy_merge_ms={:.3} map_ms={:.3} total_ms={:.3}",
            candidates.len(),
            num_classes,
            groups.len(),
            partition_refine_ms,
            class_union_ms,
            class_profiles_ms,
            greedy_merge_ms,
            map_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_hybrid_detail] candidates={} classes={} groups={} group_attempts={} target_checks={} target_rejects={} target_check_ms={:.3} disjoint_checks={} disjoint_true={} disjoint_check_ms={:.3} final_weight_checks={} final_weight_rejects={} final_weight_check_ms={:.3} transition_weight_checks={} transition_weight_rejects={} transition_weight_check_ms={:.3} group_update_ms={:.3} memberwise_indexed_probes={} memberwise_broad_probes={} memberwise_member_scans={} memberwise_overlap_checks={} summary_checks={} summary_promotions={} summary_work_promotions={} summary_snapshot_rebuilds={} summary_pending_member_scans={} classes_with_final_weight={} min_targets={} max_targets={} avg_targets={:.3} min_weights={} max_weights={} avg_weights={:.3} min_group_size={} max_group_size={}",
            candidates.len(),
            num_classes,
            groups.len(),
            group_attempts,
            target_checks,
            target_rejects,
            target_check_ms,
            disjoint_checks,
            disjoint_true,
            disjoint_check_ms,
            final_weight_checks,
            final_weight_rejects,
            final_weight_check_ms,
            transition_weight_checks,
            transition_weight_rejects,
            transition_weight_check_ms,
            group_update_ms,
            memberwise_indexed_probes,
            memberwise_broad_probes,
            memberwise_member_scans,
            memberwise_overlap_checks,
            summary_checks,
            summary_promotions,
            summary_work_promotions,
            summary_snapshot_rebuilds,
            summary_pending_member_scans,
            classes_with_final_weight,
            min_targets,
            max_targets,
            avg_targets,
            min_weights,
            max_weights,
            avg_weights,
            min_group_size,
            max_group_size,
        );
    }

    HybridColoring::colors(coloring)
}

fn partition_refine_coloring_productive(
    candidates: &[usize],
    dwa: &DWA,
    productive_transitions: &[Vec<ProductiveTransition>],
    old_to_new: &[u32],
) -> Vec<usize> {
    use std::hash::{Hash, Hasher};
    use rustc_hash::FxHasher;

    let nc = candidates.len();
    let mut hash_groups: rustc_hash::FxHashMap<u64, Vec<usize>> =
        rustc_hash::FxHashMap::default();

    for idx in 0..nc {
        let c = candidates[idx];
        let mut hasher = FxHasher::default();
        dwa.states()[c].final_weight.hash(&mut hasher);
        for transition in &productive_transitions[c] {
            let Some(mapped) = mapped_target(old_to_new, transition.target) else {
                continue;
            };
            transition.label.hash(&mut hasher);
            mapped.hash(&mut hasher);
            transition.weight.hash(&mut hasher);
        }
        productive_transitions[c].len().hash(&mut hasher);

        let sig = hasher.finish();
        hash_groups.entry(sig).or_default().push(idx);
    }

    let mut colors = vec![0usize; nc];
    let mut next_color = 0;

    for group in hash_groups.values() {
        if group.len() == 1 {
            colors[group[0]] = next_color;
            next_color += 1;
            continue;
        }

        // Verify within group for hash collisions
        let mut sub_groups: Vec<Vec<usize>> = Vec::new();
        'outer: for &idx in group {
            let c = candidates[idx];
            for sub in &mut sub_groups {
                let rep = candidates[sub[0]];
                if states_productive_equal(
                    c,
                    rep,
                    dwa,
                    productive_transitions,
                    old_to_new,
                ) {
                    sub.push(idx);
                    continue 'outer;
                }
            }
            sub_groups.push(vec![idx]);
        }

        for sub in &sub_groups {
            let color = next_color;
            next_color += 1;
            for &idx in sub {
                colors[idx] = color;
            }
        }
    }

    colors
}

/// Check if two states have identical productive signatures.
fn states_productive_equal(
    u: usize,
    v: usize,
    dwa: &DWA,
    productive_transitions: &[Vec<ProductiveTransition>],
    old_to_new: &[u32],
) -> bool {
    if dwa.states()[u].final_weight != dwa.states()[v].final_weight {
        return false;
    }
    let su = &productive_transitions[u];
    let sv = &productive_transitions[v];
    if su.len() != sv.len() {
        return false;
    }
    for (left, right) in su.iter().zip(sv) {
        if left.label != right.label {
            return false;
        }
        if mapped_target(old_to_new, left.target) != mapped_target(old_to_new, right.target) {
            return false;
        }
        if left.weight != right.weight {
            return false;
        }
    }
    true
}

fn try_all_compatible_height_0_coloring(
    candidates: &[usize],
    dwa: &DWA,
    _needed: &[Weight],
) -> Option<Vec<usize>> {
    // After push_weights, all leaf states (h=0, no transitions) are always
    // mutually compatible.  Proof:
    //
    //   For leaves, needed[u] = final_weight[u] because there are no outgoing
    //   transitions.
    //
    //   As we merge leaf states into one witness group, witness_domain and
    //   witness_final stay equal: both are the union of the same member final
    //   weights, so they grow in lockstep as each new leaf is added.
    //
    //   The compatibility check compares:
    //     witness_final ∩ overlap  vs  candidate_final ∩ overlap
    //   where overlap = witness_domain ∩ needed_candidate.
    //
    //   Since witness = witness_domain = witness_final:
    //     W ∩ (W ∩ F) = W ∩ F = F ∩ (W ∩ F)
    //   (by idempotency of set intersection), so both sides are always equal.
    //
    // Therefore we can merge all h=0 candidates into a single color without
    // checking pairwise compatibility.

    if candidates.len() <= 1 {
        return None;
    }
    if !candidates
        .iter()
        .all(|&id| dwa.states()[id].transitions.is_empty())
    {
        return None;
    }

    Some(vec![0; candidates.len()])
}

// Merge and reconstruct.

struct MergedStateBuilder {
    final_weights_pending: Vec<Weight>,
    transitions_pending: rustc_hash::FxHashMap<Label, (u32, Vec<Weight>)>,
}

impl Default for MergedStateBuilder {
    fn default() -> Self {
        Self {
            final_weights_pending: Vec::new(),
            transitions_pending: rustc_hash::FxHashMap::default(),
        }
    }
}

impl MergedStateBuilder {
    fn add_final_weight(&mut self, weight: &Weight) {
        self.final_weights_pending.push(weight.clone());
    }

    fn add_transition(&mut self, label: Label, target: u32, weight: Weight) {
        use std::collections::hash_map::Entry;
        match self.transitions_pending.entry(label) {
            Entry::Occupied(mut entry) => {
                let (existing_target, pending_weights) = entry.get_mut();
                debug_assert_eq!(*existing_target, target);
                pending_weights.push(weight);
            }
            Entry::Vacant(entry) => {
                entry.insert((target, vec![weight]));
            }
        }
    }

}

/// Build a reconstructed weight in one exact multiway sweep.
///
/// `Weight::union_all` already has a dedicated event-sweep implementation for
/// wide unions. Feeding it a tree of bounded intermediate unions adds repeated
/// sorting, allocation, and range coalescing without reducing the final work.
fn batch_build_weight(pending: Vec<Weight>, automatic_large_minimizer: bool) -> Weight {
    if reconstruction_token_range_coalescing_enabled(automatic_large_minimizer, pending.len()) {
        Weight::union_all_reconstruction(pending.iter())
    } else {
        Weight::union_all(pending.iter())
    }
}

fn merge_state_into_builder(
    old_id: usize,
    color: usize,
    dwa: &DWA,
    old_to_new: &[u32],
    builders: &mut [MergedStateBuilder],
) {
    let builder = &mut builders[color];
    let old_state = &dwa.states()[old_id];

    // Union final weights
    if let Some(fw) = &old_state.final_weight {
        builder.add_final_weight(fw);
    }

    // Merge transitions
    let n = dwa.states().len();
    for (&label, (target_raw, w_orig)) in &old_state.transitions {
        let t = *target_raw as usize;
        if t >= n {
            continue;
        }
        let target_new = old_to_new[t];
        if target_new == UNMAPPED {
            continue;
        }
        // After push_weights, w_orig is already restricted to reachable[target].
        // The merged target's needed = union(reachable[s] for all s merged into target_new),
        // which is a superset of reachable[target]. So w_orig ∩ merged_needed = w_orig.
        if !w_orig.is_empty() {
            builder.add_transition(label, target_new, w_orig.clone());
        }
    }
}

fn merge_productive_state_into_builder(
    old_id: usize,
    color: usize,
    dwa: &DWA,
    productive_transitions: &[Vec<ProductiveTransition>],
    old_to_new: &[u32],
    builders: &mut [MergedStateBuilder],
) {
    let builder = &mut builders[color];
    let old_state = &dwa.states()[old_id];
    if let Some(final_weight) = &old_state.final_weight {
        builder.add_final_weight(final_weight);
    }
    for transition in &productive_transitions[old_id] {
        let target = mapped_target(old_to_new, transition.target)
            .expect("productive transition target must be colored first");
        builder.add_transition(transition.label, target, transition.weight.clone());
    }
}

fn reconstruct_dwa(
    start_old: usize,
    old_to_new: &[u32],
    builders: Vec<MergedStateBuilder>,
    automatic_large_minimizer: bool,
) -> DWA {
    let profile_enabled = weighted_dwa_minimize_profile_enabled();
    let mut final_pending_weight_count = 0usize;
    let mut max_final_pending_weight_count = 0usize;
    let mut final_batches_over_16 = 0usize;
    let mut transition_batch_count = 0usize;
    let mut transition_pending_weight_count = 0usize;
    let mut max_transition_pending_weight_count = 0usize;
    let mut transition_batches_over_16 = 0usize;
    let mut final_union_ms = 0.0;
    let mut transition_union_ms = 0.0;
    let mut insert_ms = 0.0;
    let states: Vec<DWAState> = builders
        .into_iter()
        .map(|b| {
            let mut state = DWAState::default();
            final_pending_weight_count += b.final_weights_pending.len();
            max_final_pending_weight_count = max_final_pending_weight_count.max(b.final_weights_pending.len());
            final_batches_over_16 += usize::from(b.final_weights_pending.len() > 16);
            let final_union_started_at = Instant::now();
            let final_weight = batch_build_weight(
                b.final_weights_pending,
                automatic_large_minimizer,
            );
            final_union_ms += final_union_started_at.elapsed().as_secs_f64() * 1000.0;
            if !final_weight.is_empty() {
                state.final_weight = Some(final_weight);
            }
            for (lbl, (target, pending_weights)) in b.transitions_pending {
                transition_batch_count += 1;
                transition_pending_weight_count += pending_weights.len();
                max_transition_pending_weight_count = max_transition_pending_weight_count.max(pending_weights.len());
                transition_batches_over_16 += usize::from(pending_weights.len() > 16);
                let transition_union_started_at = Instant::now();
                let weight = batch_build_weight(pending_weights, automatic_large_minimizer);
                transition_union_ms += transition_union_started_at.elapsed().as_secs_f64() * 1000.0;
                if !weight.is_empty() {
                    let insert_started_at = Instant::now();
                    state.transitions.insert(lbl, (target, weight));
                    insert_ms += insert_started_at.elapsed().as_secs_f64() * 1000.0;
                }
            }
            state
        })
        .collect();

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_reconstruct] output_states={} final_pending_weights={} max_final_pending_weights={} final_batches_over_16={} final_union_ms={:.3} transition_batches={} transition_pending_weights={} max_transition_pending_weights={} transition_batches_over_16={} transition_union_ms={:.3} insert_ms={:.3}",
            states.len(),
            final_pending_weight_count,
            max_final_pending_weight_count,
            final_batches_over_16,
            final_union_ms,
            transition_batch_count,
            transition_pending_weight_count,
            max_transition_pending_weight_count,
            transition_batches_over_16,
            transition_union_ms,
            insert_ms,
        );
    }

    let start_new = old_to_new[start_old];
    DWA::from_parts(
        states,
        if start_new == UNMAPPED { 0 } else { start_new },
    )
}

fn canonical_dead_dwa() -> DWA {
    DWA::new(0, 0)
}

/// Exact small-input path that never materializes a transient pushed DWA.
///
/// The ordinary pipeline computes each productive transition, writes it back
/// into the state's BTreeMap, scans those maps again to build productive
/// profiles, and finally clones the same weights into reconstruction builders.
/// For small height buckets we already use exact pairwise coloring, so the
/// pushed maps are not needed for partition refinement. Keep the productive
/// transitions in one compact side vector and reconstruct directly from it.
fn try_minimize_small_pairwise_direct(dwa: &DWA) -> Option<DWA> {
    const MAX_INPUT_STATES: usize = 128;
    const MAX_HEIGHT_BUCKET: usize = 64;

    let n = dwa.states().len();
    if n == 0 || n > MAX_INPUT_STATES {
        return None;
    }
    let started_at = Instant::now();
    let topo = compute_topo_order(dwa)?;

    let mut incoming_transition_counts = vec![0usize; n];
    for state in dwa.states() {
        for (target, _) in state.transitions.values() {
            if (*target as usize) < n {
                incoming_transition_counts[*target as usize] += 1;
            }
        }
    }

    let mut needed = vec![Weight::empty(); n];
    let mut productive_transitions = vec![Vec::<ProductiveTransition>::new(); n];
    let mut intersection_cache = FxHashMap::<(usize, usize), Weight>::default();
    let mut intersection_indexes = FxHashMap::<usize, WeightIntersectionIndex>::default();
    for &source in topo.iter().rev() {
        let state = &dwa.states()[source];
        let mut reachable_parts = Vec::<Weight>::with_capacity(state.transitions.len() + 1);
        let mut reachable_is_full = false;
        if let Some(final_weight) = &state.final_weight {
            if final_weight.is_full() {
                reachable_is_full = true;
            } else if !final_weight.is_empty() {
                reachable_parts.push(final_weight.clone());
            }
        }

        let mut productive = Vec::with_capacity(state.transitions.len());
        for (&label, (target, weight)) in &state.transitions {
            let target_index = *target as usize;
            if target_index >= n || needed[target_index].is_empty() {
                continue;
            }
            let pushed_weight = if needed[target_index].is_full() {
                weight.clone()
            } else {
                let index = (incoming_transition_counts[target_index] >= 8).then(|| {
                    let key = needed[target_index].ptr_key();
                    intersection_indexes
                        .entry(key)
                        .or_insert_with(|| needed[target_index].intersection_index())
                });
                memoized_intersection(
                    &mut intersection_cache,
                    weight,
                    &needed[target_index],
                    index.as_deref(),
                )
            };
            if pushed_weight.is_empty() {
                continue;
            }
            if !reachable_is_full {
                if pushed_weight.is_full() {
                    reachable_is_full = true;
                    reachable_parts.clear();
                } else {
                    reachable_parts.push(pushed_weight.clone());
                }
            }
            productive.push(ProductiveTransition {
                label,
                target: *target,
                weight: pushed_weight,
            });
        }
        productive_transitions[source] = productive;
        needed[source] = if reachable_is_full {
            Weight::all()
        } else {
            Weight::union_all(reachable_parts.iter())
        };
    }

    let start_state = dwa.start_state() as usize;
    if start_state >= n || needed[start_state].is_empty() {
        return Some(canonical_dead_dwa());
    }

    let mut heights = vec![0usize; n];
    for &source in topo.iter().rev() {
        heights[source] = productive_transitions[source]
            .iter()
            .map(|transition| heights[transition.target as usize] + 1)
            .max()
            .unwrap_or(0);
    }
    let max_height = heights.iter().copied().max().unwrap_or(0);

    let mut reachable_from_start = vec![false; n];
    let mut stack = vec![start_state];
    while let Some(source) = stack.pop() {
        if reachable_from_start[source] {
            continue;
        }
        reachable_from_start[source] = true;
        stack.extend(
            productive_transitions[source]
                .iter()
                .map(|transition| transition.target as usize),
        );
    }

    let mut states_by_height = vec![Vec::<usize>::new(); max_height + 1];
    for state in 0..n {
        if reachable_from_start[state] && !needed[state].is_empty() {
            states_by_height[heights[state]].push(state);
        }
    }
    if states_by_height
        .iter()
        .any(|bucket| bucket.len() > MAX_HEIGHT_BUCKET)
    {
        return None;
    }

    let mut old_to_new = vec![UNMAPPED; n];
    let mut new_states = Vec::<MergedStateBuilder>::new();
    for (height, candidates) in states_by_height.iter().enumerate() {
        if candidates.is_empty() {
            continue;
        }
        let coloring = if height == 0
            && candidates
                .iter()
                .all(|&state| productive_transitions[state].is_empty())
        {
            vec![0usize; candidates.len()]
        } else {
            build_and_color_pairwise_greedy(
                dwa,
                candidates,
                &needed,
                &old_to_new,
                &productive_transitions,
            )
        };
        let base_new_id = new_states.len() as u32;
        let num_colors = coloring.iter().copied().max().map_or(0, |color| color + 1);
        for (index, &color) in coloring.iter().enumerate() {
            old_to_new[candidates[index]] = base_new_id + color as u32;
        }
        new_states.extend((0..num_colors).map(|_| MergedStateBuilder::default()));
        let builders = &mut new_states[base_new_id as usize..];
        for (index, &color) in coloring.iter().enumerate() {
            merge_productive_state_into_builder(
                candidates[index],
                color,
                dwa,
                &productive_transitions,
                &old_to_new,
                builders,
            );
        }
    }

    let minimized = reconstruct_dwa(start_state, &old_to_new, new_states, false);
    if weighted_dwa_minimize_profile_enabled() {
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_small_direct] input_states={} input_transitions={} output_states={} output_transitions={} total_ms={:.3}",
            dwa.num_states(),
            dwa.num_transitions(),
            minimized.num_states(),
            minimized.num_transitions(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(minimized)
}

// Public API.

/// Minimize an acyclic DWA using weight pushing + graph-coloring.
///
/// Falls back to the caller's DWA unchanged if the input is cyclic.
pub fn minimize_acyclic(dwa: &DWA) -> DWA {
    minimize_acyclic_owned(dwa.clone())
}

pub fn minimize_acyclic_owned(pushed: DWA) -> DWA {
    minimize_acyclic_owned_with_pointwise_class_order(pushed, PointwiseClassOrder::Stable)
}

pub fn minimize_acyclic_owned_with_pointwise_class_order(
    pushed: DWA,
    pointwise_class_order: PointwiseClassOrder,
) -> DWA {
    minimize_acyclic_owned_with_optional_live_domains(pushed, pointwise_class_order, None)
}

/// Minimize an acyclic DWA using exact live domains supplied by the
/// determinizer.
///
/// `live_domains[u]` must be the intrinsic token domain from which state `u`
/// can reach a final state.  The minimizer derives the productive transition
/// relation `W(u,a,v) intersection live_domains[v]` directly and therefore
/// skips the ordinary backward push recurrence over the DWA.
pub fn minimize_acyclic_owned_with_live_domains(
    dwa: DWA,
    live_domains: Vec<Weight>,
) -> DWA {
    minimize_acyclic_owned_with_optional_live_domains(
        dwa,
        PointwiseClassOrder::Stable,
        Some((live_domains, false)),
    )
}

/// Variant for a DWA whose edge weights have already been restricted to the
/// supplied target live domains by determinization.
pub fn minimize_acyclic_owned_with_productive_live_domains(
    dwa: DWA,
    live_domains: Vec<Weight>,
) -> DWA {
    minimize_acyclic_owned_with_optional_live_domains(
        dwa,
        PointwiseClassOrder::Stable,
        Some((live_domains, true)),
    )
}

fn minimize_acyclic_owned_with_optional_live_domains(
    mut pushed: DWA,
    pointwise_class_order: PointwiseClassOrder,
    supplied_live_domains: Option<(Vec<Weight>, bool)>,
) -> DWA {
    if pushed.states().is_empty() {
        return pushed;
    }
    if supplied_live_domains.is_none()
        && std::env::var_os("GLRMASK_WEIGHTED_MINIMIZE_DISABLE_SMALL_DIRECT").is_none()
        && let Some(minimized) = try_minimize_small_pairwise_direct(&pushed)
    {
        return minimized;
    }

    let profile_enabled = weighted_dwa_minimize_profile_enabled();
    let total_started_at = Instant::now();
    let input_states = pushed.num_states();
    let input_transitions = pushed.num_transitions();

    let (topo, needed, push_ms, edges_already_productive) =
        if let Some((needed, edges_already_productive)) = supplied_live_domains {
        if needed.len() != pushed.states().len() {
            panic!(
                "supplied weighted-DWA live-domain count {} does not match state count {}",
                needed.len(),
                pushed.states().len(),
            );
        }
        let Some(topo) = compute_topo_order(&pushed) else {
            return pushed;
        };

        if std::env::var_os("GLRMASK_VALIDATE_WEIGHTED_MINIMIZE_LIVE_DOMAINS").is_some() {
            let mut reference = pushed.clone();
            let (_, reference_topo, reference_needed) = push_weights(&mut reference);
            assert!(reference_topo.is_some(), "validated DWA must remain acyclic");
            assert_eq!(
                needed.len(),
                reference_needed.len(),
                "live-domain validation state count differs",
            );
            for (state, (derived, expected)) in
                needed.iter().zip(&reference_needed).enumerate()
            {
                assert_eq!(
                    derived, expected,
                    "NWA-derived live domain differs from DWA recurrence at state {state}",
                );
            }
        }
        (topo, needed, 0.0, edges_already_productive)
    } else {
        // Legacy reference path: compute live domains and push transition
        // weights together over the determinized DWA.
        let push_started_at = Instant::now();
        let parallel_push = std::env::var("GLRMASK_WEIGHTED_MINIMIZE_PARALLEL_PUSH")
            .map(|value| {
                let value = value.trim();
                !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
            })
            .unwrap_or(false);
        let (_, topo_from_push, reachable_from_push) = if parallel_push {
            let reference = std::env::var_os("GLRMASK_VALIDATE_WEIGHTED_MINIMIZE_PARALLEL_PUSH")
                .is_some()
                .then(|| pushed.clone());
            let result = push_weights_parallel_by_height(&mut pushed);
            if let Some(mut reference) = reference {
                let reference_result = push_weights(&mut reference);
                assert_eq!(
                    result.1, reference_result.1,
                    "parallel push produced a different topological order",
                );
                assert_eq!(
                    result.2, reference_result.2,
                    "parallel push produced different state live domains",
                );
                assert_eq!(
                    pushed.states(),
                    reference.states(),
                    "parallel push produced different productive transitions",
                );
            }
            result
        } else {
            push_weights(&mut pushed)
        };
        let push_ms = push_started_at.elapsed().as_secs_f64() * 1000.0;
        let Some(topo) = topo_from_push else {
            return pushed;
        };
        (topo, reachable_from_push, push_ms, true)
    };

    let start_state = pushed.start_state() as usize;
    #[cfg(debug_assertions)]
    if push_ms != 0.0 {
        debug_assert_pushed_weights_within_needed(&pushed, &needed);
    }
    if needed[start_state].is_empty() {
        return canonical_dead_dwa();
    }
    let productive_transitions_started_at = Instant::now();
    let productive_transitions =
        compute_productive_transitions(&pushed, &needed, edges_already_productive);
    let productive_transitions_ms =
        productive_transitions_started_at.elapsed().as_secs_f64() * 1000.0;
    let heights_started_at = Instant::now();
    let heights = compute_heights(&pushed, &topo);
    let heights_ms = heights_started_at.elapsed().as_secs_f64() * 1000.0;
    let max_height = heights.iter().max().copied().unwrap_or(0);

    let n = pushed.states().len();

    let reachable_from_start_started_at = Instant::now();
    let reachable_from_start = compute_reachable_from_start(&pushed, start_state);
    let reachable_from_start_ms =
        reachable_from_start_started_at.elapsed().as_secs_f64() * 1000.0;

    // Group states by height (only reachable states with non-empty needed sets)
    let group_by_height_started_at = Instant::now();
    let mut states_by_height: Vec<Vec<usize>> = vec![vec![]; max_height + 1];
    for (id, &h) in heights.iter().enumerate() {
        if !reachable_from_start[id] {
            continue;
        }
        if needed[id].is_empty() {
            continue;
        }
        states_by_height[h].push(id);
    }
    let group_by_height_ms = group_by_height_started_at.elapsed().as_secs_f64() * 1000.0;
    let active_states = states_by_height.iter().map(Vec::len).sum::<usize>();
    let max_bucket_size = states_by_height.iter().map(Vec::len).max().unwrap_or(0);

    // Bottom-up: color and merge
    let mut old_to_new = vec![UNMAPPED; n];
    let mut new_states: Vec<MergedStateBuilder> = Vec::new();
    let mut color_ms_total = 0.0;
    let mut merge_ms_total = 0.0;

    for h in 0..=max_height {
        let candidates = &states_by_height[h];
        if candidates.is_empty() {
            continue;
        }
        let height_started_at = Instant::now();
        let mut height_color_ms = 0.0;
        let mut fast_height0 = false;

        if h == 0 {
            let color_started_at = Instant::now();
            let all_compatible = try_all_compatible_height_0_coloring(candidates, &pushed, &needed).is_some();
            height_color_ms = color_started_at.elapsed().as_secs_f64() * 1000.0;
            if all_compatible {
                fast_height0 = true;
                let merge_started_at = Instant::now();
                let base_new_id = new_states.len() as u32;
                let num_colors = 1usize;

                for &candidate in candidates {
                    old_to_new[candidate] = base_new_id;
                }

                new_states.extend((0..num_colors).map(|_| MergedStateBuilder::default()));

                let builders = &mut new_states[base_new_id as usize..];
                for &candidate in candidates {
                    merge_productive_state_into_builder(
                        candidate,
                        0,
                        &pushed,
                        &productive_transitions,
                        &old_to_new,
                        builders,
                    );
                }
                let height_merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;
                color_ms_total += height_color_ms;
                merge_ms_total += height_merge_ms;
                if profile_enabled {
                    eprintln!(
                        "[glrmask/profile][weighted_dwa_minimize_height] height={} candidates={} colors={} fast_height0={} color_ms={:.3} merge_ms={:.3} total_ms={:.3}",
                        h,
                        candidates.len(),
                        num_colors,
                        fast_height0,
                        height_color_ms,
                        height_merge_ms,
                        height_started_at.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                continue;
            }
        }

        let color_started_at = Instant::now();
        let HybridColoring {
            colors: coloring,
            direct_builders,
        } = build_and_color_hybrid(
            &pushed,
            candidates,
            &needed,
            &old_to_new,
            &productive_transitions,
            pointwise_class_order,
        );
        height_color_ms += color_started_at.elapsed().as_secs_f64() * 1000.0;

        let merge_started_at = Instant::now();
        let base_new_id = new_states.len() as u32;
        let num_colors = coloring.iter().max().map(|&c| c + 1).unwrap_or(0);

        // Map old states to new merged states
        for (idx, &color) in coloring.iter().enumerate() {
            old_to_new[candidates[idx]] = base_new_id + color as u32;
        }

        if let Some(mut direct_builders) = direct_builders {
            debug_assert_eq!(direct_builders.len(), num_colors);
            if std::env::var_os("GLRMASK_VALIDATE_WEIGHTED_MINIMIZE_DIRECT_QUOTIENT").is_some() {
                validate_direct_builders(
                    candidates,
                    &coloring,
                    &pushed,
                    &productive_transitions,
                    &old_to_new,
                    &direct_builders,
                );
            }
            new_states.append(&mut direct_builders);
        } else {
            // Extend builders
            new_states.extend((0..num_colors).map(|_| MergedStateBuilder::default()));

            // Merge states into builders
            let builders = &mut new_states[base_new_id as usize..];
            for (idx, &color) in coloring.iter().enumerate() {
                merge_productive_state_into_builder(
                    candidates[idx],
                    color,
                    &pushed,
                    &productive_transitions,
                    &old_to_new,
                    builders,
                );
            }
        }
        let height_merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;
        color_ms_total += height_color_ms;
        merge_ms_total += height_merge_ms;

        if profile_enabled {
            eprintln!(
                "[glrmask/profile][weighted_dwa_minimize_height] height={} candidates={} colors={} fast_height0={} color_ms={:.3} merge_ms={:.3} total_ms={:.3}",
                h,
                candidates.len(),
                num_colors,
                fast_height0,
                height_color_ms,
                height_merge_ms,
                height_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    let reconstruct_started_at = Instant::now();
    let minimized = reconstruct_dwa(
        start_state,
        &old_to_new,
        new_states,
        input_states >= 1_024 && input_transitions >= 4_096,
    );
    let reconstruct_ms = reconstruct_started_at.elapsed().as_secs_f64() * 1000.0;

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize] input_states={} input_transitions={} push_ms={:.3} productive_transitions_ms={:.3} heights_ms={:.3} reachable_from_start_ms={:.3} group_by_height_ms={:.3} color_ms_total={:.3} merge_ms_total={:.3} reconstruct_ms={:.3} max_height={} active_states={} max_bucket_size={} output_states={} output_transitions={} total_ms={:.3}",
            input_states,
            input_transitions,
            push_ms,
            productive_transitions_ms,
            heights_ms,
            reachable_from_start_ms,
            group_by_height_ms,
            color_ms_total,
            merge_ms_total,
            reconstruct_ms,
            max_height,
            active_states,
            max_bucket_size,
            minimized.num_states(),
            minimized.num_transitions(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    minimized
}

#[cfg(test)]
mod tests {
    use super::{
        batch_build_weight, build_exact_group_summary, final_weights_compatible_on_domain,
        memberwise_group_compatible, minimize_acyclic, push_weights,
        overlay_compatible_token_behavior_ranges,
        sorted_weights_compatible_on_domain,
        sorted_weights_compatible_on_domain_intersection,
        weight_is_disjoint_from_domain_intersection, weights_equal_on_domain,
        weights_equal_on_domain_intersection, ClassProfile, PointwiseBehaviorInterner,
        PointwiseBehaviorMap, PointwiseBehaviorMapLayout, PointwiseProfile,
        PointwiseRegionBuildCache, PointwiseRegionInterner, TokenBehaviorRange,
        build_token_behavior_region,
    };
    use crate::weighted_u32::dwa::{DWA, DWAState};
    use crate::ds::weight::Weight;
    use range_set_blaze::RangeSetBlaze;
    use std::sync::Arc;

    fn token_set(ranges: &[(u32, u32)]) -> RangeSetBlaze<u32> {
        ranges.iter().copied().map(|(start, end)| start..=end).collect()
    }

    fn weight(entries: &[(u32, &[(u32, u32)])]) -> Weight {
        Weight::from_per_tsid_token_sets(
            entries
                .iter()
                .copied()
                .map(|(tsid, ranges)| (tsid, token_set(ranges))),
        )
    }

    #[test]
    fn reconstruction_one_sweep_matches_reduction_tree() {
        let pending: Vec<Weight> = (0..257u32)
            .map(|index| {
                Weight::from_per_tsid_token_sets(std::iter::once((
                    index % 17,
                    RangeSetBlaze::from_iter(std::iter::once((index % 31)..=(index % 31 + 2))),
                )))
            })
            .collect();

        let direct = batch_build_weight(pending.clone(), false);
        let mut tree = pending;
        while tree.len() > 64 {
            tree = tree
                .chunks(64)
                .map(|chunk| Weight::union_all(chunk.iter()))
                .collect();
        }
        let reduction_tree = Weight::union_all(tree.iter());
        assert_eq!(direct, reduction_tree);
    }

    fn assert_disjoint_matches_overlap(weight: &Weight, left: &Weight, right: &Weight) {
        let overlap = left.intersection(right);
        assert_eq!(
            weight_is_disjoint_from_domain_intersection(weight, left, right),
            weight.is_disjoint(&overlap),
        );
    }

    fn assert_equal_matches_overlap(a: &Weight, b: &Weight, left: &Weight, right: &Weight) {
        let overlap = left.intersection(right);
        assert_eq!(
            weights_equal_on_domain_intersection(a, b, left, right),
            weights_equal_on_domain(a, b, &overlap),
        );
    }

    #[test]
    fn pointwise_region_build_cache_includes_transition_target() {
        let domain_tokens = token_set(&[(0, 3)]);
        let transition_tokens = token_set(&[(0, 3)]);
        let mut behaviors = PointwiseBehaviorInterner::default();
        let mut regions = PointwiseRegionInterner::default();
        let mut cache = PointwiseRegionBuildCache::default();
        let first_transitions = [(7, 1, &transition_tokens)];
        let second_transitions = [(7, 2, &transition_tokens)];

        let first = build_token_behavior_region(
            &domain_tokens,
            None,
            &first_transitions,
            &mut behaviors,
            &mut regions,
            &mut cache,
        )
        .unwrap();
        let first_again = build_token_behavior_region(
            &domain_tokens,
            None,
            &first_transitions,
            &mut behaviors,
            &mut regions,
            &mut cache,
        )
        .unwrap();
        let second = build_token_behavior_region(
            &domain_tokens,
            None,
            &second_transitions,
            &mut behaviors,
            &mut regions,
            &mut cache,
        )
        .unwrap();

        assert!(Arc::ptr_eq(&first, &first_again));
        assert_ne!(first.as_ref(), second.as_ref());
        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cache.hits, 1);
        assert_eq!(cache.misses, 2);
    }

    #[test]
    fn direct_mapped_region_overlay_cache_matches_exact_overlay() {
        let mut regions = PointwiseRegionInterner::with_direct_overlay_slots(8);
        let left = regions.intern(vec![
            TokenBehaviorRange {
                start: 0,
                end: 3,
                behavior: 1,
            },
            TokenBehaviorRange {
                start: 8,
                end: 9,
                behavior: 2,
            },
        ]);
        let right = regions.intern(vec![
            TokenBehaviorRange {
                start: 2,
                end: 5,
                behavior: 1,
            },
            TokenBehaviorRange {
                start: 10,
                end: 12,
                behavior: 3,
            },
        ]);
        let expected = overlay_compatible_token_behavior_ranges(left.as_ref(), right.as_ref());

        let first = regions.overlay_compatible(&left, &right);
        let second = regions.overlay_compatible(&right, &left);

        assert_eq!(first.as_ref(), &expected);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(regions.direct_overlay_stats(), (8, 1, 1, 0));
    }

    #[test]
    fn dense_pointwise_behavior_map_matches_sparse() {
        let left = Arc::new(vec![TokenBehaviorRange {
            start: 0,
            end: 3,
            behavior: 1,
        }]);
        let right = Arc::new(vec![TokenBehaviorRange {
            start: 4,
            end: 7,
            behavior: 2,
        }]);
        let first = PointwiseProfile {
            by_tsid: vec![(1, Arc::clone(&left)), (3, Arc::clone(&right))],
        };
        let second = PointwiseProfile {
            by_tsid: vec![(1, Arc::clone(&right)), (2, Arc::clone(&left))],
        };

        let mut sparse = PointwiseBehaviorMap::new(PointwiseBehaviorMapLayout::Sparse);
        let mut dense = PointwiseBehaviorMap::new(PointwiseBehaviorMapLayout::Dense { slots: 4 });
        let mut sparse_regions = PointwiseRegionInterner::default();
        let mut dense_regions = PointwiseRegionInterner::default();
        for profile in [&first, &second] {
            sparse.merge_profile(profile, &mut sparse_regions);
            dense.merge_profile(profile, &mut dense_regions);
        }

        for tsid in 0..4 {
            assert_eq!(
                sparse.get(tsid).map(AsRef::as_ref),
                dense.get(tsid).map(AsRef::as_ref),
                "tsid={tsid}",
            );
        }
        assert_eq!(sparse.region_entry_count(), dense.region_entry_count());
    }

    #[test]
    fn push_weights_maintains_needed_containment_invariant() {
        let mut source = DWAState::default();
        source.transitions.insert(
            42,
            (1, Weight::from_uniform(1..=1, token_set(&[(10, 20)]))),
        );
        let mut target = DWAState::default();
        target.final_weight = Some(Weight::from_uniform(1..=1, token_set(&[(12, 14)])));

        let mut dwa = DWA::from_parts(vec![source, target], 0);
        let (_, topo, needed) = push_weights(&mut dwa);
        assert!(topo.is_some());

        let transition_weight = &dwa.states()[0].transitions.get(&42).unwrap().1;
        let target_final_weight = dwa.states()[1].final_weight.as_ref().unwrap();

        assert!(target_final_weight.is_subset(&needed[1]));
        assert!(transition_weight.is_subset(&needed[0]));
        assert!(transition_weight.is_subset(&needed[1]));
        assert_eq!(
            transition_weight,
            &Weight::from_uniform(1..=1, token_set(&[(12, 14)]))
        );
    }

    #[test]
    fn transition_compat_accepts_matching_label_equal_on_overlap_but_different_outside() {
        let overlap = weight(&[(1, &[(10, 20)])]);
        let class_weights = vec![(7, weight(&[(1, &[(10, 20)]), (2, &[(30, 30)])]))];
        let group_weights = vec![(7, weight(&[(1, &[(10, 20)]), (3, &[(40, 40)])]))];

        assert!(sorted_weights_compatible_on_domain_intersection(
            &class_weights,
            &group_weights,
            &overlap,
            &overlap,
        ));
    }

    #[test]
    fn transition_compat_rejects_class_only_label_active_on_overlap() {
        let overlap = weight(&[(1, &[(10, 20)])]);
        let class_weights = vec![(7, weight(&[(1, &[(10, 20)])]))];
        let group_weights = Vec::new();

        assert!(!sorted_weights_compatible_on_domain_intersection(
            &class_weights,
            &group_weights,
            &overlap,
            &overlap,
        ));
    }

    #[test]
    fn transition_compat_rejects_group_only_label_active_on_overlap() {
        let overlap = weight(&[(1, &[(10, 20)])]);
        let class_weights = Vec::new();
        let group_weights = vec![(7, weight(&[(1, &[(10, 20)])]))];

        assert!(!sorted_weights_compatible_on_domain_intersection(
            &class_weights,
            &group_weights,
            &overlap,
            &overlap,
        ));
    }

    #[test]
    fn transition_compat_rejects_same_target_shape_with_extra_active_label() {
        let overlap = weight(&[(1, &[(10, 20)])]);
        let class_weights = vec![(7, weight(&[(1, &[(10, 20)])]))];
        let group_weights = vec![
            (7, weight(&[(1, &[(10, 20)])])),
            (9, weight(&[(1, &[(10, 20)])])),
        ];

        assert!(!sorted_weights_compatible_on_domain_intersection(
            &class_weights,
            &group_weights,
            &overlap,
            &overlap,
        ));
    }

    #[test]
    fn transition_compat_accepts_class_and_group_weights_disjoint_from_overlap() {
        let overlap = weight(&[(1, &[(10, 20)])]);
        let class_weights = vec![(7, weight(&[(2, &[(10, 20)])]))];
        let group_weights = vec![(9, weight(&[(3, &[(10, 20)])]))];

        assert!(sorted_weights_compatible_on_domain_intersection(
            &class_weights,
            &group_weights,
            &overlap,
            &overlap,
        ));
    }

    #[test]
    fn minimize_acyclic_merges_overlapping_partial_transition_states_exactly() {
        let branch_weights = [
            Weight::from_uniform(0..=0, token_set(&[(1, 2)])),
            Weight::from_uniform(0..=0, token_set(&[(2, 3)])),
            Weight::from_uniform(0..=0, token_set(&[(3, 4)])),
        ];
        let mut start = DWAState::default();
        for (idx, label) in [10, 11, 12].into_iter().enumerate() {
            start.transitions.insert(label, ((idx + 1) as u32, Weight::all()));
        }

        let mut states = vec![start];
        for weight in &branch_weights {
            let mut branch = DWAState::default();
            branch.transitions.insert(20, (4, weight.clone()));
            states.push(branch);
        }
        let mut leaf = DWAState::default();
        leaf.final_weight = Some(Weight::all());
        states.push(leaf);
        let dwa = DWA::from_parts(states, 0);

        let words = [[10, 20], [11, 20], [12, 20], [10, 21], [11, 21], [12, 21]];
        let expected = words.map(|word| dwa.eval_word(&word));
        let minimized = minimize_acyclic(&dwa);

        assert_eq!(minimized.num_states(), 3);
        for (word, expected) in words.into_iter().zip(expected) {
            assert_eq!(minimized.eval_word(&word), expected, "word={word:?}");
        }
    }

    #[test]
    fn materialized_profile_comparison_matches_intersection_profile_comparison() {
        let class_weights = vec![
            (7, weight(&[(0, &[(1, 3)]), (1, &[(4, 5)])])),
            (9, weight(&[(0, &[(8, 9)])])),
        ];
        let group_weights = vec![
            (7, weight(&[(0, &[(2, 4)]), (1, &[(4, 5)])])),
            (8, weight(&[(0, &[(10, 11)])])),
        ];
        let class_domain = weight(&[(0, &[(1, 3)]), (1, &[(4, 5)])]);
        let group_domain = weight(&[(0, &[(2, 4)]), (1, &[(4, 6)])]);
        let overlap = class_domain.intersection(&group_domain);

        assert_eq!(
            sorted_weights_compatible_on_domain_intersection(
                &class_weights,
                &group_weights,
                &class_domain,
                &group_domain,
            ),
            sorted_weights_compatible_on_domain(&class_weights, &group_weights, &overlap),
        );
    }

    #[test]
    fn exact_group_summary_matches_memberwise_compatibility() {
        let needed = vec![
            weight(&[(0, &[(0, 9)])]),
            weight(&[(1, &[(0, 9)])]),
            weight(&[(1, &[(0, 9)])]),
        ];
        let members = vec![
            ClassProfile {
                targets: Vec::new(),
                weights: vec![(7, weight(&[(0, &[(2, 4)])]))],
                final_weight: None,
            },
            ClassProfile {
                targets: Vec::new(),
                weights: vec![(7, weight(&[(1, &[(3, 5)])]))],
                final_weight: None,
            },
        ];
        let summary = build_exact_group_summary(&[0, 1], &needed, &members);

        let compatible = ClassProfile {
            targets: Vec::new(),
            weights: vec![(7, weight(&[(1, &[(3, 5)])]))],
            final_weight: None,
        };
        let incompatible = ClassProfile {
            targets: Vec::new(),
            weights: vec![(7, weight(&[(1, &[(6, 8)])]))],
            final_weight: None,
        };

        for candidate in [&compatible, &incompatible] {
            let overlap = needed[2].intersection(&summary.needed_union);
            let via_summary = final_weights_compatible_on_domain(
                candidate.final_weight.as_ref(),
                summary.merged_final_weight.as_ref(),
                &overlap,
            ) && sorted_weights_compatible_on_domain(
                &candidate.weights,
                &summary.transition_weights,
                &overlap,
            );
            let via_members = memberwise_group_compatible(
                &needed[2],
                candidate,
                &[0, 1],
                &needed,
                &members,
            );
            assert_eq!(via_summary, via_members);
        }
        assert!(memberwise_group_compatible(
            &needed[2],
            &compatible,
            &[0, 1],
            &needed,
            &members,
        ));
        assert!(!memberwise_group_compatible(
            &needed[2],
            &incompatible,
            &[0, 1],
            &needed,
            &members,
        ));
    }

    #[test]
    fn minimize_acyclic_helpers_match_materialized_overlap_for_empty_tsid_intersection() {
        let left = Weight::from_uniform(0..=1, token_set(&[(1, 3)]));
        let right = Weight::from_uniform(4..=5, token_set(&[(1, 3)]));
        let weight_a = weight(&[(0, &[(1, 2)]), (4, &[(2, 4)])]);
        let weight_b = weight(&[(1, &[(1, 1)]), (5, &[(3, 5)])]);

        assert_disjoint_matches_overlap(&weight_a, &left, &right);
        assert_disjoint_matches_overlap(&weight_b, &left, &right);
        assert_equal_matches_overlap(&weight_a, &weight_b, &left, &right);
    }

    #[test]
    fn minimize_acyclic_helpers_match_materialized_overlap_for_disjoint_token_domains() {
        let left = Weight::from_uniform(0..=2, token_set(&[(1, 2)]));
        let right = Weight::from_uniform(0..=2, token_set(&[(5, 6)]));
        let weight_a = weight(&[(0, &[(1, 2)]), (1, &[(5, 6)])]);
        let weight_b = weight(&[(0, &[(2, 3)]), (2, &[(6, 7)])]);

        assert_disjoint_matches_overlap(&weight_a, &left, &right);
        assert_disjoint_matches_overlap(&weight_b, &left, &right);
        assert_equal_matches_overlap(&weight_a, &weight_b, &left, &right);
    }

    #[test]
    fn minimize_acyclic_helpers_match_materialized_overlap_when_weight_range_is_missing() {
        let left = Weight::from_uniform(1..=3, token_set(&[(1, 4)]));
        let right = Weight::from_uniform(2..=4, token_set(&[(2, 5)]));
        let weight_a = weight(&[(2, &[(2, 4)])]);
        let weight_b = weight(&[(2, &[(2, 4)]), (3, &[(2, 4)])]);

        assert_disjoint_matches_overlap(&weight_a, &left, &right);
        assert_disjoint_matches_overlap(&weight_b, &left, &right);
        assert_equal_matches_overlap(&weight_a, &weight_b, &left, &right);
    }

    #[test]
    fn minimize_acyclic_helpers_match_materialized_overlap_for_equal_weights_by_value() {
        let left = weight(&[(0, &[(1, 3)]), (1, &[(2, 4)])]);
        let right = weight(&[(0, &[(2, 5)]), (1, &[(1, 4)])]);
        let weight_a = weight(&[(0, &[(2, 3)]), (1, &[(2, 4)])]);
        let weight_b = weight(&[(0, &[(2, 3)]), (1, &[(2, 4)])]);

        assert_disjoint_matches_overlap(&weight_a, &left, &right);
        assert_disjoint_matches_overlap(&weight_b, &left, &right);
        assert_equal_matches_overlap(&weight_a, &weight_b, &left, &right);
    }

    #[test]
    fn minimize_acyclic_helpers_match_materialized_overlap_when_difference_is_outside_overlap() {
        let left = Weight::from_uniform(0..=1, token_set(&[(1, 2)]));
        let right = Weight::from_uniform(0..=1, token_set(&[(1, 2)]));
        let weight_a = weight(&[(0, &[(1, 2), (5, 5)]), (1, &[(1, 2)])]);
        let weight_b = weight(&[(0, &[(1, 2)]), (1, &[(1, 2)])]);

        assert_disjoint_matches_overlap(&weight_a, &left, &right);
        assert_disjoint_matches_overlap(&weight_b, &left, &right);
        assert_equal_matches_overlap(&weight_a, &weight_b, &left, &right);
    }

    #[test]
    fn minimize_acyclic_helpers_match_materialized_overlap_when_difference_is_inside_overlap() {
        let left = Weight::from_uniform(0..=1, token_set(&[(1, 3)]));
        let right = Weight::from_uniform(0..=1, token_set(&[(2, 4)]));
        let weight_a = weight(&[(0, &[(1, 2)]), (1, &[(2, 3)])]);
        let weight_b = weight(&[(0, &[(2, 3)]), (1, &[(2, 4)])]);

        assert_disjoint_matches_overlap(&weight_a, &left, &right);
        assert_disjoint_matches_overlap(&weight_b, &left, &right);
        assert_equal_matches_overlap(&weight_a, &weight_b, &left, &right);
    }
}
