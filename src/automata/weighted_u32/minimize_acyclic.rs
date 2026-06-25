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
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::dwa::{DWA, DWAState};
use crate::ds::weight::Weight;

type Label = i32;

const UNMAPPED: u32 = u32::MAX;
// Reconstruction reduces large exact weight unions in a bounded tree. 64 is
// the best measured fan-in for the global terminal-DWA merge: it avoids the
// additional rounds of 16 while keeping intermediate event sweeps bounded.
const RECONSTRUCTION_UNION_BATCH_SIZE: usize = 64;

fn weighted_dwa_minimize_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
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
    Arc::as_ptr(&weight.0) as usize
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

    let value = left.intersection(right);
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
    let mut changed = false;
    let mut target_full = 0usize;
    let mut target_empty = 0usize;
    let mut target_partial = 0usize;
    let mut reachable_parts = 0usize;
    let mut pushed_transitions = 0usize;
    let mut intersection_ms = 0.0;
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
                let new_w = memoized_intersection(&mut intersection_cache, w, &reachable[t]);
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
            "[glrmask/profile][weighted_dwa_minimize_push] states={} topo_ms={:.3} target_full={} target_empty={} target_partial={} intersection_ms={:.3} intersection_cache_entries={} reachable_parts={} union_ms={:.3} union_sizes=[{},{},{},{},{},{},{}] max_union_size={} union_key_occurrences={} union_unique_keys={} union_key_repeats={} pushed_transitions={} apply_ms={:.3}",
            n,
            topo_ms,
            target_full,
            target_empty,
            target_partial,
            intersection_ms,
            intersection_cache.len(),
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

fn compute_productive_transitions(dwa: &DWA, needed: &[Weight]) -> Vec<Vec<ProductiveTransition>> {
    let n = dwa.states().len();
    let mut result = Vec::with_capacity(n);

    for state in dwa.states() {
        let mut transitions = Vec::with_capacity(state.transitions.len());
        for (&label, (target, weight)) in &state.transitions {
            let t = *target as usize;
            if t >= n {
                continue;
            }
            // After push_weights, all remaining transitions are already productive
            // (w_pushed = w_orig ∩ reachable[t], and needed == reachable from push).
            // The intersection w_pushed ∩ needed[t] = w_pushed, so just clone.
            if needed[t].is_empty() {
                continue;
            }
            transitions.push(ProductiveTransition {
                label,
                target: *target,
                weight: weight.clone(),
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
    if Arc::ptr_eq(&w_a.0, &w_b.0) {
        return true;
    }
    if domain.is_empty() {
        return true;
    }

    let mut a_iter = w_a.0.range_values();
    let mut b_iter = w_b.0.range_values();
    let mut a_entry = a_iter.next();
    let mut b_entry = b_iter.next();

    for (d_range, d_tokens) in domain.0.range_values() {
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
    let mut weight_iter = weight.0.range_values();
    let mut weight_entry = weight_iter.next();
    let mut left_iter = left_domain.0.range_values();
    let mut right_iter = right_domain.0.range_values();
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
    if Arc::ptr_eq(&w_a.0, &w_b.0) {
        return true;
    }

    let mut a_iter = w_a.0.range_values();
    let mut b_iter = w_b.0.range_values();
    let mut left_iter = left_domain.0.range_values();
    let mut right_iter = right_domain.0.range_values();
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
}

impl PointwiseBehaviorInterner {
    fn intern(&mut self, final_active: bool, transitions: Vec<(Label, u32)>) -> u32 {
        let behavior = PointwiseBehavior { final_active, transitions };
        if let Some(&id) = self.ids.get(&behavior) {
            return id;
        }
        let id = self.ids.len() as u32;
        self.ids.insert(behavior, id);
        id
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

#[derive(Default)]
struct PointwiseRegionInterner {
    // Keep the immutable region itself as the hash key. `Arc<Vec<_>>` hashes
    // and compares by its contents, so lookup can still borrow a fresh Vec,
    // but newly interned regions no longer need a second full Vec clone solely
    // for map ownership.
    regions: FxHashMap<Arc<Vec<TokenBehaviorRange>>, ()>,
}

impl PointwiseRegionInterner {
    fn intern(&mut self, ranges: Vec<TokenBehaviorRange>) -> Arc<Vec<TokenBehaviorRange>> {
        if let Some((existing, _)) = self.regions.get_key_value(&ranges) {
            return Arc::clone(existing);
        }
        let ranges = Arc::new(ranges);
        self.regions.insert(Arc::clone(&ranges), ());
        ranges
    }
}

#[derive(Default)]
struct PointwiseProfile {
    /// Sorted by TSID. Each behavior region is immutable and shared by every
    /// TSID whose weight behavior is identical.
    by_tsid: Vec<(u32, Arc<Vec<TokenBehaviorRange>>)>,
}

#[derive(Default)]
struct PointwiseMergeGroup {
    targets_by_label: FxHashMap<Label, u32>,
    /// Exact partial behavior function of all members already in this group.
    behavior_by_tsid: FxHashMap<u32, Arc<Vec<TokenBehaviorRange>>>,
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
    let mut by_tsid = Vec::new();
    for (domain_start, domain_end, domain_tokens) in domain.compact_entries()? {
        let mut boundaries = vec![u64::from(domain_start), u64::from(domain_end) + 1];
        if let Some(final_weight) = &profile.final_weight {
            for (tsid_range, _) in final_weight.0.range_values() {
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
            for (tsid_range, _) in weight.0.range_values() {
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
                .and_then(|weight| weight.0.get(tsid_start))
                .map(|tokens| tokens.as_ref());
            let mut active_transitions = Vec::new();
            for (label, target, weight) in &transitions {
                if let Some(tokens) = weight.0.get(tsid_start) {
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
            for tsid in tsid_start..=tsid_end {
                by_tsid.push((tsid, Arc::clone(&region)));
            }
        }
    }
    Some(PointwiseProfile { by_tsid })
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
        group.behavior_by_tsid.get(tsid).is_none_or(|existing| {
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
    for (tsid, ranges) in &profile.by_tsid {
        match group.behavior_by_tsid.get_mut(tsid) {
            Some(existing) if Arc::ptr_eq(existing, ranges) => {}
            Some(existing) => {
                *existing = regions.intern(overlay_compatible_token_behavior_ranges(
                    existing.as_ref(),
                    ranges.as_ref(),
                ));
            }
            None => {
                group.behavior_by_tsid.insert(*tsid, Arc::clone(ranges));
            }
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
    profile_enabled: bool,
) -> Option<Vec<usize>> {
    let started_at = Instant::now();
    let mut interner = PointwiseBehaviorInterner::default();
    let mut regions = PointwiseRegionInterner::default();
    let mut region_build_cache = PointwiseRegionBuildCache::default();
    let profile_started_at = Instant::now();
    let mut pointwise_profiles = Vec::with_capacity(class_profiles.len());
    for (domain, profile) in class_needed_union.iter().zip(class_profiles) {
        pointwise_profiles.push(build_pointwise_profile(
            domain,
            profile,
            &mut interner,
            &mut regions,
            &mut region_build_cache,
        )?);
    }
    let profile_build_ms = profile_started_at.elapsed().as_secs_f64() * 1000.0;

    let merge_started_at = Instant::now();
    let mut groups = Vec::<PointwiseMergeGroup>::new();
    let mut group_attempts = 0usize;
    let mut target_rejects = 0usize;
    let mut behavior_rejects = 0usize;
    for class in 0..class_profiles.len() {
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
                behavior_by_tsid: FxHashMap::default(),
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
        let region_entries = groups
            .iter()
            .map(|group| {
                group
                    .behavior_by_tsid
                    .values()
                    .map(|ranges| ranges.len())
                    .sum::<usize>()
            })
            .sum::<usize>();
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_pointwise] candidates={} classes={} groups={} behaviors={} interned_regions={} regions={} region_build_cache_entries={} region_build_cache_hits={} region_build_cache_misses={} profile_build_ms={:.3} merge_ms={:.3} total_ms={:.3} group_attempts={} target_rejects={} behavior_rejects={}",
            candidates.len(),
            class_profiles.len(),
            groups.len(),
            interner.ids.len(),
            regions.regions.len(),
            region_entries,
            region_build_cache.entries.len(),
            region_build_cache.hits,
            region_build_cache.misses,
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

#[cfg(debug_assertions)]
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
fn build_and_color_hybrid(
    dwa: &DWA,
    candidates: &[usize],
    needed: &[Weight],
    old_to_new: &[u32],
    productive_transitions: &[Vec<ProductiveTransition>],
) -> Vec<usize> {
    let profile_enabled = weighted_dwa_minimize_profile_enabled();
    let total_started_at = Instant::now();

    // Step 1: Partition refinement to get fine-grained classes.
    let partition_refine_started_at = Instant::now();
    let class_coloring = partition_refine_coloring_raw(candidates, dwa, old_to_new);
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
        return class_coloring;
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

    // Each class is a partial behavior function over its pushed-needed
    // TSID/token domain. When that representation is finite, compare against
    // the exact union function of each group instead of revisiting all members.
    // A full sentinel cannot be enumerated, so it retains the generic path.
    if let Some(coloring) = try_build_and_color_pointwise(
        candidates,
        &class_coloring,
        &class_needed_union,
        &class_profiles,
        profile_enabled,
    ) {
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

    coloring
}

fn partition_refine_coloring_raw(
    candidates: &[usize],
    dwa: &DWA,
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
        // Hash raw transitions (BTreeMap iterates in label order)
        for (&label, (target, weight)) in &dwa.states()[c].transitions {
            let Some(mapped) = mapped_target(old_to_new, *target) else {
                continue;
            };
            label.hash(&mut hasher);
            mapped.hash(&mut hasher);
            weight.hash(&mut hasher);
        }
        dwa.states()[c].transitions.len().hash(&mut hasher);

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
                if states_raw_equal(c, rep, dwa, old_to_new) {
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

/// Check if two states have identical raw signatures (no needed restriction).
fn states_raw_equal(
    u: usize,
    v: usize,
    dwa: &DWA,
    old_to_new: &[u32],
) -> bool {
    if dwa.states()[u].final_weight != dwa.states()[v].final_weight {
        return false;
    }
    let su = &dwa.states()[u];
    let sv = &dwa.states()[v];
    if su.transitions.len() != sv.transitions.len() {
        return false;
    }
    // BTreeMap iterates in key order, so we can zip
    for ((&lu, (tu, wu)), (&lv, (tv, wv))) in su.transitions.iter().zip(sv.transitions.iter()) {
        if lu != lv { return false; }
        if mapped_target(old_to_new, *tu) != mapped_target(old_to_new, *tv) { return false; }
        if wu != wv { return false; }
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

/// Batch-build a Weight from a Vec of pending weights using a hybrid strategy.
fn batch_build_weight(pending: Vec<Weight>) -> Weight {
    match pending.len() {
        0 => Weight::empty(),
        1 => pending.into_iter().next().unwrap(),
        n if n <= RECONSTRUCTION_UNION_BATCH_SIZE => Weight::union_all(pending.iter()),
        _ => {
            let mut current = pending;
            while current.len() > RECONSTRUCTION_UNION_BATCH_SIZE {
                current = current
                    .chunks(RECONSTRUCTION_UNION_BATCH_SIZE)
                    .map(|chunk| Weight::union_all(chunk.iter()))
                    .collect();
            }
            Weight::union_all(current.iter())
        }
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

fn reconstruct_dwa(start_old: usize, old_to_new: &[u32], builders: Vec<MergedStateBuilder>) -> DWA {
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
            let final_weight = batch_build_weight(b.final_weights_pending);
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
                let weight = batch_build_weight(pending_weights);
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

// Public API.

/// Minimize an acyclic DWA using weight pushing + graph-coloring.
///
/// Falls back to the caller's DWA unchanged if the input is cyclic.
pub fn minimize_acyclic(dwa: &DWA) -> DWA {
    minimize_acyclic_owned(dwa.clone())
}

pub fn minimize_acyclic_owned(mut pushed: DWA) -> DWA {
    if pushed.states().is_empty() {
        return pushed;
    }

    let profile_enabled = weighted_dwa_minimize_profile_enabled();
    let total_started_at = Instant::now();
    let input_states = pushed.num_states();
    let input_transitions = pushed.num_transitions();

    // Push weights in-place on the owned determinized DWA.
    let push_started_at = Instant::now();
    let (_, topo_from_push, reachable_from_push) = push_weights(&mut pushed);
    let push_ms = push_started_at.elapsed().as_secs_f64() * 1000.0;

    // Reuse topo order from push_weights (graph structure unchanged by push).
    let topo = match topo_from_push {
        Some(t) => t,
        None => return pushed, // cyclic — fall back
    };

    // Reuse backward-reachable token sets from push_weights as needed sets.
    // Proof: push_weights computes reachable[u] = final(u) ∪ union(w(u,t) ∩ reachable[t]).
    // a fresh needed-set pass on the pushed DWA uses the same recurrence (since
    // w_pushed = w_orig ∩ reachable[t], and A ∩ A = A in the needed recurrence).
    // Both produce identical results, so we skip the redundant recomputation.
    let start_state = pushed.start_state() as usize;
    let needed = reachable_from_push;
    #[cfg(debug_assertions)]
    debug_assert_pushed_weights_within_needed(&pushed, &needed);
    if needed[start_state].is_empty() {
        return canonical_dead_dwa();
    }
    let productive_transitions_started_at = Instant::now();
    let productive_transitions = compute_productive_transitions(&pushed, &needed);
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
                    merge_state_into_builder(
                        candidate,
                        0,
                        &pushed,
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
        let coloring = build_and_color_hybrid(
            &pushed,
            candidates,
            &needed,
            &old_to_new,
            &productive_transitions,
        );
        height_color_ms += color_started_at.elapsed().as_secs_f64() * 1000.0;

        let merge_started_at = Instant::now();
        let base_new_id = new_states.len() as u32;
        let num_colors = coloring.iter().max().map(|&c| c + 1).unwrap_or(0);

        // Map old states to new merged states
        for (idx, &color) in coloring.iter().enumerate() {
            old_to_new[candidates[idx]] = base_new_id + color as u32;
        }

        // Extend builders
        new_states.extend((0..num_colors).map(|_| MergedStateBuilder::default()));

        // Merge states into builders
        let builders = &mut new_states[base_new_id as usize..];
        for (idx, &color) in coloring.iter().enumerate() {
            merge_state_into_builder(
                candidates[idx],
                color,
                &pushed,
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
    }

    let reconstruct_started_at = Instant::now();
    let minimized = reconstruct_dwa(start_state, &old_to_new, new_states);
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
        build_exact_group_summary, final_weights_compatible_on_domain,
        memberwise_group_compatible, minimize_acyclic, push_weights,
        sorted_weights_compatible_on_domain,
        sorted_weights_compatible_on_domain_intersection,
        weight_is_disjoint_from_domain_intersection, weights_equal_on_domain,
        weights_equal_on_domain_intersection, ClassProfile, PointwiseBehaviorInterner,
        PointwiseRegionBuildCache, PointwiseRegionInterner, build_token_behavior_region,
    };
    use crate::automata::weighted_u32::dwa::{DWA, DWAState};
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
