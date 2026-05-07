//! Acyclic weighted DWA minimization via weight pushing and height-layered coloring.
//!
//! The pipeline pushes weights backward to discard dead token flow, groups
//! states by topological height, colors each height bucket subject to
//! compatibility constraints, and reconstructs the minimized automaton from the
//! merged buckets.
use std::sync::Arc;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;

use super::dwa::{DWA, DWAState};
use crate::ds::weight::Weight;

type Label = i32;

const UNMAPPED: u32 = u32::MAX;

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
    let n = dwa.states().len();
    if n == 0 {
        return (false, Some(Vec::new()), Vec::new());
    }

    // 1. Topological order (Kahn's algorithm)
    let Some(topo) = compute_topo_order(dwa) else {
        return (false, None, Vec::new()); // cyclic
    };

    // 2+3 combined: backward reachable sets + push transition weights.
    // In reverse topo order, each state's targets have already been processed,
    // so we compute reachable[u] and push transitions in a single pass.
    let mut reachable: Vec<Weight> = vec![Weight::empty(); n];
    let mut intersection_cache = FxHashMap::default();
    let mut changed = false;
    for &u in topo.iter().rev() {
        let state = &dwa.states()[u];
        let mut acc = state.final_weight.as_ref().cloned().unwrap_or_else(Weight::empty);
        let mut acc_full = acc.is_full();
        let mut pushed: Vec<(Label, u32, Option<Weight>)> = Vec::new();
        for (&lbl, (target, w)) in &state.transitions {
            let t = *target as usize;
            if t >= n {
                continue;
            }
            if reachable[t].is_full() {
                if !acc_full {
                    acc = acc.union(w);
                    acc_full = acc.is_full();
                }
                // w ∩ all = w, no push needed
            } else if reachable[t].is_empty() {
                // w ∩ empty = empty, remove transition
                pushed.push((lbl, *target, None));
                // Contributes nothing to acc
            } else {
                let new_w = memoized_intersection(&mut intersection_cache, w, &reachable[t]);
                if !acc_full {
                    acc = acc.union(&new_w);
                    acc_full = acc.is_full();
                }
                if new_w != *w {
                    pushed.push((lbl, *target, if new_w.is_empty() { None } else { Some(new_w) }));
                }
            }
        }
        reachable[u] = acc;

        for (lbl, target, new_w_opt) in pushed {
            if let Some(new_w) = new_w_opt {
                dwa.states_mut()[u].transitions.insert(lbl, (target, new_w));
            } else {
                dwa.states_mut()[u].transitions.remove(&lbl);
            }
            changed = true;
        }
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
            merged.push((*existing_label, existing_weight.union(add_weight)));
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
    // Step 1: Partition refinement to get fine-grained classes.
    let class_coloring = partition_refine_coloring_raw(candidates, dwa, old_to_new);
    let num_classes = class_coloring.iter().max().map(|&c| c + 1).unwrap_or(0);
    if num_classes <= 1 {
        return class_coloring;
    }

    // Step 2: Pick one representative state per class and compute the union
    // of needed sets for each class.
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

    let class_profiles: Vec<ClassProfile> = class_rep_state
        .iter()
        .map(|&rep| build_class_profile(rep, old_to_new, productive_transitions, dwa))
        .collect();

    // Step 3: Greedy merge of classes, handling both disjoint and overlapping
    // needed sets. Instead of building an O(K²) incompatibility graph, we check
    // each class against only the small number of existing groups (~14 for kb_684).
    //
    // For overlapping classes, we maintain "merged weights" per group — the union
    // of all members' weights. Since all group members were verified compatible
    // when added, they agree on overlapping TSIDs, so the union weight is a
    // valid consensus for checking new candidates.
    struct OverlapMergeGroup {
        needed_union: Weight,
        targets: Vec<(Label, u32)>,
        merged_final_weight: Option<Weight>,
        merged_transition_weights: Vec<(Label, Weight)>,
        member_classes: Vec<usize>,
    }

    let mut groups: Vec<OverlapMergeGroup> = Vec::new();

    for class in 0..num_classes {
        let cn = &class_needed_union[class];
        let class_profile = &class_profiles[class];

        let mut placed = false;
        for g in &mut groups {
            if !sorted_targets_compatible(&class_profile.targets, &g.targets) {
                continue;
            }

            let is_disjoint = cn.is_disjoint(&g.needed_union);

            if !is_disjoint {
                // Check weight compatibility on the overlap domain without
                // materializing cn ∩ g.needed_union as a Weight.
                let weight_ok = match (&class_profile.final_weight, &g.merged_final_weight) {
                    (Some(fw), Some(gfw)) => {
                        weights_equal_on_domain_intersection(fw, gfw, cn, &g.needed_union)
                    }
                    (Some(fw), None) => {
                        weight_is_disjoint_from_domain_intersection(fw, cn, &g.needed_union)
                    }
                    (None, Some(gfw)) => {
                        weight_is_disjoint_from_domain_intersection(gfw, cn, &g.needed_union)
                    }
                    (None, None) => true,
                };
                if !weight_ok {
                    continue;
                }

                if !sorted_weights_compatible_on_domain_intersection(
                    &class_profile.weights,
                    &g.merged_transition_weights,
                    cn,
                    &g.needed_union,
                ) {
                    continue;
                }
            }

            // Compatible — merge into this group
            g.needed_union = g.needed_union.union(cn);
            merge_sorted_targets(&mut g.targets, &class_profile.targets);
            if let Some(fw) = &class_profile.final_weight {
                g.merged_final_weight = Some(match g.merged_final_weight.take() {
                    Some(existing) => existing.union(fw),
                    None => fw.clone(),
                });
            }
            merge_sorted_weights(&mut g.merged_transition_weights, &class_profile.weights);
            g.member_classes.push(class);
            placed = true;
            break;
        }

        if !placed {
            groups.push(OverlapMergeGroup {
                needed_union: cn.clone(),
                targets: class_profile.targets.clone(),
                merged_final_weight: class_profile.final_weight.clone(),
                merged_transition_weights: class_profile.weights.clone(),
                member_classes: vec![class],
            });
        }
    }

    // Step 4: Map each candidate through class → merged color.
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
    //   For leaves: needed[u] = reachable[u] = final_weight[u].
    //   So final_on_needed = final_weight, and witness_domain = witness_final
    //   (both grow by the same final_weight each iteration).
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
    transitions: rustc_hash::FxHashMap<Label, (u32, Weight)>,
}

impl Default for MergedStateBuilder {
    fn default() -> Self {
        Self {
            final_weights_pending: Vec::new(),
            transitions: rustc_hash::FxHashMap::default(),
        }
    }
}

impl MergedStateBuilder {
    fn add_final_weight(&mut self, weight: &Weight) {
        self.final_weights_pending.push(weight.clone());
    }

    fn add_transition(&mut self, label: Label, target: u32, weight: Weight) {
        use std::collections::hash_map::Entry;
        match self.transitions.entry(label) {
            Entry::Occupied(mut entry) => {
                let (existing_target, existing_weight) = entry.get_mut();
                debug_assert_eq!(*existing_target, target);
                *existing_weight = existing_weight.union(&weight);
            }
            Entry::Vacant(entry) => {
                entry.insert((target, weight));
            }
        }
    }

    fn finalize_for_reuse(&mut self) {
        let pending = std::mem::take(&mut self.final_weights_pending);
        let built = batch_build_weight(pending);
        if !built.is_empty() {
            self.final_weights_pending.push(built);
        }
    }
}

/// Batch-build a Weight from a Vec of pending weights using a hybrid strategy.
fn batch_build_weight(pending: Vec<Weight>) -> Weight {
    match pending.len() {
        0 => Weight::empty(),
        1 => pending.into_iter().next().unwrap(),
        n if n <= 16 => Weight::union_all(pending.iter()),
        _ => {
            let mut current = pending;
            while current.len() > 16 {
                current = current
                    .chunks(16)
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
    let states: Vec<DWAState> = builders
        .into_iter()
        .map(|b| {
            let mut state = DWAState::default();
            let final_weight = b.final_weights_pending.into_iter().next().unwrap_or_else(Weight::empty);
            if !final_weight.is_empty() {
                state.final_weight = Some(final_weight);
            }
            for (lbl, (target, weight)) in b.transitions {
                if !weight.is_empty() {
                    state.transitions.insert(lbl, (target, weight));
                }
            }
            state
        })
        .collect();

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
    if dwa.states().is_empty() {
        return dwa.clone();
    }

    // Clone and push weights
    let mut pushed = dwa.clone();
    let (_, topo_from_push, reachable_from_push) = push_weights(&mut pushed);

    // Reuse topo order from push_weights (graph structure unchanged by push).
    let topo = match topo_from_push {
        Some(t) => t,
        None => return dwa.clone(), // cyclic — fall back
    };

    // Reuse backward-reachable token sets from push_weights as needed sets.
    // Proof: push_weights computes reachable[u] = final(u) ∪ union(w(u,t) ∩ reachable[t]).
    // a fresh needed-set pass on the pushed DWA uses the same recurrence (since
    // w_pushed = w_orig ∩ reachable[t], and A ∩ A = A in the needed recurrence).
    // Both produce identical results, so we skip the redundant recomputation.
    let start_state = pushed.start_state() as usize;
    let needed = reachable_from_push;
    if needed[start_state].is_empty() {
        return canonical_dead_dwa();
    }
    let productive_transitions = compute_productive_transitions(&pushed, &needed);
    let heights = compute_heights(&pushed, &topo);
    let max_height = heights.iter().max().copied().unwrap_or(0);

    let n = pushed.states().len();

    let reachable_from_start = compute_reachable_from_start(&pushed, start_state);

    // Group states by height (only reachable states with non-empty needed sets)
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

    // Bottom-up: color and merge
    let mut old_to_new = vec![UNMAPPED; n];
    let mut new_states: Vec<MergedStateBuilder> = Vec::new();

    for h in 0..=max_height {
        let candidates = &states_by_height[h];
        if candidates.is_empty() {
            continue;
        }
        if h == 0 && try_all_compatible_height_0_coloring(candidates, &pushed, &needed).is_some() {
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
            for builder in builders.iter_mut() {
                builder.finalize_for_reuse();
            }
            continue;
        }

        // Hybrid coloring: partition refinement to reduce candidates into
        // equivalence classes, then graph coloring among class representatives.
        // This is O(n log n + k²) where k ≤ n, always at least as fast as
        // direct O(n²) graph coloring.
        let coloring = build_and_color_hybrid(
            &pushed,
            candidates,
            &needed,
            &old_to_new,
            &productive_transitions,
        );

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
        for builder in builders.iter_mut() {
            builder.finalize_for_reuse();
        }
    }

    reconstruct_dwa(start_state, &old_to_new, new_states)
}

#[cfg(test)]
mod tests {
    use super::{
        weight_is_disjoint_from_domain_intersection, weights_equal_on_domain,
        weights_equal_on_domain_intersection,
    };
    use crate::ds::weight::Weight;
    use range_set_blaze::RangeSetBlaze;

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
