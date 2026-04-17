//! Acyclic weighted DWA minimization via weight pushing and height-layered coloring.
//!
//! The pipeline pushes weights backward to discard dead token flow, groups
//! states by topological height, colors each height bucket subject to
//! compatibility constraints, and reconstructs the minimized automaton from the
//! merged buckets.
use std::sync::{Arc, OnceLock};

use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use super::dwa::{DWA, DWAState};
use crate::ds::weight::Weight;

fn debug_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_DEBUG_PROFILE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

type Label = i32;

const UNMAPPED: u32 = u32::MAX;

fn mapped_target(old_to_new: &[u32], target: u32) -> Option<u32> {
    let mapped = old_to_new.get(target as usize).copied().unwrap_or(UNMAPPED);
    (mapped != UNMAPPED).then_some(mapped)
}

fn compute_reachable_from_start(dwa: &DWA, start_state: usize) -> Vec<bool> {
    let mut reachable = vec![false; dwa.states.len()];
    if start_state >= dwa.states.len() {
        return reachable;
    }

    let mut stack = vec![start_state];
    while let Some(state_id) = stack.pop() {
        if reachable[state_id] {
            continue;
        }

        reachable[state_id] = true;
        for (target, _) in dwa.states[state_id].transitions.values() {
            let target = *target as usize;
            if target < dwa.states.len() && !reachable[target] {
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
    let n = dwa.states.len();
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
        let state = &dwa.states[u];
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
                dwa.states[u].transitions.insert(lbl, (target, new_w));
            } else {
                dwa.states[u].transitions.remove(&lbl);
            }
            changed = true;
        }
    }
    (changed, Some(topo), reachable)
}

// Topological analysis.

fn compute_topo_order(dwa: &DWA) -> Option<Vec<usize>> {
    let n = dwa.states.len();
    let mut in_degree = vec![0u32; n];
    for state in &dwa.states {
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
        for (_, (target, _)) in &dwa.states[u].transitions {
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
fn compute_needed_sets(dwa: &DWA, topo: &[usize]) -> Vec<Weight> {
    let n = dwa.states.len();
    let mut needed = vec![Weight::empty(); n];
    for &u in topo.iter().rev() {
        let state = &dwa.states[u];
        let mut acc = state.final_weight.as_ref().cloned().unwrap_or_else(Weight::empty);
        for (_, (target, w)) in &state.transitions {
            let t = *target as usize;
            if t >= n {
                continue;
            }
            if needed[t].is_full() {
                acc = acc.union(w);
            } else if !needed[t].is_empty() {
                let tmp = w.intersection(&needed[t]);
                acc = acc.union(&tmp);
            }
            if acc.is_full() {
                break;
            }
        }
        needed[u] = acc;
    }
    needed
}

fn compute_heights(dwa: &DWA, topo: &[usize]) -> Vec<usize> {
    let n = dwa.states.len();
    let mut heights = vec![0usize; n];
    // Process in reverse topo order so children are resolved before parents
    for &u in topo.iter().rev() {
        heights[u] = dwa.states[u]
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
    let n = dwa.states.len();
    let mut result = Vec::with_capacity(n);

    for state in &dwa.states {
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
    let n = dwa.states.len();
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
        let fw_u = dwa.states[u].final_weight.as_ref();
        let fw_v = dwa.states[v].final_weight.as_ref();
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
        let fw_u = dwa.states[u].final_weight.as_ref();
        let fw_v = dwa.states[v].final_weight.as_ref();
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

/// Build incompatibility graph: adj[i] = list of incompatible candidate indices.
fn build_incompatibility_graph(
    dwa: &DWA,
    candidates: &[usize],
    needed: &[Weight],
    old_to_new: &[u32],
    productive_transitions: &[Vec<ProductiveTransition>],
) -> Vec<Vec<usize>> {
    let nc = candidates.len();
    let mut adj: Vec<Vec<usize>> = vec![vec![]; nc];

    for i in 0..nc {
        for j in (i + 1)..nc {
            if !are_compatible(
                candidates[i],
                candidates[j],
                dwa,
                needed,
                old_to_new,
                productive_transitions,
                false,
            ) {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }

    adj
}

fn overlapping_candidate_pairs(
    candidates: &[usize],
    needed: &[Weight],
) -> Option<FxHashSet<(usize, usize)>> {
    if candidates.len() < 100 {
        return None;
    }

    let mut segments: Vec<(u32, u32, usize, Arc<range_set_blaze::RangeSetBlaze<u32>>)> = Vec::new();
    for (idx, &candidate) in candidates.iter().enumerate() {
        let compact_entries = needed[candidate].compact_entries()?;
        for (start, end, tokens) in compact_entries {
            segments.push((start, end, idx, tokens));
        }
    }

    segments.sort_unstable_by_key(|(start, end, idx, _)| (*start, *end, *idx));

    let mut overlap_pairs = FxHashSet::default();
    let mut active: Vec<(u32, usize, Arc<range_set_blaze::RangeSetBlaze<u32>>)> = Vec::new();
    // Cache: for the current segment's token set, map active token set pointer → overlaps?
    let mut disjoint_cache: FxHashMap<usize, bool> = FxHashMap::default();

    for (start, end, idx, tokens) in segments {
        active.retain(|(active_end, _, _)| *active_end >= start);
        disjoint_cache.clear();

        for (_, active_idx, active_tokens) in &active {
            if *active_idx == idx {
                continue;
            }
            let active_ptr = Arc::as_ptr(active_tokens) as usize;
            let overlaps = match disjoint_cache.entry(active_ptr) {
                std::collections::hash_map::Entry::Occupied(e) => *e.get(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    let v = Arc::ptr_eq(active_tokens, &tokens)
                        || !active_tokens.is_disjoint(tokens.as_ref());
                    *e.insert(v)
                }
            };
            if overlaps {
                let pair = if *active_idx < idx {
                    (*active_idx, idx)
                } else {
                    (idx, *active_idx)
                };
                overlap_pairs.insert(pair);
            }
        }

        active.push((end, idx, tokens));
    }

    Some(overlap_pairs)
}

fn build_incompatibility_graph_sparse(
    dwa: &DWA,
    candidates: &[usize],
    needed: &[Weight],
    old_to_new: &[u32],
    productive_transitions: &[Vec<ProductiveTransition>],
) -> Option<Vec<Vec<usize>>> {
    let debug = debug_profile_enabled();
    let t0 = std::time::Instant::now();
    let overlap_pairs = overlapping_candidate_pairs(candidates, needed)?;
    let overlap_ms = if debug { t0.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };
    let num_overlap_pairs = overlap_pairs.len();

    let nc = candidates.len();
    let mut incompatible_pairs = FxHashSet::default();

    let t1 = if debug { std::time::Instant::now() } else { t0 };
    // Sparse needed-set overlap is not sufficient on its own: even disjoint-needed
    // states are incompatible if they expose the same label but remap it to
    // different targets. Seed the sparse graph with those label-target conflicts.
    let mut label_targets: rustc_hash::FxHashMap<Label, rustc_hash::FxHashMap<u32, Vec<usize>>> =
        rustc_hash::FxHashMap::default();
    for (candidate_idx, &state_id) in candidates.iter().enumerate() {
        for transition in &productive_transitions[state_id] {
            let Some(mapped) = mapped_target(old_to_new, transition.target) else {
                continue;
            };
            label_targets
                .entry(transition.label)
                .or_default()
                .entry(mapped)
                .or_default()
                .push(candidate_idx);
        }
    }

    for target_groups in label_targets.into_values() {
        if target_groups.len() <= 1 {
            continue;
        }
        let grouped_candidates: Vec<Vec<usize>> = target_groups.into_values().collect();
        for left_group_idx in 0..grouped_candidates.len() {
            for right_group_idx in (left_group_idx + 1)..grouped_candidates.len() {
                for &left_candidate in &grouped_candidates[left_group_idx] {
                    for &right_candidate in &grouped_candidates[right_group_idx] {
                        let pair = if left_candidate < right_candidate {
                            (left_candidate, right_candidate)
                        } else {
                            (right_candidate, left_candidate)
                        };
                        incompatible_pairs.insert(pair);
                    }
                }
            }
        }
    }
    let label_conflict_ms = if debug { t1.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };
    let label_conflict_pairs = incompatible_pairs.len();

    let t2 = if debug { std::time::Instant::now() } else { t0 };
    let additional_incompatible: Vec<(usize, usize)> = overlap_pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            if incompatible_pairs.contains(&(i, j)) {
                return None;
            }
            if !are_compatible(
                candidates[i],
                candidates[j],
                dwa,
                needed,
                old_to_new,
                productive_transitions,
                true,
            ) {
                Some((i, j))
            } else {
                None
            }
        })
        .collect();
    let compat_ms = if debug { t2.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

    for pair in additional_incompatible {
        incompatible_pairs.insert(pair);
    }

    if debug {
        eprintln!("[glrmask/debug][sparse_graph] candidates={} overlap_pairs={} overlap_ms={:.1} label_conflict_pairs={} label_ms={:.1} compat_check_ms={:.1} total_incompat={}",
            nc, num_overlap_pairs, overlap_ms, label_conflict_pairs, label_conflict_ms, compat_ms, incompatible_pairs.len());
    }

    let mut adj: Vec<Vec<usize>> = vec![vec![]; nc];
    for (i, j) in incompatible_pairs {
        adj[i].push(j);
        adj[j].push(i);
    }

    Some(adj)
}

/// Compute a 128-bit signature for a candidate state.
///
/// Two states with the same signature are guaranteed compatible (they have
/// identical final weights on their needed domain, and identical productive
/// transitions with the same mapped targets and weights).  This lets us
/// deduplicate candidates before building the O(n²) incompatibility graph.
fn compute_state_signature(
    state_id: usize,
    dwa: &DWA,
    needed: &[Weight],
    old_to_new: &[u32],
    productive_transitions: &[Vec<ProductiveTransition>],
) -> u64 {
    use std::hash::{Hash, Hasher};
    use rustc_hash::FxHasher;

    let state = &dwa.states[state_id];

    let mut hasher = FxHasher::default();

    // Hash final weight restricted to needed set
    match &state.final_weight {
        Some(fw) => {
            1u8.hash(&mut hasher);
            let restricted = fw.intersection(&needed[state_id]);
            restricted.hash(&mut hasher);
        }
        None => {
            0u8.hash(&mut hasher);
        }
    }

    // Hash productive transitions (already sorted by label from compute_productive_transitions)
    let trans = &productive_transitions[state_id];
    trans.len().hash(&mut hasher);
    for t in trans {
        t.label.hash(&mut hasher);
        let mapped = mapped_target(old_to_new, t.target).unwrap_or(UNMAPPED);
        mapped.hash(&mut hasher);
        t.weight.hash(&mut hasher);
    }

    hasher.finish()
}

/// Color candidates using signature-based deduplication and a strategy that
/// adapts based on the bucket: greedy-without-graph for smaller sets, and
/// sparse overlap graph + greedy coloring for larger sets.
#[allow(dead_code)]
fn build_and_color_with_signature_dedup(
    dwa: &DWA,
    candidates: &[usize],
    needed: &[Weight],
    old_to_new: &[u32],
    productive_transitions: &[Vec<ProductiveTransition>],
) -> Vec<usize> {
    let nc = candidates.len();
    let debug = debug_profile_enabled();

    let t0 = std::time::Instant::now();
    let signatures: Vec<u64> = candidates
        .iter()
        .map(|&id| compute_state_signature(id, dwa, needed, old_to_new, productive_transitions))
        .collect();
    let sig_ms = if debug { t0.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

    let mut sig_to_rep_idx: FxHashMap<u64, usize> = FxHashMap::default();
    let mut unique_indices: Vec<usize> = Vec::new(); // indices into candidates
    for (idx, &sig) in signatures.iter().enumerate() {
        sig_to_rep_idx.entry(sig).or_insert_with(|| {
            let rep = unique_indices.len();
            unique_indices.push(idx);
            rep
        });
    }

    let rep_candidates: Vec<usize> = unique_indices.iter().map(|&i| candidates[i]).collect();

    // If signature dedup didn't reduce enough, use hybrid approach:
    // partition refinement to get classes, then graph coloring among class reps.
    const HYBRID_THRESHOLD: usize = 200;
    if rep_candidates.len() > HYBRID_THRESHOLD {
        return build_and_color_hybrid(dwa, candidates, needed, old_to_new, productive_transitions);
    }

    // Build incompatibility graph and greedy-color it.
    let t1 = if debug { std::time::Instant::now() } else { t0 };
    let rep_adj = build_incompatibility_graph_sparse(
        dwa,
        &rep_candidates,
        needed,
        old_to_new,
        productive_transitions,
    )
    .unwrap_or_else(|| {
        build_incompatibility_graph(
            dwa,
            &rep_candidates,
            needed,
            old_to_new,
            productive_transitions,
        )
    });
    let graph_ms = if debug { t1.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

    let t2 = if debug { std::time::Instant::now() } else { t0 };
    let rep_coloring = greedy_coloring(&rep_adj);
    if debug {
        let color_ms = t2.elapsed().as_secs_f64() * 1000.0;
        let num_colors = rep_coloring.iter().max().map(|c| c + 1).unwrap_or(0);
        let total_edges: usize = rep_adj.iter().map(|a| a.len()).sum::<usize>() / 2;
        eprintln!("[glrmask/debug][minimize_dedup] candidates={} unique_reps={} sig_ms={:.1} graph_ms={:.1} edges={} color_ms={:.1} colors={}",
            nc, rep_candidates.len(), sig_ms, graph_ms, total_edges, color_ms, num_colors);
    }

    let mut coloring = vec![0usize; nc];
    for (idx, &sig) in signatures.iter().enumerate() {
        let rep = sig_to_rep_idx[&sig];
        coloring[idx] = rep_coloring[rep];
    }

    coloring
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
    let debug = debug_profile_enabled();
    let t0 = std::time::Instant::now();

    // Step 1: Partition refinement to get fine-grained classes.
    let class_coloring = partition_refine_coloring_raw(candidates, dwa, old_to_new);
    let num_classes = class_coloring.iter().max().map(|&c| c + 1).unwrap_or(0);
    let _refine_ms = if debug { t0.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

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

    // Step 3: Greedy merge of classes, handling both disjoint and overlapping
    // needed sets. Instead of building an O(K²) incompatibility graph, we check
    // each class against only the small number of existing groups (~14 for kb_684).
    //
    // For overlapping classes, we maintain "merged weights" per group — the union
    // of all members' weights. Since all group members were verified compatible
    // when added, they agree on overlapping TSIDs, so the union weight is a
    // valid consensus for checking new candidates.
    let t1 = if debug { std::time::Instant::now() } else { t0 };

    struct OverlapMergeGroup {
        needed_union: Weight,
        target_map: rustc_hash::FxHashMap<i32, u32>,
        merged_final_weight: Option<Weight>,
        merged_transition_weights: rustc_hash::FxHashMap<i32, Weight>,
        member_classes: Vec<usize>,
    }

    let mut groups: Vec<OverlapMergeGroup> = Vec::new();

    for class in 0..num_classes {
        let rep = class_rep_state[class];
        let cn = &class_needed_union[class];

        // Build this class's target profile
        let tp: Vec<(i32, u32)> = productive_transitions[rep]
            .iter()
            .filter_map(|pt| {
                mapped_target(old_to_new, pt.target).map(|mt| (pt.label, mt))
            })
            .collect();

        let mut placed = false;
        for g in &mut groups {
            // Check target compatibility: no shared label maps to different targets
            let mut target_compat = true;
            for &(label, target) in &tp {
                if let Some(&existing_target) = g.target_map.get(&label) {
                    if existing_target != target {
                        target_compat = false;
                        break;
                    }
                }
            }
            if !target_compat {
                continue;
            }

            let is_disjoint = cn.is_disjoint(&g.needed_union);

            if !is_disjoint {
                // Check weight compatibility on the overlap domain.
                let overlap = cn.intersection(&g.needed_union);

                // Check final weights on overlap
                let fw_class = dwa.states[rep].final_weight.as_ref();
                let weight_ok = match (fw_class, &g.merged_final_weight) {
                    (Some(fw), Some(gfw)) => weights_equal_on_domain(fw, gfw, &overlap),
                    (Some(fw), None) => fw.is_disjoint(&overlap),
                    (None, Some(gfw)) => gfw.is_disjoint(&overlap),
                    (None, None) => true,
                };
                if !weight_ok {
                    continue;
                }

                // Check transition weights on overlap.
                // Labels present in the class:
                let mut trans_compat = true;
                let trans = &productive_transitions[rep];
                for pt in trans {
                    if let Some(gw) = g.merged_transition_weights.get(&pt.label) {
                        // Both have this label — check weights agree on overlap
                        if &pt.weight == gw {
                            continue;
                        }
                        let c_disj = pt.weight.is_disjoint(&overlap);
                        let g_disj = gw.is_disjoint(&overlap);
                        if c_disj && g_disj {
                            continue;
                        }
                        if c_disj != g_disj {
                            trans_compat = false;
                            break;
                        }
                        if !weights_equal_on_domain(&pt.weight, gw, &overlap) {
                            trans_compat = false;
                            break;
                        }
                    } else {
                        // Only class has this label — weight must not touch overlap
                        if !pt.weight.is_disjoint(&overlap) {
                            trans_compat = false;
                            break;
                        }
                    }
                }
                if trans_compat {
                    // Labels present in group but not in class
                    for (label, gw) in &g.merged_transition_weights {
                        if trans.iter().any(|pt| pt.label == *label) {
                            continue;
                        }
                        if !gw.is_disjoint(&overlap) {
                            trans_compat = false;
                            break;
                        }
                    }
                }
                if !trans_compat {
                    continue;
                }
            }

            // Compatible — merge into this group
            g.needed_union = g.needed_union.union(cn);
            for &(label, target) in &tp {
                g.target_map.insert(label, target);
            }
            if let Some(fw) = &dwa.states[rep].final_weight {
                g.merged_final_weight = Some(match g.merged_final_weight.take() {
                    Some(existing) => existing.union(fw),
                    None => fw.clone(),
                });
            }
            for pt in &productive_transitions[rep] {
                let entry = g.merged_transition_weights
                    .entry(pt.label)
                    .or_insert_with(Weight::empty);
                *entry = entry.union(&pt.weight);
            }
            g.member_classes.push(class);
            placed = true;
            break;
        }

        if !placed {
            let mut target_map = rustc_hash::FxHashMap::default();
            for &(label, target) in &tp {
                target_map.insert(label, target);
            }
            let merged_final_weight = dwa.states[rep].final_weight.clone();
            let mut merged_transition_weights = rustc_hash::FxHashMap::default();
            for pt in &productive_transitions[rep] {
                merged_transition_weights.insert(pt.label, pt.weight.clone());
            }
            groups.push(OverlapMergeGroup {
                needed_union: cn.clone(),
                target_map,
                merged_final_weight,
                merged_transition_weights,
                member_classes: vec![class],
            });
        }
    }

    let _merge_ms = if debug { t1.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

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

/// Greedy graph coloring — O(V + E).
fn greedy_coloring(adj: &[Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    if n == 0 {
        return vec![];
    }

    let mut colors = vec![usize::MAX; n];

    // Sort by decreasing degree
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| std::cmp::Reverse(adj[i].len()));

    let mut used = vec![false; n + 1]; // scratch buffer for neighbor colors

    for &u in &order {
        for &v in &adj[u] {
            if colors[v] != usize::MAX {
                used[colors[v]] = true;
            }
        }
        let mut c = 0;
        while used[c] {
            c += 1;
        }
        colors[u] = c;
        // Reset scratch
        for &v in &adj[u] {
            if colors[v] != usize::MAX {
                used[colors[v]] = false;
            }
        }
    }

    colors
}

/// Partition refinement coloring: group candidates by (final_weight, transitions).
/// Two candidates get the same color iff they have identical final weights and
/// identical productive transitions (same labels, same mapped targets, same weights).
///
/// Merge colors whose needed-set unions are disjoint and whose target
/// profiles don't conflict on any shared label. This recovers the "diamond"
/// optimization that exploits disjoint token domains.
fn partition_refine_coloring(
    candidates: &[usize],
    dwa: &DWA,
    needed: &[Weight],
    old_to_new: &[u32],
    productive_transitions: &[Vec<ProductiveTransition>],
) -> Vec<usize> {
    use std::hash::{Hash, Hasher};
    use rustc_hash::FxHasher;

    let nc = candidates.len();

    // Start from exact-signature partitions.
    let mut hash_groups: rustc_hash::FxHashMap<u64, Vec<usize>> =
        rustc_hash::FxHashMap::default();

    for idx in 0..nc {
        let c = candidates[idx];
        let mut hasher = FxHasher::default();
        dwa.states[c].final_weight.hash(&mut hasher);
        for pt in &productive_transitions[c] {
            pt.label.hash(&mut hasher);
            let mapped = old_to_new[pt.target as usize];
            mapped.hash(&mut hasher);
            pt.weight.hash(&mut hasher);
        }
        productive_transitions[c].len().hash(&mut hasher);
        let sig = hasher.finish();
        hash_groups.entry(sig).or_default().push(idx);
    }

    // Assign initial colors (verify within hash groups for collisions)
    let mut colors = vec![0usize; nc];
    let mut next_color = 0usize;
    // Per-color: representative candidate index, needed_union
    let mut color_rep: Vec<usize> = Vec::new(); // candidate index of representative
    let mut color_needed: Vec<Weight> = Vec::new();

    for group in hash_groups.values() {
        // Sub-group by actual equality
        let mut sub_groups: Vec<Vec<usize>> = Vec::new();
        'outer: for &idx in group {
            let c = candidates[idx];
            for sub in &mut sub_groups {
                let rep = candidates[sub[0]];
                if states_signature_equal(c, rep, dwa, old_to_new, productive_transitions) {
                    sub.push(idx);
                    continue 'outer;
                }
            }
            sub_groups.push(vec![idx]);
        }

        for sub in &sub_groups {
            let color = next_color;
            next_color += 1;
            // Compute needed_union for this color
            let mut nu = Weight::empty();
            for &idx in sub {
                colors[idx] = color;
                nu = nu.union(&needed[candidates[idx]]);
            }
            color_rep.push(candidates[sub[0]]);
            color_needed.push(nu);
        }
    }

    let num_initial_colors = next_color;
    if num_initial_colors <= 1 {
        return colors;
    }

    // Merge colors with disjoint needed sets and compatible targets.
    // For each color, the target_profile is the sorted (label, mapped_target) list.
    // Two colors can merge if:
    //   1. No label conflict: no shared label has different mapped targets
    //   2. Their group's needed_union stays disjoint from each other

    // Greedy grouping: iterate colors, try to place each in an existing group.
    struct MergeGroup {
        needed_union: Weight,
        // For quick label-target conflict check, store the set of (label, target) from all
        // member colors. Since disjoint-needed colors may have different target profiles,
        // we accumulate the union.
        target_map: rustc_hash::FxHashMap<i32, u32>, // label -> mapped_target (unique per group)
        member_colors: Vec<usize>,
    }

    let mut groups: Vec<MergeGroup> = Vec::new();

    for c in 0..num_initial_colors {
        let rep = color_rep[c];
        let cn = &color_needed[c];

        // Build this color's target profile
        let tp: Vec<(i32, u32)> = productive_transitions[rep]
            .iter()
            .map(|pt| (pt.label, old_to_new[pt.target as usize]))
            .collect();

        let mut placed = false;
        for g in &mut groups {
            // Check disjointness
            if !cn.is_disjoint(&g.needed_union) {
                continue;
            }
            // Check target compatibility
            let mut compat = true;
            for &(label, target) in &tp {
                if let Some(&existing_target) = g.target_map.get(&label) {
                    if existing_target != target {
                        compat = false;
                        break;
                    }
                }
            }
            if !compat {
                continue;
            }

            // Merge into this group
            g.needed_union = g.needed_union.union(cn);
            for &(label, target) in &tp {
                g.target_map.insert(label, target);
            }
            g.member_colors.push(c);
            placed = true;
            break;
        }

        if !placed {
            let mut target_map = rustc_hash::FxHashMap::default();
            for &(label, target) in &tp {
                target_map.insert(label, target);
            }
            groups.push(MergeGroup {
                needed_union: cn.clone(),
                target_map,
                member_colors: vec![c],
            });
        }
    }

    // Remap colors based on merged groups
    let mut color_to_group = vec![0usize; num_initial_colors];
    for (gid, g) in groups.iter().enumerate() {
        for &c in &g.member_colors {
            color_to_group[c] = gid;
        }
    }
    for idx in 0..nc {
        colors[idx] = color_to_group[colors[idx]];
    }

    colors
}

/// Check if two states have identical signatures (final weight + transitions).
fn states_signature_equal(
    u: usize,
    v: usize,
    dwa: &DWA,
    old_to_new: &[u32],
    productive_transitions: &[Vec<ProductiveTransition>],
) -> bool {
    // Check final weights
    if dwa.states[u].final_weight != dwa.states[v].final_weight {
        return false;
    }

    // Check productive transitions
    let trans_u = &productive_transitions[u];
    let trans_v = &productive_transitions[v];
    if trans_u.len() != trans_v.len() {
        return false;
    }
    for (tu, tv) in trans_u.iter().zip(trans_v.iter()) {
        if tu.label != tv.label {
            return false;
        }
        if mapped_target(old_to_new, tu.target) != mapped_target(old_to_new, tv.target) {
            return false;
        }
        if tu.weight != tv.weight {
            return false;
        }
    }
    true
}

/// Partition refinement using raw DWA transitions (no needed sets).
/// Groups candidates by exact (final_weight, transitions) signature.
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
        dwa.states[c].final_weight.hash(&mut hasher);
        // Hash raw transitions (BTreeMap iterates in label order)
        for (&label, (target, weight)) in &dwa.states[c].transitions {
            let Some(mapped) = mapped_target(old_to_new, *target) else {
                continue;
            };
            label.hash(&mut hasher);
            mapped.hash(&mut hasher);
            weight.hash(&mut hasher);
        }
        dwa.states[c].transitions.len().hash(&mut hasher);

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
    if dwa.states[u].final_weight != dwa.states[v].final_weight {
        return false;
    }
    let su = &dwa.states[u];
    let sv = &dwa.states[v];
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
        .all(|&id| dwa.states[id].transitions.is_empty())
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
    let old_state = &dwa.states[old_id];

    // Union final weights
    if let Some(fw) = &old_state.final_weight {
        builder.add_final_weight(fw);
    }

    // Merge transitions
    let n = dwa.states.len();
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
    DWA {
        states,
        start_state: if start_new == UNMAPPED { 0 } else { start_new },
    }
}

// Public API.

/// Default threshold: always use the full incompatibility graph approach.
const PARTITION_REFINE_THRESHOLD: usize = usize::MAX;

/// Minimize an acyclic DWA using weight pushing + graph-coloring.
///
/// Falls back to the caller's DWA unchanged if the input is cyclic.
pub fn minimize_acyclic(dwa: &DWA) -> DWA {
    minimize_acyclic_with_threshold(dwa, PARTITION_REFINE_THRESHOLD)
}

/// Like [`minimize_acyclic`], but switches from the O(n²) incompatibility
/// graph to partition-refinement coloring when a height bucket has more than
/// `partition_refine_threshold` candidates.
pub fn minimize_acyclic_with_threshold(dwa: &DWA, _partition_refine_threshold: usize) -> DWA {
    if dwa.states.is_empty() {
        return dwa.clone();
    }

    let debug = debug_profile_enabled();
    let t_start = std::time::Instant::now();

    // Clone and push weights
    let mut pushed = dwa.clone();
    let (_, topo_from_push, reachable_from_push) = push_weights(&mut pushed);

    let push_ms = if debug { t_start.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

    // Reuse topo order from push_weights (graph structure unchanged by push).
    let topo = match topo_from_push {
        Some(t) => t,
        None => return dwa.clone(), // cyclic — fall back
    };

    // Reuse backward-reachable token sets from push_weights as needed sets.
    // Proof: push_weights computes reachable[u] = final(u) ∪ union(w(u,t) ∩ reachable[t]).
    // compute_needed_sets on pushed DWA uses the same recurrence (since
    // w_pushed = w_orig ∩ reachable[t], and A ∩ A = A in the needed recurrence).
    // Both produce identical results, so we skip the redundant recomputation.
    let needed = reachable_from_push;
    let t1 = if debug { std::time::Instant::now() } else { t_start };
    let productive_transitions = compute_productive_transitions(&pushed, &needed);
    let prod_ms = if debug { t1.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };
    let heights = compute_heights(&pushed, &topo);
    let max_height = heights.iter().max().copied().unwrap_or(0);

    let n = pushed.states.len();
    let start_state = pushed.start_state as usize;

    let reachable_from_start = compute_reachable_from_start(&pushed, start_state);

    // Group states by height (only reachable states with non-empty needed sets)
    let mut states_by_height: Vec<Vec<usize>> = vec![vec![]; max_height + 1];
    for (id, &h) in heights.iter().enumerate() {
        if !reachable_from_start[id] {
            continue;
        }
        if needed[id].is_empty() && id != start_state {
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

    let t_recon = if debug { std::time::Instant::now() } else { t_start };
    let minimized = reconstruct_dwa(start_state, &old_to_new, new_states);
    if debug {
        let recon_ms = t_recon.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        let coloring_ms = total_ms - push_ms - prod_ms - recon_ms;
        eprintln!("[glrmask/debug][minimize_acyclic] states={} push_ms={:.1} prod_ms={:.1} coloring_ms={:.1} recon_ms={:.1} total_ms={:.1} minimized_states={}",
            dwa.states.len(), push_ms, prod_ms, coloring_ms, recon_ms, total_ms, minimized.states.len());
    }
    minimized
}

/// Fast minimize using signature-based partition refinement.
/// Skips push_weights and needed-set computation. Uses raw DWA transitions
/// for signatures and weight merging. Much faster but may produce slightly
/// larger output than the full graph-coloring approach.
pub fn minimize_acyclic_fast(dwa: &DWA) -> DWA {
    if dwa.states.is_empty() {
        return dwa.clone();
    }

    let Some(topo) = compute_topo_order(dwa) else {
        return dwa.clone(); // cyclic
    };
    let heights = compute_heights(dwa, &topo);
    let max_height = heights.iter().max().copied().unwrap_or(0);

    let n = dwa.states.len();
    let start_state = dwa.start_state as usize;

    let reachable_from_start = compute_reachable_from_start(dwa, start_state);

    let mut states_by_height: Vec<Vec<usize>> = vec![vec![]; max_height + 1];
    for (id, &h) in heights.iter().enumerate() {
        if !reachable_from_start[id] { continue; }
        states_by_height[h].push(id);
    }

    let mut old_to_new = vec![UNMAPPED; n];
    let mut next_new_id: u32 = 0;
    let mut any_merging = false;

    // Assign new IDs via partition refinement.
    for h in 0..=max_height {
        let candidates = &states_by_height[h];
        if candidates.is_empty() { continue; }

        let coloring = partition_refine_coloring_raw(
            candidates,
            dwa,
            &old_to_new,
        );

        let base_new_id = next_new_id;
        let num_colors = coloring.iter().max().map(|&c| c + 1).unwrap_or(0);

        if num_colors < candidates.len() {
            any_merging = true;
        }

        for (idx, &color) in coloring.iter().enumerate() {
            old_to_new[candidates[idx]] = base_new_id + color as u32;
        }
        next_new_id = base_new_id + num_colors as u32;
    }

    // Build the output DWA.
    let minimized = if !any_merging {
        // Fast path: no merging happened — just remap state IDs without cloning weights.
        let total_new = next_new_id as usize;
        let mut new_states: Vec<DWAState> = (0..total_new).map(|_| DWAState::default()).collect();
        for (old_id, &new_id) in old_to_new.iter().enumerate() {
            if new_id == UNMAPPED { continue; }
            let old_state = &dwa.states[old_id];
            let ns = &mut new_states[new_id as usize];
            ns.final_weight = old_state.final_weight.clone();
            for (&label, (target_raw, w)) in &old_state.transitions {
                let t = *target_raw as usize;
                if t >= n { continue; }
                let target_new = old_to_new[t];
                if target_new == UNMAPPED { continue; }
                ns.transitions.insert(label, (target_new, w.clone()));
            }
        }
        let start_new = old_to_new[start_state];
        DWA {
            states: new_states,
            start_state: if start_new == UNMAPPED { 0 } else { start_new },
        }
    } else {
        // Slow path: states were merged — use MergedStateBuilder.
        let mut new_states: Vec<MergedStateBuilder> = (0..next_new_id as usize)
            .map(|_| MergedStateBuilder::default())
            .collect();
        for (old_id, &new_id) in old_to_new.iter().enumerate() {
            if new_id == UNMAPPED { continue; }
            let builder = &mut new_states[new_id as usize];
            let old_state = &dwa.states[old_id];

            if let Some(fw) = &old_state.final_weight {
                builder.add_final_weight(fw);
            }

            for (&label, (target_raw, w_orig)) in &old_state.transitions {
                let t = *target_raw as usize;
                if t >= n { continue; }
                let target_new = old_to_new[t];
                if target_new == UNMAPPED { continue; }
                if !w_orig.is_empty() {
                    builder.add_transition(label, target_new, w_orig.clone());
                }
            }
        }
        for builder in new_states.iter_mut() {
            builder.finalize_for_reuse();
        }
        reconstruct_dwa(start_state, &old_to_new, new_states)
    };
    minimized
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::weighted_u32::dwa::DWA;
    use crate::ds::weight::Weight;
    use range_set_blaze::RangeSetBlaze;

    fn singleton_weight(token: u32) -> Weight {
        Weight::from_token_set_for_tsid(0, RangeSetBlaze::from_iter([token]))
    }

    #[test]
    fn test_sparse_graph_marks_label_target_conflicts_for_disjoint_needed_sets() {
        let mut dwa = DWA::new(1, 256);
        let target_left = dwa.add_state();
        let target_right = dwa.add_state();
        let mut candidates = Vec::new();

        for token in 0..101u32 {
            let candidate = dwa.add_state();
            let target = if token % 2 == 0 { target_left } else { target_right };
            let weight = singleton_weight(token);
            dwa.add_transition(candidate, 0, target, weight);
            candidates.push(candidate as usize);
        }

        let mut needed = vec![Weight::empty(); dwa.states.len()];
        needed[target_left as usize] = singleton_weight(10_000);
        needed[target_right as usize] = singleton_weight(10_001);
        for token in 0..101u32 {
            needed[candidates[token as usize]] = singleton_weight(token);
        }

        let productive_transitions = compute_productive_transitions(&dwa, &needed);
        let mut old_to_new = vec![UNMAPPED; dwa.states.len()];
        old_to_new[target_left as usize] = 9;
        old_to_new[target_right as usize] = 10;

        let adj = build_incompatibility_graph_sparse(
            &dwa,
            &candidates,
            &needed,
            &old_to_new,
            &productive_transitions,
        )
        .expect("candidate bucket should be large enough for sparse graph mode");

        assert!(
            adj[0].contains(&1) && adj[1].contains(&0),
            "disjoint-needed candidates with the same label but different remapped targets must be incompatible"
        );
    }
}
