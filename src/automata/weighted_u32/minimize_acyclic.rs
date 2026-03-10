//! Graph-coloring-based acyclic DWA minimization.
//!
//! Ported from sep1's `dwa_i32/minimization/dwa_acyclic`.
//!
//! The algorithm works in these phases:
//! 1. **Weight pushing** — intersect each transition weight with the backward
//!    reachable set of its target, removing tokens that can never reach acceptance.
//! 2. **Needed-set computation** — for each state, which token combinations can
//!    flow from that state to any final state.
//! 3. **Height-layered graph coloring** — process states bottom-up by topological
//!    height. At each height, build an incompatibility graph and color it. States
//!    with the same color get merged ("diamond" optimization).
//! 4. **Reconstruction** — build the minimized DWA from merged state builders.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::sync::Arc;

use super::dwa::{DWA, DWAState};
use crate::ds::weight::{Weight, WeightBuilder};

type Label = i32;

const UNMAPPED: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// Phase 0: Weight pushing (backward reachability pruning)
// ---------------------------------------------------------------------------

/// Push weights: intersect each transition weight with the backward-reachable
/// set of its target state.  This ensures transitions only carry tokens that
/// can actually reach acceptance, enabling more state merges.
///
/// Returns true if any weight was changed.
pub fn push_weights(dwa: &mut DWA) -> bool {
    let n = dwa.states.len();
    if n == 0 {
        return false;
    }

    // 1. Topological order (Kahn's algorithm)
    let Some(topo) = compute_topo_order(dwa) else {
        return false; // cyclic
    };

    // 2. Backward reachable sets (reverse topo order = leaves first)
    let mut reachable: Vec<Weight> = vec![Weight::empty(); n];
    for &u in topo.iter().rev() {
        let st = &dwa.states[u];
        let mut acc = WeightBuilder::new();
        if let Some(final_weight) = &st.final_weight {
            acc.union_weight(final_weight);
        }
        for (_, (target, w)) in &st.transitions {
            let t = *target as usize;
            if t >= n {
                continue;
            }
            if reachable[t].is_full() {
                acc.union_weight(w);
            } else if !reachable[t].is_empty() {
                let tmp = w.intersection(&reachable[t]);
                acc.union_weight(&tmp);
            }
            if acc.is_full() {
                break;
            }
        }
        reachable[u] = acc.build();
    }

    // 3. Intersect each transition weight with reachable[target]
    let mut changed = false;
    for u in 0..n {
        let targets_weights: Vec<(Label, u32, Weight)> = dwa.states[u]
            .transitions
            .iter()
            .map(|(&lbl, (t, w))| (lbl, *t, w.clone()))
            .collect();

        for (lbl, target, w) in targets_weights {
            let t = target as usize;
            if t >= n {
                continue;
            }
            if reachable[t].is_full() {
                continue;
            }
            let new_w = if reachable[t].is_empty() {
                Weight::empty()
            } else {
                w.intersection(&reachable[t])
            };
            if new_w != w {
                if new_w.is_empty() {
                    dwa.states[u].transitions.remove(&lbl);
                } else {
                    dwa.states[u].transitions.insert(lbl, (target, new_w));
                }
                changed = true;
            }
        }
    }
    changed
}

// ---------------------------------------------------------------------------
// Phase 1-2: Topological analysis
// ---------------------------------------------------------------------------

fn compute_topo_order(dwa: &DWA) -> Option<Vec<usize>> {
    let n = dwa.states.len();
    let mut in_degree = vec![0u32; n];
    for st in &dwa.states {
        for (_, (target, _)) in &st.transitions {
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
        let st = &dwa.states[u];
        let mut acc = WeightBuilder::new();
        if let Some(final_weight) = &st.final_weight {
            acc.union_weight(final_weight);
        }
        for (_, (target, w)) in &st.transitions {
            let t = *target as usize;
            if t >= n {
                continue;
            }
            if needed[t].is_full() {
                acc.union_weight(w);
            } else if !needed[t].is_empty() {
                let tmp = w.intersection(&needed[t]);
                acc.union_weight(&tmp);
            }
            if acc.is_full() {
                break;
            }
        }
        needed[u] = acc.build();
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

// ---------------------------------------------------------------------------
// Phase 3: Incompatibility graph + graph coloring
// ---------------------------------------------------------------------------

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
) -> bool {
    let needed_u = &needed[u];
    let needed_v = &needed[v];

    let domain_disjoint = needed_u.is_disjoint(needed_v);

    // Compute overlap lazily — only needed for non-disjoint case
    let overlap = if domain_disjoint {
        Weight::empty()
    } else {
        needed_u.intersection(needed_v)
    };

    // Check final weights on the overlapping domain (skip if disjoint)
    if !domain_disjoint {
        let fw_u = dwa.states[u]
            .final_weight
            .as_ref()
            .map(|w| w.intersection(&overlap))
            .unwrap_or_else(Weight::empty);
        let fw_v = dwa.states[v]
            .final_weight
            .as_ref()
            .map(|w| w.intersection(&overlap))
            .unwrap_or_else(Weight::empty);
        if fw_u != fw_v {
            return false;
        }
    }

    // Check transitions
    let n = dwa.states.len();
    let trans_u = &dwa.states[u].transitions;
    let trans_v = &dwa.states[v].transitions;
    let mut iter_u = trans_u.iter().peekable();
    let mut iter_v = trans_v.iter().peekable();

    while iter_u.peek().is_some() || iter_v.peek().is_some() {
        let (entry_u, entry_v) = match (iter_u.peek(), iter_v.peek()) {
            (Some(&(lu, _)), Some(&(lv, _))) => {
                if lu == lv {
                    (iter_u.next(), iter_v.next())
                } else if lu < lv {
                    (iter_u.next(), None)
                } else {
                    (None, iter_v.next())
                }
            }
            (Some(_), None) => (iter_u.next(), None),
            (None, Some(_)) => (None, iter_v.next()),
            (None, None) => break,
        };

        let (target_u, w_u_full) = match entry_u {
            Some((_, (t, w))) => {
                let t = *t as usize;
                if t < n { (Some(t), Some(w)) } else { (None, None) }
            }
            None => (None, None),
        };
        let (target_v, w_v_full) = match entry_v {
            Some((_, (t, w))) => {
                let t = *t as usize;
                if t < n { (Some(t), Some(w)) } else { (None, None) }
            }
            None => (None, None),
        };

        // Map targets to new IDs
        let mapped_u = target_u.and_then(|t| {
            let m = old_to_new[t];
            (m != UNMAPPED).then_some(m)
        });
        let mapped_v = target_v.and_then(|t| {
            let m = old_to_new[t];
            (m != UNMAPPED).then_some(m)
        });

        // CRITICAL: if both have mapped targets that differ and either has
        // non-empty weight, the builder can only store one target.
        // The states must NOT be merged.
        match (mapped_u, mapped_v) {
            (Some(mu), Some(mv)) if mu != mv => {
                let has_u = w_u_full.is_some_and(|w| !w.is_empty());
                let has_v = w_v_full.is_some_and(|w| !w.is_empty());
                if has_u || has_v {
                    return false;
                }
            }
            _ => {}
        }

        // If domains aren't disjoint, also check overlap weights and targets
        if !domain_disjoint {
            let w_u_overlap = w_u_full
                .map(|w| w.intersection(&overlap))
                .unwrap_or_else(Weight::empty);
            let w_v_overlap = w_v_full
                .map(|w| w.intersection(&overlap))
                .unwrap_or_else(Weight::empty);

            if w_u_overlap.is_empty() && w_v_overlap.is_empty() {
                continue;
            }

            if w_u_overlap != w_v_overlap {
                return false;
            }

            // On overlap, targets must match
            match (mapped_u, mapped_v) {
                (Some(mu), Some(mv)) if mu != mv => return false,
                (Some(_), None) | (None, Some(_)) => {
                    if !w_u_overlap.is_empty() {
                        return false;
                    }
                }
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
) -> Vec<Vec<usize>> {
    let nc = candidates.len();
    let mut adj: Vec<Vec<usize>> = vec![vec![]; nc];

    for i in 0..nc {
        for j in (i + 1)..nc {
            if !are_compatible(candidates[i], candidates[j], dwa, needed, old_to_new) {
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
) -> Option<HashSet<(usize, usize)>> {
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

    let mut overlap_pairs = HashSet::new();
    let mut active: Vec<(u32, usize, Arc<range_set_blaze::RangeSetBlaze<u32>>)> = Vec::new();

    for (start, end, idx, tokens) in segments {
        active.retain(|(active_end, _, _)| *active_end >= start);

        for (_, active_idx, active_tokens) in &active {
            if *active_idx == idx {
                continue;
            }
            if Arc::ptr_eq(active_tokens, &tokens) || !active_tokens.is_disjoint(tokens.as_ref()) {
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
) -> Option<Vec<Vec<usize>>> {
    let overlap_pairs = overlapping_candidate_pairs(candidates, needed)?;
    let nc = candidates.len();
    let mut adj: Vec<Vec<usize>> = vec![vec![]; nc];

    for (i, j) in overlap_pairs {
        if !are_compatible(candidates[i], candidates[j], dwa, needed, old_to_new) {
            adj[i].push(j);
            adj[j].push(i);
        }
    }

    Some(adj)
}

/// Greedy graph coloring — O(V + E), good heuristic.
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

fn try_all_compatible_height_0_coloring(
    candidates: &[usize],
    dwa: &DWA,
    needed: &[Weight],
) -> Option<Vec<usize>> {
    if candidates.len() <= 100 {
        return None;
    }
    if !candidates
        .iter()
        .all(|&id| dwa.states[id].transitions.is_empty())
    {
        return None;
    }

    let mut witness_domain = Weight::empty();
    let mut witness_final = Weight::empty();

    for &candidate in candidates {
        let needed_at_candidate = &needed[candidate];
        let final_on_needed = dwa.states[candidate]
            .final_weight
            .as_ref()
            .map(|weight| weight.intersection(needed_at_candidate))
            .unwrap_or_else(Weight::empty);

        if !witness_domain.is_empty() {
            let overlap = witness_domain.intersection(needed_at_candidate);
            if !overlap.is_empty() {
                let witness_on_overlap = witness_final.intersection(&overlap);
                let candidate_on_overlap = final_on_needed.intersection(&overlap);
                if witness_on_overlap != candidate_on_overlap {
                    return None;
                }
            }
        }

        witness_final = witness_final.union(&final_on_needed);
        witness_domain = witness_domain.union(needed_at_candidate);
    }

    Some(vec![0; candidates.len()])
}

// ---------------------------------------------------------------------------
// Phase 4: Merge + Reconstruct
// ---------------------------------------------------------------------------

struct MergedStateBuilder {
    final_weight: Weight,
    final_weight_builder: WeightBuilder,
    needed: Weight,
    needed_builder: WeightBuilder,
    transitions: BTreeMap<Label, (u32, WeightBuilder)>,
}

impl Default for MergedStateBuilder {
    fn default() -> Self {
        Self {
            final_weight: Weight::empty(),
            final_weight_builder: WeightBuilder::new(),
            needed: Weight::empty(),
            needed_builder: WeightBuilder::new(),
            transitions: BTreeMap::new(),
        }
    }
}

impl MergedStateBuilder {
    fn add_final_weight(&mut self, weight: &Weight) {
        self.final_weight_builder.union_weight(weight);
    }

    fn add_needed(&mut self, weight: &Weight) {
        self.needed_builder.union_weight(weight);
    }

    fn add_transition(&mut self, label: Label, target: u32, weight: Weight) {
        match self.transitions.entry(label) {
            std::collections::btree_map::Entry::Vacant(e) => {
                let mut weight_builder = WeightBuilder::new();
                weight_builder.union_weight(&weight);
                e.insert((target, weight_builder));
            }
            std::collections::btree_map::Entry::Occupied(mut e) => {
                let (_existing_target, existing_weight) = e.get_mut();
                existing_weight.union_weight(&weight);
            }
        }
    }

    fn finalize_for_reuse(&mut self) {
        self.final_weight = std::mem::take(&mut self.final_weight_builder).build();
        self.needed = std::mem::take(&mut self.needed_builder).build();
    }
}

fn merge_state_into_builder(
    old_id: usize,
    color: usize,
    dwa: &DWA,
    needed: &[Weight],
    old_to_new: &[u32],
    completed: &[MergedStateBuilder],
    builders: &mut [MergedStateBuilder],
) {
    let builder = &mut builders[color];
    let old_state = &dwa.states[old_id];

    // Union final weights
    if let Some(fw) = &old_state.final_weight {
        builder.add_final_weight(fw);
    }

    // Union needed sets
    builder.add_needed(&needed[old_id]);

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
        let needed_at_target = &completed[target_new as usize].needed;
        let w_effective = if needed_at_target.is_full() {
            w_orig.clone()
        } else if let Some((start, end, tokens)) = w_orig.single_compact_entry_parts() {
            needed_at_target.intersect_single_parts(start, end, &tokens)
        } else {
            w_orig.intersection(needed_at_target)
        };
        if !w_effective.is_empty() {
            builder.add_transition(label, target_new, w_effective);
        }
    }
}

fn reconstruct_dwa(start_old: usize, old_to_new: &[u32], builders: Vec<MergedStateBuilder>) -> DWA {
    let states: Vec<DWAState> = builders
        .into_iter()
        .map(|b| {
            let mut state = DWAState::default();
            if !b.final_weight.is_empty() {
                state.final_weight = Some(b.final_weight);
            }
            for (lbl, (target, weight_builder)) in b.transitions {
                let weight = weight_builder.build();
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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Minimize an acyclic DWA using weight pushing + graph-coloring.
///
/// Falls back to the caller's DWA unchanged if the input is cyclic.
pub fn minimize_acyclic(dwa: &DWA) -> DWA {
    if dwa.states.is_empty() {
        return dwa.clone();
    }

    // Clone and push weights
    let mut pushed = dwa.clone();
    push_weights(&mut pushed);

    let Some(topo) = compute_topo_order(&pushed) else {
        return dwa.clone(); // cyclic — fall back
    };

    let needed = compute_needed_sets(&pushed, &topo);
    let heights = compute_heights(&pushed, &topo);
    let max_height = heights.iter().max().copied().unwrap_or(0);

    let n = pushed.states.len();
    let start_state = pushed.start_state as usize;

    // Compute forward reachability from start
    let mut reachable_from_start = vec![false; n];
    {
        let mut stack = vec![start_state];
        while let Some(u) = stack.pop() {
            if reachable_from_start[u] {
                continue;
            }
            reachable_from_start[u] = true;
            for (_, (target, _)) in &pushed.states[u].transitions {
                let t = *target as usize;
                if t < n && !reachable_from_start[t] {
                    stack.push(t);
                }
            }
        }
    }

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

        if let Some(coloring) = try_all_compatible_height_0_coloring(candidates, &pushed, &needed) {
            let base_new_id = new_states.len() as u32;
            let num_colors = 1usize;

            for &candidate in candidates {
                old_to_new[candidate] = base_new_id;
            }

            new_states.extend((0..num_colors).map(|_| MergedStateBuilder::default()));

            let (completed, builders) = new_states.split_at_mut(base_new_id as usize);
            for &candidate in candidates {
                merge_state_into_builder(
                    candidate,
                    0,
                    &pushed,
                    &needed,
                    &old_to_new,
                    completed,
                    builders,
                );
            }
            for builder in builders.iter_mut() {
                builder.finalize_for_reuse();
            }
            continue;
        }

        // Build incompatibility graph and color it
        let adj = build_incompatibility_graph_sparse(&pushed, candidates, &needed, &old_to_new)
            .unwrap_or_else(|| build_incompatibility_graph(&pushed, candidates, &needed, &old_to_new));
        let coloring = greedy_coloring(&adj);

        let base_new_id = new_states.len() as u32;
        let num_colors = coloring.iter().max().map(|&c| c + 1).unwrap_or(0);

        // Map old states to new merged states
        for (idx, &color) in coloring.iter().enumerate() {
            old_to_new[candidates[idx]] = base_new_id + color as u32;
        }

        // Extend builders
        new_states.extend((0..num_colors).map(|_| MergedStateBuilder::default()));

        // Merge states into builders
        let (completed, builders) = new_states.split_at_mut(base_new_id as usize);
        for (idx, &color) in coloring.iter().enumerate() {
            merge_state_into_builder(
                candidates[idx],
                color,
                &pushed,
                &needed,
                &old_to_new,
                completed,
                builders,
            );
        }
        for builder in builders.iter_mut() {
            builder.finalize_for_reuse();
        }
    }

    reconstruct_dwa(start_state, &old_to_new, new_states)
}
