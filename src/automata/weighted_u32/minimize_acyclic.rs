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

use std::collections::HashSet;
use std::sync::Arc;

use super::dwa::{DWA, DWAState};
use crate::ds::weight::{Weight, WeightBuilder};

type Label = i32;

const UNMAPPED: u32 = u32::MAX;

#[derive(Default)]
struct MinimizeAcyclicProfile {
    height_buckets: usize,
    total_candidates: usize,
    max_candidates: usize,
    singleton_buckets: usize,
    all_compatible_buckets: usize,
    dense_graph_buckets: usize,
    sparse_graph_buckets: usize,
    dense_pair_candidates: usize,
    sparse_overlap_pairs: usize,
    compatibility_checks: usize,
    push_weights_ms: std::time::Duration,
    topo_needed_ms: std::time::Duration,
    graph_color_ms: std::time::Duration,
    merge_rebuild_ms: std::time::Duration,
    merge_state_calls: usize,
    merge_final_needed_ms: std::time::Duration,
    merge_transition_loop_ms: std::time::Duration,
    builder_finalize_ms: std::time::Duration,
    reconstruct_ms: std::time::Duration,
}

fn minimize_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_MINIMIZE_ACYCLIC").is_some()
}

// ---------------------------------------------------------------------------
// Phase 0: Weight pushing (backward reachability pruning)
// ---------------------------------------------------------------------------

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

    // 2. Backward reachable sets (reverse topo order = leaves first)
    let mut reachable: Vec<Weight> = vec![Weight::empty(); n];
    for &u in topo.iter().rev() {
        let st = &dwa.states[u];
        let mut acc = st.final_weight.as_ref().cloned().unwrap_or_else(Weight::empty);
        for (_, (target, w)) in &st.transitions {
            let t = *target as usize;
            if t >= n {
                continue;
            }
            if reachable[t].is_full() {
                acc = acc.union(w);
            } else if !reachable[t].is_empty() {
                let tmp = w.intersection(&reachable[t]);
                acc = acc.union(&tmp);
            }
            if acc.is_full() {
                break;
            }
        }
        reachable[u] = acc;
    }

    // 3. Intersect each transition weight with reachable[target]
    let mut changed = false;
    for u in 0..n {
        // Two-pass approach: first read-only pass collects changes, second pass applies them.
        // This avoids cloning all weights upfront.
        let changes: Vec<(Label, u32, Option<Weight>)> = dwa.states[u]
            .transitions
            .iter()
            .filter_map(|(&lbl, &(target, ref w))| {
                let t = target as usize;
                if t >= n || reachable[t].is_full() {
                    return None;
                }
                let new_w = if reachable[t].is_empty() {
                    Weight::empty()
                } else {
                    w.intersection(&reachable[t])
                };
                if new_w != *w {
                    Some((lbl, target, if new_w.is_empty() { None } else { Some(new_w) }))
                } else {
                    None
                }
            })
            .collect();

        for (lbl, target, new_w_opt) in changes {
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
        let mut acc = st.final_weight.as_ref().cloned().unwrap_or_else(Weight::empty);
        for (_, (target, w)) in &st.transitions {
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
            let productive = if needed[t].is_full() {
                weight.clone()
            } else if let Some((start, end, tokens)) = weight.single_compact_entry_parts() {
                needed[t].intersect_single_parts(start, end, &tokens)
            } else {
                weight.intersection(&needed[t])
            };
            if productive.is_empty() {
                continue;
            }
            transitions.push(ProductiveTransition {
                label,
                target: *target,
                weight: productive,
            });
        }
        result.push(transitions);
    }

    result
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
    productive_transitions: &[Vec<ProductiveTransition>],
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
    let trans_u = &productive_transitions[u];
    let trans_v = &productive_transitions[v];
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
            (Some(u_entry), None) => {
                idx_u += 1;
                (Some(u_entry), None)
            }
            (None, Some(v_entry)) => {
                idx_v += 1;
                (None, Some(v_entry))
            }
            (None, None) => break,
        };

        let (target_u, w_u_full) = match entry_u {
            Some(entry) => {
                let t = entry.target as usize;
                if t < n { (Some(t), Some(&entry.weight)) } else { (None, None) }
            }
            None => (None, None),
        };
        let (target_v, w_v_full) = match entry_v {
            Some(entry) => {
                let t = entry.target as usize;
                if t < n { (Some(t), Some(&entry.weight)) } else { (None, None) }
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
            // Fast path: if both full weights are equal, their overlap restrictions
            // are also equal. Skip expensive intersection computation.
            let weights_equal = match (w_u_full, w_v_full) {
                (Some(wu), Some(wv)) => wu == wv,
                (None, None) => true,
                _ => false,
            };

            if weights_equal {
                // Same weight → same intersection → compatible on this label.
                // Targets already verified above (mapped_u == mapped_v or both empty weight).
                continue;
            }

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
    productive_transitions: &[Vec<ProductiveTransition>],
    profile: &mut MinimizeAcyclicProfile,
) -> Vec<Vec<usize>> {
    let nc = candidates.len();
    let mut adj: Vec<Vec<usize>> = vec![vec![]; nc];
    profile.dense_graph_buckets += 1;
    profile.dense_pair_candidates += nc.saturating_sub(1) * nc / 2;

    for i in 0..nc {
        for j in (i + 1)..nc {
            profile.compatibility_checks += 1;
            if !are_compatible(
                candidates[i],
                candidates[j],
                dwa,
                needed,
                old_to_new,
                productive_transitions,
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
    productive_transitions: &[Vec<ProductiveTransition>],
    profile: &mut MinimizeAcyclicProfile,
) -> Option<Vec<Vec<usize>>> {
    let overlap_pairs = overlapping_candidate_pairs(candidates, needed)?;
    let nc = candidates.len();
    let mut adj: Vec<Vec<usize>> = vec![vec![]; nc];
    profile.sparse_graph_buckets += 1;
    profile.sparse_overlap_pairs += overlap_pairs.len();
    profile.compatibility_checks += overlap_pairs.len();

    for (i, j) in overlap_pairs {
        if !are_compatible(
            candidates[i],
            candidates[j],
            dwa,
            needed,
            old_to_new,
            productive_transitions,
        ) {
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

/// Partition refinement coloring: group candidates by (final_weight, transitions).
/// Two candidates get the same color iff they have identical final weights and
/// identical productive transitions (same labels, same mapped targets, same weights).
///
/// Phase 2: Merge colors whose needed-set unions are disjoint and whose target
/// profiles don't conflict on any shared label. This recovers the "diamond"
/// optimization that exploits disjoint token domains.
fn partition_refine_coloring(
    candidates: &[usize],
    dwa: &DWA,
    needed: &[Weight],
    old_to_new: &[u32],
    productive_transitions: &[Vec<ProductiveTransition>],
) -> Vec<usize> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let nc = candidates.len();

    // --- Phase 1: Hash-based partition by exact signature ---
    let mut hash_groups: rustc_hash::FxHashMap<u64, Vec<usize>> =
        rustc_hash::FxHashMap::default();

    for idx in 0..nc {
        let c = candidates[idx];
        let mut hasher = DefaultHasher::new();
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

    // --- Phase 2: Merge colors with disjoint needed sets + compatible targets ---
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
        if old_to_new[tu.target as usize] != old_to_new[tv.target as usize] {
            return false;
        }
        if tu.weight != tv.weight {
            return false;
        }
    }
    true
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
    transitions: rustc_hash::FxHashMap<Label, (u32, WeightBuilder)>,
}

impl Default for MergedStateBuilder {
    fn default() -> Self {
        Self {
            final_weight: Weight::empty(),
            final_weight_builder: WeightBuilder::new(),
            needed: Weight::empty(),
            needed_builder: WeightBuilder::new(),
            transitions: rustc_hash::FxHashMap::default(),
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
        use std::collections::hash_map::Entry;
        match self.transitions.entry(label) {
            Entry::Occupied(mut entry) => {
                let (existing_target, existing_weight) = entry.get_mut();
                debug_assert_eq!(*existing_target, target);
                existing_weight.union_weight(&weight);
            }
            Entry::Vacant(entry) => {
                let mut weight_builder = WeightBuilder::new();
                weight_builder.union_weight(&weight);
                entry.insert((target, weight_builder));
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
    profile: &mut MinimizeAcyclicProfile,
) {
    profile.merge_state_calls += 1;
    let phase_started_at = std::time::Instant::now();
    let builder = &mut builders[color];
    let old_state = &dwa.states[old_id];

    // Union final weights
    if let Some(fw) = &old_state.final_weight {
        builder.add_final_weight(fw);
    }

    // Union needed sets
    builder.add_needed(&needed[old_id]);
    profile.merge_final_needed_ms += phase_started_at.elapsed();

    // Merge transitions
    let phase_started_at = std::time::Instant::now();
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
    profile.merge_transition_loop_ms += phase_started_at.elapsed();
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
    let profile_enabled = minimize_profile_enabled();
    let started_at = std::time::Instant::now();
    let mut profile = MinimizeAcyclicProfile::default();

    // Clone and push weights
    let phase_started_at = std::time::Instant::now();
    let mut pushed = dwa.clone();
    let (_, topo_from_push, _reachable_pre_push) = push_weights(&mut pushed);
    profile.push_weights_ms = phase_started_at.elapsed();

    let phase_started_at = std::time::Instant::now();
    // Reuse topo order from push_weights (graph structure unchanged by push).
    let topo = match topo_from_push {
        Some(t) => t,
        None => return dwa.clone(), // cyclic — fall back
    };

    // Recompute needed sets on the pushed DWA (push may have changed weights).
    let needed = compute_needed_sets(&pushed, &topo);
    let productive_transitions = compute_productive_transitions(&pushed, &needed);
    let heights = compute_heights(&pushed, &topo);
    let max_height = heights.iter().max().copied().unwrap_or(0);
    profile.topo_needed_ms = phase_started_at.elapsed();

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
        profile.height_buckets += 1;
        profile.total_candidates += candidates.len();
        profile.max_candidates = profile.max_candidates.max(candidates.len());
        if candidates.len() == 1 {
            profile.singleton_buckets += 1;
        }

        let phase_started_at = std::time::Instant::now();
        if h == 0 && let Some(coloring) = try_all_compatible_height_0_coloring(candidates, &pushed, &needed) {
            profile.all_compatible_buckets += 1;
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
                    &mut profile,
                );
            }
            let finalize_started_at = std::time::Instant::now();
            for builder in builders.iter_mut() {
                builder.finalize_for_reuse();
            }
            profile.builder_finalize_ms += finalize_started_at.elapsed();
            profile.merge_rebuild_ms += phase_started_at.elapsed();
            continue;
        }

        // Build incompatibility graph and color it.
        // For large buckets, try sparse overlap filter first.
        let graph_started_at = std::time::Instant::now();
        let adj = build_incompatibility_graph_sparse(
            &pushed,
            candidates,
            &needed,
            &old_to_new,
            &productive_transitions,
            &mut profile,
        )
        .unwrap_or_else(|| {
            build_incompatibility_graph(
                &pushed,
                candidates,
                &needed,
                &old_to_new,
                &productive_transitions,
                &mut profile,
            )
        });
        let coloring = greedy_coloring(&adj);
        profile.graph_color_ms += graph_started_at.elapsed();

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
                &mut profile,
            );
        }
        let finalize_started_at = std::time::Instant::now();
        for builder in builders.iter_mut() {
            builder.finalize_for_reuse();
        }
        profile.builder_finalize_ms += finalize_started_at.elapsed();
        profile.merge_rebuild_ms += phase_started_at.elapsed();
    }

    let reconstruct_started_at = std::time::Instant::now();
    let minimized = reconstruct_dwa(start_state, &old_to_new, new_states);
    profile.reconstruct_ms = reconstruct_started_at.elapsed();
    profile.merge_rebuild_ms += reconstruct_started_at.elapsed();
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][minimize_acyclic] total_ms={:.3} input_states={} output_states={} height_buckets={} total_candidates={} max_candidates={} singleton_buckets={} all_compatible_buckets={} dense_graph_buckets={} sparse_graph_buckets={} dense_pair_candidates={} sparse_overlap_pairs={} compatibility_checks={} push_weights_ms={:.3} topo_needed_ms={:.3} graph_color_ms={:.3} merge_rebuild_ms={:.3} merge_state_calls={} merge_final_needed_ms={:.3} merge_transition_loop_ms={:.3} builder_finalize_ms={:.3} reconstruct_ms={:.3}",
            started_at.elapsed().as_secs_f64() * 1000.0,
            dwa.states.len(),
            minimized.states.len(),
            profile.height_buckets,
            profile.total_candidates,
            profile.max_candidates,
            profile.singleton_buckets,
            profile.all_compatible_buckets,
            profile.dense_graph_buckets,
            profile.sparse_graph_buckets,
            profile.dense_pair_candidates,
            profile.sparse_overlap_pairs,
            profile.compatibility_checks,
            profile.push_weights_ms.as_secs_f64() * 1000.0,
            profile.topo_needed_ms.as_secs_f64() * 1000.0,
            profile.graph_color_ms.as_secs_f64() * 1000.0,
            profile.merge_rebuild_ms.as_secs_f64() * 1000.0,
            profile.merge_state_calls,
            profile.merge_final_needed_ms.as_secs_f64() * 1000.0,
            profile.merge_transition_loop_ms.as_secs_f64() * 1000.0,
            profile.builder_finalize_ms.as_secs_f64() * 1000.0,
            profile.reconstruct_ms.as_secs_f64() * 1000.0,
        );
    }
    minimized
}
