//! Exact minimization for acyclic token-deterministic weighted NWAs.
//!
//! Branches carrying one grammar label may have different targets, but their
//! weights are disjoint on every live token. This is deterministic over
//! `(label, token)` and does not require subset construction. The minimizer
//! works directly with that relation, retaining multiple target branches when
//! a quotient cannot safely collapse them.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::{FxHashMap, FxHashSet};

use super::nwa::{Label, NWA};
use crate::ds::weight::Weight;
use crate::GlrMaskError;

const UNMAPPED: u32 = u32::MAX;

fn profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

fn topo_order(nwa: &NWA) -> Option<Vec<usize>> {
    let n = nwa.states().len();
    let mut indegree = vec![0usize; n];
    for state in nwa.states() {
        if !state.epsilons.is_empty() {
            return None;
        }
        for branches in state.transitions.values() {
            for (target, _) in branches {
                let target = *target as usize;
                if target >= n {
                    return None;
                }
                indegree[target] += 1;
            }
        }
    }
    let mut queue: Vec<usize> = indegree
        .iter()
        .enumerate()
        .filter_map(|(state, &degree)| (degree == 0).then_some(state))
        .collect();
    let mut head = 0usize;
    let mut result = Vec::with_capacity(n);
    while head < queue.len() {
        let state_id = queue[head];
        head += 1;
        result.push(state_id);
        for branches in nwa.states()[state_id].transitions.values() {
            for (target, _) in branches {
                let target = *target as usize;
                indegree[target] -= 1;
                if indegree[target] == 0 {
                    queue.push(target);
                }
            }
        }
    }
    (result.len() == n).then_some(result)
}

fn reachable_from_starts(nwa: &NWA) -> Vec<bool> {
    let mut reachable = vec![false; nwa.states().len()];
    let mut stack = nwa.start_states().to_vec();
    while let Some(state_id) = stack.pop() {
        let state_id = state_id as usize;
        if state_id >= reachable.len() || reachable[state_id] {
            continue;
        }
        reachable[state_id] = true;
        for branches in nwa.states()[state_id].transitions.values() {
            stack.extend(branches.iter().map(|(target, _)| *target));
        }
    }
    reachable
}

#[derive(Default)]
struct PushProfile {
    branch_intersections: usize,
    push_noop_hits: usize,
    subset_checks: usize,
    subset_hits: usize,
    subset_ms: f64,
    intersection_cache_hits: usize,
    intersection_cache_misses: usize,
    reachable_parts: usize,
    unique_reachable_parts: usize,
    union_ms: f64,
}

#[inline]
fn weight_id(weight: &Weight) -> usize {
    Arc::as_ptr(&weight.0) as usize
}

fn cached_intersection(
    cache: &mut FxHashMap<(usize, usize), Weight>,
    left: &Weight,
    right: &Weight,
    profile: &mut PushProfile,
) -> Weight {
    profile.branch_intersections += 1;
    if left.is_empty() || right.is_empty() {
        return Weight::empty();
    }
    if left.is_full() {
        return right.clone();
    }
    if right.is_full() || Arc::ptr_eq(&left.0, &right.0) {
        profile.push_noop_hits += 1;
        return left.clone();
    }
    let subset_started_at = Instant::now();
    profile.subset_checks += 1;
    if left.is_subset(right) {
        profile.subset_hits += 1;
        profile.subset_ms += subset_started_at.elapsed().as_secs_f64() * 1000.0;
        return left.clone();
    }
    profile.subset_ms += subset_started_at.elapsed().as_secs_f64() * 1000.0;
    let left_id = weight_id(left);
    let right_id = weight_id(right);
    let key = if left_id <= right_id { (left_id, right_id) } else { (right_id, left_id) };
    if let Some(existing) = cache.get(&key) {
        profile.intersection_cache_hits += 1;
        return existing.clone();
    }
    profile.intersection_cache_misses += 1;
    let result = left.intersection(right);
    cache.insert(key, result.clone());
    result
}

fn union_unique_weights(parts: impl IntoIterator<Item = Weight>, profile: &mut PushProfile) -> Weight {
    let mut unique = FxHashSet::<usize>::default();
    let mut weights = Vec::<Weight>::new();
    for weight in parts {
        if weight.is_empty() {
            continue;
        }
        profile.reachable_parts += 1;
        if unique.insert(weight_id(&weight)) {
            profile.unique_reachable_parts += 1;
            weights.push(weight);
        }
    }
    let started_at = Instant::now();
    let result = Weight::union_all(weights.iter());
    profile.union_ms += started_at.elapsed().as_secs_f64() * 1000.0;
    result
}

fn push_weights(nwa: &mut NWA, topo: &[usize]) -> (Vec<Weight>, PushProfile) {
    let mut reachable = vec![Weight::empty(); nwa.states().len()];
    let mut intersection_cache = FxHashMap::<(usize, usize), Weight>::default();
    let mut profile = PushProfile::default();
    for &state_id in topo.iter().rev() {
        let state = &mut nwa.states_mut()[state_id];
        let mut parts = Vec::with_capacity(
            state.transitions.values().map(Vec::len).sum::<usize>() + usize::from(state.final_weight.is_some()),
        );
        if let Some(final_weight) = &state.final_weight {
            if !final_weight.is_empty() {
                parts.push(final_weight.clone());
            }
        }
        for branches in state.transitions.values_mut() {
            for (target, weight) in branches.iter_mut() {
                let pushed = cached_intersection(
                    &mut intersection_cache,
                    weight,
                    &reachable[*target as usize],
                    &mut profile,
                );
                *weight = pushed;
                if !weight.is_empty() {
                    parts.push(weight.clone());
                }
            }
            branches.retain(|(_, weight)| !weight.is_empty());
        }
        state.transitions.retain(|_, branches| !branches.is_empty());
        reachable[state_id] = union_unique_weights(parts, &mut profile);
    }
    (reachable, profile)
}

fn raw_needed_weights(nwa: &NWA, topo: &[usize]) -> (Vec<Weight>, PushProfile) {
    let mut needed = vec![Weight::empty(); nwa.states().len()];
    let mut profile = PushProfile::default();
    for &state_id in topo.iter().rev() {
        let state = &nwa.states()[state_id];
        let mut parts = Vec::with_capacity(
            state.transitions.values().map(Vec::len).sum::<usize>() + usize::from(state.final_weight.is_some()),
        );
        if let Some(final_weight) = &state.final_weight {
            parts.push(final_weight.clone());
        }
        for branches in state.transitions.values() {
            parts.extend(branches.iter().map(|(_, weight)| weight.clone()));
        }
        needed[state_id] = union_unique_weights(parts, &mut profile);
    }
    (needed, profile)
}

fn heights(nwa: &NWA, topo: &[usize]) -> Vec<usize> {
    let mut heights = vec![0usize; nwa.states().len()];
    for &state_id in topo.iter().rev() {
        heights[state_id] = nwa.states()[state_id]
            .transitions
            .values()
            .flatten()
            .map(|(target, _)| heights[*target as usize] + 1)
            .max()
            .unwrap_or(0);
    }
    heights
}

type BranchProfile = BTreeMap<(Label, u32), Weight>;

fn branch_profile(nwa: &NWA, state_id: usize, old_to_new: &[u32]) -> BranchProfile {
    let mut profile = BTreeMap::new();
    for (&label, branches) in &nwa.states()[state_id].transitions {
        for (target, weight) in branches {
            let mapped_target = old_to_new[*target as usize];
            if mapped_target == UNMAPPED || weight.is_empty() {
                continue;
            }
            profile
                .entry((label, mapped_target))
                .and_modify(|existing: &mut Weight| *existing = existing.union(weight))
                .or_insert_with(|| weight.clone());
        }
    }
    profile
}

fn final_weight_matches(
    left: Option<&Weight>,
    right: Option<&Weight>,
    domain: &Weight,
) -> bool {
    let left = left.map(|weight| weight.intersection(domain)).unwrap_or_else(Weight::empty);
    let right = right.map(|weight| weight.intersection(domain)).unwrap_or_else(Weight::empty);
    left == right
}

fn profiles_match_on_domain(left: &BranchProfile, right: &BranchProfile, domain: &Weight) -> bool {
    let mut left_iter = left.iter().peekable();
    let mut right_iter = right.iter().peekable();
    loop {
        match (left_iter.peek(), right_iter.peek()) {
            (None, None) => return true,
            (Some((left_key, left_weight)), Some((right_key, right_weight))) => {
                if left_key == right_key {
                    if left_weight.intersection(domain) != right_weight.intersection(domain) {
                        return false;
                    }
                    left_iter.next();
                    right_iter.next();
                } else if left_key < right_key {
                    if !left_weight.is_disjoint(domain) {
                        return false;
                    }
                    left_iter.next();
                } else {
                    if !right_weight.is_disjoint(domain) {
                        return false;
                    }
                    right_iter.next();
                }
            }
            (Some((_, left_weight)), None) => {
                if !left_weight.is_disjoint(domain) {
                    return false;
                }
                left_iter.next();
            }
            (None, Some((_, right_weight))) => {
                if !right_weight.is_disjoint(domain) {
                    return false;
                }
                right_iter.next();
            }
        }
    }
}

fn states_compatible(
    nwa: &NWA,
    left: usize,
    right: usize,
    needed: &[Weight],
    profiles: &[BranchProfile],
) -> bool {
    let domain = needed[left].intersection(&needed[right]);
    domain.is_empty()
        || (final_weight_matches(
            nwa.states()[left].final_weight.as_ref(),
            nwa.states()[right].final_weight.as_ref(),
            &domain,
        ) && profiles_match_on_domain(&profiles[left], &profiles[right], &domain))
}

/// Minimize an acyclic NWA that is deterministic over `(label, token)`.
///
/// The output retains that property. It intentionally does not introduce a
/// product state just to make the next target independent of the token.
pub fn minimize_token_deterministic_nwa_owned(mut nwa: NWA) -> Result<NWA, GlrMaskError> {
    let input_states = nwa.states().len();
    let input_transitions = nwa.num_transitions();
    let total_started_at = Instant::now();
    let Some(topo) = topo_order(&nwa) else {
        return Err(GlrMaskError::Compilation(
            "token-deterministic NWA minimization requires an acyclic epsilon-free automaton".into(),
        ));
    };
    let reachable = reachable_from_starts(&nwa);
    let skip_push = std::env::var_os("GLRMASK_EXPERIMENTAL_ASSUME_TOKEN_NWA_TRIMMED").is_some();
    let push_started_at = Instant::now();
    let (needed, push_profile) = if skip_push {
        raw_needed_weights(&nwa, &topo)
    } else {
        push_weights(&mut nwa, &topo)
    };
    let push_ms = push_started_at.elapsed().as_secs_f64() * 1000.0;
    let heights = heights(&nwa, &topo);
    let max_height = heights.iter().copied().max().unwrap_or(0);
    let mut by_height = vec![Vec::<usize>::new(); max_height + 1];
    for (state_id, &height) in heights.iter().enumerate() {
        if reachable[state_id] && !needed[state_id].is_empty() {
            by_height[height].push(state_id);
        }
    }

    let mut old_to_new = vec![UNMAPPED; nwa.states().len()];
    let mut output = NWA::new(0, 0);
    let mut color_ms_total = 0.0;

    for candidates in by_height {
        if candidates.is_empty() {
            continue;
        }
        let profile_started_at = Instant::now();
        let mut profiles = vec![BTreeMap::new(); nwa.states().len()];
        for &state_id in &candidates {
            profiles[state_id] = branch_profile(&nwa, state_id, &old_to_new);
        }
        let profile_ms = profile_started_at.elapsed().as_secs_f64() * 1000.0;

        let color_started_at = Instant::now();
        let mut groups = Vec::<Vec<usize>>::new();
        for &candidate in &candidates {
            let mut placed = false;
            for group in &mut groups {
                if group.iter().all(|&member| {
                    states_compatible(&nwa, candidate, member, &needed, &profiles)
                }) {
                    group.push(candidate);
                    placed = true;
                    break;
                }
            }
            if !placed {
                groups.push(vec![candidate]);
            }
        }
        let color_ms = color_started_at.elapsed().as_secs_f64() * 1000.0;
        color_ms_total += profile_ms + color_ms;

        for group in groups {
            let new_state = output.add_state();
            for &old_state in &group {
                old_to_new[old_state] = new_state;
            }
        }

        // All outgoing targets have lower height and therefore are already
        // mapped. Aggregate compatible source branches by `(label, target)`.
        // Recover groups by scanning the old-to-new relation once per state.
        // the old-to-new relation once per state in this height bucket.
        let mut members_by_new = BTreeMap::<u32, Vec<usize>>::new();
        for &old_state in &candidates {
            members_by_new.entry(old_to_new[old_state]).or_default().push(old_state);
        }
        for (new_state, members) in members_by_new {
            let mut final_weight = Weight::empty();
            let mut branches = BTreeMap::<(Label, u32), Weight>::new();
            for old_state in members {
                if let Some(weight) = &nwa.states()[old_state].final_weight {
                    final_weight = final_weight.union(weight);
                }
                for (&label, source_branches) in &nwa.states()[old_state].transitions {
                    for (target, weight) in source_branches {
                        let target = old_to_new[*target as usize];
                        debug_assert_ne!(target, UNMAPPED);
                        branches
                            .entry((label, target))
                            .and_modify(|existing| *existing = existing.union(weight))
                            .or_insert_with(|| weight.clone());
                    }
                }
            }
            if !final_weight.is_empty() {
                output.set_final_weight(new_state, final_weight);
            }
            for ((label, target), weight) in branches {
                if !weight.is_empty() {
                    output.add_transition(new_state, label, target, weight);
                }
            }
        }
    }

    let starts: Vec<u32> = nwa
        .start_states()
        .iter()
        .filter_map(|start| {
            let mapped = old_to_new.get(*start as usize).copied().unwrap_or(UNMAPPED);
            (mapped != UNMAPPED).then_some(mapped)
        })
        .collect();
    output.set_start_states(starts);

    if profile_enabled() {
        eprintln!(
            "[glrmask/profile][token_deterministic_nwa_minimize] input_states={} input_transitions={} push_skipped={} push_ms={:.3} push_branch_intersections={} push_noop_hits={} push_subset_checks={} push_subset_hits={} push_subset_ms={:.3} push_intersection_cache_hits={} push_intersection_cache_misses={} push_reachable_parts={} push_unique_reachable_parts={} push_union_ms={:.3} color_ms={:.3} output_states={} output_transitions={} total_ms={:.3}",
            input_states,
            input_transitions,
            skip_push,
            push_ms,
            push_profile.branch_intersections,
            push_profile.push_noop_hits,
            push_profile.subset_checks,
            push_profile.subset_hits,
            push_profile.subset_ms,
            push_profile.intersection_cache_hits,
            push_profile.intersection_cache_misses,
            push_profile.reachable_parts,
            push_profile.unique_reachable_parts,
            push_profile.union_ms,
            color_ms_total,
            output.num_states(),
            output.num_transitions(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use range_set_blaze::RangeSetBlaze;

    use super::*;
    use crate::automata::weighted::determinize::determinize;
    use crate::automata::weighted::equivalence::find_difference;

    fn tokens(values: &[u32]) -> Weight {
        Weight::from_per_tsid_token_sets(std::iter::once((
            0,
            RangeSetBlaze::from_iter(values.iter().copied().map(|value| value..=value)),
        )))
    }

    #[test]
    fn merges_equal_token_deterministic_leaves_exactly() {
        let mut nwa = NWA::new(0, 0);
        let start = nwa.add_state();
        let left = nwa.add_state();
        let right = nwa.add_state();
        nwa.set_start_states(vec![start]);
        nwa.add_transition(start, 7, left, tokens(&[0]));
        nwa.add_transition(start, 7, right, tokens(&[1]));
        nwa.set_final_weight(left, tokens(&[0]));
        nwa.set_final_weight(right, tokens(&[1]));

        let original = determinize(&nwa).unwrap();
        let minimized = minimize_token_deterministic_nwa_owned(nwa).unwrap();
        let minimized_determinized = determinize(&minimized).unwrap();

        assert_eq!(minimized.num_states(), 2);
        assert_eq!(find_difference(&original, &minimized_determinized).unwrap(), None);
    }
}

/// Exact quotient for a token-deterministic NWA whose state weights are
/// confined to pairwise-disjoint source token domains.
///
/// `state_sources[state]` is `Some(source)` for a source-local state and
/// `None` for a mixed state (normally the combined start). States from
/// distinct sources can share one quotient state at the same height because
/// no token can reach both alternatives. A group never contains two states
/// from the same source, so no unproven within-source equivalence is assumed.
pub fn quotient_disjoint_source_nwa_owned(
    nwa: NWA,
    state_sources: &[Option<usize>],
) -> Result<NWA, GlrMaskError> {
    if state_sources.len() != nwa.states().len() {
        return Err(GlrMaskError::Compilation(
            "token-deterministic NWA source metadata length mismatch".into(),
        ));
    }
    let input_states = nwa.states().len();
    let input_transitions = nwa.num_transitions();
    let total_started_at = Instant::now();
    let Some(topo) = topo_order(&nwa) else {
        return Err(GlrMaskError::Compilation(
            "source-domain token NWA quotient requires an acyclic epsilon-free automaton".into(),
        ));
    };
    let reachable = reachable_from_starts(&nwa);
    let heights = heights(&nwa, &topo);
    let max_height = heights.iter().copied().max().unwrap_or(0);
    let mut by_height = vec![Vec::<usize>::new(); max_height + 1];
    for (state_id, &height) in heights.iter().enumerate() {
        if reachable[state_id] {
            by_height[height].push(state_id);
        }
    }

    let mut old_to_new = vec![UNMAPPED; nwa.states().len()];
    let mut output = NWA::new(0, 0);
    let mut group_count = 0usize;
    let mut merged_state_count = 0usize;

    for candidates in by_height {
        if candidates.is_empty() {
            continue;
        }
        let mut groups = Vec::<(Vec<usize>, FxHashSet<usize>, bool)>::new();
        for candidate in candidates {
            match state_sources[candidate] {
                None => {
                    // A mixed-domain state cannot be merged merely from source
                    // disjointness. Keep it isolated unless a later general
                    // quotient proves more.
                    groups.push((vec![candidate], FxHashSet::default(), true));
                }
                Some(source) => {
                    let mut placed = false;
                    for (members, sources, has_mixed) in &mut groups {
                        if !*has_mixed && !sources.contains(&source) {
                            sources.insert(source);
                            members.push(candidate);
                            placed = true;
                            break;
                        }
                    }
                    if !placed {
                        let mut sources = FxHashSet::default();
                        sources.insert(source);
                        groups.push((vec![candidate], sources, false));
                    }
                }
            }
        }

        for (members, _, _) in &groups {
            let new_state = output.add_state();
            for &old_state in members {
                old_to_new[old_state] = new_state;
            }
            merged_state_count += members.len();
        }

        for (members, _, _) in groups {
            let new_state = old_to_new[members[0]];
            let mut final_weight = Weight::empty();
            let mut branches = BTreeMap::<(Label, u32), Weight>::new();
            for old_state in members {
                if let Some(weight) = &nwa.states()[old_state].final_weight {
                    final_weight = final_weight.union(weight);
                }
                for (&label, source_branches) in &nwa.states()[old_state].transitions {
                    for (target, weight) in source_branches {
                        let target = old_to_new[*target as usize];
                        debug_assert_ne!(target, UNMAPPED);
                        branches
                            .entry((label, target))
                            .and_modify(|existing| *existing = existing.union(weight))
                            .or_insert_with(|| weight.clone());
                    }
                }
            }
            if !final_weight.is_empty() {
                output.set_final_weight(new_state, final_weight);
            }
            for ((label, target), weight) in branches {
                if !weight.is_empty() {
                    output.add_transition(new_state, label, target, weight);
                }
            }
            group_count += 1;
        }
    }

    let starts: Vec<u32> = nwa
        .start_states()
        .iter()
        .filter_map(|start| {
            let mapped = old_to_new.get(*start as usize).copied().unwrap_or(UNMAPPED);
            (mapped != UNMAPPED).then_some(mapped)
        })
        .collect();
    output.set_start_states(starts);

    if profile_enabled() {
        eprintln!(
            "[glrmask/profile][token_deterministic_nwa_source_quotient] input_states={} input_transitions={} groups={} merged_states={} output_states={} output_transitions={} total_ms={:.3}",
            input_states,
            input_transitions,
            group_count,
            merged_state_count,
            output.num_states(),
            output.num_transitions(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(output)
}
