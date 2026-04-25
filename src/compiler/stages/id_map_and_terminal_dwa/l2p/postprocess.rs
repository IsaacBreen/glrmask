//! NWA postprocessing: canonicalize, prune, collapse, disallowed-follow subtraction.
//!
//! These functions operate on a fully-built NWA (after trie walk) and prepare
//! it for determinization.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

use crate::automata::weighted::nwa::{NWA, NWAState as NWAStateType};
use crate::grammar::flat::TerminalID;
use super::equivalence_analysis::disallowed_follows::{
    build_disallowed_follow_dfa, normalize_disallowed_follows,
};
use crate::ds::bitset::BitSet;
use crate::ds::weight::Weight;

// ─── Canonicalize ────────────────────────────────────────────────────────────

fn structural_hash_nwa_state(state: &NWAStateType) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    state.final_weight.is_some().hash(&mut hasher);
    if let Some(w) = &state.final_weight {
        hash_weight(w, &mut hasher);
    }

    state.transitions.len().hash(&mut hasher);
    for (label, targets) in &state.transitions {
        label.hash(&mut hasher);
        targets.len().hash(&mut hasher);
        for (target, weight) in targets {
            target.hash(&mut hasher);
            hash_weight(weight, &mut hasher);
        }
    }

    state.epsilons.len().hash(&mut hasher);
    for (target, weight) in &state.epsilons {
        target.hash(&mut hasher);
        hash_weight(weight, &mut hasher);
    }

    hasher.finish()
}

fn hash_weight(weight: &Weight, hasher: &mut impl Hasher) {
    if weight.is_full() {
        0xFFFF_FFFFu32.hash(hasher);
        return;
    }
    for (range, tokens) in weight.0.range_values() {
        range.start().hash(hasher);
        range.end().hash(hasher);
        for r in tokens.ranges() {
            r.start().hash(hasher);
            r.end().hash(hasher);
        }
    }
}

pub(crate) fn canonicalize_acyclic_nwa(nwa: &mut NWA) {
    if nwa.states().len() <= 1 {
        return;
    }

    prune_unreachable_states(nwa);
    let topo_order = topological_order(nwa);
    if topo_order.len() != nwa.states().len() {
        return;
    }

    let old_states = nwa.states().len();
    let mut remap = vec![u32::MAX; old_states];
    let mut canonical_states: Vec<NWAStateType> = Vec::with_capacity(old_states);
    let mut hash_buckets: HashMap<u64, Vec<u32>> = HashMap::new();
    let mut merged = 0usize;

    for old_state_id in topo_order.into_iter().rev() {
        let old_state = &nwa.states()[old_state_id];

        let mut epsilons: BTreeMap<u32, Weight> = BTreeMap::new();
        for (target, weight) in &old_state.epsilons {
            let canonical_target = remap[*target as usize];
            epsilons
                .entry(canonical_target)
                .and_modify(|existing| *existing = existing.union(weight))
                .or_insert_with(|| weight.clone());
        }

        let mut transitions = BTreeMap::new();
        for (&label, targets) in &old_state.transitions {
            let mut canonical_targets: BTreeMap<u32, Weight> = BTreeMap::new();
            for (target, weight) in targets {
                let canonical_target = remap[*target as usize];
                canonical_targets
                    .entry(canonical_target)
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            }
            if !canonical_targets.is_empty() {
                transitions.insert(label, canonical_targets.into_iter().collect());
            }
        }

        let canonical_state = NWAStateType {
            final_weight: old_state.final_weight.clone(),
            transitions,
            epsilons: epsilons.into_iter().collect(),
        };

        let state_hash = structural_hash_nwa_state(&canonical_state);
        let mut canonical_id = None;
        if let Some(candidates) = hash_buckets.get(&state_hash) {
            for &candidate in candidates {
                if canonical_states[candidate as usize] == canonical_state {
                    canonical_id = Some(candidate);
                    merged += 1;
                    break;
                }
            }
        }

        let canonical_id = canonical_id.unwrap_or_else(|| {
            let new_id = canonical_states.len() as u32;
            canonical_states.push(canonical_state);
            hash_buckets.entry(state_hash).or_default().push(new_id);
            new_id
        });
        remap[old_state_id] = canonical_id;
    }

    if merged == 0 {
        return;
    }

    let mut start_states = Vec::with_capacity(nwa.start_states().len());
    let mut seen_start_states = HashSet::new();
    for &start_state in nwa.start_states() {
        let canonical_start = remap[start_state as usize];
        if seen_start_states.insert(canonical_start) {
            start_states.push(canonical_start);
        }
    }

    *nwa.states_mut() = canonical_states;
    nwa.set_start_states(start_states);
}

// ─── Prune / Reachability ────────────────────────────────────────────────────

fn retain_nwa_states(nwa: &mut NWA, retain: &[bool], drop_empty_weights: bool) -> bool {
    if retain.iter().all(|&f| f) {
        return false;
    }

    let mut remap = vec![u32::MAX; nwa.states().len()];
    let mut new_states = Vec::with_capacity(retain.iter().filter(|&&f| f).count());

    for (old_id, state) in nwa.states().iter().enumerate() {
        if retain[old_id] {
            remap[old_id] = new_states.len() as u32;
            new_states.push(state.clone());
        }
    }

    for state in &mut new_states {
        state.epsilons.retain(|(target, weight)| {
            retain[*target as usize] && (!drop_empty_weights || !weight.is_empty())
        });
        for (target, _) in &mut state.epsilons {
            *target = remap[*target as usize];
        }

        for targets in state.transitions.values_mut() {
            targets.retain(|(target, weight)| {
                retain[*target as usize] && (!drop_empty_weights || !weight.is_empty())
            });
            for (target, _) in targets.iter_mut() {
                *target = remap[*target as usize];
            }
        }
        state.transitions.retain(|_, targets| !targets.is_empty());
    }

    nwa.set_start_states(nwa
        .start_states()
        .iter()
        .copied()
        .filter(|state_id| retain[*state_id as usize])
        .map(|state_id| remap[state_id as usize])
        .collect());
    *nwa.states_mut() = new_states;
    true
}

fn compute_forward_reachable(nwa: &NWA) -> Vec<bool> {
    let mut reachable = vec![false; nwa.states().len()];
    let mut queue = VecDeque::new();

    for &start in nwa.start_states() {
        if let Some(flag) = reachable.get_mut(start as usize) {
            if !*flag {
                *flag = true;
                queue.push_back(start);
            }
        }
    }

    while let Some(state_id) = queue.pop_front() {
        let state = &nwa.states()[state_id as usize];
        for (target, _) in &state.epsilons {
            if let Some(flag) = reachable.get_mut(*target as usize) {
                if !*flag {
                    *flag = true;
                    queue.push_back(*target);
                }
            }
        }
        for (target, _) in state.transitions.values().flatten() {
            if let Some(flag) = reachable.get_mut(*target as usize) {
                if !*flag {
                    *flag = true;
                    queue.push_back(*target);
                }
            }
        }
    }

    reachable
}

pub(crate) fn prune_unreachable_states(nwa: &mut NWA) -> bool {
    if nwa.states().is_empty() {
        return false;
    }
    let reachable = compute_forward_reachable(nwa);
    retain_nwa_states(nwa, &reachable, false)
}

fn topological_order(nwa: &NWA) -> Vec<usize> {
    let mut in_degree = vec![0u32; nwa.states().len()];
    for state in nwa.states() {
        for (dst, _) in &state.epsilons {
            in_degree[*dst as usize] += 1;
        }
        for targets in state.transitions.values() {
            for (dst, _) in targets {
                in_degree[*dst as usize] += 1;
            }
        }
    }

    let mut queue = VecDeque::new();
    for (state_id, degree) in in_degree.iter().enumerate() {
        if *degree == 0 {
            queue.push_back(state_id);
        }
    }

    let mut order = Vec::with_capacity(nwa.states().len());
    while let Some(state_id) = queue.pop_front() {
        order.push(state_id);
        let state = &nwa.states()[state_id];
        for (dst, _) in &state.epsilons {
            in_degree[*dst as usize] -= 1;
            if in_degree[*dst as usize] == 0 {
                queue.push_back(*dst as usize);
            }
        }
        for targets in state.transitions.values() {
            for (dst, _) in targets {
                in_degree[*dst as usize] -= 1;
                if in_degree[*dst as usize] == 0 {
                    queue.push_back(*dst as usize);
                }
            }
        }
    }

    order
}

fn compute_coreachable_nwa(nwa: &NWA) -> Vec<bool> {
    if nwa.states().is_empty() {
        return Vec::new();
    }

    let mut reverse_edges: Vec<Vec<usize>> = vec![Vec::new(); nwa.states().len()];
    for (state_id, state) in nwa.states().iter().enumerate() {
        for (dst, weight) in &state.epsilons {
            if !weight.is_empty() {
                reverse_edges[*dst as usize].push(state_id);
            }
        }
        for targets in state.transitions.values() {
            for (dst, weight) in targets {
                if !weight.is_empty() {
                    reverse_edges[*dst as usize].push(state_id);
                }
            }
        }
    }

    let mut coreachable = vec![false; nwa.states().len()];
    let mut queue = VecDeque::new();
    for (state_id, state) in nwa.states().iter().enumerate() {
        if state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty()) {
            coreachable[state_id] = true;
            queue.push_back(state_id);
        }
    }

    while let Some(state_id) = queue.pop_front() {
        for &pred in &reverse_edges[state_id] {
            if !coreachable[pred] {
                coreachable[pred] = true;
                queue.push_back(pred);
            }
        }
    }

    coreachable
}

pub(crate) fn prune_non_coreachable_states(nwa: &mut NWA) -> bool {
    if nwa.states().is_empty() {
        return false;
    }
    let coreachable = compute_coreachable_nwa(nwa);
    retain_nwa_states(nwa, &coreachable, true)
}

// ─── Collapse always-allowed ─────────────────────────────────────────────────

fn propagate_incoming_labels(
    nwa: &NWA,
    terminals_count: usize,
) -> Vec<HashSet<TerminalID>> {
    let mut incoming = vec![HashSet::new(); nwa.states().len()];
    let mut queue = VecDeque::new();
    let mut in_queue = vec![false; nwa.states().len()];

    for &start in nwa.start_states() {
        queue.push_back(start);
        in_queue[start as usize] = true;
    }

    while let Some(state_id) = queue.pop_front() {
        in_queue[state_id as usize] = false;
        let incoming_labels = incoming[state_id as usize].clone();

        let state = &nwa.states()[state_id as usize];

        for (dst, _) in &state.epsilons {
            let labels_before = incoming[*dst as usize].len();
            incoming[*dst as usize].extend(incoming_labels.iter().copied());
            if incoming[*dst as usize].len() != labels_before && !in_queue[*dst as usize] {
                in_queue[*dst as usize] = true;
                queue.push_back(*dst);
            }
        }

        for (&label, targets) in &state.transitions {
            if label < 0 || (label as usize) >= terminals_count {
                continue;
            }
            for (dst, _) in targets {
                if incoming[*dst as usize].insert(label as TerminalID) && !in_queue[*dst as usize] {
                    in_queue[*dst as usize] = true;
                    queue.push_back(*dst);
                }
            }
        }
    }

    incoming
}

fn propagate_collapse_context(
    nwa: &NWA,
    terminals_count: usize,
) -> (Vec<HashSet<TerminalID>>, Vec<Weight>) {
    let mut incoming = vec![HashSet::new(); nwa.states().len()];
    let mut domain = vec![Weight::empty(); nwa.states().len()];
    let mut queue = VecDeque::new();
    let mut in_queue = vec![false; nwa.states().len()];

    for &start in nwa.start_states() {
        domain[start as usize] = Weight::all();
        queue.push_back(start);
        in_queue[start as usize] = true;
    }

    while let Some(state_id) = queue.pop_front() {
        in_queue[state_id as usize] = false;
        let state_domain = domain[state_id as usize].clone();
        if state_domain.is_empty() {
            continue;
        }

        let state = &nwa.states()[state_id as usize];
        let incoming_labels = incoming[state_id as usize].clone();

        for (dst, _) in &state.epsilons {
            let next_domain = domain[*dst as usize].union(&state_domain);
            let domain_changed = !next_domain.is_subset(&domain[*dst as usize]);
            if domain_changed {
                domain[*dst as usize] = next_domain;
            }

            let labels_before = incoming[*dst as usize].len();
            incoming[*dst as usize].extend(incoming_labels.iter().copied());
            let labels_changed = incoming[*dst as usize].len() != labels_before;

            if (domain_changed || labels_changed) && !in_queue[*dst as usize] {
                in_queue[*dst as usize] = true;
                queue.push_back(*dst);
            }
        }

        for (&label, targets) in &state.transitions {
            if label < 0 || (label as usize) >= terminals_count {
                continue;
            }

            for (dst, weight) in targets {
                let contrib = state_domain.intersection(weight);
                let next_domain = domain[*dst as usize].union(&contrib);
                let domain_changed = !next_domain.is_subset(&domain[*dst as usize]);
                if domain_changed {
                    domain[*dst as usize] = next_domain;
                }

                let labels_changed = incoming[*dst as usize].insert(label as TerminalID);
                if (domain_changed || labels_changed) && !in_queue[*dst as usize] {
                    in_queue[*dst as usize] = true;
                    queue.push_back(*dst);
                }
            }
        }
    }

    (incoming, domain)
}

fn allowed_labels_by_state(
    incoming: &[HashSet<TerminalID>],
    always_allowed_by_label: &[Vec<TerminalID>],
) -> Vec<HashSet<TerminalID>> {
    let mut allowed_by_state = vec![HashSet::new(); incoming.len()];

    for (state_id, labels) in incoming.iter().enumerate() {
        let Some(&first_label) = labels.iter().next() else {
            continue;
        };
        let Some(first_follows) = always_allowed_by_label.get(first_label as usize) else {
            continue;
        };

        let mut allowed: HashSet<TerminalID> = first_follows.iter().copied().collect();
        for &label in labels.iter().skip(1) {
            let Some(follows) = always_allowed_by_label.get(label as usize) else {
                continue;
            };
            allowed.retain(|terminal| follows.contains(terminal));
            if allowed.is_empty() {
                break;
            }
        }
        allowed_by_state[state_id] = allowed;
    }

    allowed_by_state
}

fn collapse_single_allowed_transitions(
    nwa: &mut NWA,
    topo_order: &[usize],
    domain: &[Weight],
    allowed_by_state: &[HashSet<TerminalID>],
    terminals_count: usize,
) -> bool {
    let mut final_weights: Vec<Option<Weight>> =
        nwa.states().iter().map(|state| state.final_weight.clone()).collect();
    let mut changed = false;

    for &state_id in topo_order.iter().rev() {
        let allowed = &allowed_by_state[state_id];
        if allowed.len() != 1 {
            continue;
        }
        let only_allowed = *allowed.iter().next().expect("singleton set checked above");

        let domain_state = &domain[state_id];
        if domain_state.is_empty() {
            continue;
        }

        let state = &mut nwa.states_mut()[state_id];
        let mut state_final_weight = final_weights[state_id].clone();
        let mut labels_to_remove = Vec::new();

        for (&label, targets) in state.transitions.iter_mut() {
            if label < 0 || (label as usize) >= terminals_count {
                continue;
            }
            if label as TerminalID != only_allowed {
                continue;
            }

            let mut new_targets = Vec::new();
            for (dst, weight) in targets.iter() {
                let Some(dst_final_weight) = final_weights[*dst as usize].as_ref() else {
                    new_targets.push((*dst, weight.clone()));
                    continue;
                };

                let reach = domain_state.intersection(weight);
                if !reach.is_empty() && reach.is_subset(dst_final_weight) {
                    let contrib = dst_final_weight.intersection(weight);
                    if !contrib.is_empty() {
                        state_final_weight = Some(match state_final_weight.take() {
                            Some(existing) => existing.union(&contrib),
                            None => contrib,
                        });
                    }
                    changed = true;
                    continue;
                }

                new_targets.push((*dst, weight.clone()));
            }

            if new_targets.is_empty() {
                labels_to_remove.push(label);
            } else {
                *targets = new_targets;
            }
        }

        for label in labels_to_remove {
            state.transitions.remove(&label);
        }

        state.final_weight = state_final_weight.clone();
        final_weights[state_id] = state_final_weight;
    }

    changed
}

pub(crate) fn collapse_always_allowed(
    nwa: &mut NWA,
    always_allowed_by_label: &[Vec<TerminalID>],
    terminals_count: usize,
) -> bool {
    if always_allowed_by_label.is_empty() || terminals_count == 0 || nwa.states().is_empty() {
        return false;
    }

    // Early exit: if no terminal has any always-allowed followers, nothing to collapse.
    if always_allowed_by_label.iter().all(|v| v.is_empty()) {
        return false;
    }

    let topo_started_at = std::time::Instant::now();
    let topo_order = topological_order(nwa);
    if topo_order.is_empty() {
        return false;
    }
    let topo_ms = topo_started_at.elapsed().as_secs_f64() * 1000.0;

    // Lightweight check: propagate only incoming labels (no Weight arithmetic)
    // and check if any state has exactly 1 allowed label before doing expensive
    // domain propagation.
    let labels_started_at = std::time::Instant::now();
    let incoming_labels = propagate_incoming_labels(nwa, terminals_count);
    let labels_ms = labels_started_at.elapsed().as_secs_f64() * 1000.0;

    let allowed_started_at = std::time::Instant::now();
    let allowed_by_state = allowed_labels_by_state(&incoming_labels, always_allowed_by_label);
    let allowed_ms = allowed_started_at.elapsed().as_secs_f64() * 1000.0;

    let any_singleton = allowed_by_state.iter().any(|s| s.len() == 1);
    if !any_singleton {
        if crate::compiler::stages::id_map_and_terminal_dwa::types::debug_profile_enabled() {
            eprintln!(
                "[glrmask/debug][collapse] early_exit=no_singletons topo_ms={:.3} labels_ms={:.3} allowed_ms={:.3}",
                topo_ms, labels_ms, allowed_ms,
            );
        }
        return false;
    }

    // Full propagation: compute domains (Weight arithmetic) only when needed.
    let propagate_started_at = std::time::Instant::now();
    let (_, domain) = propagate_collapse_context(nwa, terminals_count);
    let propagate_ms = propagate_started_at.elapsed().as_secs_f64() * 1000.0;

    let collapse_started_at = std::time::Instant::now();
    let mut changed =
        collapse_single_allowed_transitions(nwa, &topo_order, &domain, &allowed_by_state, terminals_count);
    let collapse_inner_ms = collapse_started_at.elapsed().as_secs_f64() * 1000.0;

    if prune_unreachable_states(nwa) {
        changed = true;
    }

    if crate::compiler::stages::id_map_and_terminal_dwa::types::debug_profile_enabled() {
        eprintln!(
            "[glrmask/debug][collapse] changed={} topo_ms={:.3} propagate_ms={:.3} allowed_ms={:.3} collapse_ms={:.3}",
            changed, topo_ms, propagate_ms, allowed_ms, collapse_inner_ms,
        );
    }

    changed
}

// ─── Disallowed-follow subtraction ───────────────────────────────────────────

/// Apply disallowed-follow constraints by subtracting a follow-pair DFA from
/// the NWA. Takes the pre-computed `disallowed_follows` map and `num_terminals`.
pub(crate) fn apply_disallowed_follow_constraints(
    nwa: &mut NWA,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: usize,
) {
    let normalized = normalize_disallowed_follows(num_terminals, disallowed_follows);
    if normalized.iter().all(|bits| bits.is_zero()) {
        return;
    }

    let disallowed_dfa = build_disallowed_follow_dfa(&normalized);
    *nwa = subtract_disallowed_dfa(nwa, &disallowed_dfa);
}

fn subtract_disallowed_dfa(nwa: &NWA, right: &crate::automata::unweighted::dfa::DFA) -> NWA {
    type ProdState = (u32, Option<u32>);

    let right_start = (!right.states.is_empty()).then_some(right.start_state);

    let mut result = NWA::new(0, 0);
    let mut state_ids: HashMap<ProdState, u32> = HashMap::new();
    let mut worklist: VecDeque<ProdState> = VecDeque::new();

    let get_or_create = |result: &mut NWA,
                         state_ids: &mut HashMap<ProdState, u32>,
                         worklist: &mut VecDeque<ProdState>,
                         ps: ProdState|
     -> u32 {
        if let Some(&id) = state_ids.get(&ps) {
            id
        } else {
            let id = result.add_state();
            state_ids.insert(ps, id);
            worklist.push_back(ps);
            id
        }
    };

    for &nwa_start in nwa.start_states() {
        let ps = (nwa_start, right_start);
        let id = get_or_create(&mut result, &mut state_ids, &mut worklist, ps);
        result.start_states_mut().push(id);
    }

    while let Some((nwa_sid, dfa_sid)) = worklist.pop_front() {
        let result_sid = state_ids[&(nwa_sid, dfa_sid)];
        let nwa_state = &nwa.states()[nwa_sid as usize];
        let dfa_accepting = dfa_sid
            .map(|s| right.states[s as usize].is_accepting)
            .unwrap_or(false);

        if !dfa_accepting {
            if let Some(fw) = &nwa_state.final_weight {
                result.set_final_weight(result_sid, fw.clone());
            }
        }

        for (nwa_dst, weight) in &nwa_state.epsilons {
            let ps = (*nwa_dst, dfa_sid);
            let dst_id = get_or_create(&mut result, &mut state_ids, &mut worklist, ps);
            result.add_epsilon(result_sid, dst_id, weight.clone());
        }

        for (&label, targets) in &nwa_state.transitions {
            let next_dfa = if label >= 0 {
                dfa_sid.and_then(|s| {
                    right.states[s as usize].transitions.get(&label).copied()
                })
            } else {
                dfa_sid
            };

            for (nwa_dst, weight) in targets {
                let ps = (*nwa_dst, next_dfa);
                let dst_id = get_or_create(&mut result, &mut state_ids, &mut worklist, ps);
                result.add_transition(result_sid, label, dst_id, weight.clone());
            }
        }
    }

    result
}
