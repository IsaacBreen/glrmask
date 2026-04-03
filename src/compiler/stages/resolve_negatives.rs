//! Resolve negative parser-state labels in weighted NWAs.

use std::collections::VecDeque;
use std::sync::Arc;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::automata::weighted::nwa::{NWA, NWAState};
use crate::compiler::glr::labels::{DEFAULT_LABEL, is_negative_label, negative_to_positive_label};
use crate::ds::weight::Weight;

type QueryKey = (u32, i32);
type CancellationTask = (u32, u32, i32);
type QueryWeights = Vec<Option<FxHashMap<QueryKey, Weight>>>;
type DerivedEpsilons = Vec<Option<FxHashMap<u32, Weight>>>;

fn merge_weight(entry: &mut Weight, add: Weight) -> bool {
    if add.is_empty() {
        return false;
    }
    if entry.is_empty() {
        *entry = add;
        return true;
    }
    let updated = entry.union(&add);
    if updated != *entry {
        *entry = updated;
        true
    } else {
        false
    }
}

fn intersect_with_single_weight_hint(
    left: &Weight,
    left_single: Option<&(u32, u32, Arc<RangeSetBlaze<u32>>)>,
    right: &Weight,
) -> Weight {
    if left.is_full() {
        return right.clone();
    }
    if right.is_full() {
        return left.clone();
    }
    if let Some((start, end, tokens)) = left_single {
        right.intersect_single_parts(*start, *end, tokens)
    } else {
        left.intersection(right)
    }
}

fn intersect_or_clone_right_if_subset(left: &Weight, right: &Weight) -> Weight {
    if right.is_subset(left) {
        right.clone()
    } else {
        left.intersection(right)
    }
}

#[derive(Clone)]
struct PredEdge {
    from: usize,
    weight: Weight,
}

pub(crate) fn compute_cancellations(nwa: &NWA) -> Vec<(u32, u32, Weight)> {
    compute_cancellations_range(nwa, full_state_range(nwa))
}

fn full_state_range(nwa: &NWA) -> std::ops::Range<u32> {
    0..nwa.states.len() as u32
}

fn enqueue_cancellation_task(
    worklist: &mut VecDeque<CancellationTask>,
    queued: &mut FxHashSet<CancellationTask>,
    state: u32,
    source_state: u32,
    positive_label: i32,
) {
    let task = (state, source_state, positive_label);
    if queued.insert(task) {
        worklist.push_back(task);
    }
}

fn record_query_weight(
    query_weights: &mut QueryWeights,
    state: u32,
    query_key: QueryKey,
    add: Weight,
) -> bool {
    let entry = query_weights[state as usize]
        .get_or_insert_with(FxHashMap::default)
        .entry(query_key)
        .or_insert_with(Weight::empty);
    merge_weight(entry, add)
}

fn queue_query_weight(
    query_weights: &mut QueryWeights,
    worklist: &mut VecDeque<CancellationTask>,
    queued: &mut FxHashSet<CancellationTask>,
    state: u32,
    source_state: u32,
    positive_label: i32,
    add: Weight,
) {
    if record_query_weight(query_weights, state, (source_state, positive_label), add) {
        enqueue_cancellation_task(worklist, queued, state, source_state, positive_label);
    }
}

fn propagate_query_through_derived_epsilons(
    query_weight_to_current: &Weight,
    query_single: Option<&(u32, u32, Arc<RangeSetBlaze<u32>>)>,
    current_state: u32,
    source_state: u32,
    positive_label: i32,
    query_weights: &mut QueryWeights,
    worklist: &mut VecDeque<CancellationTask>,
    queued: &mut FxHashSet<CancellationTask>,
    derived_epsilons: &DerivedEpsilons,
) {
    let Some(derived_from_current) = derived_epsilons[current_state as usize].as_ref() else {
        return;
    };

    for (&target_state, epsilon_weight) in derived_from_current {
        let propagated = intersect_with_single_weight_hint(
            query_weight_to_current,
            query_single,
            epsilon_weight,
        );
        if propagated.is_empty() {
            continue;
        }
        queue_query_weight(
            query_weights,
            worklist,
            queued,
            target_state,
            source_state,
            positive_label,
            propagated,
        );
    }
}

fn extend_derived_epsilons(
    query_weight_to_current: &Weight,
    query_single: Option<&(u32, u32, Arc<RangeSetBlaze<u32>>)>,
    source_state: u32,
    target_state: u32,
    edge_weight: &Weight,
    query_weights: &mut QueryWeights,
    worklist: &mut VecDeque<CancellationTask>,
    queued: &mut FxHashSet<CancellationTask>,
    derived_epsilons: &mut DerivedEpsilons,
) {
    let new_derived_weight = intersect_with_single_weight_hint(
        query_weight_to_current,
        query_single,
        edge_weight,
    );
    if new_derived_weight.is_empty() {
        return;
    }

    let derived_weight = derived_epsilons[source_state as usize]
        .get_or_insert_with(FxHashMap::default)
        .entry(target_state)
        .or_insert_with(Weight::empty);
    if !merge_weight(derived_weight, new_derived_weight) {
        return;
    }
    let derived_weight = derived_weight.clone();

    let Some(existing_queries) = query_weights[source_state as usize].clone() else {
        return;
    };

    for ((upstream_source_state, upstream_label), upstream_weight) in existing_queries {
        let propagated = intersect_or_clone_right_if_subset(&upstream_weight, &derived_weight);
        if propagated.is_empty() {
            continue;
        }
        queue_query_weight(
            query_weights,
            worklist,
            queued,
            target_state,
            upstream_source_state,
            upstream_label,
            propagated,
        );
    }
}

fn collect_non_empty_derived_epsilons(
    derived_epsilons: DerivedEpsilons,
) -> Vec<(u32, u32, Weight)> {
    let mut result = Vec::new();

    for (from_state, targets) in derived_epsilons.into_iter().enumerate() {
        let Some(targets) = targets else {
            continue;
        };
        for (to_state, weight) in targets {
            if !weight.is_empty() {
                result.push((from_state as u32, to_state, weight));
            }
        }
    }

    result
}

pub(crate) fn compute_cancellations_range(
    nwa: &NWA,
    range: std::ops::Range<u32>,
) -> Vec<(u32, u32, Weight)> {
    let state_count = nwa.states.len() as u32;
    if state_count == 0 {
        return Vec::new();
    }

    let mut query_weights: QueryWeights = vec![None; state_count as usize];
    let mut worklist = VecDeque::<CancellationTask>::new();
    let mut queued: FxHashSet<CancellationTask> = FxHashSet::default();
    let mut derived_epsilons: DerivedEpsilons = vec![None; state_count as usize];

    for source_state in range {
        if source_state >= state_count {
            continue;
        }

        for (&label, targets) in &nwa.states[source_state as usize].transitions {
            if !is_negative_label(label) {
                continue;
            }
            let positive_label = negative_to_positive_label(label);

            for (target_state, weight) in targets {
                if *target_state >= state_count || weight.is_empty() {
                    continue;
                }

                queue_query_weight(
                    &mut query_weights,
                    &mut worklist,
                    &mut queued,
                    *target_state,
                    source_state,
                    positive_label,
                    weight.clone(),
                );
            }
        }
    }

    while let Some((current_state, source_state, positive_label)) = worklist.pop_front() {
        queued.remove(&(current_state, source_state, positive_label));
        let Some(query_weight_to_current) = query_weights[current_state as usize]
            .as_ref()
            .and_then(|weights| weights.get(&(source_state, positive_label)))
            .cloned()
        else {
            continue;
        };
        let query_single = query_weight_to_current.single_compact_entry_parts();

        propagate_query_through_derived_epsilons(
            &query_weight_to_current,
            query_single.as_ref(),
            current_state,
            source_state,
            positive_label,
            &mut query_weights,
            &mut worklist,
            &mut queued,
            &derived_epsilons,
        );

        if let Some(positive_targets) = nwa.states[current_state as usize]
            .transitions
            .get(&positive_label)
        {
            for (target_state, edge_weight) in positive_targets {
                if *target_state < state_count {
                    extend_derived_epsilons(
                        &query_weight_to_current,
                        query_single.as_ref(),
                        source_state,
                        *target_state,
                        edge_weight,
                        &mut query_weights,
                        &mut worklist,
                        &mut queued,
                        &mut derived_epsilons,
                    );
                }
            }
        }

        if let Some(default_targets) = nwa.states[current_state as usize]
            .transitions
            .get(&DEFAULT_LABEL)
        {
            for (target_state, edge_weight) in default_targets {
                if *target_state < state_count {
                    extend_derived_epsilons(
                        &query_weight_to_current,
                        query_single.as_ref(),
                        source_state,
                        *target_state,
                        edge_weight,
                        &mut query_weights,
                        &mut worklist,
                        &mut queued,
                        &mut derived_epsilons,
                    );
                }
            }
        }

        for (target_state, epsilon_weight) in &nwa.states[current_state as usize].epsilons {
            if *target_state >= state_count {
                continue;
            }

            let propagated = intersect_with_single_weight_hint(
                &query_weight_to_current,
                query_single.as_ref(),
                epsilon_weight,
            );
            if propagated.is_empty() {
                continue;
            }

            queue_query_weight(
                &mut query_weights,
                &mut worklist,
                &mut queued,
                *target_state,
                source_state,
                positive_label,
                propagated,
            );
        }
    }

    collect_non_empty_derived_epsilons(derived_epsilons)
}

pub(crate) fn apply_cancellations_range(nwa: &mut NWA, range: std::ops::Range<u32>) {
    for (from, to, weight) in compute_cancellations_range(nwa, range) {
        nwa.add_epsilon(from, to, weight);
    }
}

fn merge_final_weight(entry: &mut Option<Weight>, add: Weight) -> bool {
    if add.is_empty() {
        return false;
    }
    match entry {
        Some(existing) => {
            let updated = existing.union(&add);
            if updated != *existing {
                *existing = updated;
                true
            } else {
                false
            }
        }
        None => {
            *entry = Some(add);
            true
        }
    }
}

fn is_live_finality_edge(target_state: u32, weight: &Weight, state_count: usize) -> bool {
    (target_state as usize) < state_count && !weight.is_empty()
}

fn for_each_live_finality_edge(
    state: &NWAState,
    state_count: usize,
    mut visit: impl FnMut(usize, &Weight),
) {
    for (target_state, weight) in &state.epsilons {
        if !is_live_finality_edge(*target_state, weight, state_count) {
            continue;
        }
        visit(*target_state as usize, weight);
    }

    for (&label, targets) in &state.transitions {
        if label != DEFAULT_LABEL && !is_negative_label(label) {
            continue;
        }

        for (target_state, weight) in targets {
            if !is_live_finality_edge(*target_state, weight, state_count) {
                continue;
            }
            visit(*target_state as usize, weight);
        }
    }
}

fn build_finality_preds_and_outdegree(nwa: &NWA) -> (Vec<Vec<PredEdge>>, Vec<usize>) {
    let state_count = nwa.states.len();
    let mut preds = vec![Vec::<PredEdge>::new(); state_count];
    let mut outdegree = vec![0usize; state_count];

    for (from_state, state) in nwa.states.iter().enumerate() {
        for_each_live_finality_edge(state, state_count, |target_state, weight| {
            preds[target_state].push(PredEdge {
                from: from_state,
                weight: weight.clone(),
            });
            outdegree[from_state] += 1;
        });
    }

    (preds, outdegree)
}

fn build_finality_reverse_topo_order(
    preds: &[Vec<PredEdge>],
    mut outdegree: Vec<usize>,
) -> Option<Vec<usize>> {
    let state_count = preds.len();

    let mut queue = VecDeque::new();
    for (state_id, degree) in outdegree.iter().enumerate() {
        if *degree == 0 {
            queue.push_back(state_id);
        }
    }

    let mut reverse_topo_order = Vec::with_capacity(state_count);
    while let Some(state_id) = queue.pop_front() {
        reverse_topo_order.push(state_id);
        for edge in &preds[state_id] {
            outdegree[edge.from] -= 1;
            if outdegree[edge.from] == 0 {
                queue.push_back(edge.from);
            }
        }
    }

    (reverse_topo_order.len() == state_count).then_some(reverse_topo_order)
}

fn collect_initial_final_weights(nwa: &NWA) -> Vec<Option<Weight>> {
    nwa.states
        .iter()
        .map(|state| state.final_weight.clone().filter(|weight| !weight.is_empty()))
        .collect()
}

fn write_final_weights(nwa: &mut NWA, reachable_final_weights: Vec<Option<Weight>>) {
    for (state_id, final_weight) in reachable_final_weights.into_iter().enumerate() {
        nwa.states[state_id].final_weight = final_weight.filter(|weight| !weight.is_empty());
    }
}

fn propagate_final_weights_to_predecessors(
    preds: &[Vec<PredEdge>],
    reachable_final_weights: &mut [Option<Weight>],
    state_id: usize,
    mut on_change: impl FnMut(usize),
) {
    let Some(reachable_final) = reachable_final_weights[state_id].clone() else {
        return;
    };

    for edge in &preds[state_id] {
        let propagated = reachable_final.intersection(&edge.weight);
        if merge_final_weight(&mut reachable_final_weights[edge.from], propagated) {
            on_change(edge.from);
        }
    }
}

fn apply_finality_fixpoint_worklist(
    nwa: &NWA,
    preds: &[Vec<PredEdge>],
    reachable_final_weights: &mut [Option<Weight>],
) {
    let n = nwa.states.len();
    let mut worklist = VecDeque::<usize>::new();
    let mut queued = vec![false; n];

    for (state_id, final_weight) in reachable_final_weights.iter().enumerate() {
        if final_weight.is_some() {
            queued[state_id] = true;
            worklist.push_back(state_id);
        }
    }

    while let Some(state_id) = worklist.pop_front() {
        queued[state_id] = false;
        propagate_final_weights_to_predecessors(
            preds,
            reachable_final_weights,
            state_id,
            |pred_state| {
                if queued[pred_state] {
                    return;
                }

                queued[pred_state] = true;
                worklist.push_back(pred_state);
            },
        );
    }
}

fn apply_finality_fixpoint_acyclic(
    preds: &[Vec<PredEdge>],
    reachable_final_weights: &mut [Option<Weight>],
    reverse_topo_order: &[usize],
) {
    for &state_id in reverse_topo_order {
        propagate_final_weights_to_predecessors(
            preds,
            reachable_final_weights,
            state_id,
            |_| {},
        );
    }
}

pub(crate) fn apply_finality_fixpoint(nwa: &mut NWA) {
    let n = nwa.states.len();
    if n == 0 {
        return;
    }
    let (preds, outdegree) = build_finality_preds_and_outdegree(nwa);
    let reverse_topo_order = build_finality_reverse_topo_order(&preds, outdegree);
    let mut reachable_final_weights = collect_initial_final_weights(nwa);

    if let Some(reverse_topo_order) = reverse_topo_order.as_deref() {
        apply_finality_fixpoint_acyclic(
            &preds,
            &mut reachable_final_weights,
            reverse_topo_order,
        );
    } else {
        apply_finality_fixpoint_worklist(nwa, &preds, &mut reachable_final_weights);
    }

    write_final_weights(nwa, reachable_final_weights);
}

pub(crate) fn remove_negative_transitions(nwa: &mut NWA) {
    for state in &mut nwa.states {
        state.transitions.retain(|label, _| !is_negative_label(*label));
    }
}

fn has_live_final_weight(state: &NWAState) -> bool {
    state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty())
}

fn has_non_default_transitions(state: &NWAState) -> bool {
    state
        .transitions
        .iter()
        .any(|(label, targets)| *label != DEFAULT_LABEL && !targets.is_empty())
}

fn is_terminal_candidate(state: &NWAState) -> bool {
    !has_non_default_transitions(state)
        && state.epsilons.is_empty()
        && has_live_final_weight(state)
}

fn default_targets_are_terminal(state: &NWAState, terminal_states: &[bool]) -> bool {
    match state.transitions.get(&DEFAULT_LABEL) {
        None => true,
        Some(targets) => targets.iter().all(|(target, _)| {
            terminal_states
                .get(*target as usize)
                .copied()
                .unwrap_or(false)
        }),
    }
}

fn grow_terminal_state_set(nwa: &NWA, terminal_states: &mut [bool]) {
    loop {
        let mut changed = false;
        for (state_id, state) in nwa.states.iter().enumerate() {
            if terminal_states[state_id] || !is_terminal_candidate(state) {
                continue;
            }

            if default_targets_are_terminal(state, terminal_states) {
                terminal_states[state_id] = true;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }
}

fn prune_terminal_default_targets(nwa: &mut NWA, terminal_states: &[bool]) {
    for state in &mut nwa.states {
        if let Some(targets) = state.transitions.get_mut(&DEFAULT_LABEL) {
            targets.retain(|(target, _)| !terminal_states[*target as usize]);
        }
        state.transitions.retain(|_, targets| !targets.is_empty());
    }
}

pub(crate) fn remove_redundant_default_transitions(nwa: &mut NWA) {
    let mut terminal_states: Vec<bool> = nwa.states.iter().map(is_terminal_candidate).collect();

    grow_terminal_state_set(nwa, &mut terminal_states);
    prune_terminal_default_targets(nwa, &terminal_states);
}

pub(crate) fn resolve_negative_codes_in_nwa(nwa: &mut NWA) {
    let state_range = full_state_range(nwa);
    apply_cancellations_range(nwa, state_range);
    apply_finality_fixpoint(nwa);
    remove_negative_transitions(nwa);
    remove_redundant_default_transitions(nwa);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::labels::{encode_negative_label, encode_positive_label};
    use std::collections::{BTreeMap, HashMap, HashSet};

    fn compute_cancellations_reference(nwa: &NWA) -> Vec<(u32, u32, Weight)> {
        let n = nwa.states.len();
        if n == 0 {
            return Vec::new();
        }

        let mut queries: HashMap<u32, HashMap<QueryKey, Weight>> = HashMap::new();
        let mut worklist = VecDeque::<(u32, u32, i32)>::new();
        let mut in_queue = vec![HashSet::<QueryKey>::new(); n];
        let mut new_eps_from: HashMap<u32, HashMap<u32, Weight>> = HashMap::new();

        let merge_into = |entry: &mut Weight, add: Weight| {
            if add.is_empty() {
                return false;
            }
            if entry.is_empty() {
                *entry = add;
                return true;
            }
            let updated = entry.union(&add);
            if updated != *entry {
                *entry = updated;
                true
            } else {
                false
            }
        };

        for a in 0..n {
            for (&label, targets) in &nwa.states[a].transitions {
                if !is_negative_label(label) {
                    continue;
                }
                let c = negative_to_positive_label(label);
                for (b, w_ab) in targets {
                    if *b as usize >= n || w_ab.is_empty() {
                        continue;
                    }
                    let query_key = (a as u32, c);
                    let entry = queries
                        .entry(*b)
                        .or_default()
                        .entry(query_key)
                        .or_insert_with(Weight::empty);
                    if merge_into(entry, w_ab.clone()) {
                        if in_queue[*b as usize].insert(query_key) {
                            worklist.push_back((*b, a as u32, c));
                        }
                    }
                }
            }
        }

        while let Some((s, a, c)) = worklist.pop_front() {
            in_queue[s as usize].remove(&(a, c));
            let Some(w_as) = queries.get(&s).and_then(|m| m.get(&(a, c))).cloned() else {
                continue;
            };

            if let Some(epsilons_from_s) = new_eps_from.get(&s) {
                for (&target, eps_w) in epsilons_from_s {
                    let prop_w = w_as.intersection(eps_w);
                    if prop_w.is_empty() {
                        continue;
                    }
                    let query_key = (a, c);
                    let entry = queries
                        .entry(target)
                        .or_default()
                        .entry(query_key)
                        .or_insert_with(Weight::empty);
                    if merge_into(entry, prop_w) {
                        if in_queue[target as usize].insert(query_key) {
                            worklist.push_back((target, a, c));
                        }
                    }
                }
            }

            let check_cancellations = |target: u32,
                                       w_st: &Weight,
                                       queries: &mut HashMap<u32, HashMap<QueryKey, Weight>>,
                                       worklist: &mut VecDeque<(u32, u32, i32)>,
                                       in_queue: &mut [HashSet<QueryKey>],
                                       new_eps_from: &mut HashMap<u32, HashMap<u32, Weight>>| {
                let new_eps_w = w_as.intersection(w_st);
                if new_eps_w.is_empty() {
                    return;
                }

                let eps_entry = new_eps_from
                    .entry(a)
                    .or_default()
                    .entry(target)
                    .or_insert_with(Weight::empty);
                let updated_eps = if eps_entry.is_empty() {
                    new_eps_w
                } else {
                    eps_entry.union(&new_eps_w)
                };
                if updated_eps == *eps_entry {
                    return;
                }
                *eps_entry = updated_eps.clone();

                if let Some(queries_at_a) = queries.get(&a).cloned() {
                    for ((a_prime, c_prime), w_a_prime_a) in queries_at_a {
                        let prop_w = w_a_prime_a.intersection(&updated_eps);
                        if prop_w.is_empty() {
                            continue;
                        }
                        let query_key = (a_prime, c_prime);
                        let entry = queries
                            .entry(target)
                            .or_default()
                            .entry(query_key)
                            .or_insert_with(Weight::empty);
                        if merge_into(entry, prop_w) {
                            if in_queue[target as usize].insert(query_key) {
                                worklist.push_back((target, a_prime, c_prime));
                            }
                        }
                    }
                }
            };

            if let Some(pos_targets) = nwa.states[s as usize].transitions.get(&c) {
                for (t, w_st) in pos_targets {
                    if *t as usize >= n {
                        continue;
                    }
                    check_cancellations(
                        *t,
                        w_st,
                        &mut queries,
                        &mut worklist,
                        &mut in_queue,
                        &mut new_eps_from,
                    );
                }
            }

            if let Some(default_targets) = nwa.states[s as usize].transitions.get(&DEFAULT_LABEL) {
                for (target, weight) in default_targets {
                    if *target as usize >= n {
                        continue;
                    }
                    check_cancellations(
                        *target,
                        weight,
                        &mut queries,
                        &mut worklist,
                        &mut in_queue,
                        &mut new_eps_from,
                    );
                }
            }

            for (t, w_st) in &nwa.states[s as usize].epsilons {
                if *t as usize >= n {
                    continue;
                }
                let prop_w = w_as.intersection(w_st);
                if prop_w.is_empty() {
                    continue;
                }
                let query_key = (a, c);
                let entry = queries
                    .entry(*t)
                    .or_default()
                    .entry(query_key)
                    .or_insert_with(Weight::empty);
                if merge_into(entry, prop_w) {
                    if in_queue[*t as usize].insert(query_key) {
                        worklist.push_back((*t, a, c));
                    }
                }
            }
        }

        let mut result = Vec::new();
        for (from, targets) in new_eps_from {
            for (to, w) in targets {
                if !w.is_empty() {
                    result.push((from, to, w));
                }
            }
        }

        result
    }

    fn normalize_cancellations(cancellations: Vec<(u32, u32, Weight)>) -> BTreeMap<(u32, u32), Weight> {
        cancellations
            .into_iter()
            .map(|(from, to, weight)| ((from, to), weight))
            .collect()
    }

    fn add_weighted_transition(nwa: &mut NWA, from: u32, label: i32, to: u32, kind: u8) {
        match kind {
            1 => nwa.add_transition(from, label, to, weight_1()),
            2 => nwa.add_transition(from, label, to, weight_12()),
            _ => {}
        }
    }

    fn add_weighted_epsilon(nwa: &mut NWA, from: u32, to: u32, kind: u8) {
        match kind {
            1 => nwa.add_epsilon(from, to, weight_1()),
            2 => nwa.add_epsilon(from, to, weight_12()),
            _ => {}
        }
    }

    fn set_weighted_final(nwa: &mut NWA, state: u32, kind: u8) {
        match kind {
            1 => nwa.set_final_weight(state, weight_1()),
            2 => nwa.set_final_weight(state, weight_12()),
            _ => {}
        }
    }

    fn apply_finality_fixpoint_reference(nwa: &mut NWA) {
        let n = nwa.states.len();
        if n == 0 {
            return;
        }

        let mut preds = vec![Vec::<PredEdge>::new(); n];
        for (from, state) in nwa.states.iter().enumerate() {
            for (target, weight) in &state.epsilons {
                if *target as usize >= n || weight.is_empty() {
                    continue;
                }
                preds[*target as usize].push(PredEdge {
                    from,
                    weight: weight.clone(),
                });
            }
            for (&label, targets) in &state.transitions {
                if label != DEFAULT_LABEL && !is_negative_label(label) {
                    continue;
                }
                for (target, weight) in targets {
                    if *target as usize >= n || weight.is_empty() {
                        continue;
                    }
                    preds[*target as usize].push(PredEdge {
                        from,
                        weight: weight.clone(),
                    });
                }
            }
        }

        let mut future_final = vec![None::<Weight>; n];
        for state_id in 0..n {
            if let Some(fw) = nwa.states[state_id].final_weight.clone() {
                if fw.is_empty() {
                    continue;
                }
                future_final[state_id] = Some(fw);
            }
        }

        apply_finality_fixpoint_worklist(nwa, &preds, &mut future_final);

        for (state_id, final_weight) in future_final.into_iter().enumerate() {
            nwa.states[state_id].final_weight = final_weight.filter(|weight| !weight.is_empty());
        }
    }

    fn final_weights(nwa: &NWA) -> Vec<Option<Weight>> {
        nwa.states.iter().map(|state| state.final_weight.clone()).collect()
    }

    fn weight_1() -> Weight {
        Weight::from_compact_ranges([(1..=1, [1..=1])])
    }

    fn weight_12() -> Weight {
        Weight::from_compact_ranges([(1..=2, [1..=2])])
    }

    #[test]
    fn test_compute_cancellations_widens_existing_eps_query() {
        let mut nwa = NWA::new(0, 0);
        for _ in 0..3 {
            nwa.add_state();
        }

        let neg1 = encode_negative_label(1);
        let pos1 = encode_positive_label(1);

        nwa.add_transition(0, neg1, 0, weight_1());
        nwa.add_transition(0, neg1, 2, weight_12());
        nwa.add_epsilon(0, 1, weight_1());
        nwa.add_transition(1, pos1, 2, weight_12());
        nwa.add_transition(2, pos1, 1, weight_12());
        nwa.add_epsilon(2, 0, weight_12());

        let cancellations = compute_cancellations(&nwa);

        assert!(
            cancellations
                .iter()
                .any(|(from, to, w)| *from == 0 && *to == 2 && *w == weight_12()),
            "expected widened epsilon 0->2 with weight_12"
        );
        assert!(
            !cancellations
                .iter()
                .any(|(from, to, w)| *from == 0 && *to == 2 && *w == weight_1()),
            "narrower 0->2 weight_1 should have been widened away"
        );
    }

    #[test]
    fn test_compute_cancellations_propagates_later_query_through_existing_epsilon() {
        let mut nwa = NWA::new(0, 0);
        for _ in 0..4 {
            nwa.add_state();
        }

        let neg1 = encode_negative_label(1);
        let pos1 = encode_positive_label(1);

        nwa.add_transition(0, neg1, 1, weight_12());
        nwa.add_transition(1, pos1, 2, weight_12());
        nwa.add_transition(3, neg1, 0, weight_1());
        nwa.add_transition(2, pos1, 1, weight_12());

        let cancellations = compute_cancellations(&nwa);

        assert!(
            cancellations
                .iter()
                .any(|(from, to, w)| *from == 0 && *to == 2 && *w == weight_12()),
            "expected initial epsilon 0->2 with weight_12"
        );
        assert!(
            cancellations
                .iter()
                .any(|(from, to, w)| *from == 3 && *to == 1 && *w == weight_1()),
            "expected later query to reuse existing epsilon and create 3->1 with weight_1"
        );
    }

    #[test]
    fn test_compute_cancellations_delta_matches_reference_on_small_family() {
        let neg1 = encode_negative_label(1);
        let pos1 = encode_positive_label(1);

        for config in 0..(3usize.pow(7)) {
            let mut code = config;
            let mut next_kind = || {
                let kind = (code % 3) as u8;
                code /= 3;
                kind
            };

            let mut nwa = NWA::new(0, 0);
            for _ in 0..4 {
                nwa.add_state();
            }

            add_weighted_transition(&mut nwa, 0, neg1, 0, next_kind());
            add_weighted_transition(&mut nwa, 0, neg1, 2, next_kind());
            add_weighted_epsilon(&mut nwa, 0, 1, next_kind());
            add_weighted_transition(&mut nwa, 1, pos1, 2, next_kind());
            add_weighted_transition(&mut nwa, 2, pos1, 1, next_kind());
            add_weighted_epsilon(&mut nwa, 2, 0, next_kind());
            add_weighted_transition(&mut nwa, 3, neg1, 0, next_kind());

            let actual = normalize_cancellations(compute_cancellations(&nwa));
            let expected = normalize_cancellations(compute_cancellations_reference(&nwa));

            assert_eq!(
                actual, expected,
                "delta propagation diverged from reference for config {}",
                config
            );
        }
    }

    #[test]
    fn test_compute_cancellations_range_matches_reference_on_small_family() {
        let neg1 = encode_negative_label(1);
        let pos1 = encode_positive_label(1);

        for config in 0..(3usize.pow(7)) {
            let mut code = config;
            let mut next_kind = || {
                let kind = (code % 3) as u8;
                code /= 3;
                kind
            };

            let mut nwa = NWA::new(0, 0);
            for _ in 0..4 {
                nwa.add_state();
            }

            add_weighted_transition(&mut nwa, 0, neg1, 0, next_kind());
            add_weighted_transition(&mut nwa, 0, neg1, 2, next_kind());
            add_weighted_epsilon(&mut nwa, 0, 1, next_kind());
            add_weighted_transition(&mut nwa, 1, pos1, 2, next_kind());
            add_weighted_transition(&mut nwa, 2, pos1, 1, next_kind());
            add_weighted_epsilon(&mut nwa, 2, 0, next_kind());
            add_weighted_transition(&mut nwa, 3, neg1, 0, next_kind());

            let actual = normalize_cancellations(compute_cancellations_range(
                &nwa,
                0..nwa.states.len() as u32,
            ));
            let expected = normalize_cancellations(compute_cancellations_reference(&nwa));

            assert_eq!(
                actual, expected,
                "range delta propagation diverged from reference for config {}",
                config
            );
        }
    }

    #[test]
    fn test_apply_finality_fixpoint_matches_reference_on_small_acyclic_family() {
        let neg1 = encode_negative_label(1);

        for config in 0..(3usize.pow(6)) {
            let mut code = config;
            let mut next_kind = || {
                let kind = (code % 3) as u8;
                code /= 3;
                kind
            };

            let mut nwa = NWA::new(0, 0);
            for _ in 0..3 {
                nwa.add_state();
            }

            set_weighted_final(&mut nwa, 0, next_kind());
            set_weighted_final(&mut nwa, 1, next_kind());
            set_weighted_final(&mut nwa, 2, next_kind());
            add_weighted_epsilon(&mut nwa, 0, 1, next_kind());
            add_weighted_transition(&mut nwa, 0, DEFAULT_LABEL, 2, next_kind());
            add_weighted_transition(&mut nwa, 1, neg1, 2, next_kind());

            let mut actual = nwa.clone();
            let mut expected = nwa.clone();
            apply_finality_fixpoint(&mut actual);
            apply_finality_fixpoint_reference(&mut expected);

            assert_eq!(
                final_weights(&actual),
                final_weights(&expected),
                "acyclic finality propagation diverged for config {}",
                config
            );
        }
    }

    #[test]
    fn test_apply_finality_fixpoint_matches_reference_on_small_cyclic_family() {
        let neg1 = encode_negative_label(1);

        for config in 0..(3usize.pow(7)) {
            let mut code = config;
            let mut next_kind = || {
                let kind = (code % 3) as u8;
                code /= 3;
                kind
            };

            let mut nwa = NWA::new(0, 0);
            for _ in 0..3 {
                nwa.add_state();
            }

            set_weighted_final(&mut nwa, 0, next_kind());
            set_weighted_final(&mut nwa, 1, next_kind());
            set_weighted_final(&mut nwa, 2, next_kind());
            add_weighted_epsilon(&mut nwa, 0, 1, next_kind());
            add_weighted_transition(&mut nwa, 1, DEFAULT_LABEL, 0, next_kind());
            add_weighted_transition(&mut nwa, 1, neg1, 2, next_kind());
            add_weighted_epsilon(&mut nwa, 2, 1, next_kind());

            let mut actual = nwa.clone();
            let mut expected = nwa.clone();
            apply_finality_fixpoint(&mut actual);
            apply_finality_fixpoint_reference(&mut expected);

            assert_eq!(
                final_weights(&actual),
                final_weights(&expected),
                "cyclic finality propagation diverged for config {}",
                config
            );
        }
    }
}
