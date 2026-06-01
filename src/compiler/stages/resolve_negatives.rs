//! Resolve negative parser-state labels in weighted NWAs.

use std::collections::VecDeque;
use std::sync::Arc;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::automata::weighted::nwa::{NWA, NWAState};
use crate::parser::glr::labels::{DEFAULT_LABEL, is_negative_label, negative_to_positive_label};
use crate::ds::weight::Weight;

type QueryKey = (u32, i32);
type CancellationTask = (u32, u32, i32);
type QueryWeights = Vec<Option<FxHashMap<QueryKey, Weight>>>;
type QueuedQueries = Vec<SmallQueuedQueries>;
type DerivedEpsilons = Vec<Option<FxHashMap<u32, Weight>>>;
type SubsetMemo = FxHashMap<(usize, usize), bool>;

#[derive(Clone, Default)]
enum SmallQueuedQueries {
    #[default]
    Empty,
    One(QueryKey),
    Many(FxHashSet<QueryKey>),
}

impl SmallQueuedQueries {
    fn insert(&mut self, query_key: QueryKey) -> bool {
        match self {
            Self::Empty => {
                *self = Self::One(query_key);
                true
            }
            Self::One(existing) => {
                if *existing == query_key {
                    false
                } else {
                    let previous = *existing;
                    let mut entries = FxHashSet::default();
                    entries.insert(previous);
                    entries.insert(query_key);
                    *self = Self::Many(entries);
                    true
                }
            }
            Self::Many(entries) => entries.insert(query_key),
        }
    }

    fn remove(&mut self, query_key: &QueryKey) {
        match self {
            Self::Empty => {}
            Self::One(existing) => {
                if existing == query_key {
                    *self = Self::Empty;
                }
            }
            Self::Many(entries) => {
                if !entries.remove(query_key) {
                    return;
                }
                match entries.len() {
                    0 => *self = Self::Empty,
                    1 => {
                        let remaining = *entries.iter().next().unwrap();
                        *self = Self::One(remaining);
                    }
                    _ => {}
                }
            }
        }
    }
}

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

fn intersect_or_clone_right_if_subset_cached(
    left: &Weight,
    right: &Weight,
    subset_memo: &mut SubsetMemo,
) -> Weight {
    let key = (left.ptr_key(), right.ptr_key());
    let subset = *subset_memo
        .entry(key)
        .or_insert_with(|| right.is_subset(left));
    if subset {
        right.clone()
    } else {
        left.intersection(right)
    }
}

#[derive(Clone, Copy)]
struct PredEdge<'a> {
    from: usize,
    weight: &'a Weight,
}

#[derive(Clone)]
struct GuardedFinalWeight {
    weight: Weight,
    /// `Some(weight)` is a proof that `weight` is a subset of the cloned,
    /// interned edge weight stored here. Owning the guard prevents pointer
    /// reuse / ABA issues while the proof exists and still lets finality
    /// propagation skip idempotent intersections along chains of equal
    /// parser-template edge weights.
    subset_of: Option<Weight>,
}

impl GuardedFinalWeight {
    fn initial(weight: Weight) -> Option<Self> {
        (!weight.is_empty()).then_some(Self {
            weight,
            subset_of: None,
        })
    }

    fn is_guarded_by(&self, edge_weight: &Weight) -> bool {
        self.subset_of
            .as_ref()
            .is_some_and(|guard| Arc::ptr_eq(&guard.0, &edge_weight.0))
    }

    fn intersection_with_edge(&self, edge_weight: &Weight) -> Option<Self> {
        if self.weight.is_empty() || edge_weight.is_empty() {
            return None;
        }

        if self.is_guarded_by(edge_weight) {
            return Some(self.clone());
        }

        if edge_weight.is_full() {
            return Some(Self {
                weight: self.weight.clone(),
                subset_of: Some(edge_weight.clone()),
            });
        }

        if self.weight.is_full() {
            return Some(Self {
                weight: edge_weight.clone(),
                subset_of: Some(edge_weight.clone()),
            });
        }

        let weight = self.weight.intersection(edge_weight);
        (!weight.is_empty()).then_some(Self {
            weight,
            subset_of: Some(edge_weight.clone()),
        })
    }
}

fn merge_guarded_final_weight(
    entry: &mut Option<GuardedFinalWeight>,
    add: GuardedFinalWeight,
) -> bool {
    if add.weight.is_empty() {
        return false;
    }

    match entry {
        Some(existing) => {
            let updated = existing.weight.union(&add.weight);
            if updated != existing.weight {
                let subset_of = if existing
                    .subset_of
                    .as_ref()
                    .zip(add.subset_of.as_ref())
                    .is_some_and(|(left, right)| Arc::ptr_eq(&left.0, &right.0))
                {
                    existing.subset_of.clone()
                } else {
                    None
                };
                existing.weight = updated;
                existing.subset_of = subset_of;
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

fn union_guarded_pending(
    pending: SmallVec<[GuardedFinalWeight; 4]>,
) -> Option<GuardedFinalWeight> {
    match pending.len() {
        0 => None,
        1 => pending.into_iter().next(),
        _ => {
            let shared_guard = pending[0].subset_of.clone().filter(|first_guard| {
                pending[1..].iter().all(|entry| {
                    entry
                        .subset_of
                        .as_ref()
                        .is_some_and(|guard| Arc::ptr_eq(&first_guard.0, &guard.0))
                })
            });
            let weight = Weight::union_all(pending.iter().map(|entry| &entry.weight));
            (!weight.is_empty()).then_some(GuardedFinalWeight {
                weight,
                subset_of: shared_guard,
            })
        }
    }
}

fn enqueue_cancellation_task(
    worklist: &mut VecDeque<CancellationTask>,
    queued: &mut QueuedQueries,
    state: u32,
    source_state: u32,
    positive_label: i32,
) {
    let task = (state, source_state, positive_label);
    if queued[state as usize].insert((source_state, positive_label)) {
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
    queued: &mut QueuedQueries,
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
    queued: &mut QueuedQueries,
    derived_epsilons: &DerivedEpsilons,
    foreign_derived: Option<&DerivedEpsilons>,
) {
    let local = derived_epsilons[current_state as usize].as_ref();
    let foreign = foreign_derived.and_then(|fd| fd[current_state as usize].as_ref());

    let propagate = |target_state: u32,
                     epsilon_weight: &Weight,
                     query_weights: &mut QueryWeights,
                     worklist: &mut VecDeque<CancellationTask>,
                     queued: &mut QueuedQueries| {
        let propagated = intersect_with_single_weight_hint(
            query_weight_to_current,
            query_single,
            epsilon_weight,
        );
        if propagated.is_empty() {
            return;
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
    };

    if let Some(local_map) = local {
        for (&target_state, epsilon_weight) in local_map {
            propagate(target_state, epsilon_weight, query_weights, worklist, queued);
        }
    }
    if let Some(foreign_map) = foreign {
        for (&target_state, epsilon_weight) in foreign_map {
            // Skip if local already has a stronger entry for this target
            // (merging handled below in queue_query_weight's dedup anyway).
            propagate(target_state, epsilon_weight, query_weights, worklist, queued);
        }
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
    queued: &mut QueuedQueries,
    derived_epsilons: &mut DerivedEpsilons,
    subset_memo: &mut SubsetMemo,
) {
    let new_derived_weight = intersect_with_single_weight_hint(
        query_weight_to_current,
        query_single,
        edge_weight,
    );
    if new_derived_weight.is_empty() {
        return;
    }

    {
        let derived_weight = derived_epsilons[source_state as usize]
            .get_or_insert_with(FxHashMap::default)
            .entry(target_state)
            .or_insert_with(Weight::empty);
        if !merge_weight(derived_weight, new_derived_weight) {
            return;
        }
    }

    let Some(derived_weight) = derived_epsilons[source_state as usize]
        .as_ref()
        .and_then(|targets| targets.get(&target_state))
    else {
        return;
    };

    let Some(existing_queries) = query_weights[source_state as usize].as_ref() else {
        return;
    };

    let mut propagated_updates = Vec::with_capacity(existing_queries.len());

    for ((upstream_source_state, upstream_label), upstream_weight) in existing_queries {
        let propagated = intersect_or_clone_right_if_subset_cached(
            upstream_weight,
            derived_weight,
            subset_memo,
        );
        if propagated.is_empty() {
            continue;
        }
        propagated_updates.push((*upstream_source_state, *upstream_label, propagated));
    }

    for (upstream_source_state, upstream_label, propagated) in propagated_updates {
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
    compute_cancellations_range_inner(nwa, range, None)
}

fn compute_cancellations_range_inner(
    nwa: &NWA,
    range: std::ops::Range<u32>,
    foreign_derived: Option<&DerivedEpsilons>,
) -> Vec<(u32, u32, Weight)> {
    let state_count = nwa.states().len() as u32;
    if state_count == 0 {
        return Vec::new();
    }

    let mut query_weights: QueryWeights = vec![None; state_count as usize];
    let mut worklist = VecDeque::<CancellationTask>::new();
    let mut queued: QueuedQueries = vec![SmallQueuedQueries::Empty; state_count as usize];
    let mut derived_epsilons: DerivedEpsilons = vec![None; state_count as usize];
    let mut subset_memo = SubsetMemo::default();

    for source_state in range {
        if source_state >= state_count {
            continue;
        }

        for (&label, targets) in nwa.states()[source_state as usize].transitions.range(..0) {
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
        queued[current_state as usize].remove(&(source_state, positive_label));
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
            foreign_derived,
        );

        if let Some(positive_targets) = nwa.states()[current_state as usize]
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
                        &mut subset_memo,
                    );
                }
            }
        }

        if let Some(default_targets) = nwa.states()[current_state as usize]
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
                        &mut subset_memo,
                    );
                }
            }
        }

        for (target_state, epsilon_weight) in &nwa.states()[current_state as usize].epsilons {
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

fn is_live_finality_edge(target_state: u32, weight: &Weight, state_count: usize) -> bool {
    (target_state as usize) < state_count && !weight.is_empty()
}

fn build_finality_preds_and_outdegree<'a>(nwa: &'a NWA) -> (Vec<Vec<PredEdge<'a>>>, Vec<usize>) {
    let state_count = nwa.states().len();
    let mut preds = vec![Vec::<PredEdge<'a>>::new(); state_count];
    let mut outdegree = vec![0usize; state_count];

    for (from_state, state) in nwa.states().iter().enumerate() {
        for (target_state, weight) in &state.epsilons {
            if !is_live_finality_edge(*target_state, weight, state_count) {
                continue;
            }
            preds[*target_state as usize].push(PredEdge {
                from: from_state,
                weight,
            });
            outdegree[from_state] += 1;
        }

        for (&label, targets) in &state.transitions {
            if label != DEFAULT_LABEL && !is_negative_label(label) {
                continue;
            }

            for (target_state, weight) in targets {
                if !is_live_finality_edge(*target_state, weight, state_count) {
                    continue;
                }
                preds[*target_state as usize].push(PredEdge {
                    from: from_state,
                    weight,
                });
                outdegree[from_state] += 1;
            }
        }
    }

    (preds, outdegree)
}

fn build_finality_reverse_topo_order(
    preds: &[Vec<PredEdge<'_>>],
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

fn collect_initial_final_weights(nwa: &NWA) -> Vec<Option<GuardedFinalWeight>> {
    nwa.states()
        .iter()
        .map(|state| {
            state
                .final_weight
                .clone()
                .and_then(GuardedFinalWeight::initial)
        })
        .collect()
}

fn write_final_weights(nwa: &mut NWA, reachable_final_weights: Vec<Option<GuardedFinalWeight>>) {
    for (state_id, final_weight) in reachable_final_weights.into_iter().enumerate() {
        nwa.states_mut()[state_id].final_weight = final_weight
            .map(|guarded| guarded.weight)
            .filter(|weight| !weight.is_empty());
    }
}

fn propagate_final_weights_to_predecessors(
    preds: &[Vec<PredEdge<'_>>],
    reachable_final_weights: &mut [Option<GuardedFinalWeight>],
    state_id: usize,
    mut on_change: impl FnMut(usize),
) {
    let Some(reachable_final) = reachable_final_weights[state_id].clone() else {
        return;
    };

    for edge in &preds[state_id] {
        let Some(propagated) = reachable_final.intersection_with_edge(edge.weight) else {
            continue;
        };
        if merge_guarded_final_weight(&mut reachable_final_weights[edge.from], propagated) {
            on_change(edge.from);
        }
    }
}

fn apply_finality_fixpoint_worklist(
    nwa: &NWA,
    preds: &[Vec<PredEdge<'_>>],
    reachable_final_weights: &mut [Option<GuardedFinalWeight>],
) {
    let n = nwa.states().len();
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
    preds: &[Vec<PredEdge<'_>>],
    reachable_final_weights: &mut [Option<GuardedFinalWeight>],
    reverse_topo_order: &[usize],
) {
    let mut pending_by_state: Vec<SmallVec<[GuardedFinalWeight; 4]>> = (0..preds.len())
        .map(|_| SmallVec::new())
        .collect();

    for (state_id, final_weight) in reachable_final_weights.iter_mut().enumerate() {
        if let Some(final_weight) = final_weight.take() {
            pending_by_state[state_id].push(final_weight);
        }
    }

    for &state_id in reverse_topo_order {
        let Some(reachable_final) = union_guarded_pending(std::mem::take(&mut pending_by_state[state_id])) else {
            continue;
        };

        for edge in &preds[state_id] {
            let Some(propagated) = reachable_final.intersection_with_edge(edge.weight) else {
                continue;
            };
            pending_by_state[edge.from].push(propagated);
        }

        reachable_final_weights[state_id] = Some(reachable_final);
    }
}

pub(crate) fn apply_finality_fixpoint(nwa: &mut NWA) {
    let n = nwa.states().len();
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
    for state in  nwa.states_mut() {
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

fn is_terminal_shape_candidate(state: &NWAState) -> bool {
    !has_non_default_transitions(state)
        && state.epsilons.is_empty()
        && has_live_final_weight(state)
}

fn default_targets_are_terminal_and_redundant(state: &NWAState, terminal_states: &[bool]) -> bool {
    let Some(final_weight) = state.final_weight.as_ref().filter(|weight| !weight.is_empty()) else {
        return false;
    };

    match state.transitions.get(&DEFAULT_LABEL) {
        None => true,
        Some(targets) => targets.iter().all(|(target, _)| {
            terminal_states
                .get(*target as usize)
                .copied()
                .unwrap_or(false)
        }) && targets.iter().all(|(_, edge_weight)| edge_weight.is_subset(final_weight)),
    }
}

fn grow_terminal_state_set(nwa: &NWA, terminal_states: &mut [bool]) {
    loop {
        let mut changed = false;
        for (state_id, state) in nwa.states().iter().enumerate() {
            if terminal_states[state_id] || !is_terminal_shape_candidate(state) {
                continue;
            }

            if default_targets_are_terminal_and_redundant(state, terminal_states) {
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
    for state in  nwa.states_mut() {
        let final_weight = state.final_weight.clone();
        if let Some(targets) = state.transitions.get_mut(&DEFAULT_LABEL) {
            targets.retain(|(target, edge_weight)| {
                if !terminal_states[*target as usize] {
                    return true;
                }
                let Some(final_weight) = final_weight.as_ref() else {
                    return true;
                };
                !edge_weight.is_subset(final_weight)
            });
        }
        state.transitions.retain(|_, targets| !targets.is_empty());
    }
}

pub(crate) fn remove_redundant_default_transitions(nwa: &mut NWA) {
    let mut terminal_states: Vec<bool> = nwa
        .states()
        .iter()
        .map(|state| is_terminal_shape_candidate(state) && !state.transitions.contains_key(&DEFAULT_LABEL))
        .collect();

    grow_terminal_state_set(nwa, &mut terminal_states);
    prune_terminal_default_targets(nwa, &terminal_states);
}

pub(crate) fn resolve_negative_codes_in_nwa(nwa: &mut NWA) {
    if !nwa.states().is_empty() {
        apply_cancellations_range(nwa, 0..nwa.states().len() as u32);
    }
    apply_finality_fixpoint(nwa);
    remove_negative_transitions(nwa);
    remove_redundant_default_transitions(nwa);
}
