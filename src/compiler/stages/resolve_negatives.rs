//! Resolve negative parser-state labels in weighted NWAs.

use std::collections::VecDeque;
use std::sync::Arc;

use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::automata::weighted::nwa::{NWA, NWAState};
use crate::compiler::glr::labels::{DEFAULT_LABEL, is_negative_label, negative_to_positive_label};
use crate::ds::weight::{ScopedWeightOpCache, Weight};

type QueryKey = (u32, i32);
type CancellationTask = (u32, u32, i32);
type QueryWeights = Vec<SmallQueryWeights>;
type DerivedEpsilons = Vec<Option<FxHashMap<u32, Weight>>>;
type SubsetMemo = FxHashMap<(usize, usize), bool>;

/// Wave scheduling pays two parallel-iterator setup costs per DAG layer.  This
/// many predecessor edges per Rayon worker is enough to amortize that work on
/// the small and large BFCL catalogs while leaving genuinely small parsers on
/// the lower-overhead serial solver.
const MIN_PARALLEL_FINALITY_EDGES_PER_WORKER: usize = 512;

#[derive(Clone, Default)]
enum SmallQueryWeights {
    #[default]
    Empty,
    One(QueryKey, Weight),
    Many(FxHashMap<QueryKey, Weight>),
}

impl SmallQueryWeights {
    fn merge(&mut self, query_key: QueryKey, add: Weight) -> bool {
        match self {
            Self::Empty => {
                if add.is_empty() {
                    return false;
                }
                *self = Self::One(query_key, add);
                true
            }
            Self::One(existing_key, existing_weight) if *existing_key == query_key => {
                merge_weight(existing_weight, add)
            }
            Self::One(existing_key, existing_weight) => {
                if add.is_empty() {
                    return false;
                }
                let previous_key = *existing_key;
                let previous_weight = existing_weight.clone();
                let mut entries = FxHashMap::with_capacity_and_hasher(4, Default::default());
                entries.insert(previous_key, previous_weight);
                entries.insert(query_key, add);
                *self = Self::Many(entries);
                true
            }
            Self::Many(entries) => {
                let entry = entries.entry(query_key).or_insert_with(Weight::empty);
                merge_weight(entry, add)
            }
        }
    }

    fn get(&self, query_key: &QueryKey) -> Option<&Weight> {
        match self {
            Self::Empty => None,
            Self::One(existing_key, weight) => (existing_key == query_key).then_some(weight),
            Self::Many(entries) => entries.get(query_key),
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::One(_, _) => 1,
            Self::Many(entries) => entries.len(),
        }
    }

    fn for_each(&self, mut f: impl FnMut(QueryKey, &Weight)) {
        match self {
            Self::Empty => {}
            Self::One(query_key, weight) => f(*query_key, weight),
            Self::Many(entries) => {
                for (&query_key, weight) in entries {
                    f(query_key, weight);
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

        let weight = self.weight.intersection_uncached(edge_weight);
        (!weight.is_empty()).then_some(Self {
            weight,
            subset_of: Some(edge_weight.clone()),
        })
    }

    /// Exact cached counterpart for serial finality propagation. The guard keeps
    /// the common nested-template case allocation-free; other repeated pairs use
    /// the invocation-local cache rather than recomputing range intersections.
    fn intersection_with_edge_cached(
        &self,
        edge_weight: &Weight,
        weight_ops: &mut ScopedWeightOpCache,
    ) -> Option<Self> {
        if self.weight.is_empty() || edge_weight.is_empty() {
            return None;
        }

        if self.is_guarded_by(edge_weight) {
            return Some(self.clone());
        }

        let weight = weight_ops.intersection(&self.weight, edge_weight);
        (!weight.is_empty()).then_some(Self {
            weight,
            subset_of: Some(edge_weight.clone()),
        })
    }
}

fn merge_guarded_final_weight(
    entry: &mut Option<GuardedFinalWeight>,
    add: GuardedFinalWeight,
    weight_ops: &mut ScopedWeightOpCache,
) -> bool {
    if add.weight.is_empty() {
        return false;
    }

    match entry {
        Some(existing) => {
            let updated = weight_ops.union(&existing.weight, &add.weight);
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
    weight_ops: &mut ScopedWeightOpCache,
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
            let weight = weight_ops.union_all(pending.iter().map(|entry| &entry.weight));
            (!weight.is_empty()).then_some(GuardedFinalWeight {
                weight,
                subset_of: shared_guard,
            })
        }
    }
}

fn record_query_weight(
    query_weights: &mut QueryWeights,
    state: u32,
    query_key: QueryKey,
    add: Weight,
) -> bool {
    query_weights[state as usize].merge(query_key, add)
}

fn queue_query_weight(
    query_weights: &mut QueryWeights,
    worklist: &mut VecDeque<CancellationTask>,
    state: u32,
    source_state: u32,
    positive_label: i32,
    add: Weight,
) {
    if record_query_weight(query_weights, state, (source_state, positive_label), add) {
        worklist.push_back((state, source_state, positive_label));
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
    derived_epsilons: &DerivedEpsilons,
    foreign_derived: Option<&DerivedEpsilons>,
) {
    let local = derived_epsilons[current_state as usize].as_ref();
    let foreign = foreign_derived.and_then(|fd| fd[current_state as usize].as_ref());

    let propagate = |target_state: u32,
                     epsilon_weight: &Weight,
                     query_weights: &mut QueryWeights,
                     worklist: &mut VecDeque<CancellationTask>| {
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
            target_state,
            source_state,
            positive_label,
            propagated,
        );
    };

    if let Some(local_map) = local {
        for (&target_state, epsilon_weight) in local_map {
            propagate(target_state, epsilon_weight, query_weights, worklist);
        }
    }
    if let Some(foreign_map) = foreign {
        for (&target_state, epsilon_weight) in foreign_map {
            // Skip if local already has a stronger entry for this target
            // (merging handled below in queue_query_weight's dedup anyway).
            propagate(target_state, epsilon_weight, query_weights, worklist);
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

    let existing_queries = &query_weights[source_state as usize];
    if existing_queries.is_empty() {
        return;
    }

    let mut propagated_updates = Vec::with_capacity(existing_queries.len());

    existing_queries.for_each(|(upstream_source_state, upstream_label), upstream_weight| {
        let propagated = intersect_or_clone_right_if_subset_cached(
            upstream_weight,
            derived_weight,
            subset_memo,
        );
        if propagated.is_empty() {
            return;
        }
        propagated_updates.push((upstream_source_state, upstream_label, propagated));
    });

    for (upstream_source_state, upstream_label, propagated) in propagated_updates {
        queue_query_weight(
            query_weights,
            worklist,
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

    let mut query_weights: QueryWeights = vec![SmallQueryWeights::Empty; state_count as usize];
    let mut worklist = VecDeque::<CancellationTask>::new();
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
                    *target_state,
                    source_state,
                    positive_label,
                    weight.clone(),
                );
            }
        }
    }

    while let Some((current_state, source_state, positive_label)) = worklist.pop_front() {
        let Some(query_weight_to_current) = query_weights[current_state as usize]
            .get(&(source_state, positive_label))
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
    weight_ops: &mut ScopedWeightOpCache,
    mut on_change: impl FnMut(usize),
) {
    let Some(reachable_final) = reachable_final_weights[state_id].clone() else {
        return;
    };

    for edge in &preds[state_id] {
        let Some(propagated) = reachable_final.intersection_with_edge_cached(edge.weight, weight_ops) else {
            continue;
        };
        if merge_guarded_final_weight(&mut reachable_final_weights[edge.from], propagated, weight_ops) {
            on_change(edge.from);
        }
    }
}

fn apply_finality_fixpoint_worklist(
    nwa: &NWA,
    preds: &[Vec<PredEdge<'_>>],
    reachable_final_weights: &mut [Option<GuardedFinalWeight>],
    weight_ops: &mut ScopedWeightOpCache,
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
            weight_ops,
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
    weight_ops: &mut ScopedWeightOpCache,
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
        let Some(reachable_final) =
            union_guarded_pending(std::mem::take(&mut pending_by_state[state_id]), weight_ops)
        else {
            continue;
        };

        for edge in &preds[state_id] {
            let Some(propagated) = reachable_final.intersection_with_edge_cached(edge.weight, weight_ops) else {
                continue;
            };
            pending_by_state[edge.from].push(propagated);
        }

        reachable_final_weights[state_id] = Some(reachable_final);
    }
}

/// Group a reverse topological traversal into dependency-free waves.  Every
/// edge goes from a later wave to an earlier wave, so all states in one wave
/// can propagate their already-complete final weights concurrently.
fn finality_reverse_topo_waves(
    preds: &[Vec<PredEdge<'_>>],
    reverse_topo_order: &[usize],
) -> Vec<Vec<usize>> {
    let mut wave_by_state = vec![0usize; preds.len()];
    let mut max_wave = 0usize;
    for &state_id in reverse_topo_order {
        let predecessor_wave = wave_by_state[state_id] + 1;
        for edge in &preds[state_id] {
            let wave = &mut wave_by_state[edge.from];
            if *wave < predecessor_wave {
                *wave = predecessor_wave;
                max_wave = max_wave.max(predecessor_wave);
            }
        }
    }

    let mut waves = (0..=max_wave).map(|_| Vec::new()).collect::<Vec<_>>();
    for &state_id in reverse_topo_order {
        waves[wave_by_state[state_id]].push(state_id);
    }
    waves
}

/// A wave normally has enough independent states to saturate the Rayon pool.
/// However, a single state can still own most of the wave's predecessor edges.
/// First compute each state's final weight, then split only states whose edge
/// count exceeds the wave mean into contiguous edge chunks.  Results are merged
/// in state/chunk/edge order, exactly matching the unsplit propagation order.
fn apply_finality_fixpoint_acyclic_parallel_waves_chunked(
    preds: &[Vec<PredEdge<'_>>],
    reachable_final_weights: &mut [Option<GuardedFinalWeight>],
    reverse_topo_order: &[usize],
) {
    let profile_detail = std::env::var("GLRMASK_PROFILE_FINALITY_WAVES_DETAIL")
        .map(|value| value == "1")
        .unwrap_or(false);
    let worker_count = rayon::current_num_threads().max(1);
    let mut pending_by_state: Vec<SmallVec<[GuardedFinalWeight; 4]>> =
        (0..preds.len()).map(|_| SmallVec::new()).collect();

    for (state_id, final_weight) in reachable_final_weights.iter_mut().enumerate() {
        if let Some(final_weight) = final_weight.take() {
            pending_by_state[state_id].push(final_weight);
        }
    }

    for (wave_index, wave) in finality_reverse_topo_waves(preds, reverse_topo_order)
        .into_iter()
        .enumerate()
    {
        let state_jobs = wave
            .into_iter()
            .map(|state_id| (state_id, std::mem::take(&mut pending_by_state[state_id])))
            .collect::<Vec<_>>();
        let state_results = state_jobs
            .into_par_iter()
            .map_init(ScopedWeightOpCache::default, |weight_ops, (state_id, pending)| {
                (state_id, union_guarded_pending(pending, weight_ops))
            })
            .collect::<Vec<_>>();

        let mut live_states = 0usize;
        let mut total_live_edges = 0usize;
        for (state_id, reachable_final) in &state_results {
            if reachable_final.is_some() {
                live_states += 1;
                total_live_edges += preds[*state_id].len();
            }
        }
        if live_states == 0 {
            continue;
        }
        let mean_live_edges = total_live_edges.div_ceil(live_states).max(1);

        // (state-result index, inclusive start, exclusive end).  The vector is
        // built in state order and each state's chunks are contiguous, so the
        // later serial merge has the same pending insertion order as one
        // unsplit state job.
        let mut tasks = Vec::<(usize, usize, usize)>::new();
        let mut split_states = 0usize;
        let mut max_chunk_edges = 0usize;
        for (result_index, (state_id, reachable_final)) in state_results.iter().enumerate() {
            if reachable_final.is_none() {
                continue;
            }
            let edge_count = preds[*state_id].len();
            if edge_count == 0 {
                continue;
            }
            let chunks = edge_count.div_ceil(mean_live_edges).min(worker_count).max(1);
            split_states += usize::from(chunks > 1);
            let chunk_len = edge_count.div_ceil(chunks);
            for start in (0..edge_count).step_by(chunk_len) {
                let end = (start + chunk_len).min(edge_count);
                max_chunk_edges = max_chunk_edges.max(end - start);
                tasks.push((result_index, start, end));
            }
        }

        let propagation_results = tasks
            .into_par_iter()
            .map(|(result_index, start, end)| {
                let (state_id, reachable_final) = &state_results[result_index];
                let reachable_final = reachable_final
                    .as_ref()
                    .expect("propagation task only exists for live finality state");
                let mut propagated = SmallVec::<[(usize, GuardedFinalWeight); 4]>::new();
                for edge in &preds[*state_id][start..end] {
                    if let Some(weight) = reachable_final.intersection_with_edge(edge.weight) {
                        propagated.push((edge.from, weight));
                    }
                }
                propagated
            })
            .collect::<Vec<_>>();
        let state_result_count = state_results.len();
        let propagation_task_count = propagation_results.len();

        for (state_id, reachable_final) in state_results {
            if let Some(reachable_final) = reachable_final {
                reachable_final_weights[state_id] = Some(reachable_final);
            }
        }
        let mut propagated_edges = 0usize;
        for propagated in propagation_results {
            propagated_edges += propagated.len();
            for (predecessor, weight) in propagated {
                pending_by_state[predecessor].push(weight);
            }
        }

        if profile_detail {
            eprintln!(
                "[glrmask/profile][finality_chunked_wave] wave={} states={} live_states={} total_live_edges={} mean_live_edges={} split_states={} tasks={} max_chunk_edges={} propagated_edges={}",
                wave_index,
                state_result_count,
                live_states,
                total_live_edges,
                mean_live_edges,
                split_states,
                propagation_task_count,
                max_chunk_edges,
                propagated_edges,
            );
        }
    }
}

pub(crate) fn apply_finality_fixpoint(nwa: &mut NWA) {
    let n = nwa.states().len();
    if n == 0 {
        return;
    }
    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
    let preds_started_at = profile_enabled.then(std::time::Instant::now);
    let (preds, outdegree) = build_finality_preds_and_outdegree(nwa);
    let finality_edge_count = preds.iter().map(Vec::len).sum::<usize>();
    let preds_ms = preds_started_at.map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
    let edge_profile = profile_enabled.then(|| {
        let mut unique_weight_ptrs = FxHashSet::default();
        let mut full_weights = 0usize;
        let mut single_entry_weights = 0usize;
        let mut wide_weight_ptrs = FxHashSet::default();
        let mut wide_edge_count = 0usize;
        let mut total_ranges = 0usize;
        let mut edge_count = 0usize;
        for edges in &preds {
            for edge in edges {
                edge_count += 1;
                unique_weight_ptrs.insert(edge.weight.ptr_key());
                full_weights += usize::from(edge.weight.is_full());
                single_entry_weights += usize::from(edge.weight.single_compact_entry_parts().is_some());
                let range_count = edge.weight.num_ranges();
                total_ranges += range_count;
                if range_count >= 32 {
                    wide_edge_count += 1;
                    wide_weight_ptrs.insert(edge.weight.ptr_key());
                }
            }
        }
        (
            edge_count,
            unique_weight_ptrs.len(),
            full_weights,
            single_entry_weights,
            total_ranges,
            wide_edge_count,
            wide_weight_ptrs.len(),
        )
    });
    let topo_started_at = profile_enabled.then(std::time::Instant::now);
    let reverse_topo_order = build_finality_reverse_topo_order(&preds, outdegree);
    let topo_ms = topo_started_at.map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
    let acyclic = reverse_topo_order.is_some();
    if profile_enabled {
        if let Some(reverse_topo_order) = reverse_topo_order.as_deref() {
            let mut layer_by_state = vec![0usize; n];
            let mut max_layer = 0usize;
            for &state_id in reverse_topo_order {
                let next_layer = layer_by_state[state_id] + 1;
                for edge in &preds[state_id] {
                    let layer = &mut layer_by_state[edge.from];
                    if *layer < next_layer {
                        *layer = next_layer;
                        max_layer = max_layer.max(next_layer);
                    }
                }
            }
            let mut counts = vec![0usize; max_layer + 1];
            for layer in layer_by_state {
                counts[layer] += 1;
            }
            eprintln!(
                "[glrmask/profile][finality_layers] layers={} max_width={} singleton_layers={} first_widths={:?}",
                counts.len(),
                counts.iter().copied().max().unwrap_or(0),
                counts.iter().filter(|&&count| count == 1).count(),
                &counts[..counts.len().min(16)],
            );
        }
    }
    let initial_started_at = profile_enabled.then(std::time::Instant::now);
    let mut reachable_final_weights = collect_initial_final_weights(nwa);
    let initial_ms = initial_started_at.map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
    let mut weight_ops = ScopedWeightOpCache::default();
    let rayon_workers = rayon::current_num_threads();
    // The serial acyclic solver reuses an invocation-local weight-operation
    // cache. On the large parser NWAs this consistently beats the chunked Rayon
    // wave solver, whose uncached intersections dominate despite parallelism.
    // Keep the parallel path available for targeted experiments.
    let force_parallel_finality = std::env::var("GLRMASK_FORCE_PARALLEL_FINALITY")
        .map(|value| value == "1")
        .unwrap_or(false);
    let use_chunked_parallel_waves = force_parallel_finality
        && acyclic
        && rayon_workers > 1
        && finality_edge_count >= MIN_PARALLEL_FINALITY_EDGES_PER_WORKER * rayon_workers;

    let solve_started_at = profile_enabled.then(std::time::Instant::now);
    if let Some(reverse_topo_order) = reverse_topo_order.as_deref() {
        if use_chunked_parallel_waves {
            apply_finality_fixpoint_acyclic_parallel_waves_chunked(
                &preds,
                &mut reachable_final_weights,
                reverse_topo_order,
            );
        } else {
            apply_finality_fixpoint_acyclic(
                &preds,
                &mut reachable_final_weights,
                reverse_topo_order,
                &mut weight_ops,
            );
        }
    } else {
        apply_finality_fixpoint_worklist(
            nwa,
            &preds,
            &mut reachable_final_weights,
            &mut weight_ops,
        );
    }
    let solve_ms = solve_started_at.map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);

    let write_started_at = profile_enabled.then(std::time::Instant::now);
    write_final_weights(nwa, reachable_final_weights);
    if let (
        Some(preds_ms),
        Some((
            edge_count,
            unique_edge_weights,
            full_weights,
            single_entry_weights,
            total_ranges,
            wide_edge_count,
            unique_wide_edge_weights,
        )),
        Some(topo_ms),
        Some(initial_ms),
        Some(solve_ms),
        Some(write_started_at),
    ) = (
        preds_ms,
        edge_profile,
        topo_ms,
        initial_ms,
        solve_ms,
        write_started_at,
    )
    {
        eprintln!(
            "[glrmask/profile][finality_fixpoint] states={} edges={} unique_edge_weights={} full_weights={} single_entry_weights={} total_weight_ranges={} wide_edges={} unique_wide_edge_weights={} acyclic={} rayon_workers={} chunked_parallel_waves={} preds_ms={:.3} topo_ms={:.3} initial_ms={:.3} solve_ms={:.3} write_ms={:.3}",
            n,
            edge_count,
            unique_edge_weights,
            full_weights,
            single_entry_weights,
            total_ranges,
            wide_edge_count,
            unique_wide_edge_weights,
            acyclic,
            rayon_workers,
            use_chunked_parallel_waves,
            preds_ms,
            topo_ms,
            initial_ms,
            solve_ms,
            write_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
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

#[cfg(test)]
fn grow_terminal_state_set_reference(nwa: &NWA, terminal_states: &mut [bool]) {
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

/// Compute the same monotone terminal-state closure as the scan-based version,
/// but visit a candidate only when one of its default successors becomes
/// terminal. This is linear in eligible default edges rather than in
/// (states × fixed-point rounds).
fn grow_terminal_state_set(nwa: &NWA, terminal_states: &mut [bool]) {
    let state_count = nwa.states().len();
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); state_count];
    let mut remaining_nonterminal_targets = vec![usize::MAX; state_count];
    let mut worklist = VecDeque::new();

    for (state_id, &is_terminal) in terminal_states.iter().enumerate() {
        if is_terminal {
            worklist.push_back(state_id);
        }
    }

    for (state_id, state) in nwa.states().iter().enumerate() {
        if terminal_states[state_id] || !is_terminal_shape_candidate(state) {
            continue;
        }
        let Some(final_weight) = state.final_weight.as_ref().filter(|weight| !weight.is_empty()) else {
            continue;
        };
        let Some(targets) = state.transitions.get(&DEFAULT_LABEL) else {
            terminal_states[state_id] = true;
            worklist.push_back(state_id);
            continue;
        };
        if targets.iter().any(|(_, edge_weight)| !edge_weight.is_subset(final_weight)) {
            continue;
        }

        let mut remaining = 0usize;
        for (target, _) in targets {
            let target = *target as usize;
            // Keep the scan implementation's behavior for malformed edges:
            // an out-of-range target is never terminal, so this state cannot
            // enter the terminal closure.
            if target >= state_count {
                remaining += 1;
                continue;
            }
            if !terminal_states[target] {
                dependents[target].push(state_id);
                remaining += 1;
            }
        }
        remaining_nonterminal_targets[state_id] = remaining;
        if remaining == 0 {
            terminal_states[state_id] = true;
            worklist.push_back(state_id);
        }
    }

    while let Some(terminal_state) = worklist.pop_front() {
        for &dependent in &dependents[terminal_state] {
            if terminal_states[dependent] {
                continue;
            }
            let remaining = &mut remaining_nonterminal_targets[dependent];
            debug_assert_ne!(*remaining, usize::MAX);
            debug_assert!(*remaining > 0);
            *remaining -= 1;
            if *remaining == 0 {
                terminal_states[dependent] = true;
                worklist.push_back(dependent);
            }
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
    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
    let cancellation_started_at = profile_enabled.then(std::time::Instant::now);
    if !nwa.states().is_empty() {
        apply_cancellations_range(nwa, 0..nwa.states().len() as u32);
    }
    let cancellation_ms = cancellation_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
    let finality_started_at = profile_enabled.then(std::time::Instant::now);
    apply_finality_fixpoint(nwa);
    let finality_ms = finality_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
    let remove_negative_started_at = profile_enabled.then(std::time::Instant::now);
    remove_negative_transitions(nwa);
    let remove_negative_ms = remove_negative_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
    let prune_defaults_started_at = profile_enabled.then(std::time::Instant::now);
    remove_redundant_default_transitions(nwa);
    if let (Some(cancellation_ms), Some(finality_ms), Some(remove_negative_ms), Some(prune_defaults_started_at)) = (
        cancellation_ms,
        finality_ms,
        remove_negative_ms,
        prune_defaults_started_at,
    ) {
        eprintln!(
            "[glrmask/profile][resolve_negatives] states={} cancellation_ms={:.3} finality_ms={:.3} remove_negative_ms={:.3} prune_defaults_ms={:.3}",
            nwa.states().len(),
            cancellation_ms,
            finality_ms,
            remove_negative_ms,
            prune_defaults_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
}


#[cfg(test)]
mod terminal_default_tests {
    use range_set_blaze::RangeSetBlaze;

    use super::*;

    fn weight(tokens: std::ops::RangeInclusive<u32>) -> Weight {
        Weight::from_token_set_for_tsid(0, RangeSetBlaze::from_iter([tokens]))
    }

    #[test]
    fn terminal_default_worklist_matches_scan_fixed_point() {
        let mut nwa = NWA::new(1, 7);
        for _ in 0..5 {
            nwa.add_state();
        }
        let all = weight(0..=7);
        // 4 is terminal initially; 3 -> 4 -> 2 -> 1 become terminal in a
        // dependency chain. State 0 has a non-default edge and must remain out.
        for state in 1..=4 {
            nwa.set_final_weight(state, all.clone());
        }
        nwa.add_transition(3, DEFAULT_LABEL, 4, all.clone());
        nwa.add_transition(2, DEFAULT_LABEL, 3, all.clone());
        nwa.add_transition(1, DEFAULT_LABEL, 2, all.clone());
        nwa.add_transition(0, 0, 1, all.clone());
        nwa.set_final_weight(0, all);

        let initial = nwa
            .states()
            .iter()
            .map(|state| is_terminal_shape_candidate(state) && !state.transitions.contains_key(&DEFAULT_LABEL))
            .collect::<Vec<_>>();
        let mut expected = initial.clone();
        let mut actual = initial;
        grow_terminal_state_set_reference(&nwa, &mut expected);
        grow_terminal_state_set(&nwa, &mut actual);
        assert_eq!(actual, expected);
        assert_eq!(actual, vec![false, true, true, true, true]);
    }

    #[test]
    fn terminal_default_worklist_rejects_non_subset_default_edge() {
        let mut nwa = NWA::new(1, 7);
        for _ in 0..2 {
            nwa.add_state();
        }
        nwa.set_final_weight(1, weight(0..=7));
        nwa.set_final_weight(0, weight(0..=3));
        nwa.add_transition(0, DEFAULT_LABEL, 1, weight(0..=7));
        let mut terminal_states = nwa
            .states()
            .iter()
            .map(|state| is_terminal_shape_candidate(state) && !state.transitions.contains_key(&DEFAULT_LABEL))
            .collect::<Vec<_>>();
        grow_terminal_state_set(&nwa, &mut terminal_states);
        assert_eq!(terminal_states, vec![false, true]);
    }

    #[test]
    fn chunked_parallel_finality_waves_match_serial_dag_propagation() {
        let mut nwa = NWA::new(1, 7);
        for _ in 0..6 {
            nwa.add_state();
        }
        let all = weight(0..=7);
        nwa.set_final_weight(4, weight(0..=3));
        nwa.set_final_weight(5, weight(4..=7));
        nwa.add_epsilon(3, 4, all.clone());
        nwa.add_transition(3, DEFAULT_LABEL, 5, all.clone());
        nwa.add_epsilon(2, 3, all.clone());
        nwa.add_transition(1, DEFAULT_LABEL, 3, all.clone());
        nwa.add_epsilon(0, 1, all);

        let (preds, outdegree) = build_finality_preds_and_outdegree(&nwa);
        let reverse_topo_order =
            build_finality_reverse_topo_order(&preds, outdegree).expect("test graph is acyclic");
        let mut serial = collect_initial_final_weights(&nwa);
        let mut chunked_parallel = serial.clone();
        let mut weight_ops = ScopedWeightOpCache::default();

        apply_finality_fixpoint_acyclic(
            &preds,
            &mut serial,
            &reverse_topo_order,
            &mut weight_ops,
        );
        apply_finality_fixpoint_acyclic_parallel_waves_chunked(
            &preds,
            &mut chunked_parallel,
            &reverse_topo_order,
        );

        for (serial_weight, chunked_weight) in serial.iter().zip(&chunked_parallel) {
            let serial_weight = serial_weight
                .as_ref()
                .map(|weight| weight.weight.clone())
                .unwrap_or_else(Weight::empty);
            let chunked_weight = chunked_weight
                .as_ref()
                .map(|weight| weight.weight.clone())
                .unwrap_or_else(Weight::empty);
            for tsid in 0..=7 {
                assert_eq!(
                    serial_weight.tokens_for_tsid(tsid),
                    chunked_weight.tokens_for_tsid(tsid),
                );
            }
        }
    }
}
