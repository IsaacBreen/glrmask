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
type DerivedEpsilons = Vec<SmallTargetWeights>;
type SubsetMemo = FxHashMap<(usize, usize), bool>;
type SingleIntersectionMemo = FxHashMap<(usize, usize), (Weight, Weight, Weight)>;

/// Wave scheduling pays two parallel-iterator setup costs per DAG layer.  This
/// many predecessor edges per Rayon worker is enough to amortize that work on
/// the small and large BFCL catalogs while leaving genuinely small parsers on
/// the lower-overhead serial solver.
const MIN_PARALLEL_FINALITY_EDGES_PER_WORKER: usize = 512;
/// Grouping identical cancellation queries pays for a wave/group build. Large
/// parser NWAs amortize it heavily; small NWAs are faster with the direct FIFO.
const GROUPED_CANCELLATION_MIN_STATES: usize = 4_096;
/// Checking for structurally dead cancellation tasks avoids weight cloning and
/// inspection on the large parser-NWA tail, but the extra transition probes do
/// not amortize on small graphs.
const DEAD_CANCELLATION_FAST_PATH_MIN_STATES: usize = 65_536;
/// Direct reverse-topological accumulation avoids one pending buffer per state
/// and wins on the large parser-NWA tail. Smaller graphs benefit more from
/// batching their few multi-edge contributions before unioning them.
const DIRECT_ACYCLIC_FINALITY_MIN_STATES: usize = 65_536;
/// Build finality predecessor/topology metadata concurrently with cancellation
/// once the graph is large enough for the extra Rayon job and owned edge refs
/// to amortize. Above the direct-finality threshold the existing parallel
/// finality solvers remain preferable until they can consume the same plan.
const PRECOMPUTED_FINALITY_MIN_STATES: usize = 8_192;

#[derive(Clone, Default)]
enum SmallQueryWeights {
    #[default]
    Empty,
    One(QueryKey, Weight),
    Many(FxHashMap<QueryKey, Weight>),
}

#[derive(Clone, Default)]
enum SmallTargetWeights {
    #[default]
    Empty,
    One(u32, Weight),
    Many(FxHashMap<u32, Weight>),
}

enum SmallTargetWeightsIntoIter {
    Empty,
    One(Option<(u32, Weight)>),
    Many(std::collections::hash_map::IntoIter<u32, Weight>),
}

impl Iterator for SmallTargetWeightsIntoIter {
    type Item = (u32, Weight);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::One(entry) => entry.take(),
            Self::Many(entries) => entries.next(),
        }
    }
}

impl SmallTargetWeights {
    fn merge(&mut self, target: u32, add: Weight) -> bool {
        match self {
            Self::Empty => {
                if add.is_empty() {
                    return false;
                }
                *self = Self::One(target, add);
                true
            }
            Self::One(existing_target, existing_weight) if *existing_target == target => {
                merge_weight(existing_weight, add)
            }
            Self::One(existing_target, existing_weight) => {
                if add.is_empty() {
                    return false;
                }
                let previous_target = *existing_target;
                let previous_weight = existing_weight.clone();
                let mut entries = FxHashMap::with_capacity_and_hasher(4, Default::default());
                entries.insert(previous_target, previous_weight);
                entries.insert(target, add);
                *self = Self::Many(entries);
                true
            }
            Self::Many(entries) => {
                let entry = entries.entry(target).or_insert_with(Weight::empty);
                merge_weight(entry, add)
            }
        }
    }

    fn get(&self, target: u32) -> Option<&Weight> {
        match self {
            Self::Empty => None,
            Self::One(existing_target, weight) => {
                (*existing_target == target).then_some(weight)
            }
            Self::Many(entries) => entries.get(&target),
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    fn for_each(&self, mut f: impl FnMut(u32, &Weight)) {
        match self {
            Self::Empty => {}
            Self::One(target, weight) => f(*target, weight),
            Self::Many(entries) => {
                for (&target, weight) in entries {
                    f(target, weight);
                }
            }
        }
    }

    fn into_entries(self) -> SmallTargetWeightsIntoIter {
        match self {
            Self::Empty => SmallTargetWeightsIntoIter::Empty,
            Self::One(target, weight) => {
                SmallTargetWeightsIntoIter::One(Some((target, weight)))
            }
            Self::Many(entries) => SmallTargetWeightsIntoIter::Many(entries.into_iter()),
        }
    }
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

fn intersect_with_single_weight_hint_cached(
    left: &Weight,
    left_single: Option<&(u32, u32, Arc<RangeSetBlaze<u32>>)>,
    right: &Weight,
    memo: &mut SingleIntersectionMemo,
) -> Weight {
    if left.is_full() {
        return right.clone();
    }
    if right.is_full() {
        return left.clone();
    }
    let Some((start, end, tokens)) = left_single else {
        return left.intersection(right);
    };

    let key = (left.ptr_key(), right.ptr_key());
    if let Some((cached_left, cached_right, result)) = memo.get(&key) {
        debug_assert!(cached_left.storage_ptr_eq(left));
        debug_assert!(cached_right.storage_ptr_eq(right));
        return result.clone();
    }

    let result = right.intersect_single_parts(*start, *end, tokens);
    // Holding both operands strongly makes pointer identity a safe cache key:
    // neither address can be recycled while the entry is resident.
    memo.insert(key, (left.clone(), right.clone(), result.clone()));
    result
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

#[derive(Clone)]
struct PredEdge {
    from: usize,
    weight: Weight,
}

/// Finality metadata that can be prepared while cancellation is still reading
/// the original NWA.  `reverse_topo_order` is a reverse topological order of
/// the *full* original graph, not merely the DEFAULT/negative/epsilon subgraph.
/// Every cancellation epsilon shortcuts an existing path in that graph, so the
/// same order remains valid after those derived edges are installed.
struct AcyclicFinalityPlan {
    preds: Vec<Vec<PredEdge>>,
    reverse_topo_order: Vec<usize>,
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
            .is_some_and(|guard| guard.storage_ptr_eq(edge_weight))
    }

    fn intersect(
        &self, edge: &Weight, operation: impl FnOnce(&Weight, &Weight) -> Weight,
    ) -> Option<Self> {
        if self.weight.is_empty() || edge.is_empty() { return None; }
        if self.is_guarded_by(edge) { return Some(self.clone()); }
        let weight = if edge.is_full() {
            self.weight.clone()
        } else if self.weight.is_full() {
            edge.clone()
        } else {
            operation(&self.weight, edge)
        };
        (!weight.is_empty()).then_some(Self { weight, subset_of: Some(edge.clone()) })
    }

    fn intersection_with_edge(&self, edge: &Weight) -> Option<Self> {
        self.intersect(edge, |left, right| left.intersection_uncached(right))
    }

    fn intersection_with_edge_cached(
        &self, edge: &Weight, ops: &mut ScopedWeightOpCache,
    ) -> Option<Self> {
        self.intersect(edge, |left, right| ops.intersection(left, right))
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
                    .is_some_and(|(left, right)| left.storage_ptr_eq(right))
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
                        .is_some_and(|guard| first_guard.storage_ptr_eq(guard))
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
    single_intersection_memo: &mut SingleIntersectionMemo,
) {
    let local = &derived_epsilons[current_state as usize];
    let foreign = foreign_derived.map(|derived| &derived[current_state as usize]);

    let mut propagate = |target_state: u32,
                         epsilon_weight: &Weight,
                         query_weights: &mut QueryWeights,
                         worklist: &mut VecDeque<CancellationTask>| {
        let propagated = intersect_with_single_weight_hint_cached(
            query_weight_to_current,
            query_single,
            epsilon_weight,
            single_intersection_memo,
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

    local.for_each(|target_state, epsilon_weight| {
        propagate(target_state, epsilon_weight, query_weights, worklist);
    });
    if let Some(foreign) = foreign {
        foreign.for_each(|target_state, epsilon_weight| {
            // Skip if local already has a stronger entry for this target
            // (merging handled below in queue_query_weight's dedup anyway).
            propagate(target_state, epsilon_weight, query_weights, worklist);
        });
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
    single_intersection_memo: &mut SingleIntersectionMemo,
) {
    let new_derived_weight = intersect_with_single_weight_hint_cached(
        query_weight_to_current,
        query_single,
        edge_weight,
        single_intersection_memo,
    );
    if new_derived_weight.is_empty() {
        return;
    }

    if !derived_epsilons[source_state as usize].merge(target_state, new_derived_weight) {
        return;
    }

    let Some(derived_weight) = derived_epsilons[source_state as usize].get(target_state) else {
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

fn extend_derived_epsilon_precomputed(
    source_state: u32,
    target_state: u32,
    new_derived_weight: &Weight,
    query_weights: &mut QueryWeights,
    worklist: &mut VecDeque<CancellationTask>,
    derived_epsilons: &mut DerivedEpsilons,
    subset_memo: &mut SubsetMemo,
) {
    if new_derived_weight.is_empty() {
        return;
    }

    if !derived_epsilons[source_state as usize].merge(target_state, new_derived_weight.clone()) {
        return;
    }

    let Some(derived_weight) = derived_epsilons[source_state as usize].get(target_state) else {
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
        if !propagated.is_empty() {
            propagated_updates.push((upstream_source_state, upstream_label, propagated));
        }
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

#[derive(Debug)]
struct CancellationQueryGroup {
    current_state: u32,
    positive_label: i32,
    weight: Weight,
    sources: SmallVec<[u32; 8]>,
}

fn build_cancellation_query_groups(
    worklist: &mut VecDeque<CancellationTask>,
    query_weights: &QueryWeights,
) -> Vec<CancellationQueryGroup> {
    let mut group_index = FxHashMap::<(u32, i32, usize), usize>::default();
    let mut groups = Vec::<CancellationQueryGroup>::new();

    while let Some((current_state, source_state, positive_label)) = worklist.pop_front() {
        let Some(weight) = query_weights[current_state as usize]
            .get(&(source_state, positive_label))
            .cloned()
        else {
            continue;
        };
        let key = (current_state, positive_label, weight.ptr_key());
        if let Some(&index) = group_index.get(&key) {
            groups[index].sources.push(source_state);
        } else {
            let index = groups.len();
            group_index.insert(key, index);
            groups.push(CancellationQueryGroup {
                current_state,
                positive_label,
                weight,
                sources: smallvec::smallvec![source_state],
            });
        }
    }
    for group in &mut groups {
        group.sources.sort_unstable();
        group.sources.dedup();
    }
    groups
}

fn queue_group_propagation(
    query_weights: &mut QueryWeights,
    worklist: &mut VecDeque<CancellationTask>,
    sources: &[u32],
    target_state: u32,
    positive_label: i32,
    propagated: &Weight,
) {
    if propagated.is_empty() {
        return;
    }
    for &source_state in sources {
        queue_query_weight(
            query_weights,
            worklist,
            target_state,
            source_state,
            positive_label,
            propagated.clone(),
        );
    }
}

fn compute_cancellations_range_grouped_inner(
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
    let mut derived_epsilons: DerivedEpsilons =
        vec![SmallTargetWeights::Empty; state_count as usize];
    let mut subset_memo = SubsetMemo::default();

    for source_state in range {
        if source_state >= state_count {
            continue;
        }
        for (&label, targets) in nwa.states()[source_state as usize].transitions.range(..0) {
            let positive_label = negative_to_positive_label(label);
            for (target_state, weight) in targets {
                if *target_state < state_count && !weight.is_empty() {
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
    }

    let profile_detail = std::env::var_os("GLRMASK_PROFILE_CANCELLATION_DETAIL").is_some();
    let initial_tasks = worklist.len();
    let initial_groups = if profile_detail {
        let mut groups = FxHashSet::<(u32, i32, usize)>::default();
        for (state, queries) in query_weights.iter().enumerate() {
            queries.for_each(|(_source, label), weight| {
                groups.insert((state as u32, label, weight.ptr_key()));
            });
        }
        groups.len()
    } else {
        0
    };
    let mut rounds = 0usize;
    let mut groups_processed = 0usize;
    let mut source_queries_processed = 0usize;

    while !worklist.is_empty() {
        rounds += 1;
        let groups = build_cancellation_query_groups(&mut worklist, &query_weights);
        groups_processed += groups.len();
        for group in groups {
            source_queries_processed += group.sources.len();
            let query_single = group.weight.single_compact_entry_parts();
            let current_state = group.current_state;
            let positive_label = group.positive_label;

            let propagate_map = |targets: &SmallTargetWeights,
                                 query_weights: &mut QueryWeights,
                                 worklist: &mut VecDeque<CancellationTask>| {
                targets.for_each(|target_state, epsilon_weight| {
                    let propagated = intersect_with_single_weight_hint(
                        &group.weight,
                        query_single.as_ref(),
                        epsilon_weight,
                    );
                    queue_group_propagation(
                        query_weights,
                        worklist,
                        &group.sources,
                        target_state,
                        positive_label,
                        &propagated,
                    );
                });
            };
            let local = &derived_epsilons[current_state as usize];
            propagate_map(local, &mut query_weights, &mut worklist);
            if let Some(foreign) =
                foreign_derived.map(|derived| &derived[current_state as usize])
            {
                propagate_map(foreign, &mut query_weights, &mut worklist);
            }

            let state = &nwa.states()[current_state as usize];
            for targets in [
                state.transitions.get(&positive_label),
                state.transitions.get(&DEFAULT_LABEL),
            ]
            .into_iter()
            .flatten()
            {
                for (target_state, edge_weight) in targets {
                    if *target_state >= state_count {
                        continue;
                    }
                    let new_derived_weight = intersect_with_single_weight_hint(
                        &group.weight,
                        query_single.as_ref(),
                        edge_weight,
                    );
                    if new_derived_weight.is_empty() {
                        continue;
                    }
                    for &source_state in &group.sources {
                        extend_derived_epsilon_precomputed(
                            source_state,
                            *target_state,
                            &new_derived_weight,
                            &mut query_weights,
                            &mut worklist,
                            &mut derived_epsilons,
                            &mut subset_memo,
                        );
                    }
                }
            }

            for (target_state, epsilon_weight) in &state.epsilons {
                if *target_state >= state_count {
                    continue;
                }
                let propagated = intersect_with_single_weight_hint(
                    &group.weight,
                    query_single.as_ref(),
                    epsilon_weight,
                );
                queue_group_propagation(
                    &mut query_weights,
                    &mut worklist,
                    &group.sources,
                    *target_state,
                    positive_label,
                    &propagated,
                );
            }
        }
    }

    let mut derived = collect_non_empty_derived_epsilons(derived_epsilons);
    derived.sort_by_key(|(from, to, _)| (*from, *to));
    if profile_detail {
        eprintln!(
            "[glrmask/profile][cancellation_grouped] states={} initial_tasks={} initial_groups={} initial_compression={:.2} rounds={} groups={} source_queries={} compression={:.2} derived_edges={}",
            state_count,
            initial_tasks,
            initial_groups,
            initial_tasks as f64 / initial_groups.max(1) as f64,
            rounds,
            groups_processed,
            source_queries_processed,
            source_queries_processed as f64 / groups_processed.max(1) as f64,
            derived.len(),
        );
    }
    derived
}

fn collect_non_empty_derived_epsilons(
    derived_epsilons: DerivedEpsilons,
) -> Vec<(u32, u32, Weight)> {
    let mut result = Vec::new();

    for (from_state, targets) in derived_epsilons.into_iter().enumerate() {
        for (to_state, weight) in targets.into_entries() {
            if !weight.is_empty() {
                result.push((from_state as u32, to_state, weight));
            }
        }
    }

    result
}


fn reverse_topological_source_layers(nwa: &NWA) -> Option<Vec<Vec<u32>>> {
    let n = nwa.states().len();
    if n == 0 {
        return Some(Vec::new());
    }
    let mut remaining_out = vec![0usize; n];
    let mut predecessors = vec![Vec::<u32>::new(); n];
    for (from, state) in nwa.states().iter().enumerate() {
        let mut add_edge = |target: u32, weight: &Weight| {
            if weight.is_empty() || target as usize >= n {
                return;
            }
            remaining_out[from] += 1;
            predecessors[target as usize].push(from as u32);
        };
        for targets in state.transitions.values() {
            for (target, weight) in targets {
                add_edge(*target, weight);
            }
        }
        for (target, weight) in &state.epsilons {
            add_edge(*target, weight);
        }
    }
    let mut current = remaining_out
        .iter()
        .enumerate()
        .filter_map(|(state, &degree)| (degree == 0).then_some(state as u32))
        .collect::<Vec<_>>();
    let mut layers = Vec::new();
    let mut visited = 0usize;
    while !current.is_empty() {
        current.sort_unstable();
        visited += current.len();
        let mut next = Vec::new();
        for &target in &current {
            for &pred in &predecessors[target as usize] {
                let degree = &mut remaining_out[pred as usize];
                debug_assert!(*degree > 0);
                *degree -= 1;
                if *degree == 0 {
                    next.push(pred);
                }
            }
        }
        layers.push(current);
        current = next;
    }
    (visited == n).then_some(layers)
}

fn compute_cancellations_reverse_topological_parallel(
    nwa: &NWA,
    allow_grouped: bool,
) -> Option<Vec<(u32, u32, Weight)>> {
    if rayon::current_num_threads() <= 1 {
        return None;
    }
    let started = std::time::Instant::now();
    let layers = reverse_topological_source_layers(nwa)?;
    let layer_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut completed: DerivedEpsilons =
        vec![SmallTargetWeights::Empty; nwa.states().len()];
    let solve_started = std::time::Instant::now();
    let mut parallel_sources = 0usize;
    let mut max_width = 0usize;
    let contiguous_runs = layers.iter().map(|layer| {
        if layer.is_empty() {
            return 0usize;
        }
        let mut sorted = layer.clone();
        sorted.sort_unstable();
        1 + sorted.windows(2).filter(|pair| pair[1] != pair[0] + 1).count()
    }).sum::<usize>();
    let max_runs_per_layer = layers.iter().map(|layer| {
        if layer.is_empty() {
            return 0usize;
        }
        let mut sorted = layer.clone();
        sorted.sort_unstable();
        1 + sorted.windows(2).filter(|pair| pair[1] != pair[0] + 1).count()
    }).max().unwrap_or(0);
    for layer in &layers {
        max_width = max_width.max(layer.len());
        parallel_sources += layer.len();
        let foreign = &completed;
        // States in one reverse-topological layer cannot reach one another:
        // every live outgoing edge points into a layer already completed.  It
        // is therefore exact to solve arbitrary groups of same-layer sources
        // against the same immutable `foreign` relation.  Coarsen to at most
        // one workspace per Rayon worker so the O(|states|) query/derived
        // vectors are amortized across hundreds of sources instead of being
        // allocated once per source.
        let workers = rayon::current_num_threads().max(1).min(layer.len().max(1));
        let chunk_size = layer.len().div_ceil(workers).max(1);
        let grouped_requested = allow_grouped
            && std::env::var("GLRMASK_ENABLE_GROUPED_CANCELLATION")
                .map(|value| {
                    let value = value.trim();
                    !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
                })
                .unwrap_or(false);
        let rows = layer
            .par_chunks(chunk_size)
            .map(|sources| {
                if grouped_requested {
                    // The grouped solver still takes a contiguous range and
                    // cannot represent arbitrary layer chunks exactly.  Keep
                    // grouped mode on the established source-local path until
                    // it has a source-slice workspace implementation.
                    sources
                        .iter()
                        .flat_map(|&source| {
                            compute_cancellations_range_grouped_inner(
                                nwa,
                                source..source + 1,
                                Some(foreign),
                            )
                        })
                        .collect::<Vec<_>>()
                } else {
                    compute_cancellations_sources_serial_inner(
                        nwa,
                        sources.iter().copied(),
                        Some(foreign),
                    )
                }
            })
            .collect::<Vec<_>>();
        for row in rows {
            for (from, to, weight) in row {
                completed[from as usize].merge(to, weight);
            }
        }
    }
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1000.0;
    let result = collect_non_empty_derived_epsilons(completed);
    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
    {
        eprintln!(
            "[glrmask/profile][parallel_reverse_topo_cancellation] states={} layers={} max_width={} sources={} contiguous_runs={} max_runs_per_layer={} derived_edges={} layer_ms={layer_ms:.3} solve_ms={solve_ms:.3} total_ms={:.3}",
            nwa.states().len(),
            layers.len(),
            max_width,
            parallel_sources,
            contiguous_runs,
            max_runs_per_layer,
            result.len(),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(result)
}

pub fn compute_cancellations_range(
    nwa: &NWA,
    range: std::ops::Range<u32>,
    allow_grouped: bool,
) -> Vec<(u32, u32, Weight)> {
    let solver_override = std::env::var("GLRMASK_CANCELLATION_SOLVER").ok();
    let use_grouped = match solver_override.as_deref().map(str::trim) {
        Some(value) if value.eq_ignore_ascii_case("grouped") => true,
        Some(value) if value.eq_ignore_ascii_case("serial") => false,
        _ => {
            let explicitly_enabled = std::env::var("GLRMASK_ENABLE_GROUPED_CANCELLATION")
                .map(|value| {
                    let value = value.trim();
                    !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
                })
                .unwrap_or(false);
            allow_grouped
                && explicitly_enabled
                && nwa.states().len() >= GROUPED_CANCELLATION_MIN_STATES
        }
    };
    if use_grouped {
        compute_cancellations_range_grouped_inner(nwa, range, None)
    } else {
        compute_cancellations_range_serial_inner(nwa, range, None)
    }
}

fn compute_cancellations_sources_serial_inner<I>(
    nwa: &NWA,
    sources: I,
    foreign_derived: Option<&DerivedEpsilons>,
) -> Vec<(u32, u32, Weight)>
where
    I: IntoIterator<Item = u32>,
{
    compute_cancellations_sources_with_pop_classes(nwa, sources, foreign_derived, None)
}

fn compute_cancellations_sources_with_pop_classes<I>(
    nwa: &NWA,
    sources: I,
    foreign_derived: Option<&DerivedEpsilons>,
    pop_classes: Option<&crate::pop_classes::PopLabelClasses>,
) -> Vec<(u32, u32, Weight)>
where
    I: IntoIterator<Item = u32>,
{
    let state_count = nwa.states().len() as u32;
    if state_count == 0 {
        return Vec::new();
    }

    let mut query_weights: QueryWeights = vec![SmallQueryWeights::Empty; state_count as usize];
    let mut worklist = VecDeque::<CancellationTask>::new();
    let mut derived_epsilons: DerivedEpsilons =
        vec![SmallTargetWeights::Empty; state_count as usize];
    let mut subset_memo = SubsetMemo::default();
    let mut single_intersection_memo = SingleIntersectionMemo::default();
    let use_dead_task_fast_path =
        state_count as usize >= DEAD_CANCELLATION_FAST_PATH_MIN_STATES;

    for source_state in sources {
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
        let state = &nwa.states()[current_state as usize];
        let local_derived = &derived_epsilons[current_state as usize];
        let foreign_derived_at_state =
            foreign_derived.map(|derived| &derived[current_state as usize]);
        let positive_targets = state.transitions.get(&positive_label);
        let default_targets = state.transitions.get(&DEFAULT_LABEL);

        if use_dead_task_fast_path
            && local_derived.is_empty()
            && foreign_derived_at_state.is_none_or(SmallTargetWeights::is_empty)
            && positive_targets.is_none()
            && default_targets.is_none()
            && pop_classes.is_none_or(|classes|
                classes.matching_targets(state, positive_label as u32).next().is_none())
            && state.epsilons.is_empty()
        {
            continue;
        }

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
            &mut single_intersection_memo,
        );

        if let Some(positive_targets) = positive_targets {
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
                        &mut single_intersection_memo,
                    );
                }
            }
        }

        if let Some(classes) = pop_classes {
            for (target_state, edge_weight) in classes.matching_targets(state, positive_label as u32) {
                extend_derived_epsilons(
                    &query_weight_to_current, query_single.as_ref(), source_state,
                    *target_state, edge_weight, &mut query_weights, &mut worklist,
                    &mut derived_epsilons, &mut subset_memo, &mut single_intersection_memo,
                );
            }
        }

        if let Some(default_targets) = default_targets {
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
                        &mut single_intersection_memo,
                    );
                }
            }
        }

        for (target_state, epsilon_weight) in &state.epsilons {
            if *target_state >= state_count {
                continue;
            }

            let propagated = intersect_with_single_weight_hint_cached(
                &query_weight_to_current,
                query_single.as_ref(),
                epsilon_weight,
                &mut single_intersection_memo,
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

fn compute_cancellations_range_serial_inner(
    nwa: &NWA,
    range: std::ops::Range<u32>,
    foreign_derived: Option<&DerivedEpsilons>,
) -> Vec<(u32, u32, Weight)> {
    compute_cancellations_sources_serial_inner(nwa, range, foreign_derived)
}

pub fn apply_cancellations_range(
    nwa: &mut NWA,
    range: std::ops::Range<u32>,
    allow_grouped: bool,
) {
    for (from, to, weight) in compute_cancellations_range(nwa, range, allow_grouped) {
        nwa.add_epsilon(from, to, weight);
    }
}

fn is_live_finality_edge(target_state: u32, weight: &Weight, state_count: usize) -> bool {
    (target_state as usize) < state_count && !weight.is_empty()
}

fn prepare_acyclic_finality_plan(nwa: &NWA) -> Option<AcyclicFinalityPlan> {
    let reverse_topo_order = reverse_topological_source_layers(nwa)?
        .into_iter().flatten().map(|state| state as usize).collect();
    let (preds, _) = build_finality_preds_and_outdegree(nwa);
    Some(AcyclicFinalityPlan { preds, reverse_topo_order })
}

fn apply_finality_fixpoint_precomputed_acyclic(nwa: &mut NWA, plan: &AcyclicFinalityPlan) {
    debug_assert_eq!(nwa.states().len(), plan.preds.len());
    let mut finals = collect_initial_final_weights(nwa);
    solve_finality_dag(&plan.preds, &mut finals, &plan.reverse_topo_order);
    write_final_weights(nwa, finals);
}

fn build_finality_preds_and_outdegree(nwa: &NWA) -> (Vec<Vec<PredEdge>>, Vec<usize>) {
    let n = nwa.states().len();
    let mut preds = vec![Vec::new(); n];
    let mut outdegree = vec![0; n];
    for (from, state) in nwa.states().iter().enumerate() {
        let mut add = |target, weight: &Weight| {
            if is_live_finality_edge(target, weight, n) {
                preds[target as usize].push(PredEdge { from, weight: weight.clone() });
                outdegree[from] += 1;
            }
        };
        for (target, weight) in &state.epsilons {
            add(*target, weight);
        }
        for (&label, targets) in &state.transitions {
            if label == DEFAULT_LABEL || is_negative_label(label) {
                for (target, weight) in targets {
                    add(*target, weight);
                }
            }
        }
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
    preds: &[Vec<PredEdge>],
    reachable_final_weights: &mut [Option<GuardedFinalWeight>],
    state_id: usize,
    weight_ops: &mut ScopedWeightOpCache,
    mut on_change: impl FnMut(usize),
) {
    let Some(reachable_final) = reachable_final_weights[state_id].clone() else {
        return;
    };

    for edge in &preds[state_id] {
        let Some(propagated) = reachable_final.intersection_with_edge_cached(&edge.weight, weight_ops) else {
            continue;
        };
        if merge_guarded_final_weight(&mut reachable_final_weights[edge.from], propagated, weight_ops) {
            on_change(edge.from);
        }
    }
}

fn apply_finality_fixpoint_worklist(
    nwa: &NWA,
    preds: &[Vec<PredEdge>],
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


/// One accumulator for a topologically scheduled fixed point. For small or
/// parallel graphs, coalesce all incoming contributions before unioning;
/// for large serial graphs, avoid retaining a pending allocation per state.
struct FinalityAccumulator<'a> {
    finals: &'a mut [Option<GuardedFinalWeight>],
    pending: Option<Vec<SmallVec<[GuardedFinalWeight; 4]>>>,
}

impl<'a> FinalityAccumulator<'a> {
    fn new(finals: &'a mut [Option<GuardedFinalWeight>], batch: bool) -> Self {
        let pending = batch.then(|| finals.iter_mut().map(|slot| {
            slot.take().into_iter().collect::<SmallVec<[GuardedFinalWeight; 4]>>()
        }).collect());
        Self { finals, pending }
    }

    fn complete(&mut self, state: usize, ops: &mut ScopedWeightOpCache) -> Option<GuardedFinalWeight> {
        if let Some(pending) = self.pending.as_mut() {
            self.finals[state] = union_guarded_pending(std::mem::take(&mut pending[state]), ops);
        }
        self.finals[state].clone()
    }

    fn add(&mut self, state: usize, weight: GuardedFinalWeight, ops: &mut ScopedWeightOpCache) {
        if let Some(pending) = self.pending.as_mut() {
            pending[state].push(weight);
        } else {
            merge_guarded_final_weight(&mut self.finals[state], weight, ops);
        }
    }

    fn complete_wave(&mut self, wave: &[usize]) -> Vec<(usize, GuardedFinalWeight)> {
        let pending = self.pending.as_mut().expect("parallel finality batches contributions");
        let jobs = wave.iter().map(|&state| (state, std::mem::take(&mut pending[state]))).collect::<Vec<_>>();
        let results = jobs.into_par_iter()
            .map_init(ScopedWeightOpCache::default, |ops, (state, pending)| {
                (state, union_guarded_pending(pending, ops))
            }).collect::<Vec<_>>();
        let mut live = Vec::new();
        for (state, result) in results {
            self.finals[state] = result.clone();
            if let Some(weight) = result { live.push((state, weight)); }
        }
        live
    }
}

/// A dependency-free wave is only a scheduling unit, not a new recurrence.
fn finality_reverse_topo_waves(preds: &[Vec<PredEdge>], order: &[usize]) -> Vec<Vec<usize>> {
    let mut level = vec![0usize; preds.len()];
    let mut waves = Vec::<Vec<usize>>::new();
    for &state in order {
        let wave = level[state];
        if waves.len() <= wave { waves.resize_with(wave + 1, Vec::new); }
        waves[wave].push(state);
        for edge in &preds[state] {
            level[edge.from] = level[edge.from].max(wave + 1);
        }
    }
    waves
}

/// Cache the algebraic operation, not its graph fanout. Owning the operands
/// keeps pointer-identity keys valid across all waves of the invocation.
#[derive(Default)]
struct ParallelFinalityProducts {
    ids: FxHashMap<(usize, usize), usize>,
    operands: Vec<(GuardedFinalWeight, Weight)>,
    results: Vec<Option<GuardedFinalWeight>>,
}

impl ParallelFinalityProducts {
    fn intern(&mut self, value: &GuardedFinalWeight, edge: &Weight) -> usize {
        let key = (value.weight.ptr_key(), edge.ptr_key());
        *self.ids.entry(key).or_insert_with(|| {
            let id = self.operands.len();
            self.operands.push((value.clone(), edge.clone()));
            id
        })
    }

    fn complete(&mut self) {
        self.results.extend(self.operands[self.results.len()..].par_iter()
            .map(|(value, edge)| value.intersection_with_edge(edge))
            .collect::<Vec<_>>());
    }
}

/// F(q) = initial(q) union sum_{q -> r} weight(q,r) intersect F(r).
/// Both serial and parallel DAG schedules finalize each state once; only
/// the intersection job scheduling and accumulator storage are selectable.
fn solve_finality_dag_with_policy(
    preds: &[Vec<PredEdge>],
    finals: &mut [Option<GuardedFinalWeight>],
    order: &[usize],
    parallel: bool,
    batch: bool,
) {
    let mut accum = FinalityAccumulator::new(finals, batch || parallel);
    let mut ops = ScopedWeightOpCache::default();
    if parallel {
        let mut products = ParallelFinalityProducts::default();
        for wave in finality_reverse_topo_waves(preds, order) {
            let mut uses = Vec::new();
            for (state, value) in accum.complete_wave(&wave) {
                for edge in &preds[state] {
                    uses.push((edge.from, products.intern(&value, &edge.weight)));
                }
            }
            products.complete();
            for (from, product) in uses {
                if let Some(value) = &products.results[product] {
                    accum.add(from, value.clone(), &mut ops);
                }
            }
        }
    } else {
        for &state in order {
            let Some(value) = accum.complete(state, &mut ops) else { continue; };
            for edge in &preds[state] {
                if let Some(value) = value.intersection_with_edge_cached(&edge.weight, &mut ops) {
                    accum.add(edge.from, value, &mut ops);
                }
            }
        }
    }
}

fn solve_finality_dag(
    preds: &[Vec<PredEdge>],
    finals: &mut [Option<GuardedFinalWeight>],
    order: &[usize],
) {
    let n = preds.len();
    let edges = preds.iter().map(Vec::len).sum::<usize>();
    let workers = rayon::current_num_threads();
    let forced = std::env::var("GLRMASK_FORCE_PARALLEL_FINALITY").is_ok_and(|value| value == "1");
    let parallel = workers > 1
        && edges >= MIN_PARALLEL_FINALITY_EDGES_PER_WORKER * workers
        && (forced || (n >= DIRECT_ACYCLIC_FINALITY_MIN_STATES
            && std::env::var_os("GLRMASK_DISABLE_DEDUP_PARALLEL_FINALITY").is_none()));
    let batch = std::env::var_os("GLRMASK_DISABLE_DIRECT_ACYCLIC_FINALITY").is_some()
        || (n < DIRECT_ACYCLIC_FINALITY_MIN_STATES
            && std::env::var_os("GLRMASK_FORCE_DIRECT_ACYCLIC_FINALITY").is_none());
    solve_finality_dag_with_policy(preds, finals, order, parallel, batch);
}

pub fn apply_finality_fixpoint(nwa: &mut NWA) {
    let started = std::time::Instant::now();
    let (preds, outdegree) = build_finality_preds_and_outdegree(nwa);
    let order = build_finality_reverse_topo_order(&preds, outdegree);
    let mut finals = collect_initial_final_weights(nwa);
    if let Some(order) = &order {
        solve_finality_dag(&preds, &mut finals, order);
    } else {
        apply_finality_fixpoint_worklist(nwa, &preds, &mut finals, &mut ScopedWeightOpCache::default());
    }
    let edges = preds.iter().map(Vec::len).sum::<usize>();
    write_final_weights(nwa, finals);
    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
    {
        eprintln!("[glrmask/profile][finality_fixpoint] states={} edges={} acyclic={} total_ms={:.3}",
            nwa.states().len(), edges, order.is_some(), started.elapsed().as_secs_f64() * 1000.0);
    }
}

pub fn remove_negative_transitions(nwa: &mut NWA) {
    if rayon::current_num_threads() > 1 && nwa.states().len() >= 4_096 {
        nwa.states_mut()
            .par_iter_mut()
            .for_each(|state| state.transitions.retain(|label, _| !is_negative_label(*label)));
    } else {
        for state in nwa.states_mut() {
            state.transitions.retain(|label, _| !is_negative_label(*label));
        }
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
    let prune = |state: &mut NWAState| {
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
    };
    if rayon::current_num_threads() > 1 && nwa.states().len() >= 4_096 {
        nwa.states_mut().par_iter_mut().for_each(prune);
    } else {
        for state in nwa.states_mut() {
            prune(state);
        }
    }
}

fn remove_negative_transitions_and_classify_terminal_states(nwa: &mut NWA) -> Vec<bool> {
    let process = |state: &mut NWAState| {
        state
            .transitions
            .retain(|label, _| !is_negative_label(*label));
        is_terminal_shape_candidate(state) && !state.transitions.contains_key(&DEFAULT_LABEL)
    };
    if rayon::current_num_threads() > 1 && nwa.states().len() >= 4_096 {
        nwa.states_mut().par_iter_mut().map(process).collect()
    } else {
        nwa.states_mut().iter_mut().map(process).collect()
    }
}

fn remove_redundant_default_transitions_with_initial(
    nwa: &mut NWA,
    mut terminal_states: Vec<bool>,
) {
    debug_assert_eq!(terminal_states.len(), nwa.states().len());
    grow_terminal_state_set(nwa, &mut terminal_states);
    prune_terminal_default_targets(nwa, &terminal_states);
}

pub fn remove_redundant_default_transitions(nwa: &mut NWA) {
    let classify = |state: &NWAState| {
        is_terminal_shape_candidate(state) && !state.transitions.contains_key(&DEFAULT_LABEL)
    };
    let mut terminal_states: Vec<bool> = if rayon::current_num_threads() > 1
        && nwa.states().len() >= 4_096
    {
        nwa.states().par_iter().map(classify).collect()
    } else {
        nwa.states().iter().map(classify).collect()
    };

    grow_terminal_state_set(nwa, &mut terminal_states);
    prune_terminal_default_targets(nwa, &terminal_states);
}

pub fn resolve_negative_codes_in_nwa(
    nwa: &mut NWA,
    allow_grouped_cancellation: bool,
) -> Option<Vec<u32>> {
    resolve_negative_codes_in_nwa_impl(nwa, allow_grouped_cancellation, None)
}

/// Run the ordinary weighted cancellation fixed point with finite consuming
/// POP classes. This changes only positive-edge matching for a pushed symbol;
/// class edges are not epsilon/default-finality edges and remain for the exact
/// positive predicate compiler. The ordinary LR path has no class catalog.
pub fn resolve_negative_codes_in_nwa_with_pop_classes(
    nwa: &mut NWA,
    classes: &crate::pop_classes::PopLabelClasses,
) -> Result<Option<Vec<u32>>, String> {
    classes.validate(nwa)?;
    Ok(resolve_negative_codes_in_nwa_impl(nwa, false, Some(classes)))
}

fn resolve_negative_codes_in_nwa_impl(
    nwa: &mut NWA,
    allow_grouped_cancellation: bool,
    pop_classes: Option<&crate::pop_classes::PopLabelClasses>,
) -> Option<Vec<u32>> {
    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
    let precompute_finality = rayon::current_num_threads() > 1
        && nwa.states().len() >= PRECOMPUTED_FINALITY_MIN_STATES
        // Larger graphs may select the deduplicated parallel finality solver;
        // leave those on the established path until the owned-plan variant has
        // an equally exact parallel implementation.
        && nwa.states().len() < DIRECT_ACYCLIC_FINALITY_MIN_STATES
        && std::env::var_os("GLRMASK_DISABLE_PRECOMPUTED_FINALITY").is_none();

    let cancellation_started_at = profile_enabled.then(std::time::Instant::now);
    let state_count = nwa.states().len() as u32;
    let use_parallel_reverse_topo =
        std::env::var_os("GLRMASK_EXPERIMENT_PARALLEL_REVERSE_TOPO_CANCELLATION").is_some()
            && rayon::current_num_threads() > 1
            && state_count >= 4_096;
    let compute_cancellations = || {
        if let Some(classes) = pop_classes {
            compute_cancellations_sources_with_pop_classes(nwa, 0..state_count, None, Some(classes))
        } else if use_parallel_reverse_topo {
            compute_cancellations_reverse_topological_parallel(nwa, allow_grouped_cancellation)
                .unwrap_or_else(|| {
                    compute_cancellations_range(nwa, 0..state_count, allow_grouped_cancellation)
                })
        } else if state_count == 0 {
            Vec::new()
        } else {
            compute_cancellations_range(nwa, 0..state_count, allow_grouped_cancellation)
        }
    };
    let precompute_finality = precompute_finality || use_parallel_reverse_topo;
    let (cancellations, mut finality_plan) = if precompute_finality {
        rayon::join(compute_cancellations, || prepare_acyclic_finality_plan(nwa))
    } else {
        (compute_cancellations(), None)
    };
    for (from, to, weight) in cancellations {
        if let Some(plan) = finality_plan.as_mut()
            && (to as usize) < plan.preds.len()
            && !weight.is_empty()
        {
            plan.preds[to as usize].push(PredEdge {
                from: from as usize,
                weight: weight.clone(),
            });
        }
        nwa.add_epsilon(from, to, weight);
    }
    let cancellation_ms = cancellation_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);

    let finality_started_at = profile_enabled.then(std::time::Instant::now);
    let used_precomputed_finality = finality_plan.is_some();
    if let Some(plan) = finality_plan.as_ref() {
        apply_finality_fixpoint_precomputed_acyclic(nwa, plan);
    } else {
        apply_finality_fixpoint(nwa);
    }
    let reusable_reverse_topo_order = finality_plan.as_ref().map(|plan| {
        plan.reverse_topo_order
            .iter()
            .map(|&state| state as u32)
            .collect::<Vec<_>>()
    });
    let finality_ms = finality_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
    let remove_negative_started_at = profile_enabled.then(std::time::Instant::now);
    let fused_cleanup = std::env::var_os(
        "GLRMASK_EXPERIMENT_FUSED_NEGATIVE_DEFAULT_CLEANUP",
    )
    .is_some();
    let initial_terminal_states = if fused_cleanup {
        Some(remove_negative_transitions_and_classify_terminal_states(nwa))
    } else {
        remove_negative_transitions(nwa);
        None
    };
    let remove_negative_ms = remove_negative_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
    let prune_defaults_started_at = profile_enabled.then(std::time::Instant::now);
    if let Some(initial_terminal_states) = initial_terminal_states {
        remove_redundant_default_transitions_with_initial(nwa, initial_terminal_states);
    } else {
        remove_redundant_default_transitions(nwa);
    }
    if let (Some(cancellation_ms), Some(finality_ms), Some(remove_negative_ms), Some(prune_defaults_started_at)) = (
        cancellation_ms,
        finality_ms,
        remove_negative_ms,
        prune_defaults_started_at,
    ) {
        eprintln!(
            "[glrmask/profile][resolve_negatives] states={} cancellation_ms={:.3} finality_ms={:.3} remove_negative_ms={:.3} prune_defaults_ms={:.3} precomputed_finality={}",
            nwa.states().len(),
            cancellation_ms,
            finality_ms,
            remove_negative_ms,
            prune_defaults_started_at.elapsed().as_secs_f64() * 1000.0,
            used_precomputed_finality,
        );
    }
    reusable_reverse_topo_order
}

#[cfg(test)]
mod terminal_default_tests {
    use range_set_blaze::RangeSetBlaze;

    use super::*;

    fn weight(tokens: std::ops::RangeInclusive<u32>) -> Weight {
        Weight::from_token_set_for_tsid(0, RangeSetBlaze::from_iter([tokens]))
    }

    fn normalize_cancellations(
        mut cancellations: Vec<(u32, u32, Weight)>,
    ) -> Vec<(u32, u32, Weight)> {
        cancellations.sort_by_key(|(from, to, _)| (*from, *to));
        cancellations
    }

    #[test]
    fn small_target_weights_promotes_merges_and_iterates_exactly() {
        let mut targets = SmallTargetWeights::Empty;
        let first = weight(0..=7);
        let first_add = weight(4..=11);
        let second = weight(16..=23);

        assert!(targets.merge(3, first.clone()));
        assert_eq!(targets.get(3), Some(&first));
        assert!(!targets.merge(3, weight(0..=3)));
        assert!(targets.merge(3, first_add));
        assert_eq!(
            targets.get(3).unwrap().tokens_for_tsid(0),
            Weight::from_token_set_for_tsid(0, RangeSetBlaze::from_iter(std::iter::once(0..=11)))
                .tokens_for_tsid(0)
        );

        assert!(targets.merge(9, second.clone()));
        assert_eq!(targets.get(9), Some(&second));
        let mut entries = targets.into_entries().collect::<Vec<_>>();
        entries.sort_by_key(|(target, _)| *target);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, 3);
        assert_eq!(entries[1], (9, second));
    }

    fn generated_cancellation_nwa(seed: u32) -> NWA {
        use crate::compiler::glr::labels::encode_negative_label;

        let state_count = 8 + (seed % 5);
        let mut nwa = NWA::new(1, 31);
        for _ in 0..state_count {
            nwa.add_state();
        }
        let weights = [
            weight(0..=7),
            weight(4..=15),
            weight(8..=23),
            weight(16..=31),
            weight(0..=31),
        ];

        for state in 0..state_count {
            let label = (state + seed) % 3;
            let negative_target = (state * 3 + seed + 1) % state_count;
            nwa.add_transition(
                state,
                encode_negative_label(label),
                negative_target,
                weights[((state + seed) as usize) % weights.len()].clone(),
            );

            for positive_label in 0..3 {
                let target = (state * 5 + positive_label + seed + 2) % state_count;
                nwa.add_transition(
                    state,
                    positive_label as i32,
                    target,
                    weights[((state + positive_label + seed + 1) as usize) % weights.len()]
                        .clone(),
                );
            }
            if (state + seed).is_multiple_of(2) {
                nwa.add_transition(
                    state,
                    DEFAULT_LABEL,
                    (state + 2) % state_count,
                    weights[((state + seed + 2) as usize) % weights.len()].clone(),
                );
            }
            if !(state + seed).is_multiple_of(3) {
                nwa.add_epsilon(
                    state,
                    (state + 1) % state_count,
                    weights[((state + seed + 3) as usize) % weights.len()].clone(),
                );
            }
        }
        nwa
    }

    #[test]
    fn grouped_cancellation_matches_serial_solver_on_generated_nwas() {
        for seed in 0..32 {
            let nwa = generated_cancellation_nwa(seed);
            let range = 0..nwa.num_states();
            let serial = normalize_cancellations(compute_cancellations_range_serial_inner(
                &nwa,
                range.clone(),
                None,
            ));
            let grouped = normalize_cancellations(compute_cancellations_range_grouped_inner(
                &nwa,
                range,
                None,
            ));
            assert_eq!(grouped, serial, "cancellation mismatch for seed {seed}");
        }
    }


    fn generated_acyclic_finality_nwa(seed: u32) -> NWA {
        use crate::compiler::glr::labels::encode_negative_label;

        let state_count = 12 + (seed % 7);
        let mut nwa = NWA::new(1, 31);
        for _ in 0..state_count {
            nwa.add_state();
        }
        let weights = [
            weight(0..=7),
            weight(4..=15),
            weight(8..=23),
            weight(16..=31),
            weight(0..=31),
        ];
        for state in 0..state_count {
            if state + 1 == state_count {
                nwa.set_final_weight(
                    state,
                    weights[((state + seed) as usize) % weights.len()].clone(),
                );
                continue;
            }
            if (state + seed).is_multiple_of(3) {
                nwa.set_final_weight(
                    state,
                    weights[((state + seed + 1) as usize) % weights.len()].clone(),
                );
            }
            let remaining = state_count - state - 1;
            let target = state + 1 + ((state * 5 + seed) % remaining);
            let label = ((state + seed) % 3) as i32;
            nwa.add_transition(
                state,
                encode_negative_label(label as u32),
                target,
                weights[((state + seed) as usize) % weights.len()].clone(),
            );
            let positive_target = state + 1 + ((state * 7 + seed + 1) % remaining);
            nwa.add_transition(
                state,
                label,
                positive_target,
                weights[((state + seed + 2) as usize) % weights.len()].clone(),
            );
            if (state + seed).is_multiple_of(2) {
                let default_target = state + 1 + ((state * 3 + seed + 2) % remaining);
                nwa.add_transition(
                    state,
                    DEFAULT_LABEL,
                    default_target,
                    weights[((state + seed + 3) as usize) % weights.len()].clone(),
                );
            }
            if !(state + seed).is_multiple_of(4) {
                let epsilon_target = state + 1 + ((state * 11 + seed + 3) % remaining);
                nwa.add_epsilon(
                    state,
                    epsilon_target,
                    weights[((state + seed + 4) as usize) % weights.len()].clone(),
                );
            }
        }
        nwa
    }

    #[test]
    fn reverse_topological_parallel_cancellation_matches_serial_on_generated_dags() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("test Rayon pool");
        pool.install(|| {
            for seed in 0..64 {
                let nwa = generated_acyclic_finality_nwa(seed);
                let serial = normalize_cancellations(compute_cancellations_range_serial_inner(
                    &nwa,
                    0..nwa.num_states(),
                    None,
                ));
                let parallel = normalize_cancellations(
                    compute_cancellations_reverse_topological_parallel(&nwa, false)
                        .expect("generated graph must admit reverse-topological layers"),
                );
                assert_eq!(parallel, serial, "cancellation mismatch for seed {seed}");
            }
        });
    }

    #[test]
    fn precomputed_finality_plan_matches_post_cancellation_scan_on_generated_dags() {
        for seed in 0..64 {
            let input = generated_acyclic_finality_nwa(seed);
            assert!(input.is_acyclic(), "generated graph must be acyclic");

            let mut expected = input.clone();
            let expected_states = expected.num_states();
            apply_cancellations_range(&mut expected, 0..expected_states, false);
            apply_finality_fixpoint(&mut expected);
            remove_negative_transitions(&mut expected);
            remove_redundant_default_transitions(&mut expected);

            let mut actual = input.clone();
            let mut plan = prepare_acyclic_finality_plan(&actual)
                .expect("generated graph must admit an acyclic finality plan");
            let cancellations =
                compute_cancellations_range(&actual, 0..actual.num_states(), false);
            for (from, to, weight) in cancellations {
                plan.preds[to as usize].push(PredEdge {
                    from: from as usize,
                    weight: weight.clone(),
                });
                actual.add_epsilon(from, to, weight);
            }
            apply_finality_fixpoint_precomputed_acyclic(&mut actual, &plan);
            remove_negative_transitions(&mut actual);
            remove_redundant_default_transitions(&mut actual);

            assert_eq!(actual.start_states(), expected.start_states(), "seed {seed}");
            assert_eq!(actual.states(), expected.states(), "seed {seed}");
        }
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

    fn reference_finality(nwa: &NWA) -> Vec<Weight> {
        let mut values = nwa.states().iter().map(|state| {
            state.final_weight.clone().unwrap_or_else(Weight::empty)
        }).collect::<Vec<_>>();
        loop {
            let previous = values.clone();
            for (from, state) in nwa.states().iter().enumerate() {
                let mut add = |target: u32, edge: &Weight| {
                    if let Some(value) = previous.get(target as usize) {
                        values[from] = values[from].union(&value.intersection(edge));
                    }
                };
                for (target, edge) in &state.epsilons { add(*target, edge); }
                for (&label, branches) in &state.transitions {
                    if label == DEFAULT_LABEL || is_negative_label(label) {
                        for (target, edge) in branches { add(*target, edge); }
                    }
                }
            }
            if previous == values { return values; }
        }
    }

    #[test]
    fn finality_schedules_match_literal_fixed_point() {
        for seed in 0..64 {
            let nwa = generated_acyclic_finality_nwa(seed);
            let expected = reference_finality(&nwa);
            let (preds, outdegree) = build_finality_preds_and_outdegree(&nwa);
            let order = build_finality_reverse_topo_order(&preds, outdegree).unwrap();
            for (parallel, batch) in [(false, false), (false, true), (true, true)] {
                let mut finals = collect_initial_final_weights(&nwa);
                solve_finality_dag_with_policy(&preds, &mut finals, &order, parallel, batch);
                let actual = finals.into_iter().map(|value| {
                    value.map(|value| value.weight).unwrap_or_else(Weight::empty)
                }).collect::<Vec<_>>();
                assert_eq!(actual, expected, "seed {seed}, parallel={parallel}, batch={batch}");
            }
        }
    }

}
