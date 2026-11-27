//! The cancellation fixed point over an arbitrary exact weight algebra.
//! Queries represent an unmatched push followed by balanced excursions;
//! a matching read grows a balanced edge, which wakes its upstream queries.
use std::collections::VecDeque;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::weight_row::SparseWeightRow;

type Query = (u32, i32);
type Task = (u32, u32, i32);
type QueryRow<W> = SparseWeightRow<Query, W>;
pub(crate) type BalancedRow<W> = SparseWeightRow<u32, W>;

pub(crate) trait CancellationWeights {
    type Value: Clone + Eq;
    fn is_empty(value: &Self::Value) -> bool;
    fn identity(value: &Self::Value) -> usize;
    fn intersection(&mut self, left: &Self::Value, right: &Self::Value) -> Self::Value;
    fn union(&mut self, left: &Self::Value, right: &Self::Value) -> Self::Value;
    fn begin_batch(&mut self, _pops: usize, _states: usize) -> bool { true }
    fn finish_batch(&mut self) {}
    fn complete(&mut self) -> bool { true }
}

pub(crate) trait CancellationGraph<W> {
    fn len(&self) -> usize;
    fn pushes(&self, source: u32, visit: impl FnMut(u32, i32, &W));
    fn epsilons(&self, source: u32, visit: impl FnMut(u32, &W));
    fn reads(&self, source: u32, label: i32, visit: impl FnMut(u32, &W));
}

#[derive(Default)]
pub(crate) struct CancellationStats {
    pub groups: usize,
    pub source_queries: usize,
    pub pops: usize,
    pub query_entries: usize,
    pub max_query_entries: usize,
}

struct Group<W> {
    current: u32,
    label: i32,
    weight: W,
    sources: SmallVec<[u32; 8]>,
}

fn grow<A: CancellationWeights, K: Copy + Eq + std::hash::Hash>(
    row: &mut SparseWeightRow<K, A::Value>,
    key: K,
    value: A::Value,
    ops: &mut A,
) -> bool {
    row.merge_with(key, value, A::is_empty, |existing, add| {
        let next = ops.union(existing, &add);
        if next == *existing {
            false
        } else {
            *existing = next;
            true
        }
    })
}

fn enqueue<A: CancellationWeights>(
    queries: &mut [QueryRow<A::Value>],
    queue: &mut VecDeque<Task>,
    current: u32,
    source: u32,
    label: i32,
    value: A::Value,
    ops: &mut A,
) {
    if let Some(row) = queries.get_mut(current as usize) {
        if grow(row, (source, label), value, ops) {
            queue.push_back((current, source, label));
        }
    }
}

fn next_groups<A: CancellationWeights>(
    queue: &mut VecDeque<Task>,
    queries: &[QueryRow<A::Value>],
    grouped: bool,
    stats: &mut CancellationStats,
) -> SmallVec<[Group<A::Value>; 1]> {
    let mut result = SmallVec::<[Group<A::Value>; 1]>::new();
    let mut indices = FxHashMap::<(u32, i32, usize), usize>::default();
    while let Some((current, source, label)) = queue.pop_front() {
        stats.pops += 1;
        let Some(weight) = queries[current as usize].get_ref(&(source, label)).cloned() else {
            continue;
        };
        let key = (current, label, A::identity(&weight));
        if grouped {
            if let Some(&index) = indices.get(&key) {
                result[index].sources.push(source);
                continue;
            }
            indices.insert(key, result.len());
        }
        result.push(Group { current, label, weight, sources: smallvec::smallvec![source] });
        if !grouped {
            break;
        }
    }
    if grouped {
        for group in &mut result {
            group.sources.sort_unstable();
            group.sources.dedup();
        }
    }
    result
}

pub(crate) fn solve<A: CancellationWeights, G: CancellationGraph<A::Value>>(
    graph: &G,
    sources: impl IntoIterator<Item = u32>,
    foreign: Option<&[BalancedRow<A::Value>]>,
    ops: &mut A,
    grouped: bool,
) -> Option<(Vec<BalancedRow<A::Value>>, CancellationStats)> {
    let n = graph.len();
    let mut queries = vec![QueryRow::default(); n];
    let mut derived = vec![BalancedRow::default(); n];
    let mut queue = VecDeque::new();
    let mut stats = CancellationStats::default();

    for source in sources {
        if source as usize >= n {
            continue;
        }
        graph.pushes(source, |target, label, weight| {
            enqueue(&mut queries, &mut queue, target, source, label, weight.clone(), ops);
        });
    }

    while !queue.is_empty() {
        if !ops.begin_batch(stats.pops, n) {
            return None;
        }
        for group in next_groups::<A>(&mut queue, &queries, grouped, &mut stats) {
            stats.groups += 1;
            stats.source_queries += group.sources.len();
            // Extend an unmatched push by any available balanced excursion.
            let mut propagate = |target: u32, edge: &A::Value| {
                if target as usize >= n {
                    return;
                }
                let value = ops.intersection(&group.weight, edge);
                if !A::is_empty(&value) {
                    for &source in &group.sources {
                        enqueue(&mut queries, &mut queue, target, source, group.label, value.clone(), ops);
                    }
                }
            };
            derived[group.current as usize].visit(&mut propagate);
            if let Some(foreign) = foreign {
                foreign[group.current as usize].visit(&mut propagate);
            }
            graph.epsilons(group.current, &mut propagate);

            // A matching read closes the push. A new or enlarged balanced
            // edge must propagate even to queries processed before it existed.
            graph.reads(group.current, group.label, |target, edge| {
                if target as usize >= n {
                    return;
                }
                let value = ops.intersection(&group.weight, edge);
                for &source in &group.sources {
                    if !grow(&mut derived[source as usize], target, value.clone(), ops) {
                        continue;
                    }
                    let balanced = derived[source as usize].get_ref(&target).unwrap();
                    // The destination may alias the upstream query row on a
                    // cyclic graph. Stage only its nonempty products.
                    let mut updates = SmallVec::<[(u32, i32, A::Value); 4]>::new();
                    queries[source as usize].visit(|(upstream, label), weight| {
                        let product = ops.intersection(weight, balanced);
                        if !A::is_empty(&product) {
                            updates.push((upstream, label, product));
                        }
                    });
                    for (upstream, label, product) in updates {
                        enqueue(&mut queries, &mut queue, target, upstream, label, product, ops);
                    }
                }
            });
        }
        ops.finish_batch();
    }
    if !ops.complete() {
        return None;
    }
    stats.query_entries = queries.iter().map(QueryRow::len).sum();
    stats.max_query_entries = queries.iter().map(QueryRow::len).max().unwrap_or(0);
    Some((derived, stats))
}
