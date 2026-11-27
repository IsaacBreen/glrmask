//! Height-layered quotient of partial deterministic weighted rows.
//!
//! Each row describes its behavior on a live domain of (TSID, token) pairs.
//! Compatible rows agree on their common domain, including absent labels and
//! finality. All occurrences of a label must also have the same quotient
//! target, even on disjoint domains. Their union is again a partial row.
//!
//! Backward liveness, exact row interning and bottom-up merging suffice;
//! compressed weight ranges are never expanded into individual coordinates.
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;

use super::dwa::{DWA, DWAState};
use super::equivalence::find_difference;
use crate::ds::weight::{SharedTokenSet, Weight, WeightIntersectionIndex};

type Label = i32;

/// Ordering policy for exact pointwise merge groups during acyclic weighted
/// minimization. The policy affects only representation choices among already
/// compatible classes; it does not alter the accepted weighted language.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PointwiseClassOrder {
    /// Preserve the stable first-seen class order.
    Stable,
    /// Place denser partial behavior functions first so their constraints are
    /// absorbed before smaller compatible fragments are grouped greedily.
    DescendingDomain,
}

const UNMAPPED: u32 = u32::MAX;
fn weighted_dwa_minimize_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

fn reconstruction_token_range_coalescing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_WEIGHT_UNION_COALESCE_TOKEN_RANGES")
            .map(|value| {
                let value = value.trim();
                !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
            })
            .unwrap_or(false)
    })
}

fn weight_body_id(weight: &Weight) -> usize {
    weight.ptr_key()
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
    index: Option<&WeightIntersectionIndex>,
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

    let value = index.map_or_else(|| left.intersection(right), |index| left.intersection_with_index(index));
    cache.insert(key, value.clone());
    value
}

fn compute_topo_order(dwa: &DWA) -> Option<Vec<usize>> {
    let n = dwa.states().len();
    let mut in_degree = vec![0u32; n];
    for state in dwa.states() {
        for (_, target, _) in state.transitions.entries() {
            let t = target as usize;
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
        for (_, target, _) in dwa.states()[u].transitions.entries() {
            let t = target as usize;
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

/// Exact reverse structural hash-cons for an acyclic weighted DWA.
///
/// States are merged only when their complete weighted suffix representation is
/// identical after targets have already been hash-consed: same final weight and
/// the same ordered `label -> (target, weight)` row. This performs no weight
/// pushing or compatibility reasoning; it is a pure DAG quotient and therefore
/// preserves the weighted language exactly.
pub fn reverse_hashcons_acyclic_owned(dwa: DWA) -> DWA {
    if dwa.states().len() <= 1 {
        return dwa;
    }
    let Some(topo) = compute_topo_order(&dwa) else {
        return dwa;
    };

    type Signature = (Option<Weight>, Vec<(Label, u32, Weight)>);
    let old_start = dwa.start_state() as usize;
    let mut old_to_new = vec![UNMAPPED; dwa.states().len()];
    let mut intern = FxHashMap::<Signature, u32>::default();
    let mut new_states = Vec::<DWAState>::new();

    for &old_id in topo.iter().rev() {
        let old = &dwa.states()[old_id];
        let transitions = old
            .transitions
            .iter()
            .map(|(&label, (target, weight))| {
                let mapped = old_to_new[*target as usize];
                debug_assert_ne!(mapped, UNMAPPED, "reverse topo target must be mapped");
                (label, mapped, weight.clone())
            })
            .collect::<Vec<_>>();
        let signature = (old.final_weight.clone(), transitions.clone());
        if let Some(&existing) = intern.get(&signature) {
            old_to_new[old_id] = existing;
            continue;
        }

        let new_id = new_states.len() as u32;
        let state = DWAState {
            transitions: transitions
                .into_iter()
                .map(|(label, target, weight)| (label, (target, weight)))
                .collect(),
            final_weight: old.final_weight.clone(),
        };
        new_states.push(state);
        intern.insert(signature, new_id);
        old_to_new[old_id] = new_id;
    }

    let start_state = old_to_new.get(old_start).copied().unwrap_or(UNMAPPED);
    debug_assert_ne!(start_state, UNMAPPED);
    DWA::from_parts(new_states, start_state)
}

#[derive(Clone, Eq, PartialEq, Hash)]
struct ArcRow {
    label: Label,
    target: u32,
    weight: Weight,
}

struct LiveRow {
    domain: Weight,
    final_weight: Weight,
    arcs: Vec<ArcRow>,
}

impl LiveRow {
    fn empty() -> Self {
        Self { domain: Weight::empty(), final_weight: Weight::empty(), arcs: Vec::new() }
    }

    fn same_behavior(&self, other: &Self) -> bool {
        self.final_weight == other.final_weight && self.arcs == other.arcs
    }

    fn signature(&self) -> u64 {
        let mut hash = FxHasher::default();
        self.final_weight.hash(&mut hash);
        self.arcs.hash(&mut hash);
        hash.finish()
    }

    fn agrees_on_overlap(&self, other: &Self) -> bool {
        if self.domain.is_disjoint(&other.domain) {
            return true;
        }
        if !restricted_weights_agree(
            &self.final_weight, &other.final_weight, &self.domain, &other.domain,
        ) {
            return false;
        }
        let mut a = self.arcs.iter().peekable();
        let mut b = other.arcs.iter().peekable();
        loop {
            let (left, right) = match (a.peek(), b.peek()) {
                (Some(x), Some(y)) if x.label == y.label => (a.next(), b.next()),
                (Some(x), Some(y)) if x.label < y.label => (a.next(), None),
                (Some(_), Some(_)) => (None, b.next()),
                (Some(_), None) => (a.next(), None),
                (None, Some(_)) => (None, b.next()),
                (None, None) => return true,
            };
            let agrees = match (left, right) {
                (Some(x), Some(y)) => restricted_weights_agree(
                    &x.weight, &y.weight, &self.domain, &other.domain,
                ),
                (Some(x), None) => x.weight.is_disjoint(&other.domain),
                (None, Some(y)) => y.weight.is_disjoint(&self.domain),
                _ => unreachable!(),
            };
            if !agrees {
                return false;
            }
        }
    }

    // The caller has established compatibility, so overlapping labels have
    // equal targets. Merge sorted rows without maps or pointwise expansion.
    fn merge(self, other: Self) -> Self {
        let mut a = self.arcs.into_iter().peekable();
        let mut b = other.arcs.into_iter().peekable();
        let mut arcs = Vec::with_capacity(a.len().max(b.len()));
        loop {
            match (a.peek(), b.peek()) {
                (Some(x), Some(y)) if x.label == y.label => {
                    let mut x = a.next().unwrap();
                    let y = b.next().unwrap();
                    debug_assert_eq!(x.target, y.target);
                    x.weight = x.weight.union(&y.weight);
                    arcs.push(x);
                }
                (Some(x), Some(y)) if x.label < y.label => arcs.push(a.next().unwrap()),
                (Some(_), Some(_)) => arcs.push(b.next().unwrap()),
                (Some(_), None) => { arcs.extend(a); break; }
                (None, Some(_)) => { arcs.extend(b); break; }
                (None, None) => break,
            }
        }
        Self {
            domain: self.domain.union(&other.domain),
            final_weight: self.final_weight.union(&other.final_weight),
            arcs,
        }
    }

    fn into_state(self) -> DWAState {
        DWAState {
            final_weight: (!self.final_weight.is_empty()).then_some(self.final_weight),
            transitions: self.arcs.into_iter()
                .map(|arc| (arc.label, (arc.target, arc.weight))).collect(),
        }
    }
}

/// A cursor returns the set covering a coordinate and the inclusive end of
/// that constant region (including gaps). All four weight streams advance
/// monotonically; no temporary intersection weight is constructed.
struct RegionCursor<'a, I> {
    current: Option<(u32, u32, &'a SharedTokenSet)>,
    rest: I,
}

impl<'a, I: Iterator<Item = (u32, u32, &'a SharedTokenSet)>> RegionCursor<'a, I> {
    fn new(mut rest: I) -> Self {
        let current = rest.next();
        Self { current, rest }
    }

    fn at(&mut self, pos: u32) -> (Option<&'a SharedTokenSet>, u32) {
        while self.current.is_some_and(|(_, end, _)| end < pos) {
            self.current = self.rest.next();
        }
        match self.current {
            Some((start, end, set)) if start <= pos => (Some(set), end),
            Some((start, _, _)) => (None, start - 1),
            None => (None, u32::MAX),
        }
    }
}

fn intersect_token_ranges<'a>(
    left: &'a RangeSetBlaze<u32>,
    right: &'a RangeSetBlaze<u32>,
) -> impl Iterator<Item = (u32, u32)> + 'a {
    let mut left = left.ranges().peekable();
    let mut right = right.ranges().peekable();
    std::iter::from_fn(move || loop {
        let l = left.peek()?;
        let r = right.peek()?;
        let lo = (*l.start()).max(*r.start());
        let hi = (*l.end()).min(*r.end());
        let le = *l.end();
        let re = *r.end();
        if le <= re { left.next(); }
        if re <= le { right.next(); }
        if lo <= hi { return Some((lo, hi)); }
    })
}

/// Given a ⊆ da and b ⊆ db, agreement on da ∩ db is equivalent to
/// a ∩ db = b ∩ da. This handles empty, full, sparse and compressed weights
/// with the same interval algebra.
fn restricted_weights_agree(a: &Weight, b: &Weight, da: &Weight, db: &Weight) -> bool {
    if a.storage_ptr_eq(b) || a == b {
        return true;
    }
    if a.is_empty() { return b.is_disjoint(da); }
    if b.is_empty() { return a.is_disjoint(db); }
    if da.storage_ptr_eq(db) || da == db { return false; }
    // ALL is encoded as a reserved sentinel, not an ordinary TSID interval.
    // Keep it out of the finite-region sweep.
    if a.is_full() { return b == db; }
    if b.is_full() { return a == da; }
    if da.is_full() { return a.intersection(db) == *b; }
    if db.is_full() { return b.intersection(da) == *a; }

    let mut a = RegionCursor::new(a.range_entries());
    let mut b = RegionCursor::new(b.range_entries());
    let mut da = RegionCursor::new(da.range_entries());
    let mut db = RegionCursor::new(db.range_entries());
    let mut pos = 0;
    loop {
        let (left_domain, le) = da.at(pos);
        let (right_domain, re) = db.at(pos);
        let mut end = le.min(re);
        if let (Some(ld), Some(rd)) = (left_domain, right_domain) {
            if !ld.is_disjoint(rd) {
                let (x, xe) = a.at(pos);
                let (y, ye) = b.at(pos);
                end = end.min(xe).min(ye);
                let agrees = match (x, y) {
                    (None, None) => true,
                    (Some(x), None) => x.is_disjoint(rd),
                    (None, Some(y)) => y.is_disjoint(ld),
                    (Some(x), Some(y)) => Arc::ptr_eq(x, y)
                        || x == y
                        || (Arc::ptr_eq(x, ld) && Arc::ptr_eq(y, rd))
                        || intersect_token_ranges(x, rd).eq(intersect_token_ranges(y, ld)),
                };
                if !agrees { return false; }
            }
        }
        if end == u32::MAX { return true; }
        pos = end + 1;
    }
}

fn batch_build_weight(weights: Vec<Weight>) -> Weight {
    if reconstruction_token_range_coalescing_enabled() {
        Weight::union_all_reconstruction(weights.iter())
    } else {
        Weight::union_all(weights.iter())
    }
}

/// A binary forest of compatible summaries. A block of 2^k members occupies
/// slot k; insertion merges each member at most once per level. Unlike a
/// repeatedly rebuilt prefix, broad domains are not copied for every class.
/// The label index cheaply rejects globally incompatible deterministic rows.
#[derive(Default)]
struct MergeGroup {
    targets: FxHashMap<Label, u32>,
    blocks: Vec<Option<LiveRow>>,
}

impl MergeGroup {
    fn accepts(&self, row: &LiveRow) -> bool {
        row.arcs.iter().all(|arc| {
            self.targets.get(&arc.label).is_none_or(|&target| target == arc.target)
        }) && self.blocks.iter().flatten().all(|block| block.agrees_on_overlap(row))
    }

    fn insert(&mut self, mut row: LiveRow) {
        for arc in &row.arcs {
            self.targets.insert(arc.label, arc.target);
        }
        for block in &mut self.blocks {
            match block.take() {
                Some(old) => row = old.merge(row),
                None => { *block = Some(row); return; }
            }
        }
        self.blocks.push(Some(row));
    }

    fn finish(self) -> DWAState {
        let blocks: Vec<_> = self.blocks.into_iter().flatten().collect();
        if blocks.len() == 1 {
            return blocks.into_iter().next().unwrap().into_state();
        }
        let mut finals = Vec::with_capacity(blocks.len());
        let mut edges: FxHashMap<Label, (u32, Vec<Weight>)> = FxHashMap::default();
        for block in blocks {
            finals.push(block.final_weight);
            for arc in block.arcs {
                let (target, weights) = edges.entry(arc.label)
                    .or_insert_with(|| (arc.target, Vec::new()));
                debug_assert_eq!(*target, arc.target);
                weights.push(arc.weight);
            }
        }
        let final_weight = batch_build_weight(finals);
        DWAState {
            final_weight: (!final_weight.is_empty()).then_some(final_weight),
            transitions: edges.into_iter()
                .map(|(label, (target, weights))| (label, (target, batch_build_weight(weights))))
                .collect(),
        }
    }
}

/// Incoming cumulative weights bound every context in which a state can be
/// reached. Even for a non-cumulative input this is a safe overapproximation.
fn forward_domains(dwa: &DWA) -> Vec<Weight> {
    let n = dwa.states().len();
    let mut incoming = vec![SmallVec::<[Weight; 2]>::new(); n];
    for state in dwa.states() {
        for (_, target, weight) in state.transitions.entries() {
            if let Some(parts) = incoming.get_mut(target as usize) {
                if !weight.is_empty() { parts.push(weight.clone()); }
            }
        }
    }
    incoming.into_iter().enumerate().map(|(i, mut parts)| {
        if i == dwa.start_state() as usize { return Weight::all(); }
        parts.sort_unstable_by_key(Weight::ptr_key);
        parts.dedup_by_key(|weight| weight.ptr_key());
        Weight::union_all(parts.iter())
    }).collect()
}

/// Both the public weight-push operation and minimization consume these same
/// live rows. The original transition maps need not be materialized or mutated
/// to find the productive subgraph.
fn live_rows(dwa: &DWA, topo: &[usize], forward: bool) -> Vec<LiveRow> {
    let n = dwa.states().len();
    let forward = forward.then(|| forward_domains(dwa));
    let mut rows: Vec<_> = (0..n).map(|_| LiveRow::empty()).collect();
    let mut fanin = vec![0usize; n];
    for state in dwa.states() {
        for (_, target, _) in state.transitions.entries() {
            if let Some(count) = fanin.get_mut(target as usize) { *count += 1; }
        }
    }
    let mut memo = FxHashMap::default();
    let mut indexes: FxHashMap<usize, WeightIntersectionIndex> = FxHashMap::default();
    for &u in topo.iter().rev() {
        let state = &dwa.states()[u];
        let source = forward.as_ref().map(|domains| &domains[u]);
        let mut final_weight = state.final_weight.clone().unwrap_or_else(Weight::empty);
        if let Some(domain) = source {
            if !final_weight.is_subset(domain) {
                final_weight = memoized_intersection(&mut memo, &final_weight, domain, None);
            }
        }
        let mut arcs = Vec::with_capacity(state.transitions.len());
        for (label, target, weight) in state.transitions.entries() {
            let t = target as usize;
            if t >= n { continue; }
            let domain = &rows[t].domain;
            let index = if fanin[t] >= 8 && !domain.is_empty() && !domain.is_full() {
                Some(&*indexes.entry(domain.ptr_key()).or_insert_with(|| domain.intersection_index()))
            } else { None };
            let mut restricted = memoized_intersection(&mut memo, weight, domain, index);
            if let Some(source) = source {
                // Incoming domains restrict reachable contexts; they are not
                // a certificate of coaccessibility. Always intersect the
                // target's backward-live domain before restricting the source.
                // This intermediate operand need not outlive the local memo,
                // so do not use its pointer as a memo key.
                if !restricted.is_subset(source) {
                    restricted = restricted.intersection(source);
                }
            }
            if !restricted.is_empty() {
                arcs.push(ArcRow { label, target, weight: restricted });
            }
        }
        // Every stored weight is within the optional incoming domain. A
        // single component covering that domain proves the union without
        // constructing it. Otherwise compute the exact productive union.
        let domain = match source {
            Some(source) if final_weight == *source ||
                arcs.iter().any(|arc| arc.weight == *source) => source.clone(),
            _ => Weight::union_all(std::iter::once(&final_weight).chain(arcs.iter().map(|arc| &arc.weight))),
        };
        rows[u] = LiveRow { domain, final_weight, arcs };
    }
    rows
}

/// Intersect each edge with the backward-live domain of its target.
/// Cyclic input is left unchanged.
pub fn push_weights(dwa: &mut DWA) -> (bool, Option<Vec<usize>>, Vec<Weight>) {
    let Some(topo) = compute_topo_order(dwa) else { return (false, None, Vec::new()); };
    let rows = live_rows(dwa, &topo, false);
    let n = rows.len();
    let mut changed = false;
    let mut domains = Vec::with_capacity(n);
    for (state, row) in dwa.states_mut().iter_mut().zip(rows) {
        let mut edits = Vec::new();
        let mut arcs = row.arcs.into_iter().peekable();
        for (label, target, weight) in state.transitions.entries() {
            if target as usize >= n { continue; }
            if arcs.peek().is_some_and(|arc| arc.label == label) {
                let arc = arcs.next().unwrap();
                if arc.weight != *weight { edits.push((label, Some((target, arc.weight)))); }
            } else {
                edits.push((label, None));
            }
        }
        changed |= !edits.is_empty();
        for (label, replacement) in edits {
            match replacement {
                Some(edge) => { state.transitions.insert(label, edge); }
                None => { state.transitions.remove(&label); }
            }
        }
        domains.push(row.domain);
    }
    (changed, Some(topo), domains)
}

struct RowClass {
    row: LiveRow,
    members: Vec<usize>,
    domains: Vec<Weight>,
}

fn merge_height(
    bucket: Vec<usize>,
    rows: &mut [LiveRow],
    mapping: &mut [u32],
    output: &mut Vec<DWAState>,
    order: PointwiseClassOrder,
) {
    if bucket.is_empty() { return; }
    // All fully accepting leaves are compatible; batch their domain union.
    if bucket.iter().all(|&u| rows[u].arcs.is_empty() && rows[u].domain == rows[u].final_weight) {
        let id = output.len() as u32;
        let mut finals = Vec::with_capacity(bucket.len());
        for u in bucket {
            mapping[u] = id;
            finals.push(std::mem::replace(&mut rows[u], LiveRow::empty()).final_weight);
        }
        let final_weight = batch_build_weight(finals);
        output.push(DWAState { final_weight: (!final_weight.is_empty()).then_some(final_weight), ..Default::default() });
        return;
    }
    let mut classes: Vec<RowClass> = Vec::new();
    let mut signatures: FxHashMap<u64, SmallVec<[usize; 1]>> = FxHashMap::default();
    for u in bucket {
        let mut row = std::mem::replace(&mut rows[u], LiveRow::empty());
        for arc in &mut row.arcs {
            arc.target = mapping[arc.target as usize];
            debug_assert_ne!(arc.target, UNMAPPED, "live target must already be mapped");
        }
        let candidates = signatures.entry(row.signature()).or_default();
        if let Some(&id) = candidates.iter().find(|&&id| classes[id].row.same_behavior(&row)) {
            classes[id].members.push(u);
            classes[id].domains.push(row.domain);
        } else {
            candidates.push(classes.len());
            classes.push(RowClass { domains: vec![row.domain.clone()], row, members: vec![u] });
        }
    }
    for class in &mut classes {
        class.row.domain = batch_build_weight(std::mem::take(&mut class.domains));
    }
    if order == PointwiseClassOrder::DescendingDomain {
        classes.sort_by_cached_key(|class| std::cmp::Reverse((
            class.row.domain.tsid_coverage_len().unwrap_or(usize::MAX), class.row.arcs.len(),
        )));
    }
    let base = output.len() as u32;
    let mut groups: Vec<MergeGroup> = Vec::new();
    for class in classes {
        let id = groups.iter().position(|group| group.accepts(&class.row)).unwrap_or_else(|| {
            groups.push(MergeGroup::default());
            groups.len() - 1
        });
        for member in class.members { mapping[member] = base + id as u32; }
        groups[id].insert(class.row);
    }
    output.extend(groups.into_iter().map(MergeGroup::finish));
}

pub fn minimize_acyclic(dwa: &DWA) -> DWA {
    minimize_acyclic_owned(dwa.clone())
}

pub fn minimize_acyclic_owned(dwa: DWA) -> DWA {
    minimize_acyclic_owned_impl(dwa, PointwiseClassOrder::Stable, false)
}

pub fn minimize_acyclic_owned_path_conditioned(dwa: DWA) -> DWA {
    minimize_acyclic_owned_impl(dwa, PointwiseClassOrder::Stable, true)
}

pub fn minimize_acyclic_owned_with_pointwise_class_order(
    dwa: DWA,
    order: PointwiseClassOrder,
) -> DWA {
    minimize_acyclic_owned_impl(dwa, order, false)
}

fn minimize_acyclic_owned_impl(dwa: DWA, order: PointwiseClassOrder, forward: bool) -> DWA {
    if dwa.states().is_empty() { return dwa; }
    let start_time = Instant::now();
    let Some(topo) = compute_topo_order(&dwa) else { return dwa; };
    let mut rows = live_rows(&dwa, &topo, forward);
    let n = rows.len();
    let start = dwa.start_state() as usize;
    if rows.get(start).is_none_or(|row| row.domain.is_empty()) {
        return DWA::new(0, 0);
    }
    let mut reachable = vec![false; n];
    reachable[start] = true;
    for &u in &topo {
        if reachable[u] {
            for arc in &rows[u].arcs { reachable[arc.target as usize] = true; }
        }
    }
    let mut heights = vec![0usize; n];
    let mut buckets: Vec<Vec<usize>> = Vec::new();
    for &u in topo.iter().rev() {
        if !reachable[u] { continue; }
        let h = rows[u].arcs.iter().map(|arc| heights[arc.target as usize] + 1).max().unwrap_or(0);
        heights[u] = h;
        if buckets.len() <= h { buckets.resize_with(h + 1, Vec::new); }
        buckets[h].push(u);
    }
    let mut mapping = vec![UNMAPPED; n];
    let mut output = Vec::new();
    let profile = weighted_dwa_minimize_profile_enabled();
    for (height, mut bucket) in buckets.into_iter().enumerate() {
        bucket.sort_unstable();
        let count = bucket.len();
        let before = output.len();
        merge_height(bucket, &mut rows, &mut mapping, &mut output, order);
        if profile {
            eprintln!("[glrmask/profile][weighted_dwa_minimize_height] height={} candidates={} colors={}",
                height, count, output.len() - before);
        }
    }
    let minimized = DWA::from_parts(output, mapping[start]);
    if profile {
        eprintln!("[glrmask/profile][weighted_dwa_minimize] input_states={} input_transitions={} output_states={} output_transitions={} total_ms={:.3}",
            n, dwa.num_transitions(), minimized.num_states(), minimized.num_transitions(), start_time.elapsed().as_secs_f64()*1000.0);
    }
    if forward && std::env::var_os("GLRMASK_ASSERT_WEIGHTED_MINIMIZE_FORWARD_DOMAINS").is_some() {
        assert!(matches!(find_difference(&dwa, &minimized), Ok(None)),
            "path-conditioned minimization changed the weighted language");
    }
    minimized
}
#[cfg(test)]
mod tests {
    use super::*;
    fn token_set(ranges: &[(u32, u32)]) -> RangeSetBlaze<u32> {
        ranges.iter().copied().map(|(start, end)| start..=end).collect()
    }

    fn weight(entries: &[(u32, &[(u32, u32)])]) -> Weight {
        Weight::from_per_tsid_token_sets(
            entries
                .iter()
                .copied()
                .map(|(tsid, ranges)| (tsid, token_set(ranges))),
        )
    }

    #[test]
    fn reverse_hashcons_merges_only_identical_weighted_suffixes() {
        let mut states = vec![DWAState::default(); 4];
        states[0]
            .transitions
            .insert(1, (1, Weight::all()));
        states[0]
            .transitions
            .insert(2, (2, Weight::all()));
        let shared_edge = weight(&[(0, &[(1, 7)])]);
        states[1]
            .transitions
            .insert(3, (3, shared_edge.clone()));
        states[2]
            .transitions
            .insert(3, (3, shared_edge));
        states[3].final_weight = Some(weight(&[(0, &[(2, 6)])]));
        let original = DWA::from_parts(states, 0);

        let hashconsed = reverse_hashcons_acyclic_owned(original.clone());
        assert_eq!(hashconsed.num_states(), 3);
        assert_eq!(
            find_difference(&original, &hashconsed).expect("acyclic equivalence must be decidable"),
            None,
        );
    }

    #[test]
    fn large_leaf_bucket() {
        let leaf_count = 65usize;
        let mut states = Vec::with_capacity(leaf_count + 1);
        let mut start = DWAState::default();
        for leaf in 0..leaf_count {
            start.transitions.insert(
                leaf as i32,
                ((leaf + 1) as u32, Weight::all()),
            );
        }
        states.push(start);
        for _ in 0..leaf_count {
            let leaf = DWAState {
                final_weight: Some(Weight::all()),
                ..Default::default()
            };
            states.push(leaf);
        }
        let dwa = DWA::from_parts(states, 0);

        let minimized = minimize_acyclic(&dwa);
        assert_eq!(
            find_difference(&dwa, &minimized).expect("equivalence check must succeed"),
            None,
        );
        assert_eq!(minimized.states().len(), 2);
        assert_eq!(minimized.states()[minimized.start_state() as usize].transitions.len(), leaf_count);
    }

    #[test]
    fn large_nonleaf_bucket() {
        // Mirror the shape that motivated widening the total-state guard:
        // many height-0 leaves, a modest height-1 bucket, and one root.  The
        // pairwise work is controlled by the 48-state non-leaf bucket even
        // though the complete automaton has well over 128 states.
        let middle_count = 48usize;
        let leaf_count = 103usize;
        let root = 0usize;
        let first_middle = 1usize;
        let first_leaf = first_middle + middle_count;
        let total_states = first_leaf + leaf_count;
        assert!(total_states > 128 && total_states <= 192);

        let mut states = vec![DWAState::default(); total_states];
        for middle in 0..middle_count {
            states[root].transitions.insert(
                middle as i32,
                ((first_middle + middle) as u32, Weight::all()),
            );
            states[first_middle + middle].transitions.insert(
                10_000,
                ((first_leaf + middle % leaf_count) as u32, Weight::all()),
            );
        }
        for leaf in 0..leaf_count {
            states[first_leaf + leaf].final_weight = Some(Weight::all());
        }

        let dwa = DWA::from_parts(states, root as u32);
        let minimized = minimize_acyclic(&dwa);
        assert_eq!(
            find_difference(&dwa, &minimized).expect("equivalence check must succeed"),
            None,
        );
        assert!(minimized.num_states() < dwa.num_states());
    }

    #[test]
    fn reconstruction_one_sweep_matches_reduction_tree() {
        let pending: Vec<Weight> = (0..257u32)
            .map(|index| {
                Weight::from_per_tsid_token_sets(std::iter::once((
                    index % 17,
                    RangeSetBlaze::from_iter(std::iter::once((index % 31)..=(index % 31 + 2))),
                )))
            })
            .collect();

        let direct = batch_build_weight(pending.clone());
        let mut tree = pending;
        while tree.len() > 64 {
            tree = tree
                .chunks(64)
                .map(|chunk| Weight::union_all(chunk.iter()))
                .collect();
        }
        let reduction_tree = Weight::union_all(tree.iter());
        assert_eq!(direct, reduction_tree);
    }

    #[test]
    fn path_conditioned_minimizer_preserves_weighted_language() {
        // The two middle states represent the same structural residual under
        // disjoint cumulative path domains. This is the shape emitted by
        // determinizing an already backward-pushed token-conditional NWA.
        let left_domain = Weight::from_uniform(1..=1, token_set(&[(10, 19)]));
        let right_domain = Weight::from_uniform(1..=1, token_set(&[(20, 29)]));
        let all_domain = left_domain.union(&right_domain);

        let mut start = DWAState::default();
        start.transitions.insert(1, (1, left_domain.clone()));
        start.transitions.insert(2, (2, right_domain.clone()));

        let mut left = DWAState::default();
        left.transitions.insert(3, (3, left_domain.clone()));
        let mut right = DWAState::default();
        right.transitions.insert(3, (3, right_domain.clone()));

        let leaf = DWAState {
            final_weight: Some(all_domain),
            ..Default::default()
        };

        let original = DWA::from_parts(vec![start, left, right, leaf], 0);
        let minimized = minimize_acyclic_owned_path_conditioned(original.clone());
        assert_eq!(
            find_difference(&original, &minimized).unwrap(),
            None,
            "path-conditioned minimization changed the weighted language",
        );
        assert!(minimized.num_states() < original.num_states());
    }

    #[test]
    fn incoming_path_domains_do_not_certify_dead_suffixes() {
        let live = weight(&[(0, &[(1, 4)])]);
        let dead = weight(&[(0, &[(5, 8)])]);
        let mut states = vec![DWAState::default(); 5];
        states[0].transitions.insert(1, (1, live.union(&dead)));
        states[0].transitions.insert(2, (2, dead.clone()));
        states[1].transitions.insert(3, (3, live.clone()));
        states[1].transitions.insert(4, (4, dead.clone()));
        states[2].final_weight = Some(Weight::empty());
        states[3].final_weight = Some(live.clone());
        states[4].final_weight = Some(live);
        let original = DWA::from_parts(states, 0);
        let minimized = minimize_acyclic_owned_path_conditioned(original.clone());
        assert_eq!(find_difference(&original, &minimized).unwrap(), None);
        assert_eq!(minimized.num_transitions(), 2);
        assert!(minimized.states().iter().all(|s| s.final_weight.as_ref().is_none_or(|w| !w.is_empty())));
    }

    #[test]
    fn push_weights_maintains_needed_containment_invariant() {
        let mut source = DWAState::default();
        source.transitions.insert(
            42,
            (1, Weight::from_uniform(1..=1, token_set(&[(10, 20)]))),
        );
        let target = DWAState {
            final_weight: Some(Weight::from_uniform(1..=1, token_set(&[(12, 14)]))),
            ..Default::default()
        };

        let mut dwa = DWA::from_parts(vec![source, target], 0);
        let (_, topo, needed) = push_weights(&mut dwa);
        assert!(topo.is_some());

        let transition_weight = &dwa.states()[0].transitions.get(&42).unwrap().1;
        let target_final_weight = dwa.states()[1].final_weight.as_ref().unwrap();

        assert!(target_final_weight.is_subset(&needed[1]));
        assert!(transition_weight.is_subset(&needed[0]));
        assert!(transition_weight.is_subset(&needed[1]));
        assert_eq!(
            transition_weight,
            &Weight::from_uniform(1..=1, token_set(&[(12, 14)]))
        );
    }

    #[test]
    fn minimize_acyclic_merges_overlapping_partial_transition_states_exactly() {
        let branch_weights = [
            Weight::from_uniform(0..=0, token_set(&[(1, 2)])),
            Weight::from_uniform(0..=0, token_set(&[(2, 3)])),
            Weight::from_uniform(0..=0, token_set(&[(3, 4)])),
        ];
        let mut start = DWAState::default();
        for (idx, label) in [10, 11, 12].into_iter().enumerate() {
            start.transitions.insert(label, ((idx + 1) as u32, Weight::all()));
        }

        let mut states = vec![start];
        for weight in &branch_weights {
            let mut branch = DWAState::default();
            branch.transitions.insert(20, (4, weight.clone()));
            states.push(branch);
        }
        let leaf = DWAState {
            final_weight: Some(Weight::all()),
            ..Default::default()
        };
        states.push(leaf);
        let dwa = DWA::from_parts(states, 0);

        let words = [[10, 20], [11, 20], [12, 20], [10, 21], [11, 21], [12, 21]];
        let expected = words.map(|word| dwa.eval_word(&word));
        let minimized = minimize_acyclic(&dwa);

        assert_eq!(minimized.num_states(), 3);
        for (word, expected) in words.into_iter().zip(expected) {
            assert_eq!(minimized.eval_word(&word), expected, "word={word:?}");
        }
    }

    fn seeded_dag_input(seed: u64) -> DWA {
        let mut rng = seed.wrapping_add(1);
        let mut next = move || {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            rng = rng.wrapping_mul(0x2545_f491_4f6c_dd1d);
            rng
        };
        let n_states = 6 + (next() % 7) as usize; // 6..=12
        let mut states = vec![DWAState::default(); n_states];
        let outer_domains: [&[(u32, &[(u32, u32)])]; 3] = [
            &[(0, &[(1, 8)]), (1, &[(3, 10)])],
            &[(0, &[(5, 12)]), (2, &[(1, 4)])],
            &[(1, &[(1, 6)]), (2, &[(7, 15)])],
        ];
        for (i, state) in states.iter_mut().enumerate() {
            let mut label = 1i32;
            while label <= 4 {
                if next() % 4 != 0 && i + 1 < n_states {
                    let span = (next() % (n_states - i - 1) as u64) as usize;
                    let target = i + 1 + span;
                    let dom = outer_domains[(next() % 3) as usize];
                    let w = if next() % 2 == 0 {
                        weight(dom)
                    } else {
                        weight(&[(0, &[(4, 9)]), (1, &[(8, 13)])])
                    };
                    state.transitions.insert(label, (target as u32, w));
                }
                label += 1;
            }
            if next() % 3 == 0 {
                state.final_weight = Some(weight(&[(0, &[(2, 5)]), (2, &[(9, 11)])]));
            }
            if next() % 5 == 0 {
                state.final_weight = Some(weight(&[(1, &[(1, 3)]), (3, &[(20, 25)])]));
            }
        }
        // Accept spine: label-5 chain with uniform weight plus matching final,
        // so every input has nonempty language by construction.
        let spine = weight(&[(0, &[(1, 8)])]);
        for (i, state) in states.iter_mut().enumerate().take(n_states - 1) {
            state.transitions.insert(5, (i as u32 + 1, spine.clone()));
        }
        states[n_states - 1].final_weight = Some(spine);
        let input = DWA::from_parts(states, 0);
        assert!(
            !input.eval_word(&vec![5; n_states - 1]).is_empty(),
            "seed {seed}: spine must accept",
        );
        input
    }

    fn assert_orders_preserve_language(seed: u64, input: &DWA) {
        use super::{minimize_acyclic_owned_with_pointwise_class_order, PointwiseClassOrder};
        let stable = minimize_acyclic_owned_with_pointwise_class_order(
            input.clone(),
            PointwiseClassOrder::Stable,
        );
        let descending = minimize_acyclic_owned_with_pointwise_class_order(
            input.clone(),
            PointwiseClassOrder::DescendingDomain,
        );
        for (name, minimized) in [("stable", &stable), ("descending", &descending)] {
            match find_difference(input, minimized) {
                Ok(None) => {}
                Ok(Some(word)) => {
                    panic!("seed {seed} {name}: differs from input at word {word:?}")
                }
                Err(e) => panic!("seed {seed} {name}: checker error: {e:?}"),
            }
        }
        match find_difference(&stable, &descending) {
            Ok(None) => {}
            Ok(Some(word)) => {
                panic!("seed {seed}: stable vs descending differ at word {word:?}")
            }
            Err(e) => panic!("seed {seed}: order-pair checker error: {e:?}"),
        }
    }

    #[test]
    fn minimize_pointwise_orders_preserve_language_seeded_family() {
        for seed in 0..64u64 {
            assert_orders_preserve_language(seed, &seeded_dag_input(seed));
        }
    }

    #[test]
    fn minimize_pointwise_orders_preserve_language_overlapping_partial_states() {
        // Two states share label 1 with overlapping (not equal) weights and
        // divergent continuations; label 2 missing from one; distinct finals.
        let mut states = vec![DWAState::default(); 5];
        states[0].transitions.insert(1, (1, weight(&[(0, &[(1, 8)])])));
        states[0].transitions.insert(2, (2, weight(&[(0, &[(1, 8)])])));
        states[1].transitions.insert(1, (3, weight(&[(0, &[(5, 12)])])));
        states[1].transitions.insert(3, (4, weight(&[(1, &[(1, 6)])])));
        states[2].transitions.insert(1, (3, weight(&[(0, &[(1, 4)])])));
        states[2].transitions.insert(4, (4, weight(&[(2, &[(7, 15)])])));
        states[3].final_weight = Some(weight(&[(0, &[(2, 5)]), (2, &[(9, 11)])]));
        states[4].final_weight = Some(weight(&[(1, &[(1, 3)]), (3, &[(20, 25)])]));
        assert_orders_preserve_language(u64::MAX, &DWA::from_parts(states, 0));
    }

    #[test]
    fn restricted_agreement_matches_materialized_intersections() {
        let weights = [
            Weight::empty(), Weight::all(),
            weight(&[(0, &[(0, 4), (6, 9)]), (3, &[(10, 12)])]),
            weight(&[(0, &[(3, 7)]), (2, &[(1, 3)]), (3, &[(11, 14)])]),
            Weight::from_uniform(1..=u32::MAX, token_set(&[(0, 1), (u32::MAX, u32::MAX)])),
            Weight::from_uniform(0..=u32::MAX, token_set(&[(1, u32::MAX)])),
        ];
        for da in &weights {
            for db in &weights {
                for a in &weights {
                    for b in &weights {
                        let a = a.intersection(da);
                        let b = b.intersection(db);
                        assert_eq!(restricted_weights_agree(&a, &b, da, db),
                            a.intersection(db) == b.intersection(da));
                    }
                }
            }
        }
    }

    fn test_row(domain: Weight, final_weight: Weight, arcs: &[(Label, u32, Weight)]) -> LiveRow {
        LiveRow { domain, final_weight, arcs: arcs.iter().map(|(label, target, weight)| ArcRow {
            label: *label, target: *target, weight: weight.clone(),
        }).collect() }
    }

    #[test]
    fn union_summary_preserves_memberwise_agreement_and_determinism() {
        let mut group = MergeGroup::default();
        let mut domains = Vec::new();
        for i in 0..65 {
            let domain = Weight::from_uniform(i..=i, token_set(&[(1, 4)]));
            let row = test_row(domain.clone(), Weight::empty(), &[(5, 7, domain.clone())]);
            assert!(group.accepts(&row));
            group.insert(row);
            domains.push(domain);
        }
        let overlap = Weight::from_uniform(12..=12, token_set(&[(3, 6)]));
        let compatible = test_row(overlap.clone(), Weight::empty(), &[(5, 7, overlap.clone())]);
        assert!(group.accepts(&compatible));
        assert!(!group.accepts(&test_row(overlap.clone(), Weight::empty(), &[])));
        assert!(!group.accepts(&test_row(overlap.clone(), overlap.clone(), &[(5, 7, overlap.clone())])));
        let disjoint = Weight::from_uniform(100..=100, token_set(&[(1, 4)]));
        assert!(!group.accepts(&test_row(disjoint.clone(), Weight::empty(), &[(5, 8, disjoint)])));
        group.insert(compatible);
        domains.push(overlap);
        let state = group.finish();
        assert_eq!(state.transitions.get(&5).unwrap().1, batch_build_weight(domains));
    }
}
