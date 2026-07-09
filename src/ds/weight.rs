use range_set_blaze::{CheckSortedDisjoint, RangeMapBlaze, RangeSetBlaze, SortedDisjointMap};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;
use dashmap::DashMap;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

// STICKY NOTE: DO NOT REMOVE THIS COMMENT.
//
// This module uses RangeSetBlaze and RangeMapBlaze as its core data structures.
// Performance characteristics are counterintuitive and differ from naive bitmaps
// or hash maps. Read this note before writing hot-path code that creates,
// mutates, or queries Weight objects.
//
// 1. RangeSetBlaze / RangeMapBlaze complexity
//    Cost is proportional to the NUMBER OF RANGES, not the numeric span covered.
//    A set covering 0..=1_000_000 as a single range is much cheaper than 1_000_001
//    individual points stored as 1_000_001 singleton ranges.
//    RangeMapBlaze merges adjacent keys only when the VALUES are also equal.
//    A Weight with 10 key-ranges each mapping to a 3-range token set can be
//    more expensive than a Weight with 50 key-ranges each mapping to a 1-range
//    token set, depending on operation mix.
//
// 2. Why remapping / rearranging IDs matters
//    If many Weights will be stored or queried together, arrange numeric IDs
//    so that each Weight's inner sets/maps form FEWER TOTAL RANGES.
//    The target is NOT "small max ID"; it is fewer ranges across the relevant
//    weights. In DWA/id-map/possible-matches-style code this means:
//    - Group IDs that co-occur in the same token sets into contiguous ranges.
//    - Recompact parser-state and terminal IDs jointly with token vocab IDs
//      when the weights are built together.
//    - A renumbering that cuts total unique ranges by 2x often beats a
//      renumbering that merely lowers the max ID.
//
// 3. Two-level interning
//    Outer Weight maps (RangeMapBlaze<u32, SharedTokenSet>) are interned via
//    GLOBAL_WEIGHTS. Inner RangeSetBlaze values are interned via
//    GLOBAL_TOKEN_SETS. Both use Arc deduplication.
//    When measuring static complexity of a collection of weights, count UNIQUE
//    interned weights once, and count UNIQUE inner rangesets once, not once per
//    key-range occurrence. Example: if the same weight appears 100 times in a
//    collection of 101 weights, its static cost is roughly that weight plus the
//    1 other unique weight, plus their shared inner rangesets. Do not multiply by 100.
//
//    Static complexity model for DWA / possible-matches recompaction:
//    - For a unique Weight, count its unique outer key ranges, plus the ranges
//      in each unique interned inner RangeSetBlaze (counted once per unique
//      inner set, not once per reference).
//    - For a collection of weights, count each unique interned Weight once and
//      each unique inner set once.
//    - This is the mental model behind minimizing
//      total_outer_ranges + total_inner_ranges in DWA::stats()-style accounting.
//
// 4. Mutation pitfalls
//    Small repeated mutations (insert one token, union one range) can be
//    surprisingly expensive because each operation may trigger normalization
//    and intern-table lookup. In hot paths, avoid building a Weight one item
//    at a time. Prefer construction APIs that collect from sorted/ranged data
//    (e.g. CompactRangeBuilder, from_per_tsid_shared)
//    so normalization and interning happen once at the end.
//
// 5. Lookup / iteration implications
//    Lookups are O(log num_key_ranges). Iteration yields ranges, not individual
//    points. Union / intersection iterate over both operands' ranges in lockstep.
//    The cheap case is when both operands have very few ranges or are the same
//    interned Arc (fast path: Arc::ptr_eq). The expensive case is many
//    misaligned small ranges on both sides.
//    Cloning is cheap because it is just an Arc clone of the interned map, but
//    only until you mutate; then a fresh normalized map must be built.
//
// 6. Practical guidance
//    - Prefer dense contiguous IDs for things that co-occur in the same sets.
//    - Recompact / remap based on the WHOLE COLLECTION of weights that will be
//      queried or stored together, not per-weight in isolation.
//    - When designing optimizers, consider the interning boundary: reducing
//      total unique interned weights and unique inner rangesets is often more
//      valuable than shrinking any single weight's local range count.
//    - If you must measure, measure total unique interned structures and total
//      ranges across the representative workload, not max ID or per-weight size.
//
// DO NOT REMOVE THIS NOTE. Future maintainers will need it.

#[derive(Debug, Clone)]
pub struct Weight(pub Arc<WeightMap>);

#[derive(Default)]
pub struct ScopedWeightOpCache {
    union_entries: FxHashMap<(usize, usize), Weight>,
    intersection_entries: FxHashMap<(usize, usize), Weight>,
    difference_entries: FxHashMap<(usize, usize), Weight>,
}

impl ScopedWeightOpCache {
    pub fn union(&mut self, left: &Weight, right: &Weight) -> Weight {
        if left.is_full() || right.is_full() {
            return Weight::all();
        }
        if left.is_empty() {
            return right.clone();
        }
        if right.is_empty() {
            return left.clone();
        }
        if Arc::ptr_eq(&left.0, &right.0) {
            return left.clone();
        }

        let key = scoped_commutative_weight_pair_key(left, right);
        if let Some(existing) = self.union_entries.get(&key) {
            return existing.clone();
        }

        let value = left.union_uncached(right);
        self.union_entries.insert(key, value.clone());
        value
    }

    pub fn intersection(&mut self, left: &Weight, right: &Weight) -> Weight {
        if left.is_empty() || right.is_empty() {
            return Weight::empty();
        }
        if Arc::ptr_eq(&left.0, &right.0) {
            return left.clone();
        }
        if left.is_full() {
            return right.clone();
        }
        if right.is_full() {
            return left.clone();
        }

        let key = scoped_commutative_weight_pair_key(left, right);
        if let Some(existing) = self.intersection_entries.get(&key) {
            return existing.clone();
        }

        let value = left.intersection_uncached(right);
        self.intersection_entries.insert(key, value.clone());
        value
    }

    pub(crate) fn intersection_entry_count(&self) -> usize {
        self.intersection_entries.len()
    }

    pub fn difference(&mut self, left: &Weight, right: &Weight) -> Weight {
        if left.is_empty() || right.is_full() {
            return Weight::empty();
        }
        if right.is_empty() {
            return left.clone();
        }
        if Arc::ptr_eq(&left.0, &right.0) {
            return Weight::empty();
        }
        if left.is_full() {
            return Weight::all();
        }

        let key = (left.ptr_key(), right.ptr_key());
        if let Some(existing) = self.difference_entries.get(&key) {
            return existing.clone();
        }

        let value = left.difference_uncached(right);
        self.difference_entries.insert(key, value.clone());
        value
    }

    /// Scoped bulk intersection with mathematical identity semantics.
    /// An empty input intersects to `Weight::all()`.
    pub fn intersection_all<'a>(&mut self, weights: impl IntoIterator<Item = &'a Weight>) -> Weight {
        let mut meaningful = SmallVec::<[&Weight; 8]>::new();
        for weight in weights {
            if weight.is_empty() {
                return Weight::empty();
            }
            if weight.is_full() {
                continue;
            }
            meaningful.push(weight);
        }

        match meaningful.len() {
            0 => Weight::all(),
            1 => meaningful[0].clone(),
            _ => {
                meaningful.sort_unstable_by_key(|weight| weight.ptr_key());
                meaningful.dedup_by_key(|weight| weight.ptr_key());
                let mut iter = meaningful.into_iter();
                let mut acc = iter.next().unwrap().clone();
                for weight in iter {
                    acc = self.intersection(&acc, weight);
                    if acc.is_empty() {
                        return acc;
                    }
                }
                acc
            }
        }
    }

    pub fn union_all<'a>(&mut self, weights: impl IntoIterator<Item = &'a Weight>) -> Weight {
        let mut meaningful = SmallVec::<[&Weight; 8]>::new();
        for weight in weights {
            if weight.is_full() {
                return Weight::all();
            }
            if weight.is_empty() {
                continue;
            }
            meaningful.push(weight);
        }

        match meaningful.len() {
            0 => Weight::empty(),
            1 => meaningful[0].clone(),
            _ if meaningful.len() > 4 => {
                meaningful.sort_unstable_by_key(|weight| weight.ptr_key());
                meaningful.dedup_by_key(|weight| weight.ptr_key());
                if meaningful.len() == 1 {
                    meaningful[0].clone()
                } else if let Some(result) = union_all_single_tsid_entries(&meaningful) {
                    result
                } else if meaningful.len() > 4 {
                    union_all_multiway(&meaningful)
                } else {
                    let mut iter = meaningful.into_iter();
                    let mut acc = iter.next().unwrap().clone();
                    for weight in iter {
                        acc = self.union(&acc, weight);
                    }
                    acc
                }
            }
            _ => {
                if let Some(result) = union_all_single_tsid_entries(&meaningful) {
                    result
                } else {
                    let mut iter = meaningful.into_iter();
                    let mut acc = iter.next().unwrap().clone();
                    for weight in iter {
                        acc = self.union(&acc, weight);
                    }
                    acc
                }
            }
        }
    }

    pub fn difference_many<'a>(
        &mut self,
        base: &Weight,
        subtracts: impl IntoIterator<Item = &'a Weight>,
    ) -> Weight {
        let mut acc = base.clone();
        for weight in subtracts {
            acc = self.difference(&acc, weight);
            if acc.is_empty() {
                return acc;
            }
        }
        acc
    }
}

pub(crate) type SharedTokenSet = Arc<RangeSetBlaze<u32>>;
type WeightMap = RangeMapBlaze<u32, SharedTokenSet>;

const INTERNER_CLEANUP_INTERVAL: usize = 1024;

// Sharded interner: DashMap provides internal striping (~16 shards) so concurrent
// intern calls can run in parallel on different keys. Previously we used a single
// `Mutex<GlobalWeightInterner>` which serialized all weight-op fresh constructions.
static GLOBAL_TOKEN_SETS: Lazy<DashMap<RangeSetBlaze<u32>, Weak<RangeSetBlaze<u32>>>> =
    Lazy::new(DashMap::new);
static GLOBAL_WEIGHTS: Lazy<DashMap<u64, Vec<Weak<WeightMap>>>> = Lazy::new(DashMap::new);
static TOKEN_INSERTS_SINCE_CLEANUP: AtomicUsize = AtomicUsize::new(0);
static WEIGHT_INSERTS_SINCE_CLEANUP: AtomicUsize = AtomicUsize::new(0);
static TOKEN_CLEANUP_IN_PROGRESS: AtomicBool = AtomicBool::new(false);
static WEIGHT_CLEANUP_IN_PROGRESS: AtomicBool = AtomicBool::new(false);
static INTERNER_CLEANUP_DEFERRAL_DEPTH: AtomicUsize = AtomicUsize::new(0);

static EMPTY_RANGESET: Lazy<SharedTokenSet> = Lazy::new(|| Arc::new(RangeSetBlaze::new()));

static EMPTY_WEIGHT: Lazy<Weight> = Lazy::new(|| Weight(Arc::new(WeightMap::new())));

static ALL_WEIGHT: Lazy<Weight> = Lazy::new(|| {
    let mut map = WeightMap::new();
    map.extend_simple(std::iter::once((
        WEIGHT_ALL_SENTINEL..=WEIGHT_ALL_SENTINEL,
        shared_rangeset(sentinel_token_set()),
    )));
    finalize_weight_map(map)
});

/// Defers expensive global interner sweeps until the last active compiler
/// exits. Fresh interning remains exact; only the timing of weak-entry removal
/// changes.
pub(crate) struct WeightInternerCleanupDeferral {
    active: bool,
}

pub(crate) fn defer_weight_interner_cleanup() -> WeightInternerCleanupDeferral {
    INTERNER_CLEANUP_DEFERRAL_DEPTH.fetch_add(1, Ordering::AcqRel);
    WeightInternerCleanupDeferral { active: true }
}

impl WeightInternerCleanupDeferral {
    pub(crate) fn finish(mut self) {
        self.release();
    }

    fn release(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        if INTERNER_CLEANUP_DEFERRAL_DEPTH.fetch_sub(1, Ordering::AcqRel) == 1 {
            interner_clear_stale();
        }
    }
}

impl Drop for WeightInternerCleanupDeferral {
    fn drop(&mut self) {
        self.release();
    }
}

fn prune_dead_token_sets() {
    GLOBAL_TOKEN_SETS.retain(|_, weak| weak.strong_count() > 0);
}

fn prune_dead_weights() {
    GLOBAL_WEIGHTS.retain(|_, bucket| {
        bucket.retain(|weak| weak.strong_count() > 0);
        !bucket.is_empty()
    });
}

/// Count a fresh interner insertion and claim a global sweep exactly once.
///
/// `DashMap::retain` scans every shard. A completed cleanup resets the counter,
/// but producers may cross another interval while that scan is still running.
/// The in-progress gate ensures those later producers never start an overlapping
/// scan; after the owner releases the gate, the accumulated counter makes a
/// subsequent insertion schedule the next sweep normally.
#[inline]
fn claim_interner_cleanup(counter: &AtomicUsize, in_progress: &AtomicBool) -> bool {
    let previous = counter.fetch_add(1, Ordering::Relaxed);
    if previous + 1 < INTERNER_CLEANUP_INTERVAL {
        return false;
    }
    if in_progress
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
        .is_err()
    {
        return false;
    }
    counter.store(0, Ordering::Release);
    true
}

fn maybe_cleanup_token_sets() {
    if INTERNER_CLEANUP_DEFERRAL_DEPTH.load(Ordering::Acquire) != 0 {
        return;
    }
    if claim_interner_cleanup(&TOKEN_INSERTS_SINCE_CLEANUP, &TOKEN_CLEANUP_IN_PROGRESS) {
        prune_dead_token_sets();
        TOKEN_CLEANUP_IN_PROGRESS.store(false, Ordering::Release);
    }
}

fn maybe_cleanup_weights() {
    if INTERNER_CLEANUP_DEFERRAL_DEPTH.load(Ordering::Acquire) != 0 {
        return;
    }
    if claim_interner_cleanup(&WEIGHT_INSERTS_SINCE_CLEANUP, &WEIGHT_CLEANUP_IN_PROGRESS) {
        prune_dead_weights();
        WEIGHT_CLEANUP_IN_PROGRESS.store(false, Ordering::Release);
    }
}

fn interner_clear_all() {
    GLOBAL_TOKEN_SETS.clear();
    GLOBAL_WEIGHTS.clear();
    TOKEN_INSERTS_SINCE_CLEANUP.store(0, Ordering::Relaxed);
    WEIGHT_INSERTS_SINCE_CLEANUP.store(0, Ordering::Relaxed);
    TOKEN_CLEANUP_IN_PROGRESS.store(false, Ordering::Release);
    WEIGHT_CLEANUP_IN_PROGRESS.store(false, Ordering::Release);
}

fn interner_clear_stale() {
    prune_dead_token_sets();
    prune_dead_weights();
    TOKEN_INSERTS_SINCE_CLEANUP.store(0, Ordering::Relaxed);
    WEIGHT_INSERTS_SINCE_CLEANUP.store(0, Ordering::Relaxed);
    TOKEN_CLEANUP_IN_PROGRESS.store(false, Ordering::Release);
    WEIGHT_CLEANUP_IN_PROGRESS.store(false, Ordering::Release);
}


fn intern_rangeset(tokens: RangeSetBlaze<u32>) -> SharedTokenSet {
    if tokens.is_empty() {
        return Arc::clone(&EMPTY_RANGESET);
    }

    // Use `entry` to obtain exclusive access to this shard for the given key.
    use dashmap::mapref::entry::Entry;
    let shared = match GLOBAL_TOKEN_SETS.entry(tokens) {
        Entry::Occupied(mut occupied) => {
            if let Some(existing) = occupied.get().upgrade() {
                return existing;
            }
            // Stale weak: replace with new Arc.
            let key = occupied.key().clone();
            let shared = Arc::new(key);
            occupied.insert(Arc::downgrade(&shared));
            shared
        }
        Entry::Vacant(vacant) => {
            let key = vacant.key().clone();
            let shared = Arc::new(key);
            vacant.insert(Arc::downgrade(&shared));
            shared
        }
    };
    // NOTE: `shared` was created after the Entry guard was released (match returned),
    // but we still must not call cleanup while holding any DashMap guard. The match
    // arms drop their guards before returning, so calling cleanup here is safe.
    maybe_cleanup_token_sets();
    shared
}

fn weight_map_fingerprint(map: &WeightMap) -> u64 {
    use std::hash::Hasher;

    let mut hasher = rustc_hash::FxHasher::default();
    for (range, tokens) in map.range_values() {
        hasher.write_u32(*range.start());
        hasher.write_u32(*range.end());
        hasher.write_usize(Arc::as_ptr(tokens) as usize);
    }
    hasher.finish()
}

fn weight_map_eq(left: &WeightMap, right: &WeightMap) -> bool {
    let mut left_iter = left.range_values();
    let mut right_iter = right.range_values();
    loop {
        match (left_iter.next(), right_iter.next()) {
            (None, None) => return true,
            (Some((left_range, left_tokens)), Some((right_range, right_tokens))) => {
                if left_range != right_range {
                    return false;
                }
                if !same_shared_token_set(left_tokens, right_tokens) {
                    return false;
                }
            }
            _ => return false,
        }
    }
}

fn intern_weight_map(map: WeightMap) -> Arc<WeightMap> {
    let fingerprint = weight_map_fingerprint(&map);
    let mut bucket = GLOBAL_WEIGHTS.entry(fingerprint).or_default();
    let mut idx = 0usize;
    while idx < bucket.len() {
        let Some(existing) = bucket[idx].upgrade() else {
            bucket.swap_remove(idx);
            continue;
        };
        if weight_map_eq(existing.as_ref(), &map) {
            return existing;
        }
        idx += 1;
    }
    let shared = Arc::new(map);
    bucket.push(Arc::downgrade(&shared));
    drop(bucket);
    maybe_cleanup_weights();
    shared
}

fn same_shared_token_set(left: &SharedTokenSet, right: &SharedTokenSet) -> bool {
    Arc::ptr_eq(left, right) || left.as_ref() == right.as_ref()
}

fn lookup_memoized_token_set_op(
    kind: TokenSetOpKind,
    left: &SharedTokenSet,
    right: &SharedTokenSet,
) -> Option<SharedTokenSet> {
    with_weight_op_memo(|memo| memo.lookup_token_set(TokenSetOpKey::for_token_sets(kind, left, right)))
}

fn store_memoized_token_set_op(
    kind: TokenSetOpKind,
    left: &SharedTokenSet,
    right: &SharedTokenSet,
    result: &SharedTokenSet,
) {
    with_weight_op_memo(|memo| {
        memo.store_token_set(TokenSetOpKey::for_token_sets(kind, left, right), left, right, result)
    });
}

fn shared_token_union(left: &SharedTokenSet, right: &SharedTokenSet) -> SharedTokenSet {
    if same_shared_token_set(left, right) || left.as_ref().is_subset(right.as_ref()) {
        Arc::clone(right)
    } else if right.as_ref().is_subset(left.as_ref()) {
        Arc::clone(left)
    } else if let Some(existing) = lookup_memoized_token_set_op(TokenSetOpKind::Union, left, right) {
        existing
    } else {
        let result = shared_rangeset(left.as_ref().clone() | right.as_ref().clone());
        store_memoized_token_set_op(TokenSetOpKind::Union, left, right, &result);
        result
    }
}

fn shared_token_union_many(tokens: &[SharedTokenSet]) -> Option<SharedTokenSet> {
    match tokens.len() {
        0 => None,
        1 => Some(Arc::clone(&tokens[0])),
        2 => Some(shared_token_union(&tokens[0], &tokens[1])),
        _ => {
            let mut ranges = Vec::<(u32, u32)>::new();
            for token_set in tokens {
                ranges.extend(
                    token_set
                        .ranges()
                        .map(|range| (*range.start(), *range.end())),
                );
            }
            if ranges.is_empty() {
                return None;
            }
            ranges.sort_unstable();

            let mut merged = Vec::with_capacity(ranges.len());
            let mut current = ranges[0];
            for (start, end) in ranges.into_iter().skip(1) {
                if start <= current.1.saturating_add(1) {
                    current.1 = current.1.max(end);
                } else {
                    merged.push(current.0..=current.1);
                    current = (start, end);
                }
            }
            merged.push(current.0..=current.1);

            Some(shared_rangeset(RangeSetBlaze::from_iter(merged)))
        }
    }
}

fn shared_token_intersection(
    left: &SharedTokenSet,
    right: &SharedTokenSet,
) -> Option<SharedTokenSet> {
    if same_shared_token_set(left, right) || left.as_ref().is_subset(right.as_ref()) {
        Some(Arc::clone(left))
    } else if right.as_ref().is_subset(left.as_ref()) {
        Some(Arc::clone(right))
    } else if let Some(existing) =
        lookup_memoized_token_set_op(TokenSetOpKind::Intersection, left, right)
    {
        (!existing.is_empty()).then_some(existing)
    } else {
        let overlap = left.as_ref() & right.as_ref();
        let result = shared_rangeset(overlap);
        store_memoized_token_set_op(TokenSetOpKind::Intersection, left, right, &result);
        (!result.is_empty()).then_some(result)
    }
}

fn shared_token_difference(
    left: &SharedTokenSet,
    right: &SharedTokenSet,
) -> Option<SharedTokenSet> {
    if same_shared_token_set(left, right) || left.as_ref().is_subset(right.as_ref()) {
        None
    } else if left.as_ref().is_disjoint(right.as_ref()) {
        Some(Arc::clone(left))
    } else {
        let difference = left.as_ref().clone() - right.as_ref().clone();
        (!difference.is_empty()).then(|| shared_rangeset(difference))
    }
}

fn union_token_sets(
    left: Option<&SharedTokenSet>,
    right: Option<&SharedTokenSet>,
) -> Option<SharedTokenSet> {
    match (left, right) {
        (Some(left_tokens), Some(right_tokens)) => {
            Some(shared_token_union(left_tokens, right_tokens))
        }
        (Some(tokens), None) | (None, Some(tokens)) => Some(Arc::clone(tokens)),
        (None, None) => None,
    }
}

fn intersect_token_sets(
    left: Option<&SharedTokenSet>,
    right: Option<&SharedTokenSet>,
) -> Option<SharedTokenSet> {
    match (left, right) {
        (Some(left_tokens), Some(right_tokens)) => {
            shared_token_intersection(left_tokens, right_tokens)
        }
        _ => None,
    }
}

fn difference_token_sets(
    left: Option<&SharedTokenSet>,
    right: Option<&SharedTokenSet>,
) -> Option<SharedTokenSet> {
    match (left, right) {
        (Some(left_tokens), Some(right_tokens)) => {
            shared_token_difference(left_tokens, right_tokens)
        }
        (Some(left_tokens), None) => Some(Arc::clone(left_tokens)),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum WeightOpKind {
    Union,
    Intersection,
    Difference,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum TokenSetOpKind {
    Union,
    Intersection,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct WeightOpKey {
    kind: WeightOpKind,
    left: usize,
    right: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct TokenSetOpKey {
    kind: TokenSetOpKind,
    left: usize,
    right: usize,
}

#[inline]
fn scoped_commutative_weight_pair_key(left: &Weight, right: &Weight) -> (usize, usize) {
    let left_key = left.ptr_key();
    let right_key = right.ptr_key();
    if left_key <= right_key {
        (left_key, right_key)
    } else {
        (right_key, left_key)
    }
}

#[inline]
fn ordered_commutative_weight_pair<'a>(
    left: &'a Weight,
    right: &'a Weight,
) -> ((usize, usize), &'a Weight, &'a Weight) {
    let left_key = left.ptr_key();
    let right_key = right.ptr_key();
    if left_key <= right_key {
        ((left_key, right_key), left, right)
    } else {
        ((right_key, left_key), right, left)
    }
}

impl WeightOpKey {
    fn new(kind: WeightOpKind, left: usize, right: usize) -> Self {
        match kind {
            WeightOpKind::Union | WeightOpKind::Intersection if left > right => Self {
                kind,
                left: right,
                right: left,
            },
            _ => Self { kind, left, right },
        }
    }

    fn for_weights(kind: WeightOpKind, left: &Weight, right: &Weight) -> Self {
        Self::new(kind, left.ptr_key(), right.ptr_key())
    }
}

impl TokenSetOpKey {
    fn new(kind: TokenSetOpKind, left: usize, right: usize) -> Self {
        if left > right {
            Self {
                kind,
                left: right,
                right: left,
            }
        } else {
            Self { kind, left, right }
        }
    }

    fn for_token_sets(kind: TokenSetOpKind, left: &SharedTokenSet, right: &SharedTokenSet) -> Self {
        Self::new(kind, Arc::as_ptr(left) as usize, Arc::as_ptr(right) as usize)
    }
}

/// Cached memo entry: stores the result AND weak references to both operands.
/// The operand weak refs guard against the ABA problem: if either operand's
/// Arc was dropped and a new Arc reuses the same address, the operand weak
/// ref will fail to upgrade, and the stale entry is discarded.
struct WeightOpMemoEntry {
    result: Weak<WeightMap>,
    left_operand: Weak<WeightMap>,
    right_operand: Weak<WeightMap>,
}

struct TokenSetOpMemoEntry {
    result: Weak<RangeSetBlaze<u32>>,
    left_operand: Weak<RangeSetBlaze<u32>>,
    right_operand: Weak<RangeSetBlaze<u32>>,
}

/// Global generation counter. Incremented by `clear_weight_op_caches()` so
/// that every thread's thread-local memo detects the invalidation on next access.
static WEIGHT_OP_MEMO_GENERATION: AtomicU64 = AtomicU64::new(0);
static WEIGHT_HASH_MEMO_GENERATION: AtomicU64 = AtomicU64::new(0);
const PUBLIC_WEIGHT_INTERSECTION_CACHE_CAP: usize = 262_144;

struct PublicWeightIntersectionMemoEntry {
    left: Weight,
    right: Weight,
    result: Weight,
}

#[derive(Default)]
struct PublicWeightIntersectionMemo {
    results: FxHashMap<(usize, usize), PublicWeightIntersectionMemoEntry>,
    generation: u64,
}

impl PublicWeightIntersectionMemo {
    fn clear_all(&mut self) {
        self.results.clear();
    }

    fn lookup(&mut self, left: &Weight, right: &Weight) -> Option<Weight> {
        let (key, ordered_left, ordered_right) = ordered_commutative_weight_pair(left, right);
        let entry = self.results.get(&key)?;
        if Arc::ptr_eq(&entry.left.0, &ordered_left.0)
            && Arc::ptr_eq(&entry.right.0, &ordered_right.0)
        {
            return Some(entry.result.clone());
        }
        self.results.remove(&key);
        None
    }

    fn store(&mut self, left: &Weight, right: &Weight, result: &Weight) {
        if self.results.len() >= PUBLIC_WEIGHT_INTERSECTION_CACHE_CAP {
            self.clear_all();
        }
        let (key, ordered_left, ordered_right) = ordered_commutative_weight_pair(left, right);
        self.results.insert(
            key,
            PublicWeightIntersectionMemoEntry {
                left: ordered_left.clone(),
                right: ordered_right.clone(),
                result: result.clone(),
            },
        );
    }
}

#[derive(Default)]
struct WeightOpMemo {
    results: FxHashMap<WeightOpKey, WeightOpMemoEntry>,
    token_set_results: FxHashMap<TokenSetOpKey, TokenSetOpMemoEntry>,
    inserts_since_cleanup: usize,
    /// The generation this memo was last synchronised with.
    generation: u64,
}

impl WeightOpMemo {
    fn maybe_cleanup(&mut self) {
        if self.inserts_since_cleanup < INTERNER_CLEANUP_INTERVAL {
            return;
        }
        self.results.retain(|_, entry| entry.result.strong_count() > 0);
        self.token_set_results
            .retain(|_, entry| entry.result.strong_count() > 0);
        self.inserts_since_cleanup = 0;
    }

    fn clear_all(&mut self) {
        self.results.clear();
        self.token_set_results.clear();
        self.inserts_since_cleanup = 0;
    }

    fn lookup(&mut self, key: WeightOpKey) -> Option<Weight> {
        let entry = self.results.get(&key)?;
        if entry.left_operand.strong_count() == 0 || entry.right_operand.strong_count() == 0 {
            self.results.remove(&key);
            return None;
        }

        entry.result.upgrade().map(Weight)
    }

    fn store(&mut self, key: WeightOpKey, left: &Weight, right: &Weight, result: &Weight) {
        self.maybe_cleanup();
        self.results.insert(
            key,
            WeightOpMemoEntry {
                result: Arc::downgrade(&result.0),
                left_operand: Arc::downgrade(&left.0),
                right_operand: Arc::downgrade(&right.0),
            },
        );
        self.inserts_since_cleanup += 1;
    }

    fn lookup_token_set(&mut self, key: TokenSetOpKey) -> Option<SharedTokenSet> {
        let entry = self.token_set_results.get(&key)?;
        if entry.left_operand.strong_count() == 0 || entry.right_operand.strong_count() == 0 {
            self.token_set_results.remove(&key);
            return None;
        }

        entry.result.upgrade()
    }

    fn store_token_set(
        &mut self,
        key: TokenSetOpKey,
        left: &SharedTokenSet,
        right: &SharedTokenSet,
        result: &SharedTokenSet,
    ) {
        self.maybe_cleanup();
        self.token_set_results.insert(
            key,
            TokenSetOpMemoEntry {
                result: Arc::downgrade(result),
                left_operand: Arc::downgrade(left),
                right_operand: Arc::downgrade(right),
            },
        );
        self.inserts_since_cleanup += 1;
    }
}

thread_local! {
    static WEIGHT_OP_MEMO: RefCell<WeightOpMemo> = RefCell::new(WeightOpMemo::default());
}

thread_local! {
    static PUBLIC_WEIGHT_INTERSECTION_MEMO: RefCell<PublicWeightIntersectionMemo> =
        RefCell::new(PublicWeightIntersectionMemo::default());
}

fn with_weight_op_memo<R>(f: impl FnOnce(&mut WeightOpMemo) -> R) -> R {
    WEIGHT_OP_MEMO.with(|memo| {
        let mut memo = memo.borrow_mut();
        let current_gen = WEIGHT_OP_MEMO_GENERATION.load(Ordering::Acquire);
        if memo.generation != current_gen {
            memo.clear_all();
            memo.generation = current_gen;
        }
        f(&mut memo)
    })
}

fn with_public_weight_intersection_memo<R>(
    f: impl FnOnce(&mut PublicWeightIntersectionMemo) -> R,
) -> R {
    PUBLIC_WEIGHT_INTERSECTION_MEMO.with(|memo| {
        let mut memo = memo.borrow_mut();
        let current_gen = WEIGHT_OP_MEMO_GENERATION.load(Ordering::Acquire);
        if memo.generation != current_gen {
            memo.clear_all();
            memo.generation = current_gen;
        }
        f(&mut memo)
    })
}

/// Cached structural hashes for interned `Weight` maps.
///
/// DWA minimization and merge code hash the same interned weights many times.
/// Computing the structural hash repeatedly walks every outer range and every
/// inner token range, which is especially costly for p0/global terminal-DWA
/// workloads. The cache key is the interned `Arc` pointer, guarded by a weak
/// reference to avoid ABA reuse; the cached value is still the full structural
/// hash, so equal weights keep identical `Hash` output even if a future caller
/// constructs an equal map under a different `Arc`.
struct WeightHashMemoEntry {
    result: u64,
    weight: Weak<WeightMap>,
}

#[derive(Default)]
struct WeightHashMemo {
    results: FxHashMap<usize, WeightHashMemoEntry>,
    inserts_since_cleanup: usize,
    generation: u64,
}

impl WeightHashMemo {
    fn maybe_cleanup(&mut self) {
        if self.inserts_since_cleanup < INTERNER_CLEANUP_INTERVAL {
            return;
        }
        self.results.retain(|_, entry| entry.weight.strong_count() > 0);
        self.inserts_since_cleanup = 0;
    }

    fn clear_all(&mut self) {
        self.results.clear();
        self.inserts_since_cleanup = 0;
    }

    fn get_or_insert(&mut self, weight: &Weight) -> u64 {
        let current_gen = WEIGHT_HASH_MEMO_GENERATION.load(Ordering::Acquire);
        if self.generation != current_gen {
            self.clear_all();
            self.generation = current_gen;
        }
        self.maybe_cleanup();
        let key = weight.ptr_key();
        if let Some(entry) = self.results.get(&key) {
            if let Some(existing) = entry.weight.upgrade() {
                if Arc::ptr_eq(&existing, &weight.0) {
                    return entry.result;
                }
            }
        }

        let result = structural_weight_hash_uncached(weight);
        self.results.insert(
            key,
            WeightHashMemoEntry {
                result,
                weight: Arc::downgrade(&weight.0),
            },
        );
        self.inserts_since_cleanup += 1;
        result
    }
}

thread_local! {
    static WEIGHT_HASH_MEMO: RefCell<WeightHashMemo> = RefCell::new(WeightHashMemo::default());
}

fn structural_weight_hash_uncached(weight: &Weight) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = rustc_hash::FxHasher::default();
    let is_full = weight.is_full();
    is_full.hash(&mut hasher);
    if !is_full {
        for (range, tokens) in weight.0.range_values() {
            range.hash(&mut hasher);
            tokens.as_ref().hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn cached_structural_weight_hash(weight: &Weight) -> u64 {
    WEIGHT_HASH_MEMO.with(|memo| memo.borrow_mut().get_or_insert(weight))
}

/// Prune only dead entries from the global weight interner.
pub fn clear_stale_weights() {
    interner_clear_stale();
}

/// Clear all live entries from the global weight/token-set interners and
/// invalidate memoized weight-operation caches that may retain those weights.
///
/// This is mainly useful for benchmarks that need to prevent interner reuse
/// from contaminating repeated compile measurements.
pub fn clear_weight_interners() {
    interner_clear_all();
    clear_weight_op_caches();
}

/// Clear weight-operation and structural-hash memo caches on **all** threads.
///
/// Increments the global generation counter so that every thread's
/// thread-local memo is lazily cleared on its next access.
pub fn clear_weight_op_caches() {
    WEIGHT_OP_MEMO_GENERATION.fetch_add(1, Ordering::Release);
    WEIGHT_HASH_MEMO_GENERATION.fetch_add(1, Ordering::Release);
}

fn lookup_memoized_weight_op(kind: WeightOpKind, left: &Weight, right: &Weight) -> Option<Weight> {
    with_weight_op_memo(|memo| memo.lookup(WeightOpKey::for_weights(kind, left, right)))
}

fn store_memoized_weight_op(kind: WeightOpKind, left: &Weight, right: &Weight, result: &Weight) {
    with_weight_op_memo(|memo| {
        memo.store(WeightOpKey::for_weights(kind, left, right), left, right, result)
    });
}

fn with_memoized_weight_op(
    kind: WeightOpKind,
    left: &Weight,
    right: &Weight,
    build: impl FnOnce() -> Weight,
) -> Weight {
    if let Some(existing) = lookup_memoized_weight_op(kind, left, right) {
        return existing;
    }

    let result = build();
    store_memoized_weight_op(kind, left, right, &result);
    result
}

pub(crate) fn finalize_weight_map(map: WeightMap) -> Weight {
    if map.ranges().next().is_none() {
        EMPTY_WEIGHT.clone()
    } else {
        Weight(intern_weight_map(map))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WeightSerdeEntry {
    tsid: [u32; 2],
    tokens: Vec<[u32; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WeightSerde {
    all: bool,
    entries: Vec<WeightSerdeEntry>,
}

fn sentinel_token_set() -> RangeSetBlaze<u32> {
    std::iter::once(WEIGHT_ALL_SENTINEL..=WEIGHT_ALL_SENTINEL).collect()
}

fn is_sentinel_token_set(tokens: &RangeSetBlaze<u32>) -> bool {
    let mut ranges = tokens.ranges();
    let Some(range) = ranges.next() else {
        return false;
    };
    ranges.next().is_none()
        && *range.start() == WEIGHT_ALL_SENTINEL
        && *range.end() == WEIGHT_ALL_SENTINEL
}

pub(crate) fn shared_rangeset(tokens: RangeSetBlaze<u32>) -> SharedTokenSet {
    intern_rangeset(tokens)
}

fn rangeset_from_ranges<I>(ranges: I) -> RangeSetBlaze<u32>
where
    I: IntoIterator<Item = std::ops::RangeInclusive<u32>>,
{
    ranges.into_iter().collect()
}

fn rangeset_to_vec(set: &RangeSetBlaze<u32>) -> Vec<[u32; 2]> {
    set.ranges()
        .map(|range| [*range.start(), *range.end()])
        .collect()
}

fn rangeset_to_string(set: &RangeSetBlaze<u32>) -> String {
    let parts: Vec<String> = set
        .ranges()
        .map(|range| {
            if range.start() == range.end() {
                format!("{}", range.start())
            } else {
                format!("{}..={}", range.start(), range.end())
            }
        })
        .collect();
    format!("{{{}}}", parts.join(","))
}

fn compress_expanded(expanded: &BTreeMap<u32, RangeSetBlaze<u32>>) -> Weight {
    let mut builder = CompactRangeBuilder::new();
    let mut current_start: Option<u32> = None;
    let mut current_end = 0u32;
    let mut current_tokens = RangeSetBlaze::new();

    for (&tsid, tokens) in expanded {
        match current_start {
            Some(_)
                if current_end.checked_add(1) == Some(tsid) && *tokens == current_tokens =>
            {
                current_end = tsid;
            }
            _ => {
                if let Some(start) = current_start.take() {
                    builder.push(
                        start,
                        current_end,
                        shared_rangeset(std::mem::take(&mut current_tokens)),
                    );
                }
                current_start = Some(tsid);
                current_end = tsid;
                current_tokens = tokens.clone();
            }
        }
    }

    if let Some(start) = current_start {
        builder.push(start, current_end, shared_rangeset(current_tokens));
    }

    builder.finish()
}

#[derive(Clone)]
struct WeightRangeEntry {
    start: u32,
    end: u32,
    tokens: SharedTokenSet,
}

#[derive(Clone)]
pub(crate) struct WeightIntersectionIndex {
    source: Weight,
    entries: Vec<WeightRangeEntry>,
}

impl WeightIntersectionIndex {
    fn new(source: &Weight) -> Self {
        Self {
            source: source.clone(),
            entries: source
                .0
                .range_values()
                .map(|(range, tokens)| WeightRangeEntry {
                    start: *range.start(),
                    end: *range.end(),
                    tokens: Arc::clone(tokens),
                })
                .collect(),
        }
    }
}

fn intersect_weight_with_index(sparse: &Weight, index: &WeightIntersectionIndex) -> Weight {
    let mut builder = CompactRangeBuilder::new();
    let mut overlap_cache: SmallVec<[(
        *const RangeSetBlaze<u32>,
        *const RangeSetBlaze<u32>,
        Option<SharedTokenSet>,
    ); 8]> = SmallVec::new();

    for (sparse_range, sparse_tokens) in sparse.0.range_values() {
        let sparse_start = *sparse_range.start();
        let sparse_end = *sparse_range.end();
        let mut entry_index = index
            .entries
            .partition_point(|entry| entry.end < sparse_start);

        while let Some(entry) = index.entries.get(entry_index) {
            if entry.start > sparse_end {
                break;
            }

            let start = sparse_start.max(entry.start);
            let end = sparse_end.min(entry.end);
            let sparse_ptr = Arc::as_ptr(sparse_tokens);
            let dense_ptr = Arc::as_ptr(&entry.tokens);
            let tokens = if let Some((_, _, cached)) = overlap_cache.iter().find(
                |(cached_sparse, cached_dense, _)| {
                    *cached_sparse == sparse_ptr && *cached_dense == dense_ptr
                },
            ) {
                cached.clone()
            } else {
                let overlap = shared_token_intersection(sparse_tokens, &entry.tokens);
                overlap_cache.push((sparse_ptr, dense_ptr, overlap.clone()));
                overlap
            };

            if let Some(tokens) = tokens {
                builder.push(start, end, tokens);
            }
            entry_index += 1;
        }
    }

    builder.finish()
}

struct CompactRangeBuilder {
    map: WeightMap,
    pending_start: Option<u32>,
    pending_end: u32,
    pending_tokens: SharedTokenSet,
}

impl CompactRangeBuilder {
    fn new() -> Self {
        Self {
            map: WeightMap::new(),
            pending_start: None,
            pending_end: 0,
            pending_tokens: Arc::clone(&EMPTY_RANGESET),
        }
    }

    fn push(&mut self, start: u32, end: u32, tokens: SharedTokenSet) {
        match self.pending_start {
            Some(_)
                if self.pending_end.checked_add(1) == Some(start)
                    && same_shared_token_set(&self.pending_tokens, &tokens) =>
            {
                self.pending_end = end;
            }
            _ => {
                self.flush();
                self.pending_start = Some(start);
                self.pending_end = end;
                self.pending_tokens = tokens;
            }
        }
    }

    fn flush(&mut self) {
        if let Some(start) = self.pending_start.take() {
            let tokens = std::mem::replace(&mut self.pending_tokens, Arc::clone(&EMPTY_RANGESET));
            self.map
                .extend_simple(std::iter::once((start..=self.pending_end, tokens)));
        }
    }

    fn finish(mut self) -> Weight {
        self.flush();
        finalize_weight_map(self.map)
    }
}

fn compact_entries(weight: &Weight) -> SmallVec<[WeightRangeEntry; 16]> {
    weight
        .0
        .range_values()
        .map(|(range, tokens)| WeightRangeEntry {
            start: *range.start(),
            end: *range.end(),
            tokens: Arc::clone(tokens),
        })
        .collect()
}

fn single_compact_entry(weight: &Weight) -> Option<WeightRangeEntry> {
    let mut entries = weight.0.range_values();
    let (range, tokens) = entries.next()?;
    if entries.next().is_some() {
        return None;
    }
    Some(WeightRangeEntry {
        start: *range.start(),
        end: *range.end(),
        tokens: Arc::clone(tokens),
    })
}

fn insert_boundary(boundaries: &mut [u64; 4], len: &mut usize, value: u64) {
    let mut pos = 0usize;
    while pos < *len && boundaries[pos] < value {
        pos += 1;
    }
    if pos < *len && boundaries[pos] == value {
        return;
    }
    let mut i = *len;
    while i > pos {
        boundaries[i] = boundaries[i - 1];
        i -= 1;
    }
    boundaries[pos] = value;
    *len += 1;
}

fn combine_single_entries<F>(
    left: &WeightRangeEntry,
    right: &WeightRangeEntry,
    mut combine: F,
) -> Weight
where
    F: FnMut(
        Option<&SharedTokenSet>,
        Option<&SharedTokenSet>,
    ) -> Option<SharedTokenSet>,
{
    let mut boundaries = [0u64; 4];
    let mut len = 0usize;
    insert_boundary(&mut boundaries, &mut len, u64::from(left.start));
    insert_boundary(&mut boundaries, &mut len, u64::from(left.end) + 1);
    insert_boundary(&mut boundaries, &mut len, u64::from(right.start));
    insert_boundary(&mut boundaries, &mut len, u64::from(right.end) + 1);

    if len < 2 {
        return Weight::empty();
    }

    let mut builder = CompactRangeBuilder::new();

    for i in 0..(len - 1) {
        let start = boundaries[i] as u32;
        let end = (boundaries[i + 1] - 1) as u32;
        let left_tokens = (left.start <= start && start <= left.end).then_some(&left.tokens);
        let right_tokens = (right.start <= start && start <= right.end).then_some(&right.tokens);
        let Some(tokens) = combine(left_tokens, right_tokens) else {
            builder.flush();
            continue;
        };
        builder.push(start, end, tokens);
    }

    builder.finish()
}

fn weight_tsid_span(weight: &Weight) -> Option<(u32, u32)> {
    let mut ranges = weight.0.ranges();
    let first = ranges.next()?;
    let mut last_end = *first.end();
    for range in ranges {
        last_end = *range.end();
    }
    Some((*first.start(), last_end))
}

fn append_weight_entries(builder: &mut CompactRangeBuilder, weight: &Weight) {
    for (range, tokens) in weight.0.range_values() {
        builder.push(*range.start(), *range.end(), Arc::clone(tokens));
    }
}

fn union_disjoint_tsid_ranges(left: &Weight, right: &Weight) -> Option<Weight> {
    let (left_start, left_end) = weight_tsid_span(left)?;
    let (right_start, right_end) = weight_tsid_span(right)?;

    let mut builder = CompactRangeBuilder::new();
    if left_end < right_start {
        append_weight_entries(&mut builder, left);
        append_weight_entries(&mut builder, right);
        Some(builder.finish())
    } else if right_end < left_start {
        append_weight_entries(&mut builder, right);
        append_weight_entries(&mut builder, left);
        Some(builder.finish())
    } else {
        None
    }
}

/// Direct multi-way union that avoids creating O(N) intermediate Weight objects.
///
/// For large inputs, this uses an event sweep over start and end boundaries.
/// The previous implementation re-scanned every started entry at every
/// boundary, which is quadratic for long overlapping ranges.  The event sweep
/// maintains the active distinct token sets incrementally instead.
fn union_all_multiway(weights: &[&Weight]) -> Weight {
    let total_entry_hint: usize = weights.iter().map(|w| w.0.ranges().count()).sum();
    let mut all_entries: Vec<WeightRangeEntry> = Vec::with_capacity(total_entry_hint);
    for weight in weights {
        for (range, tokens) in weight.0.range_values() {
            all_entries.push(WeightRangeEntry {
                start: *range.start(),
                end: *range.end(),
                tokens: Arc::clone(tokens),
            });
        }
    }

    if all_entries.is_empty() {
        return Weight::empty();
    }

    // The compact rescan path avoids tree-map overhead for the small common
    // case; the event sweep avoids pathological repeated scans for larger
    // overlapping unions.
    const INCREMENTAL_SWEEP_MIN_ENTRIES: usize = 64;
    if all_entries.len() < INCREMENTAL_SWEEP_MIN_ENTRIES {
        return union_all_multiway_rescan(all_entries);
    }
    union_all_multiway_incremental(all_entries)
}

fn union_all_multiway_rescan(mut all_entries: Vec<WeightRangeEntry>) -> Weight {
    let mut boundaries = Vec::with_capacity(all_entries.len() * 2);
    for entry in &all_entries {
        boundaries.push(u64::from(entry.start));
        boundaries.push(u64::from(entry.end) + 1);
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    if boundaries.len() < 2 {
        return Weight::empty();
    }

    all_entries.sort_unstable_by_key(|entry| entry.start);

    let mut builder = CompactRangeBuilder::new();
    let mut scan_start = 0usize;
    let mut active_tokens = Vec::<SharedTokenSet>::new();
    let mut token_union_cache: FxHashMap<Vec<usize>, SharedTokenSet> = FxHashMap::default();

    for window in boundaries.windows(2) {
        let interval_start = window[0] as u32;
        let interval_end = (window[1] - 1) as u32;

        while scan_start < all_entries.len() && all_entries[scan_start].end < interval_start {
            scan_start += 1;
        }

        active_tokens.clear();
        for entry in &all_entries[scan_start..] {
            if entry.start > interval_start {
                break;
            }
            if entry.end >= interval_end {
                active_tokens.push(Arc::clone(&entry.tokens));
            }
        }

        active_tokens.sort_unstable_by_key(|tokens| Arc::as_ptr(tokens) as usize);
        active_tokens.dedup_by_key(|tokens| Arc::as_ptr(tokens) as usize);

        let tokens = union_active_token_sets(&active_tokens, &mut token_union_cache);
        if let Some(tokens) = tokens {
            builder.push(interval_start, interval_end, tokens);
        } else {
            builder.flush();
        }
    }

    builder.finish()
}

fn union_all_multiway_incremental(mut all_entries: Vec<WeightRangeEntry>) -> Weight {
    all_entries.sort_unstable_by_key(|entry| entry.start);

    // End events are exclusive so entries ending at `boundary - 1` leave the
    // active set before entries beginning at `boundary` enter it.
    let mut end_events: Vec<(u64, SharedTokenSet)> = all_entries
        .iter()
        .map(|entry| (u64::from(entry.end) + 1, Arc::clone(&entry.tokens)))
        .collect();
    end_events.sort_unstable_by_key(|(end_exclusive, _)| *end_exclusive);

    let mut builder = CompactRangeBuilder::new();
    let mut active: BTreeMap<usize, (SharedTokenSet, usize)> = BTreeMap::new();
    let mut active_tokens = Vec::<SharedTokenSet>::new();
    let mut token_union_cache: FxHashMap<Vec<usize>, SharedTokenSet> = FxHashMap::default();
    let mut start_index = 0usize;
    let mut end_index = 0usize;

    while start_index < all_entries.len() || end_index < end_events.len() {
        let next_start = all_entries
            .get(start_index)
            .map(|entry| u64::from(entry.start));
        let next_end = end_events.get(end_index).map(|(end, _)| *end);
        let boundary = match (next_start, next_end) {
            (Some(start), Some(end)) => start.min(end),
            (Some(start), None) => start,
            (None, Some(end)) => end,
            (None, None) => break,
        };

        while end_index < end_events.len() && end_events[end_index].0 == boundary {
            let tokens = &end_events[end_index].1;
            let key = Arc::as_ptr(tokens) as usize;
            let mut remove = false;
            if let Some((_, count)) = active.get_mut(&key) {
                debug_assert!(*count > 0);
                *count -= 1;
                remove = *count == 0;
            } else {
                debug_assert!(false, "every end event must have an active start event");
            }
            if remove {
                active.remove(&key);
            }
            end_index += 1;
        }

        while start_index < all_entries.len()
            && u64::from(all_entries[start_index].start) == boundary
        {
            let tokens = Arc::clone(&all_entries[start_index].tokens);
            let key = Arc::as_ptr(&tokens) as usize;
            active
                .entry(key)
                .and_modify(|(_, count)| *count += 1)
                .or_insert((tokens, 1));
            start_index += 1;
        }

        let following_start = all_entries
            .get(start_index)
            .map(|entry| u64::from(entry.start));
        let following_end = end_events.get(end_index).map(|(end, _)| *end);
        let Some(next_boundary) = (match (following_start, following_end) {
            (Some(start), Some(end)) => Some(start.min(end)),
            (Some(start), None) => Some(start),
            (None, Some(end)) => Some(end),
            (None, None) => None,
        }) else {
            break;
        };

        if active.is_empty() {
            builder.flush();
            continue;
        }

        active_tokens.clear();
        active_tokens.extend(active.values().map(|(tokens, _)| Arc::clone(tokens)));
        let tokens = union_active_token_sets(&active_tokens, &mut token_union_cache)
            .expect("non-empty active set has a token union");
        builder.push(boundary as u32, (next_boundary - 1) as u32, tokens);
    }

    builder.finish()
}

fn union_active_token_sets(
    active_tokens: &[SharedTokenSet],
    token_union_cache: &mut FxHashMap<Vec<usize>, SharedTokenSet>,
) -> Option<SharedTokenSet> {
    match active_tokens.len() {
        0 => None,
        1 => Some(Arc::clone(&active_tokens[0])),
        2 => Some(shared_token_union(&active_tokens[0], &active_tokens[1])),
        _ => {
            let key: Vec<usize> = active_tokens
                .iter()
                .map(|tokens| Arc::as_ptr(tokens) as usize)
                .collect();
            if let Some(cached) = token_union_cache.get(&key) {
                Some(Arc::clone(cached))
            } else {
                let tokens = shared_token_union_many(active_tokens);
                if let Some(tokens) = &tokens {
                    token_union_cache.insert(key, Arc::clone(tokens));
                }
                tokens
            }
        }
    }
}

fn union_all_single_tsid_entries(weights: &[&Weight]) -> Option<Weight> {
    let mut per_tsid: BTreeMap<u32, SharedTokenSet> = BTreeMap::new();

    for weight in weights {
        let entry = single_compact_entry(weight)?;
        if entry.start != entry.end || entry.start == WEIGHT_ALL_SENTINEL {
            return None;
        }

        per_tsid
            .entry(entry.start)
            .and_modify(|existing| *existing = shared_token_union(existing, &entry.tokens))
            .or_insert(entry.tokens);
    }

    let mut builder = CompactRangeBuilder::new();
    for (tsid, tokens) in per_tsid {
        builder.push(tsid, tsid, tokens);
    }
    Some(builder.finish())
}

fn union_compact_entries(left: &Weight, right: &Weight) -> Weight {
    let left_entries = compact_entries(left);
    let right_entries = compact_entries(right);

    if left_entries.is_empty() {
        return right.clone();
    }
    if right_entries.is_empty() {
        return left.clone();
    }

    let mut builder = CompactRangeBuilder::new();
    let mut left_index = 0usize;
    let mut right_index = 0usize;
    let mut left_current = Some(left_entries[left_index].clone());
    let mut right_current = Some(right_entries[right_index].clone());

    loop {
        match (&mut left_current, &mut right_current) {
            (Some(left_entry), Some(right_entry)) => {
                if left_entry.end < right_entry.start {
                    builder.push(left_entry.start, left_entry.end, Arc::clone(&left_entry.tokens));
                    left_index += 1;
                    left_current = left_entries.get(left_index).cloned();
                    continue;
                }
                if right_entry.end < left_entry.start {
                    builder.push(right_entry.start, right_entry.end, Arc::clone(&right_entry.tokens));
                    right_index += 1;
                    right_current = right_entries.get(right_index).cloned();
                    continue;
                }

                if left_entry.start < right_entry.start {
                    builder.push(
                        left_entry.start,
                        right_entry.start - 1,
                        Arc::clone(&left_entry.tokens),
                    );
                    left_entry.start = right_entry.start;
                } else if right_entry.start < left_entry.start {
                    builder.push(
                        right_entry.start,
                        left_entry.start - 1,
                        Arc::clone(&right_entry.tokens),
                    );
                    right_entry.start = left_entry.start;
                }

                let overlap_end = left_entry.end.min(right_entry.end);
                builder.push(
                    left_entry.start,
                    overlap_end,
                    shared_token_union(&left_entry.tokens, &right_entry.tokens),
                );

                match (left_entry.end == overlap_end, right_entry.end == overlap_end) {
                    (true, true) => {
                        left_index += 1;
                        right_index += 1;
                        left_current = left_entries.get(left_index).cloned();
                        right_current = right_entries.get(right_index).cloned();
                    }
                    (true, false) => {
                        let next_start = overlap_end + 1;
                        right_entry.start = next_start;
                        left_index += 1;
                        left_current = left_entries.get(left_index).cloned();
                    }
                    (false, true) => {
                        let next_start = overlap_end + 1;
                        left_entry.start = next_start;
                        right_index += 1;
                        right_current = right_entries.get(right_index).cloned();
                    }
                    (false, false) => unreachable!(),
                }
            }
            (Some(left_entry), None) => {
                builder.push(left_entry.start, left_entry.end, Arc::clone(&left_entry.tokens));
                left_index += 1;
                left_current = left_entries.get(left_index).cloned();
            }
            (None, Some(right_entry)) => {
                builder.push(right_entry.start, right_entry.end, Arc::clone(&right_entry.tokens));
                right_index += 1;
                right_current = right_entries.get(right_index).cloned();
            }
            (None, None) => break,
        }
    }

    builder.finish()
}

fn combined_boundaries(left: &[WeightRangeEntry], right: &[WeightRangeEntry]) -> SmallVec<[u64; 32]> {
    let mut boundaries = SmallVec::<[u64; 32]>::with_capacity((left.len() + right.len()) * 2);
    for entry in left.iter().chain(right.iter()) {
        boundaries.push(u64::from(entry.start));
        boundaries.push(u64::from(entry.end) + 1);
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
}

fn active_tokens<'a>(
    entries: &'a [WeightRangeEntry],
    index: &mut usize,
    start: u32,
) -> Option<&'a SharedTokenSet> {
    while *index < entries.len() && entries[*index].end < start {
        *index += 1;
    }
    entries.get(*index).and_then(|entry| {
        (entry.start <= start && start <= entry.end).then_some(&entry.tokens)
    })
}

fn combine_compact_entries<F>(left: &Weight, right: &Weight, mut combine: F) -> Weight
where
    F: FnMut(
        Option<&SharedTokenSet>,
        Option<&SharedTokenSet>,
    ) -> Option<SharedTokenSet>,
{
    let left_entries = compact_entries(left);
    let right_entries = compact_entries(right);
    let boundaries = combined_boundaries(&left_entries, &right_entries);
    if boundaries.len() < 2 {
        return Weight::empty();
    }

    let mut left_index = 0usize;
    let mut right_index = 0usize;
    let mut builder = CompactRangeBuilder::new();

    for window in boundaries.windows(2) {
        let start = window[0] as u32;
        let end = (window[1] - 1) as u32;
        let left_tokens = active_tokens(&left_entries, &mut left_index, start);
        let right_tokens = active_tokens(&right_entries, &mut right_index, start);
        let Some(tokens) = combine(left_tokens, right_tokens) else {
            builder.flush();
            continue;
        };
        builder.push(start, end, tokens);
    }

    builder.finish()
}

fn intersect_weights(left: &Weight, right: &Weight) -> Weight {
    let mut left_iter = left.0.range_values();
    let mut right_iter = right.0.range_values();
    let mut left_entry = left_iter.next();
    let mut right_entry = right_iter.next();

    let mut builder = CompactRangeBuilder::new();
    let mut same_as_left = true;
    let mut same_as_right = true;

    loop {
        let (left_range, left_tokens, right_range, right_tokens) = match (left_entry, right_entry)
        {
            (Some((left_range, left_tokens)), Some((right_range, right_tokens))) => {
                (left_range, left_tokens, right_range, right_tokens)
            }
            (Some(_), None) => {
                same_as_left = false;
                break;
            }
            (None, Some(_)) => {
                same_as_right = false;
                break;
            }
            (None, None) => break,
        };
        let start = (*left_range.start()).max(*right_range.start());
        let end = (*left_range.end()).min(*right_range.end());
        let left_start = *left_range.start();
        let left_end = *left_range.end();
        let right_start = *right_range.start();
        let right_end = *right_range.end();

        if start <= end {
            if start != left_start || end != left_end {
                same_as_left = false;
            }
            if start != right_start || end != right_end {
                same_as_right = false;
            }
            if let Some(tokens) = shared_token_intersection(left_tokens, right_tokens) {
                if !Arc::ptr_eq(&tokens, left_tokens) {
                    same_as_left = false;
                }
                if !Arc::ptr_eq(&tokens, right_tokens) {
                    same_as_right = false;
                }
                builder.push(start, end, tokens);
            } else {
                same_as_left = false;
                same_as_right = false;
            }
        } else if left_end < right_start {
            same_as_left = false;
        } else if right_end < left_start {
            same_as_right = false;
        }

        if left_end <= right_end {
            left_entry = left_iter.next();
        } else {
            left_entry = Some((left_range, left_tokens));
        }
        if right_end <= left_end {
            right_entry = right_iter.next();
        } else {
            right_entry = Some((right_range, right_tokens));
        }
    }

    if same_as_left {
        left.clone()
    } else if same_as_right {
        right.clone()
    } else {
        builder.finish()
    }
}

fn intersect_single_entry_with_weight(single: &WeightRangeEntry, other: &Weight) -> Weight {
    let mut builder = CompactRangeBuilder::new();
    let mut overlap_cache: SmallVec<[(
        *const RangeSetBlaze<u32>,
        Option<SharedTokenSet>,
    ); 8]> = SmallVec::new();

    let bounds = CheckSortedDisjoint::new([single.start..=single.end]);
    for (range, other_tokens) in other.0.range_values().map_and_set_intersection(bounds) {
        let start = *range.start();
        let end = *range.end();

        let tokens = if same_shared_token_set(&single.tokens, other_tokens) {
            Some(Arc::clone(&single.tokens))
        } else {
            let cache_key = Arc::as_ptr(other_tokens);
            if let Some((_, cached)) = overlap_cache.iter().find(|(ptr, _)| *ptr == cache_key) {
                cached.clone()
            } else {
                let overlap = shared_token_intersection(&single.tokens, other_tokens);
                overlap_cache.push((cache_key, overlap.clone()));
                overlap
            }
        };

        let Some(tokens) = tokens else {
            builder.flush();
            continue;
        };

        builder.push(start, end, tokens);
    }

    builder.finish()
}

impl Weight {
    pub(crate) fn intersection_index(&self) -> WeightIntersectionIndex {
        WeightIntersectionIndex::new(self)
    }

    pub(crate) fn intersection_with_index(&self, index: &WeightIntersectionIndex) -> Self {
        let other = &index.source;
        if self.is_empty() || other.is_empty() {
            return Self::empty();
        }
        if Arc::ptr_eq(&self.0, &other.0) {
            return self.clone();
        }
        if self.is_full() {
            return other.clone();
        }
        if other.is_full() {
            return self.clone();
        }

        if let Some(existing) = with_public_weight_intersection_memo(|memo| memo.lookup(self, other)) {
            return existing;
        }

        let result = intersect_weight_with_index(self, index);
        with_public_weight_intersection_memo(|memo| memo.store(self, other, &result));
        result
    }

    pub(crate) fn ptr_key(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    pub(crate) fn structural_hash_cached(&self) -> u64 {
        cached_structural_weight_hash(self)
    }

    pub(crate) fn compact_entries(&self) -> Option<Vec<(u32, u32, SharedTokenSet)>> {
        if self.is_full() {
            return None;
        }

        Some(
            self.0
                .range_values()
                .map(|(range, tokens)| (*range.start(), *range.end(), Arc::clone(tokens)))
                .collect(),
        )
    }

    /// Borrowed outer-map entries for consumers that do not retain token sets.
    pub(crate) fn range_entries(
        &self,
    ) -> impl Iterator<Item = (u32, u32, &SharedTokenSet)> + '_ {
        self.0
            .range_values()
            .map(|(range, tokens)| (*range.start(), *range.end(), tokens))
    }

    pub fn empty() -> Self {
        EMPTY_WEIGHT.clone()
    }

    pub fn all() -> Self {
        ALL_WEIGHT.clone()
    }

    /// Create a weight where all tsids in the range share the same token set.
    pub fn from_uniform(tsid_range: std::ops::RangeInclusive<u32>, tokens: RangeSetBlaze<u32>) -> Self {
        if tokens.is_empty() {
            return Self::empty();
        }
        let mut map = WeightMap::new();
        map.extend_simple(std::iter::once((tsid_range, shared_rangeset(tokens))));
        finalize_weight_map(map)
    }

    /// Build a weight from per-TSID token sets without creating intermediate Weight objects.
    /// Each entry is (tsid, token_set). Entries MUST be sorted by tsid (ascending).
    /// Adjacent TSIDs with identical (Arc-equal) token sets are merged into ranges.
    pub fn from_per_tsid_token_sets(entries: impl IntoIterator<Item = (u32, RangeSetBlaze<u32>)>) -> Self {
        let mut builder = CompactRangeBuilder::new();
        for (tsid, tokens) in entries {
            if tokens.is_empty() {
                continue;
            }
            builder.push(tsid, tsid, shared_rangeset(tokens));
        }
        builder.finish()
    }

    /// Like `from_per_tsid_token_sets` but accepts pre-shared (Arc) token sets.
    /// This allows TSIDs sharing the same representative state to reuse the same
    /// Arc, enabling CompactRangeBuilder to merge them into contiguous ranges.
    pub fn from_per_tsid_shared(entries: impl IntoIterator<Item = (u32, SharedTokenSet)>) -> Self {
        let mut builder = CompactRangeBuilder::new();
        for (tsid, tokens) in entries {
            if tokens.is_empty() {
                continue;
            }
            builder.push(tsid, tsid, tokens);
        }
        builder.finish()
    }

    /// Like `from_per_tsid_shared` but accepts pre-grouped, sorted inclusive
    /// `(start_tsid, end_tsid)` runs that each share one Arc token set. This is
    /// byte-identical to pushing every TSID in `start..=end` individually (the
    /// builder merges same-Arc contiguous TSIDs anyway) but skips the per-TSID
    /// iteration when the caller already knows the contiguous coordinate runs.
    pub(crate) fn from_tsid_runs_shared(
        runs: impl IntoIterator<Item = (u32, u32, SharedTokenSet)>,
    ) -> Self {
        let mut builder = CompactRangeBuilder::new();
        for (start, end, tokens) in runs {
            if tokens.is_empty() {
                continue;
            }
            builder.push(start, end, tokens);
        }
        builder.finish()
    }

    /// Return the interned token set at one TSID without cloning its range
    /// contents. This is intentionally crate-visible for sparse coordinate
    /// remaps in the terminal-DWA compiler.
    pub(crate) fn shared_tokens_for_tsid(&self, tsid: u32) -> SharedTokenSet {
        if self.is_full() {
            return self
                .0
                .range_values()
                .next()
                .expect("full weight must retain its sentinel token set")
                .1
                .clone();
        }
        self.0
            .get(tsid)
            .cloned()
            .unwrap_or_else(|| Arc::clone(&EMPTY_RANGESET))
    }

    /// Apply a sorted, unique set of point-TSID token-set overrides without
    /// expanding the unchanged ranges of `self`. Empty override token sets
    /// remove that point from the result.
    pub(crate) fn with_sparse_tsid_overrides(
        &self,
        overrides: &[(u32, SharedTokenSet)],
    ) -> Self {
        if overrides.is_empty() || self.is_full() {
            return self.clone();
        }
        debug_assert!(overrides.windows(2).all(|pair| pair[0].0 < pair[1].0));

        let mut builder = CompactRangeBuilder::new();
        let mut override_index = 0usize;

        let push_override = |builder: &mut CompactRangeBuilder,
                             tsid: u32,
                             tokens: &SharedTokenSet| {
            if !tokens.is_empty() {
                builder.push(tsid, tsid, Arc::clone(tokens));
            }
        };

        for (start, end, tokens) in self.range_entries() {
            while override_index < overrides.len() && overrides[override_index].0 < start {
                let (tsid, override_tokens) = &overrides[override_index];
                push_override(&mut builder, *tsid, override_tokens);
                override_index += 1;
            }

            let mut cursor = start;
            while override_index < overrides.len() && overrides[override_index].0 <= end {
                let (tsid, override_tokens) = &overrides[override_index];
                if cursor < *tsid {
                    builder.push(cursor, *tsid - 1, Arc::clone(tokens));
                }
                push_override(&mut builder, *tsid, override_tokens);
                cursor = tsid.saturating_add(1);
                override_index += 1;
            }
            if cursor <= end {
                builder.push(cursor, end, Arc::clone(tokens));
            }
        }

        while override_index < overrides.len() {
            let (tsid, tokens) = &overrides[override_index];
            push_override(&mut builder, *tsid, tokens);
            override_index += 1;
        }

        builder.finish()
    }

    /// Apply sparse TSID overrides and restrict the result to `domain` in a
    /// single range walk. This is exactly equivalent to
    /// `self.with_sparse_tsid_overrides(overrides).intersection(domain)`, but
    /// avoids interning and then immediately re-reading the intermediate
    /// overridden weight.
    pub(crate) fn with_sparse_tsid_overrides_intersection(
        &self,
        overrides: &[(u32, SharedTokenSet)],
        domain: &Weight,
    ) -> Self {
        if self.is_empty() || domain.is_empty() {
            return Self::empty();
        }
        if self.is_full() {
            return self.intersection(domain);
        }
        if overrides.is_empty() {
            return self.intersection(domain);
        }
        if domain.is_full() {
            return self.with_sparse_tsid_overrides(overrides);
        }
        debug_assert!(overrides.windows(2).all(|pair| pair[0].0 < pair[1].0));

        let mut overlay = SmallVec::<[WeightRangeEntry; 16]>::new();
        let push_overlay = |entries: &mut SmallVec<[WeightRangeEntry; 16]>,
                            start: u32,
                            end: u32,
                            tokens: &SharedTokenSet| {
            if tokens.is_empty() || start > end {
                return;
            }
            if let Some(previous) = entries.last_mut()
                && previous.end.checked_add(1) == Some(start)
                && same_shared_token_set(&previous.tokens, tokens)
            {
                previous.end = end;
            } else {
                entries.push(WeightRangeEntry {
                    start,
                    end,
                    tokens: Arc::clone(tokens),
                });
            }
        };

        let mut override_index = 0usize;
        for (start, end, tokens) in self.range_entries() {
            while override_index < overrides.len() && overrides[override_index].0 < start {
                let (tsid, override_tokens) = &overrides[override_index];
                push_overlay(&mut overlay, *tsid, *tsid, override_tokens);
                override_index += 1;
            }

            let mut cursor = start;
            while override_index < overrides.len() && overrides[override_index].0 <= end {
                let (tsid, override_tokens) = &overrides[override_index];
                if cursor < *tsid {
                    push_overlay(&mut overlay, cursor, *tsid - 1, &tokens);
                }
                push_overlay(&mut overlay, *tsid, *tsid, override_tokens);
                cursor = tsid.saturating_add(1);
                override_index += 1;
            }
            if cursor <= end {
                push_overlay(&mut overlay, cursor, end, &tokens);
            }
        }
        while override_index < overrides.len() {
            let (tsid, tokens) = &overrides[override_index];
            push_overlay(&mut overlay, *tsid, *tsid, tokens);
            override_index += 1;
        }

        let mut builder = CompactRangeBuilder::new();
        let mut overlap_cache: SmallVec<[(
            *const RangeSetBlaze<u32>,
            *const RangeSetBlaze<u32>,
            Option<SharedTokenSet>,
        ); 8]> = SmallVec::new();
        let mut domain_iter = domain.0.range_values();
        let mut current_domain = domain_iter.next();
        for entry in overlay {
            while current_domain
                .as_ref()
                .is_some_and(|(range, _)| *range.end() < entry.start)
            {
                current_domain = domain_iter.next();
            }
            while let Some((range, domain_tokens)) = current_domain.as_ref() {
                if *range.start() > entry.end {
                    break;
                }
                let start = entry.start.max(*range.start());
                let end = entry.end.min(*range.end());
                let source_ptr = Arc::as_ptr(&entry.tokens);
                let domain_ptr = Arc::as_ptr(domain_tokens);
                let tokens = if let Some((_, _, cached)) = overlap_cache.iter().find(
                    |(cached_source, cached_domain, _)| {
                        *cached_source == source_ptr && *cached_domain == domain_ptr
                    },
                ) {
                    cached.clone()
                } else {
                    let overlap = shared_token_intersection(&entry.tokens, domain_tokens);
                    overlap_cache.push((source_ptr, domain_ptr, overlap.clone()));
                    overlap
                };
                if let Some(tokens) = tokens {
                    builder.push(start, end, tokens);
                }
                if *range.end() <= entry.end {
                    current_domain = domain_iter.next();
                } else {
                    break;
                }
            }
        }
        builder.finish()
    }

    /// Build the exact union of sorted point-TSID entries.
    ///
    /// The entries must be nondecreasing by TSID. Equal TSIDs are reduced with
    /// the canonical token-set union before the compact map is built, avoiding
    /// a sequence of whole-weight unions for point-entry workloads.
    pub(crate) fn union_sorted_point_entries(
        entries: impl IntoIterator<Item = (u32, SharedTokenSet)>,
    ) -> Self {
        let mut builder = CompactRangeBuilder::new();
        let mut current: Option<(u32, SharedTokenSet)> = None;

        for (tsid, tokens) in entries {
            if tokens.is_empty() {
                continue;
            }
            match current.as_mut() {
                Some((current_tsid, current_tokens)) if *current_tsid == tsid => {
                    *current_tokens = shared_token_union(current_tokens, &tokens);
                }
                Some((current_tsid, current_tokens)) => {
                    builder.push(*current_tsid, *current_tsid, Arc::clone(current_tokens));
                    current = Some((tsid, tokens));
                }
                None => current = Some((tsid, tokens)),
            }
        }

        if let Some((tsid, tokens)) = current {
            builder.push(tsid, tsid, tokens);
        }
        builder.finish()
    }

    pub fn insert(
        &mut self,
        tsid_range: std::ops::RangeInclusive<u32>,
        token_ranges: &[std::ops::RangeInclusive<u32>],
    ) {
        if self.is_full() {
            return;
        }
        let tokens = rangeset_from_ranges(token_ranges.iter().cloned());
        if tokens.is_empty() {
            return;
        }
        let mut expanded = self.expanded_entries();
        for tsid in tsid_range {
            expanded
                .entry(tsid)
                .and_modify(|existing| *existing = existing.clone() | tokens.clone())
                .or_insert_with(|| tokens.clone());
        }
        *self = compress_expanded(&expanded);
    }

    pub fn clear(&mut self) {
        *self = Self::empty();
    }

    pub fn is_full(&self) -> bool {
        let mut entries = self.0.range_values();
        let Some((range, tokens)) = entries.next() else {
            return false;
        };
        entries.next().is_none()
            && *range.start() == WEIGHT_ALL_SENTINEL
            && *range.end() == WEIGHT_ALL_SENTINEL
            && is_sentinel_token_set(tokens.as_ref())
    }

    pub fn is_empty(&self) -> bool {
        self.0.ranges().next().is_none()
    }

    pub fn num_ranges(&self) -> usize {
        self.0.ranges().count()
    }

    /// Return the set of TSID ranges covered by this weight, ignoring token
    /// subranges. `None` denotes the mathematical full weight, whose TSID
    /// coverage may overlap any finite TSID set.
    pub(crate) fn tsid_coverage(&self) -> Option<RangeSetBlaze<u32>> {
        if self.is_full() {
            return None;
        }
        Some(self.0.ranges().collect())
    }

    pub fn union(&self, other: &Self) -> Self {
        if self.is_full() || other.is_full() {
            return Self::all();
        }
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        if Arc::ptr_eq(&self.0, &other.0) {
            return self.clone();
        }
        if let Some(existing) = lookup_memoized_weight_op(WeightOpKind::Union, self, other) {
            return existing;
        }
        let result = self.union_uncached(other);
        store_memoized_weight_op(WeightOpKind::Union, self, other, &result);
        result
    }

    fn union_uncached(&self, other: &Self) -> Self {
        if let Some(result) = union_disjoint_tsid_ranges(self, other) {
            return result;
        }

        let left_single = single_compact_entry(self);
        let right_single = single_compact_entry(other);

        if let (Some(left), Some(right)) = (&left_single, &right_single) {
            combine_single_entries(&left, &right, union_token_sets)
        } else {
            union_compact_entries(self, other)
        }
    }

    pub fn union_all<'a>(weights: impl IntoIterator<Item = &'a Self>) -> Self {
        let mut meaningful = SmallVec::<[&Weight; 8]>::new();
        for weight in weights {
            if weight.is_full() {
                return Self::all();
            }
            if weight.is_empty() {
                continue;
            }
            meaningful.push(weight);
        }

        let result = match meaningful.len() {
            0 => Self::empty(),
            1 => meaningful[0].clone(),
            _ if meaningful.len() > 4 => {
            meaningful.sort_unstable_by_key(|w| w.ptr_key());
            meaningful.dedup_by_key(|w| w.ptr_key());
            if meaningful.len() == 1 {
                meaningful[0].clone()
            } else if let Some(result) = union_all_single_tsid_entries(&meaningful) {
                result
            } else if meaningful.len() > 4 {
                union_all_multiway(&meaningful)
            } else {
                let mut iter = meaningful.into_iter();
                let mut acc = iter.next().unwrap().clone();
                for weight in iter {
                    acc = acc.union(weight);
                }
                acc
            }
            }
            _ => {
                if let Some(result) = union_all_single_tsid_entries(&meaningful) {
                    result
                } else {
                    let mut iter = meaningful.into_iter();
                    let mut acc = iter.next().unwrap().clone();
                    for weight in iter {
                        acc = acc.union(weight);
                    }
                    acc
                }
            }
        };
        result
    }

    /// Exact direct multi-way union. This avoids allocating pairwise
    /// intermediate weights when several complex operands are merged once.
    pub(crate) fn union_all_direct<'a>(weights: impl IntoIterator<Item = &'a Self>) -> Self {
        let mut meaningful = SmallVec::<[&Weight; 8]>::new();
        for weight in weights {
            if weight.is_full() {
                return Self::all();
            }
            if !weight.is_empty() {
                meaningful.push(weight);
            }
        }
        meaningful.sort_unstable_by_key(|weight| weight.ptr_key());
        meaningful.dedup_by_key(|weight| weight.ptr_key());
        match meaningful.len() {
            0 => Self::empty(),
            1 => meaningful[0].clone(),
            _ => union_all_single_tsid_entries(&meaningful)
                .unwrap_or_else(|| union_all_multiway(&meaningful)),
        }
    }

    pub fn intersection(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::empty();
        }
        if Arc::ptr_eq(&self.0, &other.0) {
            return self.clone(); // Same weight → intersection is itself
        }
        if self.is_full() {
            return other.clone();
        }
        if other.is_full() {
            return self.clone();
        }

        let existing = with_public_weight_intersection_memo(|memo| memo.lookup(self, other));
        if let Some(existing) = existing {
            return existing;
        }

        let result = self.intersection_uncached_impl(other);
        with_public_weight_intersection_memo(|memo| memo.store(self, other, &result));
        result
    }

    pub(crate) fn intersection_uncached(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::empty();
        }
        if Arc::ptr_eq(&self.0, &other.0) {
            return self.clone(); // Same weight → intersection is itself
        }
        if self.is_full() {
            return other.clone();
        }
        if other.is_full() {
            return self.clone();
        }
        self.intersection_uncached_impl(other)
    }

    fn intersection_uncached_impl(&self, other: &Self) -> Self {
        if let (Some(left), Some(right)) = (single_compact_entry(self), single_compact_entry(other))
        {
            combine_single_entries(&left, &right, intersect_token_sets)
        } else if let Some(single) = single_compact_entry(self) {
            intersect_single_entry_with_weight(&single, other)
        } else if let Some(single) = single_compact_entry(other) {
            intersect_single_entry_with_weight(&single, self)
        } else {
            intersect_weights(self, other)
        }
    }

    pub fn difference(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_full() {
            return Self::empty();
        }
        if other.is_empty() {
            return self.clone();
        }
        if Arc::ptr_eq(&self.0, &other.0) {
            return Self::empty();
        }
        if self.is_full() {
            // Cannot compute all \ other without an explicit universe.
            // Return all() as a safe over-approximation.  Callers that need
            // exact complements should use the dedicated complement() method
            // which returns empty() as a no-op sentinel instead.
            return Self::all();
        }
        with_memoized_weight_op(WeightOpKind::Difference, self, other, || self.difference_uncached(other))
    }

    fn difference_uncached(&self, other: &Self) -> Self {
        combine_compact_entries(self, other, difference_token_sets)
    }

    pub fn complement(&self) -> Self {
        if self.is_full() {
            Self::empty()
        } else if self.is_empty() {
            Self::all()
        } else {
            // Cannot compute a proper per-TSID complement without an explicit
            // token/TSID universe.  Returning empty() makes the determinization
            // normalization step a no-op (target ∪ empty = target), which
            // preserves correctness at the cost of potentially more DWA states
            // (no subset collapsing via normalization).  The previous approach
            // was `all().difference(self)` which always returned `all()` due to
            // the sentinel representation, causing target subsets to collapse
            // into `Weight::all()` and producing false positives.
            Self::empty()
        }
    }

    pub fn from_token_set_for_tsid(tsid: u32, tokens: RangeSetBlaze<u32>) -> Self {
        if tokens.is_empty() {
            return Self::empty();
        }
        Self::from_uniform(tsid..=tsid, tokens)
    }

    pub fn tokens_for_tsid(&self, tsid: u32) -> RangeSetBlaze<u32> {
        if self.is_full() {
            return sentinel_token_set();
        }
        self.0
            .get(tsid)
            .map(|tokens| tokens.as_ref().clone())
            .unwrap_or_else(RangeSetBlaze::new)
    }

    pub(crate) fn single_compact_entry_parts(
        &self,
    ) -> Option<(u32, u32, SharedTokenSet)> {
        let entry = single_compact_entry(self)?;
        Some((entry.start, entry.end, entry.tokens))
    }

    pub(crate) fn outer_range_count(&self) -> usize {
        self.0.ranges().count()
    }

    pub(crate) fn single_tsid_shared_entry(&self) -> Option<(u32, SharedTokenSet)> {
        let (start, end, tokens) = self.single_compact_entry_parts()?;
        if start == end && start != WEIGHT_ALL_SENTINEL {
            Some((start, tokens))
        } else {
            None
        }
    }

    pub(crate) fn union_single_tsid_shared_entries(
        entries: impl IntoIterator<Item = (u32, SharedTokenSet)>,
    ) -> Self {
        let mut per_tsid: BTreeMap<u32, SharedTokenSet> = BTreeMap::new();

        for (tsid, tokens) in entries {
            per_tsid
                .entry(tsid)
                .and_modify(|existing| *existing = shared_token_union(existing, &tokens))
                .or_insert(tokens);
        }

        let mut builder = CompactRangeBuilder::new();
        for (tsid, tokens) in per_tsid {
            builder.push(tsid, tsid, tokens);
        }
        builder.finish()
    }

    pub(crate) fn intersect_single_parts(
        &self,
        start: u32,
        end: u32,
        tokens: &SharedTokenSet,
    ) -> Self {
        let single = WeightRangeEntry {
            start,
            end,
            tokens: Arc::clone(tokens),
        };
        intersect_single_entry_with_weight(&single, self)
    }

    pub(crate) fn for_each_intersection_tokens_with_single<F>(
        &self,
        start: u32,
        end: u32,
        single_tokens: &RangeSetBlaze<u32>,
        mut f: F,
    ) where
        F: FnMut(&RangeSetBlaze<u32>),
    {
        let mut overlap_cache: SmallVec<[(
            *const RangeSetBlaze<u32>,
            Option<SharedTokenSet>,
        ); 8]> = SmallVec::new();

        for (range, other_tokens) in self.0.range_values() {
            if end < *range.start() || *range.end() < start {
                continue;
            }

            if single_tokens == other_tokens.as_ref() {
                f(single_tokens);
                continue;
            }

            let cache_key = Arc::as_ptr(other_tokens);
            if let Some((_, cached)) = overlap_cache.iter().find(|(ptr, _)| *ptr == cache_key) {
                if let Some(cached_tokens) = cached {
                    f(cached_tokens.as_ref());
                }
                continue;
            }

            let overlap = single_tokens & other_tokens.as_ref();
            if overlap.is_empty() {
                overlap_cache.push((cache_key, None));
                continue;
            }

            let overlap_tokens = shared_rangeset(overlap);
            f(overlap_tokens.as_ref());
            overlap_cache.push((cache_key, Some(overlap_tokens)));
        }
    }

    /// Iterate over the unique (Arc-deduplicated) token sets in this weight.
    /// Each token set may cover one or more TSID ranges.
    pub fn unique_token_sets(&self) -> Vec<&RangeSetBlaze<u32>> {
        if self.is_full() || self.is_empty() {
            return Vec::new();
        }
        let mut seen: Vec<*const RangeSetBlaze<u32>> = Vec::new();
        let mut result = Vec::new();
        for (_range, tokens) in self.0.range_values() {
            let ptr = Arc::as_ptr(tokens);
            if !seen.contains(&ptr) {
                seen.push(ptr);
                result.push(tokens.as_ref());
            }
        }
        result
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        if self.is_empty() || other.is_empty() {
            return true;
        }
        if Arc::ptr_eq(&self.0, &other.0) {
            return false; // Same non-empty weight → not disjoint
        }
        if self.is_full() || other.is_full() {
            return false;
        }
        let mut left_iter = self.0.range_values();
        let mut right_iter = other.0.range_values();
        let mut left_entry = left_iter.next();
        let mut right_entry = right_iter.next();

        while let (Some((lr, lt)), Some((rr, rt))) = (&left_entry, &right_entry) {
            let start = (*lr.start()).max(*rr.start());
            let end = (*lr.end()).min(*rr.end());
            if start <= end && !lt.as_ref().is_disjoint(rt.as_ref()) {
                return false;
            }
            if lr.end() <= rr.end() {
                left_entry = left_iter.next();
            } else {
                right_entry = right_iter.next();
            }
        }
        true
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        if self.is_empty() || other.is_full() {
            return true;
        }
        if other.is_empty() || self.is_full() {
            return false;
        }
        let mut self_iter = self.0.range_values();
        let mut other_iter = other.0.range_values();
        let mut self_current = self_iter.next();
        let mut other_current = other_iter.next();
        // Track how far we've verified coverage of the current self entry
        let mut self_verified_up_to: Option<u32> = None;

        while let Some((self_range, self_tokens)) = &self_current {
            let self_start = self_verified_up_to
                .map(|v| v + 1)
                .unwrap_or(*self_range.start());

            if self_start > *self_range.end() {
                self_current = self_iter.next();
                self_verified_up_to = None;
                continue;
            }

            let Some((other_range, other_tokens)) = &other_current else {
                return false;
            };

            if *other_range.end() < self_start {
                other_current = other_iter.next();
                continue;
            }

            if *other_range.start() > self_start {
                return false;
            }

            if !self_tokens.as_ref().is_subset(other_tokens.as_ref()) {
                return false;
            }

            let covered_up_to = (*self_range.end()).min(*other_range.end());
            self_verified_up_to = Some(covered_up_to);

            if covered_up_to >= *self_range.end() {
                self_current = self_iter.next();
                self_verified_up_to = None;
            }
            if covered_up_to >= *other_range.end() {
                other_current = other_iter.next();
            }
        }
        true
    }

    /// Clip all token sets to `0..=max_token`, removing any entries that become empty.
    /// Does nothing to the ALL sentinel.
    pub fn clip_tokens(&mut self, max_token: u32) {
        if self.is_full() || self.is_empty() {
            return;
        }
        let clip: RangeSetBlaze<u32> = std::iter::once(0..=max_token).collect();
        let mut new_map = WeightMap::new();
        for (tsid_range, tokens) in self.0.range_values() {
            let clipped = tokens.as_ref() & &clip;
            if !clipped.is_empty() {
                new_map.extend_simple(std::iter::once((tsid_range, shared_rangeset(clipped))));
            }
        }
        *self = finalize_weight_map(new_map);
    }

    fn expanded_entries(&self) -> BTreeMap<u32, RangeSetBlaze<u32>> {
        if self.is_full() {
            return BTreeMap::new();
        }
        let mut out = BTreeMap::new();
        for (range, tokens) in self.0.range_values() {
            for tsid in range {
                out.insert(tsid, tokens.as_ref().clone());
            }
        }
        out
    }

    fn to_serde(&self) -> WeightSerde {
        if self.is_full() {
            return WeightSerde {
                all: true,
                entries: Vec::new(),
            };
        }
        WeightSerde {
            all: false,
            entries: self
                .0
                .range_values()
                .map(|(range, tokens)| WeightSerdeEntry {
                    tsid: [*range.start(), *range.end()],
                    tokens: rangeset_to_vec(tokens.as_ref()),
                })
                .collect(),
        }
    }
}

impl PartialEq for Weight {
    fn eq(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.0, &other.0) {
            return true;
        }
        if self.is_full() || other.is_full() {
            return self.is_full() == other.is_full();
        }
        let mut a = self.0.range_values();
        let mut b = other.0.range_values();
        loop {
            match (a.next(), b.next()) {
                (None, None) => return true,
                (Some((ra, ta)), Some((rb, tb))) => {
                    if ra != rb || ta.as_ref() != tb.as_ref() {
                        return false;
                    }
                }
                _ => return false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weight_for_tsid(tsid: u32, ranges: &[(u32, u32)]) -> Weight {
        let token_set = rangeset_from_ranges(ranges.iter().map(|(start, end)| *start..=*end));
        Weight::from_token_set_for_tsid(tsid, token_set)
    }

    #[test]
    fn interner_cleanup_deferral_releases_once() {
        let initial = INTERNER_CLEANUP_DEFERRAL_DEPTH.load(Ordering::Acquire);
        let first = defer_weight_interner_cleanup();
        let second = defer_weight_interner_cleanup();
        assert_eq!(
            INTERNER_CLEANUP_DEFERRAL_DEPTH.load(Ordering::Acquire),
            initial + 2,
        );
        second.finish();
        assert_eq!(
            INTERNER_CLEANUP_DEFERRAL_DEPTH.load(Ordering::Acquire),
            initial + 1,
        );
        first.finish();
        assert_eq!(INTERNER_CLEANUP_DEFERRAL_DEPTH.load(Ordering::Acquire), initial);
    }

    #[test]
    fn interner_cleanup_threshold_has_one_concurrent_claimant() {
        use std::sync::{Arc, Barrier};

        let counter = Arc::new(AtomicUsize::new(INTERNER_CLEANUP_INTERVAL - 1));
        let in_progress = Arc::new(AtomicBool::new(false));
        let barrier = Arc::new(Barrier::new(16));
        let workers: Vec<_> = (0..16)
            .map(|_| {
                let counter = Arc::clone(&counter);
                let in_progress = Arc::clone(&in_progress);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    usize::from(claim_interner_cleanup(&counter, &in_progress))
                })
            })
            .collect();
        let claims: usize = workers.into_iter().map(|worker| worker.join().unwrap()).sum();

        assert_eq!(claims, 1);
        assert!(in_progress.load(Ordering::Acquire));

        // Crossing additional intervals while the first caller is still
        // sweeping cannot claim another concurrent retain.
        for _ in 0..(INTERNER_CLEANUP_INTERVAL * 2) {
            assert!(!claim_interner_cleanup(&counter, &in_progress));
        }
        in_progress.store(false, Ordering::Release);
        assert!(claim_interner_cleanup(&counter, &in_progress));
    }

    #[test]
    fn scoped_weight_bulk_ops_union_all_identities() {
        let mut cache = ScopedWeightOpCache::default();

        assert_eq!(cache.union_all(std::iter::empty::<&Weight>()), Weight::empty());
        assert_eq!(cache.union_all([&Weight::empty()]), Weight::empty());
        assert_eq!(cache.union_all([&Weight::all()]), Weight::all());
    }

    #[test]
    fn scoped_weight_bulk_ops_union_all_matches_sequential_union() {
        let left = weight_for_tsid(1, &[(1, 3), (7, 8)]);
        let middle = weight_for_tsid(1, &[(2, 6)]);
        let right = weight_for_tsid(2, &[(10, 12)]);
        let weights = [&left, &middle, &right, &Weight::empty()];

        let mut cache = ScopedWeightOpCache::default();
        let bulk = cache.union_all(weights);

        let sequential = left.union(&middle).union(&right).union(&Weight::empty());
        assert_eq!(bulk, sequential);
    }

    #[test]
    fn sorted_point_entry_union_matches_sequential_weight_union() {
        let first = shared_rangeset(rangeset_from_ranges([1..=3]));
        let second = shared_rangeset(rangeset_from_ranges([3..=5]));
        let third = shared_rangeset(rangeset_from_ranges([7..=9]));
        let entries = vec![
            (2, Arc::clone(&first)),
            (2, Arc::clone(&second)),
            (4, Arc::clone(&third)),
        ];
        let direct = Weight::union_sorted_point_entries(entries.clone());
        let sequential = entries.into_iter().fold(Weight::empty(), |acc, (tsid, tokens)| {
            acc.union(&Weight::from_per_tsid_shared(std::iter::once((tsid, tokens))))
        });

        assert_eq!(direct, sequential);
        assert_eq!(direct.tokens_for_tsid(2), rangeset_from_ranges([1..=5]));
        assert_eq!(direct.tokens_for_tsid(4), rangeset_from_ranges([7..=9]));
    }

    #[test]
    fn multiway_union_matches_sequential_union_for_overlapping_ranges() {
        let mut weights = Vec::new();
        for index in 0..80u32 {
            weights.push(Weight::from_uniform(
                index..=index + 20,
                RangeSetBlaze::from_iter([index % 11..=(index % 11) + 3]),
            ));
        }
        weights.push(Weight::from_uniform(
            (u32::MAX - 2)..=u32::MAX,
            RangeSetBlaze::from_iter([99..=101]),
        ));

        let bulk = Weight::union_all(weights.iter());
        let direct = Weight::union_all_direct(weights.iter());
        let sequential = weights
            .iter()
            .fold(Weight::empty(), |acc, weight| acc.union(weight));

        assert_eq!(bulk, sequential);
        assert_eq!(direct, sequential);
    }

    #[test]
    fn indexed_intersection_matches_generic_intersection() {
        fn assert_matches(sparse: Weight, dense: Weight) {
            let index = dense.intersection_index();
            clear_weight_op_caches();
            let indexed = sparse.intersection_with_index(&index);
            clear_weight_op_caches();
            let generic = sparse.intersection_uncached(&dense);
            assert_eq!(indexed, generic);
        }

        let dense = Weight::from_per_tsid_token_sets((0..160u32).map(|tsid| {
            let tokens = match tsid % 5 {
                0 => RangeSetBlaze::from_iter([0..=7, 40..=47]),
                1 => RangeSetBlaze::from_iter([4..=13]),
                2 => RangeSetBlaze::from_iter([20..=29]),
                3 => RangeSetBlaze::from_iter([8..=11, 30..=36]),
                _ => RangeSetBlaze::from_iter([50..=65]),
            };
            (tsid * 3, tokens)
        }));
        let sparse = Weight::from_per_tsid_token_sets([
            (2, RangeSetBlaze::from_iter([3..=10])),
            (93, RangeSetBlaze::from_iter([0..=5, 44..=52])),
            (231, RangeSetBlaze::from_iter([7..=35])),
            (351, RangeSetBlaze::from_iter([30..=70])),
        ]);
        assert_matches(sparse, dense.clone());

        let index = dense.intersection_index();
        assert_eq!(Weight::all().intersection_with_index(&index), dense);
        assert_eq!(Weight::empty().intersection_with_index(&index), Weight::empty());

        for case in 0..64u32 {
            let dense = Weight::from_per_tsid_token_sets((0..192u32).map(|tsid| {
                let tokens = match (tsid / 3 + case) % 7 {
                    0 => RangeSetBlaze::from_iter([0..=9, 42..=57]),
                    1 => RangeSetBlaze::from_iter([5..=18]),
                    2 => RangeSetBlaze::from_iter([20..=37]),
                    3 => RangeSetBlaze::from_iter([11..=14, 30..=45]),
                    4 => RangeSetBlaze::from_iter([48..=66]),
                    5 => RangeSetBlaze::from_iter([3..=7, 70..=79]),
                    _ => RangeSetBlaze::from_iter([25..=31, 60..=73]),
                };
                (tsid, tokens)
            }));
            let sparse = Weight::from_per_tsid_token_sets((0..192u32).filter_map(|tsid| {
                if (tsid * 17 + case * 11) % 5 == 0 {
                    return None;
                }
                let tokens = match (tsid / 2 + case * 3) % 9 {
                    0 => RangeSetBlaze::from_iter([0..=6, 40..=53]),
                    1 => RangeSetBlaze::from_iter([4..=15]),
                    2 => RangeSetBlaze::from_iter([16..=29]),
                    3 => RangeSetBlaze::from_iter([8..=12, 28..=38]),
                    4 => RangeSetBlaze::from_iter([32..=51]),
                    5 => RangeSetBlaze::from_iter([50..=69]),
                    6 => RangeSetBlaze::from_iter([65..=82]),
                    7 => RangeSetBlaze::from_iter([2..=4, 75..=91]),
                    _ => RangeSetBlaze::from_iter([24..=27, 56..=64]),
                };
                Some((tsid, tokens))
            }));
            assert_matches(sparse, dense);
        }
    }

    #[test]
    fn sparse_tsid_overrides_intersection_matches_two_step_form() {
        let alpha = shared_rangeset(RangeSetBlaze::from_iter([1..=3]));
        let beta = shared_rangeset(RangeSetBlaze::from_iter([4..=7]));
        let gamma = shared_rangeset(RangeSetBlaze::from_iter([2..=6]));
        let delta = shared_rangeset(RangeSetBlaze::from_iter([8..=10]));
        let base = Weight::from_per_tsid_shared([
            (0, Arc::clone(&alpha)),
            (1, Arc::clone(&alpha)),
            (2, Arc::clone(&alpha)),
            (3, Arc::clone(&alpha)),
            (5, Arc::clone(&beta)),
            (6, Arc::clone(&beta)),
            (7, Arc::clone(&beta)),
        ]);
        let domain = Weight::from_per_tsid_shared([
            (1, Arc::clone(&gamma)),
            (2, Arc::clone(&gamma)),
            (4, Arc::clone(&delta)),
            (5, Arc::clone(&beta)),
            (6, Arc::clone(&gamma)),
            (8, Arc::clone(&delta)),
            (10, Arc::clone(&alpha)),
        ]);
        let overrides = [
            (1, Arc::clone(&delta)),
            (4, Arc::clone(&alpha)),
            (6, Arc::clone(&EMPTY_RANGESET)),
            (8, Arc::clone(&gamma)),
            (10, Arc::clone(&beta)),
        ];

        clear_weight_op_caches();
        let expected = base.with_sparse_tsid_overrides(&overrides).intersection(&domain);
        clear_weight_op_caches();
        let actual = base.with_sparse_tsid_overrides_intersection(&overrides, &domain);
        assert_eq!(actual, expected);
    }

    #[test]
    fn scoped_weight_bulk_ops_difference_many_identities() {
        let base = weight_for_tsid(3, &[(4, 9)]);
        let mut cache = ScopedWeightOpCache::default();

        assert_eq!(cache.difference_many(&base, std::iter::empty::<&Weight>()), base);
        assert_eq!(cache.difference_many(&base, [&base]), Weight::empty());
    }

    #[test]
    fn scoped_weight_bulk_ops_difference_many_matches_sequential_difference() {
        let base = Weight::union_all([
            &weight_for_tsid(1, &[(1, 5), (8, 10)]),
            &weight_for_tsid(2, &[(20, 24)]),
        ]);
        let subtract_left = weight_for_tsid(1, &[(2, 3), (9, 12)]);
        let subtract_right = weight_for_tsid(2, &[(21, 22)]);

        let mut cache = ScopedWeightOpCache::default();
        let bulk = cache.difference_many(&base, [&subtract_left, &subtract_right]);

        let sequential = base.difference(&subtract_left).difference(&subtract_right);
        assert_eq!(bulk, sequential);
    }
}

impl Eq for Weight {}

impl super::leveled_gss::Merge for Weight {
    fn merge(&self, other: &Self) -> Self {
        self.union(other)
    }
}

impl std::hash::Hash for Weight {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.structural_hash_cached());
    }
}

impl std::fmt::Display for Weight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return write!(f, "∅");
        }
        if self.is_full() {
            return write!(f, "ALL");
        }

        let parts: Vec<String> = self
            .0
            .range_values()
            .map(|(range, tokens)| {
                let tsid = if range.start() == range.end() {
                    format!("{}", range.start())
                } else {
                    format!("{}..={}", range.start(), range.end())
                };
                format!("{tsid}→{}", rangeset_to_string(tokens.as_ref()))
            })
            .collect();
        write!(f, "{}", parts.join("; "))
    }
}

const WEIGHT_ALL_SENTINEL: u32 = u32::MAX;

impl Serialize for Weight {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_serde().serialize(serializer)
    }
}

fn weight_from_serde_entries(entries: Vec<WeightSerdeEntry>) -> Weight {
    let mut builder = CompactRangeBuilder::new();
    for entry in entries {
        let tokens = rangeset_from_ranges(
            entry.tokens.into_iter().map(|token| token[0]..=token[1]),
        );
        if tokens.is_empty() {
            continue;
        }
        builder.push(
            entry.tsid[0],
            entry.tsid[1],
            shared_rangeset(tokens),
        );
    }
    builder.finish()
}

impl<'de> Deserialize<'de> for Weight {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let serde_weight = WeightSerde::deserialize(deserializer)?;
        if serde_weight.all {
            return Ok(Self::all());
        }
        Ok(weight_from_serde_entries(serde_weight.entries))
    }
}
