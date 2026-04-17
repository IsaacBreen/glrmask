use range_set_blaze::{CheckSortedDisjoint, RangeMapBlaze, RangeSetBlaze, SortedDisjointMap};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;
use dashmap::DashMap;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

#[derive(Debug, Clone)]
pub struct Weight(pub Arc<WeightMap>);

type SharedTokenSet = Arc<RangeSetBlaze<u32>>;
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

fn prune_dead_token_sets() {
    GLOBAL_TOKEN_SETS.retain(|_, weak| weak.strong_count() > 0);
}

fn prune_dead_weights() {
    GLOBAL_WEIGHTS.retain(|_, bucket| {
        bucket.retain(|weak| weak.strong_count() > 0);
        !bucket.is_empty()
    });
}

fn maybe_cleanup_token_sets() {
    if TOKEN_INSERTS_SINCE_CLEANUP.fetch_add(1, Ordering::Relaxed) + 1
        >= INTERNER_CLEANUP_INTERVAL
    {
        // Best-effort: swap counter to 0 and prune. Racy but harmless.
        TOKEN_INSERTS_SINCE_CLEANUP.store(0, Ordering::Relaxed);
        prune_dead_token_sets();
    }
}

fn maybe_cleanup_weights() {
    if WEIGHT_INSERTS_SINCE_CLEANUP.fetch_add(1, Ordering::Relaxed) + 1
        >= INTERNER_CLEANUP_INTERVAL
    {
        WEIGHT_INSERTS_SINCE_CLEANUP.store(0, Ordering::Relaxed);
        prune_dead_weights();
    }
}

fn interner_clear_all() {
    GLOBAL_TOKEN_SETS.clear();
    GLOBAL_WEIGHTS.clear();
    TOKEN_INSERTS_SINCE_CLEANUP.store(0, Ordering::Relaxed);
    WEIGHT_INSERTS_SINCE_CLEANUP.store(0, Ordering::Relaxed);
}

fn interner_clear_stale() {
    prune_dead_token_sets();
    prune_dead_weights();
    TOKEN_INSERTS_SINCE_CLEANUP.store(0, Ordering::Relaxed);
    WEIGHT_INSERTS_SINCE_CLEANUP.store(0, Ordering::Relaxed);
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

fn shared_token_union(left: &SharedTokenSet, right: &SharedTokenSet) -> SharedTokenSet {
    if same_shared_token_set(left, right) || left.as_ref().is_subset(right.as_ref()) {
        Arc::clone(right)
    } else if right.as_ref().is_subset(left.as_ref()) {
        Arc::clone(left)
    } else {
        shared_rangeset(left.as_ref().clone() | right.as_ref().clone())
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
    } else {
        let overlap = left.as_ref() & right.as_ref();
        (!overlap.is_empty()).then(|| shared_rangeset(overlap))
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
struct WeightOpKey {
    kind: WeightOpKind,
    left: usize,
    right: usize,
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

/// Cached memo entry: stores the result AND weak references to both operands.
/// The operand weak refs guard against the ABA problem: if either operand's
/// Arc was dropped and a new Arc reuses the same address, the operand weak
/// ref will fail to upgrade, and the stale entry is discarded.
struct WeightOpMemoEntry {
    result: Weak<WeightMap>,
    left_operand: Weak<WeightMap>,
    right_operand: Weak<WeightMap>,
}

/// Global generation counter. Incremented by `clear_weight_op_caches()` so
/// that every thread's thread-local memo detects the invalidation on next access.
static WEIGHT_OP_MEMO_GENERATION: AtomicU64 = AtomicU64::new(0);

#[derive(Default)]
struct WeightOpMemo {
    results: FxHashMap<WeightOpKey, WeightOpMemoEntry>,
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
        self.inserts_since_cleanup = 0;
    }

    fn clear_all(&mut self) {
        self.results.clear();
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
}

thread_local! {
    static WEIGHT_OP_MEMO: RefCell<WeightOpMemo> = RefCell::new(WeightOpMemo::default());
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

/// Clear the global interned-weight tables entirely.
pub fn clear_all_weights() {
    interner_clear_all();
}

/// Prune only dead entries from the global weight interner.
pub fn clear_stale_weights() {
    interner_clear_stale();
}

/// Clear weight-operation memo caches on **all** threads.
///
/// Increments the global generation counter so that every thread's
/// thread-local memo is lazily cleared on its next access.
pub fn clear_weight_op_caches() {
    WEIGHT_OP_MEMO_GENERATION.fetch_add(1, Ordering::Release);
}

/// Compatibility wrapper retaining the previous behavior.
pub fn clear_weight_caches() {
    clear_weight_op_caches();
    clear_all_weights();
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

enum NextMeaningfulWeight<'a> {
    End,
    Full,
    Weight(&'a Weight),
}

fn next_meaningful_weight<'a, I>(iter: &mut I) -> NextMeaningfulWeight<'a>
where
    I: Iterator<Item = &'a Weight>,
{
    for weight in iter {
        if weight.is_full() {
            return NextMeaningfulWeight::Full;
        }
        if !weight.is_empty() {
            return NextMeaningfulWeight::Weight(weight);
        }
    }
    NextMeaningfulWeight::End
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

/// Direct multi-way union that avoids creating O(N) intermediate Weight objects.
/// Uses a sweep-line approach: collects all range entries, sorts boundary points,
/// and computes the union of active token sets at each boundary interval.
fn union_all_multiway(weights: &[&Weight]) -> Weight {
    // Collect all (start, end, tokens) entries from all weights
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

    // Compute sorted unique boundaries
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

    // Sort entries by start for efficient scanning
    all_entries.sort_unstable_by_key(|e| e.start);

    let mut builder = CompactRangeBuilder::new();
    let mut scan_start = 0usize;

    for window in boundaries.windows(2) {
        let interval_start = window[0] as u32;
        let interval_end = (window[1] - 1) as u32;

        // Advance scan_start past entries that end before this interval
        while scan_start < all_entries.len() && all_entries[scan_start].end < interval_start {
            scan_start += 1;
        }

        // Collect all token sets active in this interval
        let mut merged_tokens: Option<SharedTokenSet> = None;
        for entry in &all_entries[scan_start..] {
            if entry.start > interval_start {
                break;
            }
            if entry.start <= interval_start && entry.end >= interval_end {
                merged_tokens = Some(match merged_tokens {
                    Some(existing) => shared_token_union(&existing, &entry.tokens),
                    None => Arc::clone(&entry.tokens),
                });
            }
        }

        if let Some(tokens) = merged_tokens {
            builder.push(interval_start, interval_end, tokens);
        } else {
            builder.flush();
        }
    }

    builder.finish()
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

    while let (Some((left_range, left_tokens)), Some((right_range, right_tokens))) = (left_entry, right_entry) {
        let start = (*left_range.start()).max(*right_range.start());
        let end = (*left_range.end()).min(*right_range.end());

        if start <= end {
            if let Some(tokens) = shared_token_intersection(left_tokens, right_tokens) {
                builder.push(start, end, tokens);
            }
        }

        let left_end = *left_range.end();
        let right_end = *right_range.end();
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

    builder.finish()
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
    pub(crate) fn ptr_key(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
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

    pub fn empty() -> Self {
        EMPTY_WEIGHT.clone()
    }

    pub fn all() -> Self {
        ALL_WEIGHT.clone()
    }

    pub fn from_compact_ranges<I, J>(entries: I) -> Self
    where
        I: IntoIterator<Item = (std::ops::RangeInclusive<u32>, J)>,
        J: IntoIterator<Item = std::ops::RangeInclusive<u32>>,
    {
        let mut out = Self::empty();
        for (tsid_range, token_ranges) in entries {
            let token_ranges: Vec<_> = token_ranges.into_iter().collect();
            out.insert(tsid_range, &token_ranges);
        }
        out
    }

    /// Create a weight where all tsids in the range share the same token set.
    /// Avoids the expensive expand/compress cycle of `from_compact_ranges`.
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

    pub fn token_union(&self) -> RangeSetBlaze<u32> {
        if self.is_full() {
            return sentinel_token_set();
        }
        let mut out = RangeSetBlaze::new();
        for (_, tokens) in self.0.range_values() {
            out = out | tokens.as_ref().clone();
        }
        out
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

    pub fn estimated_size_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.num_ranges()
                * (std::mem::size_of::<u32>() + std::mem::size_of::<Arc<RangeSetBlaze<u32>>>())
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
        with_memoized_weight_op(WeightOpKind::Union, self, other, || {
            if let (Some(left), Some(right)) = (single_compact_entry(self), single_compact_entry(other)) {
                combine_single_entries(&left, &right, union_token_sets)
            } else {
                combine_compact_entries(self, other, union_token_sets)
            }
        })
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

        match meaningful.len() {
            0 => return Self::empty(),
            1 => return meaningful[0].clone(),
            _ => {}
        }

        // Dedup by Arc pointer: union(a, a) = a, so identical weights are redundant.
        // Only worthwhile when there are enough inputs that duplicates are likely
        // and the sort cost is amortized by skipping expensive union operations.
        if meaningful.len() > 4 {
            meaningful.sort_unstable_by_key(|w| w.ptr_key());
            meaningful.dedup_by_key(|w| w.ptr_key());
            if meaningful.len() == 1 {
                return meaningful[0].clone();
            }
        }

        if let Some(result) = union_all_single_tsid_entries(&meaningful) {
            return result;
        }

        // For many inputs, use direct multiway sweep to avoid O(N) intermediate
        // Weight allocations, interner locks, and memoization overhead.
        if meaningful.len() > 4 {
            return union_all_multiway(&meaningful);
        }

        let mut iter = meaningful.into_iter();
        let mut acc = iter.next().unwrap().clone();
        for weight in iter {
            acc = acc.union(weight);
        }
        acc
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
        with_memoized_weight_op(WeightOpKind::Intersection, self, other, || {
            if let (Some(left), Some(right)) = (single_compact_entry(self), single_compact_entry(other)) {
                combine_single_entries(&left, &right, intersect_token_sets)
            } else if let Some(single) = single_compact_entry(self) {
                intersect_single_entry_with_weight(&single, other)
            } else if let Some(single) = single_compact_entry(other) {
                intersect_single_entry_with_weight(&single, self)
            } else {
                intersect_weights(self, other)
            }
        })
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
        with_memoized_weight_op(WeightOpKind::Difference, self, other, || {
            combine_compact_entries(self, other, difference_token_sets)
        })
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

impl Eq for Weight {}

impl super::leveled_gss::Merge for Weight {
    fn merge(&self, other: &Self) -> Self {
        self.union(other)
    }
}

impl std::hash::Hash for Weight {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.is_full().hash(state);
        if !self.is_full() {
            for (range, tokens) in self.0.range_values() {
                range.hash(state);
                tokens.as_ref().hash(state);
            }
        }
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

impl<'de> Deserialize<'de> for Weight {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let serde_weight = WeightSerde::deserialize(deserializer)?;
        if serde_weight.all {
            return Ok(Self::all());
        }
        Ok(Self::from_compact_ranges(serde_weight.entries.into_iter().map(|entry| {
            (
                entry.tsid[0]..=entry.tsid[1],
                entry.tokens.into_iter().map(|token| token[0]..=token[1]),
            )
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::sync::Mutex;

    static WEIGHT_CACHE_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    #[test]
    fn test_weight_empty() {
        let w = Weight::empty();
        assert!(w.is_empty());
    }

    #[test]
    fn test_weight_all_is_full() {
        let w = Weight::all();
        assert!(w.is_full());
        assert!(!w.is_empty());
    }

    #[test]
    fn test_weight_from_compact_ranges_shape() {
        let w = Weight::from_compact_ranges(vec![
            (0..=2, vec![10..=12, 20..=21]),
            (5..=5, vec![7..=9]),
        ]);
        assert!(w.estimated_size_bytes() >= std::mem::size_of::<Weight>());
    }

    #[test]
    fn test_weight_insert_shape() {
        let mut w = Weight::empty();
        w.insert(0..=2, &[10..=12, 20..=21]);
        assert!(w.estimated_size_bytes() >= std::mem::size_of::<Weight>());
    }

    #[test]
    fn test_weight_token_union_shape() {
        let w = Weight::from_compact_ranges(vec![
            (0..=2, vec![10..=12, 20..=21]),
            (5..=5, vec![7..=9]),
        ]);
        let _tokens = w.token_union();
    }

    #[test]
    fn test_weight_estimated_size_bytes_has_base_size() {
        let w = Weight::empty();
        assert!(w.estimated_size_bytes() >= std::mem::size_of::<Weight>());
    }

    #[test]
    fn test_weight_union() {
        let a = Weight::empty();
        let b = Weight::all();
        let u = a.union(&b);
        assert!(u.is_full());
    }

    #[test]
    fn test_weight_intersection() {
        let a = Weight::empty();
        let b = Weight::all();
        let i = a.intersection(&b);
        assert!(i.is_empty());
    }

    #[test]
    fn test_weight_difference() {
        let a = Weight::all();
        let b = Weight::empty();
        let d = a.difference(&b);
        assert!(d.is_full());
    }

    #[test]
    fn test_weight_union_handles_misaligned_ranges() {
        let a = Weight::from_compact_ranges(vec![(0..=2, vec![1..=2]), (5..=5, vec![7..=7])]);
        let b = Weight::from_compact_ranges(vec![(1..=5, vec![2..=3])]);

        let union = a.union(&b);

        assert_eq!(union.tokens_for_tsid(0), rangeset_from_ranges([1..=2]));
        assert_eq!(union.tokens_for_tsid(1), rangeset_from_ranges([1..=3]));
        assert_eq!(union.tokens_for_tsid(3), rangeset_from_ranges([2..=3]));
        assert_eq!(union.tokens_for_tsid(5), rangeset_from_ranges([2..=3, 7..=7]));
    }

    #[test]
    fn test_weight_intersection_handles_misaligned_ranges() {
        let a = Weight::from_compact_ranges(vec![(0..=2, vec![1..=2]), (4..=5, vec![5..=5])]);
        let b = Weight::from_compact_ranges(vec![(1..=3, vec![2..=3]), (5..=6, vec![5..=6])]);

        let intersection = a.intersection(&b);

        assert_eq!(intersection.tokens_for_tsid(0), RangeSetBlaze::new());
        assert_eq!(intersection.tokens_for_tsid(1), rangeset_from_ranges([2..=2]));
        assert_eq!(intersection.tokens_for_tsid(2), rangeset_from_ranges([2..=2]));
        assert_eq!(intersection.tokens_for_tsid(5), rangeset_from_ranges([5..=5]));
        assert_eq!(intersection.num_ranges(), 2);
    }

    #[test]
    fn test_weight_difference_handles_misaligned_ranges() {
        let a = Weight::from_compact_ranges(vec![(0..=2, vec![1..=2]), (4..=5, vec![5..=5])]);
        let b = Weight::from_compact_ranges(vec![(1..=3, vec![2..=3]), (5..=6, vec![5..=6])]);

        let difference = a.difference(&b);

        assert_eq!(difference.tokens_for_tsid(0), rangeset_from_ranges([1..=2]));
        assert_eq!(difference.tokens_for_tsid(1), rangeset_from_ranges([1..=1]));
        assert_eq!(difference.tokens_for_tsid(2), rangeset_from_ranges([1..=1]));
        assert_eq!(difference.tokens_for_tsid(4), rangeset_from_ranges([5..=5]));
        assert_eq!(difference.tokens_for_tsid(5), RangeSetBlaze::new());
    }

    #[test]
    fn test_weight_union_coalesces_adjacent_equal_segments() {
        let a = Weight::from_compact_ranges(vec![(0..=10, vec![1..=1])]);
        let b = Weight::from_compact_ranges(vec![(3..=5, vec![1..=1])]);

        let union = a.union(&b);

        assert_eq!(union, a);
        assert_eq!(union.num_ranges(), 1);
    }

    #[test]
    fn test_weight_clear() {
        let mut w = Weight::all();
        w.clear();
        assert!(w.is_empty());
    }

    #[test]
    fn test_weight_display() {
        let empty = Weight::empty();
        let all = Weight::all();
        assert_eq!(format!("{empty}"), "∅");
        assert_eq!(format!("{all}"), "ALL");
    }

    #[test]
    fn test_weight_equality() {
        let a = Weight::empty();
        let b = Weight::empty();
        assert_eq!(a, b);
        let c = Weight::all();
        assert_ne!(a, c);
    }

    #[test]
    fn test_weight_map_interning_reuses_arc_for_equal_weights() {
        let _guard = WEIGHT_CACHE_TEST_LOCK.lock().unwrap();

        let a = Weight::from_compact_ranges(vec![
            (0..=2, vec![10..=12, 20..=21]),
            (5..=5, vec![7..=9]),
        ]);
        let b = Weight::from_compact_ranges(vec![
            (0..=2, vec![10..=12, 20..=21]),
            (5..=5, vec![7..=9]),
        ]);
        assert_eq!(a, b);
        assert_eq!(a.ptr_key(), b.ptr_key());
    }

    #[test]
    fn test_weight_map_interning_reuses_arc_after_union() {
        let _guard = WEIGHT_CACHE_TEST_LOCK.lock().unwrap();

        let left = Weight::from_compact_ranges(vec![(0..=1, vec![1..=2])]);
        let right = Weight::from_compact_ranges(vec![(2..=3, vec![3..=4])]);

        let via_union = left.union(&right);
        let direct = Weight::from_compact_ranges(vec![(0..=1, vec![1..=2]), (2..=3, vec![3..=4])]);

        assert_eq!(via_union, direct);
        assert_eq!(via_union.ptr_key(), direct.ptr_key());
    }

    #[test]
    fn test_weight_op_memo_discards_entries_when_operands_are_gone() {
        let _guard = WEIGHT_CACHE_TEST_LOCK.lock().unwrap();
        clear_weight_caches();

        let stale_result = Weight::from_compact_ranges(vec![(0..=0, vec![7..=7])]);
        let stale_operand_weaks = {
            let left = Weight::from_compact_ranges(vec![(0..=0, vec![1..=1])]);
            let right = Weight::from_compact_ranges(vec![(0..=0, vec![2..=2])]);
            (
                Arc::downgrade(&left.0),
                Arc::downgrade(&right.0),
            )
        };

        let live_left = Weight::from_compact_ranges(vec![(0..=0, vec![3..=3])]);
        let live_right = Weight::from_compact_ranges(vec![(0..=0, vec![4..=4])]);
        let lookup_key = WeightOpKey::new(WeightOpKind::Union, live_left.ptr_key(), live_right.ptr_key());

        // Sync the thread-local generation before direct insertion.
        with_weight_op_memo(|memo| {
            memo.results.insert(
                lookup_key,
                WeightOpMemoEntry {
                    result: Arc::downgrade(&stale_result.0),
                    left_operand: stale_operand_weaks.0,
                    right_operand: stale_operand_weaks.1,
                },
            );
        });

        assert!(lookup_memoized_weight_op(WeightOpKind::Union, &live_left, &live_right).is_none());

        with_weight_op_memo(|memo| {
            assert!(!memo.results.contains_key(&lookup_key));
        });
    }

    #[test]
    fn test_clear_all_weights_breaks_future_intern_reuse() {
        let _guard = WEIGHT_CACHE_TEST_LOCK.lock().unwrap();
        clear_weight_caches();

        let first = Weight::from_compact_ranges(vec![(0..=1, vec![1..=2])]);
        let first_ptr = first.ptr_key();

        clear_all_weights();

        let second = Weight::from_compact_ranges(vec![(0..=1, vec![1..=2])]);
        assert_ne!(first_ptr, second.ptr_key());
    }

    #[test]
    fn test_clear_stale_weights_prunes_dead_interner_entries() {
        let _guard = WEIGHT_CACHE_TEST_LOCK.lock().unwrap();
        clear_weight_caches();

        let stale_tokens = RangeSetBlaze::from_iter([1..=2]);
        let fingerprint = {
            let mut map = WeightMap::new();
            map.extend_simple(std::iter::once((0..=1, Arc::new(stale_tokens.clone()))));
            weight_map_fingerprint(&map)
        };
        let stale_token_weak = {
            let shared = Arc::new(stale_tokens.clone());
            let weak = Arc::downgrade(&shared);
            drop(shared);
            weak
        };

        let stale_weight_weak = {
            let mut map = WeightMap::new();
            map.extend_simple(std::iter::once((0..=1, Arc::new(stale_tokens.clone()))));
            let shared = Arc::new(map);
            let weak = Arc::downgrade(&shared);
            GLOBAL_TOKEN_SETS.insert(stale_tokens.clone(), stale_token_weak);
            GLOBAL_WEIGHTS.insert(fingerprint, vec![weak.clone()]);
            drop(shared);
            weak
        };

        assert_eq!(stale_weight_weak.strong_count(), 0);

        clear_stale_weights();

        assert!(!GLOBAL_WEIGHTS.contains_key(&fingerprint));
        if let Some(live_entry) = GLOBAL_TOKEN_SETS.get(&stale_tokens) {
            assert!(live_entry.strong_count() > 0);
        }
    }

    #[test]
    fn test_clear_weight_op_caches_empties_thread_local_memo() {
        let _guard = WEIGHT_CACHE_TEST_LOCK.lock().unwrap();
        clear_weight_caches();

        let left = Weight::from_compact_ranges(vec![(0..=0, vec![1..=1])]);
        let right = Weight::from_compact_ranges(vec![(0..=0, vec![2..=2])]);
        let _ = left.union(&right);

        with_weight_op_memo(|memo| {
            assert!(!memo.results.is_empty());
        });

        clear_weight_op_caches();

        // After clear, next access should find empty memo (lazy invalidation).
        with_weight_op_memo(|memo| {
            assert!(memo.results.is_empty());
        });
    }

    #[test]
    fn test_weight_serde_empty() {
        let w = Weight::empty();
        let json = serde_json::to_string(&w).unwrap();
        let w2: Weight = serde_json::from_str(&json).unwrap();
        assert_eq!(w, w2);
    }

    #[test]
    fn test_weight_serde_all() {
        let w = Weight::all();
        let json = serde_json::to_string(&w).unwrap();
        let w2: Weight = serde_json::from_str(&json).unwrap();
        assert_eq!(w, w2);
    }
}
