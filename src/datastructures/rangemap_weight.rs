use std::cell::Cell;
use std::env;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

use dashmap::{DashMap, DashSet};
use lru::LruCache;
use once_cell::sync::Lazy;
use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};

use crate::datastructures::abstract_weight::{current_num_tsids, normalize_num_tsids, WeightBackend};
use crate::datastructures::cache;
use crate::datastructures::hybrid_bitset::RangeSet;
use profiler_macro::time_it;

const DIVIDE_CACHE_CAPACITY: usize = 50_000;
const DIVIDE_RHS_COMP_CACHE_CAPACITY: usize = 10_000;

static RANGEMAP_WEIGHT_INTERNER: Lazy<DashSet<Arc<RangeMapWeight>>> = Lazy::new(DashSet::new);

// Sharded concurrent op cache: eliminates Mutex contention during parallel determinize
// while maintaining cross-thread result sharing.
static RANGEMAP_OP_CACHE: Lazy<DashMap<OpKey, Arc<RangeMapWeight>>> = Lazy::new(DashMap::new);

// Separate cache for divide operations to avoid polluting the main op cache
static RANGEMAP_DIVIDE_CACHE: Lazy<Mutex<LruCache<(usize, usize), Arc<RangeMapWeight>>>> = Lazy::new(|| {
    Mutex::new(LruCache::new(NonZeroUsize::new(DIVIDE_CACHE_CAPACITY).unwrap()))
});
type RhsCompCache = HashMap<usize, RangeSet>;
static RANGEMAP_DIVIDE_RHS_COMP_CACHE: Lazy<Mutex<LruCache<usize, Arc<RhsCompCache>>>> =
    Lazy::new(|| {
        Mutex::new(LruCache::new(
            NonZeroUsize::new(DIVIDE_RHS_COMP_CACHE_CAPACITY).unwrap(),
        ))
    });
static FULL_TSIDS_CACHE: Lazy<Mutex<HashMap<usize, RangeSet>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static RANGEMAP_TSID_OUTER: Lazy<bool> = Lazy::new(|| {
    env::var("RANGEMAP_TSID_OUTER")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    .unwrap_or(false)
});

thread_local! {
    static OP_CACHE_OR_MISS_ACTIVE: Cell<bool> = Cell::new(false);
}

static OP_CACHE_PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);
static OP_CACHE_OR_HITS: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISSES: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_TIME_NS: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_COMPUTES: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_PREP_NS: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_UNION_ASYM_NS: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_MERGE_NS: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_INTERN_NS: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_LEFT_RANGES_TOTAL: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_RIGHT_RANGES_TOTAL: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_ASYM_COUNT: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_MERGE_COUNT: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_RANGESET_UNION_NS: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_RANGESET_UNION_COUNT: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_RANGESET_LEFT_RANGES_TOTAL: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_RANGESET_RIGHT_RANGES_TOTAL: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_SEGMENTS_TOTAL: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_SEGMENTS_BOTH: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_SEGMENTS_LEFT_ONLY: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_SEGMENTS_RIGHT_ONLY: AtomicU64 = AtomicU64::new(0);
static OP_CACHE_OR_MISS_SEGMENTS_NONE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Default)]
pub(crate) struct OpCacheOrMissDetail {
    pub(crate) prep_ns: u64,
    pub(crate) union_asym_ns: u64,
    pub(crate) merge_ns: u64,
    pub(crate) intern_ns: u64,
    pub(crate) left_ranges_total: u64,
    pub(crate) right_ranges_total: u64,
    pub(crate) asym_count: u64,
    pub(crate) merge_count: u64,
    pub(crate) rangeset_union_ns: u64,
    pub(crate) rangeset_union_count: u64,
    pub(crate) rangeset_left_ranges_total: u64,
    pub(crate) rangeset_right_ranges_total: u64,
    pub(crate) segments_total: u64,
    pub(crate) segments_both: u64,
    pub(crate) segments_left_only: u64,
    pub(crate) segments_right_only: u64,
    pub(crate) segments_none: u64,
}

pub(crate) fn set_op_cache_profile_enabled(enabled: bool) {
    OP_CACHE_PROFILE_ENABLED.store(enabled, AtomicOrdering::Relaxed);
}

pub(crate) fn reset_op_cache_or_counters() {
    OP_CACHE_OR_HITS.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISSES.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_TIME_NS.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_COMPUTES.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_PREP_NS.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_UNION_ASYM_NS.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_MERGE_NS.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_INTERN_NS.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_LEFT_RANGES_TOTAL.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_RIGHT_RANGES_TOTAL.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_ASYM_COUNT.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_MERGE_COUNT.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_RANGESET_UNION_NS.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_RANGESET_UNION_COUNT.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_RANGESET_LEFT_RANGES_TOTAL.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_RANGESET_RIGHT_RANGES_TOTAL.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_SEGMENTS_TOTAL.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_SEGMENTS_BOTH.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_SEGMENTS_LEFT_ONLY.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_SEGMENTS_RIGHT_ONLY.store(0, AtomicOrdering::Relaxed);
    OP_CACHE_OR_MISS_SEGMENTS_NONE.store(0, AtomicOrdering::Relaxed);
}

pub(crate) fn op_cache_or_counters() -> (u64, u64) {
    (
        OP_CACHE_OR_HITS.load(AtomicOrdering::Relaxed),
        OP_CACHE_OR_MISSES.load(AtomicOrdering::Relaxed),
    )
}

pub(crate) fn op_cache_or_miss_time_counters() -> (u64, u64) {
    (
        OP_CACHE_OR_MISS_TIME_NS.load(AtomicOrdering::Relaxed),
        OP_CACHE_OR_MISS_COMPUTES.load(AtomicOrdering::Relaxed),
    )
}

pub(crate) fn op_cache_or_miss_detail_counters() -> OpCacheOrMissDetail {
    OpCacheOrMissDetail {
        prep_ns: OP_CACHE_OR_MISS_PREP_NS.load(AtomicOrdering::Relaxed),
        union_asym_ns: OP_CACHE_OR_MISS_UNION_ASYM_NS.load(AtomicOrdering::Relaxed),
        merge_ns: OP_CACHE_OR_MISS_MERGE_NS.load(AtomicOrdering::Relaxed),
        intern_ns: OP_CACHE_OR_MISS_INTERN_NS.load(AtomicOrdering::Relaxed),
        left_ranges_total: OP_CACHE_OR_MISS_LEFT_RANGES_TOTAL.load(AtomicOrdering::Relaxed),
        right_ranges_total: OP_CACHE_OR_MISS_RIGHT_RANGES_TOTAL.load(AtomicOrdering::Relaxed),
        asym_count: OP_CACHE_OR_MISS_ASYM_COUNT.load(AtomicOrdering::Relaxed),
        merge_count: OP_CACHE_OR_MISS_MERGE_COUNT.load(AtomicOrdering::Relaxed),
        rangeset_union_ns: OP_CACHE_OR_MISS_RANGESET_UNION_NS.load(AtomicOrdering::Relaxed),
        rangeset_union_count: OP_CACHE_OR_MISS_RANGESET_UNION_COUNT.load(AtomicOrdering::Relaxed),
        rangeset_left_ranges_total: OP_CACHE_OR_MISS_RANGESET_LEFT_RANGES_TOTAL.load(AtomicOrdering::Relaxed),
        rangeset_right_ranges_total: OP_CACHE_OR_MISS_RANGESET_RIGHT_RANGES_TOTAL.load(AtomicOrdering::Relaxed),
        segments_total: OP_CACHE_OR_MISS_SEGMENTS_TOTAL.load(AtomicOrdering::Relaxed),
        segments_both: OP_CACHE_OR_MISS_SEGMENTS_BOTH.load(AtomicOrdering::Relaxed),
        segments_left_only: OP_CACHE_OR_MISS_SEGMENTS_LEFT_ONLY.load(AtomicOrdering::Relaxed),
        segments_right_only: OP_CACHE_OR_MISS_SEGMENTS_RIGHT_ONLY.load(AtomicOrdering::Relaxed),
        segments_none: OP_CACHE_OR_MISS_SEGMENTS_NONE.load(AtomicOrdering::Relaxed),
    }
}

// --- Profiling ---
// Legacy rangemap profiling removed; keep no-op hooks for callers.
pub fn reset_intern_wall_time() {}

pub fn reset_profiling() {}

pub fn print_profiling(_label: &str) {}

pub fn print_intern_wall_time(_label: &str) {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct OpKey {
    op: cache::BinOp,
    a: usize,
    b: usize,
}

fn op_key(op: cache::BinOp, a: &Arc<RangeMapWeight>, b: &Arc<RangeMapWeight>) -> OpKey {
    OpKey {
        op,
        a: Arc::as_ptr(a) as usize,
        b: Arc::as_ptr(b) as usize,
    }
}

fn is_interned_rangemap(weight: &Arc<RangeMapWeight>) -> bool {
    if let Some(existing) = RANGEMAP_WEIGHT_INTERNER.get(weight) {
        Arc::ptr_eq(&existing, weight)
    } else {
        false
    }
}

// Sharded concurrent op cache using DashMap: each shard has its own lock,
// so contention is spread across ~64 shards instead of a single global Mutex.
// Pointer-based keys are safe because the DashSet interner holds Arc references,
// preventing deallocation and pointer reuse during a compilation run.

fn invalidate_rangemap_op_cache_for_ptr(_ptr: usize) {
    // No-op: pointer reuse doesn't happen while weights are held in the interner.
    // The DashMap cache entries remain valid for the entire compilation run.
}

fn get_op_cache(op: cache::BinOp, a: &Arc<RangeMapWeight>, b: &Arc<RangeMapWeight>) -> Option<Arc<RangeMapWeight>> {
    let profile_enabled = OP_CACHE_PROFILE_ENABLED.load(AtomicOrdering::Relaxed);
    let key = op_key(op, a, b);
    if let Some(hit) = RANGEMAP_OP_CACHE.get(&key) {
        if profile_enabled && matches!(op, cache::BinOp::Or) {
            OP_CACHE_OR_HITS.fetch_add(1, AtomicOrdering::Relaxed);
        }
        return Some(hit.value().clone());
    }
    if matches!(op, cache::BinOp::And | cache::BinOp::Or | cache::BinOp::Xor) {
        let swapped = op_key(op, b, a);
        if let Some(hit) = RANGEMAP_OP_CACHE.get(&swapped) {
            if profile_enabled && matches!(op, cache::BinOp::Or) {
                OP_CACHE_OR_HITS.fetch_add(1, AtomicOrdering::Relaxed);
            }
            return Some(hit.value().clone());
        }
    }
    if profile_enabled && matches!(op, cache::BinOp::Or) {
        OP_CACHE_OR_MISSES.fetch_add(1, AtomicOrdering::Relaxed);
    }
    None
}

fn put_op_cache(
    op: cache::BinOp,
    a: Arc<RangeMapWeight>,
    b: Arc<RangeMapWeight>,
    result: Arc<RangeMapWeight>,
) {
    let key = op_key(op, &a, &b);
    RANGEMAP_OP_CACHE.insert(key, result);
}

fn build_rhs_comp_cache(rhs: &Arc<RangeMapWeight>) -> Arc<RhsCompCache> {
    let full_tsids = RangeMapWeight::full_inner_set(rhs.num_tsids());
    let mut out: RhsCompCache = HashMap::new();
    for (_, rv) in rhs.map.range_values() {
        let ptr = Arc::as_ptr(&rv.inner) as usize;
        out.entry(ptr).or_insert_with(|| &full_tsids - rv);
    }
    Arc::new(out)
}

fn get_rhs_comp_cache(rhs: &Arc<RangeMapWeight>) -> Option<Arc<RhsCompCache>> {
    let key = Arc::as_ptr(rhs) as usize;
    {
        let mut cache = RANGEMAP_DIVIDE_RHS_COMP_CACHE.lock().unwrap();
        if let Some(hit) = cache.get(&key) {
            return Some(hit.clone());
        }
    }
    let built = build_rhs_comp_cache(rhs);
    let mut cache = RANGEMAP_DIVIDE_RHS_COMP_CACHE.lock().unwrap();
    cache.push(key, built.clone());
    Some(built)
}

#[track_caller]
pub fn intern_rangemap(weight: RangeMapWeight) -> Arc<RangeMapWeight> {
    if let Some(existing) = RANGEMAP_WEIGHT_INTERNER.get(&weight) {
        return Arc::clone(&*existing);
    }
    let arc = Arc::new(weight);
    let inserted = RANGEMAP_WEIGHT_INTERNER.insert(arc.clone());
    if inserted {
        let ptr = Arc::as_ptr(&arc) as usize;
        invalidate_rangemap_op_cache_for_ptr(ptr);
    }
    if inserted {
        arc
    } else if let Some(existing) = RANGEMAP_WEIGHT_INTERNER.get(&arc) {
        Arc::clone(&*existing)
    } else {
        arc
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeMapWeight {
    /// Maps token_id -> set of tsid values (stored as ranges over token_id).
    pub(crate) map: RangeMapBlaze<usize, RangeSet>,
    pub(crate) num_tsids: usize,
    cached_hash: u64,
}

impl serde::Serialize for RangeMapWeight {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        // Serialize map as vec of (start, end, tsid_set) entries
        let entries: Vec<(usize, usize, &RangeSet)> = self.map.range_values()
            .map(|(r, v)| (*r.start(), *r.end(), v))
            .collect();
        let mut s = serializer.serialize_struct("RangeMapWeight", 2)?;
        s.serialize_field("entries", &entries)?;
        s.serialize_field("num_tsids", &self.num_tsids)?;
        s.end()
    }
}

impl<'de> serde::Deserialize<'de> for RangeMapWeight {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct RangeMapWeightProxy {
            entries: Vec<(usize, usize, RangeSet)>,
            num_tsids: usize,
        }
        let proxy = RangeMapWeightProxy::deserialize(deserializer)?;
        let mut map = RangeMapBlaze::new();
        for (start, end, tsid_set) in proxy.entries {
            map.ranges_insert(start..=end, tsid_set);
        }
        let cached_hash = Self::compute_hash(&map, proxy.num_tsids);
        Ok(RangeMapWeight { map, num_tsids: proxy.num_tsids, cached_hash })
    }
}

impl RangeMapWeight {
    fn tsid_outer_enabled() -> bool {
        *RANGEMAP_TSID_OUTER
    }
    fn compute_hash(map: &RangeMapBlaze<usize, RangeSet>, num_tsids: usize) -> u64 {
        let mut hasher = DefaultHasher::new();
        num_tsids.hash(&mut hasher);
        for (token_range, tsid_set) in map.range_values() {
            token_range.start().hash(&mut hasher);
            token_range.end().hash(&mut hasher);
            for tsid_range in tsid_set.ranges() {
                tsid_range.start().hash(&mut hasher);
                tsid_range.end().hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    pub(crate) fn from_map(map: RangeMapBlaze<usize, RangeSet>, num_tsids: usize) -> Self {
        let num_tsids = normalize_num_tsids(num_tsids);
        let cached_hash = Self::compute_hash(&map, num_tsids);
        Self {
            map,
            num_tsids,
            cached_hash,
        }
    }

    fn refresh_cached_hash(&mut self) {
        self.cached_hash = Self::compute_hash(&self.map, self.num_tsids);
    }

    fn map_range_count(map: &RangeMapBlaze<usize, RangeSet>) -> usize {
        map.range_values().len()
    }

    fn full_tsids(num_tsids: usize) -> RangeSet {
        let mut cache = FULL_TSIDS_CACHE.lock().unwrap();
        if let Some(cached) = cache.get(&num_tsids) {
            return cached.clone();
        }
        let full = Self::rangeset_from_ranges([0..=num_tsids.saturating_sub(1)]);
        cache.insert(num_tsids, full.clone());
        full
    }

    fn full_inner_set(num_tsids: usize) -> RangeSet {
        if Self::tsid_outer_enabled() {
            let max_token = crate::datastructures::get_max_llm_token();
            Self::rangeset_from_ranges([0..=max_token])
        } else {
            Self::full_tsids(num_tsids)
        }
    }

    fn max_inner_value(num_tsids: usize) -> usize {
        if Self::tsid_outer_enabled() {
            crate::datastructures::get_max_llm_token()
        } else {
            num_tsids.saturating_sub(1)
        }
    }

    fn rangeset_from_ranges<I: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(
        ranges: I,
    ) -> RangeSet {
        RangeSet::from(RangeSetBlaze::from_iter(ranges))
    }

    fn rangeset_complement_ranges(
        rhs: &RangeSet,
        max_tsid: usize,
        out: &mut Vec<std::ops::RangeInclusive<usize>>,
    ) {
        out.clear();
        let mut start = 0usize;
        for range in rhs.ranges() {
            if start > max_tsid {
                return;
            }
            let r_start = *range.start();
            if r_start > max_tsid {
                break;
            }
            if start < r_start {
                out.push(start..=r_start.saturating_sub(1));
            }
            let r_end = *range.end();
            if r_end >= max_tsid {
                return;
            }
            start = r_end.saturating_add(1);
        }
        if start <= max_tsid {
            out.push(start..=max_tsid);
        }
    }

    fn rangeset_union_with_complement_asymmetric(
        lhs: &RangeSet,
        rhs: &RangeSet,
        max_tsid: usize,
    ) -> RangeSet {
        if lhs.is_empty() {
            let mut comp_ranges = Vec::new();
            Self::rangeset_complement_ranges(rhs, max_tsid, &mut comp_ranges);
            return Self::rangeset_from_ranges(comp_ranges);
        }
        if rhs.is_empty() {
            return Self::rangeset_from_ranges([0..=max_tsid]);
        }
        if rhs.ranges_len() == 1 {
            if let Some(range) = rhs.ranges().next() {
                if *range.start() == 0 && *range.end() == max_tsid {
                    return lhs.clone();
                }
            }
        }

        let mut comp_ranges = Vec::new();
        Self::rangeset_complement_ranges(rhs, max_tsid, &mut comp_ranges);
        let mut lhs_iter = lhs.ranges();
        let mut rhs_iter = comp_ranges.into_iter();
        let mut lhs_next = lhs_iter.next();
        let mut rhs_next = rhs_iter.next();

        let mut out_ranges: Vec<std::ops::RangeInclusive<usize>> = Vec::new();
        let mut current_start: Option<usize> = None;
        let mut current_end: usize = 0;

        loop {
            let next_range = match (lhs_next.as_ref(), rhs_next.as_ref()) {
                (Some(l_range), Some(r_range)) => {
                    if l_range.start() <= r_range.start() {
                        let range = lhs_next.take().unwrap();
                        lhs_next = lhs_iter.next();
                        range
                    } else {
                        let range = rhs_next.take().unwrap();
                        rhs_next = rhs_iter.next();
                        range
                    }
                }
                (Some(_), None) => {
                    let range = lhs_next.take().unwrap();
                    lhs_next = lhs_iter.next();
                    range
                }
                (None, Some(_)) => {
                    let range = rhs_next.take().unwrap();
                    rhs_next = rhs_iter.next();
                    range
                }
                (None, None) => break,
            };

            let start = *next_range.start();
            let end = *next_range.end();
            if let Some(cur_start) = current_start {
                if start <= current_end.saturating_add(1) {
                    current_end = current_end.max(end);
                } else {
                    out_ranges.push(cur_start..=current_end);
                    current_start = Some(start);
                    current_end = end;
                }
            } else {
                current_start = Some(start);
                current_end = end;
            }
        }

        if let Some(cur_start) = current_start {
            out_ranges.push(cur_start..=current_end);
        }

        let out = Self::rangeset_from_ranges(out_ranges);
        out
    }

    fn ensure_right_comp<'a>(
        rv: &RangeSet,
        full_inner: &RangeSet,
        right_comp: &'a mut Option<RangeSet>,
        right_ptr: &mut Option<*const RangeSetBlaze<usize>>,
        rhs_comp_cache: Option<&RhsCompCache>,
    ) -> &'a RangeSet {
        let ptr = Arc::as_ptr(&rv.inner);
        if right_ptr.map_or(true, |prev| prev != ptr) {
            if let Some(cache) = rhs_comp_cache {
                let key = ptr as usize;
                if let Some(comp) = cache.get(&key) {
                    *right_comp = Some(comp.clone());
                    *right_ptr = Some(ptr);
                    return right_comp.as_ref().expect("missing right complement");
                }
            }
            *right_comp = Some(full_inner - rv);
            *right_ptr = Some(ptr);
        }
        right_comp.as_ref().expect("missing right complement")
    }

    pub(crate) fn new(num_tsids: usize) -> Self {
        Self::from_map(RangeMapBlaze::new(), num_tsids)
    }

    pub(crate) fn num_tsids(&self) -> usize {
        normalize_num_tsids(self.num_tsids)
    }

    /// Check if this weight is disjoint with another (no shared positions).
    /// Uses a merge-scan over sorted range maps — O(|ranges_a| + |ranges_b|)
    /// without materializing the full intersection.
    pub(crate) fn is_disjoint(&self, other: &Self) -> bool {
        if self.map.is_empty() || other.map.is_empty() {
            return true;
        }

        let mut iter_a = self.map.range_values();
        let mut iter_b = other.map.range_values();
        let mut a = iter_a.next();
        let mut b = iter_b.next();

        while let (Some((ra, va)), Some((rb, vb))) = (&a, &b) {
            if ra.end() < rb.start() {
                // ra fully before rb
                a = iter_a.next();
            } else if rb.end() < ra.start() {
                // rb fully before ra
                b = iter_b.next();
            } else {
                // Key ranges overlap — check if value sets overlap
                if !va.is_disjoint(vb) {
                    return false;
                }
                // Advance whichever range ends first
                if ra.end() <= rb.end() {
                    a = iter_a.next();
                } else {
                    b = iter_b.next();
                }
            }
        }
        true
    }

    /// Check if self is a subset of other (every position in self is also in other).
    /// Uses a merge-scan over sorted range maps without materializing the difference.
    pub(crate) fn is_subset_of(&self, other: &Self) -> bool {
        if self.map.is_empty() {
            return true;
        }
        if other.map.is_empty() {
            return false;
        }

        let mut iter_b = other.map.range_values();
        let mut b = iter_b.next();

        for (ra, va) in self.map.range_values() {
            // For each key range in self, find covering range(s) in other
            let mut pos = *ra.start();

            while pos <= *ra.end() {
                // Advance b until it could cover pos
                while let Some((rb, _)) = &b {
                    if rb.end() >= &pos {
                        break;
                    }
                    b = iter_b.next();
                }

                match &b {
                    Some((rb, vb)) if rb.start() <= &pos => {
                        // b covers pos — check value containment
                        if !va.is_subset(vb) {
                            return false;
                        }
                        // Advance pos past the covered region
                        let covered_end = std::cmp::min(*ra.end(), *rb.end());
                        if covered_end == usize::MAX {
                            break;
                        }
                        pos = covered_end + 1;
                    }
                    _ => {
                        // No range in other covers pos — self has uncovered positions
                        return false;
                    }
                }
            }
        }
        true
    }

    fn to_token_map(&self) -> BTreeMap<usize, RangeSet> {
        let mut out: BTreeMap<usize, RangeSet> = BTreeMap::new();
        for (token_range, tsid_set) in self.map.range_values() {
            for token in *token_range.start()..=*token_range.end() {
                out.insert(token, tsid_set.clone());
            }
        }
        out
    }

    fn merge_maps<F>(
        left: &RangeMapBlaze<usize, RangeSet>,
        right: &RangeMapBlaze<usize, RangeSet>,
        combine: F,
    ) -> RangeMapBlaze<usize, RangeSet>
    where
        F: Fn(Option<&RangeSet>, Option<&RangeSet>) -> RangeSet,
    {
        let mut boundaries: Vec<usize> = Vec::new();
        for (range, _) in left.range_values() {
            boundaries.push(*range.start());
            if let Some(next) = range.end().checked_add(1) {
                boundaries.push(next);
            }
        }
        for (range, _) in right.range_values() {
            boundaries.push(*range.start());
            if let Some(next) = range.end().checked_add(1) {
                boundaries.push(next);
            }
        }

        boundaries.sort_unstable();
        boundaries.dedup();

        let mut out = RangeMapBlaze::new();
        if boundaries.is_empty() {
            return out;
        }

        let mut current_start: Option<usize> = None;
        let mut current_end: usize = 0;
        let mut current_value = RangeSet::zeros();

        for (idx, &start) in boundaries.iter().enumerate() {
            let end = if idx + 1 < boundaries.len() {
                boundaries[idx + 1].saturating_sub(1)
            } else {
                usize::MAX
            };
            if start > end {
                continue;
            }

            let combined = combine(left.get(start), right.get(start));
            if combined.is_empty() {
                if let Some(range_start) = current_start.take() {
                    out.ranges_insert(range_start..=current_end, current_value.clone());
                }
                continue;
            }

            if let Some(range_start) = current_start {
                if current_value == combined && current_end.saturating_add(1) == start {
                    current_end = end;
                    continue;
                }
                out.ranges_insert(range_start..=current_end, current_value.clone());
            }

            current_start = Some(start);
            current_end = end;
            current_value = combined;
        }

        if let Some(range_start) = current_start {
            out.ranges_insert(range_start..=current_end, current_value);
        }

        out
    }

    fn intersect_asymmetric(
        small: &RangeMapBlaze<usize, RangeSet>,
        large: &RangeMapBlaze<usize, RangeSet>,
    ) -> RangeMapBlaze<usize, RangeSet> {
        let mut out = RangeMapBlaze::new();
        let mut large_iter = large.range_values();
        let mut large_current = large_iter.next();

        let mut current_start: Option<usize> = None;
        let mut current_end: usize = 0;
        let mut current_value = RangeSet::zeros();

        for (s_range, s_val) in small.range_values() {
            let s_start = *s_range.start();
            let s_end = *s_range.end();

            loop {
                let advance = match large_current.as_ref() {
                    Some((l_range, _)) if *l_range.end() < s_start => true,
                    _ => false,
                };
                if advance {
                    large_current = large_iter.next();
                } else {
                    break;
                }
            }

            let mut l_opt = large_current.take();
            while let Some((l_range, l_val)) = l_opt {
                if *l_range.start() > s_end {
                    large_current = Some((l_range, l_val));
                    break;
                }

                let overlap_start = s_start.max(*l_range.start());
                let overlap_end = s_end.min(*l_range.end());
                if overlap_start <= overlap_end {
                    let combined = s_val & l_val;
                    if combined.is_empty() {
                        if let Some(range_start) = current_start.take() {
                            out.ranges_insert(range_start..=current_end, current_value.clone());
                        }
                    } else if let Some(range_start) = current_start {
                        let is_same = Arc::ptr_eq(&current_value.inner, &combined.inner)
                            || current_value == combined;
                        if is_same && current_end.saturating_add(1) == overlap_start {
                            current_end = overlap_end;
                        } else {
                            out.ranges_insert(range_start..=current_end, current_value.clone());
                            current_start = Some(overlap_start);
                            current_end = overlap_end;
                            current_value = combined;
                        }
                    } else {
                        current_start = Some(overlap_start);
                        current_end = overlap_end;
                        current_value = combined;
                    }
                }

                if *l_range.end() <= s_end {
                    l_opt = large_iter.next();
                } else {
                    large_current = Some((l_range, l_val));
                    break;
                }
            }
        }

        if let Some(range_start) = current_start {
            out.ranges_insert(range_start..=current_end, current_value);
        }

        out
    }

    fn union_asymmetric(
        small: &RangeMapBlaze<usize, RangeSet>,
        large: &RangeMapBlaze<usize, RangeSet>,
    ) -> RangeMapBlaze<usize, RangeSet> {
        let mut result = large.clone();
        let mut large_iter = large.range_values();
        let mut large_current = large_iter.next();

        for (s_range, s_val) in small.range_values() {
            let s_start = *s_range.start();
            let s_end = *s_range.end();
            let mut cursor = s_start;

            loop {
                let advance = match large_current.as_ref() {
                    Some((l_range, _)) if *l_range.end() < cursor => true,
                    _ => false,
                };
                if advance {
                    large_current = large_iter.next();
                } else {
                    break;
                }
            }

            let mut l_opt = large_current.take();
            let mut keep_current: Option<(std::ops::RangeInclusive<usize>, &RangeSet)> = None;
            while let Some((l_range, l_val)) = l_opt {
                if *l_range.start() > s_end {
                    keep_current = Some((l_range, l_val));
                    break;
                }

                let overlap_start = cursor.max(*l_range.start());
                let overlap_end = s_end.min(*l_range.end());

                if cursor < overlap_start {
                    result.ranges_insert(cursor..=overlap_start.saturating_sub(1), s_val.clone());
                }

                if overlap_start <= overlap_end {
                    let combined = s_val | l_val;
                    result.ranges_insert(overlap_start..=overlap_end, combined);
                    cursor = overlap_end.saturating_add(1);
                }

                if cursor > s_end {
                    if *l_range.end() > s_end {
                        keep_current = Some((l_range, l_val));
                    }
                    break;
                }

                if *l_range.end() <= s_end {
                    l_opt = large_iter.next();
                } else {
                    keep_current = Some((l_range, l_val));
                    break;
                }
            }
            large_current = keep_current;
            if large_current.is_none() {
                large_current = large_iter.next();
            }

            if cursor <= s_end {
                result.ranges_insert(cursor..=s_end, s_val.clone());
            }
        }
        result
    }

    fn compress_outer_map(map: BTreeMap<usize, RangeSet>) -> RangeMapBlaze<usize, RangeSet> {
        let mut iter = map.into_iter();
        let (mut start, mut current) = iter.next().unwrap();
        let mut prev = start;
        let mut out = RangeMapBlaze::new();

        for (token, tsid_set) in iter {
            if token == prev.saturating_add(1) && tsid_set == current {
                prev = token;
                continue;
            }
            if !current.is_empty() {
                out.ranges_insert(start..=prev, current.clone());
            }
            start = token;
            prev = token;
            current = tsid_set;
        }

        if !current.is_empty() {
            out.ranges_insert(start..=prev, current);
        }

        out
    }

    fn invert_outer_map(map: &RangeMapBlaze<usize, RangeSet>) -> RangeMapBlaze<usize, RangeSet> {
        let mut tsid_map: BTreeMap<usize, RangeSet> = BTreeMap::new();
        for (token_range, tsid_set) in map.range_values() {
            if tsid_set.is_empty() {
                continue;
            }
            let token_rs = Self::rangeset_from_ranges([token_range.clone()]);
            for tsid_range in tsid_set.ranges() {
                for tsid in *tsid_range.start()..=*tsid_range.end() {
                    let entry = tsid_map.entry(tsid).or_insert_with(RangeSet::zeros);
                    *entry |= &token_rs;
                }
            }
        }

        if tsid_map.is_empty() {
            return RangeMapBlaze::new();
        }

        Self::compress_outer_map(tsid_map)
    }

    fn from_token_map(map: BTreeMap<usize, RangeSet>, num_tsids: usize) -> Self {
        let num_tsids = normalize_num_tsids(num_tsids);
        if map.is_empty() {
            return Self::new(num_tsids);
        }

        let out = Self::compress_outer_map(map);
        if Self::tsid_outer_enabled() {
            let inverted = Self::invert_outer_map(&out);
            return Self::from_map(inverted, num_tsids);
        }

        Self::from_map(out, num_tsids)
    }

    pub(crate) fn from_uniform_tsid_set(
        token_start: usize,
        token_end: usize,
        tsid_set: RangeSet,
        num_tsids: usize,
    ) -> Self {
        let num_tsids = normalize_num_tsids(num_tsids);
        if tsid_set.is_empty() || token_start > token_end {
            return Self::new(num_tsids);
        }
        if Self::tsid_outer_enabled() {
            let token_rs = Self::rangeset_from_ranges([token_start..=token_end]);
            let mut map = RangeMapBlaze::new();
            for tsid_range in tsid_set.ranges() {
                map.ranges_insert(*tsid_range.start()..=*tsid_range.end(), token_rs.clone());
            }
            return Self::from_map(map, num_tsids);
        }
        let mut map = RangeMapBlaze::new();
        map.ranges_insert(token_start..=token_end, tsid_set);
        Self::from_map(map, num_tsids)
    }

    pub(crate) fn from_rsb_with_num_tsids(rsb: &RangeSetBlaze<usize>, num_tsids: usize) -> Self {
        let num_tsids = normalize_num_tsids(num_tsids);
        if rsb.is_empty() {
            return Self::new(num_tsids);
        }

        let mut ranges = rsb.ranges();
        if let Some(range) = ranges.next() {
            if ranges.next().is_none() {
                let start = *range.start();
                let end = *range.end();
                if start <= end
                    && start % num_tsids == 0
                    && end % num_tsids == num_tsids.saturating_sub(1)
                {
                    let token_start = start / num_tsids;
                    let token_end = end / num_tsids;
                    let full_tsids = Self::full_tsids(num_tsids);
                    return Self::from_uniform_tsid_set(token_start, token_end, full_tsids, num_tsids);
                }
            }
        }
        let mut token_map: BTreeMap<usize, RangeSet> = BTreeMap::new();

        for range in rsb.ranges() {
            let start = *range.start();
            let end = *range.end();
            let start_token = start / num_tsids;
            let end_token = end / num_tsids;
            let start_tsid = start % num_tsids;
            let end_tsid = end % num_tsids;

            if start_token == end_token {
                let entry = token_map.entry(start_token).or_insert_with(RangeSet::zeros);
                *entry |= &Self::rangeset_from_ranges([start_tsid..=end_tsid]);
                continue;
            }

            // First token partial
            {
                let entry = token_map.entry(start_token).or_insert_with(RangeSet::zeros);
                *entry |= &Self::rangeset_from_ranges([start_tsid..=num_tsids - 1]);
            }

            // Middle full tokens
            if start_token + 1 <= end_token.saturating_sub(1) {
                let full = Self::rangeset_from_ranges([0..=num_tsids - 1]);
                for token in start_token + 1..=end_token - 1 {
                    let entry = token_map.entry(token).or_insert_with(RangeSet::zeros);
                    *entry |= &full;
                }
            }

            // Last token partial
            {
                let entry = token_map.entry(end_token).or_insert_with(RangeSet::zeros);
                *entry |= &Self::rangeset_from_ranges([0..=end_tsid]);
            }
        }

        Self::from_token_map(token_map, num_tsids)
    }

    fn union_all_non_negated(weights: &[&RangeMapWeight]) -> Self {
        if weights.is_empty() {
            return Self::new(current_num_tsids());
        }

        let num_tsids = weights[0].num_tsids();
        if weights.len() == 1 {
            return weights[0].clone();
        }
        if weights.len() == 2 {
            return weights[0].union_non_negated(weights[1]);
        }

        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut weight_ranges: Vec<Vec<(usize, usize, RangeSet)>> = Vec::with_capacity(weights.len());
        for weight in weights {
            assert_eq!(
                weight.num_tsids(),
                num_tsids,
                "RangeMapWeight num_tsids mismatch"
            );
            let mut ranges = Vec::with_capacity(Self::map_range_count(&weight.map));
            for (range, tsid_set) in weight.map.range_values() {
                ranges.push((*range.start(), *range.end(), tsid_set.clone()));
            }
            weight_ranges.push(ranges);
        }

        let mut heap: BinaryHeap<Reverse<(usize, u8, usize)>> = BinaryHeap::new();
        let mut indices: Vec<usize> = vec![0; weights.len()];
        let mut current_values: Vec<Option<RangeSet>> = vec![None; weights.len()];
        let mut active_indices: Vec<usize> = Vec::new();
        let mut active_positions: Vec<Option<usize>> = vec![None; weights.len()];

        for (idx, ranges) in weight_ranges.iter().enumerate() {
            if let Some((start, _, _)) = ranges.get(0) {
                heap.push(Reverse((*start, 1u8, idx)));
            }
        }

        let mut out = RangeMapBlaze::new();
        if heap.is_empty() {
            return Self::new(num_tsids);
        }

        let mut current_start: Option<usize> = None;
        let mut current_end: usize = 0;
        let mut current_value = RangeSet::zeros();

        while let Some(Reverse((boundary, kind, weight_idx))) = heap.pop() {
            let mut end_events: Vec<usize> = Vec::new();
            let mut start_events: Vec<usize> = Vec::new();

            if kind == 0 {
                end_events.push(weight_idx);
            } else {
                start_events.push(weight_idx);
            }

            while let Some(Reverse((next_boundary, next_kind, next_idx))) = heap.peek().cloned() {
                if next_boundary != boundary {
                    break;
                }
                heap.pop();
                if next_kind == 0 {
                    end_events.push(next_idx);
                } else {
                    start_events.push(next_idx);
                }
            }

            for w_idx in end_events {
                if let Some(pos) = active_positions[w_idx].take() {
                    let last_idx = active_indices.pop().expect("active_indices empty");
                    if pos < active_indices.len() {
                        active_indices[pos] = last_idx;
                        active_positions[last_idx] = Some(pos);
                    }
                }
                current_values[w_idx] = None;
                indices[w_idx] += 1;
                if let Some((next_start, _, _)) = weight_ranges[w_idx].get(indices[w_idx]) {
                    if *next_start == boundary {
                        start_events.push(w_idx);
                    } else {
                        heap.push(Reverse((*next_start, 1u8, w_idx)));
                    }
                }
            }

            for w_idx in start_events {
                let (start, end, value) = weight_ranges[w_idx][indices[w_idx]].clone();
                debug_assert_eq!(start, boundary);
                current_values[w_idx] = Some(value);
                if active_positions[w_idx].is_none() {
                    active_positions[w_idx] = Some(active_indices.len());
                    active_indices.push(w_idx);
                }
                if let Some(next) = end.checked_add(1) {
                    heap.push(Reverse((next, 0u8, w_idx)));
                }
            }

            let combined = {
                let mut active_values: Vec<&RangeSet> = Vec::with_capacity(active_indices.len());
                for &idx in &active_indices {
                    if let Some(val) = &current_values[idx] {
                        active_values.push(val);
                    }
                }
                if active_values.is_empty() {
                    RangeSet::zeros()
                } else {
                    RangeSet::bulk_union(&active_values)
                }
            };

            let next_boundary = heap.peek().map(|Reverse((b, _, _))| *b).unwrap_or(usize::MAX);
            let end = if next_boundary == usize::MAX {
                usize::MAX
            } else {
                next_boundary.saturating_sub(1)
            };
            if boundary > end {
                continue;
            }

            if combined.is_empty() {
                if let Some(range_start) = current_start.take() {
                    out.ranges_insert(range_start..=current_end, current_value.clone());
                }
            } else if let Some(range_start) = current_start {
                if current_value == combined && current_end.saturating_add(1) == boundary {
                    current_end = end;
                } else {
                    out.ranges_insert(range_start..=current_end, current_value.clone());
                    current_start = Some(boundary);
                    current_end = end;
                    current_value = combined;
                }
            } else {
                current_start = Some(boundary);
                current_end = end;
                current_value = combined;
            }

            if next_boundary == usize::MAX {
                break;
            }
        }

        if let Some(range_start) = current_start {
            out.ranges_insert(range_start..=current_end, current_value);
        }

        Self::from_map(out, num_tsids)
    }

    pub(crate) fn bulk_union(weights: &[&RangeMapWeight]) -> Arc<RangeMapWeight> {
        if weights.is_empty() {
            return intern_rangemap(RangeMapWeight::new(current_num_tsids()));
        }

        let num_tsids = weights[0].num_tsids();
        let mut non_empty: Vec<&RangeMapWeight> = Vec::with_capacity(weights.len());

        for weight in weights {
            assert_eq!(
                weight.num_tsids(),
                num_tsids,
                "RangeMapWeight num_tsids mismatch"
            );
            if weight.map.is_empty() {
                continue;
            }
            non_empty.push(*weight);
        }

        if non_empty.is_empty() {
            return intern_rangemap(RangeMapWeight::new(num_tsids));
        }
        if non_empty.len() == 1 {
            return intern_rangemap(non_empty[0].clone());
        }

        let result = Self::union_all_non_negated(&non_empty);
        intern_rangemap(result)
    }

    pub(crate) fn union_all(weights: &[Arc<RangeMapWeight>]) -> Arc<RangeMapWeight> {
        if weights.is_empty() {
            return intern_rangemap(RangeMapWeight::new(current_num_tsids()));
        }

        let num_tsids = weights[0].num_tsids();
        let mut non_empty: Vec<&RangeMapWeight> = Vec::with_capacity(weights.len());
        let mut single_arc: Option<&Arc<RangeMapWeight>> = None;

        for weight in weights {
            assert_eq!(
                weight.num_tsids(),
                num_tsids,
                "RangeMapWeight num_tsids mismatch"
            );
            if weight.map.is_empty() {
                continue;
            }
            if single_arc.is_none() {
                single_arc = Some(weight);
            }
            non_empty.push(weight.as_ref());
        }

        if non_empty.is_empty() {
            return intern_rangemap(RangeMapWeight::new(num_tsids));
        }
        if non_empty.len() == 1 {
            return single_arc.expect("missing non-empty weight").clone();
        }

        let result = Self::union_all_non_negated(&non_empty);
        intern_rangemap(result)
    }

    fn union_non_negated(&self, other: &Self) -> Self {
        let profile_active = OP_CACHE_PROFILE_ENABLED.load(AtomicOrdering::Relaxed)
            && OP_CACHE_OR_MISS_ACTIVE.with(|flag| flag.get());
        let prep_start = if profile_active { Some(Instant::now()) } else { None };
        let left_ranges = Self::map_range_count(&self.map);
        let right_ranges = Self::map_range_count(&other.map);
        if profile_active {
            OP_CACHE_OR_MISS_LEFT_RANGES_TOTAL
                .fetch_add(left_ranges as u64, AtomicOrdering::Relaxed);
            OP_CACHE_OR_MISS_RIGHT_RANGES_TOTAL
                .fetch_add(right_ranges as u64, AtomicOrdering::Relaxed);
        }
        if left_ranges == 0 {
            if let Some(start) = prep_start {
                OP_CACHE_OR_MISS_PREP_NS
                    .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
            }
            return other.clone();
        }
        if right_ranges == 0 {
            if let Some(start) = prep_start {
                OP_CACHE_OR_MISS_PREP_NS
                    .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
            }
            return self.clone();
        }
        let is_same = std::ptr::eq(self, other) || self == other;
        if is_same {
            if let Some(start) = prep_start {
                OP_CACHE_OR_MISS_PREP_NS
                    .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
            }
            return self.clone();
        }
        if left_ranges == 1 || right_ranges == 1 {
            let max_token = crate::datastructures::get_max_llm_token();
            if Self::tsid_outer_enabled() {
                let num_tsids = self.num_tsids();
                if num_tsids > 0 {
                    let full_tokens = Self::rangeset_from_ranges([0..=max_token]);
                    if left_ranges == 1 {
                        if let Some((range, token_set)) = self.map.range_values().next() {
                            if *range.start() == 0
                                && *range.end() == num_tsids.saturating_sub(1)
                                && *token_set == full_tokens
                            {
                                if let Some(start) = prep_start {
                                    OP_CACHE_OR_MISS_PREP_NS
                                        .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                                }
                                return self.clone();
                            }
                        }
                    }
                    if right_ranges == 1 {
                        if let Some((range, token_set)) = other.map.range_values().next() {
                            if *range.start() == 0
                                && *range.end() == num_tsids.saturating_sub(1)
                                && *token_set == full_tokens
                            {
                                if let Some(start) = prep_start {
                                    OP_CACHE_OR_MISS_PREP_NS
                                        .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                                }
                                return other.clone();
                            }
                        }
                    }
                }
            } else {
                if left_ranges == 1 {
                    if let Some((range, tsid_set)) = self.map.range_values().next() {
                        if *range.start() == 0
                            && *range.end() == max_token
                            && *tsid_set == Self::full_tsids(self.num_tsids)
                        {
                            if let Some(start) = prep_start {
                                OP_CACHE_OR_MISS_PREP_NS
                                    .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                            }
                            return self.clone();
                        }
                    }
                }
                if right_ranges == 1 {
                    if let Some((range, tsid_set)) = other.map.range_values().next() {
                        if *range.start() == 0
                            && *range.end() == max_token
                            && *tsid_set == Self::full_tsids(self.num_tsids)
                        {
                            if let Some(start) = prep_start {
                                OP_CACHE_OR_MISS_PREP_NS
                                    .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                            }
                            return other.clone();
                        }
                    }
                }
            }
        }

        let (smaller, larger, small_ranges, large_ranges) = if left_ranges <= right_ranges {
            (self, other, left_ranges, right_ranges)
        } else {
            (other, self, right_ranges, left_ranges)
        };

        if let Some(start) = prep_start {
            OP_CACHE_OR_MISS_PREP_NS
                .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        }

        let map = if small_ranges.saturating_mul(10) < large_ranges {
            if profile_active {
                let start = Instant::now();
                let map = Self::union_asymmetric(&smaller.map, &larger.map);
                OP_CACHE_OR_MISS_UNION_ASYM_NS
                    .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                OP_CACHE_OR_MISS_ASYM_COUNT.fetch_add(1, AtomicOrdering::Relaxed);
                map
            } else {
                Self::union_asymmetric(&smaller.map, &larger.map)
            }
        } else {
            if profile_active {
                let start = Instant::now();
                let map = Self::merge_maps(&self.map, &other.map, |left, right| {
                    OP_CACHE_OR_MISS_SEGMENTS_TOTAL.fetch_add(1, AtomicOrdering::Relaxed);
                    match (left, right) {
                        (Some(a), Some(b)) => {
                            OP_CACHE_OR_MISS_SEGMENTS_BOTH.fetch_add(1, AtomicOrdering::Relaxed);
                            OP_CACHE_OR_MISS_RANGESET_LEFT_RANGES_TOTAL
                                .fetch_add(a.ranges_len() as u64, AtomicOrdering::Relaxed);
                            OP_CACHE_OR_MISS_RANGESET_RIGHT_RANGES_TOTAL
                                .fetch_add(b.ranges_len() as u64, AtomicOrdering::Relaxed);
                            let start = Instant::now();
                            let combined = a | b;
                            OP_CACHE_OR_MISS_RANGESET_UNION_NS
                                .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                            OP_CACHE_OR_MISS_RANGESET_UNION_COUNT
                                .fetch_add(1, AtomicOrdering::Relaxed);
                            combined
                        }
                        (Some(a), None) => {
                            OP_CACHE_OR_MISS_SEGMENTS_LEFT_ONLY.fetch_add(1, AtomicOrdering::Relaxed);
                            a.clone()
                        }
                        (None, Some(b)) => {
                            OP_CACHE_OR_MISS_SEGMENTS_RIGHT_ONLY.fetch_add(1, AtomicOrdering::Relaxed);
                            b.clone()
                        }
                        (None, None) => {
                            OP_CACHE_OR_MISS_SEGMENTS_NONE.fetch_add(1, AtomicOrdering::Relaxed);
                            RangeSet::zeros()
                        }
                    }
                });
                OP_CACHE_OR_MISS_MERGE_NS
                    .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                OP_CACHE_OR_MISS_MERGE_COUNT.fetch_add(1, AtomicOrdering::Relaxed);
                map
            } else {
                Self::merge_maps(&self.map, &other.map, |left, right| match (left, right) {
                    (Some(a), Some(b)) => a | b,
                    (Some(a), None) => a.clone(),
                    (None, Some(b)) => b.clone(),
                    (None, None) => RangeSet::zeros(),
                })
            }
        };
        let result = Self::from_map(map, self.num_tsids());
        result
    }

    #[time_it("RangeMapWeight::union_fast")]
    pub(crate) fn union_fast(&self, other: &Self) -> Self {
        self.union_non_negated(other)
    }

    fn intersect_non_negated(&self, other: &Self) -> Self {
        let left_ranges = Self::map_range_count(&self.map);
        let right_ranges = Self::map_range_count(&other.map);
        if left_ranges == 0 || right_ranges == 0 {
            return Self::new(self.num_tsids());
        }

        let (smaller, larger, small_ranges, large_ranges) = if left_ranges <= right_ranges {
            (self, other, left_ranges, right_ranges)
        } else {
            (other, self, right_ranges, left_ranges)
        };

        let map = if small_ranges.saturating_mul(10) < large_ranges {
            Self::intersect_asymmetric(&smaller.map, &larger.map)
        } else {
            Self::merge_maps(&self.map, &other.map, |left, right| match (left, right) {
                (Some(a), Some(b)) => a & b,
                _ => RangeSet::zeros(),
            })
        };

        Self::from_map(map, self.num_tsids())
    }

    fn difference_non_negated(&self, other: &Self) -> Self {
        let map = Self::merge_maps(&self.map, &other.map, |left, right| match (left, right) {
            (Some(a), Some(b)) => a - b,
            (Some(a), None) => a.clone(),
            _ => RangeSet::zeros(),
        });
        Self::from_map(map, self.num_tsids())
    }

    pub(crate) fn divide(&self, other: &Self) -> Self {
        self.divide_with_rhs_comp_cache(other, None)
    }

    fn divide_with_rhs_comp_cache(
        &self,
        other: &Self,
        rhs_comp_cache: Option<&RhsCompCache>,
    ) -> Self {
        assert_eq!(self.num_tsids(), other.num_tsids(), "RangeMapWeight num_tsids mismatch");
        let num_tsids = self.num_tsids();
        let full_tsids = Self::full_inner_set(num_tsids);
        let max_tsid = Self::max_inner_value(num_tsids);

        let mut left_iter = self.map.range_values();
        let mut right_iter = other.map.range_values();
        let mut left = left_iter.next();
        let mut right = right_iter.next();

        if left.is_none() && right.is_none() {
            let result = Self::from_map(RangeMapBlaze::new(), num_tsids);
            return result;
        }

        let mut pos = match (left.as_ref(), right.as_ref()) {
            (Some((l_range, _)), Some((r_range, _))) => {
                (*l_range.start()).min(*r_range.start())
            }
            (Some((l_range, _)), None) => *l_range.start(),
            (None, Some((r_range, _))) => *r_range.start(),
            (None, None) => 0,
        };

        let mut out = RangeMapBlaze::new();
        let mut current_start: Option<usize> = None;
        let mut current_end: usize = 0;
        let mut current_value = RangeSet::zeros();

        let mut right_comp: Option<RangeSet> = None;
        let mut right_ptr: Option<*const RangeSetBlaze<usize>> = None;

        loop {
            loop {
                let advance = match left.as_ref() {
                    Some((range, _)) if pos > *range.end() => true,
                    _ => false,
                };
                if advance {
                    left = left_iter.next();
                } else {
                    break;
                }
            }
            loop {
                let advance = match right.as_ref() {
                    Some((range, _)) if pos > *range.end() => true,
                    _ => false,
                };
                if advance {
                    right = right_iter.next();
                } else {
                    break;
                }
            }
            let (left_val, next_left_change) = match left.as_ref() {
                Some((range, val)) => {
                    if pos < *range.start() {
                        (None, Some(*range.start()))
                    } else {
                        (Some(*val), range.end().checked_add(1))
                    }
                }
                None => (None, None),
            };

            let (right_val, next_right_change) = match right.as_ref() {
                Some((range, val)) => {
                    if pos < *range.start() {
                        (None, Some(*range.start()))
                    } else {
                        (Some(*val), range.end().checked_add(1))
                    }
                }
                None => (None, None),
            };
            if right_val.is_none() {
                right_comp = None;
                right_ptr = None;
            }
            let combined = match (left_val, right_val) {
                (Some(a), Some(rv)) => {
                    let lhs_ranges = a.ranges_len();
                    let rhs_ranges = rv.ranges_len();
                    if lhs_ranges.saturating_mul(2) < rhs_ranges {
                        let out = Self::rangeset_union_with_complement_asymmetric(a, rv, max_tsid);
                        out
                    } else {
                        let comp = Self::ensure_right_comp(
                            rv,
                            &full_tsids,
                            &mut right_comp,
                            &mut right_ptr,
                            rhs_comp_cache,
                        );
                        let out = a | comp;
                        out
                    }
                }
                (Some(a), None) => a.clone(),
                (None, Some(rv)) => {
                    let comp = Self::ensure_right_comp(
                        rv,
                        &full_tsids,
                        &mut right_comp,
                        &mut right_ptr,
                        rhs_comp_cache,
                    );
                    comp.clone()
                }
                (None, None) => full_tsids.clone(),
            };

            let next_change = match (next_left_change, next_right_change) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };
            let end = match next_change {
                Some(next) => next.saturating_sub(1),
                None => usize::MAX,
            };

            if combined.is_empty() {
                if let Some(range_start) = current_start.take() {
                    out.ranges_insert(range_start..=current_end, current_value.clone());
                }
            } else if let Some(range_start) = current_start {
                let is_same = Arc::ptr_eq(&current_value.inner, &combined.inner)
                    || current_value == combined;
                if is_same && current_end.saturating_add(1) == pos {
                    current_end = end;
                } else {
                    out.ranges_insert(range_start..=current_end, current_value.clone());
                    current_start = Some(pos);
                    current_end = end;
                    current_value = combined;
                }
            } else {
                current_start = Some(pos);
                current_end = end;
                current_value = combined;
            }

            if end == usize::MAX {
                break;
            }
            pos = end.saturating_add(1);
        }

        if let Some(range_start) = current_start {
            out.ranges_insert(range_start..=current_end, current_value);
        }

        let result = Self::from_map(out, num_tsids);
        result
    }

    pub(crate) fn clip_to_max(&mut self, max: usize) {
        if self.map.is_empty() {
            return;
        }

        let num_tsids = self.num_tsids();
        if num_tsids == 0 {
            self.map = RangeMapBlaze::new();
            self.refresh_cached_hash();
            return;
        }
        if let Some(current_max) = self.max_item() {
            if current_max <= max {
                return;
            }
        }
        let max_token = max / num_tsids;
        let max_tsid = max % num_tsids;
        if Self::tsid_outer_enabled() {
            let token_clip_full = Self::rangeset_from_ranges([0..=max_token]);
            let token_clip_prefix = if max_token > 0 {
                Self::rangeset_from_ranges([0..=max_token.saturating_sub(1)])
            } else {
                RangeSet::zeros()
            };
            let mut new_map = RangeMapBlaze::new();

            for (tsid_range, token_set) in self.map.range_values() {
                let start = *tsid_range.start();
                let end = *tsid_range.end();

                if start > max_tsid {
                    let clipped = token_set & &token_clip_prefix;
                    if !clipped.is_empty() {
                        new_map.ranges_insert(start..=end, clipped);
                    }
                    continue;
                }

                if end <= max_tsid {
                    let clipped = token_set & &token_clip_full;
                    if !clipped.is_empty() {
                        new_map.ranges_insert(start..=end, clipped);
                    }
                    continue;
                }

                let clipped_full = token_set & &token_clip_full;
                if !clipped_full.is_empty() {
                    new_map.ranges_insert(start..=max_tsid, clipped_full.clone());
                }

                let right_start = max_tsid.saturating_add(1);
                if right_start <= end {
                    let clipped_prefix = token_set & &token_clip_prefix;
                    if !clipped_prefix.is_empty() {
                        new_map.ranges_insert(right_start..=end, clipped_prefix);
                    }
                }
            }

            self.map = new_map;
            self.refresh_cached_hash();
            return;
        }

        let tsid_clip = Self::rangeset_from_ranges([0..=max_tsid]);

        let mut new_map = RangeMapBlaze::new();
        for (token_range, tsid_set) in self.map.range_values() {
            let start = *token_range.start();
            if start > max_token {
                break;
            }
            let end = (*token_range.end()).min(max_token);
            if start > end {
                continue;
            }

            if end < max_token {
                if !tsid_set.is_empty() {
                    new_map.ranges_insert(start..=end, tsid_set.clone());
                }
                continue;
            }

            if start < max_token {
                if !tsid_set.is_empty() {
                    new_map.ranges_insert(start..=max_token.saturating_sub(1), tsid_set.clone());
                }
            }

            let clipped = tsid_set & &tsid_clip;
            if !clipped.is_empty() {
                new_map.ranges_insert(max_token..=max_token, clipped);
            }
        }

        self.map = new_map;
        self.refresh_cached_hash();
    }

    pub(crate) fn expand_to_rsb(&self) -> RangeSetBlaze<usize> {
        if self.map.is_empty() {
            return RangeSetBlaze::new();
        }

        let num_tsids = self.num_tsids();
        let mut ranges: Vec<std::ops::RangeInclusive<usize>> = Vec::new();
        if Self::tsid_outer_enabled() {
            for (tsid_range, token_set) in self.map.range_values() {
                let tsid_start = *tsid_range.start();
                let tsid_end = *tsid_range.end();
                for token_range in token_set.ranges() {
                    for token in *token_range.start()..=*token_range.end() {
                        let base = token.saturating_mul(num_tsids);
                        ranges.push(base.saturating_add(tsid_start)..=base.saturating_add(tsid_end));
                    }
                }
            }
        } else {
            for (token_range, tsid_set) in self.map.range_values() {
                for token in *token_range.start()..=*token_range.end() {
                    for tsid_range in tsid_set.ranges() {
                        let base = token.saturating_mul(num_tsids);
                        let tsid_start = *tsid_range.start();
                        let tsid_end = *tsid_range.end();
                        ranges.push(base.saturating_add(tsid_start)..=base.saturating_add(tsid_end));
                    }
                }
            }
        }
        RangeSetBlaze::from_iter(ranges)
    }

    pub(crate) fn expand_to_rsb_bounded(&self, max: usize) -> RangeSetBlaze<usize> {
        if self.map.is_empty() {
            return RangeSetBlaze::new();
        }

        let num_tsids = self.num_tsids();
        let max_token = max / num_tsids;
        let max_tsid = max % num_tsids;
        let mut ranges: Vec<std::ops::RangeInclusive<usize>> = Vec::new();

        if Self::tsid_outer_enabled() {
            for (tsid_range, token_set) in self.map.range_values() {
                let tsid_start = *tsid_range.start();
                let tsid_end = *tsid_range.end();

                let left_end = tsid_end.min(max_tsid);
                if tsid_start <= left_end {
                    for token_range in token_set.ranges() {
                        let token_start = *token_range.start();
                        let mut token_end = *token_range.end();
                        if token_start > max_token {
                            continue;
                        }
                        token_end = token_end.min(max_token);
                        for token in token_start..=token_end {
                            let base = token.saturating_mul(num_tsids);
                            ranges.push(base.saturating_add(tsid_start)..=base.saturating_add(left_end));
                        }
                    }
                }

                if tsid_end > max_tsid && max_token > 0 {
                    let right_start = tsid_start.max(max_tsid.saturating_add(1));
                    if right_start <= tsid_end {
                        let token_limit = max_token.saturating_sub(1);
                        for token_range in token_set.ranges() {
                            let token_start = *token_range.start();
                            let mut token_end = *token_range.end();
                            if token_start > token_limit {
                                continue;
                            }
                            token_end = token_end.min(token_limit);
                            for token in token_start..=token_end {
                                let base = token.saturating_mul(num_tsids);
                                ranges.push(base.saturating_add(right_start)..=base.saturating_add(tsid_end));
                            }
                        }
                    }
                }
            }
        } else {
            for (token_range, tsid_set) in self.map.range_values() {
                let token_start = *token_range.start();
                let token_end = (*token_range.end()).min(max_token);
                if token_start > token_end {
                    continue;
                }
                for token in token_start..=token_end {
                    let base = token.saturating_mul(num_tsids);
                    for tsid_range in tsid_set.ranges() {
                        let tsid_start = *tsid_range.start();
                        let mut tsid_end = *tsid_range.end();
                        if token == max_token {
                            if tsid_start > max_tsid {
                                continue;
                            }
                            tsid_end = tsid_end.min(max_tsid);
                        }
                        ranges.push(base.saturating_add(tsid_start)..=base.saturating_add(tsid_end));
                    }
                }
            }
        }

        RangeSetBlaze::from_iter(ranges)
    }
}

impl Hash for RangeMapWeight {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.cached_hash.hash(state);
    }
}

impl WeightBackend for RangeMapWeight {
    fn empty() -> Self {
        Self::new(current_num_tsids())
    }

    fn all(max_position: usize) -> Self {
        let num_tsids = current_num_tsids();
        let num_tsids = normalize_num_tsids(num_tsids);
        if num_tsids == 0 {
            return Self::new(num_tsids);
        }

        let max_token = max_position / num_tsids;
        let max_tsid = max_position % num_tsids;
        if Self::tsid_outer_enabled() {
            let mut map = RangeMapBlaze::new();
            let full_tokens = Self::rangeset_from_ranges([0..=max_token]);

            if max_tsid == num_tsids.saturating_sub(1) {
                map.ranges_insert(0..=max_tsid, full_tokens);
                return Self::from_map(map, num_tsids);
            }

            map.ranges_insert(0..=max_tsid, full_tokens.clone());
            if max_token > 0 {
                let prefix_tokens = Self::rangeset_from_ranges([0..=max_token.saturating_sub(1)]);
                if !prefix_tokens.is_empty() && max_tsid + 1 <= num_tsids.saturating_sub(1) {
                    map.ranges_insert(max_tsid + 1..=num_tsids.saturating_sub(1), prefix_tokens);
                }
            }
            return Self::from_map(map, num_tsids);
        }
        let full_tsids = Self::rangeset_from_ranges([0..=num_tsids - 1]);
        let mut map = RangeMapBlaze::new();

        if max_token == 0 {
            let tsids = Self::rangeset_from_ranges([0..=max_tsid]);
            if !tsids.is_empty() {
                map.ranges_insert(0..=0, tsids);
            }
            return Self::from_map(map, num_tsids);
        }

        if max_tsid == num_tsids - 1 {
            map.ranges_insert(0..=max_token, full_tsids);
        } else {
            map.ranges_insert(0..=max_token - 1, full_tsids.clone());
            let last_tsids = Self::rangeset_from_ranges([0..=max_tsid]);
            if !last_tsids.is_empty() {
                map.ranges_insert(max_token..=max_token, last_tsids);
            }
        }

        Self::from_map(map, num_tsids)
    }

    fn from_position(pos: usize) -> Self {
        let num_tsids = current_num_tsids();
        let num_tsids = normalize_num_tsids(num_tsids);
        if num_tsids == 0 {
            return Self::new(num_tsids);
        }
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        let mut map = RangeMapBlaze::new();
        if Self::tsid_outer_enabled() {
            let token_set = Self::rangeset_from_ranges([token..=token]);
            map.ranges_insert(tsid..=tsid, token_set);
        } else {
            let tsid_set = Self::rangeset_from_ranges([tsid..=tsid]);
            map.ranges_insert(token..=token, tsid_set);
        }
        Self::from_map(map, num_tsids)
    }

    fn from_ranges<I: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(ranges: I) -> Self {
        let rsb = RangeSetBlaze::from_iter(ranges);
        Self::from_rsb_with_num_tsids(&rsb, current_num_tsids())
    }

    fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn len(&self) -> usize {
        let mut total: u128 = 0;
        for (token_range, tsid_set) in self.map.range_values() {
            let range_len = (*token_range.end()).saturating_sub(*token_range.start()).saturating_add(1) as u128;
            let tsid_len = tsid_set.len() as u128;
            total = total.saturating_add(range_len.saturating_mul(tsid_len));
        }
        if total > usize::MAX as u128 { usize::MAX } else { total as usize }
    }

    fn contains(&self, pos: usize) -> bool {
        let num_tsids = self.num_tsids();
        if num_tsids == 0 {
            return false;
        }
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        if Self::tsid_outer_enabled() {
            self.map.get(tsid).map_or(false, |tokens| tokens.contains(token))
        } else {
            self.map.get(token).map_or(false, |tsids| tsids.contains(tsid))
        }
    }

    fn ranges_len(&self) -> usize {
        let map_ranges = self.map.range_values().len();
        let tsid_ranges: usize = self
            .map
            .range_values()
            .map(|(_, tsid_set)| tsid_set.ranges_len())
            .sum();
        map_ranges.saturating_add(tsid_ranges)
    }

    fn num_ranges(&self) -> usize {
        self.ranges_len()
    }

    fn insert(&mut self, pos: usize) {
        let num_tsids = self.num_tsids();
        if num_tsids == 0 {
            return;
        }
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        if Self::tsid_outer_enabled() {
            let mut new_set = Self::rangeset_from_ranges([token..=token]);
            if let Some(existing) = self.map.get(tsid) {
                new_set |= existing;
            }
            self.map.ranges_insert(tsid..=tsid, new_set);
        } else {
            let mut new_set = Self::rangeset_from_ranges([tsid..=tsid]);
            if let Some(existing) = self.map.get(token) {
                new_set |= existing;
            }
            self.map.ranges_insert(token..=token, new_set);
        }
        self.refresh_cached_hash();
    }

    fn intersect(&self, other: &Self) -> Self {
        assert_eq!(self.num_tsids(), other.num_tsids(), "RangeMapWeight num_tsids mismatch");
        self.intersect_non_negated(other)
    }

    fn intersect_assign(&mut self, other: &Self) {
        *self = self.intersect(other);
    }

    fn union(&self, other: &Self) -> Self {
        assert_eq!(self.num_tsids(), other.num_tsids(), "RangeMapWeight num_tsids mismatch");
        self.union_non_negated(other)
    }

    fn union_assign(&mut self, other: &Self) {
        *self = self.union(other);
    }

    fn difference(&self, other: &Self) -> Self {
        assert_eq!(self.num_tsids(), other.num_tsids(), "RangeMapWeight num_tsids mismatch");
        self.difference_non_negated(other)
    }

    fn complement(&self, max_position: usize) -> Self {
        let all = Self::all(max_position);
        all.difference(self)
    }

    // Note: divide uses default trait implementation (self.union(&other.complement(max_position)))

    fn min_item(&self) -> Option<usize> {
        let num_tsids = self.num_tsids();
        let mut min_pos: Option<usize> = None;
        if Self::tsid_outer_enabled() {
            for (tsid_range, token_set) in self.map.range_values() {
                let tsid = *tsid_range.start();
                let token = token_set.ranges().next().map(|r| *r.start());
                if let Some(token) = token {
                    let pos = token.saturating_mul(num_tsids).saturating_add(tsid);
                    min_pos = Some(min_pos.map_or(pos, |m| m.min(pos)));
                }
            }
        } else {
            for (token_range, tsid_set) in self.map.range_values() {
                let token = *token_range.start();
                let tsid = tsid_set.ranges().next().map(|r| *r.start());
                if let Some(tsid) = tsid {
                    let pos = token.saturating_mul(num_tsids).saturating_add(tsid);
                    min_pos = Some(min_pos.map_or(pos, |m| m.min(pos)));
                }
            }
        }
        min_pos
    }

    fn max_item(&self) -> Option<usize> {
        let num_tsids = self.num_tsids();
        let mut max_pos: Option<usize> = None;
        if Self::tsid_outer_enabled() {
            for (tsid_range, token_set) in self.map.range_values() {
                let tsid = *tsid_range.end();
                let token = token_set.ranges().last().map(|r| *r.end());
                if let Some(token) = token {
                    let pos = token.saturating_mul(num_tsids).saturating_add(tsid);
                    max_pos = Some(max_pos.map_or(pos, |m| m.max(pos)));
                }
            }
        } else {
            for (token_range, tsid_set) in self.map.range_values() {
                let token = *token_range.end();
                let tsid = tsid_set.ranges().last().map(|r| *r.end());
                if let Some(tsid) = tsid {
                    let pos = token.saturating_mul(num_tsids).saturating_add(tsid);
                    max_pos = Some(max_pos.map_or(pos, |m| m.max(pos)));
                }
            }
        }
        max_pos
    }
}

impl WeightBackend for Arc<RangeMapWeight> {
    fn empty() -> Self {
        intern_rangemap(RangeMapWeight::new(current_num_tsids()))
    }

    fn all(max_position: usize) -> Self {
        intern_rangemap(<RangeMapWeight as WeightBackend>::all(max_position))
    }

    fn from_position(pos: usize) -> Self {
        intern_rangemap(<RangeMapWeight as WeightBackend>::from_position(pos))
    }

    fn from_ranges<I: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(ranges: I) -> Self {
        let rsb = RangeSetBlaze::from_iter(ranges);
        intern_rangemap(RangeMapWeight::from_rsb_with_num_tsids(&rsb, current_num_tsids()))
    }

    fn is_empty(&self) -> bool {
        WeightBackend::is_empty(self.as_ref())
    }

    fn len(&self) -> usize {
        WeightBackend::len(self.as_ref())
    }

    fn contains(&self, pos: usize) -> bool {
        WeightBackend::contains(self.as_ref(), pos)
    }

    fn ranges_len(&self) -> usize {
        WeightBackend::ranges_len(self.as_ref())
    }

    fn num_ranges(&self) -> usize {
        WeightBackend::num_ranges(self.as_ref())
    }

    fn insert(&mut self, pos: usize) {
        let mut new = (**self).clone();
        new.insert(pos);
        *self = intern_rangemap(new);
    }

    fn intersect(&self, other: &Self) -> Self {
        if Arc::ptr_eq(self, other) {
            return self.clone();
        }
        if let Some(hit) = get_op_cache(cache::BinOp::And, self, other) {
            return hit;
        }
        let out = WeightBackend::intersect(self.as_ref(), other.as_ref());
        let out = intern_rangemap(out);
        put_op_cache(cache::BinOp::And, self.clone(), other.clone(), out.clone());
        out
    }

    fn intersect_assign(&mut self, other: &Self) {
        *self = self.intersect(other);
    }

    fn union(&self, other: &Self) -> Self {
        if Arc::ptr_eq(self, other) {
            return self.clone();
        }
        if let Some(hit) = get_op_cache(cache::BinOp::Or, self, other) {
            return hit;
        }
        let profile_enabled = OP_CACHE_PROFILE_ENABLED.load(AtomicOrdering::Relaxed);
        let miss_start = if profile_enabled { Some(Instant::now()) } else { None };
        let out = if profile_enabled {
            OP_CACHE_OR_MISS_ACTIVE.with(|flag| {
                let prev = flag.replace(true);
                let out = WeightBackend::union(self.as_ref(), other.as_ref());
                flag.set(prev);
                out
            })
        } else {
            WeightBackend::union(self.as_ref(), other.as_ref())
        };
        if let Some(start) = miss_start {
            OP_CACHE_OR_MISS_TIME_NS
                .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
            OP_CACHE_OR_MISS_COMPUTES.fetch_add(1, AtomicOrdering::Relaxed);
        }
        let intern_start = if profile_enabled { Some(Instant::now()) } else { None };
        let out = intern_rangemap(out);
        if let Some(start) = intern_start {
            OP_CACHE_OR_MISS_INTERN_NS
                .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        }
        put_op_cache(cache::BinOp::Or, self.clone(), other.clone(), out.clone());
        out
    }

    fn union_assign(&mut self, other: &Self) {
        *self = self.union(other);
    }

    fn difference(&self, other: &Self) -> Self {
        if Arc::ptr_eq(self, other) {
            return intern_rangemap(RangeMapWeight::new(self.num_tsids()));
        }
        if let Some(hit) = get_op_cache(cache::BinOp::Sub, self, other) {
            return hit;
        }
        let out = WeightBackend::difference(self.as_ref(), other.as_ref());
        let out = intern_rangemap(out);
        put_op_cache(cache::BinOp::Sub, self.clone(), other.clone(), out.clone());
        out
    }

    fn complement(&self, max_position: usize) -> Self {
        let all = <Arc<RangeMapWeight> as WeightBackend>::all(max_position);
        all.difference(self)
    }

    fn min_item(&self) -> Option<usize> {
        WeightBackend::min_item(self.as_ref())
    }

    fn max_item(&self) -> Option<usize> {
        WeightBackend::max_item(self.as_ref())
    }
}

/// Cached divide for Arc<RangeMapWeight> - computes self | !other with separate cache.
pub fn divide_rangemap_cached(a: &Arc<RangeMapWeight>, b: &Arc<RangeMapWeight>) -> Arc<RangeMapWeight> {
    // Only cache if both inputs are interned (pointer stability)
    if !is_interned_rangemap(a) || !is_interned_rangemap(b) {
        let out = a.divide(b);
        let out = intern_rangemap(out);
        return out;
    }
    
    // Create key from pointers
    let key = (Arc::as_ptr(a) as usize, Arc::as_ptr(b) as usize);
    
    // Check separate divide cache
    {
        let mut cache = RANGEMAP_DIVIDE_CACHE.lock().unwrap();
        let hit = cache.get(&key).cloned();
        if let Some(hit) = hit {
            return hit;
        }
    }
    
    // Compute divide: self | !other
    let rhs_comp_cache = get_rhs_comp_cache(b);
    let out = a.divide_with_rhs_comp_cache(b, rhs_comp_cache.as_deref());
    let out = intern_rangemap(out);
    
    // Cache result in separate divide cache
    {
        let mut cache = RANGEMAP_DIVIDE_CACHE.lock().unwrap();
        cache.push(key, out.clone());
    }
    
    out
}
