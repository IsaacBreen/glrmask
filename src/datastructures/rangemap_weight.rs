use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use dashmap::DashSet;
use lru::LruCache;
use once_cell::sync::Lazy;
use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};

use crate::datastructures::abstract_weight::{current_num_tsids, normalize_num_tsids, WeightBackend};
use crate::datastructures::cache;
use crate::datastructures::hybrid_bitset::RangeSet;
use profiler_macro::time_it;

const WEIGHT_OP_CACHE_CAPACITY: usize = 100_000;
const DIVIDE_CACHE_CAPACITY: usize = 50_000;
const DIVIDE_RHS_COMP_CACHE_CAPACITY: usize = 10_000;

static RANGEMAP_WEIGHT_INTERNER: Lazy<DashSet<Arc<RangeMapWeight>>> = Lazy::new(DashSet::new);
static RANGEMAP_OP_CACHE: Lazy<Mutex<LruCache<OpKey, Arc<RangeMapWeight>>>> = Lazy::new(|| {
    Mutex::new(LruCache::new(NonZeroUsize::new(WEIGHT_OP_CACHE_CAPACITY).unwrap()))
});
static RANGEMAP_OP_CACHE_INDEX: Lazy<Mutex<HashMap<usize, HashSet<OpKey>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct InternSiteKey {
    file: &'static str,
    line: u32,
    column: u32,
}

#[derive(Debug, Default, Clone, Copy)]
struct InternSiteStats {
    calls: u64,
    hits: u64,
    misses: u64,
}

static INTERN_SITE_STATS: Lazy<Mutex<HashMap<InternSiteKey, InternSiteStats>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// --- Profiling ---
static INTERN_TOTAL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_COUNT_OR: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_COUNT_OR_CACHE_HIT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_OR_TOTAL: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_OR_UNION: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_OR_INTERN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_OR_CACHE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_COUNT_OR_ASYM: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_OR_ASYM: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_COUNT_OR_MERGE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_OR_MERGE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_COUNT_DIVIDE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_COUNT_DIVIDE_ASYM: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_DIVIDE_PATH_ASYM_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_DIVIDE_PATH_ASYM_TIME: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_DIVIDE_PATH_REG_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_DIVIDE_PATH_REG_TIME: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_DIVIDE_ASYM_LHS_RANGES_TOTAL: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_DIVIDE_ASYM_RHS_RANGES_TOTAL: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_DIVIDE_ASYM_LHS_GE_50: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_DIVIDE_ASYM_LHS_GE_100: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_DIVIDE_REG_LHS_RANGES_TOTAL: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_DIVIDE_REG_RHS_RANGES_TOTAL: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_COUNT_DIVIDE_CACHE_HIT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_TOTAL: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_RIGHT_COMP: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_ASYM_COMP: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_ASYM_MERGE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_CACHE_LOOKUP: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_CACHE_INSERT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_INNER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_INTERN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_COMBINE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_INSERT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_COUNT_DIVIDE_INSERT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_BUILD_RESULT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_ADVANCE_LEFT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_ADVANCE_RIGHT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_LOOP_TOTAL: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_OTHER: std::sync::atomic::AtomicU64 =
     std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_VALS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_TIME_DIVIDE_IS_SAME: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_INTERN_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_INTERN_HIT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_INTERN_MISS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_INTERN_TIME_TOTAL: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_INTERN_TIME_LOOKUP: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_INTERN_TIME_INSERT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_INTERN_RANGE_TOTAL: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_INTERN_RANGE_HIT_TOTAL: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROF_RANGEMAP_INTERN_RANGE_MISS_TOTAL: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn intern_site_profile_enabled() -> bool {
    static ENABLED: Lazy<bool> = Lazy::new(|| {
        std::env::var("PROFILE_RANGEMAP_INTERN_SITES")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    });
    *ENABLED
}

pub fn reset_intern_wall_time() {
    INTERN_TOTAL_NS.store(0, std::sync::atomic::Ordering::Relaxed);
}

pub fn reset_profiling() {
    PROF_RANGEMAP_COUNT_OR.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_COUNT_OR_CACHE_HIT.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_OR_TOTAL.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_OR_UNION.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_OR_INTERN.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_OR_CACHE.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_COUNT_OR_ASYM.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_OR_ASYM.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_COUNT_OR_MERGE.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_OR_MERGE.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_COUNT_DIVIDE.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_COUNT_DIVIDE_ASYM.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_DIVIDE_PATH_ASYM_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_DIVIDE_PATH_ASYM_TIME.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_DIVIDE_PATH_REG_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_DIVIDE_PATH_REG_TIME.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_DIVIDE_ASYM_LHS_RANGES_TOTAL.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_DIVIDE_ASYM_RHS_RANGES_TOTAL.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_DIVIDE_ASYM_LHS_GE_50.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_DIVIDE_ASYM_LHS_GE_100.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_DIVIDE_REG_LHS_RANGES_TOTAL.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_DIVIDE_REG_RHS_RANGES_TOTAL.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_DIVIDE_TOTAL.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_DIVIDE_RIGHT_COMP.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_DIVIDE_ASYM_COMP.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_DIVIDE_ASYM_MERGE.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_DIVIDE_CACHE_LOOKUP.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_DIVIDE_CACHE_INSERT.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_DIVIDE_INNER.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_DIVIDE_INTERN.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_DIVIDE_COMBINE.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_DIVIDE_INSERT.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_COUNT_DIVIDE_INSERT.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_DIVIDE_BUILD_RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_DIVIDE_ADVANCE_LEFT.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_DIVIDE_ADVANCE_RIGHT.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_DIVIDE_LOOP_TOTAL.store(0, std::sync::atomic::Ordering::Relaxed);
        PROF_RANGEMAP_TIME_DIVIDE_OTHER.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_DIVIDE_VALS.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_TIME_DIVIDE_IS_SAME.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_INTERN_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_INTERN_HIT.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_INTERN_MISS.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_INTERN_TIME_TOTAL.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_INTERN_TIME_LOOKUP.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_INTERN_TIME_INSERT.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_INTERN_RANGE_TOTAL.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_INTERN_RANGE_HIT_TOTAL.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_INTERN_RANGE_MISS_TOTAL.store(0, std::sync::atomic::Ordering::Relaxed);
    INTERN_SITE_STATS.lock().unwrap().clear();
}

pub fn print_profiling(label: &str) {
    let count_or = PROF_RANGEMAP_COUNT_OR.load(std::sync::atomic::Ordering::Relaxed);
    let count_or_cache_hit = PROF_RANGEMAP_COUNT_OR_CACHE_HIT
        .load(std::sync::atomic::Ordering::Relaxed);
    let time_or_total = PROF_RANGEMAP_TIME_OR_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
    let time_or_union = PROF_RANGEMAP_TIME_OR_UNION.load(std::sync::atomic::Ordering::Relaxed);
    let time_or_intern = PROF_RANGEMAP_TIME_OR_INTERN.load(std::sync::atomic::Ordering::Relaxed);
    let time_or_cache = PROF_RANGEMAP_TIME_OR_CACHE.load(std::sync::atomic::Ordering::Relaxed);
    let count_or_asym = PROF_RANGEMAP_COUNT_OR_ASYM.load(std::sync::atomic::Ordering::Relaxed);
    let time_or_asym = PROF_RANGEMAP_TIME_OR_ASYM.load(std::sync::atomic::Ordering::Relaxed);
    let count_or_merge = PROF_RANGEMAP_COUNT_OR_MERGE.load(std::sync::atomic::Ordering::Relaxed);
    let time_or_merge = PROF_RANGEMAP_TIME_OR_MERGE.load(std::sync::atomic::Ordering::Relaxed);
    let count_divide = PROF_RANGEMAP_COUNT_DIVIDE.load(std::sync::atomic::Ordering::Relaxed);
    let count_divide_asym =
        PROF_RANGEMAP_COUNT_DIVIDE_ASYM.load(std::sync::atomic::Ordering::Relaxed);
    let divide_asym_count =
        PROF_RANGEMAP_DIVIDE_PATH_ASYM_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    let divide_asym_time =
        PROF_RANGEMAP_DIVIDE_PATH_ASYM_TIME.load(std::sync::atomic::Ordering::Relaxed);
    let divide_reg_count =
        PROF_RANGEMAP_DIVIDE_PATH_REG_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    let divide_reg_time =
        PROF_RANGEMAP_DIVIDE_PATH_REG_TIME.load(std::sync::atomic::Ordering::Relaxed);
    let divide_asym_lhs_total =
        PROF_RANGEMAP_DIVIDE_ASYM_LHS_RANGES_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
    let divide_asym_rhs_total =
        PROF_RANGEMAP_DIVIDE_ASYM_RHS_RANGES_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
    let divide_asym_lhs_ge_50 =
        PROF_RANGEMAP_DIVIDE_ASYM_LHS_GE_50.load(std::sync::atomic::Ordering::Relaxed);
    let divide_asym_lhs_ge_100 =
        PROF_RANGEMAP_DIVIDE_ASYM_LHS_GE_100.load(std::sync::atomic::Ordering::Relaxed);
    let divide_reg_lhs_total =
        PROF_RANGEMAP_DIVIDE_REG_LHS_RANGES_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
    let divide_reg_rhs_total =
        PROF_RANGEMAP_DIVIDE_REG_RHS_RANGES_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
    let time_divide_total =
        PROF_RANGEMAP_TIME_DIVIDE_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
    let time_divide_right_comp =
        PROF_RANGEMAP_TIME_DIVIDE_RIGHT_COMP.load(std::sync::atomic::Ordering::Relaxed);
    let time_divide_asym_comp =
        PROF_RANGEMAP_TIME_DIVIDE_ASYM_COMP.load(std::sync::atomic::Ordering::Relaxed);
    let time_divide_asym_merge =
        PROF_RANGEMAP_TIME_DIVIDE_ASYM_MERGE.load(std::sync::atomic::Ordering::Relaxed);
    let time_divide_cache_lookup =
        PROF_RANGEMAP_TIME_DIVIDE_CACHE_LOOKUP.load(std::sync::atomic::Ordering::Relaxed);
    let time_divide_cache_insert =
        PROF_RANGEMAP_TIME_DIVIDE_CACHE_INSERT.load(std::sync::atomic::Ordering::Relaxed);
    let time_divide_inner =
        PROF_RANGEMAP_TIME_DIVIDE_INNER.load(std::sync::atomic::Ordering::Relaxed);
    let time_divide_intern =
        PROF_RANGEMAP_TIME_DIVIDE_INTERN.load(std::sync::atomic::Ordering::Relaxed);
    let time_divide_combine =
        PROF_RANGEMAP_TIME_DIVIDE_COMBINE.load(std::sync::atomic::Ordering::Relaxed);
    let time_divide_insert =
        PROF_RANGEMAP_TIME_DIVIDE_INSERT.load(std::sync::atomic::Ordering::Relaxed);
    let time_divide_build_result =
        PROF_RANGEMAP_TIME_DIVIDE_BUILD_RESULT.load(std::sync::atomic::Ordering::Relaxed);
    let count_divide_insert =
        PROF_RANGEMAP_COUNT_DIVIDE_INSERT.load(std::sync::atomic::Ordering::Relaxed);
    let time_divide_advance_left =
        PROF_RANGEMAP_TIME_DIVIDE_ADVANCE_LEFT.load(std::sync::atomic::Ordering::Relaxed);
    let time_divide_advance_right =
        PROF_RANGEMAP_TIME_DIVIDE_ADVANCE_RIGHT.load(std::sync::atomic::Ordering::Relaxed);
    let time_divide_loop_total =
        PROF_RANGEMAP_TIME_DIVIDE_LOOP_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
        let time_divide_other =
            PROF_RANGEMAP_TIME_DIVIDE_OTHER.load(std::sync::atomic::Ordering::Relaxed);
    let time_divide_vals =
        PROF_RANGEMAP_TIME_DIVIDE_VALS.load(std::sync::atomic::Ordering::Relaxed);
    let time_divide_is_same =
        PROF_RANGEMAP_TIME_DIVIDE_IS_SAME.load(std::sync::atomic::Ordering::Relaxed);
    let intern_count = PROF_RANGEMAP_INTERN_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    let intern_hit = PROF_RANGEMAP_INTERN_HIT.load(std::sync::atomic::Ordering::Relaxed);
    let intern_miss = PROF_RANGEMAP_INTERN_MISS.load(std::sync::atomic::Ordering::Relaxed);
    let intern_time_total =
        PROF_RANGEMAP_INTERN_TIME_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
    let intern_time_lookup =
        PROF_RANGEMAP_INTERN_TIME_LOOKUP.load(std::sync::atomic::Ordering::Relaxed);
    let intern_time_insert =
        PROF_RANGEMAP_INTERN_TIME_INSERT.load(std::sync::atomic::Ordering::Relaxed);
    let intern_range_total =
        PROF_RANGEMAP_INTERN_RANGE_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
    let intern_range_hit_total =
        PROF_RANGEMAP_INTERN_RANGE_HIT_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
    let intern_range_miss_total =
        PROF_RANGEMAP_INTERN_RANGE_MISS_TOTAL.load(std::sync::atomic::Ordering::Relaxed);

    let count_or_fast = count_or_asym.saturating_add(count_or_merge);
    if count_or > 0 || count_or_fast > 0 || count_divide > 0 || intern_count > 0 {
        println!("RANGEMAP_WEIGHT_PROF [{}]:", label);
        if count_or > 0 {
            println!(
                "  OR (arc): {:9} ops, {:9} us (avg {:.2} us)",
                count_or,
                time_or_total,
                time_or_total as f64 / count_or as f64
            );
            println!(
                "  OR (arc) breakdown: union={} us, intern={} us, cache={} us, cache_hits={}",
                time_or_union,
                time_or_intern,
                time_or_cache,
                count_or_cache_hit,
            );
        }
        if count_or_fast > 0 {
            let fast_time = time_or_asym.saturating_add(time_or_merge);
            println!(
                "  OR (union_fast): {:9} ops, {:9} us (avg {:.2} us)",
                count_or_fast,
                fast_time,
                fast_time as f64 / count_or_fast as f64
            );
            println!(
                "  OR (union_fast) breakdown: asym={} ({} us), merge={} ({} us)",
                count_or_asym,
                time_or_asym,
                count_or_merge,
                time_or_merge,
            );
        }
        if count_divide > 0 {
            println!(
                "  DIV: {:9} ops, {:9} us (avg {:.2} us)",
                count_divide,
                time_divide_total,
                time_divide_total as f64 / count_divide as f64
            );
            println!(
                "  DIV breakdown: right_comp={} us, combine={} us, insert={} us, inserts={}, advance_left={} us, advance_right={} us, vals={} us, is_same={} us, loop_total={} us, other={} us, asym={}",
                time_divide_right_comp,
                time_divide_combine,
                time_divide_insert,
                count_divide_insert,
                time_divide_advance_left,
                time_divide_advance_right,
                time_divide_vals,
                time_divide_is_same,
                time_divide_loop_total,
                time_divide_other,
                count_divide_asym,
            );
            println!(
                "  DIV extra: asym_comp={} us, asym_merge={} us, build_result={} us, cache_lookup={} us, cache_insert={} us, inner={} us, intern={} us",
                time_divide_asym_comp,
                time_divide_asym_merge,
                time_divide_build_result,
                time_divide_cache_lookup,
                time_divide_cache_insert,
                time_divide_inner,
                time_divide_intern,
            );
            if divide_asym_count > 0 || divide_reg_count > 0 {
                println!(
                    "  DIV paths: asym={} ops, {} us (avg {:.2} us); reg={} ops, {} us (avg {:.2} us)",
                    divide_asym_count,
                    divide_asym_time,
                    if divide_asym_count > 0 {
                        divide_asym_time as f64 / divide_asym_count as f64
                    } else {
                        0.0
                    },
                    divide_reg_count,
                    divide_reg_time,
                    if divide_reg_count > 0 {
                        divide_reg_time as f64 / divide_reg_count as f64
                    } else {
                        0.0
                    },
                );
            }
            if divide_asym_count > 0 {
                println!(
                    "  DIV asym ranges: lhs_avg={:.2}, rhs_avg={:.2}, lhs_ge_50={}, lhs_ge_100={}",
                    divide_asym_lhs_total as f64 / divide_asym_count as f64,
                    divide_asym_rhs_total as f64 / divide_asym_count as f64,
                    divide_asym_lhs_ge_50,
                    divide_asym_lhs_ge_100,
                );
            }
            if divide_reg_count > 0 {
                println!(
                    "  DIV reg ranges: lhs_avg={:.2}, rhs_avg={:.2}",
                    divide_reg_lhs_total as f64 / divide_reg_count as f64,
                    divide_reg_rhs_total as f64 / divide_reg_count as f64,
                );
            }
        }
        if intern_count > 0 {
            println!(
                "  INTERN: {:9} ops, {:9} us (avg {:.2} us), hits={}, misses={}, lookup={} us, insert={} us",
                intern_count,
                intern_time_total,
                intern_time_total as f64 / intern_count as f64,
                intern_hit,
                intern_miss,
                intern_time_lookup,
                intern_time_insert,
            );
            println!(
                "  INTERN ranges: total={}, avg_total={:.2}, avg_hit={:.2}, avg_miss={:.2}",
                intern_range_total,
                intern_range_total as f64 / intern_count as f64,
                if intern_hit > 0 {
                    intern_range_hit_total as f64 / intern_hit as f64
                } else {
                    0.0
                },
                if intern_miss > 0 {
                    intern_range_miss_total as f64 / intern_miss as f64
                } else {
                    0.0
                }
            );
        }
        if intern_site_profile_enabled() {
            let mut sites: Vec<(InternSiteKey, InternSiteStats)> = INTERN_SITE_STATS
                .lock()
                .unwrap()
                .iter()
                .map(|(k, v)| (*k, *v))
                .collect();
            if !sites.is_empty() {
                sites.sort_by(|a, b| b.1.calls.cmp(&a.1.calls));
                let show = sites.len().min(12);
                println!("  INTERN call sites (top {} by calls):", show);
                for (key, stats) in sites.into_iter().take(show) {
                    let hit_rate = if stats.calls == 0 {
                        0.0
                    } else {
                        (stats.hits as f64) * 100.0 / (stats.calls as f64)
                    };
                    println!(
                        "    {}:{}:{} calls={} hits={} misses={} hit_rate={:.1}%",
                        key.file,
                        key.line,
                        key.column,
                        stats.calls,
                        stats.hits,
                        stats.misses,
                        hit_rate
                    );
                }
            }
        }
    }
}

pub fn print_intern_wall_time(label: &str) {
    let total_ns = INTERN_TOTAL_NS.load(std::sync::atomic::Ordering::Relaxed);
    let total_ms = total_ns as f64 / 1_000_000.0;
    let interner_len = RANGEMAP_WEIGHT_INTERNER.len();
    println!(
        "RANGEMAP_INTERN_WALL [{}]: total_ns={}, total_ms={:.2}, interner_len={}",
        label, total_ns, total_ms, interner_len
    );
}

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

fn remove_op_key_from_index(index: &mut HashMap<usize, HashSet<OpKey>>, key: OpKey) {
    if let Some(set) = index.get_mut(&key.a) {
        set.remove(&key);
        if set.is_empty() {
            index.remove(&key.a);
        }
    }
    if let Some(set) = index.get_mut(&key.b) {
        set.remove(&key);
        if set.is_empty() {
            index.remove(&key.b);
        }
    }
}

fn invalidate_rangemap_op_cache_for_ptr(ptr: usize) {
    let mut cache = RANGEMAP_OP_CACHE.lock().unwrap();
    let mut index = RANGEMAP_OP_CACHE_INDEX.lock().unwrap();
    let Some(keys) = index.remove(&ptr) else { return; };
    for key in keys {
        cache.pop(&key);
        remove_op_key_from_index(&mut index, key);
    }
}

fn get_op_cache(op: cache::BinOp, a: &Arc<RangeMapWeight>, b: &Arc<RangeMapWeight>) -> Option<Arc<RangeMapWeight>> {
    if !is_interned_rangemap(a) || !is_interned_rangemap(b) {
        return None;
    }
    let mut cache = RANGEMAP_OP_CACHE.lock().unwrap();
    let key = op_key(op, a, b);
    if let Some(hit) = cache.get(&key) {
        return Some(hit.clone());
    }
    if matches!(op, cache::BinOp::And | cache::BinOp::Or | cache::BinOp::Xor) {
        let swapped = op_key(op, b, a);
        if let Some(hit) = cache.get(&swapped) {
            return Some(hit.clone());
        }
    }
    None
}

fn put_op_cache(
    op: cache::BinOp,
    a: Arc<RangeMapWeight>,
    b: Arc<RangeMapWeight>,
    result: Arc<RangeMapWeight>,
) {
    if !is_interned_rangemap(&a) || !is_interned_rangemap(&b) {
        return;
    }
    let key = op_key(op, &a, &b);
    let mut cache = RANGEMAP_OP_CACHE.lock().unwrap();
    let mut index = RANGEMAP_OP_CACHE_INDEX.lock().unwrap();
    if let Some((evicted_key, _)) = cache.push(key, result) {
        remove_op_key_from_index(&mut index, evicted_key);
    }
    index.entry(key.a).or_default().insert(key);
    index.entry(key.b).or_default().insert(key);
}

fn build_rhs_comp_cache(rhs: &Arc<RangeMapWeight>) -> Arc<RhsCompCache> {
    let full_tsids = RangeMapWeight::full_tsids(rhs.num_tsids());
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

#[time_it("intern_rangemap")]
#[track_caller]
pub fn intern_rangemap(weight: RangeMapWeight) -> Arc<RangeMapWeight> {
    let wall_start = std::time::Instant::now();
    PROF_RANGEMAP_INTERN_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let range_count = RangeMapWeight::map_range_count(&weight.map) as u64;
    PROF_RANGEMAP_INTERN_RANGE_TOTAL.fetch_add(range_count, std::sync::atomic::Ordering::Relaxed);
    let site_key = if intern_site_profile_enabled() {
        let loc = std::panic::Location::caller();
        Some(InternSiteKey {
            file: loc.file(),
            line: loc.line(),
            column: loc.column(),
        })
    } else {
        None
    };
    let total_start = std::time::Instant::now();
    let lookup_start = std::time::Instant::now();
    if let Some(existing) = RANGEMAP_WEIGHT_INTERNER.get(&weight) {
        let lookup_time = lookup_start.elapsed();
        PROF_RANGEMAP_INTERN_TIME_LOOKUP.fetch_add(
            lookup_time.as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        PROF_RANGEMAP_INTERN_HIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        PROF_RANGEMAP_INTERN_RANGE_HIT_TOTAL.fetch_add(
            range_count,
            std::sync::atomic::Ordering::Relaxed,
        );
        if let Some(key) = site_key {
            let mut map = INTERN_SITE_STATS.lock().unwrap();
            let entry = map.entry(key).or_default();
            entry.calls += 1;
            entry.hits += 1;
        }
        let existing = Arc::clone(&*existing);
        PROF_RANGEMAP_INTERN_TIME_TOTAL.fetch_add(
            total_start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        INTERN_TOTAL_NS.fetch_add(
            wall_start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        return existing;
    }
    let lookup_time = lookup_start.elapsed();
    PROF_RANGEMAP_INTERN_TIME_LOOKUP.fetch_add(
        lookup_time.as_micros() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    PROF_RANGEMAP_INTERN_MISS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    PROF_RANGEMAP_INTERN_RANGE_MISS_TOTAL.fetch_add(
        range_count,
        std::sync::atomic::Ordering::Relaxed,
    );
    if let Some(key) = site_key {
        let mut map = INTERN_SITE_STATS.lock().unwrap();
        let entry = map.entry(key).or_default();
        entry.calls += 1;
        entry.misses += 1;
    }
    let insert_start = std::time::Instant::now();
    let arc = Arc::new(weight);
    let inserted = RANGEMAP_WEIGHT_INTERNER.insert(arc.clone());
    if inserted {
        let ptr = Arc::as_ptr(&arc) as usize;
        invalidate_rangemap_op_cache_for_ptr(ptr);
    }
    let out = if inserted {
        arc
    } else if let Some(existing) = RANGEMAP_WEIGHT_INTERNER.get(&arc) {
        Arc::clone(&*existing)
    } else {
        arc
    };
    PROF_RANGEMAP_INTERN_TIME_INSERT.fetch_add(
        insert_start.elapsed().as_micros() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    PROF_RANGEMAP_INTERN_TIME_TOTAL.fetch_add(
        total_start.elapsed().as_micros() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    INTERN_TOTAL_NS.fetch_add(
        wall_start.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeMapWeight {
    /// Maps token_id -> set of tsid values (stored as ranges over token_id).
    pub(crate) map: RangeMapBlaze<usize, RangeSet>,
    pub(crate) num_tsids: usize,
    cached_hash: u64,
}

impl RangeMapWeight {
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

    fn from_map(map: RangeMapBlaze<usize, RangeSet>, num_tsids: usize) -> Self {
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
            let comp_start = std::time::Instant::now();
            Self::rangeset_complement_ranges(rhs, max_tsid, &mut comp_ranges);
            PROF_RANGEMAP_TIME_DIVIDE_ASYM_COMP.fetch_add(
                comp_start.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
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
        let comp_start = std::time::Instant::now();
        Self::rangeset_complement_ranges(rhs, max_tsid, &mut comp_ranges);
        PROF_RANGEMAP_TIME_DIVIDE_ASYM_COMP.fetch_add(
            comp_start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let merge_start = std::time::Instant::now();
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
        PROF_RANGEMAP_TIME_DIVIDE_ASYM_MERGE.fetch_add(
            merge_start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        out
    }

    fn ensure_right_comp<'a>(
        rv: &RangeSet,
        full_tsids: &RangeSet,
        right_comp: &'a mut Option<RangeSet>,
        right_ptr: &mut Option<*const RangeSetBlaze<usize>>,
        right_comp_time: &mut std::time::Duration,
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
            let right_comp_start = std::time::Instant::now();
            *right_comp = Some(full_tsids - rv);
            let elapsed = right_comp_start.elapsed();
            *right_comp_time += elapsed;
            PROF_RANGEMAP_TIME_DIVIDE_RIGHT_COMP.fetch_add(
                elapsed.as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
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

    fn from_token_map(map: BTreeMap<usize, RangeSet>, num_tsids: usize) -> Self {
        let num_tsids = normalize_num_tsids(num_tsids);
        if map.is_empty() {
            return Self::new(num_tsids);
        }

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

        Self::from_map(out, num_tsids)
    }

    pub(crate) fn from_rsb_with_num_tsids(rsb: &RangeSetBlaze<usize>, num_tsids: usize) -> Self {
        let num_tsids = normalize_num_tsids(num_tsids);
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

    #[time_it("RangeMapWeight::union_all_non_negated")]
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

            let mut combined = RangeSet::zeros();
            for &idx in &active_indices {
                if let Some(val) = &current_values[idx] {
                    combined |= val;
                }
            }

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

    #[time_it("RangeMapWeight::union_all")]
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

    #[time_it("RangeMapWeight::union_non_negated")]
    fn union_non_negated(&self, other: &Self) -> Self {
        let left_ranges = Self::map_range_count(&self.map);
        let right_ranges = Self::map_range_count(&other.map);
        if left_ranges == 0 {
            return other.clone();
        }
        if right_ranges == 0 {
            return self.clone();
        }

        let (smaller, larger, small_ranges, large_ranges) = if left_ranges <= right_ranges {
            (self, other, left_ranges, right_ranges)
        } else {
            (other, self, right_ranges, left_ranges)
        };

        let map = if small_ranges.saturating_mul(10) < large_ranges {
            PROF_RANGEMAP_COUNT_OR_ASYM.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let start = std::time::Instant::now();
            let map = Self::union_asymmetric(&smaller.map, &larger.map);
            PROF_RANGEMAP_TIME_OR_ASYM.fetch_add(
                start.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            map
        } else {
            PROF_RANGEMAP_COUNT_OR_MERGE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let start = std::time::Instant::now();
            let map = Self::merge_maps(&self.map, &other.map, |left, right| match (left, right) {
                (Some(a), Some(b)) => a | b,
                (Some(a), None) => a.clone(),
                (None, Some(b)) => b.clone(),
                (None, None) => RangeSet::zeros(),
            });
            PROF_RANGEMAP_TIME_OR_MERGE.fetch_add(
                start.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            map
        };
        Self::from_map(map, self.num_tsids())
    }

    #[time_it("RangeMapWeight::union_fast")]
    pub(crate) fn union_fast(&self, other: &Self) -> Self {
        self.union_non_negated(other)
    }

    #[time_it("RangeMapWeight::intersect_non_negated")]
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
        crate::datastructures::hybrid_bitset::PROF_COUNT_DIVIDE.fetch_add(
            1,
            std::sync::atomic::Ordering::Relaxed,
        );
        PROF_RANGEMAP_COUNT_DIVIDE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let start = std::time::Instant::now();
        let num_tsids = self.num_tsids();
        let full_tsids = Self::full_tsids(num_tsids);
        let max_tsid = num_tsids.saturating_sub(1);

        let mut left_iter = self.map.range_values();
        let mut right_iter = other.map.range_values();
        let mut left = left_iter.next();
        let mut right = right_iter.next();

        if left.is_none() && right.is_none() {
            let result = Self::from_map(RangeMapBlaze::new(), num_tsids);
            crate::datastructures::hybrid_bitset::PROF_TIME_DIVIDE.fetch_add(
                start.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            PROF_RANGEMAP_TIME_DIVIDE_TOTAL.fetch_add(
                start.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
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
            let loop_start = std::time::Instant::now();
            let mut loop_accounted = std::time::Duration::ZERO;
            let advance_left_start = std::time::Instant::now();
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
            let advance_left_time = advance_left_start.elapsed();
            PROF_RANGEMAP_TIME_DIVIDE_ADVANCE_LEFT.fetch_add(
                advance_left_time.as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            loop_accounted += advance_left_time;

            let advance_right_start = std::time::Instant::now();
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
            let advance_right_time = advance_right_start.elapsed();
            PROF_RANGEMAP_TIME_DIVIDE_ADVANCE_RIGHT.fetch_add(
                advance_right_time.as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            loop_accounted += advance_right_time;

            let vals_start = std::time::Instant::now();
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
            let vals_time = vals_start.elapsed();
            PROF_RANGEMAP_TIME_DIVIDE_VALS.fetch_add(
                vals_time.as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            loop_accounted += vals_time;

            let mut right_comp_time = std::time::Duration::ZERO;
            if right_val.is_none() {
                right_comp = None;
                right_ptr = None;
            }

            let combine_start = std::time::Instant::now();
            let combined = match (left_val, right_val) {
                (Some(a), Some(rv)) => {
                    let lhs_ranges = a.ranges_len();
                    let rhs_ranges = rv.ranges_len();
                    if lhs_ranges.saturating_mul(2) < rhs_ranges {
                        PROF_RANGEMAP_COUNT_DIVIDE_ASYM.fetch_add(
                            1,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        PROF_RANGEMAP_DIVIDE_PATH_ASYM_COUNT.fetch_add(
                            1,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        PROF_RANGEMAP_DIVIDE_ASYM_LHS_RANGES_TOTAL.fetch_add(
                            lhs_ranges as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        PROF_RANGEMAP_DIVIDE_ASYM_RHS_RANGES_TOTAL.fetch_add(
                            rhs_ranges as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        if lhs_ranges >= 50 {
                            PROF_RANGEMAP_DIVIDE_ASYM_LHS_GE_50.fetch_add(
                                1,
                                std::sync::atomic::Ordering::Relaxed,
                            );
                        }
                        if lhs_ranges >= 100 {
                            PROF_RANGEMAP_DIVIDE_ASYM_LHS_GE_100.fetch_add(
                                1,
                                std::sync::atomic::Ordering::Relaxed,
                            );
                        }
                        let branch_start = std::time::Instant::now();
                        let out = Self::rangeset_union_with_complement_asymmetric(a, rv, max_tsid);
                        PROF_RANGEMAP_DIVIDE_PATH_ASYM_TIME.fetch_add(
                            branch_start.elapsed().as_micros() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        out
                    } else {
                        PROF_RANGEMAP_DIVIDE_PATH_REG_COUNT.fetch_add(
                            1,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        PROF_RANGEMAP_DIVIDE_REG_LHS_RANGES_TOTAL.fetch_add(
                            lhs_ranges as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        PROF_RANGEMAP_DIVIDE_REG_RHS_RANGES_TOTAL.fetch_add(
                            rhs_ranges as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        let branch_start = std::time::Instant::now();
                        let comp = Self::ensure_right_comp(
                            rv,
                            &full_tsids,
                            &mut right_comp,
                            &mut right_ptr,
                            &mut right_comp_time,
                            rhs_comp_cache,
                        );
                        let out = a | comp;
                        PROF_RANGEMAP_DIVIDE_PATH_REG_TIME.fetch_add(
                            branch_start.elapsed().as_micros() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
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
                        &mut right_comp_time,
                        rhs_comp_cache,
                    );
                    comp.clone()
                }
                (None, None) => full_tsids.clone(),
            };
            let combine_time = combine_start.elapsed();
            PROF_RANGEMAP_TIME_DIVIDE_COMBINE.fetch_add(
                combine_time.as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            loop_accounted += right_comp_time;
            loop_accounted += combine_time;

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

            let mut insert_time = std::time::Duration::ZERO;
            let mut is_same_time = std::time::Duration::ZERO;
            if combined.is_empty() {
                if let Some(range_start) = current_start.take() {
                    let insert_start = std::time::Instant::now();
                    out.ranges_insert(range_start..=current_end, current_value.clone());
                    insert_time = insert_start.elapsed();
                    PROF_RANGEMAP_TIME_DIVIDE_INSERT.fetch_add(
                        insert_time.as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    PROF_RANGEMAP_COUNT_DIVIDE_INSERT.fetch_add(
                        1,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
            } else if let Some(range_start) = current_start {
                let is_same_start = std::time::Instant::now();
                let is_same = Arc::ptr_eq(&current_value.inner, &combined.inner)
                    || current_value == combined;
                is_same_time = is_same_start.elapsed();
                PROF_RANGEMAP_TIME_DIVIDE_IS_SAME.fetch_add(
                    is_same_time.as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                if is_same && current_end.saturating_add(1) == pos {
                    current_end = end;
                } else {
                    let insert_start = std::time::Instant::now();
                    out.ranges_insert(range_start..=current_end, current_value.clone());
                    insert_time = insert_start.elapsed();
                    PROF_RANGEMAP_TIME_DIVIDE_INSERT.fetch_add(
                        insert_time.as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    PROF_RANGEMAP_COUNT_DIVIDE_INSERT.fetch_add(
                        1,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    current_start = Some(pos);
                    current_end = end;
                    current_value = combined;
                }
            } else {
                current_start = Some(pos);
                current_end = end;
                current_value = combined;
            }

            loop_accounted += insert_time;
            loop_accounted += is_same_time;

            let loop_elapsed = loop_start.elapsed();
            PROF_RANGEMAP_TIME_DIVIDE_LOOP_TOTAL.fetch_add(
                loop_elapsed.as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            if let Some(other) = loop_elapsed.checked_sub(loop_accounted) {
                PROF_RANGEMAP_TIME_DIVIDE_OTHER.fetch_add(
                    other.as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }

            if end == usize::MAX {
                break;
            }
            pos = end.saturating_add(1);
        }

        if let Some(range_start) = current_start {
            let insert_start = std::time::Instant::now();
            out.ranges_insert(range_start..=current_end, current_value);
            PROF_RANGEMAP_TIME_DIVIDE_INSERT.fetch_add(
                insert_start.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            PROF_RANGEMAP_COUNT_DIVIDE_INSERT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        let build_start = std::time::Instant::now();
        let result = Self::from_map(out, num_tsids);
        PROF_RANGEMAP_TIME_DIVIDE_BUILD_RESULT.fetch_add(
            build_start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        crate::datastructures::hybrid_bitset::PROF_TIME_DIVIDE.fetch_add(
            start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        PROF_RANGEMAP_TIME_DIVIDE_TOTAL.fetch_add(
            start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        result
    }

    pub(crate) fn clip_to_max(&mut self, max: usize) {
        if self.map.is_empty() {
            return;
        }

        let num_tsids = self.num_tsids();
        let max_token = max / num_tsids;
        let max_tsid = max % num_tsids;
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
        let tsid_set = Self::rangeset_from_ranges([tsid..=tsid]);
        let mut map = RangeMapBlaze::new();
        map.ranges_insert(token..=token, tsid_set);
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
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        self.map.get(token).map_or(false, |tsids| tsids.contains(tsid))
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
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        let mut new_set = Self::rangeset_from_ranges([tsid..=tsid]);
        if let Some(existing) = self.map.get(token) {
            new_set |= existing;
        }
        self.map.ranges_insert(token..=token, new_set);
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
        for (token_range, tsid_set) in self.map.range_values() {
            let token = *token_range.start();
            let tsid = tsid_set.ranges().next().map(|r| *r.start());
            if let Some(tsid) = tsid {
                let pos = token.saturating_mul(num_tsids).saturating_add(tsid);
                min_pos = Some(min_pos.map_or(pos, |m| m.min(pos)));
            }
        }
        min_pos
    }

    fn max_item(&self) -> Option<usize> {
        let num_tsids = self.num_tsids();
        let mut max_pos: Option<usize> = None;
        for (token_range, tsid_set) in self.map.range_values() {
            let token = *token_range.end();
            let tsid = tsid_set.ranges().last().map(|r| *r.end());
            if let Some(tsid) = tsid {
                let pos = token.saturating_mul(num_tsids).saturating_add(tsid);
                max_pos = Some(max_pos.map_or(pos, |m| m.max(pos)));
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
        PROF_RANGEMAP_COUNT_OR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if Arc::ptr_eq(self, other) {
            return self.clone();
        }
        if let Some(hit) = get_op_cache(cache::BinOp::Or, self, other) {
            PROF_RANGEMAP_COUNT_OR_CACHE_HIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return hit;
        }
        let union_start = std::time::Instant::now();
        let out = WeightBackend::union(self.as_ref(), other.as_ref());
        let union_time = union_start.elapsed();
        let intern_start = std::time::Instant::now();
        let out = intern_rangemap(out);
        let intern_time = intern_start.elapsed();
        let cache_start = std::time::Instant::now();
        put_op_cache(cache::BinOp::Or, self.clone(), other.clone(), out.clone());
        let cache_time = cache_start.elapsed();
        PROF_RANGEMAP_TIME_OR_TOTAL.fetch_add(
            (union_time + intern_time + cache_time).as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        PROF_RANGEMAP_TIME_OR_UNION.fetch_add(
            union_time.as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        PROF_RANGEMAP_TIME_OR_INTERN.fetch_add(
            intern_time.as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        PROF_RANGEMAP_TIME_OR_CACHE.fetch_add(
            cache_time.as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
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
        let out = WeightBackend::complement(self.as_ref(), max_position);
        intern_rangemap(out)
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
    PROF_RANGEMAP_COUNT_DIVIDE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    
    // Only cache if both inputs are interned (pointer stability)
    if !is_interned_rangemap(a) || !is_interned_rangemap(b) {
        let inner_start = std::time::Instant::now();
        let out = a.divide(b);
        PROF_RANGEMAP_TIME_DIVIDE_INNER.fetch_add(
            inner_start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let intern_start = std::time::Instant::now();
        let out = intern_rangemap(out);
        PROF_RANGEMAP_TIME_DIVIDE_INTERN.fetch_add(
            intern_start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        return out;
    }
    
    // Create key from pointers
    let key = (Arc::as_ptr(a) as usize, Arc::as_ptr(b) as usize);
    
    // Check separate divide cache
    {
        let lookup_start = std::time::Instant::now();
        let mut cache = RANGEMAP_DIVIDE_CACHE.lock().unwrap();
        let hit = cache.get(&key).cloned();
        PROF_RANGEMAP_TIME_DIVIDE_CACHE_LOOKUP.fetch_add(
            lookup_start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        if let Some(hit) = hit {
            PROF_RANGEMAP_COUNT_DIVIDE_CACHE_HIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return hit;
        }
    }
    
    // Compute divide: self | !other
    let rhs_comp_cache = get_rhs_comp_cache(b);
    let inner_start = std::time::Instant::now();
    let out = a.divide_with_rhs_comp_cache(b, rhs_comp_cache.as_deref());
    PROF_RANGEMAP_TIME_DIVIDE_INNER.fetch_add(
        inner_start.elapsed().as_micros() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    let intern_start = std::time::Instant::now();
    let out = intern_rangemap(out);
    PROF_RANGEMAP_TIME_DIVIDE_INTERN.fetch_add(
        intern_start.elapsed().as_micros() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    
    // Cache result in separate divide cache
    {
        let cache_start = std::time::Instant::now();
        let mut cache = RANGEMAP_DIVIDE_CACHE.lock().unwrap();
        cache.push(key, out.clone());
        PROF_RANGEMAP_TIME_DIVIDE_CACHE_INSERT.fetch_add(
            cache_start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    
    out
}
