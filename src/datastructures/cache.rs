use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::hybrid_l2_bitset::HybridL2Bitset;
use lru::LruCache;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::Arc;

// --- Acc Type ---
pub type Acc<T> = Arc<T>;

// --- Operation Enum ---
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    And,
    Or,
    Xor,
    Sub,
}

// --- Cache Keys ---
// Key for L1 bitset operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct L1OpKey {
    op: BinOp,
    a: Acc<RangeSetBlaze<usize>>,
    b: Acc<RangeSetBlaze<usize>>,
}

// Key for L2 bitset operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct L2OpKey {
    op: BinOp,
    a: Acc<RangeMapBlaze<usize, HybridBitset>>,
    b: Acc<RangeMapBlaze<usize, HybridBitset>>,
}

// --- Global Caches ---
const VALUE_CACHE_CAPACITY: usize = 100_000;
const OP_CACHE_CAPACITY: usize = 100_000;

struct Caches {
    // Value caches (interning pools)
    l1_values: LruCache<RangeSetBlaze<usize>, Acc<RangeSetBlaze<usize>>>,
    l2_values: LruCache<RangeMapBlaze<usize, HybridBitset>, Acc<RangeMapBlaze<usize, HybridBitset>>>,

    // Operation caches
    l1_ops: LruCache<L1OpKey, Acc<RangeSetBlaze<usize>>>,
    l2_ops: LruCache<L2OpKey, Acc<RangeMapBlaze<usize, HybridBitset>>>,
}

impl Caches {
    fn new() -> Self {
        Caches {
            l1_values: LruCache::new(NonZeroUsize::new(VALUE_CACHE_CAPACITY).unwrap()),
            l2_values: LruCache::new(NonZeroUsize::new(VALUE_CACHE_CAPACITY).unwrap()),
            l1_ops: LruCache::new(NonZeroUsize::new(OP_CACHE_CAPACITY).unwrap()),
            l2_ops: LruCache::new(NonZeroUsize::new(OP_CACHE_CAPACITY).unwrap()),
        }
    }
}

static GLOBAL_CACHES: Lazy<Mutex<Caches>> = Lazy::new(|| Mutex::new(Caches::new()));

// --- Heuristics ---
pub const SIMPLE_BITSET_THRESHOLD: usize = 16;
pub const SIMPLE_L2_BITSET_THRESHOLD: usize = 8;

// --- Cache Access Functions ---

// L1 (HybridBitset)
pub fn intern_l1(rs: RangeSetBlaze<usize>) -> Acc<RangeSetBlaze<usize>> {
    let mut caches = GLOBAL_CACHES.lock();
    if let Some(acc) = caches.l1_values.get(&rs) {
        return acc.clone();
    }
    let acc = Acc::new(rs.clone());
    caches.l1_values.put(rs, acc.clone());
    acc
}

pub fn get_l1_op_cache(
    op: BinOp,
    a: &Acc<RangeSetBlaze<usize>>,
    b: &Acc<RangeSetBlaze<usize>>,
) -> Option<Acc<RangeSetBlaze<usize>>> {
    let mut caches = GLOBAL_CACHES.lock();
    let key = L1OpKey { op, a: a.clone(), b: b.clone() };
    caches.l1_ops.get(&key).cloned()
}

pub fn put_l1_op_cache(
    op: BinOp,
    a: Acc<RangeSetBlaze<usize>>,
    b: Acc<RangeSetBlaze<usize>>,
    result: Acc<RangeSetBlaze<usize>>,
) {
    let mut caches = GLOBAL_CACHES.lock();
    let key = L1OpKey { op, a, b };
    caches.l1_ops.put(key, result);
}

// L2 (HybridL2Bitset)
pub fn intern_l2(
    rm: RangeMapBlaze<usize, HybridBitset>,
) -> Acc<RangeMapBlaze<usize, HybridBitset>> {
    let mut caches = GLOBAL_CACHES.lock();
    if let Some(acc) = caches.l2_values.get(&rm) {
        return acc.clone();
    }
    let acc = Acc::new(rm.clone());
    caches.l2_values.put(rm, acc.clone());
    acc
}

pub fn get_l2_op_cache(
    op: BinOp,
    a: &Acc<RangeMapBlaze<usize, HybridBitset>>,
    b: &Acc<RangeMapBlaze<usize, HybridBitset>>,
) -> Option<Acc<RangeMapBlaze<usize, HybridBitset>>> {
    let mut caches = GLOBAL_CACHES.lock();
    let key = L2OpKey { op, a: a.clone(), b: b.clone() };
    caches.l2_ops.get(&key).cloned()
}

pub fn put_l2_op_cache(
    op: BinOp,
    a: Acc<RangeMapBlaze<usize, HybridBitset>>,
    b: Acc<RangeMapBlaze<usize, HybridBitset>>,
    result: Acc<RangeMapBlaze<usize, HybridBitset>>,
) {
    let mut caches = GLOBAL_CACHES.lock();
    let key = L2OpKey { op, a, b };
    caches.l2_ops.put(key, result);
}
