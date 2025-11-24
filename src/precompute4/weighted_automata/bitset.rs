// src/precompute4/weighted_automata/bitset.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use lru::LruCache;
use once_cell::sync::Lazy;
use range_set_blaze::RangeSetBlaze;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::num::NonZeroUsize;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Deref, Not, Sub, SubAssign};
use std::sync::{Arc, Mutex};
use crate::datastructures::hybrid_bitset::HybridBitset;

/// Thin wrapper around `RangeSetBlaze<usize>` with cached fingerprint and `is_all` flag.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct SimpleBitset(Arc<SimpleBitsetInner>);

#[derive(Eq, PartialEq, Ord, PartialOrd)]
pub struct SimpleBitsetInner {
    pub(crate) rsb: RangeSetBlaze<usize>,
    pub(crate) fp: u64,
    is_all: bool,
}

impl Deref for SimpleBitset {
    type Target = SimpleBitsetInner;

    #[inline]
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl Default for SimpleBitset {
    fn default() -> Self { Self::zeros() }
}

// Global interning for SimpleBitset
// Global interning for SimpleBitset
struct Interner(HashMap<u64, Vec<Arc<SimpleBitsetInner>>>);

const NUM_SHARDS: usize = 64;

struct ShardedInterner {
    shards: Vec<Mutex<Interner>>,
}

impl ShardedInterner {
    fn new() -> Self {
        let mut shards = Vec::with_capacity(NUM_SHARDS);
        for _ in 0..NUM_SHARDS {
            shards.push(Mutex::new(Interner(HashMap::new())));
        }
        Self { shards }
    }

    fn get(&self, fp: u64) -> &Mutex<Interner> {
        &self.shards[(fp as usize) % NUM_SHARDS]
    }
}

static INTERNER: Lazy<ShardedInterner> = Lazy::new(ShardedInterner::new);

static ZEROS: Lazy<SimpleBitset> =
    Lazy::new(|| SimpleBitset(Arc::new(SimpleBitsetInner { rsb: RangeSetBlaze::new(), fp: FP_ZERO, is_all: false })));
static ALL: Lazy<SimpleBitset> = Lazy::new(|| {
    SimpleBitset(Arc::new(SimpleBitsetInner { rsb: universe_rsb(), fp: FP_ALL, is_all: true }))
});

type OpCache = LruCache<(usize, usize), SimpleBitset>;
const CACHE_SIZE: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(1024) }; // Per shard

struct ShardedCache {
    shards: Vec<Mutex<OpCache>>,
}

impl ShardedCache {
    fn new() -> Self {
        let mut shards = Vec::with_capacity(NUM_SHARDS);
        for _ in 0..NUM_SHARDS {
            shards.push(Mutex::new(LruCache::new(CACHE_SIZE)));
        }
        Self { shards }
    }

    fn get(&self, key: (usize, usize)) -> &Mutex<OpCache> {
        // Simple hash for the pair of pointers
        let h = (key.0 ^ key.1.rotate_left(32)); 
        &self.shards[h % NUM_SHARDS]
    }
}

static UNION_CACHE: Lazy<ShardedCache> = Lazy::new(ShardedCache::new);
static INTERSECTION_CACHE: Lazy<ShardedCache> = Lazy::new(ShardedCache::new);
static SUB_CACHE: Lazy<ShardedCache> = Lazy::new(ShardedCache::new);

pub(crate) const FP_ZERO: u64 = 0x9E37_79B9_7F4A_7C15;
const FP_ALL: u64 = 0xD6E8_FEB8_6659_FD93;
pub(crate) const FP_K1: u64 = 0xC2B2_AE3D_27D4_EB4F;
pub(crate) const FP_K2: u64 = 0x1656_67B1_F3E1_5A6D;
const FP_K3: u64 = 0x9E37_79B1_1F37_9B97;

#[inline]
pub(crate) fn mix3(a: u64, b: u64, c: u64) -> u64 {
    let mut x = a ^ b.wrapping_mul(FP_K1);
    x = x.rotate_left(27) ^ c.wrapping_mul(FP_K2);
    x ^= x >> 33;
    x = x.wrapping_mul(FP_K3);
    x ^ (x >> 29)
}

#[inline]
fn calc_is_all_and_fp(rsb: &RangeSetBlaze<usize>) -> (bool, u64) {
    let mut it = rsb.ranges();
    if let Some(first) = it.next() {
        if *first.start() == 0 && *first.end() == usize::MAX && it.next().is_none() {
            return (true, FP_ALL);
        }
        let mut fp = mix3(FP_ZERO, *first.start() as u64, (*first.end() as u64).wrapping_mul(FP_K1));
        for r in it {
            let s = *r.start() as u64;
            let e = *r.end() as u64;
            fp = mix3(fp, s.wrapping_mul(FP_K2), e.wrapping_mul(FP_K3));
        }
        (false, fp)
    } else {
        (false, FP_ZERO)
    }
}

fn intern(rsb: RangeSetBlaze<usize>) -> SimpleBitset {
    if rsb.is_empty() {
        return ZEROS.clone();
    }
    let (is_all, fp) = calc_is_all_and_fp(&rsb);
    if is_all {
        return ALL.clone();
    }

    let mut interner = INTERNER.get(fp).lock().unwrap();
    let candidates = interner.0.entry(fp).or_default();
    for candidate in candidates.iter() {
        if candidate.rsb == rsb {
            return SimpleBitset(candidate.clone());
        }
    }

    let new_inner = Arc::new(SimpleBitsetInner { rsb, fp, is_all });
    candidates.push(new_inner.clone());
    SimpleBitset(new_inner)
}

#[inline]
fn universe_rsb() -> RangeSetBlaze<usize> { RangeSetBlaze::from_iter([0usize..=usize::MAX]) }

#[inline]
fn complement_rsb(rsb: &RangeSetBlaze<usize>) -> RangeSetBlaze<usize> { &universe_rsb() - rsb }

impl SimpleBitset {
    pub fn zeros() -> Self { ZEROS.clone() }

    pub fn ones(len: usize) -> Self {
        intern(RangeSetBlaze::from_iter([0..=len - 1]))
    }

    pub fn all() -> Self { ALL.clone() }

    pub fn from_item(item: usize) -> Self { intern(RangeSetBlaze::from_iter([item])) }

    pub fn from_ranges(ranges: &[(usize, usize)]) -> Self {
        let rsb = RangeSetBlaze::from_iter(ranges.iter().map(|&(s, e)| s..=e));
        intern(rsb)
    }

    pub fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self { intern(rsb) }
}

impl SimpleBitsetInner {
    pub fn len(&self) -> usize { self.rsb.len().try_into().unwrap_or(usize::MAX) }

    pub fn is_empty(&self) -> bool { self.rsb.is_empty() }

    #[inline]
    pub fn is_all_fast(&self) -> bool { self.is_all }

    pub fn is_disjoint(&self, other: &SimpleBitset) -> bool {
        if self.is_empty() || other.is_empty() {
            return true;
        }
        if self.is_all_fast() || other.is_all_fast() {
            return false;
        }
        (&self.rsb & &other.rsb).is_empty()
    }

    pub fn contains(&self, index: usize) -> bool { self.rsb.contains(index) }

    pub fn max_item(&self) -> Option<usize> { self.rsb.iter().last() }

    pub fn iter_up_to(&self, max: usize) -> impl Iterator<Item = usize> {
        (&self.rsb & &RangeSetBlaze::from_iter([0..=max])).into_iter()
    }

}

impl SimpleBitset {
    pub fn is_subset_of(&self, rhs: &SimpleBitset) -> bool { (self & rhs) == *self }

    pub fn complement(&self) -> SimpleBitset {
        if self.is_empty() {
            return SimpleBitset::all();
        }
        if self.is_all_fast() {
            return SimpleBitset::zeros();
        }
        let rsb = complement_rsb(&self.rsb);
        intern(rsb)
    }

    pub fn insert(&mut self, item: usize) {
        if self.is_all_fast() {
            return;
        }
        let mut rsb = self.rsb.clone();
        rsb.insert(item);
        *self = intern(rsb);
    }

    pub fn add(&mut self, item: usize) { self.insert(item); }

    pub fn remove(&mut self, item: usize) {
        if self.contains(item) {
            let mut rsb = self.rsb.clone();
            rsb.remove(item);
            *self = intern(rsb);
        }
    }

    pub fn set(&mut self, item: usize, value: bool) {
        if value {
            self.insert(item);
        } else {
            self.remove(item);
        }
    }

    pub fn clear(&mut self) { *self = Self::zeros(); }

    pub fn clip_to_range(&mut self, min: usize, max: usize) {
        if self.is_empty() {
            return;
        }
        let clip_rsb = RangeSetBlaze::from_iter([min..=max]);
        let rsb = &self.rsb & &clip_rsb;
        *self = intern(rsb);
    }

    pub fn clip_min(&mut self, min: usize) { self.clip_to_range(min, usize::MAX); }

    pub fn clip_max(&mut self, max: usize) { self.clip_to_range(0, max); }
}

impl Hash for SimpleBitset {
    fn hash<H: Hasher>(&self, state: &mut H) { state.write_u64(self.fp); }
}

impl Debug for SimpleBitset {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_all_fast() {
            write!(f, "SimpleBitset(ALL)")
        } else {
            Debug::fmt(&self.0.rsb, f)
        }
    }
}

impl Display for SimpleBitset {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_all_fast() {
            return write!(f, "ALL");
        }
        write!(f, "[")?;
        let mut ranges = self.rsb.ranges().peekable();
        while let Some(range) = ranges.next() {
            if range.start() == range.end() {
                write!(f, "{}", range.start())?;
            } else {
                write!(f, "{}..={}", range.start(), range.end())?;
            }
            if ranges.peek().is_some() {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

impl FromIterator<usize> for SimpleBitset {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        intern(RangeSetBlaze::from_iter(iter))
    }
}

impl FromIterator<std::ops::RangeInclusive<usize>> for SimpleBitset {
    fn from_iter<T: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(iter: T) -> Self {
        intern(RangeSetBlaze::from_iter(iter))
    }
}

impl From<RangeSetBlaze<usize>> for SimpleBitset {
    fn from(rsb: RangeSetBlaze<usize>) -> Self { intern(rsb) }
}

impl From<HybridBitset> for SimpleBitset {
    fn from(hb: HybridBitset) -> Self { intern(hb.inner.as_ref().clone()) }
}

impl From<SimpleBitset> for HybridBitset {
    fn from(sb: SimpleBitset) -> Self { HybridBitset::from(sb.rsb.clone()) }
}

impl<'a> BitAnd<&'a SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn bitand(self, rhs: &'a SimpleBitset) -> Self::Output {
        if Arc::ptr_eq(&self.0, &rhs.0) {
            return self.clone();
        }
        if self.is_empty() || rhs.is_empty() {
            return SimpleBitset::zeros();
        }
        if self.is_all_fast() {
            return rhs.clone();
        }
        if rhs.is_all_fast() {
            return self.clone();
        }

        let p1 = Arc::as_ptr(&self.0) as usize;
        let p2 = Arc::as_ptr(&rhs.0) as usize;
        let key = if p1 < p2 { (p1, p2) } else { (p2, p1) };

        let mut cache = INTERSECTION_CACHE.get(key).lock().unwrap();
        if let Some(result) = cache.get(&key) {
            return result.clone();
        }

        let result = intern(&self.rsb & &rhs.rsb);
        cache.put(key, result.clone());
        result
    }
}

impl<'a> BitOr<&'a SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn bitor(self, rhs: &'a SimpleBitset) -> Self::Output {
        if Arc::ptr_eq(&self.0, &rhs.0) {
            return self.clone();
        }
        if self.is_all_fast() || rhs.is_all_fast() {
            return SimpleBitset::all();
        }
        if self.is_empty() {
            return rhs.clone();
        }
        if rhs.is_empty() {
            return self.clone();
        }

        let p1 = Arc::as_ptr(&self.0) as usize;
        let p2 = Arc::as_ptr(&rhs.0) as usize;
        let key = if p1 < p2 { (p1, p2) } else { (p2, p1) };

        let mut cache = UNION_CACHE.get(key).lock().unwrap();
        if let Some(result) = cache.get(&key) {
            return result.clone();
        }

        let result = intern(&self.rsb | &rhs.rsb);
        cache.put(key, result.clone());
        result
    }
}

impl BitAndAssign<&SimpleBitset> for SimpleBitset {
    fn bitand_assign(&mut self, rhs: &SimpleBitset) {
        *self = &*self & rhs;
    }
}

impl BitOrAssign<&SimpleBitset> for SimpleBitset {
    fn bitor_assign(&mut self, rhs: &SimpleBitset) {
        *self = &*self | rhs;
    }
}

impl BitAnd<SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn bitand(self, rhs: SimpleBitset) -> Self::Output { (&self) & (&rhs) }
}

impl BitOr<SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn bitor(self, rhs: SimpleBitset) -> Self::Output { (&self) | (&rhs) }
}

impl<'a> BitAnd<&'a SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn bitand(self, rhs: &'a SimpleBitset) -> Self::Output { (&self) & rhs }
}

impl<'a> BitOr<&'a SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn bitor(self, rhs: &'a SimpleBitset) -> Self::Output { (&self) | rhs }
}

impl<'a> BitAnd<SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn bitand(self, rhs: SimpleBitset) -> Self::Output { self & (&rhs) }
}

impl<'a> BitOr<SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn bitor(self, rhs: SimpleBitset) -> Self::Output { self | (&rhs) }
}

impl SubAssign<&SimpleBitset> for SimpleBitset {
    fn sub_assign(&mut self, rhs: &SimpleBitset) {
        *self = &*self - rhs;
    }
}

impl SubAssign<SimpleBitset> for SimpleBitset {
    fn sub_assign(&mut self, rhs: SimpleBitset) { *self -= &rhs }
}

impl Sub<SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn sub(self, rhs: SimpleBitset) -> Self::Output { (&self) - (&rhs) }
}

impl<'a> Sub<&'a SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn sub(self, rhs: &'a SimpleBitset) -> Self::Output {
        if Arc::ptr_eq(&self.0, &rhs.0) {
            return SimpleBitset::zeros();
        }
        if self.is_empty() || rhs.is_empty() {
            return self.clone();
        }
        if rhs.is_all_fast() {
            return SimpleBitset::zeros();
        }

        let p1 = Arc::as_ptr(&self.0) as usize;
        let p2 = Arc::as_ptr(&rhs.0) as usize;
        let key = (p1, p2);

        let mut cache = SUB_CACHE.get(key).lock().unwrap();
        if let Some(result) = cache.get(&key) {
            return result.clone();
        }

        let result = if self.is_all_fast() {
            rhs.complement()
        } else {
            intern(&self.rsb - &rhs.rsb)
        };
        cache.put(key, result.clone());
        result
    }
}

impl Not for SimpleBitset {
    type Output = SimpleBitset;
    fn not(self) -> Self::Output {
        if self.is_empty() {
            return SimpleBitset::all();
        }
        if self.is_all_fast() {
            return SimpleBitset::zeros();
        }
        intern(complement_rsb(&self.rsb))
    }
}

impl Not for &SimpleBitset {
    type Output = SimpleBitset;
    fn not(self) -> Self::Output {
        if self.is_empty() {
            return SimpleBitset::all();
        }
        if self.is_all_fast() {
            return SimpleBitset::zeros();
        }
        intern(complement_rsb(&self.rsb))
    }
}

impl Serialize for SimpleBitset {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        #[derive(Serialize)]
        #[serde(untagged)]
        enum Repr<'a> {
            All(&'a str),
            Ranges(Vec<RangeRepr>),
        }

        #[derive(Serialize)]
        #[serde(untagged)]
        enum RangeRepr {
            Single(usize),
            Range((usize, usize)),
        }

        if self.is_all {
            Repr::All("ALL").serialize(serializer)
        } else {
            let ranges: Vec<RangeRepr> = self
                .rsb
                .ranges()
                .map(|r| {
                    if r.start() == r.end() {
                        RangeRepr::Single(*r.start())
                    } else {
                        RangeRepr::Range((*r.start(), *r.end()))
                    }
                })
                .collect();
            Repr::Ranges(ranges).serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for SimpleBitset {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        use serde::de;

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            All(String),
            Ranges(Vec<RangeRepr>),
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum RangeRepr {
            Single(usize),
            Range((usize, usize)),
        }

        match Repr::deserialize(deserializer)? {
            Repr::All(s) => {
                if s == "ALL" {
                    Ok(SimpleBitset::all())
                } else {
                    Err(de::Error::custom("expected string 'ALL' for all-bitset"))
                }
            }
            Repr::Ranges(ranges) => {
                let rsb = RangeSetBlaze::from_iter(ranges.into_iter().map(|rr| match rr {
                    RangeRepr::Single(i) => i..=i,
                    RangeRepr::Range((s, e)) => s..=e,
                }));
                Ok(intern(rsb))
            }
        }
    }
}
