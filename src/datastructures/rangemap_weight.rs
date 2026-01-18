use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use lru::LruCache;
use once_cell::sync::Lazy;
use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};

use crate::datastructures::abstract_weight::{current_num_tsids, normalize_num_tsids, WeightBackend};
use crate::datastructures::cache;
use crate::datastructures::hybrid_bitset::RangeSet;

const WEIGHT_OP_CACHE_CAPACITY: usize = 100_000;

static RANGEMAP_WEIGHT_INTERNER: Lazy<Mutex<HashSet<Arc<RangeMapWeight>>>> =
    Lazy::new(|| Mutex::new(HashSet::new()));
static RANGEMAP_OP_CACHE: Lazy<Mutex<LruCache<OpKey, Arc<RangeMapWeight>>>> = Lazy::new(|| {
    Mutex::new(LruCache::new(NonZeroUsize::new(WEIGHT_OP_CACHE_CAPACITY).unwrap()))
});
static RANGEMAP_OP_CACHE_INDEX: Lazy<Mutex<HashMap<usize, HashSet<OpKey>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static RANGEMAP_WEIGHT_PTRS: Lazy<Mutex<HashSet<usize>>> = Lazy::new(|| Mutex::new(HashSet::new()));
static FULL_TSIDS_CACHE: Lazy<Mutex<HashMap<usize, RangeSet>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

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
    let ptr = Arc::as_ptr(weight) as usize;
    {
        let ptrs = RANGEMAP_WEIGHT_PTRS.lock().unwrap();
        if ptrs.contains(&ptr) {
            return true;
        }
    }
    let interner = RANGEMAP_WEIGHT_INTERNER.lock().unwrap();
    let found = interner.iter().any(|arc| Arc::as_ptr(arc) as usize == ptr);
    if found {
        RANGEMAP_WEIGHT_PTRS.lock().unwrap().insert(ptr);
    }
    found
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

pub fn intern_rangemap(weight: RangeMapWeight) -> Arc<RangeMapWeight> {
    let mut interner = RANGEMAP_WEIGHT_INTERNER.lock().unwrap();
    if let Some(existing) = interner.get(&weight) {
        let ptr = Arc::as_ptr(existing) as usize;
        RANGEMAP_WEIGHT_PTRS.lock().unwrap().insert(ptr);
        return existing.clone();
    }
    let arc = Arc::new(weight);
    let ptr = Arc::as_ptr(&arc) as usize;
    invalidate_rangemap_op_cache_for_ptr(ptr);
    interner.insert(arc.clone());
    RANGEMAP_WEIGHT_PTRS.lock().unwrap().insert(ptr);
    arc
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeMapWeight {
    /// Maps token_id -> set of tsid values (stored as ranges over token_id).
    pub(crate) map: RangeMapBlaze<usize, RangeSet>,
    pub(crate) num_tsids: usize,
}

impl RangeMapWeight {
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

    pub(crate) fn new(num_tsids: usize) -> Self {
        Self {
            map: RangeMapBlaze::new(),
            num_tsids: normalize_num_tsids(num_tsids),
        }
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

        Self { map: out, num_tsids }
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
            Self::union_asymmetric(&smaller.map, &larger.map)
        } else {
            Self::merge_maps(&self.map, &other.map, |left, right| match (left, right) {
                (Some(a), Some(b)) => a | b,
                (Some(a), None) => a.clone(),
                (None, Some(b)) => b.clone(),
                (None, None) => RangeSet::zeros(),
            })
        };
        Self { map, num_tsids: self.num_tsids() }
    }

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

        Self { map, num_tsids: self.num_tsids() }
    }

    fn difference_non_negated(&self, other: &Self) -> Self {
        let map = Self::merge_maps(&self.map, &other.map, |left, right| match (left, right) {
            (Some(a), Some(b)) => a - b,
            (Some(a), None) => a.clone(),
            _ => RangeSet::zeros(),
        });
        Self {
            map,
            num_tsids: self.num_tsids(),
        }
    }

    pub(crate) fn divide(&self, other: &Self) -> Self {
        assert_eq!(self.num_tsids(), other.num_tsids(), "RangeMapWeight num_tsids mismatch");
        crate::datastructures::hybrid_bitset::PROF_COUNT_DIVIDE.fetch_add(
            1,
            std::sync::atomic::Ordering::Relaxed,
        );
        let start = std::time::Instant::now();
        let num_tsids = self.num_tsids();
        let full_tsids = Self::full_tsids(num_tsids);

        let mut left_iter = self.map.range_values();
        let mut right_iter = other.map.range_values();
        let mut left = left_iter.next();
        let mut right = right_iter.next();

        if left.is_none() && right.is_none() {
            let result = Self {
                map: RangeMapBlaze::new(),
                num_tsids,
            };
            crate::datastructures::hybrid_bitset::PROF_TIME_DIVIDE.fetch_add(
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

            if let Some(rv) = right_val {
                let ptr = Arc::as_ptr(&rv.inner);
                if right_ptr != Some(ptr) {
                    right_comp = Some(&full_tsids - rv);
                    right_ptr = Some(ptr);
                }
            } else {
                right_comp = None;
                right_ptr = None;
            }

            let combined = match (left_val, right_val) {
                (Some(a), Some(_)) => {
                    let comp = right_comp.as_ref().expect("missing right complement");
                    a | comp
                }
                (Some(a), None) => a.clone(),
                (None, Some(_)) => right_comp
                    .as_ref()
                    .expect("missing right complement")
                    .clone(),
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

        let result = Self {
            map: out,
            num_tsids,
        };
        crate::datastructures::hybrid_bitset::PROF_TIME_DIVIDE.fetch_add(
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
        self.num_tsids.hash(state);
        for (token_range, tsid_set) in self.map.range_values() {
            token_range.start().hash(state);
            token_range.end().hash(state);
            for tsid_range in tsid_set.ranges() {
                tsid_range.start().hash(state);
                tsid_range.end().hash(state);
            }
        }
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
            return Self { map, num_tsids };
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

        Self { map, num_tsids }
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
        Self { map, num_tsids }
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
        if Arc::ptr_eq(self, other) {
            return self.clone();
        }
        if let Some(hit) = get_op_cache(cache::BinOp::Or, self, other) {
            return hit;
        }
        let out = WeightBackend::union(self.as_ref(), other.as_ref());
        let out = intern_rangemap(out);
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
