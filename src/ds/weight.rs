#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use range_set_blaze::{CheckSortedDisjoint, RangeMapBlaze, RangeSetBlaze, SortedDisjointMap};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Weight(pub RangeMapBlaze<u32, Arc<RangeSetBlaze<u32>>>);

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

fn shared_rangeset(tokens: RangeSetBlaze<u32>) -> Arc<RangeSetBlaze<u32>> {
    Arc::new(tokens)
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

fn rangeset_to_string_with_names(
    set: &RangeSetBlaze<u32>,
    names: &BTreeMap<u32, String>,
) -> String {
    let expanded: Vec<u32> = set.iter().collect();
    if expanded.len() <= WEIGHT_NAME_EXPAND_LIMIT && expanded.iter().all(|id| names.contains_key(id)) {
        return format!(
            "{{{}}}",
            expanded
                .iter()
                .map(|id| names.get(id).cloned().unwrap_or_else(|| id.to_string()))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    rangeset_to_string(set)
}

fn compress_expanded(expanded: &BTreeMap<u32, RangeSetBlaze<u32>>) -> Weight {
    let mut map = RangeMapBlaze::new();
    let mut current_start: Option<u32> = None;
    let mut current_end = 0u32;
    let mut current_tokens = RangeSetBlaze::new();

    let mut flush = |map: &mut RangeMapBlaze<u32, Arc<RangeSetBlaze<u32>>>,
                     current_start: &mut Option<u32>,
                     current_end: &mut u32,
                     current_tokens: &mut RangeSetBlaze<u32>| {
        if let Some(start) = *current_start {
            let tokens = std::mem::replace(current_tokens, RangeSetBlaze::new());
            map.extend_simple(std::iter::once((start..=*current_end, shared_rangeset(tokens))));
        }
        *current_start = None;
    };

    for (&tsid, tokens) in expanded {
        match current_start {
            Some(start)
                if current_end.checked_add(1) == Some(tsid) && *tokens == current_tokens =>
            {
                let _ = start;
                current_end = tsid;
            }
            _ => {
                flush(&mut map, &mut current_start, &mut current_end, &mut current_tokens);
                current_start = Some(tsid);
                current_end = tsid;
                current_tokens = tokens.clone();
            }
        }
    }

    flush(&mut map, &mut current_start, &mut current_end, &mut current_tokens);
    Weight(map)
}

#[derive(Clone)]
struct WeightRangeEntry {
    start: u32,
    end: u32,
    tokens: Arc<RangeSetBlaze<u32>>,
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
        Option<&Arc<RangeSetBlaze<u32>>>,
        Option<&Arc<RangeSetBlaze<u32>>>,
    ) -> Option<Arc<RangeSetBlaze<u32>>>,
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

    let mut map = RangeMapBlaze::new();
    let mut pending_start = None;
    let mut pending_end = 0u32;
    let mut pending_tokens = shared_rangeset(RangeSetBlaze::new());

    for i in 0..(len - 1) {
        let start = boundaries[i] as u32;
        let end = (boundaries[i + 1] - 1) as u32;
        let left_tokens = (left.start <= start && start <= left.end).then_some(&left.tokens);
        let right_tokens = (right.start <= start && start <= right.end).then_some(&right.tokens);
        let Some(tokens) = combine(left_tokens, right_tokens) else {
            flush_compact_range(&mut map, &mut pending_start, &mut pending_end, &mut pending_tokens);
            continue;
        };
        push_compact_range(
            &mut map,
            &mut pending_start,
            &mut pending_end,
            &mut pending_tokens,
            start,
            end,
            tokens,
        );
    }

    flush_compact_range(&mut map, &mut pending_start, &mut pending_end, &mut pending_tokens);
    Weight(map)
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
) -> Option<&'a Arc<RangeSetBlaze<u32>>> {
    while *index < entries.len() && entries[*index].end < start {
        *index += 1;
    }
    entries.get(*index).and_then(|entry| {
        (entry.start <= start && start <= entry.end).then_some(&entry.tokens)
    })
}

fn push_compact_range(
    map: &mut RangeMapBlaze<u32, Arc<RangeSetBlaze<u32>>>,
    pending_start: &mut Option<u32>,
    pending_end: &mut u32,
    pending_tokens: &mut Arc<RangeSetBlaze<u32>>,
    start: u32,
    end: u32,
    tokens: Arc<RangeSetBlaze<u32>>,
) {
    match *pending_start {
        Some(existing_start)
            if pending_end.checked_add(1) == Some(start) && *pending_tokens == tokens =>
        {
            let _ = existing_start;
            *pending_end = end;
        }
        _ => {
            flush_compact_range(map, pending_start, pending_end, pending_tokens);
            *pending_start = Some(start);
            *pending_end = end;
            *pending_tokens = tokens;
        }
    }
}

fn flush_compact_range(
    map: &mut RangeMapBlaze<u32, Arc<RangeSetBlaze<u32>>>,
    pending_start: &mut Option<u32>,
    pending_end: &mut u32,
    pending_tokens: &mut Arc<RangeSetBlaze<u32>>,
) {
    if let Some(start) = *pending_start {
        let tokens = std::mem::replace(pending_tokens, shared_rangeset(RangeSetBlaze::new()));
        map.extend_simple(std::iter::once((start..=*pending_end, tokens)));
        *pending_start = None;
    }
}

fn combine_compact_entries<F>(left: &Weight, right: &Weight, mut combine: F) -> Weight
where
    F: FnMut(
        Option<&Arc<RangeSetBlaze<u32>>>,
        Option<&Arc<RangeSetBlaze<u32>>>,
    ) -> Option<Arc<RangeSetBlaze<u32>>>,
{
    let left_entries = compact_entries(left);
    let right_entries = compact_entries(right);
    let boundaries = combined_boundaries(&left_entries, &right_entries);
    if boundaries.len() < 2 {
        return Weight::empty();
    }

    let mut left_index = 0usize;
    let mut right_index = 0usize;
    let mut map = RangeMapBlaze::new();
    let mut pending_start = None;
    let mut pending_end = 0u32;
    let mut pending_tokens = shared_rangeset(RangeSetBlaze::new());

    for window in boundaries.windows(2) {
        let start = window[0] as u32;
        let end = (window[1] - 1) as u32;
        let left_tokens = active_tokens(&left_entries, &mut left_index, start);
        let right_tokens = active_tokens(&right_entries, &mut right_index, start);
        let Some(tokens) = combine(left_tokens, right_tokens) else {
            flush_compact_range(&mut map, &mut pending_start, &mut pending_end, &mut pending_tokens);
            continue;
        };
        push_compact_range(
            &mut map,
            &mut pending_start,
            &mut pending_end,
            &mut pending_tokens,
            start,
            end,
            tokens,
        );
    }

    flush_compact_range(&mut map, &mut pending_start, &mut pending_end, &mut pending_tokens);
    Weight(map)
}

fn intersect_weights(left: &Weight, right: &Weight) -> Weight {
    let mut left_iter = left.0.range_values();
    let mut right_iter = right.0.range_values();
    let mut left_entry = left_iter.next();
    let mut right_entry = right_iter.next();

    let mut map = RangeMapBlaze::new();
    let mut pending_start = None;
    let mut pending_end = 0u32;
    let mut pending_tokens = shared_rangeset(RangeSetBlaze::new());

    while let (Some((left_range, left_tokens)), Some((right_range, right_tokens))) = (left_entry, right_entry) {
        let start = (*left_range.start()).max(*right_range.start());
        let end = (*left_range.end()).min(*right_range.end());

        if start <= end {
            let left_tokens_ref = left_tokens.as_ref();
            let right_tokens_ref = right_tokens.as_ref();
            let tokens = if Arc::ptr_eq(left_tokens, right_tokens)
                || left_tokens_ref == right_tokens_ref
            {
                Some(Arc::clone(left_tokens))
            } else {
                let overlap = left_tokens_ref & right_tokens_ref;
                if overlap.is_empty() {
                    None
                } else {
                    Some(shared_rangeset(overlap))
                }
            };

            if let Some(tokens) = tokens {
                push_compact_range(
                    &mut map,
                    &mut pending_start,
                    &mut pending_end,
                    &mut pending_tokens,
                    start,
                    end,
                    tokens,
                );
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

    flush_compact_range(&mut map, &mut pending_start, &mut pending_end, &mut pending_tokens);
    Weight(map)
}

fn intersect_single_entry_with_weight(single: &WeightRangeEntry, other: &Weight) -> Weight {
    let mut map = RangeMapBlaze::new();
    let mut pending_start = None;
    let mut pending_end = 0u32;
    let mut pending_tokens = shared_rangeset(RangeSetBlaze::new());
    let mut overlap_cache: SmallVec<[(
        *const RangeSetBlaze<u32>,
        Option<Arc<RangeSetBlaze<u32>>>,
    ); 8]> = SmallVec::new();

    let bounds = CheckSortedDisjoint::new([single.start..=single.end]);
    for (range, other_tokens) in other.0.range_values().map_and_set_intersection(bounds) {
        let start = *range.start();
        let end = *range.end();

        let tokens = if Arc::ptr_eq(&single.tokens, other_tokens)
            || single.tokens.as_ref() == other_tokens.as_ref()
        {
            Arc::clone(&single.tokens)
        } else {
            let cache_key = Arc::as_ptr(other_tokens);
            if let Some((_, cached)) = overlap_cache.iter().find(|(ptr, _)| *ptr == cache_key) {
                let Some(cached_tokens) = cached else {
                    continue;
                };
                Arc::clone(cached_tokens)
            } else {
                let overlap = single.tokens.as_ref() & other_tokens.as_ref();
                if overlap.is_empty() {
                    overlap_cache.push((cache_key, None));
                    continue;
                }
                let overlap_tokens = shared_rangeset(overlap);
                overlap_cache.push((cache_key, Some(Arc::clone(&overlap_tokens))));
                overlap_tokens
            }
        };

        push_compact_range(
            &mut map,
            &mut pending_start,
            &mut pending_end,
            &mut pending_tokens,
            start,
            end,
            tokens,
        );
    }

    flush_compact_range(&mut map, &mut pending_start, &mut pending_end, &mut pending_tokens);
    Weight(map)
}

fn range_map_entries(weight: &Weight) -> Vec<(std::ops::RangeInclusive<u32>, RangeSetBlaze<u32>)> {
    weight
        .0
        .range_values()
    .map(|(range, tokens)| (range, tokens.as_ref().clone()))
        .collect()
}

#[derive(Debug, Clone, Default)]
pub(crate) struct WeightBuilder {
    is_full: bool,
    expanded: BTreeMap<u32, RangeSetBlaze<u32>>,
}

impl WeightBuilder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn is_full(&self) -> bool {
        self.is_full
    }

    pub(crate) fn union_weight(&mut self, weight: &Weight) {
        if self.is_full || weight.is_empty() {
            return;
        }
        if weight.is_full() {
            self.is_full = true;
            self.expanded.clear();
            return;
        }

        for (range, tokens_arc) in weight.0.range_values() {
            for tsid in range {
                match self.expanded.entry(tsid) {
                    std::collections::btree_map::Entry::Occupied(mut e) => {
                        let existing = e.get_mut();
                        // Skip union if token sets are already identical
                        if existing != tokens_arc.as_ref() {
                            *existing |= tokens_arc.as_ref();
                        }
                    }
                    std::collections::btree_map::Entry::Vacant(e) => {
                        e.insert(tokens_arc.as_ref().clone());
                    }
                }
            }
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        !self.is_full && self.expanded.is_empty()
    }

    pub(crate) fn build(self) -> Weight {
        if self.is_full {
            Weight::all()
        } else if self.expanded.is_empty() {
            Weight::empty()
        } else {
            compress_expanded(&self.expanded)
        }
    }
}

impl Weight {
    pub(crate) fn compact_entries(&self) -> Option<Vec<(u32, u32, Arc<RangeSetBlaze<u32>>)>> {
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
        Self(RangeMapBlaze::new())
    }

    pub fn all() -> Self {
        let mut map = RangeMapBlaze::new();
        map.extend_simple(std::iter::once((
            WEIGHT_ALL_SENTINEL..=WEIGHT_ALL_SENTINEL,
            shared_rangeset(sentinel_token_set()),
        )));
        Self(map)
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
        if let (Some(left), Some(right)) = (single_compact_entry(self), single_compact_entry(other)) {
            return combine_single_entries(&left, &right, |left, right| match (left, right) {
                (Some(left_tokens), Some(right_tokens)) => {
                    if Arc::ptr_eq(left_tokens, right_tokens) || left_tokens.as_ref() == right_tokens.as_ref() {
                        Some(Arc::clone(left_tokens))
                    } else {
                        Some(shared_rangeset(left_tokens.as_ref().clone() | right_tokens.as_ref().clone()))
                    }
                }
                (Some(tokens), None) | (None, Some(tokens)) => Some(Arc::clone(tokens)),
                (None, None) => None,
            });
        }
        combine_compact_entries(self, other, |left, right| match (left, right) {
            (Some(left_tokens), Some(right_tokens)) => {
                if Arc::ptr_eq(left_tokens, right_tokens) || left_tokens.as_ref() == right_tokens.as_ref() {
                    Some(Arc::clone(left_tokens))
                } else {
                    Some(shared_rangeset(left_tokens.as_ref().clone() | right_tokens.as_ref().clone()))
                }
            }
            (Some(tokens), None) | (None, Some(tokens)) => Some(Arc::clone(tokens)),
            (None, None) => None,
        })
    }

    pub fn union_all<'a>(weights: impl IntoIterator<Item = &'a Self>) -> Self {
        let mut expanded: BTreeMap<u32, RangeSetBlaze<u32>> = BTreeMap::new();

        for weight in weights {
            if weight.is_full() {
                return Self::all();
            }
            if weight.is_empty() {
                continue;
            }

            for (range, tokens_arc) in weight.0.range_values() {
                for tsid in range {
                    match expanded.entry(tsid) {
                        std::collections::btree_map::Entry::Occupied(mut e) => {
                            let existing = e.get_mut();
                            if existing != tokens_arc.as_ref() {
                                *existing |= tokens_arc.as_ref();
                            }
                        }
                        std::collections::btree_map::Entry::Vacant(e) => {
                            e.insert(tokens_arc.as_ref().clone());
                        }
                    }
                }
            }
        }

        if expanded.is_empty() {
            Self::empty()
        } else {
            compress_expanded(&expanded)
        }
    }

    pub fn intersection(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::empty();
        }
        if self.is_full() {
            return other.clone();
        }
        if other.is_full() {
            return self.clone();
        }
        if let (Some(left), Some(right)) = (single_compact_entry(self), single_compact_entry(other)) {
            return combine_single_entries(&left, &right, |left, right| match (left, right) {
                (Some(left_tokens), Some(right_tokens)) => {
                    if Arc::ptr_eq(left_tokens, right_tokens) || left_tokens.as_ref() == right_tokens.as_ref() {
                        Some(Arc::clone(left_tokens))
                    } else {
                        let tokens = left_tokens.as_ref() & right_tokens.as_ref();
                        (!tokens.is_empty()).then(|| shared_rangeset(tokens))
                    }
                }
                _ => None,
            });
        }
        if let Some(single) = single_compact_entry(self) {
            return intersect_single_entry_with_weight(&single, other);
        }
        if let Some(single) = single_compact_entry(other) {
            return intersect_single_entry_with_weight(&single, self);
        }
        intersect_weights(self, other)
    }

    pub fn difference(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_full() {
            return Self::empty();
        }
        if other.is_empty() {
            return self.clone();
        }
        if self.is_full() {
            // Cannot compute all \ other without an explicit universe.
            // Return all() as a safe over-approximation.  Callers that need
            // exact complements should use the dedicated complement() method
            // which returns empty() as a no-op sentinel instead.
            return Self::all();
        }
        combine_compact_entries(self, other, |left, right| match (left, right) {
            (Some(left_tokens), Some(right_tokens)) => {
                if Arc::ptr_eq(left_tokens, right_tokens) || left_tokens.as_ref() == right_tokens.as_ref() {
                    None
                } else {
                    let tokens = left_tokens.as_ref().clone() - right_tokens.as_ref().clone();
                    (!tokens.is_empty()).then(|| shared_rangeset(tokens))
                }
            }
            (Some(left_tokens), None) => Some(Arc::clone(left_tokens)),
            _ => None,
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

    pub fn divide(&self, other: &Self) -> Self {
        self.difference(other)
    }

    pub fn from_token_set_for_tsid(tsid: u32, tokens: RangeSetBlaze<u32>) -> Self {
        if tokens.is_empty() {
            return Self::empty();
        }
        let token_ranges: Vec<_> = tokens.ranges().collect();
        Self::from_compact_ranges(std::iter::once((tsid..=tsid, token_ranges)))
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
    ) -> Option<(u32, u32, Arc<RangeSetBlaze<u32>>)> {
        let entry = single_compact_entry(self)?;
        Some((entry.start, entry.end, entry.tokens))
    }

    pub(crate) fn intersect_single_parts(
        &self,
        start: u32,
        end: u32,
        tokens: &Arc<RangeSetBlaze<u32>>,
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
            Option<Arc<RangeSetBlaze<u32>>>,
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
        let mut new_map = RangeMapBlaze::new();
        for (tsid_range, tokens) in self.0.range_values() {
            let clipped = tokens.as_ref() & &clip;
            if !clipped.is_empty() {
                new_map.extend_simple(std::iter::once((tsid_range, Arc::new(clipped))));
            }
        }
        self.0 = new_map;
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

const WEIGHT_NAME_EXPAND_LIMIT: usize = 64;

pub struct WeightDisplayWithNames<'a> {
    weight: &'a Weight,
    tsid_names: &'a std::collections::BTreeMap<u32, String>,
    token_names: &'a std::collections::BTreeMap<u32, String>,
}

impl Weight {
    pub fn display_with_names<'a>(
        &'a self,
        tsid_names: &'a std::collections::BTreeMap<u32, String>,
        token_names: &'a std::collections::BTreeMap<u32, String>,
    ) -> WeightDisplayWithNames<'a> {
        WeightDisplayWithNames {
            weight: self,
            tsid_names,
            token_names,
        }
    }
}

impl std::fmt::Display for WeightDisplayWithNames<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.weight.is_empty() {
            return write!(f, "∅");
        }
        if self.weight.is_full() {
            return write!(f, "ALL");
        }

        let parts: Vec<String> = self
            .weight
            .0
            .range_values()
            .map(|(range, tokens)| {
                let expanded: Vec<u32> = range.clone().collect();
                let tsid = if expanded.len() <= WEIGHT_NAME_EXPAND_LIMIT
                    && expanded.iter().all(|id| self.tsid_names.contains_key(id))
                {
                    expanded
                        .iter()
                        .map(|id| self.tsid_names.get(id).cloned().unwrap_or_else(|| id.to_string()))
                        .collect::<Vec<_>>()
                        .join("|")
                } else if range.start() == range.end() {
                    self.tsid_names
                        .get(range.start())
                        .cloned()
                        .unwrap_or_else(|| range.start().to_string())
                } else {
                    format!("{}..={}", range.start(), range.end())
                };
                format!(
                    "{tsid}→{}",
                    rangeset_to_string_with_names(tokens.as_ref(), self.token_names)
                )
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
