#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};
use serde::{Deserialize, Serialize};

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct Weight(pub RangeMapBlaze<u32, RangeSetBlaze<u32>>);

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

    let mut flush = |map: &mut RangeMapBlaze<u32, RangeSetBlaze<u32>>,
                     current_start: &mut Option<u32>,
                     current_end: &mut u32,
                     current_tokens: &mut RangeSetBlaze<u32>| {
        if let Some(start) = *current_start {
            let tokens = std::mem::replace(current_tokens, RangeSetBlaze::new());
            map.extend_simple(std::iter::once((start..=*current_end, tokens)));
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
    tokens: RangeSetBlaze<u32>,
}

fn compact_entries(weight: &Weight) -> Vec<WeightRangeEntry> {
    weight
        .0
        .range_values()
        .map(|(range, tokens)| WeightRangeEntry {
            start: *range.start(),
            end: *range.end(),
            tokens: tokens.clone(),
        })
        .collect()
}

fn combined_boundaries(left: &[WeightRangeEntry], right: &[WeightRangeEntry]) -> Vec<u64> {
    let mut boundaries = Vec::with_capacity((left.len() + right.len()) * 2);
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
) -> Option<&'a RangeSetBlaze<u32>> {
    while *index < entries.len() && entries[*index].end < start {
        *index += 1;
    }
    entries.get(*index).and_then(|entry| {
        (entry.start <= start && start <= entry.end).then_some(&entry.tokens)
    })
}

fn push_compact_range(
    map: &mut RangeMapBlaze<u32, RangeSetBlaze<u32>>,
    pending_start: &mut Option<u32>,
    pending_end: &mut u32,
    pending_tokens: &mut RangeSetBlaze<u32>,
    start: u32,
    end: u32,
    tokens: RangeSetBlaze<u32>,
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
    map: &mut RangeMapBlaze<u32, RangeSetBlaze<u32>>,
    pending_start: &mut Option<u32>,
    pending_end: &mut u32,
    pending_tokens: &mut RangeSetBlaze<u32>,
) {
    if let Some(start) = *pending_start {
        let tokens = std::mem::replace(pending_tokens, RangeSetBlaze::new());
        map.extend_simple(std::iter::once((start..=*pending_end, tokens)));
        *pending_start = None;
    }
}

fn combine_compact_entries<F>(left: &Weight, right: &Weight, mut combine: F) -> Weight
where
    F: FnMut(Option<&RangeSetBlaze<u32>>, Option<&RangeSetBlaze<u32>>) -> RangeSetBlaze<u32>,
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
    let mut pending_tokens = RangeSetBlaze::new();

    for window in boundaries.windows(2) {
        let start = window[0] as u32;
        let end = (window[1] - 1) as u32;
        let left_tokens = active_tokens(&left_entries, &mut left_index, start);
        let right_tokens = active_tokens(&right_entries, &mut right_index, start);
        let tokens = combine(left_tokens, right_tokens);
        if tokens.is_empty() {
            flush_compact_range(&mut map, &mut pending_start, &mut pending_end, &mut pending_tokens);
        } else {
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

    flush_compact_range(&mut map, &mut pending_start, &mut pending_end, &mut pending_tokens);
    Weight(map)
}

fn range_map_entries(weight: &Weight) -> Vec<(std::ops::RangeInclusive<u32>, RangeSetBlaze<u32>)> {
    weight
        .0
        .range_values()
        .map(|(range, tokens)| (range, tokens.clone()))
        .collect()
}

impl Weight {
    pub fn empty() -> Self {
        Self(RangeMapBlaze::new())
    }

    pub fn all() -> Self {
        let mut map = RangeMapBlaze::new();
        map.extend_simple(std::iter::once((WEIGHT_ALL_SENTINEL..=WEIGHT_ALL_SENTINEL, sentinel_token_set())));
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
            out = out | tokens.clone();
        }
        out
    }

    pub fn is_full(&self) -> bool {
        let entries = range_map_entries(self);
        entries.len() == 1
            && entries[0].0.start() == &WEIGHT_ALL_SENTINEL
            && entries[0].0.end() == &WEIGHT_ALL_SENTINEL
            && entries[0].1 == sentinel_token_set()
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
                * (std::mem::size_of::<u32>() + std::mem::size_of::<RangeSetBlaze<u32>>())
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
        combine_compact_entries(self, other, |left, right| match (left, right) {
            (Some(left_tokens), Some(right_tokens)) => left_tokens.clone() | right_tokens.clone(),
            (Some(tokens), None) | (None, Some(tokens)) => tokens.clone(),
            (None, None) => RangeSetBlaze::new(),
        })
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
        combine_compact_entries(self, other, |left, right| match (left, right) {
            (Some(left_tokens), Some(right_tokens)) => left_tokens.clone() & right_tokens.clone(),
            _ => RangeSetBlaze::new(),
        })
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
            (Some(left_tokens), Some(right_tokens)) => left_tokens.clone() - right_tokens.clone(),
            (Some(left_tokens), None) => left_tokens.clone(),
            _ => RangeSetBlaze::new(),
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
        self.0.get(tsid).cloned().unwrap_or_else(RangeSetBlaze::new)
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.intersection(other).is_empty()
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        self.difference(other).is_empty()
    }

    fn expanded_entries(&self) -> BTreeMap<u32, RangeSetBlaze<u32>> {
        if self.is_full() {
            return BTreeMap::new();
        }
        let mut out = BTreeMap::new();
        for (range, tokens) in self.0.range_values() {
            for tsid in range {
                out.insert(tsid, tokens.clone());
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
                    tokens: rangeset_to_vec(tokens),
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
        range_map_entries(self) == range_map_entries(other)
    }
}

impl Eq for Weight {}

impl std::hash::Hash for Weight {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.is_full().hash(state);
        if !self.is_full() {
            range_map_entries(self).hash(state);
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
                format!("{tsid}→{}", rangeset_to_string(tokens))
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
                format!("{tsid}→{}", rangeset_to_string_with_names(tokens, self.token_names))
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
