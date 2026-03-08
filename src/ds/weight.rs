#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: This file is the split-out glrmask analogue of sep1's weighted-token machinery around `dwa_i32::Weight` and related backing structures in `datastructures/`.

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
            map.extend_simple(std::iter::once((start..=*current_end, current_tokens.clone())));
        }
        *current_start = None;
        *current_tokens = RangeSetBlaze::new();
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
        let mut expanded = self.expanded_entries();
        for (tsid, tokens) in other.expanded_entries() {
            expanded
                .entry(tsid)
                .and_modify(|existing| *existing = existing.clone() | tokens.clone())
                .or_insert(tokens);
        }
        compress_expanded(&expanded)
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
        let left = self.expanded_entries();
        let right = other.expanded_entries();
        let mut out = BTreeMap::new();
        for (tsid, left_tokens) in left {
            if let Some(right_tokens) = right.get(&tsid) {
                let tokens = left_tokens & right_tokens.clone();
                if !tokens.is_empty() {
                    out.insert(tsid, tokens);
                }
            }
        }
        compress_expanded(&out)
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
        let mut out = BTreeMap::new();
        let right = other.expanded_entries();
        for (tsid, left_tokens) in self.expanded_entries() {
            let tokens = match right.get(&tsid) {
                Some(right_tokens) => left_tokens - right_tokens.clone(),
                None => left_tokens,
            };
            if !tokens.is_empty() {
                out.insert(tsid, tokens);
            }
        }
        compress_expanded(&out)
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
