use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};

use crate::datastructures::abstract_weight::{current_num_tsids, normalize_num_tsids, WeightBackend};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeMapWeight {
    /// Maps token_id -> set of tsid values (stored as ranges over token_id).
    pub(crate) map: RangeMapBlaze<usize, RangeSetBlaze<usize>>,
    pub(crate) num_tsids: usize,
}

impl RangeMapWeight {
    pub(crate) fn new(num_tsids: usize) -> Self {
        Self {
            map: RangeMapBlaze::new(),
            num_tsids: normalize_num_tsids(num_tsids),
        }
    }

    pub(crate) fn num_tsids(&self) -> usize {
        normalize_num_tsids(self.num_tsids)
    }

    fn to_token_map(&self) -> BTreeMap<usize, RangeSetBlaze<usize>> {
        let mut out: BTreeMap<usize, RangeSetBlaze<usize>> = BTreeMap::new();
        for (token_range, tsid_set) in self.map.range_values() {
            for token in *token_range.start()..=*token_range.end() {
                out.insert(token, tsid_set.clone());
            }
        }
        out
    }

    fn merge_maps<F>(
        left: &RangeMapBlaze<usize, RangeSetBlaze<usize>>,
        right: &RangeMapBlaze<usize, RangeSetBlaze<usize>>,
        combine: F,
    ) -> RangeMapBlaze<usize, RangeSetBlaze<usize>>
    where
        F: Fn(Option<&RangeSetBlaze<usize>>, Option<&RangeSetBlaze<usize>>) -> RangeSetBlaze<usize>,
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
        let mut current_value = RangeSetBlaze::new();

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

    fn from_token_map(map: BTreeMap<usize, RangeSetBlaze<usize>>, num_tsids: usize) -> Self {
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
        let mut token_map: BTreeMap<usize, RangeSetBlaze<usize>> = BTreeMap::new();

        for range in rsb.ranges() {
            let start = *range.start();
            let end = *range.end();
            let start_token = start / num_tsids;
            let end_token = end / num_tsids;
            let start_tsid = start % num_tsids;
            let end_tsid = end % num_tsids;

            if start_token == end_token {
                let entry = token_map.entry(start_token).or_insert_with(RangeSetBlaze::new);
                *entry |= &RangeSetBlaze::from_iter([start_tsid..=end_tsid]);
                continue;
            }

            // First token partial
            {
                let entry = token_map.entry(start_token).or_insert_with(RangeSetBlaze::new);
                *entry |= &RangeSetBlaze::from_iter([start_tsid..=num_tsids - 1]);
            }

            // Middle full tokens
            if start_token + 1 <= end_token.saturating_sub(1) {
                let full = RangeSetBlaze::from_iter([0..=num_tsids - 1]);
                for token in start_token + 1..=end_token - 1 {
                    let entry = token_map.entry(token).or_insert_with(RangeSetBlaze::new);
                    *entry |= &full;
                }
            }

            // Last token partial
            {
                let entry = token_map.entry(end_token).or_insert_with(RangeSetBlaze::new);
                *entry |= &RangeSetBlaze::from_iter([0..=end_tsid]);
            }
        }

        Self::from_token_map(token_map, num_tsids)
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
        let full_tsids = RangeSetBlaze::from_iter([0..=num_tsids - 1]);
        let mut map = RangeMapBlaze::new();

        if max_token == 0 {
            let tsids = RangeSetBlaze::from_iter([0..=max_tsid]);
            if !tsids.is_empty() {
                map.ranges_insert(0..=0, tsids);
            }
            return Self { map, num_tsids };
        }

        if max_tsid == num_tsids - 1 {
            map.ranges_insert(0..=max_token, full_tsids);
        } else {
            map.ranges_insert(0..=max_token - 1, full_tsids.clone());
            let last_tsids = RangeSetBlaze::from_iter([0..=max_tsid]);
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
        let tsid_set = RangeSetBlaze::from_iter([tsid..=tsid]);
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
        let mut total = 0usize;
        for (token_range, tsid_set) in self.map.range_values() {
            let range_len = (*token_range.end()).saturating_sub(*token_range.start()).saturating_add(1);
            total = total.saturating_add(range_len.saturating_mul(tsid_set.ranges_len()));
        }
        total
    }

    fn insert(&mut self, pos: usize) {
        let num_tsids = self.num_tsids();
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        let mut new_set = RangeSetBlaze::from_iter([tsid..=tsid]);
        if let Some(existing) = self.map.get(token) {
            new_set |= existing;
        }
        self.map.ranges_insert(token..=token, new_set);
    }

    fn intersect(&self, other: &Self) -> Self {
        let map = Self::merge_maps(&self.map, &other.map, |left, right| {
            match (left, right) {
                (Some(a), Some(b)) => a & b,
                _ => RangeSetBlaze::new(),
            }
        });
        Self { map, num_tsids: self.num_tsids() }
    }

    fn intersect_assign(&mut self, other: &Self) {
        *self = self.intersect(other);
    }

    fn union(&self, other: &Self) -> Self {
        let map = Self::merge_maps(&self.map, &other.map, |left, right| {
            match (left, right) {
                (Some(a), Some(b)) => a | b,
                (Some(a), None) => a.clone(),
                (None, Some(b)) => b.clone(),
                (None, None) => RangeSetBlaze::new(),
            }
        });
        Self { map, num_tsids: self.num_tsids() }
    }

    fn union_assign(&mut self, other: &Self) {
        *self = self.union(other);
    }

    fn difference(&self, other: &Self) -> Self {
        let map = Self::merge_maps(&self.map, &other.map, |left, right| {
            match (left, right) {
                (Some(a), Some(b)) => a - b,
                (Some(a), None) => a.clone(),
                _ => RangeSetBlaze::new(),
            }
        });
        Self { map, num_tsids: self.num_tsids() }
    }

    fn complement(&self, max_position: usize) -> Self {
        let all = Self::all(max_position);
        all.difference(self)
    }

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
