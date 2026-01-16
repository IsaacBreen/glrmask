use range_set_blaze::RangeSetBlaze;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use crate::datastructures::abstract_weight::{current_num_tsids, normalize_num_tsids, WeightBackend};

/// Factorized weight representation as a union of (tsid_set × token_set) pairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactorizedWeight {
    repr: FactorizedRepr,
    num_tsids: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FactorizedRepr {
    Pairs(Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)>),
    Expanded(RangeSetBlaze<usize>),
}

impl FactorizedWeight {
    pub(crate) fn new(num_tsids: usize) -> Self {
        Self {
            repr: FactorizedRepr::Pairs(Vec::new()),
            num_tsids: normalize_num_tsids(num_tsids),
        }
    }
    
    /// Create a factorized weight from pairs directly.
    pub fn from_pairs(pairs: Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)>, num_tsids: usize) -> Self {
        let mut fw = Self {
            repr: FactorizedRepr::Pairs(pairs),
            num_tsids: normalize_num_tsids(num_tsids),
        };
        fw.normalize_pairs();
        fw
    }

    fn from_expanded_rsb(rsb: RangeSetBlaze<usize>, num_tsids: usize) -> Self {
        Self {
            repr: FactorizedRepr::Expanded(rsb),
            num_tsids: normalize_num_tsids(num_tsids),
        }
    }

    pub(crate) fn num_tsids(&self) -> usize {
        normalize_num_tsids(self.num_tsids)
    }

    pub fn is_expanded(&self) -> bool {
        matches!(self.repr, FactorizedRepr::Expanded(_))
    }

    pub fn expanded(&self) -> Option<&RangeSetBlaze<usize>> {
        match &self.repr {
            FactorizedRepr::Expanded(rsb) => Some(rsb),
            _ => None,
        }
    }

    pub fn pairs(&self) -> &[(RangeSetBlaze<usize>, RangeSetBlaze<usize>)] {
        match &self.repr {
            FactorizedRepr::Pairs(pairs) => pairs,
            FactorizedRepr::Expanded(_) => &[],
        }
    }

    pub fn pairs_len(&self) -> usize {
        match &self.repr {
            FactorizedRepr::Pairs(pairs) => pairs.len(),
            FactorizedRepr::Expanded(_) => 0,
        }
    }

    fn add_pair(&mut self, tsid_set: RangeSetBlaze<usize>, token_set: RangeSetBlaze<usize>) {
        if tsid_set.is_empty() || token_set.is_empty() {
            return;
        }
        let num_tsids = self.num_tsids();
        match &mut self.repr {
            FactorizedRepr::Pairs(pairs) => {
                for (existing_tsids, existing_tokens) in pairs.iter_mut() {
                    if *existing_tsids == tsid_set {
                        *existing_tokens |= &token_set;
                        return;
                    }
                }
                pairs.push((tsid_set, token_set));
            }
            FactorizedRepr::Expanded(expanded) => {
                let pair_rsb = Self::expand_pair_to_rsb(&tsid_set, &token_set, num_tsids);
                *expanded |= &pair_rsb;
            }
        }
    }

    fn normalize_pairs(&mut self) {
        const PAIR_THRESHOLD: usize = 200;
        let pairs = match &mut self.repr {
            FactorizedRepr::Pairs(pairs) => pairs,
            FactorizedRepr::Expanded(_) => return,
        };

        loop {
            let before_count = pairs.len();

            // First pass: merge by identical tsid_set
            let mut by_tsids = Vec::with_capacity(pairs.len());
            for (tsid_set, token_set) in std::mem::take(pairs) {
                if tsid_set.is_empty() || token_set.is_empty() {
                    continue;
                }
                let mut merged = false;
                for (existing_tsids, existing_tokens) in &mut by_tsids {
                    if *existing_tsids == tsid_set {
                        *existing_tokens |= &token_set;
                        merged = true;
                        break;
                    }
                }
                if !merged {
                    by_tsids.push((tsid_set, token_set));
                }
            }

            // Second pass: merge by identical token_set
            let mut by_tokens = Vec::with_capacity(by_tsids.len());
            for (tsid_set, token_set) in by_tsids {
                let mut merged = false;
                for (existing_tsids, existing_tokens) in &mut by_tokens {
                    if *existing_tokens == token_set {
                        *existing_tsids |= &tsid_set;
                        merged = true;
                        break;
                    }
                }
                if !merged {
                    by_tokens.push((tsid_set, token_set));
                }
            }

            *pairs = by_tokens;

            if pairs.len() >= before_count {
                break;
            }
        }

        if pairs.len() > PAIR_THRESHOLD {
            let expanded = self.expand_to_rsb_internal();
            self.repr = FactorizedRepr::Expanded(expanded);
        }
    }

    pub(crate) fn from_position_with_num_tsids(pos: usize, num_tsids: usize) -> Self {
        let num_tsids = normalize_num_tsids(num_tsids);
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        let tsid_set = RangeSetBlaze::from_iter([tsid..=tsid]);
        let token_set = RangeSetBlaze::from_iter([token..=token]);
        let mut weight = Self {
            repr: FactorizedRepr::Pairs(vec![(tsid_set, token_set)]),
            num_tsids,
        };
        weight.normalize_pairs();
        weight
    }

    pub(crate) fn all_with_max_position(max_position: usize, num_tsids: usize) -> Self {
        let num_tsids = normalize_num_tsids(num_tsids);
        if max_position == 0 {
            return Self::from_position_with_num_tsids(0, num_tsids);
        }

        let full_tsids = RangeSetBlaze::from_iter([0..=num_tsids - 1]);
        let full_tokens = max_position / num_tsids;
        let last_tsid = max_position % num_tsids;

        let mut weight = Self::new(num_tsids);
        if last_tsid == num_tsids - 1 {
            let token_set = RangeSetBlaze::from_iter([0..=full_tokens]);
            weight.add_pair(full_tsids, token_set);
        } else {
            if full_tokens > 0 {
                let token_set = RangeSetBlaze::from_iter([0..=full_tokens - 1]);
                weight.add_pair(full_tsids.clone(), token_set);
            }
            let token_set = RangeSetBlaze::from_iter([full_tokens..=full_tokens]);
            let tsid_set = RangeSetBlaze::from_iter([0..=last_tsid]);
            weight.add_pair(tsid_set, token_set);
        }
        weight.normalize_pairs();
        weight
    }

    pub(crate) fn from_rsb_with_num_tsids(rsb: &RangeSetBlaze<usize>, num_tsids: usize) -> Self {
        let num_tsids = normalize_num_tsids(num_tsids);
        if rsb.is_empty() {
            return Self::new(num_tsids);
        }

        let mut token_to_tsids: BTreeMap<usize, RangeSetBlaze<usize>> = BTreeMap::new();
        let full_tsid_set = RangeSetBlaze::from_iter([0..=num_tsids - 1]);

        for range in rsb.ranges() {
            let start = *range.start();
            let end = *range.end();
            let start_token = start / num_tsids;
            let end_token = end / num_tsids;
            let start_tsid = start % num_tsids;
            let end_tsid = end % num_tsids;

            if start_token == end_token {
                let entry = token_to_tsids.entry(start_token).or_insert_with(RangeSetBlaze::new);
                *entry |= &RangeSetBlaze::from_iter([start_tsid..=end_tsid]);
                continue;
            }

            let entry = token_to_tsids.entry(start_token).or_insert_with(RangeSetBlaze::new);
            *entry |= &RangeSetBlaze::from_iter([start_tsid..=num_tsids - 1]);

            if start_token + 1 <= end_token.saturating_sub(1) {
                for token in (start_token + 1)..=end_token - 1 {
                    let entry = token_to_tsids.entry(token).or_insert_with(RangeSetBlaze::new);
                    *entry |= &full_tsid_set;
                }
            }

            let entry = token_to_tsids.entry(end_token).or_insert_with(RangeSetBlaze::new);
            *entry |= &RangeSetBlaze::from_iter([0..=end_tsid]);
        }

        let mut weight = Self::new(num_tsids);
        for (token, tsid_set) in token_to_tsids {
            let token_set = RangeSetBlaze::from_iter([token..=token]);
            weight.add_pair(tsid_set, token_set);
        }
        weight.normalize_pairs();
        weight
    }

    pub fn expand_to_rsb(&self) -> RangeSetBlaze<usize> {
        if std::env::var("ALLOW_FACTORIZED_EXPANSION").is_err() {
            panic!(
                "Unexpected factorized weight expansion at: FactorizedWeight::expand_to_rsb(). Set ALLOW_FACTORIZED_EXPANSION=1 to allow."
            );
        }
        self.expand_to_rsb_internal()
    }

    pub(crate) fn expand_to_rsb_unchecked(&self) -> RangeSetBlaze<usize> {
        self.expand_to_rsb_internal()
    }

    fn expand_to_rsb_internal(&self) -> RangeSetBlaze<usize> {
        match &self.repr {
            FactorizedRepr::Expanded(rsb) => rsb.clone(),
            FactorizedRepr::Pairs(pairs) => {
                if pairs.is_empty() {
                    return RangeSetBlaze::new();
                }
                let num_tsids = self.num_tsids();
                let mut ranges: Vec<std::ops::RangeInclusive<usize>> = Vec::new();

                for (tsid_set, token_set) in pairs {
                    for token_range in token_set.ranges() {
                        let token_start = *token_range.start();
                        let token_end = *token_range.end();
                        for tsid_range in tsid_set.ranges() {
                            let tsid_start = *tsid_range.start();
                            let tsid_end = *tsid_range.end();
                            for token in token_start..=token_end {
                                let base = token.saturating_mul(num_tsids);
                                ranges.push(base.saturating_add(tsid_start)..=base.saturating_add(tsid_end));
                            }
                        }
                    }
                }

                RangeSetBlaze::from_iter(ranges)
            }
        }
    }

    fn expand_pair_to_rsb(
        tsid_set: &RangeSetBlaze<usize>,
        token_set: &RangeSetBlaze<usize>,
        num_tsids: usize,
    ) -> RangeSetBlaze<usize> {
        let mut ranges: Vec<std::ops::RangeInclusive<usize>> = Vec::new();
        for token_range in token_set.ranges() {
            let token_start = *token_range.start();
            let token_end = *token_range.end();
            for tsid_range in tsid_set.ranges() {
                let tsid_start = *tsid_range.start();
                let tsid_end = *tsid_range.end();
                for token in token_start..=token_end {
                    let base = token.saturating_mul(num_tsids);
                    ranges.push(base.saturating_add(tsid_start)..=base.saturating_add(tsid_end));
                }
            }
        }
        RangeSetBlaze::from_iter(ranges)
    }
}

fn hash_rangeset<H: Hasher>(rsb: &RangeSetBlaze<usize>, state: &mut H) {
    for range in rsb.ranges() {
        range.start().hash(state);
        range.end().hash(state);
    }
}

impl Hash for FactorizedWeight {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.num_tsids.hash(state);
        match &self.repr {
            FactorizedRepr::Pairs(pairs) => {
                0u8.hash(state);
                pairs.len().hash(state);
                for (tsid_set, token_set) in pairs {
                    hash_rangeset(tsid_set, state);
                    hash_rangeset(token_set, state);
                }
            }
            FactorizedRepr::Expanded(rsb) => {
                1u8.hash(state);
                hash_rangeset(rsb, state);
            }
        }
    }
}

impl WeightBackend for FactorizedWeight {
    fn empty() -> Self {
        FactorizedWeight::new(current_num_tsids())
    }

    fn all(max_position: usize) -> Self {
        FactorizedWeight::all_with_max_position(max_position, current_num_tsids())
    }

    fn from_position(pos: usize) -> Self {
        FactorizedWeight::from_position_with_num_tsids(pos, current_num_tsids())
    }

    fn from_ranges<I: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(ranges: I) -> Self {
        let rsb = RangeSetBlaze::from_iter(ranges);
        FactorizedWeight::from_rsb_with_num_tsids(&rsb, current_num_tsids())
    }

    fn is_empty(&self) -> bool {
        match &self.repr {
            FactorizedRepr::Pairs(pairs) => {
                pairs.is_empty() || pairs.iter().all(|(a, b)| a.is_empty() || b.is_empty())
            }
            FactorizedRepr::Expanded(rsb) => rsb.is_empty(),
        }
    }

    fn len(&self) -> usize {
        match &self.repr {
            FactorizedRepr::Pairs(pairs) => {
                let mut total: u128 = 0;
                for (tsid_set, token_set) in pairs {
                    let pair_count = tsid_set.len().saturating_mul(token_set.len());
                    total = total.saturating_add(pair_count);
                }
                if total > usize::MAX as u128 {
                    usize::MAX
                } else {
                    total as usize
                }
            }
            FactorizedRepr::Expanded(rsb) => {
                let len = rsb.len();
                if len > usize::MAX as u128 {
                    usize::MAX
                } else {
                    len as usize
                }
            }
        }
    }

    fn contains(&self, pos: usize) -> bool {
        match &self.repr {
            FactorizedRepr::Pairs(pairs) => {
                if pairs.is_empty() {
                    return false;
                }
                let num_tsids = self.num_tsids();
                let token = pos / num_tsids;
                let tsid = pos % num_tsids;
                pairs.iter().any(|(tsid_set, token_set)| {
                    tsid_set.contains(tsid) && token_set.contains(token)
                })
            }
            FactorizedRepr::Expanded(rsb) => rsb.contains(pos),
        }
    }

    fn ranges_len(&self) -> usize {
        match &self.repr {
            FactorizedRepr::Pairs(pairs) => pairs
                .iter()
                .map(|(tsid_set, token_set)| tsid_set.ranges_len() + token_set.ranges_len())
                .sum(),
            FactorizedRepr::Expanded(rsb) => rsb.ranges_len(),
        }
    }

    fn insert(&mut self, pos: usize) {
        match &mut self.repr {
            FactorizedRepr::Pairs(_) => {
                let num_tsids = self.num_tsids();
                let token = pos / num_tsids;
                let tsid = pos % num_tsids;
                let tsid_set = RangeSetBlaze::from_iter([tsid..=tsid]);
                let token_set = RangeSetBlaze::from_iter([token..=token]);
                self.add_pair(tsid_set, token_set);
                self.normalize_pairs();
            }
            FactorizedRepr::Expanded(rsb) => {
                *rsb |= &RangeSetBlaze::from_iter([pos..=pos]);
            }
        }
    }

    fn intersect(&self, other: &Self) -> Self {
        assert_eq!(self.num_tsids(), other.num_tsids(), "FactorizedWeight num_tsids mismatch");
        match (&self.repr, &other.repr) {
            (FactorizedRepr::Expanded(_), _) | (_, FactorizedRepr::Expanded(_)) => {
                let rsb = self.expand_to_rsb_internal() & other.expand_to_rsb_internal();
                FactorizedWeight::from_expanded_rsb(rsb, self.num_tsids())
            }
            (FactorizedRepr::Pairs(self_pairs), FactorizedRepr::Pairs(other_pairs)) => {
                let mut out = FactorizedWeight::new(self.num_tsids());
                for (tsid_a, token_a) in self_pairs {
                    for (tsid_b, token_b) in other_pairs {
                        let tsid_inter = tsid_a & tsid_b;
                        let token_inter = token_a & token_b;
                        if !tsid_inter.is_empty() && !token_inter.is_empty() {
                            out.add_pair(tsid_inter, token_inter);
                        }
                    }
                }
                out.normalize_pairs();
                out
            }
        }
    }

    fn intersect_assign(&mut self, other: &Self) {
        *self = self.intersect(other);
    }

    fn union(&self, other: &Self) -> Self {
        assert_eq!(self.num_tsids(), other.num_tsids(), "FactorizedWeight num_tsids mismatch");
        match (&self.repr, &other.repr) {
            (FactorizedRepr::Expanded(_), _) | (_, FactorizedRepr::Expanded(_)) => {
                let rsb = self.expand_to_rsb_internal() | other.expand_to_rsb_internal();
                FactorizedWeight::from_expanded_rsb(rsb, self.num_tsids())
            }
            (FactorizedRepr::Pairs(_), FactorizedRepr::Pairs(other_pairs)) => {
                let mut out = self.clone();
                for (tsid_set, token_set) in other_pairs {
                    out.add_pair(tsid_set.clone(), token_set.clone());
                }
                out.normalize_pairs();
                out
            }
        }
    }

    fn union_assign(&mut self, other: &Self) {
        assert_eq!(self.num_tsids(), other.num_tsids(), "FactorizedWeight num_tsids mismatch");
        match (&self.repr, &other.repr) {
            (FactorizedRepr::Expanded(_), _) | (_, FactorizedRepr::Expanded(_)) => {
                let rsb = self.expand_to_rsb_internal() | other.expand_to_rsb_internal();
                self.repr = FactorizedRepr::Expanded(rsb);
            }
            (FactorizedRepr::Pairs(_), FactorizedRepr::Pairs(other_pairs)) => {
                for (tsid_set, token_set) in other_pairs {
                    self.add_pair(tsid_set.clone(), token_set.clone());
                }
                self.normalize_pairs();
            }
        }
    }

    fn difference(&self, other: &Self) -> Self {
        assert_eq!(self.num_tsids(), other.num_tsids(), "FactorizedWeight num_tsids mismatch");
        match (&self.repr, &other.repr) {
            (FactorizedRepr::Expanded(_), _) | (_, FactorizedRepr::Expanded(_)) => {
                let rsb = self.expand_to_rsb_internal() - other.expand_to_rsb_internal();
                FactorizedWeight::from_expanded_rsb(rsb, self.num_tsids())
            }
            (FactorizedRepr::Pairs(self_pairs), FactorizedRepr::Pairs(other_pairs)) => {
                if self.is_empty() {
                    return FactorizedWeight::new(self.num_tsids());
                }
                if other.is_empty() {
                    return self.clone();
                }

                let mut out = FactorizedWeight::new(self.num_tsids());
                for (tsid_set, token_set) in self_pairs {
                    let mut remainders = vec![(tsid_set.clone(), token_set.clone())];
                    for (other_tsids, other_tokens) in other_pairs {
                        if remainders.is_empty() {
                            break;
                        }
                        let mut next = Vec::new();
                        for (rem_tsids, rem_tokens) in remainders {
                            let tsid_inter = &rem_tsids & other_tsids;
                            let token_inter = &rem_tokens & other_tokens;
                            if tsid_inter.is_empty() || token_inter.is_empty() {
                                next.push((rem_tsids, rem_tokens));
                                continue;
                            }

                            let tsid_diff = &rem_tsids - other_tsids;
                            if !tsid_diff.is_empty() {
                                next.push((tsid_diff, rem_tokens.clone()));
                            }

                            let token_diff = &rem_tokens - other_tokens;
                            if !token_diff.is_empty() && !tsid_inter.is_empty() {
                                next.push((tsid_inter, token_diff));
                            }
                        }
                        remainders = next;
                    }

                    for (rem_tsids, rem_tokens) in remainders {
                        out.add_pair(rem_tsids, rem_tokens);
                    }
                }

                out.normalize_pairs();
                out
            }
        }
    }

    fn complement(&self, max_position: usize) -> Self {
        let all = FactorizedWeight::all_with_max_position(max_position, self.num_tsids());
        all.difference(self)
    }

    fn min_item(&self) -> Option<usize> {
        match &self.repr {
            FactorizedRepr::Expanded(rsb) => rsb.ranges().next().map(|r| *r.start()),
            FactorizedRepr::Pairs(pairs) => {
                let num_tsids = self.num_tsids();
                pairs
                    .iter()
                    .filter_map(|(tsid_set, token_set)| {
                        let min_token = token_set.ranges().next().map(|r| *r.start())?;
                        let min_tsid = tsid_set.ranges().next().map(|r| *r.start())?;
                        Some(min_token.saturating_mul(num_tsids).saturating_add(min_tsid))
                    })
                    .min()
            }
        }
    }

    fn max_item(&self) -> Option<usize> {
        match &self.repr {
            FactorizedRepr::Expanded(rsb) => rsb.ranges().last().map(|r| *r.end()),
            FactorizedRepr::Pairs(pairs) => {
                let num_tsids = self.num_tsids();
                pairs
                    .iter()
                    .filter_map(|(tsid_set, token_set)| {
                        let max_token = token_set.ranges().last().map(|r| *r.end())?;
                        let max_tsid = tsid_set.ranges().last().map(|r| *r.end())?;
                        Some(max_token.saturating_mul(num_tsids).saturating_add(max_tsid))
                    })
                    .max()
            }
        }
    }

}
