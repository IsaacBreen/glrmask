use range_set_blaze::RangeSetBlaze;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use crate::datastructures::abstract_weight::{current_num_tsids, normalize_num_tsids, WeightBackend};

/// Factorized weight representation as a union of (tsid_set × token_set) pairs.
/// 
/// The representation is: ⋃ᵢ (tsid_setᵢ × token_setᵢ)
/// where × denotes the Cartesian product in the N×M space:
/// position = token * num_tsids + tsid
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactorizedWeight {
    /// Pairs of (tsid_set, token_set). The weight represents the union of all pairs.
    pairs: Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)>,
    num_tsids: usize,
}

impl FactorizedWeight {
    pub(crate) fn new(num_tsids: usize) -> Self {
        Self {
            pairs: Vec::new(),
            num_tsids: normalize_num_tsids(num_tsids),
        }
    }
    
    /// Create a factorized weight from pairs directly.
    pub fn from_pairs(pairs: Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)>, num_tsids: usize) -> Self {
        let mut fw = Self {
            pairs,
            num_tsids: normalize_num_tsids(num_tsids),
        };
        fw.normalize_pairs();
        fw
    }

    pub(crate) fn num_tsids(&self) -> usize {
        normalize_num_tsids(self.num_tsids)
    }

    pub fn pairs(&self) -> &[(RangeSetBlaze<usize>, RangeSetBlaze<usize>)] {
        &self.pairs
    }

    pub fn pairs_len(&self) -> usize {
        self.pairs.len()
    }

    fn add_pair(&mut self, tsid_set: RangeSetBlaze<usize>, token_set: RangeSetBlaze<usize>) {
        if tsid_set.is_empty() || token_set.is_empty() {
            return;
        }
        // Try to merge with existing pair that has same tsid_set
        for (existing_tsids, existing_tokens) in &mut self.pairs {
            if *existing_tsids == tsid_set {
                *existing_tokens |= &token_set;
                return;
            }
        }
        self.pairs.push((tsid_set, token_set));
    }

    /// Normalize pairs to find a more compact representation.
    /// 
    /// This applies iterative merging plus two greedy re-factorizations:
    /// 1. Merge pairs with identical tsid_sets (union their token_sets)
    /// 2. Merge pairs with identical token_sets (union their tsid_sets)
    /// 3. Rebuild by grouping tokens by their combined tsid_set
    /// 4. Rebuild by grouping tsids by their combined token_set
    /// 5. Pick the smallest representation
    fn normalize_pairs(&mut self) {
        if self.pairs.is_empty() {
            return;
        }

        let mut pairs = std::mem::take(&mut self.pairs);
        pairs.retain(|(tsid_set, token_set)| !tsid_set.is_empty() && !token_set.is_empty());

        let mut best = Self::merge_identical_pairs(pairs);
        if best.len() <= 1 {
            self.pairs = best;
            return;
        }

        let token_candidate = Self::normalize_by_tokens(&best);
        let tsid_candidate = Self::normalize_by_tsids(&best, self.num_tsids());

        for candidate in [token_candidate, tsid_candidate] {
            if candidate.is_empty() {
                continue;
            }
            let candidate = Self::merge_identical_pairs(candidate);
            if Self::is_better_candidate(&candidate, &best) {
                best = candidate;
            }
        }

        self.pairs = best;
    }

    fn merge_identical_pairs(
        mut pairs: Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)>,
    ) -> Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)> {
        loop {
            let before_count = pairs.len();

            // First pass: merge by identical tsid_set
            let mut by_tsids: Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)> = Vec::with_capacity(pairs.len());
            for (tsid_set, token_set) in pairs {
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
            let mut by_tokens: Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)> = Vec::with_capacity(by_tsids.len());
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

            pairs = by_tokens;

            if pairs.len() >= before_count {
                break;
            }
        }

        pairs
    }

    fn normalize_by_tokens(
        pairs: &[(RangeSetBlaze<usize>, RangeSetBlaze<usize>)],
    ) -> Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)> {
        let max_token = pairs
            .iter()
            .filter_map(|(_, token_set)| token_set.ranges().last().map(|r| *r.end()))
            .max();
        let Some(max_token) = max_token else {
            return Vec::new();
        };

        let mut token_tsids: Vec<RangeSetBlaze<usize>> = vec![RangeSetBlaze::new(); max_token + 1];

        for (tsid_set, token_set) in pairs {
            for token_range in token_set.ranges() {
                for token in *token_range.start()..=*token_range.end() {
                    if token_tsids[token].is_empty() {
                        token_tsids[token] = tsid_set.clone();
                    } else {
                        token_tsids[token] |= tsid_set;
                    }
                }
            }
        }

        let mut out: Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)> = Vec::new();
        for (token, tsid_set) in token_tsids.into_iter().enumerate() {
            if tsid_set.is_empty() {
                continue;
            }
            let mut merged = false;
            for (existing_tsids, existing_tokens) in &mut out {
                if *existing_tsids == tsid_set {
                    *existing_tokens |= &RangeSetBlaze::from_iter([token..=token]);
                    merged = true;
                    break;
                }
            }
            if !merged {
                let token_set = RangeSetBlaze::from_iter([token..=token]);
                out.push((tsid_set, token_set));
            }
        }

        out
    }

    fn normalize_by_tsids(
        pairs: &[(RangeSetBlaze<usize>, RangeSetBlaze<usize>)],
        num_tsids: usize,
    ) -> Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)> {
        let num_tsids = normalize_num_tsids(num_tsids);
        let mut tsid_tokens: Vec<RangeSetBlaze<usize>> = vec![RangeSetBlaze::new(); num_tsids];

        for (tsid_set, token_set) in pairs {
            for tsid_range in tsid_set.ranges() {
                for tsid in *tsid_range.start()..=*tsid_range.end() {
                    if tsid_tokens[tsid].is_empty() {
                        tsid_tokens[tsid] = token_set.clone();
                    } else {
                        tsid_tokens[tsid] |= token_set;
                    }
                }
            }
        }

        let mut out: Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)> = Vec::new();
        for (tsid, token_set) in tsid_tokens.into_iter().enumerate() {
            if token_set.is_empty() {
                continue;
            }
            let mut merged = false;
            for (existing_tsids, existing_tokens) in &mut out {
                if *existing_tokens == token_set {
                    *existing_tsids |= &RangeSetBlaze::from_iter([tsid..=tsid]);
                    merged = true;
                    break;
                }
            }
            if !merged {
                let tsid_set = RangeSetBlaze::from_iter([tsid..=tsid]);
                out.push((tsid_set, token_set));
            }
        }

        out
    }

    fn is_better_candidate(
        candidate: &[(RangeSetBlaze<usize>, RangeSetBlaze<usize>)],
        best: &[(RangeSetBlaze<usize>, RangeSetBlaze<usize>)],
    ) -> bool {
        let candidate_cost = Self::candidate_cost(candidate);
        let best_cost = Self::candidate_cost(best);
        candidate_cost < best_cost
    }

    fn candidate_cost(
        pairs: &[(RangeSetBlaze<usize>, RangeSetBlaze<usize>)],
    ) -> (usize, usize, u128) {
        let total_ranges: usize = pairs
            .iter()
            .map(|(tsid_set, token_set)| tsid_set.ranges_len() + token_set.ranges_len())
            .sum();
        let total_items: u128 = pairs
            .iter()
            .map(|(tsid_set, token_set)| {
                let tsid_len = tsid_set.len() as u128;
                let token_len = token_set.len() as u128;
                tsid_len.saturating_mul(token_len)
            })
            .sum();
        (pairs.len(), total_ranges, total_items)
    }

    pub(crate) fn from_position_with_num_tsids(pos: usize, num_tsids: usize) -> Self {
        let num_tsids = normalize_num_tsids(num_tsids);
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        let tsid_set = RangeSetBlaze::from_iter([tsid..=tsid]);
        let token_set = RangeSetBlaze::from_iter([token..=token]);
        Self {
            pairs: vec![(tsid_set, token_set)],
            num_tsids,
        }
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
            // Full row - single pair covers everything
            let token_set = RangeSetBlaze::from_iter([0..=full_tokens]);
            weight.add_pair(full_tsids, token_set);
        } else {
            // Partial last row
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

        // Group positions by token, collecting which tsids are set for each token
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
                // Same token row - add the tsid range
                let entry = token_to_tsids.entry(start_token).or_insert_with(RangeSetBlaze::new);
                *entry |= &RangeSetBlaze::from_iter([start_tsid..=end_tsid]);
                continue;
            }

            // Start token: from start_tsid to end of row
            let entry = token_to_tsids.entry(start_token).or_insert_with(RangeSetBlaze::new);
            *entry |= &RangeSetBlaze::from_iter([start_tsid..=num_tsids - 1]);

            // Full middle tokens
            if start_token + 1 <= end_token.saturating_sub(1) {
                for token in (start_token + 1)..=end_token - 1 {
                    let entry = token_to_tsids.entry(token).or_insert_with(RangeSetBlaze::new);
                    *entry |= &full_tsid_set;
                }
            }

            // End token: from 0 to end_tsid
            let entry = token_to_tsids.entry(end_token).or_insert_with(RangeSetBlaze::new);
            *entry |= &RangeSetBlaze::from_iter([0..=end_tsid]);
        }

        // Build weight with better factorization:
        // Group tokens by their tsid_set pattern
        let mut tsid_pattern_to_tokens: BTreeMap<Vec<(usize, usize)>, Vec<usize>> = BTreeMap::new();
        for (token, tsid_set) in token_to_tsids {
            // Use a canonical key for the tsid_set
            let key: Vec<(usize, usize)> = tsid_set.ranges().map(|r| (*r.start(), *r.end())).collect();
            tsid_pattern_to_tokens.entry(key).or_default().push(token);
        }

        // Create pairs grouped by tsid pattern
        let mut weight = Self::new(num_tsids);
        for (tsid_key, tokens) in tsid_pattern_to_tokens {
            let tsid_set: RangeSetBlaze<usize> = RangeSetBlaze::from_iter(
                tsid_key.into_iter().map(|(s, e)| s..=e)
            );
            let token_set = RangeSetBlaze::from_iter(tokens.into_iter().map(|t| t..=t));
            weight.add_pair(tsid_set, token_set);
        }
        weight.normalize_pairs();
        weight
    }

    pub fn expand_to_rsb(&self) -> RangeSetBlaze<usize> {
        if std::env::var("ALLOW_FACTORIZED_EXPANSION").is_err() {
            panic!(
                "Unexpected factorized weight expansion at: FactorizedWeight::expand_to_rsb(). \
                 Set ALLOW_FACTORIZED_EXPANSION=1 to allow. pairs_len={}",
                self.pairs.len()
            );
        }
        self.expand_to_rsb_internal()
    }

    pub(crate) fn expand_to_rsb_unchecked(&self) -> RangeSetBlaze<usize> {
        self.expand_to_rsb_internal()
    }

    fn expand_to_rsb_internal(&self) -> RangeSetBlaze<usize> {
        if self.pairs.is_empty() {
            return RangeSetBlaze::new();
        }
        let num_tsids = self.num_tsids();
        let mut ranges: Vec<std::ops::RangeInclusive<usize>> = Vec::new();

        for (tsid_set, token_set) in &self.pairs {
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

fn hash_rangeset<H: Hasher>(rsb: &RangeSetBlaze<usize>, state: &mut H) {
    for range in rsb.ranges() {
        range.start().hash(state);
        range.end().hash(state);
    }
}

impl Hash for FactorizedWeight {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.num_tsids.hash(state);
        self.pairs.len().hash(state);
        for (tsid_set, token_set) in &self.pairs {
            hash_rangeset(tsid_set, state);
            hash_rangeset(token_set, state);
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
        self.pairs.is_empty() || self.pairs.iter().all(|(a, b)| a.is_empty() || b.is_empty())
    }

    fn len(&self) -> usize {
        let mut total: u128 = 0;
        for (tsid_set, token_set) in &self.pairs {
            let pair_count = tsid_set.len().saturating_mul(token_set.len());
            total = total.saturating_add(pair_count);
        }
        if total > usize::MAX as u128 {
            usize::MAX
        } else {
            total as usize
        }
    }

    fn contains(&self, pos: usize) -> bool {
        if self.pairs.is_empty() {
            return false;
        }
        let num_tsids = self.num_tsids();
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        self.pairs.iter().any(|(tsid_set, token_set)| {
            tsid_set.contains(tsid) && token_set.contains(token)
        })
    }

    fn ranges_len(&self) -> usize {
        self.pairs
            .iter()
            .map(|(tsid_set, token_set)| tsid_set.ranges_len() + token_set.ranges_len())
            .sum()
    }

    fn insert(&mut self, pos: usize) {
        let num_tsids = self.num_tsids();
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        let tsid_set = RangeSetBlaze::from_iter([tsid..=tsid]);
        let token_set = RangeSetBlaze::from_iter([token..=token]);
        self.add_pair(tsid_set, token_set);
        self.normalize_pairs();
    }

    fn intersect(&self, other: &Self) -> Self {
        // Handle mixed num_tsids by falling back to raw position intersection.
        // This happens when mixing weights from different dims contexts (e.g., terminal-space vs weight-heavy space).
        if self.num_tsids() != other.num_tsids() {
            let self_rsb = self.expand_to_rsb_unchecked();
            let other_rsb = other.expand_to_rsb_unchecked();
            let result_rsb = &self_rsb & &other_rsb;
            // Use the larger num_tsids for proper factorization of the result
            let result_num_tsids = self.num_tsids().max(other.num_tsids());
            return FactorizedWeight::from_rsb_with_num_tsids(&result_rsb, result_num_tsids);
        }
        
        let mut out = FactorizedWeight::new(self.num_tsids());
        for (tsid_a, token_a) in &self.pairs {
            for (tsid_b, token_b) in &other.pairs {
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

    fn intersect_assign(&mut self, other: &Self) {
        *self = self.intersect(other);
    }

    fn union(&self, other: &Self) -> Self {
        // Handle mixed num_tsids by falling back to raw position union.
        if self.num_tsids() != other.num_tsids() {
            let self_rsb = self.expand_to_rsb_unchecked();
            let other_rsb = other.expand_to_rsb_unchecked();
            let result_rsb = &self_rsb | &other_rsb;
            let result_num_tsids = self.num_tsids().max(other.num_tsids());
            return FactorizedWeight::from_rsb_with_num_tsids(&result_rsb, result_num_tsids);
        }
        
        let mut out = self.clone();
        for (tsid_set, token_set) in &other.pairs {
            out.add_pair(tsid_set.clone(), token_set.clone());
        }
        out.normalize_pairs();
        out
    }

    fn union_assign(&mut self, other: &Self) {
        // Handle mixed num_tsids by falling back to raw position union.
        if self.num_tsids() != other.num_tsids() {
            *self = self.union(other);
            return;
        }
        
        for (tsid_set, token_set) in &other.pairs {
            self.add_pair(tsid_set.clone(), token_set.clone());
        }
        self.normalize_pairs();
    }

    fn difference(&self, other: &Self) -> Self {
        // Handle mixed num_tsids by falling back to raw position difference.
        if self.num_tsids() != other.num_tsids() {
            let self_rsb = self.expand_to_rsb_unchecked();
            let other_rsb = other.expand_to_rsb_unchecked();
            let result_rsb = &self_rsb - &other_rsb;
            let result_num_tsids = self.num_tsids().max(other.num_tsids());
            return FactorizedWeight::from_rsb_with_num_tsids(&result_rsb, result_num_tsids);
        }
        
        if self.is_empty() {
            return FactorizedWeight::new(self.num_tsids());
        }
        if other.is_empty() {
            return self.clone();
        }

        let mut out = FactorizedWeight::new(self.num_tsids());
        for (tsid_set, token_set) in &self.pairs {
            // Start with the original pair as the only remainder
            let mut remainders = vec![(tsid_set.clone(), token_set.clone())];
            
            // Subtract each pair from other
            for (other_tsids, other_tokens) in &other.pairs {
                if remainders.is_empty() {
                    break;
                }
                let mut next = Vec::new();
                for (rem_tsids, rem_tokens) in remainders {
                    let tsid_inter = &rem_tsids & other_tsids;
                    let token_inter = &rem_tokens & other_tokens;
                    
                    if tsid_inter.is_empty() || token_inter.is_empty() {
                        // No overlap - keep the remainder as is
                        next.push((rem_tsids, rem_tokens));
                        continue;
                    }

                    // Subtract the intersection, keeping:
                    // 1. (tsids not in other) × (all our tokens)
                    // 2. (tsids in other) × (tokens not in other)
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

    fn complement(&self, max_position: usize) -> Self {
        let all = FactorizedWeight::all_with_max_position(max_position, self.num_tsids());
        all.difference(self)
    }

    fn min_item(&self) -> Option<usize> {
        if self.is_empty() {
            return None;
        }
        let num_tsids = self.num_tsids();
        let mut min_pos: Option<usize> = None;
        for (tsid_set, token_set) in &self.pairs {
            if let (Some(min_token), Some(min_tsid)) = (
                token_set.ranges().next().map(|r| *r.start()),
                tsid_set.ranges().next().map(|r| *r.start()),
            ) {
                let pos = min_token * num_tsids + min_tsid;
                min_pos = Some(min_pos.map_or(pos, |m| m.min(pos)));
            }
        }
        min_pos
    }

    fn max_item(&self) -> Option<usize> {
        if self.is_empty() {
            return None;
        }
        let num_tsids = self.num_tsids();
        let mut max_pos: Option<usize> = None;
        for (tsid_set, token_set) in &self.pairs {
            if let (Some(max_token), Some(max_tsid)) = (
                token_set.ranges().last().map(|r| *r.end()),
                tsid_set.ranges().last().map(|r| *r.end()),
            ) {
                let pos = max_token * num_tsids + max_tsid;
                max_pos = Some(max_pos.map_or(pos, |m| m.max(pos)));
            }
        }
        max_pos
    }
}

impl FactorizedWeight {
    pub fn clip_max(&mut self, max: usize) {
        let num_tsids = self.num_tsids();
        let max_token = max / num_tsids;
        let max_tsid = max % num_tsids;
        
        let mut new_pairs = Vec::new();
        for (tsid_set, token_set) in std::mem::take(&mut self.pairs) {
            // Tokens strictly before max_token: keep all tsids
            let tokens_before = &token_set & &RangeSetBlaze::from_iter([0..=max_token.saturating_sub(1)]);
            if !tokens_before.is_empty() {
                new_pairs.push((tsid_set.clone(), tokens_before));
            }
            
            // Token at max_token: only keep tsids <= max_tsid
            if token_set.contains(max_token) {
                let tsids_at_max = &tsid_set & &RangeSetBlaze::from_iter([0..=max_tsid]);
                if !tsids_at_max.is_empty() {
                    let token_at_max = RangeSetBlaze::from_iter([max_token..=max_token]);
                    new_pairs.push((tsids_at_max, token_at_max));
                }
            }
        }
        self.pairs = new_pairs;
        self.normalize_pairs();
    }

    pub fn iter_ranges(&self) -> Box<dyn Iterator<Item = (usize, usize)> + '_> {
        Box::new(self.expand_to_rsb_internal().ranges().map(|r| (*r.start(), *r.end())).collect::<Vec<_>>().into_iter())
    }
}

impl serde::Serialize for FactorizedWeight {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("FactorizedWeight", 2)?;
        state.serialize_field("num_tsids", &self.num_tsids)?;
        let pairs_ser: Vec<(Vec<(usize, usize)>, Vec<(usize, usize)>)> = self
            .pairs
            .iter()
            .map(|(tsid_set, token_set)| {
                let tsid_ranges: Vec<(usize, usize)> = tsid_set.ranges().map(|r| (*r.start(), *r.end())).collect();
                let token_ranges: Vec<(usize, usize)> = token_set.ranges().map(|r| (*r.start(), *r.end())).collect();
                (tsid_ranges, token_ranges)
            })
            .collect();
        state.serialize_field("pairs", &pairs_ser)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for FactorizedWeight {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            num_tsids: usize,
            pairs: Vec<(Vec<(usize, usize)>, Vec<(usize, usize)>)>,
        }
        let helper = Helper::deserialize(deserializer)?;
        let pairs: Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)> = helper
            .pairs
            .into_iter()
            .map(|(tsid_ranges, token_ranges)| {
                let tsid_set = RangeSetBlaze::from_iter(tsid_ranges.into_iter().map(|(s, e)| s..=e));
                let token_set = RangeSetBlaze::from_iter(token_ranges.into_iter().map(|(s, e)| s..=e));
                (tsid_set, token_set)
            })
            .collect();
        Ok(FactorizedWeight::from_pairs(pairs, helper.num_tsids))
    }
}
