//! Factored Weight representation for weight-heavy DWAs.
//!
//! A FactoredWeight represents a weight as a union of 2D profiles:
//!   Weight = OR( TokenSet_i × TsidSet_i )
//!
//! This achieves ~10x compression compared to 1D RangeSet representation
//! by exploiting the 2D structure of the N×M weight space.
//!
//! Position encoding: `pos = llm_token * num_tsids + tsid`
//!
//! Operations:
//! - AND: Pairwise intersection of terms, filter empty results
//! - OR:  Concatenate terms, optionally merge same-profile terms
//! - contains: Check if any term contains the (token, tsid) pair
//!
//! NOTE: This is a pure 2D representation. No 1D expansion is cached.
//! Expansion to 1D is only for debugging purposes.

use range_set_blaze::RangeSetBlaze;
use std::fmt;

/// A weight represented as a union of 2D profiles.
/// Each term is a Cartesian product: TokenSet × TsidSet.
#[derive(Clone)]
pub struct FactoredWeight {
    /// List of (TokenRangeSet, TsidRangeSet) pairs.
    /// The weight is the union of all these Cartesian products.
    terms: Vec<(RangeSetBlaze<u16>, RangeSetBlaze<u16>)>,
    /// Number of tokenizer states (M in the N×M expansion)
    pub num_tsids: u16,
}

impl fmt::Debug for FactoredWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FactoredWeight")
            .field("num_terms", &self.terms.len())
            .field("num_tsids", &self.num_tsids)
            .finish()
    }
}

impl PartialEq for FactoredWeight {
    fn eq(&self, other: &Self) -> bool {
        self.num_tsids == other.num_tsids && self.expand_rsb_fast() == other.expand_rsb_fast()
    }
}

impl Eq for FactoredWeight {}

impl FactoredWeight {
    /// Create a new factored weight with the given terms.
    pub fn new(terms: Vec<(RangeSetBlaze<u16>, RangeSetBlaze<u16>)>, num_tsids: u16) -> Self {
        Self { terms, num_tsids }
    }
    
    /// Create an empty factored weight.
    pub fn empty(num_tsids: u16) -> Self {
        Self { terms: Vec::new(), num_tsids }
    }
    
    /// Create a factored weight from a single Cartesian product.
    pub fn from_product(tokens: RangeSetBlaze<u16>, tsids: RangeSetBlaze<u16>, num_tsids: u16) -> Self {
        if tokens.is_empty() || tsids.is_empty() {
            return Self::empty(num_tsids);
        }
        Self { terms: vec![(tokens, tsids)], num_tsids }
    }
    
    /// Create a factored weight from a token set (N-space) × all TSIDs.
    /// This is efficient for weight-heavy mode where we have tokens and want all tsids.
    pub fn from_token_set_all_tsids(tokens: RangeSetBlaze<usize>, num_tsids: u16) -> Self {
        if tokens.is_empty() || num_tsids == 0 {
            return Self::empty(num_tsids);
        }
        // Convert usize to u16 for token set
        let tok_u16: RangeSetBlaze<u16> = tokens.ranges()
            .map(|r| (*r.start() as u16)..=(*r.end() as u16))
            .collect();
        // All tsids = [0, num_tsids - 1]
        let mut all_tsids = RangeSetBlaze::new();
        all_tsids.ranges_insert(0..=(num_tsids - 1));
        Self::from_product(tok_u16, all_tsids, num_tsids)
    }
    
    /// Create a factored weight from a token set (N-space) × specific TSID.
    /// This is the correct method for weight-heavy precomputation where we know the tsid.
    pub fn from_token_set_specific_tsid(tokens: RangeSetBlaze<usize>, tsid: usize, num_tsids: u16) -> Self {
        if tokens.is_empty() || num_tsids == 0 {
            return Self::empty(num_tsids);
        }
        // Convert usize to u16 for token set
        let tok_u16: RangeSetBlaze<u16> = tokens.ranges()
            .map(|r| (*r.start() as u16)..=(*r.end() as u16))
            .collect();
        // Just this one tsid
        let mut tsid_set = RangeSetBlaze::new();
        tsid_set.ranges_insert(tsid as u16..=tsid as u16);
        Self::from_product(tok_u16, tsid_set, num_tsids)
    }
    
    /// Create a "full" weight covering all tokens × all tsids.
    pub fn full(num_tokens: usize, num_tsids: u16) -> Self {
        if num_tokens == 0 || num_tsids == 0 {
            return Self::empty(num_tsids);
        }
        let mut all_tokens = RangeSetBlaze::new();
        all_tokens.ranges_insert(0..=(num_tokens as u16 - 1));
        let mut all_tsids = RangeSetBlaze::new();
        all_tsids.ranges_insert(0..=(num_tsids - 1));
        Self::from_product(all_tokens, all_tsids, num_tsids)
    }
    
    /// Check if this weight is empty.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty() || self.terms.iter().all(|(t, s)| t.is_empty() || s.is_empty())
    }
    
    /// Check if a (token, tsid) pair is contained in this weight.
    pub fn contains(&self, token: u16, tsid: u16) -> bool {
        self.terms.iter().any(|(tok_set, tsid_set)| {
            tok_set.contains(token) && tsid_set.contains(tsid)
        })
    }
    
    /// Check if a 1D position is contained in this weight.
    /// Position = token * num_tsids + tsid
    pub fn contains_pos(&self, pos: usize) -> bool {
        let num_tsids = self.num_tsids as usize;
        let token = (pos / num_tsids) as u16;
        let tsid = (pos % num_tsids) as u16;
        self.contains(token, tsid)
    }
    
    /// Compute the union of two factored weights.
    /// Simply concatenates the terms and optionally merges same-profile terms.
    pub fn union(&self, other: &FactoredWeight) -> FactoredWeight {
        debug_assert_eq!(self.num_tsids, other.num_tsids);
        
        // Fast path: if either is empty, return the other
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        
        // Fast path: if either is full, return a "full" weight
        if self.is_full() {
            return self.clone();
        }
        if other.is_full() {
            return other.clone();
        }
        
        let mut terms = self.terms.clone();
        terms.extend(other.terms.iter().cloned());
        
        // Optional: merge terms with identical TsidSets
        let merged = merge_same_profile_terms(terms);
        
        FactoredWeight { terms: merged, num_tsids: self.num_tsids }
    }
    
    /// Compute the intersection of two factored weights.
    /// Returns a new weight containing all (token, tsid) pairs in both weights.
    pub fn intersection(&self, other: &FactoredWeight) -> FactoredWeight {
        debug_assert_eq!(self.num_tsids, other.num_tsids);
        
        // Fast path: if either is empty, return empty
        if self.is_empty() || other.is_empty() {
            return FactoredWeight::empty(self.num_tsids);
        }
        
        // Fast path: if self is full, return other
        if self.is_full() {
            return other.clone();
        }
        // Fast path: if other is full, return self
        if other.is_full() {
            return self.clone();
        }
        
        let mut result = Vec::new();
        for (tok_a, tsid_a) in &self.terms {
            for (tok_b, tsid_b) in &other.terms {
                let tok_inter = tok_a & tok_b;
                let tsid_inter = tsid_a & tsid_b;
                if !tok_inter.is_empty() && !tsid_inter.is_empty() {
                    result.push((tok_inter, tsid_inter));
                }
            }
        }
        
        FactoredWeight { terms: result, num_tsids: self.num_tsids }
    }

    /// Subtract another factored weight (A \ B).
    /// Produces a union of rectangles representing the difference.
    pub fn subtract(&self, other: &FactoredWeight) -> FactoredWeight {
        debug_assert_eq!(self.num_tsids, other.num_tsids);

        if self.is_empty() {
            return self.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        if other.is_full() {
            return FactoredWeight::empty(self.num_tsids);
        }

        let other_rects = rects_from_terms(&other.terms);
        let mut out_rects: Vec<Rect> = Vec::new();

        for rect in rects_from_terms(&self.terms) {
            let mut current = vec![rect];
            for other_rect in &other_rects {
                if current.is_empty() {
                    break;
                }
                let mut next = Vec::new();
                for r in current {
                    next.extend(subtract_rect(r, *other_rect));
                }
                current = next;
            }
            out_rects.extend(current);
        }

        if out_rects.is_empty() {
            return FactoredWeight::empty(self.num_tsids);
        }

        let mut terms = Vec::with_capacity(out_rects.len());
        for rect in out_rects {
            let mut tok_set = RangeSetBlaze::new();
            tok_set.ranges_insert(rect.tok_lo..=rect.tok_hi);
            let mut tsid_set = RangeSetBlaze::new();
            tsid_set.ranges_insert(rect.tsid_lo..=rect.tsid_hi);
            terms.push((tok_set, tsid_set));
        }

        FactoredWeight { terms: merge_same_profile_terms(terms), num_tsids: self.num_tsids }
    }

    /// Complement within the bounded N×M domain (num_tokens × num_tsids).
    pub fn complement(&self, num_tokens: usize) -> FactoredWeight {
        let full = FactoredWeight::full(num_tokens, self.num_tsids);
        full.subtract(self)
    }

    /// Insert a single 1D position (token * num_tsids + tsid).
    pub fn insert_pos(&self, pos: usize) -> FactoredWeight {
        let num_tsids = self.num_tsids as usize;
        let token = (pos / num_tsids) as u16;
        let tsid = (pos % num_tsids) as u16;
        let mut tok_set = RangeSetBlaze::new();
        tok_set.insert(token);
        let mut tsid_set = RangeSetBlaze::new();
        tsid_set.insert(tsid);
        let single = FactoredWeight::from_product(tok_set, tsid_set, self.num_tsids);
        self.union(&single)
    }

    /// Remove a single 1D position (token * num_tsids + tsid).
    pub fn remove_pos(&self, pos: usize) -> FactoredWeight {
        let num_tsids = self.num_tsids as usize;
        let token = (pos / num_tsids) as u16;
        let tsid = (pos % num_tsids) as u16;
        let mut tok_set = RangeSetBlaze::new();
        tok_set.insert(token);
        let mut tsid_set = RangeSetBlaze::new();
        tsid_set.insert(tsid);
        let single = FactoredWeight::from_product(tok_set, tsid_set, self.num_tsids);
        self.subtract(&single)
    }

    /// Set a single 1D position to true/false.
    pub fn set_pos(&self, pos: usize, value: bool) -> FactoredWeight {
        if value {
            self.insert_pos(pos)
        } else {
            self.remove_pos(pos)
        }
    }
    
    /// Convert from 1D ranges (N×M space) to factored representation.
    /// 
    /// This decomposes each 1D range into 2D rectangles and groups them
    /// by TSID profile for compact storage.
    pub fn from_1d_ranges(ranges: impl Iterator<Item = (usize, usize)>, num_tsids: usize) -> Self {
        use std::collections::HashMap;
        
        let num_tsids_u16 = num_tsids as u16;
        
        // Group by (tsid_lo, tsid_hi) profile
        let mut profile_map: HashMap<(u16, u16), RangeSetBlaze<u16>> = HashMap::new();
        
        for (start, end) in ranges {
            // Decompose to 2D rectangles
            let tok_s = (start / num_tsids) as u16;
            let tsid_s = (start % num_tsids) as u16;
            let tok_e = (end / num_tsids) as u16;
            let tsid_e = (end % num_tsids) as u16;
            
            // Generate rectangles for this range
            let rects = decompose_range_to_rects(tok_s, tsid_s, tok_e, tsid_e, num_tsids_u16);
            
            for (t1, t2, s1, s2) in rects {
                profile_map.entry((s1, s2))
                    .or_insert_with(RangeSetBlaze::new)
                    .ranges_insert(t1..=t2);
            }
        }
        
        // Convert profile map to terms
        let terms: Vec<_> = profile_map.into_iter()
            .map(|((tsid_lo, tsid_hi), tok_set)| {
                let mut tsid_set = RangeSetBlaze::new();
                tsid_set.ranges_insert(tsid_lo..=tsid_hi);
                (tok_set, tsid_set)
            })
            .collect();
        
        FactoredWeight { terms, num_tsids: num_tsids_u16 }
    }
    
    /// Convert from a RangeSetBlaze to factored representation.
    pub fn from_rsb(rsb: &RangeSetBlaze<usize>, num_tsids: usize) -> Self {
        Self::from_1d_ranges(rsb.ranges().map(|r| (*r.start(), *r.end())), num_tsids)
    }
    
    /// Get a reference to the 2D factored terms.
    pub fn terms(&self) -> &[(RangeSetBlaze<u16>, RangeSetBlaze<u16>)] {
        &self.terms
    }
    
    /// Count the total number of terms.
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }
    
    /// Count total ranges across all terms (tokens + tsids).
    pub fn total_ranges(&self) -> usize {
        self.terms.iter()
            .map(|(t, s)| t.ranges().count() + s.ranges().count())
            .sum()
    }
    
    /// Estimate storage size in bytes.
    /// Each range is (u16, u16) = 4 bytes per range.
    pub fn estimated_storage_bytes(&self) -> usize {
        self.total_ranges() * 4
    }
    
    /// Check if this weight represents "all" (contains all positions).
    /// Uses global dimensions to determine the full token × tsid space.
    pub fn is_full(&self) -> bool {
        let dims = crate::dwa_i32::get_weight_dimensions();
        let num_tokens = dims.num_tokens;
        let num_tsids = dims.num_tsids;
        
        // Check if we have a single term that covers all tokens × all tsids
        if self.terms.len() == 1 {
            let (tok_set, tsid_set) = &self.terms[0];
            // Token set should be [0..=num_tokens-1]
            let tok_full = tok_set.len() as usize == num_tokens 
                && tok_set.first() == Some(0u16) 
                && tok_set.last() == Some(num_tokens as u16 - 1);
            // Tsid set should be [0..=num_tsids-1]
            let tsid_full = tsid_set.len() as usize == num_tsids 
                && tsid_set.first() == Some(0u16) 
                && tsid_set.last() == Some(num_tsids as u16 - 1);
            return tok_full && tsid_full;
        }
        
        // For multiple terms, we'd need to check union covers all
        // This is expensive, so return false conservatively
        false
    }
    
    /// Count the total number of positions (cardinality) in this weight.
    /// This computes the sum of |TokenSet_i| × |TsidSet_i|.
    /// 
    /// Note: This may overcount if terms overlap. For exact count, use expand().len().
    pub fn len(&self) -> usize {
        self.terms.iter()
            .map(|(t, s)| t.len() as usize * s.len() as usize)
            .sum()
    }
    
    /// Project this weight onto the token dimension only.
    /// Returns the set of tokens that appear in any (token, tsid) pair.
    pub fn project_tokens(&self) -> RangeSetBlaze<usize> {
        let mut result = RangeSetBlaze::new();
        for (tok_set, _) in &self.terms {
            for r in tok_set.ranges() {
                result.ranges_insert(*r.start() as usize..=*r.end() as usize);
            }
        }
        result
    }
    
    /// Project this weight onto the TSID dimension only.
    /// Returns the set of TSIDs that appear in any (token, tsid) pair.
    pub fn project_tsids(&self) -> RangeSetBlaze<usize> {
        let mut result = RangeSetBlaze::new();
        for (_, tsid_set) in &self.terms {
            for r in tsid_set.ranges() {
                result.ranges_insert(*r.start() as usize..=*r.end() as usize);
            }
        }
        result
    }
    
    /// Expand this factored weight to the full 1D N×M-space representation.
    /// 
    /// WARNING: This is expensive and should only be used for debugging or
    /// interfacing with code that requires 1D representation.
    /// 
    /// Position = token * num_tsids + tsid
    #[cfg(any(test, debug_assertions))]
    pub fn expand(&self) -> RangeSetBlaze<usize> {
        self.expand_impl()
    }
    
    /// Internal expand implementation - for debugging only.
    pub(crate) fn expand_impl(&self) -> RangeSetBlaze<usize> {
        self.expand_rsb_fast()
    }

    /// Efficiently expand to 1D RangeSetBlaze without iterating every position.
    /// Runs in O(num_token_ranges * num_tsid_ranges) with fast-path for full TSID rows.
    fn expand_rsb_fast(&self) -> RangeSetBlaze<usize> {
        let mut result = RangeSetBlaze::new();
        let num_tsids = self.num_tsids as usize;

        for (tok_set, tsid_set) in &self.terms {
            for tok_range in tok_set.ranges() {
                let tok_start = *tok_range.start() as usize;
                let tok_end = *tok_range.end() as usize;

                for tsid_range in tsid_set.ranges() {
                    let tsid_start = *tsid_range.start() as usize;
                    let tsid_end = *tsid_range.end() as usize;

                    if tsid_start == 0 && tsid_end + 1 == num_tsids {
                        let pos_start = tok_start * num_tsids;
                        let pos_end = (tok_end + 1) * num_tsids - 1;
                        result.ranges_insert(pos_start..=pos_end);
                        continue;
                    }

                    for token in tok_start..=tok_end {
                        let pos_start = token * num_tsids + tsid_start;
                        let pos_end = token * num_tsids + tsid_end;
                        result.ranges_insert(pos_start..=pos_end);
                    }
                }
            }
        }

        result
    }

    
    /// Get number of ranges in the 1D expansion (for debugging/comparison).
    /// WARNING: This is expensive - it requires full expansion.
    #[cfg(any(test, debug_assertions))]
    pub fn num_1d_ranges(&self) -> usize {
        self.expand_impl().ranges_len() as usize
    }
    
    /// Iterate over (token, tsid) pairs in this weight.
    /// This is more efficient than expanding to 1D for iteration.
    pub fn iter_2d(&self) -> impl Iterator<Item = (u16, u16)> + '_ {
        self.terms.iter().flat_map(|(tok_set, tsid_set)| {
            tok_set.iter().flat_map(move |token| {
                tsid_set.iter().map(move |tsid| (token, tsid))
            })
        })
    }
    
    /// Iterate over 1D positions in this weight.
    /// Position = token * num_tsids + tsid
    pub fn iter_positions(&self) -> impl Iterator<Item = usize> + '_ {
        let num_tsids = self.num_tsids as usize;
        self.iter_2d().map(move |(token, tsid)| {
            token as usize * num_tsids + tsid as usize
        })
    }
    
    /// Get the minimum position in this weight (if non-empty).
    pub fn min_position(&self) -> Option<usize> {
        let num_tsids = self.num_tsids as usize;
        self.terms.iter()
            .filter(|(t, s)| !t.is_empty() && !s.is_empty())
            .map(|(t, s)| {
                let min_tok = t.first().unwrap() as usize;
                let min_tsid = s.first().unwrap() as usize;
                min_tok * num_tsids + min_tsid
            })
            .min()
    }
    
    /// Get the maximum position in this weight (if non-empty).
    pub fn max_position(&self) -> Option<usize> {
        let num_tsids = self.num_tsids as usize;
        self.terms.iter()
            .filter(|(t, s)| !t.is_empty() && !s.is_empty())
            .map(|(t, s)| {
                let max_tok = t.last().unwrap() as usize;
                let max_tsid = s.last().unwrap() as usize;
                max_tok * num_tsids + max_tsid
            })
            .max()
    }
    
    /// Check if this weight is a subset of another weight.
    /// Returns true if every (token, tsid) in self is also in other.
    pub fn is_subset_of(&self, other: &FactoredWeight) -> bool {
        // Fast path: empty is subset of anything
        if self.is_empty() {
            return true;
        }
        
        // Fast path: if other is full, then any weight is a subset
        if other.is_full() {
            return true;
        }
        
        // Fast path: if self is full but other isn't, not a subset
        if self.is_full() && !other.is_full() {
            return false;
        }
        
        // For every term (TokenSet × TsidSet) in self, check that each
        // (token, tsid) pair is contained in other.
        // This is equivalent to: self ∩ other == self
        // Or: (self - other).is_empty()
        
        // The most efficient 2D check: for each pair in self, check other.contains()
        // However, this is still O(|self|) where |self| is cardinality.
        // An alternative: intersection and check equality of terms
        
        // For efficiency, iter self's 2D pairs and check all are in other
        for (tok, tsid) in self.iter_2d() {
            if !other.contains(tok, tsid) {
                return false;
            }
        }
        true
    }
    
    /// Clip this weight to only include positions up to max (1D position).
    /// Returns a new FactoredWeight with positions > max removed.
    pub fn clip_max(&self, max_pos: usize) -> FactoredWeight {
        let num_tsids = self.num_tsids as usize;
        let max_tok = (max_pos / num_tsids) as u16;
        let max_tsid = (max_pos % num_tsids) as u16;
        
        let mut new_terms = Vec::new();
        
        for (tok_set, tsid_set) in &self.terms {
            // Clip token set: only keep tokens <= max_tok
            let clipped_tok: RangeSetBlaze<u16> = tok_set.iter()
                .filter(|&t| t <= max_tok)
                .collect();
            
            if clipped_tok.is_empty() {
                continue;
            }
            
            // For tokens < max_tok, all tsids are ok
            // For token == max_tok, only tsids <= max_tsid are ok
            
            if clipped_tok.contains(max_tok) {
                // Split into two parts:
                // 1. Tokens < max_tok: all tsids ok
                // 2. Token == max_tok: only tsids <= max_tsid
                
                let below_max_tok: RangeSetBlaze<u16> = clipped_tok.iter()
                    .filter(|&t| t < max_tok)
                    .collect();
                
                if !below_max_tok.is_empty() {
                    new_terms.push((below_max_tok, tsid_set.clone()));
                }
                
                // Token == max_tok: clip tsids
                let clipped_tsid: RangeSetBlaze<u16> = tsid_set.iter()
                    .filter(|&s| s <= max_tsid)
                    .collect();
                
                if !clipped_tsid.is_empty() {
                    let mut just_max: RangeSetBlaze<u16> = RangeSetBlaze::new();
                    just_max.insert(max_tok);
                    new_terms.push((just_max, clipped_tsid));
                }
            } else {
                // All tokens are < max_tok, so all tsids are ok
                new_terms.push((clipped_tok, tsid_set.clone()));
            }
        }
        
        FactoredWeight { terms: merge_same_profile_terms(new_terms), num_tsids: self.num_tsids }
    }
    
    /// Iterate over 1D positions up to a maximum value.
    pub fn iter_positions_up_to(&self, max: usize) -> impl Iterator<Item = usize> + '_ {
        let num_tsids = self.num_tsids as usize;
        self.iter_2d()
            .map(move |(token, tsid)| token as usize * num_tsids + tsid as usize)
            .take_while(move |&pos| pos <= max)
    }
    
    /// Hash the 2D structure (for implementing Hash on AbstractWeight).
    /// This hashes the terms in a canonical order.
    pub fn hash_2d<H: std::hash::Hasher>(&self, state: &mut H) {
        use std::hash::Hash;
        // Hash num_tsids first
        self.num_tsids.hash(state);
        
        // Collect and sort terms for canonical ordering
        let mut sorted_terms: Vec<_> = self.terms.iter()
            .map(|(tok_set, tsid_set)| {
                let tok_ranges: Vec<(u16, u16)> = tok_set.ranges().map(|r| (*r.start(), *r.end())).collect();
                let tsid_ranges: Vec<(u16, u16)> = tsid_set.ranges().map(|r| (*r.start(), *r.end())).collect();
                (tok_ranges, tsid_ranges)
            })
            .collect();
        sorted_terms.sort();
        
        for (tok_ranges, tsid_ranges) in sorted_terms {
            tok_ranges.hash(state);
            tsid_ranges.hash(state);
        }
    }
}

/// Decompose a 1D range into 2D rectangles.
/// 
/// A 1D range [start, end] in N×M space decomposes into up to 3 rectangles:
/// 1. Partial first row: [tok_s, tok_s] × [tsid_s, M-1]  (if not starting at tsid 0)
/// 2. Full middle rows: [tok_s+1, tok_e-1] × [0, M-1]   (if there are full rows)
/// 3. Partial last row:  [tok_e, tok_e] × [0, tsid_e]    (if not ending at tsid M-1)
fn decompose_range_to_rects(tok_s: u16, tsid_s: u16, tok_e: u16, tsid_e: u16, num_tsids: u16) -> Vec<(u16, u16, u16, u16)> {
    let max_tsid = num_tsids - 1;
    
    if tok_s == tok_e {
        // Single row: just one rectangle
        return vec![(tok_s, tok_s, tsid_s, tsid_e)];
    }
    
    let mut rects = Vec::new();
    
    // Partial first row (if not starting at tsid 0)
    if tsid_s > 0 {
        rects.push((tok_s, tok_s, tsid_s, max_tsid));
    }
    
    // Full middle rows
    let first_full_row = if tsid_s > 0 { tok_s + 1 } else { tok_s };
    let last_full_row = if tsid_e < max_tsid { tok_e.saturating_sub(1) } else { tok_e };
    
    if first_full_row <= last_full_row {
        rects.push((first_full_row, last_full_row, 0, max_tsid));
    }
    
    // Partial last row (if not ending at max tsid and not same as first row)
    if tsid_e < max_tsid && tok_e > tok_s {
        rects.push((tok_e, tok_e, 0, tsid_e));
    }
    
    rects
}

#[derive(Clone, Copy, Debug)]
struct Rect {
    tok_lo: u16,
    tok_hi: u16,
    tsid_lo: u16,
    tsid_hi: u16,
}

fn rects_from_terms(terms: &[(RangeSetBlaze<u16>, RangeSetBlaze<u16>)]) -> Vec<Rect> {
    let mut rects = Vec::new();
    for (tok_set, tsid_set) in terms {
        for tok_range in tok_set.ranges() {
            for tsid_range in tsid_set.ranges() {
                rects.push(Rect {
                    tok_lo: *tok_range.start(),
                    tok_hi: *tok_range.end(),
                    tsid_lo: *tsid_range.start(),
                    tsid_hi: *tsid_range.end(),
                });
            }
        }
    }
    rects
}

fn subtract_rect(a: Rect, b: Rect) -> Vec<Rect> {
    let tok_lo = a.tok_lo.max(b.tok_lo);
    let tok_hi = a.tok_hi.min(b.tok_hi);
    let tsid_lo = a.tsid_lo.max(b.tsid_lo);
    let tsid_hi = a.tsid_hi.min(b.tsid_hi);

    if tok_lo > tok_hi || tsid_lo > tsid_hi {
        return vec![a];
    }

    let mut out = Vec::new();

    if a.tok_lo < tok_lo {
        out.push(Rect {
            tok_lo: a.tok_lo,
            tok_hi: tok_lo - 1,
            tsid_lo: a.tsid_lo,
            tsid_hi: a.tsid_hi,
        });
    }

    if tok_hi < a.tok_hi {
        out.push(Rect {
            tok_lo,
            tok_hi: a.tok_hi,
            tsid_lo: a.tsid_lo,
            tsid_hi: a.tsid_hi,
        });
    }

    if a.tsid_lo < tsid_lo {
        out.push(Rect {
            tok_lo,
            tok_hi,
            tsid_lo: a.tsid_lo,
            tsid_hi: tsid_lo - 1,
        });
    }

    if tsid_hi < a.tsid_hi {
        out.push(Rect {
            tok_lo,
            tok_hi,
            tsid_lo: tsid_hi + 1,
            tsid_hi: a.tsid_hi,
        });
    }

    out
}

/// Merge terms that have identical TSID profiles by unioning their token sets.
fn merge_same_profile_terms(terms: Vec<(RangeSetBlaze<u16>, RangeSetBlaze<u16>)>) -> Vec<(RangeSetBlaze<u16>, RangeSetBlaze<u16>)> {
    use std::collections::HashMap;
    
    // Group by serialized tsid_set (since RangeSetBlaze doesn't impl Hash/Eq directly)
    let mut groups: HashMap<Vec<(u16, u16)>, RangeSetBlaze<u16>> = HashMap::new();
    let mut tsid_sets: HashMap<Vec<(u16, u16)>, RangeSetBlaze<u16>> = HashMap::new();
    
    for (tok_set, tsid_set) in terms {
        let key: Vec<(u16, u16)> = tsid_set.ranges().map(|r| (*r.start(), *r.end())).collect();
        groups.entry(key.clone())
            .and_modify(|existing| *existing |= &tok_set)
            .or_insert_with(|| tok_set.clone());
        tsid_sets.entry(key).or_insert(tsid_set);
    }
    
    groups.into_iter()
        .map(|(key, tok_set)| (tok_set, tsid_sets.remove(&key).unwrap()))
        .collect()
}

impl std::ops::BitAnd for &FactoredWeight {
    type Output = FactoredWeight;
    
    fn bitand(self, rhs: Self) -> Self::Output {
        self.intersection(rhs)
    }
}

impl std::ops::BitOr for &FactoredWeight {
    type Output = FactoredWeight;
    
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_factored_weight_contains() {
        let num_tsids = 4;
        
        // Create a weight: tokens {0, 1, 2}, tsids {1, 3}
        let mut tokens = RangeSetBlaze::new();
        tokens.ranges_insert(0..=2);
        let mut tsids = RangeSetBlaze::new();
        tsids.insert(1);
        tsids.insert(3);
        
        let fw = FactoredWeight::from_product(tokens, tsids, num_tsids);
        
        // Should contain (0,1), (0,3), (1,1), (1,3), (2,1), (2,3)
        assert!(fw.contains(0, 1));
        assert!(fw.contains(0, 3));
        assert!(fw.contains(1, 1));
        assert!(fw.contains(2, 3));
        
        // Should NOT contain
        assert!(!fw.contains(0, 0));
        assert!(!fw.contains(0, 2));
        assert!(!fw.contains(3, 1));
    }
    
    #[test]
    fn test_factored_weight_expand() {
        let num_tsids: u16 = 4;
        
        // Create a weight: tokens {0, 1}, tsids {1, 2}
        let mut tokens = RangeSetBlaze::new();
        tokens.ranges_insert(0..=1);
        let mut tsids = RangeSetBlaze::new();
        tsids.ranges_insert(1..=2);
        
        let fw = FactoredWeight::from_product(tokens, tsids, num_tsids);
        
        // Expected positions: 
        // (0,1) = 0*4+1 = 1
        // (0,2) = 0*4+2 = 2
        // (1,1) = 1*4+1 = 5
        // (1,2) = 1*4+2 = 6
        let expanded = fw.expand();
        
        assert!(expanded.contains(1));
        assert!(expanded.contains(2));
        assert!(expanded.contains(5));
        assert!(expanded.contains(6));
        
        assert!(!expanded.contains(0));
        assert!(!expanded.contains(3));
        assert!(!expanded.contains(4));
    }
    
    #[test]
    fn test_from_1d_ranges_single_row() {
        let num_tsids = 4;
        
        // Range [5, 7] in 1D = token 1, tsids 1-3
        let fw = FactoredWeight::from_1d_ranges([(5, 7)].iter().copied(), num_tsids);
        
        assert!(fw.contains(1, 1));
        assert!(fw.contains(1, 2));
        assert!(fw.contains(1, 3));
        assert!(!fw.contains(1, 0));
        assert!(!fw.contains(0, 1));
    }
    
    #[test]
    fn test_from_1d_ranges_multi_row() {
        let num_tsids = 4;
        
        // Range [2, 9] in 1D:
        // - token 0: tsids 2-3 (positions 2-3)
        // - token 1: tsids 0-3 (positions 4-7) - full row
        // - token 2: tsids 0-1 (positions 8-9)
        let fw = FactoredWeight::from_1d_ranges([(2, 9)].iter().copied(), num_tsids);
        
        // Verify by expanding back
        let expanded = fw.expand();
        for pos in 2..=9 {
            assert!(expanded.contains(pos), "Position {} should be in expanded", pos);
        }
        assert!(!expanded.contains(0));
        assert!(!expanded.contains(1));
        assert!(!expanded.contains(10));
    }
    
    #[test]
    fn test_intersection() {
        let num_tsids: u16 = 4;
        
        // fw1: tokens {0..5}, tsids {0,1}
        let mut t1 = RangeSetBlaze::new();
        t1.ranges_insert(0..=5);
        let mut s1 = RangeSetBlaze::new();
        s1.ranges_insert(0..=1);
        let fw1 = FactoredWeight::from_product(t1, s1, num_tsids);
        
        // fw2: tokens {3..10}, tsids {1,2}
        let mut t2 = RangeSetBlaze::new();
        t2.ranges_insert(3..=10);
        let mut s2 = RangeSetBlaze::new();
        s2.ranges_insert(1..=2);
        let fw2 = FactoredWeight::from_product(t2, s2, num_tsids);
        
        let result = &fw1 & &fw2;
        
        // Intersection should be: tokens {3..5}, tsids {1}
        assert!(result.contains(3, 1));
        assert!(result.contains(4, 1));
        assert!(result.contains(5, 1));
        
        // Should not contain
        assert!(!result.contains(2, 1)); // token 2 not in intersection
        assert!(!result.contains(3, 0)); // tsid 0 not in intersection
        assert!(!result.contains(3, 2)); // tsid 2 not in fw1
    }
    
    #[test]
    fn test_union_same_profile() {
        let num_tsids: u16 = 4;
        
        // Two weights with same tsid profile should merge nicely
        let mut t1 = RangeSetBlaze::new();
        t1.ranges_insert(0..=5);
        let mut s1 = RangeSetBlaze::new();
        s1.ranges_insert(0..=1);
        let fw1 = FactoredWeight::from_product(t1, s1.clone(), num_tsids);
        
        let mut t2 = RangeSetBlaze::new();
        t2.ranges_insert(10..=15);
        let fw2 = FactoredWeight::from_product(t2, s1, num_tsids);
        
        let result = &fw1 | &fw2;
        
        // Should have merged into one term
        assert_eq!(result.num_terms(), 1);
        
        // Should contain all expected pairs
        assert!(result.contains(0, 0));
        assert!(result.contains(5, 1));
        assert!(result.contains(10, 0));
        assert!(result.contains(15, 1));
    }
    
    #[test]
    fn test_roundtrip() {
        let num_tsids = 100;
        
        // Create some 1D ranges
        let ranges = vec![
            (0, 99),     // Full first row
            (100, 299),  // Two full rows
            (350, 375),  // Partial row
        ];
        
        let fw = FactoredWeight::from_1d_ranges(ranges.clone().into_iter(), num_tsids);
        let expanded = fw.expand();
        
        // Verify all original positions are present
        for (start, end) in &ranges {
            for pos in *start..=*end {
                assert!(expanded.contains(pos), "Position {} should be in expanded", pos);
            }
        }
        
        // Verify no extra positions
        // Check some positions that should NOT be present
        assert!(!expanded.contains(300)); // Between range 2 and 3
        assert!(!expanded.contains(376)); // After range 3
    }
}
