//! Dense internal-token accumulator used by Mask.
//!
//! The Parser DWA yields weights over the compact internal token universe.
//! This module carries those dense sets through parser-stack walks before final
//! materialization into the caller-visible original vocabulary mask.

use std::sync::Arc;

use crate::parser::gss::{LeveledGSS, Merge};
use crate::sets::weight::Weight;
use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

pub(super) type DenseTokenMaskCache = FxHashMap<usize, Arc<[u64]>>;
pub(super) type DenseMaskGSS = LeveledGSS<u32, DenseMaskAcc>;


#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct DenseTokenSetIntersectionKey {
    pub(super) tsid: u32,
    pub(super) dense: usize,
    pub(super) dense_len: usize,
    pub(super) token_set: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct DenseGssTransitionKey {
    pub(super) lower: usize,
    pub(super) entries: SmallVec<[(u32, usize, usize, usize); 4]>,
}


/// Dense bitmap accumulator used while walking the parser DWA.
///
/// Key:
///   parser-DWA internal tokenizer-state id.
///
/// Value:
///   dense bitmap of final shared constraint-internal token ids.
///
/// The token ids here must match parser-DWA Weight token ids. They also match
/// Constraint.can_match bitmap token ids after compile-time vocab
/// reconciliation.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct DenseMaskAcc(pub(super) SmallVec<[(u32, Arc<[u64]>); 2]>);

impl DenseMaskAcc {
    pub(super) fn from_dense(tsid: u32, dense: Vec<u64>) -> Option<Self> {
        if dense.iter().all(|&word| word == 0) {
            return None;
        }

        let dense: Arc<[u64]> = dense.into();
        let mut entries = SmallVec::new();
        entries.push((tsid, dense));
        Some(Self(entries))
    }

    pub(super) fn from_dense_arc(tsid: u32, dense: Arc<[u64]>) -> Option<Self> {
        if dense.iter().all(|&word| word == 0) {
            return None;
        }

        let mut entries = SmallVec::new();
        entries.push((tsid, dense));
        Some(Self(entries))
    }

    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub(super) fn bit_range_mask(lo_bit: usize, hi_bit: usize) -> u64 {
        debug_assert!(lo_bit <= hi_bit);
        debug_assert!(hi_bit < 64);

        let high_mask = if hi_bit == 63 {
            !0u64
        } else {
            (1u64 << (hi_bit + 1)) - 1
        };

        let low_mask = if lo_bit == 0 {
            0
        } else {
            (1u64 << lo_bit) - 1
        };

        high_mask & !low_mask
    }

    pub(super) fn for_each_token_range_word<F>(tokens: &RangeSetBlaze<u32>, word_limit: usize, mut f: F)
    where
        F: FnMut(usize, u64),
    {
        if word_limit == 0 {
            return;
        }

        let max_token_exclusive = word_limit.saturating_mul(64);
        if max_token_exclusive == 0 {
            return;
        }

        for range in tokens.ranges() {
            let lo = *range.start() as usize;
            if lo >= max_token_exclusive {
                continue;
            }

            let hi = (*range.end() as usize).min(max_token_exclusive - 1);
            if lo > hi {
                continue;
            }

            let word_lo = lo / 64;
            let word_hi = hi / 64;

            for word_idx in word_lo..=word_hi {
                let lo_bit = if word_idx == word_lo { lo % 64 } else { 0 };
                let hi_bit = if word_idx == word_hi { hi % 64 } else { 63 };
                f(word_idx, Self::bit_range_mask(lo_bit, hi_bit));
            }
        }
    }

    pub(super) fn intersect_dense_with_tokens(
        dense: &[u64],
        tokens: &RangeSetBlaze<u32>,
    ) -> Option<Arc<[u64]>> {
        if dense.is_empty() || tokens.is_empty() {
            return None;
        }

        let mut out = vec![0u64; dense.len()];
        let mut any = false;

        Self::for_each_token_range_word(tokens, dense.len(), |word_idx, token_mask| {
            let word = dense[word_idx] & token_mask;
            if word != 0 {
                out[word_idx] |= word;
                any = true;
            }
        });

        if any {
            Some(out.into())
        } else {
            None
        }
    }

    pub(super) fn intersect_dense_with_token_set(
        dense: &[u64],
        token_set: &Arc<RangeSetBlaze<u32>>,
        precomputed: &DenseTokenMaskCache,
    ) -> Option<Arc<[u64]>> {
        let key = Arc::as_ptr(token_set) as usize;

        if let Some(mask) = precomputed.get(&key) {
            let mut out = vec![0u64; dense.len()];
            let mut any = false;

            for i in 0..dense.len() {
                let word = dense[i] & mask.get(i).copied().unwrap_or(0);
                if word != 0 {
                    any = true;
                }
                out[i] = word;
            }

            if any {
                Some(out.into())
            } else {
                None
            }
        } else {
            Self::intersect_dense_with_tokens(dense, token_set)
        }
    }

    pub(super) fn or_dense_and_token_set_into(
        dense: &[u64],
        token_set: &Arc<RangeSetBlaze<u32>>,
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
    ) {
        let key = Arc::as_ptr(token_set) as usize;

        if let Some(mask) = precomputed.get(&key) {
            let n = dense.len().min(mask.len()).min(merged.len());
            for i in 0..n {
                merged[i] |= dense[i] & mask[i];
            }
        } else {
            let word_limit = dense.len().min(merged.len());
            Self::for_each_token_range_word(token_set, word_limit, |word_idx, token_mask| {
                merged[word_idx] |= dense[word_idx] & token_mask;
            });
        }
    }

    pub(super) fn intersect_with_weight(
        &self,
        weight: &Weight,
        precomputed: &DenseTokenMaskCache,
    ) -> Option<Self> {
        if self.is_empty() {
            return None;
        }

        if weight.is_full() {
            return Some(self.clone());
        }

        let mut result = SmallVec::new();

        for (tsid, dense) in &self.0 {
            let Some(token_set) = weight.0.get(*tsid) else {
                continue;
            };

            if let Some(intersection) =
                Self::intersect_dense_with_token_set(dense, token_set, precomputed)
            {
                result.push((*tsid, intersection));
            }
        }

        if result.is_empty() {
            None
        } else {
            Some(Self(result))
        }
    }

    pub(super) fn intersect_with_weight_cached(
        &self,
        weight: &Weight,
        precomputed: &DenseTokenMaskCache,
        cache: &mut FxHashMap<DenseTokenSetIntersectionKey, Option<Arc<[u64]>>>,
    ) -> Option<Self> {
        if self.is_empty() {
            return None;
        }
        if weight.is_full() {
            return Some(self.clone());
        }

        let mut result = SmallVec::new();

        for (tsid, dense) in &self.0 {
            let Some(token_set) = weight.0.get(*tsid) else {
                continue;
            };
            if let Some(intersection) = Self::intersect_dense_with_token_set_cached(
                *tsid,
                dense,
                token_set,
                precomputed,
                cache,
            ) {
                result.push((*tsid, intersection));
            }
        }

        if result.is_empty() {
            None
        } else {
            Some(Self(result))
        }
    }

    pub(super) fn intersect_dense_with_token_set_cached(
        tsid: u32,
        dense: &Arc<[u64]>,
        token_set: &Arc<RangeSetBlaze<u32>>,
        precomputed: &DenseTokenMaskCache,
        cache: &mut FxHashMap<DenseTokenSetIntersectionKey, Option<Arc<[u64]>>>,
    ) -> Option<Arc<[u64]>> {
        let key = DenseTokenSetIntersectionKey {
            tsid,
            dense: dense.as_ptr() as usize,
            dense_len: dense.len(),
            token_set: Arc::as_ptr(token_set) as usize,
        };
        if let Some(cached) = cache.get(&key) {
            return cached.clone();
        }
        if let Some(mask) = precomputed.get(&key.token_set) {
            let mut any = false;
            let mut out: Option<Vec<u64>> = None;
            for i in 0..dense.len() {
                let word = dense[i] & mask.get(i).copied().unwrap_or(0);
                any |= word != 0;
                if let Some(out) = out.as_mut() {
                    out.push(word);
                } else if word != dense[i] {
                    let mut new_out = Vec::with_capacity(dense.len());
                    new_out.extend_from_slice(&dense[..i]);
                    new_out.push(word);
                    out = Some(new_out);
                }
            }
            let result = if !any {
                None
            } else if let Some(out) = out {
                Some(out.into())
            } else {
                Some(Arc::clone(dense))
            };
            cache.insert(key, result.clone());
            return result;
        }
        let result = Self::intersect_dense_with_token_set(dense, token_set, precomputed);
        cache.insert(key, result.clone());
        result
    }

    pub(super) fn intersect_with_weight_in_place(
        &mut self,
        weight: &Weight,
        precomputed: &DenseTokenMaskCache,
    ) -> bool {
        if self.is_empty() {
            return false;
        }
        if weight.is_full() {
            return true;
        }

        let mut idx = 0usize;
        while idx < self.0.len() {
            let (tsid, dense) = &mut self.0[idx];
            let Some(token_set) = weight.0.get(*tsid) else {
                self.0.remove(idx);
                continue;
            };

            let key = Arc::as_ptr(token_set) as usize;
            if let Some(mask) = precomputed.get(&key) {
                let dense_mut = Arc::make_mut(dense);
                let mut any = false;
                for i in 0..dense_mut.len() {
                    let word = dense_mut[i] & mask.get(i).copied().unwrap_or(0);
                    any |= word != 0;
                    dense_mut[i] = word;
                }
                if any {
                    idx += 1;
                } else {
                    self.0.remove(idx);
                }
                continue;
            }

            let Some(intersection) = Self::intersect_dense_with_token_set(dense, token_set, precomputed) else {
                self.0.remove(idx);
                continue;
            };
            *dense = intersection;
            idx += 1;
        }

        !self.0.is_empty()
    }

    pub(super) fn or_into_merged(&self, merged: &mut [u64]) {
        for (_, dense) in &self.0 {
            let n = dense.len().min(merged.len());
            for i in 0..n {
                merged[i] |= dense[i];
            }
        }
    }

    pub(super) fn or_intersection_into_merged(
        &self,
        final_weight: &Weight,
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
    ) {
        if final_weight.is_full() {
            self.or_into_merged(merged);
            return;
        }

        for (tsid, dense) in &self.0 {
            let Some(token_set) = final_weight.0.get(*tsid) else {
                continue;
            };

            Self::or_dense_and_token_set_into(dense, token_set, precomputed, merged);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DenseMaskAcc, DenseTokenMaskCache};
    use range_set_blaze::RangeSetBlaze;
    use rustc_hash::FxHashMap;
    use std::sync::Arc;

    pub(super) fn precomputed_for(
        token_set: &Arc<RangeSetBlaze<u32>>,
        mask: Arc<[u64]>,
    ) -> DenseTokenMaskCache {
        let mut precomputed: FxHashMap<usize, Arc<[u64]>> = FxHashMap::default();
        precomputed.insert(Arc::as_ptr(token_set) as usize, mask);
        precomputed
    }

    #[test]
    pub(super) fn precomputed_dense_intersection_reuses_arc_when_unchanged() {
        let dense: Arc<[u64]> = Arc::from([0b1011_u64, 0b0101]);
        let token_set = Arc::new(RangeSetBlaze::from_iter([0_u32..=127]));
        let precomputed = precomputed_for(&token_set, Arc::from([!0_u64, !0_u64]));

        let mut cache = FxHashMap::default();
        let intersected = DenseMaskAcc::intersect_dense_with_token_set_cached(
            0,
            &dense,
            &token_set,
            &precomputed,
            &mut cache,
        )
        .unwrap();

        assert!(Arc::ptr_eq(&intersected, &dense));
    }

    #[test]
    pub(super) fn precomputed_dense_intersection_allocates_when_pruned() {
        let dense: Arc<[u64]> = Arc::from([0b1011_u64, 0b0101]);
        let token_set = Arc::new(RangeSetBlaze::from_iter([0_u32..=127]));
        let precomputed = precomputed_for(&token_set, Arc::from([0b0011_u64, 0b0000]));

        let mut cache = FxHashMap::default();
        let intersected = DenseMaskAcc::intersect_dense_with_token_set_cached(
            0,
            &dense,
            &token_set,
            &precomputed,
            &mut cache,
        )
        .unwrap();

        assert!(!Arc::ptr_eq(&intersected, &dense));
        assert_eq!(&*intersected, &[0b0011_u64, 0b0000]);
    }
}

impl Merge for DenseMaskAcc {
    fn merge(&self, other: &Self) -> Self {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }

        if self.0.len() == 1 && other.0.len() == 1 {
            let (left_key, left_dense) = self.0.iter().next().expect("len checked");
            let (right_key, right_dense) = other.0.iter().next().expect("len checked");
            if left_key != right_key {
                let mut entries = SmallVec::new();
                if left_key < right_key {
                    entries.push((*left_key, Arc::clone(left_dense)));
                    entries.push((*right_key, Arc::clone(right_dense)));
                } else {
                    entries.push((*right_key, Arc::clone(right_dense)));
                    entries.push((*left_key, Arc::clone(left_dense)));
                }
                return Self(entries);
            }
            if Arc::ptr_eq(left_dense, right_dense) || left_dense == right_dense {
                return self.clone();
            }
            let len = left_dense.len().max(right_dense.len());
            let mut combined = vec![0u64; len];
            for (i, &word) in left_dense.iter().enumerate() {
                combined[i] |= word;
            }
            for (i, &word) in right_dense.iter().enumerate() {
                combined[i] |= word;
            }
            let mut entries = SmallVec::new();
            entries.push((*left_key, combined.into()));
            return Self(entries);
        }

        let mut merged = self.0.clone();

        for (tsid, other_dense) in &other.0 {
            match merged.iter().position(|(existing_tsid, _)| existing_tsid == tsid) {
                Some(idx) => {
                    let dense = &mut merged[idx].1;
                    let len = dense.len().max(other_dense.len());
                    let mut combined = vec![0u64; len];

                    for (i, &word) in dense.iter().enumerate() {
                        combined[i] |= word;
                    }
                    for (i, &word) in other_dense.iter().enumerate() {
                        combined[i] |= word;
                    }

                    *dense = combined.into();
                }
                None => {
                    let insert_at = merged
                        .iter()
                        .position(|(existing_tsid, _)| existing_tsid > tsid)
                        .unwrap_or(merged.len());
                    merged.insert(insert_at, (*tsid, Arc::clone(other_dense)));
                }
            }
        }

        Self(merged)
    }
}
