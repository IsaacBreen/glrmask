pub(crate) mod profile;
pub(crate) mod queue;

use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::labels::{encode_positive_label, DEFAULT_LABEL};
use crate::ds::leveled_gss::{LeveledGSS, Merge};
use crate::ds::weight::Weight;
use crate::runtime::constraint::DenseToBufProfileStats;
use crate::runtime::state::{ConstraintState, MaskCacheData};
use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;
use std::time::Instant;

use self::profile::{
    elapsed_ns,
    emit_mask_fast_conversion_profile_line,
    emit_mask_inner_profile_line,
    emit_mask_queue_debug_line,
    mask_delta_profile_enabled,
    mask_fast_conversion_profile_enabled,
    mask_inner_profile_enabled,
    mask_queue_debug_enabled,
    mask_single_path_to_stacks_fallback_disabled,
    MaskProfile,
    MaskInnerProfileStats,
};
use self::queue::{mask_queue_mode, MaskQueue};

type DenseTokenMaskCache = FxHashMap<usize, Arc<[u64]>>;
type DenseMaskGSS = LeveledGSS<u32, DenseMaskAcc>;

const DELTA_SEED_MIN_SAVINGS: u64 = 2048;
const MASK_SINGLE_PATH_DIRECT_MAX_DEPTH: u32 = 64;
const MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct DenseTokenSetIntersectionKey {
    tsid: u32,
    dense: usize,
    dense_len: usize,
    token_set: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct DenseGssTransitionKey {
    lower: usize,
    entries: SmallVec<[(u32, usize, usize, usize); 4]>,
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
/// Constraint.possible_matches bitmap token ids after compile-time vocab
/// reconciliation.
#[derive(Clone, PartialEq, Eq, Hash)]
struct DenseMaskAcc(SmallVec<[(u32, Arc<[u64]>); 2]>);

impl DenseMaskAcc {
    fn from_dense(tsid: u32, dense: Vec<u64>) -> Option<Self> {
        if dense.iter().all(|&word| word == 0) {
            return None;
        }

        let dense: Arc<[u64]> = dense.into();
        let mut entries = SmallVec::new();
        entries.push((tsid, dense));
        Some(Self(entries))
    }

    fn from_dense_arc(tsid: u32, dense: Arc<[u64]>) -> Option<Self> {
        if dense.iter().all(|&word| word == 0) {
            return None;
        }

        let mut entries = SmallVec::new();
        entries.push((tsid, dense));
        Some(Self(entries))
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    fn bit_range_mask(lo_bit: usize, hi_bit: usize) -> u64 {
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

    fn for_each_token_range_word<F>(tokens: &RangeSetBlaze<u32>, word_limit: usize, mut f: F)
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

    fn intersect_dense_with_tokens(
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

    fn intersect_dense_with_token_set(
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

    fn or_dense_and_token_set_into(
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

    fn intersect_with_weight(
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

    fn intersect_with_weight_cached(
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

    fn intersect_dense_with_token_set_cached(
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

    fn intersect_with_weight_in_place(
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

    fn or_into_merged(&self, merged: &mut [u64]) {
        for (_, dense) in &self.0 {
            let n = dense.len().min(merged.len());
            for i in 0..n {
                merged[i] |= dense[i];
            }
        }
    }

    fn or_intersection_into_merged(
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

    fn precomputed_for(
        token_set: &Arc<RangeSetBlaze<u32>>,
        mask: Arc<[u64]>,
    ) -> DenseTokenMaskCache {
        let mut precomputed: FxHashMap<usize, Arc<[u64]>> = FxHashMap::default();
        precomputed.insert(Arc::as_ptr(token_set) as usize, mask);
        precomputed
    }

    #[test]
    fn precomputed_dense_intersection_reuses_arc_when_unchanged() {
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
    fn precomputed_dense_intersection_allocates_when_pruned() {
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

fn enqueue_gss(queue: &mut MaskQueue, target: u32, gss: DenseMaskGSS) {
    queue.enqueue(target, gss);
}

fn dense_gss_transition_key(
    gss: &DenseMaskGSS,
    weight: &Weight,
) -> Option<DenseGssTransitionKey> {
    let lower = gss.single_interface_lower_id()?;
    let mut entries = SmallVec::new();
    gss.for_each_acc(|acc| {
        for (tsid, dense) in &acc.0 {
            let token_set = weight
                .0
                .get(*tsid)
                .map(|set| Arc::as_ptr(set) as usize)
                .unwrap_or(0);
            entries.push((*tsid, dense.as_ptr() as usize, dense.len(), token_set));
        }
    });
    entries.sort_unstable();
    Some(DenseGssTransitionKey { lower, entries })
}

fn enqueue_weighted_transition(
    queue: &mut MaskQueue,
    popped: &DenseMaskGSS,
    target: u32,
    weight: &Weight,
    precomputed: &DenseTokenMaskCache,
    transition_gss_cache: &mut FxHashMap<DenseGssTransitionKey, DenseMaskGSS>,
    transition_intersection_cache: &mut FxHashMap<
        DenseTokenSetIntersectionKey,
        Option<Arc<[u64]>>,
    >,
    profile: &mut Option<MaskInnerProfileStats>,
) {
    if weight.is_full() {
        enqueue_gss(queue, target, popped.clone());
        return;
    }

    let profile_enabled = profile.is_some();
    let apply_start = if profile_enabled {
        Some(Instant::now())
    } else {
        None
    };
    let mut intersect_ns = 0u64;
    let cache_key = dense_gss_transition_key(popped, weight);
    if let Some(key) = cache_key.as_ref() {
        if let Some(cached) = transition_gss_cache.get(key) {
            if let (Some(profile), Some(start)) = (profile.as_mut(), apply_start) {
                let apply_ns = elapsed_ns(start);
                profile.transition_apply_ns += apply_ns;
                profile.transition_apply_gss_ns += apply_ns;
            }
            enqueue_gss(queue, target, cached.clone());
            return;
        }
    }

    let pruned = popped.apply_and_prune_no_promote(|allowed| {
        let intersect_start = if profile_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let intersected = allowed.intersect_with_weight_cached(
            weight,
            precomputed,
            transition_intersection_cache,
        );
        if let Some(start) = intersect_start {
            intersect_ns += elapsed_ns(start);
        }
        intersected
    });
    if let Some(key) = cache_key {
        transition_gss_cache.insert(key, pruned.clone());
    }
    if let (Some(profile), Some(start)) = (profile.as_mut(), apply_start) {
        let apply_ns = elapsed_ns(start);
        profile.transition_apply_ns += apply_ns;
        profile.transition_apply_intersect_ns += intersect_ns;
        profile.transition_apply_gss_ns += apply_ns.saturating_sub(intersect_ns);
    }

    enqueue_gss(queue, target, pruned);
}

fn enqueue_parser_state_transition(
    queue: &mut MaskQueue,
    fast_transitions: &FxHashMap<i32, (u32, Weight)>,
    parser_state: u32,
    popped: &DenseMaskGSS,
    precomputed: &DenseTokenMaskCache,
    transition_gss_cache: &mut FxHashMap<DenseGssTransitionKey, DenseMaskGSS>,
    transition_intersection_cache: &mut FxHashMap<
        DenseTokenSetIntersectionKey,
        Option<Arc<[u64]>>,
    >,
    profile: &mut Option<MaskInnerProfileStats>,
) {
    let positive_label = encode_positive_label(parser_state);

    let lookup_start = if profile.is_some() {
        Some(Instant::now())
    } else {
        None
    };
    let Some((target, weight)) = fast_transitions
        .get(&positive_label)
        .or_else(|| fast_transitions.get(&DEFAULT_LABEL))
    else {
        if let (Some(profile), Some(start)) = (profile.as_mut(), lookup_start) {
            profile.transition_lookup_ns += elapsed_ns(start);
        }
        return;
    };
    if let (Some(profile), Some(start)) = (profile.as_mut(), lookup_start) {
        profile.transition_lookup_ns += elapsed_ns(start);
    }

    queue.record_parser_dwa_transition_enqueue();
    enqueue_weighted_transition(
        queue,
        popped,
        *target,
        weight,
        precomputed,
        transition_gss_cache,
        transition_intersection_cache,
        profile,
    );
}

fn update_eos_mask(buf: &mut [u32], eos_token_id: Option<u32>, is_complete: bool) {
    let Some(eos_token_id) = eos_token_id else {
        return;
    };

    let word = eos_token_id as usize / 32;
    let bit = eos_token_id as usize % 32;

    let Some(slot) = buf.get_mut(word) else {
        return;
    };

    *slot &= !(1u32 << bit);

    if is_complete {
        *slot |= 1u32 << bit;
    }
}

fn eos_mask_bit(buf: &[u32], eos_token_id: Option<u32>) -> bool {
    let Some(eos_token_id) = eos_token_id else {
        return false;
    };
    let word = eos_token_id as usize / 32;
    let bit = eos_token_id as usize % 32;
    buf.get(word)
        .map(|slot| (*slot & (1u32 << bit)) != 0)
        .unwrap_or(false)
}

impl<'a> ConstraintState<'a> {
    fn try_fill_mask_single_path_direct(&self, buf: &mut [u32]) -> bool {
        if mask_inner_profile_enabled() || mask_delta_profile_enabled() {
            return false;
        }

        if self.state.is_empty() || self.state.len() > 4 {
            return false;
        }

        let mut paths = SmallVec::<[(u32, TerminalsDisallowed, SmallVec<[u32; 16]>); MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS]>::new();
        for (&original_tokenizer_state, gss) in &self.state {
            if gss.max_depth() > MASK_SINGLE_PATH_DIRECT_MAX_DEPTH {
                return false;
            }

            let mut stack = SmallVec::<[u32; 16]>::new();
            if let Some(terminals_disallowed) = gss.single_path_top_first_and_acc(&mut stack) {
                paths.push((original_tokenizer_state, terminals_disallowed, stack));
                continue;
            }

            // TODO: Add a direct `try_virtual_stack()` mask path and remove
            // this `to_stacks()` fallback entirely. Once mask generation can
            // consume virtual stacks without materializing concrete paths,
            // reassess whether the broader single-path fast paths are still
            // pulling their weight; `try_virtual_stack()` should cover most
            // of the cases that justify special handling.
            if mask_single_path_to_stacks_fallback_disabled() {
                return false;
            }
            if gss.path_count_at_most(MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS + 1)
                > MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS
            {
                return false;
            }
            for (stack_bottom_first, terminals_disallowed) in gss.to_stacks() {
                stack.clear();
                stack.extend(stack_bottom_first.iter().rev().copied());
                paths.push((original_tokenizer_state, terminals_disallowed, stack.clone()));
                if paths.len() > MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS {
                    return false;
                }
            }
        }

        let parser_dwa = self.constraint.parser_dwa();
        if parser_dwa.states().is_empty() {
            return false;
        }

        buf.fill(0);

        let precomputed = &self.constraint.weight_token_dense_masks;
        let dense_words = self.constraint.internal_token_dense_words;
        let mut merged = {
            let mut scratch = self.mask_scratch.lock().unwrap();
            std::mem::take(&mut scratch.merged_dense)
        };
        merged.clear();
        merged.resize(dense_words, 0);
        let mut used_direct_final = false;
        let mut direct_buf_dirty = false;

        let restore_scratch = |merged: Vec<u64>| {
            let mut scratch = self.mask_scratch.lock().unwrap();
            scratch.merged_dense = merged;
            scratch.chain_merged_dense.clear();
        };

        for (original_tokenizer_state, terminals_disallowed, stack) in paths {
            let internal_tsid = self
                .constraint
                .internal_tsid_for_state(original_tokenizer_state);
            let Some(mut acc) = self.terminals_disallowed_to_dense_acc(
                &terminals_disallowed,
                original_tokenizer_state,
                internal_tsid,
            ) else {
                continue;
            };

            let mut dwa_state_id = parser_dwa.start_state();
            let mut stack_idx = 0usize;

            loop {
                let dwa_state = &parser_dwa.states()[dwa_state_id as usize];
                if let Some(final_weight) = &dwa_state.final_weight {
                    used_direct_final = true;
                    self.merge_final_weight_to_internal(
                        final_weight,
                        &acc,
                        precomputed,
                        &mut merged,
                        Some(&mut *buf),
                        &mut direct_buf_dirty,
                    );
                }

                let Some(&parser_state) = stack.get(stack_idx) else {
                    break;
                };
                stack_idx += 1;

                let positive_label = encode_positive_label(parser_state);
                if stack_idx == 1 {
                    if let Some(accept_weight) = self
                        .constraint
                        .parser_top_accept
                        .get(&positive_label)
                        .or_else(|| self.constraint.parser_top_accept.get(&DEFAULT_LABEL))
                    {
                        used_direct_final = true;
                        self.merge_final_weight_to_internal(
                            accept_weight,
                            &acc,
                            precomputed,
                            &mut merged,
                            Some(&mut *buf),
                            &mut direct_buf_dirty,
                        );
                    }
                }
                let fast_transitions = &self.constraint.dwa_fast_transitions[dwa_state_id as usize];
                let Some((target, weight)) = fast_transitions
                    .get(&positive_label)
                    .or_else(|| fast_transitions.get(&DEFAULT_LABEL))
                else {
                    break;
                };

                if !acc.intersect_with_weight_in_place(weight, precomputed) {
                    break;
                }
                dwa_state_id = *target;
            }
        }

        if !used_direct_final && !self.is_complete() {
            restore_scratch(merged);
            return false;
        }

        if merged.iter().any(|&word| word != 0) {
            let buf_zeroed = !direct_buf_dirty;
            self.constraint.or_internal_dense_to_buf(&merged, buf, buf_zeroed);
        }

        update_eos_mask(buf, self.constraint.eos_token_id, self.is_complete());

        if direct_buf_dirty {
            self.store_mask_cache_reuse_dense(buf);
        } else {
            self.store_mask_cache(buf, &merged);
        }

        restore_scratch(merged);
        true
    }

    fn try_fill_mask_from_cache(&self, buf: &mut [u32]) -> bool {
        let cache = self.mask_cache.lock().unwrap();

        let Some(cache_data) = cache.as_ref() else {
            return false;
        };

        if cache_data.generation != self.generation {
            return false;
        }

        buf.copy_from_slice(&cache_data.mask);
        true
    }

    fn store_mask_cache(&self, buf: &[u32], merged_dense: &[u64]) {
        let mut cache = self.mask_cache.lock().unwrap();

        match cache.as_mut() {
            Some(cache_data) => {
                cache_data.generation = self.generation;

                cache_data.mask.clear();
                cache_data.mask.extend_from_slice(buf);

                cache_data.merged_dense.clear();
                cache_data.merged_dense.extend_from_slice(merged_dense);
            }
            None => {
                *cache = Some(MaskCacheData {
                    generation: self.generation,
                    mask: buf.to_vec(),
                    merged_dense: merged_dense.to_vec(),
                });
            }
        }
    }

    fn terminals_disallowed_to_dense_acc(
        &self,
        terminals_disallowed: &TerminalsDisallowed,
        original_tokenizer_state: u32,
        internal_tsid: u32,
    ) -> Option<DenseMaskAcc> {
        let base = &self.constraint.seed_universe_dense;
        if base.is_empty() {
            return None;
        }
        let terminal_masks = &self.constraint.seed_terminal_dense;

        let Some(disallowed_in_state) = terminals_disallowed.get(&original_tokenizer_state) else {
            return DenseMaskAcc::from_dense_arc(internal_tsid, Arc::clone(base));
        };

        if disallowed_in_state.is_empty() {
            return DenseMaskAcc::from_dense_arc(internal_tsid, Arc::clone(base));
        }

        let mut dense = base.to_vec();

        // TerminalsDisallowed remains keyed by ORIGINAL tokenizer state because
        // it describes tokenizer futures accumulated by the GLR parser.
        //
        // possible_matches weights themselves are already in the final shared
        // internal TSID/token spaces. `seed_terminal_dense` bridges back to
        // original tokenizer states by expanding each internal TSID through
        // `internal_tsid_to_states` during precomputation.
        for &terminal_id in disallowed_in_state {
            if let Some(mask) = terminal_masks.get(&(original_tokenizer_state, terminal_id)) {
                for (allowed_word, mask_word) in dense.iter_mut().zip(mask.iter()) {
                    *allowed_word &= !mask_word;
                }
            }
        }

        DenseMaskAcc::from_dense(internal_tsid, dense)
    }

    fn merge_final_weight_to_internal(
        &self,
        final_weight: &Weight,
        acc: &DenseMaskAcc,
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
        mut direct_buf: Option<&mut [u32]>,
        direct_buf_dirty: &mut bool,
    ) -> bool {
        let mut all_direct = true;
        if final_weight.is_full() {
            for (_, dense) in &acc.0 {
                let n = dense.len().min(merged.len());
                for i in 0..n {
                    merged[i] |= dense[i];
                }
                all_direct = false;
            }
        } else {
            for (tsid, dense) in &acc.0 {
                let Some(token_set) = final_weight.0.get(*tsid) else {
                    continue;
                };

                let handled_directly = if let Some(buf) = direct_buf.as_deref_mut() {
                    let token_set_key = Arc::as_ptr(token_set) as usize;
                    if self
                        .constraint
                        .direct_sparse_weight_token_sets
                        .contains(&token_set_key)
                        && self
                            .constraint
                            .or_dense_token_set_to_buf_sparse(dense, token_set, 2048, buf)
                            .is_some()
                    {
                        *direct_buf_dirty = true;
                        true
                    } else if self
                        .constraint
                        .or_weight_token_set_to_buf_if_contained(dense, token_set, buf)
                    {
                        *direct_buf_dirty = true;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                if !handled_directly {
                    DenseMaskAcc::or_dense_and_token_set_into(dense, token_set, precomputed, merged);
                    all_direct = false;
                }
            }
        }

        all_direct
    }

    fn merge_final_weight_for_accs(
        &self,
        final_weight: &Weight,
        accs: &[DenseMaskAcc],
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
        direct_buf: &mut Option<&mut [u32]>,
        direct_buf_dirty: &mut bool,
    ) -> bool {
        let mut all_direct = true;
        for acc in accs {
            all_direct &= self.merge_final_weight_to_internal(
                final_weight,
                acc,
                precomputed,
                merged,
                direct_buf.as_deref_mut(),
                direct_buf_dirty,
            );
        }
        all_direct
    }

    fn merge_final_weight_for_gss(
        &self,
        final_weight: &Weight,
        gss: &DenseMaskGSS,
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
        direct_buf: &mut Option<&mut [u32]>,
        direct_buf_dirty: &mut bool,
    ) -> bool {
        let mut all_direct = true;
        gss.for_each_acc(|acc| {
            all_direct &= self.merge_final_weight_to_internal(
                final_weight,
                acc,
                precomputed,
                merged,
                direct_buf.as_deref_mut(),
                direct_buf_dirty,
            );
        });
        all_direct
    }

    fn seed_mask_queue_merged(
        &self,
        start_final_weight: Option<&Weight>,
        start_fast_transitions: &FxHashMap<i32, (u32, Weight)>,
        precomputed: &DenseTokenMaskCache,
        transition_gss_cache: &mut FxHashMap<DenseGssTransitionKey, DenseMaskGSS>,
        transition_intersection_cache: &mut FxHashMap<
            DenseTokenSetIntersectionKey,
            Option<Arc<[u64]>>,
        >,
        queue: &mut MaskQueue,
        merged: &mut [u64],
        direct_buf: &mut Option<&mut [u32]>,
        direct_buf_possible: &mut bool,
        direct_buf_used: &mut bool,
        direct_buf_dirty: &mut bool,
        profile: &mut Option<MaskInnerProfileStats>,
    ) {
        for (&tokenizer_state, gss) in &self.state {
            if gss.is_empty() {
                continue;
            }

            let original_tokenizer_state = tokenizer_state;
            let internal_tsid = self.constraint.internal_tsid_for_state(original_tokenizer_state);

            let seed_decompose_start = if profile.is_some() {
                Some(Instant::now())
            } else {
                None
            };
            let (decomposed, root_accs) =
                gss.apply_transform_and_decompose(|terminals_disallowed| {
                    self.terminals_disallowed_to_dense_acc(
                        terminals_disallowed,
                        original_tokenizer_state,
                        internal_tsid,
                    )
                });
            if let (Some(profile), Some(start)) = (profile.as_mut(), seed_decompose_start) {
                profile.seed_decompose_ns += elapsed_ns(start);
            }

            if decomposed.is_empty() && root_accs.is_empty() {
                continue;
            }

            if let Some(final_weight) = start_final_weight {
                let accumulate_start = if profile.is_some() {
                    Some(Instant::now())
                } else {
                    None
                };
                *direct_buf_used = true;
                *direct_buf_possible &= self.merge_final_weight_for_accs(
                    final_weight,
                    &root_accs,
                    precomputed,
                    merged,
                    direct_buf,
                    direct_buf_dirty,
                );

                for (_, sub_gss) in &decomposed {
                    *direct_buf_possible &= self.merge_final_weight_for_gss(
                        final_weight,
                        sub_gss,
                        precomputed,
                        merged,
                        direct_buf,
                        direct_buf_dirty,
                    );
                }
                if let (Some(profile), Some(start)) = (profile.as_mut(), accumulate_start) {
                    profile.token_accumulation_ns += elapsed_ns(start);
                }
            }

            for (parser_state, popped) in &decomposed {
                let positive_label = encode_positive_label(*parser_state);
                if let Some(accept_weight) = self
                    .constraint
                    .parser_top_accept
                    .get(&positive_label)
                    .or_else(|| self.constraint.parser_top_accept.get(&DEFAULT_LABEL))
                {
                    let accumulate_start = if profile.is_some() {
                        Some(Instant::now())
                    } else {
                        None
                    };
                    *direct_buf_used = true;
                    *direct_buf_possible &= self.merge_final_weight_for_gss(
                        accept_weight,
                        popped,
                        precomputed,
                        merged,
                        direct_buf,
                        direct_buf_dirty,
                    );
                    if let (Some(profile), Some(start)) = (profile.as_mut(), accumulate_start) {
                        profile.token_accumulation_ns += elapsed_ns(start);
                    }
                }
                queue.record_seed_decompose_callback();
                enqueue_parser_state_transition(
                    queue,
                    start_fast_transitions,
                    *parser_state,
                    popped,
                    precomputed,
                    transition_gss_cache,
                    transition_intersection_cache,
                    profile,
                );
            }
        }
    }

    fn store_mask_cache_reuse_dense(&self, buf: &[u32]) {
        let mut cache = self.mask_cache.lock().unwrap();

        match cache.as_mut() {
            Some(cache_data) => {
                cache_data.generation = self.generation;
                cache_data.mask.clear();
                cache_data.mask.extend_from_slice(buf);
                cache_data.merged_dense.clear();
            }
            None => {
                *cache = Some(MaskCacheData {
                    generation: self.generation,
                    mask: buf.to_vec(),
                    merged_dense: Vec::new(),
                });
            }
        }
    }

    fn touch_mask_cache_generation(&self) {
        let mut cache = self.mask_cache.lock().unwrap();
        if let Some(cache_data) = cache.as_mut() {
            cache_data.generation = self.generation;
        }
    }

    fn fill_mask_uncached(&self, buf: &mut [u32]) {
        let _ = self.fill_mask_uncached_maybe_profile(buf, false);
    }

    fn fill_mask_uncached_maybe_profile(
        &self,
        buf: &mut [u32],
        force_profile: bool,
    ) -> Option<MaskProfile> {
        let total_start = (force_profile || mask_inner_profile_enabled()).then(Instant::now);

        if self.try_fill_mask_single_path_direct(buf) {
            return total_start.map(|start| MaskProfile {
                total_ns: elapsed_ns(start),
                single_path_direct: 1,
                ..MaskProfile::default()
            });
        }

        self.fill_mask_uncached_queue(buf, force_profile, total_start)
    }

    fn fill_mask_uncached_queue(
        &self,
        buf: &mut [u32],
        force_profile: bool,
        total_start: Option<Instant>,
    ) -> Option<MaskProfile> {
        let parser_dwa = self.constraint.parser_dwa();

        if self.state.is_empty() || parser_dwa.states().is_empty() {
            buf.fill(0);
            update_eos_mask(buf, self.constraint.eos_token_id, self.is_complete());
            self.store_mask_cache(buf, &[]);
            return total_start.map(|start| MaskProfile {
                total_ns: elapsed_ns(start),
                ..MaskProfile::default()
            });
        }

        let precomputed = &self.constraint.weight_token_dense_masks;
        let dense_words = self.constraint.internal_token_dense_words;
        let mut transition_gss_cache: FxHashMap<DenseGssTransitionKey, DenseMaskGSS> =
            FxHashMap::default();
        let mut transition_intersection_cache: FxHashMap<
            DenseTokenSetIntersectionKey,
            Option<Arc<[u64]>>,
        > = FxHashMap::default();

        let mut merged = {
            let mut scratch = self.mask_scratch.lock().unwrap();
            std::mem::take(&mut scratch.merged_dense)
        };

        buf.fill(0);
        merged.clear();
        merged.resize(dense_words, 0);
        let mut direct_buf = Some(&mut *buf);
        let mut direct_buf_possible = true;
        let mut direct_buf_used = false;
        let mut direct_buf_dirty = false;

        let mut queue = MaskQueue::new();
        let mut profile = if force_profile || mask_inner_profile_enabled() {
            Some(MaskInnerProfileStats::default())
        } else {
            None
        };
        let delta_profile_enabled = profile.is_some() && mask_delta_profile_enabled();

        let start_state = parser_dwa.start_state();
        let start_dwa_state = &parser_dwa.states()[start_state as usize];
        let start_fast_transitions = &self.constraint.dwa_fast_transitions[start_state as usize];

        self.seed_mask_queue_merged(
            start_dwa_state.final_weight.as_ref(),
            start_fast_transitions,
            precomputed,
            &mut transition_gss_cache,
            &mut transition_intersection_cache,
            &mut queue,
            &mut merged,
            &mut direct_buf,
            &mut direct_buf_possible,
            &mut direct_buf_used,
            &mut direct_buf_dirty,
            &mut profile,
        );

        loop {
            let popped = queue.pop_next();
            if let Some(profile) = profile.as_mut() {
                profile.queue_pop_ns = queue.debug_stats().pop_total_ns;
            }

            let Some((wa_state, gss)) = popped else {
                break;
            };

            let dwa_state = &parser_dwa.states()[wa_state as usize];
            let fast_transitions = &self.constraint.dwa_fast_transitions[wa_state as usize];

            if let Some(final_weight) = &dwa_state.final_weight {
                let accumulate_start = if profile.is_some() {
                    Some(Instant::now())
                } else {
                    None
                };
                direct_buf_used = true;
                direct_buf_possible &= self.merge_final_weight_for_gss(
                    final_weight,
                    &gss,
                    precomputed,
                    &mut merged,
                    &mut direct_buf,
                    &mut direct_buf_dirty,
                );
                if let (Some(profile), Some(start)) = (profile.as_mut(), accumulate_start) {
                    profile.token_accumulation_ns += elapsed_ns(start);
                }
            }

            let loop_decompose_start = if profile.is_some() {
                Some(Instant::now())
            } else {
                None
            };
            gss.for_each_decomposed(|parser_state, popped| {
                let callback_start = if profile.is_some() {
                    Some(Instant::now())
                } else {
                    None
                };
                queue.record_loop_decompose_callback();
                enqueue_parser_state_transition(
                    &mut queue,
                    fast_transitions,
                    parser_state,
                    &popped,
                    precomputed,
                    &mut transition_gss_cache,
                    &mut transition_intersection_cache,
                    &mut profile,
                );
                if let (Some(profile), Some(start)) = (profile.as_mut(), callback_start) {
                    profile.loop_decompose_callback_ns += elapsed_ns(start);
                }
            });

            if let (Some(profile), Some(start)) = (profile.as_mut(), loop_decompose_start) {
                profile.loop_decompose_total_ns += elapsed_ns(start);
            }
        }

        if mask_queue_debug_enabled() {
            let debug = queue.debug_stats();
            let line = format!(
                "[glrmask/debug][mask_queue] mode={:?} enqueue_calls={} merge_hits={} fuse_calls={} fuse_changed_depth={} stale_skips={} popped_items={} seed_decompose_callbacks={} loop_decompose_callbacks={} parser_dwa_transitions_enqueued={}",
                mask_queue_mode(),
                debug.enqueue_calls,
                debug.merge_hit_count,
                debug.fuse_calls,
                debug.fuse_changed_depth,
                debug.stale_schedule_skips,
                debug.popped_items,
                debug.seed_decompose_callbacks,
                debug.loop_decompose_callbacks,
                debug.parser_dwa_transitions_enqueued,
            );
            emit_mask_queue_debug_line(&line);
        }

        drop(direct_buf);
        let finalize_start = profile.as_ref().map(|_| Instant::now());

        let merged_has_leftovers = merged.iter().any(|&word| word != 0);
        let direct_finalized = direct_buf_used && direct_buf_possible && !merged_has_leftovers;
        let can_use_merged_cache = !direct_buf_dirty;
        let mut use_delta_seed = direct_finalized;
        let mut reuse_existing_cache_dense = false;
        if !direct_finalized && can_use_merged_cache {
            let cache = self.mask_cache.lock().unwrap();
            if let Some(cache_data) = cache.as_ref() {
                if cache_data.mask.len() == buf.len()
                    && cache_data.merged_dense.len() == merged.len()
                    && cache_data.merged_dense == merged
                {
                    let zero_start = profile.as_ref().map(|_| Instant::now());
                    buf.copy_from_slice(&cache_data.mask);
                    if let (Some(profile), Some(start)) = (profile.as_mut(), zero_start) {
                        profile.finalize_zero_ns += elapsed_ns(start);
                        profile.finalize_equal_dense_copy_seed = 1;
                        if delta_profile_enabled {
                            profile.delta_prev_available = 1;
                            profile.delta_unchanged_words = merged.len() as u64;
                            profile.delta_copy_cost_words = self.constraint.mask_len() as u64;
                            profile.delta_used_seed = 1;
                        }
                    }
                    reuse_existing_cache_dense = true;
                    use_delta_seed = true;
                }
            }
            if !use_delta_seed {
                if let Some(cache_data) = cache.as_ref().filter(|c| c.merged_dense.len() == merged.len()) {
                    let scratch_cost = self.constraint.estimate_internal_dense_to_buf_cost(&merged);
                    let copy_cost_words = self.constraint.mask_len() as u64;
                    let mut added_bits = 0u64;
                    let mut removed_bits = 0u64;
                    let mut unchanged_words = 0u64;
                    let mut unchanged_bits = 0u64;
                    let mut added_cost = 0u64;
                    let mut removed_cost = 0u64;
                    let capture_delta_summary = delta_profile_enabled;
                    let n_internal = self.constraint.internal_token_to_tokens.len();
                    let word_len = merged.len().max(cache_data.merged_dense.len());
                    for wi in 0..word_len {
                        if wi * 64 >= n_internal {
                            break;
                        }
                        let remaining = n_internal - wi * 64;
                        let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
                        let current = merged.get(wi).copied().unwrap_or(0) & valid_mask;
                        let previous = cache_data.merged_dense.get(wi).copied().unwrap_or(0) & valid_mask;
                        if capture_delta_summary && current == previous {
                            unchanged_words += 1;
                        }
                        if capture_delta_summary {
                            unchanged_bits += (!(current ^ previous) & valid_mask).count_ones() as u64;
                        }

                        let added = current & !previous;
                        if capture_delta_summary {
                            added_bits += added.count_ones() as u64;
                        }
                        if added == valid_mask {
                            if let Some(group_mask) = self.constraint.word_group_sparse_masks.get(wi) {
                                added_cost += group_mask.len() as u64;
                            } else {
                                added_cost += self
                                    .constraint
                                    .internal_bits_grouped_buf_op_cost(wi, added, valid_mask, copy_cost_words as usize)
                                    as u64;
                            }
                        } else if added != 0 {
                            added_cost += self
                                .constraint
                                .internal_bits_grouped_buf_op_cost(wi, added, valid_mask, copy_cost_words as usize)
                                as u64;
                        }

                        let removed = previous & !current;
                        if capture_delta_summary {
                            removed_bits += removed.count_ones() as u64;
                        }
                        if removed == valid_mask {
                            if let Some(group_mask) = self.constraint.word_group_sparse_masks.get(wi) {
                                removed_cost += group_mask.len() as u64;
                            } else {
                                removed_cost += self
                                    .constraint
                                    .internal_bits_grouped_buf_op_cost(wi, removed, valid_mask, copy_cost_words as usize)
                                    as u64;
                            }
                        } else if removed != 0 {
                            removed_cost += self
                                .constraint
                                .internal_bits_grouped_buf_op_cost(wi, removed, valid_mask, copy_cost_words as usize)
                                as u64;
                        }
                    }

                    let delta_cost = copy_cost_words + added_cost + removed_cost;
                    let delta_savings = scratch_cost.saturating_sub(delta_cost);

                    if delta_profile_enabled {
                        if let Some(profile) = profile.as_mut() {
                            profile.delta_prev_available = 1;
                            profile.delta_added_bits = added_bits;
                            profile.delta_removed_bits = removed_bits;
                            profile.delta_unchanged_words = unchanged_words;
                            profile.delta_unchanged_bits = unchanged_bits;
                            profile.delta_added_cost = added_cost;
                            profile.delta_removed_cost = removed_cost;
                            profile.delta_copy_cost_words = copy_cost_words;
                            profile.delta_scratch_estimated_cost = scratch_cost;
                            profile.delta_estimated_cost = delta_cost;
                            profile.delta_estimated_savings = delta_savings;
                        }
                    }
                    let delta_wins_decisively =
                        delta_savings > DELTA_SEED_MIN_SAVINGS && delta_cost.saturating_mul(2) < scratch_cost;
                    if delta_wins_decisively && cache_data.mask.len() == buf.len() {
                        let zero_start = profile.as_ref().map(|_| Instant::now());
                        buf.copy_from_slice(&cache_data.mask);
                        if let (Some(profile), Some(start)) = (profile.as_mut(), zero_start) {
                            profile.finalize_zero_ns += elapsed_ns(start);
                        }

                        let dense_to_buf_start = profile.as_ref().map(|_| Instant::now());
                        let delta_replay = self.constraint.apply_internal_dense_delta_to_buf(
                            &cache_data.merged_dense,
                            &merged,
                            buf,
                        );
                        if let Some(profile) = profile.as_mut() {
                            profile.delta_replay = delta_replay;
                            profile.finalize_delta_replay = 1;
                            if delta_profile_enabled {
                                profile.delta_used_seed = 1;
                            }
                            if let Some(start) = dense_to_buf_start {
                                profile.finalize_dense_to_buf_ns += elapsed_ns(start);
                            }
                        }
                        use_delta_seed = true;
                    }
                }
            }
        }

        if !use_delta_seed {
            let dense_to_buf = if direct_finalized || !merged_has_leftovers {
                DenseToBufProfileStats::default()
            } else {
                let buf_zeroed = !direct_buf_dirty;

                if profile.is_some() {
                    let dense_to_buf_start = Instant::now();
                    let dense_to_buf = self
                        .constraint
                        .or_internal_dense_to_buf(&merged, buf, buf_zeroed);
                    if let Some(profile) = profile.as_mut() {
                        profile.finalize_dense_to_buf_ns += elapsed_ns(dense_to_buf_start);
                    }
                    dense_to_buf
                } else {
                    let fast_conversion_start =
                        mask_fast_conversion_profile_enabled().then(Instant::now);
                    self.constraint
                        .or_internal_dense_to_buf_fast(&merged, buf, buf_zeroed);
                    if let Some(start) = fast_conversion_start {
                        let merged_set_bits =
                            merged.iter().map(|word| word.count_ones() as u64).sum::<u64>();
                        emit_mask_fast_conversion_profile_line(&format!(
                            "[glrmask/debug][mask_fast_conversion] ns={} internal_set_bits={} buf_words={} direct_buf_used={} direct_buf_possible={}",
                            elapsed_ns(start),
                            merged_set_bits,
                            buf.len(),
                            direct_buf_used,
                            direct_buf_possible
                        ));
                    }
                    DenseToBufProfileStats::default()
                }
            };
            if let Some(profile) = profile.as_mut() {
                profile.finalize_scratch_rebuild = 1;
                profile.dense_to_buf = dense_to_buf;
            }
        }
        // NOTE: NEVER EVER add any post-filter here that rechecks candidate
        // mask tokens through commit semantics. If mask and commit disagree,
        // the bug is in the seed/DWA mask construction logic itself and must
        // be fixed there. Hiding the mismatch with a second-pass filter is not
        // allowed. This note is intentional and must NEVER EVER be removed.

        let eos_unchanged = reuse_existing_cache_dense
            && eos_mask_bit(buf, self.constraint.eos_token_id) == self.is_complete();
        if !eos_unchanged {
            let eos_start = profile.as_ref().map(|_| Instant::now());
            update_eos_mask(buf, self.constraint.eos_token_id, self.is_complete());
            if let (Some(profile), Some(start)) = (profile.as_mut(), eos_start) {
                profile.finalize_eos_ns += elapsed_ns(start);
            }
        }

        let cache_start = profile.as_ref().map(|_| Instant::now());
        if can_use_merged_cache {
            if reuse_existing_cache_dense {
                if eos_unchanged {
                    self.touch_mask_cache_generation();
                } else {
                    self.store_mask_cache_reuse_dense(buf);
                }
            } else {
                self.store_mask_cache(buf, &merged);
            }
        } else {
            self.store_mask_cache_reuse_dense(buf);
        }
        if let (Some(profile), Some(start)) = (profile.as_mut(), cache_start) {
            profile.finalize_cache_ns += elapsed_ns(start);
        }
        let queue_debug = queue.debug_stats();

        if let Some(profile) = profile.as_mut() {
            if let Some(start) = finalize_start {
                profile.finalize_ns += elapsed_ns(start);
            }
            profile.queue_pop_ns = queue.debug_stats().pop_total_ns;
            if let Some(start) = total_start {
                profile.total_ns = elapsed_ns(start);
            }

            let loop_decompose_ns = profile
                .loop_decompose_total_ns
                .saturating_sub(profile.loop_decompose_callback_ns);
            let enqueue_exclusive_ns = queue_debug
                .enqueue_total_ns
                .saturating_sub(queue_debug.fuse_total_ns);
            let accounted_ns = profile.seed_decompose_ns
                + profile.queue_pop_ns
                + loop_decompose_ns
                + profile.transition_lookup_ns
                + profile.transition_apply_ns
                + profile.token_accumulation_ns
                + enqueue_exclusive_ns
                + queue_debug.fuse_total_ns
                + profile.finalize_ns;
            let other_ns = profile.total_ns.saturating_sub(accounted_ns);
            let line = format!(
                "[glrmask/debug][mask_inner] queue_mode={:?} total_ns={} seed_decompose_ns={} queue_pop_ns={} loop_decompose_ns={} transition_lookup_ns={} transition_apply_ns={} transition_apply_intersect_ns={} transition_apply_gss_ns={} token_accumulation_ns={} enqueue_merge_ns={} queue_lookup_ns={} queue_merge_ns={} queue_insert_ns={} insert_without_merge_count={} fuse_ns={} finalize_ns={} finalize_zero_ns={} finalize_dense_to_buf_ns={} finalize_eos_ns={} finalize_cache_ns={} delta_prev_available={} delta_added_bits={} delta_removed_bits={} delta_unchanged_words={} delta_unchanged_bits={} delta_added_cost={} delta_removed_cost={} delta_copy_cost_words={} delta_scratch_estimated_cost={} delta_estimated_cost={} delta_estimated_savings={} delta_used_seed={} delta_added_word_group_hits={} delta_added_word_group_entries={} delta_removed_word_group_hits={} delta_removed_word_group_entries={} delta_added_byte_group_hits={} delta_added_byte_group_entries={} delta_removed_byte_group_hits={} delta_removed_byte_group_entries={} delta_added_token_iterations={} delta_added_token_entries={} delta_removed_token_iterations={} delta_removed_token_entries={} finalize_equal_dense_copy_seed={} finalize_delta_replay={} finalize_scratch_rebuild={} dense_words_visited={} dense_complement_path_used={} dense_normal_full_word_hits={} dense_normal_group_complement_hits={} dense_complement_full_word_hits={} dense_complement_full_byte_groups={} dense_complement_full_nibble_groups={} dense_complement_remaining_bits={} dense_normal_token_iterations={} dense_complement_token_iterations={} dense_normal_sparse_entries={} dense_normal_group_complement_sparse_entries={} dense_complement_sparse_entries={} dense_complement_heavy_dense_clears={} dense_complement_max_sparse_span={} dense_group_or_sparse_entries={} dense_group_andnot_sparse_entries={} dense_group_sparse_groups={} dense_group_sparse_total_entries={} dense_group_sparse_max_entries={} dense_group_dense_storage_words={} dense_raw_token_sparse_entries={} other_ns={} enqueue_calls={} merge_hits={} popped_items={} parser_dwa_transitions_enqueued={}",
                mask_queue_mode(),
                profile.total_ns,
                profile.seed_decompose_ns,
                profile.queue_pop_ns,
                loop_decompose_ns,
                profile.transition_lookup_ns,
                profile.transition_apply_ns,
                profile.transition_apply_intersect_ns,
                profile.transition_apply_gss_ns,
                profile.token_accumulation_ns,
                enqueue_exclusive_ns,
                queue_debug.lookup_total_ns,
                queue_debug.merge_total_ns,
                queue_debug.insert_total_ns,
                queue_debug.insert_without_merge_count,
                queue_debug.fuse_total_ns,
                profile.finalize_ns,
                profile.finalize_zero_ns,
                profile.finalize_dense_to_buf_ns,
                profile.finalize_eos_ns,
                profile.finalize_cache_ns,
                profile.delta_prev_available,
                profile.delta_added_bits,
                profile.delta_removed_bits,
                profile.delta_unchanged_words,
                profile.delta_unchanged_bits,
                profile.delta_added_cost,
                profile.delta_removed_cost,
                profile.delta_copy_cost_words,
                profile.delta_scratch_estimated_cost,
                profile.delta_estimated_cost,
                profile.delta_estimated_savings,
                profile.delta_used_seed,
                profile.delta_replay.added_word_group_hits,
                profile.delta_replay.added_word_group_entries,
                profile.delta_replay.removed_word_group_hits,
                profile.delta_replay.removed_word_group_entries,
                profile.delta_replay.added_byte_group_hits,
                profile.delta_replay.added_byte_group_entries,
                profile.delta_replay.removed_byte_group_hits,
                profile.delta_replay.removed_byte_group_entries,
                profile.delta_replay.added_token_iterations,
                profile.delta_replay.added_token_entries,
                profile.delta_replay.removed_token_iterations,
                profile.delta_replay.removed_token_entries,
                profile.finalize_equal_dense_copy_seed,
                profile.finalize_delta_replay,
                profile.finalize_scratch_rebuild,
                profile.dense_to_buf.dense_words_visited,
                profile.dense_to_buf.complement_path_used,
                profile.dense_to_buf.normal_full_word_hits,
                profile.dense_to_buf.normal_group_complement_hits,
                profile.dense_to_buf.complement_full_word_hits,
                profile.dense_to_buf.complement_full_byte_groups,
                profile.dense_to_buf.complement_full_nibble_groups,
                profile.dense_to_buf.complement_remaining_bits,
                profile.dense_to_buf.normal_token_iterations,
                profile.dense_to_buf.complement_token_iterations,
                profile.dense_to_buf.normal_sparse_entries,
                profile.dense_to_buf.normal_group_complement_sparse_entries,
                profile.dense_to_buf.complement_sparse_entries,
                profile.dense_to_buf.complement_heavy_dense_clears,
                profile.dense_to_buf.complement_max_sparse_span,
                profile.dense_to_buf.group_or_sparse_entries,
                profile.dense_to_buf.group_andnot_sparse_entries,
                self.constraint.word_group_sparse_masks.len(),
                self.constraint.word_group_sparse_total_entries,
                self.constraint.word_group_sparse_max_entries,
                self.constraint.word_group_sparse_masks.len() * self.constraint.mask_len(),
                self.constraint.internal_token_buf_flat.len(),
                other_ns,
                queue_debug.enqueue_calls,
                queue_debug.merge_hit_count,
                queue_debug.popped_items,
                queue_debug.parser_dwa_transitions_enqueued,
            );
            emit_mask_inner_profile_line(&line);
        }

        let returned_profile = profile.map(|profile| {
            let mut out = MaskProfile::from_parts(profile, *queue_debug, false, false);
            if out.total_ns == 0 {
                if let Some(start) = total_start {
                    out.total_ns = elapsed_ns(start);
                }
            }
            out
        });

        let mut scratch = self.mask_scratch.lock().unwrap();
        scratch.merged_dense = merged;
        scratch.chain_merged_dense.clear();

        returned_profile
    }

    pub fn mask(&self) -> Vec<u32> {
        let mut buf = vec![0u32; self.constraint.mask_len()];
        self.fill_mask(&mut buf);
        buf
    }

    pub(crate) fn prefill_mask_cache(&self) {
        let cache = self.mask_cache.lock().unwrap();
        if cache
            .as_ref()
            .is_some_and(|cache_data| cache_data.generation == self.generation)
        {
            return;
        }
        drop(cache);

        let mut buf = vec![0u32; self.constraint.mask_len()];
        self.fill_mask_uncached(&mut buf);
    }

    pub fn fill_mask(&self, buf: &mut [u32]) {
        if self.try_fill_mask_from_cache(buf) {
            return;
        }

        self.fill_mask_uncached(buf);
    }

    pub fn fill_mask_timed_ns(&self, buf: &mut [u32]) -> u64 {
        let start = Instant::now();
        self.fill_mask(buf);
        start.elapsed().as_nanos() as u64
    }

    pub fn fill_mask_profiled(&self, buf: &mut [u32]) -> MaskProfile {
        let total_start = Instant::now();
        if self.try_fill_mask_from_cache(buf) {
            return MaskProfile {
                total_ns: elapsed_ns(total_start),
                cache_hit: 1,
                ..MaskProfile::default()
            };
        }

        self.fill_mask_uncached_maybe_profile(buf, true)
            .unwrap_or_else(|| MaskProfile {
                total_ns: elapsed_ns(total_start),
                ..MaskProfile::default()
            })
    }

    pub fn mask_game_fill_mask_and_internal_ids(&self, buf: &mut [u32]) -> Vec<u32> {
        if self.try_fill_mask_from_cache(buf) {
            let cache = self.mask_cache.lock().unwrap();
            if cache
                .as_ref()
                .is_some_and(|cache_data| !cache_data.merged_dense.is_empty())
            {
                drop(cache);
            } else {
                drop(cache);
                let _ = self.fill_mask_uncached_queue(buf, false, None);
            }
        } else {
            let _ = self.fill_mask_uncached_queue(buf, false, None);
        }

        let cache = self.mask_cache.lock().unwrap();
        let Some(cache_data) = cache.as_ref() else {
            return Vec::new();
        };

        let n_internal = self.constraint.internal_token_to_tokens.len();
        let mut out = Vec::new();
        for (word_idx, &word) in cache_data.merged_dense.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let internal = word_idx * 64 + bit;
                if internal < n_internal {
                    out.push(internal as u32);
                }
                word &= word - 1;
            }
        }
        out
    }
}
