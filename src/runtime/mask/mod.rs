pub(crate) mod profile;
pub(crate) mod queue;

use crate::automata::lexer::Lexer;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::grammar::flat::TerminalID;
use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_positive_label};
use crate::compiler::glr::parser::{
    lookahead_reduction_factor,
    lookahead_reduction_factor_row_subset,
    stack_may_advance_on,
    ParserGSS,
};
use crate::ds::bitset::BitSet;
use crate::ds::leveled_gss::{
    IndexedLeveledGss, IndexedLeveledGssNode, IndexedLowerIdentity, LeveledGSS, Merge,
};
use crate::ds::weight::Weight;
use crate::runtime::artifact::IndexedDagDenseMask;
use crate::runtime::constraint::{
    Constraint, DenseToBufProfileStats, RuntimeTokenSetRef, RuntimeWeightRef,
};
use crate::runtime::state::{ConstraintState, MaskCacheData, MaskScratch, ParserStateMap};
use range_set_blaze::RangeSetBlaze;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use self::profile::{
    elapsed_ns,
    emit_mask_fast_conversion_profile_line,
    emit_mask_inner_profile_line,
    emit_mask_queue_debug_line,
    initialize_runtime_config as initialize_mask_profile_config,
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
const MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY: usize = 64;
const MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS: usize = 128;
const MASK_SINGLE_PATH_DIRECT_INLINE_STACK_DEPTH: usize = 64;
const MASK_SINGLE_PATH_DIRECT_MAX_PLAN_OPS: usize = 1024;
// Below this much parser-stack work, grouping and compiling plans costs more
// than replaying the stacks directly even when a duplicate exists.
const MASK_SINGLE_PATH_DIRECT_MIN_PLAN_STACK_VALUES: usize = 128;
// Count concrete stack values only after the path-count gate. Ambiguous lexer
// frontiers commonly carry the same small parser language under several
// tokenizer states; 1,024 still bounds the direct walk tightly while admitting
// the bounded 33-64-path frontiers already supported by commit.
// This keeps the exact lexer/parser relation flat instead of forcing indexed-DAG
// construction after a successful bounded commit.
const MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_STACK_VALUES: usize = 1024;
const MASK_SINGLE_PATH_DIRECT_TWO_PASS_MIN_STATE_COUNT: usize =
    MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS / 2;

#[inline]
fn set_original_mask_bit(buf: &mut [u32], token_id: u32) {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    if let Some(slot) = buf.get_mut(word) {
        *slot |= 1u32 << bit;
    }
}

#[inline(always)]
fn original_mask_contains(buf: &[u32], token_id: u32) -> bool {
    buf.get(token_id as usize / 32)
        .is_some_and(|word| word & (1u32 << (token_id % 32)) != 0)
}


fn exact_component_trigger_accepted_weight(
    constraint: &Constraint,
    dwa: &crate::automata::weighted_u32::dwa::DWA,
    top_first: &[u32],
) -> Weight {
    let mut ops = crate::ds::weight::ScopedWeightOpCache::default();
    let mut state_id = dwa.start_state();
    let mut path_weight = Weight::all();
    let mut accepted = Weight::empty();
    let accumulate = |state_id: u32,
                      path_weight: &Weight,
                      accepted: &mut Weight,
                      ops: &mut crate::ds::weight::ScopedWeightOpCache| {
        if let Some(final_weight) = dwa
            .states()
            .get(state_id as usize)
            .and_then(|state| state.final_weight.as_ref())
        {
            let contribution = ops.intersection(path_weight, final_weight);
            if !contribution.is_empty() {
                *accepted = ops.union(accepted, &contribution);
            }
        }
    };
    accumulate(state_id, &path_weight, &mut accepted, &mut ops);
    for &parser_state in top_first {
        let Some(state) = dwa.states().get(state_id as usize) else {
            break;
        };
        let positive = encode_positive_label(parser_state);
        let transition = state
            .transitions
            .get(&positive)
            .or_else(|| {
                constraint
                    .parser_state_domain_label(parser_state)
                    .and_then(|label| state.transitions.get(&label))
            })
            .or_else(|| state.transitions.get(&DEFAULT_LABEL));
        let Some((target, edge_weight)) = transition else {
            break;
        };
        path_weight = ops.intersection(&path_weight, edge_weight);
        if path_weight.is_empty() {
            break;
        }
        state_id = *target;
        accumulate(state_id, &path_weight, &mut accepted, &mut ops);
    }
    accepted
}

fn single_path_direct_stack_work(
    stack_lengths: impl IntoIterator<Item = usize>,
) -> Option<usize> {
    let mut total = 0usize;
    for len in stack_lengths {
        total = total.saturating_add(len);
        if total > MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_STACK_VALUES {
            return None;
        }
    }
    Some(total)
}

#[inline]
fn single_path_direct_plan_reuse_dominates(
    path_count: usize,
    total_stack_values: usize,
    repeated_stack_values: usize,
) -> bool {
    path_count >= 3
        && total_stack_values >= MASK_SINGLE_PATH_DIRECT_MIN_PLAN_STACK_VALUES
        && repeated_stack_values
            > total_stack_values.saturating_sub(repeated_stack_values)
}

#[derive(Clone, Copy)]
enum SinglePathDirectPlanOp<'a> {
    Merge(RuntimeWeightRef<'a>),
    Intersect(RuntimeWeightRef<'a>),
}

#[derive(Clone, Copy)]
struct SinglePathDirectStackPlan {
    representative_path: usize,
    stack_fingerprint: u64,
    ops_start: usize,
    ops_end: usize,
}

#[inline]
fn single_path_direct_stack_fingerprint(stack: &[u32]) -> u64 {
    // FNV-1a is sufficient here: equality is still checked before sharing a
    // plan, so the fingerprint only avoids repeatedly comparing long stacks.
    let mut fingerprint = 0xcbf29ce484222325u64;
    for &state in stack {
        fingerprint ^= u64::from(state);
        fingerprint = fingerprint.wrapping_mul(0x100000001b3);
    }
    fingerprint ^ (stack.len() as u64).wrapping_mul(0x9e3779b97f4a7c15)
}

fn materialize_single_path_seed_intersection(
    base: &[u64],
    dense: &mut Vec<u64>,
    internal_tsid: u32,
    weight: RuntimeWeightRef<'_>,
    constraint: &Constraint,
) -> bool {
    debug_assert!(!weight.is_full());
    let Some(token_set) = weight.token_set_for_tsid(internal_tsid) else {
        dense.clear();
        return false;
    };

    dense.clear();
    dense.resize(base.len(), 0);
    if let Some(mask) = constraint.runtime_token_set_dense_mask(token_set) {
        let mut any = false;
        for (idx, dense_word) in dense.iter_mut().enumerate() {
            *dense_word = base[idx] & mask.get(idx).copied().unwrap_or(0);
            any |= *dense_word != 0;
        }
        return any;
    }

    let mut any = false;
    DenseMaskAcc::for_each_runtime_token_range_word(token_set, base.len(), |word_idx, token_mask| {
        let word = base[word_idx] & token_mask;
        dense[word_idx] |= word;
        any |= word != 0;
    });
    any
}

pub(crate) fn indexed_dag_mask_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_ENABLE_INDEXED_DAG_MASK")
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
    })
}

fn indexed_dag_mask_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_INDEXED_DAG_MASK").is_some()
}

fn dynamic_mask_equivalence_assert_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_ASSERT_DYNAMIC_MASK_EQUIVALENCE")
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
    })
}

pub(crate) fn initialize_runtime_config() {
    let _ = dynamic_mask_equivalence_assert_enabled();
    let _ = indexed_dag_mask_enabled();
    initialize_mask_profile_config();
    let _ = mask_queue_mode();
}

fn assert_dynamic_mask_equivalence(state: &ConstraintState<'_>, static_mask: &[u32]) {
    if !dynamic_mask_equivalence_assert_enabled() {
        return;
    }

    let mut dynamic_mask = vec![0u32; state.constraint.mask_len()];
    state.fill_mask_dynamic(&mut dynamic_mask);
    if static_mask == dynamic_mask {
        return;
    }

    let mut differing_token_ids = Vec::new();
    let mut differing_count = 0usize;
    for (word_index, (&static_word, &dynamic_word)) in
        static_mask.iter().zip(&dynamic_mask).enumerate()
    {
        let mut differing_bits = static_word ^ dynamic_word;
        while differing_bits != 0 {
            let bit = differing_bits.trailing_zeros() as usize;
            differing_count += 1;
            if differing_token_ids.len() < 64 {
                differing_token_ids.push(word_index * 32 + bit);
            }
            differing_bits &= differing_bits - 1;
        }
    }

    panic!(
        "dynamic/static mask mismatch at generation {}: differing_tokens={} first_differing_token_ids={:?} parser_state_keys={:?}",
        state.generation,
        differing_count,
        differing_token_ids,
        state.state.keys().copied().collect::<Vec<_>>(),
    );
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum RuntimeTokenSetKey {
    Materialized(usize),
    PackedDwa(u32),
    PackedPool(u32),
}

impl RuntimeTokenSetKey {
    #[inline]
    fn from_ref(token_set: RuntimeTokenSetRef<'_>) -> Self {
        if let Some(key) = token_set.materialized_key() {
            Self::Materialized(key)
        } else if let Some(id) = token_set.packed_id() {
            Self::PackedDwa(id)
        } else {
            Self::PackedPool(
                token_set
                    .packed_pool_id()
                    .expect("runtime token set has identity"),
            )
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct DenseTokenSetIntersectionKey {
    tsid: u32,
    dense: usize,
    dense_len: usize,
    token_set: RuntimeTokenSetKey,
}

type DenseTokenSetIntersectionSmallCache =
    SmallVec<[(Arc<[u64]>, RuntimeTokenSetKey, Option<Arc<[u64]>>); 8]>;

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

    fn from_dense_arc_for_tsids(tsids: &[u32], dense: Arc<[u64]>) -> Option<Self> {
        if tsids.is_empty() || dense.iter().all(|&word| word == 0) {
            return None;
        }

        let mut entries = SmallVec::with_capacity(tsids.len());
        for &tsid in tsids {
            entries.push((tsid, Arc::clone(&dense)));
        }
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

    fn for_each_runtime_token_range_word<F>(
        tokens: RuntimeTokenSetRef<'_>,
        word_limit: usize,
        mut f: F,
    ) where
        F: FnMut(usize, u64),
    {
        if word_limit == 0 {
            return;
        }
        let max_token_exclusive = word_limit.saturating_mul(64);
        if max_token_exclusive == 0 {
            return;
        }
        tokens.for_each_range(|start, end| {
            let lo = start as usize;
            if lo >= max_token_exclusive {
                return;
            }
            let hi = (end as usize).min(max_token_exclusive - 1);
            if lo > hi {
                return;
            }
            let word_lo = lo / 64;
            let word_hi = hi / 64;
            for word_idx in word_lo..=word_hi {
                let lo_bit = if word_idx == word_lo { lo % 64 } else { 0 };
                let hi_bit = if word_idx == word_hi { hi % 64 } else { 63 };
                f(word_idx, Self::bit_range_mask(lo_bit, hi_bit));
            }
        });
    }

    fn intersect_dense_with_runtime_token_set(
        dense: &[u64],
        token_set: RuntimeTokenSetRef<'_>,
        precomputed: &DenseTokenMaskCache,
    ) -> Option<Arc<[u64]>> {
        if let Some(key) = token_set.materialized_key() {
            if let Some(mask) = precomputed.get(&key) {
                let mut out = vec![0u64; dense.len()];
                let mut any = false;
                for i in 0..dense.len() {
                    let word = dense[i] & mask.get(i).copied().unwrap_or(0);
                    any |= word != 0;
                    out[i] = word;
                }
                return any.then(|| out.into());
            }
        }
        let mut out = vec![0u64; dense.len()];
        let mut any = false;
        Self::for_each_runtime_token_range_word(token_set, dense.len(), |word_idx, token_mask| {
            let word = dense[word_idx] & token_mask;
            if word != 0 {
                out[word_idx] |= word;
                any = true;
            }
        });
        any.then(|| out.into())
    }

    fn or_dense_and_runtime_token_set_into(
        dense: &[u64],
        token_set: RuntimeTokenSetRef<'_>,
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
    ) {
        if let Some(key) = token_set.materialized_key() {
            if let Some(mask) = precomputed.get(&key) {
                let n = dense.len().min(mask.len()).min(merged.len());
                for i in 0..n {
                    merged[i] |= dense[i] & mask[i];
                }
                return;
            }
        }
        let word_limit = dense.len().min(merged.len());
        Self::for_each_runtime_token_range_word(token_set, word_limit, |word_idx, token_mask| {
            merged[word_idx] |= dense[word_idx] & token_mask;
        });
    }

    fn intersect_with_runtime_weight_reuse(
        &self,
        weight: RuntimeWeightRef<'_>,
        precomputed: &DenseTokenMaskCache,
    ) -> Option<Self> {
        if self.is_empty() {
            return None;
        }
        if weight.is_full() {
            return Some(self.clone());
        }
        let mut entries = SmallVec::new();
        for (tsid, dense) in &self.0 {
            let Some(token_set) = weight.token_set_for_tsid(*tsid) else {
                continue;
            };
            if let Some(intersection) =
                Self::intersect_dense_with_runtime_token_set(dense, token_set, precomputed)
            {
                entries.push((*tsid, intersection));
            }
        }
        (!entries.is_empty()).then_some(Self(entries))
    }

    fn intersect_with_runtime_weight_small_cached(
        &self,
        weight: RuntimeWeightRef<'_>,
        precomputed: &DenseTokenMaskCache,
        cache: &mut DenseTokenSetIntersectionSmallCache,
    ) -> Option<Self> {
        if self.is_empty() {
            return None;
        }
        if weight.is_full() {
            return Some(self.clone());
        }
        let mut result = SmallVec::new();
        for (tsid, dense) in &self.0 {
            let Some(token_set) = weight.token_set_for_tsid(*tsid) else {
                continue;
            };
            let token_key = RuntimeTokenSetKey::from_ref(token_set);
            let cached = cache.iter().find(|(cached_dense, cached_token_set, _)| {
                Arc::ptr_eq(cached_dense, dense) && *cached_token_set == token_key
            });
            let intersection = if let Some((_, _, result)) = cached {
                result.clone()
            } else {
                let result = Self::intersect_dense_with_runtime_token_set(
                    dense,
                    token_set,
                    precomputed,
                );
                if cache.len() < cache.inline_size() {
                    cache.push((Arc::clone(dense), token_key, result.clone()));
                }
                result
            };
            if let Some(intersection) = intersection {
                result.push((*tsid, intersection));
            }
        }
        (!result.is_empty()).then_some(Self(result))
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
            let Some(token_set) = weight.token_set_for_tsid_ref(*tsid) else {
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
            let Some(token_set) = weight.token_set_for_tsid_ref(*tsid) else {
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
            token_set: RuntimeTokenSetKey::Materialized(Arc::as_ptr(token_set) as usize),
        };
        if let Some(cached) = cache.get(&key) {
            return cached.clone();
        }
        if let RuntimeTokenSetKey::Materialized(token_set_key) = key.token_set {
            if let Some(mask) = precomputed.get(&token_set_key) {
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
        }
        let result = Self::intersect_dense_with_token_set(dense, token_set, precomputed);
        cache.insert(key, result.clone());
        result
    }

    fn intersect_with_weight_small_cached(
        &self,
        weight: &Weight,
        precomputed: &DenseTokenMaskCache,
        cache: &mut DenseTokenSetIntersectionSmallCache,
    ) -> Option<Self> {
        if self.is_empty() {
            return None;
        }
        if weight.is_full() {
            return Some(self.clone());
        }

        let mut result = SmallVec::new();
        for (tsid, dense) in &self.0 {
            let Some(token_set) = weight.token_set_for_tsid_ref(*tsid) else {
                continue;
            };
            if let Some(intersection) = Self::intersect_dense_with_token_set_small_cached(
                dense,
                token_set,
                precomputed,
                cache,
            ) {
                result.push((*tsid, intersection));
            }
        }

        (!result.is_empty()).then_some(Self(result))
    }

    fn intersect_dense_with_token_set_small_cached(
        dense: &Arc<[u64]>,
        token_set: &Arc<RangeSetBlaze<u32>>,
        precomputed: &DenseTokenMaskCache,
        cache: &mut DenseTokenSetIntersectionSmallCache,
    ) -> Option<Arc<[u64]>> {
        let token_set_key = RuntimeTokenSetKey::Materialized(Arc::as_ptr(token_set) as usize);
        if let Some((_, _, result)) = cache.iter().find(|(cached_dense, cached_token_set, _)| {
            Arc::ptr_eq(cached_dense, dense) && *cached_token_set == token_set_key
        }) {
            return result.clone();
        }

        let result = Self::intersect_dense_with_token_set(dense, token_set, precomputed);
        if cache.len() < cache.inline_size() {
            cache.push((Arc::clone(dense), token_set_key, result.clone()));
        }
        result
    }

    fn intersect_dense_with_token_set_reuse(
        dense: &Arc<[u64]>,
        token_set: &Arc<RangeSetBlaze<u32>>,
        precomputed: &DenseTokenMaskCache,
    ) -> Option<Arc<[u64]>> {
        let key = Arc::as_ptr(token_set) as usize;
        if let Some(mask) = precomputed.get(&key) {
            let mut any = false;
            let mut out: Option<Vec<u64>> = None;
            for index in 0..dense.len() {
                let word = dense[index] & mask.get(index).copied().unwrap_or(0);
                any |= word != 0;
                if let Some(out) = out.as_mut() {
                    out.push(word);
                } else if word != dense[index] {
                    let mut changed = Vec::with_capacity(dense.len());
                    changed.extend_from_slice(&dense[..index]);
                    changed.push(word);
                    out = Some(changed);
                }
            }
            return if !any {
                None
            } else if let Some(out) = out {
                Some(out.into())
            } else {
                Some(Arc::clone(dense))
            };
        }

        let mut out = vec![0u64; dense.len()];
        let mut any = false;
        Self::for_each_token_range_word(token_set, dense.len(), |word_index, token_mask| {
            let word = dense[word_index] & token_mask;
            out[word_index] |= word;
            any |= word != 0;
        });
        if !any {
            return None;
        }
        if out.as_slice() == dense.as_ref() {
            Some(Arc::clone(dense))
        } else {
            Some(out.into())
        }
    }

    fn intersect_with_weight_reuse(
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
        let mut entries = SmallVec::new();
        for (tsid, dense) in &self.0 {
            let Some(token_set) = weight.token_set_for_tsid_ref(*tsid) else {
                continue;
            };
            if let Some(intersection) =
                Self::intersect_dense_with_token_set_reuse(dense, token_set, precomputed)
            {
                entries.push((*tsid, intersection));
            }
        }
        (!entries.is_empty()).then_some(Self(entries))
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
            let Some(token_set) = weight.token_set_for_tsid_ref(*tsid) else {
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

    fn merge_in_place(&mut self, other: &Self) {
        if other.is_empty() {
            return;
        }
        if self.is_empty() {
            *self = other.clone();
            return;
        }
        for (other_tsid, other_dense) in &other.0 {
            match self
                .0
                .iter()
                .position(|(existing_tsid, _)| existing_tsid == other_tsid)
            {
                Some(index) => {
                    let dense = &mut self.0[index].1;
                    if Arc::ptr_eq(dense, other_dense) {
                        continue;
                    }
                    if dense.len() == other_dense.len() {
                        let dense = Arc::make_mut(dense);
                        for (word, other_word) in dense.iter_mut().zip(other_dense.iter()) {
                            *word |= *other_word;
                        }
                    } else {
                        let len = dense.len().max(other_dense.len());
                        let mut combined = vec![0u64; len];
                        for (i, word) in dense.iter().enumerate() {
                            combined[i] |= *word;
                        }
                        for (i, word) in other_dense.iter().enumerate() {
                            combined[i] |= *word;
                        }
                        *dense = combined.into();
                    }
                }
                None => {
                    let insert_at = self
                        .0
                        .iter()
                        .position(|(existing_tsid, _)| existing_tsid > other_tsid)
                        .unwrap_or(self.0.len());
                    self.0
                        .insert(insert_at, (*other_tsid, Arc::clone(other_dense)));
                }
            }
        }
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
            let Some(token_set) = final_weight.token_set_for_tsid_ref(*tsid) else {
                continue;
            };

            Self::or_dense_and_token_set_into(dense, token_set, precomputed, merged);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        exact_component_trigger_accepted_weight, single_path_direct_plan_reuse_dominates,
        single_path_direct_stack_work,
        DenseMaskAcc,
        DenseTokenMaskCache,
        DenseTokenSetIntersectionSmallCache,
        MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY,
        MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_STACK_VALUES,
    };
    use crate::automata::lexer::Lexer;
    use crate::compiler::glr::accumulator::TerminalsDisallowed;
    use crate::compiler::glr::parser::ParserGSS;
    use crate::{Constraint as Constraint, Vocab};
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

    fn mask_contains(mask: &[u32], token: u32) -> bool {
        mask.get(token as usize / 32)
            .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
    }


    fn exact_start_trigger_contains(constraint: &Constraint, token: u32) -> bool {
        let crate::runtime::BoundaryTrigger::Exact(dwa) = &constraint.boundary_trigger else {
            panic!("constraint does not carry an Exact boundary trigger");
        };
        let state = constraint.start();
        let mut accepted = false;
        for (&tokenizer_state, gss) in state.state.iter() {
            let complete = gss.for_each_stack_top_first_bounded(128, |top_first, _| {
                let weight =
                    exact_component_trigger_accepted_weight(constraint, dwa, top_first);
                if weight.tokens_for_tsid(tokenizer_state).contains(token) {
                    accepted = true;
                }
            });
            assert!(complete, "tiny trigger test GSS traversal must complete");
        }
        accepted
    }

    #[test]
    fn exact_finish_trigger_requires_a_proper_internal_offset() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"xy".to_vec()),
            (2, b"xx".to_vec()),
            (3, b"yx".to_vec()),
        ]);
        let mut constraint = Constraint::from_ebnf(r#"start ::= "x""#, &vocab).unwrap();
        constraint.build_exact_boundary_trigger().unwrap();

        assert!(exact_start_trigger_contains(&constraint, 1));
        assert!(exact_start_trigger_contains(&constraint, 2));
        assert!(
            !exact_start_trigger_contains(&constraint, 0),
            "finish exactly at model-token end is not an internal trigger",
        );
        assert!(!exact_start_trigger_contains(&constraint, 3));
    }

    #[test]
    fn exact_entry_trigger_requires_placeholder_readiness_after_prefix() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"xy".to_vec()),
            (2, b"y".to_vec()),
        ]);
        let mut parent = Constraint::from_glrm_grammar_with_unbound_subgrammars_bindings_and_end_tokens(
            "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            &vocab,
            &[],
            &[],
        )
        .unwrap();
        parent.build_exact_boundary_trigger().unwrap();

        assert!(exact_start_trigger_contains(&parent, 1));
        assert!(
            !exact_start_trigger_contains(&parent, 0),
            "placeholder reached only at model-token end is handled by next-call control closure",
        );
        assert!(!exact_start_trigger_contains(&parent, 2));
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

    #[test]
    fn small_intersection_cache_reuses_exact_result() {
        let dense: Arc<[u64]> = Arc::from([0b1011_u64, 0b0101]);
        let token_set = Arc::new(RangeSetBlaze::from_iter([0_u32..=127]));
        let precomputed = precomputed_for(&token_set, Arc::from([0b0011_u64, 0b0100]));
        let mut cache = DenseTokenSetIntersectionSmallCache::new();

        let first = DenseMaskAcc::intersect_dense_with_token_set_small_cached(
            &dense,
            &token_set,
            &precomputed,
            &mut cache,
        )
        .unwrap();
        let second = DenseMaskAcc::intersect_dense_with_token_set_small_cached(
            &dense,
            &token_set,
            &precomputed,
            &mut cache,
        )
        .unwrap();

        assert_eq!(&*first, &[0b0011_u64, 0b0100]);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(cache.len(), 1);
        assert!(Arc::ptr_eq(&cache[0].0, &dense));
    }

    #[test]
    fn empty_possible_matches_uses_exact_seed_exclusion_scan() {
        let mut constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                t B ::= "b";
                nt start ::= A | B;
            "#,
            &Vocab::new(
                vec![
                    (0, b"a".to_vec()),
                    (1, b"b".to_vec()),
                    (2, b"ab".to_vec()),
                ]),
        )
        .expect("test constraint should compile");
        let terminal_a = constraint
            .terminal_display_names
            .iter()
            .position(|name| name == "A")
            .expect("A terminal should have a display name") as u32;
        constraint.possible_matches.clear();
        constraint.possible_matches_complete = false;

        let tokenizer_state = constraint.tokenizer.initial_state();
        let disallowed = TerminalsDisallowed::new().with_insert(tokenizer_state, terminal_a);
        let mut state = constraint.start_dynamic();
        state.state = crate::runtime::state::ParserStateMap::singleton(
            tokenizer_state,
            ParserGSS::from_stacks(&[(vec![0u32], disallowed)]),
        );

        let mut expected = vec![0u32; constraint.mask_len()];
        state.fill_mask_dynamic(&mut expected);
        let mut actual = vec![0u32; constraint.mask_len()];
        state.fill_mask(&mut actual);
        assert_eq!(actual, expected);

        let loaded = Constraint::load(&constraint.save()).expect("empty-PM constraint should roundtrip");
        assert!(loaded.possible_matches.is_empty());
        let mut loaded_state = loaded.start_dynamic();
        let loaded_tokenizer_state = loaded.tokenizer.initial_state();
        let loaded_disallowed =
            TerminalsDisallowed::new().with_insert(loaded_tokenizer_state, terminal_a);
        loaded_state.state = crate::runtime::state::ParserStateMap::singleton(
            loaded_tokenizer_state,
            ParserGSS::from_stacks(&[(vec![0u32], loaded_disallowed)]),
        );
        let mut loaded_expected = vec![0u32; loaded.mask_len()];
        loaded_state.fill_mask_dynamic(&mut loaded_expected);
        let mut loaded_actual = vec![0u32; loaded.mask_len()];
        loaded_state.fill_mask(&mut loaded_actual);
        assert_eq!(loaded_actual, loaded_expected);
    }

    #[test]
    fn literal_choice_terminal_mask_keeps_multibyte_prefix_tokens() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"S".to_vec()),
            (2, b"Se".to_vec()),
            (3, b"Service".to_vec()),
            (4, b"I".to_vec()),
            (5, b"In".to_vec()),
            (6, b"Independent".to_vec()),
            (7, b" provider assertion\"".to_vec()),
            (8, b" validation of assertion\"".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                fa assurance_body ::= {
                    start 0;
                    accept 1;
                    0 -- "\"" ("Service provider assertion\"" | "Independent validation of assertion\"") --> 1;
                };
                nt start ::= assurance_body;
            "#,
            &vocab,
        )
        .expect("literal-choice terminal should compile");

        let mut state = constraint.start();
        state.commit_token(0).expect("opening quote should commit");

        let mut static_mask = vec![0u32; constraint.mask_len()];
        state.fill_mask(&mut static_mask);
        let mut dynamic_mask = vec![0u32; constraint.mask_len()];
        state.fill_mask_dynamic(&mut dynamic_mask);

        assert_eq!(static_mask, dynamic_mask);
        for token in [1, 2, 3, 4, 5, 6] {
            assert!(
                mask_contains(&static_mask, token),
                "literal prefix token {token} should be accepted"
            );
        }
    }

    #[test]
    fn direct_mask_spills_past_the_inline_path_capacity() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                nt start ::= A;
            "#,
            &vocab,
        )
        .expect("single-terminal grammar should compile");
        let mut state = constraint.start();
        let (tokenizer_state, parser_gss) = state.state.entries[0].clone();
        state.state.entries.clear();
        for _ in 0..=MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY {
            state
                .state
                .insert_flat_alternative(tokenizer_state, parser_gss.clone());
        }
        assert_eq!(
            state.state.len(),
            MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY + 1,
        );

        let mut direct = vec![0u32; constraint.mask_len()];
        assert!(state.try_fill_mask_single_path_direct(&mut direct));
        let mut dynamic = vec![0u32; constraint.mask_len()];
        state.fill_mask_dynamic(&mut dynamic);
        assert_eq!(direct, dynamic);
        assert!(mask_contains(&direct, 0));
        assert!(!mask_contains(&direct, 1));
    }

    #[test]
    fn stack_plan_admission_depends_on_reuse_not_the_old_path_boundary() {
        for path_count in [31, 32, 33, 64] {
            assert!(!single_path_direct_plan_reuse_dominates(
                path_count,
                path_count * 8,
                0,
            ));
        }
        assert!(!single_path_direct_plan_reuse_dominates(2, 16, 8));
        assert!(!single_path_direct_plan_reuse_dominates(3, 24, 16));
        assert!(!single_path_direct_plan_reuse_dominates(9, 72, 56));
        assert!(single_path_direct_plan_reuse_dominates(9, 144, 112));
    }

    #[test]
    fn single_path_direct_stack_work_budget_accepts_shallow_ambiguity() {
        assert_eq!(single_path_direct_stack_work([8; 10]), Some(80));
        assert_eq!(
            single_path_direct_stack_work([MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_STACK_VALUES]),
            Some(MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_STACK_VALUES),
        );
        assert_eq!(
            single_path_direct_stack_work([
                MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_STACK_VALUES,
                1,
            ]),
            None,
        );
    }

    #[test]
    fn indexed_dag_mask_matches_dynamic_on_all_small_reachable_states() {
        use std::collections::BTreeSet;
        fn allowed(mask: &[u32], token: u32) -> bool {
            mask.get(token as usize / 32)
                .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
        }

        let vocab = Vocab::new(
            ["a", "b", "ab", "ba", "aa", "bb"]
                .into_iter()
                .enumerate()
                .map(|(id, bytes)| (id as u32, bytes.as_bytes().to_vec()))
                .collect(),
        );
        let grammars = [
            r#"
                start start;
                t A ::= "a" | "ab";
                t B ::= "a" | "ba";
                nt item ::= A | B;
                nt start ::= item item? item?;
            "#,
            r#"
                start start;
                t A ::= "a"+;
                t B ::= "a"+ "b"?;
                nt start ::= A A | B B | A B | B A;
            "#,
        ];

        let mut ambiguous_states = 0usize;
        for grammar in grammars {
            let constraint = Constraint::from_glrm_grammar(grammar, &vocab)
                .expect("small indexed-DAG parity grammar should compile");
            let mut frontier = vec![(constraint.start(), Vec::<u32>::new())];
            let mut seen = BTreeSet::new();
            for depth in 0..=3 {
                let mut next = Vec::new();
                for (state, path) in frontier {
                    let key = state.debug_parser_stacks();
                    if !seen.insert(format!("{key:?}")) {
                        continue;
                    }
                    let mut expected = vec![0u32; constraint.mask_len()];
                    state.fill_mask_dynamic(&mut expected);
                    if state.has_parser_ambiguity() {
                        ambiguous_states += 1;
                        let mut actual = vec![0u32; constraint.mask_len()];
                        assert!(state.fill_mask_indexed_dag(&mut actual, true));
                        assert_eq!(
                            actual, expected,
                            "indexed/dynamic mask mismatch depth={depth} path={path:?} grammar={grammar}"
                        );
                    }
                    if depth == 3 {
                        continue;
                    }
                    for (token, bytes) in constraint.token_bytes_iter() {
                        if !allowed(&expected, token) {
                            continue;
                        }
                        let mut advanced = state.clone();
                        advanced
                            .commit_bytes(bytes)
                            .expect("dynamically admitted token must commit");
                        let mut next_path = path.clone();
                        next_path.push(token);
                        next.push((advanced, next_path));
                    }
                }
                frontier = next;
            }
        }
        assert!(ambiguous_states > 0, "test must exercise indexed ambiguous states");
    }

    #[test]
    fn indexed_dag_cache_stays_exact_across_commits_and_restore() {
        let vocab = Vocab::new(
            ["a", "b", "ab", "ba", "aa", "bb"]
                .into_iter()
                .enumerate()
                .map(|(id, bytes)| (id as u32, bytes.as_bytes().to_vec()))
                .collect(),
        );
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a"+;
                t B ::= "a"+ "b"?;
                nt item ::= A | B;
                nt start ::= item item? item? item?;
            "#,
            &vocab,
        )
        .expect("persistent indexed-DAG parity grammar should compile");
        let mut state = constraint.start();
        let sequence: [&[u8]; 4] = [b"a", b"a", b"a", b"b"];
        let mut checkpoint = None;

        for (index, bytes) in sequence.into_iter().enumerate() {
            let mut expected = vec![0u32; constraint.mask_len()];
            state.fill_mask_dynamic(&mut expected);
            if state.has_parser_ambiguity() {
                let mut first = vec![0u32; constraint.mask_len()];
                let mut second = vec![0u32; constraint.mask_len()];
                assert!(state.fill_mask_indexed_dag(&mut first, true));
                assert!(state.fill_mask_indexed_dag(&mut second, true));
                assert_eq!(first, expected);
                assert_eq!(second, expected, "same-state cache hit changed the mask");
            }
            state
                .commit_bytes(bytes)
                .expect("test sequence should remain valid");
            if index == 1 {
                checkpoint = Some(state.clone());
            }
        }

        state = checkpoint.expect("checkpoint should be captured");
        let mut expected = vec![0u32; constraint.mask_len()];
        state.fill_mask_dynamic(&mut expected);
        if state.has_parser_ambiguity() {
            let mut actual = vec![0u32; constraint.mask_len()];
            assert!(state.fill_mask_indexed_dag(&mut actual, true));
            assert_eq!(actual, expected, "restored indexed mask diverged");
        }
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

#[derive(Clone, PartialEq, Eq, Hash)]
struct DenseAccIdentity(SmallVec<[(u32, usize, usize); 2]>);

fn dense_acc_identity(accumulator: &DenseMaskAcc) -> DenseAccIdentity {
    DenseAccIdentity(
        accumulator
            .0
            .iter()
            .map(|(tsid, dense)| (*tsid, dense.as_ptr() as usize, dense.len()))
            .collect(),
    )
}

#[derive(Clone)]
struct SingleLowerMemoEntry {
    dwa_state: u32,
    tsid: u32,
    result: Option<Arc<[u64]>>,
}

#[derive(Clone)]
struct SingleSegmentMemoEntry {
    offset: usize,
    dwa_state: u32,
    tsid: u32,
    result: Option<Arc<[u64]>>,
}

struct SingleSourceMemo {
    source: IndexedLowerIdentity<u32>,
    last_seen_epoch: u64,
    lower: SmallVec<[SingleLowerMemoEntry; 8]>,
    segments: SmallVec<[SingleSegmentMemoEntry; 8]>,
}

#[derive(Default)]
pub(crate) struct IndexedDagMaskRuntime {
    epoch: u64,
    live_source_count: usize,
    lower_memo: FxHashMap<(u32, usize, u32), Option<DenseMaskAcc>>,
    segment_memo: FxHashMap<(usize, usize, u32, u32), Option<DenseMaskAcc>>,
    final_memo: FxHashMap<(u32, u32), Option<DenseMaskAcc>>,
    single_source_ids: FxHashMap<usize, u32>,
    single_sources: Vec<SingleSourceMemo>,
    lower_sources: FxHashMap<usize, IndexedLowerIdentity<u32>>,
    accumulators: Vec<DenseMaskAcc>,
    accumulator_ids: FxHashMap<DenseAccIdentity, u32>,
    index_nodes: Vec<IndexedLeveledGssNode<u32, DenseMaskAcc>>,
    index_roots: Vec<u32>,
    index_lower_ids: FxHashMap<usize, u32>,
    index_upper_ids: FxHashMap<usize, u32>,
}

impl IndexedDagMaskRuntime {
    const SOURCE_SLACK: usize = 64;
    const MAX_ACCUMULATORS: usize = 65_536;
    const MAX_RETAINED_INDEX_NODES: usize = 16_384;

    fn begin_mask(&mut self) {
        self.epoch = self.epoch.wrapping_add(1).max(1);
        self.live_source_count = 0;
        if self.accumulators.len() > Self::MAX_ACCUMULATORS {
            *self = Self {
                epoch: self.epoch,
                ..Self::default()
            };
        }
    }

    fn mark_source_live(&mut self, slot: u32) {
        let source = &mut self.single_sources[slot as usize];
        if source.last_seen_epoch != self.epoch {
            source.last_seen_epoch = self.epoch;
            self.live_source_count += 1;
        }
    }

    fn prune_stale_sources_if_needed(&mut self) {
        let threshold = self
            .live_source_count
            .saturating_mul(2)
            .saturating_add(Self::SOURCE_SLACK);
        if self.single_sources.len() <= threshold {
            return;
        }
        let oldest_kept = self.epoch.saturating_sub(1);
        self.single_sources
            .retain(|source| source.last_seen_epoch >= oldest_kept);
        self.single_source_ids.clear();
        let mut retained = FxHashSet::default();
        for (slot, source) in self.single_sources.iter().enumerate() {
            let ptr = source.source.ptr_key();
            retained.insert(ptr);
            self.single_source_ids.insert(
                ptr,
                u32::try_from(slot).expect("indexed mask source slots exceeded u32"),
            );
        }
        self.lower_sources.retain(|ptr, _| retained.contains(ptr));
        self.lower_memo
            .retain(|(_, ptr, _), _| retained.contains(ptr));
        self.segment_memo
            .retain(|(ptr, _, _, _), _| retained.contains(ptr));
    }

    fn retain_index_scratch(
        &mut self,
        dag: IndexedLeveledGss<u32, DenseMaskAcc>,
        roots: Vec<u32>,
        lower_ids: FxHashMap<usize, u32>,
        upper_ids: FxHashMap<usize, u32>,
    ) {
        if dag.nodes.len() > Self::MAX_RETAINED_INDEX_NODES {
            return;
        }
        self.index_nodes = dag.nodes;
        self.index_roots = roots;
        self.index_lower_ids = lower_ids;
        self.index_upper_ids = upper_ids;
    }
}

/// Exact denotational evaluator for the parser DWA over a weighted GSS DAG.
///
/// For DWA state `q`, GSS node `G`, and token accumulator `a`, the result is
///
/// `E(q, G, a) = union_{s in [[G]]} (a intersect W(q, s))`,
///
/// where `[[G]]` is the stack language denoted by the GSS node and `W(q, s)`
/// is the union of every accepting-prefix weight encountered while the parser
/// DWA reads stack `s` from the top downward.
///
/// The implementation is the structural recurrence of this definition:
///
/// * a branch is language union, so its result is bitmap union;
/// * an interface fixes the accumulator correlated with its lower language;
/// * a DWA edge `(q, x) -> (q', w)` contributes
///   `w intersect E(q', child, a)`;
/// * the current state's final weight contributes at every readable prefix;
/// * a segment is the unary instance of the same recurrence.
///
/// Exactness follows by induction on the acyclic indexed GSS DAG, using
/// distributivity of bitmap intersection over union. Memoization changes only
/// evaluation order; its key contains every semantic argument.
struct IndexedDagMaskEvaluator<'a, 'r> {
    constraint: &'a Constraint,
    dag: &'a IndexedLeveledGss<u32, DenseMaskAcc>,
    precomputed: &'a DenseTokenMaskCache,
    runtime: &'r mut IndexedDagMaskRuntime,
    source_slots: Vec<u32>,
    upper_memo: FxHashMap<(u32, u32), Option<DenseMaskAcc>>,
    all_upper_memo: FxHashMap<u32, Option<DenseMaskAcc>>,
    upper_calls: u64,
    upper_hits: u64,
    lower_calls: u64,
    lower_hits: u64,
    segment_calls: u64,
    segment_hits: u64,
    memo_result_entries: u64,
    memo_dense_words: u64,
    memo_nonzero_words: u64,
    memo_max_nonzero_words: u64,
}

impl<'a, 'r> IndexedDagMaskEvaluator<'a, 'r> {
    fn new(
        constraint: &'a Constraint,
        dag: &'a IndexedLeveledGss<u32, DenseMaskAcc>,
        precomputed: &'a DenseTokenMaskCache,
        runtime: &'r mut IndexedDagMaskRuntime,
    ) -> Self {
        let source_slots = dag
            .nodes
            .iter()
            .map(|node| match node {
                IndexedLeveledGssNode::LowerGeneral { source, .. }
                | IndexedLeveledGssNode::LowerSegment { source, .. } => {
                    let ptr = source.ptr_key();
                    if let Some(&slot) = runtime.single_source_ids.get(&ptr) {
                        runtime.mark_source_live(slot);
                        slot
                    } else {
                        let slot = u32::try_from(runtime.single_sources.len())
                            .expect("indexed mask source slots exceeded u32");
                        runtime.single_sources.push(SingleSourceMemo {
                            source: source.clone(),
                            last_seen_epoch: runtime.epoch,
                            lower: SmallVec::new(),
                            segments: SmallVec::new(),
                        });
                        runtime.single_source_ids.insert(ptr, slot);
                        runtime.live_source_count += 1;
                        slot
                    }
                }
                IndexedLeveledGssNode::UpperBranch { .. }
                | IndexedLeveledGssNode::Interface { .. } => u32::MAX,
            })
            .collect();
        Self {
            constraint,
            dag,
            precomputed,
            runtime,
            source_slots,
            upper_memo: FxHashMap::default(),
            all_upper_memo: FxHashMap::default(),
            upper_calls: 0,
            upper_hits: 0,
            lower_calls: 0,
            lower_hits: 0,
            segment_calls: 0,
            segment_hits: 0,
            memo_result_entries: 0,
            memo_dense_words: 0,
            memo_nonzero_words: 0,
            memo_max_nonzero_words: 0,
        }
    }

    fn accumulator_id(&mut self, accumulator: &DenseMaskAcc) -> u32 {
        let identity = dense_acc_identity(accumulator);
        if let Some(id) = self.runtime.accumulator_ids.get(&identity) {
            return *id;
        }
        let id = u32::try_from(self.runtime.accumulators.len())
            .expect("indexed DAG accumulator IDs exceeded u32");
        self.runtime.accumulators.push(accumulator.clone());
        self.runtime.accumulator_ids.insert(identity, id);
        id
    }

    fn merge_result(current: &mut Option<DenseMaskAcc>, incoming: Option<DenseMaskAcc>) {
        let Some(incoming) = incoming else {
            return;
        };
        match current {
            Some(existing) => existing.merge_in_place(&incoming),
            None => *current = Some(incoming),
        }
    }

    fn intersect_result(
        &self,
        result: Option<DenseMaskAcc>,
        weight: &Weight,
    ) -> Option<DenseMaskAcc> {
        result?.intersect_with_weight_reuse(weight, self.precomputed)
    }

    fn final_for_accumulator(
        &mut self,
        dwa_state: u32,
        accumulator_id: u32,
    ) -> Option<DenseMaskAcc> {
        let key = (dwa_state, accumulator_id);
        if let Some(cached) = self.runtime.final_memo.get(&key) {
            return cached.clone();
        }
        let result = self.constraint.parser_dwa().states()[dwa_state as usize]
            .final_weight
            .as_ref()
            .and_then(|weight| {
                self.runtime.accumulators[accumulator_id as usize]
                    .intersect_with_weight_reuse(weight, self.precomputed)
            });
        self.runtime.final_memo.insert(key, result.clone());
        result
    }

    fn all_accumulators_upper(&mut self, node: u32) -> Option<DenseMaskAcc> {
        if let Some(cached) = self.all_upper_memo.get(&node) {
            return cached.clone();
        }
        let indexed = self.dag.nodes[node as usize].clone();
        let mut out = None;
        match indexed {
            IndexedLeveledGssNode::UpperBranch { empty, children } => {
                Self::merge_result(&mut out, empty);
                for (_, child) in children {
                    Self::merge_result(&mut out, self.all_accumulators_upper(child));
                }
            }
            IndexedLeveledGssNode::Interface { accumulator, .. } => {
                out = Some(accumulator);
            }
            IndexedLeveledGssNode::LowerGeneral { .. }
            | IndexedLeveledGssNode::LowerSegment { .. } => {
                debug_assert!(false, "upper accumulator traversal reached lower node");
            }
        }
        self.all_upper_memo.insert(node, out.clone());
        out
    }

    fn transition(&self, dwa_state: u32, parser_state: u32) -> Option<(u32, Weight)> {
        let transitions = &self.constraint.dwa_fast_transitions[dwa_state as usize];
        self.constraint
            .fast_parser_dwa_transition(transitions, parser_state)
            .map(|(target, weight)| (target, weight.clone()))
    }

    fn eval_upper(&mut self, dwa_state: u32, node: u32) -> Option<DenseMaskAcc> {
        self.upper_calls += 1;
        let key = (dwa_state, node);
        if let Some(cached) = self.upper_memo.get(&key) {
            self.upper_hits += 1;
            return cached.clone();
        }
        let indexed = self.dag.nodes[node as usize].clone();
        let mut out = match &indexed {
            IndexedLeveledGssNode::UpperBranch { .. } => {
                let accumulator = self.all_accumulators_upper(node)?;
                let accumulator_id = self.accumulator_id(&accumulator);
                self.final_for_accumulator(dwa_state, accumulator_id)
            }
            IndexedLeveledGssNode::Interface { accumulator, lower } => {
                let accumulator_id = self.accumulator_id(accumulator);
                self.eval_lower(dwa_state, *lower, accumulator_id)
            }
            IndexedLeveledGssNode::LowerGeneral { .. }
            | IndexedLeveledGssNode::LowerSegment { .. } => {
                debug_assert!(false, "upper evaluation reached lower node");
                None
            }
        };
        if let IndexedLeveledGssNode::UpperBranch { children, .. } = indexed {
            for (parser_state, child) in children {
                let Some((target, weight)) = self.transition(dwa_state, parser_state) else {
                    continue;
                };
                let child_result = self.eval_upper(target, child);
                let child_result = self.intersect_result(child_result, &weight);
                Self::merge_result(&mut out, child_result);
            }
        }
        self.upper_memo.insert(key, out.clone());
        out
    }

    fn profile_memo_result(&mut self, result: &Option<DenseMaskAcc>) {
        if !indexed_dag_mask_profile_enabled() {
            return;
        }
        let Some(result) = result else {
            return;
        };
        self.memo_result_entries += 1;
        for (_, dense) in &result.0 {
            let nonzero = dense.iter().filter(|word| **word != 0).count() as u64;
            self.memo_dense_words += dense.len() as u64;
            self.memo_nonzero_words += nonzero;
            self.memo_max_nonzero_words = self.memo_max_nonzero_words.max(nonzero);
        }
    }

    fn single_dense_mask_for_weight(
        &self,
        weight: &Weight,
        tsid: u32,
    ) -> IndexedDagDenseMask {
        if weight.is_full() {
            return IndexedDagDenseMask::Full;
        }
        let Some(tokens) = weight.token_set_for_tsid_ref(tsid) else {
            return IndexedDagDenseMask::Empty;
        };
        let token_key = Arc::as_ptr(tokens) as usize;
        if let Some(dense) = self.precomputed.get(&token_key) {
            return Self::single_dense_transition_mask(Arc::clone(dense));
        }
        let mut dense = vec![0u64; self.constraint.internal_token_dense_words];
        DenseMaskAcc::for_each_token_range_word(tokens, dense.len(), |index, mask| {
            dense[index] |= mask;
        });
        Self::single_dense_transition_mask(dense.into())
    }

    fn single_dense_transition_mask(words: Arc<[u64]>) -> IndexedDagDenseMask {
        let Some(start) = words.iter().position(|word| *word != 0) else {
            return IndexedDagDenseMask::Empty;
        };
        let end = words
            .iter()
            .rposition(|word| *word != 0)
            .expect("nonzero start implies nonzero end")
            + 1;
        IndexedDagDenseMask::Dense { words, start, end }
    }

    fn single_transition(
        &self,
        dwa_state: u32,
        parser_state: u32,
        tsid: u32,
    ) -> Option<(u32, &'a IndexedDagDenseMask)> {
        let row = self
            .constraint
            .indexed_dag_dense_transitions
            .get(dwa_state as usize)?;
        let transition = self
            .constraint
            .indexed_parser_dwa_transition(row, parser_state)?;
        Some((
            transition.target,
            transition.masks.get(tsid),
        ))
    }

    fn intersect_single_with_dense_mask(
        dense: &Arc<[u64]>,
        mask: &IndexedDagDenseMask,
    ) -> Option<Arc<[u64]>> {
        match mask {
            IndexedDagDenseMask::Full => Some(Arc::clone(dense)),
            IndexedDagDenseMask::Empty => None,
            IndexedDagDenseMask::Dense {
                words: mask,
                start,
                end,
            } => {
                let end = (*end).min(dense.len()).min(mask.len());
                if *start >= end {
                    return None;
                }
                if *start != 0 || end != dense.len() {
                    let mut out = vec![0u64; end];
                    let mut last_nonzero = 0usize;
                    for index in *start..end {
                        let word = dense[index] & mask[index];
                        out[index] = word;
                        if word != 0 {
                            last_nonzero = index + 1;
                        }
                    }
                    if last_nonzero == 0 {
                        return None;
                    }
                    out.truncate(last_nonzero);
                    return Some(out.into());
                }
                let mut any = false;
                let mut out: Option<Vec<u64>> = None;
                for index in 0..dense.len() {
                    let word = dense[index] & mask[index];
                    any |= word != 0;
                    if let Some(out) = out.as_mut() {
                        out.push(word);
                    } else if word != dense[index] {
                        let mut changed = Vec::with_capacity(dense.len());
                        changed.extend_from_slice(&dense[..index]);
                        changed.push(word);
                        out = Some(changed);
                    }
                }
                if !any {
                    None
                } else if let Some(out) = out {
                    Some(out.into())
                } else {
                    Some(Arc::clone(dense))
                }
            }
        }
    }

    fn merge_single_result(
        current: &mut Option<Arc<[u64]>>,
        incoming: Option<Arc<[u64]>>,
    ) {
        let Some(incoming) = incoming else {
            return;
        };
        let Some(existing) = current.as_mut() else {
            *current = Some(incoming);
            return;
        };
        if Arc::ptr_eq(existing, &incoming) {
            return;
        }
        if existing.len() == incoming.len() {
            let existing = Arc::make_mut(existing);
            for (word, incoming_word) in existing.iter_mut().zip(incoming.iter()) {
                *word |= *incoming_word;
            }
            return;
        }
        let len = existing.len().max(incoming.len());
        let mut combined = vec![0u64; len];
        for (index, word) in existing.iter().enumerate() {
            combined[index] |= *word;
        }
        for (index, word) in incoming.iter().enumerate() {
            combined[index] |= *word;
        }
        *existing = combined.into();
    }

    fn merge_single_intersection(
        current: &mut Option<Arc<[u64]>>,
        incoming: Arc<[u64]>,
        mask: &IndexedDagDenseMask,
    ) {
        match mask {
            IndexedDagDenseMask::Full => {
                Self::merge_single_result(current, Some(incoming));
            }
            IndexedDagDenseMask::Empty => {}
            IndexedDagDenseMask::Dense {
                words: mask,
                start,
                end,
            } => {
                let Some(existing) = current.as_mut() else {
                    let transition_mask = IndexedDagDenseMask::Dense {
                        words: Arc::clone(mask),
                        start: *start,
                        end: *end,
                    };
                    *current =
                        Self::intersect_single_with_dense_mask(&incoming, &transition_mask);
                    return;
                };
                if Arc::ptr_eq(existing, &incoming) {
                    return;
                }
                let end = (*end).min(incoming.len()).min(mask.len());
                if *start >= end {
                    return;
                }
                if existing.len() >= end {
                    let existing = Arc::make_mut(existing);
                    for index in *start..end {
                        existing[index] |= incoming[index] & mask[index];
                    }
                    return;
                }
                let len = existing.len().max(end);
                let mut combined = vec![0u64; len];
                for (index, word) in existing.iter().enumerate() {
                    combined[index] = *word;
                }
                for index in *start..end {
                    combined[index] |= incoming[index] & mask[index];
                }
                *existing = combined.into();
            }
        }
    }

    /// Return the parser/GSS transfer mask for one internal tokenizer state,
    /// independently of the current seed accumulator.
    ///
    /// The denotation evaluated below uses only union and intersection with
    /// parser-DWA weights. Therefore `E(q, G, a) = a âˆ© E(q, G, U)` for every
    /// seed accumulator `a` and seed universe `U`. Caching `E(q, G, U)` by
    /// `(q, G, tsid)` keeps the result valid when delayed lexer exclusions
    /// produce a different `a` at the next model token.
    fn final_for_single_transfer(
        &self,
        dwa_state: u32,
        tsid: u32,
    ) -> Option<Arc<[u64]>> {
        match self
            .constraint
            .indexed_dag_dense_finals
            .get(dwa_state as usize)?
            .get(tsid)
        {
            IndexedDagDenseMask::Full => Some(Arc::clone(&self.constraint.seed_universe_dense)),
            IndexedDagDenseMask::Dense { words, .. } => Some(Arc::clone(words)),
            IndexedDagDenseMask::Empty => None,
        }
    }

    fn eval_lower_single_transfer(
        &mut self,
        dwa_state: u32,
        node: u32,
        tsid: u32,
    ) -> Option<Arc<[u64]>> {
        self.lower_calls += 1;
        let source_slot = self.source_slots[node as usize];
        if source_slot == u32::MAX {
            debug_assert!(false, "single lower evaluation reached upper node");
            return None;
        }
        let cached = self.runtime.single_sources[source_slot as usize]
            .lower
            .iter()
            .find(|entry| entry.dwa_state == dwa_state && entry.tsid == tsid)
            .map(|entry| entry.result.clone());
        if let Some(cached) = cached {
            self.lower_hits += 1;
            return cached;
        }
        let (empty, child_count) = match &self.dag.nodes[node as usize] {
            IndexedLeveledGssNode::LowerGeneral {
                empty, children, ..
            } => (*empty, children.len()),
            IndexedLeveledGssNode::LowerSegment { .. } => {
                let out = self.eval_segment_single_transfer(dwa_state, node, 0, tsid);
                self.runtime.single_sources[source_slot as usize]
                    .lower
                    .push(SingleLowerMemoEntry {
                        dwa_state,
                        tsid,
                        result: out.clone(),
                    });
                return out;
            }
            _ => unreachable!(),
        };
        let mut out = (empty || child_count != 0)
            .then(|| self.final_for_single_transfer(dwa_state, tsid))
            .flatten();
        for child_index in 0..child_count {
            let (parser_state, child) = match &self.dag.nodes[node as usize] {
                IndexedLeveledGssNode::LowerGeneral { children, .. } => {
                    children[child_index]
                }
                _ => unreachable!(),
            };
            let Some((target, transition_mask)) =
                self.single_transition(dwa_state, parser_state, tsid)
            else {
                continue;
            };
            if matches!(transition_mask, IndexedDagDenseMask::Empty) {
                continue;
            }
            if let Some(child_result) =
                self.eval_lower_single_transfer(target, child, tsid)
            {
                Self::merge_single_intersection(&mut out, child_result, transition_mask);
            }
        }
        self.runtime.single_sources[source_slot as usize]
            .lower
            .push(SingleLowerMemoEntry {
                dwa_state,
                tsid,
                result: out.clone(),
            });
        out
    }

    fn eval_segment_single_transfer(
        &mut self,
        dwa_state: u32,
        node: u32,
        offset: usize,
        tsid: u32,
    ) -> Option<Arc<[u64]>> {
        self.segment_calls += 1;
        let source_slot = self.source_slots[node as usize];
        if source_slot == u32::MAX {
            return None;
        }
        let cached = self.runtime.single_sources[source_slot as usize]
            .segments
            .iter()
            .find(|entry| {
                entry.offset == offset && entry.dwa_state == dwa_state && entry.tsid == tsid
            })
            .map(|entry| entry.result.clone());
        if let Some(cached) = cached {
            self.segment_hits += 1;
            return cached;
        }
        let (parser_state, next, has_more) = match &self.dag.nodes[node as usize] {
            IndexedLeveledGssNode::LowerSegment {
                values,
                next,
                ..
            } => {
                let Some(&parser_state) =
                    values.get(values.len().saturating_sub(1 + offset))
                else {
                    return None;
                };
                (parser_state, *next, offset + 1 < values.len())
            }
            _ => unreachable!(),
        };
        let mut out = self.final_for_single_transfer(dwa_state, tsid);
        if let Some((target, transition_mask)) =
            self.single_transition(dwa_state, parser_state, tsid)
            && !matches!(transition_mask, IndexedDagDenseMask::Empty)
        {
            let child_result = if has_more {
                self.eval_segment_single_transfer(
                    target,
                    node,
                    offset + 1,
                    tsid,
                )
            } else {
                self.eval_lower_single_transfer(target, next, tsid)
            };
            if let Some(child_result) = child_result {
                Self::merge_single_intersection(&mut out, child_result, transition_mask);
            }
        }
        self.runtime.single_sources[source_slot as usize]
            .segments
            .push(SingleSegmentMemoEntry {
                offset,
                dwa_state,
                tsid,
                result: out.clone(),
            });
        out
    }

    fn eval_lower(
        &mut self,
        dwa_state: u32,
        node: u32,
        accumulator_id: u32,
    ) -> Option<DenseMaskAcc> {
        if let Some((tsid, dense)) = self.runtime.accumulators[accumulator_id as usize]
            .0
            .first()
            .filter(|_| self.runtime.accumulators[accumulator_id as usize].0.len() == 1)
            .map(|(tsid, dense)| (*tsid, Arc::clone(dense)))
        {
            let transfer = self.eval_lower_single_transfer(dwa_state, node, tsid)?;
            let transfer = Self::single_dense_transition_mask(transfer);
            return Self::intersect_single_with_dense_mask(&dense, &transfer)
            .and_then(|result| DenseMaskAcc::from_dense_arc(tsid, result));
        }
        self.lower_calls += 1;
        let source_ptr = match &self.dag.nodes[node as usize] {
            IndexedLeveledGssNode::LowerGeneral { source, .. }
            | IndexedLeveledGssNode::LowerSegment { source, .. } => source.ptr_key(),
            IndexedLeveledGssNode::UpperBranch { .. }
            | IndexedLeveledGssNode::Interface { .. } => {
                debug_assert!(false, "lower evaluation reached upper node");
                return None;
            }
        };
        let key = (dwa_state, source_ptr, accumulator_id);
        if let Some(cached) = self.runtime.lower_memo.get(&key) {
            self.lower_hits += 1;
            return cached.clone();
        }
        let indexed = self.dag.nodes[node as usize].clone();
        let source = match &indexed {
            IndexedLeveledGssNode::LowerGeneral { source, .. }
            | IndexedLeveledGssNode::LowerSegment { source, .. } => source.clone(),
            IndexedLeveledGssNode::UpperBranch { .. }
            | IndexedLeveledGssNode::Interface { .. } => unreachable!(),
        };
        let out = match indexed {
            IndexedLeveledGssNode::LowerGeneral {
                empty, children, ..
            } => {
                let has_paths = empty || !children.is_empty();
                let mut out = has_paths
                    .then(|| self.final_for_accumulator(dwa_state, accumulator_id))
                    .flatten();
                for (parser_state, child) in children {
                    let Some((target, weight)) = self.transition(dwa_state, parser_state) else {
                        continue;
                    };
                    let child_result = self.eval_lower(target, child, accumulator_id);
                    let child_result = self.intersect_result(child_result, &weight);
                    Self::merge_result(&mut out, child_result);
                }
                out
            }
            IndexedLeveledGssNode::LowerSegment { .. } => {
                self.eval_segment(dwa_state, node, 0, accumulator_id)
            }
            IndexedLeveledGssNode::UpperBranch { .. }
            | IndexedLeveledGssNode::Interface { .. } => unreachable!(),
        };
        self.profile_memo_result(&out);
        self.runtime
            .lower_sources
            .entry(source_ptr)
            .or_insert(source);
        self.runtime.lower_memo.insert(key, out.clone());
        out
    }

    fn eval_segment(
        &mut self,
        dwa_state: u32,
        node: u32,
        offset: usize,
        accumulator_id: u32,
    ) -> Option<DenseMaskAcc> {
        self.segment_calls += 1;
        let source_ptr = match &self.dag.nodes[node as usize] {
            IndexedLeveledGssNode::LowerSegment { source, .. } => source.ptr_key(),
            _ => {
                debug_assert!(false, "segment evaluation reached non-segment node");
                return None;
            }
        };
        let key = (source_ptr, offset, dwa_state, accumulator_id);
        if let Some(cached) = self.runtime.segment_memo.get(&key) {
            self.segment_hits += 1;
            return cached.clone();
        }
        let IndexedLeveledGssNode::LowerSegment {
            source,
            values,
            next,
        } = self.dag.nodes[node as usize].clone()
        else {
            unreachable!("segment source changed during evaluation");
        };
        let mut out = self.final_for_accumulator(dwa_state, accumulator_id);
        if let Some(&parser_state) = values.get(values.len().saturating_sub(1 + offset))
            && let Some((target, weight)) = self.transition(dwa_state, parser_state)
        {
            let child_result = if offset + 1 < values.len() {
                self.eval_segment(target, node, offset + 1, accumulator_id)
            } else {
                self.eval_lower(target, next, accumulator_id)
            };
            let child_result = self.intersect_result(child_result, &weight);
            Self::merge_result(&mut out, child_result);
        }
        self.profile_memo_result(&out);
        self.runtime
            .lower_sources
            .entry(source_ptr)
            .or_insert(source);
        self.runtime.segment_memo.insert(key, out.clone());
        out
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
                .token_set_for_tsid_ref(*tsid)
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
    weight: RuntimeWeightRef<'_>,
    precomputed: &DenseTokenMaskCache,
    transition_gss_cache: &mut FxHashMap<DenseGssTransitionKey, DenseMaskGSS>,
    transition_intersection_cache: &mut DenseTokenSetIntersectionSmallCache,
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
    let cache_key = None;
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
        let intersected = allowed.intersect_with_runtime_weight_small_cached(
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
    constraint: &Constraint,
    queue: &mut MaskQueue,
    dwa_state: u32,
    parser_state: u32,
    popped: &DenseMaskGSS,
    precomputed: &DenseTokenMaskCache,
    transition_gss_cache: &mut FxHashMap<DenseGssTransitionKey, DenseMaskGSS>,
    transition_intersection_cache: &mut DenseTokenSetIntersectionSmallCache,
    profile: &mut Option<MaskInnerProfileStats>,
) {
    let lookup_start = if profile.is_some() {
        Some(Instant::now())
    } else {
        None
    };
    let Some((target, weight)) = constraint
        .runtime_parser_dwa_transition(dwa_state, parser_state)
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
        target,
        weight,
        precomputed,
        transition_gss_cache,
        transition_intersection_cache,
        profile,
    );
}

impl<'a> ConstraintState<'a> {
    fn or_segmented_component_mask(&self, output: &mut [u32], component_mask: &[u32]) {
        // Retained components can carry private linker sentinel IDs (and other
        // component-local specials) that are deliberately absent from the
        // finished outer constraint.  Their masks are expressed in original
        // token-ID space, so never OR those private IDs straight through into
        // the caller-visible mask.  Intersect with the outer constraint's
        // actual token universe while copying set bits.
        for (word_index, (&source_word, target_word)) in component_mask
            .iter()
            .zip(output.iter_mut())
            .enumerate()
        {
            let mut remaining = source_word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros();
                let token_id = word_index as u32 * 32 + bit;
                if self.knows_token_id(token_id) {
                    *target_word |= 1u32 << bit;
                }
                remaining &= remaining - 1;
            }
        }
    }

    /// Project one visible outer tokenizer state into every raw tokenizer
    /// state of `component` that it represents.
    ///
    /// Before runtime-product compression the composed lexer is a literal
    /// disjoint union, so this is just `global = component_offset + local`
    /// (with union state zero acting as the shared reset dispatcher). If the
    /// composed constraint later installs a deterministic runtime product, a
    /// visible product state denotes a subset of those old union states; lift
    /// that retained source relation back through the same offsets rather than
    /// pretending the product state itself is in component coordinates.
    fn segmented_local_tokenizer_states(
        &self,
        component: &crate::runtime::SegmentedParserComponent,
        global_state: u32,
    ) -> SmallVec<[u32; 8]> {
        let mut locals = SmallVec::<[u32; 8]>::new();
        let component_reset = component.constraint.runtime_commit_initial_state();
        let component_offset = component.tokenizer_state_offset;
        let component_states = component.constraint.tokenizer.num_states();

        let mut add_source_union_state = |source_state: u32| {
            // Source-union state zero is the shared reset dispatcher rather
            // than one component's physical state. In a component projection
            // it denotes that component's authoritative commit reset.
            if source_state == 0 {
                locals.push(component_reset);
                return;
            }
            let Some(local) = source_state.checked_sub(component_offset) else {
                return;
            };
            if local < component_states {
                locals.push(local);
            }
        };

        if global_state == self.constraint.runtime_commit_initial_state() {
            locals.push(component_reset);
        } else if let Some(source_offset) = self.constraint.runtime_source_state_offset() {
            if global_state >= source_offset {
                add_source_union_state(global_state - source_offset);
            } else if let Some(source_states) =
                self.constraint.runtime_product_source_states(global_state)
            {
                for &source_state in source_states {
                    add_source_union_state(source_state);
                }
            }
        } else {
            add_source_union_state(global_state);
        }

        locals.sort_unstable();
        locals.dedup();
        locals
    }

    fn segmented_local_disallowed(
        &self,
        component: &crate::runtime::SegmentedParserComponent,
        source: &TerminalsDisallowed,
    ) -> TerminalsDisallowed {
        let terminal_start = component.terminal_offset;
        let terminal_end = terminal_start.saturating_add(component.constraint.table.num_terminals);
        let mut result = TerminalsDisallowed::new();
        for (global_tokenizer_state, terminals) in source.iter() {
            let local_tokenizer_states =
                self.segmented_local_tokenizer_states(component, *global_tokenizer_state);
            for local_tokenizer_state in local_tokenizer_states {
                for &terminal in terminals.iter() {
                    if terminal_start <= terminal && terminal < terminal_end {
                        result = result.with_insert(
                            local_tokenizer_state,
                            terminal - terminal_start,
                        );
                    }
                }
            }
        }
        result
    }

    fn try_fill_mask_segmented_deterministic_union_direct(&self, buf: &mut [u32]) -> bool {
        fn accepted_for_stack(
            component: &crate::runtime::SegmentedParserComponent,
            top_first: &[u32],
        ) -> Weight {
            let constraint = component.constraint.as_ref();
            let dwa = &constraint.parser_dwa;
            let mut ops = crate::ds::weight::ScopedWeightOpCache::default();
            let mut state_id = dwa.start_state();
            let mut path_weight = Weight::all();
            let mut accepted = Weight::empty();
            let accumulate = |state_id: u32,
                              path_weight: &Weight,
                              accepted: &mut Weight,
                              ops: &mut crate::ds::weight::ScopedWeightOpCache| {
                if let Some(final_weight) = dwa
                    .states()
                    .get(state_id as usize)
                    .and_then(|state| state.final_weight.as_ref())
                {
                    let contribution = ops.intersection(path_weight, final_weight);
                    if !contribution.is_empty() {
                        *accepted = ops.union(accepted, &contribution);
                    }
                }
            };
            accumulate(state_id, &path_weight, &mut accepted, &mut ops);
            for &parser_state in top_first {
                let Some(state) = dwa.states().get(state_id as usize) else {
                    break;
                };
                let positive = encode_positive_label(parser_state);
                let transition = state
                    .transitions
                    .get(&positive)
                    .or_else(|| {
                        constraint
                            .parser_state_domain_label(parser_state)
                            .and_then(|label| state.transitions.get(&label))
                    })
                    .or_else(|| state.transitions.get(&DEFAULT_LABEL));
                let Some((target, edge_weight)) = transition else {
                    break;
                };
                path_weight = ops.intersection(&path_weight, edge_weight);
                if path_weight.is_empty() {
                    break;
                }
                state_id = *target;
                accumulate(state_id, &path_weight, &mut accepted, &mut ops);
            }
            accepted
        }

        fn dense_contains(acc: &DenseMaskAcc, token: u32) -> bool {
            let word = token as usize / 64;
            let bit = token % 64;
            acc.0.iter().any(|(_, dense)| {
                dense.get(word)
                    .is_some_and(|word_value| (*word_value & (1u64 << bit)) != 0)
            })
        }

        fn or_component_weight(
            outer: &ConstraintState<'_>,
            component: &crate::runtime::SegmentedParserComponent,
            global_tokenizer_state: u32,
            accepted: &Weight,
            allowed: &DenseMaskAcc,
            root_disallow: bool,
            buf: &mut [u32],
        ) -> bool {
            let local_tokenizer_states =
                outer.segmented_local_tokenizer_states(component, global_tokenizer_state);
            if local_tokenizer_states.is_empty() {
                return true;
            }
            let source = component.constraint.as_ref();
            let blocked = root_disallow
                .then(|| component.root_disallowed_terminal)
                .flatten()
                .and_then(|terminal| source.possible_matches.get(&terminal));
            for local_tokenizer_state in local_tokenizer_states {
                for &tsid in source.internal_tsids_for_state(local_tokenizer_state) {
                    let accepted_tokens = if accepted.is_full() {
                        source.internal_token_universe()
                    } else {
                        accepted.tokens_for_tsid(tsid)
                    };
                    if accepted_tokens.is_empty() {
                        continue;
                    }
                    let blocked_tokens = blocked.map(|weight| {
                        if weight.is_full() {
                            source.internal_token_universe()
                        } else {
                            weight.tokens_for_tsid(tsid)
                        }
                    });
                    for internal_token in accepted_tokens.iter() {
                        if blocked_tokens
                            .as_ref()
                            .is_some_and(|tokens| tokens.contains(internal_token))
                        {
                            continue;
                        }
                        if source.internal_token_to_tokens.is_empty() {
                            let original = internal_token;
                            let outer_internal = outer
                                .constraint
                                .original_token_internal_at(original)
                                .unwrap_or(u32::MAX);
                            if outer_internal != u32::MAX && dense_contains(allowed, outer_internal) {
                                set_original_mask_bit(buf, original);
                            }
                        } else if let Some(originals) =
                            source.internal_token_to_tokens.get(internal_token as usize)
                        {
                            for &original in originals {
                                let outer_internal = outer
                                    .constraint
                                    .original_token_internal_at(original)
                                    .unwrap_or(u32::MAX);
                                if outer_internal != u32::MAX
                                    && dense_contains(allowed, outer_internal)
                                {
                                    set_original_mask_bit(buf, original);
                                }
                            }
                        }
                    }
                }
            }
            true
        }

        let Some(overlay) = self.constraint.static_dynamic_overlay.as_ref() else {
            return false;
        };
        let dispatch = &overlay.segmented_component_union_root_dispatch;
        if dispatch.is_empty() || overlay.segmented_parser_components.is_empty() {
            return false;
        }
        if overlay.segmented_static_baseline {
            self.fill_mask_uncached(buf);
        } else {
            buf.fill(0);
        }
        for (&global_tokenizer_state, gss) in self.state.iter() {
            let mut complete = true;
            let traversal_complete = gss.for_each_stack_top_first_bounded(128, |top_first, acc| {
                let Some(allowed) =
                    self.terminals_disallowed_to_dense_acc(acc, global_tokenizer_state)
                else {
                    complete = false;
                    return;
                };
                // Synthetic root final: every component start final contributes.
                for component in &overlay.segmented_parser_components {
                    let accepted = accepted_for_stack(component, &[]);
                    if !or_component_weight(
                        self,
                        component,
                        global_tokenizer_state,
                        &accepted,
                        &allowed,
                        true,
                        buf,
                    ) {
                        complete = false;
                        return;
                    }
                }
                let Some(&global_top) = top_first.first() else {
                    return;
                };
                let component_index = dispatch
                    .get(global_top as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                if component_index == u32::MAX {
                    return;
                }
                let Some(component) = overlay
                    .segmented_parser_components
                    .get(component_index as usize)
                else {
                    complete = false;
                    return;
                };
                let mut local_top_first = SmallVec::<[u32; 64]>::new();
                for &global_parser_state in top_first {
                    let local = component
                        .global_to_local_parser_state
                        .get(global_parser_state as usize)
                        .copied()
                        .unwrap_or(u32::MAX);
                    if local == u32::MAX {
                        break;
                    }
                    local_top_first.push(local);
                }
                if local_top_first.is_empty() {
                    return;
                }
                let accepted = accepted_for_stack(component, &local_top_first);
                if !or_component_weight(
                    self,
                    component,
                    global_tokenizer_state,
                    &accepted,
                    &allowed,
                    false,
                    buf,
                ) {
                    complete = false;
                }
            });
            if !traversal_complete || !complete {
                return false;
            }
        }
        if !self.or_segmented_boundary_shards_mask(overlay, buf) {
            return false;
        }
        true
    }

    /// Evaluate the compressed deterministic union A.  Its synthetic root is
    /// represented by `segmented_component_union_root_dispatch`; after the
    /// first parser-state read each concrete stack path is in exactly one
    /// cached component DWA body.  Root final weights are the union of every
    /// component start final, so we also project an empty stack into each
    /// component coordinate for the same branch accumulator.
    fn try_fill_mask_segmented_deterministic_union(&self, buf: &mut [u32]) -> bool {
        if std::env::var_os("GLRMASK_EXPERIMENT_SEGMENTED_DIRECT_COMPONENT_DWA_MASK").is_some() {
            return self.try_fill_mask_segmented_deterministic_union_direct(buf);
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_SEGMENTED_MASK").is_some();
        let total_started_at = profile.then(Instant::now);
        let Some(overlay) = self.constraint.static_dynamic_overlay.as_ref() else {
            return false;
        };
        let dispatch = &overlay.segmented_component_union_root_dispatch;
        if dispatch.is_empty() || overlay.segmented_parser_components.is_empty() {
            return false;
        }

        let mut projected_states = SmallVec::<[ParserStateMap; 4]>::new();
        projected_states.resize_with(
            overlay.segmented_parser_components.len(),
            ParserStateMap::default,
        );

        for (&global_tokenizer_state, gss) in self.state.iter() {
            let complete = gss.for_each_stack_top_first_bounded(128, |top_first, acc| {
                // Deterministic union-root final = union of component start
                // finals. Preserve the branch-local disallowed accumulator in
                // each component coordinate.
                for (component_index, component) in
                    overlay.segmented_parser_components.iter().enumerate()
                {
                    let local_tokenizer_states =
                        self.segmented_local_tokenizer_states(component, global_tokenizer_state);
                    if local_tokenizer_states.is_empty() {
                        continue;
                    }
                    let local_disallowed = self.segmented_local_disallowed(component, acc);
                    // An empty projected parser stack is the component-root
                    // coordinate in the segmented union.  In particular, do
                    // not seed a retained dynamic child with its standalone
                    // initial language here: reaching the child is a boundary
                    // event owned by B, not part of the child's local A mask.
                    for local_tokenizer_state in local_tokenizer_states {
                        let mut branch_disallowed = local_disallowed.clone();
                        if let Some(terminal) = component.root_disallowed_terminal {
                            branch_disallowed = branch_disallowed.with_insert(
                                local_tokenizer_state,
                                terminal,
                            );
                        }
                        let root_gss =
                            ParserGSS::from_single_stack(Vec::new(), branch_disallowed);
                        projected_states[component_index].merge_insert(
                            local_tokenizer_state,
                            root_gss,
                        );
                    }
                }

                let Some(&global_top) = top_first.first() else {
                    return;
                };
                let component_index = dispatch
                    .get(global_top as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                if component_index == u32::MAX {
                    return;
                }
                let Some(component) = overlay
                    .segmented_parser_components
                    .get(component_index as usize)
                else {
                    return;
                };
                let local_tokenizer_states =
                    self.segmented_local_tokenizer_states(component, global_tokenizer_state);
                if local_tokenizer_states.is_empty() {
                    return;
                }

                let mut local_top_first = SmallVec::<[u32; 64]>::new();
                for &global_parser_state in top_first {
                    let local = component
                        .global_to_local_parser_state
                        .get(global_parser_state as usize)
                        .copied()
                        .unwrap_or(u32::MAX);
                    if local == u32::MAX {
                        break;
                    }
                    local_top_first.push(local);
                }
                if local_top_first.is_empty() {
                    return;
                }
                local_top_first.reverse();
                let local_stack = local_top_first.into_vec();
                let local_disallowed = self.segmented_local_disallowed(component, acc);
                for local_tokenizer_state in local_tokenizer_states {
                    projected_states[component_index as usize].merge_insert(
                        local_tokenizer_state,
                        ParserGSS::from_single_stack(
                            local_stack.clone(),
                            local_disallowed.clone(),
                        ),
                    );
                }
            });
            if !complete {
                return false;
            }
        }

        if overlay.segmented_static_baseline {
            self.fill_mask_uncached(buf);
        } else {
            buf.fill(0);
        }
        let mut component_times = SmallVec::<[u64; 4]>::new();
        let required_component_mask_len = overlay
            .segmented_parser_components
            .iter()
            .zip(projected_states.iter())
            .filter_map(|(component, state)| {
                (!state.is_empty()).then(|| component.constraint.mask_len())
            })
            .max()
            .unwrap_or(0)
            .max(buf.len());
        let mut component_buf = if projected_states.iter().any(|state| !state.is_empty()) {
            let mut scratch = self.mask_scratch.lock().unwrap();
            let mut reusable = std::mem::take(&mut scratch.output_buf);
            reusable.resize(required_component_mask_len, 0);
            Some(reusable)
        } else {
            None
        };
        for (component_index, (component, state)) in overlay
            .segmented_parser_components
            .iter()
            .zip(projected_states)
            .enumerate()
        {
            if state.is_empty() {
                component_times.push(0);
                continue;
            }
            let component_buf = component_buf
                .as_mut()
                .expect("active segmented component requires a mask buffer");
            let component_started_at = profile.then(Instant::now);
            component_buf.fill(0);
            let shadow = ConstraintState {
                constraint: component.constraint.as_ref(),
                state,
                buffers: Default::default(),
                generation: self.generation,
                mask_cache: Mutex::new(None),
                mask_scratch: {
                    let scratch = self.mask_scratch.lock().unwrap();
                    scratch
                        .segmented_component_scratch
                        .get(component_index)
                        .cloned()
                        .unwrap_or_else(|| {
                            Arc::new(Mutex::new(MaskScratch::for_constraint(
                                component.constraint.as_ref(),
                            )))
                        })
                },
            };
            // Dispatch through the retained component itself. Static source
            // constraints therefore keep their parser DWA and precomputed mask
            // artifacts, while dynamic sources keep their parser/lexer walker.
            // A nested hybrid component recursively retains the same split.
            shadow.fill_mask(component_buf);
            self.or_segmented_component_mask(buf, &component_buf);
            component_times.push(component_started_at.map_or(0, elapsed_ns));
        }

        let boundary_started_at = profile.then(Instant::now);
        if !self.or_segmented_boundary_shards_mask(overlay, buf) {
            return false;
        }
        let boundary_ns = boundary_started_at.map_or(0, elapsed_ns);
        if let Some(started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][deterministic_two_dwa_mask] components={} component_ns={component_times:?} boundary_ns={} total_ns={}",
                overlay.segmented_parser_components.len(),
                boundary_ns,
                elapsed_ns(started_at),
            );
        }
        true
    }

    /// Exact common-case evaluator for a segmented component parser union.
    ///
    /// Each retained component keeps its original token/TSID coordinate and
    /// therefore its existing dense mask caches.  For a concrete parser stack,
    /// read the maximal top-first prefix whose composed LR states have a local
    /// preimage in that component.  The standalone component DWA has no
    /// transition for the first foreign state, so deeper values are semantically
    /// irrelevant; truncating there preserves every accepting prefix exactly.
    /// Ambiguous GSSes deliberately decline this fast path for now.
    fn try_fill_mask_segmented_single_paths(&self, buf: &mut [u32]) -> bool {
        let profile_all = std::env::var_os("GLRMASK_PROFILE_SEGMENTED_MASK").is_some();
        let profile_slow = std::env::var_os("GLRMASK_PROFILE_SEGMENTED_MASK_SLOW").is_some();
        let profile = profile_all || profile_slow;
        let total_started_at = profile.then(Instant::now);
        let Some(overlay) = self.constraint.static_dynamic_overlay.as_ref() else {
            return false;
        };
        if !overlay.segmented_component_union_root_dispatch.is_empty() {
            return self.try_fill_mask_segmented_deterministic_union(buf);
        }
        // v22 deliberately flattens all-static retained components into one
        // ordinary parser-DWA baseline at save time.  After load the live
        // component list can therefore be empty while B remains separately
        // serialized and authoritative.  In that shape A comes from
        // `fill_mask_uncached` below and this evaluator only needs to OR B.
        if overlay.segmented_parser_components.is_empty() && !overlay.segmented_static_baseline {
            return false;
        }

        let mut projected_states = SmallVec::<[ParserStateMap; 4]>::new();
        projected_states.resize_with(
            overlay.segmented_parser_components.len(),
            ParserStateMap::default,
        );

        for (&global_tokenizer_state, gss) in self.state.iter() {
            let complete = gss.for_each_stack_top_first_bounded(128, |top_first, acc| {
                for (component_index, component) in
                    overlay.segmented_parser_components.iter().enumerate()
                {
                    let local_tokenizer_states =
                        self.segmented_local_tokenizer_states(component, global_tokenizer_state);
                    if local_tokenizer_states.is_empty() {
                        continue;
                    }

                    let mut local_top_first = SmallVec::<[u32; 64]>::new();
                    for &global_parser_state in top_first {
                        let local = component
                            .global_to_local_parser_state
                            .get(global_parser_state as usize)
                            .copied()
                            .unwrap_or(u32::MAX);
                        if local == u32::MAX {
                            break;
                        }
                        local_top_first.push(local);
                    }
                    let local_disallowed = self.segmented_local_disallowed(component, acc);
                    let local_stack = if local_top_first.is_empty() {
                        // An empty *composed* parser stack belongs to the outer
                        // parent root.  An empty projection of a non-empty
                        // composed stack means instead that this component does
                        // not own the current stack top.  In particular, while
                        // a child is active the buried parent caller frame must
                        // not reactivate parent A (or its scoped ignore) merely
                        // because the child's top state has no parent preimage.
                        if !top_first.is_empty() || component.terminal_offset != 0 {
                            continue;
                        }
                        Vec::new()
                    } else {
                        local_top_first.reverse();
                        local_top_first.into_vec()
                    };
                    // Boundary reachability is accounted for separately by B.
                    // At a synthetic component root suppress standalone scoped
                    // ignore independently in every lifted lexer lane.
                    for local_tokenizer_state in local_tokenizer_states {
                        let mut branch_disallowed = local_disallowed.clone();
                        if local_stack.is_empty()
                            && let Some(terminal) = component.root_disallowed_terminal
                        {
                            branch_disallowed = branch_disallowed.with_insert(
                                local_tokenizer_state,
                                terminal,
                            );
                        }
                        projected_states[component_index].merge_insert(
                            local_tokenizer_state,
                            ParserGSS::from_single_stack(
                                local_stack.clone(),
                                branch_disallowed,
                            ),
                        );
                    }
                }
            });
            if !complete {
                return false;
            }
        }

        if overlay.segmented_static_baseline {
            self.fill_mask_uncached(buf);
        } else {
            buf.fill(0);
        }
        let mut component_buf = None::<Vec<u32>>;
        let mut component_times = SmallVec::<[u64; 4]>::new();
        for (component_index, (component, state)) in overlay
            .segmented_parser_components
            .iter()
            .zip(projected_states)
            .enumerate()
        {
            if state.is_empty() {
                component_times.push(0);
                continue;
            }
            let component_started_at = profile.then(Instant::now);
            let reused_initial = {
                let scratch = self.mask_scratch.lock().unwrap();
                match (
                    scratch.segmented_component_initial_states.get(component_index),
                    scratch.segmented_component_initial_masks.get(component_index),
                ) {
                    (Some(initial_state), Some(initial_mask)) if initial_state == &state => {
                        self.or_segmented_component_mask(buf, initial_mask);
                        true
                    }
                    _ => false,
                }
            };
            if reused_initial {
                component_times.push(component_started_at.map_or(0, elapsed_ns));
                continue;
            }
            if component_buf.is_none() {
                let mut scratch = self.mask_scratch.lock().unwrap();
                let mut reusable = std::mem::take(&mut scratch.output_buf);
                reusable.resize(
                    buf.len().max(component.constraint.mask_len()),
                    0,
                );
                component_buf = Some(reusable);
            } else if component_buf
                .as_ref()
                .is_some_and(|buffer| buffer.len() < component.constraint.mask_len())
            {
                component_buf
                    .as_mut()
                    .expect("segmented component mask buffer exists")
                    .resize(component.constraint.mask_len(), 0);
            }
            let component_buf = component_buf
                .as_mut()
                .expect("active segmented component requires a mask buffer");
            component_buf.fill(0);
            let shadow = ConstraintState {
                constraint: component.constraint.as_ref(),
                state,
                buffers: Default::default(),
                generation: self.generation,
                mask_cache: Mutex::new(None),
                mask_scratch: {
                    let scratch = self.mask_scratch.lock().unwrap();
                    scratch
                        .segmented_component_scratch
                        .get(component_index)
                        .cloned()
                        .unwrap_or_else(|| {
                            Arc::new(Mutex::new(MaskScratch::for_constraint(
                                component.constraint.as_ref(),
                            )))
                        })
                },
            };
            // Preserve the supplied component's runtime backend. A static
            // component reuses its compiled parser-DWA/mask machinery; a
            // dynamic component keeps using its exact lexer/parser walker.
            // Nested segmented components recurse through the same dispatch.
            shadow.fill_mask(component_buf);
            if std::env::var_os("GLRMASK_DEBUG_SEGMENTED_COMPONENT_MASK").is_some() {
                eprintln!(
                    "[glrmask/debug][segmented_component_mask] dynamic={} terminal_offset={} mask={:?} state={:?}",
                    component.constraint.uses_dynamic_runtime(),
                    component.terminal_offset,
                    component_buf,
                    shadow.state,
                );
            }
            self.or_segmented_component_mask(buf, component_buf.as_slice());
            component_times.push(component_started_at.map_or(0, elapsed_ns));
        }
        if let Some(mut reusable) = component_buf.take() {
            reusable.clear();
            self.mask_scratch.lock().unwrap().output_buf = reusable;
        }
        let boundary_started_at = profile.then(Instant::now);
        if !self.or_segmented_boundary_shards_mask(overlay, buf) {
            return false;
        }
        let boundary_ns = boundary_started_at.map_or(0, elapsed_ns);
        if let Some(started_at) = total_started_at {
            let total_ns = elapsed_ns(started_at);
            if profile_all || total_ns >= 100_000 {
                eprintln!(
                    "[glrmask/profile][segmented_parser_mask] components={} component_ns={component_times:?} boundary_ns={} total_ns={}",
                    overlay.segmented_parser_components.len(),
                    boundary_ns,
                    total_ns,
                );
            }
        }
        true
    }

    fn or_segmented_boundary_shards_mask(
        &self,
        overlay: &crate::runtime::StaticDynamicOverlayMetadata,
        buf: &mut [u32],
    ) -> bool {
        if !overlay.segmented_boundary_shards.is_empty() {
            let mut needs_direct_dynamic = false;
            let mut direct_candidates = vec![0u32; self.constraint.mask_len()];
            let mut direct_candidates_complete = true;
            for shard in &overlay.segmented_boundary_shards {
                let ok = match &shard.backend {
                    crate::runtime::SegmentedBoundaryShardBackend::StaticParser(boundary) => {
                        self.or_segmented_boundary_parser_mask(
                            boundary,
                            Some(&shard.start_parser_states),
                            shard.accepts_empty_stack,
                            buf,
                        )
                    }
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicTerminalTrie(boundary) => {
                        self.or_segmented_boundary_terminal_trie_mask(
                            boundary,
                            Some(&shard.start_parser_states),
                            shard.accepts_empty_stack,
                            buf,
                        )
                    }
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect => {
                        if !self.segmented_boundary_shard_may_be_active(shard) {
                            continue;
                        }
                        needs_direct_dynamic = true;
                        let trigger = overlay
                            .segmented_parser_components
                            .get(shard.start_component as usize)
                            .map(|component| &component.constraint.boundary_trigger);
                        match trigger {
                            Some(crate::runtime::BoundaryTrigger::Tokens(tokens)) => {
                                for &token in tokens.iter() {
                                    set_original_mask_bit(&mut direct_candidates, token);
                                }
                            }
                            Some(crate::runtime::BoundaryTrigger::Exact(dwa)) => {
                                let Some(component) = overlay
                                    .segmented_parser_components
                                    .get(shard.start_component as usize)
                                else {
                                    direct_candidates_complete = false;
                                    continue;
                                };
                                if !self.or_exact_component_trigger_candidates(
                                    component,
                                    shard,
                                    dwa,
                                    &mut direct_candidates,
                                ) {
                                    direct_candidates_complete = false;
                                }
                            }
                            Some(crate::runtime::BoundaryTrigger::None) | None => {
                                // Required zero-build-cost conservative level.
                                direct_candidates_complete = false;
                            }
                        }
                        true
                    }
                };
                if !ok {
                    return false;
                }
            }
            if needs_direct_dynamic {
                if direct_candidates_complete {
                    // Token-summary trigger: exact dynamic recognition remains
                    // authoritative, but trie traversal is restricted to the
                    // conservative union of component-owned candidate tokens.
                    super::dynamic_mask::or_mask_dynamic_candidate_additions(
                        self,
                        buf,
                        &direct_candidates,
                    );
                } else {
                    // Conservative `TriggerDetail::None` semantics: run the
                    // exact dynamic recognizer once over the composed state.
                    // It adds only language not already covered by retained A.
                    super::dynamic_mask::or_mask_dynamic_additions(self, buf);
                }
            }
            return true;
        }

        // v22/legacy in-memory compatibility while boundary shards are being
        // versioned into the wire format.
        if let Some(boundary) = overlay.segmented_boundary_parser.as_deref()
            && !self.or_segmented_boundary_parser_mask(boundary, None, true, buf)
        {
            return false;
        }
        if let Some(boundary) = overlay.segmented_boundary_terminal_trie.as_deref()
            && !self.or_segmented_boundary_terminal_trie_mask(boundary, None, true, buf)
        {
            return false;
        }
        true
    }

    fn or_segmented_boundary_terminal_trie_mask(
        &self,
        boundary: &crate::runtime::SegmentedBoundaryTerminalTrie,
        start_parser_states: Option<&crate::ds::bitset::BitSet>,
        accepts_empty_stack: bool,
        buf: &mut [u32],
    ) -> bool {
        const MAX_NWA_PRODUCT_ENTRIES: usize = 4096;

        fn dense_contains(acc: &DenseMaskAcc, token: u32) -> bool {
            let word = token as usize / 64;
            let bit = token % 64;
            acc.0.iter().any(|(_, dense)| {
                dense.get(word)
                    .is_some_and(|word_value| (*word_value & (1u64 << bit)) != 0)
            })
        }

        #[derive(Clone)]
        enum BoundaryTokenDomain {
            Full,
            Set(Arc<RangeSetBlaze<u32>>),
        }

        fn same_domain(left: &BoundaryTokenDomain, right: &BoundaryTokenDomain) -> bool {
            match (left, right) {
                (BoundaryTokenDomain::Full, BoundaryTokenDomain::Full) => true,
                (BoundaryTokenDomain::Set(left), BoundaryTokenDomain::Set(right)) => {
                    Arc::ptr_eq(left, right) || left.as_ref() == right.as_ref()
                }
                _ => false,
            }
        }

        fn intersect_domain(
            current: &BoundaryTokenDomain,
            edge: &Weight,
            tsid: u32,
        ) -> Option<BoundaryTokenDomain> {
            if edge.is_empty() {
                return None;
            }
            if edge.is_full() {
                return Some(current.clone());
            }
            let edge_tokens = edge.token_set_for_tsid_ref(tsid)?;
            match current {
                BoundaryTokenDomain::Full => {
                    Some(BoundaryTokenDomain::Set(Arc::clone(edge_tokens)))
                }
                BoundaryTokenDomain::Set(current_tokens) => {
                    if Arc::ptr_eq(current_tokens, edge_tokens)
                        || current_tokens.as_ref() == edge_tokens.as_ref()
                    {
                        return Some(current.clone());
                    }
                    if current_tokens.as_ref().is_disjoint(edge_tokens.as_ref()) {
                        return None;
                    }
                    let overlap = current_tokens.as_ref() & edge_tokens.as_ref();
                    (!overlap.is_empty())
                        .then(|| BoundaryTokenDomain::Set(Arc::new(overlap)))
                }
            }
        }

        type ProductBucket = SmallVec<[(BoundaryTokenDomain, ParserGSS); 4]>;

        fn push_product(
            bucket: &mut ProductBucket,
            domain: BoundaryTokenDomain,
            parser: ParserGSS,
        ) {
            if let Some((_, existing_parser)) = bucket
                .iter_mut()
                .find(|(existing_domain, _)| same_domain(existing_domain, &domain))
            {
                *existing_parser = existing_parser.merge(&parser);
            } else {
                bucket.push((domain, parser));
            }
        }

        for (&global_tokenizer_state, gss) in self.state.iter() {
            let tsid = boundary
                .tokenizer_state_to_tsid
                .get(global_tokenizer_state as usize)
                .copied()
                .unwrap_or(u32::MAX);
            if tsid == u32::MAX {
                continue;
            }

            let mut complete = true;
            let single_path = gss.is_single_path();
            let traversal_complete = gss.for_each_stack_top_first_bounded(128, |top_first, acc| {
                if let Some(start_parser_states) = start_parser_states {
                    match top_first.first().copied() {
                        Some(top) if !start_parser_states.contains(top as usize) => return,
                        None if !accepts_empty_stack => return,
                        _ => {}
                    }
                }
                let Some(allowed) =
                    self.terminals_disallowed_to_dense_acc(acc, global_tokenizer_state)
                else {
                    complete = false;
                    return;
                };
                let parser = if single_path {
                    gss.clone()
                } else {
                    let mut stack = SmallVec::<[u32; 64]>::from_slice(top_first);
                    stack.reverse();
                    ParserGSS::from_single_stack(stack.into_vec(), acc.clone())
                };

                let mut admit_internal_token = |internal_token: u32| -> bool {
                    let Some(originals) = boundary
                        .internal_token_to_originals
                        .get(internal_token as usize)
                    else {
                        return false;
                    };
                    for &original in originals {
                        let outer_internal = self
                            .constraint
                            .original_token_internal_at(original)
                            .unwrap_or(u32::MAX);
                        let outer_allowed = outer_internal != u32::MAX
                            && dense_contains(&allowed, outer_internal);
                        if outer_allowed {
                            set_original_mask_bit(buf, original);
                        }
                    }
                    true
                };

                if let Some(nwa) = boundary.symbolic_nwa.as_ref() {
                    let mut buckets = SmallVec::<[ProductBucket; 16]>::new();
                    buckets.resize_with(nwa.nodes.len(), ProductBucket::new);
                    for &start in &nwa.start_states {
                        let Some(bucket) = buckets.get_mut(start as usize) else {
                            complete = false;
                            return;
                        };
                        push_product(bucket, BoundaryTokenDomain::Full, parser.clone());
                    }

                    let mut product_entries = 0usize;
                    for &state_id in &nwa.topological_order {
                        let Some(bucket) = buckets.get_mut(state_id as usize) else {
                            complete = false;
                            return;
                        };
                        let entries = std::mem::take(bucket);
                        if entries.is_empty() {
                            continue;
                        }
                        product_entries = product_entries.saturating_add(entries.len());
                        if product_entries > MAX_NWA_PRODUCT_ENTRIES {
                            complete = false;
                            return;
                        }
                        let Some(node) = nwa.nodes.get(state_id as usize) else {
                            complete = false;
                            return;
                        };
                        for (domain, parser) in entries {
                            if let Some(final_weight) = node.final_weight.as_ref()
                                && let Some(output_domain) =
                                    intersect_domain(&domain, final_weight, tsid)
                            {
                                match output_domain {
                                    BoundaryTokenDomain::Full => {
                                        for internal_token in
                                            0..boundary.internal_token_to_originals.len() as u32
                                        {
                                            if !admit_internal_token(internal_token) {
                                                complete = false;
                                                return;
                                            }
                                        }
                                    }
                                    BoundaryTokenDomain::Set(tokens) => {
                                        for range in tokens.ranges() {
                                            for internal_token in range {
                                                if !admit_internal_token(internal_token) {
                                                    complete = false;
                                                    return;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            for (target, edge_weight) in &node.epsilons {
                                let Some(next_domain) =
                                    intersect_domain(&domain, edge_weight, tsid)
                                else {
                                    continue;
                                };
                                let Some(target_bucket) = buckets.get_mut(*target as usize) else {
                                    complete = false;
                                    return;
                                };
                                push_product(target_bucket, next_domain, parser.clone());
                            }
                            for transition in &node.transitions {
                                let Some(next_domain) =
                                    intersect_domain(&domain, &transition.weight, tsid)
                                else {
                                    continue;
                                };
                                let advanced = super::commit::advance_parser_stacks_table_exact(
                                    self.constraint,
                                    &parser,
                                    transition.terminal,
                                );
                                let Some(advanced) = advanced else {
                                    continue;
                                };
                                if advanced.is_empty() {
                                    continue;
                                }
                                let Some(target_bucket) =
                                    buckets.get_mut(transition.target as usize)
                                else {
                                    complete = false;
                                    return;
                                };
                                push_product(target_bucket, next_domain, advanced);
                            }
                        }
                    }
                    return;
                }

                // Legacy v21 artifacts carry an explicitly expanded trie. Keep
                // that evaluator unchanged for backwards compatibility.
                let root = boundary
                    .root_by_tsid
                    .get(tsid as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                if root == u32::MAX {
                    return;
                }
                let mut pending = Vec::<(u32, ParserGSS)>::new();
                pending.push((root, parser));
                let mut visits = 0usize;
                while let Some((node_id, parser)) = pending.pop() {
                    visits += 1;
                    if visits > boundary.nodes.len().saturating_mul(2).max(32) {
                        complete = false;
                        return;
                    }
                    let Some(node) = boundary.nodes.get(node_id as usize) else {
                        complete = false;
                        return;
                    };
                    for &internal_token in &node.outputs {
                        if !admit_internal_token(internal_token) {
                            complete = false;
                            return;
                        }
                    }
                    for &(terminal, child) in &node.children {
                        if let Some(advanced) = super::commit::advance_parser_stacks_table_exact(
                            self.constraint,
                            &parser,
                            terminal,
                        ) && !advanced.is_empty()
                        {
                            pending.push((child, advanced));
                        }
                    }
                }
            });
            if !traversal_complete || !complete {
                return false;
            }
        }
        true
    }

    /// Conservatively test whether this shard can own at least one current
    /// composed parser-stack top. Incomplete bounded GSS inspection returns
    /// true so trigger acceleration can never create a false negative.
    fn segmented_boundary_shard_may_be_active(
        &self,
        shard: &crate::runtime::SegmentedBoundaryShard,
    ) -> bool {
        let mut active = false;
        for gss in self.state.values() {
            let complete = gss.for_each_stack_top_first_bounded(128, |top_first, _| {
                if active {
                    return;
                }
                match top_first.first().copied() {
                    Some(top) => {
                        active = shard.start_parser_states.contains(top as usize);
                    }
                    None => {
                        active = shard.accepts_empty_stack;
                    }
                }
            });
            if !complete {
                return true;
            }
            if active {
                return true;
            }
        }
        false
    }

    /// OR exact component-trigger candidates into `buf`. Trigger parser labels
    /// are component-local LR IDs, but its weight coordinate is deliberately
    /// raw/original: TSID == local tokenizer-state ID and token == original
    /// model-token ID. This avoids assuming that the ordinary whole-token
    /// TSID/token quotient also preserves proper-prefix boundary observations.
    /// This routine intentionally ignores terminal exclusions: doing so can
    /// only add false-positive candidates and therefore remains a sound gate
    /// for the exact dynamic crossing evaluator.
    fn or_exact_component_trigger_candidates(
        &self,
        component: &crate::runtime::SegmentedParserComponent,
        shard: &crate::runtime::SegmentedBoundaryShard,
        dwa: &crate::automata::weighted_u32::dwa::DWA,
        buf: &mut [u32],
    ) -> bool {
        for (&global_tokenizer_state, gss) in self.state.iter() {
            let local_tokenizer_states =
                self.segmented_local_tokenizer_states(component, global_tokenizer_state);
            if local_tokenizer_states.is_empty() {
                // An unmappable lexer state is harmless only when this
                // component cannot own any stack at that lexer key. If an
                // active shard is present, declining Exact is mandatory:
                // silently skipping the key could remove real trigger tokens
                // and create a false-negative dynamic-boundary gate.
                let mut active = false;
                let traversal_complete =
                    gss.for_each_stack_top_first_bounded(128, |top_first, _| {
                        if active {
                            return;
                        }
                        active = match top_first.first().copied() {
                            Some(top) => shard.start_parser_states.contains(top as usize),
                            None => shard.accepts_empty_stack,
                        };
                    });
                if !traversal_complete || active {
                    return false;
                }
                continue;
            }
            let mut complete = true;
            let traversal_complete = gss.for_each_stack_top_first_bounded(128, |top_first, _| {
                match top_first.first().copied() {
                    Some(top) if !shard.start_parser_states.contains(top as usize) => return,
                    None if !shard.accepts_empty_stack => return,
                    _ => {}
                }

                let mut local_top_first = SmallVec::<[u32; 64]>::new();
                let mut projection_truncated = false;
                for &global_parser_state in top_first {
                    let local = component
                        .global_to_local_parser_state
                        .get(global_parser_state as usize)
                        .copied()
                        .unwrap_or(u32::MAX);
                    if local == u32::MAX {
                        projection_truncated = true;
                        break;
                    }
                    local_top_first.push(local);
                }
                if projection_truncated || (!top_first.is_empty() && local_top_first.is_empty()) {
                    // Do not evaluate a trigger language on a silently
                    // truncated local stack.  A future scoped-GSS proof may
                    // certify the first foreign state as exactly the caller
                    // boundary; until then, fallback to the unified dynamic
                    // recognizer rather than risk losing a deeper-stack
                    // readiness condition.
                    complete = false;
                    return;
                }

                let accepted = exact_component_trigger_accepted_weight(
                    component.constraint.as_ref(),
                    dwa,
                    &local_top_first,
                );
                if accepted.is_empty() {
                    return;
                }
                // Exact trigger construction always leaves the lexical path
                // support attached, so a universal final result is malformed
                // for this artifact. Decline rather than interpreting it as a
                // huge token universe and risk masking a coordinate bug.
                if accepted.is_full() {
                    complete = false;
                    return;
                }
                for &local_tokenizer_state in &local_tokenizer_states {
                    let tokens = accepted.tokens_for_tsid(local_tokenizer_state);
                    for original_token in tokens.iter() {
                        set_original_mask_bit(buf, original_token);
                    }
                }
            });
            if !traversal_complete || !complete {
                return false;
            }
        }
        true
    }

    /// Evaluate the private-coordinate deterministic boundary parser DWA over
    /// the current composed parser GSS.  The boundary machine is fully
    /// determinized and negative-free before publication; only the top-level
    /// union with the component parser DWA remains segmented at runtime.
    fn or_segmented_boundary_parser_mask(
        &self,
        boundary: &crate::runtime::SegmentedBoundaryParser,
        start_parser_states: Option<&crate::ds::bitset::BitSet>,
        accepts_empty_stack: bool,
        buf: &mut [u32],
    ) -> bool {
        fn accepted_for_stack(
            dwa: &crate::automata::weighted_u32::dwa::DWA,
            top_first: &[u32],
        ) -> Weight {
            let mut ops = crate::ds::weight::ScopedWeightOpCache::default();
            let mut state_id = dwa.start_state();
            let mut path_weight = Weight::all();
            let mut accepted = Weight::empty();
            let accumulate_final = |state_id: u32,
                                    path_weight: &Weight,
                                    accepted: &mut Weight,
                                    ops: &mut crate::ds::weight::ScopedWeightOpCache| {
                if let Some(final_weight) = dwa
                    .states()
                    .get(state_id as usize)
                    .and_then(|state| state.final_weight.as_ref())
                {
                    let contribution = ops.intersection(path_weight, final_weight);
                    if !contribution.is_empty() {
                        *accepted = ops.union(accepted, &contribution);
                    }
                }
            };
            accumulate_final(state_id, &path_weight, &mut accepted, &mut ops);

            for &parser_state in top_first {
                let label = encode_positive_label(parser_state);
                let Some(state) = dwa.states().get(state_id as usize) else {
                    break;
                };
                let Some((target, edge_weight)) = state
                    .transitions
                    .get(&label)
                    .or_else(|| state.transitions.get(&DEFAULT_LABEL))
                else {
                    break;
                };
                path_weight = ops.intersection(&path_weight, edge_weight);
                if path_weight.is_empty() {
                    break;
                }
                state_id = *target;
                accumulate_final(state_id, &path_weight, &mut accepted, &mut ops);
            }
            accepted
        }

        fn accepted_mask_for_stack(
            dwa: &crate::compiler::stages::parser_dwa::SmallBoundaryDwa,
            tsid: u32,
            top_first: &[u32],
        ) -> u64 {
            if tsid >= dwa.tsid_count as u32 {
                return 0;
            }
            let mut state_id = dwa.start_state();
            let mut path_mask = dwa.all_token_mask();
            let mut accepted = 0u64;
            let accumulate_final = |state_id: u32, path_mask: u64, accepted: &mut u64| {
                let Some(state) = dwa.states.get(state_id as usize) else {
                    return;
                };
                if state.final_weight != 0 {
                    *accepted |= path_mask & dwa.weight_mask(state.final_weight, tsid);
                }
            };
            accumulate_final(state_id, path_mask, &mut accepted);
            for &parser_state in top_first {
                let label = encode_positive_label(parser_state);
                let Some(state) = dwa.states.get(state_id as usize) else {
                    break;
                };
                let edge = state
                    .transitions
                    .iter()
                    .find(|(edge_label, _, _)| *edge_label == label)
                    .or_else(|| {
                        state
                            .transitions
                            .iter()
                            .find(|(edge_label, _, _)| *edge_label == DEFAULT_LABEL)
                    });
                let Some(&(_, target, weight)) = edge else {
                    break;
                };
                path_mask &= dwa.weight_mask(weight, tsid);
                if path_mask == 0 {
                    break;
                }
                state_id = target;
                accumulate_final(state_id, path_mask, &mut accepted);
            }
            accepted
        }

        fn dense_contains(acc: &DenseMaskAcc, token: u32) -> bool {
            let word = token as usize / 64;
            let bit = token % 64;
            acc.0.iter().any(|(_, dense)| {
                dense.get(word)
                    .is_some_and(|word_value| (*word_value & (1u64 << bit)) != 0)
            })
        }

        let debug_boundary = std::env::var_os("GLRMASK_DEBUG_SEGMENTED_BOUNDARY_MASK").is_some();
        for (&global_tokenizer_state, gss) in self.state.iter() {
            let boundary_tsid = if boundary.uses_composed_tsid_coordinate {
                self.constraint
                    .state_to_internal_tsid
                    .get(global_tokenizer_state as usize)
                    .copied()
                    .unwrap_or(u32::MAX)
            } else {
                boundary
                    .tokenizer_state_to_tsid
                    .get(global_tokenizer_state as usize)
                    .copied()
                    .unwrap_or(u32::MAX)
            };
            if debug_boundary {
                eprintln!(
                    "[glrmask/debug][segmented_boundary_state] global_tokenizer_state={} boundary_tsid={} paths={}",
                    global_tokenizer_state,
                    boundary_tsid,
                    gss.path_count_at_most(129),
                );
            }
            if boundary_tsid == u32::MAX {
                continue;
            }
            let mut complete = true;
            let traversal_complete = gss.for_each_stack_top_first_bounded(128, |top_first, acc| {
                if let Some(start_parser_states) = start_parser_states {
                    match top_first.first().copied() {
                        Some(top) if !start_parser_states.contains(top as usize) => return,
                        None if !accepts_empty_stack => return,
                        _ => {}
                    }
                }
                let Some(allowed) =
                    self.terminals_disallowed_to_dense_acc(acc, global_tokenizer_state)
                else {
                    // This segmented evaluator is an optimization. If an
                    // exclusion accumulator cannot be represented exactly in
                    // its dense scratch form, decline the whole projection so
                    // the caller runs the unified exact fallback.
                    complete = false;
                    return;
                };
                if let Some(compact) = boundary.compact_parser_dwa.as_ref() {
                    let mut accepted = accepted_mask_for_stack(compact, boundary_tsid, top_first);
                    if debug_boundary {
                        let internal = (0..compact.token_count as u32)
                            .filter(|&token| accepted & (1u64 << token) != 0)
                            .collect::<Vec<_>>();
                        let originals = internal
                            .iter()
                            .flat_map(|&token| boundary.internal_token_to_originals.get(token as usize).into_iter().flatten().copied())
                            .collect::<Vec<_>>();
                        eprintln!(
                            "[glrmask/debug][segmented_boundary_stack] top_first={:?} tsid={} accepted_internal={:?} accepted_originals={:?} compact=true",
                            top_first,
                            boundary_tsid,
                            internal,
                            originals,
                        );
                    }
                    while accepted != 0 {
                        let internal_token = accepted.trailing_zeros();
                        accepted &= accepted - 1;
                        let Some(originals) = boundary
                            .internal_token_to_originals
                            .get(internal_token as usize)
                        else {
                            complete = false;
                            return;
                        };
                        for &original in originals {
                            let outer_internal = self
                                .constraint
                                .original_token_internal_at(original)
                                .unwrap_or(u32::MAX);
                            if outer_internal != u32::MAX && dense_contains(&allowed, outer_internal) {
                                set_original_mask_bit(buf, original);
                            }
                        }
                    }
                } else {
                    let accepted = accepted_for_stack(&boundary.parser_dwa, top_first);
                    let Some(tokens) = accepted.token_set_for_tsid_ref(boundary_tsid) else {
                        if debug_boundary {
                            eprintln!(
                                "[glrmask/debug][segmented_boundary_stack] top_first={:?} tsid={} accepted_internal=[]",
                                top_first,
                                boundary_tsid,
                            );
                        }
                        return;
                    };
                    if debug_boundary {
                        let internal = tokens.ranges().flat_map(|range| range).collect::<Vec<_>>();
                        let originals = internal
                            .iter()
                            .flat_map(|&token| boundary.internal_token_to_originals.get(token as usize).into_iter().flatten().copied())
                            .collect::<Vec<_>>();
                        eprintln!(
                            "[glrmask/debug][segmented_boundary_stack] top_first={:?} tsid={} accepted_internal={:?} accepted_originals={:?}",
                            top_first,
                            boundary_tsid,
                            internal,
                            originals,
                        );
                    }
                    for range in tokens.ranges() {
                        for internal_token in range {
                            let Some(originals) = boundary
                                .internal_token_to_originals
                                .get(internal_token as usize)
                            else {
                                complete = false;
                                return;
                            };
                            for &original in originals {
                                let outer_internal = self
                                    .constraint
                                    .original_token_internal_at(original)
                                    .unwrap_or(u32::MAX);
                                if outer_internal != u32::MAX && dense_contains(&allowed, outer_internal) {
                                    set_original_mask_bit(buf, original);
                                }
                            }
                        }
                    }
                }
            });
            if !traversal_complete || !complete {
                return false;
            }
        }
        true
    }

    /// Two-DWA runtime: the ordinary mask hot path evaluates deterministic A
    /// (`constraint.parser_dwa`) first, then ORs deterministic boundary DWA B.
    /// The legacy segmented-component experiment already evaluates B inside its
    /// own path, so only apply this overlay when there are no component
    /// segments.
    fn or_two_dwa_boundary_parser_mask(&self, buf: &mut [u32]) -> bool {
        let Some(overlay) = self.constraint.static_dynamic_overlay.as_ref() else {
            return true;
        };
        if !overlay.segmented_parser_components.is_empty() {
            return true;
        }
        if !self.or_segmented_boundary_shards_mask(overlay, buf) {
            return false;
        }
        true
    }

    /// Exact overlay for an out-of-vocabulary special token reached only after
    /// a linker control chain. Ordinary parser-DWA weights remain the fast path;
    /// constraints without explicit controls pay nothing here.
    fn update_control_special_token_mask(&self, buf: &mut [u32]) {
        if self.constraint.table.control_terminals.is_empty() {
            return;
        }
        let mut previous_token_id = None;
        for special in &self.constraint.special_token_terminals {
            if self
                .constraint
                .is_late_grammar_placeholder_terminal(special.terminal_id)
            {
                continue;
            }
            if previous_token_id == Some(special.token_id) {
                continue;
            }
            previous_token_id = Some(special.token_id);
            if super::commit::advance_special_token_paths(
                self.constraint,
                &self.state,
                special.token_id,
            )
            .is_some_and(|gss| !gss.is_empty())
            {
                set_original_mask_bit(buf, special.token_id);
            }
        }
    }

    pub(crate) fn clear_late_grammar_placeholder_mask(&self, buf: &mut [u32]) {
        for special in &self.constraint.special_token_terminals {
            if !self
                .constraint
                .is_late_grammar_placeholder_terminal(special.terminal_id)
            {
                continue;
            }
            let word = special.token_id as usize / 32;
            let bit = special.token_id % 32;
            if let Some(slot) = buf.get_mut(word) {
                *slot &= !(1u32 << bit);
            }
        }
    }

    fn fill_blocked_seed_dense(
        &self,
        terminals_disallowed: &TerminalsDisallowed,
        blocked: &mut Vec<u64>,
    ) {
        const MAX_FALLBACK_MASKS: usize = 512;

        let profile = std::env::var_os("GLRMASK_PROFILE_SEED_EXCLUSIONS").is_some();
        let started = profile.then(Instant::now);
        blocked.clear();
        blocked.resize(self.constraint.seed_universe_dense.len(), 0);
        if terminals_disallowed.is_empty() {
            return;
        }

        let mut missing_pairs = SmallVec::<[(u32, TerminalID); 4]>::new();
        for (&continuation_tokenizer_state, terminals) in terminals_disallowed.iter() {
            for &terminal_id in terminals.iter() {
                if let Some(mask) = self
                    .constraint
                    .seed_terminal_dense
                    .get(&(continuation_tokenizer_state, terminal_id))
                {
                    for (blocked_word, mask_word) in blocked.iter_mut().zip(mask.iter()) {
                        *blocked_word |= mask_word;
                    }
                } else if !self.constraint.possible_matches_complete {
                    // IMPORTANT: this is a legacy-only escape hatch. The
                    // dynamic possible-matches fallback is terrible and will
                    // be removed. New compiler paths MUST provide complete
                    // exact possible matches instead of reaching this branch.
                    // DO NOT REMOVE OR WEAKEN THIS COMMENT.
                    missing_pairs.push((continuation_tokenizer_state, terminal_id));
                }
            }
        }

        let mut cache_hits = 0usize;
        let mut uncached = SmallVec::<[(u32, TerminalID); 4]>::new();
        if !missing_pairs.is_empty() {
            let cache = self
                .constraint
                .seed_terminal_dense_fallback
                .lock()
                .expect("seed exclusion cache poisoned");
            for &pair in &missing_pairs {
                if let Some(mask) = cache.get(&pair) {
                    cache_hits += 1;
                    for (blocked_word, mask_word) in blocked.iter_mut().zip(mask.iter()) {
                        *blocked_word |= mask_word;
                    }
                } else if !uncached.contains(&pair) {
                    uncached.push(pair);
                }
            }
        }

        let dynamic_started = profile.then(Instant::now);
        for (continuation_tokenizer_state, terminal_id) in uncached.iter().copied() {
            let exclusions = TerminalsDisallowed::new()
                .with_insert(continuation_tokenizer_state, terminal_id);
            let mut computed = vec![0u64; blocked.len()];
            super::dynamic_mask::or_blocked_internal_tokens_for_exclusions(
                self.constraint,
                &exclusions,
                &mut computed,
            )
            .expect("unbounded seed-exclusion scan cannot fail");
            let computed: Arc<[u64]> = computed.into();

            let selected = {
                let mut cache = self
                    .constraint
                    .seed_terminal_dense_fallback
                    .lock()
                    .expect("seed exclusion cache poisoned");
                if let Some(existing) = cache.get(&(continuation_tokenizer_state, terminal_id)) {
                    Arc::clone(existing)
                } else {
                    if cache.len() < MAX_FALLBACK_MASKS {
                        cache.insert(
                            (continuation_tokenizer_state, terminal_id),
                            Arc::clone(&computed),
                        );
                    }
                    computed
                }
            };
            for (blocked_word, mask_word) in blocked.iter_mut().zip(selected.iter()) {
                *blocked_word |= mask_word;
            }
        }

        if let Some(started) = started {
            eprintln!(
                "[glrmask/profile][seed_exclusions] total_ns={} dynamic_ns={} remembered_pairs={} fallback_pairs={} cache_hits={} computed_pairs={} missing={:?}",
                elapsed_ns(started),
                dynamic_started.map(elapsed_ns).unwrap_or(0),
                terminals_disallowed
                    .iter()
                    .map(|(_, terminals)| terminals.len())
                    .sum::<usize>(),
                missing_pairs.len(),
                cache_hits,
                uncached.len(),
                missing_pairs,
            );
        }
    }

    fn try_fill_mask_single_path_direct(&self, buf: &mut [u32]) -> bool {
        if mask_inner_profile_enabled() || mask_delta_profile_enabled() {
            return false;
        }

        if self.state.is_empty()
            || self.state.len() > MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS
        {
            return false;
        }

        let mut paths = SmallVec::<[(u32, TerminalsDisallowed, SmallVec<[u32; MASK_SINGLE_PATH_DIRECT_INLINE_STACK_DEPTH]>); MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY]>::new();
        if self.state.len() < MASK_SINGLE_PATH_DIRECT_TWO_PASS_MIN_STATE_COUNT {
            // Below half the path budget, accepted multipath states are common
            // and a separate counting traversal costs more than it saves. Keep
            // the original one-pass admission/materialization algorithm.
            for (&original_tokenizer_state, gss) in &self.state {
                if gss.max_depth() > MASK_SINGLE_PATH_DIRECT_MAX_DEPTH {
                    return false;
                }

                let mut stack = SmallVec::<[u32; MASK_SINGLE_PATH_DIRECT_INLINE_STACK_DEPTH]>::new();
                if let Some(terminals_disallowed) = gss.single_path_top_first_and_acc(&mut stack) {
                    paths.push((original_tokenizer_state, terminals_disallowed, stack));
                    continue;
                }

                if mask_single_path_to_stacks_fallback_disabled() {
                    return false;
                }
                let remaining = MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS.saturating_sub(paths.len());
                let complete = gss.for_each_stack_top_first_bounded(
                    remaining,
                    |stack_top_first, terminals_disallowed| {
                        let mut path_stack = SmallVec::<[u32; MASK_SINGLE_PATH_DIRECT_INLINE_STACK_DEPTH]>::new();
                        path_stack.extend(stack_top_first.iter().copied());
                        paths.push((
                            original_tokenizer_state,
                            terminals_disallowed.clone(),
                            path_stack,
                        ));
                    },
                );
                if !complete {
                    return false;
                }
            }
        } else {
            // Once active tokenizer states consume at least half the path
            // budget, a small amount of branching is likely to reject the
            // specialized kernel. Count without cloning stack values first;
            // only accepted states pay to materialize concrete stacks.
            let mut all_single_path = true;
            for (&original_tokenizer_state, gss) in &self.state {
                if gss.max_depth() > MASK_SINGLE_PATH_DIRECT_MAX_DEPTH {
                    return false;
                }

                let mut stack = SmallVec::<[u32; MASK_SINGLE_PATH_DIRECT_INLINE_STACK_DEPTH]>::new();
                let Some(terminals_disallowed) = gss.single_path_top_first_and_acc(&mut stack) else {
                    all_single_path = false;
                    break;
                };
                paths.push((original_tokenizer_state, terminals_disallowed, stack));
            }

            if !all_single_path {
                if mask_single_path_to_stacks_fallback_disabled() {
                    return false;
                }

                let mut total_paths = 0usize;
                let mut total_stack_values = 0usize;
                for gss in self.state.values() {
                    if gss.max_depth() > MASK_SINGLE_PATH_DIRECT_MAX_DEPTH {
                        return false;
                    }
                    let remaining =
                        MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS.saturating_sub(total_paths);
                    let complete = gss.for_each_stack_len_bounded(remaining, |stack_len, _| {
                        total_paths += 1;
                        total_stack_values = total_stack_values.saturating_add(stack_len);
                    });
                    if !complete
                        || total_stack_values > MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_STACK_VALUES
                    {
                        return false;
                    }
                }

                paths.clear();
                for (&original_tokenizer_state, gss) in &self.state {
                    let remaining =
                        MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS.saturating_sub(paths.len());
                    let complete = gss.for_each_stack_top_first_bounded(
                        remaining,
                        |stack_top_first, terminals_disallowed| {
                            let mut path_stack = SmallVec::<[u32; MASK_SINGLE_PATH_DIRECT_INLINE_STACK_DEPTH]>::new();
                            path_stack.extend(stack_top_first.iter().copied());
                            paths.push((
                                original_tokenizer_state,
                                terminals_disallowed.clone(),
                                path_stack,
                            ));
                        },
                    );
                    debug_assert!(
                        complete,
                        "admitted GSS must materialize within the path budget"
                    );
                    if !complete {
                        return false;
                    }
                }
            }
        }
        if paths.iter().any(|(tokenizer_state, _, _)| {
            self.constraint
                .internal_tsids_for_state(*tokenizer_state)
                .len()
                != 1
        }) {
            return false;
        }
        let Some(total_stack_values) =
            single_path_direct_stack_work(paths.iter().map(|(_, _, stack)| stack.len()))
        else {
            return false;
        };
        if self.constraint.runtime_parser_dwa_state_count() == 0 {
            return false;
        }

        let mut plan_ops =
            SmallVec::<[SinglePathDirectPlanOp<'_>; MASK_SINGLE_PATH_DIRECT_MAX_PLAN_OPS]>::new();
        let mut stack_plans = SmallVec::<
            [SinglePathDirectStackPlan; MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY],
        >::new();
        let mut path_plan_indices =
            SmallVec::<[u8; MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY]>::new();
        let mut repeated_stack_values = 0usize;

        // Planning pays only when repeated parser-stack traversal dominates the
        // unique work.  This replaces the old 32-path switch with a direct cost
        // comparison: three copies of one deep stack can qualify, while 64
        // unrelated stacks do not build programs merely because the frontier is
        // wide.  First group stacks without touching the parser DWA; compile
        // programs only after the reuse test succeeds.
        if paths.len() >= 3 {
            for (path_index, (_, _, stack)) in paths.iter().enumerate() {
                let stack_fingerprint = single_path_direct_stack_fingerprint(stack);
                let existing = stack_plans.iter().position(|plan| {
                    plan.stack_fingerprint == stack_fingerprint
                        && paths[plan.representative_path].2.as_slice() == stack.as_slice()
                });
                let plan_index = if let Some(existing) = existing {
                    repeated_stack_values = repeated_stack_values.saturating_add(stack.len());
                    existing
                } else {
                    let plan_index = stack_plans.len();
                    stack_plans.push(SinglePathDirectStackPlan {
                        representative_path: path_index,
                        stack_fingerprint,
                        ops_start: 0,
                        ops_end: 0,
                    });
                    plan_index
                };
                path_plan_indices.push(plan_index as u8);
            }
        }

        let should_build_stack_plans = path_plan_indices.len() == paths.len()
            && single_path_direct_plan_reuse_dominates(
                paths.len(),
                total_stack_values,
                repeated_stack_values,
            );
        let mut plans_complete = should_build_stack_plans;
        if should_build_stack_plans {
            'build_plans: for plan_index in 0..stack_plans.len() {
                let representative_path = stack_plans[plan_index].representative_path;
                let stack = &paths[representative_path].2;
                let ops_start = plan_ops.len();
                let mut dwa_state_id = self.constraint.runtime_parser_dwa_start_state();
                let mut stack_idx = 0usize;

                loop {
                    if let Some(final_weight) =
                        self.constraint.runtime_parser_dwa_final_weight(dwa_state_id)
                    {
                        if plan_ops.len() == MASK_SINGLE_PATH_DIRECT_MAX_PLAN_OPS {
                            plans_complete = false;
                            break 'build_plans;
                        }
                        plan_ops.push(SinglePathDirectPlanOp::Merge(final_weight));
                    }

                    let Some(&parser_state) = stack.get(stack_idx) else {
                        break;
                    };
                    stack_idx += 1;

                    let positive_label = encode_positive_label(parser_state);
                    if stack_idx == 1 {
                        let has_direct_regular_acceptance = self
                            .constraint
                            .direct_regular_wide_acceptance_for_parser_state(parser_state)
                            .is_some()
                            || self
                                .constraint
                                .for_each_direct_regular_l1_acceptance(parser_state, |_| {});
                        if has_direct_regular_acceptance {
                            plans_complete = false;
                            break 'build_plans;
                        }
                        if let Some(accept_weight) =
                            self.constraint.runtime_parser_top_accept(positive_label)
                        {
                            if plan_ops.len() == MASK_SINGLE_PATH_DIRECT_MAX_PLAN_OPS {
                                plans_complete = false;
                                break 'build_plans;
                            }
                            plan_ops.push(SinglePathDirectPlanOp::Merge(accept_weight));
                        }
                        let accept_parts =
                            self.constraint.runtime_parser_top_accept_parts(positive_label);
                        if !accept_parts.is_empty() {
                            for accept_weight in accept_parts {
                                if plan_ops.len() == MASK_SINGLE_PATH_DIRECT_MAX_PLAN_OPS {
                                    plans_complete = false;
                                    break 'build_plans;
                                }
                                plan_ops.push(SinglePathDirectPlanOp::Merge(accept_weight));
                            }
                        }
                    }

                    let Some((target, weight)) = self
                        .constraint
                        .runtime_parser_dwa_transition(dwa_state_id, parser_state)
                    else {
                        break;
                    };
                    if plan_ops.len() == MASK_SINGLE_PATH_DIRECT_MAX_PLAN_OPS {
                        plans_complete = false;
                        break 'build_plans;
                    }
                    plan_ops.push(SinglePathDirectPlanOp::Intersect(weight));
                    dwa_state_id = target;
                }

                stack_plans[plan_index].ops_start = ops_start;
                stack_plans[plan_index].ops_end = plan_ops.len();
            }
        }
        let use_stack_plans = plans_complete && should_build_stack_plans;
        buf.fill(0);

        let precomputed = &self.constraint.weight_token_dense_masks;
        let dense_words = self.constraint.internal_token_dense_words;
        let (mut merged, mut output_scratch, mut single_path_aux, mut single_path_acc) = {
            let mut scratch = self.mask_scratch.lock().unwrap();
            (
                std::mem::take(&mut scratch.merged_dense),
                std::mem::take(&mut scratch.output_buf),
                std::mem::take(&mut scratch.single_path_aux_dense),
                std::mem::take(&mut scratch.single_path_acc_dense),
            )
        };
        merged.clear();
        merged.resize(dense_words, 0);
        let mut used_direct_final = false;
        let mut direct_buf_dirty = false;

        let restore_scratch = |
            merged: Vec<u64>,
            output_scratch: Vec<u32>,
            single_path_aux: Vec<u64>,
            single_path_acc: Vec<u64>,
        | {
            let mut scratch = self.mask_scratch.lock().unwrap();
            scratch.merged_dense = merged;
            scratch.chain_merged_dense.clear();
            scratch.output_buf = output_scratch;
            scratch.single_path_aux_dense = single_path_aux;
            scratch.single_path_acc_dense = single_path_acc;
        };

        if use_stack_plans {
            for (path_index, (original_tokenizer_state, terminals_disallowed, _)) in
                paths.iter().enumerate()
            {
                let internal_tsid = self
                    .constraint
                    .internal_tsid_for_state(*original_tokenizer_state);
                let seed_base = &self.constraint.seed_universe_dense;
                let mut dense_is_seed = terminals_disallowed.is_empty();
                if dense_is_seed {
                    if seed_base.is_empty() {
                        continue;
                    }
                } else if !self.fill_single_path_seed_dense(
                    terminals_disallowed,
                    &mut single_path_aux,
                    &mut single_path_acc,
                ) {
                    continue;
                }

                let plan = &stack_plans[path_plan_indices[path_index] as usize];
                for op in &plan_ops[plan.ops_start..plan.ops_end] {
                    match *op {
                        SinglePathDirectPlanOp::Merge(weight) => {
                            used_direct_final = true;
                            let dense = if dense_is_seed {
                                seed_base.as_ref()
                            } else {
                                single_path_acc.as_slice()
                            };
                            self.merge_single_path_final_weight_to_internal(
                                weight,
                                internal_tsid,
                                dense,
                                precomputed,
                                &mut merged,
                                Some(&mut *buf),
                                &mut direct_buf_dirty,
                            );
                        }
                        SinglePathDirectPlanOp::Intersect(weight) => {
                            if dense_is_seed {
                                if weight.is_full() {
                                    continue;
                                }
                                if !materialize_single_path_seed_intersection(
                                    seed_base,
                                    &mut single_path_acc,
                                    internal_tsid,
                                    weight,
                                    self.constraint,
                                ) {
                                    break;
                                }
                                dense_is_seed = false;
                            } else if !Self::intersect_single_path_dense_with_weight_in_place(
                                &mut single_path_acc,
                                &mut single_path_aux,
                                internal_tsid,
                                weight,
                                self.constraint,
                            ) {
                                break;
                            }
                        }
                    }
                }
            }
        } else {
            for (original_tokenizer_state, terminals_disallowed, stack) in &paths {
                let internal_tsid = self
                    .constraint
                    .internal_tsid_for_state(*original_tokenizer_state);
                let seed_base = &self.constraint.seed_universe_dense;
                let mut dense_is_seed = terminals_disallowed.is_empty();
                if dense_is_seed {
                    if seed_base.is_empty() {
                        continue;
                    }
                } else if !self.fill_single_path_seed_dense(
                    terminals_disallowed,
                    &mut single_path_aux,
                    &mut single_path_acc,
                ) {
                    continue;
                }

                let mut dwa_state_id = self.constraint.runtime_parser_dwa_start_state();
                let mut stack_idx = 0usize;

                loop {
                    if let Some(final_weight) =
                        self.constraint.runtime_parser_dwa_final_weight(dwa_state_id)
                    {
                        used_direct_final = true;
                        let dense = if dense_is_seed {
                            seed_base.as_ref()
                        } else {
                            single_path_acc.as_slice()
                        };
                        self.merge_single_path_final_weight_to_internal(
                            final_weight,
                            internal_tsid,
                            dense,
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
                        let dense = if dense_is_seed {
                            seed_base.as_ref()
                        } else {
                            single_path_acc.as_slice()
                        };
                        let mut used_equivalent_wide_summary = false;
                        if let Some(summary) = self
                            .constraint
                            .direct_regular_wide_acceptance_for_parser_state(parser_state)
                            && let Some(accepted) = summary.dense_by_tsid.get(internal_tsid)
                        {
                            let n = dense.len().min(accepted.len()).min(merged.len());
                            for word in 0..n {
                                merged[word] |= dense[word] & accepted[word];
                            }
                            used_direct_final = true;
                            used_equivalent_wide_summary = true;
                        }

                        if !used_equivalent_wide_summary {
                            if let Some(accept_weight) =
                                self.constraint.runtime_parser_top_accept(positive_label)
                            {
                                used_direct_final = true;
                                self.merge_single_path_final_weight_to_internal(
                                    accept_weight,
                                    internal_tsid,
                                    dense,
                                    precomputed,
                                    &mut merged,
                                    Some(&mut *buf),
                                    &mut direct_buf_dirty,
                                );
                            }
                            let accept_parts =
                                self.constraint.runtime_parser_top_accept_parts(positive_label);
                            if !accept_parts.is_empty() {
                                used_direct_final = true;
                                for accept_weight in accept_parts {
                                    self.merge_single_path_final_weight_to_internal(
                                        accept_weight,
                                        internal_tsid,
                                        dense,
                                        precomputed,
                                        &mut merged,
                                        Some(&mut *buf),
                                        &mut direct_buf_dirty,
                                    );
                                }
                            }
                            let used_l1 = self.constraint.for_each_direct_regular_l1_acceptance(
                                parser_state,
                                |accept_weight| {
                                    self.merge_single_path_final_weight_to_internal(
                                        accept_weight,
                                        internal_tsid,
                                        dense,
                                        precomputed,
                                        &mut merged,
                                        Some(&mut *buf),
                                        &mut direct_buf_dirty,
                                    );
                                },
                            );
                            used_direct_final |= used_l1;
                        }
                    }

                    let Some((target, weight)) = self
                        .constraint
                        .runtime_parser_dwa_transition(dwa_state_id, parser_state)
                    else {
                        break;
                    };

                    if dense_is_seed {
                        if !weight.is_full() {
                            if !materialize_single_path_seed_intersection(
                                seed_base,
                                &mut single_path_acc,
                                internal_tsid,
                                weight,
                                self.constraint,
                            ) {
                                break;
                            }
                            dense_is_seed = false;
                        }
                    } else if !Self::intersect_single_path_dense_with_weight_in_place(
                        &mut single_path_acc,
                        &mut single_path_aux,
                        internal_tsid,
                        weight,
                        self.constraint,
                    ) {
                        break;
                    }
                    dwa_state_id = target;
                }
            }
        }
        if !used_direct_final && !self.is_accepting() {
            restore_scratch(merged, output_scratch, single_path_aux, single_path_acc);
            return false;
        }

        if merged.iter().any(|&word| word != 0) {
            let buf_zeroed = !direct_buf_dirty;
            self.constraint.or_internal_dense_to_buf_fast_with_scratch(
                &merged,
                buf,
                buf_zeroed,
                &mut output_scratch,
            );
        }
        if direct_buf_dirty {
            self.store_mask_cache_reuse_dense(buf);
        } else {
            self.store_mask_cache(buf, &merged);
        }
        restore_scratch(merged, output_scratch, single_path_aux, single_path_acc);
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

    fn fill_single_path_seed_dense(
        &self,
        terminals_disallowed: &TerminalsDisallowed,
        aux: &mut Vec<u64>,
        dense: &mut Vec<u64>,
    ) -> bool {
        let base = &self.constraint.seed_universe_dense;
        if base.is_empty() {
            dense.clear();
            return false;
        }

        dense.clear();
        dense.extend_from_slice(base);

        if terminals_disallowed.is_empty() {
            return true;
        }

        self.fill_blocked_seed_dense(terminals_disallowed, aux);

        if aux.iter().all(|&word| word == 0) {
            return true;
        }

        let mut any = false;
        for (allowed_word, blocked_word) in dense.iter_mut().zip(aux.iter().copied()) {
            *allowed_word &= !blocked_word;
            any |= *allowed_word != 0;
        }
        any
    }

    fn intersect_single_path_dense_with_weight_in_place(
        dense: &mut Vec<u64>,
        aux: &mut Vec<u64>,
        internal_tsid: u32,
        weight: RuntimeWeightRef<'_>,
        constraint: &Constraint,
    ) -> bool {
        if weight.is_full() {
            return dense.iter().any(|&word| word != 0);
        }

        let Some(token_set) = weight.token_set_for_tsid(internal_tsid) else {
            dense.fill(0);
            return false;
        };
        if let Some(mask) = constraint.runtime_token_set_dense_mask(token_set) {
            let mut any = false;
            for (idx, dense_word) in dense.iter_mut().enumerate() {
                *dense_word &= mask.get(idx).copied().unwrap_or(0);
                any |= *dense_word != 0;
            }
            return any;
        }

        aux.clear();
        aux.resize(dense.len(), 0);
        DenseMaskAcc::for_each_runtime_token_range_word(token_set, dense.len(), |word_idx, token_mask| {
            aux[word_idx] |= dense[word_idx] & token_mask;
        });
        std::mem::swap(dense, aux);
        dense.iter().any(|&word| word != 0)
    }

    fn merge_single_path_final_weight_to_internal(
        &self,
        final_weight: RuntimeWeightRef<'_>,
        internal_tsid: u32,
        dense: &[u64],
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
        mut direct_buf: Option<&mut [u32]>,
        direct_buf_dirty: &mut bool,
    ) -> bool {
        if final_weight.is_full() {
            let n = dense.len().min(merged.len());
            for idx in 0..n {
                merged[idx] |= dense[idx];
            }
            return false;
        }

        let Some(token_set) = final_weight.token_set_for_tsid(internal_tsid) else {
            return true;
        };
        if let (Some(buf), RuntimeTokenSetRef::Materialized(token_set)) =
            (direct_buf.as_deref_mut(), token_set)
        {
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
                return true;
            }
            if self
                .constraint
                .or_weight_token_set_to_buf_if_contained(dense, token_set, buf)
            {
                *direct_buf_dirty = true;
                return true;
            }
        }

        DenseMaskAcc::or_dense_and_runtime_token_set_into(dense, token_set, precomputed, merged);
        false
    }

    fn terminals_disallowed_to_dense_acc(
        &self,
        terminals_disallowed: &TerminalsDisallowed,
        tokenizer_state: u32,
    ) -> Option<DenseMaskAcc> {
        let internal_tsids = self.constraint.internal_tsids_for_state(tokenizer_state);
        let base = &self.constraint.seed_universe_dense;
        if base.is_empty() || internal_tsids.is_empty() {
            return None;
        }
        if terminals_disallowed.is_empty() {
            return DenseMaskAcc::from_dense_arc_for_tsids(
                internal_tsids,
                Arc::clone(base),
            );
        }

        let mut blocked_only = Vec::new();
        self.fill_blocked_seed_dense(terminals_disallowed, &mut blocked_only);

        if blocked_only.iter().all(|&word| word == 0) {
            return DenseMaskAcc::from_dense_arc_for_tsids(
                internal_tsids,
                Arc::clone(base),
            );
        }

        let mut dense = base.to_vec();
        for (allowed_word, blocked_word) in dense.iter_mut().zip(blocked_only) {
            *allowed_word &= !blocked_word;
        }

        DenseMaskAcc::from_dense_arc_for_tsids(internal_tsids, dense.into())
    }

    fn merge_final_weight_to_internal(
        &self,
        final_weight: RuntimeWeightRef<'_>,
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
                let Some(token_set) = final_weight.token_set_for_tsid(*tsid) else {
                    continue;
                };

                let handled_directly = if let (Some(buf), RuntimeTokenSetRef::Materialized(token_set)) =
                    (direct_buf.as_deref_mut(), token_set)
                {
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
                    DenseMaskAcc::or_dense_and_runtime_token_set_into(dense, token_set, precomputed, merged);
                    all_direct = false;
                }
            }
        }

        all_direct
    }

    fn merge_final_weight_for_accs(
        &self,
        final_weight: RuntimeWeightRef<'_>,
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
        final_weight: RuntimeWeightRef<'_>,
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

    fn try_seed_direct_regular_wide_frontier(
        &self,
        gss: &ParserGSS,
        original_tokenizer_state: u32,
        start_final_weight: Option<RuntimeWeightRef<'_>>,
        start_dwa_state: u32,
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
        direct_buf: &mut Option<&mut [u32]>,
        direct_buf_possible: &mut bool,
        direct_buf_used: &mut bool,
        direct_buf_dirty: &mut bool,
    ) -> bool {
        if !self
            .constraint
            .runtime_parser_dwa_row_is_empty(start_dwa_state)
        {
            return false;
        }
        let Some(summary) = self.constraint.direct_regular_wide_frontier_for_gss(gss) else {
            return false;
        };
        let Some(terminals_disallowed) = gss.uniform_accumulator() else {
            return false;
        };

        let Some(dense_acc) = self.terminals_disallowed_to_dense_acc(
            &terminals_disallowed,
            original_tokenizer_state,
        ) else {
            return true;
        };

        *direct_buf_used = true;
        if let Some(weight) = start_final_weight {
            *direct_buf_possible &= self.merge_final_weight_to_internal(
                weight,
                &dense_acc,
                precomputed,
                merged,
                direct_buf.as_deref_mut(),
                direct_buf_dirty,
            );
        }
        for (tsid, dense) in &dense_acc.0 {
            let Some(accepted) = summary.dense_by_tsid.get(*tsid) else {
                continue;
            };
            let n = dense.len().min(accepted.len()).min(merged.len());
            for index in 0..n {
                merged[index] |= dense[index] & accepted[index];
            }
        }
        *direct_buf_possible = false;
        true
    }

    fn seed_mask_queue_merged(
        &self,
        start_final_weight: Option<RuntimeWeightRef<'_>>,
        start_dwa_state: u32,
        precomputed: &DenseTokenMaskCache,
        transition_gss_cache: &mut FxHashMap<DenseGssTransitionKey, DenseMaskGSS>,
        transition_intersection_cache: &mut DenseTokenSetIntersectionSmallCache,
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
            if self.try_seed_direct_regular_wide_frontier(
                gss,
                original_tokenizer_state,
                start_final_weight,
                start_dwa_state,
                precomputed,
                merged,
                direct_buf,
                direct_buf_possible,
                direct_buf_used,
                direct_buf_dirty,
            ) {
                continue;
            }

            let seed_decompose_start = if profile.is_some() {
                Some(Instant::now())
            } else {
                None
            };
            // Exclusion pruning depends on which parser top makes an overlapping
            // terminal actionable. Transforming accumulators before decomposing
            // the parser frontier loses that correlation: the union of all top
            // states can let a terminal from one parser path rescue a blocked
            // token on another. Decompose by top first, then transform every
            // accumulator in that top-local sub-GSS with exactly that top state.
            let mut decomposed = Vec::new();
            gss.for_each_decomposed(|parser_state, popped| {
                let dense = popped.apply_and_prune(|terminals_disallowed| {
                    self.terminals_disallowed_to_dense_acc(
                        terminals_disallowed,
                        original_tokenizer_state,
                    )
                });
                if !dense.is_empty() {
                    decomposed.push((parser_state, dense));
                }
            });

            // Empty parser-stack paths have no actionable parser top. Preserve
            // their root accumulators separately for the parser-DWA start final
            // weight, matching apply_transform_and_decompose's old root handling.
            let mut root_accs = Vec::new();
            gss.isolate(None).for_each_acc(|terminals_disallowed| {
                if let Some(acc) = self.terminals_disallowed_to_dense_acc(
                    terminals_disallowed,
                    original_tokenizer_state,
                ) {
                    root_accs.push(acc);
                }
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
                if let Some(accept_weight) =
                    self.constraint.runtime_parser_top_accept(positive_label)
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
                let accept_parts =
                    self.constraint.runtime_parser_top_accept_parts(positive_label);
                if !accept_parts.is_empty() {
                    let accumulate_start = if profile.is_some() {
                        Some(Instant::now())
                    } else {
                        None
                    };
                    *direct_buf_used = true;
                    for accept_weight in accept_parts {
                        *direct_buf_possible &= self.merge_final_weight_for_gss(
                            accept_weight,
                            popped,
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
                let accumulate_start = profile.as_ref().map(|_| Instant::now());
                let mut l1_direct_possible = true;
                let used_l1 = self.constraint.for_each_direct_regular_l1_acceptance(
                    *parser_state,
                    |accept_weight| {
                        l1_direct_possible &= self.merge_final_weight_for_gss(
                            accept_weight,
                            popped,
                            precomputed,
                            merged,
                            direct_buf,
                            direct_buf_dirty,
                        );
                    },
                );
                if used_l1 {
                    *direct_buf_used = true;
                    *direct_buf_possible &= l1_direct_possible;
                    if let (Some(profile), Some(start)) = (profile.as_mut(), accumulate_start) {
                        profile.token_accumulation_ns += elapsed_ns(start);
                    }
                }
                queue.record_seed_decompose_callback();
                enqueue_parser_state_transition(
                    self.constraint,
                    queue,
                    start_dwa_state,
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

    fn fill_mask_indexed_dag(&self, buf: &mut [u32], force: bool) -> bool {
        if (!force && !indexed_dag_mask_enabled()) || !self.has_parser_ambiguity() {
            return false;
        }
        let parser_dwa = self.constraint.parser_dwa();
        if self.state.is_empty() || parser_dwa.states().is_empty() {
            return false;
        }
        if self.constraint.indexed_dag_dense_transitions.len() != parser_dwa.states().len()
            || self.constraint.indexed_dag_dense_finals.len() != parser_dwa.states().len()
        {
            return false;
        }

        let profile = indexed_dag_mask_profile_enabled();
        let total_started = profile.then(Instant::now);
        let precomputed = &self.constraint.weight_token_dense_masks;
        let dense_words = self.constraint.internal_token_dense_words;
        let (mut merged, mut indexed_runtime) = {
            let mut scratch = self.mask_scratch.lock().unwrap();
            (
                std::mem::take(&mut scratch.merged_dense),
                std::mem::take(&mut scratch.indexed_dag_mask),
            )
        };
        indexed_runtime.begin_mask();
        merged.clear();
        merged.resize(dense_words, 0);
        buf.fill(0);
        let mut accepted: Option<DenseMaskAcc> = None;
        let mut index_ns = 0u64;
        let mut eval_ns = 0u64;

        let merge_accepted = |accepted: &mut Option<DenseMaskAcc>, incoming: Option<DenseMaskAcc>| {
            let Some(incoming) = incoming else {
                return;
            };
            *accepted = Some(match accepted.take() {
                Some(existing) => existing.merge(&incoming),
                None => incoming,
            });
        };

        let start_state = parser_dwa.start_state();
        let start_final_weight = parser_dwa.states()[start_state as usize].final_weight.as_ref();
        let start_transitions = &self.constraint.dwa_fast_transitions[start_state as usize];
        let mut seed_intersections = DenseTokenSetIntersectionSmallCache::new();
        let lower_entries_before = indexed_runtime.lower_memo.len();
        let segment_entries_before = indexed_runtime.segment_memo.len();
        let accumulator_entries_before = indexed_runtime.accumulators.len();
        let mut seed_gsses = Vec::<DenseMaskGSS>::new();
        let mut seed_targets = Vec::<u32>::new();
        let mut seed_weights = Vec::<Weight>::new();

        for (&tokenizer_state, gss) in &self.state {
            if gss.is_empty() {
                continue;
            }
            let root_dense = gss.isolate(None).apply_and_prune(|terminals_disallowed| {
                self.terminals_disallowed_to_dense_acc(terminals_disallowed, tokenizer_state)
            });
            if let Some(final_weight) = start_final_weight {
                root_dense.for_each_acc(|accumulator| {
                    merge_accepted(
                        &mut accepted,
                        accumulator.intersect_with_weight_small_cached(
                            final_weight,
                            precomputed,
                            &mut seed_intersections,
                        ),
                    );
                });
            }

            gss.for_each_decomposed(|parser_state, popped| {
                let dense = popped.apply_and_prune(|terminals_disallowed| {
                    self.terminals_disallowed_to_dense_acc(
                        terminals_disallowed,
                        tokenizer_state,
                    )
                });
                if dense.is_empty() {
                    return;
                }
                if let Some(final_weight) = start_final_weight {
                    dense.for_each_acc(|accumulator| {
                        merge_accepted(
                            &mut accepted,
                            accumulator.intersect_with_weight_small_cached(
                                final_weight,
                                precomputed,
                                &mut seed_intersections,
                            ),
                        );
                    });
                }
                let positive_label = encode_positive_label(parser_state);
                if let Some(top_weight) = self
                    .constraint
                    .parser_top_accept
                    .get(&positive_label)
                    .or_else(|| self.constraint.parser_top_accept.get(&DEFAULT_LABEL))
                {
                    dense.for_each_acc(|accumulator| {
                        merge_accepted(
                            &mut accepted,
                            accumulator.intersect_with_weight_small_cached(
                                top_weight,
                                precomputed,
                                &mut seed_intersections,
                            ),
                        );
                    });
                }
                if let Some(top_parts) = self
                    .constraint
                    .parser_top_accept_parts
                    .get(&positive_label)
                    .or_else(|| {
                        self.constraint
                            .parser_top_accept_parts
                            .get(&DEFAULT_LABEL)
                    })
                {
                    for top_weight in top_parts {
                        dense.for_each_acc(|accumulator| {
                            merge_accepted(
                                &mut accepted,
                            accumulator.intersect_with_weight_small_cached(
                                    top_weight,
                                    precomputed,
                                    &mut seed_intersections,
                                ),
                            );
                        });
                    }
                }
                self.constraint.for_each_direct_regular_l1_acceptance(
                    parser_state,
                    |top_weight| {
                        let RuntimeWeightRef::Materialized(top_weight) = top_weight else {
                            return;
                        };
                        dense.for_each_acc(|accumulator| {
                            merge_accepted(
                                &mut accepted,
                                accumulator.intersect_with_weight_small_cached(
                                    top_weight,
                                    precomputed,
                                    &mut seed_intersections,
                                ),
                            );
                        });
                    },
                );
                let Some((target, transition_weight)) = self
                    .constraint
                    .fast_parser_dwa_transition(start_transitions, parser_state)
                else {
                    return;
                };
                seed_gsses.push(dense);
                seed_targets.push(target);
                seed_weights.push(transition_weight.clone());
            });
        }

        let mut indexed_nodes = 0usize;
        let mut upper_entries = 0usize;
        let mut lower_entries = 0usize;
        let mut segment_entries = 0usize;
        let mut upper_calls = 0u64;
        let mut upper_hits = 0u64;
        let mut lower_calls = 0u64;
        let mut lower_hits = 0u64;
        let mut segment_calls = 0u64;
        let mut segment_hits = 0u64;
        let mut memo_result_entries = 0u64;
        let mut memo_dense_words = 0u64;
        let mut memo_nonzero_words = 0u64;
        let mut memo_max_nonzero_words = 0u64;

        if !seed_gsses.is_empty() {
            let started = profile.then(Instant::now);
            let index_nodes = std::mem::take(&mut indexed_runtime.index_nodes);
            let index_roots = std::mem::take(&mut indexed_runtime.index_roots);
            let index_lower_ids = std::mem::take(&mut indexed_runtime.index_lower_ids);
            let index_upper_ids = std::mem::take(&mut indexed_runtime.index_upper_ids);
            let (dag, roots, index_lower_ids, index_upper_ids) =
                DenseMaskGSS::indexed_dag_many_reusing(
                    &seed_gsses,
                    index_nodes,
                    index_roots,
                    index_lower_ids,
                    index_upper_ids,
                );
            if let Some(started) = started {
                index_ns += elapsed_ns(started);
            }
            indexed_nodes = dag.nodes.len();
            let started = profile.then(Instant::now);
            let mut evaluator = IndexedDagMaskEvaluator::new(
                self.constraint,
                &dag,
                precomputed,
                &mut indexed_runtime,
            );
            for (((root, target), transition_weight), _) in roots
                .iter()
                .copied()
                .zip(seed_targets.into_iter())
                .zip(seed_weights.into_iter())
                .zip(seed_gsses.into_iter())
            {
                let result = evaluator.eval_upper(target, root);
                let result = result.and_then(|result| {
                    result.intersect_with_weight_small_cached(
                        &transition_weight,
                        precomputed,
                        &mut seed_intersections,
                    )
                });
                merge_accepted(&mut accepted, result);
            }
            if let Some(started) = started {
                eval_ns += elapsed_ns(started);
            }
            upper_entries = evaluator.upper_memo.len();
            lower_entries = evaluator.runtime.lower_memo.len();
            segment_entries = evaluator.runtime.segment_memo.len();
            upper_calls = evaluator.upper_calls;
            upper_hits = evaluator.upper_hits;
            lower_calls = evaluator.lower_calls;
            lower_hits = evaluator.lower_hits;
            segment_calls = evaluator.segment_calls;
            segment_hits = evaluator.segment_hits;
            memo_result_entries = evaluator.memo_result_entries;
            memo_dense_words = evaluator.memo_dense_words;
            memo_nonzero_words = evaluator.memo_nonzero_words;
            memo_max_nonzero_words = evaluator.memo_max_nonzero_words;
            indexed_runtime.retain_index_scratch(
                dag,
                roots,
                index_lower_ids,
                index_upper_ids,
            );
        }

        if let Some(accepted) = accepted {
            accepted.or_into_merged(&mut merged);
        }
        self.constraint.or_internal_dense_to_buf_fast(&merged, buf, true);
        self.store_mask_cache(buf, &merged);
        let accumulator_entries_total = indexed_runtime.accumulators.len();
        indexed_runtime.prune_stale_sources_if_needed();
        {
            let mut scratch = self.mask_scratch.lock().unwrap();
            scratch.merged_dense = merged;
            scratch.indexed_dag_mask = indexed_runtime;
        }
        if let Some(started) = total_started {
            eprintln!(
                "[glrmask/profile][indexed_dag_mask] total_ns={} index_ns={} eval_ns={} indexed_nodes={} upper_entries={} lower_entries_added={} lower_entries_total={} segment_entries_added={} segment_entries_total={} accumulator_entries_added={} accumulator_entries_total={} upper_calls={} upper_hits={} lower_calls={} lower_hits={} segment_calls={} segment_hits={} memo_result_entries={} memo_dense_words={} memo_nonzero_words={} memo_max_nonzero_words={}",
                elapsed_ns(started),
                index_ns,
                eval_ns,
                indexed_nodes,
                upper_entries,
                lower_entries.saturating_sub(lower_entries_before),
                lower_entries,
                segment_entries.saturating_sub(segment_entries_before),
                segment_entries,
                accumulator_entries_total.saturating_sub(accumulator_entries_before),
                accumulator_entries_total,
                upper_calls,
                upper_hits,
                lower_calls,
                lower_hits,
                segment_calls,
                segment_hits,
                memo_result_entries,
                memo_dense_words,
                memo_nonzero_words,
                memo_max_nonzero_words,
            );
        }
        true
    }

    fn try_fill_mask_indexed_dag(&self, buf: &mut [u32]) -> bool {
        self.fill_mask_indexed_dag(buf, false)
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

        if self.try_fill_mask_indexed_dag(buf) {
            return total_start.map(|start| MaskProfile {
                total_ns: elapsed_ns(start),
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
        if self.state.is_empty() || self.constraint.runtime_parser_dwa_state_count() == 0 {
            buf.fill(0);
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
        let mut transition_intersection_cache = DenseTokenSetIntersectionSmallCache::new();

        let mut merged = {
            let mut scratch = self.mask_scratch.lock().unwrap();
            std::mem::take(&mut scratch.merged_dense)
        };

        buf.fill(0);
        merged.clear();
        merged.resize(dense_words, 0);
        let mut direct_buf = None;
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

        let start_state = self.constraint.runtime_parser_dwa_start_state();
        let start_final_weight = self.constraint.runtime_parser_dwa_final_weight(start_state);

        self.seed_mask_queue_merged(
            start_final_weight,
            start_state,
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

            if let Some(final_weight) = self.constraint.runtime_parser_dwa_final_weight(wa_state) {
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
                    self.constraint,
                    &mut queue,
                    wa_state,
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
                    let n_internal = self.constraint.internal_token_count();
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
        let cache_start = profile.as_ref().map(|_| Instant::now());
        if can_use_merged_cache {
            if reuse_existing_cache_dense {
                self.touch_mask_cache_generation();
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
                "[glrmask/debug][mask_inner] queue_mode={:?} total_ns={} seed_decompose_ns={} queue_pop_ns={} loop_decompose_ns={} transition_lookup_ns={} transition_apply_ns={} transition_apply_intersect_ns={} transition_apply_gss_ns={} token_accumulation_ns={} enqueue_merge_ns={} queue_lookup_ns={} queue_merge_ns={} queue_insert_ns={} insert_without_merge_count={} fuse_ns={} finalize_ns={} finalize_zero_ns={} finalize_dense_to_buf_ns={} finalize_cache_ns={} delta_prev_available={} delta_added_bits={} delta_removed_bits={} delta_unchanged_words={} delta_unchanged_bits={} delta_added_cost={} delta_removed_cost={} delta_copy_cost_words={} delta_scratch_estimated_cost={} delta_estimated_cost={} delta_estimated_savings={} delta_used_seed={} delta_added_word_group_hits={} delta_added_word_group_entries={} delta_removed_word_group_hits={} delta_removed_word_group_entries={} delta_added_byte_group_hits={} delta_added_byte_group_entries={} delta_removed_byte_group_hits={} delta_removed_byte_group_entries={} delta_added_token_iterations={} delta_added_token_entries={} delta_removed_token_iterations={} delta_removed_token_entries={} finalize_equal_dense_copy_seed={} finalize_delta_replay={} finalize_scratch_rebuild={} dense_words_visited={} dense_complement_path_used={} dense_normal_full_word_hits={} dense_normal_group_complement_hits={} dense_complement_full_word_hits={} dense_complement_full_byte_groups={} dense_complement_full_nibble_groups={} dense_complement_remaining_bits={} dense_normal_token_iterations={} dense_complement_token_iterations={} dense_normal_sparse_entries={} dense_normal_group_complement_sparse_entries={} dense_complement_sparse_entries={} dense_complement_heavy_dense_clears={} dense_complement_max_sparse_span={} dense_group_or_sparse_entries={} dense_group_andnot_sparse_entries={} dense_group_sparse_groups={} dense_group_sparse_total_entries={} dense_group_sparse_max_entries={} dense_group_dense_storage_words={} dense_raw_token_sparse_entries={} other_ns={} enqueue_calls={} merge_hits={} popped_items={} parser_dwa_transitions_enqueued={}",
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
                self.constraint.internal_token_buf_flat_len(),
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

    /// Return the allowed-token mask as a packed `u32` bitset.
    pub fn mask(&self) -> Vec<u32> {
        let mut buf = vec![0u32; self.constraint.mask_len()];
        self.fill_mask(&mut buf);
        buf
    }

    pub(crate) fn prefill_mask_cache(&self) {
        if self
            .constraint
            .static_dynamic_overlay
            .as_ref()
            .is_some_and(|overlay| {
                (!overlay.segmented_parser_components.is_empty()
                    && std::env::var_os("GLRMASK_EXPERIMENT_SEGMENTED_PARSER_MASK").is_some())
                    || (overlay.segmented_mask_authoritative
                        && (!overlay.segmented_parser_components.is_empty()
                            || overlay.segmented_static_baseline))
            })
        {
            return;
        }
        let cache = self.mask_cache.lock().unwrap();
        if cache
            .as_ref()
            .is_some_and(|cache_data| cache_data.generation == self.generation)
        {
            return;
        }
        drop(cache);

        let mut buf = vec![0u32; self.constraint.mask_len()];
        if self.constraint.uses_dynamic_runtime() {
            self.fill_mask_dynamic(&mut buf);
            self.store_mask_cache_reuse_dense(&buf);
        } else {
            self.fill_mask_uncached(&mut buf);
            self.update_control_special_token_mask(&mut buf);
            self.store_mask_cache_reuse_dense(&buf);
        }
    }

    /// Fill `buf` with the allowed-token mask.
    ///
    /// `buf` must contain at least [`crate::Constraint::mask_len`] words. Any extra
    /// words are cleared.

    fn static_mask_for_reset_branch(&self, gss: &ParserGSS, buf: &mut [u32]) {
        let reset_state = self.constraint.runtime_commit_initial_state();
        let mut shadow = self.clone();
        shadow.state.clear();
        shadow.state.insert_flat_alternative(reset_state, gss.clone());
        if let Some(factored) = shadow.lookahead_factored_mask_shadow() {
            factored.fill_mask_uncached(buf);
        } else {
            shadow.fill_mask_uncached(buf);
        }
        shadow.update_control_special_token_mask(buf);
    }

    fn add_admissible_scoped_ignore_tokens(&self, buf: &mut [u32]) {
        if self.constraint.static_dynamic_overlay.is_none()
            || std::env::var_os("GLRMASK_EXPERIMENT_SCOPED_IGNORE_EXACT_OVERLAY").is_none()
        {
            return;
        }
        let reset_state = self.constraint.runtime_commit_initial_state();
        let use_fusions = std::env::var_os("GLRMASK_EXPERIMENT_SCOPED_IGNORE_FUSIONS").is_some();
        let use_residual_possible_matches =
            std::env::var_os("GLRMASK_EXPERIMENT_SCOPED_IGNORE_RESIDUAL_PM").is_some();
        let mut suffix_mask = use_fusions.then(|| vec![0u32; buf.len()]);
        let mut candidate_mask = use_fusions.then(|| vec![0u32; buf.len()]);
        for (&tokenizer_state, gss) in self.state.iter() {
            if use_residual_possible_matches
                && tokenizer_state != reset_state
                && gss.all_accs_satisfy(|blocked: &TerminalsDisallowed| blocked.is_empty())
            {
                for &terminal in &self.constraint.table.skip_terminals {
                    if !stack_may_advance_on(&self.constraint.table, gss, terminal) {
                        continue;
                    }
                    self.constraint.visit_possible_match_original_tokens(
                        tokenizer_state,
                        terminal,
                        |token| set_original_mask_bit(buf, token),
                    );
                }
            }
            if tokenizer_state != reset_state
                || !gss.all_accs_satisfy(|blocked: &TerminalsDisallowed| blocked.is_empty())
            {
                continue;
            }
            for (terminal, tokens) in &self.constraint.scoped_ignore_only_tokens {
                if !stack_may_advance_on(&self.constraint.table, gss, *terminal) {
                    continue;
                }
                for &token in tokens.iter() {
                    set_original_mask_bit(buf, token);
                }
                if !use_fusions {
                    continue;
                }
                let Some((_, fusions)) = self
                    .constraint
                    .scoped_ignore_prefix_fusions
                    .iter()
                    .find(|(candidate, _)| candidate == terminal)
                else {
                    continue;
                };
                let suffix_mask = suffix_mask
                    .as_deref_mut()
                    .expect("fusion suffix mask was requested");
                suffix_mask.fill(0);
                // A completed Skip inside one model token resets the lexer but
                // leaves this exact parser GSS unchanged. Test the remaining
                // suffix against *that branch-local reset state*, never against
                // the global union mask: another residual tokenizer branch may
                // admit the same suffix token for unrelated reasons.
                self.static_mask_for_reset_branch(gss, suffix_mask);
                for &(fused, suffix) in fusions.iter() {
                    if original_mask_contains(suffix_mask, suffix) {
                        set_original_mask_bit(
                            candidate_mask
                                .as_deref_mut()
                                .expect("fusion candidate mask was requested"),
                            fused,
                        );
                    }
                }
            }
        }
        if let Some(candidate_mask) = candidate_mask.as_deref()
            && candidate_mask.iter().any(|&word| word != 0)
        {
            super::dynamic_mask::or_mask_dynamic_candidate_additions(self, buf, candidate_mask);
        }
    }

    fn lookahead_factored_mask_shadow(&self) -> Option<Self> {
        if self.constraint.static_dynamic_overlay.is_none()
            || std::env::var_os("GLRMASK_EXPERIMENT_MASK_LOOKAHEAD_FACTOR").is_none()
        {
            return None;
        }

        // Two factors are sufficient to expose the parent continuation for the
        // nested-subgrammar return shape. Keep the experiment configurable for
        // differential work, but do not accumulate every intermediate factor:
        // only the deepest certified subset can add anything the shallower
        // subset could add, while carrying at least the same lookahead guards.
        let max_factor_depth = std::env::var("GLRMASK_EXPERIMENT_MASK_LOOKAHEAD_FACTOR_MAX_DEPTH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2)
            .min(8);
        let mut additions = SmallVec::<[(u32, ParserGSS); 4]>::new();
        for (&tokenizer_state, gss) in self.state.iter() {
            let mut factor = gss.clone();
            let mut blocked = SmallVec::<[TerminalID; 8]>::new();
            let mut deepest = None::<ParserGSS>;
            let use_fast_factor = std::env::var_os(
                "GLRMASK_EXPERIMENT_FAST_LOOKAHEAD_FACTOR",
            )
            .is_some();
            let mut chain_forced = None::<BitSet>;
            let mut final_after = None::<BitSet>;
            let mut fast_guard_representable = true;
            for _ in 0..max_factor_depth {
                if use_fast_factor
                    && let Some(overlay) = self.constraint.static_dynamic_overlay.as_ref()
                    && !overlay.non_parent_only_parser_states.is_empty()
                {
                    let Some(top) = factor.single_exclusive_top_value() else {
                        break;
                    };
                    if !overlay
                        .non_parent_only_parser_states
                        .get(top as usize)
                        .copied()
                        .unwrap_or(false)
                    {
                        break;
                    }
                }
                let next = if use_fast_factor {
                    let Some((next, forced, after)) =
                        lookahead_reduction_factor_row_subset(&self.constraint.table, &factor)
                    else {
                        break;
                    };
                    chain_forced = Some(match chain_forced.take() {
                        None => forced,
                        Some(existing) => {
                            let not_forced = existing.difference(&forced);
                            existing.difference(&not_forced)
                        }
                    });
                    final_after = Some(after);
                    next
                } else {
                    let Some((next, newly_blocked)) =
                        lookahead_reduction_factor(&self.constraint.table, &factor)
                    else {
                        break;
                    };
                    for terminal in newly_blocked {
                        if !blocked.contains(&terminal) {
                            blocked.push(terminal);
                        }
                    }
                    next
                };
                factor = next;
                deepest = Some(factor.clone());
            }
            let Some(deepest) = deepest else {
                continue;
            };
            if use_fast_factor {
                blocked.clear();
                let Some(chain_forced) = chain_forced.as_ref() else {
                    continue;
                };
                let Some(final_after) = final_after.as_ref() else {
                    continue;
                };
                for bit in final_after.difference(chain_forced).iter_ones() {
                    if bit >= self.constraint.table.num_terminals as usize {
                        fast_guard_representable = false;
                        break;
                    }
                    blocked.push(bit as TerminalID);
                }
                if !fast_guard_representable {
                    continue;
                }
            }
            let guarded = deepest.apply(|acc: &TerminalsDisallowed| {
                let mut guarded = acc.clone();
                for &terminal in &blocked {
                    guarded = guarded.with_insert(tokenizer_state, terminal);
                }
                guarded
            });
            additions.push((tokenizer_state, guarded));
        }
        if additions.is_empty() {
            return None;
        }

        let mut shadow = self.clone();
        for (tokenizer_state, factored) in additions {
            shadow
                .state
                .insert_flat_alternative(tokenizer_state, factored);
        }
        Some(shadow)
    }

    pub fn fill_mask(&self, buf: &mut [u32]) {
        let required = self.constraint.mask_len();
        assert!(buf.len() >= required, "mask buffer is smaller than constraint mask");
        let (mask, tail) = buf.split_at_mut(required);
        tail.fill(0);
        let authoritative_segmented = self
            .constraint
            .static_dynamic_overlay
            .as_ref()
            .is_some_and(|overlay| {
                overlay.segmented_mask_authoritative
                    && (!overlay.segmented_parser_components.is_empty()
                        || overlay.segmented_static_baseline)
            });
        if authoritative_segmented {
            if self.try_fill_mask_segmented_single_paths(mask) {
                self.update_control_special_token_mask(mask);
                self.clear_late_grammar_placeholder_mask(mask);
                return;
            }
            // Segmented projection is the common authoritative A/B path. For
            // an exceptional GSS shape it cannot represent, retain exactness
            // by evaluating the unified composed table dynamically. This is a
            // fallback only: supported states continue to reuse every retained
            // component backend independently.
            self.fill_mask_dynamic(mask);
            self.clear_late_grammar_placeholder_mask(mask);
            return;
        }
        if self.constraint.uses_dynamic_runtime() {
            if !self.try_fill_mask_from_cache(mask) {
                self.fill_mask_dynamic(mask);
                self.store_mask_cache_reuse_dense(mask);
            }
            self.clear_late_grammar_placeholder_mask(mask);
            return;
        }
        if std::env::var_os("GLRMASK_EXPERIMENT_SEGMENTED_PARSER_MASK").is_some()
            && self
                .constraint
                .static_dynamic_overlay
                .as_ref()
                .is_some_and(|overlay| !overlay.segmented_parser_components.is_empty())
        {
            if self.try_fill_mask_segmented_single_paths(mask) {
                self.update_control_special_token_mask(mask);
                if std::env::var_os("GLRMASK_VALIDATE_SEGMENTED_PARSER_MASK").is_some() {
                    let mut reference = vec![0u32; mask.len()];
                    self.fill_mask_uncached(&mut reference);
                    self.update_control_special_token_mask(&mut reference);
                    if reference != mask {
                        let reference_only = (0..mask.len() * 32)
                            .filter(|&token| {
                                let word = token / 32;
                                let bit = token % 32;
                                ((reference[word] >> bit) & 1) != 0
                                    && ((mask[word] >> bit) & 1) == 0
                            })
                            .take(32)
                            .collect::<Vec<_>>();
                        let segmented_only = (0..mask.len() * 32)
                            .filter(|&token| {
                                let word = token / 32;
                                let bit = token % 32;
                                ((reference[word] >> bit) & 1) == 0
                                    && ((mask[word] >> bit) & 1) != 0
                            })
                            .take(32)
                            .collect::<Vec<_>>();
                        panic!(
                            "segmented component parser mask differs from flattened reference; reference_only={reference_only:?} segmented_only={segmented_only:?}"
                        );
                    }
                }
                self.store_mask_cache_reuse_dense(mask);
                self.clear_late_grammar_placeholder_mask(mask);
                return;
            }
        }
        let cache_hit = self.try_fill_mask_from_cache(mask);
        if !cache_hit {
            let factor_profile = std::env::var_os("GLRMASK_PROFILE_LOOKAHEAD_FACTOR").is_some();
            let factor_started = factor_profile.then(Instant::now);
            if let Some(shadow) = self.lookahead_factored_mask_shadow() {
                let factor_ns = factor_started.map_or(0, elapsed_ns);
                let fill_started = factor_profile.then(Instant::now);
                shadow.fill_mask_uncached(mask);
                if !shadow.or_two_dwa_boundary_parser_mask(mask) {
                    mask.fill(0);
                }
                if let Some(fill_started) = fill_started {
                    eprintln!(
                        "[glrmask/profile][lookahead_factor] built_ns={} fill_ns={} original_branches={} shadow_branches={}",
                        factor_ns,
                        elapsed_ns(fill_started),
                        self.state.len(),
                        shadow.state.len(),
                    );
                }
                self.store_mask_cache_reuse_dense(mask);
            } else {
                if let Some(factor_started) = factor_started {
                    eprintln!(
                        "[glrmask/profile][lookahead_factor] built_none_ns={} branches={}",
                        elapsed_ns(factor_started),
                        self.state.len(),
                    );
                }
                self.fill_mask_uncached(mask);
                if !self.or_two_dwa_boundary_parser_mask(mask) {
                    mask.fill(0);
                }
            }
            self.update_control_special_token_mask(mask);
            if !self.constraint.table.control_terminals.is_empty() {
                self.store_mask_cache_reuse_dense(mask);
            }
        }
        self.add_admissible_scoped_ignore_tokens(mask);
        if std::env::var_os("GLRMASK_EXPERIMENT_STATIC_DYNAMIC_OVERLAY").is_some() {
            super::dynamic_mask::or_mask_dynamic_additions(self, mask);
        }
        self.clear_late_grammar_placeholder_mask(mask);
        assert_dynamic_mask_equivalence(self, mask);
    }

    pub(crate) fn fill_mask_timed_ns(&self, buf: &mut [u32]) -> u64 {
        let start = Instant::now();
        self.fill_mask(buf);
        start.elapsed().as_nanos() as u64
    }

    pub(crate) fn fill_mask_profiled(&self, buf: &mut [u32]) -> MaskProfile {
        let required = self.constraint.mask_len();
        assert!(buf.len() >= required, "mask buffer is smaller than constraint mask");
        let (buf, tail) = buf.split_at_mut(required);
        tail.fill(0);
        let total_start = Instant::now();
        if self.try_fill_mask_from_cache(buf) {
            self.clear_late_grammar_placeholder_mask(buf);
            return MaskProfile {
                total_ns: elapsed_ns(total_start),
                cache_hit: 1,
                ..MaskProfile::default()
            };
        }
        if self.constraint.uses_dynamic_runtime() {
            self.fill_mask_dynamic(buf);
            self.clear_late_grammar_placeholder_mask(buf);
            self.store_mask_cache_reuse_dense(buf);
            return MaskProfile {
                total_ns: elapsed_ns(total_start),
                ..MaskProfile::default()
            };
        }

        let profile = self
            .fill_mask_uncached_maybe_profile(buf, true)
            .unwrap_or_else(|| MaskProfile {
                total_ns: elapsed_ns(total_start),
                ..MaskProfile::default()
            });
        self.update_control_special_token_mask(buf);
        self.clear_late_grammar_placeholder_mask(buf);
        if !self.constraint.table.control_terminals.is_empty() {
            self.store_mask_cache_reuse_dense(buf);
        }
        profile
    }

}
