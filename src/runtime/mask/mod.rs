pub(crate) mod profile;
pub(crate) mod queue;
mod static_dwa_walk;
use static_dwa_walk::{StackWalkEvent, walk_single_stack};
#[cfg(test)]
use crate::automata::lexer::Lexer;

use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::grammar::flat::TerminalID;
use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_positive_label};
use crate::compiler::glr::parser::ParserGSS;
use crate::ds::leveled_gss::{
    IndexedLeveledGss, IndexedLeveledGssNode, IndexedLowerIdentity, LeveledGSS, Merge,
};
use crate::ds::weight::Weight;
use crate::runtime::artifact::IndexedDagDenseMask;
use crate::runtime::constraint::{
    Constraint, DenseToBufProfileStats, RuntimeTokenSetRef, RuntimeWeightRef,
};
use crate::runtime::state::{
    CommitBuffers, ConstraintState, MaskCacheData, MaskScratch, ParserStateMap,
};
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

use super::artifact::{WeightTokenMaskCache, WeightTokenMaskEntry};
type DenseMaskGSS = LeveledGSS<u32, DenseMaskAcc>;

/// Mask-local exact byte-transition reuse for recursive composition. Full
/// structural equality includes scoped lexer keys, parser stacks, and delayed
/// exclusions. Hashes only select buckets; they never prove equivalence.
/// Once the state budget is exhausted, new frontiers keep using the original
/// exact radix-edge evaluator. There is no compilation or persistent-cache cost.
#[cfg(test)]
struct RecursiveMaskTransitions {
    states: Vec<ParserStateMap>,
    rows: Vec<[u32; 256]>,
    buckets: FxHashMap<u64, SmallVec<[u32; 2]>>,
    max_states: usize,
    byte_representatives: [u8; 256],
}

#[cfg(test)]
#[derive(Clone)]
enum RecursiveMaskFrontier {
    Cached(u32),
    Uncached(ParserStateMap),
}

/// An exact alphabet congruence over every intact leaf lexer. Equal columns
/// reach identical states (and hence identical matches/futures) from *all*
/// lexer states, including resets and exclusion continuations. Thus zero-width
/// CALL/RETURN and ignored-terminal routes also cannot distinguish the bytes.
/// Bound the proof work; large, epsilon, or virtual lexers simply keep the
/// identity alphabet. This is mask-local work, not constraint compilation.
#[cfg(test)]
fn recursive_mask_byte_representatives(constraint: &Constraint) -> [u8; 256] {
    let identity = std::array::from_fn(|byte| byte as u8);
    let Ok(Some(layout)) = constraint.recursive_parser_layout() else { return identity; };
    const MAX_LEXER_STATES: usize = 256;
    if layout.leaves.is_empty() || layout.total_tokenizer_states as usize > MAX_LEXER_STATES {
        return identity;
    }
    let mut leaves = Vec::with_capacity(layout.leaves.len());
    let mut state_count = 0usize;
    for index in 0..layout.leaves.len() {
        let Some(leaf) = constraint.recursive_leaf_constraint(index) else { return identity; };
        if leaf.tokenizer.has_any_virtual_runtime() || leaf.tokenizer_has_epsilon_transitions {
            return identity;
        }
        state_count += leaf.tokenizer.num_states() as usize;
        if state_count > MAX_LEXER_STATES { return identity; }
        leaves.push(leaf);
    }
    let mut rows = Vec::with_capacity(state_count);
    for leaf in leaves {
        for state in 0..leaf.tokenizer.num_states() {
            rows.push(std::array::from_fn::<u32, 256, _>(|byte|
                leaf.tokenizer_fast_transitions.transition(&leaf.tokenizer, state, byte as u8)));
        }
    }
    use std::hash::{Hash, Hasher};
    let mut buckets = FxHashMap::<u64, SmallVec<[u8; 2]>>::default();
    let mut result = identity;
    for byte in 0..256 {
        let mut hash = rustc_hash::FxHasher::default();
        for row in &rows { row[byte].hash(&mut hash); }
        let bucket = buckets.entry(hash.finish()).or_default();
        // Compare complete columns after hashing: collisions cannot merge
        // inequivalent bytes. One matrix allocation avoids 256 column Vecs.
        if let Some(&representative) = bucket.iter().find(|&&representative|
            rows.iter().all(|row| row[byte] == row[representative as usize]))
        {
            result[byte] = representative;
        } else {
            bucket.push(byte as u8);
        }
    }
    result
}

#[cfg(test)]
impl RecursiveMaskTransitions {
    const UNKNOWN: u32 = u32::MAX;
    const DEAD: u32 = u32::MAX - 1;

    fn new(max_states: usize, byte_representatives: [u8; 256]) -> Self {
        Self {
            states: Vec::new(), rows: Vec::new(), buckets: FxHashMap::default(),
            max_states, byte_representatives,
        }
    }

    fn intern(&mut self, state: &ParserStateMap) -> Option<u32> {
        use std::hash::{Hash, Hasher};
        let mut hash = rustc_hash::FxHasher::default();
        for (lexer, gss) in &state.entries {
            lexer.hash(&mut hash);
            gss.max_depth().hash(&mut hash);
        }
        let hash = hash.finish();
        if let Some(ids) = self.buckets.get(&hash) {
            for &id in ids {
                if self.states[id as usize] == *state {
                    return Some(id);
                }
            }
        }
        if self.states.len() >= self.max_states {
            return None;
        }
        let id = self.states.len() as u32;
        self.states.push(state.clone());
        self.rows.push([Self::UNKNOWN; 256]);
        self.buckets.entry(hash).or_default().push(id);
        Some(id)
    }

    fn advance(
        &mut self,
        constraint: &Constraint,
        parent: &RecursiveMaskFrontier,
        buffers: &mut CommitBuffers,
        bytes: &[u8],
    ) -> Option<RecursiveMaskFrontier> {
        let mut id = match parent {
            RecursiveMaskFrontier::Cached(id) => *id,
            RecursiveMaskFrontier::Uncached(state) => {
                return crate::runtime::commit::advance_bytes_from_state_exact(
                    constraint, state, buffers, bytes,
                ).map(RecursiveMaskFrontier::Uncached);
            }
        };
        for (offset, &byte) in bytes.iter().enumerate() {
            let column = self.byte_representatives[byte as usize] as usize;
            let cached = self.rows[id as usize][column];
            if cached == Self::DEAD {
                return None;
            }
            if cached != Self::UNKNOWN {
                id = cached;
                continue;
            }
            let next = crate::runtime::commit::advance_bytes_from_state_exact(
                constraint, &self.states[id as usize], buffers, std::slice::from_ref(&byte),
            );
            let Some(next) = next else {
                self.rows[id as usize][column] = Self::DEAD;
                return None;
            };
            let Some(next_id) = self.intern(&next) else {
                // Budget exhaustion changes only caching, not admission or
                // traversal coverage. Finish this edge exactly, without a
                // second vocabulary walk or loss of any live alternative.
                return crate::runtime::commit::advance_bytes_from_state_exact(
                    constraint, &next, buffers, &bytes[offset + 1..],
                ).map(RecursiveMaskFrontier::Uncached);
            };
            self.rows[id as usize][column] = next_id;
            id = next_id;
        }
        Some(RecursiveMaskFrontier::Cached(id))
    }
}

const DELTA_SEED_MIN_SAVINGS: u64 = 2048;
const MASK_SINGLE_PATH_DIRECT_MAX_DEPTH: u32 = 64;
const MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY: usize = 64;
const MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS: usize = 128;
// Many independent lexer states with one parser stack are cheap to replay.
// Expanding many alternatives of ONE GSS is different: it duplicates shared
// parser suffixes and dense mask operations. Keep small local ambiguity direct,
// but let the exact indexed-DAG evaluator retain sharing for wider GSSes.
// This bounds speculation, not the language: an incomplete enumeration returns
// false before touching the output, and the next evaluator visits the full GSS.
const MASK_SINGLE_PATH_DIRECT_MAX_PATHS_PER_GSS: usize = 16;
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

/// Test-only switch for the segmented-mask stage trace. Enabled by
/// `GLRMASK_DEBUG_MASK_STAGES`; compiled only under `cfg(test)`.
#[cfg(test)]
fn segmented_mask_stage_trace_enabled() -> bool {
    crate::env::var_os("GLRMASK_DEBUG_MASK_STAGES").is_some()
}

#[cfg(test)]
fn segmented_mask_bits(buf: &[u32]) -> Vec<u32> {
    let mut bits = Vec::new();
    for (word_index, &word) in buf.iter().enumerate() {
        let mut word = word;
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            bits.push((word_index * 32 + bit) as u32);
            word &= word - 1;
        }
    }
    bits
}

#[cfg(test)]
fn segmented_mask_stage_trace(stage: &str, buf: &[u32]) {
    if !segmented_mask_stage_trace_enabled() {
        return;
    }
    let bits = segmented_mask_bits(buf);
    eprintln!(
        "[mask-stages] {stage} token8={} token9={} bits={bits:?}",
        bits.contains(&8),
        bits.contains(&9),
    );
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

/// Same dense/range intersection for ordinary and boundary single-stack masks.
/// The lookup closure specializes away when no precomputed dense row is present.
#[inline]
fn intersect_static_dense_with_weight<'w, 'm>(
    dense: &mut Vec<u64>,
    aux: &mut Vec<u64>,
    internal_tsid: u32,
    weight: RuntimeWeightRef<'w>,
    dense_mask: impl FnOnce(RuntimeTokenSetRef<'w>) -> Option<&'m [u64]>,
) -> bool {
    if weight.is_full() {
        return dense.iter().any(|&word| word != 0);
    }

    let Some(token_set) = weight.token_set_for_tsid(internal_tsid) else {
        dense.fill(0);
        return false;
    };
    if let Some(mask) = dense_mask(token_set) {
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

pub(crate) fn indexed_dag_mask_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::env::var("GLRMASK_ENABLE_INDEXED_DAG_MASK")
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
    })
}

fn indexed_dag_mask_profile_enabled() -> bool {
    crate::env::var_os("GLRMASK_PROFILE_INDEXED_DAG_MASK").is_some()
}

fn dynamic_mask_equivalence_assert_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::env::var("GLRMASK_ASSERT_DYNAMIC_MASK_EQUIVALENCE")
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

fn assert_dynamic_mask_equivalence(state: &ConstraintState, static_mask: &[u32]) {
    if !dynamic_mask_equivalence_assert_enabled() {
        return;
    }

    let mut dynamic_mask = vec![0u32; state.constraint.body_mask_len()];
    crate::compiler::boundary_transfer::permit_strict_static_dynamic(|| {
        state.fill_mask_dynamic(&mut dynamic_mask)
    });
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

/// The weak reference keeps the allocation identity reserved even if a
/// temporary materialized weight is dropped during this mask evaluation.
enum RuntimeTokenSetKey {
    Materialized(std::sync::Weak<RangeSetBlaze<u32>>),
    PackedDwa(u32),
    PackedPool(u32),
}

impl RuntimeTokenSetKey {
    #[inline]
    fn from_ref(token_set: RuntimeTokenSetRef<'_>) -> Self {
        match token_set {
            RuntimeTokenSetRef::Materialized(tokens) => Self::Materialized(Arc::downgrade(tokens)),
            RuntimeTokenSetRef::PackedDwa(tokens) => Self::PackedDwa(tokens.id()),
            RuntimeTokenSetRef::PackedPool(tokens) => Self::PackedPool(tokens.id()),
        }
    }

    #[inline]
    fn matches(&self, token_set: RuntimeTokenSetRef<'_>) -> bool {
        match (self, token_set) {
            (Self::Materialized(cached), RuntimeTokenSetRef::Materialized(tokens)) =>
                cached.as_ptr() == Arc::as_ptr(tokens),
            (Self::PackedDwa(cached), RuntimeTokenSetRef::PackedDwa(tokens)) =>
                *cached == tokens.id(),
            (Self::PackedPool(cached), RuntimeTokenSetRef::PackedPool(tokens)) =>
                *cached == tokens.id(),
            _ => false,
        }
    }
}

type DenseTokenSetIntersectionSmallCache =
    SmallVec<[(Arc<[u64]>, RuntimeTokenSetKey, Option<Arc<[u64]>>); 8]>;


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

    /// Intersect a dense row with any runtime token-set representation.
    /// Preserve the input allocation when the intersection is unchanged.
    fn intersect_dense_with_runtime_token_set(
        dense: &Arc<[u64]>,
        token_set: RuntimeTokenSetRef<'_>,
        precomputed: &WeightTokenMaskCache,
    ) -> Option<Arc<[u64]>> {
        if dense.is_empty() || token_set.is_empty() {
            return None;
        }
        if let Some(key) = token_set.materialized_key()
            && let Some(mask) = precomputed.get(&key).and_then(WeightTokenMaskEntry::dense)
        {
            let mut any = false;
            let mut changed: Option<Vec<u64>> = None;
            for (index, &input) in dense.iter().enumerate() {
                let word = input & mask.get(index).copied().unwrap_or(0);
                any |= word != 0;
                if let Some(out) = changed.as_mut() {
                    out.push(word);
                } else if word != input {
                    let mut out = Vec::with_capacity(dense.len());
                    out.extend_from_slice(&dense[..index]);
                    out.push(word);
                    changed = Some(out);
                }
            }
            return any.then(|| changed.map_or_else(|| Arc::clone(dense), Arc::from));
        }

        let mut out = vec![0u64; dense.len()];
        let mut any = false;
        Self::for_each_runtime_token_range_word(token_set, dense.len(), |word_idx, token_mask| {
            let word = dense[word_idx] & token_mask;
            out[word_idx] |= word;
            any |= word != 0;
        });
        any.then(|| {
            if out.as_slice() == dense.as_ref() {
                Arc::clone(dense)
            } else {
                out.into()
            }
        })
    }

    fn or_dense_and_runtime_token_set_into(
        dense: &[u64],
        token_set: RuntimeTokenSetRef<'_>,
        precomputed: &WeightTokenMaskCache,
        merged: &mut [u64],
    ) {
        if let Some(key) = token_set.materialized_key()
            && let Some(mask) = precomputed.get(&key).and_then(WeightTokenMaskEntry::dense) {
                let n = dense.len().min(mask.len()).min(merged.len());
                for i in 0..n {
                    merged[i] |= dense[i] & mask[i];
                }
                return;
            }
        let word_limit = dense.len().min(merged.len());
        Self::for_each_runtime_token_range_word(token_set, word_limit, |word_idx, token_mask| {
            merged[word_idx] |= dense[word_idx] & token_mask;
        });
    }

    fn intersect_with_runtime_weight_reuse(
        &self,
        weight: RuntimeWeightRef<'_>,
        precomputed: &WeightTokenMaskCache,
    ) -> Option<Self> {
        self.intersect_with_runtime_weight_by(weight, |dense, token_set| {
            Self::intersect_dense_with_runtime_token_set(dense, token_set, precomputed)
        })
    }

    fn intersect_with_runtime_weight_by<'a>(
        &self,
        weight: RuntimeWeightRef<'a>,
        mut intersect: impl FnMut(&Arc<[u64]>, RuntimeTokenSetRef<'a>) -> Option<Arc<[u64]>>,
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
            if let Some(intersection) = intersect(dense, token_set) {
                entries.push((*tsid, intersection));
            }
        }
        (!entries.is_empty()).then_some(Self(entries))
    }

    fn intersect_with_runtime_weight_small_cached(
        &self,
        weight: RuntimeWeightRef<'_>,
        precomputed: &WeightTokenMaskCache,
        cache: &mut DenseTokenSetIntersectionSmallCache,
    ) -> Option<Self> {
        self.intersect_with_runtime_weight_by(weight, |dense, token_set| {
            Self::intersect_dense_with_runtime_token_set_cached(dense, token_set, precomputed, cache)
        })
    }

    fn intersect_dense_with_runtime_token_set_cached(
        dense: &Arc<[u64]>,
        token_set: RuntimeTokenSetRef<'_>,
        precomputed: &WeightTokenMaskCache,
        cache: &mut DenseTokenSetIntersectionSmallCache,
    ) -> Option<Arc<[u64]>> {
        if let Some((_, _, result)) = cache.iter().find(|(cached_dense, cached_token_set, _)| {
            Arc::ptr_eq(cached_dense, dense) && cached_token_set.matches(token_set)
        }) {
            return result.clone();
        }
        let result = Self::intersect_dense_with_runtime_token_set(dense, token_set, precomputed);
        if cache.len() < cache.inline_size() {
            cache.push((Arc::clone(dense), RuntimeTokenSetKey::from_ref(token_set), result.clone()));
        }
        result
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

}

// ---------------------------------------------------------------------------
// Exact cap-free GSS x deterministic boundary-DWA evaluator (static shards).
//
// Production static boundary shards evaluate a determinized parser DWA over
// every concrete parser stack in the current GSS. Enumerating stacks with
// `for_each_stack_top_first_bounded(128, ...)` and declining past the cap is
// unacceptable for a claimed supported static shard: callers interpret
// `false` as "run the exact dynamic fallback". This evaluator computes the
// exact same union the enumeration denotes — union over paths of
// (DWA acceptance(path) intersect eligibility(path)) — by dynamic programming
// over the indexed GSS DAG x DWA product with memoized shared tails. No
// path-count cap affects correctness.
//
// Denotation (top-first read, prefix finals at every readable prefix,
// DEFAULT-label fallback, per-path eligibility correlation):
// - `lower_eval(q, L)` is the union over suffixes in `[[L]]` of the DWA
//   acceptance read from `q`, including the final at the empty prefix and at
//   every consumed prefix. Memoized on `(q, L)`; a shared tail is evaluated
//   once per DWA state no matter how many top prefixes reach it.
// - `eval_upper(q, U)` groups that union by accumulator identity (the
//   interface/branch-empty node id whose accumulator correlates with the
//   path), so filtering afterwards preserves the per-path correlation
//   `union_p (accept(p) intersect eligible(p))` rather than the unsound
//   `union accept(p) intersect union eligible(p)`.
// - Intersection distributes over union, so stamping a shared suffix result
//   with each incoming edge weight is exact.
// - The top-stack start-component/empty-stack filter applies only to the
//   first label (or the empty stack); deeper recursion is filter-free.
#[cfg(test)]
type BoundaryDagGroupMap = FxHashMap<u32, Weight>;

#[cfg(test)]
struct BoundaryWeightDagEvaluator<'a> {
    dwa: &'a crate::automata::weighted_u32::dwa::DWA,
    dag: &'a IndexedLeveledGss<u32, TerminalsDisallowed>,
    ops: crate::ds::weight::ScopedWeightOpCache,
    upper_memo: FxHashMap<(u32, u32), BoundaryDagGroupMap>,
    lower_memo: FxHashMap<(u32, u32), Weight>,
    groups_memo: FxHashMap<u32, FxHashSet<u32>>,
    nonempty_memo: FxHashMap<u32, bool>,
    dwa_steps: u64,
    weight_unions: u64,
    weight_intersections: u64,
}

#[cfg(test)]
impl<'a> BoundaryWeightDagEvaluator<'a> {
    fn new(
        dwa: &'a crate::automata::weighted_u32::dwa::DWA,
        dag: &'a IndexedLeveledGss<u32, TerminalsDisallowed>,
    ) -> Self {
        Self {
            dwa,
            dag,
            ops: crate::ds::weight::ScopedWeightOpCache::default(),
            upper_memo: FxHashMap::default(),
            lower_memo: FxHashMap::default(),
            groups_memo: FxHashMap::default(),
            nonempty_memo: FxHashMap::default(),
            dwa_steps: 0,
            weight_unions: 0,
            weight_intersections: 0,
        }
    }

    fn union_into(&mut self, out: &mut Weight, incoming: &Weight) {
        if incoming.is_empty() {
            return;
        }
        self.weight_unions += 1;
        let merged = self.ops.union(out, incoming);
        *out = merged;
    }

    fn intersect(&mut self, a: &Weight, b: &Weight) -> Weight {
        self.weight_intersections += 1;
        self.ops.intersection(a, b)
    }

    fn dwa_edge(&mut self, state: u32, parser_state: u32) -> Option<(u32, Weight)> {
        self.dwa_steps += 1;
        let st = self.dwa.states().get(state as usize)?;
        let (target, weight) = st
            .transitions
            .get(&encode_positive_label(parser_state))
            .or_else(|| st.transitions.get(&DEFAULT_LABEL))?;
        Some((*target, weight.clone()))
    }

    fn dwa_final(&self, state: u32) -> Weight {
        self.dwa
            .states()
            .get(state as usize)
            .and_then(|st| st.final_weight.clone())
            .unwrap_or_else(Weight::empty)
    }

    /// Structural non-emptiness of a lower node's stack language.
    fn lower_nonempty(&mut self, node: u32) -> bool {
        if let Some(&cached) = self.nonempty_memo.get(&node) {
            return cached;
        }
        let dag = self.dag;
        let result = match &dag.nodes[node as usize] {
            IndexedLeveledGssNode::LowerGeneral { empty, children, .. } => {
                *empty || children.iter().any(|(_, child)| self.lower_nonempty(*child))
            }
            // A segment denotes its fixed prefix ++ tail, so the language is
            // nonempty iff the tail is (degenerate empty-values segments
            // denote exactly the tail).
            IndexedLeveledGssNode::LowerSegment { next, .. } => self.lower_nonempty(*next),
            IndexedLeveledGssNode::UpperBranch { .. } | IndexedLeveledGssNode::Interface { .. } => {
                debug_assert!(false, "lower_nonempty reached upper node");
                false
            }
        };
        self.nonempty_memo.insert(node, result);
        result
    }

    /// Accumulator-group identities with a nonempty path language below an
    /// upper node (DWA-independent). Groups are keyed by the
    /// interface/branch-empty node id carrying the correlated accumulator.
    fn upper_groups(&mut self, node: u32) -> FxHashSet<u32> {
        if let Some(cached) = self.groups_memo.get(&node) {
            return cached.clone();
        }
        let dag = self.dag;
        let result = match &dag.nodes[node as usize] {
            IndexedLeveledGssNode::Interface { lower, .. } => {
                let mut set = FxHashSet::default();
                if self.lower_nonempty(*lower) {
                    set.insert(node);
                }
                set
            }
            IndexedLeveledGssNode::UpperBranch { empty, children } => {
                let mut set = FxHashSet::default();
                if empty.is_some() {
                    set.insert(node);
                }
                for (_, child) in children {
                    set.extend(self.upper_groups(*child));
                }
                set
            }
            IndexedLeveledGssNode::LowerGeneral { .. }
            | IndexedLeveledGssNode::LowerSegment { .. } => {
                debug_assert!(false, "upper_groups reached lower node");
                FxHashSet::default()
            }
        };
        self.groups_memo.insert(node, result.clone());
        result
    }

    /// Consume one segment's fixed values top-first, then its tail. The caller
    /// guarantees the tail language is nonempty. Intermediate DWA finals are
    /// accumulated exactly as the per-path walk would.
    fn step_segment(&mut self, dwa_state: u32, values: &[u32], next: u32) -> Weight {
        let mut out = self.dwa_final(dwa_state);
        let mut cumulative = Weight::all();
        let mut state = dwa_state;
        let mut live = true;
        for value in values.iter().rev() {
            let Some((target, edge_weight)) = self.dwa_edge(state, *value) else {
                live = false;
                break;
            };
            cumulative = self.intersect(&cumulative, &edge_weight);
            if cumulative.is_empty() {
                live = false;
                break;
            }
            state = target;
            let final_here = self.dwa_final(state);
            let stamped = self.intersect(&cumulative, &final_here);
            self.union_into(&mut out, &stamped);
        }
        if live {
            let tail = self.lower_eval(state, next);
            // `tail` repeats `final(state)`, already added above; union is
            // idempotent so stamping the whole tail stays exact.
            let stamped = self.intersect(&cumulative, &tail);
            self.union_into(&mut out, &stamped);
        }
        out
    }

    fn lower_eval(&mut self, dwa_state: u32, node: u32) -> Weight {
        if let Some(cached) = self.lower_memo.get(&(dwa_state, node)) {
            return cached.clone();
        }
        let dag = self.dag;
        let result = match dag.nodes[node as usize].clone() {
            IndexedLeveledGssNode::LowerGeneral { empty: _, children, .. } => {
                // Empty-prefix acceptance belongs to every stack in a
                // nonempty language, not just the empty stack.
                if !self.lower_nonempty(node) {
                    return Weight::empty();
                }
                let mut out = self.dwa_final(dwa_state);
                for (value, child) in children {
                    let Some((target, edge_weight)) = self.dwa_edge(dwa_state, value) else {
                        continue;
                    };
                    let child_result = self.lower_eval(target, child);
                    if child_result.is_empty() {
                        continue;
                    }
                    let stamped = self.intersect(&edge_weight, &child_result);
                    self.union_into(&mut out, &stamped);
                }
                out
            }
            IndexedLeveledGssNode::LowerSegment { values, next, .. } => {
                if self.lower_nonempty(next) {
                    self.step_segment(dwa_state, &values, next)
                } else {
                    Weight::empty()
                }
            }
            IndexedLeveledGssNode::UpperBranch { .. } | IndexedLeveledGssNode::Interface { .. } => {
                debug_assert!(false, "lower_eval reached upper node");
                Weight::empty()
            }
        };
        self.lower_memo.insert((dwa_state, node), result.clone());
        result
    }

    fn eval_upper(&mut self, dwa_state: u32, node: u32) -> BoundaryDagGroupMap {
        if let Some(cached) = self.upper_memo.get(&(dwa_state, node)) {
            return cached.clone();
        }
        let dag = self.dag;
        let result = match dag.nodes[node as usize].clone() {
            IndexedLeveledGssNode::Interface { lower, .. } => {
                let weight = self.lower_eval(dwa_state, lower);
                let mut map = BoundaryDagGroupMap::default();
                if !weight.is_empty() {
                    map.insert(node, weight);
                }
                map
            }
            IndexedLeveledGssNode::UpperBranch { children, .. } => {
                let mut map = BoundaryDagGroupMap::default();
                for (value, child) in children {
                    let Some((target, edge_weight)) = self.dwa_edge(dwa_state, value) else {
                        continue;
                    };
                    let child_map = self.eval_upper(target, child);
                    for (group, weight) in child_map {
                        if weight.is_empty() {
                            continue;
                        }
                        let stamped = self.intersect(&edge_weight, &weight);
                        match map.get_mut(&group) {
                            Some(existing) => self.union_into(existing, &stamped),
                            None => {
                                map.insert(group, stamped);
                            }
                        }
                    }
                }
                // Empty-prefix acceptance belongs to every surviving path's
                // group; the per-group add preserves path-specific
                // eligibility correlation (each path's empty-prefix
                // contribution is filtered by its own accumulator later).
                let groups = self.upper_groups(node);
                let final_weight = self.dwa_final(dwa_state);
                if !final_weight.is_empty() {
                    for group in groups {
                        match map.get_mut(&group) {
                            Some(existing) => self.union_into(existing, &final_weight),
                            None => {
                                map.insert(group, final_weight.clone());
                            }
                        }
                    }
                }
                map
            }
            IndexedLeveledGssNode::LowerGeneral { .. }
            | IndexedLeveledGssNode::LowerSegment { .. } => {
                debug_assert!(false, "eval_upper reached lower node");
                BoundaryDagGroupMap::default()
            }
        };
        self.upper_memo.insert((dwa_state, node), result.clone());
        result
    }

    /// First-level top-filtered lower evaluation for an interface-rooted DAG:
    /// only the top label (or the empty stack) is filtered, deeper recursion
    /// is filter-free.
    fn lower_eval_top_filtered(
        &mut self,
        dwa_state: u32,
        node: u32,
        top_live: &dyn Fn(Option<u32>) -> bool,
    ) -> Weight {
        let dag = self.dag;
        match dag.nodes[node as usize].clone() {
            IndexedLeveledGssNode::LowerGeneral { empty, children, .. } => {
                // Empty-prefix acceptance belongs to every surviving stack,
                // not just the empty stack.
                let live_empty = empty && top_live(None);
                let live_language = live_empty
                    || children
                        .iter()
                        .any(|(value, child)| top_live(Some(*value)) && self.lower_nonempty(*child));
                let mut out = if live_language {
                    self.dwa_final(dwa_state)
                } else {
                    Weight::empty()
                };
                for (value, child) in children {
                    if !top_live(Some(value)) {
                        continue;
                    }
                    let Some((target, edge_weight)) = self.dwa_edge(dwa_state, value) else {
                        continue;
                    };
                    let child_result = self.lower_eval(target, child);
                    if child_result.is_empty() {
                        continue;
                    }
                    let stamped = self.intersect(&edge_weight, &child_result);
                    self.union_into(&mut out, &stamped);
                }
                out
            }
            IndexedLeveledGssNode::LowerSegment { values, next, .. } => {
                if values.is_empty() {
                    return self.lower_eval_top_filtered(dwa_state, next, top_live);
                }
                let top = *values.last().expect("nonempty segment has a top value");
                if !top_live(Some(top)) || !self.lower_nonempty(next) {
                    return Weight::empty();
                }
                // The top passed the filter; the rest is filter-free.
                self.step_segment(dwa_state, &values, next)
            }
            IndexedLeveledGssNode::UpperBranch { .. } | IndexedLeveledGssNode::Interface { .. } => {
                debug_assert!(false, "lower_eval_top_filtered reached upper node");
                Weight::empty()
            }
        }
    }

    /// Evaluate the whole DAG from the DWA start state. `top_live` filters
    /// the top stack value (`None` for the empty stack): start-component
    /// ownership or empty-stack acceptance. Returns group-id -> accepted
    /// weight; the caller resolves each group id to its accumulator and
    /// filters eligibility per group.
    fn eval_root(&mut self, top_live: &dyn Fn(Option<u32>) -> bool) -> BoundaryDagGroupMap {
        let dag = self.dag;
        let start = self.dwa.start_state();
        let root = dag.root;
        match dag.nodes[root as usize].clone() {
            IndexedLeveledGssNode::Interface { lower, .. } => {
                let weight = self.lower_eval_top_filtered(start, lower, top_live);
                let mut map = BoundaryDagGroupMap::default();
                if !weight.is_empty() {
                    map.insert(root, weight);
                }
                map
            }
            IndexedLeveledGssNode::UpperBranch { empty, children } => {
                let mut map = BoundaryDagGroupMap::default();
                let mut live_groups = FxHashSet::default();
                if empty.is_some() && top_live(None) {
                    live_groups.insert(root);
                }
                for (value, child) in children {
                    if !top_live(Some(value)) {
                        continue;
                    }
                    // Every language group below this child survives the top
                    // filter, so it owns the empty-prefix acceptance even
                    // when its DWA extensions are all dead.
                    live_groups.extend(self.upper_groups(child));
                    let Some((target, edge_weight)) = self.dwa_edge(start, value) else {
                        continue;
                    };
                    let child_map = self.eval_upper(target, child);
                    for (group, weight) in child_map {
                        if weight.is_empty() {
                            continue;
                        }
                        let stamped = self.intersect(&edge_weight, &weight);
                        match map.get_mut(&group) {
                            Some(existing) => self.union_into(existing, &stamped),
                            None => {
                                map.insert(group, stamped);
                            }
                        }
                    }
                }
                let final_weight = self.dwa_final(start);
                if !final_weight.is_empty() {
                    for group in live_groups {
                        match map.get_mut(&group) {
                            Some(existing) => self.union_into(existing, &final_weight),
                            None => {
                                map.insert(group, final_weight.clone());
                            }
                        }
                    }
                }
                map
            }
            IndexedLeveledGssNode::LowerGeneral { .. }
            | IndexedLeveledGssNode::LowerSegment { .. } => {
                debug_assert!(false, "boundary DAG root is not an upper node");
                BoundaryDagGroupMap::default()
            }
        }
    }

    /// Resolve a group id to the accumulator correlated with its paths.
    fn group_accumulator(&self, group: u32) -> Option<TerminalsDisallowed> {
        match &self.dag.nodes[group as usize] {
            IndexedLeveledGssNode::Interface { accumulator, .. } => Some(accumulator.clone()),
            IndexedLeveledGssNode::UpperBranch { empty, .. } => empty.clone(),
            IndexedLeveledGssNode::LowerGeneral { .. }
            | IndexedLeveledGssNode::LowerSegment { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        exact_component_trigger_accepted_weight, single_path_direct_plan_reuse_dominates,
        single_path_direct_stack_work,
        DenseMaskAcc,
        WeightTokenMaskCache,
        DenseTokenSetIntersectionSmallCache,
        RuntimeTokenSetRef,
        MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY,
        MASK_SINGLE_PATH_DIRECT_MAX_PATHS_PER_GSS,
        MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_STACK_VALUES,
    };
    use crate::automata::lexer::Lexer;
    use crate::compiler::glr::accumulator::TerminalsDisallowed;
    use crate::compiler::glr::parser::ParserGSS;
    use crate::{runtime::Constraint as Constraint, Grammar, Vocab};
    use range_set_blaze::RangeSetBlaze;
    use std::sync::Arc;

    fn precomputed_for(
        token_set: &Arc<RangeSetBlaze<u32>>,
        mask: Arc<[u64]>,
    ) -> WeightTokenMaskCache {
        let mut precomputed = WeightTokenMaskCache::default();
        precomputed.insert(
            Arc::as_ptr(token_set) as usize,
            super::WeightTokenMaskEntry::new(
                token_set,
                Some(mask),
                crate::runtime::artifact::WeightTokenOutput::None,
            ),
        );
        precomputed
    }

    fn mask_contains(mask: &[u32], token: u32) -> bool {
        mask.get(token as usize / 32)
            .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
    }

    #[test]
    fn component_mask_word_copy_matches_exact_membership_dense_sparse_and_specials() {
        // Dense prefixes ending on and between word boundaries; sparse holes;
        // byte aliases; out-of-vocabulary specials; and longer output buffers.
        let vocabularies = [
            (0..2).collect::<Vec<u32>>(),
            (0..32).collect::<Vec<u32>>(),
            (0..35).collect::<Vec<u32>>(),
            vec![0, 3, 31, 32, 65, 96],
        ];
        for ids in vocabularies {
            let vocab = Vocab::from_entries_unchecked(ids.iter().map(|&id| (id, b"a".to_vec())).collect());
            let original = crate::ConstraintSpec::builder(
                Grammar::glrm("glrm 1; start document; extern token SPECIAL; nt document = \"a\" | SPECIAL;"),
                &vocab,
            ).unwrap().bind_token("SPECIAL", [70, 130]).unwrap().build().unwrap().compile().unwrap();
            let loaded = Constraint::load(original.save(), &vocab).unwrap();
            for constraint in [&original, &loaded] {
                let state = constraint.start();
                for seed in [0u32, 0x5555_5555, 0xffff_ffff] {
                    let mut actual = vec![seed; constraint.mask_bit_len().div_ceil(32) + 2];
                    let mut expected = actual.clone();
                    let source: Vec<_> = (0..actual.len() + 1)
                        .map(|i| if seed == 0 { u32::MAX } else { seed.rotate_left(i as u32) })
                        .collect();
                    for (word, (&bits, output)) in source.iter().zip(&mut expected).enumerate() {
                        for bit in 0..32 {
                            if bits & (1u32 << bit) != 0 && state.knows_token_id(word as u32 * 32 + bit) {
                                *output |= 1u32 << bit;
                            }
                        }
                    }
                    state.or_segmented_component_mask(&mut actual, &source);
                    assert_eq!(actual, expected, "ids={ids:?}, seed={seed}");
                }
            }
        }
    }

    #[test]
    fn segmented_static_result_cache_tracks_exact_scoped_state() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"[".to_vec()), (1, b"a".to_vec()), (2, b"]".to_vec()),
            (3, b"!".to_vec()), (4, b"[aa]!".to_vec()), (5, b" ".to_vec()),
        ]);
        let child = Constraint::compile(Grammar::glrm(
            r#"start child; ignore WS; t WS = " "+; t WORD = /[a-z]+/; nt child = WORD;"#
        ), &vocab).unwrap();
        let parent = Constraint::compile(Grammar::glrm(
            r#"start document; extern grammar child; nt document = "[" child "]!";"#
        ), &vocab).unwrap();
        let bound = parent.bind_grammar("child", child).unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());
        let loaded = Constraint::load(bound.save(), &vocab).unwrap();
        for constraint in [&bound, &loaded] {
            let mut state = constraint.start();
            let mut unchanged_cache_reuses = 0;
            for token in [0,1,1,1,5,2,3] {
                let mask = state.mask();
                assert!(mask_contains(&mask, token));
                let mut cached = vec![0; mask.len()];
                assert!(state.try_fill_mask_from_cache(&mut cached));
                assert_eq!(cached, mask);
                let before = state.state.clone();
                assert!(state.commit_token(token));
                if before == state.state {
                    assert!(state.try_fill_mask_from_cache(&mut cached));
                    unchanged_cache_reuses += 1;
                } else {
                    assert!(!state.try_fill_mask_from_cache(&mut cached));
                }
                assert_eq!(state.mask(), state.clone().mask(), "cached/fresh token={token}");
            }
            assert!(unchanged_cache_reuses > 0, "fixture must exercise a repeated-state cache hit");
            assert!(state.is_accepting());
            assert!(!state.commit_token(0));
            assert!(state.mask().iter().all(|&word| word == 0), "failed commit invalidates cache");
            assert_eq!(state.mask(), state.clone().mask());
        }
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"x".to_vec()),
            (1, b"xy".to_vec()),
            (2, b"y".to_vec()),
        ]);
        let mut parent = Constraint::from_glrm_grammar_with_subgrammars_bindings_and_end_tokens(
            "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            &[],
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
    fn recursive_exact_full_walk_ignores_outer_table_and_tokenizer() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"Xa!".to_vec()),
            (4, b"a!".to_vec()),
        ]);
        let child = Constraint::compile(
            Grammar::glrm("glrm 1; start child; nt child = \"a\";"),
            &vocab,
        )
        .unwrap();
        let parent = Constraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar child; nt document = \"X\" child \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = parent
            .bind_grammar_dynamic_boundary("child", child)
            .unwrap();
        let monolithic = Constraint::compile(
            Grammar::glrm("glrm 1; start document; nt document = \"X\" \"a\" \"!\";"),
            &vocab,
        )
        .unwrap();

        let mut poisoned = Constraint::load(bound.save(), &vocab).unwrap();
        poisoned.recursive_parser_layout().unwrap().unwrap();
        let root = poisoned
            .static_dynamic_overlay
            .as_ref()
            .unwrap()
            .segmented_parser_components[0]
            .constraint
            .clone();
        poisoned.tokenizer = root.tokenizer.clone();
        poisoned.tokenizer_fast_transitions = root.tokenizer_fast_transitions.clone();
        poisoned.tokenizer_has_epsilon_transitions = root.tokenizer_has_epsilon_transitions;
        assert!(!poisoned.table.is_present(), "native root must retain no LR table");

        let mut actual = poisoned.start();
        let mut expected = monolithic.start();
        for prefix_token in [None, Some(0), Some(1)] {
            if let Some(token) = prefix_token {
                assert!(actual.commit_token(token));
                assert!(expected.commit_token(token));
            }
            let mut fallback = vec![0u32; poisoned.body_mask_len()];
            actual.fill_recursive_mask_by_exact_full_walk(&mut fallback);
            let expected_mask = expected.mask();
            assert_eq!(fallback, expected_mask);

            let mut dynamic_reference = vec![0u32; poisoned.body_mask_len()];
            actual.fill_mask_dynamic(&mut dynamic_reference);
            assert_eq!(dynamic_reference, expected_mask);

            let mut profiled = vec![0u32; poisoned.body_mask_len()];
            actual.fill_mask_profiled(&mut profiled);
            assert_eq!(profiled, expected_mask);
        }
    }

    /// The former recursive walker deliberately scanned dead descendants.
    /// Keep that independent control flow as a regression oracle for DFS jumps.
    fn recursive_unpruned_reference(state: &super::ConstraintState) -> Vec<u32> {
        let vocab = state.constraint.dynamic_mask_vocab_for_runtime();
        let trie = vocab.trie.as_ref();
        let mut buffers = super::CommitBuffers::default();
        let mut output = vec![0u32; state.constraint.mask_bit_len().div_ceil(32)];
        let mut stack = vec![None; usize::from(trie.full_walk_max_parent_depth()) + 2];
        stack[0] = Some(state.state.clone());
        let mark = |node, output: &mut [u32]| {
            if let Some(canonical) = trie.node(node).token_id {
                for &id in vocab.token_ids(canonical).unwrap() {
                    if !state.constraint.has_special_token_id(id) {
                        super::set_original_mask_bit(output, id);
                    }
                }
            }
        };
        // The root's zero-byte aliases are not model-token byte transitions.
        // Exact special-token paths for those IDs are evaluated below.
        for edge in trie.walk_edges() {
            let depth = edge.parent_depth as usize;
            let next = stack[depth].as_ref().and_then(|parent| {
                crate::runtime::commit::advance_bytes_from_state_exact(
                    &state.constraint, parent, &mut buffers, trie.walk_edge_bytes(edge),
                )
            });
            if next.is_some() {
                mark(edge.child, &mut output);
            }
            stack[depth + 1] = next;
        }
        for special in &state.constraint.special_token_terminals {
            if state.constraint.is_late_grammar_placeholder_terminal(special.terminal_id) { continue; }
            for token in special.token_ids() {
                if recursive_token_domain_reference(state, &mut buffers, token) {
                    super::set_original_mask_bit(&mut output, token);
                }
            }
        }
        output
    }

    fn recursive_token_domain_reference(
        state: &super::ConstraintState,
        buffers: &mut super::CommitBuffers,
        token_id: u32,
    ) -> bool {
        if state.constraint.token_bytes_for_id(token_id).is_some_and(|b| b.is_empty()) {
            // commit_bytes(empty) is intentionally a no-op; it is not a proof
            // of model-token admission. Match the explicit current public API
            // contract without suppressing any nonempty or special-ID probes.
            state.constraint.has_special_token_id(token_id)
                && crate::runtime::commit::advance_special_token_paths(
                    &state.constraint, &state.state, token_id,
                ).is_some_and(|gss| !gss.is_empty())
        } else {
            crate::runtime::commit::token_admissible_from_state_exact(
                &state.constraint, &state.state, buffers, token_id,
            )
        }
    }

    #[test]
    fn recursive_dead_subtree_jumps_match_unpruned_and_pointwise_oracles() {
        // Large dead families share ancestors with live siblings. Duplicate
        // byte spellings, an empty token, and a special-token spelling within
        // a dead byte family exercise independent endpoint/semantic routing.
        let mut entries = vec![
            (0, b"X".to_vec()), (1, b"[".to_vec()), (2, b"a".to_vec()),
            (3, b"]".to_vec()), (4, b"!".to_vec()), (5, b"a]!".to_vec()),
            (6, b"[a]!".to_vec()), (7, b"X[a]!".to_vec()),
            (8, b"X[]!".to_vec()), (9, Vec::new()), (10, b"a".to_vec()),
            (11, b"qdead-special".to_vec()), (12, b"qdead-special".to_vec()),
            (13, b"]!".to_vec()), (14, b"[]!".to_vec()),
        ];
        for i in 0..512u32 {
            entries.push((32 + i, format!("qdead-{i:04}-suffix").into_bytes()));
            entries.push((600 + i, format!("X[ab-{i:04}-suffix").into_bytes()));
        }
        let ids: Vec<u32> = entries.iter().map(|(id, _)| *id).chain([9001]).collect();
        let vocab = Vocab::from_entries_unchecked(entries);
        for nullable in [false, true] {
            let leaf_source = if nullable {
                "glrm 1; start leaf; extern token SPECIAL; nt leaf = \"a\"? | SPECIAL;"
            } else {
                "glrm 1; start leaf; extern token SPECIAL; nt leaf = \"a\" | SPECIAL;"
            };
            let leaf = crate::ConstraintSpec::builder(Grammar::glrm(leaf_source), &vocab)
                .unwrap().bind_token("SPECIAL", [11, 9001]).unwrap()
                .build().unwrap().compile().unwrap();
            let middle_parent = Constraint::compile(Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
            ), &vocab).unwrap();
            let middle = middle_parent.bind_grammar_dynamic_boundary("leaf", leaf).unwrap();
            let outer_parent = Constraint::compile(Grammar::glrm(
                "glrm 1; start outer; extern grammar middle; nt outer = \"X\" middle \"!\";",
            ), &vocab).unwrap();
            let bound = outer_parent.bind_grammar_dynamic_boundary("middle", middle).unwrap();
            let loaded = Constraint::load(bound.save(), &vocab).unwrap();
            for constraint in [&bound, &loaded] {
                assert!(constraint.uses_compact_segmented_parser_runtime());
                let check = |state: &super::ConstraintState| {
                    let mut actual = vec![0u32; constraint.mask_bit_len().div_ceil(32)];
                    state.fill_recursive_mask_by_exact_full_walk(&mut actual);
                    assert_eq!(actual, recursive_unpruned_reference(state), "nullable={nullable}");
                    for budget in [0, 1, 2] {
                        let mut bounded = vec![0u32; constraint.mask_bit_len().div_ceil(32)];
                        state.or_recursive_dynamic_full_walk_exact_with_budget(&mut bounded, budget);
                        assert_eq!(bounded, actual, "cache budget={budget}, nullable={nullable}");
                    }
                    let mut buffers = super::CommitBuffers::default();
                    for &id in &ids {
                        let exact = recursive_token_domain_reference(state, &mut buffers, id);
                        let allowed = actual[id as usize / 32] & (1 << (id % 32)) != 0;
                        assert_eq!(allowed, exact, "token={id}, nullable={nullable}");
                    }
                    assert_eq!(state.mask(), actual);
                };
                for prefix in [b"".as_slice(), b"X", b"X[", b"X[a", b"X[a]", b"X[a]!"] {
                    let mut state = constraint.start();
                    assert!(state.commit_bytes(prefix));
                    check(&state);
                }
                for special in [11, 9001] {
                    let mut state = constraint.start();
                    assert!(state.commit_bytes(b"X["));
                    let mask = state.mask();
                    assert_ne!(mask[special / 32] & (1 << (special % 32)), 0);
                    assert_eq!(mask[0] & (1 << 12), 0, "byte alias has no special route");
                    assert!(state.commit_token(special as u32));
                    check(&state);
                    assert!(state.commit_bytes(b"]!"));
                    check(&state);
                }
                if nullable {
                    let mut state = constraint.start();
                    assert!(state.commit_bytes(b"X[]!"));
                    check(&state);
                }
            }
        }
    }

    #[test]
    fn recursive_transition_cache_matches_radix_on_text_and_ambiguity() {
        let mut words = vec![Vec::<u8>::new()];
        let mut layer = vec![Vec::<u8>::new()];
        for _ in 0..4 {
            let mut next = Vec::new();
            for prefix in layer {
                for &byte in b"abcX[]! " {
                    let mut word = prefix.clone();
                    word.push(byte);
                    next.push(word);
                }
            }
            words.extend(next.iter().cloned());
            layer = next;
        }
        let vocab = Vocab::from_entries_unchecked(words.into_iter().enumerate()
            .map(|(id, bytes)| (id as u32, bytes)).collect::<Vec<_>>());
        for leaf_source in [
            "glrm 1; start leaf; t TEXT = /[a-c ]{1,8}/; nt leaf = TEXT;",
            "glrm 1; start leaf; t WORD = /[a-c]{1,4}/; nt leaf = WORD | WORD WORD;",
            "glrm 1; start leaf; ignore WS; t WS = \" \"+; t WORD = /[a-c]{1,4}/; nt leaf = WORD;",
        ] {
            let leaf = Constraint::compile(Grammar::glrm(leaf_source), &vocab).unwrap();
            let middle = Constraint::compile(Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
            ), &vocab).unwrap().bind_grammar_dynamic_boundary("leaf", leaf).unwrap();
            let bound = Constraint::compile(Grammar::glrm(
                "glrm 1; start outer; extern grammar middle; nt outer = \"X\" middle \"!\";",
            ), &vocab).unwrap().bind_grammar_dynamic_boundary("middle", middle).unwrap();
            let alphabet = super::recursive_mask_byte_representatives(&bound);
            assert_eq!(alphabet[b'a' as usize], alphabet[b'b' as usize]);
            assert_eq!(alphabet[b'b' as usize], alphabet[b'c' as usize]);
            assert_ne!(alphabet[b'a' as usize], alphabet[b']' as usize]);
            for prefix in [b"".as_slice(), b"X", b"X[", b"X[a", b"X[abc", b"X[abc]", b"X[abc]!"] {
                let mut state = bound.start();
                assert!(state.commit_bytes(prefix));
                let expected = recursive_unpruned_reference(&state);
                for budget in [0, 2, 256] {
                    let mut actual = vec![0u32; bound.mask_bit_len().div_ceil(32)];
                    state.or_recursive_dynamic_full_walk_exact_with_budget(&mut actual, budget);
                    assert_eq!(actual, expected, "prefix={prefix:?} budget={budget} leaf={leaf_source}");
                }
            }
        }
    }

    #[test]
    fn precomputed_dense_intersection_reuses_arc_when_unchanged() {
        let dense: Arc<[u64]> = Arc::from([0b1011_u64, 0b0101]);
        let token_set = Arc::new(RangeSetBlaze::from_iter(std::iter::once(0_u32..=127)));
        let precomputed = precomputed_for(&token_set, Arc::from([!0_u64, !0_u64]));

        let mut cache = DenseTokenSetIntersectionSmallCache::new();
        let intersected = DenseMaskAcc::intersect_dense_with_runtime_token_set_cached(
            &dense,
            RuntimeTokenSetRef::Materialized(&token_set),
            &precomputed,
            &mut cache,
        )
        .unwrap();

        assert!(Arc::ptr_eq(&intersected, &dense));
    }

    #[test]
    fn precomputed_dense_intersection_allocates_when_pruned() {
        let dense: Arc<[u64]> = Arc::from([0b1011_u64, 0b0101]);
        let token_set = Arc::new(RangeSetBlaze::from_iter(std::iter::once(0_u32..=127)));
        let precomputed = precomputed_for(&token_set, Arc::from([0b0011_u64, 0b0000]));

        let mut cache = DenseTokenSetIntersectionSmallCache::new();
        let intersected = DenseMaskAcc::intersect_dense_with_runtime_token_set_cached(
            &dense,
            RuntimeTokenSetRef::Materialized(&token_set),
            &precomputed,
            &mut cache,
        )
        .unwrap();

        assert!(!Arc::ptr_eq(&intersected, &dense));
        assert_eq!(&*intersected, &[0b0011_u64, 0b0000]);
    }

    #[test]
    fn small_intersection_cache_reserves_temporary_token_identity() {
        let dense: Arc<[u64]> = Arc::from([0b11_u64]);
        let tokens = Arc::new(RangeSetBlaze::from_iter([0_u32]));
        let mut cache = DenseTokenSetIntersectionSmallCache::new();
        let result = DenseMaskAcc::intersect_dense_with_runtime_token_set_cached(
            &dense,
            RuntimeTokenSetRef::Materialized(&tokens),
            &Default::default(),
            &mut cache,
        ).unwrap();
        assert_eq!(&*result, &[1]);
        assert_eq!(Arc::weak_count(&tokens), 1);
        drop(tokens);
        let other = Arc::new(RangeSetBlaze::from_iter([1_u32]));
        let result = DenseMaskAcc::intersect_dense_with_runtime_token_set_cached(
            &dense,
            RuntimeTokenSetRef::Materialized(&other),
            &Default::default(),
            &mut cache,
        ).unwrap();
        assert_eq!(&*result, &[2]);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn small_intersection_cache_reuses_exact_result() {
        let dense: Arc<[u64]> = Arc::from([0b1011_u64, 0b0101]);
        let token_set = Arc::new(RangeSetBlaze::from_iter(std::iter::once(0_u32..=127)));
        let precomputed = precomputed_for(&token_set, Arc::from([0b0011_u64, 0b0100]));
        let mut cache = DenseTokenSetIntersectionSmallCache::new();

        let first = DenseMaskAcc::intersect_dense_with_runtime_token_set_cached(
            &dense,
            RuntimeTokenSetRef::Materialized(&token_set),
            &precomputed,
            &mut cache,
        )
        .unwrap();
        let second = DenseMaskAcc::intersect_dense_with_runtime_token_set_cached(
            &dense,
            RuntimeTokenSetRef::Materialized(&token_set),
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
                t A = "a";
                t B = "b";
                nt start = A | B;
            "#,
            &Vocab::from_entries_unchecked(
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
        constraint.non_dwa_weights.materialized_mut().possible_matches.clear();
        constraint.possible_matches_complete = false;

        let tokenizer_state = constraint.tokenizer.initial_state();
        let disallowed = TerminalsDisallowed::new().with_insert(tokenizer_state, terminal_a);
        let mut state = constraint.start_dynamic();
        state.state = crate::runtime::state::ParserStateMap::singleton(
            tokenizer_state,
            ParserGSS::from_stacks(&[(vec![0u32], disallowed)]),
        );

        let mut expected = vec![0u32; constraint.body_mask_len()];
        state.fill_mask_dynamic(&mut expected);
        let mut actual = vec![0u32; constraint.body_mask_len()];
        state.fill_mask(&mut actual);
        assert_eq!(actual, expected);

        let loaded = Constraint::load(constraint.save(), &crate::public_api::constraint_vocab(&constraint)).expect("empty-PM constraint should roundtrip");
        assert!(loaded.runtime_possible_match_terminals().next().is_none());
        let mut loaded_state = loaded.start_dynamic();
        let loaded_tokenizer_state = loaded.tokenizer.initial_state();
        let loaded_disallowed =
            TerminalsDisallowed::new().with_insert(loaded_tokenizer_state, terminal_a);
        loaded_state.state = crate::runtime::state::ParserStateMap::singleton(
            loaded_tokenizer_state,
            ParserGSS::from_stacks(&[(vec![0u32], loaded_disallowed)]),
        );
        let mut loaded_expected = vec![0u32; loaded.body_mask_len()];
        loaded_state.fill_mask_dynamic(&mut loaded_expected);
        let mut loaded_actual = vec![0u32; loaded.body_mask_len()];
        loaded_state.fill_mask(&mut loaded_actual);
        assert_eq!(loaded_actual, loaded_expected);
    }

    #[test]
    fn literal_choice_terminal_mask_keeps_multibyte_prefix_tokens() {
        let vocab = Vocab::from_entries_unchecked(vec![
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
                nt assurance_body = fa {
                    start 0;
                    accept 1;
                    edge 0 -> 1: "\"" ("Service provider assertion\"" | "Independent validation of assertion\"");
                };
                nt start = assurance_body;
            "#,
            &vocab,
        )
        .expect("literal-choice terminal should compile");

        let mut state = constraint.start();
        assert!(state.commit_token(0), "opening quote should commit");

        let mut static_mask = vec![0u32; constraint.body_mask_len()];
        state.fill_mask(&mut static_mask);
        let mut dynamic_mask = vec![0u32; constraint.body_mask_len()];
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
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let literal = "a".repeat(MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY * 2);
        let grammar = format!("start start; t A = \"{literal}\"; nt start = A;");
        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab)
            .expect("long finite terminal should compile");
        let mut state = constraint.start();
        let parser_gss = state.state.values().next().unwrap().clone();
        let keys = (0..constraint.tokenizer.num_states())
            .filter(|&key| constraint.internal_tsids_for_state(key).len() == 1)
            .take(MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY + 1).collect::<Vec<_>>();
        assert_eq!(keys.len(), MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY + 1);
        state.state.clear();
        for key in keys { state.state.merge_insert(key, parser_gss.clone()); }
        let mut direct = vec![0u32; constraint.body_mask_len()];
        assert!(state.try_fill_mask_single_path_direct(&mut direct));
        let mut dynamic = vec![0u32; constraint.body_mask_len()];
        state.fill_mask_dynamic(&mut dynamic);
        assert_eq!(direct, dynamic);
        assert!(mask_contains(&direct, 0));
        assert!(!mask_contains(&direct, 1));
    }

    #[test]
    fn direct_mask_declines_wide_shared_gss_without_partial_output() {
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let constraint = Constraint::from_glrm_grammar(
            r#"start start; t A = "a"; nt start = A;"#,
            &vocab,
        )
        .expect("routing-test grammar should compile");
        let mut state = constraint.start();
        let tokenizer_state = *state.state.keys().next().unwrap();
        // Deliberately synthetic parser labels: admission must decline BEFORE
        // looking up any of them or evaluating an incomplete subset of paths.
        // Both the one-pass and the many-lexer two-pass admission paths matter.
        let wide = ParserGSS::from_stacks(
            &(0..=MASK_SINGLE_PATH_DIRECT_MAX_PATHS_PER_GSS)
                .map(|path| (vec![0, 10_000 + path as u32, 7], TerminalsDisallowed::default()))
                .collect::<Vec<_>>(),
        );
        for lexer_branches in [1, MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY] {
            state.state.entries.clear();
            for index in 0..lexer_branches {
                state.state.merge_insert(tokenizer_state + index as u32, wide.clone());
            }
            let mut mask = vec![0x5a5a_5a5a; constraint.body_mask_len()];
            let before = mask.clone();
            assert!(!state.try_fill_mask_single_path_direct(&mut mask));
            assert_eq!(mask, before, "declining admission must not publish a partial mask");
        }
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

        let vocab = Vocab::from_entries_unchecked(
            ["a", "b", "ab", "ba", "aa", "bb"]
                .into_iter()
                .enumerate()
                .map(|(id, bytes)| (id as u32, bytes.as_bytes().to_vec()))
                .collect(),
        );
        let grammars = [
            r#"
                start start;
                t A = "a" | "ab";
                t B = "a" | "ba";
                nt item = A | B;
                nt start = item item? item?;
            "#,
            r#"
                start start;
                t A = "a"+;
                t B = "a"+ "b"?;
                nt start = A A | B B | A B | B A;
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
                    let mut expected = vec![0u32; constraint.body_mask_len()];
                    state.fill_mask_dynamic(&mut expected);
                    if state.has_parser_ambiguity() {
                        ambiguous_states += 1;
                        let mut actual = vec![0u32; constraint.body_mask_len()];
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
                        assert!(advanced
                            .commit_bytes(bytes), "dynamically admitted token must commit");
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
        let vocab = Vocab::from_entries_unchecked(
            ["a", "b", "ab", "ba", "aa", "bb"]
                .into_iter()
                .enumerate()
                .map(|(id, bytes)| (id as u32, bytes.as_bytes().to_vec()))
                .collect(),
        );
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A = "a"+;
                t B = "a"+ "b"?;
                nt item = A | B;
                nt start = item item? item? item?;
            "#,
            &vocab,
        )
        .expect("persistent indexed-DAG parity grammar should compile");
        let mut state = constraint.start();
        let sequence: [&[u8]; 4] = [b"a", b"a", b"a", b"b"];
        let mut checkpoint = None;

        for (index, bytes) in sequence.into_iter().enumerate() {
            let mut expected = vec![0u32; constraint.body_mask_len()];
            state.fill_mask_dynamic(&mut expected);
            if state.has_parser_ambiguity() {
                let mut first = vec![0u32; constraint.body_mask_len()];
                let mut second = vec![0u32; constraint.body_mask_len()];
                assert!(state.fill_mask_indexed_dag(&mut first, true));
                assert!(state.fill_mask_indexed_dag(&mut second, true));
                assert_eq!(first, expected);
                assert_eq!(second, expected, "same-state cache hit changed the mask");
            }
            assert!(state
                .commit_bytes(bytes), "test sequence should remain valid");
            if index == 1 {
                checkpoint = Some(state.clone());
            }
        }

        state = checkpoint.expect("checkpoint should be captured");
        let mut expected = vec![0u32; constraint.body_mask_len()];
        state.fill_mask_dynamic(&mut expected);
        if state.has_parser_ambiguity() {
            let mut actual = vec![0u32; constraint.body_mask_len()];
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
    precomputed: &'a WeightTokenMaskCache,
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
        precomputed: &'a WeightTokenMaskCache,
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
        result?.intersect_with_runtime_weight_reuse(
            RuntimeWeightRef::Materialized(weight),
            self.precomputed,
        )
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
        let result = self.constraint.parser_dwa().expect("indexed DAG requires owned DWA").states()[dwa_state as usize]
            .final_weight
            .as_ref()
            .and_then(|weight| {
                self.runtime.accumulators[accumulator_id as usize]
                    .intersect_with_runtime_weight_reuse(
                        RuntimeWeightRef::Materialized(weight),
                        self.precomputed,
                    )
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
        if let Some(dense) = self.precomputed.get(&token_key).and_then(WeightTokenMaskEntry::dense) {
            return Self::single_dense_transition_mask(Arc::clone(dense));
        }
        let mut dense = vec![0u64; self.constraint.internal_token_dense_words];
        DenseMaskAcc::for_each_runtime_token_range_word(
            RuntimeTokenSetRef::Materialized(tokens),
            dense.len(),
            |index, mask| dense[index] |= mask,
        );
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
    /// parser-DWA weights. Therefore `E(q, G, a) = a ∩ E(q, G, U)` for every
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
                let &parser_state =
                    values.get(values.len().saturating_sub(1 + offset))?;
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

fn enqueue_weighted_transition(
    queue: &mut MaskQueue,
    popped: &DenseMaskGSS,
    target: u32,
    weight: RuntimeWeightRef<'_>,
    precomputed: &WeightTokenMaskCache,
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
    precomputed: &WeightTokenMaskCache,
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
        transition_intersection_cache,
        profile,
    );
}

/// One exact weighted-GSS queue traversal for ordinary and boundary parser
/// DWAs. Namespace selection and weight representation are supplied outside
/// the traversal; queue scheduling, decomposition and profiling are shared.
/// Closures are monomorphised, so ordinary lookup/intersection fast paths keep
/// their existing implementation rather than acquiring virtual dispatch.
/// Inline the executor into each provider so extracting the shared loop does
/// not leave the ordinary hot path behind an additional closure-call boundary.
#[inline(always)]
fn walk_static_dwa_gss_queue<W>(
    queue: &mut MaskQueue,
    profile: &mut Option<MaskInnerProfileStats>,
    mut final_weight: impl FnMut(u32) -> Option<W>,
    mut transition: impl FnMut(u32, u32) -> Option<(u32, W)>,
    mut accumulate: impl FnMut(W, &DenseMaskGSS),
    mut enqueue: impl FnMut(&mut MaskQueue, u32, W, &DenseMaskGSS, &mut Option<MaskInnerProfileStats>),
) {
    loop {
        let popped = queue.pop_next();
        if let Some(profile) = profile.as_mut() {
            profile.queue_pop_ns = queue.debug_stats().pop_total_ns;
        }
        let Some((wa_state, gss)) = popped else { break; };
        if let Some(weight) = final_weight(wa_state) {
            let started = profile.as_ref().map(|_| Instant::now());
            accumulate(weight, &gss);
            if let (Some(profile), Some(started)) = (profile.as_mut(), started) {
                profile.token_accumulation_ns += elapsed_ns(started);
            }
        }
        let decompose_started = profile.as_ref().map(|_| Instant::now());
        gss.for_each_decomposed(|parser_state, popped| {
            let callback_started = profile.as_ref().map(|_| Instant::now());
            queue.record_loop_decompose_callback();
            let lookup_started = profile.as_ref().map(|_| Instant::now());
            let edge = transition(wa_state, parser_state);
            if let (Some(profile), Some(started)) = (profile.as_mut(), lookup_started) {
                profile.transition_lookup_ns += elapsed_ns(started);
            }
            if let Some((target, weight)) = edge {
                queue.record_parser_dwa_transition_enqueue();
                enqueue(queue, target, weight, &popped, profile);
            }
            if let (Some(profile), Some(started)) = (profile.as_mut(), callback_started) {
                profile.loop_decompose_callback_ns += elapsed_ns(started);
            }
        });
        if let (Some(profile), Some(started)) = (profile.as_mut(), decompose_started) {
            profile.loop_decompose_total_ns += elapsed_ns(started);
        }
    }
}

fn boundary_queue_seed(
    stacks: &LeveledGSS<u32, ()>,
    seed: &DenseMaskAcc,
    top_live: &impl Fn(Option<u32>) -> bool,
) -> DenseMaskGSS {
    let dense = stacks.apply(|_| seed.clone());
    let mut filtered = if top_live(None) { dense.isolate(None) } else { DenseMaskGSS::empty() };
    dense.for_each_decomposed(|top, popped| {
        if top_live(Some(top)) {
            filtered = filtered.merge(&popped.push(top));
        }
    });
    filtered
}

fn merge_boundary_dense_result(output: &mut Option<DenseMaskAcc>, incoming: Option<DenseMaskAcc>) {
    if let Some(incoming) = incoming {
        if let Some(output) = output.as_mut() { output.merge_in_place(&incoming); }
        else { *output = Some(incoming); }
    }
}

/// Provider adapter for private, materialized boundary-DWA weights. The graph
/// traversal and dense intersection/cache machinery are the ordinary engine's.
fn boundary_weight_mask_via_shared_queue(
    dwa: &crate::automata::weighted_u32::dwa::DWA,
    tsids: &[u32],
    token_count: usize,
    stacks: &LeveledGSS<u32, ()>,
    top_live: &impl Fn(Option<u32>) -> bool,
) -> Option<DenseMaskAcc> {
    if token_count == 0 || tsids.is_empty() { return None; }
    let mut full = vec![u64::MAX; token_count.div_ceil(64)];
    if !token_count.is_multiple_of(64) { *full.last_mut().unwrap() = (1u64 << (token_count % 64)) - 1; }
    let seed = DenseMaskAcc::from_dense_arc_for_tsids(tsids, Arc::from(full))?;
    let mut queue = MaskQueue::new();
    queue.enqueue(dwa.start_state(), boundary_queue_seed(stacks, &seed, top_live));
    let precomputed = WeightTokenMaskCache::default();
    let mut intersection_cache = DenseTokenSetIntersectionSmallCache::new();
    let mut output = None;
    walk_static_dwa_gss_queue(
        &mut queue,
        &mut None,
        |state| dwa.states().get(state as usize).and_then(|row| row.final_weight.as_ref())
            .map(RuntimeWeightRef::Materialized),
        |state, parser| {
            let row = &dwa.states().get(state as usize)?.transitions;
            row.get(&encode_positive_label(parser)).or_else(|| row.get(&DEFAULT_LABEL))
                .map(|(target, weight)| (*target, RuntimeWeightRef::Materialized(weight)))
        },
        |weight, gss| {
            gss.for_each_acc(|acc| {
                merge_boundary_dense_result(&mut output,
                    acc.intersect_with_runtime_weight_reuse(weight, &precomputed));
            });
        },
        |queue, target, weight, popped, profile| {
            enqueue_weighted_transition(queue, popped, target, weight, &precomputed,
                &mut intersection_cache, profile);
        },
    );
    output
}

impl ConstraintState {
    fn or_segmented_component_mask(&self, output: &mut [u32], component_mask: &[u32]) {
        // Retained components can carry private linker sentinel IDs (and other
        // component-local specials) that are deliberately absent from the
        // finished outer constraint.  Their masks are expressed in original
        // token-ID space, so never OR those private IDs straight through into
        // the caller-visible mask.  Intersect with the outer constraint's
        // actual token universe while copying set bits.
        // Most model vocabularies have exactly the byte IDs 0..N. Prove that
        // from the existing immutable index: N distinct nonnegative IDs with
        // maximum N-1 leave no holes. No vocabulary scan, derived table or
        // composition-build work is needed. Copy this proven prefix with the
        // same word-wise OR used by the ordinary mask path, and retain exact
        // pointwise membership for sparse IDs / private-special-token tails.
        let (byte_count, max_byte_id) = (
            self.constraint.token_bytes.len(),
            self.constraint.token_bytes.max_token_id(),
        );
        let dense_byte_count = if byte_count != 0
            && max_byte_id.is_some_and(|id| u64::from(id) + 1 == byte_count as u64)
        { byte_count } else { 0 };
        let full_words = (dense_byte_count / 32).min(output.len()).min(component_mask.len());
        for (target, &source) in output[..full_words].iter_mut().zip(&component_mask[..full_words]) {
            *target |= source;
        }
        for (word_index, (&source_word, target_word)) in component_mask
            .iter()
            .zip(output.iter_mut())
            .enumerate()
            .skip(full_words)
        {
            let known_prefix = if word_index == dense_byte_count / 32 && dense_byte_count % 32 != 0 {
                (1u32 << (dense_byte_count % 32)) - 1
            } else { 0 };
            *target_word |= source_word & known_prefix;
            let mut remaining = source_word & !*target_word;
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

    /// Project an outer recursive tokenizer state into its owning component.
    fn segmented_local_tokenizer_states(
        &self,
        component_index: usize,
        global_state: u32,
    ) -> SmallVec<[u32; 8]> {
        self.constraint
            .recursive_tokenizer_state_for_component(component_index, global_state)
            .into_iter()
            .collect()
    }

    fn segmented_local_disallowed(
        &self,
        component_index: usize,
        component: &crate::runtime::SegmentedParserComponent,
        source: &TerminalsDisallowed,
    ) -> TerminalsDisallowed {
        let terminal_start = component.terminal_offset;
        let terminal_end = terminal_start.saturating_add(component.constraint.parser_terminal_count());
        let mut result = TerminalsDisallowed::new();
        for (global_tokenizer_state, terminals) in source.iter() {
            let local_tokenizer_states = self.segmented_local_tokenizer_states(component_index, *global_tokenizer_state);
            for local_tokenizer_state in local_tokenizer_states {
                for &terminal in terminals.iter() {
                    if let Some(local_terminal) = self
                            .constraint
                            .recursive_terminal_for_component(component_index, terminal)
                        {
                            result = result.with_insert(local_tokenizer_state, local_terminal);
                            continue;
                        }
                    if terminal_start <= terminal && terminal < terminal_end {
                        result = result.with_insert(
                            local_tokenizer_state,
                            terminal - terminal_start,
                        );
                    }
                    for &(alias, local_terminal) in &component.global_terminal_aliases {
                        if alias == terminal {
                            result = result.with_insert(local_tokenizer_state, local_terminal);
                        }
                    }
                }
            }
        }
        result
    }

    /// Evaluate each component's exact local mask, then its boundary shards.
    /// Scoped state intervals identify the owning component.
    fn try_fill_mask_segmented(&self, buf: &mut [u32]) -> bool {
        let profile = crate::env::var_os("GLRMASK_PROFILE_SEGMENTED_MASK").is_some();
        let total_started_at = profile.then(Instant::now);
        let Some(overlay) = self.constraint.static_dynamic_overlay.as_ref() else {
            return false;
        };
        if overlay.segmented_parser_components.is_empty() {
            return false;
        }

        let mut projected_states = SmallVec::<[ParserStateMap; 4]>::new();
        projected_states.resize_with(
            overlay.segmented_parser_components.len(),
            ParserStateMap::default,
        );

        for (&global_tokenizer_state, gss) in self.state.iter() {
            // A control-closed recursive frontier can represent exponentially
            // many valid stacks. Project its shared graph instead of declaring
            // a static-mask failure after enumerating 128 concrete paths.
            if !gss.is_single_path() {
                for (component_index, component) in overlay.segmented_parser_components.iter().enumerate() {
                    let local_tokenizer_states = self.segmented_local_tokenizer_states(component_index, global_tokenizer_state);
                    if local_tokenizer_states.is_empty() { continue; }
                    let projected = gss.map_nonempty_top_prefixes(|&global|
                        self.constraint.compact_segmented_parser_local_state(component_index, global));
                    if projected.is_empty() { continue; }
                    let projected = projected.apply_and_prune(|acc|
                        Some(self.segmented_local_disallowed(component_index, component, acc)));
                    for local_state in local_tokenizer_states {
                        projected_states[component_index].merge_insert(local_state, projected.clone());
                    }
                }
                continue;
            }
            let complete = gss.for_each_stack_top_first_bounded(128, |top_first, acc| {
                let Some(&global_top) = top_first.first() else {
                    return;
                };
                let Some(component_index) =
                    self.constraint.compact_segmented_parser_component(global_top).map(|(component, _)| component)
                else {
                    return;
                };
                let Some(component) = overlay
                    .segmented_parser_components
                    .get(component_index)
                else {
                    return;
                };
                let local_tokenizer_states = self.segmented_local_tokenizer_states(component_index, global_tokenizer_state);
                if local_tokenizer_states.is_empty() {
                    return;
                }

                let mut local_top_first = SmallVec::<[u32; 64]>::new();
                for &global_parser_state in top_first {
                    let Some(local) = self.constraint.compact_segmented_parser_local_state(component_index, global_parser_state) else {
                        break;
                    };
                    local_top_first.push(local);
                }
                if local_top_first.is_empty() {
                    return;
                }
                local_top_first.reverse();
                let local_stack = local_top_first.into_vec();
                let local_disallowed =
                    self.segmented_local_disallowed(component_index, component, acc);
                for local_tokenizer_state in local_tokenizer_states {
                    projected_states[component_index].merge_insert(
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

        buf.fill(0);
        #[cfg(test)]
        segmented_mask_stage_trace("baseline", buf);
        let mut component_times = SmallVec::<[u64; 4]>::new();
        let required_component_mask_len = overlay
            .segmented_parser_components
            .iter()
            .zip(projected_states.iter())
            .filter(|&(component, state)| !state.is_empty() ).map(|(component, state)| component.constraint.body_mask_len())
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
            terminated: false,
            markers: None,
                constraint: component.constraint.as_ref().clone(),
                state,
                buffers: CommitBuffers::for_mask_only_shadow(),
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
            self.or_segmented_component_mask(buf, component_buf);
            #[cfg(test)]
            segmented_mask_stage_trace(&format!("component-{component_index}"), buf);
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
        #[cfg(test)]
        segmented_mask_stage_trace("shards", buf);
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

    /// Test-only: report one boundary shard's mask contribution by snapshot
    /// diff, together with the effective top state / source context that made
    /// it active. Enabled by `GLRMASK_DEBUG_MASK_STAGES`.
    #[cfg(test)]
    fn trace_segmented_shard_delta(
        &self,
        component_index: usize,
        shard: &crate::runtime::SegmentedBoundaryShard,
        before: &[u32],
        after: &[u32],
    ) {
        if !segmented_mask_stage_trace_enabled() {
            return;
        }
        let mut added = Vec::new();
        for (word_index, (&before_word, &after_word)) in
            before.iter().zip(after.iter()).enumerate()
        {
            let mut delta = after_word & !before_word;
            while delta != 0 {
                let bit = delta.trailing_zeros() as usize;
                added.push((word_index * 32 + bit) as u32);
                delta &= delta - 1;
            }
        }
        let mut contexts = Vec::new();
        for (&global_tokenizer_state, gss) in self.state.iter() {
            let _ = gss.for_each_stack_top_first_bounded(128, |top_first, _| {
                match top_first.first().copied() {
                    Some(top) => {
                        let owner = self.constraint
                            .compact_segmented_parser_component(top)
                            .map(|(component, _)| component as u32);
                        contexts.push(format!(
                            "gts={global_tokenizer_state} top={top} owner={owner:?} active={}",
                            owner == Some(component_index as u32),
                        ));
                    }
                    None => contexts.push(format!(
                        "gts={global_tokenizer_state} empty_stack active={}",
                        component_index == 0,
                    )),
                }
            });
        }
        let backend = match &shard.backend {
            crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_) => "StaticParser",
            crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect => "DynamicDirect",
        };
        let candidates = shard
            .candidate_tokens
            .as_ref()
            .map(|tokens| tokens.iter().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        eprintln!(
            "[mask-stages] shard[{component_index}] backend={backend} start_component={} candidates={candidates:?} added_bits={added:?} token8={} token9={} contexts={contexts:?}",
            component_index,
            added.contains(&8),
            added.contains(&9),
        );
    }

    // Keep the diagnostic scope outside the implementation's stack-frame
    // prologue/epilogue. Otherwise its large shadow-state frame is incorrectly
    // charged to non-boundary work by whole-minus-dispatch accounting.
    #[inline(always)]
    fn or_segmented_boundary_shards_mask(
        &self,
        overlay: &crate::runtime::StaticDynamicOverlayMetadata,
        buf: &mut [u32],
    ) -> bool {
        #[cfg(any(test, feature = "internal-api"))]
        let _boundary_cpu = super::boundary_cpu_profile::Span::enter(super::boundary_cpu_profile::Kind::Dispatch);
        self.or_segmented_boundary_shards_mask_impl(overlay, buf)
    }

    #[inline(never)]
    fn or_segmented_boundary_shards_mask_impl(
        &self,
        overlay: &crate::runtime::StaticDynamicOverlayMetadata,
        buf: &mut [u32],
    ) -> bool {
        if overlay
            .segmented_parser_components
            .iter()
            .any(|component| component.boundary.is_some())
        {
            let mut needs_direct_dynamic = false;
            // Vocabulary preparation is binding-local and done once; execution
            // remains the same shared optimized dynamic walker as ordinary masks.
            let restricted = crate::env::var_os("GLRMASK_DISABLE_BOUNDARY_VOCAB").is_none();
            for (component_index, component) in
                overlay.segmented_parser_components.iter().enumerate()
            {
                let Some(shard) = component.boundary.as_ref() else {
                    continue;
                };
                #[cfg(test)]
                let before_shard = segmented_mask_stage_trace_enabled().then(|| buf.to_vec());
                let ok = match &shard.backend {
                    crate::runtime::SegmentedBoundaryShardBackend::StaticParser(boundary) => {
                        self.or_segmented_boundary_parser_mask(
                            boundary,
                            component_index as u32,
                            buf,
                        )
                    }
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect => {
                        #[cfg(any(test, feature = "internal-api"))]
                        let _dynamic_cpu = super::boundary_cpu_profile::Span::enter(super::boundary_cpu_profile::Kind::Dynamic);
                        if !self.segmented_boundary_shard_may_be_active(component_index) {
                            continue;
                        }
                        // Strict-static trap (milestone H): a claimed static
                        // path must never silently evaluate DynamicDirect
                        // masks. Env-gated so exact dynamic compositions are
                        // unaffected; strict-static tests set
                        // GLRMASK_STRICT_STATIC_TRAP_DYNAMIC=1 and any firing
                        // DynamicDirect shard panics loudly here instead of
                        // contributing hidden dynamic admissions.
                        crate::compiler::boundary_transfer::strict_static_trap_dynamic_for_state(
                            "segmented_dynamic_direct_shard",
                            self.constraint.uses_dynamic_runtime(),
                        );
                        let handled = restricted && shard.candidate_tokens.as_deref().is_some_and(|ids| {
                            let prepared = shard.mask_vocabulary.get_or_init(|| {
                                super::dynamic_mask::PreparedMaskVocabulary::new(&self.constraint, ids)
                            });
                            prepared.as_ref().ok().is_some_and(|vocab| {
                                if crate::env::var_os("GLRMASK_DISABLE_BOUNDARY_FRONTIER_SELECTION").is_some() {
                                    return vocab.or_mask(self, buf).expect("exact vocabulary mask failed");
                                }
                                let Some(frontier) = self.segmented_boundary_initial_frontier(component_index)
                                    else { return false; };
                                if frontier.is_empty() { return true; }
                                if crate::env::var_os("GLRMASK_PROFILE_DYNAMIC_BOUNDARY_CANDIDATE_TRIE").is_some() {
                                    eprintln!("[glrmask/profile][boundary_initial_frontier] component={} before={} after={}",
                                        component_index, self.state.len(), frontier.len());
                                }
                                let view = ConstraintState {
                                    terminated: self.terminated,
            markers: None,
                                    constraint: self.constraint.clone(),
                                    state: frontier,
                                    buffers: CommitBuffers::for_mask_only_shadow(),
                                    generation: self.generation,
                                    mask_cache: Mutex::new(None),
                                    mask_scratch: Arc::clone(&self.mask_scratch),
                                };
                                vocab.or_mask(&view, buf).expect("exact vocabulary mask failed")
                            })
                        });
                        needs_direct_dynamic |= !handled;
                        true
                    }
                };
                #[cfg(test)]
                if let Some(before_shard) = before_shard {
                    self.trace_segmented_shard_delta(component_index, shard, &before_shard, buf);
                }
                if !ok {
                    return false;
                }
            }
            if needs_direct_dynamic {
                #[cfg(any(test, feature = "internal-api"))]
                let _fallback_cpu = super::boundary_cpu_profile::Span::enter(super::boundary_cpu_profile::Kind::Dynamic);
                // A DynamicDirect shard owned a live stack on a path that also
                // evaluated static shards. The per-shard arm above already
                // traps loudly under the strict flag; this second trap names
                // the unified/recursive walker fallback itself so a missed arm
                // cannot silently contribute exact dynamic admissions.
                crate::compiler::boundary_transfer::strict_static_trap_dynamic_for_state(
                    "segmented_boundary_needs_direct_dynamic",
                    self.constraint.uses_dynamic_runtime(),
                );
                if self.constraint.uses_compact_segmented_parser_runtime() {
                    // Use the same shared provider walk as all-dynamic
                    // composition, then union with the static contribution.
                    let mut dynamic = vec![0u32; self.constraint.mask_bit_len().div_ceil(32)];
                    self.fill_mask_dynamic(&mut dynamic);
                    for (out, value) in buf.iter_mut().zip(dynamic) { *out |= value; }
                } else {
                    // The unified strict walker evaluates the complete exact
                    // composed language and ORs it with the already-computed A
                    // baseline. No B-domain candidate discovery is needed.
                    super::dynamic_mask::or_mask_dynamic_additions(self, buf);
                }
            }
            return true;
        }

        true
    }

    /// Independent test oracle retained from the pre-unification recursive
    /// walker. Production recursive masking uses the ordinary shared traversal
    /// with finite/config scoped providers in dynamic_mask.
    #[cfg(test)]
    fn or_recursive_dynamic_full_walk_exact(&self, buf: &mut [u32]) {
        self.or_recursive_dynamic_full_walk_exact_with_budget(buf, 256);
    }

    #[cfg(test)]
    fn or_recursive_dynamic_full_walk_exact_with_budget(&self, buf: &mut [u32], state_budget: usize) {
        crate::compiler::boundary_transfer::strict_static_trap_dynamic_for_state(
            "or_recursive_dynamic_full_walk_exact",
            self.constraint.uses_dynamic_runtime(),
        );
        let vocab = self.constraint.dynamic_mask_vocab_for_runtime();
        let trie = vocab.trie.as_ref();
        let stack_len = usize::from(trie.full_walk_max_parent_depth()).saturating_add(2);
        let mut state_stack = vec![None; stack_len.max(1)];
        let alphabet = if state_budget == 0 { std::array::from_fn(|byte| byte as u8) }
            else { recursive_mask_byte_representatives(&self.constraint) };
        // A proved finite deterministic leaf alphabet cannot produce a wide
        // epsilon frontier. The recursive queue does not use speculative flat
        // commits, so it needs no eager pool of 256 flat-frontier GSS objects.
        // Keep the established scratch policy for unproved/virtual lexers.
        let mut buffers = if alphabet.iter().enumerate().any(|(i, &b)| i != b as usize) {
            CommitBuffers::for_finite_recursive_mask()
        } else {
            CommitBuffers::default()
        };
        let mut transitions = RecursiveMaskTransitions::new(state_budget, alphabet);
        state_stack[0] = Some(match transitions.intern(&self.state) {
            Some(id) => RecursiveMaskFrontier::Cached(id),
            None => RecursiveMaskFrontier::Uncached(self.state.clone()),
        });

        let mark_node = |node: u32, live: bool, buf: &mut [u32]| {
            if !live {
                return;
            }
            let Some(canonical) = trie.node(node).token_id else {
                return;
            };
            let Some(token_ids) = vocab.token_ids(canonical) else {
                return;
            };
            for &token_id in token_ids {
                // A token id with explicit special-token semantics denotes the
                // union of its byte and special routes. Probe it once through
                // authoritative token commit below rather than accepting from
                // only the byte route here.
                if !self.constraint.has_special_token_id(token_id) {
                    set_original_mask_bit(buf, token_id);
                }
            }
        };

        // A zero-byte vocabulary entry lives on the radix-trie root. It is not
        // an ordinary model-token byte route because consuming it would make
        // no byte progress. Explicit grammar special-token semantics for the
        // same ID are handled pointwise below.
        let edges = trie.walk_edges();
        let mut edge_index = 0;
        while let Some(edge) = edges.get(edge_index) {
            let parent_depth = edge.parent_depth as usize;
            let child_depth = parent_depth + 1;
            let child_state = state_stack[parent_depth].as_ref().and_then(|parent| {
                transitions.advance(
                    &self.constraint,
                    parent,
                    &mut buffers,
                    trie.walk_edge_bytes(edge),
                )
            });
            let live = child_state.is_some();
            state_stack[child_depth] = child_state;
            if !live {
                // Exact byte advancement has rejected this prefix. Previously
                // its descendants were still visited, but their absent parent
                // states suppressed every semantic advance and token mark.
                // Skip precisely that same work using the existing DFS jump;
                // no boundary-candidate or lexer-liveness approximation is
                // involved. The next edge's parent is an already-live ancestor,
                // so stale deeper stack slots cannot be read before replacement.
                // Special-token routes remain independent and are probed below.
                debug_assert!(edge.subtree_end as usize > edge_index);
                edge_index = edge.subtree_end as usize;
                continue;
            }
            mark_node(edge.child, live, buf);
            edge_index += 1;
        }

        // Special-token semantics can add a parser path independent of the
        // token's byte spelling, so those ids deliberately stay pointwise.
        let mut pointwise_candidates = self
            .constraint
            .special_token_terminals
            .iter()
            .filter(|special| {
                !self
                    .constraint
                    .is_late_grammar_placeholder_terminal(special.terminal_id)
            })
            .flat_map(|special| special.token_ids())
            .collect::<Vec<_>>();
        pointwise_candidates.sort_unstable();
        pointwise_candidates.dedup();
        for token_id in pointwise_candidates {
            let admitted = if self
                .constraint
                .token_bytes_for_id(token_id)
                .is_some_and(|bytes| bytes.is_empty())
            {
                // The byte trie represents an empty vocabulary token at its
                // root. Exact special-token IDs with that spelling are
                // deliberately pointwise: an empty byte spelling must not
                // become a zero-progress alternate route that bypasses the
                // parser's special-token terminal.
                crate::runtime::commit::advance_special_token_paths(
                    &self.constraint,
                    &self.state,
                    token_id,
                )
                .is_some_and(|gss| !gss.is_empty())
            } else {
                crate::runtime::commit::token_admissible_from_state_exact(
                    &self.constraint,
                    &self.state,
                    &mut buffers,
                    token_id,
                )
            };
            if admitted {
                set_original_mask_bit(buf, token_id);
            }
        }
    }

    /// Independent pre-unification mask oracle, compiled only for tests.
    #[cfg(test)]
    pub(crate) fn fill_recursive_mask_by_exact_full_walk(&self, buf: &mut [u32]) {
        crate::compiler::boundary_transfer::strict_static_trap_dynamic_for_state(
            "fill_recursive_mask_by_exact_full_walk",
            self.constraint.uses_dynamic_runtime(),
        );
        buf.fill(0);
        self.or_recursive_dynamic_full_walk_exact(buf);
    }

    /// Conservatively test whether this shard can own at least one current
    /// composed parser-stack top. Incomplete bounded GSS inspection returns
    /// true so trigger acceleration can never create a false negative.
    fn segmented_boundary_shard_may_be_active(
        &self,
        component_index: usize,
    ) -> bool {
        // Ownership depends only on the top, not on an enumeration of entire
        // stacks. peek_values is an exact union over all GSS top alternatives.
        for gss in self.state.values() {
            if component_index == 0 && !gss.isolate(None).is_empty() { return true; }
            for top in gss.peek_values() {
                let belongs = self.constraint.compact_segmented_parser_component(top)
                    .is_some_and(|(owner, _)| owner == component_index);
                if belongs { return true; }
            }
        }
        false
    }

    /// Select the initial relation belonging to one vocabulary shard. Keep
    /// whole lower stacks and exclusion accumulators; never enumerate or cut
    /// stack paths. This is input selection for the ordinary shared walker.
    fn segmented_boundary_initial_frontier(
        &self,
        component_index: usize,
    ) -> Option<ParserStateMap> {
        let mut selected = ParserStateMap::default();
        for (&lexer, gss) in self.state.iter() {
            if gss.is_empty() { continue; }
            let tops = gss.peek_values();
            let mut keep = SmallVec::<[u32; 4]>::new();
            for &top in &tops {
                let (owner, _) = self.constraint.compact_segmented_parser_component(top)?;
                if owner == component_index { keep.push(top); }
            }
            let empty = gss.isolate(None);
            let branch = if keep.len() == tops.len() && (component_index == 0 || empty.is_empty()) {
                gss.clone()
            } else {
                let mut branch = if component_index == 0 { empty } else { ParserGSS::empty() };
                for top in keep { branch = branch.merge(&gss.isolate(Some(top))); }
                branch
            };
            if !branch.is_empty() { selected.merge_insert(lexer, branch); }
        }
        Some(selected)
    }

    /// Admit one exact-DAG group's accepted Weight into `buf`, preserving the
    /// per-path eligibility correlation (the group already unions exactly the
    /// paths sharing its accumulator). Returns `false` to decline when an
    /// internal token has no original mapping; the caller then runs the exact
    /// fallback (which the strict-static trap turns into a loud panic).
    fn admit_boundary_weight_group(
        &self,
        boundary: &crate::runtime::SegmentedBoundaryParser,
        accepted: &Weight,
        boundary_tsids: &[u32],
        allowed: &DenseMaskAcc,
        buf: &mut [u32],
    ) -> bool {
        fn dense_contains(acc: &DenseMaskAcc, token: u32) -> bool {
            let word = token as usize / 64;
            let bit = token % 64;
            acc.0.iter().any(|(_, dense)| {
                dense.get(word)
                    .is_some_and(|word_value| (*word_value & (1u64 << bit)) != 0)
            })
        }
        for &boundary_tsid in boundary_tsids {
            let Some(tokens) = accepted.token_set_for_tsid_ref(boundary_tsid) else {
                continue;
            };
            for range in tokens.ranges() {
                for internal_token in range {
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
                        if outer_internal != u32::MAX && dense_contains(allowed, outer_internal)
                        {
                            set_original_mask_bit(buf, original);
                        }
                    }
                }
            }
        }
        true
    }

    fn or_boundary_dag_via_shared_queue(
        &self,
        boundary: &crate::runtime::SegmentedBoundaryParser,
        parser_dwa: &crate::automata::weighted_u32::dwa::DWA,
        boundary_tsids: &[u32],
        global_tokenizer_state: u32,
        gss: &ParserGSS,
        top_live: &impl Fn(Option<u32>) -> bool,
        buf: &mut [u32],
    ) -> bool {
        let started = crate::env::var_os("GLRMASK_PROFILE_STATIC_BOUNDARY").is_some().then(Instant::now);
        let mut group_count = 0;
        for (stacks, accumulator) in gss.partition_by_accumulator() {
            group_count += 1;
            let Some(allowed) = self.terminals_disallowed_to_dense_acc(
                &accumulator, global_tokenizer_state,
            ) else { return false; };
            // Partition by exact exclusion accumulator BEFORE evaluation, so
            // accepted language and output eligibility remain correlated.
            // Partitioning operates on the compact GSS, never enumerating a
            // bounded list of individual stacks.
            let accepted = boundary_weight_mask_via_shared_queue(
                parser_dwa, boundary_tsids, boundary.internal_token_to_originals.len(),
                &stacks, top_live,
            );
            let Some(accepted) = accepted else { continue; };
            for (_, dense) in &accepted.0 {
                for (word, &bits) in dense.iter().enumerate() {
                    let mut bits = bits;
                    while bits != 0 {
                        let internal = word * 64 + bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let Some(originals) = boundary.internal_token_to_originals.get(internal)
                            else { return false; };
                        for &original in originals {
                            let Some(outer_internal) = self.constraint.original_token_internal_at(original)
                                .filter(|&token| token != u32::MAX) else { continue; };
                            if allowed.0.iter().any(|(_, dense)| {
                                dense.get(outer_internal as usize / 64)
                                    .is_some_and(|word| word & (1u64 << (outer_internal % 64)) != 0)
                            }) { set_original_mask_bit(buf, original); }
                        }
                    }
                }
            }
        }
        if let Some(started) = started {
            eprintln!("[glrmask/profile][static_boundary_dag] backend=shared_queue groups={group_count} total_ns={} complete=true", elapsed_ns(started));
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
        start_component: u32,
        buf: &mut [u32],
    ) -> bool {
        #[cfg(test)]
        fn reference_accepted_for_stack(
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

        fn accepted_dense_for_stack(
            dwa: &crate::automata::weighted_u32::dwa::DWA,
            tsid: u32,
            token_count: usize,
            top_first: &[u32],
        ) -> Vec<u64> {
            let mut path = vec![u64::MAX; token_count.div_ceil(64)];
            if !token_count.is_multiple_of(64) {
                *path.last_mut().expect("nonempty token universe") =
                    (1u64 << (token_count % 64)) - 1;
            }
            let mut accepted = vec![0; path.len()];
            let mut aux = Vec::new();
            // Boundary token IDs belong to this shard, not to the outer
            // constraint. Reuse the normal dense/range primitives but never
            // consult a cache keyed in the outer token coordinate.
            let precomputed = WeightTokenMaskCache::default();
            walk_single_stack::<false, _>(
                dwa.start_state(), top_first,
                |state| dwa.states().get(state as usize)?.final_weight.as_ref(),
                |state, parser| {
                    let row = &dwa.states().get(state as usize)?.transitions;
                    row.get(&encode_positive_label(parser))
                        .or_else(|| row.get(&DEFAULT_LABEL))
                        .map(|(target, weight)| (*target, weight))
                },
                |event| {
                    match event {
                        StackWalkEvent::Final(weight) => {
                            let weight = RuntimeWeightRef::Materialized(weight);
                            if weight.is_full() {
                                for (out, &bits) in accepted.iter_mut().zip(&path) {
                                    *out |= bits;
                                }
                            } else if let Some(tokens) = weight.token_set_for_tsid(tsid) {
                                DenseMaskAcc::or_dense_and_runtime_token_set_into(
                                    &path, tokens, &precomputed, &mut accepted,
                                );
                            }
                        }
                        StackWalkEvent::Intersect(weight) => {
                            return intersect_static_dense_with_weight(
                                &mut path, &mut aux, tsid,
                                RuntimeWeightRef::Materialized(weight), |_| None,
                            );
                        }
                        StackWalkEvent::Top(_) => unreachable!("top hook is compiled out"),
                    }
                    true
                },
            );
            #[cfg(test)]
            {
                // Independent pre-refactor symbolic walk: compare the whole
                // private mask before outer eligibility filtering can hide a
                // traversal, TSID, or padding error.
                let reference = reference_accepted_for_stack(dwa, top_first);
                let mut expected = vec![0u64; accepted.len()];
                for token in 0..token_count {
                    if reference.is_full() || reference.token_set_for_tsid_ref(tsid)
                        .is_some_and(|tokens| tokens.contains(token as u32))
                    {
                        expected[token / 64] |= 1u64 << (token % 64);
                    }
                }
                assert_eq!(accepted, expected, "boundary shared walk tsid={tsid} stack={top_first:?}");
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

        let parser_dwa = &boundary.parser_dwa;
        let top_live = |top: Option<u32>| match top {
            Some(value) => self.constraint
                .compact_segmented_parser_component(value)
                .is_some_and(|(owner, _)| owner == start_component as usize),
            None => start_component == 0,
        };
        let debug_boundary = crate::env::var_os("GLRMASK_DEBUG_SEGMENTED_BOUNDARY_MASK").is_some();
        for (&global_tokenizer_state, gss) in self.state.iter() {
            let boundary_tsids = match &boundary.coordinate {
                crate::runtime::BoundaryTsidCoordinate::Composed => self.constraint
                    .runtime_internal_tsids_for_tokenizer_state(global_tokenizer_state)
                    .unwrap_or_default(),
                crate::runtime::BoundaryTsidCoordinate::ShardLocal(map) => map
                    .get(global_tokenizer_state as usize)
                    .copied()
                    .filter(|&tsid| tsid != u32::MAX)
                    .into_iter()
                    .collect::<SmallVec<[u32; 4]>>(),
            };
            if debug_boundary {
                eprintln!(
                    "[glrmask/debug][segmented_boundary_state] global_tokenizer_state={} boundary_tsids={:?} paths={}",
                    global_tokenizer_state,
                    boundary_tsids,
                    gss.path_count_at_most(129),
                );
            }
            if boundary_tsids.is_empty() {
                continue;
            }
            let mut complete = true;
            // Cap-free exact evaluation: single-path GSSes keep the direct
            // per-stack walk (exactly one path, so the bound can never bite),
            // while ambiguous GSSes run the memoized DAG x DWA product
            // evaluator so >128 / shared-tail path counts stay exact with no
            // decline and no hidden dynamic fallback.
            let traversal_complete = if gss.is_single_path() {
                // Isolated static-boundary profiling (Priority 4): single-path
                // evaluations dominate ordinary states; time them separately
                // from DynamicDirect and commits.
                let single_started =
                    crate::env::var_os("GLRMASK_PROFILE_STATIC_BOUNDARY").is_some().then(Instant::now);
                let mut single_paths = 0usize;
                // Phase breakdown for hotspot attribution on slow evaluations
                // (allowed = exclusion accumulator projection, walk = DWA
                // product walk, admit = internal->original mapping/filter).
                let mut phase_allowed_ns: u64 = 0;
                let mut phase_walk_ns: u64 = 0;
                let mut phase_admit_ns: u64 = 0;
                let mut single_depth = 0usize;
                let complete_single = gss.for_each_stack_top_first_bounded(128, |top_first, acc| {
                    single_paths += 1;
                    single_depth = single_depth.max(top_first.len());
                if !top_live(top_first.first().copied()) {
                    return;
                }
                let phase_mark = single_started.is_some().then(Instant::now);
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
                if let Some(mark) = phase_mark {
                    phase_allowed_ns += elapsed_ns(mark);
                }
                let mut debug_internal = Vec::new();
                let mut debug_originals = Vec::new();
                for &boundary_tsid in &boundary_tsids {
                    let walk_mark = single_started.is_some().then(Instant::now);
                    let accepted = accepted_dense_for_stack(
                        parser_dwa, boundary_tsid,
                        boundary.internal_token_to_originals.len(), top_first,
                    );
                    if let Some(mark) = walk_mark {
                        phase_walk_ns += elapsed_ns(mark);
                    }
                    let admit_mark = single_started.is_some().then(Instant::now);
                    for (word_index, mut bits) in accepted.into_iter().enumerate() {
                        while bits != 0 {
                            let internal_token = (word_index * 64) as u32 + bits.trailing_zeros();
                            bits &= bits - 1;
                            let Some(originals) = boundary
                                .internal_token_to_originals
                                .get(internal_token as usize)
                            else {
                                complete = false;
                                return;
                            };
                            if debug_boundary {
                                debug_internal.push(internal_token);
                                debug_originals.extend_from_slice(originals);
                            }
                            for &original in originals {
                                let outer_internal = self
                                    .constraint
                                    .original_token_internal_at(original)
                                    .unwrap_or(u32::MAX);
                                if outer_internal != u32::MAX
                                    && dense_contains(&allowed, outer_internal)
                                {
                                    set_original_mask_bit(buf, original);
                                }
                            }
                        }
                    }
                    if let Some(mark) = admit_mark {
                        phase_admit_ns += elapsed_ns(mark);
                    }
                }
                if debug_boundary {
                    debug_internal.sort_unstable();
                    debug_internal.dedup();
                    debug_originals.sort_unstable();
                    debug_originals.dedup();
                    eprintln!(
                        "[glrmask/debug][segmented_boundary_stack] top_first={:?} tsids={:?} accepted_internal={:?} accepted_originals={:?}",
                        top_first,
                        boundary_tsids,
                        debug_internal,
                        debug_originals,
                    );
                }
                });
                if let Some(single_started) = single_started {
                    let total_ns = elapsed_ns(single_started);
                    eprintln!(
                        "[glrmask/profile][static_boundary_dag] gss_nodes=0 upper_memo=0 lower_memo=0 groups=0 dwa_steps=0 weight_unions=0 weight_intersections=0 total_ns={total_ns} complete={complete} compact=single paths={single_paths} depth={single_depth} allowed_ns={phase_allowed_ns} walk_ns={phase_walk_ns} admit_ns={phase_admit_ns}",
                    );
                }
                complete_single
            } else {
                // Exact memoized DAG x DWA product evaluation (no path cap).
                // Groups preserve the per-path eligibility correlation:
                // union over groups of (accepted(group) intersect
                // eligible(group)), never union-accepts independently of
                // union-eligibility.
                #[cfg(test)]
                let mut reference_mask = buf.to_vec();
                let shared_complete = self.or_boundary_dag_via_shared_queue(
                    boundary, parser_dwa, &boundary_tsids,
                    global_tokenizer_state, gss, &top_live, buf,
                );
                #[cfg(test)]
                {
                let buf = reference_mask.as_mut_slice();
                let mut complete = true;
                let dag = gss.indexed_dag();
                let profile_static =
                    crate::env::var_os("GLRMASK_PROFILE_STATIC_BOUNDARY").is_some();
                let dag_started = profile_static.then(Instant::now);
                let mut evaluator = BoundaryWeightDagEvaluator::new(parser_dwa, &dag);
                let groups = evaluator.eval_root(&top_live);
                for (group, accepted) in &groups {
                    let Some(acc) = evaluator.group_accumulator(*group) else {
                        complete = false;
                        break;
                    };
                    let Some(allowed) = self.terminals_disallowed_to_dense_acc(
                        &acc,
                        global_tokenizer_state,
                    ) else {
                        complete = false;
                        break;
                    };
                    if !self.admit_boundary_weight_group(
                        boundary,
                        accepted,
                        &boundary_tsids,
                        &allowed,
                        buf,
                    ) {
                        complete = false;
                        break;
                    }
                }
                if let Some(dag_started) = dag_started {
                    eprintln!(
                        "[glrmask/profile][static_boundary_dag] gss_nodes={} upper_memo={} lower_memo={} groups={} dwa_steps={} weight_unions={} weight_intersections={} total_ns={} complete={complete} compact=false",
                        dag.nodes.len(),
                        evaluator.upper_memo.len(),
                        evaluator.lower_memo.len(),
                        groups.len(),
                        evaluator.dwa_steps,
                        evaluator.weight_unions,
                        evaluator.weight_intersections,
                        elapsed_ns(dag_started),
                    );
                }
                assert_eq!(shared_complete, complete, "shared boundary queue completeness differs from DAG oracle");
                }
                #[cfg(test)]
                if shared_complete {
                    assert_eq!(buf, reference_mask.as_slice(), "shared boundary queue mask differs from exact DAG oracle");
                }
                shared_complete
            };
            if !traversal_complete || !complete {
                return false;
            }
        }
        true
    }

    /// Exact overlay for an out-of-vocabulary special token reached only after
    /// a linker control chain. Ordinary parser-DWA weights remain the fast path;
    /// constraints without explicit controls pay nothing here.
    fn update_control_special_token_mask(&self, buf: &mut [u32]) {
        if self.constraint.special_token_terminals.is_empty() { return; }
        // A data-only constructor's static byte DWA deliberately has no exact
        // token edges. Query the same complete parser relation for those IDs;
        // exact-only tokens must never be reinterpreted as their vocabulary bytes.
        if !self.constraint.parser_has_controls()
            && !self.constraint.uses_compact_segmented_parser_runtime()
            && !self.constraint.has_template_parser()
        {
            return;
        }
        for special in &self.constraint.special_token_terminals {
            if self.constraint.is_late_grammar_placeholder_terminal(special.terminal_id) { continue; }
            if super::commit::advance_special_terminal_paths(
                &self.constraint, &self.state, special.terminal_id,
            ).is_some_and(|gss| !gss.is_empty()) {
                special.or_mask(buf);
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
        self.enforce_empty_byte_token_domain(buf);
    }

    /// A zero-byte model ID is not an ordinary byte transition. Enforce the
    /// shared commit endpoint policy at final output, retaining an
    /// empty-spelled special ID only via its exact live action. Root end IDs
    /// are applied separately after this body-domain filter.
    fn enforce_empty_byte_token_domain(&self, buf: &mut [u32]) {
        if self.constraint.empty_byte_token_ids.is_empty()
            || !(self.constraint.has_template_parser()
                || self.constraint.uses_dynamic_runtime()
                || self.constraint.uses_compact_segmented_parser_runtime())
        { return; }
        // Liveness belongs to a parser terminal, not to every token in its
        // interval. In particular, an all-vocabulary empty-spelled binding must
        // not perform one parser advance per model ID. This inventory is sorted
        // and unique, whether prepared from Vocab or reconstructed during load.
        let empty_ids = self.constraint.empty_byte_token_ids.as_ref();
        let mut live_ranges = smallvec::SmallVec::<[(u32, u32); 8]>::new();
        for special in &self.constraint.special_token_terminals {
            if self.constraint.is_late_grammar_placeholder_terminal(special.terminal_id) {
                continue;
            }
            let first_empty = empty_ids.partition_point(|&id| id < special.token_id);
            if empty_ids.get(first_empty).is_none_or(|&id| id > special.token_id_end) {
                continue;
            }
            if super::commit::advance_special_terminal_paths(
                &self.constraint, &self.state, special.terminal_id,
            ).is_some_and(|stack| !stack.is_empty()) {
                live_ranges.push((special.token_id, special.token_id_end));
            }
        }
        retain_live_empty_byte_ranges(buf, empty_ids, &mut live_ranges);
    }

    fn fill_blocked_seed_dense(
        &self,
        terminals_disallowed: &TerminalsDisallowed,
        blocked: &mut Vec<u64>,
    ) {
        const MAX_FALLBACK_MASKS: usize = 512;

        let profile = crate::env::var_os("GLRMASK_PROFILE_SEED_EXCLUSIONS").is_some();
        let started = profile.then(Instant::now);
        blocked.clear();
        blocked.resize(self.constraint.seed_universe_dense.len(), 0);
        if terminals_disallowed.is_empty() {
            return;
        }

        let mut missing_pairs = SmallVec::<[(u32, TerminalID); 4]>::new();
        for (&continuation_tokenizer_state, terminals) in terminals_disallowed.iter() {
            let exclusion_state = self.constraint.static_exclusion_state(continuation_tokenizer_state)
                .expect("validated recursive lexer state must have a static observation");
            for &terminal_id in terminals.iter() {
                if let Some(mask) = self
                    .constraint
                    .seed_terminal_dense
                    .get(&(exclusion_state, terminal_id))
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
                &self.constraint,
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
                let remaining = MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS
                    .saturating_sub(paths.len())
                    .min(MASK_SINGLE_PATH_DIRECT_MAX_PATHS_PER_GSS);
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
                    let remaining = MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS
                        .saturating_sub(total_paths)
                        .min(MASK_SINGLE_PATH_DIRECT_MAX_PATHS_PER_GSS);
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
                    let remaining = MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS
                        .saturating_sub(paths.len())
                        .min(MASK_SINGLE_PATH_DIRECT_MAX_PATHS_PER_GSS);
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
                walk_single_stack::<true, _>(
                    self.constraint.runtime_parser_dwa_start_state(),
                    stack,
                    |state| self.constraint.runtime_parser_dwa_final_weight(state),
                    |state, parser| self.constraint.runtime_parser_dwa_transition(state, parser),
                    |event| {
                        match event {
                            StackWalkEvent::Final(final_weight) => {
                                if plan_ops.len() == MASK_SINGLE_PATH_DIRECT_MAX_PLAN_OPS {
                                    plans_complete = false;
                                    return false;
                                }
                                plan_ops.push(SinglePathDirectPlanOp::Merge(final_weight));
                                // Every continuation only intersects this
                                // path's candidate set. Full acceptance has
                                // already emitted that entire set.
                                if final_weight.is_full() {
                                    return false;
                                }
                            }
                            StackWalkEvent::Top(parser_state) => {
                                let positive_label = encode_positive_label(parser_state);
                                let has_direct_regular_acceptance = self
                                    .constraint
                                    .direct_regular_wide_acceptance_for_parser_state(parser_state)
                                    .is_some()
                                    || self
                                        .constraint
                                        .for_each_direct_regular_l1_acceptance(parser_state, |_| {});
                                if has_direct_regular_acceptance {
                                    plans_complete = false;
                                    return false;
                                }
                                if let Some(accept_weight) =
                                    self.constraint.runtime_parser_top_accept(positive_label)
                                {
                                    if plan_ops.len() == MASK_SINGLE_PATH_DIRECT_MAX_PLAN_OPS {
                                        plans_complete = false;
                                        return false;
                                    }
                                    plan_ops.push(SinglePathDirectPlanOp::Merge(accept_weight));
                                }
                                let accept_parts =
                                    self.constraint.runtime_parser_top_accept_parts(positive_label);
                                if !accept_parts.is_empty() {
                                    for accept_weight in accept_parts {
                                        if plan_ops.len() == MASK_SINGLE_PATH_DIRECT_MAX_PLAN_OPS {
                                            plans_complete = false;
                                            return false;
                                        }
                                        plan_ops.push(SinglePathDirectPlanOp::Merge(accept_weight));
                                    }
                                }
                            }
                            StackWalkEvent::Intersect(weight) => {
                                if plan_ops.len() == MASK_SINGLE_PATH_DIRECT_MAX_PLAN_OPS {
                                    plans_complete = false;
                                    return false;
                                }
                                plan_ops.push(SinglePathDirectPlanOp::Intersect(weight));
                            }
                        }
                        true
                    },
                );
                if !plans_complete {
                    break 'build_plans;
                }

                stack_plans[plan_index].ops_start = ops_start;
                stack_plans[plan_index].ops_end = plan_ops.len();
            }
        }
        let use_stack_plans = plans_complete && should_build_stack_plans;
        buf.fill(0);

        let precomputed = &self.constraint.weight_token_masks;
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
                            if weight.is_full() {
                                break;
                            }
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
                                    &self.constraint,
                                ) {
                                    break;
                                }
                                dense_is_seed = false;
                            } else if !Self::intersect_single_path_dense_with_weight_in_place(
                                &mut single_path_acc,
                                &mut single_path_aux,
                                internal_tsid,
                                weight,
                                &self.constraint,
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

                walk_single_stack::<true, _>(
                    self.constraint.runtime_parser_dwa_start_state(),
                    stack,
                    |state| self.constraint.runtime_parser_dwa_final_weight(state),
                    |state, parser| self.constraint.runtime_parser_dwa_transition(state, parser),
                    |event| {
                        match event {
                            StackWalkEvent::Final(final_weight) => {
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
                                // This stops only the current correlated
                                // parser/lexer path. Later paths may still
                                // contribute tokens, but this path cannot:
                                // its future masks are subsets of `dense`.
                                if final_weight.is_full() {
                                    return false;
                                }
                            }
                            StackWalkEvent::Top(parser_state) => {
                                let positive_label = encode_positive_label(parser_state);
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
                            StackWalkEvent::Intersect(weight) => {
                                if dense_is_seed {
                                    if !weight.is_full() {
                                        if !materialize_single_path_seed_intersection(
                                            seed_base,
                                            &mut single_path_acc,
                                            internal_tsid,
                                            weight,
                                            &self.constraint,
                                        ) {
                                            return false;
                                        }
                                        dense_is_seed = false;
                                    }
                                } else if !Self::intersect_single_path_dense_with_weight_in_place(
                                    &mut single_path_acc,
                                    &mut single_path_aux,
                                    internal_tsid,
                                    weight,
                                    &self.constraint,
                                ) {
                                    return false;
                                }
                            }
                        }
                        true
                    },
                );
            }
        }
        if !used_direct_final && !self.frontier_complete() {
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
        intersect_static_dense_with_weight(dense, aux, internal_tsid, weight,
            |tokens| constraint.runtime_token_set_dense_mask(tokens))
    }

    fn merge_single_path_final_weight_to_internal(
        &self,
        final_weight: RuntimeWeightRef<'_>,
        internal_tsid: u32,
        dense: &[u64],
        precomputed: &WeightTokenMaskCache,
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
            if self.constraint.try_or_materialized_final_token_set_to_buf(dense, token_set, buf) {
                *direct_buf_dirty = true;
                return true;
            }
        }

        if let (Some(buf), RuntimeTokenSetRef::PackedPool(tokens)) =
            (direct_buf, token_set)
            && self.constraint.or_packed_final_tokens_to_buf(dense, tokens, buf)
        {
            *direct_buf_dirty = true;
            return true;
        }

        DenseMaskAcc::or_dense_and_runtime_token_set_into(dense, token_set, precomputed, merged);
        false
    }

    fn terminals_disallowed_to_dense_acc(
        &self,
        terminals_disallowed: &TerminalsDisallowed,
        tokenizer_state: u32,
    ) -> Option<DenseMaskAcc> {
        let internal_tsids = self
            .constraint
            .runtime_internal_tsids_for_tokenizer_state(tokenizer_state)?;
        let base = &self.constraint.seed_universe_dense;
        if base.is_empty() || internal_tsids.is_empty() {
            return None;
        }
        if terminals_disallowed.is_empty() {
            return DenseMaskAcc::from_dense_arc_for_tsids(
                &internal_tsids,
                Arc::clone(base),
            );
        }

        let mut blocked_only = Vec::new();
        self.fill_blocked_seed_dense(terminals_disallowed, &mut blocked_only);

        if blocked_only.iter().all(|&word| word == 0) {
            return DenseMaskAcc::from_dense_arc_for_tsids(
                &internal_tsids,
                Arc::clone(base),
            );
        }

        let mut dense = base.to_vec();
        for (allowed_word, blocked_word) in dense.iter_mut().zip(blocked_only) {
            *allowed_word &= !blocked_word;
        }

        DenseMaskAcc::from_dense_arc_for_tsids(&internal_tsids, dense.into())
    }

    fn merge_final_weight_to_internal(
        &self,
        final_weight: RuntimeWeightRef<'_>,
        acc: &DenseMaskAcc,
        precomputed: &WeightTokenMaskCache,
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
                    let handled = self.constraint
                        .try_or_materialized_final_token_set_to_buf(dense, token_set, buf);
                    *direct_buf_dirty |= handled;
                    handled
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
        precomputed: &WeightTokenMaskCache,
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
        precomputed: &WeightTokenMaskCache,
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
        precomputed: &WeightTokenMaskCache,
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
        precomputed: &WeightTokenMaskCache,
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
                    &self.constraint,
                    queue,
                    start_dwa_state,
                    *parser_state,
                    popped,
                    precomputed,
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
        let Some(weights) = self.constraint.non_dwa_weights.materialized() else { return false; };
        let Some(parser_dwa) = self.constraint.parser_dwa() else { return false; };
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
        let precomputed = &self.constraint.weight_token_masks;
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
                        accumulator.intersect_with_runtime_weight_small_cached(
                            RuntimeWeightRef::Materialized(final_weight),
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
                            accumulator.intersect_with_runtime_weight_small_cached(
                                RuntimeWeightRef::Materialized(final_weight),
                                precomputed,
                                &mut seed_intersections,
                            ),
                        );
                    });
                }
                let positive_label = encode_positive_label(parser_state);
                if let Some(top_weight) = weights.top_accept(positive_label)
                {
                    dense.for_each_acc(|accumulator| {
                        merge_accepted(
                            &mut accepted,
                            accumulator.intersect_with_runtime_weight_small_cached(
                                RuntimeWeightRef::Materialized(top_weight),
                                precomputed,
                                &mut seed_intersections,
                            ),
                        );
                    });
                }
                if let Some(top_parts) = weights.parser_top_accept_parts.get(&positive_label).or_else(|| weights.parser_top_accept_parts.get(&DEFAULT_LABEL))
                {
                    for top_weight in top_parts {
                        dense.for_each_acc(|accumulator| {
                            merge_accepted(
                                &mut accepted,
                            accumulator.intersect_with_runtime_weight_small_cached(
                                RuntimeWeightRef::Materialized(top_weight),
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
                                accumulator.intersect_with_runtime_weight_small_cached(
                                RuntimeWeightRef::Materialized(top_weight),
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
                &self.constraint,
                &dag,
                precomputed,
                &mut indexed_runtime,
            );
            for (((root, target), transition_weight), _) in roots
                .iter()
                .copied()
                .zip(seed_targets)
                .zip(seed_weights)
                .zip(seed_gsses)
            {
                let result = evaluator.eval_upper(target, root);
                let result = result.and_then(|result| {
                    result.intersect_with_runtime_weight_small_cached(
                        RuntimeWeightRef::Materialized(&transition_weight),
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

        let precomputed = &self.constraint.weight_token_masks;
        let dense_words = self.constraint.internal_token_dense_words;
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
            &mut transition_intersection_cache,
            &mut queue,
            &mut merged,
            &mut direct_buf,
            &mut direct_buf_possible,
            &mut direct_buf_used,
            &mut direct_buf_dirty,
            &mut profile,
        );

        walk_static_dwa_gss_queue(
            &mut queue,
            &mut profile,
            |state| self.constraint.runtime_parser_dwa_final_weight(state),
            |state, parser| self.constraint.runtime_parser_dwa_transition(state, parser),
            |final_weight, gss| {
                direct_buf_used = true;
                direct_buf_possible &= self.merge_final_weight_for_gss(
                    final_weight,
                    gss,
                    precomputed,
                    &mut merged,
                    &mut direct_buf,
                    &mut direct_buf_dirty,
                );
            },
            |queue, target, weight, popped, profile| {
                enqueue_weighted_transition(
                    queue,
                    popped,
                    target,
                    weight,
                    precomputed,
                    &mut transition_intersection_cache,
                    profile,
                );
            },
        );

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

        let finalize_start = profile.as_ref().map(|_| Instant::now());

        let merged_has_leftovers = merged.iter().any(|&word| word != 0);
        let direct_finalized = direct_buf_used && direct_buf_possible && !merged_has_leftovers;
        let can_use_merged_cache = !direct_buf_dirty;
        let mut use_delta_seed = direct_finalized;
        let mut reuse_existing_cache_dense = false;
        if !direct_finalized && can_use_merged_cache {
            let cache = self.mask_cache.lock().unwrap();
            if let Some(cache_data) = cache.as_ref()
                && cache_data.mask.len() == buf.len()
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
                            profile.delta_copy_cost_words = self.constraint.body_mask_len() as u64;
                            profile.delta_used_seed = 1;
                        }
                    }
                    reuse_existing_cache_dense = true;
                    use_delta_seed = true;
                }
            if !use_delta_seed
                && let Some(cache_data) = cache.as_ref().filter(|c| c.merged_dense.len() == merged.len()) {
                    let scratch_cost = self.constraint.estimate_internal_dense_to_buf_cost(&merged);
                    let copy_cost_words = self.constraint.body_mask_len() as u64;
                    let mut added_bits = 0u64;
                    let mut removed_bits = 0u64;
                    let mut unchanged_words = 0u64;
                    let mut unchanged_bits = 0u64;
                    let (added_cost, removed_cost) = self.constraint
                        .estimate_internal_dense_delta_cost(&cache_data.merged_dense, &merged);
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
                        let removed = previous & !current;
                        if capture_delta_summary {
                            removed_bits += removed.count_ones() as u64;
                        }
                    }

                    let delta_cost = copy_cost_words + added_cost + removed_cost;
                    let delta_savings = scratch_cost.saturating_sub(delta_cost);

                    if delta_profile_enabled
                        && let Some(profile) = profile.as_mut() {
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
                self.constraint.token_projection.groups().len(),
                self.constraint.token_projection.groups().iter().map(Vec::len).sum::<usize>(),
                self.constraint.token_projection.groups().iter().map(Vec::len).max().unwrap_or(0),
                self.constraint.token_projection.groups().len() * self.constraint.body_mask_len(),
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
            if out.total_ns == 0
                && let Some(start) = total_start {
                    out.total_ns = elapsed_ns(start);
                }
            out
        });

        let mut scratch = self.mask_scratch.lock().unwrap();
        scratch.merged_dense = merged;
        scratch.chain_merged_dense.clear();

        returned_profile
    }

    /// Which tokens are allowed next, as a bitmask: token `i` is bit `i % 32` of word `i / 32`.
    pub fn mask(&self) -> Vec<u32> {
        let mut buf = vec![0u32; self.constraint.mask_bit_len().div_ceil(32)];
        self.fill_mask(&mut buf);
        buf
    }

    /// Like `mask()`, but writes the words into `buf`. Extra words are set to 0.
    ///
    /// # Panics
    ///
    /// Panics if `buf` is shorter than `self.mask_bit_len().div_ceil(32)`.
    pub fn fill_mask(&self, buf: &mut [u32]) {
        assert!(buf.len() >= self.constraint.mask_bit_len().div_ceil(32), "mask buffer has fewer words than required by mask_bit_len()");
        if self.terminated {
            buf.fill(0);
            return;
        }
        self.fill_body_mask(buf);
        // Also finalize cached and ordinary dynamic masks: internal body paths
        // may return before their composed-mask endpoint filter runs.
        self.enforce_empty_byte_token_domain(buf);
        if !self.constraint.end_tokens.is_empty() {
            let accepting = self.frontier_complete();
            for &id in self.constraint.end_tokens.iter() {
                let word = &mut buf[id as usize / 32];
                let bit = 1u32 << (id % 32);
                // End IDs are reserved generation controls, not byte pieces.
                if accepting { *word |= bit; } else { *word &= !bit; }
            }
        }
    }

    /// The mask's length in bits (the largest token ID plus one).
    pub fn mask_bit_len(&self) -> usize {
        self.constraint.mask_bit_len()
    }

    pub(crate) fn fill_body_mask(&self, buf: &mut [u32]) {
        let required = self.constraint.body_mask_len();
        assert!(buf.len() >= required, "mask buffer is smaller than constraint mask");
        let (mask, tail) = buf.split_at_mut(required);
        tail.fill(0);
        let overlay = self.constraint.static_dynamic_overlay.as_ref();
        let authoritative_segmented = self.constraint.uses_compact_segmented_parser_runtime();
        let authoritative_dynamic_direct = overlay.is_some_and(|overlay| {
            authoritative_segmented
                && overlay.segmented_parser_components.iter().all(|component| {
                    component.boundary.as_ref().is_some_and(|shard| {
                        matches!(
                            shard.backend,
                            crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
                        )
                    })
                })
        });
        let prepared_boundary_domain = crate::env::var_os("GLRMASK_DISABLE_BOUNDARY_VOCAB").is_none()
            && overlay.is_some_and(|overlay| overlay.segmented_parser_components.iter().any(|component| {
                component.boundary.as_ref().is_some_and(|shard| {
                    matches!(shard.backend, crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect)
                        && shard.candidate_tokens.is_some()
                })
            }));
        if authoritative_dynamic_direct && !prepared_boundary_domain {
            // DynamicDirect already computes the complete exact composed
            // language. Do not first construct component A masks only to OR the
            // same full language over them again. This is explicitly dynamic
            // evaluation (e.g. a differential reference side), so it runs
            // with a strict-static permit; static-side fallbacks below must
            // never be wrapped.
            crate::compiler::boundary_transfer::permit_strict_static_dynamic(|| {
                self.fill_mask_dynamic(mask)
            });
            self.clear_late_grammar_placeholder_mask(mask);
            return;
        }
        if authoritative_segmented {
            // The ordinary generation cache is valid for the complete
            // composed mask too. Commit retains it only when the exact
            // scoped parser/lexer state (including exclusions) is unchanged.
            if self.try_fill_mask_from_cache(mask) {
                self.clear_late_grammar_placeholder_mask(mask);
                return;
            }
            let filled = if authoritative_dynamic_direct {
                crate::compiler::boundary_transfer::permit_strict_static_dynamic(|| {
                    self.try_fill_mask_segmented(mask)
                })
            } else { self.try_fill_mask_segmented(mask) };
            if filled {
                self.update_control_special_token_mask(mask);
                self.clear_late_grammar_placeholder_mask(mask);
                self.store_mask_cache_reuse_dense(mask);
                return;
            }
            // Segmented projection is the common authoritative A/B path. A
            // recursive exceptional GSS must stay in the same scoped provider
            // coordinate, so dynamic evaluation selects a scoped provider for
            // the same shared walker rather than the transitional outer
            // parser/tokenizer. This fallback is a
            // hidden dynamic mask on a claimed static path: trap loudly under
            // the strict-static flag instead of silently contributing exact
            // dynamic admissions.
            if authoritative_dynamic_direct {
                crate::compiler::boundary_transfer::permit_strict_static_dynamic(|| self.fill_mask_dynamic(mask));
            } else {
                crate::compiler::boundary_transfer::strict_static_trap_dynamic(
                    "authoritative_segmented_projection_decline",
                );
                self.fill_mask_dynamic(mask);
            }
            self.clear_late_grammar_placeholder_mask(mask);
            return;
        }
        if self.constraint.uses_dynamic_runtime() {
            // Diagnostic escape hatch used by dynamic-mask performance tests:
            // bypass both the state-local last-mask cache and the lower-level
            // dynamic memo so repeated calls expose recomputation cost.
            let disable_dynamic_mask_cache = !super::dynamic_mask::dynamic_mask_cache_enabled();
            let profile_dynamic_cache = crate::env::var("GLRMASK_PROFILE_DYNAMIC_STATE_CACHE_GENERATION")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                == Some(self.generation);
            let local_started = profile_dynamic_cache.then(Instant::now);
            let local_hit = !disable_dynamic_mask_cache && self.try_fill_mask_from_cache(mask);
            let local_us = local_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1e6);
            if !local_hit {
                let dynamic_started = profile_dynamic_cache.then(Instant::now);
                self.fill_mask_dynamic(mask);
                let dynamic_us = dynamic_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1e6);
                let store_started = profile_dynamic_cache.then(Instant::now);
                if !disable_dynamic_mask_cache {
                    self.store_mask_cache_reuse_dense(mask);
                }
                let store_us = store_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1e6);
                if profile_dynamic_cache {
                    eprintln!(
                        "[glrmask/profile][dynamic_state_cache] generation={} local_hit=false local_us={:.1} dynamic_us={:.1} store_us={:.1}",
                        self.generation,
                        local_us,
                        dynamic_us,
                        store_us,
                    );
                }
            } else if profile_dynamic_cache {
                eprintln!(
                    "[glrmask/profile][dynamic_state_cache] generation={} local_hit=true local_us={:.1}",
                    self.generation,
                    local_us,
                );
            }
            let clear_started = profile_dynamic_cache.then(Instant::now);
            self.clear_late_grammar_placeholder_mask(mask);
            if let Some(clear_started) = clear_started {
                eprintln!(
                    "[glrmask/profile][dynamic_state_cache] generation={} clear_us={:.1}",
                    self.generation,
                    clear_started.elapsed().as_secs_f64() * 1e6,
                );
            }
            return;
        }
        let cache_hit = self.try_fill_mask_from_cache(mask);
        if !cache_hit {
            self.fill_mask_uncached(mask);
            self.update_control_special_token_mask(mask);
            // Template exact-token admissions are overlaid on the byte DWA.
            // Replace any pre-overlay cached mask before it can be reused.
            if self.constraint.parser_has_controls()
                || self.constraint.uses_compact_segmented_parser_runtime()
                || (self.constraint.has_template_parser() && !self.constraint.special_token_terminals.is_empty())
            {
                self.store_mask_cache_reuse_dense(mask);
            }
        }
        if crate::env::var_os("GLRMASK_EXPERIMENT_STATIC_DYNAMIC_OVERLAY").is_some() {
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
        if self.terminated || !self.constraint.end_tokens.is_empty() {
            let started = Instant::now();
            self.fill_mask(buf);
            return MaskProfile { total_ns: elapsed_ns(started), ..MaskProfile::default() };
        }
        let required = self.constraint.body_mask_len();
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
        if self.constraint.uses_compact_segmented_parser_runtime() {
            // The recursive provider-native mask has different profiling
            // phases from the historical flattened/static evaluator. Preserve
            // exact semantics first; detailed segmented profiling can be added
            // independently without routing a recursive state through the
            // transitional outer parser/tokenizer.
            self.fill_mask(buf);
            return MaskProfile {
                total_ns: elapsed_ns(total_start),
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
        if self.constraint.parser_has_controls()
            || self.constraint.uses_compact_segmented_parser_runtime()
            || (self.constraint.has_template_parser() && !self.constraint.special_token_terminals.is_empty())
        {
            self.store_mask_cache_reuse_dense(buf);
        }
        profile
    }

}

#[cfg(test)]
mod boundary_dag_exact_tests {
    //! Exactness tests for the cap-free GSS x boundary-DWA product evaluator.
    //!
    //! Each test builds an adversarial ambiguous GSS with >128 distinct paths
    //! over shared tails (which makes the old
    //! `for_each_stack_top_first_bounded(128, _)` evaluator decline) and
    //! compares the memoized DAG evaluator against a literal per-path
    //! reference walk written independently in test code.

    use super::BoundaryWeightDagEvaluator;
    use crate::automata::weighted_u32::dwa::{DWA, DWAState};
    use crate::compiler::glr::accumulator::TerminalsDisallowed;
    use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_positive_label};
    use crate::compiler::glr::parser::ParserGSS;
    use crate::ds::weight::Weight;
    use range_set_blaze::RangeSetBlaze;

    const TOKEN_COUNT: usize = 6;
    const ALL_TOKENS: u64 = (1 << TOKEN_COUNT) - 1;

    /// 256 distinct stacks sharing a common bottom tail, in bottom-to-top order.
    fn adversarial_stacks() -> Vec<(Vec<u32>, TerminalsDisallowed)> {
        (0..256u32)
            .map(|bits| {
                let mut stack = vec![900, 901, 902];
                for level in 0..8u32 {
                    stack.push(2 * level + ((bits >> level) & 1));
                }
                (stack, TerminalsDisallowed::new())
            })
            .collect()
    }

    fn adversarial_gss() -> ParserGSS {
        ParserGSS::from_stacks(&adversarial_stacks())
    }

    /// Positive edges, explicit empty guard, DEFAULT, prefix finals and dead end.
    /// Coordinate 1 rotates each token set and kills the +0 transition.
    fn adversarial_dwa(rows: u32) -> DWA {
        let weight = |mask: u64, kill_second_row: bool| {
            Weight::from_per_tsid_token_sets((0..rows).filter_map(|tsid| {
                let bits = if tsid == 0 {
                    mask
                } else if kill_second_row {
                    0
                } else {
                    ((mask << 1) | (mask >> 5)) & ALL_TOKENS
                };
                (bits != 0).then(|| {
                    let tokens = (0..TOKEN_COUNT as u32)
                        .filter(|&token| bits & (1 << token) != 0)
                        .collect::<RangeSetBlaze<u32>>();
                    (tsid, tokens)
                })
            }))
        };
        let states = vec![
            DWAState {
                transitions: [
                    (encode_positive_label(0), (1, weight(0b000011, true))),
                    (encode_positive_label(14), (2, Weight::empty())),
                    (DEFAULT_LABEL, (2, Weight::all())),
                ].into_iter().collect(),
                final_weight: Some(weight(0b000101, false)),
            },
            DWAState {
                transitions: [(encode_positive_label(2), (2, weight(0b001010, false)))]
                    .into_iter().collect(),
                final_weight: Some(weight(0b000010, false)),
            },
            DWAState {
                transitions: Default::default(),
                final_weight: Some(weight(0b001000, false)),
            },
        ];
        DWA::from_parts(states, 0)
    }

    fn weight_bits(weight: &Weight, tsid: u32) -> u64 {
        if weight.is_full() {
            ALL_TOKENS
        } else {
            weight.token_set_for_tsid_ref(tsid).map_or(0, |tokens| {
                (0..TOKEN_COUNT as u32)
                    .filter(|&token| tokens.contains(token))
                    .fold(0, |bits, token| bits | (1 << token))
            })
        }
    }

    /// Literal per-path oracle with independent bitwise weight operations.
    fn reference_mask_for_stack(dwa: &DWA, tsid: u32, top_first: &[u32]) -> u64 {
        let mut state_id = dwa.start_state();
        let mut path_mask = ALL_TOKENS;
        let mut accepted = 0;
        let accumulate = |state_id: u32, path_mask: u64, accepted: &mut u64| {
            if let Some(final_weight) = dwa.states()[state_id as usize].final_weight.as_ref() {
                *accepted |= path_mask & weight_bits(final_weight, tsid);
            }
        };
        accumulate(state_id, path_mask, &mut accepted);
        for &parser_state in top_first {
            let transitions = &dwa.states()[state_id as usize].transitions;
            let Some((target, weight)) = transitions.get(&encode_positive_label(parser_state))
                .or_else(|| transitions.get(&DEFAULT_LABEL))
            else {
                break;
            };
            path_mask &= weight_bits(weight, tsid);
            if path_mask == 0 {
                break;
            }
            state_id = *target;
            accumulate(state_id, path_mask, &mut accepted);
        }
        accepted
    }

    fn reference_union_all_paths(
        dwa: &DWA,
        tsid: u32,
        gss: &ParserGSS,
        top_live: &dyn Fn(Option<u32>) -> bool,
    ) -> (u64, usize) {
        let mut union = 0;
        let mut count = 0;
        let complete = gss.for_each_stack_top_first_bounded(100_000, |top_first, _| {
            if top_live(top_first.first().copied()) {
                count += 1;
                union |= reference_mask_for_stack(dwa, tsid, top_first);
            }
        });
        assert!(complete, "reference enumeration must complete");
        (union, count)
    }

    #[test]
    fn weight_dag_evaluator_matches_literal_paths_past_128() {
        let gss = adversarial_gss();
        assert!(!gss.is_single_path());
        assert_eq!(gss.path_count_at_most(129), 129);
        assert!(!gss.for_each_stack_top_first_bounded(128, |_, _| {}));
        let dwa = adversarial_dwa(1);
        let dag = gss.indexed_dag();
        let top_live = |_: Option<u32>| true;
        let (expected, path_count) = reference_union_all_paths(&dwa, 0, &gss, &top_live);
        assert_eq!(path_count, 256);
        let mut evaluator = BoundaryWeightDagEvaluator::new(&dwa, &dag);
        let groups = evaluator.eval_root(&top_live);
        let mut actual = 0;
        for (group, weight) in &groups {
            assert!(evaluator.group_accumulator(*group).is_some());
            actual |= weight_bits(weight, 0);
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn weight_dag_evaluator_respects_top_filter_and_default() {
        let gss = adversarial_gss();
        let dwa = adversarial_dwa(1);
        let dag = gss.indexed_dag();
        // Top 14 has an explicit empty guard; top 15 uses DEFAULT.
        for (top, expected) in [(14, 0b000101), (15, 0b001101)] {
            let top_live = |value: Option<u32>| value == Some(top);
            let (literal, path_count) = reference_union_all_paths(&dwa, 0, &gss, &top_live);
            assert_eq!(path_count, 128);
            assert_eq!(literal, expected);
            let mut evaluator = BoundaryWeightDagEvaluator::new(&dwa, &dag);
            let groups = evaluator.eval_root(&top_live);
            let actual = groups.values().fold(0, |bits, weight| bits | weight_bits(weight, 0));
            assert_eq!(actual, expected);
        }
        let positive = ParserGSS::from_stacks(&[(vec![2, 0], TerminalsDisallowed::new())]);
        let dag = positive.indexed_dag();
        let mut evaluator = BoundaryWeightDagEvaluator::new(&dwa, &dag);
        let actual = evaluator.eval_root(&|_| true).values()
            .fold(0, |bits, weight| bits | weight_bits(weight, 0));
        assert_eq!(actual, 0b000111);
    }

    #[test]
    fn weight_dag_evaluator_keeps_per_path_acc_groups() {
        let mut stacks = adversarial_stacks();
        for (index, (_, acc)) in stacks.iter_mut().enumerate() {
            if index % 2 == 0 {
                *acc = acc.clone().with_insert(7, 9);
            }
        }
        let gss = ParserGSS::from_stacks(&stacks);
        assert!(!gss.is_single_path());
        let dwa = adversarial_dwa(1);
        let dag = gss.indexed_dag();
        let top_live = |_: Option<u32>| true;
        let mut evaluator = BoundaryWeightDagEvaluator::new(&dwa, &dag);
        let groups = evaluator.eval_root(&top_live);
        assert!(groups.len() >= 2, "distinct accumulator groups collapsed");
        let (expected, _) = reference_union_all_paths(&dwa, 0, &gss, &top_live);
        let actual = groups.values().fold(0, |bits, weight| bits | weight_bits(weight, 0));
        assert_eq!(actual, expected);
    }

    #[test]
    fn weight_dag_evaluator_empty_stack_filtering() {
        let gss = ParserGSS::from_stacks(&[(Vec::new(), TerminalsDisallowed::new())]);
        let dwa = adversarial_dwa(1);
        let dag = gss.indexed_dag();
        let mut evaluator = BoundaryWeightDagEvaluator::new(&dwa, &dag);
        let with_empty = evaluator.eval_root(&|_| true);
        assert_eq!(with_empty.len(), 1);
        assert_eq!(weight_bits(with_empty.values().next().unwrap(), 0), 0b000101);
        let mut evaluator = BoundaryWeightDagEvaluator::new(&dwa, &dag);
        assert!(evaluator.eval_root(&|top| top.is_some()).is_empty());
    }

    #[test]
    fn shared_boundary_queue_preserves_correlated_groups_and_coordinates() {
        let mut stacks = adversarial_stacks();
        for (index, (_, acc)) in stacks.iter_mut().enumerate() {
            if index % 2 == 0 {
                *acc = acc.clone().with_insert(7, 9);
            }
        }
        stacks.push((vec![2, 0], TerminalsDisallowed::new()));
        stacks.push((Vec::new(), TerminalsDisallowed::new()));
        let dwa = adversarial_dwa(2);
        for gss in [ParserGSS::from_stacks(&stacks), ParserGSS::empty()] {
            for filter in 0..3 {
                let top_live = |top: Option<u32>| match filter {
                    0 => true,
                    1 => top.is_some_and(|value| value != 14),
                    _ => false,
                };
                for (language, accumulator) in gss.partition_by_accumulator() {
                    let original = language.apply(|_| accumulator.clone());
                    let dag = original.indexed_dag();
                    let mut evaluator = BoundaryWeightDagEvaluator::new(&dwa, &dag);
                    let groups = evaluator.eval_root(&top_live);
                    for coordinates in [&[0, 1][..], &[1][..], &[0][..]] {
                        let actual = super::boundary_weight_mask_via_shared_queue(
                            &dwa, coordinates, TOKEN_COUNT, &language, &top_live,
                        );
                        for tsid in 0..2 {
                            let expected = if coordinates.contains(&tsid) {
                                let (literal, _) =
                                    reference_union_all_paths(&dwa, tsid, &original, &top_live);
                                let dag_bits = groups.values().fold(0, |bits, weight| {
                                    bits | weight_bits(weight, tsid)
                                });
                                assert_eq!(literal, dag_bits);
                                literal
                            } else {
                                0
                            };
                            let observed = actual.as_ref().map_or(0, |allowed| {
                                allowed.0.iter()
                                    .filter(|(coordinate, _)| *coordinate == tsid)
                                    .fold(0, |bits, (_, words)| bits | words[0])
                            });
                            assert_eq!(observed, expected,
                                "coordinate={tsid} selected={coordinates:?} group={accumulator:?} filter={filter}");
                        }
                    }
                }
            }
        }
    }
}

/// Clear a sorted, unique empty-spelling inventory outside the live exact
/// intervals. Allowed spans are skipped by binary search, including overlaps;
/// this never introduces a bit that the body mask did not already contain.
fn retain_live_empty_byte_ranges(
    mask: &mut [u32],
    empty_ids: &[u32],
    live_ranges: &mut [(u32, u32)],
) {
    fn clear_ids(mask: &mut [u32], ids: &[u32]) {
        let Some((&start, &end)) = ids.first().zip(ids.last()) else { return; };
        // Sorted unique IDs with this cardinality constitute a complete span.
        // Clear dense uncovered spans wordwise rather than once per model ID.
        if u64::from(end) - u64::from(start) + 1 == ids.len() as u64 {
            let first = start as usize / 32;
            if first >= mask.len() { return; }
            let last = (end as usize / 32).min(mask.len() - 1);
            let first_bits = u32::MAX << (start % 32);
            let last_bits = if last == end as usize / 32 {
                u32::MAX >> (31 - end % 32)
            } else { u32::MAX };
            if first == last {
                mask[first] &= !(first_bits & last_bits);
            } else {
                mask[first] &= !first_bits;
                mask[first + 1..last].fill(0);
                mask[last] &= !last_bits;
            }
        } else {
            for &id in ids {
                if let Some(word) = mask.get_mut(id as usize / 32) {
                    *word &= !(1u32 << (id % 32));
                }
            }
        }
    }
    live_ranges.sort_unstable();
    let mut remaining = empty_ids;
    for &(start, end) in live_ranges.iter() {
        let before = remaining.partition_point(|&id| id < start);
        clear_ids(mask, &remaining[..before]);
        remaining = &remaining[before..];
        let through = remaining.partition_point(|&id| id <= end);
        remaining = &remaining[through..];
        if remaining.is_empty() { return; }
    }
    clear_ids(mask, remaining);
}

#[cfg(test)]
mod empty_byte_range_filter_tests {
    use super::retain_live_empty_byte_ranges;

    #[test]
    fn interval_filter_matches_independent_pointwise_policy() {
        let domains = [
            (0..128).collect::<Vec<u32>>(),
            (0..128).filter(|id| id % 3 == 0).collect(),
            (17..96).collect(),
            vec![0, 31, 32, 63, 64, 95, 127, u32::MAX],
        ];
        let selections = [vec![], vec![(0, 127)], vec![(31, 63)],
            vec![(90, 127), (0, 32), (31, 96)],
            vec![(u32::MAX - 1, u32::MAX)], vec![(60, 95), (0, 1)]];
        for empty in &domains {
            for ranges in &selections {
                for words in [0, 1, 4] {
                    for seed in [0, u32::MAX, 0xa5a5_5a5a] {
                        let mut actual = vec![seed; words];
                        let mut expected = actual.clone();
                        for &id in empty {
                            if !ranges.iter().any(|&(start, end)| start <= id && id <= end)
                                && let Some(word) = expected.get_mut(id as usize / 32) {
                                    *word &= !(1u32 << (id % 32));
                                }
                        }
                        retain_live_empty_byte_ranges(&mut actual, empty, &mut ranges.clone());
                        assert_eq!(actual, expected, "domain={empty:?}, ranges={ranges:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn full_and_nearly_full_128k_empty_domains_skip_allowed_spans() {
        let ids = (0..131_072).collect::<Vec<u32>>();
        let mut mask = vec![u32::MAX; 4096];
        retain_live_empty_byte_ranges(&mut mask, &ids, &mut [(0, 131_071)]);
        assert!(mask.iter().all(|&word| word == u32::MAX));
        retain_live_empty_byte_ranges(&mut mask, &ids, &mut [(65_537, 131_071), (0, 65_535)]);
        assert_eq!(mask.iter().map(|word| word.count_ones()).sum::<u32>(), 131_071);
        assert_eq!(mask[65_536 / 32] & 1, 0);
    }
}
