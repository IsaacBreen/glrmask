use crate::automata::lexer::Lexer;
use std::sync::{Arc, Mutex};

use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::{
    ParserGSS, stacks_finished, stacks_finished_control_closed,
};
use crate::ds::bitset::BitSet;
use crate::ds::leveled_gss::GssSemanticKeyInterner;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::constraint::Constraint;

pub(crate) const LINEAR_STACK_RESERVE: usize = 64;
pub(crate) const INLINE_PARSER_STATE_CAPACITY: usize = 64;

/// Parser paths grouped by tokenizer state, stored inline for the common small
/// frontier. Entries remain sorted by tokenizer-state ID. The bounded flat
/// runtime may temporarily retain multiple single-stack entries with the same
/// key instead of materializing them into one allocating GSS; general paths
/// normalize equal-key entries when they need map semantics.
#[derive(Clone, Default, PartialEq, Eq)]
pub(crate) struct ParserStateMap {
    pub(crate) entries: SmallVec<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>,
}

impl std::fmt::Debug for ParserStateMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.entries.iter().map(|(k, v)| (k, v))).finish()
    }
}

impl ParserStateMap {
    pub(crate) fn singleton(key: u32, value: ParserGSS) -> Self {
        let mut entries = SmallVec::new();
        entries.push((key, value));
        Self { entries }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize { self.entries.len() }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool { self.entries.is_empty() }

    #[inline]
    pub(crate) fn clear(&mut self) { self.entries.clear(); }

    #[inline]
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&u32, &ParserGSS)> {
        self.entries.iter().map(|(key, value)| (key, value))
    }

    #[inline]
    pub(crate) fn values(&self) -> impl Iterator<Item = &ParserGSS> {
        self.entries.iter().map(|(_, value)| value)
    }

    #[inline]
    pub(crate) fn values_mut(&mut self) -> impl Iterator<Item = &mut ParserGSS> {
        self.entries.iter_mut().map(|(_, value)| value)
    }

    #[inline]
    pub(crate) fn keys(&self) -> impl Iterator<Item = &u32> {
        self.entries.iter().map(|(key, _)| key)
    }

    fn equal_range(&self, key: u32) -> std::ops::Range<usize> {
        let start = self.entries.partition_point(|(entry_key, _)| *entry_key < key);
        let end = self.entries.partition_point(|(entry_key, _)| *entry_key <= key);
        start..end
    }

    /// Return the sole entry for `key`. A duplicate-key flat frontier is not a
    /// map value and deliberately returns `None` so map-only optimizations fall
    /// back rather than observing just one parser alternative.
    pub(crate) fn get(&self, key: &u32) -> Option<&ParserGSS> {
        let range = self.equal_range(*key);
        (range.len() == 1).then(|| &self.entries[range.start].1)
    }

    pub(crate) fn get_mut(&mut self, key: &u32) -> Option<&mut ParserGSS> {
        let range = self.equal_range(*key);
        if range.len() != 1 {
            return None;
        }
        Some(&mut self.entries[range.start].1)
    }

    pub(crate) fn values_for_key(&self, key: u32) -> impl Iterator<Item = &ParserGSS> {
        let range = self.equal_range(key);
        self.entries[range].iter().map(|(_, value)| value)
    }

    pub(crate) fn has_duplicate_keys(&self) -> bool {
        self.entries.windows(2).any(|pair| pair[0].0 == pair[1].0)
    }

    /// Materialize duplicate-key flat alternatives into one GSS per tokenizer
    /// state. This is the explicit boundary between the bounded allocation-free
    /// representation and general map/GSS algorithms.
    pub(crate) fn normalize_duplicate_keys(&mut self) {
        if !self.has_duplicate_keys() {
            return;
        }
        let old = std::mem::take(&mut self.entries);
        for (key, value) in old {
            self.merge_insert(key, value);
        }
    }

    /// Map-style insertion. Any bounded flat alternatives already stored under
    /// this key are merged into the returned old value before replacement.
    pub(crate) fn insert(&mut self, key: u32, value: ParserGSS) -> Option<ParserGSS> {
        let range = self.equal_range(key);
        if range.is_empty() {
            self.entries.insert(range.start, (key, value));
            return None;
        }

        let mut old = std::mem::replace(&mut self.entries[range.start].1, value);
        for _ in range.start + 1..range.end {
            let (_, duplicate) = self.entries.remove(range.start + 1);
            old = old.merge(&duplicate);
        }
        Some(old)
    }

    pub(crate) fn pop_first(&mut self) -> Option<(u32, ParserGSS)> {
        (!self.entries.is_empty()).then(|| self.entries.remove(0))
    }

    pub(crate) fn merge_insert(&mut self, key: u32, mut value: ParserGSS) {
        let range = self.equal_range(key);
        if range.is_empty() {
            self.entries.insert(range.start, (key, value));
            return;
        }
        for _ in range.clone() {
            let (_, existing) = self.entries.remove(range.start);
            value = value.merge(&existing);
        }
        self.entries.insert(range.start, (key, value));
    }

    /// Insert one correlated flat alternative without merging equal keys.
    ///
    /// This is the relation-preserving counterpart to `merge_insert`: callers
    /// use it when the same tokenizer state is paired with distinct parser
    /// languages that must remain separate for a later lexer transition.
    pub(crate) fn insert_flat_alternative(&mut self, key: u32, value: ParserGSS) {
        let index = self.entries.partition_point(|(entry_key, _)| *entry_key <= key);
        self.entries.insert(index, (key, value));
    }

    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&u32, &mut ParserGSS) -> bool) {
        let mut index = 0;
        while index < self.entries.len() {
            let should_keep = {
                let (key, value) = &mut self.entries[index];
                keep(key, value)
            };
            if should_keep {
                index += 1;
            } else {
                self.entries.remove(index);
            }
        }
    }

    /// Replace the sole tokenizer-state key without moving or reallocating its GSS.
    pub(crate) fn replace_single_key(&mut self, key: u32) -> bool {
        self.replace_single_keys(std::slice::from_ref(&key))
    }

    /// Associate the sole existing GSS with a small sorted set of tokenizer
    /// states. Arc clones are allocation-free and eight entries fit inline.
    pub(crate) fn replace_single_keys(&mut self, keys: &[u32]) -> bool {
        if self.entries.len() != 1 || keys.is_empty() || keys.len() > INLINE_PARSER_STATE_CAPACITY {
            return false;
        }
        if keys.windows(2).any(|pair| pair[0] >= pair[1]) {
            return false;
        }
        let (_, gss) = self.entries.pop().unwrap();
        for &key in &keys[..keys.len() - 1] {
            self.entries.push((key, gss.clone()));
        }
        self.entries.push((*keys.last().unwrap(), gss));
        true
    }


    /// Replace a small set of structurally identical parser states with one
    /// shared GSS under a new sorted set of tokenizer-state keys.
    pub(crate) fn replace_uniform_gss_keys(&mut self, keys: &[u32]) -> bool {
        if self.entries.is_empty()
            || keys.is_empty()
            || keys.len() > INLINE_PARSER_STATE_CAPACITY
            || keys.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return false;
        }
        let representative = &self.entries[0].1;
        if self
            .entries
            .iter()
            .skip(1)
            .any(|(_, gss)| !representative.ptr_eq(gss) && representative != gss)
        {
            return false;
        }

        let old = std::mem::take(&mut self.entries);
        let (_, gss) = old.into_iter().next().expect("nonempty state checked");
        for &key in &keys[..keys.len() - 1] {
            self.entries.push((key, gss.clone()));
        }
        self.entries.push((*keys.last().unwrap(), gss));
        true
    }
}

impl FromIterator<(u32, ParserGSS)> for ParserStateMap {
    fn from_iter<T: IntoIterator<Item = (u32, ParserGSS)>>(iter: T) -> Self {
        let mut result = Self::default();
        for (key, value) in iter {
            if let Some(existing) = result.insert(key, value) {
                let current = result.get_mut(&key).unwrap();
                *current = existing.merge(current);
            }
        }
        result
    }
}

impl IntoIterator for ParserStateMap {
    type Item = (u32, ParserGSS);
    type IntoIter = smallvec::IntoIter<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>;

    fn into_iter(self) -> Self::IntoIter { self.entries.into_iter() }
}

impl<'a> IntoIterator for &'a ParserStateMap {
    type Item = (&'a u32, &'a ParserGSS);
    type IntoIter = std::iter::Map<
        std::slice::Iter<'a, (u32, ParserGSS)>,
        fn(&(u32, ParserGSS)) -> (&u32, &ParserGSS),
    >;

    fn into_iter(self) -> Self::IntoIter {
        fn pair_refs(pair: &(u32, ParserGSS)) -> (&u32, &ParserGSS) { (&pair.0, &pair.1) }
        self.entries.iter().map(pair_refs)
    }
}


/// Cached fill_mask result, keyed on generation counter.
pub(crate) struct MaskCacheData {
    pub generation: u64,
    pub mask: Vec<u32>,
    /// The merged internal token dense bitmap used to compute this mask.
    /// Enables incremental updates when the state changes slightly.
    pub merged_dense: Vec<u64>,
}

#[derive(Default)]
pub(crate) struct MaskScratch {
    pub merged_dense: Vec<u64>,
    pub chain_merged_dense: Vec<u64>,
    pub output_buf: Vec<u32>,
    /// Reused by the single-path direct mask kernel when constructing the
    /// terminal-disallowed seed and non-precomputed intersections.
    pub single_path_aux_dense: Vec<u64>,
    /// Mutable single-TSID accumulator for the allocation-free direct kernel.
    pub single_path_acc_dense: Vec<u64>,
    /// Bounded live-state cache for indexed-DAG masking. Cached products are
    /// exact pure-function results keyed by retained immutable GSS nodes.
    pub indexed_dag_mask: crate::runtime::mask::IndexedDagMaskRuntime,
    /// Reusable scratch for retained segmented component evaluators. These are
    /// allocated when the outer state is created, not in the mask hot path.
    /// `Arc<Mutex<_>>` lets a short-lived projected `ConstraintState` borrow the
    /// component scratch without moving it out of the parent scratch object.
    pub segmented_component_scratch: Vec<Arc<Mutex<MaskScratch>>>,
}

impl MaskScratch {
    pub(crate) fn for_constraint(_constraint: &Constraint) -> Self {
        Self::default()
    }
}

impl Clone for CommitBuffers {
    fn clone(&self) -> Self {
        // Don't clone scratch buffers — start fresh
        Self::default()
    }
}

impl CommitBuffers {
    pub fn clear_all(&mut self) {
        self.advance_result_cache.clear();
        self.semantic_frontier_keys.clear();
        self.pending_state.clear();
        self.seen_matches.clear();
        self.terminal_result_cache.clear();
        self.small_exec_result.end_state.clear();
        self.small_exec_result.matches.clear();
        self.reusable_tokenizer_exec.states.clear();
        self.reusable_tokenizer_exec.matches.clear();
        for bucket in &mut self.processing_queue {
            bucket.clear();
        }
    }

    pub(crate) fn reset_all(&mut self) {
        self.clear_all();
        self.admission_cache.clear();
        self.parser_relative_mask_eq_cache.clear();
        self.template_advance_runtime.reset_all();
    }
}

pub struct ConstraintState {
    pub(crate) terminated: bool,
    pub(crate) markers: Option<Box<crate::runtime::markers::MarkerState>>,
    pub(crate) constraint: Constraint,
    pub(crate) state: ParserStateMap,
    pub(crate) buffers: CommitBuffers,
    /// Monotonically increasing counter, bumped on every commit.
    /// Used for cheap cache invalidation in fill_mask.
    pub(crate) generation: u64,
    /// Cached fill_mask result: returned directly when state matches cached snapshot.
    /// Not cloned — clone starts with empty cache.
    pub(crate) mask_cache: Mutex<Option<MaskCacheData>>,
    /// Reusable scratch buffers for fill_mask to avoid per-call allocation.
    pub(crate) mask_scratch: Arc<Mutex<MaskScratch>>,
}

impl Clone for ConstraintState {

    fn clone(&self) -> Self {
        ConstraintState {
            terminated: self.terminated,
            markers: self.markers.clone(),
            constraint: self.constraint.clone(),
            state: self.state.clone(),
            buffers: CommitBuffers::for_constraint(&self.constraint),
            generation: self.generation,
            mask_cache: Mutex::new(None),
            mask_scratch: Arc::new(Mutex::new(MaskScratch::for_constraint(&self.constraint))),
        }
    }
}

impl std::fmt::Debug for ConstraintState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstraintState")
            .field("terminated", &self.terminated)
            .field("state_len", &self.state.len())
            .field("mask_cached", &self.mask_cache.lock().unwrap().is_some())
            .finish()
    }
}

impl ConstraintState {
    pub(crate) fn copy_snapshot_from(&mut self, source: &ConstraintState) {
        assert!(Arc::ptr_eq(&self.constraint.data, &source.constraint.data));
        self.terminated = source.terminated;
        self.markers = source.markers.clone();
        self.state = source.state.clone();
        self.buffers = CommitBuffers::for_constraint(&self.constraint);
        self.generation = source.generation;
        self.mask_cache = Mutex::new(None);
        self.mask_scratch = Arc::new(Mutex::new(MaskScratch::for_constraint(&self.constraint)));
    }

    pub(crate) fn reserve_linear_stack_hot_path(&mut self) {
        // Runtime frontiers are commonly a small set of correlated tokenizer
        // continuations, each carrying one concrete parser stack. Detach and
        // reserve every such GSS before timed decoding so later branch updates
        // can reuse their storage in place. Non-linear entries simply decline.
        for gss in self.state.values_mut() {
            let _ = gss.reserve_single_segment_capacity(LINEAR_STACK_RESERVE);
        }
    }


    /// The bytes that must come next.
    pub fn forced_bytes(&self) -> Vec<u8> {
        crate::runtime::forced_bytes_for_states(std::slice::from_ref(self))
    }

    /// Whether the output so far could end here.
    pub fn is_accepting(&self) -> bool {
        if !self.constraint.end_tokens.is_empty() {
            return self.terminated;
        }
        frontier_is_accepting(&self.constraint, &self.state, self.terminated)
    }

    /// Whether the parser frontier is a complete grammar match, ignoring any
    /// end-token policy.
    pub(crate) fn frontier_complete(&self) -> bool {
        frontier_is_accepting(&self.constraint, &self.state, false)
    }

    pub(crate) fn parser_root_count(&self) -> usize {
        self.state.values().map(|gss| gss.peek_values().len()).sum()
    }

    pub(crate) fn parser_path_count(&self, limit: usize) -> usize {
        self.state.values().map(|gss| gss.path_count_at_most(limit)).sum::<usize>().min(limit)
    }

    pub(crate) fn has_parser_ambiguity(&self) -> bool {
        self.parser_path_count(2) > 1
    }

    /// Return all flattened parser stacks for debugging.
    /// Each entry is (tokenizer_state, Vec<(stack_of_parser_states, disallowed_terminals)>).
    pub(crate) fn debug_parser_stacks(&self) -> Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)> {
        let mut grouped = std::collections::BTreeMap::<
            u32,
            Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>,
        >::new();
        for (&ts, gss) in self.state.iter() {
            let stacks = gss
                .to_stacks(4_096)
                .expect("stack enumeration exceeded explicit limit");
            let out = grouped.entry(ts).or_default();
            out.extend(stacks.into_iter().map(|(stack, acc)| {
                let disallowed = acc
                    .iter()
                    .map(|(key, values)| (*key, values.iter().copied().collect()))
                    .collect();
                (stack, disallowed)
            }));
        }
        for stacks in grouped.values_mut() {
            stacks.sort();
            stacks.dedup();
        }
        grouped.into_iter().collect()
    }

    /// Whether nothing more can be committed.
    #[must_use]
    pub fn is_terminated(&self) -> bool {
        if self.terminated {
            return true;
        }
        if self.state.is_empty() {
            return true;
        }
        // A complete body under an end-token root can still commit its end token.
        if !self.constraint.end_tokens.is_empty() && self.frontier_complete() {
            return false;
        }
        !self.state.iter().any(|(&tokenizer_state, gss)| {
            super::commit::entry_can_continue(&self.constraint, tokenizer_state, gss)
        })
    }

    #[cfg(any(test,feature="internal-api"))]
    #[doc(hidden)]
    pub fn forced_canonical_reference(&self)->Vec<u32> {
        super::commit::forcing_reference::force(std::slice::from_ref(self),None)
    }
    #[cfg(any(test,feature="internal-api"))]
    #[doc(hidden)]
    pub fn forced_token_ids_reference(&self)->Vec<u32> {
        super::commit::forcing_reference::force_token_ids(std::slice::from_ref(self),None)
    }

    #[cfg(any(test, feature="internal-api"))]
    pub(super) fn mask_for_forced(&self, dynamic: bool) -> Vec<u32> {
        if !self.constraint.end_tokens.is_empty() { return self.mask(); }
        if dynamic {
            let mut mask = vec![0u32; self.constraint.body_mask_len()];
            self.fill_mask_dynamic(&mut mask);
            mask
        } else { self.mask() }
    }

}

/// Whether the parser frontier completes the grammar, ignoring end-token policy.
pub(super) fn frontier_is_accepting(constraint: &Constraint, state: &ParserStateMap, terminated: bool) -> bool {
        if terminated { return true; }
        if constraint.uses_compact_segmented_parser_runtime() {
            let Some(root_reset) = constraint.recursive_tokenizer_reset_state(0) else {
                return false;
            };
            return state.values_for_key(root_reset).any(|stack| {
                !stack.is_empty()
                    && constraint
                        .compact_segmented_parser_is_finished(stack)
                        .unwrap_or(false)
            });
        }
        let product_initial = constraint.tokenizer.initial_state();
        let commit_initial = constraint.runtime_commit_initial_state();
        state
            .values_for_key(product_initial)
            .chain(
                (commit_initial != product_initial)
                    .then(|| state.values_for_key(commit_initial))
                    .into_iter()
                    .flatten(),
            )
            .any(|stack| {
                if let Some(parser) = constraint.template_parser() {
                    return parser.finished(stack);
                }
                !stack.is_empty()
                    && constraint
                        .sparse_direct_regular_gss_is_complete(stack)
                        .unwrap_or_else(|| {
                            if let Some(finished) =
                                constraint.compact_segmented_parser_is_finished(stack)
                            {
                                finished
                            } else if !constraint.parser_has_controls() {
                                stacks_finished(&constraint.table, stack)
                            } else {
                                stacks_finished_control_closed(&constraint.table, stack)
                            }
                        })
            })
    }
