use crate::automata::lexer::Lexer;
use std::sync::{Arc, Mutex, Weak};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::thread::LocalKey;
use super::ConstraintData;

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
// Work buffers retain their tuned capacity; live sequence storage is smaller.
pub(crate) const INLINE_PARSER_STATE_CAPACITY: usize = 64;
pub(crate) const LIVE_FRONTIER_CAPACITY: usize = 4;
pub(crate) type FrontierEntries = SmallVec<[(u32, ParserGSS); LIVE_FRONTIER_CAPACITY]>;

/// Parser paths grouped by tokenizer state, stored inline for the common small
/// frontier. Entries remain sorted by tokenizer-state ID. The bounded flat
/// runtime may temporarily retain multiple single-stack entries with the same
/// key instead of materializing them into one allocating GSS; general paths
/// normalize equal-key entries when they need map semantics.
#[derive(Clone, Default, PartialEq, Eq)]
pub(crate) struct ParserStateMap {
    pub(crate) entries: FrontierEntries,
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
    /// states. Arc clones are allocation-free; the vector may spill independently.
    pub(crate) fn replace_single_keys(&mut self, keys: &[u32]) -> bool {
        if self.entries.len() != 1 || keys.is_empty() {
            return false;
        }
        if keys.windows(2).any(|pair| pair[0] >= pair[1]) {
            return false;
        }
        let (_, gss) = self.entries.pop().unwrap();
        self.entries.reserve(keys.len());
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

        let (_, gss) = self.entries.swap_remove(0);
        self.entries.clear();
        self.entries.reserve(keys.len());
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
    type IntoIter = smallvec::IntoIter<[(u32, ParserGSS); LIVE_FRONTIER_CAPACITY]>;

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


// A bounded, per-thread cache of idle scratch leases, scoped to the exact
// immutable compiled constraint. Weak identity prevents pointer ABA without
// retaining a compiled grammar. A checkout removes the slot before invoking
// any parser or mask code, so nested/reentrant calls obtain independent
// storage and no RefCell borrow or synchronization guard spans runtime work.
// The lease is !Send; a public state contains no lease and can move freely
// between threads. On unwind discard the affected scratch.
type ScratchSlot<T> = (Weak<ConstraintData>, Box<T>);
type ScratchPool<T> = RefCell<Vec<ScratchSlot<T>>>;
const SCRATCH_POOL_LIMIT: usize = 8;

thread_local! {
    static COMMIT_SCRATCH: ScratchPool<CommitBuffers> = const { RefCell::new(Vec::new()) };
    static MASK_SCRATCH: ScratchPool<MaskScratch> = const { RefCell::new(Vec::new()) };
}

pub(crate) struct ScratchLease<T: 'static> {
    slot: Option<ScratchSlot<T>>,
    pool: &'static LocalKey<ScratchPool<T>>,
    _thread_bound: PhantomData<Rc<()>>,
}

impl<T: 'static> Deref for ScratchLease<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T { &self.slot.as_ref().expect("live scratch lease").1 }
}

impl<T: 'static> DerefMut for ScratchLease<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T { &mut self.slot.as_mut().expect("live scratch lease").1 }
}

impl<T: 'static> Drop for ScratchLease<T> {
    fn drop(&mut self) {
        if std::thread::panicking() { return; }
        let mut slot = self.slot.take();
        // Drop any displaced value outside the TLS borrow. TLS destruction
        // and reentrant acquisition must not panic in Drop.
        let mut displaced = None;
        let _ = self.pool.try_with(|cell| {
            let Ok(mut slots) = cell.try_borrow_mut() else { return; };
            // Retain nested leases too, so subsequent reentrant work does
            // not have to rebuild scratch for the same constraint.
            if slots.len() >= SCRATCH_POOL_LIMIT {
                displaced = Some(slots.remove(0));
            }
            slots.push(slot.take().expect("live scratch lease"));
        });
        drop(displaced);
    }
}

#[inline]
fn checkout_scratch<T: 'static>(
    pool: &'static LocalKey<ScratchPool<T>>,
    constraint: &Constraint,
    make: impl FnOnce(&Constraint) -> T,
) -> ScratchLease<T> {
    let ptr = Arc::as_ptr(&constraint.data);
    let slot = pool.try_with(|cell| {
        let mut slots = cell.try_borrow_mut().ok()?;
        let index = slots.iter().rposition(|(key, _)| key.as_ptr() == ptr)?;
        Some(slots.remove(index))
    }).ok().flatten().unwrap_or_else(|| (Arc::downgrade(&constraint.data), Box::new(make(constraint))));
    ScratchLease { slot: Some(slot), pool, _thread_bound: PhantomData }
}

#[inline]
pub(crate) fn commit_scratch(constraint: &Constraint) -> ScratchLease<CommitBuffers> {
    checkout_scratch(&COMMIT_SCRATCH, constraint, CommitBuffers::for_constraint)
}

#[inline]
pub(crate) fn mask_scratch(constraint: &Constraint) -> ScratchLease<MaskScratch> {
    checkout_scratch(&MASK_SCRATCH, constraint, MaskScratch::for_constraint)
}

pub(crate) fn warm_scratch(constraint: &Constraint) {
    drop(commit_scratch(constraint));
    drop(mask_scratch(constraint));
}

/// Cached fill_mask result, keyed on generation counter.
pub(crate) struct MaskCacheData {
    pub generation: u64,
    pub mask: Vec<u32>,
    /// The merged internal token dense bitmap used to compute this mask.
    /// Enables incremental updates when the state changes slightly.
    pub merged_dense: Vec<u64>,
}

/// A pointer-sized lazy per-sequence cache. The pointee is installed at most
/// once through a shared reference; clearing and reclamation require &mut self.
/// A populated cache uses the same Mutex as before, now beside its payload.
pub(crate) struct StateMaskCache {
    ptr: std::sync::atomic::AtomicPtr<Mutex<MaskCacheData>>,
}

impl StateMaskCache {
    pub(crate) const fn new() -> Self {
        Self { ptr: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }
    }

    #[inline]
    fn cell(&self) -> Option<&Mutex<MaskCacheData>> {
        // SAFETY: only a fully initialized Box is published (Release CAS).
        // Through &self it cannot be removed, replaced, or freed. The returned
        // borrow prevents obtaining the &mut self required to clear/drop it.
        unsafe { self.ptr.load(std::sync::atomic::Ordering::Acquire).as_ref() }
    }

    #[inline]
    pub(crate) fn lock(&self) -> std::sync::LockResult<Option<std::sync::MutexGuard<'_, MaskCacheData>>> {
        match self.cell() {
            None => Ok(None),
            Some(cell) => cell.lock().map(Some)
                .map_err(|e| std::sync::PoisonError::new(Some(e.into_inner()))),
        }
    }

    #[inline]
    pub(crate) fn get_mut(&mut self) -> std::sync::LockResult<Option<&mut MaskCacheData>> {
        // SAFETY: the unique borrow excludes concurrent shared access; the
        // pointer, if non-null, owns a live boxed Mutex.
        match unsafe { self.ptr.get_mut().as_mut() } {
            None => Ok(None),
            Some(cell) => cell.get_mut().map(Some)
                .map_err(|e| std::sync::PoisonError::new(Some(e.into_inner()))),
        }
    }

    pub(crate) fn store(&self, generation: u64, mask: &[u32], dense: &[u64]) {
        use std::sync::atomic::Ordering::{Acquire, AcqRel};
        let cell = match self.cell() {
            Some(cell) => cell,
            None => {
                let candidate = Box::into_raw(Box::new(Mutex::new(MaskCacheData {
                    generation, mask: mask.to_vec(), merged_dense: dense.to_vec(),
                })));
                match self.ptr.compare_exchange(std::ptr::null_mut(), candidate, AcqRel, Acquire) {
                    Ok(_) => return,
                    Err(existing) => {
                        // SAFETY: the failed CAS did not publish candidate; we
                        // retain sole ownership. The winner is fully initialized
                        // and cannot be freed while this shared borrow is live.
                        unsafe {
                            drop(Box::from_raw(candidate));
                            &*existing
                        }
                    }
                }
            }
        };
        let mut data = cell.lock().unwrap();
        data.generation = generation;
        data.mask.clear();
        data.mask.extend_from_slice(mask);
        data.merged_dense.clear();
        data.merged_dense.extend_from_slice(dense);
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        let ptr = std::mem::replace(self.ptr.get_mut(), std::ptr::null_mut());
        if !ptr.is_null() {
            // SAFETY: &mut self excludes all outstanding borrows/guards.
            // The pointer was produced by Box::into_raw and has one owner.
            unsafe { drop(Box::from_raw(ptr)); }
        }
    }
}

impl Drop for StateMaskCache {
    fn drop(&mut self) { self.clear(); }
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
    /// Exact component initial states/masks, prepared outside mask timing. A
    /// composed projection frequently lands exactly at a component entry; in
    /// that case reuse the component's already-cached initial mask rather than
    /// invoking its uncached mask engine through a synthetic shadow state.
    pub segmented_component_initial_states: Vec<ParserStateMap>,
    pub segmented_component_initial_masks: Vec<Vec<u32>>,
}

impl MaskScratch {
    pub(crate) fn for_constraint(constraint: &Constraint) -> Self {
        let dense_words = constraint.internal_token_dense_words;
        let mut segmented_component_initial_states = Vec::new();
        let mut segmented_component_initial_masks = Vec::new();
        if let Some(overlay) = constraint.static_dynamic_overlay.as_ref() {
            let authoritative_dynamic_direct = overlay.segmented_mask_authoritative
                && !overlay.segmented_parser_components.is_empty()
                && overlay.segmented_parser_components.iter().all(|component| {
                    component.boundary.as_ref().is_some_and(|shard| {
                        matches!(
                            shard.backend,
                            crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
                        )
                    })
                });
            if !authoritative_dynamic_direct {
                segmented_component_initial_states.reserve(overlay.segmented_parser_components.len());
                segmented_component_initial_masks.reserve(overlay.segmented_parser_components.len());
                for component in &overlay.segmented_parser_components {
                    let source = component.constraint.as_ref();
                    segmented_component_initial_states.push(source.initial_state_map());
                    segmented_component_initial_masks.push(source.start().mask());
                }
            }
        }
        Self {
            merged_dense: Vec::with_capacity(dense_words),
            chain_merged_dense: Vec::with_capacity(dense_words),
            output_buf: Vec::with_capacity(constraint.body_mask_len()),
            // Allocate and touch these before any timed runtime operation. The
            // direct kernel clears/reuses them without changing capacity.
            single_path_aux_dense: vec![0; dense_words],
            single_path_acc_dense: vec![0; dense_words],
            indexed_dag_mask: crate::runtime::mask::IndexedDagMaskRuntime::default(),
            segmented_component_initial_states,
            segmented_component_initial_masks,
        }
    }
}

/// Exact admission facts retained for one structurally persistent parser GSS.
///
/// `gss` is a strong Arc-backed clone, so pointer identity cannot be recycled
/// while the entry exists. `tested`/`admitted` record pointwise exact terminal
/// admission. `boolean_queries` caches existential results for the common
/// single tokenizer-end-state case without forcing a full admitted-set closure.
#[derive(Debug, Clone)]
pub(crate) struct ParserAdmissionCacheEntry {
    pub(crate) gss: ParserGSS,
    pub(crate) tested: BitSet,
    pub(crate) admitted: BitSet,
    pub(crate) boolean_queries: SmallVec<[(BitSet, bool); 8]>,
}

/// Proven equality of the next-token vocabulary observation for two exact
/// lexer states under one parser-admitted terminal set. The proof itself is
/// parser-stack independent once the admitted set is fixed: before the first
/// finalization only these terminals are observable, and after an equal
/// admitted finalization both sides advance the same current GSS and reset to
/// the same lexer state.
#[derive(Debug, Clone)]
pub(crate) struct ParserRelativeMaskEqCacheEntry {
    pub(crate) left: u32,
    pub(crate) right: u32,
    pub(crate) admitted: BitSet,
}

/// Reusable scratch buffers for `commit_bytes_impl`, retained between calls
/// to avoid repeated heap allocation.
#[derive(Debug)]
pub(crate) struct CommitBuffers {
    pub advance_result_cache: FxHashMap<(usize, u32), (ParserGSS, ParserGSS)>,
    pub semantic_frontier_keys: GssSemanticKeyInterner<u32, TerminalsDisallowed>,
    pub admission_cache: SmallVec<[ParserAdmissionCacheEntry; 8]>,
    pub parser_relative_mask_eq_cache: SmallVec<[ParserRelativeMaskEqCacheEntry; 8]>,
    pub pending_state: FxHashMap<u32, ParserGSS>,
    pub seen_matches: FxHashSet<(usize, u32)>,
    pub terminal_result_cache: FxHashMap<u32, ParserGSS>,
    pub exec_results: FxHashMap<u32, crate::automata::lexer::tokenizer::TokenizerExecResult>,
    pub small_exec_result: crate::automata::lexer::tokenizer::TokenizerExecResult,
    pub reusable_tokenizer_exec:
        crate::runtime::commit::tokenizer_scan::ReusableTokenizerExecScratch,
    pub prune_tokenizer_exec:
        crate::runtime::commit::tokenizer_scan::ReusableTokenizerExecScratch,
    // Read-only mask projections do not need the large commit-only bucket
    // arrays. Ordinary states still prepare them eagerly at construction;
    // owning them indirectly also keeps recursive mask/state frames small.
    pub small_queue: Option<Box<crate::runtime::commit::SmallCommitQueueScratch>>,
    // The fixed frontier arrays are large enough that embedding them here
    // makes several nested state-construction frames exceed a native musl
    // pthread stack. Retain one eagerly prepared allocation across commits.
    pub flat_frontier: Option<Box<crate::runtime::commit::FlatFrontierScratch>>,
    pub linear_stack_original: Vec<u32>,
    pub linear_stack_work: Vec<u32>,
    pub processing_queue: Vec<FxHashMap<u32, ParserGSS>>,
    pub template_advance_runtime: crate::runtime::commit::TemplateAdvanceRuntime,
}

impl Default for CommitBuffers {
    fn default() -> Self {
        Self::with_flat_frontier_preallocation(256)
    }
}

impl CommitBuffers {
    /// Short-lived read-only mask projections never execute a token commit.
    /// Do not construct the speculative flat-commit GSS pool for each mask.
    /// Keep ordinary ConstraintState/commit allocation policy unchanged.
    pub(crate) fn for_mask_only_shadow() -> Self {
        Self::with_allocation_policy::<true>(0)
    }

    /// Finite recursive mask probes use exact queue/lexer advancement, not the
    /// speculative flat-commit pool. Other callers retain their existing pool.
    pub(crate) fn for_finite_recursive_mask() -> Self {
        Self::with_flat_frontier_preallocation(0)
    }

    fn with_flat_frontier_preallocation(preallocated_gss: usize) -> Self {
        Self::with_allocation_policy::<false>(preallocated_gss)
    }

    fn with_allocation_policy<const MASK_ONLY: bool>(preallocated_gss: usize) -> Self {
        Self {
            advance_result_cache: FxHashMap::default(),
            semantic_frontier_keys: GssSemanticKeyInterner::with_capacity(if MASK_ONLY { 0 } else { 256 }),
            admission_cache: SmallVec::new(),
            parser_relative_mask_eq_cache: SmallVec::new(),
            pending_state: FxHashMap::default(),
            seen_matches: FxHashSet::default(),
            terminal_result_cache: FxHashMap::default(),
            exec_results: FxHashMap::default(),
            small_exec_result: crate::automata::lexer::tokenizer::TokenizerExecResult {
                end_state: crate::automata::lexer::tokenizer::TokenizerStateSet::new(),
                matches: Vec::with_capacity(if MASK_ONLY { 0 } else { 8 }),
            },
            reusable_tokenizer_exec:
                crate::runtime::commit::tokenizer_scan::ReusableTokenizerExecScratch::with_capacity(if MASK_ONLY { 0 } else { 512 }),
            prune_tokenizer_exec:
                crate::runtime::commit::tokenizer_scan::ReusableTokenizerExecScratch::with_capacity(if MASK_ONLY { 0 } else { 512 }),
            small_queue: if MASK_ONLY { None } else {
                Some(crate::runtime::commit::SmallCommitQueueScratch::boxed())
            },
            flat_frontier: {
                static LAZY_MASK_STACKS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                if MASK_ONLY && *LAZY_MASK_STACKS.get_or_init(|| {
                    crate::env::var("GLRMASK_LAZY_MASK_COMMIT_SCRATCH")
                        .map_or(true, |v| matches!(v.as_str(), "1" | "true"))
                }) {
                    None
                } else {
                    Some(crate::runtime::commit::FlatFrontierScratch::boxed_with_preallocated_gss(preallocated_gss))
                }
            },
            linear_stack_original: Vec::with_capacity(if MASK_ONLY { 0 } else { LINEAR_STACK_RESERVE }),
            linear_stack_work: Vec::with_capacity(if MASK_ONLY { 0 } else { LINEAR_STACK_RESERVE }),
            processing_queue: Vec::new(),
            template_advance_runtime:
                if MASK_ONLY { crate::runtime::commit::TemplateAdvanceRuntime::for_mask_only_shadow() }
                else { crate::runtime::commit::TemplateAdvanceRuntime::default() },
        }
    }

    pub(crate) fn for_constraint(constraint: &Constraint) -> Self {
        const ORDINARY_DYNAMIC_PREALLOCATED_GSS: usize = 32;
        let ordinary_dynamic = constraint.uses_dynamic_runtime()
            && constraint.static_dynamic_overlay.is_none()
            && !constraint.table_has_ambiguity()
            && !constraint.tokenizer_has_epsilon_transitions;
        if ordinary_dynamic {
            Self::with_flat_frontier_preallocation(ORDINARY_DYNAMIC_PREALLOCATED_GSS)
        } else {
            Self::default()
        }
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
        self.exec_results.clear();
        self.small_exec_result.end_state.clear();
        self.small_exec_result.matches.clear();
        self.reusable_tokenizer_exec.states.clear();
        self.reusable_tokenizer_exec.matches.clear();
        self.prune_tokenizer_exec.states.clear();
        self.prune_tokenizer_exec.matches.clear();
        if let Some(queue) = self.small_queue.as_mut() { queue.clear(); }
        if let Some(frontier) = self.flat_frontier.as_mut() { frontier.clear(); }
        self.linear_stack_original.clear();
        self.linear_stack_work.clear();
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
    pub(crate) markers: Option<Arc<crate::runtime::markers::MarkerState>>,
    pub(crate) constraint: Constraint,
    // Small live frontier; scratch and bounded commit algorithms retain their own capacity.
    pub(crate) state: ParserStateMap,
    /// Low 63 bits: commit generation; high bit: explicit end-token termination.
    /// The cache is cleared on generation rollover.
    pub(crate) epoch: u64,
    /// Cached fill_mask result: returned directly when state matches cached snapshot.
    /// Not cloned — clone starts with empty cache.
    pub(crate) mask_cache: StateMaskCache,
}

impl Clone for ConstraintState {

    fn clone(&self) -> Self {
        ConstraintState {
            markers: self.markers.clone(),
            constraint: self.constraint.clone(),
            state: self.state.clone(),
            epoch: self.epoch,
            mask_cache: StateMaskCache::new(),
        }
    }
}

impl std::fmt::Debug for ConstraintState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstraintState")
            .field("terminated", &self.terminated())
            .field("state_len", &self.state.len())
            .field("mask_cached", &self.mask_cache.lock().unwrap().is_some())
            .finish()
    }
}

impl ConstraintState {
    const TERMINATED_BIT: u64 = 1 << 63;
    const GENERATION_MASK: u64 = Self::TERMINATED_BIT - 1;

    #[inline]
    pub(crate) fn generation(&self) -> u64 {
        self.epoch & Self::GENERATION_MASK
    }

    #[inline]
    pub(crate) fn terminated(&self) -> bool {
        self.epoch & Self::TERMINATED_BIT != 0
    }

    #[inline]
    pub(crate) fn advance_generation(&mut self, terminated: bool) {
        let next = self.epoch.wrapping_add(1) & Self::GENERATION_MASK;
        // An arbitrarily old cache must not become valid again after rollover.
        if next == 0 { self.mask_cache.clear(); }
        self.epoch = next | (u64::from(terminated) << 63);
    }

    pub(crate) fn copy_snapshot_from(&mut self, source: &ConstraintState) {
        assert!(Arc::ptr_eq(&self.constraint.data, &source.constraint.data));
        self.markers = source.markers.clone();
        self.state = source.state.clone();
        self.epoch = source.epoch;
        self.mask_cache.clear();
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
            return self.terminated();
        }
        frontier_is_accepting(&self.constraint, &self.state, self.terminated())
    }

    /// Whether the parser frontier is a complete grammar match, ignoring any
    /// end-token policy.
    pub(crate) fn frontier_complete(&self) -> bool {
        frontier_is_accepting(&self.constraint, &self.state, false)
    }

    #[cfg(feature = "internal-api")]
    pub(crate) fn frontier_shape(&self) -> (usize, usize, usize, usize, Vec<u32>) {
        let entries = &self.state.entries;
        let mut keys = Vec::new();
        let mut ptr_duplicates = 0;
        let mut equal_duplicates = 0;
        for (i, (key, gss)) in entries.iter().enumerate() {
            if keys.last() != Some(key) { keys.push(*key); }
            let previous = &entries[..i];
            if previous.iter().any(|(k, g)| k == key && gss.ptr_eq(g)) {
                ptr_duplicates += 1;
            } else if previous.iter().any(|(k, g)| k == key && gss == g) {
                equal_duplicates += 1;
            }
        }
        (entries.len(), keys.len(), ptr_duplicates, equal_duplicates,
         entries.iter().map(|(k, _)| *k).collect())
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
        if self.terminated() {
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
                if let Some(parser) = &constraint.template_parser {
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


#[cfg(test)]
mod clone_and_scratch_tests {
    use super::*;
    use crate::{Grammar, Vocab};

    fn constraint(source: &str) -> Constraint {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"c".to_vec())]).unwrap();
        Constraint::compile(Grammar::glrm(source), &vocab).unwrap()
    }

    #[test]
    fn sequence_state_layout() {
        eprintln!("state={} frontier={} entry={} constraint={} markers={} cache={}",
            std::mem::size_of::<ConstraintState>(), std::mem::size_of::<ParserStateMap>(),
            std::mem::size_of::<(u32, ParserGSS)>(), std::mem::size_of::<Constraint>(),
            std::mem::size_of::<Option<Arc<crate::runtime::markers::MarkerState>>>(),
            std::mem::size_of::<StateMaskCache>());
    }

    #[test]
    fn packed_epoch_preserves_termination_and_invalidates_rollover() {
        let c = constraint("start root; nt root = 'a'+;");
        let mut state = c.start();
        state.epoch = ConstraintState::GENERATION_MASK;
        state.mask_cache.store(0, &[1], &[]);
        state.advance_generation(true);
        assert_eq!(state.generation(), 0);
        assert!(state.terminated());
        assert!(state.mask_cache.lock().unwrap().is_none());
        let mut clone = state.clone();
        assert!(clone.terminated());
        clone.advance_generation(false);
        assert_eq!(clone.generation(), 1);
        assert!(!clone.terminated());
        assert!(state.terminated());
        state.copy_snapshot_from(&clone);
        assert_eq!(state.generation(), 1);
        assert!(!state.terminated());
        #[cfg(target_pointer_width = "64")]
        assert!(std::mem::size_of::<ConstraintState>() <= 112);
    }

    #[test]
    fn compact_mask_cache_is_lazy_and_concurrent() {
        let mut cache = StateMaskCache::new();
        assert_eq!(std::mem::size_of::<StateMaskCache>(), std::mem::size_of::<*const ()>());
        assert!(cache.lock().unwrap().is_none());
        let start = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let (cache, start) = (&cache, &start);
                scope.spawn(move || {
                    start.wait();
                    cache.store(7, &[1, 2], &[3, 4]);
                    let guard = cache.lock().unwrap().unwrap();
                    assert_eq!((guard.generation, guard.mask.as_slice(), guard.merged_dense.as_slice()),
                        (7, &[1, 2][..], &[3, 4][..]));
                });
            }
        });
        cache.get_mut().unwrap().unwrap().generation = 8;
        cache.store(9, &[5], &[]);
        let guard = cache.lock().unwrap().unwrap();
        assert_eq!(guard.generation, 9);
        assert_eq!(guard.mask, [5]);
        assert!(guard.merged_dense.is_empty());
        drop(guard);
        cache.clear();
        assert!(cache.lock().unwrap().is_none());
        cache.store(10, &[6], &[]);
        assert_eq!(cache.lock().unwrap().unwrap().generation, 10);
    }

    #[test]
    fn constraint_mask_length_cache_is_shared_and_cow_invalidated() {
        let original = constraint("start root; nt root = 'a'+;");
        assert_eq!(std::mem::size_of::<Constraint>(), std::mem::size_of::<*const ()>());
        let old_len = original.mask_bit_len();
        let mut changed = original.clone();
        assert_eq!(changed.cached_mask_bit_len.get(), Some(&old_len));
        changed.end_tokens = Arc::from([99u32]);
        assert_eq!(changed.cached_mask_bit_len.get(), None);
        assert_eq!(changed.mask_bit_len(), 100);
        assert_eq!(original.mask_bit_len(), old_len);
        assert!(!Arc::ptr_eq(&original.data, &changed.data));
    }


    // Deliberately opt-in: use the same cached GPT-2 + padding vocabulary as
    // the clone investigation; no full corpus or timing matrix.
    #[test]
    #[ignore = "developer profile; set GLRMASK_FRONTIER_PROFILE_VOCAB to a token-hex file"]
    fn frontier_width_profile() {
        let path = std::env::var_os("GLRMASK_FRONTIER_PROFILE_VOCAB").expect("token-hex path");
        let text = std::fs::read_to_string(path).unwrap();
        let entries: Vec<_> = text.lines().enumerate().map(|(id,line)| {
            let bytes = line.as_bytes().as_chunks::<2>().0.iter()
                .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(),16).unwrap())
                .collect::<Vec<_>>();
            (id as u32, bytes)
        }).collect();
        let mut byte_ids = [u32::MAX;256];
        for (id,bytes) in &entries {
            if let [b] = bytes.as_slice() {
                let slot = &mut byte_ids[*b as usize];
                if *slot == u32::MAX { *slot = *id; }
            }
        }
        assert!(byte_ids.iter().all(|&id| id != u32::MAX));
        let vocab = Vocab::new(entries).unwrap();
        let opts = crate::CompileOptions::default().mask_strategy(crate::MaskStrategy::FAST_COMPILE);
        let profile = |name: &str, compiled: &crate::Constraint, bytes: &[u8]| {
            let mut s = compiled.start();
            let mut histogram = std::collections::BTreeMap::new();
            *histogram.entry(s.state.len()).or_insert(0usize) += 1;
            for (i,b) in bytes.iter().enumerate().take(512) {
                assert!(s.commit_token(byte_ids[*b as usize]), "{name}: byte {i}");
                *histogram.entry(s.state.len()).or_insert(0) += 1;
            }
            eprintln!("FRONTIER {name} {histogram:?}");
        };
        let js = Grammar::from_glrm(include_str!("../programmatic_js/javascript.glrm")).unwrap()
            .compile_with(&vocab,opts.clone()).unwrap();
        for (name,bytes) in [
            ("js-dangling",include_bytes!("../../tests/data/javascript/16_dangling_else_and_nested_prefixes.js").as_slice()),
            ("js-deep",include_bytes!("../../tests/data/javascript/19_deep_nesting.js").as_slice()),
            ("js-comprehensive",include_bytes!("../../tests/data/javascript/22_comprehensive_integration.js").as_slice()),
        ] { profile(name,&js,bytes); }
        let nested = Grammar::from_json_schema(r#"{"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"id":{"type":"integer"},"name":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}}},"required":["id","name","tags"],"additionalProperties":false}}},"required":["items"],"additionalProperties":false}"#).unwrap()
            .compile_with(&vocab,opts.clone()).unwrap();
        profile("json-nested",&nested,br#"{"items": [{"id": 123, "name": "alpha", "tags": ["x", "y"]}, {"id": 456, "name": "beta", "tags": ["z"]}]}"#);
        let union = Grammar::from_json_schema(r#"{"type":"array","items":{"anyOf":[{"type":"string","pattern":"^ab[0-9]+$"},{"type":"string","pattern":"^abc.*$"},{"type":"object","properties":{"key":{"type":"string"},"value":{"type":"integer"}},"required":["key","value"],"additionalProperties":false},{"type":"integer"}]}}"#).unwrap()
            .compile_with(&vocab,opts).unwrap();
        profile("json-union",&union,br#"["ab123", "abcxyz", {"key": "word", "value": -12}, 42, "ab456"]"#);
    }


    #[test]
    fn scratch_checkout_is_reentrant_constraint_scoped_and_reusable() {
        let c = constraint("start root; nt root = 'a'+;");
        let other = constraint("start root; nt root = 'b'+;");
        let mut outer = commit_scratch(&c);
        outer.linear_stack_original.push(98765);
        let outer_ptr = &*outer as *const CommitBuffers;
        let inner = commit_scratch(&c);
        assert_ne!(outer_ptr, &*inner as *const CommitBuffers);
        assert!(inner.linear_stack_original.is_empty());
        let different = commit_scratch(&other);
        assert!(different.linear_stack_original.is_empty());
        drop(different);
        drop(inner);
        drop(outer);
        let mut reused = commit_scratch(&c);
        assert_eq!(outer_ptr, &*reused as *const CommitBuffers);
        assert_eq!(reused.linear_stack_original.pop(), Some(98765));
        drop(reused);
        let a = mask_scratch(&c);
        let b = mask_scratch(&c);
        assert!(!std::ptr::eq(&*a, &*b));
    }

    #[test]
    fn scratch_is_not_reused_after_unwind() {
        let c = constraint("start root; nt root = 'a'+;");
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut scratch = commit_scratch(&c);
            scratch.linear_stack_original.push(98765);
            panic!("simulated runtime unwind");
        }));
        assert!(failed.is_err());
        assert!(!commit_scratch(&c).linear_stack_original.contains(&98765));
    }

    #[test]
    fn bulk_rekey_reuses_spilled_frontier_storage() {
        let c = constraint("start root; nt root = 'a'+;");
        let initial = c.start();
        let gss = initial.state.values().next().unwrap().clone();
        let mut frontier = ParserStateMap::singleton(0, gss.clone());
        let keys: Vec<u32> = (0..65).collect();
        assert!(frontier.replace_single_keys(&keys));
        assert!(frontier.entries.spilled());
        assert!(frontier.values().all(|value| value.ptr_eq(&gss)));
        let storage = frontier.entries.as_ptr();
        assert!(frontier.replace_uniform_gss_keys(&[100]));
        assert_eq!(frontier.entries.as_ptr(), storage);
        assert!(frontier.replace_single_keys(&keys));
        assert_eq!(frontier.entries.as_ptr(), storage);
        assert_eq!(frontier.keys().copied().collect::<Vec<_>>(), keys);
        assert!(frontier.values().all(|value| value.ptr_eq(&gss)));
        assert!(!frontier.replace_uniform_gss_keys(&[2, 1]));
        assert_eq!(frontier.keys().copied().collect::<Vec<_>>(), keys);
    }

    #[test]
    fn sequence_clone_shares_gss_nodes_but_not_frontier_storage() {
        let c = constraint("start root; nt root = 'a' ('b' | 'c');");
        let mut original = c.start();
        assert!(original.commit_token(0));
        let expected = original.mask();
        let mut accepted = original.clone();
        let mut rejected = original.clone();
        for ((a, b), d) in original.state.values().zip(accepted.state.values()).zip(rejected.state.values()) {
            assert!(a.ptr_eq(b) && a.ptr_eq(d));
        }
        assert!(!std::ptr::eq(&original.state, &accepted.state));
        assert!(accepted.commit_token(1));
        assert!(accepted.is_accepting());
        assert!(!rejected.commit_token(0));
        assert!(!rejected.is_accepting());
        assert!(rejected.is_terminated());
        assert!(rejected.mask().iter().all(|&word| word == 0));
        assert_eq!(original.mask(), expected);
        assert!(!original.is_accepting());
        assert!(original.commit_token(2));
        assert!(original.is_accepting());
    }
}
