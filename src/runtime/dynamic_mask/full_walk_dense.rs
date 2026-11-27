use super::*;
#[cfg(test)]
use super::boundary_cache::FullWalkBoundaryDirectCache;
use crate::runtime::artifact::{AdaptiveByteMemo, DynamicLazyUnionState};
use rustc_hash::FxHashSet;

#[inline(always)]
fn canonicalize_lazy_targets(targets: &mut SmallVec<[u32; 8]>) {
    if targets.windows(2).any(|pair| pair[0] > pair[1]) {
        targets.sort_unstable();
    }
    targets.dedup();
}

#[inline]
fn physical_sole_live_terminal(tokenizer: &Tokenizer, state: u32) -> Option<TerminalID> {
    let mut only = None::<TerminalID>;
    for terminal in tokenizer
        .matched_terminals_slice(state)
        .iter()
        .copied()
        .chain(
            tokenizer
                .possible_future_terminals(state)
                .iter_ones()
                .map(|terminal| terminal as TerminalID),
        )
    {
        match only {
            None => only = Some(terminal),
            Some(existing) if existing == terminal => {}
            Some(_) => return None,
        }
    }
    only
}

#[inline]
fn bitset_sole_live_terminal(matched: &BitSet, futures: &BitSet) -> Option<TerminalID> {
    let mut only = None::<TerminalID>;
    for terminal in matched
        .iter_ones()
        .chain(futures.iter_ones())
        .map(|terminal| terminal as TerminalID)
    {
        match only {
            None => only = Some(terminal),
            Some(existing) if existing == terminal => {}
            Some(_) => return None,
        }
    }
    only
}

// Parser admission rows may carry one additional EOF bit. Lexer futures
// contain only ordinary terminals and have zero padding, so intersect their
// shared words directly; EOF can never become a lexical witness.
#[inline(always)]
fn lexer_parser_rows_intersect(parser: &BitSet, lexer: &BitSet) -> bool {
    debug_assert!(parser.len() == lexer.len() || parser.len() == lexer.len() + 1);
    parser.words().iter().zip(lexer.words()).any(|(a, b)| a & b != 0)
}

trait FullWalkTransitionTable {
    type Cell: Copy;

    fn cell(&self, state: u32, byte: u8) -> Self::Cell;

    fn cell_is_dead(cell: Self::Cell) -> bool;

    fn cell_has_finalizer(cell: Self::Cell) -> bool;

    fn cell_target(cell: Self::Cell) -> u32;

    /// Exact self-transition classes of an already materialized table row.
    /// Lazy tables override this: proving an alphabet must not create a large
    /// collection of speculative derivative states for unused vocabulary bytes.
    fn proven_identity_byte_classes(
        &self,
        _tokenizer: &Tokenizer,
        state: u32,
    ) -> Option<FullWalkIdentityByteClasses> {
        Some(full_walk_identity_byte_classes(state, |byte| {
            let cell = self.cell(state, byte);
            (!Self::cell_is_dead(cell))
                .then(|| (Self::cell_target(cell), Self::cell_has_finalizer(cell)))
        }))
    }

    /// Exact alternatives sharing one parser coordinate. Ordinary rows are
    /// atomic. A lazy union may expose its bounded physical member set, but
    /// never derivative approximations or members from another table.
    #[inline]
    fn continuation_witness_components(&self, state: u32) -> SmallVec<[u32; 8]> {
        smallvec::smallvec![state]
    }

    #[inline(always)]
    fn transition(&self, state: u32, byte: u8) -> u32 {
        let cell = self.cell(state, byte);
        if Self::cell_is_dead(cell) {
            u32::MAX
        } else {
            Self::cell_target(cell)
        }
    }

    #[inline(always)]
    fn state_count(&self, tokenizer: &Tokenizer) -> usize {
        tokenizer.num_states() as usize
    }

    #[inline(always)]
    fn finalizer_code(&self, state: u32, base: &[u32]) -> u32 {
        unsafe { *base.get_unchecked(state as usize) }
    }

    #[inline(always)]
    fn single_finalizer_continues(&self, state: u32, base: &[u8]) -> bool {
        unsafe { *base.get_unchecked(state as usize) != 0 }
    }

    #[inline]
    fn matched_terminals(&self, tokenizer: &Tokenizer, state: u32) -> SmallVec<[TerminalID; 4]> {
        tokenizer.matched_terminals_slice(state).iter().copied().collect()
    }


    #[inline(always)]
    fn future_contains(&self, tokenizer: &Tokenizer, state: u32, terminal: TerminalID) -> bool {
        tokenizer.possible_future_terminals(state).contains(terminal as usize)
    }

    #[inline(always)]
    fn future_intersects(&self, tokenizer: &Tokenizer, state: u32, terminals: &BitSet) -> bool {
        lexer_parser_rows_intersect(terminals, tokenizer.possible_future_terminals(state))
    }

    /// Return the sole matched/future terminal at this exact walk coordinate.
    /// `None` means either no terminal is live or more than one distinct
    /// terminal is live.  Synthetic subset/union coordinates override this to
    /// read their already-materialized exact terminal metadata.
    #[inline]
    fn sole_live_terminal(&self, tokenizer: &Tokenizer, state: u32) -> Option<TerminalID> {
        physical_sole_live_terminal(tokenizer, state)
    }

    #[inline]
    fn union_states(&self, _states: &[u32]) -> Option<u32> {
        None
    }

    #[inline(always)]
    fn dense_output_hint(&self, _state: u32) -> Option<bool> {
        None
    }

    #[inline(always)]
    fn cache_dense_output_hint(&self, _state: u32, _dense: bool) {}
}

#[derive(Clone, Copy)]
struct FullWalkFlat16<'a> {
    transitions: &'a [u16],
}

impl FullWalkTransitionTable for FullWalkFlat16<'_> {
    type Cell = u16;

    #[inline(always)]
    fn cell(&self, state: u32, byte: u8) -> u16 {
        unsafe {
            *self
                .transitions
                .get_unchecked((state as usize).wrapping_mul(256) + byte as usize)
        }
    }

    #[inline(always)]
    fn cell_is_dead(cell: u16) -> bool {
        cell == u16::MAX
    }

    #[inline(always)]
    fn cell_has_finalizer(cell: u16) -> bool {
        cell & 0x8000 != 0
    }

    #[inline(always)]
    fn cell_target(cell: u16) -> u32 {
        u32::from(cell & 0x7fff)
    }

}


struct FullWalkLazyUnion<'a> {
    base_transitions16: Option<&'a [u16]>,
    base_transitions32: Option<&'a [u32]>,
    base_state_count: u32,
    tokenizer: Arc<Tokenizer>,
    overflowed: &'a std::cell::Cell<bool>,
    use_union_pair_memo: bool,
    cache: std::cell::UnsafeCell<std::sync::MutexGuard<'a, DynamicLazyUnionCache>>,
}

impl<'a> FullWalkLazyUnion<'a> {
    const UNBUILT: u32 = u32::MAX - 1;
    const SOFT_MAX_EXTENSION_STATES: usize = 4096;
    const RESERVED_EXTENSION_STATES: usize = 8192;
    const MAX_UNION_PAIR_MEMO: usize = 2048;

    fn new(
        tokenizer: &Arc<Tokenizer>,
        base_transitions16: Option<&'a [u16]>,
        base_transitions32: Option<&'a [u32]>,
        mut cache: std::sync::MutexGuard<'a, DynamicLazyUnionCache>,
        root_states: &[u32],
        overflowed: &'a std::cell::Cell<bool>,
    ) -> Option<(Self, u32)> {
        if root_states.len() < 2 {
            return None;
        }
        let base_state_count = tokenizer.num_states();
        if base_state_count >= 0x8000_0000u32.saturating_sub(Self::RESERVED_EXTENSION_STATES as u32) {
            return None;
        }
        if !cache.source.as_ref().is_some_and(|source| Arc::ptr_eq(source, tokenizer)) {
            Self::clear_cache(&mut cache);
            cache.source = Some(Arc::clone(tokenizer));
        }
        if cache.states.len() >= Self::SOFT_MAX_EXTENSION_STATES {
            // `clear_cache()` also drops the sparse physical transition rows.
            // Perform the soft-limit reset before sizing `base_rows`; doing
            // this in the opposite order leaves `base_rows` empty and makes
            // the unchecked hot-path lookup in `base_cell()` immediately
            // out-of-bounds on the next physical transition.
            Self::clear_cache(&mut cache);
        }
        if base_transitions16.is_none() && base_transitions32.is_none()
            && cache.base_rows.len() != base_state_count as usize
        {
            cache
                .base_rows
                .resize_with(base_state_count as usize, || None);
        }
        static UNION_PAIR_MEMO: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_union_pair_memo = full_walk_acceleration_enabled() && *UNION_PAIR_MEMO.get_or_init(|| {
            !crate::env::var("GLRMASK_DISABLE_LAZY_UNION_PAIR_MEMO").ok().is_some_and(|value| {
                matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
            })
        });
        let table = Self {
            base_transitions16,
            base_transitions32,
            base_state_count,
            tokenizer: Arc::clone(tokenizer),
            overflowed,
            use_union_pair_memo,
            cache: std::cell::UnsafeCell::new(cache),
        };
        let root = table.intern_states(root_states)?;
        Some((table, root))
    }

    #[inline]
    fn clear_cache(cache: &mut DynamicLazyUnionCache) {
        cache.base_rows.clear();
        cache.state_by_pair.clear();
        cache.state_by_union_pair.clear();
        cache.state_by_hash.clear();
        cache.states.clear();
        cache.dense_output_hints.clear();
    }

    fn intern_states(&self, states: &[u32]) -> Option<u32> {
        // Cached physical subsets already have canonical IDs, but discovering
        // that ID used to re-expand and sort both inputs on every call. Exact
        // input-coordinate pairs are stable until clear_cache resets the whole
        // namespace, at which point this memo is cleared too.
        let pair_key = if self.use_union_pair_memo {
            match states {
                [first, second] => Some((u64::from((*first).min(*second)) << 32) | u64::from((*first).max(*second))),
                _ => None,
            }
        } else { None };
        if let Some(key) = pair_key {
            let cache = unsafe { &*self.cache.get() };
            if let Some(result) = cache.state_by_union_pair.get(&key) {
                return Some(*result);
            }
        }
        let mut physical = SmallVec::<[u32; 8]>::new();
        {
            let cache = unsafe { &*self.cache.get() };
            for &state in states {
                if state < self.base_state_count {
                    physical.push(state);
                } else {
                    let subset = &cache.states.get((state - self.base_state_count) as usize)?.members;
                    physical.extend_from_slice(subset);
                }
            }
        }
        let cache = unsafe { &mut *self.cache.get() };
        let result = Self::intern_physical_inner(self.base_state_count, cache, physical);
        if let (Some(key), Some(result)) = (pair_key, result) {
            let memo = &mut cache.state_by_union_pair;
            if memo.len() < Self::MAX_UNION_PAIR_MEMO {
                memo.insert(key, result);
            }
        }
        if result.is_none() {
            self.overflowed.set(true);
        }
        result
    }

    fn intern_physical_inner(
        base_state_count: u32,
        cache: &mut DynamicLazyUnionCache,
        mut states: SmallVec<[u32; 8]>,
        ) -> Option<u32> {
        states.sort_unstable();
        states.dedup();
        Self::intern_sorted_physical_inner(base_state_count, cache, states)
    }

    fn intern_sorted_physical_inner(
        base_state_count: u32,
        cache: &mut DynamicLazyUnionCache,
        states: SmallVec<[u32; 8]>,
        ) -> Option<u32> {
        debug_assert!(states.windows(2).all(|pair| pair[0] < pair[1]));
        match states.as_slice() {
            [] => return None,
            [state] => return Some(*state),
            _ => {}
        }
        if states.iter().any(|&state| state >= base_state_count) {
            return None;
        }
        let pair_key = match states.as_slice() {
            [first, second] => Some((u64::from(*first) << 32) | u64::from(*second)),
            _ => None,
        };
        let hash = if let Some(key) = pair_key {
            if let Some(&state) = cache.state_by_pair.get(&key) {
                return Some(state);
            }
            None
        } else {
            let mut hasher = FxHasher::default();
            states.as_slice().hash(&mut hasher);
            let hash = hasher.finish();
            if let Some(ids) = cache.state_by_hash.get(&hash) {
                for &id in ids {
                    if cache.states[(id - base_state_count) as usize].members == states {
                        return Some(id);
                    }
                }
            }
            Some(hash)
        };
        if cache.states.len() >= Self::RESERVED_EXTENSION_STATES {
            return None;
        }
        let id = base_state_count.checked_add(cache.states.len() as u32)?;
        if let Some(key) = pair_key {
            cache.state_by_pair.insert(key, id);
        } else if let Some(hash) = hash {
            cache.state_by_hash.entry(hash).or_default().push(id);
        }
        cache.states.push(DynamicLazyUnionState {
            members: states,
            row: AdaptiveByteMemo::default(),
            metadata: None,
        });
        Some(id)
    }

    #[inline(always)]
    fn extension_index(&self, state: u32) -> usize {
        (state - self.base_state_count) as usize
    }

    #[inline(always)]
    fn checked_extension_index(&self, state: u32) -> Option<usize> {
        if state < self.base_state_count {
            return None;
        }
        let index = self.extension_index(state);
        let valid = unsafe {
            let cache = &*self.cache.get();
            index < cache.states.len()
        };
        if !valid {
            if crate::env::var_os("GLRMASK_PROFILE_INVALID_LAZY_UNION_STATE").is_some() {
                let cache = unsafe { &*self.cache.get() };
                eprintln!(
                    "[glrmask/profile][invalid_lazy_union_state] state={} base={} index={} subsets={} reserved={}",
                    state,
                    self.base_state_count,
                    index,
                    cache.states.len(),
                    Self::RESERVED_EXTENSION_STATES,
                );
            }
            // Lazy-union execution is an optional bounded acceleration. An
            // out-of-domain state must never become a correctness or memory
            // safety boundary: flag the attempt so the caller discards this
            // partial walk and reruns through the ordinary exact walker.
            self.overflowed.set(true);
            return None;
        }
        Some(index)
    }

    #[inline(always)]
    fn uncached_base_cell(&self, state: u32, byte: u8) -> u32 {
        if let Some(base_transitions) = self.base_transitions16 {
            let cell = unsafe {
                *base_transitions
                    .get_unchecked((state as usize).wrapping_mul(256) + byte as usize)
            };
            return if cell == u16::MAX {
                u32::MAX
            } else {
                u32::from(cell & 0x7fff)
                    | if cell & 0x8000 != 0 { 0x8000_0000 } else { 0 }
            };
        }
        if let Some(base_transitions) = self.base_transitions32 {
            return unsafe {
                *base_transitions
                    .get_unchecked((state as usize).wrapping_mul(256) + byte as usize)
            };
        }
        let tokenizer = self.tokenizer.as_ref();
        let target = tokenizer.dynamic_direct_transition(state, byte);
        if target == u32::MAX {
            return u32::MAX;
        }
        debug_assert!(target < 0x8000_0000);
        target
            | if tokenizer.matched_terminal_bitset(target).is_empty() {
                0
            } else {
                0x8000_0000
            }
    }

    fn base_identity_byte_classes(
        &self,
        tokenizer: &Tokenizer,
        state: u32,
    ) -> Option<FullWalkIdentityByteClasses> {
        if self.base_transitions16.is_some() || self.base_transitions32.is_some() {
            // Existing dense rows are cheap read-only evidence. Do not call
            // cell_raw(), which could materialize new union derivatives.
            return Some(full_walk_identity_byte_classes(state, |byte| {
                let cell = self.uncached_base_cell(state, byte);
                (cell != u32::MAX).then_some((cell & 0x7fff_ffff, cell & 0x8000_0000 != 0))
            }));
        }
        // This query declines virtual/compressed coordinates. Plain physical
        // transitions and their self-loop range query use the same immutable
        // byte-transition semantics as dynamic_direct_transition().
        tokenizer.dynamic_direct_transition_count(state)?;
        let bytes = Lexer::self_loop_bytes(tokenizer, state).to_words();
        let mut result = FullWalkIdentityByteClasses::default();
        if tokenizer.matched_terminal_bitset(state).is_empty() {
            result.ordinary = bytes;
        } else {
            result.finalizing = bytes;
        }
        Some(result)
    }

    #[inline(always)]
    fn base_cell(&self, state: u32, byte: u8) -> u32 {
        if self.base_transitions16.is_some() || self.base_transitions32.is_some() {
            return self.uncached_base_cell(state, byte);
        }
        // Scalar-dispatch tokenizers can expose thousands of physical states,
        // while one mask normally probes only a few bytes from each newly seen
        // state. Building a complete 256-cell row on the first probe creates a
        // large tail spike; never caching the probe makes common repeated states
        // needlessly expensive. Allocate the row lazily and fill only requested
        // cells, so each exact physical (state, byte) transition is paid once.
        let cached = unsafe {
            let cache = &mut *self.cache.get();
            let slot = cache.base_rows.get_unchecked_mut(state as usize);
            let row = slot.get_or_insert_with(|| Box::new([Self::UNBUILT; 256]));
            *row.get_unchecked(byte as usize)
        };
        if cached != Self::UNBUILT {
            return cached;
        }
        let value = self.uncached_base_cell(state, byte);
        unsafe {
            let cache = &mut *self.cache.get();
            *cache
                .base_rows
                .get_unchecked_mut(state as usize)
                .as_mut()
                .unwrap_unchecked()
                .get_unchecked_mut(byte as usize) = value;
        }
        value
    }

    #[inline(always)]
    fn cached_virtual_cell(&self, index: usize, byte: u8) -> Option<u32> {
        let cache = unsafe { &*self.cache.get() };
        cache.states.get(index)?.row.get(byte)
    }

    #[inline(always)]
    fn cache_virtual_cell(&self, index: usize, byte: u8, value: u32) {
        let cache = unsafe { &mut *self.cache.get() };
        unsafe { cache.states.get_unchecked_mut(index) }.row.insert(byte, value);
    }

    /// After a wide virtual subset has already paid for several independent
    /// byte probes, switch sparse physical components to row-wise construction
    /// only when their cheap packed-row lengths prove that enumerating every
    /// outgoing edge is no more work than another handful of subset probes.
    fn maybe_materialize_sparse_wide_row(&self, index: usize) -> bool {
        const WIDE_SUBSET_MIN: usize = 64;
        const PROBE_AMORTIZATION: usize = 8;
        if self.base_transitions16.is_some() || self.base_transitions32.is_some() {
            return false;
        }
        let members = {
            let cache = unsafe { &*self.cache.get() };
            let Some(DynamicLazyUnionState { row: AdaptiveByteMemo::Sparse(cells), members, .. }) = cache.states.get(index) else {
                return false;
            };
            if cells.len() < PROBE_AMORTIZATION {
                return false;
            }
            if members.len() < WIDE_SUBSET_MIN {
                return false;
            }
            members.clone()
        };
        let tokenizer = self.tokenizer.as_ref();
        let budget = members.len().saturating_mul(PROBE_AMORTIZATION);
        let mut outgoing = 0usize;
        for &member in &members {
            let Some(count) = tokenizer.dynamic_direct_transition_count(member) else {
                return false;
            };
            outgoing = outgoing.saturating_add(count);
            if outgoing > budget {
                return false;
            }
        }
        // Near-one-edge-per-state dispatcher layers are already cheap to probe
        // and do not amortize the fixed row-build machinery. Conversely, dense
        // rows were rejected by `budget` above. Materialize only the middle
        // regime where repeated subset probes dominate but sparse enumeration
        // is still bounded.
        if outgoing.saturating_mul(2) < members.len().saturating_mul(3) {
            return false;
        }

        let mut targets_by_byte: [SmallVec<[u32; 8]>; 256] =
            std::array::from_fn(|_| SmallVec::new());
        let mut touched = SmallVec::<[u8; 32]>::new();
        let mut finalizing = [false; 256];
        for member in members {
            for (byte, target) in tokenizer.transitions_from(member) {
                let targets = unsafe { targets_by_byte.get_unchecked_mut(byte as usize) };
                if targets.is_empty() {
                    touched.push(byte);
                }
                targets.push(target);
                if !finalizing[byte as usize]
                    && !tokenizer.matched_terminal_bitset(target).is_empty()
                {
                    finalizing[byte as usize] = true;
                }
            }
        }
        let mut row = Box::new([u32::MAX; 256]);
        for byte in touched {
            let targets = unsafe { targets_by_byte.get_unchecked_mut(byte as usize) };
            canonicalize_lazy_targets(targets);
            let target = match targets.as_slice() {
                [] => continue,
                [single] => *single,
                _ => {
                    let cache = unsafe { &mut *self.cache.get() };
                    let Some(target) = Self::intern_sorted_physical_inner(
                        self.base_state_count,
                        cache,
                        std::mem::take(targets),
                    ) else {
                        self.overflowed.set(true);
                        return false;
                    };
                    target
                }
            };
            row[byte as usize] = target
                | if finalizing[byte as usize] {
                    0x8000_0000
                } else {
                    0
                };
        }
        unsafe {
            (&mut *self.cache.get()).states[index].row = AdaptiveByteMemo::Dense(row);
        }
        true
    }

    #[inline(always)]
    fn cell_raw(&self, state: u32, byte: u8) -> u32 {
        if state < self.base_state_count {
            return self.base_cell(state, byte);
        }
        let Some(index) = self.checked_extension_index(state) else {
            return u32::MAX;
        };
        if let Some(cached) = self.cached_virtual_cell(index, byte) {
            return cached;
        }
        if self.maybe_materialize_sparse_wide_row(index) {
            return self
                .cached_virtual_cell(index, byte)
                .unwrap_or(u32::MAX);
        }

        let mut targets = SmallVec::<[u32; 8]>::new();
        let mut finalizer_bits = 0u32;
        // Virtual derivatives cache their own resulting cell, so member probes
        // deliberately bypass the physical-row cache. That makes this scan
        // read-only with respect to `DynamicLazyUnionCache`: borrow the canonical
        // subset in place instead of cloning a potentially heap-backed wide
        // union on every newly observed vocabulary byte.
        {
            let cache = unsafe { &*self.cache.get() };
            for &member in &cache.states[index].members {
                let cell = self.uncached_base_cell(member, byte);
                if cell == u32::MAX {
                    continue;
                }
                finalizer_bits |= cell & 0x8000_0000;
                targets.push(cell & 0x7fff_ffff);
            }
        }
        canonicalize_lazy_targets(&mut targets);
        let target = match targets.as_slice() {
            [] => u32::MAX,
            [state] => *state,
            _ => {
                let cache = unsafe { &mut *self.cache.get() };
                let Some(target) =
                    Self::intern_sorted_physical_inner(
                        self.base_state_count,
                        cache,
                        targets,
                    )
                else {
                    self.overflowed.set(true);
                    return u32::MAX;
                };
                target
            }
        };
        let value = if target == u32::MAX {
            u32::MAX
        } else {
            target | finalizer_bits
        };
        self.cache_virtual_cell(index, byte, value);
        value
    }

    #[inline(always)]
    fn ensure_virtual_metadata(&self, state: u32) -> bool {
        debug_assert!(state >= self.base_state_count);
        let Some(index) = self.checked_extension_index(state) else {
            return false;
        };
        if unsafe { (&*self.cache.get()).states[index].metadata.is_none() } {
            self.build_virtual_metadata(index);
        }
        true
    }

    // Every finalizer/future lookup hits the small cached branch above. Keep
    // first-use bitset construction out of those inlined vocabulary-walk
    // sites, while retaining the checked extension-coordinate boundary.
    #[cold]
    #[inline(never)]
    fn build_virtual_metadata(&self, index: usize) {
        let tokenizer = self.tokenizer.as_ref();
        let mut matched = BitSet::new(tokenizer.num_terminals() as usize);
        let mut futures = BitSet::new(tokenizer.num_terminals() as usize);
        {
            let cache = unsafe { &*self.cache.get() };
            for &member in &cache.states[index].members {
                matched.union_with(tokenizer.matched_terminal_bitset(member));
                futures.union_with(tokenizer.possible_future_terminals(member));
            }
        }
        let (first, second) = {
            let mut iter = matched.iter_ones().map(|terminal| terminal as TerminalID);
            (iter.next(), iter.next())
        };
        let finalizer_code = match (first, second) {
            (None, _) => u32::MAX,
            (Some(terminal), None) => terminal,
            _ => u32::MAX - 1,
        };
        let single_finalizer_continues = first
            .filter(|_| second.is_none())
            .is_some_and(|terminal| futures.contains(terminal as usize));
        let metadata = DynamicLazyUnionMetadata {
            finalizer_code,
            single_finalizer_continues: u8::from(single_finalizer_continues),
            matched,
            futures,
        };
        let cache = unsafe { &mut *self.cache.get() };
        if cache.states[index].metadata.is_none() {
            cache.states[index].metadata = Some(metadata);
        }
    }
}

impl FullWalkTransitionTable for FullWalkLazyUnion<'_> {
    type Cell = u32;

    #[inline(always)]
    fn cell(&self, state: u32, byte: u8) -> u32 {
        self.cell_raw(state, byte)
    }

    #[inline(always)] fn cell_is_dead(cell: u32) -> bool { cell == u32::MAX }
    #[inline(always)] fn cell_has_finalizer(cell: u32) -> bool { cell & 0x8000_0000 != 0 }
    #[inline(always)] fn cell_target(cell: u32) -> u32 { cell & 0x7fff_ffff }

    fn continuation_witness_components(&self, state: u32) -> SmallVec<[u32; 8]> {
        if state < self.base_state_count { return smallvec::smallvec![state]; }
        let Some(index) = self.checked_extension_index(state) else {
            return SmallVec::new();
        };
        let cache = unsafe { &*self.cache.get() };
        let members = &cache.states[index].members;
        if members.len() > 8 { return SmallVec::new(); }
        // The owned inline copy releases the UnsafeCell borrow before any
        // later metadata or transition query can grow the lazy-union cache.
        SmallVec::from_slice(members)
    }

    fn proven_identity_byte_classes(
        &self,
        tokenizer: &Tokenizer,
        state: u32,
    ) -> Option<FullWalkIdentityByteClasses> {
        if state < self.base_state_count {
            return self.base_identity_byte_classes(tokenizer, state);
        }
        let index = self.checked_extension_index(state)?;
        let cache = unsafe { &*self.cache.get() };
        let members = &cache.states[index].members;
        if members.len() > 8 { return None; }
        let (&first, rest) = members.split_first()?;
        let mut result = self.base_identity_byte_classes(tokenizer, first)?;
        // Every member staying itself is sufficient for the exact union to
        // stay itself. A union edge finalizes iff any member edge finalizes.
        // No derivative state is interned by this proof.
        for &member in rest {
            result.intersect_union_member(self.base_identity_byte_classes(tokenizer, member)?);
        }
        Some(result)
    }

    #[inline(always)]
    fn state_count(&self, _tokenizer: &Tokenizer) -> usize {
        self.base_state_count as usize + Self::RESERVED_EXTENSION_STATES
    }

    #[inline(always)]
    fn finalizer_code(&self, state: u32, base: &[u32]) -> u32 {
        if state < self.base_state_count {
            if !base.is_empty() {
                return unsafe { *base.get_unchecked(state as usize) };
            }
            let tokenizer = self.tokenizer.as_ref();
            return match tokenizer.matched_terminals_slice(state) {
                [] => u32::MAX,
                [terminal] => *terminal,
                _ => u32::MAX - 1,
            };
        }
        if !self.ensure_virtual_metadata(state) {
            return u32::MAX;
        }
        let cache = unsafe { &*self.cache.get() };
        cache.states[self.extension_index(state)].metadata
            .as_ref().expect("virtual subset metadata missing").finalizer_code
    }

    #[inline(always)]
    fn single_finalizer_continues(&self, state: u32, base: &[u8]) -> bool {
        if state < self.base_state_count {
            if !base.is_empty() {
                return unsafe { *base.get_unchecked(state as usize) != 0 };
            }
            let tokenizer = self.tokenizer.as_ref();
            return match tokenizer.matched_terminals_slice(state) {
                [terminal] => tokenizer
                    .possible_future_terminals(state)
                    .contains(*terminal as usize),
                _ => false,
            };
        }
        if !self.ensure_virtual_metadata(state) {
            return false;
        }
        let cache = unsafe { &*self.cache.get() };
        cache.states[self.extension_index(state)].metadata
            .as_ref().expect("virtual subset metadata missing").single_finalizer_continues != 0
    }

    #[inline]
    fn matched_terminals(&self, tokenizer: &Tokenizer, state: u32) -> SmallVec<[TerminalID; 4]> {
        if state < self.base_state_count {
            return tokenizer.matched_terminals_slice(state).iter().copied().collect();
        }
        if !self.ensure_virtual_metadata(state) {
            return SmallVec::new();
        }
        let cache = unsafe { &*self.cache.get() };
        cache.states[self.extension_index(state)].metadata
            .as_ref().expect("virtual subset metadata missing").matched
            .iter_ones().map(|terminal| terminal as TerminalID).collect()
    }


    #[inline(always)]
    fn future_contains(&self, tokenizer: &Tokenizer, state: u32, terminal: TerminalID) -> bool {
        if state < self.base_state_count {
            return tokenizer.possible_future_terminals(state).contains(terminal as usize);
        }
        if !self.ensure_virtual_metadata(state) {
            return false;
        }
        let cache = unsafe { &*self.cache.get() };
        cache.states[self.extension_index(state)].metadata
            .as_ref().expect("virtual subset metadata missing").futures.contains(terminal as usize)
    }

    #[inline(always)]
    fn future_intersects(&self, tokenizer: &Tokenizer, state: u32, terminals: &BitSet) -> bool {
        if state < self.base_state_count {
            return lexer_parser_rows_intersect(terminals, tokenizer.possible_future_terminals(state));
        }
        if !self.ensure_virtual_metadata(state) {
            return false;
        }
        let cache = unsafe { &*self.cache.get() };
        lexer_parser_rows_intersect(terminals, &cache.states[self.extension_index(state)].metadata
            .as_ref().expect("virtual subset metadata missing").futures)
    }

    #[inline]
    fn sole_live_terminal(&self, tokenizer: &Tokenizer, state: u32) -> Option<TerminalID> {
        if state < self.base_state_count {
            return physical_sole_live_terminal(tokenizer, state);
        }
        if !self.ensure_virtual_metadata(state) {
            return None;
        }
        let cache = unsafe { &*self.cache.get() };
        let metadata = cache.states[self.extension_index(state)].metadata
            .as_ref()
            .expect("virtual subset metadata missing");
        bitset_sole_live_terminal(&metadata.matched, &metadata.futures)
    }

    #[inline]
    fn union_states(&self, states: &[u32]) -> Option<u32> {
        self.intern_states(states)
    }

    #[inline(always)]
    fn dense_output_hint(&self, state: u32) -> Option<bool> {
        let cache = unsafe { &*self.cache.get() };
        match cache.dense_output_hints.get(&state).copied() {
            Some(2) => Some(true),
            Some(1) => Some(false),
            _ => None,
        }
    }

    #[inline(always)]
    fn cache_dense_output_hint(&self, state: u32, dense: bool) {
        let cache = unsafe { &mut *self.cache.get() };
        cache.dense_output_hints.insert(state, if dense { 2 } else { 1 });
    }
}

#[derive(Clone, Copy)]
struct FullWalkFlat32<'a> {
    transitions: &'a [u32],
}

impl FullWalkTransitionTable for FullWalkFlat32<'_> {
    type Cell = u32;

    #[inline(always)]
    fn cell(&self, state: u32, byte: u8) -> u32 {
        unsafe {
            *self
                .transitions
                .get_unchecked((state as usize).wrapping_mul(256) + byte as usize)
        }
    }

    #[inline(always)]
    fn cell_is_dead(cell: u32) -> bool {
        cell == u32::MAX
    }

    #[inline(always)]
    fn cell_has_finalizer(cell: u32) -> bool {
        cell & 0x8000_0000 != 0
    }

    #[inline(always)]
    fn cell_target(cell: u32) -> u32 {
        cell & 0x7fff_ffff
    }

}

#[derive(Clone, Copy, Debug, Default)]
struct LlgMasterDecision {
    /// Exact whole-token safe-string radius proved at the current lexer/parser
    /// frontier. `u16::MAX` denotes the complete unbounded safe+ language.
    safe_radius: u16,
    whitespace: bool,
}

impl LlgMasterDecision {
    #[inline(always)]
    fn admits_root_class(self, class: u16) -> bool {
        let safe_chars = crate::runtime::dynamic_mask_llg_master_safe_chars(class);
        (safe_chars != 0 && safe_chars <= self.safe_radius)
            || (self.whitespace
                && crate::runtime::dynamic_mask_llg_master_is_whitespace(class))
    }

    #[inline(always)]
    fn is_empty(self) -> bool {
        self.safe_radius == 0 && !self.whitespace
    }
}

/// Prove the exact master safe+/whitespace certificate while every original
/// lexer root still carries its exact source state. This must run before any
/// same-parser lexer-root union, because the union coordinate intentionally
/// discards the one-source provenance used by projected/symbolic proofs.
#[cold]
#[inline(never)]
fn precollapse_master_decision(
    state: &ConstraintState,
    vocab: &DynamicMaskVocab,
    root_branches: &DynamicBranches,
    lexer_state_count: usize,
) -> Option<LlgMasterDecision> {
    // Profiling flags are intentionally read once per proof attempt rather
    // than inside the hot parser/proof helpers. These knobs are sometimes
    // enabled dynamically by diagnostic harnesses, so unlike ordinary feature
    // flags they should not be process-cached.
    let profile_mask = dynamic_mask_profile_enabled(state.generation);
    let profile_proof = dynamic_mask_proof_profile_enabled() && profile_mask;
    let proof_definitions_available = vocab.llg_master_trie().is_some()
        && vocab.llg_slice_by_cache_id(LLG_SAFE_PLUS_SLICE as u32).is_some()
        && vocab.llg_slice_by_cache_id(LLG_WHITESPACE_SLICE as u32).is_some();
    if profile_mask {
        eprintln!(
            "[glrmask/profile][precollapse_master_entry] roots={} pending_guard={} missing_exact={} definitions={}",
            root_branches.len(),
            root_branches
                .iter()
                .any(|branch| !branch.initial_prune_guard.is_passed()),
            root_branches
                .iter()
                .any(|branch| branch.exact_tokenizer_state.is_none()),
            proof_definitions_available,
        );
    }
    if root_branches.is_empty()
        || root_branches
            .iter()
            .any(|branch| !branch.initial_prune_guard.is_passed())
        || root_branches
            .iter()
            .any(|branch| branch.exact_tokenizer_state.is_none())
        || !proof_definitions_available
    {
        return None;
    }

    let safe_plus_slice = vocab
        .llg_slice_by_cache_id(LLG_SAFE_PLUS_SLICE as u32)
        .expect("safe+ proof slice missing");
    let whitespace_slice = vocab
        .llg_slice_by_cache_id(LLG_WHITESPACE_SLICE as u32)
        .expect("whitespace proof slice missing");
    // Parser admission can be materially more expensive than the cheap
    // lexical support test below, and a failed pre-collapse proof otherwise
    // pays it again when the exact full walk constructs its own parser cache.
    // Every master proof in this function already requires one parser-admitted
    // terminal whose byte support covers the complete slice alphabet. Before
    // asking the parser anything, conservatively test the larger set of all
    // lexically-live terminals. If even that superset has no such terminal,
    // neither safe+/radius nor whitespace can possibly produce a certificate.
    // This is a necessary-condition gate only; it never admits a slice.
    let tokenizer = &state.constraint.tokenizer;
    let lexical_slice_candidate = |slice: &crate::runtime::artifact::DynamicMaskSliceTrie| {
        root_branches.iter().any(|branch| {
            let source = branch
                .exact_tokenizer_state
                .expect("precollapse proof requires exact source");
            tokenizer
                .matched_terminals_slice(source)
                .iter()
                .copied()
                .chain(
                    tokenizer
                        .possible_future_terminals(source)
                        .iter()
                        .map(|terminal| terminal as TerminalID),
                )
                .any(|terminal| {
                    tokenizer
                        .terminal_byte_support(terminal)
                        .is_some_and(|support| slice.slice_token_bytes().is_subset(&support))
                })
        })
    };
    if !lexical_slice_candidate(safe_plus_slice)
        && !lexical_slice_candidate(whitespace_slice)
    {
        if profile_mask {
            eprintln!("[glrmask/profile][precollapse_master] declined=lexical_support");
        }
        return None;
    }

    // These are immutable properties of the two proof DFAs. Compute them once
    // per surviving decision rather than rescanning all 256 bytes for every
    // root and again for the radius fallback. Keep them after the cheap lexical
    // gate above so proof-ineligible states pay none of this work.
    let safe_plus_first_bytes = safe_plus_slice.first_bytes();
    let whitespace_first_bytes = whitespace_slice.first_bytes();

    let (mut parser_cache, root_parser_nodes) =
        super::FullWalkParserCache::from_roots(root_branches, Some(lexer_state_count));
    let mut outcomes = [false; LLG_PROOF_SLOT_COUNT];
    // When the unbounded safe+ proof fails, the bounded-radius fallback asks
    // exactly the same immutable lexer eligibility question. Retain the
    // already-filtered candidates rather than repeating support/residual work.
    let mut safe_radius_candidates = Vec::<(u32, SmallVec<[TerminalID; 8]>)>::new();
    for slice_index in [LLG_SAFE_PLUS_SLICE, LLG_WHITESPACE_SLICE] {
        let slice = vocab
            .llg_slice_by_cache_id(slice_index as u32)
            .expect("requested proof slice missing");
        let slice_first_bytes = if slice_index == LLG_SAFE_PLUS_SLICE {
            safe_plus_first_bytes
        } else {
            whitespace_first_bytes
        };
        'sources: for (root_index, branch) in root_branches.iter().enumerate() {
            let source = branch
                .exact_tokenizer_state
                .expect("precollapse proof requires exact source");
            let admitted_started = profile_proof.then(std::time::Instant::now);
            let admitted = parser_cache
                .admitted(&state.constraint, root_parser_nodes[root_index])
                .clone();
            if let Some(started) = admitted_started {
                eprintln!(
                    "[glrmask/profile][proof_phase] kind=admitted slice={} count={} ms={:.3}",
                    slice_index,
                    admitted.count_ones(),
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }

            // Prepared build-time certificates are exact positive proofs for
            // this source state and slice.  A parser-admitted proving terminal
            // lets us bypass all lexer eligibility filtering and runtime
            // quotient/product work.  Empty/missing rows deliberately fall
            // through to the existing exact proof machinery.
            let prepared_slice_slot = if slice_index == LLG_SAFE_PLUS_SLICE {
                Some(0usize)
            } else if slice_index == LLG_WHITESPACE_SLICE {
                Some(1usize)
            } else {
                None
            };
            if let Some(slice_slot) = prepared_slice_slot
                && let Some(&terminal) = vocab
                    .prepared_master_provers(source, slice_slot)
                    .iter()
                    .find(|&&terminal| admitted.contains(terminal as usize))
                {
                    if profile_mask {
                        eprintln!(
                            "[glrmask/profile][precollapse_master_proof] slice={} source={} terminal={} proof=prepared",
                            slice_index, source, terminal,
                        );
                    }
                    outcomes[slice_index] = true;
                    break 'sources;
                }

            // Support and physical-residual coverage are immutable lexer facts
            // for this `(source, terminal, slice)`. Derive the candidate list
            // once and reuse it for virtual proof, quotient proof, and (for
            // safe+) the bounded-radius fallback below.
            // Parser admission alone may include hundreds of terminals
            // unrelated to this exact lexer source. Byte support is not a
            // liveness certificate. Refuse impossible candidates BEFORE any
            // per-terminal construction; this only removes an acceleration.
            let filter_source_live =
                crate::env::var_os("GLRMASK_DISABLE_PROOF_SOURCE_LIVE_GATE").is_none();
            let eligible = admitted
                .iter_ones()
                .map(|terminal| terminal as TerminalID)
                .filter(|&terminal| {
                    (!filter_source_live || tokenizer.terminal_state_is_live(source, terminal))
                        && tokenizer
                        .terminal_byte_support(terminal)
                        .is_some_and(|support| slice.slice_token_bytes().is_subset(&support))
                        && tokenizer
                            .physical_terminal_residual_covers_first_bytes(
                                source,
                                terminal,
                                slice_first_bytes,
                            )
                            != Some(false)
                })
                .collect::<SmallVec<[TerminalID; 8]>>();
            if profile_proof && slice_index == LLG_SAFE_PLUS_SLICE {
                for &terminal in &eligible {
                    for partition in 0usize..9 {
                        let Some(partition_dfa) = crate::compiler::stages::id_map_and_terminal_dwa::classify::vocab_partition_effective_dfa(partition) else {
                            continue;
                        };
                        let started = std::time::Instant::now();
                        let result = residual_regex_slice_prefix_contained(
                            &state.constraint.tokenizer,
                            source,
                            terminal,
                            partition_dfa.as_ref(),
                            32 * 1024,
                        );
                        eprintln!(
                            "[glrmask/profile][partition_residual_probe] source={} terminal={} partition={} result={:?} us={:.1}",
                            source,
                            terminal,
                            partition,
                            result,
                            started.elapsed().as_secs_f64() * 1e6,
                        );
                        let projected_started = std::time::Instant::now();
                        let projected = vocab.projected_terminal_slice_contained(
                            terminal,
                            source,
                            0x1000 + partition as u32,
                            partition_dfa.as_ref(),
                        );
                        eprintln!(
                            "[glrmask/profile][partition_projected_probe] source={} terminal={} partition={} result={:?} us={:.1}",
                            source,
                            terminal,
                            partition,
                            projected,
                            projected_started.elapsed().as_secs_f64() * 1e6,
                        );
                    }
                }
            }
            if slice_index == LLG_SAFE_PLUS_SLICE {
                safe_radius_candidates.push((source, eligible.clone()));
            }

            for &terminal in &eligible {
                if prepared_slice_slot
                    .and_then(|slot| {
                        vocab.prepared_master_proof_result(source, slot, terminal)
                    })
                    == Some(false)
                {
                    continue;
                }
                let proof_started = profile_proof.then(std::time::Instant::now);
                let virtual_proof = virtual_residual_slice_prefix_contained(
                    &state.constraint.tokenizer,
                    source,
                    terminal,
                    slice.dfa(),
                    1_536,
                );
                if let Some(started) = proof_started {
                    eprintln!(
                        "[glrmask/profile][proof_phase] kind=virtual slice={} terminal={} source={} result={:?} ms={:.3}",
                        slice_index,
                        terminal,
                        source,
                        virtual_proof,
                        started.elapsed().as_secs_f64() * 1e3,
                    );
                }
                if virtual_proof == Some(true) {
                    if profile_mask {
                        eprintln!(
                            "[glrmask/profile][precollapse_master_proof] slice={} source={} terminal={} proof=virtual",
                            slice_index, source, terminal,
                        );
                    }
                    outcomes[slice_index] = true;
                    break 'sources;
                }
            }

            let needs_quotient = eligible.iter().copied().any(|terminal| {
                prepared_slice_slot
                    .and_then(|slot| {
                        vocab.prepared_master_proof_result(source, slot, terminal)
                    })
                    .is_none()
            });
            // A grammar-quotiented dynamic vocabulary has already paid to
            // reduce the model vocabulary to grammar-equivalence representatives.
            // Building a second terminal-projection quotient lazily here can cost
            // orders of magnitude more than simply walking that small trie. Keep
            // consuming explicitly/prepared quotients when present, but do not
            // synthesize them online for the O2 runtime.
            if needs_quotient && !eligible.is_empty() && !vocab.is_grammar_quotiented() {
                if crate::env::var_os("GLRMASK_DISABLE_DEMAND_TERMINAL_PROOF").is_none() {
                    // The admitted candidate list is already exact. Preparing all
                    // broad terminals here puts unrelated grammar-wide work on a
                    // single mask's cold path. Reuse the existing per-terminal
                    // cache without changing the containment proof or its scope.
                    for &terminal in &eligible {
                        if prepared_slice_slot.and_then(|slot| {
                            vocab.prepared_master_proof_result(source, slot, terminal)
                        }).is_none() {
                            vocab.prepare_runtime_projected_terminal_quotient(
                                &state.constraint.tokenizer, terminal,
                            );
                        }
                    }
                } else {
                    vocab.prepare_runtime_projected_terminal_quotients(
                        &state.constraint.tokenizer,
                        &safe_plus_slice.slice_token_bytes(),
                        None,
                    );
                }
            }

            for &terminal in &eligible {
                if prepared_slice_slot
                    .and_then(|slot| {
                        vocab.prepared_master_proof_result(source, slot, terminal)
                    })
                    .is_some()
                {
                    continue;
                }
                let proof_started = profile_proof.then(std::time::Instant::now);
                let quotient_proof = vocab.projected_terminal_slice_contained(
                    terminal,
                    source,
                    slice.cache_id(),
                    slice.dfa(),
                );
                if let Some(started) = proof_started {
                    eprintln!(
                        "[glrmask/profile][proof_phase] kind=quotient slice={} terminal={} source={} result={:?} ms={:.3}",
                        slice_index,
                        terminal,
                        source,
                        quotient_proof,
                        started.elapsed().as_secs_f64() * 1e3,
                    );
                }
                if quotient_proof == Some(true) {
                    if profile_mask {
                        eprintln!(
                            "[glrmask/profile][precollapse_master_proof] slice={} source={} terminal={} proof=quotient",
                            slice_index, source, terminal,
                        );
                    }
                    outcomes[slice_index] = true;
                    break 'sources;
                }
            }
        }
    }

    if !outcomes[LLG_SAFE_PLUS_SLICE] {
        let max_vocab_safe_chars = u32::from(vocab.llg_master_max_safe_chars());
        let mut answers = Vec::<Option<u32>>::new();
        for (source, terminals) in &safe_radius_candidates {
                for &terminal in terminals {
                    let projected_started = profile_proof.then(std::time::Instant::now);
                    let projected_radius = vocab
                        .prepared_safe_radius(*source, terminal)
                        .map(u32::from)
                        .or_else(|| {
                            vocab.projected_terminal_slice_repeat_radius(
                                terminal,
                                *source,
                                safe_plus_slice.cache_id(),
                                safe_plus_slice.dfa(),
                                max_vocab_safe_chars,
                                16 * 1024,
                            )
                        });
                    if let Some(started) = projected_started {
                        eprintln!(
                            "[glrmask/profile][proof_phase] kind=projected_radius terminal={} source={} result={:?} ms={:.3}",
                            terminal,
                            source,
                            projected_radius,
                            started.elapsed().as_secs_f64() * 1e3,
                        );
                    }
                    let virtual_started = profile_proof.then(std::time::Instant::now);
                    let virtual_radius = virtual_residual_safe_repeat_radius(
                        &state.constraint.tokenizer,
                        *source,
                        terminal,
                        safe_plus_slice.dfa(),
                        max_vocab_safe_chars,
                        16 * 1024,
                    );
                    if let Some(started) = virtual_started {
                        eprintln!(
                            "[glrmask/profile][proof_phase] kind=virtual_radius terminal={} source={} result={:?} ms={:.3}",
                            terminal,
                            source,
                            virtual_radius,
                            started.elapsed().as_secs_f64() * 1e3,
                        );
                    }
                    let radius = match (projected_radius, virtual_radius) {
                        (Some(left), Some(right)) => Some(left.max(right)),
                        (left, right) => left.or(right),
                    };
                    answers.push(radius);
                }
            }
        let safe_radius = answers
            .into_iter()
            .flatten()
            .filter_map(|radius| u16::try_from(radius).ok())
            .max()
            .unwrap_or(0);
        let decision = LlgMasterDecision {
            safe_radius,
            whitespace: outcomes[LLG_WHITESPACE_SLICE],
        };
        return (!decision.is_empty()).then_some(decision);
    }

    let safe_radius = u16::MAX;
    let decision = LlgMasterDecision {
        safe_radius,
        whitespace: outcomes[LLG_WHITESPACE_SLICE],
    };
    (!decision.is_empty()).then_some(decision)
}

/// Execute a finite projection whose only live epsilon structure is a reset
/// dispatcher directly over its raw scalar component rows. The dispatcher
/// closure and any other multi-state root config are represented by the same
/// exact lazy-union table already used for deterministic same-parser unions.
///
/// This removes compile-time whole-product determinization without changing
/// the walk language. If the bounded derived-state cache cannot represent the
/// observed execution, return `false`; the caller reruns through the ordinary
/// exact NFA configuration walker.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_scalar_dispatch(
    state: &ConstraintState,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
    transitions16: Option<&[u16]>,
    transitions32: Option<&[u32]>,
    finalizer_code: Option<&[u32]>,
    single_finalizer_continues: Option<&[u8]>,
    ignored_output: Option<&[u32]>,
) -> Result<bool, String> {
    let tokenizer = lexer_scan_cache.tokenizer();
    if !tokenizer.has_scalar_deterministic_dispatch() {
        return Ok(false);
    }
    let profile = dynamic_mask_profile_enabled(state.generation);

    // A pending token-start guard is an exact output filter: its remembered
    // lexer/terminal pairs reject precisely those vocabulary tokens whose
    // bytes produce a blocked match. The predicate is deliberately independent
    // of parser resets during the candidate token. Factor it out of traversal
    // so each correlated root can use the ordinary Passed fast path (including
    // master slicing), then subtract that root's exact blocked-token mask.
    // Root branches are language alternatives, so OR the filtered root masks.
    if root_branches
        .iter()
        .any(|branch| !branch.initial_prune_guard.is_passed())
        && !joint_initial_guard_walk_enabled(
            root_branches,
            state.constraint.tokenizer.initial_state(),
        )
    {
        let mut merged = vec![0u32; buf.len()];
        let mut scratch = vec![0u32; buf.len()];
        for branch in root_branches {
            scratch.fill(0);
            let mut one = DynamicBranches::new();
            let blocked = branch
                .initial_prune_guard
                .blocked_output_mask(&state.constraint, buf.len())?;
            let mut unguarded = branch.clone();
            unguarded.initial_prune_guard = InitialPruneGuard::Passed;
            one.push(unguarded);
            if !try_scalar_dispatch(
                state,
                vocab,
                trie,
                &one,
                lexer_scan_cache,
                &mut scratch,
                transitions16,
                transitions32,
                finalizer_code,
                single_finalizer_continues,
                ignored_output,
            )? {
                return Ok(false);
            }
            if let Some(blocked) = blocked {
                for (word, &blocked) in scratch.iter_mut().zip(blocked.iter()) {
                    *word &= !blocked;
                }
            }
            for (dst, &word) in merged.iter_mut().zip(&scratch) {
                *dst |= word;
            }
        }
        if profile {
            eprintln!(
                "[glrmask/profile][scalar_dispatch] factored_pending_roots={}",
                root_branches.len(),
            );
        }
        buf.copy_from_slice(&merged);
        return Ok(true);
    }

    // The exact containment proofs below can be much more expensive than a
    // sparse vocabulary walk. Use the already-materialized finite mask
    // projection as a necessary-condition gate: if the union of immediate
    // bytes from the current root configs cannot even cover a slice's
    // first-byte language, that slice cannot help this mask. This gate may
    // conservatively skip an optimization; it never admits a token.
    let master_may_apply = if root_branches.iter().all(|branch| branch.initial_prune_guard.is_passed())
        && vocab.llg_master_trie().is_some()
        && let (Some(safe_plus), Some(whitespace)) = (
            vocab.llg_slice_by_cache_id(LLG_SAFE_PLUS_SLICE as u32),
            vocab.llg_slice_by_cache_id(LLG_WHITESPACE_SLICE as u32),
        )
    {
        let mut root_first_bytes = U8Set::empty();
        let mut physical = SmallVec::<[u32; 8]>::new();
        for branch in root_branches {
            lexer_scan_cache.physical_states_for_config(branch.tokenizer_config, &mut physical)?;
            for &raw_state in &physical {
                if tokenizer.state_has_epsilon_transitions(raw_state) {
                    continue;
                }
                if let Some(transitions) = transitions16 {
                    let row = (raw_state as usize).wrapping_mul(256);
                    if row + 256 > transitions.len() {
                        continue;
                    }
                    for byte in 0u16..=255 {
                        if unsafe { *transitions.get_unchecked(row + byte as usize) } != u16::MAX {
                            root_first_bytes.insert(byte as u8);
                        }
                    }
                } else if let Some(transitions) = transitions32 {
                    let row = (raw_state as usize).wrapping_mul(256);
                    if row + 256 > transitions.len() {
                        continue;
                    }
                    for byte in 0u16..=255 {
                        if unsafe { *transitions.get_unchecked(row + byte as usize) } != u32::MAX {
                            root_first_bytes.insert(byte as u8);
                        }
                    }
                } else {
                    root_first_bytes |= tokenizer.state_first_bytes(raw_state);
                }
            }
        }
        safe_plus
            .first_bytes()
            .is_subset(&root_first_bytes)
            || whitespace.first_bytes().is_subset(&root_first_bytes)
    } else {
        false
    };
    let master_started = profile.then(std::time::Instant::now);
    // A single exact root does not need a pre-collapse certificate: no lexer
    // provenance is about to be lost, and the later single-root master path can
    // prove the same slice directly (including the cheap virtual bounded-radius
    // fast path). Reserve the expensive pre-collapse prover for actual unions.
    let skip_single_virtual_precollapse = master_may_apply
        && root_branches.len() == 1
        && root_branches[0].exact_tokenizer_state.is_some_and(|source| {
            state
                .constraint
                .tokenizer
                .singleton_epsilon_closure(source)
                .into_iter()
                .any(|residual_state| {
                    state
                        .constraint
                        .tokenizer
                        .virtual_residual_terminal_for_state(residual_state)
                        .is_some()
                })
        });
    let precollapse_master_decision = (master_may_apply
        && !skip_single_virtual_precollapse
        && crate::env::var_os("GLRMASK_EXPERIMENT_DISABLE_PRECOLLAPSE_MASTER").is_none())
        .then(|| {
            precollapse_master_decision(
                state,
                vocab,
                root_branches,
                tokenizer.num_states() as usize,
            )
        })
        .flatten();
    if let Some(started) = master_started {
        eprintln!(
            "[glrmask/profile][scalar_dispatch_precollapse_master] eligible={} hit={} elapsed_ms={:.3}",
            master_may_apply,
            precollapse_master_decision.is_some(),
            started.elapsed().as_secs_f64() * 1e3,
        );
    }
    // Carry the certificate, not the replacement trie. The inner walk first
    // compares the certified residual work against this ordinary trie and
    // selects the master only if profitable. Replacing it here makes that
    // comparison use the larger master as its own baseline and also leaves
    // the wrong trie in place when the inner gate declines the certificate.
    let Some(reset_states) = tokenizer.sorted_deterministic_dispatch_roots() else {
        return Ok(false);
    };
    // Do not speculatively determinize the complete dispatcher subset
    // graph here. The vocabulary walk observes only a small portion of
    // that graph, while full construction can cost many milliseconds or
    // exceed the dense extension reserve. The exact lazy table below
    // materializes only derivatives actually requested by trie edges.
    let Some(subset_cache) = vocab.try_lock_lazy_union_cache() else {
        return Ok(false);
    };
    let overflowed = std::cell::Cell::new(false);
    let tokenizer_owner = vocab.mask_runtime_tokenizer_owner()
        .unwrap_or(&state.constraint.tokenizer);
    if !std::ptr::eq(tokenizer_owner.as_ref(), tokenizer) {
        return Ok(false);
    }
    let Some((lazy_transitions, initial_lexer_state)) = FullWalkLazyUnion::new(
        tokenizer_owner,
        transitions16,
        transitions32,
        subset_cache,
        &reset_states,
        &overflowed,
    ) else {
        return Ok(false);
    };

    let mut physical = SmallVec::<[u32; 8]>::new();
    let mut transformed = DynamicBranches::new();
    for branch in root_branches {
        lexer_scan_cache.physical_states_for_config(branch.tokenizer_config, &mut physical)?;
        physical.retain(|tokenizer_state| {
            let tokenizer_state = *tokenizer_state;
            if !tokenizer.state_has_epsilon_transitions(tokenizer_state) {
                return true;
            }
            debug_assert!(tokenizer.transitions_from(tokenizer_state).next().is_none());
            false
        });
        physical.sort_unstable();
        physical.dedup();
        if physical.is_empty() {
            continue;
        }
        let tokenizer_config = match physical.as_slice() {
            [single] => *single,
            _ => {
                let Some(state) = lazy_transitions.union_states(&physical) else {
                    return Ok(false);
                };
                state
            }
        };
        transformed.push(DynamicBranch {
            tokenizer_config,
            exact_tokenizer_state: branch.exact_tokenizer_state,
            parser_filtered_root: branch.parser_filtered_root,
            parser_filtered_transparent: branch.parser_filtered_transparent,
            residual_continuation_terminal: branch.residual_continuation_terminal,
            gss: branch.gss.clone(),
            shared_root_admission: branch.shared_root_admission.clone(),
            initial_prune_guard: branch.initial_prune_guard.clone(),
        });
    }
    if transformed.is_empty() {
        return Ok(false);
    }

    let mut scratch = vec![0u32; buf.len()];
    let result = if transformed.len() == 1 {
        try_full_walk_mask_with_table_from_initial_in_output_scope::<_, true>(
            state,
            vocab,
            trie,
            precollapse_master_decision,
            &transformed,
            lexer_scan_cache,
            &mut scratch,
            lazy_transitions,
            finalizer_code.unwrap_or(&[]),
            single_finalizer_continues.unwrap_or(&[]),
            initial_lexer_state,
            ignored_output,
        )
    } else {
        try_full_walk_mask_with_table_from_initial_in_output_scope::<_, false>(
            state,
            vocab,
            trie,
            precollapse_master_decision,
            &transformed,
            lexer_scan_cache,
            &mut scratch,
            lazy_transitions,
            finalizer_code.unwrap_or(&[]),
            single_finalizer_continues.unwrap_or(&[]),
            initial_lexer_state,
            ignored_output,
        )
    }?;
    if overflowed.get() {
        let mut cache = vocab.lock_lazy_union_cache();
        FullWalkLazyUnion::clear_cache(&mut cache);
        return Ok(false);
    }
    if !result {
        return Ok(false);
    }
    if profile {
        eprintln!("[glrmask/profile][scalar_dispatch] cache=lazy_walk_success");
    }
    buf.copy_from_slice(&scratch);
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_flat16<const HOT_SINGLE_ROOT: bool>(
    state: &ConstraintState,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
    transitions: &[u16],
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
) -> Result<bool, String> {
    // A token-start maximal-munch guard is an output filter over the whole
    // candidate model token. Carrying one pending guard inside a multi-root
    // deterministic walk prevents the pre-collapse master proof from applying
    // to every root and can turn an otherwise sliced broad mask into a full
    // vocabulary traversal. Factor each correlated root exactly as in the
    // scalar-dispatch path: evaluate the root with the guard marked Passed,
    // subtract the immutable blocked-token mask, then union root languages.
    if root_branches
        .iter()
        .any(|branch| !branch.initial_prune_guard.is_passed())
        && !joint_initial_guard_walk_enabled(
            root_branches,
            state.constraint.tokenizer.initial_state(),
        )
    {
        let profile = dynamic_mask_profile_enabled(state.generation);
        let mut merged = vec![0u32; buf.len()];
        let mut scratch = vec![0u32; buf.len()];
        for branch in root_branches {
            scratch.fill(0);
            let blocked_started = profile.then(std::time::Instant::now);
            let blocked = branch
                .initial_prune_guard
                .blocked_output_mask(&state.constraint, buf.len())?;
            if let Some(started) = blocked_started {
                eprintln!(
                    "[glrmask/profile][flat16_pending_filter] build_ms={:.3} words={}",
                    started.elapsed().as_secs_f64() * 1e3,
                    blocked.as_ref().map_or(0, |mask| mask.len()),
                );
            }

            let mut one = DynamicBranches::new();
            let mut unguarded = branch.clone();
            unguarded.initial_prune_guard = InitialPruneGuard::Passed;
            one.push(unguarded);
            let root_started = profile.then(std::time::Instant::now);
            if !try_flat16::<HOT_SINGLE_ROOT>(
                state,
                vocab,
                trie,
                &one,
                lexer_scan_cache,
                &mut scratch,
                transitions,
                finalizer_code,
                single_finalizer_continues,
            )? {
                return Ok(false);
            }
            if let Some(started) = root_started {
                eprintln!(
                    "[glrmask/profile][flat16_pending_filter] root_walk_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }
            if let Some(blocked) = blocked {
                for (word, &blocked) in scratch.iter_mut().zip(blocked.iter()) {
                    *word &= !blocked;
                }
            }
            for (dst, &word) in merged.iter_mut().zip(&scratch) {
                *dst |= word;
            }
        }
        buf.copy_from_slice(&merged);
        return Ok(true);
    }

    let profile_flat16 = dynamic_mask_flat16_profile_enabled();
    let proof_started = profile_flat16.then(std::time::Instant::now);
    // A single root cannot lose exact-source provenance through root
    // collapsing. Let the ordinary one-root proof below use the active dense
    // transition table directly before considering the heavier projected
    // quotient. Pre-collapse is only needed when multiple roots may be merged
    // into a coordinate that no longer identifies one exact source state.
    let precollapse_master_decision = (root_branches.len() >= 2
        && root_branches.iter().all(|branch| branch.initial_prune_guard.is_passed())
        && crate::env::var_os("GLRMASK_EXPERIMENT_DISABLE_PRECOLLAPSE_MASTER").is_none()).then(|| {
        precollapse_master_decision(
            state,
            vocab,
            root_branches,
            lexer_scan_cache.tokenizer().num_states() as usize,
        )
    }).flatten();
    if let Some(started) = proof_started {
        eprintln!(
            "[glrmask/profile][flat16_phases] proof_ms={:.3} proof_hit={}",
            started.elapsed().as_secs_f64() * 1e3,
            precollapse_master_decision.is_some(),
        );
    }
    // Preserve the ordinary trie until the inner profitability gate accepts
    // the pre-collapse certificate; see the scalar-dispatch path above.
    if lexer_scan_cache.subset_union_requested
        && root_branches.len() >= 2
        && root_branches.iter().all(|branch| branch.initial_prune_guard.is_passed())
        && root_branches
            .iter()
            .skip(1)
            .all(|branch| branch.gss.ptr_eq(&root_branches[0].gss))
    {
        let mut root_states = root_branches
            .iter()
            .map(|branch| branch.tokenizer_config)
            .collect::<Vec<_>>();
        root_states.sort_unstable();
        root_states.dedup();
        if root_states.len() >= 2 {
            let Some(subset_cache) = vocab.try_lock_lazy_union_cache() else {
                // The persistent lazy-subset cache is shared by all sequences
                // using this constraint. Never serialize concurrent mask fills
                // behind that derived acceleration state: contention falls back
                // to the ordinary exact multi-root walker.
                return try_full_walk_mask_with_table::<_, HOT_SINGLE_ROOT>(
                    state,
                    vocab,
                    trie,
                    precollapse_master_decision,
                    root_branches,
                    lexer_scan_cache,
                    buf,
                    FullWalkFlat16 { transitions },
                    finalizer_code,
                    single_finalizer_continues,
                );
            };
            let overflowed = std::cell::Cell::new(false);
            let tokenizer_owner = vocab.mask_runtime_tokenizer_owner()
                .unwrap_or(&state.constraint.tokenizer);
            if std::ptr::eq(tokenizer_owner.as_ref(), lexer_scan_cache.tokenizer()) {
            if let Some((subset_transitions, root_state)) = FullWalkLazyUnion::new(
                tokenizer_owner,
                Some(transitions),
                None,
                subset_cache,
                &root_states,
                &overflowed,
            ) {
                let mut collapsed = DynamicBranches::new();
                collapsed.push(DynamicBranch {
                    tokenizer_config: root_state,
                    exact_tokenizer_state: None,
                    parser_filtered_root: false,
                    parser_filtered_transparent: false,
                    residual_continuation_terminal: None,
                    gss: root_branches[0].gss.clone(),
                    shared_root_admission: root_branches[0].shared_root_admission.clone(),
                    initial_prune_guard: InitialPruneGuard::Passed,
                });
                let result = try_full_walk_mask_with_table::<_, true>(
                    state,
                    vocab,
                    trie,
                    precollapse_master_decision,
                    &collapsed,
                    lexer_scan_cache,
                    buf,
                    subset_transitions,
                    finalizer_code,
                    single_finalizer_continues,
                );
                if !overflowed.get() {
                    return result;
                }
                // Do not carry a saturated partial interner into the next
                // mask. The retry below does not use this cache, so clearing
                // here affects only future lazy-union attempts.
                {
                    let mut cache = vocab.lock_lazy_union_cache();
                    FullWalkLazyUnion::clear_cache(&mut cache);
                }
                // The lazy execution extension is deliberately bounded so
                // parser-node boundary caches can stay dense. If one mask
                // discovers more subset states than that execution reserve,
                // discard the partial mask and rerun through the ordinary
                // exact multi-root Flat16 walker instead of making the cache
                // capacity a correctness limit.
                return try_full_walk_mask_with_table::<_, HOT_SINGLE_ROOT>(
                    state,
                    vocab,
                    trie,
                    precollapse_master_decision,
                    root_branches,
                    lexer_scan_cache,
                    buf,
                    FullWalkFlat16 { transitions },
                    finalizer_code,
                    single_finalizer_continues,
                );
            }
            }
        }
    }
    let walk_started = profile_flat16.then(std::time::Instant::now);
    let result = try_full_walk_mask_with_table::<_, HOT_SINGLE_ROOT>(
        state,
        vocab,
        trie,
        precollapse_master_decision,
        root_branches,
        lexer_scan_cache,
        buf,
        FullWalkFlat16 { transitions },
        finalizer_code,
        single_finalizer_continues,
    );
    if let Some(started) = walk_started {
        eprintln!(
            "[glrmask/profile][flat16_phases] walk_ms={:.3}",
            started.elapsed().as_secs_f64() * 1e3,
        );
    }
    result
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_flat32<const HOT_SINGLE_ROOT: bool>(
    state: &ConstraintState,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
    transitions: &[u32],
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
) -> Result<bool, String> {
    try_full_walk_mask_with_table::<_, HOT_SINGLE_ROOT>(
        state,
        vocab,
        trie,
        None,
        root_branches,
        lexer_scan_cache,
        buf,
        FullWalkFlat32 { transitions },
        finalizer_code,
        single_finalizer_continues,
    )
}

/// Exact lexer-only identity evidence, separating ordinary and finalizing
/// self-edges for union certificates and continuing-component witnesses.
#[derive(Clone, Copy, Default)]
struct FullWalkIdentityByteClasses {
    ordinary: [u64; 4],
    finalizing: [u64; 4],
}

impl FullWalkIdentityByteClasses {
    fn intersect_union_member(&mut self, member: Self) {
        for word in 0..4 {
            let self_any = self.ordinary[word] | self.finalizing[word];
            let member_any = member.ordinary[word] | member.finalizing[word];
            self.ordinary[word] &= member.ordinary[word];
            self.finalizing[word] = (self_any & member_any) & !self.ordinary[word];
        }
    }
}

fn full_walk_identity_byte_classes(
    state: u32,
    mut transition: impl FnMut(u8) -> Option<(u32, bool)>,
) -> FullWalkIdentityByteClasses {
    let mut classes = FullWalkIdentityByteClasses::default();
    for byte in 0..=u8::MAX {
        let Some((target, finalizing)) = transition(byte) else { continue; };
        if target != state { continue; }
        let words = if finalizing { &mut classes.finalizing } else { &mut classes.ordinary };
        words[byte as usize >> 6] |= 1u64 << (byte & 63);
    }
    classes
}

#[derive(Default)]
struct FullWalkIdentityProofCache {
    rows: FxHashMap<u32, Option<FullWalkIdentityByteClasses>>,
}

/// A marker is irrelevant only when *every* original token alias is outside
/// this root's output responsibility. Looking at a canonical representative
/// alone would be unsound when aliases span multiple output words.
pub(super) fn full_walk_marker_fully_ignored(
    marker: u64,
    ignored: &[u32],
    aliases_ignored: impl FnOnce(u32) -> bool,
) -> bool {
    if marker & DYNAMIC_TOKEN_MARKER_FALLBACK == 0 {
        let word = (marker >> 32) as usize;
        let bits = marker as u32;
        return ignored.get(word).is_some_and(|value| value & bits == bits);
    }
    let Some(canonical) = (marker & !DYNAMIC_TOKEN_MARKER_FALLBACK)
        .checked_sub(1).and_then(|id| u32::try_from(id).ok())
    else {
        return false;
    };
    aliases_ignored(canonical)
}

/// Exact prefix counts in the actual walk's canonical endpoint order. This
/// index is local to one root mask, not keyed by parser states or persisted.
#[cfg(test)]
struct FullWalkOutputScope {
    required_prefix: Vec<u32>,
}

#[cfg(test)]
std::thread_local! {
    pub(super) static TEST_OUTPUT_SCOPE_SKIPPED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}


/// Lazy equivalent of the scope prefix index for preorder marker intervals.
/// The next responsible marker refutes every containing interval without
/// re-scanning it. As starts advance, each marker is inspected at most once;
/// markers already bypassed by an exact lexer/parser proof are never inspected.
/// Backward queries reset the certificate rather than assuming caller order.
#[derive(Default)]
pub(super) struct FullWalkOutputScopeCursor {
    last_start: usize,
    ignored_through: usize,
    next_required: Option<usize>,
}

impl FullWalkOutputScopeCursor {
    pub(super) fn covers(
        &mut self,
        markers: &[u64],
        start: usize,
        end: usize,
        mut is_ignored: impl FnMut(u64) -> bool,
    ) -> bool {
        if start > end || end > markers.len() { return false; }
        if start < self.last_start { *self = Self::default(); }
        self.last_start = start;
        if start == end { return true; }
        if let Some(required) = self.next_required {
            if required >= start { return required >= end; }
            self.next_required = None;
        }
        for index in start.max(self.ignored_through)..end {
            if !is_ignored(markers[index]) {
                self.ignored_through = index;
                self.next_required = Some(index);
                return false;
            }
        }
        self.ignored_through = self.ignored_through.max(end);
        true
    }
}

#[cfg(test)]
impl FullWalkOutputScope {
    fn new(markers: &[u64], mut is_ignored: impl FnMut(u64) -> bool) -> Option<Self> {
        const MAX_MARKERS: usize = 1 << 20;
        if markers.len() > MAX_MARKERS { return None; }
        let mut required_prefix = Vec::new();
        required_prefix.try_reserve_exact(markers.len().checked_add(1)?).ok()?;
        let mut required = 0u32;
        required_prefix.push(required);
        for &marker in markers {
            required += u32::from(!is_ignored(marker));
            required_prefix.push(required);
        }
        Some(Self { required_prefix })
    }

    fn covers(&self, start: usize, end: usize) -> bool {
        start <= end
            && self.required_prefix.get(start).zip(self.required_prefix.get(end))
                .is_some_and(|(left, right)| left == right)
    }
}

impl FullWalkIdentityProofCache {
    fn get<T: FullWalkTransitionTable>(
        &mut self,
        transitions: &T,
        tokenizer: &Tokenizer,
        lexer: u32,
    ) -> Option<FullWalkIdentityByteClasses> {
        if let Some(&classes) = self.rows.get(&lexer) { return classes; }
        // At most 64 local row proofs, with no persistent or serialized state.
        if self.rows.len() >= 64 { return None; }
        let classes = transitions.proven_identity_byte_classes(tokenizer, lexer);
        self.rows.insert(lexer, classes);
        classes
    }
}


/// An alphabet of exact self-loops is closed under concatenation. When every
/// byte in a vocabulary subtree belongs to it, all descendant token endpoints
/// see exactly the current full parser/lexer/guard state. Unseen bytes never
/// qualify; this is not a character-class or language approximation.
#[inline]
fn identity_alphabet_covers_subtree(identity: &[u64; 4], subtree: [u64; 4]) -> bool {
    subtree.iter().zip(identity).all(|(&needed, &known)| needed & !known == 0)
}

/// A lower-bound hit proves true; an upper-bound miss proves false.
/// The unresolved middle must use the existing exact stack simulation.
#[inline(always)]
#[cfg(test)]
fn full_walk_row_liveness_bound(
    guaranteed_live: bool,
    necessary_live: impl FnOnce() -> bool,
) -> Option<bool> {
    if guaranteed_live {
        Some(true)
    } else if !necessary_live() {
        Some(false)
    } else {
        None
    }
}

#[cfg(test)]
mod boundary_direct_cache_tests {
    use super::FullWalkBoundaryDirectCache;

    #[test]
    fn full_tags_and_parser_rows_prevent_false_cache_hits() {
        let mut cache = FullWalkBoundaryDirectCache::new();
        cache.push_row(); cache.push_row();
        assert_eq!(cache.get(0, 0), 0);
        cache.set(0, 0, 2); cache.set(1, 0, 1);
        assert_eq!(cache.get(0, 0), 2); assert_eq!(cache.get(1, 0), 1);
        cache.set(0, 16, 1);
        assert_eq!(cache.get(0, 0), 0); assert_eq!(cache.get(0, 16), 1);
        assert_eq!(cache.get(1, 0), 1);
        cache.set(0, u32::MAX, 2);
        assert_eq!(cache.get(0, u32::MAX), 2);
        assert_eq!(cache.get(0, 15), 0);
    }

    #[test]
    fn dense_promotion_preserves_all_retained_cells_and_unknowns() {
        for width in [17, 63, 129, 257, 1025] {
            let mut cache = FullWalkBoundaryDirectCache::new();
            let mut rows = vec![Vec::new(); 23];
            for node in 0..rows.len() {
                cache.push_row();
                for lexer in 0..width as u32 {
                    cache.set(node, lexer, 1 + ((node as u32 + lexer) % 2) as u8);
                }
            }
            let expected: Vec<Vec<u8>> = (0..rows.len()).map(|node|
                (0..width as u32).map(|lexer| cache.get(node, lexer)).collect()).collect();
            cache.expand_into(&mut rows, width);
            assert_eq!(rows, expected);
            assert!(rows.iter().all(|row| row.len() == width));
        }
    }
}

/// Exact direct dynamic-mask path for bounded deterministic lexer coordinates.
///
/// This performs the exact vocabulary walk, skipping only a child subtree once
/// the current lexer branch is already proven dead. It does not use subtree
/// certificates, segment-effect caches, recognizer-state interning, or any
/// speculative admission rule. Unsupported lexer or composition shapes return
/// `false` and use the existing exact fallback.
fn residual_regex_slice_prefix_contained(
    tokenizer: &Tokenizer,
    source: u32,
    terminal: TerminalID,
    slice: &crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa,
    work_limit: usize,
) -> Option<bool> {
    let coordinates = tokenizer.terminal_residual_coordinates()?;
    let residual_state = coordinates
        .row(source)?
        .iter()
        .find_map(|&(candidate, state)| (candidate == terminal).then_some(state))?;
    let residual = coordinates.terminal_dfa(terminal)?;
    let mut seen = FxHashSet::<(u32, u32)>::default();
    let mut queue = std::collections::VecDeque::from([(slice.start_state(), residual_state)]);
    let mut work = 0usize;
    while let Some((slice_state, regex_state)) = queue.pop_front() {
        if !seen.insert((slice_state, regex_state)) {
            continue;
        }
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let slice_target = slice.step(slice_state, byte);
            if !slice.can_reach_accepting(slice_target) {
                continue;
            }
            work = work.saturating_add(1);
            if work > work_limit {
                return None;
            }
            let Some(regex_target) = residual.step(regex_state, byte) else {
                return Some(false);
            };
            let regex_live = residual.finalizers(regex_target).contains(0)
                || residual.possible_future_group_ids(regex_target).contains(0);
            if !regex_live {
                return Some(false);
            }
            if !seen.contains(&(slice_target, regex_target)) {
                queue.push_back((slice_target, regex_target));
            }
        }
    }
    Some(true)
}


const LLG_SAFE_PLUS_SLICE: usize = 0;
const LLG_WHITESPACE_SLICE: usize = 3;
// Proof cache IDs 0 and 3 are intentionally stable. Slot count only needs to
// cover the largest stable ID; there are no fixed bounded-length proof slots.
const LLG_PROOF_SLOT_COUNT: usize = LLG_WHITESPACE_SLICE + 1;

pub(super) fn virtual_residual_slice_prefix_contained(
    tokenizer: &Tokenizer,
    source: u32,
    terminal: TerminalID,
    slice: &crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa,
    work_limit: usize,
) -> Option<bool> {
    let mut found = false;
    let mut unknown = false;
    for residual_state in tokenizer.singleton_epsilon_closure(source) {
        if tokenizer.virtual_residual_terminal_for_state(residual_state) != Some(terminal) {
            continue;
        }
        found = true;
        match tokenizer.virtual_residual_parser_transparent_byte_dfa(
            residual_state,
            slice.start_state(),
            slice.class_count(),
            slice.byte_to_class_map(),
            slice.transition_table(),
            slice.can_reach_accepting_map(),
            slice.has_finite_language(),
            work_limit,
        ) {
            Some(true) => return Some(true),
            Some(false) => {}
            None => unknown = true,
        }
    }
    if !found || unknown {
        None
    } else {
        Some(false)
    }
}

pub(super) fn virtual_residual_safe_repeat_radius(
    tokenizer: &Tokenizer,
    source: u32,
    terminal: TerminalID,
    safe_plus: &crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa,
    max_repetitions: u32,
    work_limit: usize,
) -> Option<u32> {
    let mut best = None;
    let mut found = false;
    for residual_state in tokenizer.singleton_epsilon_closure(source) {
        if tokenizer.virtual_residual_terminal_for_state(residual_state) != Some(terminal) {
            continue;
        }
        found = true;
        if let Some(radius) = tokenizer.virtual_residual_parser_transparent_byte_dfa_repeat_radius(
            residual_state,
            safe_plus.start_state(),
            safe_plus.class_count(),
            safe_plus.byte_to_class_map(),
            safe_plus.transition_table(),
            safe_plus.accepting_map(),
            safe_plus.can_reach_accepting_map(),
            max_repetitions,
            work_limit,
        ) {
            best = Some(best.map_or(radius, |current: u32| current.max(radius)));
        }
    }
    found.then_some(best).flatten()
}
fn direct_slice_prefix_contained<T: FullWalkTransitionTable>(
    transitions: &T,
    tokenizer: &Tokenizer,
    start: u32,
    terminal: TerminalID,
    slice: &crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa,
    work_limit: usize,
) -> Option<bool> {
    let mut seen = FxHashSet::<(u32, u32)>::default();
    let mut queue = std::collections::VecDeque::from([(slice.start_state(), start)]);
    let mut work = 0usize;
    while let Some((slice_state, lexer_state)) = queue.pop_front() {
        if !seen.insert((slice_state, lexer_state)) {
            continue;
        }
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let slice_target = slice.step(slice_state, byte);
            if !slice.can_reach_accepting(slice_target) {
                continue;
            }
            work = work.saturating_add(1);
            if work > work_limit {
                return None;
            }
            let cell = transitions.cell(lexer_state, byte);
            if T::cell_is_dead(cell) {
                return Some(false);
            }
            let target = T::cell_target(cell);
            let terminal_live = transitions.future_contains(tokenizer, target, terminal)
                || transitions
                    .matched_terminals(tokenizer, target)
                    .contains(&terminal);
            if !terminal_live {
                return Some(false);
            }
            if !seen.contains(&(slice_target, target)) {
                queue.push_back((slice_target, target));
            }
        }
    }
    Some(true)
}

fn debug_direct_slice_prefix_counterexample<T: FullWalkTransitionTable>(
    transitions: &T,
    tokenizer: &Tokenizer,
    start: u32,
    terminal: TerminalID,
    slice: &crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa,
    work_limit: usize,
) -> (Option<bool>, Vec<u8>, &'static str) {
    let mut seen = FxHashSet::<(u32, u32)>::default();
    let mut queue = std::collections::VecDeque::from([(
        slice.start_state(),
        start,
        Vec::<u8>::new(),
    )]);
    let mut work = 0usize;
    while let Some((slice_state, lexer_state, path)) = queue.pop_front() {
        if !seen.insert((slice_state, lexer_state)) {
            continue;
        }
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let slice_target = slice.step(slice_state, byte);
            if !slice.can_reach_accepting(slice_target) {
                continue;
            }
            work += 1;
            if work > work_limit {
                return (None, path, "budget");
            }
            let mut next_path = path.clone();
            next_path.push(byte);
            let cell = transitions.cell(lexer_state, byte);
            if T::cell_is_dead(cell) {
                return (Some(false), next_path, "dead-transition");
            }
            let target = T::cell_target(cell);
            let terminal_live = transitions.future_contains(tokenizer, target, terminal)
                || transitions
                    .matched_terminals(tokenizer, target)
                    .contains(&terminal);
            if !terminal_live {
                return (Some(false), next_path, "terminal-not-live");
            }
            if next_path.len() < 96 && !seen.contains(&(slice_target, target)) {
                queue.push_back((slice_target, target, next_path));
            }
        }
    }
    (Some(true), Vec::new(), "contained")
}


#[inline]
fn is_transparent_json_string_chunk_terminal(constraint: &Constraint, terminal: TerminalID) -> bool {
    constraint
        .terminal_display_names
        .get(terminal as usize)
        .is_some_and(|name| name.starts_with("json_string_char_"))
}

fn transparent_json_string_terminals(
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    parser_node: u32,
) -> BitSet {
    let mut transparent = parser_cache.admitted(constraint, parser_node).clone();
    let terminals = transparent.iter_ones().collect::<Vec<_>>();
    for terminal in terminals {
        if !is_transparent_json_string_chunk_terminal(constraint, terminal as TerminalID) {
            transparent.clear(terminal);
        }
    }
    transparent
}

/// Exact bounded containment through GLRMask's implementation-level JSON-string
/// chunk terminals. Long bounded semantic strings are deliberately lowered into
/// a sequence of `json_string_char_*` parser terminals. llguidance keeps this as
/// one lexeme, so a faithful slice proof must allow those internal finalizations
/// and parser advances while refusing every unrelated grammar terminal.
fn bounded_string_chunk_slice_contained<T: FullWalkTransitionTable>(
    transitions: &T,
    tokenizer: &Tokenizer,
    start_lexer_state: u32,
    reset_lexer_state: u32,
    initial_parser_node: u32,
    initial_guard: FullWalkPruneGuard,
    slice: &crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa,
    finalizer_code: &[u32],
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    work_limit: usize,
) -> Option<bool> {
    const NONE: u32 = u32::MAX;
    const MULTI: u32 = u32::MAX - 1;

    let mut start_branches = FullWalkBranches::new();
    start_branches.push(FullWalkBranch {
        lexer_state: start_lexer_state,
        parser_node: initial_parser_node,
        prune_guard: initial_guard,
    });
    let mut queue = std::collections::VecDeque::from([(
        slice.start_state(),
        start_branches,
    )]);
    let mut seen = FxHashSet::<(u32, FullWalkBranches)>::default();
    let mut work = 0usize;

    while let Some((slice_state, mut branches)) = queue.pop_front() {
        branches.sort_unstable();
        branches.dedup();
        if !seen.insert((slice_state, branches.clone())) {
            continue;
        }

        for byte in 0u16..=255 {
            let byte = byte as u8;
            let slice_target = slice.step(slice_state, byte);
            if !slice.can_reach_accepting(slice_target) {
                continue;
            }
            work = work.saturating_add(1);
            if work > work_limit {
                return None;
            }

            let mut next = FullWalkBranches::new();
            for branch in &branches {
                let Some(advanced_guard) = branch.prune_guard.advance(tokenizer, transitions, byte)
                else {
                    continue;
                };
                let target = transitions.transition(branch.lexer_state, byte);
                if target == u32::MAX {
                    continue;
                }

                let code = transitions.finalizer_code(target, finalizer_code);
                if code == MULTI {
                    for terminal in transitions.matched_terminals(tokenizer, target) {
                        if !is_transparent_json_string_chunk_terminal(constraint, terminal) {
                            continue;
                        }
                        if !parser_cache.terminal_not_known_inadmissible(
                            constraint,
                            branch.parser_node,
                            terminal,
                        ) {
                            continue;
                        }
                        if let Some(parser_node) =
                            parser_cache.advance(constraint, branch.parser_node, terminal)
                        {
                            full_walk_push_unique(
                                &mut next,
                                FullWalkBranch {
                                    lexer_state: reset_lexer_state,
                                    parser_node,
                                    prune_guard: advanced_guard.remember_terminal_match(
                                        tokenizer,
                                        transitions,
                                        target,
                                        terminal,
                                    ),
                                },
                            );
                        }
                    }
                } else if code != NONE
                    && is_transparent_json_string_chunk_terminal(constraint, code)
                    && parser_cache.terminal_not_known_inadmissible(
                        constraint,
                        branch.parser_node,
                        code,
                    )
                    && let Some(parser_node) =
                        parser_cache.advance(constraint, branch.parser_node, code)
                {
                    full_walk_push_unique(
                        &mut next,
                        FullWalkBranch {
                            lexer_state: reset_lexer_state,
                            parser_node,
                            prune_guard: advanced_guard.remember_terminal_match(
                                tokenizer,
                                transitions,
                                target,
                                code,
                            ),
                        },
                    );
                }

                // Continue the current lexical branch only while some parser-
                // admitted transparent string-chunk terminal can still complete.
                let transparent = transparent_json_string_terminals(
                    parser_cache,
                    constraint,
                    branch.parser_node,
                );
                if !transparent.is_empty()
                    && transitions.future_intersects(tokenizer, target, &transparent)
                {
                    full_walk_push_unique(
                        &mut next,
                        FullWalkBranch {
                            lexer_state: target,
                            parser_node: branch.parser_node,
                            prune_guard: advanced_guard,
                        },
                    );
                }
            }

            if next.is_empty() {
                return Some(false);
            }
            next.sort_unstable();
            next.dedup();
            if !seen.contains(&(slice_target, next.clone())) {
                queue.push_back((slice_target, next));
            }
        }
    }
    Some(true)
}

#[inline(never)]
fn try_full_walk_mask_with_table<T: FullWalkTransitionTable, const HOT_SINGLE_ROOT: bool>(
    state: &ConstraintState,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    llg_master_decision: Option<LlgMasterDecision>,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
    transitions: T,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
) -> Result<bool, String> {
    debug_assert!(trie.full_walk_max_parent_depth() < 255);

    let initial_lexer_state = lexer_scan_cache
        .config_for_raw_start(vocab.mask_runtime_state(state.constraint.tokenizer.initial_state()))?;

    try_full_walk_mask_with_table_from_initial::<T, HOT_SINGLE_ROOT>(
        state,
        vocab,
        trie,
        llg_master_decision,
        root_branches,
        lexer_scan_cache,
        buf,
        transitions,
        finalizer_code,
        single_finalizer_continues,
        initial_lexer_state,
    )
}

#[inline(never)]
fn try_full_walk_mask_with_table_from_initial<
    T: FullWalkTransitionTable,
    const HOT_SINGLE_ROOT: bool,
>(
    state: &ConstraintState,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    llg_master_decision: Option<LlgMasterDecision>,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
    transitions: T,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    initial_lexer_state: u32,
) -> Result<bool, String> {
    try_full_walk_mask_with_table_from_initial_in_output_scope::<T, HOT_SINGLE_ROOT>(
        state, vocab, trie, llg_master_decision, root_branches, lexer_scan_cache,
        buf, transitions, finalizer_code, single_finalizer_continues,
        initial_lexer_state, None,
    )
}

/// Dense rows supply the same lexical relation and exact parser queries as
/// the other providers. Their synthetic coordinates never escape this adapter.
struct DenseProductTransitions<'a, T> {
    table: &'a T,
    tokenizer: &'a Tokenizer,
    vocab: &'a DynamicMaskVocab,
    finalizers: &'a [u32],
    continuing: &'a [u8],
    ignored_output: Option<&'a [u32]>,
    prune_prefixes: bool,
    proofs: FullWalkIdentityProofCache,
    witness_enabled: bool,
}

impl<T: FullWalkTransitionTable> super::FullWalkTransitionTable for DenseProductTransitions<'_, T> {
    type Cell = T::Cell;
    #[inline(always)]
    fn cell(&mut self, state: u32, byte: u8) -> Self::Cell { self.table.cell(state, byte) }
    #[inline(always)]
    fn cell_is_dead(cell: Self::Cell) -> bool { T::cell_is_dead(cell) }
    #[inline(always)]
    fn cell_has_finalizer(cell: Self::Cell) -> bool { T::cell_has_finalizer(cell) }
    #[inline(always)]
    fn cell_target(cell: Self::Cell) -> u32 { T::cell_target(cell) }
    fn root_state(&mut self, state: u32) -> Result<u32, String> { Ok(state) }
    fn finalizer_code(&self, state: u32) -> u32 { self.table.finalizer_code(state, self.finalizers) }
    fn single_finalizer_continues(&mut self, state: u32) -> bool {
        self.table.single_finalizer_continues(state, self.continuing)
    }
    fn matched_terminals(&self, state: u32) -> SmallVec<[TerminalID; 4]> {
        self.table.matched_terminals(self.tokenizer, state)
    }
    fn future_contains(&mut self, state: u32, terminal: TerminalID) -> bool {
        self.table.future_contains(self.tokenizer, state, terminal)
    }
    fn future_intersects(&mut self, state: u32, terminals: &BitSet) -> bool {
        self.table.future_intersects(self.tokenizer, state, terminals)
    }
    fn merge_states(&mut self, states: &[u32]) -> Option<u32> {
        self.table.union_states(states)
            .or_else(|| self.vocab.mask_projection_state_for_projection_states(states))
    }
    fn dense_state_count(&self) -> Option<usize> { Some(self.table.state_count(self.tokenizer)) }
    fn token_boundary_allowed(
        &mut self, cache: &mut super::FullWalkParserCache, constraint: &Constraint,
        initial: u32, lexer: u32, parser: u32,
    ) -> bool {
        lexer == initial || cache.physical_token_boundary_allowed_table(constraint, self, parser, lexer)
    }
    fn exact_raw_state(&self, _state: u32) -> Option<u32> { None }
    fn exact_next_bytes(&mut self, state: u32) -> Option<U8Set> {
        if state >= self.tokenizer.num_states() { return None; }
        let mut bytes = U8Set::empty();
        for byte in 0..=255u8 {
            if !T::cell_is_dead(self.table.cell(state, byte)) { bytes.insert(byte); }
        }
        Some(bytes)
    }
    fn parser_conditioned_dead_skip_default(&self) -> bool { self.prune_prefixes }
    fn pre_admitted_original_mask(&self) -> Option<&[u32]> { self.ignored_output }
    fn product_transition_cache_capacity(&self) -> usize {
        if full_walk_acceleration_enabled() { 1024 } else { 0 }
    }
    fn memoize_scalar_products(&self) -> bool { false }
    fn proves_live_subtree(
        &mut self, product: &super::FullWalkProduct, alphabet: [u64; 4],
        cache: &mut super::FullWalkParserCache, constraint: &Constraint,
    ) -> bool {
        if !self.witness_enabled || product.len() > 8 { return false; }
        for &(lexer, parser) in &product.plain {
            for member in self.table.continuation_witness_components(lexer) {
                let Some(classes) = self.proofs.get(self.table, self.tokenizer, member) else { continue; };
                let loops = std::array::from_fn(|i| classes.ordinary[i] | classes.finalizing[i]);
                if identity_alphabet_covers_subtree(&loops, alphabet)
                    && cache.physical_token_boundary_allowed_table(constraint, self, parser, member)
                {
                    return true;
                }
            }
        }
        false
    }
}
#[inline(never)]
fn try_full_walk_mask_with_table_from_initial_in_output_scope<
    T: FullWalkTransitionTable,
    const HOT_SINGLE_ROOT: bool,
>(
    state: &ConstraintState,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    llg_master_decision: Option<LlgMasterDecision>,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
    transitions: T,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    initial_lexer_state: u32,
    ignored_output: Option<&[u32]>,
) -> Result<bool, String> {
    debug_assert!(trie.full_walk_max_parent_depth() < 255);

    let profile_walk = dynamic_mask_profile_enabled(state.generation);
    let profile_kernel = crate::env::var("GLRMASK_PROFILE_DYNAMIC_KERNEL_GENERATION")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        == Some(state.generation);
    

    let (mut parser_cache, root_parser_nodes) = super::FullWalkParserCache::from_roots(
        root_branches,
        Some(transitions.state_count(lexer_scan_cache.tokenizer())),
    );
    if profile_walk {
        for (root_index, &parser_node) in root_parser_nodes.iter().enumerate() {
            let lexer_state = root_branches[root_index].tokenizer_config;
            let tokenizer = lexer_scan_cache.tokenizer();
            if lexer_state < tokenizer.num_states() {
                let matched = tokenizer
                    .matched_terminal_bitset(lexer_state)
                    .iter_ones()
                    .map(|terminal| terminal as TerminalID)
                    .collect::<Vec<_>>();
                let futures = tokenizer
                    .possible_future_terminals(lexer_state)
                    .iter_ones()
                    .map(|terminal| terminal as TerminalID)
                    .collect::<Vec<_>>();
                eprintln!(
                    "[glrmask/profile][root_lexer_observation] generation={} root={} lexer={} matched={:?} futures={:?}",
                    state.generation,
                    root_index,
                    lexer_state,
                    matched,
                    futures,
                );
            } else {
                eprintln!(
                    "[glrmask/profile][root_lexer_observation] generation={} root={} lexer={} subset_extension=true",
                    state.generation,
                    root_index,
                    lexer_state,
                );
            }
            let admitted = parser_cache.admitted(&state.constraint, parser_node);
            let ids = admitted
                .iter_ones()
                .map(|terminal| terminal as TerminalID)
                .collect::<Vec<_>>();
            eprintln!(
                "[glrmask/profile][root_admitted_terminals] generation={} root={} count={} ids={:?}",
                state.generation,
                root_index,
                ids.len(),
                ids,
            );
            for &terminal in &ids {
                if let Some(expr) = state.constraint.retained_terminal_expr(terminal) {
                    eprintln!(
                        "[glrmask/profile][root_admitted_terminal_expr] generation={} root={} terminal={} expr={:?}",
                        state.generation,
                        root_index,
                        terminal,
                        expr,
                    );
                }
            }
        }
    }

    // With no pending terminal-exclusion guard, a zero-initialized mask is an
    // exact representation for the common singleton-parser frontier: the walk
    // can set surviving token endpoints instead of starting full and clearing
    // every dead subtree. Pending prune guards are deliberately excluded here
    // because their accumulator correlation is not represented by the plain
    // parser-admission singleton test.
    static SINGLETON_POSITIVE_REBUILD_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let singleton_positive_rebuild = *SINGLETON_POSITIVE_REBUILD_ENABLED
        .get_or_init(|| crate::env::var_os("GLRMASK_DISABLE_SINGLETON_POSITIVE_REBUILD").is_none())
        && state.constraint.ignore_terminal.is_none()
        && root_branches
            .iter()
            .all(|branch| branch.initial_prune_guard.is_passed())
        && {
            let mut only_terminal = None::<TerminalID>;
            let mut singleton = true;
            for &parser_node in &root_parser_nodes {
                let mut admitted = parser_cache
                    .admitted(&state.constraint, parser_node)
                    .iter_ones()
                    .map(|terminal| terminal as TerminalID);
                let Some(terminal) = admitted.next() else {
                    singleton = false;
                    break;
                };
                if admitted.next().is_some() {
                    singleton = false;
                    break;
                }
                match only_terminal {
                    None => only_terminal = Some(terminal),
                    Some(existing) if existing == terminal => {}
                    Some(_) => {
                        singleton = false;
                        break;
                    }
                }
            }
            singleton && only_terminal.is_some()
        };
    // The parser row can admit many terminals even when the current lexer
    // residual can produce only one of them. In that common narrow-mask shape,
    // requiring parser admission itself to be singleton needlessly forces the
    // negative-polarity path. Use the exact walk coordinate's sole live
    // terminal instead, then verify that terminal against parser admission.
    // This is purely an output-polarity choice: the semantic walk is unchanged.
    static EFFECTIVE_SINGLETON_POSITIVE_REBUILD_ENABLED: std::sync::OnceLock<bool> =
        std::sync::OnceLock::new();
    let effective_singleton_positive_rebuild = *EFFECTIVE_SINGLETON_POSITIVE_REBUILD_ENABLED
        .get_or_init(|| {
            crate::env::var_os("GLRMASK_DISABLE_EFFECTIVE_SINGLETON_POSITIVE_REBUILD").is_none()
        })
        && root_branches
            .iter()
            .all(|branch| branch.initial_prune_guard.is_passed())
        && {
            let tokenizer = lexer_scan_cache.tokenizer();
            let mut only_terminal = None::<TerminalID>;
            let mut singleton = true;
            for (root_index, root) in root_branches.iter().enumerate() {
                let lexer_state = root.tokenizer_config;
                let admitted = parser_cache.admitted(&state.constraint, root_parser_nodes[root_index]);
                let Some(root_terminal) = transitions.sole_live_terminal(tokenizer, lexer_state)
                else {
                    singleton = false;
                    break;
                };
                if Some(root_terminal) != state.constraint.ignore_terminal
                    && !admitted.contains(root_terminal as usize)
                {
                    singleton = false;
                    break;
                }
                match only_terminal {
                    None => only_terminal = Some(root_terminal),
                    Some(existing) if existing == root_terminal => {}
                    Some(_) => {
                        singleton = false;
                        break;
                    }
                }
            }
            singleton && only_terminal.is_some()
        };
    // The pre-collapse Flat16 path can hand us a master certificate directly.
    // Other deterministic one-root paths still have the same exact lexer/parser
    // information. Before running an exact proof, filter admitted terminals by
    // the necessary byte-support condition for the safe+ language. This keeps
    // the cold proof off ordinary narrow terminals even when a parser state has
    // many alternatives, while broad string terminals remain eligible.
    let mut llg_master_decision = llg_master_decision;
    static GENERIC_MASTER_PROOF_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let generic_master_proof_enabled = *GENERIC_MASTER_PROOF_ENABLED.get_or_init(|| {
        crate::env::var_os("GLRMASK_DISABLE_GENERIC_MASTER_PROOF").is_none()
    });
    let generic_master_candidates = if generic_master_proof_enabled
        && llg_master_decision.is_none()
        && HOT_SINGLE_ROOT
        && root_branches.len() == 1
        && root_branches[0].initial_prune_guard.is_passed()
        && vocab.llg_master_trie().is_some()
    {
        vocab
            .llg_slice_by_cache_id(LLG_SAFE_PLUS_SLICE as u32)
            .and_then(|safe_plus| {
                let admitted = parser_cache.admitted(&state.constraint, root_parser_nodes[0]);
                let mut candidates = SmallVec::<[TerminalID; 4]>::new();
                let lexer_state = root_branches[0].tokenizer_config;
                let filter_first_bytes =
                    crate::env::var_os("GLRMASK_DISABLE_GENERIC_MASTER_FIRST_BYTES").is_none();
                let first_bytes = filter_first_bytes.then(|| safe_plus.first_bytes());
                let exact_source = root_branches[0].exact_tokenizer_state;
                for terminal in admitted.iter_ones().map(|terminal| terminal as TerminalID) {
                    let lexer_live = transitions.future_contains(
                        lexer_scan_cache.tokenizer(),
                        lexer_state,
                        terminal,
                    ) || transitions
                        .matched_terminals(lexer_scan_cache.tokenizer(), lexer_state)
                        .contains(&terminal);
                    if lexer_live
                        && state
                            .constraint
                            .tokenizer
                            .terminal_byte_support(terminal)
                            .is_some_and(|support| safe_plus.slice_token_bytes().is_subset(&support))
                        // A wide parser row need not imply a wide proof search.
                        // Use the exact source's cheap first-byte necessary
                        // condition BEFORE applying the candidate-count limit.
                        // Unknown/virtual sources remain conservative candidates.
                        && !first_bytes.zip(exact_source).is_some_and(|(bytes, source)| {
                            state.constraint.tokenizer.physical_terminal_residual_covers_first_bytes(
                                source, terminal, bytes,
                            ) == Some(false)
                        })
                    {
                        candidates.push(terminal);
                    }
                }
                // The old <=2 gate protects the expensive wholly-online proof
                // path. Once this source has prepared master-prover rows, the
                // expensive batch work has already been done and the remaining
                // exact direct/quotient fallbacks are the validated wide path.
                // Keep the conservative limit for sources with no prepared row.
                let prepared_wide = root_branches[0]
                    .exact_tokenizer_state
                    .is_some_and(|source| vocab.has_prepared_master_prover_row(source));
                let candidate_limit = if prepared_wide { 8 } else { 2 };
                if crate::env::var_os("GLRMASK_DIAG_CANDIDATE_GATES").is_some() {
                    eprintln!("[candidate_gate] generation={} lexer={} exact={:?} candidates={} limit={} first_bytes_gate={}",
                        state.generation, lexer_state, exact_source, candidates.len(), candidate_limit, filter_first_bytes);
                }
                (!candidates.is_empty() && candidates.len() <= candidate_limit)
                    .then_some(candidates)
            })
    } else {
        None
    };
    if let Some(safe_plus_candidates) = generic_master_candidates {
        let parser_node = root_parser_nodes[0];
        let lexer_state = root_branches[0].tokenizer_config;
        let exact_source = root_branches[0].exact_tokenizer_state;
        let admitted = parser_cache.admitted(&state.constraint, parser_node).clone();
        let safe_plus = vocab
            .llg_slice_by_cache_id(LLG_SAFE_PLUS_SLICE as u32)
            .expect("dynamic-radius master requires safe+ proof DFA");
        let whitespace = vocab
            .llg_slice_by_cache_id(LLG_WHITESPACE_SLICE as u32)
            .expect("dynamic-radius master requires whitespace proof DFA");
        let whitespace_candidates = admitted
            .iter_ones()
            .map(|terminal| terminal as TerminalID)
            .filter(|&terminal| {
                state
                    .constraint
                    .tokenizer
                    .terminal_byte_support(terminal)
                    .is_some_and(|support| whitespace.slice_token_bytes().is_subset(&support))
            })
            .collect::<SmallVec<[TerminalID; 8]>>();

        let prove_slice = |
            slice: &crate::runtime::artifact::DynamicMaskSliceTrie,
            terminals: &[TerminalID],
        | -> bool {
            terminals.iter().copied().any(|terminal| {
                if let Some(source) = exact_source {
                    let prepared_slot = if slice.cache_id() == LLG_SAFE_PLUS_SLICE as u32 {
                        Some(0usize)
                    } else if slice.cache_id() == LLG_WHITESPACE_SLICE as u32 {
                        Some(1usize)
                    } else {
                        None
                    };
                    if let Some(prepared) = prepared_slot.and_then(|slot| {
                        vocab.prepared_master_proof_result(source, slot, terminal)
                    }) {
                        return prepared;
                    }
                }
                let live = transitions.future_contains(
                    lexer_scan_cache.tokenizer(),
                    lexer_state,
                    terminal,
                ) || transitions
                    .matched_terminals(lexer_scan_cache.tokenizer(), lexer_state)
                    .contains(&terminal);
                if !live {
                    return false;
                }
                if let Some(cached) = vocab.cached_direct_slice_contained(
                    terminal,
                    lexer_state,
                    slice.cache_id(),
                ) {
                    return cached;
                }
                // The active transition table is already an exact mask-runtime
                // coordinate. For one-root states, try the source-local
                // containment proof before materializing any terminal quotient.
                // A positive result is definitive; a negative/unknown result
                // merely falls through to the existing exact quotient and
                // symbolic proofs.
                if direct_slice_prefix_contained(
                    &transitions,
                    lexer_scan_cache.tokenizer(),
                    lexer_state,
                    terminal,
                    slice.dfa(),
                    32 * 1024,
                ) == Some(true)
                {
                    vocab.cache_direct_slice_contained(
                        terminal,
                        lexer_state,
                        slice.cache_id(),
                        true,
                    );
                    return true;
                }
                if let Some(quotient) = exact_source.and_then(|source| {
                    vocab.projected_terminal_slice_contained(
                        terminal,
                        source,
                        slice.cache_id(),
                        slice.dfa(),
                    )
                }) {
                    vocab.cache_direct_slice_contained(
                        terminal,
                        lexer_state,
                        slice.cache_id(),
                        quotient,
                    );
                    return quotient;
                }
                let symbolic = exact_source.and_then(|source| {
                    virtual_residual_slice_prefix_contained(
                        &state.constraint.tokenizer,
                        source,
                        terminal,
                        slice.dfa(),
                        1_536,
                    )
                });
                let proved = symbolic == Some(true);
                vocab.cache_direct_slice_contained(
                    terminal,
                    lexer_state,
                    slice.cache_id(),
                    proved,
                );
                proved
            })
        };

        let compute_safe_radius = || -> Option<u16> {
            let source = exact_source?;
            let max_vocab_safe_chars = u32::from(vocab.llg_master_max_safe_chars());
            Some(
                safe_plus_candidates
                    .iter()
                    .copied()
                    .filter_map(|terminal| {
                        let projected = vocab
                            .prepared_safe_radius(source, terminal)
                            .map(u32::from)
                            .or_else(|| {
                                vocab.projected_terminal_slice_repeat_radius(
                                    terminal,
                                    source,
                                    safe_plus.cache_id(),
                                    safe_plus.dfa(),
                                    max_vocab_safe_chars,
                                    16 * 1024,
                                )
                            });
                        let symbolic = virtual_residual_safe_repeat_radius(
                            &state.constraint.tokenizer,
                            source,
                            terminal,
                            safe_plus.dfa(),
                            max_vocab_safe_chars,
                            16 * 1024,
                        );
                        match (projected, symbolic) {
                            (Some(left), Some(right)) => Some(left.max(right)),
                            (left, right) => left.or(right),
                        }
                    })
                    .filter_map(|radius| u16::try_from(radius).ok())
                    .max()
                    .unwrap_or(0),
            )
        };
        let prefer_bounded_radius = crate::env::var_os("GLRMASK_DISABLE_VIRTUAL_RADIUS_FIRST").is_none()
            && exact_source.is_some_and(|source| {
            safe_plus_candidates.iter().copied().any(|terminal| {
                state
                    .constraint
                    .tokenizer
                    .singleton_epsilon_closure(source)
                    .into_iter()
                    .any(|residual_state| {
                        state
                            .constraint
                            .tokenizer
                            .virtual_residual_terminal_for_state(residual_state)
                            == Some(terminal)
                    })
            })
        });
        let radius_before_unbounded = prefer_bounded_radius.then(compute_safe_radius).flatten();
        // For vocabulary masking an exact positive bounded radius is already a
        // complete certificate for every precomputed safe-slice token within
        // that radius. Do not first pay to prove the strictly stronger
        // unbounded safe+ language when the bounded certificate is available.
        let safe_plus_proved = if radius_before_unbounded.is_some_and(|radius| radius != 0) {
            false
        } else {
            prove_slice(safe_plus, safe_plus_candidates.as_slice())
        };
        let safe_radius = if safe_plus_proved {
            u16::MAX
        } else {
            radius_before_unbounded
                .or_else(compute_safe_radius)
                .unwrap_or(0)
        };
        let whitespace_proved = prove_slice(whitespace, whitespace_candidates.as_slice());
        let decision = LlgMasterDecision {
            safe_radius,
            whitespace: whitespace_proved,
        };
        if !decision.is_empty() {
            llg_master_decision = Some(decision);
        }
    }
    // Certify an upper bound separately from the positive safe radius. A
    // successful lower-radius proof alone NEVER licenses rejecting long tokens.
    // Every source in this one exact root's epsilon closure must have a finite
    // body envelope that cannot finish its terminal on a safe-string atom.
    static SAFE_ENVELOPE_UPPER_ENABLED: OnceLock<bool> = OnceLock::new();
    let mut safe_envelope_upper = if *SAFE_ENVELOPE_UPPER_ENABLED.get_or_init(|| {
        crate::env::var_os("GLRMASK_DISABLE_SAFE_ENVELOPE_UPPER").is_none()
    }) && HOT_SINGLE_ROOT && root_branches.len() == 1
        && root_branches[0].initial_prune_guard.is_passed()
        && vocab.llg_master_trie().is_some()
        && !vocab.is_grammar_quotiented()
    {
        (|| -> Option<u16> {
            let source = root_branches[0].exact_tokenizer_state?;
            let tok = &state.constraint.tokenizer;
            let slice = vocab.llg_slice_by_cache_id(LLG_SAFE_PLUS_SLICE as u32)?;
            let dfa = slice.dfa();
            let mut upper = None::<u32>;
            for raw in tok.singleton_epsilon_closure(source).iter().copied() {
                // Pure structural epsilon nodes add no consuming behavior.
                if tok.state_has_epsilon_transitions(raw)
                    && tok.transitions_from(raw).next().is_none()
                    && tok.matched_terminal_bitset(raw).is_empty()
                {
                    continue;
                }
                let bound = tok.virtual_residual_safe_atom_length_upper_bound(
                    raw,
                    dfa.start_state(),
                    dfa.class_count(),
                    dfa.byte_to_class_map(),
                    dfa.transition_table(),
                    dfa.accepting_map(),
                    dfa.can_reach_accepting_map(),
                )?;
                upper = Some(upper.map_or(bound, |old| old.max(bound)));
            }
            let upper = u16::try_from(upper?).ok()?;
            // No vocabulary token can exceed the maximum observed safe length.
            if upper >= vocab.llg_master_max_safe_chars() {
                return None;
            }
            if let Some(lower) = llg_master_decision
                && (lower.whitespace || lower.safe_radius > upper) {
                    return None;
                }
            Some(upper)
        })()
    } else {
        None
    };
    if safe_envelope_upper.is_some() && llg_master_decision.is_none() {
        llg_master_decision = Some(LlgMasterDecision::default());
    }
    let safe_interval_work = |decision: LlgMasterDecision| -> Option<usize> {
        let base = vocab.llg_master_residual_ops(decision.safe_radius, decision.whitespace)?;
        if let Some(upper) = safe_envelope_upper {
            let above_upper = vocab.llg_master_residual_ops(upper, decision.whitespace)?;
            let all_safe_skipped = vocab.llg_master_residual_ops(u16::MAX, decision.whitespace)?;
            Some(base.checked_sub(above_upper)?.checked_add(all_safe_skipped)?)
        } else {
            Some(base)
        }
    };
    if crate::env::var_os("GLRMASK_DIAG_SAFE_ENVELOPE_UPPER").is_some() {
        eprintln!(
            "[safe_envelope_upper] generation={} lower={:?} upper={:?} work={:?}",
            state.generation,
            llg_master_decision.map(|v| v.safe_radius),
            safe_envelope_upper,
            llg_master_decision.and_then(safe_interval_work)
        );
    }

    // Exact master proofs are not automatically profitable. Selecting the
    // partitioned master trie also abandons the optimized ordinary-trie hot
    // lane; for small bounded radii the residual master walk can be comparable
    // to, or even larger than, the ordinary vocabulary walk. Gate the route on
    // exact precomputed strict-walk volume without changing proof semantics.
    if let Some(decision) = llg_master_decision {
        let max_permille = crate::env::var("GLRMASK_EXPERIMENT_CONFIG_MASTER_MAX_RESIDUAL_PERMILLE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1000);
        let ordinary_ops = trie.full_walk_ops().len();
        let profitable = safe_interval_work(decision)
            .is_some_and(|residual_ops| {
                residual_ops.saturating_mul(1000)
                    <= ordinary_ops.saturating_mul(max_permille)
            });
        if crate::env::var_os("GLRMASK_DIAG_CANDIDATE_GATES").is_some() {
            eprintln!("[candidate_gate_master] generation={} radius={} whitespace={} ordinary={} residual={:?} profitable={}",
                state.generation, decision.safe_radius, decision.whitespace, ordinary_ops,
                vocab.llg_master_residual_ops(decision.safe_radius, decision.whitespace), profitable);
        }
        if !profitable {
            llg_master_decision = None;
        }
    }
    if llg_master_decision.is_none() {
        safe_envelope_upper = None;
    }
    let trie = llg_master_decision
        .and_then(|_| vocab.llg_master_trie())
        .map_or(trie, |slice| slice.trie());
    debug_assert!(trie.full_walk_max_parent_depth() < 255);

    // Choose output polarity only after every master/slice proof above has
    // completed. A late master decision changes both the trie being walked and
    // the set of tokens represented by skipped subtrees, so initializing a
    // sparse zero mask before that decision would omit the newly certified
    // admitted slice. The three modes below are exact representations of the
    // same walk result:
    //   * master: seed the already-proved admitted slice and add residual hits;
    //   * singleton: seed zero and add the few surviving endpoints;
    //   * ordinary: seed all tokens and clear rejected subtrees/endpoints.
    // For ordinary passed-guard walks the accepted and rejected output sets
    // are both exact representations of the same logical result.  The legacy
    // path chose one polarity before the walk, which can force O(vocab)
    // mutations for a sparse result (or the reverse for a dense result).
    // Defer ordinary materialization and record compact trie events so the
    // cheaper exact side can be selected after the logical walk.  This is an
    // output-representation choice only: the parser/lexer walk and acceptance
    // decisions are identical for both polarities.
    static FORCE_POSITIVE_REBUILD: OnceLock<bool> = OnceLock::new();
    let force_positive_rebuild = *FORCE_POSITIVE_REBUILD
        .get_or_init(|| crate::env::var_os("GLRMASK_EXPERIMENT_FORCE_POSITIVE_REBUILD").is_some())
        && llg_master_decision.is_none()
        && state.constraint.ignore_terminal.is_none()
        && root_branches
            .iter()
            .all(|branch| branch.initial_prune_guard.is_passed())
        && vocab.residual_original_token_words_for(trie).is_none();
    // Grammar-quotiented O2 masks can be extremely dense even when the parser
    // admits only one terminal: one quotient terminal may represent nearly the
    // whole model vocabulary.  Do not let parser singleton-ness force a sparse
    // positive rebuild in that case.  Record the exact walk result and choose
    // the cheaper output polarity after the walk instead.  O1 retains the
    // existing singleton-positive policy.
    let quotient_adaptive_polarity = vocab.is_grammar_quotiented();
    // A two-terminal scalar frontier can be dense (for example a
    // nullable string). Deferred polarity blocks the already-exact dense hot
    // walker, even though that walker can set endpoints without constructing
    // both output sides. Select its positive output before walking substantial
    // full-vocabulary tries. This changes only output scheduling: every token
    // still follows the same lexer/parser transitions and failed hot attempts
    // retain the ordinary exact fallback. Correlated guards, quotient vocabularies and profiled walks keep their
    // existing policy. Lexer IDs are interpreted only by the existing executed
    // transition table; projected IDs are not owner-tokenizer coordinates.
    static SCALAR_HOT_POSITIVE_REBUILD_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let scalar_hot_positive_rebuild = !quotient_adaptive_polarity
        && !singleton_positive_rebuild && !effective_singleton_positive_rebuild
        && llg_master_decision.is_none()
        && state.constraint.ignore_terminal.is_none()
        && !profile_walk && !profile_kernel
        && *SCALAR_HOT_POSITIVE_REBUILD_ENABLED.get_or_init(|| {
            crate::env::var_os("GLRMASK_DISABLE_DENSE_HOT_LANE").is_none()
        })
        && HOT_SINGLE_ROOT && root_branches.len() == 1
        && root_branches[0].initial_prune_guard.is_passed()
        && trie.full_walk_ops().len() >= 4096
        && parser_cache.admitted(&state.constraint, root_parser_nodes[0]).count_ones() == 2;
    let deferred_output = !force_positive_rebuild
        && !scalar_hot_positive_rebuild
        && llg_master_decision.is_none()
        && (quotient_adaptive_polarity
            || (!singleton_positive_rebuild && !effective_singleton_positive_rebuild))
        && state.constraint.ignore_terminal.is_none()
        && root_branches
            .iter()
            .all(|branch| branch.initial_prune_guard.is_passed())
        && vocab.residual_original_token_words_for(trie).is_none();

    let mut positive_rebuild = if let Some(decision) = llg_master_decision {
        if !vocab.copy_llg_master_admitted_words(decision.safe_radius, decision.whitespace, buf) {
            return Err("dynamic-radius LLG master trie is missing admitted-token words".to_owned());
        }
        true
    } else if !deferred_output
        && (singleton_positive_rebuild
            || effective_singleton_positive_rebuild
            || scalar_hot_positive_rebuild
            || force_positive_rebuild)
    {
        buf.fill(0);
        true
    } else if deferred_output {
        // No output mutation occurs until the walk has counted both exact
        // materialization sides.  Initialize defensively; the chosen side is
        // written in full before return.
        buf.fill(0);
        false
    } else {
        let all_words = vocab.all_original_token_words();
        let copy_len = buf.len().min(all_words.len());
        buf[..copy_len].copy_from_slice(&all_words[..copy_len]);
        if copy_len < buf.len() {
            buf[copy_len..].fill(0);
        }
        if let Some(residual_words) = vocab.residual_original_token_words_for(trie) {
            for (target, &residual) in buf.iter_mut().zip(residual_words) {
                *target &= !residual;
            }
            true
        } else {
            false
        }
    };

    // Choose interior pruning from exact root evidence, not a different
    // byte executor. On broad roots a direct row lookup and endpoint query
    // are enough; on parser-dead roots prune before descending.
    let tokenizer = lexer_scan_cache.tokenizer();
    let mut adapter = DenseProductTransitions {
        table: &transitions, tokenizer, vocab, finalizers: finalizer_code,
        continuing: single_finalizer_continues, ignored_output,
        prune_prefixes: true, proofs: FullWalkIdentityProofCache::default(),
        witness_enabled: full_walk_acceleration_enabled()
            && crate::env::var_os("GLRMASK_DISABLE_LIVE_BRANCH_WITNESS").is_none(),
    };
    parser_cache.profile |= profile_walk || profile_kernel;
    let ordinary_trie = llg_master_decision.is_none()
        && vocab.residual_original_token_words_for(trie).is_none();
    if HOT_SINGLE_ROOT && root_branches.len() == 1 && ordinary_trie
        && root_branches[0].initial_prune_guard.is_passed()
        && !profile_walk && !profile_kernel && trie.has_full_walk_root_byte_index()
        && crate::env::var_os("GLRMASK_DISABLE_DENSE_HOT_LANE").is_none()
    {
        let lexer = root_branches[0].tokenizer_config;
        let parser = root_parser_nodes[0];
        let mut parser_dead_mass = 0usize;
        for byte in 0..=255u8 {
            let Some((start, _, _)) = trie.full_walk_root_byte_range(byte) else { continue; };
            let cell = transitions.cell(lexer, byte);
            if T::cell_is_dead(cell) || T::cell_has_finalizer(cell) { continue; }
            if !parser_cache.physical_token_boundary_allowed_table(
                &state.constraint, &mut adapter, parser, T::cell_target(cell),
            ) {
                let (child, _) = trie.full_walk_dead_subtree(start as usize);
                parser_dead_mass += vocab.subtree_original_tokens_for(trie, child).len();
                if parser_dead_mass >= 256 { break; }
            }
        }
        adapter.prune_prefixes = parser_dead_mass >= 256;
        if positive_rebuild && !adapter.prune_prefixes
            && transitions.dense_output_hint(lexer) == Some(true)
        {
            let all_words = vocab.all_original_token_words();
            let copy_len = buf.len().min(all_words.len());
            buf[..copy_len].copy_from_slice(&all_words[..copy_len]);
            buf[copy_len..].fill(0);
            positive_rebuild = false;
        }
    }
    let result = super::execute_full_walk_product(
        state, vocab, trie, root_branches, buf, &mut adapter, parser_cache,
        &root_parser_nodes, initial_lexer_state,
        super::FullWalkTraversalPlan {
            master_decision: llg_master_decision.map(|d| (d.safe_radius, d.whitespace)),
            safe_envelope_upper, first_match_direct_root: None,
            positive_rebuild, deferred_output,
            allow_sparse_root: ordinary_trie && positive_rebuild,
            dense_root_bytes: true, include_empty_root: false,
        },
    )?;
    if ordinary_trie && HOT_SINGLE_ROOT && root_branches.len() == 1
        && ignored_output.is_none() && result
    {
        let admitted = buf.iter().map(|w| w.count_ones() as usize).sum::<usize>();
        let total = vocab.all_original_token_words().iter().map(|w| w.count_ones() as usize).sum::<usize>();
        transitions.cache_dense_output_hint(root_branches[0].tokenizer_config,
            total != 0 && admitted >= total.saturating_sub(total / 8));
    }
    Ok(result)
}
#[cfg(test)]
mod full_walk_acceleration_tests {
    use super::*;


    // A deliberately literal branch-list oracle. It owns parser languages,
    // not interned IDs, and never uses the product step or its memoization.
    #[derive(Clone)]
    struct ReferenceBranch {
        lexer: u32,
        parser: ParserStacks,
        exclusions: Vec<(u32, TerminalID)>,
    }

    fn reference_step<T: FullWalkTransitionTable>(
        table: &T, tokenizer: &Tokenizer, constraint: &Constraint,
        branches: Vec<ReferenceBranch>, byte: u8,
    ) -> Vec<ReferenceBranch> {
        let mut next = Vec::new();
        for mut branch in branches {
            let mut exclusions = Vec::new();
            let mut rejected = false;
            for (lexer, terminal) in branch.exclusions.drain(..) {
                let target = table.transition(lexer, byte);
                if target == u32::MAX { continue; }
                if table.matched_terminals(tokenizer, target).contains(&terminal) {
                    rejected = true;
                    break;
                }
                if table.future_contains(tokenizer, target, terminal) {
                    exclusions.push((target, terminal));
                }
            }
            if rejected { continue; }
            let target = table.transition(branch.lexer, byte);
            if target == u32::MAX { continue; }
            for terminal in table.matched_terminals(tokenizer, target) {
                let ignore = Some(terminal) == constraint.ignore_terminal;
                let parser = if ignore { Some(branch.parser.clone()) }
                    else { parser_child(constraint, &branch.parser, terminal) };
                if let Some(parser) = parser {
                    let mut guard = exclusions.clone();
                    if !ignore && table.future_contains(tokenizer, target, terminal) {
                        guard.push((target, terminal));
                    }
                    next.push(ReferenceBranch { lexer: tokenizer.start_state(), parser, exclusions: guard });
                }
            }
            let admitted = exact_parser_admission_for_stacks(constraint, &branch.parser);
            if table.future_intersects(tokenizer, target, &admitted)
                || constraint.ignore_terminal.is_some_and(|t| table.future_contains(tokenizer, target, t))
            {
                next.push(ReferenceBranch { lexer: target, parser: branch.parser, exclusions });
            }
        }
        next
    }

    fn reference_boundary<T: FullWalkTransitionTable>(
        table: &T, tokenizer: &Tokenizer, constraint: &Constraint,
        branches: &[ReferenceBranch],
    ) -> bool {
        branches.iter().any(|branch| branch.lexer == tokenizer.start_state()
            || constraint.ignore_terminal.is_some_and(|t| table.future_contains(tokenizer, branch.lexer, t))
            || table.future_intersects(tokenizer, branch.lexer,
                &exact_parser_admission_for_stacks(constraint, &branch.parser)))
    }

    #[test]
    fn product_matches_literal_guarded_branch_oracle_and_witness() {
        use crate::{DynamicConstraint, Grammar, Vocab};
        use super::super::FullWalkTransitionTable as ProductTable;
        let vocab = Vocab::from_entries_unchecked(["a", "b", "aa", "ab", " "].into_iter().enumerate()
            .map(|(id, text)| (id as u32, text.as_bytes().to_vec())).collect());
        let dynamic = DynamicConstraint::compile(Grammar::glrm(
            "start s; ignore WS; t WS = ' '+; t A = 'a'+; t B = 'b'; nt s = A B;"), &vocab).unwrap();
        let constraint = &dynamic.inner;
        let tokenizer = &constraint.tokenizer;
        let walk_vocab = crate::compiler::constraint_possible_matches::runtime_dynamic_vocab_for_vocab(&vocab);
        let states = tokenizer.num_states() as usize;
        let mut rows = vec![u32::MAX; states * 256];
        let finals = (0..states).map(|state| match tokenizer.matched_terminals_slice(state as u32) {
            [] => u32::MAX,
            [terminal] => *terminal,
            _ => u32::MAX - 1,
        }).collect::<Vec<_>>();
        let continues = (0..states).map(|state| {
            u8::from(finals[state] != u32::MAX && finals[state] != u32::MAX - 1
                && tokenizer.possible_future_terminals(state as u32).contains(finals[state] as usize))
        }).collect::<Vec<_>>();
        for state in 0..states {
            for (byte, target) in tokenizer.transitions_from(state as u32) {
                rows[state * 256 + byte as usize] = target
                    | if finals[target as usize] != u32::MAX { 0x8000_0000 } else { 0 };
            }
        }
        let transitions = FullWalkFlat32 { transitions: &rows };
        let mut adapter = DenseProductTransitions {
            table: &transitions, tokenizer, vocab: &walk_vocab,
            finalizers: &finals, continuing: &continues, ignored_output: None,
            prune_prefixes: true, proofs: FullWalkIdentityProofCache::default(),
            witness_enabled: true,
        };
        let stacks = constraint.start().state.entries[0].1.apply(|_| ());
        let (mut cache, _) = FullWalkParserCache::from_roots(&DynamicBranches::new(), Some(states));
        let parser = cache.push_stacks(stacks.clone());
        let alphabet = U8Set::single(b'a').to_words();
        let lexer = (0..states as u32).find(|&lexer| {
            let cell = transitions.cell(lexer, b'a');
            !FullWalkFlat32::cell_is_dead(cell)
                && FullWalkFlat32::cell_target(cell) == lexer
                && FullWalkFlat32::cell_has_finalizer(cell)
                && cache.physical_token_boundary_allowed_table(constraint, &mut adapter, parser, lexer)
        }).expect("fixture must exercise a live finalizing self-edge");
        let mut root = FullWalkProduct::default();
        root.push(lexer, parser, FullWalkPruneGuard::Passed);
        root.push(tokenizer.start_state(), parser, FullWalkPruneGuard::Passed);
        assert!(adapter.proves_live_subtree(&root, alphabet, &mut cache, constraint));
        let mut guarded = FullWalkProduct::default();
        guarded.push(lexer, parser, FullWalkPruneGuard::Pending(smallvec::smallvec![
            FullWalkPruneMemory::Exact(lexer, finals[lexer as usize])
        ]));
        assert!(!adapter.proves_live_subtree(&guarded, alphabet, &mut cache, constraint));
        assert!(!adapter.proves_live_subtree(&root, U8Set::single(b'b').to_words(), &mut cache, constraint));
        for length in 1..=5u32 {
            for mut code in 0..3usize.pow(length) {
                let mut product = root.clone();
                let mut reference = vec![
                    ReferenceBranch { lexer, parser: stacks.clone(), exclusions: Vec::new() },
                    ReferenceBranch { lexer: tokenizer.start_state(), parser: stacks.clone(), exclusions: Vec::new() },
                ];
                let mut kernel = FullWalkKernel {
                    transitions: &mut adapter, parser_cache: &mut cache, constraint,
                    initial: tokenizer.start_state(), union_cache: FxHashMap::default(),
                    scoped_pruning: true, probe_budget: 0, non_pruning: 0,
                };
                for _ in 0..length {
                    let byte = [b'a', b'b', b' '][code % 3];
                    code /= 3;
                    product = kernel.step(&product, byte);
                    reference = reference_step(&transitions, tokenizer, constraint, reference, byte);
                    assert_eq!(kernel.boundary_allowed(&product),
                        reference_boundary(&transitions, tokenizer, constraint, &reference));
                }
            }
        }
        let union_storage = std::sync::Mutex::new(DynamicLazyUnionCache::default());
        let overflowed = std::cell::Cell::new(false);
        let tokenizer_owner = Arc::new(tokenizer.clone());
        let (union, union_root) = FullWalkLazyUnion::new(
            &tokenizer_owner, None, Some(&rows), union_storage.lock().unwrap(),
            &[lexer, tokenizer.start_state()], &overflowed,
        ).unwrap();
        let mut union_adapter = DenseProductTransitions {
            table: &union, tokenizer, vocab: &walk_vocab, finalizers: &finals,
            continuing: &continues, ignored_output: None, prune_prefixes: true,
            proofs: FullWalkIdentityProofCache::default(), witness_enabled: true,
        };
        let (mut union_cache, _) = FullWalkParserCache::from_roots(
            &DynamicBranches::new(), Some(union.state_count(tokenizer)));
        let parser = union_cache.push_stacks(stacks);
        let mut product = FullWalkProduct::default();
        product.push(union_root, parser, FullWalkPruneGuard::Passed);
        assert!(union_adapter.proves_live_subtree(&product, alphabet, &mut union_cache, constraint));
        let mut kernel = FullWalkKernel {
            transitions: &mut union_adapter, parser_cache: &mut union_cache, constraint,
            initial: tokenizer.start_state(), union_cache: FxHashMap::default(),
            scoped_pruning: true, probe_budget: 0, non_pruning: 0,
        };
        for _ in 0..6 {
            product = kernel.step(&product, b'a');
            assert!(kernel.boundary_allowed(&product));
        }
        assert!(!overflowed.get());
    }
    #[test]
    fn row_liveness_bounds_are_exact_for_all_small_terminal_sets() {
        let mut proofs = 0;
        let mut unresolved = 0;
        for upper in 0u32..16 {
            for lower in 0u32..16 {
                if lower & !upper != 0 { continue; }
                for admitted in 0u32..16 {
                    if lower & !admitted != 0 || admitted & !upper != 0 { continue; }
                    for futures in 0u32..16 {
                        match full_walk_row_liveness_bound(lower & futures != 0, || upper & futures != 0) {
                            Some(value) => { assert_eq!(value, admitted & futures != 0); proofs += 1; }
                            None => { assert_eq!(lower & futures, 0); assert_ne!(upper & futures, 0); unresolved += 1; }
                        }
                    }
                }
            }
        }
        assert!(proofs > 1000 && unresolved > 100);
        assert_eq!(full_walk_row_liveness_bound(true, || panic!("proven positive must not query upper bound")), Some(true));
    }

    #[test]
    fn row_liveness_bounds_match_compiled_and_loaded_parser_frontiers() {
        use crate::{DynamicConstraint, Grammar, Vocab};
        let vocab = Vocab::from_entries_unchecked([
            "a", "b", "c", "aa", "ab", "ba", "bb", " ", "acb",
        ].into_iter().enumerate()
            .map(|(id, text)| (id as u32, text.as_bytes().to_vec())).collect());
        let grammars = [
            "start s; t A = 'a'; t B = 'b'; t C = 'c'; nt s = A s B | C;",
            "start s; t A = 'a'; t B = 'b'; t C = 'c'; nt s = A s B | A A | A B | C;",
            "start s; ignore WS; t WS = ' '+; t A = 'a'+; t B = 'ab'; t C = 'b'+; nt s = A s C | B | A C;",
        ];
        let mut positive = 0;
        let mut negative = 0;
        let mut compared = 0;
        for grammar in grammars {
            let compiled = DynamicConstraint::compile(Grammar::glrm(grammar), &vocab).unwrap();
            let loaded = DynamicConstraint::load_with_vocab(&compiled.save(), &vocab).unwrap();
            for dynamic in [&compiled, &loaded] {
                let constraint = &dynamic.inner;
                // These predicates read terminal metadata, never byte cells.
                let transitions = FullWalkFlat32 { transitions: &[] };
                let mut adapter = DenseProductTransitions {
                    table: &transitions, tokenizer: &constraint.tokenizer,
                    vocab: constraint.dynamic_mask_vocab_for_runtime(),
                    finalizers: &[], continuing: &[], ignored_output: None,
                    prune_prefixes: true, proofs: FullWalkIdentityProofCache::default(),
                    witness_enabled: true,
                };
                let mut pending = constraint.start().state.entries.iter()
                    .map(|(_, gss)| gss.apply(|_| ())).collect::<Vec<_>>();
                let mut seen = std::collections::HashSet::new();
                for _ in 0..5 {
                    let mut next = Vec::new();
                    for stacks in pending {
                        let mut key = stacks.to_stacks(4096).expect("small grammar path bound");
                        key.sort();
                        if !seen.insert(key) { continue; }
                        let (mut cache, _) = FullWalkParserCache::from_roots(
                            &DynamicBranches::new(), Some(constraint.tokenizer.num_states() as usize),
                        );
                        cache.row_liveness_enabled = true;
                        cache.direct_boundary_rows = Some(FullWalkBoundaryDirectCache::new());
                        let node = cache.push_stacks(stacks.clone());
                        let admitted = exact_parser_admission_for_stacks(constraint, &stacks);
                        for lexer in 0..constraint.tokenizer.num_states() {
                            let exact = transitions.future_intersects(&constraint.tokenizer, lexer, &admitted);
                            if let Some(answer) = cache.row_future_allowed(
                                constraint, &mut adapter, node, lexer,
                            ) {
                                assert_eq!(answer, exact, "grammar={grammar} lexer={lexer}");
                                positive += usize::from(answer);
                                negative += usize::from(!answer);
                            }
                            compared += 1;
                        }
                        // Independently check the complete cached predicate,
                        // including ignore, positive/negative hits, and disabled fallback.
                        for lexer in 0..constraint.tokenizer.num_states() {
                            let exact = transitions.future_intersects(&constraint.tokenizer, lexer, &admitted)
                                || constraint.ignore_terminal.is_some_and(|terminal|
                                    transitions.future_contains(&constraint.tokenizer, lexer, terminal));
                            assert_eq!(cache.physical_token_boundary_allowed_table(
                                constraint, &mut adapter, node, lexer), exact);
                            assert_eq!(cache.physical_token_boundary_allowed_table(
                                constraint, &mut adapter, node, lexer), exact);
                        }
                        cache.row_liveness_enabled = false;
                        assert!(cache.row_future_allowed(constraint, &mut adapter, node, 0).is_none());
                        for terminal in 0..constraint.parser_terminal_count() {
                            if let Some(child) = parser_child(constraint, &stacks, terminal) {
                                next.push(child);
                            }
                        }
                    }
                    pending = next;
                }
            }
        }
        assert!(compared > 100, "test must cover actual parser/lexer combinations");
        assert!(positive > 0 && negative > 0, "both proof directions must execute");
    }


    #[test]
    fn root_output_scope_marker_checks_all_aliases_across_output_words() {
        let direct = (1u64 << 32) | 0b101;
        assert!(full_walk_marker_fully_ignored(direct, &[0, 0b111], |_| panic!("direct")));
        assert!(!full_walk_marker_fully_ignored(direct, &[0, 0b001], |_| panic!("direct")));
        assert!(!full_walk_marker_fully_ignored(direct, &[u32::MAX], |_| panic!("direct")));
        let aliased = DYNAMIC_TOKEN_MARKER_FALLBACK | 4;
        for mask in [vec![1, 0], vec![0, 1], vec![1, 1]] {
            let covered = full_walk_marker_fully_ignored(aliased, &mask, |id| {
                assert_eq!(id, 3);
                [0_u32, 32].iter().all(|id| mask[*id as usize / 32] & (1 << (*id % 32)) != 0)
            });
            assert_eq!(covered, mask == [1, 1]);
        }
        assert!(!full_walk_marker_fully_ignored(DYNAMIC_TOKEN_MARKER_FALLBACK, &[], |_| panic!("invalid")));
        assert!(!full_walk_marker_fully_ignored(u64::MAX, &[], |_| panic!("invalid")));
    }


    #[test]
    fn root_output_scope_cursor_matches_all_intervals_and_scans_each_marker_once() {
        let markers: Vec<u64> = (0..8).collect();
        for ignored in 0_u32..256 {
            let mut cursor = FullWalkOutputScopeCursor::default();
            let mut visits = [0_usize; 8];
            for start in 0..=markers.len() {
                for end in (start..=markers.len()).rev() {
                    let expected = markers[start..end].iter().all(|&m| ignored & (1 << m) != 0);
                    let actual = cursor.covers(&markers, start, end, |m| {
                        visits[m as usize] += 1;
                        ignored & (1 << m) != 0
                    });
                    assert_eq!(actual, expected, "ignored={ignored} interval={start}..{end}");
                }
            }
            assert!(visits.iter().all(|&n| n <= 1), "repeated scans: {visits:?}");
            // A caller restoring an earlier traversal must not reuse a later
            // ignored-prefix certificate. Test nonmonotone order too.
            for start in (0..=markers.len()).rev() {
                for end in start..=markers.len() {
                    let expected = markers[start..end].iter().all(|&m| ignored & (1 << m) != 0);
                    assert_eq!(cursor.covers(&markers, start, end, |m| ignored & (1 << m) != 0), expected);
                }
            }
            assert!(!cursor.covers(&markers, 3, 2, |_| panic!("invalid interval")));
            assert!(!cursor.covers(&markers, 0, 9, |_| panic!("invalid interval")));
        }
        let mut cursor = FullWalkOutputScopeCursor::default();
        assert!(cursor.covers(&[], 0, 0, |_| panic!("empty range")));
    }

    #[test]
    fn root_output_scope_prefix_ranges_match_literal_marker_responsibility() {
        let markers = [0, 1, 2, 3, 4, 5, 6, 7];
        for ignored in 0..256u32 {
            let scope = FullWalkOutputScope::new(&markers, |marker| ignored & (1 << marker) != 0).unwrap();
            for start in 0..=8 {
                for end in start..=8 {
                    assert_eq!(scope.covers(start, end),
                        (start..end).all(|marker| ignored & (1 << marker) != 0));
                }
            }
            assert!(!scope.covers(4, 3));
            assert!(!scope.covers(0, 9));
            assert!(!scope.covers(usize::MAX, usize::MAX));
        }
        assert!(FullWalkOutputScope::new(&[], |_| false).unwrap().covers(0, 0));
    }

    #[test]
    fn root_output_scope_dont_care_algebra_preserves_guarded_union() {
        for accepted in 0..16u32 {
            for blocked in 0..16u32 {
                let ignored = accepted | blocked;
                for root in 0..16u32 {
                    for arbitrary in 0..16u32 {
                        let partial = (root & !ignored) | (arbitrary & ignored);
                        assert_eq!(accepted | (partial & !blocked), accepted | (root & !blocked));
                    }
                }
            }
        }
    }

    #[test]
    fn structural_identity_classes_separate_finalizers_and_never_include_dead_or_changed_targets() {
        let state = 37;
        let mut calls = 0;
        let classes = full_walk_identity_byte_classes(state, |byte| {
            calls += 1;
            match byte {
                0 | 255 => Some((state, false)),
                b'a'..=b'z' => Some((state, true)),
                b'A'..=b'Z' => Some((state + 1, false)),
                b'0'..=b'9' => Some((state + 1, true)),
                _ => None,
            }
        });
        assert_eq!(calls, 256);
        for byte in 0..=u8::MAX {
            let bit = 1u64 << (byte & 63);
            assert_eq!(classes.ordinary[byte as usize >> 6] & bit != 0,
                       byte == 0 || byte == 255);
            assert_eq!(classes.finalizing[byte as usize >> 6] & bit != 0,
                       byte.is_ascii_lowercase());
        }
        let shifted = full_walk_identity_byte_classes(state + 1, |byte| {
            Some((state, byte % 2 == 0))
        });
        assert_eq!(shifted.ordinary, [0; 4]);
        assert_eq!(shifted.finalizing, [0; 4]);
    }


    #[test]
    fn identity_classes_of_union_require_every_member_and_keep_finalizer_or() {
        let mut left = full_walk_identity_byte_classes(1, |byte| match byte {
            0 | b'a' | b'b' => Some((1, false)),
            b'c' | 255 => Some((1, true)),
            _ => None,
        });
        let right = full_walk_identity_byte_classes(2, |byte| match byte {
            b'b' | b'c' => Some((2, false)),
            b'a' | 255 => Some((2, true)),
            _ => None,
        });
        left.intersect_union_member(right);
        assert_eq!(left.ordinary, U8Set::single(b'b').to_words());
        assert_eq!(left.finalizing, U8Set::from_bytes(&[b'a', b'c', 255]).to_words());
        left.intersect_union_member(FullWalkIdentityByteClasses::default());
        assert_eq!(left.ordinary, [0; 4]);
        assert_eq!(left.finalizing, [0; 4]);
    }

    #[test]
    fn structural_identity_proof_cache_is_bounded_and_keeps_exact_row_keys() {
        let tokenizer = crate::automata::lexer::tokenizer::arbitrary_flat32_test_tokenizer();
        let mut rows = vec![u16::MAX; 65 * 256];
        for state in 0..65 {
            rows[state * 256 + b'a' as usize] = state as u16;
        }
        let table = FullWalkFlat16 { transitions: &rows };
        let mut cache = FullWalkIdentityProofCache::default();
        for state in 0..64 {
            let classes = cache.get(&table, &tokenizer, state).unwrap();
            assert_eq!(classes.ordinary, U8Set::single(b'a').to_words());
            assert_eq!(classes.finalizing, [0; 4]);
        }
        assert_eq!(cache.rows.len(), 64);
        assert!(cache.get(&table, &tokenizer, 64).is_none());
        assert!(cache.get(&table, &tokenizer, 0).is_some());
        assert_eq!(cache.rows.len(), 64);
    }


    #[test]
    fn identity_subtree_alphabet_requires_every_byte() {
        let mut identity = [0_u64; 4];
        identity[b'a' as usize >> 6] |= 1 << (b'a' & 63);
        identity[3] |= 1 << 63;
        assert!(identity_alphabet_covers_subtree(&identity, identity));
        assert!(identity_alphabet_covers_subtree(&identity, [0; 4]));
        for byte in 0_u16..=255 {
            let mut alphabet = [0_u64; 4];
            alphabet[byte as usize >> 6] = 1 << (byte & 63);
            assert_eq!(identity_alphabet_covers_subtree(&identity, alphabet),
                byte == u16::from(b'a') || byte == 255);
        }
    }


    #[test]
    fn dense_parser_canonical_preserves_complete_stack_languages() {
        let (mut cache, _) = FullWalkParserCache::from_roots(&DynamicBranches::new(), Some(9));
        cache.canonicalize = true;
        let first = ParserStacks::from_stacks(&[
            (vec![0, 1, 7], ()), (vec![0, 2, 7], ()),
        ]);
        let id = cache.push_stacks(first.clone());
        // Pass the cost gate without requiring a grammar-specific shape.
        for value in 1..32 {
            cache.push_stacks(ParserStacks::from_single_stack(vec![value + 100], ()));
        }
        let same = ParserStacks::merge_many([
            ParserStacks::from_single_stack(vec![0, 2, 7], ()),
            ParserStacks::from_single_stack(vec![0, 1, 7], ()),
        ]);
        assert_eq!(cache.intern_dense_parser_stacks(same), id);
        let different = ParserStacks::from_single_stack(vec![0, 3, 7], ());
        let different_id = cache.intern_dense_parser_stacks(different.clone());
        assert_ne!(different_id, id, "equal parser tops are not an equivalence");
        assert_eq!(cache.intern_dense_parser_stacks(different), different_id);
        let empty_word = first.merge(&ParserStacks::from_single_stack(Vec::new(), ()));
        assert_ne!(cache.intern_dense_parser_stacks(empty_word), id, "empty-stack acceptance is part of the key");
        assert!(cache.canonical.as_ref().unwrap().hits >= 2);
    }

    #[test]
    fn dense_parser_canonical_budget_failure_falls_back_without_aliasing() {
        let (mut cache, _) = FullWalkParserCache::from_roots(&DynamicBranches::new(), Some(9));
        cache.canonicalize = true;
        for value in 0..32 {
            cache.push_stacks(ParserStacks::from_single_stack(vec![value], ()));
        }
        cache.canonical = Some(Box::new(FullWalkParserCanonical {
            keys: crate::ds::leveled_gss::GssSemanticKeyInterner::with_budget(1, 1, 1),
            nodes: FxHashMap::from_iter([(0, 0)]),
            hits: 0,
        }));
        let gss = ParserStacks::from_single_stack(vec![88, 99], ());
        let first = cache.intern_dense_parser_stacks(gss.clone());
        let second = cache.intern_dense_parser_stacks(gss.clone());
        assert_ne!(first, 0);
        assert_ne!(first, second);
        assert!(cache.nodes[first as usize].gss.ptr_eq(&gss));
        assert!(cache.nodes[second as usize].gss.ptr_eq(&gss));
        assert!(cache.canonical.as_ref().unwrap().keys.is_exhausted());
    }


}

#[cfg(test)]
mod wide_scalar_dispatch_tests {
    use super::*;

    #[test]
    fn lazy_union_pair_memo_preserves_virtual_members_aliases_and_bounds() {
        let tokenizer = Arc::new(crate::automata::lexer::tokenizer::arbitrary_flat32_test_tokenizer());
        let cache = std::sync::Mutex::new(DynamicLazyUnionCache::default());
        let overflowed = std::cell::Cell::new(false);
        let (mut table, first) = FullWalkLazyUnion::new(
            &tokenizer, None, None, cache.lock().unwrap(), &[0, 32_768], &overflowed,
        ).unwrap();
        table.use_union_pair_memo = true;
        let second = table.intern_states(&[1, 32_768]).unwrap();
        let inputs = [0, 1, 2, 32_768, first, second];
        for &left in &inputs {
            for &right in &inputs {
                let result = table.intern_states(&[left, right]).unwrap();
                let reversed = table.intern_states(&[right, left]).unwrap();
                assert_eq!(result, reversed);
                let cache = unsafe { &*table.cache.get() };
                let members = |state: u32| -> Vec<u32> {
                    if state < table.base_state_count { vec![state] }
                    else { cache.states[(state - table.base_state_count) as usize].members.to_vec() }
                };
                let mut expected = members(left);
                expected.extend(members(right));
                expected.sort_unstable(); expected.dedup();
                assert_eq!(members(result), expected, "the full physical union is the oracle");
            }
        }
        let invalid = table.base_state_count + FullWalkLazyUnion::RESERVED_EXTENSION_STATES as u32;
        assert_eq!(table.intern_states(&[0, invalid]), None);
        assert!(table.continuation_witness_components(invalid).is_empty());
        assert_eq!(table.continuation_witness_components(first).as_slice(), &[0, 32_768]);
        let wide = table.intern_states(&(0..9).collect::<Vec<_>>()).unwrap();
        assert!(table.continuation_witness_components(wide).is_empty(),
            "oversized exact unions must fall back rather than truncate their member set");
        let before = unsafe { (&*table.cache.get()).state_by_union_pair.len() };
        // Fill only the memo's bounded index with valid known results. Once
        // full, a new union still executes normally but is not retained.
        {
            let cache = unsafe { &mut *table.cache.get() };
            let memo = &mut cache.state_by_union_pair;
            for i in 1..=(FullWalkLazyUnion::MAX_UNION_PAIR_MEMO - before) {
                memo.insert((1u64 << 63) + i as u64, first);
            }
            assert_eq!(memo.len(), FullWalkLazyUnion::MAX_UNION_PAIR_MEMO);
        }
        let joined = table.intern_states(&[10, 11]).unwrap();
        let cache = unsafe { &*table.cache.get() };
        assert_eq!(cache.states[(joined - table.base_state_count) as usize].members.as_slice(), &[10, 11]);
        assert_eq!(cache.state_by_union_pair.len(), FullWalkLazyUnion::MAX_UNION_PAIR_MEMO);
    }

    #[test]
    fn lazy_scalar_dispatch_rows_support_states_beyond_flat16() {
        let tokenizer = Arc::new(crate::automata::lexer::tokenizer::arbitrary_flat32_test_tokenizer());
        let cache = std::sync::Mutex::new(DynamicLazyUnionCache::default());
        let guard = cache.lock().expect("lazy-union cache lock");
        let overflowed = std::cell::Cell::new(false);
        let (table, root) = FullWalkLazyUnion::new(
            &tokenizer,
            None,
            None,
            guard,
            &[0, 32_768],
            &overflowed,
        )
        .expect("wide lazy scalar-dispatch table");

        assert!(root >= tokenizer.num_states());
        assert_eq!(table.transition(0, b'a'), 32_768);
        assert_eq!(table.transition(0, b'b'), u32::MAX);
        assert!(!overflowed.get());
    }

    #[test]
    fn lazy_scalar_dispatch_soft_reset_reinitializes_physical_rows() {
        let tokenizer = Arc::new(crate::automata::lexer::tokenizer::arbitrary_flat32_test_tokenizer());
        let mut seeded = DynamicLazyUnionCache { source: Some(Arc::clone(&tokenizer)), ..Default::default() };
        seeded.state_by_union_pair.insert(17, u32::MAX);
        seeded
            .base_rows
            .resize_with(tokenizer.num_states() as usize, || None);
        seeded.states.resize_with(
            FullWalkLazyUnion::SOFT_MAX_EXTENSION_STATES,
            || DynamicLazyUnionState { members: SmallVec::from_slice(&[0, 32_768]), row: Default::default(), metadata: None },
        );

        let cache = std::sync::Mutex::new(seeded);
        let guard = cache.lock().expect("lazy-union cache lock");
        let overflowed = std::cell::Cell::new(false);
        let (table, _root) = FullWalkLazyUnion::new(
            &tokenizer,
            None,
            None,
            guard,
            &[0, 32_768],
            &overflowed,
        )
        .expect("lazy table after soft-limit reset");

        {
            let cache = unsafe { &*table.cache.get() };
            assert_eq!(cache.base_rows.len(), tokenizer.num_states() as usize);
            assert!(!cache.state_by_union_pair.contains_key(&17),
                "old extension coordinates must not survive a soft reset");
        }
        assert_eq!(table.transition(0, b'a'), 32_768);
        assert!(!overflowed.get());
    }

    #[test]
    fn lazy_scalar_dispatch_repeated_soft_resets_keep_physical_rows_live() {
        let tokenizer = Arc::new(crate::automata::lexer::tokenizer::arbitrary_flat32_test_tokenizer());
        let cache = std::sync::Mutex::new(DynamicLazyUnionCache::default());

        for _ in 0..3 {
            {
                let mut seeded = cache.lock().expect("lazy-union cache lock");
                seeded.source = Some(Arc::clone(&tokenizer));
                seeded
                    .base_rows
                    .resize_with(tokenizer.num_states() as usize, || None);
                seeded.states.resize_with(
                    FullWalkLazyUnion::SOFT_MAX_EXTENSION_STATES,
                    || DynamicLazyUnionState { members: SmallVec::from_slice(&[0, 32_768]), row: Default::default(), metadata: None },
                );
            }

            let guard = cache.lock().expect("lazy-union cache lock");
            let overflowed = std::cell::Cell::new(false);
            let (table, _root) = FullWalkLazyUnion::new(
                &tokenizer,
                None,
                None,
                guard,
                &[0, 32_768],
                &overflowed,
            )
            .expect("lazy table after repeated soft-limit reset");

            {
                let cache = unsafe { &*table.cache.get() };
                assert_eq!(cache.base_rows.len(), tokenizer.num_states() as usize);
            }
            assert_eq!(table.transition(0, b'a'), 32_768);
            assert!(!overflowed.get());
        }
    }
}
