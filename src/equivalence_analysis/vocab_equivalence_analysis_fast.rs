//! Fast Implementation of Vocab Equivalence Analysis
//!
//! This module provides a high-performance algorithm for computing vocabulary
//! token equivalence classes. Two tokens are equivalent if they produce identical
//! parsing behavior across all initial tokenizer states.
//!
//! The algorithm uses:
//! - Batched iterative refinement over initial states
//! - Parallel signature computation using rayon
//! - Precomputed DFA with optimized memory layout
//! - Incremental suffix hash caching
//!
//! Complexity: O(tokens × states × avg_token_length) with parallelism

// PERMANENT WARNING: Do NOT add caching to file or shortcuts that skip/restrict
// states/tokens for equivalence analysis. Full correctness is mandatory.
// In-memory memoization is fine, but no "cheating" optimizations that drop work.

use crate::dfa_u8::Tokenizer;
use crate::r#macro::is_debug_level_enabled;
use ahash::{AHasher, RandomState};
use hashbrown::HashMap;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::BTreeSet;
use std::hash::{BuildHasher, Hasher};

pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;

// --- TYPE ALIASES AND CONSTANTS ---

type EdgeList = SmallVec<[(usize, usize); 4]>;
type GroupList = SmallVec<[usize; 4]>;
type FinalizerList = SmallVec<[Finalizer; 4]>;

const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;
const NONE_STATE: u32 = u32::MAX;
const NONE_POS: u32 = u32::MAX;

// --- CORE DATA STRUCTURES ---

#[derive(Clone, Copy)]
struct Finalizer {
    gid: usize,
    non_greedy: bool,
}

#[derive(Clone)]
enum FutureMode {
    AlwaysTerminate,
    AlwaysContinue,
    Guarded(GroupList),
}

/// Precomputed DFA with optimized data layout for fast execution.
struct PrecomputedDfa {
    start_state: usize,
    transitions: Vec<[u32; 256]>,
    finalizers: Vec<FinalizerList>,
    future_modes: Vec<FutureMode>,
    guard_masks: Vec<Option<Box<[u64]>>>,
    has_transitions: Vec<bool>,
    num_groups: usize,
    mask_words: usize,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
}

/// Scratch space for position-0 DFA execution across all initial states.
struct Pos0Scratch {
    current_states: Vec<usize>,
    done: Vec<bool>,
    active_indices: Vec<usize>,
    end_states: Vec<Option<usize>>,
    matched_bits: Vec<u64>,
    mask_words: usize,
    match_positions: Vec<u32>,
    match_gen: Vec<u32>,
    cur_gen: u32,
    touched_groups: Vec<GroupList>,
    touched_positions: Vec<usize>,
    touched_states: Vec<usize>,
    base_offsets: Vec<usize>,
    results: Vec<(Option<usize>, EdgeList)>,
    seen_target: Vec<bool>,
    all_targets: Vec<usize>,
}

/// Scratch space for suffix hash computation.
struct SuffixScratch {
    match_positions: Vec<u32>,
    touched_positions: GroupList,
    visited: Vec<bool>,
    queue: Vec<usize>,
    order: Vec<usize>,
    nodes: Vec<Option<(u64, EdgeList)>>,
    pos_hashes: Vec<u64>,
    projected_cache: HashMap<(usize, usize), u64>,
}

// --- HASH UTILITIES ---

static HASH_RANDOM_STATE: Lazy<RandomState> =
    Lazy::new(|| RandomState::with_seeds(HASH_SEED1, HASH_SEED2, HASH_SEED3, HASH_SEED4));

#[inline]
fn new_hasher() -> AHasher {
    HASH_RANDOM_STATE.build_hasher()
}

#[inline]
fn hash_group_list(list: &[usize]) -> u64 {
    let mut hasher = new_hasher();
    hasher.write_u8(1);
    hasher.write_u64(list.len() as u64);
    for &value in list {
        hasher.write_u64(value as u64);
    }
    hasher.finish()
}

// --- DFA PRECOMPUTATION ---

fn precompute_dfa(regex: &Tokenizer) -> PrecomputedDfa {
    let dfa = regex.dfa();
    crate::debug!(4, "Precomputing DFA with {} states", dfa.states.len());
    assert!(
        dfa.states.len() <= u32::MAX as usize,
        "DFA too large for packed transitions"
    );

    // Determine maximum group ID
    let mut max_gid: Option<usize> = None;
    for state in &dfa.states {
        if let Some(m) = state.finalizers.iter().max() {
            max_gid = Some(max_gid.map_or(m, |cur| cur.max(m)));
        }
        if let Some(m) = state.possible_future_group_ids.iter().max() {
            max_gid = Some(max_gid.map_or(*m, |cur| cur.max(*m)));
        }
    }
    if let Some(m) = dfa.non_greedy_finalizers.iter().max() {
        max_gid = Some(max_gid.map_or(*m, |cur| cur.max(*m)));
    }

    let num_groups = max_gid.map(|m| m + 1).unwrap_or(0);
    let mask_words = (num_groups + 63) / 64;

    // Build transition tables and finalizer lists
    let mut transitions: Vec<[u32; 256]> = Vec::with_capacity(dfa.states.len());
    let mut finalizers: Vec<FinalizerList> = Vec::with_capacity(dfa.states.len());
    let mut possible_future: Vec<GroupList> = Vec::with_capacity(dfa.states.len());
    let mut has_transitions: Vec<bool> = Vec::with_capacity(dfa.states.len());

    for state in &dfa.states {
        let mut table = [NONE_STATE; 256];
        for (byte, &target) in state.transitions.iter() {
            table[byte as usize] = target as u32;
        }
        transitions.push(table);
        finalizers.push(
            state
                .finalizers
                .iter()
                .map(|gid| Finalizer {
                    gid,
                    non_greedy: false,
                })
                .collect(),
        );
        possible_future.push(state.possible_future_group_ids.iter().copied().collect());
        has_transitions.push(!state.transitions.is_empty());
    }

    // Mark non-greedy finalizers
    let mut non_greedy_flags = vec![false; num_groups];
    for &gid in &dfa.non_greedy_finalizers {
        if gid < num_groups {
            non_greedy_flags[gid] = true;
        }
    }
    for finals in &mut finalizers {
        for f in finals.iter_mut() {
            f.non_greedy = non_greedy_flags.get(f.gid).copied().unwrap_or(false);
        }
    }

    // Compute future modes + guarded bitmasks
    let mut future_modes: Vec<FutureMode> = Vec::with_capacity(possible_future.len());
    let mut guard_masks: Vec<Option<Box<[u64]>>> = Vec::with_capacity(possible_future.len());

    for future in possible_future.iter() {
        if future.is_empty() {
            future_modes.push(FutureMode::AlwaysTerminate);
            guard_masks.push(None);
            continue;
        }

        let mut guard: GroupList = GroupList::new();
        let mut always_continue = false;
        for &gid in future {
            if gid >= num_groups || !non_greedy_flags[gid] {
                always_continue = true;
                break;
            }
            guard.push(gid);
        }

        if always_continue {
            future_modes.push(FutureMode::AlwaysContinue);
            guard_masks.push(None);
            continue;
        }

        guard.sort_unstable();
        guard.dedup();

        if mask_words == 0 {
            // num_groups==0 implies possible_future is empty, which was handled above.
            future_modes.push(FutureMode::AlwaysTerminate);
            guard_masks.push(None);
            continue;
        }

        let mut mask = vec![0u64; mask_words];
        for &gid in guard.iter() {
            let word = gid >> 6;
            let bit = 1u64 << (gid & 63);
            mask[word] |= bit;
        }

        future_modes.push(FutureMode::Guarded(guard));
        guard_masks.push(Some(mask.into_boxed_slice()));
    }

    // Precompute completion hashes
    let none_completion_hash = {
        let mut hasher = new_hasher();
        hasher.write_u8(0);
        hasher.finish()
    };

    let completion_hash: Vec<u64> = possible_future
        .iter()
        .map(|vec| hash_group_list(vec))
        .collect();

    PrecomputedDfa {
        start_state: dfa.start_state,
        transitions,
        finalizers,
        future_modes,
        guard_masks,
        has_transitions,
        num_groups,
        mask_words,
        completion_hash,
        none_completion_hash,
    }
}

// --- SCRATCH SPACE IMPLEMENTATIONS ---

impl Pos0Scratch {
    fn new(num_states: usize, num_groups: usize) -> Self {
        let base_offsets: Vec<usize> = (0..num_states)
            .map(|idx| idx.saturating_mul(num_groups))
            .collect();
        let mask_words = (num_groups + 63) / 64;
        let match_len = num_states.saturating_mul(num_groups);
        Pos0Scratch {
            current_states: vec![0; num_states],
            done: vec![false; num_states],
            active_indices: Vec::new(),
            end_states: vec![None; num_states],
            matched_bits: vec![0u64; num_states.saturating_mul(mask_words)],
            mask_words,
            match_positions: vec![0u32; match_len],
            match_gen: vec![0u32; match_len],
            cur_gen: 1,
            touched_groups: vec![GroupList::new(); num_states],
            touched_positions: Vec::new(),
            touched_states: Vec::new(),
            base_offsets,
            results: Vec::with_capacity(num_states),
            seen_target: Vec::new(),
            all_targets: Vec::new(),
        }
    }

    fn reset(&mut self, initial_states: &[usize], num_groups: usize) {
        let len = initial_states.len();
        if len > self.current_states.len() {
            self.current_states.resize(len, 0);
            self.done.resize(len, false);
            self.end_states.resize(len, None);
            self.matched_bits.resize(len.saturating_mul(self.mask_words), 0);
            let new_len = len.saturating_mul(num_groups);
            self.match_positions.resize(new_len, 0);
            self.match_gen.resize(new_len, 0);
            self.touched_groups.resize(len, GroupList::new());
            self.base_offsets.clear();
            for i in 0..len {
                self.base_offsets.push(i * num_groups);
            }
            self.results.resize(len, (None, EdgeList::new()));
        }

        self.current_states[..len].clone_from_slice(initial_states);
        self.done.fill(false);
        self.active_indices.clear();
        self.end_states[..len].fill(None);

        // Advance generation instead of clearing `match_positions`.
        // If we ever wrap to 0, clear the generation array once.
        self.cur_gen = self.cur_gen.wrapping_add(1);
        if self.cur_gen == 0 {
            self.match_gen.fill(0);
            self.cur_gen = 1;
        }

        self.touched_positions.clear();

        // Clear touched_groups and matched_bits efficiently
        for &state_idx in &self.touched_states {
            if state_idx < self.touched_groups.len() {
                self.touched_groups[state_idx].clear();
            }
            if self.mask_words > 0 {
                let base = state_idx.saturating_mul(self.mask_words);
                let end = base.saturating_add(self.mask_words);
                if end <= self.matched_bits.len() {
                    self.matched_bits[base..end].fill(0);
                }
            }
        }
        self.touched_states.clear();

        if num_groups == 0 {
            return;
        }

        if self.results.len() < self.current_states.len() {
            self.results.resize_with(self.current_states.len(), || (None, EdgeList::new()));
        }
    }
}

/// Execute DFA from all initial states on a token, returning end states and unique target positions.
///
/// This is the hot-path variant used by vocab equivalence analysis. It avoids allocating/sorting
/// per-state edge lists; instead, it records (gid -> match position) in `match_positions` and
/// the set of touched gids in `touched_groups`, which `compute_chunk_signature` later hashes
/// using the precomputed suffix-cache.
fn compute_pos0_end_states_and_targets(
    pre: &PrecomputedDfa,
    scratch: &mut Pos0Scratch,
    slice: &[u8],
    initial_states: &[usize],
) {
    let num_states = initial_states.len();
    let num_groups = pre.num_groups;
    let len = slice.len();

    scratch.reset(initial_states, num_groups);

    // Prepare all_targets tracking
    let all_targets = &mut scratch.all_targets;

    // Clear seen_target only for positions we saw last time
    let seen_target = &mut scratch.seen_target;
    for &pos in all_targets.iter() {
        if pos < seen_target.len() {
            seen_target[pos] = false;
        }
    }
    all_targets.clear();

    let needed_seen = len + 1;
    if seen_target.len() < needed_seen {
        seen_target.resize(needed_seen, false);
    }

    let current_states = &mut scratch.current_states;
    let done = &mut scratch.done;
    let active_indices = &mut scratch.active_indices;
    let match_positions = &mut scratch.match_positions;
    let match_gen = &mut scratch.match_gen;
    let cur_gen = scratch.cur_gen;
    let touched_groups = &mut scratch.touched_groups;
    let touched_positions = &mut scratch.touched_positions;
    let touched_states = &mut scratch.touched_states;
    let matched_bits = &mut scratch.matched_bits;
    let mask_words = scratch.mask_words;
    let base_offsets = &scratch.base_offsets;

    active_indices.clear();
    let has_bytes = !slice.is_empty();
    let first_byte = if has_bytes { slice[0] } else { 0 };

    // Process initial finalizers
    for (i, &state) in initial_states.iter().enumerate() {
        let base = base_offsets[i];
        for f in &pre.finalizers[state] {
            let gid = f.gid;
            if gid < num_groups {
                let idx = base + gid;
                if match_gen[idx] != cur_gen {
                    match_gen[idx] = cur_gen;
                    match_positions[idx] = 0;
                    let groups = &mut touched_groups[i];
                    if groups.is_empty() {
                        touched_states.push(i);
                    }
                    groups.push(gid);

                    if mask_words > 0 {
                        let word = gid >> 6;
                        let bit = 1u64 << (gid & 63);
                        matched_bits[i * mask_words + word] |= bit;
                    }
                }
            }
        }
        if !pre.has_transitions[state] {
            done[i] = true;
            continue;
        }

        if has_bytes {
            let next_state = pre.transitions[state][first_byte as usize];
            if next_state == NONE_STATE {
                done[i] = true;
                continue;
            }
        }

        active_indices.push(i);
    }

    // Process each byte of the token
    if has_bytes && !active_indices.is_empty() {
        let mut active_len = active_indices.len();
        for (pos, &byte) in slice.iter().enumerate() {
            let position = (pos + 1) as u32;
            let mut next_len = 0usize;

            unsafe {
                for idx in 0..active_len {
                    let i = *active_indices.get_unchecked(idx);
                    let base = *base_offsets.get_unchecked(i);
                    let current = *current_states.get_unchecked(i);
                    let next_state = *pre
                        .transitions
                        .get_unchecked(current)
                        .get_unchecked(byte as usize);

                    if next_state != NONE_STATE {
                        let next_state = next_state as usize;
                        *current_states.get_unchecked_mut(i) = next_state;

                        for f in pre.finalizers.get_unchecked(next_state) {
                            let gid = f.gid;
                            if gid < num_groups {
                                let idx = base + gid;
                                let slot_pos = match_positions.get_unchecked_mut(idx);
                                let slot_gen = match_gen.get_unchecked_mut(idx);
                                let was_none = *slot_gen != cur_gen;
                                if f.non_greedy {
                                    if was_none {
                                        *slot_gen = cur_gen;
                                        *slot_pos = position;
                                    }
                                } else {
                                    *slot_gen = cur_gen;
                                    *slot_pos = position;
                                }

                                if was_none {
                                    let groups = touched_groups.get_unchecked_mut(i);
                                    if groups.is_empty() {
                                        touched_states.push(i);
                                    }
                                    groups.push(gid);

                                    if mask_words > 0 {
                                        let word = gid >> 6;
                                        let bit = 1u64 << (gid & 63);
                                        *matched_bits
                                            .get_unchecked_mut(i * mask_words + word) |= bit;
                                    }
                                }
                            }
                        }

                        let terminate = match pre.future_modes.get_unchecked(next_state) {
                            FutureMode::AlwaysTerminate => true,
                            FutureMode::AlwaysContinue => false,
                            FutureMode::Guarded(_guard) => {
                                if mask_words == 0 {
                                    true
                                } else {
                                    let guard_mask = pre
                                        .guard_masks
                                        .get_unchecked(next_state)
                                        .as_ref()
                                        .unwrap();
                                    let bits_base = i * mask_words;
                                    let mut all_met = true;
                                    for w in 0..mask_words {
                                        let required = *guard_mask.get_unchecked(w);
                                        if required
                                            & !*matched_bits.get_unchecked(bits_base + w)
                                            != 0
                                        {
                                            all_met = false;
                                            break;
                                        }
                                    }
                                    all_met
                                }
                            }
                        };

                        if terminate {
                            *done.get_unchecked_mut(i) = true;
                        }
                    } else {
                        *done.get_unchecked_mut(i) = true;
                    }

                    if !*done.get_unchecked(i) {
                        *active_indices.get_unchecked_mut(next_len) = i;
                        next_len += 1;
                    }
                }
            }

            active_len = next_len;
            if active_len == 0 {
                break;
            }
        }
    }

    // Collect end states and targets
    for i in 0..num_states {
        let end_state = if done[i] || !pre.has_transitions[current_states[i]] {
            None
        } else {
            Some(current_states[i])
        };

        scratch.end_states[i] = end_state;

        if num_groups > 0 {
            let base = base_offsets[i];
            for &gid in &touched_groups[i] {
                let pos_val = match_positions[base + gid];
                if pos_val > 0 {
                    let pos_usize = pos_val as usize;
                    if pos_usize <= len && !seen_target[pos_usize] {
                        seen_target[pos_usize] = true;
                        all_targets.push(pos_usize);
                    }
                }
            }
        }
    }

    // Results are stored in-place:
    // - `scratch.end_states[..num_states]`
    // - `scratch.all_targets`
}

impl SuffixScratch {
    fn new(num_groups: usize) -> Self {
        SuffixScratch {
            match_positions: vec![NONE_POS; num_groups],
            touched_positions: GroupList::new(),
            visited: Vec::new(),
            queue: Vec::new(),
            order: Vec::new(),
            nodes: Vec::new(),
            pos_hashes: Vec::new(),
            projected_cache: HashMap::new(),
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.match_positions.fill(NONE_POS);
        self.touched_positions.clear();
    }

    #[inline]
    fn ensure_capacity(&mut self, len: usize) {
        let needed = len + 1;

        // Only clear entries that were actually visited in the previous run
        for &pos in &self.queue {
            if pos < self.visited.len() {
                self.visited[pos] = false;
            }
            if pos < self.nodes.len() {
                self.nodes[pos] = None;
            }
            if pos < self.pos_hashes.len() {
                self.pos_hashes[pos] = 0;
            }
        }

        // Resize if needed
        if self.visited.len() < needed {
            self.visited.resize(needed, false);
        }
        if self.nodes.len() < needed {
            self.nodes.resize(needed, None);
        }
        if self.pos_hashes.len() < needed {
            self.pos_hashes.resize(needed, 0);
        }

        self.queue.clear();
        self.order.clear();
        self.projected_cache.clear();
    }
}


// --- CORE EXECUTION: SUFFIX HASH COMPUTATION ---

/// Execute DFA on a suffix starting from position base_pos.
#[inline]
fn execute_suffix(
    pre: &PrecomputedDfa,
    slice: &[u8],
    base_pos: usize,
    scratch: &mut SuffixScratch,
) -> (Option<usize>, EdgeList) {
    let num_groups = pre.num_groups;

    if num_groups > 0 {
        scratch.reset();
    }

    let match_positions = &mut scratch.match_positions;
    let touched = &mut scratch.touched_positions;

    let mut current = pre.start_state;
    let mut done = false;

    // Initial finalizers
    if num_groups > 0 {
        for f in &pre.finalizers[current] {
            let gid = f.gid;
            if gid < num_groups {
                let slot = &mut match_positions[gid];
                let was_none = *slot == NONE_POS;
                if f.non_greedy {
                    if was_none {
                        *slot = 0;
                    }
                } else {
                    *slot = 0;
                }
                if was_none {
                    touched.push(gid);
                }
            }
        }
    }

    if !pre.has_transitions[current] {
        done = true;
    }

    // Process each byte
    for (idx, &byte) in slice.iter().enumerate() {
        if done {
            break;
        }

        let next_state = pre.transitions[current][byte as usize];
        if next_state != NONE_STATE {
            let next_state = next_state as usize;
            current = next_state;
            let position = (idx + 1) as u32;

            if num_groups > 0 {
                for f in &pre.finalizers[current] {
                    let gid = f.gid;
                    if gid < num_groups {
                        let slot = &mut match_positions[gid];
                        let was_none = *slot == NONE_POS;
                        if f.non_greedy {
                            if was_none {
                                *slot = position;
                            }
                        } else {
                            *slot = position;
                        }

                        if was_none {
                            touched.push(gid);
                        }
                    }
                }
            }

            let terminate = match &pre.future_modes[current] {
                FutureMode::AlwaysTerminate => true,
                FutureMode::AlwaysContinue => false,
                FutureMode::Guarded(guard) => {
                    guard.iter().all(|&gid| match_positions[gid] != NONE_POS)
                }
            };

            if terminate {
                done = true;
            }
        } else {
            done = true;
        }
    }

    let end_state = if done || !pre.has_transitions[current] {
        None
    } else {
        Some(current)
    };

    let mut edges: EdgeList = SmallVec::new();
    if num_groups > 0 {
        touched.sort_unstable();
        for &gid in touched.iter() {
            let pos_val = match_positions[gid];
            if pos_val != NONE_POS && pos_val != 0 {
                edges.push((gid, base_pos + pos_val as usize));
            }
        }
    }

    (end_state, edges)
}

/// Compute suffix hashes incrementally, updating the cache.
fn compute_suffix_hashes_incremental(
    pre: &PrecomputedDfa,
    slice: &[u8],
    new_targets: &[usize],
    cache: &mut Vec<Option<u64>>,
    scratch: &mut SuffixScratch,
    _suffix_group_mask: Option<&[bool]>,
    group_to_class: Option<&[usize]>,
) {
    // Build suffix DAG (also used by projected hash computation)
    build_suffix_dag(pre, slice, new_targets, scratch);

    // Compute unprojected hashes from the DAG
    // Process in reverse order (bottom-up for DAG)
    scratch.order.sort_unstable_by(|a, b| b.cmp(a));

    for &pos in &scratch.order {
        if cache[pos].is_some() {
            continue;
        }
        if let Some((completion_hash, ref edges)) = scratch.nodes[pos] {
            let mut hasher = new_hasher();
            hasher.write_u64(completion_hash);

            for &(group_id, target) in edges.iter() {
                let target_hash = cache[target].unwrap_or(0);
                hasher.write_u64(group_id as u64);
                hasher.write_u64(target_hash);
            }

            cache[pos] = Some(hasher.finish());
        }
    }
    scratch.order.clear();
}

/// Build the suffix DAG without computing hashes.
/// After this call, `scratch.nodes[pos]` contains `(completion_hash, edges)` for each
/// reachable suffix position. The DAG can be used for projected hash computation.
fn build_suffix_dag(
    pre: &PrecomputedDfa,
    slice: &[u8],
    new_targets: &[usize],
    scratch: &mut SuffixScratch,
) {
    scratch.ensure_capacity(slice.len());

    // Queue positions that need computation
    for &pos in new_targets {
        if pos <= slice.len() && scratch.nodes[pos].is_none() && !scratch.visited[pos] {
            scratch.visited[pos] = true;
            scratch.queue.push(pos);
        }
    }

    if scratch.queue.is_empty() {
        return;
    }

    // BFS to discover all reachable positions
    let mut cursor = 0;
    while cursor < scratch.queue.len() {
        let pos = scratch.queue[cursor];
        cursor += 1;

        let (end_state, edges) = execute_suffix(pre, &slice[pos..], pos, scratch);

        for &(_, target) in &edges {
            if target <= slice.len() && scratch.nodes[target].is_none() && !scratch.visited[target] {
                scratch.visited[target] = true;
                scratch.queue.push(target);
            }
        }

        let completion_hash = end_state
            .map(|id| pre.completion_hash[id])
            .unwrap_or(pre.none_completion_hash);
        scratch.nodes[pos] = Some((completion_hash, edges));
        scratch.order.push(pos);
    }
}

/// Compute a projected suffix hash for a specific position, only considering
/// edges whose group is allowed after `parent_group`. Uses memoization via `projected_cache`.
///
/// This mirrors `prune_trellis_recursive`: at each level, only edges in
/// `ever_allowed_by_group[parent_group]` are included in the hash.
fn compute_projected_suffix_hash(
    pos: usize,
    parent_group: usize,
    ever_allowed_by_group: &[Vec<bool>],
    nodes: &[Option<(u64, EdgeList)>],
    projected_cache: &mut HashMap<(usize, usize), u64>,
    group_to_class: Option<&[usize]>,
) -> u64 {
    let key = (pos, parent_group);
    if let Some(&cached) = projected_cache.get(&key) {
        return cached;
    }

    let hash = if let Some((completion_hash, ref edges)) = nodes[pos] {
        let mut hasher = new_hasher();
        hasher.write_u64(completion_hash);

        let allowed = if parent_group < ever_allowed_by_group.len() {
            Some(&ever_allowed_by_group[parent_group])
        } else {
            None
        };

        for &(group_id, target) in edges.iter() {
            // Check if group_id is allowed after parent_group
            let is_allowed = match allowed {
                Some(mask) => group_id < mask.len() && mask[group_id],
                None => true, // No follow info -> allow all
            };
            if !is_allowed {
                continue;
            }
            // Recurse with group_id as the new parent
            let target_hash = compute_projected_suffix_hash(
                target, group_id, ever_allowed_by_group, nodes, projected_cache, group_to_class,
            );
            hasher.write_u64(group_id as u64);
            hasher.write_u64(target_hash);
        }

        hasher.finish()
    } else {
        0
    };

    projected_cache.insert(key, hash);
    hash
}

// --- SIGNATURE COMPUTATION ---

/// Compute the signature for a token given a chunk of initial states.
fn compute_chunk_signature(
    pre: &PrecomputedDfa,
    token: &[u8],
    chunk_states: &[usize],
    pos0: &mut Pos0Scratch,
    suffix_scratch: &mut SuffixScratch,
    cache: &mut Vec<Option<u64>>,
    suffix_group_mask: Option<&[bool]>,
    ever_allowed_by_group: Option<&[Vec<bool>]>,
    group_to_class: Option<&[usize]>,
) -> u64 {
    compute_pos0_end_states_and_targets(pre, pos0, token, chunk_states);

    // Only compute suffix hashes when there are match targets
    if !pos0.all_targets.is_empty() {
        compute_suffix_hashes_incremental(pre, token, &pos0.all_targets, cache, suffix_scratch, suffix_group_mask, group_to_class);
    }

    // If ever_allowed_by_group is provided, we'll compute projected hashes
    // that prune suffix edges based on which group was matched at position 0.
    let use_projected = ever_allowed_by_group.is_some();

    let num_groups = pre.num_groups;
    let include_groups = num_groups > 0;

    // Fast path: combine per-state signatures using wrapping_mul (avoids creating
    // a top-level AHasher). Only states with group matches need a full hasher.
    let mut sig: u64 = HASH_SEED3;
    for i in 0..chunk_states.len() {
        let completion_hash = pos0.end_states[i]
            .map(|id| pre.completion_hash[id])
            .unwrap_or(pre.none_completion_hash);

        let state_sig = if include_groups && !pos0.touched_groups[i].is_empty() {
            // This state has group matches - hash them
            let groups = &mut pos0.touched_groups[i];
            if groups.len() > 1 {
                groups.sort_unstable();
            }
            let base = pos0.base_offsets[i];
            let mut h = new_hasher();
            h.write_u64(completion_hash);
            for &gid in groups.iter() {
                let pos_val = pos0.match_positions[base + gid];
                if pos_val > 0 {
                    let target_hash = if use_projected {
                        let ea = ever_allowed_by_group.unwrap();
                        compute_projected_suffix_hash(
                            pos_val as usize,
                            gid,
                            ea,
                            &suffix_scratch.nodes,
                            &mut suffix_scratch.projected_cache,
                            group_to_class,
                        )
                    } else {
                        cache[pos_val as usize].unwrap_or(0)
                    };
                    h.write_u64(gid as u64);
                    h.write_u64(target_hash);
                }
            }
            h.finish()
        } else {
            // No group matches at this state - just use completion hash directly
            completion_hash
        };

        // Order-preserving combination of per-state signatures
        sig = sig.wrapping_mul(HASH_SEED1).wrapping_add(state_sig);
    }

    sig
}

// --- MAIN ENTRY POINT ---

/// Find vocab equivalence classes of tokens based on DFA behavior.
/// Uses iterative state-based refinement with batching and parallel processing.
/// 
/// Note: For large state counts, the caller should pre-reduce using
/// `state_equivalence_analysis::find_state_equivalence_classes`
/// before calling this function. This is typically done in constraint.rs.
///
/// # Arguments
/// * `regex` - The tokenizer DFA
/// * `strings` - Vocabulary tokens to analyze
/// * `initial_states` - Tokenizer states to consider for equivalence
///
/// # Returns
/// Sets of token indices that are equivalent (produce identical parsing behavior).
pub fn find_vocab_equivalence_classes<S: AsRef<[u8]> + Sync>(
    regex: &Tokenizer,
    strings: &[S],
    initial_states: &[usize],
) -> VocabEquivalenceResult {
    find_vocab_equivalence_classes_with_follow(regex, strings, initial_states, None, None, None)
}

/// Find vocab equivalence classes with optional follow-set pruning.
///
/// `suffix_group_mask`: if provided, suffix hashes will only include edges for groups where
/// `mask[gid] == true`. Groups not in the mask are ignored in suffix positions, causing
/// tokens that differ only in those groups to be merged. The mask should be `true` for
/// groups that can appear after any other group (i.e., groups that appear in at least one
/// follow set).
///
/// `ever_allowed_by_group`: if provided, per-group follow masks. `ever_allowed_by_group[g]`
/// is a bool mask: `mask[h] == true` means group h can follow group g. When this is
/// provided, suffix hashes use projected computation that prunes edges per-context.
pub fn find_vocab_equivalence_classes_with_follow<S: AsRef<[u8]> + Sync>(
    regex: &Tokenizer,
    strings: &[S],
    initial_states: &[usize],
    suffix_group_mask: Option<&[bool]>,
    ever_allowed_by_group: Option<&[Vec<bool>]>,
    group_to_class: Option<&[usize]>,
) -> VocabEquivalenceResult {
    use std::time::Instant;
    
    let profile_eq = std::env::var("PROFILE_BUILD_TOKENIZER").is_ok();
    let total_start = Instant::now();
    let pre = precompute_dfa(regex);
    let precompute_time = total_start.elapsed();
    if profile_eq {
        eprintln!("TIMING: equiv::precompute_dfa {:?}", precompute_time);
    }
    let reduced_initial_states: Vec<usize> = initial_states.to_vec();

    if is_debug_level_enabled(3) {
        crate::debug!(
            3,
            "fast vocab equivalence: num_states={} num_groups={} precompute={:?}",
            reduced_initial_states.len(),
            pre.num_groups,
            precompute_time,
        );
    }

    let num_tokens = strings.len();
    let num_states = reduced_initial_states.len();

    if num_states == 0 || num_tokens == 0 {
        return BTreeSet::from_iter(vec![(0..num_tokens).collect()]);
    }

    // Analyze state transition sparsity for large state sets
    if num_states > 2000 {
        let mut total_transitions = 0usize;
        let mut states_with_few_transitions = 0usize;
        for &sid in &reduced_initial_states {
            let trans = &pre.transitions[sid];
            let count = trans.iter().filter(|&&t| t != NONE_STATE).count();
            total_transitions += count;
            if count < 10 {
                states_with_few_transitions += 1;
            }
        }
        let avg_transitions = total_transitions as f64 / num_states as f64;
        crate::debug!(
            3,
            "State transition analysis: avg_transitions={:.1}, sparse_states={}/{} ({:.1}%)",
            avg_transitions,
            states_with_few_transitions,
            num_states,
            100.0 * states_with_few_transitions as f64 / num_states as f64
        );
    }

    let num_groups = pre.num_groups;

    // === Self-loop trie pruning analysis ===
    if profile_eq {
        let analysis_trie = VocabTrie::build(strings);
        let mut total_prunable = 0u64;
        let mut total_checks = 0u64;
        let mut total_trie_edges = 0u64;
        let mut total_none_tokens = 0u64;
        let mut per_state: Vec<(usize, u64, u64, u64, u64)> = Vec::new();

        // Total bytes in flat approach: sum of all token lengths
        let total_token_bytes: u64 = strings.iter().map(|s| s.as_ref().len() as u64).sum();

        // For each initial state, walk the trie and count prunable tokens
        for &state_id in &reduced_initial_states {
            let (prunable, checks, trie_edges, none_tok) = count_selfloop_prunable(
                &analysis_trie, &pre, state_id, 0,
            );
            total_prunable += prunable;
            total_checks += checks;
            total_trie_edges += trie_edges;
            total_none_tokens += none_tok;
            per_state.push((state_id, prunable, checks, trie_edges, none_tok));
        }

        let flat_total_transitions = total_token_bytes * num_states as u64;
        eprintln!(
            "TIMING: equiv::selfloop_analysis: {} states × {} tokens = {} pairs",
            num_states, num_tokens, num_states as u64 * num_tokens as u64,
        );
        eprintln!(
            "  tokens prunable={} ({:.1}%), none_state={} ({:.1}%), remaining={} ({:.1}%)",
            total_prunable,
            100.0 * total_prunable as f64 / (num_states as f64 * num_tokens as f64),
            total_none_tokens,
            100.0 * total_none_tokens as f64 / (num_states as f64 * num_tokens as f64),
            num_states as u64 * num_tokens as u64 - total_prunable - total_none_tokens,
            100.0 * (num_states as u64 * num_tokens as u64 - total_prunable - total_none_tokens) as f64 / (num_states as f64 * num_tokens as f64),
        );
        eprintln!(
            "  byte transitions: flat={}, trie_walk={} ({:.1}% of flat), trie_nodes={}",
            flat_total_transitions,
            total_trie_edges,
            100.0 * total_trie_edges as f64 / flat_total_transitions as f64,
            analysis_trie.num_nodes(),
        );
        // Per-state breakdown: show states with highest pruning
        per_state.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!("  Per-state pruning (top 10):");
        for &(state_id, prunable, checks, trie_edges, none_tok) in per_state.iter().take(10) {
            let pct = 100.0 * prunable as f64 / num_tokens as f64;
            let trans = &pre.transitions[state_id];
            let self_loop_count = (0..256usize).filter(|&b| trans[b] == state_id as u32).count();
            let non_none_count = (0..256usize).filter(|&b| trans[b] != NONE_STATE).count();
            let finalizer_groups: Vec<usize> = pre.finalizers[state_id].iter().map(|f| f.gid).collect();
            eprintln!(
                "    state={}: prunable={} ({:.1}%), none_st={}, trie_edges={}, self_loop={}/256, non_none={}/256, finalizers={:?}",
                state_id, prunable, pct, none_tok, trie_edges, self_loop_count, non_none_count, finalizer_groups,
            );
        }
    }

    // Process states in batches for memory efficiency.
    // Smaller batches improve cache locality for match_positions array
    // (batch_size * num_groups * 4 bytes per thread) and enable early pruning of singletons.
    let batch_size = if num_states < 200 { num_states } else { 200 };

    let mut active_indices: Vec<usize> = (0..num_tokens).collect();
    let mut partition: Vec<usize> = vec![0; num_tokens];
    let mut next_class_id = 1usize;

    if is_debug_level_enabled(4) {
        crate::debug!(
            4,
            "  Iterative refinement: {} tokens, {} states, batch_size={}",
            num_tokens,
            num_states,
            batch_size
        );
    }

    let mut batch_count = 0;
    
    // Timing accumulators
    let mut total_refine_time = std::time::Duration::ZERO;

    for batch_start in (0..num_states).step_by(batch_size) {
        if active_indices.is_empty() {
            break;
        }

        let batch_end = (batch_start + batch_size).min(num_states);
        let batch = &reduced_initial_states[batch_start..batch_end];

        // Compute partial signatures for active tokens
        let batch_start_time = Instant::now();
        let active_sigs: Vec<(usize, u64)> = active_indices
            .par_iter()
            .map_init(
                || {
                    (
                        Pos0Scratch::new(batch.len(), num_groups),
                        SuffixScratch::new(num_groups),
                        vec![None; 256],
                    )
                },
                |state, &token_idx| {
                    let (scratch_pos0, scratch_suffix, scratch_cache) = state;
                    let token = strings[token_idx].as_ref();

                    if scratch_cache.len() <= token.len() {
                        scratch_cache.resize(token.len() + 1, None);
                    }
                    scratch_cache.iter_mut().for_each(|x| *x = None);

                    let sig = compute_chunk_signature(
                        &pre, token, batch, scratch_pos0, scratch_suffix, scratch_cache,
                        suffix_group_mask, ever_allowed_by_group, group_to_class,
                    );
                    (token_idx, sig)
                },
            )
            .collect();
        let batch_compute_time = batch_start_time.elapsed();

        // Group by (old_class, new_signature) to refine partition
        let refine_start = Instant::now();
        let mut refinement: HashMap<(usize, u64), Vec<usize>> =
            HashMap::with_capacity(active_sigs.len() / 2);
        for (token_idx, sig) in active_sigs {
            let old_class = partition[token_idx];
            refinement
                .entry((old_class, sig))
                .or_insert_with(Vec::new)
                .push(token_idx);
        }

        // Group refinement entries by old_class
        let mut by_old_class: HashMap<usize, Vec<(u64, Vec<usize>)>> = HashMap::new();
        for ((old_class, sig), tokens) in refinement {
            by_old_class
                .entry(old_class)
                .or_insert_with(Vec::new)
                .push((sig, tokens));
        }

        // Update partition and find still-active tokens
        let mut new_active_indices = Vec::with_capacity(active_indices.len());

        for (_old_class, sub_groups) in by_old_class {
            let mut first = true;
            for (_sig, tokens) in sub_groups {
                let class_to_use = if first {
                    first = false;
                    _old_class
                } else {
                    let id = next_class_id;
                    next_class_id += 1;
                    id
                };

                for &token_idx in &tokens {
                    partition[token_idx] = class_to_use;
                }

                if tokens.len() > 1 {
                    new_active_indices.extend(tokens);
                }
            }
        }
        total_refine_time += refine_start.elapsed();

        active_indices = new_active_indices;
        batch_count += 1;

        if is_debug_level_enabled(5) {
            let num_classes = {
                let mut seen: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
                for &c in &partition {
                    seen.insert(c);
                }
                seen.len()
            };
            crate::debug!(
                5,
                "    Batch {}: {} active tokens, {} classes, compute={:?}",
                batch_count,
                active_indices.len(),
                num_classes,
                batch_compute_time,
            );
        }
    }

    if profile_eq {
        eprintln!("TIMING: equiv::batch_compute total={:?} refine={:?} batches={}", 
            total_start.elapsed(), total_refine_time, batch_count);
    }

    // Build final groups from partition
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::with_capacity(next_class_id);
    for (token_idx, &class_id) in partition.iter().enumerate() {
        groups.entry(class_id).or_insert_with(Vec::new).push(token_idx);
    }

    if is_debug_level_enabled(4) {
        crate::debug!(
            4,
            "  Computed {} vocab equivalence classes in {} batches",
            groups.len(),
            batch_count
        );
    }

    groups.into_values().collect()
}

// --- TRIE-BASED BATCH SIGNATURE COMPUTATION ---

/// Compact byte-level trie for vocabulary prefix sharing.
/// Reduces DFA transitions by ~69% by sharing work for common token prefixes.
struct VocabTrie {
    /// Flat array of trie nodes. Node 0 is the root.
    nodes: Vec<TrieNode>,
}

struct TrieNode {
    /// Children sorted by byte for deterministic traversal.
    /// (byte, child_node_index)
    children: SmallVec<[(u8, u32); 4]>,
    /// Token index if this node is a leaf (complete token), else u32::MAX.
    token_idx: u32,
    /// Number of tokens reachable from this subtree (for active filtering).
    subtree_size: u32,
    /// Set of all bytes that appear in this subtree (edges from this node and all descendants).
    /// Represented as a 256-bit bitset (4 × u64).
    future_bytes: [u64; 4],
}

impl VocabTrie {
    fn build<S: AsRef<[u8]>>(tokens: &[S]) -> Self {
        let mut nodes = Vec::with_capacity(tokens.len() * 2);
        nodes.push(TrieNode {
            children: SmallVec::new(),
            token_idx: u32::MAX,
            subtree_size: 0,
            future_bytes: [0u64; 4],
        });

        for (idx, token) in tokens.iter().enumerate() {
            let mut current = 0u32;
            for &byte in token.as_ref() {
                let pos = nodes[current as usize]
                    .children
                    .iter()
                    .position(|&(b, _)| b == byte);
                current = match pos {
                    Some(p) => nodes[current as usize].children[p].1,
                    None => {
                        let new_idx = nodes.len() as u32;
                        nodes.push(TrieNode {
                            children: SmallVec::new(),
                            token_idx: u32::MAX,
                            subtree_size: 0,
                            future_bytes: [0u64; 4],
                        });
                        nodes[current as usize].children.push((byte, new_idx));
                        new_idx
                    }
                };
            }
            nodes[current as usize].token_idx = idx as u32;
        }

        // Sort children by byte for deterministic ordering
        for node in &mut nodes {
            node.children.sort_unstable_by_key(|&(b, _)| b);
        }

        // Compute subtree sizes and future byte sets (post-order)
        fn compute_subtree_info(nodes: &mut [TrieNode], idx: u32) -> (u32, [u64; 4]) {
            let has_token = if nodes[idx as usize].token_idx != u32::MAX { 1 } else { 0 };
            let children: SmallVec<[(u8, u32); 4]> = nodes[idx as usize].children.clone();
            let mut size = has_token;
            let mut future = [0u64; 4];
            for &(byte, child_idx) in &children {
                let (child_size, child_future) = compute_subtree_info(nodes, child_idx);
                size += child_size;
                // Add edge byte to future set
                let word = (byte as usize) >> 6;
                let bit = 1u64 << (byte & 63);
                future[word] |= bit;
                // Union with child's future set
                for i in 0..4 {
                    future[i] |= child_future[i];
                }
            }
            nodes[idx as usize].subtree_size = size;
            nodes[idx as usize].future_bytes = future;
            (size, future)
        }
        compute_subtree_info(&mut nodes, 0);

        VocabTrie { nodes }
    }

    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Check if all bytes in the future byte set have self-loop transitions at the given DFA state.
    /// Returns true if transitions[state][b] == state for all bytes b in future_bytes.
    fn all_future_bytes_self_loop(&self, node_idx: u32, transitions: &[[u32; 256]], state: usize) -> bool {
        let fb = &self.nodes[node_idx as usize].future_bytes;
        // If no future bytes, trivially true (leaf node)
        if fb[0] | fb[1] | fb[2] | fb[3] == 0 {
            return true;
        }
        let trans = &transitions[state];
        let state_u32 = state as u32;
        for word_idx in 0..4 {
            let mut bits = fb[word_idx];
            while bits != 0 {
                let bit_pos = bits.trailing_zeros() as usize;
                let byte_val = word_idx * 64 + bit_pos;
                if trans[byte_val] != state_u32 {
                    return false;
                }
                bits &= bits - 1; // Clear lowest set bit
            }
        }
        true
    }
}

/// Count tokens prunable by self-loop optimization for a given initial DFA state.
/// Returns (prunable_tokens, nodes_checked, trie_edges_walked, none_state_tokens).
fn count_selfloop_prunable(
    trie: &VocabTrie,
    pre: &PrecomputedDfa,
    initial_state: usize,
    root_node: u32,
) -> (u64, u64, u64, u64) {
    let mut prunable = 0u64;
    let mut checks = 0u64;
    let mut trie_edges = 0u64;
    let mut none_tokens = 0u64;

    // DFS stack: (node_idx, dfa_state)
    let mut stack: Vec<(u32, usize)> = vec![(root_node, initial_state)];

    while let Some((node_idx, state)) = stack.pop() {
        checks += 1;
        let node = &trie.nodes[node_idx as usize];

        // Check self-loop condition: all future bytes self-loop at current state
        if trie.all_future_bytes_self_loop(node_idx, &pre.transitions, state) {
            prunable += node.subtree_size as u64;
            continue;
        }

        // Not prunable — recurse into children
        for &(byte, child_idx) in &node.children {
            let next_state = pre.transitions[state][byte as usize];
            if next_state != NONE_STATE {
                trie_edges += 1;
                stack.push((child_idx, next_state as usize));
            } else {
                none_tokens += trie.nodes[child_idx as usize].subtree_size as u64;
            }
        }
    }

    (prunable, checks, trie_edges, none_tokens)
}
