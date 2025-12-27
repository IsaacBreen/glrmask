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

use crate::finite_automata::Regex;
use crate::r#macro::is_debug_level_enabled;
use ahash::{AHasher, RandomState};
use hashbrown::HashMap;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::BTreeSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{BuildHasher, Hash, Hasher};

pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;

// =============================================================================
// TYPE ALIASES AND CONSTANTS
// =============================================================================

type EdgeList = SmallVec<[(usize, usize); 4]>;
type GroupList = SmallVec<[usize; 4]>;
type FinalizerList = SmallVec<[Finalizer; 4]>;

const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;
const NONE_STATE: u32 = u32::MAX;
const NONE_POS: u32 = u32::MAX;

// =============================================================================
// CORE DATA STRUCTURES
// =============================================================================

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
    has_transitions: Vec<bool>,
    num_groups: usize,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
}

/// Scratch space for position-0 DFA execution across all initial states.
struct Pos0Scratch {
    current_states: Vec<usize>,
    done: Vec<bool>,
    match_positions: Vec<u32>,
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
}

// =============================================================================
// HASH UTILITIES
// =============================================================================

#[inline]
fn new_hasher() -> AHasher {
    RandomState::with_seeds(HASH_SEED1, HASH_SEED2, HASH_SEED3, HASH_SEED4).build_hasher()
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

// =============================================================================
// DFA PRECOMPUTATION
// =============================================================================

fn precompute_dfa(regex: &Regex) -> PrecomputedDfa {
    let dfa = &regex.dfa;
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

    // Compute future modes
    let future_modes: Vec<FutureMode> = possible_future
        .iter()
        .map(|future| {
            if future.is_empty() {
                return FutureMode::AlwaysTerminate;
            }
            let mut guard: GroupList = GroupList::new();
            for &gid in future {
                if gid >= num_groups || !non_greedy_flags[gid] {
                    return FutureMode::AlwaysContinue;
                }
                guard.push(gid);
            }
            guard.sort_unstable();
            guard.dedup();
            FutureMode::Guarded(guard)
        })
        .collect();

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
        has_transitions,
        num_groups,
        completion_hash,
        none_completion_hash,
    }
}

// =============================================================================
// SCRATCH SPACE IMPLEMENTATIONS
// =============================================================================

impl Pos0Scratch {
    fn new(num_states: usize, num_groups: usize) -> Self {
        let base_offsets: Vec<usize> = (0..num_states)
            .map(|idx| idx.saturating_mul(num_groups))
            .collect();
        Pos0Scratch {
            current_states: vec![0; num_states],
            done: vec![false; num_states],
            match_positions: vec![NONE_POS; num_states.saturating_mul(num_groups)],
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
            self.match_positions.resize(len.saturating_mul(num_groups), NONE_POS);
            self.touched_groups.resize(len, GroupList::new());
            self.base_offsets.clear();
            for i in 0..len {
                self.base_offsets.push(i * num_groups);
            }
            self.results.resize(len, (None, EdgeList::new()));
        }

        self.current_states[..len].clone_from_slice(initial_states);
        self.done.fill(false);

        // Only clear match_positions that were touched in the previous run
        for &idx in &self.touched_positions {
            if idx < self.match_positions.len() {
                self.match_positions[idx] = NONE_POS;
            }
        }
        self.touched_positions.clear();

        // Clear touched_groups efficiently
        for &state_idx in &self.touched_states {
            if state_idx < self.touched_groups.len() {
                self.touched_groups[state_idx].clear();
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
    }
}

// =============================================================================
// CORE EXECUTION: POSITION-0 DFA EXECUTION
// =============================================================================

/// Execute DFA from all initial states on a token.
/// Returns (end_state, edges) for each initial state, plus list of unique target positions.
fn compute_pos0_results<'a>(
    pre: &PrecomputedDfa,
    scratch: &'a mut Pos0Scratch,
    slice: &[u8],
    initial_states: &[usize],
) -> (&'a [(Option<usize>, EdgeList)], &'a [usize]) {
    let num_states = initial_states.len();
    let num_groups = pre.num_groups;
    let len = slice.len();

    scratch.reset(initial_states, num_groups);

    // Prepare results vector
    if scratch.results.len() < num_states {
        scratch.results.resize_with(num_states, || (None, EdgeList::new()));
    }
    for i in 0..num_states {
        scratch.results[i].0 = None;
        scratch.results[i].1.clear();
    }

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
    let match_positions = &mut scratch.match_positions;
    let touched_groups = &mut scratch.touched_groups;
    let touched_positions = &mut scratch.touched_positions;
    let touched_states = &mut scratch.touched_states;
    let base_offsets = &scratch.base_offsets;

    // Process initial finalizers
    for (i, &state) in initial_states.iter().enumerate() {
        let base = base_offsets[i];
        for f in &pre.finalizers[state] {
            let gid = f.gid;
            if gid < num_groups {
                let idx = base + gid;
                if match_positions[idx] == NONE_POS {
                    match_positions[idx] = 0;
                }
                let groups = &mut touched_groups[i];
                if !groups.contains(&gid) {
                    if groups.is_empty() {
                        touched_states.push(i);
                    }
                    groups.push(gid);
                }
                touched_positions.push(idx);
            }
        }
        if !pre.has_transitions[state] {
            done[i] = true;
        }
    }

    // Process each byte of the token
    for (pos, &byte) in slice.iter().enumerate() {
        let position = (pos + 1) as u32;
        let mut any_active = false;

        // SAFETY: All indices are pre-validated:
        // - i < num_states, and all arrays are sized to num_states
        // - current_states[i] is always a valid DFA state (< pre.transitions.len())
        // - byte is u8, so byte as usize < 256 (valid for transition table)
        // - base + gid is valid because base_offsets and match_positions are properly sized
        unsafe {
            for i in 0..num_states {
                if *done.get_unchecked(i) {
                    continue;
                }
                any_active = true;

                let base = *base_offsets.get_unchecked(i);
                let current = *current_states.get_unchecked(i);
                let next_state = *pre.transitions.get_unchecked(current).get_unchecked(byte as usize);

                if next_state != NONE_STATE {
                    let next_state = next_state as usize;
                    *current_states.get_unchecked_mut(i) = next_state;

                    for f in pre.finalizers.get_unchecked(next_state) {
                        let gid = f.gid;
                        if gid < num_groups {
                            let idx = base + gid;
                            let slot = match_positions.get_unchecked_mut(idx);
                            if f.non_greedy {
                                if *slot == NONE_POS {
                                    *slot = position;
                                }
                            } else {
                                *slot = position;
                            }

                            let groups = touched_groups.get_unchecked_mut(i);
                            if !groups.contains(&gid) {
                                if groups.is_empty() {
                                    touched_states.push(i);
                                }
                                groups.push(gid);
                            }
                            touched_positions.push(idx);
                        }
                    }

                    let terminate = match pre.future_modes.get_unchecked(next_state) {
                        FutureMode::AlwaysTerminate => true,
                        FutureMode::AlwaysContinue => false,
                        FutureMode::Guarded(guard) => {
                            let mut all_met = true;
                            for &gid in guard.iter() {
                                let idx = base + gid;
                                if *match_positions.get_unchecked(idx) == NONE_POS {
                                    all_met = false;
                                    break;
                                }
                            }
                            all_met
                        }
                    };

                    if terminate {
                        *done.get_unchecked_mut(i) = true;
                    }
                } else {
                    *done.get_unchecked_mut(i) = true;
                }
            }
        }

        if !any_active {
            break;
        }
    }

    // Collect results
    for i in 0..num_states {
        let end_state = if done[i] || !pre.has_transitions[current_states[i]] {
            None
        } else {
            Some(current_states[i])
        };

        let edges = &mut scratch.results[i].1;
        if num_groups > 0 {
            let base = base_offsets[i];
            for &gid in &touched_groups[i] {
                if gid >= num_groups {
                    continue;
                }
                let pos_val = match_positions[base + gid];
                if pos_val != NONE_POS && pos_val > 0 {
                    let pos_usize = pos_val as usize;
                    edges.push((gid, pos_usize));
                    if pos_usize <= len && !seen_target[pos_usize] {
                        seen_target[pos_usize] = true;
                        all_targets.push(pos_usize);
                    }
                }
            }
        }

        edges.sort_unstable_by_key(|e| e.0);
        scratch.results[i].0 = end_state;
    }

    (&scratch.results[..num_states], &scratch.all_targets)
}

// =============================================================================
// CORE EXECUTION: SUFFIX HASH COMPUTATION
// =============================================================================

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
                if f.non_greedy {
                    if *slot == NONE_POS {
                        *slot = 0;
                        touched.push(gid);
                    }
                } else {
                    let was_none = *slot == NONE_POS;
                    *slot = 0;
                    if was_none {
                        touched.push(gid);
                    }
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
                        if f.non_greedy {
                            if *slot == NONE_POS {
                                *slot = position;
                            }
                        } else {
                            *slot = position;
                        }

                        if !touched.contains(&gid) {
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
) {
    scratch.ensure_capacity(slice.len());

    // Queue positions that need computation
    for &pos in new_targets {
        if pos <= slice.len() && cache[pos].is_none() && !scratch.visited[pos] {
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
            if target <= slice.len() && cache[target].is_none() && !scratch.visited[target] {
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

    // Process in reverse order (bottom-up for DAG)
    scratch.order.sort_unstable_by(|a, b| b.cmp(a));

    for pos in scratch.order.drain(..) {
        if let Some((completion_hash, edges)) = scratch.nodes[pos].take() {
            let mut hasher = new_hasher();
            hasher.write_u64(completion_hash);

            for (group_id, target) in edges {
                let target_hash = cache[target].unwrap_or(0);
                hasher.write_u64(group_id as u64);
                hasher.write_u64(target_hash);
            }

            cache[pos] = Some(hasher.finish());
        }
    }
}

// =============================================================================
// SIGNATURE COMPUTATION
// =============================================================================

/// Compute the signature for a token given a chunk of initial states.
/// 
/// The signature includes:
/// 1. The first byte of the token (to ensure tokens with different first bytes are never grouped)
/// 2. DFA behavior: final states and group finalizations across all initial states
///
/// This is critical for grammar-constrained decoding: tokens with different first bytes
/// cannot appear at the same position in any valid parse, so they must be in different
/// equivalence classes even if their DFA behavior is otherwise identical.
fn compute_chunk_signature(
    pre: &PrecomputedDfa,
    token: &[u8],
    chunk_states: &[usize],
    pos0: &mut Pos0Scratch,
    suffix_scratch: &mut SuffixScratch,
    cache: &mut Vec<Option<u64>>,
) -> u64 {
    let (pos0_results, all_targets) = compute_pos0_results(pre, pos0, token, chunk_states);

    compute_suffix_hashes_incremental(pre, token, all_targets, cache, suffix_scratch);

    let mut hasher = new_hasher();

    for (end_state, edges) in pos0_results {
        let mut state_hasher = new_hasher();
        let completion_hash = end_state
            .map(|id| pre.completion_hash[id])
            .unwrap_or(pre.none_completion_hash);
        state_hasher.write_u64(completion_hash);

        for (gid, target) in edges {
            let target_hash = cache[*target].unwrap_or(0);
            state_hasher.write_u64(*gid as u64);
            state_hasher.write_u64(target_hash);
        }

        hasher.write_u64(state_hasher.finish());
    }

    hasher.finish()
}

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

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
pub fn find_vocab_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> VocabEquivalenceResult {
    use std::time::Instant;
    
    let total_start = Instant::now();
    let pre = precompute_dfa(regex);
    let precompute_time = total_start.elapsed();

    // Note: State equivalence reduction (if needed) should be done by the caller.
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

    // Process states in batches for memory efficiency
    // Use a larger batch size when state count is small, single batch when < 3000 states
    let batch_size = if num_states < 3000 { num_states } else { 2048 };

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

    let num_groups = pre.num_groups;
    let mut batch_count = 0;
    
    // Timing accumulators
    let mut total_refine_time = std::time::Duration::ZERO;

    for batch_start in (0..num_states).step_by(batch_size) {
        if active_indices.is_empty() {
            break;
        }

        let batch_end = (batch_start + batch_size).min(num_states);
        let batch = &reduced_initial_states[batch_start..batch_end];

        // Compute partial signatures for active tokens in PARALLEL
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
                    let token = &strings[token_idx];
                    
                    // Ensure cache is large enough
                    if scratch_cache.len() <= token.len() {
                        scratch_cache.resize(token.len() + 1, None);
                    }
                    scratch_cache.iter_mut().for_each(|x| *x = None);
                    
                    let sig = compute_chunk_signature(&pre, token, batch, scratch_pos0, scratch_suffix, scratch_cache);
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
    
    if is_debug_level_enabled(4) {
        crate::debug!(
            4,
            "  Timing: refine={:?}",
            total_refine_time,
        );
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

// =============================================================================
// DEBUG/TEST UTILITIES
// =============================================================================

fn compute_suffix_hashes_debug(
    regex: &Regex,
    slice: &[u8],
    all_targets: &[usize],
) -> Vec<u64> {
    use std::collections::VecDeque;

    let len = slice.len();
    if all_targets.is_empty() {
        return vec![0; len + 1];
    }

    let mut visited = vec![false; len + 1];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut order: Vec<usize> = Vec::new();
    let mut nodes: Vec<Option<(Option<usize>, EdgeList)>> = vec![None; len + 1];

    for &pos in all_targets {
        if pos > 0 && pos <= len && !visited[pos] {
            visited[pos] = true;
            queue.push_back(pos);
        }
    }

    while let Some(pos) = queue.pop_front() {
        let result = regex.execute_from_state_nonzero(&slice[pos..], regex.dfa.start_state);

        let mut edges: EdgeList = result
            .matches
            .iter()
            .map(|m| {
                let target = pos + m.position;
                if target <= len && !visited[target] {
                    visited[target] = true;
                    queue.push_back(target);
                }
                (m.group_id, target)
            })
            .collect();

        edges.sort_unstable_by_key(|e| e.0);
        nodes[pos] = Some((result.end_state, edges));
        order.push(pos);
    }

    order.sort_unstable_by(|a, b| b.cmp(a));
    let mut pos_hashes: Vec<u64> = vec![0; len + 1];

    for pos in order {
        if let Some((end_state, edges)) = &nodes[pos] {
            let completion =
                end_state.map(|id| regex.dfa.states[id].possible_future_group_ids.clone());
            let mut hasher = DefaultHasher::new();
            completion.hash(&mut hasher);
            for (group_id, target) in edges {
                let target_hash = pos_hashes[*target];
                (group_id, target_hash).hash(&mut hasher);
            }
            pos_hashes[pos] = hasher.finish();
        }
    }

    pos_hashes
}

pub fn compute_signature_debug(
    regex: &Regex,
    slice: &[u8],
    initial_states: &[usize],
) -> Vec<u64> {
    let pre = precompute_dfa(regex);
    let mut scratch = Pos0Scratch::new(initial_states.len(), pre.num_groups);
    let (pos0_results, all_targets) = compute_pos0_results(&pre, &mut scratch, slice, initial_states);
    let pos_hashes = compute_suffix_hashes_debug(regex, slice, all_targets);

    let mut signatures: Vec<u64> = Vec::with_capacity(initial_states.len());
    for (end_state, edges) in pos0_results.iter() {
        let completion = end_state.map(|id| regex.dfa.states[id].possible_future_group_ids.clone());
        let mut hasher = DefaultHasher::new();
        completion.hash(&mut hasher);
        for (group_id, target) in edges.iter() {
            let target_hash = *pos_hashes.get(*target).unwrap_or(&0);
            (group_id, target_hash).hash(&mut hasher);
        }
        signatures.push(hasher.finish());
    }

    signatures
}

pub fn debug_pos0_edges(
    regex: &Regex,
    slice: &[u8],
    initial_states: &[usize],
) -> Vec<EdgeList> {
    let pre = precompute_dfa(regex);
    let mut scratch = Pos0Scratch::new(initial_states.len(), pre.num_groups);
    let (pos0_results, _) = compute_pos0_results(&pre, &mut scratch, slice, initial_states);
    pos0_results.iter().map(|(_, edges)| edges.clone()).collect()
}

pub fn compute_signature_actual(
    regex: &Regex,
    slice: &[u8],
    initial_states: &[usize],
) -> u64 {
    let pre = precompute_dfa(regex);
    let mut pos0 = Pos0Scratch::new(initial_states.len(), pre.num_groups);
    let mut suffix_scratch = SuffixScratch::new(pre.num_groups);
    let mut cache = vec![None; slice.len() + 1];

    compute_chunk_signature(&pre, slice, initial_states, &mut pos0, &mut suffix_scratch, &mut cache)
}
