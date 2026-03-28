//! Fast vocab equivalence analysis based on DFA behavior signatures.
//!
//! Each token is classified by its match positions, suffix structure, and end
//! states across all tokenizer starts.

use super::super::compat::{compute_byte_classes, TokenizerView};
use ahash::{AHasher, RandomState};
use hashbrown::HashMap;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::compiler::stages::equivalence_analysis::disallowed_follows::normalize_disallowed_follows;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;

pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;

type EdgeList = SmallVec<[(usize, usize); 4]>;

struct DagNode {
    hash: u64,
    edges: EdgeList,
    end_state: usize,
}

const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;
const NONE: u32 = u32::MAX;
const STATE_NONE: usize = usize::MAX;
const VOCAB_MATCH_POSITIONS_GROUP_BYTES: usize = 256 * 1024;
const SELF_LOOP_ACTIVE_LEN_LIMIT: usize = 512;

/// Flat DFA with byte-class-compressed transposed transition tables.
///
/// Byte equivalence classes group bytes that produce identical transitions across
/// all DFA states. The transposed layout `trans_by_class[class * num_states + state]`
/// gives optimal cache locality: for a given byte, all state lookups hit a single
/// contiguous array chunk that fits in L1 cache.
struct Dfa {
    start_state: usize,
    num_states: usize,
    /// Byte-to-class mapping (byte equivalence classes).
    byte_to_class: [u8; 256],
    /// Transposed transition table: `trans_by_class[class * num_states + state]`.
    /// For a given byte class, all state transitions are contiguous in memory.
    trans_by_class: Vec<u32>,
    finalizers: Vec<SmallVec<[usize; 4]>>,
    is_dead_end: Vec<bool>,
    num_groups: usize,
    possible_future_groups: Vec<SmallVec<[usize; 4]>>,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
    /// Per-state bitset: which bytes cause a self-loop (transition back to same state).
    self_loop_bytes: Vec<U8Set>,
    disallowed_follows: Vec<BitSet>,
}

impl Dfa {
    /// Get completion hash for a state (or none_completion_hash for STATE_NONE).
    #[inline]
    fn completion(&self, state: usize) -> u64 {
        if state < self.completion_hash.len() {
            self.completion_hash[state]
        } else {
            self.none_completion_hash
        }
    }

    #[inline]
    fn completion_with_disallowed(&self, state: usize, disallowed: Option<&BitSet>) -> u64 {
        let Some(disallowed) = disallowed.filter(|bits| !bits.is_zero()) else {
            return self.completion(state);
        };
        if state >= self.possible_future_groups.len() {
            return self.none_completion_hash;
        }
        let mut h = new_hasher();
        h.write_u8(2);
        h.write_u64(hash_filtered_group_list(&self.possible_future_groups[state], disallowed));
        h.finish()
    }

    /// Look up transition: given a DFA state and a byte, return the next state (u32).
    #[inline]
    fn transition(&self, state: usize, byte: u8) -> u32 {
        let class = self.byte_to_class[byte as usize] as usize;
        unsafe { *self.trans_by_class.get_unchecked(class * self.num_states + state) }
    }

    #[inline]
    fn disallowed_for(&self, gid: usize) -> &BitSet {
        &self.disallowed_follows[gid]
    }
}

/// Combined scratch space for batch DFA execution and suffix DAG construction.
struct Scratch {
    // Batch execution across initial states
    current_states: Vec<usize>,
    active_indices: Vec<usize>,
    match_positions: Vec<u32>,
    /// Per-state multi-word dirty bitmask.  Layout: `[dirty_words * num_states]`
    /// where state `i`'s dirty mask occupies indices `[i*dirty_words .. (i+1)*dirty_words]`.
    dirty_group_masks: Vec<u64>,
    /// Number of u64 words per state in `dirty_group_masks` (= ceil(num_groups/64)).
    dirty_words: usize,
    targets: Vec<usize>,
    target_gids: HashMap<usize, SmallVec<[usize; 16]>>,
    single_target_pos: usize,
    single_target_gids: SmallVec<[usize; 16]>,
    // Suffix DAG
    dag: HashMap<usize, DagNode>,
    dag_queue: Vec<usize>,
    dag_disallowed: HashMap<usize, BitSet>,
    single_target_hash_pos: usize,
    single_target_hash: u64,
    suffix_match_positions: Vec<u32>,
    suffix_dirty_groups: SmallVec<[usize; 16]>,
}

static HASH_RANDOM_STATE: Lazy<RandomState> =
    Lazy::new(|| RandomState::with_seeds(HASH_SEED1, HASH_SEED2, HASH_SEED3, HASH_SEED4));
static VOCAB_UNGROUPED_BATCH: Lazy<bool> =
    Lazy::new(|| env_flag_enabled("GLRMASK_VOCAB_UNGROUPED_BATCH"));
static VOCAB_SIGNATURE_PROFILE_TOTALS: Lazy<VocabSignatureProfileTotals> =
    Lazy::new(VocabSignatureProfileTotals::new);

struct VocabSignatureProfileTotals {
    tokens: AtomicU64,
    zero_target_tokens: AtomicU64,
    single_target_tokens: AtomicU64,
    multi_target_tokens: AtomicU64,
    tokens_with_targets: AtomicU64,
    target_positions: AtomicU64,
    run_batch_ns: AtomicU64,
    run_batch_inner_ns: AtomicU64,
    collect_targets_ns: AtomicU64,
    try_single_target_ns: AtomicU64,
    try_single_target_hits: AtomicU64,
    try_single_target_fallbacks: AtomicU64,
    hash_suffix_calls: AtomicU64,
    hash_suffix_ns: AtomicU64,
    hash_suffix_setup_ns: AtomicU64,
    hash_suffix_bfs_ns: AtomicU64,
    hash_suffix_run_suffix_ns: AtomicU64,
    hash_suffix_propagate_ns: AtomicU64,
    hash_suffix_hash_ns: AtomicU64,
    hash_suffix_nodes: AtomicU64,
    finish_ns: AtomicU64,
}

impl VocabSignatureProfileTotals {
    const fn new() -> Self {
        Self {
            tokens: AtomicU64::new(0),
            zero_target_tokens: AtomicU64::new(0),
            single_target_tokens: AtomicU64::new(0),
            multi_target_tokens: AtomicU64::new(0),
            tokens_with_targets: AtomicU64::new(0),
            target_positions: AtomicU64::new(0),
            run_batch_ns: AtomicU64::new(0),
            run_batch_inner_ns: AtomicU64::new(0),
            collect_targets_ns: AtomicU64::new(0),
            try_single_target_ns: AtomicU64::new(0),
            try_single_target_hits: AtomicU64::new(0),
            try_single_target_fallbacks: AtomicU64::new(0),
            hash_suffix_calls: AtomicU64::new(0),
            hash_suffix_ns: AtomicU64::new(0),
            hash_suffix_setup_ns: AtomicU64::new(0),
            hash_suffix_bfs_ns: AtomicU64::new(0),
            hash_suffix_run_suffix_ns: AtomicU64::new(0),
            hash_suffix_propagate_ns: AtomicU64::new(0),
            hash_suffix_hash_ns: AtomicU64::new(0),
            hash_suffix_nodes: AtomicU64::new(0),
            finish_ns: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        self.tokens.store(0, Ordering::Relaxed);
        self.zero_target_tokens.store(0, Ordering::Relaxed);
        self.single_target_tokens.store(0, Ordering::Relaxed);
        self.multi_target_tokens.store(0, Ordering::Relaxed);
        self.tokens_with_targets.store(0, Ordering::Relaxed);
        self.target_positions.store(0, Ordering::Relaxed);
        self.run_batch_ns.store(0, Ordering::Relaxed);
        self.run_batch_inner_ns.store(0, Ordering::Relaxed);
        self.collect_targets_ns.store(0, Ordering::Relaxed);
        self.try_single_target_ns.store(0, Ordering::Relaxed);
        self.try_single_target_hits.store(0, Ordering::Relaxed);
        self.try_single_target_fallbacks.store(0, Ordering::Relaxed);
        self.hash_suffix_calls.store(0, Ordering::Relaxed);
        self.hash_suffix_ns.store(0, Ordering::Relaxed);
        self.hash_suffix_setup_ns.store(0, Ordering::Relaxed);
        self.hash_suffix_bfs_ns.store(0, Ordering::Relaxed);
        self.hash_suffix_run_suffix_ns.store(0, Ordering::Relaxed);
        self.hash_suffix_propagate_ns.store(0, Ordering::Relaxed);
        self.hash_suffix_hash_ns.store(0, Ordering::Relaxed);
        self.hash_suffix_nodes.store(0, Ordering::Relaxed);
        self.finish_ns.store(0, Ordering::Relaxed);
    }
}

#[inline]
fn new_hasher() -> AHasher {
    HASH_RANDOM_STATE.build_hasher()
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let trimmed = value.trim();
            !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn compile_profile_enabled() -> bool {
    env_flag_enabled("GLRMASK_PROFILE_COMPILE") || env_flag_enabled("GLRMASK_PROFILE_COMPILE_SUMMARY")
}

fn vocab_reachability_profile_enabled() -> bool {
    env_flag_enabled("GLRMASK_PROFILE_VOCAB_REACHABILITY")
}

fn vocab_batch_size_override() -> Option<usize> {
    std::env::var("GLRMASK_VOCAB_EQUIV_BATCH_SIZE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
}

fn vocab_state_group_size(num_states: usize, num_groups: usize) -> usize {
    if *VOCAB_UNGROUPED_BATCH || num_states <= 1 || num_groups == 0 {
        return num_states;
    }

    let bytes_per_state = num_groups.saturating_mul(std::mem::size_of::<u32>());
    let group_size = (VOCAB_MATCH_POSITIONS_GROUP_BYTES / bytes_per_state).max(1);
    group_size.min(num_states)
}

fn diversity_state_order_enabled() -> bool {
    !env_flag_enabled("GLRMASK_DISABLE_DIVERSITY_STATE_ORDER")
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn elapsed_ns(started_at: Instant) -> u64 {
    started_at.elapsed().as_nanos() as u64
}

fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

fn reachable_state_count(tokenizer: &TokenizerView, initial_states: &[usize]) -> usize {
    let dfa = tokenizer.dfa();
    if initial_states.is_empty() {
        return 0;
    }

    let mut seen = vec![false; dfa.states.len()];
    let mut queue = VecDeque::new();

    for &state in initial_states {
        if state < seen.len() && !seen[state] {
            seen[state] = true;
            queue.push_back(state);
        }
    }

    let mut count = 0usize;
    while let Some(state) = queue.pop_front() {
        count += 1;
        for &target in &dfa.states[state].transitions {
            if target == NONE {
                continue;
            }
            let target = target as usize;
            if !seen[target] {
                seen[target] = true;
                queue.push_back(target);
            }
        }
    }

    count
}

fn states_by_transition_diversity(dfa: &Dfa, states: &[usize]) -> Vec<usize> {
    let num_classes = dfa
        .byte_to_class
        .iter()
        .copied()
        .max()
        .map_or(0usize, |max_class| max_class as usize + 1);

    let mut ranked: Vec<(usize, usize)> = states
        .iter()
        .copied()
        .map(|state_id| {
            let mut targets = BTreeSet::new();
            for class in 0..num_classes {
                targets.insert(dfa.trans_by_class[class * dfa.num_states + state_id]);
            }
            (state_id, targets.len())
        })
        .collect();

    ranked.sort_unstable_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.into_iter().map(|(state_id, _)| state_id).collect()
}

#[inline]
fn hash_group_list(iter: impl ExactSizeIterator<Item = usize>) -> u64 {
    let mut h = new_hasher();
    h.write_u8(1);
    h.write_u64(iter.len() as u64);
    for v in iter {
        h.write_u64(v as u64);
    }
    h.finish()
}

#[inline]
fn hash_filtered_group_list(groups: &[usize], disallowed: &BitSet) -> u64 {
    let mut h = new_hasher();
    h.write_u8(1);
    let mut count = 0usize;
    for &gid in groups {
        if !disallowed.contains(gid) {
            count += 1;
        }
    }
    h.write_u64(count as u64);
    for &gid in groups {
        if !disallowed.contains(gid) {
            h.write_u64(gid as u64);
        }
    }
    h.finish()
}

fn build_dfa(
    tokenizer: &TokenizerView,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    byte_to_class_override: Option<&[u8; 256]>,
) -> Dfa {
    let profile_compile = compile_profile_enabled();
    let build_started_at = Instant::now();
    let dfa = tokenizer.dfa();
    assert!(dfa.states.len() <= u32::MAX as usize, "DFA too large");

    // Compute num_groups from all group IDs referenced in the DFA
    let num_groups = dfa
        .states
        .iter()
        .flat_map(|s| {
            s.finalizers
                .iter().copied()
                .chain(s.possible_future_group_ids.iter().copied())
        })
        .max()
        .map_or(0, |m| m + 1);

    let state_scan_started_at = Instant::now();
    let mut finalizers = Vec::with_capacity(dfa.states.len());
    let mut is_dead_end = Vec::with_capacity(dfa.states.len());
    let mut possible_future_groups = Vec::with_capacity(dfa.states.len());
    let mut completion_hash = Vec::with_capacity(dfa.states.len());
    let mut self_loop_bytes = Vec::with_capacity(dfa.states.len());

    for (state_idx, state) in dfa.states.iter().enumerate() {
        finalizers.push(state.finalizers.iter().copied().collect());

        is_dead_end.push(state.possible_future_group_ids.is_empty());
        let future_groups: SmallVec<[usize; 4]> =
            state.possible_future_group_ids.iter().copied().collect();
        completion_hash.push(hash_group_list(future_groups.iter().copied()));
        possible_future_groups.push(future_groups);

        let mut bits = U8Set::empty();
        for (byte_idx, &target) in state.transitions.iter().enumerate() {
            if target == state_idx as u32 {
                bits.insert(byte_idx as u8);
            }
        }
        self_loop_bytes.push(bits);
    }
    let state_scan_ms = elapsed_ms(state_scan_started_at);

    let none_completion_hash = {
        let mut h = new_hasher();
        h.write_u8(0);
        h.finish()
    };

    // Compute byte equivalence classes: group bytes with identical transitions across all states.
    let byte_classes_started_at = Instant::now();
    let num_dfa_states = dfa.states.len();
    let byte_to_class = byte_to_class_override.copied().unwrap_or_else(|| compute_byte_classes(dfa));
    let num_classes = byte_to_class
        .iter()
        .copied()
        .max()
        .map_or(0usize, |max_class| max_class as usize + 1);
    let mut class_repr = vec![0u8; num_classes];
    let mut class_seen = vec![false; num_classes];
    for b in 0..=255u8 {
        let class = byte_to_class[b as usize] as usize;
        if !class_seen[class] {
            class_seen[class] = true;
            class_repr[class] = b;
        }
    }
    let byte_classes_ms = elapsed_ms(byte_classes_started_at);

    // Build transposed transition table: trans_by_class[class * num_states + state]
    let transpose_started_at = Instant::now();
    let mut trans_by_class = vec![NONE; num_classes * num_dfa_states];
    for c in 0..num_classes {
        let repr = class_repr[c] as usize;
        let base = c * num_dfa_states;
        for s in 0..num_dfa_states {
            trans_by_class[base + s] = dfa.states[s].transitions[repr];
        }
    }
    let transpose_ms = elapsed_ms(transpose_started_at);

    if profile_compile {
        eprintln!(
            "[glrmask/profile][vocab_build_dfa] dfa_states={} num_groups={} byte_classes={} state_scan_ms={:.3} byte_classes_ms={:.3} transpose_ms={:.3} total_ms={:.3}",
            num_dfa_states,
            num_groups,
            num_classes,
            state_scan_ms,
            byte_classes_ms,
            transpose_ms,
            elapsed_ms(build_started_at),
        );
    }

    Dfa {
        start_state: dfa.start_state,
        num_states: num_dfa_states,
        byte_to_class,
        trans_by_class,
        finalizers,
        is_dead_end,
        num_groups,
        possible_future_groups,
        completion_hash,
        none_completion_hash,
        self_loop_bytes,
        disallowed_follows: normalize_disallowed_follows(num_groups, disallowed_follows),
    }
}

fn intersect_node_disallowed(
    scratch: &mut Scratch,
    pos: usize,
    incoming: &BitSet,
) {
    if let Some(existing) = scratch.dag_disallowed.get_mut(&pos) {
        *existing = existing.intersection(incoming);
    } else {
        scratch.dag_disallowed.insert(pos, incoming.clone());
    }
}

fn node_disallows_gid(scratch: &Scratch, pos: usize, gid: usize) -> bool {
    scratch
        .dag_disallowed
        .get(&pos)
        .map(|bits| bits.contains(gid))
        .unwrap_or(false)
}

impl Scratch {
    fn new(num_states: usize, num_groups: usize) -> Self {
        let dirty_words = (num_groups + 63) / 64;
        Scratch {
            current_states: vec![0; num_states],
            active_indices: Vec::new(),
            match_positions: vec![NONE; num_states * num_groups],
            dirty_group_masks: vec![0; num_states * dirty_words.max(1)],
            dirty_words,
            targets: Vec::new(),
            target_gids: HashMap::new(),
            single_target_pos: usize::MAX,
            single_target_gids: SmallVec::new(),
            dag: HashMap::new(),
            dag_queue: Vec::new(),
            dag_disallowed: HashMap::new(),
            single_target_hash_pos: usize::MAX,
            single_target_hash: 0,
            suffix_match_positions: vec![NONE; num_groups],
            suffix_dirty_groups: SmallVec::new(),
        }
    }
}

#[inline]
fn mark_dirty_group(scratch: &mut Scratch, state_idx: usize, gid: usize) {
    let word_idx = gid / 64;
    let bit = gid % 64;
    let flat_idx = state_idx * scratch.dirty_words + word_idx;
    scratch.dirty_group_masks[flat_idx] |= 1u64 << bit;
}

fn ensure_target_gids_map(
    target_gids: &mut HashMap<usize, SmallVec<[usize; 16]>>,
    single_target_pos: usize,
    single_target_gids: &[usize],
) {
    if target_gids.is_empty() && single_target_pos != usize::MAX {
        let mut gids = SmallVec::new();
        gids.extend(single_target_gids.iter().copied());
        target_gids.insert(single_target_pos, gids);
    }
}

fn record_target_gid(
    targets: &mut Vec<usize>,
    target_gids: &mut HashMap<usize, SmallVec<[usize; 16]>>,
    single_target_pos: &mut usize,
    single_target_gids: &mut SmallVec<[usize; 16]>,
    pos: usize,
    gid: usize,
) {
    if targets.is_empty() {
        targets.push(pos);
        *single_target_pos = pos;
        single_target_gids.clear();
        single_target_gids.push(gid);
        return;
    }

    if target_gids.is_empty() && targets.len() == 1 && *single_target_pos == pos {
        if !single_target_gids.contains(&gid) {
            single_target_gids.push(gid);
        }
        return;
    }

    ensure_target_gids_map(target_gids, *single_target_pos, single_target_gids.as_slice());

    let gids = target_gids.entry(pos).or_default();
    if gids.is_empty() {
        targets.push(pos);
    }
    if !gids.contains(&gid) {
        gids.push(gid);
    }
}

fn run_batch_inner(
    dfa: &Dfa,
    scratch: &mut Scratch,
    slice: &[u8],
    state_offset: usize,
    num_states: usize,
) {
    let num_groups = dfa.num_groups;
    let len = slice.len();
    let dirty_words = scratch.dirty_words;

    scratch.active_indices.clear();
    {
        let mask_start = state_offset * dirty_words;
        let mask_end = (state_offset + num_states) * dirty_words;
        for v in scratch.dirty_group_masks[mask_start..mask_end].iter_mut() {
            *v = 0;
        }
    }

    let has_bytes = !slice.is_empty();
    let first_byte = if has_bytes { slice[0] } else { 0 };

    // Initialize active states. Position-0 finalizer matches are NOT recorded
    // here because collect_targets filters them out (requires pv > 0) and
    // finish_token_signature skips them too. Omitting position-0 recording
    // avoids creating dirty-tracking entries for the ~97% of tokens that never
    // match at any position > 0, eliminating unnecessary bitmask
    // overhead in collect_targets and finish_token_signature.
    for i in state_offset..state_offset + num_states {
        let state = scratch.current_states[i];
        if dfa.is_dead_end[state] {
            scratch.current_states[i] = STATE_NONE;
            continue;
        }
        if has_bytes && dfa.transition(state, first_byte) == NONE {
            scratch.current_states[i] = STATE_NONE;
            continue;
        }
        scratch.active_indices.push(i);
    }

    // Walk each byte (hot path)
    if has_bytes && !scratch.active_indices.is_empty() {
        let mut active_len = scratch.active_indices.len();
        for (pos, &byte) in slice.iter().enumerate() {
            let position = (pos + 1) as u32;
            let mut next_len = 0usize;
            let class = dfa.byte_to_class[byte as usize] as usize;
            let class_base = class * dfa.num_states;
            for idx in 0..active_len {
                let i = scratch.active_indices[idx];
                let base = i * num_groups;
                let next_state = unsafe { *dfa.trans_by_class.get_unchecked(class_base + scratch.current_states[i]) };
                if next_state != NONE {
                    let ns = next_state as usize;
                    scratch.current_states[i] = ns;
                    for &gid in &dfa.finalizers[ns] {
                        if gid < num_groups {
                            let ix = base + gid;
                            if scratch.match_positions[ix] == NONE {
                                mark_dirty_group(scratch, i, gid);
                            }
                            scratch.match_positions[ix] = position;
                        }
                    }
                    if dfa.is_dead_end[ns] {
                        scratch.current_states[i] = STATE_NONE;
                    }
                } else {
                    scratch.current_states[i] = STATE_NONE;
                }
                if scratch.current_states[i] != STATE_NONE {
                    scratch.active_indices[next_len] = i;
                    next_len += 1;
                }
            }
            active_len = next_len;
            if active_len == 0 {
                break;
            }

            // Self-loop early exit: if all active states self-loop on every remaining byte,
            // greedy match positions advance to token_length and we can stop.
            if pos + 1 < len && active_len <= SELF_LOOP_ACTIVE_LEN_LIMIT {
                // Intersect self_loop_bytes for all active states
                let mut sl = U8Set::all();
                for idx in 0..active_len {
                    let i = scratch.active_indices[idx];
                    let s = scratch.current_states[i];
                    sl &= dfa.self_loop_bytes[s];
                }
                // Check if all remaining bytes are in the intersection
                let all_self_loop = slice[pos + 1..].iter().all(|&b| sl.contains(b));
                if all_self_loop {
                    let token_len = len as u32;
                    for idx in 0..active_len {
                        let i = scratch.active_indices[idx];
                        let base = i * num_groups;
                        let s = scratch.current_states[i];
                        for &gid in &dfa.finalizers[s] {
                            if gid < num_groups {
                                let ix = base + gid;
                                if scratch.match_positions[ix] == NONE {
                                    mark_dirty_group(scratch, i, gid);
                                }
                                scratch.match_positions[ix] = token_len;
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
}

fn collect_targets(
    scratch: &mut Scratch,
    num_groups: usize,
    len: usize,
    state_offset: usize,
    num_states: usize,
) {
    if num_groups == 0 {
        return;
    }

    let dirty_words = scratch.dirty_words;

    let Scratch {
        dirty_group_masks,
        match_positions,
        targets,
        target_gids,
        single_target_pos,
        single_target_gids,
        ..
    } = scratch;

    for i in state_offset..state_offset + num_states {
        let base = i * num_groups;
        let mask_base = i * dirty_words;
        for w in 0..dirty_words {
            let mut dirty_mask = dirty_group_masks[mask_base + w];
            while dirty_mask != 0 {
                let bit = dirty_mask.trailing_zeros() as usize;
                dirty_mask &= dirty_mask - 1;
                let gid = w * 64 + bit;
                if gid >= num_groups {
                    break;
                }
                let pv = match_positions[base + gid];
                if pv != NONE && pv > 0 && (pv as usize) <= len {
                    record_target_gid(
                        targets,
                        target_gids,
                        single_target_pos,
                        single_target_gids,
                        pv as usize,
                        gid,
                    );
                }
            }
        }
    }
}

/// Run DFA from all initial states on a token, recording end states and match positions.
/// Uses dirty bitmask tracking to avoid O(num_states * num_groups) memset.
/// INVARIANT: match_positions entries are NONE except for dirty entries from a previous
/// call that must have been cleaned up by the caller (token_signature does this).
fn run_batch(
    dfa: &Dfa,
    scratch: &mut Scratch,
    slice: &[u8],
    initial_states: &[usize],
    state_group_size: usize,
    profile_signature_detail: bool,
) {
    let num_states = initial_states.len();
    let num_groups = dfa.num_groups;
    let len = slice.len();
    let run_batch_started_at = profile_signature_detail.then(Instant::now);
    let mut run_batch_inner_ns = 0u64;
    let mut collect_targets_ns = 0u64;

    if num_states == 0 {
        scratch.targets.clear();
        return;
    }

    scratch.current_states[..num_states].clone_from_slice(initial_states);
    scratch.targets.clear();
    scratch.target_gids.clear();
    scratch.single_target_pos = usize::MAX;
    scratch.single_target_gids.clear();

    if state_group_size >= num_states {
        let run_batch_inner_started_at = profile_signature_detail.then(Instant::now);
        run_batch_inner(dfa, scratch, slice, 0, num_states);
        run_batch_inner_ns += run_batch_inner_started_at.map_or(0, elapsed_ns);

        let collect_targets_started_at = profile_signature_detail.then(Instant::now);
        collect_targets(scratch, num_groups, len, 0, num_states);
        collect_targets_ns += collect_targets_started_at.map_or(0, elapsed_ns);
    } else {
        for state_offset in (0..num_states).step_by(state_group_size) {
            let group_len = (state_offset + state_group_size).min(num_states) - state_offset;

            let run_batch_inner_started_at = profile_signature_detail.then(Instant::now);
            run_batch_inner(dfa, scratch, slice, state_offset, group_len);
            run_batch_inner_ns += run_batch_inner_started_at.map_or(0, elapsed_ns);

            let collect_targets_started_at = profile_signature_detail.then(Instant::now);
            collect_targets(scratch, num_groups, len, state_offset, group_len);
            collect_targets_ns += collect_targets_started_at.map_or(0, elapsed_ns);
        }
    }

    if profile_signature_detail {
        let totals = &*VOCAB_SIGNATURE_PROFILE_TOTALS;
        totals
            .run_batch_ns
            .fetch_add(run_batch_started_at.map_or(0, elapsed_ns), Ordering::Relaxed);
        totals
            .run_batch_inner_ns
            .fetch_add(run_batch_inner_ns, Ordering::Relaxed);
        totals
            .collect_targets_ns
            .fetch_add(collect_targets_ns, Ordering::Relaxed);
    }
}

fn hash_suffixes(
    dfa: &Dfa,
    slice: &[u8],
    scratch: &mut Scratch,
    profile_signature_detail: bool,
) -> usize {
    let total_started_at = profile_signature_detail.then(Instant::now);
    let setup_started_at = profile_signature_detail.then(Instant::now);
    let len = slice.len();
    scratch.dag.clear();
    scratch.dag_queue.clear();
    scratch.dag_disallowed.clear();

    for (&pos, gids) in &scratch.target_gids {
        if let Some((&first_gid, rest)) = gids.split_first() {
            let mut combined = dfa.disallowed_for(first_gid).clone();
            for &gid in rest {
                combined = combined.intersection(dfa.disallowed_for(gid));
            }
            scratch.dag_disallowed.insert(pos, combined);
        }
    }
    let setup_ns = setup_started_at.map_or(0, elapsed_ns);

    // BFS from target positions: run suffix DFA at each, discover new positions from edges
    for &pos in &scratch.targets {
        if pos < len && !scratch.dag.contains_key(&pos) {
            scratch.dag_queue.push(pos);
            scratch.dag.insert(
                pos,
                DagNode {
                    hash: 0,
                    edges: EdgeList::new(),
                    end_state: STATE_NONE,
                },
            );
        }
    }

    let mut cursor = 0;
    let bfs_started_at = profile_signature_detail.then(Instant::now);
    let mut run_suffix_ns = 0u64;
    while cursor < scratch.dag_queue.len() {
        let pos = scratch.dag_queue[cursor];
        cursor += 1;
        let run_suffix_started_at = profile_signature_detail.then(Instant::now);
        let (end_state, edges) = run_suffix(
            dfa,
            &slice[pos..],
            pos,
            &mut scratch.suffix_match_positions,
            &mut scratch.suffix_dirty_groups,
        );
        run_suffix_ns += run_suffix_started_at.map_or(0, elapsed_ns);
        for &(_, target) in &edges {
            if target < len && !scratch.dag.contains_key(&target) {
                scratch.dag_queue.push(target);
                scratch.dag.insert(
                    target,
                    DagNode {
                        hash: 0,
                        edges: EdgeList::new(),
                        end_state: STATE_NONE,
                    },
                );
            }
        }
        scratch.dag.insert(
            pos,
            DagNode {
                hash: 0,
                edges,
                end_state: end_state.unwrap_or(STATE_NONE),
            },
        );
    }
    let bfs_ns = bfs_started_at.map_or(0, elapsed_ns);

    let propagate_started_at = profile_signature_detail.then(Instant::now);
    scratch.dag_queue.sort_unstable();
    for idx in 0..scratch.dag_queue.len() {
        let pos = scratch.dag_queue[idx];
        let edges = scratch.dag[&pos].edges.clone();

        let first_hop_target = edges.iter().map(|&(_, t)| t).min();
        let first_hop_blocked = first_hop_target.is_some_and(|ft| {
            edges.iter()
                .filter(|&&(_, t)| t == ft)
                .all(|&(gid, _)| node_disallows_gid(scratch, pos, gid))
        });

        for &(gid, target) in &edges {
            if node_disallows_gid(scratch, pos, gid) {
                continue;
            }
            if first_hop_blocked && Some(target) != first_hop_target {
                continue;
            }
            if target < len {
                intersect_node_disallowed(scratch, target, dfa.disallowed_for(gid));
            }
        }
    }
    let propagate_ns = propagate_started_at.map_or(0, elapsed_ns);

    // Hash bottom-up: process deeper positions first
    let hash_started_at = profile_signature_detail.then(Instant::now);
    scratch.dag_queue.sort_unstable_by(|a, b| b.cmp(a));
    for idx in 0..scratch.dag_queue.len() {
        let pos = scratch.dag_queue[idx];
        let node = scratch.dag.get(&pos).unwrap();
        let edges = node.edges.clone();
        let end_state = node.end_state;
        let mut h = new_hasher();
        h.write_u64(dfa.completion_with_disallowed(end_state, scratch.dag_disallowed.get(&pos)));

        // Multi-segment edge fix: the earliest-target edges represent
        // "first hop" (single-segment) choices from this position.
        // Edges at later positions represent multi-segment paths that
        // necessarily pass through the first hop's segment. If ALL
        // first-hop groups are disallowed, later edges are unreachable.
        let first_hop_target = edges.iter().map(|&(_, t)| t).min();
        let first_hop_blocked = first_hop_target.is_some_and(|ft| {
            edges.iter()
                .filter(|&&(_, t)| t == ft)
                .all(|&(gid, _)| node_disallows_gid(scratch, pos, gid))
        });

        for &(gid, target) in &edges {
            if node_disallows_gid(scratch, pos, gid) {
                continue;
            }
            // Skip later-hop edges when ALL first-hop edges are disallowed
            if first_hop_blocked && Some(target) != first_hop_target {
                continue;
            }
            h.write_u64(gid as u64);
            h.write_u64(scratch.dag.get(&target).map_or(0, |node| node.hash));
        }
        scratch.dag.get_mut(&pos).unwrap().hash = h.finish();
    }
    let hash_ns = hash_started_at.map_or(0, elapsed_ns);

    if profile_signature_detail {
        let totals = &*VOCAB_SIGNATURE_PROFILE_TOTALS;
        totals.hash_suffix_calls.fetch_add(1, Ordering::Relaxed);
        totals
            .hash_suffix_ns
            .fetch_add(total_started_at.map_or(0, elapsed_ns), Ordering::Relaxed);
        totals
            .hash_suffix_setup_ns
            .fetch_add(setup_ns, Ordering::Relaxed);
        totals
            .hash_suffix_bfs_ns
            .fetch_add(bfs_ns, Ordering::Relaxed);
        totals
            .hash_suffix_run_suffix_ns
            .fetch_add(run_suffix_ns, Ordering::Relaxed);
        totals
            .hash_suffix_propagate_ns
            .fetch_add(propagate_ns, Ordering::Relaxed);
        totals
            .hash_suffix_hash_ns
            .fetch_add(hash_ns, Ordering::Relaxed);
        totals
            .hash_suffix_nodes
            .fetch_add(scratch.dag_queue.len() as u64, Ordering::Relaxed);
    }

    scratch.dag_queue.len()
}

/// Run DFA on a suffix from start_state, returning (end_state, edges to match positions).
fn run_suffix(
    dfa: &Dfa,
    slice: &[u8],
    base_pos: usize,
    match_positions: &mut [u32],
    dirty_groups: &mut SmallVec<[usize; 16]>,
) -> (Option<usize>, EdgeList) {
    let num_groups = dfa.num_groups;
    dirty_groups.clear();
    let mut current = dfa.start_state;
    let mut done = dfa.is_dead_end[current];

    for &gid in &dfa.finalizers[current] {
        if gid < num_groups && match_positions[gid] == NONE {
            dirty_groups.push(gid);
            match_positions[gid] = 0;
        }
    }

    for (idx, &byte) in slice.iter().enumerate() {
        if done {
            break;
        }
        let ns = dfa.transition(current, byte);
        if ns == NONE {
            done = true;
            break;
        }
        current = ns as usize;
        let position = (idx + 1) as u32;
        for &gid in &dfa.finalizers[current] {
            if gid < num_groups {
                if match_positions[gid] == NONE {
                    dirty_groups.push(gid);
                }
                match_positions[gid] = position;
            }
        }
        if dfa.is_dead_end[current] {
            done = true;
        }
    }

    let end_state = if done { None } else { Some(current) };
    let edges: EdgeList = dirty_groups
        .iter()
        .filter_map(|&gid| {
            let pv = match_positions[gid];
            (pv != NONE && pv != 0).then(|| (gid, base_pos + pv as usize))
        })
        .collect();
    for &gid in dirty_groups.iter() {
        match_positions[gid] = NONE;
    }
    (end_state, edges)
}

fn try_hash_single_target_suffix(
    dfa: &Dfa,
    slice: &[u8],
    scratch: &mut Scratch,
) -> Option<usize> {
    let pos = scratch.single_target_pos;
    if pos == usize::MAX {
        return None;
    }
    let len = slice.len();

    if pos >= len {
        scratch.single_target_hash_pos = pos;
        scratch.single_target_hash = 0;
        return Some(0);
    }

    let (&first_gid, rest) = scratch.single_target_gids.split_first()?;
    let mut root_disallowed = dfa.disallowed_for(first_gid).clone();
    for &gid in rest {
        root_disallowed = root_disallowed.intersection(dfa.disallowed_for(gid));
    }

    let (end_state, edges) = run_suffix(
        dfa,
        &slice[pos..],
        pos,
        &mut scratch.suffix_match_positions,
        &mut scratch.suffix_dirty_groups,
    );
    if edges.iter().any(|&(_, target)| target < len) {
        return None;
    }

    let first_hop_target = edges.iter().map(|&(_, target)| target).min();
    let first_hop_blocked = first_hop_target.is_some_and(|target| {
        edges
            .iter()
            .filter(|&&(_, edge_target)| edge_target == target)
            .all(|&(gid, _)| root_disallowed.contains(gid))
    });

    let mut h = new_hasher();
    h.write_u64(dfa.completion_with_disallowed(
        end_state.unwrap_or(STATE_NONE),
        Some(&root_disallowed),
    ));
    for &(gid, target) in &edges {
        if root_disallowed.contains(gid) {
            continue;
        }
        if first_hop_blocked && Some(target) != first_hop_target {
            continue;
        }
        h.write_u64(gid as u64);
        h.write_u64(0);
    }

    scratch.single_target_hash_pos = pos;
    scratch.single_target_hash = h.finish();
    Some(1)
}

/// Compute a token's full signature over a batch of initial states.
/// Also cleans up match_positions for dirty groups (maintaining the NONE invariant).
fn finish_token_signature(
    dfa: &Dfa,
    chunk_states: &[usize],
    scratch: &mut Scratch,
) -> u64 {
    let num_groups = dfa.num_groups;
    let dirty_words = scratch.dirty_words;
    let dag = &scratch.dag;
    let single_target_hash_pos = scratch.single_target_hash_pos;
    let single_target_hash = scratch.single_target_hash;
    let mut sig: u64 = HASH_SEED3;
    for i in 0..chunk_states.len() {
        let completion = dfa.completion(scratch.current_states[i]);
        let base = i * num_groups;
        let mask_base = i * dirty_words;

        let mut any_dirty = false;
        for w in 0..dirty_words {
            if scratch.dirty_group_masks[mask_base + w] != 0 {
                any_dirty = true;
                break;
            }
        }

        let state_sig = if any_dirty {
            let mut h = new_hasher();
            h.write_u64(completion);
            for w in 0..dirty_words {
                let mut dirty_mask = scratch.dirty_group_masks[mask_base + w];
                while dirty_mask != 0 {
                    let bit = dirty_mask.trailing_zeros() as usize;
                    dirty_mask &= dirty_mask - 1;
                    let gid = w * 64 + bit;
                    if gid >= num_groups {
                        break;
                    }
                    let pv = scratch.match_positions[base + gid];
                    if pv != NONE && pv > 0 {
                        h.write_u64(gid as u64);
                        let target = pv as usize;
                        let target_hash = if single_target_hash_pos == target {
                            single_target_hash
                        } else {
                            dag.get(&target).map_or(0, |node| node.hash)
                        };
                        h.write_u64(target_hash);
                    }
                    scratch.match_positions[base + gid] = NONE;
                }
                scratch.dirty_group_masks[mask_base + w] = 0;
            }
            h.finish()
        } else {
            completion
        };

        sig = sig.wrapping_mul(HASH_SEED1).wrapping_add(state_sig);
    }
    sig
}

fn token_signature(
    dfa: &Dfa,
    token: &[u8],
    chunk_states: &[usize],
    state_group_size: usize,
    scratch: &mut Scratch,
    profile_signature_detail: bool,
) -> u64 {
    if profile_signature_detail {
        VOCAB_SIGNATURE_PROFILE_TOTALS
            .tokens
            .fetch_add(1, Ordering::Relaxed);
    }
    scratch.single_target_hash_pos = usize::MAX;
    scratch.single_target_hash = 0;
    run_batch(
        dfa,
        scratch,
        token,
        chunk_states,
        state_group_size,
        profile_signature_detail,
    );
    let target_count = scratch.targets.len();
    if profile_signature_detail {
        let totals = &*VOCAB_SIGNATURE_PROFILE_TOTALS;
        totals
            .target_positions
            .fetch_add(target_count as u64, Ordering::Relaxed);
        match target_count {
            0 => {
                totals
                    .zero_target_tokens
                    .fetch_add(1, Ordering::Relaxed);
            }
            1 => {
                totals
                    .single_target_tokens
                    .fetch_add(1, Ordering::Relaxed);
                totals
                    .tokens_with_targets
                    .fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                totals
                    .multi_target_tokens
                    .fetch_add(1, Ordering::Relaxed);
                totals
                    .tokens_with_targets
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    if target_count == 1 {
        let try_single_target_started_at = profile_signature_detail.then(Instant::now);
        if let Some(dag_nodes) = try_hash_single_target_suffix(dfa, token, scratch) {
            let _ = dag_nodes;
            if profile_signature_detail {
                let totals = &*VOCAB_SIGNATURE_PROFILE_TOTALS;
                totals
                    .try_single_target_ns
                    .fetch_add(try_single_target_started_at.map_or(0, elapsed_ns), Ordering::Relaxed);
                totals
                    .try_single_target_hits
                    .fetch_add(1, Ordering::Relaxed);
            }
        } else {
            if profile_signature_detail {
                let totals = &*VOCAB_SIGNATURE_PROFILE_TOTALS;
                totals
                    .try_single_target_ns
                    .fetch_add(try_single_target_started_at.map_or(0, elapsed_ns), Ordering::Relaxed);
                totals
                    .try_single_target_fallbacks
                    .fetch_add(1, Ordering::Relaxed);
            }
            ensure_target_gids_map(
                &mut scratch.target_gids,
                scratch.single_target_pos,
                scratch.single_target_gids.as_slice(),
            );
            hash_suffixes(dfa, token, scratch, profile_signature_detail);
        }
    } else if target_count > 0 {
        hash_suffixes(dfa, token, scratch, profile_signature_detail);
    }

    let finish_started_at = profile_signature_detail.then(Instant::now);
    let sig = finish_token_signature(dfa, chunk_states, scratch);
    if profile_signature_detail {
        VOCAB_SIGNATURE_PROFILE_TOTALS
            .finish_ns
            .fetch_add(finish_started_at.map_or(0, elapsed_ns), Ordering::Relaxed);
    }
    sig
}

// ----- DFS Trie Walk for Prefix Sharing -----

const TRIE_CHUNK_SIZE: usize = 128;
const TRIE_WALK_MIN_TOKENS: usize = 256;

static TRIE_WALK_DISABLED: Lazy<bool> =
    Lazy::new(|| env_flag_enabled("GLRMASK_DISABLE_TRIE_WALK"));

struct DepthChangeLog {
    /// (state_idx, old_state_value)
    state_changes: Vec<(usize, usize)>,
    /// (match_positions_flat_idx, old_value)
    match_changes: Vec<(usize, u32)>,
    /// (state_idx, old_dirty_mask)
    dirty_changes: Vec<(usize, u64)>,
}

impl DepthChangeLog {
    fn new() -> Self {
        Self {
            state_changes: Vec::new(),
            match_changes: Vec::new(),
            dirty_changes: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.state_changes.clear();
        self.match_changes.clear();
        self.dirty_changes.clear();
    }
}

struct TrieWalkState {
    depth_logs: Vec<DepthChangeLog>,
}

impl TrieWalkState {
    fn new() -> Self {
        Self {
            depth_logs: Vec::new(),
        }
    }

    fn ensure_depth(&mut self, depth: usize) {
        while self.depth_logs.len() <= depth {
            self.depth_logs.push(DepthChangeLog::new());
        }
    }
}

/// Walk one byte forward at the given depth, recording all state changes
/// to the change log for later backtracking.
fn dfs_step(
    dfa: &Dfa,
    scratch: &mut Scratch,
    trie: &mut TrieWalkState,
    byte: u8,
    depth: usize,
    batch_len: usize,
) {
    trie.ensure_depth(depth);
    let log = &mut trie.depth_logs[depth];
    log.clear();

    let num_groups = dfa.num_groups;
    let dirty_words = scratch.dirty_words;
    let position = (depth + 1) as u32;
    let class = dfa.byte_to_class[byte as usize] as usize;
    let class_base = class * dfa.num_states;

    for i in 0..batch_len {
        let old_state = scratch.current_states[i];
        if old_state == STATE_NONE {
            continue;
        }

        let next_state_raw =
            unsafe { *dfa.trans_by_class.get_unchecked(class_base + old_state) };
        if next_state_raw == NONE {
            log.state_changes.push((i, old_state));
            scratch.current_states[i] = STATE_NONE;
            continue;
        }

        let ns = next_state_raw as usize;
        log.state_changes.push((i, old_state));
        scratch.current_states[i] = ns;

        let base = i * num_groups;
        for &gid in &dfa.finalizers[ns] {
            if gid < num_groups {
                let ix = base + gid;
                let old_mp = scratch.match_positions[ix];
                if old_mp == NONE {
                    let word_idx = gid / 64;
                    let bit = gid % 64;
                    let flat_idx = i * dirty_words + word_idx;
                    log.dirty_changes
                        .push((flat_idx, scratch.dirty_group_masks[flat_idx]));
                    scratch.dirty_group_masks[flat_idx] |= 1u64 << bit;
                }
                log.match_changes.push((ix, old_mp));
                scratch.match_positions[ix] = position;
            }
        }

        if dfa.is_dead_end[ns] {
            scratch.current_states[i] = STATE_NONE;
        }
    }
}

/// Undo all changes recorded at a single depth level.
fn dfs_undo_depth(scratch: &mut Scratch, log: &DepthChangeLog) {
    for &(ix, old_mp) in log.match_changes.iter().rev() {
        scratch.match_positions[ix] = old_mp;
    }
    for &(flat_idx, old_dirty) in log.dirty_changes.iter().rev() {
        scratch.dirty_group_masks[flat_idx] = old_dirty;
    }
    for &(i, old_state) in log.state_changes.iter().rev() {
        scratch.current_states[i] = old_state;
    }
}

/// Backtrack from current_depth to target_depth by undoing changes.
fn dfs_backtrack(
    scratch: &mut Scratch,
    trie: &TrieWalkState,
    current_depth: usize,
    target_depth: usize,
) {
    for depth in (target_depth..current_depth).rev() {
        dfs_undo_depth(scratch, &trie.depth_logs[depth]);
    }
}

/// Compute token signature without modifying scratch state (no cleanup).
/// Uses multi-word bitmask dirty tracking for any number of groups.
fn finish_token_signature_no_cleanup(
    dfa: &Dfa,
    num_initial_states: usize,
    scratch: &Scratch,
) -> u64 {
    let num_groups = dfa.num_groups;
    let dirty_words = scratch.dirty_words;
    let dag = &scratch.dag;
    let single_target_hash_pos = scratch.single_target_hash_pos;
    let single_target_hash = scratch.single_target_hash;
    let mut sig: u64 = HASH_SEED3;
    for i in 0..num_initial_states {
        let completion = dfa.completion(scratch.current_states[i]);
        let base = i * num_groups;
        let mask_base = i * dirty_words;

        let mut any_dirty = false;
        for w in 0..dirty_words {
            if scratch.dirty_group_masks[mask_base + w] != 0 {
                any_dirty = true;
                break;
            }
        }

        let state_sig = if any_dirty {
            let mut h = new_hasher();
            h.write_u64(completion);
            for w in 0..dirty_words {
                let mut dm = scratch.dirty_group_masks[mask_base + w];
                while dm != 0 {
                    let bit = dm.trailing_zeros() as usize;
                    dm &= dm - 1;
                    let gid = w * 64 + bit;
                    if gid >= num_groups {
                        break;
                    }
                    let pv = scratch.match_positions[base + gid];
                    if pv != NONE && pv > 0 {
                        h.write_u64(gid as u64);
                        let target = pv as usize;
                        let target_hash = if single_target_hash_pos == target {
                            single_target_hash
                        } else {
                            dag.get(&target).map_or(0, |node| node.hash)
                        };
                        h.write_u64(target_hash);
                    }
                }
            }
            h.finish()
        } else {
            completion
        };
        sig = sig.wrapping_mul(HASH_SEED1).wrapping_add(state_sig);
    }
    sig
}

/// Process a sorted chunk of tokens using DFS trie walk with prefix sharing.
/// Tokens must be sorted by byte content. Returns (token_idx, signature) pairs.
fn trie_walk_chunk_signatures<S: AsRef<[u8]> + Sync>(
    dfa: &Dfa,
    strings: &[S],
    chunk: &[usize],
    batch: &[usize],
    state_group_size: usize,
    scratch: &mut Scratch,
    trie: &mut TrieWalkState,
    profile: bool,
) -> Vec<(usize, u64)> {
    let batch_len = batch.len();
    let num_groups = dfa.num_groups;
    let mut results = Vec::with_capacity(chunk.len());

    // Initialize: copy initial states and mark dead-ends
    scratch.current_states[..batch_len].clone_from_slice(batch);
    {
        let dirty_words = scratch.dirty_words;
        let mask_end = batch_len * dirty_words;
        for v in scratch.dirty_group_masks[..mask_end].iter_mut() {
            *v = 0;
        }
    }
    for i in 0..batch_len {
        if dfa.is_dead_end[scratch.current_states[i]] {
            scratch.current_states[i] = STATE_NONE;
        }
    }

    let mut current_depth: usize = 0;
    let mut prev_token: &[u8] = &[];

    for &token_idx in chunk {
        let token = strings[token_idx].as_ref();
        let token_len = token.len();

        // Compute LCP with previous token
        let lcp = prev_token
            .iter()
            .zip(token.iter())
            .take_while(|(a, b)| a == b)
            .count();

        // Backtrack to LCP depth
        if current_depth > lcp {
            dfs_backtrack(scratch, trie, current_depth, lcp);
        }

        // Walk forward from LCP to token end
        for d in lcp..token_len {
            dfs_step(dfa, scratch, trie, token[d], d, batch_len);
        }
        current_depth = token_len;

        // Collect targets across all state groups
        scratch.targets.clear();
        scratch.target_gids.clear();
        scratch.single_target_pos = usize::MAX;
        scratch.single_target_gids.clear();
        scratch.single_target_hash_pos = usize::MAX;
        scratch.single_target_hash = 0;

        if state_group_size >= batch_len {
            collect_targets(scratch, num_groups, token_len, 0, batch_len);
        } else {
            for state_offset in (0..batch_len).step_by(state_group_size) {
                let group_len =
                    (state_offset + state_group_size).min(batch_len) - state_offset;
                collect_targets(scratch, num_groups, token_len, state_offset, group_len);
            }
        }

        // Hash suffixes
        let target_count = scratch.targets.len();
        if target_count == 1 {
            if try_hash_single_target_suffix(dfa, token, scratch).is_none() {
                ensure_target_gids_map(
                    &mut scratch.target_gids,
                    scratch.single_target_pos,
                    scratch.single_target_gids.as_slice(),
                );
                hash_suffixes(dfa, token, scratch, profile);
            }
        } else if target_count > 0 {
            hash_suffixes(dfa, token, scratch, profile);
        }

        // Compute signature without cleanup (DFS backtrack handles state restoration)
        let sig = finish_token_signature_no_cleanup(dfa, batch_len, scratch);

        if profile {
            let totals = &*VOCAB_SIGNATURE_PROFILE_TOTALS;
            totals.tokens.fetch_add(1, Ordering::Relaxed);
            totals
                .target_positions
                .fetch_add(target_count as u64, Ordering::Relaxed);
            match target_count {
                0 => {
                    totals.zero_target_tokens.fetch_add(1, Ordering::Relaxed);
                }
                1 => {
                    totals
                        .single_target_tokens
                        .fetch_add(1, Ordering::Relaxed);
                    totals
                        .tokens_with_targets
                        .fetch_add(1, Ordering::Relaxed);
                }
                _ => {
                    totals
                        .multi_target_tokens
                        .fetch_add(1, Ordering::Relaxed);
                    totals
                        .tokens_with_targets
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        results.push((token_idx, sig));
        prev_token = token;
    }

    // Final backtrack to restore scratch to clean state
    if current_depth > 0 {
        dfs_backtrack(scratch, trie, current_depth, 0);
    }

    results
}

pub fn find_vocab_equivalence_classes_with_follow<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> VocabEquivalenceResult {
    find_vocab_equivalence_classes_with_follow_and_byte_classes(
        tokenizer,
        strings,
        initial_states,
        disallowed_follows,
        None,
    )
}

pub fn find_vocab_equivalence_classes_with_follow_and_byte_classes<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    byte_to_class: Option<&[u8; 256]>,
) -> VocabEquivalenceResult {
    let profile_compile = compile_profile_enabled();
    let reachable_states = vocab_reachability_profile_enabled()
        .then(|| reachable_state_count(tokenizer, initial_states));
    let build_dfa_started_at = Instant::now();
    let dfa = build_dfa(tokenizer, disallowed_follows, byte_to_class);
    let build_dfa_ms = elapsed_ms(build_dfa_started_at);
    let order_states_started_at = Instant::now();
    let ordered_states = if diversity_state_order_enabled() {
        states_by_transition_diversity(&dfa, initial_states)
    } else {
        initial_states.to_vec()
    };
    let order_states_ms = elapsed_ms(order_states_started_at);
    let num_tokens = strings.len();
    let num_initial_states = ordered_states.len();

    if num_initial_states == 0 || num_tokens == 0 {
        return BTreeSet::from_iter(vec![(0..num_tokens).collect()]);
    }

    let num_groups = dfa.num_groups;
    // Use all states in a single batch when feasible.  A single batch avoids
    // repeated token sorting, trie walk reinitialisation, rayon sync points
    // between batches, and redundant finish_token_signature iterations.
    // Memory per rayon thread is bounded by the match_positions working set
    // (batch_size * num_groups * 4 bytes).  With the 16 MB target, the worst
    // case across an 8-thread pool is ~128 MB — fine for a one-shot compile.
    let default_batch_size = {
        let target_bytes = 16_000_000usize;
        let per_state_bytes = num_groups.max(1) * std::mem::size_of::<u32>();
        (target_bytes / per_state_bytes).clamp(500, 50_000)
    };
    let batch_size = vocab_batch_size_override()
        .unwrap_or_else(|| num_initial_states.min(default_batch_size));
    let mut active_indices: Vec<usize> = (0..num_tokens).collect();
    let mut partition = vec![0usize; num_tokens];
    let mut next_class_id = 1usize;
    let mut batch_total_ms = 0.0;
    let mut signature_total_ms = 0.0;
    let mut refine_total_ms = 0.0;

    if profile_compile {
        VOCAB_SIGNATURE_PROFILE_TOTALS.reset();
    }

    for (batch_index, batch_start) in (0..num_initial_states).step_by(batch_size).enumerate() {
        if active_indices.is_empty() {
            break;
        }

        let batch_started_at = Instant::now();
        let active_before = active_indices.len();
        let batch_end = (batch_start + batch_size).min(num_initial_states);
        let batch = &ordered_states[batch_start..batch_end];
        let state_group_size = vocab_state_group_size(batch.len(), num_groups);
        let signature_started_at = Instant::now();
        let use_trie_walk = active_indices.len() >= TRIE_WALK_MIN_TOKENS
            && !*TRIE_WALK_DISABLED;
        let active_sigs: Vec<(usize, u64)> = if use_trie_walk {
            let mut sorted_indices = active_indices.clone();
            sorted_indices.sort_unstable_by(|&a, &b| {
                strings[a].as_ref().cmp(strings[b].as_ref())
            });
            let chunk_results: Vec<Vec<(usize, u64)>> = sorted_indices
                .par_chunks(TRIE_CHUNK_SIZE)
                .map_init(
                    || {
                        (
                            Scratch::new(batch.len(), num_groups),
                            TrieWalkState::new(),
                        )
                    },
                    |(scratch, trie_state), chunk| {
                        trie_walk_chunk_signatures(
                            &dfa,
                            strings,
                            chunk,
                            batch,
                            state_group_size,
                            scratch,
                            trie_state,
                            profile_compile,
                        )
                    },
                )
                .collect();
            chunk_results.into_iter().flatten().collect()
        } else {
            active_indices
                .par_iter()
                .map_init(
                    || Scratch::new(batch.len(), num_groups),
                    |scratch, &token_idx| {
                        let token = strings[token_idx].as_ref();
                        (
                            token_idx,
                            token_signature(
                                &dfa,
                                token,
                                batch,
                                state_group_size,
                                scratch,
                                profile_compile,
                            ),
                        )
                    },
                )
                .collect()
        };
        let signature_ms = elapsed_ms(signature_started_at);

        let refine_started_at = Instant::now();
        let mut refinement: HashMap<(usize, u64), Vec<usize>> =
            HashMap::with_capacity(active_sigs.len() / 2);
        for (ti, sig) in active_sigs {
            refinement
                .entry((partition[ti], sig))
                .or_default()
                .push(ti);
        }

        let mut new_active = Vec::with_capacity(active_indices.len());
        let mut seen_classes = vec![false; next_class_id.max(1)];
        for ((old_class, _), tokens) in refinement {
            let class_id = if !seen_classes[old_class] {
                seen_classes[old_class] = true;
                old_class
            } else {
                let id = next_class_id;
                next_class_id += 1;
                id
            };
            for &ti in &tokens {
                partition[ti] = class_id;
            }
            if tokens.len() > 1 {
                new_active.extend(tokens);
            }
        }
        active_indices = new_active;
        let refine_ms = elapsed_ms(refine_started_at);
        let batch_ms = elapsed_ms(batch_started_at);
        batch_total_ms += batch_ms;
        signature_total_ms += signature_ms;
        refine_total_ms += refine_ms;

        if profile_compile {
            let state_group_count = batch.len().div_ceil(state_group_size.max(1));
            eprintln!(
                "[glrmask/profile][vocab] batch={} states={} state_group_size={} state_groups={} active_before={} active_after={} classes={} signature_ms={:.3} refine_ms={:.3} ms={:.3}",
                batch_index,
                batch.len(),
                state_group_size,
                state_group_count,
                active_before,
                active_indices.len(),
                next_class_id,
                signature_ms,
                refine_ms,
                batch_ms,
            );
        }
    }

    let materialize_started_at = Instant::now();
    let mut groups = vec![Vec::new(); next_class_id.max(1)];
    for (token_idx, &class_id) in partition.iter().enumerate() {
        groups[class_id].push(token_idx);
    }
    let result: VocabEquivalenceResult = groups.into_iter().filter(|group| !group.is_empty()).collect();
    let materialize_ms = elapsed_ms(materialize_started_at);

    if profile_compile {
        let totals = &*VOCAB_SIGNATURE_PROFILE_TOTALS;
        eprintln!(
            "[glrmask/profile][vocab_summary] dfa_states={} reachable_states={} relevant_states={} tokens={} build_dfa_ms={:.3} order_states_ms={:.3} batch_ms={:.3} signature_ms={:.3} refine_ms={:.3} materialize_ms={:.3} classes={}",
            dfa.num_states,
            reachable_states.unwrap_or(0),
            num_initial_states,
            num_tokens,
            build_dfa_ms,
            order_states_ms,
            batch_total_ms,
            signature_total_ms,
            refine_total_ms,
            materialize_ms,
            result.len(),
        );
        eprintln!(
            "[glrmask/profile][vocab_signature] tokens={} zero_target_tokens={} single_target_tokens={} multi_target_tokens={} tokens_with_targets={} target_positions={} run_batch_work_ms={:.3} run_batch_inner_work_ms={:.3} collect_targets_work_ms={:.3} try_single_target_work_ms={:.3} try_single_target_hits={} try_single_target_fallbacks={} hash_suffix_calls={} hash_suffix_work_ms={:.3} hash_suffix_setup_work_ms={:.3} hash_suffix_bfs_work_ms={:.3} hash_suffix_run_suffix_work_ms={:.3} hash_suffix_propagate_work_ms={:.3} hash_suffix_hash_work_ms={:.3} hash_suffix_nodes={} finish_work_ms={:.3}",
            totals.tokens.load(Ordering::Relaxed),
            totals.zero_target_tokens.load(Ordering::Relaxed),
            totals.single_target_tokens.load(Ordering::Relaxed),
            totals.multi_target_tokens.load(Ordering::Relaxed),
            totals.tokens_with_targets.load(Ordering::Relaxed),
            totals.target_positions.load(Ordering::Relaxed),
            ns_to_ms(totals.run_batch_ns.load(Ordering::Relaxed)),
            ns_to_ms(totals.run_batch_inner_ns.load(Ordering::Relaxed)),
            ns_to_ms(totals.collect_targets_ns.load(Ordering::Relaxed)),
            ns_to_ms(totals.try_single_target_ns.load(Ordering::Relaxed)),
            totals.try_single_target_hits.load(Ordering::Relaxed),
            totals.try_single_target_fallbacks.load(Ordering::Relaxed),
            totals.hash_suffix_calls.load(Ordering::Relaxed),
            ns_to_ms(totals.hash_suffix_ns.load(Ordering::Relaxed)),
            ns_to_ms(totals.hash_suffix_setup_ns.load(Ordering::Relaxed)),
            ns_to_ms(totals.hash_suffix_bfs_ns.load(Ordering::Relaxed)),
            ns_to_ms(totals.hash_suffix_run_suffix_ns.load(Ordering::Relaxed)),
            ns_to_ms(totals.hash_suffix_propagate_ns.load(Ordering::Relaxed)),
            ns_to_ms(totals.hash_suffix_hash_ns.load(Ordering::Relaxed)),
            totals.hash_suffix_nodes.load(Ordering::Relaxed),
            ns_to_ms(totals.finish_ns.load(Ordering::Relaxed)),
        );
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::ast::{bytes, choice};
    use crate::compiler::compile::build_tokenizer_from_exprs;
    use crate::compiler::stages::equivalence_analysis::compat::TokenizerView;
    use std::collections::BTreeMap;

    #[test]
    fn test_disallowed_follow_merges_ab_and_ac() {
        let tokenizer = build_tokenizer_from_exprs(&[
            choice(vec![bytes(b"a"), bytes(b"b")]),
            choice(vec![bytes(b"b"), bytes(b"c")]),
        ]);
        let tokenizer_view = TokenizerView::new(&tokenizer);
        let tokens = vec![b"ab".to_vec(), b"ac".to_vec()];
        let initial_states = vec![tokenizer_view.initial_state_id()];

        let mut disallowed = BTreeMap::new();
        let mut bitset = BitSet::new(2);
        bitset.set(0);
        disallowed.insert(0u32, bitset);

        let classes = find_vocab_equivalence_classes_with_follow(
            &tokenizer_view,
            &tokens,
            &initial_states,
            &disallowed,
        );

        assert!(
            classes.iter().any(|class| class == &vec![0, 1]),
            "expected ab and ac to be equivalent with disallowed follows, got {classes:?}"
        );
    }

    #[test]
    fn test_json_array_vocab_equivalence_with_follows() {
        let tokenizer = build_tokenizer_from_exprs(&[
            bytes(b"a"),
            bytes(b"bc"),
        ]);
        let tokenizer_view = TokenizerView::new(&tokenizer);
        let tokens = vec![b"a".to_vec(), b"ab".to_vec()];
        let initial_states = vec![tokenizer_view.initial_state_id()];

        let mut disallowed = BTreeMap::new();
        let mut bitset = BitSet::new(2);
        bitset.set(0);
        disallowed.insert(0u32, bitset);

        let classes = find_vocab_equivalence_classes_with_follow(
            &tokenizer_view,
            &tokens,
            &initial_states,
            &disallowed,
        );

        // Should be two classes
        assert!(
            classes.iter().any(|class| class == &vec![0]) && classes.iter().any(|class| class == &vec![1]),
            "expected ',' and ',-' to be in separate classes, got {classes:?}"
        );
    }
}
