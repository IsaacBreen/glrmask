//! Medium-cost vocab equivalence analysis with first-byte bucketing.
//!
//! Tokens that share early byte structure are grouped together, with bulk
//! handling for self-loop-heavy cases and per-token fallback work for the rest.

use super::super::compat::TokenizerView;
use ahash::{AHasher, RandomState};
use hashbrown::HashMap;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{BuildHasher, Hasher};

use super::super::disallowed_follows::normalize_disallowed_follows;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;

pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;

type EdgeList = SmallVec<[(usize, usize); 4]>;

const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;
const NONE: u32 = u32::MAX;
const STATE_NONE: usize = usize::MAX;

/// Flat DFA with 256-byte transition tables and self-loop bitsets.
struct Dfa {
    start_state: usize,
    transitions: Vec<[u32; 256]>,
    finalizers: Vec<SmallVec<[usize; 4]>>,
    is_dead_end: Vec<bool>,
    num_groups: usize,
    possible_future_groups: Vec<SmallVec<[usize; 4]>>,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
    self_loop_bytes: Vec<U8Set>,
    empty_suffix_hash: u64,
    /// Bitmap: bit `gid` is set if state has a finalizer for group `gid`
    finalizer_bits: Vec<u32>,
    /// Bitmap: bit `gid` is set if the finalizer is greedy (NOT non-greedy)
    greedy_bits: Vec<u32>,
    disallowed_follows: Vec<BitSet>,
}

impl Dfa {
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
        h.write_u64(disallowed.len() as u64);
        for &word in disallowed.words() {
            h.write_u64(word);
        }
        h.write_u64(hash_filtered_group_list(&self.possible_future_groups[state], disallowed));
        h.finish()
    }

    #[inline]
    fn disallowed_for(&self, gid: usize) -> &BitSet {
        &self.disallowed_follows[gid]
    }

    #[inline]
    fn empty_suffix_hash_for(&self, gid: usize) -> u64 {
        let disallowed = self.disallowed_for(gid);
        if disallowed.is_zero() {
            return self.empty_suffix_hash;
        }
        let end_state = if self.is_dead_end[self.start_state] {
            STATE_NONE
        } else {
            self.start_state
        };
        let mut h = new_hasher();
        h.write_u64(self.completion_with_disallowed(end_state, Some(disallowed)));
        h.finish()
    }
}

// Hashing.

static HASH_RANDOM_STATE: Lazy<RandomState> =
    Lazy::new(|| RandomState::with_seeds(HASH_SEED1, HASH_SEED2, HASH_SEED3, HASH_SEED4));

#[inline]
fn new_hasher() -> AHasher {
    HASH_RANDOM_STATE.build_hasher()
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

// DFA build.

fn build_dfa(tokenizer: &TokenizerView, disallowed_follows: &BTreeMap<u32, BitSet>) -> Dfa {
    let dfa = tokenizer.dfa();
    assert!(dfa.states.len() <= u32::MAX as usize, "DFA too large");

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

    let mut transitions = Vec::with_capacity(dfa.states.len());
    let mut finalizers = Vec::with_capacity(dfa.states.len());
    let mut is_dead_end = Vec::with_capacity(dfa.states.len());
    let mut possible_future_groups = Vec::with_capacity(dfa.states.len());
    let mut completion_hash = Vec::with_capacity(dfa.states.len());

    for (s, state) in dfa.states.iter().enumerate() {
        let mut table = [NONE; 256];
        for (byte_idx, &target) in dfa.transitions_for(s).iter().enumerate() {
            table[byte_idx] = target;
        }
        transitions.push(table);
        finalizers.push(state.finalizers.iter().copied().collect());
        is_dead_end.push(state.possible_future_group_ids.is_empty());
        let future_groups: SmallVec<[usize; 4]> =
            state.possible_future_group_ids.iter().copied().collect();
        completion_hash.push(hash_group_list(future_groups.iter().copied()));
        possible_future_groups.push(future_groups);
    }

    let none_completion_hash = {
        let mut h = new_hasher();
        h.write_u8(0);
        h.finish()
    };

    let self_loop_bytes: Vec<U8Set> = (0..transitions.len())
        .map(|s| {
            let mut bits = U8Set::empty();
            for b in 0..=255u8 {
                if transitions[s][b as usize] == s as u32 {
                    bits.insert(b);
                }
            }
            bits
        })
        .collect();

    // Build finalizer bitmaps for fast advance_states
    let mut finalizer_bits: Vec<u32> = vec![0u32; transitions.len()];
    let mut greedy_bits: Vec<u32> = vec![0u32; transitions.len()];
    for (s, fins) in finalizers.iter().enumerate() {
        for &gid in fins {
            if (gid as u32) < 32 {
                finalizer_bits[s] |= 1u32 << gid;
                greedy_bits[s] |= 1u32 << gid;
            }
        }
    }

    let empty_suffix_hash = {
        let end = if is_dead_end[dfa.start_state] {
            None
        } else {
            Some(dfa.start_state)
        };
        let ch = if let Some(s) = end {
            completion_hash[s]
        } else {
            none_completion_hash
        };
        let mut h = new_hasher();
        h.write_u64(ch);
        h.finish()
    };

    Dfa {
        start_state: dfa.start_state,
        transitions,
        finalizers,
        is_dead_end,
        num_groups,
        possible_future_groups,
        completion_hash,
        none_completion_hash,
        self_loop_bytes,
        empty_suffix_hash,
        finalizer_bits,
        greedy_bits,
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

// Self-loop check helpers.

/// Check if all states at (d_states, d_mp) can be bulk-assigned.
/// Returns (can_bulk, bulk_hash) if possible.
fn try_bulk_assign(
    dfa: &Dfa,
    depth_states: &[u32],
    depth_match_positions: &[u32],
    num_initial_states: usize,
    num_groups: usize,
    subtree_bytes: &U8Set,
) -> Option<u64> {
    // Self-loop check: intersect self_loop_bytes for all alive states
    let mut sl_inter = U8Set::all();
    let mut any_alive = false;
    for state_index in 0..num_initial_states {
        let current_state = depth_states[state_index];
        if current_state != NONE {
            any_alive = true;
            sl_inter &= dfa.self_loop_bytes[current_state as usize];
        }
    }

    if !any_alive || !subtree_bytes.is_subset(&sl_inter) {
        return None;
    }

    try_bulk_assign_no_selfloop(
        dfa,
        depth_states,
        depth_match_positions,
        num_initial_states,
        num_groups,
    )
}

/// Compute bulk hash after self-loop check already passed.
/// Checks can_bulk (greedy match positions) and computes the hash.
fn try_bulk_assign_no_selfloop(
    dfa: &Dfa,
    depth_states: &[u32],
    depth_match_positions: &[u32],
    num_initial_states: usize,
    num_groups: usize,
) -> Option<u64> {
    // can_bulk check: every match position > 0 must be for a group where
    // the current state has a greedy finalizer
    let can_bulk = (0..num_initial_states).all(|state_index| {
        let current_state = depth_states[state_index];
        let base = state_index * num_groups;
        (0..num_groups).all(|gid| {
            let pv = depth_match_positions[base + gid];
            if pv > 0 && pv != NONE {
                current_state != NONE
                    && dfa.finalizers[current_state as usize]
                        .iter()
                        .any(|&state_gid| state_gid == gid)
            } else {
                true
            }
        })
    });

    if !can_bulk {
        return None;
    }

    // Compute bulk hash
    let mut hash = HASH_SEED3;
    for state_index in 0..num_initial_states {
        let end_state = depth_states[state_index];
        let base = state_index * num_groups;
        let completion = if end_state == NONE {
            dfa.none_completion_hash
        } else {
            dfa.completion_hash[end_state as usize]
        };
        let has_any = (0..num_groups).any(|gid| depth_match_positions[base + gid] != NONE);
        let sig = if has_any {
            let mut h = new_hasher();
            h.write_u64(completion);
            for gid in 0..num_groups {
                let pv = depth_match_positions[base + gid];
                if pv != NONE && pv > 0 {
                    h.write_u64(gid as u64);
                    h.write_u64(dfa.empty_suffix_hash_for(gid));
                }
            }
            h.finish()
        } else {
            completion
        };
        hash = hash.wrapping_mul(HASH_SEED1).wrapping_add(sig);
    }

    Some(hash)
}

/// Advance DFA states by one byte, updating states and match positions.
/// Uses bitmap finalizers for fast inner loop when group counts stay small, and falls back to
/// full finalizer list for larger group counts.
fn advance_states(
    dfa: &Dfa,
    parent_states: &[u32],
    parent_match_positions: &[u32],
    child_states: &mut [u32],
    child_match_positions: &mut [u32],
    byte: u8,
    depth: u32,
    num_initial_states: usize,
    num_groups: usize,
) {
    child_match_positions[..num_initial_states * num_groups]
        .copy_from_slice(&parent_match_positions[..num_initial_states * num_groups]);
    let use_bitmap = num_groups <= 32;
    for state_index in 0..num_initial_states {
        let parent_state = parent_states[state_index];
        let match_base = state_index * num_groups;
        if parent_state == NONE {
            child_states[state_index] = NONE;
        } else {
            let next_state = dfa.transitions[parent_state as usize][byte as usize];
            if next_state == NONE {
                child_states[state_index] = NONE;
            } else {
                let next_state_index = next_state as usize;
                if use_bitmap {
                    let finalizer_bits = dfa.finalizer_bits[next_state_index];
                    if finalizer_bits != 0 {
                        let greedy_bits = dfa.greedy_bits[next_state_index];
                        let mut bits = finalizer_bits;
                        while bits != 0 {
                            let gid = bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            if gid < num_groups {
                                if (greedy_bits >> gid) & 1 == 1
                                    || child_match_positions[match_base + gid] == NONE
                                {
                                    child_match_positions[match_base + gid] = depth;
                                }
                            }
                        }
                    }
                } else {
                    for &gid in &dfa.finalizers[next_state_index] {
                        if gid < num_groups {
                            child_match_positions[match_base + gid] = depth;
                        }
                    }
                }
                child_states[state_index] = if dfa.is_dead_end[next_state_index] {
                    NONE
                } else {
                    next_state
                };
            }
        }
    }
}

// Per-token signature computation.

struct Scratch {
    current_states: Vec<usize>,
    active_indices: Vec<usize>,
    match_positions: Vec<u32>,
    targets: Vec<usize>,
    dag: HashMap<usize, (u64, EdgeList)>,
    dag_end_states: HashMap<usize, usize>,
    dag_queue: Vec<usize>,
    dag_disallowed: HashMap<usize, BitSet>,
}

impl Scratch {
    fn new(num_states: usize, num_groups: usize) -> Self {
        Scratch {
            current_states: vec![STATE_NONE; num_states],
            active_indices: Vec::new(),
            match_positions: vec![NONE; num_states * num_groups],
            targets: Vec::new(),
            dag: HashMap::new(),
            dag_end_states: HashMap::new(),
            dag_queue: Vec::new(),
            dag_disallowed: HashMap::new(),
        }
    }
}

/// Run DFA from all initial states on a token with self-loop early exit.
fn run_batch(
    dfa: &Dfa,
    scratch: &mut Scratch,
    slice: &[u8],
    initial_states: &[usize],
) {
    let num_states = initial_states.len();
    let num_groups = dfa.num_groups;
    let len = slice.len();

    scratch.current_states[..num_states].clone_from_slice(initial_states);
    scratch.active_indices.clear();
    scratch.match_positions[..num_states * num_groups].fill(NONE);

    let has_bytes = !slice.is_empty();
    let first_byte = if has_bytes { slice[0] } else { 0 };

    for (i, &state) in initial_states.iter().enumerate() {
        let base = i * num_groups;
        for &gid in &dfa.finalizers[state] {
            if gid < num_groups && scratch.match_positions[base + gid] == NONE {
                scratch.match_positions[base + gid] = 0;
            }
        }
        if dfa.is_dead_end[state] {
            scratch.current_states[i] = STATE_NONE;
            continue;
        }
        if has_bytes && dfa.transitions[state][first_byte as usize] == NONE {
            scratch.current_states[i] = STATE_NONE;
            continue;
        }
        scratch.active_indices.push(i);
    }

    if has_bytes && !scratch.active_indices.is_empty() {
        let mut active_len = scratch.active_indices.len();
        for (pos, &byte) in slice.iter().enumerate() {
            let position = (pos + 1) as u32;
            let mut next_len = 0usize;
            for idx in 0..active_len {
                let i = scratch.active_indices[idx];
                let base = i * num_groups;
                let next_state = dfa.transitions[scratch.current_states[i]][byte as usize];
                if next_state != NONE {
                    let ns = next_state as usize;
                    scratch.current_states[i] = ns;
                    for &gid in &dfa.finalizers[ns] {
                        if gid < num_groups {
                            let ix = base + gid;
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

            // Self-loop early exit
            if pos + 1 < len {
                let mut sl = U8Set::all();
                for idx in 0..active_len {
                    let i = scratch.active_indices[idx];
                    let s = scratch.current_states[i];
                    sl &= dfa.self_loop_bytes[s];
                }
                let all_self_loop = slice[pos + 1..].iter().all(|&b| sl.contains(b));
                if all_self_loop {
                    let token_len = len as u32;
                    for idx in 0..active_len {
                        let i = scratch.active_indices[idx];
                        let base = i * num_groups;
                        let s = scratch.current_states[i];
                        for &gid in &dfa.finalizers[s] {
                            if gid < num_groups {
                                scratch.match_positions[base + gid] = token_len;
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    scratch.targets.clear();
    if num_groups > 0 {
        for base in (0..num_states * num_groups).step_by(num_groups) {
            for gid in 0..num_groups {
                let pv = scratch.match_positions[base + gid];
                if pv != NONE && pv > 0 && (pv as usize) <= len {
                    scratch.targets.push(pv as usize);
                }
            }
        }
    }
    scratch.targets.sort_unstable();
    scratch.targets.dedup();
}

fn run_suffix(
    dfa: &Dfa,
    slice: &[u8],
    base_pos: usize,
    match_positions: &mut [u32],
) -> (Option<usize>, EdgeList) {
    let num_groups = dfa.num_groups;
    match_positions[..num_groups].fill(NONE);
    let mut current_state = dfa.start_state;
    let mut done = dfa.is_dead_end[current_state];

    for &gid in &dfa.finalizers[current_state] {
        if gid < num_groups && match_positions[gid] == NONE {
            match_positions[gid] = 0;
        }
    }

    for (idx, &byte) in slice.iter().enumerate() {
        if done {
            break;
        }
        let next_state = dfa.transitions[current_state][byte as usize];
        if next_state == NONE {
            done = true;
            break;
        }
        current_state = next_state as usize;
        let position = (idx + 1) as u32;
        for &gid in &dfa.finalizers[current_state] {
            if gid < num_groups {
                match_positions[gid] = position;
            }
        }
        if dfa.is_dead_end[current_state] {
            done = true;
        }
    }

    let end_state = if done { None } else { Some(current_state) };
    let edges: EdgeList = (0..num_groups)
        .filter_map(|gid| {
            let pv = match_positions[gid];
            (pv != NONE && pv != 0).then(|| (gid, base_pos + pv as usize))
        })
        .collect();
    (end_state, edges)
}

fn hash_suffixes(dfa: &Dfa, slice: &[u8], scratch: &mut Scratch) {
    let len = slice.len();
    let num_groups = dfa.num_groups;
    let mut suffix_match_positions = vec![NONE; num_groups];
    scratch.dag.clear();
    scratch.dag_end_states.clear();
    scratch.dag_queue.clear();
    scratch.dag_disallowed.clear();

    for &pos in &scratch.targets {
        if pos <= len && !scratch.dag.contains_key(&pos) {
            scratch.dag_queue.push(pos);
            scratch.dag.insert(pos, (0, EdgeList::new()));
        }
    }

    for base in (0..scratch.current_states.len() * num_groups).step_by(num_groups) {
        for gid in 0..num_groups {
            let pv = scratch.match_positions[base + gid];
            if pv != NONE && pv > 0 {
                intersect_node_disallowed(scratch, pv as usize, dfa.disallowed_for(gid));
            }
        }
    }

    let mut cursor = 0;
    while cursor < scratch.dag_queue.len() {
        let pos = scratch.dag_queue[cursor];
        cursor += 1;
        let (end_state, edges) =
            run_suffix(dfa, &slice[pos..], pos, &mut suffix_match_positions);
        for &(_, target) in &edges {
            if target <= len && !scratch.dag.contains_key(&target) {
                scratch.dag_queue.push(target);
                scratch.dag.insert(target, (0, EdgeList::new()));
            }
        }
        scratch.dag_end_states.insert(pos, end_state.unwrap_or(STATE_NONE));
        scratch.dag.insert(pos, (0, edges));
    }

    scratch.dag_queue.sort_unstable();
    for idx in 0..scratch.dag_queue.len() {
        let pos = scratch.dag_queue[idx];
        let (_, edges) = scratch.dag[&pos].clone();
        for &(gid, target) in &edges {
            if node_disallows_gid(scratch, pos, gid) {
                continue;
            }
            if target <= len {
                intersect_node_disallowed(scratch, target, dfa.disallowed_for(gid));
            }
        }
    }

    scratch.dag_queue.sort_unstable_by(|a, b| b.cmp(a));
    for idx in 0..scratch.dag_queue.len() {
        let pos = scratch.dag_queue[idx];
        let (_, edges) = scratch.dag[&pos].clone();
        let end_state = scratch.dag_end_states.get(&pos).copied().unwrap_or(STATE_NONE);
        let mut h = new_hasher();
        h.write_u64(dfa.completion_with_disallowed(end_state, scratch.dag_disallowed.get(&pos)));
        for &(gid, target) in &edges {
            if node_disallows_gid(scratch, pos, gid) {
                continue;
            }
            h.write_u64(gid as u64);
            h.write_u64(scratch.dag.get(&target).map_or(0, |e| e.0));
        }
        scratch.dag.get_mut(&pos).unwrap().0 = h.finish();
    }
}

fn token_signature(
    dfa: &Dfa,
    token: &[u8],
    initial_states: &[usize],
    scratch: &mut Scratch,
) -> u64 {
    run_batch(dfa, scratch, token, initial_states);
    if !scratch.targets.is_empty() {
        hash_suffixes(dfa, token, scratch);
    }

    let mut sig: u64 = HASH_SEED3;
    for i in 0..initial_states.len() {
        let completion = dfa.completion(scratch.current_states[i]);
        let base = i * dfa.num_groups;
        let mp = &scratch.match_positions[base..base + dfa.num_groups];
        let state_sig = if mp.iter().any(|&pv| pv != NONE) {
            let mut h = new_hasher();
            h.write_u64(completion);
            for (gid, &pv) in mp.iter().enumerate() {
                if pv != NONE && pv > 0 {
                    h.write_u64(gid as u64);
                    h.write_u64(scratch.dag.get(&(pv as usize)).map_or(0, |e| e.0));
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

// Recursive classification.

/// Recursively classify tokens by byte prefix, bulk-assigning where possible.
/// `sorted` must be a lexicographically sorted slice of token indices that all share
/// the same prefix bytes[0..depth). Tokens that reach their full length without
/// being bulk-assigned are pushed to `non_bulk`. Hash assignments are collected
/// into `assignments` for thread-safe parallel use.
fn classify_sorted_collect<S: AsRef<[u8]>>(
    dfa: &Dfa,
    strings: &[S],
    sorted: &[usize],
    parent_states: &[u32],
    parent_match_positions: &[u32],
    depth: usize,
    num_initial_states: usize,
    num_groups: usize,
    assignments: &mut Vec<(usize, u64)>,
    non_bulk: &mut Vec<usize>,
) {
    if sorted.is_empty() {
        return;
    }

    // Skip leaves (len <= depth) — they come first in lexicographic order
    let mut pos = 0;
    while pos < sorted.len() && strings[sorted[pos]].as_ref().len() <= depth {
        non_bulk.push(sorted[pos]);
        pos += 1;
    }
    if pos >= sorted.len() {
        return;
    }
    let longer = &sorted[pos..];

    let mut child_states = vec![NONE; num_initial_states];
    let mut child_match_positions = vec![NONE; num_initial_states * num_groups];

    // Iterate groups by byte[depth] — contiguous in sorted order
    let mut i = 0;
    while i < longer.len() {
        let b = strings[longer[i]].as_ref()[depth];
        let group_start = i;
        i += 1;
        while i < longer.len() && strings[longer[i]].as_ref()[depth] == b {
            i += 1;
        }
        let group = &longer[group_start..i];

        advance_states(
            dfa,
            parent_states,
            parent_match_positions,
            &mut child_states,
            &mut child_match_positions,
            b,
            (depth + 1) as u32,
            num_initial_states,
            num_groups,
        );

        // Compute byte_set for this group (all bytes at positions > depth)
        let mut byte_set = U8Set::empty();
        for &ti in group {
            for &bb in &strings[ti].as_ref()[depth + 1..] {
                byte_set.insert(bb);
            }
        }

        if let Some(hash) = try_bulk_assign(
            dfa,
            &child_states,
            &child_match_positions,
            num_initial_states,
            num_groups,
            &byte_set,
        ) {
            for &ti in group {
                assignments.push((ti, hash));
            }
            continue;
        }

        classify_sorted_collect(
            dfa,
            strings,
            group,
            &child_states,
            &child_match_positions,
            depth + 1,
            num_initial_states,
            num_groups,
            assignments,
            non_bulk,
        );
    }
}

// Public API.

/// Medium-cost vocab equivalence analysis with recursive byte-level bucketing.
pub fn find_vocab_equivalence_classes_with_follow<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> VocabEquivalenceResult {
    let dfa = build_dfa(tokenizer, disallowed_follows);
    let num_initial_states = initial_states.len();
    let num_groups = dfa.num_groups;
    let num_tokens = strings.len();

    if num_initial_states == 0 || num_tokens == 0 {
        return BTreeSet::from_iter(vec![(0..num_tokens).collect()]);
    }

    let mut hashes = vec![0u64; num_tokens];

    let mut depth0_states = vec![NONE; num_initial_states];
    let mut depth0_match_positions = vec![NONE; num_initial_states * num_groups];
    for state_index in 0..num_initial_states {
        let initial_state = initial_states[state_index];
        let match_base = state_index * num_groups;
        for &gid in &dfa.finalizers[initial_state] {
            if gid < num_groups && depth0_match_positions[match_base + gid] == NONE {
                depth0_match_positions[match_base + gid] = 0;
            }
        }
        depth0_states[state_index] = if dfa.is_dead_end[initial_state] {
            NONE
        } else {
            initial_state as u32
        };
    }

    // Pre-sort tokens lexicographically.
    let mut sorted_indices: Vec<usize> = (0..num_tokens).collect();
    sorted_indices.sort_unstable_by(|&a, &b| strings[a].as_ref().cmp(strings[b].as_ref()));

    // Handle empty tokens (leaves at depth 0) — they come first in sorted order
    let mut empty_end = 0;
    while empty_end < sorted_indices.len()
        && strings[sorted_indices[empty_end]].as_ref().is_empty()
    {
        empty_end += 1;
    }
    let empty_tokens = &sorted_indices[..empty_end];
    let longer = &sorted_indices[empty_end..];

    // Find depth-0 group boundaries (by byte[0])
    let mut group_ranges: Vec<(usize, usize)> = Vec::new();
    if !longer.is_empty() {
        let mut start = 0;
        for i in 1..longer.len() {
            if strings[longer[i]].as_ref()[0] != strings[longer[i - 1]].as_ref()[0] {
                group_ranges.push((start, i));
                start = i;
            }
        }
        group_ranges.push((start, longer.len()));
    }

    // Process depth-0 groups in parallel
    let group_results: Vec<(Vec<(usize, u64)>, Vec<usize>)> = group_ranges
        .par_iter()
        .map(|&(start, end)| {
            let group = &longer[start..end];
            let b = strings[group[0]].as_ref()[0];

            let mut child_states = vec![NONE; num_initial_states];
            let mut child_match_positions = vec![NONE; num_initial_states * num_groups];
            advance_states(
                &dfa,
                &depth0_states,
                &depth0_match_positions,
                &mut child_states,
                &mut child_match_positions,
                b,
                1,
                num_initial_states,
                num_groups,
            );

            let mut byte_set = U8Set::empty();
            for &ti in group {
                for &bb in &strings[ti].as_ref()[1..] {
                    byte_set.insert(bb);
                }
            }

            let mut assignments = Vec::new();
            let mut non_bulk = Vec::new();

            if let Some(hash) = try_bulk_assign(
                &dfa,
                &child_states,
                &child_match_positions,
                num_initial_states,
                num_groups,
                &byte_set,
            ) {
                for &ti in group {
                    assignments.push((ti, hash));
                }
            } else {
                classify_sorted_collect(
                    &dfa,
                    strings,
                    group,
                    &child_states,
                    &child_match_positions,
                    1,
                    num_initial_states,
                    num_groups,
                    &mut assignments,
                    &mut non_bulk,
                );
            }

            (assignments, non_bulk)
        })
        .collect();

    // Merge results
    let mut non_bulk_tokens: Vec<usize> = Vec::from(empty_tokens);
    for (assignments, non_bulk) in group_results {
        for (ti, hash) in assignments {
            hashes[ti] = hash;
        }
        non_bulk_tokens.extend(non_bulk);
    }


    // Process non-bulk tokens in parallel.
    let non_bulk_hashes: Vec<(usize, u64)> = non_bulk_tokens
        .par_iter()
        .map_init(
            || Scratch::new(num_initial_states, num_groups),
            |scratch, &ti| {
                let token = strings[ti].as_ref();
                (ti, token_signature(&dfa, token, initial_states, scratch))
            },
        )
        .collect();

    for (ti, h) in non_bulk_hashes {
        hashes[ti] = h;
    }
    // Group by hash
    let mut groups: HashMap<u64, Vec<usize>> = HashMap::with_capacity(num_tokens / 4);
    for (ti, &h) in hashes.iter().enumerate() {
        groups.entry(h).or_default().push(ti);
    }

    groups.into_values().collect()
}

// Partition comparison.

