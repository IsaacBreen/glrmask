#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(clippy::all)]
#![allow(unreachable_code)]
#![allow(unused_assignments)]
//! Flat vocab equivalence analysis: classify tokens by first-byte DFA behavior.
//!
//! Instead of building a trie, groups tokens by their first byte(s) and
//! uses self-loop detection for bulk assignment. Non-bulk tokens are
//! processed in parallel with rayon.

use super::super::compat::{Sep1Tokenizer, FlatDfa, FlatDfaState, GroupID};
use ahash::{AHasher, RandomState};
use hashbrown::HashMap;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::BTreeSet;
use std::hash::{BuildHasher, Hasher};

pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;

type EdgeList = SmallVec<[(usize, usize); 4]>;

const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;
const NONE: u32 = u32::MAX;
const STATE_NONE: usize = usize::MAX;

#[derive(Clone, Copy)]
struct Finalizer {
    gid: usize,
}

/// Flat DFA with 256-byte transition tables and self-loop bitsets.
struct Dfa {
    start_state: usize,
    transitions: Vec<[u32; 256]>,
    finalizers: Vec<SmallVec<[Finalizer; 4]>>,
    is_dead_end: Vec<bool>,
    num_groups: usize,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
    self_loop_bytes: Vec<[u64; 4]>,
    empty_suffix_hash: u64,
    /// Bitmap: bit `gid` is set if state has a finalizer for group `gid`
    finalizer_bits: Vec<u32>,
    /// Bitmap: bit `gid` is set if the finalizer is greedy (NOT non-greedy)
    greedy_bits: Vec<u32>,
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
}

// ---- Hashing ----

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

// ---- DFA build ----

fn build_dfa(regex: &Sep1Tokenizer) -> Dfa {
    let dfa = regex.dfa();
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
    let mut completion_hash = Vec::with_capacity(dfa.states.len());

    for state in &dfa.states {
        let mut table = [NONE; 256];
        for (byte_idx, &target) in state.transitions.iter().enumerate() { let byte = byte_idx as u8;
            table[byte_idx] = target;
        }
        transitions.push(table);
        finalizers.push(
            state
                .finalizers
                .iter()
                .map(|&gid| Finalizer {
                    gid,
                })
                .collect(),
        );
        is_dead_end.push(state.possible_future_group_ids.is_empty());
        completion_hash.push(hash_group_list(
            state.possible_future_group_ids.iter().copied(),
        ));
    }

    let none_completion_hash = {
        let mut h = new_hasher();
        h.write_u8(0);
        h.finish()
    };

    let self_loop_bytes: Vec<[u64; 4]> = (0..transitions.len())
        .map(|s| {
            let mut bits = [0u64; 4];
            for b in 0..=255u8 {
                if transitions[s][b as usize] == s as u32 {
                    bits[b as usize >> 6] |= 1u64 << (b & 63);
                }
            }
            bits
        })
        .collect();

    // Build finalizer bitmaps for fast advance_states
    let mut finalizer_bits: Vec<u32> = vec![0u32; transitions.len()];
    let mut greedy_bits: Vec<u32> = vec![0u32; transitions.len()];
    for (s, fins) in finalizers.iter().enumerate() {
        for f in fins as &[Finalizer] {
            if (f.gid as u32) < 32 {
                finalizer_bits[s] |= 1u32 << f.gid;
                greedy_bits[s] |= 1u32 << f.gid;
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
        completion_hash,
        none_completion_hash,
        self_loop_bytes,
        empty_suffix_hash,
        finalizer_bits,
        greedy_bits,
    }
}

// ---- Self-loop check helpers ----

#[inline]
fn u8set_is_subset(a: &[u64; 4], b: &[u64; 4]) -> bool {
    (a[0] & !b[0]) == 0 && (a[1] & !b[1]) == 0 && (a[2] & !b[2]) == 0 && (a[3] & !b[3]) == 0
}

/// Check if all states at (d_states, d_mp) can be bulk-assigned.
/// Returns (can_bulk, bulk_hash) if possible.
fn try_bulk_assign(
    dfa: &Dfa,
    d_states: &[u32],
    d_mp: &[u32],
    ni: usize,
    ng: usize,
    subtree_bytes: &[u64; 4],
) -> Option<u64> {
    // Self-loop check: intersect self_loop_bytes for all alive states
    let mut sl_inter = [!0u64; 4];
    let mut any_alive = false;
    for si in 0..ni {
        let cs = d_states[si];
        if cs != NONE {
            any_alive = true;
            let sl = &dfa.self_loop_bytes[cs as usize];
            for i in 0..4 {
                sl_inter[i] &= sl[i];
            }
        }
    }

    if !any_alive || !u8set_is_subset(subtree_bytes, &sl_inter) {
        return None;
    }

    try_bulk_assign_no_selfloop(dfa, d_states, d_mp, ni, ng)
}

/// Compute bulk hash after self-loop check already passed.
/// Checks can_bulk (greedy match positions) and computes the hash.
fn try_bulk_assign_no_selfloop(
    dfa: &Dfa,
    d_states: &[u32],
    d_mp: &[u32],
    ni: usize,
    ng: usize,
) -> Option<u64> {

    // can_bulk check: every mp > 0 must be for a group where
    // the current state has a greedy finalizer
    let can_bulk = (0..ni).all(|si| {
        let cs = d_states[si];
        let base = si * ng;
        (0..ng).all(|gid| {
            let pv = d_mp[base + gid];
            if pv > 0 && pv != NONE {
                cs != NONE
                    && dfa.finalizers[cs as usize]
                        .iter()
                        .any(|f| f.gid == gid)
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
    for si in 0..ni {
        let es = d_states[si];
        let base = si * ng;
        let completion = if es == NONE {
            dfa.none_completion_hash
        } else {
            dfa.completion_hash[es as usize]
        };
        let has_any = (0..ng).any(|gid| d_mp[base + gid] != NONE);
        let sig = if has_any {
            let mut h = new_hasher();
            h.write_u64(completion);
            for gid in 0..ng {
                let pv = d_mp[base + gid];
                if pv != NONE && pv > 0 {
                    h.write_u64(gid as u64);
                    h.write_u64(dfa.empty_suffix_hash);
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
/// Uses bitmap finalizers for fast inner loop when ng <= 32, falls back to
/// full finalizer list for larger group counts.
fn advance_states(
    dfa: &Dfa,
    parent_states: &[u32],
    parent_mp: &[u32],
    child_states: &mut [u32],
    child_mp: &mut [u32],
    byte: u8,
    depth: u32,
    ni: usize,
    ng: usize,
) {
    child_mp[..ni * ng].copy_from_slice(&parent_mp[..ni * ng]);
    let use_bitmap = ng <= 32;
    for si in 0..ni {
        let ps = parent_states[si];
        let mp_base = si * ng;
        if ps == NONE {
            child_states[si] = NONE;
        } else {
            let ns = dfa.transitions[ps as usize][byte as usize];
            if ns == NONE {
                child_states[si] = NONE;
            } else {
                let ns_u = ns as usize;
                if use_bitmap {
                    // Fast path: bitmap for ng <= 32
                    let fin = dfa.finalizer_bits[ns_u];
                    if fin != 0 {
                        let greedy = dfa.greedy_bits[ns_u];
                        let mut bits = fin;
                        while bits != 0 {
                            let gid = bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            if gid < ng {
                                if (greedy >> gid) & 1 == 1 || child_mp[mp_base + gid] == NONE {
                                    child_mp[mp_base + gid] = depth;
                                }
                            }
                        }
                    }
                } else {
                    // Full finalizer list for ng > 32
                    for f in &dfa.finalizers[ns_u] {
                        let gid = f.gid;
                        if gid < ng {
                            child_mp[mp_base + gid] = depth;
                        }
                    }
                }
                child_states[si] = if dfa.is_dead_end[ns_u] { NONE } else { ns };
            }
        }
    }
}

// ---- Per-token signature computation (reused from fast version) ----

struct Scratch {
    current_states: Vec<usize>,
    active_indices: Vec<usize>,
    match_positions: Vec<u32>,
    targets: Vec<usize>,
    dag: HashMap<usize, (u64, EdgeList)>,
    dag_queue: Vec<usize>,
}

impl Scratch {
    fn new(num_states: usize, num_groups: usize) -> Self {
        Scratch {
            current_states: vec![STATE_NONE; num_states],
            active_indices: Vec::new(),
            match_positions: vec![NONE; num_states * num_groups],
            targets: Vec::new(),
            dag: HashMap::new(),
            dag_queue: Vec::new(),
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
        for f in &dfa.finalizers[state] {
            if f.gid < num_groups && scratch.match_positions[base + f.gid] == NONE {
                scratch.match_positions[base + f.gid] = 0;
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
                    for f in &dfa.finalizers[ns] {
                        if f.gid < num_groups {
                            let ix = base + f.gid;
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
                let mut sl = [!0u64; 4];
                for idx in 0..active_len {
                    let i = scratch.active_indices[idx];
                    let s = scratch.current_states[i];
                    for k in 0..4 {
                        sl[k] &= dfa.self_loop_bytes[s][k];
                    }
                }
                let all_self_loop = slice[pos + 1..].iter().all(|&b| {
                    sl[b as usize >> 6] & (1u64 << (b & 63)) != 0
                });
                if all_self_loop {
                    let token_len = len as u32;
                    for idx in 0..active_len {
                        let i = scratch.active_indices[idx];
                        let base = i * num_groups;
                        let s = scratch.current_states[i];
                        for f in &dfa.finalizers[s] {
                            if f.gid < num_groups {
                                scratch.match_positions[base + f.gid] = token_len;
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
    let ng = dfa.num_groups;
    match_positions[..ng].fill(NONE);
    let mut current = dfa.start_state;
    let mut done = dfa.is_dead_end[current];

    for f in &dfa.finalizers[current] {
        if f.gid < ng && match_positions[f.gid] == NONE {
            match_positions[f.gid] = 0;
        }
    }

    for (idx, &byte) in slice.iter().enumerate() {
        if done {
            break;
        }
        let ns = dfa.transitions[current][byte as usize];
        if ns == NONE {
            done = true;
            break;
        }
        current = ns as usize;
        let position = (idx + 1) as u32;
        for f in &dfa.finalizers[current] {
            if f.gid < ng {
                match_positions[f.gid] = position;
            }
        }
        if dfa.is_dead_end[current] {
            done = true;
        }
    }

    let end_state = if done { None } else { Some(current) };
    let edges: EdgeList = (0..ng)
        .filter_map(|gid| {
            let pv = match_positions[gid];
            (pv != NONE && pv != 0).then(|| (gid, base_pos + pv as usize))
        })
        .collect();
    (end_state, edges)
}

fn hash_suffixes(dfa: &Dfa, slice: &[u8], scratch: &mut Scratch) {
    let len = slice.len();
    let ng = dfa.num_groups;
    let mut suffix_mp = vec![NONE; ng];
    scratch.dag.clear();
    scratch.dag_queue.clear();

    for &pos in &scratch.targets {
        if pos <= len && !scratch.dag.contains_key(&pos) {
            scratch.dag_queue.push(pos);
            scratch.dag.insert(pos, (0, EdgeList::new()));
        }
    }

    let mut cursor = 0;
    while cursor < scratch.dag_queue.len() {
        let pos = scratch.dag_queue[cursor];
        cursor += 1;
        let (end_state, edges) = run_suffix(dfa, &slice[pos..], pos, &mut suffix_mp);
        for &(_, target) in &edges {
            if target <= len && !scratch.dag.contains_key(&target) {
                scratch.dag_queue.push(target);
                scratch.dag.insert(target, (0, EdgeList::new()));
            }
        }
        let ch = dfa.completion(end_state.unwrap_or(STATE_NONE));
        scratch.dag.insert(pos, (ch, edges));
    }

    scratch.dag_queue.sort_unstable_by(|a, b| b.cmp(a));
    for idx in 0..scratch.dag_queue.len() {
        let pos = scratch.dag_queue[idx];
        let (ch, edges) = scratch.dag[&pos].clone();
        let mut h = new_hasher();
        h.write_u64(ch);
        for &(gid, target) in &edges {
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

// ---- Recursive classification (sorted-slice approach) ----

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
    parent_mp: &[u32],
    depth: usize,
    ni: usize,
    ng: usize,
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

    let mut child_states = vec![NONE; ni];
    let mut child_mp = vec![NONE; ni * ng];

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
            dfa, parent_states, parent_mp,
            &mut child_states, &mut child_mp,
            b, (depth + 1) as u32, ni, ng,
        );

        // Compute byte_set for this group (all bytes at positions > depth)
        let mut byte_set = [0u64; 4];
        for &ti in group {
            for &bb in &strings[ti].as_ref()[depth + 1..] {
                byte_set[bb as usize >> 6] |= 1u64 << (bb & 63);
            }
        }

        if let Some(hash) = try_bulk_assign(dfa, &child_states, &child_mp, ni, ng, &byte_set) {
            for &ti in group {
                assignments.push((ti, hash));
            }
            continue;
        }

        classify_sorted_collect(
            dfa, strings, group,
            &child_states, &child_mp,
            depth + 1, ni, ng, assignments, non_bulk,
        );
    }
}

// ---- Public API ----

pub fn find_vocab_equivalence_classes<S: AsRef<[u8]> + Sync>(
    regex: &Sep1Tokenizer,
    strings: &[S],
    initial_states: &[usize],
) -> VocabEquivalenceResult {
    find_vocab_equivalence_classes_with_follow(regex, strings, initial_states, None, None, None)
}

/// Flat vocab equivalence analysis with recursive byte-level classification.
///
/// Phase 1: Recursively group tokens by byte prefix. At each depth, advance
/// DFA states and check if all alive states self-loop on all remaining bytes
/// in the subtree. If so, bulk-assign a shared hash. Otherwise recurse deeper.
///
/// Phase 2: Process non-bulk tokens (leaves) in parallel with rayon.
pub fn find_vocab_equivalence_classes_with_follow<S: AsRef<[u8]> + Sync>(
    regex: &Sep1Tokenizer,
    strings: &[S],
    initial_states: &[usize],
    _suffix_group_mask: Option<&[bool]>,
    _ever_allowed_by_group: Option<&[Vec<bool>]>,
    _group_to_class: Option<&[usize]>,
) -> VocabEquivalenceResult {
    let t0 = std::time::Instant::now();
    let dfa = build_dfa(regex);
    let ni = initial_states.len();
    let ng = dfa.num_groups;
    let nt = strings.len();

    if ni == 0 || nt == 0 {
        return BTreeSet::from_iter(vec![(0..nt).collect()]);
    }

    let mut hashes = vec![0u64; nt];

    // Compute initial states/mp at depth 0
    let mut d0_states = vec![NONE; ni];
    let mut d0_mp = vec![NONE; ni * ng];
    for si in 0..ni {
        let s = initial_states[si];
        let mp_base = si * ng;
        for f in &dfa.finalizers[s] {
            if f.gid < ng && d0_mp[mp_base + f.gid] == NONE {
                d0_mp[mp_base + f.gid] = 0;
            }
        }
        d0_states[si] = if dfa.is_dead_end[s] { NONE } else { s as u32 };
    }

    let t1 = std::time::Instant::now();

    // Phase 1: Pre-sort tokens lexicographically
    let mut sorted_indices: Vec<usize> = (0..nt).collect();
    sorted_indices.sort_unstable_by(|&a, &b| strings[a].as_ref().cmp(strings[b].as_ref()));

    let t1b = std::time::Instant::now();

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

            let mut child_states = vec![NONE; ni];
            let mut child_mp = vec![NONE; ni * ng];
            advance_states(
                &dfa, &d0_states, &d0_mp,
                &mut child_states, &mut child_mp,
                b, 1, ni, ng,
            );

            let mut byte_set = [0u64; 4];
            for &ti in group {
                for &bb in &strings[ti].as_ref()[1..] {
                    byte_set[bb as usize >> 6] |= 1u64 << (bb & 63);
                }
            }

            let mut assignments = Vec::new();
            let mut non_bulk = Vec::new();

            if let Some(hash) = try_bulk_assign(&dfa, &child_states, &child_mp, ni, ng, &byte_set) {
                for &ti in group {
                    assignments.push((ti, hash));
                }
            } else {
                classify_sorted_collect(
                    &dfa, strings, group,
                    &child_states, &child_mp,
                    1, ni, ng, &mut assignments, &mut non_bulk,
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

    let t2 = std::time::Instant::now();

    // Phase 2: Process non-bulk tokens in parallel
    let non_bulk_hashes: Vec<(usize, u64)> = non_bulk_tokens
        .par_iter()
        .map_init(
            || Scratch::new(ni, ng),
            |scratch, &ti| {
                let token = strings[ti].as_ref();
                (ti, token_signature(&dfa, token, initial_states, scratch))
            },
        )
        .collect();

    for (ti, h) in non_bulk_hashes {
        hashes[ti] = h;
    }

    let t3 = std::time::Instant::now();

    // Group by hash
    let mut groups: HashMap<u64, Vec<usize>> = HashMap::with_capacity(nt / 4);
    for (ti, &h) in hashes.iter().enumerate() {
        groups.entry(h).or_default().push(ti);
    }

    let t4 = std::time::Instant::now();
    // sep1_debug!(
        // 2,
        // "Vocab equiv FLAT: dfa={:?}, sort={:?}, walk={:?} (non_bulk={}), par_compute={:?}, group={:?}, total={:?}",
        // t1 - t0,
        // t1b - t1,
        // t2 - t1b,
        // non_bulk_tokens.len(),
        // t3 - t2,
        // t4 - t3,
        // t4 - t0
    // );

    groups.into_values().collect()
}

// ---- Partition comparison (same as simple version) ----

fn vocab_is_refinement(finer: &VocabEquivalenceResult, coarser: &VocabEquivalenceResult) -> bool {
    let mut token_to_coarse: HashMap<usize, usize> = HashMap::new();
    for (cid, class) in coarser.iter().enumerate() {
        for &ti in class {
            token_to_coarse.insert(ti, cid);
        }
    }
    for class in finer {
        let first_coarse = token_to_coarse.get(&class[0]).copied();
        if !class
            .iter()
            .all(|&ti| token_to_coarse.get(&ti).copied() == first_coarse)
        {
            return false;
        }
    }
    true
}

pub fn partition_is_at_least_as_fine(
    a: &VocabEquivalenceResult,
    b: &VocabEquivalenceResult,
) -> bool {
    vocab_is_refinement(a, b)
}

pub fn partitions_are_comparable(
    a: &VocabEquivalenceResult,
    b: &VocabEquivalenceResult,
) -> bool {
    vocab_is_refinement(a, b) || vocab_is_refinement(b, a)
}

pub fn partitions_are_equivalent(
    a: &VocabEquivalenceResult,
    b: &VocabEquivalenceResult,
) -> bool {
    vocab_is_refinement(a, b) && vocab_is_refinement(b, a)
}
