//! Fast vocab equivalence analysis: partition tokens by DFA behavior signatures.
//!
//! For each token, computes a signature encoding its DFA behavior across all
//! initial states (match positions, suffix DAG structure, end states).
//! Tokens with identical signatures form equivalence classes.

// Do NOT add caching shortcuts that skip states/tokens. Full correctness mandatory.

use crate::dfa_u8::Tokenizer;
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
    non_greedy: bool,
}

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
    num_classes: usize,
    /// Transposed transition table: `trans_by_class[class * num_states + state]`.
    /// For a given byte class, all state transitions are contiguous in memory.
    trans_by_class: Vec<u32>,
    finalizers: Vec<SmallVec<[Finalizer; 4]>>,
    is_dead_end: Vec<bool>,
    num_groups: usize,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
    /// Per-state bitset: which bytes cause a self-loop (transition back to same state).
    self_loop_bytes: Vec<[u64; 4]>,
    /// Maps group ID → follow-class ID (from follow matrix).
    /// Groups with the same follow-class have identical rows and columns in the
    /// ever_allowed_by_group matrix. Used only for looking up which follow-class
    /// a group belongs to, NOT for replacing group IDs in hashes.
    group_to_follow_class: Option<Vec<usize>>,
    /// Per-follow-class visibility: `follow_class_visible[c1][c2]` = true iff class c2
    /// can follow class c1. Used for projected suffix hashing — when a group in class c1
    /// matches, the suffix hash only includes groups visible to c1.
    follow_class_visible: Option<Vec<Vec<bool>>>,
    /// Number of follow classes (= max class ID + 1 when group_to_follow_class is set).
    num_follow_classes: usize,
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

    /// Map a group ID to its follow-class ID (if follow data is present).
    /// Returns the raw group ID if no follow data.
    #[inline]
    fn follow_class_of(&self, gid: usize) -> usize {
        match &self.group_to_follow_class {
            Some(map) if gid < map.len() => map[gid],
            _ => gid,
        }
    }

    /// Look up transition: given a DFA state and a byte, return the next state (u32).
    #[inline]
    fn transition(&self, state: usize, byte: u8) -> u32 {
        let class = self.byte_to_class[byte as usize] as usize;
        unsafe { *self.trans_by_class.get_unchecked(class * self.num_states + state) }
    }
}

/// Combined scratch space for batch DFA execution and suffix DAG construction.
struct Scratch {
    // Batch execution across initial states
    current_states: Vec<usize>,
    active_indices: Vec<usize>,
    match_positions: Vec<u32>,
    /// Per-state list of group IDs touched during the last run_batch call.
    /// Used to avoid O(num_states * num_groups) memset and scan.
    dirty_groups: Vec<SmallVec<[usize; 16]>>,
    targets: Vec<usize>,
    // Suffix DAG
    dag: HashMap<usize, (u64, EdgeList)>,
    dag_queue: Vec<usize>,
    /// Per-position per-follow-class hashes for projected suffix hashing.
    /// Indexed by position → SmallVec of hashes (one per follow class).
    dag_projected: HashMap<usize, SmallVec<[u64; 16]>>,
}

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

fn build_dfa(regex: &Tokenizer) -> Dfa {
    build_dfa_with_class_map(regex, None, None)
}

fn build_dfa_with_class_map(
    regex: &Tokenizer,
    group_to_class: Option<&[usize]>,
    ever_allowed_by_group: Option<&[Vec<bool>]>,
) -> Dfa {
    let dfa = regex.dfa();
    assert!(dfa.states.len() <= u32::MAX as usize, "DFA too large");

    // Compute num_groups from all group IDs referenced in the DFA
    let num_groups = dfa
        .states
        .iter()
        .flat_map(|s| {
            s.finalizers
                .iter()
                .chain(s.possible_future_group_ids.iter().copied())
        })
        .chain(dfa.non_greedy_finalizers.iter().copied())
        .max()
        .map_or(0, |m| m + 1);

    let mut non_greedy_flags = vec![false; num_groups];
    for &gid in &dfa.non_greedy_finalizers {
        if gid < num_groups {
            non_greedy_flags[gid] = true;
        }
    }

    let mut transitions = Vec::with_capacity(dfa.states.len());
    let mut finalizers = Vec::with_capacity(dfa.states.len());
    let mut is_dead_end = Vec::with_capacity(dfa.states.len());
    let mut completion_hash = Vec::with_capacity(dfa.states.len());

    for state in &dfa.states {
        let mut table = [NONE; 256];
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
                    non_greedy: non_greedy_flags.get(gid).copied().unwrap_or(false),
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

    // Precompute self-loop byte sets per state
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

    // Compute byte equivalence classes: group bytes with identical transitions across all states.
    let num_dfa_states = transitions.len();
    let mut byte_to_class = [0u8; 256];
    let mut class_repr = [0u8; 256]; // representative byte for each class
    let mut num_classes = 0usize;

    for b in 0..=255u8 {
        let mut found = false;
        for c in 0..num_classes {
            let repr = class_repr[c] as usize;
            let same = (0..num_dfa_states).all(|s| transitions[s][b as usize] == transitions[s][repr]);
            if same {
                byte_to_class[b as usize] = c as u8;
                found = true;
                break;
            }
        }
        if !found {
            byte_to_class[b as usize] = num_classes as u8;
            class_repr[num_classes] = b;
            num_classes += 1;
        }
    }

    // Build transposed transition table: trans_by_class[class * num_states + state]
    let mut trans_by_class = vec![NONE; num_classes * num_dfa_states];
    for c in 0..num_classes {
        let repr = class_repr[c] as usize;
        let base = c * num_dfa_states;
        for s in 0..num_dfa_states {
            trans_by_class[base + s] = transitions[s][repr];
        }
    }

    crate::debug!(2, "  DFA byte classes: {} classes from 256 bytes ({} states, table={:.1}KB)",
        num_classes, num_dfa_states,
        (num_classes * num_dfa_states * 4) as f64 / 1024.0);

    Dfa {
        start_state: dfa.start_state,
        num_states: num_dfa_states,
        byte_to_class,
        num_classes,
        trans_by_class,
        finalizers,
        is_dead_end,
        num_groups,
        completion_hash,
        none_completion_hash,
        self_loop_bytes,
        group_to_follow_class: group_to_class.map(|s| s.to_vec()),
        follow_class_visible: Dfa::build_follow_class_visible(group_to_class, ever_allowed_by_group),
        num_follow_classes: group_to_class.map_or(0, |g2c| g2c.iter().copied().max().map_or(0, |m| m + 1)),
    }
}

impl Dfa {
    /// Build per-follow-class visibility matrix from group_to_class and ever_allowed_by_group.
    /// `follow_class_visible[c1][c2]` = true iff class c2 can follow class c1.
    fn build_follow_class_visible(
        group_to_class: Option<&[usize]>,
        ever_allowed_by_group: Option<&[Vec<bool>]>,
    ) -> Option<Vec<Vec<bool>>> {
        let g2c = group_to_class?;
        let eabg = ever_allowed_by_group?;
        let num_classes = g2c.iter().copied().max().map_or(0, |m| m + 1);
        if num_classes == 0 {
            return None;
        }
        let num_groups = g2c.len();
        // For each class, find a representative group
        let mut class_rep: Vec<Option<usize>> = vec![None; num_classes];
        for (gid, &cid) in g2c.iter().enumerate() {
            if class_rep[cid].is_none() {
                class_rep[cid] = Some(gid);
            }
        }
        // Build class-level visibility: class c2 is visible from class c1
        // iff the representative of c1 allows the representative of c2.
        // (All groups in a class have the same follow row and column.)
        let mut visible: Vec<Vec<bool>> = vec![vec![false; num_classes]; num_classes];
        for c1 in 0..num_classes {
            if let Some(rep1) = class_rep[c1] {
                if rep1 < eabg.len() {
                    let row = &eabg[rep1];
                    for c2 in 0..num_classes {
                        if let Some(rep2) = class_rep[c2] {
                            if rep2 < row.len() && row[rep2] {
                                visible[c1][c2] = true;
                            }
                        }
                    }
                }
            }
        }
        crate::debug!(2, "  Follow class visibility: {} classes, density {:.1}%",
            num_classes,
            100.0 * visible.iter().flat_map(|r| r.iter()).filter(|&&b| b).count() as f64
                / (num_classes * num_classes) as f64);
        Some(visible)
    }
}

impl Scratch {
    fn new(num_states: usize, num_groups: usize) -> Self {
        Scratch {
            current_states: vec![0; num_states],
            active_indices: Vec::new(),
            match_positions: vec![NONE; num_states * num_groups],
            dirty_groups: vec![SmallVec::new(); num_states],
            targets: Vec::new(),
            dag: HashMap::new(),
            dag_queue: Vec::new(),
            dag_projected: HashMap::new(),
        }
    }
}

/// Run DFA from all initial states on a token, recording end states and match positions.
/// Uses dirty_groups tracking to avoid O(num_states * num_groups) memset.
/// INVARIANT: match_positions entries are NONE except for dirty entries from a previous
/// call that must have been cleaned up by the caller (token_signature does this).
fn run_batch(
    dfa: &Dfa,
    scratch: &mut Scratch,
    slice: &[u8],
    initial_states: &[usize],
) {
    let num_states = initial_states.len();
    let num_groups = dfa.num_groups;
    let len = slice.len();

    // Reset scratch — DON'T fill match_positions (maintained by dirty cleanup)
    scratch.current_states[..num_states].clone_from_slice(initial_states);
    scratch.active_indices.clear();
    for dg in scratch.dirty_groups[..num_states].iter_mut() {
        dg.clear();
    }

    let has_bytes = !slice.is_empty();
    let first_byte = if has_bytes { slice[0] } else { 0 };

    // Process initial finalizers
    for (i, &state) in initial_states.iter().enumerate() {
        let base = i * num_groups;
        for f in &dfa.finalizers[state] {
            if f.gid < num_groups && scratch.match_positions[base + f.gid] == NONE {
                scratch.match_positions[base + f.gid] = 0;
                scratch.dirty_groups[i].push(f.gid);
            }
        }
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
                    for f in &dfa.finalizers[ns] {
                        if f.gid < num_groups {
                            let ix = base + f.gid;
                            if !f.non_greedy || scratch.match_positions[ix] == NONE {
                                scratch.match_positions[ix] = position;
                            }
                            scratch.dirty_groups[i].push(f.gid);
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
            if pos + 1 < len {
                // Intersect self_loop_bytes for all active states
                let mut sl = [!0u64; 4];
                for idx in 0..active_len {
                    let i = scratch.active_indices[idx];
                    let s = scratch.current_states[i];
                    for k in 0..4 {
                        sl[k] &= dfa.self_loop_bytes[s][k];
                    }
                }
                // Check if all remaining bytes are in the intersection
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
                            if f.gid < num_groups && !f.non_greedy {
                                scratch.match_positions[base + f.gid] = token_len;
                                scratch.dirty_groups[i].push(f.gid);
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    // Collect unique match target positions — only from dirty groups
    scratch.targets.clear();
    if num_groups > 0 {
        for si in 0..num_states {
            let base = si * num_groups;
            for &gid in &scratch.dirty_groups[si] {
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

/// Run DFA on a suffix from start_state, returning (end_state, edges to match positions).
fn run_suffix(
    dfa: &Dfa,
    slice: &[u8],
    base_pos: usize,
    match_positions: &mut [u32],
) -> (Option<usize>, EdgeList) {
    let num_groups = dfa.num_groups;
    match_positions[..num_groups].fill(NONE);
    let mut current = dfa.start_state;
    let mut done = dfa.is_dead_end[current];

    for f in &dfa.finalizers[current] {
        if f.gid < num_groups && match_positions[f.gid] == NONE {
            match_positions[f.gid] = 0;
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
        for f in &dfa.finalizers[current] {
            if f.gid < num_groups {
                if !f.non_greedy || match_positions[f.gid] == NONE {
                    match_positions[f.gid] = position;
                }
            }
        }
        if dfa.is_dead_end[current] {
            done = true;
        }
    }

    let end_state = if done { None } else { Some(current) };
    let edges: EdgeList = (0..num_groups)
        .filter_map(|gid| {
            let pv = match_positions[gid];
            (pv != NONE && pv != 0).then(|| (gid, base_pos + pv as usize))
        })
        .collect();
    (end_state, edges)
}

/// Build suffix DAG via BFS from match target positions and hash bottom-up.
fn hash_suffixes(
    dfa: &Dfa,
    slice: &[u8],
    scratch: &mut Scratch,
) {
    let len = slice.len();
    let mut suffix_mp = vec![NONE; dfa.num_groups];
    scratch.dag.clear();
    scratch.dag_queue.clear();

    // BFS from target positions: run suffix DFA at each, discover new positions from edges
    for &pos in &scratch.targets {
        if pos <= len && !scratch.dag.contains_key(&pos) {
            scratch.dag_queue.push(pos);
            scratch.dag.insert(pos, (0, EdgeList::new())); // placeholder
        }
    }

    let mut cursor = 0;
    while cursor < scratch.dag_queue.len() {
        let pos = scratch.dag_queue[cursor];
        cursor += 1;
        let (end_state, edges) =
            run_suffix(dfa, &slice[pos..], pos, &mut suffix_mp);
        for &(_, target) in &edges {
            if target <= len && !scratch.dag.contains_key(&target) {
                scratch.dag_queue.push(target);
                scratch.dag.insert(target, (0, EdgeList::new()));
            }
        }
        let ch = dfa.completion(end_state.unwrap_or(STATE_NONE));
        scratch.dag.insert(pos, (ch, edges));
    }

    // Hash bottom-up: process deeper positions first
    scratch.dag_queue.sort_unstable_by(|a, b| b.cmp(a));

    if let Some(ref fcv) = dfa.follow_class_visible {
        // Projected suffix hashing: compute per-follow-class hashes at each DAG node.
        //
        // hash(P, C) = combine(completion, [(gid, hash(target, follow_class_of(gid)))
        //                                   for (gid, target) in edges(P)
        //                                   if visible(C, follow_class_of(gid))])
        //
        // Raw group IDs are preserved in the hash (not mapped to classes).
        // Only the FILTERING of edges uses follow-class visibility, and the
        // RECURSIVE class selection uses follow_class_of(edge_gid).
        let nc = dfa.num_follow_classes;
        scratch.dag_projected.clear();

        for idx in 0..scratch.dag_queue.len() {
            let pos = scratch.dag_queue[idx];
            let (ch, ref edges) = scratch.dag[&pos];

            let mut class_hashes: SmallVec<[u64; 16]> = SmallVec::with_capacity(nc);
            for c in 0..nc {
                let vis = &fcv[c];
                let mut h = new_hasher();
                h.write_u64(ch);
                for &(gid, target) in edges {
                    let gid_class = dfa.follow_class_of(gid);
                    if gid_class < vis.len() && vis[gid_class] {
                        h.write_u64(gid as u64);  // RAW group ID
                        // Recurse: project the target by the edge group's follow-class
                        let th = scratch.dag_projected.get(&target)
                            .map_or(0, |hashes| if gid_class < hashes.len() { hashes[gid_class] } else { 0 });
                        h.write_u64(th);
                    }
                }
                class_hashes.push(h.finish());
            }
            scratch.dag_projected.insert(pos, class_hashes);
        }
    } else {
        // No follow data: hash with raw group IDs, all edges included
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
}

/// Compute a token's full signature over a batch of initial states.
/// Also cleans up match_positions for dirty groups (maintaining the NONE invariant).
fn token_signature(
    dfa: &Dfa,
    token: &[u8],
    chunk_states: &[usize],
    scratch: &mut Scratch,
) -> u64 {
    run_batch(dfa, scratch, token, chunk_states);
    if !scratch.targets.is_empty() {
        hash_suffixes(dfa, token, scratch);
    }

    let num_groups = dfa.num_groups;
    let mut sig: u64 = HASH_SEED3;
    for i in 0..chunk_states.len() {
        let completion = dfa.completion(scratch.current_states[i]);
        let base = i * num_groups;

        // Sort and dedup dirty groups for deterministic hashing
        let dirty = &mut scratch.dirty_groups[i];
        dirty.sort_unstable();
        dirty.dedup();

        let state_sig = if !dirty.is_empty() {
            // Check if any dirty group has a non-zero match position
            let has_match = dirty.iter().any(|&gid| {
                let pv = scratch.match_positions[base + gid];
                pv != NONE && pv > 0
            });
            if has_match {
                let mut h = new_hasher();
                h.write_u64(completion);
                if dfa.follow_class_visible.is_some() {
                    // Projected suffix hashing: use per-follow-class suffix hash.
                    // For group G matching at suffix position PV, the suffix hash
                    // is projected by follow_class_of(G)'s visibility.
                    // Raw group IDs are used in the hash (not mapped to classes).
                    for &gid in dirty.iter() {
                        let pv = scratch.match_positions[base + gid];
                        if pv != NONE && pv > 0 {
                            let fc = dfa.follow_class_of(gid);
                            let dh = scratch.dag_projected.get(&(pv as usize))
                                .map_or(0, |hashes| if fc < hashes.len() { hashes[fc] } else { 0 });
                            h.write_u64(gid as u64);  // RAW group ID
                            h.write_u64(dh);
                        }
                    }
                } else {
                    for &gid in dirty.iter() {
                        let pv = scratch.match_positions[base + gid];
                        if pv != NONE && pv > 0 {
                            h.write_u64(gid as u64);
                            h.write_u64(scratch.dag.get(&(pv as usize)).map_or(0, |e| e.0));
                        }
                    }
                }
                h.finish()
            } else {
                // Groups matched at position 0 only — still use hasher for distinction
                let mut h = new_hasher();
                h.write_u64(completion);
                h.finish()
            }
        } else {
            completion
        };

        // Clean up: reset match_positions for dirty groups back to NONE
        for &gid in dirty.iter() {
            scratch.match_positions[base + gid] = NONE;
        }

        sig = sig.wrapping_mul(HASH_SEED1).wrapping_add(state_sig);
    }
    sig
}

pub fn find_vocab_equivalence_classes<S: AsRef<[u8]> + Sync>(
    regex: &Tokenizer,
    strings: &[S],
    initial_states: &[usize],
) -> VocabEquivalenceResult {
    find_vocab_equivalence_classes_with_follow(regex, strings, initial_states, None, None, None)
}

/// Find vocab equivalence classes, optionally merging groups that belong to
/// the same grammar‐equivalence class (via `group_to_class`).
pub fn find_vocab_equivalence_classes_with_follow<S: AsRef<[u8]> + Sync>(
    regex: &Tokenizer,
    strings: &[S],
    initial_states: &[usize],
    _suffix_group_mask: Option<&[bool]>,
    ever_allowed_by_group: Option<&[Vec<bool>]>,
    group_to_class: Option<&[usize]>,
) -> VocabEquivalenceResult {
    let t0 = std::time::Instant::now();
    let dfa = build_dfa_with_class_map(regex, group_to_class, ever_allowed_by_group);
    let t1 = std::time::Instant::now();
    crate::debug!(2, "  Projected suffix hashing: follow_class_visible={}, group_to_follow_class={}, num_follow_classes={}",
        dfa.follow_class_visible.is_some(), dfa.group_to_follow_class.is_some(), dfa.num_follow_classes);
    let num_tokens = strings.len();
    let num_states = initial_states.len();

    if num_states == 0 || num_tokens == 0 {
        return BTreeSet::from_iter(vec![(0..num_tokens).collect()]);
    }

    let num_groups = dfa.num_groups;
    // Use large batches now that dirty_groups tracking avoids O(batch_size * ng) memset.
    // Memory per thread: batch_size * ng * 4 bytes for match_positions (allocated once, sparse use).
    let batch_size = num_states.min(5000);
    let mut active_indices: Vec<usize> = (0..num_tokens).collect();
    let mut partition = vec![0usize; num_tokens];
    let mut next_class_id = 1usize;

    for batch_start in (0..num_states).step_by(batch_size) {
        if active_indices.is_empty() {
            break;
        }
        let batch_end = (batch_start + batch_size).min(num_states);
        let batch = &initial_states[batch_start..batch_end];

        let t_par0 = std::time::Instant::now();

        let active_sigs: Vec<(usize, u64)> = active_indices
            .par_iter()
            .map_init(
                || Scratch::new(batch.len(), num_groups),
                |scratch, &token_idx| {
                    let token = strings[token_idx].as_ref();
                    (token_idx, token_signature(&dfa, token, batch, scratch))
                },
            )
            .collect();

        let t_par1 = std::time::Instant::now();

        // Refine partition: group tokens by (old_class, signature)
        let mut refinement: HashMap<(usize, u64), Vec<usize>> =
            HashMap::with_capacity(active_sigs.len() / 2);
        for (ti, sig) in active_sigs {
            refinement
                .entry((partition[ti], sig))
                .or_default()
                .push(ti);
        }

        // Assign class IDs: first sub-group of each old class keeps the old ID
        let mut new_active = Vec::with_capacity(active_indices.len());
        let mut seen_classes: HashMap<usize, ()> = HashMap::new();
        for ((old_class, _), tokens) in refinement {
            let class_id = if seen_classes.insert(old_class, ()).is_none() {
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
        crate::debug!(2, "  batch {}: par_iter={:?}, refine={:?}",
            batch_start, t_par1 - t_par0, std::time::Instant::now() - t_par1);
    }

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::with_capacity(next_class_id);
    for (ti, &cid) in partition.iter().enumerate() {
        groups.entry(cid).or_default().push(ti);
    }
    let t2 = std::time::Instant::now();
    crate::debug!(2, "Vocab equiv FAST: dfa={:?}, par_compute={:?}, total={:?}",
        t1 - t0, t2 - t1, t2 - t0);
    groups.into_values().collect()
}
