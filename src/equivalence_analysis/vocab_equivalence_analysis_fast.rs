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

/// Flat DFA with 256-byte transition tables.
struct Dfa {
    start_state: usize,
    transitions: Vec<[u32; 256]>,
    finalizers: Vec<SmallVec<[Finalizer; 4]>>,
    is_dead_end: Vec<bool>,
    num_groups: usize,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
    /// Per-state bitset: which bytes cause a self-loop (transition back to same state).
    self_loop_bytes: Vec<[u64; 4]>,
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
}

/// Combined scratch space for batch DFA execution and suffix DAG construction.
struct Scratch {
    // Batch execution across initial states
    current_states: Vec<usize>,
    active_indices: Vec<usize>,
    match_positions: Vec<u32>,
    targets: Vec<usize>,
    // Suffix DAG
    dag: HashMap<usize, (u64, EdgeList)>,
    dag_queue: Vec<usize>,
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

    Dfa {
        start_state: dfa.start_state,
        transitions,
        finalizers,
        is_dead_end,
        num_groups,
        completion_hash,
        none_completion_hash,
        self_loop_bytes,
    }
}

impl Scratch {
    fn new(num_states: usize, num_groups: usize) -> Self {
        Scratch {
            current_states: vec![0; num_states],
            active_indices: Vec::new(),
            match_positions: vec![NONE; num_states * num_groups],
            targets: Vec::new(),
            dag: HashMap::new(),
            dag_queue: Vec::new(),
        }
    }
}

/// Run DFA from all initial states on a token, recording end states and match positions.
fn run_batch(
    dfa: &Dfa,
    scratch: &mut Scratch,
    slice: &[u8],
    initial_states: &[usize],
) {
    let num_states = initial_states.len();
    let num_groups = dfa.num_groups;
    let len = slice.len();

    // Reset scratch
    scratch.current_states[..num_states].clone_from_slice(initial_states);
    scratch.active_indices.clear();
    scratch.match_positions[..num_states * num_groups].fill(NONE);

    let has_bytes = !slice.is_empty();
    let first_byte = if has_bytes { slice[0] } else { 0 };

    // Process initial finalizers
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

    // Walk each byte (hot path)
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
                            if !f.non_greedy || scratch.match_positions[ix] == NONE {
                                scratch.match_positions[ix] = position;
                            }
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
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    // Collect unique match target positions
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
        let ns = dfa.transitions[current][byte as usize];
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

/// Compute a token's full signature over a batch of initial states.
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

    let mut sig: u64 = HASH_SEED3;
    for i in 0..chunk_states.len() {
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

pub fn find_vocab_equivalence_classes<S: AsRef<[u8]> + Sync>(
    regex: &Tokenizer,
    strings: &[S],
    initial_states: &[usize],
) -> VocabEquivalenceResult {
    find_vocab_equivalence_classes_with_follow(regex, strings, initial_states, None, None, None)
}

/// Find vocab equivalence classes. The last three parameters are accepted for
/// API compatibility but unused internally.
pub fn find_vocab_equivalence_classes_with_follow<S: AsRef<[u8]> + Sync>(
    regex: &Tokenizer,
    strings: &[S],
    initial_states: &[usize],
    _suffix_group_mask: Option<&[bool]>,
    _ever_allowed_by_group: Option<&[Vec<bool>]>,
    _group_to_class: Option<&[usize]>,
) -> VocabEquivalenceResult {
    let t0 = std::time::Instant::now();
    let dfa = build_dfa(regex);
    let t1 = std::time::Instant::now();
    let num_tokens = strings.len();
    let num_states = initial_states.len();

    if num_states == 0 || num_tokens == 0 {
        return BTreeSet::from_iter(vec![(0..num_tokens).collect()]);
    }

    let num_groups = dfa.num_groups;
    let batch_size = num_states.min(200);
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
