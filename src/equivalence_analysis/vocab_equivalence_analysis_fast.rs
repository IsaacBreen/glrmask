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
type GroupList = SmallVec<[usize; 4]>;

const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;
const NONE: u32 = u32::MAX;

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
    has_transitions: Vec<bool>,
    num_groups: usize,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
}

/// Combined scratch space for batch DFA execution and suffix DAG construction.
struct Scratch {
    // Batch execution across initial states
    current_states: Vec<usize>,
    done: Vec<bool>,
    active_indices: Vec<usize>,
    end_states: Vec<Option<usize>>,
    match_positions: Vec<u32>,
    touched_groups: Vec<GroupList>,
    targets: Vec<usize>,
    // Suffix DAG
    suffix_match_positions: Vec<u32>,
    visited: Vec<bool>,
    queue: Vec<usize>,
    order: Vec<usize>,
    nodes: Vec<Option<(u64, EdgeList)>>,
    cache: Vec<Option<u64>>,
}

static HASH_RANDOM_STATE: Lazy<RandomState> =
    Lazy::new(|| RandomState::with_seeds(HASH_SEED1, HASH_SEED2, HASH_SEED3, HASH_SEED4));

#[inline]
fn new_hasher() -> AHasher {
    HASH_RANDOM_STATE.build_hasher()
}

#[inline]
fn hash_group_list(list: &[usize]) -> u64 {
    let mut h = new_hasher();
    h.write_u8(1);
    h.write_u64(list.len() as u64);
    for &v in list {
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
    let mut has_transitions = Vec::with_capacity(dfa.states.len());
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
        has_transitions.push(!state.transitions.is_empty());
        completion_hash.push(hash_group_list(
            &state
                .possible_future_group_ids
                .iter()
                .copied()
                .collect::<Vec<_>>(),
        ));
    }

    let none_completion_hash = {
        let mut h = new_hasher();
        h.write_u8(0);
        h.finish()
    };

    Dfa {
        start_state: dfa.start_state,
        transitions,
        finalizers,
        is_dead_end,
        has_transitions,
        num_groups,
        completion_hash,
        none_completion_hash,
    }
}

impl Scratch {
    fn new(num_states: usize, num_groups: usize) -> Self {
        Scratch {
            current_states: vec![0; num_states],
            done: vec![false; num_states],
            active_indices: Vec::new(),
            end_states: vec![None; num_states],
            match_positions: vec![NONE; num_states * num_groups],
            touched_groups: vec![GroupList::new(); num_states],
            targets: Vec::new(),
            suffix_match_positions: vec![NONE; num_groups],
            visited: Vec::new(),
            queue: Vec::new(),
            order: Vec::new(),
            nodes: Vec::new(),
            cache: vec![None; 256],
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
    scratch.done.fill(false);
    scratch.active_indices.clear();
    scratch.end_states[..num_states].fill(None);
    scratch.match_positions[..num_states * num_groups].fill(NONE);
    for tg in scratch.touched_groups[..num_states].iter_mut() {
        tg.clear();
    }

    let has_bytes = !slice.is_empty();
    let first_byte = if has_bytes { slice[0] } else { 0 };

    // Process initial finalizers
    for (i, &state) in initial_states.iter().enumerate() {
        let base = i * num_groups;
        for f in &dfa.finalizers[state] {
            if f.gid < num_groups && scratch.match_positions[base + f.gid] == NONE {
                scratch.match_positions[base + f.gid] = 0;
                scratch.touched_groups[i].push(f.gid);
            }
        }
        if !dfa.has_transitions[state] {
            scratch.done[i] = true;
            continue;
        }
        if has_bytes && dfa.transitions[state][first_byte as usize] == NONE {
            scratch.done[i] = true;
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
                            let was_none = scratch.match_positions[ix] == NONE;
                            if !f.non_greedy || was_none {
                                scratch.match_positions[ix] = position;
                            }
                            if was_none {
                                scratch.touched_groups[i].push(f.gid);
                            }
                        }
                    }
                    if dfa.is_dead_end[ns] {
                        scratch.done[i] = true;
                    }
                } else {
                    scratch.done[i] = true;
                }
                if !scratch.done[i] {
                    scratch.active_indices[next_len] = i;
                    next_len += 1;
                }
            }
            active_len = next_len;
            if active_len == 0 {
                break;
            }
        }
    }

    // Collect end states and unique match target positions
    scratch.targets.clear();
    for i in 0..num_states {
        scratch.end_states[i] = if scratch.done[i]
            || !dfa.has_transitions[scratch.current_states[i]]
        {
            None
        } else {
            Some(scratch.current_states[i])
        };
        if num_groups > 0 {
            let base = i * num_groups;
            for &gid in &scratch.touched_groups[i] {
                let pv = scratch.match_positions[base + gid];
                if pv > 0 {
                    let p = pv as usize;
                    if p <= len {
                        scratch.targets.push(p);
                    }
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
    let mut touched = GroupList::new();
    let mut current = dfa.start_state;
    let mut done = dfa.is_dead_end[current];

    for f in &dfa.finalizers[current] {
        if f.gid < num_groups && match_positions[f.gid] == NONE {
            match_positions[f.gid] = 0;
            touched.push(f.gid);
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
                let was_none = match_positions[f.gid] == NONE;
                if !f.non_greedy || was_none {
                    match_positions[f.gid] = position;
                }
                if was_none {
                    touched.push(f.gid);
                }
            }
        }
        if dfa.is_dead_end[current] {
            done = true;
        }
    }

    let end_state = if done || !dfa.has_transitions[current] {
        None
    } else {
        Some(current)
    };
    touched.sort_unstable();
    let edges: EdgeList = touched
        .iter()
        .filter_map(|&gid| {
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

    // Reset from previous call
    for &pos in &scratch.queue {
        if pos < scratch.visited.len() {
            scratch.visited[pos] = false;
        }
        if pos < scratch.nodes.len() {
            scratch.nodes[pos] = None;
        }
    }
    scratch.queue.clear();
    scratch.order.clear();
    let needed = len + 1;
    if scratch.visited.len() < needed {
        scratch.visited.resize(needed, false);
    }
    if scratch.nodes.len() < needed {
        scratch.nodes.resize(needed, None);
    }

    // Seed BFS with target positions
    for &pos in &scratch.targets {
        if pos <= len && scratch.nodes[pos].is_none() && !scratch.visited[pos] {
            scratch.visited[pos] = true;
            scratch.queue.push(pos);
        }
    }
    if scratch.queue.is_empty() {
        return;
    }

    // BFS: discover all reachable suffix positions
    let mut cursor = 0;
    while cursor < scratch.queue.len() {
        let pos = scratch.queue[cursor];
        cursor += 1;
        let (end_state, edges) =
            run_suffix(dfa, &slice[pos..], pos, &mut scratch.suffix_match_positions);
        for &(_, target) in &edges {
            if target <= len && scratch.nodes[target].is_none() && !scratch.visited[target] {
                scratch.visited[target] = true;
                scratch.queue.push(target);
            }
        }
        let ch = end_state
            .map(|id| dfa.completion_hash[id])
            .unwrap_or(dfa.none_completion_hash);
        scratch.nodes[pos] = Some((ch, edges));
        scratch.order.push(pos);
    }

    // Hash bottom-up (deeper positions first)
    scratch.order.sort_unstable_by(|a, b| b.cmp(a));
    for &pos in &scratch.order {
        if scratch.cache[pos].is_some() {
            continue;
        }
        if let Some((ch, ref edges)) = scratch.nodes[pos] {
            let mut h = new_hasher();
            h.write_u64(ch);
            for &(gid, target) in edges.iter() {
                h.write_u64(gid as u64);
                h.write_u64(scratch.cache[target].unwrap_or(0));
            }
            scratch.cache[pos] = Some(h.finish());
        }
    }
    scratch.order.clear();
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
        let completion = scratch.end_states[i]
            .map(|id| dfa.completion_hash[id])
            .unwrap_or(dfa.none_completion_hash);

        let state_sig = if dfa.num_groups > 0 && !scratch.touched_groups[i].is_empty() {
            let groups = &mut scratch.touched_groups[i];
            if groups.len() > 1 {
                groups.sort_unstable();
            }
            let base = i * dfa.num_groups;
            let mut h = new_hasher();
            h.write_u64(completion);
            for &gid in groups.iter() {
                let pv = scratch.match_positions[base + gid];
                if pv > 0 {
                    h.write_u64(gid as u64);
                    h.write_u64(scratch.cache[pv as usize].unwrap_or(0));
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
    let dfa = build_dfa(regex);
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

        let active_sigs: Vec<(usize, u64)> = active_indices
            .par_iter()
            .map_init(
                || Scratch::new(batch.len(), num_groups),
                |scratch, &token_idx| {
                    let token = strings[token_idx].as_ref();
                    if scratch.cache.len() <= token.len() {
                        scratch.cache.resize(token.len() + 1, None);
                    }
                    scratch.cache.iter_mut().for_each(|x| *x = None);
                    (token_idx, token_signature(&dfa, token, batch, scratch))
                },
            )
            .collect();

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
        let mut first_of_class: HashMap<usize, bool> = HashMap::new();
        for ((old_class, _), tokens) in refinement {
            let class_id = {
                let seen = first_of_class.entry(old_class).or_insert(false);
                if *seen {
                    let id = next_class_id;
                    next_class_id += 1;
                    id
                } else {
                    *seen = true;
                    old_class
                }
            };
            for &ti in &tokens {
                partition[ti] = class_id;
            }
            if tokens.len() > 1 {
                new_active.extend(tokens);
            }
        }
        active_indices = new_active;
    }

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::with_capacity(next_class_id);
    for (ti, &cid) in partition.iter().enumerate() {
        groups.entry(cid).or_default().push(ti);
    }
    groups.into_values().collect()
}
