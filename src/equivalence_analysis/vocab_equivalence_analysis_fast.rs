//! Fast vocabulary equivalence analysis via iterative DFA signature refinement.
//!
//! Computes which vocabulary tokens produce identical parsing behavior across
//! all initial tokenizer states, using batched parallel signature computation.

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
type FinalizerList = SmallVec<[Finalizer; 4]>;
const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;
const NONE_STATE: u32 = u32::MAX;
const NONE_POS: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Finalizer { gid: usize, non_greedy: bool }
#[derive(Clone, Copy, PartialEq)]
enum FutureMode { Terminate, Continue }

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

struct Pos0Scratch {
    current_states: Vec<usize>,
    done: Vec<bool>,
    active_indices: Vec<usize>,
    end_states: Vec<Option<usize>>,
    match_positions: Vec<u32>,
    match_gen: Vec<u32>,
    cur_gen: u32,
    touched_groups: Vec<GroupList>,
    touched_states: Vec<usize>,
    base_offsets: Vec<usize>,
    seen_target: Vec<bool>,
    all_targets: Vec<usize>,
}

struct SuffixScratch {
    match_positions: Vec<u32>,
    visited: Vec<bool>,
    queue: Vec<usize>,
    order: Vec<usize>,
    nodes: Vec<Option<(u64, EdgeList)>>,
}

static HASH_RANDOM_STATE: Lazy<RandomState> =
    Lazy::new(|| RandomState::with_seeds(HASH_SEED1, HASH_SEED2, HASH_SEED3, HASH_SEED4));
#[inline]
fn new_hasher() -> AHasher { HASH_RANDOM_STATE.build_hasher() }
#[inline]
fn hash_group_list(list: &[usize]) -> u64 {
    let mut h = new_hasher();
    h.write_u8(1);
    h.write_u64(list.len() as u64);
    for &v in list { h.write_u64(v as u64); }
    h.finish()
}

fn precompute_dfa(regex: &Tokenizer) -> PrecomputedDfa {
    let dfa = regex.dfa();
    assert!(dfa.states.len() <= u32::MAX as usize, "DFA too large");
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
    let mut transitions = Vec::with_capacity(dfa.states.len());
    let mut finalizers: Vec<FinalizerList> = Vec::with_capacity(dfa.states.len());
    let mut possible_future: Vec<GroupList> = Vec::with_capacity(dfa.states.len());
    let mut has_transitions = Vec::with_capacity(dfa.states.len());
    for state in &dfa.states {
        let mut table = [NONE_STATE; 256];
        for (byte, &target) in state.transitions.iter() {
            table[byte as usize] = target as u32;
        }
        transitions.push(table);
        finalizers.push(state.finalizers.iter()
            .map(|gid| Finalizer { gid, non_greedy: false }).collect());
        possible_future.push(state.possible_future_group_ids.iter().copied().collect());
        has_transitions.push(!state.transitions.is_empty());
    }
    let mut non_greedy_flags = vec![false; num_groups];
    for &gid in &dfa.non_greedy_finalizers {
        if gid < num_groups { non_greedy_flags[gid] = true; }
    }
    for finals in &mut finalizers {
        for f in finals.iter_mut() {
            f.non_greedy = non_greedy_flags.get(f.gid).copied().unwrap_or(false);
        }
    }
    let future_modes: Vec<FutureMode> = possible_future.iter()
        .map(|f| if f.is_empty() { FutureMode::Terminate } else { FutureMode::Continue })
        .collect();
    let none_completion_hash = { let mut h = new_hasher(); h.write_u8(0); h.finish() };
    let completion_hash: Vec<u64> = possible_future.iter().map(|v| hash_group_list(v)).collect();
    PrecomputedDfa {
        start_state: dfa.start_state, transitions, finalizers, future_modes,
        has_transitions, num_groups, completion_hash, none_completion_hash,
    }
}

impl Pos0Scratch {
    fn new(num_states: usize, num_groups: usize) -> Self {
        Pos0Scratch {
            current_states: vec![0; num_states], done: vec![false; num_states],
            active_indices: Vec::new(), end_states: vec![None; num_states],
            match_positions: vec![0u32; num_states.saturating_mul(num_groups)],
            match_gen: vec![0u32; num_states.saturating_mul(num_groups)], cur_gen: 1,
            touched_groups: vec![GroupList::new(); num_states], touched_states: Vec::new(),
            base_offsets: (0..num_states).map(|i| i.saturating_mul(num_groups)).collect(),
            seen_target: Vec::new(), all_targets: Vec::new(),
        }
    }
    fn reset(&mut self, initial_states: &[usize], num_groups: usize) {
        let len = initial_states.len();
        if len > self.current_states.len() {
            self.current_states.resize(len, 0);
            self.done.resize(len, false);
            self.end_states.resize(len, None);
            self.match_positions.resize(len.saturating_mul(num_groups), 0);
            self.match_gen.resize(len.saturating_mul(num_groups), 0);
            self.touched_groups.resize(len, GroupList::new());
            self.base_offsets = (0..len).map(|i| i * num_groups).collect();
        }
        self.current_states[..len].clone_from_slice(initial_states);
        self.done.fill(false);
        self.active_indices.clear();
        self.end_states[..len].fill(None);
        self.cur_gen = self.cur_gen.wrapping_add(1);
        if self.cur_gen == 0 { self.match_gen.fill(0); self.cur_gen = 1; }
        for &si in &self.touched_states {
            if si < self.touched_groups.len() { self.touched_groups[si].clear(); }
        }
        self.touched_states.clear();
    }
}

/// Run DFA from all initial states on a token. Results stored in scratch.
fn compute_pos0(
    pre: &PrecomputedDfa, scratch: &mut Pos0Scratch, slice: &[u8], initial_states: &[usize],
) {
    let (num_states, num_groups, len) = (initial_states.len(), pre.num_groups, slice.len());
    scratch.reset(initial_states, num_groups);
    let all_targets = &mut scratch.all_targets;
    let seen_target = &mut scratch.seen_target;
    for &pos in all_targets.iter() { if pos < seen_target.len() { seen_target[pos] = false; } }
    all_targets.clear();
    if seen_target.len() < len + 1 { seen_target.resize(len + 1, false); }
    let current_states = &mut scratch.current_states;
    let done = &mut scratch.done;
    let active_indices = &mut scratch.active_indices;
    let match_positions = &mut scratch.match_positions;
    let match_gen = &mut scratch.match_gen;
    let cur_gen = scratch.cur_gen;
    let touched_groups = &mut scratch.touched_groups;
    let touched_states = &mut scratch.touched_states;
    let base_offsets = &scratch.base_offsets;
    active_indices.clear();
    let has_bytes = !slice.is_empty();
    let first_byte = if has_bytes { slice[0] } else { 0 };
    for (i, &state) in initial_states.iter().enumerate() {
        let base = base_offsets[i];
        for f in &pre.finalizers[state] {
            if f.gid < num_groups && match_gen[base + f.gid] != cur_gen {
                match_gen[base + f.gid] = cur_gen;
                match_positions[base + f.gid] = 0;
                let groups = &mut touched_groups[i];
                if groups.is_empty() { touched_states.push(i); }
                groups.push(f.gid);
            }
        }
        if !pre.has_transitions[state] { done[i] = true; continue; }
        if has_bytes && pre.transitions[state][first_byte as usize] == NONE_STATE {
            done[i] = true; continue;
        }
        active_indices.push(i);
    }
    if has_bytes && !active_indices.is_empty() {
        let mut active_len = active_indices.len();
        for (pos, &byte) in slice.iter().enumerate() {
            let position = (pos + 1) as u32;
            let mut next_len = 0usize;
            unsafe {
                for idx in 0..active_len {
                    let i = *active_indices.get_unchecked(idx);
                    let base = *base_offsets.get_unchecked(i);
                    let ns = *pre.transitions
                        .get_unchecked(*current_states.get_unchecked(i))
                        .get_unchecked(byte as usize);
                    if ns != NONE_STATE {
                        let ns = ns as usize;
                        *current_states.get_unchecked_mut(i) = ns;
                        for f in pre.finalizers.get_unchecked(ns) {
                            if f.gid < num_groups {
                                let idx = base + f.gid;
                                let slot_gen = match_gen.get_unchecked_mut(idx);
                                let was_none = *slot_gen != cur_gen;
                                if !f.non_greedy || was_none {
                                    *slot_gen = cur_gen;
                                    *match_positions.get_unchecked_mut(idx) = position;
                                }
                                if was_none {
                                    let groups = touched_groups.get_unchecked_mut(i);
                                    if groups.is_empty() { touched_states.push(i); }
                                    groups.push(f.gid);
                                }
                            }
                        }
                        if *pre.future_modes.get_unchecked(ns) == FutureMode::Terminate {
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
            if active_len == 0 { break; }
        }
    }
    // Collect end states and unique target positions
    for i in 0..num_states {
        scratch.end_states[i] = if done[i] || !pre.has_transitions[current_states[i]] {
            None
        } else {
            Some(current_states[i])
        };
        if num_groups > 0 {
            let base = base_offsets[i];
            for &gid in &touched_groups[i] {
                let pv = match_positions[base + gid];
                if pv > 0 {
                    let p = pv as usize;
                    if p <= len && !seen_target[p] { seen_target[p] = true; all_targets.push(p); }
                }
            }
        }
    }
}

impl SuffixScratch {
    fn new(num_groups: usize) -> Self {
        SuffixScratch {
            match_positions: vec![NONE_POS; num_groups],
            visited: Vec::new(), queue: Vec::new(), order: Vec::new(),
            nodes: Vec::new(),
        }
    }
    fn ensure_capacity(&mut self, len: usize) {
        let needed = len + 1;
        for &pos in &self.queue {
            if pos < self.visited.len() { self.visited[pos] = false; }
            if pos < self.nodes.len() { self.nodes[pos] = None; }
        }
        if self.visited.len() < needed { self.visited.resize(needed, false); }
        if self.nodes.len() < needed { self.nodes.resize(needed, None); }
        self.queue.clear();
        self.order.clear();
    }
}

/// Run DFA on suffix slice[base_pos..] from start_state, returning end state and edges.
fn execute_suffix(
    pre: &PrecomputedDfa, slice: &[u8], base_pos: usize, mpos: &mut [u32],
) -> (Option<usize>, EdgeList) {
    let ng = pre.num_groups;
    mpos[..ng].fill(NONE_POS);
    let mut touched = GroupList::new();
    let mut current = pre.start_state;
    let mut done = !pre.has_transitions[current];
    // Initial finalizers: all slots are NONE_POS, so first write always succeeds
    for f in &pre.finalizers[current] {
        if f.gid < ng && mpos[f.gid] == NONE_POS { mpos[f.gid] = 0; touched.push(f.gid); }
    }
    for (idx, &byte) in slice.iter().enumerate() {
        if done { break; }
        let ns = pre.transitions[current][byte as usize];
        if ns == NONE_STATE { done = true; break; }
        current = ns as usize;
        let position = (idx + 1) as u32;
        for f in &pre.finalizers[current] {
            if f.gid < ng {
                let was_none = mpos[f.gid] == NONE_POS;
                if f.non_greedy { if was_none { mpos[f.gid] = position; } }
                else { mpos[f.gid] = position; }
                if was_none { touched.push(f.gid); }
            }
        }
        done = pre.future_modes[current] == FutureMode::Terminate;
    }
    let end = if done || !pre.has_transitions[current] { None } else { Some(current) };
    touched.sort_unstable();
    let edges: EdgeList = touched.iter()
        .filter_map(|&gid| (mpos[gid] != NONE_POS && mpos[gid] != 0)
            .then(|| (gid, base_pos + mpos[gid] as usize)))
        .collect();
    (end, edges)
}

/// Build suffix DAG via BFS and compute hashes bottom-up.
fn compute_suffix_hashes(
    pre: &PrecomputedDfa, slice: &[u8], new_targets: &[usize],
    cache: &mut Vec<Option<u64>>, scratch: &mut SuffixScratch,
) {
    scratch.ensure_capacity(slice.len());
    for &pos in new_targets {
        if pos <= slice.len() && scratch.nodes[pos].is_none() && !scratch.visited[pos] {
            scratch.visited[pos] = true;
            scratch.queue.push(pos);
        }
    }
    if scratch.queue.is_empty() { return; }
    // BFS to build DAG
    let mut cursor = 0;
    while cursor < scratch.queue.len() {
        let pos = scratch.queue[cursor];
        cursor += 1;
        let (end_state, edges) = execute_suffix(pre, &slice[pos..], pos, &mut scratch.match_positions);
        for &(_, target) in &edges {
            if target <= slice.len() && scratch.nodes[target].is_none() && !scratch.visited[target] {
                scratch.visited[target] = true;
                scratch.queue.push(target);
            }
        }
        let ch = end_state.map(|id| pre.completion_hash[id]).unwrap_or(pre.none_completion_hash);
        scratch.nodes[pos] = Some((ch, edges));
        scratch.order.push(pos);
    }
    // Compute hashes bottom-up
    scratch.order.sort_unstable_by(|a, b| b.cmp(a));
    for &pos in &scratch.order {
        if cache[pos].is_some() { continue; }
        if let Some((ch, ref edges)) = scratch.nodes[pos] {
            let mut h = new_hasher();
            h.write_u64(ch);
            for &(gid, target) in edges.iter() {
                h.write_u64(gid as u64);
                h.write_u64(cache[target].unwrap_or(0));
            }
            cache[pos] = Some(h.finish());
        }
    }
    scratch.order.clear();
}

fn compute_chunk_signature(
    pre: &PrecomputedDfa, token: &[u8], chunk_states: &[usize], pos0: &mut Pos0Scratch,
    suffix: &mut SuffixScratch, cache: &mut Vec<Option<u64>>,
) -> u64 {
    compute_pos0(pre, pos0, token, chunk_states);
    if !pos0.all_targets.is_empty() {
        compute_suffix_hashes(pre, token, &pos0.all_targets, cache, suffix);
    }
    let mut sig: u64 = HASH_SEED3;
    for i in 0..chunk_states.len() {
        let ch = pos0.end_states[i]
            .map(|id| pre.completion_hash[id]).unwrap_or(pre.none_completion_hash);
        let state_sig = if pre.num_groups > 0 && !pos0.touched_groups[i].is_empty() {
            let groups = &mut pos0.touched_groups[i];
            if groups.len() > 1 { groups.sort_unstable(); }
            let base = pos0.base_offsets[i];
            let mut h = new_hasher();
            h.write_u64(ch);
            for &gid in groups.iter() {
                let pv = pos0.match_positions[base + gid];
                if pv > 0 {
                    h.write_u64(gid as u64);
                    h.write_u64(cache[pv as usize].unwrap_or(0));
                }
            }
            h.finish()
        } else { ch };
        sig = sig.wrapping_mul(HASH_SEED1).wrapping_add(state_sig);
    }
    sig
}

pub fn find_vocab_equivalence_classes<S: AsRef<[u8]> + Sync>(
    regex: &Tokenizer, strings: &[S], initial_states: &[usize],
) -> VocabEquivalenceResult {
    find_vocab_equivalence_classes_with_follow(regex, strings, initial_states, None, None, None)
}

/// Find vocab equivalence classes with optional follow-set pruning.
/// `ever_allowed_by_group`: per-group follow masks for projected suffix hashing.
/// `suffix_group_mask` and `group_to_class`: accepted for API compat, unused internally.
pub fn find_vocab_equivalence_classes_with_follow<S: AsRef<[u8]> + Sync>(
    regex: &Tokenizer, strings: &[S], initial_states: &[usize],
    _suffix_group_mask: Option<&[bool]>, ever_allowed_by_group: Option<&[Vec<bool>]>,
    _group_to_class: Option<&[usize]>,
) -> VocabEquivalenceResult {
    let pre = precompute_dfa(regex);
    let (num_tokens, num_states) = (strings.len(), initial_states.len());
    if num_states == 0 || num_tokens == 0 {
        return BTreeSet::from_iter(vec![(0..num_tokens).collect()]);
    }
    let num_groups = pre.num_groups;
    let batch_size = if num_states < 200 { num_states } else { 200 };
    let mut active_indices: Vec<usize> = (0..num_tokens).collect();
    let mut partition = vec![0usize; num_tokens];
    let mut next_class_id = 1usize;
    for batch_start in (0..num_states).step_by(batch_size) {
        if active_indices.is_empty() { break; }
        let batch = &initial_states[batch_start..(batch_start + batch_size).min(num_states)];
        let active_sigs: Vec<(usize, u64)> = active_indices.par_iter()
            .map_init(
                || (Pos0Scratch::new(batch.len(), num_groups),
                    SuffixScratch::new(num_groups), vec![None; 256]),
                |state, &ti| {
                    let (p0, sf, sc) = state;
                    let token = strings[ti].as_ref();
                    if sc.len() <= token.len() { sc.resize(token.len() + 1, None); }
                    sc.iter_mut().for_each(|x| *x = None);
                    (ti, compute_chunk_signature(&pre, token, batch, p0, sf, sc))
                },
            ).collect();
        let mut refinement: HashMap<(usize, u64), Vec<usize>> =
            HashMap::with_capacity(active_sigs.len() / 2);
        for (ti, sig) in active_sigs {
            refinement.entry((partition[ti], sig)).or_default().push(ti);
        }
        let mut by_old: HashMap<usize, Vec<Vec<usize>>> = HashMap::new();
        for ((oc, _), tokens) in refinement {
            by_old.entry(oc).or_default().push(tokens);
        }
        let mut new_active = Vec::with_capacity(active_indices.len());
        for (old_class, sub_groups) in by_old {
            let mut first = true;
            for tokens in sub_groups {
                let cid = if first { first = false; old_class }
                    else { let id = next_class_id; next_class_id += 1; id };
                for &ti in &tokens { partition[ti] = cid; }
                if tokens.len() > 1 { new_active.extend(tokens); }
            }
        }
        active_indices = new_active;
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::with_capacity(next_class_id);
    for (ti, &cid) in partition.iter().enumerate() { groups.entry(cid).or_default().push(ti); }
    groups.into_values().collect()
}
