//! Fast vocab equivalence analysis: partition tokens by DFA behavior signatures.
//!
//! For each token, computes a signature encoding its DFA behavior across all
//! initial states (match positions, suffix DAG structure, end states).
//! Tokens with identical signatures form equivalence classes.

// Do NOT add caching shortcuts that skip states/tokens. Full correctness mandatory.

use super::super::compat::{Sep1Tokenizer, FlatDfa, FlatDfaState, GroupID};
use ahash::{AHasher, RandomState};
use hashbrown::HashMap;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{BuildHasher, Hasher};

use crate::ds::bitset::BitSet;

pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;

type EdgeList = SmallVec<[(usize, usize); 4]>;

const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;
const NONE: u32 = u32::MAX;
const STATE_NONE: usize = usize::MAX;

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
    finalizers: Vec<SmallVec<[usize; 4]>>,
    is_dead_end: Vec<bool>,
    num_groups: usize,
    possible_future_groups: Vec<SmallVec<[usize; 4]>>,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
    /// Per-state bitset: which bytes cause a self-loop (transition back to same state).
    self_loop_bytes: Vec<[u64; 4]>,
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
        h.write_u64(disallowed.len() as u64);
        for &word in disallowed.words() {
            h.write_u64(word);
        }
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
    /// Per-state list of group IDs touched during the last run_batch call.
    /// Used to avoid O(num_states * num_groups) memset and scan.
    dirty_groups: Vec<SmallVec<[usize; 16]>>,
    targets: Vec<usize>,
    // Suffix DAG
    dag: HashMap<usize, (u64, EdgeList)>,
    dag_end_states: HashMap<usize, usize>,
    dag_queue: Vec<usize>,
    dag_disallowed: HashMap<usize, BitSet>,
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

fn normalize_disallowed_follows(
    num_groups: usize,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> Vec<BitSet> {
    let mut normalized = vec![BitSet::new(num_groups); num_groups];
    for gid in 0..num_groups {
        if let Some(bits) = disallowed_follows.get(&(gid as u32)) {
            let mut out = BitSet::new(num_groups);
            for bit in bits.iter() {
                if bit < num_groups {
                    out.set(bit);
                }
            }
            normalized[gid] = out;
        }
    }
    normalized
}

fn build_dfa(regex: &Sep1Tokenizer, disallowed_follows: &BTreeMap<u32, BitSet>) -> Dfa {
    let dfa = regex.dfa();
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

    let mut transitions = Vec::with_capacity(dfa.states.len());
    let mut finalizers = Vec::with_capacity(dfa.states.len());
    let mut is_dead_end = Vec::with_capacity(dfa.states.len());
    let mut possible_future_groups = Vec::with_capacity(dfa.states.len());
    let mut completion_hash = Vec::with_capacity(dfa.states.len());

    for state in &dfa.states {
        let mut table = [NONE; 256];
        for (byte_idx, &target) in state.transitions.iter().enumerate() { let byte = byte_idx as u8;
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

    Dfa {
        start_state: dfa.start_state,
        num_states: num_dfa_states,
        byte_to_class,
        num_classes,
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
        Scratch {
            current_states: vec![0; num_states],
            active_indices: Vec::new(),
            match_positions: vec![NONE; num_states * num_groups],
            dirty_groups: vec![SmallVec::new(); num_states],
            targets: Vec::new(),
            dag: HashMap::new(),
            dag_end_states: HashMap::new(),
            dag_queue: Vec::new(),
            dag_disallowed: HashMap::new(),
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
        for &gid in &dfa.finalizers[state] {
            if gid < num_groups && scratch.match_positions[base + gid] == NONE {
                scratch.match_positions[base + gid] = 0;
                scratch.dirty_groups[i].push(gid);
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
                    for &gid in &dfa.finalizers[ns] {
                        if gid < num_groups {
                            let ix = base + gid;
                            scratch.match_positions[ix] = position;
                            scratch.dirty_groups[i].push(gid);
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
                        for &gid in &dfa.finalizers[s] {
                            if gid < num_groups {
                                scratch.match_positions[base + gid] = token_len;
                                scratch.dirty_groups[i].push(gid);
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

    for &gid in &dfa.finalizers[current] {
        if gid < num_groups && match_positions[gid] == NONE {
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
                match_positions[gid] = position;
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
    scratch.dag_end_states.clear();
    scratch.dag_queue.clear();
    scratch.dag_disallowed.clear();

    // BFS from target positions: run suffix DFA at each, discover new positions from edges
    for &pos in &scratch.targets {
        if pos <= len && !scratch.dag.contains_key(&pos) {
            scratch.dag_queue.push(pos);
            scratch.dag.insert(pos, (0, EdgeList::new())); // placeholder
        }
    }

    for si in 0..scratch.dirty_groups.len() {
        let base = si * dfa.num_groups;
        let dirty = scratch.dirty_groups[si].clone();
        for gid in dirty {
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
            run_suffix(dfa, &slice[pos..], pos, &mut suffix_mp);
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

    // Hash bottom-up: process deeper positions first
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
                for &gid in dirty.iter() {
                    let pv = scratch.match_positions[base + gid];
                    if pv != NONE && pv > 0 {
                        h.write_u64(gid as u64);
                        h.write_u64(scratch.dag.get(&(pv as usize)).map_or(0, |e| e.0));
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

pub fn find_vocab_equivalence_classes_with_follow<S: AsRef<[u8]> + Sync>(
    regex: &Sep1Tokenizer,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> VocabEquivalenceResult {
    let dfa = build_dfa(regex, disallowed_follows);
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
    }

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::with_capacity(next_class_id);
    for (ti, &cid) in partition.iter().enumerate() {
        groups.entry(cid).or_default().push(ti);
    }
    groups.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::ast::{bytes, choice};
    use crate::compiler::compile::build_tokenizer_from_exprs;
    use crate::compiler::stages::equivalence_analysis::compat::Sep1Tokenizer;
    use std::collections::BTreeMap;
    use std::path::Path;

    fn build_gpt2_unicode_to_byte_map() -> BTreeMap<char, u8> {
        let mut byte_values: Vec<u32> = (b'!' as u32..=b'~' as u32).collect();
        byte_values.extend(0xA1u32..=0xACu32);
        byte_values.extend(0xAEu32..=0xFFu32);

        let mut unicode_values = byte_values.clone();
        let mut extra = 0u32;
        for byte in 0u32..=255u32 {
            if !byte_values.contains(&byte) {
                byte_values.push(byte);
                unicode_values.push(256 + extra);
                extra += 1;
            }
        }

        let mut unicode_to_byte = BTreeMap::new();
        for (byte, codepoint) in byte_values.into_iter().zip(unicode_values.into_iter()) {
            let ch = char::from_u32(codepoint).expect("valid GPT-2 codepoint");
            unicode_to_byte.insert(ch, byte as u8);
        }
        unicode_to_byte
    }

    fn load_cached_gpt2_vocab_bytes() -> Vec<Vec<u8>> {
        let vocab_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../constraint-framework-analysis/.cache/vocab_cache/vocab.json");
        let raw = std::fs::read_to_string(&vocab_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", vocab_path.display()));
        let vocab: BTreeMap<String, u32> =
            serde_json::from_str(&raw).expect("cached GPT-2 vocab should parse");
        let unicode_to_byte = build_gpt2_unicode_to_byte_map();
        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        let mut tokens = vec![Vec::new(); max_id + 1];

        for (token_str, token_id) in vocab {
            let token_bytes: Vec<u8> = token_str
                .chars()
                .map(|ch| unicode_to_byte[&ch])
                .collect();
            tokens[token_id as usize] = token_bytes;
        }

        tokens
    }

    #[test]
    fn test_disallowed_follow_merges_ab_and_ac() {
        let tokenizer = build_tokenizer_from_exprs(&[
            choice(vec![bytes(b"a"), bytes(b"b")]),
            choice(vec![bytes(b"b"), bytes(b"c")]),
        ]);
        let sep1_tok = Sep1Tokenizer::new(&tokenizer);
        let tokens = vec![b"ab".to_vec(), b"ac".to_vec()];
        let initial_states = vec![sep1_tok.initial_state_id()];

        let mut disallowed = BTreeMap::new();
        let mut bitset = BitSet::new(2);
        bitset.set(0);
        disallowed.insert(0u32, bitset);

        let classes = find_vocab_equivalence_classes_with_follow(
            &sep1_tok,
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
    fn test_completion_with_disallowed_distinguishes_detector_state() {
        let dfa = Dfa {
            start_state: 0,
            num_states: 1,
            byte_to_class: [0; 256],
            num_classes: 1,
            trans_by_class: vec![NONE],
            finalizers: vec![SmallVec::new()],
            is_dead_end: vec![false],
            num_groups: 3,
            possible_future_groups: vec![smallvec::smallvec![0usize]],
            completion_hash: vec![123],
            none_completion_hash: 456,
            self_loop_bytes: vec![[0; 4]],
            disallowed_follows: vec![BitSet::new(3); 3],
        };
        let mut disallowed_a = BitSet::new(3);
        disallowed_a.set(1);
        let mut disallowed_b = BitSet::new(3);
        disallowed_b.set(2);

        assert_ne!(
            dfa.completion_with_disallowed(0, Some(&disallowed_a)),
            dfa.completion_with_disallowed(0, Some(&disallowed_b)),
            "different disallowed-follow detector states must not collapse when filtered futures match"
        );
    }

    #[test]
    #[ignore]
    fn diagnose_g_gb_witness_fast_suffix_hashes() {
        let lark = include_str!("../../../../../tests/fixtures/github_hard_o56012_split_quotes.lark");
        let grammar = crate::import::lark::parse_lark(lark).expect("lark should parse");
        let analyzed = crate::compiler::glr::analysis::AnalyzedGrammar::from_grammar_def(&grammar);
        let disallowed_follows = crate::compiler::compile::compute_disallowed_follows(&analyzed);
        let tokenizer = crate::compiler::compile::build_tokenizer(&grammar);
        let sep1 = Sep1Tokenizer::new(&tokenizer);
        let dfa = build_dfa(&sep1, &disallowed_follows);
        let initial_states = [269usize];
        let tokens = [b"G".as_slice(), b"GB".as_slice()];

        for (ti, token) in tokens.iter().enumerate() {
            println!("\n=== fast token {ti} {:?} ===", String::from_utf8_lossy(token));
            let mut scratch = Scratch::new(initial_states.len(), dfa.num_groups);
            run_batch(&dfa, &mut scratch, token, &initial_states);

            let mut dirty = scratch.dirty_groups[0].clone();
            dirty.sort_unstable();
            dirty.dedup();
            println!("end_state={:?}", scratch.current_states[0]);
            println!("targets={:?}", scratch.targets);
            println!("dirty groups / positions:");
            for &gid in &dirty {
                let pos = scratch.match_positions[gid];
                println!("  gid {gid} -> {pos}");
            }

            if !scratch.targets.is_empty() {
                hash_suffixes(&dfa, token, &mut scratch);
                let mut positions = scratch.dag_queue.clone();
                positions.sort_unstable();
                println!("suffix DAG:");
                for pos in positions {
                    let (hash, edges) = scratch.dag[&pos].clone();
                    let end_state = scratch.dag_end_states.get(&pos).copied().unwrap_or(STATE_NONE);
                    println!("  pos {pos}: end_state={end_state} hash=0x{hash:016X} edges={edges:?}");
                }
            }

            let mut sig_scratch = Scratch::new(initial_states.len(), dfa.num_groups);
            let sig = token_signature(&dfa, token, &initial_states, &mut sig_scratch);
            println!("signature=0x{sig:016X}");
        }
    }

    #[test]
    #[ignore]
    fn diagnose_g_gb_on_full_vocab_reduced_states() {
        let lark = include_str!("../../../../../tests/fixtures/github_hard_o56012_split_quotes.lark");
        let grammar = crate::import::lark::parse_lark(lark).expect("lark should parse");
        let analyzed = crate::compiler::glr::analysis::AnalyzedGrammar::from_grammar_def(&grammar);
        let disallowed_follows = crate::compiler::compile::compute_disallowed_follows(&analyzed);
        let tokenizer = crate::compiler::compile::build_tokenizer(&grammar);
        let sep1 = Sep1Tokenizer::new(&tokenizer);

        let full_tokens = load_cached_gpt2_vocab_bytes();
        assert_eq!(full_tokens[38], b"G");
        assert_eq!(full_tokens[4579], b"GB");

        let states: Vec<usize> = (0..sep1.dfa().states.len()).collect();
        let mapping = crate::compiler::stages::equivalence_analysis::state::fast::find_state_equivalence_classes(
            &sep1,
            &full_tokens,
            &states,
        );

        let rep_269 = mapping[269];
        let reduced_states: Vec<usize> = {
            let mut set = BTreeSet::new();
            for &rep in &mapping {
                set.insert(rep);
            }
            set.into_iter().collect()
        };

        println!("full-vocab rep(269) = {rep_269}");
        println!("full-vocab reduced state count = {}", reduced_states.len());
        println!("state 269 is representative: {}", reduced_states.contains(&269));

        let pair_tokens = vec![full_tokens[38].clone(), full_tokens[4579].clone()];
        let pair_classes = find_vocab_equivalence_classes_with_follow(
            &sep1,
            &pair_tokens,
            &reduced_states,
            &disallowed_follows,
        );
        println!("pair classes on full-vocab reduced states: {pair_classes:?}");

        let dfa = build_dfa(&sep1, &disallowed_follows);
        let mut left_scratch = Scratch::new(reduced_states.len(), dfa.num_groups);
        let mut right_scratch = Scratch::new(reduced_states.len(), dfa.num_groups);
        let left_sig = token_signature(&dfa, &full_tokens[38], &reduced_states, &mut left_scratch);
        let right_sig = token_signature(&dfa, &full_tokens[4579], &reduced_states, &mut right_scratch);
        println!("signature(G)=0x{left_sig:016X}");
        println!("signature(GB)=0x{right_sig:016X}");
        println!("signatures equal: {}", left_sig == right_sig);

        let full_classes = find_vocab_equivalence_classes_with_follow(
            &sep1,
            &full_tokens,
            &reduced_states,
            &disallowed_follows,
        );
        let same_full_class = full_classes
            .iter()
            .any(|class| class.contains(&38) && class.contains(&4579));
        println!("same full fast class: {same_full_class}");
    }

    #[test]
    #[ignore]
    fn diagnose_o56012_current_witness_fast_suffix_hashes() {
        let lark = include_str!("../../../../../tests/fixtures/github_hard_o56012_split_quotes.lark");
        let grammar = crate::import::lark::parse_lark(lark).expect("lark should parse");
        let (normalized, tokenizer) =
            crate::compiler::grammar::transforms::prepare_grammar_for_compile(&grammar);
        let analyzed = crate::compiler::glr::analysis::AnalyzedGrammar::from_grammar_def(&normalized);
        let disallowed_follows = crate::compiler::compile::compute_disallowed_follows(&analyzed);
        let sep1 = Sep1Tokenizer::new(&tokenizer);
        let dfa = build_dfa(&sep1, &disallowed_follows);
        let initial_states = [9686usize];
        let tokens = [b",\"".as_slice(), b",'\"".as_slice()];

        for (ti, token) in tokens.iter().enumerate() {
            println!("\n=== fast token {ti} {:?} ===", String::from_utf8_lossy(token));
            let mut scratch = Scratch::new(initial_states.len(), dfa.num_groups);
            run_batch(&dfa, &mut scratch, token, &initial_states);

            let mut dirty = scratch.dirty_groups[0].clone();
            dirty.sort_unstable();
            dirty.dedup();
            println!("end_state={:?}", scratch.current_states[0]);
            println!("targets={:?}", scratch.targets);
            println!("dirty groups / positions:");
            for &gid in &dirty {
                let pos = scratch.match_positions[gid];
                println!("  gid {gid} -> {pos}");
            }

            if !scratch.targets.is_empty() {
                hash_suffixes(&dfa, token, &mut scratch);
                let mut positions = scratch.dag_queue.clone();
                positions.sort_unstable();
                println!("suffix DAG:");
                for pos in positions {
                    let (hash, edges) = scratch.dag[&pos].clone();
                    let end_state = scratch.dag_end_states.get(&pos).copied().unwrap_or(STATE_NONE);
                    let disallowed = scratch.dag_disallowed.get(&pos).cloned();
                    println!(
                        "  pos {pos}: end_state={end_state} hash=0x{hash:016X} disallowed={:?} edges={edges:?}",
                        disallowed
                    );
                }
            }

            let mut sig_scratch = Scratch::new(initial_states.len(), dfa.num_groups);
            let sig = token_signature(&dfa, token, &initial_states, &mut sig_scratch);
            println!("signature=0x{sig:016X}");
        }
    }
}
