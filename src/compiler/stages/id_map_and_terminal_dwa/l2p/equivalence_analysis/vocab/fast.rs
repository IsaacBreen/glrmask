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
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{BuildHasher, Hasher};

use super::super::disallowed_follows::normalize_disallowed_follows;
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

/// Precomputed transition-only data that is identical across partitions.
///
/// `filter_for_terminals` only changes finalizers and possible_future_group_ids,
/// not transitions. So `trans_by_class`, `byte_to_class`, and `self_loop_bytes`
/// can be computed once and shared across all partition vocab equivalence calls.
pub struct SharedVocabDfaBase {
    byte_to_class: [u8; 256],
    pub num_classes: usize,
    trans_by_class: Vec<u32>,
    self_loop_bytes: Vec<U8Set>,
    none_completion_hash: u64,
    /// Hash of the full transition table used to build this cache.
    /// Used to detect incompatible DFAs that happen to share the same state count.
    transition_hash: u64,
}

impl SharedVocabDfaBase {
    /// Build from a FlatDfa. Called lazily via OnceLock on first use.
    pub fn build_from_dfa(dfa: &super::super::compat::FlatDfa) -> Self {
        let num_dfa_states = dfa.states.len();
        let byte_to_class = compute_byte_classes(dfa);
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

        // Fused row-major construction: one pass over DFA states instead of
        // 51 column-major passes for trans_by_class + 1 pass for self_loop_bytes.
        let mut trans_by_class = vec![NONE; num_classes * num_dfa_states];
        let mut self_loop_bytes = Vec::with_capacity(num_dfa_states);
        for s in 0..dfa.states.len() {
            for c in 0..num_classes {
                trans_by_class[c * num_dfa_states + s] =
                    dfa.trans(s, class_repr[c] as usize);
            }
            let mut bits = U8Set::empty();
            for (byte_idx, &target) in dfa.transitions_for(s).iter().enumerate() {
                if target == s as u32 {
                    bits.insert(byte_idx as u8);
                }
            }
            self_loop_bytes.push(bits);
        }

        let none_completion_hash = {
            let mut h = new_hasher();
            h.write_u8(0);
            h.finish()
        };

        let transition_hash = {
            let mut h = new_hasher();
            for s in 0..dfa.states.len() {
                for &t in dfa.transitions_for(s) {
                    h.write_u32(t);
                }
            }
            h.finish()
        };

        SharedVocabDfaBase {
            byte_to_class,
            num_classes,
            trans_by_class,
            self_loop_bytes,
            none_completion_hash,
            transition_hash,
        }
    }

    /// Build directly from a flat transition table (`[u32; num_states * 256]`).
    /// Skips FlatDfa/metadata construction — only needs transition data.
    pub fn build_from_flat_trans(flat_trans: &[u32]) -> Self {
        let num_dfa_states = flat_trans.len() / 256;
        assert_eq!(flat_trans.len(), num_dfa_states * 256);

        // Compute byte classes using column hashing (same logic as compute_byte_classes).
        let byte_to_class = {
            let mut column_hashes = [0u64; 256];
            for s in 0..num_dfa_states {
                let base = s * 256;
                for b in 0..256 {
                    column_hashes[b] = column_hashes[b]
                        .wrapping_mul(0x517cc1b727220a95)
                        .wrapping_add(flat_trans[base + b] as u64);
                }
            }
            let mut sorted_indices: [u8; 256] = std::array::from_fn(|i| i as u8);
            sorted_indices.sort_unstable_by_key(|&b| column_hashes[b as usize]);
            let mut btc = [0u8; 256];
            let mut next_class = 0u8;
            btc[sorted_indices[0] as usize] = 0;
            for i in 1..256 {
                let curr = sorted_indices[i];
                let h = column_hashes[curr as usize];
                if h != column_hashes[sorted_indices[i - 1] as usize] {
                    next_class += 1;
                    btc[curr as usize] = next_class;
                } else {
                    let mut assigned = false;
                    for j in (0..i).rev() {
                        let prev = sorted_indices[j];
                        if column_hashes[prev as usize] != h {
                            break;
                        }
                        let same = (0..num_dfa_states).all(|s| {
                            let base = s * 256;
                            flat_trans[base + curr as usize] == flat_trans[base + prev as usize]
                        });
                        if same {
                            btc[curr as usize] = btc[prev as usize];
                            assigned = true;
                            break;
                        }
                    }
                    if !assigned {
                        next_class += 1;
                        btc[curr as usize] = next_class;
                    }
                }
            }
            btc
        };

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

        let mut trans_by_class = vec![NONE; num_classes * num_dfa_states];
        let mut self_loop_bytes = Vec::with_capacity(num_dfa_states);
        for s in 0..num_dfa_states {
            let base = s * 256;
            for c in 0..num_classes {
                trans_by_class[c * num_dfa_states + s] =
                    flat_trans[base + class_repr[c] as usize];
            }
            let mut bits = U8Set::empty();
            for byte_idx in 0..256 {
                if flat_trans[base + byte_idx] == s as u32 {
                    bits.insert(byte_idx as u8);
                }
            }
            self_loop_bytes.push(bits);
        }

        let none_completion_hash = {
            let mut h = new_hasher();
            h.write_u8(0);
            h.finish()
        };

        let transition_hash = {
            let mut h = new_hasher();
            for &t in flat_trans {
                h.write_u32(t);
            }
            h.finish()
        };

        SharedVocabDfaBase {
            byte_to_class,
            num_classes,
            trans_by_class,
            self_loop_bytes,
            none_completion_hash,
            transition_hash,
        }
    }

    /// Return the precomputed byte-to-class mapping.
    pub fn byte_to_class(&self) -> [u8; 256] {
        self.byte_to_class
    }

    /// Check if this cache was built from a DFA with the given state count
    /// and identical transitions (verified via transition hash).
    pub fn is_compatible_with_state_count(&self, num_dfa_states: usize) -> bool {
        self.trans_by_class.len() == self.num_classes * num_dfa_states
            && self.self_loop_bytes.len() == num_dfa_states
    }

    /// Check full compatibility: state count AND transition hash must match.
    /// Two DFAs with the same state count but different transitions (e.g. from
    /// different simplify_for_terminals outcomes) must not share the cache.
    pub fn is_compatible_with_dfa(&self, dfa: &super::super::compat::FlatDfa) -> bool {
        let num_dfa_states = dfa.states.len();
        if !self.is_compatible_with_state_count(num_dfa_states) {
            return false;
        }
        let mut h = new_hasher();
        for s in 0..num_dfa_states {
            for &t in dfa.transitions_for(s) {
                h.write_u32(t);
            }
        }
        h.finish() == self.transition_hash
    }
}

/// Cache type for lazy SharedVocabDfaBase initialization across partitions.
pub type SharedVocabDfaCache = std::sync::OnceLock<SharedVocabDfaBase>;

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
    dirty_state_flags: Vec<u8>,
    /// Per-state multi-word dirty bitmask.  Layout: `[dirty_words * num_states]`
    /// where state `i`'s dirty mask occupies indices `[i*dirty_words .. (i+1)*dirty_words]`.
    dirty_group_masks: Vec<u64>,
    /// Number of u64 words per state in `dirty_group_masks` (= ceil(num_groups/64)).
    dirty_words: usize,
    targets: Vec<usize>,
    target_gids: HashMap<usize, SmallVec<[usize; 16]>>,
    single_target_pos: usize,
    single_target_gids: SmallVec<[usize; 16]>,
    single_target_seen: Vec<u32>,
    single_target_seen_epoch: u32,
    // Suffix DAG
    dag_nodes: Vec<Option<DagNode>>,
    dag_queue: Vec<usize>,
    dag_disallowed: Vec<Option<BitSet>>,
    single_target_hash_pos: usize,
    single_target_hash: u64,
    suffix_match_positions: Vec<u32>,
    suffix_dirty_groups: SmallVec<[usize; 16]>,
}

static HASH_RANDOM_STATE: Lazy<RandomState> =
    Lazy::new(|| RandomState::with_seeds(HASH_SEED1, HASH_SEED2, HASH_SEED3, HASH_SEED4));
static VOCAB_UNGROUPED_BATCH: Lazy<bool> =
    Lazy::new(|| env_flag_enabled("GLRMASK_VOCAB_UNGROUPED_BATCH"));

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
    build_dfa_with_group_filter(tokenizer, disallowed_follows, byte_to_class_override, None, None)
}

/// Build the analysis DFA, optionally filtering to only active groups.
///
/// When `active_groups` is provided, only groups marked `true` are included
/// in finalizers and possible_future_groups. This is used for L2+-only
/// equivalence analysis where L1 terminal groups are excluded.
///
/// When `shared_cache` is provided, lazily initializes a SharedVocabDfaBase on
/// first call and reuses it for subsequent calls. This avoids redundant
/// computation of transition tables and self-loop bytes across partitions.
fn build_dfa_with_group_filter(
    tokenizer: &TokenizerView,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    byte_to_class_override: Option<&[u8; 256]>,
    active_groups: Option<&[bool]>,
    shared_cache: Option<&SharedVocabDfaCache>,
) -> Dfa {
    let dfa = tokenizer.dfa();
    assert!(dfa.states.len() <= u32::MAX as usize, "DFA too large");

    // Lazily initialize the shared base from this TokenizerView's DFA.
    let shared_base: Option<&SharedVocabDfaBase> = shared_cache.map(|cache| {
        cache.get_or_init(|| SharedVocabDfaBase::build_from_dfa(dfa))
    });

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

    let mut finalizers = Vec::with_capacity(dfa.states.len());
    let mut is_dead_end = Vec::with_capacity(dfa.states.len());
    let mut possible_future_groups = Vec::with_capacity(dfa.states.len());
    let mut completion_hash = Vec::with_capacity(dfa.states.len());

    for (_state_idx, state) in dfa.states.iter().enumerate() {
        let filtered_finalizers: SmallVec<[usize; 4]> = if let Some(ag) = active_groups {
            state.finalizers.iter().copied().filter(|&gid| ag.get(gid).copied().unwrap_or(false)).collect()
        } else {
            state.finalizers.iter().copied().collect()
        };
        finalizers.push(filtered_finalizers);

        let future_groups: SmallVec<[usize; 4]> = if let Some(ag) = active_groups {
            state.possible_future_group_ids.iter().copied().filter(|&gid| ag.get(gid).copied().unwrap_or(false)).collect()
        } else {
            state.possible_future_group_ids.iter().copied().collect()
        };
        is_dead_end.push(future_groups.is_empty());
        completion_hash.push(hash_group_list(future_groups.iter().copied()));
        possible_future_groups.push(future_groups);
    }

    let num_dfa_states = dfa.states.len();

    // Use shared base if available and compatible, otherwise compute from scratch.
    // When simplify_for_terminals minimizes the DFA (changing transitions),
    // the shared base may be incompatible and must be skipped.
    let compatible_shared_base = shared_base.filter(|base| {
        base.is_compatible_with_dfa(dfa)
    });
    let (byte_to_class, trans_by_class, self_loop_bytes, none_completion_hash) =
        if let Some(base) = compatible_shared_base {
            let btc = byte_to_class_override.copied().unwrap_or(base.byte_to_class);
            (btc, base.trans_by_class.clone(), base.self_loop_bytes.clone(), base.none_completion_hash)
        } else {
            let btc = byte_to_class_override.copied().unwrap_or_else(|| compute_byte_classes(dfa));
            let num_classes = btc
                .iter()
                .copied()
                .max()
                .map_or(0usize, |max_class| max_class as usize + 1);
            let mut class_repr = vec![0u8; num_classes];
            let mut class_seen = vec![false; num_classes];
            for b in 0..=255u8 {
                let class = btc[b as usize] as usize;
                if !class_seen[class] {
                    class_seen[class] = true;
                    class_repr[class] = b;
                }
            }

            let mut tbc = vec![NONE; num_classes * num_dfa_states];
            for c in 0..num_classes {
                let repr = class_repr[c] as usize;
                let bbase = c * num_dfa_states;
                for s in 0..num_dfa_states {
                    tbc[bbase + s] = dfa.trans(s, repr);
                }
            }

            let mut slb = Vec::with_capacity(num_dfa_states);
            for s in 0..dfa.states.len() {
                let mut bits = U8Set::empty();
                for (byte_idx, &target) in dfa.transitions_for(s).iter().enumerate() {
                    if target == s as u32 {
                        bits.insert(byte_idx as u8);
                    }
                }
                slb.push(bits);
            }

            let nch = {
                let mut h = new_hasher();
                h.write_u8(0);
                h.finish()
            };

            (btc, tbc, slb, nch)
        };

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
    ensure_position_slot(&mut scratch.dag_disallowed, pos);
    if let Some(existing) = scratch.dag_disallowed[pos].as_mut() {
        *existing = existing.intersection(incoming);
    } else {
        scratch.dag_disallowed[pos] = Some(incoming.clone());
    }
}

fn node_disallows_gid(scratch: &Scratch, pos: usize, gid: usize) -> bool {
    scratch
        .dag_disallowed
        .get(pos)
        .and_then(|bits| bits.as_ref())
        .map(|bits| bits.contains(gid))
        .unwrap_or(false)
}

#[inline]
fn ensure_position_slot<T>(slots: &mut Vec<Option<T>>, pos: usize) {
    if pos >= slots.len() {
        slots.resize_with(pos + 1, || None);
    }
}

impl Scratch {
    fn new(num_states: usize, num_groups: usize) -> Self {
        let dirty_words = (num_groups + 63) / 64;
        Scratch {
            current_states: vec![0; num_states],
            active_indices: Vec::new(),
            match_positions: vec![NONE; num_states * num_groups],
            dirty_state_flags: vec![0; num_states],
            dirty_group_masks: vec![0; num_states * dirty_words.max(1)],
            dirty_words,
            targets: Vec::new(),
            target_gids: HashMap::new(),
            single_target_pos: usize::MAX,
            single_target_gids: SmallVec::new(),
            single_target_seen: vec![0; num_groups],
            single_target_seen_epoch: 1,
            dag_nodes: Vec::new(),
            dag_queue: Vec::new(),
            dag_disallowed: Vec::new(),
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
    scratch.dirty_state_flags[state_idx] = 1;
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

fn advance_seen_epoch(seen: &mut [u32], epoch: &mut u32) {
    *epoch = epoch.wrapping_add(1);
    if *epoch == 0 {
        seen.fill(0);
        *epoch = 1;
    }
}

fn record_target_gid(
    targets: &mut Vec<usize>,
    target_gids: &mut HashMap<usize, SmallVec<[usize; 16]>>,
    single_target_pos: &mut usize,
    single_target_gids: &mut SmallVec<[usize; 16]>,
    single_target_seen: &mut [u32],
    single_target_seen_epoch: &mut u32,
    pos: usize,
    gid: usize,
) {
    if targets.is_empty() {
        targets.push(pos);
        *single_target_pos = pos;
        single_target_gids.clear();
        advance_seen_epoch(single_target_seen, single_target_seen_epoch);
        if gid < single_target_seen.len() {
            single_target_seen[gid] = *single_target_seen_epoch;
        }
        single_target_gids.push(gid);
        return;
    }

    if target_gids.is_empty() && targets.len() == 1 && *single_target_pos == pos {
        if gid < single_target_seen.len() {
            if single_target_seen[gid] != *single_target_seen_epoch {
                single_target_seen[gid] = *single_target_seen_epoch;
                single_target_gids.push(gid);
            }
        } else if !single_target_gids.contains(&gid) {
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
    for flag in scratch.dirty_state_flags[state_offset..state_offset + num_states].iter_mut() {
        *flag = 0;
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
        dirty_state_flags,
        dirty_group_masks,
        match_positions,
        targets,
        target_gids,
        single_target_pos,
        single_target_gids,
        single_target_seen,
        single_target_seen_epoch,
        ..
    } = scratch;

    for i in state_offset..state_offset + num_states {
        if dirty_state_flags[i] == 0 {
            continue;
        }
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
                        single_target_seen,
                        single_target_seen_epoch,
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
    _profile_signature_detail: bool,
) {
    let num_states = initial_states.len();
    let num_groups = dfa.num_groups;
    let len = slice.len();

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
        run_batch_inner(dfa, scratch, slice, 0, num_states);

        collect_targets(
            scratch,
            num_groups,
            len,
            0,
            num_states,
        );
    } else {
        for state_offset in (0..num_states).step_by(state_group_size) {
            let group_len = (state_offset + state_group_size).min(num_states) - state_offset;

            run_batch_inner(dfa, scratch, slice, state_offset, group_len);

            collect_targets(
                scratch,
                num_groups,
                len,
                state_offset,
                group_len,
            );
        }
    }
}

fn hash_suffixes(
    dfa: &Dfa,
    slice: &[u8],
    scratch: &mut Scratch,
    _profile_signature_detail: bool,
) -> usize {
    let len = slice.len();
    scratch.dag_nodes.clear();
    scratch.dag_queue.clear();
    scratch.dag_disallowed.clear();

    for (&pos, gids) in &scratch.target_gids {
        if let Some((&first_gid, rest)) = gids.split_first() {
            let mut combined = dfa.disallowed_for(first_gid).clone();
            for &gid in rest {
                combined = combined.intersection(dfa.disallowed_for(gid));
            }
            ensure_position_slot(&mut scratch.dag_disallowed, pos);
            scratch.dag_disallowed[pos] = Some(combined);
        }
    }

    // BFS from target positions: run suffix DFA at each, discover new positions from edges
    for &pos in &scratch.targets {
        if pos < len {
            ensure_position_slot(&mut scratch.dag_nodes, pos);
            if scratch.dag_nodes[pos].is_some() {
                continue;
            }
            scratch.dag_queue.push(pos);
            scratch.dag_nodes[pos] = Some(DagNode {
                hash: 0,
                edges: EdgeList::new(),
                end_state: STATE_NONE,
            });
        }
    }

    let mut cursor = 0;
    while cursor < scratch.dag_queue.len() {
        let pos = scratch.dag_queue[cursor];
        cursor += 1;
        let (end_state, edges) = run_suffix(
            dfa,
            &slice[pos..],
            pos,
            &mut scratch.suffix_match_positions,
            &mut scratch.suffix_dirty_groups,
        );
        for &(_, target) in &edges {
            if target < len {
                ensure_position_slot(&mut scratch.dag_nodes, target);
                if scratch.dag_nodes[target].is_some() {
                    continue;
                }
                scratch.dag_queue.push(target);
                scratch.dag_nodes[target] = Some(DagNode {
                    hash: 0,
                    edges: EdgeList::new(),
                    end_state: STATE_NONE,
                });
            }
        }
        scratch.dag_nodes[pos] = Some(DagNode {
            hash: 0,
            edges,
            end_state: end_state.unwrap_or(STATE_NONE),
        });
    }

    scratch.dag_queue.sort_unstable();
    for idx in 0..scratch.dag_queue.len() {
        let pos = scratch.dag_queue[idx];
        let edges = scratch.dag_nodes[pos].as_ref().unwrap().edges.clone();

        for &(gid, target) in &edges {
            if node_disallows_gid(scratch, pos, gid) {
                continue;
            }
            if target < len {
                intersect_node_disallowed(scratch, target, dfa.disallowed_for(gid));
            }
        }
    }

    // Hash bottom-up: process deeper positions first. Reuse the ascending
    // topological order from propagation and walk it in reverse instead of
    // paying for a second sort.
    for idx in (0..scratch.dag_queue.len()).rev() {
        let pos = scratch.dag_queue[idx];
        let node = scratch.dag_nodes[pos].as_ref().unwrap();
        let edges = node.edges.clone();
        let end_state = node.end_state;
        let mut h = new_hasher();
        h.write_u64(dfa.completion_with_disallowed(
            end_state,
            scratch.dag_disallowed.get(pos).and_then(|bits| bits.as_ref()),
        ));

        for &(gid, target) in &edges {
            if node_disallows_gid(scratch, pos, gid) {
                continue;
            }
            h.write_u64(gid as u64);
            h.write_u64(
                scratch
                    .dag_nodes
                    .get(target)
                    .and_then(|node| node.as_ref())
                    .map_or(0, |node| node.hash),
            );
        }
        scratch.dag_nodes[pos].as_mut().unwrap().hash = h.finish();
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

    let mut h = new_hasher();
    h.write_u64(dfa.completion_with_disallowed(
        end_state.unwrap_or(STATE_NONE),
        Some(&root_disallowed),
    ));
    for &(gid, _target) in &edges {
        if root_disallowed.contains(gid) {
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
    let dag = &scratch.dag_nodes;
    let single_target_hash_pos = scratch.single_target_hash_pos;
    let single_target_hash = scratch.single_target_hash;
    let mut sig: u64 = HASH_SEED3;
    for i in 0..chunk_states.len() {
        let completion = dfa.completion(scratch.current_states[i]);
        let base = i * num_groups;
        let mask_base = i * dirty_words;

        let state_sig = if scratch.dirty_state_flags[i] != 0 {
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
                            dag.get(target)
                                .and_then(|node| node.as_ref())
                                .map_or(0, |node| node.hash)
                        };
                        h.write_u64(target_hash);
                    }
                    scratch.match_positions[base + gid] = NONE;
                }
                scratch.dirty_group_masks[mask_base + w] = 0;
            }
            scratch.dirty_state_flags[i] = 0;
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
    _profile_signature_detail: bool,
) -> u64 {
    scratch.single_target_hash_pos = usize::MAX;
    scratch.single_target_hash = 0;
    run_batch(
        dfa,
        scratch,
        token,
        chunk_states,
        state_group_size,
        false,
    );
    let target_count = scratch.targets.len();
    if target_count == 1 {
        if let Some(dag_nodes) = try_hash_single_target_suffix(dfa, token, scratch) {
            let _ = dag_nodes;
        } else {
            ensure_target_gids_map(
                &mut scratch.target_gids,
                scratch.single_target_pos,
                scratch.single_target_gids.as_slice(),
            );
            hash_suffixes(dfa, token, scratch, false);
        }
    } else if target_count > 0 {
        hash_suffixes(dfa, token, scratch, false);
    }

    finish_token_signature(dfa, chunk_states, scratch)
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
    /// (state_idx, old_dirty_state_flag)
    dirty_state_flag_changes: Vec<(usize, u8)>,
}

impl DepthChangeLog {
    fn new() -> Self {
        Self {
            state_changes: Vec::new(),
            match_changes: Vec::new(),
            dirty_changes: Vec::new(),
            dirty_state_flag_changes: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.state_changes.clear();
        self.match_changes.clear();
        self.dirty_changes.clear();
        self.dirty_state_flag_changes.clear();
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
                    if scratch.dirty_state_flags[i] == 0 {
                        log.dirty_state_flag_changes
                            .push((i, scratch.dirty_state_flags[i]));
                        scratch.dirty_state_flags[i] = 1;
                    }
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
    for &(state_idx, old_flag) in log.dirty_state_flag_changes.iter().rev() {
        scratch.dirty_state_flags[state_idx] = old_flag;
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

/// Fast token signature when no dirty flags are set (no targets).
/// Uses 4-way loop unrolling to break the serial multiply-add dependency chain
/// in the original `finish_token_signature_no_cleanup`, reducing latency from
/// ~4 cycles/element to ~1 cycle/element on pipelined CPUs.
#[inline(never)]
fn finish_token_signature_clean(
    dfa: &Dfa,
    num_initial_states: usize,
    scratch: &Scratch,
) -> u64 {
    let c = HASH_SEED1;
    let c2 = c.wrapping_mul(c);
    let c3 = c2.wrapping_mul(c);
    let c4 = c3.wrapping_mul(c);
    let mut sig: u64 = HASH_SEED3;
    let n4 = (num_initial_states / 4) * 4;
    for i in (0..n4).step_by(4) {
        let x0 = dfa.completion(scratch.current_states[i]);
        let x1 = dfa.completion(scratch.current_states[i + 1]);
        let x2 = dfa.completion(scratch.current_states[i + 2]);
        let x3 = dfa.completion(scratch.current_states[i + 3]);
        let term = x0
            .wrapping_mul(c3)
            .wrapping_add(x1.wrapping_mul(c2))
            .wrapping_add(x2.wrapping_mul(c))
            .wrapping_add(x3);
        sig = sig.wrapping_mul(c4).wrapping_add(term);
    }
    for i in n4..num_initial_states {
        sig = sig
            .wrapping_mul(c)
            .wrapping_add(dfa.completion(scratch.current_states[i]));
    }
    sig
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
    let dag = &scratch.dag_nodes;
    let single_target_hash_pos = scratch.single_target_hash_pos;
    let single_target_hash = scratch.single_target_hash;
    let mut sig: u64 = HASH_SEED3;
    for i in 0..num_initial_states {
        let completion = dfa.completion(scratch.current_states[i]);
        let base = i * num_groups;
        let mask_base = i * dirty_words;

        let state_sig = if scratch.dirty_state_flags[i] != 0 {
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
                            dag.get(target)
                                .and_then(|node| node.as_ref())
                                .map_or(0, |node| node.hash)
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
    _profile: bool,
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
    for flag in scratch.dirty_state_flags[..batch_len].iter_mut() {
        *flag = 0;
    }
    for i in 0..batch_len {
        if dfa.is_dead_end[scratch.current_states[i]] {
            scratch.current_states[i] = STATE_NONE;
        }
    }

    let mut current_depth: usize = 0;
    let mut prev_token: &[u8] = &[];
    // dirty_state_flag_changes records states that transition 0→1 at each depth.
    let mut dirty_count: usize = 0;

    for &token_idx in chunk {
        let token = strings[token_idx].as_ref();
        let token_len = token.len();

        let lcp = prev_token
            .iter()
            .zip(token.iter())
            .take_while(|(left, right)| left == right)
            .count();

        if current_depth > lcp {
            for depth in (lcp..current_depth).rev() {
                dirty_count -= trie.depth_logs[depth].dirty_state_flag_changes.len();
            }
            dfs_backtrack(scratch, trie, current_depth, lcp);
        }

        for depth in lcp..token_len {
            dfs_step(dfa, scratch, trie, token[depth], depth, batch_len);
            dirty_count += trie.depth_logs[depth].dirty_state_flag_changes.len();
        }
        current_depth = token_len;

        let sig = if dirty_count == 0 {
            finish_token_signature_clean(dfa, batch_len, scratch)
        } else {
            scratch.targets.clear();
            scratch.target_gids.clear();
            scratch.single_target_pos = usize::MAX;
            scratch.single_target_gids.clear();
            scratch.single_target_hash_pos = usize::MAX;
            scratch.single_target_hash = 0;

            if state_group_size >= batch_len {
                collect_targets(
                    scratch,
                    num_groups,
                    token_len,
                    0,
                    batch_len,
                );
            } else {
                for state_offset in (0..batch_len).step_by(state_group_size) {
                    let group_len =
                        (state_offset + state_group_size).min(batch_len) - state_offset;
                    collect_targets(
                        scratch,
                        num_groups,
                        token_len,
                        state_offset,
                        group_len,
                    );
                }
            }

            let target_count = scratch.targets.len();
            if target_count == 1 {
                if try_hash_single_target_suffix(dfa, token, scratch).is_none() {
                    ensure_target_gids_map(
                        &mut scratch.target_gids,
                        scratch.single_target_pos,
                        scratch.single_target_gids.as_slice(),
                    );
                    hash_suffixes(dfa, token, scratch, false);
                }
            } else if target_count > 0 {
                hash_suffixes(dfa, token, scratch, false);
            }

            finish_token_signature_no_cleanup(dfa, batch_len, scratch)
        };

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
    find_vocab_equivalence_classes_with_group_filter(
        tokenizer, strings, initial_states, disallowed_follows, byte_to_class, None, None,
    )
}

/// Vocab equivalence with optional group filtering.
///
/// Minimum reduction ratio (compact/original) to trigger compaction.
const COMPACT_DFA_MAX_RATIO: f64 = 0.85;
/// Minimum number of initial states to consider compaction worthwhile.
const COMPACT_DFA_MIN_STATES: usize = 500;
/// Minimum work estimate (states × tokens) to justify the compaction overhead.
const COMPACT_DFA_MIN_WORK: usize = 10_000_000;

/// Build a compact DFA containing only states reachable from `initial_states`
/// via byte classes actually used by the partition's tokens.  Returns the
/// compact DFA and the remapped initial state indices, or None if compaction
/// is not beneficial.
fn compact_dfa_for_tokens<S: AsRef<[u8]>>(
    dfa: &Dfa,
    initial_states: &[usize],
    strings: &[S],
) -> Option<(Dfa, Vec<usize>)> {
    if initial_states.len() < COMPACT_DFA_MIN_STATES
        || strings.is_empty()
        || initial_states.len() * strings.len() < COMPACT_DFA_MIN_WORK
    {
        return None;
    }

    // Relevant byte classes from the partition's tokens.
    let mut byte_used = [false; 256];
    for s in strings {
        for &b in s.as_ref() {
            byte_used[b as usize] = true;
        }
    }
    let num_classes = dfa.byte_to_class.iter().copied().max().map_or(0usize, |m| m as usize + 1);
    let mut class_used = vec![false; num_classes];
    for b in 0..=255u8 {
        if byte_used[b as usize] {
            class_used[dfa.byte_to_class[b as usize] as usize] = true;
        }
    }
    let relevant_classes: Vec<usize> = class_used
        .iter()
        .enumerate()
        .filter_map(|(i, &u)| if u { Some(i) } else { None })
        .collect();

    // BFS from initial_states following only relevant-class transitions.
    let mut visited = vec![false; dfa.num_states];
    let mut queue = std::collections::VecDeque::new();
    for &s in initial_states {
        if s < dfa.num_states && !visited[s] {
            visited[s] = true;
            queue.push_back(s);
        }
    }
    while let Some(s) = queue.pop_front() {
        for &c in &relevant_classes {
            let t = dfa.trans_by_class[c * dfa.num_states + s];
            if t != NONE {
                let t = t as usize;
                if !visited[t] {
                    visited[t] = true;
                    queue.push_back(t);
                }
            }
        }
    }

    let restricted_reachable = visited.iter().filter(|&&v| v).count();
    let ratio = restricted_reachable as f64 / dfa.num_states as f64;
    if ratio > COMPACT_DFA_MAX_RATIO {
        return None;
    }

    // Build state remapping.
    let mut original_to_compact = vec![u32::MAX; dfa.num_states];
    let mut compact_to_original = Vec::with_capacity(restricted_reachable);
    for (i, &v) in visited.iter().enumerate() {
        if v {
            original_to_compact[i] = compact_to_original.len() as u32;
            compact_to_original.push(i);
        }
    }
    let compact_num = compact_to_original.len();

    // Build compact transition table (all classes, compact states).
    let mut compact_trans = vec![NONE; num_classes * compact_num];
    for c in 0..num_classes {
        let src_base = c * dfa.num_states;
        let dst_base = c * compact_num;
        for cs in 0..compact_num {
            let orig = compact_to_original[cs];
            let t = dfa.trans_by_class[src_base + orig];
            if t != NONE {
                let mapped = original_to_compact[t as usize];
                // Only keep transition if target is in the compact set.
                if mapped != u32::MAX {
                    compact_trans[dst_base + cs] = mapped;
                }
            }
        }
    }

    // Compact per-state arrays.
    let compact_finalizers: Vec<SmallVec<[usize; 4]>> =
        compact_to_original.iter().map(|&s| dfa.finalizers[s].clone()).collect();
    let compact_is_dead_end: Vec<bool> =
        compact_to_original.iter().map(|&s| dfa.is_dead_end[s]).collect();
    let compact_completion_hash: Vec<u64> =
        compact_to_original.iter().map(|&s| dfa.completion_hash[s]).collect();
    let compact_self_loop: Vec<U8Set> =
        compact_to_original.iter().map(|&s| dfa.self_loop_bytes[s]).collect();
    let compact_pfg: Vec<SmallVec<[usize; 4]>> =
        compact_to_original.iter().map(|&s| dfa.possible_future_groups[s].clone()).collect();

    // Remap initial states.
    let compact_initial: Vec<usize> = initial_states
        .iter()
        .map(|&s| original_to_compact[s] as usize)
        .collect();

    let compact_dfa = Dfa {
        start_state: if dfa.start_state < original_to_compact.len() && original_to_compact[dfa.start_state] != u32::MAX {
            original_to_compact[dfa.start_state] as usize
        } else {
            0
        },
        num_states: compact_num,
        byte_to_class: dfa.byte_to_class,
        trans_by_class: compact_trans,
        finalizers: compact_finalizers,
        is_dead_end: compact_is_dead_end,
        num_groups: dfa.num_groups,
        possible_future_groups: compact_pfg,
        completion_hash: compact_completion_hash,
        none_completion_hash: dfa.none_completion_hash,
        self_loop_bytes: compact_self_loop,
        disallowed_follows: dfa.disallowed_follows.clone(),
    };

    Some((compact_dfa, compact_initial))
}

/// When `active_groups` is provided, the DFA only tracks groups marked `true`.
/// L1 terminal groups can be excluded this way for a L2+-only analysis.
pub fn find_vocab_equivalence_classes_with_group_filter<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    byte_to_class: Option<&[u8; 256]>,
    active_groups: Option<&[bool]>,
    shared_cache: Option<&SharedVocabDfaCache>,
) -> VocabEquivalenceResult {
    let dfa = build_dfa_with_group_filter(tokenizer, disallowed_follows, byte_to_class, active_groups, shared_cache);

    // Compact DFA: restrict to states reachable from initial_states via the
    // partition's token bytes.  This can dramatically shrink the transition
    // table (e.g. 77260 → 36598 for p0 of o62058) improving cache locality.
    let compacted = compact_dfa_for_tokens(&dfa, initial_states, strings);
    let (dfa_ref, initial_states_ref): (&Dfa, &[usize]) = if let Some((ref cdfa, ref cstates)) = compacted {
        (cdfa, cstates)
    } else {
        (&dfa, initial_states)
    };

    let ordered_states = if diversity_state_order_enabled() {
        states_by_transition_diversity(dfa_ref, initial_states_ref)
    } else {
        initial_states_ref.to_vec()
    };
    let num_tokens = strings.len();
    let num_initial_states = ordered_states.len();

    if num_initial_states == 0 || num_tokens == 0 {
        return BTreeSet::from_iter(vec![(0..num_tokens).collect()]);
    }

    let num_groups = dfa_ref.num_groups;
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

    for (_batch_index, batch_start) in (0..num_initial_states).step_by(batch_size).enumerate() {
        if active_indices.is_empty() {
            break;
        }

        let batch_end = (batch_start + batch_size).min(num_initial_states);
        let batch = &ordered_states[batch_start..batch_end];
        let state_group_size = vocab_state_group_size(batch.len(), num_groups);
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
                            dfa_ref,
                            strings,
                            chunk,
                            batch,
                            state_group_size,
                            scratch,
                            trie_state,
                            false,
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
                                dfa_ref,
                                token,
                                batch,
                                state_group_size,
                                scratch,
                                false,
                            ),
                        )
                    },
                )
                .collect()
        };

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
    }

    let mut groups = vec![Vec::new(); next_class_id.max(1)];
    for (token_idx, &class_id) in partition.iter().enumerate() {
        groups[class_id].push(token_idx);
    }
    groups.into_iter().filter(|group| !group.is_empty()).collect()
}

