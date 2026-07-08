//! Fast vocab equivalence analysis based on DFA behavior signatures.
//!
//! Each token is classified by its match positions, suffix structure, and end
//! states across all tokenizer starts.

use super::super::compat::{compute_byte_classes, FlatDfa, FlatTransitionCache, TokenizerView};
use ahash::{AHasher, RandomState};
use hashbrown::HashMap;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use super::super::disallowed_follows::normalize_disallowed_follows;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::compiler::stages::id_map_and_terminal_dwa::types::compile_profile_enabled;

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
    trans_by_class: Arc<[u32]>,
    finalizers: Vec<SmallVec<[usize; 4]>>,
    is_dead_end: Vec<bool>,
    num_groups: usize,
    possible_future_groups: Vec<SmallVec<[usize; 4]>>,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
    /// Per-state bitset: which bytes cause a self-loop (transition back to same state).
    self_loop_bytes: Arc<[U8Set]>,
    disallowed_follows: Vec<BitSet>,
}

/// Precomputed transition-only data that is identical across partitions.
///
/// `filter_for_terminals` only changes finalizers and possible_future_group_ids,
/// not transitions. So the compressed transition layouts, `byte_to_class`, and
/// `self_loop_bytes` can be computed once and shared across all partition
/// equivalence calls.
pub struct SharedVocabDfaBase {
    byte_to_class: [u8; 256],
    pub num_classes: usize,
    class_representatives: Arc<[u8]>,
    /// The two dense compressed layouts are independently lazy. A reduced
    /// representative analysis can use the shared byte partition without
    /// paying to materialize raw-state layouts it never walks.
    trans_by_class: OnceLock<Arc<[u32]>>,
    /// Row-major transition table: `state * num_classes + class`.
    trans_by_state_class: OnceLock<Arc<[u32]>>,
    self_loop_bytes: Arc<[U8Set]>,
    none_completion_hash: u64,
    /// Exact source table for pointer-fast and value-exact compatibility checks.
    source_transitions: Arc<[u32]>,
}

fn compute_byte_classes_and_self_loops(
    dfa: &super::super::compat::FlatDfa,
) -> ([u8; 256], Vec<U8Set>) {
    let __probe = std::env::var_os("GLRMASK_PROBE_BYTE_CLASSES").is_some();
    let __t0 = std::time::Instant::now();
    // Exact byte-class discovery already requires a full DFA-table pass. Build
    // self-loop masks in that same pass rather than paying for another scan.
    //
    // Two independent 64-bit rolling hashes per byte-column form a 128-bit
    // column fingerprint. Two byte-columns are equal iff their fingerprints
    // match; a false collision between *distinct* columns has probability
    // ~2^-128 across 256 columns, so grouping purely by fingerprint is exact
    // in practice and removes the previous cache-hostile strided column
    // comparison entirely. The strict-reference validator remains the backstop.
    let mut hash_a = [0u64; 256];
    let mut hash_b = [0u64; 256];
    let mut self_loop_bytes = Vec::with_capacity(dfa.states.len());
    for state in 0..dfa.states.len() {
        let mut self_loops = U8Set::empty();
        let base = state * 256;
        for byte in 0..256usize {
            let target = dfa.transitions[base + byte];
            hash_a[byte] = hash_a[byte]
                .wrapping_mul(0x517cc1b727220a95)
                .wrapping_add(target as u64 + 1);
            hash_b[byte] = hash_b[byte]
                .wrapping_mul(0x9e3779b97f4a7c15)
                .wrapping_add((target as u64).rotate_left(17) ^ 0xD1B54A32D192ED03);
            if target == state as u32 {
                self_loops.insert(byte as u8);
            }
        }
        self_loop_bytes.push(self_loops);
    }
    let __hash_ms = __t0.elapsed().as_secs_f64() * 1000.0;
    let __t1 = std::time::Instant::now();

    let mut byte_to_class = [0u8; 256];
    let mut sorted_indices: [u8; 256] = std::array::from_fn(|i| i as u8);
    sorted_indices.sort_unstable_by_key(|&byte| (hash_a[byte as usize], hash_b[byte as usize]));
    let mut next_class = 0u8;
    byte_to_class[sorted_indices[0] as usize] = 0;
    for i in 1..256 {
        let current = sorted_indices[i] as usize;
        let previous = sorted_indices[i - 1] as usize;
        if hash_a[current] != hash_a[previous] || hash_b[current] != hash_b[previous] {
            next_class = next_class.wrapping_add(1);
        }
        byte_to_class[current] = next_class;
    }
    if __probe {
        let __disambig_ms = __t1.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[glrmask/probe][byte_classes] states={} num_classes={} hash_pass_ms={:.3} group_ms={:.3}",
            dfa.states.len(),
            next_class as usize + 1,
            __hash_ms,
            __disambig_ms,
        );
    }
    (byte_to_class, self_loop_bytes)
}

/// Like [`compute_byte_classes_and_self_loops`] but only distinguishes the
/// `relevant_bytes`. All irrelevant bytes are mapped to a single reserved
/// class (0); relevant bytes are grouped exactly by their 128-bit column
/// fingerprint into classes `1..`. The per-state scan touches only the
/// relevant columns, so cost is proportional to the number of relevant bytes.
///
/// Self-loop masks are likewise only populated for relevant bytes. This is
/// exact for callers that only ever walk relevant bytes (e.g. token-based L2P
/// equivalence): the collapsed irrelevant class is never indexed.
fn compute_byte_classes_and_self_loops_relevant(
    dfa: &super::super::compat::FlatDfa,
    relevant_bytes: &[bool; 256],
) -> ([u8; 256], Vec<U8Set>) {
    let relevant: Vec<usize> = (0..256usize).filter(|&byte| relevant_bytes[byte]).collect();
    let mut hash_a = [0u64; 256];
    let mut hash_b = [0u64; 256];
    let mut self_loop_bytes = Vec::with_capacity(dfa.states.len());
    for state in 0..dfa.states.len() {
        let mut self_loops = U8Set::empty();
        let base = state * 256;
        for &byte in &relevant {
            let target = dfa.transitions[base + byte];
            hash_a[byte] = hash_a[byte]
                .wrapping_mul(0x517cc1b727220a95)
                .wrapping_add(target as u64 + 1);
            hash_b[byte] = hash_b[byte]
                .wrapping_mul(0x9e3779b97f4a7c15)
                .wrapping_add((target as u64).rotate_left(17) ^ 0xD1B54A32D192ED03);
            if target == state as u32 {
                self_loops.insert(byte as u8);
            }
        }
        self_loop_bytes.push(self_loops);
    }

    // Class 0 is the reserved "irrelevant" dump class. Relevant bytes are
    // grouped by fingerprint into classes starting at 1.
    let mut byte_to_class = [0u8; 256];
    let mut sorted = relevant.clone();
    sorted.sort_unstable_by_key(|&byte| (hash_a[byte], hash_b[byte]));
    let mut next_class = 0u8;
    for (index, &byte) in sorted.iter().enumerate() {
        if index == 0
            || hash_a[byte] != hash_a[sorted[index - 1]]
            || hash_b[byte] != hash_b[sorted[index - 1]]
        {
            next_class = next_class.wrapping_add(1);
        }
        byte_to_class[byte] = next_class;
    }
    (byte_to_class, self_loop_bytes)
}

impl SharedVocabDfaBase {
    fn class_representatives(byte_to_class: &[u8; 256], num_classes: usize) -> Arc<[u8]> {
        let mut representatives = vec![0u8; num_classes];
        let mut seen = vec![false; num_classes];
        for byte in 0..=255u8 {
            let class = byte_to_class[byte as usize] as usize;
            if !seen[class] {
                seen[class] = true;
                representatives[class] = byte;
            }
        }
        Arc::from(representatives)
    }

    fn build_layout(&self, transposed: bool) -> Arc<[u32]> {
        let num_states = self.source_transitions.len() / 256;
        let mut transitions = vec![NONE; self.num_classes * num_states];
        for state in 0..num_states {
            let source_base = state * 256;
            for class in 0..self.num_classes {
                let target = self.source_transitions
                    [source_base + self.class_representatives[class] as usize];
                let destination = if transposed {
                    class * num_states + state
                } else {
                    state * self.num_classes + class
                };
                transitions[destination] = target;
            }
        }
        Arc::from(transitions)
    }

    /// Build from transition data derived from sparse lexer rows. Byte classes
    /// and self-loop bytes are shared immediately; the two dense layouts remain
    /// lazy until a caller actually needs their access pattern.
    pub fn build_from_flat_transition_cache(cache: &FlatTransitionCache) -> Self {
        let byte_to_class = cache.byte_to_class;
        let num_classes = byte_to_class
            .iter()
            .copied()
            .max()
            .map_or(0usize, |max_class| max_class as usize + 1);
        let none_completion_hash = {
            let mut h = new_hasher();
            h.write_u8(0);
            h.finish()
        };
        Self {
            byte_to_class,
            num_classes,
            class_representatives: Self::class_representatives(&byte_to_class, num_classes),
            trans_by_class: OnceLock::new(),
            trans_by_state_class: OnceLock::new(),
            self_loop_bytes: Arc::clone(&cache.self_loop_bytes),
            none_completion_hash,
            source_transitions: Arc::clone(&cache.transitions),
        }
    }

    /// Build from a FlatDfa. Called lazily via OnceLock on first use.
    pub fn build_from_dfa(dfa: &super::super::compat::FlatDfa) -> Self {
        let (byte_to_class, self_loop_bytes) = compute_byte_classes_and_self_loops(dfa);
        let num_classes = byte_to_class
            .iter()
            .copied()
            .max()
            .map_or(0usize, |max_class| max_class as usize + 1);
        let none_completion_hash = {
            let mut h = new_hasher();
            h.write_u8(0);
            h.finish()
        };
        Self {
            byte_to_class,
            num_classes,
            class_representatives: Self::class_representatives(&byte_to_class, num_classes),
            trans_by_class: OnceLock::new(),
            trans_by_state_class: OnceLock::new(),
            self_loop_bytes: Arc::from(self_loop_bytes),
            none_completion_hash,
            source_transitions: Arc::clone(&dfa.transitions),
        }
    }

    /// Build a byte-class base that only distinguishes the *relevant* bytes
    /// (those marked `true`), collapsing every other byte into a single class.
    ///
    /// This is exact for any analysis whose transition walks only ever follow
    /// relevant bytes — e.g. the C-seeded L2P state/vocab equivalence, which
    /// only steps through the partition's token bytes. Because irrelevant byte
    /// columns are never indexed, merging them is unobservable. Restricting the
    /// scan to relevant columns makes construction proportional to the number
    /// of relevant bytes instead of the full 256-wide alphabet.
    pub fn build_from_dfa_relevant(
        dfa: &super::super::compat::FlatDfa,
        relevant_bytes: &[bool; 256],
    ) -> Self {
        let (byte_to_class, self_loop_bytes) =
            compute_byte_classes_and_self_loops_relevant(dfa, relevant_bytes);
        let num_classes = byte_to_class
            .iter()
            .copied()
            .max()
            .map_or(0usize, |max_class| max_class as usize + 1);
        let none_completion_hash = {
            let mut h = new_hasher();
            h.write_u8(0);
            h.finish()
        };
        Self {
            byte_to_class,
            num_classes,
            class_representatives: Self::class_representatives(&byte_to_class, num_classes),
            trans_by_class: OnceLock::new(),
            trans_by_state_class: OnceLock::new(),
            self_loop_bytes: Arc::from(self_loop_bytes),
            none_completion_hash,
            source_transitions: Arc::clone(&dfa.transitions),
        }
    }

    /// Return the precomputed byte-to-class mapping.
    pub fn byte_to_class(&self) -> [u8; 256] {
        self.byte_to_class
    }

    /// Borrow the precomputed byte-to-class mapping for a compatible DFA.
    pub fn byte_to_class_ref(&self) -> &[u8; 256] {
        &self.byte_to_class
    }

    /// Borrow the class-major compressed layout for token-equivalence walks.
    pub fn transitions_by_class(&self) -> &[u32] {
        self.trans_by_class
            .get_or_init(|| self.build_layout(true))
            .as_ref()
    }

    pub fn transitions_by_class_arc(&self) -> Arc<[u32]> {
        Arc::clone(self.trans_by_class.get_or_init(|| self.build_layout(true)))
    }

    /// Borrow the row-major compressed layout for state-by-token walks.
    pub fn transitions_by_state_class(&self) -> &[u32] {
        self.trans_by_state_class
            .get_or_init(|| self.build_layout(false))
            .as_ref()
    }

    /// Check full compatibility without forcing either lazy layout.
    pub fn is_compatible_with_dfa(&self, dfa: &super::super::compat::FlatDfa) -> bool {
        let num_dfa_states = dfa.states.len();
        self.self_loop_bytes.len() == num_dfa_states
            && self.source_transitions.len() == dfa.transitions.len()
            && (Arc::ptr_eq(&self.source_transitions, &dfa.transitions)
                || self.source_transitions.as_ref() == dfa.transitions.as_ref())
    }
}

/// Cache type for lazy SharedVocabDfaBase initialization across partitions.
pub type SharedVocabDfaCache = std::sync::OnceLock<SharedVocabDfaBase>;

/// Deterministic stage-local identity for one immutable vocabulary-analysis
/// DFA. The stage fixes the source tokenizer and raw transition relation; this
/// key therefore contains only the view-dependent fields materialized into the
/// DFA itself.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AnalysisDfaCacheKey {
    num_groups: usize,
    active_group_len: usize,
    active_group_words: Vec<u64>,
    normalized_disallowed_follows: Vec<BitSet>,
}

fn dfa_num_groups(dfa: &super::super::compat::FlatDfa) -> usize {
    dfa.states
        .iter()
        .flat_map(|state| {
            state
                .finalizers
                .iter()
                .copied()
                .chain(state.possible_future_group_ids.iter().copied())
        })
        .max()
        .map_or(0, |group| group + 1)
}

fn pack_active_groups(active_groups: Option<&[bool]>, default_len: usize) -> (usize, Vec<u64>) {
    let active_len = active_groups.map_or(default_len, <[bool]>::len);
    let mut words = vec![0u64; active_len.div_ceil(64)];
    for group in 0..active_len {
        if active_groups.map_or(true, |groups| groups[group]) {
            words[group / 64] |= 1u64 << (group % 64);
        }
    }
    (active_len, words)
}

impl AnalysisDfaCacheKey {
    fn for_analysis_view(
        tokenizer: &TokenizerView,
        active_groups: Option<&[bool]>,
        effective_disallowed_follows: &BTreeMap<u32, BitSet>,
    ) -> Self {
        let num_groups = dfa_num_groups(tokenizer.dfa());
        let (active_group_len, active_group_words) = pack_active_groups(active_groups, num_groups);
        Self {
            num_groups,
            active_group_len,
            active_group_words,
            normalized_disallowed_follows: normalize_disallowed_follows(
                num_groups,
                effective_disallowed_follows,
            ),
        }
    }
}

/// Stage-local cache for immutable full vocabulary-analysis DFAs.
///
/// Source identity is fixed by the surrounding terminal-DWA stage. Entries
/// are separated by the canonical active-terminal view and the normalized
/// effective disallowed-follows relation. In particular, terminal
/// interchangeability changes neither the raw transition infrastructure nor
/// this cache's source identity.
#[derive(Default)]
pub struct SharedVocabAnalysisDfaCache {
    entries: Mutex<HashMap<AnalysisDfaCacheKey, Arc<OnceLock<Arc<Dfa>>>>>,
}

impl SharedVocabAnalysisDfaCache {
    fn get_or_init(
        &self,
        key: AnalysisDfaCacheKey,
        build: impl FnOnce() -> Dfa,
    ) -> Arc<Dfa> {
        let entry = {
            let mut entries = self.entries.lock().expect("vocab analysis DFA cache poisoned");
            entries
                .entry(key)
                .or_insert_with(|| Arc::new(OnceLock::new()))
                .clone()
        };
        Arc::clone(entry.get_or_init(|| Arc::new(build())))
    }
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
    dirty_state_flags: Vec<u8>,
    /// Per-state multi-word dirty bitmask.  Layout: `[dirty_words * num_states]`
    /// where state `i`'s dirty mask occupies indices `[i*dirty_words .. (i+1)*dirty_words]`.
    dirty_group_masks: Vec<u64>,
    /// Number of u64 words per state in `dirty_group_masks` (= ceil(num_groups/64)).
    dirty_words: usize,
    num_groups: usize,
    /// For the active DFS path, how many initial states have their latest
    /// match for `(position, group)` at this slot.
    trie_target_group_counts: Vec<u32>,
    /// Number of live groups at each token position in the active DFS path.
    trie_target_position_counts: Vec<u32>,
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
static VOCAB_ROW_CERT_DIAG: Lazy<bool> =
    Lazy::new(|| env_flag_enabled("GLRMASK_VOCAB_EQUIV_ROW_CERT_DIAG"));

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

fn vocab_verify_token_pair_override() -> Option<(usize, usize)> {
    let value = std::env::var("GLRMASK_VOCAB_VERIFY_TOKEN_PAIR").ok()?;
    let mut parts = value.split(',').map(str::trim);
    let left = parts.next()?.parse::<usize>().ok()?;
    let right = parts.next()?.parse::<usize>().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((left, right))
}

fn vocab_verify_token_pair_from_final_classes_enabled() -> bool {
    env_flag_enabled("GLRMASK_VOCAB_VERIFY_TOKEN_PAIR_FROM_FINAL_CLASSES")
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

    // Compute the raw group axis before filtering. The active view removes
    // observations but does not change the terminal-ID coordinate used by
    // follow constraints or the cache key.
    let num_groups = dfa_num_groups(dfa);

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
    let compatible_shared_base = shared_base.filter(|base| {
        base.is_compatible_with_dfa(dfa)
    });
    let (byte_to_class, trans_by_class, self_loop_bytes, none_completion_hash) =
        if let Some(base) = compatible_shared_base {
            let btc = byte_to_class_override.copied().unwrap_or(base.byte_to_class);
            (
                btc,
                base.transitions_by_class_arc(),
                Arc::clone(&base.self_loop_bytes),
                base.none_completion_hash,
            )
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

            (btc, Arc::from(tbc), Arc::from(slb), nch)
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
            num_groups,
            trie_target_group_counts: Vec::new(),
            trie_target_position_counts: Vec::new(),
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

    /// Reuse the allocation for a later state batch with the same vocabulary
    /// group axis. Token-signature cleanup leaves the active prefix clean.
    fn ensure_capacity(&mut self, num_states: usize, num_groups: usize) {
        let dirty_words = (num_groups + 63) / 64;
        if self.dirty_words != dirty_words || self.single_target_seen.len() != num_groups {
            *self = Self::new(num_states, num_groups);
            return;
        }
        if self.current_states.len() >= num_states {
            return;
        }
        self.current_states.resize(num_states, 0);
        self.match_positions.resize(num_states * num_groups, NONE);
        self.dirty_state_flags.resize(num_states, 0);
        self.dirty_group_masks
            .resize(num_states * dirty_words.max(1), 0);
    }
}

fn reset_trie_target_aggregate(scratch: &mut Scratch, max_token_len: usize) {
    let slots = (max_token_len + 1).saturating_mul(scratch.num_groups);
    scratch.trie_target_group_counts.resize(slots, 0);
    scratch.trie_target_group_counts[..slots].fill(0);
    scratch.trie_target_position_counts.resize(max_token_len + 1, 0);
    scratch.trie_target_position_counts[..=max_token_len].fill(0);
}

#[inline(always)]
fn update_trie_target_aggregate(
    scratch: &mut Scratch,
    gid: usize,
    previous_position: u32,
    next_position: u32,
) {
    if previous_position == next_position || gid >= scratch.num_groups {
        return;
    }

    let remove = |scratch: &mut Scratch, position: usize, gid: usize| {
        if position == 0 {
            return;
        }
        let index = position * scratch.num_groups + gid;
        debug_assert!(scratch.trie_target_group_counts[index] > 0);
        scratch.trie_target_group_counts[index] -= 1;
        if scratch.trie_target_group_counts[index] == 0 {
            debug_assert!(scratch.trie_target_position_counts[position] > 0);
            scratch.trie_target_position_counts[position] -= 1;
        }
    };
    let add = |scratch: &mut Scratch, position: usize, gid: usize| {
        if position == 0 {
            return;
        }
        let index = position * scratch.num_groups + gid;
        if scratch.trie_target_group_counts[index] == 0 {
            scratch.trie_target_position_counts[position] += 1;
        }
        scratch.trie_target_group_counts[index] += 1;
    };

    if previous_position != NONE {
        remove(scratch, previous_position as usize, gid);
    }
    if next_position != NONE {
        add(scratch, next_position as usize, gid);
    }
}

/// Materialize the union of live `(position, group)` pairs maintained during a
/// trie walk. This is exactly what `collect_targets` obtains by scanning every
/// state-local dirty mask, but its work is proportional to live positions.
fn collect_trie_targets(scratch: &mut Scratch, token_len: usize) {
    scratch.targets.clear();
    scratch.target_gids.clear();
    scratch.single_target_pos = usize::MAX;
    scratch.single_target_gids.clear();

    let mut targets_with_gids: SmallVec<[(usize, SmallVec<[usize; 16]>); 4]> = SmallVec::new();
    for position in 1..=token_len {
        if scratch.trie_target_position_counts[position] == 0 {
            continue;
        }
        let base = position * scratch.num_groups;
        let mut gids = SmallVec::<[usize; 16]>::new();
        for gid in 0..scratch.num_groups {
            if scratch.trie_target_group_counts[base + gid] != 0 {
                gids.push(gid);
            }
        }
        debug_assert!(!gids.is_empty());
        targets_with_gids.push((position, gids));
    }

    for (position, _) in &targets_with_gids {
        scratch.targets.push(*position);
    }
    if targets_with_gids.len() == 1 {
        let (position, gids) = targets_with_gids.pop().expect("single target exists");
        scratch.single_target_pos = position;
        scratch.single_target_gids = gids;
    } else {
        for (position, gids) in targets_with_gids {
            scratch.target_gids.insert(position, gids);
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

fn fill_state_observation_words_and_cleanup(
    dfa: &Dfa,
    batch_len: usize,
    scratch: &mut Scratch,
    out: &mut [u64],
) {
    let num_groups = dfa.num_groups;
    let dirty_words = scratch.dirty_words;
    let dag = &scratch.dag_nodes;
    let single_target_hash_pos = scratch.single_target_hash_pos;
    let single_target_hash = scratch.single_target_hash;

    for i in 0..batch_len {
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

        out[i] = state_sig;
    }
}

fn compute_token_state_observation_words(
    dfa: &Dfa,
    token: &[u8],
    batch: &[usize],
    state_group_size: usize,
    scratch: &mut Scratch,
    out: &mut [u64],
) {
    scratch.single_target_hash_pos = usize::MAX;
    scratch.single_target_hash = 0;
    run_batch(dfa, scratch, token, batch, state_group_size, false);

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

    fill_state_observation_words_and_cleanup(dfa, batch.len(), scratch, out);
}

fn first_distinguishing_state_for_token_pair_with_count<S: AsRef<[u8]>>(
    dfa: &Dfa,
    strings: &[S],
    left_token_idx: usize,
    right_token_idx: usize,
    ordered_states: &[usize],
    ordered_original_states: &[usize],
    batch_size: usize,
) -> (Option<usize>, usize) {
    if left_token_idx >= strings.len() || right_token_idx >= strings.len() {
        return (None, 0);
    }

    let left_token = strings[left_token_idx].as_ref();
    let right_token = strings[right_token_idx].as_ref();
    let mut states_checked = 0usize;
    let mut left_words = vec![0u64; batch_size.max(1)];
    let mut right_words = vec![0u64; batch_size.max(1)];

    for batch_start in (0..ordered_states.len()).step_by(batch_size.max(1)) {
        let batch_end = (batch_start + batch_size.max(1)).min(ordered_states.len());
        let batch = &ordered_states[batch_start..batch_end];
        let batch_len = batch.len();
        let state_group_size = vocab_state_group_size(batch_len, dfa.num_groups);
        let mut left_scratch = Scratch::new(batch_len, dfa.num_groups);
        let mut right_scratch = Scratch::new(batch_len, dfa.num_groups);

        compute_token_state_observation_words(
            dfa,
            left_token,
            batch,
            state_group_size,
            &mut left_scratch,
            &mut left_words[..batch_len],
        );
        compute_token_state_observation_words(
            dfa,
            right_token,
            batch,
            state_group_size,
            &mut right_scratch,
            &mut right_words[..batch_len],
        );

        for i in 0..batch_len {
            states_checked += 1;
            if left_words[i] != right_words[i] {
                return (Some(ordered_original_states[batch_start + i]), states_checked);
            }
        }
    }

    (None, states_checked)
}

fn first_distinguishing_state_for_token_pair<S: AsRef<[u8]>>(
    dfa: &Dfa,
    strings: &[S],
    left_token_idx: usize,
    right_token_idx: usize,
    ordered_states: &[usize],
    ordered_original_states: &[usize],
    batch_size: usize,
) -> Option<usize> {
    first_distinguishing_state_for_token_pair_with_count(
        dfa,
        strings,
        left_token_idx,
        right_token_idx,
        ordered_states,
        ordered_original_states,
        batch_size,
    )
    .0
}

fn log_vocab_pair_verification<S: AsRef<[u8]>>(
    dfa: &Dfa,
    strings: &[S],
    left_token_idx: usize,
    right_token_idx: usize,
    ordered_states: &[usize],
    ordered_original_states: &[usize],
    batch_size: usize,
) {
    if left_token_idx >= strings.len() || right_token_idx >= strings.len() {
        eprintln!(
            "[glrmask/profile][vocab_pair_verify] left={} right={} result=invalid states={} total_ms=0.000",
            left_token_idx,
            right_token_idx,
            ordered_states.len(),
        );
        return;
    }

    let started_at = std::time::Instant::now();
    let (witness_state, states_checked) = first_distinguishing_state_for_token_pair_with_count(
        dfa,
        strings,
        left_token_idx,
        right_token_idx,
        ordered_states,
        ordered_original_states,
        batch_size,
    );

    if let Some(witness_state) = witness_state {
        eprintln!(
            "[glrmask/profile][vocab_pair_verify] left={} right={} result=different witness_state={} states_checked={} total_ms={:.3}",
            left_token_idx,
            right_token_idx,
            witness_state,
            states_checked,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    } else {
        eprintln!(
            "[glrmask/profile][vocab_pair_verify] left={} right={} result=equivalent states={} total_ms={:.3}",
            left_token_idx,
            right_token_idx,
            ordered_states.len(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
}

fn run_vocab_row_cert_diag<S: AsRef<[u8]> + Sync>(
    dfa: &Dfa,
    strings: &[S],
    ordered_states: &[usize],
    batch_size: usize,
    state_group_size_for_batch: impl Fn(usize) -> usize,
    groups: &[Vec<usize>],
) {
    let started_at = std::time::Instant::now();
    let rep_token_indices: Vec<usize> = groups.iter().map(|group| group[0]).collect();
    let mut row_keys = vec![Vec::<u64>::with_capacity(rep_token_indices.len()); ordered_states.len()];

    for batch_start in (0..ordered_states.len()).step_by(batch_size.max(1)) {
        let batch_end = (batch_start + batch_size.max(1)).min(ordered_states.len());
        let batch = &ordered_states[batch_start..batch_end];
        let batch_len = batch.len();
        let state_group_size = state_group_size_for_batch(batch_len);
        let mut scratch = Scratch::new(batch_len, dfa.num_groups);
        let mut state_words = vec![0u64; batch_len];

        for &token_idx in &rep_token_indices {
            let token = strings[token_idx].as_ref();
            compute_token_state_observation_words(
                dfa,
                token,
                batch,
                state_group_size,
                &mut scratch,
                &mut state_words,
            );
            for i in 0..batch_len {
                row_keys[batch_start + i].push(state_words[i]);
            }
        }
    }

    let mut row_class_counts: HashMap<Vec<u64>, usize> =
        HashMap::with_capacity(row_keys.len() / 2);
    for row in row_keys {
        *row_class_counts.entry(row).or_insert(0) += 1;
    }

    let row_classes = row_class_counts.len();
    let largest_block = row_class_counts.values().copied().max().unwrap_or(0);
    let singleton_rows = row_class_counts
        .values()
        .copied()
        .filter(|&count| count == 1)
        .count();
    let reduction_pct = if ordered_states.is_empty() {
        0.0
    } else {
        100.0 * (1.0 - row_classes as f64 / ordered_states.len() as f64)
    };

    eprintln!(
        "[glrmask/profile][vocab_row_cert_diag] states={} rep_tokens={} row_classes={} reduction_pct={:.2} largest_block={} singleton_rows={} total_ms={:.3}",
        ordered_states.len(),
        rep_token_indices.len(),
        row_classes,
        reduction_pct,
        largest_block,
        singleton_rows,
        started_at.elapsed().as_secs_f64() * 1000.0,
    );
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

struct ScratchWorker {
    scratch: Scratch,
    trie_state: TrieWalkState,
}

#[derive(Default)]
struct ScratchPool {
    available: Mutex<Vec<ScratchWorker>>,
    allocations: AtomicUsize,
    reuses: AtomicUsize,
}

impl ScratchPool {
    fn checkout(&self, num_states: usize, num_groups: usize) -> ScratchLease<'_> {
        let worker = self.available.lock().unwrap().pop();
        let mut worker = match worker {
            Some(worker) => {
                self.reuses.fetch_add(1, Ordering::Relaxed);
                worker
            }
            None => {
                self.allocations.fetch_add(1, Ordering::Relaxed);
                ScratchWorker {
                    scratch: Scratch::new(num_states, num_groups),
                    trie_state: TrieWalkState::new(),
                }
            }
        };
        worker.scratch.ensure_capacity(num_states, num_groups);
        ScratchLease {
            pool: self,
            worker: Some(worker),
        }
    }

    fn stats(&self) -> (usize, usize) {
        (
            self.allocations.load(Ordering::Relaxed),
            self.reuses.load(Ordering::Relaxed),
        )
    }
}

struct ScratchLease<'a> {
    pool: &'a ScratchPool,
    worker: Option<ScratchWorker>,
}

impl ScratchLease<'_> {
    fn worker_mut(&mut self) -> &mut ScratchWorker {
        self.worker.as_mut().expect("scratch lease must hold its worker")
    }
}

impl Drop for ScratchLease<'_> {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            self.pool.available.lock().unwrap().push(worker);
        }
    }
}

#[derive(Clone, Copy, Default)]
struct TrieWalkChunkStats {
    dfs_step_ms: f64,
    collect_targets_ms: f64,
    single_target_suffix_ms: f64,
    multi_target_suffix_ms: f64,
    finish_signature_ms: f64,
    dfs_steps: usize,
    dfs_steps_without_new_dirty: usize,
    dfs_states_visited: usize,
    dfs_dead_transitions: usize,
    dfs_dead_without_new_dirty: usize,
    dfs_new_dirty_groups: usize,
    dfs_new_dirty_states: usize,
    clean_tokens: usize,
    dirty_tokens: usize,
    single_target_tokens: usize,
    multi_target_tokens: usize,
    total_targets: usize,
}

impl TrieWalkChunkStats {
    fn add_assign(&mut self, other: Self) {
        self.dfs_step_ms += other.dfs_step_ms;
        self.collect_targets_ms += other.collect_targets_ms;
        self.single_target_suffix_ms += other.single_target_suffix_ms;
        self.multi_target_suffix_ms += other.multi_target_suffix_ms;
        self.finish_signature_ms += other.finish_signature_ms;
        self.dfs_steps += other.dfs_steps;
        self.dfs_steps_without_new_dirty += other.dfs_steps_without_new_dirty;
        self.dfs_states_visited += other.dfs_states_visited;
        self.dfs_dead_transitions += other.dfs_dead_transitions;
        self.dfs_dead_without_new_dirty += other.dfs_dead_without_new_dirty;
        self.dfs_new_dirty_groups += other.dfs_new_dirty_groups;
        self.dfs_new_dirty_states += other.dfs_new_dirty_states;
        self.clean_tokens += other.clean_tokens;
        self.dirty_tokens += other.dirty_tokens;
        self.single_target_tokens += other.single_target_tokens;
        self.multi_target_tokens += other.multi_target_tokens;
        self.total_targets += other.total_targets;
    }
}

#[derive(Clone, Copy, Default)]
struct DfsStepStats {
    states_visited: usize,
    dead_transitions: usize,
    dead_without_new_dirty: usize,
    new_dirty_groups: usize,
    new_dirty_states: usize,
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
                update_trie_target_aggregate(scratch, gid, old_mp, position);
                scratch.match_positions[ix] = position;
            }
        }

        if dfa.is_dead_end[ns] {
            scratch.current_states[i] = STATE_NONE;
        }
    }
}

fn dfs_step_profiled(
    dfa: &Dfa,
    scratch: &mut Scratch,
    trie: &mut TrieWalkState,
    byte: u8,
    depth: usize,
    batch_len: usize,
) -> DfsStepStats {
    trie.ensure_depth(depth);
    let log = &mut trie.depth_logs[depth];
    log.clear();

    let mut step_stats = DfsStepStats::default();

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
        step_stats.states_visited += 1;

        let mut state_new_dirty_groups = 0usize;

        let next_state_raw =
            unsafe { *dfa.trans_by_class.get_unchecked(class_base + old_state) };
        if next_state_raw == NONE {
            log.state_changes.push((i, old_state));
            scratch.current_states[i] = STATE_NONE;
            step_stats.dead_transitions += 1;
            step_stats.dead_without_new_dirty += 1;
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
                        step_stats.new_dirty_states += 1;
                    }
                    scratch.dirty_group_masks[flat_idx] |= 1u64 << bit;
                    step_stats.new_dirty_groups += 1;
                    state_new_dirty_groups += 1;
                }
                log.match_changes.push((ix, old_mp));
                update_trie_target_aggregate(scratch, gid, old_mp, position);
                scratch.match_positions[ix] = position;
            }
        }

        if dfa.is_dead_end[ns] {
            scratch.current_states[i] = STATE_NONE;
            step_stats.dead_transitions += 1;
            if state_new_dirty_groups == 0 {
                step_stats.dead_without_new_dirty += 1;
            }
        }
    }

    step_stats
}

/// Undo all changes recorded at a single depth level.
fn dfs_undo_depth(scratch: &mut Scratch, log: &DepthChangeLog) {
    for &(ix, old_mp) in log.match_changes.iter().rev() {
        let current_mp = scratch.match_positions[ix];
        let gid = ix % scratch.num_groups;
        update_trie_target_aggregate(scratch, gid, current_mp, old_mp);
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
    profile: bool,
) -> (Vec<(usize, u64)>, TrieWalkChunkStats) {
    let batch_len = batch.len();
    let num_groups = dfa.num_groups;
    let mut results = Vec::with_capacity(chunk.len());
    let elapsed_ms = |started_at: Option<Instant>| {
        started_at.map_or(0.0, |instant| instant.elapsed().as_secs_f64() * 1000.0)
    };
    let mut stats = TrieWalkChunkStats::default();

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
    let max_token_len = chunk
        .iter()
        .map(|&token_idx| strings[token_idx].as_ref().len())
        .max()
        .unwrap_or(0);
    reset_trie_target_aggregate(scratch, max_token_len);

    let mut current_depth: usize = 0;
    let mut prev_token: &[u8] = &[];
    // dirty_state_flag_changes records states that transition 0→1 at each depth.
    let mut dirty_count: usize = 0;

    for &token_idx in chunk {
        let token = strings[token_idx].as_ref();
        let token_len = token.len();
        let dfs_started_at = profile.then(Instant::now);

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
            if profile {
                let step_stats =
                    dfs_step_profiled(dfa, scratch, trie, token[depth], depth, batch_len);
                dirty_count += trie.depth_logs[depth].dirty_state_flag_changes.len();
                stats.dfs_steps += 1;
                if step_stats.new_dirty_groups == 0 {
                    stats.dfs_steps_without_new_dirty += 1;
                }
                stats.dfs_states_visited += step_stats.states_visited;
                stats.dfs_dead_transitions += step_stats.dead_transitions;
                stats.dfs_dead_without_new_dirty += step_stats.dead_without_new_dirty;
                stats.dfs_new_dirty_groups += step_stats.new_dirty_groups;
                stats.dfs_new_dirty_states += step_stats.new_dirty_states;
            } else {
                dfs_step(dfa, scratch, trie, token[depth], depth, batch_len);
                dirty_count += trie.depth_logs[depth].dirty_state_flag_changes.len();
            }
        }
        stats.dfs_step_ms += elapsed_ms(dfs_started_at);
        current_depth = token_len;

        let sig = if dirty_count == 0 {
            if profile {
                stats.clean_tokens += 1;
            }
            let finish_started_at = profile.then(Instant::now);
            let signature = finish_token_signature_clean(dfa, batch_len, scratch);
            stats.finish_signature_ms += elapsed_ms(finish_started_at);
            signature
        } else {
            if profile {
                stats.dirty_tokens += 1;
            }
            scratch.single_target_hash_pos = usize::MAX;
            scratch.single_target_hash = 0;

            let collect_targets_started_at = profile.then(Instant::now);
            collect_trie_targets(scratch, token_len);
            stats.collect_targets_ms += elapsed_ms(collect_targets_started_at);

            let target_count = scratch.targets.len();
            if profile {
                stats.total_targets += target_count;
            }
            if target_count == 1 {
                if profile {
                    stats.single_target_tokens += 1;
                }
                let single_target_started_at = profile.then(Instant::now);
                if try_hash_single_target_suffix(dfa, token, scratch).is_none() {
                    ensure_target_gids_map(
                        &mut scratch.target_gids,
                        scratch.single_target_pos,
                        scratch.single_target_gids.as_slice(),
                    );
                    hash_suffixes(dfa, token, scratch, false);
                }
                stats.single_target_suffix_ms += elapsed_ms(single_target_started_at);
            } else if target_count > 0 {
                if profile {
                    stats.multi_target_tokens += 1;
                }
                let multi_target_started_at = profile.then(Instant::now);
                hash_suffixes(dfa, token, scratch, false);
                stats.multi_target_suffix_ms += elapsed_ms(multi_target_started_at);
            }

            let finish_started_at = profile.then(Instant::now);
            let signature = finish_token_signature_no_cleanup(dfa, batch_len, scratch);
            stats.finish_signature_ms += elapsed_ms(finish_started_at);
            signature
        };

        results.push((token_idx, sig));
        prev_token = token;
    }

    // Final backtrack to restore scratch to clean state
    if current_depth > 0 {
        dfs_backtrack(scratch, trie, current_depth, 0);
    }

    (results, stats)
}

/// Vocab equivalence with optional group filtering.
///
/// Minimum reduction ratio (compact/original) to trigger compaction.
const COMPACT_DFA_MAX_RATIO: f64 = 0.85;
/// Minimum number of initial states to consider compaction worthwhile.
const COMPACT_DFA_MIN_STATES: usize = 500;
/// Minimum work estimate (states × tokens) to justify the compaction overhead.
const COMPACT_DFA_MIN_WORK: usize = 10_000_000;
const INPUT_VIEW_COMPACT_MIN_STATES: usize = 512;
const INPUT_VIEW_COMPACT_MAX_RATIO: f64 = 0.85;
const INPUT_VIEW_COMPACT_MAX_TOKENS: usize = 16;

/// Restrict a tokenizer view to exactly the states reachable from the state
/// representatives and lexer start along bytes that appear in `strings`.
///
/// This happens before building the analysis DFA, rather than after it.  The
/// finite-vocabulary equivalence pass cannot observe any omitted transition:
/// every token walk starts from one of these roots and consumes only these
/// bytes.  Keeping the start root also preserves the suffix observations used
/// by the trellis hash.
fn compact_tokenizer_view_for_tokens<S: AsRef<[u8]>>(
    tokenizer: &TokenizerView,
    initial_states: &[usize],
    strings: &[S],
) -> Option<(TokenizerView, Vec<usize>, Vec<usize>)> {
    let source = tokenizer.dfa();
    if source.states.len() < INPUT_VIEW_COMPACT_MIN_STATES || strings.is_empty() {
        return None;
    }

    let mut relevant_bytes = [false; 256];
    for string in strings {
        for &byte in string.as_ref() {
            relevant_bytes[byte as usize] = true;
        }
    }
    if !relevant_bytes.iter().any(|&used| used) {
        return None;
    }

    let mut reachable = vec![false; source.states.len()];
    let mut queue = VecDeque::new();
    let visit = |state: usize, reachable: &mut [bool], queue: &mut VecDeque<usize>| {
        if state < reachable.len() && !reachable[state] {
            reachable[state] = true;
            queue.push_back(state);
        }
    };
    visit(source.start_state, &mut reachable, &mut queue);
    for &state in initial_states {
        visit(state, &mut reachable, &mut queue);
    }
    while let Some(state) = queue.pop_front() {
        for (byte, &used) in relevant_bytes.iter().enumerate() {
            if !used {
                continue;
            }
            let target = source.trans(state, byte);
            if target != NONE {
                visit(target as usize, &mut reachable, &mut queue);
            }
        }
    }

    let reachable_count = reachable.iter().filter(|&&state| state).count();
    if reachable_count as f64 / source.states.len() as f64 > INPUT_VIEW_COMPACT_MAX_RATIO {
        return None;
    }

    let mut original_to_compact = vec![NONE; source.states.len()];
    let mut compact_to_original = Vec::with_capacity(reachable_count);
    for (original, &is_reachable) in reachable.iter().enumerate() {
        if is_reachable {
            original_to_compact[original] = compact_to_original.len() as u32;
            compact_to_original.push(original);
        }
    }

    let mut transitions = vec![NONE; reachable_count * 256];
    let mut states = Vec::with_capacity(reachable_count);
    for (compact, &original) in compact_to_original.iter().enumerate() {
        states.push(source.states[original].clone());
        let compact_base = compact * 256;
        for (byte, &used) in relevant_bytes.iter().enumerate() {
            if !used {
                continue;
            }
            let target = source.trans(original, byte);
            if target != NONE {
                let mapped = original_to_compact[target as usize];
                debug_assert_ne!(mapped, NONE, "reachable token edge omitted from compact view");
                transitions[compact_base + byte] = mapped;
            }
        }
    }

    let compact_initial = initial_states
        .iter()
        .map(|&state| original_to_compact[state] as usize)
        .collect();
    let start_state = original_to_compact[source.start_state] as usize;
    Some((
        TokenizerView {
            flat_dfa: FlatDfa {
                states,
                start_state,
                transitions: Arc::from(transitions),
            },
        },
        compact_initial,
        compact_to_original,
    ))
}

/// Build a compact DFA containing only states reachable from `initial_states`
/// via byte classes actually used by the partition's tokens.  Returns the
/// compact DFA and the remapped initial state indices, or None if compaction
/// is not beneficial.
fn compact_dfa_for_tokens<S: AsRef<[u8]>>(
    dfa: &Dfa,
    initial_states: &[usize],
    strings: &[S],
) -> Option<(Dfa, Vec<usize>, Vec<usize>)> {
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
        trans_by_class: Arc::from(compact_trans),
        finalizers: compact_finalizers,
        is_dead_end: compact_is_dead_end,
        num_groups: dfa.num_groups,
        possible_future_groups: compact_pfg,
        completion_hash: compact_completion_hash,
        none_completion_hash: dfa.none_completion_hash,
        self_loop_bytes: Arc::from(compact_self_loop),
        disallowed_follows: dfa.disallowed_follows.clone(),
    };

    Some((compact_dfa, compact_initial, compact_to_original))
}

/// When `active_groups` is provided, the DFA only tracks groups marked `true`.
/// L1 terminal groups can be excluded this way for a L2+-only analysis.
pub(crate) fn find_vocab_equivalence_classes_with_group_filter_profiled<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    byte_to_class: Option<&[u8; 256]>,
    active_groups: Option<&[bool]>,
    shared_cache: Option<&SharedVocabDfaCache>,
    shared_analysis_dfa_cache: Option<&SharedVocabAnalysisDfaCache>,
) -> (VocabEquivalenceResult, f64) {
    let profiling = compile_profile_enabled();
    let elapsed_ms = |started_at: Option<Instant>| {
        started_at.map_or(0.0, |instant| instant.elapsed().as_secs_f64() * 1000.0)
    };

    let total_started_at = profiling.then(Instant::now);
    let build_dfa_started_at = Instant::now();
    let input_compacted = if shared_analysis_dfa_cache.is_none()
        && shared_cache.is_none()
        && active_groups.is_none()
        && strings.len() <= INPUT_VIEW_COMPACT_MAX_TOKENS
    {
        compact_tokenizer_view_for_tokens(tokenizer, initial_states, strings)
    } else {
        None
    };
    let (tokenizer_for_dfa, initial_states_for_dfa, input_compact_to_original): (
        &TokenizerView,
        &[usize],
        Option<&Vec<usize>>,
    ) = if let Some((ref compact_view, ref compact_initial, ref compact_to_original)) = input_compacted {
        (compact_view, compact_initial, Some(compact_to_original))
    } else {
        (tokenizer, initial_states, None)
    };
    let dfa = if let Some(cache) = shared_analysis_dfa_cache {
        let key = AnalysisDfaCacheKey::for_analysis_view(
            tokenizer_for_dfa,
            active_groups,
            disallowed_follows,
        );
        cache.get_or_init(key, || {
            build_dfa_with_group_filter(
                tokenizer_for_dfa,
                disallowed_follows,
                byte_to_class,
                active_groups,
                shared_cache,
            )
        })
    } else {
        Arc::new(build_dfa_with_group_filter(
            tokenizer_for_dfa,
            disallowed_follows,
            if input_compact_to_original.is_some() {
                None
            } else {
                byte_to_class
            },
            active_groups,
            shared_cache,
        ))
    };
    let build_dfa_ms = build_dfa_started_at.elapsed().as_secs_f64() * 1000.0;

    // Compact DFA: restrict to states reachable from initial_states via the
    // partition's token bytes.  This can dramatically shrink the transition
    // table (e.g. 77260 → 36598 for p0 of o62058) improving cache locality.
    let compact_dfa_started_at = profiling.then(Instant::now);
    let compacted = compact_dfa_for_tokens(dfa.as_ref(), initial_states_for_dfa, strings);
    let compact_dfa_ms = elapsed_ms(compact_dfa_started_at);
    let (dfa_ref, initial_states_ref, compact_to_original): (&Dfa, &[usize], Option<&Vec<usize>>) = if let Some((ref cdfa, ref cstates, ref compact_to_original)) = compacted {
        (cdfa, cstates, Some(compact_to_original))
    } else {
        (&dfa, initial_states_for_dfa, None)
    };
    let compacted_states = compact_to_original.map_or(dfa.num_states, |states| states.len());
    let num_tokens = strings.len();
    let num_initial_states = initial_states_ref.len();

    if num_initial_states == 0 || num_tokens == 0 {
        return (BTreeSet::from_iter(vec![(0..num_tokens).collect()]), build_dfa_ms);
    }

    if initial_states_ref
        .iter()
        .all(|&state| dfa_ref.is_dead_end[state])
    {
        if profiling {
            eprintln!(
                "[glrmask/profile][vocab_equiv] strings={} initial_states={} batches=0 used_trie_walk=false active_final={} original_states={} effective_states={} compacted={} build_dfa_ms={:.3} compact_dfa_ms={:.3} state_order_ms=0.000 sort_tokens_ms=0.000 signature_ms=0.000 refinement_ms=0.000 final_groups_ms=0.000 dfs_step_ms=0.000 collect_targets_ms=0.000 single_target_suffix_ms=0.000 multi_target_suffix_ms=0.000 finish_signature_ms=0.000 dfs_steps=0 dfs_steps_without_new_dirty=0 dfs_states_visited=0 dfs_dead_transitions=0 dfs_dead_without_new_dirty=0 dfs_new_dirty_groups=0 dfs_new_dirty_states=0 clean_tokens={} dirty_tokens=0 single_target_tokens=0 multi_target_tokens=0 total_targets=0 total_ms={:.3}",
                num_tokens,
                num_initial_states,
                num_tokens,
                dfa.num_states,
                compacted_states,
                compact_to_original.is_some(),
                build_dfa_ms,
                compact_dfa_ms,
                num_tokens,
                elapsed_ms(total_started_at),
            );
        }
        return (BTreeSet::from_iter(vec![(0..num_tokens).collect()]), build_dfa_ms);
    }

    let state_order_started_at = profiling.then(Instant::now);
    let ordered_states = if diversity_state_order_enabled() {
        states_by_transition_diversity(dfa_ref, initial_states_ref)
    } else {
        initial_states_ref.to_vec()
    };
    let state_order_ms = elapsed_ms(state_order_started_at);
    let ordered_original_states = if let Some(compact_to_original) = compact_to_original {
        ordered_states
            .iter()
            .map(|&state| {
                let input_state = compact_to_original[state];
                input_compact_to_original
                    .map(|source| source[input_state])
                    .unwrap_or(input_state)
            })
            .collect::<Vec<_>>()
    } else if let Some(input_compact_to_original) = input_compact_to_original {
        ordered_states
            .iter()
            .map(|&state| input_compact_to_original[state])
            .collect::<Vec<_>>()
    } else {
        ordered_states.clone()
    };

    let num_groups = dfa_ref.num_groups;
    // A 500-state default is empirically better for the TTFM tail than the
    // older memory-target-derived large batch when combined with the no-unit
    // GLR default; keep the env override for A/B.
    let default_batch_size = 500usize;
    let batch_size = vocab_batch_size_override()
        .unwrap_or_else(|| num_initial_states.min(default_batch_size));
    let mut active_indices: Vec<usize> = (0..num_tokens).collect();
    let mut partition = vec![0usize; num_tokens];
    let mut next_class_id = 1usize;
    let mut sort_tokens_ms = 0.0;
    let mut signature_ms = 0.0;
    let mut refinement_ms = 0.0;
    let mut batches = 0usize;
    let mut used_trie_walk = false;
    let mut trie_walk_stats = TrieWalkChunkStats::default();
    let scratch_pool = Arc::new(ScratchPool::default());

    // A single Rayon worker previously rebuilt the full scratch arena for every
    // state batch. The arena is deliberately reset by the signature routines,
    // so one owner can safely reuse it across all batches without altering the
    // parallel path or the refinement semantics.
    let single_threaded = rayon::current_num_threads() == 1;
    let mut single_thread_scratch = single_threaded.then(|| Scratch::new(batch_size, num_groups));
    let mut single_thread_trie = single_threaded.then(TrieWalkState::new);

    for (_batch_index, batch_start) in (0..num_initial_states).step_by(batch_size).enumerate() {
        if active_indices.is_empty() {
            break;
        }
        batches += 1;

        let batch_end = (batch_start + batch_size).min(num_initial_states);
        let batch = &ordered_states[batch_start..batch_end];
        let state_group_size = vocab_state_group_size(batch.len(), num_groups);
        let use_trie_walk = active_indices.len() >= TRIE_WALK_MIN_TOKENS
            && !*TRIE_WALK_DISABLED;
        used_trie_walk |= use_trie_walk;
        let active_sigs: Vec<(usize, u64)> = if use_trie_walk {
            let mut sorted_indices = active_indices.clone();
            let sort_started_at = profiling.then(Instant::now);
            sorted_indices.sort_unstable_by(|&a, &b| {
                strings[a].as_ref().cmp(strings[b].as_ref())
            });
            sort_tokens_ms += elapsed_ms(sort_started_at);
            let signature_started_at = profiling.then(Instant::now);
            let mut flat_results = Vec::with_capacity(sorted_indices.len());
            if let (Some(scratch), Some(trie_state)) = (
                single_thread_scratch.as_mut(),
                single_thread_trie.as_mut(),
            ) {
                let (chunk_result, chunk_stats) = trie_walk_chunk_signatures(
                    dfa_ref,
                    strings,
                    &sorted_indices,
                    batch,
                    state_group_size,
                    scratch,
                    trie_state,
                    profiling,
                );
                flat_results.extend(chunk_result);
                if profiling {
                    trie_walk_stats.add_assign(chunk_stats);
                }
            } else {
                let scratch_pool = Arc::clone(&scratch_pool);
                let chunk_results: Vec<(Vec<(usize, u64)>, TrieWalkChunkStats)> = sorted_indices
                    .par_chunks(TRIE_CHUNK_SIZE)
                    .map_init(
                        || scratch_pool.checkout(batch.len(), num_groups),
                        |lease, chunk| {
                            let worker = lease.worker_mut();
                            trie_walk_chunk_signatures(
                                dfa_ref,
                                strings,
                                chunk,
                                batch,
                                state_group_size,
                                &mut worker.scratch,
                                &mut worker.trie_state,
                                profiling,
                            )
                        },
                    )
                    .collect();
                for (chunk_result, chunk_stats) in chunk_results {
                    flat_results.extend(chunk_result);
                    if profiling {
                        trie_walk_stats.add_assign(chunk_stats);
                    }
                }
            }
            signature_ms += elapsed_ms(signature_started_at);
            flat_results
        } else {
            let signature_started_at = profiling.then(Instant::now);
            let result = if let Some(scratch) = single_thread_scratch.as_mut() {
                active_indices
                    .iter()
                    .map(|&token_idx| {
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
                    })
                    .collect()
            } else {
                let scratch_pool = Arc::clone(&scratch_pool);
                active_indices
                    .par_iter()
                    .map_init(
                        || scratch_pool.checkout(batch.len(), num_groups),
                        |lease, &token_idx| {
                            let token = strings[token_idx].as_ref();
                            let worker = lease.worker_mut();
                            (
                                token_idx,
                                token_signature(
                                    dfa_ref,
                                    token,
                                    batch,
                                    state_group_size,
                                    &mut worker.scratch,
                                    false,
                                ),
                            )
                        },
                    )
                    .collect()
            };
            signature_ms += elapsed_ms(signature_started_at);
            result
        };

        let refinement_started_at = profiling.then(Instant::now);
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
        refinement_ms += elapsed_ms(refinement_started_at);
    }

    let final_groups_started_at = profiling.then(Instant::now);
    let mut groups = vec![Vec::new(); next_class_id.max(1)];
    for (token_idx, &class_id) in partition.iter().enumerate() {
        groups[class_id].push(token_idx);
    }
    let groups: Vec<Vec<usize>> = groups.into_iter().filter(|group| !group.is_empty()).collect();
    let final_groups_ms = elapsed_ms(final_groups_started_at);

    if let Some((left_token_idx, right_token_idx)) = vocab_verify_token_pair_override() {
        log_vocab_pair_verification(
            dfa_ref,
            strings,
            left_token_idx,
            right_token_idx,
            &ordered_states,
            &ordered_original_states,
            batch_size,
        );
    }

    if vocab_verify_token_pair_from_final_classes_enabled() {
        if let Some(group) = groups.iter().find(|group| group.len() >= 2) {
            log_vocab_pair_verification(
                dfa_ref,
                strings,
                group[0],
                group[1],
                &ordered_states,
                &ordered_original_states,
                batch_size,
            );
        }
        if groups.len() >= 2 {
            log_vocab_pair_verification(
                dfa_ref,
                strings,
                groups[0][0],
                groups[1][0],
                &ordered_states,
                &ordered_original_states,
                batch_size,
            );
        }
    }

    if *VOCAB_ROW_CERT_DIAG {
        run_vocab_row_cert_diag(
            dfa_ref,
            strings,
            &ordered_states,
            batch_size,
            |batch_len| vocab_state_group_size(batch_len, num_groups),
            &groups,
        );
    }

    let (scratch_pool_allocations, scratch_pool_reuses) = scratch_pool.stats();

    if profiling {
        eprintln!(
            "[glrmask/profile][vocab_equiv] strings={} initial_states={} batches={} used_trie_walk={} active_final={} original_states={} effective_states={} compacted={} build_dfa_ms={:.3} compact_dfa_ms={:.3} state_order_ms={:.3} sort_tokens_ms={:.3} signature_ms={:.3} refinement_ms={:.3} final_groups_ms={:.3} dfs_step_ms={:.3} collect_targets_ms={:.3} single_target_suffix_ms={:.3} multi_target_suffix_ms={:.3} finish_signature_ms={:.3} dfs_steps={} dfs_steps_without_new_dirty={} dfs_states_visited={} dfs_dead_transitions={} dfs_dead_without_new_dirty={} dfs_new_dirty_groups={} dfs_new_dirty_states={} clean_tokens={} dirty_tokens={} single_target_tokens={} multi_target_tokens={} total_targets={} scratch_pool_allocations={} scratch_pool_reuses={} total_ms={:.3}",
            num_tokens,
            num_initial_states,
            batches,
            used_trie_walk,
            active_indices.len(),
            dfa.num_states,
            dfa_ref.num_states,
            compact_to_original.is_some(),
            build_dfa_ms,
            compact_dfa_ms,
            state_order_ms,
            sort_tokens_ms,
            signature_ms,
            refinement_ms,
            final_groups_ms,
            trie_walk_stats.dfs_step_ms,
            trie_walk_stats.collect_targets_ms,
            trie_walk_stats.single_target_suffix_ms,
            trie_walk_stats.multi_target_suffix_ms,
            trie_walk_stats.finish_signature_ms,
            trie_walk_stats.dfs_steps,
            trie_walk_stats.dfs_steps_without_new_dirty,
            trie_walk_stats.dfs_states_visited,
            trie_walk_stats.dfs_dead_transitions,
            trie_walk_stats.dfs_dead_without_new_dirty,
            trie_walk_stats.dfs_new_dirty_groups,
            trie_walk_stats.dfs_new_dirty_states,
            trie_walk_stats.clean_tokens,
            trie_walk_stats.dirty_tokens,
            trie_walk_stats.single_target_tokens,
            trie_walk_stats.multi_target_tokens,
            trie_walk_stats.total_targets,
            scratch_pool_allocations,
            scratch_pool_reuses,
            elapsed_ms(total_started_at),
        );
    }

    (groups.into_iter().collect(), build_dfa_ms)
}

/// Result-only compatibility entry point. Callers that need precise phase
/// accounting should use the profiled sibling above.
pub fn find_vocab_equivalence_classes_with_group_filter<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    byte_to_class: Option<&[u8; 256]>,
    active_groups: Option<&[bool]>,
    shared_cache: Option<&SharedVocabDfaCache>,
    shared_analysis_dfa_cache: Option<&SharedVocabAnalysisDfaCache>,
) -> VocabEquivalenceResult {
    find_vocab_equivalence_classes_with_group_filter_profiled(
        tokenizer,
        strings,
        initial_states,
        disallowed_follows,
        byte_to_class,
        active_groups,
        shared_cache,
        shared_analysis_dfa_cache,
    )
    .0
}

#[cfg(test)]
mod shared_base_tests {
    use super::*;
    use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::compat::{
        FlatDfa, FlatDfaState, TokenizerView,
    };
    use std::sync::Arc;

    fn sample_dfa() -> FlatDfa {
        let mut transitions = vec![u32::MAX; 3 * 256];
        transitions[b'a' as usize] = 1;
        transitions[b'b' as usize] = 2;
        transitions[256 + b'a' as usize] = 1;
        transitions[256 + b'b' as usize] = 2;
        transitions[2 * 256 + b'a' as usize] = 2;
        transitions[2 * 256 + b'b' as usize] = 1;
        FlatDfa {
            states: vec![
                FlatDfaState {
                    finalizers: vec![],
                    possible_future_group_ids: vec![0],
                },
                FlatDfaState {
                    finalizers: vec![0],
                    possible_future_group_ids: vec![0],
                },
                FlatDfaState {
                    finalizers: vec![1],
                    possible_future_group_ids: vec![0, 1],
                },
            ],
            start_state: 0,
            transitions: Arc::from(transitions),
        }
    }

    fn padded_sample_dfa() -> FlatDfa {
        let mut dfa = sample_dfa();
        let target_state_count = 600;
        dfa.states.resize_with(target_state_count, || FlatDfaState {
            finalizers: Vec::new(),
            possible_future_group_ids: Vec::new(),
        });
        let mut transitions = dfa.transitions.to_vec();
        transitions.resize(target_state_count * 256, u32::MAX);
        dfa.transitions = Arc::from(transitions);
        dfa
    }

    #[test]
    fn tiny_vocab_input_compaction_matches_uncompacted_analysis() {
        let view = TokenizerView {
            flat_dfa: padded_sample_dfa(),
        };
        let tokens: Vec<&[u8]> = vec![b"a", b"b", b"aa", b"ba"];
        let initial_states = vec![0usize, 1, 2];
        let disallowed = BTreeMap::<u32, BitSet>::new();

        let compacted = find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &initial_states,
            &disallowed,
            None,
            None,
            None,
            None,
        );
        let uncached_base = SharedVocabDfaCache::default();
        let uncompacted = find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &initial_states,
            &disallowed,
            None,
            None,
            Some(&uncached_base),
            None,
        );

        assert_eq!(compacted, uncompacted);
    }

    #[test]
    fn shared_analysis_dfa_cache_matches_uncached_vocab_equivalence() {
        let view = TokenizerView { flat_dfa: sample_dfa() };
        let tokens: Vec<&[u8]> = vec![b"a", b"b", b"aa", b"ba"];
        let initial_states = vec![0usize, 1, 2];
        let disallowed = BTreeMap::<u32, BitSet>::new();

        let uncached = find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &initial_states,
            &disallowed,
            None,
            None,
            None,
            None,
        );
        let cache = SharedVocabAnalysisDfaCache::default();
        let cached_first = find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &initial_states,
            &disallowed,
            None,
            None,
            None,
            Some(&cache),
        );
        let cached_hit = find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &initial_states,
            &disallowed,
            None,
            None,
            None,
            Some(&cache),
        );

        assert_eq!(cached_first, uncached);
        assert_eq!(cached_hit, uncached);
        assert_eq!(cache.entries.lock().unwrap().len(), 1);
    }

    #[test]
    fn shared_analysis_dfa_cache_keys_filtered_views_and_normalized_follows() {
        let view = TokenizerView { flat_dfa: sample_dfa() };
        let tokens: Vec<&[u8]> = vec![b"a", b"b", b"aa", b"ba"];
        let initial_states = vec![0usize, 1, 2];
        let cache = SharedVocabAnalysisDfaCache::default();

        let active_zero = [true, false];
        let active_all = [true, true];
        let absent_follows = BTreeMap::<u32, BitSet>::new();
        let explicit_empty_follows = BTreeMap::from([(0u32, BitSet::new(2))]);
        let mut blocked_row = BitSet::new(2);
        blocked_row.set(1);
        let blocked_follows = BTreeMap::from([(0u32, blocked_row)]);

        let cached_zero = find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &initial_states,
            &absent_follows,
            None,
            Some(&active_zero),
            None,
            Some(&cache),
        );
        let cached_zero_explicit_empty = find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &initial_states,
            &explicit_empty_follows,
            None,
            Some(&active_zero),
            None,
            Some(&cache),
        );
        assert_eq!(cached_zero_explicit_empty, cached_zero);
        assert_eq!(
            cache.entries.lock().unwrap().len(),
            1,
            "absent and explicit empty follow rows must canonicalize to one key",
        );

        let uncached_all = find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &initial_states,
            &absent_follows,
            None,
            Some(&active_all),
            None,
            None,
        );
        let cached_all = find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &initial_states,
            &absent_follows,
            None,
            Some(&active_all),
            None,
            Some(&cache),
        );
        assert_eq!(cached_all, uncached_all);
        assert_eq!(cache.entries.lock().unwrap().len(), 2);

        let uncached_blocked = find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &initial_states,
            &blocked_follows,
            None,
            Some(&active_all),
            None,
            None,
        );
        let cached_blocked = find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &initial_states,
            &blocked_follows,
            None,
            Some(&active_all),
            None,
            Some(&cache),
        );
        assert_eq!(cached_blocked, uncached_blocked);
        assert_eq!(cache.entries.lock().unwrap().len(), 3);
    }

    #[test]
    fn trie_target_aggregate_matches_dirty_mask_scan() {
        let mut scratch = Scratch::new(2, 3);
        reset_trie_target_aggregate(&mut scratch, 4);

        let set_match = |scratch: &mut Scratch, state: usize, gid: usize, position: u32| {
            let index = state * scratch.num_groups + gid;
            let previous = scratch.match_positions[index];
            if previous == NONE {
                mark_dirty_group(scratch, state, gid);
            }
            update_trie_target_aggregate(scratch, gid, previous, position);
            scratch.match_positions[index] = position;
        };

        set_match(&mut scratch, 0, 0, 1);
        set_match(&mut scratch, 1, 0, 2);
        set_match(&mut scratch, 0, 1, 3);
        set_match(&mut scratch, 1, 2, 4);
        // Replacing a state-local latest match must remove only its old pair.
        set_match(&mut scratch, 0, 0, 4);

        collect_targets(&mut scratch, 3, 4, 0, 2);
        let mut expected_targets = scratch.targets.clone();
        expected_targets.sort_unstable();
        let expected_target_gids = scratch.target_gids.clone();
        let expected_single_target_pos = scratch.single_target_pos;
        let expected_single_target_gids = scratch.single_target_gids.clone();

        collect_trie_targets(&mut scratch, 4);
        assert_eq!(scratch.targets, expected_targets);
        assert_eq!(scratch.target_gids, expected_target_gids);
        if expected_targets.len() == 1 {
            assert_eq!(scratch.single_target_pos, expected_single_target_pos);
            assert_eq!(scratch.single_target_gids, expected_single_target_gids);
        } else {
            // The legacy scan leaves these stale after materializing the map;
            // multi-target consumers use only `targets` and `target_gids`.
            assert_eq!(scratch.single_target_pos, usize::MAX);
            assert!(scratch.single_target_gids.is_empty());
        }
    }

    #[test]
    fn shared_base_row_major_layout_matches_flat_dfa() {
        let dfa = sample_dfa();
        let base = SharedVocabDfaBase::build_from_dfa(&dfa);
        let byte_to_class = base.byte_to_class_ref();
        let row_major = base.transitions_by_state_class();

        for state in 0..dfa.states.len() {
            for byte in 0..=255usize {
                let class = byte_to_class[byte] as usize;
                assert_eq!(
                    row_major[state * base.num_classes + class],
                    dfa.trans(state, byte),
                    "state={state}, byte={byte}"
                );
            }
        }
        assert!(base.is_compatible_with_dfa(&dfa));

        let independently_allocated = FlatDfa {
            states: dfa.states.clone(),
            start_state: dfa.start_state,
            transitions: Arc::from(dfa.transitions.to_vec()),
        };
        assert!(base.is_compatible_with_dfa(&independently_allocated));
    }
}

#[cfg(test)]
mod scratch_pool_tests {
    use super::*;

    #[test]
    fn scratch_keeps_same_group_buffers_for_smaller_batches() {
        let mut scratch = Scratch::new(8, 5);
        let positions_ptr = scratch.match_positions.as_ptr();
        scratch.ensure_capacity(3, 5);
        assert_eq!(scratch.match_positions.as_ptr(), positions_ptr);
        assert!(scratch.current_states.len() >= 8);

        scratch.ensure_capacity(12, 5);
        assert!(scratch.current_states.len() >= 12);
        assert!(scratch.match_positions.len() >= 12 * 5);

        scratch.ensure_capacity(4, 6);
        assert_eq!(scratch.current_states.len(), 4);
        assert_eq!(scratch.match_positions.len(), 4 * 6);
        assert!(scratch.match_positions.iter().all(|&value| value == NONE));
    }

    #[test]
    fn scratch_pool_returns_leased_workers() {
        let pool = ScratchPool::default();
        {
            let _lease = pool.checkout(4, 3);
        }
        {
            let _lease = pool.checkout(2, 3);
        }
        assert_eq!(pool.stats(), (1, 1));
    }
}
