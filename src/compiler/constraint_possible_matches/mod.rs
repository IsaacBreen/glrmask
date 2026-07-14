use crate::automata::lexer::Lexer;
use std::collections::BTreeMap;
use std::hash::Hasher;
use std::sync::Mutex;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::constraint_possible_matches::collector::{
    IntervalPossibleMatchMap, TerminalRangeGroup, TrieClassBuildResult,
};
use crate::compiler::pm_profile::elapsed_ms;
use crate::compiler::possible_matches::PossibleMatchesComputer;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap, MappedArtifact};
use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::compat::{
    FlatDfa, FlatDfaState, TokenizerView,
};
use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::vocab::fast as vocab_equivalence_analysis;
use crate::ds::bitset::BitSet;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::ds::weight::{shared_rangeset, Weight};
use crate::grammar::flat::TerminalID;
use crate::vocab::VocabDerivedArtifact;
use crate::Vocab;

pub(crate) mod collector;

pub(crate) type RuntimePossibleMatchesByTerminal = BTreeMap<TerminalID, Weight>;
pub(crate) type SignatureClassId = u32;
type StateTerminalLabel = (u32, TerminalID);

#[derive(Debug, Clone)]
pub(crate) struct PossibleMatchVocabMap {
    pub(crate) original_to_internal: Vec<u32>,
    pub(crate) internal_to_originals: Vec<Vec<u32>>,
}

// WARNING: terminal-DWA equivalence maps must never be reused for possible
// matches. Terminal-DWA equivalence does not imply possible-matches
// equivalence. This warning must never be removed under any circumstances.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConstraintPossibleMatchesConfig {
    defer_to_dynamic_mask: bool,
}

impl ConstraintPossibleMatchesConfig {
    /// Materialize the full possible-match table during compilation.
    pub(crate) const EAGER: Self = Self {
        defer_to_dynamic_mask: false,
    };
    /// Keep compile-time PM empty. Runtime static masking remains exact until
    /// token-start terminal exclusions appear, then falls back to the exact
    /// dynamic masker for that mask call.
    pub(crate) const DEFER_TO_DYNAMIC_MASK: Self = Self {
        defer_to_dynamic_mask: true,
    };
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ConstraintPossibleMatchesProfile {
    pub(crate) vocab_equiv_ms: f64,
    pub(crate) possible_matches_collect_ms: f64,
    pub(crate) possible_match_vocab_ms: f64,
}

#[derive(Debug)]
pub(crate) struct ConstraintPossibleMatchesComputation {
    pub(crate) mapped_possible_matches: MappedArtifact<RuntimePossibleMatchesByTerminal>,
    pub(crate) runtime_dynamic_vocab: RuntimeDynamicMaskVocabArtifacts,
    pub(crate) profile: ConstraintPossibleMatchesProfile,
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeDynamicMaskVocabArtifacts {
    pub(crate) trie: Arc<VocabPrefixTree>,
    pub(crate) token_aliases: Arc<Vec<Vec<u32>>>,
}

#[derive(Debug, Clone)]
struct OrderedVocab {
    original_slot_count: usize,
    ordered_to_originals: Arc<Vec<Vec<u32>>>,
    ordered_token_bytes: Vec<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct OrderedVocabTrieArtifacts {
    ordered_vocab: Arc<OrderedVocab>,
    trie: Arc<VocabPrefixTree>,
}

impl VocabDerivedArtifact for OrderedVocabTrieArtifacts {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OrderedVocabCacheFingerprint {
    token_count: usize,
    max_token_id: u32,
    total_bytes: usize,
    hash: u64,
}

#[derive(Debug, Clone)]
struct OrderedVocabCacheEntry {
    fingerprint: OrderedVocabCacheFingerprint,
    source_original_to_ordered: Arc<[u32]>,
    artifacts: OrderedVocabTrieArtifacts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OrderedVocabCacheStatus {
    Disabled,
    Hit,
    Miss,
}

impl OrderedVocabCacheStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Hit => "hit",
            Self::Miss => "miss",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct OrderedVocabCacheProfile {
    status: OrderedVocabCacheStatus,
    probe_ns: u128,
    verify_ns: u128,
    ordered_vocab_build_ns: u128,
    trie_build_ns: u128,
    cache_entries: usize,
    capacity: usize,
}

#[derive(Debug, Clone, Copy)]
struct SweepEvent {
    add: bool,
    group_id: u32,
}

#[derive(Debug, Clone)]
struct SweepGroup {
    label_ids: Box<[u32]>,
}

#[derive(Debug, Default, Clone, Copy)]
struct SweepBuildStats {
    used_state_classes: usize,
    terminal_groups: usize,
    terminal_labels: usize,
    group_label_refs: usize,
    total_intervals: usize,
    total_events: usize,
}

pub(crate) fn build_internal_token_bytes_from_groups(
    vocab: &Vocab,
    internal_to_originals: &[Vec<u32>],
) -> BTreeMap<u32, Vec<u8>> {
    internal_to_originals.iter().enumerate().filter_map(|(internal_token_id, originals)| {
        let bytes = originals.iter().find_map(|original| vocab.entries.get(original))?.clone();
        Some((internal_token_id as u32, bytes))
    }).collect()
}

fn build_ordered_vocab(token_bytes: &BTreeMap<u32, Vec<u8>>) -> OrderedVocab {
    let original_slot_count = token_bytes.keys().next_back().map(|token_id| *token_id as usize + 1).unwrap_or(0);
    let mut entries: Vec<(u32, &[u8])> = token_bytes
        .iter()
        .map(|(&token_id, bytes)| (token_id, bytes.as_slice()))
        .collect();
    entries.sort_unstable_by(|left, right| left.1.cmp(right.1).then_with(|| left.0.cmp(&right.0)));

    let mut ordered_to_originals = Vec::new();
    let mut ordered_token_bytes = Vec::new();
    let mut index = 0usize;
    while index < entries.len() {
        let bytes = entries[index].1;
        let mut originals = Vec::new();
        while index < entries.len() && entries[index].1 == bytes {
            originals.push(entries[index].0);
            index += 1;
        }
        originals.sort_unstable();
        originals.dedup();
        ordered_token_bytes.push(bytes.to_vec());
        ordered_to_originals.push(originals);
    }

    OrderedVocab {
        original_slot_count,
        ordered_to_originals: Arc::new(ordered_to_originals),
        ordered_token_bytes,
    }
}

fn build_ordered_vocab_prefix_tree(ordered_vocab: &OrderedVocab) -> VocabPrefixTree {
    let entries: Vec<(usize, &[u8])> = ordered_vocab.ordered_token_bytes.iter().enumerate().map(|(ordered_id, bytes)| (ordered_id, bytes.as_slice())).collect();
    VocabPrefixTree::build_presorted(&entries)
}

fn ordered_vocab_cache_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_PM_ORDERED_VOCAB_CACHE")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true)
    })
}

fn ordered_vocab_cache_capacity() -> usize {
    static CAPACITY: OnceLock<usize> = OnceLock::new();
    *CAPACITY.get_or_init(|| {
        std::env::var("GLRMASK_PM_ORDERED_VOCAB_CACHE_CAPACITY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(4)
    })
}

fn ordered_vocab_cache() -> &'static Mutex<Vec<OrderedVocabCacheEntry>> {
    static CACHE: OnceLock<Mutex<Vec<OrderedVocabCacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn ordered_vocab_cache_fingerprint(
    token_bytes: &BTreeMap<u32, Vec<u8>>,
) -> OrderedVocabCacheFingerprint {
    let mut hasher = rustc_hash::FxHasher::default();
    let mut token_count = 0usize;
    let mut max_token_id = 0u32;
    let mut total_bytes = 0usize;
    for (&token_id, bytes) in token_bytes {
        hasher.write_u32(token_id);
        hasher.write_usize(bytes.len());
        hasher.write(bytes);
        token_count += 1;
        max_token_id = token_id;
        total_bytes += bytes.len();
    }
    OrderedVocabCacheFingerprint {
        token_count,
        max_token_id,
        total_bytes,
        hash: hasher.finish(),
    }
}

fn ordered_vocab_cache_source_matches(
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    source_original_to_ordered: &[u32],
    ordered_vocab: &OrderedVocab,
) -> bool {
    if ordered_vocab.ordered_token_bytes.len() != ordered_vocab.ordered_to_originals.len() {
        return false;
    }

    let cached_token_count: usize = ordered_vocab
        .ordered_to_originals
        .iter()
        .map(|originals| originals.len())
        .sum();
    if token_bytes.len() != cached_token_count {
        return false;
    }

    let actual_slot_count = token_bytes
        .keys()
        .next_back()
        .map(|token_id| *token_id as usize + 1)
        .unwrap_or(0);
    if actual_slot_count != ordered_vocab.original_slot_count {
        return false;
    }

    if source_original_to_ordered.len() != ordered_vocab.original_slot_count {
        return false;
    }

    for (&original_id, actual_bytes) in token_bytes {
        let Some(&ordered_id) = source_original_to_ordered.get(original_id as usize) else {
            return false;
        };
        let Some(cached_bytes) = ordered_vocab.ordered_token_bytes.get(ordered_id as usize) else {
            return false;
        };
        if actual_bytes != cached_bytes {
            return false;
        }
    }

    true
}

fn ordered_vocab_cache_source_original_to_ordered(
    ordered_vocab: &OrderedVocab,
) -> Arc<[u32]> {
    let mut original_to_ordered = vec![u32::MAX; ordered_vocab.original_slot_count];
    for (ordered_id, originals) in ordered_vocab.ordered_to_originals.iter().enumerate() {
        for &original_id in originals {
            let slot = &mut original_to_ordered[original_id as usize];
            debug_assert_eq!(*slot, u32::MAX);
            *slot = ordered_id as u32;
        }
    }
    original_to_ordered.into()
}

fn compile_profile_requested() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

fn emit_ordered_vocab_cache_profile(profile: OrderedVocabCacheProfile) {
    if !compile_profile_requested() {
        return;
    }
    eprintln!(
        "[glrmask/profile][ordered_vocab_cache] status={} probe_ms={:.3} verify_ms={:.3} ordered_vocab_ms={:.3} vocab_prefix_tree_ms={:.3} cache_entries={} capacity={}",
        profile.status.as_str(),
        profile.probe_ns as f64 / 1_000_000.0,
        profile.verify_ns as f64 / 1_000_000.0,
        profile.ordered_vocab_build_ns as f64 / 1_000_000.0,
        profile.trie_build_ns as f64 / 1_000_000.0,
        profile.cache_entries,
        profile.capacity,
    );
}

fn get_ordered_vocab_trie_artifacts(
    token_bytes: &BTreeMap<u32, Vec<u8>>,
) -> (OrderedVocabTrieArtifacts, OrderedVocabCacheProfile) {
    let capacity = ordered_vocab_cache_capacity();
    if !ordered_vocab_cache_enabled() || capacity == 0 {
        let ordered_vocab_started_at = Instant::now();
        let ordered_vocab = Arc::new(build_ordered_vocab(token_bytes));
        let ordered_vocab_build_ns = ordered_vocab_started_at.elapsed().as_nanos();
        let trie_started_at = Instant::now();
        let trie = Arc::new(build_ordered_vocab_prefix_tree(ordered_vocab.as_ref()));
        let trie_build_ns = trie_started_at.elapsed().as_nanos();
        return (
            OrderedVocabTrieArtifacts { ordered_vocab, trie },
            OrderedVocabCacheProfile {
                status: OrderedVocabCacheStatus::Disabled,
                probe_ns: 0,
                verify_ns: 0,
                ordered_vocab_build_ns,
                trie_build_ns,
                cache_entries: 0,
                capacity,
            },
        );
    }

    let probe_started_at = Instant::now();
    let fingerprint = ordered_vocab_cache_fingerprint(token_bytes);
    let mut verify_ns = 0u128;

    {
        let mut cache = ordered_vocab_cache().lock().unwrap();
        let mut hit_index = None;
        for (index, entry) in cache.iter().enumerate() {
            if entry.fingerprint != fingerprint {
                continue;
            }
            let verify_started_at = Instant::now();
            let is_match = ordered_vocab_cache_source_matches(
                token_bytes,
                entry.source_original_to_ordered.as_ref(),
                entry.artifacts.ordered_vocab.as_ref(),
            );
            verify_ns += verify_started_at.elapsed().as_nanos();
            if is_match {
                hit_index = Some(index);
                break;
            }
        }

        if let Some(index) = hit_index {
            let entry = cache.remove(index);
            let artifacts = entry.artifacts.clone();
            cache.push(entry);
            let cache_entries = cache.len();
            return (
                artifacts,
                OrderedVocabCacheProfile {
                    status: OrderedVocabCacheStatus::Hit,
                    probe_ns: probe_started_at.elapsed().as_nanos(),
                    verify_ns,
                    ordered_vocab_build_ns: 0,
                    trie_build_ns: 0,
                    cache_entries,
                    capacity,
                },
            );
        }
    }

    let ordered_vocab_started_at = Instant::now();
    let ordered_vocab = Arc::new(build_ordered_vocab(token_bytes));
    let ordered_vocab_build_ns = ordered_vocab_started_at.elapsed().as_nanos();
    let trie_started_at = Instant::now();
    let trie = Arc::new(build_ordered_vocab_prefix_tree(ordered_vocab.as_ref()));
    let trie_build_ns = trie_started_at.elapsed().as_nanos();
    let source_original_to_ordered = ordered_vocab_cache_source_original_to_ordered(ordered_vocab.as_ref());
    let entry = OrderedVocabCacheEntry {
        fingerprint,
        source_original_to_ordered,
        artifacts: OrderedVocabTrieArtifacts {
            ordered_vocab: Arc::clone(&ordered_vocab),
            trie: Arc::clone(&trie),
        },
    };

    let cache_entries = {
        let mut cache = ordered_vocab_cache().lock().unwrap();
        if cache.len() >= capacity {
            cache.remove(0);
        }
        cache.push(entry);
        cache.len()
    };

    (
        OrderedVocabTrieArtifacts { ordered_vocab, trie },
        OrderedVocabCacheProfile {
            status: OrderedVocabCacheStatus::Miss,
            probe_ns: probe_started_at.elapsed().as_nanos(),
            verify_ns,
            ordered_vocab_build_ns,
            trie_build_ns,
            cache_entries,
            capacity,
        },
    )
}

fn get_ordered_vocab_trie_artifacts_for_vocab(
    vocab: &Vocab,
) -> (OrderedVocabTrieArtifacts, OrderedVocabCacheProfile) {
    let capacity = ordered_vocab_cache_capacity();
    if !ordered_vocab_cache_enabled() || capacity == 0 {
        return get_ordered_vocab_trie_artifacts(&*vocab.entries);
    }

    let probe_started_at = Instant::now();
    if let Some(artifacts) = vocab.vocab_derived_cache_get::<OrderedVocabTrieArtifacts>() {
        return (
            artifacts.as_ref().clone(),
            OrderedVocabCacheProfile {
                status: OrderedVocabCacheStatus::Hit,
                probe_ns: probe_started_at.elapsed().as_nanos(),
                verify_ns: 0,
                ordered_vocab_build_ns: 0,
                trie_build_ns: 0,
                cache_entries: 1,
                capacity,
            },
        );
    }

    let ordered_vocab_started_at = Instant::now();
    let ordered_vocab = Arc::new(build_ordered_vocab(&*vocab.entries));
    let ordered_vocab_build_ns = ordered_vocab_started_at.elapsed().as_nanos();
    let trie_started_at = Instant::now();
    let trie = Arc::new(build_ordered_vocab_prefix_tree(ordered_vocab.as_ref()));
    let trie_build_ns = trie_started_at.elapsed().as_nanos();
    let artifacts = OrderedVocabTrieArtifacts { ordered_vocab, trie };
    vocab.vocab_derived_cache_set(Arc::new(artifacts.clone()));

    (
        artifacts,
        OrderedVocabCacheProfile {
            status: OrderedVocabCacheStatus::Miss,
            probe_ns: probe_started_at.elapsed().as_nanos(),
            verify_ns: 0,
            ordered_vocab_build_ns,
            trie_build_ns,
            cache_entries: 1,
            capacity,
        },
    )
}

#[allow(dead_code)]
pub(crate) fn dense_word_count(token_slots: u32) -> usize { (token_slots as usize + 63) / 64 }

#[allow(dead_code)]
pub(crate) fn max_original_token_slot(token_bytes: &BTreeMap<u32, Vec<u8>>) -> u32 {
    token_bytes.keys().next_back().map(|token_id| token_id.saturating_add(1)).unwrap_or(0)
}

fn range_set_from_sorted_ids(ids: &[u32]) -> RangeSetBlaze<u32> {
    let Some((&first, rest)) = ids.split_first() else { return RangeSetBlaze::new(); };
    let mut ranges = Vec::new();
    let mut start = first;
    let mut end = first;
    for &id in rest {
        if id == end + 1 { end = id; }
        else { ranges.push(start..=end); start = id; end = id; }
    }
    ranges.push(start..=end);
    RangeSetBlaze::from_iter(ranges)
}

fn range_set_from_u128_mask(mask: u128) -> RangeSetBlaze<u32> {
    if mask == 0 {
        return RangeSetBlaze::new();
    }

    let mut ranges = Vec::new();
    let mut bits = mask;
    while bits != 0 {
        let start = bits.trailing_zeros();
        let mut end = start;
        bits &= !(1u128 << start);
        while bits != 0 {
            let next = bits.trailing_zeros();
            if next != end + 1 {
                break;
            }
            end = next;
            bits &= !(1u128 << next);
        }
        ranges.push(start..=end);
    }

    RangeSetBlaze::from_iter(ranges)
}

#[inline]
fn pm_vocab_equiv_enabled() -> bool {
    std::env::var("GLRMASK_PM_VOCAB_EQUIV")
        .map(|value| {
            let trimmed = value.trim();
            trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(false)
}

#[inline]
fn pm_vocab_equiv_supported(tokenizer: &Tokenizer) -> bool {
    let _ = tokenizer;
    true
}

#[derive(Clone, Copy)]
struct PmTokenOutcome {
    terminals: u128,
    end_state: u32,
}

#[derive(Clone, Copy)]
struct NfaPmTokenOutcome {
    terminal_set: u32,
    end_config: u32,
}

#[derive(Default)]
struct NfaPmAnalysis<'a> {
    tokenizer: Option<&'a Tokenizer>,
    config_ids: FxHashMap<Vec<u32>, u32>,
    configs: Vec<Box<[u32]>>,
    transitions: FxHashMap<(u32, u8), u32>,
    terminal_set_ids: FxHashMap<Vec<u64>, u32>,
    terminal_sets: Vec<Box<[u64]>>,
    union_cache: FxHashMap<(u32, u32), u32>,
}

impl<'a> NfaPmAnalysis<'a> {
    fn new(tokenizer: &'a Tokenizer) -> Self {
        let mut analysis = Self {
            tokenizer: Some(tokenizer),
            ..Self::default()
        };
        analysis.intern_terminal_set(Vec::new());
        analysis
    }

    #[inline]
    fn tokenizer(&self) -> &'a Tokenizer {
        self.tokenizer.expect("NFA PM tokenizer missing")
    }

    fn intern_config(&mut self, states: Vec<u32>) -> u32 {
        if let Some(&id) = self.config_ids.get(&states) {
            return id;
        }
        let id = self.configs.len() as u32;
        self.config_ids.insert(states.clone(), id);
        self.configs.push(states.into_boxed_slice());
        id
    }

    fn config_for_raw_state(&mut self, raw_state: u32) -> u32 {
        let closure = self.tokenizer().execute_from_state_end_only(&[], raw_state);
        self.intern_config(closure.to_vec())
    }

    fn step_config(&mut self, config: u32, byte: u8) -> u32 {
        if let Some(&target) = self.transitions.get(&(config, byte)) {
            return target;
        }
        let states = self.configs[config as usize].to_vec();
        let targets = self.tokenizer().step_all(&states, byte);
        let target = if targets.is_empty() {
            u32::MAX
        } else {
            self.intern_config(targets.to_vec())
        };
        self.transitions.insert((config, byte), target);
        target
    }

    fn intern_terminal_set(&mut self, words: Vec<u64>) -> u32 {
        if let Some(&id) = self.terminal_set_ids.get(&words) {
            return id;
        }
        let id = self.terminal_sets.len() as u32;
        self.terminal_set_ids.insert(words.clone(), id);
        self.terminal_sets.push(words.into_boxed_slice());
        id
    }

    fn matched_terminal_set_for_config(&mut self, config: u32) -> u32 {
        let word_count = (self.tokenizer().num_terminals() as usize).div_ceil(64);
        let mut words = vec![0u64; word_count];
        for &state in self.configs[config as usize].iter() {
            for terminal in self.tokenizer().matched_terminals_iter(state) {
                let terminal = terminal as usize;
                words[terminal >> 6] |= 1u64 << (terminal & 63);
            }
        }
        self.intern_terminal_set(words)
    }

    fn union_terminal_sets(&mut self, left: u32, right: u32) -> u32 {
        if left == 0 {
            return right;
        }
        if right == 0 || left == right {
            return left;
        }
        let key = if left < right { (left, right) } else { (right, left) };
        if let Some(&id) = self.union_cache.get(&key) {
            return id;
        }
        let left_words = self.terminal_sets[left as usize].to_vec();
        let right_words = &self.terminal_sets[right as usize];
        let mut words = left_words;
        if words.len() < right_words.len() {
            words.resize(right_words.len(), 0);
        }
        for (word, &right_word) in words.iter_mut().zip(right_words.iter()) {
            *word |= right_word;
        }
        let id = self.intern_terminal_set(words);
        self.union_cache.insert(key, id);
        id
    }

    fn advance_outcomes(
        &mut self,
        parent: &[NfaPmTokenOutcome],
        segment: &[u8],
    ) -> Vec<NfaPmTokenOutcome> {
        let mut child = Vec::with_capacity(parent.len());
        for &outcome in parent {
            let mut terminal_set = outcome.terminal_set;
            let mut current_config = outcome.end_config;
            if current_config != u32::MAX {
                for &byte in segment {
                    current_config = self.step_config(current_config, byte);
                    if current_config == u32::MAX {
                        break;
                    }
                    let matched = self.matched_terminal_set_for_config(current_config);
                    terminal_set = self.union_terminal_sets(terminal_set, matched);
                }
            }
            child.push(NfaPmTokenOutcome {
                terminal_set,
                end_config: current_config,
            });
        }
        child
    }
}

struct NfaPmVocabEquivBuilder<'a, 'b> {
    ordered_vocab: &'a OrderedVocab,
    analysis: &'b mut NfaPmAnalysis<'a>,
    signature_buckets: FxHashMap<u64, Vec<u32>>,
    signatures: Vec<Vec<u32>>,
    original_to_internal: Vec<u32>,
    internal_to_originals: Vec<Vec<u32>>,
    representative_original_ids: Vec<u32>,
}

impl<'a, 'b> NfaPmVocabEquivBuilder<'a, 'b> {
    fn new(ordered_vocab: &'a OrderedVocab, analysis: &'b mut NfaPmAnalysis<'a>) -> Self {
        Self {
            ordered_vocab,
            analysis,
            signature_buckets: FxHashMap::default(),
            signatures: Vec::new(),
            original_to_internal: vec![u32::MAX; ordered_vocab.original_slot_count],
            internal_to_originals: Vec::new(),
            representative_original_ids: Vec::new(),
        }
    }

    fn signature_hash(outcomes: &[NfaPmTokenOutcome]) -> u64 {
        outcomes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, outcome| {
            mix_pm_signature_word(hash, outcome.terminal_set as u64)
        })
    }

    fn intern_signature(&mut self, outcomes: &[NfaPmTokenOutcome]) -> u32 {
        let hash = Self::signature_hash(outcomes);
        if let Some(bucket) = self.signature_buckets.get(&hash) {
            for &signature_id in bucket {
                let signature = &self.signatures[signature_id as usize];
                if signature.len() == outcomes.len()
                    && signature
                        .iter()
                        .zip(outcomes.iter())
                        .all(|(&left, right)| left == right.terminal_set)
                {
                    return signature_id;
                }
            }
        }
        let signature_id = self.signatures.len() as u32;
        self.signatures.push(outcomes.iter().map(|outcome| outcome.terminal_set).collect());
        self.signature_buckets.entry(hash).or_default().push(signature_id);
        signature_id
    }

    fn record_token(&mut self, ordered_token_id: usize, outcomes: &[NfaPmTokenOutcome]) {
        let class_id = self.intern_signature(outcomes);
        let class_idx = class_id as usize;
        while self.internal_to_originals.len() <= class_idx {
            self.internal_to_originals.push(Vec::new());
            self.representative_original_ids.push(u32::MAX);
        }
        let Some(originals) = self.ordered_vocab.ordered_to_originals.get(ordered_token_id) else {
            return;
        };
        for &original in originals {
            if let Some(slot) = self.original_to_internal.get_mut(original as usize) {
                *slot = class_id;
            }
            if self.representative_original_ids[class_idx] == u32::MAX {
                self.representative_original_ids[class_idx] = original;
            }
            self.internal_to_originals[class_idx].push(original);
        }
    }

    fn visit(&mut self, node: &VocabPrefixTreeNode, outcomes: &[NfaPmTokenOutcome]) {
        if node.has_token() {
            self.record_token(node.token_id(), outcomes);
        }
        for (segment, child) in node.iter_children() {
            let child_outcomes = self.analysis.advance_outcomes(outcomes, segment);
            self.visit(child, &child_outcomes);
        }
    }

    fn finish(mut self) -> ManyToOneIdMap {
        for originals in &mut self.internal_to_originals {
            originals.sort_unstable();
            originals.dedup();
        }
        ManyToOneIdMap {
            original_to_internal: self.original_to_internal,
            internal_to_originals: self.internal_to_originals,
            representative_original_ids: self.representative_original_ids,
        }
    }
}

#[inline]
fn mix_pm_signature_word(hash: u64, word: u64) -> u64 {
    let mixed = word.wrapping_add(0x9e37_79b9_7f4a_7c15);
    hash ^ mixed
        .wrapping_add(hash << 6)
        .wrapping_add(hash >> 2)
        .wrapping_mul(0x517c_c1b7_2722_0a95)
}

fn pm_signature_hash(outcomes: &[PmTokenOutcome]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for outcome in outcomes {
        hash = mix_pm_signature_word(hash, outcome.terminals as u64);
        hash = mix_pm_signature_word(hash, (outcome.terminals >> 64) as u64);
    }
    hash
}

fn pm_signature_matches(signature: &[u128], outcomes: &[PmTokenOutcome]) -> bool {
    signature.len() == outcomes.len()
        && signature
            .iter()
            .zip(outcomes.iter())
            .all(|(&left, right)| left == right.terminals)
}

fn intern_pm_token_signature(
    outcomes: &[PmTokenOutcome],
    buckets: &mut FxHashMap<u64, Vec<u32>>,
    signatures: &mut Vec<Vec<u128>>,
) -> u32 {
    let hash = pm_signature_hash(outcomes);
    if let Some(bucket) = buckets.get(&hash) {
        for &signature_id in bucket {
            if pm_signature_matches(&signatures[signature_id as usize], outcomes) {
                return signature_id;
            }
        }
    }

    let signature_id = signatures.len() as u32;
    let signature = outcomes
        .iter()
        .map(|outcome| outcome.terminals)
        .collect::<Vec<_>>();
    signatures.push(signature);
    buckets.entry(hash).or_default().push(signature_id);
    signature_id
}

fn advance_pm_token_outcomes(
    parent: &[PmTokenOutcome],
    segment: &[u8],
    byte_transitions: &[Vec<u32>],
    matched_terminal_masks: &[u128],
) -> Vec<PmTokenOutcome> {
    let mut child = Vec::with_capacity(parent.len());
    for &outcome in parent {
        let mut terminals = outcome.terminals;
        let mut current_state = outcome.end_state;
        if current_state != u32::MAX {
            for &byte in segment {
                let next_state = byte_transitions[byte as usize][current_state as usize];
                if next_state == u32::MAX {
                    current_state = u32::MAX;
                    break;
                }
                current_state = next_state;
                terminals |= matched_terminal_masks[current_state as usize];
            }
        }
        child.push(PmTokenOutcome {
            terminals,
            end_state: current_state,
        });
    }
    child
}

struct PmVocabEquivBuilder<'a> {
    ordered_vocab: &'a OrderedVocab,
    byte_transitions: &'a [Vec<u32>],
    matched_terminal_masks: &'a [u128],
    signature_buckets: FxHashMap<u64, Vec<u32>>,
    signatures: Vec<Vec<u128>>,
    original_to_internal: Vec<u32>,
    internal_to_originals: Vec<Vec<u32>>,
    representative_original_ids: Vec<u32>,
}

impl<'a> PmVocabEquivBuilder<'a> {
    fn new(
        ordered_vocab: &'a OrderedVocab,
        byte_transitions: &'a [Vec<u32>],
        matched_terminal_masks: &'a [u128],
    ) -> Self {
        Self {
            ordered_vocab,
            byte_transitions,
            matched_terminal_masks,
            signature_buckets: FxHashMap::default(),
            signatures: Vec::new(),
            original_to_internal: vec![u32::MAX; ordered_vocab.original_slot_count],
            internal_to_originals: Vec::new(),
            representative_original_ids: Vec::new(),
        }
    }

    fn record_token(&mut self, ordered_token_id: usize, outcomes: &[PmTokenOutcome]) {
        let class_id = intern_pm_token_signature(
            outcomes,
            &mut self.signature_buckets,
            &mut self.signatures,
        );
        let class_idx = class_id as usize;
        while self.internal_to_originals.len() <= class_idx {
            self.internal_to_originals.push(Vec::new());
            self.representative_original_ids.push(u32::MAX);
        }
        let Some(originals) = self.ordered_vocab.ordered_to_originals.get(ordered_token_id) else {
            return;
        };
        for &original in originals {
            if let Some(slot) = self.original_to_internal.get_mut(original as usize) {
                *slot = class_id;
            }
            if self.representative_original_ids[class_idx] == u32::MAX {
                self.representative_original_ids[class_idx] = original;
            }
            self.internal_to_originals[class_idx].push(original);
        }
    }

    fn visit(&mut self, node: &VocabPrefixTreeNode, outcomes: &[PmTokenOutcome]) {
        if node.has_token() {
            self.record_token(node.token_id(), outcomes);
        }
        for (segment, child) in node.iter_children() {
            let child_outcomes = advance_pm_token_outcomes(
                outcomes,
                segment,
                self.byte_transitions,
                self.matched_terminal_masks,
            );
            self.visit(child, &child_outcomes);
        }
    }

    fn finish(mut self) -> ManyToOneIdMap {
        for originals in &mut self.internal_to_originals {
            originals.sort_unstable();
            originals.dedup();
        }
        ManyToOneIdMap {
            original_to_internal: self.original_to_internal,
            internal_to_originals: self.internal_to_originals,
            representative_original_ids: self.representative_original_ids,
        }
    }
}

fn compute_pm_vocab_equivalence_map(
    tokenizer: &Tokenizer,
    ordered_vocab: &OrderedVocab,
    trie: &VocabPrefixTree,
) -> ManyToOneIdMap {
    if tokenizer.has_epsilon_transitions() {
        let mut analysis = NfaPmAnalysis::new(tokenizer);
        let root_outcomes = (0..tokenizer.num_states())
            .map(|state| NfaPmTokenOutcome {
                terminal_set: 0,
                end_config: analysis.config_for_raw_state(state),
            })
            .collect::<Vec<_>>();
        let mut builder = NfaPmVocabEquivBuilder::new(ordered_vocab, &mut analysis);
        builder.visit(&trie.root, &root_outcomes);
        return builder.finish();
    }
    let num_states = tokenizer.num_states() as usize;
    let mut byte_transitions = vec![vec![u32::MAX; num_states]; 256];
    for state_idx in 0..num_states {
        for (byte, target) in tokenizer.transitions_from(state_idx as u32) {
            byte_transitions[byte as usize][state_idx] = target;
        }
    }

    let mut matched_terminal_masks = Vec::with_capacity(num_states);
    for state in 0..tokenizer.num_states() {
        let mut mask = 0u128;
        for terminal in tokenizer.matched_terminals_iter(state) {
            if terminal < 128 {
                mask |= 1u128 << terminal;
            }
        }
        matched_terminal_masks.push(mask);
    }

    // For a deterministic-component dispatch, the synthetic reset state's PM
    // behavior is the union of the component-root behaviors.  Equality at all
    // physical component states therefore implies equality at the dispatch
    // state, so including it as a dead scalar row would be both unnecessary
    // and incorrect.
    let dispatch_start = tokenizer
        .has_deterministic_dispatch()
        .then(|| tokenizer.start_state());
    let root_outcomes = (0..tokenizer.num_states())
        .filter(|state| Some(*state) != dispatch_start)
        .map(|state| PmTokenOutcome {
            terminals: 0,
            end_state: state,
        })
        .collect::<Vec<_>>();
    let mut builder = PmVocabEquivBuilder::new(
        ordered_vocab,
        &byte_transitions,
        &matched_terminal_masks,
    );
    builder.visit(&trie.root, &root_outcomes);
    builder.finish()
}

fn compute_pm_vocab_equivalence_map_fast(
    tokenizer: &Tokenizer,
    ordered_vocab: &OrderedVocab,
) -> ManyToOneIdMap {
    let num_states = tokenizer.num_states() as usize;
    let mut transitions = vec![u32::MAX; num_states * 256];
    let states = (0..num_states)
        .map(|state_idx| {
            let base = state_idx * 256;
            for (byte, target) in tokenizer.transitions_from(state_idx as u32) {
                transitions[base + byte as usize] = target;
            }
            FlatDfaState {
                finalizers: tokenizer
                    .matched_terminals_iter(state_idx as u32)
                    .map(|terminal| terminal as usize)
                    .collect(),
                // The equivalence signature is based on terminals actually
                // reached while consuming a token, but the fast walker also
                // uses future groups to identify genuinely dead states.  An
                // empty vector here marks every non-final state dead and can
                // collapse the entire vocabulary into one bogus class.
                possible_future_group_ids: tokenizer
                    .possible_future_terminals_iter(state_idx as u32)
                    .map(|terminal| terminal as usize)
                    .collect(),
            }
        })
        .collect::<Vec<_>>();
    let tokenizer_view = TokenizerView {
        flat_dfa: FlatDfa {
            states,
            start_state: tokenizer.start_state() as usize,
            transitions: Arc::from(transitions),
        },
    };
    let strings = ordered_vocab
        .ordered_token_bytes
        .iter()
        .map(|bytes| bytes.as_slice())
        .collect::<Vec<_>>();
    let dispatch_start = tokenizer
        .has_deterministic_dispatch()
        .then(|| tokenizer.start_state() as usize);
    let initial_states = (0..tokenizer.num_states() as usize)
        .filter(|state| Some(*state) != dispatch_start)
        .collect::<Vec<_>>();
    let disallowed_follows = BTreeMap::<u32, BitSet>::new();
    let classes = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter(
        &tokenizer_view,
        &strings,
        &initial_states,
        &disallowed_follows,
        None,
        None,
        None,
        None,
    );

    let mut original_to_internal = vec![u32::MAX; ordered_vocab.original_slot_count];
    let mut internal_to_originals = Vec::new();
    let mut representative_original_ids = Vec::new();
    for class in classes {
        let internal = internal_to_originals.len() as u32;
        let mut originals = Vec::new();
        for ordered_id in class {
            if let Some(ordered_originals) = ordered_vocab.ordered_to_originals.get(ordered_id) {
                for &original in ordered_originals {
                    if let Some(slot) = original_to_internal.get_mut(original as usize) {
                        *slot = internal;
                    }
                    originals.push(original);
                }
            }
        }
        originals.sort_unstable();
        originals.dedup();
        let representative = originals.first().copied().unwrap_or(u32::MAX);
        internal_to_originals.push(originals);
        representative_original_ids.push(representative);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

fn used_state_class_ids(state_classes: &[u32]) -> Vec<u32> {
    let mut ids: Vec<u32> = state_classes.iter().copied().filter(|&class_id| class_id != u32::MAX).collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn next_nonzero_stamp(generation: &mut u32, stamps: &mut [u32]) -> u32 {
    *generation = generation.wrapping_add(1);
    if *generation == 0 {
        stamps.fill(0);
        *generation = 1;
    }
    *generation
}

fn push_sweep_event(events: &mut [Vec<SweepEvent>], event_positions: &mut Vec<u32>, position: u32, event: SweepEvent) {
    let Some(bucket) = events.get_mut(position as usize) else { return; };
    if bucket.is_empty() { event_positions.push(position); }
    bucket.push(event);
}

fn intern_state_terminal_label(
    label_ids: &mut FxHashMap<StateTerminalLabel, u32>,
    labels_by_id: &mut Vec<StateTerminalLabel>,
    label: StateTerminalLabel,
) -> u32 {
    if let Some(&label_id) = label_ids.get(&label) {
        label_id
    } else {
        let label_id = labels_by_id.len() as u32;
        labels_by_id.push(label);
        label_ids.insert(label, label_id);
        label_id
    }
}

fn build_sweep_events(
    class_maps: &[Arc<IntervalPossibleMatchMap>],
    state_classes: &[u32],
    num_ordered_tokens: usize,
) -> (Vec<Vec<SweepEvent>>, Vec<u32>, Vec<SweepGroup>, Vec<StateTerminalLabel>, SweepBuildStats) {
    let mut events = vec![Vec::new(); num_ordered_tokens + 1];
    let mut event_positions = Vec::new();
    let mut groups = Vec::<SweepGroup>::new();
    let mut labels_by_id = Vec::<StateTerminalLabel>::new();
    let mut label_ids = FxHashMap::<StateTerminalLabel, u32>::default();
    let mut stats = SweepBuildStats::default();

    let used_state_classes = used_state_class_ids(state_classes);
    stats.used_state_classes = used_state_classes.len();

    for class_id in used_state_classes {
        let Some(class_map) = class_maps.get(class_id as usize) else { continue; };
        for entry in class_map.iter() {
            if entry.terminals.is_empty() || entry.ranges.is_empty() { continue; }

            let mut group_label_ids = Vec::with_capacity(entry.terminals.len());
            for &terminal_id in entry.terminals.iter() {
                group_label_ids.push(intern_state_terminal_label(&mut label_ids, &mut labels_by_id, (class_id, terminal_id)));
            }
            group_label_ids.sort_unstable();
            group_label_ids.dedup();
            if group_label_ids.is_empty() { continue; }

            let group_id = groups.len() as u32;
            stats.group_label_refs += group_label_ids.len();
            groups.push(SweepGroup { label_ids: group_label_ids.into_boxed_slice() });

            for &(lo, mut hi) in entry.ranges.iter() {
                if num_ordered_tokens == 0 { continue; }
                let max_token = num_ordered_tokens as u32 - 1;
                if lo > max_token { continue; }
                hi = hi.min(max_token);
                if lo > hi { continue; }
                stats.total_intervals += 1;
                push_sweep_event(&mut events, &mut event_positions, lo, SweepEvent { add: true, group_id });
                stats.total_events += 1;
                let after = hi.saturating_add(1);
                if after <= num_ordered_tokens as u32 {
                    push_sweep_event(&mut events, &mut event_positions, after, SweepEvent { add: false, group_id });
                    stats.total_events += 1;
                }
            }
        }
    }

    event_positions.sort_unstable();
    event_positions.dedup();
    stats.terminal_groups = groups.len();
    stats.terminal_labels = labels_by_id.len();
    (events, event_positions, groups, labels_by_id, stats)
}

#[inline]
fn active_group_hash(group_id: u32) -> u64 {
    let mut value = (group_id as u64).wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn insert_active_group_id(
    active_group_ids: &mut Vec<u32>,
    active_group_positions: &mut [u32],
    active_group_fingerprint: &mut u64,
    group_id: u32,
) {
    let slot = &mut active_group_positions[group_id as usize];
    if *slot != u32::MAX {
        return;
    }
    *slot = active_group_ids.len() as u32;
    active_group_ids.push(group_id);
    *active_group_fingerprint ^= active_group_hash(group_id);
}

fn remove_active_group_id(
    active_group_ids: &mut Vec<u32>,
    active_group_positions: &mut [u32],
    active_group_fingerprint: &mut u64,
    group_id: u32,
) {
    let remove_index = active_group_positions[group_id as usize] as usize;
    debug_assert!(remove_index < active_group_ids.len());
    let removed_group_id = active_group_ids.swap_remove(remove_index);
    debug_assert_eq!(removed_group_id, group_id);
    if remove_index < active_group_ids.len() {
        let moved_group_id = active_group_ids[remove_index];
        active_group_positions[moved_group_id as usize] = remove_index as u32;
    }
    active_group_positions[group_id as usize] = u32::MAX;
    *active_group_fingerprint ^= active_group_hash(group_id);
}

fn apply_sweep_events(
    active_group_counts: &mut [u32],
    events: &[SweepEvent],
    active_group_ids: &mut Vec<u32>,
    active_group_positions: &mut [u32],
    active_group_fingerprint: &mut u64,
) {
    for event in events.iter().filter(|event| !event.add) {
        let count = &mut active_group_counts[event.group_id as usize];
        assert!(*count > 0, "pmv sweep removal underflow for group_id={}", event.group_id);
        if *count == 1 {
            remove_active_group_id(active_group_ids, active_group_positions, active_group_fingerprint, event.group_id);
        }
        *count -= 1;
    }
    for event in events.iter().filter(|event| event.add) {
        let count = &mut active_group_counts[event.group_id as usize];
        if *count == 0 {
            insert_active_group_id(active_group_ids, active_group_positions, active_group_fingerprint, event.group_id);
        }
        *count += 1;
    }
}

fn active_group_key_matches(
    active_group_counts: &[u32],
    active_group_ids: &[u32],
    sorted_key: &[u32],
) -> bool {
    if active_group_ids.len() != sorted_key.len() {
        return false;
    }
    sorted_key.iter().all(|&group_id| active_group_counts[group_id as usize] > 0)
}

fn build_signature_from_active_groups(
    active_group_counts: &[u32],
    active_group_count: usize,
    groups: &[SweepGroup],
    labels_by_id: &[StateTerminalLabel],
    label_stamps: &mut [u32],
    stamp_generation: &mut u32,
) -> Vec<StateTerminalLabel> {
    if active_group_count == 0 { return Vec::new(); }
    let stamp = next_nonzero_stamp(stamp_generation, label_stamps);
    let mut signature = Vec::new();
    for (group_id, group) in groups.iter().enumerate() {
        if active_group_counts[group_id] == 0 { continue; }
        for &label_id in group.label_ids.iter() {
            let stamp_slot = &mut label_stamps[label_id as usize];
            if *stamp_slot != stamp {
                *stamp_slot = stamp;
                signature.push(labels_by_id[label_id as usize]);
            }
        }
    }
    signature.sort_unstable();
    signature
}

fn build_signature_from_active_group_ids(
    active_group_ids: &[u32],
    groups: &[SweepGroup],
    labels_by_id: &[StateTerminalLabel],
    label_stamps: &mut [u32],
    stamp_generation: &mut u32,
) -> Vec<StateTerminalLabel> {
    if active_group_ids.is_empty() { return Vec::new(); }

    let stamp = next_nonzero_stamp(stamp_generation, label_stamps);
    let mut signature = Vec::new();
    for &group_id in active_group_ids {
        let Some(group) = groups.get(group_id as usize) else { continue; };
        for &label_id in group.label_ids.iter() {
            let stamp_slot = &mut label_stamps[label_id as usize];
            if *stamp_slot != stamp {
                *stamp_slot = stamp;
                signature.push(labels_by_id[label_id as usize]);
            }
        }
    }
    signature.sort_unstable();
    signature
}

fn build_possible_match_vocab_and_weights_from_interval_maps(
    class_maps: &[Arc<IntervalPossibleMatchMap>],
    state_classes: &[u32],
    ordered_vocab: &OrderedVocab,
) -> (PossibleMatchVocabMap, RuntimePossibleMatchesByTerminal) {
    let num_ordered_tokens = ordered_vocab.ordered_to_originals.len();
    let pmv_detail_enabled = std::env::var("GLRMASK_PROFILE_PMV_DETAIL")
        .map(|value| value == "1")
        .unwrap_or(false);

    if group_pmv_legacy_enabled() {
        if pmv_detail_enabled {
            eprintln!("[glrmask/profile][pmv_detail] stage=legacy_expanded enabled=1");
        }
        return build_legacy_possible_match_vocab_and_weights_from_interval_maps(class_maps, state_classes, ordered_vocab);
    }

    let sweep_events_started_at = Instant::now();
    let (events, event_positions, groups, labels_by_id, sweep_build_stats) =
        build_sweep_events(class_maps, state_classes, num_ordered_tokens);
    let sweep_events_ms = elapsed_ms(sweep_events_started_at);

    let mut signature_to_id: FxHashMap<Vec<StateTerminalLabel>, SignatureClassId> = FxHashMap::default();
    let mut active_group_signature_to_signature_id: FxHashMap<u64, Vec<(Vec<u32>, SignatureClassId)>> = FxHashMap::default();
    let mut signature_labels: Vec<Vec<StateTerminalLabel>> = Vec::new();
    let mut original_to_internal = vec![u32::MAX; ordered_vocab.original_slot_count];
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut active_group_counts = vec![0u32; groups.len()];
    let mut active_group_ids = Vec::<u32>::new();
    let mut active_group_positions = vec![u32::MAX; groups.len()];
    let mut active_group_fingerprint = 0u64;
    let mut label_stamps = vec![0u32; labels_by_id.len()];
    let mut stamp_generation = 0u32;

    let sweep_started_at = Instant::now();
    let mut signature_build_ms = 0.0;
    let mut signature_lookup_ms = 0.0;
    let mut assignment_ms = 0.0;
    let mut sweep_segments = 0usize;
    let mut active_group_signature_cache_hits = 0usize;
    let mut active_group_signature_cache_misses = 0usize;
    let mut active_group_signature_build_ms = 0.0;
    let mut label_signature_build_ms = 0.0;
    let mut total_active_signature_len = 0usize;
    let mut max_active_signature_len = 0usize;
    let mut total_active_group_len = 0usize;
    let mut max_active_group_len = 0usize;

    let mut event_index = 0usize;
    let mut position = 0usize;
    while position < num_ordered_tokens {
        while event_index < event_positions.len() && event_positions[event_index] as usize == position {
            apply_sweep_events(
                &mut active_group_counts,
                &events[position],
                &mut active_group_ids,
                &mut active_group_positions,
                &mut active_group_fingerprint,
            );
            event_index += 1;
        }

        let next_position = event_positions.get(event_index).map(|&next| (next as usize).min(num_ordered_tokens)).unwrap_or(num_ordered_tokens);
        let active_group_signature_started_at = Instant::now();
        sweep_segments += 1;
        total_active_group_len += active_group_ids.len();
        max_active_group_len = max_active_group_len.max(active_group_ids.len());
        let cached_signature_id = active_group_signature_to_signature_id
            .get(&active_group_fingerprint)
            .and_then(|bucket| {
                bucket.iter().find_map(|(sorted_key, signature_id)| {
                    if active_group_key_matches(&active_group_counts, &active_group_ids, sorted_key) {
                        Some(*signature_id)
                    } else {
                        None
                    }
                })
            });
        active_group_signature_build_ms += elapsed_ms(active_group_signature_started_at);

        let signature_lookup_started_at = Instant::now();
        let signature_id = if let Some(existing) = cached_signature_id {
            active_group_signature_cache_hits += 1;
            existing
        } else {
            active_group_signature_cache_misses += 1;
            let label_signature_started_at = Instant::now();
            let signature = build_signature_from_active_group_ids(
                &active_group_ids,
                &groups,
                &labels_by_id,
                &mut label_stamps,
                &mut stamp_generation,
            );
            label_signature_build_ms += elapsed_ms(label_signature_started_at);

            let signature_id = if let Some(&existing) = signature_to_id.get(&signature) {
                existing
            } else {
                let new_id = signature_labels.len() as SignatureClassId;
                signature_to_id.insert(signature.clone(), new_id);
                signature_labels.push(signature);
                internal_to_originals.push(Vec::new());
                new_id
            };
            let active_group_key_started_at = Instant::now();
            let mut active_group_key = active_group_ids.clone();
            active_group_key.sort_unstable();
            active_group_signature_build_ms += elapsed_ms(active_group_key_started_at);
            active_group_signature_to_signature_id
                .entry(active_group_fingerprint)
                .or_default()
                .push((active_group_key, signature_id));
            signature_id
        };
        signature_lookup_ms += elapsed_ms(signature_lookup_started_at);
        signature_build_ms = active_group_signature_build_ms + label_signature_build_ms;

        let signature_len = signature_labels
            .get(signature_id as usize)
            .map(|labels| labels.len())
            .unwrap_or(0);
        total_active_signature_len += signature_len;
        max_active_signature_len = max_active_signature_len.max(signature_len);

        let assignment_started_at = Instant::now();
        for ordered_id in position..next_position {
            for &original in &ordered_vocab.ordered_to_originals[ordered_id] {
                if let Some(slot) = original_to_internal.get_mut(original as usize) { *slot = signature_id; }
            }
        }
        assignment_ms += elapsed_ms(assignment_started_at);
        position = next_position;
    }
    let sweep_ms = elapsed_ms(sweep_started_at);

    let internal_to_originals_started_at = Instant::now();
    for (original, &signature_id) in original_to_internal.iter().enumerate() {
        if signature_id != u32::MAX {
            internal_to_originals[signature_id as usize].push(original as u32);
        }
    }
    let sort_dedup_ms = elapsed_ms(internal_to_originals_started_at);

    let ids_by_label_started_at = Instant::now();
    let use_bitmask_ids_by_label = signature_labels.len() <= u128::BITS as usize;
    let mut label_entries = 0usize;
    let mut ids_by_label: BTreeMap<TerminalID, BTreeMap<u32, Vec<u32>>> = BTreeMap::new();
    let mut pair_masks = FxHashMap::<(TerminalID, u32), u128>::default();
    if use_bitmask_ids_by_label {
        for (signature_id, labels) in signature_labels.iter().enumerate() {
            let bit = 1u128 << signature_id;
            for &(class_id, terminal_id) in labels {
                label_entries += 1;
                *pair_masks.entry((terminal_id, class_id)).or_insert(0) |= bit;
            }
        }
    } else {
        for (signature_id, labels) in signature_labels.iter().enumerate() {
            let signature_id = signature_id as u32;
            for &(class_id, terminal_id) in labels {
                label_entries += 1;
                ids_by_label.entry(terminal_id).or_default().entry(class_id).or_default().push(signature_id);
            }
        }
    }
    let ids_by_label_ms = elapsed_ms(ids_by_label_started_at);

    let weight_build_started_at = Instant::now();
    let mut state_token_sets = 0usize;
    let mut bitmask_unique_masks = 0usize;
    let mut bitmask_mask_cache_hits = 0usize;
    let mut bitmask_mask_cache_misses = 0usize;
    let possible_matches: RuntimePossibleMatchesByTerminal = if use_bitmask_ids_by_label {
        let mut by_terminal: BTreeMap<TerminalID, Vec<(u32, u128)>> = BTreeMap::new();
        for ((terminal_id, class_id), mask) in pair_masks {
            by_terminal.entry(terminal_id).or_default().push((class_id, mask));
        }
        let mut shared_token_set_by_mask = FxHashMap::<u128, std::sync::Arc<RangeSetBlaze<u32>>>::default();
        by_terminal.into_iter().map(|(terminal_id, mut by_state)| {
            by_state.sort_unstable_by_key(|(state, _)| *state);
            let mut entries = Vec::new();
            for (state, mask) in by_state {
                if mask == 0 {
                    continue;
                }
                let shared_token_set = if let Some(existing) = shared_token_set_by_mask.get(&mask) {
                    bitmask_mask_cache_hits += 1;
                    existing.clone()
                } else {
                    bitmask_mask_cache_misses += 1;
                    let token_set = shared_rangeset(range_set_from_u128_mask(mask));
                    shared_token_set_by_mask.insert(mask, token_set.clone());
                    token_set
                };
                state_token_sets += 1;
                entries.push((state, shared_token_set));
            }
            if !entries.is_empty() {
                bitmask_unique_masks = shared_token_set_by_mask.len();
            }
            (terminal_id, Weight::from_per_tsid_shared(entries.into_iter()))
        }).filter(|(_, weight)| !weight.is_empty()).collect()
    } else {
        ids_by_label.into_iter().map(|(terminal_id, by_state)| {
            let mut entries = Vec::new();
            // `ids` are appended while iterating `signature_labels` in increasing
            // `signature_id` order, and labels are deduped within each signature,
            // so each bucket is already strictly increasing and unique.
            for (state, ids) in by_state {
                let token_set = range_set_from_sorted_ids(&ids);
                if !token_set.is_empty() {
                    state_token_sets += 1;
                    entries.push((state, shared_rangeset(token_set)));
                }
            }
            (terminal_id, Weight::from_per_tsid_shared(entries.into_iter()))
        }).filter(|(_, weight)| !weight.is_empty()).collect()
    };
    let terminal_ids = possible_matches.len();
    let weight_build_ms = elapsed_ms(weight_build_started_at);

    if pmv_detail_enabled {
        let mean_active_signature_len = if sweep_segments == 0 {
            0.0
        } else {
            total_active_signature_len as f64 / sweep_segments as f64
        };
        let mean_active_group_len = if sweep_segments == 0 {
            0.0
        } else {
            total_active_group_len as f64 / sweep_segments as f64
        };
        eprintln!(
            "[glrmask/profile][pmv_detail] stage=group_sweep_events sweep_events_ms={:.3} event_positions={} total_group_events={} used_state_classes={} total_group_intervals={} terminal_groups={} terminal_labels={} group_label_refs={}",
            sweep_events_ms,
            event_positions.len(),
            sweep_build_stats.total_events,
            sweep_build_stats.used_state_classes,
            sweep_build_stats.total_intervals,
            sweep_build_stats.terminal_groups,
            sweep_build_stats.terminal_labels,
            sweep_build_stats.group_label_refs,
        );
        eprintln!(
            "[glrmask/profile][pmv_detail] stage=sweep sweep_ms={:.3} segments={} signature_build_ms={:.3} signature_lookup_ms={:.3} assignment_ms={:.3} active_group_signature_cache_hits={} active_group_signature_cache_misses={} active_group_signature_build_ms={:.3} label_signature_build_ms={:.3} unique_signatures={} max_active_signature_len={} mean_active_signature_len={:.3} max_active_groups={} mean_active_groups={:.3}",
            sweep_ms,
            sweep_segments,
            signature_build_ms,
            signature_lookup_ms,
            assignment_ms,
            active_group_signature_cache_hits,
            active_group_signature_cache_misses,
            active_group_signature_build_ms,
            label_signature_build_ms,
            signature_labels.len(),
            max_active_signature_len,
            mean_active_signature_len,
            max_active_group_len,
            mean_active_group_len,
        );
        eprintln!(
            "[glrmask/profile][pmv_detail] stage=sort_dedup sort_dedup_ms={:.3} internal_signature_classes={}",
            sort_dedup_ms,
            internal_to_originals.len(),
        );
        eprintln!(
            "[glrmask/profile][pmv_detail] stage=ids_by_label ids_by_label_ms={:.3} label_entries={} terminal_ids={} bitmask_path_used={}",
            ids_by_label_ms,
            label_entries,
            terminal_ids,
            use_bitmask_ids_by_label,
        );
        eprintln!(
            "[glrmask/profile][pmv_detail] stage=weights weights_ms={:.3} terminal_ids={} state_token_sets={} bitmask_path_used={} bitmask_unique_masks={} bitmask_mask_cache_hits={} bitmask_mask_cache_misses={}",
            weight_build_ms,
            terminal_ids,
            state_token_sets,
            use_bitmask_ids_by_label,
            bitmask_unique_masks,
            bitmask_mask_cache_hits,
            bitmask_mask_cache_misses,
        );
    }

    let possible_match_vocab = PossibleMatchVocabMap { original_to_internal, internal_to_originals };
    if group_pmv_validation_enabled() {
        validate_group_pmv_outputs(class_maps, state_classes, ordered_vocab, &possible_match_vocab, &possible_matches);
    }

    (possible_match_vocab, possible_matches)
}


type ExpandedIntervalPossibleMatchMap = BTreeMap<TerminalID, Vec<(u32, u32)>>;

#[derive(Debug, Clone, Copy)]
struct LegacySweepEvent {
    add: bool,
    label_id: u32,
}

fn normalize_token_ranges(ranges: &mut Vec<(u32, u32)>) {
    if ranges.len() <= 1 { return; }
    ranges.sort_unstable();
    let mut write = 0usize;
    for read in 1..ranges.len() {
        let (start, end) = ranges[read];
        let current = &mut ranges[write];
        if start <= current.1.saturating_add(1) {
            current.1 = current.1.max(end);
        } else {
            write += 1;
            ranges[write] = (start, end);
        }
    }
    ranges.truncate(write + 1);
}

fn append_expanded_ranges(
    map: &mut ExpandedIntervalPossibleMatchMap,
    terminal: TerminalID,
    ranges: &[(u32, u32)],
) {
    if !ranges.is_empty() {
        map.entry(terminal).or_default().extend_from_slice(ranges);
    }
}

fn normalize_expanded_interval_map(map: &mut ExpandedIntervalPossibleMatchMap) {
    map.retain(|_, ranges| {
        normalize_token_ranges(ranges);
        !ranges.is_empty()
    });
}

fn expand_interval_class_maps(
    class_maps: &[Arc<IntervalPossibleMatchMap>],
) -> Vec<Arc<ExpandedIntervalPossibleMatchMap>> {
    class_maps.iter().map(|class_map| {
        let mut expanded = ExpandedIntervalPossibleMatchMap::new();
        for entry in class_map.iter() {
            for &terminal_id in entry.terminals.iter() {
                append_expanded_ranges(&mut expanded, terminal_id, &entry.ranges);
            }
        }
        normalize_expanded_interval_map(&mut expanded);
        Arc::new(expanded)
    }).collect()
}

fn push_legacy_sweep_event(
    events: &mut [Vec<LegacySweepEvent>],
    event_positions: &mut Vec<u32>,
    position: u32,
    event: LegacySweepEvent,
) {
    let Some(bucket) = events.get_mut(position as usize) else { return; };
    if bucket.is_empty() { event_positions.push(position); }
    bucket.push(event);
}

fn build_legacy_sweep_events(
    class_maps: &[Arc<ExpandedIntervalPossibleMatchMap>],
    state_classes: &[u32],
    num_ordered_tokens: usize,
) -> (Vec<Vec<LegacySweepEvent>>, Vec<u32>, Vec<StateTerminalLabel>) {
    let mut events = vec![Vec::new(); num_ordered_tokens + 1];
    let mut event_positions = Vec::new();
    let mut labels_by_id = Vec::<StateTerminalLabel>::new();
    let mut label_ids = FxHashMap::<StateTerminalLabel, u32>::default();

    for class_id in used_state_class_ids(state_classes) {
        let Some(class_map) = class_maps.get(class_id as usize) else { continue; };
        for (&terminal_id, ranges) in class_map.iter() {
            let label_id = intern_state_terminal_label(&mut label_ids, &mut labels_by_id, (class_id, terminal_id));
            for &(lo, mut hi) in ranges.iter() {
                if num_ordered_tokens == 0 { continue; }
                let max_token = num_ordered_tokens as u32 - 1;
                if lo > max_token { continue; }
                hi = hi.min(max_token);
                if lo > hi { continue; }
                push_legacy_sweep_event(&mut events, &mut event_positions, lo, LegacySweepEvent { add: true, label_id });
                let after = hi.saturating_add(1);
                if after <= num_ordered_tokens as u32 {
                    push_legacy_sweep_event(&mut events, &mut event_positions, after, LegacySweepEvent { add: false, label_id });
                }
            }
        }
    }

    event_positions.sort_unstable();
    event_positions.dedup();
    (events, event_positions, labels_by_id)
}

fn apply_legacy_sweep_events(
    active_counts: &mut [u32],
    events: &[LegacySweepEvent],
    active_label_count: &mut usize,
) {
    for event in events.iter().filter(|event| !event.add) {
        let count = &mut active_counts[event.label_id as usize];
        assert!(*count > 0, "legacy pmv sweep removal underflow for label_id={}", event.label_id);
        if *count == 1 {
            *active_label_count -= 1;
        }
        *count -= 1;
    }
    for event in events.iter().filter(|event| event.add) {
        let count = &mut active_counts[event.label_id as usize];
        if *count == 0 {
            *active_label_count += 1;
        }
        *count += 1;
    }
}

fn build_legacy_possible_match_vocab_and_weights_from_interval_maps(
    class_maps: &[Arc<IntervalPossibleMatchMap>],
    state_classes: &[u32],
    ordered_vocab: &OrderedVocab,
) -> (PossibleMatchVocabMap, RuntimePossibleMatchesByTerminal) {
    let expanded_class_maps = expand_interval_class_maps(class_maps);
    let num_ordered_tokens = ordered_vocab.ordered_to_originals.len();
    let (events, event_positions, labels_by_id) =
        build_legacy_sweep_events(&expanded_class_maps, state_classes, num_ordered_tokens);

    let mut signature_to_id: FxHashMap<Vec<StateTerminalLabel>, SignatureClassId> = FxHashMap::default();
    let mut signature_labels: Vec<Vec<StateTerminalLabel>> = Vec::new();
    let mut original_to_internal = vec![u32::MAX; ordered_vocab.original_slot_count];
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut active_counts = vec![0u32; labels_by_id.len()];
    let mut active_label_count = 0usize;

    let mut event_index = 0usize;
    let mut position = 0usize;
    while position < num_ordered_tokens {
        while event_index < event_positions.len() && event_positions[event_index] as usize == position {
            apply_legacy_sweep_events(&mut active_counts, &events[position], &mut active_label_count);
            event_index += 1;
        }

        let next_position = event_positions.get(event_index).map(|&next| (next as usize).min(num_ordered_tokens)).unwrap_or(num_ordered_tokens);
        let mut signature = Vec::with_capacity(active_label_count);
        for (label_id, &label) in labels_by_id.iter().enumerate() {
            if active_counts[label_id] > 0 {
                signature.push(label);
            }
        }
        signature.sort_unstable();

        let signature_id = if let Some(&existing) = signature_to_id.get(&signature) { existing } else {
            let new_id = signature_labels.len() as SignatureClassId;
            signature_to_id.insert(signature.clone(), new_id);
            signature_labels.push(signature);
            internal_to_originals.push(Vec::new());
            new_id
        };

        for ordered_id in position..next_position {
            for &original in &ordered_vocab.ordered_to_originals[ordered_id] {
                if let Some(slot) = original_to_internal.get_mut(original as usize) { *slot = signature_id; }
                internal_to_originals[signature_id as usize].push(original);
            }
        }
        position = next_position;
    }

    for originals in &mut internal_to_originals { originals.sort_unstable(); originals.dedup(); }

    let mut ids_by_label: BTreeMap<TerminalID, BTreeMap<u32, Vec<u32>>> = BTreeMap::new();
    for (signature_id, labels) in signature_labels.iter().enumerate() {
        let signature_id = signature_id as u32;
        for &(class_id, terminal_id) in labels {
            ids_by_label.entry(terminal_id).or_default().entry(class_id).or_default().push(signature_id);
        }
    }

    let possible_matches = ids_by_label.into_iter().map(|(terminal_id, by_state)| {
        let mut entries = Vec::new();
        for (state, mut ids) in by_state {
            ids.sort_unstable();
            ids.dedup();
            let token_set = range_set_from_sorted_ids(&ids);
            if !token_set.is_empty() {
                entries.push((state, shared_rangeset(token_set)));
            }
        }
        (terminal_id, Weight::from_per_tsid_shared(entries.into_iter()))
    }).filter(|(_, weight)| !weight.is_empty()).collect();

    (PossibleMatchVocabMap { original_to_internal, internal_to_originals }, possible_matches)
}

fn validate_group_pmv_outputs(
    class_maps: &[Arc<IntervalPossibleMatchMap>],
    state_classes: &[u32],
    ordered_vocab: &OrderedVocab,
    actual_vocab: &PossibleMatchVocabMap,
    actual_matches: &RuntimePossibleMatchesByTerminal,
) {
    let started_at = Instant::now();
    let (expected_vocab, expected_matches) =
        build_legacy_possible_match_vocab_and_weights_from_interval_maps(class_maps, state_classes, ordered_vocab);

    if actual_vocab.original_to_internal != expected_vocab.original_to_internal {
        let mut mismatch = None;
        for idx in 0..actual_vocab.original_to_internal.len().min(expected_vocab.original_to_internal.len()) {
            let actual = actual_vocab.original_to_internal[idx];
            let expected = expected_vocab.original_to_internal[idx];
            if actual != expected {
                mismatch = Some((idx, actual, expected));
                break;
            }
        }
        panic!("group PMV validation failed: original_to_internal mismatch at {:?}", mismatch);
    }
    if actual_vocab.internal_to_originals != expected_vocab.internal_to_originals {
        let mut mismatch = None;
        for idx in 0..actual_vocab.internal_to_originals.len().min(expected_vocab.internal_to_originals.len()) {
            let actual = &actual_vocab.internal_to_originals[idx];
            let expected = &expected_vocab.internal_to_originals[idx];
            if actual != expected {
                mismatch = Some((idx, actual.clone(), expected.clone()));
                break;
            }
        }
        panic!("group PMV validation failed: internal_to_originals mismatch at {:?}; actual_len={} expected_len={}", mismatch, actual_vocab.internal_to_originals.len(), expected_vocab.internal_to_originals.len());
    }
    if actual_matches != &expected_matches {
        let mut terminal_ids: Vec<TerminalID> = actual_matches.keys().chain(expected_matches.keys()).copied().collect();
        terminal_ids.sort_unstable();
        terminal_ids.dedup();
        let mismatch = terminal_ids.into_iter().find(|terminal_id| actual_matches.get(terminal_id) != expected_matches.get(terminal_id));
        panic!("group PMV validation failed: possible match weight mismatch for terminal {:?}", mismatch);
    }

    if std::env::var_os("GLRMASK_PROFILE_PMV_DETAIL").is_some() {
        eprintln!("[glrmask/profile][pmv_validate] legacy_expand_compare_ms={:.3}", elapsed_ms(started_at));
    }
}

fn group_pmv_validation_enabled() -> bool {
    std::env::var("GLRMASK_VALIDATE_GROUP_PMV")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn group_pmv_legacy_enabled() -> bool {
    std::env::var("GLRMASK_PM_USE_LEGACY_PMV")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

const PM_NFA_POWERSET_DEFAULT_MAX_STATES: usize = 12_000;
const PM_NFA_POWERSET_NARROW_MAX_STATES: usize = 32_768;
const PM_NFA_POWERSET_NARROW_MAX_TERMINALS: usize = 256;

fn nfa_powerset_collect_default(state_count: usize, root_terminal_union: usize) -> bool {
    state_count <= PM_NFA_POWERSET_DEFAULT_MAX_STATES
        || (state_count <= PM_NFA_POWERSET_NARROW_MAX_STATES
            && root_terminal_union <= PM_NFA_POWERSET_NARROW_MAX_TERMINALS)
}

fn nfa_powerset_collect_enabled(state_count: usize, root_terminal_union: usize) -> bool {
    std::env::var("GLRMASK_PM_NFA_POWERSET_COLLECT")
        .map(|value| {
            let trimmed = value.trim();
            trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or_else(|_| nfa_powerset_collect_default(state_count, root_terminal_union))
}

struct PossibleMatchPowersetView {
    tokenizer_view: TokenizerView,
    raw_start_to_view: Vec<u32>,
    boundary_state: Vec<u32>,
    is_end: Vec<bool>,
}

fn intern_possible_match_config(
    mut config: Vec<u32>,
    is_closed: bool,
    config_ids: &mut FxHashMap<Vec<u32>, u32>,
    configs: &mut Vec<Box<[u32]>>,
    config_is_closed: &mut Vec<bool>,
) -> Option<u32> {
    config.sort_unstable();
    config.dedup();
    if config.is_empty() {
        return None;
    }
    if let Some(&id) = config_ids.get(&config) {
        config_is_closed[id as usize] |= is_closed;
        return Some(id);
    }
    let id = configs.len() as u32;
    config_ids.insert(config.clone(), id);
    configs.push(config.into_boxed_slice());
    config_is_closed.push(is_closed);
    Some(id)
}

fn build_possible_match_powerset_view(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
) -> PossibleMatchPowersetView {
    let singleton_closures = tokenizer.all_singleton_epsilon_closures();
    let mut config_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut configs = Vec::<Box<[u32]>>::new();
    let mut config_is_closed = Vec::<bool>::new();
    let raw_start_to_view = (0..tokenizer.num_states())
        .map(|raw_state| {
            intern_possible_match_config(
                singleton_closures[raw_state as usize].to_vec(),
                true,
                &mut config_ids,
                &mut configs,
                &mut config_is_closed,
            )
            .expect("epsilon closure of tokenizer state must be nonempty")
        })
        .collect::<Vec<_>>();

    let active_bytes = relevant_bytes
        .iter()
        .enumerate()
        .filter_map(|(byte, &active)| active.then_some(byte as u8))
        .collect::<Vec<_>>();
    let mut states = Vec::<FlatDfaState>::new();
    let mut transition_rows = Vec::<Box<[u32; 256]>>::new();
    let mut boundary_state = Vec::<u32>::new();
    let mut is_end = Vec::<bool>::new();
    let mut target_marks = vec![0u32; tokenizer.num_states() as usize];
    let mut target_generation = 0u32;
    let mut target_config = Vec::<u32>::new();
    let mut config_index = 0usize;
    while config_index < configs.len() {
        let config = configs[config_index].to_vec();
        let mut finalizers = config
            .iter()
            .flat_map(|&raw_state| tokenizer.matched_terminals_iter(raw_state))
            .map(|terminal| terminal as usize)
            .collect::<Vec<_>>();
        finalizers.sort_unstable();
        finalizers.dedup();
        states.push(FlatDfaState {
            finalizers,
            possible_future_group_ids: Vec::new(),
        });

        let live_config = config
            .iter()
            .copied()
            .filter(|&raw_state| !tokenizer.is_end(raw_state))
            .collect::<Vec<_>>();
        is_end.push(live_config.is_empty());
        boundary_state.push(
            intern_possible_match_config(
                live_config,
                false,
                &mut config_ids,
                &mut configs,
                &mut config_is_closed,
            )
            .unwrap_or(u32::MAX),
        );

        let mut row = Box::new([u32::MAX; 256]);
        let source_is_closed = config_is_closed[config_index];
        for &byte in &active_bytes {
            target_generation = target_generation.wrapping_add(1);
            if target_generation == 0 {
                target_marks.fill(0);
                target_generation = 1;
            }
            target_config.clear();
            // Ordinary powerset configurations are already epsilon-closed, so
            // re-expanding every member's source closure is redundant. A
            // boundary projection may have accepting/end states removed and
            // is not necessarily closed; those configurations retain the
            // exact source-closure walk.
            for &source in &config {
                let sources = if source_is_closed {
                    std::slice::from_ref(&source)
                } else {
                    singleton_closures[source as usize].as_ref()
                };
                for &closed_source in sources {
                    let target = tokenizer.get_transition(closed_source, byte);
                    if target == u32::MAX {
                        continue;
                    }
                    for &reachable in singleton_closures[target as usize].iter() {
                        let mark = &mut target_marks[reachable as usize];
                        if *mark != target_generation {
                            *mark = target_generation;
                            target_config.push(reachable);
                        }
                    }
                }
            }
            target_config.sort_unstable();
            if let Some(target) = intern_possible_match_config(
                target_config.clone(),
                true,
                &mut config_ids,
                &mut configs,
                &mut config_is_closed,
            ) {
                row[byte as usize] = target;
            }
        }
        transition_rows.push(row);
        config_index += 1;
    }

    let mut transitions = Vec::with_capacity(states.len() * 256);
    for row in transition_rows {
        transitions.extend_from_slice(row.as_ref());
    }
    debug_assert_eq!(states.len(), configs.len());
    debug_assert_eq!(config_is_closed.len(), configs.len());
    debug_assert_eq!(boundary_state.len(), states.len());
    debug_assert_eq!(is_end.len(), states.len());

    PossibleMatchPowersetView {
        tokenizer_view: TokenizerView {
            flat_dfa: FlatDfa {
                states,
                start_state: raw_start_to_view[tokenizer.start_state() as usize] as usize,
                transitions: Arc::from(transitions),
            },
        },
        raw_start_to_view,
        boundary_state,
        is_end,
    }
}

fn sparse_root_collect_enabled() -> bool {
    std::env::var("GLRMASK_PM_SPARSE_ROOT_COLLECT")
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

fn sparse_root_state_limit() -> usize {
    std::env::var("GLRMASK_PM_SPARSE_ROOT_MAX_STATES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(128)
}

fn sparse_root_terminal_limit() -> usize {
    std::env::var("GLRMASK_PM_SPARSE_ROOT_MAX_TERMINALS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16)
}

fn root_terminal_union_count(tokenizer: &Tokenizer, states: &[u32]) -> usize {
    let mut seen = vec![false; tokenizer.num_terminals() as usize];
    let mut count = 0usize;
    for &state in states {
        for terminal in tokenizer
            .matched_terminals_iter(state)
            .chain(tokenizer.possible_future_terminals_iter(state))
        {
            let slot = terminal as usize;
            if slot < seen.len() && !seen[slot] {
                seen[slot] = true;
                count += 1;
            }
        }
    }
    count
}

fn interval_map_from_sparse_matches(
    matches: &FxHashMap<TerminalID, RangeSetBlaze<u32>>,
) -> IntervalPossibleMatchMap {
    let mut by_ranges = BTreeMap::<Vec<(u32, u32)>, Vec<TerminalID>>::new();
    for (&terminal, token_ids) in matches {
        let ranges: Vec<(u32, u32)> = token_ids
            .ranges()
            .map(|range| (*range.start(), *range.end()))
            .collect();
        if !ranges.is_empty() {
            by_ranges.entry(ranges).or_default().push(terminal);
        }
    }

    let mut map = Vec::with_capacity(by_ranges.len());
    for (ranges, mut terminals) in by_ranges {
        terminals.sort_unstable();
        terminals.dedup();
        if !terminals.is_empty() {
            map.push(TerminalRangeGroup {
                terminals: terminals.into_boxed_slice(),
                ranges,
            });
        }
    }
    map.sort_unstable_by(|left, right| {
        left.terminals
            .as_ref()
            .cmp(right.terminals.as_ref())
            .then_with(|| left.ranges.cmp(&right.ranges))
    });
    map
}

fn collect_sparse_root_possible_matches(
    tokenizer: &Tokenizer,
    root: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
    entries: &[u32],
    canonical_state: Option<&[u32]>,
) -> TrieClassBuildResult {
    let mut computer = PossibleMatchesComputer::new_with_canonical_state(tokenizer, canonical_state);
    let mut state_classes = vec![u32::MAX; tokenizer.num_states() as usize];
    let mut class_maps = Vec::<Arc<IntervalPossibleMatchMap>>::new();
    let mut map_to_class = FxHashMap::<IntervalPossibleMatchMap, u32>::default();

    for &state in entries {
        let sparse_matches = computer.possible_matches_for_node(root, state);
        let interval_map = interval_map_from_sparse_matches(sparse_matches.as_ref());
        let class_id = if let Some(&class_id) = map_to_class.get(&interval_map) {
            class_id
        } else {
            let class_id = class_maps.len() as u32;
            map_to_class.insert(interval_map.clone(), class_id);
            class_maps.push(Arc::new(interval_map));
            class_id
        };

        if let Some(slot) = state_classes.get_mut(state as usize) {
            *slot = class_id;
        }
    }

    TrieClassBuildResult {
        state_classes,
        class_maps,
    }
}

fn attach_structured_dispatch_possible_matches(
    tokenizer: &Tokenizer,
    root: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
    result: &mut TrieClassBuildResult,
) {
    if !tokenizer.has_deterministic_dispatch() {
        return;
    }

    let start = tokenizer.start_state();
    let dispatch = collect_sparse_root_possible_matches(tokenizer, root, &[start], None);
    let dispatch_class = dispatch
        .state_classes
        .get(start as usize)
        .copied()
        .filter(|&class| class != u32::MAX)
        .expect("structured lexer dispatch PM row must be collected");
    let dispatch_map = dispatch.class_maps[dispatch_class as usize].as_ref();
    let class = result
        .class_maps
        .iter()
        .position(|existing| existing.as_ref() == dispatch_map)
        .map(|class| class as u32)
        .unwrap_or_else(|| {
            let class = result.class_maps.len() as u32;
            result.class_maps.push(Arc::new(dispatch_map.clone()));
            class
        });
    result.state_classes[start as usize] = class;
}

pub(crate) fn compute_constraint_possible_matches(
    tokenizer: &Tokenizer,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    config: ConstraintPossibleMatchesConfig,
) -> ConstraintPossibleMatchesComputation {
    let artifacts_and_profile = get_ordered_vocab_trie_artifacts(token_bytes);
    if config.defer_to_dynamic_mask {
        let (artifacts, profile) = artifacts_and_profile;
        emit_ordered_vocab_cache_profile(profile);
        let runtime_dynamic_vocab = runtime_dynamic_vocab_artifacts(&artifacts);
        return empty_possible_matches_computation(
            tokenizer,
            token_bytes.len(),
            runtime_dynamic_vocab,
        );
    }
    compute_constraint_possible_matches_with_artifacts(
        tokenizer,
        token_bytes.len(),
        artifacts_and_profile,
        None,
    )
}

fn runtime_dynamic_vocab_artifacts(
    artifacts: &OrderedVocabTrieArtifacts,
) -> RuntimeDynamicMaskVocabArtifacts {
    RuntimeDynamicMaskVocabArtifacts {
        trie: Arc::clone(&artifacts.trie),
        token_aliases: Arc::clone(&artifacts.ordered_vocab.ordered_to_originals),
    }
}

/// Neutral PM artifact for the deferred mode. All dimensions are deliberately
/// unmapped so PM cannot force tokenizer-state or vocabulary splits during ID
/// reconciliation; the independently retained dynamic vocabulary is the exact
/// fallback representation.
fn empty_possible_matches_computation(
    tokenizer: &Tokenizer,
    original_token_count: usize,
    runtime_dynamic_vocab: RuntimeDynamicMaskVocabArtifacts,
) -> ConstraintPossibleMatchesComputation {
    let possible_matches_id_map = InternalIdMap {
        tokenizer_states: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            vec![u32::MAX; tokenizer.num_states() as usize],
            0,
        ),
        vocab_tokens: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            vec![u32::MAX; original_token_count],
            0,
        ),
    };
    ConstraintPossibleMatchesComputation {
        mapped_possible_matches: MappedArtifact::new(
            RuntimePossibleMatchesByTerminal::new(),
            possible_matches_id_map,
        ),
        runtime_dynamic_vocab,
        profile: ConstraintPossibleMatchesProfile::default(),
    }
}

fn compute_constraint_possible_matches_with_artifacts(
    tokenizer: &Tokenizer,
    original_token_count: usize,
    artifacts_and_profile: (OrderedVocabTrieArtifacts, OrderedVocabCacheProfile),
    initial_vocab_map: Option<&ManyToOneIdMap>,
) -> ConstraintPossibleMatchesComputation {
    let pm_started_at = Instant::now();

    let (artifacts, ordered_vocab_cache_profile) = artifacts_and_profile;
    emit_ordered_vocab_cache_profile(ordered_vocab_cache_profile);
    let runtime_dynamic_vocab = runtime_dynamic_vocab_artifacts(&artifacts);
    let ordered_vocab = artifacts.ordered_vocab;
    let trie = artifacts.trie;

    let structured_dispatch = tokenizer.has_deterministic_dispatch();
    let dispatch_start = structured_dispatch.then(|| tokenizer.start_state());
    let trie_build_states: Vec<u32> = (0..tokenizer.num_states())
        .filter(|state| Some(*state) != dispatch_start)
        .collect();

    let root_terminal_union = root_terminal_union_count(tokenizer, &trie_build_states);
    let use_nfa_powerset_collect = tokenizer.has_epsilon_transitions()
        && !structured_dispatch
        && nfa_powerset_collect_enabled(tokenizer.num_states() as usize, root_terminal_union);
    let use_sparse_root_collect = (tokenizer.has_epsilon_transitions() && !structured_dispatch)
        || (sparse_root_collect_enabled()
            && trie_build_states.len() <= sparse_root_state_limit()
            && root_terminal_union <= sparse_root_terminal_limit());

    let mut trie_class_result = if use_nfa_powerset_collect {
        let mut relevant_bytes = [false; 256];
        for bytes in &ordered_vocab.ordered_token_bytes {
            for &byte in bytes {
                relevant_bytes[byte as usize] = true;
            }
        }
        let view_started_at = Instant::now();
        let powerset = build_possible_match_powerset_view(tokenizer, &relevant_bytes);
        let view_build_ms = elapsed_ms(view_started_at);
        let mut view_entries = trie_build_states
            .iter()
            .map(|&raw_state| powerset.raw_start_to_view[raw_state as usize])
            .collect::<Vec<_>>();
        view_entries.sort_unstable();
        view_entries.dedup();
        let (view_result, _) =
            collector::collect_possible_matches_interval_trie_class_build_for_flat_view(
                &powerset.tokenizer_view,
                tokenizer.num_terminals() as usize,
                &powerset.is_end,
                &trie.root,
                &view_entries,
                Some(&powerset.boundary_state),
            );
        let mut state_classes = vec![u32::MAX; tokenizer.num_states() as usize];
        for &raw_state in &trie_build_states {
            let view_state = powerset.raw_start_to_view[raw_state as usize] as usize;
            state_classes[raw_state as usize] = view_result.state_classes[view_state];
        }
        if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][trie_build_nfa_powerset] raw_states={} view_states={} root_view_states={} classes={} view_build_ms={:.3}",
                trie_build_states.len(),
                powerset.tokenizer_view.dfa().states.len(),
                view_entries.len(),
                view_result.class_maps.len(),
                view_build_ms,
            );
        }
        TrieClassBuildResult {
            state_classes,
            class_maps: view_result.class_maps,
        }
    } else if use_sparse_root_collect {
        if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][trie_build_sparse_root] states={} terminals={} max_states={} max_terminals={}",
                trie_build_states.len(),
                root_terminal_union,
                sparse_root_state_limit(),
                sparse_root_terminal_limit(),
            );
        }
        collect_sparse_root_possible_matches(
            tokenizer,
            &trie.root,
            &trie_build_states,
            None,
        )
    } else {
        collector::collect_possible_matches_interval_trie_class_build_with_classes(
            tokenizer,
            &trie.root,
            &trie_build_states,
            None,
        )
        .0
    };
    attach_structured_dispatch_possible_matches(tokenizer, &trie.root, &mut trie_class_result);

    let possible_matches_collect_ms = elapsed_ms(pm_started_at);

    let possible_match_vocab_started_at = Instant::now();
    let (possible_match_vocab, possible_matches) = build_possible_match_vocab_and_weights_from_interval_maps(&trie_class_result.class_maps, &trie_class_result.state_classes, ordered_vocab.as_ref());

    let local_vocab_map = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
        possible_match_vocab.original_to_internal.clone(),
        possible_match_vocab.internal_to_originals.len() as u32,
    );
    let vocab_tokens = if let Some(initial_vocab_map) = initial_vocab_map {
        initial_vocab_map.compose(&local_vocab_map)
    } else {
        local_vocab_map
    };

    let possible_matches_id_map = InternalIdMap {
        tokenizer_states: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            trie_class_result.state_classes.clone(),
            trie_class_result.state_classes.iter().copied().filter(|&class_id| class_id != u32::MAX).max().map(|class_id| class_id + 1).unwrap_or(0),
        ),
        vocab_tokens,
    };

    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some() || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
        eprintln!("[glrmask/profile][possible_match_vocab] original_tokens={} ordered_byte_tokens={} possible_match_tokens={}", original_token_count, ordered_vocab.ordered_to_originals.len(), possible_matches_id_map.vocab_tokens.internal_to_originals.len());
    }

    let possible_match_vocab_ms = elapsed_ms(possible_match_vocab_started_at);

    ConstraintPossibleMatchesComputation {
        mapped_possible_matches: MappedArtifact::new(possible_matches, possible_matches_id_map),
        runtime_dynamic_vocab,
        profile: ConstraintPossibleMatchesProfile {
            vocab_equiv_ms: 0.0,
            possible_matches_collect_ms,
            possible_match_vocab_ms,
        },
    }
}

pub(crate) fn compute_constraint_possible_matches_for_vocab(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    config: ConstraintPossibleMatchesConfig,
) -> ConstraintPossibleMatchesComputation {
    if config.defer_to_dynamic_mask {
        let (full_artifacts, full_profile) = get_ordered_vocab_trie_artifacts_for_vocab(vocab);
        emit_ordered_vocab_cache_profile(full_profile);
        let runtime_dynamic_vocab = runtime_dynamic_vocab_artifacts(&full_artifacts);
        return empty_possible_matches_computation(
            tokenizer,
            vocab.entries.len(),
            runtime_dynamic_vocab,
        );
    }

    if pm_vocab_equiv_enabled() && pm_vocab_equiv_supported(tokenizer) {
        let (full_artifacts, full_profile) = get_ordered_vocab_trie_artifacts_for_vocab(vocab);
        let runtime_dynamic_vocab = runtime_dynamic_vocab_artifacts(&full_artifacts);
        emit_ordered_vocab_cache_profile(full_profile);
        let vocab_equiv_started_at = Instant::now();
        let use_naive = std::env::var("GLRMASK_PM_VOCAB_EQUIV_NAIVE")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let pm_vocab_map = if use_naive || tokenizer.has_epsilon_transitions() {
            compute_pm_vocab_equivalence_map(
                tokenizer,
                full_artifacts.ordered_vocab.as_ref(),
                full_artifacts.trie.as_ref(),
            )
        } else {
            compute_pm_vocab_equivalence_map_fast(tokenizer, full_artifacts.ordered_vocab.as_ref())
        };
        if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][pm_vocab_equiv] original_tokens={} pm_vocab_classes={} mode={} ms={:.3}",
                vocab.entries.len(),
                pm_vocab_map.internal_to_originals.len(),
                if tokenizer.has_epsilon_transitions() {
                    "nfa_exact"
                } else if use_naive {
                    "naive"
                } else {
                    "fast"
                },
                elapsed_ms(vocab_equiv_started_at),
            );
        }
        let compact_token_bytes =
            build_internal_token_bytes_from_groups(vocab, &pm_vocab_map.internal_to_originals);
        let vocab_equiv_ms = elapsed_ms(vocab_equiv_started_at);
        let mut computation = compute_constraint_possible_matches_with_artifacts(
            tokenizer,
            vocab.entries.len(),
            get_ordered_vocab_trie_artifacts(&compact_token_bytes),
            Some(&pm_vocab_map),
        );
        computation.runtime_dynamic_vocab = runtime_dynamic_vocab;
        computation.profile.vocab_equiv_ms = vocab_equiv_ms;
        return computation;
    }

    compute_constraint_possible_matches_with_artifacts(
        tokenizer,
        vocab.entries.len(),
        get_ordered_vocab_trie_artifacts_for_vocab(vocab),
        None,
    )
}

pub(crate) fn prepare_vocab_for_possible_matches(vocab: &Vocab) {
    let _ = get_ordered_vocab_trie_artifacts_for_vocab(vocab);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer;
    use crate::compiler::pipeline::build_tokenizer_from_exprs_partitioned_with_adaptive;
    use std::collections::BTreeSet;

    fn directly_matched_terminals(
        tokenizer: &Tokenizer,
        start_state: u32,
        bytes: &[u8],
    ) -> BTreeSet<u32> {
        let mut states = tokenizer.execute_from_state_end_only(&[], start_state);
        let mut terminals = BTreeSet::new();
        for &byte in bytes {
            states = tokenizer.step_all(&states, byte);
            for &state in &states {
                terminals.extend(tokenizer.matched_terminals_iter(state));
            }
            if states.is_empty() {
                break;
            }
        }
        terminals
    }

    #[test]
    fn structured_dispatch_possible_matches_match_direct_state_set_execution() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b" ".to_vec())),
                min: 1,
                max: None,
            },
        ];
        let tokenizer = build_tokenizer_from_exprs_partitioned_with_adaptive(
            &expressions,
            None,
            &[0, 1, 2, 2],
            false,
        );
        assert!(tokenizer.has_deterministic_dispatch());
        assert!(pm_vocab_equiv_supported(&tokenizer));

        let entries = vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"b".to_vec()),
            (3, b" a".to_vec()),
            (4, b"a ".to_vec()),
            (5, b"x".to_vec()),
            (6, b"ab".to_vec()),
        ];
        let vocab = Vocab::new(entries.clone(), None);
        let computation = compute_constraint_possible_matches_for_vocab(
            &tokenizer,
            &vocab,
            ConstraintPossibleMatchesConfig::EAGER,
        );
        let mapped = &computation.mapped_possible_matches;

        for state in 0..tokenizer.num_states() {
            let internal_state = mapped.id_map().tokenizer_states.original_to_internal[state as usize];
            assert_ne!(internal_state, u32::MAX, "state={state}");
            for (token_id, bytes) in &entries {
                let internal_token = mapped.id_map().vocab_tokens.original_to_internal[*token_id as usize];
                assert_ne!(internal_token, u32::MAX, "token={token_id}");
                let expected = directly_matched_terminals(&tokenizer, state, bytes);
                for terminal in 0..tokenizer.num_terminals() {
                    let actual = mapped
                        .artifact()
                        .get(&terminal)
                        .is_some_and(|weight| {
                            weight
                                .tokens_for_tsid(internal_state)
                                .contains(internal_token)
                        });
                    assert_eq!(
                        actual,
                        expected.contains(&terminal),
                        "state={state} token={token_id} bytes={bytes:?} terminal={terminal}",
                    );
                }
            }
        }
    }

    #[test]
    fn epsilon_nfa_possible_match_collector_defaults_by_state_scale() {
        assert!(nfa_powerset_collect_default(914, 1_707));
        assert!(nfa_powerset_collect_default(8_108, 1_000));
        assert!(nfa_powerset_collect_default(10_355, 1_000));
        assert!(nfa_powerset_collect_default(
            PM_NFA_POWERSET_DEFAULT_MAX_STATES,
            usize::MAX,
        ));
        assert!(!nfa_powerset_collect_default(18_943, 1_707));
        assert!(nfa_powerset_collect_default(26_965, 192));
        assert!(!nfa_powerset_collect_default(
            PM_NFA_POWERSET_NARROW_MAX_STATES + 1,
            192,
        ));
        assert!(!nfa_powerset_collect_default(26_965, 1_707));
    }

    #[test]
    fn epsilon_powerset_interval_collector_matches_sparse_nfa_rows() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        assert!(tokenizer.has_epsilon_transitions());
        assert!(!tokenizer.has_deterministic_dispatch());

        let vocab = Vocab::new(
            vec![
                (0, b"".to_vec()),
                (1, b"a".to_vec()),
                (2, b"aa".to_vec()),
                (3, b"ab".to_vec()),
                (4, b"b".to_vec()),
                (5, b"ba".to_vec()),
                (6, b"x".to_vec()),
            ],
            None,
        );
        let artifacts = get_ordered_vocab_trie_artifacts_for_vocab(&vocab).0;
        let raw_states = (0..tokenizer.num_states()).collect::<Vec<_>>();
        let sparse = collect_sparse_root_possible_matches(
            &tokenizer,
            &artifacts.trie.root,
            &raw_states,
            None,
        );

        let mut relevant_bytes = [false; 256];
        for bytes in &artifacts.ordered_vocab.ordered_token_bytes {
            for &byte in bytes {
                relevant_bytes[byte as usize] = true;
            }
        }
        let powerset = build_possible_match_powerset_view(&tokenizer, &relevant_bytes);
        let mut view_entries = powerset.raw_start_to_view.clone();
        view_entries.sort_unstable();
        view_entries.dedup();
        let (powerset_rows, _) =
            collector::collect_possible_matches_interval_trie_class_build_for_flat_view(
                &powerset.tokenizer_view,
                tokenizer.num_terminals() as usize,
                &powerset.is_end,
                &artifacts.trie.root,
                &view_entries,
                Some(&powerset.boundary_state),
            );
        let sparse_expanded = expand_interval_class_maps(&sparse.class_maps);
        let powerset_expanded = expand_interval_class_maps(&powerset_rows.class_maps);

        for raw_state in raw_states {
            let sparse_class = sparse.state_classes[raw_state as usize];
            assert_ne!(sparse_class, u32::MAX, "raw_state={raw_state}");
            let view_state = powerset.raw_start_to_view[raw_state as usize] as usize;
            let powerset_class = powerset_rows.state_classes[view_state];
            assert_ne!(powerset_class, u32::MAX, "raw_state={raw_state}");
            assert_eq!(
                sparse_expanded[sparse_class as usize].as_ref(),
                powerset_expanded[powerset_class as usize].as_ref(),
                "raw_state={raw_state} view_state={view_state}",
            );
        }
    }

    #[test]
    fn epsilon_pm_vocab_equivalence_distinguishes_terminals_above_127() {
        let mut expressions = Vec::new();
        let mut partitions = Vec::new();
        for terminal in 0..130u32 {
            expressions.push(Expr::U8Seq(vec![terminal as u8]));
            partitions.push(terminal % 3);
        }
        let tokenizer = build_tokenizer_from_exprs_partitioned_with_adaptive(
            &expressions,
            None,
            &partitions,
            false,
        );
        assert!(tokenizer.has_epsilon_transitions());

        let vocab = Vocab::new(vec![(0, vec![128]), (1, vec![129])], None);
        let full_artifacts = get_ordered_vocab_trie_artifacts_for_vocab(&vocab).0;
        let classes = compute_pm_vocab_equivalence_map(
            &tokenizer,
            full_artifacts.ordered_vocab.as_ref(),
            full_artifacts.trie.as_ref(),
        );

        assert_ne!(
            classes.original_to_internal[0],
            classes.original_to_internal[1],
            "PM vocab equivalence must include terminal IDs above the old u128 ceiling",
        );
    }
}
