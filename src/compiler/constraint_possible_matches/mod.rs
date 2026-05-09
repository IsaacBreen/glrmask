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
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::ds::weight::{shared_rangeset, Weight};
use crate::grammar::flat::TerminalID;
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

#[derive(Debug, Clone)]
pub(crate) struct ConstraintPossibleMatchesConfig<'a> {
    pub(crate) initial_state_map: Option<&'a ManyToOneIdMap>,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ConstraintPossibleMatchesProfile {
    pub(crate) possible_matches_collect_ms: f64,
    pub(crate) possible_match_vocab_ms: f64,
}

#[derive(Debug)]
pub(crate) struct ConstraintPossibleMatchesComputation {
    pub(crate) mapped_possible_matches: MappedArtifact<RuntimePossibleMatchesByTerminal>,
    pub(crate) profile: ConstraintPossibleMatchesProfile,
}

#[derive(Debug, Clone)]
struct OrderedVocab {
    original_slot_count: usize,
    ordered_to_originals: Vec<Vec<u32>>,
    ordered_token_bytes: Vec<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct OrderedVocabTrieArtifacts {
    ordered_vocab: Arc<OrderedVocab>,
    trie: Arc<VocabPrefixTree>,
}

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

    OrderedVocab { original_slot_count, ordered_to_originals, ordered_token_bytes }
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

fn compose_state_classes_with_initial_map(state_classes: &[u32], initial_state_map: &ManyToOneIdMap) -> Vec<u32> {
    let num_dfa_states = initial_state_map.original_to_internal.len();
    let mut composed_state_classes = vec![u32::MAX; num_dfa_states];
    for (initial_internal, originals) in initial_state_map.internal_to_originals.iter().enumerate() {
        let Some(&initial_rep) = initial_state_map.representative_original_ids.get(initial_internal) else { continue; };
        let Some(&class_id) = state_classes.get(initial_rep as usize) else { continue; };
        if class_id == u32::MAX { continue; }
        for &original in originals { composed_state_classes[original as usize] = class_id; }
    }
    composed_state_classes
}

fn canonical_states_from_initial_map(initial_state_map: &ManyToOneIdMap, num_states: u32) -> Vec<u32> {
    let mut canonical: Vec<u32> = (0..num_states).collect();
    for (state, &internal) in initial_state_map.original_to_internal.iter().enumerate() {
        if internal == u32::MAX { continue; }
        let Some(&representative) = initial_state_map.representative_original_ids.get(internal as usize) else { continue; };
        if representative == u32::MAX { continue; }
        if let Some(slot) = canonical.get_mut(state) { *slot = representative; }
    }
    canonical
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

fn apply_sweep_events(active_group_counts: &mut [u32], events: &[SweepEvent], active_group_count: &mut usize) {
    for event in events.iter().filter(|event| !event.add) {
        let count = &mut active_group_counts[event.group_id as usize];
        assert!(*count > 0, "pmv sweep removal underflow for group_id={}", event.group_id);
        if *count == 1 {
            *active_group_count -= 1;
        }
        *count -= 1;
    }
    for event in events.iter().filter(|event| event.add) {
        let count = &mut active_group_counts[event.group_id as usize];
        if *count == 0 {
            *active_group_count += 1;
        }
        *count += 1;
    }
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

fn build_active_group_ids(
    active_group_counts: &[u32],
    active_group_count: usize,
) -> Vec<u32> {
    if active_group_count == 0 { return Vec::new(); }

    let mut active_group_ids = Vec::with_capacity(active_group_count);
    for (group_id, &count) in active_group_counts.iter().enumerate() {
        if count > 0 {
            active_group_ids.push(group_id as u32);
        }
    }
    active_group_ids
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
    let mut active_group_signature_to_signature_id: FxHashMap<Vec<u32>, SignatureClassId> = FxHashMap::default();
    let mut signature_labels: Vec<Vec<StateTerminalLabel>> = Vec::new();
    let mut original_to_internal = vec![u32::MAX; ordered_vocab.original_slot_count];
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut active_group_counts = vec![0u32; groups.len()];
    let mut active_group_count = 0usize;
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
            apply_sweep_events(&mut active_group_counts, &events[position], &mut active_group_count);
            event_index += 1;
        }

        let next_position = event_positions.get(event_index).map(|&next| (next as usize).min(num_ordered_tokens)).unwrap_or(num_ordered_tokens);
        let active_group_signature_started_at = Instant::now();
        let active_group_ids = build_active_group_ids(&active_group_counts, active_group_count);
        active_group_signature_build_ms += elapsed_ms(active_group_signature_started_at);
        sweep_segments += 1;
        total_active_group_len += active_group_ids.len();
        max_active_group_len = max_active_group_len.max(active_group_ids.len());

        let signature_lookup_started_at = Instant::now();
        let signature_id = if let Some(&existing) = active_group_signature_to_signature_id.get(&active_group_ids) {
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
            active_group_signature_to_signature_id.insert(active_group_ids, signature_id);
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
                internal_to_originals[signature_id as usize].push(original);
            }
        }
        assignment_ms += elapsed_ms(assignment_started_at);
        position = next_position;
    }
    let sweep_ms = elapsed_ms(sweep_started_at);

    let sort_dedup_started_at = Instant::now();
    for originals in &mut internal_to_originals { originals.sort_unstable(); originals.dedup(); }
    let sort_dedup_ms = elapsed_ms(sort_dedup_started_at);

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
            for (state, mut ids) in by_state {
                ids.sort_unstable();
                ids.dedup();
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
    let mut seen = vec![false; tokenizer.num_terminals as usize];
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

pub(crate) fn compute_constraint_possible_matches(
    tokenizer: &Tokenizer,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    config: ConstraintPossibleMatchesConfig,
) -> ConstraintPossibleMatchesComputation {
    let pm_started_at = Instant::now();

    let (artifacts, ordered_vocab_cache_profile) = get_ordered_vocab_trie_artifacts(token_bytes);
    emit_ordered_vocab_cache_profile(ordered_vocab_cache_profile);
    let ordered_vocab = artifacts.ordered_vocab;
    let trie = artifacts.trie;

    let trie_build_states: Vec<u32> = match config.initial_state_map {
        Some(init_map) => init_map.representative_original_ids.clone(),
        None => (0..tokenizer.num_states()).collect(),
    };
    let canonical_states = config
        .initial_state_map
        .map(|init_map| canonical_states_from_initial_map(init_map, tokenizer.num_states()));

    let root_terminal_union = root_terminal_union_count(tokenizer, &trie_build_states);
    let use_sparse_root_collect = sparse_root_collect_enabled()
        && trie_build_states.len() <= sparse_root_state_limit()
        && root_terminal_union <= sparse_root_terminal_limit();

    let mut trie_class_result = if use_sparse_root_collect {
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
            canonical_states.as_deref(),
        )
    } else {
        collector::collect_possible_matches_interval_trie_class_build_with_classes(
            tokenizer,
            &trie.root,
            &trie_build_states,
            canonical_states.as_deref(),
        )
        .0
    };

    if let Some(init_map) = config.initial_state_map {
        trie_class_result.state_classes = compose_state_classes_with_initial_map(&trie_class_result.state_classes, init_map);
    }

    let possible_matches_collect_ms = elapsed_ms(pm_started_at);

    let possible_match_vocab_started_at = Instant::now();
    let (possible_match_vocab, possible_matches) = build_possible_match_vocab_and_weights_from_interval_maps(&trie_class_result.class_maps, &trie_class_result.state_classes, ordered_vocab.as_ref());

    let possible_matches_id_map = InternalIdMap {
        tokenizer_states: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            trie_class_result.state_classes.clone(),
            trie_class_result.state_classes.iter().copied().filter(|&class_id| class_id != u32::MAX).max().map(|class_id| class_id + 1).unwrap_or(0),
        ),
        vocab_tokens: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            possible_match_vocab.original_to_internal.clone(),
            possible_match_vocab.internal_to_originals.len() as u32,
        ),
    };

    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some() || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
        eprintln!("[glrmask/profile][possible_match_vocab] original_tokens={} ordered_byte_tokens={} possible_match_tokens={}", token_bytes.len(), ordered_vocab.ordered_to_originals.len(), possible_matches_id_map.vocab_tokens.internal_to_originals.len());
    }

    let possible_match_vocab_ms = elapsed_ms(possible_match_vocab_started_at);

    ConstraintPossibleMatchesComputation {
        mapped_possible_matches: MappedArtifact::new(possible_matches, possible_matches_id_map),
        profile: ConstraintPossibleMatchesProfile { possible_matches_collect_ms, possible_match_vocab_ms },
    }
}
