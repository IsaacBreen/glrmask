//! Ordered-vocabulary construction and cache management for scan-relation building.
//!
//! The scan relation is computed over a byte-sorted vocabulary trie, but the
//! public input vocabulary is indexed by original tokenizer ids.  This module
//! owns exactly that quotient/ordering boundary and nothing about parser stacks
//! or automata weights.

use super::prelude::*;
use super::types::*;

pub(super) struct OrderedVocab {
    pub(super) original_slot_count: usize,
    pub(super) ordered_to_originals: Vec<Vec<u32>>,
    pub(super) ordered_token_bytes: Vec<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub(super) struct OrderedVocabTrieArtifacts {
    pub(super) ordered_vocab: Arc<OrderedVocab>,
    pub(super) trie: Arc<VocabPrefixTree>,
}

impl VocabDerivedArtifact for OrderedVocabTrieArtifacts {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct OrderedVocabCacheFingerprint {
    token_count: usize,
    max_token_id: u32,
    total_bytes: usize,
    hash: u64,
}

#[derive(Debug, Clone)]
pub(super) struct OrderedVocabCacheEntry {
    fingerprint: OrderedVocabCacheFingerprint,
    source_original_to_ordered: Arc<[u32]>,
    artifacts: OrderedVocabTrieArtifacts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OrderedVocabCacheStatus {
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
pub(super) struct OrderedVocabCacheProfile {
    status: OrderedVocabCacheStatus,
    probe_ns: u128,
    verify_ns: u128,
    ordered_vocab_build_ns: u128,
    trie_build_ns: u128,
    cache_entries: usize,
    capacity: usize,
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
        std::env::var("GLRMASK_SCAN_RELATION_ORDERED_VOCAB_CACHE")
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
        std::env::var("GLRMASK_SCAN_RELATION_ORDERED_VOCAB_CACHE_CAPACITY")
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

pub(super) fn emit_ordered_vocab_cache_profile(profile: OrderedVocabCacheProfile) {
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

pub(super) fn get_ordered_vocab_trie_artifacts(
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

pub(super) fn get_ordered_vocab_trie_artifacts_for_vocab(
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

pub(super) fn range_set_from_sorted_ids(ids: &[u32]) -> RangeSetBlaze<u32> {
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

pub(super) fn range_set_from_u128_mask(mask: u128) -> RangeSetBlaze<u32> {
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
