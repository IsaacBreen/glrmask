use std::collections::BTreeMap;
use std::hash::Hash;
use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};
use rustc_hash::FxHashMap;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::constraint_possible_matches::collector::DensePossibleMatchMap;
use crate::compiler::pm_profile::elapsed_ms;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::ds::weight::{Weight, finalize_weight_map, shared_rangeset};
use crate::grammar::flat::TerminalID;

pub(crate) mod collector;

pub(crate) type DensePossibleMatchesByState = BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>;
pub(crate) type RuntimePossibleMatchesByTerminal = BTreeMap<TerminalID, Weight>;
pub(crate) type PossibleMatchSignature = Vec<(u32, TerminalID)>;
pub(crate) type SeedStateSignature = Vec<u32>;
pub(crate) type SignatureClassId = u32;

#[derive(Debug, Clone)]
pub(crate) struct ConstraintVocabMap {
    pub(crate) original_to_internal: Vec<u32>,
    pub(crate) internal_to_originals: Vec<Vec<u32>>,
    pub(crate) old_internal_to_constraint: Vec<Vec<u32>>,
}

#[derive(Debug, Clone)]
pub(crate) struct ConstraintPossibleMatchesConfig<'a> {
    pub(crate) initial_state_map: Option<&'a ManyToOneIdMap>,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ConstraintPossibleMatchesProfile {
    pub(crate) possible_matches_collect_ms: f64,
    pub(crate) constraint_vocab_ms: f64,
}

#[derive(Debug)]
pub(crate) struct ConstraintPossibleMatchesComputation {
    pub(crate) possible_matches: RuntimePossibleMatchesByTerminal,
    pub(crate) constraint_vocab: ConstraintVocabMap,
    pub(crate) profile: ConstraintPossibleMatchesProfile,
}

pub(crate) fn build_internal_token_bytes_from_groups(
    vocab: &Vocab,
    internal_to_originals: &[Vec<u32>],
) -> BTreeMap<u32, Vec<u8>> {
    internal_to_originals
        .iter()
        .enumerate()
        .filter_map(|(internal_token_id, originals)| {
            let bytes = originals
                .iter()
                .find_map(|original| vocab.entries.get(original))?
                .clone();
            Some((internal_token_id as u32, bytes))
        })
        .collect()
}

pub(crate) fn dense_word_count(token_slots: u32) -> usize {
    (token_slots as usize + 63) / 64
}

fn set_dense_bit(words: &mut [u64], token_id: u32) {
    let word = token_id as usize / 64;
    let bit = token_id % 64;

    if let Some(slot) = words.get_mut(word) {
        *slot |= 1u64 << bit;
    }
}

pub(crate) fn dense_bit_is_set(words: &[u64], token_id: u32) -> bool {
    let word = token_id as usize / 64;
    let bit = token_id % 64;

    words
        .get(word)
        .map(|word| ((*word >> bit) & 1) != 0)
        .unwrap_or(false)
}

fn for_each_dense_bit(words: &[u64], mut f: impl FnMut(u32)) {
    for (word_idx, &word) in words.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let bit = bits.trailing_zeros();
            let token_id = word_idx as u32 * 64 + bit;
            f(token_id);
            bits &= bits - 1;
        }
    }
}

fn range_set_from_sorted_ids(ids: &[u32]) -> RangeSetBlaze<u32> {
    let Some((&first, rest)) = ids.split_first() else {
        return RangeSetBlaze::new();
    };

    let mut ranges = Vec::new();
    let mut start = first;
    let mut end = first;
    for &id in rest {
        if id == end + 1 {
            end = id;
        } else {
            ranges.push(start..=end);
            start = id;
            end = id;
        }
    }
    ranges.push(start..=end);
    RangeSetBlaze::from_iter(ranges)
}


fn unique_same_byte_groups(tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>) -> Vec<Arc<[u32]>> {
    let mut groups: Vec<_> = tokens_with_same_bytes
        .iter()
        .filter_map(|(&token_id, group)| {
            (group.first().copied() == Some(token_id)).then(|| Arc::clone(group))
        })
        .collect();
    groups.sort_unstable_by_key(|group| group[0]);
    groups
}

fn build_group_constraint_internal_ids(
    groups: &[Arc<[u32]>],
    original_to_constraint_internal: &[u32],
) -> FxHashMap<u32, Arc<[u32]>> {
    let mut result = FxHashMap::default();
    for group in groups {
        let leader = group[0];
        let mut ids = Vec::new();
        for &token_id in group.iter() {
            let Some(&constraint_internal_id) = original_to_constraint_internal.get(token_id as usize) else {
                continue;
            };
            if constraint_internal_id != u32::MAX {
                ids.push(constraint_internal_id);
            }
        }
        ids.sort_unstable();
        ids.dedup();
        result.insert(leader, Arc::from(ids));
    }
    result
}

fn remap_dense_bitmap_with_original_to_internal_ids(
    original_bitmap: &[u64],
    original_to_internal_ids: &[Arc<[u32]>],
    num_words: usize,
) -> RangeSetBlaze<u32> {
    let mut remapped = vec![0u64; num_words];
    for_each_dense_bit(original_bitmap, |original_token_id| {
        let Some(internal_ids) = original_to_internal_ids.get(original_token_id as usize) else {
            return;
        };
        for &internal_id in internal_ids.iter() {
            set_dense_bit(&mut remapped, internal_id);
        }
    });

    let mut ids = Vec::new();
    for_each_dense_bit(&remapped, |token_id| ids.push(token_id));
    range_set_from_sorted_ids(&ids)
}

/// Compose a short `state_classes` (covering only representative states) back
/// to the full DFA state space using `initial_state_map`.
/// Compose dense root `state_classes` through `initial_state_map`.
/// The collector returns classes indexed by original DFA state id for the
/// states it was seeded with; each initial-map class inherits the class of its
/// representative.
fn compose_state_classes_with_initial_map(
    state_classes: &[u32],
    initial_state_map: &ManyToOneIdMap,
) -> Vec<u32> {
    let num_dfa_states = initial_state_map.original_to_internal.len();
    let mut composed_state_classes = vec![u32::MAX; num_dfa_states];
    for (initial_internal, originals) in initial_state_map.internal_to_originals.iter().enumerate() {
        let Some(&initial_rep) = initial_state_map.representative_original_ids.get(initial_internal) else {
            continue;
        };
        let Some(&class_id) = state_classes.get(initial_rep as usize) else {
            continue;
        };
        if class_id == u32::MAX {
            continue;
        }
        for &original in originals {
            composed_state_classes[original as usize] = class_id;
        }
    }
    composed_state_classes
}

fn intern_signatures_by_group<T>(
    groups: Vec<Arc<[u32]>>,
    mut signatures_by_group: FxHashMap<u32, T>,
) -> FxHashMap<u32, SignatureClassId>
where
    T: Default + Eq + Hash,
{
    let mut signature_to_id: FxHashMap<T, SignatureClassId> = FxHashMap::default();
    let mut token_to_id = FxHashMap::default();
    let mut next_id: SignatureClassId = 0;

    for group in groups {
        let leader = group[0];
        let signature = signatures_by_group.remove(&leader).unwrap_or_default();
        let signature_id = *signature_to_id.entry(signature).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        for &token_id in group.iter() {
            token_to_id.insert(token_id, signature_id);
        }
    }

    token_to_id
}

pub(crate) fn intern_signature_ids<T>(
    signatures: FxHashMap<u32, T>,
) -> FxHashMap<u32, SignatureClassId>
where
    T: Eq + Hash,
{
    let mut signature_to_id: FxHashMap<T, SignatureClassId> = FxHashMap::default();
    let mut token_to_id = FxHashMap::default();
    let mut next_id: SignatureClassId = 0;

    for (token_id, signature) in signatures {
        let signature_id = *signature_to_id.entry(signature).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        token_to_id.insert(token_id, signature_id);
    }
    token_to_id
}

pub(crate) fn constraint_vocab_from_terminal_dwa_vocab(parser_vocab: &ManyToOneIdMap) -> ConstraintVocabMap {
    let old_internal_to_constraint = (0..parser_vocab.internal_to_originals.len())
        .map(|internal_id| vec![internal_id as u32])
        .collect();

    ConstraintVocabMap {
        original_to_internal: parser_vocab.original_to_internal.clone(),
        internal_to_originals: parser_vocab.internal_to_originals.clone(),
        old_internal_to_constraint,
    }
}

pub(crate) fn constraint_vocab_is_identity(constraint_vocab: &ConstraintVocabMap) -> bool {
    constraint_vocab
        .old_internal_to_constraint
        .iter()
        .enumerate()
        .all(|(internal_id, mapped)| mapped.len() == 1 && mapped[0] == internal_id as u32)
}

fn remap_dense_maps_to_constraint_vocab(
    maps: impl Iterator<Item = (u32, DensePossibleMatchMap)>,
    original_to_constraint_internal: &[u32],
    constraint_token_count: u32,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
) -> RuntimePossibleMatchesByTerminal {
    let num_words = dense_word_count(constraint_token_count);
    let mut remap_cache: FxHashMap<(TerminalID, Box<[u64]>), RangeSetBlaze<u32>> = FxHashMap::default();
    let groups = unique_same_byte_groups(tokens_with_same_bytes);
    let max_token_slot = tokens_with_same_bytes
        .keys()
        .max()
        .map(|token_id| *token_id as usize + 1)
        .unwrap_or(0);
    let mut group_leaders = vec![u32::MAX; max_token_slot];
    for group in &groups {
        let leader = group[0];
        for &token_id in group.iter() {
            group_leaders[token_id as usize] = leader;
        }
    }
    let group_constraint_internal_ids = build_group_constraint_internal_ids(&groups, original_to_constraint_internal);

    // Accumulate per-terminal (state_id, token_set) entries.
    let mut terminal_entries: BTreeMap<TerminalID, Vec<(u32, RangeSetBlaze<u32>)>> = BTreeMap::new();

    for (state_id, terminals) in maps {
        for (&terminal_id, original_bitmap) in &terminals {
            let cache_key = (terminal_id, original_bitmap.clone());
            let token_set = remap_cache
                .entry(cache_key)
                .or_insert_with(|| {
                    let mut remapped = vec![0u64; num_words];
                    let mut any = false;
                    for_each_dense_bit(original_bitmap, |original_token_id| {
                        let Some(&leader) = group_leaders.get(original_token_id as usize) else {
                            return;
                        };
                        if leader == u32::MAX {
                            return;
                        }
                        let Some(constraint_internal_ids) = group_constraint_internal_ids.get(&leader) else {
                            return;
                        };
                        for &constraint_internal_id in constraint_internal_ids.iter() {
                            set_dense_bit(&mut remapped, constraint_internal_id);
                            any = true;
                        }
                    });
                    if any {
                        let mut ids = Vec::new();
                        for_each_dense_bit(&remapped, |token_id| ids.push(token_id));
                        range_set_from_sorted_ids(&ids)
                    } else {
                        RangeSetBlaze::new()
                    }
                })
                .clone();

            if !token_set.is_empty() {
                terminal_entries.entry(terminal_id).or_default().push((state_id, token_set));
            }
        }
    }

    terminal_entries
        .into_iter()
        .map(|(terminal_id, mut entries)| {
            entries.sort_by_key(|(state, _)| *state);
            let weight = Weight::from_per_tsid_token_sets(
                entries.into_iter().map(|(state, tokens)| (state, tokens)),
            );
            (terminal_id, weight)
        })
        .collect()
}

fn build_terminal_vocab_token_entries(
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    original_to_internal: &[u32],
) -> Vec<(usize, Vec<u8>)> {
    token_bytes
        .iter()
        .filter_map(|(&original_token_id, bytes)| {
            let internal_id = original_to_internal
                .get(original_token_id as usize)
                .copied()
                .unwrap_or(u32::MAX);
            (internal_id != u32::MAX).then(|| (internal_id as usize, bytes.clone()))
        })
        .collect()
}

fn range_set_from_dense_words(words: &[u64]) -> RangeSetBlaze<u32> {
    let mut ids = Vec::new();
    for_each_dense_bit(words, |token_id| ids.push(token_id));
    range_set_from_sorted_ids(&ids)
}

pub(crate) fn class_maps_to_runtime_possible_matches_shared(
    class_maps: &[Arc<DensePossibleMatchMap>],
    state_classes: &[u32],
) -> RuntimePossibleMatchesByTerminal {
    let mut bitmap_cache: FxHashMap<Box<[u64]>, Arc<RangeSetBlaze<u32>>> = FxHashMap::default();
    let mut class_token_set_cache: FxHashMap<u32, BTreeMap<TerminalID, Arc<RangeSetBlaze<u32>>>> = FxHashMap::default();

    for (class_id, class_map) in class_maps.iter().enumerate() {
        let mut terminal_sets = BTreeMap::new();
        for (&terminal_id, bitmap) in class_map.iter() {
            let token_set = bitmap_cache
                .entry(bitmap.clone())
                .or_insert_with(|| shared_rangeset(range_set_from_dense_words(bitmap)))
                .clone();
            if !token_set.is_empty() {
                terminal_sets.insert(terminal_id, token_set);
            }
        }
        class_token_set_cache.insert(class_id as u32, terminal_sets);
    }

    let mut terminal_entries: BTreeMap<TerminalID, Vec<(u32, Arc<RangeSetBlaze<u32>>)>> = BTreeMap::new();
    for (state_id, &class_id) in state_classes.iter().enumerate() {
        if class_id == u32::MAX {
            continue;
        }
        let Some(terminal_sets) = class_token_set_cache.get(&class_id) else {
            continue;
        };
        for (&terminal_id, token_set) in terminal_sets {
            terminal_entries
                .entry(terminal_id)
                .or_default()
                .push((state_id as u32, Arc::clone(token_set)));
        }
    }

    terminal_entries
        .into_iter()
        .map(|(terminal_id, mut entries)| {
            entries.sort_by_key(|(state, _)| *state);
            let weight = Weight::from_per_tsid_shared(entries.into_iter());
            (terminal_id, weight)
        })
        .collect()
}

fn remap_token_set_to_constraint_vocab(
    old_tokens: &RangeSetBlaze<u32>,
    old_internal_to_constraint: &[Vec<u32>],
) -> RangeSetBlaze<u32> {
    let mut new_ids = Vec::new();
    for old_internal_token in old_tokens.iter() {
        if let Some(mapped_ids) = old_internal_to_constraint.get(old_internal_token as usize) {
            new_ids.extend_from_slice(mapped_ids);
        }
    }
    new_ids.sort_unstable();
    new_ids.dedup();
    range_set_from_sorted_ids(&new_ids)
}

fn remap_arc_token_set_to_constraint_vocab(
    token_set: &Arc<RangeSetBlaze<u32>>,
    old_internal_to_constraint: &[Vec<u32>],
    token_set_cache: &mut FxHashMap<usize, Arc<RangeSetBlaze<u32>>>,
) -> Arc<RangeSetBlaze<u32>> {
    let cache_key = Arc::as_ptr(token_set) as usize;
    if let Some(cached) = token_set_cache.get(&cache_key) {
        return Arc::clone(cached);
    }
    let remapped = shared_rangeset(remap_token_set_to_constraint_vocab(token_set, old_internal_to_constraint));
    token_set_cache.insert(cache_key, Arc::clone(&remapped));
    remapped
}

pub(crate) fn remap_weight_to_constraint_vocab(
    weight: &Weight,
    old_internal_to_constraint: &[Vec<u32>],
    token_set_cache: &mut FxHashMap<usize, Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    if weight.is_full() {
        return Weight::all();
    }

    let mut remapped = RangeMapBlaze::new();
    for (start, end, token_set) in weight.compact_entries().unwrap_or_default() {
        let remapped_token_set = remap_arc_token_set_to_constraint_vocab(&token_set, old_internal_to_constraint, token_set_cache);
        if !remapped_token_set.is_empty() {
            remapped.extend_simple(std::iter::once((start..=end, remapped_token_set)));
        }
    }
    finalize_weight_map(remapped)
}

pub(crate) fn compute_constraint_possible_matches(
    tokenizer: &Tokenizer,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    internal_ids: &InternalIdMap,
    config: ConstraintPossibleMatchesConfig,
) -> ConstraintPossibleMatchesComputation {
    let pm_started_at = Instant::now();

    let token_entries = build_terminal_vocab_token_entries(
        token_bytes,
        &internal_ids.vocab_tokens.original_to_internal,
    );
    let trie = VocabPrefixTree::build_owned(token_entries);

    let terminal_vocab_token_count = internal_ids.vocab_tokens.num_internal_ids();

    let trie_build_states: Vec<u32> = match config.initial_state_map {
        Some(init_map) => init_map.representative_original_ids.clone(),
        None => (0..tokenizer.num_states()).collect(),
    };
    let (mut trie_class_result, _) = collector::collect_possible_matches_dense_trie_class_build_with_classes(
        tokenizer,
        &trie.root,
        terminal_vocab_token_count,
        &trie_build_states,
    );
    if let Some(init_map) = config.initial_state_map {
        trie_class_result.state_classes = compose_state_classes_with_initial_map(
            &trie_class_result.state_classes,
            init_map,
        );
    }
    let possible_matches_collect_ms = elapsed_ms(pm_started_at);

    let constraint_vocab_started_at = Instant::now();

    // Fast path: possible-matches are produced in terminal-DWA vocab space,
    // so there is no need to split vocab tokens by possible-match or seed-state
    // signatures. Use an identity constraint-vocab that preserves the existing
    // terminal-DWA token groupings.
    let constraint_vocab = constraint_vocab_from_terminal_dwa_vocab(&internal_ids.vocab_tokens);

    let possible_matches = class_maps_to_runtime_possible_matches_shared(
        &trie_class_result.class_maps,
        &trie_class_result.state_classes,
    );

    let constraint_vocab_ms = elapsed_ms(constraint_vocab_started_at);

    ConstraintPossibleMatchesComputation {
        possible_matches,
        constraint_vocab,
        profile: ConstraintPossibleMatchesProfile {
            possible_matches_collect_ms,
            constraint_vocab_ms,
        },
    }
}