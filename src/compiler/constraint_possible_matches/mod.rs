use std::collections::BTreeMap;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::constraint_possible_matches::collector::DensePossibleMatchMap;
use crate::compiler::pm_profile::elapsed_ms;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::ds::weight::{Weight, shared_rangeset};
use crate::grammar::flat::TerminalID;

pub(crate) mod collector;

pub(crate) type DensePossibleMatchesByState = BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>;
pub(crate) type RuntimePossibleMatchesByTerminal = BTreeMap<TerminalID, Weight>;
pub(crate) type PossibleMatchSignature = Vec<(u32, TerminalID)>;
pub(crate) type SeedStateSignature = Vec<u32>;
pub(crate) type SignatureClassId = u32;

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
    pub(crate) possible_matches: RuntimePossibleMatchesByTerminal,
    pub(crate) id_map: InternalIdMap,
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

pub(crate) fn max_original_token_slot(token_bytes: &BTreeMap<u32, Vec<u8>>) -> u32 {
    token_bytes
        .keys()
        .next_back()
        .map(|token_id| token_id.saturating_add(1))
        .unwrap_or(0)
}

fn set_dense_bit(words: &mut [u64], token_id: u32) {
    let word = token_id as usize / 64;
    let bit = token_id % 64;

    if let Some(slot) = words.get_mut(word) {
        *slot |= 1u64 << bit;
    }
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

pub(crate) fn build_tokens_with_same_bytes(
    token_bytes: &BTreeMap<u32, Vec<u8>>,
) -> FxHashMap<u32, Arc<[u32]>> {
    let mut by_bytes: BTreeMap<Vec<u8>, Vec<u32>> = BTreeMap::new();
    for (&token_id, bytes) in token_bytes {
        by_bytes.entry(bytes.clone()).or_default().push(token_id);
    }

    let mut tokens_with_same_bytes = FxHashMap::default();
    for (_, mut token_ids) in by_bytes {
        token_ids.sort_unstable();
        token_ids.dedup();
        let shared: Arc<[u32]> = Arc::from(token_ids.clone());
        for &token_id in &token_ids {
            tokens_with_same_bytes.insert(token_id, Arc::clone(&shared));
        }
    }

    tokens_with_same_bytes
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

fn build_same_byte_group_leaders(
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    groups: &[Arc<[u32]>],
) -> Vec<u32> {
    let mut leaders = vec![u32::MAX; max_original_token_slot(token_bytes) as usize];
    for group in groups {
        let leader = group[0];
        for &token_id in group.iter() {
            leaders[token_id as usize] = leader;
        }
    }
    leaders
}

fn build_group_possible_match_internal_ids(
    groups: &[Arc<[u32]>],
    original_to_possible_match_internal: &[u32],
) -> FxHashMap<u32, Arc<[u32]>> {
    let mut result = FxHashMap::default();
    for group in groups {
        let leader = group[0];
        let mut ids = Vec::new();
        for &token_id in group.iter() {
            let Some(&possible_match_internal_id) = original_to_possible_match_internal.get(token_id as usize) else {
                continue;
            };
            if possible_match_internal_id != u32::MAX {
                ids.push(possible_match_internal_id);
            }
        }
        ids.sort_unstable();
        ids.dedup();
        result.insert(leader, Arc::from(ids));
    }
    result
}

fn build_original_to_possible_match_internal_ids(
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
    original_to_possible_match_internal: &[u32],
) -> Vec<Arc<[u32]>> {
    let max_token_slot = tokens_with_same_bytes
        .keys()
        .max()
        .map(|token_id| *token_id as usize + 1)
        .unwrap_or(0);
    let mut result = vec![Arc::<[u32]>::from([]); max_token_slot];

    for group in unique_same_byte_groups(tokens_with_same_bytes) {
        let mut internal_ids = Vec::new();
        for &token_id in group.iter() {
            let Some(&internal_id) = original_to_possible_match_internal.get(token_id as usize) else {
                continue;
            };
            if internal_id != u32::MAX {
                internal_ids.push(internal_id);
            }
        }
        internal_ids.sort_unstable();
        internal_ids.dedup();
        let shared: Arc<[u32]> = Arc::from(internal_ids);
        for &token_id in group.iter() {
            if (token_id as usize) < result.len() {
                result[token_id as usize] = Arc::clone(&shared);
            }
        }
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

fn used_state_class_ids(state_classes: &[u32]) -> Vec<u32> {
    let mut ids: Vec<u32> = state_classes
        .iter()
        .copied()
        .filter(|&class_id| class_id != u32::MAX)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
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

pub(crate) fn build_possible_match_signature_ids_from_trie_classes(
    class_maps: &[Arc<DensePossibleMatchMap>],
    state_classes: &[u32],
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
) -> FxHashMap<u32, SignatureClassId> {
    let groups = unique_same_byte_groups(tokens_with_same_bytes);
    let group_leaders = build_same_byte_group_leaders(token_bytes, &groups);
    let used_class_ids = used_state_class_ids(state_classes);
    let mut signatures_by_group: FxHashMap<u32, PossibleMatchSignature> =
        groups.iter().map(|group| (group[0], Vec::new())).collect();

    let signature_entries: Vec<(u32, (u32, TerminalID))> = used_class_ids
        .par_iter()
        .flat_map_iter(|&class_id| {
            let mut entries = Vec::new();
            let Some(terminals) = class_maps.get(class_id as usize) else {
                return entries;
            };
            for (&terminal_id, bitmap) in terminals.iter() {
                for_each_dense_bit(bitmap, |original_token_id| {
                    let Some(&leader) = group_leaders.get(original_token_id as usize) else {
                        return;
                    };
                    if leader != u32::MAX {
                        entries.push((leader, (class_id, terminal_id)));
                    }
                });
            }
            entries
        })
        .collect();

    for (leader, item) in signature_entries {
        if let Some(signature) = signatures_by_group.get_mut(&leader) {
            signature.push(item);
        }
    }

    for signature in signatures_by_group.values_mut() {
        signature.sort_unstable();
        signature.dedup();
    }

    intern_signatures_by_group(groups, signatures_by_group)
}

pub(crate) fn build_seed_state_signature_ids_from_trie_classes_exact(
    tokenizer: &Tokenizer,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
    state_classes: &[u32],
) -> FxHashMap<u32, SignatureClassId> {
    let groups = unique_same_byte_groups(tokens_with_same_bytes);

    let mut class_to_rep: FxHashMap<u32, u32> = FxHashMap::default();
    for (state, &class_id) in state_classes.iter().enumerate() {
        if class_id == u32::MAX {
            continue;
        }
        class_to_rep.entry(class_id).or_insert(state as u32);
    }
    let mut rep_entries: Vec<(u32, u32)> = class_to_rep.into_iter().collect();
    rep_entries.sort_unstable_by_key(|(class_id, _)| *class_id);
    let state_class_count = rep_entries.len();
    let num_leaders = groups.len();

    let mut leader_idx_by_token = vec![u32::MAX; max_original_token_slot(token_bytes) as usize];
    let mut leader_entries = Vec::with_capacity(num_leaders);
    for (leader_idx, group) in groups.iter().enumerate() {
        let leader = group[0];
        if let Some(slot) = leader_idx_by_token.get_mut(leader as usize) {
            *slot = leader_idx as u32;
        }
        let bytes = token_bytes
            .get(&leader)
            .expect("leader must have bytes")
            .clone();
        leader_entries.push((leader as usize, bytes));
    }
    let leader_trie = VocabPrefixTree::build_owned(leader_entries);

    let num_states = tokenizer.num_states() as usize;
    let flat_transitions: Vec<[u32; 256]> = (0..num_states)
        .map(|state_idx| {
            let dfa_state = &tokenizer.dfa.states()[state_idx];
            let mut flat = [u32::MAX; 256];
            for (byte, &target) in dfa_state.transitions.iter() {
                flat[byte as usize] = target;
            }
            flat
        })
        .collect();
    let terminal_states: Vec<bool> = (0..num_states)
        .map(|state| !tokenizer.dfa.finalizers(state as u32).is_zero())
        .collect();

    let class_words = (state_class_count + 63) / 64;
    let rep_word_mask: Vec<(usize, u64)> = (0..state_class_count)
        .map(|rep_idx| (rep_idx / 64, 1u64 << (rep_idx % 64)))
        .collect();
    let signatures_atomic: Vec<AtomicU64> = (0..num_leaders * class_words)
        .map(|_| AtomicU64::new(0))
        .collect();

    rep_entries
        .par_iter()
        .enumerate()
        .for_each(|(rep_idx, (_class_id, rep_state))| {
            let (word, mask) = rep_word_mask[rep_idx];
            let mut on_match = |leader_idx: u32| {
                let offset = leader_idx as usize * class_words + word;
                signatures_atomic[offset].fetch_or(mask, Ordering::Relaxed);
            };
            collect_seed_signature_matches_from_trie(
                &flat_transitions,
                &terminal_states,
                &leader_trie.root,
                *rep_state,
                &leader_idx_by_token,
                &mut on_match,
            );
        });

    let signatures_flat: Vec<u64> = signatures_atomic
        .into_iter()
        .map(AtomicU64::into_inner)
        .collect();

    use std::hash::Hasher;
    use rustc_hash::FxHasher;

    fn hash_slice(slice: &[u64]) -> u64 {
        let mut hasher = FxHasher::default();
        for &word in slice {
            hasher.write_u64(word);
        }
        hasher.finish()
    }

    let mut leader_hashes: Vec<(u64, usize)> = Vec::with_capacity(num_leaders);
    for leader_idx in 0..num_leaders {
        let base = leader_idx * class_words;
        let hash = hash_slice(&signatures_flat[base..base + class_words]);
        leader_hashes.push((hash, leader_idx));
    }
    leader_hashes.sort_unstable_by_key(|(hash, _)| *hash);

    let mut leader_to_interned: Vec<SignatureClassId> = vec![0; num_leaders];
    let mut next_class: SignatureClassId = 0;
    let mut index = 0;
    while index < leader_hashes.len() {
        let hash = leader_hashes[index].0;
        let mut end = index + 1;
        while end < leader_hashes.len() && leader_hashes[end].0 == hash {
            end += 1;
        }
        let mut slice_to_class: FxHashMap<&[u64], SignatureClassId> = FxHashMap::default();
        for entry in &leader_hashes[index..end] {
            let leader_idx = entry.1;
            let base = leader_idx * class_words;
            let slice = &signatures_flat[base..base + class_words];
            let class_id = *slice_to_class.entry(slice).or_insert_with(|| {
                let id = next_class;
                next_class += 1;
                id
            });
            leader_to_interned[leader_idx] = class_id;
        }
        index = end;
    }

    let mut token_to_id = FxHashMap::with_capacity_and_hasher(
        tokens_with_same_bytes.len(),
        Default::default(),
    );
    for (leader_idx, group) in groups.iter().enumerate() {
        let interned_id = leader_to_interned[leader_idx];
        for &token_id in group.iter() {
            token_to_id.insert(token_id, interned_id);
        }
    }

    token_to_id
}

#[inline]
fn push_reachable_leader_indices<F>(
    node: &VocabPrefixTreeNode,
    leader_idx_by_token: &[u32],
    on_match: &mut F,
) where
    F: FnMut(u32),
{
    for range in node.reachable_token_ids().ranges() {
        for token_id in *range.start()..=*range.end() {
            let idx = leader_idx_by_token
                .get(token_id)
                .copied()
                .unwrap_or(u32::MAX);
            if idx != u32::MAX {
                on_match(idx);
            }
        }
    }
}

#[inline]
fn scan_seed_signature_edge(
    flat_transitions: &[[u32; 256]],
    terminal_states: &[bool],
    mut state: u32,
    edge: &[u8],
) -> Option<(u32, bool)> {
    for &byte in edge {
        let next = flat_transitions[state as usize][byte as usize];
        if next == u32::MAX {
            return None;
        }
        state = next;
        if terminal_states[state as usize] {
            return Some((state, true));
        }
    }
    Some((state, false))
}

fn collect_seed_signature_matches_from_trie<F>(
    flat_transitions: &[[u32; 256]],
    terminal_states: &[bool],
    node: &VocabPrefixTreeNode,
    state: u32,
    leader_idx_by_token: &[u32],
    on_match: &mut F,
) where
    F: FnMut(u32),
{
    if node.has_token() {
        let token_id = node.token_id();
        let idx = leader_idx_by_token
            .get(token_id)
            .copied()
            .unwrap_or(u32::MAX);
        if idx != u32::MAX {
            on_match(idx);
        }
    }

    for (edge, child) in node.iter_children() {
        let Some((next_state, accepted)) =
            scan_seed_signature_edge(flat_transitions, terminal_states, state, edge)
        else {
            continue;
        };

        if accepted {
            push_reachable_leader_indices(child, leader_idx_by_token, on_match);
        } else {
            collect_seed_signature_matches_from_trie(
                flat_transitions,
                terminal_states,
                child,
                next_state,
                leader_idx_by_token,
                on_match,
            );
        }
    }
}

pub(crate) fn build_possible_match_vocab_map(
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    possible_match_signature_ids: &FxHashMap<u32, SignatureClassId>,
    seed_state_signature_ids: &FxHashMap<u32, SignatureClassId>,
) -> PossibleMatchVocabMap {
    let max_original_slot = token_bytes
        .keys()
        .next_back()
        .map(|token_id| *token_id as usize + 1)
        .unwrap_or(0);

    let mut original_to_internal = vec![u32::MAX; max_original_slot];
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut groups: BTreeMap<(SignatureClassId, SignatureClassId), Vec<u32>> = BTreeMap::new();

    for &original_token_id in token_bytes.keys() {
        let signature = possible_match_signature_ids
            .get(&original_token_id)
            .copied()
            .unwrap_or_default();
        let seed_signature = seed_state_signature_ids
            .get(&original_token_id)
            .copied()
            .unwrap_or_default();
        groups
            .entry((signature, seed_signature))
            .or_default()
            .push(original_token_id);
    }

    for (_, mut originals) in groups {
        originals.sort_unstable();
        originals.dedup();
        let new_internal_id = internal_to_originals.len() as u32;
        for &original_token_id in &originals {
            original_to_internal[original_token_id as usize] = new_internal_id;
        }
        internal_to_originals.push(originals);
    }

    PossibleMatchVocabMap {
        original_to_internal,
        internal_to_originals,
    }
}

pub(crate) fn remap_possible_matches_to_possible_match_vocab(
    raw_possible_matches: DensePossibleMatchesByState,
    original_to_possible_match_internal: &[u32],
    possible_match_token_count: u32,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
) -> RuntimePossibleMatchesByTerminal {
    remap_dense_maps_to_possible_match_vocab(
        raw_possible_matches.into_iter(),
        original_to_possible_match_internal,
        possible_match_token_count,
        tokens_with_same_bytes,
    )
}

fn remap_dense_maps_to_possible_match_vocab(
    maps: impl Iterator<Item = (u32, DensePossibleMatchMap)>,
    original_to_possible_match_internal: &[u32],
    possible_match_token_count: u32,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
) -> RuntimePossibleMatchesByTerminal {
    let num_words = dense_word_count(possible_match_token_count);
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
    let group_possible_match_internal_ids =
        build_group_possible_match_internal_ids(&groups, original_to_possible_match_internal);

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
                        let Some(possible_match_internal_ids) = group_possible_match_internal_ids.get(&leader) else {
                            return;
                        };
                        for &possible_match_internal_id in possible_match_internal_ids.iter() {
                            set_dense_bit(&mut remapped, possible_match_internal_id);
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

pub(crate) fn remap_class_maps_to_possible_match_vocab(
    class_maps: &[Arc<DensePossibleMatchMap>],
    state_classes: &[u32],
    original_to_possible_match_internal: &[u32],
    possible_match_token_count: u32,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
) -> RuntimePossibleMatchesByTerminal {
    let num_words = dense_word_count(possible_match_token_count);
    let mut remap_cache: FxHashMap<(TerminalID, Box<[u64]>), RangeSetBlaze<u32>> = FxHashMap::default();
    let mut class_token_set_cache: FxHashMap<u32, BTreeMap<TerminalID, RangeSetBlaze<u32>>> = FxHashMap::default();
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
    let group_possible_match_internal_ids =
        build_group_possible_match_internal_ids(&groups, original_to_possible_match_internal);

    let mut terminal_entries: BTreeMap<TerminalID, Vec<(u32, RangeSetBlaze<u32>)>> = BTreeMap::new();

    for class_id in used_state_class_ids(state_classes) {
        let Some(class_map) = class_maps.get(class_id as usize) else {
            continue;
        };

        let remapped_terminals = if let Some(cached) = class_token_set_cache.get(&class_id) {
            cached.clone()
        } else {
            let mut remapped = BTreeMap::new();
            for (&terminal_id, original_bitmap) in class_map.iter() {
                let cache_key = (terminal_id, original_bitmap.clone());
                let token_set = remap_cache
                    .entry(cache_key)
                    .or_insert_with(|| {
                        let mut remapped_bits = vec![0u64; num_words];
                        let mut any = false;
                        for_each_dense_bit(original_bitmap, |original_token_id| {
                            let Some(&leader) = group_leaders.get(original_token_id as usize) else {
                                return;
                            };
                            if leader == u32::MAX {
                                return;
                            }
                            let Some(possible_match_internal_ids) =
                                group_possible_match_internal_ids.get(&leader)
                            else {
                                return;
                            };
                            for &possible_match_internal_id in possible_match_internal_ids.iter() {
                                set_dense_bit(&mut remapped_bits, possible_match_internal_id);
                                any = true;
                            }
                        });
                        if any {
                            let mut ids = Vec::new();
                            for_each_dense_bit(&remapped_bits, |token_id| ids.push(token_id));
                            range_set_from_sorted_ids(&ids)
                        } else {
                            RangeSetBlaze::new()
                        }
                    })
                    .clone();
                if !token_set.is_empty() {
                    remapped.insert(terminal_id, token_set);
                }
            }
            class_token_set_cache.insert(class_id, remapped.clone());
            remapped
        };

        for (terminal_id, token_set) in remapped_terminals {
            terminal_entries.entry(terminal_id).or_default().push((class_id, token_set));
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

pub(crate) fn compute_constraint_possible_matches(
    tokenizer: &Tokenizer,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    config: ConstraintPossibleMatchesConfig,
) -> ConstraintPossibleMatchesComputation {
    let pm_started_at = Instant::now();

    let token_entries: Vec<(usize, Vec<u8>)> = token_bytes
        .iter()
        .map(|(&token_id, bytes)| (token_id as usize, bytes.clone()))
        .collect();
    let trie = VocabPrefixTree::build_owned(token_entries);
    let original_token_slots = max_original_token_slot(token_bytes);

    let trie_build_states: Vec<u32> = match config.initial_state_map {
        Some(init_map) => init_map.representative_original_ids.clone(),
        None => (0..tokenizer.num_states()).collect(),
    };
    let (mut trie_class_result, _) = collector::collect_possible_matches_dense_trie_class_build_with_classes(
        tokenizer,
        &trie.root,
        original_token_slots,
        &trie_build_states,
    );
    if let Some(init_map) = config.initial_state_map {
        trie_class_result.state_classes = compose_state_classes_with_initial_map(
            &trie_class_result.state_classes,
            init_map,
        );
    }
    let possible_matches_collect_ms = elapsed_ms(pm_started_at);

    let possible_match_vocab_started_at = Instant::now();
    let tokens_with_same_bytes = build_tokens_with_same_bytes(token_bytes);
    let possible_match_signature_ids = build_possible_match_signature_ids_from_trie_classes(
        &trie_class_result.class_maps,
        &trie_class_result.state_classes,
        token_bytes,
        &tokens_with_same_bytes,
    );
    let seed_state_signature_ids = build_seed_state_signature_ids_from_trie_classes_exact(
        tokenizer,
        token_bytes,
        &tokens_with_same_bytes,
        &trie_class_result.state_classes,
    );
    let possible_match_vocab = build_possible_match_vocab_map(
        token_bytes,
        &possible_match_signature_ids,
        &seed_state_signature_ids,
    );
    let possible_matches = remap_class_maps_to_possible_match_vocab(
        &trie_class_result.class_maps,
        &trie_class_result.state_classes,
        &possible_match_vocab.original_to_internal,
        possible_match_vocab.internal_to_originals.len() as u32,
        &tokens_with_same_bytes,
    );
    let possible_matches_id_map = InternalIdMap {
        tokenizer_states: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            trie_class_result.state_classes.clone(),
            trie_class_result
                .state_classes
                .iter()
                .copied()
                .filter(|&class_id| class_id != u32::MAX)
                .max()
                .map(|class_id| class_id + 1)
                .unwrap_or(0),
        ),
        vocab_tokens: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            possible_match_vocab.original_to_internal.clone(),
            possible_match_vocab.internal_to_originals.len() as u32,
        ),
    };

    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
    {
        eprintln!(
            "[glrmask/profile][possible_match_vocab] original_tokens={} possible_match_tokens={}",
            token_bytes.len(),
            possible_matches_id_map.vocab_tokens.internal_to_originals.len(),
        );
    }

    let possible_match_vocab_ms = elapsed_ms(possible_match_vocab_started_at);

    ConstraintPossibleMatchesComputation {
        possible_matches,
        id_map: possible_matches_id_map,
        profile: ConstraintPossibleMatchesProfile {
            possible_matches_collect_ms,
            possible_match_vocab_ms,
        },
    }
}
