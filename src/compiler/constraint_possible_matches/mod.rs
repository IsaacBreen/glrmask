use std::collections::BTreeMap;
use std::hash::Hash;
use std::sync::Arc;

use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::Vocab;
use crate::compiler::constraint_possible_matches::collector::DensePossibleMatchMap;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::ds::weight::{Weight, finalize_weight_map, shared_rangeset};
use crate::grammar::flat::TerminalID;
use crate::runtime::Constraint;

pub(crate) mod collector;

pub(crate) type DensePossibleMatchesByState = BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>;
pub(crate) type RuntimePossibleMatchesByState = BTreeMap<u32, Weight>;
pub(crate) type PossibleMatchSignature = Vec<(u32, TerminalID)>;
pub(crate) type SeedStateSignature = Vec<u32>;
pub(crate) type SignatureClassId = u32;

#[derive(Debug)]
pub(crate) struct ConstraintVocabMap {
    pub(crate) original_to_internal: Vec<u32>,
    pub(crate) internal_to_originals: Vec<Vec<u32>>,
    pub(crate) old_internal_to_constraint: Vec<Vec<u32>>,
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

fn build_state_class_members(state_classes: &[u32], num_classes: usize) -> Vec<Vec<u32>> {
    let mut members = vec![Vec::new(); num_classes];
    for (state, &class_id) in state_classes.iter().enumerate() {
        if class_id == u32::MAX {
            continue;
        }
        if let Some(class_members) = members.get_mut(class_id as usize) {
            class_members.push(state as u32);
        }
    }
    members
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

pub(crate) fn build_possible_match_signatures(
    raw_possible_matches: &DensePossibleMatchesByState,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
) -> FxHashMap<u32, PossibleMatchSignature> {
    let groups = unique_same_byte_groups(tokens_with_same_bytes);
    let group_leaders = build_same_byte_group_leaders(token_bytes, &groups);
    let mut signatures_by_group: FxHashMap<u32, PossibleMatchSignature> =
        groups.iter().map(|group| (group[0], Vec::new())).collect();

    for (&original_tokenizer_state, terminals) in raw_possible_matches {
        for (&terminal_id, bitmap) in terminals {
            for_each_dense_bit(bitmap, |original_token_id| {
                let Some(&leader) = group_leaders.get(original_token_id as usize) else {
                    return;
                };
                if leader == u32::MAX {
                    return;
                }
                if let Some(signature) = signatures_by_group.get_mut(&leader) {
                    signature.push((original_tokenizer_state, terminal_id));
                }
            });
        }
    }

    for signature in signatures_by_group.values_mut() {
        signature.sort_unstable();
        signature.dedup();
    }

    let mut signatures = FxHashMap::default();
    for group in groups {
        let signature = signatures_by_group.remove(&group[0]).unwrap_or_default();
        for &token_id in group.iter() {
            signatures.insert(token_id, signature.clone());
        }
    }
    signatures
}

pub(crate) fn build_possible_match_signature_ids_from_trie_classes(
    class_maps: &[Arc<DensePossibleMatchMap>],
    state_classes: &[u32],
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
) -> FxHashMap<u32, SignatureClassId> {
    let groups = unique_same_byte_groups(tokens_with_same_bytes);
    let group_leaders = build_same_byte_group_leaders(token_bytes, &groups);
    let class_members = build_state_class_members(state_classes, class_maps.len());
    let mut signatures_by_group: FxHashMap<u32, PossibleMatchSignature> =
        groups.iter().map(|group| (group[0], Vec::new())).collect();

    for (class_id, terminals) in class_maps.iter().enumerate() {
        let members = &class_members[class_id];
        if members.is_empty() {
            continue;
        }
        for (&terminal_id, bitmap) in terminals.iter() {
            for_each_dense_bit(bitmap, |original_token_id| {
                let Some(&leader) = group_leaders.get(original_token_id as usize) else {
                    return;
                };
                if leader == u32::MAX {
                    return;
                }
                if let Some(signature) = signatures_by_group.get_mut(&leader) {
                    signature.push((class_id as u32, terminal_id));
                }
            });
        }
    }

    for signature in signatures_by_group.values_mut() {
        signature.sort_unstable();
        signature.dedup();
    }

    intern_signatures_by_group(groups, signatures_by_group)
}

pub(crate) fn build_possible_match_signatures_by_internal_tsid(
    raw_possible_matches: &DensePossibleMatchesByState,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
    state_to_internal_tsid: &[u32],
) -> FxHashMap<u32, PossibleMatchSignature> {
    let groups = unique_same_byte_groups(tokens_with_same_bytes);
    let group_leaders = build_same_byte_group_leaders(token_bytes, &groups);
    let mut signatures_by_group: FxHashMap<u32, PossibleMatchSignature> =
        groups.iter().map(|group| (group[0], Vec::new())).collect();

    for (&original_tokenizer_state, terminals) in raw_possible_matches {
        let internal_tsid = state_to_internal_tsid
            .get(original_tokenizer_state as usize)
            .copied()
            .unwrap_or(original_tokenizer_state);
        for (&terminal_id, bitmap) in terminals {
            for_each_dense_bit(bitmap, |original_token_id| {
                let Some(&leader) = group_leaders.get(original_token_id as usize) else {
                    return;
                };
                if leader == u32::MAX {
                    return;
                }
                if let Some(signature) = signatures_by_group.get_mut(&leader) {
                    signature.push((internal_tsid, terminal_id));
                }
            });
        }
    }

    for signature in signatures_by_group.values_mut() {
        signature.sort_unstable();
        signature.dedup();
    }

    let mut signatures = FxHashMap::default();
    for group in groups {
        let signature = signatures_by_group.remove(&group[0]).unwrap_or_default();
        for &token_id in group.iter() {
            signatures.insert(token_id, signature.clone());
        }
    }
    signatures
}

pub(crate) fn build_seed_state_signatures_from_possible_matches(
    raw_possible_matches: &DensePossibleMatchesByState,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
) -> FxHashMap<u32, SeedStateSignature> {
    let groups = unique_same_byte_groups(tokens_with_same_bytes);
    let group_leaders = build_same_byte_group_leaders(token_bytes, &groups);
    let mut signatures_by_group: FxHashMap<u32, SeedStateSignature> =
        groups.iter().map(|group| (group[0], Vec::new())).collect();

    for (&original_tokenizer_state, terminals) in raw_possible_matches {
        for bitmap in terminals.values() {
            for_each_dense_bit(bitmap, |original_token_id| {
                let Some(&leader) = group_leaders.get(original_token_id as usize) else {
                    return;
                };
                if leader == u32::MAX {
                    return;
                }
                if let Some(signature) = signatures_by_group.get_mut(&leader) {
                    signature.push(original_tokenizer_state);
                }
            });
        }
    }

    for signature in signatures_by_group.values_mut() {
        signature.sort_unstable();
        signature.dedup();
    }

    let mut signatures = FxHashMap::default();
    for group in groups {
        let signature = signatures_by_group.remove(&group[0]).unwrap_or_default();
        for &token_id in group.iter() {
            signatures.insert(token_id, signature.clone());
        }
    }
    signatures
}

pub(crate) fn build_seed_state_signature_ids_from_trie_classes(
    class_maps: &[Arc<DensePossibleMatchMap>],
    state_classes: &[u32],
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
) -> FxHashMap<u32, SignatureClassId> {
    let groups = unique_same_byte_groups(tokens_with_same_bytes);
    let group_leaders = build_same_byte_group_leaders(token_bytes, &groups);
    let class_members = build_state_class_members(state_classes, class_maps.len());
    let mut signatures_by_group: FxHashMap<u32, SeedStateSignature> =
        groups.iter().map(|group| (group[0], Vec::new())).collect();

    for (class_id, terminals) in class_maps.iter().enumerate() {
        let members = &class_members[class_id];
        if members.is_empty() {
            continue;
        }

        let mut covered_leaders = Vec::new();
        for bitmap in terminals.values() {
            for_each_dense_bit(bitmap, |original_token_id| {
                let Some(&leader) = group_leaders.get(original_token_id as usize) else {
                    return;
                };
                if leader != u32::MAX {
                    covered_leaders.push(leader);
                }
            });
        }

        covered_leaders.sort_unstable();
        covered_leaders.dedup();

        for leader in covered_leaders {
            if let Some(signature) = signatures_by_group.get_mut(&leader) {
                signature.push(class_id as u32);
            }
        }
    }

    for signature in signatures_by_group.values_mut() {
        signature.sort_unstable();
        signature.dedup();
    }

    intern_signatures_by_group(groups, signatures_by_group)
}

pub(crate) fn build_seed_state_signatures_from_possible_matches_by_internal_tsid(
    raw_possible_matches: &DensePossibleMatchesByState,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
    state_to_internal_tsid: &[u32],
) -> FxHashMap<u32, SeedStateSignature> {
    let groups = unique_same_byte_groups(tokens_with_same_bytes);
    let group_leaders = build_same_byte_group_leaders(token_bytes, &groups);
    let mut signatures_by_group: FxHashMap<u32, SeedStateSignature> =
        groups.iter().map(|group| (group[0], Vec::new())).collect();

    for (&original_tokenizer_state, terminals) in raw_possible_matches {
        let internal_tsid = state_to_internal_tsid
            .get(original_tokenizer_state as usize)
            .copied()
            .unwrap_or(original_tokenizer_state);
        for bitmap in terminals.values() {
            for_each_dense_bit(bitmap, |original_token_id| {
                let Some(&leader) = group_leaders.get(original_token_id as usize) else {
                    return;
                };
                if leader == u32::MAX {
                    return;
                }
                if let Some(signature) = signatures_by_group.get_mut(&leader) {
                    signature.push(internal_tsid);
                }
            });
        }
    }

    for signature in signatures_by_group.values_mut() {
        signature.sort_unstable();
        signature.dedup();
    }

    let mut signatures = FxHashMap::default();
    for group in groups {
        let signature = signatures_by_group.remove(&group[0]).unwrap_or_default();
        for &token_id in group.iter() {
            signatures.insert(token_id, signature.clone());
        }
    }
    signatures
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

pub(crate) fn build_constraint_vocab_map(
    parser_vocab: &ManyToOneIdMap,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    possible_match_signature_ids: &FxHashMap<u32, SignatureClassId>,
    seed_state_signature_ids: &FxHashMap<u32, SignatureClassId>,
) -> ConstraintVocabMap {
    let max_original_slot = token_bytes
        .keys()
        .next_back()
        .map(|token_id| *token_id as usize + 1)
        .unwrap_or(0);

    let mut original_to_internal = vec![
        u32::MAX;
        parser_vocab.original_to_internal.len().max(max_original_slot)
    ];
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut old_internal_to_constraint = vec![Vec::<u32>::new(); parser_vocab.internal_to_originals.len()];

    for (old_internal_id, originals) in parser_vocab.internal_to_originals.iter().enumerate() {
        let mut groups: BTreeMap<(SignatureClassId, SignatureClassId), Vec<u32>> = BTreeMap::new();
        for &original_token_id in originals {
            if !token_bytes.contains_key(&original_token_id) {
                continue;
            }
            let forward = parser_vocab
                .original_to_internal
                .get(original_token_id as usize)
                .copied()
                .unwrap_or(u32::MAX);
            debug_assert_eq!(forward, old_internal_id as u32);
            let signature = possible_match_signature_ids.get(&original_token_id).cloned().unwrap_or_default();
            let seed_signature = seed_state_signature_ids.get(&original_token_id).cloned().unwrap_or_default();
            groups.entry((signature, seed_signature)).or_default().push(original_token_id);
        }

        for (_, mut originals) in groups {
            originals.sort_unstable();
            originals.dedup();
            let new_internal_id = internal_to_originals.len() as u32;
            for &original_token_id in &originals {
                if original_token_id as usize >= original_to_internal.len() {
                    original_to_internal.resize(original_token_id as usize + 1, u32::MAX);
                }
                original_to_internal[original_token_id as usize] = new_internal_id;
            }
            old_internal_to_constraint[old_internal_id].push(new_internal_id);
            internal_to_originals.push(originals);
        }
    }

    ConstraintVocabMap {
        original_to_internal,
        internal_to_originals,
        old_internal_to_constraint,
    }
}

pub(crate) fn remap_possible_matches_to_constraint_vocab(
    raw_possible_matches: DensePossibleMatchesByState,
    original_to_constraint_internal: &[u32],
    constraint_token_count: u32,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
) -> RuntimePossibleMatchesByState {
    let num_words = dense_word_count(constraint_token_count);
    let mut remap_cache: FxHashMap<Vec<(TerminalID, Box<[u64]>)>, Weight> = FxHashMap::default();
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

    raw_possible_matches
        .into_iter()
        .map(|(original_tokenizer_state, terminals)| {
            let cache_key: Vec<(TerminalID, Box<[u64]>)> = terminals
                .iter()
                .map(|(&terminal_id, bitmap)| (terminal_id, bitmap.clone()))
                .collect();

            let weight = remap_cache
                .entry(cache_key)
                .or_insert_with(|| {
                    let remapped_terminals = terminals
                        .iter()
                        .filter_map(|(&terminal_id, original_bitmap)| {
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
                                Some((terminal_id, {
                                    let mut ids = Vec::new();
                                    for_each_dense_bit(&remapped, |token_id| ids.push(token_id));
                                    range_set_from_sorted_ids(&ids)
                                }))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();
                    Weight::from_per_tsid_token_sets(remapped_terminals)
                })
                .clone();

            (original_tokenizer_state, weight)
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

pub(crate) fn assert_possible_matches_equivalent_within_internal_tsids(constraint: &Constraint) {
    let mut merged_class_count = 0usize;
    let mut merged_state_count = 0usize;
    let mut max_class_size = 0usize;
    for states in &constraint.internal_tsid_to_states {
        if states.len() <= 1 {
            continue;
        }
        merged_class_count += 1;
        merged_state_count += states.len();
        max_class_size = max_class_size.max(states.len());
        let representative = constraint.possible_matches_for_state(states[0]);
        for &state in &states[1..] {
            let actual = constraint.possible_matches_for_state(state);
            assert_eq!(actual, representative);
        }
    }
    eprintln!(
        "[glrmask/diag][pm_equiv] tokenizer_states={} internal_tsids={} merged_classes={} merged_states={} max_class_size={}",
        constraint.state_to_internal_tsid.len(),
        constraint.internal_tsid_to_states.len(),
        merged_class_count,
        merged_state_count,
        max_class_size,
    );
}

pub(crate) fn emit_possible_matches_unique_counts(constraint: &Constraint) {
    let unique_all: FxHashSet<_> = constraint.possible_matches.values().cloned().collect();
    let unique_reps: FxHashSet<_> = constraint
        .internal_tsid_to_states
        .iter()
        .filter_map(|states| states.first())
        .filter_map(|state| constraint.possible_matches.get(state).cloned())
        .collect();
    eprintln!(
        "[glrmask/diag][pm_unique] tokenizer_states={} internal_tsids={} unique_all_states={} unique_rep_states={}",
        constraint.possible_matches.len(),
        constraint.internal_tsid_to_states.len(),
        unique_all.len(),
        unique_reps.len(),
    );
}

pub(crate) fn expand_possible_matches_to_original_states(
    representative_matches: &BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>,
    state_classes: &[Vec<u32>],
    representative_states: &[u32],
) -> BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>> {
    let mut expanded = BTreeMap::new();
    for (internal_tsid, states) in state_classes.iter().enumerate() {
        let representative_state = representative_states.get(internal_tsid).copied().unwrap_or(u32::MAX);
        let matches = representative_matches.get(&representative_state).cloned().unwrap_or_default();
        for &state in states {
            expanded.insert(state, matches.clone());
        }
    }
    expanded
}