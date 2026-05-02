use std::collections::BTreeMap;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::constraint_possible_matches::collector::DensePossibleMatchMap;
use crate::compiler::pm_profile::elapsed_ms;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::ds::weight::{Weight, finalize_weight_map, shared_rangeset};
use crate::grammar::flat::TerminalID;
use crate::runtime::Constraint;

pub(crate) mod bruteforce;
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
    pub(crate) debug_compile_stages: bool,
    pub(crate) profile_summary_enabled: bool,
    pub(crate) use_internal_tsid_representatives: bool,
    pub(crate) trie_class_build_enabled: bool,
    pub(crate) diag_root_signature: bool,
    pub(crate) initial_state_map: Option<&'a ManyToOneIdMap>,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ConstraintPossibleMatchesProfile {
    pub(crate) possible_matches_collect_ms: f64,
    pub(crate) constraint_vocab_ms: f64,
    pub(crate) same_bytes_ms: f64,
    pub(crate) possible_match_signatures_ms: f64,
    pub(crate) seed_state_signatures_ms: f64,
    pub(crate) build_map_ms: f64,
    pub(crate) remap_possible_matches_ms: f64,
    pub(crate) total_ms: f64,
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

fn build_original_to_constraint_internal_ids(
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
    original_to_constraint_internal: &[u32],
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
            let Some(&internal_id) = original_to_constraint_internal.get(token_id as usize) else {
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

/// Exact trie-based seed-state signature builder.
///
/// Walks the tokenizer DFA over a prefix trie of group-leader byte strings,
/// recording which original tokenizer states can reach each leader.  The
/// result is then expanded from leaders to every token in the same-byte
/// group, sorted/deduplicated, and returned as a map from original token id
/// to its seed-state signature.
/// Brute-force exact seed-state signature builder.
///
/// For each group leader, simulates the tokenizer from every nonterminal
/// state.  Terminal states are intentionally omitted: all tokens share the
/// same terminal-state component, so it does not affect the constraint-vocab
/// grouping.
pub(crate) fn build_seed_state_signatures_trie(
    tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
) -> FxHashMap<u32, SeedStateSignature> {
    let groups = unique_same_byte_groups(tokens_with_same_bytes);
    let num_states = tokenizer.num_states() as usize;

    let terminal_states: Vec<bool> = (0..num_states)
        .map(|state| !tokenizer.dfa.finalizers(state as u32).is_zero())
        .collect();

    // Compute signatures per group leader in parallel.
    let signatures_by_leader: Vec<(u32, SeedStateSignature)> = groups
        .par_iter()
        .map(|group| {
            let leader = group[0];
            let bytes = token_bytes.get(&leader).expect("leader must have bytes");
            let mut signature: SeedStateSignature = Vec::new();

            for state in 0..num_states as u32 {
                if terminal_states[state as usize] {
                    continue;
                }
                if can_scan_token(tokenizer, bytes, state) {
                    signature.push(state);
                }
            }

            signature.sort_unstable();
            signature.dedup();
            (leader, signature)
        })
        .collect();

    let mut signatures_by_leader_map: FxHashMap<u32, SeedStateSignature> =
        signatures_by_leader.into_iter().collect();

    // Expand from leaders to all tokens in each same-byte group.
    let mut signatures = FxHashMap::default();
    for group in groups {
        let signature = signatures_by_leader_map.remove(&group[0]).unwrap_or_default();
        for &token_id in group.iter() {
            signatures.insert(token_id, signature.clone());
        }
    }

    signatures
}

/// Fast seed-state signature builder that reuses trie state classes.
///
/// `state_classes` comes from `collect_possible_matches_dense_trie_class_build_with_classes`.
/// States in the same class are guaranteed to have identical token-matching
/// behavior, so we only simulate one representative per class.
pub(crate) fn build_seed_state_signature_ids_from_trie_classes_exact(
    tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
    state_classes: &[u32],
) -> FxHashMap<u32, SignatureClassId> {
    let t_start = std::time::Instant::now();

    let t_groups = std::time::Instant::now();
    let groups = unique_same_byte_groups(tokens_with_same_bytes);
    let groups_ms = t_groups.elapsed().as_secs_f64() * 1000.0;

    let t_map = std::time::Instant::now();
    let mut class_to_rep: FxHashMap<u32, u32> = FxHashMap::default();
    let mut class_members: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for (state, &class_id) in state_classes.iter().enumerate() {
        if class_id == u32::MAX {
            continue;
        }
        class_to_rep.entry(class_id).or_insert(state as u32);
        class_members.entry(class_id).or_default().push(state as u32);
    }
    let mut rep_entries: Vec<(u32, u32)> = class_to_rep.into_iter().collect();
    rep_entries.sort_unstable_by_key(|(class_id, _)| *class_id);
    let state_class_count = rep_entries.len();
    let map_ms = t_map.elapsed().as_secs_f64() * 1000.0;

    let num_leaders = groups.len();

    // Build a vocabulary prefix trie over the leader tokens for seed-signature
    // simulation. This replaces the per-group can_scan_token loop with a
    // prefix-shared traversal that avoids re-scanning common prefixes for every
    // representative state.
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
    let leader_trie = VocabPrefixTree::build(&leader_entries);

    // Precompute flat DFA transitions and terminal-state booleans for the
    // seed-signature trie walk, replacing hot tokenizer.step/finalizers calls
    // with direct array lookups.
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

    let t_sim = std::time::Instant::now();
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
    let sim_ms = t_sim.elapsed().as_secs_f64() * 1000.0;

    let inv_ms = 0.0;

    let t_flat = std::time::Instant::now();
    let signatures_flat: Vec<u64> = signatures_atomic
        .into_iter()
        .map(AtomicU64::into_inner)
        .collect();
    let flat_ms = t_flat.elapsed().as_secs_f64() * 1000.0;

    let t_expand = std::time::Instant::now();
    // Hash-sort-group dense signatures to avoid expensive Vec cloning and
    // HashMap operations with class_words-word keys.
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
    leader_hashes.sort_unstable_by_key(|(h, _)| *h);

    let mut leader_to_interned: Vec<SignatureClassId> = vec![0; num_leaders];
    let mut next_class: SignatureClassId = 0;
    let mut i = 0;
    while i < leader_hashes.len() {
        let hash = leader_hashes[i].0;
        // Find the end of this hash bucket.
        let mut j = i + 1;
        while j < leader_hashes.len() && leader_hashes[j].0 == hash {
            j += 1;
        }
        // Within the bucket, intern unique slices using a small HashMap.
        let mut slice_to_class: FxHashMap<&[u64], SignatureClassId> = FxHashMap::default();
        for k in i..j {
            let leader_idx = leader_hashes[k].1;
            let base = leader_idx * class_words;
            let slice = &signatures_flat[base..base + class_words];
            let class_id = *slice_to_class.entry(slice).or_insert_with(|| {
                let id = next_class;
                next_class += 1;
                id
            });
            leader_to_interned[leader_idx] = class_id;
        }
        i = j;
    }

    let signature_class_count = next_class as usize;

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
    let expand_ms = t_expand.elapsed().as_secs_f64() * 1000.0;

    let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][seed_sig_classes] groups={} state_classes={} reps={} words={} flat_len={} signature_classes={} groups_ms={:.3} map_ms={:.3} sim_ms={:.3} inv_ms={:.3} flat_ms={:.3} expand_ms={:.3} total_ms={:.3}",
            groups.len(),
            class_members.len(),
            rep_entries.len(),
            class_words,
            num_leaders * class_words,
            signature_class_count,
            groups_ms,
            map_ms,
            sim_ms,
            inv_ms,
            flat_ms,
            expand_ms,
            total_ms,
        );
    }

    token_to_id
}

#[inline]
fn push_reachable_leader_indices<F>(
    node: &VocabPrefixTreeNode,
    leader_idx_by_token: &[u32],
    on_match: &mut F,
)
where
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
)
where
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
        let Some((next_state, accepted)) = scan_seed_signature_edge(flat_transitions, terminal_states, state, edge) else {
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

#[inline]
fn can_scan_token(
    tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
    bytes: &[u8],
    start_state: u32,
) -> bool {
    let mut state = start_state;
    for &byte in bytes {
        let Some(next) = tokenizer.step(state, byte) else {
            return false;
        };
        state = next;
        if !tokenizer.dfa.finalizers(state).is_zero() {
            return true;
        }
    }
    true
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
    let used_class_ids = used_state_class_ids(state_classes);
    let mut signatures_by_group: FxHashMap<u32, SeedStateSignature> =
        groups.iter().map(|group| (group[0], Vec::new())).collect();

    let signature_entries: Vec<(u32, u32)> = used_class_ids
        .par_iter()
        .flat_map_iter(|&class_id| {
            let mut covered_leaders = Vec::new();
            let Some(terminals) = class_maps.get(class_id as usize) else {
                return Vec::new();
            };
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
            covered_leaders
                .into_iter()
                .map(|leader| (leader, class_id))
                .collect::<Vec<_>>()
        })
        .collect();

    for (leader, class_id) in signature_entries {
        if let Some(signature) = signatures_by_group.get_mut(&leader) {
            signature.push(class_id);
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
) -> RuntimePossibleMatchesByTerminal {
    remap_dense_maps_to_constraint_vocab(
        raw_possible_matches.into_iter(),
        original_to_constraint_internal,
        constraint_token_count,
        tokens_with_same_bytes,
    )
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

pub(crate) fn remap_class_maps_to_existing_vocab_fast(
    class_maps: &[Arc<DensePossibleMatchMap>],
    state_classes: &[u32],
    original_to_constraint_internal: &[u32],
    constraint_token_count: u32,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
) -> RuntimePossibleMatchesByTerminal {
    let num_words = dense_word_count(constraint_token_count);
    let original_to_internal_ids = build_original_to_constraint_internal_ids(
        tokens_with_same_bytes,
        original_to_constraint_internal,
    );

    let mut bitmap_cache: FxHashMap<Box<[u64]>, RangeSetBlaze<u32>> = FxHashMap::default();
    let mut class_token_set_cache: FxHashMap<u32, BTreeMap<TerminalID, Arc<RangeSetBlaze<u32>>>> = FxHashMap::default();

    for (class_id, class_map) in class_maps.iter().enumerate() {
        let mut terminal_sets = BTreeMap::new();
        for (&terminal_id, original_bitmap) in class_map.iter() {
            let token_set = bitmap_cache
                .entry(original_bitmap.clone())
                .or_insert_with(|| {
                    remap_dense_bitmap_with_original_to_internal_ids(
                        original_bitmap,
                        &original_to_internal_ids,
                        num_words,
                    )
                })
                .clone();
            if !token_set.is_empty() {
                terminal_sets.insert(terminal_id, shared_rangeset(token_set));
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

pub(crate) fn remap_class_maps_to_constraint_vocab(
    class_maps: &[Arc<DensePossibleMatchMap>],
    state_classes: &[u32],
    original_to_constraint_internal: &[u32],
    constraint_token_count: u32,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
) -> RuntimePossibleMatchesByTerminal {
    let num_words = dense_word_count(constraint_token_count);
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
    let group_constraint_internal_ids = build_group_constraint_internal_ids(&groups, original_to_constraint_internal);

    // Accumulate per-terminal (state_id, token_set) entries.
    let mut terminal_entries: BTreeMap<TerminalID, Vec<(u32, RangeSetBlaze<u32>)>> = BTreeMap::new();

    for (state, &class_id) in state_classes.iter().enumerate() {
        if class_id == u32::MAX {
            continue;
        }
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
                            let Some(constraint_internal_ids) = group_constraint_internal_ids.get(&leader) else {
                                return;
                            };
                            for &constraint_internal_id in constraint_internal_ids.iter() {
                                set_dense_bit(&mut remapped_bits, constraint_internal_id);
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
            terminal_entries.entry(terminal_id).or_default().push((state as u32, token_set));
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
    eprintln!(
        "[glrmask/diag][pm_unique] terminals={} internal_tsids={} unique_terminal_weights={}",
        constraint.possible_matches.len(),
        constraint.internal_tsid_to_states.len(),
        unique_all.len(),
    );
}

pub(crate) fn compute_constraint_possible_matches(
    tokenizer: &Tokenizer,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    internal_ids: &InternalIdMap,
    config: ConstraintPossibleMatchesConfig,
) -> ConstraintPossibleMatchesComputation {
    let total_started_at = Instant::now();

    let pm_started_at = Instant::now();
    if config.debug_compile_stages {
        eprintln!("[glrmask/debug][compile_stage] raw_possible_matches_begin");
    }

    let token_entries: Vec<(usize, Vec<u8>)> = token_bytes
        .iter()
        .map(|(&token_id, bytes)| (token_id as usize, bytes.clone()))
        .collect();
    let trie_build_started_at = Instant::now();
    let trie = VocabPrefixTree::build_owned(token_entries);
    let trie_build_ms = elapsed_ms(trie_build_started_at);

    let collect_started_at = Instant::now();
    let original_token_slots = max_original_token_slot(token_bytes);

    let (raw_possible_matches, trie_class_result, dense_profile, profile_label, profile_state_count) =
        if config.use_internal_tsid_representatives {
            eprintln!(
                "[glrmask/diag][pm_reps_unsafe] using internal-tsid representatives as a diagnostic shortcut; this is not production-valid by construction"
            );
            let representative_states = &internal_ids.tokenizer_states.representative_original_ids;
            if config.diag_root_signature {
                let root_signature_count = collector::count_root_child_internal_tsid_signatures(
                    tokenizer,
                    &trie.root,
                    representative_states,
                    &internal_ids.tokenizer_states.original_to_internal,
                );
                eprintln!(
                    "[glrmask/diag][pm_root_sig] rep_states={} unique_root_signatures={}",
                    representative_states.len(),
                    root_signature_count,
                );
            }
            let (representative_matches, profile) = collector::collect_possible_matches_by_selected_original_tsid_dense(
                tokenizer,
                &trie.root,
                original_token_slots,
                representative_states,
            );
            (
                representative_matches,
                None,
                profile,
                "constraint_original_tokens_internal_tsid_reps",
                representative_states.len() as u32,
            )
        } else if config.trie_class_build_enabled {
            let trie_build_states: Vec<u32> = match config.initial_state_map {
                Some(init_map) => init_map.representative_original_ids.clone(),
                None => (0..tokenizer.num_states()).collect(),
            };
            let (mut trie_class_result, profile) = collector::collect_possible_matches_dense_trie_class_build_with_classes(
                tokenizer,
                &trie.root,
                original_token_slots,
                &trie_build_states,
            );
            // Compose dense root state_classes through initial_state_map.
            // The collector returns classes indexed by original DFA state id
            // for the states it was seeded with. Each initial-map class inherits
            // the class of its representative.
            if let Some(init_map) = config.initial_state_map {
                trie_class_result.state_classes = compose_state_classes_with_initial_map(
                    &trie_class_result.state_classes,
                    init_map,
                );
            }
            (
                BTreeMap::new(),
                Some(trie_class_result),
                profile,
                "constraint_original_tokens",
                trie_build_states.len() as u32,
            )
        } else {
            let (matches, profile) = collector::collect_possible_matches_by_original_tsid_dense(
                tokenizer,
                &trie.root,
                original_token_slots,
            );
            (
                matches,
                None,
                profile,
                "constraint_original_tokens",
                tokenizer.num_states(),
            )
        };

    let collect_ms = elapsed_ms(collect_started_at);
    collector::emit_possible_matches_profile_summary(
        profile_label,
        token_bytes.len(),
        profile_state_count,
        trie_build_ms,
        collect_ms,
        &dense_profile,
    );

    let possible_matches_collect_ms = elapsed_ms(pm_started_at);
    if config.debug_compile_stages {
        eprintln!(
            "[glrmask/debug][compile_stage] raw_possible_matches_done trie_build_ms={:.3} collect_ms={:.3} profile_label={} profile_states={}",
            trie_build_ms,
            collect_ms,
            profile_label,
            profile_state_count,
        );
        eprintln!(
            "[glrmask/debug][compile_stage] constraint_vocab_begin possible_matches_ms={:.3}",
            possible_matches_collect_ms,
        );
    }

    let constraint_vocab_started_at = Instant::now();

    let tokens_with_same_bytes_started_at = Instant::now();
    if config.debug_compile_stages {
        eprintln!("[glrmask/debug][compile_stage] constraint_vocab_same_bytes_begin");
    }
    let tokens_with_same_bytes = build_tokens_with_same_bytes(token_bytes);
    let same_bytes_ms = elapsed_ms(tokens_with_same_bytes_started_at);
    if config.debug_compile_stages {
        eprintln!(
            "[glrmask/debug][compile_stage] constraint_vocab_same_bytes_done ms={:.3}",
            same_bytes_ms,
        );
    }
    if config.profile_summary_enabled {
        eprintln!(
            "[glrmask/profile][constraint_vocab_step] step=same_bytes ms={:.3}",
            same_bytes_ms,
        );
    }

    // Fast path: possible-matches are produced in terminal-DWA vocab space,
    // so there is no need to split vocab tokens by possible-match or seed-state
    // signatures. Use an identity constraint-vocab that preserves the existing
    // terminal-DWA token groupings.
    let possible_match_signatures_ms = 0.0;
    let seed_state_signatures_ms = 0.0;
    let build_map_ms = 0.0;
    let constraint_vocab = constraint_vocab_from_terminal_dwa_vocab(&internal_ids.vocab_tokens);
    if config.debug_compile_stages {
        eprintln!(
            "[glrmask/debug][compile_stage] constraint_vocab_identity vocab_tokens={}",
            constraint_vocab.internal_to_originals.len(),
        );
    }
    if config.profile_summary_enabled {
        eprintln!(
            "[glrmask/profile][constraint_vocab_step] step=possible_match_signatures ms={:.3}",
            possible_match_signatures_ms,
        );
        eprintln!(
            "[glrmask/profile][constraint_vocab_step] step=seed_state_signatures ms={:.3}",
            seed_state_signatures_ms,
        );
        eprintln!(
            "[glrmask/profile][constraint_vocab_step] step=build_map ms={:.3}",
            build_map_ms,
        );
    }

    let constraint_token_count = constraint_vocab.internal_to_originals.len() as u32;
    let remap_possible_matches_started_at = Instant::now();
    if config.debug_compile_stages {
        eprintln!(
            "[glrmask/debug][compile_stage] constraint_vocab_remap_possible_matches_begin constraint_token_count={}",
            constraint_token_count,
        );
    }
    let possible_matches = if let Some(trie_class_result) = trie_class_result.as_ref() {
        remap_class_maps_to_existing_vocab_fast(
            &trie_class_result.class_maps,
            &trie_class_result.state_classes,
            &constraint_vocab.original_to_internal,
            constraint_token_count,
            &tokens_with_same_bytes,
        )
    } else {
        let raw_possible_matches = if config.use_internal_tsid_representatives {
            expand_possible_matches_to_original_states(
                &raw_possible_matches,
                &internal_ids.tokenizer_states.internal_to_originals,
                &internal_ids.tokenizer_states.representative_original_ids,
            )
        } else {
            raw_possible_matches
        };
        remap_possible_matches_to_constraint_vocab(
            raw_possible_matches,
            &constraint_vocab.original_to_internal,
            constraint_token_count,
            &tokens_with_same_bytes,
        )
    };
    let remap_possible_matches_ms = elapsed_ms(remap_possible_matches_started_at);
    if config.debug_compile_stages {
        eprintln!(
            "[glrmask/debug][compile_stage] constraint_vocab_remap_possible_matches_done ms={:.3}",
            remap_possible_matches_ms,
        );
    }
    if config.profile_summary_enabled {
        eprintln!(
            "[glrmask/profile][constraint_vocab_step] step=remap_possible_matches ms={:.3}",
            remap_possible_matches_ms,
        );
    }

    let constraint_vocab_ms = elapsed_ms(constraint_vocab_started_at);
    if config.debug_compile_stages {
        eprintln!(
            "[glrmask/debug][compile_stage] constraint_vocab_done ms={:.3}",
            constraint_vocab_ms,
        );
    }

    ConstraintPossibleMatchesComputation {
        possible_matches,
        constraint_vocab,
        profile: ConstraintPossibleMatchesProfile {
            possible_matches_collect_ms,
            constraint_vocab_ms,
            same_bytes_ms,
            possible_match_signatures_ms,
            seed_state_signatures_ms,
            build_map_ms,
            remap_possible_matches_ms,
            total_ms: elapsed_ms(total_started_at),
        },
    }
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