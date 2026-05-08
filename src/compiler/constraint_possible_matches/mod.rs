
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;
use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::constraint_possible_matches::collector::IntervalPossibleMatchMap;
use crate::compiler::pm_profile::elapsed_ms;
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
#[derive(Debug, Clone, Copy)]
struct SweepEvent {
    add: bool,
    label: StateTerminalLabel,
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
fn build_ordered_vocab(token_bytes: &BTreeMap<u32, Vec<u8>>) -> OrderedVocab {
    let original_slot_count = token_bytes
        .keys()
        .next_back()
        .map(|token_id| *token_id as usize + 1)
        .unwrap_or(0);
    let mut entries: Vec<(Vec<u8>, u32)> = token_bytes
        .iter()
        .map(|(&token_id, bytes)| (bytes.clone(), token_id))
        .collect();
    entries.sort_unstable_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    let mut ordered_to_originals = Vec::new();
    let mut ordered_token_bytes = Vec::new();
    let mut index = 0usize;
    while index < entries.len() {
        let bytes = entries[index].0.clone();
        let mut originals = Vec::new();
        while index < entries.len() && entries[index].0 == bytes {
            let original = entries[index].1;
            originals.push(original);
            index += 1;
        }
        originals.sort_unstable();
        originals.dedup();
        ordered_token_bytes.push(bytes);
        ordered_to_originals.push(originals);
    }
    OrderedVocab {
        original_slot_count,
        ordered_to_originals,
        ordered_token_bytes,
    }
}
fn build_ordered_vocab_prefix_tree(ordered_vocab: &OrderedVocab) -> VocabPrefixTree {
    let entries: Vec<(usize, &[u8])> = ordered_vocab
        .ordered_token_bytes
        .iter()
        .enumerate()
        .map(|(ordered_id, bytes)| (ordered_id, bytes.as_slice()))
        .collect();
    VocabPrefixTree::build_presorted(&entries)
}
#[allow(dead_code)]
pub(crate) fn dense_word_count(token_slots: u32) -> usize {
    (token_slots as usize + 63) / 64
}
#[allow(dead_code)]
pub(crate) fn max_original_token_slot(token_bytes: &BTreeMap<u32, Vec<u8>>) -> u32 {
    token_bytes
        .keys()
        .next_back()
        .map(|token_id| token_id.saturating_add(1))
        .unwrap_or(0)
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
fn push_sweep_event(
    events: &mut [Vec<SweepEvent>],
    event_positions: &mut Vec<u32>,
    position: u32,
    event: SweepEvent,
) {
    let Some(bucket) = events.get_mut(position as usize) else {
        return;
    };
    if bucket.is_empty() {
        event_positions.push(position);
    }
    bucket.push(event);
}
fn build_sweep_events(
    class_maps: &[Arc<IntervalPossibleMatchMap>],
    state_classes: &[u32],
    num_ordered_tokens: usize,
) -> (Vec<Vec<SweepEvent>>, Vec<u32>) {
    let mut events = vec![Vec::new(); num_ordered_tokens + 1];
    let mut event_positions = Vec::new();
    for class_id in used_state_class_ids(state_classes) {
        let Some(class_map) = class_maps.get(class_id as usize) else {
            continue;
        };
        for (&terminal_id, ranges) in class_map.iter() {
            let label = (class_id, terminal_id);
            for &(lo, mut hi) in ranges {
                if num_ordered_tokens == 0 {
                    continue;
                }
                let max_token = num_ordered_tokens as u32 - 1;
                if lo > max_token {
                    continue;
                }
                hi = hi.min(max_token);
                if lo > hi {
                    continue;
                }
                push_sweep_event(
                    &mut events,
                    &mut event_positions,
                    lo,
                    SweepEvent { add: true, label },
                );
                let after = hi.saturating_add(1);
                if after <= num_ordered_tokens as u32 {
                    push_sweep_event(
                        &mut events,
                        &mut event_positions,
                        after,
                        SweepEvent { add: false, label },
                    );
                }
            }
        }
    }
    event_positions.sort_unstable();
    event_positions.dedup();
    (events, event_positions)
}
fn apply_sweep_events(active: &mut BTreeSet<StateTerminalLabel>, events: &[SweepEvent]) {
    for event in events.iter().filter(|event| !event.add) {
        active.remove(&event.label);
    }
    for event in events.iter().filter(|event| event.add) {
        active.insert(event.label);
    }
}
fn build_possible_match_vocab_and_weights_from_interval_maps(
    class_maps: &[Arc<IntervalPossibleMatchMap>],
    state_classes: &[u32],
    ordered_vocab: &OrderedVocab,
) -> (PossibleMatchVocabMap, RuntimePossibleMatchesByTerminal) {
    let num_ordered_tokens = ordered_vocab.ordered_to_originals.len();
    let (events, event_positions) = build_sweep_events(class_maps, state_classes, num_ordered_tokens);
    let mut signature_to_id: FxHashMap<Vec<StateTerminalLabel>, SignatureClassId> = FxHashMap::default();
    let mut signature_labels: Vec<Vec<StateTerminalLabel>> = Vec::new();
    let mut original_to_internal = vec![u32::MAX; ordered_vocab.original_slot_count];
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut active = BTreeSet::<StateTerminalLabel>::new();
    let mut event_index = 0usize;
    let mut position = 0usize;
    while position < num_ordered_tokens {
        while event_index < event_positions.len() && event_positions[event_index] as usize == position {
            apply_sweep_events(&mut active, &events[position]);
            event_index += 1;
        }
        let next_position = event_positions
            .get(event_index)
            .map(|&next| (next as usize).min(num_ordered_tokens))
            .unwrap_or(num_ordered_tokens);
        let signature: Vec<StateTerminalLabel> = active.iter().copied().collect();
        let signature_id = if let Some(&existing) = signature_to_id.get(&signature) {
            existing
        } else {
            let new_id = signature_labels.len() as SignatureClassId;
            signature_to_id.insert(signature.clone(), new_id);
            signature_labels.push(signature);
            internal_to_originals.push(Vec::new());
            new_id
        };
        for ordered_id in position..next_position {
            for &original in &ordered_vocab.ordered_to_originals[ordered_id] {
                if let Some(slot) = original_to_internal.get_mut(original as usize) {
                    *slot = signature_id;
                }
                internal_to_originals[signature_id as usize].push(original);
            }
        }
        position = next_position;
    }
    for originals in &mut internal_to_originals {
        originals.sort_unstable();
        originals.dedup();
    }
    let mut ids_by_label: BTreeMap<TerminalID, BTreeMap<u32, Vec<u32>>> = BTreeMap::new();
    for (signature_id, labels) in signature_labels.iter().enumerate() {
        let signature_id = signature_id as u32;
        for &(class_id, terminal_id) in labels {
            ids_by_label
                .entry(terminal_id)
                .or_default()
                .entry(class_id)
                .or_default()
                .push(signature_id);
        }
    }
    let possible_matches = ids_by_label
        .into_iter()
        .map(|(terminal_id, by_state)| {
            let entries = by_state.into_iter().filter_map(|(state, mut ids)| {
                ids.sort_unstable();
                ids.dedup();
                let token_set = range_set_from_sorted_ids(&ids);
                if token_set.is_empty() {
                    None
                } else {
                    Some((state, shared_rangeset(token_set)))
                }
            });
            (terminal_id, Weight::from_per_tsid_shared(entries))
        })
        .filter(|(_, weight)| !weight.is_empty())
        .collect();
    (
        PossibleMatchVocabMap {
            original_to_internal,
            internal_to_originals,
        },
        possible_matches,
    )
}
pub(crate) fn compute_constraint_possible_matches(
    tokenizer: &Tokenizer,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    config: ConstraintPossibleMatchesConfig,
) -> ConstraintPossibleMatchesComputation {
    let pm_started_at = Instant::now();
    let ordered_vocab = build_ordered_vocab(token_bytes);
    let trie = build_ordered_vocab_prefix_tree(&ordered_vocab);
    let trie_build_states: Vec<u32> = match config.initial_state_map {
        Some(init_map) => init_map.representative_original_ids.clone(),
        None => (0..tokenizer.num_states()).collect(),
    };
    let (mut trie_class_result, _) = collector::collect_possible_matches_interval_trie_class_build_with_classes(
        tokenizer,
        &trie.root,
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
    let (possible_match_vocab, possible_matches) = build_possible_match_vocab_and_weights_from_interval_maps(
        &trie_class_result.class_maps,
        &trie_class_result.state_classes,
        &ordered_vocab,
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
            "[glrmask/profile][possible_match_vocab] original_tokens={} ordered_byte_tokens={} possible_match_tokens={}",
            token_bytes.len(),
            ordered_vocab.ordered_to_originals.len(),
            possible_matches_id_map.vocab_tokens.internal_to_originals.len(),
        );
    }
    let possible_match_vocab_ms = elapsed_ms(possible_match_vocab_started_at);
    ConstraintPossibleMatchesComputation {
        mapped_possible_matches: MappedArtifact::new(possible_matches, possible_matches_id_map),
        profile: ConstraintPossibleMatchesProfile {
            possible_matches_collect_ms,
            possible_match_vocab_ms,
        },
    }
}

