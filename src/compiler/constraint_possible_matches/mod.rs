use std::collections::BTreeMap;
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
    label_id: u32,
}

#[derive(Debug, Default, Clone, Copy)]
struct SweepBuildStats {
    used_state_classes: usize,
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
    let mut entries: Vec<(Vec<u8>, u32)> = token_bytes.iter().map(|(&token_id, bytes)| (bytes.clone(), token_id)).collect();
    entries.sort_unstable_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

    let mut ordered_to_originals = Vec::new();
    let mut ordered_token_bytes = Vec::new();
    let mut index = 0usize;
    while index < entries.len() {
        let bytes = entries[index].0.clone();
        let mut originals = Vec::new();
        while index < entries.len() && entries[index].0 == bytes {
            originals.push(entries[index].1);
            index += 1;
        }
        originals.sort_unstable();
        originals.dedup();
        ordered_token_bytes.push(bytes);
        ordered_to_originals.push(originals);
    }

    OrderedVocab { original_slot_count, ordered_to_originals, ordered_token_bytes }
}

fn build_ordered_vocab_prefix_tree(ordered_vocab: &OrderedVocab) -> VocabPrefixTree {
    let entries: Vec<(usize, &[u8])> = ordered_vocab.ordered_token_bytes.iter().enumerate().map(|(ordered_id, bytes)| (ordered_id, bytes.as_slice())).collect();
    VocabPrefixTree::build_presorted(&entries)
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

fn push_sweep_event(events: &mut [Vec<SweepEvent>], event_positions: &mut Vec<u32>, position: u32, event: SweepEvent) {
    let Some(bucket) = events.get_mut(position as usize) else { return; };
    if bucket.is_empty() { event_positions.push(position); }
    bucket.push(event);
}

fn build_sweep_events(
    class_maps: &[Arc<IntervalPossibleMatchMap>],
    state_classes: &[u32],
    num_ordered_tokens: usize,
) -> (Vec<Vec<SweepEvent>>, Vec<u32>, Vec<StateTerminalLabel>, SweepBuildStats) {
    let mut events = vec![Vec::new(); num_ordered_tokens + 1];
    let mut event_positions = Vec::new();
    let mut labels_by_id = Vec::<StateTerminalLabel>::new();
    let mut label_ids = FxHashMap::<StateTerminalLabel, u32>::default();
    let mut stats = SweepBuildStats::default();

    let used_state_classes = used_state_class_ids(state_classes);
    stats.used_state_classes = used_state_classes.len();

    for class_id in used_state_classes {
        let Some(class_map) = class_maps.get(class_id as usize) else { continue; };
        for (&terminal_id, ranges) in class_map.iter() {
            let label = (class_id, terminal_id);
            let label_id = if let Some(&label_id) = label_ids.get(&label) {
                label_id
            } else {
                let label_id = labels_by_id.len() as u32;
                labels_by_id.push(label);
                label_ids.insert(label, label_id);
                label_id
            };
            for &(lo, mut hi) in ranges {
                if num_ordered_tokens == 0 { continue; }
                let max_token = num_ordered_tokens as u32 - 1;
                if lo > max_token { continue; }
                hi = hi.min(max_token);
                if lo > hi { continue; }
                stats.total_intervals += 1;
                push_sweep_event(&mut events, &mut event_positions, lo, SweepEvent { add: true, label_id });
                stats.total_events += 1;
                let after = hi.saturating_add(1);
                if after <= num_ordered_tokens as u32 {
                    push_sweep_event(&mut events, &mut event_positions, after, SweepEvent { add: false, label_id });
                    stats.total_events += 1;
                }
            }
        }
    }

    event_positions.sort_unstable();
    event_positions.dedup();
    (events, event_positions, labels_by_id, stats)
}

fn apply_sweep_events(active_counts: &mut [u32], events: &[SweepEvent], active_label_count: &mut usize) {
    for event in events.iter().filter(|event| !event.add) {
        let count = &mut active_counts[event.label_id as usize];
        assert!(*count > 0, "pmv sweep removal underflow for label_id={}", event.label_id);
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

fn build_possible_match_vocab_and_weights_from_interval_maps(
    class_maps: &[Arc<IntervalPossibleMatchMap>],
    state_classes: &[u32],
    ordered_vocab: &OrderedVocab,
) -> (PossibleMatchVocabMap, RuntimePossibleMatchesByTerminal) {
    let num_ordered_tokens = ordered_vocab.ordered_to_originals.len();
    let pmv_detail_enabled = std::env::var("GLRMASK_PROFILE_PMV_DETAIL")
        .map(|value| value == "1")
        .unwrap_or(false);

    let sweep_events_started_at = Instant::now();
    let (events, event_positions, labels_by_id, sweep_build_stats) =
        build_sweep_events(class_maps, state_classes, num_ordered_tokens);
    let sweep_events_ms = elapsed_ms(sweep_events_started_at);

    let mut signature_to_id: FxHashMap<Vec<StateTerminalLabel>, SignatureClassId> = FxHashMap::default();
    let mut signature_labels: Vec<Vec<StateTerminalLabel>> = Vec::new();
    let mut original_to_internal = vec![u32::MAX; ordered_vocab.original_slot_count];
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut active_counts = vec![0u32; labels_by_id.len()];
    let mut active_label_count = 0usize;

    let sweep_started_at = Instant::now();
    let mut signature_build_ms = 0.0;
    let mut signature_lookup_ms = 0.0;
    let mut assignment_ms = 0.0;
    let mut sweep_segments = 0usize;
    let mut total_active_signature_len = 0usize;
    let mut max_active_signature_len = 0usize;

    let mut event_index = 0usize;
    let mut position = 0usize;
    while position < num_ordered_tokens {
        while event_index < event_positions.len() && event_positions[event_index] as usize == position {
            apply_sweep_events(&mut active_counts, &events[position], &mut active_label_count);
            event_index += 1;
        }

        let next_position = event_positions.get(event_index).map(|&next| (next as usize).min(num_ordered_tokens)).unwrap_or(num_ordered_tokens);
        let signature_started_at = Instant::now();
        let mut signature = Vec::with_capacity(active_label_count);
        for (label_id, &label) in labels_by_id.iter().enumerate() {
            if active_counts[label_id] > 0 {
                signature.push(label);
            }
        }
        signature_build_ms += elapsed_ms(signature_started_at);
        sweep_segments += 1;
        total_active_signature_len += signature.len();
        max_active_signature_len = max_active_signature_len.max(signature.len());

        let signature_lookup_started_at = Instant::now();
        let signature_id = if let Some(&existing) = signature_to_id.get(&signature) { existing } else {
            let new_id = signature_labels.len() as SignatureClassId;
            signature_to_id.insert(signature.clone(), new_id);
            signature_labels.push(signature);
            internal_to_originals.push(Vec::new());
            new_id
        };
        signature_lookup_ms += elapsed_ms(signature_lookup_started_at);

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
    let mut ids_by_label: BTreeMap<TerminalID, BTreeMap<u32, Vec<u32>>> = BTreeMap::new();
    let mut label_entries = 0usize;
    for (signature_id, labels) in signature_labels.iter().enumerate() {
        let signature_id = signature_id as u32;
        for &(class_id, terminal_id) in labels {
            label_entries += 1;
            ids_by_label.entry(terminal_id).or_default().entry(class_id).or_default().push(signature_id);
        }
    }
    let ids_by_label_ms = elapsed_ms(ids_by_label_started_at);

    let weight_build_started_at = Instant::now();
    let terminal_ids = ids_by_label.len();
    let mut state_token_sets = 0usize;
    let possible_matches = ids_by_label.into_iter().map(|(terminal_id, by_state)| {
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
    }).filter(|(_, weight)| !weight.is_empty()).collect();
    let weight_build_ms = elapsed_ms(weight_build_started_at);

    if pmv_detail_enabled {
        let mean_active_signature_len = if sweep_segments == 0 {
            0.0
        } else {
            total_active_signature_len as f64 / sweep_segments as f64
        };
        eprintln!(
            "[glrmask/profile][pmv_detail] stage=sweep_events sweep_events_ms={:.3} event_positions={} total_events={} used_state_classes={} total_intervals={}",
            sweep_events_ms,
            event_positions.len(),
            sweep_build_stats.total_events,
            sweep_build_stats.used_state_classes,
            sweep_build_stats.total_intervals,
        );
        eprintln!(
            "[glrmask/profile][pmv_detail] stage=sweep sweep_ms={:.3} segments={} signature_build_ms={:.3} signature_lookup_ms={:.3} assignment_ms={:.3} unique_signatures={} max_active_signature_len={} mean_active_signature_len={:.3}",
            sweep_ms,
            sweep_segments,
            signature_build_ms,
            signature_lookup_ms,
            assignment_ms,
            signature_labels.len(),
            max_active_signature_len,
            mean_active_signature_len,
        );
        eprintln!(
            "[glrmask/profile][pmv_detail] stage=sort_dedup sort_dedup_ms={:.3} internal_signature_classes={}",
            sort_dedup_ms,
            internal_to_originals.len(),
        );
        eprintln!(
            "[glrmask/profile][pmv_detail] stage=ids_by_label ids_by_label_ms={:.3} label_entries={} terminal_ids={}",
            ids_by_label_ms,
            label_entries,
            terminal_ids,
        );
        eprintln!(
            "[glrmask/profile][pmv_detail] stage=weights weights_ms={:.3} terminal_ids={} state_token_sets={}",
            weight_build_ms,
            terminal_ids,
            state_token_sets,
        );
    }

    (PossibleMatchVocabMap { original_to_internal, internal_to_originals }, possible_matches)
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
    let canonical_states = config
        .initial_state_map
        .map(|init_map| canonical_states_from_initial_map(init_map, tokenizer.num_states()));

    let (mut trie_class_result, _) = collector::collect_possible_matches_interval_trie_class_build_with_classes(tokenizer, &trie.root, &trie_build_states, canonical_states.as_deref());
    if let Some(init_map) = config.initial_state_map {
        trie_class_result.state_classes = compose_state_classes_with_initial_map(&trie_class_result.state_classes, init_map);
    }
    let possible_matches_collect_ms = elapsed_ms(pm_started_at);

    let possible_match_vocab_started_at = Instant::now();
    let (possible_match_vocab, possible_matches) = build_possible_match_vocab_and_weights_from_interval_maps(&trie_class_result.class_maps, &trie_class_result.state_classes, &ordered_vocab);

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
