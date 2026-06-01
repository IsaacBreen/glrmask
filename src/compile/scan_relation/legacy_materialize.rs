//! Legacy expanded sweep for validation and emergency fallback.
//!
//! This module intentionally keeps the older expansion path isolated.  It is
//! useful as an oracle for the grouped sweep, but publication-facing readers
//! should not have to read it to understand the main algorithm.

use super::prelude::*;
use super::collector::IntervalCanMatchMap;
use super::ordered_vocab::{range_set_from_sorted_ids, OrderedVocab};
use super::types::*;
use super::vocab_materialize::{intern_state_terminal_label, used_state_class_ids};

type ExpandedIntervalCanMatchMap = BTreeMap<TerminalID, Vec<(u32, u32)>>;

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
    map: &mut ExpandedIntervalCanMatchMap,
    terminal: TerminalID,
    ranges: &[(u32, u32)],
) {
    if !ranges.is_empty() {
        map.entry(terminal).or_default().extend_from_slice(ranges);
    }
}

fn normalize_expanded_interval_map(map: &mut ExpandedIntervalCanMatchMap) {
    map.retain(|_, ranges| {
        normalize_token_ranges(ranges);
        !ranges.is_empty()
    });
}

fn expand_interval_class_maps(
    class_maps: &[Arc<IntervalCanMatchMap>],
) -> Vec<Arc<ExpandedIntervalCanMatchMap>> {
    class_maps.iter().map(|class_map| {
        let mut expanded = ExpandedIntervalCanMatchMap::new();
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
    class_maps: &[Arc<ExpandedIntervalCanMatchMap>],
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
        assert!(*count > 0, "legacy scan-relation-vocab sweep removal underflow for label_id={}", event.label_id);
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

pub(super) fn build_legacy_scan_relation_vocab_and_weights_from_interval_maps(
    class_maps: &[Arc<IntervalCanMatchMap>],
    state_classes: &[u32],
    ordered_vocab: &OrderedVocab,
) -> (ScanRelationVocabMap, RuntimeCanMatchByTerminal) {
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

    let can_match = ids_by_label.into_iter().map(|(terminal_id, by_state)| {
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

    (ScanRelationVocabMap { original_to_internal, internal_to_originals }, can_match)
}

pub(super) fn validate_group_scan_relation_vocab_outputs(
    class_maps: &[Arc<IntervalCanMatchMap>],
    state_classes: &[u32],
    ordered_vocab: &OrderedVocab,
    actual_vocab: &ScanRelationVocabMap,
    actual_matches: &RuntimeCanMatchByTerminal,
) {
    let started_at = Instant::now();
    let (expected_vocab, expected_matches) =
        build_legacy_scan_relation_vocab_and_weights_from_interval_maps(class_maps, state_classes, ordered_vocab);

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
        panic!("group scan-relation vocab validation failed: original_to_internal mismatch at {:?}", mismatch);
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
        panic!("group scan-relation vocab validation failed: internal_to_originals mismatch at {:?}; actual_len={} expected_len={}", mismatch, actual_vocab.internal_to_originals.len(), expected_vocab.internal_to_originals.len());
    }
    if actual_matches != &expected_matches {
        let mut terminal_ids: Vec<TerminalID> = actual_matches.keys().chain(expected_matches.keys()).copied().collect();
        terminal_ids.sort_unstable();
        terminal_ids.dedup();
        let mismatch = terminal_ids.into_iter().find(|terminal_id| actual_matches.get(terminal_id) != expected_matches.get(terminal_id));
        panic!("group scan-relation vocab validation failed: can-match weight mismatch for terminal {:?}", mismatch);
    }

    if std::env::var_os("GLRMASK_PROFILE_SCAN_RELATION_VOCAB_DETAIL").is_some() {
        eprintln!("[glrmask/profile][scan_relation_vocab_validate] legacy_expand_compare_ms={:.3}", elapsed_ms(started_at));
    }
}
