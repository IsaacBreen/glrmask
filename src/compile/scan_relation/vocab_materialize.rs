//! Sweep-line materialization of CanMatch weights.
//!
//! The collector gives a compact interval description: state-class ids paired
//! with terminal sets and token ranges.  This module turns that description into
//! a runtime vocabulary quotient plus `Weight`s.  The key invariant is that two
//! tokens share a scan-relation internal id iff their active `(state, terminal)`
//! label set is identical.

use super::prelude::*;
use super::collector::IntervalCanMatchMap;
use super::legacy_materialize::{
    build_legacy_scan_relation_vocab_and_weights_from_interval_maps,
    validate_group_scan_relation_vocab_outputs,
};
use super::ordered_vocab::{range_set_from_sorted_ids, range_set_from_u128_mask, OrderedVocab};
use super::root_collect::{
    group_scan_relation_vocab_legacy_enabled,
    group_scan_relation_vocab_validation_enabled,
};
use super::types::*;

pub(super) fn used_state_class_ids(state_classes: &[u32]) -> Vec<u32> {
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

pub(super) fn intern_state_terminal_label(
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
    class_maps: &[Arc<IntervalCanMatchMap>],
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
        assert!(*count > 0, "scan-relation-vocab sweep removal underflow for group_id={}", event.group_id);
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

pub(super) fn build_scan_relation_vocab_and_weights_from_interval_maps(
    class_maps: &[Arc<IntervalCanMatchMap>],
    state_classes: &[u32],
    ordered_vocab: &OrderedVocab,
) -> (ScanRelationVocabMap, RuntimeCanMatchByTerminal) {
    let num_ordered_tokens = ordered_vocab.ordered_to_originals.len();
    let scan_relation_vocab_detail_enabled = std::env::var("GLRMASK_PROFILE_SCAN_RELATION_VOCAB_DETAIL")
        .map(|value| value == "1")
        .unwrap_or(false);

    if group_scan_relation_vocab_legacy_enabled() {
        if scan_relation_vocab_detail_enabled {
            eprintln!("[glrmask/profile][scan_relation_vocab_detail] stage=legacy_expanded enabled=1");
        }
        return build_legacy_scan_relation_vocab_and_weights_from_interval_maps(class_maps, state_classes, ordered_vocab);
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
    let can_match: RuntimeCanMatchByTerminal = if use_bitmask_ids_by_label {
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
    let terminal_ids = can_match.len();
    let weight_build_ms = elapsed_ms(weight_build_started_at);

    if scan_relation_vocab_detail_enabled {
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
            "[glrmask/profile][scan_relation_vocab_detail] stage=group_sweep_events sweep_events_ms={:.3} event_positions={} total_group_events={} used_state_classes={} total_group_intervals={} terminal_groups={} terminal_labels={} group_label_refs={}",
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
            "[glrmask/profile][scan_relation_vocab_detail] stage=sweep sweep_ms={:.3} segments={} signature_build_ms={:.3} signature_lookup_ms={:.3} assignment_ms={:.3} active_group_signature_cache_hits={} active_group_signature_cache_misses={} active_group_signature_build_ms={:.3} label_signature_build_ms={:.3} unique_signatures={} max_active_signature_len={} mean_active_signature_len={:.3} max_active_groups={} mean_active_groups={:.3}",
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
            "[glrmask/profile][scan_relation_vocab_detail] stage=sort_dedup sort_dedup_ms={:.3} internal_signature_classes={}",
            sort_dedup_ms,
            internal_to_originals.len(),
        );
        eprintln!(
            "[glrmask/profile][scan_relation_vocab_detail] stage=ids_by_label ids_by_label_ms={:.3} label_entries={} terminal_ids={} bitmask_path_used={}",
            ids_by_label_ms,
            label_entries,
            terminal_ids,
            use_bitmask_ids_by_label,
        );
        eprintln!(
            "[glrmask/profile][scan_relation_vocab_detail] stage=weights weights_ms={:.3} terminal_ids={} state_token_sets={} bitmask_path_used={} bitmask_unique_masks={} bitmask_mask_cache_hits={} bitmask_mask_cache_misses={}",
            weight_build_ms,
            terminal_ids,
            state_token_sets,
            use_bitmask_ids_by_label,
            bitmask_unique_masks,
            bitmask_mask_cache_hits,
            bitmask_mask_cache_misses,
        );
    }

    let scan_relation_vocab = ScanRelationVocabMap { original_to_internal, internal_to_originals };
    if group_scan_relation_vocab_validation_enabled() {
        validate_group_scan_relation_vocab_outputs(class_maps, state_classes, ordered_vocab, &scan_relation_vocab, &can_match);
    }

    (scan_relation_vocab, can_match)
}
