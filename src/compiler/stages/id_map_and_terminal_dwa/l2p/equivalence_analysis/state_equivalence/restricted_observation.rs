//! Exact state quotient for L2P's restricted observation model.
//!
//! This pass does not modify the lexer DFA.  It computes a `ManyToOneIdMap`
//! over its original state IDs.  Its observation alphabet consists of the
//! bytes present in the current vocabulary partition, and its terminal labels
//! are restricted to the active L2P terminals (TI representatives when TI is
//! enabled).  Future-terminal labels are read from the original DFA and never
//! recomputed after restricting bytes.

use std::time::Instant;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;

use super::max_length::active_byte_representatives;

const NO_CANDIDATE: usize = usize::MAX;

#[inline]
fn signature_fingerprint_step(mut hash: u64, word: u64) -> u64 {
    hash ^= word.wrapping_add(0x9e37_79b9_7f4a_7c15);
    hash = hash.rotate_left(27).wrapping_mul(0x3c79_ac49_2ba7_b653);
    hash ^= hash >> 33;
    hash
}

#[inline]
fn signature_slot_fingerprint(slot: usize, word: u64) -> u64 {
    signature_fingerprint_step(
        0xd6e8_feb8_6659_fd93_u64 ^ (slot as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        word,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TargetLabels {
    finalizers: SmallVec<[u32; 4]>,
    future_finalizers: SmallVec<[u32; 4]>,
}

fn target_label_ids(tokenizer: &Tokenizer, active_groups: Option<&[bool]>) -> Vec<u32> {
    let num_states = tokenizer.num_states() as usize;
    let mut ids = vec![0u32; num_states];
    let mut labels = Vec::<TargetLabels>::new();
    let mut first_by_fingerprint = FxHashMap::<u64, usize>::default();
    first_by_fingerprint.reserve(num_states / 2);
    let mut next_same_fingerprint = Vec::<usize>::new();

    for state in 0..num_states {
        // Tokenizer terminal iterators traverse the backing BitSet, which is
        // ascending and duplicate-free. Preserve that canonical order without
        // per-state sorting or heap allocation for the usual tiny label sets.
        let mut finalizers = SmallVec::<[u32; 4]>::new();
        let mut future_finalizers = SmallVec::<[u32; 4]>::new();
        let mut fingerprint = 0x4d5f_3a17_9b28_c6e1_u64;

        match active_groups {
            Some(groups) => {
                for terminal in tokenizer.matched_terminals_iter(state as u32) {
                    if groups.get(terminal as usize).copied().unwrap_or(false) {
                        finalizers.push(terminal);
                        fingerprint = signature_fingerprint_step(fingerprint, terminal as u64);
                    }
                }
                fingerprint = signature_fingerprint_step(fingerprint, u64::MAX);
                for terminal in tokenizer.possible_future_terminals_iter(state as u32) {
                    if groups.get(terminal as usize).copied().unwrap_or(false) {
                        future_finalizers.push(terminal);
                        fingerprint = signature_fingerprint_step(fingerprint, terminal as u64);
                    }
                }
            }
            None => {
                for terminal in tokenizer.matched_terminals_iter(state as u32) {
                    finalizers.push(terminal);
                    fingerprint = signature_fingerprint_step(fingerprint, terminal as u64);
                }
                fingerprint = signature_fingerprint_step(fingerprint, u64::MAX);
                for terminal in tokenizer.possible_future_terminals_iter(state as u32) {
                    future_finalizers.push(terminal);
                    fingerprint = signature_fingerprint_step(fingerprint, terminal as u64);
                }
            }
        }

        let mut matching = first_by_fingerprint
            .get(&fingerprint)
            .copied()
            .unwrap_or(NO_CANDIDATE);
        while matching != NO_CANDIDATE {
            let previous = &labels[matching];
            if previous.finalizers == finalizers
                && previous.future_finalizers == future_finalizers
            {
                break;
            }
            matching = next_same_fingerprint[matching];
        }

        let id = if matching != NO_CANDIDATE {
            matching
        } else {
            let id = labels.len();
            let previous = first_by_fingerprint.insert(fingerprint, id);
            labels.push(TargetLabels {
                finalizers,
                future_finalizers,
            });
            next_same_fingerprint.push(previous.unwrap_or(NO_CANDIDATE));
            id
        };
        ids[state] = id as u32;
    }

    ids
}

/// Build the candidate-state partition inherited from an earlier pass.
///
/// Every raw lexer state remains represented.  A prior map contributes one
/// candidate for each of its classes; raw states absent from that map are kept
/// as singleton candidates instead of being silently dropped.
fn candidate_partition(
    num_states: usize,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> (Vec<Vec<u32>>, Vec<usize>, Vec<usize>) {
    let mut members = Vec::<Vec<u32>>::new();
    let mut representatives = Vec::<usize>::new();
    let mut raw_to_candidate = vec![NO_CANDIDATE; num_states];

    if let Some(map) = initial_state_map {
        for originals in &map.internal_to_originals {
            let mut candidate_members = Vec::with_capacity(originals.len());
            for &raw in originals {
                let raw = raw as usize;
                if raw < num_states && raw_to_candidate[raw] == NO_CANDIDATE {
                    candidate_members.push(raw as u32);
                }
            }
            if candidate_members.is_empty() {
                continue;
            }
            let candidate = members.len();
            let representative = candidate_members[0] as usize;
            for &raw in &candidate_members {
                raw_to_candidate[raw as usize] = candidate;
            }
            representatives.push(representative);
            members.push(candidate_members);
        }
    }

    for raw in 0..num_states {
        if raw_to_candidate[raw] != NO_CANDIDATE {
            continue;
        }
        let candidate = members.len();
        raw_to_candidate[raw] = candidate;
        representatives.push(raw);
        members.push(vec![raw as u32]);
    }

    (members, representatives, raw_to_candidate)
}

fn map_from_candidate_classes(
    candidate_members: &[Vec<u32>],
    candidate_representatives: &[usize],
    candidate_classes: &[u32],
    num_states: usize,
) -> ManyToOneIdMap {
    let num_classes = candidate_classes
        .iter()
        .copied()
        .max()
        .map_or(0, |class| class + 1);
    let mut original_to_internal = vec![u32::MAX; num_states];
    let mut internal_to_originals = vec![Vec::new(); num_classes as usize];
    let mut representative_original_ids = vec![u32::MAX; num_classes as usize];

    for ((members, &representative), &class) in candidate_members
        .iter()
        .zip(candidate_representatives)
        .zip(candidate_classes)
    {
        let bucket = &mut internal_to_originals[class as usize];
        if bucket.is_empty() {
            representative_original_ids[class as usize] = representative as u32;
        }
        for &raw in members {
            original_to_internal[raw as usize] = class;
            bucket.push(raw);
        }
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

/// Compute the coarsest fixed point of the restricted observation recurrence:
///
/// `s ↦ ((b, class(dst(s,b)), F(dst(s,b)), U(dst(s,b))) for b in bytes)`.
///
/// `F` and `U` are the original DFA's finalizer and future-finalizer sets,
/// filtered only by `active_groups`.  They are intentionally not recomputed
/// from the byte-restricted transition relation.
pub(crate) fn compute_state_map(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    initial_state_map: Option<&ManyToOneIdMap>,
    active_groups: Option<&[bool]>,
    byte_to_class: Option<&[u8; 256]>,
) -> ManyToOneIdMap {
    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
    let total_started_at = Instant::now();
    let num_states = tokenizer.num_states() as usize;
    if num_states == 0 {
        return ManyToOneIdMap::from_original_to_internal_allowing_unmapped(Vec::new(), 0);
    }

    let active_bytes = active_byte_representatives(Some(relevant_bytes), byte_to_class);
    let target_labels_started_at = Instant::now();
    let target_labels = target_label_ids(tokenizer, active_groups);
    let target_labels_ms = target_labels_started_at.elapsed().as_secs_f64() * 1000.0;
    let target_label_classes = target_labels
        .iter()
        .copied()
        .max()
        .map_or(0usize, |label| label as usize + 1);

    let candidate_partition_started_at = Instant::now();
    let (candidate_members, candidate_representatives, raw_to_candidate) =
        candidate_partition(num_states, initial_state_map);
    let candidate_partition_ms = candidate_partition_started_at.elapsed().as_secs_f64() * 1000.0;
    let num_candidates = candidate_representatives.len();

    let observation_cache_started_at = Instant::now();
    let observation_width = active_bytes.len();
    let signature_width = observation_width;
    let mut observed_targets = vec![0u64; num_candidates * observation_width];
    let mut signatures = vec![0u64; num_candidates * signature_width];
    let mut signature_fingerprints = vec![0u64; num_candidates];
    // Keep every observed source-slot edge. The refinement cache below updates
    // exactly those signature words when a destination candidate changes
    // class, rather than rebuilding all words of every affected source.
    let mut reverse_offsets = vec![0usize; num_candidates + 1];
    for (candidate, &state) in candidate_representatives.iter().enumerate() {
        let observation_start = candidate * observation_width;
        for (slot, &byte) in active_bytes.iter().enumerate() {
            let target = tokenizer.get_transition(state as u32, byte);
            if target == u32::MAX {
                continue;
            }
            let target_candidate = raw_to_candidate[target as usize];
            debug_assert_ne!(target_candidate, NO_CANDIDATE);
            let labels = target_labels[target as usize] as u64 + 1;
            let observation = (labels << 32) | (target_candidate as u64 + 1);
            observed_targets[observation_start + slot] = observation;
            let signature_word = (labels << 32) | 1;
            signatures[observation_start + slot] = signature_word;
            signature_fingerprints[candidate] ^=
                signature_slot_fingerprint(slot, signature_word);
            reverse_offsets[target_candidate + 1] += 1;
        }
    }
    let observation_cache_ms = observation_cache_started_at.elapsed().as_secs_f64() * 1000.0;

    // Inverse links use original candidate IDs only; this is not a second
    // tokenizer coordinate.
    let reverse_cache_started_at = Instant::now();
    for candidate in 1..=num_candidates {
        reverse_offsets[candidate] += reverse_offsets[candidate - 1];
    }
    let mut reverse_cursor = reverse_offsets[..num_candidates].to_vec();
    let mut reverse_edges = vec![0u64; reverse_offsets[num_candidates]];
    for source in 0..num_candidates {
        let observation_start = source * observation_width;
        for (slot, &observation) in observed_targets
            [observation_start..observation_start + observation_width]
            .iter()
            .enumerate()
        {
            if observation != 0 {
                let target = (observation as u32 - 1) as usize;
                let edge = reverse_cursor[target];
                reverse_edges[edge] = (source as u64) << 8 | slot as u64;
                reverse_cursor[target] += 1;
            }
        }
    }
    let reverse_cache_ms = reverse_cache_started_at.elapsed().as_secs_f64() * 1000.0;
    // Each refinement only splits the preceding classes. A class can change
    // only when one of its observations enters a class split in the preceding
    // round, so untouched classes retain their exact prior signature.
    let refinement_started_at = Instant::now();
    let mut current_classes = vec![0u32; num_candidates];
    let mut next_classes = vec![0u32; num_candidates];
    let mut class_members = vec![(0..num_candidates).collect::<Vec<_>>()];
    let mut first_candidate_by_fingerprint = FxHashMap::<u64, usize>::default();
    first_candidate_by_fingerprint.reserve(num_candidates);
    let mut next_same_fingerprint = vec![NO_CANDIDATE; num_candidates];
    let mut refinement_rounds = 0usize;
    let mut refined_candidate_visits = 0usize;
    let mut refined_class_visits = 0usize;
    let mut dirty_classes = vec![0usize];
    let mut dirty_flags = vec![true];

    while !dirty_classes.is_empty() {
        refinement_rounds += 1;
        next_classes.copy_from_slice(&current_classes);
        let mut moved_candidates = Vec::new();
        let current_dirty_classes = std::mem::take(&mut dirty_classes);

        for class in current_dirty_classes {
            dirty_flags[class] = false;
            if class_members[class].len() <= 1 {
                continue;
            }

            let members = std::mem::take(&mut class_members[class]);
            refined_candidate_visits += members.len();
            refined_class_visits += 1;
            let new_class_base = class_members.len();
            let mut retained_members = Vec::with_capacity(members.len());
            let mut split_members = Vec::<Vec<usize>>::new();
            first_candidate_by_fingerprint.clear();

            for candidate in members {
                let signature_start = candidate * signature_width;
                let signature_end = signature_start + signature_width;
                let fingerprint = signature_fingerprints[candidate];

                let matching_candidate = {
                    let mut matching_candidate = first_candidate_by_fingerprint
                        .get(&fingerprint)
                        .copied()
                        .unwrap_or(NO_CANDIDATE);
                    while matching_candidate != NO_CANDIDATE {
                        let matching_start = matching_candidate * signature_width;
                        if signatures[signature_start..signature_end]
                            == signatures[matching_start..matching_start + signature_width]
                        {
                            break;
                        }
                        matching_candidate = next_same_fingerprint[matching_candidate];
                    }
                    matching_candidate
                };

                let next_class = if matching_candidate != NO_CANDIDATE {
                    next_classes[matching_candidate] as usize
                } else {
                    let next_class = if retained_members.is_empty() {
                        class
                    } else {
                        let next_class = new_class_base + split_members.len();
                        split_members.push(Vec::new());
                        next_class
                    };
                    let previous = first_candidate_by_fingerprint.insert(fingerprint, candidate);
                    next_same_fingerprint[candidate] = previous.unwrap_or(NO_CANDIDATE);
                    next_class
                };

                next_classes[candidate] = next_class as u32;
                if next_class == class {
                    retained_members.push(candidate);
                } else {
                    split_members[next_class - new_class_base].push(candidate);
                }
            }

            debug_assert!(!retained_members.is_empty());
            class_members[class] = retained_members;
            for members in &split_members {
                moved_candidates.extend(members.iter().copied());
            }
            class_members.extend(split_members);
            dirty_flags.resize(class_members.len(), false);
        }

        if moved_candidates.is_empty() {
            let refinement_ms = refinement_started_at.elapsed().as_secs_f64() * 1000.0;
            let map_materialize_started_at = Instant::now();
            let result = map_from_candidate_classes(
                &candidate_members,
                &candidate_representatives,
                &current_classes,
                num_states,
            );
            let map_materialize_ms = map_materialize_started_at.elapsed().as_secs_f64() * 1000.0;
            if profile_enabled {
                eprintln!(
                    "[glrmask/profile][restricted_observation] states={} candidates={} active_bytes={} target_label_classes={} target_labels_ms={:.3} candidate_partition_ms={:.3} observation_cache_ms={:.3} refinement_ms={:.3} map_materialize_ms={:.3} rounds={} reps={} total_ms={:.3}",
                    num_states,
                    num_candidates,
                    active_bytes.len(),
                    target_label_classes,
                    target_labels_ms,
                    candidate_partition_ms,
                    observation_cache_ms,
                    refinement_ms,
                    map_materialize_ms,
                    refinement_rounds,
                    class_members.len(),
                    total_started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            return result;
        }

        for target in moved_candidates {
            let target_class = next_classes[target] as u64 + 1;
            for &edge in &reverse_edges[reverse_offsets[target]..reverse_offsets[target + 1]] {
                let source = (edge >> 8) as usize;
                let slot = (edge & 0xff) as usize;
                let signature_index = source * signature_width + slot;
                let old_word = signatures[signature_index];
                let new_word = (old_word & 0xffff_ffff_0000_0000) | target_class;
                debug_assert_ne!(old_word, 0);
                debug_assert_ne!(old_word, new_word);
                signatures[signature_index] = new_word;
                signature_fingerprints[source] ^=
                    signature_slot_fingerprint(slot, old_word)
                        ^ signature_slot_fingerprint(slot, new_word);
                let source_class = next_classes[source] as usize;
                if class_members[source_class].len() > 1 && !dirty_flags[source_class] {
                    dirty_flags[source_class] = true;
                    dirty_classes.push(source_class);
                }
            }
        }
        std::mem::swap(&mut current_classes, &mut next_classes);
    }

    let refinement_ms = refinement_started_at.elapsed().as_secs_f64() * 1000.0;
    let map_materialize_started_at = Instant::now();
    let result = map_from_candidate_classes(
        &candidate_members,
        &candidate_representatives,
        &current_classes,
        num_states,
    );
    let map_materialize_ms = map_materialize_started_at.elapsed().as_secs_f64() * 1000.0;
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][restricted_observation] states={} candidates={} active_bytes={} target_label_classes={} target_labels_ms={:.3} candidate_partition_ms={:.3} observation_cache_ms={:.3} reverse_cache_ms={:.3} refinement_ms={:.3} map_materialize_ms={:.3} rounds={} reps={} total_ms={:.3}",
            num_states,
            num_candidates,
            active_bytes.len(),
            target_label_classes,
            target_labels_ms,
            candidate_partition_ms,
            observation_cache_ms,
            reverse_cache_ms,
            refinement_ms,
            map_materialize_ms,
            refinement_rounds,
            class_members.len(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    result
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex;

    fn tokenizer(expressions: Vec<Expr>) -> Tokenizer {
        let terminal_count = expressions.len() as u32;
        build_regex(&expressions).into_tokenizer(
            terminal_count,
            Some(Arc::from(expressions.into_boxed_slice())),
        )
    }

    fn class_of(map: &ManyToOneIdMap, state: u32) -> u32 {
        map.original_to_internal[state as usize]
    }

    fn reference_state_map(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        initial_state_map: Option<&ManyToOneIdMap>,
        active_groups: Option<&[bool]>,
        byte_to_class: Option<&[u8; 256]>,
    ) -> ManyToOneIdMap {
        let num_states = tokenizer.num_states() as usize;
        let active_bytes = active_byte_representatives(Some(relevant_bytes), byte_to_class);
        let target_labels = target_label_ids(tokenizer, active_groups);
        let (candidate_members, candidate_representatives, raw_to_candidate) =
            candidate_partition(num_states, initial_state_map);
        let num_candidates = candidate_representatives.len();
        let mut current_classes = vec![0u32; num_candidates];
        let mut current_class_count = usize::from(num_candidates != 0);
        let mut signature = vec![0u64; 1 + active_bytes.len()];

        for _ in 0..num_candidates {
            let mut next_classes = vec![0u32; num_candidates];
            let mut classes_by_signature = FxHashMap::<Vec<u64>, u32>::default();

            for (candidate, &state) in candidate_representatives.iter().enumerate() {
                signature[0] = current_classes[candidate] as u64;
                for (slot, &byte) in active_bytes.iter().enumerate() {
                    let target = tokenizer.get_transition(state as u32, byte);
                    signature[slot + 1] = if target == u32::MAX {
                        0
                    } else {
                        let target_candidate = raw_to_candidate[target as usize];
                        let target_class = current_classes[target_candidate] as u64 + 1;
                        let labels = target_labels[target as usize] as u64 + 1;
                        (labels << 32) | target_class
                    };
                }
                let next_class = classes_by_signature.len() as u32;
                next_classes[candidate] = *classes_by_signature
                    .entry(signature.clone())
                    .or_insert(next_class);
            }

            let next_class_count = classes_by_signature.len();
            if next_class_count == current_class_count {
                return map_from_candidate_classes(
                    &candidate_members,
                    &candidate_representatives,
                    &current_classes,
                    num_states,
                );
            }
            current_classes = next_classes;
            current_class_count = next_class_count;
        }
        unreachable!()
    }

    fn assert_same_partition(left: &ManyToOneIdMap, right: &ManyToOneIdMap) {
        assert_eq!(left.original_to_internal.len(), right.original_to_internal.len());
        for state in 0..left.original_to_internal.len() {
            for other in 0..left.original_to_internal.len() {
                assert_eq!(
                    left.original_to_internal[state] == left.original_to_internal[other],
                    right.original_to_internal[state] == right.original_to_internal[other],
                    "states {state} and {other} differ",
                );
            }
        }
    }

    #[test]
    fn frozen_future_labels_survive_byte_restriction_and_obey_active_mask() {
        // `x` and `y` are deliberately absent from the restricted byte set.
        // The states after `a` and `b` nevertheless remain distinguishable
        // through their `c` successors' *original* future-terminal labels.
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"acx".to_vec()),
            Expr::U8Seq(b"bcy".to_vec()),
        ]);
        let start = tokenizer.initial_state_id();
        let after_a = tokenizer.get_transition(start, b'a');
        let after_b = tokenizer.get_transition(start, b'b');
        let after_ac = tokenizer.get_transition(after_a, b'c');
        let after_bc = tokenizer.get_transition(after_b, b'c');
        assert_ne!(after_a, u32::MAX);
        assert_ne!(after_b, u32::MAX);
        assert_ne!(after_ac, u32::MAX);
        assert_ne!(after_bc, u32::MAX);
        assert!(tokenizer.possible_future_terminals_iter(after_ac).any(|t| t == 0));
        assert!(tokenizer.possible_future_terminals_iter(after_bc).any(|t| t == 1));

        let mut only_c = [false; 256];
        only_c[b'c' as usize] = true;

        let all_active = compute_state_map(&tokenizer, &only_c, None, Some(&[true, true]), None);
        assert_ne!(
            class_of(&all_active, after_a),
            class_of(&all_active, after_b),
            "frozen future-terminal labels after `c` must remain observable even though x/y are restricted out",
        );

        let none_active =
            compute_state_map(&tokenizer, &only_c, None, Some(&[false, false]), None);
        assert_eq!(
            class_of(&none_active, after_a),
            class_of(&none_active, after_b),
            "active-terminal filtering must remove the only remaining observation",
        );

        let no_bytes = [false; 256];
        let no_byte_observation =
            compute_state_map(&tokenizer, &no_bytes, None, Some(&[true, true]), None);
        assert_eq!(
            class_of(&no_byte_observation, after_a),
            class_of(&no_byte_observation, after_b),
            "without c in the byte set the future labels are not reached by the recurrence",
        );
    }

    #[test]
    fn class_local_refinement_matches_dense_reference() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"abx".to_vec()),
            Expr::U8Seq(b"aby".to_vec()),
            Expr::U8Seq(b"acx".to_vec()),
            Expr::U8Seq(b"acy".to_vec()),
            Expr::U8Seq(b"bbx".to_vec()),
            Expr::U8Seq(b"bby".to_vec()),
        ]);
        let mut bytes = [false; 256];
        for &byte in b"abcxy" {
            bytes[byte as usize] = true;
        }
        let mut byte_to_class = [0u8; 256];
        for byte in 0..256 {
            byte_to_class[byte] = byte as u8;
        }
        let raw_groups = (0..tokenizer.num_states())
            .map(|state| state % 3)
            .collect::<Vec<_>>();
        let initial = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(raw_groups, 3);
        let active = [true, false, true, false, true, true];

        for initial_state_map in [None, Some(&initial)] {
            for active_groups in [None, Some(active.as_slice())] {
                for byte_classes in [None, Some(&byte_to_class)] {
                    let reference = reference_state_map(
                        &tokenizer,
                        &bytes,
                        initial_state_map,
                        active_groups,
                        byte_classes,
                    );
                    let actual = compute_state_map(
                        &tokenizer,
                        &bytes,
                        initial_state_map,
                        active_groups,
                        byte_classes,
                    );
                    assert_same_partition(&reference, &actual);
                }
            }
        }
    }
}
