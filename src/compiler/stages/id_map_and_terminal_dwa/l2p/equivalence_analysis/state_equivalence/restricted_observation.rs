//! Exact state quotient for L2P's restricted observation model.
//!
//! This pass does not modify the lexer DFA.  It computes a `ManyToOneIdMap`
//! over its original state IDs.  Its observation alphabet consists of the
//! bytes present in the current vocabulary partition, and its terminal labels
//! are restricted to the active L2P terminals (TI representatives when TI is
//! enabled).  Future-terminal labels are read from the original DFA and never
//! recomputed after restricting bytes.

use rustc_hash::FxHashMap;

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;

use super::max_length::active_byte_representatives;

const NO_CANDIDATE: usize = usize::MAX;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct TargetLabels {
    finalizers: Vec<u32>,
    future_finalizers: Vec<u32>,
}

#[inline]
fn terminal_is_active(terminal: u32, active_groups: Option<&[bool]>) -> bool {
    active_groups.map_or(true, |groups| {
        groups
            .get(terminal as usize)
            .copied()
            .unwrap_or(false)
    })
}

fn target_label_ids(tokenizer: &Tokenizer, active_groups: Option<&[bool]>) -> Vec<u32> {
    let mut ids = vec![0u32; tokenizer.num_states() as usize];
    let mut ids_by_labels = FxHashMap::<TargetLabels, u32>::default();

    for state in 0..tokenizer.num_states() as usize {
        let mut finalizers = tokenizer
            .matched_terminals_iter(state as u32)
            .filter(|&terminal| terminal_is_active(terminal, active_groups))
            .collect::<Vec<_>>();
        let mut future_finalizers = tokenizer
            .possible_future_terminals_iter(state as u32)
            .filter(|&terminal| terminal_is_active(terminal, active_groups))
            .collect::<Vec<_>>();
        // Lexer iterators are normally ordered, but make the observation
        // canonical at this boundary rather than relying on that detail.
        finalizers.sort_unstable();
        finalizers.dedup();
        future_finalizers.sort_unstable();
        future_finalizers.dedup();

        let next = ids_by_labels.len() as u32;
        let id = *ids_by_labels
            .entry(TargetLabels {
                finalizers,
                future_finalizers,
            })
            .or_insert(next);
        ids[state] = id;
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
    let num_states = tokenizer.num_states() as usize;
    if num_states == 0 {
        return ManyToOneIdMap::from_original_to_internal_allowing_unmapped(Vec::new(), 0);
    }

    let active_bytes = active_byte_representatives(Some(relevant_bytes), byte_to_class);
    let target_labels = target_label_ids(tokenizer, active_groups);
    let (candidate_members, candidate_representatives, raw_to_candidate) =
        candidate_partition(num_states, initial_state_map);
    let num_candidates = candidate_representatives.len();

    // At depth zero every candidate has the same recursive characterization.
    // Each refinement prefixes the previous class, making the partition
    // monotone while its hash-table key remains the complete collision-safe
    // signature.
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
                    debug_assert_ne!(target_candidate, NO_CANDIDATE);
                    let target_class = current_classes[target_candidate] as u64 + 1;
                    let labels = target_labels[target as usize] as u64 + 1;
                    (labels << 32) | target_class
                };
            }

            let next_class = classes_by_signature.len() as u32;
            let class = *classes_by_signature
                .entry(signature.clone())
                .or_insert(next_class);
            next_classes[candidate] = class;
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

    unreachable!("restricted-observation partition refinement did not stabilize");
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
}
