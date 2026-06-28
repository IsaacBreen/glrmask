//! Exact semantic comparison for partition-local terminal DWA artifacts.
//!
//! A terminal DWA evaluates a terminal-label word by intersecting its transition
//! and final weights. Equivalent artifacts can distribute the same restriction
//! across different edges, so structural edge-weight equality is too strong.
//!
//! The comparison projects artifacts to Boolean DWAs. Coordinates are grouped by
//! the pair of internal state/token IDs they occupy in the baseline and
//! candidate id maps. Every original coordinate in one group induces exactly the
//! same two Boolean DWAs, so checking one representative is an exact quotient of
//! the original coordinate product rather than sampling.

use std::collections::{BTreeSet, VecDeque};

use crate::automata::weighted_u32::dwa::DWA;
use crate::compiler::stages::id_map_and_terminal_dwa::types::LocalIdMapTerminalDwa;
use crate::ds::weight::Weight;

const UNMAPPED: u32 = u32::MAX;

pub(crate) fn compare(
    baseline: &LocalIdMapTerminalDwa,
    candidate: &LocalIdMapTerminalDwa,
) -> Result<(), String> {
    let state_classes = paired_internal_classes(
        &baseline.id_map.tokenizer_states.original_to_internal,
        &candidate.id_map.tokenizer_states.original_to_internal,
    );
    let token_classes = paired_internal_classes(
        &baseline.id_map.vocab_tokens.original_to_internal,
        &candidate.id_map.vocab_tokens.original_to_internal,
    );

    for &(baseline_state, candidate_state) in &state_classes {
        for &(baseline_token, candidate_token) in &token_classes {
            compare_boolean_dfas(
                &baseline.dwa,
                &candidate.dwa,
                baseline_state,
                baseline_token,
                candidate_state,
                candidate_token,
            )?;
        }
    }
    Ok(())
}

/// Return every distinct pair of internal IDs occupied by an original axis
/// coordinate. A pair with both entries unmapped is absent from both artifacts
/// and cannot affect either weighted language.
fn paired_internal_classes(left: &[u32], right: &[u32]) -> Vec<(u32, u32)> {
    let mut classes = BTreeSet::new();
    for original in 0..left.len().max(right.len()) {
        let left_id = left.get(original).copied().unwrap_or(UNMAPPED);
        let right_id = right.get(original).copied().unwrap_or(UNMAPPED);
        if left_id != UNMAPPED || right_id != UNMAPPED {
            classes.insert((left_id, right_id));
        }
    }
    classes.into_iter().collect()
}

fn outgoing_labels(dwa: &DWA, state: Option<u32>) -> Vec<i32> {
    state
        .and_then(|state| dwa.states().get(state as usize))
        .map(|state| state.transitions.keys().copied().collect())
        .unwrap_or_default()
}

fn accepts_final(dwa: &DWA, state: Option<u32>, internal_state: u32, internal_token: u32) -> bool {
    state
        .and_then(|id| dwa.states().get(id as usize))
        .and_then(|node| node.final_weight.as_ref())
        .is_some_and(|weight| contains(weight, internal_state, internal_token))
}

fn enabled_target(
    dwa: &DWA,
    state: Option<u32>,
    label: i32,
    internal_state: u32,
    internal_token: u32,
) -> Option<u32> {
    let state = state?;
    let (target, weight) = dwa.states()[state as usize].transitions.get(&label)?;
    contains(weight, internal_state, internal_token).then_some(*target)
}

fn contains(weight: &Weight, internal_state: u32, internal_token: u32) -> bool {
    internal_state != UNMAPPED
        && internal_token != UNMAPPED
        && (weight.is_full() || weight.tokens_for_tsid(internal_state).contains(internal_token))
}

fn compare_boolean_dfas(
    baseline: &DWA,
    candidate: &DWA,
    baseline_state_id: u32,
    baseline_token_id: u32,
    candidate_state_id: u32,
    candidate_token_id: u32,
) -> Result<(), String> {
    let mut pending = VecDeque::from([(
        Some(baseline.start_state()),
        Some(candidate.start_state()),
        Vec::<i32>::new(),
    )]);
    let mut seen = BTreeSet::<(Option<u32>, Option<u32>)>::new();

    while let Some((baseline_state, candidate_state, word)) = pending.pop_front() {
        if !seen.insert((baseline_state, candidate_state)) {
            continue;
        }
        let baseline_accepts = accepts_final(
            baseline,
            baseline_state,
            baseline_state_id,
            baseline_token_id,
        );
        let candidate_accepts = accepts_final(
            candidate,
            candidate_state,
            candidate_state_id,
            candidate_token_id,
        );
        if baseline_accepts != candidate_accepts {
            return Err(format!(
                "baseline_tsid={baseline_state_id} baseline_token={baseline_token_id} candidate_tsid={candidate_state_id} candidate_token={candidate_token_id} word={word:?} baseline_accepts={baseline_accepts} candidate_accepts={candidate_accepts}",
            ));
        }

        let labels = outgoing_labels(baseline, baseline_state)
            .into_iter()
            .chain(outgoing_labels(candidate, candidate_state))
            .collect::<BTreeSet<_>>();
        for label in labels {
            let next_baseline = enabled_target(
                baseline,
                baseline_state,
                label,
                baseline_state_id,
                baseline_token_id,
            );
            let next_candidate = enabled_target(
                candidate,
                candidate_state,
                label,
                candidate_state_id,
                candidate_token_id,
            );
            if next_baseline.is_none() && next_candidate.is_none() {
                continue;
            }
            let mut next_word = word.clone();
            next_word.push(label);
            pending.push_back((next_baseline, next_candidate, next_word));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::paired_internal_classes;

    #[test]
    fn coordinate_pair_classes_merge_only_semantically_identical_coordinates() {
        let left = [0, 0, 1, u32::MAX, 1];
        let right = [4, 4, 9, 8, u32::MAX];
        assert_eq!(
            paired_internal_classes(&left, &right),
            vec![(0, 4), (1, 9), (1, u32::MAX), (u32::MAX, 8)],
        );
    }
}
