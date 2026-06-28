//! Exact semantic comparison for partition-local terminal DWA artifacts.
//!
//! A terminal DWA evaluates a terminal-label word by intersecting its transition
//! and final weights. Equivalent artifacts can distribute the same restriction
//! across different edges, so structural edge-weight equality is too strong.

use std::collections::{BTreeSet, VecDeque};

use crate::automata::weighted_u32::dwa::DWA;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::compiler::stages::id_map_and_terminal_dwa::types::LocalIdMapTerminalDwa;
use crate::ds::weight::Weight;

pub(crate) fn compare(
    baseline: &LocalIdMapTerminalDwa,
    candidate: &LocalIdMapTerminalDwa,
) -> Result<(), String> {
    let states = original_domain(
        &baseline.id_map.tokenizer_states.original_to_internal,
        &candidate.id_map.tokenizer_states.original_to_internal,
    );
    let tokens = original_domain(
        &baseline.id_map.vocab_tokens.original_to_internal,
        &candidate.id_map.vocab_tokens.original_to_internal,
    );
    for original_state in states {
        for original_token in tokens.iter().copied() {
            compare_boolean_dfas(
                &baseline.dwa,
                &baseline.id_map,
                &candidate.dwa,
                &candidate.id_map,
                original_state,
                original_token,
            )?;
        }
    }
    Ok(())
}

fn original_domain(left: &[u32], right: &[u32]) -> BTreeSet<u32> {
    left.iter()
        .enumerate()
        .chain(right.iter().enumerate())
        .filter_map(|(original, &internal)| (internal != u32::MAX).then_some(original as u32))
        .collect()
}

fn outgoing_labels(dwa: &DWA, state: Option<u32>) -> Vec<i32> {
    state
        .and_then(|state| dwa.states().get(state as usize))
        .map(|state| state.transitions.keys().copied().collect())
        .unwrap_or_default()
}

fn accepts_final(dwa: &DWA, map: &InternalIdMap, state: Option<u32>, s: u32, t: u32) -> bool {
    state
        .and_then(|id| dwa.states().get(id as usize))
        .and_then(|node| node.final_weight.as_ref())
        .is_some_and(|weight| contains(weight, map, s, t))
}

fn enabled_target(dwa: &DWA, map: &InternalIdMap, state: Option<u32>, label: i32, s: u32, t: u32) -> Option<u32> {
    let state = state?;
    let (target, weight) = dwa.states()[state as usize].transitions.get(&label)?;
    contains(weight, map, s, t).then_some(*target)
}

fn contains(weight: &Weight, map: &InternalIdMap, s: u32, t: u32) -> bool {
    let Some(&si) = map.tokenizer_states.original_to_internal.get(s as usize) else {
        return false;
    };
    let Some(&ti) = map.vocab_tokens.original_to_internal.get(t as usize) else {
        return false;
    };
    si != u32::MAX
        && ti != u32::MAX
        && (weight.is_full() || weight.tokens_for_tsid(si).contains(ti))
}

fn compare_boolean_dfas(
    baseline: &DWA,
    baseline_map: &InternalIdMap,
    candidate: &DWA,
    candidate_map: &InternalIdMap,
    original_state: u32,
    original_token: u32,
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
            baseline_map,
            baseline_state,
            original_state,
            original_token,
        );
        let candidate_accepts = accepts_final(
            candidate,
            candidate_map,
            candidate_state,
            original_state,
            original_token,
        );
        if baseline_accepts != candidate_accepts {
            return Err(format!(
                "state={original_state} token={original_token} word={word:?} baseline_accepts={baseline_accepts} candidate_accepts={candidate_accepts}",
            ));
        }

        let labels = outgoing_labels(baseline, baseline_state)
            .into_iter()
            .chain(outgoing_labels(candidate, candidate_state))
            .collect::<BTreeSet<_>>();
        for label in labels {
            let next_baseline = enabled_target(
                baseline,
                baseline_map,
                baseline_state,
                label,
                original_state,
                original_token,
            );
            let next_candidate = enabled_target(
                candidate,
                candidate_map,
                candidate_state,
                label,
                original_state,
                original_token,
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
