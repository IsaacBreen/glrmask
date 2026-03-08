//! Resolve negative parser-state labels in weighted NWAs.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::HashSet;

use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::labels::{encode_negative_label, DEFAULT_LABEL};
use crate::ds::weight::Weight;

fn is_negative_label(label: i32) -> bool {
    label < 0 && label != DEFAULT_LABEL
}

pub(crate) fn compute_cancellations(nwa: &NWA) -> Vec<(u32, u32, Weight)> {
    let mut out = Vec::new();

    for (state_id, state) in nwa.states.iter().enumerate() {
        for (&label, targets) in &state.transitions {
            if label < 0 || label == DEFAULT_LABEL {
                continue;
            }

            let negative_label = encode_negative_label(label as u32);
            for (mid, first_weight) in targets {
                let Some(mid_state) = nwa.states.get(*mid as usize) else {
                    continue;
                };
                let Some(cancel_targets) = mid_state.transitions.get(&negative_label) else {
                    continue;
                };
                for (dst, second_weight) in cancel_targets {
                    out.push((state_id as u32, *dst, first_weight.intersection(second_weight)));
                }
            }
        }
    }

    out
}

pub(crate) fn apply_cancellations(nwa: &mut NWA) {
    for (from, to, weight) in compute_cancellations(nwa) {
        if !weight.is_empty() {
            nwa.add_epsilon(from, to, weight);
        }
    }
}

pub(crate) fn apply_finality_fixpoint(nwa: &mut NWA) {
    let mut changed = true;
    while changed {
        changed = false;

        for state_id in 0..nwa.states.len() {
            let mut additions = Vec::new();

            for (dst, weight) in nwa.states[state_id].epsilons.clone() {
                if let Some(final_weight) = nwa.states[dst as usize].final_weight.as_ref() {
                    additions.push(weight.intersection(final_weight));
                }
            }
            if let Some(default_targets) = nwa.states[state_id].transitions.get(&DEFAULT_LABEL).cloned() {
                for (dst, weight) in default_targets {
                    if let Some(final_weight) = nwa.states[dst as usize].final_weight.as_ref() {
                        additions.push(weight.intersection(final_weight));
                    }
                }
            }
            // Negative transitions represent GSS pushes.  When a negative
            // transition leads to a state with a final_weight, the source
            // state is also final — the push modifies the GSS but doesn't
            // gate token eligibility.  Without this propagation the
            // final_weight is stranded behind the negative edge and lost
            // when `remove_negative_transitions` runs.
            for (&label, targets) in nwa.states[state_id].transitions.clone().iter() {
                if !is_negative_label(label) {
                    continue;
                }
                for (dst, weight) in targets {
                    if let Some(final_weight) = nwa.states[*dst as usize].final_weight.as_ref() {
                        additions.push(weight.intersection(final_weight));
                    }
                }
            }

            for addition in additions {
                if addition.is_empty() {
                    continue;
                }
                let updated = match &nwa.states[state_id].final_weight {
                    Some(existing) => existing.union(&addition),
                    None => addition,
                };
                let was_same = nwa.states[state_id]
                    .final_weight
                    .as_ref()
                    .map(|existing| existing == &updated)
                    .unwrap_or(false);
                if !was_same {
                    nwa.states[state_id].final_weight = Some(updated);
                    changed = true;
                }
            }
        }
    }
}

pub(crate) fn remove_negative_transitions(nwa: &mut NWA) {
    for state in &mut nwa.states {
        state.transitions.retain(|label, _| !is_negative_label(*label));
    }
}

pub(crate) fn remove_redundant_default_transitions(nwa: &mut NWA) {
    let n = nwa.states.len();
    let mut is_terminal = vec![false; n];

    for state_id in 0..n {
        let state = &nwa.states[state_id];
        let has_non_default = state.transitions.iter().any(|(label, targets)| {
            *label != DEFAULT_LABEL && !targets.is_empty()
        });
        let is_final = state.final_weight.as_ref().map(|weight| !weight.is_empty()).unwrap_or(false);
        if !has_non_default && state.epsilons.is_empty() && is_final {
            is_terminal[state_id] = true;
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for state_id in 0..n {
            if is_terminal[state_id] {
                continue;
            }
            let state = &nwa.states[state_id];
            let has_non_default = state.transitions.iter().any(|(label, targets)| {
                *label != DEFAULT_LABEL && !targets.is_empty()
            });
            let is_final = state.final_weight.as_ref().map(|weight| !weight.is_empty()).unwrap_or(false);
            if has_non_default || !state.epsilons.is_empty() || !is_final {
                continue;
            }
            let default_targets_terminal = state
                .transitions
                .get(&DEFAULT_LABEL)
                .map(|targets| targets.iter().all(|(target, _)| is_terminal[*target as usize]))
                .unwrap_or(true);
            if default_targets_terminal {
                is_terminal[state_id] = true;
                changed = true;
            }
        }
    }

    for state in &mut nwa.states {
        if let Some(targets) = state.transitions.get_mut(&DEFAULT_LABEL) {
            targets.retain(|(target, _)| !is_terminal[*target as usize]);
        }
        state.transitions.retain(|_, targets| !targets.is_empty());
    }
}

pub(crate) fn resolve_negative_codes_in_nwa(nwa: &mut NWA) {
    apply_cancellations(nwa);
    apply_finality_fixpoint(nwa);
    remove_negative_transitions(nwa);
    remove_redundant_default_transitions(nwa);
}
