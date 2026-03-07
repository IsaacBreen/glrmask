//! Template-DFA compilation from terminal characterizations.
// SEP1_MAP: This placeholder file corresponds directly to sep1's `precompute4/template_dfa.rs` template-automaton builder.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::compiler::glr::labels::{encode_negative_label, encode_positive_label, DEFAULT_LABEL};
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::templates::characterize::TerminalCharacterization;

#[derive(Debug, Clone, Default)]
pub struct Templates {
    pub by_terminal: BTreeMap<TerminalID, UnweightedDfa>,
}

impl Templates {
    pub(crate) fn from_characterizations(
        characterizations: &BTreeMap<TerminalID, TerminalCharacterization>,
    ) -> Self {
        let by_terminal = characterizations
            .iter()
            .map(|(&terminal, characterization)| (terminal, build_template_dfa(characterization)))
            .collect();
        Self { by_terminal }
    }
}

fn build_template_dfa(tc: &TerminalCharacterization) -> UnweightedDfa {
    let mut dfa = UnweightedDfa::new();
    let start_state = dfa.start_state;
    let mut nt_nodes = BTreeMap::new();

    for &nt in &tc.all_nts {
        let state = dfa.add_state();
        nt_nodes.insert(nt, state);
    }

    for &(initial_state, shift_state) in &tc.shifts {
        let accept = ensure_path(
            &mut dfa,
            start_state,
            &[
                encode_positive_label(initial_state),
                encode_negative_label(initial_state),
                encode_negative_label(shift_state),
            ],
        );
        dfa.set_accepting(accept, true);
    }

    for &(initial_state, pop_count, nt) in &tc.reduces {
        let Some(&target_nt) = nt_nodes.get(&nt) else {
            continue;
        };
        let mut path = Vec::with_capacity(1 + pop_count);
        path.push(encode_positive_label(initial_state));
        path.extend(std::iter::repeat(DEFAULT_LABEL).take(pop_count));
        ensure_path_to_existing(&mut dfa, start_state, &path, target_nt);
    }

    for &(src_nt, revealed_state, goto_state, shift_state) in &tc.nt_escapes {
        let Some(&src_state) = nt_nodes.get(&src_nt) else {
            continue;
        };
        let accept = ensure_path(
            &mut dfa,
            src_state,
            &[
                encode_positive_label(revealed_state),
                encode_negative_label(revealed_state),
                encode_negative_label(goto_state),
                encode_negative_label(shift_state),
            ],
        );
        dfa.set_accepting(accept, true);
    }

    for &(src_nt, revealed_state, pop_count, dst_nt) in &tc.nt_rereduces {
        let (Some(&src_state), Some(&dst_state)) = (nt_nodes.get(&src_nt), nt_nodes.get(&dst_nt)) else {
            continue;
        };
        let mut path = Vec::with_capacity(1 + pop_count);
        path.push(encode_positive_label(revealed_state));
        path.extend(std::iter::repeat(DEFAULT_LABEL).take(pop_count));
        ensure_path_to_existing(&mut dfa, src_state, &path, dst_state);
    }

    dfa
}

fn ensure_path(dfa: &mut UnweightedDfa, from: u32, labels: &[i32]) -> u32 {
    let mut state = from;
    for &label in labels {
        let next = if let Some(&next) = dfa.states[state as usize].transitions.get(&label) {
            next
        } else {
            let new_state = dfa.add_state();
            dfa.add_transition(state, label, new_state);
            new_state
        };
        state = next;
    }
    state
}

fn ensure_path_to_existing(dfa: &mut UnweightedDfa, from: u32, labels: &[i32], target: u32) {
    if labels.is_empty() {
        return;
    }

    let mut state = from;
    for &label in &labels[..labels.len() - 1] {
        state = if let Some(&next) = dfa.states[state as usize].transitions.get(&label) {
            next
        } else {
            let new_state = dfa.add_state();
            dfa.add_transition(state, label, new_state);
            new_state
        };
    }

    let final_label = labels[labels.len() - 1];
    if let Some(existing_target) = dfa.states[state as usize].transitions.get(&final_label).copied() {
        merge_states(dfa, existing_target, target);
    } else {
        dfa.add_transition(state, final_label, target);
    }
}

fn merge_states(dfa: &mut UnweightedDfa, keep: u32, merge: u32) {
    if keep == merge {
        return;
    }

    let merge_state = dfa.states[merge as usize].clone();
    if merge_state.is_accepting {
        dfa.set_accepting(keep, true);
    }

    for (label, target) in merge_state.transitions {
        if let Some(existing_target) = dfa.states[keep as usize].transitions.get(&label).copied() {
            merge_states(dfa, existing_target, target);
        } else {
            dfa.add_transition(keep, label, target);
        }
    }
}