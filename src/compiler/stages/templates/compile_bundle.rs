//! Template bundle assembly into a weighted NWA.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::templates::compile_dfa::Templates;
use crate::ds::weight::Weight;

impl Templates {
    pub(crate) fn build_bundle(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> NWA {
        let mut raw_bundle = NWA::new(0, 0);
        let start = raw_bundle.add_state();
        raw_bundle.start_states.push(start);

        for (&terminal, weight) in terminal_weights {
            if weight.is_empty() {
                continue;
            }
            let Some(template) = self.by_terminal.get(&terminal) else {
                continue;
            };
            append_template(&mut raw_bundle, start, template, weight);
        }

        let bundle_dwa = minimize(
            &determinize(&raw_bundle).expect(
                "template bundle determinization failed during multi-template bundle assembly",
            ),
        );
        dwa_to_nwa(&bundle_dwa)
    }
}

fn append_template(nwa: &mut NWA, bundle_start: u32, dfa: &UnweightedDfa, entry_weight: &Weight) {
    if dfa.states.is_empty() {
        return;
    }

    let offset = nwa.states.len() as u32;
    for _state in &dfa.states {
        nwa.add_state();
    }

    nwa.add_epsilon(bundle_start, offset + dfa.start_state, entry_weight.clone());

    for (state_id, state) in dfa.states.iter().enumerate() {
        let from = offset + state_id as u32;
        if state.is_accepting {
            nwa.set_final_weight(from, Weight::all());
        }
        for (&label, &target) in &state.transitions {
            nwa.add_transition(from, label, offset + target, Weight::all());
        }
    }
}

fn dwa_to_nwa(dwa: &DWA) -> NWA {
    let mut nwa = NWA::new(0, 0);
    for _ in &dwa.states {
        nwa.add_state();
    }

    nwa.start_states.push(dwa.start_state);
    for (state_id, state) in dwa.states.iter().enumerate() {
        if let Some(final_weight) = state.final_weight.clone() {
            nwa.set_final_weight(state_id as u32, final_weight);
        }
        for (&label, (target, weight)) in &state.transitions {
            nwa.add_transition(state_id as u32, label, *target, weight.clone());
        }
    }

    nwa
}
