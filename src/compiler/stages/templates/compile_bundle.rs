//! Template bundle assembly into a weighted NWA.
// SEP1_MAP: sep1 performs the nearest work inside `precompute4/parser_dwa.rs` when it assembles parser-NWA pieces after template construction; glrmask keeps that boundary as its own placeholder file.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::weighted::determinize::determinize;
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
        let mut nwa = NWA::new(0, 0);
        nwa.states.clear();
        nwa.start_states.clear();
        let mut appended = 0usize;

        for (&terminal, weight) in terminal_weights {
            if weight.is_empty() {
                continue;
            }
            let Some(template) = self.by_terminal.get(&terminal) else {
                continue;
            };
            append_template(&mut nwa, template, weight);
            appended += 1;
        }

        if appended <= 1 {
            return nwa;
        }

        let bundle_dwa = minimize(
            &determinize(&nwa).expect(
                "template bundle determinization failed during multi-template bundle assembly",
            ),
        );

        let mut rebuilt = NWA::new(0, 0);
        rebuilt.states.clear();
        rebuilt.start_states.push(bundle_dwa.start_state);
        for _ in &bundle_dwa.states {
            rebuilt.add_state();
        }
        for (state_id, state) in bundle_dwa.states.iter().enumerate() {
            let from = state_id as u32;
            if let Some(final_weight) = state.final_weight.clone() {
                rebuilt.set_final_weight(from, final_weight);
            }
            for (&label, (target, weight)) in &state.transitions {
                rebuilt.add_transition(from, label, *target, weight.clone());
            }
        }

        rebuilt
    }
}

fn append_template(nwa: &mut NWA, dfa: &UnweightedDfa, final_weight: &Weight) {
    if dfa.states.is_empty() {
        return;
    }

    let offset = nwa.states.len() as u32;
    for _state in &dfa.states {
        nwa.add_state();
    }

    nwa.start_states.push(offset + dfa.start_state);

    for (state_id, state) in dfa.states.iter().enumerate() {
        let from = offset + state_id as u32;
        if state.is_accepting {
            nwa.set_final_weight(from, final_weight.clone());
        }
        for (&label, &target) in &state.transitions {
            nwa.add_transition(from, label, offset + target, Weight::all());
        }
    }
}