//! Template bundle assembly into a weighted NWA.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, HashMap};

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::unweighted_u32::nfa::NFA as UnweightedNfa;
use crate::automata::unweighted_u32::determinize::determinize as unweighted_determinize;
use crate::automata::unweighted_u32::minimize::minimize as unweighted_minimize;
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
        if terminal_weights.len() == 1 {
            let mut bundle_nwa = NWA::new(0, 0);
            let start = bundle_nwa.add_state();
            bundle_nwa.start_states.push(start);

            let (&terminal, weight) = terminal_weights.iter().next().expect("single-entry bundle");
            if !weight.is_empty() {
                if let Some(template) = self.by_terminal.get(&terminal) {
                    append_template(&mut bundle_nwa, start, template, weight);
                }
            }

            // Determinize + minimize single-entry bundle too.
            let bundle_dwa = determinize(&bundle_nwa).expect("single bundle determinize failed");
            let minimized = minimize(&bundle_dwa);
            return dwa_to_nwa(&minimized);
        }

        let profile_enabled = std::env::var_os("GLRMASK_PROFILE_PARSER_DWA").is_some();

        // Group entries by weight so we can merge templates that share weights
        // using fast unweighted DFA operations.
        let mut weight_groups: HashMap<&Weight, Vec<TerminalID>> = HashMap::new();
        for (&terminal, weight) in terminal_weights {
            if weight.is_empty() {
                continue;
            }
            if self.by_terminal.contains_key(&terminal) {
                weight_groups.entry(weight).or_default().push(terminal);
            }
        }

        let num_groups = weight_groups.len();

        // Build a merged unweighted DFA for each weight group.
        let unweighted_started = std::time::Instant::now();
        let mut group_dfas: Vec<(&Weight, UnweightedDfa)> = Vec::with_capacity(num_groups);
        for (weight, terminals) in &weight_groups {
            if terminals.len() == 1 {
                // Single terminal in group — use template DFA directly.
                if let Some(template) = self.by_terminal.get(&terminals[0]) {
                    group_dfas.push((weight, template.clone()));
                }
            } else {
                // Multiple terminals sharing a weight — union their DFAs via NFA.
                let merged = union_unweighted_dfas(
                    terminals.iter().filter_map(|t| self.by_terminal.get(t)),
                );
                group_dfas.push((weight, merged));
            }
        }
        let unweighted_ms = unweighted_started.elapsed().as_secs_f64() * 1000.0;

        // Build the weighted NWA with one epsilon per weight group.
        let mut bundle_nwa = NWA::new(0, 0);
        let start = bundle_nwa.add_state();
        bundle_nwa.start_states.push(start);

        for (weight, dfa) in &group_dfas {
            append_template(&mut bundle_nwa, start, dfa, weight);
        }

        let nwa_states = bundle_nwa.states.len();

        // Weighted determinize + minimize to ensure each bundle is minimal.
        let det_started = std::time::Instant::now();
        let bundle_dwa = determinize(&bundle_nwa).expect("bundle determinize failed");
        let det_ms = det_started.elapsed().as_secs_f64() * 1000.0;
        let dwa_states = bundle_dwa.states.len();

        let min_started = std::time::Instant::now();
        let minimized = minimize(&bundle_dwa);
        let min_ms = min_started.elapsed().as_secs_f64() * 1000.0;
        let min_states = minimized.states.len();

        let result = dwa_to_nwa(&minimized);

        if profile_enabled {
            eprintln!(
                "[glrmask/profile][bundle_detmin] entries={} groups={} nwa_states={} dwa_states={} min_states={} unweighted_ms={:.1} det_ms={:.1} min_ms={:.1}",
                terminal_weights.len(), num_groups, nwa_states, dwa_states, min_states, unweighted_ms, det_ms, min_ms,
            );
        }

        result
    }
}

/// Union multiple unweighted DFAs into one DFA via NFA union + determinize + minimize.
fn union_unweighted_dfas<'a>(dfas: impl Iterator<Item = &'a UnweightedDfa>) -> UnweightedDfa {
    let mut nfa = UnweightedNfa::new_empty();
    let shared_start = nfa.add_state();
    nfa.start_states.push(shared_start);

    for dfa in dfas {
        if dfa.states.is_empty() {
            continue;
        }
        let offset = nfa.states.len() as u32;
        for _ in &dfa.states {
            nfa.add_state();
        }
        // Epsilon from shared start to this DFA's start.
        nfa.add_epsilon(shared_start, offset + dfa.start_state);
        for (state_id, state) in dfa.states.iter().enumerate() {
            let from = offset + state_id as u32;
            if state.is_accepting {
                nfa.set_accepting(from);
            }
            for (&label, &target) in &state.transitions {
                nfa.add_transition(from, label, offset + target);
            }
        }
    }

    let det = unweighted_determinize(&nfa);
    unweighted_minimize(&det)
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
