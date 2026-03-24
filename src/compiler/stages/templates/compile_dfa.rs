//! Template-DFA compilation from terminal characterizations.
//!
//! Builds each template as a lightweight NFA (fresh intermediate states per
//! path, epsilon-connected to NT nodes) and then determinizes + minimizes to
//! produce an acyclic unweighted DFA.

use std::collections::BTreeMap;

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::unweighted_u32::determinize::determinize;
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic as minimize_dfa;
use crate::automata::unweighted_u32::nfa::NFA;
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
        use rayon::prelude::*;

        let by_terminal: BTreeMap<TerminalID, UnweightedDfa> = characterizations
            .par_iter()
            .map(|(&terminal, characterization)| {
                let nfa = build_template_nfa(characterization);
                let dfa = minimize_dfa(&determinize(&nfa));
                (terminal, dfa)
            })
            .collect();

        Self { by_terminal }
    }
}

fn build_nonterminal_nodes(
    nfa: &mut NFA,
    characterization: &TerminalCharacterization,
) -> BTreeMap<u32, u32> {
    let mut nonterminal_nodes = BTreeMap::new();
    for &nonterminal in &characterization.all_nts {
        let state = nfa.add_state();
        nonterminal_nodes.insert(nonterminal, state);
    }
    nonterminal_nodes
}

fn append_default_pop_chain(nfa: &mut NFA, mut from: u32, pop_count: usize, target: u32) {
    for pop_index in 0..pop_count {
        let to = if pop_index == pop_count - 1 {
            target
        } else {
            nfa.add_state()
        };
        nfa.add_transition(from, DEFAULT_LABEL, to);
        from = to;
    }
}

fn add_positive_transition_chain(
    nfa: &mut NFA,
    from: u32,
    revealed_state: u32,
    pop_count: usize,
    target: u32,
) {
    let first_target = if pop_count == 0 {
        target
    } else {
        nfa.add_state()
    };
    nfa.add_transition(from, encode_positive_label(revealed_state), first_target);
    append_default_pop_chain(nfa, first_target, pop_count, target);
}

/// Build an unweighted NFA from a terminal characterization.
///
/// Each shift/reduce/escape/re-reduce path gets its own fresh intermediate
/// states, connected to the shared start state (via epsilon) and to shared
/// NT-node states.
fn build_template_nfa(characterization: &TerminalCharacterization) -> NFA {
    let mut nfa = NFA::new();
    let start = 0u32; // NFA::new() creates state 0 as start

    let nonterminal_nodes = build_nonterminal_nodes(&mut nfa, characterization);

    for &(initial_state, shift_state) in &characterization.shifts {
        let s0 = nfa.add_state();
        let s1 = nfa.add_state();
        let s2 = nfa.add_state();
        let s3 = nfa.add_state();

        nfa.add_epsilon(start, s0);
        nfa.add_transition(s0, encode_positive_label(initial_state), s1);
        nfa.add_transition(s1, encode_negative_label(initial_state), s2);
        nfa.add_transition(s2, encode_negative_label(shift_state), s3);
        nfa.set_accepting(s3);
    }

    for &(initial_state, pop_count, nonterminal) in &characterization.reduces {
        let Some(&target_nonterminal_state) = nonterminal_nodes.get(&nonterminal) else {
            continue;
        };

        let s0 = nfa.add_state();
        nfa.add_epsilon(start, s0);

        add_positive_transition_chain(
            &mut nfa,
            s0,
            initial_state,
            pop_count,
            target_nonterminal_state,
        );
    }

    for &(source_nonterminal, revealed_state, goto_state, shift_state) in &characterization.nt_escapes {
        let Some(&source_state) = nonterminal_nodes.get(&source_nonterminal) else {
            continue;
        };

        let s0 = nfa.add_state();
        let s1 = nfa.add_state();
        let s2 = nfa.add_state();
        let s3 = nfa.add_state();
        let s4 = nfa.add_state();

        nfa.add_epsilon(source_state, s0);
        nfa.add_transition(s0, encode_positive_label(revealed_state), s1);
        nfa.add_transition(s1, encode_negative_label(revealed_state), s2);
        nfa.add_transition(s2, encode_negative_label(goto_state), s3);
        nfa.add_transition(s3, encode_negative_label(shift_state), s4);
        nfa.set_accepting(s4);
    }

    for &(source_nonterminal, revealed_state, pop_count, target_nonterminal) in &characterization.nt_rereduces {
        let (Some(&source_state), Some(&target_state)) =
            (nonterminal_nodes.get(&source_nonterminal), nonterminal_nodes.get(&target_nonterminal))
        else {
            continue;
        };

        let s0 = nfa.add_state();
        nfa.add_epsilon(source_state, s0);
        add_positive_transition_chain(&mut nfa, s0, revealed_state, pop_count, target_state);
    }

    nfa
}
