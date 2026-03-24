//! Template-DFA compilation from terminal characterizations.
//!
//! Builds each template as a lightweight NFA (fresh intermediate states per
//! path, epsilon-connected to NT nodes) and then determinizes + minimizes to
//! produce an acyclic unweighted DFA.  The NFA approach mirrors sep1's
//! `build_nfa_from_terminal_characterization` and avoids the self-loops that
//! the old direct-DFA builder created when two reduces shared a label prefix
//! and target NT but had different pop counts.

use std::collections::BTreeMap;

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::unweighted_u32::determinize::determinize;
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic as minimize_dfa;
use crate::automata::unweighted_u32::nfa::NFA;
use crate::compiler::glr::labels::{encode_negative_label, encode_positive_label, DEFAULT_LABEL};
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::templates::characterize::TerminalCharacterization;

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct Templates {
    pub by_terminal: BTreeMap<TerminalID, UnweightedDfa>,
}

impl Templates {
    pub(crate) fn from_characterizations(
        characterizations: &BTreeMap<TerminalID, TerminalCharacterization>,
    ) -> Self {
        #[cfg(feature = "rayon")]
        {
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

        #[cfg(not(feature = "rayon"))]
        {
            let by_terminal = characterizations
                .iter()
                .map(|(&terminal, characterization)| {
                    let nfa = build_template_nfa(characterization);
                    let dfa = minimize_dfa(&determinize(&nfa));
                    (terminal, dfa)
                })
                .collect();
            Self { by_terminal }
        }
    }
}

// ---------------------------------------------------------------------------
// NFA construction (mirrors sep1's build_nfa_from_terminal_characterization)
// ---------------------------------------------------------------------------

/// Build an unweighted NFA from a terminal characterization.
///
/// Each shift/reduce/escape/re-reduce path gets its own fresh intermediate
/// states, connected to the shared start state (via epsilon) and to shared
/// NT-node states.  This avoids the self-loops that the old direct-DFA builder
/// introduced when two reduces shared a label prefix but had different pop
/// counts pointing at the same NT node.
fn build_template_nfa(tc: &TerminalCharacterization) -> NFA {
    let mut nfa = NFA::new();
    let start = 0u32; // NFA::new() creates state 0 as start

    // Shared node for each nonterminal.
    let mut nt_nodes = BTreeMap::new();
    for &nt in &tc.all_nts {
        let state = nfa.add_state();
        nt_nodes.insert(nt, state);
    }

    // -- Initial shifts -------------------------------------------------------
    // start --ε--> s0 --[+initial]--> s1 --[-initial]--> s2 --[-shift]--> s3 [accept]
    for &(initial_state, shift_state) in &tc.shifts {
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

    // -- Initial reduces ------------------------------------------------------
    // start --ε--> s0 --[+initial]--> (chain of DEFAULT pops) --> nt_node
    for &(initial_state, pop_count, nt) in &tc.reduces {
        let Some(&target_nt) = nt_nodes.get(&nt) else {
            continue;
        };

        let s0 = nfa.add_state();
        nfa.add_epsilon(start, s0);

        let first_target = if pop_count == 0 {
            target_nt
        } else {
            nfa.add_state()
        };
        nfa.add_transition(s0, encode_positive_label(initial_state), first_target);

        let mut from = first_target;
        for i in 0..pop_count {
            let to = if i == pop_count - 1 {
                target_nt
            } else {
                nfa.add_state()
            };
            nfa.add_transition(from, DEFAULT_LABEL, to);
            from = to;
        }
    }

    // -- NT escapes -----------------------------------------------------------
    // src_nt --ε--> s0 --[+rev]--> s1 --[-rev]--> s2 --[-goto]--> s3 --[-shift]--> s4 [accept]
    for &(src_nt, revealed_state, goto_state, shift_state) in &tc.nt_escapes {
        let Some(&src_state) = nt_nodes.get(&src_nt) else {
            continue;
        };

        let s0 = nfa.add_state();
        let s1 = nfa.add_state();
        let s2 = nfa.add_state();
        let s3 = nfa.add_state();
        let s4 = nfa.add_state();

        nfa.add_epsilon(src_state, s0);
        nfa.add_transition(s0, encode_positive_label(revealed_state), s1);
        nfa.add_transition(s1, encode_negative_label(revealed_state), s2);
        nfa.add_transition(s2, encode_negative_label(goto_state), s3);
        nfa.add_transition(s3, encode_negative_label(shift_state), s4);
        nfa.set_accepting(s4);
    }

    // -- NT re-reduces --------------------------------------------------------
    // src_nt --ε--> s0 --[+rev]--> (chain of DEFAULT pops) --> dst_nt
    for &(src_nt, revealed_state, pop_count, dst_nt) in &tc.nt_rereduces {
        let (Some(&src_state), Some(&dst_state)) =
            (nt_nodes.get(&src_nt), nt_nodes.get(&dst_nt))
        else {
            continue;
        };

        let s0 = nfa.add_state();
        nfa.add_epsilon(src_state, s0);

        let first_target = if pop_count == 0 {
            dst_state
        } else {
            nfa.add_state()
        };
        nfa.add_transition(s0, encode_positive_label(revealed_state), first_target);

        let mut from = first_target;
        for i in 0..pop_count {
            let to = if i == pop_count - 1 {
                dst_state
            } else {
                nfa.add_state()
            };
            nfa.add_transition(from, DEFAULT_LABEL, to);
            from = to;
        }
    }

    nfa
}
