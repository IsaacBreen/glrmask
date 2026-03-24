//! Template-DFA compilation from terminal characterizations.
//!
//! Builds each template as a lightweight NFA (fresh intermediate states per
//! path, epsilon-connected to NT nodes) and then determinizes + minimizes to
//! produce an acyclic unweighted DFA.

use std::collections::BTreeMap;
use std::time::Instant;

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::unweighted_u32::determinize::determinize;
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic as minimize_dfa;
use crate::automata::unweighted_u32::nfa::NFA;
use crate::compiler::glr::labels::{encode_negative_label, encode_positive_label, DEFAULT_LABEL};
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::templates::characterize::TerminalCharacterization;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TemplateCompileProfile {
    pub(crate) build_nfa_ms: f64,
    pub(crate) determinize_ms: f64,
    pub(crate) minimize_ms: f64,
    pub(crate) total_ms: f64,
    pub(crate) num_terminals: usize,
    pub(crate) unique_characterizations: usize,
    pub(crate) max_characterization_multiplicity: usize,
    pub(crate) total_nfa_states: usize,
    pub(crate) max_nfa_states: usize,
    pub(crate) total_nfa_transitions: usize,
    pub(crate) max_nfa_transitions: usize,
    pub(crate) total_dfa_states: usize,
    pub(crate) max_dfa_states: usize,
    pub(crate) total_dfa_transitions: usize,
    pub(crate) max_dfa_transitions: usize,
}

impl TemplateCompileProfile {
    pub(crate) fn avg_nfa_states(&self) -> f64 {
        average(self.total_nfa_states, self.num_terminals)
    }

    pub(crate) fn avg_nfa_transitions(&self) -> f64 {
        average(self.total_nfa_transitions, self.num_terminals)
    }

    pub(crate) fn avg_dfa_states(&self) -> f64 {
        average(self.total_dfa_states, self.num_terminals)
    }

    pub(crate) fn avg_dfa_transitions(&self) -> f64 {
        average(self.total_dfa_transitions, self.num_terminals)
    }

    fn observe_compilation(&mut self, sample: &TemplateCompilationSample) {
        self.build_nfa_ms += sample.build_nfa_ms;
        self.determinize_ms += sample.determinize_ms;
        self.minimize_ms += sample.minimize_ms;
        self.total_ms += sample.total_ms();
        self.num_terminals += 1;
        self.total_nfa_states += sample.nfa_states;
        self.max_nfa_states = self.max_nfa_states.max(sample.nfa_states);
        self.total_nfa_transitions += sample.nfa_transitions;
        self.max_nfa_transitions = self.max_nfa_transitions.max(sample.nfa_transitions);
        self.total_dfa_states += sample.dfa_states;
        self.max_dfa_states = self.max_dfa_states.max(sample.dfa_states);
        self.total_dfa_transitions += sample.dfa_transitions;
        self.max_dfa_transitions = self.max_dfa_transitions.max(sample.dfa_transitions);
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct TemplateCompilationSample {
    build_nfa_ms: f64,
    determinize_ms: f64,
    minimize_ms: f64,
    nfa_states: usize,
    nfa_transitions: usize,
    dfa_states: usize,
    dfa_transitions: usize,
}

impl TemplateCompilationSample {
    fn total_ms(&self) -> f64 {
        self.build_nfa_ms + self.determinize_ms + self.minimize_ms
    }
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn average(total: usize, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total as f64 / count as f64
    }
}

fn nfa_size(nfa: &NFA) -> (usize, usize) {
    let transitions = nfa
        .states
        .iter()
        .map(|state| {
            state
                .transitions
                .values()
                .map(Vec::len)
                .sum::<usize>()
                + state.epsilons.len()
        })
        .sum();
    (nfa.states.len(), transitions)
}

fn dfa_size(dfa: &UnweightedDfa) -> (usize, usize) {
    let transitions = dfa
        .states
        .iter()
        .map(|state| state.transitions.len())
        .sum();
    (dfa.states.len(), transitions)
}

fn compile_template_with_profile(
    characterization: &TerminalCharacterization,
) -> (UnweightedDfa, TemplateCompilationSample) {
    let build_nfa_started_at = Instant::now();
    let nfa = build_template_nfa(characterization);
    let build_nfa_ms = elapsed_ms(build_nfa_started_at);
    let (nfa_states, nfa_transitions) = nfa_size(&nfa);

    let determinize_started_at = Instant::now();
    let determinized = determinize(&nfa);
    let determinize_ms = elapsed_ms(determinize_started_at);

    let minimize_started_at = Instant::now();
    let dfa = minimize_dfa(&determinized);
    let minimize_ms = elapsed_ms(minimize_started_at);
    let (dfa_states, dfa_transitions) = dfa_size(&dfa);

    (
        dfa,
        TemplateCompilationSample {
            build_nfa_ms,
            determinize_ms,
            minimize_ms,
            nfa_states,
            nfa_transitions,
            dfa_states,
            dfa_transitions,
        },
    )
}

pub(crate) fn emit_template_profile_summary(
    characterize_ms: f64,
    profile: &TemplateCompileProfile,
) {
    eprintln!(
        "[glrmask/profile][templates] characterize_ms={:.3} compile_ms={:.3} build_nfa_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} num_terminals={} unique_characterizations={} max_characterization_multiplicity={} avg_nfa_states={:.1} max_nfa_states={} avg_nfa_transitions={:.1} max_nfa_transitions={} avg_dfa_states={:.1} max_dfa_states={} avg_dfa_transitions={:.1} max_dfa_transitions={} total_ms={:.3}",
        characterize_ms,
        profile.total_ms,
        profile.build_nfa_ms,
        profile.determinize_ms,
        profile.minimize_ms,
        profile.num_terminals,
        profile.unique_characterizations,
        profile.max_characterization_multiplicity,
        profile.avg_nfa_states(),
        profile.max_nfa_states,
        profile.avg_nfa_transitions(),
        profile.max_nfa_transitions,
        profile.avg_dfa_states(),
        profile.max_dfa_states,
        profile.avg_dfa_transitions(),
        profile.max_dfa_transitions,
        characterize_ms + profile.total_ms,
    );
}

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

    pub(crate) fn from_characterizations_profiled(
        characterizations: &BTreeMap<TerminalID, TerminalCharacterization>,
    ) -> (Self, TemplateCompileProfile) {
        use rayon::prelude::*;

        let mut multiplicities = BTreeMap::<&TerminalCharacterization, usize>::new();
        for characterization in characterizations.values() {
            *multiplicities.entry(characterization).or_default() += 1;
        }

        let compiled: Vec<(TerminalID, UnweightedDfa, TemplateCompilationSample)> =
            characterizations
                .par_iter()
                .map(|(&terminal, characterization)| {
                    let (dfa, sample) = compile_template_with_profile(characterization);
                    (terminal, dfa, sample)
                })
                .collect();

        let mut profile = TemplateCompileProfile {
            unique_characterizations: multiplicities.len(),
            max_characterization_multiplicity: multiplicities.values().copied().max().unwrap_or(0),
            ..TemplateCompileProfile::default()
        };

        let by_terminal = compiled
            .into_iter()
            .map(|(terminal, dfa, sample)| {
                profile.observe_compilation(&sample);
                (terminal, dfa)
            })
            .collect();

        (Self { by_terminal }, profile)
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
