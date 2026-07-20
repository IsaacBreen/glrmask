//! Collapse long same-terminal runs only after proving parser-effect stabilization.
//!
//! The final post-TI terminal automaton can contain vocabulary-length paths such
//! as `UNARY_OPERATOR^24`. Grammar follow admissibility alone cannot erase those
//! labels: each occurrence may still change the parser stack. For each terminal
//! with a long run, this pass compiles the exact parser stack-effect automata for
//! `T`, `T^2`, ... and finds a certified idempotent power `k` where
//! `E(T^k) = E(T^(k+1))`. Equality is a congruence under effect composition, so
//! all repetitions beyond `k` can be lowered to epsilon. The rewritten weighted
//! terminal NWA is then determinized and minimized before parser-DWA composition.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::time::Instant;

use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::equivalence::find_difference;
use crate::automata::weighted::nwa::{NWA, NWAState};
use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::stages::equiv_types::{InternalIdMap, MappedArtifact};
use crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalDwaFamilies;
use crate::compiler::stages::parser_dwa::build_parser_dwa_from_terminal_dwa_with_precomputed_templates;
use crate::compiler::stages::templates::Templates;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;
use crate::Vocab;

const MIN_CANDIDATE_RUN: u8 = 8;
const MAX_DISCOVERED_RUN: u8 = 8;
const MAX_CERTIFIED_CAP: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RunMode {
    None,
    Capped { terminal: TerminalID, count: u8 },
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TerminalRunCollapseProfile {
    pub candidate_terminals: usize,
    pub certified_terminals: usize,
    pub states_before: usize,
    pub states_after: usize,
    pub transitions_before: usize,
    pub transitions_after: usize,
    pub certificate_ms: f64,
    pub rewrite_ms: f64,
}

fn dwa_to_nwa(dwa: &DWA) -> NWA {
    let states = dwa
        .states()
        .iter()
        .map(|state| NWAState {
            final_weight: state.final_weight.clone(),
            transitions: state
                .transitions
                .iter()
                .map(|(&label, (target, weight))| (label, vec![(*target, weight.clone())]))
                .collect(),
            epsilons: Vec::new(),
        })
        .collect();
    NWA::from_parts(states, vec![dwa.start_state()])
}

fn as_nwa(automaton: &TerminalAutomaton) -> NWA {
    match automaton {
        TerminalAutomaton::Dwa(dwa) => dwa_to_nwa(dwa),
        TerminalAutomaton::TokenDeterministicNwa(nwa)
        | TerminalAutomaton::EpsilonNwa(nwa) => nwa.clone(),
    }
}

fn terminal_power(terminal: TerminalID, exponent: usize) -> TerminalAutomaton {
    let mut dwa = DWA::new(1, 0);
    let mut state = dwa.start_state();
    for _ in 0..exponent {
        let next = dwa.add_state();
        dwa.add_transition(state, terminal as i32, next, Weight::all());
        state = next;
    }
    dwa.set_final_weight(state, Weight::all());
    TerminalAutomaton::Dwa(dwa)
}

fn repeated_label_maxima(automaton: &TerminalAutomaton) -> BTreeMap<TerminalID, u8> {
    let nwa = as_nwa(automaton);
    let mut maxima = BTreeMap::<TerminalID, u8>::new();
    let mut queue = VecDeque::<(u32, Option<TerminalID>, u8)>::new();
    let mut seen = HashSet::<(u32, Option<TerminalID>, u8)>::new();
    for &start in nwa.start_states() {
        queue.push_back((start, None, 0));
    }

    while let Some((state_id, last, run)) = queue.pop_front() {
        if !seen.insert((state_id, last, run)) {
            continue;
        }
        let Some(state) = nwa.states().get(state_id as usize) else {
            continue;
        };
        for (target, weight) in &state.epsilons {
            if !weight.is_empty() {
                queue.push_back((*target, last, run));
            }
        }
        for (&label, targets) in &state.transitions {
            if label < 0 {
                continue;
            }
            let terminal = label as TerminalID;
            let next_run = if last == Some(terminal) {
                run.saturating_add(1).min(MAX_DISCOVERED_RUN)
            } else {
                1
            };
            maxima
                .entry(terminal)
                .and_modify(|existing| *existing = (*existing).max(next_run))
                .or_insert(next_run);
            for (target, weight) in targets {
                if !weight.is_empty() {
                    queue.push_back((*target, Some(terminal), next_run));
                }
            }
        }
    }
    maxima
}

fn parser_effect_for_power(
    terminal: TerminalID,
    exponent: usize,
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
    vocab: &Vocab,
    id_map: &InternalIdMap,
) -> DWA {
    let terminal_dwa = terminal_power(terminal, exponent);
    build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
        table,
        grammar,
        &terminal_dwa,
        templates,
        vocab,
        id_map,
        false,
    )
}

fn effects_equal(left: &DWA, right: &DWA, context: &str) -> bool {
    find_difference(left, right)
        .unwrap_or_else(|error| panic!("{context}: {error}"))
        .is_none()
}

fn certify_caps(
    candidates: &BTreeMap<TerminalID, u8>,
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
    vocab: &Vocab,
    id_map: &InternalIdMap,
) -> BTreeMap<TerminalID, u8> {
    let mut caps = BTreeMap::new();
    for (&terminal, &max_run) in candidates {
        if max_run < MIN_CANDIDATE_RUN {
            continue;
        }

        // Equality of consecutive powers is a monoid stabilization certificate:
        // if E(T^k) = E(T^(k+1)), right-composition by E(T) gives equality for
        // every larger power. Build one additional power as a defensive check
        // against mistakes in the effect construction or equivalence routine.
        let max_exponent = usize::from(max_run)
            .min(MAX_CERTIFIED_CAP + 2)
            .max(3);
        let mut previous = parser_effect_for_power(
            terminal, 1, table, grammar, templates, vocab, id_map,
        );
        let verify_stabilization = cfg!(debug_assertions)
            || std::env::var_os("GLRMASK_VERIFY_TERMINAL_RUN_CERTIFICATES").is_some();
        let mut pending_cap = None;

        for exponent in 2..=max_exponent {
            let current = parser_effect_for_power(
                terminal,
                exponent,
                table,
                grammar,
                templates,
                vocab,
                id_map,
            );
            let equal = effects_equal(
                &previous,
                &current,
                "terminal-run parser-effect certificate comparison failed",
            );

            if let Some(cap) = pending_cap.take() {
                if equal {
                    caps.insert(terminal, cap as u8);
                    break;
                }
                // Consecutive equality should be stable under composition. If
                // the defensive confirmation fails, reject that candidate and
                // continue searching rather than applying an unsound rewrite.
            } else {
                let cap = exponent - 1;
                if equal && cap <= MAX_CERTIFIED_CAP && usize::from(max_run) > cap {
                    if !verify_stabilization {
                        caps.insert(terminal, cap as u8);
                        break;
                    }
                    pending_cap = Some(cap);
                }
            }

            previous = current;
        }
    }
    caps
}

fn cap_runs(automaton: &TerminalAutomaton, caps: &BTreeMap<TerminalID, u8>) -> DWA {
    let source = as_nwa(automaton);
    let mut result = NWA::new(0, 0);
    let mut product_ids = HashMap::<(u32, RunMode), u32>::new();
    let mut queue = VecDeque::<(u32, RunMode)>::new();

    let get_or_create = |source_state: u32,
                         mode: RunMode,
                         result: &mut NWA,
                         product_ids: &mut HashMap<(u32, RunMode), u32>,
                         queue: &mut VecDeque<(u32, RunMode)>|
     -> u32 {
        if let Some(&existing) = product_ids.get(&(source_state, mode)) {
            return existing;
        }
        let id = result.add_state();
        if let Some(final_weight) = source.states()[source_state as usize].final_weight.clone() {
            result.set_final_weight(id, final_weight);
        }
        product_ids.insert((source_state, mode), id);
        queue.push_back((source_state, mode));
        id
    };

    let mut starts = Vec::new();
    for &source_start in source.start_states() {
        starts.push(get_or_create(
            source_start,
            RunMode::None,
            &mut result,
            &mut product_ids,
            &mut queue,
        ));
    }
    result.set_start_states(starts);

    while let Some((source_state, mode)) = queue.pop_front() {
        let from = product_ids[&(source_state, mode)];
        let state = &source.states()[source_state as usize];

        for (target, weight) in &state.epsilons {
            if weight.is_empty() {
                continue;
            }
            let to = get_or_create(
                *target,
                mode,
                &mut result,
                &mut product_ids,
                &mut queue,
            );
            result.add_epsilon(from, to, weight.clone());
        }

        for (&label, targets) in &state.transitions {
            for (target, weight) in targets {
                if weight.is_empty() {
                    continue;
                }
                let (next_mode, erase_label) = if label >= 0 {
                    let terminal = label as TerminalID;
                    if let Some(&cap) = caps.get(&terminal) {
                        match mode {
                            RunMode::Capped {
                                terminal: previous,
                                count,
                            } if previous == terminal => {
                                if count >= cap {
                                    (RunMode::Capped { terminal, count: cap }, true)
                                } else {
                                    (
                                        RunMode::Capped {
                                            terminal,
                                            count: count + 1,
                                        },
                                        false,
                                    )
                                }
                            }
                            _ => (RunMode::Capped { terminal, count: 1 }, false),
                        }
                    } else {
                        (RunMode::None, false)
                    }
                } else {
                    (RunMode::None, false)
                };
                let to = get_or_create(
                    *target,
                    next_mode,
                    &mut result,
                    &mut product_ids,
                    &mut queue,
                );
                if erase_label {
                    result.add_epsilon(from, to, weight.clone());
                } else {
                    result.add_transition(from, label, to, weight.clone());
                }
            }
        }
    }

    let determinized = determinize(&result)
        .expect("certified terminal-run rewrite determinization failed");
    minimize(&determinized)
}

pub(crate) fn collapse_certified_terminal_runs(
    families: &mut TerminalDwaFamilies,
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
    vocab: &Vocab,
) -> TerminalRunCollapseProfile {
    if std::env::var_os("GLRMASK_DISABLE_CERTIFIED_TERMINAL_RUN_COLLAPSE").is_some() {
        return TerminalRunCollapseProfile::default();
    }
    let Some(l2p) = families.l2p.as_ref() else {
        return TerminalRunCollapseProfile::default();
    };
    let before = l2p.artifact().stats();
    let candidates = repeated_label_maxima(l2p.artifact())
        .into_iter()
        .filter(|(_, max_run)| *max_run >= MIN_CANDIDATE_RUN)
        .collect::<BTreeMap<_, _>>();
    if candidates.is_empty() {
        return TerminalRunCollapseProfile {
            states_before: before.states,
            states_after: before.states,
            transitions_before: before.transitions,
            transitions_after: before.transitions,
            ..TerminalRunCollapseProfile::default()
        };
    }

    let certificate_started_at = Instant::now();
    let caps = certify_caps(
        &candidates,
        table,
        grammar,
        templates,
        vocab,
        l2p.id_map(),
    );
    let certificate_ms = certificate_started_at.elapsed().as_secs_f64() * 1000.0;
    if caps.is_empty() {
        return TerminalRunCollapseProfile {
            candidate_terminals: candidates.len(),
            states_before: before.states,
            states_after: before.states,
            transitions_before: before.transitions,
            transitions_after: before.transitions,
            certificate_ms,
            ..TerminalRunCollapseProfile::default()
        };
    }

    let rewrite_started_at = Instant::now();
    let family = families.l2p.take().expect("L2P family checked above");
    let (automaton, id_map) = family.into_parts();
    let rewritten = cap_runs(&automaton, &caps);
    let after = rewritten.stats();
    let after_states = after.states;
    let after_transitions = after.transitions;
    families.l2p = Some(MappedArtifact::new(
        TerminalAutomaton::Dwa(rewritten),
        id_map,
    ));
    let rewrite_ms = rewrite_started_at.elapsed().as_secs_f64() * 1000.0;

    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_DEBUG_CERTIFIED_TERMINAL_RUNS").is_some()
    {
        eprintln!(
            "[glrmask/profile][certified_terminal_runs] candidates={:?} caps={:?} states={}->{} transitions={}->{} certificate_ms={:.3} rewrite_ms={:.3}",
            candidates,
            caps,
            before.states,
            after_states,
            before.transitions,
            after_transitions,
            certificate_ms,
            rewrite_ms,
        );
    }

    TerminalRunCollapseProfile {
        candidate_terminals: candidates.len(),
        certified_terminals: caps.len(),
        states_before: before.states,
        states_after: after_states,
        transitions_before: before.transitions,
        transitions_after: after_transitions,
        certificate_ms,
        rewrite_ms,
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::stages::equiv_types::ManyToOneIdMap;
    use crate::compiler::stages::templates::characterize::characterize_terminals_profiled;
    use crate::grammar::ast::lower;
    use crate::grammar::glrm::from_glrm;

    fn word_automaton(labels: &[i32], epsilon_after: Option<usize>) -> TerminalAutomaton {
        let mut nwa = NWA::new(0, 0);
        let start = nwa.add_state();
        nwa.set_start_states(vec![start]);
        let mut state = start;
        for (index, &label) in labels.iter().enumerate() {
            if epsilon_after == Some(index) {
                let next = nwa.add_state();
                nwa.add_epsilon(state, next, Weight::all());
                state = next;
            }
            let next = nwa.add_state();
            nwa.add_transition(state, label, next, Weight::all());
            state = next;
        }
        nwa.set_final_weight(state, Weight::all());
        TerminalAutomaton::EpsilonNwa(nwa)
    }

    fn identity_id_map() -> InternalIdMap {
        InternalIdMap {
            tokenizer_states: ManyToOneIdMap {
                original_to_internal: vec![0],
                internal_to_originals: vec![vec![0]],
                representative_original_ids: vec![0],
            },
            vocab_tokens: ManyToOneIdMap {
                original_to_internal: vec![0],
                internal_to_originals: vec![vec![0]],
                representative_original_ids: vec![0],
            },
            deferred_vocab_singleton_original_ids: None,
        }
    }

    #[test]
    fn bounded_repeat_certifies_only_after_the_language_is_exhausted() {
        let named = from_glrm(
            r#"
start start;
t A ::= "a";
t B ::= "b";
nt item ::= A | B;
nt start ::= item item? item?;
"#,
        )
        .unwrap();
        let grammar_def = lower(&named).unwrap();
        let grammar = AnalyzedGrammar::from_grammar_def(&grammar_def);
        let table = GLRTable::build(&grammar);
        let (characterizations, _) = characterize_terminals_profiled(&table, &grammar);
        let templates = Templates::from_characterizations(&characterizations);
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let terminal_a = (0..grammar.num_terminals)
            .find(|&terminal| grammar.terminal_display_name(terminal) == "A")
            .expect("terminal A");

        let caps = certify_caps(
            &BTreeMap::from([(terminal_a, 8)]),
            &table,
            &grammar,
            &templates,
            &vocab,
            &identity_id_map(),
        );

        assert_eq!(caps.get(&terminal_a), Some(&4));
    }

    #[test]
    fn cap_runs_erases_only_excess_repetitions() {
        let source = word_automaton(&[7, 7, 7, 7, 8, 7, 7, 7], None);
        let rewritten = cap_runs(&source, &BTreeMap::from([(7, 2)]));

        assert!(!rewritten.eval_word(&[7, 7, 8, 7, 7]).is_empty());
        assert!(rewritten.eval_word(&[7, 7, 7, 8, 7, 7]).is_empty());
        assert!(rewritten.eval_word(&[7, 7, 8, 7, 7, 7]).is_empty());
    }

    #[test]
    fn epsilon_edges_do_not_reset_a_certified_run() {
        let source = word_automaton(&[7, 7, 7], Some(1));
        let rewritten = cap_runs(&source, &BTreeMap::from([(7, 2)]));

        assert!(!rewritten.eval_word(&[7, 7]).is_empty());
        assert!(rewritten.eval_word(&[7, 7, 7]).is_empty());
    }

    #[test]
    fn a_different_terminal_resets_the_run_counter() {
        let source = word_automaton(&[7, 7, 7, 8, 7, 7, 7], None);
        let rewritten = cap_runs(&source, &BTreeMap::from([(7, 2)]));

        assert!(!rewritten.eval_word(&[7, 7, 8, 7, 7]).is_empty());
    }
}
