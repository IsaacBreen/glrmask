#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::resolve_negatives::resolve_negative_codes_in_nwa;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::templates::Templates;
use crate::compiler::stages::templates::characterize::characterize_terminals;
use crate::compiler::terminal_dwa::build_terminal_dwa;
use crate::ds::weight::Weight;

type BundleEntry = BTreeMap<TerminalID, Weight>;

#[derive(Debug, Clone)]
struct BundleEdge {
    target_state: u32,
    bundle_id: usize,
}

#[derive(Debug, Clone)]
struct TerminalDag {
    start_state: u32,
    final_weights: Vec<Option<Weight>>,
    outgoing_edges: Vec<Vec<BundleEdge>>,
}

#[derive(Debug, Clone)]
struct TerminalDagDecomposition {
    dag: TerminalDag,
    bundle_entries: Vec<BundleEntry>,
}

fn append_dwa(into: &mut DWA, other: &DWA) -> u32 {
    let offset = into.states.len() as u32;
    for _ in &other.states {
        into.add_state();
    }

    for (state_id, state) in other.states.iter().enumerate() {
        let dst_state = offset + state_id as u32;
        if let Some(final_weight) = state.final_weight.clone() {
            into.set_final_weight(dst_state, final_weight);
        }
        for (&label, (target, weight)) in &state.transitions {
            into.add_transition(dst_state, label, offset + *target, weight.clone());
        }
    }

    offset + other.start_state
}

fn append_nwa(into: &mut NWA, other: &NWA) {
    let offset = into.states.len() as u32;
    for _ in &other.states {
        into.add_state();
    }

    for (state_id, state) in other.states.iter().enumerate() {
        let dst_state = offset + state_id as u32;
        if let Some(final_weight) = state.final_weight.clone() {
            into.set_final_weight(dst_state, final_weight);
        }
        for (&label, targets) in &state.transitions {
            for (target, weight) in targets {
                into.add_transition(dst_state, label, offset + *target, weight.clone());
            }
        }
        for (target, weight) in &state.epsilons {
            into.add_epsilon(dst_state, offset + *target, weight.clone());
        }
    }

    into.start_states
        .extend(other.start_states.iter().map(|state| offset + *state));
}

fn accepting_nwa(final_weight: &Weight) -> Option<NWA> {
    if final_weight.is_empty() {
        return None;
    }

    let mut nwa = NWA::new(0, 0);
    let start = nwa.add_state();
    nwa.start_states.push(start);
    nwa.set_final_weight(start, final_weight.clone());
    Some(nwa)
}

fn template_dfa_to_nwa(template: &UnweightedDfa) -> NWA {
    let mut nwa = NWA::new(0, 0);
    for _ in &template.states {
        nwa.add_state();
    }

    nwa.start_states.push(template.start_state);
    for (state_id, state) in template.states.iter().enumerate() {
        if state.is_accepting {
            nwa.set_final_weight(state_id as u32, Weight::all());
        }
        for (&label, target) in &state.transitions {
            nwa.add_transition(state_id as u32, label, *target, Weight::all());
        }
    }

    nwa
}

fn build_bundle_nwa(
    bundle: &BundleEntry,
    templates: &BTreeMap<TerminalID, UnweightedDfa>,
) -> NWA {
    let mut combined = NWA::new(0, 0);
    let combined_start = combined.add_state();
    combined.start_states.push(combined_start);

    for (&terminal, weight) in bundle {
        if weight.is_empty() {
            continue;
        }

        let template = templates
            .get(&terminal)
            .expect("missing template for terminal bundle entry");
        let template_nwa = template_dfa_to_nwa(template);
        let body = combined.append_with_body(&template_nwa);
        for &body_start in &body.start_states {
            combined.add_epsilon(combined_start, body_start, weight.clone());
        }
    }

    combined
}

fn build_bundle_nwas(
    bundle_entries: &[BundleEntry],
    templates: &BTreeMap<TerminalID, UnweightedDfa>,
) -> Vec<NWA> {
    bundle_entries
        .iter()
        .map(|bundle| build_bundle_nwa(bundle, templates))
        .collect()
}

fn group_terminal_edges_by_target(
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    state_id: u32,
) -> BTreeMap<u32, BundleEntry> {
    let Some(state) = terminal_dwa.states.get(state_id as usize) else {
        return BTreeMap::new();
    };

    let mut groups = BTreeMap::<u32, BundleEntry>::new();
    for (&label, (target, weight)) in &state.transitions {
        if label < 0 || label as u32 >= grammar.num_terminals {
            continue;
        }

        groups
            .entry(*target)
            .or_default()
            .entry(label as TerminalID)
            .and_modify(|existing| *existing = existing.union(weight))
            .or_insert_with(|| weight.clone());
    }

    groups
}

fn decompose_terminal_dwa(
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
) -> TerminalDagDecomposition {
    let mut bundle_entries = Vec::<BundleEntry>::new();
    let mut outgoing_edges = vec![Vec::<BundleEdge>::new(); terminal_dwa.states.len()];

    for state_id in 0..terminal_dwa.states.len() as u32 {
        let grouped_edges = group_terminal_edges_by_target(terminal_dwa, grammar, state_id);
        for (target_state, bundle_entry) in grouped_edges {
            let bundle_id = bundle_entries.len();
            bundle_entries.push(bundle_entry);
            outgoing_edges[state_id as usize].push(BundleEdge {
                target_state,
                bundle_id,
            });
        }
    }

    TerminalDagDecomposition {
        dag: TerminalDag {
            start_state: terminal_dwa.start_state,
            final_weights: terminal_dwa
                .states
                .iter()
                .map(|state| state.final_weight.clone())
                .collect(),
            outgoing_edges,
        },
        bundle_entries,
    }
}

fn concatenate_nwas(left: &NWA, right: &NWA) -> Option<NWA> {
    if left.start_states.is_empty() || right.start_states.is_empty() {
        return None;
    }

    let mut arena = NWA::new(0, 0);
    let right_body = arena.append_with_body(right);
    let left_body = arena.concatenate_in_place(left, &right_body);
    arena.start_states = left_body.start_states.clone();
    Some(arena)
}

fn union_optional_nwa(acc: &mut Option<NWA>, next: NWA) {
    match acc {
        Some(existing) => append_nwa(existing, &next),
        None => *acc = Some(next),
    }
}

fn compose_terminal_dag_state(
    state_id: u32,
    dag: &TerminalDag,
    bundle_nwas: &[NWA],
    memo: &mut BTreeMap<u32, Option<NWA>>,
) -> Option<NWA> {
    if let Some(cached) = memo.get(&state_id) {
        return cached.clone();
    }

    let mut composed: Option<NWA> = dag
        .final_weights
        .get(state_id as usize)
        .and_then(|weight| weight.as_ref())
        .and_then(accepting_nwa);

    let Some(edges) = dag.outgoing_edges.get(state_id as usize) else {
        memo.insert(state_id, composed.clone());
        return composed;
    };

    for edge in edges {
        let Some(continuation) = compose_terminal_dag_state(edge.target_state, dag, bundle_nwas, memo)
        else {
            continue;
        };
        let Some(branch_bundle) = bundle_nwas.get(edge.bundle_id) else {
            continue;
        };
        let Some(branch_with_continuation) = concatenate_nwas(branch_bundle, &continuation) else {
            continue;
        };
        union_optional_nwa(&mut composed, branch_with_continuation);
    }

    memo.insert(state_id, composed.clone());
    composed
}

fn wrap_parser_dwa_with_tokenizer_seeds(
    table: &GLRTable,
    tokenizer: &Tokenizer,
    core_dwa: &DWA,
) -> DWA {
    let core_is_dead = core_dwa
        .states
        .get(core_dwa.start_state as usize)
        .map(|state| state.final_weight.is_none() && state.transitions.is_empty())
        .unwrap_or(true);
    if core_is_dead {
        return DWA::new(0, 0);
    }

    let mut wrapped = DWA::new(0, 0);
    let core_start = append_dwa(&mut wrapped, core_dwa);

    for tokenizer_state in 0..tokenizer.num_states() {
        let tsid_label = table.num_terminals as i32 + tokenizer_state as i32;
        wrapped.add_transition(wrapped.start_state, tsid_label, core_start, Weight::all());
    }

    wrapped
}

pub fn build_parser_dwa(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
) -> DWA {
    let terminal_dwa = build_terminal_dwa(grammar, tokenizer, vocab, id_map);
    build_parser_dwa_from_terminal_dwa(table, grammar, tokenizer, &terminal_dwa)
}

pub(crate) fn build_parser_dwa_from_terminal_dwa(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    terminal_dwa: &DWA,
) -> DWA {
    let characterizations = characterize_terminals(table, grammar);
    let templates = Templates::from_characterizations(&characterizations);
    let decomposition = decompose_terminal_dwa(terminal_dwa, grammar);
    let bundle_nwas = build_bundle_nwas(&decomposition.bundle_entries, &templates.by_terminal);

    let mut memo = BTreeMap::new();
    let Some(mut parser_nwa) = compose_terminal_dag_state(
        decomposition.dag.start_state,
        &decomposition.dag,
        &bundle_nwas,
        &mut memo,
    ) else {
        return DWA::new(0, 0);
    };

    resolve_negative_codes_in_nwa(&mut parser_nwa);
    parser_nwa.subtract_final_weights_from_outgoing();

    let mut core_dwa = minimize(
        &determinize(&parser_nwa)
            .expect("parser NWA determinization failed despite acyclic terminal/template composition"),
    );

    wrap_parser_dwa_with_tokenizer_seeds(table, tokenizer, &core_dwa)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::grammar::model::GrammarDef;
    use crate::compiler::grammar::model::tests::*;

    fn make_vocab_and_preprocessing(gdef: &GrammarDef) -> (Vocab, Tokenizer, InternalIdMap) {
        let tok = crate::compiler::compile::build_tokenizer(gdef);

        let mut entries: Vec<(u32, Vec<u8>)> = Vec::new();
        for (i, td) in gdef.terminals.iter().enumerate() {
            entries.push((i as u32, td.name().as_bytes().to_vec()));
        }
        let vocab = Vocab::new(entries, None);
        let id_map = InternalIdMap::build(&tok, &vocab);
        (vocab, tok, id_map)
    }

    #[test]
    fn test_build_parser_dwa_simple() {
        let gdef = simple_ab_grammar();
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);
        let (vocab, tok, vp) = make_vocab_and_preprocessing(&gdef);

        let dwa = build_parser_dwa(&table, &gg, &tok, &vocab, &vp);
        assert!(dwa.num_states() > 0);
    }

    #[test]
    fn test_build_parser_dwa_choice() {
        let gdef = choice_grammar();
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);
        let (vocab, tok, vp) = make_vocab_and_preprocessing(&gdef);

        let dwa = build_parser_dwa(&table, &gg, &tok, &vocab, &vp);
        assert!(dwa.num_states() > 0);
    }
}
