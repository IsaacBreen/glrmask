#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
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
use crate::compiler::terminal_dwa::build_terminal_dwa_with_prefix_weights;
use crate::ds::weight::Weight;

type Bundle = BTreeMap<TerminalID, Weight>;

#[derive(Debug, Clone)]
struct Branch {
    target: u32,
    bundle: NWA,
}

#[derive(Debug, Clone)]
struct StateSummary {
    final_weight: Option<Weight>,
    branches: Vec<Branch>,
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

fn group_terminal_edges_by_target(
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    state_id: u32,
) -> BTreeMap<u32, Bundle> {
    let Some(state) = terminal_dwa.states.get(state_id as usize) else {
        return BTreeMap::new();
    };

    let mut groups = BTreeMap::<u32, Bundle>::new();
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

fn build_state_summaries(
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
) -> Vec<StateSummary> {
    terminal_dwa
        .states
        .iter()
        .enumerate()
        .map(|(state_id, state)| StateSummary {
            final_weight: state.final_weight.clone(),
            branches: group_terminal_edges_by_target(terminal_dwa, grammar, state_id as u32)
                .into_iter()
                .map(|(target, bundle)| Branch {
                    target,
                    bundle: templates.build_bundle(&bundle),
                })
                .collect(),
        })
        .collect()
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
        Some(existing) => {
            let body = existing.append_with_body(&next);
            existing.start_states.extend(body.start_states);
        }
        None => *acc = Some(next),
    }
}

fn compose_state(
    state_id: u32,
    states: &[StateSummary],
    memo: &mut BTreeMap<u32, Option<NWA>>,
) -> Option<NWA> {
    if let Some(cached) = memo.get(&state_id) {
        return cached.clone();
    }

    let Some(state) = states.get(state_id as usize) else {
        return None;
    };

    let mut composed: Option<NWA> = state.final_weight.as_ref().and_then(accepting_nwa);

    for branch in &state.branches {
        let Some(continuation) = compose_state(branch.target, states, memo) else {
            continue;
        };
        let Some(branch_with_continuation) = concatenate_nwas(&branch.bundle, &continuation) else {
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
    start_prefix_weight: Option<Weight>,
    id_map: &InternalIdMap,
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
    let core_start_state = wrapped.states[core_start as usize].clone();

    for tokenizer_state in 0..tokenizer.num_states() {
        let tsid_label = table.num_terminals as i32 + tokenizer_state as i32;
        let internal_tsid = id_map
            .tokenizer_states
            .original_to_internal
            .get(tokenizer_state as usize)
            .copied()
            .unwrap_or(tokenizer_state);

        let prefix_weight = start_prefix_weight.as_ref().and_then(|weight| {
            let tokens = weight.tokens_for_tsid(internal_tsid);
            if tokens.is_empty() {
                None
            } else {
                Some(Weight::from_token_set_for_tsid(internal_tsid, tokens))
            }
        });

        let target = if let Some(prefix_weight) = prefix_weight {
            let seed_state = wrapped.add_state();
            wrapped.set_final_weight(seed_state, prefix_weight);
            for (&label, (next, weight)) in &core_start_state.transitions {
                wrapped.add_transition(seed_state, label, *next, weight.clone());
            }
            seed_state
        } else {
            core_start
        };

        wrapped.add_transition(wrapped.start_state, tsid_label, target, Weight::all());
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
    let (terminal_dwa, start_prefix_weight) =
        build_terminal_dwa_with_prefix_weights(grammar, tokenizer, vocab, id_map);
    build_parser_dwa_from_terminal_dwa(
        table,
        grammar,
        tokenizer,
        &terminal_dwa,
        start_prefix_weight,
        id_map,
    )
}

pub(crate) fn build_parser_dwa_from_terminal_dwa(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    terminal_dwa: &DWA,
    start_prefix_weight: Option<Weight>,
    id_map: &InternalIdMap,
) -> DWA {
    let characterizations = characterize_terminals(table, grammar);
    let templates = Templates::from_characterizations(&characterizations);
    let states = build_state_summaries(terminal_dwa, grammar, &templates);

    let mut memo = BTreeMap::new();
    let Some(mut parser_nwa) = compose_state(terminal_dwa.start_state, &states, &mut memo)
    else {
        return DWA::new(0, 0);
    };

    resolve_negative_codes_in_nwa(&mut parser_nwa);
    parser_nwa.subtract_final_weights_from_outgoing();

    let core_dwa = minimize(
        &determinize(&parser_nwa)
            .expect("parser NWA determinization failed despite acyclic terminal/template composition"),
    );

    wrap_parser_dwa_with_tokenizer_seeds(table, tokenizer, &core_dwa, start_prefix_weight, id_map)
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
