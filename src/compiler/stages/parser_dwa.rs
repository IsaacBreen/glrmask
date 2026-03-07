

















#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: This file is the direct glrmask counterpart to sep1's `precompute4/parser_dwa.rs`, with the same high-level job but a much thinner current implementation.

use std::collections::{BTreeMap, BTreeSet};

use range_set_blaze::RangeSetBlaze;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::{DWA, DWAState};
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::labels::encode_positive_label;
use crate::compiler::glr::table::{Action, GLRTable};
use crate::compiler::glr::labels::{DEFAULT_LABEL, is_negative_label, negative_to_positive_label};
use crate::compiler::grammar::model::{NonterminalID, TerminalID};
use crate::compiler::resolve_negatives::resolve_negative_codes_in_nwa;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::templates::characterize::characterize_terminals;
use crate::compiler::stages::templates::Templates;
use crate::compiler::terminal_dwa::{build_terminal_dwa, TerminalDWA};
use crate::Vocab;
use crate::ds::weight::Weight;












fn find_cycle_in_non_accepting_states(dwa: &DWA) -> Option<Vec<usize>> {
    let n = dwa.states.len();
    let non_accepting: Vec<bool> = dwa.states.iter().map(|s| s.final_weight.is_none()).collect();
    let start = dwa.start_state as usize;
    if start >= n || !non_accepting[start] {
        return None; 
    }

    let mut color = vec![0u8; n]; 
    let mut parent = vec![usize::MAX; n];

    fn visit_cycle_path(
        u: usize,
        states: &[DWAState],
        non_accepting: &[bool],
        color: &mut [u8],
        parent: &mut [usize],
    ) -> Option<usize> {
        color[u] = 1;
        for (v, _) in states[u].transitions.values() {
            let v = *v as usize;
            if v >= color.len() || !non_accepting[v] {
                continue; 
            }
            match color[v] {
                1 => {
                    parent[v] = u;
                    return Some(v); 
                }
                0 => {
                    parent[v] = u;
                    if let Some(cs) = visit_cycle_path(v, states, non_accepting, color, parent) {
                        return Some(cs);
                    }
                }
                _ => {}
            }
        }
        color[u] = 2;
        None
    }

    
    
    if let Some(cycle_start) = visit_cycle_path(start, &dwa.states, &non_accepting, &mut color, &mut parent) {
        
        let mut path = vec![cycle_start];
        let mut cur = parent[cycle_start];
        while cur != cycle_start && cur != usize::MAX {
            path.push(cur);
            cur = parent[cur];
        }
        path.push(cycle_start);
        path.reverse();
        return Some(path);
    }
    None
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

fn concatenate_nwa_in_place(left: &mut NWA, right: &NWA) {
    let accepting_states: Vec<(u32, Weight)> = left
        .states
        .iter()
        .enumerate()
        .filter_map(|(state_id, state)| {
            state.final_weight.clone().map(|weight| (state_id as u32, weight))
        })
        .collect();

    let offset = left.states.len() as u32;
    append_nwa(left, right);
    let continuation_starts: Vec<u32> = right.start_states.iter().map(|state| offset + *state).collect();

    for (state_id, weight) in accepting_states {
        if let Some(state) = left.states.get_mut(state_id as usize) {
            state.final_weight = None;
        }
        for start in &continuation_starts {
            left.add_epsilon(state_id, *start, weight.clone());
        }
    }
}

fn terminal_branches_for_tokenizer_state(
    terminal_dwa: &TerminalDWA,
    grammar: &AnalyzedGrammar,
    terminal_state: u32,
    tokenizer_state: u32,
) -> Vec<(TerminalID, u32, Weight)> {
    let Some(state) = terminal_dwa.nwa.states.get(terminal_state as usize) else {
        return Vec::new();
    };

    let mut branches = Vec::new();
    for (&label, targets) in &state.transitions {
        if label < 0 || label as u32 >= grammar.num_terminals {
            continue;
        }

        for (target, weight) in targets {
            let tokens = weight.tokens_for_tsid(tokenizer_state);
            if tokens.is_empty() {
                continue;
            }
            branches.push((
                label as TerminalID,
                *target,
                Weight::from_token_set_for_tsid(tokenizer_state, tokens),
            ));
        }
    }

    branches
}

fn terminal_branch_groups_for_tokenizer_state(
    terminal_dwa: &TerminalDWA,
    grammar: &AnalyzedGrammar,
    terminal_state: u32,
    tokenizer_state: u32,
) -> BTreeMap<u32, BTreeMap<TerminalID, Weight>> {
    let mut groups = BTreeMap::<u32, BTreeMap<TerminalID, Weight>>::new();
    for (terminal, target, weight) in terminal_branches_for_tokenizer_state(
        terminal_dwa,
        grammar,
        terminal_state,
        tokenizer_state,
    ) {
        groups
            .entry(target)
            .or_default()
            .entry(terminal)
            .and_modify(|existing: &mut Weight| *existing = existing.union(&weight))
            .or_insert(weight);
    }
    groups
}

fn terminal_weights_for_tokenizer_state(
    terminal_dwa: &TerminalDWA,
    grammar: &AnalyzedGrammar,
    root_state: u32,
    tokenizer_state: u32,
) -> BTreeMap<TerminalID, Weight> {
    let mut weights = BTreeMap::new();
    for (terminal, _target, weight) in terminal_branches_for_tokenizer_state(
        terminal_dwa,
        grammar,
        root_state,
        tokenizer_state,
    ) {
        weights
            .entry(terminal)
            .and_modify(|existing: &mut Weight| *existing = existing.union(&weight))
            .or_insert(weight);
    }

    weights
}

fn build_branch_bundle(
    templates: &Templates,
    terminal_dwa: &TerminalDWA,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
    parser_state: u32,
    tokenizer_state: u32,
    target_state: u32,
    terminal_group: &BTreeMap<TerminalID, Weight>,
) -> Option<NWA> {
    let parser_terminal_weights: BTreeMap<_, _> = terminal_group
        .iter()
        .filter(|(terminal, _)| !table.actions(parser_state, **terminal).is_empty())
        .map(|(&terminal, weight)| (terminal, weight.clone()))
        .collect();
    if parser_terminal_weights.is_empty() {
        return None;
    }

    let mut bundle = templates.build_bundle(&parser_terminal_weights);
    let continuation_groups = terminal_branch_groups_for_tokenizer_state(
        terminal_dwa,
        grammar,
        target_state,
        tokenizer_state,
    );

    let mut continuation_union = NWA::new(0, 0);
    continuation_union.states.clear();
    continuation_union.start_states.clear();
    let mut has_continuation = false;

    for (next_target, next_group) in &continuation_groups {
        if let Some(next_bundle) = build_branch_bundle(
            templates,
            terminal_dwa,
            grammar,
            table,
            parser_state,
            tokenizer_state,
            *next_target,
            next_group,
        ) {
            append_nwa(&mut continuation_union, &next_bundle);
            has_continuation = true;
        }
    }

    if has_continuation {
        concatenate_nwa_in_place(&mut bundle, &continuation_union);
    }

    Some(bundle)
}



pub fn build_parser_dwa(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
) -> DWA {
    let terminal_dwa = build_terminal_dwa(grammar, tokenizer, vocab, id_map);
    build_parser_dwa_from_terminal_dwa(table, grammar, tokenizer, &terminal_dwa, id_map)
}

pub(crate) fn build_parser_dwa_from_terminal_dwa(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    terminal_dwa: &TerminalDWA,
    id_map: &InternalIdMap,
) -> DWA {
    // SEP1_MAP: Like sep1's parser-DWA builder, this stage composes terminal automata, template automata, and parser-table behavior into the final parser DWA.
    let mut dwa = DWA::new(id_map.num_tsids(), id_map.max_token_id());
    let characterizations = characterize_terminals(table, grammar);
    let templates = Templates::from_characterizations(&characterizations);

    fn tsid_seed_label(table: &GLRTable, tokenizer_state: u32) -> i32 {
        table.num_terminals as i32 + tokenizer_state as i32
    }

    for tokenizer_state in 0..tokenizer.num_states() {
        let internal_tsid = id_map
            .tokenizer_states
            .original_to_internal
            .get(tokenizer_state as usize)
            .copied()
            .unwrap_or(tokenizer_state);
        let root_state = terminal_dwa
            .tsid_roots
            .get(internal_tsid as usize)
            .copied()
            .unwrap_or(0);
        let branch_groups = root_terminal_branch_groups_for_tokenizer_state(
            terminal_dwa,
            grammar,
            root_state,
            tokenizer_state,
        );
        let terminal_weights =
            terminal_weights_for_tokenizer_state(terminal_dwa, grammar, root_state, tokenizer_state);

        let seed_state = dwa.add_state();
        let mut seed_weight = Weight::empty();
        for weight in terminal_weights.values() {
            seed_weight = seed_weight.union(weight);
        }
        dwa.add_transition(
            dwa.start_state,
            tsid_seed_label(table, tokenizer_state),
            seed_state,
            seed_weight,
        );

        for parser_state in 0..table.num_states {
            let mut bundle_nwa = NWA::new(0, 0);
            bundle_nwa.states.clear();
            bundle_nwa.start_states.clear();
            let mut has_group_bundle = false;

            for (target_state, terminal_group) in &branch_groups {
                let Some(group_bundle) = build_branch_bundle(
                    &templates,
                    terminal_dwa,
                    grammar,
                    table,
                    parser_state,
                    tokenizer_state,
                    *target_state,
                    terminal_group,
                ) else {
                    continue;
                };

                append_nwa(&mut bundle_nwa, &group_bundle);
                has_group_bundle = true;
            }

            if !has_group_bundle {
                continue;
            }

            resolve_negative_codes_in_nwa(&mut bundle_nwa);
            let bundle_dwa = minimize(
                &determinize(&bundle_nwa).expect(
                    "template bundle determinization failed despite acyclic terminal characterization",
                ),
            );

            let bundle_is_dead = bundle_dwa
                .states
                .get(bundle_dwa.start_state as usize)
                .map(|state| state.final_weight.is_none() && state.transitions.is_empty())
                .unwrap_or(true);
            if bundle_is_dead {
                continue;
            }

            let bundle_start = append_dwa(&mut dwa, &bundle_dwa);
            dwa.add_transition(
                seed_state,
                encode_positive_label(parser_state),
                bundle_start,
                Weight::all(),
            );
        }
    }
    dwa
}

fn root_terminal_branch_groups_for_tokenizer_state(
    terminal_dwa: &TerminalDWA,
    grammar: &AnalyzedGrammar,
    root_state: u32,
    tokenizer_state: u32,
) -> BTreeMap<u32, BTreeMap<TerminalID, Weight>> {
    terminal_branch_groups_for_tokenizer_state(
            terminal_dwa,
            grammar,
            root_state,
            tokenizer_state,
    )
}





#[cfg(test)]
mod tests {
    use super::*;
    use range_set_blaze::RangeSetBlaze;
    use crate::Vocab;
    use crate::automata::lexer::tokenizer::Tokenizer;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::grammar::model::GrammarDef;
    use crate::compiler::grammar::model::tests::*;

    fn make_vocab_and_preprocessing(
        gdef: &GrammarDef,
    ) -> (Vocab, Tokenizer, InternalIdMap) {
        let tok = Tokenizer::from_grammar_def(gdef);
        
        let mut entries: Vec<(u32, Vec<u8>)> = Vec::new();
        for (i, td) in gdef.terminals.iter().enumerate() {
            entries.push((i as u32, td.name.as_bytes().to_vec()));
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
