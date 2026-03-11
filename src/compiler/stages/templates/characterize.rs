//! NOTE: terminal characterization is intentionally deferred.
//! Keep only the minimal data shape and entrypoint for this cleanup pass.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::{Action, GLRTable};
use crate::compiler::grammar::model::{NonterminalID, TerminalID};

type InitialShift = (u32, u32);

type InitialReduce = (u32, usize, NonterminalID);

type NtEscape = (NonterminalID, u32, u32, u32);

type NtRereduce = (NonterminalID, u32, usize, NonterminalID);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TerminalCharacterization {
    pub shifts: Vec<InitialShift>,
    pub reduces: Vec<InitialReduce>,
    pub nt_escapes: Vec<NtEscape>,
    pub nt_rereduces: Vec<NtRereduce>,
    pub all_nts: BTreeSet<NonterminalID>,
}

impl TerminalCharacterization {
    pub fn find_cycle(&self) -> Option<Vec<NonterminalID>> {
        let mut adjacency = BTreeMap::<NonterminalID, BTreeSet<NonterminalID>>::new();
        for (src_nt, _revealed_state, _pop_count, dst_nt) in &self.nt_rereduces {
            adjacency.entry(*src_nt).or_default().insert(*dst_nt);
        }

        let mut colors = BTreeMap::<NonterminalID, u8>::new();
        let mut path = Vec::new();

        fn dfs(
            node: NonterminalID,
            adjacency: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
            colors: &mut BTreeMap<NonterminalID, u8>,
            path: &mut Vec<NonterminalID>,
        ) -> Option<Vec<NonterminalID>> {
            colors.insert(node, 1);
            path.push(node);

            if let Some(neighbors) = adjacency.get(&node) {
                for &neighbor in neighbors {
                    match colors.get(&neighbor).copied().unwrap_or(0) {
                        1 => {
                            let cycle_start = path.iter().position(|nt| *nt == neighbor).unwrap_or(0);
                            let mut cycle = path[cycle_start..].to_vec();
                            cycle.push(neighbor);
                            return Some(cycle);
                        }
                        0 => {
                            if let Some(cycle) = dfs(neighbor, adjacency, colors, path) {
                                return Some(cycle);
                            }
                        }
                        _ => {}
                    }
                }
            }

            path.pop();
            colors.insert(node, 2);
            None
        }

        for &nt in adjacency.keys() {
            if colors.get(&nt).copied().unwrap_or(0) == 0 {
                if let Some(cycle) = dfs(nt, &adjacency, &mut colors, &mut path) {
                    return Some(cycle);
                }
            }
        }

        None
    }
}

pub(crate) fn characterize_terminals(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
) -> BTreeMap<TerminalID, TerminalCharacterization> {
    (0..grammar.num_terminals)
        .map(|terminal| (terminal, characterize_terminal(table, grammar, terminal)))
        .collect()
}

fn characterize_terminal(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    terminal: TerminalID,
) -> TerminalCharacterization {
    let mut shifts = BTreeSet::new();
    let mut reduces = BTreeSet::new();
    let mut nt_escapes = BTreeSet::new();
    let mut nt_rereduces = BTreeSet::new();

    for state in 0..table.num_states {
        let Some(action) = table.action(state, terminal) else {
            continue;
        };
        match action {
            Action::Shift(shift_state) => {
                shifts.insert((state, *shift_state));
            }
            Action::Reduce(rule_id) => {
                let rule = &table.rules[*rule_id as usize];
                let len = rule.rhs.len();
                if len > 0 {
                    reduces.insert((state, len - 1, rule.lhs));
                }
            }
            Action::Split { shift, reduces: split_reduces, .. } => {
                if let Some(shift_state) = shift {
                    shifts.insert((state, *shift_state));
                }
                for rule_id in split_reduces {
                    let rule = &table.rules[*rule_id as usize];
                    let len = rule.rhs.len();
                    if len > 0 {
                        reduces.insert((state, len - 1, rule.lhs));
                    }
                }
            }
            Action::Accept => {}
        }
    }

    for revealed_state in 0..table.num_states {
        if let Some(gotos) = table.goto.get(revealed_state as usize) {
            for (&nonterminal, &goto_state) in gotos {
                explore_from_goto(
                    table,
                    terminal,
                    nonterminal,
                    revealed_state,
                    goto_state,
                    &mut nt_escapes,
                    &mut nt_rereduces,
                );
            }
        }
    }

    let characterization = TerminalCharacterization {
        shifts: shifts.into_iter().collect(),
        reduces: reduces.into_iter().collect(),
        nt_escapes: nt_escapes.into_iter().collect(),
        nt_rereduces: nt_rereduces.into_iter().collect(),
        all_nts: (0..grammar.num_nonterminals).collect(),
    };

    if let Some(cycle) = characterization.find_cycle() {
        panic!(
            "terminal characterization for terminal {} contains a reduction cycle: {:?}",
            terminal,
            cycle
        );
    }

    characterization
}

fn explore_from_goto(
    table: &GLRTable,
    terminal: TerminalID,
    stack_nt: NonterminalID,
    revealed_state: u32,
    start_state: u32,
    nt_escapes: &mut BTreeSet<NtEscape>,
    nt_rereduces: &mut BTreeSet<NtRereduce>,
) {
    let mut worklist = VecDeque::new();
    let mut visited = BTreeSet::new();

    visited.insert(start_state);
    worklist.push_back(start_state);

    while let Some(current_state) = worklist.pop_front() {
        let Some(action) = table.action(current_state, terminal) else {
            continue;
        };
        match action {
            Action::Shift(shift_state) => {
                nt_escapes.insert((stack_nt, revealed_state, current_state, *shift_state));
            }
            Action::Reduce(rule_id) => {
                let rule = &table.rules[*rule_id as usize];
                handle_reduce(
                    table,
                    stack_nt,
                    revealed_state,
                    rule.rhs.len(),
                    rule.lhs,
                    &mut visited,
                    &mut worklist,
                    nt_rereduces,
                );
            }
            Action::Split { shift, reduces: split_reduces, .. } => {
                if let Some(shift_state) = shift {
                    nt_escapes.insert((stack_nt, revealed_state, current_state, *shift_state));
                }
                for rule_id in split_reduces {
                    let rule = &table.rules[*rule_id as usize];
                    handle_reduce(
                        table,
                        stack_nt,
                        revealed_state,
                        rule.rhs.len(),
                        rule.lhs,
                        &mut visited,
                        &mut worklist,
                        nt_rereduces,
                    );
                }
            }
            Action::Accept => {}
        }
    }
}

fn handle_reduce(
    table: &GLRTable,
    stack_nt: NonterminalID,
    revealed_state: u32,
    len: usize,
    reduce_nt: NonterminalID,
    visited: &mut BTreeSet<u32>,
    worklist: &mut VecDeque<u32>,
    nt_rereduces: &mut BTreeSet<NtRereduce>,
) {
    if len == 1 {
        if let Some(next_goto_state) = table.goto_target(revealed_state, reduce_nt) {
            if visited.insert(next_goto_state) {
                worklist.push_back(next_goto_state);
            }
        }
    } else if len > 1 {
        nt_rereduces.insert((stack_nt, revealed_state, len - 2, reduce_nt));
    }
}
