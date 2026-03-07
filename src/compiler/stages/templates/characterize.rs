//! Parser-side terminal characterization.
//!
//! This stage analyzes the GLR table to determine the stack patterns that make
//! each terminal valid. Those structural characterizations are later consumed
//! by the terminal stage and template-compilation stage.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::compiler::glr::analysis::GLRGrammar;
use crate::compiler::glr::table::{Action, GLRTable};
use crate::compiler::grammar::ast::{NonterminalId, TerminalId};

/// Shift: from parser state `from`, terminal T shifts to state `to`.
type InitialShift = (u32, u32);

/// Reduce: from parser state `from`, terminal T reduces rule with
/// `pop_count` states and LHS nonterminal `nt`.
type InitialReduce = (u32, usize, NonterminalId);

/// After reducing to nonterminal `nt_from`, if the revealed state is `revealed`,
/// then goto(revealed, nt_from) = `goto_state`, and from `goto_state`, terminal T
/// can shift to `shift_state`.
type NtEscape = (NonterminalId, u32, u32, u32);

/// After reducing to nonterminal `nt_from`, if the revealed state is `revealed`,
/// goto(revealed, nt_from) = `goto_state`, and from `goto_state`, terminal T
/// reduces again.
type NtRereduce = (NonterminalId, u32, usize, NonterminalId);

/// Stack pattern characterization for a single terminal.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TerminalCharacterization {
    pub shifts: Vec<InitialShift>,
    pub reduces: Vec<InitialReduce>,
    pub nt_escapes: Vec<NtEscape>,
    pub nt_rereduces: Vec<NtRereduce>,
    /// All nonterminals involved in reduce cascades.
    pub all_nts: BTreeSet<NonterminalId>,
}

/// Characterize terminals: find all parser-stack patterns that allow them.
pub(crate) fn characterize_terminals(
    table: &GLRTable,
    grammar: &GLRGrammar,
) -> BTreeMap<TerminalId, TerminalCharacterization> {
    let mut result = BTreeMap::new();
    let num_states = table.num_states;

    for t in 0..grammar.num_terminals {
        let mut tc = TerminalCharacterization {
            shifts: Vec::new(),
            reduces: Vec::new(),
            nt_escapes: Vec::new(),
            nt_rereduces: Vec::new(),
            all_nts: BTreeSet::new(),
        };

        for s in 0..num_states {
            for action in table.actions(s, t) {
                match action {
                    Action::Shift(to) => {
                        tc.shifts.push((s, *to));
                    }
                    Action::Reduce(rule_idx) => {
                        let rule = &table.rules[*rule_idx as usize];
                        let pop_count = rule.rhs.len();
                        let nt = rule.lhs;
                        tc.reduces.push((s, pop_count, nt));
                        tc.all_nts.insert(nt);
                    }
                    Action::Accept => {}
                }
            }
        }

        let mut visited_nts: BTreeSet<NonterminalId> = BTreeSet::new();
        let mut nt_queue: VecDeque<NonterminalId> = tc.all_nts.iter().copied().collect();

        while let Some(nt) = nt_queue.pop_front() {
            if !visited_nts.insert(nt) {
                continue;
            }

            for revealed in 0..num_states {
                if let Some(goto_state) = table.goto_target(revealed, nt) {
                    for action in table.actions(goto_state, t) {
                        match action {
                            Action::Shift(shift_to) => {
                                tc.nt_escapes.push((nt, revealed, goto_state, *shift_to));
                            }
                            Action::Reduce(rule_idx) => {
                                let rule = &table.rules[*rule_idx as usize];
                                let pop2 = rule.rhs.len();
                                let nt2 = rule.lhs;
                                tc.nt_rereduces.push((nt, revealed, pop2, nt2));
                                tc.all_nts.insert(nt2);
                                if !visited_nts.contains(&nt2) {
                                    nt_queue.push_back(nt2);
                                }
                            }
                            Action::Accept => {}
                        }
                    }
                }
            }
        }

        if !tc.shifts.is_empty() || !tc.reduces.is_empty() {
            result.insert(t, tc);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar::ast::tests::simple_ab_grammar;

    #[test]
    fn test_characterize_simple_ab() {
        let gdef = simple_ab_grammar();
        let grammar = GLRGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&grammar);
        let chars = characterize_terminals(&table, &grammar);

        assert!(chars.contains_key(&0));
        assert!(!chars[&0].shifts.is_empty());
        assert!(chars.contains_key(&1));
    }
}
