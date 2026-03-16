use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::compiler::glr::table::{Action, GLRTable};
use crate::compiler::grammar::model::{NonterminalID, TerminalID};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ReplaceSafeAnalysis {
    pub shift_candidates: usize,
    pub safe_shifts: usize,
    pub goto_candidates: usize,
    pub safe_gotos: usize,
    pub safe_shift_terminals: Vec<BTreeSet<TerminalID>>,
    pub safe_goto_nonterminals: Vec<BTreeSet<NonterminalID>>,
}

impl ReplaceSafeAnalysis {
    pub(crate) fn shift_is_safe(&self, state: u32, terminal: TerminalID) -> bool {
        self.safe_shift_terminals
            .get(state as usize)
            .is_some_and(|terminals| terminals.contains(&terminal))
    }

    pub(crate) fn goto_is_safe(&self, state: u32, nonterminal: NonterminalID) -> bool {
        self.safe_goto_nonterminals
            .get(state as usize)
            .is_some_and(|nonterminals| nonterminals.contains(&nonterminal))
    }
}

pub(crate) fn analyze_replace_safe(table: &GLRTable) -> ReplaceSafeAnalysis {
    let mut analysis = ReplaceSafeAnalysis {
        safe_shift_terminals: vec![BTreeSet::new(); table.num_states as usize],
        safe_goto_nonterminals: vec![BTreeSet::new(); table.num_states as usize],
        ..ReplaceSafeAnalysis::default()
    };

    for state in 0..table.num_states {
        for (&terminal, action) in &table.action[state as usize] {
            let Some(shift_target) = shift_target(action) else {
                continue;
            };
            analysis.shift_candidates += 1;
            if action_has_conflict(action) {
                continue;
            }
            if immediate_replace_safe_from_hidden_src(table, state, shift_target) {
                analysis.safe_shifts += 1;
                analysis.safe_shift_terminals[state as usize].insert(terminal);
            }
        }

        for (&nonterminal, &target) in &table.goto[state as usize] {
            analysis.goto_candidates += 1;
            if immediate_replace_safe_from_hidden_src(table, state, target) {
                analysis.safe_gotos += 1;
                analysis.safe_goto_nonterminals[state as usize].insert(nonterminal);
            }
        }
    }

    analysis
}

fn shift_target(action: &Action) -> Option<u32> {
    match action {
        Action::Shift(target) => Some(*target),
        Action::Split { shift, .. } => *shift,
        Action::Reduce(_) | Action::Accept => None,
    }
}

fn action_has_conflict(action: &Action) -> bool {
    matches!(action, Action::Split { shift: Some(_), reduces, accept: _ } if !reduces.is_empty())
}

fn immediate_replace_safe_from_hidden_src(table: &GLRTable, hidden_src: u32, current_state: u32) -> bool {
    table.action[current_state as usize]
        .values()
        .all(|action| action_is_safe_from_hidden_src(table, hidden_src, action))
}

fn action_is_safe_from_hidden_src(table: &GLRTable, hidden_src: u32, action: &Action) -> bool {
    match action {
        Action::Shift(_) => false,
        Action::Reduce(rule_id) => reduction_is_safe(table, hidden_src, *rule_id),
        Action::Split {
            shift: Some(_),
            reduces,
            accept: _,
        } => reduces.is_empty(),
        Action::Split {
            shift: None,
            reduces,
            accept: _,
        } => reduces
            .iter()
            .copied()
            .all(|rule_id| reduction_is_safe(table, hidden_src, rule_id)),
        Action::Accept => true,
    }
}

fn reduction_is_safe(table: &GLRTable, hidden_src: u32, rule_id: u32) -> bool {
    let rule = &table.rules[rule_id as usize];
    rule.rhs.len() == 1 && table.goto_target(hidden_src, rule.lhs).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::grammar::model::{GrammarDef, Rule, Symbol, Terminal};

    fn table_for(grammar: GrammarDef) -> GLRTable {
        GLRTable::build(&AnalyzedGrammar::from_grammar_def(&grammar))
    }

    #[test]
    fn test_replace_safe_marks_len1_reduce_after_shift_as_safe() {
        let grammar = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            start: 0,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..Default::default()
        };
        let table = table_for(grammar);
        let analysis = analyze_replace_safe(&table);

        assert!(analysis.shift_is_safe(0, 0));
        assert!(analysis.goto_is_safe(0, 1));
    }

    #[test]
    fn test_replace_safe_rejects_future_shift_region() {
        let grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"b".to_vec(),
                },
            ],
            ..Default::default()
        };
        let table = table_for(grammar);
        let analysis = analyze_replace_safe(&table);

        assert!(!analysis.shift_is_safe(0, 0));
    }

    #[test]
    fn test_replace_safe_rejects_goto_with_future_shift() {
        let grammar = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"b".to_vec(),
                },
            ],
            ..Default::default()
        };
        let table = table_for(grammar);
        let analysis = analyze_replace_safe(&table);

        assert!(!analysis.goto_is_safe(0, 1));
    }
}