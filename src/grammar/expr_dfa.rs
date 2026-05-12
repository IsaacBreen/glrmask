use crate::automata::unweighted_u32::dfa::{DFA, Label};

use super::ast::GrammarExpr;

/// A DFA whose transition labels are indices into `symbols`.
///
/// This keeps the transition graph compact while allowing each transition
/// symbol to be an arbitrary [`GrammarExpr`]. A transition label is valid when
/// it is non-negative and less than `symbols.len()`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExprDFA {
    pub dfa: DFA,
    pub symbols: Vec<GrammarExpr>,
}

impl ExprDFA {
    pub fn new(dfa: DFA, symbols: Vec<GrammarExpr>) -> Self {
        Self { dfa, symbols }
    }

    pub fn symbol_for_label(&self, label: Label) -> Option<&GrammarExpr> {
        usize::try_from(label).ok().and_then(|index| self.symbols.get(index))
    }
}

#[cfg(test)]
mod tests {
    use crate::grammar::ast::{lower, NamedGrammar, NamedRule};
    use crate::grammar::flat::Symbol;

    use super::*;

    #[test]
    fn lowers_expr_dfa_transition_symbols() {
        let mut dfa = DFA::new();
        let accept = dfa.add_state();
        dfa.add_transition(0, 0, accept);
        dfa.set_accepting(accept, true);

        let grammar = NamedGrammar {
            rules: vec![NamedRule {
                name: "start".into(),
                expr: GrammarExpr::ExprDFA(Box::new(ExprDFA::new(
                    dfa,
                    vec![GrammarExpr::Literal(b"a".to_vec())],
                ))),
                is_terminal: false,
                is_internal: false,
            }],
            start: "start".into(),
            ignore: None,
        };

        let lowered = lower(&grammar).expect("ExprDFA should lower");
        assert_eq!(lowered.terminals.len(), 1);
        assert!(lowered
            .rules
            .iter()
            .any(|rule| matches!(rule.rhs.as_slice(), [Symbol::Terminal(_), Symbol::Nonterminal(_)])));
    }
}
