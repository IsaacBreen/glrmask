//! Lowering for `GrammarExpr::ExprNFA`.
//!
//! An ExprNFA is already an automaton over grammar-expression labels. Lowering
//! turns each automaton state into a nonterminal and each labeled transition into
//! a production.

use crate::automata::unweighted::dfa::DFA;
use crate::grammar_ir::expr_nfa::ExprNFA;
use crate::grammar_ir::flat::{NonterminalID, Rule, Symbol};
use crate::GlrMaskError;

impl super::Lowerer {
    pub(super) fn expr_nfa_state_nonterminals(
        &mut self,
        state_count: usize,
        start: usize,
        hint: &str,
        start_lhs: Option<NonterminalID>,
    ) -> Result<Vec<NonterminalID>, GlrMaskError> {
        if start >= state_count {
            return Err(GlrMaskError::GrammarParse(format!(
                "ExprNFA start state {} is out of range for {} states",
                start, state_count
            )));
        }

        let mut nts = Vec::with_capacity(state_count);
        for state_index in 0..state_count {
            if Some(state_index) == start_lhs.map(|_| start) {
                nts.push(start_lhs.unwrap());
            } else {
                let (_, nt) = self.fresh_nonterminal(hint);
                nts.push(nt);
            }
        }
        Ok(nts)
    }

    pub(super) fn emit_expr_dfa_leftlinear(
        &mut self,
        lhs: NonterminalID,
        expr_nfa: &ExprNFA,
        dfa: &DFA,
    ) -> Result<(), GlrMaskError> {
        let state_count = dfa.states.len();
        let start = dfa.start_state as usize;
        let nts = self.expr_nfa_state_nonterminals(state_count, start, "expr_nfa_prefix", None)?;
        let start_nt = nts[start];
        self.rules.push(Rule {
            lhs: start_nt,
            rhs: Vec::new(),
        });

        for (state_index, state) in dfa.states.iter().enumerate() {
            for (label, target) in &state.transitions {
                let target = *target as usize;
                if target >= state_count {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition from state {state_index} targets out-of-range state {target}"
                    )));
                }
                let symbol_expr = expr_nfa.symbol_for_label(*label).ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition label {label} is not a valid symbol index"
                    ))
                })?;
                let symbol = self.lower_expr_terminalish(symbol_expr)?;
                self.rules.push(Rule {
                    lhs: nts[target],
                    rhs: vec![Symbol::Nonterminal(nts[state_index]), symbol],
                });
            }
        }

        for (state_index, state) in dfa.states.iter().enumerate() {
            if state.is_accepting {
                self.rules.push(Rule {
                    lhs,
                    rhs: vec![Symbol::Nonterminal(nts[state_index])],
                });
            }
        }

        Ok(())
    }

    pub(super) fn emit_expr_dfa_leftlinear_nonnullable(
        &mut self,
        lhs: NonterminalID,
        expr_nfa: &ExprNFA,
        dfa: &DFA,
    ) -> Result<(), GlrMaskError> {
        let state_count = dfa.states.len();
        let start = dfa.start_state as usize;
        if start >= state_count {
            return Err(GlrMaskError::GrammarParse(format!(
                "ExprNFA start state {} is out of range for {} states",
                dfa.start_state, state_count
            )));
        }

        let nullable_nts = self.expr_nfa_state_nonterminals(
            state_count,
            start,
            "expr_nfa_nullable_prefix",
            None,
        )?;
        let nonnullable_nts = self.expr_nfa_state_nonterminals(
            state_count,
            start,
            "expr_nfa_nonnullable_prefix",
            None,
        )?;

        self.rules.push(Rule {
            lhs: nullable_nts[start],
            rhs: Vec::new(),
        });

        for (state_index, state) in dfa.states.iter().enumerate() {
            for (label, target) in &state.transitions {
                let target = *target as usize;
                if target >= state_count {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition from state {state_index} targets out-of-range state {target}"
                    )));
                }
                let symbol_expr = expr_nfa.symbol_for_label(*label).ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition label {label} is not a valid symbol index"
                    ))
                })?;

                if self.expr_is_nullable(symbol_expr) {
                    let symbol = self.lower_expr_terminalish(symbol_expr)?;
                    self.rules.push(Rule {
                        lhs: nullable_nts[target],
                        rhs: vec![Symbol::Nonterminal(nullable_nts[state_index]), symbol],
                    });
                }

                if let Some(symbol) = self.lower_nonnullable_expr_symbol(symbol_expr)? {
                    self.rules.push(Rule {
                        lhs: nonnullable_nts[target],
                        rhs: vec![Symbol::Nonterminal(nullable_nts[state_index]), symbol],
                    });
                }

                let symbol = self.lower_expr_terminalish(symbol_expr)?;
                self.rules.push(Rule {
                    lhs: nonnullable_nts[target],
                    rhs: vec![Symbol::Nonterminal(nonnullable_nts[state_index]), symbol],
                });
            }
        }

        for (state_index, state) in dfa.states.iter().enumerate() {
            if state.is_accepting {
                self.rules.push(Rule {
                    lhs,
                    rhs: vec![Symbol::Nonterminal(nonnullable_nts[state_index])],
                });
            }
        }

        Ok(())
    }

    pub(super) fn emit_expr_nfa(&mut self, lhs: NonterminalID, expr_nfa: &ExprNFA) -> Result<(), GlrMaskError> {
        let dfa = expr_nfa.determinize_and_minimize();
        self.emit_expr_dfa_leftlinear(lhs, expr_nfa, &dfa)
    }

    pub(super) fn emit_expr_nfa_nonnullable(
        &mut self,
        lhs: NonterminalID,
        expr_nfa: &ExprNFA,
    ) -> Result<(), GlrMaskError> {
        let dfa = expr_nfa.determinize_and_minimize();
        self.emit_expr_dfa_leftlinear_nonnullable(lhs, expr_nfa, &dfa)
    }

}

