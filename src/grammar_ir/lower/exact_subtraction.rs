//! Local exact-alternative subtraction during grammar lowering.
//!
//! This is not the whole-grammar exact-subtraction transform in
//! `grammar_ir::transforms::exact_subtraction`.  It handles the smaller case
//! where lowering sees `A - B` and can remove alternatives exactly because the
//! left-hand side names a nonterminal with explicit alternatives.

use std::collections::{HashMap, HashSet};

use crate::grammar_ir::ast::GrammarExpr;
use crate::grammar_ir::expr_nfa::ExprNFA;
use crate::GlrMaskError;

impl super::Lowerer {
    fn exact_subtraction_alternatives(
        &self,
        lhs_name: &str,
        exclude: &GrammarExpr,
    ) -> Result<Vec<GrammarExpr>, GlrMaskError> {
        match exclude {
            GrammarExpr::Choice(options) => {
                let mut out = Vec::new();
                for option in options {
                    out.extend(self.exact_subtraction_alternatives(lhs_name, option)?);
                }
                Ok(out)
            }
            GrammarExpr::Grouped(inner) => Ok(Self::top_level_alternatives(inner)),
            GrammarExpr::Ref(name) => {
                let Some(false) = self.named_rule_is_terminal.get(name).copied() else {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "{lhs_name} - {name} requires {name} to name a nonterminal rule"
                    )));
                };
                let referenced_expr = self.named_rule_exprs.get(name).ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!(
                        "unknown rule referenced in exact alternative subtraction: {name}"
                    ))
                })?;
                Ok(Self::top_level_alternatives(referenced_expr))
            }
            other => Ok(Self::top_level_alternatives(other)),
        }
    }

    fn canonical_exact_expr(&self, expr: &GrammarExpr) -> GrammarExpr {
        let mut visiting = HashSet::new();
        let mut memo = HashMap::new();
        self.canonical_exact_expr_inner(expr, &mut visiting, &mut memo)
    }

    fn canonical_exact_expr_inner(
        &self,
        expr: &GrammarExpr,
        visiting: &mut HashSet<String>,
        memo: &mut HashMap<String, GrammarExpr>,
    ) -> GrammarExpr {
        match Self::strip_grouping(expr) {
            GrammarExpr::Ref(name) => {
                if self.named_rule_is_terminal.get(name).copied().unwrap_or(false) {
                    return GrammarExpr::Ref(name.clone());
                }
                let Some(referenced) = self.named_rule_exprs.get(name) else {
                    return GrammarExpr::Ref(name.clone());
                };
                if let Some(canonical) = memo.get(name) {
                    return canonical.clone();
                }
                if !visiting.insert(name.clone()) {
                    return GrammarExpr::Ref(name.clone());
                }
                let canonical = self.canonical_exact_expr_inner(referenced, visiting, memo);
                visiting.remove(name);
                memo.insert(name.clone(), canonical.clone());
                canonical
            }
            GrammarExpr::Grouped(inner) => self.canonical_exact_expr_inner(inner, visiting, memo),
            GrammarExpr::Sequence(items) => GrammarExpr::Sequence(
                items
                    .iter()
                    .map(|item| self.canonical_exact_expr_inner(item, visiting, memo))
                    .collect(),
            ),
            GrammarExpr::Choice(items) => GrammarExpr::Choice(
                items
                    .iter()
                    .map(|item| self.canonical_exact_expr_inner(item, visiting, memo))
                    .collect(),
            ),
            GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
                expr: Box::new(self.canonical_exact_expr_inner(expr, visiting, memo)),
                exclude: Box::new(self.canonical_exact_expr_inner(exclude, visiting, memo)),
            },
            GrammarExpr::Intersect { expr, intersect } => GrammarExpr::Intersect {
                expr: Box::new(self.canonical_exact_expr_inner(expr, visiting, memo)),
                intersect: Box::new(self.canonical_exact_expr_inner(intersect, visiting, memo)),
            },
            GrammarExpr::Optional(inner) => GrammarExpr::Optional(Box::new(
                self.canonical_exact_expr_inner(inner, visiting, memo),
            )),
            GrammarExpr::Repeat(inner) => GrammarExpr::Repeat(Box::new(
                self.canonical_exact_expr_inner(inner, visiting, memo),
            )),
            GrammarExpr::RepeatOne(inner) => GrammarExpr::RepeatOne(Box::new(
                self.canonical_exact_expr_inner(inner, visiting, memo),
            )),
            GrammarExpr::RepeatRange { expr, min, max } => GrammarExpr::RepeatRange {
                expr: Box::new(self.canonical_exact_expr_inner(expr, visiting, memo)),
                min: *min,
                max: *max,
            },
            GrammarExpr::SeparatedSequence {
                items,
                separator,
                allow_empty,
            } => GrammarExpr::SeparatedSequence {
                items: items
                    .iter()
                    .map(|(item, required)| {
                        (
                            self.canonical_exact_expr_inner(item, visiting, memo),
                            *required,
                        )
                    })
                    .collect(),
                separator: Box::new(self.canonical_exact_expr_inner(separator, visiting, memo)),
                allow_empty: *allow_empty,
            },
            GrammarExpr::ExprNFA(expr_nfa) => GrammarExpr::ExprNFA(Box::new(ExprNFA {
                nfa: expr_nfa.nfa.clone(),
                symbols: expr_nfa
                    .symbols
                    .iter()
                    .map(|symbol| self.canonical_exact_expr_inner(symbol, visiting, memo))
                    .collect(),
            })),
            GrammarExpr::Epsilon
            | GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => Self::strip_grouping(expr).clone(),
        }
    }

    pub(super) fn exact_nonterminal_subtraction_expr(
        &self,
        expr: &GrammarExpr,
    ) -> Result<Option<GrammarExpr>, GlrMaskError> {
        let GrammarExpr::Exclude { expr: lhs_expr, exclude } = expr else {
            return Ok(None);
        };
        let GrammarExpr::Ref(lhs_name) = Self::strip_grouping(lhs_expr) else {
            return Ok(None);
        };
        let Some(false) = self.named_rule_is_terminal.get(lhs_name).copied() else {
            return Ok(None);
        };

        let lhs_rule_expr = self.named_rule_exprs.get(lhs_name).ok_or_else(|| {
            GlrMaskError::GrammarParse(format!(
                "unknown nonterminal referenced in exact alternative subtraction: {lhs_name}"
            ))
        })?;
        let mut remaining = Self::top_level_alternatives(lhs_rule_expr);
        let mut remaining_keys = remaining
            .iter()
            .map(|candidate| self.canonical_exact_expr(candidate))
            .collect::<Vec<_>>();
        for remove_alt in self.exact_subtraction_alternatives(lhs_name, exclude)? {
            let remove_alt_key = self.canonical_exact_expr(&remove_alt);
            let Some(position) = remaining_keys
                .iter()
                .position(|candidate| candidate == &remove_alt_key)
            else {
                return Err(GlrMaskError::GrammarParse(format!(
                    "no exact alternative {:?} in {}",
                    remove_alt, lhs_name
                )));
            };
            remaining.remove(position);
            remaining_keys.remove(position);
        }

        Ok(Some(match remaining.len() {
            0 => GrammarExpr::Choice(Vec::new()),
            1 => remaining.pop().unwrap(),
            _ => GrammarExpr::Choice(remaining),
        }))
    }
}

