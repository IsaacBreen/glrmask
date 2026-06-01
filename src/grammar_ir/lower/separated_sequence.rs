//! Lowering for `GrammarExpr::SeparatedSequence`.
//!
//! A separated sequence is not just ordinary repetition: absence of an optional
//! item and derivation of the empty string are distinct facts.  This module owns
//! the separator-placement invariant.

use crate::grammar_ir::ast::{CommaSepShape, GrammarExpr};
use crate::grammar_ir::flat::{Rule, Symbol};
use crate::GlrMaskError;

use super::repeat::repeat_tree_shape;

/// Read the `CommaSepShape` from the `GLRMASK_ORDERED_OBJECT_SHAPE` environment variable.
pub(crate) fn comma_sep_shape() -> CommaSepShape {
    match std::env::var("GLRMASK_ORDERED_OBJECT_SHAPE")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("left") => CommaSepShape::Left,
        Some("balanced") => CommaSepShape::Balanced,
        Some("left-balanced") | Some("left_balanced") | Some("leftbalanced") => {
            CommaSepShape::LeftBalanced
        }
        Some("right") | Some("factored") => CommaSepShape::Right,
        None => CommaSepShape::Left,
        Some(_) => CommaSepShape::Left,
    }
}

impl super::Lowerer {
    pub(super) fn lower_separated_sequence_repetition_item_nonempty_symbol(
        &mut self,
        inner: &GrammarExpr,
        separator: &GrammarExpr,
        min: usize,
        max: Option<usize>,
    ) -> Result<Option<Symbol>, GlrMaskError> {
        let Some(item_sym) = self.lower_nonnullable_expr_symbol(inner)? else {
            return Ok(None);
        };

        let sep_sym = self.lower_expr_terminalish(separator)?;
        let (_, pair_nt) = self.fresh_nonterminal("sep_rep_pair");
        self.rules.push(Rule {
            lhs: pair_nt,
            rhs: vec![sep_sym, item_sym.clone()],
        });
        let pair_symbol = Symbol::Nonterminal(pair_nt);
        let shape = repeat_tree_shape();

        if max.is_none() {
            let (_, rep_nt) = self.fresh_nonterminal("sep_rep_plus");
            self.rules.push(Rule {
                lhs: rep_nt,
                rhs: vec![item_sym.clone()],
            });
            self.rules.push(Rule {
                lhs: rep_nt,
                rhs: vec![Symbol::Nonterminal(rep_nt), pair_symbol],
            });
            return Ok(Some(Symbol::Nonterminal(rep_nt)));
        }

        let max = max.expect("finite bound expected when max.is_none() is false");
        if min > max {
            return Ok(None);
        }
        if max == 0 {
            return Ok(None);
        }

        let min = min.max(1);

        let prefix_sym = if min == 1 {
            item_sym.clone()
        } else {
            let (_, prefix_nt) = self.fresh_nonterminal("sep_rep_prefix");
            let prefix_tail_nt = self.repeat_exact_nonterminal(&pair_symbol, min - 1, shape);
            self.rules.push(Rule {
                lhs: prefix_nt,
                rhs: vec![item_sym.clone(), Symbol::Nonterminal(prefix_tail_nt)],
            });
            Symbol::Nonterminal(prefix_nt)
        };

        if min == max {
            return Ok(Some(prefix_sym));
        }

        let extra_nt = self.repeat_range_nonterminal(&pair_symbol, 0, max - min, shape);
        let (_, result_nt) = self.fresh_nonterminal("sep_rep_range");
        self.rules.push(Rule {
            lhs: result_nt,
            rhs: vec![prefix_sym, Symbol::Nonterminal(extra_nt)],
        });
        Ok(Some(Symbol::Nonterminal(result_nt)))
    }

    pub(super) fn lower_separated_sequence_item_nonempty_symbol(
        &mut self,
        item_expr: &GrammarExpr,
        separator: &GrammarExpr,
    ) -> Result<Option<Symbol>, GlrMaskError> {
        match item_expr {
            GrammarExpr::Repeat(inner) => {
                self.lower_separated_sequence_repetition_item_nonempty_symbol(inner, separator, 1, None)
            }
            GrammarExpr::RepeatOne(inner) => {
                self.lower_separated_sequence_repetition_item_nonempty_symbol(inner, separator, 1, None)
            }
            GrammarExpr::RepeatRange { expr, min, max } => {
                let required = (*min).max(1);
                self.lower_separated_sequence_repetition_item_nonempty_symbol(expr, separator, required, Some(*max))
            }
            _ => self.lower_nonnullable_expr_symbol(item_expr),
        }
    }

    /// Lower a `SeparatedSequence` into a grammar symbol.
    ///
    /// Returns `(symbol, can_be_empty)` where `can_be_empty` is `true` if the
    /// symbol can derive the empty string (i.e., all items are optional).
    ///
    /// The tree is split according to `shape`, mirroring the same algorithm used
    /// for JSON Schema ordered objects.
    pub(super) fn lower_separated_sequence_inner(
        &mut self,
        items: &[(GrammarExpr, bool)],
        separator: &GrammarExpr,
        shape: CommaSepShape,
    ) -> Result<(Symbol, bool), GlrMaskError> {
        debug_assert!(!items.is_empty());

        if items.len() == 1 {
            let (item_expr, is_required) = &items[0];
            // Always route through lower_sepseq_item_nonempty_symbol so that the
            // separator is correctly threaded through repetition items.
            // e.g. RepeatOne(item) must become `item (sep item)*`, not bare `item+`.
            // For non-repetition items the function falls through to
            // lower_nonnullable_expr_symbol which handles them correctly.
            let item_sym = self.lower_separated_sequence_item_nonempty_symbol(item_expr, separator)?;
            // Return can_be_empty=true for optional items as a signal to the parent to add
            // a "without this item and its preceding separator" alternative.  We do NOT emit
            // an epsilon rule here — that would create dangling separators in the parent rule
            // (e.g. "key": , ).  The caller of lower_separated_sequence_inner handles the
            // all-optional empty case via an explicit separate alternative (e.g. "{}").
            let can_be_empty = !is_required || self.expr_is_nullable(item_expr);
            return Ok((item_sym.unwrap_or_else(|| self.lower_expr(&GrammarExpr::Epsilon)), can_be_empty));
        }

        let mid = match shape {
            CommaSepShape::Balanced => items.len() / 2,
            CommaSepShape::Left => items.len() - 1,
            CommaSepShape::Right => 1,
            CommaSepShape::LeftBalanced => {
                let first_optional = items.iter().position(|(_, required)| !required);
                match first_optional {
                    None => items.len() - 1,
                    Some(0) => items.len() / 2,
                    Some(idx) => idx,
                }
            }
        };

        let sep_sym = self.lower_expr_terminalish(separator)?;
        let (left_sym, left_can_be_empty) =
            self.lower_separated_sequence_inner(&items[..mid], separator, shape)?;
        let (right_sym, right_can_be_empty) =
            self.lower_separated_sequence_inner(&items[mid..], separator, shape)?;

        let (_, nt) = self.fresh_nonterminal("sep_seq");

        // STICKY NOTE: DO NOT REMOVE THIS WARNING UNDER ANY CIRCUMSTANCES.
        // In generic SeparatedSequence lowering, "item derives empty" is NOT the
        // same thing as "item is absent": required nullable items can still be
        // structurally present and participate in separator placement/arity.
        // A naive right-linear lowering that treats nullable items as skippable
        // absence changes the accepted language by collapsing those cases.
        // Always: left sep right
        self.rules.push(Rule {
            lhs: nt,
            rhs: vec![left_sym.clone(), sep_sym, right_sym.clone()],
        });
        // If right side can be empty: left alone is valid
        if right_can_be_empty {
            self.rules.push(Rule { lhs: nt, rhs: vec![left_sym.clone()] });
        }
        // If left side can be empty: right alone is valid
        if left_can_be_empty {
            self.rules.push(Rule { lhs: nt, rhs: vec![right_sym.clone()] });
        }

        // Both sides can be empty: propagate the flag upward so the grandparent can add a
        // "without this subtree and its separator" alternative.  Do NOT emit nt -> ε here;
        // that would produce dangling separators in the enclosing rule.
        let can_be_empty = left_can_be_empty && right_can_be_empty;

        Ok((Symbol::Nonterminal(nt), can_be_empty))
    }

}

