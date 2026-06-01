//! Conversion between grammar expressions and lexer expressions.
//!
//! These helpers answer local semantic questions needed by lowering: whether an
//! expression is nullable, whether ExprNFA placement is legal, and how terminal
//! grammar syntax becomes lexer-level regular expressions.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::regex::parse_regex;
use crate::ds::u8set::U8Set;
use crate::grammar_ir::ast::{GrammarExpr, NamedGrammar};
use crate::grammar_ir::flat::Rule;
use crate::grammar_ir::render::lark::u8set_to_class_def;
use crate::{GlrMaskError, Result};

use super::char_class_pattern;

/// Convert a GrammarExpr to an Expr tree, resolving terminal Ref nodes
/// via the `terminal_bodies` map and caching results in `terminal_expr_cache`.
pub(super) fn grammar_expr_to_expr(
    expr: &GrammarExpr,
    terminal_bodies: &HashMap<String, GrammarExpr>,
    terminal_expr_cache: &mut HashMap<String, Arc<Expr>>,
    visiting: &mut HashSet<String>,
) -> Result<Expr, GlrMaskError> {
    Ok(match expr {
        GrammarExpr::Grouped(inner) => {
            return grammar_expr_to_expr(inner, terminal_bodies, terminal_expr_cache, visiting);
        }
        GrammarExpr::Literal(bytes) => Expr::U8Seq(bytes.clone()),
        GrammarExpr::CharClass { def, negate, utf8 } => {
            let pattern = char_class_pattern(def, *negate);
            parse_regex(&pattern, *utf8)
        }
        GrammarExpr::RawRegex(pattern) => parse_regex(pattern, true),
        GrammarExpr::LexerDfa(dfa) => Expr::Dfa(dfa.clone()),
        GrammarExpr::AnyByte => Expr::U8Class(U8Set::from_range(0, 255)),
        GrammarExpr::Epsilon => Expr::Epsilon,
        GrammarExpr::Sequence(parts) => {
            let exprs: Vec<Expr> = parts.iter().map(|p| grammar_expr_to_expr(p, terminal_bodies, terminal_expr_cache, visiting)).collect::<Result<_, _>>()?;
            if exprs.len() == 1 {
                exprs.into_iter().next().unwrap()
            } else {
                Expr::Seq(exprs)
            }
        }
        GrammarExpr::Choice(options) => {
            let exprs: Vec<Expr> = options.iter().map(|o| grammar_expr_to_expr(o, terminal_bodies, terminal_expr_cache, visiting)).collect::<Result<_, _>>()?;
            if exprs.len() == 1 {
                exprs.into_iter().next().unwrap()
            } else {
                Expr::Choice(exprs)
            }
        }
        GrammarExpr::Exclude { expr, exclude } => Expr::Exclude {
            expr: Box::new(grammar_expr_to_expr(expr, terminal_bodies, terminal_expr_cache, visiting)?),
            exclude: Box::new(grammar_expr_to_expr(exclude, terminal_bodies, terminal_expr_cache, visiting)?),
        },
        GrammarExpr::Intersect { expr, intersect } => Expr::Intersect {
            expr: Box::new(grammar_expr_to_expr(expr, terminal_bodies, terminal_expr_cache, visiting)?),
            intersect: Box::new(grammar_expr_to_expr(intersect, terminal_bodies, terminal_expr_cache, visiting)?),
        },
        GrammarExpr::Optional(inner) => Expr::Repeat {
            expr: Box::new(grammar_expr_to_expr(inner, terminal_bodies, terminal_expr_cache, visiting)?),
            min: 0,
            max: Some(1),
        },
        GrammarExpr::Repeat(inner) => Expr::Repeat {
            expr: Box::new(grammar_expr_to_expr(inner, terminal_bodies, terminal_expr_cache, visiting)?),
            min: 0,
            max: None,
        },
        GrammarExpr::RepeatOne(inner) => Expr::Repeat {
            expr: Box::new(grammar_expr_to_expr(inner, terminal_bodies, terminal_expr_cache, visiting)?),
            min: 1,
            max: None,
        },
        GrammarExpr::RepeatRange { expr, min, max } => Expr::Repeat {
            expr: Box::new(grammar_expr_to_expr(expr, terminal_bodies, terminal_expr_cache, visiting)?),
            min: *min,
            max: Some(*max),
        },
        GrammarExpr::Ref(name) => {
            // Look up in cache first
            if let Some(cached) = terminal_expr_cache.get(name) {
                return Ok(Expr::Shared(cached.clone()));
            }
            // Must be a terminal rule — look up its body and resolve it
            if let Some(body) = terminal_bodies.get(name).cloned() {
                if !visiting.insert(name.clone()) {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "cycle detected in terminal rule references: {name}"
                    )));
                }
                let expr = grammar_expr_to_expr(&body, terminal_bodies, terminal_expr_cache, visiting)?;
                let arc = Arc::new(expr);
                terminal_expr_cache.insert(name.clone(), arc.clone());
                visiting.remove(name);
                Expr::Shared(arc)
            } else {
                return Err(GlrMaskError::GrammarParse(format!(
                    "unresolved Ref({name}) in terminal body — not found in terminal rules"
                )));
            }
        }
        GrammarExpr::SeparatedSequence { .. } => {
            return Err(GlrMaskError::GrammarParse(
                "GrammarExpr::SeparatedSequence cannot appear inside a terminal rule".into(),
            ));
        }
        GrammarExpr::ExprNFA(_) => {
            return Err(GlrMaskError::GrammarParse(
                "GrammarExpr::ExprNFA cannot appear inside a terminal rule".into(),
            ));
        }
    })
}

pub(super) fn grammar_expr_is_nullable(
    expr: &GrammarExpr,
    rule_nullable: &HashMap<String, bool>,
) -> bool {
    match expr {
        GrammarExpr::Ref(name) => rule_nullable.get(name).copied().unwrap_or(false),
        GrammarExpr::Grouped(inner) => grammar_expr_is_nullable(inner, rule_nullable),
        GrammarExpr::Sequence(parts) => parts.iter().all(|part| grammar_expr_is_nullable(part, rule_nullable)),
        GrammarExpr::Choice(options) => options.iter().any(|option| grammar_expr_is_nullable(option, rule_nullable)),
        GrammarExpr::Epsilon => true,
        GrammarExpr::Exclude { expr, exclude } => {
            grammar_expr_is_nullable(expr, rule_nullable)
                && !grammar_expr_is_nullable(exclude, rule_nullable)
        }
        GrammarExpr::Intersect { expr, intersect } => {
            grammar_expr_is_nullable(expr, rule_nullable)
                && grammar_expr_is_nullable(intersect, rule_nullable)
        }
        GrammarExpr::Optional(_) | GrammarExpr::Repeat(_) => true,
        GrammarExpr::RepeatOne(inner) => grammar_expr_is_nullable(inner, rule_nullable),
        GrammarExpr::RepeatRange { expr, min, .. } => {
            *min == 0 || grammar_expr_is_nullable(expr, rule_nullable)
        }
        GrammarExpr::Literal(bytes) => bytes.is_empty(),
        GrammarExpr::CharClass { def, negate, utf8 } => {
            parse_regex(&char_class_pattern(def, *negate), *utf8).is_nullable()
        }
        GrammarExpr::RawRegex(pattern) => parse_regex(pattern, true).is_nullable(),
        GrammarExpr::LexerDfa(dfa) => !dfa.finalizers(0).is_empty(),
        GrammarExpr::AnyByte => false,
        GrammarExpr::SeparatedSequence { items, allow_empty, .. } => {
            *allow_empty
                && items
                    .iter()
                    .all(|(item, is_required)| !*is_required || grammar_expr_is_nullable(item, rule_nullable))
        }
        GrammarExpr::ExprNFA(expr_nfa) => {
            let dfa = expr_nfa.determinize_and_minimize();
            let state_count = dfa.states.len();
            let start = dfa.start_state as usize;
            if start >= state_count {
                return false;
            }

            let mut nullable_from_state = vec![false; state_count];
            for (state_index, state) in dfa.states.iter().enumerate() {
                nullable_from_state[state_index] = state.is_accepting;
            }

            let mut changed = true;
            while changed {
                changed = false;
                for (state_index, state) in dfa.states.iter().enumerate() {
                    if nullable_from_state[state_index] {
                        continue;
                    }
                    for (label, target) in &state.transitions {
                        let target = *target as usize;
                        let Some(symbol) = expr_nfa.symbol_for_label(*label) else {
                            continue;
                        };
                        if target < state_count
                            && nullable_from_state[target]
                            && grammar_expr_is_nullable(symbol, rule_nullable)
                        {
                            nullable_from_state[state_index] = true;
                            changed = true;
                            break;
                        }
                    }
                }
            }
            nullable_from_state[start]
        }
    }
}

pub(super) fn compute_rule_nullability(grammar: &NamedGrammar) -> HashMap<String, bool> {
    let mut nullable = grammar
        .rules
        .iter()
        .map(|rule| (rule.name.clone(), false))
        .collect::<HashMap<_, _>>();

    loop {
        let mut changed = false;
        for rule in &grammar.rules {
            let is_nullable = grammar_expr_is_nullable(&rule.expr, &nullable);
            if is_nullable && !nullable.get(&rule.name).copied().unwrap_or(false) {
                nullable.insert(rule.name.clone(), true);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    nullable
}

pub(super) fn validate_expr_nfa_placement(grammar: &NamedGrammar) -> Result<(), GlrMaskError> {
    fn walk(expr: &GrammarExpr, top_level: bool, rule_name: &str) -> Result<(), GlrMaskError> {
        match expr {
            GrammarExpr::ExprNFA(_) => {
                if !top_level {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "GrammarExpr::ExprNFA must be the complete expression of a nonterminal rule; found nested in {rule_name}"
                    )));
                }
            }
            GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
                for part in parts {
                    walk(part, false, rule_name)?;
                }
            }
            GrammarExpr::Grouped(inner) => {
                walk(inner, false, rule_name)?;
            }
            GrammarExpr::Exclude { expr, exclude } => {
                walk(expr, false, rule_name)?;
                walk(exclude, false, rule_name)?;
            }
            GrammarExpr::Intersect { expr, intersect } => {
                walk(expr, false, rule_name)?;
                walk(intersect, false, rule_name)?;
            }
            GrammarExpr::Optional(inner)
            | GrammarExpr::Repeat(inner)
            | GrammarExpr::RepeatOne(inner)
            | GrammarExpr::RepeatRange { expr: inner, .. } => {
                walk(inner, false, rule_name)?;
            }
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                for (item, _) in items {
                    walk(item, false, rule_name)?;
                }
                walk(separator, false, rule_name)?;
            }
            GrammarExpr::Ref(_)
            | GrammarExpr::Epsilon
            | GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => {}
        }
        Ok(())
    }

    for rule in &grammar.rules {
        if rule.is_terminal && matches!(rule.expr, GrammarExpr::ExprNFA(_)) {
            return Err(GlrMaskError::GrammarParse(format!(
                "GrammarExpr::ExprNFA cannot be used as terminal rule {}",
                rule.name
            )));
        }
        walk(&rule.expr, true, &rule.name)?;
    }
    Ok(())
}

/// Convert a lexer-level [`Expr`] into an equivalent [`GrammarExpr`].
///
/// Every `Expr` variant has a `GrammarExpr` counterpart, so this is lossless.
/// `Expr::U8Class(U8Set)` is converted to `GrammarExpr::CharClass` using a
/// range-encoded string representation.
pub fn expr_to_grammar_expr(expr: &Expr) -> GrammarExpr {
    match expr {
        Expr::Shared(inner) => expr_to_grammar_expr(inner),
        Expr::U8Seq(bytes) => GrammarExpr::Literal(bytes.clone()),
        Expr::U8Class(set) => GrammarExpr::CharClass {
            def: u8set_to_class_def(set),
            negate: false,
            utf8: false,
        },
        Expr::Dfa(dfa) => GrammarExpr::LexerDfa(dfa.clone()),
        Expr::Epsilon => GrammarExpr::Epsilon,
        Expr::Seq(parts) => {
            let items: Vec<_> = parts.iter().map(expr_to_grammar_expr).collect();
            match items.len() {
                0 => GrammarExpr::Epsilon,
                1 => items.into_iter().next().unwrap(),
                _ => GrammarExpr::Sequence(items),
            }
        }
        Expr::Choice(alts) => {
            let items: Vec<_> = alts.iter().map(expr_to_grammar_expr).collect();
            match items.len() {
                0 => GrammarExpr::Epsilon,
                1 => items.into_iter().next().unwrap(),
                _ => GrammarExpr::Choice(items),
            }
        }
        Expr::Exclude { expr, exclude } => GrammarExpr::Exclude {
            expr: Box::new(expr_to_grammar_expr(expr)),
            exclude: Box::new(expr_to_grammar_expr(exclude)),
        },
        Expr::Intersect { expr, intersect } => GrammarExpr::Intersect {
            expr: Box::new(expr_to_grammar_expr(expr)),
            intersect: Box::new(expr_to_grammar_expr(intersect)),
        },
        Expr::Repeat { expr: inner, min, max } => {
            let g = expr_to_grammar_expr(inner);
            match (*min, *max) {
                (0, None) => GrammarExpr::Repeat(Box::new(g)),
                (1, None) => GrammarExpr::RepeatOne(Box::new(g)),
                (0, Some(1)) => GrammarExpr::Optional(Box::new(g)),
                (n, Some(m)) => GrammarExpr::RepeatRange { expr: Box::new(g), min: n, max: m },
                (n, None) => {
                    // n+ : express as exactly-n followed by zero-or-more
                    GrammarExpr::Sequence(vec![
                        GrammarExpr::RepeatRange { expr: Box::new(g.clone()), min: n, max: n },
                        GrammarExpr::Repeat(Box::new(g)),
                    ])
                }
            }
        }
    }
}


pub(super) fn dedup_rules_preserving_first_occurrence(rules: &mut Vec<Rule>) {
    let mut seen = HashSet::new();
    rules.retain(|rule| seen.insert(rule.clone()));
}
