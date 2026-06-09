use std::collections::{HashMap, HashSet};

use crate::grammar::ast::{GrammarExpr, Quantifier, NamedGrammar};

/// Conservative normalization for `NamedGrammar` ASTs.
///
/// This pass is intentionally syntax-directed and language-preserving. It does
/// not try to discover arbitrary grammar equivalences; it only removes wrapper
/// structure that can obscure later grammar-shape optimizations.
pub fn simplify_named_grammar(grammar: &mut NamedGrammar) -> SimplifyStats {
    let mut stats = SimplifyStats::default();

    loop {
        let before = stats;

        for rule in &mut grammar.rules {
            let expr = std::mem::replace(&mut rule.expr, GrammarExpr::Epsilon);
            rule.expr = simplify_expr(expr, &mut stats);
        }

        inline_single_use_sequence_and_choice_rules(grammar, &mut stats);

        if stats == before {
            break;
        }
    }

    stats
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SimplifyStats {
    pub singleton_sequences: usize,
    pub singleton_choices: usize,
    pub flattened_sequences: usize,
    pub flattened_choices: usize,
    pub removed_sequence_epsilons: usize,
    pub repeat_simplifications: usize,
    pub inlined_rules: usize,
}

fn simplify_expr(expr: GrammarExpr, stats: &mut SimplifyStats) -> GrammarExpr {
    match expr {
        GrammarExpr::Sequence(parts) => simplify_sequence(parts, stats),
        GrammarExpr::Choice(options) => simplify_choice(options, stats),
        GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
            expr: Box::new(simplify_expr(*expr, stats)),
            exclude: Box::new(simplify_expr(*exclude, stats)),
        },
        GrammarExpr::Intersect { expr, intersect } => GrammarExpr::Intersect {
            expr: Box::new(simplify_expr(*expr, stats)),
            intersect: Box::new(simplify_expr(*intersect, stats)),
        },
        GrammarExpr::Quantified(inner, quantifier) => match quantifier {
            Quantifier::Optional => simplify_optional(*inner, stats),
            Quantifier::ZeroPlus => simplify_repeat(*inner, stats),
            Quantifier::OnePlus => simplify_repeat_one(*inner, stats),
            Quantifier::Range(min, max) => simplify_repeat_range(*inner, min, max, stats),
        },
        GrammarExpr::SeparatedSequence {
            items,
            separator,
            allow_empty,
        } => GrammarExpr::SeparatedSequence {
            items: items
                .into_iter()
                .map(|(item, quantifier)| (simplify_expr(item, stats), quantifier))
                .collect(),
            separator: Box::new(simplify_expr(*separator, stats)),
            allow_empty,
        },
        GrammarExpr::ExprNFA(mut expr_nfa) => {
            expr_nfa.symbols = expr_nfa
                .symbols
                .into_iter()
                .map(|symbol| simplify_expr(symbol, stats))
                .collect();
            GrammarExpr::ExprNFA(expr_nfa)
        }
        atom => atom,
    }
}

fn simplify_sequence(parts: Vec<GrammarExpr>, stats: &mut SimplifyStats) -> GrammarExpr {
    let mut out = Vec::new();
    for part in parts {
        match simplify_expr(part, stats) {
            GrammarExpr::Sequence(nested) => {
                stats.flattened_sequences += 1;
                out.extend(nested);
            }
            GrammarExpr::Epsilon => {
                stats.removed_sequence_epsilons += 1;
            }
            other => out.push(other),
        }
    }

    match out.len() {
        0 => GrammarExpr::Epsilon,
        1 => {
            stats.singleton_sequences += 1;
            out.pop().unwrap()
        }
        _ => GrammarExpr::Sequence(out),
    }
}

fn simplify_choice(options: Vec<GrammarExpr>, stats: &mut SimplifyStats) -> GrammarExpr {
    let mut out = Vec::new();
    for option in options {
        match simplify_expr(option, stats) {
            GrammarExpr::Choice(nested) => {
                stats.flattened_choices += 1;
                out.extend(nested);
            }
            other => out.push(other),
        }
    }

    match out.len() {
        0 => GrammarExpr::Epsilon,
        1 => {
            stats.singleton_choices += 1;
            out.pop().unwrap()
        }
        _ => GrammarExpr::Choice(out),
    }
}

fn simplify_optional(inner: GrammarExpr, stats: &mut SimplifyStats) -> GrammarExpr {
    match simplify_expr(inner, stats) {
        GrammarExpr::Epsilon => {
            stats.repeat_simplifications += 1;
            GrammarExpr::Epsilon
        }
        GrammarExpr::Quantified(inner, Quantifier::Optional) => {
            stats.repeat_simplifications += 1;
            GrammarExpr::Quantified(inner, Quantifier::Optional)
        }
        GrammarExpr::Quantified(inner, Quantifier::ZeroPlus) | GrammarExpr::Quantified(inner, Quantifier::OnePlus) => {
            stats.repeat_simplifications += 1;
            GrammarExpr::Quantified(inner, Quantifier::ZeroPlus)
        }
        other => GrammarExpr::Quantified(Box::new(other), Quantifier::Optional),
    }
}

fn simplify_repeat(inner: GrammarExpr, stats: &mut SimplifyStats) -> GrammarExpr {
    match simplify_expr(inner, stats) {
        GrammarExpr::Epsilon => {
            stats.repeat_simplifications += 1;
            GrammarExpr::Epsilon
        }
        GrammarExpr::Quantified(inner, Quantifier::Optional) | GrammarExpr::Quantified(inner, Quantifier::ZeroPlus) | GrammarExpr::Quantified(inner, Quantifier::OnePlus) => {
            stats.repeat_simplifications += 1;
            GrammarExpr::Quantified(inner, Quantifier::ZeroPlus)
        }
        other => GrammarExpr::Quantified(Box::new(other), Quantifier::ZeroPlus),
    }
}

fn simplify_repeat_one(inner: GrammarExpr, stats: &mut SimplifyStats) -> GrammarExpr {
    match simplify_expr(inner, stats) {
        GrammarExpr::Epsilon => {
            stats.repeat_simplifications += 1;
            GrammarExpr::Epsilon
        }
        GrammarExpr::Quantified(inner, Quantifier::OnePlus) => {
            stats.repeat_simplifications += 1;
            GrammarExpr::Quantified(inner, Quantifier::OnePlus)
        }
        GrammarExpr::Quantified(inner, Quantifier::Optional) | GrammarExpr::Quantified(inner, Quantifier::ZeroPlus) => {
            stats.repeat_simplifications += 1;
            GrammarExpr::Quantified(inner, Quantifier::ZeroPlus)
        }
        other => GrammarExpr::Quantified(Box::new(other), Quantifier::OnePlus),
    }
}

fn simplify_repeat_range(
    inner: GrammarExpr,
    min: usize,
    max: Option<usize>,
    stats: &mut SimplifyStats,
) -> GrammarExpr {
    let inner = simplify_expr(inner, stats);
    match (min, max) {
        (0, Some(0)) => {
            stats.repeat_simplifications += 1;
            GrammarExpr::Epsilon
        }
        (1, Some(1)) => {
            stats.repeat_simplifications += 1;
            inner
        }
        (0, Some(1)) => {
            stats.repeat_simplifications += 1;
            simplify_optional(inner, stats)
        }
        _ => GrammarExpr::Quantified(Box::new(inner), Quantifier::Range(min, max)),
    }
}

fn inline_single_use_sequence_and_choice_rules(grammar: &mut NamedGrammar, stats: &mut SimplifyStats) {
    let ref_counts = reference_counts(grammar);
    let rule_exprs = grammar
        .rules
        .iter()
        .map(|rule| (rule.name.clone(), rule.expr.clone()))
        .collect::<HashMap<_, _>>();
    let protected = protected_rule_names(grammar);
    let mut removed = HashSet::new();

    for rule in &mut grammar.rules {
        if rule.is_terminal {
            continue;
        }
        inline_refs_in_expr(
            &mut rule.expr,
            &rule_exprs,
            &ref_counts,
            &protected,
            &mut removed,
            stats,
        );
    }

    if removed.is_empty() {
        return;
    }

    grammar.rules.retain(|rule| !removed.contains(&rule.name));
}

fn reference_counts(grammar: &NamedGrammar) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for rule in &grammar.rules {
        collect_ref_counts(&rule.expr, &mut counts);
    }
    counts
}

fn collect_ref_counts(expr: &GrammarExpr, counts: &mut HashMap<String, usize>) {
    match expr {
        GrammarExpr::Ref(name) => {
            *counts.entry(name.clone()).or_insert(0) += 1;
        }
        GrammarExpr::Grouped(inner) => {
            collect_ref_counts(inner, counts);
        }
        GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
            for part in parts {
                collect_ref_counts(part, counts);
            }
        }
        GrammarExpr::Exclude { expr, exclude } => {
            collect_ref_counts(expr, counts);
            collect_ref_counts(exclude, counts);
        }
        GrammarExpr::Intersect { expr, intersect } => {
            collect_ref_counts(expr, counts);
            collect_ref_counts(intersect, counts);
        }
        GrammarExpr::Quantified(inner, Quantifier::Optional) | GrammarExpr::Quantified(inner, Quantifier::ZeroPlus) | GrammarExpr::Quantified(inner, Quantifier::OnePlus) => {
            collect_ref_counts(inner, counts);
        }
        GrammarExpr::Quantified(expr, Quantifier::Range(_, _)) => collect_ref_counts(expr, counts),
        GrammarExpr::SeparatedSequence {
            items, separator, ..
        } => {
            for (item, _) in items {
                collect_ref_counts(item, counts);
            }
            collect_ref_counts(separator, counts);
        }
        GrammarExpr::ExprNFA(expr_nfa) => {
            for symbol in &expr_nfa.symbols {
                collect_ref_counts(symbol, counts);
            }
        }
        GrammarExpr::Epsilon
        | GrammarExpr::Literal(_)
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte => {}
    }
}

fn protected_rule_names(grammar: &NamedGrammar) -> HashSet<String> {
    let mut protected = HashSet::new();
    protected.insert(grammar.start.clone());
    if let Some(ignore) = &grammar.ignore {
        protected.insert(ignore.clone());
    }
    for rule in &grammar.rules {
        if rule.is_terminal || matches!(rule.expr, GrammarExpr::ExprNFA(_)) {
            protected.insert(rule.name.clone());
        }
    }
    protected
}

fn inline_refs_in_expr(
    expr: &mut GrammarExpr,
    rule_exprs: &HashMap<String, GrammarExpr>,
    ref_counts: &HashMap<String, usize>,
    protected: &HashSet<String>,
    removed: &mut HashSet<String>,
    stats: &mut SimplifyStats,
) {
    match expr {
        GrammarExpr::Grouped(inner) => {
            inline_refs_in_expr(inner, rule_exprs, ref_counts, protected, removed, stats);
        }
        GrammarExpr::Sequence(parts) => {
            inline_sequence_refs(parts, rule_exprs, ref_counts, protected, removed, stats);
            for part in parts {
                inline_refs_in_expr(part, rule_exprs, ref_counts, protected, removed, stats);
            }
        }
        GrammarExpr::Choice(options) => {
            inline_choice_refs(options, rule_exprs, ref_counts, protected, removed, stats);
            for option in options {
                inline_refs_in_expr(option, rule_exprs, ref_counts, protected, removed, stats);
            }
        }
        GrammarExpr::Exclude { expr, exclude } => {
            inline_refs_in_expr(expr, rule_exprs, ref_counts, protected, removed, stats);
            inline_refs_in_expr(exclude, rule_exprs, ref_counts, protected, removed, stats);
        }
        GrammarExpr::Intersect { expr, intersect } => {
            inline_refs_in_expr(expr, rule_exprs, ref_counts, protected, removed, stats);
            inline_refs_in_expr(intersect, rule_exprs, ref_counts, protected, removed, stats);
        }
        GrammarExpr::Quantified(inner, Quantifier::Optional) | GrammarExpr::Quantified(inner, Quantifier::ZeroPlus) | GrammarExpr::Quantified(inner, Quantifier::OnePlus) => {
            inline_refs_in_expr(inner, rule_exprs, ref_counts, protected, removed, stats);
        }
        GrammarExpr::Quantified(expr, Quantifier::Range(_, _)) => {
            inline_refs_in_expr(expr, rule_exprs, ref_counts, protected, removed, stats);
        }
        GrammarExpr::SeparatedSequence {
            items, separator, ..
        } => {
            for (item, _) in items {
                inline_refs_in_expr(item, rule_exprs, ref_counts, protected, removed, stats);
            }
            inline_refs_in_expr(separator, rule_exprs, ref_counts, protected, removed, stats);
        }
        GrammarExpr::ExprNFA(expr_nfa) => {
            for symbol in &mut expr_nfa.symbols {
                inline_refs_in_expr(symbol, rule_exprs, ref_counts, protected, removed, stats);
            }
        }
        GrammarExpr::Ref(_)
        | GrammarExpr::Epsilon
        | GrammarExpr::Literal(_)
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte => {}
    }
}

fn inline_sequence_refs(
    parts: &mut Vec<GrammarExpr>,
    rule_exprs: &HashMap<String, GrammarExpr>,
    ref_counts: &HashMap<String, usize>,
    protected: &HashSet<String>,
    removed: &mut HashSet<String>,
    stats: &mut SimplifyStats,
) {
    if parts.len() <= 1 {
        return;
    }

    let mut out = Vec::new();
    for part in std::mem::take(parts) {
        if let Some((name, inlined_parts)) =
            single_use_ref_to_sequence(&part, rule_exprs, ref_counts, protected)
        {
            removed.insert(name);
            stats.inlined_rules += 1;
            out.extend(inlined_parts);
        } else {
            out.push(part);
        }
    }
    *parts = out;
}

fn inline_choice_refs(
    options: &mut Vec<GrammarExpr>,
    rule_exprs: &HashMap<String, GrammarExpr>,
    ref_counts: &HashMap<String, usize>,
    protected: &HashSet<String>,
    removed: &mut HashSet<String>,
    stats: &mut SimplifyStats,
) {
    if options.len() <= 1 {
        return;
    }

    let mut out = Vec::new();
    for option in std::mem::take(options) {
        if let Some((name, inlined_options)) =
            single_use_ref_to_choice(&option, rule_exprs, ref_counts, protected)
        {
            removed.insert(name);
            stats.inlined_rules += 1;
            out.extend(inlined_options);
        } else {
            out.push(option);
        }
    }
    *options = out;
}

fn single_use_ref_to_sequence(
    expr: &GrammarExpr,
    rule_exprs: &HashMap<String, GrammarExpr>,
    ref_counts: &HashMap<String, usize>,
    protected: &HashSet<String>,
) -> Option<(String, Vec<GrammarExpr>)> {
    let GrammarExpr::Ref(name) = expr else {
        return None;
    };
    if protected.contains(name) || ref_counts.get(name).copied().unwrap_or(0) != 1 {
        return None;
    }
    match rule_exprs.get(name)? {
        GrammarExpr::Sequence(parts) => Some((name.clone(), parts.clone())),
        GrammarExpr::Epsilon => Some((name.clone(), Vec::new())),
        _ => None,
    }
}

fn single_use_ref_to_choice(
    expr: &GrammarExpr,
    rule_exprs: &HashMap<String, GrammarExpr>,
    ref_counts: &HashMap<String, usize>,
    protected: &HashSet<String>,
) -> Option<(String, Vec<GrammarExpr>)> {
    let GrammarExpr::Ref(name) = expr else {
        return None;
    };
    if protected.contains(name) || ref_counts.get(name).copied().unwrap_or(0) != 1 {
        return None;
    }
    match rule_exprs.get(name)? {
        GrammarExpr::Choice(options) => Some((name.clone(), options.clone())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::simplify_named_grammar;
    use crate::grammar::ast::{GrammarExpr, Quantifier, NamedGrammar, NamedRule};

    fn lit(s: &str) -> GrammarExpr {
        GrammarExpr::Literal(s.as_bytes().to_vec())
    }

    fn nt(name: &str, expr: GrammarExpr) -> NamedRule {
        NamedRule {
            name: name.to_string(),
            expr,
            is_terminal: false,
            is_internal: false,
        }
    }

    #[test]
    fn flattens_singleton_and_nested_sequences_and_choices() {
        let mut grammar = NamedGrammar {
            rules: vec![nt(
                "start",
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Sequence(vec![lit("a")]),
                    GrammarExpr::Choice(vec![GrammarExpr::Choice(vec![lit("b"), lit("c")])]),
                ]),
            )],
            start: "start".into(),
            ignore: None,
        };

        simplify_named_grammar(&mut grammar);

        assert_eq!(
            grammar.rules[0].expr,
            GrammarExpr::Sequence(vec![
                lit("a"),
                GrammarExpr::Choice(vec![lit("b"), lit("c")]),
            ])
        );
    }

    #[test]
    fn inlines_single_use_sequence_rule_into_longer_sequence() {
        let mut grammar = NamedGrammar {
            rules: vec![
                nt("start", GrammarExpr::Sequence(vec![lit("x"), GrammarExpr::Ref("mid".into())])),
                nt("mid", GrammarExpr::Sequence(vec![lit("a"), lit("b")])),
            ],
            start: "start".into(),
            ignore: None,
        };

        let stats = simplify_named_grammar(&mut grammar);

        assert_eq!(stats.inlined_rules, 1);
        assert_eq!(grammar.rules.len(), 1);
        assert_eq!(
            grammar.rules[0].expr,
            GrammarExpr::Sequence(vec![lit("x"), lit("a"), lit("b")])
        );
    }

    #[test]
    fn does_not_inline_single_use_sequence_rule_as_whole_rule() {
        let mut grammar = NamedGrammar {
            rules: vec![
                nt("start", GrammarExpr::Ref("mid".into())),
                nt("mid", GrammarExpr::Sequence(vec![lit("a"), lit("b")])),
            ],
            start: "start".into(),
            ignore: None,
        };

        simplify_named_grammar(&mut grammar);

        assert_eq!(grammar.rules.len(), 2);
    }

    #[test]
    fn inlines_single_use_choice_rule_into_longer_choice() {
        let mut grammar = NamedGrammar {
            rules: vec![
                nt("start", GrammarExpr::Choice(vec![lit("x"), GrammarExpr::Ref("alts".into())])),
                nt("alts", GrammarExpr::Choice(vec![lit("a"), lit("b")])),
            ],
            start: "start".into(),
            ignore: None,
        };

        let stats = simplify_named_grammar(&mut grammar);

        assert_eq!(stats.inlined_rules, 1);
        assert_eq!(grammar.rules.len(), 1);
        assert_eq!(
            grammar.rules[0].expr,
            GrammarExpr::Choice(vec![lit("x"), lit("a"), lit("b")])
        );
    }

    #[test]
    fn simplifies_safe_repeat_shapes() {
        let mut grammar = NamedGrammar {
            rules: vec![nt(
                "start",
                GrammarExpr::Quantified(Box::new(GrammarExpr::Quantified(Box::new(lit("a")), Quantifier::OnePlus)), Quantifier::Optional),
            )],
            start: "start".into(),
            ignore: None,
        };

        simplify_named_grammar(&mut grammar);

        assert_eq!(grammar.rules[0].expr, GrammarExpr::Quantified(Box::new(lit("a")), Quantifier::ZeroPlus));
    }
}
