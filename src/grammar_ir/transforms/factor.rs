//! Grammar IR transform.

//! Choice factoring on parsed `GrammarExpr` rules.
//!
//! Factoring is a grammar optimization: it extracts common sub-choices into
//! helper rules, reducing parser state counts and improving DWA minimization.
//! This operates on the `NamedGrammar` AST before lowering.
//!
//! The decision of which rules to factor uses the grammar's explicit `terminals`
//! set rather than name-prefix heuristics.

use std::collections::{HashMap, HashSet};

use crate::grammar_ir::ast::{GrammarExpr, NamedGrammar, NamedRule};

fn contains_regex_features(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::Grouped(inner) => contains_regex_features(inner),
        GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte => true,
        GrammarExpr::Literal(_) | GrammarExpr::Ref(_) | GrammarExpr::Epsilon => false,
        GrammarExpr::Sequence(exprs) | GrammarExpr::Choice(exprs) => {
            exprs.iter().any(contains_regex_features)
        }
        GrammarExpr::Exclude { expr, exclude } => {
            contains_regex_features(expr) || contains_regex_features(exclude)
        }
        GrammarExpr::Intersect { expr, intersect } => {
            contains_regex_features(expr) || contains_regex_features(intersect)
        }
        GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner)
        | GrammarExpr::RepeatRange { expr: inner, .. } => contains_regex_features(inner),
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
            items.iter().any(|(item, _)| contains_regex_features(item))
                || contains_regex_features(separator)
        }
        GrammarExpr::ExprNFA(expr_nfa) => expr_nfa.symbols.iter().any(contains_regex_features),
    }
}

fn epsilon_expr() -> GrammarExpr {
    GrammarExpr::Sequence(Vec::new())
}

fn colon_literal() -> GrammarExpr {
    GrammarExpr::Literal(b":".to_vec())
}

pub fn factor_named_grammar(grammar: NamedGrammar) -> NamedGrammar {
    let terminal_names = grammar.terminal_names_set();
    let rules = ChoiceFactorer::new(grammar.rules, &terminal_names).factor_all();
    NamedGrammar {
        rules,
        start: grammar.start,
        ignore: grammar.ignore,
    }
}

struct ChoiceFactorer {
    rules: HashMap<String, GrammarExpr>,
    ordered_rules: Vec<(String, bool, bool)>,
    terminals: HashSet<String>,
    recursive_rules: HashSet<String>,
    new_rules: Vec<NamedRule>,
    factor_cache: HashMap<Vec<GrammarExpr>, String>,
}

impl ChoiceFactorer {
    fn new(rules: Vec<NamedRule>, terminals: &HashSet<String>) -> Self {
        let mut ordered_rules: Vec<(String, bool, bool)> = Vec::new();
        let mut seen_names = HashSet::<String>::new();
        let mut rules_by_name: HashMap<String, GrammarExpr> = HashMap::new();

        for rule in rules {
            if seen_names.insert(rule.name.clone()) {
                ordered_rules.push((rule.name.clone(), rule.is_terminal, rule.is_internal));
            }

            rules_by_name
                .entry(rule.name)
                .and_modify(|existing| {
                    let merged = match existing.clone() {
                        GrammarExpr::Choice(mut options) => {
                            options.push(rule.expr.clone());
                            GrammarExpr::Choice(options)
                        }
                        previous => GrammarExpr::Choice(vec![previous, rule.expr.clone()]),
                    };
                    *existing = merged;
                })
                .or_insert(rule.expr);
        }

        let recursive_rules = Self::find_recursive_rules(&rules_by_name);

        Self {
            rules: rules_by_name,
            ordered_rules,
            terminals: terminals.clone(),
            recursive_rules,
            new_rules: Vec::new(),
            factor_cache: HashMap::new(),
        }
    }

    fn factor_all(mut self) -> Vec<NamedRule> {
        for (name, is_terminal, is_internal) in self.ordered_rules.clone() {
            let expr = self
                .rules
                .get(&name)
                .cloned()
                .expect("rule order and rule map should stay aligned");

            let factored_expr = if self.should_factor_rule(&name, &expr) {
                self.factor_expr(expr, &name)
            } else {
                expr
            };

            self.new_rules.push(NamedRule {
                name,
                expr: factored_expr,
                is_terminal,
                is_internal,
            });
        }

        self.new_rules
    }

    fn should_factor_rule(&self, name: &str, expr: &GrammarExpr) -> bool {
        name.starts_with('_')
            && !self.terminals.contains(name)
            && !contains_regex_features(expr)
    }

    fn factor_expr(&mut self, expr: GrammarExpr, rule_name: &str) -> GrammarExpr {
        match expr {
            GrammarExpr::Choice(alternatives) if alternatives.len() > 1 => {
                self.factor_choice(alternatives, rule_name)
            }
            GrammarExpr::Sequence(exprs) => GrammarExpr::Sequence(
                exprs
                    .into_iter()
                    .map(|expr| self.factor_expr(expr, rule_name))
                    .collect(),
            ),
            GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
                expr: Box::new(self.factor_expr(*expr, rule_name)),
                exclude: Box::new(self.factor_expr(*exclude, rule_name)),
            },
            GrammarExpr::Optional(expr) => {
                GrammarExpr::Optional(Box::new(self.factor_expr(*expr, rule_name)))
            }
            GrammarExpr::Repeat(expr) => {
                GrammarExpr::Repeat(Box::new(self.factor_expr(*expr, rule_name)))
            }
            GrammarExpr::RepeatOne(expr) => {
                GrammarExpr::RepeatOne(Box::new(self.factor_expr(*expr, rule_name)))
            }
            GrammarExpr::RepeatRange { expr, min, max } => GrammarExpr::RepeatRange {
                expr: Box::new(self.factor_expr(*expr, rule_name)),
                min,
                max,
            },
            other => other,
        }
    }

    fn factor_choice(&mut self, alternatives: Vec<GrammarExpr>, rule_name: &str) -> GrammarExpr {
        if alternatives.len() < 2 {
            return alternatives.into_iter().next().unwrap_or_else(epsilon_expr);
        }

        let (mut safe_alternatives, recursive_alternatives): (Vec<_>, Vec<_>) = alternatives
            .into_iter()
            .partition(|alternative| self.is_safe_alternative(alternative));

        let mut final_choices = Vec::new();

        if safe_alternatives.len() > 1 {
            let helper_name = self.create_helper_rule(
                safe_alternatives,
                format!("{}_safe", rule_name),
            );
            final_choices.push(GrammarExpr::Ref(helper_name));
        } else if let Some(safe_alternative) = safe_alternatives.pop() {
            final_choices.push(safe_alternative);
        }

        let tail_groups = self.group_by_tail(&recursive_alternatives);
        for (tail, heads) in tail_groups {
            if heads.len() > 1 || Self::is_complex_head(&heads[0]) {
                let helper_name = if heads.len() == 1 {
                    self.create_helper_rule(heads.clone(), format!("{}_key", rule_name))
                } else {
                    self.create_helper_rule(heads.clone(), format!("{}_keys", rule_name))
                };

                final_choices.push(GrammarExpr::Sequence(vec![
                    GrammarExpr::Ref(helper_name),
                    colon_literal(),
                    tail,
                ]));
            } else {
                final_choices.push(GrammarExpr::Sequence(vec![
                    heads.into_iter().next().expect("single head should exist"),
                    colon_literal(),
                    tail,
                ]));
            }
        }

        for alternative in &recursive_alternatives {
            if !Self::has_tail_pattern(alternative) {
                final_choices.push(alternative.clone());
            }
        }

        match final_choices.len() {
            0 => epsilon_expr(),
            1 => final_choices
                .into_iter()
                .next()
                .expect("single factored choice should exist"),
            _ => GrammarExpr::Choice(final_choices),
        }
    }

    fn is_safe_alternative(&self, expr: &GrammarExpr) -> bool {
        let refs = Self::collect_refs(expr);
        !refs.iter().any(|name| self.recursive_rules.contains(name))
    }

    fn collect_refs(expr: &GrammarExpr) -> HashSet<String> {
        let mut refs = HashSet::new();
        Self::collect_refs_impl(expr, &mut refs);
        refs
    }

    fn collect_refs_impl(expr: &GrammarExpr, refs: &mut HashSet<String>) {
        match expr {
            GrammarExpr::Grouped(inner) => {
                Self::collect_refs_impl(inner, refs);
            }
            GrammarExpr::Ref(name) => {
                refs.insert(name.clone());
            }
            GrammarExpr::Sequence(exprs) | GrammarExpr::Choice(exprs) => {
                for expr in exprs {
                    Self::collect_refs_impl(expr, refs);
                }
            }
            GrammarExpr::Exclude { expr, exclude } => {
                Self::collect_refs_impl(expr, refs);
                Self::collect_refs_impl(exclude, refs);
            }
            GrammarExpr::Intersect { expr, intersect } => {
                Self::collect_refs_impl(expr, refs);
                Self::collect_refs_impl(intersect, refs);
            }
            GrammarExpr::Optional(expr)
            | GrammarExpr::Repeat(expr)
            | GrammarExpr::RepeatOne(expr)
            | GrammarExpr::RepeatRange { expr, .. } => Self::collect_refs_impl(expr, refs),
            GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::Epsilon
            | GrammarExpr::AnyByte => {}
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                for (item, _) in items {
                    Self::collect_refs_impl(item, refs);
                }
                Self::collect_refs_impl(separator, refs);
            }
            GrammarExpr::ExprNFA(expr_nfa) => {
                for symbol in &expr_nfa.symbols {
                    Self::collect_refs_impl(symbol, refs);
                }
            }
        }
    }

    fn group_by_tail(&self, alternatives: &[GrammarExpr]) -> HashMap<GrammarExpr, Vec<GrammarExpr>> {
        let mut groups = HashMap::<GrammarExpr, Vec<GrammarExpr>>::new();

        for alternative in alternatives {
            if let Some((head, tail)) = Self::extract_tail_pattern(alternative) {
                if self.is_safe_alternative(&head) {
                    groups.entry(tail).or_default().push(head);
                }
            }
        }

        groups
    }

    fn has_tail_pattern(expr: &GrammarExpr) -> bool {
        Self::extract_tail_pattern(expr).is_some()
    }

    fn extract_tail_pattern(expr: &GrammarExpr) -> Option<(GrammarExpr, GrammarExpr)> {
        let GrammarExpr::Sequence(parts) = expr else {
            return None;
        };
        if parts.len() < 3 {
            return None;
        }

        let (tail, prefix) = parts.split_last()?;
        let (separator, head_parts) = prefix.split_last()?;
        match separator {
            GrammarExpr::Literal(literal) if literal == b":" && matches!(tail, GrammarExpr::Ref(_)) => {
                let head = if head_parts.len() == 1 {
                    head_parts[0].clone()
                } else {
                    GrammarExpr::Sequence(head_parts.to_vec())
                };
                Some((head, tail.clone()))
            }
            _ => None,
        }
    }

    fn is_complex_head(expr: &GrammarExpr) -> bool {
        match expr {
            GrammarExpr::Grouped(inner) => Self::is_complex_head(inner),
            GrammarExpr::Sequence(parts) => parts.len() > 2,
            GrammarExpr::Choice(_) => true,
            GrammarExpr::Exclude { .. } => true,
            GrammarExpr::Optional(_)
            | GrammarExpr::Repeat(_)
            | GrammarExpr::RepeatOne(_)
            | GrammarExpr::RepeatRange { .. } => true,
            _ => false,
        }
    }

    fn create_helper_rule(&mut self, alternatives: Vec<GrammarExpr>, base_name: String) -> String {
        if let Some(existing) = self.factor_cache.get(&alternatives) {
            return existing.clone();
        }

        let mut helper_name = format!("__{}", base_name);
        let base = helper_name.clone();
        let mut collision_index = 1;

        while self.rules.contains_key(&helper_name)
            || self.new_rules.iter().any(|r| r.name == helper_name)
        {
            helper_name = format!("{}_{}", base, collision_index);
            collision_index += 1;
        }

        let helper_expr = if alternatives.len() == 1 {
            alternatives[0].clone()
        } else {
            GrammarExpr::Choice(alternatives.clone())
        };

        self.new_rules.push(NamedRule {
            name: helper_name.clone(),
            expr: helper_expr,
            is_terminal: false,
            is_internal: false,
        });
        self.factor_cache.insert(alternatives, helper_name.clone());
        helper_name
    }

    fn find_recursive_rules(rules: &HashMap<String, GrammarExpr>) -> HashSet<String> {
        let mut recursive = HashSet::new();
        let mut deps = HashMap::<String, Vec<String>>::new();

        for (name, expr) in rules {
            let mut refs = HashSet::new();
            Self::collect_refs_static(expr, &mut refs);
            deps.insert(name.clone(), refs.into_iter().collect());
        }

        for start in rules.keys() {
            if Self::can_reach_self(start, &deps) {
                recursive.insert(start.clone());
            }
        }

        recursive
    }

    fn collect_refs_static(expr: &GrammarExpr, refs: &mut HashSet<String>) {
        match expr {
            GrammarExpr::Grouped(inner) => {
                Self::collect_refs_static(inner, refs);
            }
            GrammarExpr::Ref(name) => {
                refs.insert(name.clone());
            }
            GrammarExpr::Sequence(exprs) | GrammarExpr::Choice(exprs) => {
                for expr in exprs {
                    Self::collect_refs_static(expr, refs);
                }
            }
            GrammarExpr::Exclude { expr, exclude } => {
                Self::collect_refs_static(expr, refs);
                Self::collect_refs_static(exclude, refs);
            }
            GrammarExpr::Intersect { expr, intersect } => {
                Self::collect_refs_static(expr, refs);
                Self::collect_refs_static(intersect, refs);
            }
            GrammarExpr::Optional(expr)
            | GrammarExpr::Repeat(expr)
            | GrammarExpr::RepeatOne(expr)
            | GrammarExpr::RepeatRange { expr, .. } => Self::collect_refs_static(expr, refs),
            GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::Epsilon
            | GrammarExpr::AnyByte => {}
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                for (item, _) in items {
                    Self::collect_refs_static(item, refs);
                }
                Self::collect_refs_static(separator, refs);
            }
            GrammarExpr::ExprNFA(expr_nfa) => {
                for symbol in &expr_nfa.symbols {
                    Self::collect_refs_static(symbol, refs);
                }
            }
        }
    }

    fn can_reach_self(start: &str, deps: &HashMap<String, Vec<String>>) -> bool {
        let mut visited = HashSet::from([start.to_string()]);
        let mut stack = vec![start.to_string()];

        while let Some(node) = stack.pop() {
            if let Some(neighbors) = deps.get(&node) {
                for neighbor in neighbors {
                    if neighbor == start {
                        return true;
                    }
                    if visited.insert(neighbor.clone()) {
                        stack.push(neighbor.clone());
                    }
                }
            }
        }

        false
    }
}
