//! Choice factoring on parsed `GrammarExpr` rules.
//!
//! Factoring is a grammar optimization: it extracts common sub-choices into
//! helper rules, reducing parser state counts and improving DWA minimization.
//! This operates on the `NamedGrammar` AST before lowering.
//!
//! The decision of which rules to factor uses the grammar's explicit `terminals`
//! set rather than name-prefix heuristics.

use std::collections::{HashMap, HashSet};

use super::ast::{GrammarExpr, NamedGrammar, NamedRule};

fn contains_regex_features(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::TerminalExpr(_)
        | GrammarExpr::AnyByte => true,
        GrammarExpr::Literal(_) | GrammarExpr::Ref(_) => false,
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
        GrammarExpr::SeparatedSequence { items, separator } => {
            items.iter().any(|(item, _)| contains_regex_features(item))
                || contains_regex_features(separator)
        }
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

#[cfg(test)]
pub fn factor_grammar_rules(
    rules: Vec<NamedRule>,
    terminals: &HashSet<String>,
) -> Vec<NamedRule> {
    ChoiceFactorer::new(rules, terminals).factor_all()
}

struct ChoiceFactorer {
    rules: HashMap<String, GrammarExpr>,
    ordered_rules: Vec<(String, bool)>,
    terminals: HashSet<String>,
    recursive_rules: HashSet<String>,
    new_rules: Vec<NamedRule>,
    factor_cache: HashMap<Vec<GrammarExpr>, String>,
}

impl ChoiceFactorer {
    fn new(rules: Vec<NamedRule>, terminals: &HashSet<String>) -> Self {
        let mut ordered_rules: Vec<(String, bool)> = Vec::new();
        let mut seen_names = HashSet::<String>::new();
        let mut rules_by_name: HashMap<String, GrammarExpr> = HashMap::new();

        for rule in rules {
            if seen_names.insert(rule.name.clone()) {
                ordered_rules.push((rule.name.clone(), rule.is_terminal));
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
        for (name, is_terminal) in self.ordered_rules.clone() {
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
                is_internal: false,
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
            | GrammarExpr::TerminalExpr(_)
            | GrammarExpr::AnyByte => {}
            GrammarExpr::SeparatedSequence { items, separator } => {
                for (item, _) in items {
                    Self::collect_refs_impl(item, refs);
                }
                Self::collect_refs_impl(separator, refs);
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
            | GrammarExpr::TerminalExpr(_)
            | GrammarExpr::AnyByte => {}
            GrammarExpr::SeparatedSequence { items, separator } => {
                for (item, _) in items {
                    Self::collect_refs_static(item, refs);
                }
                Self::collect_refs_static(separator, refs);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(bytes: &[u8]) -> GrammarExpr {
        GrammarExpr::Literal(bytes.to_vec())
    }

    fn terminals(names: &[&str]) -> HashSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn nt(name: &str, expr: GrammarExpr) -> NamedRule {
        NamedRule { name: name.to_string(), expr, is_terminal: false, is_internal: false }
    }

    fn term(name: &str, expr: GrammarExpr) -> NamedRule {
        NamedRule { name: name.to_string(), expr, is_terminal: true, is_internal: false }
    }

    #[test]
    fn test_factor_nonterminal_choice_into_helper() {
        let factored = factor_grammar_rules(
            vec![
                nt("start", GrammarExpr::Ref("_value".to_string())),
                nt("_value", GrammarExpr::Choice(vec![lit(b"a"), lit(b"b")])),
            ],
            &terminals(&[]),
        );

        let value_rule = factored
            .iter()
            .find(|r| r.name == "_value")
            .expect("factored _value rule should exist");
        let helper_name = match &value_rule.expr {
            GrammarExpr::Ref(name) => name.clone(),
            other => panic!("expected _value to be replaced by helper ref, got {:?}", other),
        };

        let helper_rule = factored
            .iter()
            .find(|r| r.name == helper_name)
            .expect("helper rule should be created for safe alternatives");
        assert!(matches!(
            &helper_rule.expr,
            GrammarExpr::Choice(options) if options == &vec![lit(b"a"), lit(b"b")]
        ));
    }

    #[test]
    fn test_factor_recursive_key_tail_choice_into_key_helper() {
        let factored = factor_grammar_rules(
            vec![
                nt("start", GrammarExpr::Ref("_pair".to_string())),
                nt("_pair", GrammarExpr::Choice(vec![
                    GrammarExpr::Sequence(vec![lit(b"a"), lit(b":"), GrammarExpr::Ref("_pair".to_string())]),
                    GrammarExpr::Sequence(vec![lit(b"b"), lit(b":"), GrammarExpr::Ref("_pair".to_string())]),
                ])),
            ],
            &terminals(&[]),
        );

        let pair_rule = factored
            .iter()
            .find(|r| r.name == "_pair")
            .expect("factored recursive rule should exist");
        let helper_name = match &pair_rule.expr {
            GrammarExpr::Sequence(parts)
                if parts.len() == 3
                    && parts[1] == lit(b":")
                    && parts[2] == GrammarExpr::Ref("_pair".to_string()) =>
            {
                match &parts[0] {
                    GrammarExpr::Ref(name) => name.clone(),
                    other => panic!("expected helper ref in factored head position, got {:?}", other),
                }
            }
            other => panic!("expected key-tail factored sequence, got {:?}", other),
        };

        let helper_rule = factored
            .iter()
            .find(|r| r.name == helper_name)
            .expect("key helper rule should be created");
        assert!(matches!(
            &helper_rule.expr,
            GrammarExpr::Choice(options) if options == &vec![lit(b"a"), lit(b"b")]
        ));
    }

    #[test]
    fn test_factor_named_grammar_merges_repeated_rule_definitions_into_choice() {
        let factored = factor_grammar_rules(
            vec![
                nt("start", GrammarExpr::Ref("a".to_string())),
                nt("a", GrammarExpr::Sequence(vec![])),
                nt("a", lit(b"f")),
            ],
            &terminals(&[]),
        );

        let a_rule = factored
            .iter()
            .find(|r| r.name == "a")
            .expect("merged rule 'a' should exist");
        assert!(matches!(
            &a_rule.expr,
            GrammarExpr::Choice(options)
                if options == &vec![GrammarExpr::Sequence(vec![]), lit(b"f")]
        ));
        assert_eq!(factored.iter().filter(|r| r.name == "a").count(), 1);
    }

    #[test]
    fn test_skip_terminal_rules() {
        // Terminal rules should not be factored regardless of name conventions
        let original = vec![
            term("_value", GrammarExpr::Choice(vec![lit(b"a"), lit(b"b")])),
        ];
        let factored = factor_grammar_rules(original.clone(), &terminals(&["_value"]));
        assert_eq!(factored, original);
    }

    #[test]
    fn test_skip_rules_with_regex_features() {
        let original = vec![
            nt("word", GrammarExpr::Choice(vec![
                GrammarExpr::CharClass { def: "a-z".to_string(), negate: false, utf8: false },
                lit(b"x"),
            ])),
        ];
        let factored = factor_grammar_rules(original.clone(), &terminals(&[]));
        assert_eq!(factored, original);
    }
}
