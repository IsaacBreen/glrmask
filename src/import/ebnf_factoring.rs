//! EBNF choice factoring on parsed `GrammarExpr` rules.

use std::collections::{HashMap, HashSet};

use crate::grammar::ast::{GrammarExpr, NamedGrammar};

fn contains_regex_features(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::CharClass { .. } | GrammarExpr::RawRegex(_) | GrammarExpr::AnyByte
        | GrammarExpr::CompiledTerminal { .. } => true,
        GrammarExpr::Literal(_) | GrammarExpr::Ref(_) => false,
        GrammarExpr::Sequence(exprs) | GrammarExpr::Choice(exprs) => {
            exprs.iter().any(contains_regex_features)
        }
        GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner) => contains_regex_features(inner),
    }
}

pub(crate) fn factor_named_grammar(grammar: NamedGrammar) -> NamedGrammar {
    NamedGrammar {
        rules: factor_grammar_rules(grammar.rules),
        start: grammar.start,
    }
}

pub(crate) fn factor_grammar_rules(rules: Vec<(String, GrammarExpr)>) -> Vec<(String, GrammarExpr)> {
    ChoiceFactorer::new(rules).factor_all()
}

struct ChoiceFactorer {
    rules: HashMap<String, GrammarExpr>,
    rule_order: Vec<String>,
    recursive_rules: HashSet<String>,
    new_rules: Vec<(String, GrammarExpr)>,
    helper_counter: usize,
    factor_cache: HashMap<Vec<GrammarExpr>, String>,
}

impl ChoiceFactorer {
    fn new(rules: Vec<(String, GrammarExpr)>) -> Self {
        let rule_order = rules.iter().map(|(name, _)| name.clone()).collect::<Vec<_>>();
        let rules = rules.into_iter().collect::<HashMap<_, _>>();
        let recursive_rules = Self::find_recursive_rules(&rules);

        Self {
            rules,
            rule_order,
            recursive_rules,
            new_rules: Vec::new(),
            helper_counter: 0,
            factor_cache: HashMap::new(),
        }
    }

    fn factor_all(mut self) -> Vec<(String, GrammarExpr)> {
        for name in self.rule_order.clone() {
            let expr = self
                .rules
                .get(&name)
                .cloned()
                .expect("rule order and rule map should stay aligned");

            let should_factor = name.starts_with('_')
                && !name.starts_with("_json")
                && !contains_regex_features(&expr);

            let factored_expr = if should_factor {
                self.factor_expr(expr, &name)
            } else {
                expr
            };

            self.new_rules.push((name, factored_expr));
        }

        self.new_rules
    }

    fn factor_expr(&mut self, expr: GrammarExpr, context_name: &str) -> GrammarExpr {
        match expr {
            GrammarExpr::Choice(alternatives) if alternatives.len() > 1 => {
                self.factor_choice(alternatives, context_name)
            }
            GrammarExpr::Sequence(exprs) => GrammarExpr::Sequence(
                exprs
                    .into_iter()
                    .map(|expr| self.factor_expr(expr, context_name))
                    .collect(),
            ),
            GrammarExpr::Optional(expr) => {
                GrammarExpr::Optional(Box::new(self.factor_expr(*expr, context_name)))
            }
            GrammarExpr::Repeat(expr) => {
                GrammarExpr::Repeat(Box::new(self.factor_expr(*expr, context_name)))
            }
            GrammarExpr::RepeatOne(expr) => {
                GrammarExpr::RepeatOne(Box::new(self.factor_expr(*expr, context_name)))
            }
            other => other,
        }
    }

    fn factor_choice(&mut self, alternatives: Vec<GrammarExpr>, context_name: &str) -> GrammarExpr {
        if alternatives.len() < 2 {
            return alternatives
                .into_iter()
                .next()
                .unwrap_or_else(|| GrammarExpr::Sequence(vec![]));
        }

        let mut safe_alts = Vec::new();
        let mut unsafe_alts = Vec::new();

        for alternative in alternatives {
            if self.is_safe_alternative(&alternative) {
                safe_alts.push(alternative);
            } else {
                unsafe_alts.push(alternative);
            }
        }

        let mut final_choices = Vec::new();

        if safe_alts.len() > 1 {
            let helper_name = self.create_helper_rule(safe_alts, format!("{}_safe", context_name));
            final_choices.push(GrammarExpr::Ref(helper_name));
        } else if let Some(safe_alt) = safe_alts.into_iter().next() {
            final_choices.push(safe_alt);
        }

        let tail_groups = self.group_by_tail(&unsafe_alts);
        for (tail, heads) in tail_groups {
            if heads.len() > 1 || self.is_complex_head(&heads[0]) {
                let helper_name = if heads.len() == 1 {
                    self.create_helper_rule(heads.clone(), format!("{}_key", context_name))
                } else {
                    self.create_helper_rule(heads.clone(), format!("{}_keys", context_name))
                };

                final_choices.push(GrammarExpr::Sequence(vec![
                    GrammarExpr::Ref(helper_name),
                    GrammarExpr::Literal(b":".to_vec()),
                    tail,
                ]));
            } else {
                final_choices.push(GrammarExpr::Sequence(vec![
                    heads.into_iter().next().expect("single head should exist"),
                    GrammarExpr::Literal(b":".to_vec()),
                    tail,
                ]));
            }
        }

        for alternative in &unsafe_alts {
            if !self.has_tail_pattern(alternative) {
                final_choices.push(alternative.clone());
            }
        }

        if final_choices.is_empty() {
            GrammarExpr::Sequence(vec![])
        } else if final_choices.len() == 1 {
            final_choices.into_iter().next().expect("single factored choice should exist")
        } else {
            GrammarExpr::Choice(final_choices)
        }
    }

    fn is_safe_alternative(&self, expr: &GrammarExpr) -> bool {
        let refs = self.collect_refs(expr);
        !refs.iter().any(|name| self.recursive_rules.contains(name))
    }

    fn collect_refs(&self, expr: &GrammarExpr) -> HashSet<String> {
        let mut refs = HashSet::new();
        self.collect_refs_impl(expr, &mut refs);
        refs
    }

    fn collect_refs_impl(&self, expr: &GrammarExpr, refs: &mut HashSet<String>) {
        let _ = self;
        match expr {
            GrammarExpr::Ref(name) => {
                refs.insert(name.clone());
            }
            GrammarExpr::Sequence(exprs) | GrammarExpr::Choice(exprs) => {
                for expr in exprs {
                    self.collect_refs_impl(expr, refs);
                }
            }
            GrammarExpr::Optional(expr)
            | GrammarExpr::Repeat(expr)
            | GrammarExpr::RepeatOne(expr) => self.collect_refs_impl(expr, refs),
            GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::AnyByte
            | GrammarExpr::CompiledTerminal { .. } => {}
        }
    }

    fn group_by_tail(&self, alternatives: &[GrammarExpr]) -> HashMap<GrammarExpr, Vec<GrammarExpr>> {
        let mut groups = HashMap::<GrammarExpr, Vec<GrammarExpr>>::new();

        for alternative in alternatives {
            if let Some((head, tail)) = self.extract_tail_pattern(alternative) {
                if self.is_safe_alternative(&head) {
                    groups.entry(tail).or_default().push(head);
                }
            }
        }

        groups
    }

    fn has_tail_pattern(&self, expr: &GrammarExpr) -> bool {
        self.extract_tail_pattern(expr).is_some()
    }

    fn extract_tail_pattern(&self, expr: &GrammarExpr) -> Option<(GrammarExpr, GrammarExpr)> {
        let _ = self;
        if let GrammarExpr::Sequence(parts) = expr {
            if parts.len() >= 3 {
                let colon_index = parts.len() - 2;
                if let GrammarExpr::Literal(literal) = &parts[colon_index] {
                    if literal == b":" {
                        let head = if colon_index == 1 {
                            parts[0].clone()
                        } else {
                            GrammarExpr::Sequence(parts[..colon_index].to_vec())
                        };
                        let tail = parts[parts.len() - 1].clone();
                        if matches!(tail, GrammarExpr::Ref(_)) {
                            return Some((head, tail));
                        }
                    }
                }
            }
        }
        None
    }

    fn is_complex_head(&self, expr: &GrammarExpr) -> bool {
        let _ = self;
        match expr {
            GrammarExpr::Sequence(parts) => parts.len() > 2,
            GrammarExpr::Choice(_) => true,
            GrammarExpr::Optional(_) | GrammarExpr::Repeat(_) | GrammarExpr::RepeatOne(_) => true,
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
            || self.new_rules.iter().any(|(name, _)| name == &helper_name)
        {
            helper_name = format!("{}_{}", base, collision_index);
            collision_index += 1;
        }

        let helper_expr = if alternatives.len() == 1 {
            alternatives[0].clone()
        } else {
            GrammarExpr::Choice(alternatives.clone())
        };

        self.new_rules.push((helper_name.clone(), helper_expr));
        self.factor_cache.insert(alternatives, helper_name.clone());
        self.helper_counter += 1;
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
            GrammarExpr::Optional(expr)
            | GrammarExpr::Repeat(expr)
            | GrammarExpr::RepeatOne(expr) => Self::collect_refs_static(expr, refs),
            GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::AnyByte
            | GrammarExpr::CompiledTerminal { .. } => {}
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

    #[test]
    fn test_factor_safe_internal_choice_into_helper() {
        let factored = factor_grammar_rules(vec![
            ("start".to_string(), GrammarExpr::Ref("_value".to_string())),
            (
                "_value".to_string(),
                GrammarExpr::Choice(vec![lit(b"a"), lit(b"b")]),
            ),
        ]);

        let value_rule = factored
            .iter()
            .find(|(name, _)| name == "_value")
            .expect("factored _value rule should exist");
        let helper_name = match &value_rule.1 {
            GrammarExpr::Ref(name) => name.clone(),
            other => panic!("expected _value to be replaced by helper ref, got {:?}", other),
        };

        let helper_rule = factored
            .iter()
            .find(|(name, _)| name == &helper_name)
            .expect("helper rule should be created for safe alternatives");
        assert!(matches!(
            &helper_rule.1,
            GrammarExpr::Choice(options) if options == &vec![lit(b"a"), lit(b"b")]
        ));
    }

    #[test]
    fn test_factor_recursive_key_tail_choice_into_key_helper() {
        let factored = factor_grammar_rules(vec![
            ("start".to_string(), GrammarExpr::Ref("_pair".to_string())),
            (
                "_pair".to_string(),
                GrammarExpr::Choice(vec![
                    GrammarExpr::Sequence(vec![lit(b"a"), lit(b":"), GrammarExpr::Ref("_pair".to_string())]),
                    GrammarExpr::Sequence(vec![lit(b"b"), lit(b":"), GrammarExpr::Ref("_pair".to_string())]),
                ]),
            ),
        ]);

        let pair_rule = factored
            .iter()
            .find(|(name, _)| name == "_pair")
            .expect("factored recursive rule should exist");
        let helper_name = match &pair_rule.1 {
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
            .find(|(name, _)| name == &helper_name)
            .expect("key helper rule should be created");
        assert!(matches!(
            &helper_rule.1,
            GrammarExpr::Choice(options) if options == &vec![lit(b"a"), lit(b"b")]
        ));
    }

    #[test]
    fn test_skip_non_internal_rules() {
        let original = vec![(
            "start".to_string(),
            GrammarExpr::Choice(vec![lit(b"a"), lit(b"b")]),
        )];
        let factored = factor_grammar_rules(original.clone());
        assert_eq!(factored, original);
    }
}