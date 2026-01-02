//! EBNF Choice Factoring
//!
//! This module implements choice factoring on parsed grammar rules (GrammarExpr),
//! improving compilation time by reducing the number of alternatives in large choice expressions.
//!
//! ## Key Improvements Over String-Based Implementation
//!
//! The previous implementation operated on raw EBNF text using regex parsing. This new version:
//! - Works on structured `GrammarExpr` data instead of strings
//! - Uses proper AST traversal instead of regex-based text manipulation
//! - Performs principled dependency analysis to detect recursive rules
//! - Handles complex nested expressions correctly
//!
//! ## Factoring Strategy
//!
//! For rules with large choice expressions (e.g., 100+ alternatives), factoring groups alternatives
//! to reduce parser complexity:
//!
//! 1. **Safe alternatives** (no recursive references) are grouped into helper rules
//! 2. **Key-value patterns** like `"key" ':' tail` are factored by grouping keys
//! 3. **Recursive base types** are left alone (empirically better for DFA minimization)
//!
//! ## Exception: `_json` Prefix
//!
//! Rules with the `_json` prefix are not factored. This is based on:
//! - Stable naming convention from the JSON schema converter
//! - Empirical validation: excluding these rules produces significantly better DFA minimization
//!   (e.g., 66K→13K states vs 73K→16K states on ApolloRouter schema)
//! - These rules represent fundamental JSON structural types (_json_value, _json_object, etc.)
//!
//! While this is a string-based check, it's well-justified and based on observable behavior
//! rather than arbitrary decision.

use crate::interface::interface::GrammarExpr;
use std::collections::{HashMap, HashSet};

/// Check if an expression contains regex features (CharClass or AnyChar).
/// Rules containing these features will be treated as terminals later,
/// so they should not be factored (helpers would not be recognized as terminals).
fn contains_regex_features(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::AnyChar => true,
        GrammarExpr::CharClass(_) => true,
        GrammarExpr::Literal(_) => false,
        GrammarExpr::Ref(_) => false,
        GrammarExpr::Sequence(exprs) | GrammarExpr::Choice(exprs) => {
            exprs.iter().any(contains_regex_features)
        }
        GrammarExpr::Optional(inner) | GrammarExpr::Repeat(inner) => {
            contains_regex_features(inner)
        }
    }
}

/// Factor choices in a list of grammar rules.
/// This operates on parsed GrammarExpr structures, not raw EBNF text.
pub fn factor_grammar_rules(rules: Vec<(String, GrammarExpr)>) -> Vec<(String, GrammarExpr)> {
    let mut factorer = ChoiceFactorer::new(rules);
    factorer.factor_all()
}

/// Main factoring engine
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
        let rule_order: Vec<String> = rules.iter().map(|(name, _)| name.clone()).collect();
        let rules: HashMap<String, GrammarExpr> = rules.into_iter().collect();
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
        for name in &self.rule_order.clone() {
            let expr = self.rules.get(name).unwrap().clone();
            
            // Only factor rules that are internal helpers (start with '_')
            // Skip rules with the _json prefix (empirically validated exception)
            // Skip rules with regex features (CharClass, AnyChar) - they will be treated as terminals
            // and their helper rules would not be recognized as terminals, causing errors
            let should_factor = name.starts_with('_') 
                && !name.starts_with("_json")
                && !contains_regex_features(&expr);
            
            let factored_expr = if should_factor {
                self.factor_expr(expr, name)
            } else {
                expr
            };
            
            self.new_rules.push((name.clone(), factored_expr));
        }
        
        crate::debug!(4, "EBNF factoring: {} rules -> {} rules ({} helpers created)", 
                      self.rule_order.len(), self.new_rules.len(), self.helper_counter);
        self.new_rules
    }

    /// Factor a single expression
    fn factor_expr(&mut self, expr: GrammarExpr, context_name: &str) -> GrammarExpr {
        match expr {
            GrammarExpr::Choice(alternatives) if alternatives.len() > 1 => {
                self.factor_choice(alternatives, context_name)
            }
            // Recursively process other expression types
            GrammarExpr::Sequence(exprs) => {
                GrammarExpr::Sequence(
                    exprs.into_iter()
                        .map(|e| self.factor_expr(e, context_name))
                        .collect()
                )
            }
            GrammarExpr::Optional(e) => {
                GrammarExpr::Optional(Box::new(self.factor_expr(*e, context_name)))
            }
            GrammarExpr::Repeat(e) => {
                GrammarExpr::Repeat(Box::new(self.factor_expr(*e, context_name)))
            }
            other => other,
        }
    }

    /// Factor a choice expression by grouping safe alternatives
    fn factor_choice(&mut self, alternatives: Vec<GrammarExpr>, context_name: &str) -> GrammarExpr {
        if alternatives.len() < 2 {
            return if alternatives.len() == 1 {
                alternatives.into_iter().next().unwrap()
            } else {
                GrammarExpr::Sequence(vec![]) // Empty choice -> epsilon
            };
        }

        // Classify alternatives as safe or unsafe
        let mut safe_alts = Vec::new();
        let mut unsafe_alts = Vec::new();
        
        for alt in alternatives {
            if self.is_safe_alternative(&alt) {
                safe_alts.push(alt);
            } else {
                unsafe_alts.push(alt);
            }
        }

        let mut final_choices = Vec::new();

        // Group safe alternatives into a single helper rule
        if safe_alts.len() > 1 {
            let helper_name = self.create_helper_rule(safe_alts, format!("{}_safe", context_name));
            final_choices.push(GrammarExpr::Ref(helper_name));
        } else if safe_alts.len() == 1 {
            final_choices.push(safe_alts.into_iter().next().unwrap());
        }

        // Group unsafe alternatives by their tail pattern
        let tail_groups = self.group_by_tail(&unsafe_alts);
        
        for (tail, heads) in tail_groups {
            if heads.len() > 1 || self.is_complex_head(&heads[0]) {
                let helper_name = if heads.len() == 1 {
                    self.create_helper_rule(heads.clone(), format!("{}_key", context_name))
                } else {
                    self.create_helper_rule(heads.clone(), format!("{}_keys", context_name))
                };
                
                // Reconstruct: helper ':' tail
                final_choices.push(GrammarExpr::Sequence(vec![
                    GrammarExpr::Ref(helper_name),
                    GrammarExpr::Literal(b":".to_vec()),
                    tail,
                ]));
            } else {
                // Single simple head, don't create helper
                final_choices.push(GrammarExpr::Sequence(vec![
                    heads.into_iter().next().unwrap(),
                    GrammarExpr::Literal(b":".to_vec()),
                    tail,
                ]));
            }
        }

        // Add remaining unsafe alternatives that don't follow the key:value pattern
        for alt in &unsafe_alts {
            if !self.has_tail_pattern(alt) {
                final_choices.push(alt.clone());
            }
        }

        if final_choices.is_empty() {
            GrammarExpr::Sequence(vec![]) // Epsilon
        } else if final_choices.len() == 1 {
            final_choices.into_iter().next().unwrap()
        } else {
            GrammarExpr::Choice(final_choices)
        }
    }

    /// Check if an alternative is "safe" (doesn't reference recursive rules)
    fn is_safe_alternative(&self, expr: &GrammarExpr) -> bool {
        let refs = self.collect_refs(expr);
        !refs.iter().any(|r| self.is_recursive(r))
    }

    /// Check if a rule is recursive (references itself transitively)
    fn is_recursive(&self, name: &str) -> bool {
        self.recursive_rules.contains(name)
    }

    /// Collect all rule references in an expression
    fn collect_refs(&self, expr: &GrammarExpr) -> HashSet<String> {
        let mut refs = HashSet::new();
        self.collect_refs_impl(expr, &mut refs);
        refs
    }

    fn collect_refs_impl(&self, expr: &GrammarExpr, refs: &mut HashSet<String>) {
        match expr {
            GrammarExpr::Ref(name) => {
                refs.insert(name.clone());
            }
            GrammarExpr::Sequence(exprs) | GrammarExpr::Choice(exprs) => {
                for e in exprs {
                    self.collect_refs_impl(e, refs);
                }
            }
            GrammarExpr::Optional(e) | GrammarExpr::Repeat(e) => {
                self.collect_refs_impl(e, refs);
            }
            GrammarExpr::Literal(_) | GrammarExpr::CharClass(_) | GrammarExpr::AnyChar => {}
        }
    }

    /// Group alternatives by their tail (for key:value patterns)
    fn group_by_tail(&self, alternatives: &[GrammarExpr]) -> HashMap<GrammarExpr, Vec<GrammarExpr>> {
        let mut groups: HashMap<GrammarExpr, Vec<GrammarExpr>> = HashMap::new();
        
        for alt in alternatives {
            if let Some((head, tail)) = self.extract_tail_pattern(alt) {
                if self.is_safe_alternative(&head) {
                    groups.entry(tail).or_insert_with(Vec::new).push(head);
                }
            }
        }
        
        groups
    }

    /// Check if an alternative has a tail pattern (e.g., head ':' tail)
    fn has_tail_pattern(&self, expr: &GrammarExpr) -> bool {
        self.extract_tail_pattern(expr).is_some()
    }

    /// Extract head and tail from patterns like: head ':' tail
    fn extract_tail_pattern(&self, expr: &GrammarExpr) -> Option<(GrammarExpr, GrammarExpr)> {
        if let GrammarExpr::Sequence(parts) = expr {
            // Look for patterns ending with ':' followed by a reference
            if parts.len() >= 3 {
                // Check if second-to-last is ':'
                let colon_idx = parts.len() - 2;
                if let GrammarExpr::Literal(lit) = &parts[colon_idx] {
                    if lit == b":" {
                        // Head is everything before ':', tail is after
                        let head = if colon_idx == 1 {
                            parts[0].clone()
                        } else {
                            GrammarExpr::Sequence(parts[..colon_idx].to_vec())
                        };
                        let tail = parts[parts.len() - 1].clone();
                        
                        // Only consider it a tail pattern if tail is a reference
                        if matches!(tail, GrammarExpr::Ref(_)) {
                            return Some((head, tail));
                        }
                    }
                }
            }
        }
        None
    }

    /// Check if a head expression is complex enough to warrant factoring
    fn is_complex_head(&self, expr: &GrammarExpr) -> bool {
        match expr {
            GrammarExpr::Sequence(parts) => parts.len() > 2,
            GrammarExpr::Choice(_) => true,
            GrammarExpr::Repeat(_) | GrammarExpr::Optional(_) => true,
            _ => false,
        }
    }

    /// Create a new helper rule with the given alternatives
    fn create_helper_rule(&mut self, alternatives: Vec<GrammarExpr>, base_name: String) -> String {
        // Check cache first
        let cache_key = alternatives.clone();
        if let Some(existing_name) = self.factor_cache.get(&cache_key) {
            return existing_name.clone();
        }

        // Generate unique name
        let mut helper_name = format!("__{}", base_name);
        let mut collision_idx = 1;
        let base = helper_name.clone();
        
        while self.rules.contains_key(&helper_name) || 
              self.new_rules.iter().any(|(name, _)| name == &helper_name) {
            helper_name = format!("{}_{}", base, collision_idx);
            collision_idx += 1;
        }

        // Create the rule
        let expr = if alternatives.len() == 1 {
            alternatives.into_iter().next().unwrap()
        } else {
            GrammarExpr::Choice(alternatives.clone())
        };

        self.new_rules.push((helper_name.clone(), expr));
        self.factor_cache.insert(cache_key, helper_name.clone());
        self.helper_counter += 1;

        helper_name
    }

    /// Find all recursive rules (rules that reference themselves transitively)
    fn find_recursive_rules(rules: &HashMap<String, GrammarExpr>) -> HashSet<String> {
        let mut recursive = HashSet::new();
        
        // Build dependency graph
        let mut deps: HashMap<String, Vec<String>> = HashMap::new();
        for (name, expr) in rules {
            let mut refs = HashSet::new();
            Self::collect_refs_static(expr, &mut refs);
            deps.insert(name.clone(), refs.into_iter().collect());
        }
        
        // For each rule, check if it can reach itself via DFS
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
                for e in exprs {
                    Self::collect_refs_static(e, refs);
                }
            }
            GrammarExpr::Optional(e) | GrammarExpr::Repeat(e) => {
                Self::collect_refs_static(e, refs);
            }
            GrammarExpr::Literal(_) | GrammarExpr::CharClass(_) | GrammarExpr::AnyChar => {}
        }
    }

    fn can_reach_self(start: &str, deps: &HashMap<String, Vec<String>>) -> bool {
        let mut visited = HashSet::new();
        let mut stack = vec![start.to_string()];
        visited.insert(start.to_string());
        
        while let Some(node) = stack.pop() {
            if let Some(neighbors) = deps.get(&node) {
                for neighbor in neighbors {
                    if neighbor == start {
                        return true;
                    }
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        stack.push(neighbor.clone());
                    }
                }
            }
        }
        
        false
    }
}
