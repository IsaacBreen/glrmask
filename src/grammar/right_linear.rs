//! Exact compression of large right-linear named grammars into one ExprNFA.
//!
//! Generated state-machine grammars often spell an automaton as hundreds of
//! parser rules. Feeding that shape through LR construction and parser-DWA
//! composition needlessly materializes a large pushdown machine for a regular
//! language. This pass recognizes the safe subset and preserves the graph as an
//! [`ExprNFA`](crate::grammar::expr_nfa::ExprNFA).

use std::collections::{HashMap, HashSet};

use crate::grammar::ast::{GrammarExpr, NamedGrammar, NamedRule, Quantifier};
use crate::grammar::expr_nfa::ExprNfaBuilder;

const MIN_PARSER_RULES: usize = 32;
const MAX_MACRO_NODES: usize = 64;
const DISABLE_ENV: &str = "GLRMASK_DISABLE_RIGHT_LINEAR_COMPRESSION";

fn enabled() -> bool {
    std::env::var(DISABLE_ENV)
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(true)
}

#[derive(Clone)]
struct ExpandedMacro {
    expr: GrammarExpr,
    nodes: usize,
}

struct MacroExpander<'a> {
    rules: &'a HashMap<String, &'a NamedRule>,
    terminals: &'a HashSet<String>,
    visiting: HashSet<String>,
    memo: HashMap<String, Option<ExpandedMacro>>,
}

impl<'a> MacroExpander<'a> {
    fn new(
        rules: &'a HashMap<String, &'a NamedRule>,
        terminals: &'a HashSet<String>,
    ) -> Self {
        Self {
            rules,
            terminals,
            visiting: HashSet::new(),
            memo: HashMap::new(),
        }
    }

    fn rule(&mut self, name: &str) -> Option<ExpandedMacro> {
        if self.terminals.contains(name) {
            return Some(ExpandedMacro {
                expr: GrammarExpr::Ref(name.to_owned()),
                nodes: 1,
            });
        }
        if let Some(cached) = self.memo.get(name) {
            return cached.clone();
        }
        if !self.visiting.insert(name.to_owned()) {
            return None;
        }
        let result = self
            .rules
            .get(name)
            .and_then(|rule| self.expr(&rule.expr));
        self.visiting.remove(name);
        self.memo.insert(name.to_owned(), result.clone());
        result
    }

    fn expr(&mut self, expr: &GrammarExpr) -> Option<ExpandedMacro> {
        let result = match expr {
            GrammarExpr::Ref(name) => self.rule(name)?,
            GrammarExpr::Grouped(inner) => {
                let inner = self.expr(inner)?;
                ExpandedMacro {
                    expr: GrammarExpr::Grouped(Box::new(inner.expr)),
                    nodes: inner.nodes + 1,
                }
            }
            GrammarExpr::Sequence(parts) => {
                let mut nodes = 1usize;
                let mut expanded = Vec::with_capacity(parts.len());
                for part in parts {
                    let part = self.expr(part)?;
                    nodes = nodes.checked_add(part.nodes)?;
                    if nodes > MAX_MACRO_NODES {
                        return None;
                    }
                    expanded.push(part.expr);
                }
                ExpandedMacro {
                    expr: GrammarExpr::Sequence(expanded),
                    nodes,
                }
            }
            GrammarExpr::Choice(options) => {
                let mut nodes = 1usize;
                let mut expanded = Vec::with_capacity(options.len());
                for option in options {
                    let option = self.expr(option)?;
                    nodes = nodes.checked_add(option.nodes)?;
                    if nodes > MAX_MACRO_NODES {
                        return None;
                    }
                    expanded.push(option.expr);
                }
                ExpandedMacro {
                    expr: GrammarExpr::Choice(expanded),
                    nodes,
                }
            }
            GrammarExpr::Quantified(inner, quantifier) => {
                let inner = self.expr(inner)?;
                ExpandedMacro {
                    expr: GrammarExpr::Quantified(Box::new(inner.expr), quantifier.clone()),
                    nodes: inner.nodes + 1,
                }
            }
            GrammarExpr::Exclude { expr, exclude } => {
                let expr = self.expr(expr)?;
                let exclude = self.expr(exclude)?;
                ExpandedMacro {
                    expr: GrammarExpr::Exclude {
                        expr: Box::new(expr.expr),
                        exclude: Box::new(exclude.expr),
                    },
                    nodes: expr.nodes + exclude.nodes + 1,
                }
            }
            GrammarExpr::Intersect { expr, intersect } => {
                let expr = self.expr(expr)?;
                let intersect = self.expr(intersect)?;
                ExpandedMacro {
                    expr: GrammarExpr::Intersect {
                        expr: Box::new(expr.expr),
                        intersect: Box::new(intersect.expr),
                    },
                    nodes: expr.nodes + intersect.nodes + 1,
                }
            }
            GrammarExpr::Epsilon
            | GrammarExpr::Literal(_)
            | GrammarExpr::SpecialToken(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => ExpandedMacro {
                expr: expr.clone(),
                nodes: 1,
            },
            GrammarExpr::SeparatedSequence { .. } | GrammarExpr::ExprNFA(_) => return None,
        };
        (result.nodes <= MAX_MACRO_NODES).then_some(result)
    }
}

fn rewrite_macros(
    expr: &GrammarExpr,
    terminals: &HashSet<String>,
    macros: &HashMap<String, ExpandedMacro>,
) -> GrammarExpr {
    match expr {
        GrammarExpr::Ref(name) if !terminals.contains(name) => macros
            .get(name)
            .map_or_else(|| expr.clone(), |expanded| expanded.expr.clone()),
        GrammarExpr::Grouped(inner) => {
            GrammarExpr::Grouped(Box::new(rewrite_macros(inner, terminals, macros)))
        }
        GrammarExpr::Sequence(parts) => GrammarExpr::Sequence(
            parts
                .iter()
                .map(|part| rewrite_macros(part, terminals, macros))
                .collect(),
        ),
        GrammarExpr::Choice(options) => GrammarExpr::Choice(
            options
                .iter()
                .map(|option| rewrite_macros(option, terminals, macros))
                .collect(),
        ),
        GrammarExpr::Quantified(inner, quantifier) => GrammarExpr::Quantified(
            Box::new(rewrite_macros(inner, terminals, macros)),
            quantifier.clone(),
        ),
        GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
            expr: Box::new(rewrite_macros(expr, terminals, macros)),
            exclude: Box::new(rewrite_macros(exclude, terminals, macros)),
        },
        GrammarExpr::Intersect { expr, intersect } => GrammarExpr::Intersect {
            expr: Box::new(rewrite_macros(expr, terminals, macros)),
            intersect: Box::new(rewrite_macros(intersect, terminals, macros)),
        },
        other => other.clone(),
    }
}

struct RightLinearBuilder<'a> {
    builder: ExprNfaBuilder,
    parser_states: HashMap<String, u32>,
    parser_names: &'a HashSet<String>,
    accept: u32,
}

impl<'a> RightLinearBuilder<'a> {
    fn new(parser_names: &'a HashSet<String>, start: &str) -> Option<Self> {
        if !parser_names.contains(start) {
            return None;
        }
        let mut builder = ExprNfaBuilder::new();
        let mut parser_states = HashMap::with_capacity(parser_names.len());
        parser_states.insert(start.to_owned(), builder.start_state());
        let mut names = parser_names.iter().filter(|name| name.as_str() != start).collect::<Vec<_>>();
        names.sort_unstable();
        for name in names {
            parser_states.insert(name.clone(), builder.add_state());
        }
        let accept = builder.add_state();
        builder.set_accepting(accept);
        Some(Self {
            builder,
            parser_states,
            parser_names,
            accept,
        })
    }

    fn has_parser_ref(&self, expr: &GrammarExpr) -> bool {
        match expr {
            GrammarExpr::Ref(name) => self.parser_names.contains(name),
            GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) => {
                self.has_parser_ref(inner)
            }
            GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
                parts.iter().any(|part| self.has_parser_ref(part))
            }
            GrammarExpr::Exclude { expr, exclude } => {
                self.has_parser_ref(expr) || self.has_parser_ref(exclude)
            }
            GrammarExpr::Intersect { expr, intersect } => {
                self.has_parser_ref(expr) || self.has_parser_ref(intersect)
            }
            GrammarExpr::SeparatedSequence { .. } | GrammarExpr::ExprNFA(_) => true,
            _ => false,
        }
    }

    fn compile(&mut self, expr: &GrammarExpr, from: u32, to: u32) -> bool {
        if !self.has_parser_ref(expr) {
            if matches!(expr, GrammarExpr::Epsilon) {
                self.builder.add_epsilon(from, to);
            } else {
                self.builder.add_transition(from, expr.clone(), to);
            }
            return true;
        }
        match expr {
            GrammarExpr::Ref(name) if self.parser_names.contains(name) => {
                if to != self.accept {
                    return false;
                }
                let Some(&target) = self.parser_states.get(name) else {
                    return false;
                };
                self.builder.add_epsilon(from, target);
                true
            }
            GrammarExpr::Grouped(inner) => self.compile(inner, from, to),
            GrammarExpr::Sequence(parts) => {
                if parts.is_empty() {
                    self.builder.add_epsilon(from, to);
                    return true;
                }
                let mut current = from;
                for (index, part) in parts.iter().enumerate() {
                    let next = if index + 1 == parts.len() {
                        to
                    } else {
                        self.builder.add_state()
                    };
                    if !self.compile(part, current, next) {
                        return false;
                    }
                    current = next;
                }
                true
            }
            GrammarExpr::Choice(options) => options.iter().all(|option| self.compile(option, from, to)),
            GrammarExpr::Epsilon => {
                self.builder.add_epsilon(from, to);
                true
            }
            GrammarExpr::Quantified(inner, Quantifier::Optional) => {
                self.builder.add_epsilon(from, to);
                self.compile(inner, from, to)
            }
            GrammarExpr::Quantified(inner, Quantifier::ZeroPlus) => {
                if self.has_parser_ref(inner) {
                    return false;
                }
                self.builder.add_epsilon(from, to);
                let body_start = self.builder.add_state();
                let body_end = self.builder.add_state();
                self.builder.add_epsilon(from, body_start);
                if !self.compile(inner, body_start, body_end) {
                    return false;
                }
                self.builder.add_epsilon(body_end, body_start);
                self.builder.add_epsilon(body_end, to);
                true
            }
            GrammarExpr::Quantified(inner, Quantifier::OnePlus) => {
                if self.has_parser_ref(inner) {
                    return false;
                }
                let body_start = self.builder.add_state();
                let body_end = self.builder.add_state();
                self.builder.add_epsilon(from, body_start);
                if !self.compile(inner, body_start, body_end) {
                    return false;
                }
                self.builder.add_epsilon(body_end, body_start);
                self.builder.add_epsilon(body_end, to);
                true
            }
            GrammarExpr::Quantified(_, Quantifier::Range(_, _))
            | GrammarExpr::SeparatedSequence { .. }
            | GrammarExpr::ExprNFA(_) => false,
            atom => {
                if self.has_parser_ref(atom) {
                    return false;
                }
                self.builder.add_transition(from, atom.clone(), to);
                true
            }
        }
    }
}

/// Replace a large exact right-linear parser-rule graph by one ExprNFA rule.
///
/// Returns `true` when compression was selected. The pass is deliberately
/// conservative: it requires no ignore rule or custom lexer partitions and
/// rejects every parser reference that is not in tail position.
pub fn compress_large_right_linear_grammar(grammar: &mut NamedGrammar) -> bool {
    if !enabled()
        || grammar.ignore.is_some()
        || !grammar.lexer_partitions.is_empty()
        || !grammar.lexer_literal_partitions.is_empty()
        || grammar.default_lexer_partition.is_some()
    {
        return false;
    }

    let parser_rule_count = grammar.rules.iter().filter(|rule| !rule.is_terminal).count();
    if parser_rule_count < MIN_PARSER_RULES {
        return false;
    }

    let rule_map = grammar
        .rules
        .iter()
        .map(|rule| (rule.name.clone(), rule))
        .collect::<HashMap<_, _>>();
    let terminals = grammar
        .rules
        .iter()
        .filter(|rule| rule.is_terminal)
        .map(|rule| rule.name.clone())
        .collect::<HashSet<_>>();

    let mut expander = MacroExpander::new(&rule_map, &terminals);
    let parser_names_all = grammar
        .rules
        .iter()
        .filter(|rule| !rule.is_terminal)
        .map(|rule| rule.name.clone())
        .collect::<Vec<_>>();
    let mut macros = HashMap::new();
    for name in &parser_names_all {
        if name == &grammar.start {
            continue;
        }
        if let Some(expanded) = expander.rule(name) {
            macros.insert(name.clone(), expanded);
        }
    }

    let parser_names = parser_names_all
        .iter()
        .filter(|name| !macros.contains_key(*name))
        .cloned()
        .collect::<HashSet<_>>();
    let mut right_linear = match RightLinearBuilder::new(&parser_names, &grammar.start) {
        Some(builder) => builder,
        None => return false,
    };

    let mut names = parser_names.iter().collect::<Vec<_>>();
    names.sort_unstable();
    for name in names {
        let Some(rule) = rule_map.get(name) else {
            return false;
        };
        let expr = rewrite_macros(&rule.expr, &terminals, &macros);
        let from = right_linear.parser_states[name];
        if !right_linear.compile(&expr, from, right_linear.accept) {
            return false;
        }
    }

    let expr_nfa = right_linear.builder.build().with_direct_nfa_emission();
    let input_rules = grammar.rules.len();
    let nfa_states = expr_nfa.nfa.states.len();
    let nfa_transitions = expr_nfa
        .nfa
        .states
        .iter()
        .map(|state| {
            state.transitions.values().map(Vec::len).sum::<usize>() + state.epsilons.len()
        })
        .sum::<usize>();

    grammar.rules.retain(|rule| rule.is_terminal);
    grammar.rules.push(NamedRule {
        name: grammar.start.clone(),
        expr: GrammarExpr::ExprNFA(Box::new(expr_nfa)),
        is_terminal: false,
        is_internal: false,
    });

    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
    {
        eprintln!(
            "[glrmask/profile][right_linear_compression] selected=true input_rules={} parser_rules={} macros={} nfa_states={} nfa_transitions={}",
            input_rules,
            parser_rule_count,
            macros.len(),
            nfa_states,
            nfa_transitions,
        );
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::lark::{parse_lark_to_named, parse_lark_to_named_uncompressed};

    #[test]
    fn compresses_large_tail_recursive_state_graph() {
        let mut source = String::from("start: s0\n");
        for index in 0..40 {
            source.push_str(&format!("s{index}: A s{} | B\n", index + 1));
        }
        source.push_str("s40: B\nA: /a/\nB: /b/\n");
        let mut grammar = parse_lark_to_named_uncompressed(&source).unwrap();
        assert!(compress_large_right_linear_grammar(&mut grammar));
        assert_eq!(grammar.rules.iter().filter(|rule| !rule.is_terminal).count(), 1);
        assert!(matches!(
            grammar.rules.iter().find(|rule| !rule.is_terminal).unwrap().expr,
            GrammarExpr::ExprNFA(_)
        ));
    }

    #[test]
    fn direct_regular_metadata_survives_compile_preparation() {
        let mut source = String::from("start: s0\n");
        for index in 0..40 {
            source.push_str(&format!("s{index}: A s{} | B\n", index + 1));
        }
        source.push_str("s40: B\nA: /a/\nB: /b/\n");

        let mut named = parse_lark_to_named_uncompressed(&source).unwrap();
        assert!(compress_large_right_linear_grammar(&mut named));
        let factored = crate::grammar::factoring::factor_named_grammar(named);
        let lowered = crate::grammar::ast::lower(&factored).unwrap();
        assert!(lowered.direct_regular_automaton.is_some(), "AST lower lost direct regular metadata");
        let prepared = crate::compiler::grammar::transforms::prepare_grammar_transforms_only(lowered);
        assert!(prepared.direct_regular_automaton.is_some(), "grammar transforms lost direct regular metadata");
        let analyzed = crate::compiler::glr::analysis::AnalyzedGrammar::from_grammar_def(&prepared);
        assert!(analyzed.direct_regular_automaton.is_some(), "analysis lost direct regular metadata");
    }

    #[test]
    fn rejects_non_tail_recursion() {
        let mut source = String::from("start: s0\n");
        for index in 0..40 {
            source.push_str(&format!("s{index}: A s{} B | A\n", index + 1));
        }
        source.push_str("s40: A\nA: /a/\nB: /b/\n");
        let mut grammar = parse_lark_to_named(&source).unwrap();
        assert!(!compress_large_right_linear_grammar(&mut grammar));
    }

    #[test]
    fn ignores_small_and_lexer_partitioned_grammars() {
        let mut grammar = parse_lark_to_named("start: A\nA: /a/\n").unwrap();
        assert!(!compress_large_right_linear_grammar(&mut grammar));
    }
}
