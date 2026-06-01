//! Lowering from named grammar IR to flat compiler grammar.
//!
//! This module contains the semantics-preserving translation
//!
//! ```text
//! NamedGrammar  --->  GrammarDef
//! ```
//!
//! It is intentionally split by mathematical role: repeat lowering, separated
//! sequence lowering, ExprNFA lowering, local exact-alternative subtraction, and
//! terminal-expression conversion each live in their own files.

mod exact_subtraction;
mod expr_nfa_lower;
mod repeat;
pub(crate) mod separated_sequence;
mod terminal_expr;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use crate::automata::lexer::ast::Expr;
use crate::grammar_ir::ast::{GrammarExpr, NamedGrammar};
use crate::grammar_ir::flat::{
    GrammarDef, NonterminalID, Rule, Symbol, Terminal, TerminalID,
};
use crate::{GlrMaskError, Result};

use repeat::repeat_tree_shape;
use separated_sequence::comma_sep_shape;
use terminal_expr::{
    compute_rule_nullability, dedup_rules_preserving_first_occurrence, grammar_expr_to_expr,
    grammar_expr_is_nullable, validate_expr_nfa_placement,
};

pub use terminal_expr::expr_to_grammar_expr;

use crate::grammar_ir::render::lark::regex_escape_byte;

fn char_class_pattern(def: &str, negate: bool) -> String {
    if negate {
        format!("[^{}]", def)
    } else {
        format!("[{}]", def)
    }
}

pub(super) struct Lowerer {
    pub(super) rules: Vec<Rule>,
    pub(super) terminal_map: BTreeMap<String, TerminalID>,
    pub(super) terminals: Vec<Terminal>,
    pub(super) nonterminal_ids: BTreeMap<String, NonterminalID>,
    pub(super) generated_nonterminal_counter: u32,
    pub(super) terminal_names: BTreeMap<TerminalID, String>,
    pub(super) internal_terminal_names: HashSet<String>,
    pub(super) named_rule_exprs: HashMap<String, GrammarExpr>,
    pub(super) named_rule_is_terminal: HashMap<String, bool>,
    pub(super) rule_nullable: HashMap<String, bool>,
    pub(super) terminal_bodies: HashMap<String, GrammarExpr>,
    pub(super) terminal_expr_cache: HashMap<String, Arc<Expr>>,
    pub(super) nonnullable_named_rule_cache: HashMap<String, NonterminalID>,
    /// Shared cache for repeat-exact nonterminals, keyed by (symbol, count).
    pub(super) repeat_exact_cache: BTreeMap<(Symbol, usize), NonterminalID>,
    /// Shared cache for repeat-range nonterminals, keyed by (symbol, min, max).
    /// Only used for Left/Right shapes (bucket-based decomposition).
    pub(super) repeat_range_cache: BTreeMap<(Symbol, usize, usize), NonterminalID>,
    /// Shared cache for repeat-max nonterminals, keyed by (symbol, max).
    /// Used by LeftBalanced/Balanced shapes for O(log N) range decomposition.
    pub(super) repeat_max_cache: BTreeMap<(Symbol, usize), NonterminalID>,
    /// Shared cache for repeat-min1-max nonterminals, keyed by (symbol, max).
    /// repeat_min1_max_N matches exactly 1..N elements (N >= 1).
    pub(super) repeat_min1_max_cache: BTreeMap<(Symbol, usize), NonterminalID>,
}

impl Lowerer {
    fn new() -> Self {
        Self {
            rules: Vec::new(),
            terminal_map: BTreeMap::new(),
            terminals: Vec::new(),
            nonterminal_ids: BTreeMap::new(),
            generated_nonterminal_counter: 0,
            terminal_names: BTreeMap::new(),
            internal_terminal_names: HashSet::new(),
            named_rule_exprs: HashMap::new(),
            named_rule_is_terminal: HashMap::new(),
            rule_nullable: HashMap::new(),
            terminal_bodies: HashMap::new(),
            terminal_expr_cache: HashMap::new(),
            nonnullable_named_rule_cache: HashMap::new(),
            repeat_exact_cache: BTreeMap::new(),
            repeat_range_cache: BTreeMap::new(),
            repeat_max_cache: BTreeMap::new(),
            repeat_min1_max_cache: BTreeMap::new(),
        }
    }

    fn nonterminal_id(&mut self, name: &str) -> NonterminalID {
        if let Some(&id) = self.nonterminal_ids.get(name) {
            id
        } else {
            let id = self.nonterminal_ids.len() as NonterminalID;
            self.nonterminal_ids.insert(name.to_string(), id);
            id
        }
    }

    fn fresh_nonterminal(&mut self, hint: &str) -> (String, NonterminalID) {
        let name = format!("__{}_{}", hint, self.generated_nonterminal_counter);
        self.generated_nonterminal_counter += 1;
        let id = self.nonterminal_id(&name);
        (name, id)
    }

    fn expr_is_nullable(&self, expr: &GrammarExpr) -> bool {
        grammar_expr_is_nullable(expr, &self.rule_nullable)
    }

    fn strip_grouping(expr: &GrammarExpr) -> &GrammarExpr {
        match expr {
            GrammarExpr::Grouped(inner) => Self::strip_grouping(inner),
            _ => expr,
        }
    }

    fn top_level_alternatives(expr: &GrammarExpr) -> Vec<GrammarExpr> {
        match Self::strip_grouping(expr) {
            GrammarExpr::Choice(options) => options
                .iter()
                .map(|option| Self::strip_grouping(option).clone())
                .collect(),
            other => vec![other.clone()],
        }
    }

    fn resolve_terminal_expr(
        &mut self,
        owner_name: Option<&str>,
        expr: &GrammarExpr,
    ) -> Result<Expr, GlrMaskError> {
        let mut visiting = HashSet::new();
        if let Some(name) = owner_name {
            visiting.insert(name.to_string());
        }
        grammar_expr_to_expr(
            expr,
            &self.terminal_bodies,
            &mut self.terminal_expr_cache,
            &mut visiting,
        )
    }

    fn nonnullable_terminal_symbol(
        &mut self,
        expr: &GrammarExpr,
    ) -> Result<Option<Symbol>, GlrMaskError> {
        match expr {
            GrammarExpr::Literal(bytes) => {
                if bytes.is_empty() {
                    return Ok(None);
                }
                let pattern = bytes.iter().map(|&b| regex_escape_byte(b)).collect::<String>();
                let tid = self.terminal_id(&String::from_utf8_lossy(bytes), &pattern, false);
                Ok(Some(Symbol::Terminal(tid)))
            }
            GrammarExpr::Grouped(inner) => self.nonnullable_terminal_symbol(inner),
            GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::AnyByte
            | GrammarExpr::Exclude { .. }
            | GrammarExpr::Intersect { .. } => {
                let expr = self.resolve_terminal_expr(None, expr)?;
                let expr = if expr.is_nullable() {
                    Expr::Exclude {
                        expr: Box::new(expr),
                        exclude: Box::new(Expr::Epsilon),
                    }
                    .optimize()
                } else {
                    expr
                };
                let name = format!("__nonnullable_terminal_{}", self.generated_nonterminal_counter);
                let tid = self.register_terminal_expr(&name, expr);
                Ok(Some(Symbol::Terminal(tid)))
            }
            _ => Ok(None),
        }
    }

    fn lower_nonnullable_named_rule(&mut self, name: &str) -> Result<Symbol, GlrMaskError> {
        if let Some(&nt) = self.nonnullable_named_rule_cache.get(name) {
            return Ok(Symbol::Nonterminal(nt));
        }

        let expr = self
            .named_rule_exprs
            .get(name)
            .cloned()
            .ok_or_else(|| GlrMaskError::GrammarParse(format!("unknown rule referenced from SeparatedSequence: {name}")))?;
        let is_terminal = *self.named_rule_is_terminal.get(name).unwrap_or(&false);

        // If the referenced named rule is already nonnullable, reuse its
        // ordinary lowered symbol instead of synthesizing a second alias.
        if !self.rule_nullable.get(name).copied().unwrap_or(false)
            && !(is_terminal && self.internal_terminal_names.contains(name))
        {
            return Ok(Symbol::Nonterminal(self.nonterminal_id(name)));
        }

        let (_, nt) = self.fresh_nonterminal("nonnullable_rule");
        self.nonnullable_named_rule_cache.insert(name.to_string(), nt);

        if is_terminal {
            let terminal_expr = self.resolve_terminal_expr(Some(name), &expr)?;
            let terminal_expr = if terminal_expr.is_nullable() {
                Expr::Exclude {
                    expr: Box::new(terminal_expr),
                    exclude: Box::new(Expr::Epsilon),
                }
                .optimize()
            } else {
                terminal_expr
            };

            if !matches!(terminal_expr, Expr::Epsilon) {
                let terminal_name = format!("__nonnullable_ref_{name}");
                let tid = self.register_terminal_expr(&terminal_name, terminal_expr);
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![Symbol::Terminal(tid)],
                });
            }
        } else {
            self.emit_nonnullable_expr(nt, &expr)?;
        }

        Ok(Symbol::Nonterminal(nt))
    }

    fn lower_nonnullable_expr_symbol(
        &mut self,
        expr: &GrammarExpr,
    ) -> Result<Option<Symbol>, GlrMaskError> {
        match expr {
            GrammarExpr::Epsilon => Ok(None),
            GrammarExpr::Literal(bytes) if bytes.is_empty() => Ok(None),
            GrammarExpr::Grouped(inner) => self.lower_nonnullable_expr_symbol(inner),
            GrammarExpr::Ref(name) => Ok(Some(self.lower_nonnullable_named_rule(name)?)),
            GrammarExpr::Optional(inner) => self.lower_nonnullable_expr_symbol(inner),
            GrammarExpr::Exclude { .. } => {
                if let Some(lowered) = self.exact_nonterminal_subtraction_expr(expr)? {
                    return self.lower_nonnullable_expr_symbol(&lowered);
                }
                self.nonnullable_terminal_symbol(expr)
            }
            GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::AnyByte
            | GrammarExpr::Intersect { .. } => self.nonnullable_terminal_symbol(expr),
            _ => {
                let (_, nt) = self.fresh_nonterminal("nonnullable_expr");
                self.emit_nonnullable_expr(nt, expr)?;
                Ok(Some(Symbol::Nonterminal(nt)))
            }
        }
    }

    fn emit_nonnullable_sequence(
        &mut self,
        lhs: NonterminalID,
        parts: &[GrammarExpr],
    ) -> Result<(), GlrMaskError> {
        if parts.iter().any(|part| !self.expr_is_nullable(part)) {
            let mut rhs = Vec::with_capacity(parts.len());
            for part in parts {
                rhs.push(self.lower_expr_terminalish(part)?);
            }
            self.rules.push(Rule { lhs, rhs });
            return Ok(());
        }

        for (nonempty_index, nonempty_part) in parts.iter().enumerate() {
            let Some(nonempty_symbol) = self.lower_nonnullable_expr_symbol(nonempty_part)? else {
                continue;
            };

            let mut rhs = Vec::with_capacity(parts.len());
            for (index, part) in parts.iter().enumerate() {
                if index == nonempty_index {
                    rhs.push(nonempty_symbol.clone());
                } else {
                    rhs.push(self.lower_expr_terminalish(part)?);
                }
            }
            self.rules.push(Rule { lhs, rhs });
        }
        Ok(())
    }

    fn emit_nonnullable_expr(
        &mut self,
        lhs: NonterminalID,
        expr: &GrammarExpr,
    ) -> Result<(), GlrMaskError> {
        match expr {
            GrammarExpr::Grouped(inner) => {
                self.emit_nonnullable_expr(lhs, inner)?;
            }
            GrammarExpr::Ref(name) => {
                let symbol = self.lower_nonnullable_named_rule(name)?;
                self.rules.push(Rule { lhs, rhs: vec![symbol] });
            }
            GrammarExpr::Exclude { .. } => {
                if let Some(lowered) = self.exact_nonterminal_subtraction_expr(expr)? {
                    self.emit_nonnullable_expr(lhs, &lowered)?;
                } else if let Some(symbol) = self.nonnullable_terminal_symbol(expr)? {
                    self.rules.push(Rule { lhs, rhs: vec![symbol] });
                }
            }
            GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte
            | GrammarExpr::Intersect { .. } => {
                if let Some(symbol) = self.nonnullable_terminal_symbol(expr)? {
                    self.rules.push(Rule { lhs, rhs: vec![symbol] });
                }
            }
            GrammarExpr::Sequence(parts) => {
                self.emit_nonnullable_sequence(lhs, parts)?;
            }
            GrammarExpr::Choice(options) => {
                for option in options {
                    self.emit_nonnullable_expr(lhs, option)?;
                }
            }
            GrammarExpr::Optional(inner) => {
                self.emit_nonnullable_expr(lhs, inner)?;
            }
            GrammarExpr::Repeat(inner) | GrammarExpr::RepeatOne(inner) => {
                if let Some(symbol) = self.lower_nonnullable_expr_symbol(inner)? {
                    self.rules.push(Rule {
                        lhs,
                        rhs: vec![symbol.clone()],
                    });
                    self.rules.push(Rule {
                        lhs,
                        rhs: vec![Symbol::Nonterminal(lhs), symbol],
                    });
                }
            }
            GrammarExpr::RepeatRange { expr, min, max } => {
                let Some(symbol) = self.lower_nonnullable_expr_symbol(expr)? else {
                    return Ok(());
                };
                let adjusted_min = if self.expr_is_nullable(expr) {
                    1
                } else {
                    *min
                };
                if adjusted_min > *max {
                    return Ok(());
                }
                let shape = repeat_tree_shape();
                let range_nonterminal = self.repeat_range_nonterminal(
                    &symbol,
                    adjusted_min,
                    *max,
                    shape,
                );
                self.rules.push(Rule {
                    lhs,
                    rhs: vec![Symbol::Nonterminal(range_nonterminal)],
                });
            }
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                let shape = comma_sep_shape();
                let (symbol, _) = self.lower_separated_sequence_inner(items, separator, shape)?;
                self.rules.push(Rule {
                    lhs,
                    rhs: vec![symbol],
                });
            }
            GrammarExpr::ExprNFA(expr_nfa) => {
                self.emit_expr_nfa_nonnullable(lhs, expr_nfa)?;
            }
            GrammarExpr::Epsilon => {}
        }
        Ok(())
    }

    fn terminal_id(&mut self, name: &str, pattern: &str, utf8: bool) -> TerminalID {
        let pattern_key = format!("{pattern}:{utf8}");
        if let Some(&id) = self.terminal_map.get(&pattern_key) {
            return id;
        }
        let id = self.terminals.len() as TerminalID;
        self.terminal_map.insert(pattern_key, id);
        self.terminal_names.insert(id, name.to_string());
        let name_bytes = name.as_bytes();
        let literal_pattern: String = name_bytes.iter().map(|&byte| regex_escape_byte(byte)).collect();
        if literal_pattern == pattern && !utf8 {
            self.terminals.push(Terminal::Literal {
                id,
                bytes: name_bytes.to_vec(),
            });
        } else {
            self.terminals.push(Terminal::Pattern {
                id,
                pattern: pattern.to_string(),
                utf8,
            });
        }
        id
    }


    fn lower_expr(&mut self, expr: &GrammarExpr) -> Symbol {
        fn emit(lowerer: &mut Lowerer, lhs: NonterminalID, expr: &GrammarExpr) -> Result<(), GlrMaskError> {
            match expr {
                GrammarExpr::Grouped(inner) => emit(lowerer, lhs, inner)?,
                GrammarExpr::Sequence(parts) => {
                    let mut rhs = Vec::new();
                    for part in parts {
                        rhs.push(lowerer.lower_expr(part));
                    }
                    lowerer.rules.push(Rule { lhs, rhs });
                }
                GrammarExpr::Choice(options) => {
                    for option in options {
                        emit(lowerer, lhs, option)?;
                    }
                }
                GrammarExpr::Optional(inner) => {
                    lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                    emit(lowerer, lhs, inner)?;
                }
                GrammarExpr::Repeat(inner) => {
                    let symbol = lowerer.lower_expr(inner);
                    lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                    lowerer.rules.push(Rule {
                        lhs,
                        rhs: vec![Symbol::Nonterminal(lhs), symbol],
                    });
                }
                GrammarExpr::RepeatOne(inner) => {
                    let symbol = lowerer.lower_expr(inner);
                    lowerer.rules.push(Rule {
                        lhs,
                        rhs: vec![symbol.clone()],
                    });
                    lowerer.rules.push(Rule {
                        lhs,
                        rhs: vec![Symbol::Nonterminal(lhs), symbol],
                    });
                }
                GrammarExpr::RepeatRange { expr, min, max } => {
                    lowerer.emit_repeat_range(lhs, expr, *min, *max)?;
                }
                GrammarExpr::SeparatedSequence { items, separator, allow_empty } => {
                    let shape = comma_sep_shape();
                    let (sym, can_be_empty) =
                        lowerer.lower_separated_sequence_inner(items, separator, shape)?;
                    lowerer.rules.push(Rule { lhs, rhs: vec![sym] });
                    if *allow_empty && can_be_empty {
                        lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                    }
                }
                GrammarExpr::Epsilon => {
                    lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                }
                GrammarExpr::ExprNFA(expr_nfa) => {
                    lowerer.emit_expr_nfa(lhs, expr_nfa)?;
                }
                _ => {
                    if let Some(lowered) = lowerer.exact_nonterminal_subtraction_expr(expr)? {
                        emit(lowerer, lhs, &lowered)?;
                        return Ok(());
                    }
                    let symbol = lowerer.lower_expr_terminalish(expr)?;
                    lowerer.rules.push(Rule {
                        lhs,
                        rhs: vec![symbol],
                    });
                }
            }
            Ok(())
        }

        let (_, nonterminal) = self.fresh_nonterminal("expr");
        emit(self, nonterminal, expr)
            .expect("grammar lowering should not fail for internal expression emission");
        Symbol::Nonterminal(nonterminal)
    }

    fn lower_expr_terminalish(&mut self, expr: &GrammarExpr) -> Result<Symbol, GlrMaskError> {
        Ok(match expr {
            GrammarExpr::Grouped(inner) => return self.lower_expr_terminalish(inner),
            GrammarExpr::Ref(name) => {
                if !self.named_rule_exprs.contains_key(name) {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "unknown rule referenced from nonterminal context: {name}"
                    )));
                }
                if self.internal_terminal_names.contains(name) {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "internal-only terminal {name} referenced from nonterminal context"
                    )));
                }
                Symbol::Nonterminal(self.nonterminal_id(name))
            }
            GrammarExpr::Literal(bytes) => {
                let pattern = bytes.iter().map(|&b| regex_escape_byte(b)).collect::<String>();
                Symbol::Terminal(self.terminal_id(&String::from_utf8_lossy(bytes), &pattern, false))
            }
            GrammarExpr::CharClass { def, negate, utf8 } => {
                let pattern = char_class_pattern(def, *negate);
                Symbol::Terminal(self.terminal_id(&pattern, &pattern, *utf8))
            }
            GrammarExpr::RawRegex(pattern) => {
                // assume utf8 true for raw regex from lark/ebnf
                Symbol::Terminal(self.terminal_id(pattern, pattern, true))
            }
            GrammarExpr::LexerDfa(_) => {
                let expr = self.resolve_terminal_expr(None, expr)?;
                let name = format!("__terminal_expr_{}", self.generated_nonterminal_counter);
                Symbol::Terminal(self.register_terminal_expr(&name, expr))
            }
            GrammarExpr::Epsilon => {
                // Epsilon as an inline NT atom: create a nonterminal with an empty production.
                let (_, nt) = self.fresh_nonterminal("eps");
                self.rules.push(Rule { lhs: nt, rhs: Vec::new() });
                Symbol::Nonterminal(nt)
            }
            GrammarExpr::Exclude { .. } | GrammarExpr::Intersect { .. } => {
                if let Some(lowered) = self.exact_nonterminal_subtraction_expr(expr)? {
                    return Ok(self.lower_expr(&lowered));
                }
                let expr = self.resolve_terminal_expr(None, expr)?;
                let name = format!("__terminal_expr_{}", self.generated_nonterminal_counter);
                Symbol::Terminal(self.register_terminal_expr(&name, expr))
            }
            GrammarExpr::AnyByte => {
                Symbol::Terminal(self.terminal_id(".", ".", false))
            }
            GrammarExpr::Sequence(_)
            | GrammarExpr::Choice(_)
            | GrammarExpr::Optional(_)
            | GrammarExpr::Repeat(_)
            | GrammarExpr::RepeatOne(_)
            | GrammarExpr::RepeatRange { .. }
            | GrammarExpr::SeparatedSequence { .. } => self.lower_expr(expr),
            | GrammarExpr::ExprNFA(_) => {
                return Err(GlrMaskError::GrammarParse(
                    "GrammarExpr::ExprNFA must be the complete expression of a nonterminal rule"
                        .into(),
                ));
            }
        })
    }


    fn register_terminal_expr(&mut self, name: &str, expr: Expr) -> TerminalID {
        if let Some(id) = self.terminals.iter().find_map(|terminal| match terminal {
            Terminal::Expr { id, expr: existing } if *existing == expr => Some(*id),
            _ => None,
        }) {
            return id;
        }

        let id = self.terminals.len() as TerminalID;
        self.terminal_names.insert(id, name.to_string());
        self.terminals.push(Terminal::Expr { id, expr });
        id
    }
}

pub fn lower(grammar: &NamedGrammar) -> Result<GrammarDef, GlrMaskError> {
    validate_expr_nfa_placement(grammar)?;

    let mut lowerer = Lowerer::new();
    lowerer.named_rule_exprs = grammar
        .rules
        .iter()
        .map(|rule| (rule.name.clone(), rule.expr.clone()))
        .collect();
    lowerer.named_rule_is_terminal = grammar
        .rules
        .iter()
        .map(|rule| (rule.name.clone(), rule.is_terminal))
        .collect();
    lowerer.rule_nullable = compute_rule_nullability(grammar);

    // Collect internal terminal names for validation.
    lowerer.internal_terminal_names = grammar
        .rules
        .iter()
        .filter(|r| r.is_terminal && r.is_internal)
        .map(|r| r.name.clone())
        .collect();

    for rule in &grammar.rules {
        if rule.is_terminal && rule.is_internal {
            continue; // don't allocate nonterminal IDs for internal terminals
        }
        lowerer.nonterminal_id(&rule.name);
    }

    // Build a map of terminal rule bodies for resolving Ref nodes inside terminal exprs.
    lowerer.terminal_bodies = grammar
        .rules
        .iter()
        .filter(|r| r.is_terminal)
        .map(|r| (r.name.clone(), r.expr.clone()))
        .collect();

    for rule in &grammar.rules {
        // Terminal rules: convert the entire body to a single Terminal::Expr.
        // Refs to other terminal rules are resolved via Expr::Shared.
        if rule.is_terminal {
            let expr = lowerer.resolve_terminal_expr(Some(&rule.name), &rule.expr)?;
            let arc = Arc::new(expr.clone());
            lowerer.terminal_expr_cache.insert(rule.name.clone(), arc);

            if rule.is_internal {
                // Internal-only: cached for Shared resolution, no terminal or production.
                continue;
            }

            let lhs = lowerer.nonterminal_id(&rule.name);
            let tid = lowerer.register_terminal_expr(&rule.name, expr);
            lowerer.rules.push(Rule { lhs, rhs: vec![Symbol::Terminal(tid)] });
            continue;
        }

        let lhs = lowerer.nonterminal_id(&rule.name);

        match &rule.expr {
            GrammarExpr::Grouped(inner) => {
                let symbol = lowerer.lower_expr_terminalish(inner)?;
                lowerer.rules.push(Rule { lhs, rhs: vec![symbol] });
            }
            GrammarExpr::Sequence(parts) => {
                let rhs = parts.iter().map(|part| lowerer.lower_expr_terminalish(part)).collect::<Result<Vec<_>, _>>()?;
                lowerer.rules.push(Rule { lhs, rhs });
            }
            GrammarExpr::Choice(options) => {
                for option in options {
                    match option {
                        GrammarExpr::Sequence(parts) => {
                            let rhs = parts.iter().map(|part| lowerer.lower_expr_terminalish(part)).collect::<Result<Vec<_>, _>>()?;
                            lowerer.rules.push(Rule { lhs, rhs });
                        }
                        _ => {
                            let symbol = lowerer.lower_expr_terminalish(option)?;
                            lowerer.rules.push(Rule { lhs, rhs: vec![symbol] });
                        }
                    }
                }
            }
            GrammarExpr::Optional(inner) => {
                lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                let symbol = lowerer.lower_expr_terminalish(inner)?;
                lowerer.rules.push(Rule { lhs, rhs: vec![symbol] });
            }
            GrammarExpr::Repeat(inner) => {
                let symbol = lowerer.lower_expr_terminalish(inner)?;
                lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                lowerer.rules.push(Rule { lhs, rhs: vec![Symbol::Nonterminal(lhs), symbol] });
            }
            GrammarExpr::RepeatOne(inner) => {
                let symbol = lowerer.lower_expr_terminalish(inner)?;
                lowerer.rules.push(Rule { lhs, rhs: vec![symbol.clone()] });
                lowerer.rules.push(Rule { lhs, rhs: vec![Symbol::Nonterminal(lhs), symbol] });
            }
            GrammarExpr::RepeatRange { expr, min, max } => {
                lowerer.emit_repeat_range(lhs, expr, *min, *max)?;
            }
            GrammarExpr::ExprNFA(expr_nfa) => {
                lowerer.emit_expr_nfa(lhs, expr_nfa)?;
            }
            _ => {
                let symbol = lowerer.lower_expr_terminalish(&rule.expr)?;
                lowerer.rules.push(Rule { lhs, rhs: vec![symbol] });
            }
        }
    }

    let start = lowerer
        .nonterminal_ids
        .get(&grammar.start)
        .copied()
        .ok_or_else(|| {
            GlrMaskError::GrammarParse(format!("undefined start rule: {}", grammar.start))
        })?;
    let nonterminal_names = lowerer
        .nonterminal_ids
        .iter()
        .filter(|(name, _)| !name.starts_with("__"))
        .map(|(name, id)| (*id, name.clone()))
        .collect();

    let ignore_terminal = grammar.ignore.as_ref().and_then(|ignore_name| {
        lowerer
            .terminal_names
            .iter()
            .find_map(|(&id, name)| (name == ignore_name).then_some(id))
    });

    dedup_rules_preserving_first_occurrence(&mut lowerer.rules);

    Ok(GrammarDef {
        rules: lowerer.rules,
        start,
        terminals: lowerer.terminals,
        nonterminal_names,
        terminal_names: lowerer.terminal_names,
        ignore_terminal,
    })
}
