//! Named grammar IR: syntax before compiler lowering.
//!
//! This module contains only data definitions and local graph utilities for the
//! named grammar representation.  It deliberately does not own lowering,
//! renderer, parser, or transform algorithms.
//!
//! Mathematical contract:
//! - [`GrammarExpr`] is syntax.
//! - [`NamedRule`] binds a name to syntax and marks whether that name denotes a terminal.
//! - [`NamedGrammar`] is an ordered rule set plus a start symbol and optional ignore terminal.
//! - Lowering to the compiler grammar lives in [`crate::grammar_ir::lower`].
//! - Observation/serialization lives in [`crate::grammar_ir::render`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::automata::lexer::dfa::DFA as LexerDFA;
use crate::grammar_ir::expr_nfa::ExprNFA;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GrammarExpr {
    Ref(String),
    Grouped(Box<GrammarExpr>),
    Sequence(Vec<GrammarExpr>),
    Choice(Vec<GrammarExpr>),
    /// Empty string / epsilon. Equivalent to `Sequence([])` for grammar purposes;
    /// maps to `Expr::Epsilon` in terminal-expression context.
    Epsilon,
    Exclude {
        expr: Box<GrammarExpr>,
        exclude: Box<GrammarExpr>,
    },
    Intersect {
        expr: Box<GrammarExpr>,
        intersect: Box<GrammarExpr>,
    },
    Optional(Box<GrammarExpr>),
    Repeat(Box<GrammarExpr>),
    RepeatOne(Box<GrammarExpr>),
    RepeatRange {
        expr: Box<GrammarExpr>,
        min: usize,
        max: usize,
    },
    Literal(Vec<u8>),
    CharClass { def: String, negate: bool, utf8: bool },
    RawRegex(String),
    LexerDfa(Arc<LexerDFA>),
    AnyByte,
    /// A separator-delimited sequence of items where some items are optional.
    ///
    /// `items` is an ordered list of `(item_expr, is_required)` pairs.
    /// The sequence allows any subset of items (respecting order) where all
    /// required items are present and optional items may be omitted.
    /// Items that are present are joined by `separator` between consecutive ones.
    ///
    /// This generalises the "ordered object" pattern from JSON Schema (comma-separated
    /// key-value pairs where some keys are optional) to arbitrary grammars.
    SeparatedSequence {
        items: Vec<(GrammarExpr, bool)>,
        separator: Box<GrammarExpr>,
        allow_empty: bool,
    },
    /// An NFA whose transition labels are indices into a side table of grammar
    /// expressions.
    ///
    /// This is only valid as the complete expression of a nonterminal rule.
    /// GLRM likewise serializes it as a named top-level definition.
    ExprNFA(Box<ExprNFA>),
}

/// Controls the tree shape used when lowering [`GrammarExpr::SeparatedSequence`].
///
/// The shape determines how the item list is recursively split into subtrees,
/// which affects parse-path counts and grammar size. Configure via the
/// `GLRMASK_ORDERED_OBJECT_SHAPE` environment variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommaSepShape {
    /// Split at the midpoint (balanced binary tree).
    Balanced,
    /// Always split one item from the left (left-linear tree). Default.
    Left,
    /// Always split one item from the right (right-linear / factored tree).
    Right,
    /// Split at the first optional item boundary; fall back to balanced.
    LeftBalanced,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NamedRule {
    pub name: String,
    pub expr: GrammarExpr,
    pub is_terminal: bool,
    /// Internal-only terminals exist solely as sub-expressions of other
    /// terminal rules (resolved via `Expr::Shared`). They do not produce
    /// their own `TerminalID` or parser production.
    pub is_internal: bool,
}

#[derive(Debug, Clone)]
pub struct NamedGrammar {
    pub rules: Vec<NamedRule>,
    pub start: String,
    /// Name of the terminal rule whose body should be used as the ignore pattern.
    /// Set by Lark's `%ignore` directive.
    pub ignore: Option<String>,
}

impl NamedGrammar {
    /// Returns the set of rule names marked as terminals.
    pub fn terminal_names_set(&self) -> HashSet<String> {
        self.rules
            .iter()
            .filter(|r| r.is_terminal)
            .map(|r| r.name.clone())
            .collect()
    }

    /// Remove rules that are not reachable from the start rule (or ignore rule).
    ///
    /// Traverses `GrammarExpr::Ref` edges to find all rules reachable from
    /// `self.start` (and `self.ignore` if set), then returns a new grammar
    /// containing only those rules in their original order.
    pub fn prune_unreachable(&self) -> Self {
        fn collect_refs(expr: &GrammarExpr, out: &mut HashSet<String>) {
            match expr {
                GrammarExpr::Ref(name) => { out.insert(name.clone()); }
                GrammarExpr::Grouped(inner) => collect_refs(inner, out),
                GrammarExpr::Sequence(items) => { for e in items { collect_refs(e, out); } }
                GrammarExpr::Choice(alts) => { for e in alts { collect_refs(e, out); } }
                GrammarExpr::Exclude { expr, exclude } => {
                    collect_refs(expr, out); collect_refs(exclude, out);
                }
                GrammarExpr::Intersect { expr, intersect } => {
                    collect_refs(expr, out); collect_refs(intersect, out);
                }
                GrammarExpr::Optional(e) | GrammarExpr::Repeat(e) | GrammarExpr::RepeatOne(e) => {
                    collect_refs(e, out);
                }
                GrammarExpr::RepeatRange { expr, .. } => collect_refs(expr, out),
                GrammarExpr::SeparatedSequence { items, separator, .. } => {
                    for (e, _) in items { collect_refs(e, out); }
                    collect_refs(separator, out);
                }
                GrammarExpr::ExprNFA(expr_nfa) => {
                    for symbol in &expr_nfa.symbols {
                        collect_refs(symbol, out);
                    }
                }
                GrammarExpr::Epsilon | GrammarExpr::Literal(_)
                | GrammarExpr::CharClass { .. } | GrammarExpr::RawRegex(_)
                | GrammarExpr::LexerDfa(_)
                | GrammarExpr::AnyByte => {}
            }
        }

        let rule_map: HashMap<String, &NamedRule> = self.rules.iter()
            .map(|r| (r.name.clone(), r))
            .collect();

        let mut reachable: HashSet<String> = HashSet::new();
        let mut worklist: Vec<String> = vec![self.start.clone()];
        if let Some(ref ign) = self.ignore {
            worklist.push(ign.clone());
        }

        while let Some(name) = worklist.pop() {
            if !reachable.insert(name.clone()) { continue; }
            if let Some(rule) = rule_map.get(&name) {
                let mut refs = HashSet::new();
                collect_refs(&rule.expr, &mut refs);
                for r in refs {
                    if !reachable.contains(&r) {
                        worklist.push(r);
                    }
                }
            }
        }

        let rules = self.rules.iter()
            .filter(|r| reachable.contains(&r.name))
            .cloned()
            .collect();

        NamedGrammar { rules, start: self.start.clone(), ignore: self.ignore.clone() }
    }
}


impl NamedGrammar {
    /// Dump the grammar in a Lark-like human-readable form.
    ///
    /// Rendering is intentionally delegated to `grammar_ir::render::lark`; the
    /// named grammar data type keeps this convenience method only so existing
    /// callers do not need to know about the renderer module.
    pub fn to_lark(&self) -> String {
        crate::grammar_ir::render::lark::to_lark(self)
    }
}
