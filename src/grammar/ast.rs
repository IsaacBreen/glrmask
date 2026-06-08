use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use crate::GlrMaskError;
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::dfa::DFA as LexerDFA;
use crate::automata::unweighted_u32::dfa::DFA;
use crate::automata::lexer::regex::parse_regex;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::{
    GrammarDef, NonterminalID, Rule, Symbol, Terminal, TerminalID,
};
use crate::grammar::expr_nfa::ExprNFA;

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

/// Read the `CommaSepShape` from the `GLRMASK_ORDERED_OBJECT_SHAPE` environment variable.
pub fn comma_sep_shape() -> CommaSepShape {
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

    /// Dump the grammar in a Lark-like human-readable format.
    ///
    /// `GrammarExpr` variants with no direct Lark equivalent (e.g. `Exclude`,
    /// `TerminalExpr`) are preserved as inline comments so the dump still
    /// reflects the full original grammar structure.
    pub fn to_lark(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        // Start rule
        writeln!(out, "// start: {}", self.start).unwrap();
        if let Some(ref ign) = self.ignore {
            writeln!(out, "%ignore {}", ign).unwrap();
        }
        writeln!(out).unwrap();

        // Terminal rules first, then nonterminal rules
        let terminals: Vec<_> = self.rules.iter().filter(|r| r.is_terminal).collect();
        let nonterminals: Vec<_> = self.rules.iter().filter(|r| !r.is_terminal).collect();

        if !nonterminals.is_empty() {
            writeln!(out, "// === Nonterminal rules ===").unwrap();
            for rule in &nonterminals {
                let prefix = if rule.is_internal { "// [internal] " } else { "" };
                write!(out, "{}{}: ", prefix, rule.name).unwrap();
                grammar_expr_to_lark(&rule.expr, &mut out, false);
                writeln!(out).unwrap();
            }
            writeln!(out).unwrap();
        }

        if !terminals.is_empty() {
            writeln!(out, "// === Terminal rules ===").unwrap();
            for rule in &terminals {
                let prefix = if rule.is_internal { "// [internal] " } else { "" };
                write!(out, "{}{}: ", prefix, rule.name).unwrap();
                grammar_expr_to_lark(&rule.expr, &mut out, false);
                writeln!(out).unwrap();
            }
        }

        out
    }
}

/// Format a `GrammarExpr` in Lark-like syntax. `parens` controls whether
/// compound expressions get wrapped in parentheses for disambiguation.
fn grammar_expr_to_lark(expr: &GrammarExpr, out: &mut String, parens: bool) {
    grammar_expr_to_lark_with_indent(expr, out, parens, 0);
}

fn grammar_expr_to_lark_with_indent(
    expr: &GrammarExpr,
    out: &mut String,
    parens: bool,
    indent: usize,
) {
    use std::fmt::Write;
    match expr {
        GrammarExpr::Ref(name) => {
            out.push_str(name);
        }
        GrammarExpr::Grouped(inner) => {
            out.push('(');
            grammar_expr_to_lark_with_indent(inner, out, false, indent);
            out.push(')');
        }
        GrammarExpr::Sequence(items) => {
            if parens && items.len() > 1 {
                out.push('(');
            }
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                grammar_expr_to_lark_with_indent(item, out, true, indent);
            }
            if parens && items.len() > 1 {
                out.push(')');
            }
        }
        GrammarExpr::Choice(alts) => {
            let multiline = alts.len() > 6;
            if parens && alts.len() > 1 {
                out.push('(');
            }
            for (i, alt) in alts.iter().enumerate() {
                if i > 0 {
                    if multiline {
                        out.push('\n');
                        for _ in 0..(indent + 4) {
                            out.push(' ');
                        }
                        out.push_str("| ");
                    } else {
                        out.push_str(" | ");
                    }
                }
                let child_indent = if multiline { indent + 6 } else { indent };
                grammar_expr_to_lark_with_indent(alt, out, true, child_indent);
            }
            if parens && alts.len() > 1 {
                if multiline {
                    out.push('\n');
                    for _ in 0..indent {
                        out.push(' ');
                    }
                }
                out.push(')');
            }
        }
        GrammarExpr::Literal(bytes) => {
            // Try UTF-8 first; fall back to hex
            if let Ok(s) = std::str::from_utf8(bytes) {
                write!(out, "\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")).unwrap();
            } else {
                let hex_str: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
                write!(out, "/*hex:{}*/", hex_str).unwrap();
            }
        }
        GrammarExpr::Optional(inner) => {
            grammar_expr_to_lark_with_indent(inner, out, true, indent);
            out.push('?');
        }
        GrammarExpr::Repeat(inner) => {
            grammar_expr_to_lark_with_indent(inner, out, true, indent);
            out.push('*');
        }
        GrammarExpr::RepeatOne(inner) => {
            grammar_expr_to_lark_with_indent(inner, out, true, indent);
            out.push('+');
        }
        GrammarExpr::RepeatRange { expr: inner, min, max } => {
            grammar_expr_to_lark_with_indent(inner, out, true, indent);
            write!(out, "~{}..{}", min, max).unwrap();
        }
        GrammarExpr::Epsilon => {
            out.push_str("/*eps*/");
        }
        GrammarExpr::Exclude { expr: inner, exclude } => {
            write!(out, "/*Exclude(").unwrap();
            grammar_expr_to_lark_with_indent(inner, out, false, indent);
            write!(out, " \\ ").unwrap();
            grammar_expr_to_lark_with_indent(exclude, out, false, indent);
            write!(out, ")*/").unwrap();
        }
        GrammarExpr::Intersect { expr: inner, intersect } => {
            write!(out, "/*Intersect(").unwrap();
            grammar_expr_to_lark_with_indent(inner, out, false, indent);
            write!(out, " & ").unwrap();
            grammar_expr_to_lark_with_indent(intersect, out, false, indent);
            write!(out, ")*/").unwrap();
        }
        GrammarExpr::CharClass { def, negate, utf8 } => {
            if *negate {
                write!(out, "[^{}]", def).unwrap();
            } else {
                write!(out, "[{}]", def).unwrap();
            }
            if *utf8 {
                write!(out, "/*utf8*/").unwrap();
            }
        }
        GrammarExpr::RawRegex(pattern) => {
            write!(out, "/{}/", pattern).unwrap();
        }
        GrammarExpr::LexerDfa(dfa) => {
            write!(out, "/*LexerDfa(states={})*/", dfa.num_states()).unwrap();
        }
        GrammarExpr::AnyByte => {
            out.push_str("/./ /*AnyByte*/");
        }
        GrammarExpr::SeparatedSequence { items, separator, allow_empty } => {
            write!(out, "/*SeparatedSequence(sep=").unwrap();
            grammar_expr_to_lark_with_indent(separator, out, false, indent);
            write!(out, ", allow_empty={}, items=[", allow_empty).unwrap();
            for (i, (item, required)) in items.iter().enumerate() {
                if i > 0 { write!(out, ", ").unwrap(); }
                grammar_expr_to_lark_with_indent(item, out, true, indent);
                if !required { write!(out, "?").unwrap(); }
            }
            write!(out, "])*/").unwrap();
        }
        GrammarExpr::ExprNFA(expr_nfa) => {
            write!(
                out,
                "/*ExprNFA(states={}, symbols={})*/",
                expr_nfa.nfa.states.len(),
                expr_nfa.symbols.len()
            )
            .unwrap();
        }
    }
}

struct Lowerer {
    rules: Vec<Rule>,
    terminal_map: BTreeMap<String, TerminalID>,
    terminals: Vec<Terminal>,
    nonterminal_ids: BTreeMap<String, NonterminalID>,
    generated_nonterminal_counter: u32,
    terminal_names: BTreeMap<TerminalID, String>,
    internal_terminal_names: HashSet<String>,
    named_rule_exprs: HashMap<String, GrammarExpr>,
    named_rule_is_terminal: HashMap<String, bool>,
    rule_nullable: HashMap<String, bool>,
    terminal_bodies: HashMap<String, GrammarExpr>,
    terminal_expr_cache: HashMap<String, Arc<Expr>>,
    nonnullable_named_rule_cache: HashMap<String, NonterminalID>,
    /// Shared cache for repeat-exact nonterminals, keyed by (symbol, count).
    repeat_exact_cache: BTreeMap<(Symbol, usize), NonterminalID>,
    /// Shared cache for repeat-range nonterminals, keyed by (symbol, min, max).
    /// Only used for Left/Right shapes (bucket-based decomposition).
    repeat_range_cache: BTreeMap<(Symbol, usize, usize), NonterminalID>,
    /// Shared cache for repeat-max nonterminals, keyed by (symbol, max).
    /// Used by LeftBalanced/Balanced shapes for O(log N) range decomposition.
    repeat_max_cache: BTreeMap<(Symbol, usize), NonterminalID>,
    /// Shared cache for repeat-min1-max nonterminals, keyed by (symbol, max).
    /// repeat_min1_max_N matches exactly 1..N elements (N >= 1).
    repeat_min1_max_cache: BTreeMap<(Symbol, usize), NonterminalID>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RepeatTreeShape {
    Balanced,
    Left,
    Right,
    /// Deterministic bounded-range lowering that uses a countdown chain.
    Countdown,
    /// Balanced exact decomposition (O(log N) tree depth) with balanced
    /// range alternation (O(log N) close-bracket resolution).
    LeftBalanced,
}

fn repeat_tree_shape() -> RepeatTreeShape {
    match std::env::var("GLRMASK_REPEAT_TREE_SHAPE").ok().as_deref() {
        Some(v) => repeat_tree_shape_from_value(v),
        None => RepeatTreeShape::Balanced,
    }
}

fn repeat_tree_shape_from_value(value: &str) -> RepeatTreeShape {
    match value {
        "left" => RepeatTreeShape::Left,
        "balanced" => RepeatTreeShape::Balanced,
        "countdown" | "deterministic" => RepeatTreeShape::Countdown,
        "leftbalanced" | "left_balanced" => RepeatTreeShape::LeftBalanced,
        _ => RepeatTreeShape::Right,
    }
}

fn right_repeat_range_front_bucket() -> usize {
    std::env::var("GLRMASK_RIGHT_REPEAT_RANGE_FRONT_BUCKET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128)
}

fn left_repeat_range_back_bucket() -> usize {
    std::env::var("GLRMASK_LEFT_REPEAT_RANGE_BACK_BUCKET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128)
}

fn exact_repeat_split(count: usize, shape: RepeatTreeShape) -> (usize, usize) {
    debug_assert!(count > 1);
    match shape {
        RepeatTreeShape::Balanced | RepeatTreeShape::Countdown | RepeatTreeShape::LeftBalanced => {
            let left = count / 2;
            (left, count - left)
        }
        RepeatTreeShape::Left => (count - 1, 1),
        RepeatTreeShape::Right => (1, count - 1),
    }
}

fn range_repeat_split(min: usize, max: usize, shape: RepeatTreeShape) -> (usize, usize) {
    debug_assert!(min < max);
    let width = max - min + 1;
    match shape {
        RepeatTreeShape::Balanced | RepeatTreeShape::Countdown | RepeatTreeShape::LeftBalanced => {
            let left_width = width / 2;
            let split = min + left_width - 1;
            (split, width - left_width)
        }
        RepeatTreeShape::Left => (max - 1, 1),
        RepeatTreeShape::Right => (min, width - 1),
    }
}

fn highest_power_of_two_le(n: usize) -> usize {
    debug_assert!(n > 0);
    1usize << ((usize::BITS - 1 - n.leading_zeros()) as usize)
}

fn char_class_pattern(def: &str, negate: bool) -> String {
    if negate {
        format!("[^{def}]")
    } else {
        format!("[{def}]")
    }
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

    fn exact_subtraction_alternatives(
        &self,
        lhs_name: &str,
        exclude: &GrammarExpr,
    ) -> Result<Vec<GrammarExpr>, GlrMaskError> {
        match exclude {
            GrammarExpr::Choice(options) => {
                let mut out = Vec::new();
                for option in options {
                    out.extend(self.exact_subtraction_alternatives(lhs_name, option)?);
                }
                Ok(out)
            }
            GrammarExpr::Grouped(inner) => Ok(Self::top_level_alternatives(inner)),
            GrammarExpr::Ref(name) => {
                let Some(false) = self.named_rule_is_terminal.get(name).copied() else {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "{lhs_name} - {name} requires {name} to name a nonterminal rule"
                    )));
                };
                let referenced_expr = self.named_rule_exprs.get(name).ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!(
                        "unknown rule referenced in exact alternative subtraction: {name}"
                    ))
                })?;
                Ok(Self::top_level_alternatives(referenced_expr))
            }
            other => Ok(Self::top_level_alternatives(other)),
        }
    }

    fn canonical_exact_expr(&self, expr: &GrammarExpr) -> GrammarExpr {
        let mut visiting = HashSet::new();
        let mut memo = HashMap::new();
        self.canonical_exact_expr_inner(expr, &mut visiting, &mut memo)
    }

    fn canonical_exact_expr_inner(
        &self,
        expr: &GrammarExpr,
        visiting: &mut HashSet<String>,
        memo: &mut HashMap<String, GrammarExpr>,
    ) -> GrammarExpr {
        match Self::strip_grouping(expr) {
            GrammarExpr::Ref(name) => {
                if self.named_rule_is_terminal.get(name).copied().unwrap_or(false) {
                    return GrammarExpr::Ref(name.clone());
                }
                let Some(referenced) = self.named_rule_exprs.get(name) else {
                    return GrammarExpr::Ref(name.clone());
                };
                if let Some(canonical) = memo.get(name) {
                    return canonical.clone();
                }
                if !visiting.insert(name.clone()) {
                    return GrammarExpr::Ref(name.clone());
                }
                let canonical = self.canonical_exact_expr_inner(referenced, visiting, memo);
                visiting.remove(name);
                memo.insert(name.clone(), canonical.clone());
                canonical
            }
            GrammarExpr::Grouped(inner) => self.canonical_exact_expr_inner(inner, visiting, memo),
            GrammarExpr::Sequence(items) => GrammarExpr::Sequence(
                items
                    .iter()
                    .map(|item| self.canonical_exact_expr_inner(item, visiting, memo))
                    .collect(),
            ),
            GrammarExpr::Choice(items) => GrammarExpr::Choice(
                items
                    .iter()
                    .map(|item| self.canonical_exact_expr_inner(item, visiting, memo))
                    .collect(),
            ),
            GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
                expr: Box::new(self.canonical_exact_expr_inner(expr, visiting, memo)),
                exclude: Box::new(self.canonical_exact_expr_inner(exclude, visiting, memo)),
            },
            GrammarExpr::Intersect { expr, intersect } => GrammarExpr::Intersect {
                expr: Box::new(self.canonical_exact_expr_inner(expr, visiting, memo)),
                intersect: Box::new(self.canonical_exact_expr_inner(intersect, visiting, memo)),
            },
            GrammarExpr::Optional(inner) => GrammarExpr::Optional(Box::new(
                self.canonical_exact_expr_inner(inner, visiting, memo),
            )),
            GrammarExpr::Repeat(inner) => GrammarExpr::Repeat(Box::new(
                self.canonical_exact_expr_inner(inner, visiting, memo),
            )),
            GrammarExpr::RepeatOne(inner) => GrammarExpr::RepeatOne(Box::new(
                self.canonical_exact_expr_inner(inner, visiting, memo),
            )),
            GrammarExpr::RepeatRange { expr, min, max } => GrammarExpr::RepeatRange {
                expr: Box::new(self.canonical_exact_expr_inner(expr, visiting, memo)),
                min: *min,
                max: *max,
            },
            GrammarExpr::SeparatedSequence {
                items,
                separator,
                allow_empty,
            } => GrammarExpr::SeparatedSequence {
                items: items
                    .iter()
                    .map(|(item, required)| {
                        (
                            self.canonical_exact_expr_inner(item, visiting, memo),
                            *required,
                        )
                    })
                    .collect(),
                separator: Box::new(self.canonical_exact_expr_inner(separator, visiting, memo)),
                allow_empty: *allow_empty,
            },
            GrammarExpr::ExprNFA(expr_nfa) => GrammarExpr::ExprNFA(Box::new(ExprNFA {
                nfa: expr_nfa.nfa.clone(),
                symbols: expr_nfa
                    .symbols
                    .iter()
                    .map(|symbol| self.canonical_exact_expr_inner(symbol, visiting, memo))
                    .collect(),
            })),
            GrammarExpr::Epsilon
            | GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => Self::strip_grouping(expr).clone(),
        }
    }

    fn exact_nonterminal_subtraction_expr(
        &self,
        expr: &GrammarExpr,
    ) -> Result<Option<GrammarExpr>, GlrMaskError> {
        let GrammarExpr::Exclude { expr: lhs_expr, exclude } = expr else {
            return Ok(None);
        };
        let GrammarExpr::Ref(lhs_name) = Self::strip_grouping(lhs_expr) else {
            return Ok(None);
        };
        let Some(false) = self.named_rule_is_terminal.get(lhs_name).copied() else {
            return Ok(None);
        };

        let lhs_rule_expr = self.named_rule_exprs.get(lhs_name).ok_or_else(|| {
            GlrMaskError::GrammarParse(format!(
                "unknown nonterminal referenced in exact alternative subtraction: {lhs_name}"
            ))
        })?;
        let mut remaining = Self::top_level_alternatives(lhs_rule_expr);
        let mut remaining_keys = remaining
            .iter()
            .map(|candidate| self.canonical_exact_expr(candidate))
            .collect::<Vec<_>>();
        for remove_alt in self.exact_subtraction_alternatives(lhs_name, exclude)? {
            let remove_alt_key = self.canonical_exact_expr(&remove_alt);
            let Some(position) = remaining_keys
                .iter()
                .position(|candidate| candidate == &remove_alt_key)
            else {
                continue;
            };
            remaining.remove(position);
            remaining_keys.remove(position);
        }

        Ok(Some(match remaining.len() {
            0 => GrammarExpr::Choice(Vec::new()),
            1 => remaining.pop().unwrap(),
            _ => GrammarExpr::Choice(remaining),
        }))
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

    fn repeat_exact_nonterminal(
        &mut self,
        symbol: &Symbol,
        count: usize,
        shape: RepeatTreeShape,
    ) -> NonterminalID {
        let key = (symbol.clone(), count);
        if let Some(&nonterminal) = self.repeat_exact_cache.get(&key) {
            return nonterminal;
        }

        let (_, nonterminal) = self.fresh_nonterminal("repeat_exact");
        self.repeat_exact_cache.insert(key, nonterminal);
        match count {
            0 => self.rules.push(Rule {
                lhs: nonterminal,
                rhs: Vec::new(),
            }),
            1 => self.rules.push(Rule {
                lhs: nonterminal,
                rhs: vec![symbol.clone()],
            }),
            _ => {
                let (left, right) = exact_repeat_split(count, shape);
                let left_nonterminal =
                    self.repeat_exact_nonterminal(symbol, left, shape);
                let right_nonterminal =
                    self.repeat_exact_nonterminal(symbol, right, shape);
                self.rules.push(Rule {
                    lhs: nonterminal,
                    rhs: vec![
                        Symbol::Nonterminal(left_nonterminal),
                        Symbol::Nonterminal(right_nonterminal),
                    ],
                });
            }
        }
        nonterminal
    }

    fn repeat_max_nonterminal(
        &mut self,
        symbol: &Symbol,
        max: usize,
    ) -> NonterminalID {
        let key = (symbol.clone(), max);
        if let Some(&nt) = self.repeat_max_cache.get(&key) {
            return nt;
        }

        if max == 0 {
            let (_, nt) = self.fresh_nonterminal("repeat_max");
            self.repeat_max_cache.insert(key, nt);
            self.rules.push(Rule {
                lhs: nt,
                rhs: Vec::new(),
            });
            return nt;
        }

        let (_, nt) = self.fresh_nonterminal("repeat_max");
        self.repeat_max_cache.insert(key, nt);

        if let Some(span) = max.checked_add(1).filter(|span| span.is_power_of_two()) {
            let high = span / 2;
            debug_assert!(high > 0);

            let low_nt = self.repeat_max_nonterminal(symbol, high - 1);
            let exact_high_nt =
                self.repeat_exact_nonterminal(symbol, high, RepeatTreeShape::LeftBalanced);

            self.rules.push(Rule {
                lhs: nt,
                rhs: vec![Symbol::Nonterminal(low_nt)],
            });

            let rhs = if high == 1 {
                vec![Symbol::Nonterminal(exact_high_nt)]
            } else {
                vec![
                    Symbol::Nonterminal(exact_high_nt),
                    Symbol::Nonterminal(low_nt),
                ]
            };
            self.rules.push(Rule { lhs: nt, rhs });
        } else {
            let high = highest_power_of_two_le(max);
            debug_assert!(high > 0);
            debug_assert!(high <= max);

            let below_nt = self.repeat_max_nonterminal(symbol, high - 1);
            let exact_high_nt =
                self.repeat_exact_nonterminal(symbol, high, RepeatTreeShape::LeftBalanced);

            self.rules.push(Rule {
                lhs: nt,
                rhs: vec![Symbol::Nonterminal(below_nt)],
            });

            if max == high {
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![Symbol::Nonterminal(exact_high_nt)],
                });
            } else {
                let tail_nt = self.repeat_max_nonterminal(symbol, max - high);
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![
                        Symbol::Nonterminal(exact_high_nt),
                        Symbol::Nonterminal(tail_nt),
                    ],
                });
            }
        }

        nt
    }

    /// Returns a nonterminal matching exactly 1..=max occurrences of `symbol`.
    /// Defined as: `symbol repeat_max_{max-1}` — the first element is mandatory,
    /// the rest (0..max-1) are handled by `repeat_max`.
    fn repeat_min1_max_nonterminal(&mut self, symbol: &Symbol, max: usize) -> NonterminalID {
        debug_assert!(max >= 1);
        let key = (symbol.clone(), max);
        if let Some(&nt) = self.repeat_min1_max_cache.get(&key) {
            return nt;
        }

        let (_, nt) = self.fresh_nonterminal("repeat_min1_max");
        self.repeat_min1_max_cache.insert(key, nt);

        if max == 1 {
            self.rules.push(Rule {
                lhs: nt,
                rhs: vec![symbol.clone()],
            });
            return nt;
        }

        let tail_nt = self.repeat_max_nonterminal(symbol, max - 1);
        self.rules.push(Rule {
            lhs: nt,
            rhs: vec![symbol.clone(), Symbol::Nonterminal(tail_nt)],
        });
        nt
    }

    fn repeat_range_nonterminal(
        &mut self,
        symbol: &Symbol,
        min: usize,
        max: usize,
        shape: RepeatTreeShape,
    ) -> NonterminalID {
        debug_assert!(min <= max);
        if min == max {
            return self.repeat_exact_nonterminal(symbol, min, shape);
        }

        let key = (symbol.clone(), min, max);
        if let Some(&nonterminal) = self.repeat_range_cache.get(&key) {
            return nonterminal;
        }

        if shape == RepeatTreeShape::Countdown {
            return self.repeat_range_nonterminal_countdown(symbol, min, max, shape);
        }

        match shape {
            RepeatTreeShape::LeftBalanced | RepeatTreeShape::Balanced => {
                return self.repeat_range_nonterminal_balanced(symbol, min, max, shape);
            }
            _ => {}
        }

        let (_, nonterminal) = self.fresh_nonterminal("repeat_range");
        self.repeat_range_cache.insert(key, nonterminal);

        match shape {
            RepeatTreeShape::Right if (max - min + 1) > right_repeat_range_front_bucket() => {
                let cutoff = (min + right_repeat_range_front_bucket() - 1).min(max);
                for count in min..=cutoff {
                    let exact_nonterminal =
                        self.repeat_exact_nonterminal(symbol, count, shape);
                    self.rules.push(Rule {
                        lhs: nonterminal,
                        rhs: vec![Symbol::Nonterminal(exact_nonterminal)],
                    });
                }
                if cutoff < max {
                    let tail_nonterminal = self.repeat_range_nonterminal(
                        symbol,
                        cutoff + 1,
                        max,
                        shape,
                    );
                    self.rules.push(Rule {
                        lhs: nonterminal,
                        rhs: vec![Symbol::Nonterminal(tail_nonterminal)],
                    });
                }
                return nonterminal;
            }
            RepeatTreeShape::Left if (max - min + 1) > left_repeat_range_back_bucket() => {
                let cutoff = max.saturating_sub(left_repeat_range_back_bucket() - 1).max(min);
                if min < cutoff {
                    let head_nonterminal = self.repeat_range_nonterminal(
                        symbol,
                        min,
                        cutoff - 1,
                        shape,
                    );
                    self.rules.push(Rule {
                        lhs: nonterminal,
                        rhs: vec![Symbol::Nonterminal(head_nonterminal)],
                    });
                }
                for count in cutoff..=max {
                    let exact_nonterminal =
                        self.repeat_exact_nonterminal(symbol, count, shape);
                    self.rules.push(Rule {
                        lhs: nonterminal,
                        rhs: vec![Symbol::Nonterminal(exact_nonterminal)],
                    });
                }
                return nonterminal;
            }
            _ => {}
        }
        let (split, _) = range_repeat_split(min, max, shape);
        let left_nonterminal = self.repeat_range_nonterminal(
            symbol,
            min,
            split,
            shape,
        );
        let right_nonterminal = self.repeat_range_nonterminal(
            symbol,
            split + 1,
            max,
            shape,
        );
        self.rules.push(Rule {
            lhs: nonterminal,
            rhs: vec![Symbol::Nonterminal(left_nonterminal)],
        });
        self.rules.push(Rule {
            lhs: nonterminal,
            rhs: vec![Symbol::Nonterminal(right_nonterminal)],
        });
        nonterminal
    }

    fn repeat_range_nonterminal_countdown(
        &mut self,
        symbol: &Symbol,
        min: usize,
        max: usize,
        shape: RepeatTreeShape,
    ) -> NonterminalID {
        let key = (symbol.clone(), min, max);
        if let Some(&nonterminal) = self.repeat_range_cache.get(&key) {
            return nonterminal;
        }

        let (_, nonterminal) = self.fresh_nonterminal("repeat_range");
        self.repeat_range_cache.insert(key, nonterminal);

        if min == 0 {
            self.rules.push(Rule {
                lhs: nonterminal,
                rhs: Vec::new(),
            });
        }

        let tail_nonterminal = if min > 0 {
            self.repeat_range_nonterminal(symbol, min - 1, max - 1, shape)
        } else {
            self.repeat_range_nonterminal(symbol, 0, max - 1, shape)
        };
        self.rules.push(Rule {
            lhs: nonterminal,
            rhs: vec![symbol.clone(), Symbol::Nonterminal(tail_nonterminal)],
        });
        nonterminal
    }

    fn repeat_range_nonterminal_balanced(
        &mut self,
        symbol: &Symbol,
        min: usize,
        max: usize,
        shape: RepeatTreeShape,
    ) -> NonterminalID {
        debug_assert!(min <= max);
        debug_assert!(matches!(shape, RepeatTreeShape::LeftBalanced | RepeatTreeShape::Balanced));
        if min == max {
            return self.repeat_exact_nonterminal(symbol, min, shape);
        }
        let delta = max - min;
        let max_nt = self.repeat_max_nonterminal(symbol, delta);
        if min == 0 {
            max_nt
        } else {
            let exact_nt = self.repeat_exact_nonterminal(symbol, min, shape);
            let (_, result_nt) = self.fresh_nonterminal("repeat_range");
            self.rules.push(Rule {
                lhs: result_nt,
                rhs: vec![
                    Symbol::Nonterminal(exact_nt),
                    Symbol::Nonterminal(max_nt),
                ],
            });
            result_nt
        }
    }

    fn emit_repeat_range(
        &mut self,
        lhs: NonterminalID,
        inner: &GrammarExpr,
        min: usize,
        max: usize,
    ) -> Result<(), GlrMaskError> {
        debug_assert!(min <= max);
        let symbol = self.lower_expr_terminalish(inner)?;
        let shape = repeat_tree_shape();
        let range_nonterminal = self.repeat_range_nonterminal(
            &symbol,
            min,
            max,
            shape,
        );
        self.rules.push(Rule {
            lhs,
            rhs: vec![Symbol::Nonterminal(range_nonterminal)],
        });
        Ok(())
    }

    fn expr_nfa_state_nonterminals(
        &mut self,
        state_count: usize,
        start: usize,
        hint: &str,
        start_lhs: Option<NonterminalID>,
    ) -> Result<Vec<NonterminalID>, GlrMaskError> {
        if start >= state_count {
            return Err(GlrMaskError::GrammarParse(format!(
                "ExprNFA start state {} is out of range for {} states",
                start, state_count
            )));
        }

        let mut nts = Vec::with_capacity(state_count);
        for state_index in 0..state_count {
            if Some(state_index) == start_lhs.map(|_| start) {
                nts.push(start_lhs.unwrap());
            } else {
                let (_, nt) = self.fresh_nonterminal(hint);
                nts.push(nt);
            }
        }
        Ok(nts)
    }

    fn emit_expr_dfa_leftlinear(
        &mut self,
        lhs: NonterminalID,
        expr_nfa: &ExprNFA,
        dfa: &DFA,
    ) -> Result<(), GlrMaskError> {
        let state_count = dfa.states.len();
        let start = dfa.start_state as usize;
        let nts = self.expr_nfa_state_nonterminals(state_count, start, "expr_nfa_prefix", None)?;
        let start_nt = nts[start];
        self.rules.push(Rule {
            lhs: start_nt,
            rhs: Vec::new(),
        });

        for (state_index, state) in dfa.states.iter().enumerate() {
            for (label, target) in &state.transitions {
                let target = *target as usize;
                if target >= state_count {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition from state {state_index} targets out-of-range state {target}"
                    )));
                }
                let symbol_expr = expr_nfa.symbol_for_label(*label).ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition label {label} is not a valid symbol index"
                    ))
                })?;
                let symbol = self.lower_expr_terminalish(symbol_expr)?;
                self.rules.push(Rule {
                    lhs: nts[target],
                    rhs: vec![Symbol::Nonterminal(nts[state_index]), symbol],
                });
            }
        }

        for (state_index, state) in dfa.states.iter().enumerate() {
            if state.is_accepting {
                self.rules.push(Rule {
                    lhs,
                    rhs: vec![Symbol::Nonterminal(nts[state_index])],
                });
            }
        }

        Ok(())
    }

    fn emit_expr_dfa_leftlinear_nonnullable(
        &mut self,
        lhs: NonterminalID,
        expr_nfa: &ExprNFA,
        dfa: &DFA,
    ) -> Result<(), GlrMaskError> {
        let state_count = dfa.states.len();
        let start = dfa.start_state as usize;
        if start >= state_count {
            return Err(GlrMaskError::GrammarParse(format!(
                "ExprNFA start state {} is out of range for {} states",
                dfa.start_state, state_count
            )));
        }

        let nullable_nts = self.expr_nfa_state_nonterminals(
            state_count,
            start,
            "expr_nfa_nullable_prefix",
            None,
        )?;
        let nonnullable_nts = self.expr_nfa_state_nonterminals(
            state_count,
            start,
            "expr_nfa_nonnullable_prefix",
            None,
        )?;

        self.rules.push(Rule {
            lhs: nullable_nts[start],
            rhs: Vec::new(),
        });

        for (state_index, state) in dfa.states.iter().enumerate() {
            for (label, target) in &state.transitions {
                let target = *target as usize;
                if target >= state_count {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition from state {state_index} targets out-of-range state {target}"
                    )));
                }
                let symbol_expr = expr_nfa.symbol_for_label(*label).ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition label {label} is not a valid symbol index"
                    ))
                })?;

                if self.expr_is_nullable(symbol_expr) {
                    let symbol = self.lower_expr_terminalish(symbol_expr)?;
                    self.rules.push(Rule {
                        lhs: nullable_nts[target],
                        rhs: vec![Symbol::Nonterminal(nullable_nts[state_index]), symbol],
                    });
                }

                if let Some(symbol) = self.lower_nonnullable_expr_symbol(symbol_expr)? {
                    self.rules.push(Rule {
                        lhs: nonnullable_nts[target],
                        rhs: vec![Symbol::Nonterminal(nullable_nts[state_index]), symbol],
                    });
                }

                let symbol = self.lower_expr_terminalish(symbol_expr)?;
                self.rules.push(Rule {
                    lhs: nonnullable_nts[target],
                    rhs: vec![Symbol::Nonterminal(nonnullable_nts[state_index]), symbol],
                });
            }
        }

        for (state_index, state) in dfa.states.iter().enumerate() {
            if state.is_accepting {
                self.rules.push(Rule {
                    lhs,
                    rhs: vec![Symbol::Nonterminal(nonnullable_nts[state_index])],
                });
            }
        }

        Ok(())
    }

    fn emit_expr_nfa(&mut self, lhs: NonterminalID, expr_nfa: &ExprNFA) -> Result<(), GlrMaskError> {
        let dfa = expr_nfa.determinize_and_minimize();
        self.emit_expr_dfa_leftlinear(lhs, expr_nfa, &dfa)
    }

    fn emit_expr_nfa_nonnullable(
        &mut self,
        lhs: NonterminalID,
        expr_nfa: &ExprNFA,
    ) -> Result<(), GlrMaskError> {
        let dfa = expr_nfa.determinize_and_minimize();
        self.emit_expr_dfa_leftlinear_nonnullable(lhs, expr_nfa, &dfa)
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

    fn lower_sepseq_repetition_item_nonempty_symbol(
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

    fn lower_sepseq_item_nonempty_symbol(
        &mut self,
        item_expr: &GrammarExpr,
        separator: &GrammarExpr,
    ) -> Result<Option<Symbol>, GlrMaskError> {
        match item_expr {
            GrammarExpr::Repeat(inner) => {
                self.lower_sepseq_repetition_item_nonempty_symbol(inner, separator, 1, None)
            }
            GrammarExpr::RepeatOne(inner) => {
                self.lower_sepseq_repetition_item_nonempty_symbol(inner, separator, 1, None)
            }
            GrammarExpr::RepeatRange { expr, min, max } => {
                let required = (*min).max(1);
                self.lower_sepseq_repetition_item_nonempty_symbol(expr, separator, required, Some(*max))
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
    fn lower_separated_sequence_inner(
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
            let item_sym = self.lower_sepseq_item_nonempty_symbol(item_expr, separator)?;
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

    /// Register a pre-resolved terminal Expr, deduplicating by value.
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

/// Convert a GrammarExpr to an Expr tree, resolving terminal Ref nodes
/// via the `terminal_bodies` map and caching results in `terminal_expr_cache`.
fn grammar_expr_to_expr(
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

fn grammar_expr_is_nullable(
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

fn compute_rule_nullability(grammar: &NamedGrammar) -> HashMap<String, bool> {
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

fn validate_expr_nfa_placement(grammar: &NamedGrammar) -> Result<(), GlrMaskError> {
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

/// Encode a [`U8Set`] as a character-class definition string (without the surrounding `[...]`).
///
/// Uses range notation where possible. Always produces a non-negated form.
pub(crate) fn u8set_to_class_def(set: &U8Set) -> String {
    let mut out = String::new();
    let bytes: Vec<u8> = set.iter().collect();
    let mut i = 0usize;
    while i < bytes.len() {
        let start = bytes[i];
        let mut end = start;
        i += 1;
        while i < bytes.len() && bytes[i] == end.wrapping_add(1) && end < 255 {
            end = bytes[i];
            i += 1;
        }
        push_class_char(&mut out, start);
        if end != start {
            if end == start + 1 {
                push_class_char(&mut out, end);
            } else {
                out.push('-');
                push_class_char(&mut out, end);
            }
        }
    }
    out
}

fn push_class_char(out: &mut String, b: u8) {
    use std::fmt::Write;
    match b {
        b'\\' => out.push_str("\\\\"),
        b']' => out.push_str("\\]"),
        b'-' => out.push_str("\\-"),
        b'^' => out.push_str("\\^"),
        0x20..=0x7E => out.push(b as char),
        _ => write!(out, "\\x{:02X}", b).unwrap(),
    }
}

fn dedup_rules_preserving_first_occurrence(rules: &mut Vec<Rule>) {
    let mut seen = HashSet::new();
    rules.retain(|rule| seen.insert(rule.clone()));
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

fn escape_byte(b: u8) -> String {
    match b {
        b'\n' => "\\n".into(),
        b'\r' => "\\r".into(),
        b'\t' => "\\t".into(),
        b'\\' => "\\\\".into(),
        b'"' => "\\\"".into(),
        byte if byte.is_ascii_graphic() || byte == b' ' => (byte as char).to_string(),
        byte => format!("\\x{byte:02x}"),
    }
}

fn regex_escape_byte(b: u8) -> String {
    match b {
        b'.' | b'+' | b'*' | b'?' | b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'|' | b'^' | b'$' | b'\\' => {
            format!("\\{}", b as char)
        }
        _ => escape_byte(b),
    }
}

#[cfg(test)]
mod tests {
    use super::{lower, GrammarExpr, NamedGrammar, NamedRule};

    fn nonterminal(name: &str, expr: GrammarExpr) -> NamedRule {
        NamedRule {
            name: name.to_string(),
            expr,
            is_terminal: false,
            is_internal: false,
        }
    }

    fn terminal(name: &str, expr: GrammarExpr) -> NamedRule {
        NamedRule {
            name: name.to_string(),
            expr,
            is_terminal: true,
            is_internal: false,
        }
    }

    fn literal(text: &str) -> GrammarExpr {
        GrammarExpr::Literal(text.as_bytes().to_vec())
    }

    fn subtract(lhs: &str, exclude: GrammarExpr) -> GrammarExpr {
        GrammarExpr::Exclude {
            expr: Box::new(GrammarExpr::Ref(lhs.to_string())),
            exclude: Box::new(exclude),
        }
    }

    #[test]
    fn exact_subtraction_matches_nonterminal_alias_body() {
        let grammar = NamedGrammar {
            rules: vec![
                terminal("JSON_STRING_BODY", literal("body\"")),
                nonterminal(
                    "json_string",
                    GrammarExpr::Sequence(vec![
                        literal("\""),
                        GrammarExpr::Ref("JSON_STRING_BODY".to_string()),
                    ]),
                ),
                nonterminal(
                    "json_value",
                    GrammarExpr::Choice(vec![
                        GrammarExpr::Ref("json_string".to_string()),
                        literal("0"),
                    ]),
                ),
                nonterminal(
                    "start",
                    subtract("json_value", GrammarExpr::Ref("json_string".to_string())),
                ),
            ],
            start: "start".to_string(),
            ignore: None,
        };

        lower(&grammar).unwrap();
    }

    #[test]
    fn exact_subtraction_canonicalization_is_cycle_safe() {
        let grammar = NamedGrammar {
            rules: vec![
                nonterminal(
                    "loop",
                    GrammarExpr::Sequence(vec![
                        GrammarExpr::Ref("loop".to_string()),
                        literal("y"),
                    ]),
                ),
                nonterminal(
                    "A",
                    GrammarExpr::Choice(vec![
                        GrammarExpr::Ref("loop".to_string()),
                        literal("x"),
                    ]),
                ),
                nonterminal("start", subtract("A", literal("z"))),
            ],
            start: "start".to_string(),
            ignore: None,
        };

        let err = lower(&grammar).unwrap_err();
        assert!(format!("{err}").contains("no exact alternative"), "{err}");
    }

    #[test]
    fn lower_deduplicates_identical_rules() {
        let grammar = NamedGrammar {
            rules: vec![nonterminal(
                "start",
                GrammarExpr::Choice(vec![literal("a"), literal("a")]),
            )],
            start: "start".to_string(),
            ignore: None,
        };

        let gdef = lower(&grammar).unwrap();
        let start_rules = gdef
            .rules
            .iter()
            .filter(|rule| rule.lhs == gdef.start)
            .count();
        assert_eq!(start_rules, 1, "duplicate alternatives should not create duplicate rules");
    }

    #[test]
    fn nonnullable_sequence_with_nonnullable_part_reduces_rules() {
        let grammar = NamedGrammar {
            rules: vec![
                nonterminal(
                    "body",
                    GrammarExpr::Choice(vec![
                        literal("abc"),
                        GrammarExpr::Epsilon,
                    ]),
                ),
                nonterminal(
                    "item",
                    GrammarExpr::RepeatOne(Box::new(GrammarExpr::Sequence(vec![
                        literal("{"),
                        GrammarExpr::Ref("body".to_string()),
                        literal("}"),
                    ]))),
                ),
                nonterminal("start", GrammarExpr::Ref("item".to_string())),
            ],
            start: "start".to_string(),
            ignore: None,
        };

        let gdef = lower(&grammar).unwrap();
        let brace_rules_count = gdef
            .rules
            .iter()
            .filter(|rule| {
                matches!(
                    rule.rhs.first(),
                    Some(crate::grammar::flat::Symbol::Terminal(tid))
                        if gdef.terminal_display_name(*tid) == "{"
                )
            })
            .count();
        assert_eq!(
            brace_rules_count,
            1,
            "nonnullable sequence should not synthesize duplicate brace-start alternatives"
        );
    }
}
