use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use crate::GlrMaskError;
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::regex::parse_regex;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::{
    GrammarDef, NonterminalID, Rule, Symbol, Terminal, TerminalID,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GrammarExpr {
    Ref(String),
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
    },
}

/// Controls the tree shape used when lowering [`GrammarExpr::SeparatedSequence`].
///
/// The shape determines how the item list is recursively split into subtrees,
/// which affects parse-path counts and grammar size. Configure via the
/// `GLRMASK_ORDERED_OBJECT_SHAPE` environment variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommaSepShape {
    /// Split at the midpoint (balanced binary tree). Default.
    Balanced,
    /// Always split one item from the left (left-linear tree).
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
        None => CommaSepShape::Balanced,
        Some(_) => CommaSepShape::Balanced,
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
                GrammarExpr::SeparatedSequence { items, separator } => {
                    for (e, _) in items { collect_refs(e, out); }
                    collect_refs(separator, out);
                }
                GrammarExpr::Epsilon | GrammarExpr::Literal(_)
                | GrammarExpr::CharClass { .. } | GrammarExpr::RawRegex(_)
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
        GrammarExpr::AnyByte => {
            out.push_str("/./ /*AnyByte*/");
        }
        GrammarExpr::SeparatedSequence { items, separator } => {
            write!(out, "/*SeparatedSequence(sep=").unwrap();
            grammar_expr_to_lark_with_indent(separator, out, false, indent);
            write!(out, ", items=[").unwrap();
            for (i, (item, required)) in items.iter().enumerate() {
                if i > 0 { write!(out, ", ").unwrap(); }
                grammar_expr_to_lark_with_indent(item, out, true, indent);
                if !required { write!(out, "?").unwrap(); }
            }
            write!(out, "])*/").unwrap();
        }
    }
}

fn format_terminal_expr(expr: &Expr) -> String {
    fn push_class_byte(out: &mut String, b: u8) {
        use std::fmt::Write;
        match b {
            b'\\' => out.push_str("\\\\"),
            b']' => out.push_str("\\]"),
            b'-' => out.push_str("\\-"),
            0x20..=0x7E => out.push(char::from(b)),
            _ => {
                write!(out, "\\x{b:02X}").unwrap();
            }
        }
    }

    fn render(expr: &Expr, out: &mut String, needs_parens: bool) {
        use std::fmt::Write;
        match expr {
            Expr::U8Seq(bytes) => {
                out.push('"');
                for &b in bytes {
                    match b {
                        b'\\' => out.push_str("\\\\"),
                        b'"' => out.push_str("\\\""),
                        b'\n' => out.push_str("\\n"),
                        b'\r' => out.push_str("\\r"),
                        b'\t' => out.push_str("\\t"),
                        0x20..=0x7E => out.push(char::from(b)),
                        _ => {
                            write!(out, "\\x{b:02X}").unwrap();
                        }
                    }
                }
                out.push('"');
            }
            Expr::U8Class(set) => {
                out.push('[');
                let bytes: Vec<u8> = set.iter().collect();
                let mut i = 0usize;
                while i < bytes.len() {
                    let start = bytes[i];
                    let mut end = start;
                    i += 1;
                    while i < bytes.len() && bytes[i] == end.saturating_add(1) {
                        end = bytes[i];
                        i += 1;
                    }

                    if start == end {
                        push_class_byte(out, start);
                    } else {
                        push_class_byte(out, start);
                        out.push('-');
                        push_class_byte(out, end);
                    }
                }
                out.push(']');
            }
            Expr::Seq(parts) => {
                let wrap = needs_parens && parts.len() > 1;
                if wrap {
                    out.push('(');
                }
                for (idx, part) in parts.iter().enumerate() {
                    if idx > 0 {
                        out.push(' ');
                    }
                    render(part, out, true);
                }
                if wrap {
                    out.push(')');
                }
            }
            Expr::Choice(options) => {
                let wrap = needs_parens && options.len() > 1;
                if wrap {
                    out.push('(');
                }
                for (idx, option) in options.iter().enumerate() {
                    if idx > 0 {
                        out.push_str(" | ");
                    }
                    render(option, out, true);
                }
                if wrap {
                    out.push(')');
                }
            }
            Expr::Exclude { expr, exclude } => {
                out.push('(');
                render(expr, out, false);
                out.push_str(" \\ ");
                render(exclude, out, false);
                out.push(')');
            }
            Expr::Intersect { expr, intersect } => {
                out.push('(');
                render(expr, out, false);
                out.push_str(" & ");
                render(intersect, out, false);
                out.push(')');
            }
            Expr::Repeat { expr, min, max } => {
                render(expr, out, true);
                match (*min, *max) {
                    (0, None) => out.push('*'),
                    (1, None) => out.push('+'),
                    (0, Some(1)) => out.push('?'),
                    (m, Some(n)) if m == n => {
                        write!(out, "{{{m}}}").unwrap();
                    }
                    (m, Some(n)) => {
                        write!(out, "{{{m},{n}}}").unwrap();
                    }
                    (m, None) => {
                        write!(out, "{{{m},}}").unwrap();
                    }
                }
            }
            Expr::Shared(inner) => {
                out.push_str("shared(");
                render(inner, out, false);
                out.push(')');
            }
            Expr::Epsilon => out.push_str("<epsilon>"),
        }
    }

    let mut pretty = String::new();
    render(expr, &mut pretty, false);
    pretty
}

struct Lowerer {
    rules: Vec<Rule>,
    terminal_map: BTreeMap<String, TerminalID>,
    terminals: Vec<Terminal>,
    nonterminal_ids: BTreeMap<String, NonterminalID>,
    generated_nonterminal_counter: u32,
    terminal_names: BTreeMap<TerminalID, String>,
    internal_terminal_names: HashSet<String>,
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
    /// Balanced exact decomposition (O(log N) tree depth) with balanced
    /// range alternation (O(log N) close-bracket resolution).
    LeftBalanced,
}

fn repeat_tree_shape() -> RepeatTreeShape {
    match std::env::var("GLRMASK_REPEAT_TREE_SHAPE").ok().as_deref() {
        Some(v) => repeat_tree_shape_from_value(v),
        None => RepeatTreeShape::LeftBalanced,
    }
}

fn repeat_tree_shape_from_value(value: &str) -> RepeatTreeShape {
    match value {
        "left" => RepeatTreeShape::Left,
        "balanced" => RepeatTreeShape::Balanced,
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
        RepeatTreeShape::Balanced | RepeatTreeShape::LeftBalanced => {
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
        RepeatTreeShape::Balanced | RepeatTreeShape::LeftBalanced => {
            let left_width = width / 2;
            let split = min + left_width - 1;
            (split, width - left_width)
        }
        RepeatTreeShape::Left => (max - 1, 1),
        RepeatTreeShape::Right => (min, width - 1),
    }
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

        if max == 1 {
            self.rules.push(Rule { lhs: nt, rhs: Vec::new() });
            self.rules.push(Rule { lhs: nt, rhs: vec![symbol.clone()] });
        } else {
            // Unambiguous split: 0..half elements use repeat_max_half;
            // half+1..max elements require exactly `half` up front then 1..max-half more.
            // The two alternatives are disjoint → no GSS path explosion.
            let half = max / 2;
            let half_nt = self.repeat_max_nonterminal(symbol, half);
            let exact_half_nt = self.repeat_exact_nonterminal(symbol, half, RepeatTreeShape::LeftBalanced);
            let min1_tail_nt = self.repeat_min1_max_nonterminal(symbol, max - half);
            self.rules.push(Rule { lhs: nt, rhs: vec![Symbol::Nonterminal(half_nt)] });
            self.rules.push(Rule {
                lhs: nt,
                rhs: vec![Symbol::Nonterminal(exact_half_nt), Symbol::Nonterminal(min1_tail_nt)],
            });
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

        match shape {
            RepeatTreeShape::LeftBalanced | RepeatTreeShape::Balanced => {
                return self.repeat_range_nonterminal_balanced(symbol, min, max, shape);
            }
            _ => {}
        }

        let key = (symbol.clone(), min, max);
        if let Some(&nonterminal) = self.repeat_range_cache.get(&key) {
            return nonterminal;
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
        let exact_nt = self.repeat_exact_nonterminal(symbol, min, shape);
        let max_nt = self.repeat_max_nonterminal(symbol, delta);
        if min == 0 {
            max_nt
        } else {
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

    fn lower_expr(&mut self, expr: &GrammarExpr) -> Symbol {
        fn emit(lowerer: &mut Lowerer, lhs: NonterminalID, expr: &GrammarExpr) -> Result<(), GlrMaskError> {
            match expr {
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
                GrammarExpr::SeparatedSequence { items, separator } => {
                    let shape = comma_sep_shape();
                    let (sym, _) = lowerer.lower_separated_sequence_inner(items, separator, shape)?;
                    lowerer.rules.push(Rule { lhs, rhs: vec![sym] });
                }
                GrammarExpr::Epsilon => {
                    lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                }
                _ => {
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
            GrammarExpr::Ref(name) => {
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
            GrammarExpr::Epsilon => {
                // Epsilon as an inline NT atom: create a nonterminal with an empty production.
                let (_, nt) = self.fresh_nonterminal("eps");
                self.rules.push(Rule { lhs: nt, rhs: Vec::new() });
                Symbol::Nonterminal(nt)
            }
            GrammarExpr::Exclude { .. } => {
                return Err(GlrMaskError::GrammarParse(
                    "GrammarExpr::Exclude must be extracted into a terminal rule before lowering"
                        .into(),
                ));
            }
            GrammarExpr::Intersect { .. } => {
                return Err(GlrMaskError::GrammarParse(
                    "GrammarExpr::Intersect must be extracted into a terminal rule before lowering"
                        .into(),
                ));
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
        })
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
            let item_sym = self.lower_expr_terminalish(item_expr)?;
            // Return can_be_empty=true for optional items as a signal to the parent to add
            // a "without this item and its preceding separator" alternative.  We do NOT emit
            // an epsilon rule here — that would create dangling separators in the parent rule
            // (e.g. "key": , ).  The caller of lower_separated_sequence_inner handles the
            // all-optional empty case via an explicit separate alternative (e.g. "{}").
            return Ok((item_sym, !is_required));
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
        GrammarExpr::Literal(bytes) => Expr::U8Seq(bytes.clone()),
        GrammarExpr::CharClass { def, negate, utf8 } => {
            let pattern = char_class_pattern(def, *negate);
            parse_regex(&pattern, *utf8)
        }
        GrammarExpr::RawRegex(pattern) => parse_regex(pattern, true),
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
    })
}

/// Convert a lexer-level [`Expr`] into an equivalent [`GrammarExpr`].
///
/// Every `Expr` variant has a `GrammarExpr` counterpart, so this is lossless.
/// `Expr::U8Class(U8Set)` is converted to `GrammarExpr::CharClass` using a
/// range-encoded string representation.
pub fn expr_to_grammar_expr(expr: &Expr) -> GrammarExpr {
    match expr {
        Expr::U8Seq(bytes) => GrammarExpr::Literal(bytes.clone()),
        Expr::U8Class(set) => GrammarExpr::CharClass {
            def: u8set_to_class_def(set),
            negate: false,
            utf8: false,
        },
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
        Expr::Shared(inner) => expr_to_grammar_expr(inner),
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

/// Promote large alternations of literals in non-terminal rules to terminal rules.
///
/// When a `Choice` of ≥ `threshold` `Literal` options appears in a non-terminal
/// rule, create a new UPPERCASE terminal rule containing that choice (compiled as
/// a regex DFA) and replace the original `Choice` with a `Ref` to the new rule.
/// This avoids creating thousands of LR productions for large enums.
pub fn promote_large_literal_alts(grammar: &mut NamedGrammar, threshold: usize) {
    let mut new_rules: Vec<NamedRule> = Vec::new();
    let mut cache: HashMap<Vec<Vec<u8>>, String> = HashMap::new();
    let mut counter = 0usize;

    for rule in &mut grammar.rules {
        if rule.is_terminal {
            continue;
        }
        promote_expr_literals(
            &mut rule.expr,
            threshold,
            &mut new_rules,
            &mut cache,
            &mut counter,
        );
    }

    grammar.rules.extend(new_rules);
}

fn promote_expr_literals(
    expr: &mut GrammarExpr,
    threshold: usize,
    new_rules: &mut Vec<NamedRule>,
    cache: &mut HashMap<Vec<Vec<u8>>, String>,
    counter: &mut usize,
) {
    match expr {
        GrammarExpr::Choice(options) => {
            if options.len() >= threshold
                && options
                    .iter()
                    .all(|o| matches!(o, GrammarExpr::Literal(_)))
            {
                let mut literal_options: Vec<Vec<u8>> = options
                    .iter()
                    .filter_map(|option| match option {
                        GrammarExpr::Literal(bytes) => Some(bytes.clone()),
                        _ => None,
                    })
                    .collect();
                literal_options.sort();

                let rule_name = cache
                    .entry(literal_options)
                    .or_insert_with(|| {
                        let name = format!("ENUM_{}", *counter);
                        *counter += 1;
                        new_rules.push(NamedRule {
                            name: name.clone(),
                            expr: std::mem::replace(expr, GrammarExpr::Literal(Vec::new())),
                            is_terminal: true,
                            is_internal: false,
                        });
                        name
                    })
                    .clone();

                *expr = GrammarExpr::Ref(rule_name);
                return;
            }
            for option in options.iter_mut() {
                promote_expr_literals(option, threshold, new_rules, cache, counter);
            }
        }
        GrammarExpr::Exclude { expr, exclude } => {
            promote_expr_literals(expr, threshold, new_rules, cache, counter);
            promote_expr_literals(exclude, threshold, new_rules, cache, counter);
        }
        GrammarExpr::Sequence(parts) => {
            for part in parts.iter_mut() {
                promote_expr_literals(part, threshold, new_rules, cache, counter);
            }
        }
        GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner)
        | GrammarExpr::RepeatRange { expr: inner, .. } => {
            promote_expr_literals(inner, threshold, new_rules, cache, counter);
        }
        GrammarExpr::SeparatedSequence { items, separator } => {
            for (item, _) in items.iter_mut() {
                promote_expr_literals(item, threshold, new_rules, cache, counter);
            }
            promote_expr_literals(separator, threshold, new_rules, cache, counter);
        }
        _ => {}
    }
}

pub fn lower(grammar: &NamedGrammar) -> Result<GrammarDef, GlrMaskError> {
    let mut lowerer = Lowerer::new();

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
    let terminal_bodies: HashMap<String, GrammarExpr> = grammar
        .rules
        .iter()
        .filter(|r| r.is_terminal)
        .map(|r| (r.name.clone(), r.expr.clone()))
        .collect();
    let mut terminal_expr_cache: HashMap<String, Arc<Expr>> = HashMap::new();

    for rule in &grammar.rules {
        // Terminal rules: convert the entire body to a single Terminal::Expr.
        // Refs to other terminal rules are resolved via Expr::Shared.
        if rule.is_terminal {
            let mut visiting = HashSet::new();
            visiting.insert(rule.name.clone());
            let expr = grammar_expr_to_expr(
                &rule.expr,
                &terminal_bodies,
                &mut terminal_expr_cache,
                &mut visiting,
            )?;
            let arc = Arc::new(expr.clone());
            terminal_expr_cache.insert(rule.name.clone(), arc);

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
            _ => {
                let symbol = lowerer.lower_expr_terminalish(&rule.expr)?;
                lowerer.rules.push(Rule { lhs, rhs: vec![symbol] });
            }
        }
    }

    let start = lowerer.nonterminal_id(&grammar.start);
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
    use super::*;
    use std::collections::BTreeSet;

    fn nt(name: &str, expr: GrammarExpr) -> NamedRule {
        NamedRule { name: name.into(), expr, is_terminal: false, is_internal: false }
    }

    fn term(name: &str, expr: GrammarExpr) -> NamedRule {
        NamedRule { name: name.into(), expr, is_terminal: true, is_internal: false }
    }

    fn derivable_terminal_counts(
        grammar: &GrammarDef,
        target_tid: TerminalID,
        nonterminal: NonterminalID,
        memo: &mut BTreeMap<NonterminalID, BTreeSet<usize>>,
    ) -> BTreeSet<usize> {
        if let Some(cached) = memo.get(&nonterminal) {
            return cached.clone();
        }

        let mut result = BTreeSet::new();
        for rule in grammar.rules.iter().filter(|rule| rule.lhs == nonterminal) {
            let mut totals = BTreeSet::from([0usize]);
            for symbol in &rule.rhs {
                let counts = match symbol {
                    Symbol::Terminal(tid) => BTreeSet::from([usize::from(*tid == target_tid)]),
                    Symbol::Nonterminal(next) => derivable_terminal_counts(grammar, target_tid, *next, memo),
                };
                let mut next_totals = BTreeSet::new();
                for left in &totals {
                    for right in &counts {
                        next_totals.insert(left + right);
                    }
                }
                totals = next_totals;
            }
            result.extend(totals);
        }

        memo.insert(nonterminal, result.clone());
        result
    }

    #[test]
    fn test_lower_simple_sequence() {
        let g = NamedGrammar {
            rules: vec![nt(
                "start",
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b"a".to_vec()),
                    GrammarExpr::Literal(b"b".to_vec()),
                ]),
            )],
            start: "start".into(),
            ignore: None,
        };
        let gdef = lower(&g).unwrap();
        assert_eq!(gdef.start, 0);
        assert!(!gdef.rules.is_empty());
        assert_eq!(gdef.num_terminals(), 2);
    }

    #[test]
    fn test_lower_choice() {
        let g = NamedGrammar {
            rules: vec![nt(
                "start",
                GrammarExpr::Choice(vec![
                    GrammarExpr::Literal(b"a".to_vec()),
                    GrammarExpr::Literal(b"b".to_vec()),
                ]),
            )],
            start: "start".into(),
            ignore: None,
        };
        let gdef = lower(&g).unwrap();
        
        let start_rules: Vec<_> = gdef.rules.iter().filter(|r| r.lhs == 0).collect();
        assert_eq!(start_rules.len(), 2);
    }

    #[test]
    fn test_lower_optional() {
        let g = NamedGrammar {
            rules: vec![nt(
                "start",
                GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"a".to_vec()))),
            )],
            start: "start".into(),
            ignore: None,
        };
        let gdef = lower(&g).unwrap();
        
        assert!(gdef.rules.len() >= 2);
    }

    /// Regression case covering nullability handling in `from_exprs`.
    #[test]
    fn test_lower_nullability_uses_epsilon_rules_not_empty_terminals() {
        let g = NamedGrammar {
            rules: vec![nt(
                "start",
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"x".to_vec()))),
                    GrammarExpr::Sequence(vec![]),
                    GrammarExpr::Literal(b"z".to_vec()),
                ]),
            )],
            start: "start".into(),
            ignore: None,
        };
        let gdef = lower(&g).unwrap();

        assert_eq!(gdef.terminals.len(), 2, "only the concrete x/z literals should become terminals");
        assert!(gdef.terminals.iter().any(|terminal| matches!(terminal, Terminal::Literal { bytes, .. } if bytes == b"x")));
        assert!(gdef.terminals.iter().any(|terminal| matches!(terminal, Terminal::Literal { bytes, .. } if bytes == b"z")));
        assert!(
            !gdef
                .terminals
                .iter()
                .any(|terminal| matches!(terminal, Terminal::Literal { bytes, .. } if bytes.is_empty())),
            "nullable pieces should lower through epsilon productions, not through empty terminals"
        );

        assert!(
            gdef.rules.iter().any(|rule| rule.lhs != gdef.start && rule.rhs.is_empty()),
            "lowering nullable pieces should introduce helper epsilon productions"
        );
        assert!(
            gdef.rules.iter().any(|rule| {
                rule.lhs == gdef.start
                    && rule.rhs.len() == 3
                    && matches!(rule.rhs[0], Symbol::Nonterminal(_))
                    && matches!(rule.rhs[1], Symbol::Nonterminal(_))
                    && matches!(rule.rhs[2], Symbol::Terminal(_))
            }),
            "the start rule should sequence the optional helper, the explicit epsilon helper, and the trailing literal"
        );
    }

    #[test]
    fn test_lower_repeat() {
        let g = NamedGrammar {
            rules: vec![nt(
                "start",
                GrammarExpr::RepeatOne(Box::new(GrammarExpr::Literal(b"a".to_vec()))),
            )],
            start: "start".into(),
            ignore: None,
        };
        let gdef = lower(&g).unwrap();
        
        assert!(gdef.rules.len() >= 2);
    }

    #[test]
    fn test_lower_repeat_range_derives_disjoint_counts() {
        let g = NamedGrammar {
            rules: vec![nt(
                "start",
                GrammarExpr::RepeatRange {
                    expr: Box::new(GrammarExpr::Literal(b"a".to_vec())),
                    min: 3,
                    max: 5,
                },
            )],
            start: "start".into(),
            ignore: None,
        };
        let gdef = lower(&g).unwrap();
        let a_tid = gdef
            .terminals
            .iter()
            .find_map(|terminal| match terminal {
                Terminal::Literal { id, bytes } if bytes == b"a" => Some(*id),
                _ => None,
            })
            .expect("lowered grammar should contain the literal terminal");

        let mut memo = BTreeMap::new();
        let counts = derivable_terminal_counts(&gdef, a_tid, gdef.start, &mut memo);
        assert_eq!(counts, BTreeSet::from([3usize, 4, 5]));
        assert!(
            gdef.rules.iter().all(|rule| rule.rhs.len() <= 2),
            "repeat-range lowering should stay binary and avoid long optional ladders"
        );
    }

    #[test]
    fn test_exact_repeat_split_respects_left_tree_shape() {
        assert_eq!(repeat_tree_shape_from_value("left"), RepeatTreeShape::Left);
        assert_eq!(exact_repeat_split(13, RepeatTreeShape::Left), (12, 1));
        assert_eq!(range_repeat_split(3, 13, RepeatTreeShape::Left), (12, 1));
    }

    #[test]
    fn test_exact_repeat_split_respects_right_tree_shape() {
        assert_eq!(repeat_tree_shape_from_value("right"), RepeatTreeShape::Right);
        assert_eq!(exact_repeat_split(13, RepeatTreeShape::Right), (1, 12));
        assert_eq!(range_repeat_split(3, 13, RepeatTreeShape::Right), (3, 10));
    }

    #[test]
    fn test_lower_multi_rule() {
        let g = NamedGrammar {
            rules: vec![
                nt(
                    "start",
                    GrammarExpr::Sequence(vec![
                        GrammarExpr::Ref("item".into()),
                        GrammarExpr::Literal(b".".to_vec()),
                    ]),
                ),
                nt(
                    "item",
                    GrammarExpr::Choice(vec![
                        GrammarExpr::Literal(b"a".to_vec()),
                        GrammarExpr::Literal(b"b".to_vec()),
                    ]),
                ),
            ],
            start: "start".into(),
            ignore: None,
        };
        let gdef = lower(&g).unwrap();
        assert_eq!(gdef.start, 0); 
        assert!(gdef.num_nonterminals() >= 2);
    }

    #[test]
    fn test_lower_retains_useful_names_but_not_helper_nonterminals() {
        let g = NamedGrammar {
            rules: vec![
                nt(
                    "start",
                    GrammarExpr::Sequence(vec![
                        GrammarExpr::Ref("named_nt".into()),
                        GrammarExpr::Literal(b"term1".to_vec()),
                    ]),
                ),
                nt(
                    "named_nt",
                    GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"term2".to_vec()))),
                ),
            ],
            start: "start".into(),
            ignore: None,
        };

        let gdef = lower(&g).unwrap();

        let nonterminal_names: Vec<&str> = gdef
            .nonterminal_names
            .values()
            .map(|name| name.as_str())
            .collect();
        assert!(nonterminal_names.contains(&"start"));
        assert!(nonterminal_names.contains(&"named_nt"));
        assert!(!nonterminal_names.iter().any(|name| name.starts_with("__")));

        let terminal_names: Vec<&str> = gdef
            .terminal_names
            .values()
            .map(|name| name.as_str())
            .collect();
        assert!(terminal_names.contains(&"term1"));
        assert!(terminal_names.contains(&"term2"));
    }

    #[test]
    fn test_lower_terminal_exclude_rule_to_expr_exclude() {
        let g = NamedGrammar {
            rules: vec![
                nt("start", GrammarExpr::Ref("ANY_BUT_A".into())),
                term(
                    "ANY_BUT_A",
                    GrammarExpr::Exclude {
                        expr: Box::new(GrammarExpr::AnyByte),
                        exclude: Box::new(GrammarExpr::Literal(b"a".to_vec())),
                    },
                ),
            ],
            start: "start".into(),
            ignore: None,
        };

        let gdef = lower(&g).unwrap();
        let terminal = gdef
            .terminals
            .iter()
            .find(|terminal| gdef.terminal_display_name(terminal.id()) == "ANY_BUT_A")
            .expect("lowered terminal should exist");

        assert!(matches!(
            terminal,
            Terminal::Expr {
                expr: Expr::Exclude { .. },
                ..
            }
        ));
    }

    // ── SeparatedSequence tests ────────────────────────────────────────────

    fn make_sep_seq_grammar(items: Vec<(GrammarExpr, bool)>, sep: GrammarExpr) -> NamedGrammar {
        NamedGrammar {
            rules: vec![nt(
                "start",
                GrammarExpr::SeparatedSequence { items, separator: Box::new(sep) },
            )],
            start: "start".into(),
            ignore: None,
        }
    }

    fn sep_terminal_id(gdef: &GrammarDef) -> TerminalID {
        gdef.terminals
            .iter()
            .find_map(|t| match t {
                Terminal::Literal { id, bytes } if bytes == b"," => Some(*id),
                _ => None,
            })
            .expect("separator terminal ',' should exist in lowered grammar")
    }

    fn item_terminal_id(gdef: &GrammarDef, byte: u8) -> TerminalID {
        gdef.terminals
            .iter()
            .find_map(|t| match t {
                Terminal::Literal { id, bytes } if bytes == &[byte] => Some(*id),
                _ => None,
            })
            .expect("item terminal should exist in lowered grammar")
    }

    /// `SeparatedSequence` with all required items accepts exactly `n-1` separators.
    #[test]
    fn test_separated_sequence_all_required_has_fixed_sep_count() {
        // items: a(req), b(req), c(req) → must produce exactly "a,b,c" (2 separators)
        let g = make_sep_seq_grammar(
            vec![
                (GrammarExpr::Literal(b"a".to_vec()), true),
                (GrammarExpr::Literal(b"b".to_vec()), true),
                (GrammarExpr::Literal(b"c".to_vec()), true),
            ],
            GrammarExpr::Literal(b",".to_vec()),
        );
        let gdef = lower(&g).unwrap();
        let sep_tid = sep_terminal_id(&gdef);
        let mut memo = BTreeMap::new();
        let sep_counts = derivable_terminal_counts(&gdef, sep_tid, gdef.start, &mut memo);
        assert_eq!(sep_counts, BTreeSet::from([2usize]), "3 required items must produce exactly 2 separators");
    }

    /// Optional items let the parent skip the separator — no dangling commas.
    #[test]
    fn test_separated_sequence_optional_item_no_dangling_sep() {
        // items: a(req), b(opt), c(req) → can produce "a,b,c" (2 seps) or "a,c" (1 sep).
        // Must NOT produce any path with a dangling separator.
        let g = make_sep_seq_grammar(
            vec![
                (GrammarExpr::Literal(b"a".to_vec()), true),
                (GrammarExpr::Literal(b"b".to_vec()), false),
                (GrammarExpr::Literal(b"c".to_vec()), true),
            ],
            GrammarExpr::Literal(b",".to_vec()),
        );
        let gdef = lower(&g).unwrap();
        let sep_tid = sep_terminal_id(&gdef);
        let mut memo = BTreeMap::new();
        let sep_counts = derivable_terminal_counts(&gdef, sep_tid, gdef.start, &mut memo);
        assert_eq!(
            sep_counts,
            BTreeSet::from([1usize, 2]),
            "optional middle item allows 1 or 2 separators, never 0 or 3"
        );
        // Verify that no grammar rule has an empty rhs for the sep_seq NTs,
        // which would allow a dangling separator.
        let sep_rules_with_epsilon: Vec<_> = gdef
            .rules
            .iter()
            .filter(|r| r.lhs != gdef.start && r.rhs.is_empty())
            .collect();
        assert!(
            sep_rules_with_epsilon.is_empty(),
            "no epsilon rules should be introduced by SeparatedSequence (dangling separator guard)"
        );
    }

    /// Two optional items: accepts a,b or a alone or b alone — but never epsilon.
    #[test]
    fn test_separated_sequence_all_optional_no_epsilon_rule() {
        // items: a(opt), b(opt) → can_be_empty=true but no epsilon rule emitted.
        let g = make_sep_seq_grammar(
            vec![
                (GrammarExpr::Literal(b"a".to_vec()), false),
                (GrammarExpr::Literal(b"b".to_vec()), false),
            ],
            GrammarExpr::Literal(b",".to_vec()),
        );
        let gdef = lower(&g).unwrap();

        // The grammar should accept "a,b" (1 sep), "a" (0 seps), "b" (0 seps).
        let sep_tid = sep_terminal_id(&gdef);
        let a_tid = item_terminal_id(&gdef, b'a');
        let b_tid = item_terminal_id(&gdef, b'b');

        let mut memo = BTreeMap::new();
        let sep_counts = derivable_terminal_counts(&gdef, sep_tid, gdef.start, &mut memo);
        assert_eq!(sep_counts, BTreeSet::from([0usize, 1]), "two optional items → 0 or 1 separators");

        memo.clear();
        let a_counts = derivable_terminal_counts(&gdef, a_tid, gdef.start, &mut memo);
        assert_eq!(a_counts, BTreeSet::from([0usize, 1]), "a is optional → 0 or 1 occurrences");

        memo.clear();
        let b_counts = derivable_terminal_counts(&gdef, b_tid, gdef.start, &mut memo);
        assert_eq!(b_counts, BTreeSet::from([0usize, 1]), "b is optional → 0 or 1 occurrences");

        // No epsilon rules anywhere — caller handles the "nothing present" case.
        let any_epsilon = gdef.rules.iter().any(|r| r.rhs.is_empty());
        assert!(!any_epsilon, "SeparatedSequence must not emit epsilon rules");
    }

    /// Single required item: lowers to just the item symbol, no wrapper rule needed.
    #[test]
    fn test_separated_sequence_single_required() {
        let g = make_sep_seq_grammar(
            vec![(GrammarExpr::Literal(b"x".to_vec()), true)],
            GrammarExpr::Literal(b",".to_vec()),
        );
        let gdef = lower(&g).unwrap();
        // Only the 'x' terminal; the separator should not appear.
        assert_eq!(gdef.terminals.len(), 1, "single required item: only one terminal");
        assert!(matches!(&gdef.terminals[0], Terminal::Literal { bytes, .. } if bytes == b"x"));
    }

    /// Single optional item: lowers to just the item symbol (no epsilon or wrapper NT emitted).
    #[test]
    fn test_separated_sequence_single_optional() {
        let g = make_sep_seq_grammar(
            vec![(GrammarExpr::Literal(b"x".to_vec()), false)],
            GrammarExpr::Literal(b",".to_vec()),
        );
        let gdef = lower(&g).unwrap();
        // Only the 'x' terminal; no separator.
        assert_eq!(gdef.terminals.len(), 1, "single optional item: only one terminal");
        // No epsilon rules.
        let any_epsilon = gdef.rules.iter().any(|r| r.rhs.is_empty());
        assert!(!any_epsilon, "single optional item must not produce an epsilon rule");
    }
}
