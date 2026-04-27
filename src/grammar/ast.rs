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
        allow_empty: bool,
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
                GrammarExpr::SeparatedSequence { items, separator, .. } => {
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
        None => RepeatTreeShape::LeftBalanced,
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
            GrammarExpr::Ref(name) => Ok(Some(self.lower_nonnullable_named_rule(name)?)),
            GrammarExpr::Optional(inner) => self.lower_nonnullable_expr_symbol(inner),
            GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::AnyByte
            | GrammarExpr::Exclude { .. }
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
            GrammarExpr::Ref(name) => {
                let symbol = self.lower_nonnullable_named_rule(name)?;
                self.rules.push(Rule { lhs, rhs: vec![symbol] });
            }
            GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::AnyByte
            | GrammarExpr::Exclude { .. }
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

        let first_optional = items.iter().position(|(_, required)| !required);
        if let Some(split_idx) = first_optional {
            let required_prefix = &items[..split_idx];
            let optional_suffix = &items[split_idx..];
            let required_prefix_is_nonnullable = !required_prefix.is_empty()
                && required_prefix
                    .iter()
                    .all(|(expr, required)| *required && !self.expr_is_nullable(expr));
            let suffix_is_all_optional = optional_suffix.iter().all(|(_, required)| !*required);

            if required_prefix_is_nonnullable && suffix_is_all_optional {
                let sep_sym = self.lower_expr_terminalish(separator)?;
                let (prefix_sym, _) =
                    self.lower_separated_sequence_inner(required_prefix, separator, shape)?;
                let (suffix_sym, suffix_can_be_empty) =
                    self.lower_separated_sequence_inner(optional_suffix, separator, shape)?;

                debug_assert!(suffix_can_be_empty);

                let (_, nt) = self.fresh_nonterminal("sep_seq");
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![prefix_sym.clone()],
                });
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![prefix_sym, sep_sym, suffix_sym],
                });
                return Ok((Symbol::Nonterminal(nt), false));
            }
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

fn grammar_expr_is_nullable(
    expr: &GrammarExpr,
    rule_nullable: &HashMap<String, bool>,
) -> bool {
    match expr {
        GrammarExpr::Ref(name) => rule_nullable.get(name).copied().unwrap_or(false),
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
        GrammarExpr::AnyByte => false,
        GrammarExpr::SeparatedSequence { items, allow_empty, .. } => {
            *allow_empty
                && items
                    .iter()
                    .all(|(item, is_required)| !*is_required || grammar_expr_is_nullable(item, rule_nullable))
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
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
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
    fn test_repeat_tree_shape_parses_countdown() {
        assert_eq!(
            repeat_tree_shape_from_value("countdown"),
            RepeatTreeShape::Countdown,
        );
        assert_eq!(
            repeat_tree_shape_from_value("deterministic"),
            RepeatTreeShape::Countdown,
        );
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
                GrammarExpr::SeparatedSequence {
                    items,
                    separator: Box::new(sep),
                    allow_empty: true,
                },
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

    /// Two optional items: accepts a,b or a alone or b alone — and now epsilon.
    #[test]
    fn test_separated_sequence_all_optional_no_epsilon_rule() {
        // items: a(opt), b(opt) → can_be_empty=true and top-level epsilon emitted.
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

        // Nullable SeparatedSequence now emits an epsilon helper production.
        let has_epsilon = gdef.rules.iter().any(|r| r.rhs.is_empty());
        assert!(has_epsilon, "nullable SeparatedSequence should emit an epsilon rule");
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

    /// Single optional item: lowers to item plus an epsilon alternative.
    #[test]
    fn test_separated_sequence_single_optional() {
        let g = make_sep_seq_grammar(
            vec![(GrammarExpr::Literal(b"x".to_vec()), false)],
            GrammarExpr::Literal(b",".to_vec()),
        );
        let gdef = lower(&g).unwrap();
        // Only the 'x' terminal; no separator.
        assert_eq!(gdef.terminals.len(), 1, "single optional item: only one terminal");
        // Nullable SeparatedSequence emits an epsilon helper production.
        let has_epsilon = gdef.rules.iter().any(|r| r.rhs.is_empty());
        assert!(has_epsilon, "single optional item should produce an epsilon rule");
    }

    #[test]
    fn test_separated_sequence_nullable_required_item_skips_separator_when_empty() {
        let g = make_sep_seq_grammar(
            vec![
                (GrammarExpr::Literal(b"a".to_vec()), true),
                (
                    GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"b".to_vec()))),
                    true,
                ),
                (GrammarExpr::Literal(b"c".to_vec()), true),
            ],
            GrammarExpr::Literal(b",".to_vec()),
        );
        let gdef = lower(&g).unwrap();

        let sep_tid = sep_terminal_id(&gdef);
        let b_tid = item_terminal_id(&gdef, b'b');

        let mut memo = BTreeMap::new();
        let sep_counts = derivable_terminal_counts(&gdef, sep_tid, gdef.start, &mut memo);
        assert_eq!(
            sep_counts,
            BTreeSet::from([1usize, 2]),
            "nullable required middle item should not force a doubled separator when empty"
        );

        memo.clear();
        let b_counts = derivable_terminal_counts(&gdef, b_tid, gdef.start, &mut memo);
        assert_eq!(b_counts, BTreeSet::from([0usize, 1]), "nullable middle item should still allow omission");
    }

    #[test]
    fn test_separated_sequence_nullable_terminal_ref_skips_separator_when_empty() {
        let g = NamedGrammar {
            rules: vec![
                nt(
                    "start",
                    GrammarExpr::SeparatedSequence {
                        items: vec![
                            (GrammarExpr::Literal(b"a".to_vec()), true),
                            (GrammarExpr::Ref("MAYBE_B".into()), true),
                            (GrammarExpr::Literal(b"c".to_vec()), true),
                        ],
                        separator: Box::new(GrammarExpr::Literal(b",".to_vec())),
                        allow_empty: true,
                    },
                ),
                term("MAYBE_B", GrammarExpr::RawRegex("b?".into())),
            ],
            start: "start".into(),
            ignore: None,
        };
        let gdef = lower(&g).unwrap();

        let sep_tid = sep_terminal_id(&gdef);

        let mut memo = BTreeMap::new();
        let sep_counts = derivable_terminal_counts(&gdef, sep_tid, gdef.start, &mut memo);
        assert_eq!(
            sep_counts,
            BTreeSet::from([1usize, 2]),
            "nullable terminal refs inside SeparatedSequence should not emit doubled separators"
        );
    }

    #[test]
    fn test_separated_sequence_nonnullable_terminal_ref_reuses_named_wrapper() {
        let g = NamedGrammar {
            rules: vec![
                nt(
                    "start",
                    GrammarExpr::SeparatedSequence {
                        items: vec![(GrammarExpr::Ref("KEY".into()), true)],
                        separator: Box::new(GrammarExpr::Literal(b",".to_vec())),
                        allow_empty: true,
                    },
                ),
                term("KEY", GrammarExpr::RawRegex("x".into())),
            ],
            start: "start".into(),
            ignore: None,
        };
        let gdef = lower(&g).unwrap();

        let key_nt = gdef
            .nonterminal_names
            .iter()
            .find_map(|(id, name)| (name == "KEY").then_some(*id))
            .expect("named terminal KEY should have a wrapper nonterminal");
        let key_tid = gdef
            .terminals
            .iter()
            .find_map(|terminal| (gdef.terminal_display_name(terminal.id()) == "KEY").then_some(terminal.id()))
            .expect("named terminal KEY should exist");

        let key_terminal_wrappers: Vec<_> = gdef
            .rules
            .iter()
            .filter(|rule| rule.rhs == vec![Symbol::Terminal(key_tid)])
            .collect();
        assert_eq!(
            key_terminal_wrappers.len(),
            1,
            "nonnullable terminal refs in SeparatedSequence should not synthesize an extra alias"
        );
        assert_eq!(
            key_terminal_wrappers[0].lhs,
            key_nt,
            "the only terminal wrapper should be the named rule's own wrapper"
        );
    }

    #[test]
    fn test_separated_sequence_repeat_range_inserts_internal_separators() {
        let g = make_sep_seq_grammar(
            vec![
                (GrammarExpr::Literal(b"a".to_vec()), true),
                (
                    GrammarExpr::RepeatRange {
                        expr: Box::new(GrammarExpr::Literal(b"b".to_vec())),
                        min: 0,
                        max: 3,
                    },
                    true,
                ),
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
            BTreeSet::from([1usize, 2, 3, 4]),
            "b{{0,3}} inside sepseq should become b (sep b){{0,2}} when present"
        );
    }

    #[test]
    fn test_separated_sequence_grouped_repeat_range_does_not_insert_internal_separators() {
        let g = make_sep_seq_grammar(
            vec![
                (GrammarExpr::Literal(b"a".to_vec()), true),
                (
                    GrammarExpr::Sequence(vec![GrammarExpr::RepeatRange {
                        expr: Box::new(GrammarExpr::Literal(b"b".to_vec())),
                        min: 2,
                        max: 3,
                    }]),
                    true,
                ),
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
            BTreeSet::from([2usize]),
            "grouped repeat item should stay a single sepseq item"
        );
    }
}
