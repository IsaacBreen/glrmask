#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

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
    TerminalExpr(Expr),
    Exclude {
        expr: Box<GrammarExpr>,
        exclude: Box<GrammarExpr>,
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct NamedRule {
    pub name: String,
    pub expr: GrammarExpr,
    pub is_terminal: bool,
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
}

struct Lowerer {
    rules: Vec<Rule>,
    terminal_map: BTreeMap<String, TerminalID>,
    terminals: Vec<Terminal>,
    nt_map: BTreeMap<String, NonterminalID>,
    anon_counter: u32,
    terminal_names: BTreeMap<TerminalID, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RepeatTreeShape {
    Balanced,
    Left,
    Right,
}

fn repeat_tree_shape() -> RepeatTreeShape {
    repeat_tree_shape_from_value(&std::env::var("GLRMASK_TREE_SHAPE").unwrap_or_default())
}

fn repeat_tree_shape_from_value(value: &str) -> RepeatTreeShape {
    match value {
        "left" => RepeatTreeShape::Left,
        "balanced" => RepeatTreeShape::Balanced,
        _ => RepeatTreeShape::Right,
    }
}

fn highest_power_of_two_less_than(value: usize) -> usize {
    debug_assert!(value > 1);
    1usize << ((usize::BITS - 1) - (value - 1).leading_zeros())
}

const RIGHT_REPEAT_RANGE_FRONT_BUCKET: usize = 128;
const LEFT_REPEAT_RANGE_BACK_BUCKET: usize = 128;

fn exact_repeat_split(count: usize, shape: RepeatTreeShape) -> (usize, usize) {
    debug_assert!(count > 1);
    match shape {
        RepeatTreeShape::Balanced => {
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
        RepeatTreeShape::Balanced => {
            let left_width = width / 2;
            let split = min + left_width - 1;
            (split, width - left_width)
        }
        RepeatTreeShape::Left => (max - 1, 1),
        RepeatTreeShape::Right => (min, width - 1),
    }
}

impl Lowerer {
    fn new() -> Self {
        Self {
            rules: Vec::new(),
            terminal_map: BTreeMap::new(),
            terminals: Vec::new(),
            nt_map: BTreeMap::new(),
            anon_counter: 0,
            terminal_names: BTreeMap::new(),
        }
    }

    fn nt_id(&mut self, name: &str) -> NonterminalID {
        if let Some(&id) = self.nt_map.get(name) {
            id
        } else {
            let id = self.nt_map.len() as NonterminalID;
            self.nt_map.insert(name.to_string(), id);
            id
        }
    }

    fn fresh_nt(&mut self, hint: &str) -> (String, NonterminalID) {
        let name = format!("__{}_{}", hint, self.anon_counter);
        self.anon_counter += 1;
        let id = self.nt_id(&name);
        (name, id)
    }

    fn terminal_id(&mut self, name: &str, pattern: &str, utf8: bool) -> TerminalID {
        let key = format!("{}:{}", pattern, utf8);
        if let Some(&id) = self.terminal_map.get(&key) {
            return id;
        }
        let id = self.terminals.len() as TerminalID;
        self.terminal_map.insert(key, id);
        self.terminal_names.insert(id, name.to_string());
        // Decide variant: if the pattern is the same as the escaped literal of
        // the name bytes, store as Literal; otherwise store as Pattern.
        let name_bytes = name.as_bytes();
        let escaped: String = name_bytes.iter().map(|&b| regex_escape_byte(b)).collect();
        if escaped == pattern && !utf8 {
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

    fn repeat_exact_nt(
        &mut self,
        item: &Symbol,
        count: usize,
        shape: RepeatTreeShape,
        cache: &mut BTreeMap<usize, NonterminalID>,
    ) -> NonterminalID {
        if let Some(&nt) = cache.get(&count) {
            return nt;
        }

        let (_, nt) = self.fresh_nt("repeat_exact");
        cache.insert(count, nt);
        match count {
            0 => self.rules.push(Rule { lhs: nt, rhs: Vec::new() }),
            1 => self.rules.push(Rule {
                lhs: nt,
                rhs: vec![item.clone()],
            }),
            _ => {
                let (left, right) = exact_repeat_split(count, shape);
                let left_nt = self.repeat_exact_nt(item, left, shape, cache);
                let right_nt = self.repeat_exact_nt(item, right, shape, cache);
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![Symbol::Nonterminal(left_nt), Symbol::Nonterminal(right_nt)],
                });
            }
        }
        nt
    }

    fn repeat_range_nt(
        &mut self,
        item: &Symbol,
        min: usize,
        max: usize,
        shape: RepeatTreeShape,
        exact_cache: &mut BTreeMap<usize, NonterminalID>,
        range_cache: &mut BTreeMap<(usize, usize), NonterminalID>,
    ) -> NonterminalID {
        debug_assert!(min <= max);
        if min == max {
            return self.repeat_exact_nt(item, min, shape, exact_cache);
        }
        if let Some(&nt) = range_cache.get(&(min, max)) {
            return nt;
        }

        let (_, nt) = self.fresh_nt("repeat_range");
        range_cache.insert((min, max), nt);
        match shape {
            RepeatTreeShape::Right if (max - min + 1) > RIGHT_REPEAT_RANGE_FRONT_BUCKET => {
                let cutoff = (min + RIGHT_REPEAT_RANGE_FRONT_BUCKET - 1).min(max);
                for count in min..=cutoff {
                    let exact_nt = self.repeat_exact_nt(item, count, shape, exact_cache);
                    self.rules.push(Rule {
                        lhs: nt,
                        rhs: vec![Symbol::Nonterminal(exact_nt)],
                    });
                }
                if cutoff < max {
                    let tail_nt = self.repeat_range_nt(
                        item,
                        cutoff + 1,
                        max,
                        shape,
                        exact_cache,
                        range_cache,
                    );
                    self.rules.push(Rule {
                        lhs: nt,
                        rhs: vec![Symbol::Nonterminal(tail_nt)],
                    });
                }
                return nt;
            }
            RepeatTreeShape::Left if (max - min + 1) > LEFT_REPEAT_RANGE_BACK_BUCKET => {
                let cutoff = max.saturating_sub(LEFT_REPEAT_RANGE_BACK_BUCKET - 1).max(min);
                if min < cutoff {
                    let head_nt = self.repeat_range_nt(
                        item,
                        min,
                        cutoff - 1,
                        shape,
                        exact_cache,
                        range_cache,
                    );
                    self.rules.push(Rule {
                        lhs: nt,
                        rhs: vec![Symbol::Nonterminal(head_nt)],
                    });
                }
                for count in cutoff..=max {
                    let exact_nt = self.repeat_exact_nt(item, count, shape, exact_cache);
                    self.rules.push(Rule {
                        lhs: nt,
                        rhs: vec![Symbol::Nonterminal(exact_nt)],
                    });
                }
                return nt;
            }
            _ => {}
        }
        let (split, _) = range_repeat_split(min, max, shape);
        let left_nt =
            self.repeat_range_nt(item, min, split, shape, exact_cache, range_cache);
        let right_nt =
            self.repeat_range_nt(item, split + 1, max, shape, exact_cache, range_cache);
        self.rules.push(Rule {
            lhs: nt,
            rhs: vec![Symbol::Nonterminal(left_nt)],
        });
        self.rules.push(Rule {
            lhs: nt,
            rhs: vec![Symbol::Nonterminal(right_nt)],
        });
        nt
    }

    fn emit_repeat_range(
        &mut self,
        lhs: NonterminalID,
        inner: &GrammarExpr,
        min: usize,
        max: usize,
    ) -> Result<(), GlrMaskError> {
        debug_assert!(min <= max);
        let item = self.lower_expr_terminalish(inner)?;
        let shape = repeat_tree_shape();
        let mut exact_cache = BTreeMap::new();
        let mut range_cache = BTreeMap::new();
        let range_nt =
            self.repeat_range_nt(&item, min, max, shape, &mut exact_cache, &mut range_cache);
        self.rules.push(Rule {
            lhs,
            rhs: vec![Symbol::Nonterminal(range_nt)],
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
                    let item = lowerer.lower_expr(inner);
                    lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                    lowerer.rules.push(Rule { lhs, rhs: vec![Symbol::Nonterminal(lhs), item] });
                }
                GrammarExpr::RepeatOne(inner) => {
                    let item = lowerer.lower_expr(inner);
                    lowerer.rules.push(Rule { lhs, rhs: vec![item.clone()] });
                    lowerer.rules.push(Rule { lhs, rhs: vec![Symbol::Nonterminal(lhs), item] });
                }
                GrammarExpr::RepeatRange { expr, min, max } => {
                    lowerer.emit_repeat_range(lhs, expr, *min, *max)?;
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

        let (_, nt) = self.fresh_nt("expr");
        emit(self, nt, expr).expect("grammar lowering should not fail for internal expression emission");
        Symbol::Nonterminal(nt)
    }

    fn lower_expr_terminalish(&mut self, expr: &GrammarExpr) -> Result<Symbol, GlrMaskError> {
        Ok(match expr {
            GrammarExpr::Ref(name) => Symbol::Nonterminal(self.nt_id(name)),
            GrammarExpr::Literal(bytes) => {
                let pattern = bytes.iter().map(|&b| regex_escape_byte(b)).collect::<String>();
                Symbol::Terminal(self.terminal_id(&String::from_utf8_lossy(bytes), &pattern, false))
            }
            GrammarExpr::CharClass { def, negate, utf8 } => {
                let pattern = if *negate {
                    format!("[^{def}]")
                } else {
                    format!("[{def}]")
                };
                Symbol::Terminal(self.terminal_id(&pattern, &pattern, *utf8))
            }
            GrammarExpr::RawRegex(pattern) => {
                // assume utf8 true for raw regex from lark/ebnf
                Symbol::Terminal(self.terminal_id(pattern, pattern, true))
            }
            GrammarExpr::TerminalExpr(expr) => {
                Symbol::Terminal(self.register_terminal_expr("<expr>", expr.clone()))
            }
            GrammarExpr::Exclude { .. } => {
                return Err(GlrMaskError::GrammarParse(
                    "GrammarExpr::Exclude must be extracted into a terminal rule before lowering"
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
            | GrammarExpr::RepeatRange { .. } => self.lower_expr(expr),
        })
    }

    /// Register a pre-resolved terminal Expr, deduplicating by value.
    fn register_terminal_expr(&mut self, name: &str, expr: Expr) -> TerminalID {
        // Dedup by the Expr tree itself
        for (i, t) in self.terminals.iter().enumerate() {
            if let Terminal::Expr { expr: existing, .. } = t {
                if *existing == expr {
                    return i as TerminalID;
                }
            }
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
    ge: &GrammarExpr,
    terminal_bodies: &HashMap<String, GrammarExpr>,
    terminal_expr_cache: &mut HashMap<String, Arc<Expr>>,
    visiting: &mut HashSet<String>,
) -> Result<Expr, GlrMaskError> {
    Ok(match ge {
        GrammarExpr::Literal(bytes) => Expr::U8Seq(bytes.clone()),
        GrammarExpr::CharClass { def, negate, utf8 } => {
            let pattern = if *negate {
                format!("[^{def}]")
            } else {
                format!("[{def}]")
            };
            parse_regex(&pattern, *utf8)
        }
        GrammarExpr::RawRegex(pattern) => parse_regex(pattern, true),
        GrammarExpr::AnyByte => Expr::U8Class(U8Set::from_range(0, 255)),
        GrammarExpr::TerminalExpr(expr) => expr.clone(),
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
    })
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
                let mut sorted_bytes: Vec<Vec<u8>> = options
                    .iter()
                    .map(|o| match o {
                        GrammarExpr::Literal(b) => b.clone(),
                        _ => unreachable!(),
                    })
                    .collect();
                sorted_bytes.sort();

                let rule_name = cache
                    .entry(sorted_bytes)
                    .or_insert_with(|| {
                        let name = format!("ENUM_{}", *counter);
                        *counter += 1;
                        new_rules.push(NamedRule {
                            name: name.clone(),
                            expr: std::mem::replace(expr, GrammarExpr::Literal(Vec::new())),
                            is_terminal: true,
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
        _ => {}
    }
}

pub fn lower(grammar: &NamedGrammar) -> Result<GrammarDef, GlrMaskError> {
    let mut lowerer = Lowerer::new();

    for rule in &grammar.rules {
        lowerer.nt_id(&rule.name);
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
        let lhs = lowerer.nt_id(&rule.name);

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
            let tid = lowerer.register_terminal_expr(&rule.name, expr);
            lowerer.rules.push(Rule { lhs, rhs: vec![Symbol::Terminal(tid)] });
            continue;
        }

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

    let start = lowerer.nt_id(&grammar.start);
    let nonterminal_names = lowerer
        .nt_map
        .iter()
        .filter(|(name, _)| !name.starts_with("__"))
        .map(|(name, id)| (*id, name.clone()))
        .collect();

    let ignore_terminal = if let Some(ref ignore_name) = grammar.ignore {
        // Find the terminal created for the ignore rule.
        // The ignore rule has is_terminal=true, so it was lowered above
        // as NT → Terminal. The terminal has the ignore name in terminal_names.
        let tid = lowerer.terminal_names.iter()
            .find(|(_, name)| *name == ignore_name)
            .map(|(&id, _)| id);
        tid
    } else {
        None
    };

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
        NamedRule { name: name.into(), expr, is_terminal: false }
    }

    fn term(name: &str, expr: GrammarExpr) -> NamedRule {
        NamedRule { name: name.into(), expr, is_terminal: true }
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

    /// Adapted from sep1 `test_nullability_handling_in_from_exprs`.
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
}
