use crate::constraint::GrammarConstraint;
use crate::debug;
use crate::finite_automata::{greedy_group, groups, Expr, ExprGroup, GroupID, QuantifierType, Regex};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::parser::GLRParser;
use crate::glr::table::{assign_non_terminal_ids, generate_glr_parser, generate_glr_parser_with_maps, generate_glr_parser_with_terminal_map, NonTerminalID, TerminalID};
use crate::interface::ebnf::{EbnfParseResult, EbnfParser};
use crate::interface::lark::{LarkParser, LarkParseResult};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::types::TerminalID as GrammarTokenID;
use crate::datastructures::u8set::U8Set;
use crate::glr::analyze::simplify_grammar;
use crate::glr::grammar::regex_name;
// May not be used directly here anymore
use bimap::BiBTreeMap;
use json_convertible_derive::JSONConvertible;
use kdam::tqdm;
use std::collections::BTreeMap as StdMap;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use std::fs;

type LLMToken<'a> = &'a [u8];
type LLMTokenMap = BiBTreeMap<Vec<u8>, LLMTokenID>;

// --- Nullability analysis for regex expressions ---
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum ExprNullability {
    NeverNull,
    CanBeNull,
    AlwaysNull,
}

/// Check the nullability of a regex expression.
/// Returns whether the expression can/must/cannot match the empty string.
pub fn get_expr_nullability(expr: &Expr) -> ExprNullability {
    fn _get_nullability(
        expr: &Expr,
        cache: &mut HashMap<*const Expr, ExprNullability>,
    ) -> ExprNullability {
        match expr {
            Expr::U8Seq(bytes) => {
                if bytes.is_empty() {
                    ExprNullability::AlwaysNull
                } else {
                    ExprNullability::NeverNull
                }
            }
            Expr::U8Class(_) => ExprNullability::NeverNull,
            Expr::Quantifier(inner, q_type) => match q_type {
                QuantifierType::ZeroOrMore => ExprNullability::CanBeNull,
                QuantifierType::OneOrMore => _get_nullability(inner, cache),
                QuantifierType::ZeroOrOne => ExprNullability::CanBeNull,
            },
            Expr::Choice(exprs) => {
                let nullabilities: Vec<ExprNullability> = exprs
                    .iter()
                    .map(|e| _get_nullability(e, cache))
                    .collect();
                if nullabilities.iter().any(|n| {
                    matches!(n, ExprNullability::AlwaysNull | ExprNullability::CanBeNull)
                }) {
                    ExprNullability::CanBeNull
                } else {
                    ExprNullability::NeverNull
                }
            }
            Expr::Seq(exprs) => {
                let nullabilities: Vec<ExprNullability> = exprs
                    .iter()
                    .map(|e| _get_nullability(e, cache))
                    .collect();
                if nullabilities
                    .iter()
                    .all(|n| matches!(n, ExprNullability::AlwaysNull | ExprNullability::CanBeNull))
                {
                    ExprNullability::CanBeNull
                } else if nullabilities
                    .iter()
                    .any(|n| *n == ExprNullability::NeverNull)
                {
                    ExprNullability::NeverNull
                } else {
                    ExprNullability::NeverNull
                }
            }
            Expr::Epsilon => ExprNullability::AlwaysNull,
            Expr::Shared(arc) => {
                let ptr = Arc::as_ptr(arc) as *const Expr;
                if let Some(&cached_nullability) = cache.get(&ptr) {
                    cached_nullability
                } else {
                    let nullability = _get_nullability(arc.as_ref(), cache);
                    cache.insert(ptr, nullability);
                    nullability
                }
            }
        }
    }
    let mut cache: HashMap<*const Expr, ExprNullability> = HashMap::new();
    _get_nullability(expr, &mut cache)
}

// --- GrammarExpr: Definition of grammar structure before compilation ---
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GrammarExpr {
    Ref(String),
    Sequence(Vec<GrammarExpr>),
    Choice(Vec<GrammarExpr>),
    Optional(Box<GrammarExpr>),
    Repeat(Box<GrammarExpr>), // Zero or more repetition
    Literal(Vec<u8>),
    CharClass(String),
    AnyChar,
}

/// Intermediate type for GrammarExpr JSON serialization (maintains backward compatibility)
#[derive(JSONConvertible)]
enum GrammarExprJSON {
    Ref { name: String },
    Sequence { exprs: Vec<GrammarExprJSON> },
    Choice { exprs: Vec<GrammarExprJSON> },
    Optional { expr: Box<GrammarExprJSON> },
    Repeat { expr: Box<GrammarExprJSON> },
    Literal { bytes: Vec<u8> },
    CharClass { def: String },
    AnyChar,
}

impl GrammarExprJSON {
    fn from_expr(e: &GrammarExpr) -> Self {
        match e {
            GrammarExpr::Ref(name) => GrammarExprJSON::Ref { name: name.clone() },
            GrammarExpr::Sequence(exprs) => GrammarExprJSON::Sequence {
                exprs: exprs.iter().map(GrammarExprJSON::from_expr).collect(),
            },
            GrammarExpr::Choice(exprs) => GrammarExprJSON::Choice {
                exprs: exprs.iter().map(GrammarExprJSON::from_expr).collect(),
            },
            GrammarExpr::Optional(expr) => GrammarExprJSON::Optional {
                expr: Box::new(GrammarExprJSON::from_expr(expr)),
            },
            GrammarExpr::Repeat(expr) => GrammarExprJSON::Repeat {
                expr: Box::new(GrammarExprJSON::from_expr(expr)),
            },
            GrammarExpr::Literal(bytes) => GrammarExprJSON::Literal { bytes: bytes.clone() },
            GrammarExpr::CharClass(s) => GrammarExprJSON::CharClass { def: s.clone() },
            GrammarExpr::AnyChar => GrammarExprJSON::AnyChar,
        }
    }

    fn to_expr(self) -> GrammarExpr {
        match self {
            GrammarExprJSON::Ref { name } => GrammarExpr::Ref(name),
            GrammarExprJSON::Sequence { exprs } => {
                GrammarExpr::Sequence(exprs.into_iter().map(|e| e.to_expr()).collect())
            }
            GrammarExprJSON::Choice { exprs } => {
                GrammarExpr::Choice(exprs.into_iter().map(|e| e.to_expr()).collect())
            }
            GrammarExprJSON::Optional { expr } => GrammarExpr::Optional(Box::new(expr.to_expr())),
            GrammarExprJSON::Repeat { expr } => GrammarExpr::Repeat(Box::new(expr.to_expr())),
            GrammarExprJSON::Literal { bytes } => GrammarExpr::Literal(bytes),
            GrammarExprJSON::CharClass { def } => GrammarExpr::CharClass(def),
            GrammarExprJSON::AnyChar => GrammarExpr::AnyChar,
        }
    }
}

impl JSONConvertible for GrammarExpr {
    fn to_json(&self) -> JSONNode {
        GrammarExprJSON::from_expr(self).to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        GrammarExprJSON::from_json(node).map(|e| e.to_expr())
    }
}

// Helper functions to construct GrammarExpr
pub fn r#ref(name: &str) -> GrammarExpr {
    GrammarExpr::Ref(name.to_string())
}
pub fn sequence(exprs: Vec<GrammarExpr>) -> GrammarExpr {
    GrammarExpr::Sequence(exprs)
}
pub fn choice(exprs: Vec<GrammarExpr>) -> GrammarExpr {
    GrammarExpr::Choice(exprs)
}
pub fn optional(expr: GrammarExpr) -> GrammarExpr {
    GrammarExpr::Optional(Box::new(expr))
}
pub fn repeat(expr: GrammarExpr) -> GrammarExpr {
    GrammarExpr::Repeat(Box::new(expr))
}
pub fn literal(bytes: Vec<u8>) -> GrammarExpr {
    GrammarExpr::Literal(bytes)
}

// --- GrammarDefinition: Abstract representation of the grammar ---
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrammarDefinition {
    pub productions: Vec<Production>,
    pub start_production_id: usize, // Index into productions
    pub literal_to_group_id: BiBTreeMap<Vec<u8>, usize>,
    pub regex_name_to_group_id: BiBTreeMap<String, usize>,
    pub group_id_to_expr: BTreeMap<usize, Expr>,
    pub ignore_terminal_id: Option<TerminalID>,
    pub external_name_to_group_id: BiBTreeMap<String, usize>,
}

impl GrammarDefinition {
    pub fn terminal_to_group_id(&self) -> BiBTreeMap<Terminal, usize> {
        let mut terminal_to_group_id = BiBTreeMap::new();
        for (name, group_id) in &self.regex_name_to_group_id {
            let terminal = Terminal::RegexName(name.clone());
            terminal_to_group_id.insert(terminal, *group_id);
        }
        for (literal, group_id) in &self.literal_to_group_id {
            let terminal = Terminal::Literal(literal.clone());
            terminal_to_group_id.insert(terminal, *group_id);
        }
        for (name, group_id) in &self.external_name_to_group_id {
            let terminal = Terminal::RegexName(name.clone());
            terminal_to_group_id.insert(terminal, *group_id);
        }
        terminal_to_group_id
    }
}

impl JSONConvertible for GrammarDefinition {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("productions".to_string(), self.productions.to_json());
        obj.insert(
            "start_production_id".to_string(),
            self.start_production_id.to_json(),
        );
        obj.insert(
            "ignore_terminal_id".to_string(),
            self.ignore_terminal_id.to_json(),
        );
        obj.insert(
            "external_name_to_group_id".to_string(),
            self.external_name_to_group_id.to_json(),
        );

        let mut regexes_json_list = Vec::new();
        let mut sorted_regexes_info: Vec<(usize, String, Expr)> = Vec::new();
        for (name, group_id) in &self.regex_name_to_group_id {
            let expr = self
                .group_id_to_expr
                .get(group_id)
                .unwrap_or_else(|| {
                    panic!(
                        "Internal consistency error: group_id {} for name '{}' not found in group_id_to_expr.",
                        group_id, name
                    )
                })
                .clone();
            sorted_regexes_info.push((*group_id, name.clone(), expr));
        }
        sorted_regexes_info.sort_by_key(|(group_id, _, _)| *group_id);
        for (group_id, name, expr) in sorted_regexes_info {
            let mut terminal_obj = StdMap::new();
            terminal_obj.insert("name".to_string(), name.to_json());
            terminal_obj.insert("group_id".to_string(), group_id.to_json());
            terminal_obj.insert("expr".to_string(), expr.to_json());
            regexes_json_list.push(JSONNode::Object(terminal_obj));
        }
        obj.insert(
            "regex_terminals".to_string(),
            JSONNode::Array(regexes_json_list),
        );

        let mut literals_json_list = Vec::new();
        let mut sorted_literals_info: Vec<(usize, Vec<u8>)> = self
            .literal_to_group_id
            .iter()
            .map(|(val, gid)| (*gid, val.clone()))
            .collect();
        sorted_literals_info.sort_by_key(|(group_id, _)| *group_id);

        for (group_id, val) in sorted_literals_info {
            let mut literal_obj = StdMap::new();
            literal_obj.insert("value".to_string(), val.to_json());
            literal_obj.insert("group_id".to_string(), group_id.to_json());
            literals_json_list.push(JSONNode::Object(literal_obj));
        }
        obj.insert(
            "literal_terminals".to_string(),
            JSONNode::Array(literals_json_list),
        );

        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let productions = obj
                    .remove("productions")
                    .ok_or_else(|| "Missing field productions for GrammarDefinition".to_string())
                    .and_then(Vec::<Production>::from_json)?;
                let start_production_id = obj
                    .remove("start_production_id")
                    .ok_or_else(|| {
                        "Missing field start_production_id for GrammarDefinition".to_string()
                    })
                    .and_then(usize::from_json)?;
                let ignore_terminal_id = obj
                    .remove("ignore_terminal_id")
                    .ok_or_else(|| {
                        "Missing field ignore_terminal_id for GrammarDefinition".to_string()
                    })
                    .and_then(Option::<TerminalID>::from_json)?;
                let external_name_to_group_id = obj
                    .remove("external_name_to_group_id")
                    .map(|node| BiBTreeMap::<String, usize>::from_json(node))
                    .transpose()?
                    .unwrap_or_default();

                let mut new_literal_to_group_id = BiBTreeMap::new();
                let mut new_regex_name_to_group_id = BiBTreeMap::new();
                let mut new_group_id_to_expr = BTreeMap::new();

                let regex_terminals_node = obj.remove("regex_terminals").ok_or_else(|| {
                    "Missing field regex_terminals for GrammarDefinition".to_string()
                })?;
                if let JSONNode::Array(terminals_list) = regex_terminals_node {
                    for terminal_node in terminals_list {
                        if let JSONNode::Object(mut terminal_obj) = terminal_node {
                            let name = String::from_json(
                                terminal_obj.remove("name").ok_or("Missing name")?,
                            )?;
                            let group_id = usize::from_json(
                                terminal_obj
                                    .remove("group_id")
                                    .ok_or("Missing group_id")?,
                            )?;
                            let expr =
                                Expr::from_json(terminal_obj.remove("expr").ok_or("Missing expr")?)?;
                            new_regex_name_to_group_id.insert(name, group_id);
                            new_group_id_to_expr.insert(group_id, expr);
                        }
                    }
                }

                let literal_terminals_node = obj.remove("literal_terminals").ok_or_else(|| {
                    "Missing field literal_terminals for GrammarDefinition".to_string()
                })?;
                if let JSONNode::Array(literals_list) = literal_terminals_node {
                    for literal_node in literals_list {
                        if let JSONNode::Object(mut literal_obj) = literal_node {
                            let value = Vec::<u8>::from_json(
                                literal_obj.remove("value").ok_or("Missing value")?,
                            )?;
                            let group_id = usize::from_json(
                                literal_obj
                                    .remove("group_id")
                                    .ok_or("Missing group_id")?,
                            )?;
                            new_literal_to_group_id.insert(value.clone(), group_id);
                            new_group_id_to_expr.insert(group_id, Expr::U8Seq(value));
                        }
                    }
                }

                Ok(GrammarDefinition {
                    productions,
                    start_production_id,
                    regex_name_to_group_id: new_regex_name_to_group_id,
                    literal_to_group_id: new_literal_to_group_id,
                    group_id_to_expr: new_group_id_to_expr,
                    ignore_terminal_id,
                    external_name_to_group_id,
                })
            }
            _ => Err("Expected JSONNode::Object for GrammarDefinition".to_string()),
        }
    }
}

impl Display for GrammarDefinition {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GrammarDefinition:")?;
        writeln!(f, "  Start Production ID: {}", self.start_production_id)?;
        writeln!(f, "  Productions ({}):", self.productions.len())?;
        for production in &self.productions {
            write!(f, "    {} -> ", production.lhs.0)?;
            for (i, symbol) in production.rhs.iter().enumerate() {
                match symbol {
                    Symbol::Terminal(terminal) => write!(f, "{}", terminal.to_string())?,
                    Symbol::NonTerminal(non_terminal) => write!(f, "{}", non_terminal.0)?,
                }
                if i < production.rhs.len() - 1 {
                    write!(f, " ")?;
                }
            }
            writeln!(f)?;
        }
        writeln!(f)?;
        Ok(())
    }
}

pub fn display_productions(productions: &[Production]) -> String {
    let mut result = String::new();
    for prod in productions {
        result.push_str(&format!(
            "{} -> {}\n",
            prod.lhs.0,
            prod.rhs
                .iter()
                .map(|symbol| match symbol {
                    Symbol::Terminal(t) => t.to_string(),
                    Symbol::NonTerminal(nt) => nt.to_string(),
                })
                .collect::<Vec<_>>()
                .join(" ")
        ));
    }
    result
}

impl GrammarDefinition {
    /// Converts the grammar definition back into a string in EBNF format.
    ///
    /// This is useful for debugging and inspecting grammars, especially after minimization.
    /// Note that this is a direct translation of the production rules and will not reconstruct
    /// higher-level EBNF operators like `*`, `+`, or `?`.
    pub fn to_ebnf(&self) -> String {
        let mut ebnf_string = String::new();

        // Group productions by LHS
        let mut prods_by_lhs: BTreeMap<NonTerminal, Vec<&[Symbol]>> = BTreeMap::new();
        for prod in &self.productions {
            prods_by_lhs.entry(prod.lhs.clone()).or_default().push(&prod.rhs);
        }

        // Process non-terminals in alphabetical order for deterministic output.
        for (nt, rhss) in prods_by_lhs {
            ebnf_string.push_str(&format!("{} ::= ", nt.0));

            for (i, rhs) in rhss.iter().enumerate() {
                if i > 0 {
                    ebnf_string.push_str("\n  | ");
                }

                if rhs.is_empty() {
                    // Epsilon production is an empty sequence before the semicolon.
                } else {
                    let rhs_str: Vec<String> = rhs
                        .iter()
                        .map(|symbol| match symbol {
                            Symbol::NonTerminal(nt) => nt.0.clone(),
                            Symbol::Terminal(t) => t.to_string(),
                        })
                        .collect();
                    ebnf_string.push_str(&rhs_str.join(" "));
                }
            }
            ebnf_string.push_str(" ;\n");
        }
        ebnf_string
    }

    pub fn add_external_terminal(&mut self, name: &str) -> usize {
        if let Some(group_id) = self.external_name_to_group_id.get_by_left(name) {
            return *group_id;
        }
        if self.regex_name_to_group_id.contains_left(name) {
            panic!(
                "External terminal name '{}' conflicts with an existing terminal in the grammar.",
                name
            );
        }

        let all_gids: BTreeSet<usize> = self
            .group_id_to_expr
            .keys()
            .copied()
            .chain(self.external_name_to_group_id.right_values().copied())
            .collect();

        let new_group_id = all_gids.iter().max().map(|max_id| max_id + 1).unwrap_or(0);

        self.external_name_to_group_id
            .insert(name.to_string(), new_group_id);
        new_group_id
    }
}

impl GrammarDefinition {
    /// Generates a unique indexed name (e.g., Base[0], Base[1]) avoiding collisions.
    fn generate_unique_indexed_name(
        base_name: &str,
        counters: &mut HashMap<String, usize>,
        all_existing_names: &mut HashSet<String>,
    ) -> String {
        let idx_ref = counters.entry(base_name.to_string()).or_insert(0);
        let mut current_idx = *idx_ref;
        loop {
            let new_name = format!("{}[{}]", base_name, current_idx);
            if !all_existing_names.contains(&new_name) {
                all_existing_names.insert(new_name.clone());
                *idx_ref = current_idx + 1;
                return new_name;
            }
            current_idx += 1;
        }
    }

    /// Generates a unique name for a terminal derived from a literal byte sequence.
    fn generate_unique_indexed_name_for_literal(
        bytes: &[u8],
        counters: &mut HashMap<String, usize>,
        all_names: &mut HashSet<String>,
    ) -> String {
        match String::from_utf8(bytes.to_vec()) {
            Ok(s) if !s.is_empty() && !s.contains('[') && !s.contains(']') && !s.contains('\"') => {
                let s = format!("\"{}\"", s.escape_debug().to_string());
                if !all_names.contains(&s) {
                    all_names.insert(s.clone());
                    s
                } else {
                    Self::generate_unique_indexed_name(&s, counters, all_names)
                }
            }
            _ => {
                let base_name = format!(
                    "\"{}\"",
                    String::from_utf8_lossy(bytes).escape_debug().to_string()
                );
                Self::generate_unique_indexed_name(&base_name, counters, all_names)
            }
        }
    }

    /// Handles nullable terminals in the grammar by:
    /// - Removing always-null terminals from production RHSs
    /// - Replacing may-be-null terminals with optional non-terminals
    /// 
    /// If `already_processed` is provided, terminals in that set are skipped.
    /// This is used when calling this after optimization to avoid re-processing
    /// terminals that were already handled in from_exprs.
    pub fn handle_nullable_terminals_except(&mut self, already_processed: &HashSet<String>) {
        // Collect all existing names to avoid collisions when generating new non-terminal names
        let mut all_names: HashSet<String> = self
            .productions
            .iter()
            .map(|p| p.lhs.0.clone())
            .collect();
        for name in self.regex_name_to_group_id.left_values() {
            all_names.insert(name.clone());
        }
        
        let mut per_base_counters: HashMap<String, usize> = HashMap::new();
        
        // Identify nullable terminals
        let mut always_null_terminals: HashSet<String> = HashSet::new();
        let mut may_be_null_terminals: HashSet<String> = HashSet::new();

        for (terminal_name, group_id) in self.regex_name_to_group_id.iter() {
            // Skip terminals that were already processed
            if already_processed.contains(terminal_name) {
                continue;
            }
            
            let expr = self
                .group_id_to_expr
                .get(group_id)
                .expect("regex_name_to_group_id / group_id_to_expr out of sync");

            match get_expr_nullability(expr) {
                ExprNullability::AlwaysNull => {
                    always_null_terminals.insert(terminal_name.clone());
                }
                ExprNullability::CanBeNull => {
                    may_be_null_terminals.insert(terminal_name.clone());
                }
                ExprNullability::NeverNull => {}
            }
        }

        if always_null_terminals.is_empty() && may_be_null_terminals.is_empty() {
            return; // Nothing to do
        }

        debug!(
            4,
            "Removing {} always-null terminals after optimization",
            always_null_terminals.len()
        );
        
        // Remove always-null terminals from production RHSs
        for prod in self.productions.iter_mut() {
            prod.rhs.retain(|sym| match sym {
                Symbol::Terminal(Terminal::RegexName(t)) => !always_null_terminals.contains(t),
                _ => true,
            });
        }

        debug!(
            4,
            "Processing {} may-be-null terminals after optimization",
            may_be_null_terminals.len()
        );
        
        // Replace may-be-null terminals with optional non-terminals
        for terminal_name in &may_be_null_terminals {
            let opt_nt_name = Self::generate_unique_indexed_name(
                &format!("{}Opt", terminal_name.trim_matches('"')),
                &mut per_base_counters,
                &mut all_names,
            );
            let opt_nt = NonTerminal(opt_nt_name.clone());

            for prod in self.productions.iter_mut() {
                for sym in &mut prod.rhs {
                    if let Symbol::Terminal(Terminal::RegexName(t)) = sym {
                        if t == terminal_name {
                            *sym = Symbol::NonTerminal(opt_nt.clone());
                        }
                    }
                }
            }

            // Add the optional non-terminal productions: one with the terminal, one epsilon
            self.productions.push(Production {
                lhs: opt_nt.clone(),
                rhs: vec![Symbol::Terminal(regex_name(terminal_name))],
            });
            self.productions.push(Production {
                lhs: opt_nt.clone(),
                rhs: Vec::new(), // epsilon
            });
        }
    }
    
    /// Handles nullable terminals in the grammar. This processes ALL terminals.
    /// Use `handle_nullable_terminals_except` if you want to skip already-processed terminals.
    pub fn handle_nullable_terminals(&mut self) {
        self.handle_nullable_terminals_except(&HashSet::new());
    }

    /// Converts a `GrammarExpr` into a sequence of `Symbol`s and a list of newly created `Production`s.
    fn convert_grammar_expr_to_symbols(
        expr: &GrammarExpr,
        current_rule_name_or_path: &str,
        literal_to_group_id: &mut BiBTreeMap<Vec<u8>, usize>,
        nonterminal_names: &HashSet<&str>,
        regex_name_to_group_id: &mut BiBTreeMap<String, usize>,
        regex_expr_to_group_id: &mut BiBTreeMap<Expr, usize>,
        next_terminal_group_id: &mut usize,
        per_base_counters: &mut HashMap<String, usize>,
        all_names: &mut HashSet<String>,
        memo: &mut BTreeMap<GrammarExpr, NonTerminal>,
    ) -> Result<(Vec<Symbol>, Vec<Production>), String> {
        match expr {
            GrammarExpr::AnyChar => Err(
                "AnyChar (`.`) is only allowed inside terminal definitions (rules with uppercase names)."
                    .to_string(),
            ),
            GrammarExpr::CharClass(class_def) => Err(format!(
                "Character class `{}` is only allowed inside terminal definitions (rules with uppercase names).",
                class_def
            )),
            GrammarExpr::Literal(bytes) => {
                let literal_expr = Expr::U8Seq(bytes.clone());

                if !regex_expr_to_group_id.contains_left(&literal_expr) {
                    let gid = *next_terminal_group_id;
                    *next_terminal_group_id += 1;
                    regex_expr_to_group_id.insert(literal_expr.clone(), gid);
                    literal_to_group_id.insert(bytes.clone(), gid);
                }

                Ok((
                    vec![Symbol::Terminal(Terminal::Literal(bytes.clone()))],
                    Vec::new(),
                ))
            }
            GrammarExpr::Ref(name) => {
                if nonterminal_names.contains(name.as_str()) {
                    Ok((vec![Symbol::NonTerminal(NonTerminal(name.clone()))], Vec::new()))
                } else {
                    Ok((vec![Symbol::Terminal(regex_name(name))], Vec::new()))
                }
            }
            GrammarExpr::Sequence(exprs) => {
                let mut combined_symbols = Vec::new();
                let mut combined_productions = Vec::new();
                for e in exprs {
                    let (symbols, new_productions) = Self::convert_grammar_expr_to_symbols(
                        e,
                        current_rule_name_or_path,
                        literal_to_group_id,
                        nonterminal_names,
                        regex_name_to_group_id,
                        regex_expr_to_group_id,
                        next_terminal_group_id,
                        per_base_counters,
                        all_names,
                        memo,
                    )?;
                    combined_symbols.extend(symbols);
                    combined_productions.extend(new_productions);
                }
                Ok((combined_symbols, combined_productions))
            }
            GrammarExpr::Choice(exprs) => {
                if let Some(nt) = memo.get(expr) {
                    return Ok((vec![Symbol::NonTerminal(nt.clone())], Vec::new()));
                }
                let choice_nt_name = Self::generate_unique_indexed_name(
                    current_rule_name_or_path,
                    per_base_counters,
                    all_names,
                );
                let nt = NonTerminal(choice_nt_name.clone());

                let mut choice_defining_productions = Vec::new();
                let mut children_productions_from_arms = Vec::new();

                for expr_choice_item in exprs {
                    let (rhs_symbols_for_arm, productions_from_arm_processing) =
                        Self::convert_grammar_expr_to_symbols(
                            expr_choice_item,
                            &choice_nt_name,
                            literal_to_group_id,
                            nonterminal_names,
                            regex_name_to_group_id,
                            regex_expr_to_group_id,
                            next_terminal_group_id,
                            per_base_counters,
                            all_names,
                            memo,
                        )?;
                    choice_defining_productions.push(Production {
                        lhs: nt.clone(),
                        rhs: rhs_symbols_for_arm,
                    });
                    children_productions_from_arms.extend(productions_from_arm_processing);
                }

                let mut all_new_productions = choice_defining_productions;
                all_new_productions.extend(children_productions_from_arms);

                memo.insert(expr.clone(), nt.clone());
                Ok((vec![Symbol::NonTerminal(nt)], all_new_productions))
            }
            GrammarExpr::Optional(expr_box) => {
                Self::convert_grammar_expr_to_symbols(
                    &GrammarExpr::Choice(vec![*expr_box.clone(), GrammarExpr::Sequence(vec![])]),
                    current_rule_name_or_path,
                    literal_to_group_id,
                    nonterminal_names,
                    regex_name_to_group_id,
                    regex_expr_to_group_id,
                    next_terminal_group_id,
                    per_base_counters,
                    all_names,
                    memo,
                )
            }
            GrammarExpr::Repeat(expr_box) => {
                if let Some(nt) = memo.get(expr) {
                    return Ok((vec![Symbol::NonTerminal(nt.clone())], Vec::new()));
                }
                let repeat_nt_name = Self::generate_unique_indexed_name(
                    current_rule_name_or_path,
                    per_base_counters,
                    all_names,
                );
                let nt = NonTerminal(repeat_nt_name.clone());

                let (expr_symbols, productions_from_expr_box) =
                    Self::convert_grammar_expr_to_symbols(
                        expr_box,
                        &repeat_nt_name,
                        literal_to_group_id,
                        nonterminal_names,
                        regex_name_to_group_id,
                        regex_expr_to_group_id,
                        next_terminal_group_id,
                        per_base_counters,
                        all_names,
                        memo,
                    )?;

                let mut current_level_productions = Vec::new();
                if !expr_symbols.is_empty() {
                    current_level_productions.push(Production {
                        lhs: nt.clone(),
                        rhs: {
                            let mut r = Vec::new();
                            r.push(Symbol::NonTerminal(nt.clone()));
                            r.extend(expr_symbols);
                            r
                        },
                    });
                }
                current_level_productions.push(Production {
                    lhs: nt.clone(),
                    rhs: vec![],
                });

                let mut all_new_productions = current_level_productions;
                all_new_productions.extend(productions_from_expr_box);

                memo.insert(expr.clone(), nt.clone());
                Ok((vec![Symbol::NonTerminal(nt)], all_new_productions))
            }
        }
    }

    fn convert_grammar_expr_to_regex_expr(
        grammar_expr: &GrammarExpr,
        unresolved_terminals: &BTreeMap<String, GrammarExpr>,
        memo: &mut BTreeMap<String, Expr>,
        resolving_stack: &mut HashSet<String>,
    ) -> Result<Expr, String> {
        match grammar_expr {
            GrammarExpr::AnyChar => Ok(Expr::U8Class(U8Set::all())),
            GrammarExpr::Literal(bytes) => Ok(Expr::U8Seq(bytes.clone())),
            GrammarExpr::CharClass(class_def) => {
                let content = &class_def[1..class_def.len() - 1];
                let (negated, content) = if content.starts_with('^') {
                    (true, &content[1..])
                } else {
                    (false, content)
                };

                let mut u8set = U8Set::none();
                let mut it = content.chars().peekable();

                let mut parse_char =
                    |it: &mut std::iter::Peekable<std::str::Chars>| -> Result<Option<char>, String> {
                        if let Some(c) = it.next() {
                            if c == '\\' {
                                if let Some(escaped_char) = it.next() {
                                    Ok(Some(match escaped_char {
                                        'n' => '\n',
                                        't' => '\t',
                                        'r' => '\r',
                                        'b' => '\u{0008}',
                                        'f' => '\u{000C}',
                                        'v' => '\u{000B}',
                                        '\\' => '\\',
                                        other => other,
                                    }))
                                } else {
                                    Err(format!(
                                        "Dangling escape in char class: {}",
                                        class_def
                                    ))
                                }
                            } else {
                                Ok(Some(c))
                            }
                        } else {
                            Ok(None)
                        }
                    };

                while let Some(start_char) = parse_char(&mut it)? {
                    if it.peek() == Some(&'-') {
                        it.next();
                        if let Some(end_char) = parse_char(&mut it)? {
                            for i in (start_char as u8)..=(end_char as u8) {
                                u8set.insert(i);
                            }
                        } else {
                            u8set.insert(start_char as u8);
                            u8set.insert(b'-');
                        }
                    } else {
                        u8set.insert(start_char as u8);
                    }
                }
                Ok(Expr::U8Class(if negated {
                    u8set.complement()
                } else {
                    u8set
                }))
            }
            GrammarExpr::Ref(name) => {
                if let Some(resolved_expr) = memo.get(name) {
                    return Ok(resolved_expr.clone());
                }

                if resolving_stack.contains(name) {
                    return Err(format!(
                        "Cyclic reference in terminal definitions involving '{}'",
                        name
                    ));
                }

                if let Some(terminal_expr) = unresolved_terminals.get(name) {
                    resolving_stack.insert(name.clone());
                    let result = Self::convert_grammar_expr_to_regex_expr(
                        terminal_expr,
                        unresolved_terminals,
                        memo,
                        resolving_stack,
                    );
                    resolving_stack.remove(name);

                    let resolved_expr = result?;
                    memo.insert(name.clone(), resolved_expr.clone());
                    Ok(resolved_expr)
                } else {
                    Err(format!(
                        "Non-terminal reference '{}' found in a terminal definition. Terminal definitions cannot contain non-terminal references.",
                        name
                    ))
                }
            }
            GrammarExpr::Sequence(exprs) => {
                if exprs.is_empty() {
                    return Ok(Expr::Epsilon);
                }
                let mut sub_exprs = Vec::new();
                for e in exprs {
                    sub_exprs.push(Self::convert_grammar_expr_to_regex_expr(
                        e,
                        unresolved_terminals,
                        memo,
                        resolving_stack,
                    )?);
                }
                Ok(Expr::Seq(sub_exprs))
            }
            GrammarExpr::Choice(exprs) => {
                let mut sub_exprs = Vec::new();
                for e in exprs {
                    sub_exprs.push(Self::convert_grammar_expr_to_regex_expr(
                        e,
                        unresolved_terminals,
                        memo,
                        resolving_stack,
                    )?);
                }
                Ok(Expr::Choice(sub_exprs))
            }
            GrammarExpr::Optional(expr) => {
                let sub_expr = Self::convert_grammar_expr_to_regex_expr(
                    expr,
                    unresolved_terminals,
                    memo,
                    resolving_stack,
                )?;
                Ok(Expr::Quantifier(
                    Box::new(sub_expr),
                    QuantifierType::ZeroOrOne,
                ))
            }
            GrammarExpr::Repeat(expr) => {
                let sub_expr = Self::convert_grammar_expr_to_regex_expr(
                    expr,
                    unresolved_terminals,
                    memo,
                    resolving_stack,
                )?;
                Ok(Expr::Quantifier(
                    Box::new(sub_expr),
                    QuantifierType::ZeroOrMore,
                ))
            }
        }
    }

    /// Constructs a `GrammarDefinition` from a list of grammar expressions.
    pub fn from_exprs(
        grammar_exprs: Vec<(String, GrammarExpr)>,
        regex_exprs: Vec<(String, Expr)>,
    ) -> Result<Self, String> {
        Self::from_exprs_with_ignore(grammar_exprs, regex_exprs, None)
    }

    /// Constructs a `GrammarDefinition` from a list of `(name, GrammarExpr)` tuples
    /// with optional ignore symbol.
    pub fn from_exprs_with_ignore(
        grammar_exprs: Vec<(String, GrammarExpr)>,
        regex_exprs: Vec<(String, Expr)>,
        ignore_symbol_name: Option<&str>,
    ) -> Result<Self, String> {
        if grammar_exprs.is_empty() {
            return Err("Grammar expressions list cannot be empty.".to_string());
        }

        let mut literal_to_group_id: BiBTreeMap<Vec<u8>, usize> = BiBTreeMap::new();
        let mut regex_name_to_group_id: BiBTreeMap<String, usize> = BiBTreeMap::new();
        let mut group_id_to_expr: BTreeMap<usize, Expr> = BTreeMap::new();
        let mut next_terminal_group_id = 0;

        // Process predefined terminals
        for (name, expr) in regex_exprs {
            if regex_name_to_group_id.contains_left(&name) {
                return Err(format!("Duplicate terminal name defined: {}", name));
            }
            let group_id = next_terminal_group_id;
            regex_name_to_group_id.insert(name, group_id);
            group_id_to_expr.insert(group_id, expr);
            next_terminal_group_id += 1;
        }

        let mut all_names: HashSet<String> =
            grammar_exprs.iter().map(|(name, _)| name.clone()).collect();
        all_names.extend(regex_name_to_group_id.left_values().cloned());
        let mut per_base_counters: HashMap<String, usize> = HashMap::new();
        let mut memo: BTreeMap<GrammarExpr, NonTerminal> = BTreeMap::new();

        let mut start_production_name = "start'".to_string();
        let nonterminal_names_from_rules: HashSet<&str> =
            grammar_exprs.iter().map(|(name, _)| name.as_str()).collect();
        while nonterminal_names_from_rules.contains(start_production_name.as_str())
            || all_names.contains(&start_production_name)
        {
            start_production_name.push('\'');
        }
        all_names.insert(start_production_name.clone());
        debug!(5, "Augmented start_production_name: {:?}", start_production_name);

        let mut productions = vec![Production {
            lhs: NonTerminal(start_production_name.clone()),
            rhs: vec![Symbol::NonTerminal(NonTerminal(
                grammar_exprs[0].0.clone(),
            ))],
        }];
        let start_production_id = 0;

        let it = grammar_exprs.iter();
        #[cfg(not(rustrover))]
        let it = tqdm!(
            grammar_exprs.iter(),
            disable = !crate::r#macro::should_show_progress_bars(),
            leave = false,
            desc = "Converting grammar expressions to productions"
        );
        let mut anon_regex_expr_to_group_id = BiBTreeMap::new();
        for (name, expr) in it {
            let lhs = NonTerminal(name.clone());
            let lhs_name_str = name;

            if let GrammarExpr::Choice(choices) = expr {
                for choice_expr in choices {
                    let (rhs_symbols_for_arm, new_productions_for_arm) =
                        Self::convert_grammar_expr_to_symbols(
                            choice_expr,
                            lhs_name_str,
                            &mut literal_to_group_id,
                            &nonterminal_names_from_rules,
                            &mut regex_name_to_group_id,
                            &mut anon_regex_expr_to_group_id,
                            &mut next_terminal_group_id,
                            &mut per_base_counters,
                            &mut all_names,
                            &mut memo,
                        )?;
                    productions.push(Production {
                        lhs: lhs.clone(),
                        rhs: rhs_symbols_for_arm,
                    });
                    productions.extend(new_productions_for_arm);
                }
            } else {
                let (rhs_symbols, new_productions_for_rhs) =
                    Self::convert_grammar_expr_to_symbols(
                        expr,
                        lhs_name_str,
                        &mut literal_to_group_id,
                        &nonterminal_names_from_rules,
                        &mut regex_name_to_group_id,
                        &mut anon_regex_expr_to_group_id,
                        &mut next_terminal_group_id,
                        &mut per_base_counters,
                        &mut all_names,
                        &mut memo,
                    )?;
                productions.push(Production { lhs, rhs: rhs_symbols });
                productions.extend(new_productions_for_rhs);
            }
        }

        for (expr, group_id) in anon_regex_expr_to_group_id {
            group_id_to_expr.insert(group_id, expr);
        }

        // Nullability analysis for terminals:
        //   - always-null terminals are removed from RHSs;
        //   - sometimes-null terminals are desugared into optional non-terminals.
        let mut always_null_terminals: HashSet<String> = HashSet::new();
        let mut may_be_null_terminals: HashSet<String> = HashSet::new();

        for (terminal_name, group_id) in regex_name_to_group_id.iter() {
            let expr = group_id_to_expr
                .get(group_id)
                .expect("regex_name_to_group_id / group_id_to_expr out of sync");

            match get_expr_nullability(expr) {
                ExprNullability::AlwaysNull => {
                    always_null_terminals.insert(terminal_name.clone());
                }
                ExprNullability::CanBeNull => {
                    may_be_null_terminals.insert(terminal_name.clone());
                }
                ExprNullability::NeverNull => {}
            }
        }

        debug!(
            4,
            "Removing {} always-null terminals",
            always_null_terminals.len()
        );
        let mut updated_productions: Vec<Production> = Vec::with_capacity(productions.len());
        for prod in productions.into_iter() {
            let filtered_rhs: Vec<Symbol> = prod
                .rhs
                .into_iter()
                .filter(|sym| match sym {
                    Symbol::Terminal(Terminal::RegexName(t)) => {
                        !always_null_terminals.contains(t)
                    }
                    _ => true,
                })
                .collect();

            updated_productions.push(Production {
                lhs: prod.lhs,
                rhs: filtered_rhs,
            });
        }

        let mut productions = updated_productions;

        debug!(
            4,
            "Processing {} may-be-null terminals",
            may_be_null_terminals.len()
        );
        for terminal_name in &may_be_null_terminals {
            let opt_nt_name = Self::generate_unique_indexed_name(
                &format!("{}Opt", terminal_name.trim_matches('"')),
                &mut per_base_counters,
                &mut all_names,
            );
            let opt_nt = NonTerminal(opt_nt_name.clone());

            for prod in productions.iter_mut() {
                for sym in &mut prod.rhs {
                    if let Symbol::Terminal(Terminal::RegexName(t)) = sym {
                        if t == terminal_name {
                            *sym = Symbol::NonTerminal(opt_nt.clone());
                        }
                    }
                }
            }

            productions.push(Production {
                lhs: opt_nt.clone(),
                rhs: vec![Symbol::Terminal(regex_name(&terminal_name))],
            });
            productions.push(Production {
                lhs: opt_nt.clone(),
                rhs: Vec::new(),
            });
        }

        let mut def = GrammarDefinition {
            productions,
            start_production_id,
            literal_to_group_id,
            regex_name_to_group_id,
            group_id_to_expr,
            ignore_terminal_id: None,
            external_name_to_group_id: BiBTreeMap::new(),
        };

        // Set ignore terminal before optimization so it's preserved
        if let Some(ignore_name) = ignore_symbol_name {
            let group_id = def
                .regex_name_to_group_id
                .get_by_left(ignore_name)
                .ok_or_else(|| {
                    format!(
                        "Ignore symbol '{}' is not a defined terminal in the grammar.",
                        ignore_name
                    )
                })?;
            def.ignore_terminal_id = Some(TerminalID(*group_id));
        }

        def.optimize();

        Ok(def)
    }

    /// Constructs a `GrammarDefinition` from parsed grammar rules.
    /// This is the common implementation used by both `from_ebnf` and `from_lark`.
    fn from_parsed_rules(
        grammar_exprs: Vec<(String, GrammarExpr)>,
        ignore_symbol_name: Option<String>,
    ) -> Result<Self, String> {
        fn is_terminal_name(name: &str) -> bool {
            name.chars().next().map_or(false, |c| c.is_uppercase())
        }

        let mut terminals: BTreeMap<String, GrammarExpr> = BTreeMap::new();
        for (name, expr) in &grammar_exprs {
            if is_terminal_name(name) {
                terminals.insert(name.clone(), expr.clone());
            }
        }

        fn gather_referenced_terminals(
            expr: &GrammarExpr,
            terminals: &BTreeMap<String, GrammarExpr>,
            referenced_terminals: &mut HashSet<String>,
        ) {
            match expr {
                GrammarExpr::AnyChar => {}
                GrammarExpr::CharClass(_) => {}
                GrammarExpr::Literal(_) => {}
                GrammarExpr::Ref(name) => {
                    if terminals.contains_key(name) {
                        referenced_terminals.insert(name.clone());
                    }
                }
                GrammarExpr::Sequence(exprs) | GrammarExpr::Choice(exprs) => {
                    for e in exprs {
                        gather_referenced_terminals(e, terminals, referenced_terminals);
                    }
                }
                GrammarExpr::Optional(expr_box) | GrammarExpr::Repeat(expr_box) => {
                    gather_referenced_terminals(&*expr_box, terminals, referenced_terminals);
                }
            }
        }

        let mut referenced_terminals = HashSet::new();
        for (name, expr) in grammar_exprs.iter() {
            if !is_terminal_name(name) {
                gather_referenced_terminals(expr, &terminals, &mut referenced_terminals);
            }
        }
        if let Some(ignore_name) = &ignore_symbol_name {
            if is_terminal_name(ignore_name) {
                referenced_terminals.insert(ignore_name.clone());
            }
        }

        let non_terminal_rules: Vec<(String, GrammarExpr)> = grammar_exprs
            .into_iter()
            .filter(|(name, _)| !is_terminal_name(name))
            .collect();

        let terminal_defs: Vec<(String, Expr)> = terminals
            .clone()
            .into_iter()
            .filter(|(name, _)| referenced_terminals.contains(name))
            .map(|(name, grammar_expr)| {
                let mut memo = BTreeMap::new();
                let mut resolving_stack = HashSet::new();
                let regex_expr = Self::convert_grammar_expr_to_regex_expr(
                    &grammar_expr,
                    &terminals,
                    &mut memo,
                    &mut resolving_stack,
                )
                .unwrap();
                (name, regex_expr)
            })
            .collect();

        let grammar_def = GrammarDefinition::from_exprs_with_ignore(
            non_terminal_rules, 
            terminal_defs,
            ignore_symbol_name.as_deref(),
        )?;

        Ok(grammar_def)
    }

    /// Constructs a `GrammarDefinition` from an EBNF string.
    /// 
    /// EBNF format uses `::=` for rule definitions and `;` terminators:
    /// ```text
    /// rule ::= expr;
    /// ```
    pub fn from_ebnf(ebnf_source: &str) -> Result<Self, String> {
        let ebnf = EbnfParser::new(ebnf_source).and_then(|mut p| p.parse())?;
        Self::from_parsed_rules(ebnf.grammar_rules, ebnf.ignore_symbol_name)
    }

    /// Constructs a `GrammarDefinition` from a Lark grammar string.
    /// 
    /// Lark format uses `:` for rule definitions and newlines as terminators:
    /// ```text
    /// rule: expr
    /// ```
    pub fn from_lark(lark_source: &str) -> Result<Self, String> {
        let lark = LarkParser::new(lark_source).and_then(|mut p| p.parse())?;
        Self::from_parsed_rules(lark.grammar_rules, lark.ignore_symbol_name)
    }

    /// Constructs a `GrammarDefinition` from an EBNF file.
    pub fn from_ebnf_file(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read EBNF file '{}': {}", path, e))?;
        Self::from_ebnf(&content)
            .map_err(|e| format!("Failed to parse EBNF file '{}':\n{}", path, e))
    }

    /// Constructs a `GrammarDefinition` from a Lark grammar file.
    pub fn from_lark_file(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read Lark file '{}': {}", path, e))?;
        Self::from_lark(&content)
            .map_err(|e| format!("Failed to parse Lark file '{}':\n{}", path, e))
    }

    /// Helper to get terminal expressions ordered by group ID for tokenizer construction.
    pub fn get_terminal_expressions_for_tokenizer(&self) -> Vec<ExprGroup> {
        if self.group_id_to_expr.is_empty() {
            return Vec::new();
        }

        let max_group_id = *self.group_id_to_expr.keys().max().unwrap_or(&0);
        let mut expr_groups_vec: Vec<ExprGroup> =
            vec![greedy_group(Expr::Epsilon); max_group_id + 1];

        for (group_id, expr) in &self.group_id_to_expr {
            if *group_id < expr_groups_vec.len() {
                expr_groups_vec[*group_id] = greedy_group(expr.clone());
            } else {
                debug!(
                    0,
                    "Warning: Group ID {} is out of bounds for tokenizer expressions vector (len {}). Terminal {:?} might be missing.",
                    group_id,
                    expr_groups_vec.len(),
                    expr
                );
            }
        }
        expr_groups_vec
    }
}

// --- CompiledGrammar: Grammar with compiled tokenizer and parser ---
#[derive(Clone)]
pub struct CompiledGrammar {
    pub definition: Arc<GrammarDefinition>,
    pub tokenizer: Regex,
    pub glr_parser: GLRParser,
}

/// Intermediate type for CompiledGrammar JSON serialization
#[derive(JSONConvertible)]
struct CompiledGrammarJSON {
    definition: GrammarDefinition,
    tokenizer: Regex,
    glr_parser: GLRParser,
}

impl CompiledGrammarJSON {
    fn from_compiled(c: &CompiledGrammar) -> Self {
        CompiledGrammarJSON {
            definition: (*c.definition).clone(),
            tokenizer: c.tokenizer.clone(),
            glr_parser: c.glr_parser.clone(),
        }
    }

    fn to_compiled(self) -> CompiledGrammar {
        CompiledGrammar {
            definition: Arc::new(self.definition),
            tokenizer: self.tokenizer,
            glr_parser: self.glr_parser,
        }
    }
}

impl JSONConvertible for CompiledGrammar {
    fn to_json(&self) -> JSONNode {
        CompiledGrammarJSON::from_compiled(self).to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        CompiledGrammarJSON::from_json(node).map(|c| c.to_compiled())
    }
}

impl CompiledGrammar {
    /// Creates a `CompiledGrammar` from an `Arc<GrammarDefinition>`.
    pub fn from_definition(definition: Arc<GrammarDefinition>) -> Self {
        debug!(3, "Building tokenizer from definition");
        let terminal_expr_groups = definition.get_terminal_expressions_for_tokenizer();
        let tokenizer_expr_groups_obj = groups(terminal_expr_groups);
        let tokenizer = tokenizer_expr_groups_obj.build();

        debug!(3, "Building GLR parser from definition");
        let mut terminal_map: BiBTreeMap<Terminal, TerminalID> =
            definition
                .regex_name_to_group_id
                .iter()
                .map(|(name, group_id)| {
                    (Terminal::RegexName(name.clone()), TerminalID(*group_id))
                })
                .collect();
        for (val_bytes, group_id) in &definition.literal_to_group_id {
            terminal_map.insert(
                Terminal::Literal(val_bytes.clone()),
                TerminalID(*group_id),
            );
        }
        for (name, group_id) in &definition.external_name_to_group_id {
            terminal_map.insert(
                Terminal::RegexName(name.clone()),
                TerminalID(*group_id),
            );
        }
        let glr_parser = generate_glr_parser_with_terminal_map(
            &definition.productions,
            terminal_map,
            definition.ignore_terminal_id,
        );

        Self {
            definition,
            tokenizer,
            glr_parser,
        }
    }

    // Accessors
    pub fn productions(&self) -> &Vec<Production> {
        &self.definition.productions
    }
    pub fn start_production_id(&self) -> usize {
        self.definition.start_production_id
    }
    pub fn regex_name_to_group_id(&self) -> &BiBTreeMap<String, usize> {
        &self.definition.regex_name_to_group_id
    }
    pub fn tokenizer(&self) -> &Regex {
        &self.tokenizer
    }
    pub fn glr_parser(&self) -> &GLRParser {
        &self.glr_parser
    }
}

impl Display for CompiledGrammar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CompiledGrammar:")?;
        writeln!(
            f,
            "  Definition (Arc<GrammarDefinition>):"
        )?;
        writeln!(
            f,
            "    Start Production ID: {}",
            self.definition.start_production_id
        )?;
        writeln!(
            f,
            "  Productions ({}):",
            self.definition.productions.len()
        )?;
        for production in &self.definition.productions {
            write!(f, "      {} -> ", production.lhs.0)?;
            for (i, symbol) in production.rhs.iter().enumerate() {
                match symbol {
                    Symbol::Terminal(terminal) => write!(f, "{}", terminal)?,
                    Symbol::NonTerminal(non_terminal) => write!(f, "{}", non_terminal.0)?,
                }
                if i < production.rhs.len() - 1 {
                    write!(f, " ")?;
                }
            }
            writeln!(f)?;
        }
        writeln!(
            f,
            "    Terminals (Name to GroupID, {}):",
            self.definition.regex_name_to_group_id.len()
        )?;
        let mut terminals_sorted: Vec<_> =
            self.definition.regex_name_to_group_id.iter().collect();
        terminals_sorted.sort_by_key(|&(_, group_id)| group_id);
        for (name, group_id) in terminals_sorted {
            writeln!(f, "      {}: {:?}", name, group_id)?;
        }

        writeln!(
            f,
            "  Tokenizer (States: {}): {}",
            self.tokenizer.dfa.states.len(),
            &self.tokenizer.dfa
        )?;
        writeln!(
            f,
            "  GLR Parser (States: {}): {}",
            self.glr_parser.table.len(),
            &self.glr_parser
        )?;
        Ok(())
    }
}

// --- Incremental Parser ---
use crate::glr::parser::GLRParserState;
use crate::tokenizer::{ExecuteResult, LLMTokenID, TokenizerStateID};

#[derive(Clone)]
pub struct IncrementalParser<'a> {
    grammar: &'a CompiledGrammar,
    pub(crate) state: BTreeMap<TokenizerStateID, GLRParserState<'a>>,
}

impl<'a> IncrementalParser<'a> {
    pub fn new(grammar: &'a CompiledGrammar) -> Self {
        let initial_glr_state = grammar.glr_parser().init_glr_parser(None);
        let initial_tokenizer_state = grammar.tokenizer().initial_state_id();
        let state = BTreeMap::from([(initial_tokenizer_state, initial_glr_state)]);
        Self { grammar, state }
    }

    pub fn feed(&mut self, bytes: &[u8]) {
        crate::debug!(
            3,
            "Processing input bytes: {:?} with {} active tokenizer states",
            bytes,
            self.state.len()
        );
        let mut next_states: BTreeMap<TokenizerStateID, GLRParserState<'a>> = BTreeMap::new();
        let mut queue: BTreeMap<(usize, TokenizerStateID), GLRParserState<'a>> = BTreeMap::new();

        for (tokenizer_state_id, glr_state) in std::mem::take(&mut self.state) {
            queue.insert((0, tokenizer_state_id), glr_state);
        }

        while let Some(((position, current_tokenizer_state_id), current_glr_state)) =
            queue.pop_first()
        {
            let results: ExecuteResult = self
                .grammar
                .tokenizer()
                .execute_from_state(&bytes[position..], current_tokenizer_state_id);

            crate::debug!(
                4,
                "Processing position {} in state {}. Matches: {}",
                position,
                current_tokenizer_state_id.0,
                results.matches.len()
            );
            for token in results.matches {
                crate::debug!(
                    4,
                    "Found match for token {:?} ({}) with width {}",
                    token.id,
                    self.grammar
                        .definition
                        .regex_name_to_group_id
                        .get_by_right(&token.id)
                        .unwrap_or(&"UNKNOWN_TOKEN_NAME".to_string()),
                    token.width
                );
                let grammar_token_id = TerminalID(token.id);
                let mut next_glr_state = current_glr_state.clone();
                next_glr_state.step(grammar_token_id);

                if next_glr_state.is_ok() {
                    if position + token.width == bytes.len() {
                        let next_tokenizer_state_id =
                            self.grammar.tokenizer().initial_state_id();
                        next_states
                            .entry(next_tokenizer_state_id)
                            .and_modify(|existing_state| {
                                existing_state.merge_with(next_glr_state.clone())
                            })
                            .or_insert(next_glr_state.clone());
                    } else {
                        let next_tokenizer_state_id =
                            self.grammar.tokenizer().initial_state_id();
                        queue
                            .entry((position + token.width, next_tokenizer_state_id))
                            .and_modify(|existing_state| {
                                existing_state.merge_with(next_glr_state.clone())
                            })
                            .or_insert(next_glr_state);
                    }
                }
            }

            if let Some(end_state_id) = results.end_state {
                let possible_final_grammar_tokens: BTreeSet<_> = self
                    .grammar
                    .tokenizer()
                    .tokens_accessible_from_state(TokenizerStateID(end_state_id));
                for possible_final_grammar_token in possible_final_grammar_tokens {
                    let mut final_glr_state = current_glr_state.clone();
                    final_glr_state.step(possible_final_grammar_token);
                    if final_glr_state.is_ok() {
                        let next_tokenizer_state_id = TokenizerStateID(end_state_id);
                        next_states
                            .entry(next_tokenizer_state_id)
                            .and_modify(|existing_state| {
                                existing_state.merge_with(current_glr_state.clone())
                            })
                            .or_insert(current_glr_state.clone());
                    }
                }
            }
        }
        self.state = next_states;
    }

    pub fn is_valid(&self) -> bool {
        self.state.values().any(|glr_state| glr_state.is_ok())
    }
}
