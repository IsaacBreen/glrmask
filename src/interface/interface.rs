use crate::constraint::GrammarConstraint;
use crate::debug;
use crate::finite_automata::{greedy_group, groups, Expr, ExprGroup, GroupID, QuantifierType, Regex};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::parser::GLRParser;
use crate::glr::table::{assign_non_terminal_ids, generate_glr_parser, generate_glr_parser_with_maps, generate_glr_parser_with_terminal_map, NonTerminalID, TerminalID};
use crate::interface::ebnf::{EbnfParseResult, EbnfParser};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::types::TerminalID as GrammarTokenID;
use crate::datastructures::u8set::U8Set;
use crate::glr::analyze::simplify_grammar;
use crate::glr::grammar::regex_name;
// May not be used directly here anymore
use bimap::BiBTreeMap;
use kdam::tqdm;
use std::collections::BTreeMap as StdMap;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use std::fs;

type LLMToken<'a> = &'a [u8];
type LLMTokenMap = BiBTreeMap<Vec<u8>, LLMTokenID>;

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

impl JSONConvertible for GrammarExpr {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        match self {
            GrammarExpr::Ref(name) => {
                obj.insert("variant".to_string(), JSONNode::String("Ref".to_string()));
                obj.insert("name".to_string(), name.to_json());
            }
            GrammarExpr::Sequence(exprs) => {
                obj.insert("variant".to_string(), JSONNode::String("Sequence".to_string()));
                obj.insert("exprs".to_string(), exprs.to_json());
            }
            GrammarExpr::Choice(exprs) => {
                obj.insert("variant".to_string(), JSONNode::String("Choice".to_string()));
                obj.insert("exprs".to_string(), exprs.to_json());
            }
            GrammarExpr::Optional(expr_box) => {
                obj.insert("variant".to_string(), JSONNode::String("Optional".to_string()));
                obj.insert("expr".to_string(), expr_box.to_json());
            }
            GrammarExpr::Repeat(expr_box) => {
                obj.insert("variant".to_string(), JSONNode::String("Repeat".to_string()));
                obj.insert("expr".to_string(), expr_box.to_json());
            }
            GrammarExpr::Literal(bytes) => {
                obj.insert("variant".to_string(), JSONNode::String("Literal".to_string()));
                obj.insert("bytes".to_string(), bytes.to_json());
            }
            GrammarExpr::CharClass(s) => {
                obj.insert("variant".to_string(), JSONNode::String("CharClass".to_string()));
                obj.insert("def".to_string(), s.to_json());
            }
            GrammarExpr::AnyChar => {
                obj.insert("variant".to_string(), JSONNode::String("AnyChar".to_string()));
            }
        }
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let variant = obj
                    .remove("variant")
                    .ok_or_else(|| "Missing field variant for GrammarExpr".to_string())
                    .and_then(String::from_json)?;
                match variant.as_str() {
                    "Ref" => {
                        let name = obj
                            .remove("name")
                            .ok_or_else(|| "Missing field name for Ref".to_string())
                            .and_then(String::from_json)?;
                        Ok(GrammarExpr::Ref(name))
                    }
                    "Sequence" => {
                        let exprs = obj
                            .remove("exprs")
                            .ok_or_else(|| "Missing field exprs for Sequence".to_string())
                            .and_then(Vec::<GrammarExpr>::from_json)?;
                        Ok(GrammarExpr::Sequence(exprs))
                    }
                    "Choice" => {
                        let exprs = obj
                            .remove("exprs")
                            .ok_or_else(|| "Missing field exprs for Choice".to_string())
                            .and_then(Vec::<GrammarExpr>::from_json)?;
                        Ok(GrammarExpr::Choice(exprs))
                    }
                    "Optional" => {
                        let expr_node = obj
                            .remove("expr")
                            .ok_or_else(|| "Missing field expr for Optional".to_string())?;
                        Ok(GrammarExpr::Optional(Box::new(GrammarExpr::from_json(
                            expr_node,
                        )?)))
                    }
                    "Repeat" => {
                        let expr_node = obj
                            .remove("expr")
                            .ok_or_else(|| "Missing field expr for Repeat".to_string())?;
                        Ok(GrammarExpr::Repeat(Box::new(GrammarExpr::from_json(
                            expr_node,
                        )?)))
                    }
                    "Literal" => {
                        let bytes = obj
                            .remove("bytes")
                            .ok_or_else(|| "Missing field bytes for Literal".to_string())
                            .and_then(Vec::<u8>::from_json)?;
                        Ok(GrammarExpr::Literal(bytes))
                    }
                    "CharClass" => {
                        let s = obj
                            .remove("def")
                            .ok_or_else(|| "Missing field def for CharClass".to_string())
                            .and_then(String::from_json)?;
                        Ok(GrammarExpr::CharClass(s))
                    }
                    "AnyChar" => Ok(GrammarExpr::AnyChar),
                    _ => Err(format!("Unknown variant {} for GrammarExpr", variant)),
                }
            }
            _ => Err("Expected JSONNode::Object for GrammarExpr".to_string()),
        }
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
    pub fn optimize(&mut self) {
        crate::interface::optimization::optimize_grammar(self);
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
            disable = !PROGRESS_BAR_ENABLED,
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

        #[derive(Copy, Clone, PartialEq)]
        enum Nullability {
            NeverNull,
            CanBeNull,
            AlwaysNull,
        }

        fn get_nullability(expr: Expr) -> Nullability {
            fn _get_nullability(
                expr: Expr,
                cache: &mut HashMap<Arc<Expr>, Nullability>,
            ) -> Nullability {
                match expr {
                    Expr::U8Seq(bytes) => {
                        if bytes.is_empty() {
                            Nullability::AlwaysNull
                        } else {
                            Nullability::NeverNull
                        }
                    }
                    Expr::U8Class(_) => Nullability::NeverNull,
                    Expr::Quantifier(expr, q_type) => match q_type {
                        QuantifierType::ZeroOrMore => Nullability::CanBeNull,
                        QuantifierType::OneOrMore => _get_nullability(*expr, cache),
                        QuantifierType::ZeroOrOne => Nullability::CanBeNull,
                    },
                    Expr::Choice(exprs) => {
                        let nullabilities: Vec<Nullability> = exprs
                            .iter()
                            .map(|e| _get_nullability(e.clone(), cache))
                            .collect();
                        if nullabilities.iter().any(|n| {
                            matches!(n, Nullability::AlwaysNull | Nullability::CanBeNull)
                        }) {
                            Nullability::CanBeNull
                        } else {
                            Nullability::NeverNull
                        }
                    }
                    Expr::Seq(exprs) => {
                        let nullabilities: Vec<Nullability> = exprs
                            .iter()
                            .map(|e| _get_nullability(e.clone(), cache))
                            .collect();
                        if nullabilities
                            .iter()
                            .all(|n| matches!(n, Nullability::AlwaysNull | Nullability::CanBeNull))
                        {
                            Nullability::CanBeNull
                        } else if nullabilities
                            .iter()
                            .any(|n| *n == Nullability::NeverNull)
                        {
                            Nullability::NeverNull
                        } else {
                            Nullability::NeverNull
                        }
                    }
                    Expr::Epsilon => Nullability::AlwaysNull,
                    Expr::Shared(expr) => {
                        if let Some(&cached_nullability) = cache.get(&expr) {
                            cached_nullability
                        } else {
                            let nullability = _get_nullability(expr.as_ref().clone(), cache);
                            cache.insert(expr.clone(), nullability);
                            nullability
                        }
                    }
                }
            }
            let mut cache: HashMap<Arc<Expr>, Nullability> = HashMap::new();
            _get_nullability(expr, &mut cache)
        }

        // Nullability analysis for terminals:
        //   - always-null terminals are removed from RHSs;
        //   - sometimes-null terminals are desugared into optional non-terminals.
        let mut always_null_terminals: HashSet<String> = HashSet::new();
        let mut may_be_null_terminals: HashSet<String> = HashSet::new();

        for (terminal_name, group_id) in regex_name_to_group_id.iter() {
            let expr = group_id_to_expr
                .get(group_id)
                .expect("regex_name_to_group_id / group_id_to_expr out of sync")
                .clone();

            match get_nullability(expr) {
                Nullability::AlwaysNull => {
                    always_null_terminals.insert(terminal_name.clone());
                }
                Nullability::CanBeNull => {
                    may_be_null_terminals.insert(terminal_name.clone());
                }
                Nullability::NeverNull => {}
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

        Ok(GrammarDefinition {
            productions,
            start_production_id,
            literal_to_group_id,
            regex_name_to_group_id,
            group_id_to_expr,
            ignore_terminal_id: None,
            external_name_to_group_id: BiBTreeMap::new(),
        })
    }

    /// Constructs a `GrammarDefinition` from an EBNF string.
    pub fn from_ebnf(ebnf_source: &str) -> Result<Self, String> {
        let ebnf = EbnfParser::new(ebnf_source).and_then(|mut p| p.parse())?;
        let grammar_exprs = ebnf.grammar_rules;

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
        if let Some(ignore_name) = &ebnf.ignore_symbol_name {
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

        let mut grammar_def = GrammarDefinition::from_exprs(non_terminal_rules, terminal_defs)?;

        if let Some(ignore_name) = &ebnf.ignore_symbol_name {
            let group_id = grammar_def
                .regex_name_to_group_id
                .get_by_left(ignore_name)
                .ok_or_else(|| {
                    format!(
                        "Ignore symbol '{}' is not a defined terminal in the grammar.",
                        ignore_name
                    )
                })?;
            grammar_def.ignore_terminal_id = Some(TerminalID(*group_id));
        }

        Ok(grammar_def)
    }

    /// Constructs a `GrammarDefinition` from an EBNF file.
    pub fn from_ebnf_file(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read EBNF file '{}': {}", path, e))?;
        Self::from_ebnf(&content)
            .map_err(|e| format!("Failed to parse EBNF file '{}':\n{}", path, e))
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

impl JSONConvertible for CompiledGrammar {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("definition".to_string(), self.definition.to_json());
        obj.insert("tokenizer".to_string(), self.tokenizer.to_json());
        obj.insert("glr_parser".to_string(), self.glr_parser.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let definition_node = obj
                    .remove("definition")
                    .ok_or_else(|| "Missing field definition for CompiledGrammar".to_string())?;
                let definition = Arc::new(GrammarDefinition::from_json(definition_node)?);

                let tokenizer_node = obj
                    .remove("tokenizer")
                    .ok_or_else(|| "Missing field tokenizer for CompiledGrammar".to_string())?;
                let tokenizer = Regex::from_json(tokenizer_node)?;

                let glr_parser_node = obj
                    .remove("glr_parser")
                    .ok_or_else(|| "Missing field glr_parser for CompiledGrammar".to_string())?;
                let glr_parser = GLRParser::from_json(glr_parser_node)?;

                Ok(CompiledGrammar {
                    definition,
                    tokenizer,
                    glr_parser,
                })
            }
            _ => Err("Expected JSONNode::Object for CompiledGrammar".to_string()),
        }
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
use crate::profiler::PROGRESS_BAR_ENABLED;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datastructures::hybrid_bitset::HybridBitset;
    use crate::finite_automata::{eat_u8, eat_u8_seq};
    use crate::interface::tokenizer_combinators::{
        eat_u8_fast, eat_u8_negation_fast, eat_u8_range_fast, repeat0_fast,
    };
    use crate::{choice_fast, groups, seq_fast};
    use bitvec::prelude::*;

    use crate::finite_automata::Expr as RegexExpr;
    use crate::glr::grammar::{NonTerminal as NT, Production as Prod, Symbol as Sym, Terminal};

    fn bitvec_with_capacity_and_values(capacity: usize, values: Vec<usize>) -> HybridBitset {
        let mut bitvec = BitVec::new();
        bitvec.resize(capacity, false);
        for value in values {
            if value < capacity {
                bitvec.set(value, true);
            }
        }
        bitvec.into()
    }

    #[test]
    fn test_precompute_for_python_name_token_with_names() {
        let ignore_expr = repeat0_fast(choice_fast!(
            eat_u8_fast(b' '),
            seq_fast!(
                eat_u8_fast(b'#'),
                repeat0_fast(eat_u8_negation_fast(b'\n')),
                eat_u8_fast(b'\n')
            )
        ));
        let digit_expr = eat_u8_range_fast(b'0', b'9');
        let alph_lower_expr = eat_u8_range_fast(b'a', b'z');
        let alph_upper_expr = eat_u8_range_fast(b'A', b'Z');
        let underscore_expr = eat_u8_fast(b'_');

        let name_start_expr =
            choice_fast!(alph_lower_expr.clone(), alph_upper_expr.clone(), underscore_expr.clone());
        let name_middle_expr = choice_fast!(name_start_expr.clone(), digit_expr.clone());
        let name_expr = seq_fast!(
            ignore_expr.clone(),
            name_start_expr.clone(),
            repeat0_fast(name_middle_expr.clone())
        );

        let tokenizer = groups![
            greedy_group(ignore_expr),
            greedy_group(digit_expr),
            greedy_group(alph_lower_expr),
            greedy_group(alph_upper_expr),
            greedy_group(underscore_expr),
            greedy_group(name_start_expr),
            greedy_group(name_middle_expr),
            greedy_group(name_expr)
        ]
        .build();

        let llm_tokens: Vec<Vec<u8>> =
            (0..2).map(|i| format!("abcdefghijk{}", i).as_bytes().to_vec()).collect();
        let llm_token_map: LLMTokenMap = llm_tokens
            .iter()
            .enumerate()
            .map(|(i, token)| (token.clone(), LLMTokenID(i)))
            .collect();
        let max_llm_token_id = llm_tokens.len();

        let mut regex_name_to_group_id = BiBTreeMap::new();
        regex_name_to_group_id.insert(regex_name(&"ignore"), 0);
        regex_name_to_group_id.insert(regex_name(&"digit"), 1);
        regex_name_to_group_id.insert(regex_name(&"alph_lower"), 2);
        regex_name_to_group_id.insert(regex_name(&"alph_upper"), 3);
        regex_name_to_group_id.insert(regex_name(&"underscore"), 4);
        regex_name_to_group_id.insert(regex_name(&"name_start"), 5);
        regex_name_to_group_id.insert(regex_name(&"name_middle"), 6);
        regex_name_to_group_id.insert(regex_name(&"name"), 7);

        let dummy_productions = vec![Prod {
            lhs: NT("S".to_string()),
            rhs: vec![],
        }];
        let dummy_glr_parser = generate_glr_parser(&dummy_productions, None);

        let constraint = GrammarConstraint::new(
            tokenizer,
            dummy_glr_parser,
            llm_token_map,
            regex_name_to_group_id,
            max_llm_token_id,
        );
        println!("Precomputation (implicitly done by GrammarConstraint::new) successful.");
    }

    #[test]
    fn test_incremental_parser_simple() {
        let terminals = vec![
            ("a".to_string(), eat_u8(b'a')),
            ("b".to_string(), eat_u8(b'b')),
            ("c".to_string(), eat_u8(b'c')),
        ];
        let rules = vec![(
            "S".to_string(),
            choice(vec![
                sequence(vec![crate::interface::r#ref("a"), crate::interface::r#ref("b")]),
                sequence(vec![crate::interface::r#ref("a"), crate::interface::r#ref("c")]),
            ]),
        )];
        let grammar_def = GrammarDefinition::from_exprs(rules, terminals).unwrap();
        let grammar = CompiledGrammar::from_definition(std::sync::Arc::new(grammar_def));
        let mut parser = IncrementalParser::new(&grammar);

        assert!(parser.is_valid());

        parser.feed(b"a");
        assert!(parser.is_valid());
        assert_eq!(parser.state.len(), 1, "Expected 1 state after feeding 'a'");
        assert!(parser
            .state
            .contains_key(&grammar.tokenizer().initial_state_id()));

        let mut parser_ab = parser.clone();
        parser_ab.feed(b"b");
        assert!(parser_ab.is_valid());

        let mut parser_ac = IncrementalParser::new(&grammar);
        parser_ac.feed(b"a");
        parser_ac.feed(b"c");
        assert!(parser_ac.is_valid());

        let mut parser_ad = IncrementalParser::new(&grammar);
        parser_ad.feed(b"a");
        parser_ad.feed(b"d");
        assert!(!parser_ad.is_valid());
    }

    #[test]
    fn test_minimal_python_example_with_compiled_grammar() {
        let terminals = vec![
            (
                "NUMBER".to_string(),
                crate::interface::tokenizer_combinators::repeat1_fast(
                    crate::interface::tokenizer_combinators::eat_u8_range_fast(b'0', b'9'),
                ),
            ),
            (
                "PLUS".to_string(),
                crate::interface::tokenizer_combinators::eat_u8_fast(b'+'),
            ),
        ];

        let rules = vec![(
            "S".to_string(),
            sequence(vec![
                crate::interface::r#ref("NUMBER"),
                crate::interface::r#ref("PLUS"),
                crate::interface::r#ref("NUMBER"),
                crate::interface::r#ref("PLUS"),
                crate::interface::r#ref("NUMBER"),
            ]),
        )];

        println!("Building grammar...");
        let grammar_def = GrammarDefinition::from_exprs(rules, terminals).unwrap();
        let compiled_grammar =
            CompiledGrammar::from_definition(std::sync::Arc::new(grammar_def));

        let mut llm_token_map = bimap::BiBTreeMap::new();
        let mut llm_tokens_vec: Vec<Vec<u8>> = Vec::new();
        for i in 0..=9 {
            let digit_byte = b'0' + i;
            let token = vec![digit_byte];
            llm_token_map.insert(token.clone(), LLMTokenID(i as usize));
            llm_tokens_vec.push(token);
        }
        let plus_token = vec![b'+'];
        let plus_token_id = 10usize;
        llm_token_map.insert(plus_token.clone(), LLMTokenID(plus_token_id));
        llm_tokens_vec.push(plus_token);

        let max_llm_token_id = plus_token_id + 1;
        let eof_llm_token_id = max_llm_token_id;

        println!("Creating constraint...");
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar,
            llm_token_map.clone(),
            max_llm_token_id,
        );

        println!("Initializing state...");
        let mut state = grammar_constraint.init();

        let input_token_ids = vec![
            LLMTokenID(1),
            LLMTokenID(2),
            LLMTokenID(3),
            LLMTokenID(10),
            LLMTokenID(4),
            LLMTokenID(5),
            LLMTokenID(6),
            LLMTokenID(10),
        ];

        println!("Committing tokens...");
        for token_id in input_token_ids {
            assert!(
                state.get_mask().contains(token_id.0),
                "Token ID {} not in mask. Mask: {:?}",
                token_id.0,
                state.get_mask().iter_bits().collect::<Vec<_>>()
            );
            state.commit(token_id);
        }

        println!("Getting final mask...");
        let final_mask = state.get_mask();

        for i in 0..=9 {
            assert!(
                final_mask.contains(i),
                "Expected digit '{}' (LLM Token ID {}) to be allowed. Mask: {:?}",
                (b'0' + i as u8) as char,
                i,
                final_mask.iter_bits().collect::<Vec<_>>()
            );
        }
        assert!(
            !final_mask.contains(plus_token_id),
            "Expected '+' (LLM Token ID {}) to be disallowed. Mask: {:?}",
            plus_token_id,
            final_mask.iter_bits().collect::<Vec<_>>()
        );
        if final_mask.len() > eof_llm_token_id {
            assert!(
                !final_mask.contains(eof_llm_token_id),
                "Expected EOF (ID {}) to be disallowed at this stage",
                eof_llm_token_id
            );
        }

        println!("Final mask check passed.");
    }

    #[test]
    fn test_sentence_grammar_from_prompt() {
        let terminals = vec![
            ("a".to_string(), eat_u8(b'a')),
            ("the".to_string(), eat_u8_seq(b"the".to_vec())),
            ("apple".to_string(), eat_u8_seq(b"apple".to_vec())),
            ("banana".to_string(), eat_u8_seq(b"banana".to_vec())),
            ("person".to_string(), eat_u8_seq(b"person".to_vec())),
            (" ".to_string(), eat_u8(b' ')),
            ("eats".to_string(), eat_u8_seq(b"eats".to_vec())),
            ("likes".to_string(), eat_u8_seq(b"likes".to_vec())),
            ("is".to_string(), eat_u8_seq(b"is".to_vec())),
            ("tasty".to_string(), eat_u8_seq(b"tasty".to_vec())),
            ("red".to_string(), eat_u8_seq(b"red".to_vec())),
            ("happy".to_string(), eat_u8_seq(b"happy".to_vec())),
            (".".to_string(), eat_u8(b'.')),
            ("and".to_string(), eat_u8_seq(b"and".to_vec())),
        ];

        let expr_A = choice(vec![
            crate::interface::r#ref("a"),
            crate::interface::r#ref("the"),
            crate::interface::r#ref("apple"),
            crate::interface::r#ref("banana"),
            crate::interface::r#ref("person"),
        ]);
        let expr_IGNORE = crate::interface::r#ref(" ");
        let expr_B = choice(vec![
            crate::interface::r#ref("eats"),
            crate::interface::r#ref("likes"),
            crate::interface::r#ref("is"),
            crate::interface::r#ref("tasty"),
            crate::interface::r#ref("red"),
            crate::interface::r#ref("happy"),
            crate::interface::r#ref("."), //
            crate::interface::r#ref("and"),
        ]);

        let expr_start = sequence(vec![
            crate::interface::r#ref("A"),
            crate::interface::r#ref("IGNORE"),
            crate::interface::r#ref("B"),
        ]);

        let grammar_exprs = vec![
            ("start".to_string(), expr_start),
            ("A".to_string(), expr_A),
            ("IGNORE".to_string(), expr_IGNORE),
            ("B".to_string(), expr_B),
        ];

        println!("Building grammar for sentence test...");
        let grammar_def =
            GrammarDefinition::from_exprs(grammar_exprs, terminals).expect("Failed to create grammar definition");
        let compiled_grammar =
            CompiledGrammar::from_definition(std::sync::Arc::new(grammar_def));
        println!("{}", compiled_grammar);

        let mut llm_token_map = bimap::BiBTreeMap::new();
        let mut next_llm_id_val = 0;

        let mut add_token = |s: &str| {
            let token_bytes = s.as_bytes().to_vec();
            if let Some(existing_id) = llm_token_map.get_by_left(&token_bytes) {
                return *existing_id;
            }
            let id = LLMTokenID(next_llm_id_val);
            llm_token_map.insert(token_bytes, id);
            next_llm_id_val += 1;
            id
        };

        let tok_a = add_token("a");
        let tok_the = add_token("the");
        let tok_apple = add_token("apple");
        let tok_banana = add_token("banana");
        let tok_person = add_token("person");

        let tok_space = add_token(" ");

        let tok_eats = add_token("eats");
        let tok_likes = add_token("likes");
        let tok_is = add_token("is");
        let tok_tasty = add_token("tasty");
        let tok_red = add_token("red");
        let tok_happy = add_token("happy");
        let tok_dot = add_token(".");
        let tok_and = add_token("and");

        let tok_e = add_token("e");
        let tok_eth = add_token("eth");

        let max_original_llm_token_id =
            if next_llm_id_val == 0 { 0 } else { next_llm_id_val - 1 };

        let eof_llm_token_id = LLMTokenID(next_llm_id_val);

        let ids_to_mask = |ids: &[LLMTokenID]| -> HybridBitset {
            let mut bs = HybridBitset::zeros();
            for id in ids {
                bs.insert(id.0);
            }
            bs
        };

        println!("Creating constraint for sentence test...");
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar,
            llm_token_map.clone(),
            max_original_llm_token_id,
        );

        println!("Initializing state for sentence test...");
        let mut state = grammar_constraint.init();

        let mut expected_A_tokens = vec![tok_a, tok_the, tok_apple, tok_banana, tok_person];
        let mut current_mask = state.get_mask();
        assert_eq!(
            current_mask,
            ids_to_mask(&expected_A_tokens),
            "Initial mask should allow tokens for A"
        );

        state.commit(tok_apple);
        current_mask = state.get_mask();
        let expected_IGNORE_tokens = vec![tok_space];
        assert_eq!(
            current_mask,
            ids_to_mask(&expected_IGNORE_tokens),
            "Mask after 'apple' should allow token for IGNORE (' ')"
        );

        state.commit(tok_space);
        current_mask = state.get_mask();
        let mut expected_B_tokens = vec![
            tok_a, tok_eats, tok_likes, tok_is, tok_tasty, tok_red, tok_happy, tok_dot, tok_and,
            tok_e,
        ];
        assert_eq!(
            current_mask,
            ids_to_mask(&expected_B_tokens),
            "Mask after 'apple ' should allow tokens for B"
        );

        state.commit(tok_eats);
        current_mask = state.get_mask();
        let mut expected_eof_mask = HybridBitset::zeros();
        assert_eq!(current_mask, expected_eof_mask);

        println!("Sentence grammar test completed successfully.");
    }

    #[test]
    fn test_sentence_grammar_from_prompt_simplified() {
        let terminals = vec![
            ("A_T".to_string(), eat_u8_seq(b"ab".to_vec())),
            ("B_T".to_string(), eat_u8_seq(b"bc".to_vec())),
        ];

        let expr_A = crate::interface::r#ref("A_T");
        let expr_B = crate::interface::r#ref("B_T");

        let expr_start = sequence(vec![
            crate::interface::r#ref("A"),
            crate::interface::r#ref("B"),
        ]);

        let grammar_exprs = vec![
            ("start".to_string(), expr_start),
            ("A".to_string(), expr_A),
            ("B".to_string(), expr_B),
        ];

        println!("Building grammar for sentence test...");
        let grammar_def =
            GrammarDefinition::from_exprs(grammar_exprs, terminals).expect("Failed to create grammar definition");
        let compiled_grammar =
            CompiledGrammar::from_definition(std::sync::Arc::new(grammar_def));
        println!("{}", compiled_grammar);

        let mut llm_token_map = bimap::BiBTreeMap::new();
        let mut next_llm_id_val = 0;

        let mut add_token = |s: &str| {
            let token_bytes = s.as_bytes().to_vec();
            if let Some(existing_id) = llm_token_map.get_by_left(&token_bytes) {
                return *existing_id;
            }
            let id = LLMTokenID(next_llm_id_val);
            llm_token_map.insert(token_bytes, id);
            next_llm_id_val += 1;
            id
        };

        let tok_b = add_token("b");

        let max_original_llm_token_id =
            if next_llm_id_val == 0 { 0 } else { next_llm_id_val - 1 };

        let eof_llm_token_id = LLMTokenID(next_llm_id_val);

        let ids_to_mask = |ids: &[LLMTokenID]| -> HybridBitset {
            let mut bs = HybridBitset::zeros();
            for id in ids {
                bs.insert(id.0);
            }
            bs
        };

        println!("Creating constraint for sentence test...");
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar,
            llm_token_map.clone(),
            max_original_llm_token_id,
        );

        println!("Initializing state for sentence test...");
        let mut state = grammar_constraint.init();

        let expected_A_tokens = vec![];
        let current_mask = state.get_mask();
        assert_eq!(
            current_mask,
            ids_to_mask(&expected_A_tokens),
            "Initial mask should allow tokens for A"
        );
    }

    #[test]
    fn test_python_reported_bug_def_rep_space_f() {
        let terminals = vec![
            ("SPACE".to_string(), eat_u8(b' ')),
            ("F".to_string(), eat_u8(b'f')),
        ];
        let start_expr = sequence(vec![
            repeat(crate::interface::r#ref("SPACE")),
            crate::interface::r#ref("F"),
        ]);
        let exprs = vec![("start".to_string(), start_expr)];
        let grammar_def =
            GrammarDefinition::from_exprs(exprs, terminals).expect("Failed to create grammar definition");
        let compiled_grammar =
            CompiledGrammar::from_definition(std::sync::Arc::new(grammar_def));
        println!("Compiled Grammar: {}", compiled_grammar);

        let mut llm_token_map = BiBTreeMap::new();
        let tok_space_id = LLMTokenID(0);
        let tok_f_space_id = LLMTokenID(1);

        llm_token_map.insert(b" ".to_vec(), tok_space_id);
        llm_token_map.insert(b" f".to_vec(), tok_f_space_id);

        let max_original_llm_token_id = 2;
        let dummy_eof_placeholder = 0;

        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar,
            llm_token_map.clone(),
            max_original_llm_token_id,
        );
        let mut state = grammar_constraint.init();

        let initial_mask = state.get_mask();

        assert!(
            initial_mask.contains(tok_f_space_id.0),
            "BUG REPLICATION: Initial mask should contain ' f' (ID {}), but it does not. Mask: {:?}",
            tok_f_space_id.0,
            &initial_mask
        );

        assert!(
            initial_mask.contains(tok_space_id.0),
            "Initial mask should contain ' ' (ID {}). Mask: {:?}",
            tok_space_id.0,
            &initial_mask
        );
    }

    #[test]
    fn test_nullability_handling_in_from_exprs() {
        let terminals = vec![
            (
                "X_OPT".to_string(),
                RegexExpr::Quantifier(Box::new(eat_u8(b'x')), QuantifierType::ZeroOrOne),
            ),
            ("EPS".to_string(), RegexExpr::Epsilon),
            ("Z".to_string(), eat_u8(b'z')),
        ];
        let rules = vec![(
            "Root".to_string(),
            sequence(vec![
                crate::interface::r#ref("X_OPT"),
                crate::interface::r#ref("EPS"),
                crate::interface::r#ref("Z"),
            ]),
        )];

        let grammar_def =
            GrammarDefinition::from_exprs(rules, terminals).expect("Failed to create GrammarDefinition");

        use crate::glr::grammar::regex_name;

        let name_term_x_opt = "X_OPT".to_string();
        let _term_x_opt_gid = *grammar_def
            .regex_name_to_group_id
            .get_by_left(&name_term_x_opt)
            .unwrap_or_else(|| {
                panic!(
                    "Could not find group ID for sometimes-null terminal name: {}",
                    name_term_x_opt
                )
            });

        let name_term_eps = "EPS".to_string();
        let _term_eps_gid = *grammar_def
            .regex_name_to_group_id
            .get_by_left(&name_term_eps)
            .unwrap_or_else(|| {
                panic!(
                    "Could not find group ID for always-null terminal name: {}",
                    name_term_eps
                )
            });

        let name_term_z = "Z".to_string();
        let _term_z_gid = *grammar_def
            .regex_name_to_group_id
            .get_by_left(&name_term_z)
            .unwrap_or_else(|| {
                panic!(
                    "Could not find group ID for never-null terminal name: {}",
                    name_term_z
                )
            });

        let mut nt_optional_term_x_opt_name = "".to_string();
        let mut found_prod_to_terminal = false;
        let mut found_prod_to_epsilon = false;

        for prod in &grammar_def.productions {
            if prod.rhs.len() == 1 {
                if let Sym::Terminal(Terminal::RegexName(t)) = &prod.rhs[0] {
                    if t == &name_term_x_opt {
                        let candidate_nt_name = prod.lhs.0.clone();
                        if grammar_def.productions.iter().any(|p| {
                            p.lhs.0 == candidate_nt_name && p.rhs.is_empty()
                        }) {
                            nt_optional_term_x_opt_name = candidate_nt_name;
                            found_prod_to_terminal = true;
                            break;
                        }
                    }
                }
            }
        }

        if !nt_optional_term_x_opt_name.is_empty() {
            if grammar_def.productions.iter().any(|p| {
                p.lhs.0 == nt_optional_term_x_opt_name && p.rhs.is_empty()
            }) {
                found_prod_to_epsilon = true;
            }
        }

        assert!(
            found_prod_to_terminal,
            "Could not find production NT -> {} for the optional NT",
            name_term_x_opt
        );
        assert!(
            found_prod_to_epsilon,
            "Could not find production {} -> epsilon for the optional NT",
            nt_optional_term_x_opt_name
        );
        assert!(
            !nt_optional_term_x_opt_name.is_empty(),
            "Could not find the generated optional NT for {}",
            name_term_x_opt
        );

        let augmented_start_nt_name =
            grammar_def.productions[grammar_def.start_production_id].lhs.0.clone();

        let expected_prods_set = BTreeSet::from([
            Prod {
                lhs: NT(augmented_start_nt_name),
                rhs: vec![Sym::NonTerminal(NT("Root".to_string()))],
            },
            Prod {
                lhs: NT("Root".to_string()),
                rhs: vec![
                    Sym::NonTerminal(NT(nt_optional_term_x_opt_name.clone())),
                    Sym::Terminal(regex_name(&name_term_z)),
                ],
            },
            Prod {
                lhs: NT(nt_optional_term_x_opt_name.clone()),
                rhs: vec![Sym::Terminal(regex_name(&name_term_x_opt))],
            },
            Prod {
                lhs: NT(nt_optional_term_x_opt_name.clone()),
                rhs: vec![],
            },
        ]);

        let actual_prods_set: BTreeSet<_> =
            grammar_def.productions.iter().cloned().collect();

        if expected_prods_set != actual_prods_set {
            println!(
                "Expected productions ({}) vs Actual productions ({})",
                expected_prods_set.len(),
                actual_prods_set.len()
            );
            println!("Expected (not found in actual):");
            for p in expected_prods_set.difference(&actual_prods_set) {
                println!(
                    "  {} -> {}",
                    p.lhs.0,
                    p.rhs
                        .iter()
                        .map(|s| match s {
                            Sym::Terminal(t) => t.to_string(),
                            Sym::NonTerminal(nt) => nt.to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
            println!("Actual (not found in expected):");
            for p in actual_prods_set.difference(&expected_prods_set) {
                println!(
                    "  {} -> {}",
                    p.lhs.0,
                    p.rhs
                        .iter()
                        .map(|s| match s {
                            Sym::Terminal(t) => t.to_string(),
                            Sym::NonTerminal(nt) => nt.to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
        }

        assert_eq!(
            actual_prods_set.len(),
            expected_prods_set.len(),
            "Number of productions mismatch"
        );
        assert_eq!(
            actual_prods_set, expected_prods_set,
            "Production sets do not match"
        );

        for prod in &grammar_def.productions {
            for sym in &prod.rhs {
                if let Sym::Terminal(t) = sym {
                    assert_ne!(
                        t,
                        &regex_name(&name_term_eps),
                        "Always-null terminal '{}' should not appear in the RHS of any final production (found in {} -> ...)",
                        name_term_eps,
                        prod.lhs.0
                    );
                }
            }
        }
    }
}
