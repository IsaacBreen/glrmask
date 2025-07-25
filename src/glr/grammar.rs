use std::collections::{BTreeMap, BTreeSet};
use crate::json_serialization::{JSONConvertible, JSONNode}; // Added
use std::collections::BTreeMap as StdMap;
use std::fmt::{Display, Formatter};
// Added for derive macro pattern

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[derive(Hash)]
pub struct NonTerminal(pub String);

impl JSONConvertible for NonTerminal {
    fn to_json(&self) -> JSONNode { self.0.to_json() }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        String::from_json(node).map(NonTerminal)
    }
}

impl Display for NonTerminal {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Terminal {
    Regex(String),
    Literal(Vec<u8>),
}

impl JSONConvertible for Terminal {
    fn to_json(&self) -> JSONNode {
        match self {
            Terminal::Regex(name) => JSONNode::Object({
                let mut obj = StdMap::new();
                obj.insert("type".to_string(), JSONNode::String("Regex".to_string()));
                obj.insert("value".to_string(), JSONNode::String(name.clone()));
                obj
            }),
            Terminal::Literal(bytes) => JSONNode::Object({
                let mut obj = StdMap::new();
                obj.insert("type".to_string(), JSONNode::String("Literal".to_string()));
                obj.insert("value".to_string(), JSONNode::String(String::from_utf8_lossy(bytes).to_string()));
                obj
            }),
        }
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let type_field = obj.remove("type").ok_or_else(|| "Missing field type for Terminal".to_string())?;
                let value_field = obj.remove("value").ok_or_else(|| "Missing field value for Terminal".to_string())?;

                match String::from_json(type_field)?.as_str() {
                    "Regex" => Ok(Terminal::Regex(String::from_json(value_field)?)),
                    "Literal" => Ok(Terminal::Literal(Vec::from_json(value_field)?)),
                    _ => Err("Unknown type for Terminal".to_string()),
                }
            }
            _ => Err("Expected JSONNode::Object for Terminal".to_string()),
        }
    }
}

impl Display for Terminal {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Terminal::Regex(name) => write!(f, "{}", name),
            Terminal::Literal(bytes) => write!(f, "{:?}", String::from_utf8_lossy(bytes)),
        }
    }
}

pub fn terminal(name: &str) -> Terminal {
    Terminal::Regex(name.to_string())
}

pub fn literal(bytes: Vec<u8>) -> Terminal {
    Terminal::Literal(bytes)
}

impl Terminal {
    pub fn terminal(name: &str) -> Self {
        Terminal::Regex(name.to_string())
    }

    pub fn literal(bytes: Vec<u8>) -> Self {
        Terminal::Literal(bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Symbol {
    Terminal(Terminal),
    NonTerminal(NonTerminal),
}

impl JSONConvertible for Symbol {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        match self {
            Symbol::Terminal(t) => {
                obj.insert("variant".to_string(), JSONNode::String("Terminal".to_string()));
                obj.insert("value".to_string(), t.to_json());
            }
            Symbol::NonTerminal(nt) => {
                obj.insert("variant".to_string(), JSONNode::String("NonTerminal".to_string()));
                obj.insert("value".to_string(), nt.to_json());
            }
        }
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let variant = obj.remove("variant").ok_or_else(|| "Missing field variant for Symbol".to_string())
                                   .and_then(String::from_json)?;
                let value_node = obj.remove("value").ok_or_else(|| "Missing field value for Symbol".to_string())?;
                match variant.as_str() {
                    "Terminal" => Terminal::from_json(value_node).map(Symbol::Terminal),
                    "NonTerminal" => NonTerminal::from_json(value_node).map(Symbol::NonTerminal),
                    _ => Err(format!("Unknown variant {} for Symbol", variant)),
                }
            }
            _ => Err("Expected JSONNode::Object for Symbol".to_string()),
        }
    }
}


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Production {
    pub lhs: NonTerminal,
    pub rhs: Vec<Symbol>,
}

// Manual impl for Production (could be derived)
impl JSONConvertible for Production {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("lhs".to_string(), self.lhs.to_json());
        obj.insert("rhs".to_string(), self.rhs.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let lhs = obj.remove("lhs").ok_or_else(|| "Missing field lhs for Production".to_string())
                                 .and_then(NonTerminal::from_json)?;
                let rhs = obj.remove("rhs").ok_or_else(|| "Missing field rhs for Production".to_string())
                                 .and_then(Vec::<Symbol>::from_json)?;
                Ok(Production { lhs, rhs })
            }
            _ => Err("Expected JSONNode::Object for Production".to_string()),
        }
    }
}

impl Display for Production {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ->", self.lhs.0)?;
        for symbol in &self.rhs {
            match symbol {
                Symbol::Terminal(terminal) => write!(f, " {}", terminal)?,
                Symbol::NonTerminal(non_terminal) => write!(f, " {}", non_terminal.0)?,
            }
        }
        Ok(())
    }
}

pub fn nt(name: &str) -> Symbol {
    Symbol::NonTerminal(NonTerminal(name.to_string()))
}

pub fn t(name: &str) -> Symbol {
    Symbol::Terminal(terminal(name))
}

pub fn prod(name: &str, rhs: Vec<Symbol>) -> Production {
    Production {
        lhs: NonTerminal(name.to_string()),
        rhs,
    }
}

pub fn compute_nullable_nonterminals(productions: &[Production]) -> BTreeSet<NonTerminal> {
    let mut nullable_nonterminals = BTreeSet::new();
    let mut changed = true;

    while changed {
        changed = false;
        for production in productions {
            // Rule 1: A -> ε makes A nullable
            if production.rhs.is_empty() && !nullable_nonterminals.contains(&production.lhs) {
                nullable_nonterminals.insert(production.lhs.clone());
                changed = true;
            // Rule 2: A -> X1 X2 ... Xn makes A nullable if all Xi are nullable non-terminals
            } else if !production.rhs.is_empty() // Ensure RHS is not empty to avoid re-checking Rule 1
                      && production.rhs.iter().all(|symbol| {
                          matches!(symbol, Symbol::NonTerminal(nt) if nullable_nonterminals.contains(nt))
                      })
                      && !nullable_nonterminals.contains(&production.lhs)
            {
                nullable_nonterminals.insert(production.lhs.clone());
                changed = true;
            }
        }
    }

    nullable_nonterminals
}

pub fn compute_first_sets_for_nonterminals(productions: &[Production]) -> BTreeMap<NonTerminal, BTreeSet<Terminal>> {
    let nullable_nonterminals = compute_nullable_nonterminals(productions);
    let mut first_sets: BTreeMap<NonTerminal, BTreeSet<Terminal>> = BTreeMap::new();

    // Initialize for all non-terminals to avoid panics and handle non-terminals that only appear on RHS.
    for p in productions {
        first_sets.entry(p.lhs.clone()).or_default();
        for s in &p.rhs {
            if let Symbol::NonTerminal(nt) = s {
                first_sets.entry(nt.clone()).or_default();
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;

        for production in productions {
            let lhs = &production.lhs;
            let rhs = &production.rhs;

            let old_size = first_sets.get(lhs).unwrap().len();

            for symbol in rhs {
                if let Symbol::NonTerminal(nt) = symbol {
                    let first_nt = first_sets.get(nt).cloned().unwrap_or_default(); // Handle case where nt might not be in first_sets yet
                    first_sets.get_mut(lhs).unwrap().extend(first_nt);

                    if !nullable_nonterminals.contains(nt) {
                        break;
                    }
                } else if let Symbol::Terminal(t) = symbol { // Added this case
                    first_sets.get_mut(lhs).unwrap().insert(t.clone());
                    break;
                }
            }

            if first_sets.get(lhs).unwrap().len() != old_size {
                changed = true;
            }
        }
    }

    first_sets
}

pub fn compute_follow_sets_for_nonterminals(
    productions: &[Production],
    start_production_id: usize,
    first_sets: &BTreeMap<NonTerminal, BTreeSet<Terminal>>,
    nullable_nonterminals: &BTreeSet<NonTerminal>,
) -> BTreeMap<NonTerminal, BTreeSet<Option<Terminal>>> {
    let mut follow_sets: BTreeMap<NonTerminal, BTreeSet<Option<Terminal>>> = BTreeMap::new();

    // Initialize for all non-terminals
    for production in productions {
        follow_sets.entry(production.lhs.clone()).or_default();
        for symbol in &production.rhs { // Ensure all non-terminals in RHS are in follow_sets
            if let Symbol::NonTerminal(nt) = symbol {
                follow_sets.entry(nt.clone()).or_default();
            }
        }
    }

    // Rule 1: Place EOF (None) in FOLLOW(S) where S is the start symbol.
    if !productions.is_empty() {
        let start_nt = &productions[start_production_id].lhs;
        follow_sets.entry(start_nt.clone()).or_default().insert(None);
    }

    let mut changed = true;
    while changed {
        changed = false;

        for production in productions {
            let lhs = &production.lhs;
            let rhs = &production.rhs;

            for (i, symbol) in rhs.iter().enumerate() {
                if let Symbol::NonTerminal(nt) = symbol {
                    let old_len = follow_sets.get(nt).unwrap().len();

                    let mut suffix_is_nullable = true;
                    for next_symbol in &rhs[i + 1..] {
                        match next_symbol {
                            Symbol::Terminal(t_next) => {
                                follow_sets.get_mut(nt).unwrap().insert(Some(t_next.clone()));
                                suffix_is_nullable = false;
                                break;
                            }
                            Symbol::NonTerminal(nt_next) => {
                                let first_next = first_sets.get(nt_next).cloned().unwrap_or_default();
                                follow_sets.get_mut(nt).unwrap().extend(first_next.iter().cloned().map(Some));

                                if !nullable_nonterminals.contains(nt_next) {
                                    suffix_is_nullable = false;
                                    break;
                                }
                            }
                        }
                    }

                    if suffix_is_nullable {
                        let follow_lhs = follow_sets.get(lhs).unwrap().clone();
                        follow_sets.get_mut(nt).unwrap().extend(follow_lhs);
                    }

                    if follow_sets.get(nt).unwrap().len() != old_len {
                        changed = true;
                    }
                }
            }
        }
    }

    follow_sets
}
