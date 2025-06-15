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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[derive(Hash)]
pub struct Terminal(pub String);

impl JSONConvertible for Terminal {
    fn to_json(&self) -> JSONNode { self.0.to_json() }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        String::from_json(node).map(Terminal)
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
                Symbol::Terminal(terminal) => write!(f, " {}", terminal.0)?,
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
    Symbol::Terminal(Terminal(name.to_string()))
}

pub fn prod(name: &str, rhs: Vec<Symbol>) -> Production {
    Production {
        lhs: NonTerminal(name.to_string()),
        rhs,
    }
}

pub fn compute_epsilon_nonterminals(productions: &[Production]) -> BTreeSet<NonTerminal> {
    let mut epsilon_nonterminals = BTreeSet::new();
    let mut changed = true;

    while changed {
        changed = false;
        for production in productions {
            if production.rhs.is_empty() && !epsilon_nonterminals.contains(&production.lhs) {
                epsilon_nonterminals.insert(production.lhs.clone());
                changed = true;
            } else if production.rhs.iter().all(|symbol| {
                matches!(symbol, Symbol::NonTerminal(nt) if epsilon_nonterminals.contains(nt))
            }) && !epsilon_nonterminals.contains(&production.lhs)
            {
                epsilon_nonterminals.insert(production.lhs.clone());
                changed = true;
            }
        }
    }

    epsilon_nonterminals
}

pub fn compute_first_sets(productions: &[Production]) -> BTreeMap<NonTerminal, BTreeSet<Terminal>> {
    let epsilon_nonterminals = compute_epsilon_nonterminals(productions);
    let mut first_sets: BTreeMap<NonTerminal, BTreeSet<Terminal>> = BTreeMap::new();

    for production in productions {
        let lhs = &production.lhs;
        first_sets.entry(lhs.clone()).or_default();
        
        for symbol in &production.rhs {
            match symbol {
                Symbol::Terminal(t) => {
                    first_sets.get_mut(lhs).unwrap().insert(t.clone());
                    break;
                }
                Symbol::NonTerminal(nt) => {
                    if !epsilon_nonterminals.contains(nt) {
                        break;
                    }
                }
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;

        for production in productions {
            let lhs = &production.lhs;
            let rhs = &production.rhs;

            let old_size = first_sets.get_mut(lhs).unwrap().len();

            for symbol in rhs {
                if let Symbol::NonTerminal(nt) = symbol {
                    let first_nt = first_sets.get(nt).cloned().unwrap_or_default(); // Handle case where nt might not be in first_sets yet
                    first_sets.get_mut(lhs).unwrap().extend(first_nt);

                    if !epsilon_nonterminals.contains(nt) {
                        break;
                    }
                } else if let Symbol::Terminal(t) = symbol { // Added this case
                    break;
                }
            }

            if first_sets.get_mut(lhs).unwrap().len() != old_size {
                changed = true;
            }
        }
    }

    first_sets
}

pub fn compute_follow_sets(productions: &[Production]) -> BTreeMap<NonTerminal, BTreeSet<Terminal>> {
    let first_sets = compute_first_sets(productions);
    let epsilon_nonterminals = compute_epsilon_nonterminals(productions);
    let mut follow_sets: BTreeMap<NonTerminal, BTreeSet<Terminal>> = BTreeMap::new();

    for production in productions {
        follow_sets.entry(production.lhs.clone()).or_default();
        for symbol in &production.rhs { // Ensure all non-terminals in RHS are in follow_sets
            if let Symbol::NonTerminal(nt) = symbol {
                follow_sets.entry(nt.clone()).or_default();
            }
        }
    }


    let mut changed = true;
    while changed {
        changed = false;

        for production in productions {
            let lhs = &production.lhs;
            let rhs = &production.rhs;

            for (i, symbol) in rhs.iter().enumerate() {
                if let Symbol::NonTerminal(nt) = symbol {
                    let old_size = follow_sets.get_mut(nt).unwrap().len();

                    let mut nullable = true;
                    for next_symbol in &rhs[i + 1..] {
                        match next_symbol {
                            Symbol::Terminal(t_next) => {
                                follow_sets.get_mut(nt).unwrap().insert(t_next.clone());
                                nullable = false;
                                break;
                            }
                            Symbol::NonTerminal(nt_next) => {
                                let first_next = first_sets.get(nt_next).cloned().unwrap_or_default();
                                follow_sets.get_mut(nt).unwrap().extend(first_next.iter().cloned());
                                
                                if !epsilon_nonterminals.contains(nt_next) {
                                    nullable = false;
                                    break;
                                }
                            }
                        }
                    }

                    if nullable {
                        let follow_lhs = follow_sets.get(lhs).unwrap().clone();
                        follow_sets.get_mut(nt).unwrap().extend(follow_lhs);
                    }

                    if follow_sets.get_mut(nt).unwrap().len() != old_size {
                        changed = true;
                    }
                }
            }
        }
    }

    follow_sets
}
