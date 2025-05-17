use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NonTerminal(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Terminal(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Symbol {
    Terminal(Terminal),
    NonTerminal(NonTerminal),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Production {
    pub lhs: NonTerminal,
    pub rhs: Vec<Symbol>,
}

use crate::json_serialization::{JSONNode, JSONConvertible};

impl JSONConvertible for NonTerminal {
    fn to_json(&self) -> JSONNode { self.0.to_json() }
    fn from_json(node: &JSONNode) -> Result<Self, String> { String::from_json(node).map(NonTerminal) }
}

impl JSONConvertible for Terminal {
    fn to_json(&self) -> JSONNode { self.0.to_json() }
    fn from_json(node: &JSONNode) -> Result<Self, String> { String::from_json(node).map(Terminal) }
}

impl JSONConvertible for Symbol {
    fn to_json(&self) -> JSONNode {
        match self {
            Symbol::Terminal(t) => crate::json_serialization::struct_to_json_object(vec![
                ("type", JSONNode::String("Terminal".to_string())),
                ("value", t.to_json()),
            ]),
            Symbol::NonTerminal(nt) => crate::json_serialization::struct_to_json_object(vec![
                ("type", JSONNode::String("NonTerminal".to_string())),
                ("value", nt.to_json()),
            ]),
        }
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        let map = crate::json_serialization::json_object_to_btreemap(node)?;
        let type_str = map.get("type").ok_or_else(|| "Missing 'type' field for Symbol".to_string())
            .and_then(String::from_json)?;
        let value_node = map.get("value").ok_or_else(|| "Missing 'value' field for Symbol".to_string())?;

        match type_str.as_str() {
            "Terminal" => Terminal::from_json(value_node).map(Symbol::Terminal),
            "NonTerminal" => NonTerminal::from_json(value_node).map(Symbol::NonTerminal),
            _ => Err(format!("Unknown Symbol type: {}", type_str)),
        }
    }
}

impl JSONConvertible for Production {
    fn to_json(&self) -> JSONNode {
        crate::json_serialization::struct_to_json_object(vec![
            ("lhs", self.lhs.to_json()),
            ("rhs", self.rhs.to_json()),
        ])
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        let map = crate::json_serialization::json_object_to_btreemap(node)?;
        Ok(Production {
            lhs: map.get("lhs").ok_or_else(|| "Missing 'lhs' field for Production".to_string()).and_then(NonTerminal::from_json)?,
            rhs: map.get("rhs").ok_or_else(|| "Missing 'rhs' field for Production".to_string()).and_then(Vec::<Symbol>::from_json)?,
        })
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
                    let first_nt = first_sets[nt].clone();
                    first_sets.get_mut(lhs).unwrap().extend(first_nt);

                    if !epsilon_nonterminals.contains(nt) {
                        break;
                    }
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
                                let first_next = &first_sets[nt_next];
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
