use std::collections::{BTreeMap, BTreeSet};
use crate::json_serialization::{JSONConvertible, JSONNode};
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

pub fn regex(name: &str) -> Terminal {
    Terminal::Regex(name.to_string())
}

pub fn literal(bytes: &[u8]) -> Terminal {
    Terminal::Literal(bytes.to_vec())
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
    Symbol::Terminal(regex(name))
}

pub fn prod(name: &str, rhs: Vec<Symbol>) -> Production {
    Production {
        lhs: NonTerminal(name.to_string()),
        rhs,
    }
}
