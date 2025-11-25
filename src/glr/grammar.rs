use crate::json_serialization::{JSONConvertible, JSONNode};
use json_convertible_derive::JSONConvertible;
use std::collections::BTreeMap as StdMap;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible)]
pub struct NonTerminal(pub String);

impl From<&str> for NonTerminal {
    fn from(s: &str) -> Self {
        NonTerminal(s.to_string())
    }
}

impl Display for NonTerminal {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Terminal {
    RegexName(String),
    Literal(Vec<u8>),
}

impl JSONConvertible for Terminal {
    fn to_json(&self) -> JSONNode {
        match self {
            Terminal::RegexName(name) => {
                let mut obj = StdMap::new();
                obj.insert("type".to_string(), JSONNode::String("Regex".to_string()));
                obj.insert("value".to_string(), JSONNode::String(name.clone()));
                JSONNode::Object(obj)
            }
            Terminal::Literal(bytes) => {
                let mut obj = StdMap::new();
                obj.insert("type".to_string(), JSONNode::String("Literal".to_string()));
                obj.insert("value".to_string(), bytes.to_json());
                JSONNode::Object(obj)
            }
        }
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let type_field = obj
                    .remove("type")
                    .ok_or_else(|| "Missing field type for Terminal".to_string())?;
                let value_field = obj
                    .remove("value")
                    .ok_or_else(|| "Missing field value for Terminal".to_string())?;
                match String::from_json(type_field)?.as_str() {
                    "Regex" => Ok(Terminal::RegexName(String::from_json(value_field)?)),
                    "Literal" => Vec::<u8>::from_json(value_field).map(Terminal::Literal),
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
            Terminal::RegexName(name) => {
                // Use bare identifier if possible, otherwise quote.
                let mut chars = name.chars();
                let is_ident = if let Some(first) = chars.next() {
                    (first.is_ascii_alphabetic() || first == '_')
                        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
                } else {
                    false
                };
                if is_ident {
                    write!(f, "{name}")
                } else {
                    write!(
                        f,
                        "'{}'",
                        name.replace('\\', "\\\\").replace('\'', "\\'")
                    )
                }
            }
            Terminal::Literal(bytes) => {
                let s = String::from_utf8_lossy(bytes);
                write!(
                    f,
                    "'{}'",
                    s.replace('\\', "\\\\").replace('\'', "\\'")
                )
            }
        }
    }
}

pub fn regex_name(name: &str) -> Terminal {
    Terminal::RegexName(name.to_string())
}

pub fn literal(bytes: &[u8]) -> Terminal {
    Terminal::Literal(bytes.to_vec())
}

impl Terminal {
    pub fn terminal(name: &str) -> Self {
        Terminal::RegexName(name.to_string())
    }

    pub fn regex_name(name: &str) -> Self {
        Terminal::RegexName(name.to_string())
    }

    pub fn literal(bytes: Vec<u8>) -> Self {
        Terminal::Literal(bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible)]
pub enum Symbol {
    Terminal(Terminal),
    NonTerminal(NonTerminal),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, JSONConvertible)]
pub struct Production {
    pub lhs: NonTerminal,
    pub rhs: Vec<Symbol>,
}

impl Display for Production {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ->", self.lhs.0)?;
        for symbol in &self.rhs {
            match symbol {
                Symbol::Terminal(t) => write!(f, " {t}")?,
                Symbol::NonTerminal(nt) => write!(f, " {}", nt.0)?,
            }
        }
        Ok(())
    }
}

pub fn nt(name: &str) -> Symbol {
    Symbol::NonTerminal(NonTerminal(name.to_string()))
}

pub fn t(name: &str) -> Symbol {
    Symbol::Terminal(regex_name(name))
}

pub fn prod(name: &str, rhs: Vec<Symbol>) -> Production {
    Production {
        lhs: NonTerminal(name.to_string()),
        rhs,
    }
}
