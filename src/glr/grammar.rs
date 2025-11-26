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

/// Intermediate type for Terminal JSON serialization (maintains backward compatibility)
/// Uses "type" field with values "Regex" and "Literal" instead of standard "variant"
#[derive(JSONConvertible)]
enum TerminalJSON {
    Regex { value: String },
    Literal { value: Vec<u8> },
}

impl TerminalJSON {
    fn from_terminal(t: &Terminal) -> Self {
        match t {
            Terminal::RegexName(name) => TerminalJSON::Regex { value: name.clone() },
            Terminal::Literal(bytes) => TerminalJSON::Literal { value: bytes.clone() },
        }
    }

    fn to_terminal(self) -> Terminal {
        match self {
            TerminalJSON::Regex { value } => Terminal::RegexName(value),
            TerminalJSON::Literal { value } => Terminal::Literal(value),
        }
    }
}

impl JSONConvertible for Terminal {
    fn to_json(&self) -> JSONNode {
        // Use "type" key instead of "variant" for backward compatibility
        match TerminalJSON::from_terminal(self).to_json() {
            JSONNode::Object(mut obj) => {
                if let Some(variant) = obj.remove("variant") {
                    obj.insert("type".to_string(), variant);
                }
                JSONNode::Object(obj)
            }
            other => other,
        }
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        // Accept "type" key for backward compatibility
        match node {
            JSONNode::Object(mut obj) => {
                if let Some(type_val) = obj.remove("type") {
                    obj.insert("variant".to_string(), type_val);
                }
                TerminalJSON::from_json(JSONNode::Object(obj)).map(|t| t.to_terminal())
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
