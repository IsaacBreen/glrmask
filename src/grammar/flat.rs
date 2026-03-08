#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use serde::{Deserialize, Serialize};

use crate::automata::regex::Expr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarDef {
    pub rules: Vec<Rule>,
    pub start: NonterminalID,
    pub terminals: Vec<Terminal>,
}

pub type NonterminalID = u32;

pub type TerminalID = u32;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rule {
    pub lhs: NonterminalID,
    pub rhs: Vec<Symbol>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Symbol {
    Terminal(TerminalID),
    Nonterminal(NonterminalID),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Terminal {
    /// An exact byte sequence (e.g., a keyword or punctuation).
    Literal { id: TerminalID, bytes: Vec<u8> },
    /// A regex pattern string (e.g., `[a-z]+` or `\\d`).
    Pattern { id: TerminalID, pattern: String },
    /// A pre-parsed regex expression.
    Expr { id: TerminalID, expr: Expr },
}

impl Terminal {
    /// Return the terminal's numeric ID.
    pub fn id(&self) -> TerminalID {
        match self {
            Terminal::Literal { id, .. } => *id,
            Terminal::Pattern { id, .. } => *id,
            Terminal::Expr { id, .. } => *id,
        }
    }

    /// Return a display name for the terminal.
    pub fn name(&self) -> String {
        match self {
            Terminal::Literal { bytes, .. } => String::from_utf8_lossy(bytes).into_owned(),
            Terminal::Pattern { pattern, .. } => pattern.clone(),
            Terminal::Expr { expr, .. } => format!("{:?}", expr),
        }
    }

    /// Return the regex pattern string for the terminal.
    /// For literals, this escapes the bytes into a regex-safe pattern.
    /// For `Expr` variants, returns the debug representation (callers should
    /// prefer working with the `Expr` directly when possible).
    pub fn pattern(&self) -> String {
        match self {
            Terminal::Literal { bytes, .. } => {
                bytes.iter().map(|&b| escape_byte_for_regex(b)).collect()
            }
            Terminal::Pattern { pattern, .. } => pattern.clone(),
            Terminal::Expr { expr, .. } => format!("{:?}", expr),
        }
    }
}

/// Escape a single byte into its regex-pattern representation.
fn escape_byte_for_regex(b: u8) -> String {
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

impl GrammarDef {
    pub fn num_terminals(&self) -> u32 {
        self.terminals.len() as u32
    }

    pub fn num_nonterminals(&self) -> u32 {
        self.rules
            .iter()
            .flat_map(|rule| {
                std::iter::once(rule.lhs).chain(rule.rhs.iter().filter_map(|sym| match sym {
                    Symbol::Nonterminal(id) => Some(*id),
                    Symbol::Terminal(_) => None,
                }))
            })
            .max()
            .map(|id| id + 1)
            .unwrap_or(0)
    }

    pub fn terminal_pattern(&self, terminal: TerminalID) -> String {
        self.terminals
            .iter()
            .find(|t| t.id() == terminal)
            .map(|t| t.pattern())
            .unwrap_or_default()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn literal(id: u32, s: &str) -> Terminal {
        Terminal::Literal {
            id,
            bytes: s.as_bytes().to_vec(),
        }
    }

    pub fn simple_ab_grammar() -> GrammarDef {
        GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![literal(0, "a"), literal(1, "b")],
        }
    }

    pub fn choice_grammar() -> GrammarDef {
        GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            start: 0,
            terminals: vec![literal(0, "a"), literal(1, "b")],
        }
    }

    pub fn two_nt_grammar() -> GrammarDef {

        GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            start: 0,
            terminals: vec![literal(0, "a"), literal(1, "b")],
        }
    }

    pub fn nested_nt_grammar() -> GrammarDef {

        GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Nonterminal(2)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            start: 0,
            terminals: vec![literal(0, "a"), literal(1, "b")],
        }
    }

    pub fn three_terminal_grammar() -> GrammarDef {
        GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Terminal(1),
                    Symbol::Terminal(2),
                ],
            }],
            start: 0,
            terminals: vec![literal(0, "a"), literal(1, "b"), literal(2, "c")],
        }
    }

    pub fn nested_two_rhs_grammar() -> GrammarDef {

        GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(2)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
                },
            ],
            start: 0,
            terminals: vec![literal(0, "a"), literal(1, "b"), literal(2, "c")],
        }
    }

    #[test]
    fn test_grammar_def_basics() {
        let g = simple_ab_grammar();
        assert_eq!(g.num_terminals(), 2);
        assert_eq!(g.num_nonterminals(), 1);
    }
}
