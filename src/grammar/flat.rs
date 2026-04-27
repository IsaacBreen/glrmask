use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::automata::regex::Expr;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GrammarDef {
    pub rules: Vec<Rule>,
    pub start: NonterminalID,
    pub terminals: Vec<Terminal>,
    #[serde(default)]
    pub nonterminal_names: BTreeMap<NonterminalID, String>,
    #[serde(default)]
    pub terminal_names: BTreeMap<TerminalID, String>,
    #[serde(default)]
    pub ignore_terminal: Option<TerminalID>,
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

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Symbol::Terminal(id) => write!(f, "T{}", id),
            Symbol::Nonterminal(id) => write!(f, "NT{}", id),
        }
    }
}

impl Symbol {
    fn nonterminal_id(&self) -> Option<NonterminalID> {
        match self {
            Symbol::Nonterminal(nonterminal) => Some(*nonterminal),
            Symbol::Terminal(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Terminal {
    /// An exact byte sequence (e.g., a keyword or punctuation).
    Literal { id: TerminalID, bytes: Vec<u8> },
    /// A regex pattern string (e.g., `[a-z]+` or `\\d`).
    Pattern { id: TerminalID, pattern: String, utf8: bool },
    /// A pre-parsed regex expression.
    Expr { id: TerminalID, expr: Expr },
}

impl Rule {
    fn nonterminal_ids(&self) -> impl Iterator<Item = NonterminalID> + '_ {
        std::iter::once(self.lhs).chain(self.rhs.iter().filter_map(Symbol::nonterminal_id))
    }
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
}

impl GrammarDef {
    pub fn num_terminals(&self) -> u32 {
        self.terminals.len() as u32
    }

    pub fn num_nonterminals(&self) -> u32 {
        self.rules
            .iter()
            .flat_map(|rule| rule.nonterminal_ids())
            .max()
            .map(|id| id + 1)
            .unwrap_or(0)
    }

    pub fn terminal_display_name(&self, terminal: TerminalID) -> String {
        self.terminal_names
            .get(&terminal)
            .cloned()
            .or_else(|| self.terminal_by_id(terminal).map(Terminal::name))
            .unwrap_or_else(|| format!("T{terminal}"))
    }

    fn terminal_by_id(&self, terminal: TerminalID) -> Option<&Terminal> {
        self.terminals
            .iter()
            .find(|terminal_def| terminal_def.id() == terminal)
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

    fn test_grammar(rules: Vec<Rule>, terminals: Vec<Terminal>) -> GrammarDef {
        GrammarDef {
            rules,
            start: 0,
            terminals,
            ..Default::default()
        }
    }

    pub fn simple_ab_grammar() -> GrammarDef {
        test_grammar(
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            vec![literal(0, "a"), literal(1, "b")],
        )
    }

    pub fn choice_grammar() -> GrammarDef {
        test_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            vec![literal(0, "a"), literal(1, "b")],
        )
    }

    pub fn two_nt_grammar() -> GrammarDef {
        test_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            vec![literal(0, "a"), literal(1, "b")],
        )
    }

    pub fn nested_nt_grammar() -> GrammarDef {
        test_grammar(
            vec![
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
            vec![literal(0, "a"), literal(1, "b")],
        )
    }

    pub fn three_terminal_grammar() -> GrammarDef {
        test_grammar(
            vec![Rule {
                lhs: 0,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Terminal(1),
                    Symbol::Terminal(2),
                ],
            }],
            vec![literal(0, "a"), literal(1, "b"), literal(2, "c")],
        )
    }

    pub fn nested_two_rhs_grammar() -> GrammarDef {
        test_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(2)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
                },
            ],
            vec![literal(0, "a"), literal(1, "b"), literal(2, "c")],
        )
    }

    #[test]
    fn test_grammar_def_basics() {
        let g = simple_ab_grammar();
        assert_eq!(g.num_terminals(), 2);
        assert_eq!(g.num_nonterminals(), 1);
    }
}
