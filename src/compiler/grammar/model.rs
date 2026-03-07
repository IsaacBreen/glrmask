//! Flattened grammar model.
//!
//! This is the canonical flattened grammar representation that all importers
//! (EBNF, Lark, JSON Schema) compile down to before parser-table analysis.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use serde::{Deserialize, Serialize};

/// A grammar definition consisting of production rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarDef {
    /// Production rules.
    pub rules: Vec<Rule>,
    /// The start nonterminal.
    pub start: NonterminalID,
    /// Terminal metadata.
    pub terminals: Vec<Terminal>,
    /// TerminalID → regex/pattern string used to build the tokenizer.
    pub terminal_patterns: Vec<String>,
}

/// A nonterminal ID.
pub type NonterminalID = u32;

/// A terminal ID.
pub type TerminalID = u32;

/// A production rule: `lhs -> rhs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    /// Left-hand side nonterminal.
    pub lhs: NonterminalID,
    /// Right-hand side: sequence of symbols.
    pub rhs: Vec<Symbol>,
}

/// A symbol in a production rule's right-hand side.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Symbol {
    /// A terminal symbol.
    Terminal(TerminalID),
    /// A nonterminal symbol.
    Nonterminal(NonterminalID),
}

/// Metadata for a terminal symbol in the flattened grammar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Terminal {
    /// Unique ID of this terminal.
    pub id: TerminalID,
    /// Human-readable name.
    pub name: String,
}

impl GrammarDef {
    /// Number of terminals.
    pub fn num_terminals(&self) -> u32 {
        unimplemented!()
    }

    /// Number of nonterminals (determined by scanning rules).
    pub fn num_nonterminals(&self) -> u32 {
        unimplemented!()
    }

    /// Lookup the tokenizer pattern for a terminal by ID.
    pub fn terminal_pattern(&self, terminal: TerminalID) -> &str {
        let _ = terminal;
        unimplemented!()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn terminal(id: u32, name: &str) -> Terminal {
        Terminal {
            id,
            name: name.into(),
        }
    }

    /// Helper: build a tiny grammar "S → a b" with 1 rule, 2 terminals.
    pub fn simple_ab_grammar() -> GrammarDef {
        GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![terminal(0, "a"), terminal(1, "b")],
            terminal_patterns: vec!["a".into(), "b".into()],
        }
    }

    /// Helper: build a grammar with a choice: "S → a | b".
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
            terminals: vec![terminal(0, "a"), terminal(1, "b")],
            terminal_patterns: vec!["a".into(), "b".into()],
        }
    }

    /// Helper: build a grammar "S → A b, A → a" with 2 nonterminals.
    pub fn two_nt_grammar() -> GrammarDef {
        // NT 0 = S, NT 1 = A
        // T 0 = a, T 1 = b
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
            terminals: vec![terminal(0, "a"), terminal(1, "b")],
            terminal_patterns: vec!["a".into(), "b".into()],
        }
    }

    /// Helper: build a grammar "S → A B, A → a, B → b" with 3 nonterminals.
    pub fn nested_nt_grammar() -> GrammarDef {
        // NT 0 = S, NT 1 = A, NT 2 = B
        // T 0 = a, T 1 = b
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
            terminals: vec![terminal(0, "a"), terminal(1, "b")],
            terminal_patterns: vec!["a".into(), "b".into()],
        }
    }

    /// Helper: build a grammar "S → a b c" with 3 terminals.
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
            terminals: vec![terminal(0, "a"), terminal(1, "b"), terminal(2, "c")],
            terminal_patterns: vec!["a".into(), "b".into(), "c".into()],
        }
    }

    /// Helper: build a grammar "S → A c, A → a b" with a nonterminal that produces two terminals.
    pub fn nested_two_rhs_grammar() -> GrammarDef {
        // NT 0 = S, NT 1 = A
        // T 0 = a, T 1 = b, T 2 = c
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
            terminals: vec![terminal(0, "a"), terminal(1, "b"), terminal(2, "c")],
            terminal_patterns: vec!["a".into(), "b".into(), "c".into()],
        }
    }

    #[test]
    fn test_grammar_def_basics() {
        let g = simple_ab_grammar();
        assert_eq!(g.num_terminals(), 2);
        assert_eq!(g.num_nonterminals(), 1);
    }
}
