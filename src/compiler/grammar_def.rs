//! Grammar intermediate representation.
//!
//! The canonical internal representation of a grammar that all frontends
//! (EBNF, Lark, JSON Schema) compile down to.

use serde::{Deserialize, Serialize};

/// A grammar definition consisting of production rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarDef {
    /// Production rules indexed by nonterminal ID.
    pub rules: Vec<Rule>,
    /// The start nonterminal.
    pub start: NonterminalId,
    /// Terminal definitions (regex patterns for each terminal).
    pub terminals: Vec<TerminalDef>,
}

/// A nonterminal ID.
pub type NonterminalId = u32;

/// A terminal ID.
pub type TerminalId = u32;

/// A production rule: `lhs -> rhs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    /// Left-hand side nonterminal.
    pub lhs: NonterminalId,
    /// Right-hand side: sequence of symbols.
    pub rhs: Vec<Symbol>,
}

/// A symbol in a production rule's right-hand side.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Symbol {
    /// A terminal symbol.
    Terminal(TerminalId),
    /// A nonterminal symbol.
    Nonterminal(NonterminalId),
}

/// Definition of a terminal symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalDef {
    /// Unique ID of this terminal.
    pub id: TerminalId,
    /// Human-readable name.
    pub name: String,
    /// Regex pattern that this terminal matches.
    pub pattern: String,
}
