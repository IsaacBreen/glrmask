//! GLR grammar representation for table construction.

use super::super::grammar_def::{GrammarDef, NonterminalId, Rule, TerminalId};

/// A grammar augmented for GLR table construction.
///
/// Includes the augmented start rule `S' -> S $`.
#[derive(Debug, Clone)]
pub struct GlrGrammar {
    /// All production rules (index 0 is the augmented start rule).
    pub rules: Vec<Rule>,
    /// Number of terminals (IDs 0..num_terminals).
    pub num_terminals: u32,
    /// Number of nonterminals.
    pub num_nonterminals: u32,
    /// The augmented start nonterminal.
    pub start: NonterminalId,
    /// EOF terminal ID.
    pub eof_terminal: TerminalId,
}

impl GlrGrammar {
    /// Build a GLR grammar from a grammar definition.
    pub fn from_grammar_def(def: &GrammarDef) -> Self {
        // TODO: augment with S' -> S $
        let num_terminals = def.terminals.len() as u32;
        let eof_terminal = num_terminals; // One past the last terminal
        let num_nonterminals = def
            .rules
            .iter()
            .map(|r| r.lhs + 1)
            .max()
            .unwrap_or(0);

        Self {
            rules: def.rules.clone(),
            num_terminals: num_terminals + 1, // +1 for EOF
            num_nonterminals,
            start: def.start,
            eof_terminal,
        }
    }
}
