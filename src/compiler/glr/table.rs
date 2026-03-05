//! GLR parse table (action + goto tables).

use serde::{Deserialize, Serialize};

use super::grammar::GlrGrammar;

/// An action in the GLR parse table.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    /// Shift to state.
    Shift(u32),
    /// Reduce by rule.
    Reduce(u32),
    /// Accept the input.
    Accept,
}

/// A GLR parse table with action and goto tables.
///
/// Supports ambiguous grammars (multiple actions per cell).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlrTable {
    /// `action[state][terminal]` = list of actions.
    pub action: Vec<Vec<Vec<Action>>>,
    /// `goto[state][nonterminal]` = target state (or u32::MAX for error).
    pub goto: Vec<Vec<u32>>,
    /// Number of states.
    pub num_states: u32,
}

impl GlrTable {
    /// Build parse tables from a GLR grammar using LR(0) item sets + SLR(1) lookaheads.
    pub fn build(_grammar: &GlrGrammar) -> Self {
        // TODO: Implement LR(0) item set construction + SLR(1) table generation
        Self {
            action: Vec::new(),
            goto: Vec::new(),
            num_states: 0,
        }
    }
}
