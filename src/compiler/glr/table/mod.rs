use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use super::analysis::{AnalyzedGrammar, EOF};
use crate::grammar::flat::{NonterminalID, Rule, Symbol, TerminalID};

mod action;
mod build;
mod optimize;
mod row;

pub use action::{Action, GuardedStackShift, StackShift, StackShiftGuard};

use build::{build_table, Item, PendingAction};
use optimize::merge_same_core_lr1_states;

use row::{ActionRow, GotoRow};

const DISABLE_DEFAULT_ACTION_ROWS_ENV: &str = "GLRMASK_DISABLE_DEFAULT_ACTION_ROWS";

fn default_action_rows_enabled() -> bool {
    !std::env::var(DISABLE_DEFAULT_ACTION_ROWS_ENV)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GLRTable {
    pub action: Vec<ActionRow>,
    pub goto: Vec<GotoRow>,
    pub num_states: u32,
    pub num_terminals: u32,
    pub num_rules: u32,
    pub rules: Vec<Rule>,
    #[serde(default)]
    pub nonterminal_display_names: Vec<String>,
    /// Set of (state, terminal) pairs where the shift was created by the
    /// transfer mechanism. The characterization should treat these as
    /// non-replace to avoid creating pop-0 reduces in the template NFA.
    pub forwarded_shifts: FxHashSet<(u32, TerminalID)>,
}

impl GLRTable {
    pub fn build(grammar: &AnalyzedGrammar) -> Self {
        build_table(grammar)
    }

    pub(crate) fn compress_default_action_rows(&mut self) {
        for row in &mut self.action {
            row.compress_default(self.num_terminals);
        }
    }

    #[inline]
    pub fn action(&self, state: u32, terminal: TerminalID) -> Option<&Action> {
        self.action
            .get(state as usize)
            .and_then(|by_terminal| by_terminal.get(&terminal))
    }

    #[inline]
    pub fn goto_target(&self, state: u32, nt: NonterminalID) -> Option<(u32, bool)> {
        self.goto
            .get(state as usize)
            .and_then(|by_nt| by_nt.get(&nt).copied())
    }

    #[inline]
    pub fn nonterminal_display_name(&self, nt: NonterminalID) -> Option<&str> {
        self.nonterminal_display_names
            .get(nt as usize)
            .map(String::as_str)
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use super::{Action, GLRTable};
    use super::row::{ActionRow, GotoRow};
    use crate::grammar::flat::{NonterminalID, TerminalID};

    pub(crate) fn build_test_table(
        num_states: u32,
        num_terminals: u32,
        action_rows: &[&[(TerminalID, Action)]],
        goto_rows: &[&[(NonterminalID, (u32, bool))]],
    ) -> GLRTable {
        GLRTable {
            action: action_rows
                .iter()
                .map(|row| ActionRow::from_iter(row.iter().cloned()))
                .collect(),
            goto: goto_rows
                .iter()
                .map(|row| GotoRow::from_iter(row.iter().cloned()))
                .collect(),
            num_states,
            num_terminals,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            forwarded_shifts: Default::default(),
        }
    }
}