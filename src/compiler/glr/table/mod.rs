use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use super::analysis::{AnalyzedGrammar, EOF};
use crate::ds::bitset::BitSet;
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
    /// Terminal support used by cheap admission/mask queries.
    ///
    /// `action` is the optimized execution table. Some execution actions are
    /// guarded stack effects, whose guards must be evaluated when executing the
    /// action. This side table is captured before guard-producing stack-effect
    /// lowering and is kept in sync across state remapping/merging. A bit set in
    /// this vector answers only the admission question: "can a reachable parser
    /// path with this top state advance on this terminal?"  That lets
    /// `stack_may_advance_on*` be pure row-presence checks without inspecting an
    /// optimized action body.
    #[serde(default)]
    pub advance: Vec<BitSet>,
    /// Set of (state, terminal) pairs where the shift was created by the
    /// transfer mechanism. The characterization should treat these as
    /// non-replace to avoid creating pop-0 reduces in the template NFA.
    pub forwarded_shifts: FxHashSet<(u32, TerminalID)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TableAmbiguityKind {
    Split,
    StackShifts,
    GuardedStackShifts,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TableAmbiguity {
    pub state: u32,
    pub terminal: TerminalID,
    pub kind: TableAmbiguityKind,
    pub alternatives: usize,
}

fn action_ambiguity(action: &Action) -> Option<(TableAmbiguityKind, usize)> {
    match action {
        Action::Split {
            shift,
            reduces,
            accept,
        } => {
            let alternatives = usize::from(shift.is_some()) + reduces.len() + usize::from(*accept);
            (alternatives > 1).then_some((TableAmbiguityKind::Split, alternatives))
        }
        Action::StackShifts(shifts) => {
            (shifts.len() > 1).then_some((TableAmbiguityKind::StackShifts, shifts.len()))
        }
        Action::GuardedStackShifts(shifts) => {
            (shifts.len() > 1).then_some((TableAmbiguityKind::GuardedStackShifts, shifts.len()))
        }
        _ => None,
    }
}

impl GLRTable {
    pub fn build(grammar: &AnalyzedGrammar) -> Self {
        build_table(grammar)
    }

    #[inline]
    fn terminal_bit(&self, terminal: TerminalID) -> Option<usize> {
        if terminal == EOF {
            Some(self.num_terminals as usize)
        } else if terminal < self.num_terminals {
            Some(terminal as usize)
        } else {
            None
        }
    }

    #[inline]
    fn has_advance_rows(&self) -> bool {
        self.advance.len() == self.num_states as usize
    }

    pub(crate) fn rebuild_advance_rows_from_actions(&mut self) {
        self.advance = action_presence_rows(&self.action, self.num_terminals);
    }

    #[inline]
    pub(crate) fn advance_row_allows(&self, state: u32, terminal: TerminalID) -> bool {
        if self.has_advance_rows() {
            let Some(bit) = self.terminal_bit(terminal) else {
                return false;
            };
            return self
                .advance
                .get(state as usize)
                .is_some_and(|row| row.contains(bit));
        }

        // Compatibility fallback for hand-built test tables and older serialized
        // artifacts that do not carry the side table. Newly compiled tables build
        // `advance` before guard-producing optimizations run.
        self.action(state, terminal).is_some()
    }

    #[inline]
    pub(crate) fn advance_row_intersects(&self, state: u32, terminals: &BitSet) -> bool {
        if self.has_advance_rows()
            && let Some(row) = self.advance.get(state as usize)
        {
            return row
                .words()
                .iter()
                .zip(terminals.words())
                .any(|(left, right)| (*left & *right) != 0);
        }

        self.action.get(state as usize).is_some_and(|actions| {
            actions.keys().any(|terminal| {
                self.terminal_bit(terminal)
                    .is_some_and(|bit| terminals.contains(bit))
            })
        })
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

    pub fn ambiguous_actions(&self) -> Vec<TableAmbiguity> {
        let mut ambiguities = Vec::new();
        for (state, row) in self.action.iter().enumerate() {
            for (terminal, action) in row {
                if let Some((kind, alternatives)) = action_ambiguity(action) {
                    ambiguities.push(TableAmbiguity {
                        state: state as u32,
                        terminal,
                        kind,
                        alternatives,
                    });
                }
            }
        }
        ambiguities
    }

    pub fn has_ambiguity(&self) -> bool {
        self.action.iter().enumerate().any(|(_, row)| {
            row.into_iter()
                .any(|(_, action)| action_ambiguity(action).is_some())
        })
    }

    #[inline]
    pub fn goto_target(&self, state: u32, nt: NonterminalID) -> Option<(u32, bool)> {
        self.goto
            .get(state as usize)
            .and_then(|by_nt| by_nt.get(&nt).copied())
    }

    pub(super) fn validate_structure(&self, context: &str) {
        let expected_len = self.num_states as usize;
        if self.action.len() != expected_len {
            panic!(
                "{context}: action row count {} does not match num_states {}",
                self.action.len(),
                self.num_states,
            );
        }
        if self.goto.len() != expected_len {
            panic!(
                "{context}: goto row count {} does not match num_states {}",
                self.goto.len(),
                self.num_states,
            );
        }
        if !self.advance.is_empty() && self.advance.len() != expected_len {
            panic!(
                "{context}: advance row count {} does not match num_states {}",
                self.advance.len(),
                self.num_states,
            );
        }

        let validate_target = |source_state: u32,
                               label_kind: &str,
                               label_value: u32,
                               path: &str,
                               target: u32| {
            if target >= self.num_states {
                panic!(
                    "{context}: state {} {} {} has invalid {} target {} >= num_states {}",
                    source_state,
                    label_kind,
                    label_value,
                    path,
                    target,
                    self.num_states,
                );
            }
        };

        for (source_state, row) in self.action.iter().enumerate() {
            let source_state = source_state as u32;
            for (terminal, action) in row.iter() {
                match action {
                    Action::Shift(target, _) => {
                        validate_target(source_state, "terminal", terminal, "Action::Shift", *target);
                    }
                    Action::StackShifts(shifts) => {
                        for (shift_idx, shift) in shifts.iter().enumerate() {
                            for (push_idx, &target) in shift.pushes.iter().enumerate() {
                                validate_target(
                                    source_state,
                                    "terminal",
                                    terminal,
                                    &format!("Action::StackShifts[{shift_idx}].pushes[{push_idx}]"),
                                    target,
                                );
                            }
                        }
                    }
                    Action::GuardedStackShifts(shifts) => {
                        for (shift_idx, shift) in shifts.iter().enumerate() {
                            for (guard_idx, guard) in shift.guards.iter().enumerate() {
                                for (state_idx, &target) in guard.states.iter().enumerate() {
                                    validate_target(
                                        source_state,
                                        "terminal",
                                        terminal,
                                        &format!("Action::GuardedStackShifts[{shift_idx}].guards[{guard_idx}].states[{state_idx}]"),
                                        target,
                                    );
                                }
                            }
                            for (push_idx, &target) in shift.pushes.iter().enumerate() {
                                validate_target(
                                    source_state,
                                    "terminal",
                                    terminal,
                                    &format!("Action::GuardedStackShifts[{shift_idx}].pushes[{push_idx}]"),
                                    target,
                                );
                            }
                        }
                    }
                    Action::Split { shift: Some((target, _)), .. } => {
                        validate_target(
                            source_state,
                            "terminal",
                            terminal,
                            "Action::Split.shift",
                            *target,
                        );
                    }
                    Action::Split { shift: None, .. } | Action::Reduce(..) | Action::Accept => {}
                }
            }
        }

        for (source_state, row) in self.goto.iter().enumerate() {
            let source_state = source_state as u32;
            for (&nonterminal, &(target, _)) in row.iter() {
                validate_target(source_state, "nonterminal", nonterminal, "goto", target);
            }
        }

        for &(state, terminal) in &self.forwarded_shifts {
            if state >= self.num_states {
                panic!(
                    "{context}: forwarded_shifts has invalid state {} for terminal {} >= num_states {}",
                    state,
                    terminal,
                    self.num_states,
                );
            }
        }
    }

    #[inline]
    pub fn nonterminal_display_name(&self, nt: NonterminalID) -> Option<&str> {
        self.nonterminal_display_names
            .get(nt as usize)
            .map(String::as_str)
    }
}

fn action_presence_rows(action: &[ActionRow], num_terminals: u32) -> Vec<BitSet> {
    let mut rows = Vec::with_capacity(action.len());
    for action_row in action {
        rows.push(action_presence_row(action_row, num_terminals));
    }
    rows
}

fn action_presence_row(action_row: &ActionRow, num_terminals: u32) -> BitSet {
    let mut row = BitSet::new(num_terminals as usize + 1);
    for terminal in action_row.keys() {
        let bit = if terminal == EOF {
            num_terminals as usize
        } else if terminal < num_terminals {
            terminal as usize
        } else {
            continue;
        };
        row.set(bit);
    }
    row
}

impl GLRTable {
    pub(crate) fn extend_advance_rows_from_actions(&mut self) {
        if self.advance.is_empty() {
            return;
        }

        for action_row in self.action.iter().skip(self.advance.len()) {
            self.advance
                .push(action_presence_row(action_row, self.num_terminals));
        }
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
        let action: Vec<_> = action_rows
            .iter()
            .map(|row| ActionRow::from_iter(row.iter().cloned()))
            .collect();
        let advance = super::action_presence_rows(&action, num_terminals);
        GLRTable {
            action,
            goto: goto_rows
                .iter()
                .map(|row| GotoRow::from_iter(row.iter().cloned()))
                .collect(),
            num_states,
            num_terminals,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            advance,
            forwarded_shifts: Default::default(),
        }
    }
}

#[cfg(test)]
mod ambiguity_tests {
    use super::testing::build_test_table;
    use super::{Action, GuardedStackShift, StackShift, TableAmbiguityKind};

    #[test]
    fn ambiguous_actions_reports_split_and_stack_shift_fanout() {
        let token = 0;
        let table = build_test_table(
            4,
            1,
            &[
                &[(token, Action::Shift(1, false))],
                &[(
                    token,
                    Action::Split {
                        shift: Some((2, false)),
                        reduces: vec![(0, 1)],
                        accept: false,
                    },
                )],
                &[(
                    token,
                    Action::StackShifts(vec![
                        StackShift {
                            pop: 1,
                            pushes: vec![2],
                        },
                        StackShift {
                            pop: 2,
                            pushes: vec![3],
                        },
                    ]),
                )],
                &[(
                    token,
                    Action::GuardedStackShifts(vec![
                        GuardedStackShift {
                            guards: Vec::new(),
                            pop: 1,
                            pushes: vec![2],
                        },
                        GuardedStackShift {
                            guards: Vec::new(),
                            pop: 1,
                            pushes: vec![3],
                        },
                    ]),
                )],
            ],
            &[&[], &[], &[], &[]],
        );

        let ambiguities = table.ambiguous_actions();
        assert!(table.has_ambiguity());
        assert_eq!(ambiguities.len(), 3);
        assert_eq!(ambiguities[0].kind, TableAmbiguityKind::Split);
        assert_eq!(ambiguities[0].alternatives, 2);
        assert_eq!(ambiguities[1].kind, TableAmbiguityKind::StackShifts);
        assert_eq!(ambiguities[1].alternatives, 2);
        assert_eq!(ambiguities[2].kind, TableAmbiguityKind::GuardedStackShifts);
        assert_eq!(ambiguities[2].alternatives, 2);
    }

    #[test]
    #[should_panic(expected = "validator test action target")]
    fn validate_structure_panics_on_invalid_action_target() {
        let token = 0;
        let table = build_test_table(
            2,
            1,
            &[&[(token, Action::Shift(2, false))], &[]],
            &[&[], &[]],
        );

        table.validate_structure("validator test action target");
    }
}
