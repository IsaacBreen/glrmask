//! Extracted Commit submodule.
//!
//! This file is part of the publication cleanup split of the Commit transition relation.

use super::*;

pub(super) enum ActionableTerminals {
    SingleState(u32),
    ManyStates(SmallVec<[u32; 8]>),
}

impl ActionableTerminals {
    pub(super) fn from_gss(_constraint: &Constraint, gss: &ParserGSS) -> Option<Self> {
        if let Some(state_id) = gss.single_top_value() {
            return Some(Self::SingleState(state_id));
        }

        let states = gss.peek_values();
        if states.is_empty() {
            None
        } else {
            Some(Self::ManyStates(states))
        }
    }

    pub(super) fn contains(&self, constraint: &Constraint, terminal: u32) -> bool {
        match self {
            Self::SingleState(state_id) => constraint.table.advance_row_allows(*state_id, terminal),
            Self::ManyStates(states) => states
                .iter()
                .any(|state_id| constraint.table.advance_row_allows(*state_id, terminal)),
        }
    }
}

pub(super) fn is_ignored_terminal(ignore_terminal: Option<u32>, terminal: u32) -> bool {
    Some(terminal) == ignore_terminal
}

pub(super) fn is_actionable_terminal(
    actionable_terminals: Option<&ActionableTerminals>,
    constraint: &Constraint,
    terminal: u32,
) -> bool {
    !actionable_terminals
        .is_some_and(|actionable| !actionable.contains(constraint, terminal))
}

pub(super) fn collect_unique_actionable_matches(
    constraint: &Constraint,
    actionable_terminals: Option<&ActionableTerminals>,
    ignore_terminal: Option<u32>,
    matches: &[TokenizerMatch],
    reusable_seen_matches: Option<&mut FxHashSet<(usize, u32)>>,
) -> SmallVec<[NormalizedMatch; 8]> {
    let mut normalized = SmallVec::<[NormalizedMatch; 8]>::new();

    if matches.len() <= SMALL_NORMALIZED_MATCH_LINEAR_SCAN_MAX {
        'matches: for matched in matches {
            let ignored = is_ignored_terminal(ignore_terminal, matched.id);
            if !ignored && !is_actionable_terminal(actionable_terminals, constraint, matched.id) {
                continue;
            }
            for existing in &normalized {
                if existing.width == matched.width && existing.terminal_id == matched.id {
                    continue 'matches;
                }
            }
            normalized.push(NormalizedMatch {
                terminal_id: matched.id,
                width: matched.width,
                ignored,
            });
        }
        return normalized;
    }

    if let Some(seen_matches) = reusable_seen_matches {
        seen_matches.clear();
        for matched in matches {
            let ignored = is_ignored_terminal(ignore_terminal, matched.id);
            if !ignored && !is_actionable_terminal(actionable_terminals, constraint, matched.id) {
                continue;
            }
            if !seen_matches.insert((matched.width, matched.id)) {
                continue;
            }
            normalized.push(NormalizedMatch {
                terminal_id: matched.id,
                width: matched.width,
                ignored,
            });
        }
        return normalized;
    }

    let mut seen_matches = FxHashSet::default();
    for matched in matches {
        let ignored = is_ignored_terminal(ignore_terminal, matched.id);
        if !ignored && !is_actionable_terminal(actionable_terminals, constraint, matched.id) {
            continue;
        }
        if !seen_matches.insert((matched.width, matched.id)) {
            continue;
        }
        normalized.push(NormalizedMatch {
            terminal_id: matched.id,
            width: matched.width,
            ignored,
        });
    }
    normalized
}

