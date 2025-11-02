use crate::glr::parser::GLRParser;
use crate::glr::table::{NonTerminalID, Stage7ShiftsAndReducesLookaheadValue, StateID, TerminalID};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReduceCharacterization {
    pub terminal: TerminalID,
    pub nonterminal: NonTerminalID,
    // (revealed_state, pop_n, nonterminal)
    pub reveal_and_rereduces: BTreeSet<(StateID, usize, NonTerminalID)>,
    // (revealed_state, goto_state, shift_state)
    pub reveal_goto_shift_escapes: BTreeSet<(StateID, StateID, StateID)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BelowBottomCharacterization {
    pub terminal: TerminalID,
    // (initial_state, shift_state)
    pub initial_shifts: BTreeSet<(StateID, StateID)>,
    // (initial_state, pop_n, nonterminal)
    pub initial_reduces: BTreeSet<(StateID, usize, NonTerminalID)>,
    pub reduce_characterizations: BTreeMap<NonTerminalID, ReduceCharacterization>,
}

pub fn compute_all_characterizations(parser: &GLRParser) -> BTreeMap<TerminalID, BelowBottomCharacterization> {
    let mut all_chars = BTreeMap::new();
    for &terminal_id in parser.terminal_map.right_values() {
        all_chars.insert(terminal_id, compute_below_bottom_characterization(parser, terminal_id));
    }
    all_chars
}

pub fn compute_below_bottom_characterization(parser: &GLRParser, terminal_id: TerminalID) -> BelowBottomCharacterization {
    let mut char = BelowBottomCharacterization {
        terminal: terminal_id,
        initial_shifts: BTreeSet::new(),
        initial_reduces: BTreeSet::new(),
        reduce_characterizations: BTreeMap::new(),
    };

    // --- 1. Compute initial actions ---
    for (&initial_state, row) in &parser.table {
        if let Some(action) = row.shifts_and_reduces_full.get(&terminal_id) {
            match action {
                Stage7ShiftsAndReducesLookaheadValue::Shift(shift_state) => {
                    char.initial_shifts.insert((initial_state, *shift_state));
                }
                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, .. } => {
                    char.initial_reduces.insert((initial_state, *len, *nonterminal_id));
                }
                Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                    if let Some(shift_state) = shift {
                        char.initial_shifts.insert((initial_state, *shift_state));
                    }
                    for (len, nts) in reduces {
                        for (nt_id, _) in nts {
                            char.initial_reduces.insert((initial_state, *len, *nt_id));
                        }
                    }
                }
            }
        }
    }

    // --- 2. Compute reduce characterizations ---
    for &nt_id in parser.non_terminal_map.right_values() {
        let mut reduce_char = ReduceCharacterization {
            terminal: terminal_id,
            nonterminal: nt_id,
            reveal_and_rereduces: BTreeSet::new(),
            reveal_goto_shift_escapes: BTreeSet::new(),
        };

        // Iterate over all possible revealed states
        for (&revealed_state, row) in &parser.table {
            // Check if this state has a GOTO for our non-terminal
            if let Some(goto) = row.gotos.get(&nt_id) {
                if let Some(goto_state) = goto.state_id {
                    // This is a valid starting point for a chain.
                    let mut worklist: VecDeque<StateID> = VecDeque::new();
                    worklist.push_back(goto_state);
                    let mut visited: BTreeSet<StateID> = BTreeSet::new();
                    visited.insert(goto_state);

                    while let Some(current_state) = worklist.pop_front() {
                        if let Some(action) = parser.table.get(&current_state).and_then(|r| r.shifts_and_reduces_full.get(&terminal_id)) {
                            match action {
                                Stage7ShiftsAndReducesLookaheadValue::Shift(shift_state) => {
                                    reduce_char.reveal_goto_shift_escapes.insert((revealed_state, current_state, *shift_state));
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id: reduce_nt, len, .. } => {
                                    if *len == 1 {
                                        // Unit reduction chain
                                        if let Some(next_goto) = parser.table.get(&revealed_state).and_then(|r| r.gotos.get(reduce_nt)) {
                                            if let Some(next_goto_state) = next_goto.state_id {
                                                if visited.insert(next_goto_state) {
                                                    worklist.push_back(next_goto_state);
                                                }
                                            }
                                        }
                                    } else if *len > 1 {
                                        // Pop below revealed state
                                        reduce_char.reveal_and_rereduces.insert((revealed_state, *len - 2, *reduce_nt));
                                    }
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                                    if let Some(shift_state) = shift {
                                        reduce_char.reveal_goto_shift_escapes.insert((revealed_state, current_state, *shift_state));
                                    }
                                    for (len, nts) in reduces {
                                        for (reduce_nt, _) in nts {
                                            if *len == 1 {
                                                if let Some(next_goto) = parser.table.get(&revealed_state).and_then(|r| r.gotos.get(reduce_nt)) {
                                                    if let Some(next_goto_state) = next_goto.state_id {
                                                        if visited.insert(next_goto_state) {
                                                            worklist.push_back(next_goto_state);
                                                        }
                                                    }
                                                }
                                            } else if *len > 1 {
                                                reduce_char.reveal_and_rereduces.insert((revealed_state, *len - 2, *reduce_nt));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !reduce_char.reveal_and_rereduces.is_empty() || !reduce_char.reveal_goto_shift_escapes.is_empty() {
            char.reduce_characterizations.insert(nt_id, reduce_char);
        }
    }

    char
}
