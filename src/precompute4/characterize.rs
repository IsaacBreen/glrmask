use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::{self, Display, Formatter};

use crate::glr::parser::GLRParser;
use crate::glr::table::{get_row, iter_rows, NonTerminalID, Stage7ShiftsAndReducesLookaheadValue, StateID, TerminalID};

/// (initial_state, shift_state)
type InitialShift = (StateID, StateID);
/// (initial_state, reduction_len_minus_one, reduced_nonterminal)
type InitialReduce = (StateID, usize, NonTerminalID);
/// (revealed_state, remaining_len_minus_one, reduced_nonterminal)
type RevealAndRereduce = (StateID, usize, NonTerminalID);
/// (revealed_state, goto_state, shift_state)
type RevealGotoShiftEscape = (StateID, StateID, StateID);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReduceCharacterization {
    pub terminal: TerminalID,
    pub nonterminal: NonTerminalID,
    pub reveal_and_rereduces: BTreeSet<RevealAndRereduce>,
    pub reveal_goto_shift_escapes: BTreeSet<RevealGotoShiftEscape>,
}

impl ReduceCharacterization {
    fn new(terminal: TerminalID, nonterminal: NonTerminalID) -> Self {
        Self {
            terminal,
            nonterminal,
            reveal_and_rereduces: BTreeSet::new(),
            reveal_goto_shift_escapes: BTreeSet::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.reveal_and_rereduces.is_empty() && self.reveal_goto_shift_escapes.is_empty()
    }
}

impl Display for ReduceCharacterization {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "    Reduce Char for NT {}:", self.nonterminal.0)?;
        if !self.reveal_and_rereduces.is_empty() {
            writeln!(f, "      Reveal-and-rereduces:")?;
            for (revealed, len, nt) in &self.reveal_and_rereduces {
                writeln!(f, "        - revealed: {}, len: {}, reduce_nt: {}", revealed.0, len, nt.0)?;
            }
        }
        if !self.reveal_goto_shift_escapes.is_empty() {
            writeln!(f, "      Reveal-goto-shift escapes:")?;
            for (revealed, goto, shift) in &self.reveal_goto_shift_escapes {
                writeln!(f, "        - revealed: {}, goto: {}, shift: {}", revealed.0, goto.0, shift.0)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BelowBottomCharacterization {
    pub terminal: TerminalID,
    pub initial_shifts: BTreeSet<InitialShift>,
    pub initial_reduces: BTreeSet<InitialReduce>,
    pub reduce_characterizations: BTreeMap<NonTerminalID, ReduceCharacterization>,
    pub all_nts: BTreeSet<NonTerminalID>,
}

impl Display for BelowBottomCharacterization {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Characterization for Terminal {}:", self.terminal.0)?;
        if !self.initial_shifts.is_empty() {
            writeln!(f, "  Initial Shifts:")?;
            for (initial, shift) in &self.initial_shifts {
                writeln!(f, "    - initial: {}, shift: {}", initial.0, shift.0)?;
            }
        }
        if !self.initial_reduces.is_empty() {
            writeln!(f, "  Initial Reduces:")?;
            for (initial, len, nt) in &self.initial_reduces {
                writeln!(f, "    - initial: {}, len: {}, nt: {}", initial.0, len, nt.0)?;
            }
        }
        if !self.reduce_characterizations.is_empty() {
            writeln!(f, "  Reduce Characterizations:")?;
            for rc in self.reduce_characterizations.values() {
                write!(f, "{rc}")?;
            }
        }
        Ok(())
    }
}

pub fn compute_all_characterizations(parser: &GLRParser) -> BTreeMap<TerminalID, BelowBottomCharacterization> {
    parser
        .terminal_map
        .right_values()
        .cloned()
        .map(|terminal_id| (terminal_id, compute_below_bottom_characterization(parser, terminal_id)))
        .collect()
}

pub fn compute_below_bottom_characterization(parser: &GLRParser, terminal_id: TerminalID) -> BelowBottomCharacterization {
    let all_nts: BTreeSet<_> = parser.non_terminal_map.right_values().cloned().collect();
    let (initial_shifts, initial_reduces) = collect_initial_actions(parser, terminal_id);
    let reduce_characterizations = collect_reduce_characterizations(parser, terminal_id);

    let result = BelowBottomCharacterization {
        terminal: terminal_id,
        initial_shifts,
        initial_reduces,
        reduce_characterizations,
        all_nts,
    };

    crate::debug!(5, "Computed Below-Bottom Characterization for terminal {}:\n{}", terminal_id.0, result);
    result
}

fn collect_initial_actions(
    parser: &GLRParser,
    terminal_id: TerminalID,
) -> (BTreeSet<InitialShift>, BTreeSet<InitialReduce>) {
    use Stage7ShiftsAndReducesLookaheadValue::*;

    let mut initial_shifts = BTreeSet::new();
    let mut initial_reduces = BTreeSet::new();

    for (&initial_state, row) in iter_rows(&parser.table) {
        if let Some(action) = row.get_shifts_and_reduces_for_terminal(&terminal_id) {
            match action {
                Shift(shift_state) => {
                    initial_shifts.insert((initial_state, shift_state));
                }
                Reduce { nonterminal_id, len, .. } => {
                    if len > 0 {
                        initial_reduces.insert((initial_state, len - 1, nonterminal_id));
                    }
                }
                Split { shift, reduces } => {
                    if let Some(shift_state) = shift {
                        initial_shifts.insert((initial_state, shift_state));
                    }
                    for (len, nts) in reduces {
                        if len > 0 {
                            for (nt_id, _) in nts {
                                initial_reduces.insert((initial_state, len - 1, nt_id));
                            }
                        }
                    }
                }
            }
        }
    }

    (initial_shifts, initial_reduces)
}

fn collect_reduce_characterizations(
    parser: &GLRParser,
    terminal_id: TerminalID,
) -> BTreeMap<NonTerminalID, ReduceCharacterization> {
    let mut result: BTreeMap<NonTerminalID, ReduceCharacterization> = BTreeMap::new();

    for (revealed_state, row) in iter_rows(&parser.table) {
        for (&nt_id, goto) in &row.gotos {
            if let Some(goto_state) = goto.state_id {
                let reduce_char = result.entry(nt_id).or_insert_with(|| ReduceCharacterization::new(terminal_id, nt_id));
                explore_from_goto(parser, terminal_id, *revealed_state, goto_state, reduce_char);
            }
        }
    }

    result.retain(|_, rc| !rc.is_empty());
    result
}

fn explore_from_goto(
    parser: &GLRParser,
    terminal_id: TerminalID,
    revealed_state: StateID,
    start_state: StateID,
    reduce_char: &mut ReduceCharacterization,
) {
    use Stage7ShiftsAndReducesLookaheadValue::*;

    let mut worklist = VecDeque::new();
    let mut visited = BTreeSet::new();

    visited.insert(start_state);
    worklist.push_back(start_state);

    while let Some(current_state) = worklist.pop_front() {
        let Some(row) = get_row(&parser.table, current_state) else { continue };
        let Some(action) = row.get_shifts_and_reduces_for_terminal(&terminal_id) else { continue };

        match action {
            Shift(shift_state) => {
                reduce_char.reveal_goto_shift_escapes.insert((revealed_state, current_state, shift_state));
            }
            Reduce { nonterminal_id: reduce_nt, len, .. } => {
                handle_reduce(parser, revealed_state, len, reduce_nt, &mut visited, &mut worklist, reduce_char);
            }
            Split { shift, reduces } => {
                if let Some(shift_state) = shift {
                    reduce_char.reveal_goto_shift_escapes.insert((revealed_state, current_state, shift_state));
                }
                for (len, nts) in reduces {
                    for (reduce_nt, _) in nts {
                        handle_reduce(parser, revealed_state, len, reduce_nt, &mut visited, &mut worklist, reduce_char);
                    }
                }
            }
        }
    }
}

fn handle_reduce(
    parser: &GLRParser,
    revealed_state: StateID,
    len: usize,
    reduce_nt: NonTerminalID,
    visited: &mut BTreeSet<StateID>,
    worklist: &mut VecDeque<StateID>,
    reduce_char: &mut ReduceCharacterization,
) {
    if len == 1 {
        if let Some(next_goto_state) = get_row(&parser.table, revealed_state)
            .and_then(|row| row.gotos.get(&reduce_nt))
            .and_then(|goto| goto.state_id)
        {
            if visited.insert(next_goto_state) {
                worklist.push_back(next_goto_state);
            }
        }
    } else if len > 1 {
        reduce_char.reveal_and_rereduces.insert((revealed_state, len - 2, reduce_nt));
    }
}
