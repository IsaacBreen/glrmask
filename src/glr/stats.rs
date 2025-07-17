use crate::glr::parser::GLRParser;
use crate::glr::table::{Stage7ShiftsAndReduces, StateID};
use std::collections::BTreeMap;
use std::fmt;

/// Contains statistics about a generated GLR parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GLRStats {
    /// Total number of productions in the final grammar used by the parser.
    pub num_productions: usize,
    /// Total number of terminals.
    pub num_terminals: usize,
    /// Total number of non-terminals.
    pub num_non_terminals: usize,
    /// Total number of states in the LR(0) automaton.
    pub num_states: usize,
    /// Number of (state, terminal) pairs with shift-reduce conflicts.
    pub num_shift_reduce_conflicts: usize,
    /// Number of (state, terminal) pairs with reduce-reduce conflicts.
    pub num_reduce_reduce_conflicts: usize,
    /// Maps state ID to the number of (S/R, R/R) conflicts originating from that state.
    pub conflicts_by_state: BTreeMap<StateID, (usize, usize)>,
}

impl fmt::Display for GLRStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "--- GLR Parser Stats ---")?;
        writeln!(f, "Grammar:")?;
        writeln!(f, "  - Productions: {}", self.num_productions)?;
        writeln!(f, "  - Terminals: {}", self.num_terminals)?;
        writeln!(f, "  - Non-Terminals: {}", self.num_non_terminals)?;
        writeln!(f, "Parse Table:")?;
        writeln!(f, "  - States: {}", self.num_states)?;
        writeln!(f, "  - Shift-Reduce Conflicts (by state/terminal): {}", self.num_shift_reduce_conflicts)?;
        writeln!(f, "  - Reduce-Reduce Conflicts (by state/terminal): {}", self.num_reduce_reduce_conflicts)?;
        if !self.conflicts_by_state.is_empty() {
            writeln!(f, "Conflicts by State (StateID: (S/R, R/R)):")?;
            for (state_id, (sr, rr)) in &self.conflicts_by_state {
                if *sr > 0 || *rr > 0 {
                    writeln!(f, "  - State {}: ({}, {})", state_id.0, sr, rr)?;
                }
            }
        }
        writeln!(f, "------------------------")?;
        Ok(())
    }
}

/// Computes various statistics about the generated GLR parser table.
///
/// This function analyzes the final parse table to count states, conflicts, etc.
///
/// # Arguments
/// * `parser` - A reference to the `GLRParser` to be analyzed.
///
/// # Returns
/// A `GLRStats` struct containing the computed statistics.
pub fn get_stats(parser: &GLRParser) -> GLRStats {
    let mut num_shift_reduce_conflicts = 0;
    let mut num_reduce_reduce_conflicts = 0;
    let mut conflicts_by_state: BTreeMap<StateID, (usize, usize)> = BTreeMap::new();

    for (state_id, row) in &parser.stage_7_table {
        let mut sr_conflicts_in_state = 0;
        let mut rr_conflicts_in_state = 0;

        for (_, action) in &row.shifts_and_reduces {
            match action {
                Stage7ShiftsAndReduces::Reduce { production_ids, .. } => {
                    if production_ids.len() > 1 {
                        num_reduce_reduce_conflicts += 1;
                        rr_conflicts_in_state += 1;
                    }
                }
                Stage7ShiftsAndReduces::Split { shift, reduces } => {
                    if shift.is_some() {
                        num_shift_reduce_conflicts += 1;
                        sr_conflicts_in_state += 1;
                    }
                    let total_reduce_productions = reduces.values().flat_map(|nts| nts.values()).map(|pids| pids.len()).sum::<usize>();
                    if total_reduce_productions > 1 {
                        num_reduce_reduce_conflicts += 1;
                        rr_conflicts_in_state += 1;
                    }
                }
                _ => {}
            }
        }

        if sr_conflicts_in_state > 0 || rr_conflicts_in_state > 0 {
            conflicts_by_state.insert(*state_id, (sr_conflicts_in_state, rr_conflicts_in_state));
        }
    }

    GLRStats {
        num_productions: parser.productions.len(),
        num_terminals: parser.terminal_map.len(),
        num_non_terminals: parser.non_terminal_map.len(),
        num_states: parser.item_set_map.len(),
        num_shift_reduce_conflicts,
        num_reduce_reduce_conflicts,
        conflicts_by_state,
    }
}
