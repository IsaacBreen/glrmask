use crate::glr::parser::GLRParser;
use crate::glr::table::{Stage7ShiftsAndReduces, StateID};
use std::collections::BTreeMap;
use std::fmt;

/// Statistics for a single state in the parse table.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StateStats {
    /// Total number of terminal lookups (actions).
    pub total_actions: usize,
    /// Number of actions that are pure shifts.
    pub num_shifts: usize,
    /// Number of actions that are pure reduces.
    pub num_reduces: usize,
    /// Number of actions that are splits (conflicts).
    pub num_splits: usize,
    /// Number of goto transitions on non-terminals.
    pub num_gotos: usize,
    /// Counts occurrences of each unique action type in the state.
    pub unique_actions: BTreeMap<Stage7ShiftsAndReduces, usize>,
}

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
    /// Detailed statistics for each state.
    pub state_stats: BTreeMap<StateID, StateStats>,
}

impl fmt::Display for GLRStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "--- GLR Parser Stats ---")?;
        writeln!(f, "Grammar: {} productions, {} terminals, {} non-terminals", self.num_productions, self.num_terminals, self.num_non_terminals)?;
        writeln!(f, "Parse Table: {} states, {} S/R conflicts, {} R/R conflicts", self.num_states, self.num_shift_reduce_conflicts, self.num_reduce_reduce_conflicts)?;

        writeln!(f, "\n--- State Details ---")?;
        for (state_id, stats) in &self.state_stats {
            writeln!(
                f,
                "State {:<3} │ Actions: {:<3} (S:{}, R:{}, Sp:{}) │ Gotos: {:<2} │ Unique Actions: {}",
                state_id.0,
                stats.total_actions,
                stats.num_shifts,
                stats.num_reduces,
                stats.num_splits,
                stats.num_gotos,
                stats.unique_actions.len()
            )?;

            const MAX_UNIQUE_TO_DISPLAY: usize = 5;
            if !stats.unique_actions.is_empty() && stats.unique_actions.len() <= MAX_UNIQUE_TO_DISPLAY {
                for (action, count) in &stats.unique_actions {
                    let compact_action = match action {
                        Stage7ShiftsAndReduces::Shift(s) => format!("S→{}", s.0),
                        Stage7ShiftsAndReduces::Reduce { nonterminal_id, len, production_ids } => {
                            format!("R(NT:{},len:{},#p:{})", nonterminal_id.0, len, production_ids.len())
                        }
                        Stage7ShiftsAndReduces::Split { shift, reduces } => {
                            let s_part = shift.map_or("".to_string(), |s| format!("S→{}", s.0));
                            let r_parts: Vec<String> = reduces.iter().map(|(len, nts)| {
                                let nt_count = nts.len();
                                let p_count: usize = nts.values().map(|pids| pids.len()).sum();
                                format!("R(len:{},#nt:{},#p:{})", len, nt_count, p_count)
                            }).collect();

                            if s_part.is_empty() {
                                format!("Split[{}]", r_parts.join(", "))
                            } else {
                                format!("Split[{}, {}]", s_part, r_parts.join(", "))
                            }
                        }
                    };
                    writeln!(f, "          └─ (x{:<2}) {}", count, compact_action)?;
                }
            }
        }
        writeln!(f, "---------------------")?;
        Ok(())
    }
}

/// Computes various statistics about the generated GLR parser table.
pub fn get_stats(parser: &GLRParser) -> GLRStats {
    let mut num_shift_reduce_conflicts = 0;
    let mut num_reduce_reduce_conflicts = 0;
    let mut all_state_stats: BTreeMap<StateID, StateStats> = BTreeMap::new();

    for (state_id, row) in &parser.stage_7_table {
        let mut current_state_stats = StateStats::default();
        current_state_stats.num_gotos = row.gotos.len();
        current_state_stats.total_actions = row.shifts_and_reduces.len();

        for (_, action) in &row.shifts_and_reduces {
            // Update unique actions count
            *current_state_stats.unique_actions.entry(action.clone()).or_insert(0) += 1;

            match action {
                Stage7ShiftsAndReduces::Shift(_) => {
                    current_state_stats.num_shifts += 1;
                }
                Stage7ShiftsAndReduces::Reduce { production_ids, .. } => {
                    current_state_stats.num_reduces += 1;
                    if production_ids.len() > 1 {
                        num_reduce_reduce_conflicts += 1;
                    }
                }
                Stage7ShiftsAndReduces::Split { shift, reduces } => {
                    current_state_stats.num_splits += 1;
                    if shift.is_some() {
                        num_shift_reduce_conflicts += 1;
                    }
                    let total_reduce_productions = reduces.values().flat_map(|nts| nts.values()).map(|pids| pids.len()).sum::<usize>();
                    if total_reduce_productions > 1 {
                        num_reduce_reduce_conflicts += 1;
                    }
                }
            }
        }
        all_state_stats.insert(*state_id, current_state_stats);
    }

    GLRStats {
        num_productions: parser.productions.len(),
        num_terminals: parser.terminal_map.len(),
        num_non_terminals: parser.non_terminal_map.len(),
        num_states: parser.item_set_map.len(),
        num_shift_reduce_conflicts,
        num_reduce_reduce_conflicts,
        state_stats: all_state_stats,
    }
}
