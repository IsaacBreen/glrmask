use std::collections::{BTreeMap, BTreeSet};
use crate::glr::grammar::{NonTerminal, Production, Symbol};
use crate::glr::table::{ProductionID, Stage6Table, Stage6ShiftsAndReduces};

pub fn simplify_1_reduction_chains(
    stage_6_table: &mut Stage6Table,
    productions: &[Production],
    _start_production_id: usize, // May not be needed
) {
    // 1. Identify all unit productions (A -> B)
    let unit_productions: BTreeMap<ProductionID, NonTerminal> = productions
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if p.rhs.len() == 1 {
                if let Symbol::NonTerminal(nt) = &p.rhs[0] {
                    return Some((ProductionID(i), nt.clone()));
                }
            }
            None
        })
        .collect();

    if unit_productions.is_empty() {
        return;
    }

    // 2. Iteratively expand actions until a fixed point is reached.
    loop {
        let mut changed = false;
        let read_table = stage_6_table.clone();

        for (current_state_items, current_row) in stage_6_table.iter_mut() {
            // A map to store newly derived actions for the current state.
            // Keyed by lookahead terminal.
            let mut new_actions_for_current_state = BTreeMap::<_, Stage6ShiftsAndReduces>::new();

            // For each existing action in the current state...
            for (lookahead_terminal, action) in &current_row.shifts_and_reduces {
                // ...find any unit reductions.
                for p_id in &action.reduces {
                    if let Some(rhs_non_terminal) = unit_productions.get(p_id) {
                        // This is a unit reduction A -> B, where B is rhs_non_terminal.
                        // We need to find the actions of B in the context of the current state.
                        // These are the actions in the state reached by GOTO(current_state, B).
                        if let Some(goto_b_items) = read_table.get(current_state_items).and_then(|r| r.gotos.get(rhs_non_terminal)) {
                            if let Some(goto_b_row) = read_table.get(goto_b_items) {
                                // The actions of the GOTO state on the same lookahead terminal
                                // should be added to the current state's actions.
                                if let Some(actions_from_goto_state) = goto_b_row.shifts_and_reduces.get(lookahead_terminal) {
                                    let entry = new_actions_for_current_state
                                        .entry(lookahead_terminal.clone())
                                        .or_default();

                                    // Merge shift action
                                    if let Some(shift_items) = &actions_from_goto_state.shift {
                                        if let Some(existing_shift) = &entry.shift {
                                            if existing_shift != shift_items {
                                                // This indicates a shift/shift conflict introduced by eliminating unit productions.
                                                // This shouldn't happen in a valid LR(1) grammar, but we should be aware of it.
                                                panic!("Shift/shift conflict introduced during unit production elimination");
                                            }
                                        } else {
                                            entry.shift = Some(shift_items.clone());
                                        }
                                    }

                                    // Merge reduce actions
                                    entry.reduces.extend(actions_from_goto_state.reduces.iter().cloned());
                                }
                            }
                        }
                    }
                }
            }

            // If we found new actions, merge them into the current row.
            if !new_actions_for_current_state.is_empty() {
                for (lookahead_terminal, new_action_part) in new_actions_for_current_state {
                    let entry = current_row.shifts_and_reduces.entry(lookahead_terminal).or_default();

                    // Merge shift
                    if let Some(new_shift) = new_action_part.shift {
                        if entry.shift.is_none() {
                            entry.shift = Some(new_shift);
                            changed = true;
                        }
                    }

                    // Merge reduces
                    let old_reduces_len = entry.reduces.len();
                    entry.reduces.extend(new_action_part.reduces);
                    if entry.reduces.len() != old_reduces_len {
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    // 3. Final cleanup: remove all unit production IDs from all reduce sets.
    for row in stage_6_table.values_mut() {
        for action in row.shifts_and_reduces.values_mut() {
            action.reduces.retain(|p_id| !unit_productions.contains_key(p_id));
        }
    }
}