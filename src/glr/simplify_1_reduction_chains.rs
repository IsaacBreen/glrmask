use std::collections::{BTreeMap, BTreeSet};
use crate::glr::grammar::{NonTerminal, Production, Symbol};
use crate::glr::items::Item;
use crate::glr::table::{ProductionID, Stage6Table};


/// Implements a variant of Pager's algorithm to eliminate unit productions (e.g., A -> B)
/// from a Stage 6 parse table. This can make the parser more efficient by removing
/// chains of reductions.
///
/// The process is iterative:
/// 1. For each state `j` with a unit reduction `A -> B` on lookahead `t`:
/// 2. Find all predecessor states `i` such that `goto(i, B) = j`.
/// 3. For each such `i`, find the target state `k = goto(i, A)`.
/// 4. The actions of state `k` on lookahead `t` are merged into the actions of state `j` on `t`.
/// 5. The original unit reduction `A -> B` is removed from `j`'s actions.
/// 6. This is repeated until no more changes occur.
pub fn simplify_1_reduction_chains(
    stage_6_table: &mut Stage6Table,
    productions: &[Production],
    _start_production_id: usize,
) {
    // 1. Identify all unit productions (A -> B).
    let unit_productions: BTreeMap<ProductionID, (NonTerminal, NonTerminal)> = productions
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if p.rhs.len() == 1 {
                if let Symbol::NonTerminal(ref nt_rhs) = p.rhs[0] {
                    return Some((ProductionID(i), (p.lhs.clone(), nt_rhs.clone())));
                }
            }
            None
        })
        .collect();

    if unit_productions.is_empty() {
        crate::debug!(3, "No unit productions found to eliminate.");
        return; // Nothing to do.
    }
    crate::debug!(3, "Found {} unit productions to analyze for elimination.", unit_productions.len());

    // Create a temporary mapping from state (item set) to a simple ID for logging.
    let state_to_id: BTreeMap<_, _> = stage_6_table.keys().enumerate().map(|(i, s)| (s, i)).collect();

    // 2. Build a reverse goto map for efficient lookup of predecessor states.
    // Map: destination_state -> Vec<(source_state, non_terminal)>
    let mut reverse_gotos: BTreeMap<&BTreeSet<Item>, Vec<(&BTreeSet<Item>, &NonTerminal)>> = BTreeMap::new();
    for (state_i, row_i) in stage_6_table.iter() {
        for (nt_b, state_j) in &row_i.gotos {
            reverse_gotos.entry(state_j).or_default().push((state_i, nt_b));
        }
    }

    // 3. Iteratively apply the elimination rule until no more changes occur.
    loop {
        let mut changed = false;
        let table_clone = stage_6_table.clone(); // Read from a stable clone.

        // Iterate over all states `j` in the table.
        for (state_j, row_j) in stage_6_table.iter_mut() {
            // Iterate over all lookaheads `t` for state `j`.
            for (terminal_t, action_j) in row_j.shifts_and_reduces.iter_mut() {
                
                let pids_in_action: Vec<_> = action_j.reduces.iter().cloned().collect();
                let mut pids_to_remove = BTreeSet::new();
                let mut actions_to_add = Vec::new(); // Collect actions from states `k` to merge.

                // For each reduction in the current action...
                for &pid in &pids_in_action {
                    // Check if it's a unit production `A -> B`.
                    if let Some((nt_a, nt_b)) = unit_productions.get(&pid) {
                        // It is. Mark it for removal.
                        pids_to_remove.insert(pid);

                        // Find all predecessor states `i` such that goto(i, B) = j.
                        if let Some(predecessors) = reverse_gotos.get(state_j) {
                            for (state_i, symbol) in predecessors {
                                if *symbol == nt_b {
                                    // Found a predecessor `i`.
                                    // Find k = goto(i, A).
                                    let row_i = &table_clone[*state_i];
                                    if let Some(state_k) = row_i.gotos.get(nt_a) {
                                        // Found state k.
                                        // Get actions of k on lookahead t.
                                        let row_k = &table_clone[state_k];
                                        if let Some(action_k) = row_k.shifts_and_reduces.get(terminal_t) {
                                            // Schedule to merge actions of k into j.
                                            actions_to_add.push(action_k.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // If we found any unit productions to process...
                if !pids_to_remove.is_empty() {
                    changed = true;

                    // Remove the unit productions from the action.
                    let original_reduce_count = action_j.reduces.len();
                    action_j.reduces.retain(|pid| !pids_to_remove.contains(pid));
                    let removed_count = original_reduce_count - action_j.reduces.len();
                    let state_j_id = state_to_id[state_j];
                    crate::debug!(4, "State {}, Terminal '{}': Removed {} unit reduction(s).", state_j_id, terminal_t, removed_count);

                    // Merge the new actions.
                    for action_to_add in actions_to_add {
                        // Merge reduces
                        let before_len = action_j.reduces.len();
                        action_j.reduces.extend(action_to_add.reduces);
                        if action_j.reduces.len() > before_len {
                             crate::debug!(4, "  -> Added {} new reduction(s).", action_j.reduces.len() - before_len);
                        }

                        // Merge shift
                        if let Some(shift_to_add) = action_to_add.shift {
                            if let Some(existing_shift) = &action_j.shift {
                                if existing_shift != &shift_to_add {
                                    // This is a problem: a shift/shift conflict.
                                    // This shouldn't happen in a valid LR automaton construction,
                                    // as it implies goto(j, t) is ambiguous.
                                    panic!(
                                        "Unit production elimination created a shift/shift conflict on terminal '{}'",
                                        terminal_t
                                    );
                                }
                            } else {
                                action_j.shift = Some(shift_to_add);
                                crate::debug!(4, "  -> Added a new shift action.");
                            }
                        }
                    }
                }
            }
        }

        if !changed {
            crate::debug!(3, "Unit production elimination reached a fixed point.");
            break;
        }
    }
}
