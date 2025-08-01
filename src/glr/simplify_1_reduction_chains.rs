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
    // --- 1. Identify unit productions and create helper maps ---
    let unit_productions: BTreeMap<ProductionID, (&NonTerminal, &NonTerminal)> = productions
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if p.rhs.len() == 1 {
                if let Symbol::NonTerminal(nt) = &p.rhs[0] {
                    return Some((ProductionID(i), &p.lhs, nt));
                }
            }
            None
        })
        .collect();

    if unit_productions.is_empty() {
        return;
    }
    let unit_prod_ids: BTreeSet<ProductionID> = unit_productions.keys().cloned().collect();

    // --- 2. Iteratively update actions ---
    loop {
        let mut changed = false;
        let table_clone = stage_6_table.clone();

        for (j_state_items, j_row) in stage_6_table.iter_mut() {
            for (terminal, action) in j_row.shifts_and_reduces.iter_mut() {
                let unit_reduces_to_process: Vec<ProductionID> = action
                    .reduces
                    .iter()
                    .filter(|&pid| unit_prod_ids.contains(pid))
                    .cloned()
                    .collect();

                if unit_reduces_to_process.is_empty() {
                    continue;
                }

                action.reduces.retain(|pid| !unit_prod_ids.contains(pid));
                changed = true;

                for p_id in &unit_reduces_to_process {
                    let (lhs_nt, rhs_nt) = unit_productions[p_id];

                    for (i_state_items, i_row) in &table_clone {
                        if let Some(goto_state) = i_row.gotos.get(rhs_nt) {
                            if goto_state == j_state_items {
                                if let Some(k_state_items) = i_row.gotos.get(lhs_nt) {
                                    if let Some(k_action) = table_clone.get(k_state_items).and_then(|k_row| k_row.shifts_and_reduces.get(terminal)) {
                                        if let Some(k_shift) = &k_action.shift {
                                            if action.shift.is_none() { action.shift = Some(k_shift.clone()); changed = true; } 
                                            else { assert_eq!(action.shift, k_action.shift, "Shift/shift conflict during unit production elimination"); }
                                        }
                                        let pre_merge_len = action.reduces.len();
                                        action.reduces.extend(k_action.reduces.iter().cloned());
                                        if action.reduces.len() != pre_merge_len { changed = true; }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if !changed { break; }
    }
}
