use std::collections::{BTreeMap, BTreeSet};
use crate::glr::grammar::{Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{ProductionID, Stage6Table};

pub fn simplify_1_reduction_chains(
    stage_6_table: &mut Stage6Table,
    productions: &[Production],
    _start_production_id: usize,
) {
    // This function implements unit-production elimination by propagating non-unit reductions
    // backward through unit-reduction chains in the parse table.

    // 1. Identify all unit productions. A unit production is of the form A -> B.
    let unit_production_ids: BTreeSet<ProductionID> = productions
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if p.rhs.len() == 1 && matches!(p.rhs[0], Symbol::NonTerminal(_)) {
                Some(ProductionID(i))
            } else {
                None
            }
        })
        .collect();

    if unit_production_ids.is_empty() {
        return; // No unit productions to eliminate.
    }

    // 2. For each (state, lookahead) pair, compute the set of non-unit productions
    //    that are ultimately reachable through chains of unit reductions.
    let mut final_reductions: BTreeMap<(BTreeSet<Item>, Terminal), BTreeSet<ProductionID>> = BTreeMap::new();

    // Initialize with the direct non-unit reductions for each state and lookahead.
    for (item_set, row) in stage_6_table.iter() {
        for (terminal, action) in &row.shifts_and_reduces {
            let non_unit_reduces: BTreeSet<_> = action.reduces.iter()
                .filter(|pid| !unit_production_ids.contains(pid))
                .cloned()
                .collect();
            if !non_unit_reduces.is_empty() {
                final_reductions.insert((item_set.clone(), terminal.clone()), non_unit_reduces);
            }
        }
    }

    // 3. Iteratively propagate these final reductions backward through unit-reduction links
    //    until a fixed point is reached.
    let mut changed = true;
    while changed {
        changed = false;
        for (item_set, row) in stage_6_table.iter() {
            for (terminal, action) in &row.shifts_and_reduces {
                let mut newly_added_reductions = BTreeSet::new();
                for pid in &action.reduces {
                    if unit_production_ids.contains(pid) {
                        // This is a unit production A -> B.
                        let prod = &productions[pid.0];
                        let lhs = &prod.lhs;
                        // After reducing by A -> B, the parser would GOTO on A from the current state.
                        if let Some(goto_state) = row.gotos.get(lhs) {
                            if let Some(target_reduces) = final_reductions.get(&(goto_state.clone(), terminal.clone())) {
                                newly_added_reductions.extend(target_reduces.iter().cloned());
                            }
                        }
                    }
                }

                if !newly_added_reductions.is_empty() {
                    let current_reduces = final_reductions.entry((item_set.clone(), terminal.clone())).or_default();
                    let old_len = current_reduces.len();
                    current_reduces.extend(newly_added_reductions);
                    if current_reduces.len() > old_len {
                        changed = true;
                    }
                }
            }
        }
    }

    // 4. Update the table: remove all unit reductions and add the computed final non-unit reductions.
    for (item_set, row) in stage_6_table.iter_mut() {
        for (terminal, action) in row.shifts_and_reduces.iter_mut() {
            // Remove all unit reductions.
            action.reduces.retain(|pid| !unit_production_ids.contains(pid));
            // Add the computed non-unit reductions.
            if let Some(final_reds) = final_reductions.get(&(item_set.clone(), terminal.clone())) {
                action.reduces.extend(final_reds);
            }
        }
    }
}
