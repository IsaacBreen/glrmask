use std::collections::{BTreeMap, BTreeSet};
use bimap::BiBTreeMap;
use crate::glr::grammar::{NonTerminal, Production, Symbol};
use crate::glr::items::Item;
use crate::glr::table::{NonTerminalID, Stage6Table};

/// Implements a simplified version of Pager's algorithm to eliminate unit productions.
/// This function modifies the Stage 6 table in place.
pub fn eliminate_unit_productions(
    stage_6_table: &mut Stage6Table,
    productions: &[Production],
    non_terminal_map: &mut BiBTreeMap<NonTerminal, NonTerminalID>,
) {
    // 1. Identify unit productions, nodes, and leaves.
    let mut unit_productions = BTreeSet::new();
    let mut nodes = BTreeSet::new();
    let mut leaves = BTreeSet::new();
    let mut non_terminals = BTreeSet::new();

    for p in productions {
        non_terminals.insert(p.lhs.clone());
        if p.rhs.len() == 1 {
            if let Symbol::NonTerminal(rhs_nt) = &p.rhs[0] {
                unit_productions.insert((p.lhs.clone(), rhs_nt.clone()));
                nodes.insert(p.lhs.clone());
            }
        }
    }

    for nt in non_terminals {
        if !nodes.contains(&nt) {
            leaves.insert(nt);
        }
    }

    // 2. Compute descendant chains for each node.
    let mut descendants: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
    for (lhs, rhs) in &unit_productions {
        descendants.entry(lhs.clone()).or_default().insert(rhs.clone());
    }

    let mut changed = true;
    while changed {
        changed = false;
        for (node, descs) in descendants.clone() {
            for desc in descs {
                if let Some(next_descs) = descendants.get(&desc) {
                    let current_len = descendants.get(&node).unwrap().len();
                    descendants.get_mut(&node).unwrap().extend(next_descs.clone());
                    if descendants.get(&node).unwrap().len() != current_len {
                        changed = true;
                    }
                }
            }
        }
    }

    crate::debug!(3, "Unit production nodes: {:?}", nodes);
    crate::debug!(3, "Unit production leaves: {:?}", leaves);
    crate::debug!(4, "Unit production descendants: {:?}", descendants);

    // 3. Iteratively merge states.
    let mut changed = true;
    while changed {
        changed = false;
        let mut new_table = stage_6_table.clone();
        let mut merges: BTreeMap<BTreeSet<Item>, BTreeSet<Item>> = BTreeMap::new();

        for (item_set, row) in stage_6_table.iter() {
            for (nt, goto_target_set) in &row.gotos {
                // Check if this goto leads to a state that is a simple unit reduction.
                let is_unit_reduction_state = stage_6_table.get(goto_target_set)
                    .map_or(false, |target_row| {
                        target_row.shifts_and_reduces.is_empty() &&
                            target_row.gotos.is_empty() &&
                            goto_target_set.len() == 1 &&
                            goto_target_set.iter().next().unwrap().dot_at_end() &&
                            unit_productions.contains(&(
                                nt.clone(),
                                if let Symbol::NonTerminal(rhs_nt) = &goto_target_set.iter().next().unwrap().production.rhs[0] {
                                    rhs_nt.clone()
                                } else {
                                    // Should not happen for unit productions
                                    return false;
                                }))
                    });

                if is_unit_reduction_state {
                    let reduction_item = goto_target_set.iter().next().unwrap();
                    if let Symbol::NonTerminal(descendant_nt) = &reduction_item.production.rhs[0] {
                        // This is a reduction from nt -> descendant_nt.
                        // We need to merge this state with the state we would go to on descendant_nt.
                        if let Some(final_target_set) = row.gotos.get(descendant_nt) {
                            if goto_target_set != final_target_set {
                                crate::debug!(4, "Merging state for reduction {} -> {}", nt, descendant_nt);
                                merges.insert(goto_target_set.clone(), final_target_set.clone());
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        if changed {
            // Apply merges
            for (from, to) in &merges {
                if let Some(from_row) = new_table.remove(&from) {
                    let to_row = new_table.get_mut(&to).unwrap();
                    // Merge `from_row` into `to_row`. For Stage6, this is complex.
                    // A simpler approach for now is just to update transitions.
                }
            }

            // Update transitions
            for row in new_table.values_mut() {
                for goto_target_set in row.gotos.values_mut() {
                    if let Some(new_target) = merges.get(goto_target_set) {
                        *goto_target_set = new_target.clone();
                    }
                }
            }
            *stage_6_table = new_table;
        }
    }

    // 4. Remove goto transitions on node non-terminals.
    for row in stage_6_table.values_mut() {
        row.gotos.retain(|nt, _| !nodes.contains(nt));
    }

    // 5. Remove node non-terminals from the map so they don't get IDs in stage 7.
    for node in nodes {
        non_terminal_map.remove_by_left(&node);
    }
}
