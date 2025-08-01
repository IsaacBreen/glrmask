use std::collections::{BTreeMap, BTreeSet};
use crate::glr::grammar::{NonTerminal, Production, Symbol};
use crate::glr::items::Item;
use crate::glr::table::{ProductionID, Stage6Row, Stage6ShiftsAndReduces, Stage6Table};

pub fn simplify_1_reduction_chains(
    stage_6_table: &mut Stage6Table,
    productions: &[Production],
    start_production_id: usize,
) {
    let production_map: BTreeMap<_, _> = productions.iter().enumerate().map(|(i, p)| (ProductionID(i), p)).collect();

    // 1. Identify unit productions (A -> B)
    let unit_productions: BTreeMap<ProductionID, (&NonTerminal, &NonTerminal)> = production_map
        .iter()
        .filter_map(|(&id, p)| {
            if p.rhs.len() == 1 {
                if let Symbol::NonTerminal(nt) = &p.rhs[0] {
                    return Some((id, (&p.lhs, nt)));
                }
            }
            None
        })
        .collect();

    if unit_productions.is_empty() {
        return;
    }

    // 2. Build descendant/ancestor relationships
    let mut descendants: BTreeMap<&NonTerminal, BTreeSet<&NonTerminal>> = BTreeMap::new();
    let mut ancestors: BTreeMap<&NonTerminal, BTreeSet<&NonTerminal>> = BTreeMap::new();
    let all_non_terminals: BTreeSet<_> = productions.iter().map(|p| &p.lhs).collect();

    for nt in &all_non_terminals {
        descendants.entry(nt).or_default().insert(nt);
        ancestors.entry(nt).or_default().insert(nt);
    }

    for (_, (lhs, rhs)) in &unit_productions {
        descendants.get_mut(lhs).unwrap().insert(rhs);
        ancestors.get_mut(rhs).unwrap().insert(lhs);
    }

    // Transitive closure for descendants
    let mut changed = true;
    while changed {
        changed = false;
        for nt in &all_non_terminals {
            let current_descendants = descendants.get(nt).unwrap().clone();
            for d in current_descendants {
                let new_descendants = descendants.get(d).unwrap().clone();
                let mut my_descendants = descendants.get_mut(nt).unwrap();
                let old_len = my_descendants.len();
                my_descendants.extend(new_descendants);
                if my_descendants.len() != old_len {
                    changed = true;
                }
            }
        }
    }

    let nodes: BTreeSet<_> = unit_productions.values().map(|(lhs, _)| *lhs).collect();
    let leaves: BTreeSet<_> = all_non_terminals.iter().filter(|nt| !nodes.contains(*nt)).cloned().collect();

    // 3. Iteratively merge states
    let mut new_states: BTreeMap<BTreeSet<BTreeSet<Item>>, BTreeSet<Item>> = BTreeMap::new();
    let mut changed = true;

    while changed {
        changed = false;
        let current_table = stage_6_table.clone();
        let original_keys: Vec<_> = current_table.keys().cloned().collect();

        for item_set in &original_keys {
            let row = current_table.get(item_set).unwrap();
            for leaf in &leaves {
                if let Some(descendants_of_leaf) = descendants.get(leaf) {
                    // Find all transitions from this state on NTs that can derive `leaf`
                    let states_to_merge_keys: BTreeSet<_> = row.gotos.iter()
                        .filter(|(nt, _)| descendants_of_leaf.contains(nt))
                        .map(|(_, target_set)| target_set.clone())
                        .collect();

                    if states_to_merge_keys.len() > 1 {
                        // Check if any of the target states have a unit reduction
                        let has_unit_reduction = states_to_merge_keys.iter().any(|target_set| {
                            current_table.get(target_set).map_or(false, |target_row| {
                                target_row.shifts_and_reduces.values().any(|action| {
                                    !action.reduces.is_empty() && action.reduces.iter().any(|pid| unit_productions.contains_key(pid))
                                })
                            })
                        });

                        if !has_unit_reduction { continue; }

                        // Merge states
                        let merged_state_key = states_to_merge_keys.clone();
                        let merged_row = if let Some(cached_set) = new_states.get(&merged_state_key) {
                            stage_6_table.get(cached_set).unwrap().clone()
                        } else {
                            let mut new_row = Stage6Row::default();
                            let mut new_item_set = BTreeSet::new();
                            for key in &states_to_merge_keys {
                                if let Some(row_to_merge) = current_table.get(key) {
                                    new_item_set.extend(key.iter().cloned());
                                    // Merge shifts_and_reduces
                                    for (term, action) in &row_to_merge.shifts_and_reduces {
                                        let entry = new_row.shifts_and_reduces.entry(term.clone()).or_default();
                                        if let Some(shift_set) = &action.shift {
                                            entry.shift.get_or_insert_with(BTreeSet::new).extend(shift_set.iter().cloned());
                                        }
                                        entry.reduces.extend(action.reduces.iter());
                                    }
                                    // Merge gotos
                                    for (nt, goto_set) in &row_to_merge.gotos {
                                        new_row.gotos.entry(nt.clone()).or_default().extend(goto_set.iter().cloned());
                                    }
                                }
                            }
                            new_states.insert(merged_state_key, new_item_set.clone());
                            stage_6_table.insert(new_item_set, new_row.clone());
                            changed = true;
                            new_row
                        };

                        // Update transitions in the original row
                        let mut original_row = stage_6_table.get_mut(item_set).unwrap();
                        let merged_item_set = new_states.get(&states_to_merge_keys).unwrap();
                        for (nt, goto_set) in original_row.gotos.iter_mut() {
                            if states_to_merge_keys.contains(goto_set) {
                                *goto_set = merged_item_set.clone();
                            }
                        }
                    }
                }
            }
        }
    }

    // 4. Clean up: Remove transitions on "node" non-terminals and update reduces
    let final_table = std::mem::take(stage_6_table);
    for (item_set, mut row) in final_table {
        // Remove gotos on node non-terminals
        row.gotos.retain(|nt, _| !nodes.contains(nt));

        // Update reduces to point to leaf productions
        for action in row.shifts_and_reduces.values_mut() {
            let mut new_reduces = BTreeSet::new();
            let mut changed_reduces = false;
            for &pid in &action.reduces {
                if let Some((lhs, _)) = unit_productions.get(&pid) {
                    // This is a unit production, find its leaf equivalent
                    let leaf_descendants: Vec<_> = descendants.get(lhs).unwrap().iter().filter(|d| leaves.contains(*d)).collect();
                    if let Some(&&leaf) = leaf_descendants.first() {
                         // Find a non-unit production for the leaf to substitute
                        if let Some((new_pid, _)) = production_map.iter().find(|(_, p)| p.lhs == *leaf && !unit_productions.contains_key(p.0)) {
                             new_reduces.insert(*new_pid);
                             changed_reduces = true;
                        } else {
                             new_reduces.insert(pid); // Keep original if no substitute found
                        }
                    } else {
                        new_reduces.insert(pid); // Keep original if no leaf found
                    }
                } else {
                    new_reduces.insert(pid);
                }
            }
            if changed_reduces {
                action.reduces = new_reduces;
            }
        }
        stage_6_table.insert(item_set, row);
    }
}
