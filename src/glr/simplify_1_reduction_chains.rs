use std::collections::{BTreeMap, BTreeSet, VecDeque};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::table::{ProductionID, Stage6Table, Stage6Row, Stage6ShiftsAndReduces};
use crate::glr::items::Item;

/// Finds the start state of the LR automaton.
/// It assumes the start state is the one containing an item for the first production
/// with the dot at the beginning. This is a heuristic that works for typical augmented grammars.
fn find_start_state<'a>(table: &'a Stage6Table, start_prod: &Production) -> Option<&'a BTreeSet<Item>> {
    table.keys().find(|item_set| {
        item_set.iter().any(|item| {
            item.production == *start_prod && item.dot_position == 0
        })
    })
}

/// Performs a reachability analysis (BFS) to find all states reachable from the start state.
fn compute_reachability(table: &Stage6Table, start_state: &BTreeSet<Item>) -> BTreeSet<BTreeSet<Item>> {
    let mut reachable = BTreeSet::new();
    let mut worklist = VecDeque::new();

    if table.contains_key(start_state) {
        worklist.push_back(start_state.clone());
        reachable.insert(start_state.clone());
    }

    while let Some(state_items) = worklist.pop_front() {
        if let Some(row) = table.get(&state_items) {
            // Successors from shifts
            for action in row.shifts_and_reduces.values() {
                if let Some(target_state) = &action.shift {
                    if reachable.insert(target_state.clone()) {
                        worklist.push_back(target_state.clone());
                    }
                }
            }
            // Successors from gotos
            for target_state in row.gotos.values() {
                if reachable.insert(target_state.clone()) {
                    worklist.push_back(target_state.clone());
                }
            }
        }
    }
    reachable
}

/// Implements Pager's algorithm to eliminate unit productions from a Stage 6 LR parse table.
/// This can reduce the number of states and parsing steps, but may increase table generation time.
pub fn simplify_1_reduction_chains(
    stage_6_table: &mut Stage6Table,
    productions: &[Production],
    start_production_id: usize,
) {
    // --- 1. Pre-computation: Identify unit productions, nodes, leaves, and derivation chains ---
    let unit_productions: Vec<(ProductionID, &Production)> = productions
        .iter()
        .enumerate()
        .filter(|(_, p)| p.rhs.len() == 1)
        .map(|(i, p)| (ProductionID(i), p))
        .collect();

    if unit_productions.is_empty() {
        return; // Nothing to do
    }

    let unit_production_ids: BTreeSet<ProductionID> = unit_productions.iter().map(|(id, _)| *id).collect();

    let mut nodes: BTreeSet<Symbol> = BTreeSet::new();
    let mut rhs_symbols: BTreeSet<Symbol> = BTreeSet::new();
    for (_, p) in &unit_productions {
        nodes.insert(Symbol::NonTerminal(p.lhs.clone()));
        rhs_symbols.insert(p.rhs[0].clone());
    }
    let leaves: BTreeSet<Symbol> = rhs_symbols.difference(&nodes).cloned().collect();

    let mut derives_map: BTreeMap<Symbol, BTreeSet<Symbol>> = BTreeMap::new();
    for s in nodes.iter().chain(leaves.iter()).chain(rhs_symbols.iter()) {
        derives_map.entry(s.clone()).or_default().insert(s.clone()); // Reflexive
    }

    let mut changed = true;
    while changed {
        changed = false;
        for (_, p) in &unit_productions {
            let lhs_sym = Symbol::NonTerminal(p.lhs.clone());
            let rhs_sym = p.rhs[0].clone();
            if let Some(rhs_derives) = derives_map.get(&rhs_sym).cloned() {
                let lhs_derives = derives_map.entry(lhs_sym).or_default();
                let old_len = lhs_derives.len();
                lhs_derives.extend(rhs_derives);
                if lhs_derives.len() != old_len {
                    changed = true;
                }
            }
        }
    }

    // --- 2. Iteratively build the new table with combined states ---
    let mut new_table = Stage6Table::new();
    let mut combination_cache: BTreeMap<BTreeSet<BTreeSet<Item>>, BTreeSet<Item>> = BTreeMap::new();
    let original_states: Vec<_> = stage_6_table.keys().cloned().collect();

    // Create new states by combining existing ones.
    for state_items in &original_states {
        let old_row = stage_6_table.get(state_items).unwrap();
        for leaf in &leaves {
            let sources: Vec<_> = derives_map.iter()
                .filter(|(_, derives)| derives.contains(leaf))
                .map(|(s, _)| s.clone())
                .collect();

            let mut target_states = BTreeSet::new();
            for s in &sources {
                match s {
                    Symbol::Terminal(t) => {
                        if let Some(action) = old_row.shifts_and_reduces.get(t) {
                            if let Some(target) = &action.shift {
                                target_states.insert(target.clone());
                            }
                        }
                    }
                    Symbol::NonTerminal(nt) => {
                        if let Some(target) = old_row.gotos.get(nt) {
                            target_states.insert(target.clone());
                        }
                    }
                }
            }

            if !target_states.is_empty() {
                combination_cache.entry(target_states.clone()).or_insert_with(|| {
                    let mut combined_row = Stage6Row::default();
                    let mut combined_items = BTreeSet::new();

                    for target_state in &target_states {
                        combined_items.extend(target_state.iter().cloned());
                        if let Some(target_row) = stage_6_table.get(target_state) {
                            for (terminal, action) in &target_row.shifts_and_reduces {
                                let new_action = combined_row.shifts_and_reduces.entry(terminal.clone()).or_default();
                                if action.shift.is_some() {
                                    new_action.shift = action.shift.clone();
                                }
                                let proper_reduces = action.reduces.difference(&unit_production_ids).cloned();
                                new_action.reduces.extend(proper_reduces);
                            }
                            for (nt, goto_target) in &target_row.gotos {
                                combined_row.gotos.insert(nt.clone(), goto_target.clone());
                            }
                        }
                    }
                    new_table.insert(combined_items.clone(), combined_row);
                    combined_items
                });
            }
        }
    }

    // Populate the new table with modified transitions.
    for state_items in &original_states {
        let old_row = stage_6_table.get(state_items).unwrap();
        let mut new_row = Stage6Row::default();

        // Copy non-node gotos
        for (nt, target) in &old_row.gotos {
            if !nodes.contains(&Symbol::NonTerminal(nt.clone())) {
                new_row.gotos.insert(nt.clone(), target.clone());
            }
        }

        // Handle shifts and leaf transitions
        let mut handled_terminals = BTreeSet::new();
        for leaf in &leaves {
            if let Symbol::Terminal(leaf_terminal) = leaf {
                let sources: Vec<_> = derives_map.iter()
                    .filter(|(_, derives)| derives.contains(leaf))
                    .map(|(s, _)| s.clone())
                    .collect();

                let mut target_states = BTreeSet::new();
                for s in &sources {
                    if let Symbol::Terminal(t) = s {
                        if let Some(action) = old_row.shifts_and_reduces.get(t) {
                            if let Some(target) = &action.shift {
                                target_states.insert(target.clone());
                                handled_terminals.insert(t.clone());
                            }
                        }
                    }
                }

                if !target_states.is_empty() {
                    let combined_state_items = combination_cache.get(&target_states).unwrap();
                    new_row.shifts_and_reduces.insert(leaf_terminal.clone(), Stage6ShiftsAndReduces {
                        shift: Some(combined_state_items.clone()),
                        reduces: BTreeSet::new(), // Reductions are merged into the combined state's row
                    });
                }
            }
        }

        // Copy remaining shifts and reduces
        for (terminal, action) in &old_row.shifts_and_reduces {
            if !handled_terminals.contains(terminal) {
                new_row.shifts_and_reduces.insert(terminal.clone(), action.clone());
            }
        }
        new_table.insert(state_items.clone(), new_row);
    }

    // --- 3. Remap Gotos for reductions whose LHS is a node ---
    let mut gotos_to_add: BTreeMap<BTreeSet<Item>, Vec<(NonTerminal, BTreeSet<Item>)>> = BTreeMap::new();
    for (state, row) in &new_table {
        for leaf in &leaves {
            if let Symbol::NonTerminal(leaf_nt) = leaf {
                if let Some(target) = row.gotos.get(leaf_nt) {
                    for (node, derives) in &derives_map {
                        if let Symbol::NonTerminal(node_nt) = node {
                            if derives.contains(leaf) && node != leaf {
                                gotos_to_add.entry(state.clone()).or_default().push((node_nt.clone(), target.clone()));
                            }
                        }
                    }
                }
            }
        }
    }
    for (state, additions) in gotos_to_add {
        if let Some(row) = new_table.get_mut(&state) {
            for (nt, target) in additions {
                row.gotos.insert(nt, target);
            }
        }
    }

    // --- 4. Cleanup: Remove unreachable states ---
    let start_state = find_start_state(&new_table, &productions[start_production_id])
        .expect("Start state not found in table")
        .clone();
    let reachable_states = compute_reachability(&new_table, &start_state);
    new_table.retain(|state, _| reachable_states.contains(state));

    *stage_6_table = new_table;
}