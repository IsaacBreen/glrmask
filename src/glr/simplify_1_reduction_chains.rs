use std::collections::{BTreeMap, BTreeSet, VecDeque};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{ProductionID, Stage6Table, Stage6Row, Stage6ShiftsAndReduces, StateID};
use bimap::BiBTreeMap;

/// Helper struct for the intermediate representation of the table during transformation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct IntermediateRow {
    shifts_and_reduces: BTreeMap<Terminal, IntermediateAction>,
    gotos: BTreeMap<NonTerminal, StateID>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct IntermediateAction {
    shift: Option<StateID>,
    reduces: BTreeSet<ProductionID>,
}

/// Analyzes the grammar to identify unit productions, nodes, leaves, and computes
/// the transitive closure of the unit production relation.
///
/// A "unit production" is defined as `A -> B` where both A and B are non-terminals.
/// A "node" is a non-terminal on the LHS of a unit production.
/// A "leaf" is a non-terminal on the RHS of a unit production that is not a node.
fn analyze_grammar(productions: &[Production]) -> (
    BTreeSet<ProductionID>, // unit_productions (A -> B)
    BTreeSet<NonTerminal>,  // nodes
    BTreeSet<NonTerminal>,  // leaves
    BTreeMap<NonTerminal, BTreeSet<NonTerminal>>, // derives_unit_star (transitive closure)
    BTreeSet<ProductionID>, // non_unit_productions
) {
    let mut unit_productions = BTreeSet::new();
    let mut non_unit_productions = BTreeSet::new();
    let mut derives_unit = BTreeMap::<NonTerminal, BTreeSet<NonTerminal>>::new();
    let mut all_nonterminals = BTreeSet::<NonTerminal>::new();

    for (i, p) in productions.iter().enumerate() {
        all_nonterminals.insert(p.lhs.clone());
        for s in &p.rhs {
            if let Symbol::NonTerminal(nt) = s {
                all_nonterminals.insert(nt.clone());
            }
        }

        if p.rhs.len() == 1 {
            if let Symbol::NonTerminal(rhs_nt) = &p.rhs[0] {
                unit_productions.insert(ProductionID(i));
                derives_unit.entry(p.lhs.clone()).or_default().insert(rhs_nt.clone());
            } else {
                non_unit_productions.insert(ProductionID(i));
            }
        } else {
            non_unit_productions.insert(ProductionID(i));
        }
    }

    let nodes: BTreeSet<NonTerminal> = derives_unit.keys().cloned().collect();
    let mut leaves = BTreeSet::new();
    for rhs_nts in derives_unit.values() {
        for nt in rhs_nts {
            if !nodes.contains(nt) {
                leaves.insert(nt.clone());
            }
        }
    }

    // Compute transitive-reflexive closure for derives_unit
    let mut derives_unit_star: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
    for nt in &all_nonterminals {
        derives_unit_star.entry(nt.clone()).or_default().insert(nt.clone());
    }
    for (s1, s2_set) in &derives_unit {
        derives_unit_star.entry(s1.clone()).or_default().extend(s2_set.iter().cloned());
    }

    let mut changed = true;
    while changed {
        changed = false;
        let keys: Vec<NonTerminal> = derives_unit_star.keys().cloned().collect();
        for k in keys {
            let derives_k = derives_unit_star.get(&k).unwrap().clone();
            let mut new_derives = BTreeSet::new();
            for d in derives_k {
                if let Some(derives_d) = derives_unit_star.get(&d) {
                    new_derives.extend(derives_d.iter().cloned());
                }
            }
            let current_set = derives_unit_star.get_mut(&k).unwrap();
            let old_len = current_set.len();
            current_set.extend(new_derives);
            if current_set.len() != old_len {
                changed = true;
            }
        }
    }

    (unit_productions, nodes, leaves, derives_unit_star, non_unit_productions)
}

/// Converts the `Stage6Table` into a more manageable intermediate representation
/// where states are identified by simple `StateID`s.
fn build_intermediate_representation(
    stage_6_table: &Stage6Table,
    productions: &[Production],
) -> (
    BTreeMap<StateID, IntermediateRow>,
    BiBTreeMap<BTreeSet<Item>, StateID>,
    BTreeMap<StateID, BTreeSet<Item>>,
    StateID,
) {
    let mut item_set_to_id = BiBTreeMap::new();
    let mut next_state_id = 0;

    // Find start state by looking for the item corresponding to the augmented start production.
    // Assumes productions[0] is the augmented start production (e.g., S' -> S).
    let start_prod_lhs = &productions[0].lhs;
    let start_item_set = stage_6_table.keys().find(|items| {
        items.iter().any(|item| &item.production.lhs == start_prod_lhs && item.dot_position == 0)
    }).expect("Start state not found in table");

    let mut sorted_keys: Vec<_> = stage_6_table.keys().cloned().collect();
    sorted_keys.sort(); // Ensure deterministic ID assignment

    for item_set in sorted_keys {
        if !item_set_to_id.contains_left(&item_set) {
            item_set_to_id.insert(item_set, StateID(next_state_id));
            next_state_id += 1;
        }
    }

    let start_state_id = *item_set_to_id.get_by_left(start_item_set).unwrap();

    let mut intermediate_table = BTreeMap::new();
    for (item_set, row) in stage_6_table {
        let state_id = *item_set_to_id.get_by_left(item_set).unwrap();
        let mut new_row = IntermediateRow::default();

        for (terminal, action) in &row.shifts_and_reduces {
            let new_action = IntermediateAction {
                shift: action.shift.as_ref().map(|s| *item_set_to_id.get_by_left(s).unwrap()),
                reduces: action.reduces.clone(),
            };
            new_row.shifts_and_reduces.insert(terminal.clone(), new_action);
        }

        for (nt, goto_set) in &row.gotos {
            new_row.gotos.insert(nt.clone(), *item_set_to_id.get_by_left(goto_set).unwrap());
        }
        intermediate_table.insert(state_id, new_row);
    }

    let id_to_item_set = item_set_to_id.iter().map(|(k, v)| (*v, k.clone())).collect();

    (intermediate_table, item_set_to_id, id_to_item_set, start_state_id)
}

/// Performs a graph traversal to find all states reachable from the start state.
fn find_reachable_states(start_id: StateID, table: &BTreeMap<StateID, IntermediateRow>) -> BTreeSet<StateID> {
    let mut reachable = BTreeSet::new();
    let mut worklist = VecDeque::from([start_id]);
    reachable.insert(start_id);

    while let Some(s_id) = worklist.pop_front() {
        if let Some(row) = table.get(&s_id) {
            for action in row.shifts_and_reduces.values() {
                if let Some(next_id) = action.shift {
                    if reachable.insert(next_id) {
                        worklist.push_back(next_id);
                    }
                }
            }
            for &next_id in row.gotos.values() {
                if reachable.insert(next_id) {
                    worklist.push_back(next_id);
                }
            }
        }
    }
    reachable
}

/// Implements Pager's algorithm to eliminate unit productions from an LR parsing table.
///
/// The algorithm works by:
/// 1. Identifying unit productions (`A -> B`), nodes (LHS of unit prods), and leaves (RHS of unit prods that are not nodes).
/// 2. Creating new "combined" states that merge the actions of states reachable via a chain of unit productions.
/// 3. Rerouting transitions on nodes to transitions on leaves that point to these new combined states.
/// 4. Removing original transitions on nodes.
/// 5. Remapping the LHS of non-unit productions from nodes to corresponding leaves.
/// 6. Pruning any states that become unreachable after these transformations.
pub fn simplify_1_reduction_chains(
    stage_6_table: &mut Stage6Table,
    productions: &mut Vec<Production>,
) {
    if stage_6_table.is_empty() {
        return;
    }

    // --- 1. Analyze Grammar ---
    let (unit_productions, nodes, leaves, derives_unit_star, non_unit_productions) =
        analyze_grammar(productions);

    if unit_productions.is_empty() {
        return; // Nothing to do
    }

    // --- 2. Build Intermediate Representation ---
    let (
        mut intermediate_table,
        _item_set_to_id,
        mut id_to_item_set,
        start_state_id,
    ) = build_intermediate_representation(stage_6_table, productions);

    // --- 3. Main Loop: Create Combined States ---
    let mut worklist: VecDeque<StateID> = intermediate_table.keys().cloned().collect();
    let mut combined_states: BTreeMap<BTreeSet<StateID>, StateID> = BTreeMap::new();
    let mut next_state_id = intermediate_table.len();
    let mut processed_states = BTreeSet::new();

    while let Some(s_id) = worklist.pop_front() {
        if !processed_states.insert(s_id) {
            continue;
        }

        let mut s_row = intermediate_table.get(&s_id).unwrap().clone();

        for leaf in &leaves {
            let symbols_deriving_leaf: BTreeSet<&NonTerminal> = derives_unit_star
                .iter()
                .filter(|(_, derives)| derives.contains(leaf))
                .map(|(s, _)| s)
                .collect();

            let mut t_set = BTreeSet::new();
            for nt in symbols_deriving_leaf {
                if let Some(goto_id) = s_row.gotos.get(nt) {
                    t_set.insert(*goto_id);
                }
            }

            if t_set.is_empty() {
                continue;
            }

            let t_new_id = *combined_states.entry(t_set.clone()).or_insert_with(|| {
                let new_id = StateID(next_state_id);
                next_state_id += 1;

                let mut new_row = IntermediateRow::default();
                for &t_id in &t_set {
                    if let Some(t_row) = intermediate_table.get(&t_id) {
                        for (terminal, action) in &t_row.shifts_and_reduces {
                            let new_action = new_row.shifts_and_reduces.entry(terminal.clone()).or_default();
                            if action.shift.is_some() { new_action.shift = action.shift; }
                            for &reduce_id in &action.reduces {
                                if non_unit_productions.contains(&reduce_id) {
                                    new_action.reduces.insert(reduce_id);
                                }
                            }
                        }
                        for (nt, &goto_id) in &t_row.gotos {
                            new_row.gotos.insert(nt.clone(), goto_id);
                        }
                    }
                }
                intermediate_table.insert(new_id, new_row);
                worklist.push_back(new_id);
                new_id
            });

            s_row.gotos.insert(leaf.clone(), t_new_id);
        }
        intermediate_table.insert(s_id, s_row);
    }

    // --- 4. Post-processing ---
    for row in intermediate_table.values_mut() {
        row.gotos.retain(|nt, _| !nodes.contains(nt));
    }

    let reachable_states = find_reachable_states(start_state_id, &intermediate_table);
    intermediate_table.retain(|id, _| reachable_states.contains(id));

    let mut node_to_leaf_map: BTreeMap<NonTerminal, NonTerminal> = BTreeMap::new();
    for node in &nodes {
        if let Some(derives) = derives_unit_star.get(node) {
            if let Some(leaf) = derives.iter().find(|l| leaves.contains(l)) {
                node_to_leaf_map.insert(node.clone(), leaf.clone());
            }
        }
    }

    for (prod_id, prod) in productions.iter_mut().enumerate() {
        if !unit_productions.contains(&ProductionID(prod_id)) {
            if let Some(leaf_nt) = node_to_leaf_map.get(&prod.lhs) {
                prod.lhs = leaf_nt.clone();
            }
        }
    }

    // --- 5. Finalization: Rebuild stage_6_table ---
    for (t_set, &new_id) in &combined_states {
        if !id_to_item_set.contains_key(&new_id) {
            let mut combined_items = BTreeSet::new();
            for &old_id in t_set {
                if let Some(items) = id_to_item_set.get(&old_id) {
                    combined_items.extend(items.iter().cloned());
                }
            }
            id_to_item_set.insert(new_id, combined_items);
        }
    }

    stage_6_table.clear();
    for (state_id, row) in intermediate_table {
        let item_set = id_to_item_set.get(&state_id).cloned().unwrap_or_default();
        let mut new_row = Stage6Row::default();

        for (terminal, action) in row.shifts_and_reduces {
            if action.shift.is_some() || !action.reduces.is_empty() {
                let new_action = Stage6ShiftsAndReduces {
                    shift: action.shift.and_then(|id| id_to_item_set.get(&id).cloned()),
                    reduces: action.reduces,
                };
                new_row.shifts_and_reduces.insert(terminal, new_action);
            }
        }

        for (nt, goto_id) in row.gotos {
            new_row.gotos.insert(nt, id_to_item_set.get(&goto_id).unwrap().clone());
        }
        stage_6_table.insert(item_set, new_row);
    }
}