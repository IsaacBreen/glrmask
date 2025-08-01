use std::collections::{BTreeMap, BTreeSet};
use crate::glr::grammar::{NonTerminal, Production, Symbol};
use crate::glr::items::Item;
use crate::glr::table::{ProductionID, Stage6Table};

pub fn simplify_1_reduction_chains(
    stage_6_table: &mut Stage6Table,
    productions: &[Production],
    _start_production_id: usize, // Not used, but part of the signature
) {
    crate::debug!(3, "Simplifying 1-reduction chains...");

    // Step 1: Identify unit productions (A -> B) and non-unit productions.
    let mut unit_productions: BTreeMap<ProductionID, (NonTerminal, NonTerminal)> = BTreeMap::new();
    let mut non_unit_productions: BTreeSet<ProductionID> = BTreeSet::new();

    for (i, p) in productions.iter().enumerate() {
        let prod_id = ProductionID(i);
        if p.rhs.len() == 1 {
            if let Symbol::NonTerminal(ref rhs_nt) = p.rhs[0] {
                unit_productions.insert(prod_id, (p.lhs.clone(), rhs_nt.clone()));
                continue;
            }
        }
        non_unit_productions.insert(prod_id);
    }

    if unit_productions.is_empty() {
        crate::debug!(3, "No unit productions found. Skipping simplification.");
        return;
    }
    crate::debug!(3, "Found {} unit productions.", unit_productions.len());

    // Step 2: Build the unit production graph and compute its transitive closure (A ->* B).
    let mut unit_reach: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
    let all_nonterminals: BTreeSet<NonTerminal> = productions.iter()
        .flat_map(|p| {
            let mut nts = vec![p.lhs.clone()];
            for s in &p.rhs {
                if let Symbol::NonTerminal(nt) = s {
                    nts.push(nt.clone());
                }
            }
            nts
        })
        .collect();

    // Initialize with self-loops (A ->* A)
    for nt in &all_nonterminals {
        unit_reach.entry(nt.clone()).or_default().insert(nt.clone());
    }
    // Add direct unit production edges (A -> B)
    for (_, (lhs, rhs)) in &unit_productions {
        unit_reach.entry(lhs.clone()).or_default().insert(rhs.clone());
    }

    // Compute transitive closure using fixed-point iteration.
    let mut changed = true;
    while changed {
        changed = false;
        for nt_i in all_nonterminals.iter() {
            // Collect neighbors to avoid borrowing issues while modifying the map
            let neighbors: Vec<NonTerminal> = unit_reach.get(nt_i).unwrap().iter().cloned().collect();
            for nt_j in neighbors {
                if let Some(successors_of_j) = unit_reach.get(&nt_j).cloned() {
                    let current_successors = unit_reach.get_mut(nt_i).unwrap();
                    let old_len = current_successors.len();
                    current_successors.extend(successors_of_j);
                    if current_successors.len() != old_len {
                        changed = true;
                    }
                }
            }
        }
    }

    // Step 3: For each non-terminal `A`, compute the set of non-unit productions
    // `B -> ...` where `A ->* B`.
    let mut target_reductions: BTreeMap<NonTerminal, BTreeSet<ProductionID>> = BTreeMap::new();
    for (lhs_nt, reachable_nts) in &unit_reach {
        let mut targets = BTreeSet::new();
        for reachable_nt in reachable_nts {
            // Find all non-unit productions with lhs `reachable_nt`
            for &prod_id in &non_unit_productions {
                if productions[prod_id.0].lhs == *reachable_nt {
                    targets.insert(prod_id);
                }
            }
        }
        target_reductions.insert(lhs_nt.clone(), targets);
    }

    // Step 4: Iterate through the table and replace unit reductions with their non-unit targets.
    let mut changes_count = 0;
    for row in stage_6_table.values_mut() {
        for action in row.shifts_and_reduces.values_mut() {
            // Optimization: skip if no unit productions are in this reduce set.
            if !action.reduces.iter().any(|pid| unit_productions.contains_key(pid)) {
                continue;
            }

            let original_reduces = std::mem::take(&mut action.reduces);
            let mut new_reduces = BTreeSet::new();

            for prod_id in original_reduces {
                if non_unit_productions.contains(&prod_id) {
                    new_reduces.insert(prod_id);
                } else {
                    // It's a unit production.
                    let (lhs, _) = unit_productions.get(&prod_id).unwrap();
                    if let Some(targets) = target_reductions.get(lhs) {
                        new_reduces.extend(targets);
                        changes_count += 1;
                    }
                }
            }
            action.reduces = new_reduces;
        }
    }
    crate::debug!(3, "Replaced {} unit reduction instances.", changes_count);
}
