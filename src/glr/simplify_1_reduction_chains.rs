use std::collections::{BTreeMap, BTreeSet};
use crate::glr::grammar::{NonTerminal, Production, Symbol};
use crate::glr::table::{ProductionID, Stage6Table};

/// Eliminates unit productions (e.g., `A -> B`) from the parse table.
///
/// This optimization is performed on the `Stage6Table` before states and actions are
/// finalized with IDs. It works by modifying the reduce actions in the table.
///
/// The core idea is based on the property that if a grammar contains a unit production
/// `A -> B`, then whenever `A` is reducible, `B` is also reducible. This means we can
/// replace a reduction by `A -> B` with all the reductions possible for `B`. This
/// process is applied transitively.
///
/// For example, if we have:
/// - `A -> B`
/// - `B -> C`
/// - `C -> 'terminal'`
///
/// And a state has a reduce action for `A -> B`, this function will replace that
/// reduction with a reduction for `C -> 'terminal'`. This avoids the intermediate
/// steps of reducing `C` to `B` and then `B` to `A` during parsing.
///
/// # Algorithm
/// 1.  Identify all unit productions in the grammar. A production is a unit production
///     if its right-hand side consists of a single non-terminal.
/// 2.  For each state and each lookahead terminal in the `Stage6Table`:
///     a.  Find the set of `reduces` for the corresponding action.
///     b.  Perform a fixed-point iteration:
///         i.  Start with the existing set of reductions.
///         ii. If this set contains a unit production `A -> B`, add all productions
///             of `B` to the set.
///         iii. Repeat until no new productions can be added.
///     c.  After the expansion is complete, remove all the unit productions from the
///         set, leaving only the non-unit productions that were derived.
/// 3.  The `Stage6Table` is updated in place with these modified reduction sets.
///
/// # Arguments
/// * `stage_6_table` - A mutable reference to the `Stage6Table` to be modified.
/// * `productions` - A slice of all productions in the grammar.
pub fn simplify_1_reduction_chains(
    stage_6_table: &mut Stage6Table,
    productions: &[Production],
) {
    // 1. Identify all unit productions (A -> B) and map their ID to the RHS non-terminal.
    let unit_prod_rhs: BTreeMap<ProductionID, &NonTerminal> = productions
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if p.rhs.len() == 1 {
                if let Symbol::NonTerminal(nt_rhs) = &p.rhs[0] {
                    Some((ProductionID(i), nt_rhs))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    if unit_prod_rhs.is_empty() {
        crate::debug!(3, "No unit productions found, skipping elimination.");
        return;
    }
    crate::debug!(2, "Found {} unit productions to eliminate.", unit_prod_rhs.len());

    // 2. Create a map from each non-terminal to the list of its productions.
    let prods_by_lhs: BTreeMap<&NonTerminal, Vec<ProductionID>> = productions
        .iter()
        .enumerate()
        .fold(BTreeMap::new(), |mut acc, (i, p)| {
            acc.entry(&p.lhs).or_default().push(ProductionID(i));
            acc
        });

    // 3. Iterate over the table and expand reductions.
    for row in stage_6_table.values_mut() {
        for action in row.shifts_and_reduces.values_mut() {
            // If there are no unit productions in this action's reduce set, skip.
            if !action.reduces.iter().any(|pid| unit_prod_rhs.contains_key(pid)) {
                continue;
            }

            let mut expanded_reduces = action.reduces.clone();
            let mut worklist: Vec<ProductionID> = action.reduces.iter().cloned().collect();
            let mut processed_in_worklist = BTreeSet::new();

            while let Some(pid) = worklist.pop() {
                if !processed_in_worklist.insert(pid) {
                    continue;
                }

                // If pid is a unit production, add all productions of its RHS non-terminal to the worklist.
                if let Some(rhs_nt) = unit_prod_rhs.get(&pid) {
                    if let Some(prods_of_rhs) = prods_by_lhs.get(rhs_nt) {
                        for &new_pid in prods_of_rhs {
                            if expanded_reduces.insert(new_pid) {
                                // If we added a new production, it might be a unit production itself,
                                // so add it to the worklist to be processed.
                                worklist.push(new_pid);
                            }
                        }
                    }
                }
            }

            // 4. After expansion, filter out all the original unit productions from the set.
            let original_len = action.reduces.len();
            action.reduces = expanded_reduces
                .into_iter()
                .filter(|pid| !unit_prod_rhs.contains_key(pid))
                .collect();

            if action.reduces.len() != original_len {
                crate::debug!(4, "Simplified unit reductions in state. Before: {}, After: {}", original_len, action.reduces.len());
            }
        }
    }
}
