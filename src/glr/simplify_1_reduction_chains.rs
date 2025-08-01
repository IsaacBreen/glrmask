use std::collections::{BTreeMap, BTreeSet};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{ProductionID, Stage6ShiftsAndReduces, Stage6Table};

/// A context structure to hold all the read-only data needed for the `resolve` function.
/// This avoids passing many arguments and helps with managing lifetimes.
struct Context<'a> {
    productions: &'a [Production],
    unit_prods: &'a BTreeMap<ProductionID, NonTerminal>,
    kernel_to_state: &'a BTreeMap<BTreeSet<Item>, &'a BTreeSet<Item>>,
    predecessor_map: &'a BTreeMap<&'a BTreeSet<Item>, Vec<(&'a BTreeSet<Item>, Symbol)>>,
    stage_6_table: &'a Stage6Table,
}

/// Recursively resolves unit reduction chains for a given state `s` and lookahead terminal `t`.
///
/// It follows chains of unit reductions (`A -> B`) and accumulates the actions (shifts and non-unit reduces)
/// from the final states in the chains. It uses memoization to avoid recomputing results and a
/// `visited` set to handle cycles (e.g., `A -> B`, `B -> A`).
fn resolve<'a>(
    s: &'a BTreeSet<Item>,
    t: &'a Terminal,
    ctx: &Context<'a>,
    memo: &mut BTreeMap<(&'a BTreeSet<Item>, &'a Terminal), (Option<BTreeSet<Item>>, BTreeSet<ProductionID>)>,
    visited: &mut BTreeSet<&'a BTreeSet<Item>>,
) -> (Option<BTreeSet<Item>>, BTreeSet<ProductionID>) {
    // Check memoization table first.
    if let Some(cached) = memo.get(&(s, t)) {
        return cached.clone();
    }

    // Check for cycles in the reduction chain.
    if !visited.insert(s) {
        return (None, BTreeSet::new()); // Cycle detected, terminate this path.
    }

    let actions = &ctx.stage_6_table[s].shifts_and_reduces[t];

    // Initialize final actions with the current state's shift action.
    let mut final_shift = actions.shift.clone();
    let mut final_reduces = BTreeSet::new();
    let mut unit_reduces_to_follow = BTreeSet::new();

    // Separate unit reductions from other reductions.
    for &p_id in &actions.reduces {
        if ctx.unit_prods.contains_key(&p_id) {
            unit_reduces_to_follow.insert(p_id);
        } else {
            final_reduces.insert(p_id);
        }
    }

    // If there are unit reductions, follow them.
    if !unit_reduces_to_follow.is_empty() {
        for p_id in unit_reduces_to_follow {
            let production = &ctx.productions[p_id.0];
            let nt_a = &production.lhs; // The LHS of the unit production (e.g., A in A -> B)
            let nt_b = ctx.unit_prods.get(&p_id).unwrap(); // The RHS (e.g., B)

            // Find the predecessor state that led to `s` via a GOTO on `nt_b`.
            if let Some(preds) = ctx.predecessor_map.get(&s) {
                let nt_b_symbol = Symbol::NonTerminal(nt_b.clone());
                if let Some((s_prev, _)) = preds.iter().find(|(_, sym)| sym == &nt_b_symbol) {
                    // Now find the state we would go to from `s_prev` on `nt_a`.
                    if let Some(s_a_kernel) = ctx.stage_6_table[s_prev].gotos.get(nt_a) {
                        if let Some(s_a) = ctx.kernel_to_state.get(s_a_kernel) {
                            // Recursively resolve actions for the new state.
                            let (rec_shift, rec_reduces) = resolve(s_a, t, ctx, memo, visited);

                            // Merge the resolved actions.
                            if let Some(rec_s) = rec_shift {
                                final_shift.get_or_insert_with(BTreeSet::new).extend(rec_s);
                            }
                            final_reduces.extend(rec_reduces);
                        }
                    }
                }
            }
        }
    }

    visited.remove(s); // Backtrack

    let result = (final_shift, final_reduces);
    memo.insert((s, t), result.clone());
    result
}

pub fn simplify_1_reduction_chains(
    stage_6_table: &mut Stage6Table,
    productions: &[Production],
    start_production_id: usize,
) {
    crate::debug!(2, "Simplifying 1-reduction chains...");

    let unit_prods: BTreeMap<ProductionID, NonTerminal> = productions.iter().enumerate().filter_map(|(i, p)| {
        if p.rhs.len() == 1 {
            if let Symbol::NonTerminal(nt) = &p.rhs[0] { return Some((ProductionID(i), nt.clone())); }
        }
        None
    }).collect();

    if unit_prods.is_empty() { crate::debug!(2, "No unit productions found."); return; }
    crate::debug!(3, "Found {} unit productions.", unit_prods.len());

    let start_nt = &productions[start_production_id].lhs;
    let mut kernel_to_state: BTreeMap<BTreeSet<Item>, &BTreeSet<Item>> = BTreeMap::new();
    for state in stage_6_table.keys() {
        let kernel: BTreeSet<Item> = state.iter().filter(|item| item.dot_position > 0 || item.production.lhs == *start_nt).cloned().collect();
        kernel_to_state.insert(kernel, state);
    }

    let mut predecessor_map: BTreeMap<&BTreeSet<Item>, Vec<(&BTreeSet<Item>, Symbol)>> = BTreeMap::new();
    for (s_prev, row) in stage_6_table.iter() {
        for (nt, kernel) in &row.gotos { if let Some(s_next) = kernel_to_state.get(kernel) { predecessor_map.entry(s_next).or_default().push((s_prev, Symbol::NonTerminal(nt.clone()))); } }
        for (t, action) in &row.shifts_and_reduces { if let Some(kernel) = &action.shift { if let Some(s_next) = kernel_to_state.get(kernel) { predecessor_map.entry(s_next).or_default().push((s_prev, Symbol::Terminal(t.clone()))); } } }
    }

    let mut memo = BTreeMap::new();
    let mut new_actions = BTreeMap::new();
    let ctx = Context { productions, unit_prods: &unit_prods, kernel_to_state: &kernel_to_state, predecessor_map: &predecessor_map, stage_6_table };

    for (s, row) in stage_6_table.iter() {
        for t in row.shifts_and_reduces.keys() {
            let mut visited = BTreeSet::new();
            let resolved_actions = resolve(s, t, &ctx, &mut memo, &mut visited);
            new_actions.insert((s.clone(), t.clone()), resolved_actions);
        }
    }

    for ((s, t), (shift, reduces)) in new_actions {
        let action = Stage6ShiftsAndReduces { shift, reduces };
        if let Some(row) = stage_6_table.get_mut(&s) { row.shifts_and_reduces.insert(t, action); }
    }
    crate::debug!(2, "Finished simplifying 1-reduction chains.");
}
