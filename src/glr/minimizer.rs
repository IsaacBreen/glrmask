use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use std::collections::{BTreeMap, BTreeSet, VecDeque, HashMap};
use crate::glr::analyze::remove_productions_with_undefined_nonterminals;
use crate::interface::display_productions;

/// Removes productions that contain terminals not in the `interesting_terminals` set.
pub fn remove_productions_with_uninteresting_terminals(
    productions: &[Production],
    interesting_terminals: &BTreeSet<Terminal>,
) -> Vec<Production> {
    productions
        .iter()
        .filter(|prod| {
            prod.rhs.iter().all(|symbol| match symbol {
                Symbol::NonTerminal(_) => true, // Keep non-terminals for now
                Symbol::Terminal(t) => interesting_terminals.contains(t),
            })
        })
        .cloned()
        .collect()
}

/// Finds all non-terminals that are part of any cycle in the grammar's dependency graph.
/// Helper for `substitute_single_productions_and_report` to find all nodes in any cycle.
fn find_cycles_dfs(
    u: &NonTerminal,
    adj: &BTreeMap<NonTerminal, BTreeSet<NonTerminal>>,
    visited: &mut BTreeSet<NonTerminal>,
    recursion_stack: &mut BTreeSet<NonTerminal>,
    path: &mut Vec<NonTerminal>,
    nts_in_cycle: &mut BTreeSet<NonTerminal>,
) {
    visited.insert(u.clone());
    recursion_stack.insert(u.clone());
    path.push(u.clone());

    if let Some(neighbors) = adj.get(u) {
        for v in neighbors {
            if recursion_stack.contains(v) {
                // Cycle detected. Mark all nodes from the start of the cycle in the current path.
                if let Some(cycle_start_index) = path.iter().position(|n| n == v) {
                    for i in cycle_start_index..path.len() {
                        nts_in_cycle.insert(path[i].clone());
                    }
                }
            } else if !visited.contains(v) {
                find_cycles_dfs(v, adj, visited, recursion_stack, path, nts_in_cycle);
            }
        }
    }

    path.pop();
    recursion_stack.remove(u);
}

fn find_all_nts_in_cycles(productions: &[Production]) -> BTreeSet<NonTerminal> {
    let mut adj: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
    let mut all_nts = BTreeSet::new();
    for p in productions {
        all_nts.insert(p.lhs.clone());
        for s in &p.rhs {
            if let Symbol::NonTerminal(nt) = s {
                all_nts.insert(nt.clone());
            }
        }
    }

    for nt in &all_nts {
        adj.entry(nt.clone()).or_default();
    }
    for p in productions {
        for s in &p.rhs {
            if let Symbol::NonTerminal(nt) = s {
                adj.get_mut(&p.lhs).unwrap().insert(nt.clone());
            }
        }
    }

    let mut nts_in_cycle = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let mut recursion_stack = BTreeSet::new();
    let mut path = Vec::new();

    for nt in all_nts {
        if !visited.contains(&nt) {
            find_cycles_dfs(&nt, &adj, &mut visited, &mut recursion_stack, &mut path, &mut nts_in_cycle);
        }
    }
    nts_in_cycle
}

/// Iteratively substitutes non-terminals that have only one production rule.
/// It returns the new set of productions and a set of non-terminals that were substituted.
/// The original productions for the substituted non-terminals are kept.
pub fn substitute_single_productions_and_report(
    productions: &[Production],
    start_nt: &NonTerminal,
    max_rhs_len: usize,
) -> (Vec<Production>, BTreeSet<NonTerminal>) {
    let mut current_prods = productions.to_vec();
    let mut all_substituted_nts = BTreeSet::new();

    loop {
        let before_prods = current_prods.clone();

        // 1. Find all non-terminals involved in cycles. They cannot be substituted.
        let nts_in_cycle = find_all_nts_in_cycles(&current_prods);

        // 2. Group productions by LHS to find single-production non-terminals.
        let mut prods_by_lhs: BTreeMap<NonTerminal, Vec<&Production>> = BTreeMap::new();
        for p in &current_prods {
            prods_by_lhs.entry(p.lhs.clone()).or_default().push(p);
        }

        // 3. Find new candidates for substitution.
        let mut substitutions: BTreeMap<NonTerminal, Vec<Symbol>> = BTreeMap::new();
        for (nt, prods) in &prods_by_lhs {
            if prods.len() == 1 &&
               nt != start_nt &&
               !nts_in_cycle.contains(nt) &&
               prods[0].rhs.len() <= max_rhs_len {
                substitutions.insert(nt.clone(), prods[0].rhs.clone());
            }
        }

        if substitutions.is_empty() {
            break; // No more substitutions can be made, terminate.
        }

        // 4. Record the non-terminals we are substituting in this pass.
        all_substituted_nts.extend(substitutions.keys().cloned());

        // 5. Perform substitution on all productions.
        let mut next_prods = Vec::new();
        for prod in &current_prods {
            let new_rhs = prod.rhs.iter().flat_map(|symbol| {
                if let Symbol::NonTerminal(nt) = symbol {
                    if let Some(subst_rhs) = substitutions.get(nt) {
                        subst_rhs.clone()
                    } else {
                        vec![symbol.clone()]
                    }
                } else {
                    vec![symbol.clone()]
                }
            }).collect();

            next_prods.push(Production {
                lhs: prod.lhs.clone(),
                rhs: new_rhs,
            });
        }
        current_prods = next_prods;

        if current_prods == before_prods {
            break;
        }
    }
    (current_prods, all_substituted_nts)
}

/// Removes productions whose LHS is in the given set of non-terminals.
pub fn remove_productions_for_nts(productions: &[Production], nts_to_remove: &BTreeSet<NonTerminal>) -> Vec<Production> {
    productions
        .iter()
        .filter(|p| !nts_to_remove.contains(&p.lhs))
        .cloned()
        .collect()
}

/// Removes productions whose LHS non-terminal is not reachable from the start symbol.
pub fn eliminate_unreachable_productions(
    productions: &[Production],
    start_nt: &NonTerminal,
) -> Vec<Production> {
    let mut reachable_nts: BTreeSet<NonTerminal> = BTreeSet::new();
    let mut worklist: VecDeque<NonTerminal> = VecDeque::new();

    reachable_nts.insert(start_nt.clone());
    worklist.push_back(start_nt.clone());

    while let Some(nt) = worklist.pop_front() {
        for prod in productions {
            if prod.lhs == nt {
                for symbol in &prod.rhs {
                    if let Symbol::NonTerminal(rhs_nt) = symbol {
                        if reachable_nts.insert(rhs_nt.clone()) {
                            worklist.push_back(rhs_nt.clone());
                        }
                    }
                }
            }
        }
    }

    productions
        .iter()
        .filter(|prod| reachable_nts.contains(&prod.lhs))
        .cloned()
        .collect()
}

/// Applies a series of simplification steps to a grammar to reduce it for a specific test case.
pub fn simplify_grammar_for_test_case(
    productions: &[Production],
    start_production_id: usize,
    interesting_terminals: &BTreeSet<Terminal>,
) -> (Vec<Production>, usize) {
    let start_nt = &productions[start_production_id].lhs;

    // 1. Remove productions with terminals not in our test case.
    let mut current_productions = remove_productions_with_uninteresting_terminals(productions, interesting_terminals);
    println!("simplify_grammar_for_test_case: After removing uninteresting terminals: {} productions", current_productions.len());
    if current_productions.len() < 500 {
        println!("Current productions:\n{}", display_productions(&current_productions));
    }

    // 2. Iteratively apply other simplifications until a fixed point is reached.
    loop {
        let before_count = current_productions.len();

        // Substitute non-terminals with a single production rule.
        const MAX_SUBSTITUTION_RHS_LEN: usize = 10;
        let (substituted_with_defs, substituted_nts) = substitute_single_productions_and_report(&current_productions, start_nt, MAX_SUBSTITUTION_RHS_LEN);
        let substituted = remove_productions_for_nts(&substituted_with_defs, &substituted_nts);
        if substituted.len() != current_productions.len() {
             println!("simplify_grammar_for_test_case: After substituting single productions: {} productions", substituted.len());
            if substituted.len() < 500 {
                println!("Substituted productions:\n{}", display_productions(&substituted));
            }
        }

        // Find the index of the start production to exempt it from removal.
        let current_start_prod_id = substituted.iter().position(|p| p.lhs == *start_nt);

        // Remove productions that now refer to undefined non-terminals.
        let exempt_indices = if let Some(id) = current_start_prod_id { vec![id] } else { vec![] };
        let cleaned = remove_productions_with_undefined_nonterminals(&substituted, &exempt_indices);
        if cleaned.len() != substituted.len() {
            println!("simplify_grammar_for_test_case: After removing undefined non-terminals: {} productions", cleaned.len());
            if cleaned.len() < 500 {
                println!("Cleaned productions:\n{}", display_productions(&cleaned));
            }
        }

        // Remove productions that are no longer reachable from the start symbol.
        let reachable = eliminate_unreachable_productions(&cleaned, start_nt);
        if reachable.len() != cleaned.len() {
            println!("simplify_grammar_for_test_case: After eliminating unreachable productions: {} productions", reachable.len());
            if reachable.len() < 500 {
                println!("Reachable productions:\n{}", display_productions(&reachable));
            }
        }

        println!("Current productions after simplification: {}", reachable.len());

        if reachable.len() == before_count {
            break; // Fixed point reached
        }
        current_productions = reachable;
    }

    let final_start_id = current_productions.iter().position(|p| p.lhs == *start_nt)
        .expect("Start production was removed during simplification");

    (current_productions, final_start_id)
}

/// Eliminates "unit productions" (rules of the form A -> B) from a grammar.
///
/// This function replaces chains of unit productions with direct productions, effectively
/// bypassing the intermediate non-terminals. For example, if the grammar contains:
///   `expr -> term`
///   `term -> factor`
///   `factor -> ID`
/// and `term` and `factor` are only used in these unit productions, this function
/// would effectively allow `expr` to directly produce whatever `ID` produces,
/// and the intermediate rules might be removed if they become unproductive.
///
/// # Arguments
/// * `productions` - The list of productions to simplify.
///
/// # Returns
/// A new `Vec<Production>` with unit productions eliminated.
pub fn eliminate_unit_productions(productions: &[Production]) -> Vec<Production> {
    // 1. Build a graph of unit productions: A -> B means an edge from A to B.
    let mut unit_graph: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
    let mut non_unit_productions: Vec<Production> = Vec::new();
    let all_nts: BTreeSet<NonTerminal> = productions.iter().map(|p| p.lhs.clone()).collect();

    for nt in &all_nts {
        unit_graph.entry(nt.clone()).or_default();
    }

    for p in productions {
        if p.rhs.len() == 1 {
            if let Symbol::NonTerminal(rhs_nt) = &p.rhs[0] {
                // This is a unit production: p.lhs -> rhs_nt
                unit_graph.entry(p.lhs.clone()).or_default().insert(rhs_nt.clone());
                continue; // Don't add unit productions to the new list yet.
            }
        }
        non_unit_productions.push(p.clone());
    }

    // 2. Compute the transitive closure of the unit production graph.
    // This tells us all non-terminals `B` that `A` can derive through a chain of unit productions.
    let mut closure: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
    for nt in &all_nts {
        let mut reachable = BTreeSet::new();
        let mut queue = VecDeque::new();

        reachable.insert(nt.clone());
        queue.push_back(nt.clone());

        while let Some(current_nt) = queue.pop_front() {
            if let Some(neighbors) = unit_graph.get(&current_nt) {
                for neighbor in neighbors {
                    if reachable.insert(neighbor.clone()) {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }
        closure.insert(nt.clone(), reachable);
    }

    // 3. Create the new set of productions.
    let mut new_productions = Vec::new();
    let mut added_prods = BTreeSet::new();

    // For each non-terminal `A`...
    for a_nt in &all_nts {
        // Find all non-terminals `B` that `A` can derive via unit productions (A =>* B).
        if let Some(reachable_from_a) = closure.get(a_nt) {
            for b_nt in reachable_from_a {
                // Find all non-unit productions that start with `B` (B -> γ).
                for p in &non_unit_productions {
                    if &p.lhs == b_nt {
                        // Add a new production `A -> γ`.
                        let new_prod = Production {
                            lhs: a_nt.clone(),
                            rhs: p.rhs.clone(),
                        };
                        if added_prods.insert(new_prod.clone()) {
                            new_productions.push(new_prod);
                        }
                    }
                }
            }
        }
    }

    // It's possible that some original non-unit productions are no longer needed if their
    // LHS was only part of a unit-production chain that has now been bypassed.
    // However, for safety and simplicity, we can keep them and let a subsequent
    // `eliminate_unreachable_productions` pass clean them up.
    // Let's add them back if they haven't been effectively recreated.
    for p in non_unit_productions {
        if added_prods.insert(p.clone()) {
            new_productions.push(p);
        }
    }

    new_productions
}

