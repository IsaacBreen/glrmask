use crate::glr::analyze::remove_productions_with_undefined_nonterminals;
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::interface::display_productions;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Removes productions that contain terminals not in the `interesting_terminals` set.
pub fn remove_productions_with_uninteresting_terminals(
    productions: &[Production],
    interesting_terminals: &BTreeSet<Terminal>,
) -> Vec<Production> {
    productions
        .iter()
        .filter(|prod| {
            prod.rhs.iter().all(|symbol| match symbol {
                Symbol::NonTerminal(_) => true,
                Symbol::Terminal(t) => interesting_terminals.contains(t),
            })
        })
        .cloned()
        .collect()
}

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
pub fn substitute_single_productions_and_report(
    productions: &[Production],
    start_nt: &NonTerminal,
    max_rhs_len: usize,
) -> (Vec<Production>, BTreeSet<NonTerminal>) {
    let mut current_prods = productions.to_vec();
    let mut all_substituted_nts = BTreeSet::new();

    loop {
        let before_prods = current_prods.clone();

        let nts_in_cycle = find_all_nts_in_cycles(&current_prods);

        let mut prods_by_lhs: BTreeMap<NonTerminal, Vec<&Production>> = BTreeMap::new();
        for p in &current_prods {
            prods_by_lhs.entry(p.lhs.clone()).or_default().push(p);
        }

        let mut substitutions: BTreeMap<NonTerminal, Vec<Symbol>> = BTreeMap::new();
        for (nt, prods) in &prods_by_lhs {
            if prods.len() == 1
                && nt != start_nt
                && !nts_in_cycle.contains(nt)
                && prods[0].rhs.len() <= max_rhs_len
            {
                substitutions.insert(nt.clone(), prods[0].rhs.clone());
            }
        }

        if substitutions.is_empty() {
            break;
        }

        all_substituted_nts.extend(substitutions.keys().cloned());

        let mut next_prods = Vec::new();
        for prod in &current_prods {
            let new_rhs = prod
                .rhs
                .iter()
                .flat_map(|symbol| {
                    if let Symbol::NonTerminal(nt) = symbol {
                        if let Some(subst_rhs) = substitutions.get(nt) {
                            subst_rhs.clone()
                        } else {
                            vec![symbol.clone()]
                        }
                    } else {
                        vec![symbol.clone()]
                    }
                })
                .collect();

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
pub fn remove_productions_for_nts(
    productions: &[Production],
    nts_to_remove: &BTreeSet<NonTerminal>,
) -> Vec<Production> {
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
    interesting_terminals: &BTreeSet<Terminal>,
) -> (Vec<Production>, usize) {
    if productions.is_empty() {
        return (vec![], 0);
    }
    let start_nt = &productions[0].lhs;

    let mut current_productions =
        remove_productions_with_uninteresting_terminals(productions, interesting_terminals);
    println!(
        "simplify_grammar_for_test_case: After removing uninteresting terminals: {} productions",
        current_productions.len()
    );
    if current_productions.len() < 500 {
        println!(
            "Current productions:\n{}",
            display_productions(&current_productions)
        );
    }

    loop {
        let before_count = current_productions.len();

        const MAX_SUBSTITUTION_RHS_LEN: usize = 10;
        let (substituted_with_defs, substituted_nts) =
            substitute_single_productions_and_report(&current_productions, start_nt, MAX_SUBSTITUTION_RHS_LEN);
        let substituted = remove_productions_for_nts(&substituted_with_defs, &substituted_nts);
        if substituted.len() != current_productions.len() {
            println!(
                "simplify_grammar_for_test_case: After substituting single productions: {} productions",
                substituted.len()
            );
            if substituted.len() < 500 {
                println!(
                    "Substituted productions:\n{}",
                    display_productions(&substituted)
                );
            }
        }

        let current_start_prod_id = substituted.iter().position(|p| p.lhs == *start_nt);

        let exempt_indices = if let Some(id) = current_start_prod_id {
            vec![id]
        } else {
            vec![]
        };
        let cleaned =
            remove_productions_with_undefined_nonterminals(&substituted, &exempt_indices);
        if cleaned.len() != substituted.len() {
            println!(
                "simplify_grammar_for_test_case: After removing undefined non-terminals: {} productions",
                cleaned.len()
            );
            if cleaned.len() < 500 {
                println!(
                    "Cleaned productions:\n{}",
                    display_productions(&cleaned)
                );
            }
        }

        let reachable = eliminate_unreachable_productions(&cleaned, start_nt);
        if reachable.len() != cleaned.len() {
            println!(
                "simplify_grammar_for_test_case: After eliminating unreachable productions: {} productions",
                reachable.len()
            );
            if reachable.len() < 500 {
                println!(
                    "Reachable productions:\n{}",
                    display_productions(&reachable)
                );
            }
        }

        println!(
            "Current productions after simplification: {}",
            reachable.len()
        );

        if reachable.len() == before_count {
            current_productions = reachable;
            break;
        }
        current_productions = reachable;
    }

    let final_start_id = current_productions
        .iter()
        .position(|p| p.lhs == *start_nt)
        .expect("Start production was removed during simplification");

    (current_productions, final_start_id)
}
