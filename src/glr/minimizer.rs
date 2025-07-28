use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
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

/// Iteratively substitutes non-terminals that have only one production rule.
pub fn substitute_single_productions(
    productions: &[Production],
    start_nt: &NonTerminal,
) -> Vec<Production> {
    let mut current_prods = productions.to_vec();
    loop {
        let mut prods_by_lhs: BTreeMap<NonTerminal, Vec<&Production>> = BTreeMap::new();
        for p in &current_prods {
            prods_by_lhs.entry(p.lhs.clone()).or_default().push(p);
        }

        let mut substitutions: BTreeMap<NonTerminal, Vec<Symbol>> = BTreeMap::new();
        for (nt, prods) in &prods_by_lhs {
            if prods.len() == 1 && nt != start_nt {
                let single_prod = prods[0];
                // Avoid direct recursion `A -> A ...` which would cause infinite substitution
                if !single_prod.rhs.iter().any(|s| s == &Symbol::NonTerminal(nt.clone())) {
                    substitutions.insert(nt.clone(), single_prod.rhs.clone());
                }
            }
        }

        if substitutions.is_empty() {
            break;
        }

        let mut next_prods = Vec::new();
        for prod in &current_prods {
            // Don't include the definitions of the substituted non-terminals
            if substitutions.contains_key(&prod.lhs) {
                continue;
            }

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
    }
    current_prods
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
) -> Vec<Production> {
    let start_nt = &productions[start_production_id].lhs;

    // 1. Remove productions with terminals not in our test case.
    let mut current_productions = remove_productions_with_uninteresting_terminals(productions, interesting_terminals);
    println!("After removing uninteresting terminals: {} productions", current_productions.len());
    if current_productions.len() < 20 {
        println!("Current productions:\n{}", display_productions(&current_productions));
    }

    // 2. Iteratively apply other simplifications until a fixed point is reached.
    loop {
        let before_count = current_productions.len();

        // Substitute non-terminals with a single production rule.
        let substituted = substitute_single_productions(&current_productions, start_nt);
        if substituted.len() != current_productions.len() {
             println!("After substituting single productions: {} productions", substituted.len());
            if substituted.len() < 20 {
                println!("Substituted productions:\n{}", display_productions(&substituted));
            }
        }

        // Remove productions that now refer to undefined non-terminals.
        let cleaned = remove_productions_with_undefined_nonterminals(&substituted, &[start_production_id]);
        if cleaned.len() != substituted.len() {
            println!("After removing undefined non-terminals: {} productions", cleaned.len());
            if cleaned.len() < 20 {
                println!("Cleaned productions:\n{}", display_productions(&cleaned));
            }
        }

        // Remove productions that are no longer reachable from the start symbol.
        let reachable = eliminate_unreachable_productions(&cleaned, start_nt);
        if reachable.len() != cleaned.len() {
            println!("After eliminating unreachable productions: {} productions", reachable.len());
            if reachable.len() < 20 {
                println!("Reachable productions:\n{}", display_productions(&reachable));
            }
        }

        if reachable.len() == before_count {
            break; // Fixed point reached
        }
        current_productions = reachable;
    }

    current_productions
}
