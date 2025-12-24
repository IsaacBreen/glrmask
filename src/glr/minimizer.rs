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

/// Left-factor productions to extract common prefixes.
/// 
/// For productions like:
///   A → α β
///   A → α γ
/// 
/// Creates:
///   A → α A'
///   A' → β | γ
/// 
/// This reduces LR states by delaying the decision point until after the common prefix.
/// 
/// Returns the modified productions and the set of new nonterminals created.
pub fn left_factor_grammar(
    productions: &[Production],
    unique_name_gen: &mut impl FnMut(&str) -> String,
) -> (Vec<Production>, BTreeSet<NonTerminal>) {
    if productions.is_empty() {
        return (Vec::new(), BTreeSet::new());
    }
    
    let mut result_prods: Vec<Production> = Vec::new();
    let mut new_nonterminals: BTreeSet<NonTerminal> = BTreeSet::new();
    
    // Remember the start production's LHS
    let start_lhs = productions[0].lhs.clone();
    
    // Group productions by LHS, using BTreeMap for deterministic ordering
    let mut prods_by_lhs: BTreeMap<NonTerminal, Vec<Vec<Symbol>>> = BTreeMap::new();
    for prod in productions {
        prods_by_lhs.entry(prod.lhs.clone())
            .or_default()
            .push(prod.rhs.clone());
    }
    
    // Process start production first to ensure it remains at index 0
    if let Some(start_alts) = prods_by_lhs.remove(&start_lhs) {
        let factored = factor_alternatives(&start_lhs, start_alts, unique_name_gen, &mut new_nonterminals);
        result_prods.extend(factored);
    }
    
    // Process remaining productions
    for (lhs, alternatives) in prods_by_lhs {
        let factored = factor_alternatives(&lhs, alternatives, unique_name_gen, &mut new_nonterminals);
        result_prods.extend(factored);
    }
    
    (result_prods, new_nonterminals)
}

/// Factor alternatives for a single nonterminal.
/// 
/// This function avoids creating epsilon productions by:
/// 1. Only factoring when ALL alternatives have a non-empty suffix after the common prefix
/// 2. Grouping alternatives by their first symbol and factoring each group separately
fn factor_alternatives(
    lhs: &NonTerminal,
    alternatives: Vec<Vec<Symbol>>,
    unique_name_gen: &mut impl FnMut(&str) -> String,
    new_nonterminals: &mut BTreeSet<NonTerminal>,
) -> Vec<Production> {
    if alternatives.len() <= 1 {
        // Nothing to factor
        return alternatives.into_iter()
            .map(|rhs| Production { lhs: lhs.clone(), rhs })
            .collect();
    }
    
    // Group alternatives by their first symbol
    let mut groups: BTreeMap<Option<Symbol>, Vec<Vec<Symbol>>> = BTreeMap::new();
    for alt in alternatives {
        let first = alt.first().cloned();
        groups.entry(first).or_default().push(alt);
    }
    
    // If there's only one group, or one of the groups is epsilon, 
    // we can't factor without creating epsilon productions
    if groups.len() == 1 {
        // All alternatives start with the same symbol, try to factor
        let (first_sym, alts) = groups.into_iter().next().unwrap();
        
        if first_sym.is_none() {
            // All alternatives are epsilon - just return them
            return alts.into_iter()
                .map(|rhs| Production { lhs: lhs.clone(), rhs })
                .collect();
        }
        
        // Check: would factoring create an epsilon production?
        // This happens if any alternative is exactly the common prefix (suffix would be empty)
        let min_len = alts.iter().map(|a| a.len()).min().unwrap_or(0);
        if min_len == 0 {
            // One of the alternatives is empty - can't factor
            return alts.into_iter()
                .map(|rhs| Production { lhs: lhs.clone(), rhs })
                .collect();
        }
        
        // Find the actual longest common prefix
        let common_prefix_len = find_longest_common_prefix(&alts);
        
        // Don't factor if it would create an epsilon production
        // (i.e., if any alternative's length equals the common prefix length)
        if alts.iter().any(|alt| alt.len() == common_prefix_len) {
            // Factoring would create epsilon - don't factor
            return alts.into_iter()
                .map(|rhs| Production { lhs: lhs.clone(), rhs })
                .collect();
        }
        
        if common_prefix_len == 0 {
            // No common prefix - return as-is
            return alts.into_iter()
                .map(|rhs| Production { lhs: lhs.clone(), rhs })
                .collect();
        }
        
        // Safe to factor - all suffixes will be non-empty
        let common_prefix: Vec<Symbol> = alts[0][..common_prefix_len].to_vec();
        
        // Create new nonterminal for the factored part
        let new_nt_name = unique_name_gen(&lhs.0);
        let new_nt = NonTerminal(new_nt_name);
        new_nonterminals.insert(new_nt.clone());
        
        // Create main production: A → prefix A'
        let mut main_rhs = common_prefix;
        main_rhs.push(Symbol::NonTerminal(new_nt.clone()));
        
        // Create factored alternatives (suffixes after common prefix)
        let suffixes: Vec<Vec<Symbol>> = alts.into_iter()
            .map(|alt| alt[common_prefix_len..].to_vec())
            .collect();
        
        // Recursively factor the suffixes (there may be more common prefixes)
        let mut result = vec![Production { lhs: lhs.clone(), rhs: main_rhs }];
        let factored_suffixes = factor_alternatives(&new_nt, suffixes, unique_name_gen, new_nonterminals);
        result.extend(factored_suffixes);
        
        return result;
    }
    
    // Multiple groups with different first symbols - process each group separately
    // and emit all resulting productions
    let mut result: Vec<Production> = Vec::new();
    for (_first, alts) in groups {
        let factored = factor_alternatives(lhs, alts, unique_name_gen, new_nonterminals);
        result.extend(factored);
    }
    result
}

/// Find the length of the longest common prefix among all alternatives.
fn find_longest_common_prefix(alternatives: &[Vec<Symbol>]) -> usize {
    if alternatives.is_empty() {
        return 0;
    }
    
    let min_len = alternatives.iter().map(|a| a.len()).min().unwrap_or(0);
    
    for i in 0..min_len {
        let first = &alternatives[0][i];
        if !alternatives.iter().all(|alt| alt.get(i) == Some(first)) {
            return i;
        }
    }
    
    min_len
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
