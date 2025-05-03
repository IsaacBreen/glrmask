use std::collections::{BTreeMap, BTreeSet, VecDeque};
use crate::glr::grammar::{NonTerminal, Production, Symbol};

/// Computes the set of non-terminals that can derive the empty string (epsilon).
pub fn compute_nullable_nonterminals(productions: &[Production]) -> BTreeSet<NonTerminal> {
    let mut nullable_nonterminals = BTreeSet::new();
    let mut changed = true;

    while changed {
        changed = false;
        for production in productions {
            // Rule 1: A -> ε makes A nullable
            if production.rhs.is_empty() && !nullable_nonterminals.contains(&production.lhs) {
                nullable_nonterminals.insert(production.lhs.clone());
                changed = true;
            // Rule 2: A -> X1 X2 ... Xn makes A nullable if all Xi are nullable non-terminals
            } else if !production.rhs.is_empty() // Ensure RHS is not empty to avoid re-checking Rule 1
                      && production.rhs.iter().all(|symbol| {
                          matches!(symbol, Symbol::NonTerminal(nt) if nullable_nonterminals.contains(nt))
                      })
                      && !nullable_nonterminals.contains(&production.lhs)
            {
                nullable_nonterminals.insert(production.lhs.clone());
                changed = true;
            }
        }
    }

    nullable_nonterminals
}


/// Helper function for detecting cycles using Depth First Search.
fn detect_cycles_recursive(
    nt: &NonTerminal,
    graph: &BTreeMap<NonTerminal, BTreeSet<NonTerminal>>,
    visiting: &mut BTreeSet<NonTerminal>, // Nodes currently in the recursion stack for the current path
    visited: &mut BTreeSet<NonTerminal>,  // Nodes that have been fully explored (all descendants visited)
    path: &mut Vec<NonTerminal>,          // Current path for error reporting
) -> Result<(), String> {
    visiting.insert(nt.clone());
    path.push(nt.clone());

    // Explore neighbors
    if let Some(neighbors) = graph.get(nt) {
        for neighbor in neighbors {
            if visiting.contains(neighbor) {
                // Cycle detected: neighbor is already in the current recursion stack
                path.push(neighbor.clone()); // Add the node that closes the cycle to the path
                // Find where the cycle starts in the current path
                let cycle_start_index = path.iter().position(|n| n == neighbor).unwrap_or(0);
                let cycle_nodes: Vec<_> = path[cycle_start_index..].iter().map(|n| n.0.clone()).collect();
                let recursion_type = if cycle_nodes.len() == 2 && cycle_nodes[0] == cycle_nodes[1] { // A -> A case
                    "Direct"
                } else {
                    "Indirect"
                };
                return Err(format!(
                    "{} length-1 recursion cycle detected involving: {:?}",
                    recursion_type, cycle_nodes
                ));
            }
            // If the neighbor hasn't been fully explored yet, recurse
            if !visited.contains(neighbor) {
                // Propagate error if cycle found in recursive call
                detect_cycles_recursive(neighbor, graph, visiting, visited, path)?;
            }
            // Else: neighbor is in visited but not visiting, meaning it was fully explored from a different path, no cycle here.
        }
    }

    // Finished exploring descendants of nt
    visiting.remove(nt); // Remove from current recursion stack
    visited.insert(nt.clone()); // Mark as fully explored
    path.pop(); // Backtrack path

    Ok(())
}


/// Validates the grammar for common issues.
///
/// Checks for:
/// 1. Undefined non-terminals (non-terminals used in RHS but never defined in LHS).
/// 2. Length-1 recursion (direct or indirect), considering nullable prefixes.
///    - e.g., `A ::= A`
///    - e.g., `A ::= B A` where `B` is nullable
///    - e.g., `A ::= B`, `B ::= C`, `C ::= A`
///    - e.g., `A ::= X B`, `B ::= Y C`, `C ::= Z A` where `X, Y, Z` are nullable
pub fn validate(productions: &[Production]) -> Result<(), String> {
    // --- Check 1: Missing Productions ---
    let mut lhs_nonterms: BTreeSet<NonTerminal> = BTreeSet::new();
    let mut rhs_nonterms: BTreeSet<NonTerminal> = BTreeSet::new();
    let mut all_nonterminals: BTreeSet<NonTerminal> = BTreeSet::new(); // Collect all NTs

    for prod in productions {
        lhs_nonterms.insert(prod.lhs.clone());
        all_nonterminals.insert(prod.lhs.clone());
        for symbol in &prod.rhs {
            if let Symbol::NonTerminal(nt) = symbol {
                rhs_nonterms.insert(nt.clone());
                all_nonterminals.insert(nt.clone());
            }
        }
    }

    let missing_nonterms: BTreeSet<_> = rhs_nonterms.difference(&lhs_nonterms).collect();
    if !missing_nonterms.is_empty() {
        let missing_nonterm_strings: BTreeSet<_> = missing_nonterms.into_iter().map(|nt| nt.0.clone()).collect();
        // Provide more context in the error message
        let rhs_strings: BTreeSet<_> = rhs_nonterms.iter().map(|nt| nt.0.clone()).collect();
        let lhs_strings: BTreeSet<_> = lhs_nonterms.iter().map(|nt| nt.0.clone()).collect();
        return Err(format!(
            "Validation Error: Non-terminal(s) used in rule RHS but never defined in LHS: {:?}. All RHS non-terminals: {:?}. All LHS non-terminals: {:?}",
            missing_nonterm_strings, rhs_strings, lhs_strings
        ));
    }

    // --- Check 2: Length-1 Recursion ---
    let nullable_nonterminals = compute_nullable_nonterminals(productions);
    crate::debug!(3, "Nullable non-terminals: {:?}", nullable_nonterminals.iter().map(|nt| &nt.0).collect::<Vec<_>>());

    // Build a graph where an edge A -> B exists if A can derive B potentially preceded by nullable symbols.
    let mut unit_graph: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
    for nt in &all_nonterminals { // Initialize graph with all nodes
        unit_graph.entry(nt.clone()).or_default();
    }

    for prod in productions {
        let lhs = &prod.lhs;
        let mut first_non_nullable_nt: Option<&NonTerminal> = None;

        for symbol in &prod.rhs {
            match symbol {
                Symbol::Terminal(_) => {
                    // Found a terminal before any non-nullable non-terminal.
                    // This rule cannot contribute to length-1 recursion.
                    first_non_nullable_nt = None;
                    break;
                }
                Symbol::NonTerminal(nt) => {
                    if !nullable_nonterminals.contains(nt) {
                        // Found the first non-nullable non-terminal.
                        first_non_nullable_nt = Some(nt);
                        break;
                    }
                    // This non-terminal is nullable, continue scanning.
                }
            }
        }

        // If the first non-nullable symbol in the RHS is a non-terminal B,
        // add a directed edge A -> B to the graph.
        if let Some(target_nt) = first_non_nullable_nt {
            unit_graph.entry(lhs.clone()).or_default().insert(target_nt.clone());
            crate::debug!(4, "Unit graph edge added: {} -> {}", lhs.0, target_nt.0);
        }
    }

    // Detect cycles in the unit graph using DFS
    let mut visiting = BTreeSet::new(); // Nodes currently in DFS stack
    let mut visited = BTreeSet::new();  // Nodes fully explored

    for nt in &all_nonterminals {
        if !visited.contains(nt) { // Only start DFS from unvisited nodes
            let mut path = Vec::new(); // Track path for error reporting
            detect_cycles_recursive(nt, &unit_graph, &mut visiting, &mut visited, &mut path)?;
        }
    }

    // --- Validation Successful ---
    Ok(())
}


// TODO: This function is marked as broken and is not modified by this request.
pub fn drop_dead(productions: &[Production]) -> Vec<Production> {
    // todo: this function is broken
    let mut nt_reachables: BTreeMap<&NonTerminal, BTreeSet<&NonTerminal>> = BTreeMap::new();

    for prod in productions {
        let rhs_nonterms: BTreeSet<_> = prod.rhs.iter()
            .filter_map(|symbol| {
                if let Symbol::NonTerminal(nt) = symbol {
                    Some(nt)
                } else {
                    None
                }
            })
            .collect();
        nt_reachables.insert(&prod.lhs, rhs_nonterms);
    }

    loop {
        let mut changed = false;
        for (nt, reachables) in nt_reachables.clone() {
            let old_len = nt_reachables[nt].len();
            for reachable in reachables {
                if let Some(reachable_reachables) = nt_reachables.get(reachable).cloned() {
                    nt_reachables.get_mut(nt).unwrap().extend(reachable_reachables);
                }
            }
            if nt_reachables[nt].len() != old_len {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let start_prod = &productions[0];
    let mut reachable_from_start = BTreeSet::new();
    for symbol in &start_prod.rhs {
        if let Symbol::NonTerminal(nt) = symbol {
            reachable_from_start.insert(nt);
            if let Some(nt_reachables) = nt_reachables.get(nt).cloned() {
                reachable_from_start.extend(nt_reachables);
            }
        }
    }

    let new_productions: Vec<_> = productions.iter()
        .filter(|prod| reachable_from_start.contains(&prod.lhs) || *prod == start_prod)
        .cloned()
        .collect();

    crate::debug!(2, "Dropped {} productions", productions.len() - new_productions.len());

    new_productions
}