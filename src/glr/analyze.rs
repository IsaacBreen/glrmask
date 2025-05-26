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
                // Get only the nodes involved in the cycle itself
                let cycle_nodes_in_path: Vec<_> = path[cycle_start_index..].iter().map(|n| n.0.as_str()).collect();

                // Format the cycle path string: "A -> B -> C -> A"
                let cycle_path_str = cycle_nodes_in_path.join(" -> ");

                let recursion_type = if cycle_nodes_in_path.len() == 2 && cycle_nodes_in_path[0] == cycle_nodes_in_path[1] { // A -> A case
                    "Direct"
                } else {
                    "Indirect"
                };

                // Remove the temporary node added to path for cycle detection before returning error
                path.pop();

                return Err(format!(
                    "{} length-1 recursion cycle detected: {}",
                    recursion_type, cycle_path_str
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
///    A rule `A ::= α` contributes to a potential cycle if `α` consists of
/// 3. Left-nullable left recursion: Rules of the form `A ::= B1 ... Bk A ...` where `k > 0`
///    zero or more nullable non-terminals, followed by a single non-terminal `B`,
///    and nothing else (i.e., `A ::= Nullable* B`).
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

    // Build a graph where an edge A -> B exists if a rule A ::= Nullable* B exists.
    let mut unit_graph: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
    for nt in &all_nonterminals { // Initialize graph with all nodes
        unit_graph.entry(nt.clone()).or_default();
    }

    // --- Corrected Logic for Building Unit Graph ---
    for prod in productions {
        let lhs = &prod.lhs;
        let rhs = &prod.rhs;

        let mut first_non_nullable_symbol: Option<&Symbol> = None;
        let mut first_non_nullable_idx: Option<usize> = None;

        // Find the first non-nullable symbol and its index
        for (idx, symbol) in rhs.iter().enumerate() {
            match symbol {
                Symbol::Terminal(_) => {
                    first_non_nullable_symbol = Some(symbol);
                    first_non_nullable_idx = Some(idx);
                    break; // Found first non-nullable, stop scanning
                }
                Symbol::NonTerminal(nt) => {
                    if !nullable_nonterminals.contains(nt) {
                        first_non_nullable_symbol = Some(symbol);
                        first_non_nullable_idx = Some(idx);
                        break; // Found first non-nullable, stop scanning
                    }
                    // It's a nullable non-terminal, continue scanning
                }
            }
        }

        match first_non_nullable_symbol {
            Some(Symbol::NonTerminal(target_nt)) => {
                // Found a non-terminal B as the first non-nullable symbol at index k
                let k = first_non_nullable_idx.unwrap();

                // Check if B is the *last* symbol in the RHS
                if k == rhs.len() - 1 {
                    // Rule is of the form A ::= Nullable* B. Add edge A -> B.
                    unit_graph.entry(lhs.clone()).or_default().insert(target_nt.clone());
                    crate::debug!(4, "Unit graph edge added ({} -> {} from rule: {} ::= {:?})", lhs.0, target_nt.0, lhs.0, prod.rhs);
                } else {
                    // Rule is A ::= Nullable* B NonEmptySuffix... Does not contribute.
                    crate::debug!(5, "Rule skipped for unit graph ({} ::= {:?}): Non-terminal {} followed by other symbols.", lhs.0, prod.rhs, target_nt.0);
                }
            }
            Some(Symbol::Terminal(_)) => {
                // First non-nullable symbol is a terminal. Rule is A ::= Nullable* Terminal ... Does not contribute.
                crate::debug!(5, "Rule skipped for unit graph ({} ::= {:?}): First non-nullable symbol is a terminal.", lhs.0, prod.rhs);
            }
            None => {
                // All symbols in RHS are nullable, or RHS is empty. Rule is A ::= Nullable*. Does not contribute.
                crate::debug!(5, "Rule skipped for unit graph ({} ::= {:?}): RHS is fully nullable or empty.", lhs.0, prod.rhs);
            }
        }
    }
    // --- End Corrected Logic ---


    // Detect cycles in the unit graph using DFS
    let mut visiting = BTreeSet::new(); // Nodes currently in DFS stack
    let mut visited = BTreeSet::new();  // Nodes fully explored

    // Sort the non-terminals for deterministic traversal order (optional, but good for consistency)
    let sorted_nonterminals: Vec<_> = all_nonterminals.iter().collect();

    for nt in sorted_nonterminals {
        if !visited.contains(nt) { // Only start DFS from unvisited nodes
            let mut path = Vec::new(); // Track path for error reporting
            detect_cycles_recursive(nt, &unit_graph, &mut visiting, &mut visited, &mut path)?;
        }
    }

    // --- Check 3: Left-Nullable Left Recursion ---
    // Detect rules like A ::= B A ... where B is nullable.
    for prod in productions {
        let lhs = &prod.lhs;
        let rhs = &prod.rhs;

        // Iterate through RHS symbols to find the recursive non-terminal A
        for (i, symbol) in rhs.iter().enumerate() {
            if let Symbol::NonTerminal(nt) = symbol {
                if nt == lhs { // Found potential left recursion: A ::= ... A ...
                    // Check if all preceding symbols (if any) are nullable non-terminals
                    let prefix = &rhs[0..i];
                    if !prefix.is_empty() { // Only check if there's a prefix
                        let prefix_is_nullable = prefix.iter().all(|sym| match sym {
                            Symbol::NonTerminal(prefix_nt) => nullable_nonterminals.contains(prefix_nt),
                            Symbol::Terminal(_) => false, // Terminals are not nullable
                        });

                        if prefix_is_nullable {
                            return Err(format!("Validation Error: Left-nullable left recursion detected in rule '{} ::= {:?}'. The prefix '{:?}' before the recursive non-terminal '{}' is nullable.", lhs.0, rhs, prefix, lhs.0));
                        }
                    }
                    // If the prefix is empty (direct left recursion A ::= A ...) or not fully nullable,
                    // we don't flag it here (GLR can handle standard left recursion).
                    // We only care about the case where a nullable sequence precedes the recursion.
                    break; // Found the first instance of A on RHS, no need to check further in this rule
                }
            }
        }
    }

    // --- Validation Successful ---
    Ok(())
}

/// Removes productions that use non-terminals on their RHS which are never defined on the LHS
/// of any *remaining* production. This process is repeated until no more productions can be removed.
///
/// This is useful for cleaning up grammars before further analysis or parser generation,
/// especially if the grammar might contain references to non-terminals that have no rules.
pub fn remove_productions_with_undefined_nonterminals(initial_productions: &[Production]) -> Vec<Production> {
    let mut current_productions = initial_productions.to_vec();

    loop {
        let mut defined_lhs_nonterminals: BTreeSet<NonTerminal> = BTreeSet::new();
        for prod in &current_productions {
            defined_lhs_nonterminals.insert(prod.lhs.clone());
        }
        let mut removed_productions = Vec::new();
        let mut kept_productions = Vec::new();
        for prod in current_productions {
            let keep = prod.rhs.iter().all(|symbol| match symbol {
                Symbol::Terminal(_) => true, // Terminals are always defined
                Symbol::NonTerminal(nt) => defined_lhs_nonterminals.contains(nt),
            });
            if keep {
                kept_productions.push(prod);
            } else {
                removed_productions.push(prod);
            }
        }
        current_productions = kept_productions;
        if removed_productions.is_empty() {
            break;
        }
        crate::debug!(2, "Removing {} productions with undefined non-terminals.", removed_productions.len());
        let all_rhs_nonterminals: BTreeSet<NonTerminal> = removed_productions.iter().flat_map(|prod| prod.rhs.iter().filter_map(|symbol| match symbol {
            Symbol::NonTerminal(nt) => Some(nt.clone()),
            _ => None,
        })).collect();
        crate::debug!(2, "Missing non-terminals ({}) in productions:", all_rhs_nonterminals.len());
        for nt in all_rhs_nonterminals.difference(&defined_lhs_nonterminals) {
            crate::debug!(2, "  {}", nt.0);
        }
        crate::debug!(2, "Removed productions:");
        for prod in removed_productions {
            crate::debug!(2, "  {}", prod);
        }
    }

    current_productions
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

/// Computes the set of non-terminals that can derive a string containing at least one of the interesting_symbols.
fn compute_can_derive_interesting(
    productions: &[Production],
    interesting_symbols: &BTreeSet<Symbol>,
) -> BTreeSet<NonTerminal> {
    let mut can_derive_interesting = BTreeSet::new();
    let mut changed = true;

    while changed {
        changed = false;
        for production in productions {
            if can_derive_interesting.contains(&production.lhs) {
                continue;
            }

            let mut rhs_can_lead_to_interesting = false;
            for symbol_in_rhs in &production.rhs {
                match symbol_in_rhs {
                    Symbol::Terminal(_) => {
                        if interesting_symbols.contains(symbol_in_rhs) {
                            rhs_can_lead_to_interesting = true;
                            break;
                        }
                    }
                    Symbol::NonTerminal(nt_in_rhs) => {
                        // An RHS non-terminal leads to interesting if it IS an interesting symbol itself,
                        // OR it can derive an interesting symbol.
                        if interesting_symbols.contains(symbol_in_rhs) || can_derive_interesting.contains(nt_in_rhs) {
                            rhs_can_lead_to_interesting = true;
                            break;
                        }
                    }
                }
            }

            if rhs_can_lead_to_interesting {
                if can_derive_interesting.insert(production.lhs.clone()) {
                    changed = true;
                }
            }
        }
    }
    can_derive_interesting
}

/// Computes the set of non-terminals that are reachable by derivation from any non-terminal in interesting_symbols.
/// If interesting_symbols contains no non-terminals, this returns an empty set.
fn compute_reachable_from_interesting_nts(
    productions: &[Production],
    interesting_symbols: &BTreeSet<Symbol>,
) -> BTreeSet<NonTerminal> {
    let mut seed_interesting_nts = BTreeSet::new();
    for s in interesting_symbols {
        if let Symbol::NonTerminal(nt) = s {
            seed_interesting_nts.insert(nt.clone());
        }
    }

    if seed_interesting_nts.is_empty() {
        return BTreeSet::new();
    }

    let mut reachable_set = seed_interesting_nts.clone();
    let mut worklist: VecDeque<NonTerminal> = seed_interesting_nts.into_iter().collect();

    while let Some(nt_lhs_from_worklist) = worklist.pop_front() {
        // Find productions where nt_lhs_from_worklist is the LHS
        for production in productions {
            if production.lhs == nt_lhs_from_worklist {
                for symbol_in_rhs in &production.rhs {
                    if let Symbol::NonTerminal(nt_in_rhs) = symbol_in_rhs {
                        if !reachable_set.contains(nt_in_rhs) {
                             if reachable_set.insert(nt_in_rhs.clone()) { // Check insert result
                                worklist.push_back(nt_in_rhs.clone());
                            }
                        }
                    }
                }
            }
        }
    }
    reachable_set
}

/// Filters productions based on relevance to a set of "interesting" symbols.
///
/// The goal is to keep only those productions that are necessary to parse or define
/// structures related to the `interesting_symbols`.
///
/// A production `P: LHS -> RHS` is kept if its `LHS` non-terminal is "involved".
/// A non-terminal `N` is considered "involved" if it meets any of these criteria,
/// or is reachable through productions from another "involved" non-terminal:
/// 1. `N` itself is one of the `interesting_symbols` (if `N` is a non-terminal).
/// 2. `N` can derive a string containing at least one of the `interesting_symbols`.
///
/// The process is:
///   a. Identify all non-terminals satisfying (1) or (2). This forms a "seed set" of directly relevant non-terminals.
///   b. Perform a reachability analysis (graph traversal) starting from this seed set. Any non-terminal
///      that appears on the right-hand side of a production whose left-hand side is in the (expanding) set of
///      involved non-terminals also becomes involved.
///   c. Finally, all productions whose LHS is an "involved" non-terminal are kept.
///
/// If `interesting_symbols` is empty, no productions will be kept.
pub fn filter_productions_by_reachability(
    initial_productions: &[Production],
    interesting_symbols: &BTreeSet<Symbol>,
) -> Vec<Production> {
    if interesting_symbols.is_empty() {
        crate::debug!(2, "filter_productions_by_reachability: interesting_symbols is empty, returning no productions.");
        return Vec::new();
    }

    // Step 1: Find all non-terminals that can derive an interesting symbol.
    // N is in can_derive_set if N =>* ... S_int ... where S_int is in interesting_symbols.
    let can_derive_set = compute_can_derive_interesting(initial_productions, interesting_symbols);
    crate::debug!(3, "filter_productions_by_reachability: CanDeriveInteresting set: {:?}", can_derive_set.iter().map(|nt| &nt.0).collect::<Vec<_>>());

    // Step 2: Create the initial seed set of "directly relevant" symbols for overall reachability.
    // These are non-terminals that are either interesting themselves or can derive an interesting symbol.
    let mut seed_symbols_for_reachability = BTreeSet::new();
    for nt in &can_derive_set {
        seed_symbols_for_reachability.insert(Symbol::NonTerminal(nt.clone()));
    }
    for s in interesting_symbols {
        if let Symbol::NonTerminal(_) = s {
            // Add only if it's an NT. If it's already covered by can_derive_set, BTreeSet handles duplicates.
            seed_symbols_for_reachability.insert(s.clone());
        }
    }
    crate::debug!(3, "filter_productions_by_reachability: Seed symbols for overall reachability: {:?}", seed_symbols_for_reachability.iter().map(|s| match s { Symbol::NonTerminal(nt) => nt.0.as_str(), Symbol::Terminal(t) => t.0.as_str() }).collect::<Vec<_>>());

    // Step 3: Compute all non-terminals "involved" with the interesting symbols.
    // This includes the initial seed NTs and any NTs reachable from them via productions.
    // If X is in seed_symbols_for_reachability (as an NT), and X -> Y Z, then Y and Z (if NTs) become involved.
    let all_involved_nts = compute_reachable_from_interesting_nts(initial_productions, &seed_symbols_for_reachability);
    crate::debug!(3, "filter_productions_by_reachability: All involved NTs (LHS must be in this set): {:?}", all_involved_nts.iter().map(|nt| &nt.0).collect::<Vec<_>>());

    // Step 4: Filter productions. A production is kept if its LHS is an "involved" non-terminal.
    let mut kept_productions = Vec::new();
    for production in initial_productions {
        if all_involved_nts.contains(&production.lhs) {
            kept_productions.push(production.clone());
        } else {
            crate::debug!(
                4,
                "Filtering out production: {} (LHS {} not in all_involved_nts)",
                production,
                production.lhs.0
            );
        }
    }
    
    // Note: `generate_glr_parser_with_maps` (which will likely consume this output)
    // internally calls `remove_productions_with_undefined_nonterminals` and `validate`,
    // so further cleanup here might be redundant if that's the next step.
    // However, it's good that this filter produces a self-contained set as much as possible.
    // If `all_involved_nts` is empty (e.g. interesting_symbols only contains terminals not derivable),
    // then kept_productions will be empty, which is correct.
    kept_productions
}
