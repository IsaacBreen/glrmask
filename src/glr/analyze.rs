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
pub fn remove_productions_with_undefined_nonterminals(initial_productions: &[Production], exempt: &[usize]) -> Vec<Production> {
    let mut current_productions: Vec<(usize, Production)> = initial_productions.into_iter().cloned().enumerate().collect();

    loop {
        let mut defined_lhs_nonterminals: BTreeSet<NonTerminal> = BTreeSet::new();
        for (i, prod) in &current_productions {
            defined_lhs_nonterminals.insert(prod.lhs.clone());
        }
        let mut removed_productions: Vec<(usize, Production)> = Vec::new();
        let mut kept_productions: Vec<(usize, Production)> = Vec::new();
        for (i, prod) in current_productions {
            let keep = prod.rhs.iter().all(|symbol| match symbol {
                Symbol::Terminal(_) => true, // Terminals are always defined
                Symbol::NonTerminal(nt) => defined_lhs_nonterminals.contains(nt),
            }) || exempt.contains(&i);
            if keep {
                kept_productions.push((i, prod));
            } else {
                removed_productions.push((i, prod));
            }
        }
        current_productions = kept_productions;
        if removed_productions.is_empty() {
            break;
        }
        crate::debug!(2, "Removing {} productions with undefined non-terminals.", removed_productions.len());
        let all_rhs_nonterminals: BTreeSet<NonTerminal> = removed_productions.iter().flat_map(|(i, prod)| prod.rhs.iter().filter_map(|symbol| match symbol {
            Symbol::NonTerminal(nt) => Some(nt.clone()),
            _ => None,
        })).collect();
        crate::debug!(2, "Missing non-terminals ({}) in productions:", all_rhs_nonterminals.len());
        for nt in all_rhs_nonterminals.difference(&defined_lhs_nonterminals) {
            crate::debug!(2, "  {}", nt.0);
        }
        crate::debug!(2, "Removed productions:");
        for (i, prod) in removed_productions {
            crate::debug!(2, "  {}", prod);
        }
    }

    current_productions.into_iter().map(|(_, prod)| prod).collect()
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

/// Filters productions to keep only those relevant to deriving specified "interesting" symbols.
///
/// A production `P: L -> R` is kept if:
/// 1. Its LHS (`L`) can derive an interesting symbol (i.e., `L` is in `can_derive_set`).
/// AND
/// 2. Its RHS (`R`), for this specific production, can also derive an interesting symbol.
///
/// The RHS `R` "can derive an interesting symbol" if:
/// 1. `R` directly contains an interesting symbol (terminal or non-terminal).
/// 2. `R` contains a non-terminal `N` that is itself in `can_derive_set`.
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

    // --- Pre-computation ---
    // 1. Non-terminals that can derive an interesting symbol.
    let can_derive_set = compute_can_derive_interesting(initial_productions, interesting_symbols);
    crate::debug!(3, "filter_productions_by_reachability: CanDeriveInteresting set: {:?}", can_derive_set.iter().map(|nt| &nt.0).collect::<Vec<_>>());

    // The following sets are no longer directly used in the simplified logic,
    // but compute_can_derive_interesting is the core.
    // let reachable_from_interesting_nt_set = compute_reachable_from_interesting_nts(initial_productions, interesting_symbols);
    // crate::debug!(3, "filter_productions_by_reachability: ReachableFromInterestingNTs set: {:?}", reachable_from_interesting_nt_set.iter().map(|nt| &nt.0).collect::<Vec<_>>());
    // let mut bootstrap_lhs_nts = BTreeSet::new();
    // for production in initial_productions {
    //     for symbol_in_rhs in &production.rhs {
    //         if matches!(symbol_in_rhs, Symbol::Terminal(_)) && interesting_symbols.contains(symbol_in_rhs) {
    //             bootstrap_lhs_nts.insert(production.lhs.clone());
    //             break;
    //         }
    //     }
    // }
    // crate::debug!(3, "filter_productions_by_reachability: BootstrapLHS NTs (from interesting terminals): {:?}", bootstrap_lhs_nts.iter().map(|nt| &nt.0).collect::<Vec<_>>());

    // --- Filtering Loop ---
    let mut kept_productions = Vec::new();
    for production in initial_productions {
        let lhs = &production.lhs;

        // Condition A: LHS can derive an interesting symbol.
        let lhs_can_derive_interesting = can_derive_set.contains(lhs);

        // Condition B: RHS of *this specific* production can derive an interesting symbol.
        let mut rhs_can_derive_interesting_for_this_rule = false;
        for symbol_in_rhs in &production.rhs {
            match symbol_in_rhs {
                Symbol::Terminal(_) => {
                    if interesting_symbols.contains(symbol_in_rhs) {
                        rhs_can_derive_interesting_for_this_rule = true;
                        break;
                    }
                }
                Symbol::NonTerminal(nt_in_rhs) => {
                    // An RHS non-terminal leads to interesting if:
                    // 1. It IS an interesting symbol itself (e.g. if interesting_symbols can contain NTs)
                    // 2. OR it can derive an interesting symbol (i.e., nt_in_rhs is in can_derive_set)
                    if interesting_symbols.contains(symbol_in_rhs) || can_derive_set.contains(nt_in_rhs) {
                        rhs_can_derive_interesting_for_this_rule = true;
                        break;
                    }
                }
            }
        }

        if lhs_can_derive_interesting && rhs_can_derive_interesting_for_this_rule {
            kept_productions.push(production.clone());
        } else {
            crate::debug!(
                4,
                "Filtering out production: {} (LHS can derive interesting: {}, RHS of this rule can derive interesting: {})",
                production,
                lhs_can_derive_interesting,
                rhs_can_derive_interesting_for_this_rule
            );
        }
    }
    
    kept_productions
}

/// Computes the set of non-terminals that can derive a terminal string.
fn compute_productive_nonterminals(productions: &[Production]) -> BTreeSet<NonTerminal> {
    let mut productive_nts = BTreeSet::new();
    if productions.is_empty() {
        return productive_nts;
    }

    let mut changed = true;
    while changed {
        changed = false;
        for prod in productions {
            if productive_nts.contains(&prod.lhs) {
                continue;
            }

            let rhs_is_productive = prod.rhs.iter().all(|symbol| match symbol {
                Symbol::Terminal(_) => true,
                Symbol::NonTerminal(nt) => productive_nts.contains(nt),
            });

            if rhs_is_productive {
                if productive_nts.insert(prod.lhs.clone()) {
                    changed = true;
                }
            }
        }
    }
    productive_nts
}

/// Removes rules involving non-productive non-terminals.
/// If the start_symbol becomes non-productive, returns an empty set of productions.
fn remove_non_productive_rules(productions: &[Production], start_symbol: &NonTerminal) -> Vec<Production> {
    if productions.is_empty() { return Vec::new(); }
    let productive_nts = compute_productive_nonterminals(productions);

    if !productive_nts.contains(start_symbol) && productions.iter().any(|p| p.lhs == *start_symbol) {
        crate::debug!(2, "Simplify: Start symbol {} became non-productive. Removing all productions.", start_symbol.0);
        return Vec::new();
    }

    productions.iter().filter(|prod| {
        productive_nts.contains(&prod.lhs) &&
        prod.rhs.iter().all(|symbol| match symbol {
            Symbol::Terminal(_) => true,
            Symbol::NonTerminal(nt) => productive_nts.contains(nt),
        })
    }).cloned().collect()
}

/// Computes non-terminals reachable from the start_symbol.
fn compute_reachable_nonterminals(productions: &[Production], start_symbol: &NonTerminal) -> BTreeSet<NonTerminal> {
    let mut reachable_nts = BTreeSet::new();
    if productions.is_empty() || !productions.iter().any(|p| p.lhs == *start_symbol) {
        return reachable_nts;
    }

    reachable_nts.insert(start_symbol.clone());
    let mut worklist: VecDeque<NonTerminal> = VecDeque::new();
    worklist.push_back(start_symbol.clone());

    while let Some(nt_lhs) = worklist.pop_front() {
        for prod in productions {
            if prod.lhs == nt_lhs {
                for symbol_in_rhs in &prod.rhs {
                    if let Symbol::NonTerminal(nt_in_rhs) = symbol_in_rhs {
                        if reachable_nts.insert(nt_in_rhs.clone()) {
                            worklist.push_back(nt_in_rhs.clone());
                        }
                    }
                }
            }
        }
    }
    reachable_nts
}

/// Removes rules whose LHS is not reachable from the start_symbol.
/// If the start_symbol itself becomes unreachable, returns an empty set.
fn remove_unreachable_rules(productions: &[Production], start_symbol: &NonTerminal) -> Vec<Production> {
    if productions.is_empty() { return Vec::new(); }
    let reachable_nts = compute_reachable_nonterminals(productions, start_symbol);

    if !reachable_nts.contains(start_symbol) && productions.iter().any(|p| p.lhs == *start_symbol) {
         crate::debug!(2, "Simplify: Start symbol {} became unreachable. Removing all productions.", start_symbol.0);
        return Vec::new();
    }

    productions.iter().filter(|prod| reachable_nts.contains(&prod.lhs)).cloned().collect()
}

/// Inlines epsilon productions. For L -> X N Y where N is nullable, adds L -> X Y.
/// Relies on subsequent cleanup to remove original N -> ε if N becomes unused/unproductive.
fn inline_epsilon_productions(productions: &[Production]) -> (Vec<Production>, bool) {
    let nullable_set = compute_nullable_nonterminals(productions);
    let mut new_productions = BTreeSet::new(); // Use BTreeSet for deduplication
    let mut made_change = false;

    for prod in productions {
        if prod.rhs.is_empty() { // Keep existing A -> ε rules for now
            new_productions.insert(prod.clone());
            continue;
        }

        let nullable_indices: Vec<usize> = prod.rhs.iter().enumerate()
            .filter_map(|(i, symbol)| match symbol {
                Symbol::NonTerminal(nt) if nullable_set.contains(nt) => Some(i),
                _ => None,
            }).collect();

        if nullable_indices.is_empty() {
            new_productions.insert(prod.clone());
            continue;
        }

        // Iterate through all 2^k combinations of omitting/keeping nullable symbols
        let num_nullable_in_rhs = nullable_indices.len();
        for i in 0..(1 << num_nullable_in_rhs) {
            let mut current_rhs = Vec::new();
            let mut nullable_ptr = 0;
            for (original_idx, symbol) in prod.rhs.iter().enumerate() {
                if nullable_ptr < nullable_indices.len() && original_idx == nullable_indices[nullable_ptr] {
                    // This is a nullable symbol from our list. Check the i-th bit.
                    if (i >> nullable_ptr) & 1 == 1 { // Bit is 1: keep this occurrence
                        current_rhs.push(symbol.clone());
                    } else { // Bit is 0: omit this occurrence
                        // If we omit, it's a change from the original full rule
                    }
                    nullable_ptr += 1;
                } else {
                    current_rhs.push(symbol.clone());
                }
            }

            // Add the new production if it's not an exact copy of the original RHS AND
            // it's not an L -> L self-loop (unless original was L -> L).
            // An empty RHS is a valid new epsilon production.
            let is_self_loop = current_rhs.len() == 1 && current_rhs[0] == Symbol::NonTerminal(prod.lhs.clone());
            let original_is_self_loop = prod.rhs.len() == 1 && prod.rhs[0] == Symbol::NonTerminal(prod.lhs.clone());

            if !is_self_loop || original_is_self_loop { // Avoid creating new L->L from L->N L
                if new_productions.insert(Production { lhs: prod.lhs.clone(), rhs: current_rhs }) {
                    // If insert is true, it's a new unique production.
                    // We need to check if it's different from the original to claim `made_change`.
                    // This check is tricky. Simpler: compare input set with output set at the end.
                }
            }
        }
    }
    
    let final_prods: Vec<Production> = new_productions.into_iter().collect();
    let input_set: BTreeSet<_> = productions.iter().cloned().collect();
    let final_set: BTreeSet<_> = final_prods.iter().cloned().collect();
    if input_set != final_set {
        made_change = true;
    }

    (final_prods, made_change)
}

/// Inlines unit productions (A -> B).
fn inline_unit_productions(productions: &[Production]) -> (Vec<Production>, bool) {
    let mut non_unit_productions: Vec<Production> = Vec::new();
    let mut unit_relations: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
    let mut all_nts = BTreeSet::new();

    for prod in productions {
        all_nts.insert(prod.lhs.clone());
        if prod.rhs.len() == 1 && matches!(&prod.rhs[0], Symbol::NonTerminal(_)) {
            let rhs_nt = match &prod.rhs[0] { Symbol::NonTerminal(nt) => nt, _ => unreachable!() };
            if prod.lhs != *rhs_nt { // Not A -> A
                unit_relations.entry(prod.lhs.clone()).or_default().insert(rhs_nt.clone());
            } else { non_unit_productions.push(prod.clone()); } // Keep A -> A
        } else {
            non_unit_productions.push(prod.clone());
        }
    }

    let mut unit_closure: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
    for nt_lhs in all_nts.iter() {
        let mut reachable_by_unit = BTreeSet::new();
        let mut q: VecDeque<_> = vec![nt_lhs.clone()].into();
        let mut visited_dfs = BTreeSet::new();
        while let Some(curr) = q.pop_front() {
            if !visited_dfs.insert(curr.clone()) { continue; }
            if let Some(direct_units) = unit_relations.get(&curr) {
                for unit_rhs in direct_units {
                    if curr != *unit_rhs { reachable_by_unit.insert(unit_rhs.clone()); }
                    q.push_back(unit_rhs.clone());
                }
            }
        }
        if !reachable_by_unit.is_empty() { unit_closure.insert(nt_lhs.clone(), reachable_by_unit); }
    }

    let mut result_prods = BTreeSet::new();
    for p in &non_unit_productions { result_prods.insert(p.clone()); }

    for (lhs_a, derives_bs) in &unit_closure {
        for rhs_b in derives_bs {
            for non_unit_prod in &non_unit_productions {
                if non_unit_prod.lhs == *rhs_b { // B -> gamma (non-unit)
                    result_prods.insert(Production { lhs: lhs_a.clone(), rhs: non_unit_prod.rhs.clone() });
                }
            }
        }
    }
    
    let final_prods: Vec<Production> = result_prods.into_iter().collect();
    let changed = productions.iter().cloned().collect::<BTreeSet<_>>() != final_prods.iter().cloned().collect::<BTreeSet<_>>();
    (final_prods, changed)
}

/// Substitutes non-terminals that have a single, non-recursive production.
fn substitute_single_production_nonterminals(productions: &[Production], start_symbol: &NonTerminal) -> (Vec<Production>, bool) {
    let mut current_productions: Vec<Production> = productions.to_vec();
    let mut any_substitution_in_loop = false;

    loop {
        let mut prod_map: BTreeMap<NonTerminal, Vec<&Production>> = BTreeMap::new();
        for p in &current_productions { prod_map.entry(p.lhs.clone()).or_default().push(p); }

        let mut substitutable_nts: BTreeMap<NonTerminal, Vec<Symbol>> = BTreeMap::new();
        for (nt, prods_for_nt) in &prod_map {
            if nt == start_symbol || prods_for_nt.len() != 1 { continue; }
            let single_prod = prods_for_nt[0];
            if !single_prod.rhs.iter().any(|s| matches!(s, Symbol::NonTerminal(r_nt) if r_nt == nt)) {
                substitutable_nts.insert(nt.clone(), single_prod.rhs.clone());
            }
        }

        if substitutable_nts.is_empty() { break; }
        any_substitution_in_loop = true;
        let mut next_iteration_productions = Vec::new();
        for prod_to_modify in &current_productions {
            if substitutable_nts.contains_key(&prod_to_modify.lhs) { continue; } // Rule defining the substituted NT is dropped

            let mut new_rhs = Vec::new();
            for symbol in &prod_to_modify.rhs {
                if let Symbol::NonTerminal(nt_in_rhs) = symbol {
                    if let Some(sub_rhs) = substitutable_nts.get(nt_in_rhs) {
                        new_rhs.extend_from_slice(sub_rhs);
                        continue;
                    }
                }
                new_rhs.push(symbol.clone());
            }
            next_iteration_productions.push(Production { lhs: prod_to_modify.lhs.clone(), rhs: new_rhs });
        }
        current_productions = next_iteration_productions;
    }
    (current_productions, any_substitution_in_loop)
}

/// Simplifies a grammar definition by applying various common techniques iteratively.
/// Aims to improve performance by reducing productions or restructuring them.
/// The actual parse tree generated does NOT have to be preserved, only the accepted language.
pub fn simplify_grammar(initial_productions: &[Production], start_symbol: &NonTerminal) -> Vec<Production> {
    if initial_productions.is_empty() || !initial_productions.iter().any(|p| p.lhs == *start_symbol) {
        crate::debug!(1, "Simplify: Initial productions empty or start symbol {} not defined. Returning empty.", start_symbol.0);
        return Vec::new();
    }

    let mut current_productions = initial_productions.to_vec();
    crate::debug!(2, "Simplify: Initial {} productions.", current_productions.len());

    loop {
        let productions_at_loop_start = current_productions.clone();

        current_productions = remove_non_productive_rules(&current_productions, start_symbol);
        if current_productions.is_empty() { crate::debug!(2, "Simplify: Became empty after non-productive removal."); break; }
        current_productions = remove_unreachable_rules(&current_productions, start_symbol);
        if current_productions.is_empty() { crate::debug!(2, "Simplify: Became empty after unreachable removal."); break; }
        
        let (prods_eps, _) = inline_epsilon_productions(&current_productions);
        current_productions = prods_eps;
        current_productions = remove_non_productive_rules(&current_productions, start_symbol);
        if current_productions.is_empty() { crate::debug!(2, "Simplify: Became empty after eps-inline cleanup."); break; }
        current_productions = remove_unreachable_rules(&current_productions, start_symbol);
        if current_productions.is_empty() { crate::debug!(2, "Simplify: Became empty after eps-inline cleanup."); break; }

        let (prods_unit, _) = inline_unit_productions(&current_productions);
        current_productions = prods_unit;
        current_productions = remove_non_productive_rules(&current_productions, start_symbol);
        if current_productions.is_empty() { crate::debug!(2, "Simplify: Became empty after unit-inline cleanup."); break; }
        current_productions = remove_unreachable_rules(&current_productions, start_symbol);
        if current_productions.is_empty() { crate::debug!(2, "Simplify: Became empty after unit-inline cleanup."); break; }

        let (prods_sub, _) = substitute_single_production_nonterminals(&current_productions, start_symbol);
        current_productions = prods_sub;
        current_productions = remove_non_productive_rules(&current_productions, start_symbol);
        if current_productions.is_empty() { crate::debug!(2, "Simplify: Became empty after substitute cleanup."); break; }
        current_productions = remove_unreachable_rules(&current_productions, start_symbol);
        if current_productions.is_empty() { crate::debug!(2, "Simplify: Became empty after substitute cleanup."); break; }

        current_productions.sort();
        let mut productions_at_loop_start_sorted = productions_at_loop_start;
        productions_at_loop_start_sorted.sort();

        if current_productions == productions_at_loop_start_sorted {
            crate::debug!(2, "Simplify: Fixed point reached with {} productions.", current_productions.len());
            break;
        }
        crate::debug!(2, "Simplify: Iteration complete. Productions count: {}. Continuing.", current_productions.len());
    }

    current_productions.sort(); // Final sort for deterministic output
    crate::debug!(2, "Simplify: Final simplified grammar has {} productions.", current_productions.len());
    current_productions
}

