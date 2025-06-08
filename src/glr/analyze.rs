use std::collections::{BTreeMap, BTreeSet, VecDeque, HashMap}; // Added HashMap
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal, compute_first_sets}; // Added compute_first_sets

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

pub fn simplify_grammar(initial_productions: &[Production], start_production_id: usize) -> (Vec<Production>, usize) {
    todo!()
}

/// Computes a map from each terminal to a set of terminals that can immediately follow it in the grammar.
/// This considers sequences T1 -> T2, and T1 -> NT -> T2 where NT can be a sequence of non-terminals.
pub fn compute_terminal_follow_sets(
    productions: &[Production],
    nullable_nonterminals: &BTreeSet<NonTerminal>,
    first_sets: &BTreeMap<NonTerminal, BTreeSet<Terminal>>,
) -> BTreeMap<Terminal, BTreeSet<Terminal>> {
    let mut terminal_follows: BTreeMap<Terminal, BTreeSet<Terminal>> = BTreeMap::new();

    for prod in productions {
        for i in 0..prod.rhs.len() {
            if let Symbol::Terminal(t1) = &prod.rhs[i] {
                // t1 is the current terminal we are finding follows for.
                // Look at symbols S(i+1), S(i+2), ...
                for j in (i + 1)..prod.rhs.len() {
                    let next_symbol_in_rhs = &prod.rhs[j];
                    match next_symbol_in_rhs {
                        Symbol::Terminal(t2) => {
                            terminal_follows.entry(t1.clone()).or_default().insert(t2.clone());
                            break; // Found a terminal, stop scanning further for this t1 in this rule.
                        }
                        Symbol::NonTerminal(nt_after_t1) => {
                            if let Some(first_of_nt) = first_sets.get(nt_after_t1) {
                                terminal_follows.entry(t1.clone()).or_default().extend(first_of_nt.iter().cloned());
                            }
                            if !nullable_nonterminals.contains(nt_after_t1) {
                                break; // nt_after_t1 is not nullable, so it blocks further symbols from following t1.
                            }
                            // nt_after_t1 is nullable, continue to the next symbol in rhs for t1.
                        }
                    }
                }
            }
        }
    }
    terminal_follows
}

/// Helper to find the rightmost non-nullable symbol in a production's RHS.
fn find_rightmost_non_nullable<'a>(
    rhs: &'a [Symbol],
    nullable_nonterminals: &BTreeSet<NonTerminal>,
) -> Option<(usize, &'a Symbol)> {
    for (k, sym) in rhs.iter().enumerate().rev() {
        let is_nullable = match sym {
            Symbol::Terminal(_) => false,
            Symbol::NonTerminal(nt) => nullable_nonterminals.contains(nt),
        };
        if !is_nullable {
            return Some((k, sym));
        }
    }
    None
}

/// Resolves right-recursion in a grammar by converting it to left-recursion,
/// which GLR parsers can handle. This process handles both direct and indirect
/// right-recursion.
///
/// The algorithm is analogous to the standard algorithm for eliminating left-recursion:
/// 1. Order the non-terminals `N_1, ..., N_k`.
/// 2. For each `N_i` in order:
///    a. Substitute any indirect right-recursion `N_i -> ... N_j ...` (where `j < i`)
///       by replacing `N_j` with its productions. This is repeated until no such
///       indirect recursions remain.
///    b. Eliminate any direct right-recursion for `N_i` by introducing a new
///       non-terminal. A rule `N_i -> α N_i` is converted to a left-recursive
///       form `N_i -> N_i' β` and `N_i' -> N_i' α | ε`.
pub fn resolve_right_recursion(
    productions: &mut Vec<Production>,
    mut new_name_generator: impl FnMut(&str) -> String,
) {
    // 1. Get all non-terminals and establish an order.
    let mut all_nonterminals_set = BTreeSet::new();
    for p in productions.iter() {
        all_nonterminals_set.insert(p.lhs.clone());
        for s in &p.rhs {
            if let Symbol::NonTerminal(nt) = s {
                all_nonterminals_set.insert(nt.clone());
            }
        }
    }
    let ordered_nonterminals: Vec<NonTerminal> = all_nonterminals_set.into_iter().collect();
    let nt_to_order: HashMap<NonTerminal, usize> = ordered_nonterminals
        .iter()
        .enumerate()
        .map(|(i, nt)| (nt.clone(), i))
        .collect();

    // 2. Compute nullable non-terminals from the current set of productions.
    let nullable_nonterminals = compute_nullable_nonterminals(productions);

    let mut current_productions = productions.clone();

    // 3. Main loop for eliminating indirect and then direct right-recursion.
    for i in 0..ordered_nonterminals.len() {
        let nt_i = &ordered_nonterminals[i];

        // a. Substitute to eliminate indirect right-recursion
        let mut changed_in_substitution_loop = true;
        while changed_in_substitution_loop {
            changed_in_substitution_loop = false;

            let substitution_map: HashMap<NonTerminal, Vec<Vec<Symbol>>> = (0..i)
                .map(|j| {
                    let nt_j = &ordered_nonterminals[j];
                    let prods = current_productions.iter()
                        .filter(|p| &p.lhs == nt_j)
                        .map(|p| p.rhs.clone())
                        .collect();
                    (nt_j.clone(), prods)
                })
                .collect();

            let mut next_iteration_productions = Vec::new();
            for prod_i in current_productions {
                if &prod_i.lhs != nt_i {
                    next_iteration_productions.push(prod_i);
                    continue;
                }

                if let Some((k, Symbol::NonTerminal(nt_j))) = find_rightmost_non_nullable(&prod_i.rhs, &nullable_nonterminals) {
                    if let Some(prods_for_j) = substitution_map.get(nt_j) {
                        // Indirect right-recursion detected: N_i -> α N_j NullableSuffix
                        changed_in_substitution_loop = true;
                        
                        for prod_j_rhs in prods_for_j {
                            let mut new_rhs = prod_i.rhs[..k].to_vec();
                            new_rhs.extend_from_slice(prod_j_rhs);
                            new_rhs.extend_from_slice(&prod_i.rhs[k+1..]);
                            next_iteration_productions.push(Production {
                                lhs: nt_i.clone(),
                                rhs: new_rhs,
                            });
                        }
                    } else {
                        next_iteration_productions.push(prod_i);
                    }
                } else {
                    next_iteration_productions.push(prod_i);
                }
            }
            current_productions = next_iteration_productions;
        }

        // b. Eliminate direct right-recursion for nt_i.
        let (prods_for_nti, other_prods): (Vec<_>, Vec<_>) = current_productions.into_iter().partition(|p| &p.lhs == nt_i);
        current_productions = other_prods;

        let mut recursive_alphas = Vec::new();
        let mut non_recursive_gammas = Vec::new();

        for p in prods_for_nti {
            if let Some((k, Symbol::NonTerminal(nt))) = find_rightmost_non_nullable(&p.rhs, &nullable_nonterminals) {
                if nt == nt_i {
                    // This is a recursive rule: N_i -> α N_i NullableSuffix
                    let alpha = p.rhs[..k].to_vec();
                    recursive_alphas.push(alpha);
                    continue;
                }
            }
            // If we are here, it's a non-recursive rule.
            non_recursive_gammas.push(p.rhs);
        }

        if !recursive_alphas.is_empty() {
            // Create new non-terminal N_i'
            let new_nt_name = new_name_generator(&format!("{}RR", nt_i.0));
            let new_nt = NonTerminal(new_nt_name);

            // Create productions for N_i' -> N_i' α | ε
            for alpha in recursive_alphas {
                let mut new_rhs = vec![Symbol::NonTerminal(new_nt.clone())];
                new_rhs.extend(alpha);
                current_productions.push(Production {
                    lhs: new_nt.clone(),
                    rhs: new_rhs,
                });
            }
            // Add N_i' -> ε
            current_productions.push(Production {
                lhs: new_nt.clone(),
                rhs: vec![],
            });

            // Modify original rules: N_i -> N_i' γ
            for gamma in &non_recursive_gammas {
                let mut new_rhs = vec![Symbol::NonTerminal(new_nt.clone())];
                new_rhs.extend(gamma.clone());
                current_productions.push(Production {
                    lhs: nt_i.clone(),
                    rhs: new_rhs,
                });
            }
            // If all rules for N_i were right-recursive, it can only produce infinite sequences.
            // In this case, we make N_i -> N_i' so it can at least produce ε via N_i'.
            if non_recursive_gammas.is_empty() {
                current_productions.push(Production {
                    lhs: nt_i.clone(),
                    rhs: vec![Symbol::NonTerminal(new_nt.clone())],
                });
            }
        } else {
            // No direct right-recursion, just add the rules back.
            for gamma in non_recursive_gammas {
                current_productions.push(Production {
                    lhs: nt_i.clone(),
                    rhs: gamma,
                });
            }
        }
    }

    // Update the original productions vector
    *productions = current_productions;
}
