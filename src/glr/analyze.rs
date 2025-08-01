use std::cmp::PartialEq;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use bimap::BiBTreeMap;
use kdam::{tqdm, BarExt};
use crate::glr::automaton::{compute_first_sets_for_nonterminals, compute_follow_sets_for_nonterminals, compute_nonterminal_nullability, compute_closure, Nullability};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::table::{Goto, NonTerminalID, Table, StateID};


/// Checks for non-terminals used in rule RHS but never defined in LHS.
pub fn check_for_undefined_non_terminals(productions: &[Production]) -> Vec<String> {
    let mut lhs_nonterms: BTreeSet<NonTerminal> = BTreeSet::new();
    let mut rhs_nonterms: BTreeSet<NonTerminal> = BTreeSet::new();

    for prod in productions {
        lhs_nonterms.insert(prod.lhs.clone());
        for symbol in &prod.rhs {
            if let Symbol::NonTerminal(nt) = symbol {
                rhs_nonterms.insert(nt.clone());
            }
        }
    }

    let missing_nonterms: BTreeSet<_> = rhs_nonterms.difference(&lhs_nonterms).collect();
    if !missing_nonterms.is_empty() {
        let missing_nonterm_strings: BTreeSet<_> = missing_nonterms.into_iter().map(|nt| nt.0.clone()).collect();
        vec![format!(
            "Non-terminal(s) used in rule RHS but never defined in LHS: {:?}",
            missing_nonterm_strings
        )]
    } else {
        Vec::new()
    }
}

/// Helper for check_for_length_1_recursion. Detects all elementary cycles in a graph.
fn detect_all_cycles_recursive(
    nt: &NonTerminal,
    graph: &BTreeMap<NonTerminal, BTreeSet<NonTerminal>>,
    visiting: &mut BTreeSet<NonTerminal>, // Nodes currently in the recursion stack for the current path
    visited: &mut BTreeSet<NonTerminal>,  // Nodes that have been fully explored
    path: &mut Vec<NonTerminal>,          // Current path for cycle detection
    cycles: &mut BTreeSet<Vec<NonTerminal>>, // Set to store unique, canonicalized cycles
) {
    visiting.insert(nt.clone());
    path.push(nt.clone());

    if let Some(neighbors) = graph.get(nt) {
        for neighbor in neighbors {
            if visiting.contains(neighbor) {
                // Cycle detected. Extract it from the current path.
                let cycle_start_index = path.iter().position(|n| n == neighbor).unwrap_or(0);
                let mut cycle: Vec<_> = path[cycle_start_index..].to_vec();

                // Canonicalize the cycle by rotating it to start with the lexicographically smallest element.
                // This ensures that A -> B -> A and B -> A -> B are treated as the same cycle.
                if !cycle.is_empty() {
                    let min_node_pos = cycle.iter().enumerate()
                        .min_by_key(|&(_, n)| n)
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    cycle.rotate_left(min_node_pos);
                }
                cycles.insert(cycle);
                continue; // Continue to find other cycles from this node.
            }

            if !visited.contains(neighbor) {
                detect_all_cycles_recursive(neighbor, graph, visiting, visited, path, cycles);
            }
        }
    }

    visiting.remove(nt);
    visited.insert(nt.clone());
    path.pop();
}

/// Checks for length-1 recursion (e.g., A ::= A, A ::= B; B ::= A), considering nullable prefixes.
pub fn check_for_length_1_recursion(productions: &[Production]) -> Vec<String> {
    let nullability_map = compute_nonterminal_nullability(productions);
    let all_nonterminals: BTreeSet<NonTerminal> = productions.iter().flat_map(|p| {
        let mut nts = vec![p.lhs.clone()];
        for s in &p.rhs {
            if let Symbol::NonTerminal(nt) = s {
                nts.push(nt.clone());
            }
        }
        nts
    }).collect();

    // Build a graph where an edge A -> B exists if a rule A ::= Nullable* B Nullable* exists.
    let mut unit_graph: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
    for nt in &all_nonterminals {
        unit_graph.entry(nt.clone()).or_default();
    }

    for prod in productions {
        let lhs = &prod.lhs;
        let rhs = &prod.rhs;

        let non_nullable_symbols: Vec<&Symbol> = rhs
            .iter()
            .filter(|symbol| match symbol {
                Symbol::Terminal(_) => true,
                Symbol::NonTerminal(nt) => !nullability_map.get(nt).map_or(false, |n| n.is_nullable()),
            })
            .collect();

        if non_nullable_symbols.len() == 1 {
            if let Symbol::NonTerminal(target_nt) = non_nullable_symbols[0] {
                unit_graph
                    .entry(lhs.clone())
                    .or_default()
                    .insert(target_nt.clone());
            }
        }
    }

    // Detect all unique cycles in the unit graph using DFS.
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let mut cycles = BTreeSet::new();
    let sorted_nonterminals: Vec<_> = all_nonterminals.iter().collect();

    for nt in sorted_nonterminals {
        if !visited.contains(nt) {
            let mut path = Vec::new();
            detect_all_cycles_recursive(nt, &unit_graph, &mut visiting, &mut visited, &mut path, &mut cycles);
        }
    }

    // Format errors for each unique cycle found.
    cycles.into_iter().map(|cycle| {
        let mut cycle_nodes_for_display: Vec<_> = cycle.iter().map(|n| n.0.as_str()).collect();
        cycle_nodes_for_display.push(cycle[0].0.as_str()); // Close the loop for display.

        let cycle_path_str = cycle_nodes_for_display.join(" -> ");
        let recursion_type = if cycle.len() == 1 { "Direct" } else { "Indirect" };

        format!("{} length-1 recursion cycle detected: {}", recursion_type, cycle_path_str)
    }).collect()
}

/// Checks for left-nullable left recursion (e.g., A ::= B A ..., where B is nullable).
pub fn check_for_left_nullable_left_recursion(productions: &[Production]) -> Vec<String> {
    let nullability_map = compute_nonterminal_nullability(productions);
    let mut errors = Vec::new();

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
                            Symbol::NonTerminal(prefix_nt) => nullability_map.get(prefix_nt).map_or(false, |n| n.is_nullable()),
                            Symbol::Terminal(_) => false, // Terminals are not nullable
                        });

                        if prefix_is_nullable {
                            errors.push(format!("Left-nullable left recursion detected in rule '{} ::= {:?}'. The prefix '{:?}' before the recursive non-terminal '{}' is nullable.", lhs.0, rhs, prefix, lhs.0));
                        }
                    }
                    // We only care about the first instance of recursion in a rule.
                    break;
                }
            }
        }
    }
    errors
}

/// Validates the grammar for common issues, collecting all errors.
///
/// Checks for:
/// 1. Undefined non-terminals.
/// 2. Length-1 recursion (direct or indirect).
/// 3. Left-nullable left recursion.
pub fn validate(productions: &[Production]) -> Result<(), String> {
    let mut errors = Vec::new();

    errors.extend(check_for_undefined_non_terminals(productions));
    errors.extend(check_for_length_1_recursion(productions));
    errors.extend(check_for_left_nullable_left_recursion(productions));

    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Grammar validation failed with {} error(s):\n- {}",
            errors.len(),
            errors.join("\n- ")
        ))
    }
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

pub fn simplify_grammar(initial_productions: &[Production]) -> Vec<Production> {
    todo!()
}

/// Helper function to find the last symbol in a rule's RHS that is not a nullable non-terminal.
/// Returns the index and the symbol if found.
fn find_last_non_nullable_symbol<'a>(
    rhs: &'a [Symbol],
    nullability_map: &BTreeMap<NonTerminal, Nullability>,
) -> Option<(usize, &'a Symbol)> {
    for (i, symbol) in rhs.iter().enumerate().rev() {
        let is_nullable = match symbol {
            Symbol::NonTerminal(nt) => nullability_map.get(nt).map_or(false, |n| n.is_nullable()),
            Symbol::Terminal(_) => false,
        };
        if !is_nullable {
            return Some((i, symbol));
        }
    }
    None
}

pub fn compute_terminal_follow_sets(productions: &[Production]) -> BTreeMap<Terminal, BTreeSet<Terminal>> {
    let first_sets = compute_first_sets_for_nonterminals(productions);
    let nullability_map = compute_nonterminal_nullability(productions);
    let nonterminal_follow_sets = compute_follow_sets_for_nonterminals(productions, &first_sets, &nullability_map);

    let mut terminal_follows: BTreeMap<Terminal, BTreeSet<Terminal>> = BTreeMap::new();

    for production in productions {
        let lhs = &production.lhs;
        let rhs = &production.rhs;

        for (i, symbol) in rhs.iter().enumerate() {
            if let Symbol::Terminal(t) = symbol {
                // We found a terminal 't'. Now, find what can follow it in this rule.
                let mut all_following_are_nullable = true;

                // Look at the rest of the production's RHS (the suffix)
                for next_symbol in &rhs[i + 1..] {
                    match next_symbol {
                        Symbol::Terminal(next_t) => {
                            // The next symbol is a terminal. It's in the follow set.
                            terminal_follows.entry(t.clone()).or_default().insert(next_t.clone());
                            all_following_are_nullable = false;
                            break; // Found a non-nullable symbol, so we're done with this suffix.
                        }
                        Symbol::NonTerminal(next_nt) => {
                            // The next symbol is a non-terminal. Add its FIRST set.
                            if let Some(first_set_for_next_nt) = first_sets.get(next_nt) {
                                terminal_follows
                                    .entry(t.clone())
                                    .or_default()
                                    .extend(first_set_for_next_nt.iter().cloned());
                            }
                            // If the non-terminal is not nullable, we stop looking further.
                            if !nullability_map.get(next_nt).map_or(false, |n| n.is_nullable()) {
                                all_following_are_nullable = false;
                                break;
                            }
                        }
                    }
                }

                // If the rest of the RHS was empty or consisted entirely of nullable non-terminals,
                // then FOLLOW(t) must also include FOLLOW(lhs).
                if all_following_are_nullable {
                    if let Some(follow_set_for_lhs) = nonterminal_follow_sets.get(lhs) {
                        terminal_follows
                            .entry(t.clone())
                            .or_default()
                            // filter_map removes None (EOF) and unwraps Some(T) to T
                            .extend(follow_set_for_lhs.iter().filter_map(|opt_t| opt_t.clone()));
                    }
                }
            }
        }
    }

    terminal_follows
}

/// Creates a closure that generates unique non-terminal names, suitable for `resolve_right_recursion`.
///
/// The generator ensures that new names do not conflict with existing non-terminal names
/// or with names it has generated previously. A typical generated name for a non-terminal `A`
/// would be `A_rr`, `A_rr_2`, etc., to avoid collisions.
///
/// # Arguments
/// * `all_nonterminals` - A set of all non-terminal names currently in the grammar.
///
/// # Returns
/// A closure `FnMut(&str) -> String` that can be passed to `resolve_right_recursion`.
pub fn create_unique_name_generator(all_nonterminals: &BTreeSet<NonTerminal>) -> impl FnMut(&str) -> String {
    let mut existing_names: BTreeSet<String> = all_nonterminals.iter().map(|nt| nt.0.clone()).collect();

    move |base_name: &str| {
        // First attempt: base_name + "_rr" (for right-recursion elimination)
        let mut new_name = format!("{}_rr", base_name);
        let mut counter = 1;

        // Check for collisions and increment a suffix if needed
        while existing_names.contains(&new_name) {
            counter += 1;
            new_name = format!("{}_rr_{}", base_name, counter);
        }

        // Reserve the new name to avoid future collisions within the generator.
        existing_names.insert(new_name.clone());
        new_name
    }
}

pub fn resolve_right_recursion(
    productions: &mut Vec<Production>,
    mut new_name_generator: impl FnMut(&str) -> String,
) {
    todo!("resolve_right_recursion");
}

pub fn resolve_direct_right_recursion(
    productions: &mut Vec<Production>,
    mut new_name_generator: impl FnMut(&str) -> String,
) {
    // This function transforms direct right-recursion into left-recursion while preserving
    // the original order of productions as much as possible.
    //
    // The transformation for a non-terminal `A` with rules:
    //   A -> α₁ A | ... | αₘ A  (right-recursive rules, where αᵢ is a prefix)
    //   A -> β₁ | ... | βₙ      (non-recursive rules)
    // is to replace them with:
    //   A  -> A' β₁ | ... | A' βₙ
    //   A' -> A' α₁ | ... | A' αₘ | ε
    // where `A'` is a new non-terminal. This generates the same language `(α₁|...|αₘ)* (β₁|...|βₙ)`
    // using left-recursion.
    //
    // The implementation preserves order by iterating through the original production list.
    // When it first encounters a rule for a right-recursive non-terminal, it replaces all
    // rules for that non-terminal with the new, transformed set of rules at that position.

    // 1. Group productions by LHS to easily access all rules for a given non-terminal.
    // The BTreeMap is just for efficient lookup; we don't iterate over its keys for ordering.
    let mut prods_by_lhs: BTreeMap<NonTerminal, Vec<Production>> = BTreeMap::new();
    for prod in productions.iter().cloned() {
        prods_by_lhs.entry(prod.lhs.clone()).or_default().push(prod);
    }

    // 2. Identify all non-terminals that have simple direct right-recursive rules.
    let mut recursive_nts = BTreeSet::new();
    for (nt, prods_for_nt) in &prods_by_lhs {
        let is_recursive = prods_for_nt.iter().any(|p| {
            // A rule `A -> α A` is simple direct right-recursive if:
            // a) `α` is not empty (i.e., rule is not `A -> A`).
            if p.rhs.len() < 2 { return false; }
            // b) The last symbol is `A`.
            if p.rhs.last() != Some(&Symbol::NonTerminal(p.lhs.clone())) { return false; }
            // c) `α` (the prefix) does not contain `A`.
            let alpha = &p.rhs[..p.rhs.len() - 1];
            !alpha.contains(&Symbol::NonTerminal(p.lhs.clone()))
        });
        if is_recursive {
            recursive_nts.insert(nt.clone());
        }
    }

    // 3. Build the new production list, preserving order.
    let mut new_productions = Vec::new();
    let mut processed_recursive_nts = BTreeSet::new();

    for prod in productions.iter().cloned() {
        let lhs = &prod.lhs;

        if !recursive_nts.contains(lhs) {
            // This production's LHS is not right-recursive, so we can keep the production as is.
            new_productions.push(prod);
            continue;
        }

        // The LHS is right-recursive. We need to process all of its rules at once.
        // We do this only the first time we encounter a rule for this LHS.
        if processed_recursive_nts.contains(lhs) {
            // We've already handled this NT and its rules, so we skip this and subsequent productions for it.
            continue;
        }
        processed_recursive_nts.insert(lhs.clone());

        // --- Perform the transformation for `lhs` ---
        let prods_for_nt = prods_by_lhs.get(lhs).unwrap();

        let (recursive_rules, other_rules): (Vec<_>, Vec<_>) = prods_for_nt.iter().cloned().partition(|p| {
            if p.rhs.len() < 2 { return false; }
            if p.rhs.last() != Some(&Symbol::NonTerminal(p.lhs.clone())) { return false; }
            let alpha = &p.rhs[..p.rhs.len() - 1];
            !alpha.contains(&Symbol::NonTerminal(p.lhs.clone()))
        });

        // `recursive_rules` is guaranteed to be non-empty because `lhs` is in `recursive_nts`.
        let new_nt = NonTerminal(new_name_generator(&lhs.0));
        crate::debug!(2, "Resolving direct right-recursion for '{}', creating new non-terminal '{}'", lhs.0, new_nt.0);

        // Create new rules for the original non-terminal `A`: `A -> A' βⱼ`.
        // The order of these new rules is based on the original order of the `β` rules.
        for non_rec_rule in &other_rules {
            let mut new_rhs = vec![Symbol::NonTerminal(new_nt.clone())];
            new_rhs.extend_from_slice(&non_rec_rule.rhs);
            let new_prod = Production { lhs: lhs.clone(), rhs: new_rhs };
            crate::debug!(2, "  Transforming non-recursive rule '{}' -> '{}'", non_rec_rule, new_prod);
            new_productions.push(new_prod);
        }

        // Create rules for the new non-terminal `A'`: `A' -> A' αᵢ` and `A' -> ε`.
        for rec_rule in &recursive_rules {
            let alpha = &rec_rule.rhs[..rec_rule.rhs.len() - 1];
            let mut new_rhs = vec![Symbol::NonTerminal(new_nt.clone())];
            new_rhs.extend_from_slice(alpha);
            let new_prod = Production { lhs: new_nt.clone(), rhs: new_rhs };
            crate::debug!(2, "  Transforming recursive rule '{}' -> '{}'", rec_rule, new_prod);
            new_productions.push(new_prod);
        }
        let epsilon_prod = Production { lhs: new_nt.clone(), rhs: vec![] }; // A' -> ε
        crate::debug!(2, "  Adding new epsilon rule: '{}'", epsilon_prod);
        new_productions.push(epsilon_prod);
    }

    // 4. Replace the original productions with the new set.
    productions.clear();
    productions.extend(new_productions);
}

pub fn inline_null_productions(productions: &[Production]) -> Vec<Production> {
    let nullability_map = compute_nonterminal_nullability(productions);

    let mut final_productions = Vec::new();
    final_productions.push(productions[0].clone());

    for original_prod in &productions[0..] {
        // For each original production, we generate all possible new productions
        // by removing any combination of nullable non-terminals from its RHS.

        // A worklist of RHS variants to process.
        let mut worklist: VecDeque<Vec<Symbol>> = VecDeque::new();
        let mut generated_rhss: Vec<Vec<Symbol>> = Vec::new();

        // Start with the original RHS.
        worklist.push_back(original_prod.rhs.clone());

        'worklist: while let Some(current_rhs) = worklist.pop_front() {
            // Iterate over the symbols of the current RHS variant.
            for i in 0..current_rhs.len() {
                if let Symbol::NonTerminal(nt) = &current_rhs[i] {
                    // If we find a nullable non-terminal...
                    let nullability = nullability_map.get(nt).copied().unwrap_or(Nullability::NotNull);
                    if nullability.is_nullable() {
                        // ...create a new RHS variant with it removed.
                        let mut new_rhs = current_rhs.clone();
                        new_rhs.remove(i);

                        generated_rhss.push(new_rhs.clone());
                        worklist.push_back(new_rhs);
                        if nullability == Nullability::Null {
                            continue 'worklist;
                        }
                    }
                }
            }
            generated_rhss.push(current_rhs.clone());
        }

        // Add all generated variants as new productions with the original LHS.
        for rhs in generated_rhss.into_iter().rev() {
            final_productions.push(Production {
                lhs: original_prod.lhs.clone(),
                rhs,
            });
        }
    }

    // Finally, remove all productions that are now null (e.g., A -> ε),
    // as they have been inlined.
    let start_prod_nonterms: BTreeSet<_> = productions[0].rhs.iter().filter_map(|s| match s {
        Symbol::NonTerminal(nt) => Some(nt.clone()),
        _ => None,
    }).collect();
    final_productions.retain(|p| !p.rhs.is_empty() || start_prod_nonterms.contains(&p.lhs));
    // Remove duplicates
    final_productions.dedup();
    final_productions
}

pub fn inline_unit_productions(productions: &[Production]) -> Vec<Production> {
    todo!()
}
