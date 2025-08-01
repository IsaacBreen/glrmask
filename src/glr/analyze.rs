use std::cmp::PartialEq;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use bimap::BiBTreeMap;
use kdam::{tqdm, BarExt};
use crate::glr::automaton::{compute_first_sets_for_nonterminals, compute_follow_sets_for_nonterminals, compute_nullable_nonterminals, compute_closure};
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
    let nullable_nonterminals = compute_nullable_nonterminals(productions);
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
                Symbol::NonTerminal(nt) => !nullable_nonterminals.contains(nt),
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
    let nullable_nonterminals = compute_nullable_nonterminals(productions);
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
                            Symbol::NonTerminal(prefix_nt) => nullable_nonterminals.contains(prefix_nt),
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

pub fn validate_start_production_ends_with_terminal(
    productions: &[Production],
    start_production_id: usize,
) -> Result<(), String> {
    if start_production_id >= productions.len() {
        return Err(format!("Invalid start production ID: {}. Must be less than the number of productions: {}", start_production_id, productions.len()));
    }

    let start_prod = &productions[start_production_id];
    if !start_prod.rhs.is_empty() && !matches!(start_prod.rhs.last(), Some(Symbol::Terminal(_))) {
        return Err(format!("Start production [{}] does not end with a terminal symbol. Last symbol in RHS is: {:?}", start_prod, start_prod.rhs.last()));
    }

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

/// Helper function to find the last symbol in a rule's RHS that is not a nullable non-terminal.
/// Returns the index and the symbol if found.
fn find_last_non_nullable_symbol<'a>(
    rhs: &'a [Symbol],
    nullable_set: &BTreeSet<NonTerminal>,
) -> Option<(usize, &'a Symbol)> {
    for (i, symbol) in rhs.iter().enumerate().rev() {
        let is_nullable = match symbol {
            Symbol::NonTerminal(nt) => nullable_set.contains(nt),
            Symbol::Terminal(_) => false,
        };
        if !is_nullable {
            return Some((i, symbol));
        }
    }
    None
}

pub fn compute_terminal_follow_sets(productions: &[Production], start_production_id: usize) -> BTreeMap<Terminal, BTreeSet<Terminal>> {
    let first_sets = compute_first_sets_for_nonterminals(productions);
    let nullable_nonterminals = compute_nullable_nonterminals(productions);
    let nonterminal_follow_sets = compute_follow_sets_for_nonterminals(productions, start_production_id, &first_sets, &nullable_nonterminals);

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
                            if !nullable_nonterminals.contains(next_nt) {
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

/// Checks if two goto maps are compatible.
///
/// Two goto maps are compatible if, for all non-terminal IDs they have in common,
/// the corresponding `Goto` actions are identical.
fn are_gotos_compatible(
    gotos1: &BTreeMap<NonTerminalID, Goto>,
    gotos2: &BTreeMap<NonTerminalID, Goto>,
) -> bool {
    // Iterate over the keys of the first map.
    for (nt_id, goto1) in gotos1 {
        // If the second map also contains this key, check if the values are equal.
        if let Some(goto2) = gotos2.get(nt_id) {
            if goto1 != goto2 {
                // Incompatibility found.
                return false;
            }
        }
    }
    // No incompatibilities found.
    true
}

/// Finds pairs of states in a parse table that have identical actions but only
/// "compatible" gotos.
///
/// This is useful for identifying states that might be candidates for merging in
/// LALR(1) parsers, or for general table analysis.
///
/// A pair of states is reported if:
/// 1. Their shift/reduce actions are identical across all phases (including default reduces).
/// 2. Their `goto` maps are "compatible". Two goto maps are compatible if for every
///    non-terminal key they have in common, the target `Goto` action is also identical.
///
/// # Arguments
/// * `table` - The `Table` to analyze.
///
/// # Returns
/// A vector of tuples, where each tuple `(StateID, StateID)` represents a pair of
/// compatible states.
pub fn find_compatible_states(table: &Table) -> Vec<(StateID, StateID)> {
    let mut compatible_pairs = Vec::new();
    let states_and_rows: Vec<_> = table.iter().collect();

    // Iterate over all unique pairs of states.
    for i in 0..states_and_rows.len() {
        for j in (i + 1)..states_and_rows.len() {
            let (state_id1, row1) = states_and_rows[i];
            let (state_id2, row2) = states_and_rows[j];

            // Condition 1: Check for identical actions across all phases.
            let actions_are_identical = row1.shifts_and_reduces_without_default_reduce == row2.shifts_and_reduces_without_default_reduce
                && row1.shifts_and_reduces_full == row2.shifts_and_reduces_full
                && row1.default_reduce == row2.default_reduce;

            if actions_are_identical {
                // Condition 2: Check for compatible gotos.
                if are_gotos_compatible(&row1.gotos, &row2.gotos) {
                    compatible_pairs.push((*state_id1, *state_id2));
                }
            }
        }
    }

    compatible_pairs
}

/// Helper struct for `eliminate_unit_productions` to manage pre-computed grammar info.
struct UnitProductionInfo {
    unit_prod_ids: BTreeSet<usize>,
    nodes: BTreeSet<NonTerminal>,
    leaves: BTreeSet<NonTerminal>,
    derives_map: BTreeMap<NonTerminal, BTreeSet<NonTerminal>>, // Transitive closure A => B
    nt_to_leaf_map: BTreeMap<NonTerminal, NonTerminal>, // Map from a node to an arbitrary leaf
}

/// Pre-computes information about unit productions (A -> B).
fn compute_unit_production_info(productions: &[Production]) -> UnitProductionInfo {
    let unit_prod_ids: BTreeSet<usize> = productions.iter().enumerate()
        .filter(|(_, p)| p.rhs.len() == 1 && matches!(p.rhs[0], Symbol::NonTerminal(_)))
        .map(|(i, _)| i)
        .collect();

    let mut unit_graph: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
    let all_nts: BTreeSet<NonTerminal> = productions.iter().flat_map(|p| {
        let mut nts = vec![p.lhs.clone()];
        if let Some(Symbol::NonTerminal(nt)) = p.rhs.get(0) {
            nts.push(nt.clone());
        }
        nts
    }).collect();

    for nt in &all_nts {
        unit_graph.entry(nt.clone()).or_default();
    }
    for &i in &unit_prod_ids {
        let p = &productions[i];
        if let Symbol::NonTerminal(rhs_nt) = &p.rhs[0] {
            unit_graph.get_mut(&p.lhs).unwrap().insert(rhs_nt.clone());
        }
    }

    let nodes: BTreeSet<NonTerminal> = unit_graph.iter().filter(|(_, v)| !v.is_empty()).map(|(k, _)| k.clone()).collect();
    let leaves: BTreeSet<NonTerminal> = unit_graph.values().flatten().filter(|nt| !nodes.contains(*nt)).cloned().collect();

    // Compute transitive closure (derives_map) using Floyd-Warshall-like iteration
    let mut derives_map = unit_graph.clone();
    for k in &all_nts {
        for i in &all_nts {
            for j in &all_nts {
                if derives_map[i].contains(k) && derives_map[k].contains(j) {
                    derives_map.get_mut(i).unwrap().insert(j.clone());
                }
            }
        }
    }

    // For each node, find an arbitrary leaf it can derive
    let mut nt_to_leaf_map = BTreeMap::new();
    for node in &nodes {
        if let Some(derived_leaves) = derives_map.get(node).and_then(|derived| {
            derived.iter().filter(|nt| leaves.contains(*nt)).next()
        }) {
            nt_to_leaf_map.insert(node.clone(), derived_leaves.clone());
        }
    }

    UnitProductionInfo { unit_prod_ids, nodes, leaves, derives_map, nt_to_leaf_map }
}

use crate::glr::table::{Stage6Row, Stage6ShiftsAndReduces};
use crate::glr::items::Item;

type Stage6Table = BTreeMap<BTreeSet<Item>, Stage6Row>;

/// Checks if a state has a unit reduction action.
fn has_unit_reduction(
    state: &BTreeSet<Item>,
    table: &Stage6Table,
    unit_prod_ids: &BTreeSet<usize>,
) -> bool {
    if let Some(row) = table.get(state) {
        row.shifts_and_reduces.values().any(|action|
            !action.reduces.is_empty() && action.reduces.iter().any(|pid| unit_prod_ids.contains(&pid.0))
        )
    } else {
        false
    }
}

/// Merges multiple states into a single new state.
fn combine_states<'a>(
    states_to_combine: &BTreeSet<&'a BTreeSet<Item>>,
    table: &Stage6Table,
    memo: &mut BTreeMap<BTreeSet<&'a BTreeSet<Item>>, BTreeSet<Item>>,
    new_table: &mut Stage6Table,
) -> BTreeSet<Item> {
    if let Some(combined) = memo.get(states_to_combine) {
        return combined.clone();
    }

    let mut new_item_set = BTreeSet::new();
    let mut new_row = Stage6Row::default();

    for &state in states_to_combine {
        new_item_set.extend(state.iter().cloned());
        if let Some(row_to_merge) = table.get(state) {
            // Merge gotos
            for (nt, goto_target) in &row_to_merge.gotos {
                if let Some(existing_target) = new_row.gotos.get(nt) {
                    assert_eq!(existing_target, goto_target, "Unit production elimination created GOTO conflict");
                } else {
                    new_row.gotos.insert(nt.clone(), goto_target.clone());
                }
            }
            // Merge shifts and reduces
            for (term, action) in &row_to_merge.shifts_and_reduces {
                let new_action = new_row.shifts_and_reduces.entry(term.clone()).or_default();
                if let Some(shift_target) = &action.shift {
                    if new_action.shift.is_some() && &new_action.shift != &action.shift {
                        panic!("Unit production elimination created shift/shift conflict");
                    }
                    new_action.shift = Some(shift_target.clone());
                }
                new_action.reduces.extend(action.reduces.iter());
            }
        }
    }

    // Check if an equivalent state already exists in the new table
    if let Some(existing_state) = new_table.iter().find(|(_, row)| **row == new_row).map(|(k, _)| k.clone()) {
        memo.insert(states_to_combine.clone(), existing_state.clone());
        return existing_state;
    }

    memo.insert(states_to_combine.clone(), new_item_set.clone());
    new_table.insert(new_item_set.clone(), new_row);
    new_item_set
}

/// Implements Pager's algorithm to eliminate unit productions from a Stage 6 parse table.
pub fn eliminate_unit_productions(
    stage_6_table: &mut Stage6Table,
    productions: &mut Vec<Production>,
    start_production_id: usize,
) {
    let initial_state_count = stage_6_table.len();
    let info = compute_unit_production_info(productions);

    // --- Iterative Merging ---
    let mut current_table = stage_6_table.clone();

    loop {
        let mut changed = false;
        let mut next_table = current_table.clone();
        let mut memo: BTreeMap<BTreeSet<&BTreeSet<Item>>, BTreeSet<Item>> = BTreeMap::new();
        let states_to_process: Vec<_> = current_table.keys().collect();

        for s in states_to_process {
            let s_row = current_table.get(s).unwrap();
            let mut new_s_row = s_row.clone();

            for leaf in &info.leaves {
                if let Some(t_successor) = s_row.gotos.get(leaf) {
                    if has_unit_reduction(t_successor, &current_table, &info.unit_prod_ids) {
                        let mut states_to_combine: BTreeSet<&BTreeSet<Item>> = BTreeSet::new();
                        // Find all ancestors of `leaf` that have a GOTO from `s`.
                        for (nt, derived_nts) in &info.derives_map {
                            if derived_nts.contains(leaf) {
                                if let Some(ancestor_successor) = s_row.gotos.get(nt) {
                                    states_to_combine.insert(ancestor_successor);
                                }
                            }
                        }
                        // Also include the leaf's own successor
                        if let Some(leaf_successor) = s_row.gotos.get(leaf) {
                            states_to_combine.insert(leaf_successor);
                        }

                        if !states_to_combine.is_empty() {
                            let combined_state = combine_states(&states_to_combine, &current_table, &mut memo, &mut next_table);
                            if new_s_row.gotos.insert(leaf.clone(), combined_state) != Some(t_successor.clone()) {
                                changed = true;
                            }
                        }
                    }
                }
            }
            next_table.insert(s.clone(), new_s_row);
        }

        if !changed {
            break;
        }
        current_table = next_table;
    }

    // --- Final Cleanup Steps ---
    // 3. Delete transitions on nodes.
    for row in current_table.values_mut() {
        row.gotos.retain(|nt, _| !info.nodes.contains(nt));
    }

    // 4. Delete unreachable states.
    let start_item = Item { production: productions[start_production_id].clone(), dot_position: 0, lookahead: None };
    let start_state_key = current_table.keys().find(|k| k.contains(&start_item)).expect("Start state not found").clone();

    let mut reachable_states: BTreeSet<BTreeSet<Item>> = BTreeSet::new();
    let mut worklist: VecDeque<BTreeSet<Item>> = VecDeque::from([start_state_key]);

    while let Some(state) = worklist.pop_front() {
        if reachable_states.contains(&state) { continue; }
        if let Some(row) = current_table.get(&state) {
            for action in row.shifts_and_reduces.values() {
                if let Some(s) = &action.shift { worklist.push_back(s.clone()); }
            }
            for goto in row.gotos.values() {
                worklist.push_back(goto.clone());
            }
        }
        reachable_states.insert(state);
    }
    current_table.retain(|s, _| reachable_states.contains(s));

    // 5. Modify productions list for final parser.
    let mut new_productions = Vec::new();
    let mut old_pid_to_new_pid: BTreeMap<usize, usize> = BTreeMap::new();
    let mut new_idx = 0;

    for (old_idx, p) in productions.iter().enumerate() {
        if !info.unit_prod_ids.contains(&old_idx) {
            let mut new_p = p.clone();
            if info.nodes.contains(&p.lhs) {
                let leaf = info.nt_to_leaf_map.get(&p.lhs).expect("Node should derive a leaf");
                new_p.lhs = leaf.clone();
            }
            new_productions.push(new_p);
            old_pid_to_new_pid.insert(old_idx, new_idx);
            new_idx += 1;
        }
    }

    // Remap production IDs in the final table.
    for row in current_table.values_mut() {
        for action in row.shifts_and_reduces.values_mut() {
            action.reduces = action.reduces.iter()
                .filter_map(|old_pid| old_pid_to_new_pid.get(&old_pid.0))
                .map(|&new_idx| crate::glr::table::ProductionID(new_idx))
                .collect();
        }
    }

    let final_state_count = current_table.len();
    crate::debug!(2, "Unit production elimination complete. States reduced from {} to {}.", initial_state_count, final_state_count);

    *stage_6_table = current_table;
    *productions = new_productions;
}
