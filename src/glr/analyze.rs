use std::collections::HashMap;
use crate::glr::automaton::{
    compute_closure, compute_first_sets_for_nonterminals, compute_follow_sets_for_nonterminals,
    compute_nonterminal_nullability, compute_null_nonterminals, compute_nullable_nonterminals,
    Nullability,
};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::table::{Goto, NonTerminalID, StateID, Table};
use bimap::BiBTreeMap;
use kdam::{tqdm, BarExt};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use crate::finite_automata::{Expr, QuantifierType};
use crate::glr::grammar::regex_name;

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
        let missing_nonterm_strings: BTreeSet<_> =
            missing_nonterms.into_iter().map(|nt| nt.0.clone()).collect();
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
                    let min_node_pos = cycle
                        .iter()
                        .enumerate()
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
    let all_nonterminals: BTreeSet<NonTerminal> = productions
        .iter()
        .flat_map(|p| {
            let mut nts = vec![p.lhs.clone()];
            for s in &p.rhs {
                if let Symbol::NonTerminal(nt) = s {
                    nts.push(nt.clone());
                }
            }
            nts
        })
        .collect();

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
            detect_all_cycles_recursive(
                nt,
                &unit_graph,
                &mut visiting,
                &mut visited,
                &mut path,
                &mut cycles,
            );
        }
    }

    // Format errors for each unique cycle found.
    cycles
        .into_iter()
        .map(|cycle| {
            let mut cycle_nodes_for_display: Vec<_> = cycle.iter().map(|n| n.0.as_str()).collect();
            cycle_nodes_for_display.push(cycle[0].0.as_str()); // Close the loop for display.

            let cycle_path_str = cycle_nodes_for_display.join(" -> ");
            let recursion_type = if cycle.len() == 1 {
                "Direct"
            } else {
                "Indirect"
            };

            format!("{recursion_type} length-1 recursion cycle detected: {cycle_path_str}")
        })
        .collect()
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
                if nt == lhs {
                    // Found potential left recursion: A ::= ... A ...
                    // Check if all preceding symbols (if any) are nullable non-terminals
                    let prefix = &rhs[0..i];
                    if !prefix.is_empty() {
                        // Only check if there's a prefix
                        let prefix_is_nullable = prefix.iter().all(|sym| match sym {
                            Symbol::NonTerminal(prefix_nt) =>
                                nullable_nonterminals.contains(prefix_nt),
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

/// Computes the set of productive non-terminals (those that can derive a terminal string).
fn compute_productive_non_terminals(productions: &[Production]) -> BTreeSet<NonTerminal> {
    let mut productive_nts = BTreeSet::new();
    let mut changed = true;

    while changed {
        changed = false;
        for prod in productions {
            if productive_nts.contains(&prod.lhs) {
                continue;
            }

            // A rule's RHS is productive if all its symbols are productive.
            // Terminals are inherently productive. Non-terminals are productive if they are in our set.
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

/// Checks for non-terminals that cannot derive any terminal string.
pub fn check_for_non_productive_non_terminals(productions: &[Production]) -> Vec<String> {
    // Collect all non-terminals defined on the LHS. If a non-terminal is only used on the RHS,
    // `check_for_undefined_non_terminals` will have already caught it.
    let all_nonterminals: BTreeSet<NonTerminal> =
        productions.iter().map(|p| p.lhs.clone()).collect();
    let productive_nts = compute_productive_non_terminals(productions);

    let non_productive_nts: BTreeSet<_> = all_nonterminals.difference(&productive_nts).collect();

    if !non_productive_nts.is_empty() {
        let mut non_productive_strings: Vec<_> =
            non_productive_nts.into_iter().map(|nt| nt.0.clone()).collect();
        non_productive_strings.sort(); // For deterministic error messages
        vec![format!(
            "Non-terminal(s) are non-productive (cannot derive a terminal string): {:?}",
            non_productive_strings
        )]
    } else {
        Vec::new()
    }
}

/// Validates the grammar for common issues, collecting all errors.
///
/// Checks for:
/// 1. Undefined non-terminals.
/// 2. Non-productive non-terminals.
/// 3. Length-1 recursion (direct or indirect).
/// 4. Left-nullable left recursion.
pub fn validate(productions: &[Production]) -> Result<(), String> {
    let mut errors = Vec::new();

    errors.extend(check_for_undefined_non_terminals(productions));
    errors.extend(check_for_non_productive_non_terminals(productions));
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
pub fn remove_productions_with_undefined_nonterminals(
    initial_productions: &[Production],
    exempt: &[usize],
) -> Vec<Production> {
    let mut current_productions: Vec<(usize, Production)> =
        initial_productions.iter().cloned().enumerate().collect();

    loop {
        let mut defined_lhs_nonterminals: BTreeSet<NonTerminal> = BTreeSet::new();
        for (i, prod) in &current_productions {
            defined_lhs_nonterminals.insert(prod.lhs.clone());
        }
        let mut removed_productions: Vec<(usize, Production)> = Vec::new();
        let mut kept_productions: Vec<(usize, Production)> = Vec::new();
        for (i, prod) in current_productions {
            let keep = prod
                .rhs
                .iter()
                .all(|symbol| match symbol {
                    Symbol::Terminal(_) => true, // Terminals are always defined
                    Symbol::NonTerminal(nt) => defined_lhs_nonterminals.contains(nt),
                })
                || exempt.contains(&i);
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
        crate::debug!(
            3,
            "Removing {} productions with undefined non-terminals.",
            removed_productions.len()
        );
        let all_rhs_nonterminals: BTreeSet<NonTerminal> = removed_productions
            .iter()
            .flat_map(|(_i, prod)| {
                prod.rhs.iter().filter_map(|symbol| match symbol {
                    Symbol::NonTerminal(nt) => Some(nt.clone()),
                    _ => None,
                })
            })
            .collect();
        crate::debug!(
            2,
            "Missing non-terminals ({}) in productions:",
            all_rhs_nonterminals.len()
        );
        for nt in all_rhs_nonterminals.difference(&defined_lhs_nonterminals) {
            crate::debug!(4, "  {}", nt.0);
        }
        crate::debug!(5, "Removed productions:");
        for (i, prod) in removed_productions {
            crate::debug!(5, "  {}", prod);
        }
    }

    current_productions
        .into_iter()
        .map(|(_, prod)| prod)
        .collect()
}

pub fn remove_unreachable_productions(productions: &[Production], start_production_id: usize) -> Vec<Production> {
    if productions.is_empty() {
        return Vec::new();
    }

    let start_lhs = &productions[start_production_id].lhs;
    let mut reachable_nts = BTreeSet::new();
    let mut worklist = VecDeque::new();

    reachable_nts.insert(start_lhs.clone());
    worklist.push_back(start_lhs.clone());

    // Index productions by LHS for faster lookup
    let mut prods_by_lhs: BTreeMap<NonTerminal, Vec<&Production>> = BTreeMap::new();
    for prod in productions {
        prods_by_lhs.entry(prod.lhs.clone()).or_default().push(prod);
    }

    while let Some(nt) = worklist.pop_front() {
        if let Some(prod_list) = prods_by_lhs.get(&nt) {
            for prod in prod_list {
                for sym in &prod.rhs {
                    if let Symbol::NonTerminal(child_nt) = sym {
                        if reachable_nts.insert(child_nt.clone()) {
                            worklist.push_back(child_nt.clone());
                        }
                    }
                }
            }
        }
    }

    let new_productions: Vec<_> = productions
        .iter()
        .filter(|p| reachable_nts.contains(&p.lhs))
        .cloned()
        .collect();

    if new_productions.len() < productions.len() {
        crate::debug!(3, "Removed {} unreachable productions", productions.len() - new_productions.len());
    }

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
                        if interesting_symbols.contains(symbol_in_rhs)
                            || can_derive_interesting.contains(nt_in_rhs)
                        {
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
                            if reachable_set.insert(nt_in_rhs.clone()) {
                                // Check insert result
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
        crate::debug!(
            2,
            "filter_productions_by_reachability: interesting_symbols is empty, returning no productions."
        );
        return Vec::new();
    }

    // --- Pre-computation ---
    // 1. Non-terminals that can derive an interesting symbol.
    let can_derive_set =
        compute_can_derive_interesting(initial_productions, interesting_symbols);
    crate::debug!(
        3,
        "filter_productions_by_reachability: CanDeriveInteresting set: {:?}",
        can_derive_set.iter().map(|nt| &nt.0).collect::<Vec<_>>()
    );

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
                    if interesting_symbols.contains(symbol_in_rhs)
                        || can_derive_set.contains(nt_in_rhs)
                    {
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

pub fn compute_terminal_follow_sets(
    productions: &[Production],
) -> BTreeMap<Terminal, BTreeSet<Terminal>> {
    let nullable_nonterminals = compute_nullable_nonterminals(productions);
    let first_sets =
        compute_first_sets_for_nonterminals(productions, &nullable_nonterminals);
    let nonterminal_follow_sets =
        compute_follow_sets_for_nonterminals(productions, &first_sets, &nullable_nonterminals);

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
                            terminal_follows
                                .entry(t.clone())
                                .or_default()
                                .insert(next_t.clone());
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
pub fn create_unique_name_generator(
    all_nonterminals: &BTreeSet<NonTerminal>,
) -> impl FnMut(&str) -> String {
    let mut existing_names: BTreeSet<String> =
        all_nonterminals.iter().map(|nt| nt.0.clone()).collect();

    move |base_name: &str| {
        // First attempt: base_name + "_rr" (for right-recursion elimination)
        let mut new_name = format!("{base_name}_rr");
        let mut counter = 1;

        // Check for collisions and increment a suffix if needed
        while existing_names.contains(&new_name) {
            counter += 1;
            new_name = format!("{base_name}_rr_{counter}");
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
            if p.rhs.len() < 2 {
                return false;
            }
            // b) The last symbol is `A`.
            if p.rhs.last() != Some(&Symbol::NonTerminal(p.lhs.clone())) {
                return false;
            }
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

        let (recursive_rules, other_rules): (Vec<_>, Vec<_>) =
            prods_for_nt.iter().cloned().partition(|p| {
                if p.rhs.len() < 2 {
                    return false;
                }
                if p.rhs.last() != Some(&Symbol::NonTerminal(p.lhs.clone())) {
                    return false;
                }
                let alpha = &p.rhs[..p.rhs.len() - 1];
                !alpha.contains(&Symbol::NonTerminal(p.lhs.clone()))
            });

        // `recursive_rules` is guaranteed to be non-empty because `lhs` is in `recursive_nts`.
        let new_nt = NonTerminal(new_name_generator(&lhs.0));
        crate::debug!(
            5,
            "Resolving direct right-recursion for '{}' -> '{}'",
            lhs.0,
            new_nt.0
        );

        // Create new rules for the original non-terminal `A`: `A -> A' βⱼ`.
        // The order of these new rules is based on the original order of the `β` rules.
        for non_rec_rule in &other_rules {
            let mut new_rhs = vec![Symbol::NonTerminal(new_nt.clone())];
            new_rhs.extend_from_slice(&non_rec_rule.rhs);
            let new_prod = Production {
                lhs: lhs.clone(),
                rhs: new_rhs,
            };
            crate::debug!(
                5,
                "  Transforming non-recursive rule '{}' -> '{}'",
                non_rec_rule,
                new_prod
            );
            new_productions.push(new_prod);
        }

        // Create rules for the new non-terminal `A'`: `A' -> A' αᵢ` and `A' -> ε`.
        for rec_rule in &recursive_rules {
            let alpha = &rec_rule.rhs[..rec_rule.rhs.len() - 1];
            let mut new_rhs = vec![Symbol::NonTerminal(new_nt.clone())];
            new_rhs.extend_from_slice(alpha);
            let new_prod = Production {
                lhs: new_nt.clone(),
                rhs: new_rhs,
            };
            crate::debug!(
                5,
                "  Transforming recursive rule '{}' -> '{}'",
                rec_rule,
                new_prod
            );
            new_productions.push(new_prod);
        }
        let epsilon_prod = Production {
            lhs: new_nt.clone(),
            rhs: vec![],
        }; // A' -> ε
        crate::debug!(5, "  Adding new epsilon rule: '{}'", epsilon_prod);
        new_productions.push(epsilon_prod);
    }

    // 4. Replace the original productions with the new set.
    productions.clear();
    productions.extend(new_productions);
}

pub fn inline_null_productions(productions: &[Production]) -> Vec<Production> {
    if productions.is_empty() {
        return Vec::new();
    }

    let nullability = compute_nonterminal_nullability(productions);
    let nullable_nonterminals: BTreeSet<_> = nullability
        .iter()
        .filter_map(|(nt, status)| {
            if *status == Nullability::Nullable || *status == Nullability::Null {
                Some(nt.clone())
            } else {
                None
            }
        })
        .collect();
    let start_symbol = &productions[0].lhs;
    let start_symbol_is_nullable = nullable_nonterminals.contains(start_symbol);

    let mut seen = BTreeSet::<Production>::new();
    let mut out = Vec::<Production>::new();

    let start_prod = productions[0].clone();
    seen.insert(start_prod.clone());
    out.push(start_prod);

    for prod in &productions[1..] {
        // Generate all RHS variants by taking the Cartesian product of options for each symbol.
        let rhs_variants: Vec<Vec<Symbol>> = prod.rhs.iter().fold(vec![vec![]], |acc, sym| {
            let sym_options = match sym {
                Symbol::Terminal(_) => vec![Some(sym.clone())],
                Symbol::NonTerminal(nt) => match nullability.get(nt) {
                    Some(Nullability::Null) => vec![None], // Must be removed
                    Some(Nullability::Nullable) => vec![Some(sym.clone()), None], // Optional
                    _ => vec![Some(sym.clone())], // Must be kept
                },
            };

            acc.into_iter()
                .flat_map(|variant| {
                    sym_options.iter().map(move |opt| {
                        let mut new_variant = variant.clone();
                        if let Some(s) = opt {
                            new_variant.push(s.clone());
                        }
                        new_variant
                    })
                })
                .collect()
        });

        for rhs in rhs_variants {
            let new_prod = Production {
                lhs: prod.lhs.clone(),
                rhs,
            };
            if seen.insert(new_prod.clone()) {
                out.push(new_prod);
            }
        }
    }

    let start_rhs_nts: BTreeSet<_> = productions[0]
        .rhs
        .iter()
        .filter_map(|s| {
            if let Symbol::NonTerminal(nt) = s {
                Some(nt.clone())
            } else {
                None
            }
        })
        .collect();

    out.into_iter()
        .filter(|p| {
            if !p.rhs.is_empty() {
                true
            } else {
                // Epsilon production. Keep if its LHS is in start_rhs_nts,
                // or if it's an epsilon production for a nullable start symbol.
                start_rhs_nts.contains(&p.lhs)
                    || (p.lhs == *start_symbol && start_symbol_is_nullable)
            }
        })
        .collect()
}

pub fn inline_unit_productions(productions: &[Production]) -> Vec<Production> {
    todo!()
}

/// Rewrites productions by inserting dummy terminals before their grouped original terminals.
pub fn rewrite_productions_with_dummies(
    original_productions: &[Production],
    dummy_map: &BTreeMap<String, BTreeSet<Terminal>>,
) -> (Vec<Production>, BTreeSet<Terminal>) {
    if dummy_map.is_empty() {
        return (original_productions.to_vec(), BTreeSet::new());
    }

    let mut original_to_dummy: BTreeMap<Terminal, String> = BTreeMap::new();
    for (dummy_name, originals) in dummy_map {
        for original_terminal in originals {
            original_to_dummy.insert(original_terminal.clone(), dummy_name.clone());
        }
    }

    let mut new_productions = Vec::new();
    let mut new_dummy_terminals = BTreeSet::new();

    for prod in original_productions {
        let mut new_rhs = Vec::new();
        for symbol in &prod.rhs {
            if let Symbol::Terminal(t) = symbol {
                if let Some(dummy_name) = original_to_dummy.get(t) {
                    let dummy_terminal = Terminal::RegexName(dummy_name.clone());
                    new_rhs.push(Symbol::Terminal(dummy_terminal.clone()));
                    new_dummy_terminals.insert(dummy_terminal);
                }
            }
            new_rhs.push(symbol.clone());
        }
        new_productions.push(Production {
            lhs: prod.lhs.clone(),
            rhs: new_rhs,
        });
    }

    (new_productions, new_dummy_terminals)
}

fn get_expr_for_terminal(
    t: &Terminal,
    literal_to_group_id: &BiBTreeMap<Vec<u8>, usize>,
    regex_name_to_group_id: &BiBTreeMap<String, usize>,
    group_id_to_expr: &BTreeMap<usize, Expr>,
) -> Option<Expr> {
    match t {
        Terminal::Literal(bytes) => {
            // Even if it's a literal, we prefer the group_id_to_expr version if available,
            // as it might be optimized or shared. But Expr::U8Seq is safe fallback.
            if let Some(gid) = literal_to_group_id.get_by_left(bytes) {
                group_id_to_expr.get(gid).cloned()
            } else {
                Some(Expr::U8Seq(bytes.clone()))
            }
        }
        Terminal::RegexName(name) => {
            let gid = regex_name_to_group_id.get_by_left(name)?;
            group_id_to_expr.get(gid).cloned()
        }
    }
}

fn create_new_terminal(
    expr: Expr,
    base_name: &str,
    regex_name_to_group_id: &mut BiBTreeMap<String, usize>,
    group_id_to_expr: &mut BTreeMap<usize, Expr>,
) -> Terminal {
    let mut counter = 0;
    let name = loop {
        let n = format!("__OPT_{}_{}", base_name, counter);
        if !regex_name_to_group_id.contains_left(&n) {
            break n;
        }
        counter += 1;
    };

    // Find next free group id
    let new_gid = group_id_to_expr.keys().max().map_or(0, |k| k + 1);
    
    group_id_to_expr.insert(new_gid, expr);
    regex_name_to_group_id.insert(name.clone(), new_gid);
    
    Terminal::RegexName(name)
}

fn is_nullable(expr: &Expr) -> bool {
    match expr {
        Expr::Epsilon => true,
        Expr::U8Seq(s) => s.is_empty(),
        Expr::U8Class(_) => false,
        Expr::Choice(opts) => opts.iter().any(is_nullable),
        Expr::Seq(seq) => seq.iter().all(is_nullable),
        Expr::Quantifier(e, q) => match q {
            QuantifierType::ZeroOrMore | QuantifierType::ZeroOrOne => true,
            QuantifierType::OneOrMore => is_nullable(e),
        },
        Expr::Shared(e) => is_nullable(e),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ResolvedSymbol {
    Expr(Expr),
    SelfRef,
}

fn efficient_choice(exprs: Vec<Expr>) -> Expr {
    let mut flat = Vec::with_capacity(exprs.len());
    for e in exprs {
        if let Expr::Choice(subs) = e {
            flat.extend(subs);
        } else {
            flat.push(e);
        }
    }
    if flat.len() == 1 {
        flat.pop().unwrap()
    } else {
        Expr::Choice(flat)
    }
}

fn efficient_seq(exprs: Vec<Expr>) -> Expr {
    let mut flat = Vec::with_capacity(exprs.len());
    for e in exprs {
        if let Expr::Seq(subs) = e {
            flat.extend(subs);
        } else {
            flat.push(e);
        }
    }
    if flat.len() == 1 {
        flat.pop().unwrap()
    } else {
        Expr::Seq(flat)
    }
}

const MAX_REGEX_COMPLEXITY: usize = 50_000_000;

fn get_expr_complexity(expr: &Expr) -> usize {
    match expr {
        Expr::U8Seq(_) => 1,
        Expr::U8Class(_) => 1,
        Expr::Shared(inner) => get_expr_complexity(inner),
        Expr::Quantifier(inner, _) => get_expr_complexity(inner) + 1,
        Expr::Choice(exprs) => exprs.iter().map(get_expr_complexity).sum::<usize>() + 1,
        Expr::Seq(exprs) => exprs.iter().map(get_expr_complexity).sum::<usize>() + 1,
        Expr::Epsilon => 0,
    }
}

fn resolve_production_rhs(
    rhs: &[Symbol],
    nt: &NonTerminal,
    literal_to_group_id: &BiBTreeMap<Vec<u8>, usize>,
    regex_name_to_group_id: &BiBTreeMap<String, usize>,
    group_id_to_expr: &BTreeMap<usize, Expr>,
    nts_to_replace: &BTreeMap<NonTerminal, (Terminal, bool)>,
    ignore_expr: &Option<Expr>,
) -> Option<Vec<ResolvedSymbol>> {
    let mut seq = Vec::new();
    for (i, sym) in rhs.iter().enumerate() {
        if i > 0 {
            if let Some(ref ie) = ignore_expr {
                seq.push(ResolvedSymbol::Expr(Expr::Quantifier(
                    Box::new(ie.clone()),
                    QuantifierType::ZeroOrMore,
                )));
            }
        }

        match sym {
            Symbol::Terminal(t) => {
                let e = get_expr_for_terminal(
                    t,
                    literal_to_group_id,
                    regex_name_to_group_id,
                    group_id_to_expr,
                )?;
                seq.push(ResolvedSymbol::Expr(e));
            }
            Symbol::NonTerminal(n) => {
                if n == nt {
                    seq.push(ResolvedSymbol::SelfRef);
                } else if let Some((t, nullable)) = nts_to_replace.get(n) {
                    let e = get_expr_for_terminal(
                        t,
                        literal_to_group_id,
                        regex_name_to_group_id,
                        group_id_to_expr,
                    )?;
                    if *nullable {
                        seq.push(ResolvedSymbol::Expr(Expr::Choice(vec![e, Expr::Epsilon])));
                    } else {
                        seq.push(ResolvedSymbol::Expr(e));
                    }
                } else {
                    // Depends on unconverted NT
                    return None;
                }
            }
        }
    }
    if seq.is_empty() {
        seq.push(ResolvedSymbol::Expr(Expr::Epsilon));
    }
    Some(seq)
}

fn get_non_null_part(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::Epsilon => None,
        Expr::U8Seq(s) => if s.is_empty() { None } else { Some(expr.clone()) },
        Expr::U8Class(_) => Some(expr.clone()),
        Expr::Shared(inner) => get_non_null_part(inner),
        Expr::Quantifier(inner, q) => match q {
            QuantifierType::OneOrMore => Some(expr.clone()),
            QuantifierType::ZeroOrMore | QuantifierType::ZeroOrOne => {
                // Non-null part of A* or A? is A+ or A (simplification)
                // Actually A* -> non-null is A+. A? -> non-null is A.
                // Assuming inner is non-nullable for A+. If A is nullable, A+ is nullable.
                let inner_non_null = get_non_null_part(inner)?;
                match q {
                    QuantifierType::ZeroOrMore => Some(Expr::Quantifier(Box::new(inner_non_null), QuantifierType::OneOrMore)),
                    QuantifierType::ZeroOrOne => Some(inner_non_null),
                    _ => unreachable!(),
                }
            }
        },
        Expr::Choice(opts) => {
            let non_null_opts: Vec<Expr> = opts.iter().filter_map(get_non_null_part).collect();
            if non_null_opts.is_empty() { None } else { Some(efficient_choice(non_null_opts)) }
        },
        Expr::Seq(seq) => {
             if is_nullable(expr) { None } else { Some(expr.clone()) }
        }
    }
}

fn check_is_noop(prods: &[Production], term: &Terminal, nullable: bool) -> bool {
    if nullable {
        if prods.len() != 2 { return false; }
        let has_term = prods.iter().any(|p| p.rhs.len() == 1 && p.rhs[0] == Symbol::Terminal(term.clone()));
        let has_eps = prods.iter().any(|p| p.rhs.is_empty());
        has_term && has_eps
    } else {
        if prods.len() != 1 { return false; }
        prods[0].rhs.len() == 1 && prods[0].rhs[0] == Symbol::Terminal(term.clone())
    }
}

pub fn optimize_grammar(
    productions: &mut Vec<Production>,
    regex_name_to_group_id: &mut BiBTreeMap<String, usize>,
    literal_to_group_id: &mut BiBTreeMap<Vec<u8>, usize>,
    group_id_to_expr: &mut BTreeMap<usize, Expr>,
    ignore_terminal_id: Option<crate::types::TerminalID>,
    start_symbol: &NonTerminal,
) {
    let mut changed = true;
    while changed {
        changed = false;
        changed |= convert_regular_nts_to_terminals(
            productions,
            regex_name_to_group_id,
            literal_to_group_id,
            group_id_to_expr,
            start_symbol,
            ignore_terminal_id,
        );
        changed |= merge_adjacent_terminals(
            productions,
            regex_name_to_group_id,
            literal_to_group_id,
            group_id_to_expr,
            ignore_terminal_id,
        );
    }
}

fn convert_regular_nts_to_terminals(
    productions: &mut Vec<Production>,
    regex_name_to_group_id: &mut BiBTreeMap<String, usize>,
    literal_to_group_id: &mut BiBTreeMap<Vec<u8>, usize>,
    group_id_to_expr: &mut BTreeMap<usize, Expr>,
    start_symbol: &NonTerminal,
    ignore_terminal_id: Option<crate::types::TerminalID>,
) -> bool {
    let mut prods_by_lhs: BTreeMap<NonTerminal, Vec<Production>> = BTreeMap::new();
    for p in productions.iter() {
        prods_by_lhs.entry(p.lhs.clone()).or_default().push(p.clone());
    }

    let mut nts_to_replace: BTreeMap<NonTerminal, (Terminal, bool)> = BTreeMap::new();

    let ignore_expr = if let Some(tid) = ignore_terminal_id {
        group_id_to_expr.get(&tid.0).cloned()
    } else {
        None
    };

    // We perform a fixed-point iteration here.
    // This allows chains like A->B, B->C, C->Terminal to be fully resolved in one go.
    let mut pending_nts: Vec<NonTerminal> = prods_by_lhs.keys().cloned().collect();
    let mut loop_changed = true;
    let mut any_conversion_happened = false;

    // Reverse map for structural sharing of regexes
    let mut expr_to_group_id: HashMap<Expr, usize> = group_id_to_expr.iter().map(|(k, v)| (v.clone(), *k)).collect();

    // Perform topological sort on pending_nts to process dependencies first.
    // This reduces the number of passes from O(N) to O(1) for deep dependency chains.
    {
        let mut adj: BTreeMap<&NonTerminal, BTreeSet<&NonTerminal>> = BTreeMap::new();
        let pending_set: BTreeSet<&NonTerminal> = pending_nts.iter().collect();

        for nt in &pending_nts {
            let deps = adj.entry(nt).or_default();
            if let Some(prods) = prods_by_lhs.get(nt) {
                for prod in prods {
                    for sym in &prod.rhs {
                        if let Symbol::NonTerminal(dep_nt) = sym {
                            if dep_nt != nt && pending_set.contains(dep_nt) {
                                deps.insert(dep_nt);
                            }
                        }
                    }
                }
            }
        }

        let mut visited = BTreeSet::new();
        let mut sorted = Vec::with_capacity(pending_nts.len());
        for nt in &pending_nts {
             if !visited.contains(nt) {
                 topo_visit(nt, &adj, &mut visited, &mut sorted);
             }
        }
        pending_nts = sorted;
    }

    while loop_changed {
        loop_changed = false;
        let mut next_pending = Vec::with_capacity(pending_nts.len());

        for nt in pending_nts {
            let prods = &prods_by_lhs[&nt];

            // Prevent infinite loop: if start symbol is already a single terminal, stop.
            if nt == *start_symbol && prods.len() == 1 && prods[0].rhs.len() == 1 {
                if let Symbol::Terminal(_) = &prods[0].rhs[0] {
                    continue;
                }
            }

            // Analyze productions
            let mut base_exprs: Vec<Expr> = Vec::new();
            let mut left_rec_exprs: Vec<Expr> = Vec::new();
            let mut right_rec_exprs: Vec<Expr> = Vec::new();
            let mut failed = false;

            for p in prods {
                if let Some(seq) = resolve_production_rhs(
                    &p.rhs,
                    &nt,
                    literal_to_group_id,
                    regex_name_to_group_id,
                    group_id_to_expr,
                    &nts_to_replace,
                    &ignore_expr,
                ) {
                    let self_count = seq.iter().filter(|s| matches!(s, ResolvedSymbol::SelfRef)).count();
                    if self_count > 1 {
                        // println!("NT {} failed: multiple self refs", nt.0);
                        failed = true; // Multiple self-refs (e.g. center embedding or A -> A A) hard to convert simply
                        break;
                    }

                    if self_count == 0 {
                        // Base case
                        let exprs: Vec<Expr> = seq.into_iter().map(|s| match s { ResolvedSymbol::Expr(e) => e, _ => unreachable!() }).collect();
                        base_exprs.push(efficient_seq(exprs));
                    } else {
                        // Recursive case
                        // Check position
                        if let ResolvedSymbol::SelfRef = seq[0] {
                             // Left recursive: [Self, Rest...]
                             let exprs: Vec<Expr> = seq.into_iter().skip(1).map(|s| match s { ResolvedSymbol::Expr(e) => e, _ => unreachable!() }).collect();
                             left_rec_exprs.push(efficient_seq(exprs));
                        } else if let ResolvedSymbol::SelfRef = seq[seq.len()-1] {
                             // Right recursive: [Prefix..., Self]
                             let exprs: Vec<Expr> = seq.clone().into_iter().take(seq.len()-1).map(|s| match s { ResolvedSymbol::Expr(e) => e, _ => unreachable!() }).collect();
                             right_rec_exprs.push(efficient_seq(exprs));
                        } else {
                            failed = true; // Center embedding
                            // println!("NT {} failed: center embedding", nt.0);
                            break;
                        }
                    }
                } else {
                    failed = true; // Dependency not yet resolved
                    break;
                }
            }

            if !failed {
                // Valid structure found. Ensure recursion type consistency.
                if !left_rec_exprs.is_empty() && !right_rec_exprs.is_empty() {
                     // Mixed recursion is complex (e.g. Palindromes), skip.
                     next_pending.push(nt);
                     continue;
                }

                let base_choice = efficient_choice(base_exprs);
                let final_expr = if !left_rec_exprs.is_empty() {
                    let loop_choice = efficient_choice(left_rec_exprs);
                    Expr::Seq(vec![base_choice, Expr::Quantifier(Box::new(loop_choice), QuantifierType::ZeroOrMore)])
                } else if !right_rec_exprs.is_empty() {
                    let loop_choice = efficient_choice(right_rec_exprs);
                    Expr::Seq(vec![Expr::Quantifier(Box::new(loop_choice), QuantifierType::ZeroOrMore), base_choice])
                } else {
                    base_choice
                };

                let mut is_nt_nullable = false;
                let mut expr_for_terminal = final_expr.clone();

                if is_nullable(&final_expr) {
                    if let Some(non_null) = get_non_null_part(&final_expr) {
                         expr_for_terminal = non_null;
                         is_nt_nullable = true;
                    } else {
                        // Purely nullable (e.g. empty string or epsilon), cannot convert to terminal
                        // println!("NT {} failed: purely nullable", nt.0);
                        continue;
                    }
                }
                
                if get_expr_complexity(&expr_for_terminal) > MAX_REGEX_COMPLEXITY {
                    // println!("NT {} failed: complexity limit", nt.0);
                    continue;
                }

                // Deduplicate: Check if this expression already has a terminal
                let (new_term, created_new) = if let Some(&gid) = expr_to_group_id.get(&expr_for_terminal) {
                    // Reuse existing terminal
                    if let Some(term_name) = regex_name_to_group_id.get_by_right(&gid) {
                         (Terminal::RegexName(term_name.clone()), false)
                    } else if let Some(bytes) = literal_to_group_id.get_by_right(&gid) {
                         (Terminal::Literal(bytes.clone()), false)
                    } else {
                        panic!("Group ID {} found in expr map but missing from both regex and literal maps", gid);
                    }
                } else {
                    // Create new terminal
                    let t = create_new_terminal(expr_for_terminal.clone(), &nt.0, regex_name_to_group_id, group_id_to_expr);
                    // Update reverse map
                    if let Terminal::RegexName(ref name) = t {
                        if let Some(&gid) = regex_name_to_group_id.get_by_left(name) {
                             expr_to_group_id.insert(expr_for_terminal, gid);
                        }
                    }
                    (t, true)
                };

                // If we reused a terminal, check if this replacement is actually a no-op (grammar already matches)
                let is_noop = !created_new && check_is_noop(prods, &new_term, is_nt_nullable);

                if !is_noop {
                    nts_to_replace.insert(nt.clone(), (new_term, is_nt_nullable));
                    loop_changed = true;
                    any_conversion_happened = true;
                }
                // If no-op, we don't add to nts_to_replace, effectively skipping it for this pass.
            } else {
                next_pending.push(nt);
            }
        }
        pending_nts = next_pending;
    }

    if !nts_to_replace.is_empty() {
        crate::debug!(3, "Converted {} regular non-terminals to terminals", nts_to_replace.len());

        // Remove productions defining replaced NTs
        // If NT is nullable, we must KEEP it but redefine it as NT -> T | epsilon.
        // If NT is not nullable, we remove it and replace usages.
        
        let mut new_prods = Vec::new();
        for p in productions.iter() {
            if let Some((term, nullable)) = nts_to_replace.get(&p.lhs) {
                // This production is one of the old definitions of the replaced NT. Skip it.
                // We will add the new definition later if needed.
            } else {
                // Keep productions for other NTs
                let mut p_clone = p.clone();
                for s in p_clone.rhs.iter_mut() {
                    if let Symbol::NonTerminal(nt) = s {
                        if let Some((t, nullable)) = nts_to_replace.get(nt) {
                            if !*nullable {
                                *s = Symbol::Terminal(t.clone());
                            }
                            // If nullable, we keep the NonTerminal symbol! 
                            // The NonTerminal itself will be redefined below.
                        }
                    }
                }
                new_prods.push(p_clone);
            }
        }

        // If start symbol was replaced (non-nullable case), add its new production to new_prods
        if let Some((term, false)) = nts_to_replace.get(start_symbol) {
            new_prods.push(Production {
                lhs: start_symbol.clone(),
                rhs: vec![Symbol::Terminal(term.clone())],
            });
        }

        // Add new definitions for replaced NTs
        for (nt, (term, nullable)) in &nts_to_replace {
            if *nullable {
                // NT -> T
                new_prods.push(Production {
                    lhs: nt.clone(),
                    rhs: vec![Symbol::Terminal(term.clone())],
                });
                // NT -> epsilon
                new_prods.push(Production {
                    lhs: nt.clone(),
                    rhs: vec![],
                });
            }
        }

        *productions = new_prods;
        return true;
    }

    any_conversion_happened
}

fn topo_visit<'a>(
    nt: &'a NonTerminal,
    adj: &BTreeMap<&'a NonTerminal, BTreeSet<&'a NonTerminal>>,
    visited: &mut BTreeSet<&'a NonTerminal>,
    sorted: &mut Vec<NonTerminal>,
) {
    visited.insert(nt);
    if let Some(deps) = adj.get(nt) {
        for dep in deps {
            if !visited.contains(dep) {
                topo_visit(dep, adj, visited, sorted);
            }
        }
    }
    sorted.push(nt.clone());
}

fn merge_adjacent_terminals(
    productions: &mut Vec<Production>,
    regex_name_to_group_id: &mut BiBTreeMap<String, usize>,
    literal_to_group_id: &mut BiBTreeMap<Vec<u8>, usize>,
    group_id_to_expr: &mut BTreeMap<usize, Expr>,
    ignore_terminal_id: Option<crate::types::TerminalID>,
) -> bool {
    let mut changed = false;
    
    // Resolve ignore expr
    let ignore_expr = if let Some(tid) = ignore_terminal_id {
        // Find gid for tid. tid.0 is the group_id (usually) or we need reverse lookup.
        // TerminalID is just usize.
        group_id_to_expr.get(&tid.0).cloned()
    } else {
        None
    };

    let ignore_seq = if let Some(ie) = ignore_expr {
        Some(Expr::Quantifier(Box::new(ie), QuantifierType::ZeroOrMore))
    } else {
        None
    };

    for p in productions.iter_mut() {
        if p.rhs.len() < 2 {
            continue;
        }

        let mut new_rhs = Vec::new();
        let mut i = 0;
        while i < p.rhs.len() {
            if i + 1 < p.rhs.len() {
                if let (Symbol::Terminal(t1), Symbol::Terminal(t2)) = (&p.rhs[i], &p.rhs[i+1]) {
                    // Merge t1 and t2
                    let e1 = get_expr_for_terminal(t1, literal_to_group_id, regex_name_to_group_id, group_id_to_expr);
                    let e2 = get_expr_for_terminal(t2, literal_to_group_id, regex_name_to_group_id, group_id_to_expr);

                    if let (Some(e1), Some(e2)) = (e1, e2) {
                        let combined_expr = if let Some(ref ie) = ignore_seq {
                            Expr::Seq(vec![e1, ie.clone(), e2])
                        } else {
                            Expr::Seq(vec![e1, e2])
                        };

                        if get_expr_complexity(&combined_expr) > MAX_REGEX_COMPLEXITY {
                            break;
                        }

                        let new_term = create_new_terminal(
                            combined_expr, 
                            "merged", 
                            regex_name_to_group_id, 
                            group_id_to_expr
                        );

                        new_rhs.push(Symbol::Terminal(new_term));
                        i += 2;
                        changed = true;
                        continue;
                    }
                }
            }
            new_rhs.push(p.rhs[i].clone());
            i += 1;
        }
        p.rhs = new_rhs;
    }
    
    changed
}
