use crate::glr::automaton::{
    compute_first_sets_for_nonterminals, compute_follow_sets_for_nonterminals,
    compute_nonterminal_nullability, compute_nullable_nonterminals, Nullability,
};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Checks for non-terminals used in rule RHS but never defined in LHS.
pub fn check_for_undefined_non_terminals(productions: &[Production]) -> Vec<String> {
    let mut lhs_nonterms = BTreeSet::new();
    let mut rhs_nonterms = BTreeSet::new();

    for prod in productions {
        lhs_nonterms.insert(prod.lhs.clone());
        rhs_nonterms.extend(prod.rhs.iter().filter_map(|s| match s {
            Symbol::NonTerminal(nt) => Some(nt.clone()),
            _ => None,
        }));
    }

    let missing: Vec<_> = rhs_nonterms
        .difference(&lhs_nonterms)
        .map(|nt| nt.0.clone())
        .collect();

    if missing.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "Non-terminal(s) used in rule RHS but never defined in LHS: {:?}",
            missing
        )]
    }
}

/// Helper for check_for_length_1_recursion. Detects all elementary cycles in a graph.
fn detect_all_cycles_recursive(
    nt: &NonTerminal,
    graph: &BTreeMap<NonTerminal, BTreeSet<NonTerminal>>,
    visiting: &mut BTreeSet<NonTerminal>,
    visited: &mut BTreeSet<NonTerminal>,
    path: &mut Vec<NonTerminal>,
    cycles: &mut BTreeSet<Vec<NonTerminal>>,
) {
    visiting.insert(nt.clone());
    path.push(nt.clone());

    if let Some(neighbors) = graph.get(nt) {
        for neighbor in neighbors {
            if visiting.contains(neighbor) {
                if let Some(start) = path.iter().position(|n| n == neighbor) {
                    let mut cycle: Vec<_> = path[start..].to_vec();
                    if !cycle.is_empty() {
                        let min_pos = cycle
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, n)| *n)
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        cycle.rotate_left(min_pos);
                    }
                    cycles.insert(cycle);
                }
                continue;
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

fn find_rightmost_position(
    rhs: &[Symbol],
    target: &NonTerminal,
    nullable: &BTreeSet<NonTerminal>,
) -> Option<usize> {
    for (idx, symbol) in rhs.iter().enumerate().rev() {
        if let Symbol::NonTerminal(nt) = symbol {
            if nt == target {
                let suffix_nullable = rhs[idx + 1..].iter().all(|s| match s {
                    Symbol::NonTerminal(snt) => nullable.contains(snt),
                    Symbol::Terminal(_) => false,
                });
                if suffix_nullable {
                    return Some(idx);
                }
            }

            if !nullable.contains(nt) {
                break;
            }
        } else {
            break;
        }
    }
    None
}

fn break_right_recursion_with_ordered_inlining(
    productions: &mut Vec<Production>,
    right_recursive_nts: &BTreeSet<NonTerminal>,
) -> bool {
    let mut nullable = compute_nullable_nonterminals(productions);
    let mut ordered_nts = Vec::new();
    let mut seen = BTreeSet::new();
    for prod in productions.iter() {
        if right_recursive_nts.contains(&prod.lhs) && seen.insert(prod.lhs.clone()) {
            ordered_nts.push(prod.lhs.clone());
        }
    }

    if ordered_nts.len() < 2 {
        return false;
    }

    let mut changed = false;

    for idx in (0..ordered_nts.len()).rev() {
        let ai = &ordered_nts[idx];
        let mut ai_prods: Vec<_> = productions
            .iter()
            .filter(|p| &p.lhs == ai)
            .cloned()
            .collect();
        let mut replaced_ai = false;

        for aj_idx in idx + 1..ordered_nts.len() {
            let aj = &ordered_nts[aj_idx];
            if ai == aj {
                continue;
            }

            let aj_prods: Vec<_> = productions
                .iter()
                .filter(|p| &p.lhs == aj)
                .cloned()
                .collect();
            if aj_prods.is_empty() {
                continue;
            }

            let mut next_ai_prods = Vec::with_capacity(ai_prods.len());
            let mut replaced_pair = false;

            for prod in ai_prods.into_iter() {
                if let Some(pos) = find_rightmost_position(&prod.rhs, aj, &nullable) {
                    replaced_pair = true;
                    let prefix: Vec<_> = prod.rhs[..pos].to_vec();
                    let suffix: Vec<_> = prod.rhs[pos + 1..].to_vec();

                    for b_prod in &aj_prods {
                        let mut rhs = prefix.clone();
                        rhs.extend(b_prod.rhs.iter().cloned());
                        rhs.extend(suffix.iter().cloned());
                        next_ai_prods.push(Production {
                            lhs: ai.clone(),
                            rhs,
                        });
                    }
                } else {
                    next_ai_prods.push(prod);
                }
            }

            if replaced_pair {
                crate::debug!(
                    5,
                    "Ordered inlining fallback: expanded {} inside {}",
                    aj.0,
                    ai.0
                );
                replaced_ai = true;
                ai_prods = next_ai_prods;
            } else {
                ai_prods = next_ai_prods;
            }
        }

        if replaced_ai {
            changed = true;
            productions.retain(|p| &p.lhs != ai);
            productions.extend(ai_prods.clone());
            nullable = compute_nullable_nonterminals(productions);
        }
    }

    changed
}
/// Checks for length-1 recursion (e.g., A ::= A, A ::= B; B ::= A), considering nullable prefixes.
pub fn check_for_length_1_recursion(productions: &[Production]) -> Vec<String> {
    let nullable_nonterminals = compute_nullable_nonterminals(productions);
    let all_nonterminals: BTreeSet<NonTerminal> = productions
        .iter()
        .flat_map(|p| {
            std::iter::once(p.lhs.clone()).chain(p.rhs.iter().filter_map(|s| match s {
                Symbol::NonTerminal(nt) => Some(nt.clone()),
                _ => None,
            }))
        })
        .collect();

    // Build a graph where there is an edge A -> B if a rule A ::= Nullable* B Nullable* exists.
    let mut unit_graph: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
    for nt in &all_nonterminals {
        unit_graph.entry(nt.clone()).or_default();
    }

    for prod in productions {
        let non_nullable_symbols: Vec<&Symbol> = prod
            .rhs
            .iter()
            .filter(|symbol| match symbol {
                Symbol::Terminal(_) => true,
                Symbol::NonTerminal(nt) => !nullable_nonterminals.contains(nt),
            })
            .collect();

        if non_nullable_symbols.len() == 1 {
            if let Symbol::NonTerminal(target_nt) = non_nullable_symbols[0] {
                unit_graph
                    .entry(prod.lhs.clone())
                    .or_default()
                    .insert(target_nt.clone());
            }
        }
    }

    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let mut cycles = BTreeSet::new();
    let mut path = Vec::new();

    for nt in &all_nonterminals {
        if !visited.contains(nt) {
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

    cycles
        .into_iter()
        .map(|cycle| {
            let mut names: Vec<_> = cycle.iter().map(|n| n.0.as_str()).collect();
            names.push(cycle[0].0.as_str());
            let recursion_type = if cycle.len() == 1 {
                "Direct"
            } else {
                "Indirect"
            };
            format!(
                "{recursion_type} length-1 recursion cycle detected: {}",
                names.join(" -> ")
            )
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

        for (i, symbol) in rhs.iter().enumerate() {
            if let Symbol::NonTerminal(nt) = symbol {
                if nt == lhs {
                    let prefix = &rhs[0..i];
                    if !prefix.is_empty() {
                        let prefix_is_nullable = prefix.iter().all(|sym| match sym {
                            Symbol::NonTerminal(prefix_nt) => {
                                nullable_nonterminals.contains(prefix_nt)
                            }
                            Symbol::Terminal(_) => false,
                        });
                        if prefix_is_nullable {
                            errors.push(format!(
                                "Left-nullable left recursion detected in rule '{} ::= {:?}'. \
                                 The prefix '{:?}' before the recursive non-terminal '{}' is nullable.",
                                lhs.0, rhs, prefix, lhs.0
                            ));
                        }
                    }
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

            let rhs_is_productive = prod.rhs.iter().all(|symbol| match symbol {
                Symbol::Terminal(_) => true,
                Symbol::NonTerminal(nt) => productive_nts.contains(nt),
            });

            if rhs_is_productive && productive_nts.insert(prod.lhs.clone()) {
                changed = true;
            }
        }
    }
    productive_nts
}

/// Checks for non-terminals that cannot derive any terminal string.
pub fn check_for_non_productive_non_terminals(productions: &[Production]) -> Vec<String> {
    let all_nonterminals: BTreeSet<NonTerminal> =
        productions.iter().map(|p| p.lhs.clone()).collect();
    let productive_nts = compute_productive_non_terminals(productions);

    let non_productive: Vec<_> = all_nonterminals
        .difference(&productive_nts)
        .map(|nt| nt.0.clone())
        .collect();

    if non_productive.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "Non-terminal(s) are non-productive (cannot derive a terminal string): {:?}",
            non_productive
        )]
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
pub fn remove_productions_with_undefined_nonterminals(
    initial_productions: &[Production],
    exempt: &[usize],
) -> Vec<Production> {
    let mut current: Vec<(usize, Production)> =
        initial_productions.iter().cloned().enumerate().collect();

    loop {
        let defined_lhs: BTreeSet<NonTerminal> =
            current.iter().map(|(_, prod)| prod.lhs.clone()).collect();

        let mut removed = Vec::new();
        let mut kept = Vec::new();

        for (i, prod) in current {
            let keep = exempt.contains(&i)
                || prod.rhs.iter().all(|symbol| match symbol {
                    Symbol::Terminal(_) => true,
                    Symbol::NonTerminal(nt) => defined_lhs.contains(nt),
                });
            if keep {
                kept.push((i, prod));
            } else {
                removed.push((i, prod));
            }
        }

        if removed.is_empty() {
            current = kept;
            break;
        }

        crate::debug!(
            5,
            "Removing {} productions with undefined non-terminals.",
            removed.len()
        );

        let all_rhs_nonterminals: BTreeSet<NonTerminal> = removed
            .iter()
            .flat_map(|(_, prod)| {
                prod.rhs.iter().filter_map(|symbol| match symbol {
                    Symbol::NonTerminal(nt) => Some(nt.clone()),
                    _ => None,
                })
            })
            .collect();

        crate::debug!(
            4,
            "Missing non-terminals ({}) in productions:",
            all_rhs_nonterminals.len()
        );
        for nt in all_rhs_nonterminals.difference(&defined_lhs) {
            crate::debug!(6, "  {}", nt.0);
        }

        crate::debug!(7, "Removed productions:");
        for (_, prod) in &removed {
            crate::debug!(7, "  {}", prod);
        }

        current = kept;
    }

    current.into_iter().map(|(_, prod)| prod).collect()
}

// TODO: This function is known to be incomplete; kept here for compatibility.
pub fn drop_dead(productions: &[Production]) -> Vec<Production> {
    // todo: this function is broken
    let mut nt_reachables: BTreeMap<&NonTerminal, BTreeSet<&NonTerminal>> = BTreeMap::new();

    for prod in productions {
        let rhs_nonterms: BTreeSet<_> = prod
            .rhs
            .iter()
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
                    nt_reachables
                        .get_mut(nt)
                        .unwrap()
                        .extend(reachable_reachables);
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

    let new_productions: Vec<_> = productions
        .iter()
        .filter(|prod| reachable_from_start.contains(&prod.lhs) || *prod == start_prod)
        .cloned()
        .collect();

    crate::debug!(
        4,
        "Dropped {} productions",
        productions.len() - new_productions.len()
    );

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

            let rhs_can_lead = production.rhs.iter().any(|symbol| match symbol {
                Symbol::Terminal(_) => interesting_symbols.contains(symbol),
                Symbol::NonTerminal(nt) => {
                    interesting_symbols.contains(symbol) || can_derive_interesting.contains(nt)
                }
            });

            if rhs_can_lead && can_derive_interesting.insert(production.lhs.clone()) {
                changed = true;
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
    let seed_interesting_nts: BTreeSet<NonTerminal> = interesting_symbols
        .iter()
        .filter_map(|s| match s {
            Symbol::NonTerminal(nt) => Some(nt.clone()),
            _ => None,
        })
        .collect();

    if seed_interesting_nts.is_empty() {
        return BTreeSet::new();
    }

    let mut reachable_set = seed_interesting_nts.clone();
    let mut worklist: VecDeque<NonTerminal> = seed_interesting_nts.into_iter().collect();

    while let Some(nt_lhs_from_worklist) = worklist.pop_front() {
        for production in productions.iter().filter(|p| p.lhs == nt_lhs_from_worklist) {
            for symbol_in_rhs in &production.rhs {
                if let Symbol::NonTerminal(nt_in_rhs) = symbol_in_rhs {
                    if reachable_set.insert(nt_in_rhs.clone()) {
                        worklist.push_back(nt_in_rhs.clone());
                    }
                }
            }
        }
    }
    reachable_set
}

/// Filters productions to keep only those relevant to deriving specified "interesting" symbols.
pub fn filter_productions_by_reachability(
    initial_productions: &[Production],
    interesting_symbols: &BTreeSet<Symbol>,
) -> Vec<Production> {
    if interesting_symbols.is_empty() {
        crate::debug!(4, "filter_productions_by_reachability: interesting_symbols is empty, returning no productions.");
        return Vec::new();
    }

    let can_derive_set = compute_can_derive_interesting(initial_productions, interesting_symbols);
    crate::debug!(
        5,
        "filter_productions_by_reachability: CanDeriveInteresting set: {:?}",
        can_derive_set.iter().map(|nt| &nt.0).collect::<Vec<_>>()
    );

    let mut kept_productions = Vec::new();
    for production in initial_productions {
        let lhs_can_derive_interesting = can_derive_set.contains(&production.lhs);

        let rhs_can_derive_interesting_for_this_rule =
            production
                .rhs
                .iter()
                .any(|symbol_in_rhs| match symbol_in_rhs {
                    Symbol::Terminal(_) => interesting_symbols.contains(symbol_in_rhs),
                    Symbol::NonTerminal(nt_in_rhs) => {
                        interesting_symbols.contains(symbol_in_rhs)
                            || can_derive_set.contains(nt_in_rhs)
                    }
                });

        if lhs_can_derive_interesting && rhs_can_derive_interesting_for_this_rule {
            kept_productions.push(production.clone());
        } else {
            crate::debug!(6, "Filtering out production: {} (LHS can derive interesting: {}, RHS of this rule can derive interesting: {})", production, lhs_can_derive_interesting, rhs_can_derive_interesting_for_this_rule);
        }
    }

    kept_productions
}

pub fn simplify_grammar(initial_productions: &[Production]) -> Vec<Production> {
    todo!()
}

/// Helper function to find the last symbol in a rule's RHS that is not a nullable non-terminal.
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
    let first_sets = compute_first_sets_for_nonterminals(productions, &nullable_nonterminals);
    let nonterminal_follow_sets =
        compute_follow_sets_for_nonterminals(productions, &first_sets, &nullable_nonterminals);

    let mut terminal_follows: BTreeMap<Terminal, BTreeSet<Terminal>> = BTreeMap::new();

    for production in productions {
        let lhs = &production.lhs;
        let rhs = &production.rhs;

        for (i, symbol) in rhs.iter().enumerate() {
            if let Symbol::Terminal(t) = symbol {
                let mut all_following_are_nullable = true;

                for next_symbol in &rhs[i + 1..] {
                    match next_symbol {
                        Symbol::Terminal(next_t) => {
                            terminal_follows
                                .entry(t.clone())
                                .or_default()
                                .insert(next_t.clone());
                            all_following_are_nullable = false;
                            break;
                        }
                        Symbol::NonTerminal(next_nt) => {
                            if let Some(first_set_for_next_nt) = first_sets.get(next_nt) {
                                terminal_follows
                                    .entry(t.clone())
                                    .or_default()
                                    .extend(first_set_for_next_nt.iter().cloned());
                            }
                            if !nullable_nonterminals.contains(next_nt) {
                                all_following_are_nullable = false;
                                break;
                            }
                        }
                    }
                }

                if all_following_are_nullable {
                    if let Some(follow_set_for_lhs) = nonterminal_follow_sets.get(lhs) {
                        terminal_follows
                            .entry(t.clone())
                            .or_default()
                            .extend(follow_set_for_lhs.iter().filter_map(|opt_t| opt_t.clone()));
                    }
                }
            }
        }
    }

    terminal_follows
}

/// Creates a closure that generates unique non-terminal names, suitable for `resolve_right_recursion`.
pub fn create_unique_name_generator(
    all_nonterminals: &BTreeSet<NonTerminal>,
) -> impl FnMut(&str) -> String {
    let mut existing_names: BTreeSet<String> =
        all_nonterminals.iter().map(|nt| nt.0.clone()).collect();

    move |base_name: &str| {
        let mut new_name = format!("{base_name}_rr");
        let mut counter = 1;

        while existing_names.contains(&new_name) {
            counter += 1;
            new_name = format!("{base_name}_rr_{counter}");
        }

        existing_names.insert(new_name.clone());
        new_name
    }
}

pub fn resolve_right_recursion(
    productions: &mut Vec<Production>,
    new_name_generator: &mut impl FnMut(&str) -> String,
) {
    // This function eliminates all right recursion (direct and indirect) by:
    // 1. Finding all NTs that are right-recursive (can derive A →* ... A)
    // 2. Inlining unit productions to expose direct recursion
    // 3. Applying the standard right-to-left recursion transformation

    // Track previous right-recursive sets to detect when we're making no progress
    let mut prev_right_recursive: Option<BTreeSet<NonTerminal>> = None;
    let mut no_progress_count = 0;
    const MAX_NO_PROGRESS: usize = 50;

    loop {
        let nullable = compute_nullable_nonterminals(productions);

        // Helper to find ALL potential rightmost non-terminals (considering nullable suffixes)
        // Returns all NTs that could be the rightmost, depending on which optional elements are present
        let get_rightmost_nts = |rhs: &[Symbol]| -> Vec<NonTerminal> {
            let mut result = Vec::new();
            for symbol in rhs.iter().rev() {
                match symbol {
                    Symbol::NonTerminal(nt) => {
                        // This NT could be the rightmost (when symbols to its right are empty)
                        result.push(nt.clone());
                        if !nullable.contains(nt) {
                            // This NT is non-nullable, so it's always present when reached
                            // No need to look further left
                            break;
                        }
                        // NT is nullable, continue looking left for more possibilities
                    }
                    Symbol::Terminal(_) => {
                        // Hit a non-nullable terminal, stop looking left
                        break;
                    }
                }
            }
            result
        };

        // Build right-reachability graph
        // Add edges for ALL potential rightmost NTs (including nullable ones)
        let mut right_reachable: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
        for prod in productions.iter() {
            for rightmost_nt in get_rightmost_nts(&prod.rhs) {
                right_reachable
                    .entry(prod.lhs.clone())
                    .or_default()
                    .insert(rightmost_nt);
            }
        }

        // Compute transitive closure
        loop {
            let mut changed = false;
            let keys: Vec<_> = right_reachable.keys().cloned().collect();
            for a in &keys {
                let reachable: Vec<_> = right_reachable
                    .get(a)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .collect();
                for b in reachable {
                    if let Some(reachable_from_b) = right_reachable.get(&b).cloned() {
                        let set = right_reachable.entry(a.clone()).or_default();
                        for c in reachable_from_b {
                            if set.insert(c) {
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Find right-recursive non-terminals
        let right_recursive_nts: BTreeSet<_> = right_reachable
            .iter()
            .filter(|(nt, reachable)| reachable.contains(*nt))
            .filter(|(nt, _)| !nt.0.ends_with("_rr") && !nt.0.contains("_rr_"))
            .map(|(nt, _)| nt.clone())
            .collect();

        if right_recursive_nts.is_empty() {
            break;
        }

        // Check if any are directly right-recursive (can be handled by resolve_direct_right_recursion)
        let has_direct = productions.iter().any(|p| {
            let is_direct =
                right_recursive_nts.contains(&p.lhs) && is_simple_direct_right_recursive(p);
            if is_direct {
                crate::debug!(5, "Found direct right recursion for {}", p.lhs.0);
            }
            is_direct
        });

        // Check if we're making progress
        if let Some(ref prev) = prev_right_recursive {
            if &right_recursive_nts == prev && !has_direct {
                no_progress_count += 1;
                if no_progress_count >= MAX_NO_PROGRESS {
                    crate::debug!(5, "No progress in eliminating right recursion after {} iterations, giving up. Remaining: {:?}",
                        MAX_NO_PROGRESS, right_recursive_nts.iter().map(|nt| &nt.0).collect::<Vec<_>>());
                    break;
                }
            } else {
                no_progress_count = 0;
            }
        }
        prev_right_recursive = Some(right_recursive_nts.clone());

        crate::debug!(
            5,
            "Found right-recursive non-terminals: {:?}",
            right_recursive_nts
                .iter()
                .map(|nt| &nt.0)
                .collect::<Vec<_>>()
        );

        if has_direct {
            // Apply the direct right recursion transformation
            resolve_direct_right_recursion(productions, &mut *new_name_generator);
        } else {
            // Check for "hidden right recursion" like S → a S B where B is nullable
            // This is effectively S → a S
            let has_hidden_right_recursion = productions.iter().any(|p| {
                if !right_recursive_nts.contains(&p.lhs) {
                    return false;
                }
                // Check if there's a non-terminal equal to LHS followed only by nullable symbols
                for (i, sym) in p.rhs.iter().enumerate() {
                    if let Symbol::NonTerminal(nt) = sym {
                        if nt == &p.lhs {
                            // Check if all symbols after this are nullable
                            let suffix = &p.rhs[i + 1..];
                            if !suffix.is_empty() && suffix.iter().all(|s| {
                                matches!(s, Symbol::NonTerminal(snt) if nullable.contains(snt))
                            }) {
                                // Also make sure this is not at the very end (that's simple direct)
                                // and that there's something before it
                                if i > 0 {
                                    return true;
                                }
                            }
                        }
                    }
                }
                false
            });

            if has_hidden_right_recursion {
                // Handle hidden right recursion by removing the nullable suffix
                // S → a S B becomes S → a S (effectively)
                // We'll inline the nullable NTs' empty productions
                crate::debug!(5, "Handling hidden right recursion");

                let mut new_prods = Vec::new();
                for prod in productions.iter() {
                    if !right_recursive_nts.contains(&prod.lhs) {
                        new_prods.push(prod.clone());
                        continue;
                    }

                    // Check if this production has hidden right recursion
                    let mut has_hidden = false;
                    let mut self_position = None;
                    for (i, sym) in prod.rhs.iter().enumerate() {
                        if let Symbol::NonTerminal(nt) = sym {
                            if nt == &prod.lhs {
                                let suffix = &prod.rhs[i + 1..];
                                if !suffix.is_empty() && suffix.iter().all(|s| {
                                    matches!(s, Symbol::NonTerminal(snt) if nullable.contains(snt))
                                }) {
                                    if i > 0 {
                                        has_hidden = true;
                                        self_position = Some(i);
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    if has_hidden {
                        let pos = self_position.unwrap();
                        // Create a new production without the nullable suffix
                        // S → a S B becomes S → a S
                        // This transforms hidden right recursion into direct right recursion
                        // which will be handled in the next iteration
                        let new_rhs: Vec<_> = prod.rhs[..=pos].to_vec();
                        new_prods.push(Production {
                            lhs: prod.lhs.clone(),
                            rhs: new_rhs,
                        });
                        // NOTE: We do NOT keep the original. The nullable suffix productions
                        // are handled by inline_null_productions elsewhere.
                    } else {
                        new_prods.push(prod.clone());
                    }
                }

                *productions = new_prods;
            } else {
                // Check for "both-ends self-recursion" like E → E + E
                // This pattern: LHS at position 0 AND at the end, with something in between
                // These need special handling: we extract the middle operators
                let has_both_ends_self_recursion = productions.iter().any(|p| {
                    if !right_recursive_nts.contains(&p.lhs) {
                        return false;
                    }
                    if p.rhs.len() < 3 {
                        return false; // Need at least: E op E
                    }
                    // Check if RHS starts AND ends with the LHS
                    let starts_with_self =
                        matches!(p.rhs.first(), Some(Symbol::NonTerminal(nt)) if nt == &p.lhs);
                    let ends_with_self =
                        matches!(p.rhs.last(), Some(Symbol::NonTerminal(nt)) if nt == &p.lhs);
                    starts_with_self && ends_with_self
                });

                if has_both_ends_self_recursion {
                    // Handle E → E + E type recursion (both left and right recursive)
                    // Transform to: E → E E', E' → + E E' | * E E' | ε
                    crate::debug!(5, "Handling both-ends self-recursion (E → E + E pattern)");

                    let mut new_prods = Vec::new();
                    let mut processed = BTreeSet::new();

                    // Group by LHS
                    let prods_by_lhs: BTreeMap<NonTerminal, Vec<Production>> = {
                        let mut map = BTreeMap::new();
                        for prod in productions.iter() {
                            map.entry(prod.lhs.clone())
                                .or_insert_with(Vec::new)
                                .push(prod.clone());
                        }
                        map
                    };

                    for prod in productions.iter() {
                        if processed.contains(&prod.lhs) {
                            continue;
                        }

                        if !right_recursive_nts.contains(&prod.lhs) {
                            new_prods.push(prod.clone());
                            continue;
                        }

                        // Check if this NT has both-ends self-recursion (E → E + E pattern)
                        let prods_for_nt = prods_by_lhs.get(&prod.lhs).unwrap();
                        let has_both_ends_recursion = prods_for_nt.iter().any(|p| {
                            if p.rhs.len() < 3 {
                                return false;
                            }
                            // Must start AND end with the LHS
                            let starts_with_self = matches!(p.rhs.first(), Some(Symbol::NonTerminal(nt)) if nt == &p.lhs);
                            let ends_with_self = matches!(p.rhs.last(), Some(Symbol::NonTerminal(nt)) if nt == &p.lhs);
                            starts_with_self && ends_with_self
                        });

                        if !has_both_ends_recursion {
                            new_prods.push(prod.clone());
                            continue;
                        }

                        processed.insert(prod.lhs.clone());

                        // Create new NT for the tail recursion
                        let new_nt = NonTerminal(new_name_generator(&prod.lhs.0));
                        crate::debug!(
                            5,
                            "Transforming {} with both-ends self-recursion, creating {}",
                            prod.lhs.0,
                            new_nt.0
                        );

                        // Partition productions into:
                        // - Both-ends recursive (E → E + E): becomes E' → + E E'
                        // - Simple right recursive (E → a E): becomes E' → a E E'
                        // - Non-recursive (E → id): becomes E → id E'

                        for p in prods_for_nt {
                            let starts_with_self = matches!(p.rhs.first(), Some(Symbol::NonTerminal(nt)) if nt == &p.lhs);
                            let ends_with_self = matches!(p.rhs.last(), Some(Symbol::NonTerminal(nt)) if nt == &p.lhs);

                            if starts_with_self && ends_with_self && p.rhs.len() >= 3 {
                                // E → E + E becomes E' → + E E'
                                // Extract the middle part (between first E and last E)
                                let middle: Vec<_> = p.rhs[1..p.rhs.len() - 1].to_vec();

                                // E' → middle E E' (where middle is "+" for "E + E")
                                let mut new_rhs = middle;
                                new_rhs.push(Symbol::NonTerminal(p.lhs.clone()));
                                new_rhs.push(Symbol::NonTerminal(new_nt.clone()));
                                new_prods.push(Production {
                                    lhs: new_nt.clone(),
                                    rhs: new_rhs,
                                });
                            } else {
                                // E → id becomes E → id E'
                                let mut new_rhs = p.rhs.clone();
                                new_rhs.push(Symbol::NonTerminal(new_nt.clone()));
                                new_prods.push(Production {
                                    lhs: p.lhs.clone(),
                                    rhs: new_rhs,
                                });
                            }
                        }

                        // E' → ε
                        new_prods.push(Production {
                            lhs: new_nt.clone(),
                            rhs: vec![],
                        });
                    }

                    *productions = new_prods;
                } else {
                    // No direct recursion found - we have only indirect recursion through unit productions
                    // Inline unit productions involving right-recursive NTs to expose direct recursion

                    // A unit production is A → B where B is a single non-terminal
                    let unit_prods: Vec<_> = productions
                        .iter()
                        .filter(|p| {
                            p.rhs.len() == 1
                            && matches!(p.rhs.first(), Some(Symbol::NonTerminal(nt)) if right_recursive_nts.contains(nt))
                            && right_recursive_nts.contains(&p.lhs)
                        })
                        .cloned()
                        .collect();

                    if unit_prods.is_empty() {
                        // No unit productions to inline - fall back to deterministic ordered inlining
                        crate::debug!(
                            5,
                            "No unit productions found, using ordered inlining fallback"
                        );

                        if !break_right_recursion_with_ordered_inlining(
                            productions,
                            &right_recursive_nts,
                        ) {
                            crate::debug!(
                                5,
                                "Could not break right recursion cycle for: {:?}, giving up",
                                right_recursive_nts
                                    .iter()
                                    .map(|nt| &nt.0)
                                    .collect::<Vec<_>>()
                            );
                            break;
                        }
                    } else {
                        // Inline unit productions
                        crate::debug!(
                            5,
                            "Inlining unit productions: {:?}",
                            unit_prods
                                .iter()
                                .map(|p| format!("{}", p))
                                .collect::<Vec<_>>()
                        );

                        let mut new_prods = Vec::new();
                        let prods_by_lhs: BTreeMap<NonTerminal, Vec<Production>> = {
                            let mut map = BTreeMap::new();
                            for prod in productions.iter() {
                                map.entry(prod.lhs.clone())
                                    .or_insert_with(Vec::new)
                                    .push(prod.clone());
                            }
                            map
                        };

                        for prod in productions.iter() {
                            // Check if this is a unit production A → B where B is right-recursive
                            if prod.rhs.len() == 1 {
                                if let Some(Symbol::NonTerminal(nt)) = prod.rhs.first() {
                                    if right_recursive_nts.contains(nt)
                                        && right_recursive_nts.contains(&prod.lhs)
                                    {
                                        // Drop direct self-recursion A -> A
                                        if nt == &prod.lhs {
                                            continue;
                                        }
                                        // Inline: replace A → B with A → γ for each B → γ
                                        if let Some(b_prods) = prods_by_lhs.get(nt) {
                                            for b_prod in b_prods {
                                                new_prods.push(Production {
                                                    lhs: prod.lhs.clone(),
                                                    rhs: b_prod.rhs.clone(),
                                                });
                                            }
                                        }
                                        continue;
                                    }
                                }
                            }
                            new_prods.push(prod.clone());
                        }

                        *productions = new_prods;
                    }
                }
            }
        }
    }
}

fn is_simple_direct_right_recursive(prod: &Production) -> bool {
    if prod.rhs.len() < 2 {
        return false;
    }
    match prod.rhs.last() {
        Some(Symbol::NonTerminal(nt)) if nt == &prod.lhs => {
            let prefix = &prod.rhs[..prod.rhs.len() - 1];
            !prefix.contains(&Symbol::NonTerminal(prod.lhs.clone()))
        }
        _ => false,
    }
}

pub fn resolve_direct_right_recursion(
    productions: &mut Vec<Production>,
    mut new_name_generator: impl FnMut(&str) -> String,
) {
    // Group productions by LHS to easily access all rules for a given non-terminal.
    let mut prods_by_lhs: BTreeMap<NonTerminal, Vec<Production>> = BTreeMap::new();
    for prod in productions.iter().cloned() {
        prods_by_lhs.entry(prod.lhs.clone()).or_default().push(prod);
    }

    // Identify all non-terminals that have simple direct right-recursive rules.
    let mut recursive_nts = BTreeSet::new();
    for (nt, prods_for_nt) in &prods_by_lhs {
        if prods_for_nt.iter().any(is_simple_direct_right_recursive) {
            recursive_nts.insert(nt.clone());
        }
    }

    // Build the new production list, preserving order as much as possible.
    let mut new_productions = Vec::new();
    let mut processed_recursive_nts = BTreeSet::new();

    for prod in productions.iter().cloned() {
        let lhs = &prod.lhs;

        if !recursive_nts.contains(lhs) {
            new_productions.push(prod);
            continue;
        }

        if processed_recursive_nts.contains(lhs) {
            continue;
        }
        processed_recursive_nts.insert(lhs.clone());

        let prods_for_nt = prods_by_lhs.get(lhs).expect("LHS group missing");
        let (recursive_rules, other_rules): (Vec<_>, Vec<_>) = prods_for_nt
            .iter()
            .cloned()
            .partition(is_simple_direct_right_recursive);

        // Check if we already have a helper NT for this LHS (from a previous resolution)
        // Look for a rule A -> ... A_rr
        let existing_helper = other_rules.iter().find_map(|p| {
            if let Some(Symbol::NonTerminal(nt)) = p.rhs.last() {
                if nt.0.starts_with(&format!("{}_rr", lhs.0)) {
                    return Some(nt.clone());
                }
            }
            None
        });

        let (new_nt, reused) = if let Some(helper) = existing_helper {
            crate::debug!(7, "Reusing existing helper '{}' for '{}'", helper.0, lhs.0);
            (helper, true)
        } else {
            (NonTerminal(new_name_generator(&lhs.0)), false)
        };

        if !reused {
            crate::debug!(
                7,
                "Resolving direct right-recursion for '{}' -> '{}'",
                lhs.0,
                new_nt.0
            );
        }

        // A -> βⱼ A'
        for non_rec_rule in &other_rules {
            // If reusing helper, check if this rule is already transformed
            if reused {
                if let Some(Symbol::NonTerminal(last)) = non_rec_rule.rhs.last() {
                    if last == &new_nt {
                        // Already transformed: A -> ... A_rr
                        new_productions.push(non_rec_rule.clone());
                        continue;
                    }
                }
            }

            let mut new_rhs = Vec::with_capacity(non_rec_rule.rhs.len() + 1);
            new_rhs.extend_from_slice(&non_rec_rule.rhs);
            new_rhs.push(Symbol::NonTerminal(new_nt.clone()));
            let new_prod = Production {
                lhs: lhs.clone(),
                rhs: new_rhs,
            };
            crate::debug!(
                7,
                "  Transforming non-recursive rule '{}' -> '{}'",
                non_rec_rule,
                new_prod
            );
            new_productions.push(new_prod);
        }

        // A' -> αᵢ A'
        for rec_rule in &recursive_rules {
            let alpha = &rec_rule.rhs[..rec_rule.rhs.len() - 1];
            let mut new_rhs = Vec::with_capacity(alpha.len() + 1);
            new_rhs.extend_from_slice(alpha);
            new_rhs.push(Symbol::NonTerminal(new_nt.clone()));
            let new_prod = Production {
                lhs: new_nt.clone(),
                rhs: new_rhs,
            };
            crate::debug!(
                7,
                "  Transforming recursive rule '{}' -> '{}'",
                rec_rule,
                new_prod
            );
            new_productions.push(new_prod);
        }

        // A' -> ε
        if !reused {
            let epsilon_prod = Production {
                lhs: new_nt.clone(),
                rhs: Vec::new(),
            };
            crate::debug!(7, "  Adding new epsilon rule: '{}'", epsilon_prod);
            new_productions.push(epsilon_prod);
        }
    }

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
        let rhs_variants: Vec<Vec<Symbol>> = prod.rhs.iter().fold(vec![vec![]], |acc, sym| {
            let sym_options = match sym {
                Symbol::Terminal(_) => vec![Some(sym.clone())],
                Symbol::NonTerminal(nt) => match nullability.get(nt) {
                    Some(Nullability::Null) => vec![None],
                    Some(Nullability::Nullable) => vec![Some(sym.clone()), None],
                    _ => vec![Some(sym.clone())],
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
