use crate::glr::automaton::{
    compute_first_sets_for_nonterminals,
    compute_follow_sets_for_nonterminals,
    compute_nonterminal_nullability,
    compute_nullable_nonterminals,
    Nullability,
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

/// Checks for length-1 recursion (e.g., A ::= A, A ::= B; B ::= A), considering nullable prefixes.
pub fn check_for_length_1_recursion(productions: &[Production]) -> Vec<String> {
    let nullable_nonterminals = compute_nullable_nonterminals(productions);
    let all_nonterminals: BTreeSet<NonTerminal> = productions
        .iter()
        .flat_map(|p| {
            std::iter::once(p.lhs.clone()).chain(
                p.rhs
                    .iter()
                    .filter_map(|s| match s {
                        Symbol::NonTerminal(nt) => Some(nt.clone()),
                        _ => None,
                    }),
            )
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
            let recursion_type = if cycle.len() == 1 { "Direct" } else { "Indirect" };
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
        let defined_lhs: BTreeSet<NonTerminal> = current
            .iter()
            .map(|(_, prod)| prod.lhs.clone())
            .collect();

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
            3,
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
            2,
            "Missing non-terminals ({}) in productions:",
            all_rhs_nonterminals.len()
        );
        for nt in all_rhs_nonterminals.difference(&defined_lhs) {
            crate::debug!(4, "  {}", nt.0);
        }

        crate::debug!(5, "Removed productions:");
        for (_, prod) in &removed {
            crate::debug!(5, "  {}", prod);
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

    let new_productions: Vec<_> = productions
        .iter()
        .filter(|prod| reachable_from_start.contains(&prod.lhs) || *prod == start_prod)
        .cloned()
        .collect();

    crate::debug!(
        2,
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
        crate::debug!(
            2,
            "filter_productions_by_reachability: interesting_symbols is empty, returning no productions."
        );
        return Vec::new();
    }

    let can_derive_set =
        compute_can_derive_interesting(initial_productions, interesting_symbols);
    crate::debug!(
        3,
        "filter_productions_by_reachability: CanDeriveInteresting set: {:?}",
        can_derive_set
            .iter()
            .map(|nt| &nt.0)
            .collect::<Vec<_>>()
    );

    let mut kept_productions = Vec::new();
    for production in initial_productions {
        let lhs_can_derive_interesting = can_derive_set.contains(&production.lhs);

        let rhs_can_derive_interesting_for_this_rule =
            production.rhs.iter().any(|symbol_in_rhs| match symbol_in_rhs {
                Symbol::Terminal(_) => interesting_symbols.contains(symbol_in_rhs),
                Symbol::NonTerminal(nt_in_rhs) => {
                    interesting_symbols.contains(symbol_in_rhs)
                        || can_derive_set.contains(nt_in_rhs)
                }
            });

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
    mut new_name_generator: impl FnMut(&str) -> String,
) {
    todo!("resolve_right_recursion");
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
        let (recursive_rules, other_rules): (Vec<_>, Vec<_>) =
            prods_for_nt.iter().cloned().partition(is_simple_direct_right_recursive);

        let new_nt = NonTerminal(new_name_generator(&lhs.0));
        crate::debug!(
            5,
            "Resolving direct right-recursion for '{}' -> '{}'",
            lhs.0,
            new_nt.0
        );

        // A -> A' βⱼ
        for non_rec_rule in &other_rules {
            let mut new_rhs = Vec::with_capacity(non_rec_rule.rhs.len() + 1);
            new_rhs.push(Symbol::NonTerminal(new_nt.clone()));
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

        // A' -> A' αᵢ
        for rec_rule in &recursive_rules {
            let alpha = &rec_rule.rhs[..rec_rule.rhs.len() - 1];
            let mut new_rhs = Vec::with_capacity(alpha.len() + 1);
            new_rhs.push(Symbol::NonTerminal(new_nt.clone()));
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

        // A' -> ε
        let epsilon_prod = Production {
            lhs: new_nt.clone(),
            rhs: Vec::new(),
        };
        crate::debug!(5, "  Adding new epsilon rule: '{}'", epsilon_prod);
        new_productions.push(epsilon_prod);
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
        let rhs_variants: Vec<Vec<Symbol>> =
            prod.rhs.iter().fold(vec![vec![]], |acc, sym| {
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
