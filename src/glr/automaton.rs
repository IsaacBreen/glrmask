use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use profiler_macro::time_it;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Nullability {
    /// Can only derive strings containing terminals; cannot derive ε.
    NotNull,
    /// Can derive ε, and can also derive strings containing terminals.
    Nullable,
    /// Can only derive ε (or other non-terminals that are themselves Null).
    /// Cannot derive any string containing a terminal.
    Null,
}

pub fn compute_nonterminal_nullability(productions: &[Production]) -> BTreeMap<NonTerminal, Nullability> {
    if productions.is_empty() {
        return BTreeMap::new();
    }

    let mut all_nonterminals = BTreeSet::new();
    for p in productions {
        all_nonterminals.insert(p.lhs.clone());
        for s in &p.rhs {
            if let Symbol::NonTerminal(nt) = s {
                all_nonterminals.insert(nt.clone());
            }
        }
    }

    let mut nullability: BTreeMap<NonTerminal, (bool, bool)> = all_nonterminals
        .iter()
        .map(|nt| (nt.clone(), (false, false))) // (is_nullable, is_productive)
        .collect();

    let mut nt_dependencies: BTreeMap<NonTerminal, Vec<usize>> = BTreeMap::new();
    for (i, p) in productions.iter().enumerate() {
        for s in &p.rhs {
            if let Symbol::NonTerminal(nt) = s {
                nt_dependencies.entry(nt.clone()).or_default().push(i);
            }
        }
    }

    // Pass 1: Epsilon-derivation (is_nullable)
    let mut nullable_counters: Vec<usize> = productions.iter().map(|p| p.rhs.len()).collect();
    let mut worklist: VecDeque<NonTerminal> = VecDeque::new();

    for (i, p) in productions.iter().enumerate() {
        if p.rhs.is_empty() {
            let lhs = &p.lhs;
            if !nullability.get(lhs).unwrap().0 {
                nullability.get_mut(lhs).unwrap().0 = true;
                worklist.push_back(lhs.clone());
            }
        }
    }

    while let Some(nt) = worklist.pop_front() {
        if let Some(dependent_prods) = nt_dependencies.get(&nt) {
            for &prod_idx in dependent_prods {
                nullable_counters[prod_idx] -= 1;
                if nullable_counters[prod_idx] == 0 {
                    let lhs = &productions[prod_idx].lhs;
                    if !nullability.get(lhs).unwrap().0 {
                        nullability.get_mut(lhs).unwrap().0 = true;
                        worklist.push_back(lhs.clone());
                    }
                }
            }
        }
    }

    // Pass 2: Terminal-derivation (is_productive)
    let mut productive_counters: Vec<usize> = vec![0; productions.len()];
    for (i, p) in productions.iter().enumerate() {
        for s in &p.rhs {
            if let Symbol::NonTerminal(_) = s {
                productive_counters[i] += 1;
            }
        }
    }

    for (i, p) in productions.iter().enumerate() {
        if productive_counters[i] == 0 && !p.rhs.is_empty() {
            let lhs = &p.lhs;
            if !nullability.get(lhs).unwrap().1 {
                nullability.get_mut(lhs).unwrap().1 = true;
                worklist.push_back(lhs.clone());
            }
        }
    }

    while let Some(nt) = worklist.pop_front() {
        if let Some(dependent_prods) = nt_dependencies.get(&nt) {
            for &prod_idx in dependent_prods {
                productive_counters[prod_idx] -= 1;
                if productive_counters[prod_idx] == 0 {
                    let lhs = &productions[prod_idx].lhs;
                    if !nullability.get(lhs).unwrap().1 {
                        nullability.get_mut(lhs).unwrap().1 = true;
                        worklist.push_back(lhs.clone());
                    }
                }
            }
        }
    }

    // Combine results
    all_nonterminals
        .into_iter()
        .map(|nt| {
            let (is_nullable, is_productive) = nullability[&nt];
            let status = match (is_nullable, is_productive) {
                (true, false) => Nullability::Null,
                (true, true) => Nullability::Nullable,
                (false, true) => Nullability::NotNull,
                (false, false) => Nullability::NotNull, // Non-productive
            };
            (nt, status)
        })
        .collect()
}

pub fn compute_null_nonterminals(productions: &[Production]) -> BTreeSet<NonTerminal> {
    compute_nonterminal_nullability(productions)
        .into_iter()
        .filter_map(|(nt, status)| (status == Nullability::Null).then_some(nt))
        .collect()
}

pub fn compute_nullable_nonterminals(productions: &[Production]) -> BTreeSet<NonTerminal> {
    crate::debug!(3, "Computing nullable non-terminals");
    compute_nonterminal_nullability(productions)
        .into_iter()
        .filter_map(|(nt, status)| {
            (status == Nullability::Nullable || status == Nullability::Null).then_some(nt)
        })
        .collect()
}

pub fn compute_first_sets_for_nonterminals(
    productions: &[Production],
) -> BTreeMap<NonTerminal, BTreeSet<Terminal>> {
    crate::debug!(3, "Computing first sets for non-terminals");
    let nullable_nonterminals = compute_nullable_nonterminals(productions);
    let mut first_sets: BTreeMap<NonTerminal, BTreeSet<Terminal>> = BTreeMap::new();
    let mut prods_by_lhs: BTreeMap<NonTerminal, Vec<&Production>> = BTreeMap::new();
    let mut worklist = VecDeque::new();
    let mut influences: BTreeMap<NonTerminal, Vec<NonTerminal>> = BTreeMap::new();

    for p in productions {
        prods_by_lhs.entry(p.lhs.clone()).or_default().push(p);
        first_sets.entry(p.lhs.clone()).or_default();
        let mut prev_are_nullable = true;
        for s in &p.rhs {
            if !prev_are_nullable {
                break;
            }
            if let Symbol::NonTerminal(nt) = s {
                influences.entry(nt.clone()).or_default().push(p.lhs.clone());
                if !nullable_nonterminals.contains(nt) {
                    prev_are_nullable = false;
                }
            } else {
                prev_are_nullable = false;
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for (lhs, prods) in &prods_by_lhs {
            let old_size = first_sets.get(lhs).unwrap().len();
            for p in prods {
                for s in &p.rhs {
                    match s {
                        Symbol::Terminal(t) => {
                            first_sets.get_mut(lhs).unwrap().insert(t.clone());
                            break;
                        }
                        Symbol::NonTerminal(nt) => {
                            let first_nt = first_sets.get(nt).cloned().unwrap_or_default();
                            first_sets.get_mut(lhs).unwrap().extend(first_nt);
                            if !nullable_nonterminals.contains(nt) {
                                break;
                            }
                        }
                    }
                }
            }
            if first_sets.get(lhs).unwrap().len() != old_size {
                changed = true;
            }
        }
    }

    first_sets
}

pub fn compute_follow_sets_for_nonterminals(
    productions: &[Production],
    first_sets: &BTreeMap<NonTerminal, BTreeSet<Terminal>>,
    nullable_nonterminals: &BTreeSet<NonTerminal>,
) -> BTreeMap<NonTerminal, BTreeSet<Option<Terminal>>> {
    crate::debug!(3, "Computing follow sets for non-terminals");
    let mut follow_sets: BTreeMap<NonTerminal, BTreeSet<Option<Terminal>>> = BTreeMap::new();

    // Initialize for all non-terminals
    for production in productions {
        follow_sets.entry(production.lhs.clone()).or_default();
        for symbol in &production.rhs {
            if let Symbol::NonTerminal(nt) = symbol {
                follow_sets.entry(nt.clone()).or_default();
            }
        }
    }

    // Rule 1: Place EOF (None) in FOLLOW(S) where S is the start symbol.
    if !productions.is_empty() {
        let start_nt = &productions[0].lhs;
        follow_sets.entry(start_nt.clone()).or_default().insert(None);
    }

    let mut changed = true;
    while changed {
        changed = false;

        for production in productions {
            let lhs = &production.lhs;
            let rhs = &production.rhs;

            for (i, symbol) in rhs.iter().enumerate() {
                if let Symbol::NonTerminal(nt) = symbol {
                    let old_len = follow_sets.get(nt).unwrap().len();

                    let mut suffix_is_nullable = true;
                    for next_symbol in &rhs[i + 1..] {
                        match next_symbol {
                            Symbol::Terminal(t_next) => {
                                follow_sets.get_mut(nt).unwrap().insert(Some(t_next.clone()));
                                suffix_is_nullable = false;
                                break;
                            }
                            Symbol::NonTerminal(nt_next) => {
                                let first_next = first_sets.get(nt_next).cloned().unwrap_or_default();
                                follow_sets
                                    .get_mut(nt)
                                    .unwrap()
                                    .extend(first_next.iter().cloned().map(Some));

                                if !nullable_nonterminals.contains(nt_next) {
                                    suffix_is_nullable = false;
                                    break;
                                }
                            }
                        }
                    }

                    if suffix_is_nullable {
                        let follow_lhs = follow_sets.get(lhs).unwrap().clone();
                        if follow_sets.get_mut(nt).unwrap().extend(follow_lhs) > 0 {
                           // The set was modified, but extend returns () in stable Rust.
                           // We rely on the length check below.
                        }
                    }

                    if follow_sets.get(nt).unwrap().len() != old_len {
                        changed = true;
                    }
                }
            }
        }
    }

    follow_sets
}

#[time_it]
pub fn compute_closure(
    items: &BTreeSet<Item>,
    prods_by_lhs: &BTreeMap<NonTerminal, Vec<usize>>,
    productions: &[Production],
) -> BTreeSet<Item> {
    // crate::debug!(3, "Computing closure");
    let mut closure = items.clone();
    let mut worklist: VecDeque<Item> = items.iter().cloned().collect();

    while let Some(item) = worklist.pop_front() {
        let prod = &productions[item.production_id];
        if let Some(Symbol::NonTerminal(nt)) = prod.rhs.get(item.dot_position) {
            if let Some(prod_indices) = prods_by_lhs.get(nt) {
                for &prod_idx in prod_indices {
                    let new_item = Item {
                        production_id: prod_idx,
                        dot_position: 0,
                    };
                    if closure.insert(new_item) {
                        worklist.push_back(new_item);
                    }
                }
            }
        }
    }
    closure
}

pub fn compute_goto(items: &BTreeSet<Item>, productions: &[Production]) -> BTreeSet<Item> {
    items
        .iter()
        .filter_map(|item| {
            let prod = &productions[item.production_id];
            if item.dot_position < prod.rhs.len() {
                Some(Item {
                    production_id: item.production_id,
                    dot_position: item.dot_position + 1,
                })
            } else {
                None
            }
        })
        .collect()
}

pub fn split_on_dot(
    items: &BTreeSet<Item>,
    productions: &[Production],
) -> BTreeMap<Option<Symbol>, BTreeSet<Item>> {
    let mut result: BTreeMap<Option<Symbol>, BTreeSet<Item>> = BTreeMap::new();
    for item in items {
        let prod = &productions[item.production_id];
        let key = prod.rhs.get(item.dot_position).cloned();
        result.entry(key).or_default().insert(item.clone());
    }
    result
}
