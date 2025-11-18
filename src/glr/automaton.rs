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

    // 1. Collect all non-terminals from the grammar.
    let mut all_nonterminals = BTreeSet::new();
    for p in productions {
        all_nonterminals.insert(p.lhs.clone());
        for s in &p.rhs {
            if let Symbol::NonTerminal(nt) = s {
                all_nonterminals.insert(nt.clone());
            }
        }
    }

    // Pass 1: Determine which non-terminals can derive ε.
    let mut can_derive_epsilon = BTreeSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        for p in productions {
            if !can_derive_epsilon.contains(&p.lhs) {
                let rhs_is_nullable = p.rhs.iter().all(|symbol| match symbol {
                    Symbol::Terminal(_) => false,
                    Symbol::NonTerminal(nt) => can_derive_epsilon.contains(nt),
                });
                if rhs_is_nullable {
                    can_derive_epsilon.insert(p.lhs.clone());
                    changed = true;
                }
            }
        }
    }

    // Pass 2: Determine which non-terminals are productive (can derive a terminal string).
    let mut can_derive_terminal = BTreeSet::new();
    changed = true;
    while changed {
        changed = false;
        for p in productions {
            if !can_derive_terminal.contains(&p.lhs) {
                let rhs_is_productive = p.rhs.iter().any(|symbol| match symbol {
                    Symbol::Terminal(_) => true,
                    Symbol::NonTerminal(nt) => can_derive_terminal.contains(nt),
                });
                if rhs_is_productive {
                    can_derive_terminal.insert(p.lhs.clone());
                    changed = true;
                }
            }
        }
    }

    // Combine results.
    all_nonterminals
        .into_iter()
        .map(|nt| {
            let is_nullable = can_derive_epsilon.contains(&nt);
            let is_productive = can_derive_terminal.contains(&nt);

            let status = match (is_nullable, is_productive) {
                (true, false) => Nullability::Null,
                (true, true) => Nullability::Nullable,
                (false, true) => Nullability::NotNull,
                // A non-productive non-terminal that cannot derive ε is just a dead end.
                // It doesn't fit neatly, but NotNull is the safest classification.
                (false, false) => Nullability::NotNull,
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
    // TODO: should this account for EOF? Return `BTreeMap<NonTerminal, BTreeSet<Option<Terminal>>>`?
    crate::debug!(3, "Computing first sets for non-terminals");
    let nullable_nonterminals = compute_nullable_nonterminals(productions);
    let mut first_sets: BTreeMap<NonTerminal, BTreeSet<Terminal>> = BTreeMap::new();

    // Initialize for all non-terminals to avoid panics and handle non-terminals that only appear on RHS.
    for p in productions {
        first_sets.entry(p.lhs.clone()).or_default();
        for s in &p.rhs {
            if let Symbol::NonTerminal(nt) = s {
                first_sets.entry(nt.clone()).or_default();
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;

        for production in productions {
            let lhs = &production.lhs;
            let rhs = &production.rhs;

            let old_size = first_sets.get(lhs).unwrap().len();

            for symbol in rhs {
                if let Symbol::NonTerminal(nt) = symbol {
                    let first_nt = first_sets.get(nt).cloned().unwrap_or_default();
                    first_sets.get_mut(lhs).unwrap().extend(first_nt);

                    if !nullable_nonterminals.contains(nt) {
                        break;
                    }
                } else if let Symbol::Terminal(t) = symbol {
                    first_sets.get_mut(lhs).unwrap().insert(t.clone());
                    break;
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
                        follow_sets.get_mut(nt).unwrap().extend(follow_lhs);
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
