use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::{Item, LRMode, LR_MODE};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

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

pub fn compute_first_sets_for_nonterminals(productions: &[Production]) -> BTreeMap<NonTerminal, BTreeSet<Terminal>> {
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
                    let first_nt = first_sets.get(nt).cloned().unwrap_or_default(); // Handle case where nt might not be in first_sets yet
                    first_sets.get_mut(lhs).unwrap().extend(first_nt);

                    if !nullable_nonterminals.contains(nt) {
                        break;
                    }
                } else if let Symbol::Terminal(t) = symbol { // Added this case
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
    start_production_id: usize,
    first_sets: &BTreeMap<NonTerminal, BTreeSet<Terminal>>,
    nullable_nonterminals: &BTreeSet<NonTerminal>,
) -> BTreeMap<NonTerminal, BTreeSet<Option<Terminal>>> {
    let mut follow_sets: BTreeMap<NonTerminal, BTreeSet<Option<Terminal>>> = BTreeMap::new();

    // Initialize for all non-terminals
    for production in productions {
        follow_sets.entry(production.lhs.clone()).or_default();
        for symbol in &production.rhs { // Ensure all non-terminals in RHS are in follow_sets
            if let Symbol::NonTerminal(nt) = symbol {
                follow_sets.entry(nt.clone()).or_default();
            }
        }
    }

    // Rule 1: Place EOF (None) in FOLLOW(S) where S is the start symbol.
    if !productions.is_empty() {
        let start_nt = &productions[start_production_id].lhs;
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
                                follow_sets.get_mut(nt).unwrap().extend(first_next.iter().cloned().map(Some));

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

pub fn compute_first_set_for_item(
    item: &Item,
    productions: &[Production],
    first_sets: &BTreeMap<NonTerminal, BTreeSet<Terminal>>,
    nullable_nonterminals: &BTreeSet<NonTerminal>,
) -> BTreeSet<Option<Terminal>> {
    if let Some((symbol, next_item)) = item.next() {
        match symbol {
            Symbol::Terminal(t) => {
                // If the next symbol is a terminal, the first is just that terminal
                BTreeSet::from([Some(t)])
            }
            Symbol::NonTerminal(nt) => {
                let mut first_set: BTreeSet<_> = first_sets.get(&nt).cloned().unwrap_or_default().into_iter()
                    .map(Some)
                    .collect();

                if nullable_nonterminals.contains(&nt) {
                    // If the non-terminal is nullable, we also need to include the firsts for the next item
                    let next_firsts = compute_first_set_for_item(
                        &next_item,
                        productions,
                        first_sets,
                        nullable_nonterminals,
                    );
                    first_set.extend(next_firsts);
                }
                first_set
            }
        }
    } else {
        // The dot is at the end. The first is the lookahead.
        BTreeSet::from([item.lookahead.clone()])
    }
}

pub fn compute_closure(
    items: &BTreeSet<Item>,
    productions: &[Production],
    first_sets: &BTreeMap<NonTerminal, BTreeSet<Terminal>>,
    nullable_nonterminals: &BTreeSet<NonTerminal>,
    follow_sets: &BTreeMap<NonTerminal, BTreeSet<Option<Terminal>>>,

) -> BTreeSet<Item> {
    // crate::debug!(3, "Computing closure");
    let mut closure = items.clone();
    let mut worklist: VecDeque<Item> = items.iter().cloned().collect();

    while let Some(item) = worklist.pop_front() {
        if let Some((Symbol::NonTerminal(nt), next_item)) = item.next() {
            for prod in productions.iter().filter(|p| p.lhs == nt) {
                let lookaheads = compute_first_set_for_item(&next_item, productions, &first_sets, &nullable_nonterminals);
                for lookahead in lookaheads {
                    let new_item = Item {
                        production: prod.clone(),
                        dot_position: 0,
                        lookahead,
                    };
                    if closure.insert(new_item.clone()) {
                        worklist.push_back(new_item);
                    }
                }
            }
        }
    }

    if LR_MODE == LRMode::LALR {
        let mut lalr_closure = BTreeSet::new();
        let mut reduce_item_cores: BTreeMap<(Production, usize), BTreeSet<Option<Terminal>>> = BTreeMap::new();

        // Separate reduce and non-reduce items, and group reduce items by core
        for item in closure {
        reduce_item_cores.entry((item.production, item.dot_position)).or_default();
        }

        // Process reduce items by replacing their specific lookaheads with the full FOLLOW set.
        for ((prod, dot_pos), _) in reduce_item_cores {
            if let Some(follows) = follow_sets.get(&prod.lhs) {
                for lookahead in follows {
                    lalr_closure.insert(Item { production: prod.clone(), dot_position: dot_pos, lookahead: lookahead.clone() });
                }
            }
        }
        return lalr_closure;
    }
    closure
}

#[allow(dead_code)]
pub fn compute_goto(items: &BTreeSet<Item>) -> BTreeSet<Item> {
    items.iter()
        .filter_map(|item| item.next())
        .map(|(_, next_item)| next_item)
        .collect()
}

#[allow(dead_code)]
pub fn split_on_dot(items: &BTreeSet<Item>) -> BTreeMap<Option<Symbol>, BTreeSet<Item>> {
    let mut result: BTreeMap<Option<Symbol>, BTreeSet<Item>> = BTreeMap::new();
    for item in items {
        result
            .entry(item.production.rhs.get(item.dot_position).cloned())
            .or_default()
            .insert(item.clone());
    }
    result
}
