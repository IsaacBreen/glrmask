use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::{Item, LRMode, LR_MODE};
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
    all_nonterminals.into_iter().map(|nt| {
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
    }).collect()
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
        .filter_map(|(nt, status)| (status == Nullability::Nullable || status == Nullability::Null).then_some(nt))
        .collect()
}

pub fn compute_first_sets_for_nonterminals(productions: &[Production]) -> BTreeMap<NonTerminal, BTreeSet<Terminal>> {
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

            // Start with the FOLLOW set of the rule's LHS.
            // This will be propagated to any non-terminals at the end of the rule.
            let mut trailer = follow_sets.get(lhs).unwrap().clone();

            for symbol in rhs.iter().rev() {
                match symbol {
                    Symbol::Terminal(t) => {
                        // The new trailer is just the FIRST set of this terminal.
                        trailer = BTreeSet::from([Some(t.clone())]);
                    }
                    Symbol::NonTerminal(nt) => {
                        // The FOLLOW set of the current non-terminal `nt` is updated with the trailer.
                        let follow_nt = follow_sets.get_mut(nt).unwrap();
                        let old_len = follow_nt.len();
                        follow_nt.extend(trailer.iter().cloned());
                        if follow_nt.len() != old_len {
                            changed = true;
                        }

                        // Update the trailer for the next symbol to the left.
                        if nullable_nonterminals.contains(nt) {
                            // If `nt` is nullable, the new trailer is FIRST(nt) U old_trailer.
                            if let Some(first_nt) = first_sets.get(nt) {
                                trailer.extend(first_nt.iter().cloned().map(Some));
                            }
                        } else {
                            // If `nt` is not nullable, the new trailer is just FIRST(nt).
                            trailer = first_sets.get(nt)
                                .cloned()
                                .unwrap_or_default()
                                .into_iter()
                                .map(Some)
                                .collect();
                        }
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

    match LR_MODE {
        LRMode::LALR | LRMode::LALR_EX_GOTO => {
            let item_set_is_next_for_any_shift = items.iter().any(|item| {
                item.dot_position
                    .checked_sub(1)
                    .and_then(|pos| item.production.rhs.get(pos))
                    .map_or(false, |symbol| matches!(symbol, Symbol::Terminal(_)))
            });

            if LR_MODE == LRMode::LALR_EX_GOTO && !item_set_is_next_for_any_shift {
                crate::debug!(4, "Item set is next for any shift. Computing LR(1) closure.");
                closure
            } else {
                crate::debug!(4, "Item set is not next for any shift. Computing LALR closure.");
                let mut lalr_closure = BTreeSet::new();
                let mut reduce_item_cores: BTreeMap<(Production, usize), BTreeSet<Option<Terminal>>> = BTreeMap::new();

                // Separate reduce and non-reduce items, and group reduce items by core
                for item in closure {
                    reduce_item_cores.entry((item.production, item.dot_position)).or_default().insert(item.lookahead);
                }

                // Process reduce items by replacing their specific lookaheads with the full FOLLOW set.
                for ((prod, dot_pos), existing_lookaheads) in reduce_item_cores {
                    if let Some(follows) = follow_sets.get(&prod.lhs) {
                        for lookahead in follows {
                            if lookahead == &None && !existing_lookaheads.contains(&None) {
                                continue;
                            }
                            lalr_closure.insert(Item { production: prod.clone(), dot_position: dot_pos, lookahead: lookahead.clone() });
                        }
                    }
                }
                lalr_closure
            }
        }
        LRMode::LR1 => closure,
    }
}

pub fn compute_goto(items: &BTreeSet<Item>) -> BTreeSet<Item> {
    items.iter()
        .filter_map(|item| item.next())
        .map(|(_, next_item)| next_item)
        .collect()
}

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
