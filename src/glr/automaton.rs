use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::{Item, LRMode, LR_MODE};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nullability {
    /// Can derive a terminal, but not epsilon.
    NotNull,
    /// Can derive epsilon, and can also derive a terminal.
    Nullable,
    /// Can only derive epsilon (or other Null non-terminals), never a terminal.
    Null,
}

impl Nullability {
    pub fn is_nullable(&self) -> bool {
        matches!(self, Nullability::Nullable | Nullability::Null)
    }
}

pub fn compute_nonterminal_nullability(productions: &[Production]) -> BTreeMap<NonTerminal, Nullability> {
    if productions.is_empty() {
        return BTreeMap::new();
    }

    // 1. Find all non-terminals that can derive epsilon.
    let mut can_derive_epsilon = BTreeSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        for p in productions {
            if p.rhs.is_empty() {
                if can_derive_epsilon.insert(p.lhs.clone()) {
                    changed = true;
                }
            } else if p.rhs.iter().all(|s| match s {
                Symbol::NonTerminal(nt) => can_derive_epsilon.contains(nt),
                Symbol::Terminal(_) => false,
            }) {
                if can_derive_epsilon.insert(p.lhs.clone()) {
                    changed = true;
                }
            }
        }
    }

    // 2. Find all non-terminals that can derive a terminal string (are "productive").
    let mut productive_nts = BTreeSet::new();
    changed = true;
    while changed {
        changed = false;
        for p in productions {
            if productive_nts.contains(&p.lhs) {
                continue;
            }
            if p.rhs.iter().any(|s| match s {
                Symbol::Terminal(_) => true,
                Symbol::NonTerminal(nt) => productive_nts.contains(nt),
            }) {
                if productive_nts.insert(p.lhs.clone()) {
                    changed = true;
                }
            }
            changed = true;
        }
    }

    // 3. Collect all non-terminals and combine results.
    let mut all_nonterminals = BTreeSet::new();
    for p in productions {
        all_nonterminals.insert(p.lhs.clone());
        for s in &p.rhs {
            if let Symbol::NonTerminal(nt) = s {
                all_nonterminals.insert(nt.clone());
            }
        }
    }

    let mut nullability_map = BTreeMap::new();
    for nt in all_nonterminals {
        let is_nullable = can_derive_epsilon.contains(&nt);
        let is_productive = productive_nts.contains(&nt);

        let nullability = match (is_nullable, is_productive) {
            (true, true) => Nullability::Nullable,
            (true, false) => Nullability::Null,
            (false, _) => Nullability::NotNull, // A non-productive, non-nullable NT is useless, but we classify it as NotNull.
        };
        nullability_map.insert(nt, nullability);
    }

    nullability_map
}

pub fn compute_first_sets_for_nonterminals(productions: &[Production]) -> BTreeMap<NonTerminal, BTreeSet<Terminal>> {
    let nullability_map = compute_nonterminal_nullability(productions);
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

                    if !nullability_map.get(nt).map_or(false, |n| n.is_nullable()) {
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
    nullability_map: &BTreeMap<NonTerminal, Nullability>,
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
                                follow_sets.get_mut(nt).unwrap().extend(first_next.iter().cloned().map(Some));

                                if !nullability_map.get(nt_next).map_or(false, |n| n.is_nullable()) {
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
    nullability_map: &BTreeMap<NonTerminal, Nullability>,
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

                if nullability_map.get(&nt).map_or(false, |n| n.is_nullable()) {
                    // If the non-terminal is nullable, we also need to include the firsts for the next item
                    let next_firsts = compute_first_set_for_item(
                        &next_item,
                        productions,
                        first_sets,
                        nullability_map,
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
    nullability_map: &BTreeMap<NonTerminal, Nullability>,
    follow_sets: &BTreeMap<NonTerminal, BTreeSet<Option<Terminal>>>,

) -> BTreeSet<Item> {
    // crate::debug!(3, "Computing closure");
    let mut closure = items.clone();
    let mut worklist: VecDeque<Item> = items.iter().cloned().collect();

    while let Some(item) = worklist.pop_front() {
        if let Some((Symbol::NonTerminal(nt), next_item)) = item.next() {
            for prod in productions.iter().filter(|p| p.lhs == nt) {
                let lookaheads = compute_first_set_for_item(&next_item, productions, &first_sets, &nullability_map);
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
