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

pub fn compute_nonterminal_nullability(
    productions: &[Production],
) -> BTreeMap<NonTerminal, Nullability> {
    use bimap::BiBTreeMap;
    use std::iter;

    if productions.is_empty() {
        return BTreeMap::new();
    }

    // 1. Assign integer IDs to non-terminals.
    let mut nt_map = BiBTreeMap::new();
    let mut all_nts = Vec::new();
    for p in productions {
        for nt in iter::once(&p.lhs).chain(p.rhs.iter().filter_map(|s| match s {
            Symbol::NonTerminal(nt) => Some(nt),
            _ => None,
        })) {
            if !nt_map.contains_left(nt) {
                let id = all_nts.len();
                nt_map.insert(nt.clone(), id);
                all_nts.push(nt.clone());
            }
        }
    }
    let num_nts = all_nts.len();

    // Map productions to use NT IDs.
    let prods_with_ids: Vec<(usize, Vec<Option<usize>>)> = productions
        .iter()
        .map(|p| {
            let lhs_id = *nt_map.get_by_left(&p.lhs).unwrap();
            let rhs_ids: Vec<Option<usize>> = p
                .rhs
                .iter()
                .map(|s| match s {
                    Symbol::Terminal(_) => None,
                    Symbol::NonTerminal(nt) => Some(*nt_map.get_by_left(nt).unwrap()),
                })
                .collect();
            (lhs_id, rhs_ids)
        })
        .collect();

    let n_prods = productions.len();
    let mut nt_rhs_occurs_by_id: Vec<Vec<usize>> = vec![Vec::new(); num_nts];
    for (idx, (_, rhs_ids)) in prods_with_ids.iter().enumerate() {
        for id_opt in rhs_ids {
            if let Some(id) = id_opt {
                nt_rhs_occurs_by_id[*id].push(idx);
            }
        }
    }

    // For ε computation: only productions whose RHS contains no terminals matter.
    let mut eps_relevant = vec![true; n_prods];
    let mut eps_unsatisfied = vec![0usize; n_prods];

    for (idx, (_, rhs_ids)) in prods_with_ids.iter().enumerate() {
        let mut nonterm_count = 0usize;
        for id_opt in rhs_ids {
            if id_opt.is_none() {
                // Terminal
                eps_relevant[idx] = false;
                break;
            }
            nonterm_count += 1;
        }
        if eps_relevant[idx] {
            eps_unsatisfied[idx] = nonterm_count;
        }
    }

    // ---------------------------------------------------------------------
    // Phase 1: compute the set of non-terminals that can derive ε.
    // ---------------------------------------------------------------------
    let mut can_derive_epsilon_by_id = vec![false; num_nts];
    let mut queue_eps: VecDeque<usize> = VecDeque::new();

    for (idx, (lhs_id, _)) in prods_with_ids.iter().enumerate() {
        if eps_relevant[idx] && eps_unsatisfied[idx] == 0 {
            if !can_derive_epsilon_by_id[*lhs_id] {
                can_derive_epsilon_by_id[*lhs_id] = true;
                queue_eps.push_back(*lhs_id);
            }
        }
    }

    while let Some(nt_id) = queue_eps.pop_front() {
        for &p_idx in &nt_rhs_occurs_by_id[nt_id] {
            if !eps_relevant[p_idx] {
                continue;
            }
            if eps_unsatisfied[p_idx] == 0 {
                continue;
            }
            eps_unsatisfied[p_idx] -= 1;
            if eps_unsatisfied[p_idx] == 0 {
                let lhs_id = prods_with_ids[p_idx].0;
                if !can_derive_epsilon_by_id[lhs_id] {
                    can_derive_epsilon_by_id[lhs_id] = true;
                    queue_eps.push_back(lhs_id);
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Phase 2: compute the set of non-terminals that can derive some string
    // containing at least one terminal.
    // ---------------------------------------------------------------------
    let mut prod_unsatisfied = vec![0usize; n_prods];
    let mut prod_has_terminal_or_prod_nt = vec![false; n_prods];

    for (idx, (_, rhs_ids)) in prods_with_ids.iter().enumerate() {
        for id_opt in rhs_ids {
            match id_opt {
                None => {
                    // Terminal
                    prod_has_terminal_or_prod_nt[idx] = true;
                }
                Some(nt_id) => {
                    if !can_derive_epsilon_by_id[*nt_id] {
                        prod_unsatisfied[idx] += 1;
                    }
                }
            }
        }
    }

    let mut can_derive_terminal_by_id = vec![false; num_nts];
    let mut queue_term: VecDeque<usize> = VecDeque::new();

    for (idx, (lhs_id, _)) in prods_with_ids.iter().enumerate() {
        if prod_has_terminal_or_prod_nt[idx] && prod_unsatisfied[idx] == 0 {
            if !can_derive_terminal_by_id[*lhs_id] {
                can_derive_terminal_by_id[*lhs_id] = true;
                queue_term.push_back(*lhs_id);
            }
        }
    }

    while let Some(nt_id) = queue_term.pop_front() {
        for &p_idx in &nt_rhs_occurs_by_id[nt_id] {
            if !can_derive_epsilon_by_id[nt_id] && prod_unsatisfied[p_idx] > 0 {
                prod_unsatisfied[p_idx] -= 1;
            }
            if !prod_has_terminal_or_prod_nt[p_idx] {
                prod_has_terminal_or_prod_nt[p_idx] = true;
            }

            if prod_unsatisfied[p_idx] == 0 && prod_has_terminal_or_prod_nt[p_idx] {
                let lhs_id = prods_with_ids[p_idx].0;
                if !can_derive_terminal_by_id[lhs_id] {
                    can_derive_terminal_by_id[lhs_id] = true;
                    queue_term.push_back(lhs_id);
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Combine ε-derivability and terminal-derivability into Nullability.
    // ---------------------------------------------------------------------
    all_nts
        .into_iter()
        .map(|nt| {
            let id = *nt_map.get_by_left(&nt).unwrap();
            let is_nullable = can_derive_epsilon_by_id[id];
            let is_productive = can_derive_terminal_by_id[id];

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
    nullable_nonterminals: &BTreeSet<NonTerminal>,
) -> BTreeMap<NonTerminal, BTreeSet<Terminal>> {
    crate::debug!(3, "Computing first sets for non-terminals");
    use bimap::BiBTreeMap;
    use std::collections::HashSet;
    use std::iter;

    // 1. Assign integer IDs to non-terminals for performance.
    let mut nt_map = BiBTreeMap::new();
    let mut all_nts = Vec::new();
    for p in productions {
        for nt in iter::once(&p.lhs).chain(p.rhs.iter().filter_map(|s| match s {
            Symbol::NonTerminal(nt) => Some(nt),
            _ => None,
        })) {
            if !nt_map.contains_left(nt) {
                let id = all_nts.len();
                nt_map.insert(nt.clone(), id);
                all_nts.push(nt.clone());
            }
        }
    }
    let num_nts = all_nts.len();
    let nullable_ids: HashSet<usize> = nullable_nonterminals
        .iter()
        .filter_map(|nt| nt_map.get_by_left(nt).copied())
        .collect();

    // 2. Use Vecs indexed by ID for data structures.
    let mut first_sets_by_id: Vec<HashSet<Terminal>> = vec![HashSet::new(); num_nts];
    let mut deps_by_id: Vec<Vec<usize>> = vec![Vec::new(); num_nts];
    let mut worklist: VecDeque<(usize, Terminal)> = VecDeque::new();

    // 3. Build dependency graph and seed worklist with direct terminals.
    for p in productions {
        let lhs_id = *nt_map.get_by_left(&p.lhs).unwrap();
        let mut prefix_nullable = true;

        for sym in &p.rhs {
            if !prefix_nullable {
                break;
            }
            match sym {
                Symbol::Terminal(t) => {
                    if first_sets_by_id[lhs_id].insert(t.clone()) {
                        worklist.push_back((lhs_id, t.clone()));
                    }
                    prefix_nullable = false;
                }
                Symbol::NonTerminal(nt) => {
                    let nt_id = *nt_map.get_by_left(nt).unwrap();
                    deps_by_id[nt_id].push(lhs_id);
                    if !nullable_ids.contains(&nt_id) {
                        prefix_nullable = false;
                    }
                }
            }
        }
    }

    // 4. Propagate terminals along the dependency graph.
    while let Some((nt_id, terminal)) = worklist.pop_front() {
        for &dependent_lhs_id in &deps_by_id[nt_id] {
            if first_sets_by_id[dependent_lhs_id].insert(terminal.clone()) {
                worklist.push_back((dependent_lhs_id, terminal.clone()));
            }
        }
    }

    // 5. Convert back to the required BTreeMap format for deterministic output.
    all_nts
        .into_iter()
        .map(|nt| {
            let id = *nt_map.get_by_left(&nt).unwrap();
            let set = first_sets_by_id[id].iter().cloned().collect();
            (nt, set)
        })
        .collect()
}

pub fn compute_follow_sets_for_nonterminals(
    productions: &[Production],
    first_sets: &BTreeMap<NonTerminal, BTreeSet<Terminal>>,
    nullable_nonterminals: &BTreeSet<NonTerminal>,
) -> BTreeMap<NonTerminal, BTreeSet<Option<Terminal>>> {
    crate::debug!(3, "Computing follow sets for non-terminals");
    use bimap::BiBTreeMap;
    use std::iter;

    if productions.is_empty() {
        return BTreeMap::new();
    }

    // 1. Assign integer IDs to non-terminals.
    let mut nt_map = BiBTreeMap::new();
    let mut all_nts = Vec::new();
    for p in productions {
        for nt in iter::once(&p.lhs).chain(p.rhs.iter().filter_map(|s| match s {
            Symbol::NonTerminal(nt) => Some(nt),
            _ => None,
        })) {
            if !nt_map.contains_left(nt) {
                let id = all_nts.len();
                nt_map.insert(nt.clone(), id);
                all_nts.push(nt.clone());
            }
        }
    }
    let num_nts = all_nts.len();

    // 2. Use Vecs indexed by ID for data structures.
    let mut follow_sets_by_id: Vec<BTreeSet<Option<Terminal>>> = vec![BTreeSet::new(); num_nts];
    let mut edges_by_id: Vec<Vec<usize>> = vec![Vec::new(); num_nts];

    // Rule 1: EOF (None) is in FOLLOW(S) where S is the start symbol.
    let start_nt = &productions[0].lhs;
    let start_id = *nt_map.get_by_left(start_nt).unwrap();
    follow_sets_by_id[start_id].insert(None);

    // Rules 2 & 3: For each A -> α B β, add FIRST(β) \ {ε} to FOLLOW(B),
    // and if β is nullable, add an edge A -> B for propagation of FOLLOW(A).
    for production in productions {
        let lhs_id = *nt_map.get_by_left(&production.lhs).unwrap();
        let rhs = &production.rhs;
        let n = rhs.len();

        for i in 0..n {
            if let Symbol::NonTerminal(ref b_nt) = rhs[i] {
                let b_id = *nt_map.get_by_left(b_nt).unwrap();
                let mut first_of_suffix: BTreeSet<Terminal> = BTreeSet::new();
                let mut suffix_nullable = true;

                for symbol in &rhs[i + 1..] {
                    match symbol {
                        Symbol::Terminal(t) => {
                            first_of_suffix.insert(t.clone());
                            suffix_nullable = false;
                            break;
                        }
                        Symbol::NonTerminal(nt) => {
                            if let Some(first_nt) = first_sets.get(nt) {
                                first_of_suffix.extend(first_nt.iter().cloned());
                            }
                            if !nullable_nonterminals.contains(nt) {
                                suffix_nullable = false;
                                break;
                            }
                        }
                    }
                }

                let dest = &mut follow_sets_by_id[b_id];
                for t in first_of_suffix {
                    dest.insert(Some(t));
                }

                if suffix_nullable {
                    edges_by_id[lhs_id].push(b_id);
                }
            }
        }
    }

    // Worklist algorithm to propagate FOLLOW sets along the edges A -> B.
    let mut worklist: VecDeque<usize> = VecDeque::new();
    let mut in_queue: BTreeSet<usize> = BTreeSet::new();

    for id in 0..num_nts {
        if !follow_sets_by_id[id].is_empty() {
            worklist.push_back(id);
            in_queue.insert(id);
        }
    }

    while let Some(a_id) = worklist.pop_front() {
        in_queue.remove(&a_id);
        let src = follow_sets_by_id[a_id].clone();

        if src.is_empty() {
            continue;
        }

        for &b_id in &edges_by_id[a_id] {
            let dest = &mut follow_sets_by_id[b_id];
            let old_len = dest.len();
            dest.extend(src.iter().cloned());
            if dest.len() != old_len && !in_queue.contains(&b_id) {
                worklist.push_back(b_id);
                in_queue.insert(b_id);
            }
        }
    }

    // 5. Convert back to the required BTreeMap format.
    all_nts
        .into_iter()
        .map(|nt| {
            let id = *nt_map.get_by_left(&nt).unwrap();
            (nt, follow_sets_by_id[id].clone())
        })
        .collect()
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
