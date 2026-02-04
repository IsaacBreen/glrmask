use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{NonTerminalID, TerminalID};
use profiler_macro::time_it;
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

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
    if productions.is_empty() {
        return BTreeMap::new();
    }

    // Collect all non-terminals and record where they appear on the RHS.
    let mut all_nonterminals = BTreeSet::new();
    let n_prods = productions.len();
    let mut nt_rhs_occurs: BTreeMap<NonTerminal, Vec<usize>> = BTreeMap::new();

    // For ε computation: only productions whose RHS contains no terminals matter.
    let mut eps_relevant = vec![true; n_prods];
    let mut eps_unsatisfied = vec![0usize; n_prods];

    for (idx, p) in productions.iter().enumerate() {
        all_nonterminals.insert(p.lhs.clone());

        let mut has_terminal = false;
        let mut nonterm_count = 0usize;

        for sym in &p.rhs {
            match sym {
                Symbol::Terminal(_) => {
                    has_terminal = true;
                }
                Symbol::NonTerminal(nt) => {
                    all_nonterminals.insert(nt.clone());
                    nonterm_count += 1;
                    // Record each occurrence so that we can update counts accurately.
                    nt_rhs_occurs.entry(nt.clone()).or_default().push(idx);
                }
            }
        }

        if has_terminal {
            eps_relevant[idx] = false;
        } else {
            eps_unsatisfied[idx] = nonterm_count;
        }
    }

    // ---------------------------------------------------------------------
    // Phase 1: compute the set of non-terminals that can derive ε.
    // ---------------------------------------------------------------------
    let mut can_derive_epsilon = BTreeSet::new();
    let mut queue_eps: VecDeque<NonTerminal> = VecDeque::new();

    for (idx, p) in productions.iter().enumerate() {
        if eps_relevant[idx] && eps_unsatisfied[idx] == 0 {
            if can_derive_epsilon.insert(p.lhs.clone()) {
                queue_eps.push_back(p.lhs.clone());
            }
        }
    }

    while let Some(nt) = queue_eps.pop_front() {
        if let Some(prods_using) = nt_rhs_occurs.get(&nt) {
            for &p_idx in prods_using {
                if !eps_relevant[p_idx] {
                    continue;
                }
                if eps_unsatisfied[p_idx] == 0 {
                    continue;
                }
                eps_unsatisfied[p_idx] -= 1;
                if eps_unsatisfied[p_idx] == 0 {
                    let lhs = productions[p_idx].lhs.clone();
                    if can_derive_epsilon.insert(lhs.clone()) {
                        queue_eps.push_back(lhs);
                    }
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

    for (idx, p) in productions.iter().enumerate() {
        for sym in &p.rhs {
            match sym {
                Symbol::Terminal(_) => {
                    prod_has_terminal_or_prod_nt[idx] = true;
                }
                Symbol::NonTerminal(nt) => {
                    if !can_derive_epsilon.contains(nt) {
                        prod_unsatisfied[idx] += 1;
                    }
                }
            }
        }
    }

    let mut can_derive_terminal = BTreeSet::new();
    let mut queue_term: VecDeque<NonTerminal> = VecDeque::new();

    for (idx, p) in productions.iter().enumerate() {
        if prod_has_terminal_or_prod_nt[idx] && prod_unsatisfied[idx] == 0 {
            if can_derive_terminal.insert(p.lhs.clone()) {
                queue_term.push_back(p.lhs.clone());
            }
        }
    }

    while let Some(nt) = queue_term.pop_front() {
        if let Some(prods_using) = nt_rhs_occurs.get(&nt) {
            for &p_idx in prods_using {
                if !can_derive_epsilon.contains(&nt) && prod_unsatisfied[p_idx] > 0 {
                    prod_unsatisfied[p_idx] -= 1;
                }

                if !prod_has_terminal_or_prod_nt[p_idx] {
                    prod_has_terminal_or_prod_nt[p_idx] = true;
                }

                if prod_unsatisfied[p_idx] == 0 && prod_has_terminal_or_prod_nt[p_idx] {
                    let lhs = productions[p_idx].lhs.clone();
                    if can_derive_terminal.insert(lhs.clone()) {
                        queue_term.push_back(lhs);
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Combine ε-derivability and terminal-derivability into Nullability.
    // ---------------------------------------------------------------------
    all_nonterminals
        .into_iter()
        .map(|nt| {
            let is_nullable = can_derive_epsilon.contains(&nt);
            let is_productive = can_derive_terminal.contains(&nt);

            let status = match (is_nullable, is_productive) {
                (true, false) => Nullability::Null,
                (true, true) => Nullability::Nullable,
                (false, true) => Nullability::NotNull,
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
    crate::debug!(6, "Computing nullable non-terminals");
    let start = std::time::Instant::now();
    let res = compute_nonterminal_nullability(productions)
        .into_iter()
        .filter_map(|(nt, status)| {
            (status == Nullability::Nullable || status == Nullability::Null).then_some(nt)
        })
        .collect();
    crate::debug!(6, "Computed nullable non-terminals in {:.2?}", start.elapsed());
    res
}

pub fn compute_first_sets_for_nonterminals(
    productions: &[Production],
    nullable_nonterminals: &BTreeSet<NonTerminal>,
) -> BTreeMap<NonTerminal, BTreeSet<Terminal>> {
    crate::debug!(5, "Computing first sets for non-terminals");
    let start = std::time::Instant::now();
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
    let res = all_nts
        .into_iter()
        .map(|nt| {
            let id = *nt_map.get_by_left(&nt).unwrap();
            let set = first_sets_by_id[id].iter().cloned().collect();
            (nt, set)
        })
        .collect();
    crate::debug!(5, "Computed first sets in {:.2?}", start.elapsed());
    res
}

pub fn compute_follow_sets_for_nonterminals(
    productions: &[Production],
    first_sets: &BTreeMap<NonTerminal, BTreeSet<Terminal>>,
    nullable_nonterminals: &BTreeSet<NonTerminal>,
) -> BTreeMap<NonTerminal, BTreeSet<Option<Terminal>>> {
    crate::debug!(5, "Computing follow sets for non-terminals");
    let start = std::time::Instant::now();

    let mut follow_sets: BTreeMap<NonTerminal, BTreeSet<Option<Terminal>>> = BTreeMap::new();
    let mut edges: BTreeMap<NonTerminal, Vec<NonTerminal>> = BTreeMap::new();

    // Initialize FOLLOW sets for all non-terminals that appear.
    for production in productions {
        follow_sets.entry(production.lhs.clone()).or_default();
        for symbol in &production.rhs {
            if let Symbol::NonTerminal(nt) = symbol {
                follow_sets.entry(nt.clone()).or_default();
            }
        }
    }

    if productions.is_empty() {
        return follow_sets;
    }

    // Rule 1: EOF (None) is in FOLLOW(S) where S is the start symbol.
    let start_nt = productions[0].lhs.clone();
    follow_sets.entry(start_nt.clone()).or_default().insert(None);

    // Rules 2 & 3: For each A -> α B β, add FIRST(β) \ {ε} to FOLLOW(B),
    // and if β is nullable, add an edge A -> B for propagation of FOLLOW(A).
    for production in productions {
        let lhs = &production.lhs;
        let rhs = &production.rhs;
        let n = rhs.len();

        for i in 0..n {
            if let Symbol::NonTerminal(ref b_nt) = rhs[i] {
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

                let follow_b = follow_sets.entry(b_nt.clone()).or_default();
                for t in first_of_suffix {
                    follow_b.insert(Some(t));
                }

                if suffix_nullable {
                    edges.entry(lhs.clone()).or_default().push(b_nt.clone());
                }
            }
        }
    }

    // Worklist algorithm to propagate FOLLOW sets along the edges A -> B.
    let mut worklist: VecDeque<NonTerminal> = VecDeque::new();
    let mut in_queue: BTreeSet<NonTerminal> = BTreeSet::new();

    for (nt, set) in &follow_sets {
        if !set.is_empty() {
            worklist.push_back(nt.clone());
            in_queue.insert(nt.clone());
        }
    }

    while let Some(a_nt) = worklist.pop_front() {
        in_queue.remove(&a_nt);
        let src = match follow_sets.get(&a_nt) {
            Some(s) => s.clone(),
            None => continue,
        };

        if let Some(targets) = edges.get(&a_nt) {
            for b_nt in targets {
                let dest = follow_sets.entry(b_nt.clone()).or_default();
                let old_len = dest.len();
                dest.extend(src.iter().cloned());
                if dest.len() != old_len && !in_queue.contains(b_nt) {
                    worklist.push_back(b_nt.clone());
                    in_queue.insert(b_nt.clone());
                }
            }
        }
    }

    crate::debug!(5, "Computed follow sets in {:.2?}", start.elapsed());
    follow_sets
}

#[time_it]
pub fn compute_closure(
    items: &BTreeSet<Item>,
    prods_by_lhs: &BTreeMap<NonTerminal, Vec<usize>>,
    productions: &[Production],
) -> BTreeSet<Item> {
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

#[allow(dead_code)]
pub fn compute_first_sets_ids(
    _light_productions: &[Vec<usize>],
    _num_terminals: usize,
    _num_nonterminals: usize,
    _nullable_nts: &HashSet<usize>,
) -> Vec<BTreeSet<TerminalID>> {
    panic!("Use compute_first_sets_ids_with_lhs instead");
}

pub fn compute_first_sets_ids_with_lhs(
    light_productions: &[Vec<usize>],
    lhs_ids: &[usize],
    num_terminals: usize,
    num_nonterminals: usize,
    nullable_nts: &HashSet<usize>,
) -> Vec<BTreeSet<TerminalID>> {
    let mut first_sets = vec![BTreeSet::new(); num_nonterminals];
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); num_nonterminals];
    let mut worklist: VecDeque<(usize, TerminalID)> = VecDeque::new();

    for (prod_idx, rhs) in light_productions.iter().enumerate() {
        let lhs_id = lhs_ids[prod_idx];
        let mut prefix_nullable = true;

        for &sym_id in rhs {
            if !prefix_nullable {
                break;
            }
            if sym_id < num_terminals {
                let t_id = TerminalID(sym_id);
                if first_sets[lhs_id].insert(t_id) {
                    worklist.push_back((lhs_id, t_id));
                }
                prefix_nullable = false;
            } else {
                let nt_id = sym_id - num_terminals;
                deps[nt_id].push(lhs_id);
                if !nullable_nts.contains(&nt_id) {
                    prefix_nullable = false;
                }
            }
        }
    }

    while let Some((nt_id, term_id)) = worklist.pop_front() {
        for &dep_lhs in &deps[nt_id] {
            if first_sets[dep_lhs].insert(term_id) {
                worklist.push_back((dep_lhs, term_id));
            }
        }
    }

    first_sets
}

pub fn compute_follow_sets_ids(
    light_productions: &[Vec<usize>],
    lhs_ids: &[usize],
    first_sets: &[BTreeSet<TerminalID>],
    nullable_nts: &HashSet<usize>,
    num_terminals: usize,
    num_nonterminals: usize,
    start_nt_id: usize,
) -> Vec<BTreeSet<Option<TerminalID>>> {
    let mut follow_sets = vec![BTreeSet::new(); num_nonterminals];
    let mut edges: Vec<Vec<usize>> = vec![Vec::new(); num_nonterminals];

    follow_sets[start_nt_id].insert(None);

    for (prod_idx, rhs) in light_productions.iter().enumerate() {
        let lhs_id = lhs_ids[prod_idx];

        for i in 0..rhs.len() {
            let sym_id = rhs[i];
            if sym_id >= num_terminals {
                let b_nt = sym_id - num_terminals;
                let mut suffix_nullable = true;

                for &next_sym in &rhs[i + 1..] {
                    if next_sym < num_terminals {
                        follow_sets[b_nt].insert(Some(TerminalID(next_sym)));
                        suffix_nullable = false;
                        break;
                    } else {
                        let next_nt = next_sym - num_terminals;
                        for &t in &first_sets[next_nt] {
                            follow_sets[b_nt].insert(Some(t));
                        }
                        if !nullable_nts.contains(&next_nt) {
                            suffix_nullable = false;
                            break;
                        }
                    }
                }

                if suffix_nullable {
                    edges[lhs_id].push(b_nt);
                }
            }
        }
    }

    let mut worklist: VecDeque<usize> = (0..num_nonterminals).collect();
    let mut in_queue: Vec<bool> = vec![true; num_nonterminals];

    while let Some(u) = worklist.pop_front() {
        in_queue[u] = false;
        let src_set: Vec<Option<TerminalID>> = follow_sets[u].iter().cloned().collect();

        for &v in &edges[u] {
            let mut changed = false;
            for &t in &src_set {
                if follow_sets[v].insert(t) {
                    changed = true;
                }
            }
            if changed && !in_queue[v] {
                in_queue[v] = true;
                worklist.push_back(v);
            }
        }
    }

    follow_sets
}
