use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::grammar::{CompactProduction, CompactSymbol, NonTerminalID, TerminalID};
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
                    // This non-terminal already counts as "able to derive a string"
                    // if it can derive ε. Otherwise we wait until it becomes
                    // terminal-productive.
                    if !can_derive_epsilon.contains(nt) {
                        prod_unsatisfied[idx] += 1;
                    }
                }
            }
        }
    }

    let mut can_derive_terminal = BTreeSet::new();
    let mut queue_term: VecDeque<NonTerminal> = VecDeque::new();

    // Seed with productions whose RHS already contains a terminal and whose
    // remaining non-terminals are known to derive *some* string.
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
                // If this occurrence was counted as "unsatisfied" (i.e. the
                // non-terminal was not known nullable), decrement the counter.
                if !can_derive_epsilon.contains(&nt) && prod_unsatisfied[p_idx] > 0 {
                    prod_unsatisfied[p_idx] -= 1;
                }

                // The production now definitely has a source of terminals:
                // either a direct terminal or this productive non-terminal.
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
    let start = std::time::Instant::now();
    let res = compute_nonterminal_nullability(productions)
        .into_iter()
        .filter_map(|(nt, status)| {
            (status == Nullability::Nullable || status == Nullability::Null).then_some(nt)
        })
        .collect();
    crate::debug!(3, "Computed nullable non-terminals in {:.2?}", start.elapsed());
    res
}

pub fn compute_first_sets_for_nonterminals(
    productions: &[Production],
    nullable_nonterminals: &BTreeSet<NonTerminal>,
) -> BTreeMap<NonTerminal, BTreeSet<Terminal>> {
    crate::debug!(3, "Computing first sets for non-terminals");
    let start = std::time::Instant::now();
    use std::iter;
    use bimap::BiBTreeMap;
    use std::collections::HashSet;

    // 1. Assign integer IDs to non-terminals for performance.
    let mut nt_map = BiBTreeMap::new();
    let mut all_nts = Vec::new();
    for p in productions {
        for nt in iter::once(&p.lhs).chain(p.rhs.iter().filter_map(|s| match s {
                Symbol::NonTerminal(nt) => Some(nt),
                _ => None,
            }))
        {
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
    crate::debug!(3, "Computed first sets in {:.2?}", start.elapsed());
    res
}

pub fn compute_follow_sets_for_nonterminals(
    productions: &[Production],
    first_sets: &BTreeMap<NonTerminal, BTreeSet<Terminal>>,
    nullable_nonterminals: &BTreeSet<NonTerminal>,
) -> BTreeMap<NonTerminal, BTreeSet<Option<Terminal>>> {
    crate::debug!(3, "Computing follow sets for non-terminals");
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

    crate::debug!(3, "Computed follow sets in {:.2?}", start.elapsed());
    follow_sets
}

fn strongconnect(
    v: usize,
    index: &mut usize,
    stack: &mut Vec<usize>,
    indices: &mut Vec<Option<usize>>,
    lowlinks: &mut Vec<usize>,
    on_stack: &mut Vec<bool>,
    sccs: &mut Vec<Vec<usize>>,
    dependencies: &[Vec<NonTerminalID>],
) {
    indices[v] = Some(*index);
    lowlinks[v] = *index;
    *index += 1;
    stack.push(v);
    on_stack[v] = true;

    for w in &dependencies[v] {
        let w_idx = w.0;
        if indices[w_idx].is_none() {
            strongconnect(w_idx, index, stack, indices, lowlinks, on_stack, sccs, dependencies);
            lowlinks[v] = lowlinks[v].min(lowlinks[w_idx]);
        } else if on_stack[w_idx] {
            lowlinks[v] = lowlinks[v].min(indices[w_idx].unwrap());
        }
    }

    if lowlinks[v] == indices[v].unwrap() {
        let mut scc = Vec::new();
        loop {
            let w = stack.pop().unwrap();
            on_stack[w] = false;
            scc.push(w);
            if w == v { break; }
        }
        sccs.push(scc);
    }
}

pub fn precompute_closures(
    productions: &[CompactProduction],
    num_non_terminals: usize,
) -> Vec<BTreeSet<Item>> {
    let mut closures: Vec<BTreeSet<Item>> = vec![BTreeSet::new(); num_non_terminals];
    let mut direct_items: Vec<Vec<Item>> = vec![Vec::new(); num_non_terminals];
    let mut dependencies: Vec<Vec<NonTerminalID>> = vec![Vec::new(); num_non_terminals];

    for (prod_idx, prod) in productions.iter().enumerate() {
        let item = Item { production_id: prod_idx, dot_position: 0 };
        direct_items[prod.lhs.0].push(item);
        if let Some(CompactSymbol::NonTerminal(nt)) = prod.rhs.get(0) {
            dependencies[prod.lhs.0].push(*nt);
        }
    }

    let mut index = 0;
    let mut stack = Vec::new();
    let mut indices = vec![None; num_non_terminals];
    let mut lowlinks = vec![0; num_non_terminals];
    let mut on_stack = vec![false; num_non_terminals];
    let mut sccs = Vec::new();

    for i in 0..num_non_terminals {
        if indices[i].is_none() {
            strongconnect(
                i, &mut index, &mut stack, &mut indices, &mut lowlinks, &mut on_stack, &mut sccs, &dependencies
            );
        }
    }

    for scc in sccs {
        let mut scc_items = BTreeSet::new();
        for &nt_idx in &scc {
            for item in &direct_items[nt_idx] {
                scc_items.insert(*item);
            }
        }
        
        for &nt_idx in &scc {
            for &dep_nt in &dependencies[nt_idx] {
                if !scc.contains(&dep_nt.0) {
                    let dep_items = &closures[dep_nt.0];
                    scc_items.extend(dep_items.iter().cloned());
                }
            }
        }
        
        for &nt_idx in &scc {
            closures[nt_idx] = scc_items.clone();
        }
    }
    
    closures
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

#[time_it]
pub fn compute_closure_compact(
    items: &BTreeSet<Item>,
    precomputed_closures: &[BTreeSet<Item>],
    productions: &[CompactProduction],
) -> BTreeSet<Item> {
    let mut closure = items.clone();
    for item in items {
        let prod = &productions[item.production_id];
        if let Some(CompactSymbol::NonTerminal(nt_id)) = prod.rhs.get(item.dot_position) {
            closure.extend(precomputed_closures[nt_id.0].iter().cloned());
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

pub fn compute_goto_compact(items: &BTreeSet<Item>, productions: &[CompactProduction]) -> BTreeSet<Item> {
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

pub fn split_on_dot_compact(
    items: &BTreeSet<Item>,
    productions: &[CompactProduction],
) -> BTreeMap<Option<CompactSymbol>, BTreeSet<Item>> {
    let mut result: BTreeMap<Option<CompactSymbol>, BTreeSet<Item>> = BTreeMap::new();
    for item in items {
        let prod = &productions[item.production_id];
        let key = prod.rhs.get(item.dot_position).cloned();
        result.entry(key).or_default().insert(item.clone());
    }
    result
}

pub fn compute_nullable_nonterminals_compact(
    productions: &[CompactProduction],
    num_non_terminals: usize,
) -> Vec<bool> {
    let mut nullable = vec![false; num_non_terminals];
    let mut changed = true;
    while changed {
        changed = false;
        for p in productions {
            if !nullable[p.lhs.0] {
                let all_rhs_nullable = p.rhs.iter().all(|s| match s {
                    CompactSymbol::Terminal(_) => false,
                    CompactSymbol::NonTerminal(nt) => nullable[nt.0],
                });
                if all_rhs_nullable {
                    nullable[p.lhs.0] = true;
                    changed = true;
                }
            }
        }
    }
    nullable
}

pub fn compute_first_sets_compact(
    productions: &[CompactProduction],
    nullable: &[bool],
    num_non_terminals: usize,
) -> Vec<BTreeSet<TerminalID>> {
    let mut first_sets = vec![BTreeSet::new(); num_non_terminals];
    let mut changed = true;
    while changed {
        changed = false;
        for p in productions {
            let lhs_id = p.lhs.0;
            let mut rhs_nullable = true;
            for s in &p.rhs {
                if !rhs_nullable {
                    break;
                }
                match s {
                    CompactSymbol::Terminal(t) => {
                        if first_sets[lhs_id].insert(*t) {
                            changed = true;
                        }
                        rhs_nullable = false;
                    }
                    CompactSymbol::NonTerminal(nt) => {
                        let nt_id = nt.0;
                        // We can't borrow first_sets[lhs_id] mutably and first_sets[nt_id] immutably at same time easily in loop
                        // So we clone the set to add
                        let to_add: Vec<TerminalID> = first_sets[nt_id].iter().cloned().collect();
                        for t in to_add {
                            if first_sets[lhs_id].insert(t) {
                                changed = true;
                            }
                        }
                        if !nullable[nt_id] {
                            rhs_nullable = false;
                        }
                    }
                }
            }
        }
    }
    first_sets
}

pub fn compute_follow_sets_compact(
    productions: &[CompactProduction],
    first_sets: &[BTreeSet<TerminalID>],
    nullable: &[bool],
    num_non_terminals: usize,
) -> Vec<BTreeSet<Option<TerminalID>>> {
    let mut follow_sets = vec![BTreeSet::new(); num_non_terminals];
    if productions.is_empty() {
        return follow_sets;
    }
    // Rule 1: Start symbol gets EOF (None)
    follow_sets[productions[0].lhs.0].insert(None);

    let mut changed = true;
    while changed {
        changed = false;
        for p in productions {
            let lhs_id = p.lhs.0;
            for i in 0..p.rhs.len() {
                if let CompactSymbol::NonTerminal(b_nt) = p.rhs[i] {
                    let b_id = b_nt.0;
                    let mut suffix_nullable = true;
                    let mut suffix_first = BTreeSet::new();

                    for j in (i + 1)..p.rhs.len() {
                        match &p.rhs[j] {
                            CompactSymbol::Terminal(t) => {
                                suffix_first.insert(*t);
                                suffix_nullable = false;
                                break;
                            }
                            CompactSymbol::NonTerminal(g_nt) => {
                                suffix_first.extend(first_sets[g_nt.0].iter().cloned());
                                if !nullable[g_nt.0] {
                                    suffix_nullable = false;
                                    break;
                                }
                            }
                        }
                    }

                    let old_len = follow_sets[b_id].len();
                    for t in suffix_first {
                        follow_sets[b_id].insert(Some(t));
                    }
                    if suffix_nullable {
                        let lhs_follow: Vec<_> = follow_sets[lhs_id].iter().cloned().collect();
                        follow_sets[b_id].extend(lhs_follow);
                    }
                    if follow_sets[b_id].len() != old_len {
                        changed = true;
                    }
                }
            }
        }
    }
    follow_sets
}
