fn max_nt_id(rules: &[Rule]) -> u32 {
    rules
        .iter()
        .flat_map(|rule| {
            std::iter::once(rule.lhs).chain(rule.rhs.iter().filter_map(|symbol| match symbol {
                Symbol::Nonterminal(nonterminal) => Some(*nonterminal),
                Symbol::Terminal(_) => None,
            }))
        })
        .max()
        .unwrap_or(0)
}

fn add_boundary_nonterminals<'a>(
    symbols: impl Iterator<Item = &'a Symbol>,
    nullable: &BTreeSet<NonterminalID>,
    targets: &mut BTreeSet<NonterminalID>,
) {
    for symbol in symbols {
        match symbol {
            Symbol::Nonterminal(nonterminal) => {
                targets.insert(*nonterminal);
                if !nullable.contains(nonterminal) {
                    break;
                }
            }
            Symbol::Terminal(_) => break,
        }
    }
}

fn build_right_reachability_graph(
    rules: &[Rule],
    nullable: &BTreeSet<NonterminalID>,
) -> BTreeMap<NonterminalID, BTreeSet<NonterminalID>> {
    let mut graph = BTreeMap::<NonterminalID, BTreeSet<NonterminalID>>::new();
    for rule in rules {
        add_boundary_nonterminals(
            rule.rhs.iter().rev(),
            nullable,
            graph.entry(rule.lhs).or_default(),
        );
    }
    graph
}

fn find_indirect_rr_cycle(
    graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
) -> Option<Vec<NonterminalID>> {
    find_cycle(graph, 1, false)
}

fn find_cycle(
    graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
    min_cycle_len: usize,
    skip_self_loops: bool,
) -> Option<Vec<NonterminalID>> {
    fn dfs(
        node: NonterminalID,
        graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
        colors: &mut BTreeMap<NonterminalID, u8>,
        stack: &mut Vec<NonterminalID>,
        min_cycle_len: usize,
        skip_self_loops: bool,
    ) -> Option<Vec<NonterminalID>> {
        colors.insert(node, 1);
        stack.push(node);
        for &next in graph.get(&node).into_iter().flatten() {
            if skip_self_loops && next == node {
                continue;
            }

            match colors.get(&next).copied().unwrap_or(0) {
                0 => {
                    if let Some(cycle) = dfs(
                        next,
                        graph,
                        colors,
                        stack,
                        min_cycle_len,
                        skip_self_loops,
                    ) {
                        return Some(cycle);
                    }
                }
                1 => {
                    if let Some(start) = stack.iter().position(|&entry| entry == next) {
                        let cycle = stack[start..].to_vec();
                        if cycle.len() >= min_cycle_len {
                            return Some(cycle);
                        }
                    }
                }
                _ => {}
            }
        }
        stack.pop();
        colors.insert(node, 2);
        None
    }

    let mut colors = BTreeMap::new();
    let mut stack = Vec::new();
    for &node in graph.keys() {
        if colors.get(&node).copied().unwrap_or(0) == 0 {
            if let Some(cycle) = dfs(
                node,
                graph,
                &mut colors,
                &mut stack,
                min_cycle_len,
                skip_self_loops,
            ) {
                return Some(cycle);
            }
        }
    }
    None
}

/// Build a graph where an edge A → B means B appears at the left edge of
/// a production for A (possibly after nullable symbols).
fn build_left_reachability_graph(
    rules: &[Rule],
    nullable: &BTreeSet<NonterminalID>,
) -> BTreeMap<NonterminalID, BTreeSet<NonterminalID>> {
    let mut graph = BTreeMap::<NonterminalID, BTreeSet<NonterminalID>>::new();
    for rule in rules {
        add_boundary_nonterminals(rule.rhs.iter(), nullable, graph.entry(rule.lhs).or_default());
    }
    graph
}

/// Find an indirect left-recursive cycle (length ≥ 2) in the left-reachability
/// graph.  Direct self-loops (A → A …) are excluded — they are fine for GLR.
fn find_indirect_lr_cycle(
    graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
) -> Option<Vec<NonterminalID>> {
    find_cycle(graph, 2, false)
}

fn find_nontrivial_sccs(
    graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
) -> Vec<BTreeSet<NonterminalID>> {
    let mut nodes: Vec<NonterminalID> = graph.keys().copied().collect();
    for neighbors in graph.values() {
        nodes.extend(neighbors.iter().copied());
    }
    nodes.sort_unstable();
    nodes.dedup();

    if nodes.is_empty() {
        return Vec::new();
    }

    let node_to_idx: HashMap<NonterminalID, usize> = nodes
        .iter()
        .enumerate()
        .map(|(index, &node)| (node, index))
        .collect();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    let mut reverse_adjacency: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];

    for (&node, neighbors) in graph {
        let from_index = node_to_idx[&node];
        for &neighbor in neighbors {
            let to_index = node_to_idx[&neighbor];
            adjacency[from_index].push(to_index);
            reverse_adjacency[to_index].push(from_index);
        }
    }

    let mut visited = vec![false; nodes.len()];
    let mut finish_order = Vec::with_capacity(nodes.len());
    for start in 0..nodes.len() {
        if visited[start] {
            continue;
        }
        let mut stack = vec![(start, 0usize)];
        visited[start] = true;
        while let Some((node_index, neighbor_index)) = stack.last_mut() {
            if *neighbor_index < adjacency[*node_index].len() {
                let next = adjacency[*node_index][*neighbor_index];
                *neighbor_index += 1;
                if !visited[next] {
                    visited[next] = true;
                    stack.push((next, 0));
                }
            } else {
                finish_order.push(*node_index);
                stack.pop();
            }
        }
    }

    visited.fill(false);
    let mut sccs = Vec::new();
    for &start in finish_order.iter().rev() {
        if visited[start] {
            continue;
        }
        let mut component_indices = Vec::new();
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(node_index) = stack.pop() {
            component_indices.push(node_index);
            for &next in &reverse_adjacency[node_index] {
                if !visited[next] {
                    visited[next] = true;
                    stack.push(next);
                }
            }
        }

        if component_indices.len() >= 2 {
            let component: BTreeSet<NonterminalID> = component_indices
                .into_iter()
                .map(|node_index| nodes[node_index])
                .collect();
            sccs.push(component);
        }
    }

    sccs
}

/// Find a cycle of length ≥ 2 in the graph (self-loops are skipped).
/// Used by `eliminate_right_recursion` to find indirect cycles.
fn find_cycle_excluding_self_loops(
    graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
) -> Option<Vec<NonterminalID>> {
    find_cycle(graph, 2, true)
}

/// Inline right-end: for rules `from_nt → α to_nt β` where β is all-nullable,
/// replace the `to_nt` occurrence with each of `to_nt`'s alternative RHSs.
///
/// This breaks indirect right-recursive cycles by removing the edge
/// `from_nt → to_nt` in the right-reachability graph.
fn inline_right_end(
    rules: &mut Vec<Rule>,
    from_nt: NonterminalID,
    to_nt: NonterminalID,
    nullable: &BTreeSet<NonterminalID>,
) {
    let to_rhss: Vec<Vec<Symbol>> = rules
        .iter()
        .filter(|r| r.lhs == to_nt)
        .map(|r| r.rhs.clone())
        .collect();
    if to_rhss.is_empty() {
        return;
    }

    let mut new_rules = Vec::new();
    for rule in rules.iter() {
        if rule.lhs != from_nt {
            new_rules.push(rule.clone());
            continue;
        }
        let pos = find_right_end_position(&rule.rhs, to_nt, nullable);
        if let Some(pos) = pos {
            for to_rhs in &to_rhss {
                let mut rhs = rule.rhs[..pos].to_vec();
                rhs.extend(to_rhs.iter().cloned());
                rhs.extend(rule.rhs[pos + 1..].iter().cloned());
                new_rules.push(Rule { lhs: from_nt, rhs });
            }
        } else {
            new_rules.push(rule.clone());
        }
    }
    *rules = new_rules;
}

/// Find the rightmost position of `target_nt` in `rhs` such that everything
/// after it is a nullable nonterminal.  Returns `None` if no such position.
fn find_right_end_position(
    rhs: &[Symbol],
    target_nt: NonterminalID,
    nullable: &BTreeSet<NonterminalID>,
) -> Option<usize> {
    for i in (0..rhs.len()).rev() {
        match &rhs[i] {
            Symbol::Nonterminal(nt) if *nt == target_nt => return Some(i),
            Symbol::Nonterminal(nt) if nullable.contains(nt) => continue,
            _ => return None,
        }
    }
    None
}

fn is_direct_right_recursive(rule: &Rule) -> bool {
    matches!(rule.rhs.last(), Some(Symbol::Nonterminal(nonterminal)) if *nonterminal == rule.lhs)
}

/// Resolve direct right recursion for a single nonterminal.
///
/// Given recursive rules `A → α A` and base rules `A → β`, transform to:
/// - Base rules (unchanged): `A → β`
/// - Composed rules: `A → new_nt β` (for each base rule)
/// - Tail rules: `new_nt → α` (body of each recursive rule, without trailing A)
/// - Left-recursive tails: `new_nt → new_nt α`
///
/// Note: if α is empty (rule `A → A`), this produces `new_nt → ε`.
/// The subsequent ε-elimination pass handles that.
fn resolve_direct_rr_single_nt(
    rules: &mut Vec<Rule>,
    nt: NonterminalID,
    new_nt: NonterminalID,
) {
    let (recursive, non_recursive): (Vec<Rule>, Vec<Rule>) = rules
        .iter()
        .filter(|r| r.lhs == nt)
        .cloned()
        .partition(|r| is_direct_right_recursive(r));

    if recursive.is_empty() {
        return;
    }

    // Keep all rules NOT for this NT
    let mut new_rules: Vec<Rule> = rules.iter().filter(|r| r.lhs != nt).cloned().collect();

    // Keep base rules: A → β
    new_rules.extend(non_recursive.iter().cloned());

    // Add A → new_nt β for each base rule
    for base in &non_recursive {
        let mut rhs = vec![Symbol::Nonterminal(new_nt)];
        rhs.extend(base.rhs.iter().cloned());
        new_rules.push(Rule { lhs: nt, rhs });
    }

    // Add new_nt → α (body without trailing A) for each recursive rule
    for rec in &recursive {
        let body = rec.rhs[..rec.rhs.len() - 1].to_vec();
        new_rules.push(Rule { lhs: new_nt, rhs: body });
    }

    // Add new_nt → new_nt α (left-recursive) for each recursive rule
    for rec in &recursive {
        let body = &rec.rhs[..rec.rhs.len() - 1];
        let mut rhs = vec![Symbol::Nonterminal(new_nt)];
        rhs.extend(body.iter().cloned());
        new_rules.push(Rule { lhs: new_nt, rhs });
    }

    *rules = new_rules;
}

/// Resolve direct right recursion for multiple nonterminals in a single pass.
///
/// Each entry maps a right-recursive NT to its fresh replacement NT.
/// Equivalent to calling `resolve_direct_rr_single_nt` for each NT independently,
/// but avoids the O(NTs × rules) cost of repeated full-vector rebuilds.
fn resolve_direct_rr_batched(
    rules: &mut Vec<Rule>,
    rr_map: &BTreeMap<NonterminalID, NonterminalID>,
) {
    // Partition rules by whether they belong to a right-recursive NT.
    let mut recursive_by_nt: BTreeMap<NonterminalID, Vec<Rule>> = BTreeMap::new();
    let mut non_recursive_by_nt: BTreeMap<NonterminalID, Vec<Rule>> = BTreeMap::new();
    let mut new_rules = Vec::with_capacity(rules.len() * 2);

    for rule in rules.iter() {
        if rr_map.contains_key(&rule.lhs) {
            if is_direct_right_recursive(rule) {
                recursive_by_nt.entry(rule.lhs).or_default().push(rule.clone());
            } else {
                non_recursive_by_nt.entry(rule.lhs).or_default().push(rule.clone());
            }
        } else {
            new_rules.push(rule.clone());
        }
    }

    for (&nt, &new_nt) in rr_map {
        let rec_rules = recursive_by_nt.remove(&nt).unwrap_or_default();
        let base_rules = non_recursive_by_nt.remove(&nt).unwrap_or_default();

        if rec_rules.is_empty() {
            new_rules.extend(base_rules);
            continue;
        }

        // Keep base rules: A → β
        new_rules.extend(base_rules.iter().cloned());

        // Add A → new_nt β for each base rule
        for base in &base_rules {
            let mut rhs = vec![Symbol::Nonterminal(new_nt)];
            rhs.extend(base.rhs.iter().cloned());
            new_rules.push(Rule { lhs: nt, rhs });
        }

        // Add new_nt → α (body without trailing A) for each recursive rule
        for rec in &rec_rules {
            let body = rec.rhs[..rec.rhs.len() - 1].to_vec();
            new_rules.push(Rule { lhs: new_nt, rhs: body });
        }

        // Add new_nt → new_nt α (left-recursive) for each recursive rule
        for rec in &rec_rules {
            let body = &rec.rhs[..rec.rhs.len() - 1];
            let mut rhs = vec![Symbol::Nonterminal(new_nt)];
            rhs.extend(body.iter().cloned());
            new_rules.push(Rule { lhs: new_nt, rhs });
        }
    }

    *rules = new_rules;
}
