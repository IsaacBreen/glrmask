fn eliminate_hidden_left_recursion(
    rules: &mut Vec<Rule>,
    nullable: &BTreeSet<NonterminalID>,
    normalize_iteration: usize,
) {
    let profile_enabled = compile_profile_enabled();

    loop {
        let rules_len = rules.len();
        let lr_graph_started_at = profile_enabled.then(Instant::now);
        let lr_graph = build_left_reachability_graph(rules, nullable);
        let build_left_reachability_graph_ms = lr_graph_started_at
            .map(elapsed_ms)
            .unwrap_or(0.0);
        let left_reachability_node_count = lr_graph.len();
        let left_reachability_edge_count: usize =
            lr_graph.values().map(BTreeSet::len).sum();
        let cycle_sccs = find_nontrivial_sccs(&lr_graph);
        if cycle_sccs.is_empty() && cfg!(debug_assertions) {
            debug_assert!(find_indirect_lr_cycle(&lr_graph).is_none());
        }
        if cycle_sccs.is_empty() {
            if profile_enabled {
                emit_normalize_profile(
                    "hidden_left_recursion_helper_counters",
                    Some(normalize_iteration),
                    build_left_reachability_graph_ms,
                    rules_len,
                    rules_len,
                    &format!(
                        " rules_len={} build_rhs_by_lhs_ms=0.000 rhs_lhs_bucket_count=0 rhs_total_entries=0 build_left_reachability_graph_ms={:.3} left_reachability_node_count={} left_reachability_edge_count={}",
                        rules_len,
                        build_left_reachability_graph_ms,
                        left_reachability_node_count,
                        left_reachability_edge_count,
                    ),
                );
            }
            return;
        }
        let rhs_by_lhs_started_at = profile_enabled.then(Instant::now);
        let rhs_by_lhs = build_rhs_by_lhs(rules);
        let build_rhs_by_lhs_ms = rhs_by_lhs_started_at.map(elapsed_ms).unwrap_or(0.0);
        let rhs_lhs_bucket_count = rhs_by_lhs.len();
        let rhs_total_entries: usize = rhs_by_lhs.values().map(BTreeSet::len).sum();

        if profile_enabled {
            emit_normalize_profile(
                "hidden_left_recursion_helper_counters",
                Some(normalize_iteration),
                build_left_reachability_graph_ms + build_rhs_by_lhs_ms,
                rules_len,
                rules_len,
                &format!(
                    " rules_len={} build_rhs_by_lhs_ms={:.3} rhs_lhs_bucket_count={} rhs_total_entries={} build_left_reachability_graph_ms={:.3} left_reachability_node_count={} left_reachability_edge_count={}",
                    rules_len,
                    build_rhs_by_lhs_ms,
                    rhs_lhs_bucket_count,
                    rhs_total_entries,
                    build_left_reachability_graph_ms,
                    left_reachability_node_count,
                    left_reachability_edge_count,
                ),
            );
        }

        let mut additions = Vec::new();
        let mut replaced_cycle_rules = HashSet::new();
        let cycle_scc_by_nt: HashMap<NonterminalID, usize> = cycle_sccs
            .iter()
            .enumerate()
            .flat_map(|(scc_index, cycle_nodes)| {
                cycle_nodes
                    .iter()
                    .copied()
                    .map(move |nonterminal| (nonterminal, scc_index))
            })
            .collect();
        for rule in rules.iter() {
            let Some(&scc_index) = cycle_scc_by_nt.get(&rule.lhs) else {
                continue;
            };
            let cycle_nodes = &cycle_sccs[scc_index];

            let prefix_end = nullable_prefix_len(&rule.rhs, nullable);
            // For each skip length, if next symbol is a cycle member, add a shortened rule.
            for skip in 1..=prefix_end {
                let suffix = &rule.rhs[skip..];
                if let Some(Symbol::Nonterminal(nt)) = suffix.first() {
                    if cycle_nodes.contains(nt) {
                        additions.push(Rule {
                            lhs: rule.lhs,
                            rhs: suffix.to_vec(),
                        });
                    }
                }
            }

            let first_after_nullable = rule.rhs.get(prefix_end);
            if let Some(Symbol::Nonterminal(next_nt)) = first_after_nullable {
                if *next_nt != rule.lhs && cycle_nodes.contains(next_nt) {
                    let expansions =
                        expand_cycle_head_paths(&rhs_by_lhs, &cycle_nodes, *next_nt, rule.lhs);
                    let mut produced_replacement = false;
                    for expansion in expansions {
                        let mut rhs = expansion;
                        rhs.extend_from_slice(&rule.rhs[prefix_end + 1..]);
                        additions.push(Rule { lhs: rule.lhs, rhs });
                        produced_replacement = true;
                    }
                    if produced_replacement {
                        replaced_cycle_rules.insert(rule.clone());
                    }
                }
            }
        }

        let existing: HashSet<Rule> = rules.iter().cloned().collect();
        let mut unique_additions = HashSet::new();
        additions.retain(|rule| {
            !existing.contains(rule) && unique_additions.insert(rule.clone())
        });

        if !replaced_cycle_rules.is_empty() {
            rules.retain(|rule| !replaced_cycle_rules.contains(rule));
        }

        if additions.is_empty() {
            return;
        }
        rules.extend(additions);
    }
}

fn expand_cycle_head_paths(
    rhs_by_lhs: &BTreeMap<NonterminalID, BTreeSet<Vec<Symbol>>>,
    cycle_nodes: &BTreeSet<NonterminalID>,
    current: NonterminalID,
    goal: NonterminalID,
) -> Vec<Vec<Symbol>> {
    fn expand(
        rhs_by_lhs: &BTreeMap<NonterminalID, BTreeSet<Vec<Symbol>>>,
        cycle_nodes: &BTreeSet<NonterminalID>,
        current: NonterminalID,
        goal: NonterminalID,
        visited: &mut BTreeSet<NonterminalID>,
        out: &mut Vec<Vec<Symbol>>,
    ) {
        if !visited.insert(current) {
            return;
        }

        if let Some(expansions) = rhs_by_lhs.get(&current) {
            for expansion_rhs in expansions {
                match expansion_rhs.first() {
                    Some(Symbol::Nonterminal(head))
                        if cycle_nodes.contains(head) && *head != goal =>
                    {
                        let mut nested = Vec::new();
                        expand(
                            rhs_by_lhs,
                            cycle_nodes,
                            *head,
                            goal,
                            visited,
                            &mut nested,
                        );
                        for mut nested_rhs in nested {
                            nested_rhs.extend_from_slice(&expansion_rhs[1..]);
                            out.push(nested_rhs);
                        }
                    }
                    _ => out.push(expansion_rhs.clone()),
                }
            }
        }

        visited.remove(&current);
    }

    let mut out = Vec::new();
    expand(
        rhs_by_lhs,
        cycle_nodes,
        current,
        goal,
        &mut BTreeSet::new(),
        &mut out,
    );
    out
}

fn nullable_prefix_len(rhs: &[Symbol], nullable: &BTreeSet<NonterminalID>) -> usize {
    rhs.iter()
        .take_while(|symbol| matches!(symbol, Symbol::Nonterminal(nt) if nullable.contains(nt)))
        .count()
}

