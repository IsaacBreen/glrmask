pub struct AnalyzedGrammar {
    pub rules: Vec<Rule>,
    pub num_terminals: u32,
    pub terminal_display_names: Vec<String>,
    pub num_nonterminals: u32,
    pub nonterminal_display_names: Vec<String>,
    pub nullable: BTreeSet<NonterminalID>,
    pub first: Vec<BitSet>,
    pub follow: Vec<BitSet>,
    /// Index: nonterminal -> list of rule indices with that nonterminal as LHS.
    pub rules_by_lhs: Vec<Vec<u32>>,
}

impl AnalyzedGrammar {
    pub fn from_grammar_def(g: &GrammarDef) -> Self {
        let num_terminals = g.num_terminals();
        let mut rules = Vec::with_capacity(g.rules.len() + 1);
        let augmented_start = g.num_nonterminals();
        rules.push(Rule {
            lhs: augmented_start,
            rhs: vec![Symbol::Nonterminal(g.start)],
        });
        rules.extend(g.rules.iter().cloned());

        let num_nonterminals = augmented_start + 1;
        let nullable = compute_nullable(&rules, num_nonterminals);
        let first = compute_first(&rules, num_nonterminals, num_terminals, &nullable);
        let follow = compute_follow(
            &rules,
            num_nonterminals,
            num_terminals,
            augmented_start,
            &first,
            &nullable,
        );

        let mut rules_by_lhs = vec![Vec::new(); num_nonterminals as usize];
        for (i, r) in rules.iter().enumerate() {
            if (r.lhs as usize) < rules_by_lhs.len() {
                rules_by_lhs[r.lhs as usize].push(i as u32);
            }
        }

        Self {
            rules,
            num_terminals,
            terminal_display_names: (0..num_terminals)
                .map(|terminal| g.terminal_display_name(terminal))
                .collect(),
            num_nonterminals,
            nonterminal_display_names: (0..num_nonterminals)
                .map(|nonterminal| {
                    if nonterminal == augmented_start {
                        "<augmented-start>".to_string()
                    } else {
                        g.nonterminal_names
                            .get(&nonterminal)
                            .cloned()
                            .unwrap_or_else(|| format!("N{nonterminal}"))
                    }
                })
                .collect(),
            nullable,
            first,
            follow,
            rules_by_lhs,
        }
    }

    pub fn terminal_display_name(&self, terminal: TerminalID) -> &str {
        self.terminal_display_names
            .get(terminal as usize)
            .map(String::as_str)
            .unwrap_or("<unknown-terminal>")
    }

    /// Assert the pre-table-build grammar is in the normal form required by
    /// GLR table construction and downstream characterization.
    pub fn check_table_build_normal_form(&self) -> Result<(), String> {
        let mut violations: Vec<String> = Vec::new();
        if let Err(msg) = self.check_no_nullable_nonterminals() {
            violations.push(msg);
        }
        if let Err(msg) = self.check_no_reachable_zero_length_productions() {
            violations.push(msg);
        }
        if let Err(msg) = self.check_recursion_boundedness() {
            violations.push(msg);
        }
        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations.join("\n"))
        }
    }

    pub fn debug_check_grammar_preconditions(&self) -> Result<(), String> {
        self.check_table_build_normal_form()
    }

    pub fn check_no_nullable_nonterminals(&self) -> Result<(), String> {
        let reachable = self.reachable_nonterminals();
        let synthetic_start = self.num_nonterminals.saturating_sub(1);
        if !self.nullable.is_empty() {
            let ids: Vec<u32> = self
                .nullable
                .iter()
                .filter(|&&nt| nt != synthetic_start && reachable.contains(&nt))
                .copied()
                .collect();
            if !ids.is_empty() {
                return Err(format!(
                    "nullable nonterminals reachable at the table-build boundary: {:?}. \
                     Rules with epsilon-productions or all-nullable RHS create \
                     reduce chains that the characterisation stage cannot \
                     handle when combined with recursion.",
                    ids,
                ));
            }
        }
        Ok(())
    }

    pub fn check_no_reachable_zero_length_productions(&self) -> Result<(), String> {
        let reachable = self.reachable_nonterminals();
        let zero_len_rules: Vec<String> = self
            .rules
            .iter()
            .enumerate()
            .filter(|(_, rule)| reachable.contains(&rule.lhs) && rule.rhs.is_empty())
            .map(|(index, rule)| format!("rule#{index}: lhs=N{}", rule.lhs))
            .collect();

        if zero_len_rules.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "zero-length productions reachable at the table-build boundary: {}",
                zero_len_rules.join(", ")
            ))
        }
    }

    pub fn check_recursion_boundedness(&self) -> Result<(), String> {
        let mut violations: Vec<String> = Vec::new();
        let reachable = self.reachable_nonterminals();

        let rr_graph = filter_graph_to_reachable(
            build_right_reachability_graph(&self.rules, &self.nullable),
            &reachable,
        );
        if let Some(cycle) = find_indirect_rr_cycle(&rr_graph) {
            violations.push(format!(
                "right-recursive cycle detected: {:?}. \
                 Right recursion causes unbounded reduce chains in \
                 terminal characterisation. Convert to left recursion \
                 or inline the cycle.",
                cycle,
            ));
        }

        let lr_graph = filter_graph_to_reachable(
            build_left_reachability_graph(&self.rules, &self.nullable),
            &reachable,
        );
        if let Some(cycle) = find_indirect_lr_cycle(&lr_graph) {
            if cycle.len() >= 2 {
                violations.push(format!(
                    "indirect left-recursive cycle detected: {:?}. \
                     Indirect left recursion may create unbounded GSS \
                     growth. Inline or rewrite the cycle.",
                    cycle,
                ));
            }
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations.join("\n"))
        }
    }

    fn reachable_nonterminals(&self) -> BTreeSet<NonterminalID> {
        let synthetic_start = self.num_nonterminals.saturating_sub(1);
        let mut reachable = BTreeSet::from([synthetic_start]);
        let mut queue = VecDeque::from([synthetic_start]);

        while let Some(nonterminal) = queue.pop_front() {
            for &rule_index in self.rules_by_lhs.get(nonterminal as usize).into_iter().flatten() {
                let rule = &self.rules[rule_index as usize];
                for next_nonterminal in rule.rhs.iter().filter_map(|symbol| match symbol {
                    Symbol::Nonterminal(nonterminal) => Some(*nonterminal),
                    Symbol::Terminal(_) => None,
                }) {
                    if reachable.insert(next_nonterminal) {
                        queue.push_back(next_nonterminal);
                    }
                }
            }
        }

        reachable
    }
}

/// Eliminate right recursion by first inlining indirect cycles and then
/// rewriting direct right recursion into left recursion.
pub(crate) fn eliminate_right_recursion(
    rules: &mut Vec<Rule>,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
) {
    // Resolve indirect right recursion by inlining right ends.
    const MAX_INDIRECT_ROUNDS: usize = 200;
    for _ in 0..MAX_INDIRECT_ROUNDS {
        let num_nt = max_nt_id(rules) + 1;
        let nullable = compute_nullable(rules, num_nt);
        let graph = build_right_reachability_graph(rules, &nullable);
        match find_cycle_excluding_self_loops(&graph) {
            Some(cycle) => {
                let from = cycle[0];
                let to = cycle[1 % cycle.len()];
                inline_right_end(rules, from, to, &nullable);
            }
            None => break,
        }
    }

    // Resolve direct right recursion for all nonterminals in a single pass.
    let rr_nts: BTreeMap<NonterminalID, NonterminalID> = rules
        .iter()
        .filter(|r| is_direct_right_recursive(r))
        .map(|r| r.lhs)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|nt| (nt, fresh_nt()))
        .collect();

    if !rr_nts.is_empty() {
        resolve_direct_rr_batched(rules, &rr_nts);
    }
}
