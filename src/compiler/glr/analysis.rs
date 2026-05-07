use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

use crate::grammar::flat::{GrammarDef, NonterminalID, Rule, Symbol, TerminalID};

pub const EOF: TerminalID = u32::MAX;

#[derive(Debug, Clone)]
pub struct AnalyzedGrammar {
    pub rules: Vec<Rule>,
    pub num_terminals: u32,
    pub terminal_display_names: Vec<String>,
    pub num_nonterminals: u32,
    pub nullable: BTreeSet<NonterminalID>,
    pub first: Vec<BTreeSet<TerminalID>>,
    pub follow: Vec<BTreeSet<TerminalID>>,
    /// Index: nonterminal -> list of rule indices with that nonterminal as LHS.
    pub rules_by_lhs: Vec<Vec<u32>>,
}

impl AnalyzedGrammar {
    pub fn from_grammar_def(g: &GrammarDef) -> Self {
        let mut rules = Vec::with_capacity(g.rules.len() + 1);
        let augmented_start = g.num_nonterminals();
        rules.push(Rule {
            lhs: augmented_start,
            rhs: vec![Symbol::Nonterminal(g.start)],
        });
        rules.extend(g.rules.iter().cloned());

        let num_nonterminals = augmented_start + 1;
        let nullable = compute_nullable(&rules, num_nonterminals);
        let first = compute_first(&rules, num_nonterminals, &nullable);
        let follow = compute_follow(&rules, num_nonterminals, augmented_start, &first, &nullable);

        let mut rules_by_lhs = vec![Vec::new(); num_nonterminals as usize];
        for (i, r) in rules.iter().enumerate() {
            if (r.lhs as usize) < rules_by_lhs.len() {
                rules_by_lhs[r.lhs as usize].push(i as u32);
            }
        }

        Self {
            rules,
            num_terminals: g.num_terminals(),
            terminal_display_names: (0..g.num_terminals())
                .map(|terminal| g.terminal_display_name(terminal))
                .collect(),
            num_nonterminals,
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

fn inline_null_productions_exhaustive(rules: &[Rule], num_nt: u32) -> Vec<Rule> {
    let nullable = compute_nullable(rules, num_nt);
    if nullable.is_empty() {
        return rules.to_vec();
    }

    let mut seen = HashSet::<Rule>::new();
    let mut out = Vec::new();

    for rule in rules {
        let nullable_positions: Vec<usize> = rule
            .rhs
            .iter()
            .enumerate()
            .filter_map(|(i, sym)| match sym {
                Symbol::Nonterminal(nt) if nullable.contains(nt) => Some(i),
                _ => None,
            })
            .collect();

        let k = nullable_positions.len();
        // Safety guard: refuse power-set expansion beyond 20 nullable positions
        assert!(
            k <= 20,
            "production for NT {} has {} nullable positions; refusing power-set",
            rule.lhs, k,
        );

        for mask in 0u64..(1u64 << k) {
            let new_rhs: Vec<Symbol> = rule
                .rhs
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    match nullable_positions.binary_search(i) {
                        Ok(idx) => mask & (1u64 << idx) != 0, // bit set → keep
                        Err(_) => true,                        // non-nullable → always keep
                    }
                })
                .map(|(_, sym)| sym.clone())
                .collect();

            // Drop ε-rules
            if new_rhs.is_empty() {
                continue;
            }

            let candidate = Rule {
                lhs: rule.lhs,
                rhs: new_rhs,
            };
            if seen.insert(candidate.clone()) {
                out.push(candidate);
            }
        }
    }

    out
}

fn find_nullable_runs(
    rhs: &[Symbol],
    nullable: &BTreeSet<NonterminalID>,
    threshold: usize,
) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut run_start = None;

    for (idx, symbol) in rhs.iter().enumerate() {
        let is_nullable = matches!(symbol, Symbol::Nonterminal(nt) if nullable.contains(nt));
        if is_nullable {
            if run_start.is_none() {
                run_start = Some(idx);
            }
        } else if let Some(start) = run_start.take() {
            let len = idx - start;
            if len > threshold {
                runs.push((start, idx - 1));
            }
        }
    }

    if let Some(start) = run_start {
        let len = rhs.len() - start;
        if len > threshold {
            runs.push((start, rhs.len() - 1));
        }
    }

    runs
}

fn compute_nonempty_productive(rules: &[Rule], num_nt: u32) -> BTreeSet<NonterminalID> {
    let mut productive_any = BTreeSet::new();
    let mut nonempty_productive = BTreeSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        for rule in rules {
            if rule.lhs >= num_nt {
                continue;
            }

            let mut rhs_productive = true;
            let mut rhs_nonempty = false;
            for symbol in &rule.rhs {
                match symbol {
                    Symbol::Terminal(_) => rhs_nonempty = true,
                    Symbol::Nonterminal(nonterminal) => {
                        if !productive_any.contains(nonterminal) {
                            rhs_productive = false;
                            break;
                        }
                        rhs_nonempty |= nonempty_productive.contains(nonterminal);
                    }
                }
            }

            if rhs_productive {
                changed |= productive_any.insert(rule.lhs);
                if rhs_nonempty {
                    changed |= nonempty_productive.insert(rule.lhs);
                }
            }
        }
    }
    nonempty_productive
}

fn compress_nullable_runs_with_optional_tree(rules: &[Rule], num_nt: u32) -> Vec<Rule> {
    let nullable = compute_nullable(rules, num_nt);
    if nullable.is_empty() {
        return rules.to_vec();
    }
    let nonempty_productive = compute_nonempty_productive(rules, num_nt);

    let mut run_count = 0usize;
    let mut max_run_len = 0usize;
    for rule in rules {
        for (start, end) in find_nullable_runs(&rule.rhs, &nullable, 1) {
            run_count += 1;
            max_run_len = max_run_len.max(end - start + 1);
        }
    }
    if run_count == 0 {
        return rules.to_vec();
    }

    let mut by_lhs = BTreeMap::<NonterminalID, Vec<Vec<Symbol>>>::new();
    for rule in rules {
        by_lhs.entry(rule.lhs).or_default().push(rule.rhs.clone());
    }

    let mut next_nt = max_nt_id(rules) + 1;
    let mut fresh_nt = || {
        let id = next_nt;
        next_nt += 1;
        id
    };

    let mut nn_cache = BTreeMap::<NonterminalID, NonterminalID>::new();
    let mut result = Vec::<Rule>::new();
    for rule in rules {
        let runs = find_nullable_runs(&rule.rhs, &nullable, 1);
        if runs.is_empty() {
            result.push(rule.clone());
            continue;
        }

        let mut new_rhs = rule.rhs.clone();
        for &(start, end) in runs.iter().rev() {
            let segment: Vec<Symbol> = new_rhs.drain(start..=end).collect();
            let Some(root_nn) = build_non_nullable_tree(
                &segment,
                2,
                &mut fresh_nt,
                &mut result,
                &nullable,
                &nonempty_productive,
                &by_lhs,
                &mut nn_cache,
            ) else {
                continue;
            };
            let root_opt = fresh_nt();
            result.push(Rule {
                lhs: root_opt,
                rhs: vec![Symbol::Nonterminal(root_nn)],
            });
            result.push(Rule {
                lhs: root_opt,
                rhs: vec![],
            });
            new_rhs.insert(start, Symbol::Nonterminal(root_opt));
        }

        result.push(Rule {
            lhs: rule.lhs,
            rhs: new_rhs,
        });
    }

    dedup_rules(&mut result);
    result
}

fn build_non_nullable_tree(
    segment: &[Symbol],
    k: usize,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
    new_rules: &mut Vec<Rule>,
    nullable: &BTreeSet<NonterminalID>,
    nonempty_productive: &BTreeSet<NonterminalID>,
    by_lhs: &BTreeMap<NonterminalID, Vec<Vec<Symbol>>>,
    nn_cache: &mut BTreeMap<NonterminalID, NonterminalID>,
) -> Option<NonterminalID> {
    let k = k.max(2);
    let n = segment.len();
    if n == 0 {
        return None;
    }

    let nn_segment: Vec<Symbol> = segment
        .iter()
        .filter_map(|symbol| match symbol {
            Symbol::Terminal(terminal) => Some(Symbol::Terminal(*terminal)),
            Symbol::Nonterminal(nonterminal) if !nullable.contains(nonterminal) => {
                Some(Symbol::Nonterminal(*nonterminal))
            }
            Symbol::Nonterminal(nonterminal) if nonempty_productive.contains(nonterminal) => {
                get_or_create_non_nullable_nt(
                    *nonterminal,
                    fresh_nt,
                    new_rules,
                    nullable,
                    nonempty_productive,
                    by_lhs,
                    nn_cache,
                )
                .map(Symbol::Nonterminal)
            }
            Symbol::Nonterminal(_) => None,
        })
        .collect();
    let n = nn_segment.len();
    if n == 0 {
        return None;
    }

    if n <= k {
        let leaf_nt = fresh_nt();
        for mask in 1u64..(1u64 << n) {
            let rhs: Vec<Symbol> = nn_segment
                .iter()
                .enumerate()
                .filter(|(idx, _)| ((mask >> idx) & 1) == 1)
                .map(|(_, symbol)| symbol.clone())
                .collect();
            new_rules.push(Rule { lhs: leaf_nt, rhs });
        }
        return Some(leaf_nt);
    }

    // Keep the default right-heavy decomposition; alternate shapes were only
    // used for internal experiments.
    let (first, rest) = nn_segment.split_at(1);
    let chunks: Vec<&[Symbol]> = if rest.is_empty() {
        vec![first]
    } else {
        vec![first, rest]
    };
    let chunk_nts: Vec<NonterminalID> = chunks
        .into_iter()
        .map(|chunk| {
            build_non_nullable_tree(
                chunk,
                k,
                fresh_nt,
                new_rules,
                nullable,
                nonempty_productive,
                by_lhs,
                nn_cache,
            )
            .expect("nonempty chunk should have a nonnullable tree")
        })
        .collect();
    let chunk_symbols: Vec<Symbol> = chunk_nts
        .into_iter()
        .map(Symbol::Nonterminal)
        .collect();
    build_non_nullable_tree(
        &chunk_symbols,
        k,
        fresh_nt,
        new_rules,
        nullable,
        nonempty_productive,
        by_lhs,
        nn_cache,
    )
}

fn get_or_create_non_nullable_nt(
    nt: NonterminalID,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
    new_rules: &mut Vec<Rule>,
    nullable: &BTreeSet<NonterminalID>,
    nonempty_productive: &BTreeSet<NonterminalID>,
    by_lhs: &BTreeMap<NonterminalID, Vec<Vec<Symbol>>>,
    nn_cache: &mut BTreeMap<NonterminalID, NonterminalID>,
) -> Option<NonterminalID> {
    if !nullable.contains(&nt) {
        return Some(nt);
    }
    if !nonempty_productive.contains(&nt) {
        return None;
    }
    if let Some(&cached) = nn_cache.get(&nt) {
        return Some(cached);
    }

    let Some(alts) = by_lhs.get(&nt) else {
        return None;
    };
    let nn_nt = fresh_nt();
    nn_cache.insert(nt, nn_nt);
    let mut emitted = false;
    for rhs in alts {
        if rhs.is_empty() {
            continue;
        }
        if rhs.iter().any(|symbol| match symbol {
            Symbol::Terminal(_) => true,
            Symbol::Nonterminal(inner) => !nullable.contains(inner),
        }) {
            new_rules.push(Rule { lhs: nn_nt, rhs: rhs.clone() });
            emitted = true;
            continue;
        }

        let Some(rhs_nn) = build_non_nullable_tree(
            rhs,
            2,
            fresh_nt,
            new_rules,
            nullable,
            nonempty_productive,
            by_lhs,
            nn_cache,
        ) else {
            continue;
        };
        new_rules.push(Rule {
            lhs: nn_nt,
            rhs: vec![Symbol::Nonterminal(rhs_nn)],
        });
        emitted = true;
    }

    if !emitted {
        nn_cache.remove(&nt);
        return None;
    }
    Some(nn_nt)
}

/// Inline null productions (ε-elimination).
///
/// Preprocess long nullable runs with a balanced binary tree before doing the
/// existing exhaustive elimination, to avoid the raw power-set blowups that
/// occur when many nullable nonterminals appear consecutively.
pub(crate) fn inline_null_productions(rules: &[Rule], num_nt: u32) -> Vec<Rule> {
    let preprocessed = compress_nullable_runs_with_optional_tree(rules, num_nt);
    let result = inline_null_productions_exhaustive(&preprocessed, max_nt_id(&preprocessed) + 1);
    result
}

/// Eliminate hidden left recursion.
///
/// Hidden left recursion occurs when `A → β B …` where every symbol in
/// `β` is nullable and `B` is in an indirect left-recursive cycle with `A`.
/// We add shortened rules with the nullable prefix removed, exposing the
/// left recursion so it becomes direct (which GLR handles natively).
fn eliminate_hidden_left_recursion(
    rules: &mut Vec<Rule>,
    nullable: &BTreeSet<NonterminalID>,
) {
    const MAX_ITERATIONS: usize = 20;

    for _ in 0..MAX_ITERATIONS {
        let lr_graph = build_left_reachability_graph(rules, nullable);
        let cycle = match find_indirect_lr_cycle(&lr_graph) {
            Some(c) => c,
            None => break,
        };
        let cycle_nodes: BTreeSet<NonterminalID> = cycle.into_iter().collect();

        let mut additions = Vec::new();
        for rule in rules.iter() {
            if !cycle_nodes.contains(&rule.lhs) {
                continue;
            }

            let prefix_end = nullable_prefix_len(&rule.rhs, nullable);
            // For each skip length, if next symbol is a cycle member, add shortened rule
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
        }

        if additions.is_empty() {
            break;
        }
        rules.extend(additions);
    }
}

fn nullable_prefix_len(rhs: &[Symbol], nullable: &BTreeSet<NonterminalID>) -> usize {
    rhs.iter()
        .take_while(|symbol| matches!(symbol, Symbol::Nonterminal(nt) if nullable.contains(nt)))
        .count()
}

/// Remove rules for nonterminals not reachable from the start symbol.
fn remove_unreachable_rules(rules: &[Rule], start: NonterminalID) -> Vec<Rule> {
    // Build index: lhs → rule indices for O(1) lookup per NT.
    let mut rules_by_lhs = BTreeMap::<NonterminalID, Vec<usize>>::new();
    for (i, rule) in rules.iter().enumerate() {
        rules_by_lhs.entry(rule.lhs).or_default().push(i);
    }

    let mut reachable = BTreeSet::new();
    let mut worklist = vec![start];
    while let Some(nt) = worklist.pop() {
        if !reachable.insert(nt) {
            continue;
        }
        if let Some(indexes) = rules_by_lhs.get(&nt) {
            for &idx in indexes {
                for sym in &rules[idx].rhs {
                    if let Symbol::Nonterminal(n) = sym {
                        if !reachable.contains(n) {
                            worklist.push(*n);
                        }
                    }
                }
            }
        }
    }
    rules
        .iter()
        .filter(|r| reachable.contains(&r.lhs))
        .cloned()
        .collect()
}

fn build_rhs_by_lhs(rules: &[Rule]) -> BTreeMap<NonterminalID, BTreeSet<Vec<Symbol>>> {
    let mut rhs_by_lhs = BTreeMap::<NonterminalID, BTreeSet<Vec<Symbol>>>::new();
    for rule in rules {
        rhs_by_lhs
            .entry(rule.lhs)
            .or_default()
            .insert(rule.rhs.clone());
    }
    rhs_by_lhs
}

fn compute_expandable_single_productions(
    rhs_by_lhs: &BTreeMap<NonterminalID, BTreeSet<Vec<Symbol>>>,
) -> (BTreeMap<NonterminalID, Vec<Symbol>>, BTreeSet<NonterminalID>) {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum VisitState {
        Visiting,
        Expandable,
        NotExpandable,
    }

    fn visit(
        nt: NonterminalID,
        unique_rhs_by_lhs: &BTreeMap<NonterminalID, Vec<Symbol>>,
        state: &mut BTreeMap<NonterminalID, VisitState>,
    ) -> bool {
        if let Some(existing) = state.get(&nt).copied() {
            return match existing {
                VisitState::Visiting => false,
                VisitState::Expandable => true,
                VisitState::NotExpandable => false,
            };
        }

        let Some(rhs) = unique_rhs_by_lhs.get(&nt) else {
            return false;
        };

        state.insert(nt, VisitState::Visiting);
        let expandable = rhs.iter().all(|symbol| match symbol {
            Symbol::Terminal(_) => true,
            Symbol::Nonterminal(child) => {
                if unique_rhs_by_lhs.contains_key(child) {
                    visit(*child, unique_rhs_by_lhs, state)
                } else {
                    true
                }
            }
        });
        state.insert(
            nt,
            if expandable {
                VisitState::Expandable
            } else {
                VisitState::NotExpandable
            },
        );
        expandable
    }

    let unique_rhs_by_lhs: BTreeMap<NonterminalID, Vec<Symbol>> = rhs_by_lhs
        .iter()
        .filter_map(|(&nt, rhss)| {
            if rhss.len() == 1 {
                rhss.iter().next().cloned().map(|rhs| (nt, rhs))
            } else {
                None
            }
        })
        .collect();

    let mut state = BTreeMap::<NonterminalID, VisitState>::new();
    let mut expandable = BTreeSet::new();
    for &nt in unique_rhs_by_lhs.keys() {
        if visit(nt, &unique_rhs_by_lhs, &mut state) {
            expandable.insert(nt);
        }
    }

    (unique_rhs_by_lhs, expandable)
}

fn flatten_rhs_symbols(
    rhs: &[Symbol],
    unique_rhs_by_lhs: &BTreeMap<NonterminalID, Vec<Symbol>>,
    expandable_single_productions: &BTreeSet<NonterminalID>,
    flatten_cache: &mut HashMap<NonterminalID, Option<Vec<Symbol>>>,
) -> Vec<Symbol> {
    const MAX_FLATTENED_RHS_LEN: usize = 4096;

    fn flatten_symbol(
        symbol: &Symbol,
        out: &mut Vec<Symbol>,
        unique_rhs_by_lhs: &BTreeMap<NonterminalID, Vec<Symbol>>,
        expandable_single_productions: &BTreeSet<NonterminalID>,
        flatten_cache: &mut HashMap<NonterminalID, Option<Vec<Symbol>>>,
    ) {
        match symbol {
            Symbol::Terminal(_) => out.push(symbol.clone()),
            Symbol::Nonterminal(nt)
                if expandable_single_productions.contains(nt) =>
            {
                if let Some(cached) = flatten_cache.get(nt) {
                    match cached {
                        Some(flattened) if out.len() + flattened.len() <= MAX_FLATTENED_RHS_LEN => {
                            out.extend(flattened.iter().cloned());
                        }
                        _ => out.push(symbol.clone()),
                    }
                    return;
                }

                if let Some(expanded_rhs) = unique_rhs_by_lhs.get(nt) {
                    let mut flattened_nt = Vec::new();
                    for expanded_symbol in expanded_rhs {
                        flatten_symbol(
                            expanded_symbol,
                            &mut flattened_nt,
                            unique_rhs_by_lhs,
                            expandable_single_productions,
                            flatten_cache,
                        );
                        if flattened_nt.len() > MAX_FLATTENED_RHS_LEN {
                            flatten_cache.insert(*nt, None);
                            out.push(symbol.clone());
                            return;
                        }
                    }

                    flatten_cache.insert(*nt, Some(flattened_nt.clone()));
                    out.extend(flattened_nt);
                } else {
                    out.push(symbol.clone());
                }
            }
            Symbol::Nonterminal(_) => out.push(symbol.clone()),
        }
    }

    let mut flattened = Vec::new();
    for symbol in rhs {
        flatten_symbol(
            symbol,
            &mut flattened,
            unique_rhs_by_lhs,
            expandable_single_productions,
            flatten_cache,
        );
        if flattened.len() > MAX_FLATTENED_RHS_LEN {
            return rhs.to_vec();
        }
    }
    flattened
}

/// Deduplicate rules, preserving order of first occurrence.
enum RuleDedupKey<'a> {
    Borrowed(NonterminalID, &'a [Symbol]),
    Owned(NonterminalID, Vec<Symbol>),
}

impl RuleDedupKey<'_> {
    fn lhs(&self) -> NonterminalID {
        match self {
            RuleDedupKey::Borrowed(lhs, _) | RuleDedupKey::Owned(lhs, _) => *lhs,
        }
    }

    fn rhs(&self) -> &[Symbol] {
        match self {
            RuleDedupKey::Borrowed(_, rhs) => rhs,
            RuleDedupKey::Owned(_, rhs) => rhs,
        }
    }
}

impl PartialEq for RuleDedupKey<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.lhs() == other.lhs() && self.rhs() == other.rhs()
    }
}

impl Eq for RuleDedupKey<'_> {}

impl Hash for RuleDedupKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.lhs().hash(state);
        self.rhs().hash(state);
    }
}

fn dedup_rules(rules: &mut Vec<Rule>) {
    let rhs_by_lhs = build_rhs_by_lhs(rules);
    let (unique_rhs_by_lhs, expandable_single_productions) =
        compute_expandable_single_productions(&rhs_by_lhs);
    let mut keep = Vec::with_capacity(rules.len());
    {
        let mut seen = HashSet::with_capacity(rules.len());
        let mut flatten_cache = HashMap::<NonterminalID, Option<Vec<Symbol>>>::new();
        for rule in rules.iter() {
            let can_flatten = rule.rhs.iter().any(|symbol| {
                matches!(symbol, Symbol::Nonterminal(nt) if expandable_single_productions.contains(nt))
            });
            let key = if can_flatten {
                RuleDedupKey::Owned(
                    rule.lhs,
                    flatten_rhs_symbols(
                        &rule.rhs,
                        &unique_rhs_by_lhs,
                        &expandable_single_productions,
                        &mut flatten_cache,
                    ),
                )
            } else {
                RuleDedupKey::Borrowed(rule.lhs, &rule.rhs)
            };
            keep.push(seen.insert(key));
        }
    }

    let mut keep_iter = keep.into_iter();
    rules.retain(|_| keep_iter.next().unwrap_or(false));
}

fn is_reflexive_unit_rule(rule: &Rule) -> bool {
    matches!(rule.rhs.as_slice(), [Symbol::Nonterminal(nonterminal)] if *nonterminal == rule.lhs)
}

pub(crate) fn merge_identical_nonterminals(
    rules: &[Rule],
    start: NonterminalID,
) -> Vec<Rule> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Build rhs_by_lhs: the set of productions for each nonterminal.
    let rhs_by_lhs = build_rhs_by_lhs(rules);

    if rhs_by_lhs.len() <= 1 {
        return rules.to_vec();
    }

    let nts: Vec<NonterminalID> = rhs_by_lhs.keys().copied().collect();

    // Fast O(1) lookup from NT ID → index, replacing BTreeMap (which is
    // O(log n) and dominates the hot loop when there are millions of
    // lookups across refinement iterations).
    let max_nt_id = *nts.last().unwrap() as usize;
    let mut nt_to_idx_fast = vec![u32::MAX; max_nt_id + 1];
    for (i, &nt) in nts.iter().enumerate() {
        nt_to_idx_fast[nt as usize] = i as u32;
    }

    // Pre-index production sets for O(1) access by NT index.
    let rhs_by_idx: Vec<&BTreeSet<Vec<Symbol>>> =
        nts.iter().map(|nt| &rhs_by_lhs[nt]).collect();
    let (unique_rhs_by_lhs, expandable_single_productions) =
        compute_expandable_single_productions(&rhs_by_lhs);
    let mut flatten_cache = HashMap::<NonterminalID, Option<Vec<Symbol>>>::new();
    let flattened_rhs_by_idx: Vec<Vec<Vec<Symbol>>> = nts
        .iter()
        .map(|nt| {
            rhs_by_lhs[nt]
                .iter()
                .map(|rhs| {
                    flatten_rhs_symbols(
                        rhs,
                        &unique_rhs_by_lhs,
                        &expandable_single_productions,
                        &mut flatten_cache,
                    )
                })
                .collect()
        })
        .collect();

    // ── Partition refinement (top-down) ──────────────────────────────────
    //
    // Instead of the classical bottom-up "find-one-merge, re-scan" loop
    // (which cascades for O(chain_length) iterations), we use partition
    // refinement: start with the *coarsest* partition consistent with
    // terminal structure, then iteratively refine by incorporating the
    // partition classes of referenced nonterminals.  This discovers all
    // transitively-isomorphic nonterminals in O(refinement_depth) passes,
    // with O(n) work per pass.
    //
    // Each refinement hashes each NT's normalised production set (with
    // self-refs → sentinel, other NT refs → current class ID), then
    // normalises the resulting class IDs by order of first appearance so
    // that the convergence check is stable.

    const SELF_SENTINEL: u64 = u64::MAX;
    const GENERIC_NT: u64 = u64::MAX - 1;

    // Hash a single NT's production set given current class assignments.
    // Uses commutative accumulation (wrapping_add of rotated hashes) to
    // avoid Vec allocation and sorting.
    let hash_nt = |nt_idx: usize, class_of: &[u64]| -> u64 {
        let nt = nts[nt_idx];
        let mut sig: u64 = 0;
        for rhs in &flattened_rhs_by_idx[nt_idx] {
            let mut h = DefaultHasher::new();
            for s in rhs {
                match s {
                    Symbol::Terminal(t) => {
                        0u8.hash(&mut h);
                        t.hash(&mut h);
                    }
                    Symbol::Nonterminal(n) if *n == nt => {
                        1u8.hash(&mut h);
                        SELF_SENTINEL.hash(&mut h);
                    }
                    Symbol::Nonterminal(n) => {
                        let ni = *n as usize;
                        if ni <= max_nt_id && nt_to_idx_fast[ni] != u32::MAX {
                            1u8.hash(&mut h);
                            class_of[nt_to_idx_fast[ni] as usize].hash(&mut h);
                        } else {
                            2u8.hash(&mut h);
                            n.hash(&mut h);
                        }
                    }
                }
            }
            // Commutative combine: wrapping_add of scrambled prod hash.
            let ph = h.finish();
            sig = sig.wrapping_add(ph.wrapping_mul(0x9E3779B97F4A7C15));
        }
        sig
    };

    // Normalise returns (normalised_vec, n_distinct_classes).
    let normalise_counted = |raw: &[u64]| -> (Vec<u64>, usize) {
        let mut map = HashMap::<u64, u64>::with_capacity(raw.len());
        let mut nc: u64 = 0;
        let v: Vec<u64> = raw
            .iter()
            .map(|&v| {
                *map.entry(v).or_insert_with(|| {
                    let c = nc;
                    nc += 1;
                    c
                })
            })
            .collect();
        (v, nc as usize)
    };

    // Quick pre-check: hash with REAL NT IDs (no GENERIC_NT). If every
    // NT already has a unique signature, there are no isomorphisms and
    // we can skip the expensive partition refinement entirely. This
    // handles the common "confirmatory call" case in O(n).
    {
        let real_classes: Vec<u64> = (0..nts.len())
            .map(|i| {
                // Use the NT's own index as its class (finest partition).
                let nt = nts[i];
                let mut sig: u64 = 0;
                for rhs in &flattened_rhs_by_idx[i] {
                    let mut h = DefaultHasher::new();
                    for s in rhs {
                        match s {
                            Symbol::Terminal(t) => {
                                0u8.hash(&mut h);
                                t.hash(&mut h);
                            }
                            Symbol::Nonterminal(n) if *n == nt => {
                                1u8.hash(&mut h);
                                SELF_SENTINEL.hash(&mut h);
                            }
                            Symbol::Nonterminal(n) => {
                                2u8.hash(&mut h);
                                n.hash(&mut h);
                            }
                        }
                    }
                    let ph = h.finish();
                    sig = sig.wrapping_add(ph.wrapping_mul(0x9E3779B97F4A7C15));
                }
                sig
            })
            .collect();
        let mut seen = HashSet::with_capacity(real_classes.len());
        if real_classes.iter().all(|h| seen.insert(*h)) {
            // Every NT has a unique raw-ID signature → no merges possible.
            return rules.to_vec();
        }
    }

    // Initial partition: all NT refs → GENERIC_NT.
    let initial_classes: Vec<u64> = {
        let init = vec![GENERIC_NT; nts.len()];
        (0..nts.len()).map(|i| hash_nt(i, &init)).collect()
    };
    let (mut class_of, n_classes) = normalise_counted(&initial_classes);

    // If every NT is already in its own class, no merges are possible.
    if n_classes == nts.len() {
        return rules.to_vec();
    }

    // ── Compute processing order: ascending "depth" in the reference DAG ──
    //
    // NTs at depth 0 have no references to other NTs (leaves). Depth d means
    // the longest path to a leaf is d hops. Processing in ascending depth
    // with in-place updates (Gauss-Seidel) propagates chain information in
    // ONE pass instead of one-per-chain-link. For purely acyclic chains of
    // depth D, this reduces D iterations to ~1.
    //
    // We compute exact depths via SCC condensation (Kosaraju's) + DAG depth,
    // which handles cycles correctly (NTs in the same SCC share a depth) and
    // works for arbitrarily deep chains.
    let processing_order: Vec<usize> = {
        let n = nts.len();

        // Build adjacency: for each NT index, which other NT indices does it reference?
        let mut refs_of: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, &nt) in nts.iter().enumerate() {
            for rhs in rhs_by_idx[i].iter() {
                for s in rhs {
                    if let Symbol::Nonterminal(r) = s {
                        if *r != nt {
                            let ri = *r as usize;
                            if ri <= max_nt_id && nt_to_idx_fast[ri] != u32::MAX {
                                refs_of[i].push(nt_to_idx_fast[ri] as usize);
                            }
                        }
                    }
                }
            }
        }

        // ── Kosaraju's SCC algorithm (O(V+E), iterative) ──

        // Step 1: iterative DFS on original graph → finish order.
        let mut visited = vec![false; n];
        let mut finish_order = Vec::with_capacity(n);
        for start in 0..n {
            if visited[start] { continue; }
            let mut stk: Vec<(usize, usize)> = vec![(start, 0)];
            visited[start] = true;
            while let Some((v, ni)) = stk.last_mut() {
                if *ni < refs_of[*v].len() {
                    let w = refs_of[*v][*ni];
                    *ni += 1;
                    if !visited[w] {
                        visited[w] = true;
                        stk.push((w, 0));
                    }
                } else {
                    finish_order.push(*v);
                    stk.pop();
                }
            }
        }

        // Step 2: reverse graph + DFS in reverse finish order → SCC IDs.
        // Kosaraju's numbers SCCs in topological order (sources first in the
        // original dependent→dependency edge direction).
        let mut rev_refs: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, neighbors) in refs_of.iter().enumerate() {
            for &j in neighbors {
                rev_refs[j].push(i);
            }
        }
        let mut scc_id = vec![0u32; n];
        let mut next_scc = 0u32;
        visited.fill(false);
        for &start in finish_order.iter().rev() {
            if visited[start] { continue; }
            let mut stk = vec![start];
            visited[start] = true;
            while let Some(v) = stk.pop() {
                scc_id[v] = next_scc;
                for &w in &rev_refs[v] {
                    if !visited[w] {
                        visited[w] = true;
                        stk.push(w);
                    }
                }
            }
            next_scc += 1;
        }
        let num_sccs = next_scc as usize;

        // Step 3: condensed DAG depth (longest path to a sink = leaf depth 0).
        // Build inter-SCC edges, deduplicate, then compute depth in reverse
        // topological order (SCCs numbered source-first, so iterate high→low).
        let mut scc_edges: Vec<Vec<u32>> = vec![Vec::new(); num_sccs];
        for (i, neighbors) in refs_of.iter().enumerate() {
            for &j in neighbors {
                if scc_id[i] != scc_id[j] {
                    scc_edges[scc_id[i] as usize].push(scc_id[j]);
                }
            }
        }
        for edges in &mut scc_edges {
            edges.sort_unstable();
            edges.dedup();
        }
        let mut scc_depth = vec![0u32; num_sccs];
        for s in (0..num_sccs).rev() {
            for &dst in &scc_edges[s] {
                scc_depth[s] = scc_depth[s].max(scc_depth[dst as usize] + 1);
            }
        }

        // Map SCC depth back to NTs, sort by ascending depth (leaves first).
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by_key(|&i| scc_depth[scc_id[i] as usize]);
        order
    };

    // Refine until stable, using depth-ordered in-place (Gauss-Seidel) updates.
    let mut iters = 0u32;
    loop {
        iters += 1;
        let prev_class_of = class_of.clone();
        // In-place update: process NTs in depth order so that deeper NTs
        // (which depend on shallower ones) see already-updated classes.
        let mut raw = vec![0u64; nts.len()];
        for &i in &processing_order {
            raw[i] = hash_nt(i, &class_of);
            // Update class_of in-place for Gauss-Seidel propagation.
            class_of[i] = raw[i];
        }
        let (new_class_of, nc) = normalise_counted(&raw);
        class_of = new_class_of;
        if class_of == prev_class_of {
            break;
        }
        // Early termination: every NT is in its own class → no merges.
        if nc == nts.len() {
            return rules.to_vec();
        }
    }

    // ── Build merge map from final partition ─────────────────────────────
    // Within each equivalence class, pick a representative (prefer start,
    // otherwise lowest NT ID — which is naturally first since `nts` is
    // sorted from a BTreeMap).

    let mut class_to_rep: BTreeMap<u64, NonterminalID> = BTreeMap::new();
    for (idx, &nt) in nts.iter().enumerate() {
        let class = class_of[idx];
        let rep = class_to_rep.entry(class).or_insert(nt);
        if nt == start {
            *rep = start;
        }
    }

    let mut merge_map: BTreeMap<NonterminalID, NonterminalID> = BTreeMap::new();
    for (idx, &nt) in nts.iter().enumerate() {
        let rep = class_to_rep[&class_of[idx]];
        if nt != rep {
            merge_map.insert(nt, rep);
        }
    }

    if merge_map.is_empty() {
        return rules.to_vec();
    }

    // Apply merge map to produce the deduplicated rule set.
    let apply = |nt: NonterminalID| -> NonterminalID {
        *merge_map.get(&nt).unwrap_or(&nt)
    };

    let mut result = Vec::new();
    let mut seen = HashSet::with_capacity(rules.len());
    for rule in rules {
        let lhs = apply(rule.lhs);
        let rhs: Vec<Symbol> = rule
            .rhs
            .iter()
            .map(|symbol| match symbol {
                Symbol::Terminal(terminal) => Symbol::Terminal(*terminal),
                Symbol::Nonterminal(nonterminal) => Symbol::Nonterminal(apply(*nonterminal)),
            })
            .collect();
        let merged = Rule { lhs, rhs };
        if is_reflexive_unit_rule(&merged) {
            continue;
        }
        if seen.insert(merged.clone()) {
            result.push(merged);
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Grammar Normalization Pipeline
// ─────────────────────────────────────────────────────────────────────────────
//
// Transforms a grammar so that it satisfies the preconditions required by the
// terminal-characterization stage:
//
//   1. No nullable nonterminals — every nonterminal derives at least one
//      terminal symbol.
//   2. No right recursion — neither direct (A → α A) nor indirect
//      (A →* α B, B →* β A).
//   3. No indirect left recursion — only direct left recursion (A → A α) is
//      permitted (safe for GLR).
//
// The normalization loop repeatedly inlines null productions, eliminates
// right recursion, and exposes hidden left recursion until the grammar stops
// changing, then runs the final epsilon-elimination and unreachable pruning.

/// Run the full grammar normalization pipeline (in place).
///
/// Mutates `rules` so that they satisfy the characterization
/// preconditions (no nullable NTs, no right recursion, no indirect LR).
/// `start` is used only for unreachable-production pruning and is never
/// changed.
pub fn normalize_grammar(rules: &mut Vec<Rule>, start: NonterminalID) {
    use std::cell::Cell;

    let next_nt = Cell::new(max_nt_id(rules) + 1);
    let mut fresh_nt = || {
        let id = next_nt.get();
        next_nt.set(id + 1);
        id
    };

    let mut iteration = 0;
    loop {
        let snap = rules.clone();
        replace_rules_with_resync(rules, &next_nt, inline_null_productions);
        with_resynced_next_nonterminal(rules, &next_nt, |rules| {
            eliminate_right_recursion(rules, &mut fresh_nt);
        });
        with_resynced_next_nonterminal(rules, &next_nt, |rules| {
            let nullable = compute_nullable(rules, max_nt_id(rules) + 1);
            eliminate_hidden_left_recursion(rules, &nullable);
        });
        dedup_rules(rules);
        let converged = *rules == snap;
        iteration += 1;

        if converged {
            break;
        }
    }

    replace_rules_with_resync(rules, &next_nt, inline_null_productions);
    *rules = remove_unreachable_rules(rules, start);
    dedup_rules(rules);
}

fn replace_rules_with_resync(
    rules: &mut Vec<Rule>,
    next_nt: &std::cell::Cell<u32>,
    update: impl FnOnce(&[Rule], u32) -> Vec<Rule>,
) {
    *rules = update(rules, next_nt.get());
    resync_next_nonterminal(rules, next_nt);
}

fn with_resynced_next_nonterminal(
    rules: &mut Vec<Rule>,
    next_nt: &std::cell::Cell<u32>,
    update: impl FnOnce(&mut Vec<Rule>),
) {
    update(rules);
    resync_next_nonterminal(rules, next_nt);
}

fn resync_next_nonterminal(rules: &[Rule], next_nt: &std::cell::Cell<u32>) {
    next_nt.set(max_nt_id(rules) + 1);
}

fn compute_nullable(rules: &[Rule], num_nt: u32) -> BTreeSet<NonterminalID> {
    let mut nullable = BTreeSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        for rule in rules {
            if rule.lhs >= num_nt {
                continue;
            }
            let rhs_nullable = rule.rhs.is_empty()
                || rule.rhs.iter().all(|symbol| match symbol {
                    Symbol::Terminal(_) => false,
                    Symbol::Nonterminal(nonterminal) => nullable.contains(nonterminal),
                });
            if rhs_nullable && nullable.insert(rule.lhs) {
                changed = true;
            }
        }
    }
    nullable
}

fn compute_first(
    rules: &[Rule],
    num_nt: u32,
    nullable: &BTreeSet<NonterminalID>,
) -> Vec<BTreeSet<TerminalID>> {
    let mut first = vec![BTreeSet::new(); num_nt as usize];
    let mut changed = true;
    while changed {
        changed = false;
        for rule in rules {
            let lhs = rule.lhs as usize;
            for symbol in &rule.rhs {
                match symbol {
                    Symbol::Terminal(terminal) => {
                        changed |= first[lhs].insert(*terminal);
                        break;
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        let additions = first[*nonterminal as usize].clone();
                        let old_len = first[lhs].len();
                        first[lhs].extend(additions);
                        changed |= first[lhs].len() != old_len;
                        if !nullable.contains(nonterminal) {
                            break;
                        }
                    }
                }
            }
        }
    }
    first
}

fn compute_follow(
    rules: &[Rule],
    num_nt: u32,
    start: NonterminalID,
    first: &[BTreeSet<TerminalID>],
    nullable: &BTreeSet<NonterminalID>,
) -> Vec<BTreeSet<TerminalID>> {
    let mut follow = vec![BTreeSet::new(); num_nt as usize];
    if let Some(start_follow) = follow.get_mut(start as usize) {
        start_follow.insert(EOF);
    }

    let mut changed = true;
    while changed {
        changed = false;
        for rule in rules {
            let lhs_follow = follow[rule.lhs as usize].clone();
            for (index, symbol) in rule.rhs.iter().enumerate() {
                let Symbol::Nonterminal(nonterminal) = symbol else {
                    continue;
                };

                let suffix = &rule.rhs[index + 1..];
                let mut additions = BTreeSet::new();
                let mut suffix_nullable = true;
                for suffix_symbol in suffix {
                    match suffix_symbol {
                        Symbol::Terminal(terminal) => {
                            additions.insert(*terminal);
                            suffix_nullable = false;
                            break;
                        }
                        Symbol::Nonterminal(next_nonterminal) => {
                            additions.extend(first[*next_nonterminal as usize].iter().copied());
                            if !nullable.contains(next_nonterminal) {
                                suffix_nullable = false;
                                break;
                            }
                        }
                    }
                }
                if suffix_nullable {
                    additions.extend(lhs_follow.iter().copied());
                }

                let target = &mut follow[*nonterminal as usize];
                let old_len = target.len();
                target.extend(additions);
                changed |= target.len() != old_len;
            }
        }
    }

    follow
}

fn filter_graph_to_reachable(
    graph: BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
    reachable: &BTreeSet<NonterminalID>,
) -> BTreeMap<NonterminalID, BTreeSet<NonterminalID>> {
    graph
        .into_iter()
        .filter(|(nonterminal, _)| reachable.contains(nonterminal))
        .map(|(nonterminal, edges)| {
            (
                nonterminal,
                edges
                    .into_iter()
                    .filter(|edge| reachable.contains(edge))
                    .collect(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::flat::{GrammarDef, Terminal};

    fn analyzed_grammar(rules: Vec<Rule>, start: NonterminalID) -> AnalyzedGrammar {
        AnalyzedGrammar::from_grammar_def(&GrammarDef {
            rules,
            start,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..GrammarDef::default()
        })
    }

    fn bounded_language(
        rules: &[Rule],
        start: NonterminalID,
        num_nonterminals: u32,
        max_len: usize,
    ) -> BTreeSet<Vec<TerminalID>> {
        fn rhs_language(
            rhs: &[Symbol],
            languages: &[BTreeSet<Vec<TerminalID>>],
            max_len: usize,
        ) -> BTreeSet<Vec<TerminalID>> {
            let mut acc = BTreeSet::from([Vec::new()]);
            for symbol in rhs {
                let part = match symbol {
                    Symbol::Terminal(terminal) => {
                        BTreeSet::from([vec![*terminal]])
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        languages[*nonterminal as usize].clone()
                    }
                };
                let mut next = BTreeSet::new();
                for prefix in &acc {
                    for suffix in &part {
                        if prefix.len() + suffix.len() <= max_len {
                            let mut combined = prefix.clone();
                            combined.extend(suffix);
                            next.insert(combined);
                        }
                    }
                }
                acc = next;
                if acc.is_empty() {
                    break;
                }
            }
            acc
        }

        let mut languages = vec![BTreeSet::new(); num_nonterminals as usize];
        loop {
            let mut changed = false;
            for rule in rules {
                let derived = rhs_language(&rule.rhs, &languages, max_len);
                let target = &mut languages[rule.lhs as usize];
                let old_len = target.len();
                target.extend(derived);
                changed |= target.len() != old_len;
            }
            if !changed {
                break;
            }
        }
        languages[start as usize].clone()
    }

    #[test]
    fn table_build_normal_form_rejects_nullable_zero_length_rules() {
        let grammar = analyzed_grammar(vec![Rule { lhs: 0, rhs: Vec::new() }], 0);

        let error = grammar.check_table_build_normal_form().unwrap_err();
        assert!(error.contains("nullable nonterminals reachable"));
        assert!(error.contains("zero-length productions reachable"));
    }

    #[test]
    fn table_build_normal_form_rejects_direct_right_recursion() {
        let grammar = analyzed_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            0,
        );

        let error = grammar.check_table_build_normal_form().unwrap_err();
        assert!(error.contains("right-recursive cycle detected"));
    }

    #[test]
    fn table_build_normal_form_rejects_indirect_left_recursion() {
        let grammar = analyzed_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            0,
        );

        let error = grammar.check_table_build_normal_form().unwrap_err();
        assert!(error.contains("indirect left-recursive cycle detected"));
    }

    #[test]
    fn table_build_normal_form_accepts_simple_nonnullable_grammar() {
        let grammar = analyzed_grammar(
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            0,
        );

        assert!(grammar.check_table_build_normal_form().is_ok());
    }

    #[test]
    fn nullable_run_compression_preserves_nullable_only_nonempty_derivations() {
        let rules = vec![
            Rule {
                lhs: 3,
                rhs: vec![Symbol::Terminal(1)],
            },
            Rule { lhs: 2, rhs: vec![] },
            Rule {
                lhs: 2,
                rhs: vec![Symbol::Nonterminal(2), Symbol::Nonterminal(3)],
            },
            Rule {
                lhs: 0,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Nonterminal(1),
                    Symbol::Nonterminal(2),
                    Symbol::Terminal(0),
                ],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Terminal(2)],
            },
            Rule {
                lhs: 5,
                rhs: vec![Symbol::Terminal(3)],
            },
            Rule { lhs: 4, rhs: vec![] },
            Rule {
                lhs: 4,
                rhs: vec![Symbol::Nonterminal(4), Symbol::Nonterminal(5)],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Nonterminal(4)],
            },
        ];

        let source = bounded_language(&rules, 0, 6, 3);
        let transformed_rules = inline_null_productions(&rules, 6);
        let transformed =
            bounded_language(&transformed_rules, 0, max_nt_id(&transformed_rules) + 1, 3);
        let mut normalized_rules = rules.clone();
        normalize_grammar(&mut normalized_rules, 0);
        let normalized =
            bounded_language(&normalized_rules, 0, max_nt_id(&normalized_rules) + 1, 3);

        assert!(source.contains(&vec![0, 3, 0]));
        assert_eq!(transformed, source);
        assert_eq!(normalized, source);
    }
}
