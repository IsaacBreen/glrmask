use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::compiler::grammar::model::{GrammarDef, NonterminalID, Rule, Symbol, TerminalID};

pub const EOF: TerminalID = u32::MAX;

#[derive(Debug, Clone)]
pub struct AnalyzedGrammar {
    pub rules: Vec<Rule>,
    #[allow(dead_code)]
    pub start: NonterminalID,
    pub num_terminals: u32,
    pub num_nonterminals: u32,
    pub nullable: BTreeSet<NonterminalID>,
    pub first: Vec<BTreeSet<TerminalID>>,
    pub follow: Vec<BTreeSet<TerminalID>>,
    /// Index: nonterminal → list of rule indices with that nonterminal as LHS.
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
            start: augmented_start,
            num_terminals: g.num_terminals(),
            num_nonterminals,
            nullable,
            first,
            follow,
            rules_by_lhs,
        }
    }

    /// Debug check: asserts the grammar has no right recursion, no indirect left
    /// recursion, and no nullable nonterminals.
    ///
    /// Calls [`check_no_nullable_nonterminals`] and
    /// [`check_recursion_boundedness`] and merges their results.
    pub fn debug_check_grammar_preconditions(&self) -> Result<(), String> {
        let mut violations: Vec<String> = Vec::new();
        if let Err(msg) = self.check_no_nullable_nonterminals() {
            violations.push(msg);
        }
        if let Err(msg) = self.check_recursion_boundedness() {
            violations.push(msg);
        }
        if violations.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "grammar precondition violations ({} found):\n{}",
                violations.len(),
                violations.iter()
                    .enumerate()
                    .map(|(i, v)| format!("  {}. {}", i + 1, v))
                    .collect::<Vec<_>>()
                    .join("\n"),
            ))
        }
    }

    /// Check that no nonterminal is nullable (derives ε).
    pub fn check_no_nullable_nonterminals(&self) -> Result<(), String> {
        if !self.nullable.is_empty() {
            let ids: Vec<u32> = self.nullable.iter()
                .filter(|&&nt| nt < self.num_nonterminals - 1) // skip augmented start
                .copied()
                .collect();
            if !ids.is_empty() {
                return Err(format!(
                    "nullable nonterminals detected: {:?}. \
                     Rules with ε-productions or all-nullable RHS create \
                     reduce chains that the characterisation stage cannot \
                     handle when combined with recursion.",
                    ids,
                ));
            }
        }
        Ok(())
    }

    /// Check that the grammar has no right-recursive or indirect
    /// left-recursive cycles.
    pub fn check_recursion_boundedness(&self) -> Result<(), String> {
        let mut violations: Vec<String> = Vec::new();

        let rr_graph = build_right_reachability_graph(&self.rules, &self.nullable);
        if let Some(cycle) = find_indirect_rr_cycle(&rr_graph) {
            violations.push(format!(
                "right-recursive cycle detected: {:?}. \
                 Right recursion causes unbounded reduce chains in \
                 terminal characterisation.  Convert to left recursion \
                 or inline the cycle.",
                cycle,
            ));
        }

        let lr_graph = build_left_reachability_graph(&self.rules, &self.nullable);
        if let Some(cycle) = find_indirect_lr_cycle(&lr_graph) {
            if cycle.len() >= 2 {
                violations.push(format!(
                    "indirect left-recursive cycle detected: {:?}. \
                     Indirect left recursion may create unbounded GSS \
                     growth.  Inline or rewrite the cycle.",
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

fn build_right_reachability_graph(
    rules: &[Rule],
    nullable: &BTreeSet<NonterminalID>,
) -> BTreeMap<NonterminalID, BTreeSet<NonterminalID>> {
    let mut graph = BTreeMap::<NonterminalID, BTreeSet<NonterminalID>>::new();
    for rule in rules {
        let suffix = rule
            .rhs
            .iter()
            .rev()
            .take_while(|symbol| match symbol {
                Symbol::Nonterminal(nonterminal) => nullable.contains(nonterminal),
                Symbol::Terminal(_) => false,
            })
            .collect::<Vec<_>>();
        for symbol in suffix.into_iter().rev() {
            if let Symbol::Nonterminal(nonterminal) = symbol {
                graph.entry(rule.lhs).or_default().insert(*nonterminal);
            }
        }
        if let Some(Symbol::Nonterminal(nonterminal)) = rule.rhs.last() {
            graph.entry(rule.lhs).or_default().insert(*nonterminal);
        }
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
        for symbol in &rule.rhs {
            match symbol {
                Symbol::Nonterminal(nonterminal) => {
                    graph.entry(rule.lhs).or_default().insert(*nonterminal);
                    if !nullable.contains(nonterminal) {
                        break;
                    }
                }
                Symbol::Terminal(_) => break,
            }
        }
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

fn compress_nullable_runs_with_optional_tree(rules: &[Rule], num_nt: u32) -> Vec<Rule> {
    let nullable = compute_nullable(rules, num_nt);
    if nullable.is_empty() {
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
            let root_nn = build_non_nullable_tree(
                &segment,
                2,
                &mut fresh_nt,
                &mut result,
                &nullable,
                &by_lhs,
                &mut nn_cache,
            );
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
    by_lhs: &BTreeMap<NonterminalID, Vec<Vec<Symbol>>>,
    nn_cache: &mut BTreeMap<NonterminalID, NonterminalID>,
) -> NonterminalID {
    let k = k.max(2);
    let n = segment.len();
    if n == 0 {
        let nt = fresh_nt();
        new_rules.push(Rule { lhs: nt, rhs: vec![] });
        return nt;
    }

    let nn_segment: Vec<Symbol> = segment
        .iter()
        .map(|symbol| match symbol {
            Symbol::Terminal(terminal) => Symbol::Terminal(*terminal),
            Symbol::Nonterminal(nonterminal) => Symbol::Nonterminal(get_or_create_non_nullable_nt(
                *nonterminal,
                fresh_nt,
                new_rules,
                nullable,
                by_lhs,
                nn_cache,
            )),
        })
        .collect();

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
        return leaf_nt;
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
            build_non_nullable_tree(chunk, k, fresh_nt, new_rules, nullable, by_lhs, nn_cache)
        })
        .collect();
    let chunk_symbols: Vec<Symbol> = chunk_nts
        .into_iter()
        .map(Symbol::Nonterminal)
        .collect();
    build_non_nullable_tree(&chunk_symbols, k, fresh_nt, new_rules, nullable, by_lhs, nn_cache)
}

fn get_or_create_non_nullable_nt(
    nt: NonterminalID,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
    new_rules: &mut Vec<Rule>,
    nullable: &BTreeSet<NonterminalID>,
    by_lhs: &BTreeMap<NonterminalID, Vec<Vec<Symbol>>>,
    nn_cache: &mut BTreeMap<NonterminalID, NonterminalID>,
) -> NonterminalID {
    if !nullable.contains(&nt) {
        return nt;
    }
    if let Some(&cached) = nn_cache.get(&nt) {
        return cached;
    }

    let Some(alts) = by_lhs.get(&nt) else {
        return nt;
    };
    let kept: Vec<Vec<Symbol>> = alts
        .iter()
        .filter(|alt| {
            !alt.is_empty()
                && alt.iter().any(|symbol| match symbol {
                    Symbol::Terminal(_) => true,
                    Symbol::Nonterminal(inner) => !nullable.contains(inner),
                })
        })
        .cloned()
        .collect();

    if kept.is_empty() {
        return nt;
    }

    let nn_nt = fresh_nt();
    nn_cache.insert(nt, nn_nt);
    for rhs in kept {
        new_rules.push(Rule { lhs: nn_nt, rhs });
    }
    nn_nt
}

/// Inline null productions (ε-elimination).
///
/// Preprocess long nullable runs with a balanced binary tree before doing the
/// existing exhaustive elimination, to avoid the raw power-set blowups that
/// occur when many nullable nonterminals appear consecutively.
pub(crate) fn inline_null_productions(rules: &[Rule], num_nt: u32) -> Vec<Rule> {
    let preprocessed = compress_nullable_runs_with_optional_tree(rules, num_nt);
    inline_null_productions_exhaustive(&preprocessed, max_nt_id(&preprocessed) + 1)
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

/// Deduplicate rules, preserving order of first occurrence.
fn dedup_rules(rules: &mut Vec<Rule>) {
    let mut seen = HashSet::with_capacity(rules.len());
    rules.retain(|r| seen.insert(r.clone()));
}

pub(crate) fn merge_identical_nonterminals(
    rules: &[Rule],
    start: NonterminalID,
) -> Vec<Rule> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Sentinel for normalizing self-references when comparing isomorphic NTs.
    let self_sentinel: NonterminalID = u32::MAX;

    let mut rhs_by_lhs = BTreeMap::<NonterminalID, BTreeSet<Vec<Symbol>>>::new();
    for rule in rules {
        rhs_by_lhs.entry(rule.lhs).or_default().insert(rule.rhs.clone());
    }

    // Normalize a production set by replacing self-references with a sentinel.
    // This allows detecting isomorphic self-referencing nonterminals like:
    //   NT_A: { [T1], [NT_A, T1] }  and  NT_B: { [T1], [NT_B, T1] }
    let normalize_rhs_set =
        |nt: NonterminalID, rhs_set: &BTreeSet<Vec<Symbol>>| -> BTreeSet<Vec<Symbol>> {
            rhs_set
                .iter()
                .map(|rhs| {
                    rhs.iter()
                        .map(|sym| match sym {
                            Symbol::Nonterminal(n) if *n == nt => {
                                Symbol::Nonterminal(self_sentinel)
                            }
                            other => other.clone(),
                        })
                        .collect()
                })
                .collect()
        };

    let compute_hash = |nt: NonterminalID, rhs_set: &BTreeSet<Vec<Symbol>>| -> u64 {
        let normalized = normalize_rhs_set(nt, rhs_set);
        let mut hasher = DefaultHasher::new();
        for rhs in &normalized {
            rhs.hash(&mut hasher);
        }
        hasher.finish()
    };

    let mut hash_buckets = BTreeMap::<u64, Vec<NonterminalID>>::new();
    for (&lhs, rhs_set) in &rhs_by_lhs {
        hash_buckets
            .entry(compute_hash(lhs, rhs_set))
            .or_default()
            .push(lhs);
    }

    let mut merge_map = BTreeMap::<NonterminalID, NonterminalID>::new();
    for nts in hash_buckets.values() {
        if nts.len() < 2 {
            continue;
        }
        for i in 0..nts.len() {
            if merge_map.contains_key(&nts[i]) {
                continue;
            }
            let norm_i = normalize_rhs_set(nts[i], &rhs_by_lhs[&nts[i]]);
            for j in (i + 1)..nts.len() {
                if merge_map.contains_key(&nts[j]) {
                    continue;
                }
                let norm_j = normalize_rhs_set(nts[j], &rhs_by_lhs[&nts[j]]);
                if norm_i == norm_j {
                    let (keep, remove) = if nts[j] == start {
                        (nts[j], nts[i])
                    } else {
                        (nts[i], nts[j])
                    };
                    merge_map.insert(remove, keep);
                }
            }
        }
    }

    if merge_map.is_empty() {
        return rules.to_vec();
    }

    let apply_merge = |nt: NonterminalID| merge_map.get(&nt).copied().unwrap_or(nt);

    let mut result = Vec::new();
    let mut seen = HashSet::with_capacity(rules.len());
    for rule in rules {
        let lhs = apply_merge(rule.lhs);
        let rhs: Vec<Symbol> = rule
            .rhs
            .iter()
            .map(|symbol| match symbol {
                Symbol::Terminal(terminal) => Symbol::Terminal(*terminal),
                Symbol::Nonterminal(nonterminal) => Symbol::Nonterminal(apply_merge(*nonterminal)),
            })
            .collect();
        let merged = Rule { lhs, rhs };
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

    let debug_profile = std::env::var("GLRMASK_DEBUG_PROFILE")
        .map(|v| { let n = v.trim().to_ascii_lowercase(); !matches!(n.as_str(), "" | "0" | "false" | "no" | "off") })
        .unwrap_or(false);

    let next_nt = Cell::new(max_nt_id(rules) + 1);
    let mut fresh_nt = || {
        let id = next_nt.get();
        next_nt.set(id + 1);
        id
    };

    let mut iteration = 0;
    loop {
        let iter_start = std::time::Instant::now();
        let snap = rules.clone();
        let clone_ms = iter_start.elapsed().as_secs_f64() * 1000.0;

        let t0 = std::time::Instant::now();
        replace_rules_with_resync(rules, &next_nt, inline_null_productions);
        let inline_null_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        with_resynced_next_nonterminal(rules, &next_nt, |rules| {
            eliminate_right_recursion(rules, &mut fresh_nt);
        });
        let elim_right_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = std::time::Instant::now();
        with_resynced_next_nonterminal(rules, &next_nt, |rules| {
            let nullable = compute_nullable(rules, max_nt_id(rules) + 1);
            eliminate_hidden_left_recursion(rules, &nullable);
        });
        let elim_hidden_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let t3 = std::time::Instant::now();
        dedup_rules(rules);
        let dedup_ms = t3.elapsed().as_secs_f64() * 1000.0;

        let t4 = std::time::Instant::now();
        let converged = *rules == snap;
        let compare_ms = t4.elapsed().as_secs_f64() * 1000.0;

        if debug_profile {
            eprintln!(
                "[glrmask/debug][normalize] iter={} rules={} clone_ms={:.3} inline_null_ms={:.3} elim_right_ms={:.3} elim_hidden_ms={:.3} dedup_ms={:.3} compare_ms={:.3} total_ms={:.3}",
                iteration, rules.len(), clone_ms, inline_null_ms, elim_right_ms, elim_hidden_ms, dedup_ms, compare_ms,
                iter_start.elapsed().as_secs_f64() * 1000.0,
            );
        }
        iteration += 1;

        if converged {
            break;
        }
    }

    let post_t0 = std::time::Instant::now();
    replace_rules_with_resync(rules, &next_nt, inline_null_productions);
    let post_inline_null_ms = post_t0.elapsed().as_secs_f64() * 1000.0;

    let post_t1 = std::time::Instant::now();
    *rules = remove_unreachable_rules(rules, start);
    let post_remove_ms = post_t1.elapsed().as_secs_f64() * 1000.0;

    let post_t2 = std::time::Instant::now();
    dedup_rules(rules);
    let post_dedup_ms = post_t2.elapsed().as_secs_f64() * 1000.0;

    if debug_profile {
        eprintln!(
            "[glrmask/debug][normalize] post_loop rules={} inline_null_ms={:.3} remove_unreachable_ms={:.3} dedup_ms={:.3}",
            rules.len(), post_inline_null_ms, post_remove_ms, post_dedup_ms,
        );
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar::model::tests::*;
    use crate::compiler::grammar::model::{GrammarDef, Rule, Symbol, Terminal};

    fn term(id: u32, name: &str) -> Terminal {
        Terminal::Literal { id, bytes: name.as_bytes().to_vec() }
    }

    #[test]
    fn test_glr_grammar_simple() {
        let g = AnalyzedGrammar::from_grammar_def(&simple_ab_grammar());
        
        assert_eq!(g.rules.len(), 2);
        assert_eq!(g.num_nonterminals, 2); 
        assert_eq!(g.num_terminals, 2);
        assert!(g.nullable.is_empty());
        
        assert!(g.first[0].contains(&0));
        assert!(!g.first[0].contains(&1));
        
        assert!(g.follow[0].contains(&EOF));
    }

    #[test]
    fn test_glr_grammar_choice() {
        let g = AnalyzedGrammar::from_grammar_def(&choice_grammar());
        
        assert!(g.first[0].contains(&0));
        assert!(g.first[0].contains(&1));
    }

    #[test]
    fn test_glr_grammar_two_nt() {
        let g = AnalyzedGrammar::from_grammar_def(&two_nt_grammar());
        
        assert!(g.first[0].contains(&0)); 
        assert!(g.first[1].contains(&0)); 
        
        assert!(g.follow[1].contains(&1)); 
    }

    /// Simple non-recursive grammar passes all checks.
    #[test]
    fn test_preconditions_simple_grammar_passes() {
        let g = AnalyzedGrammar::from_grammar_def(&simple_ab_grammar());
        assert!(g.debug_check_grammar_preconditions().is_ok());
    }

    /// Grammar with nullable nonterminal is flagged.
    #[test]
    fn test_preconditions_nullable_detected() {
        // S -> A | ε  ;  A -> 'a'
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
                Rule { lhs: 0, rhs: vec![] }, // S -> ε
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },
            ],
            start: 0,
            terminals: vec![term(0, "a")],
            ..Default::default()
        };
        let g = AnalyzedGrammar::from_grammar_def(&gdef);
        let result = g.debug_check_grammar_preconditions();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("nullable"));
    }

    /// Grammar with right recursion is flagged.
    #[test]
    fn test_preconditions_right_recursion_detected() {
        // S -> 'a' S | 'a'
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(0)] },
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] },
            ],
            start: 0,
            terminals: vec![term(0, "a")],
            ..Default::default()
        };
        let g = AnalyzedGrammar::from_grammar_def(&gdef);
        let result = g.debug_check_grammar_preconditions();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("right-recursive"));
    }

    /// Grammar with indirect left recursion is flagged.
    #[test]
    fn test_preconditions_indirect_left_recursion_detected() {
        // A -> B 'a'  ;  B -> A 'b'
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(0)] },
                Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(1)] },
            ],
            start: 0,
            terminals: vec![term(0, "a"), term(1, "b")],
            ..Default::default()
        };
        let g = AnalyzedGrammar::from_grammar_def(&gdef);
        let result = g.debug_check_grammar_preconditions();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("indirect left-recursive"));
    }

    /// Direct left recursion (S -> S 'a' | 'a') should NOT be flagged as
    /// "indirect left recursion" — it is safe for GLR.
    #[test]
    fn test_preconditions_direct_left_recursion_ok() {
        // S -> S 'a' | 'a'  (direct LR, should only flag nullable if empty alt is absent)
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(0)] },
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] },
            ],
            start: 0,
            terminals: vec![term(0, "a")],
            ..Default::default()
        };
        let g = AnalyzedGrammar::from_grammar_def(&gdef);
        let result = g.debug_check_grammar_preconditions();
        // Should NOT contain "indirect left-recursive"
        match &result {
            Ok(()) => {} // fine — no violations
            Err(msg) => assert!(!msg.contains("indirect left-recursive"),
                "direct left recursion should not be flagged as indirect: {}", msg),
        }
    }

    // ── Normalization tests ──────────────────────────────────────────────

    /// inline_null_productions: simple ε-rule is removed, power-set
    /// variants are generated.
    #[test]
    fn test_inline_null_productions_basic() {
        // A -> B 'a'  ;  B -> 'b' | ε
        let rules = vec![
            Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(0)] },
            Rule { lhs: 1, rhs: vec![Symbol::Terminal(1)] },
            Rule { lhs: 1, rhs: vec![] }, // B -> ε
        ];
        let result = inline_null_productions(&rules, 2);
        // B -> ε is removed; A -> B 'a' spawns A -> 'a' (B omitted)
        assert!(result.iter().all(|r| !r.rhs.is_empty()), "no ε-rules in output");
        assert!(result.iter().any(|r| r.lhs == 0 && r.rhs == vec![Symbol::Terminal(0)]),
            "A -> 'a' should be generated");
        assert!(result.iter().any(|r| r.lhs == 0 && r.rhs == vec![Symbol::Nonterminal(1), Symbol::Terminal(0)]),
            "original A -> B 'a' should still be present");
    }

    /// inline_null_productions: no-op when nothing is nullable.
    #[test]
    fn test_inline_null_productions_nothing_nullable() {
        let rules = vec![
            Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)] },
        ];
        let result = inline_null_productions(&rules, 1);
        assert_eq!(result, rules);
    }

    /// normalize_grammar: an ε-producing grammar is cleaned up so the
    /// analysed grammar passes all precondition checks.
    #[test]
    fn test_normalize_removes_nullables() {
        // S -> A ; A -> ε | 'a'
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
                Rule { lhs: 1, rhs: vec![] },
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },
            ],
            start: 0,
            terminals: vec![term(0, "a")],
            ..Default::default()
        };
        let mut rules = gdef.rules.clone();
        normalize_grammar(&mut rules, gdef.start);
        let norm = GrammarDef {
            rules,
            start: gdef.start,
            terminals: gdef.terminals.clone(),
            ..Default::default()
        };
        let g = AnalyzedGrammar::from_grammar_def(&norm);
        assert!(g.check_no_nullable_nonterminals().is_ok(),
            "after normalization no NT should be nullable");
    }

    /// normalize_grammar: right recursion is converted to left recursion.
    #[test]
    fn test_normalize_eliminates_right_recursion() {
        // S -> 'a' S | 'a'
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(0)] },
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] },
            ],
            start: 0,
            terminals: vec![term(0, "a")],
            ..Default::default()
        };
        let mut rules = gdef.rules.clone();
        normalize_grammar(&mut rules, gdef.start);
        let norm = GrammarDef {
            rules,
            start: gdef.start,
            terminals: gdef.terminals.clone(),
            ..Default::default()
        };
        let g = AnalyzedGrammar::from_grammar_def(&norm);
        assert!(g.check_recursion_boundedness().is_ok(),
            "after normalization there should be no right recursion");
    }

    /// normalize_grammar produces a grammar that passes ALL precondition
    /// checks for a grammar that originally violated all three.
    #[test]
    fn test_normalize_full_pipeline() {
        // S -> A ; A -> B 'a' | ε ; B -> 'b' A (indirect RR via A→B→A)
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
                Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(2), Symbol::Terminal(0)] },
                Rule { lhs: 1, rhs: vec![] }, // A -> ε
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(1), Symbol::Nonterminal(1)] },
            ],
            start: 0,
            terminals: vec![term(0, "a"), term(1, "b")],
            ..Default::default()
        };
        // Before normalization, the grammar has nullable NTs
        let g_before = AnalyzedGrammar::from_grammar_def(&gdef);
        assert!(g_before.debug_check_grammar_preconditions().is_err());

        let mut rules = gdef.rules.clone();
        normalize_grammar(&mut rules, gdef.start);
        let norm = GrammarDef {
            rules,
            start: gdef.start,
            terminals: gdef.terminals.clone(),
            ..Default::default()
        };
        let g = AnalyzedGrammar::from_grammar_def(&norm);
        assert!(g.debug_check_grammar_preconditions().is_ok(),
            "normalized grammar should pass all precondition checks");
    }

    /// Split checks: check_no_nullable_nonterminals returns Ok for non-nullable grammar.
    #[test]
    fn test_split_check_nullable_ok() {
        let g = AnalyzedGrammar::from_grammar_def(&simple_ab_grammar());
        assert!(g.check_no_nullable_nonterminals().is_ok());
    }

    /// Split checks: check_recursion_boundedness returns Ok for non-recursive grammar.
    #[test]
    fn test_split_check_recursion_ok() {
        let g = AnalyzedGrammar::from_grammar_def(&simple_ab_grammar());
        assert!(g.check_recursion_boundedness().is_ok());
    }
}
