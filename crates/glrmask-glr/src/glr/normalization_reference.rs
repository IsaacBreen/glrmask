//! Independent ordered rewrite schedule for normalization tests.
//! The common CFG analyses, nullable expansion and exact deduplication
//! are shared with production; the recursive rewrites still use their
//! original owned-vector implementation to check publication and fresh IDs.

use super::analysis::{
    build_left_reachability_graph, build_rhs_by_lhs, build_right_reachability_graph,
    compute_nullable, dedup_rules, find_cycle_excluding_self_loops, find_indirect_lr_cycle,
    find_nontrivial_sccs, has_indirect_left_recursion_with_nullable,
    has_indirect_right_recursion_fast,
    inline_null_productions_unbudgeted as inline_null_productions, is_reflexive_unit_rule,
    max_nt_id, remove_unreachable_rules,
};
use crate::grammar::flat::{NonterminalID, Rule, Symbol};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::time::Instant;

fn compile_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn emit_normalize_profile(
    stage: &str,
    iteration: Option<usize>,
    elapsed_ms: f64,
    rules_before: usize,
    rules_after: usize,
    extra: &str,
) {
    match iteration {
        Some(iteration) => eprintln!(
            "[glrmask-profile] normalize_grammar stage={} iteration={} ms={:.3} rules_before={} rules_after={}{}",
            stage, iteration, elapsed_ms, rules_before, rules_after, extra,
        ),
        None => eprintln!(
            "[glrmask-profile] normalize_grammar stage={} ms={:.3} rules_before={} rules_after={}{}",
            stage, elapsed_ms, rules_before, rules_after, extra,
        ),
    }
}

/// Eliminate right recursion by first inlining indirect cycles and then
/// rewriting direct right recursion into left recursion.
pub fn eliminate_right_recursion(
    rules: &mut Vec<Rule>,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
) -> bool {
    // Resolve indirect right recursion by inlining right ends.
    const MAX_INDIRECT_ROUNDS: usize = 200;
    let num_nt = max_nt_id(rules) + 1;
    let nullable = compute_nullable(rules, num_nt);
    let mut completed = !has_indirect_right_recursion_fast(rules, &nullable, num_nt);
    if !completed {
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
                None => {
                    completed = true;
                    break;
                }
            }
        }
    }

    // Indirect substitution can introduce X -> X after earlier unit cleanup.
    // Removing these rules preserves finite terminal languages. Do so before
    // selecting helpers, so stripping a sole X cannot create an empty RHS.
    rules.retain(|rule| !is_reflexive_unit_rule(rule));

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
    completed
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
                recursive_by_nt
                    .entry(rule.lhs)
                    .or_default()
                    .push(rule.clone());
            } else {
                non_recursive_by_nt
                    .entry(rule.lhs)
                    .or_default()
                    .push(rule.clone());
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
            new_rules.push(Rule {
                lhs: new_nt,
                rhs: body,
            });
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

/// Eliminate hidden left recursion.
///
/// Hidden left recursion occurs when `A → β B …` where every symbol in
/// `β` is nullable and `B` is in an indirect left-recursive cycle with `A`.
/// We add shortened rules with the nullable prefix removed, exposing the
/// left recursion so it becomes direct (which GLR handles natively).
fn eliminate_hidden_left_recursion(
    rules: &mut Vec<Rule>,
    nullable: &BTreeSet<NonterminalID>,
    normalize_iteration: usize,
) -> bool {
    let profile_enabled = compile_profile_enabled();
    let mut changed = false;
    let mut refreshed_nullable = None;

    loop {
        let nullable = refreshed_nullable.as_ref().unwrap_or(nullable);
        let rules_len = rules.len();
        let lr_graph_started_at = profile_enabled.then(Instant::now);
        let lr_graph = build_left_reachability_graph(rules, nullable);
        let build_left_reachability_graph_ms = lr_graph_started_at.map(elapsed_ms).unwrap_or(0.0);
        let left_reachability_node_count = lr_graph.len();
        let left_reachability_edge_count: usize = lr_graph.values().map(BTreeSet::len).sum();
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
            return changed;
        }
        if let Some(repaired) = super::left_corner_closure::repair(rules, &cycle_sccs, nullable) {
            *rules = repaired;
            refreshed_nullable = Some(compute_nullable(rules, max_nt_id(rules) + 1));
            changed = true;
            continue;
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
                if let Some(Symbol::Nonterminal(nt)) = suffix.first()
                    && cycle_nodes.contains(nt)
                {
                    additions.push(Rule {
                        lhs: rule.lhs,
                        rhs: suffix.to_vec(),
                    });
                }
            }

            let first_after_nullable = rule.rhs.get(prefix_end);
            if let Some(Symbol::Nonterminal(next_nt)) = first_after_nullable
                && *next_nt != rule.lhs
                && cycle_nodes.contains(next_nt)
            {
                let expansions =
                    expand_cycle_head_paths(&rhs_by_lhs, cycle_nodes, *next_nt, rule.lhs);
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

        let existing: HashSet<Rule> = rules.iter().cloned().collect();
        let mut unique_additions = HashSet::new();
        additions.retain(|rule| !existing.contains(rule) && unique_additions.insert(rule.clone()));

        if !replaced_cycle_rules.is_empty() {
            rules.retain(|rule| !replaced_cycle_rules.contains(rule));
            changed = true;
        }

        if additions.is_empty() {
            return changed;
        }
        rules.extend(additions);
        changed = true;
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
                        expand(rhs_by_lhs, cycle_nodes, *head, goal, visited, &mut nested);
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

/// Run the full grammar normalization pipeline (in place).
///
/// Mutates `rules` so that they satisfy the characterization
/// preconditions (no nullable NTs, no right recursion, no indirect LR).
/// `start` is used only for unreachable-production pruning and is never
/// changed.
pub fn normalize_grammar(rules: &mut Vec<Rule>, start: NonterminalID) {
    if !rules.iter().any(|rule| rule.lhs == start) {
        rules.clear();
        return;
    }
    use std::cell::Cell;

    let profiling = compile_profile_enabled();
    let next_nt = Cell::new(max_nt_id(rules) + 1);
    let mut fresh_nt = || {
        let id = next_nt.get();
        next_nt.set(id + 1);
        id
    };

    let mut iteration = 0;
    let mut nullable_eliminated_before_exit = false;
    loop {
        let iteration_rules_before = rules.len();
        let iteration_started_at = profiling.then(Instant::now);

        let clone_started_at = profiling.then(Instant::now);
        let snap = rules.clone();
        if let Some(started_at) = clone_started_at {
            emit_normalize_profile(
                "clone_snapshot",
                Some(iteration + 1),
                elapsed_ms(started_at),
                iteration_rules_before,
                snap.len(),
                "",
            );
        }

        let inline_rules_before = rules.len();
        let inline_started_at = profiling.then(Instant::now);
        replace_rules_with_resync(rules, &next_nt, inline_null_productions);
        // A -> A contributes no terminal derivation. Epsilon elimination can
        // expose it, and rewriting it as right recursion would manufacture a
        // fresh nullable self-cycle on every subsequent iteration.
        rules.retain(|rule| !is_reflexive_unit_rule(rule));
        // Reflexive-unit cleanup can remove the last start production.
        if !rules.iter().any(|rule| rule.lhs == start) {
            rules.clear();
            return;
        }
        let inline_unchanged = *rules == snap;
        if let Some(started_at) = inline_started_at {
            emit_normalize_profile(
                "inline_null_productions",
                Some(iteration + 1),
                elapsed_ms(started_at),
                inline_rules_before,
                rules.len(),
                &format!(" phase=fixed_point changed={}", !inline_unchanged),
            );
        }

        let rr_rules_before = rules.len();
        let rr_started_at = profiling.then(Instant::now);
        let right_recursion_completed = with_resynced_next_nonterminal(rules, &next_nt, |rules| {
            eliminate_right_recursion(rules, &mut fresh_nt)
        });
        // RR can also remove all start productions. Stop before HLR allocates helpers.
        if !rules.iter().any(|rule| rule.lhs == start) {
            rules.clear();
            return;
        }
        if let Some(started_at) = rr_started_at {
            emit_normalize_profile(
                "eliminate_right_recursion",
                Some(iteration + 1),
                elapsed_ms(started_at),
                rr_rules_before,
                rules.len(),
                "",
            );
        }

        let hlr_rules_before = rules.len();
        let hlr_started_at = profiling.then(Instant::now);
        let mut nullable_count = 0usize;
        let hidden_left_recursion_changed =
            with_resynced_next_nonterminal(rules, &next_nt, |rules| {
                let nullable = compute_nullable(rules, max_nt_id(rules) + 1);
                nullable_count = nullable.len();
                // With no nullable prefix, the pass still resolves ordinary
                // indirect left recursion. Avoid its more expensive SCC and rule
                // expansion work when a cheap reachability check proves there is
                // no such cycle.
                if nullable.is_empty()
                    && !has_indirect_left_recursion_with_nullable(rules, &nullable)
                {
                    false
                } else {
                    eliminate_hidden_left_recursion(rules, &nullable, iteration + 1)
                }
            });

        if let Some(started_at) = hlr_started_at {
            emit_normalize_profile(
                "compute_nullable_and_eliminate_hidden_left_recursion",
                Some(iteration + 1),
                elapsed_ms(started_at),
                hlr_rules_before,
                rules.len(),
                &format!(" nullable_count={nullable_count}"),
            );
        }

        let dedup_rules_before = rules.len();
        let dedup_started_at = profiling.then(Instant::now);
        dedup_rules(rules);

        if let Some(started_at) = dedup_started_at {
            emit_normalize_profile(
                "dedup_rules",
                Some(iteration + 1),
                elapsed_ms(started_at),
                dedup_rules_before,
                rules.len(),
                " phase=fixed_point",
            );
        }

        let equality_started_at = profiling.then(Instant::now);
        let converged = *rules == snap;
        if let Some(started_at) = equality_started_at {
            emit_normalize_profile(
                "equality_convergence_check",
                Some(iteration + 1),
                elapsed_ms(started_at),
                rules.len(),
                snap.len(),
                &format!(" converged={converged}"),
            );
        }

        if let Some(started_at) = iteration_started_at {
            emit_normalize_profile(
                "fixed_point_iteration_total",
                Some(iteration + 1),
                elapsed_ms(started_at),
                iteration_rules_before,
                rules.len(),
                &format!(" converged={converged}"),
            );
        }

        if compute_nullable(rules, max_nt_id(rules) + 1).is_empty()
            && right_recursion_completed
            && !hidden_left_recursion_changed
        {
            // Check the rewritten grammar: right-recursion elimination can
            // introduce nullable helpers even when its input had none.
            // Post-merge transforms perform their own indirect-LR recovery.
            nullable_eliminated_before_exit = true;
            break;
        }
        iteration += 1;

        if converged {
            break;
        }
    }

    let post_inline_rules_before = rules.len();
    let post_inline_snapshot = (!nullable_eliminated_before_exit).then(|| rules.clone());
    let post_inline_started_at = profiling.then(Instant::now);
    if !nullable_eliminated_before_exit {
        replace_rules_with_resync(rules, &next_nt, inline_null_productions);
        rules.retain(|rule| !is_reflexive_unit_rule(rule));
    }
    let post_inline_changed = post_inline_snapshot
        .as_ref()
        .is_some_and(|snapshot| *rules != *snapshot);
    if let Some(started_at) = post_inline_started_at {
        let post_inline_extra = if nullable_eliminated_before_exit {
            " phase=post_loop skipped=nullable_eliminated".to_string()
        } else {
            format!(" phase=post_loop changed={post_inline_changed}")
        };
        emit_normalize_profile(
            "inline_null_productions",
            None,
            elapsed_ms(started_at),
            post_inline_rules_before,
            rules.len(),
            &post_inline_extra,
        );
    }

    let unreachable_rules_before = rules.len();
    let unreachable_started_at = profiling.then(Instant::now);
    *rules = remove_unreachable_rules(rules, start);
    if let Some(started_at) = unreachable_started_at {
        emit_normalize_profile(
            "remove_unreachable_rules",
            None,
            elapsed_ms(started_at),
            unreachable_rules_before,
            rules.len(),
            "",
        );
    }

    let final_dedup_rules_before = rules.len();
    let final_dedup_started_at = profiling.then(Instant::now);
    if post_inline_changed {
        dedup_rules(rules);
    }
    if let Some(started_at) = final_dedup_started_at {
        emit_normalize_profile(
            "dedup_rules",
            None,
            elapsed_ms(started_at),
            final_dedup_rules_before,
            rules.len(),
            if post_inline_changed {
                " phase=final"
            } else {
                " phase=final skipped=post_inline_unchanged"
            },
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

fn resync_next_nonterminal(rules: &[Rule], next_nt: &std::cell::Cell<u32>) {
    next_nt.set(max_nt_id(rules) + 1);
}
fn with_resynced_next_nonterminal<R>(
    rules: &mut Vec<Rule>,
    next_nt: &std::cell::Cell<u32>,
    update: impl FnOnce(&mut Vec<Rule>) -> R,
) -> R {
    let result = update(rules);
    resync_next_nonterminal(rules, next_nt);
    result
}
