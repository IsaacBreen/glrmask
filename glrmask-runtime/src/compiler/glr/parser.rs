use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::EOF;
use super::table::{Action, GLRTable};
use crate::compiler::grammar_def::TerminalID;
use crate::ds::leveled_gss::{LeveledGSS, LeveledGSSSummary, Merge};

pub type TerminalsDisallowed = BTreeMap<u32, BTreeSet<u32>>;

impl Merge for TerminalsDisallowed {
    fn merge(&self, other: &Self) -> Self {
        let mut merged = self.clone();
        for (state, terminals) in other {
            merged
                .entry(*state)
                .or_default()
                .extend(terminals.iter().copied());
        }
        merged
    }
}

pub type ParserGSS = LeveledGSS<u32, TerminalsDisallowed>;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct AdvanceStacksDebugMetrics {
    pub input_summary: LeveledGSSSummary,
    pub output_summary: LeveledGSSSummary,
    pub reduce_closure_iterations: usize,
    pub frontier_states_total: usize,
    pub frontier_states_max: usize,
    pub reduce_rules_considered: usize,
    pub popn_calls: usize,
    pub popn_nonempty: usize,
    pub goto_lookups: usize,
    pub goto_hits: usize,
    pub reductions_emitted: usize,
    pub absorb_targets: usize,
    pub shift_state_candidates: usize,
    pub shift_targets_hit: usize,
    pub shifted_results: usize,
    pub reduce_rule_considered_counts: BTreeMap<u32, usize>,
    pub reduce_rule_emitted_counts: BTreeMap<u32, usize>,
    pub reduce_rhs_len_emitted_counts: BTreeMap<usize, usize>,
    pub reduce_lhs_emitted_counts: BTreeMap<u32, usize>,
    pub reduce_state_emitted_counts: BTreeMap<u32, usize>,
    pub goto_from_counts: BTreeMap<u32, usize>,
    pub goto_target_counts: BTreeMap<u32, usize>,
}

#[allow(dead_code)]
pub struct GLRParser {
    pub table: GLRTable,
    pub stack: ParserGSS,
}

#[allow(dead_code)]
impl GLRParser {
    pub fn new(table: GLRTable) -> Self {
        let stack = ParserGSS::from_stacks(&[(vec![0], BTreeMap::new())]);
        Self { table, stack }
    }

    pub fn step(&self, token: TerminalID) -> (Self, bool) {
        let next_stack = advance_stacks(&self.table, &self.stack, token);
        let progressed = !next_stack.is_empty();
        (
            Self {
                table: self.table.clone(),
                stack: next_stack,
            },
            progressed,
        )
    }

    pub fn valid_terminals(&self) -> Vec<TerminalID> {
        valid_terminals_for_stacks(&self.table, &self.stack)
    }
}

fn reduce_closure_for_lookahead(
    table: &GLRTable,
    stacks: &[Vec<u32>],
    lookahead: TerminalID,
) -> Vec<Vec<u32>> {
    let mut visited = BTreeSet::new();
    let mut queue = VecDeque::new();

    for stack in stacks {
        if visited.insert(stack.clone()) {
            queue.push_back(stack.clone());
        }
    }

    while let Some(stack) = queue.pop_front() {
        let Some(&state) = stack.last() else {
            continue;
        };
        let Some(action) = table.action(state, lookahead) else {
            continue;
        };
        let reduce_rule_ids: &[u32] = match action {
            Action::Reduce(rule_id) => std::slice::from_ref(rule_id),
            Action::Split { reduces, .. } => reduces.as_slice(),
            Action::Shift(_) | Action::Accept => &[],
        };
        for rule_id in reduce_rule_ids {
            let rule = &table.rules[*rule_id as usize];
            if stack.len() < rule.rhs.len() + 1 {
                continue;
            }
            let keep_len = stack.len() - rule.rhs.len();
            let mut reduced = stack[..keep_len].to_vec();
            let Some(&goto_from) = reduced.last() else {
                continue;
            };
            let Some(target) = table.goto_target(goto_from, rule.lhs) else {
                continue;
            };
            reduced.push(target);
            if visited.insert(reduced.clone()) {
                queue.push_back(reduced);
            }
        }
    }

    visited.into_iter().collect()
}

pub(crate) fn advance_stacks(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> ParserGSS {
    advance_stacks_with_metrics(table, stack, token, None)
}

pub(crate) fn advance_stacks_with_metrics(
    table: &GLRTable,
    stack: &ParserGSS,
    token: TerminalID,
    mut metrics: Option<&mut AdvanceStacksDebugMetrics>,
) -> ParserGSS {
    let mut current = stack.clone();
    let mut processed = vec![false; table.num_states as usize];

    if let Some(metrics) = metrics.as_deref_mut() {
        metrics.input_summary = stack.summary();
    }

    loop {
        let frontier = current.peek_values();
        let new_states: Vec<u32> = frontier
            .iter()
            .filter(|&&state| !processed[state as usize])
            .copied()
            .collect();
        if new_states.is_empty() {
            break;
        }

        if let Some(metrics) = metrics.as_deref_mut() {
            metrics.reduce_closure_iterations += 1;
            metrics.frontier_states_total += new_states.len();
            metrics.frontier_states_max = metrics.frontier_states_max.max(new_states.len());
        }

        let mut any_reduced = false;
        let mut pending_bases_by_target = BTreeMap::<u32, ParserGSS>::new();
        for state in new_states {
            processed[state as usize] = true;
            let reduce_rules: &[u32] = match table.action(state, token) {
                Some(Action::Reduce(rule_id)) => std::slice::from_ref(rule_id),
                Some(Action::Split { reduces, .. }) => reduces.as_slice(),
                _ => &[],
            };
            if let Some(metrics) = metrics.as_deref_mut() {
                metrics.reduce_rules_considered += reduce_rules.len();
                for &rule_id in reduce_rules {
                    *metrics.reduce_rule_considered_counts.entry(rule_id).or_default() += 1;
                }
            }
            let subtree = current.isolate(Some(state));
            for &rule_id in reduce_rules {
                let rule = &table.rules[rule_id as usize];
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.popn_calls += 1;
                }
                let popped = subtree.popn(rule.rhs.len() as isize);
                if popped.is_empty() {
                    continue;
                }
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.popn_nonempty += 1;
                }
                for goto_from in popped.peek_values() {
                    if let Some(metrics) = metrics.as_deref_mut() {
                        metrics.goto_lookups += 1;
                    }
                    if let Some(target) = table.goto_target(goto_from, rule.lhs) {
                        let base = popped.isolate(Some(goto_from));
                        pending_bases_by_target
                            .entry(target)
                            .and_modify(|existing| *existing = existing.merge(&base))
                            .or_insert(base);
                        any_reduced = true;
                        if let Some(metrics) = metrics.as_deref_mut() {
                            metrics.goto_hits += 1;
                            metrics.reductions_emitted += 1;
                            *metrics.reduce_rule_emitted_counts.entry(rule_id).or_default() += 1;
                            *metrics
                                .reduce_rhs_len_emitted_counts
                                .entry(rule.rhs.len())
                                .or_default() += 1;
                            *metrics.reduce_lhs_emitted_counts.entry(rule.lhs).or_default() += 1;
                            *metrics.reduce_state_emitted_counts.entry(state).or_default() += 1;
                            *metrics.goto_from_counts.entry(goto_from).or_default() += 1;
                            *metrics.goto_target_counts.entry(target).or_default() += 1;
                        }
                    }
                }
            }
        }
        if !any_reduced {
            break;
        }
        if let Some(metrics) = metrics.as_deref_mut() {
            metrics.absorb_targets += pending_bases_by_target.len();
        }
        for (target, base) in pending_bases_by_target {
            current = current.absorb_push(target, &base);
        }
    }

    let mut shifted_results = Vec::new();
    for state in current.peek_values() {
        if let Some(metrics) = metrics.as_deref_mut() {
            metrics.shift_state_candidates += 1;
        }
        let shift_target = match table.action(state, token) {
            Some(Action::Shift(target)) => Some(*target),
            Some(Action::Split { shift: Some(target), .. }) => Some(*target),
            _ => None,
        };
        if let Some(target) = shift_target {
            let subtree = current.isolate(Some(state));
            shifted_results.push(subtree.push(target));
            if let Some(metrics) = metrics.as_deref_mut() {
                metrics.shift_targets_hit += 1;
                metrics.shifted_results += 1;
            }
        }
    }
    let out = ParserGSS::merge_many(shifted_results);
    if let Some(metrics) = metrics.as_deref_mut() {
        metrics.output_summary = out.summary();
    }
    out
}

pub(crate) fn stacks_finished(table: &GLRTable, stack: &ParserGSS) -> bool {
    let stacks: Vec<Vec<u32>> = stack.to_stacks().into_iter().map(|(stack, _)| stack).collect();
    reduce_closure_for_lookahead(table, &stacks, EOF)
        .into_iter()
        .any(|stack| {
            stack.last().is_some_and(|state| {
                matches!(
                    table.action(*state, EOF),
                    Some(Action::Accept) | Some(Action::Split { accept: true, .. })
                )
            })
        })
}

pub(crate) fn valid_terminals_for_stacks(table: &GLRTable, stack: &ParserGSS) -> Vec<TerminalID> {
    let stacks: Vec<Vec<u32>> = stack.to_stacks().into_iter().map(|(stack, _)| stack).collect();
    (0..table.num_terminals)
        .filter(|&terminal| !reduce_closure_for_lookahead(table, &stacks, terminal).is_empty())
        .collect()
}
