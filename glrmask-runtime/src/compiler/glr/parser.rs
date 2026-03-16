use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::EOF;
use super::table::{Action, GLRTable};
use crate::compiler::grammar_def::TerminalID;
use crate::ds::leveled_gss::{LeveledGSS, Merge};

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
    let mut current = stack.clone();
    let mut processed = vec![false; table.num_states as usize];

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

        let mut any_reduced = false;
        let mut pending_bases_by_target = BTreeMap::<u32, ParserGSS>::new();
        for state in new_states {
            processed[state as usize] = true;
            let reduce_rules: &[u32] = match table.action(state, token) {
                Some(Action::Reduce(rule_id)) => std::slice::from_ref(rule_id),
                Some(Action::Split { reduces, .. }) => reduces.as_slice(),
                _ => &[],
            };
            let subtree = current.isolate(Some(state));
            for &rule_id in reduce_rules {
                let rule = &table.rules[rule_id as usize];
                let popped = subtree.popn(rule.rhs.len() as isize);
                if popped.is_empty() {
                    continue;
                }
                for goto_from in popped.peek_values() {
                    if let Some(target) = table.goto_target(goto_from, rule.lhs) {
                        let base = popped.isolate(Some(goto_from));
                        pending_bases_by_target
                            .entry(target)
                            .and_modify(|existing| *existing = existing.merge(&base))
                            .or_insert(base);
                        any_reduced = true;
                    }
                }
            }
        }
        if !any_reduced {
            break;
        }
        for (target, base) in pending_bases_by_target {
            current = current.absorb_push(target, &base);
        }
    }

    let mut shifted_results = Vec::new();
    for state in current.peek_values() {
        let shift_target = match table.action(state, token) {
            Some(Action::Shift(target)) => Some(*target),
            Some(Action::Split { shift: Some(target), .. }) => Some(*target),
            _ => None,
        };
        if let Some(target) = shift_target {
            let subtree = current.isolate(Some(state));
            shifted_results.push(subtree.push(target));
        }
    }
    ParserGSS::merge_many(shifted_results)
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
