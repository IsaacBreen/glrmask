use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::analysis::EOF;
use super::table::{Action, GLRTable};
use crate::compiler::grammar::model::TerminalID;
use crate::ds::bitset::BitSet;
use crate::ds::leveled_gss::{LeveledGSS, LeveledGSSSummary, Merge};
use smallvec::SmallVec;

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
    pub subtree_isolate_ns: u64,
    pub pop_cache_build_ns: u64,
    pub base_isolate_ns: u64,
    pub absorb_push_ns: u64,
    pub shift_top_values_ns: u64,
    pub bookkeeping_ns: u64,
}

fn finalize_advance_timing(
    metrics: &mut Option<&mut AdvanceStacksDebugMetrics>,
    started_at: Option<std::time::Instant>,
) {
    if let (Some(metrics), Some(started_at)) = (metrics.as_deref_mut(), started_at) {
        let measured = metrics.subtree_isolate_ns
            + metrics.pop_cache_build_ns
            + metrics.base_isolate_ns
            + metrics.absorb_push_ns
            + metrics.shift_top_values_ns;
        let elapsed = started_at.elapsed().as_nanos() as u64;
        metrics.bookkeeping_ns = elapsed.saturating_sub(measured);
    }
}

#[cfg(test)]
pub(crate) struct GLRParser {
    pub table: GLRTable,
    pub stack: ParserGSS,
}

#[cfg(test)]
impl GLRParser {
    pub(crate) fn new(table: GLRTable) -> Self {
        let stack = ParserGSS::from_stacks(&[(vec![0], BTreeMap::new())]);
        Self { table, stack }
    }

    pub(crate) fn step(&self, token: TerminalID) -> (Self, bool) {
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

    pub(crate) fn valid_terminals(&self) -> Vec<TerminalID> {
        valid_terminals_for_stacks(&self.table, &self.stack)
    }
}

#[cfg(test)]
fn dedup_stacks(stacks: impl IntoIterator<Item = Vec<u32>>) -> Vec<Vec<u32>> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for stack in stacks {
        if seen.insert(stack.clone()) {
            out.push(stack);
        }
    }
    out
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

#[cfg(test)]
fn advance_stack_vectors(
    table: &GLRTable,
    stacks: &[Vec<u32>],
    token: TerminalID,
) -> Vec<Vec<u32>> {
    let closure = reduce_closure_for_lookahead(table, stacks, token);
    let mut next = Vec::new();
    for stack in closure {
        let Some(&state) = stack.last() else {
            continue;
        };
        match table.action(state, token) {
            Some(Action::Shift(target)) => {
                let mut shifted = stack.clone();
                shifted.push(*target);
                next.push(shifted);
            }
            Some(Action::Split { shift: Some(target), .. }) => {
                let mut shifted = stack.clone();
                shifted.push(*target);
                next.push(shifted);
            }
            _ => {}
        }
    }
    dedup_stacks(next)
}

fn stacks_accept(table: &GLRTable, stacks: &[Vec<u32>]) -> bool {
    reduce_closure_for_lookahead(table, stacks, EOF)
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

#[cfg(test)]
fn valid_terminals_for_stack_vectors(
    table: &GLRTable,
    stacks: &[Vec<u32>],
) -> Vec<TerminalID> {
    (0..table.num_terminals)
        .filter(|&terminal| !advance_stack_vectors(table, stacks, terminal).is_empty())
        .collect()
}

#[cfg(test)]
pub(crate) fn advance_stacks(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> ParserGSS {
    advance_stacks_with_metrics(table, stack, token, None)
}

pub(crate) fn stack_may_advance_on(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> bool {
    stack.peek_values().into_iter().any(|state| {
        matches!(
            table.action(state, token),
            Some(Action::Shift(_))
                | Some(Action::Reduce(_))
                | Some(Action::Split { .. })
                | Some(Action::Accept)
        )
    })
}

/// Returns true if any terminal in the given bitset may advance the parser stack,
/// or if the parser has a Reduce/Accept action on EOF (since reductions may
/// transition to states that can then shift on future terminals).
pub(crate) fn stack_may_advance_on_any(
    table: &GLRTable,
    stack: &ParserGSS,
    terminals: &BitSet,
) -> bool {
    stack.peek_values().into_iter().any(|state| {
        if let Some(actions_for_state) = table.action.get(state as usize) {
            actions_for_state.keys().any(|&terminal| {
                let relevant = terminals.contains(terminal as usize) || terminal == EOF;
                relevant
                    && matches!(
                        actions_for_state.get(&terminal),
                        Some(Action::Shift(_))
                            | Some(Action::Reduce(_))
                            | Some(Action::Split { .. })
                            | Some(Action::Accept)
                    )
            })
        } else {
            false
        }
    })
}

pub(crate) fn advance_stacks_with_metrics(
    table: &GLRTable,
    stack: &ParserGSS,
    token: TerminalID,
    mut metrics: Option<&mut AdvanceStacksDebugMetrics>,
) -> ParserGSS {
    let t_total = metrics.as_ref().map(|_| std::time::Instant::now());

    if let Some(state) = stack.single_exclusive_top_value() {
        let out = match table.action(state, token) {
            Some(Action::Shift(target)) => {
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.input_summary = stack.summary();
                    metrics.shift_state_candidates = 1;
                    metrics.shift_targets_hit = 1;
                    metrics.shifted_results = 1;
                }
                stack.push(*target)
            }
            Some(Action::Split {
                shift: Some(target),
                reduces,
                accept,
            }) if reduces.is_empty() && !*accept => {
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.input_summary = stack.summary();
                    metrics.shift_state_candidates = 1;
                    metrics.shift_targets_hit = 1;
                    metrics.shifted_results = 1;
                }
                stack.push(*target)
            }
            Some(Action::Reduce(_))
            | Some(Action::Accept)
            | Some(Action::Split { .. }) => ParserGSS::empty(),
            None => {
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.input_summary = stack.summary();
                    metrics.output_summary = LeveledGSSSummary::default();
                }
                finalize_advance_timing(&mut metrics, t_total);
                return ParserGSS::empty();
            }
        };
        if let Some(metrics) = metrics.as_deref_mut() {
            metrics.output_summary = out.summary();
        }
        if !out.is_empty() {
            finalize_advance_timing(&mut metrics, t_total);
            return out;
        }
    }

    let frontier = stack.peek_values();
    if frontier.is_empty() {
        if let Some(metrics) = metrics.as_deref_mut() {
            metrics.input_summary = stack.summary();
            metrics.output_summary = LeveledGSSSummary::default();
        }
        finalize_advance_timing(&mut metrics, t_total);
        return ParserGSS::empty();
    }

    let mut pure_shift_targets = SmallVec::<[(u32, u32); 8]>::new();
    let mut pure_shift_only = true;
    let mut any_action = false;
    for state in frontier.iter().copied() {
        match table.action(state, token) {
            Some(Action::Shift(target)) => {
                any_action = true;
                pure_shift_targets.push((state, *target));
            }
            Some(Action::Split {
                shift: Some(target),
                reduces,
                accept,
            }) if reduces.is_empty() && !*accept => {
                any_action = true;
                pure_shift_targets.push((state, *target));
            }
            Some(Action::Reduce(_))
            | Some(Action::Accept)
            | Some(Action::Split { .. }) => {
                any_action = true;
                pure_shift_only = false;
                break;
            }
            None => {}
        }
    }
    if !any_action {
        if let Some(metrics) = metrics.as_deref_mut() {
            metrics.input_summary = stack.summary();
            metrics.output_summary = LeveledGSSSummary::default();
        }
        finalize_advance_timing(&mut metrics, t_total);
        return ParserGSS::empty();
    }
    if pure_shift_only && !pure_shift_targets.is_empty() {
        if let Some(metrics) = metrics.as_deref_mut() {
            metrics.input_summary = stack.summary();
            metrics.shift_state_candidates = frontier.len();
        }
        let shifted_result_count = pure_shift_targets.len();
        let t_shift = metrics
            .as_ref()
            .map(|_| std::time::Instant::now());
        let out = stack.shift_top_values(pure_shift_targets);
        if let Some(metrics) = metrics.as_deref_mut() {
            metrics.shift_targets_hit = shifted_result_count;
            metrics.shifted_results = shifted_result_count;
        }
        if let (Some(metrics), Some(t_shift)) = (metrics.as_deref_mut(), t_shift) {
            metrics.shift_top_values_ns += t_shift.elapsed().as_nanos() as u64;
        }
        if let Some(metrics) = metrics.as_deref_mut() {
            metrics.output_summary = out.summary();
        }
        finalize_advance_timing(&mut metrics, t_total);
        return out;
    }

    // Reduce closure: iteratively apply all reduce actions on the GSS directly.
    let mut current = stack.clone();
    let mut processed = vec![false; table.num_states as usize];

    if let Some(metrics) = metrics.as_deref_mut() {
        metrics.input_summary = stack.summary();
    }

    loop {
        let mut new_states = SmallVec::<[u32; 8]>::new();
        if let Some(state) = current.single_top_value() {
            if !processed[state as usize] {
                new_states.push(state);
            }
        } else {
            new_states.extend(
                current
                    .peek_values()
                    .into_iter()
                    .filter(|&state| !processed[state as usize]),
            );
        }
        if new_states.is_empty() {
            break;
        }

        if let Some(metrics) = metrics.as_deref_mut() {
            metrics.reduce_closure_iterations += 1;
            metrics.frontier_states_total += new_states.len();
            metrics.frontier_states_max = metrics.frontier_states_max.max(new_states.len());
        }

        let mut any_reduced = false;
        let mut pending_bases_by_target = SmallVec::<[(u32, ParserGSS); 8]>::new();
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
            let t_subtree = metrics
                .as_ref()
                .map(|_| std::time::Instant::now());
            let subtree = current.isolate(Some(state));
            if let (Some(metrics), Some(t_subtree)) = (metrics.as_deref_mut(), t_subtree) {
                metrics.subtree_isolate_ns += t_subtree.elapsed().as_nanos() as u64;
            }
            let mut base_cache = SmallVec::<[((usize, u32), ParserGSS); 4]>::new();
            for &rule_id in reduce_rules {
                let rule = &table.rules[rule_id as usize];
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.popn_calls += 1;
                }
                let rhs_len = rule.rhs.len();
                let popped = subtree.popn(rhs_len as isize);
                if popped.is_empty() {
                    continue;
                }
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.popn_nonempty += 1;
                }
                let mut handle_goto_from = |goto_from: u32,
                                            metrics: &mut Option<&mut AdvanceStacksDebugMetrics>| {
                    if let Some(metrics) = metrics.as_deref_mut() {
                        metrics.goto_lookups += 1;
                    }
                    if let Some(target) = table.goto_target(goto_from, rule.lhs) {
                        let base = if let Some((_, cached)) = base_cache.iter().find(
                            |((cached_rhs_len, cached_goto_from), _)| {
                                *cached_rhs_len == rhs_len && *cached_goto_from == goto_from
                            },
                        ) {
                            cached.clone()
                        } else {
                            let t_base_isolate = metrics
                                .as_ref()
                                .map(|_| std::time::Instant::now());
                            let isolated = popped.isolate(Some(goto_from));
                            if let (Some(metrics), Some(t_base_isolate)) =
                                (metrics.as_deref_mut(), t_base_isolate)
                            {
                                metrics.base_isolate_ns +=
                                    t_base_isolate.elapsed().as_nanos() as u64;
                            }
                            base_cache.push(((rhs_len, goto_from), isolated.clone()));
                            isolated
                        };
                        if let Some((_, existing)) = pending_bases_by_target
                            .iter_mut()
                            .find(|(existing_target, _)| *existing_target == target)
                        {
                            *existing = existing.merge(&base);
                        } else {
                            pending_bases_by_target.push((target, base));
                        }
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
                };

                if let Some(goto_from) = popped.single_top_value() {
                    handle_goto_from(goto_from, &mut metrics);
                } else {
                    for goto_from in popped.peek_values() {
                        handle_goto_from(goto_from, &mut metrics);
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
            let t_absorb = metrics
                .as_ref()
                .map(|_| std::time::Instant::now());
            current = current.absorb_push(target, &base);
            if let (Some(metrics), Some(t_absorb)) = (metrics.as_deref_mut(), t_absorb) {
                metrics.absorb_push_ns += t_absorb.elapsed().as_nanos() as u64;
            }
        }
    }

    // Shift phase: for each state with a shift action, push the target.
    let mut shift_pairs = SmallVec::<[(u32, u32); 8]>::new();
    let mut handle_shift_state = |state: u32, metrics: &mut Option<&mut AdvanceStacksDebugMetrics>| {
        if let Some(metrics) = metrics.as_deref_mut() {
            metrics.shift_state_candidates += 1;
        }
        let shift_target = match table.action(state, token) {
            Some(Action::Shift(target)) => Some(*target),
            Some(Action::Split { shift: Some(target), .. }) => Some(*target),
            _ => None,
        };
        if let Some(target) = shift_target {
            shift_pairs.push((state, target));
            if let Some(metrics) = metrics.as_deref_mut() {
                metrics.shift_targets_hit += 1;
                metrics.shifted_results += 1;
            }
        }
    };

    if let Some(state) = current.single_top_value() {
        handle_shift_state(state, &mut metrics);
    } else {
        for state in current.peek_values() {
            handle_shift_state(state, &mut metrics);
        }
    }
    let t_shift = metrics
        .as_ref()
        .map(|_| std::time::Instant::now());
    let out = current.shift_top_values(shift_pairs);
    if let (Some(metrics), Some(t_shift)) = (metrics.as_deref_mut(), t_shift) {
        metrics.shift_top_values_ns += t_shift.elapsed().as_nanos() as u64;
    }
    if let Some(metrics) = metrics.as_deref_mut() {
        metrics.output_summary = out.summary();
    }
    finalize_advance_timing(&mut metrics, t_total);
    out
}

pub(crate) fn stacks_finished(table: &GLRTable, stack: &ParserGSS) -> bool {
    let stacks: Vec<Vec<u32>> = stack.to_stacks().into_iter().map(|(stack, _)| stack).collect();
    stacks_accept(table, &stacks)
}

#[cfg(test)]
pub(crate) fn valid_terminals_for_stacks(table: &GLRTable, stack: &ParserGSS) -> Vec<TerminalID> {
    let stacks: Vec<Vec<u32>> = stack.to_stacks().into_iter().map(|(stack, _)| stack).collect();
    valid_terminals_for_stack_vectors(table, &stacks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::grammar::model::tests::*;
    use crate::compiler::grammar::model::{GrammarDef, Rule, Symbol, Terminal};

    fn build_parser(gdef: &GrammarDef) -> GLRParser {
        let grammar = AnalyzedGrammar::from_grammar_def(gdef);
        let table = GLRTable::build(&grammar);
        GLRParser::new(table)
    }

    fn make_grammar(rules: Vec<Rule>, start: u32, terminals: Vec<Terminal>) -> GrammarDef {
        GrammarDef {
            rules,
            start,
            terminals,
            ..Default::default()
        }
    }

    fn accepts(parser: &GLRParser, input: &[TerminalID]) -> bool {
        let mut current = GLRParser {
            table: parser.table.clone(),
            stack: parser.stack.clone(),
        };
        for &token in input {
            let (next, progressed) = current.step(token);
            if !progressed {
                return false;
            }
            current = next;
        }
        stacks_finished(&current.table, &current.stack)
    }

    #[test]
    fn test_advance_stacks_preserves_accumulator_state() {
        let gdef = simple_ab_grammar();
        let grammar = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&grammar);

        let mut acc = BTreeMap::new();
        acc.insert(7, BTreeSet::from([11]));
        let gss = ParserGSS::from_stacks(&[(vec![0], acc.clone())]);

        let advanced = advance_stacks(&table, &gss, 0);
        let stacks = advanced.to_stacks();

        assert_eq!(stacks.len(), 1);
        assert_eq!(stacks[0].1, acc);
    }

    #[test]
    fn test_parse_simple_ab() {
        let gdef = simple_ab_grammar(); 
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0, 1])); 
        assert!(!accepts(&parser, &[0])); 
        assert!(!accepts(&parser, &[1, 0])); 
        assert!(!accepts(&parser, &[])); 
    }

    #[test]
    fn test_parse_choice() {
        let gdef = choice_grammar(); 
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0])); 
        assert!(accepts(&parser, &[1])); 
        assert!(!accepts(&parser, &[0, 1])); 
        assert!(!accepts(&parser, &[])); 
    }

    #[test]
    fn test_parse_two_nt() {
        let gdef = two_nt_grammar(); 
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0, 1])); 
        assert!(!accepts(&parser, &[0])); 
        assert!(!accepts(&parser, &[1])); 
    }

    #[test]
    fn test_parse_ambiguous() {
        
        let gdef = make_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Nonterminal(0),
                        Symbol::Terminal(1),
                        Symbol::Nonterminal(0),
                    ],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            0,
            vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"+".to_vec(),
                },
            ],
        );
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0])); 
        assert!(accepts(&parser, &[0, 1, 0])); 
        assert!(accepts(&parser, &[0, 1, 0, 1, 0])); 
        assert!(!accepts(&parser, &[1])); 
        assert!(!accepts(&parser, &[0, 1])); 
    }

    #[test]
    fn test_parse_nullable() {
        
        let gdef = make_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![],
                }, 
            ],
            0,
            vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
        );
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[])); 
        assert!(accepts(&parser, &[0])); 
        assert!(!accepts(&parser, &[0, 0])); 
    }

    #[test]
    fn test_valid_terminals() {
        let gdef = simple_ab_grammar(); 
        let parser = build_parser(&gdef);
        let valid = parser.valid_terminals();
        assert!(valid.contains(&0)); 
        assert!(!valid.contains(&1)); 
    }

    fn tdef(id: u32, name: &str) -> Terminal {
        Terminal::Literal { id, bytes: name.as_bytes().to_vec() }
    }

    #[test]
    fn test_ported_glr_left_recursive() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(0)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(1)] },                          
            ],
            0,
            vec![tdef(0, "a"), tdef(1, "b")],
        );
        let parser = build_parser(&gdef);
        
        assert!(accepts(&parser, &[1]),       "\"b\" accepted");
        assert!(accepts(&parser, &[1, 0]),    "\"ba\" accepted");
        assert!(accepts(&parser, &[1, 0, 0]), "\"baa\" accepted");
        
        assert!(!accepts(&parser, &[0]),    "\"a\" rejected (must start with 'b')");
        assert!(!accepts(&parser, &[1, 1]), "\"bb\" rejected (two 'b's)");
    }

    #[test]
    fn test_ported_glr_right_recursive() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(0)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(1)] },                          
            ],
            0,
            vec![tdef(0, "a"), tdef(1, "b")],
        );
        let parser = build_parser(&gdef);
        
        assert!(accepts(&parser, &[1]),          "\"b\" accepted");
        assert!(accepts(&parser, &[0, 1]),       "\"ab\" accepted");
        assert!(accepts(&parser, &[0, 0, 1]),    "\"aab\" accepted");
        assert!(accepts(&parser, &[0, 0, 0, 1]), "\"aaab\" accepted");
        
        assert!(!accepts(&parser, &[0]),     "\"a\" rejected (must end in 'b')");
        assert!(!accepts(&parser, &[1, 0]),  "\"ba\" rejected");
        assert!(!accepts(&parser, &[1, 1]),  "\"bb\" rejected");
    }

    #[test]
    fn test_ported_glr_expression_grammar() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(1), Symbol::Nonterminal(1)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },                                               
                Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(2), Symbol::Nonterminal(2)] }, 
                Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(2)] },                                               
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(3), Symbol::Nonterminal(0), Symbol::Terminal(4)] },    
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },                                                  
            ],
            0,
            vec![tdef(0, "i"), tdef(1, "+"), tdef(2, "*"), tdef(3, "("), tdef(4, ")")],
        );
        let parser = build_parser(&gdef);
        
        assert!(accepts(&parser, &[0]),                   "\"i\" accepted");
        assert!(accepts(&parser, &[0, 1, 0]),             "\"i+i\" accepted");
        assert!(accepts(&parser, &[0, 2, 0]),             "\"i*i\" accepted");
        assert!(accepts(&parser, &[0, 1, 0, 2, 0]),       "\"i+i*i\" accepted");
        assert!(accepts(&parser, &[3, 0, 1, 0, 4, 2, 0]), "\"(i+i)*i\" accepted");
        
        assert!(!accepts(&parser, &[0, 1]),       "\"i+\" rejected (incomplete)");
        assert!(!accepts(&parser, &[0, 1, 1, 0]), "\"i++i\" rejected (invalid)");
        assert!(!accepts(&parser, &[]),           "\"\" rejected (empty)");
        assert!(!accepts(&parser, &[4]),          "\")\" rejected");
        assert!(!accepts(&parser, &[3, 0]),       "\"(i\" rejected (unclosed paren)");
    }

    #[test]
    fn test_ported_glr_reduce_reduce_conflict() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(2)] }, 
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },    
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },    
            ],
            0,
            vec![tdef(0, "x")],
        );
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0]),  "\"x\" accepted despite reduce/reduce conflict");
        assert!(!accepts(&parser, &[]), "\"\" rejected");
    }

    #[test]
    fn test_ported_glr_epsilon_ambiguity() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Nonterminal(2)] }, 
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },  
                Rule { lhs: 1, rhs: vec![] },                     
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },  
                Rule { lhs: 2, rhs: vec![] },                     
            ],
            0,
            vec![tdef(0, "x")],
        );
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[]),       "\"\" accepted (A→ε, B→ε)");
        assert!(accepts(&parser, &[0]),      "\"x\" accepted (A→x,B→ε or A→ε,B→x)");
        assert!(accepts(&parser, &[0, 0]),   "\"xx\" accepted (A→x, B→x)");
        assert!(!accepts(&parser, &[0, 0, 0]), "\"xxx\" rejected");
    }

    #[test]
    fn test_ported_glr_highly_ambiguous() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Nonterminal(0)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] },                             
            ],
            0,
            vec![tdef(0, "a")],
        );
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0]),       "\"a\" accepted");
        assert!(accepts(&parser, &[0, 0]),    "\"aa\" accepted");
        assert!(accepts(&parser, &[0, 0, 0]), "\"aaa\" accepted (many parse trees)");
        assert!(!accepts(&parser, &[]),       "\"\" rejected (S not nullable)");
    }

    #[test]
    fn test_ported_glr_nullable_before_terminal() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(0)] }, 
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(1)] }, 
                Rule { lhs: 1, rhs: vec![] },                    
            ],
            0,
            vec![tdef(0, "c"), tdef(1, "d")],
        );
        let parser = build_parser(&gdef);
        
        assert!(accepts(&parser, &[1, 0]), "\"dc\" accepted (A → d c)");
        assert!(accepts(&parser, &[0]),    "\"c\" accepted (A → ε c via B→ε)");
        
        assert!(!accepts(&parser, &[1]),   "\"d\" rejected (missing 'c')");
        assert!(!accepts(&parser, &[]),    "\"\" rejected (A always requires 'c')");
    }

    #[test]
    fn test_ported_glr_ambiguous_dangling_else() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1), Symbol::Terminal(2), Symbol::Nonterminal(0)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1), Symbol::Terminal(2), Symbol::Nonterminal(0), Symbol::Terminal(3), Symbol::Nonterminal(0)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(4)] }, 
            ],
            0,
            vec![tdef(0, "if"), tdef(1, "id"), tdef(2, "then"), tdef(3, "else"), tdef(4, "other")],
        );
        let parser = build_parser(&gdef);
        
        assert!(accepts(&parser, &[0, 1, 2, 0, 1, 2, 4, 3, 4]),
            "ambiguous 'if id then if id then other else other' should be accepted");
        
        assert!(accepts(&parser, &[4]),          "\"other\" accepted");
        assert!(accepts(&parser, &[0, 1, 2, 4]), "\"if id then other\" accepted");
        assert!(!accepts(&parser, &[0, 1, 2]),   "\"if id then\" rejected (incomplete)");
    }

    #[test]
    fn test_close_token_wrapper_family_causes_reduction_spike() {
        const OPEN: u32 = 0;
        const NUM: u32 = 1;
        const COMMA: u32 = 2;
        const CLOSE: u32 = 3;

        const START: u32 = 0;
        const BODY: u32 = 1;
        const TAIL_ELEM: u32 = 2;
        const TAIL_PACK: u32 = 3;
        const FIRST_WRAP: u32 = 10;
        const WRAPPER_COUNT: usize = 24;

        let mut rules = vec![
            Rule {
                lhs: START,
                rhs: vec![
                    Symbol::Terminal(OPEN),
                    Symbol::Terminal(NUM),
                    Symbol::Nonterminal(BODY),
                    Symbol::Terminal(CLOSE),
                ],
            },
            Rule {
                lhs: BODY,
                rhs: vec![Symbol::Nonterminal(TAIL_PACK)],
            },
            Rule {
                lhs: TAIL_ELEM,
                rhs: vec![Symbol::Terminal(COMMA), Symbol::Terminal(NUM)],
            },
            Rule {
                lhs: TAIL_PACK,
                rhs: vec![Symbol::Nonterminal(TAIL_ELEM)],
            },
            Rule {
                lhs: TAIL_PACK,
                rhs: vec![
                    Symbol::Nonterminal(TAIL_ELEM),
                    Symbol::Nonterminal(TAIL_ELEM),
                ],
            },
        ];

        for i in 0..WRAPPER_COUNT {
            let wrap_nt = FIRST_WRAP + i as u32;
            rules.push(Rule {
                lhs: wrap_nt,
                rhs: vec![Symbol::Nonterminal(TAIL_PACK)],
            });
            rules.push(Rule {
                lhs: BODY,
                rhs: vec![Symbol::Nonterminal(wrap_nt)],
            });
        }

        let gdef = make_grammar(
            rules,
            START,
            vec![tdef(OPEN, "["), tdef(NUM, "n"), tdef(COMMA, ","), tdef(CLOSE, "]")],
        );
        let parser = build_parser(&gdef);

        let mut current = GLRParser {
            table: parser.table.clone(),
            stack: parser.stack.clone(),
        };
        for &token in &[OPEN, NUM, COMMA, NUM, COMMA, NUM] {
            let (next, progressed) = current.step(token);
            assert!(progressed, "prefix token {token} should progress");
            current = next;
        }

        let mut metrics = AdvanceStacksDebugMetrics::default();
        let advanced =
            advance_stacks_with_metrics(&current.table, &current.stack, CLOSE, Some(&mut metrics));
        let fast_advanced = advance_stacks(&current.table, &current.stack, CLOSE);

        assert!(!advanced.is_empty(), "close token should remain parseable");
        assert_eq!(
            fast_advanced.to_stacks().into_iter().collect::<BTreeSet<_>>(),
            advanced.to_stacks().into_iter().collect::<BTreeSet<_>>(),
            "metrics and non-metrics advance paths should agree"
        );
        assert!(
            metrics.reductions_emitted >= WRAPPER_COUNT * 2,
            "expected wrapper family to trigger many reductions, got {}",
            metrics.reductions_emitted
        );
        assert!(
            metrics
                .reduce_rhs_len_emitted_counts
                .get(&1)
                .copied()
                .unwrap_or(0)
                >= WRAPPER_COUNT * 2,
            "expected unary wrapper reductions to dominate: {:?}",
            metrics.reduce_rhs_len_emitted_counts
        );
        assert!(
            metrics
                .reduce_rhs_len_emitted_counts
                .get(&2)
                .copied()
                .unwrap_or(0)
                >= 1,
            "expected the pair-packing rule to participate: {:?}",
            metrics.reduce_rhs_len_emitted_counts
        );

        let wrapper_reductions: usize = (0..WRAPPER_COUNT)
            .map(|i| {
                metrics
                    .reduce_lhs_emitted_counts
                    .get(&(FIRST_WRAP + i as u32))
                    .copied()
                    .unwrap_or(0)
            })
            .sum();
        assert!(
            wrapper_reductions >= WRAPPER_COUNT,
            "expected wrapper nonterminals to account for many reductions, got {wrapper_reductions}"
        );
    }
}
