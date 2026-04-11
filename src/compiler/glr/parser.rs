#[cfg(test)]
use std::cell::Cell;
#[cfg(test)]
use std::collections::BTreeMap;
#[cfg(any(test, debug_assertions))]
use std::collections::BTreeSet;
#[cfg(any(test, debug_assertions))]
use std::collections::VecDeque;
#[cfg(test)]
use std::sync::Arc;

use super::accumulator::TerminalsDisallowed;
use super::analysis::EOF;
use super::table::{Action, GLRTable};
use crate::compiler::grammar::model::TerminalID;
use crate::ds::bitset::BitSet;
use crate::ds::leveled_gss::LeveledGSS;
use smallvec::SmallVec;

pub type ParserGSS = LeveledGSS<u32, TerminalsDisallowed>;

type ReduceSources = SmallVec<[(u32, ParserGSS); 4]>;
type GotoBatch = SmallVec<[(u32, ParserGSS); 8]>;


pub(crate) fn advance_stacks(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> ParserGSS {
    advance_stacks_core(table, stack.clone(), token)
}

/// Like `advance_stacks` but takes ownership of the GSS, avoiding an
/// unnecessary Arc clone when the caller doesn't need the original.
pub(crate) fn advance_stacks_owned(table: &GLRTable, stack: ParserGSS, token: TerminalID) -> ParserGSS {
    advance_stacks_core(table, stack, token)
}

/// Advance the GSS by one token.
///
/// First try the deterministic single-chain path: repeatedly reduce a flat LR
/// stack, and finish immediately if that path ends in a pure shift.
///
/// If the frontier is ambiguous, or the deterministic path stops without a
/// pure shift, fall back to the GLR path: build the reduce closure to a
/// fixpoint and return the shifted next frontier.
fn advance_stacks_core(table: &GLRTable, mut gss: ParserGSS, token: TerminalID) -> ParserGSS {
    // Fast path: single state with a pure shift action (most common case).
    if let Some(state) = gss.single_exclusive_top_value() {
        if let Some(Action::Shift(target)) = table.action(state, token) {
            return gss.push(*target);
        }
    }

    if advance_deterministically(table, &mut gss, token) {
        return gss;
    }

    advance_nondeterministically(table, gss, token)
}

fn shift_frontier(table: &GLRTable, gss: ParserGSS, token: TerminalID) -> ParserGSS {
    let mut shift_pairs = SmallVec::<[(u32, u32); 8]>::new();
    for state in gss.peek_values() {
        if let Some(target) = table.action(state, token).and_then(Action::shift_target) {
            shift_pairs.push((state, target));
        }
    }
    gss.remap_top_values_owned(shift_pairs)
}

fn apply_gotos(mut gss: ParserGSS, gotos: GotoBatch) -> ParserGSS {
    for (target, base) in gotos {
        gss = gss.absorb_push_same_acc(target, &base);
    }
    gss
}

fn add_goto(gotos: &mut GotoBatch, target: u32, base: ParserGSS) {
    if let Some((_, existing)) = gotos.iter_mut().find(|(t, _)| *t == target) {
        *existing = existing.merge(&base);
    } else {
        gotos.push((target, base));
    }
}

fn reduce_sources(gss: &ParserGSS, state: u32, rhs_len: usize) -> ReduceSources {
    gss.isolate_pop_bases(state, rhs_len as isize)
}

/// Advance an ambiguous frontier.
///
/// `closure` accumulates unshifted branches that still need GLR reduce-closure
/// processing. `shifted` accumulates branches that have already advanced past
/// the current token and therefore belong in the returned next frontier.
///
/// Each wave starts with a fresh `next` frontier. Shiftable isolated branches
/// are moved directly into `shifted`; newly reduced branches are merged into
/// `next` and become the closure for the next wave.
fn advance_nondeterministically(
    table: &GLRTable,
    mut closure: ParserGSS,
    token: TerminalID,
) -> ParserGSS {
    let mut shifted = ParserGSS::empty();

    loop {
        let mut next = ParserGSS::empty();

        for state in closure.peek_values() {
            let Some(action) = table.action(state, token) else {
                continue;
            };
            let mut isolated = closure.isolate(Some(state));
            if advance_deterministically(table, &mut isolated, token) {
                shifted = shifted.merge(&isolated);
                continue;
            }

            if let Some(target) = action.shift_target() {
                shifted = shifted.merge(&isolated.push(target));
            }

            for &rule_id in action.reduce_rule_ids() {
                let rule = &table.rules[rule_id as usize];
                for (goto_from, base) in reduce_sources(&closure, state, rule.rhs.len()) {
                    let Some(target) = table.goto_target(goto_from, rule.lhs) else {
                        continue;
                    };

                    let mut branch = base.push(target);
                    if advance_deterministically(table, &mut branch, token) {
                        shifted = shifted.merge(&branch);
                    } else {
                        next = next.merge(&branch);
                    }
                }
            }
        }

        if next.is_empty() {
            return shifted;
        }
        closure = next;
    }
}

/// Standard LR reduce loop for the deterministic case.
///
/// When the GSS frontier is a single linear chain (no ambiguity), the GSS
/// degenerates to an ordinary flat parse stack.  This applies the textbook
/// LR reduce loop directly: inspect the top state's action, pop |rhs|
/// symbols, push the goto target, repeat — until a non-reduce action is
/// reached or the chain becomes ambiguous.
///
/// If this deterministic pass ends in a pure shift, it performs that shift
/// itself and returns true to signal that the parser step is finished.
/// Otherwise it mutates `gss` and returns false so the caller can continue
/// with the nondeterministic reduce closure.
fn advance_deterministically(
    table: &GLRTable,
    gss: &mut ParserGSS,
    token: TerminalID,
) -> bool {
    let Some(mut stack) = gss.try_virtual_stack() else {
        return false; // Ambiguous frontier — skip to the general GLR path.
    };

    #[cfg(test)]
    note_vstack_hit();

    loop {
        let Some(&state) = stack.top() else {
            break;
        };

        match table.action(state, token) {
            Some(Action::Reduce(rule_id)) => {
                let rule = &table.rules[*rule_id as usize];
                if rule.rhs.len() < stack.len() {
                    // Pop |rhs| symbols and push the goto target.
                    stack.pop(rule.rhs.len());
                    let goto_from = *stack.top().unwrap();
                    match table.goto_target(goto_from, rule.lhs) {
                        Some(target) => stack.push(target),
                        None => {
                            *gss = ParserGSS::empty();
                            return false;
                        }
                    }
                } else {
                    // This reduce reaches or crosses the deterministic chain's
                    // floor. Finish it at the GSS level, batch the gotos, and
                    // keep going deterministically if the rebuilt frontier is
                    // still a single chain.
                    let current = stack.into_gss();
                    let popped = current.popn(rule.rhs.len() as isize);
                    let mut gotos = GotoBatch::new();
                    for goto_from in popped.peek_values() {
                        let base = popped.isolate(Some(goto_from));
                        if let Some(target) = table.goto_target(goto_from, rule.lhs) {
                            add_goto(&mut gotos, target, base);
                        }
                    }
                    let rebuilt = apply_gotos(current, gotos);
                    let Some(next_stack) = rebuilt.try_virtual_stack() else {
                        *gss = rebuilt;
                        return false;
                    };
                    stack = next_stack;
                }
            }
            Some(Action::Shift(target)) => {
                *gss = stack.into_gss().push(*target);
                return true;
            }
            Some(Action::Split { .. }) => {
                break;
            }
            Some(Action::Accept) => {
                break;
            }
            None => break,
        }
    }

    *gss = stack.into_gss();
    false
}

pub(crate) fn stack_may_advance_on(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> bool {
    stack.peek_values().into_iter().any(|state| table.action(state, token).is_some())
}

/// Profiled version of `advance_stacks_core`.
/// Returns (result_gss, profile) where profile contains detailed timing.
#[derive(Debug, Clone, Default)]
pub struct AdvanceProfile {
    pub pure_shift: bool,
    pub deterministic_entered: bool,
    pub deterministic_finished: bool,
    pub nondeterministic_entered: bool,
    pub vstack_len: u32,
    pub n_reduces_above_floor: u32,
    pub n_floor_crossings: u32,
    pub n_nondet_waves: u32,
    pub n_nondet_branches: u32,
    pub top_states: u32,
    pub gss_depth: u32,
    pub total_ns: u64,
    pub det_ns: u64,
    pub nondet_ns: u64,
    /// 0 = not entered, 1 = shift (finished), 2 = split, 3 = accept, 4 = no action, 5 = no top, 6 = vstack fail, 7 = floor cross vstack fail
    pub det_exit_reason: u32,
    pub det_exit_state: u32,
}

pub(crate) fn advance_stacks_profiled(
    table: &GLRTable,
    stack: &ParserGSS,
    token: TerminalID,
) -> (ParserGSS, AdvanceProfile) {
    use std::time::Instant;
    let t_total = Instant::now();
    let mut profile = AdvanceProfile::default();

    let summary = stack.summary();
    profile.top_states = stack.peek_values().len() as u32;
    profile.gss_depth = summary.max_depth;

    let mut gss = stack.clone();

    // Fast path: single state with a pure shift action
    if let Some(state) = gss.single_exclusive_top_value() {
        if let Some(Action::Shift(target)) = table.action(state, token) {
            profile.pure_shift = true;
            let result = gss.push(*target);
            profile.total_ns = t_total.elapsed().as_nanos() as u64;
            return (result, profile);
        }
    }

    // Try deterministic path
    let t_det = Instant::now();
    let det_result = advance_deterministically_profiled(table, &mut gss, token, &mut profile);
    profile.det_ns = t_det.elapsed().as_nanos() as u64;

    if det_result {
        profile.deterministic_finished = true;
        profile.total_ns = t_total.elapsed().as_nanos() as u64;
        return (gss, profile);
    }

    // Nondeterministic
    let t_nondet = Instant::now();
    profile.nondeterministic_entered = true;
    let result = advance_nondeterministically_profiled(table, gss, token, &mut profile);
    profile.nondet_ns = t_nondet.elapsed().as_nanos() as u64;
    profile.total_ns = t_total.elapsed().as_nanos() as u64;
    (result, profile)
}

fn advance_deterministically_profiled(
    table: &GLRTable,
    gss: &mut ParserGSS,
    token: TerminalID,
    profile: &mut AdvanceProfile,
) -> bool {
    let Some(mut stack) = gss.try_virtual_stack() else {
        profile.det_exit_reason = 6; // vstack fail
        return false;
    };

    profile.deterministic_entered = true;
    profile.vstack_len = stack.len() as u32;

    loop {
        let Some(&state) = stack.top() else {
            profile.det_exit_reason = 5; // no top
            break;
        };
        match table.action(state, token) {
            Some(Action::Reduce(rule_id)) => {
                let rule = &table.rules[*rule_id as usize];
                if rule.rhs.len() < stack.len() {
                    profile.n_reduces_above_floor += 1;
                    stack.pop(rule.rhs.len());
                    let goto_from = *stack.top().unwrap();
                    match table.goto_target(goto_from, rule.lhs) {
                        Some(target) => stack.push(target),
                        None => {
                            *gss = ParserGSS::empty();
                            profile.det_exit_reason = 4; // no goto
                            return false;
                        }
                    }
                } else {
                    profile.n_floor_crossings += 1;
                    let current = stack.into_gss();
                    let popped = current.popn(rule.rhs.len() as isize);
                    let mut gotos = GotoBatch::new();
                    for goto_from in popped.peek_values() {
                        let base = popped.isolate(Some(goto_from));
                        if let Some(target) = table.goto_target(goto_from, rule.lhs) {
                            add_goto(&mut gotos, target, base);
                        }
                    }
                    let rebuilt = apply_gotos(current, gotos);
                    let Some(next_stack) = rebuilt.try_virtual_stack() else {
                        *gss = rebuilt;
                        profile.det_exit_reason = 7; // floor cross vstack fail
                        return false;
                    };
                    stack = next_stack;
                }
            }
            Some(Action::Shift(target)) => {
                *gss = stack.into_gss().push(*target);
                profile.det_exit_reason = 1; // shift (finished)
                return true;
            }
            Some(Action::Split { .. }) => {
                profile.det_exit_reason = 2; // split
                profile.det_exit_state = state;
                break;
            }
            Some(Action::Accept) => {
                profile.det_exit_reason = 3; // accept
                profile.det_exit_state = state;
                break;
            }
            None => {
                profile.det_exit_reason = 4; // no action
                profile.det_exit_state = state;
                break;
            }
        }
    }

    *gss = stack.into_gss();
    false
}

fn advance_nondeterministically_profiled(
    table: &GLRTable,
    mut closure: ParserGSS,
    token: TerminalID,
    profile: &mut AdvanceProfile,
) -> ParserGSS {
    let mut shifted = ParserGSS::empty();

    loop {
        profile.n_nondet_waves += 1;
        let mut next = ParserGSS::empty();

        for state in closure.peek_values() {
            profile.n_nondet_branches += 1;
            let Some(action) = table.action(state, token) else { continue; };

            let mut isolated = closure.isolate(Some(state));
            if advance_deterministically(table, &mut isolated, token) {
                shifted = shifted.merge(&isolated);
                continue;
            }

            if let Some(target) = action.shift_target() {
                shifted = shifted.merge(&isolated.push(target));
            }

            for &rule_id in action.reduce_rule_ids() {
                let rule = &table.rules[rule_id as usize];
                for (goto_from, base) in reduce_sources(&closure, state, rule.rhs.len()) {
                    let Some(target) = table.goto_target(goto_from, rule.lhs) else { continue; };
                    let mut branch = base.push(target);
                    if advance_deterministically(table, &mut branch, token) {
                        shifted = shifted.merge(&branch);
                    } else {
                        next = next.merge(&branch);
                    }
                }
            }
        }

        if next.is_empty() { return shifted; }
        closure = next;
    }
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
        table.action.get(state as usize).is_some_and(|actions| {
            actions.keys().any(|&terminal| {
                terminals.contains(terminal as usize) || terminal == EOF
            })
        })
    })
}

pub(crate) fn stacks_finished(table: &GLRTable, stack: &ParserGSS) -> bool {
    if stack.is_empty() {
        return false;
    }

    let has_eof_action = stack
        .peek_values()
        .iter()
        .any(|&state| table.action(state, EOF).is_some());

    #[cfg(debug_assertions)]
    if has_eof_action {
        debug_assert!(
            stacks_accept(table, &stack_vectors(stack)),
            "IELR(1) fast check overapproximated for states {:?}",
            stack.peek_values(),
        );
    }

    has_eof_action
}


// ─── Test & debug infrastructure ──────────────────────────────────


#[cfg(test)]
thread_local! {
    static VSTACK_HIT_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
fn note_vstack_hit() {
    VSTACK_HIT_COUNT.with(|count| count.set(count.get() + 1));
}

#[cfg(test)]
fn take_vstack_hit_count() -> usize {
    VSTACK_HIT_COUNT.with(|count| {
        let hits = count.get();
        count.set(0);
        hits
    })
}


#[cfg(test)]
pub(crate) struct GLRParser {
    pub table: GLRTable,
    pub stack: ParserGSS,
}

#[cfg(test)]
impl GLRParser {
    pub(crate) fn new(table: GLRTable) -> Self {
        let stack = ParserGSS::from_stacks(&[(vec![0], TerminalsDisallowed::new())]);
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

#[cfg(any(test, debug_assertions))]
fn stack_vectors(stack: &ParserGSS) -> Vec<Vec<u32>> {
    stack.to_stacks().into_iter().map(|(stack, _)| stack).collect()
}

#[cfg(any(test, debug_assertions))]
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
        let reduce_rule_ids = action.reduce_rule_ids();
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
        if let Some(target) = table.action(state, token).and_then(Action::shift_target) {
            let mut shifted = stack.clone();
            shifted.push(target);
            next.push(shifted);
        }
    }
    dedup_stacks(next)
}

#[cfg(any(test, debug_assertions))]
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
pub(crate) fn valid_terminals_for_stacks(table: &GLRTable, stack: &ParserGSS) -> Vec<TerminalID> {
    valid_terminals_for_stack_vectors(table, &stack_vectors(stack))
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

    /// Differential test: for every token in `input`, advance both the GSS
    /// (which uses VirtualStack) and the flat reference implementation, then
    /// verify the resulting stack sets are identical.
    fn assert_advance_matches_reference(parser: &GLRParser, input: &[TerminalID]) {
        let mut gss = parser.stack.clone();
        let mut vecs = stack_vectors(&gss);
        for (i, &token) in input.iter().enumerate() {
            let gss_advanced = advance_stacks(&parser.table, &gss, token);
            let vec_advanced = advance_stack_vectors(&parser.table, &vecs, token);

            let mut gss_stacks = dedup_stacks(stack_vectors(&gss_advanced));
            gss_stacks.sort();
            let mut ref_stacks = dedup_stacks(vec_advanced.clone());
            ref_stacks.sort();

            assert_eq!(
                gss_stacks, ref_stacks,
                "Mismatch at step {i} (token {token}):\n  GSS stacks: {:?}\n  Ref stacks: {:?}",
                gss_stacks, ref_stacks
            );
            gss = gss_advanced;
            vecs = vec_advanced;
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

        let mut acc_inner = BTreeMap::new();
        acc_inner.insert(7, BTreeSet::from([11]));
        let acc = TerminalsDisallowed(Arc::new(acc_inner));
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
    fn test_advance_stacks_uses_virtual_stack_on_reduce_then_shift() {
        // Grammar: S -> A '+' ; A -> 'i'
        // After reading 'i', the next '+' triggers a deterministic reduce
        // followed by a shift, which should go through the VirtualStack path.
        let gdef = make_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            0,
            vec![tdef(0, "i"), tdef(1, "+")],
        );
        let parser = build_parser(&gdef);

        let after_i = advance_stacks(&parser.table, &parser.stack, 0);
        assert!(after_i.try_virtual_stack().is_some(), "single-path stack should admit VirtualStack");

        take_vstack_hit_count();
        let after_plus = advance_stacks(&parser.table, &after_i, 1);

        assert!(!after_plus.is_empty(), "reduce-then-shift path should stay alive");
        assert!(take_vstack_hit_count() > 0, "advance_stacks should hit try_vstack_reduces");
    }

    #[test]
    fn test_glr_left_recursive() {
        
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
    fn test_glr_right_recursive() {
        
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
    fn test_glr_expression_grammar() {
        
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
    fn test_glr_reduce_reduce_conflict() {
        
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
    fn test_glr_epsilon_ambiguity() {
        
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
    fn test_glr_highly_ambiguous() {
        
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
    fn test_glr_nullable_before_terminal() {
        
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
    fn test_glr_ambiguous_dangling_else() {
        
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
    fn test_close_token_wrapper_family_remains_parseable() {
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

        let advanced = advance_stacks(&current.table, &current.stack, CLOSE);

        assert!(!advanced.is_empty(), "close token should remain parseable");
        assert!(
            stacks_finished(&current.table, &advanced),
            "close token should reduce the wrapper family to a finished parse"
        );
    }

    /// Differential test: GSS advance (with VirtualStack) matches the flat
    /// reference implementation for grammars with epsilon productions and
    /// nullable nonterminals that can create intermediate `empty: true` nodes.
    #[test]
    fn test_vstack_matches_reference_nullable_grammars() {
        // Grammar 1: S → A B, A → 'x' | ε, B → 'x' | ε
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
        assert_advance_matches_reference(&parser, &[0]);
        assert_advance_matches_reference(&parser, &[0, 0]);

        // Grammar 2: S → S S | 'a' (highly ambiguous)
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Nonterminal(0)] },
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] },
            ],
            0,
            vec![tdef(0, "a")],
        );
        let parser = build_parser(&gdef);
        assert_advance_matches_reference(&parser, &[0, 0, 0, 0, 0]);

        // Grammar 3: S → A 'c', A → 'd' | ε (nullable before terminal)
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
        assert_advance_matches_reference(&parser, &[1, 0]);
        assert_advance_matches_reference(&parser, &[0]);

        // Grammar 4: S → A, A → A 'a' | 'b' (left-recursive chain)
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
                Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(0)] },
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(1)] },
            ],
            0,
            vec![tdef(0, "a"), tdef(1, "b")],
        );
        let parser = build_parser(&gdef);
        assert_advance_matches_reference(&parser, &[1, 0, 0, 0]);

        // Grammar 5: Expression grammar (deep reduce chains across floor)
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
        assert_advance_matches_reference(&parser, &[0, 1, 0, 2, 0]);
        assert_advance_matches_reference(&parser, &[3, 0, 1, 0, 4, 2, 0]);

        // Grammar 6: Reduce/reduce conflict
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
        assert_advance_matches_reference(&parser, &[0]);

        // Grammar 7: Wrapper family (many nonterminals, deep stack)
        {
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
                Rule { lhs: START, rhs: vec![Symbol::Terminal(OPEN), Symbol::Terminal(NUM), Symbol::Nonterminal(BODY), Symbol::Terminal(CLOSE)] },
                Rule { lhs: BODY, rhs: vec![Symbol::Nonterminal(TAIL_PACK)] },
                Rule { lhs: TAIL_ELEM, rhs: vec![Symbol::Terminal(COMMA), Symbol::Terminal(NUM)] },
                Rule { lhs: TAIL_PACK, rhs: vec![Symbol::Nonterminal(TAIL_ELEM)] },
                Rule { lhs: TAIL_PACK, rhs: vec![Symbol::Nonterminal(TAIL_ELEM), Symbol::Nonterminal(TAIL_ELEM)] },
            ];
            for i in 0..WRAPPER_COUNT {
                let wrap_nt = FIRST_WRAP + i as u32;
                rules.push(Rule { lhs: wrap_nt, rhs: vec![Symbol::Nonterminal(TAIL_PACK)] });
                rules.push(Rule { lhs: BODY, rhs: vec![Symbol::Nonterminal(wrap_nt)] });
            }
            let gdef = make_grammar(rules, START, vec![tdef(OPEN, "["), tdef(NUM, "n"), tdef(COMMA, ","), tdef(CLOSE, "]")]);
            let parser = build_parser(&gdef);
            assert_advance_matches_reference(&parser, &[OPEN, NUM, COMMA, NUM, COMMA, NUM, CLOSE]);
        }
    }
}
