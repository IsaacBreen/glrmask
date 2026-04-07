use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use super::analysis::EOF;
use super::table::{Action, GLRTable};
use crate::compiler::grammar::model::TerminalID;
use crate::ds::bitset::BitSet;
use crate::ds::leveled_gss::{LeveledGSS, Merge};
use smallvec::SmallVec;

/// Accumulator stored in the GSS.  Wraps the underlying BTreeMap in an Arc
/// so that Clone is O(1) (reference-count increment) instead of O(n).
/// Mutation is done by cloning the inner map on write.
#[derive(Clone, Debug)]
pub struct TerminalsDisallowed(pub(crate) Arc<BTreeMap<u32, BTreeSet<u32>>>);

impl TerminalsDisallowed {
    pub fn new() -> Self {
        TerminalsDisallowed(Arc::new(BTreeMap::new()))
    }

    /// Return a new TerminalsDisallowed with an additional entry inserted.
    pub fn with_insert(&self, state: u32, terminal: u32) -> Self {
        let mut inner = (*self.0).clone();
        inner.entry(state).or_default().insert(terminal);
        TerminalsDisallowed(Arc::new(inner))
    }
}

/// Deref to BTreeMap for transparent read-only access to all BTreeMap methods.
impl Deref for TerminalsDisallowed {
    type Target = BTreeMap<u32, BTreeSet<u32>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq for TerminalsDisallowed {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || *self.0 == *other.0
    }
}

impl Eq for TerminalsDisallowed {}

impl Hash for TerminalsDisallowed {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl Merge for TerminalsDisallowed {
    fn merge(&self, other: &Self) -> Self {
        if Arc::ptr_eq(&self.0, &other.0) {
            return self.clone();
        }
        let mut merged = (*self.0).clone();
        for (state, terminals) in other.0.iter() {
            merged
                .entry(*state)
                .or_default()
                .extend(terminals.iter().copied());
        }
        TerminalsDisallowed(Arc::new(merged))
    }
}

pub type ParserGSS = LeveledGSS<u32, TerminalsDisallowed>;

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

fn shift_target(action: Option<&Action>) -> Option<u32> {
    match action {
        Some(Action::Shift(target)) => Some(*target),
        Some(Action::Split {
            shift: Some(target),
            ..
        }) => Some(*target),
        _ => None,
    }
}

fn stack_vectors(stack: &ParserGSS) -> Vec<Vec<u32>> {
    stack.to_stacks().into_iter().map(|(stack, _)| stack).collect()
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
        if let Some(target) = shift_target(table.action(state, token)) {
            let mut shifted = stack.clone();
            shifted.push(target);
            next.push(shifted);
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

pub(crate) fn advance_stacks(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> ParserGSS {
    advance_stacks_core(table, stack.clone(), token)
}

/// Like `advance_stacks` but takes ownership of the GSS, avoiding an
/// unnecessary Arc clone when the caller doesn't need the original.
pub(crate) fn advance_stacks_owned(table: &GLRTable, stack: ParserGSS, token: TerminalID) -> ParserGSS {
    advance_stacks_core(table, stack, token)
}

fn advance_stacks_core(table: &GLRTable, stack: ParserGSS, token: TerminalID) -> ParserGSS {
    if let Some(state) = stack.single_exclusive_top_value() {
        match table.action(state, token) {
            Some(Action::Shift(target)) => return stack.push(*target),
            Some(Action::Split {
                shift: Some(target),
                reduces,
                accept,
            }) if reduces.is_empty() && !*accept => return stack.push(*target),
            Some(Action::Reduce(_)) | Some(Action::Accept) | Some(Action::Split { .. }) => {}
            None => return ParserGSS::empty(),
        }
    }

    let frontier = stack.peek_values();
    if frontier.is_empty() {
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
        return ParserGSS::empty();
    }
    if pure_shift_only && !pure_shift_targets.is_empty() {
        return stack.shift_top_values(pure_shift_targets);
    }

    // Owned: no clone needed. First Arc::make_mut won't clone if refcount == 1.
    let mut current = stack;

    // Use SmallVec for processed states — linear scan is faster than FxHashSet
    // for the typical case of ≤16 unique states in the reduce closure.
    let mut processed = SmallVec::<[u32; 16]>::new();

    // Collect initial unprocessed states from the GSS top values
    let mut new_states = SmallVec::<[u32; 8]>::new();
    if let Some(state) = current.single_top_value() {
        new_states.push(state);
    } else {
        current.for_each_top_value(|state| {
            new_states.push(state);
        });
    }

    let mut pending_bases_by_target = SmallVec::<[(u32, ParserGSS); 8]>::new();

    loop {
        if new_states.is_empty() {
            break;
        }

        let mut any_reduced = false;
        pending_bases_by_target.clear();
        for state in new_states.drain(..) {
            if processed.contains(&state) {
                continue;
            }
            processed.push(state);
            let reduce_rules: &[u32] = match table.action(state, token) {
                Some(Action::Reduce(rule_id)) => std::slice::from_ref(rule_id),
                Some(Action::Split { reduces, .. }) => reduces.as_slice(),
                _ => &[],
            };
            for &rule_id in reduce_rules {
                let rule = &table.rules[rule_id as usize];
                let rhs_len = rule.rhs.len();
                let lhs = rule.lhs;

                let bases = current.isolate_popn_bases(state, rhs_len as isize);
                for (goto_from, base) in bases {
                    if let Some(target) = table.goto_target(goto_from, lhs) {
                        if let Some((_, existing)) = pending_bases_by_target
                            .iter_mut()
                            .find(|(t, _)| *t == target)
                        {
                            *existing = existing.merge(&base);
                        } else {
                            pending_bases_by_target.push((target, base));
                        }
                        any_reduced = true;
                    }
                }
            }
        }
        if !any_reduced {
            break;
        }
        // Absorb results and use goto targets directly as next iteration's new_states
        for (target, base) in pending_bases_by_target.drain(..) {
            current = current.absorb_push_same_acc(target, &base);
            new_states.push(target);
        }
    }

    let mut shift_pairs = SmallVec::<[(u32, u32); 8]>::new();
    if let Some(state) = current.single_top_value() {
        if let Some(target) = shift_target(table.action(state, token)) {
            shift_pairs.push((state, target));
        }
    } else {
        current.for_each_top_value(|state| {
            if let Some(target) = shift_target(table.action(state, token)) {
                shift_pairs.push((state, target));
            }
        });
    }
    current.shift_top_values(shift_pairs)
}

/// Profiled version of `advance_stacks` that returns timing breakdown.
#[derive(Debug, Default)]
pub struct AdvanceProfile {
    pub isolate_ns: u64,
    pub popn_ns: u64,
    pub base_isolate_ns: u64,
    pub merge_ns: u64,
    pub absorb_push_ns: u64,
    pub shift_ns: u64,
    pub n_loop_iters: u32,
    pub n_reduces: u32,
}

pub(crate) fn advance_stacks_profiled(
    table: &GLRTable,
    stack: &ParserGSS,
    token: TerminalID,
) -> (ParserGSS, AdvanceProfile) {
    use std::time::Instant;
    let mut profile = AdvanceProfile::default();

    // Single-state fast paths (no profiling needed since they're trivially fast)
    if let Some(state) = stack.single_top_value() {
        match table.action(state, token) {
            Some(Action::Shift(target)) => return (stack.push(*target), profile),
            Some(Action::Split {
                shift: Some(target),
                reduces,
                accept,
            }) if reduces.is_empty() && !*accept => return (stack.push(*target), profile),
            Some(Action::Reduce(_)) | Some(Action::Accept) | Some(Action::Split { .. }) => {}
            None => return (ParserGSS::empty(), profile),
        }
    }

    let frontier = stack.peek_values();
    if frontier.is_empty() {
        return (ParserGSS::empty(), profile);
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
        return (ParserGSS::empty(), profile);
    }
    if pure_shift_only && !pure_shift_targets.is_empty() {
        let t0 = Instant::now();
        let result = stack.shift_top_values(pure_shift_targets);
        profile.shift_ns = t0.elapsed().as_nanos() as u64;
        return (result, profile);
    }

    let mut current = stack.clone();
    let mut processed = SmallVec::<[u32; 16]>::new();

    // Collect initial unprocessed states from the GSS top values
    let mut new_states = SmallVec::<[u32; 8]>::new();
    if let Some(state) = current.single_top_value() {
        new_states.push(state);
    } else {
        current.for_each_top_value(|state| {
            new_states.push(state);
        });
    }

    let mut pending_bases_by_target = SmallVec::<[(u32, ParserGSS); 8]>::new();

    loop {
        profile.n_loop_iters += 1;
        if new_states.is_empty() {
            break;
        }

        let mut any_reduced = false;
        pending_bases_by_target.clear();
        for state in new_states.drain(..) {
            if processed.contains(&state) {
                continue;
            }
            processed.push(state);
            let reduce_rules: &[u32] = match table.action(state, token) {
                Some(Action::Reduce(rule_id)) => std::slice::from_ref(rule_id),
                Some(Action::Split { reduces, .. }) => reduces.as_slice(),
                _ => &[],
            };
            for &rule_id in reduce_rules {
                profile.n_reduces += 1;
                let rule = &table.rules[rule_id as usize];
                let rhs_len = rule.rhs.len();
                let lhs = rule.lhs;
                let t0 = Instant::now();
                let bases = current.isolate_popn_bases(state, rhs_len as isize);
                profile.isolate_ns += t0.elapsed().as_nanos() as u64;

                for (goto_from, base) in bases {
                    if let Some(target) = table.goto_target(goto_from, lhs) {
                        if let Some((_, existing)) = pending_bases_by_target
                            .iter_mut()
                            .find(|(t, _)| *t == target)
                        {
                            let t0 = Instant::now();
                            *existing = existing.merge(&base);
                            profile.merge_ns += t0.elapsed().as_nanos() as u64;
                        } else {
                            pending_bases_by_target.push((target, base));
                        }
                        any_reduced = true;
                    }
                }
            }
        }
        if !any_reduced {
            break;
        }
        // Absorb results and use goto targets directly as next iteration's new_states
        for (target, base) in pending_bases_by_target.drain(..) {
            let t0 = Instant::now();
            current = current.absorb_push_same_acc(target, &base);
            profile.absorb_push_ns += t0.elapsed().as_nanos() as u64;
            new_states.push(target);
        }
    }

    let mut shift_pairs = SmallVec::<[(u32, u32); 8]>::new();
    if let Some(state) = current.single_top_value() {
        if let Some(target) = shift_target(table.action(state, token)) {
            shift_pairs.push((state, target));
        }
    } else {
        current.for_each_top_value(|state| {
            if let Some(target) = shift_target(table.action(state, token)) {
                shift_pairs.push((state, target));
            }
        });
    }
    let t0 = Instant::now();
    let result = current.shift_top_values(shift_pairs);
    profile.shift_ns = t0.elapsed().as_nanos() as u64;
    (result, profile)
}

/// Flat-stack alternative to `advance_stacks` for small GSSs.
///
/// Converts the GSS to flat Vec stacks, applies reduce closure + shift using
/// simple Vec operations, and rebuilds via `from_stacks`. This avoids the
/// overhead of LeveledGSS operations (Arc, HashMap, recursive structure)
/// which dominates runtime for small GSSs.
///
/// Returns `None` if the GSS has more paths than `max_paths`, in which case
/// the caller should fall back to the standard `advance_stacks`.
pub(crate) fn advance_stacks_flat(
    table: &GLRTable,
    stack: &ParserGSS,
    token: TerminalID,
    max_paths: usize,
) -> Option<ParserGSS> {
    let flat = stack.to_stacks_bounded(max_paths)?;
    if flat.is_empty() {
        return Some(ParserGSS::empty());
    }

    // Reduce closure on flat stacks.
    // Each stack is (Vec<u32>, TerminalsDisallowed). The annotation is carried
    // along unchanged through reduces (it's associated with the stack prefix
    // that doesn't change during reduce closure).
    let mut visited: std::collections::HashSet<Vec<u32>> = std::collections::HashSet::new();
    let mut queue: VecDeque<(Vec<u32>, TerminalsDisallowed)> = VecDeque::new();

    for (s, a) in &flat {
        if visited.insert(s.clone()) {
            queue.push_back((s.clone(), a.clone()));
        }
    }

    let mut shifted: Vec<(Vec<u32>, TerminalsDisallowed)> = Vec::new();

    while let Some((s, ann)) = queue.pop_front() {
        let Some(&state) = s.last() else { continue };

        // Check for shift action
        if let Some(target) = shift_target(table.action(state, token)) {
            let mut shifted_stack = s.clone();
            shifted_stack.push(target);
            shifted.push((shifted_stack, ann.clone()));
        }

        // Apply reduce rules
        let reduce_rule_ids: &[u32] = match table.action(state, token) {
            Some(Action::Reduce(rule_id)) => std::slice::from_ref(rule_id),
            Some(Action::Split { reduces, .. }) => reduces.as_slice(),
            _ => &[],
        };

        for &rule_id in reduce_rule_ids {
            let rule = &table.rules[rule_id as usize];
            let rhs_len = rule.rhs.len();
            if s.len() < rhs_len + 1 {
                continue;
            }
            let keep_len = s.len() - rhs_len;
            let goto_from = s[keep_len - 1];
            if let Some(target) = table.goto_target(goto_from, rule.lhs) {
                let mut reduced = s[..keep_len].to_vec();
                reduced.push(target);
                if visited.insert(reduced.clone()) {
                    // Carry annotation from the prefix (it's associated with the
                    // lower part of the stack which is preserved through reduces)
                    queue.push_back((reduced, ann.clone()));
                }
            }
        }
    }

    if shifted.is_empty() {
        Some(ParserGSS::empty())
    } else {
        Some(ParserGSS::from_stacks(&shifted))
    }
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

pub(crate) fn stacks_finished(table: &GLRTable, stack: &ParserGSS) -> bool {
    if stack.is_empty() {
        return false;
    }

    // IELR(1) fast check: if any root state has any action for EOF,
    // the parser can accept. With IELR(1)'s precise lookahead, a reduce
    // for EOF in a state implies the reduce chain leads to accept.
    let has_eof_action = stack
        .peek_values()
        .iter()
        .any(|&state| table.action(state, EOF).is_some());

    // Debug assertion: verify the fast check matches the full check
    #[cfg(debug_assertions)]
    {
        let full_result = if has_eof_action {
            // Only pay for full check when fast check says true
            let vecs = stack_vectors(stack);
            stacks_accept(table, &vecs)
        } else {
            false
        };
        // full_result can only be true if has_eof_action is true
        // But has_eof_action true does NOT guarantee full_result true
        // If this fires, the simple IELR(1) check is insufficient
        if has_eof_action && !full_result {
            // Log but don't crash — the check is an overapproximation
            debug_assert!(
                full_result,
                "stacks_finished: IELR(1) fast check returned true but full check returned false.\n\
                 Root states with EOF action: {:?}",
                stack.peek_values().iter().filter(|&&s| table.action(s, EOF).is_some()).collect::<Vec<_>>()
            );
        }
    }

    has_eof_action
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
}
